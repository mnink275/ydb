#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include <ydb/library/actors/core/actorsystem.h>
#include <ydb/library/actors/core/executor_pool_basic.h>
#include <ydb/library/actors/core/scheduler_basic.h>
#include <ydb/library/actors/core/log.h>
#include <ydb/library/actors/core/actor_bootstrapped.h>
#include <ydb/library/actors/util/should_continue.h>
#include <util/system/sigset.h>
#include <util/generic/xrange.h>
#include <util/system/guard.h>

#include "include/cache.hpp"
#include "events.hpp"

using namespace std::chrono_literals;
using namespace NActors;

static TProgramShouldContinue ShouldContinue;

void OnTerminate(int) {
  ShouldContinue.ShouldStop();
}

namespace utils {

int64_t PrintRSS() {
  // https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
  std::ifstream file("/proc/self/statm");
  if (!file.is_open()) {
    throw std::runtime_error("Can't open /proc/self/statm");
  }

  int tSize = 0;
  int resident = 0;
  int share = 0;
  file >> tSize >> resident >> share;

  int64_t page_size_kb = sysconf(_SC_PAGE_SIZE) /
                         1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;
  return static_cast<int64_t>(rss);  // in KB
}

}  // namespace utils

struct BenchmarkResult {
  static constexpr auto CPU_WORKING_CLOCK_GHz = 4.0;

  size_t hitCount{0};
  size_t totalCount{0};
  std::chrono::nanoseconds benchmarkTime{0ns};
  std::chrono::nanoseconds updatesTime{0ns};
  int64_t RSS{0};

  void Print() const {
    std::cout << "RSS: " << RSS / 1024.0 << " MB" << std::endl;
    std::cout << "Hit ratio: "
              << (100 * static_cast<double>(hitCount) / totalCount) << " %"
              << std::endl;

    const auto getAverageTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(benchmarkTime -
                                                             updatesTime)
            .count() /
        totalCount;
    std::cout << "Get average time: " << getAverageTime << std::endl;

    const auto updateAverageTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(updatesTime)
            .count() /
        (totalCount - hitCount);
    std::cout << "Update average time: " << updateAverageTime << std::endl;

    const auto opAverageTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(benchmarkTime)
            .count() /
        totalCount;
    std::cout << "Op average time: " << opAverageTime << std::endl;
  }

  BenchmarkResult& operator+=(const BenchmarkResult& other) {
    hitCount += other.hitCount;
    totalCount += other.totalCount;
    benchmarkTime += other.benchmarkTime;
    updatesTime += other.updatesTime;
    RSS += other.RSS;
    return *this;
  }
};

namespace ink::actor {

class TPageBasedActor : public TActorBootstrapped<TPageBasedActor> {
public:
  TPageBasedActor(NKikimr::NMiniKQL::TScopedAlloc& alloc,
                  const NKikimr::NMiniKQL::TTypeBuilder& typeBuilder,
                  const NKikimr::NMiniKQL::THolderFactory& holderFactory)
    : PageBasedCache(typeBuilder, holderFactory),
      Alloc(alloc) {}

  ~TPageBasedActor() {
    Cout << "Total count: " << TotalCount << Endl;
    Cout << "Hit count: " << HitCount << Endl;
    if (TotalCount > 0) {
      Cout << "Hit Ratio: " << 1.0 * HitCount / TotalCount << Endl;
    }
  }

  STFUNC(StateWait) {
    switch (ev->GetTypeRewrite()) {
      hFunc(TEvCache::TEvGet, Handle);
      hFunc(TEvCache::TEvUpdate, Handle);
      hFunc(TEvCache::TEvTerminate, Handle);
    }

    ++HandledEvents;
  }

  void Bootstrap() {
    Become(&TThis::StateWait);
  }

 private:
  void Handle(TEvCache::TEvGet::TPtr& ev) {
    auto guard = BindAllocator();

    const auto start = std::chrono::high_resolution_clock::now();
    const auto value = PageBasedCache.Get(ev->Get()->Key, ev->Get()->Now);
    GetsTime += std::chrono::high_resolution_clock::now() - start;

    if (value.HasValue()) {
      HitCount++;
    } else {
      const auto start = std::chrono::high_resolution_clock::now();
      PageBasedCache.Update(ev->Get()->Key, std::move(ev->Get()->Value), ev->Get()->Now + FarFuture);
      UpdatesTime += std::chrono::high_resolution_clock::now() - start;
    }

    TotalCount++;
  }

  void Handle(TEvCache::TEvUpdate::TPtr& ev) {
    auto guard = BindAllocator();

    const auto start = std::chrono::high_resolution_clock::now();
    PageBasedCache.Update(ev->Get()->Key, std::move(ev->Get()->Value), ev->Get()->Expiration);
    UpdatesTime += std::chrono::high_resolution_clock::now() - start;
  }

  void Handle(TEvCache::TEvTerminate::TPtr& ev) {
    Y_ASSERT(ev->Get()->KeyCount == TotalCount);

    ShouldContinue.ShouldStop();
  }

  TGuard<NKikimr::NMiniKQL::TScopedAlloc> BindAllocator() {
    return Guard(Alloc);
  }

 private:
  cache::Cache PageBasedCache;
  NKikimr::NMiniKQL::TScopedAlloc& Alloc;

  const ui32 FarFuture{3600}; // 1 hour

  size_t HitCount{0};
  size_t TotalCount{0};

  std::chrono::high_resolution_clock::time_point GetsTime{0ns};
  std::chrono::high_resolution_clock::time_point UpdatesTime{0ns};
};


class TBenchmarkActor : public TActorBootstrapped<TBenchmarkActor> {
 public:
  TBenchmarkActor(NKikimr::NMiniKQL::TScopedAlloc& alloc,
                 const std::string& filename, TActorId cacheActor)
    : Alloc(alloc),
      Filename(filename),
      CacheActor(cacheActor) {}

  void Bootstrap() {
    Cout << "Benchmark started" << Endl;

    std::ifstream input(Filename);
    if (!input.is_open()) {
      throw std::runtime_error("Can't open file");
    }

    auto guard = BindAllocator();

    const size_t kBatchSize = 700'000'000;
    std::vector<uint32_t> benchmark_keys;
    benchmark_keys.reserve(kBatchSize);
    size_t keyCount = 0;

    for (uint32_t key{}; input >> key;) {
      keyCount++;

      benchmark_keys.emplace_back(key);
      if (benchmark_keys.size() == kBatchSize) {
        SendBatch(std::move(benchmark_keys));

        benchmark_keys.clear();
        Cout << "Batch handled" << Endl;
      }
    }
    if (!benchmark_keys.empty()) {
      SendBatch(std::move(benchmark_keys));
      Cout << "Batch handled" << Endl;
    }

    Cout << "Bootstrap Finished" << Endl;
    Send(CacheActor, new TEvCache::TEvTerminate{keyCount});
  }

 private:
  TGuard<NKikimr::NMiniKQL::TScopedAlloc> BindAllocator() {
    return Guard(Alloc);
  }

  void SendBatch(std::vector<uint32_t>&& keys) {
    for (auto key : keys) {
      Send(CacheActor, new TEvCache::TEvGet(key, cache::UVPod{key + 10}, utils::Now()));
    }
  }

 private:
  NKikimr::NMiniKQL::TScopedAlloc& Alloc;
  std::string Filename;
  TActorId CacheActor;
};

}  // ink::actor

THolder<TActorSystemSetup> BuildActorSystemSetup(ui32 threads, ui32 pools) {
  Y_ABORT_UNLESS(threads > 0 && threads < 100);
  Y_ABORT_UNLESS(pools > 0 && pools < 10);

  auto setup = MakeHolder<TActorSystemSetup>();

  setup->NodeId = 1;

  setup->ExecutorsCount = pools;
  setup->Executors.Reset(new TAutoPtr<IExecutorPool>[pools]);
  for (ui32 idx : xrange(pools)) {
    setup->Executors[idx] = new TBasicExecutorPool(idx, threads, 50);
  }

  setup->Scheduler = new TBasicSchedulerThread(TSchedulerConfig(512, 0));

  return setup;
}

int main() {
  using namespace std::chrono_literals;
  using namespace cache;

#define PAGE_BASED_CACHE true

#if PAGE_BASED_CACHE
  NKikimr::NMiniKQL::TScopedAlloc alloc{__LOCATION__};
  NKikimr::NMiniKQL::TTypeEnvironment typeEnv{alloc};
  NKikimr::NMiniKQL::TTypeBuilder typeBuilder{typeEnv};
  NKikimr::NMiniKQL::TMemoryUsageInfo memInfo{"Memory"};
  NKikimr::NMiniKQL::THolderFactory holderFactory{alloc.Ref(), memInfo};


  std::cout << "Cache size: " << CACHE_SIZE << '\n';
  std::cout << "Configuration: " << LARGE_PAGE_SHIFT << ' ' << SMALL_PAGE_SHIFT
            << ' ' << SMALL_PAGE_SIZE_SHIFT << ' ' << "(" << LOADED_PAGE_NUMBER
            << " loaded)" << '\n';

  std::cout << "TinyLFU " << "(" << TLFU_SIZE << ", " << SAMPLE_SIZE << ")\n";
  if (USE_LRU) std::cout << "LRU " << (USE_LRU ? LRU_SIZE : 0) << std::endl;
  if (USE_BF)
    std::cout << "Bloom filter " << (USE_BF ? "ON" : "OFF") << std::endl;
  if (USE_SIMD) std::cout << "SIMD " << (USE_SIMD ? "ON" : "OFF") << std::endl;
#endif

  std::string filename = "/home/ilyaeroshev/ydb/yql/essentials/minikql/computation/caches/dataset/P14.txt";
  // std::string filename = "dataset/Financial1.txt";

#ifdef _unix_
  signal(SIGPIPE, SIG_IGN);
#endif
  signal(SIGINT, &OnTerminate);
  signal(SIGTERM, &OnTerminate);

  auto actorSystemSetup = BuildActorSystemSetup(1, 1);
  TActorSystem actorSystem(actorSystemSetup);

  actorSystem.Start();

  const auto cacheActorId = actorSystem.Register(
    new ink::actor::TPageBasedActor{alloc, typeBuilder, holderFactory});
  const auto benchmarkActorId = actorSystem.Register(
    new ink::actor::TBenchmarkActor{alloc, filename, cacheActorId});
  Y_UNUSED(benchmarkActorId);

  alloc.Release();
  while (ShouldContinue.PollState() == TProgramShouldContinue::Continue) {
    Sleep(TDuration::MilliSeconds(200));
  }

  actorSystem.Stop();
  actorSystem.Cleanup();

  return ShouldContinue.GetReturnCode();
}
