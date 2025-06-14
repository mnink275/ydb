#include <thread>
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

#include "actors/page_based_cache.hpp"
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

class TBenchmarkActor : public TActorBootstrapped<TBenchmarkActor> {
 public:
  TBenchmarkActor(const std::string& filename, TActorId cacheActor)
    : Filename(filename),
      CacheActor(cacheActor) {}

  ~TBenchmarkActor() {
    Cout << "Total count: " << TotalCount << Endl;
    Cout << "Hit count: " << HitCount << Endl;
    if (TotalCount > 0) {
      Cout << "Hit ratio: "
              << (100 * static_cast<double>(HitCount) / TotalCount) << " %"
              << Endl;
    }
  }

  STFUNC(StateWait) {
    switch (ev->GetTypeRewrite()) {
      hFunc(TEvCache::TEvTerminate, Handle);
      hFunc(TEvCache::TEvGetResult, Handle);
      hFunc(TEvCache::TEvLargePageNotLoaded, Handle);
    }

    ++HandledEvents;
  }

  void Bootstrap() {
    Cout << "Benchmark started" << Endl;

    std::ifstream input{Filename};
    if (!input.is_open()) {
      throw std::runtime_error("Can't open file");
    }

    const size_t kBatchSize = 2 * (1ULL << 30); // 2 GB
    for (uint32_t key{}; input >> key;) {
      BenchmarkKeys.push(key);
      if (BenchmarkKeys.size() >= kBatchSize) {
        throw std::runtime_error{"ALERT: dataset is larger than kBatchSize"};
      }
    }
    KeysNum = BenchmarkKeys.size();

    Cout << "Bootstrap Finished" << Endl;
    Become(&TThis::StateWait);
    SendNextKey();
  }

 private:
  void Handle(TEvCache::TEvGetResult::TPtr& ev) {
    auto& data = *ev->Get();
    if (data.Payload.HasValue()) {
      HitCount++;
    } else {
      SendUpdateEvent(data.Key);
    }

    if (TotalCount % 10'000'000 == 0) {
      Cout << "Handled Get Results: " << TotalCount << Endl;
    }

    SendNextKey();
  }

  void Handle(TEvCache::TEvLargePageNotLoaded::TPtr& ev) {
    // TODO: this update emulates unexpected Pure C++ Cache version, that affects LargePages' frequency
    SendUpdateEvent(ev->Get()->Key);

    SendNextKey();
  }

  void Handle(TEvCache::TEvTerminate::TPtr&) {
    if (++TerminatedLargePageActorsCount == cache::LOADED_PAGE_NUMBER) {
      ShouldContinue.ShouldStop();
    }
  }

  void SendUpdateEvent(const ui32 key) {
    const auto expiration = utils::Now() + FarFuture;
    Send(CacheActor, new TEvCache::TEvUpdate{key, cache::UVPod{key + 10}, expiration});
  }

  void SendNextKey() {
    if (BenchmarkKeys.empty()) {
      Send(CacheActor, new TEvCache::TEvTerminate{SelfId()});
      return;
    }

    ++TotalCount;

    const auto key = BenchmarkKeys.front();
    BenchmarkKeys.pop();
    Send(CacheActor, new TEvCache::TEvGet{key, utils::Now()});
  }

 private:
  std::string Filename;
  TActorId CacheActor;

  const ui32 FarFuture{3600}; // 1 hour
  size_t HitCount{0};
  size_t TotalCount{0};
  size_t KeysNum{0};

  size_t TerminatedLargePageActorsCount{0};


  std::chrono::high_resolution_clock::time_point StartBenchmarkTime{0ns};
  std::queue<uint32_t> BenchmarkKeys;
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
  // std::string filename = "/home/ilyaeroshev/ydb/yql/essentials/minikql/computation/caches/dataset/Financial1.txt";

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
    new ink::actor::TBenchmarkActor{filename, cacheActorId});

  Cout << "Cache Actor ID: " << cacheActorId << Endl;
  Cout << "Benchmark Actor ID: " << benchmarkActorId << Endl;

  alloc.Release();
  while (ShouldContinue.PollState() == TProgramShouldContinue::Continue) {
    Sleep(TDuration::MilliSeconds(200));
  }

  actorSystem.Stop();
  actorSystem.Cleanup();

  return ShouldContinue.GetReturnCode();
}
