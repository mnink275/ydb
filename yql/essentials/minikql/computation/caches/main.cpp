#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>

#include "include/cache.hpp"

using namespace std::chrono_literals;

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
  std::chrono::nanoseconds getsTime{0ns};
  std::chrono::nanoseconds updatesTime{0ns};
  int64_t RSS{0};

  void Print() const {
    std::cout << "RSS: " << RSS / 1024.0 << " MB" << std::endl;
    std::cout << "Hit ratio: "
              << (100 * static_cast<double>(hitCount) / totalCount) << " %"
              << std::endl;

    const auto getEstimatedTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(benchmarkTime -
                                                             updatesTime)
            .count() /
        totalCount;
    std::cout << "Get estimated time: " << getEstimatedTime << std::endl;

    const auto getAverageTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(getsTime)
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
    getsTime += other.getsTime;
    updatesTime += other.updatesTime;
    RSS += other.RSS;
    return *this;
  }
};

template <class TCache, typename... CacheArgs>
BenchmarkResult RunBenchmark(const auto& keys, TCache& cache) {
  const auto beforeBenchmarkRSS = utils::PrintRSS();

  size_t hitCount = 0;
  size_t totalCount = 0;

  auto gets_time = 0ns;
  auto updates_time = 0ns;
  auto start = std::chrono::high_resolution_clock::now();

  const auto now = utils::Now();
  const auto far_future = now + 3600;
  for (auto key : keys) {
    // auto start_get = std::chrono::high_resolution_clock::now();
    const auto value = cache.Get(key, far_future);
    // gets_time += std::chrono::high_resolution_clock::now() - start_get;

    if (value.HasValue()) {
      hitCount++;
    } else {
      auto start_update = std::chrono::high_resolution_clock::now();
      cache.Update(key, cache::UVPod{key + 10}, far_future);
      updates_time += std::chrono::high_resolution_clock::now() - start_update;
    }

    totalCount++;
  }

  const auto benchmarkTime = std::chrono::high_resolution_clock::now() - start;

  return BenchmarkResult{hitCount, totalCount, benchmarkTime, gets_time, updates_time,
                         utils::PrintRSS() - beforeBenchmarkRSS};
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

  const size_t kBatchSize = 700'000'000;
  std::vector<uint32_t> benchmark_keys;
  benchmark_keys.reserve(kBatchSize);

  BenchmarkResult total_result;

  // std::string filename = "dataset/f_960M_43M.txt";
  // std::string filename = "dataset/ws_458M_14M.txt";
  // std::string filename = "dataset/n_10M_578K.txt";
  // std::string filename = "dataset/f_640M_28M.txt";
  // std::string filename = "dataset/WebSearch2.txt";
  std::string filename = "dataset/P14.txt";

  std::ifstream input(filename);
  if (!input.is_open()) {
    throw std::runtime_error("Can't open file");
  }

  const auto beforeCacheInitRSS = utils::PrintRSS();

#if PAGE_BASED_CACHE
  using TCache = Cache;
  TCache cache{typeBuilder, holderFactory};
#else
  // using TCache = LRU;
  using TCache = LRU<uint32_t>;
  // const size_t CACHE_SIZE = 131584;
  // const size_t CACHE_SIZE = 524288;
  const size_t CACHE_SIZE = 3'000'000;
  TCache cache{CACHE_SIZE};
#endif

  total_result.RSS += utils::PrintRSS() - beforeCacheInitRSS;

  for (uint32_t key{}; input >> key;) {
    benchmark_keys.emplace_back(key);
    if (benchmark_keys.size() == kBatchSize) {
      auto result = RunBenchmark<TCache>(benchmark_keys, cache);

      total_result += result;

      benchmark_keys.clear();
      std::cout << "Batch handled" << std::endl;
    }
  }
  if (!benchmark_keys.empty()) {
    auto result = RunBenchmark<TCache>(benchmark_keys, cache);
    total_result += result;
    std::cout << "Batch handled" << std::endl;
  }

  total_result.Print();
}
