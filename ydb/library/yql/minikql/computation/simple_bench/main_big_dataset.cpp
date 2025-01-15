#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <unordered_set>
#include <thread>

#include "../mkql_key_payload_value_lru_cache.h"

#include <ydb/library/yql/minikql/mkql_type_builder.h>
#include <ydb/library/yql/minikql/mkql_string_util.h>
#include <util/generic/xrange.h>

#include <chrono>

using namespace NKikimr;

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

    int64_t page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    double rss = resident * page_size_kb;
    return static_cast<int64_t>(rss); // in KB
}

}  // namespace utils

std::vector<uint32_t> unique(std::vector<uint32_t> arr) {
    std::sort(arr.begin(), arr.end());
    arr.resize(std::unique(arr.begin(), arr.end()) - arr.begin());
    return arr;
}

std::vector<uint32_t> LoadFromFile(const std::string& filename) {
    std::ifstream input(filename);
    if (!input.is_open()) {
        throw std::runtime_error(std::string{"Filed to open file: "} + filename);
    }

    std::vector<uint32_t> keys;
    keys.reserve(1'000'000);

    for (uint32_t key; input >> key;) {
        keys.emplace_back(key);
    }

    std::cout << "Total keys count: " << keys.size() << std::endl;
    std::cout << "Unique keys count: " << unique(keys).size() << std::endl;

    return keys;
}

struct BenchmarkResult {
    static constexpr auto CPU_WORKING_CLOCK_GHz = 4.0;

    size_t hitCount;
    size_t totalCount;
    std::chrono::nanoseconds benchmarkTime;
    int64_t RSS;

    void Print() {
        std::cout << "RSS: " << RSS << " KB" << std::endl;
        std::cout << "Hit ratio: " << 100 * static_cast<double>(hitCount) / totalCount << " %" << std::endl;

        const auto averageTime = std::chrono::duration_cast<std::chrono::nanoseconds>(benchmarkTime).count() / totalCount;

        std::cout << "Operation average time: "
                  << averageTime
                  << " ns (" << static_cast<uint64_t>(averageTime * CPU_WORKING_CLOCK_GHz) << " ticks)" << std::endl;
    }
};

template <class TCache, typename... CacheArgs>
BenchmarkResult RunBenchmark(const auto& keys, CacheArgs&&... cache_args) {
    using namespace std::chrono_literals;
    const auto beforeBenchmarkRSS =  utils::PrintRSS();

    // std::cout << "Check RSS before" << std::endl;
    // std::this_thread::sleep_for(5s);

    TCache cache(std::forward<CacheArgs>(cache_args)...);

    size_t hitCount = 0;
    size_t totalCount = 0;
    auto start = std::chrono::high_resolution_clock::now();

    const auto ttl_now = std::chrono::steady_clock::now();
    const auto ttl_dt = std::chrono::hours(1);
    for (auto key : keys) {
        if (cache.Get(NUdf::TUnboxedValuePod{key}, ttl_now)) {
            hitCount++;
        } else {
            cache.Update(NUdf::TUnboxedValuePod{key}, NUdf::TUnboxedValuePod{0}, ttl_now + ttl_dt);
        }

        totalCount++;
    }

    const auto benchmarkTime = std::chrono::high_resolution_clock::now() - start;
    // std::cout << "Check RSS after" << std::endl;
    // std::this_thread::sleep_for(5s);

    return BenchmarkResult{hitCount, totalCount, benchmarkTime, utils::PrintRSS() - beforeBenchmarkRSS};
}


int main() {
  using namespace std::chrono_literals;
  using TCache = NMiniKQL::TUnboxedKeyValueLruCacheWithTtl;

  NMiniKQL::TScopedAlloc main_alloc{__LOCATION__};
  NMiniKQL::TTypeEnvironment typeEnv(main_alloc);
  NMiniKQL::TTypeBuilder typeBuilder(typeEnv);

  auto benchmark_keys = LoadFromFile("Financial1.txt");

  auto result = RunBenchmark<TCache>(
    benchmark_keys,
    100'000,
    typeBuilder.NewDataType(NUdf::EDataSlot::Uint32));
  result.Print();
}
