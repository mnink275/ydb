#include "../mkql_key_payload_value_lru_cache.h"

#include <ydb/library/yql/minikql/mkql_type_builder.h>
#include <ydb/library/yql/minikql/mkql_string_util.h>
#include <util/generic/xrange.h>

#include <ydb/core/ymq/actor/sha256.h>

#include "concurrent_lru_wrapper.hpp"
#include "sharded_ydb_lru_cache.hpp"

#include <fstream>
#include <chrono>

using namespace NKikimr;

namespace utils {

struct Options {
    size_t CacheSize;
    double Mean;
    double Stddev;
    size_t ThreadCount;
    std::optional<size_t> MakeStaleProbability;
};

Options GetOptions(int argc, char** argv) {
    Options options{
        .CacheSize = 0,
        .Mean = 0.0,
        .Stddev = 0.0,
        .ThreadCount = 1,
        .MakeStaleProbability = std::nullopt
    };

    for (int i = 1; i < argc; ++i) {
        auto arg = std::string_view(argv[i]);
        if (arg == "--cache-size" || arg == "-s") {
            options.CacheSize = std::stoull(argv[++i]);
        } else if (arg == "--mean" || arg == "-m") {
            options.Mean = std::stod(argv[++i]);
        } else if (arg == "--stddev" || arg == "-d") {
            options.Stddev = std::stod(argv[++i]);
        } else if (arg == "--thread-count" || arg == "-t") {
            options.ThreadCount = std::stoull(argv[++i]);
        } else if (arg == "--stale-probability" || arg == "-p") {
            options.MakeStaleProbability = std::stoull(argv[++i]);
            if (options.MakeStaleProbability > 100) throw std::runtime_error("Stale probability must be in range [0, 100]");
        } else {
            throw std::runtime_error("Unknown option: " + std::string(argv[i]));
        }
    }

    if (options.CacheSize == 0) {
        throw std::runtime_error("Cache size must be more than 0");
    }

    if (options.Stddev == 0.0) {
        options.Stddev = options.CacheSize;
    }

    return options;
}

long PrintRSS() {
    // https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
    std::ifstream file("/proc/self/statm");
    if (!file.is_open()) {
        throw std::runtime_error("Can't open /proc/self/statm");
    }

    int tSize = 0, resident = 0, share = 0;
    file >> tSize >> resident >> share;

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    double rss = resident * page_size_kb;
    return static_cast<long>(rss) / 1024; // in MB
}

}  // namespace utils

namespace {

template <class T>
class Statistic {
 public:
  void Load(const NMiniKQL::TKeyPayloadPairVector& data) {
    for(const auto& [key, value] : data) {
      ++Hist[key.Get<T>()];
    }
  }

  std::vector<size_t> Get() const {
    std::vector<size_t> res;
    res.reserve(Hist.size());

    for(const auto& [_, y] : Hist) {
      res.push_back(y);
    }

    std::sort(res.begin(), res.end(), std::greater<>());

    return res;
  }

  void GetHist(const std::string_view filename = "hist.py") const {
    std::ofstream out(filename.data(), std::ios::trunc);

    out << "import matplotlib.pyplot as plt\n";
    out << "ys = [";
    for(auto y: Get()) {
      out << y << ",";
    }
    out << "]\n";
    out << "xs = [i for i in range(len(ys))]\n";
    out << "plt.plot(xs, ys)\n";
    out << "plt.grid(True)\n";
    out << "plt.xlabel('Elements')\n";
    out << "plt.ylabel('Count')\n";
    out << "plt.show()\n";
  }

  void Clear() {
    Hist.clear();
  }

  size_t UniqueCount() const {
    return Hist.size();
  }

  void Print(const size_t scale = 100) const {
    size_t i = 0;
    for (const auto& [_, y] : Hist) {
      if (y / scale == 0) continue;
      std::cout << std::setw(4) << ++i << ' ' << std::string(y / scale, '*') << '\n';
    }
  }

 private:
  std::map<T, size_t> Hist;
};

template <NUdf::EDataSlot DataSlot, bool collectStats = false>
class IKeysGenerator {
public:
    virtual ~IKeysGenerator() = default;
    virtual NYql::NUdf::TUnboxedValuePod NextKey() = 0;
    size_t UniqueCount() const {
        return Freqs.size();
    }

protected:
    NYql::NUdf::TUnboxedValuePod NextKeyImpl(const auto& genKey) {
        if constexpr (DataSlot == NUdf::EDataSlot::String) {
            auto strKey = NKikimr::NSQS::CalcSHA256(std::to_string(genKey));
            return NMiniKQL::MakeString(strKey);
        } else if constexpr (DataSlot == NUdf::EDataSlot::Int64) {
            auto intKey = static_cast<i64>(genKey);
            if constexpr (collectStats) ++Freqs[intKey];
            return NYql::NUdf::TUnboxedValuePod{intKey};
        } else {
            throw std::runtime_error(TString{"Unsupported data slot"} + std::to_string(static_cast<int>(DataSlot)));
        }
    }

private:
    std::map<i64, size_t> Freqs;
};

template <NUdf::EDataSlot DataSlot>
class NormalDistributionKeyGenerator final : public IKeysGenerator<DataSlot> {
public:
    NormalDistributionKeyGenerator(
        NMiniKQL::TScopedAlloc& alloc,
        double mean = 0.0,
        double stddev = 10.0,
        size_t seed = std::random_device{}())
        : Generator(seed),
          Distribution(mean, stddev),
          Alloc(alloc) {};

    NYql::NUdf::TUnboxedValuePod NextKey() override {
        auto genKey = std::round(Distribution(Generator));
        return this->NextKeyImpl(genKey);
    }

private:
    std::mt19937 Generator;
    std::normal_distribution<double> Distribution;
    NMiniKQL::TScopedAlloc& Alloc;
};


class UniformStaleItemDecider final {
public:
    UniformStaleItemDecider(
        i64 left = 0,
        i64 right = 100,
        size_t seed = std::random_device{}())
        : Generator(seed),
          Distribution(left, right) {};

    size_t NextKey() {
        auto genKey = Distribution(Generator);
        return genKey;
    }

private:
    std::mt19937 Generator;
    std::uniform_int_distribution<size_t> Distribution;
};

class HashEqualUnboxedValue {
public:
    HashEqualUnboxedValue(const NKikimr::NMiniKQL::TType* keyType)
        : KeyTypeHelper(keyType) {}

    auto hash(auto& key) const {
        return KeyTypeHelper.GetValueHash()(key);
    }

    bool equal(auto& lhs, auto& rhs) const {
        return KeyTypeHelper.GetValueEqual()(lhs, rhs);
    }

private:
    const NKikimr::NMiniKQL::TKeyTypeContanerHelper<true, true, false> KeyTypeHelper;
};

template <NUdf::EDataSlot DataSlot, class TCache>
class CacheFixture {
public:
    CacheFixture(size_t cacheSize, const NMiniKQL::TKeyPayloadPairVector& keys, size_t seed, double mean, double stddev, std::optional<size_t> staleProbability)
        : Alloc(__LOCATION__),
          TypeEnv(Alloc),
          TypeBuilder(TypeEnv),
          Cache(cacheSize, TypeBuilder.NewDataType(DataSlot)),
          KeysGenerator(Alloc, mean, stddev, seed),
          StaleItemDecider(0, 100, seed),
          MakeStaleProbability(staleProbability) {
        for (auto& [key, value] : keys) {
            const auto now = std::chrono::steady_clock::now();
            const auto dt = std::chrono::minutes(1);
            Cache.Update(NUdf::TUnboxedValuePod{key}, NUdf::TUnboxedValuePod{value}, now + dt);
        }
    }

    void Get() {
        auto now = std::chrono::steady_clock::now();
        const auto key = KeysGenerator.NextKey();
        if (!MakeStaleProbability) {
            auto value = Cache.Get(NYql::NUdf::TUnboxedValuePod{key}, now);
            Y_DO_NOT_OPTIMIZE_AWAY(value);
            return;
        }

        bool make_stale = (StaleItemDecider.NextKey() < MakeStaleProbability);
        // Cache.Prune(now);
        auto value = Cache.Get(NYql::NUdf::TUnboxedValuePod{key}, now);
        if (make_stale) {
            static const auto LARGE_DURATION = std::chrono::hours(1);
            auto res = Cache.Get(NYql::NUdf::TUnboxedValuePod{key}, now + LARGE_DURATION);
            if (res) throw std::runtime_error("Stale item was found");
            static const auto DB_REQUEST_MOCK_DURATION = std::chrono::milliseconds(3);
            std::this_thread::sleep_for(DB_REQUEST_MOCK_DURATION);
            Cache.Update(NYql::NUdf::TUnboxedValuePod{key}, NUdf::TUnboxedValuePod{0}, std::move(now));
        }
    }

private:
    NMiniKQL::TScopedAlloc Alloc;
    NMiniKQL::TTypeEnvironment TypeEnv;
    NMiniKQL::TTypeBuilder TypeBuilder;
    TCache Cache;

    NormalDistributionKeyGenerator<DataSlot> KeysGenerator;
    UniformStaleItemDecider StaleItemDecider;
    std::optional<size_t> MakeStaleProbability;
};


template <NUdf::EDataSlot EDataSlot, class TCache>
void RunGetBenchmarkMultirhead(TString name, auto& keys, utils::Options options, size_t seed = std::random_device{}()) {
    if (options.ThreadCount > 1 && std::same_as<TCache, NMiniKQL::TUnboxedKeyValueLruCacheWithTtl>) {
        throw std::runtime_error("TUnboxedKeyValueLruCacheWithTtl doesn't support multithreading");
    }

    using namespace std::chrono_literals;
    std::cout << name << std::endl;

    const auto beforeBenchmarkRSS =  utils::PrintRSS();
    const auto benchmarkTime = 5s;

    CacheFixture<EDataSlot, TCache> cacheFixture{options.CacheSize, keys, seed, options.Mean, options.Stddev, options.MakeStaleProbability};
    
    std::vector<std::thread> threads;
    std::vector<ui64> counts(options.ThreadCount, 0);
    for (size_t i = 0; i < options.ThreadCount; ++i) {
        threads.emplace_back([&cacheFixture, &counts, i, benchmarkTime] {
            NMiniKQL::TScopedAlloc thread_local_alloc{__LOCATION__};

            auto start = std::chrono::high_resolution_clock::now();
            while (std::chrono::high_resolution_clock::now() - start < benchmarkTime) {
                cacheFixture.Get();
                ++counts[i];
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    const auto afterBenchmarkRSS = utils::PrintRSS();
    std::cout << "RSS before: " << beforeBenchmarkRSS << " MB" << std::endl;
    std::cout << "RSS after: " << afterBenchmarkRSS << " MB" << std::endl;
    std::cout << "RSS: " << afterBenchmarkRSS - beforeBenchmarkRSS << " MB" << std::endl;

    ui64 sumCount = std::accumulate(counts.begin(), counts.end(), 0ull);
    std::cout << "Data seed: " << seed << std::endl;
    std::cout << "Get() count: " << sumCount << std::endl;
    std::cout << "Get() average time: "
              << options.ThreadCount * std::chrono::duration_cast<std::chrono::nanoseconds>(benchmarkTime).count() / sumCount 
              << " ns" << std::endl;
    std::cout << std::endl;
}

}

int main(int argc, char** argv) {
    using namespace std::chrono_literals;

    NMiniKQL::TScopedAlloc main_alloc{__LOCATION__};

    using TYdbCache = NMiniKQL::TUnboxedKeyValueLruCacheWithTtl;
    using TYdbCacheSharded = NMiniKQL::TUnboxedValueLruCacheWithTTLSharded<256>;
    using TConcurrentCache = ConcurrentCacheWrapper<HashEqualUnboxedValue>;

    auto options = utils::GetOptions(argc, argv);
    const auto INITIAL_KEYS_COUNT = options.CacheSize * 2;

    NormalDistributionKeyGenerator<NUdf::EDataSlot::Int64> i64Generator{main_alloc, options.Mean, options.Stddev};
    NormalDistributionKeyGenerator<NUdf::EDataSlot::String> stringGenerator{main_alloc, options.Mean, options.Stddev};

    std::cout << "Cache size: " << options.CacheSize << '\n';
    std::cout << "Mean of normal distribution: " << options.Mean << '\n';
    std::cout << "Standard deviation of normal distribution: " << options.Stddev << '\n';

    {
        // i64 as a key, i64 as a value
        NMiniKQL::TKeyPayloadPairVector intCacheInitData;
        for (size_t i = 0; i < INITIAL_KEYS_COUNT; ++i) {
            intCacheInitData.emplace_back(i64Generator.NextKey(), i64Generator.NextKey());
        }
        Statistic<i64> intStat;
        intStat.Load(intCacheInitData);
        intStat.GetHist("int_hist.py");

        const auto uniqueKeysCount = intStat.UniqueCount();
        std::cout << "Unique data count: " << uniqueKeysCount << '\n';
        std::cout << "Initial cache fullness: " << 100 * std::min(uniqueKeysCount, options.CacheSize) / options.CacheSize << " %" << '\n';

        std::cout << '\n';

        std::cout << "Init i64 data RSS: " << utils::PrintRSS() << " MB" << std::endl;


        const size_t intSeed = std::random_device{}(); // seed for both benchmarks must be the same
        if (options.ThreadCount == 1) {
            RunGetBenchmarkMultirhead<NUdf::EDataSlot::Int64, TYdbCache>("TYdbCache: i64", intCacheInitData, options, intSeed);
        } else {
            RunGetBenchmarkMultirhead<NUdf::EDataSlot::Int64, TYdbCacheSharded>("TYdbCache: i64", intCacheInitData, options, intSeed);
        }
        RunGetBenchmarkMultirhead<NUdf::EDataSlot::Int64, TConcurrentCache>("TConcurrentCache: i64", intCacheInitData, options, intSeed);
    }

    {
        // TStringValue as a key, i64 as a value
        NMiniKQL::TKeyPayloadPairVector stringValueCacheInitData;
        for (size_t i = 0; i < INITIAL_KEYS_COUNT; ++i) {
            stringValueCacheInitData.emplace_back(stringGenerator.NextKey(), i64Generator.NextKey());
        }

        std::cout << "Init StringValue data RSS: " << utils::PrintRSS() << " MB" << std::endl;

        const size_t stringValueSeed = std::random_device{}(); // seed for both benchmarks must be the same
        if (options.ThreadCount == 1) {
            RunGetBenchmarkMultirhead<NUdf::EDataSlot::String, TYdbCache>("TYdbCache: TStringValue", stringValueCacheInitData, options, stringValueSeed);
        } else {
            RunGetBenchmarkMultirhead<NUdf::EDataSlot::String, TYdbCacheSharded>("TYdbCache: TStringValue", stringValueCacheInitData, options, stringValueSeed);
        }
        RunGetBenchmarkMultirhead<NUdf::EDataSlot::String, TConcurrentCache>("TConcurrentCache: TStringValue", stringValueCacheInitData, options, stringValueSeed);
    }
}
