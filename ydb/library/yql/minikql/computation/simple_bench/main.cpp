#include "../mkql_key_payload_value_lru_cache.h"

#include <ydb/library/yql/minikql/mkql_type_builder.h>
#include <util/generic/xrange.h>

#include <ydb/core/ymq/actor/sha256.h>

#include "../concurrent-lru-cache.h"
#include "ydb/library/yql/minikql/mkql_string_util.h"

#include <optional>
#include <fstream>
#include <chrono>

using namespace NKikimr;

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

template <NUdf::EDataSlot DataSlot>
class KeysGenerator final {
public:
    KeysGenerator(
        NMiniKQL::TScopedAlloc& alloc,
        double mean = 0.0,
        double stddev = 10.0,
        size_t seed = std::random_device{}())
        : Generator(seed),
          Distribution(mean, stddev),
          Alloc(alloc) {};

    NYql::NUdf::TUnboxedValuePod NextKey() {
        auto genKey = std::round(Distribution(Generator));
        if constexpr (DataSlot == NUdf::EDataSlot::String) {
            auto strKey = NKikimr::NSQS::CalcSHA256(std::to_string(genKey));
            return NMiniKQL::MakeString(strKey);
        } else if constexpr (DataSlot == NUdf::EDataSlot::Int64) {
            auto intKey = static_cast<i64>(genKey);
            ++Freqs[intKey];
            return NYql::NUdf::TUnboxedValuePod{intKey};
        } else {
            throw std::runtime_error(TString{"Unsupported data slot"} + std::to_string(static_cast<int>(DataSlot)));
        }
    }

    size_t UniqueCount() {
        return Freqs.size();
    }

private:
    std::mt19937 Generator;
    std::normal_distribution<double> Distribution;
    NMiniKQL::TScopedAlloc& Alloc;
    std::map<i64, i64> Freqs;
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

template <class HashEqual>
class ConcurrentCacheWrapper {
public:
    using TCache = HPHP::ConcurrentLRUCache<NUdf::TUnboxedValue, NUdf::TUnboxedValue, HashEqualUnboxedValue>;

    ConcurrentCacheWrapper(size_t maxSize, HashEqual hash_equal)
        : Cache(maxSize, hash_equal) {}

    void Update(NUdf::TUnboxedValue&& key, NUdf::TUnboxedValue&& value, std::chrono::time_point<std::chrono::steady_clock>&& /*expiration*/) {
        Cache.insert(std::move(key), std::move(value));
    }

    std::optional<NUdf::TUnboxedValue> Get(const NUdf::TUnboxedValue key, const std::chrono::time_point<std::chrono::steady_clock>& /*now*/) {
        TCache::ConstAccessor acs;
        if (Cache.find(acs, key)) {
            return *acs;
        }
        return std::nullopt;
    }

private:
    TCache Cache;
};

template <NUdf::EDataSlot DataSlot, class TCache>
class CacheFixture {
public:
    CacheFixture(size_t cacheSize, const NMiniKQL::TKeyPayloadPairVector& keys, size_t seed, double mean, double stddev)
        : Alloc(__LOCATION__),
          TypeEnv(Alloc),
          TypeBuilder(TypeEnv),
          Cache(cacheSize, TypeBuilder.NewDataType(DataSlot)),
          KeysGenerator(Alloc, mean, stddev, seed) {
        for (auto& [key, value] : keys) {
            const auto now = std::chrono::steady_clock::now();
            const auto dt = std::chrono::minutes(1);
            Cache.Update(NUdf::TUnboxedValuePod{key}, NUdf::TUnboxedValuePod{value}, now + dt);
        }
    }

    void Get() {
        const auto now = std::chrono::steady_clock::now();
        auto res = Cache.Get(KeysGenerator.NextKey(), now);
        Y_DO_NOT_OPTIMIZE_AWAY(res);
    }

private:
    NMiniKQL::TScopedAlloc Alloc;
    NMiniKQL::TTypeEnvironment TypeEnv;
    NMiniKQL::TTypeBuilder TypeBuilder;
    TCache Cache;

    KeysGenerator<DataSlot> KeysGenerator;
};

struct BenchContext {
    const size_t CacheSize;
    const double Mean;
    const double Stddev;
};

template <NUdf::EDataSlot EDataSlot, class TCache>
void RunGetBenchmark(TString name, auto& keys, BenchContext benchCtx, size_t seed = std::random_device{}()) {
    using namespace std::chrono_literals;
    std::cout << name << std::endl;

    CacheFixture<EDataSlot, TCache> concurrentCacheFixture{benchCtx.CacheSize, keys, seed, benchCtx.Mean, benchCtx.Stddev};
    auto start = std::chrono::high_resolution_clock::now();
    size_t count = 0;
    while (std::chrono::high_resolution_clock::now() - start < 5s) {
        concurrentCacheFixture.Get();
        ++count;
    }

    std::cout << "Get() count: " <<  count << std::endl;
    std::cout << "Get() average time: " << i64(1'000'000'000) * 5 / count  << " ns" << std::endl;
    std::cout << std::endl;
}

template <class Clock>
void DisplayPrecision() {
    // https://stackoverflow.com/questions/8386128/how-to-get-the-precision-of-high-resolution-clock
    std::chrono::duration<double, std::nano> ns = typename Clock::duration(1);
    std::cout << ns.count() << " ns\n";
}

}

int main() {
    DisplayPrecision<std::chrono::high_resolution_clock>();
    DisplayPrecision<std::chrono::steady_clock>();
    DisplayPrecision<std::chrono::system_clock>();

    NMiniKQL::TScopedAlloc main_alloc{__LOCATION__};

    using TYdbCache = NMiniKQL::TUnboxedKeyValueLruCacheWithTtl;
    using TConcurrentCache = ConcurrentCacheWrapper<HashEqualUnboxedValue>;

    constexpr size_t INITIAL_KEYS_COUNT = 1e5;
    BenchContext benchCtx{
        .CacheSize = static_cast<size_t>(1e4),
        .Mean = 0.0,
        .Stddev = 1e4
    };
    KeysGenerator<NUdf::EDataSlot::Int64> i64Generator{main_alloc, benchCtx.Mean, benchCtx.Stddev};
    KeysGenerator<NUdf::EDataSlot::String> stringGenerator{main_alloc, benchCtx.Mean, benchCtx.Stddev};

    std::cout << "Cache size: " << benchCtx.CacheSize << '\n';
    std::cout << "Mean of normal distribution: " << benchCtx.Mean << '\n';
    std::cout << "Standart deviation of normal distribution: " << benchCtx.Stddev << '\n';

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
    std::cout << "Unique i64 count: " << i64Generator.UniqueCount() << '\n';
    std::cout << "Initial cache fullness: " << 100 * uniqueKeysCount / std::min(uniqueKeysCount, benchCtx.CacheSize) << " %" << '\n';

    std::cout << '\n';

    const size_t intSeed = std::random_device{}(); // seed for both benchmarks must be the same
    RunGetBenchmark<NUdf::EDataSlot::Int64, TYdbCache>("TYdbCache: i64", intCacheInitData, benchCtx, intSeed);
    RunGetBenchmark<NUdf::EDataSlot::Int64, TConcurrentCache>("TConcurrentCache: i64", intCacheInitData, benchCtx, intSeed);

    // TStringValue as a key, i64 as a value
    NMiniKQL::TKeyPayloadPairVector stringValueCacheInitData;
    for (size_t i = 0; i < INITIAL_KEYS_COUNT; ++i) {
        stringValueCacheInitData.emplace_back(stringGenerator.NextKey(), i64Generator.NextKey());
    }
    // Statistic<NYql::NUdf::TStringValue> stringStat;
    // stringStat.Load(stringValueCacheInitData);
    // std::cout << "Unique data count: " << stringStat.UniqueCount() << '\n';
    // stringStat.GetHist("string_hist.py");

    const size_t stringValueSeed = std::random_device{}(); // seed for both benchmarks must be the same
    RunGetBenchmark<NUdf::EDataSlot::String, TYdbCache>("TYdbCache: TStringValue", stringValueCacheInitData, benchCtx, stringValueSeed);
    RunGetBenchmark<NUdf::EDataSlot::String, TConcurrentCache>("TConcurrentCache: TStringValue", stringValueCacheInitData, benchCtx, stringValueSeed);
}
