#include "mkql_key_payload_value_lru_cache.h"
#include "simple_bench/concurrent_lru_wrapper.hpp"
#include "simple_bench/sharded_ydb_lru_cache.hpp"

#include <ydb/library/yql/minikql/mkql_type_builder.h>
#include <library/cpp/testing/unittest/registar.h>

namespace NKikimr::NMiniKQL {

namespace {

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

using TConcurrentCacheWrapper = ConcurrentCacheWrapper<HashEqualUnboxedValue>;

}

template <class TCache>
void TestSimple() {
    TScopedAlloc alloc(__LOCATION__);
    TTypeEnvironment typeEnv(alloc);
    TTypeBuilder typeBuilder(typeEnv);
    TCache cache(10, typeBuilder.NewDataType(NUdf::EDataSlot::Int32));

    const auto t0 = std::chrono::steady_clock::now();
    const auto dt = std::chrono::seconds(1);

    UNIT_ASSERT_VALUES_EQUAL(0, cache.Size());
    // Insert data
    cache.Update(NUdf::TUnboxedValuePod{10}, NUdf::TUnboxedValuePod{100}, t0 + 10 * dt);
    cache.Update(NUdf::TUnboxedValuePod{20}, NUdf::TUnboxedValuePod{200}, t0 + 20 * dt);
    cache.Update(NUdf::TUnboxedValuePod{30}, NUdf::TUnboxedValuePod{}, t0 + 30 * dt); //empty(absent) value
    UNIT_ASSERT_VALUES_EQUAL(3, cache.Size());
    // Get data
    for (size_t i = 0; i != 3; ++i){
        {
            auto v = cache.Get(NUdf::TUnboxedValuePod{10}, t0 + (i * 3 + 0) * dt );
            UNIT_ASSERT(v);
            UNIT_ASSERT_VALUES_EQUAL(100, v->template Get<i32>());
        }
        {
            auto v = cache.Get(NUdf::TUnboxedValuePod{20}, t0 + (i * 3 + 1) * dt);
            UNIT_ASSERT(v);
            UNIT_ASSERT_VALUES_EQUAL(200, v->template Get<i32>());
        }
        {
            auto v = cache.Get(NUdf::TUnboxedValuePod{30}, t0 + (i * 3 + 2) * dt);
            UNIT_ASSERT(v);
            UNIT_ASSERT(!v->HasValue());
        }
        UNIT_ASSERT_VALUES_EQUAL(3, cache.Size());
    }

    { // Update value
        auto v = cache.Get(NUdf::TUnboxedValuePod{10}, t0 + dt);
        UNIT_ASSERT(v);
        UNIT_ASSERT_VALUES_EQUAL(100, v->template Get<i32>());
        UNIT_ASSERT_VALUES_EQUAL(3, cache.Size());

        cache.Update(NUdf::TUnboxedValuePod{10}, NUdf::TUnboxedValuePod{1000}, t0 + 15 * dt);

        v = cache.Get(NUdf::TUnboxedValuePod{10}, t0 + 14 * dt);
        UNIT_ASSERT(v);
        UNIT_ASSERT_VALUES_EQUAL(1000, v->template Get<i32>());
        UNIT_ASSERT_VALUES_EQUAL(3, cache.Size());
    }

    { // Get outdated
        auto v = cache.Get(NUdf::TUnboxedValuePod{10}, t0 + 15 * dt );
        UNIT_ASSERT(!v);
        UNIT_ASSERT_VALUES_EQUAL(2, cache.Size());
    }

    { //still exists
        auto v = cache.Get(NUdf::TUnboxedValuePod{20}, t0 + 15 * dt);
        UNIT_ASSERT(v);
        UNIT_ASSERT_VALUES_EQUAL(200, v->template Get<i32>());
    }

    // Get all outdated
    UNIT_ASSERT(!cache.Get(NUdf::TUnboxedValuePod{20}, t0 + 30 * dt));
    UNIT_ASSERT_VALUES_EQUAL(1, cache.Size());

    UNIT_ASSERT(!cache.Get(NUdf::TUnboxedValuePod{30}, t0 + 30 * dt));
    UNIT_ASSERT_VALUES_EQUAL(0, cache.Size());
}

template <class TCache>
void TestLruEviction() {
    TScopedAlloc alloc(__LOCATION__);
    TTypeEnvironment typeEnv(alloc);
    TTypeBuilder typeBuilder(typeEnv);
    TCache cache(10, typeBuilder.NewDataType(NUdf::EDataSlot::Int32));

    const auto t0 = std::chrono::steady_clock::now();
    const auto dt = std::chrono::seconds(1);

    UNIT_ASSERT_VALUES_EQUAL(0, cache.Size());
    // Insert data
    for (size_t i = 1; i <= 10; ++i){
        cache.Update(NUdf::TUnboxedValuePod{i}, NUdf::TUnboxedValuePod{}, t0 + dt);
    }

    UNIT_ASSERT_VALUES_EQUAL(10, cache.Size());

    { // The first element should be evicted
        cache.Update(NUdf::TUnboxedValuePod{11}, NUdf::TUnboxedValuePod{}, t0 + dt);
        UNIT_ASSERT(!cache.Get(NUdf::TUnboxedValuePod{1}, t0 + 0 * dt));
        UNIT_ASSERT_VALUES_EQUAL(10, cache.Size());
    }

    { // The second element should be touched and the third should be evicted
        UNIT_ASSERT(cache.Get(NUdf::TUnboxedValuePod{2}, t0 + 0 * dt));
        cache.Update(NUdf::TUnboxedValuePod{12}, NUdf::TUnboxedValuePod{}, t0 + dt);
        UNIT_ASSERT(!cache.Get(NUdf::TUnboxedValuePod{3}, t0 + 0 * dt));
        UNIT_ASSERT_VALUES_EQUAL(10, cache.Size());
    }
}

Y_UNIT_TEST_SUITE(TAllLruCachesTest) {

    Y_UNIT_TEST(SimpleTUnboxedKeyValueLruCacheWithTtl) {
        TestSimple<TUnboxedKeyValueLruCacheWithTtl>();
    }

    Y_UNIT_TEST(SimpleTConcurrentCacheWrapper) {
        TestSimple<TConcurrentCacheWrapper>();
    }

    Y_UNIT_TEST(SimpleTUnboxedValueLruCacheWithTTLSharded) {
        TestSimple<TUnboxedValueLruCacheWithTTLSharded<256>>();
    }


    Y_UNIT_TEST(LruEviction) {
        TestLruEviction<TUnboxedKeyValueLruCacheWithTtl>();
    }

    Y_UNIT_TEST(LruEvictionConcurrentCacheWrapper) {
        TestLruEviction<TConcurrentCacheWrapper>();
    }

    // TODO: Sharded LRU Cache behaves differently than TestLruEviction expects
    // Y_UNIT_TEST(LruEvictionTUnboxedValueLruCacheWithTTLSharded) {
    //     TestLruEviction<TUnboxedValueLruCacheWithTTLSharded<256>>();
    // }


    Y_UNIT_TEST(GarbageCollection) {
        //TBD
    }
}

} //namespace namespace NKikimr::NMiniKQL
