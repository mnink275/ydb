#pragma once

#include <ydb/library/yql/minikql/computation/mkql_key_payload_value_lru_cache.h>

#include <mutex>

namespace NKikimr::NMiniKQL {

template <size_t shardCount = 256>
class TUnboxedValueLruCacheWithTTLSharded {
public:
    TUnboxedValueLruCacheWithTTLSharded(size_t maxSize, const TType* keyType) {
        for (size_t i = 0; i < shardCount; ++i) {
            Caches[i].emplace(maxSize / shardCount, keyType);
        }
    }

    void Update(NYql::NUdf::TUnboxedValue&& key, NYql::NUdf::TUnboxedValue&& value, std::chrono::time_point<std::chrono::steady_clock>&& expiration) {
        const size_t shard = GetShardIdx(key);
        std::lock_guard lock(Mutexes[shard]);
        Caches[shard]->Update(std::move(key), std::move(value), std::move(expiration));
    }

    std::optional<NYql::NUdf::TUnboxedValue> Get(const NYql::NUdf::TUnboxedValue key, const std::chrono::time_point<std::chrono::steady_clock>& now) {
        const size_t shard = GetShardIdx(key);
        std::lock_guard lock(Mutexes[shard]);
        // Caches[shard]->Prune(now);
        return Caches[shard]->Get(key, now);
    }

    size_t Size() const {
        size_t size = 0;
        for (size_t i = 0; i < shardCount; ++i) {
            size += Caches[i]->Size();
        }
        return size;
    }

private:
    size_t GetShardIdx(const NYql::NUdf::TUnboxedValue& key) const {
        if (key.IsString()) {
            return key.AsRawStringValue()->Data()[0];
        } else {
            return key.Get<i64>() % shardCount;
        }
    }

    std::array<std::mutex, shardCount> Mutexes{};
    std::array<std::optional<TUnboxedKeyValueLruCacheWithTtl>, shardCount> Caches{};
};

}
