#pragma once

#include <ydb/library/yql/public/udf/udf_value.h>

#include "../concurrent-lru-cache.h"

#include <optional>

template <class HashEqual>
class ConcurrentCacheWrapper final {
public:
    using TCache = HPHP::ConcurrentLRUCache<
        NKikimr::NUdf::TUnboxedValue,
        NKikimr::NUdf::TUnboxedValue,
        HashEqual>;

    ConcurrentCacheWrapper(size_t maxSize, HashEqual hash_equal)
        : Cache(maxSize, hash_equal) {}

    void Update(NKikimr::NUdf::TUnboxedValue&& key, NKikimr::NUdf::TUnboxedValue&& value, std::chrono::time_point<std::chrono::steady_clock>&& /*expiration*/) {
        Cache.insert(std::move(key), std::move(value));
    }

    void Prune(const std::chrono::time_point<std::chrono::steady_clock>& /*now*/) {
        // There is no need to prune the cache because the entries are never outdated
    }

    std::optional<NKikimr::NUdf::TUnboxedValue> Get(const NKikimr::NUdf::TUnboxedValue key, const std::chrono::time_point<std::chrono::steady_clock>& /*now*/) {
        typename TCache::ConstAccessor acs;
        if (Cache.find(acs, key)) {
            return *acs;
        }
        return std::nullopt;
    }

private:
    TCache Cache;
};
