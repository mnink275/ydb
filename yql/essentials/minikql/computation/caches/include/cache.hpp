#pragma once

#include <cassert>
#include <filesystem>

#include "cache_config.hpp"
#include "large_page_provider.hpp"
#include "lru.hpp"

namespace cache {

class Cache {
 public:
  Cache(const NKikimr::NMiniKQL::TTypeBuilder& typeBuilder,
        const NKikimr::NMiniKQL::THolderFactory& holderFactory,
        std::filesystem::path dir_path = "./data")
      : tiny_lfu_(),
        provider_(std::move(dir_path), tiny_lfu_, typeBuilder, holderFactory)
#if USE_LRU_FLAG
        ,
        lru_(static_cast<size_t>(LRU_SIZE))
#endif
  {
  }

  UV Get(Key key, uint32_t now) {
#if USE_LRU_FLAG
    if (auto value = lru_.Get(key, now)) return value;
#endif

    auto maybe_large_page = provider_.Get</*CalledOnUpdate=*/false>(key);

    if (!maybe_large_page.has_value()) return EMPTY_UV;

    return maybe_large_page->Get(key, now);
  }

  void Update(Key key, UV&& value, uint32_t expiration_time) {
#if USE_LRU_FLAG
    auto lru_evicted = lru_.Update(key, std::move(value), expiration_time);
    if (!lru_evicted) return;

    key = lru_evicted->key;
    value = std::move(lru_evicted->value);
    expiration_time = std::move(lru_evicted->expiration_time);
#endif

    auto maybe_large_page = provider_.Get</*CalledOnUpdate=*/true>(key);

    if (!maybe_large_page.has_value()) return;

    maybe_large_page->Update(key, std::move(value), expiration_time);
  }

  void Store() const { provider_.Store(); }

 private:
  TTinyLFU tiny_lfu_{};
  LargePageProvider provider_;

#if USE_LRU_FLAG
  LRU<Key> lru_;
#endif
};

}  // namespace cache
