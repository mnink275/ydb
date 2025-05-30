#pragma once

#include <yql/essentials/public/udf/udf_value.h>
#include <yql/essentials/minikql/computation/mkql_computation_node_pack.h>

#include "cache_config.hpp"
#include "tiny_lfu_cms.hpp"
#include "utils.hpp"

#include <immintrin.h>

namespace cache {

inline size_t SmallPageIndex(Key key) noexcept {
  key &= (1ull << (8ull * sizeof(Key) - LARGE_PAGE_SHIFT)) - 1ull;
  return key % SMALL_PAGE_NUMBER;
}

// inline size_t SmallPageIndex(Key key) noexcept {
//     key &= (1ull << (8ull * sizeof(Key) - LARGE_PAGE_SHIFT)) - 1ull;
//     return key >> (8ull * sizeof(Key) - LARGE_PAGE_SHIFT - SMALL_PAGE_SHIFT);
// }

#if USE_SIMD_FLAG

inline size_t FindKeyIdxSIMD8(
    Key key, const std::array<Key, SMALL_PAGE_SIZE>& records) noexcept {
  const auto x = _mm256_set1_epi32(key);
  // const auto x = _mm256_broadcastd_epi32(_mm_loadu_si32(&key));
  const auto N = records.size();
  assert(N % 8 == 0);

  assert(reinterpret_cast<std::uintptr_t>(records.data()) % 32 == 0);
  for (size_t block_id = 0; block_id < N; block_id += 8) {
    const auto y =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(&records[block_id]));
    const auto m = _mm256_cmpeq_epi32(x, y);
    if (!_mm256_testz_si256(m, m)) {
      const auto mask = _mm256_movemask_ps(_mm256_castsi256_ps(m));
      return block_id + __builtin_ctz(mask);
    }
  }

  return N;
}

inline size_t FindKeyIdxSIMD16(
    Key key, const std::array<Key, SMALL_PAGE_SIZE>& records) noexcept {
  const auto x = _mm256_set1_epi32(key);
  const auto N = records.size();
  assert(N % 8 == 0);

  assert(reinterpret_cast<std::uintptr_t>(records.data()) % 32 == 0);
  for (size_t block_id = 0; block_id < N; block_id += 16) {
    const auto y1 =
        _mm256_load_si256(reinterpret_cast<const __m256i*>(&records[block_id]));
    const auto y2 = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(&records[block_id + 8]));
    const auto m1 = _mm256_cmpeq_epi32(x, y1);
    const auto m2 = _mm256_cmpeq_epi32(x, y2);
    const auto m = _mm256_or_si256(m1, m2);
    if (!_mm256_testz_si256(m, m)) {
      const auto mask = (_mm256_movemask_ps(_mm256_castsi256_ps(m2)) << 8) +
                        _mm256_movemask_ps(_mm256_castsi256_ps(m1));
      return block_id + __builtin_ctz(mask);
    }
  }

  return N;
}

#endif

inline size_t FindKeyIdx(
    Key key, const std::array<Key, SMALL_PAGE_SIZE>& records) noexcept {
  const auto* it = std::find(records.begin(), records.end(), key);
  return std::distance(records.begin(), it);
}

class SmallPageAdvanced {
 public:
  struct Payload {
    uint32_t expiration_time;
    NKikimr::NUdf::TUnboxedValue value;
  };

  explicit SmallPageAdvanced(TTinyLFU& tiny_lfu) noexcept
      : tiny_lfu_(tiny_lfu) {
    Clear();
  }

#if ENABLE_STATISTICS_FLAG
  double GetFillFactor() const {
    uint32_t cnt = 0;
    for (const auto& r : records_) {
      if (r != INVALID_HASH) {
        ++cnt;
      }
    }
    return cnt / static_cast<double>(records_.size());
  }

  uint64_t GetNumEvictionsHighFreq() const { return num_evictions_high_freq_; }
  uint64_t GetNumDroppedKeysLowFreq() const { return drop_keys_due_low_freq_; }
#endif

  void Clear() noexcept {
    records_.fill(INVALID_HASH);
    payload_.fill({0, UVPod{INVALID_HASH}});
    last_free_slot_ = 0;
  }

  void Load(const char* buffer) noexcept {
#if USE_BF_FLAG
    bloom_filter_.Load(file);
#endif

    utils::LoadArrayFromBuffer(buffer, records_);
    std::advance(buffer, records_.size() * sizeof(records_[0]));

    static constexpr auto TTimeSize = sizeof(Payload::expiration_time);
    for (size_t i = 0; i < payload_.size(); ++i) {
      utils::BinaryRead(
        buffer + i * TTimeSize,
        &payload_[i].expiration_time,
        TTimeSize);
    }
    std::advance(buffer, payload_.size() * TTimeSize);

    utils::BinaryRead(buffer, &last_free_slot_, sizeof(last_free_slot_));
    std::advance(buffer, sizeof(last_free_slot_));
  }

  void LoadUV(const UV& list_iter) noexcept {
#if USE_BF_FLAG
    bloom_filter_.Load(file);
#endif
    UV list_value;
    for (size_t i = 0; i < SMALL_PAGE_SIZE; ++i) {
      list_iter.Next(list_value);
      payload_[i].value = std::move(list_value);
    }
  }

  void LoadUV(UV& value, size_t idx) noexcept {
    payload_[idx].value = std::move(value);
  }

  void Store(char* buffer) const noexcept {
    utils::StoreArrayToBuffer(buffer, records_);
    std::advance(buffer, records_.size() * sizeof(records_[0]));

    static constexpr auto TTimeSize = sizeof(Payload::expiration_time);
    for (size_t i = 0; i < payload_.size(); ++i) {
      utils::BinaryWrite(
        buffer + i * TTimeSize,
        &payload_[i].expiration_time,
        TTimeSize);
    }
    std::advance(buffer, payload_.size() * TTimeSize);

    utils::BinaryWrite(buffer, &last_free_slot_, sizeof(last_free_slot_));
    std::advance(buffer, sizeof(last_free_slot_));
  }

  template <bool FastPack>
  void StoreUV(NKikimr::NMiniKQL::TValuePackerTransport<FastPack>& packer) const noexcept {
#if USE_BF_FLAG
    bloom_filter_.Store(file);
#endif
    for (size_t i = 0; i < SMALL_PAGE_SIZE; ++i) {
      packer.AddItem(payload_[i].value);
    }
  }

  UV Get(Key key, uint32_t now) noexcept {
#if USE_BF_FLAG
    if (!bloom_filter_.Test(key)) {
      return false;
    }
#endif

#if USE_SIMD_FLAG
    const auto i = FindKeyIdxSIMD16(key, records_);
#else
    const auto i = FindKeyIdx(key, records_);
#endif
    if (i < records_.size()) {
      if (CheckEvictedByTTL(i, now)) return EMPTY_UV;
      tiny_lfu_.Add(key);
      const auto new_pos = Raise(i);
      return payload_[new_pos].value;
    }

    return EMPTY_UV;
  }

  bool Update(Key key, UV&& value, uint32_t expiration_time) noexcept {
    if (records_.back() == INVALID_HASH) {
      assert(last_free_slot_ < records_.size());
      assert(records_[last_free_slot_] == INVALID_HASH);

      records_[last_free_slot_] = key;
      payload_[last_free_slot_] = {expiration_time, std::move(value)};
      tiny_lfu_.Add(key);

#if USE_BF_FLAG
      bloom_filter_.Add(key);
#endif

      Raise(last_free_slot_);
      last_free_slot_++;
      return true;
    }

    auto victim = records_.back();

    auto est_victim = tiny_lfu_.Estimate(victim);
    auto est_key = tiny_lfu_.Estimate(key);
    if (est_victim < est_key) {
      records_.back() = key;
      payload_.back() = {expiration_time, std::move(value)};
      tiny_lfu_.Add(key);

#if ENABLE_STATISTICS_FLAG
      num_evictions_high_freq_ += (est_victim != 0);
#endif

#if USE_BF_FLAG
      bloom_filter_.Add(key);
#endif

      Raise(records_.size() - 1);
      return true;
    }
#if ENABLE_STATISTICS_FLAG
    drop_keys_due_low_freq_++;
#endif
    return false;
  }

  bool operator==(const SmallPageAdvanced& other) const noexcept {
    return records_ == other.records_;
  }

 private:
  size_t Raise(
      size_t i) noexcept {  // поднимает запись i в соответствии с частотой
    while (i > 0 && tiny_lfu_.Estimate(records_[i - 1]) <
                        tiny_lfu_.Estimate(records_[i])) {
      std::swap(records_[i - 1], records_[i]);
      std::swap(payload_[i - 1], payload_[i]);
      --i;
    }

    return i;
  }

  void SiftDown(size_t i) noexcept {
    while (i + 1 < SMALL_PAGE_SIZE && records_[i + 1] != INVALID_HASH) {
      std::swap(records_[i], records_[i + 1]);
      std::swap(payload_[i], payload_[i + 1]);
      ++i;
    }
    last_free_slot_--;
    assert(last_free_slot_ == i);
  }

  bool CheckEvictedByTTL(size_t idx, uint32_t now) {
    bool should_evict = false;
    if constexpr (cache::TTL_EVICTION_PROB > 0.0) {
      static std::mt19937 gen(BERNOULLI_SEED ? BERNOULLI_SEED
                                             : std::random_device{}());
      static std::bernoulli_distribution dist(cache::TTL_EVICTION_PROB);
      should_evict = dist(gen);
    } else {
      should_evict = payload_[idx].expiration_time < now;
    }

    if (should_evict) {
      records_[idx] = INVALID_HASH;
      SiftDown(idx);
      return true;
    }

    return false;
  }

 private:
  alignas(32) std::array<Key, SMALL_PAGE_SIZE> records_{};
  std::array<Payload, SMALL_PAGE_SIZE> payload_{};

  uint32_t last_free_slot_{0};
  static_assert((1ull << sizeof(last_free_slot_) * 8) >= SMALL_PAGE_SIZE);

  TTinyLFU& tiny_lfu_;

 public:
  static constexpr size_t kTrivialDataSizeInBytes =
      SMALL_PAGE_SIZE * sizeof(Key) +
      SMALL_PAGE_SIZE * sizeof(Payload::expiration_time) +
      sizeof(last_free_slot_);

#if USE_BF_FLAG
  BloomFilter<Key, SMALL_PAGE_SIZE * 6> bloom_filter_{
      [](Key key) { return static_cast<size_t>(key) * 2654435761 % 2 ^ 32; },
      [](Key key) {
        key += ~(key << 15);
        key ^= (key >> 10);
        key += (key << 3);
        key ^= (key >> 6);
        key += ~(key << 11);
        key ^= (key >> 16);
        return static_cast<size_t>(key);
      },
      [](Key key) {
        int c2 = 0x27d4eb2d;
        key = (key ^ 61) ^ (key >> 16);
        key = key + (key << 3);
        key = key ^ (key >> 4);
        key = key * c2;
        key = key ^ (key >> 15);
        return static_cast<size_t>(key);
      },
      [](Key key) {
        key = (key + 0x7ed55d16) + (key << 12);
        key = (key ^ 0xc761c23c) ^ (key >> 19);
        key = (key + 0x165667b1) + (key << 5);
        key = (key + 0xd3a2646c) ^ (key << 9);
        key = (key + 0xfd7046c5) + (key << 3);
        key = (key ^ 0xb55a4f09) ^ (key >> 16);
        return static_cast<size_t>(key);
      }};
#endif

#if ENABLE_STATISTICS_FLAG
  uint64_t num_evictions_high_freq_{0};
  uint64_t drop_keys_due_low_freq_{0};
#endif
};

using SmallPage = SmallPageAdvanced;

}  // namespace cache
