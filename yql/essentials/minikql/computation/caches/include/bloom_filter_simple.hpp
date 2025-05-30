#pragma once

#include <bitset>
#include <cassert>
#include <fstream>
#include <vector>

#include "utils.hpp"

namespace cache {

template <class T, size_t kSize>
class BloomFilter {
 public:
  using HashFunc = size_t (*)(T key);

  template <class... Funcs>
  explicit BloomFilter(Funcs&&... funcs)
      : hash_funcs_({funcs...}), bloom_filter_() {}

  void Add(T key) {
    for (const auto& hash_func : hash_funcs_) {
      bloom_filter_[hash_func(key) % kSize] = true;
    }
  }

  bool Test(T key) const {
    for (const auto& hash_func : hash_funcs_) {
      if (!bloom_filter_[hash_func(key) % kSize]) return false;
    }

    return true;
  }

  void Clear() { bloom_filter_.reset(); }

  [[nodiscard]]
  double LoadFactor() const {
    return static_cast<double>(bloom_filter_.count()) / kSize;
  }

  void Load(std::ifstream& file) {
    std::vector<unsigned char> buf((kSize + 7) >> 3);
    utils::BinaryRead(file, buf.data(), buf.size());
    bloom_filter_ = bitset_from_bytes<kSize>(buf);
  }

  void Store(std::ofstream& file) const {
    auto bytes = bitset_to_bytes(bloom_filter_);
    utils::BinaryWrite(file, bytes.data(), bytes.size());
  }

  bool operator==(const BloomFilter& other) const {
    return bloom_filter_ == other.bloom_filter_;
  }

 private:
  // https://stackoverflow.com/a/7463972
  template <size_t N>
  std::vector<unsigned char> bitset_to_bytes(const std::bitset<N>& bs) const {
    std::vector<unsigned char> result((N + 7) >> 3);
    for (size_t j = 0; j < N; ++j) result[j >> 3] |= (bs[j] << (j & 7));
    return result;
  }

  template <size_t N>
  std::bitset<N> bitset_from_bytes(
      const std::vector<unsigned char>& buf) const {
    assert(buf.size() == ((N + 7) >> 3));
    std::bitset<N> result;
    for (size_t j = 0; j < N; ++j) result[j] = ((buf[j >> 3] >> (j & 7)) & 1);
    return result;
  }

  std::vector<HashFunc> hash_funcs_;
  std::bitset<kSize> bloom_filter_;
};

}  // namespace cache
