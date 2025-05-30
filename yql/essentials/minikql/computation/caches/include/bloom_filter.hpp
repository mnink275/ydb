#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>

#include "utils.hpp"

namespace cache {

namespace detail {

constexpr double kLog2 = 0.693147181;
constexpr double kLogFalsePositiveRate = -4.605170186;  // std::log(0.01)

constexpr uint32_t CalcNumBits(uint32_t capacity) {
  return std::max(1024U,
                  std::bit_ceil(static_cast<uint32_t>(
                      capacity * -kLogFalsePositiveRate / (kLog2 * kLog2))));
}

}  // namespace detail

template <size_t Capacity>
class BloomFilter final {
 public:
  static constexpr auto kNumBits = detail::CalcNumBits(Capacity);
  static constexpr auto kNumHashFunc =
      std::max(2U, static_cast<uint32_t>(0.7 * kNumBits / Capacity));
  static constexpr auto kDataSize = (kNumBits + 63) / 64;

  bool Add(uint64_t key) noexcept {
    const auto h1 = static_cast<uint32_t>(key);
    const auto h2 = static_cast<uint32_t>(key >> 32);
    bool was_added = true;
    for (uint32_t i = 0; i < kNumHashFunc; i++) {
      const auto bit = (h1 + (i * h2)) & (kNumBits - 1);
      const auto mask = 1ULL << (bit % 64);
      auto& block = data_[bit / 64];
      const auto prev_bit_value = (block & mask) >> (bit % 64);
      block |= mask;
      was_added &= prev_bit_value;
    }

    return was_added;
  }

  bool Test(uint64_t key) const noexcept {
    const auto h1 = static_cast<uint32_t>(key);
    const auto h2 = static_cast<uint32_t>(key >> 32);
    bool was_added = true;
    for (uint32_t i = 0; i < kNumHashFunc; i++) {
      const auto bit = (h1 + (i * h2)) & (kNumBits - 1);
      const auto mask = 1ULL << (bit % 64);
      const auto block = data_[bit / 64];
      was_added &= (block & mask) >> (bit % 64);
    }

    return was_added;
  }

  void Clear() noexcept { std::fill(data_.begin(), data_.end(), 0); }

  void Load(std::ifstream& file) {
    utils::BinaryRead(file, data_.data(), data_.size() * sizeof(data_[0]));
  }

  void Store(std::ofstream& file) const {
    utils::BinaryWrite(file, data_.data(), data_.size() * sizeof(data_[0]));
  }

  bool operator==(const BloomFilter& other) const {
    return kNumBits == other.kNumBits && kNumHashFunc == other.kNumHashFunc &&
           data_ == other.data_;
  }

 private:
  std::array<uint64_t, kDataSize> data_{};
};

}  // namespace cache
