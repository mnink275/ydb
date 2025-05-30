#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>

#include "bloom_filter.hpp"
#include "cm_sketch.hpp"
#include "utils.hpp"

namespace cache {

template <class T, size_t SampleSize, size_t NumCounters, bool UseDoorKeeper>
class TinyLFU;

template <class T, size_t SampleSize, size_t NumCounters>
class TinyLFU<T, SampleSize, NumCounters, false> final {
 public:
  using TGlobalCounter = uint32_t;
  static_assert(SampleSize <= std::numeric_limits<TGlobalCounter>::max());

  void Add(T key) noexcept {
    if (key == std::numeric_limits<T>::max()) return;

    sketch_.Add(key);

    if constexpr (SampleSize > 0) {
      if (++global_counter_ >= SampleSize) Reset();
    }
  }

  size_t Estimate(T key) const noexcept {
    if (key == std::numeric_limits<T>::max()) return 0;

    auto frequency = sketch_.Estimate(key);
    return frequency;
  }

  void Load(std::ifstream& file) {
    utils::BinaryRead(file, &global_counter_, sizeof(global_counter_));
    sketch_.Load(file);
  }

  void Store(std::ofstream& file) const {
    utils::BinaryWrite(file, &global_counter_, sizeof(global_counter_));
    sketch_.Store(file);
  }

  void Reset() noexcept {
    sketch_.Reset();
    global_counter_ = 0;
  }

  void Clear() noexcept {
    sketch_.Clear();
    global_counter_ = 0;
  }

 private:
  CountMinSketch<NumCounters> sketch_;
  TGlobalCounter global_counter_{0};
};

template <class T, size_t SampleSize, size_t NumCounters>
class TinyLFU<T, SampleSize, NumCounters, true> final {
 public:
  using TGlobalCounter = uint32_t;
  static_assert(SampleSize <= std::numeric_limits<TGlobalCounter>::max());

  void Add(T key) noexcept {
    if (key == std::numeric_limits<T>::max()) return;

    auto was_added = door_keeper_.Add(key);
    if (was_added) {
      sketch_.Add(key);
    }

    if constexpr (SampleSize > 0) {
      if (++global_counter_ >= SampleSize) Reset();
    }
  }

  size_t Estimate(T key) const noexcept {
    if (key == std::numeric_limits<T>::max()) return 0;

    auto frequency = sketch_.Estimate(key);
    if (door_keeper_.Test(key)) frequency++;
    return frequency;
  }

  void Load(std::ifstream& file) {
    utils::BinaryRead(file, &global_counter_, sizeof(global_counter_));
    door_keeper_.Load(file);
    sketch_.Load(file);
  }

  void Store(std::ofstream& file) const {
    utils::BinaryWrite(file, &global_counter_, sizeof(global_counter_));
    door_keeper_.Store(file);
    sketch_.Store(file);
  }

  void Reset() noexcept {
    sketch_.Reset();
    door_keeper_.Clear();
    global_counter_ = 0;
  }

  void Clear() noexcept {
    sketch_.Clear();
    door_keeper_.Clear();
    global_counter_ = 0;
  }

  bool operator==(const TinyLFU& other) const noexcept {
    return sketch_ == other.sketch_ && door_keeper_ == other.door_keeper_ &&
           global_counter_ == other.global_counter_;
  }

 private:
  CountMinSketch<NumCounters> sketch_;
  BloomFilter<SampleSize> door_keeper_;
  TGlobalCounter global_counter_{0};
};

}  // namespace cache
