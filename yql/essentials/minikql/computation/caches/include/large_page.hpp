#pragma once

#include <array>

#include <yql/essentials/minikql/computation/mkql_computation_node_pack.h>

#include "cache_config.hpp"
#include "small_page.hpp"
#include "utils.hpp"

namespace cache {

inline size_t LargePageIndex(Key key) noexcept {
  return key >> (8ull * sizeof(Key) - LARGE_PAGE_SHIFT);
}

class LargePage {
 public:
  explicit LargePage(TTinyLFU& tiny_lfu,
                     const NKikimr::NMiniKQL::TTypeBuilder& type_builder,
                     const NKikimr::NMiniKQL::THolderFactory& holder_factory)
      : small_pages_(utils::MakeArray<SMALL_PAGE_NUMBER>(SmallPage{tiny_lfu})),
        type_builder_(type_builder),
        holder_factory_(holder_factory) {
  }

  void Clear() noexcept {
    for (auto& page : small_pages_) {
      page.Clear();
    }
  }

#if 1
  void Load(std::ifstream& file, std::uintmax_t file_size) {
    assert(file_size >= kTrivialDataSizeInBytes);

    // non-UV data (keys + TTL + last_free_slot_)
    static std::array<char, kTrivialDataSizeInBytes> data{};
    file.read(data.data(), data.size());

    for (size_t i = 0; i < SMALL_PAGE_NUMBER; ++i) {
      small_pages_[i].Load(data.data() + i * SmallPage::kTrivialDataSizeInBytes);
    }

    // UV-data (payload)
    static auto buff = std::string(file_size - kTrivialDataSizeInBytes, '#');
    file.read(buff.data(), buff.size());

    NYql::TChunkedBuffer chuncked_buff{TString{buff}};

    static auto ui32_type = type_builder_.NewDataType(NYql::NUdf::EDataSlot::Int32);
    static auto list_type = type_builder_.NewListType(ui32_type);
    // std::cout << "Load()" << std::endl;
    static NKikimr::NMiniKQL::TValuePackerTransport<FastPack> unpacker{list_type};
    auto list = unpacker.Unpack(std::move(chuncked_buff), holder_factory_);
    const auto iter = list.GetListIterator();
    for (size_t i = 0; i < SMALL_PAGE_NUMBER; ++i) {
      small_pages_[i].LoadUV(iter);
    }
  }
#else
  void Load(std::ifstream& file, std::uintmax_t file_size) {
    assert(file_size >= kTrivialDataSizeInBytes);

    // non-UV data (keys + TTL + last_free_slot_)
    static std::array<char, kTrivialDataSizeInBytes> data{};
    file.read(data.data(), data.size());

    for (size_t i = 0; i < SMALL_PAGE_NUMBER; ++i) {
      small_pages_[i].Load(data.data() + i * SmallPage::kTrivialDataSizeInBytes);
    }

    // UV-data (payload)
    static auto buff = std::string(file_size - kTrivialDataSizeInBytes, '#');
    file.read(buff.data(), buff.size());

    NYql::TChunkedBuffer chuncked_buff{TString{buff}};

    static auto ui32_type = type_builder_.NewDataType(NYql::NUdf::EDataSlot::Int32);

    static NKikimr::NMiniKQL::TValuePackerTransport<FastPack> unpacker{ui32_type};

    NKikimr::NMiniKQL::TUnboxedValueBatch items;
    unpacker.UnpackBatch(std::move(chuncked_buff), holder_factory_, items);

    size_t small_page_idx = 0;
    size_t value_idx = 0;
    items.ForEachRow([&](UV& value) {
      small_pages_[small_page_idx].LoadUV(value, value_idx++);
      if (value_idx >= SMALL_PAGE_SIZE) {
        value_idx = 0;
        small_page_idx++;
      }
    });
  }
#endif

  void Store(utils::OFileStream& file) const {
    // non-UV data (keys + TTL + last_free_slot_)
    static std::array<char, kTrivialDataSizeInBytes> buff{};
    for (size_t i = 0; i < SMALL_PAGE_NUMBER; ++i) {
      small_pages_[i].Store(buff.data() + i * SmallPage::kTrivialDataSizeInBytes);
    }
    file.Write(buff.data(), kTrivialDataSizeInBytes);

    // UV-data (payload.value)
    NKikimr::NMiniKQL::TValuePackerTransport<FastPack> packer{type_builder_.NewDataType(NYql::NUdf::EDataSlot::Int32)};
    for (size_t i = 0; i < SMALL_PAGE_NUMBER; ++i) {
      small_pages_[i].StoreUV(packer);
    }

    NYql::TChunkedBuffer result = packer.Finish();
    result.CopyTo(file);
  }

  UV Get(Key key, uint32_t now) noexcept {
    return small_pages_[SmallPageIndex(key)].Get(key, now);
  }

  void Update(Key key, UV&& value, uint32_t expiration_time) noexcept {
    small_pages_[SmallPageIndex(key)].Update(key, std::move(value), expiration_time);
  }

#if ENABLE_STATISTICS_FLAG
  std::vector<double> GetSmallPagesFillFactors() {
    std::vector<double> res;
    res.reserve(small_pages_.size());
    for (const auto& page : small_pages_) {
      res.push_back(page.GetFillFactor());
    }
    return res;
  }

  uint64_t GetNumEvictionsHighFreq() const {
    uint64_t res = 0;
    for (const auto& page : small_pages_) {
      res += page.GetNumEvictionsHighFreq();
    }
    return res;
  }

  uint64_t GetNumDroppedKeysLowFreq() const {
    uint64_t res = 0;
    for (const auto& page : small_pages_) {
      res += page.GetNumDroppedKeysLowFreq();
    }
    return res;
  }
#endif

  bool operator==(const LargePage& other) const noexcept {
    for (size_t i = 0; i < SMALL_PAGE_NUMBER; ++i) {
      if (small_pages_[i] != other.small_pages_[i]) return false;
    }
    return true;
  }

 private:
  static constexpr std::size_t kTrivialDataSizeInBytes =
      SMALL_PAGE_NUMBER * SmallPage::kTrivialDataSizeInBytes;
  // static_assert(sizeof(SmallPage) == 1);

  std::array<SmallPage, SMALL_PAGE_NUMBER> small_pages_;
  const NKikimr::NMiniKQL::TTypeBuilder& type_builder_;
  const NKikimr::NMiniKQL::THolderFactory& holder_factory_;
};

}  // namespace cache
