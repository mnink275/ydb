#pragma once

#include <algorithm>
#include <filesystem>

#if ENABLE_STATISTICS_FLAG
#include <iostream>
#include <iomanip>
#endif

#include "cache_config.hpp"
#include "large_page.hpp"

namespace cache {

class LargePageProvider {
 public:
  LargePageProvider(std::filesystem::path dir_path,
                    TTinyLFU& tiny_lfu,
                    const NKikimr::NMiniKQL::TTypeBuilder& type_builder,
                    const NKikimr::NMiniKQL::THolderFactory& holder_factory)
      : dir_path_(std::move(dir_path)),
        storage_(std::make_unique<Storage>(tiny_lfu, type_builder, holder_factory)) {
    static_assert(LOADED_PAGE_NUMBER <= LARGE_PAGE_NUMBER);
    static_assert(LARGE_PAGE_SHIFT + SMALL_PAGE_SHIFT + SMALL_PAGE_SIZE_SHIFT <=
                  8 * sizeof(Key));
    if (!std::filesystem::exists(dir_path_))
      std::filesystem::create_directory(dir_path_);

    size_t storage_index = 0;

    for (auto [_, page_index] : LoadHeader()) {
      // TODO: сделать ленивую загрузку
      page_infos_[page_index].storage_index = storage_index;
      loaded_frequencies_[storage_index] =
          std::make_pair(page_infos_[page_index].frequency, page_index);
      LoadPage(storage_index);
      ++storage_index;
    }
  }

  template <bool CalledOnUpdate>
  LargePage* Get(Key key) {
    if (time_ == LARGE_PAGE_PERIOD) {
      DivFrequency();
      time_ = 0;
    }

    ++time_;

    const size_t page_index = LargePageIndex(key);
    page_infos_[page_index].frequency += 1;
    if (auto page_ptr = GetLoadedPage(page_index); page_ptr) {
      loaded_frequencies_[page_infos_[page_index].storage_index].first += 1;
      return page_ptr;
    }

    if (worst_frequency_estimation_ + FREQUENCY_THRESHOLD <
        page_infos_[page_index].frequency) {
      // update estimation

      worst_frequency_estimation_ = std::numeric_limits<size_t>::max();
      size_t storage_index = kNPos;
      for (size_t index = 0; index < LOADED_PAGE_NUMBER; ++index) {
        if (loaded_frequencies_[index].first < worst_frequency_estimation_) {
          worst_frequency_estimation_ = loaded_frequencies_[index].first;
          storage_index = index;
          // Можно не дублировать частоту, смотреть её по индексу
          // (loaded_frequencies_[storage_index].second)
        }
      }
      assert(worst_frequency_estimation_ != std::numeric_limits<size_t>::max());
      const size_t worst_page = loaded_frequencies_[storage_index].second;
      assert(page_infos_[worst_page].storage_index == storage_index);

      if (worst_frequency_estimation_ + FREQUENCY_THRESHOLD <
          page_infos_[page_index].frequency) {
        StorePage(storage_index);

        page_infos_[page_index].storage_index = storage_index;
        page_infos_[worst_page].storage_index = kNPos;

        loaded_frequencies_[storage_index] =
            std::make_pair(page_infos_[page_index].frequency, page_index);

        LoadPage(storage_index);

        // TODO update worst_loaded_page ?

        return &(storage_->large_pages[storage_index]);
      }
    }
#if ENABLE_STATISTICS_FLAG
    if (CalledOnUpdate) dropped_keys_++;
#endif

    return nullptr;
  }

  void Store() const {
    StoreHeader();
    for (size_t i = 0; i < LOADED_PAGE_NUMBER; ++i) {
      StorePage(i);
    }
  }

#if ENABLE_STATISTICS_FLAG
  size_t large_page_loads_{0};
  uint64_t dropped_keys_{0};
#endif

  ~LargePageProvider() {
    if constexpr (ENABLE_STATISTICS_FLAG) PrintStatistics();
  }

 private:
  struct LargePageInfo {
    size_t frequency{0};  // всегда < 2*period, можно оптимизировать размер
    size_t storage_index{kNPos};
  };

  struct Storage {
    explicit Storage(TTinyLFU& tiny_lfu,
                     const NKikimr::NMiniKQL::TTypeBuilder& type_builder,
                     const NKikimr::NMiniKQL::THolderFactory& holder_factory)
        : large_pages(
              utils::MakeArray<LOADED_PAGE_NUMBER>(LargePage{tiny_lfu, type_builder, holder_factory})) {}

    std::array<LargePage, LOADED_PAGE_NUMBER> large_pages;
  };

  std::filesystem::path GetFilePath(size_t i) const {
    return dir_path_ /
           std::filesystem::path("page" + std::to_string(i) + ".bin");
  }

  std::filesystem::path GetHeaderPath() const {
    return dir_path_ / std::filesystem::path("header.bin");
  }

  [[nodiscard]] std::vector<std::pair<size_t, size_t>> LoadHeader() {
    const std::filesystem::path file_path = GetHeaderPath();

    std::vector<std::pair<size_t, size_t>> best_pages;
    best_pages.reserve(LARGE_PAGE_NUMBER);

    if (std::filesystem::exists(file_path)) {
      std::ifstream file(file_path, std::ios_base::binary);
      for (size_t i = 0; i < page_infos_.size(); ++i) {
        utils::BinaryRead(file, &page_infos_[i].frequency,
                          sizeof(page_infos_[i].frequency));
        best_pages.emplace_back(page_infos_[i].frequency, i);
      }
      std::sort(best_pages.begin(), best_pages.end(),
                [](const auto& lhs, const auto& rhs) {
                  return lhs.first > rhs.first;
                });
      best_pages.resize(LOADED_PAGE_NUMBER);
    } else {
      while (best_pages.size() < LOADED_PAGE_NUMBER) {
        best_pages.emplace_back(0, best_pages.size());
      }
    }

    worst_frequency_estimation_ = best_pages.back().first;
    assert(best_pages.size() == LOADED_PAGE_NUMBER);
    return best_pages;
  }

  void StoreHeader() const {
    std::ofstream file(GetHeaderPath(),
                       std::ios_base::binary | std::ios_base::trunc);
    for (const auto& page : page_infos_) {
      utils::BinaryWrite(file, &page.frequency, sizeof(page.frequency));
    }
  }

  LargePage* GetLoadedPage(size_t page_index) {
    const size_t storage_index = page_infos_[page_index].storage_index;
    return storage_index != kNPos ? &(storage_->large_pages[storage_index])
                                 : nullptr;
  }

  void LoadPage(size_t storage_index) {
    assert(storage_index != kNPos);
#if ENABLE_STATISTICS_FLAG
    large_page_loads_++;
#endif
    const std::filesystem::path file_path =
        GetFilePath(loaded_frequencies_[storage_index].second);
    if (std::filesystem::exists(file_path)) {
      std::ifstream file(file_path, std::ios_base::binary);
      storage_->large_pages[storage_index].Load(
          file, std::filesystem::file_size(file_path));  // TODO: убрать копирование, при большом размере
                  // stack-overflow
    } else {
      storage_->large_pages[storage_index].Clear();
    }
  }

  void StorePage(size_t storage_index) const {
    assert(storage_index != kNPos);

    utils::OFileStream file(GetFilePath(loaded_frequencies_[storage_index].second));

    storage_->large_pages[storage_index].Store(file);
  }

  void DivFrequency() {  // делит все частоты на 2
    for (size_t i = 0; i < page_infos_.size(); ++i) {
      page_infos_[i].frequency >>= 1;
    }
    worst_frequency_estimation_ >>= 1;
    for (size_t i = 0; i < LOADED_PAGE_NUMBER; ++i) {
      loaded_frequencies_[i].first >>= 1;
    }
  }

  void PrintStatistics() const {
#if ENABLE_STATISTICS_FLAG
    std::cout << '\n';
    std::cout << "Кол-во свопов больших страниц (RAM <-> диск): "
              << large_page_loads_ << std::endl;
    std::cout << "Кол-во отброшенных ключей при Update (если соотв. LargePage "
                 "не загружена в RAM): "
              << dropped_keys_ << std::endl;

    uint64_t evictions_high_freq = 0;
    for (size_t i = 0; i < LOADED_PAGE_NUMBER; ++i) {
      evictions_high_freq += storage_->large_pages[i].GetNumEvictionsHighFreq();
    }
    std::cout << "Кол-во вытесненных ключей из страницы (при переполнении): "
              << evictions_high_freq << std::endl;

    uint64_t dropped_keys_low_freq = 0;
    for (size_t i = 0; i < LOADED_PAGE_NUMBER; ++i) {
      dropped_keys_low_freq +=
          storage_->large_pages[i].GetNumDroppedKeysLowFreq();
    }
    std::cout << "Кол-во отброшенных ключей при Update (из-за низкой частоты): "
              << dropped_keys_low_freq << std::endl;

    std::vector<double> fill_factors;
    const size_t SMALL_PAGE_NUM_OVERALL =
        LOADED_PAGE_NUMBER * SMALL_PAGE_NUMBER;
    fill_factors.reserve(SMALL_PAGE_NUM_OVERALL);
    for (size_t i = 0; i < LOADED_PAGE_NUMBER; ++i) {
      auto factors = storage_->large_pages[i].GetSmallPagesFillFactors();
      fill_factors.insert(fill_factors.end(), factors.begin(), factors.end());
    }

    const double kSample = 0.1;
    std::vector<size_t> freqs(10, 0);
    for (auto freq : fill_factors) {
      const auto sample_idx =
          std::min(9ul, static_cast<size_t>(freq / kSample));
      freqs[sample_idx]++;
    }
    std::cout << "Распределение страниц по лоад-фактору после бенчмарка:"
              << std::endl;
    for (size_t idx = freqs.size(); idx-- > 0;) {
      if (freqs[idx] == 0) continue;
      std::cout << std::fixed << std::setprecision(1) << "[" << idx * kSample
                << "-" << (idx + 1) * kSample << "): ";
      std::cout << std::fixed << std::setprecision(2)
                << 100.0 * freqs[idx] / SMALL_PAGE_NUM_OVERALL << "%"
                << std::endl;
    }
#endif
  }

  static constexpr size_t kNPos = std::numeric_limits<size_t>::max();

  const std::string dir_path_;
  std::unique_ptr<Storage> storage_;
  std::array<LargePageInfo, LARGE_PAGE_NUMBER> page_infos_;
  size_t worst_frequency_estimation_;  // частота загруженных страниц не меньше
                                       // этой оценки
  std::array<std::pair<size_t, size_t>, LOADED_PAGE_NUMBER>
      loaded_frequencies_;  // <частота, индекс page_infos_> в дубликат частот
                            // страниц для быстрого обновления
                            // worst_frequency_estimation_
  size_t time_{0};
};

}  // namespace cache
