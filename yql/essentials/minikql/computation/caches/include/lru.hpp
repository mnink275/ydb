#pragma once

#include <chrono>
#include <random>

#include "cache_config.hpp"

#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>
#include <boost/intrusive/unordered_set.hpp>
#include <boost/intrusive/unordered_set_hook.hpp>

#include <optional>

// Based on:
// https://github.com/apolukhin/apolukhin.github.io/blob/master/presentations/C%2B%2B%20Faster.cpp

namespace cache {

namespace details {

using LinkMode = boost::intrusive::link_mode<
#ifdef NDEBUG
    boost::intrusive::normal_link
#else
    boost::intrusive::safe_link
#endif
    >;

using ListBaseHook = boost::intrusive::list_base_hook<LinkMode>;
using UnorderedSetBaseHook =
    boost::intrusive::unordered_set_base_hook<LinkMode>;

template <class Key>
struct Node final : public ListBaseHook, public UnorderedSetBaseHook {
  explicit Node(Key key, UV&& value, uint32_t expiration_time)
      : key(std::move(key)),
        value(std::move(value)),
        expiration_time(expiration_time) {}

  Key key;
  UV value;
  uint32_t expiration_time;
};

template <class SomeKey>
const SomeKey& GetKey(const Node<SomeKey>& node) noexcept {
  return node.key;
}

template <class T>
const T& GetKey(const T& key) noexcept {
  return key;
}

}  // namespace details

template <class Key, class Hash = std::hash<Key>,
          class Equal = std::equal_to<Key>>
class LRU final {
 public:
  explicit LRU(size_t max_size)
      : buckets_(max_size ? max_size : 1),
        map_(BucketTraits(buckets_.data(), buckets_.size())) {}

  LRU(LRU&& lru) = delete;
  LRU(const LRU& lru) = delete;

  LRU& operator=(LRU&& lru) = delete;
  LRU& operator=(const LRU& lru) = delete;

  ~LRU() {
    while (!list_.empty()) {
      ExtractNode(list_.begin());
    }
  }

  std::optional<details::Node<Key>> Update(Key key, UV&& value, uint32_t expiration_time) {
    std::optional<details::Node<Key>> evicted;
    if (map_.size() == buckets_.size()) {
      auto node = ExtractNode(list_.begin());
      evicted = *node;
      *node = LruNode{key, std::move(value), expiration_time};
      InsertNode(std::move(node));
    } else {
      auto node = std::make_unique<LruNode>(key, std::move(value), expiration_time);
      InsertNode(std::move(node));
    }

    return evicted;
  }

  UV Get(Key key, uint32_t now) {
    auto it = map_.find(key, map_.hash_function(), map_.key_eq());
    if (it == map_.end()) return EMPTY_UV;

    bool should_evict = false;
    if constexpr (TTL_EVICTION_PROB > 0.0) {
      static std::mt19937 gen(BERNOULLI_SEED ? BERNOULLI_SEED
                                             : std::random_device{}());
      static std::bernoulli_distribution dist(TTL_EVICTION_PROB);
      should_evict = dist(gen);
    } else {
      should_evict = it->expiration_time < now;
    }

    if (should_evict) {
      ExtractNode(list_.iterator_to(*it));
      return EMPTY_UV;
    }

    list_.splice(list_.end(), list_, list_.iterator_to(*it));
    return list_.back().value;
  }

 private:
  using LruNode = details::Node<Key>;
  using List =
      boost::intrusive::list<LruNode,
                             boost::intrusive::constant_time_size<false>>;

  std::unique_ptr<LruNode> ExtractNode(typename List::iterator it) noexcept {
    std::unique_ptr<LruNode> ret(&*it);
    map_.erase(map_.iterator_to(*it));
    list_.erase(it);
    return ret;
  }

  void InsertNode(std::unique_ptr<LruNode>&& node) noexcept {
    if (!node) return;

    map_.insert(*node);
    list_.insert(list_.end(), *node);

    node.release();
  }

  struct LruNodeHash : Hash {
    template <class NodeOrKey>
    auto operator()(const NodeOrKey& x) const {
      return Hash::operator()(details::GetKey(x));
    }
  };

  struct LruNodeEqual : Equal {
    template <class NodeOrKey1, class NodeOrKey2>
    auto operator()(const NodeOrKey1& x, const NodeOrKey2& y) const {
      return Equal::operator()(details::GetKey(x), details::GetKey(y));
    }
  };

  using Map = boost::intrusive::unordered_set<
      LruNode, boost::intrusive::constant_time_size<true>,
      boost::intrusive::hash<LruNodeHash>,
      boost::intrusive::equal<LruNodeEqual>>;
  using BucketTraits = typename Map::bucket_traits;
  using BucketType = typename Map::bucket_type;

 private:
  std::vector<BucketType> buckets_;
  Map map_;
  List list_;
};

}  // namespace cache
