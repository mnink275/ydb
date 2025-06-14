#pragma once

#include <cassert>
#include <filesystem>

#include <yql/essentials/public/udf/udf_value.h>
#include <yql/essentials/minikql/computation/mkql_computation_node_pack.h>
#include <ydb/library/actors/core/actorsystem.h>
#include <ydb/library/actors/core/executor_pool_basic.h>
#include <ydb/library/actors/core/scheduler_basic.h>
#include <ydb/library/actors/core/log.h>
#include <ydb/library/actors/core/actor_bootstrapped.h>
#include <ydb/library/actors/util/should_continue.h>
#include <util/system/sigset.h>
#include <util/generic/xrange.h>
#include <util/system/guard.h>

#include "../events.hpp"
#include "../include/cache_config.hpp"
#include "../include/large_page_provider.hpp"
#include "../include/lru.hpp"

namespace ink::actor {

using namespace NActors;
using namespace std::chrono_literals;

class TPageBasedActor : public TActorBootstrapped<TPageBasedActor> {
public:
  TPageBasedActor(NKikimr::NMiniKQL::TScopedAlloc& alloc,
                  const NKikimr::NMiniKQL::TTypeBuilder& typeBuilder,
                  const NKikimr::NMiniKQL::THolderFactory& holderFactory,
                  std::filesystem::path dir_path = "./data")
    : Alloc(alloc)
    , TinyLFU()
    , Provider(std::move(dir_path))
    , TypeBuilder(typeBuilder)
    , HolderFactory(holderFactory)
#if USE_LRU_FLAG
    , lru_(static_cast<size_t>(cache::LRU_SIZE))
#endif
  {}

  STFUNC(StateWait) {
    switch (ev->GetTypeRewrite()) {
      hFunc(TEvCache::TEvGet, Handle);
      hFunc(TEvCache::TEvUpdate, Handle);
      hFunc(TEvCache::TEvTerminate, Handle);
    }

    ++HandledEvents;
  }

  void Bootstrap() {
    Become(&TThis::StateWait);

    auto toBeRegistredLargePageActors = Provider.InitLargePageActors(Alloc, TinyLFU, TypeBuilder, HolderFactory);
    std::array<NActors::TActorId, cache::LOADED_PAGE_NUMBER> largePageActorIds;
    Y_ASSERT(toBeRegistredLargePageActors.size() == cache::LOADED_PAGE_NUMBER);
    for (size_t i = 0; i < toBeRegistredLargePageActors.size(); ++i) {
      largePageActorIds[i] = Register(toBeRegistredLargePageActors[i]);
    }

    Provider.InitRegistredActorIds(std::move(largePageActorIds));
  }

 private:
  void Handle(TEvCache::TEvGet::TPtr& ev) {
    const auto result = Provider.Get<false>(ev->Get()->Key);
    if (!result) {
      Send(ev->Sender, new TEvCache::TEvLargePageNotLoaded{ev->Get()->Key});
      return;
    }

    auto [largePageActorId, swapEventPtr] = *result;
    if (swapEventPtr) {
      Send(largePageActorId, swapEventPtr);
    }

    Send(ev->Forward(largePageActorId));
  }

  void Handle(TEvCache::TEvUpdate::TPtr& ev) {
    const auto result = Provider.Get<true>(ev->Get()->Key);
    if (!result) return;

    auto [largePageActorId, swapEventPtr] = *result;
    if (swapEventPtr) {
      Send(largePageActorId, swapEventPtr);
    }

    Send(ev->Forward(largePageActorId));
  }

  void Handle(TEvCache::TEvTerminate::TPtr& ev) {
    const static auto BenchmarkActorId = ev->Get()->SenderId;

    for (auto largePageActorId : Provider.GetLargePageActorIds()) {
      Send(largePageActorId, new TEvCache::TEvTerminate{BenchmarkActorId});
    }
  }

 private:
  NKikimr::NMiniKQL::TScopedAlloc& Alloc;

  cache::TTinyLFU TinyLFU{};
  cache::LargePageProvider Provider;

  const NKikimr::NMiniKQL::TTypeBuilder& TypeBuilder;
  const NKikimr::NMiniKQL::THolderFactory& HolderFactory;
#if USE_LRU_FLAG
  cache::LRU<cache::Key> lru_;
#endif
};

}  // namespace ink::actor
