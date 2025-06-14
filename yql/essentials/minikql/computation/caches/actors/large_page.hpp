#pragma once

#include <ydb/library/actors/core/log.h>
#include <ydb/library/actors/core/actorsystem.h>
#include <ydb/library/actors/core/actor_bootstrapped.h>
#include <ydb/library/actors/util/should_continue.h>
#include <yql/essentials/public/udf/udf_value.h>
#include <yql/essentials/minikql/computation/mkql_computation_node_pack.h>
#include <util/system/guard.h>

#include "../events.hpp"
#include "../include/large_page.hpp"

namespace ink::actor {

using namespace NActors;

class TLargePageActor : public TActorBootstrapped<TLargePageActor> {
 public:
  TLargePageActor(NKikimr::NMiniKQL::TScopedAlloc& alloc,
                  cache::TTinyLFU& tinyLfu,
                  const NKikimr::NMiniKQL::TTypeBuilder& typeBuilder,
                  const NKikimr::NMiniKQL::THolderFactory& holderFactory,
                  std::filesystem::path&& currPath)
    : Alloc(alloc)
    , LargePage(tinyLfu, typeBuilder, holderFactory) {
      Load(std::move(currPath));
    }

  STFUNC(StateWait) {
    switch (ev->GetTypeRewrite()) {
      hFunc(TEvCache::TEvGet, Handle);
      hFunc(TEvCache::TEvUpdate, Handle);
      hFunc(TEvCache::TEvSwap, Handle);
      hFunc(TEvCache::TEvTerminate, Handle);
    }

    ++HandledEvents;
  }

  void Bootstrap() {
    Become(&TThis::StateWait);
  }

 private:
  void Handle(TEvCache::TEvGet::TPtr& ev) {
    HandledGetRequestsCount++;

    auto& data = *ev->Get();
    auto value = LargePage.Get(data.Key, data.Now);
    Send(ev->Sender, new TEvCache::TEvGetResult{data.Key, std::move(value)});
  }

  void Handle(TEvCache::TEvUpdate::TPtr& ev) {
    auto& data = *ev->Get();
    Y_ASSERT(data.Payload.HasValue());
    LargePage.Update(data.Key, std::move(data.Payload), data.Expiration);
  }

  void Handle(TEvCache::TEvSwap::TPtr& ev) {
    Store(std::move(ev->Get()->StorePath));
    Load(std::move(ev->Get()->LoadPath));
  }

  void Handle(TEvCache::TEvTerminate::TPtr& ev) {
    Send(ev->Get()->SenderId, new TEvCache::TEvTerminate{SelfId()});
  }

 private:
  void Store(std::filesystem::path&& filePath) {
    auto guard = BindAllocator();

    utils::OFileStream file{filePath};
    LargePage.Store(file);
  }

  void Load(std::filesystem::path&& filePath) {
    auto guard = BindAllocator();

    if (std::filesystem::exists(filePath)) {
      std::ifstream file(filePath, std::ios_base::binary);
      // TODO: убрать копирование, при большом размере stack-overflow
      LargePage.Load(file, std::filesystem::file_size(filePath));
    } else {
      LargePage.Clear();
    }
  }

  TGuard<NKikimr::NMiniKQL::TScopedAlloc> BindAllocator() {
    return Guard(Alloc);
  }

 private:
  NKikimr::NMiniKQL::TScopedAlloc& Alloc;
  cache::LargePage LargePage;

  size_t HandledGetRequestsCount{0};
};

}  // namespace ink::actor
