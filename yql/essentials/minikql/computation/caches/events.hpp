#pragma once

#include <ydb/library/actors/core/events.h>
#include <ydb/library/actors/core/event_local.h>

#include "include/cache_config.hpp"

namespace ink::actor {

using namespace NActors;

struct TEvCache {
    enum EEv {
        EvGet = EventSpaceBegin(TEvents::ES_USERSPACE + 1),
        EvGetResult,
        EvLargePageNotLoaded,
        EvUpdate,
        EvStorePage,
        EvLoadPage,
        EvSwapPage,
        EvTerminate
    };

    struct TEvGet : public TEventLocal<TEvGet, EvGet> {
        const ui32 Key;
        const ui32 Now;

        TEvGet(ui32 key, ui32 now)
            : Key(key)
            , Now(now)
        {}
    };

    struct TEvGetResult : public TEventLocal<TEvGetResult, EvGetResult> {
        const ui32 Key;
        cache::UV Payload;

        TEvGetResult(ui32 key, cache::UV&& payload)
            : Key(key)
            , Payload(std::move(payload)) {}
    };

    struct TEvLargePageNotLoaded : public TEventLocal<TEvLargePageNotLoaded, EvLargePageNotLoaded> {
        const ui32 Key;

        TEvLargePageNotLoaded(ui32 key) : Key(key) {}
    };

    struct TEvUpdate : public TEventLocal<TEvUpdate, EvUpdate> {
        const ui32 Key;
        cache::UV Payload;
        const ui32 Expiration;

        TEvUpdate(ui32 key, cache::UV&& payload, ui32 expiration)
            : Key(key)
            , Payload(std::move(payload))
            , Expiration(expiration)
        {}
    };

    struct TEvSwap : public TEventLocal<TEvSwap, EvSwapPage> {
        std::filesystem::path StorePath;
        std::filesystem::path LoadPath;

        TEvSwap(std::filesystem::path&& storePath, std::filesystem::path&& loadPath)
            : StorePath(storePath)
            , LoadPath(loadPath) {}
    };

    struct TEvTerminate : public TEventLocal<TEvTerminate, EvTerminate> {
        NActors::TActorId SenderId;

        TEvTerminate(NActors::TActorId senderId) : SenderId(senderId) {}
    };
};

}  // namespace ink::actor
