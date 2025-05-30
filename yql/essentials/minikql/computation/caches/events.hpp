#pragma once

#include <ydb/library/actors/core/events.h>
#include <ydb/library/actors/core/event_local.h>

#include "include/cache_config.hpp"

namespace ink::actor {

using namespace NActors;

struct TEvCache {
    enum EEv {
        EvGet = EventSpaceBegin(TEvents::ES_USERSPACE + 1),
        EvUpdate,
        EvTerminate
    };

    struct TEvGet : public TEventLocal<TEvGet, EvGet> {
        const ui32 Key;
        const ui32 Now;
        cache::UV Value;

        TEvGet(ui32 key, cache::UV&& value, ui32 now)
            : Key(key)
            , Now(now)
            , Value(value)
        {}
    };

    struct TEvUpdate : public TEventLocal<TEvUpdate, EvUpdate> {
        const ui32 Key;
        cache::UV Value;
        const ui32 Expiration;

        TEvUpdate(ui32 key, cache::UV&& value, ui32 expiration)
            : Key(key)
            , Value(value)
            , Expiration(expiration)
        {}
    };

    struct TEvTerminate : public TEventLocal<TEvTerminate, EvTerminate> {
        size_t KeyCount;

        TEvTerminate(size_t keyCount) : KeyCount(keyCount) {}
    };
};

}  // namespace ink::actor
