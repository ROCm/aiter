// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../opus_moe_backward_common.cuh"
#include "bf16/opus_moe_down_bwd_pipeline_gfx950.cuh"
#include "bf16/opus_moe_bias_bwd_pipeline_gfx950.cuh"
#include "bf16/opus_moe_route_dx_pipeline_gfx950.cuh"
#include "bf16/opus_moe_route_reduce_pipeline_gfx950.cuh"
#include "bf16/opus_moe_router_bwd_pipeline_gfx950.cuh"
#include "bf16/opus_moe_weight_bwd_pipeline_gfx950.cuh"
#include "opus_moe_backward_manifest.h"

#include "aiter_hip_common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <hip/hip_runtime.h>

namespace opus_moe_backward::gfx950
{
namespace detail
{

template<typename Kargs>
using Launcher = void (*)(const Kargs&, hipStream_t);

template<typename Kargs>
struct Entry
{
    int kid;
    const char* name;
    Launcher<Kargs> launcher;
};

template<typename Kargs, std::size_t N>
inline Launcher<Kargs> lookup(const std::array<Entry<Kargs>, N>& entries,
                              int requested_kid,
                              int auto_kid,
                              Family family)
{
    AITER_CHECK(!entries.empty(),
                "opus_moe_backward: gfx950 BF16 family '",
                family_name(family),
                "' has no registered kernel instances");
    if(entries.empty())
        return nullptr;

    const int kid = requested_kid == kKernelAuto ? auto_kid : requested_kid;
    const auto it = std::lower_bound(
        entries.begin(), entries.end(), kid, [](const Entry<Kargs>& entry, int value) {
            return entry.kid < value;
        });
    AITER_CHECK(it != entries.end() && it->kid == kid,
                "opus_moe_backward: kernel id ",
                kid,
                " is not registered for gfx950 BF16 family '",
                family_name(family),
                "'");
    return it == entries.end() || it->kid != kid ? nullptr : it->launcher;
}

static constexpr std::array<Entry<DownBwdKargs>,
                            OPUS_MOE_BACKWARD_DOWN_BWD_MANIFEST_SIZE>
    kDownBwd = {{OPUS_MOE_BACKWARD_DOWN_BWD_MANIFEST_ENTRIES}};
static constexpr std::array<Entry<RouteDxKargs>,
                            OPUS_MOE_BACKWARD_ROUTE_DX_MANIFEST_SIZE>
    kRouteDx = {{OPUS_MOE_BACKWARD_ROUTE_DX_MANIFEST_ENTRIES}};
static constexpr std::array<Entry<RouteReduceKargs>,
                            OPUS_MOE_BACKWARD_ROUTE_REDUCE_MANIFEST_SIZE>
    kRouteReduce = {{OPUS_MOE_BACKWARD_ROUTE_REDUCE_MANIFEST_ENTRIES}};
static constexpr std::array<Entry<Dw1Kargs>, OPUS_MOE_BACKWARD_DW1_MANIFEST_SIZE>
    kDw1 = {{OPUS_MOE_BACKWARD_DW1_MANIFEST_ENTRIES}};
static constexpr std::array<Entry<Dw2Kargs>, OPUS_MOE_BACKWARD_DW2_MANIFEST_SIZE>
    kDw2 = {{OPUS_MOE_BACKWARD_DW2_MANIFEST_ENTRIES}};
static constexpr std::array<Entry<RouterBwdKargs>,
                            OPUS_MOE_BACKWARD_ROUTER_BWD_MANIFEST_SIZE>
    kRouterBwd = {{OPUS_MOE_BACKWARD_ROUTER_BWD_MANIFEST_ENTRIES}};
static constexpr std::array<Entry<BiasBwdKargs>,
                            OPUS_MOE_BACKWARD_BIAS_BWD_MANIFEST_SIZE>
    kBiasBwd = {{OPUS_MOE_BACKWARD_BIAS_BWD_MANIFEST_ENTRIES}};

static_assert(kDownBwd.size() == OPUS_MOE_BACKWARD_DOWN_BWD_MANIFEST_SIZE);
static_assert(kRouteDx.size() == OPUS_MOE_BACKWARD_ROUTE_DX_MANIFEST_SIZE);
static_assert(kRouteReduce.size() == OPUS_MOE_BACKWARD_ROUTE_REDUCE_MANIFEST_SIZE);
static_assert(kDw1.size() == OPUS_MOE_BACKWARD_DW1_MANIFEST_SIZE);
static_assert(kDw2.size() == OPUS_MOE_BACKWARD_DW2_MANIFEST_SIZE);
static_assert(kRouterBwd.size() == OPUS_MOE_BACKWARD_ROUTER_BWD_MANIFEST_SIZE);
static_assert(kBiasBwd.size() == OPUS_MOE_BACKWARD_BIAS_BWD_MANIFEST_SIZE);

} // namespace detail

inline detail::Launcher<DownBwdKargs> dispatch_down_bwd(int kid)
{
    return detail::lookup(detail::kDownBwd, kid, 2, Family::DownBwd);
}

inline detail::Launcher<RouteDxKargs> dispatch_route_dx(int kid)
{
    return detail::lookup(detail::kRouteDx, kid, 5, Family::RouteDx);
}

inline detail::Launcher<RouteReduceKargs> dispatch_route_reduce(int kid)
{
    return detail::lookup(detail::kRouteReduce, kid, 0, Family::RouteReduce);
}

inline detail::Launcher<Dw1Kargs> dispatch_dw1(int kid)
{
    return detail::lookup(detail::kDw1, kid, 5, Family::Dw1);
}

inline detail::Launcher<Dw2Kargs> dispatch_dw2(int kid)
{
    return detail::lookup(detail::kDw2, kid, 3, Family::Dw2);
}

inline detail::Launcher<RouterBwdKargs> dispatch_router_bwd(int kid)
{
    return detail::lookup(detail::kRouterBwd, kid, 0, Family::RouterBwd);
}

inline detail::Launcher<BiasBwdKargs> dispatch_bias_bwd(int kid)
{
    return detail::lookup(detail::kBiasBwd, kid, 0, Family::BiasBwd);
}

} // namespace opus_moe_backward::gfx950
