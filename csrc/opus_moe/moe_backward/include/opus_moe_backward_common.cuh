// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <hip/hip_bfloat16.h>

namespace opus_moe_backward
{

constexpr int kKernelAuto = -1;
constexpr uint32_t kPackedTokenMask = 0x00ffffffu;
constexpr int kPackedTopkBits = 8;
constexpr int kMaxPackedTopk = 1 << kPackedTopkBits;

enum class Family : int
{
    DownBwd = 0,
    RouteDx = 1,
    RouteReduce = 2,
    Dw1 = 3,
    Dw2 = 4,
    RouterBwd = 5,
    BiasBwd = 6,
};

enum class RouteLayout : int
{
    SortedRouteMajor = 0,
    TokenSlotMajor = 1,
    CompactRouteMajor = 2,
};

constexpr const char* family_name(Family family) noexcept
{
    switch(family)
    {
    case Family::DownBwd: return "down_bwd";
    case Family::RouteDx: return "route_dx";
    case Family::RouteReduce: return "route_reduce";
    case Family::Dw1: return "dw1";
    case Family::Dw2: return "dw2";
    case Family::RouterBwd: return "router_bwd";
    case Family::BiasBwd: return "bias_bwd";
    }
    return "unknown";
}

static __host__ __device__ __forceinline__ int packed_token_id(int32_t packed)
{
    return static_cast<int>(static_cast<uint32_t>(packed) & kPackedTokenMask);
}

static __host__ __device__ __forceinline__ int packed_topk_slot(int32_t packed)
{
    return static_cast<int>(static_cast<uint32_t>(packed) >> 24);
}

static __host__ __device__ __forceinline__ int64_t logical_route_id(int token,
                                                                    int slot,
                                                                    int topk)
{
    return static_cast<int64_t>(token) * topk + slot;
}

struct DecodedRoute
{
    int token;
    int slot;
    int logical;
    bool valid;
};

// Shared sorting ABI.  Fixed and compact routing use the same physical layout,
// but the aliases below keep their meanings explicit at every call site.
// Only the pointers required by a kernel family need to be populated.
// num_valid_ids[0] is the padded active sorted-row count P.
struct RouteMetadata
{
    // SortedRouteMajor uses packed token/slot ids; CompactRouteMajor uses
    // compact logical route ids.  The compile-time RouteLayout trait prevents
    // the two encodings from being interpreted by the same kernel instance.
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    union
    {
        const int32_t* __restrict__ reverse_sorted;
        const int32_t* __restrict__ route_to_token;
    };
    union
    {
        // Padded expert-major sorted-row offsets for K2/K4/K5.
        const int32_t* __restrict__ expert_offsets;
        // Compact per-token logical-route offsets for K3 only.
        const int32_t* __restrict__ token_route_offsets;
    };

    int token_num;
    union
    {
        int topk;
        int route_count;
    };
    int num_experts;
    int sort_block_m;
    int sorted_capacity;
    int sorted_block_capacity;
};

template<RouteLayout Layout>
static __device__ __forceinline__ DecodedRoute
decode_sorted_route(const RouteMetadata& route, int32_t encoded, bool in_range)
{
    if constexpr(Layout == RouteLayout::CompactRouteMajor)
    {
        const int logical = encoded;
        const bool route_valid =
            in_range && logical >= 0 && logical < route.route_count;
        const int token = route_valid ? route.route_to_token[logical]
                                      : route.token_num;
        return DecodedRoute{token,
                            0,
                            logical,
                            route_valid && token >= 0 &&
                                token < route.token_num};
    }
    else
    {
        const int token = packed_token_id(encoded);
        const int slot = packed_topk_slot(encoded);
        const bool valid = in_range && token < route.token_num &&
                           slot < route.topk;
        return DecodedRoute{
            token,
            slot,
            static_cast<int>(logical_route_id(token, slot, route.topk)),
            valid};
    }
}

template<RouteLayout Layout>
static __device__ __forceinline__ int64_t
route_value_offset(const RouteMetadata& route,
                   const DecodedRoute& decoded,
                   int64_t fixed_token_stride)
{
    if constexpr(Layout == RouteLayout::CompactRouteMajor)
        return static_cast<int64_t>(decoded.logical);
    else
        return static_cast<int64_t>(decoded.token) * fixed_token_stride +
               decoded.slot;
}

template<RouteLayout Layout>
static __host__ __device__ __forceinline__ int
logical_route_count(const RouteMetadata& route)
{
    if constexpr(Layout == RouteLayout::CompactRouteMajor)
        return route.route_count;
    else
        return route.token_num * route.topk;
}

// K1: dO @ W2, dScore partial/reduce, SwiGLU backward, and score * activation.
// Saved route tensors are sorted-route-major in the first native contract.
struct DownBwdKargs
{
    const hip_bfloat16* __restrict__ d_out;   // [T, D]
    const hip_bfloat16* __restrict__ z;       // [P, 2I]
    const hip_bfloat16* __restrict__ w2;      // [E, D, I]
    const float* __restrict__ scores;         // [T, K]
    RouteMetadata route;

    hip_bfloat16* __restrict__ d_z;           // [P, 2I]
    hip_bfloat16* __restrict__ a_scaled;      // [P, I]
    float* __restrict__ d_scores;             // [T, K]
    float* __restrict__ d_scores_workspace;   // [T*K, ceil_div(I, block_n)]

    int model_dim;
    int inter_dim;
    int d_scores_parts;

    int64_t stride_do_t;
    int64_t stride_z_r;
    int64_t stride_w2_e;
    int64_t stride_w2_d;
    int64_t stride_score_t;
    int64_t stride_dz_r;
    int64_t stride_a_scaled_r;
    int64_t stride_ds_t;
    int64_t stride_ds_workspace_r;
};

// K2: expert-grouped varlen-M dZ @ W1.  Recover token/slot from
// sorted_token_ids and write d_x_route in logical [token,slot] order so K3 can
// consume adjacent top-k rows without an extra permutation or random gather.
struct RouteDxKargs
{
    const hip_bfloat16* __restrict__ d_z;      // [P, 2I]
    const hip_bfloat16* __restrict__ w1;       // [E, 2I, D]
    RouteMetadata route;
    hip_bfloat16* __restrict__ d_x_route;      // [P, D]

    int model_dim;
    int inter_dim;

    int64_t stride_dz_r;
    int64_t stride_w1_e;
    int64_t stride_w1_i;
    int64_t stride_dx_route_r;
};

// K3: reduce adjacent logical [token,slot] route rows in FP32.
struct RouteReduceKargs
{
    const hip_bfloat16* __restrict__ d_x_route; // [P, D]
    RouteMetadata route;
    hip_bfloat16* __restrict__ d_x;             // [T, D]

    int model_dim;
    int64_t stride_dx_route_r;
    int64_t stride_dx_t;
};

// K4: per-expert varlen-K dZ^T @ X.  FP32 workspace is required by split-K
// traits and may be null for a direct, single-CTA implementation.
struct Dw1Kargs
{
    const hip_bfloat16* __restrict__ x;         // [T, D]
    const hip_bfloat16* __restrict__ d_z;       // [P, 2I]
    RouteMetadata route;
    hip_bfloat16* __restrict__ d_w1;            // [E, 2I, D]
    float* __restrict__ workspace;

    int model_dim;
    int inter_dim;
    int split_k;

    int64_t stride_x_t;
    int64_t stride_dz_r;
    int64_t stride_dw1_e;
    int64_t stride_dw1_i;
    int64_t stride_workspace_split;
};

// K5: per-expert varlen-K dO^T @ (score * activation).
struct Dw2Kargs
{
    const hip_bfloat16* __restrict__ d_out;     // [T, D]
    const hip_bfloat16* __restrict__ a_scaled;  // [P, I]
    RouteMetadata route;
    hip_bfloat16* __restrict__ d_w2;            // [E, D, I]
    float* __restrict__ workspace;

    int model_dim;
    int inter_dim;
    int split_k;

    int64_t stride_do_t;
    int64_t stride_a_scaled_r;
    int64_t stride_dw2_e;
    int64_t stride_dw2_d;
    int64_t stride_workspace_split;
};

// Fused selected-softmax backward and scatter.  Top-k ids are the exact
// discrete routes chosen by forward; backward never selects routes again.
struct RouterBwdKargs
{
    const float* __restrict__ d_scores;        // [T, K]
    const float* __restrict__ scores;          // [T, K]
    const int32_t* __restrict__ topk_ids;       // [T, K]
    const int32_t* __restrict__ token_route_offsets; // compact: [T+1]
    float* __restrict__ d_logits;               // [T, E]

    int token_num;
    int topk;
    int num_experts;

    int64_t stride_ds_t;
    int64_t stride_score_t;
    int64_t stride_topk_id_t;
    int64_t stride_dl_t;
};

// Optional expert-bias gradients.  The launcher selects whole kernels from
// these flags; the no-bias K1--K5 path never constructs this kargs object.
struct BiasBwdKargs
{
    const hip_bfloat16* __restrict__ d_out;    // [T, D]
    const hip_bfloat16* __restrict__ d_z;      // [P, 2I]
    const float* __restrict__ scores;          // [T, K]
    const hip_bfloat16* __restrict__ b2;       // [E, D]
    RouteMetadata route;

    float* __restrict__ d_scores;              // [T, K], add bias term
    hip_bfloat16* __restrict__ d_b1;            // [E, 2I]
    hip_bfloat16* __restrict__ d_b2;            // [E, D]

    int model_dim;
    int inter_dim;
    bool compute_dscore;
    bool compute_db1;
    bool compute_db2;

    int64_t stride_do_t;
    int64_t stride_dz_r;
    int64_t stride_score_t;
    int64_t stride_b2_e;
    int64_t stride_ds_t;
    int64_t stride_db1_e;
    int64_t stride_db2_e;
};

} // namespace opus_moe_backward
