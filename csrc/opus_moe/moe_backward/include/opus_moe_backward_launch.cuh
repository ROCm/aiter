// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_backward.h"
#include "opus_gemm_arch.cuh"
#include "gfx950/opus_moe_backward_dispatch_gfx950.cuh"

#include "aiter_hip_common.h"
#include "aiter_stream.h"

#include <cstdint>

namespace opus_moe_backward
{
namespace detail
{

inline void check_route_metadata(const RouteMetadata& route,
                                 Family family,
                                 bool needs_sorted,
                                 bool needs_reverse,
                                 bool needs_expert_offsets)
{
    AITER_CHECK(route.token_num > 0,
                family_name(family),
                ": token_num must be positive, got ",
                route.token_num);
    AITER_CHECK(route.token_num <= static_cast<int>(kPackedTokenMask),
                family_name(family),
                ": token_num does not fit the 24-bit sorted-token encoding");
    AITER_CHECK(route.topk > 0 && route.topk <= kMaxPackedTopk,
                family_name(family),
                ": topk must be in [1, ",
                kMaxPackedTopk,
                "], got ",
                route.topk);
    if(needs_sorted || needs_expert_offsets)
        AITER_CHECK(route.num_experts > 0,
                    family_name(family),
                    ": num_experts must be positive");

    if(needs_sorted)
    {
        AITER_CHECK(route.sorted_token_ids != nullptr,
                    family_name(family),
                    ": sorted_token_ids must not be null");
        AITER_CHECK(route.sorted_expert_ids != nullptr,
                    family_name(family),
                    ": sorted_expert_ids must not be null");
        AITER_CHECK(route.num_valid_ids != nullptr,
                    family_name(family),
                    ": num_valid_ids must not be null");
        AITER_CHECK(route.sort_block_m > 0,
                    family_name(family),
                    ": sort_block_m must be positive");
        AITER_CHECK(route.sorted_capacity > 0 && route.sorted_block_capacity > 0,
                    family_name(family),
                    ": sorted metadata capacities must be positive");
    }
    if(needs_reverse)
        AITER_CHECK(route.reverse_sorted != nullptr,
                    family_name(family),
                    ": reverse_sorted must not be null");
    if(needs_expert_offsets)
        AITER_CHECK(route.expert_offsets != nullptr,
                    family_name(family),
                    ": expert_offsets must not be null");
}

inline void check_problem_dims(int model_dim, int inter_dim, Family family)
{
    AITER_CHECK(model_dim > 0,
                family_name(family),
                ": model_dim must be positive, got ",
                model_dim);
    AITER_CHECK(inter_dim > 0,
                family_name(family),
                ": inter_dim must be positive, got ",
                inter_dim);
}

inline void check_stride(int64_t stride, const char* name, Family family)
{
    AITER_CHECK(stride > 0,
                family_name(family),
                ": ",
                name,
                " must be a positive element stride, got ",
                stride);
}

inline void check_gfx950_or_fail()
{
    if(opus_get_gfx_arch() == OpusGfxArch::Gfx950)
        return;
    const auto& info = opus_get_arch_info();
    AITER_CHECK(false,
                "opus_moe_backward: only gfx950 is implemented; current device ",
                info.dev,
                " has gcnArchName='",
                info.name,
                "'");
}

template<typename Launcher, typename Kargs>
inline void invoke(Launcher launcher, const Kargs& kargs, hipStream_t stream, Family family)
{
    AITER_CHECK(launcher != nullptr,
                "opus_moe_backward: null launcher for family '",
                family_name(family),
                "'");
    if(launcher != nullptr)
        launcher(kargs, stream);
}

inline int select_fixed_route_dx_kernel_id(const RouteDxKargs& kargs,
                                            int requested_kernel_id)
{
    if(requested_kernel_id != kKernelAuto)
        return requested_kernel_id;

    constexpr int legacy_kid = 5;
    constexpr int cohort4_m2_kid = 7;
    constexpr int cohort10_m5_kid = 9;
    constexpr int bn256_m3_kid = 13;
    constexpr int output_n_tile = 128;
    // A cohort is useful only when there are enough route and N tiles to
    // trade a bounded amount of W1 reuse for much shorter dZ reuse distance.
    // The production cohort uses BK32: two async stages then occupy about half
    // the LDS and substantially fewer VGPRs than BK64 while preserving the
    // same MFMA work and exact BF16 result.
    // The policy depends on launch geometry and reusable sorting metadata,
    // not on an exact model tuple.  Keep the legacy layout for degenerate
    // grids and for contracts without expert offsets.
    if(kargs.route.expert_offsets == nullptr ||
       kargs.route.sorted_block_capacity < 4 ||
       kargs.model_dim <= output_n_tile)
        return legacy_kid;
    constexpr uint64_t l2_friendly_bytes = 128ull * 1024ull * 1024ull;
    const uint64_t gate_up_dim =
        2ull * static_cast<uint64_t>(kargs.inter_dim);
    const uint64_t dz_bytes =
        static_cast<uint64_t>(kargs.route.sorted_capacity) * gate_up_dim *
        sizeof(hip_bfloat16);
    const uint64_t w1_bytes =
        static_cast<uint64_t>(kargs.route.num_experts) * gate_up_dim *
        static_cast<uint64_t>(kargs.model_dim) * sizeof(hip_bfloat16);
    if(dz_bytes + w1_bytes <= l2_friendly_bytes)
        return cohort4_m2_kid;
    return kargs.model_dim % 256 == 0 ? bn256_m3_kid : cohort10_m5_kid;
}

inline bool select_fixed_down_bn256_geometry(const DownBwdKargs& kargs)
{
    if(kargs.route.expert_offsets == nullptr || kargs.inter_dim < 512 ||
       kargs.inter_dim % 256 != 0)
        return false;

    // BN256/M6 groups six adjacent sorter tiles and the compact launcher
    // rounds groups to four-group cohorts.  Below this grid size its wider
    // accumulator footprint cannot hide the lower workgroup count; above it,
    // halving gathered dO and dScore-part traffic wins across D/E/K families.
    constexpr uint64_t min_ctas = 384;
    const uint64_t sorted_blocks =
        static_cast<uint64_t>(kargs.route.sorted_block_capacity);
    const uint64_t groups =
        (sorted_blocks + 5ull) / 6ull +
        static_cast<uint64_t>(kargs.route.num_experts);
    const uint64_t padded_groups = ((groups + 3ull) / 4ull) * 4ull;
    const uint64_t parts = static_cast<uint64_t>(kargs.inter_dim / 256);
    return padded_groups * parts >= min_ctas;
}

inline int select_fixed_down_kernel_id(const DownBwdKargs& kargs,
                                       int requested_kernel_id)
{
    if(requested_kernel_id != kKernelAuto)
        return requested_kernel_id;

    constexpr int legacy_kid = 2;
    constexpr int five_route_tiles_predecoded_kid = 9;
    constexpr int six_route_tiles_predecoded_kid = 10;
    constexpr int bn256_six_route_tiles_predecoded_kid = 11;
    constexpr int bn256_deferred_z_wait_kid = 12;
    constexpr int deferred_z_wait_min_routes = 65536;
    constexpr uint64_t grouped_min_ctas = 256;
    constexpr uint64_t six_route_min_ctas = 640;
    if(kargs.route.expert_offsets == nullptr || kargs.d_scores_parts <= 1)
        return legacy_kid;

    if(select_fixed_down_bn256_geometry(kargs))
        return kargs.route.sorted_capacity >= deferred_z_wait_min_routes
                   ? bn256_deferred_z_wait_kid
                   : bn256_six_route_tiles_predecoded_kid;

    // Select from the useful M6 launch geometry rather than a model tuple or
    // an indirect tensor-byte proxy.  Each compact group owns up to six
    // sorted route tiles; the extra expert groups preserve boundaries when a
    // group straddles two experts.  Small grids need the legacy kernel's finer
    // load balance, medium grids favor M5, and large grids have enough CTAs to
    // amortize M6's wider W2 reuse without starving the 256 gfx950 CUs.
    const uint64_t sorted_blocks =
        static_cast<uint64_t>(kargs.route.sorted_block_capacity);
    const uint64_t m6_groups =
        (sorted_blocks + 5ull) / 6ull +
        static_cast<uint64_t>(kargs.route.num_experts);
    const uint64_t m6_ctas =
        m6_groups * static_cast<uint64_t>(kargs.d_scores_parts);
    if(m6_ctas < grouped_min_ctas)
        return legacy_kid;
    return m6_ctas < six_route_min_ctas
               ? five_route_tiles_predecoded_kid
               : six_route_tiles_predecoded_kid;
}

inline int select_fixed_dw1_kernel_id(const Dw1Kargs& kargs,
                                      int requested_kernel_id)
{
    if(requested_kernel_id != kKernelAuto)
        return requested_kernel_id;

    // Select only from runtime geometry and working-set size.  The BM128
    // double-stage path needs roughly three BK32 reduction tiles per expert
    // on average to hide its next-tile prefetch, and 2I must cover whole
    // BM128 output tiles.  Short or incompatible problems retain BM64.
    constexpr uint64_t l2_friendly_bytes = 128ull * 1024ull * 1024ull;
    constexpr int legacy_kid = 5;
    // The cohort paths use gfx950 buffer_load_*_lds so all three K32 operand
    // vectors land directly in their swizzled tiles.  Two experts retain
    // inter-expert load balance without extending the source reuse window.
    constexpr int cohort2_kid = 8;
    constexpr int cohort2_bm128_double_lds_kid = 9;
    constexpr int cohort2_bm256_wave4_double_lds_kid = 11;
    constexpr int reverse_cohort4_bm256_wave4_double_lds_kid = 13;
    if(kargs.route.num_experts < 2)
        return legacy_kid;
    const uint64_t source_row_elements =
        2ull * static_cast<uint64_t>(kargs.inter_dim) +
        static_cast<uint64_t>(kargs.model_dim);
    const uint64_t source_bytes =
        static_cast<uint64_t>(kargs.route.sorted_capacity) *
        source_row_elements * sizeof(hip_bfloat16);
    const uint64_t two_tile_rows =
        64ull * static_cast<uint64_t>(kargs.route.num_experts);
    const bool bm128_compatible =
        (2 * kargs.inter_dim) % 128 == 0 &&
        static_cast<uint64_t>(kargs.route.sorted_capacity) > two_tile_rows;
    if(bm128_compatible)
    {
        // BM256 lets four waves share one gathered-X tile, with each wave
        // owning eight independent native accumulator tiles.  The larger
        // register footprint pays off once the reduction is long enough and
        // the output grid either covers six CTAs per gfx950 CU or forms the
        // smaller four-CTA-per-CU full-residency grid.  Express the crossover
        // only in host-visible grouped-GEMM geometry; no model tuple or
        // routing distribution is synchronized back to the host.
        const uint64_t average_padded_routes =
            static_cast<uint64_t>(kargs.route.sorted_capacity) /
            static_cast<uint64_t>(kargs.route.num_experts);
        const bool bm256_compatible = (2 * kargs.inter_dim) % 256 == 0;
        const uint64_t bm256_output_tiles =
            static_cast<uint64_t>(kargs.route.num_experts) *
            static_cast<uint64_t>((2 * kargs.inter_dim) / 256) *
            static_cast<uint64_t>(kargs.model_dim / 128);
        constexpr uint64_t min_average_routes = 2048;
        constexpr uint64_t compact_full_residency_tiles = 1024;
        constexpr uint64_t compact_grid_alignment = 512;
        constexpr uint64_t min_output_tiles = 1536;
        const bool compact_full_residency_grid =
            bm256_output_tiles >= compact_full_residency_tiles &&
            bm256_output_tiles < min_output_tiles &&
            bm256_output_tiles % compact_grid_alignment == 0;
        if(bm256_compatible && average_padded_routes >= min_average_routes &&
           (bm256_output_tiles >= min_output_tiles ||
            compact_full_residency_grid))
        {
            // K2 walks sorted routes from low to high expert IDs.  With a
            // long reduction, start K4 at the newest dZ rows and interleave
            // four experts so producer-consumer L2 reuse does not fight
            // route-count imbalance.  Shorter reductions and wider I retain
            // cohort2, whose smaller live expert set wins those families.
            constexpr uint64_t reverse_min_average_routes = 3072;
            constexpr int reverse_max_inter_dim = 1024;
            if(average_padded_routes >= reverse_min_average_routes &&
               kargs.inter_dim <= reverse_max_inter_dim)
                return reverse_cohort4_bm256_wave4_double_lds_kid;
            return cohort2_bm256_wave4_double_lds_kid;
        }
        return cohort2_bm128_double_lds_kid;
    }
    return source_bytes > l2_friendly_bytes ? cohort2_kid : legacy_kid;
}

inline int select_fixed_dw2_kernel_id(const Dw2Kargs& kargs,
                                      int requested_kernel_id)
{
    if(requested_kernel_id != kKernelAuto)
        return requested_kernel_id;

    constexpr uint64_t l2_friendly_bytes = 128ull * 1024ull * 1024ull;
    constexpr uint64_t large_working_set_bytes = 512ull * 1024ull * 1024ull;
    constexpr int legacy_kid = 3;
    constexpr int cohort4_direct_lds_kid = 5;
    constexpr int cohort1_direct_lds_kid = 6;
    constexpr int cohort2_direct_lds_kid = 7;
    constexpr int cohort1_bm128_dual_lds_kid = 9;
    constexpr int bm128x128_adaptive_routes_kid = 10;
    constexpr int bm256x128_direct_kid = 11;
    constexpr int bm256x128_k32_triple_lds_kid = 12;
    constexpr int bm256x128_k32_prefetch_ab_triple_lds_kid = 13;
    constexpr int bm256x128_k32_native_b32_zero_pad_kid = 16;
    constexpr int bm128x256_k32_native_b32_zero_pad_grid3d_kid = 18;
    if(kargs.route.num_experts < 4)
        return legacy_kid;
    // A wide output grid amortizes the larger output tile.  Require at least
    // two K64 reduction tiles per expert on average so source reuse offsets
    // its wider LDS and accumulator footprint.  These tests use only
    // host-visible geometry and introduce no routing synchronization.
    const bool bm128x128_compatible =
        kargs.model_dim % 128 == 0 && kargs.inter_dim % 128 == 0;
    const uint64_t wide_output_blocks =
        static_cast<uint64_t>(kargs.route.num_experts) *
        static_cast<uint64_t>(kargs.model_dim / 128) *
        static_cast<uint64_t>(kargs.inter_dim / 128);
    const bool bm256x128_compatible =
        kargs.model_dim % 256 == 0 && kargs.inter_dim % 128 == 0;
    const uint64_t bm256_output_blocks =
        static_cast<uint64_t>(kargs.route.num_experts) *
        static_cast<uint64_t>(kargs.model_dim / 256) *
        static_cast<uint64_t>(kargs.inter_dim / 128);
    const uint64_t bm256_blocks_per_expert =
        static_cast<uint64_t>(kargs.model_dim / 256) *
        static_cast<uint64_t>(kargs.inter_dim / 128);
    const uint64_t average_padded_routes =
        static_cast<uint64_t>(kargs.route.sorted_capacity) /
        static_cast<uint64_t>(kargs.route.num_experts);
    if(bm128x128_compatible && wide_output_blocks >= 512 &&
       average_padded_routes >= 128)
    {
        // On a 256-CU gfx950, 4096 BM256xBN128 tiles provide sixteen CTA
        // rounds.  Four waves then share a_scaled across twice as many dO
        // rows, and each wave pipelines four K16 fragments through eight
        // independent native C tiles.  Bound the per-expert run length so
        // sparse routing cannot leave a very wide expert-major tail. The
        // BM256 family wins from short balanced reductions through
        // 24k-route experts and maximally skewed routing, avoiding the launch
        // and empty-grid cost of the retired two-kernel route split.
        constexpr uint64_t bm256_min_output_blocks = 4096;
        constexpr uint64_t bm256_max_blocks_per_expert = 64;
        if(bm256x128_compatible &&
           bm256_output_blocks >= bm256_min_output_blocks &&
           bm256_blocks_per_expert <= bm256_max_blocks_per_expert)
        {
            // K32 uses the same BM256xBN128 output tile with three 24-KiB
            // operand stages, overlapping two future route tiles with the
            // current MFMA.  The extra pipeline boundary pays off from
            // roughly 1024 padded routes per expert; shorter reductions
            // retain K64's lower loop/barrier count.  On long reductions,
            // queue both second-K16 LDS fragments before the first MFMA
            // chain.  The crossover is expressed only in reduction/output
            // geometry: the containing branch already requires at least
            // 4096 BM256xBN128 output blocks.
            constexpr uint64_t k32_min_average_routes = 1024;
            constexpr uint64_t k32_prefetch_ab_min_average_routes = 3072;
            // The balanced BM128xBN256 tile preserves CTA count, MFMA count,
            // and LDS footprint while halving repeated random dO gathers.
            // Prefer it whenever the forward-owned zero-padded activation
            // cache can support the native BN256 operand layout.
            if(average_padded_routes >= k32_min_average_routes &&
               kargs.a_scaled_padding_zero && kargs.inter_dim % 256 == 0)
                return bm128x256_k32_native_b32_zero_pad_grid3d_kid;
            if(average_padded_routes >=
               k32_prefetch_ab_min_average_routes)
                return kargs.a_scaled_padding_zero
                           ? bm256x128_k32_native_b32_zero_pad_kid
                           : bm256x128_k32_prefetch_ab_triple_lds_kid;
            return average_padded_routes >= k32_min_average_routes
                       ? bm256x128_k32_triple_lds_kid
                       : bm256x128_direct_kid;
        }
        return bm128x128_adaptive_routes_kid;
    }
    const uint64_t source_row_elements =
        static_cast<uint64_t>(kargs.model_dim) +
        static_cast<uint64_t>(kargs.inter_dim);
    const uint64_t source_bytes =
        static_cast<uint64_t>(kargs.route.sorted_capacity) *
        source_row_elements * sizeof(hip_bfloat16);
    if(source_bytes <= l2_friendly_bytes)
        return legacy_kid;
    // Once the combined gathered-dO/a_scaled working set is much larger
    // than L2, finish one expert at a time and double the output-M tile.  The
    // larger tile halves the output grid while the dual LDS stages keep K64
    // loads overlapped; it is also robust when only a subset of experts is
    // active because the decision does not depend on host-visible counts.
    if(source_bytes > large_working_set_bytes)
        return cohort1_bm128_dual_lds_kid;
    // In the medium regime retain the largest bounded cohort that does not
    // pad the expert grid.  Empty expert CTAs are otherwise material at these
    // shorter runtimes (for example E=10/14 versus E=12/16).
    if(kargs.route.num_experts % 4 == 0)
        return cohort4_direct_lds_kid;
    if(kargs.route.num_experts % 2 == 0)
        return cohort2_direct_lds_kid;
    return cohort1_direct_lds_kid;
}

} // namespace detail

} // namespace opus_moe_backward

// Checked family launches and fixed-pipeline scheduling.
namespace opus_moe_backward
{

void launch_down_bwd_bf16(const DownBwdKargs& kargs,
                          int kernel_id,
                          hipStream_t stream)
{
    constexpr Family family = Family::DownBwd;
    detail::check_route_metadata(kargs.route, family, true, false, false);
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.d_out != nullptr && kargs.z != nullptr && kargs.w2 != nullptr &&
                    kargs.scores != nullptr && kargs.d_z != nullptr &&
                    kargs.a_scaled != nullptr && kargs.d_scores != nullptr,
                "down_bwd: required input/output pointer is null");
    AITER_CHECK(kargs.d_scores_parts > 0,
                "down_bwd: d_scores_parts must be positive");
    if(kargs.d_scores_parts > 1)
        AITER_CHECK(kargs.d_scores_workspace != nullptr,
                    "down_bwd: d_scores_workspace is required for multipart reduction");
    detail::check_stride(kargs.stride_do_t, "stride_do_t", family);
    detail::check_stride(kargs.stride_z_r, "stride_z_r", family);
    detail::check_stride(kargs.stride_w2_e, "stride_w2_e", family);
    detail::check_stride(kargs.stride_w2_d, "stride_w2_d", family);
    detail::check_stride(kargs.stride_score_t, "stride_score_t", family);
    detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
    detail::check_stride(kargs.stride_a_scaled_r, "stride_a_scaled_r", family);
    detail::check_stride(kargs.stride_ds_t, "stride_ds_t", family);
    if(kargs.d_scores_parts > 1)
        detail::check_stride(
            kargs.stride_ds_workspace_r, "stride_ds_workspace_r", family);

    detail::check_gfx950_or_fail();
    const int selected_kernel_id =
        detail::select_fixed_down_kernel_id(kargs, kernel_id);
    if(selected_kernel_id == 19)
    {
        const uint64_t dz_bytes =
            static_cast<uint64_t>(kargs.route.sorted_capacity) *
            static_cast<uint64_t>(kargs.stride_dz_r) *
            sizeof(hip_bfloat16);
        AITER_CHECK(dz_bytes <= 0xffffffffull,
                    "down_bwd: MUBUF dZ store requires d_z below 4 GiB");
    }
    detail::invoke(
        gfx950::dispatch_down_bwd(selected_kernel_id), kargs, stream, family);
}

void launch_route_dx_bf16(const RouteDxKargs& kargs,
                          int kernel_id,
                          hipStream_t stream)
{
    constexpr Family family = Family::RouteDx;
    detail::check_route_metadata(kargs.route, family, true, false, false);
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.d_z != nullptr && kargs.w1 != nullptr && kargs.d_x_route != nullptr,
                "route_dx: required input/output pointer is null");
    detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
    detail::check_stride(kargs.stride_w1_e, "stride_w1_e", family);
    detail::check_stride(kargs.stride_w1_i, "stride_w1_i", family);
    detail::check_stride(kargs.stride_dx_route_r, "stride_dx_route_r", family);

    detail::check_gfx950_or_fail();
    const int selected_kernel_id =
        detail::select_fixed_route_dx_kernel_id(kargs, kernel_id);
    if(selected_kernel_id == 21)
    {
        const uint64_t route_output_bytes =
            static_cast<uint64_t>(kargs.route.sorted_capacity) *
            static_cast<uint64_t>(kargs.stride_dx_route_r) *
            sizeof(hip_bfloat16);
        AITER_CHECK(route_output_bytes <= 0xffffffffull,
                    "route_dx: MUBUF output store requires d_x_route below "
                    "4 GiB");
    }
    detail::invoke(
        gfx950::dispatch_route_dx(selected_kernel_id), kargs, stream, family);
}

void launch_route_reduce_bf16(const RouteReduceKargs& kargs,
                              int kernel_id,
                              hipStream_t stream)
{
    constexpr Family family = Family::RouteReduce;
    detail::check_route_metadata(kargs.route, family, false, false, false);
    AITER_CHECK(kargs.model_dim > 0,
                "route_reduce: model_dim must be positive, got ",
                kargs.model_dim);
    AITER_CHECK(kargs.d_x_route != nullptr && kargs.d_x != nullptr,
                "route_reduce: required input/output pointer is null");
    detail::check_stride(kargs.stride_dx_route_r, "stride_dx_route_r", family);
    detail::check_stride(kargs.stride_dx_t, "stride_dx_t", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_route_reduce(kernel_id), kargs, stream, family);
}

void launch_router_bwd_fp32(const RouterBwdKargs& kargs,
                            int kernel_id,
                            hipStream_t stream)
{
    constexpr Family family = Family::RouterBwd;
    AITER_CHECK(kargs.token_num > 0,
                "router_bwd: token_num must be positive, got ",
                kargs.token_num);
    AITER_CHECK(kargs.topk == 1 || kargs.topk == 2 ||
                    kargs.topk == 4 || kargs.topk == 8,
                "router_bwd: topk must be in {1,2,4,8}");
    AITER_CHECK(kargs.num_experts >= kargs.topk,
                "router_bwd: num_experts must be at least topk");
    AITER_CHECK(kargs.d_scores != nullptr && kargs.scores != nullptr &&
                    kargs.topk_ids != nullptr && kargs.d_logits != nullptr,
                "router_bwd: required input/output pointer is null");
    detail::check_stride(kargs.stride_ds_t, "stride_ds_t", family);
    detail::check_stride(kargs.stride_score_t, "stride_score_t", family);
    detail::check_stride(kargs.stride_topk_id_t, "stride_topk_id_t", family);
    detail::check_stride(kargs.stride_dl_t, "stride_dl_t", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_router_bwd(kernel_id), kargs, stream, family);
}

void launch_bias_bwd_bf16(const BiasBwdKargs& kargs,
                          int kernel_id,
                          hipStream_t stream)
{
    constexpr Family family = Family::BiasBwd;
    AITER_CHECK(kargs.compute_dscore || kargs.compute_db1 || kargs.compute_db2,
                "bias_bwd: at least one output must be requested");
    AITER_CHECK(kargs.route.token_num > 0 && kargs.route.num_experts > 0,
                "bias_bwd: token and expert counts must be positive");
    AITER_CHECK(kargs.route.topk == 1 || kargs.route.topk == 2 ||
                    kargs.route.topk == 4 || kargs.route.topk == 8,
                "bias_bwd: topk must be in {1,2,4,8}");
    AITER_CHECK(kargs.route.sorted_token_ids != nullptr &&
                    kargs.route.num_valid_ids != nullptr,
                "bias_bwd: sorted_token_ids and num_valid_ids are required");
    if(kargs.compute_dscore)
        AITER_CHECK(kargs.route.sorted_expert_ids != nullptr &&
                        kargs.d_out != nullptr && kargs.b2 != nullptr &&
                        kargs.d_scores != nullptr,
                    "bias_bwd: dscore inputs/outputs are required");
    if(kargs.compute_db1)
        AITER_CHECK(kargs.route.expert_offsets != nullptr &&
                        kargs.d_z != nullptr && kargs.d_b1 != nullptr,
                    "bias_bwd: db1 inputs/outputs are required");
    if(kargs.compute_db2)
        AITER_CHECK(kargs.route.expert_offsets != nullptr &&
                        kargs.d_out != nullptr && kargs.scores != nullptr &&
                        kargs.d_b2 != nullptr,
                    "bias_bwd: db2 inputs/outputs are required");
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    if(kargs.compute_dscore || kargs.compute_db2)
    {
        detail::check_stride(kargs.stride_do_t, "stride_do_t", family);
        detail::check_stride(kargs.stride_score_t, "stride_score_t", family);
    }
    if(kargs.compute_dscore)
    {
        detail::check_stride(kargs.stride_b2_e, "stride_b2_e", family);
        detail::check_stride(kargs.stride_ds_t, "stride_ds_t", family);
    }
    if(kargs.compute_db1)
    {
        detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
        detail::check_stride(kargs.stride_db1_e, "stride_db1_e", family);
    }
    if(kargs.compute_db2)
        detail::check_stride(kargs.stride_db2_e, "stride_db2_e", family);

    detail::check_gfx950_or_fail();
    detail::invoke(gfx950::dispatch_bias_bwd(kernel_id), kargs, stream, family);
}

void launch_dw1_bf16(const Dw1Kargs& kargs,
                     int kernel_id,
                     hipStream_t stream)
{
    constexpr Family family = Family::Dw1;
    detail::check_route_metadata(kargs.route, family, false, false, true);
    AITER_CHECK(kargs.route.sorted_token_ids != nullptr &&
                    kargs.route.num_valid_ids != nullptr,
                "dw1: sorted_token_ids and num_valid_ids must not be null");
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.x != nullptr && kargs.d_z != nullptr && kargs.d_w1 != nullptr,
                "dw1: required input/output pointer is null");
    AITER_CHECK(kargs.split_k > 0, "dw1: split_k must be positive");
    if(kargs.split_k > 1)
        AITER_CHECK(kargs.workspace != nullptr,
                    "dw1: FP32 workspace is required when split_k > 1");
    detail::check_stride(kargs.stride_x_t, "stride_x_t", family);
    detail::check_stride(kargs.stride_dz_r, "stride_dz_r", family);
    detail::check_stride(kargs.stride_dw1_e, "stride_dw1_e", family);
    detail::check_stride(kargs.stride_dw1_i, "stride_dw1_i", family);
    if(kargs.split_k > 1)
        detail::check_stride(
            kargs.stride_workspace_split, "stride_workspace_split", family);

    detail::check_gfx950_or_fail();
    const int selected_kernel_id =
        detail::select_fixed_dw1_kernel_id(kargs, kernel_id);
    detail::invoke(
        gfx950::dispatch_dw1(selected_kernel_id), kargs, stream, family);
}

void launch_dw2_bf16(const Dw2Kargs& kargs,
                     int kernel_id,
                     hipStream_t stream)
{
    constexpr Family family = Family::Dw2;
    detail::check_route_metadata(kargs.route, family, false, false, true);
    AITER_CHECK(kargs.route.sorted_token_ids != nullptr &&
                    kargs.route.num_valid_ids != nullptr,
                "dw2: sorted_token_ids and num_valid_ids must not be null");
    detail::check_problem_dims(kargs.model_dim, kargs.inter_dim, family);
    AITER_CHECK(kargs.d_out != nullptr && kargs.a_scaled != nullptr &&
                    kargs.d_w2 != nullptr,
                "dw2: required input/output pointer is null");
    AITER_CHECK(kargs.split_k > 0, "dw2: split_k must be positive");
    if(kargs.split_k > 1)
        AITER_CHECK(kargs.workspace != nullptr,
                    "dw2: FP32 workspace is required when split_k > 1");
    detail::check_stride(kargs.stride_do_t, "stride_do_t", family);
    detail::check_stride(kargs.stride_a_scaled_r, "stride_a_scaled_r", family);
    detail::check_stride(kargs.stride_dw2_e, "stride_dw2_e", family);
    detail::check_stride(kargs.stride_dw2_d, "stride_dw2_d", family);
    if(kargs.split_k > 1)
        detail::check_stride(
            kargs.stride_workspace_split, "stride_workspace_split", family);

    detail::check_gfx950_or_fail();
    const int selected_kernel_id =
        detail::select_fixed_dw2_kernel_id(kargs, kernel_id);
    AITER_CHECK((selected_kernel_id != 16 &&
                 selected_kernel_id != 18) ||
                    kargs.a_scaled_padding_zero,
                "dw2: native-B K32 kernels require exact-zero a_scaled "
                "padding from the full K1--K5 cache contract");
    detail::invoke(
        gfx950::dispatch_dw2(selected_kernel_id), kargs, stream, family);
}

void launch_sorted_x_blocked_g2_bf16(const SortedXBlockedG2Kargs& kargs,
                                     hipStream_t stream)
{
    AITER_CHECK(kargs.x != nullptr && kargs.sorted_token_ids != nullptr &&
                    kargs.num_valid_ids != nullptr &&
                    kargs.x_sorted != nullptr,
                "sorted-X blocked-G2: required input/output pointer is null");
    AITER_CHECK(kargs.token_num > 0 &&
                    kargs.token_num <= static_cast<int>(kPackedTokenMask),
                "sorted-X blocked-G2: token count must fit packed metadata");
    AITER_CHECK(kargs.model_dim > 0 && kargs.model_dim % 32 == 0,
                "sorted-X blocked-G2: D must be positive and divisible by 32");
    AITER_CHECK(kargs.sorted_capacity > 0,
                "sorted-X blocked-G2: sorted capacity must be positive");
    AITER_CHECK(kargs.stride_x_t == kargs.model_dim &&
                    kargs.stride_x_sorted_r == kargs.model_dim,
                "sorted-X blocked-G2: X and cache must be contiguous");
    detail::check_gfx950_or_fail();
    gfx950::sorted_x_blocked_g2_launch_gfx950(kargs, stream);
}

namespace detail
{

// Keep the production fixed-top-k pipeline visible as one flat K1--K5
// sequence. Validation and kargs construction remain at the public boundary.
inline void launch_fixed_pipeline(const DownBwdKargs& down,
                                  const RouteDxKargs& route_dx,
                                  const RouteReduceKargs& route_reduce,
                                  const Dw1Kargs& dw1,
                                  const Dw2Kargs& dw2,
                                  int down_kernel_id,
                                  int route_dx_kernel_id,
                                  int route_reduce_kernel_id,
                                  int dw1_kernel_id,
                                  int dw2_kernel_id,
                                  bool x_dw1_blocked_g2,
                                  hipStream_t stream)
{
    check_gfx950_or_fail();
    constexpr int blocked_down_sparse_owner_kid = 18;
    constexpr int blocked_down_mubuf_store_kid = 19;
    constexpr int blocked_route_dx_kid = 20;
    constexpr int blocked_route_dx_mubuf_store_kid = 21;
    constexpr int blocked_dw1_kid = 25;
    const bool blocked_down = down_kernel_id == blocked_down_sparse_owner_kid ||
                              down_kernel_id == blocked_down_mubuf_store_kid;
    if(down_kernel_id == blocked_down_mubuf_store_kid)
    {
        const uint64_t dz_store_bytes =
            static_cast<uint64_t>(down.route.sorted_capacity) *
            static_cast<uint64_t>(down.stride_dz_r) *
            sizeof(hip_bfloat16);
        AITER_CHECK(dz_store_bytes <= 0xffffffffull,
                    "fixed full pipeline: MUBUF dZ store requires d_z below "
                    "4 GiB");
    }
    const bool blocked_route_dx =
        route_dx_kernel_id == blocked_route_dx_kid ||
        route_dx_kernel_id == blocked_route_dx_mubuf_store_kid;
    if(route_dx_kernel_id == blocked_route_dx_mubuf_store_kid)
    {
        const uint64_t route_output_bytes =
            static_cast<uint64_t>(route_dx.route.sorted_capacity) *
            static_cast<uint64_t>(route_dx.stride_dx_route_r) *
            sizeof(hip_bfloat16);
        AITER_CHECK(route_output_bytes <= 0xffffffffull,
                    "fixed full pipeline: MUBUF route output store requires "
                    "d_x_route below 4 GiB");
    }
    const bool blocked_dw1 = dw1_kernel_id == blocked_dw1_kid;
    AITER_CHECK(x_dw1_blocked_g2 == blocked_dw1,
                "fixed full pipeline: blocked-G2 sorted-X requires K4 kernel 25");
    const bool any_blocked_dz = blocked_down || blocked_route_dx || blocked_dw1;
    AITER_CHECK(!any_blocked_dz ||
                    (blocked_down && blocked_route_dx && blocked_dw1),
                "fixed full pipeline: blocked dZ K1/K2/K4 instances must "
                "be selected together");

    int selected_route_dx =
        select_fixed_route_dx_kernel_id(route_dx, route_dx_kernel_id);
    int selected_route_reduce =
        route_reduce_kernel_id == kKernelAuto ? 0 : route_reduce_kernel_id;

    // Large K>=4 route families benefit from keeping K2's output in natural
    // expert-sorted order and moving the inverse permutation to K3.  Reuse
    // the existing working-set selector: kids 9 and 13 identify the large M5
    // and BN256-M3 paths whose dZ+W1 footprint exceeds the L2-friendly regime.
    // Explicit selection of either half also completes the pair, while
    // incompatible explicit pairs fail instead of silently interpreting the
    // workspace incorrectly.
    constexpr int logical_route_dx_kid = 9;
    constexpr int sorted_route_dx_kid = 11;
    constexpr int logical_bn256_route_dx_kid = 13;
    constexpr int sorted_bn256_route_dx_kid = 14;
    constexpr int sorted_bn256_b_first_route_dx_kid = 15;
    constexpr int sorted_bn256_m5_b_first_route_dx_kid = 16;
    constexpr int sorted_bn256_m5_binary_route_dx_kid = 17;
    constexpr int sorted_bn512_m3_binary_route_dx_kid = 18;
    constexpr int sorted_bn512_m3_binary_n_fast_route_dx_kid = 19;
    constexpr int b_first_min_routes = 250000;
    constexpr uint64_t m5_min_average_routes = 1536;
    constexpr uint64_t bn512_min_dz_bytes = 1024ull * 1024ull * 1024ull;
    constexpr int sorted_route_reduce_kid = 1;
    constexpr int distributed_ids_route_reduce_kid = 3;
    const auto is_sorted_route_reduce_kid = [&](int kid) {
        return kid == sorted_route_reduce_kid ||
               kid == distributed_ids_route_reduce_kid;
    };
    const auto select_sorted_route_reduce_kid = [&]() {
        return route_reduce.model_dim == 2048
                   ? distributed_ids_route_reduce_kid
                   : sorted_route_reduce_kid;
    };
    const auto sorted_route_kid = [&](int logical_kid) {
        if(logical_kid != logical_bn256_route_dx_kid)
            return sorted_route_dx_kid;

        const uint64_t average_padded_routes =
            static_cast<uint64_t>(route_dx.route.sorted_capacity) /
            static_cast<uint64_t>(route_dx.route.num_experts);
        if(average_padded_routes >= m5_min_average_routes)
        {
            const uint64_t route_dz_bytes =
                static_cast<uint64_t>(route_dx.route.sorted_capacity) *
                static_cast<uint64_t>(2 * route_dx.inter_dim) *
                sizeof(hip_bfloat16);
            // In the full mixed-kernel schedule, very large dZ streams can
            // amortize BN512's one-workgroup residency by sharing each dZ
            // load across twice as many output columns.  Keep the narrower
            // M5 kernel for smaller working sets and standalone dispatch.
            if(route_dx.model_dim % 512 == 0 &&
               route_dz_bytes >= bn512_min_dz_bytes)
            {
                const int output_n_tiles = route_dx.model_dim / 512;
                return output_n_tiles == 4 || output_n_tiles == 8
                           ? sorted_bn512_m3_binary_n_fast_route_dx_kid
                           : sorted_bn512_m3_binary_route_dx_kid;
            }
            return sorted_bn256_m5_binary_route_dx_kid;
        }

        // The BN256/M3 stage moves about 16 KiB of W1 but only 6 KiB of dZ
        // per stage.  On long sorted streams, issuing W1 first starts the
        // vmcnt-critical transfer earlier; shorter streams retain A-first at
        // the measured crossover.  This uses only runtime route geometry and
        // deliberately avoids binding the policy to a model tuple.
        return route_dx.route.sorted_capacity >= b_first_min_routes
                   ? sorted_bn256_b_first_route_dx_kid
                   : sorted_bn256_route_dx_kid;
    };
    const auto is_logical_large_route_kid = [&](int kid) {
        return kid == logical_route_dx_kid ||
               kid == logical_bn256_route_dx_kid;
    };
    const auto is_sorted_route_kid = [&](int kid) {
        return kid == sorted_route_dx_kid ||
               kid == sorted_bn256_route_dx_kid ||
               kid == sorted_bn256_b_first_route_dx_kid ||
               kid == sorted_bn256_m5_b_first_route_dx_kid ||
               kid == sorted_bn256_m5_binary_route_dx_kid ||
               kid == sorted_bn512_m3_binary_route_dx_kid ||
               kid == sorted_bn512_m3_binary_n_fast_route_dx_kid ||
               kid == blocked_route_dx_kid ||
               kid == blocked_route_dx_mubuf_store_kid;
    };
    const bool auto_sorted_route_pair =
        route_dx_kernel_id == kKernelAuto &&
        route_reduce_kernel_id == kKernelAuto &&
        is_logical_large_route_kid(selected_route_dx) &&
        route_dx.route.topk >= 4;
    if(auto_sorted_route_pair)
    {
        selected_route_dx = sorted_route_kid(selected_route_dx);
        selected_route_reduce = select_sorted_route_reduce_kid();
    }
    else if(is_sorted_route_kid(route_dx_kernel_id) &&
            route_reduce_kernel_id == kKernelAuto)
    {
        selected_route_reduce = select_sorted_route_reduce_kid();
    }
    else if(is_sorted_route_reduce_kid(route_reduce_kernel_id) &&
            route_dx_kernel_id == kKernelAuto)
    {
        selected_route_dx = sorted_route_kid(selected_route_dx);
    }
    AITER_CHECK(is_sorted_route_kid(selected_route_dx) ==
                    is_sorted_route_reduce_kid(selected_route_reduce),
                "fixed route pipeline: a sorted route workspace kernel and "
                "a sorted route-reduce kernel must be selected together");

    // K4 and K2 both consume K1's much larger dZ stream.  For the bounded
    // mid-width D family once dZ has outgrown the cache-friendly regime,
    // consume it with K4 immediately, then let K2's read refresh dZ before
    // route reduction.  K5 normally moves last so dZ stays warm across the
    // two dominant grouped GEMMs.  For the largest forward-cache working
    // sets, K5 can run before K1 and leave all dZ consumers adjacent.
    // Small working sets and narrower or wider grids retain the legacy order.
    const int model_dim = dw1.model_dim;
    const bool mid_width_d = model_dim >= 1536 && model_dim <= 2048;
    constexpr uint64_t dz_reorder_min_bytes = 512ull * 1024ull * 1024ull;
    constexpr uint64_t dz_k5_before_route_min_bytes =
        1024ull * 1024ull * 1024ull;
    const uint64_t dz_bytes =
        static_cast<uint64_t>(down.route.sorted_capacity) *
        static_cast<uint64_t>(2 * down.inter_dim) * sizeof(hip_bfloat16);
    const bool uses_saved_a_and_sorted_x =
        (down_kernel_id == 16 ||
         down_kernel_id == blocked_down_sparse_owner_kid ||
         down_kernel_id == blocked_down_mubuf_store_kid) &&
        (dw1_kernel_id == 15 || dw1_kernel_id == 18 ||
         blocked_dw1);
    const bool launch_saved_dw2_before_down =
        mid_width_d && uses_saved_a_and_sorted_x &&
        dz_bytes >= dz_k5_before_route_min_bytes;

    // With both forward-owned caches, K5 is independent of K1.  Run it first
    // for the largest working sets so K1, K4, and K2 can consume dZ back to
    // back.  This is selected from runtime byte geometry rather than a model
    // tuple, and is legal only for the explicit saved-a_scaled contract.
    if(launch_saved_dw2_before_down)
        invoke(gfx950::dispatch_dw2(
                   select_fixed_dw2_kernel_id(dw2, dw2_kernel_id)),
               dw2,
               stream,
               Family::Dw2);

    invoke(gfx950::dispatch_down_bwd(
               select_fixed_down_kernel_id(down, down_kernel_id)),
           down,
           stream,
           Family::DownBwd);
    if(mid_width_d && dz_bytes >= dz_reorder_min_bytes)
    {
        invoke(gfx950::dispatch_dw1(
                   select_fixed_dw1_kernel_id(dw1, dw1_kernel_id)),
               dw1,
               stream,
               Family::Dw1);
        // Keep K2/K3 as one producer-consumer pair after K4.  This order also
        // gives both dominant grouped GEMMs adjacent access to K1's dZ stream.
        invoke(gfx950::dispatch_route_dx(selected_route_dx),
               route_dx,
               stream,
               Family::RouteDx);
        invoke(gfx950::dispatch_route_reduce(selected_route_reduce),
               route_reduce,
               stream,
               Family::RouteReduce);
        if(!launch_saved_dw2_before_down)
            invoke(gfx950::dispatch_dw2(
                       select_fixed_dw2_kernel_id(dw2, dw2_kernel_id)),
                   dw2,
                   stream,
                   Family::Dw2);
    }
    else
    {
        invoke(gfx950::dispatch_route_dx(selected_route_dx),
               route_dx,
               stream,
               Family::RouteDx);
        invoke(gfx950::dispatch_route_reduce(selected_route_reduce),
               route_reduce,
               stream,
               Family::RouteReduce);
        invoke(gfx950::dispatch_dw1(
                   select_fixed_dw1_kernel_id(dw1, dw1_kernel_id)),
               dw1,
               stream,
               Family::Dw1);
        invoke(gfx950::dispatch_dw2(
                   select_fixed_dw2_kernel_id(dw2, dw2_kernel_id)),
               dw2,
               stream,
               Family::Dw2);
    }
}

} // namespace detail

} // namespace opus_moe_backward
