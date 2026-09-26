// gfx942 host launch helpers for the experimental exact-local-Mqk route.
//
// These helpers intentionally live outside policy.hpp and hip_launch_common.cu.
// A future opt-in adapter can reuse the existing launch structs while keeping
// the experimental kernels and their packed-four-tile ABI architecture-local.
#pragma once

#include "../hip_common.hpp"
#include "k1_kda_bt64_fused_prepare_neumann_local_mqk_experimental.hpp"
#include "k2_kda_csplit_bt64_out_hybrid_local_balanced_experimental.hpp"

namespace flashkda_hip::gfx942::experimental {

// `a.segment_a` addresses scratch owned by the common launcher.  The default
// producer uses its first four tiles as packed
// [H*total_segments,4,16,16] exact local-Mqk storage.  C16_LAYOUT instead
// treats it as the ordinary contiguous [H*total_tiles,16,16] ws_mqk arena.
template <bool C16_LAYOUT>
inline void launch_fused_prepare_local_mqk_impl(
        const FusedPrepareNeumannLaunch& a) {
    const dim3 grid = a.is_varlen
        ? dim3(a.total_segments, a.H)
        : dim3((a.NT + 3) / 4, a.N * a.H);
    auto launch = [&]<bool VL, bool USE_DPP>() {
        k1_kda_bt64_fused_prepare_neumann_local_mqk_kernel<
            VL, USE_DPP, false, C16_LAYOUT>
            <<<grid, 256, 0, a.stream>>>(
                a.q, a.k, a.g, a.beta, a.A_log, a.dt_bias,
                a.scale, a.gate_scale, a.kd, a.qd, a.kr, a.gt,
                a.decay, a.inv, a.cross32, a.cross64, a.beta_cache,
                a.segment_a, VL ? a.cu_seqlens : nullptr,
                VL ? a.tile_prefix : nullptr,
                VL ? a.pair_prefix : nullptr,
                VL ? a.segment_prefix : nullptr,
                a.N, a.total_tiles, a.total_pairs, a.total_segments,
                a.T_seq, a.H, a.NT);
    };
    if (a.is_varlen) {
        if (a.use_dpp)
            launch.template operator()<true, true>();
        else
            launch.template operator()<true, false>();
    } else if (a.use_dpp) {
        launch.template operator()<false, true>();
    } else {
        launch.template operator()<false, false>();
    }
}

inline void launch_fused_prepare_local_mqk(
        const FusedPrepareNeumannLaunch& a) {
    launch_fused_prepare_local_mqk_impl<false>(a);
}

inline void launch_fused_prepare_local_mqk_c16(
        const FusedPrepareNeumannLaunch& a) {
    launch_fused_prepare_local_mqk_impl<true>(a);
}

// Consumes the first four packed local tiles in `a.segment_a`.  It rebuilds
// only the six strict cross-chunk tiles and writes the complete output.
inline void launch_k6_hybrid_local_output(
        const Csplit64K6OutputLaunch& a) {
    const dim3 grid = a.is_varlen
        ? dim3(a.total_segments, a.H)
        : dim3((a.NT + 3) / 4, a.N * a.H);
    if (a.is_varlen) {
        k2_kda_csplit_bt64_out_hybrid_local_balanced_kernel<true>
            <<<grid, 512, 0, a.stream>>>(
                a.cs_u, a.cs_sin, a.qd, a.kr, a.gt, a.segment_a, a.out,
                a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                a.N, a.total_tiles, a.total_segments,
                a.T_seq, a.H, a.NT);
    } else {
        k2_kda_csplit_bt64_out_hybrid_local_balanced_kernel<false>
            <<<grid, 512, 0, a.stream>>>(
                a.cs_u, a.cs_sin, a.qd, a.kr, a.gt, a.segment_a, a.out,
                nullptr, nullptr, nullptr, a.N, a.total_tiles,
                a.total_segments, a.T_seq, a.H, a.NT);
    }
}

// Segment-range form used by the N=1 P3/P4 stream pipeline.  `segment_begin`
// is global for packed input and sequence-local for dense input, matching the
// established P3 range ABI.  The policy currently admits only N=1, while the
// dense y geometry remains valid for a future equal-length N>1 experiment.
inline void launch_k6_hybrid_local_output_range(
        const Csplit64K6OutputLaunch& a, hipStream_t stream,
        int segment_begin, int segment_count) {
    if (segment_count <= 0)
        return;
    const dim3 grid = a.is_varlen
        ? dim3(segment_count, a.H)
        : dim3(segment_count, a.N * a.H);
    if (a.is_varlen) {
        k2_kda_csplit_bt64_out_hybrid_local_balanced_range_kernel<true>
            <<<grid, 512, 0, stream>>>(
                a.cs_u, a.cs_sin, a.qd, a.kr, a.gt, a.segment_a, a.out,
                a.cu_seqlens, a.tile_prefix, a.segment_prefix,
                a.N, a.total_tiles, a.total_segments,
                a.T_seq, a.H, a.NT, segment_begin, segment_count);
    } else {
        k2_kda_csplit_bt64_out_hybrid_local_balanced_range_kernel<false>
            <<<grid, 512, 0, stream>>>(
                a.cs_u, a.cs_sin, a.qd, a.kr, a.gt, a.segment_a, a.out,
                nullptr, nullptr, nullptr, a.N, a.total_tiles,
                a.total_segments, a.T_seq, a.H, a.NT,
                segment_begin, segment_count);
    }
}

}  // namespace flashkda_hip::gfx942::experimental
