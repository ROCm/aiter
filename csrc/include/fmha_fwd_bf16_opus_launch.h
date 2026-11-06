// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free launch layer for the OPUS gfx950 bf16 flash-attention forward kernels.
// Shared by the torch entry point (`fmha_fwd_bf16_opus_fwd`, csrc/py_itfs_cu/
// fmha_fwd_bf16_opus_kernels.cu) and the standalone C++ benchmark (op_tests/cpp/mha/
// benchmark_mha_fwd.cpp), so the grid shape, the causal head/tail merge and the large-KV
// descriptor choice are decided in one place no matter which front end ran.
//
// Callers validate their own inputs and hand over raw pointers plus strides; this layer
// only refuses what the kernels themselves cannot express (data type, head-dim combination,
// 32-bit buffer extent).
//
// Single-header and IMPL-guarded like the kernel headers it pulls in: define
// FMHA_FWD_BF16_OPUS_LAUNCH_IMPL in the one translation unit that should own the device
// kernel instantiation. Including it without the macro yields just the argument struct.
#pragma once

#include "aiter_hip_common.h"
#include <hip/hip_runtime.h>
#include <string>

//   Batch mode: q [B, N, H, D_QK]  k [B, N, H_KV, D_QK]  v [B, N, H_KV, D_V]
//               o [B, N, H, D_V]   lse [B, H, N]
//   Group mode: q/k/v/o packed along a single sequence dim, each group located through the
//               seqstart arrays; lse [H, total_q]
// Strides are in elements and describe the caller's actual layout rather than being derived
// here, which is what lets a permuted or packed buffer run without a copy.
struct fmha_fwd_bf16_opus_args
{
    const void* q_ptr = nullptr;
    const void* k_ptr = nullptr;
    const void* v_ptr = nullptr;
    void* o_ptr       = nullptr;
    // nullptr => the kernel skips the log-sum-exp store entirely.
    void* lse_ptr = nullptr;

    // Group / varlen, int32, length batch+1. A null seqstart_q_ptr selects batch mode.
    // The *_pad arrays give the physical row offsets (equal to the non-pad arrays when
    // there is no kv padding); the non-pad ones give the real lengths used for masking.
    const int* seqstart_q_ptr     = nullptr;
    const int* seqstart_k_ptr     = nullptr;
    const int* seqstart_q_pad_ptr = nullptr;
    const int* seqstart_k_pad_ptr = nullptr;

    int batch   = 0; // number of sequences, i.e. number of groups in group mode
    int nhead   = 0;
    int nhead_k = 0;
    // Drive the grid and the kv traversal; the max over all groups in group mode.
    int seqlen_q = 0;
    int seqlen_k = 0;
    int hdim_q   = 0;
    int hdim_v   = 0;

    int stride_q_b = 0, stride_q_n = 0, stride_q_h = 0;
    int stride_k_b = 0, stride_k_n = 0, stride_k_h = 0;
    int stride_v_b = 0, stride_v_n = 0, stride_v_h = 0;
    int stride_o_b = 0, stride_o_n = 0, stride_o_h = 0;
    int stride_lse_b = 0, stride_lse_h = 0;

    // Applied to Q?K^T. Pass <= 0 to get the default 1/sqrt(hdim_q).
    float softmax_scale = 0.f;
    // Bottom-right aligned when seqlen_q != seqlen_k.
    bool causal = false;
    // Sliding-window attention. >= 0 bounds the window on the left: a query at
    // KV-coordinate j sees keys [j - window_size_left, j]. -1 (the default) is unbounded,
    // i.e. plain causal. Needs causal; right > 0 is not implemented.
    int window_size_left = -1;

    // Group mode only. Length of the flattened Q-block axis, so the grid no longer gives
    // every group the longest group's block count. 0 (the default) keeps the rectangular
    // grid.
    //
    // This is the grid extent, not a free-standing block count, so its value follows from
    // which of the three placement schemes below the caller selects -- each one packs the
    // flat index differently and so has its own extent:
    //
    //   block_map_ptr    any length >= the number of real blocks; see its note
    //   block_start_ptr  any length >= block_start[batch];        see its note
    //   neither          exactly sum(seqlen_q) / 256 + batch
    //
    // The last one is the scheme with nothing precomputed, and the only one whose extent
    // the caller does not choose: the kernel places a block by inverting seqstart_q, which
    // spaces the groups by that expression, so a grid shorter than it drops real blocks
    // rather than merely retiring surplus ones. Note it is NOT sum_g ceil(seqlen_q[g] /
    // 256) -- that sum is smaller whenever the lengths are not multiples of 256, and three
    // groups of 255 would lose the third group's only block.
    //
    // Worth supplying when the group lengths are uneven. The rectangle gives every group
    // the LONGEST group's block count, so one sequence at max_seqlen drags the whole
    // batch: at B=8 with seven ~8.4k sequences plus one at 16,354 it cost 73% more
    // workgroups and 53% more time for 11% more work. When the lengths are close the
    // rectangle wastes under 10% and this is not worth the per-block scalar lookup.
    int total_q_blocks = 0;
    // Device array of length batch+1 holding the cumulative Q-block counts (so
    // block_start[0] == 0 and block_start[batch] == sum_g ceil(seqlen_q[g] / 256)).
    //
    // Only worth building if the caller is producing it anyway: the lookup it replaces is
    // already a binary search over seqstart_q, and this one saves nothing but the divide.
    // A kernel to build it costs ~5 us of launch latency, which is a quarter of the
    // attention kernel on the smallest shapes.
    const int* block_start_ptr = nullptr;
    // Device array of packed (group << 16) | block pairs, one per flat index, which places
    // a block in a single scalar load. Takes precedence over block_start_ptr. Build it
    // block-major -- block 0 of every group, then block 1, and so on -- which keeps the
    // residency pattern the rectangular grid has and with it the L2 hit rate; a group-major
    // layout costs 8% (hit rate 54.9% -> 34.8%) even with the lookup removed.
    //
    // Entries past the last real block must be negative, so a grid sized from an upper
    // bound can retire its surplus. See opus_gqa_kargs::ptr_block_map.
    const int* block_map_ptr = nullptr;

    std::string data_type = "bf16";
};

#ifdef FMHA_FWD_BF16_OPUS_LAUNCH_IMPL

#define FMHA_FWD_HD128_BF16_OPUS_IMPL
#include "fmha_fwd_hd128_bf16_opus.h"
#define FMHA_FWD_HD192_V128_BF16_OPUS_IMPL
#include "fmha_fwd_hd192_v128_bf16_opus.h"

#include <cmath>
#include <type_traits>

namespace fmha_fwd_bf16_opus_detail {

inline float resolve_scale(float softmax_scale, int hdim_q)
{
    if(softmax_scale > 0.f)
        return softmax_scale;
    return 1.0f / std::sqrt(static_cast<float>(hdim_q));
}

// D_QK = D_V = 128 (symmetric), batch mode only.
inline bool launch_d128(const fmha_fwd_bf16_opus_args& a, hipStream_t stream)
{
    // A kv extent reaching 2^32 wraps the async-load offset and silently produces wrong
    // output, so refuse it instead.
    const int kv_stride_n = (a.stride_k_n > a.stride_v_n ? a.stride_k_n : a.stride_v_n);
    if(static_cast<long long>(a.seqlen_k) * kv_stride_n * 2LL >= (1LL << 32))
    {
        return false;
    }

    if(a.batch == 0 || a.seqlen_q == 0 || a.nhead == 0)
    {
        return true;
    }

    opus_gqa_kargs kargs{};
    kargs.ptr_q = a.q_ptr;
    kargs.ptr_k = a.k_ptr;
    kargs.ptr_v = a.v_ptr;
    kargs.ptr_o = a.o_ptr;
    kargs.B     = a.batch;
    kargs.N     = a.seqlen_q;
    kargs.N_KV  = a.seqlen_k;
    kargs.H     = a.nhead;
    kargs.H_KV  = a.nhead_k;
    kargs.D     = a.hdim_q;

    kargs.stride_q_b = a.stride_q_b;
    kargs.stride_q_n = a.stride_q_n;
    kargs.stride_q_h = a.stride_q_h;
    kargs.stride_o_b = a.stride_o_b;
    kargs.stride_o_n = a.stride_o_n;
    kargs.stride_o_h = a.stride_o_h;
    kargs.stride_k_b = a.stride_k_b;
    kargs.stride_k_n = a.stride_k_n;
    kargs.stride_k_h = a.stride_k_h;
    kargs.stride_v_b = a.stride_v_b;
    kargs.stride_v_n = a.stride_v_n;
    kargs.stride_v_h = a.stride_v_h;

    // The kernel applies scale * log2(e) to Q.
    kargs.softmax_scale = resolve_scale(a.softmax_scale, a.hdim_q);

    kargs.ptr_lse      = a.lse_ptr;
    kargs.stride_lse_b = a.stride_lse_b;
    kargs.stride_lse_h = a.stride_lse_h;

    // Group / varlen. A null seqstart_q selects batch mode, where these stay null and the
    // GROUP_MODE branches compile out entirely.
    const bool is_group = (a.seqstart_q_ptr != nullptr);
    if(is_group && (!a.seqstart_k_ptr || !a.seqstart_q_pad_ptr || !a.seqstart_k_pad_ptr))
        return false;
    kargs.ptr_seqstart_q     = is_group ? a.seqstart_q_ptr : nullptr;
    kargs.ptr_seqstart_k     = is_group ? a.seqstart_k_ptr : nullptr;
    kargs.ptr_seqstart_q_pad = is_group ? a.seqstart_q_pad_ptr : nullptr;
    kargs.ptr_seqstart_k_pad = is_group ? a.seqstart_k_pad_ptr : nullptr;
    kargs.total_q_blocks     = is_group ? a.total_q_blocks : 0;
    kargs.ptr_block_map      = is_group ? a.block_map_ptr : nullptr;
    kargs.ptr_block_start    = is_group ? a.block_start_ptr : nullptr;

    // A window wider than the longest KV is no window at all; fold it away so those shapes
    // take the cheaper non-LOCAL kernel.
    const bool is_local    = a.causal && a.window_size_left >= 0 && a.window_size_left < a.seqlen_k;
    kargs.window_size_left = is_local ? a.window_size_left : -1;

    auto launch = [&](auto traits_tag) {
        using Traits          = decltype(traits_tag);
        const int num_q_tiles = ceil_div(a.seqlen_q, Traits::Q_TILE_SIZE);
        const int num_q_blk   = ceil_div(num_q_tiles, Traits::NUM_WARPS);
        // Group mode has two groupings, see opus_gqa_kargs::total_q_blocks.
        //
        // Flat: the caller counted sum_g ceil(seqlen_q[g] / q_block) for us, so y carries
        // exactly the blocks that exist. Rectangular: rotated axes (head=x, group=y,
        // Q-block=z) so the shorter groups' empty tail blocks short-circuit together; the
        // Q-block extent is the longest group. Batch mode keeps q-block=y, batch=z.
        const bool flat = Traits::GROUP_MODE && kargs.total_q_blocks > 0;
        dim3 grid       = flat ? dim3(a.nhead, kargs.total_q_blocks, 1)
                               : (Traits::GROUP_MODE ? dim3(a.nhead, a.batch, num_q_blk)
                                                     : dim3(a.nhead, num_q_blk, a.batch));
        dim3 block(Traits::BLOCK_SIZE);
        gqa_d128_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    // causal x group x local. LOCAL implies CAUSAL, so there is no non-causal window.
    if(is_group)
    {
        if(a.causal)
        {
            if(is_local)
                launch(opus_gqa_traits<32, 64, 128, 8, true, true, true>{});
            else
                launch(opus_gqa_traits<32, 64, 128, 8, true, true, false>{});
        }
        else
            launch(opus_gqa_traits<32, 64, 128, 8, false, true, false>{});
    }
    else if(a.causal)
    {
        if(is_local)
            launch(opus_gqa_traits<32, 64, 128, 8, true, false, true>{});
        else
            launch(opus_gqa_traits<32, 64, 128, 8, true, false, false>{});
    }
    else
    {
        launch(opus_gqa_traits<32, 64, 128, 8, false, false, false>{});
    }
    return true;
}

// D_QK = 192 / D_V = 128 (asymmetric), batch + group / varlen.
inline bool launch_d192_v128(const fmha_fwd_bf16_opus_args& a, hipStream_t stream)
{
    constexpr int Q_TILE_SIZE = 32, NUM_WARPS = 8;
    constexpr int Q_BLOCK = Q_TILE_SIZE * NUM_WARPS; // 256

    const bool is_group = (a.seqstart_q_ptr != nullptr);
    if(is_group && (!a.seqstart_k_ptr || !a.seqstart_q_pad_ptr || !a.seqstart_k_pad_ptr))
    {
        return false;
    }
    const int num_q_blocks = ceil_div(a.seqlen_q, Q_BLOCK);
    if(a.batch == 0 || a.nhead == 0 || num_q_blocks == 0)
    {
        return true;
    }

    opus_gqa_d192_kargs kargs{};
    kargs.ptr_q = a.q_ptr;
    kargs.ptr_k = a.k_ptr;
    kargs.ptr_v = a.v_ptr;
    kargs.ptr_o = a.o_ptr;
    kargs.B     = a.batch;
    kargs.N     = a.seqlen_q;
    kargs.N_KV  = a.seqlen_k;
    kargs.H     = a.nhead;
    kargs.H_KV  = a.nhead_k;
    kargs.D_QK  = a.hdim_q;
    kargs.D_V   = a.hdim_v;

    kargs.stride_q_b = a.stride_q_b;
    kargs.stride_q_n = a.stride_q_n;
    kargs.stride_q_h = a.stride_q_h;
    kargs.stride_o_b = a.stride_o_b;
    kargs.stride_o_n = a.stride_o_n;
    kargs.stride_o_h = a.stride_o_h;
    kargs.stride_k_b = a.stride_k_b;
    kargs.stride_k_n = a.stride_k_n;
    kargs.stride_k_h = a.stride_k_h;
    kargs.stride_v_b = a.stride_v_b;
    kargs.stride_v_n = a.stride_v_n;
    kargs.stride_v_h = a.stride_v_h;

    // The kernel folds in log2(e) for its exp2 softmax.
    kargs.softmax_scale = resolve_scale(a.softmax_scale, a.hdim_q);

    kargs.ptr_lse      = a.lse_ptr;
    kargs.stride_lse_b = a.stride_lse_b;
    kargs.stride_lse_h = a.stride_lse_h;

    kargs.ptr_seqstart_q     = a.seqstart_q_ptr;
    kargs.ptr_seqstart_k     = a.seqstart_k_ptr;
    kargs.ptr_seqstart_q_pad = a.seqstart_q_pad_ptr;
    kargs.ptr_seqstart_k_pad = a.seqstart_k_pad_ptr;

    // Head/tail merge (causal load balance): the host is the single source of truth; the
    // kernel reads the OPT_MERGE_HEADTAIL bit and never recomputes it.
    const long long full_wgs = static_cast<long long>(num_q_blocks) * a.nhead * a.batch;
    const bool merge_ht      = a.causal && full_wgs >= static_cast<long long>(HEADTAIL_MIN_WG);
    kargs.opt                = merge_ht ? OPT_MERGE_HEADTAIL : 0;

    const int q_blk_dim = merge_ht ? ceil_div(num_q_blocks, 2) : num_q_blocks;
    // Group mode uses the rotated axis order of the production asm GROUP_MODE
    // (head=x, group=y, Q-block=z); batch mode keeps config A with q-block=x.
    dim3 grid = is_group ? dim3(a.nhead, a.batch, q_blk_dim) : dim3(q_blk_dim, a.nhead, a.batch);
    dim3 block(NUM_WARPS * 64);

    auto launch = [&](auto traits_tag) {
        using Traits = decltype(traits_tag);
        gqa_d192_v128_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    // Per-tile descriptor rebasing is only needed once a buffer's per-head extent stops
    // fitting the 32-bit num_records; under that the single-descriptor path is exact and
    // cheaper. Decided per buffer: K's rows are 1.5x wider than V's here, so K crosses the
    // limit first. seqlen_k is max_seqlen_k in group mode, so this bounds every group.
    const long long k_slice_bytes = static_cast<long long>(a.seqlen_k) * a.stride_k_n * 2LL;
    const long long v_slice_bytes = static_cast<long long>(a.seqlen_k) * a.stride_v_n * 2LL;
    const bool large_k            = k_slice_bytes >= (1LL << 32);
    const bool large_v            = v_slice_bytes >= (1LL << 32);

    auto launch_by_mode = [&](auto large_k_tag, auto large_v_tag) {
        constexpr bool LK = decltype(large_k_tag)::value;
        constexpr bool LV = decltype(large_v_tag)::value;
        if(is_group)
        {
            if(a.causal)
                launch(opus_gqa_d192_traits<32, 64, 8, true, true, LK, LV>{});
            else
                launch(opus_gqa_d192_traits<32, 64, 8, false, true, LK, LV>{});
        }
        else
        {
            if(a.causal)
                launch(opus_gqa_d192_traits<32, 64, 8, true, false, LK, LV>{});
            else
                launch(opus_gqa_d192_traits<32, 64, 8, false, false, LK, LV>{});
        }
    };

    // (small K, large V) needs strides that invert the usual 192/128 row widths; rebasing K
    // too is still correct there, so it folds into the all-large form rather than costing a
    // fourth pair of instantiations.
    if(!large_k && !large_v)
        launch_by_mode(std::false_type{}, std::false_type{});
    else if(large_k && !large_v)
        launch_by_mode(std::true_type{}, std::false_type{});
    else
        launch_by_mode(std::true_type{}, std::true_type{});

    return true;
}

} // namespace fmha_fwd_bf16_opus_detail

// Launches the opus forward kernel that matches the argument head dims on `stream`.
// Returns false when the input is outside what the kernels cover: a data type other than
// bf16, a (hdim_q, hdim_v) other than (128,128) or (192,128), group mode asked of the
// batch-only symmetric kernel, or (for the 128/128 kernel only) a kv extent past the
// 32-bit async-load offset limit.
inline bool fmha_fwd_bf16_opus_launch(const fmha_fwd_bf16_opus_args& a, hipStream_t stream)
{
    static const std::string arch_id = get_gpu_arch();
    if((arch_id != "gfx950") || (a.data_type != "bf16"))
    {
        AITER_LOG_WARNING("unsupported condition in opus fwd!!! data_type: " << a.data_type);
        return false;
    }

    const bool is_group = (a.seqstart_q_ptr != nullptr);

    if(a.hdim_q == 128 && a.hdim_v == 128)
    {
        // One kernel template covers batch, group / varlen and the sliding window.
        return fmha_fwd_bf16_opus_detail::launch_d128(a, stream);
    }
    if(a.hdim_q == 192 && a.hdim_v == 128)
    {
        return fmha_fwd_bf16_opus_detail::launch_d192_v128(a, stream);
    }
    return false;
}

#endif // FMHA_FWD_BF16_OPUS_LAUNCH_IMPL
