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
    // Optional attention sink: one fp32 logit per query head, [nhead]. nullptr => no sink.
    // Symmetric D=64/128 kernels only.
    const void* sink_ptr = nullptr;

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
    // Pass exactly `sum(seqlen_q) / 256 + batch`. This is the grid extent, not a
    // free-standing block count: the kernel places a block by inverting seqstart_q, which
    // spaces the groups by that expression, so a shorter grid drops real blocks rather
    // than merely retiring surplus ones. Note it is NOT sum_g ceil(seqlen_q[g] / 256) --
    // that sum is smaller whenever the lengths are not multiples of 256, and three groups
    // of 255 would lose the third group's only block.
    int total_q_blocks = 0;

    // Group mode only. sum(seqlen_q), i.e. the packed Q row count (q.size(0)), from which
    // the launcher derives the flat extent. When > 0 it wins over total_q_blocks; both 0
    // keep the rectangular grid.
    int total_q = 0;

    // Group mode only. Paged KV: block_table [batch, stride_bt_b] int32 page ids, pages of
    // 128 rows; k/v point at the pool and stride_k_page / stride_v_page step one page. The
    // kernel then reads each KV tile through its own descriptor, so the pool is not bound by
    // the 32-bit extent; seqstart_k still carries each group's KV length.
    const int* block_table_ptr = nullptr;
    int stride_bt_b            = 0;
    long long stride_k_page    = 0;
    long long stride_v_page    = 0;

    // Paged only. Split-KV: num_splits > 1 cuts each Q block's KV range into that many
    // shares (grid z; page-aligned without a window). Their fp32 partials go to o_acc
    // [num_splits, total_q, H, 128] and lse_acc [num_splits, H, total_q], contiguous and
    // caller-allocated, and a combine kernel then writes o / lse. Needs total_q.
    int num_splits    = 1;
    void* o_acc_ptr   = nullptr;
    void* lse_acc_ptr = nullptr;

    std::string data_type = "bf16";
};

#ifdef FMHA_FWD_BF16_OPUS_LAUNCH_IMPL

#define FMHA_FWD_HD128_BF16_OPUS_IMPL
#include "fmha_fwd_hd128_bf16_opus.h"
#define FMHA_FWD_HD128_BF16_OPUS_SWA_IMPL
#include "fmha_fwd_hd128_bf16_opus_swa.h"
#define FMHA_FWD_HD192_V128_BF16_OPUS_IMPL
#include "fmha_fwd_hd192_v128_bf16_opus.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace fmha_fwd_bf16_opus_detail {

constexpr int kMaxSplits = 16;

// Split-KV merge: o[q, h] = sum_s exp(lse_s - lse) * o_acc[s, q, h], lse = log sum_s exp(lse_s).
// One 32-lane group per (q, h), four head-dim elements per lane. The module builds with
// -ffast-math, so a share with no keys (lse = -inf) or a -inf sink is recognized by its bit
// pattern, never by comparing against infinity, and is skipped. Rows outside
// [seqstart_q[0], seqstart_q[batch]) belong to no group; their partials were never written.
__global__ void __launch_bounds__(256)
    opus_splitkv_combine_kernel(const float* __restrict__ o_acc,
                                const float* __restrict__ lse_acc,
                                unsigned short* __restrict__ o,
                                float* __restrict__ lse,
                                const float* __restrict__ sink,
                                const int* __restrict__ seqstart_q,
                                int batch,
                                int num_splits,
                                int total_q,
                                int nhead,
                                int stride_o_n,
                                int stride_o_h,
                                int stride_lse_h)
{
    constexpr int D            = 128;
    constexpr unsigned NEG_INF = 0xff800000u;
    const int pair             = blockIdx.x * 8 + threadIdx.x / 32;
    if(pair >= total_q * nhead)
        return;
    const int qi = pair / nhead, h = pair % nhead, d = (threadIdx.x % 32) * 4;
    if(qi < seqstart_q[0] || qi >= seqstart_q[batch])
        return;
    const long long acc_split = static_cast<long long>(total_q) * nhead * D;
    const long long lse_split = static_cast<long long>(nhead) * total_q;

    float ls[kMaxSplits];
    float m  = 0.f;
    bool any = false;
#pragma unroll
    for(int s = 0; s < kMaxSplits; ++s)
    {
        ls[s] = s < num_splits ? lse_acc[s * lse_split + static_cast<long long>(h) * total_q + qi]
                               : 0.f;
        if(s < num_splits && __float_as_uint(ls[s]) != NEG_INF)
        {
            m   = any ? fmaxf(m, ls[s]) : ls[s];
            any = true;
        }
    }
    // A sink is one more term with no value, added once per row; none or -inf adds nothing.
    // The empty asm keeps -ffast-math from folding the -inf test into the fmaxf below.
    unsigned sink_bits = sink != nullptr ? __float_as_uint(sink[h]) : NEG_INF;
    asm volatile("" : "+v"(sink_bits));
    const bool has_sink = sink_bits != NEG_INF;
    const float sink_h  = __uint_as_float(sink_bits);
    if(has_sink)
    {
        m   = any ? fmaxf(m, sink_h) : sink_h;
        any = true;
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float sum  = 0.f;
#pragma unroll
    for(int s = 0; s < kMaxSplits; ++s)
    {
        if(s >= num_splits)
            break;
        if(__float_as_uint(ls[s]) == NEG_INF)
            continue;
        const float w  = __expf(ls[s] - m);
        const float4 x = *reinterpret_cast<const float4*>(
            o_acc + s * acc_split + (static_cast<long long>(qi) * nhead + h) * D + d);
        acc.x += w * x.x;
        acc.y += w * x.y;
        acc.z += w * x.z;
        acc.w += w * x.w;
        sum += w;
    }
    if(has_sink) // when the sink is m its term is exactly 1, which also covers +inf
        sum += sink_h >= m ? 1.f : __expf(sink_h - m);
    if(any)
    {
        const float inv = 1.f / sum;
        acc.x *= inv;
        acc.y *= inv;
        acc.z *= inv;
        acc.w *= inv;
    }
    auto bf16 = [](float f) -> unsigned int { // round to nearest even, NaN kept
        return __builtin_bit_cast(unsigned short, static_cast<__bf16>(f));
    };
    *reinterpret_cast<uint2*>(o + static_cast<long long>(qi) * stride_o_n +
                              static_cast<long long>(h) * stride_o_h + d) =
        make_uint2(bf16(acc.x) | (bf16(acc.y) << 16), bf16(acc.z) | (bf16(acc.w) << 16));
    if(lse != nullptr && threadIdx.x % 32 == 0)
        lse[static_cast<long long>(h) * stride_lse_h + qi] =
            any ? m + logf(sum) : __uint_as_float(NEG_INF);
}

// Split count for num_splits == 0. The D=128 kernel fits one workgroup per CU (its LDS), so
// the grid runs in ceil(workgroups / CUs) waves and a ragged last wave costs a full one.
// Pick the count with the shortest critical path, waves(S) / S, taking more splits only for
// a 3% gain (each adds combine traffic), with shares of at least 4096 keys.
inline int d128_auto_splits(int nhead,
                            int total_q,
                            int batch,
                            int max_seqlen_q,
                            int max_seqlen_k,
                            int window_size_left,
                            int num_cus)
{
    // Workgroups that hold rows: at most the flat grid's extent, at most batch full-length
    // groups; exact for one group or equal lengths. A window bounds each one's keys.
    const int wgs   = nhead * std::min(total_q / 256 + batch, batch * ceil_div(max_seqlen_q, 256));
    const int keys =
        window_size_left >= 0 ? std::min(max_seqlen_k, window_size_left + 1) : max_seqlen_k;
    const int max_s = std::min(kMaxSplits, std::max(1, keys / 4096));
    if(wgs <= 0 || num_cus <= 0)
        return 1;
    int best      = 1;
    double best_t = ceil_div(wgs, num_cus);
    for(int s = 2; s <= max_s; ++s)
    {
        const double t = static_cast<double>(ceil_div(wgs * s, num_cus)) / s;
        if(t < best_t * 0.97)
        {
            best   = s;
            best_t = t;
        }
    }
    return best;
}

inline float resolve_scale(float softmax_scale, int hdim_q)
{
    if(softmax_scale > 0.f)
        return softmax_scale;
    return 1.0f / std::sqrt(static_cast<float>(hdim_q));
}

// Fill the d128 kargs fields that do not depend on the traits. Returns false if the
// arguments are inconsistent.
inline bool fill_d128_kargs(opus_gqa_kargs& kargs, const fmha_fwd_bf16_opus_args& a,
                            bool& is_group, bool& is_local)
{
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
    kargs.ptr_sink     = a.sink_ptr;

    // Group / varlen. A null seqstart_q selects batch mode, where these stay null and the
    // GROUP_MODE branches compile out entirely.
    is_group = (a.seqstart_q_ptr != nullptr);
    if(is_group && (!a.seqstart_k_ptr || !a.seqstart_q_pad_ptr || !a.seqstart_k_pad_ptr))
        return false;
    kargs.ptr_seqstart_q     = is_group ? a.seqstart_q_ptr : nullptr;
    kargs.ptr_seqstart_k     = is_group ? a.seqstart_k_ptr : nullptr;
    kargs.ptr_seqstart_q_pad = is_group ? a.seqstart_q_pad_ptr : nullptr;
    kargs.ptr_seqstart_k_pad = is_group ? a.seqstart_k_pad_ptr : nullptr;

    // A window at least as wide as the longest KV bounds nothing; fold it to plain causal.
    is_local               = a.causal && a.window_size_left >= 0 && a.window_size_left < a.seqlen_k;
    kargs.window_size_left = is_local ? a.window_size_left : -1;

    kargs.ptr_block_table = a.block_table_ptr;
    kargs.stride_bt_b     = a.stride_bt_b;
    kargs.stride_k_page   = a.stride_k_page;
    kargs.stride_v_page   = a.stride_v_page;
    // Paged KV is group mode.
    if(a.block_table_ptr != nullptr && !is_group)
        return false;

    kargs.num_splits          = a.num_splits > 1 ? a.num_splits : 1;
    kargs.ptr_o_acc           = a.o_acc_ptr;
    kargs.ptr_lse_acc         = a.lse_acc_ptr;
    kargs.stride_oacc_h       = a.hdim_v;
    kargs.stride_oacc_n       = a.nhead * a.hdim_v;
    kargs.stride_oacc_split   = static_cast<long long>(a.total_q) * a.nhead * a.hdim_v;
    kargs.stride_lseacc_h     = a.total_q;
    kargs.stride_lseacc_split = static_cast<long long>(a.nhead) * a.total_q;
    if(a.num_splits > 1 &&
       (a.block_table_ptr == nullptr || a.total_q <= 0 || a.num_splits > kMaxSplits ||
        a.o_acc_ptr == nullptr || a.lse_acc_ptr == nullptr))
        return false;
    return true;
}

// Guards for launch_d128. `ok` distinguishes "refuse" from "nothing to do".
inline bool d128_preflight(const fmha_fwd_bf16_opus_args& a, bool& ok)
{
    ok = false;
    // A kv extent reaching 2^32 wraps the async-load offset and silently produces wrong
    // output, so refuse it instead.
    // Paged KV rebases every tile, so there the bound applies to one 64-row tile.
    const int kv_stride_n = (a.stride_k_n > a.stride_v_n ? a.stride_k_n : a.stride_v_n);
    const long long kv_rows = a.block_table_ptr != nullptr ? 64LL : a.seqlen_k;
    if(kv_rows * kv_stride_n * 2LL >= (1LL << 32))
        return false;
    // A paged tile's extent is rows * row stride, so rows must not overlap.
    if(a.block_table_ptr != nullptr && (a.stride_k_n < a.hdim_q || a.stride_v_n < a.hdim_q))
        return false;
    ok = true;
    // Nothing to launch, but not an error.
    return !(a.batch == 0 || a.seqlen_q == 0 || a.nhead == 0);
}

// D_QK = D_V in {64, 128} (symmetric). Covers batch mode, group / varlen and (D=128
// only) the sliding window; which one is selected follows from the arguments, see below.
inline bool launch_d128(const fmha_fwd_bf16_opus_args& a, hipStream_t stream)
{
    bool ok = false;
    if(!d128_preflight(a, ok))
        return ok;

    opus_gqa_kargs kargs{};
    bool is_group = false, is_local = false;
    if(!fill_d128_kargs(kargs, a, is_group, is_local))
        return false;

    auto launch = [&](auto traits_tag) {
        using Traits          = decltype(traits_tag);
        const int num_q_tiles = ceil_div(a.seqlen_q, Traits::Q_TILE_SIZE);
        const int num_q_blk   = ceil_div(num_q_tiles, Traits::NUM_WARPS);
        // Flat extent from the row count and this kernel's Q block; see
        // fmha_fwd_bf16_opus_args::total_q.
        constexpr int q_block = Traits::Q_TILE_SIZE * Traits::NUM_WARPS;
        kargs.total_q_blocks  = !is_group ? 0
                                : (a.total_q > 0 ? a.total_q / q_block + a.batch
                                                 : a.total_q_blocks);
        // Group mode has two groupings, see opus_gqa_kargs::total_q_blocks.
        //
        // Flat: y carries total_q / q_block + batch blocks, the spacing the kernel's
        // seqstart_q inversion defines. That is at most `batch` over the number that hold
        // rows -- not sum_g ceil(seqlen_q[g] / q_block), which the host cannot compute
        // without reading cu_seqlens back off the device. Rectangular: rotated axes
        // (head=x, group=y, Q-block=z) so the shorter groups' empty tail blocks
        // short-circuit together; the Q-block extent is the longest group. Batch mode
        // keeps q-block=y, batch=z.
        const bool flat = Traits::GROUP_MODE && kargs.total_q_blocks > 0;
        dim3 grid = flat ? dim3(a.nhead, kargs.total_q_blocks, Traits::SPLIT ? kargs.num_splits : 1)
                         : (Traits::GROUP_MODE ? dim3(a.nhead, a.batch, num_q_blk)
                                               : dim3(a.nhead, num_q_blk, a.batch));
        dim3 block(Traits::BLOCK_SIZE);
        // The sliding window has its own form of this kernel (fmha_fwd_hd128_bf16_opus_swa.h);
        // everything else stays on the shared one.
        if constexpr(Traits::LOCAL)
            gqa_d128_swa_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        else
            gqa_d128_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
        if constexpr(Traits::SPLIT)
        {
            const int pairs = a.total_q * a.nhead;
            opus_splitkv_combine_kernel<<<ceil_div(pairs, 8), 256, 0, stream>>>(
                static_cast<const float*>(a.o_acc_ptr),
                static_cast<const float*>(a.lse_acc_ptr),
                static_cast<unsigned short*>(a.o_ptr),
                static_cast<float*>(a.lse_ptr),
                static_cast<const float*>(a.sink_ptr),
                a.seqstart_q_ptr,
                a.batch,
                kargs.num_splits,
                a.total_q,
                a.nhead,
                a.stride_o_n,
                a.stride_o_h,
                a.stride_lse_h);
            HIP_CALL_LAUNCH(hipGetLastError());
        }
    };

    // D=64 shares the tile shape (Q=32, KV=64, 8 waves); only D_TILE_SIZE differs, and
    // every derived count in opus_gqa_traits follows from it. It has no LOCAL instance;
    // fmha_fwd_bf16_opus_launch refuses a window before we get here.
    if(a.hdim_q == 64)
    {
        if(is_group)
        {
            if(a.causal)
                launch(opus_gqa_traits<32, 64, 64, 8, true, true>{});
            else
                launch(opus_gqa_traits<32, 64, 64, 8, false, true>{});
        }
        else if(a.causal)
            launch(opus_gqa_traits<32, 64, 64, 8, true, false>{});
        else
            launch(opus_gqa_traits<32, 64, 64, 8, false, false>{});
        return true;
    }

    // causal x group x local (x paged). LOCAL implies CAUSAL, so there is no non-causal window.
    if(a.block_table_ptr != nullptr && kargs.num_splits > 1)
    {
        if(is_local)
            launch(opus_gqa_traits<32, 64, 128, 8, true, true, true, true, true>{});
        else if(a.causal)
            launch(opus_gqa_traits<32, 64, 128, 8, true, true, false, true, true>{});
        else
            launch(opus_gqa_traits<32, 64, 128, 8, false, true, false, true, true>{});
    }
    else if(a.block_table_ptr != nullptr)
    {
        if(is_local)
            launch(opus_gqa_traits<32, 64, 128, 8, true, true, true, true>{});
        else if(a.causal)
            launch(opus_gqa_traits<32, 64, 128, 8, true, true, false, true>{});
        else
            launch(opus_gqa_traits<32, 64, 128, 8, false, true, false, true>{});
    }
    else if(is_group)
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
// Returns false when the input is outside what the kernels cover: a GPU other than
// gfx950, a data type other than bf16, a (hdim_q, hdim_v) outside {(64,64), (128,128),
// (192,128)}, or a kv extent past the 32-bit async-load offset limit. Every head-dim
// kernel takes batch and group / varlen; only the 128/128 one takes a sliding window.
inline bool fmha_fwd_bf16_opus_launch(const fmha_fwd_bf16_opus_args& a, hipStream_t stream)
{
    static const std::string arch_id = get_gpu_arch();
    if((arch_id != "gfx950") || (a.data_type != "bf16"))
    {
        AITER_LOG_WARNING("unsupported condition in opus fwd!!! data_type: " << a.data_type);
        return false;
    }

    // A finite left window needs causal: LOCAL implies CAUSAL, so fill_d128_kargs would
    // otherwise fold the window away and launch full attention.
    if(a.window_size_left >= 0 && !a.causal)
    {
        AITER_LOG_WARNING("opus fwd: window_size_left requires causal");
        return false;
    }
    // The 192/128 forward has no windowed instance and never reads window_size_left, so
    // refuse a finite window rather than ignore it.
    if(a.hdim_q == 192 && a.hdim_v == 128 && a.window_size_left >= 0)
    {
        AITER_LOG_WARNING("opus fwd: hdim 192/128 has no sliding-window kernel");
        return false;
    }
    // Likewise D=64: the sliding-window kernel is D=128 only.
    if(a.hdim_q == 64 && a.window_size_left >= 0)
    {
        AITER_LOG_WARNING("opus fwd: hdim 64 has no sliding-window kernel");
        return false;
    }
    // Paged and split KV are D=128 only.
    if(a.hdim_q != 128 && (a.block_table_ptr != nullptr || a.num_splits > 1))
    {
        AITER_LOG_WARNING("opus fwd: paged and split KV need hdim 128");
        return false;
    }
    // Only the symmetric kernels and the split combine read the sink; refuse it where it
    // would be ignored.
    if(a.sink_ptr != nullptr && a.hdim_q != a.hdim_v)
    {
        AITER_LOG_WARNING("opus fwd: attention sink needs a symmetric head dim");
        return false;
    }

    if(a.hdim_q == a.hdim_v && (a.hdim_q == 64 || a.hdim_q == 128))
    {
        return fmha_fwd_bf16_opus_detail::launch_d128(a, stream);
    }
    if(a.hdim_q == 192 && a.hdim_v == 128)
    {
        return fmha_fwd_bf16_opus_detail::launch_d192_v128(a, stream);
    }
    return false;
}

#endif // FMHA_FWD_BF16_OPUS_LAUNCH_IMPL
