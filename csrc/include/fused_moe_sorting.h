// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused MoE sorting: router-sort + moe_buf zeroing in a single launch.
//
// The stock path costs three dependent launches -- grouped_topk, then Opus
// P0_v2, then P23 -- and three HBM round trips, which at decode shapes is
// almost all of the time: the mandatory traffic at M=64/E=257/topk=8 is
// ~830 KB, i.e. ~0.1 us at HBM speed against ~10.7 us measured. The op is
// latency- and launch-bound, not bandwidth-bound, so the win comes from
// collapsing launches rather than from moving bytes faster.
//
// What makes one launch possible is the mesh representation. Opus and FlyDSL
// build a dense [E][M] mesh at 1-4 bytes per cell, but only topk/E = 3.1% of
// the cells are ever set, which is why their single-workgroup path gives up
// above M=24. Here the mesh is one *bit* per cell:
//
//     mask[e][w]  bit t  <=>  token (w*64 + t) is routed to expert e
//
// popcount then yields the per-expert count in one instruction instead of an
// M-byte scan, a 64-bit word maps exactly onto a wave64, and LDS drops ~32x
// versus an i32 mesh -- which pulls "the whole sort fits in one workgroup"
// from M<=24 up past M=2048, covering the entire decode range.
//
// Layout contract (identical to CK/Opus/FlyDSL, see op_tests/test_moe_sorting.py):
//   * packed id      = (slot << 24) | token_id
//   * sentinel       = (topk << 24) | num_tokens   [num_tokens = static capacity]
//   * each expert padded up to a multiple of unit_size; empty expert = 0 blocks
//   * sorted_expert_ids holds the *local* expert index when expert_mask is set
//   * num_valid_ids[0] = padded slot total, [1] = number of real tokens
//   * moe_buf is zeroed by this same launch; a 0-element moe_buf is a no-op

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

// Developer knob: compile with -DFUSED_STOP_PHASE=n to make block 0 return
// right after phase n, so a kernel-trace attributes time per phase. Dead code
// at the default; outputs are garbage when set, so timing-only.
#ifndef FUSED_STOP_PHASE
#define FUSED_STOP_PHASE 99
#endif

// Waves per router block (RW) is a template parameter of the fused-router
// kernel, chosen by the host from the token count -- see max_fused_tokens'
// neighbour router_waves_for() in the .cu for the measured trade-off.

// Spacing of router blocks in blockIdx. Kept at 1: router blocks must be the
// lowest block ids so they are dispatched first. A stride of 8 was tried to
// put every router block on block 0's XCD (workgroups go round-robin over
// XCDs, each with its own L2) so the scratch hand-off would stay in one L2 --
// it lost badly, 10.24 vs 7.40 us at M=64, because the last router block then
// sits at blockIdx ~120 and starts microseconds after block 0. Dispatch order
// dominates cache locality here.
#ifndef FUSED_ROUTER_BLOCK_STRIDE
#define FUSED_ROUTER_BLOCK_STRIDE 1
#endif

#ifndef AITER_FUSED_SORT_LOG2E
#define AITER_FUSED_SORT_LOG2E 1.44269504088896340736 // log2(e), as in aiter
#endif

namespace aiter {

// ---------------------------------------------------------------------------
// Shared-memory plan. Kept in one place so host and device agree byte for byte.
//
// One presence bit per (expert, ASSIGNMENT), plus a per-word running popcount
// so an assignment's slot within its expert is a single LDS read rather than a
// walk over every word of the mask.
//
// Indexing by assignment (t * topk + j) rather than by token is what makes
// duplicates correct. A token may legitimately name the same expert in more
// than one of its top-k slots -- a real router never does, since it knocks each
// winner down before the next pass, but topk_ids is a caller-supplied tensor
// and CK/Opus/FlyDSL all count occurrences. One bit per (expert, token) cannot
// represent two, so those assignments silently collapsed into one.
//
// It also removes a subtlety rather than adding one: the reference orders an
// expert's rows by torch.where over [tokens, topk], which is exactly ascending
// assignment index, so walking the bits in order reproduces it directly.
struct FusedMoeSortingSmem
{
    int words_per_expert; // ceil(capacity / 64)
    size_t mask_u64;      // E * words_per_expert, one presence bit per (e, token)
    size_t wordpre_i32;   // E * words_per_expert, running popcount per word
    size_t weights_f32;   // M * topk, only when the router is fused in
    size_t ids_i32;       // M * topk, only when the router is fused in
    size_t scores_f32;    // (block / 64) * router_experts, only when fused
    bool stage_weights;
    size_t bytes;

    // router_experts > 0 selects the fused-router (F1-b) layout.
    //
    // Two things are deliberately absent.
    //
    // There are no slot bitplanes. An earlier layout spent three extra planes
    // encoding the top-k slot so the scatter would not have to re-read
    // topk_ids -- but once the scatter became one thread per assignment, the
    // slot is just `idx % topk`, free from the loop index. Dropping them cuts
    // the mask 4x and pushes the fuse threshold out correspondingly.
    //
    // The router's score scratch is sized by waves in flight, not by M: the
    // router runs one wave per token (as aiter's grouped_topk does), so it
    // needs NWAVES * E floats however many tokens there are.
    static FusedMoeSortingSmem make(int num_tokens,
                                    int num_experts,
                                    int topk,
                                    int block,
                                    bool stage_weights,
                                    int router_experts = 0)
    {
        FusedMoeSortingSmem s{};
        s.words_per_expert = (static_cast<size_t>(num_tokens) * topk + 63) / 64;
        // +1 word of padding per expert row. Without it the row stride is a
        // multiple of the 64-bank x 4-byte cycle on gfx950, so in the pass that
        // walks one expert per thread every lane lands on the same bank: bank
        // conflicts measured 91% of LDS accesses at M=256 (21694 of 23750) and
        // the kernel ran 14 us against 6 us for the rivals. The odd stride
        // costs E extra words and spreads the lanes across banks.
        s.mask_u64      = static_cast<size_t>(num_experts) * (s.words_per_expert + 1);
        s.wordpre_i32   = s.mask_u64; // same padded shape
        s.stage_weights = stage_weights;
        s.weights_f32   = stage_weights ? static_cast<size_t>(num_tokens) * topk : 0;
        s.ids_i32       = router_experts > 0 ? static_cast<size_t>(num_tokens) * topk : 0;
        s.scores_f32    = router_experts > 0 ? static_cast<size_t>(block / 64) * router_experts : 0;
        // mask | weights | scores | wordpre | ids | count[E] | cumsum[E] | skip[E] | scan[block]
        s.bytes = s.mask_u64 * sizeof(uint64_t) + (s.weights_f32 + s.scores_f32) * sizeof(float) +
                  (s.wordpre_i32 + s.ids_i32 + static_cast<size_t>(num_experts) * 3 + block) *
                      sizeof(int32_t);
        return s;
    }
};

// ---------------------------------------------------------------------------
// Block-wide exclusive scan over `n` ints already in LDS.
//
// n is E, which is 257 for GLM-5.2 -- deliberately *not* assumed to fit the
// block, since 257 > 256 breaks any "one element per thread" shortcut (and E
// can be 385 elsewhere). Each thread serially folds a contiguous chunk, the
// per-thread totals are scanned, then each thread rewrites its chunk.
//
// The cross-thread step runs on wave shuffles, not a Hillis-Steele pass over
// LDS: with BLOCK=1024 the latter costs 20 __syncthreads per scan, and at
// these shapes the scan is pure overhead sitting on the critical path of the
// one workgroup that owns the sort. Shuffles bring it down to three barriers.
template <int BLOCK>
__device__ inline int fused_block_exclusive_scan(int32_t* data, int n, int32_t* tmp)
{
    constexpr int NW = BLOCK / 64;
    const int tid    = threadIdx.x;
    const int lane   = tid & 63;
    const int wave   = tid >> 6;

    const int chunk = (n + BLOCK - 1) / BLOCK;
    const int beg   = tid * chunk;
    const int end   = min(beg + chunk, n);

    int local = 0;
    for(int i = beg; i < end; ++i)
        local += data[i];

    // Inclusive scan inside the wave.
    int v = local;
    for(int off = 1; off < 64; off <<= 1)
    {
        const int u = __shfl_up(v, off, 64);
        if(lane >= off)
            v += u;
    }
    if(lane == 63)
        tmp[wave] = v;
    __syncthreads();

    // Scan the NW wave totals (NW <= 16) inside wave 0.
    if(tid < NW)
    {
        int w = tmp[tid];
        for(int off = 1; off < NW; off <<= 1)
        {
            const int u = __shfl_up(w, off, 64);
            if(tid >= off)
                w += u;
        }
        tmp[tid] = w;
    }
    __syncthreads();

    const int wave_off = (wave == 0) ? 0 : tmp[wave - 1];
    const int total    = tmp[NW - 1];

    int run = wave_off + v - local; // exclusive prefix of this thread's chunk
    for(int i = beg; i < end; ++i)
    {
        const int x = data[i];
        data[i]     = run;
        run += x;
    }
    __syncthreads();
    return total;
}

// Inclusive prefix sum across a wave64 in DPP: four row_shr steps inside each
// 16-lane row, then row_bcast:15 into rows 1 and 3 and row_bcast:31 into rows
// 2 and 3. Six VALU ops with 1-2 wait states each, versus the __shfl_up loop
// this replaced, whose six steps compile to ds_bpermute -- an LDS round trip of
// ~120 cycles apiece, in a dependent chain. ATT put the largest barrier stall
// of the whole kernel (25K cycles across 16 waves) on the waves parked behind
// that chain.
__device__ inline int fused_wave_inclusive_scan_i32(int v)
{
#if defined(__GFX9__)
    // bound_ctrl=true: lanes whose source is out of range read 0; row_mask
    // limits which rows are written, the others keep `old` (= 0 here).
    v += __builtin_amdgcn_update_dpp(0, v, 0x111, 0xf, 0xf, true); // row_shr:1
    v += __builtin_amdgcn_update_dpp(0, v, 0x112, 0xf, 0xf, true); // row_shr:2
    v += __builtin_amdgcn_update_dpp(0, v, 0x114, 0xf, 0xf, true); // row_shr:4
    v += __builtin_amdgcn_update_dpp(0, v, 0x118, 0xf, 0xf, true); // row_shr:8
    v += __builtin_amdgcn_update_dpp(0, v, 0x142, 0xa, 0xf, true); // row_bcast:15 -> rows 1,3
    v += __builtin_amdgcn_update_dpp(0, v, 0x143, 0xc, 0xf, true); // row_bcast:31 -> rows 2,3
    return v;
#else
    const int lane = threadIdx.x & 63;
    for(int off = 1; off < 64; off <<= 1)
    {
        const int u = __shfl_up(v, off, 64);
        if(lane >= off)
            v += u;
    }
    return v;
#endif
}

// Exclusive scan of data[0..n) by ONE wave (the caller picks which), no block
// barriers inside. Each lane folds a contiguous chunk, the chunk totals go
// through a shuffle scan, then each lane rewrites its chunk. Returns the total
// in every lane of the wave.
//
// This replaced the block-wide scan (fused_block_exclusive_scan) for the sort's
// phase 3. That one spent three block barriers to scan 257 numbers -- with 16
// waves each barrier is ~135 ns -- and measured 0.96 us for the phase. n here
// is E, at most a few hundred, so one wave with a short serial chunk per lane
// is both cheaper and simpler; the only barrier left is the one the caller
// needs anyway before other waves read the result.
__device__ inline int fused_wave_exclusive_scan(int32_t* data, int n)
{
    const int lane  = threadIdx.x & 63;
    const int chunk = (n + 63) / 64;
    const int beg   = lane * chunk;
    const int end   = min(beg + chunk, n);

    int local = 0;
    for(int i = beg; i < end; ++i)
        local += data[i];

    const int incl  = fused_wave_inclusive_scan_i32(local);
    const int total = __builtin_amdgcn_readlane(incl, 63);

    int run = incl - local;
    for(int i = beg; i < end; ++i)
    {
        const int v = data[i];
        data[i]     = run;
        run += v;
    }
    return total;
}

// ---------------------------------------------------------------------------
// moe_buf zeroing, run by every block except block 0.
__device__ inline void fused_zero_moe_buf(void* moe_buf, size_t nbytes, int zero_blocks)
{
    if(moe_buf == nullptr || nbytes == 0 || zero_blocks <= 0)
        return;

    // blockIdx.x == 0 owns the sort; the zero blocks are numbered from 1.
    const size_t bid     = blockIdx.x - 1;
    const size_t stride  = static_cast<size_t>(blockDim.x) * zero_blocks;
    const size_t n_vec   = nbytes / sizeof(uint4);
    uint4* vec           = reinterpret_cast<uint4*>(moe_buf);
    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);

    for(size_t i = bid * blockDim.x + threadIdx.x; i < n_vec; i += stride)
        vec[i] = zero_vec;

    // Tail below one 16 B vector: a single thread finishes it off.
    const size_t tail_beg = n_vec * sizeof(uint4);
    if(bid == 0 && threadIdx.x == 0)
    {
        char* p = reinterpret_cast<char*>(moe_buf);
        for(size_t i = tail_beg; i < nbytes; ++i)
            p[i] = 0;
    }
}

// Router arguments, only meaningful when FUSE_TOPK. Bundled so the two kernel
// signatures stay readable.
struct FusedRouterArgs
{
    const void* gating;  // [tokens, router_experts]
    bool gating_is_fp32; // else bf16
    int router_experts;  // n_routed_experts -- NOT the sort's E
    bool need_renorm;
    bool is_softmax; // else sigmoid (GLM-5.2 uses sigmoid)
    float routed_scaling_factor;
    // Cross-block handshake for multi-block routing (see fused_moe_sorting_body).
    // Caller-owned int32, zero before the first launch; the kernel leaves it
    // zero. nullptr => route inside block 0 only.
    int32_t* sync;
    int router_blocks; // blocks 0..router_blocks-1 each route NW tokens
};

// Wave reductions that mirror aiter's `wave_reduce` exactly.
//
// Not a symmetric xor butterfly: on GFX9 the last two stages are row_bcast:15
// and row_bcast:31, which combine asymmetrically, and `wave_reduce` applies
// reduce_op(remote, local) so a tie keeps `local`. Both details are load
// bearing. Float addition is not associative, so a different tree shape
// changes the softmax denominator; and arg-max ties are decided purely by
// combine order -- with bf16 logits, 20 of 64 token rows carried duplicate
// scores inside the top-8 in measurement, so an xor butterfly picked
// different experts and C1 failed. dpp_ctrl values and the
// row_mask/bank_mask/bound_ctrl defaults are taken from hip_reduce.h.
template <int CTRL>
__device__ inline int fused_dpp_mov(int v)
{ return __builtin_amdgcn_mov_dpp(v, CTRL, 0xf, 0xf, false); }

template <int CTRL>
__device__ inline void fused_argmax_step(float& v, int& i)
{
    const float rv = __builtin_bit_cast(float, fused_dpp_mov<CTRL>(__builtin_bit_cast(int, v)));
    const int ri   = fused_dpp_mov<CTRL>(i);
    if(rv > v) // reduce_op(remote, local) -> strict >, ties keep local
    {
        v = rv;
        i = ri;
    }
}

// Arg-max over the wave. Afterwards the answer lives in lane 63, which is
// where aiter reads it from (readlane(WARP_SIZE - 1)).
__device__ inline void fused_wave_argmax(float& v, int& i)
{
#if defined(__GFX9__)
    fused_argmax_step<0xb1>(v, i);  // quad_perm:[1,0,3,2]
    fused_argmax_step<0x4e>(v, i);  // quad_perm:[2,3,0,1]
    fused_argmax_step<0x141>(v, i); // row_half_mirror
    fused_argmax_step<0x140>(v, i); // row_mirror
    fused_argmax_step<0x142>(v, i); // row_bcast:15
    fused_argmax_step<0x143>(v, i); // row_bcast:31
    v = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
    i = __builtin_amdgcn_readlane(i, 63);
#else
    for(int off = 1; off < 64; off <<= 1)
    {
        const float ov = __shfl_xor(v, off, 64);
        const int oi   = __shfl_xor(i, off, 64);
        if(ov > v)
        {
            v = ov;
            i = oi;
        }
    }
#endif
}

template <int CTRL>
__device__ inline float fused_dpp_movf(float v)
{ return __builtin_bit_cast(float, fused_dpp_mov<CTRL>(__builtin_bit_cast(int, v))); }

// Same tree for the plain float reductions the softmax path needs. aiter calls
// these with threadBroadcast = true, so the result is shuffled out of lane 63
// to every lane.
template <bool IS_MAX>
__device__ inline float fused_wave_reduce_f32(float v)
{
#if defined(__GFX9__)
    constexpr int ctrls[6] = {0xb1, 0x4e, 0x141, 0x140, 0x142, 0x143};
    (void)ctrls;
    auto comb = [](float remote, float local) {
        return IS_MAX ? (remote > local ? remote : local) : (remote + local);
    };
    v = comb(fused_dpp_movf<0xb1>(v), v);
    v = comb(fused_dpp_movf<0x4e>(v), v);
    v = comb(fused_dpp_movf<0x141>(v), v);
    v = comb(fused_dpp_movf<0x140>(v), v);
    v = comb(fused_dpp_movf<0x142>(v), v);
    v = comb(fused_dpp_movf<0x143>(v), v);
    v = __shfl(v, 63, 64);
#else
    for(int off = 1; off < 64; off <<= 1)
    {
        const float o = __shfl_xor(v, off, 64);
        v             = IS_MAX ? (o > v ? o : v) : (o + v);
    }
#endif
    return v;
}

// ---------------------------------------------------------------------------
// Fused router with scores held in REGISTERS (sigmoid scoring only).
//
// The LDS-resident router below cost ~5 us per token per wave at M=64 -- 19.6 us
// for the whole router phase against 4.2 us for the entire sort -- because each
// of its topk arg-max passes goes through LDS (reads, the knock-out write, a
// wave barrier) and a wave handles its tokens back to back, so that latency
// chain is paid ~4 times over. Nothing in the algorithm needs LDS: with vec 4,
// lane L owns experts 4L..4L+3, which is NREG = 4 floats. So the scores live in
// registers, the knock-out is an unrolled compare, and each pass is a handful
// of VALU ops plus the same DPP tree. The expert->lane mapping is exactly
// aiter's VEC=4 vectorised scan, so tie-breaking -- and therefore bit-exactness
// -- is unchanged.
//
// Row loads are issued for a whole batch of TB tokens before any is scored, so
// a wave pays one HBM latency per batch instead of one per token.
//
// Softmax is not handled here: aiter scores the softmax path with an
// element-stride (lane L sums experts L, L+64, ...) whose float-sum order this
// mapping cannot reproduce; the host routes softmax to the LDS router.
// Routes tokens t = t_begin + wave, t_begin + wave + stride, ... while < m_eff,
// TB of them per batch. `sids`/`sweight` may be LDS or global (generic).
// Multi-block routing passes stride = capacity so each wave does one token.
template <int BLOCK, int NREG, int TB = 4>
__device__ inline void fused_router_topk_reg(const FusedRouterArgs& r,
                                             int t_begin,
                                             int stride,
                                             int m_eff,
                                             int topk,
                                             int32_t* sids,
                                             float* sweight)
{
    constexpr int NW  = BLOCK / 64;
    constexpr int VEC = 4;
    constexpr int R   = NREG / VEC; // vectors per lane
    static_assert(NREG % VEC == 0, "NREG must be a multiple of the vector width");

    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int ER   = r.router_experts;
    const int NVEC = ER / VEC;

    for(int t0 = t_begin + wave; t0 < m_eff; t0 += stride * TB)
    {
        // Issue every row load of the batch up front.
        float g[TB][NREG];
#pragma unroll
        for(int b = 0; b < TB; ++b)
        {
            const int t = t0 + b * stride;
#pragma unroll
            for(int rr = 0; rr < R; ++rr)
            {
                const int v     = lane + 64 * rr;
                const bool ok   = (t < m_eff) && (v < NVEC);
                const size_t of = static_cast<size_t>(t) * ER + static_cast<size_t>(v) * VEC;
                if(ok)
                {
                    if(r.gating_is_fp32)
                    {
                        const float4 x = *reinterpret_cast<const float4*>(
                            static_cast<const float*>(r.gating) + of);
                        g[b][rr * VEC + 0] = x.x;
                        g[b][rr * VEC + 1] = x.y;
                        g[b][rr * VEC + 2] = x.z;
                        g[b][rr * VEC + 3] = x.w;
                    }
                    else
                    {
                        const uint2 x = *reinterpret_cast<const uint2*>(
                            static_cast<const uint16_t*>(r.gating) + of);
                        g[b][rr * VEC + 0] = __uint_as_float(x.x << 16);
                        g[b][rr * VEC + 1] = __uint_as_float(x.x & 0xffff0000u);
                        g[b][rr * VEC + 2] = __uint_as_float(x.y << 16);
                        g[b][rr * VEC + 3] = __uint_as_float(x.y & 0xffff0000u);
                    }
                }
                else
                {
#pragma unroll
                    for(int i = 0; i < VEC; ++i)
                        g[b][rr * VEC + i] = -INFINITY; // never a candidate
                }
            }
        }

#pragma unroll
        for(int b = 0; b < TB; ++b)
        {
            const int t = t0 + b * stride; // wave-uniform, so the DPP below is full-wave
            if(t >= m_eff)
                break;

            float s[NREG];
#pragma unroll
            for(int j = 0; j < NREG; ++j)
            {
                const float x = g[b][j];
                s[j] = (x == -INFINITY)
                           ? -INFINITY
                           : __builtin_amdgcn_rcpf(
                                 1.0f + exp2f(static_cast<float>(-AITER_FUSED_SORT_LOG2E * x)));
            }

            float sum  = 0.0f;
            int my_id  = 0;
            float my_w = 0.0f;
            for(int k = 0; k < topk; ++k)
            {
                float mv = -INFINITY;
                int mi   = k;
#pragma unroll
                for(int rr = 0; rr < R; ++rr)
#pragma unroll
                    for(int i = 0; i < VEC; ++i)
                    {
                        const float x = s[rr * VEC + i];
                        if(x > mv)
                        {
                            mv = x;
                            mi = (lane + 64 * rr) * VEC + i;
                        }
                    }
                fused_wave_argmax(mv, mi);
                // Knock the winner out wherever it lives -- an unrolled compare, so
                // no dynamic register indexing (which would spill to scratch).
#pragma unroll
                for(int rr = 0; rr < R; ++rr)
#pragma unroll
                    for(int i = 0; i < VEC; ++i)
                        if((lane + 64 * rr) * VEC + i == mi)
                            s[rr * VEC + i] = -INFINITY;
                if(lane == k)
                {
                    my_id = mi;
                    my_w  = mv;
                }
                sum += mv;
            }

            const float scale =
                r.need_renorm ? r.routed_scaling_factor / sum : r.routed_scaling_factor;
            if(lane < topk)
            {
                sids[static_cast<size_t>(t) * topk + lane]    = my_id;
                sweight[static_cast<size_t>(t) * topk + lane] = my_w * scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused router: scoring + top-k, one wave per token.
//
// Mirrors aiter's grouped_topk_kernel for the n_group == 1 case, which is what
// GLM-5.2 uses (n_group=1, topk_group=1, sigmoid, norm_topk_prob, scale 2.5),
// so the group-selection stage degenerates away entirely. aiter launches that
// kernel as dim3 grid(num_tokens), dim3 block(warp_size) -- literally one wave
// per token -- so giving each wave here one token reproduces its arithmetic
// with the same reduction shape.
//
// Bit-exactness notes, since C1 admits no tolerance:
//   * the sigmoid must use the same rcp/exp2 intrinsics, and C_LOG2E is a
//     *double* literal in aiter, so the multiply happens in double and narrows
//     to float only at the exp2f call. Reproduced exactly.
//   * top-k is k sequential arg-max passes, each knocking its winner down to
//     -INFINITY. With distinct scores the winner is unique, so the reduction
//     tree cannot change the answer; only exact ties could, and sigmoid of
//     distinct logits does not produce them.
//   * the renormalisation sum accumulates in pass order k = 0..topk-1 over the
//     broadcast maxima, identically in every lane, so it is order-stable.

template <int BLOCK>
__device__ inline void fused_router_topk(
    const FusedRouterArgs& r, int m_eff, int topk, float* sscores, int32_t* sids, float* sweight)
{
    constexpr int NW = BLOCK / 64;
    const int lane   = threadIdx.x & 63;
    const int wave   = threadIdx.x >> 6;
    const int ER     = r.router_experts;

    float* sc = sscores + static_cast<size_t>(wave) * ER;

    for(int t = wave; t < m_eff; t += NW)
    {
        // Score the row.
        const size_t row = static_cast<size_t>(t) * ER;
        if(r.is_softmax)
        {
            float mx = -INFINITY;
            for(int e = lane; e < ER; e += 64)
            {
                const float g =
                    r.gating_is_fp32
                        ? static_cast<const float*>(r.gating)[row + e]
                        : __bfloat162float(static_cast<const __hip_bfloat16*>(r.gating)[row + e]);
                sc[e] = g;
                mx    = g > mx ? g : mx;
            }
            mx        = fused_wave_reduce_f32<true>(mx);
            float sum = 0.0f;
            for(int e = lane; e < ER; e += 64)
            {
                const float v = expf(sc[e] - mx);
                sc[e]         = v;
                sum += v;
            }
            sum = fused_wave_reduce_f32<false>(sum);
            for(int e = lane; e < ER; e += 64)
                sc[e] /= sum;
        }
        else
        {
            for(int e = lane; e < ER; e += 64)
            {
                const float g =
                    r.gating_is_fp32
                        ? static_cast<const float*>(r.gating)[row + e]
                        : __bfloat162float(static_cast<const __hip_bfloat16*>(r.gating)[row + e]);
                sc[e] = __builtin_amdgcn_rcpf(
                    1.0f + exp2f(static_cast<float>(-AITER_FUSED_SORT_LOG2E * g)));
            }
        }
        __builtin_amdgcn_wave_barrier();

        // k sequential arg-max passes.
        //
        // The scan is vectorised the way aiter's is -- it strides by *vector*,
        // not by element, so with vec 4 lane L owns experts 4L..4L+3. That
        // mapping is not cosmetic: ties are broken by which lane holds the
        // value, and bf16 logits tie often (8 mantissa bits over 256 experts),
        // so a plain element stride silently picks different experts.
        const int VEC  = (ER % 4 == 0) ? 4 : ((ER % 4 == 2) ? 2 : 1);
        const int NVEC = ER / VEC;

        float sum  = 0.0f;
        int my_id  = 0;
        float my_w = 0.0f;
        for(int k = 0; k < topk; ++k)
        {
            float mv = -INFINITY;
            int mi   = k;
            for(int v = lane; v < NVEC; v += 64)
            {
                for(int i = 0; i < VEC; ++i)
                {
                    const int e   = v * VEC + i;
                    const float x = sc[e];
                    if(x > mv)
                    {
                        mv = x;
                        mi = e;
                    }
                }
            }
            fused_wave_argmax(mv, mi);
            sc[mi] = -INFINITY; // every lane writes the same value to the same slot
            __builtin_amdgcn_wave_barrier();
            if(lane == k)
            {
                my_id = mi;
                my_w  = mv;
            }
            sum += mv;
        }

        const float scale = r.need_renorm ? r.routed_scaling_factor / sum : r.routed_scaling_factor;
        if(lane < topk)
        {
            sids[static_cast<size_t>(t) * topk + lane]    = my_id;
            sweight[static_cast<size_t>(t) * topk + lane] = my_w * scale;
        }
        __builtin_amdgcn_wave_barrier();
    }
}

// ---------------------------------------------------------------------------
// The fused kernel.
//
// Block 0 performs the entire sort out of LDS; blocks 1.. zero moe_buf in
// parallel with it. Phases:
//   0  clear the bitmask
//   1  one thread per assignment, atomicOr its bit (plus slot planes)
//   2  count[e] = sum of popcounts; pad up to unit_size; masked expert -> 0
//   3  two block-wide exclusive scans: padded offsets, and masked-expert count
//      (the latter converts a global expert id into the local one EP expects)
//   4  one wave per expert: emit block ids, scatter packed ids + weights, then
//      write the sentinel over the padding tail only
template <int BLOCK, bool STAGE_W, bool FUSE_TOPK, int RNREG = 0, int RW = 4>
__device__ inline void
fused_moe_sorting_body(const int32_t* __restrict__ topk_ids,
                       const float* __restrict__ topk_weights,
                       const int32_t* __restrict__ expert_mask,      // nullable
                       const int32_t* __restrict__ num_local_tokens, // nullable, device-side
                       int32_t* __restrict__ sorted_ids,
                       float* __restrict__ sorted_weights,
                       int32_t* __restrict__ sorted_expert_ids,
                       int32_t* __restrict__ num_valid_ids,
                       void* __restrict__ moe_buf,
                       int num_tokens, // static capacity = topk_ids.shape[0]
                       int topk,
                       int num_experts,
                       int unit_size,
                       size_t moe_buf_bytes,
                       int zero_blocks,
                       FusedRouterArgs router)
{
    // ---- multi-block routing (FUSE_TOPK with router.sync) --------------------
    //
    // The router's per-token chain is ~3.9 us and cannot be shortened much
    // inside one block, but it is embarrassingly parallel across tokens: aiter
    // hides it by running one block per token on 64 CUs. So blocks
    // 0..router_blocks-1 each route NW tokens, one per wave, and hand the
    // result to block 0 through the caller's output buffers -- sorted_ids and
    // sorted_weights are >= M*topk long and are not written until phase 4, so
    // their prefix is free scratch until then -- plus a release/acquire
    // handshake on `router.sync`. Blocks 1.. then continue into moe_buf
    // zeroing. All blocks are co-resident (grid <= CU count), so block 0's
    // spin cannot deadlock.
    if constexpr(FUSE_TOPK)
    {
        constexpr int RS = FUSED_ROUTER_BLOCK_STRIDE;
        const int rblk   = static_cast<int>(blockIdx.x) / RS; // router-block index
        if(router.sync != nullptr && blockIdx.x != 0 && (blockIdx.x % RS) == 0 &&
           rblk < router.router_blocks)
        {
            if constexpr(RNREG > 0)
                if((threadIdx.x >> 6) < RW)
                    fused_router_topk_reg<BLOCK, RNREG, 1>(
                        router,
                        rblk * RW,
                        num_tokens, // stride: one token per wave
                        min(num_local_tokens ? num_local_tokens[0] : num_tokens, num_tokens),
                        topk,
                        sorted_ids,
                        sorted_weights);
            __syncthreads();
            if(threadIdx.x == 0)
            {
                __threadfence();
                __hip_atomic_fetch_add(router.sync, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }

    if(blockIdx.x != 0)
    {
        fused_zero_moe_buf(moe_buf, moe_buf_bytes, zero_blocks);
        return;
    }

    extern __shared__ char fused_smem[];

    // Sized from the static capacity, so the launch is identical under a
    // CUDA-graph replay whatever num_local_tokens turns out to be.
    const int W = static_cast<int>((static_cast<size_t>(num_tokens) * topk + 63) / 64);
    // Padded row stride -- see FusedMoeSortingSmem::make. W stays the logical
    // word count; WP is what the indexing uses.
    const int WP = W + 1;

    uint64_t* smask      = reinterpret_cast<uint64_t*>(fused_smem);
    const size_t pstride = static_cast<size_t>(num_experts) * WP; // padded mask/prefix
    // Staging topk_weights in LDS is what makes the scatter fast. Read from
    // HBM it is a dependent, fully exposed gather: one workgroup has far too
    // few waves to hide ~700 ns of latency across ~220 active experts, which
    // measured 42 us at M=64 -- 4x slower than the two-kernel Opus path it
    // replaces. Phase 1 already streams every assignment, so it picks the
    // weight up on the way past for free.
    float* sweight = reinterpret_cast<float*>(smask + pstride);
    // Per-wave score scratch for the fused router: one wave owns one token, so
    // this is NWAVES * router_experts, independent of M.
    float* sscores = sweight + (STAGE_W ? static_cast<size_t>(num_tokens) * topk : 0);
    // Running popcount of each expert's earlier words, so the scatter can turn
    // a token into its slot with one LDS read instead of walking every word.
    int32_t* swordpre = reinterpret_cast<int32_t*>(
        sscores + (FUSE_TOPK ? static_cast<size_t>(BLOCK / 64) * router.router_experts : 0));
    int32_t* sids   = swordpre + pstride;
    int32_t* scount = sids + (FUSE_TOPK ? static_cast<size_t>(num_tokens) * topk : 0);
    int32_t* scum   = scount + num_experts;
    int32_t* sskip  = scum + num_experts;
    int32_t* stmp   = sskip + num_experts;

    const int tid = threadIdx.x;

    // Real token count. Read on-device: host-side decisions use the static
    // capacity only, so the same launch config replays under a CUDA graph.
    int m_eff = num_tokens;
    if(num_local_tokens != nullptr)
        m_eff = min(num_local_tokens[0], num_tokens);
    const int total_assign = m_eff * topk;

    // ---- phase 0: clear the bitmask -------------------------------------
    // With multi-block routing block 0's non-router waves do this while the
    // router waves route (below); the barrier after routing covers both.
    const bool clear_overlaps_routing = FUSE_TOPK && (router.sync != nullptr) && (RNREG > 0);
    if(!clear_overlaps_routing)
    {
        for(size_t i = tid; i < pstride; i += BLOCK)
            smask[i] = 0ull;
        __syncthreads();
    }

    if(FUSED_STOP_PHASE <= 0 && num_tokens >= 0) // runtime conjunct defeats DCE of earlier phases
        return;
    // ---- phase 1: populate the bitmask ------------------------------------
    if constexpr(FUSE_TOPK)
    {
        // F1-b: run the router here so topk_ids/topk_weights never reach HBM.
        if constexpr(RNREG > 0)
            if(router.sync != nullptr)
            {
                // Block 0 routes its own RW tokens straight into LDS ...
                if((threadIdx.x >> 6) < RW)
                    fused_router_topk_reg<BLOCK, RNREG, 1>(
                        router, 0, num_tokens, min(m_eff, RW), topk, sids, sweight);
                else // ... while the other waves clear the mask (phase 0, overlapped).
                    for(size_t i = tid - RW * 64; i < pstride; i += BLOCK - RW * 64)
                        smask[i] = 0ull;
                // ... then waits for the other router blocks and pulls theirs in.
                if(threadIdx.x == 0)
                {
                    const int want = router.router_blocks - 1;
                    while(__hip_atomic_load(
                              router.sync, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < want)
                        __builtin_amdgcn_s_sleep(1);
                    __threadfence();
                    *router.sync = 0; // leave it ready for the next launch
                }
                __syncthreads();
                for(int i = RW * topk + tid; i < total_assign; i += BLOCK)
                {
                    sids[i]    = sorted_ids[i];
                    sweight[i] = sorted_weights[i];
                }
            }
            else
                fused_router_topk_reg<BLOCK, RNREG>(
                    router, 0, BLOCK / 64, m_eff, topk, sids, sweight);
        else
            fused_router_topk<BLOCK>(router, m_eff, topk, sscores, sids, sweight);
        __syncthreads();
    }

    for(int idx = tid; idx < total_assign; idx += BLOCK)
    {
        const int e = FUSE_TOPK ? sids[idx] : topk_ids[idx];
        if constexpr(STAGE_W)
        {
            if constexpr(!FUSE_TOPK)
                sweight[idx] = topk_weights[idx];
        }
        if(e < 0 || e >= num_experts)
            continue;
        const int w        = idx >> 6;
        const uint64_t bit = 1ull << (idx & 63);
        atomicOr(&smask[static_cast<size_t>(e) * WP + w], bit);
    }
    __syncthreads();

    if(FUSED_STOP_PHASE <= 1 && num_tokens >= 0)
        return;
    // ---- phase 2: per-expert counts and padded sizes ---------------------
    // One thread per expert, walking its W words.
    //
    // A wave-per-expert variant was tried, to parallelise the word walk after
    // indexing by assignment pushed W from ceil(M/64) to ceil(M*topk/64). It
    // was worse: it puts only BLOCK/64 = 16 experts in flight against 1024
    // here, and at E=257 that costs ~17 serial rounds where this costs one.
    // Measured at M=1 it turned 3.1 us into 10.8 us. The word walk is only
    // worth parallelising once W exceeds the round count it would add, which
    // does not happen inside the supported token range (W <= 32 at M=256).
    for(int e = tid; e < num_experts; e += BLOCK)
    {
        int c = 0;
        for(int w = 0; w < W; ++w)
        {
            swordpre[static_cast<size_t>(e) * WP + w] = c;
            c += __popcll(smask[static_cast<size_t>(e) * WP + w]);
        }
        const bool off = (expert_mask != nullptr) && (expert_mask[e] == 0);
        scount[e]      = off ? 0 : c;
        scum[e]        = off ? 0 : ((c + unit_size - 1) / unit_size) * unit_size;
        sskip[e]       = off ? 1 : 0;
    }
    __syncthreads();

    if(FUSED_STOP_PHASE <= 2 && num_tokens >= 0)
        return;
    // ---- phase 3: scans (wave 0 only) ------------------------------------
    if(tid < 64)
    {
        const int total_padded_w = fused_wave_exclusive_scan(scum, num_experts);
        // Without an expert_mask no expert is ever skipped, so the local expert
        // id is just the global one and the second scan is dead work.
        if(expert_mask != nullptr)
            (void)fused_wave_exclusive_scan(sskip, num_experts);
        if(tid == 0)
        {
            stmp[0]          = total_padded_w; // broadcast to the block
            num_valid_ids[0] = total_padded_w;
            num_valid_ids[1] = m_eff;
        }
    }
    __syncthreads();
    const int total_padded = stmp[0];

    if(FUSED_STOP_PHASE <= 3 && num_tokens >= 0)
        return;
    // ---- phase 4: emit ----------------------------------------------------
    //
    // Deliberately NOT a loop over experts. That shape serialises ~222 active
    // experts onto however many waves the single workgroup has, and each
    // iteration re-enters LDS for count/base/mask before it can store, so the
    // dependent LDS latency -- not the store traffic -- sets the pace; it
    // measured ~10 us at M=64. All three pieces below are instead flat and
    // fully parallel across the whole block.
    const int sentinel = (topk << 24) | num_tokens;

    // 4a. Sentinel across the live range, coalesced. This fills the real slots
    //     too; 4c overwrites them after a barrier. Note the range is
    //     [0, total_padded) -- the *valid* output -- not the static capacity
    //     that FlyDSL's one-shot path pre-fills, which is far larger.
    // total_padded is a multiple of unit_size; when that is a multiple of 4 the
    // whole live range is 16 B-aligned (torch buffers are), so four slots go
    // per store. 7100 slots at M=64 drop from 7 rounds to 2.
    if((total_padded & 3) == 0 && (reinterpret_cast<uintptr_t>(sorted_ids) & 15) == 0)
    {
        uint4* v4      = reinterpret_cast<uint4*>(sorted_ids);
        const uint4 sv = make_uint4(sentinel, sentinel, sentinel, sentinel);
        for(int k = tid; k < (total_padded >> 2); k += BLOCK)
            v4[k] = sv;
    }
    else
    {
        for(int k = tid; k < total_padded; k += BLOCK)
            sorted_ids[k] = sentinel;
    }

    // 4b. Block ids: one thread per expert, so E=257 is a single pass.
    for(int e = tid; e < num_experts; e += BLOCK)
    {
        const int c = scount[e];
        if(c == 0)
            continue;
        const int nblk    = (c + unit_size - 1) / unit_size;
        const int bofs    = scum[e] / unit_size;
        const int local_e = (expert_mask != nullptr) ? (e - sskip[e]) : e;
        for(int b = 0; b < nblk; ++b)
            sorted_expert_ids[bofs + b] = local_e;
    }

    __syncthreads(); // 4a must land before 4c overwrites the real slots
    if(FUSED_STOP_PHASE <= 4 && num_tokens >= 0)
        return;

    // 4c. Scatter, one thread per assignment rather than one wave per expert.
    //     A token's slot inside its expert is just the number of that expert's
    //     bits below it, so the destination is computable directly from the
    //     bitmask with no cross-thread coordination:
    //         pos = base[e] + popcount(bits of e below this token)
    //     which also reproduces the reference's ascending-token ordering.
    for(int idx = tid; idx < total_assign; idx += BLOCK)
    {
        const int e = FUSE_TOPK ? sids[idx] : topk_ids[idx];
        if(e < 0 || e >= num_experts)
            continue;
        const int t = idx / topk;
        const int j = idx - t * topk;
        const int w = idx >> 6;

        if(expert_mask != nullptr && expert_mask[e] == 0)
            continue; // masked expert contributes nothing

        const size_t erow = static_cast<size_t>(e) * WP;
        const int rank =
            swordpre[erow + w] + __popcll(smask[erow + w] & ((1ull << (idx & 63)) - 1ull));

        const int pos       = scum[e] + rank;
        sorted_ids[pos]     = (j << 24) | t;
        sorted_weights[pos] = (STAGE_W || FUSE_TOPK) ? sweight[idx] : topk_weights[idx];
    }
}

// F1-a: sorting only; the caller still runs its own router.
template <int BLOCK, bool STAGE_W>
__global__ void __launch_bounds__(BLOCK)
    fused_moe_sorting_bitmask(const int32_t* __restrict__ topk_ids,
                              const float* __restrict__ topk_weights,
                              const int32_t* __restrict__ expert_mask,
                              const int32_t* __restrict__ num_local_tokens,
                              int32_t* __restrict__ sorted_ids,
                              float* __restrict__ sorted_weights,
                              int32_t* __restrict__ sorted_expert_ids,
                              int32_t* __restrict__ num_valid_ids,
                              void* __restrict__ moe_buf,
                              int num_tokens,
                              int topk,
                              int num_experts,
                              int unit_size,
                              size_t moe_buf_bytes,
                              int zero_blocks)
{
    fused_moe_sorting_body<BLOCK, STAGE_W, false>(topk_ids,
                                                  topk_weights,
                                                  expert_mask,
                                                  num_local_tokens,
                                                  sorted_ids,
                                                  sorted_weights,
                                                  sorted_expert_ids,
                                                  num_valid_ids,
                                                  moe_buf,
                                                  num_tokens,
                                                  topk,
                                                  num_experts,
                                                  unit_size,
                                                  moe_buf_bytes,
                                                  zero_blocks,
                                                  FusedRouterArgs{});
}

// F1-b: router folded in. topk_ids / topk_weights never leave LDS, which is
// the point -- it removes a launch *and* an HBM round trip, and §3 identifies
// the round trip as the real cost at decode shapes.
template <int BLOCK, bool STAGE_W, int RNREG, int RW>
__global__ void __launch_bounds__(BLOCK)
    fused_moe_sorting_topk_bitmask(const int32_t* __restrict__ expert_mask,
                                   const int32_t* __restrict__ num_local_tokens,
                                   int32_t* __restrict__ sorted_ids,
                                   float* __restrict__ sorted_weights,
                                   int32_t* __restrict__ sorted_expert_ids,
                                   int32_t* __restrict__ num_valid_ids,
                                   void* __restrict__ moe_buf,
                                   int num_tokens,
                                   int topk,
                                   int num_experts,
                                   int unit_size,
                                   size_t moe_buf_bytes,
                                   int zero_blocks,
                                   FusedRouterArgs router)
{
    fused_moe_sorting_body<BLOCK, STAGE_W, true, RNREG, RW>(nullptr,
                                                            nullptr,
                                                            expert_mask,
                                                            num_local_tokens,
                                                            sorted_ids,
                                                            sorted_weights,
                                                            sorted_expert_ids,
                                                            num_valid_ids,
                                                            moe_buf,
                                                            num_tokens,
                                                            topk,
                                                            num_experts,
                                                            unit_size,
                                                            moe_buf_bytes,
                                                            zero_blocks,
                                                            router);
}

} // namespace aiter

// ---------------------------------------------------------------------------
// Host entry points (definitions live in py_itfs_cu/fused_moe_sorting_kernels.cu)
#ifndef MOE_SORTING_NO_TORCH_DECL
#include "aiter_tensor.h"
#include <optional>

int fused_moe_sorting_get_workspace_size(int tokens, int num_experts, int topk, int unit_size);

bool fused_moe_sorting_is_supported(int tokens, int num_experts, int topk, int unit_size);

bool fused_moe_sorting_topk_is_supported(
    int tokens, int num_experts, int topk, int unit_size, int router_experts);

void fused_moe_sorting_topk_fwd(aiter_tensor_t& gating_output,
                                aiter_tensor_t& sorted_token_ids,
                                aiter_tensor_t& sorted_weights,
                                aiter_tensor_t& sorted_expert_ids,
                                aiter_tensor_t& num_valid_ids,
                                aiter_tensor_t& moe_buf,
                                int num_experts,
                                int topk,
                                int unit_size,
                                bool need_renorm,
                                bool is_softmax,
                                float routed_scaling_factor,
                                std::optional<aiter_tensor_t> local_expert_mask = std::nullopt,
                                std::optional<aiter_tensor_t> num_local_tokens  = std::nullopt,
                                std::optional<aiter_tensor_t> sync              = std::nullopt);

void fused_moe_sorting_fwd(aiter_tensor_t& topk_ids,
                           aiter_tensor_t& topk_weights,
                           aiter_tensor_t& sorted_token_ids,
                           aiter_tensor_t& sorted_weights,
                           aiter_tensor_t& sorted_expert_ids,
                           aiter_tensor_t& num_valid_ids,
                           aiter_tensor_t& moe_buf,
                           int num_experts,
                           int unit_size,
                           std::optional<aiter_tensor_t> local_expert_mask = std::nullopt,
                           std::optional<aiter_tensor_t> num_local_tokens  = std::nullopt);
#endif
