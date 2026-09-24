#pragma once

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// MiniMax-M3 decode lightning-indexer block scoring (bf16 Q x fp8 K), gfx950.
//   score[h, b*Q + tok, blk] = max_p( fp32dot(bf16(K), Q) * sm_scale * log2e )
// fp32 accumulate, four v_mfma_f32_16x16x32_bf16 per M-tile ascending k, maxNum
// fold. Q is never quantised: an fp8 MFMA would round it and save nothing.

#include <cstdint>
#include <type_traits>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include <bit>
#include <opus/opus.hpp>
#endif

namespace aiter {
namespace sparse_attn {

// Kernel arguments. Plain C types only: this struct is visible on BOTH the host
// and the device pass, because the launcher (a later file) fills it host-side.
// Field names are kept identical to the frozen prototype so the port can be
// audited field-for-field against the accepted revision.
struct opus_decode_score_args
{
    const void* __restrict__ q_ptr;        // [total_q, H, D] bf16
    const void* __restrict__ ik_ptr;       // [num_blocks, 128, D] fp8 OCP e4m3
    void* __restrict__ score_ptr;          // [H, total_q, max_block] fp32 (out)
    const void* __restrict__ bt_ptr;       // [batch, max_blocks] int32
    const void* __restrict__ seq_lens_ptr; // [batch] int32
    long long q_numel;                     // elements, whole tensor
    long long ik_numel;                    // elements, whole cache
    long long score_numel;                 // elements
    long long bt_numel;                    // int32 elements
    int batch;
    int chunk_blocks;                      // blocks per workgroup
    long long stride_q_n, stride_q_h;      // elements (stride_q_d == 1, host-asserted)
    long long stride_ik_blk;               // elements (stride_ik_pos == D, stride_ik_d == 1)
    long long stride_s_h, stride_s_b;      // elements (stride_s_k == 1)
    long long stride_bt_b;                 // int32 elements
    float sm_scale;                        // scaled by log2e IN-KERNEL (ledger D1)
};

// Wave geometry shared by the launcher and the kernel (gfx950 / wave64).
constexpr int kOpusWarpSize = 64;
constexpr int kOpusNumWarps = 4; // 256 threads/workgroup, as the reference launch

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

namespace opus_index_score {

using namespace ::opus;

// mfma_f32_16x16x32_bf16 fragment maps (wave64):
//   A (16 x 32): lane l holds A[l % 16][(l / 16) * 8 + j], j = 0..7
//   B (32 x 16): lane l holds B[(l / 16) * 8 + j][l % 16], j = 0..7
//   C (16 x 16): lane l holds C[(l / 16) * 4 + j][l % 16], j = 0..3
constexpr int kMfmaM = 16, kMfmaN = 16, kMfmaK = 32;
constexpr int kPackA = kMfmaM * kMfmaK / kOpusWarpSize; // 8
constexpr int kPackB = kMfmaN * kMfmaK / kOpusWarpSize; // 8
constexpr int kPackC = kMfmaM * kMfmaN / kOpusWarpSize; // 4
constexpr int kBlockN = 16; // == the reference's BLOCK_SIZE_N at every matrix point

// V1 staging geometry (fp8 path; bytes). One M-tile = 16 rows x 128 B = 2 KiB,
// staged as two 1024-B halves: per-M-tile double buffering keeps LDS at
// 16 KiB/workgroup = 8 waves/SIMD (invariant I2).
constexpr int kTileBytes = kMfmaM * 128 /*HEAD_DIM, static_asserted below*/
                           * 1 /*sizeof(fp8)*/;         // 2048 (GMEM tile stride)
constexpr int kHalfBytes = kTileBytes / 2;              // 1024 (GMEM half offset)

// fp8 (OCP e4m3) -> bf16, EXACT, 2 instructions per 4 bytes with scale 1.0.
// Precedent: aiter csrc/kernels/mla/opus/dsa_v32_splitkv.hpp.
__device__ __forceinline__ vector_t<bf16_t, kPackA>
fp8x8_to_bf16x8(const vector_t<fp8_t, kPackA>& v)
{
    vector_t<bf16_t, kPackA> r;
    const auto& w = reinterpret_cast<const vector_t<u32_t, kPackA / 4>&>(v);
    auto* pk      = reinterpret_cast<vector_t<bf16_t, 2>*>(&r);
    static_for<kPackA / 4>([&](auto d) {
        pk[d.value * 2 + 0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[d.value], 1.0f, false);
        pk[d.value * 2 + 1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(w[d.value], 1.0f, true);
    });
    return r;
}

template <typename K_T>
__device__ __forceinline__ vector_t<bf16_t, kPackA> k_to_bf16(const vector_t<K_T, kPackA>& v)
{
    if constexpr(std::is_same_v<K_T, bf16_t>)
        return v; // bf16 cache: identity (excluded path, ledger D0)
    else
        return fp8x8_to_bf16x8(v); // fp8 cache: exact lift
}

// One wave's work for its whole 128-token blocks. K_T is fp8_t only; the
// parameter is kept so bf16 is a compile-time refusal rather than a deletion.
// HEAD_DIM 128, BLOCK_K 128, NUM_HEADS x MAX_Q <= 16, AUX_K in {0, 3}.
template <typename K_T,
          int HEAD_DIM,
          int BLOCK_K,
          int NUM_HEADS,
          int MAX_Q,
          int AUX_K,
          int PAGE_HOIST = 16,
          bool SWIZZLE   = true,
          int HALF_PAD   = 0>
__device__ void decode_index_score_impl(const opus_decode_score_args& a)
{
    static_assert(HEAD_DIM % kMfmaK == 0, "HEAD_DIM must be a multiple of 32");
    static_assert(BLOCK_K % kMfmaM == 0, "BLOCK_K must be a multiple of 16");
    static_assert(NUM_HEADS * MAX_Q <= kBlockN,
                  "this kernel specialises BLOCK_SIZE_N = 16 (max NUM_HEADS*MAX_Q = 4*4)");
    // Layer K refusals, fail-closed (integration.md section 8.2).
#ifndef OPUS_IDX_SCORE_ALLOW_BF16_K
    static_assert(std::is_same_v<K_T, fp8_t>,
                  "bf16 index-K is excluded by human decision (integration.md ledger D0): "
                  "the fallback is the existing Python ValueError, not this kernel");
#else
    // PROBE BUILD ONLY. Defined by opus_decode_index_score_bf16_probe.cu and by
    // nothing else; the shipped translation units never see it, so the tokens
    // they compile are unchanged (verified by comparing their object hashes
    // before and after this edit -- the T1 pattern applied to our own TUs).
    static_assert(std::is_same_v<K_T, fp8_t> || std::is_same_v<K_T, bf16_t>,
                  "probe build: fp8 or bf16 index-K only");
#endif
    static_assert(PAGE_HOIST == 16 && SWIZZLE && HALF_PAD == 0,
                  "aiter builds the accepted config 3 only (PAGE_HOIST=16, SWIZZLE=on, "
                  "HALF_PAD=0); the phase-1 staircase points are not shipped");

    constexpr int kMTiles  = BLOCK_K / kMfmaM;  // 8
    constexpr int kKChunks = HEAD_DIM / kMfmaK; // 4
    constexpr int kNValid  = NUM_HEADS * MAX_Q; // valid columns of the 16-wide N tile

    const int b       = block_id_x();
    const int chunk   = block_id_y();
    const int warp    = __builtin_amdgcn_readfirstlane(thread_id_x() / kOpusWarpSize);
    const int lane    = lane_id();
    const int lane_mn = lane % 16; // A row-in-tile; B,C column n
    const int lane_g  = lane / 16; // A,B k-group; C m-group

    auto g_seq = make_gmem(reinterpret_cast<const int*>(a.seq_lens_ptr),
                           (unsigned)(a.batch * (long long)sizeof(int)));
    auto g_bt  = make_gmem(reinterpret_cast<const int*>(a.bt_ptr),
                           (unsigned)(a.bt_numel * (long long)sizeof(int)));
    auto g_q   = make_gmem(reinterpret_cast<const bf16_t*>(a.q_ptr),
                           (unsigned)(a.q_numel * (long long)sizeof(bf16_t)));
    auto g_s   = make_gmem(reinterpret_cast<fp32_t*>(a.score_ptr),
                           (unsigned)(a.score_numel * (long long)sizeof(fp32_t)));
    // The index-K descriptor is PER BLOCK: make_gmem(ik_ptr + page*stride, one
    // block), built inside the block loop. A whole-cache descriptor saturated
    // at 2 GiB because the load takes a 32-bit byte offset.

    const int seq_len     = g_seq.load(b)[0]; // scalar load returns vector_t<T,1>
    const int num_blocks  = (seq_len + BLOCK_K - 1) / BLOCK_K;
    const int chunk_start = chunk * a.chunk_blocks;
    const int chunk_end   = min(chunk_start + a.chunk_blocks, num_blocks);
    if(chunk_start >= chunk_end)
        return; // uniform: the whole workgroup exits
    // NOTE (OQ-4, verified): blocks >= num_blocks are left UNTOUCHED, exactly as
    // the frozen reference and as the incumbent aiter score op. The consumer
    // bounds its own reads (pa_sparse_block_topk does).

    // Hoist the chunk's page list: PAGE_HOIST consecutive int32,
    // workgroup-uniform, clamped in-bounds. Blocks past the hoist
    // (chunk_blocks > PAGE_HOIST) fall back to the bounds-checked buffer load.
    const int* bt_row = reinterpret_cast<const int*>(a.bt_ptr) + b * a.stride_bt_b;
    int pages[PAGE_HOIST];
    static_for<PAGE_HOIST>([&](auto i) {
        const int idx  = min(chunk_start + (int)i.value, chunk_end - 1);
        pages[i.value] = bt_row[idx];
    });

    // ---- B fragments: the query tile, hoisted out of the block loop ----
    // Column n = tok * NUM_HEADS + head (the reference's off_n ordering).
    // TRAP A: padded columns' Q rows belong to OTHER requests and are in range,
    // so they are ZEROED EXPLICITLY, never left to hardware OOB.
    const int n          = lane_mn;
    const bool n_valid   = n < kNValid;
    const int tok        = n / NUM_HEADS;
    const int head       = n - tok * NUM_HEADS;
    const int q_row      = b * MAX_Q + tok;
    const int causal_len = seq_len - MAX_Q + tok + 1; // reference L1029

    vector_t<bf16_t, kPackB> b_frag[kKChunks];
    static_for<kKChunks>([&](auto kc) {
        if(n_valid)
            b_frag[kc.value] = g_q.template load<kPackB>(
                (int)(q_row * a.stride_q_n + head * a.stride_q_h) + kc.value * kMfmaK
                + lane_g * kPackB);
        else
            b_frag[kc.value] = vector_t<bf16_t, kPackB>{}; // zero
    });

    auto mma = make_mfma<bf16_t, bf16_t, fp32_t>(
        number<kMfmaM>{}, number<kMfmaN>{}, number<kMfmaK>{});
    const fp32_t neg_inf = -numeric_limits<fp32_t>::infinity();
    // Ledger D1: sm_scale*log2e in fp32, IN-KERNEL, applied after the dot
    // (reference L1031). The incumbent's scale-less signature belongs to a
    // different operator and is deliberately not matched.
    const fp32_t sm_scale_log2e = a.sm_scale * 1.4426950409f;

    // ---- V1 staging machinery (fp8 path) ----
    constexpr bool kUseLdsStage = std::is_same_v<K_T, fp8_t>;
    static_assert(!kUseLdsStage || HEAD_DIM == 128,
                  "the V1 LDS layout is derived for HEAD_DIM=128 (every matrix point)");
    static_assert(!kUseLdsStage || kTileBytes == kMfmaM * HEAD_DIM * (int)sizeof(K_T),
                  "staging tile size must equal one M-tile of K_T (16 rows x HEAD_DIM)");
    static_assert(!kUseLdsStage || kHalfBytes == kTileBytes / 2, "half = tile/2");
    constexpr int kHalfStride   = kHalfBytes + HALF_PAD;   // LDS half-1 offset
    constexpr int kLdsTileBytes = kHalfStride + kHalfBytes; // LDS buffer stride
    static_assert(!kUseLdsStage || kHalfStride % 16 == 0,
                  "the LDS half stride must be 16-B aligned (dwordx4 direct-to-LDS writes)");
    // 4 waves x B=2 buffers x one 2-KiB M-tile = 16 KiB/workgroup -> 8 waves/SIMD
    // (invariant I2). The 1-byte dummy keeps a non-staging instantiation at zero
    // LDS.
    __shared__ __align__(128) char lds_raw[kUseLdsStage ? kOpusNumWarps * 2 * kLdsTileBytes : 1];
    auto s_buf = make_smem(reinterpret_cast<K_T*>(&lds_raw[warp * 2 * kLdsTileBytes]));

    // Per-lane staging constants (bytes; fp8: elements == bytes). One async
    // instruction covers one contiguous 1024 B gmem range = 8 whole 128 B
    // lines, each requested exactly once.
    int lane_async_h0, lane_async_h1;
    if constexpr(!SWIZZLE)
    {
        // TRANSPOSE: lane l fetches (row l%8, piece l/8)
        lane_async_h0 = (lane % 8) * 128 + (lane / 8) * 16;
        lane_async_h1 = kHalfBytes + (lane % 8) * 128 + (lane / 8) * 16;
    }
    else
    {
        // XOR SWIZZLE (pi_X): lane l fetches (row l/8, piece (l%8)^(l/8))
        lane_async_h0 = (lane / 8) * 128 + ((lane % 8) ^ (lane / 8)) * 16;
        lane_async_h1 = kHalfBytes + (lane / 8) * 128 + ((lane % 8) ^ (lane / 8)) * 16;
    }
    // Fragment ds_read: lane (r = l%16, g = l/16) reads 8 B of logical
    // (row r, col kc*32 + g*8). The XOR swizzle breaks the affine kc stride, so
    // the four kc bases are held as loop-invariant per-lane VGPRs.
    int lane_ds_kc[kKChunks];
    if constexpr(!SWIZZLE)
    {
        const int base = (lane_mn / 8) * kHalfStride + (lane_g / 2) * 128 + (lane_mn % 8) * 16
                         + (lane_g % 2) * 8;
        static_for<kKChunks>([&](auto kc) { lane_ds_kc[kc.value] = base + kc.value * 256; });
    }
    else
    {
        static_for<kKChunks>([&](auto kc) {
            lane_ds_kc[kc.value] = (lane_mn / 8) * kHalfStride + (lane_mn % 8) * 128
                                   + (((2 * kc.value + lane_g / 2) ^ (lane_mn % 8)) * 16)
                                   + (lane_g % 2) * 8;
        });
    }
    // Drain the Q/seq/block-table loads so the block loop's vmcnt accounting
    // below counts ONLY the staging asyncs.
    s_waitcnt_vmcnt(number<0>{});

    // ---- one wave per 128-token block: wave w takes blocks cs+w, +4, ... ----
    for(int blk = chunk_start + warp; blk < chunk_end; blk += kOpusNumWarps)
    {
        const int iblk = blk - chunk_start;
        int page;
        if(iblk < PAGE_HOIST)
        {
            page = pages[iblk]; // dynamic index into the uniform hoist
        }
        else
        {
            page = g_bt.load((int)(b * a.stride_bt_b) + blk)[0];
            // Drain immediately: the value feeds the per-block descriptor (the
            // staging asyncs depend on it), and the loop's vmcnt accounting must
            // count only the staging asyncs.
            s_waitcnt_vmcnt(number<0>{});
        }
        // The descriptor is REBASED per block, so per-lane voffsets stay under
        // 16 KiB and the int32 byte-offset limit at a 2 GiB cache is gone.
        // The range check is exact per block -- but it bounds offsets WITHIN
        // the page, not the page index; that is what the clamp below is for.
        const int64_t k_base   = (int64_t)page * a.stride_ik_blk;
        const int64_t k_remain = a.ik_numel - k_base;
        const int64_t k_want   = (int64_t)BLOCK_K * HEAD_DIM;
        const int64_t k_extent = (k_base < 0 || k_remain <= 0)
                                     ? 0
                                     : (k_remain < k_want ? k_remain : k_want);
        auto g_k = make_gmem(reinterpret_cast<const K_T*>(a.ik_ptr) + k_base,
                             (unsigned)(k_extent * (long long)sizeof(K_T)));

        // Causal-mask hoist: only the last block(s) of a request can be
        // partially masked; the min over columns of causal_len is
        // seq_len - MAX_Q + 1 (tok = 0).
        const int pos_base   = blk * BLOCK_K;
        const bool need_mask = (pos_base + BLOCK_K - 1) >= (seq_len - MAX_Q + 1);

        // Seed with NaN, not -inf: v_max_f32 returns the non-NaN operand and
        // yields NaN only when both are NaN, which reproduces the reference's
        // seedless max-tree semantics. A -inf seed would eat NaNs.
        fp32_t col_max = numeric_limits<fp32_t>::quiet_nan();
        // The NaN-seed argument requires max<fp32_t> to be the FLOAT
        // specialisation (v_max_f32); the generic max<T> is a select and would
        // eat the seed. Retyping this fold means re-deriving the argument.
        auto fold_tile = [&](const vector_t<fp32_t, kPackC>& acc, int mt) {
            const int token0 = mt * kMfmaM + lane_g * kPackC;
            static_for<kPackC>([&](auto j) {
                fp32_t v = acc[j.value] * sm_scale_log2e;
                if(need_mask && (pos_base + token0 + j.value) >= causal_len)
                    v = neg_inf;
                col_max = max(col_max, v);
            });
        };

        if constexpr(kUseLdsStage)
        {
            // V1 staged pipeline. stage_tile: two halves per tile; each
            // instruction moves 64 lanes x 16 B from ONE contiguous 1024-B gmem
            // range (piece-permuted across lanes) into LDS[base + TID*16].
            auto stage_tile = [&](int mt, int buf) {
                // Within-block element offsets (fp8: elements == bytes);
                // lane_async_h1 already includes the +kHalfBytes half offset.
                const int tile = mt * kTileBytes;
                g_k.template async_load<16, 0, AUX_K>(
                    &lds_raw[(warp * 2 + buf) * kLdsTileBytes], tile + lane_async_h0);
                g_k.template async_load<16, 0, AUX_K>(
                    &lds_raw[(warp * 2 + buf) * kLdsTileBytes + kHalfStride],
                    tile + lane_async_h1);
            };
            stage_tile(0, 0);
            if(kMTiles > 1)
                stage_tile(1, 1);
            static_for<kMTiles>([&](auto mt) {
                // Wait for THIS tile's 2 asyncs to land in LDS (the next tile's 2
                // may still be in flight); the last tile has no successor, so
                // wait for all. The counts are exact because everything before
                // the block loop was drained, and the g_bt fallback drains where
                // it loads.
                if(mt.value == kMTiles - 1)
                    s_waitcnt_vmcnt(number<0>{});
                else
                    s_waitcnt_vmcnt(number<2>{});
                // Read the 4 A-fragments back (ds_read_b64).
                vector_t<K_T, kPackA> frag[kKChunks];
                static_for<kKChunks>([&](auto kc) {
                    frag[kc.value] = s_buf.template load<kPackA>(
                        (mt.value & 1) * kLdsTileBytes + lane_ds_kc[kc.value]);
                });
                s_waitcnt_lgkmcnt(number<0>{}); // frags in regs; buffer reusable
                if(mt.value + 2 < kMTiles)
                    stage_tile(mt.value + 2, mt.value & 1);
                // Convert + MFMA. K = 128 accumulates as 4 sequential
                // mfma_f32_16x16x32_bf16 in ASCENDING k order -- this ordering is
                // the bit-equality contract (invariant I4), not a preference.
                vector_t<fp32_t, kPackC> acc = {};
                static_for<kKChunks>(
                    [&](auto kc) { acc = mma(k_to_bf16<K_T>(frag[kc.value]), b_frag[kc.value], acc); });
                fold_tile(acc, mt.value);
            });
        }
        else
        {
            // Direct path (no LDS trip). Unreachable in aiter: the static_assert
            // above refuses every K_T but fp8_t, and fp8_t stages. Retained
            // verbatim because the bf16 index-K path is a recorded phase-1
            // exclusion (ledger D0), not a deletion.
            static_for<kMTiles>([&](auto mt) {
                vector_t<fp32_t, kPackC> acc = {};
                static_for<kKChunks>([&](auto kc) {
                    auto raw = g_k.template load<kPackA, AUX_K>(
                        (mt.value * kMfmaM + lane_mn) * HEAD_DIM + kc.value * kMfmaK
                        + lane_g * kPackA);
                    acc = mma(k_to_bf16<K_T>(raw), b_frag[kc.value], acc);
                });
                fold_tile(acc, mt.value);
            });
        }

        // Cross-lane max over the 4 m-groups: lanes l, l^16, l^32, l^48 hold the
        // same column n = l % 16 (the C fragment's m-group is l / 16). Idiom as
        // in aiter's pa_sparse_prefill_opus.h / fmha kernels.
        u32_t bits = std::bit_cast<u32_t>(col_max);
        auto r32   = __builtin_amdgcn_permlane32_swap(bits, bits, false, true);
        col_max    = max(std::bit_cast<fp32_t>(r32[0]), std::bit_cast<fp32_t>(r32[1]));
        bits       = std::bit_cast<u32_t>(col_max);
        auto r16   = __builtin_amdgcn_permlane16_swap(bits, bits, false, true);
        col_max    = max(std::bit_cast<fp32_t>(r16[0]), std::bit_cast<fp32_t>(r16[1]));

        // TRAP A: explicit predicate on the store. Invariant I1: the store's aux
        // is 0 (the opus default used here) and must stay 0 -- the top-k
        // selector reads this buffer immediately after this kernel.
        if(lane_g == 0 && n_valid)
            g_s.store(col_max, (int)(head * a.stride_s_h + q_row * a.stride_s_b) + blk);
    }
}

} // namespace opus_index_score

#else // not the gfx950 device pass

// Arch refusal, POSITIVE: these symbols exist on every pass, so a non-gfx950
// build reaches a deliberate static_assert instead of silently vanishing. An
// absent symbol is the undefined branch, not a rejection.

#if defined(__HIP_DEVICE_COMPILE__)
#pragma message("opus_decode_index_score: non-gfx950 device pass -- compiling the refusal stub (__builtin_trap). This path runs on gfx950 only; the host-side arch check rejects before any launch.")
#endif

namespace opus_index_score {

template <typename K_T,
          int HEAD_DIM,
          int BLOCK_K,
          int NUM_HEADS,
          int MAX_Q,
          int AUX_K,
          int PAGE_HOIST = 16,
          bool SWIZZLE   = true,
          int HALF_PAD   = 0>
__device__ void decode_index_score_impl(const opus_decode_score_args& a)
{
    // The same Layer-K refusals as the real body, expressed without opus types
    // so they also hold on the host pass. sizeof(K_T) == 1 is a deliberate
    // PROXY for "K_T is fp8_t": bf16_t is 2 bytes, so this refuses the excluded
    // bf16 index-K path at compile time here too; the exact
    // std::is_same_v<K_T, fp8_t> assertion is in the gfx950 body above.
    static_assert(sizeof(K_T) == 1,
                  "bf16 index-K is excluded by human decision (integration.md ledger D0): "
                  "the fallback is the existing Python ValueError, not this kernel");
    static_assert(HEAD_DIM == 128 && BLOCK_K == 128,
                  "this kernel is built for head_dim = block_size = 128 only");
    static_assert(NUM_HEADS * MAX_Q <= 16,
                  "this kernel specialises BLOCK_SIZE_N = 16 (max NUM_HEADS*MAX_Q = 4*4)");
    static_assert(PAGE_HOIST == 16 && SWIZZLE && HALF_PAD == 0,
                  "aiter builds the accepted config 3 only (PAGE_HOIST=16, SWIZZLE=on, "
                  "HALF_PAD=0); the phase-1 staircase points are not shipped");
    (void)a;
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap(); // non-gfx950 device pass: refuse, never compute
#endif
}

} // namespace opus_index_score

#endif // __HIP_DEVICE_COMPILE__ && __gfx950__

} // namespace sparse_attn
} // namespace aiter
