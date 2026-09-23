#pragma once

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// opus_decode_index_score_kernels.cuh -- kernel TEMPLATE for the MiniMax-M3
// decode lightning-indexer block scoring (bf16 index-Q x fp8 index-K), ported
// from the phase-1 accepted prototype.
//
//   source          the porting study's accepted revision,
//                   kernel/decode_index_score.cu
//   configuration   accepted "config 3" = PAGE_HOIST=16, SWIZZLE=on,
//                   HALF_PAD=0 (AUX_K is the one tuned axis, {0, 3})
//   aiter baseline  25a909e, branch integration/decode-index-score
//
// THIS HEADER IS THE DEVICE TEMPLATE ONLY. No __global__ entry points, no
// launcher, no variant table, no ctypes entry: those are separate files
// (the integration study's P2/P3 stages).
//
// This kernel is an ADDITION, not a replacement. The incumbent MSA sparse
// block-select family (pa_sparse_block_select_kernels.cuh) implements a
// DIFFERENT operator contract -- it quantizes Q to fp8, takes no sm_scale, and
// carries init/local sentinels. Nothing here touches it; existing variants stay
// byte-identical (integration.md section 3).
//
// Operator contract (FROZEN in phase 1 -- spec.md section 1; reference
// ref/index_topk.py L973-1054):
//
//   score[h, b*MAX_Q + tok, blk]
//       = max_p( fp32dot(bf16(K[page][p][:]), Q[b*MAX_Q+tok][h][:]) * sm_scale*log2e )
//   -inf where blk*128 + p >= seq_len - MAX_Q + tok + 1
//
//   * Q is bf16 and is NEVER quantized. The fp8 key is lifted to bf16 exactly
//     (every OCP e4m3 value, subnormals included, is representable in bf16).
//     An fp8 MFMA would round Q, break bit-equality, and save nothing -- Q is
//     ~1 KiB against ~500 MiB of K.
//   * MFMA is v_mfma_f32_16x16x32_bf16, four per M-tile, ASCENDING k order.
//   * sm_scale_log2e = sm_scale * 1.4426950409f, computed IN THE KERNEL in fp32
//     and applied to the fp32 dot AFTER the dot. Never reassociate.
//   * causal -inf BEFORE the max; the max runs in fp32.
//   * correctness bar is bit-pattern equality on view(torch.int32) vs the
//     FROZEN MiniMax reference (not vs the aiter incumbent, which rounds Q and
//     accumulates in a different order). +0.0 vs -0.0 is a failure; NaN
//     position is the gate.
//
// FIVE STRUCTURAL INVARIANTS (integration-brief.md section 5). Breaking any one
// is a Class C change: stop, do not absorb it here.
//
//   I1  The score store's cache policy is aux = 0, PERMANENTLY. The 0.5 MiB
//       score buffer is read immediately afterwards by the top-k pass; a
//       non-temporal store cannot help and can slow the consumer.
//   I2  B <= 2 (two LDS staging buffers per wave). LDS per wave is B * 2048 B;
//       at 1,280 B granularity and 8 wave slots per SIMD, B=3 drops to 6
//       waves/SIMD.
//   I3  The page offset rides in the buffer descriptor's VOFFSET, NEVER in
//       soffset. CDNA4 ISA 9.1.5.2: "the sgpr_offset is not a part of the
//       offset term" -- soffset silently forfeits the hardware bounds check.
//   I4  The LDS trip is a PURE RELOCATION. Changing the k-slot grouping changes
//       fp32 accumulation order and breaks bit-equality.
//   I5  The XOR swizzle pi_X: lane l -> (row = l/8, piece = (l%8) XOR (l/8)) is
//       LOAD-BEARING -- worth 4.3%, mechanism-confirmed by a blocking counter
//       gate (TCP accesses 3.773x down against a 4.00x model prediction, misses
//       unchanged at 0.999x). Any layout change must preserve 1.00 L1 accesses
//       per 128 B line.
//
// TRAP A (phase-1 lead directive): the score store carries an EXPLICIT
// predicate. At BLOCK_SIZE_N = 16 with NUM_HEADS*MAX_Q < 16 the padded columns
// map to rows of OTHER requests, which are IN RANGE of the score buffer --
// hardware buffer OOB does NOT mask them, and an unpredicated store would
// silently corrupt a neighbouring request. The same reasoning applies to the Q
// load: padded columns are zeroed explicitly, never left to OOB.
//
// SCOPE: fp8 (OCP e4m3) index-K only. bf16 index-K is EXCLUDED BY HUMAN
// DECISION (integration.md ledger D0) -- a deliberate exclusion, not an
// oversight; its fallback is the EXISTING ValueError on the Python side. The
// bf16 branch below is retained verbatim but refused at compile time by a
// static_assert (Layer K, fail-closed), so the exclusion is visible rather than
// silently deleted, and phase-1's returning bf16 work does not have to
// reconstruct it.

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

// mfma_f32_16x16x32_bf16 fragment maps (wave64), per the opus mfma_adaptor
// shapes and the CDNA4 ISA; cross-checked against aiter's
// pa_sparse_prefill_opus.h / fmha_fwd_hd128 usage:
//   A (M x K = 16 x 32): lane l holds A[m = l % 16][k = (l / 16) * 8 + j], j = 0..7
//   B (K x N = 32 x 16): lane l holds B[k = (l / 16) * 8 + j][n = l % 16], j = 0..7
//   C (M x N = 16 x 16): lane l holds C[m = (l / 16) * 4 + j][n = l % 16], j = 0..3
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

// One wave's work for its whole 128-token blocks. Template parameters:
//   K_T       index-cache dtype. fp8_t only in aiter (ledger D0 refuses bf16_t
//             at compile time; the parameter is kept so the exclusion is a
//             refusal rather than a deletion).
//   HEAD_DIM  index head dim (128 at every matrix point; multiple of kMfmaK)
//   BLOCK_K   tokens per KV block (128 == SPARSE_BLOCK_SIZE)
//   NUM_HEADS, MAX_Q  compile-time, so BLOCK_SIZE_N and the padded-column
//             predicate are compile-time constants. (H, Q) in {1,4} x {1,4};
//             production is H = 1 (TP4 and TP8 both).
//   AUX_K     index-K LOAD cache policy -- the one tuned axis, {0, 3}. The
//             score STORE is aux = 0 unconditionally (invariant I1).
//   PAGE_HOIST / SWIZZLE / HALF_PAD  pinned to the accepted config 3 and
//             static_asserted: the phase-1 staircase (configs 0-5) does not
//             cross into aiter (integration.md P1).
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
    // The index-K descriptor is PER BLOCK (formulation P3): make_gmem(ik_ptr +
    // page*stride, one block), constructed inside the block loop below. The
    // whole-cache descriptor (P1) saturated at a 2 GiB byte footprint because
    // the load takes the byte offset as int; P3's per-block voffsets are
    // < 16 KiB, so the limit is gone and the matrix's top corner (batch 128 x
    // ctx 131072 x fp8 = exactly 2 GiB) is reachable.

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

    // Per-lane staging constants (bytes; fp8: elements == bytes). The async
    // instruction writes LDS[base + TID*16] (hardware-linear) and fetches the
    // gmem logical piece pi(l) of the half, so the 64 pieces per instruction are
    // ONE CONTIGUOUS 1024-B gmem range = 8 whole 128-B lines, each requested
    // exactly once (1.0x DRAM line-requests).
    //
    // LAYOUT, invariant I5: the L1 coalescing unit is the 16-lane PHASE, not the
    // 16-B piece: two lanes holding adjacent pieces of the same row merge into
    // one access. TRANSPOSE: a phase touches 8 lines x 32 B = 4.00 accesses per
    // line (measured 3.915, TCP). SWIZZLE: a phase covers rows 0-1 COMPLETELY
    // (lanes 0-7 take pieces 0-7 of row 0; lanes 8-15 take (0..7) XOR 1 = all 8
    // pieces of row 1) = 2 lines x 128 B = 1.00 accesses per line, the hardware
    // floor. The swizzle is the shipped layout and is worth 4.3%; the transpose
    // branch is retained as the control the counter gate was run against.
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
    // (row r, col kc*32 + g*8). TRANSPOSE keeps an affine kc stride (+256);
    // SWIZZLE does not (the XOR breaks it), so the four kc bases become four
    // loop-invariant per-lane VGPRs (+3 VGPRs, zero extra instructions in the
    // block loop).
    //
    // BANK NOTE: gfx950 LDS is 64 banks x 4 B (CDNA4 ISA 11.1 p103; the MI350
    // arch table 4.6 is a generational contrast table, MI300X 32 -> MI350X 64).
    // Both layouts put the two tile halves on the SAME banks at HALF_PAD=0 (half
    // stride 1024 B = 256 dwords = 0 mod 64), so every fragment-read phase runs
    // 2-WAY CONFLICTED. The HALF_PAD=+128 fix was MEASURED at -0.3% (a correctly
    // derived conflict that does not pay) and is therefore NOT shipped:
    // negative-results.md, and the shipped point is HALF_PAD=0.
    //
    // B2 (both layouts): verified in phase 1 by exhaustive simulation including
    // layout inversion -- every LDS byte holds exactly the logical byte its
    // fragment wants. Invariant I4: this is a PURE RELOCATION; the k-slot
    // grouping, and therefore the fp32 accumulation order, is unchanged.
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
        // Invariant I3: the descriptor is REBASED per block and the per-lane
        // offsets ride in VOFFSET. ~4 SALU per block; the hardware range check
        // is EXACT per block; and the per-lane voffsets stay within-block
        // (< 16 KiB), so the int32 saturation at a 2 GiB cache is gone. soffset
        // is never used for the page offset: CDNA4 ISA 9.1.5.2 excludes it from
        // the range check.
        //
        // What that check does NOT do is bound the page itself -- see below.
        // PAGE BOUND (review finding D04 / rule B6, upheld). The comment above
        // claimed the hardware range check bounded the PAGE offset. It never
        // did: the descriptor is rebased BY the page, so its extent bounds
        // offsets WITHIN the page and says nothing about the page index --
        // and a.ik_numel, which exists precisely to carry that bound, was
        // written by the launcher and read NOWHERE in the tree. A block table
        // holding a page past the cache therefore addressed memory outside the
        // tensor without faulting, and every test used a well-formed table so
        // nothing noticed.
        //
        // The extent is now clamped against what is actually left in the cache
        // from this page's base. A page past the end yields extent 0, which the
        // hardware range check turns into "every load returns 0" -- the same
        // fail-closed behaviour the original comment assumed it already had,
        // and which the -inf pre-fill contract then makes visible.
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

        // Seed with NaN, not -inf. fmaxf (v_max_f32) returns the non-NaN operand
        // and yields NaN only when BOTH operands are NaN, so a NaN seed makes the
        // whole fold -- the in-lane chain AND the cross-lane permlane max, which
        // are all fmaxf -- compute exactly the reference's seedless tl.max tree
        // semantics: "max of the non-NaN inputs, or NaN when every input is NaN".
        // A -inf seed EATS NaNs (fmaxf(-inf, NaN) = -inf), which is how an
        // all-NaN column came out -inf while the reference yielded NaN. This
        // also propagates the input NaN payload instead of synthesizing one. A
        // fully-masked column folds -inf values, which are non-NaN and eat the
        // seed -> -inf, as required.
        fp32_t col_max = numeric_limits<fp32_t>::quiet_nan();
        // FRAGILITY NOTE (phase-1 review, confirmed): the NaN-seed argument
        // relies on max<fp32_t> instantiating the FLOAT specialization --
        // __builtin_fmaxf -> v_max_f32. opus::max's generic max<T> is a plain
        // select (a > b ? a : b), which would eat the NaN seed with different
        // semantics. fp32_t is exactly float. If this fold is ever retyped to
        // bf16/fp16, the NaN-seed argument must be RE-DERIVED; it does not
        // transfer.
        //
        // Shared fold: scale AFTER the dot (fp32), causal -inf BEFORE the max,
        // fold this lane's 4 tokens into the running column max (max is
        // commutative).
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

// ARCH REFUSAL, POSITIVE (lead ruling, team-message-fe4c3f2e).
//
// The symbols below EXIST on the host pass and on every non-gfx950 device pass.
// An absent symbol is not a rejection -- it is the undefined branch, and a path
// that silently vanishes on another architecture violates the section 8.2 rule
// that every input is rejected at exactly one layer. So:
//
//   * the template is DEFINED, not merely declared -- a link error can never be
//     the mechanism of refusal;
//   * on a non-gfx950 DEVICE pass its body is __builtin_trap() -- this module's
//     own convention for a gfx950-only MFMA path
//     (pa_sparse_block_select_kernels.cuh mfma_scale, L29-37) -- so a launch
//     that somehow evades the host-side arch check dies immediately and loudly
//     instead of computing something;
//   * the Layer-K compile-time refusals (variant space, config point, K dtype
//     width) fire HERE TOO, on every pass, so a bad variant is rejected at
//     compile time regardless of architecture;
//   * a #pragma message marks the stub, so a non-gfx950 build of this TU is
//     visible in the build log rather than silent.
//
// WHY NOT #error, the form the ruling named first: aiter builds this module
// MULTI-ARCH -- aiter/jit/core.py:604 splits GPU_ARCHS on ';' and core.py:1176
// emits one --offload-arch per entry, so GPU_ARCHS="gfx942;gfx950" runs this
// device pass once per architecture over the same TU. A bare #error would fail
// that build outright and take the INCUMBENT sparse block-select path down with
// it -- a T1 regression on code this port is required not to touch. The #error
// form becomes correct if and only if the new path gets its own gfx950-only JIT
// module, which is a P3 build-integration decision. Flagged to the lead as an
// owed ruling, not taken unilaterally here.

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
