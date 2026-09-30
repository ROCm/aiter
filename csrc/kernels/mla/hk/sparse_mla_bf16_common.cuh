// SPDX-License-Identifier: MIT
// Derived from ROCm/aiter PR #3459 (e961cb0c), HipKittens d3cd9b31.
// Sparse MLA with a shared BF16 K/V latent: D512, no appended RoPE, H16/H64.
#pragma once

#include "hk_mla_v40_fwd_decode_gen1_common.cuh"
#include <cmath>
#include <cstdint>

namespace sparse_mla_bf16 {

template <int Heads>
struct Traits
{
    static_assert(Heads == 16 || Heads == 64);
    static constexpr uint32_t kBlockM = Heads, kNumWarps = Heads / 16;
    static constexpr uint32_t kBlockN = 32, kBlockK = 32, kTileM = 16;
    static constexpr uint32_t kQkHeadDim = 512, kVoHeadDim = 512;
    static constexpr uint32_t kQkNopeHeadDim = 512, kQkRopeHeadDim = 0;
    static constexpr uint32_t kNumThreads = 64 * kNumWarps;
    static constexpr uint32_t kRoundMode  = 0; // nearest even, including BF16 ties
    // H16 keeps one KV tile for four LDS-limited workgroups/CU; H64 uses two
    // tiles to overlap next-KV DMA with current MFMA. Q staging overlays this
    // allocation only in the prologue; terminal O bounce overlays dead KV0.
    static constexpr uint32_t kKvBytes     = 32 * 512 * 2;
    static constexpr uint32_t kKvBuffers   = Heads == 64 ? 2u : 1u;
    static constexpr uint32_t kIndexOffset = kKvBuffers * kKvBytes;
    static constexpr uint32_t kLdsBytes    = kIndexOffset + 256;
    static_assert(kLdsBytes <= 80 * 1024);
    static_assert(kNumWarps * 4352 <= kKvBytes);
};

struct Params
{
    const uint16_t* q;
    const uint16_t* kv;
    const int32_t* indptr;
    const int32_t* indices;
    uint16_t* out;
    float* lse;
    float* partial_o;
    float* partial_lse;
    int32_t queries;
    int32_t splits;
    int64_t slots;
    int64_t q_stride0;
    int64_t q_stride1;
    int64_t kv_stride0;
    float scale;
};

// These BF16 LDS readers are copied from PR #3459 unchanged. They intentionally
// have no FP8/scale/RoPE loaders or traits: the new writer supplies their exact
// sb8-permuted, row-bank-swizzled 32x512 BF16 layout.
template <typename T>
struct Bf16KvManager
{
    static constexpr uint32_t kLoadBlockN = 32, kSubBlockRows = 16;
    static constexpr uint32_t kSubBlockCols = 32, kSubBlockBytes = 1024;
    static constexpr uint32_t kNumRowTiles = 2;
    __device__ __forceinline__ static constexpr uint32_t sub_block_byte_offset(uint32_t row_tile,
                                                                               uint32_t col_tile)
    {
        return (col_tile * kNumRowTiles + row_tile) * kSubBlockBytes;
    }
    // ---- LDS -> VGPR readout for QK / PV mfmas -----------------------------
    // QK A-tile load: ds_read_b128 of one 16 x 32 bf16 sub-block into 4 vgprs.
    // (kRowOffset, kColOffset) selects which (row_tile, col_tile) of the pong;
    // the per-lane offset within the sub-block follows the mfma_f32_16x16x32_bf16
    // A-operand layout (lane = (row_in_tile, group_in_row)).
    // kRowOffset spans the FULL logical tile (0/16 for kBlockN=32; 0/16/32/48 for
    // kBlockN=64). The sub-tile (kRowOffset/32) is folded into the ds_read imm as
    // kSubIdx*kSubPongBytes so A and B share the same base ptr -- avoids a per-call
    // base+kSubPong address VGPR that CSE holds across QK and overlaps a pinned reg.
    template <uint32_t kRowOffset, uint32_t kColOffset, hkdart::all RT>
    __device__ __forceinline__ static void load_k_to_gpr(RT& dst, const uintptr_t p_lds_kv)
    {
        static_assert(kRowOffset % kSubBlockRows == 0,
                      "load_k_to_gpr: kRowOffset must be a multiple of 16.");
        static_assert(kColOffset % kSubBlockCols == 0,
                      "load_k_to_gpr: kColOffset must be a multiple of 32.");
        static_assert(kRowOffset < T::kBlockN, "load_k_to_gpr: kRowOffset out of range.");
        static_assert(kColOffset < T::kQkHeadDim, "load_k_to_gpr: kColOffset out of range.");
        // Sub-tile B (kRowOffset >= 32) -> +kSubPongBytes in the ds_read imm.
        constexpr uint32_t kSubIdx       = kRowOffset / kLoadBlockN;
        constexpr uint32_t kRowInSub     = kRowOffset % kLoadBlockN;
        constexpr uint32_t kSubPongBytes = kLoadBlockN * T::kQkHeadDim * sizeof(hk::bf16);

        // mfma_f32_16x16x32_bf16 A-operand layout: lane t holds 8 bf16 from
        //   row r = lane%16, cols [c, c+8) where c = (lane/16) * 8.
        // 8 bf16 = 4 dwords -> ds_read_b128.
        constexpr uint32_t kMfmaRows       = 16;
        constexpr uint32_t kMfmaElemPerThr = 8;

        const uint32_t lane_idx = opus::lane_id();
        const uint32_t row      = lane_idx % kMfmaRows;
        const uint32_t col      = (lane_idx / kMfmaRows) * kMfmaElemPerThr;

        // Un-swizzle: writer XORs intra-sub-block byte position by 32 on
        // sub-tile-rows 1 & 3 (rows 4..7 and 12..15) to break the 2-way bank
        // conflict; the reader applies the same XOR on the col-byte component.
        const uint32_t row_bank_swap = ((row >> 2) & 1u) << 5;
        const uint32_t in_sb_byte =
            row * (kSubBlockCols * sizeof(hk::bf16)) + ((col * sizeof(hk::bf16)) ^ row_bank_swap);

        // Constexpr sub-block selector (compiles to immediate offset); the sub-tile
        // (A/B) contributes kSubIdx*kSubPongBytes so the base ptr stays p_lds_kv.
        constexpr uint32_t kFixedOffset =
            sub_block_byte_offset(kRowInSub / kSubBlockRows, kColOffset / kSubBlockCols) +
            kSubIdx * kSubPongBytes;

        // RT must hold a single 4-vgpr range (16 bf16 mfma A-tile = 4 vgprs/lane).
        using range_type = hkdart::get_nth_range_t<typename RT::register_ranges, 0>;
        static_assert(range_type::lo + 3 == range_type::hi,
                      "ds_read_b128 requires 4 consecutive registers");

        const uintptr_t p_lds_kv_lane = p_lds_kv + in_sb_byte;
        hkm::ds_read_b128<range_type::lo>(static_cast<uint32_t>(p_lds_kv_lane), kFixedOffset);
    }

    // PV A-tile load: ds_read_b64_tr_b16 (bf16 transpose-read) of one 16-row x 16-col
    // bf16 patch from a sub-block, results land in 2 dwords/lane in (GPR, GPR+1).
    //
    // PV math is V^T @ P^T = O^T computed via mma_ABt(oaccu, kv, p_mfma) (= kv @ p_mfma^T,
    // matching the QK convention of K^T @ Q^T = P^T). So `kv` is the A operand of
    // v_mfma_f32_16x16x32_bf16, holding V^T values reorganized into the mfma A layout.
    //
    // Within each 16-lane group, lane t's 4 bf16 (4 bf16 = 1 b64 = 2 dwords/lane) are
    // (after HW transpose):
    //   output_lane[g*16 + l] holds V[g*4+0..g*4+3, kColOffset + l]
    // for g = lane_group_idx (0..3), l = lane_in_group (0..15). I.e. each lane gets
    // 4 K-rows of one V-col. Caller stitches two row halves (kRowOffset = 0, then 16)
    // into a single mfma A operand spanning 8 K-rows (= mfma K = 0..7).
    //
    // Per-lane source address (within the selected 16x32 sub-block):
    //   in_sb_byte = (lane >> 2) * (kSubBlockCols * sizeof(bf16)) + (lane & 3) * 8
    //              = lane_row * 64 + lane_col_quad * 8
    // (row stride = 32 bf16 cols * 2 B = 64 B; each "col_quad" = 4 bf16 = 8 B.)
    //
    // Compile-time fixed_offset selects:
    //   * the sub-block (row_tile = kRowOffset/16, col_tile = kColOffset/32), and
    //   * the 16-col half within that 32-col sub-block: kColOffset%32 -> +0 or +32 B.
    //
    // Un-swizzle: writer XORs intra-sub-block byte position by 32 on rows
    // whose sub-tile-row index is odd (rows 4..7 and 12..15 within the 16-row
    // sub-block). The reader applies the same XOR. With this swizzle both
    // cycles of ds_read_b64_tr_b16 (lanes 0..31 covering rows 0..7, lanes
    // 32..63 covering rows 8..15) hit 32 distinct conflict slots -- fully
    // conflict-free for this per-wave instruction mapping.
    // kRowOffset spans the FULL logical tile (0/16 = sub-tile A, 32/48 = sub-tile B).
    // The sub-tile (kRowOffset/32) is folded into the ds_read imm as
    // kSubIdx*kSubPongBytes so the base ptr stays p_lds_v (no +kSubPong VGPR).
    template <uint32_t kRowOffset, uint32_t kColOffset, uint32_t GPR>
    __device__ __forceinline__ static void load_transposed_v_to_gpr(const uintptr_t p_lds_v)
    {
        static_assert((kRowOffset % kSubBlockRows == 0u) && (kRowOffset < T::kBlockN),
                      "load_transposed_v_to_gpr: kRowOffset must be 0/16 (A) or 32/48 (B).");
        static_assert(
            (kColOffset % 16u == 0u) && (kColOffset < T::kVoHeadDim),
            "load_transposed_v_to_gpr: kColOffset must be a multiple of 16, < kVoHeadDim.");

        constexpr uint32_t kSubIdx       = kRowOffset / kLoadBlockN; // 0 (A) or 1 (B)
        constexpr uint32_t kRowInSub     = kRowOffset % kLoadBlockN; // 0 or 16
        constexpr uint32_t kSubPongBytes = kLoadBlockN * T::kQkHeadDim * sizeof(hk::bf16);
        constexpr uint32_t kRowTile      = kRowInSub / kSubBlockRows;  // 0 or 1
        constexpr uint32_t kColTile      = kColOffset / kSubBlockCols; // 0..15
        constexpr uint32_t kColInSbBytes = (kColOffset % kSubBlockCols) * sizeof(hk::bf16);
        // Bank-swizzle re-expressed as a conditional ±32 delta so that
        // (kFixedOffset + kColInSbBytes) stays fully constexpr in the
        // ds_read_b64_tr_b16 immediate offset: XOR-by-32 against a constexpr
        // value flips bit 5, equivalent to "+32 if bit was 0, else -32".
        // The sign is compile-time (from kColInSbBytes's bit 5); only the
        // boolean `is_swz` (1 bit per lane) is runtime. Avoids materialising
        // kColInSbBytes as a runtime VGPR (vs. the plain XOR formulation),
        // which freed 2 unpinned VGPRs in the audit.
        constexpr int32_t kSwzDelta = (kColInSbBytes & 32u) ? -32 : +32;
        constexpr uint32_t kFixedOffset =
            sub_block_byte_offset(kRowTile, kColTile) + kColInSbBytes + kSubIdx * kSubPongBytes;

        const uint32_t lane_idx  = opus::lane_id();
        const uint32_t row_in_sb = lane_idx >> 2;
        const uint32_t is_swz    = (row_in_sb >> 2) & 1u;
        const uint32_t in_sb     = row_in_sb * (kSubBlockCols * sizeof(hk::bf16)) +
                               (lane_idx & 3u) * 8u + is_swz * static_cast<uint32_t>(kSwzDelta);
        const uint32_t addr = static_cast<uint32_t>(p_lds_v) + in_sb;

        hkm::ds_read_b64_tr_b16<GPR>(addr, kFixedOffset);
    }

    // bf16 ds_read_b64_tr_b16 already lands in the mfma A-operand layout -- no
    // intra-lane v_swap_b32 fixup needed (V32's fp8 path interleaved cols c and
    // c+16 into the same 2 GPRs and required a swap; the b16 transpose does not).
    // Kept as a no-op for caller parity with KvManager8bitsV3.
    template <uint32_t GPR_0, uint32_t GPR_1>
    __device__ __forceinline__ static void finalize_load_transposed_v_to_gpr()
    {
    }
};

// PR PV schedule, changed only to accept the BF16-only LDS manager.
template <bool kIsFirstIter, bool kDoRescale, typename T>
__device__ __forceinline__ void
pv_gemm(Bf16KvManager<T>& kv_manager, const uintptr_t p_lds_v, const float rescale)
{
    using R                       = HkMlaV40Regs<T>;
    using comp_t                  = typename R::comp_t;
    constexpr uint32_t k_o_begin  = R::k_o_begin;
    constexpr uint32_t k_v0_begin = R::k_v0_begin;
    constexpr uint32_t k_k0_begin = R::k_k0_begin;
    constexpr uint32_t k_k1_begin = R::k_k1_begin;
    constexpr uint32_t k_k2_begin = R::k_k2_begin;
    // kBlockN=64 (m16x8 double-tile) contracts BOTH 32-row sub-tiles in one call:
    // sub-tile A (rows 0:32) uses p_mfma_a, B (rows 32:64) uses p_mfma_b. kBlockN=32
    // has only A. Picked per-iter by row_base (see below), no caller arg.
    typename R::p_mfma_a_t p_mfma_a;
    typename R::p_mfma_b_t p_mfma_b;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;
    typename R::pv_v_3_t pv_v_3;

    // D-tiling: each D-iter emits 2 oaccu 16x16 sub-tiles (32 D-cols); 512/32 = 16
    // D-iters per 32-row KV sub-tile. kBlockN=64 runs BOTH sub-tiles in one call, so
    // num_pv_iter = kNumKvSub * 16. iter [0,16) = sub-tile A, [16,32) = sub-tile B.
    constexpr uint32_t kDIters     = T::kVoHeadDim / (2u * T::kTileM); // 16
    constexpr uint32_t num_pv_iter = R::kNumKvSub * kDIters;           // 16 or 32
    // Flat V base tiles across all sub-tiles: 32 per sub-tile (S_0..S_31 each).
    constexpr uint32_t kVTilesPerSub = 2u * kDIters;                 // 32
    constexpr uint32_t kNumVTiles    = R::kNumKvSub * kVTilesPerSub; // 32 or 64

    auto pk_mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        const float2 r2         = {r, r};
        asm volatile("v_pk_mul_f32 v[%0:%1], %2, v[%0:%1]" : : "n"(base), "n"(base + 1), "v"(r2));
    };
    auto mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base), "v"(r));
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base + 1), "v"(r));
    };

    // Issue both ds_read_b64 for base tile S_jj (within its sub-tile) into a
    // round-robin slot (4 slots). kRowBase selects the KV sub-tile (0=A, 32=B),
    // folded into the V ds_read imm by load_transposed_v_to_gpr.
    auto load_S = [&]<uint32_t kRowBase, uint32_t jj, uint32_t slot>() {
        constexpr uint32_t base = (slot == 0u)   ? k_v0_begin
                                  : (slot == 1u) ? k_k0_begin
                                  : (slot == 2u) ? k_k1_begin
                                                 : k_k2_begin;
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 0u, jj * 16u, base + 0>(p_lds_v);
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 16u, jj * 16u, base + 2>(p_lds_v);
    };
    // mfma: oaccu_dst (+)= pv_v_{slot}^T @ p_mfma[sub-tile]. 3-arg init only on the
    // FIRST sub-tile of a first-iter call (kRowBase==0); sub-tile B always accumulates
    // onto A's oaccu (same oaccu tile, second N-half of P).
    auto do_mma = [&]<uint32_t kRowBase, uint32_t slot, typename OA>(OA& oaccu_dst) {
        auto run = [&](auto& v) {
            if constexpr(kIsFirstIter && kRowBase == 0u)
                hk::mma_ABt(oaccu_dst, v, p_mfma_a);
            else if constexpr(kRowBase == 0u)
                hk::mma_ABt(oaccu_dst, v, p_mfma_a, oaccu_dst);
            else
                hk::mma_ABt(oaccu_dst, v, p_mfma_b, oaccu_dst);
        };
        if constexpr(slot == 0u)
            run(pv_v_0);
        else if constexpr(slot == 1u)
            run(pv_v_1);
        else if constexpr(slot == 2u)
            run(pv_v_2);
        else
            run(pv_v_3);
    };

    if constexpr(kDoRescale)
    {
        opus::static_for<2>([&](auto s) {
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 0u>{});
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 2u>{});
        });
    }

    // Prologue: preload the first 3 flat V tiles (S_0,S_1,S_2 of sub-tile A) into
    // slots 0,1,2. ONE prologue for the whole call -- sub-tile B's S_0..S_2 are
    // prefetched by sub-tile A's tail iters via the flat prefetch index below.
    load_S.template operator()<0u, 0u, 0u>();
    load_S.template operator()<0u, 1u, 1u>();
    load_S.template operator()<0u, 2u, 2u>();

    opus::static_for<num_pv_iter>([&](auto i) {
        constexpr uint32_t iter     = i.value;
        constexpr uint32_t col_idx  = iter % kDIters;         // 0..15 D-tile within sub-tile
        constexpr uint32_t row_base = (iter / kDIters) * 32u; // 0 (A) or 32 (B)
        constexpr bool has_next     = (iter + 1u) < num_pv_iter;
        // Rescale only folds on the FIRST sub-tile pass (row_base==0) and only when a
        // NEXT D-tile in that pass exists (col_idx+1 < kDIters) -- else oaccu overflows.
        constexpr bool resc_next = kDoRescale && (row_base == 0u) && (col_idx + 1u < kDIters);
        constexpr uint32_t next_oaccu_base = k_o_begin + (col_idx + 1u) * 8u;

        // Flat V-tile index across all sub-tiles: prefetch runs on this so sub-tile A's
        // tail prefetches sub-tile B's S_0.. (ONE prologue). Consume uses col_idx.
        constexpr uint32_t jf_lo = 2u * iter;      // flat mfma idx (a)
        constexpr uint32_t jf_hi = 2u * iter + 1u; // flat mfma idx (b)
        // 4-slot round-robin (flat tile T -> slot T%4). Reading slot j%4 while refilling
        // slot (j+3)%4 keeps the refill ds_read's dst != the mfma's src reg.
        constexpr uint32_t slot_lo = jf_lo % 4u;
        constexpr uint32_t slot_hi = jf_hi % 4u;

        constexpr uint32_t oaccu_base = k_o_begin + col_idx * 8u;
        using oaccu_a_r =
            hkdart::split_many_t<hkdart::type_list<hkdart::range<oaccu_base + 0, oaccu_base + 3>>,
                                 4>;
        using oaccu_b_r =
            hkdart::split_many_t<hkdart::type_list<hkdart::range<oaccu_base + 4, oaccu_base + 7>>,
                                 4>;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_a_r> oaccu_a;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_b_r> oaccu_b;

        // Prefetch S at flat index jf+3 -> its own sub-tile row_base + local jj%32.
        auto prefetch = [&]<uint32_t jf>() {
            if constexpr(jf < kNumVTiles)
            {
                constexpr uint32_t pf_row = (jf / kVTilesPerSub) * 32u;
                constexpr uint32_t pf_jj  = jf % kVTilesPerSub;
                load_S.template operator()<pf_row, pf_jj, jf % 4u>();
            }
        };

        // ---- mfma_a (flat jf_lo, reads slot_lo) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 2, -1));
        prefetch.template operator()<jf_lo + 3u>();
        do_mma.template operator()<row_base, slot_lo>(oaccu_a);
        if constexpr(resc_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 0>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 0>{});
        }

        // ---- mfma_b (flat jf_hi, reads slot_hi) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 0, -1));
        prefetch.template operator()<jf_hi + 3u>();
        do_mma.template operator()<row_base, slot_hi>(oaccu_b);
        if constexpr(resc_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 2>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 2>{});
        }
    });
}

// Pinned-register access is deliberately opaque, as in the upstream hk::art
// helpers. The entry's compiler scratch cap and clobber map are part of its ABI.
template <uint32_t Reg>
__device__ __forceinline__ float read_f32()
{
    float x;
    asm volatile("v_mov_b32 %0, v[%1]" : "=v"(x) : "n"(Reg));
    return x;
}
template <uint32_t Reg>
__device__ __forceinline__ int32_t read_i32()
{
    int32_t x;
    asm volatile("v_mov_b32 %0, v[%1]" : "=v"(x) : "n"(Reg));
    return x;
}
template <uint32_t Reg>
__device__ __forceinline__ void write_f32(float x)
{
    asm volatile("v_mov_b32 v[%0], %1" : : "n"(Reg), "v"(x));
}
template <uint32_t Reg>
__device__ __forceinline__ void write_u32(uint32_t x)
{
    asm volatile("v_mov_b32 v[%0], %1" : : "n"(Reg), "v"(x));
}
__device__ __forceinline__ void drain()
{
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, 0));
    __builtin_amdgcn_sched_barrier(0);
}

// The common aligned layout moves an entire eight-BF16 MFMA packet in one
// 128-bit load. The ABI still admits only two-byte alignment: padded/misaligned
// tensors use the exact scalar-pair path below. All values are copied bitwise.
template <typename T>
__device__ __forceinline__ void load_q(const Params& p)
{
    using R             = HkMlaV40Regs<T>;
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t head = (threadIdx.x >> 6) * 16u + (lane & 15u);
    const uint64_t offset =
        uint64_t(blockIdx.x) * uint64_t(p.q_stride0) + uint64_t(head) * uint64_t(p.q_stride1);
    const uint16_t* row = p.q + offset;
    const bool aligned  = ((reinterpret_cast<uintptr_t>(p.q) | (uint64_t(p.q_stride0) * 2u) |
                           (uint64_t(p.q_stride1) * 2u)) &
                          15u) == 0;
    opus::static_for<16>([&](auto c_) {
        constexpr uint32_t c = c_.value;
        const uint32_t col   = sb8_inv_perm_col_elems(c * 32u + (lane >> 4) * 8u);
        if(aligned)
        {
            // sb8 changes the packet order, never the eight values inside it.
            const v4ui packet = *reinterpret_cast<const v4ui*>(row + col);
            opus::static_for<4>([&](auto j_) {
                constexpr uint32_t j = j_.value;
                write_u32<R::k_q_vgpr_begin + c * 4u + j>(packet[j]);
            });
        }
        else
        {
            opus::static_for<4>([&](auto j_) {
                constexpr uint32_t j = j_.value;
                const uint32_t pair =
                    uint32_t(row[col + j * 2u]) | (uint32_t(row[col + j * 2u + 1u]) << 16);
                write_u32<R::k_q_vgpr_begin + c * 4u + j>(pair);
            });
        }
        // Do not hoist an entire 512-column Q row into compiler-managed VGPRs.
        __builtin_amdgcn_sched_barrier(0);
    });
    drain();
}

// Prologue-only Q staging reuses the not-yet-live KV ping/pong allocation.
// H64 occupies exactly64KiB (four private16KiB wave regions); H16 occupies16KiB.
// After every wave reads its pinned Q and crosses the terminal barrier, these
// bytes are dead and KV may overwrite them. No extra steady-state LDS is used.
template <typename T>
__device__ __forceinline__ void load_q_dma(const Params& p, uint16_t* staging)
{
    using R = HkMlaV40Regs<T>;
    static_assert(T::kNumWarps * 16u * 512u * 2u <= T::kIndexOffset);
    const bool aligned = ((reinterpret_cast<uintptr_t>(p.q) | (uint64_t(p.q_stride0) * 2u) |
                           (uint64_t(p.q_stride1) * 2u)) &
                          15u) == 0;
    if(!aligned)
    {
        load_q<T>(p); // preserve the original two-byte-alignment ABI
        return;
    }
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const uint32_t r = lane >> 2, quad = lane & 3u;
    const uint32_t head = wave * 16u + r;
    const uint64_t row_offset =
        uint64_t(blockIdx.x) * uint64_t(p.q_stride0) + uint64_t(head) * uint64_t(p.q_stride1);
    const uint32_t base = __builtin_amdgcn_readfirstlane(
                              static_cast<uint32_t>(reinterpret_cast<uintptr_t>(staging))) +
                          wave * 16384u;
    // Queue all16 packets without a VMEM wait between chunks. Every source and
    // destination is aligned; Q values remain their supplied BF16 bit patterns.
    for(uint32_t c = 0; c < 16u; ++c)
    {
        const uint32_t L           = c * 32u + ((quad * 8u) ^ (((r >> 2) & 1u) * 16u));
        const uint32_t col         = sb8_inv_perm_col_elems(L);
        const uint64_t address     = reinterpret_cast<uintptr_t>(p.q + row_offset + uint64_t(col));
        const uint32_t destination = base + c * 1024u;
        asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                     :
                     : "s"(destination), "v"(address)
                     : "m0", "memory");
    }
    drain();
    __syncthreads(); // all Q packets have reached their wave-private LDS region
    const uint32_t row = lane & 15u, col = (lane >> 4) * 8u;
    const uint32_t read_address = base + row * 64u + ((col * 2u) ^ (((row >> 2) & 1u) * 32u));
    opus::static_for<16>([&](auto c_) {
        constexpr uint32_t c = c_.value;
        hkm::ds_read_b128<R::k_q_vgpr_begin + c * 4u>(read_address, c * 1024u);
    });
    drain();
    __syncthreads(); // Q LDS lifetime ends before any wave starts writing KV
}

template <typename T>
__device__ __forceinline__ void
stage_kv(const Params& p, uint16_t* tile, int32_t* selected, int64_t start, int64_t end)
{
    const uint32_t tid = threadIdx.x;
    if(tid < 32)
    {
        // CSR entry and physical slot are both guarded before their load.
        int32_t slot = -1;
        if(int64_t(start) + tid < int64_t(end))
            slot = p.indices[int64_t(start) + tid];
        selected[tid] = slot;
    }
    __syncthreads();
    // Keep each load address 64 bit through multiplication. An invalid slot
    // does not produce a pointer, including when the entire cache is empty.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(p.kv) | (uint64_t(p.kv_stride0) * 2u)) & 15u) == 0;
    if(aligned)
    {
        // Eight contiguous BF16 values map to one aligned 16-byte LDS packet:
        // sb8 permutation and bank XOR preserve all intra-packet bits. This
        // reduces address/index work and global/LDS instructions by eight.
        // The PR's K ring and P registers are dead during this stage. Reuse
        // v44..v59 as four in-flight vector packets while Q64..127 and O128..255
        // remain resident. Compiler scratch stays strictly below v44.
        using R                      = HkMlaV40Regs<T>;
        constexpr uint32_t first_reg = R::k_scratch_budget;
        constexpr uint32_t packets   = 4;
        static_assert(first_reg == 44 && first_reg + packets * 4u <= R::k_q_vgpr_begin);
        static_assert((32u * 64u) % (packets * T::kNumThreads) == 0);
        for(uint32_t first = tid; first < 32u * 64u; first += packets * T::kNumThreads)
        {
            // Full EXEC initializes every lane before any predicated global
            // load. Invalid lanes thus publish zero without forming a pointer.
            opus::static_for<packets * 4u>([&](auto reg) {
                asm volatile("v_mov_b32 v[%0], 0" : : "n"(first_reg + reg.value));
            });
            opus::static_for<packets>([&](auto i_) {
                constexpr uint32_t i = i_.value, reg = first_reg + i * 4u;
                uint32_t packet = first + i * T::kNumThreads;
                // Prevent four independently strength-reduced address streams
                // from staying live across the pinned prefetch and escaping
                // compiler scratch v0..43 (the same break-CSE idiom as PR O).
                asm volatile("" : "+v"(packet));
                const uint32_t row = packet / 64u, col = (packet % 64u) * 8u;
                const int32_t slot = selected[row];
                if(slot >= 0 && int64_t(slot) < p.slots)
                {
                    const uint64_t offset = uint64_t(slot) * uint64_t(p.kv_stride0) + uint64_t(col);
                    const uint64_t address = reinterpret_cast<uintptr_t>(p.kv + offset);
                    // Flat 64-bit address; never truncate to a cache-wide
                    // buffer descriptor or a 32-bit byte offset.
                    asm volatile("global_load_dwordx4 v[%0:%1], %2, off"
                                 :
                                 : "n"(reg), "n"(reg + 3u), "v"(address)
                                 : "memory");
                }
            });
            // Some packets can issue no VMEM instruction (all lanes invalid),
            // so a fixed partial vmcnt threshold would be incorrect here.
            drain();
            opus::static_for<packets>([&](auto i_) {
                constexpr uint32_t i = i_.value, reg = first_reg + i * 4u;
                uint32_t packet = first + i * T::kNumThreads;
                // Prevent four independently strength-reduced address streams
                // from staying live across the pinned prefetch and escaping
                // compiler scratch v0..43 (the same break-CSE idiom as PR O).
                asm volatile("" : "+v"(packet));
                const uint32_t row = packet / 64u, col = (packet % 64u) * 8u;
                const uint32_t L    = sb8_perm_col_elems(col);
                const uint32_t r    = row & 15u;
                const uint32_t byte = (L / 32u * 2u + row / 16u) * 1024u + r * 64u +
                                      ((2u * (L & 31u)) ^ (((r >> 2) & 1u) * 32u));
                hkm::ds_write_b128<reg>(reinterpret_cast<uintptr_t>(tile) + byte, 0);
            });
            drain(); // retire every LDS writer before its pinned source is reused
        }
    }
    else
    {
        for(uint32_t linear = tid; linear < 32u * 512u; linear += T::kNumThreads)
        {
            const uint32_t row = linear / 512u, col = linear % 512u;
            const int32_t slot = selected[row];
            uint16_t value     = 0;
            if(slot >= 0 && int64_t(slot) < p.slots)
                value = p.kv[uint64_t(slot) * uint64_t(p.kv_stride0) + uint64_t(col)];
            const uint32_t L    = sb8_perm_col_elems(col);
            const uint32_t r    = row & 15u;
            const uint32_t byte = (L / 32u * 2u + row / 16u) * 1024u + r * 64u +
                                  ((2u * (L & 31u)) ^ (((r >> 2) & 1u) * 32u));
            tile[byte / 2u] = value;
        }
    }
    drain();
    __syncthreads();
}

// Gfx950 full-64-bit global-to-LDS DMA. M0 supplies a wave-uniform destination
// base; lane t lands sixteen bytes at M0+t*16. The PR subblock mapping undoes
// bank XOR and sb8 on the global source side, making the destination contiguous.
// Counter semantics: DMA completion is VMEM (vmcnt), not LGKM. This helper only
// retires ordinary LDS operations, deliberately leaving valid DMA in flight.
template <typename T>
__device__ __forceinline__ void
stage_kv_async(const Params& p, uint16_t* tile, int32_t* selected, int64_t start, int64_t end)
{
    const uint32_t tid = threadIdx.x;
    if(tid < 32u)
    {
        int32_t slot = -1;
        if(start + tid < end)
            slot = p.indices[start + tid];
        selected[tid] = slot;
    }
    __syncthreads(); // index publication precedes every DMA; none is issued yet
    const uint32_t lane = tid & 63u;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const uint32_t base =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile)));
    for(uint32_t block = wave; block < 32u; block += T::kNumWarps)
    {
        const uint32_t r = lane >> 2, quad = lane & 3u;
        const uint32_t row         = (block & 1u) * 16u + r;
        const uint32_t L           = (block >> 1) * 32u + ((quad * 8u) ^ (((r >> 2) & 1u) * 16u));
        const uint32_t col         = sb8_inv_perm_col_elems(L);
        const int32_t slot         = selected[row];
        const uint32_t destination = base + block * 1024u;
        if(slot >= 0 && int64_t(slot) < p.slots)
        {
            const uint64_t offset  = uint64_t(slot) * uint64_t(p.kv_stride0) + uint64_t(col);
            const uint64_t address = reinterpret_cast<uintptr_t>(p.kv + offset);
            // M0->VMEM hazard gap follows LLVM's gfx950 DMA lowering. Source
            // addresses are only formed inside the validity predicate.
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                         :
                         : "s"(destination), "v"(address)
                         : "m0", "memory");
        }
        else
        {
            // Inactive DMA lanes do not zero memory. Clear exactly their
            // private sixteen-byte landing so stale previous-pong data cannot
            // enter QK/PV, including all-invalid first/middle/final tiles.
            const v4ui zero                      = {0, 0, 0, 0};
            *reinterpret_cast<v4ui*>(reinterpret_cast<uintptr_t>(tile) + block * 1024u +
                                     lane * 16u) = zero;
        }
    }
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    __builtin_amdgcn_sched_barrier(0);
}

// The QK ring and issue/wait/MFMA order are the PR m16x4 schedule. K comes from
// the same sb8 LDS layout; Q uses exactly the corresponding D-axis permutation.
template <typename T>
__device__ __forceinline__ void qk_gemm(uintptr_t tile)
{
    using R         = HkMlaV40Regs<T>;
    using mfma_ab_t = hk::bf16;
    typename R::p_comp_lo_t lo;
    typename R::p_comp_hi_t hi;
    auto ring = [](uint32_t s) constexpr {
        return s % 3u == 0u ? R::k_k0_begin : s % 3u == 1u ? R::k_k1_begin : R::k_k2_begin;
    };
    auto issue = [&]<uint32_t S>() {
        constexpr uint32_t base = ring(S);
        using kr = hkdart::split_many_t<hkdart::type_list<hkdart::range<base, base + 3u>>, 4>;
        hk::art<mfma_ab_t, 16, 32, hk::row_l, hk::rt_16x32_s, kr> k;
        Bf16KvManager<T>::template load_k_to_gpr<(S % 2u) * 16u, (S / 2u) * 32u>(k, tile);
    };
    auto mma = [&]<uint32_t S>() {
        constexpr uint32_t base = ring(S), qb = R::k_q_vgpr_begin + (S / 2u) * 4u;
        using kr = hkdart::split_many_t<hkdart::type_list<hkdart::range<base, base + 3u>>, 4>;
        using qr = hkdart::split_many_t<hkdart::type_list<hkdart::range<qb, qb + 3u>>, 4>;
        hk::art<mfma_ab_t, 16, 32, hk::row_l, hk::rt_16x32_s, kr> k;
        hk::art<mfma_ab_t, 16, 32, hk::row_l, hk::rt_16x32_s, qr> q;
        if constexpr(S == 0)
            hk::mma_ABt(lo, k, q);
        else if constexpr(S == 1)
            hk::mma_ABt(hi, k, q);
        else if constexpr((S % 2u) == 0)
            hk::mma_ABt(lo, k, q, lo);
        else
            hk::mma_ABt(hi, k, q, hi);
    };
    opus::static_for<3>([&](auto s) { issue.template operator()<s.value>(); });
    opus::static_for<32>([&](auto s_) {
        constexpr uint32_t s = s_.value, remain = 31u - s;
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(remain < 2u ? remain : 2u, -1));
        mma.template operator()<s>();
        if constexpr(s + 3u < 32u)
            issue.template operator()<s + 3u>();
    });
    // The PR hides the final MFMA-to-VALU RAW latency behind its next-KV
    // conversion/staging. Our synchronous BF16 path has no such intervening
    // work. LLVM cannot see the numeric pinned-register dependence, and an
    // lgkmcnt/vmcnt drain does not wait for MFMA arithmetic. Reserve a
    // conservative 16 issue cycles before softmax first reads p56..p63.
    // Without this gap the final D-axis slice is missing in one-hot Q tests.
    asm volatile("s_nop 7\n\ts_nop 7");
    __builtin_amdgcn_sched_barrier(0);
}

template <typename T>
__device__ __forceinline__ void
softmax(const Params& p, const int32_t* selected, float& maximum, float& denominator)
{
    using R               = HkMlaV40Regs<T>;
    constexpr float log2e = 1.4426950408889634074f;
    const uint32_t group  = (threadIdx.x & 63u) >> 4;
    // QK's K ring is dead after its explicit MFMA latency gap. Prefetch the
    // eight mask indices into v44..51 while keeping live scores in v56..63.
    // One LGKM drain replaces eight serialized scalar reads; pending next-KV
    // DMA (VMEM counter) remains in flight. PV may reuse the K ring afterward.
    constexpr uint32_t mask_reg = R::k_scratch_budget;
    static_assert(mask_reg + 8u <= R::k_p_comp_begin);
    const uint32_t mask_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(selected)) + group * 4u * sizeof(int32_t);
    hkm::ds_read_b128<mask_reg>(mask_address, 0);
    hkm::ds_read_b128<mask_reg + 4u>(mask_address, 16u * sizeof(int32_t));
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    __builtin_amdgcn_sched_barrier(0);
    float local_max = -INFINITY;
    opus::static_for<8>([&](auto i_) {
        constexpr uint32_t i = i_.value;
        const int32_t slot   = read_i32<mask_reg + i>();
        float score          = read_f32<R::k_p_comp_begin + i>();
        // Scaling is applied after the supplied BF16 QK FP32 accumulation.
        score = (score * p.scale) * log2e;
        if(slot < 0 || int64_t(slot) >= p.slots)
            score = -INFINITY;
        write_f32<R::k_p_comp_begin + i>(score);
        local_max = fmaxf(local_max, score);
    });
    local_max           = fmaxf(local_max, __shfl_xor(local_max, 16, 64));
    local_max           = fmaxf(local_max, __shfl_xor(local_max, 32, 64));
    const float new_max = fmaxf(maximum, local_max);
    const float alpha   = denominator > 0.0f ? __builtin_amdgcn_exp2f(maximum - new_max) : 0.0f;
    maximum             = new_max;
    typename R::oaccu_t o;
    hk::mul_vgpr(o, o, alpha);
    float sum = 0.0f;
    opus::static_for<8>([&](auto i_) {
        constexpr uint32_t i = i_.value;
        const float score    = read_f32<R::k_p_comp_begin + i>();
        // All-empty tiles never evaluate exp2(-inf - -inf).
        const float weight = score == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(score - maximum);
        write_f32<R::k_p_comp_begin + i>(weight);
        sum += weight;
    });
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    denominator = denominator * alpha + sum;
    // Exactly the upstream low-to-high P packing order and BF16 MFMA operands.
    pack_2f32_to_bf16_pair_pinned<R::k_p_mfma_begin + 0, R::k_p_comp_begin + 0>();
    pack_2f32_to_bf16_pair_pinned<R::k_p_mfma_begin + 1, R::k_p_comp_begin + 2>();
    pack_2f32_to_bf16_pair_pinned<R::k_p_mfma_begin + 2, R::k_p_comp_begin + 4>();
    pack_2f32_to_bf16_pair_pinned<R::k_p_mfma_begin + 3, R::k_p_comp_begin + 6>();
}

template <int Heads, bool Split>
__device__ __forceinline__ void sparse_mla_legacy(Params p)
{
    using T = Traits<Heads>;
    using R = HkMlaV40Regs<T>;
    static_assert(R::k_scratch_budget == 44);
    // Q/O live throughout; K/P ranges intentionally overlay PV operands.
    hkdart::clobber<typename R::q_vgpr_ranges>();
    hkdart::clobber<typename R::p_comp_ranges>();
    hkdart::clobber<typename R::o_ranges>();
    hkdart::clobber<typename R::kv_top_ranges>();
    hkdart::clobber<typename R::kv_bot_ranges>();
    hkdart::clobber<typename R::kv_alt_top_ranges>();
    extern __shared__ __align__(16) unsigned char storage[];
    auto* tile                    = reinterpret_cast<uint16_t*>(storage);
    auto* next_tile               = reinterpret_cast<uint16_t*>(storage + T::kKvBytes);
    auto* selected                = reinterpret_cast<int32_t*>(storage + T::kIndexOffset);
    auto* next_selected           = selected + 32;
    const uintptr_t output_bounce = reinterpret_cast<uintptr_t>(storage);
    const uint32_t query = blockIdx.x, partition = blockIdx.y;
    const int32_t row_start = p.indptr[query], row_end = p.indptr[query + 1u];
    const int32_t length = row_end - row_start;
    const int32_t begin  = row_start + int64_t(length) * partition / p.splits;
    const int32_t end    = row_start + int64_t(length) * (partition + 1u) / p.splits;
    load_q_dma<T>(p, tile);
    typename R::oaccu_t o;
    hk::zero(o);
    float maximum = -INFINITY, denominator = 0.0f;
    Bf16KvManager<T> manager;
    const bool async_aligned =
        T::kKvBuffers == 2u &&
        ((reinterpret_cast<uintptr_t>(p.kv) | (uint64_t(p.kv_stride0) * 2u)) & 15u) == 0;
    if(async_aligned && begin < end)
    {
        stage_kv_async<T>(p, tile, selected, begin, end);
        drain();
        __syncthreads(); // tile0 prologue: every DMA/zero has completed
    }
    for(int64_t pos = begin; pos < end; pos += 32)
    {
        if(async_aligned)
        {
            if(pos + 32 < end)
                stage_kv_async<T>(p, next_tile, next_selected, pos + 32, end);
            // No VMEM wait here: next-pong DMA overlaps current-pong compute.
        }
        else
        {
            // Preserve the proven BF16-alignment-only path synchronously.
            stage_kv<T>(p, tile, selected, pos, end);
        }
        const uintptr_t tile_addr = reinterpret_cast<uintptr_t>(tile);
        qk_gemm<T>(tile_addr);
        softmax<T>(p, selected, maximum, denominator);
        // Reuse PR's exact PV four-slot LDS/MFMA schedule. Rescale occurred once
        // above, so this call only accumulates into initialized FP32 O registers.
        pv_gemm<false, false, T>(manager, tile_addr, 1.0f);
        drain();         // next DMA complete AND every current LDS reader retired
        __syncthreads(); // publish next tile; old current may now become next
        if(async_aligned)
        {
            auto* old_tile     = tile;
            tile               = next_tile;
            next_tile          = old_tile;
            auto* old_selected = selected;
            selected           = next_selected;
            next_selected      = old_selected;
        }
    }
    const float inv_sum = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    hk::mul_vgpr(o, o, inv_sum);
    constexpr float ln2 = 0.6931471805599453094f;
    const float lse =
        denominator > 0.0f ? (maximum + __builtin_amdgcn_logf(denominator)) * ln2 : -INFINITY;
    const uint32_t lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const uint64_t slot = Split ? uint64_t(query) * uint64_t(p.splits) + partition : query;
    if(lane < 16u)
    {
        const uint64_t h = uint64_t(wave) * 16u + lane;
        if constexpr(Split)
            p.partial_lse[slot * Heads + h] = lse;
        else if(p.lse)
            p.lse[slot * Heads + h] = lse;
    }
    // Rebase the descriptor to one query/partition (64-bit) before the upstream
    // manager's local 32-bit output offsets. This also works above 4 GiB.
    if constexpr(Split)
    {
        OManager32bitsV4Gen1Swizzle<T, float> output;
        float* dst = p.partial_o + slot * Heads * 512u;
        opus::static_for<8>([&](auto i) {
            output.template output_to_vram_pair<R::k_o_begin + i.value * 16u, i.value * 64u, false>(
                dst, wave, 0, 0, output_bounce, Heads);
            drain();
        });
    }
    else
    {
        OManager16bitsV4Gen1Swizzle<T, hk::bf16> output;
        auto* dst = reinterpret_cast<hk::bf16*>(p.out + slot * Heads * 512u);
        opus::static_for<8>([&](auto i) {
            output.template output_to_vram_pair<R::k_o_begin + i.value * 16u, i.value * 64u, false>(
                dst, wave, 0, 0, output_bounce, Heads);
            drain();
        });
    }
}

// Guarded normalized-FP32 partial reducer. Each thread owns one D element;
// all-empty partitions contribute exactly zero without exp(-inf - -inf).
__global__ __launch_bounds__(256) void reduce(Params p, int heads)
{
    const uint32_t query = blockIdx.x, head = blockIdx.y;
    const uint32_t d0    = threadIdx.x;
    const uint64_t first = uint64_t(query) * uint64_t(p.splits) * uint64_t(heads) + head;
    float maximum        = -INFINITY;
    for(int s = 0; s < p.splits; ++s)
        maximum = fmaxf(maximum, p.partial_lse[first + uint64_t(s) * heads]);
    float sum = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for(int s = 0; s < p.splits; ++s)
    {
        const uint64_t row = first + uint64_t(s) * heads;
        const float value  = p.partial_lse[row];
        const float weight = value == -INFINITY ? 0.0f : expf(value - maximum);
        sum += weight;
        a0 += weight * p.partial_o[row * 512u + d0];
        a1 += weight * p.partial_o[row * 512u + d0 + 256u];
    }
    const float reciprocal = sum > 0.0f ? 1.0f / sum : 0.0f;
    const uint64_t out_row = (uint64_t(query) * uint64_t(heads) + head) * 512u;
    // Explicit nearest-even conversion, including subnormal and tie semantics.
    uint32_t pair;
    const float v0 = a0 * reciprocal, v1 = a1 * reciprocal;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pair) : "v"(v0), "v"(v1));
    p.out[out_row + d0]        = uint16_t(pair);
    p.out[out_row + d0 + 256u] = uint16_t(pair >> 16);
    if(threadIdx.x == 0 && p.lse)
        p.lse[uint64_t(query) * heads + head] = sum > 0.0f ? maximum + logf(sum) : -INFINITY;
}

} // namespace sparse_mla_bf16
