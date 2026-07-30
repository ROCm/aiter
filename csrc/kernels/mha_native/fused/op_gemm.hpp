#pragma once
#include "op_lds.hpp"

// ================================================================
// op_gemm.hpp — the two GEMMs of the D64 FMHA fwd kernel
// ================================================================
//
// ROLE IN THE PIPELINE
//   GEMM0:  S_acc = Q . K^T   (Q . K, scores)          -> feeds softmax
//   GEMM1:  O_acc += P . V    (attention-weighted sum) -> feeds epilogue
//   Both run on v_mfma_f32_32x32x8_bf16 (the CDNA3 32x32x8 bf16 MFMA). A-operand
//   comes from LDS (staged by op_lds.hpp); B-operand comes from registers.
//
// MFMA SHAPE (32x32x8 bf16, 1k variant)
//   One instruction computes C[32x32] += A[32x8] . B[8x32] over a wave (64
//   lanes). Each lane supplies a v4h (4 bf16) slice of A and of B, and holds 16
//   fp32 accumulators (v16f) of C. Two MFMA "passes" of 4 bf16 cover the 8-deep
//   K of one MFMA; multiple ksteps walk the full contraction dimension.
//
// UNIVERSAL "TransposedC" REGISTER LAYOUT  (the single most important fact)
//   The accumulator C is laid out so each lane owns ONE M-row across all its 16
//   registers, and the register index walks the free (N for GEMM0, hdim for
//   GEMM1) dimension:
//       m_row = (lane%32) + 32*warp                         [one row per lane]
//       free(r) = (r/8)*16 + (lane/32)*8 + (r%8),  r=0..15  [the other dim]
//   Because the MFMA output maps each lane to a column of B, choosing B = Q (for
//   GEMM0) / P (for GEMM1) is what makes "each lane owns one M-row" true — that
//   is WHY this kernel uses the CK convention A=LDS, B=register rather than the
//   other assignment. softmax (op_softmax.hpp) and the epilogue depend on this
//   exact distribution.
//
// k_sub PAIRING
//   k_sub = lane/32 splits the 64-lane wave into two halves. Lanes in the same
//   k_sub half hold the SAME 32 free-dim columns (the two halves are
//   complementary). GEMM0 reads K for its half; the softmax later merges the two
//   halves with one ds_bpermute (no butterfly).
//
// SWIZZLE
//   GEMM0 applies SwizzleA (swap bits 2 and 3 of the K column index, see swz())
//   when reading K from LDS, which reorders the MFMA output into the
//   groups-of-8 column pattern CK's golden S_acc uses. GEMM1 does NOT swizzle V.
//
// bf16 CASTS
//   All fp32->bf16 conversions here TRUNCATE (drop the low 16 bits), not
//   round-to-nearest-even. This matches CK and keeps ISA/numerical parity.

typedef float v16f __attribute__((ext_vector_type(16)));
typedef short v4h __attribute__((ext_vector_type(4)));

// Pack two dwords (4 bf16) into v4h, the per-lane operand slice an MFMA pass
// consumes. lo/hi are two adjacent dwords (4 bf16 total) of K/Q/V/P.
__device__ __forceinline__ v4h pack_short4(int lo, int hi) {
    v4h r;
    __builtin_memcpy(&r, &lo, 4);
    __builtin_memcpy(reinterpret_cast<char*>(&r) + 4, &hi, 4);
    return r;
}

// SwizzleA: swap bits 2 and 3 of x (x in [0,32)).
// Applied to the K(A) column index in GEMM0 so the MFMA deposits S_acc in the
// groups-of-8 N-column pattern of CK's golden output (see the n_col formula in
// op_softmax.hpp). Without the swap the columns would land permuted relative to
// what softmax/epilogue expect. It is a pure index remap — no data is moved.
__device__ __forceinline__ int swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

// K LDS element offset (padded layout, single-buffer relative).
// READER side of the layout async_copy_k_subtile() WRITES: j = seqlen_k row
// (0..63), d = headdim (0..63). The odd strides (136, 544) are the padding that
// keeps consecutive ds_reads on distinct LDS banks (avoids bank conflicts).
// Returns a bf16-element offset; callers multiply by 2 for bytes.
// Verified in Phase 0/1 K1/K2:
//   offset(j,d) = (j%4)*136 + ((j/4)%4)*32 + (j/16)*544 + (d%32) + (d/32)*2304
__device__ __forceinline__ int lds_elem_offset(int j, int d) {
    return (j % 4) * 136 + ((j / 4) % 4) * 32 + (j / 16) * 544
         + (d % 32) + (d / 32) * 2304;
}

// V LDS element offset (padded layout, single-buffer relative).
// READER side of the layout store_v_to_lds() WRITES: n = seqlen_k (only n%32
// matters within a 32-row sub-tile), d = headdim. The 72 = kPaddedRowStride
// padding again spreads ds_reads across banks. bf16-element offset.
// Verified in Phase 1 K5/K6:
//   k = n % 32; offset(n,d) = (k/8)*576 + (d/8)*72 + (d%8)*8 + (k%8)
__device__ __forceinline__ int v_lds_elem_offset(int n, int d) {
    int k = n % 32;
    return (k / 8) * 576 + (d / 8) * 72 + (d % 8) * 8 + (k % 8);
}

// Convert float to bf16 by TRUNCATION (drop low 16 bits; no rounding). Matches
// CK; used by the legacy GEMM1 path. The live GEMM1 truncates inline via
// v_perm_b32 selector 0x07060302, which picks the same high 16 bits.
__device__ __forceinline__ uint16_t f32_to_bf16_trunc(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return static_cast<uint16_t>(u >> 16);
}

// Convert bf16 (in uint16_t) to float.
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return __builtin_bit_cast(float, u);
}

// ---- Accumulator helpers ----

__device__ __forceinline__ void clear_acc(v16f& acc) {
    acc = v16f{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
}

// ================================================================
// Phase 2: GEMM0 — S_acc = Q × K^T (sub-tile granularity)
// ================================================================

// Extract the Q register slice for a given K(headdim) sub-tile index.
// Q was loaded (in pipeline.hpp) as 4 v4i covering the full 64 hdim, with this
// lane holding hdim = kstep*16 + k_sub*8 + 0..7 in q_regs[kstep]. The headdim
// contraction is processed in two 32-wide sub-tiles, each consuming 2 ksteps:
//   k_subtile_idx=0 (hdim 0..31)  -> ksteps 0,1
//   k_subtile_idx=1 (hdim 32..63) -> ksteps 2,3
__device__ __forceinline__ const v4i* slice_q(const v4i* q_regs, int k_subtile_idx) {
    return q_regs + k_subtile_idx * 2;
}

// One sub-tile of GEMM0: read K from LDS, execute 2 ksteps x 2 passes x 2 N-tiles
// = 8 MFMA. Two calls (k_subtile_idx 0 then 1) accumulate the full 64-deep
// headdim contraction into s_acc_n0 / s_acc_n1.
//
// OPERANDS: K is the A-operand (from LDS), Q is the B-operand (from registers).
// B=Q is deliberate: per the TransposedC layout, the lane->B-column mapping makes
// each lane own one Q M-row of the result S_acc (see file header).
//
// SwizzleA: each lane reads K at column seqk = swz(lane & 31). The two N-tiles
// are the two 32-wide halves of the 64-column score tile (seqk 0..31 and 32..63).
//
// Each kstep covers 16 hdim, split into two MFMA passes of 4 bf16 each (the MFMA
// is 8-deep in K = two 4-bf16 passes).
//
// Parameters:
//   s_acc_n0, s_acc_n1: accumulators for N-tile 0 (seqk 0..31) and 1 (32..63)
//   q_slice: the 2 v4i for this sub-tile (from slice_q())
//   lds:     LDS base pointer
//   buf_idx: which rotating LDS buffer holds this K sub-tile
__device__ __forceinline__ void gemm0_subtile(
    v16f& s_acc_n0, v16f& s_acc_n1,
    const v4i* q_slice,
    char* lds,
    int buf_idx)
{
    const int lane_id = threadIdx.x & 63;
    const int k_sub   = lane_id >> 5;   // which 32-lane half (selects hdim octet)

    // SwizzleA applied to the K column this lane reads. seqk0 is in N-tile 0
    // (0..31), seqk1 the same column shifted into N-tile 1 (32..63).
    const int seqk0 = swz(lane_id & 31);        // SwizzleA: N-tile 0
    const int seqk1 = 32 + swz(lane_id & 31);   // SwizzleA: N-tile 1
    const int buf_byte_base = buf_base_bytes(buf_idx);

    #pragma unroll
    for (int kstep = 0; kstep < 2; ++kstep) {
        // This lane's 8-wide headdim slice for this kstep (k_sub picks the octet).
        const int d_base = kstep * 16 + k_sub * 8;

        // K reads from LDS (A operand) for both N-tiles. v4i = 4 dwords = 8 bf16
        // = the full 8-deep K of one MFMA (consumed as two 4-bf16 passes below).
        const v4i k0 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + lds_elem_offset(seqk0, d_base) * 2);
        const v4i k1 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + lds_elem_offset(seqk1, d_base) * 2);
        const v4i q = q_slice[kstep];

        // Pass 0: hdim d_base+0..3 (low 4 bf16 of the 8-deep K).
        {
            v4h a0 = pack_short4(k0[0], k0[1]);
            v4h a1 = pack_short4(k1[0], k1[1]);
            v4h b  = pack_short4(q[0],  q[1]);
            // a from K (LDS), b from Q (reg); accumulate in place. One MFMA per
            // N-tile shares the same Q b-operand.
            s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, s_acc_n0, 0, 0, 0);
            s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, s_acc_n1, 0, 0, 0);
        }
        // Pass 1: hdim d_base+4..7 (high 4 bf16 of the 8-deep K).
        {
            v4h a0 = pack_short4(k0[2], k0[3]);
            v4h a1 = pack_short4(k1[2], k1[3]);
            v4h b  = pack_short4(q[2],  q[3]);
            s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, s_acc_n0, 0, 0, 0);
            s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, s_acc_n1, 0, 0, 0);
        }
    }
}

// ================================================================
// Phase 2: GEMM1 — O_acc += P × V (sub-tile granularity)
// ================================================================

// Pack P (the fp32 softmax probabilities, in accumulator layout) to bf16 for the
// MFMA B-operand of GEMM1. P is B (not A) for the same TransposedC reason as Q in
// GEMM0: B=P makes each lane own one M-row of O_acc.
//
// TRUNCATION via v_perm_b32 selector 0x07060302: the selector picks bytes 3,2 of
// the first source and 7,6 of the second — i.e. the HIGH 16 bits of each fp32 =
// bf16 truncation, packing two fp32 into one dword. No round-to-nearest.
// Produces 4 v4h per N-tile half.
//
// (This standalone packer is kept for reference; the live GEMM1 below packs P
// inline so the compiler can interleave packing with MFMA co-execution.)
//   v_subtile_idx=0: P from p_n0 (seqk 0..31);  =1: P from p_n1 (seqk 32..63)
__device__ __forceinline__ void pack_p_subtile(
    v4h (&p_packed)[4],
    const v16f& p_half)
{
    constexpr unsigned kFp32ToBf16Sel = 0x07060302;
    const auto* u = reinterpret_cast<const unsigned*>(&p_half);

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        unsigned lo = __builtin_amdgcn_perm(
            u[4 * s + 1], u[4 * s], kFp32ToBf16Sel);
        unsigned hi = __builtin_amdgcn_perm(
            u[4 * s + 3], u[4 * s + 2], kFp32ToBf16Sel);
        p_packed[s] = pack_short4(lo, hi);
    }
}

// Extract the P slice for a given V sub-tile index. In GEMM1 the contraction is
// over seqlen_k, which P splits into two N-tile halves (p_n0 = seqk 0..31,
// p_n1 = seqk 32..63 — exactly the two GEMM0 accumulators). One gemm1_subtile
// call per half walks the full 64-deep seqlen_k contraction into O_acc.
__device__ __forceinline__ const v16f& slice_p(
    const v16f& p_n0, const v16f& p_n1, int v_subtile_idx)
{
    return (v_subtile_idx == 0) ? p_n0 : p_n1;
}

// One sub-tile of GEMM1: O_acc += P_half . V. Reads V from LDS (A-operand, NOT
// swizzled), packs P to bf16 inline (B-operand), executes 8 MFMA
// (2 ksteps x 2 passes x 2 hdim-tiles). Fusing the pack with the MFMA lets the
// compiler overlap the v_perm packing VALU with MFMA co-execution.
//
// o_acc_d0 / o_acc_d1 are the two 32-wide hdim halves of the output O tile
// (hdim 0..31 and 32..63), in the same TransposedC layout as GEMM0's S_acc.
__device__ __forceinline__ void gemm1_subtile(
    v16f& o_acc_d0, v16f& o_acc_d1,
    const v16f& p_half,
    char* lds,
    int buf_idx)
{
    constexpr unsigned kFp32ToBf16Sel = 0x07060302;  // bf16-truncate + pack (see pack_p_subtile)
    const int lane_id = threadIdx.x & 63;
    const int k_sub   = lane_id >> 5;       // selects the seqlen_k octet
    const int hdim_pos = lane_id & 31;      // this lane's hdim column (0..31)
    const int buf_byte_base = buf_base_bytes(buf_idx);
    const auto* u = reinterpret_cast<const unsigned*>(&p_half);  // P as raw dwords

    #pragma unroll
    for (int kstep = 0; kstep < 2; ++kstep) {
        // This lane's 8-deep seqlen_k slice for this kstep.
        const int seqk_local = kstep * 16 + k_sub * 8;
        // V from LDS (A-operand), one read per hdim half (cols 0..31 and 32..63).
        // NO SwizzleA here — V is read in straight layout (unlike K in GEMM0).
        const v4i v0 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + v_lds_elem_offset(seqk_local, hdim_pos) * 2);
        const v4i v1 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + v_lds_elem_offset(seqk_local, hdim_pos + 32) * 2);

        // Pass 0: pack P[s] inline (s = kstep*2), then MFMA the low 4 bf16.
        {
            int s = kstep * 2;
            unsigned lo = __builtin_amdgcn_perm(u[4*s+1], u[4*s], kFp32ToBf16Sel);
            unsigned hi = __builtin_amdgcn_perm(u[4*s+3], u[4*s+2], kFp32ToBf16Sel);
            v4h b_val = pack_short4(lo, hi);
            v4h a0 = pack_short4(v0[0], v0[1]);   // V hdim-half 0
            v4h a1 = pack_short4(v1[0], v1[1]);   // V hdim-half 1
            // a from V (LDS), b from packed P; the two MFMA share b_val.
            o_acc_d0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b_val, o_acc_d0, 0, 0, 0);
            o_acc_d1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b_val, o_acc_d1, 0, 0, 0);
        }

        // Pass 1: pack P[s] inline (s = kstep*2+1), then MFMA the high 4 bf16.
        {
            int s = kstep * 2 + 1;
            unsigned lo = __builtin_amdgcn_perm(u[4*s+1], u[4*s], kFp32ToBf16Sel);
            unsigned hi = __builtin_amdgcn_perm(u[4*s+3], u[4*s+2], kFp32ToBf16Sel);
            v4h b_val = pack_short4(lo, hi);
            v4h a0 = pack_short4(v0[2], v0[3]);
            v4h a1 = pack_short4(v1[2], v1[3]);
            o_acc_d0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b_val, o_acc_d0, 0, 0, 0);
            o_acc_d1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b_val, o_acc_d1, 0, 0, 0);
        }
    }
}

// ================================================================
// Legacy functions — DEAD CODE, kept for reference (no live caller).
//
// gemm0() here is the pre-sub-tile GEMM0 (full 64-deep headdim in one call, old
// k_lds_offset layout, no SwizzleA). gemm1_bpermute() is the pre-MFMA GEMM1: a
// scalar FMA loop that broadcasts P across lanes with ds_bpermute instead of the
// P.V MFMA. Both are superseded by gemm0_subtile / gemm1_subtile above. They
// were called by an old `_device.hpp` entry that has since been removed; nothing
// in this kernel references them now (the live pipeline.hpp calls
// gemm0_subtile). `inline` so they emit no code while uncalled. New readers can
// skip this block.
// ================================================================

__device__ inline void gemm0(v16f& s_acc_n0, v16f& s_acc_n1,
                              const v4i* q_regs,
                              char* lds,
                              int lds_buf0_bytes,
                              int lds_buf1_bytes,
                              int lane_id) {
    int n_local = lane_id & 31;
    int k_sub   = lane_id >> 5;

    auto do_k0 = [&](int lds_buf_bytes, int q_base) {
        for (int kstep = 0; kstep < 4; kstep++) {
            int k_start = kstep * 8;
            int off_n0 = lds_buf_bytes + k_lds_offset(n_local, k_start) * 2;
            v4i k_n0 = *reinterpret_cast<const v4i*>(lds + off_n0);
            int off_n1 = lds_buf_bytes + k_lds_offset(32 + n_local, k_start) * 2;
            v4i k_n1 = *reinterpret_cast<const v4i*>(lds + off_n1);
            v4h a_n0 = pack_short4(k_n0[k_sub * 2], k_n0[k_sub * 2 + 1]);
            v4h a_n1 = pack_short4(k_n1[k_sub * 2], k_n1[k_sub * 2 + 1]);
            v4i q_reg = q_regs[q_base + kstep];
            v4h b_val = pack_short4(q_reg[k_sub * 2], q_reg[k_sub * 2 + 1]);
            s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_n0, b_val, s_acc_n0, 0, 0, 0);
            s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_n1, b_val, s_acc_n1, 0, 0, 0);
        }
    };
    do_k0(lds_buf0_bytes, 0);
    __builtin_amdgcn_sched_barrier(0);
    do_k0(lds_buf1_bytes, 4);
}

__device__ inline void gemm1_bpermute(v16f& o_acc_n0, v16f& o_acc_n1,
                                       const v16f& p_n0, const v16f& p_n1,
                                       const __hip_bfloat16* v_base,
                                       int stride_v,
                                       int kv_offset,
                                       int seqlen_k,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int n_pos = lane_id & 31;

    for (int j = 0; j < kN0; j++) {
        int v_row = kv_offset + j;
        float v_val_n0 = 0.0f, v_val_n1 = 0.0f;
        if (v_row < seqlen_k) {
            const __hip_bfloat16* v_row_ptr = v_base + static_cast<int64_t>(v_row) * stride_v;
            uint16_t bf_n0, bf_n1;
            __builtin_memcpy(&bf_n0, &v_row_ptr[n_pos], 2);
            __builtin_memcpy(&bf_n1, &v_row_ptr[32 + n_pos], 2);
            v_val_n0 = bf16_to_f32(bf_n0);
            v_val_n1 = bf16_to_f32(bf_n1);
        }
        int src_lane = k_sub * 32 + (j & 31);
        for (int i = 0; i < 16; i++) {
            float p_val;
            if (j < 32) {
                p_val = bpermute_f32(src_lane, p_n0[i]);
            } else {
                p_val = bpermute_f32(src_lane, p_n1[i]);
            }
            float p_trunc = bf16_to_f32(f32_to_bf16_trunc(p_val));
            o_acc_n0[i] += p_trunc * v_val_n0;
            o_acc_n1[i] += p_trunc * v_val_n1;
        }
    }
}
