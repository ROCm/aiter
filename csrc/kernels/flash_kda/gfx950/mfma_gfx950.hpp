// gfx950-only MFMA helpers for FlashKDA scan kernels.
//
// Keep these primitives out of the shared CDNA3/CDNA4 header: a translation
// unit may contain both gfx942 and gfx950 device passes, and only gfx950 has
// v_mfma_f32_16x16x32_bf16.  The contract helper below deliberately retains
// an x16 fallback so an arch-private launcher can still be compiled into a
// multi-architecture fat binary.
#pragma once

#include "../mfma.hpp"

namespace flashkda_hip::gfx950 {

// Issue CDNA4's flat global-to-LDS DMA without exposing an LLVM LDS write.
// The builtin form conservatively inserts vmcnt(0) before the first unrelated
// LDS read, which serializes a double-buffered pipeline.  Keeping the issue in
// one opaque asm block lets callers overlap it with current-arena MFMA and
// fence it explicitly before publishing the destination arena.
template <int Bytes, typename T, typename U>
__device__ __forceinline__ void global_to_lds_async(
        T* lds, const U* global) {
    static_assert(Bytes == 4 || Bytes == 16,
                  "gfx950 global-to-LDS supports dword or dwordx4 here");
#if defined(__gfx950__)
    const unsigned lds_addr = __builtin_amdgcn_readfirstlane(
        static_cast<unsigned>(reinterpret_cast<uintptr_t>(lds)));
    const uintptr_t global_addr = reinterpret_cast<uintptr_t>(global);
    if constexpr (Bytes == 4) {
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "global_load_lds_dword %1, off"
            : : "s"(lds_addr), "v"(global_addr) : "memory");
    } else {
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "global_load_lds_dwordx4 %1, off"
            : : "s"(lds_addr), "v"(global_addr) : "memory");
    }
#else
    // Semantic fallback for a gfx950 launcher carried in a multi-arch device
    // image.  Runtime policy never selects this operator off gfx950, but a
    // real copy keeps fat-binary builds auditable and safe to invoke.  The
    // native instruction implicitly advances the LDS destination by lane;
    // spell that addressing out for the ordinary global->VGPR->LDS path.
    constexpr int Dwords = Bytes / sizeof(uint32_t);
    const int lane = threadIdx.x & 63;
    // The underlying objects may be bf16 or float.  A may_alias word keeps
    // the bitwise fallback legal under -O3 strict aliasing while preserving
    // the vector-width code generation used by the fat-binary device pass.
    using alias_u32 = uint32_t __attribute__((__may_alias__));
    auto* dst = reinterpret_cast<alias_u32*>(lds) + lane * Dwords;
    const auto* src = reinterpret_cast<const alias_u32*>(global);
    #pragma unroll
    for (int i = 0; i < Dwords; ++i)
        dst[i] = src[i];
#endif
}

// D[m,n] = sum_k A[m,k] * B[n,k] for row-major bf16 operands in LDS.
//
// gfx950's 16x16x32 instruction consumes eight bf16 values per lane.  Lane
// group g=(lane>>4) owns k=[8g,8g+8), while lane&15 selects the matrix row.
// The output fragment is identical to the established x16 instruction, so
// callers do not need to change their output publication order.
template <int Kd, int LDA, int LDB>
__device__ __forceinline__ f32x4 contract_last_x32(
        const __bf16* __restrict__ a,
        const __bf16* __restrict__ b,
        int lane) {
    static_assert(Kd > 0 && Kd % 32 == 0,
                  "gfx950 x32 contraction requires K to be a multiple of 32");

    const int row = lane & 15;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

#if defined(__gfx950__)
    const int kb = (lane >> 4) * 8;
    #pragma unroll
    for (int k0 = 0; k0 < Kd; k0 += 32) {
        bf16x8 af, bf;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            af[i] = a[row * LDA + k0 + kb + i];
            bf[i] = b[row * LDB + k0 + kb + i];
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            af, bf, acc, 0, 0, 0);
    }
#else
    // This path is never selected at runtime by the gfx950 launcher.  It is
    // nevertheless a complete semantic fallback so the same TU can carry a
    // gfx942 device image when setup.py requests both supported targets.
    const int kb = (lane >> 4) * 4;
    #pragma unroll
    for (int k0 = 0; k0 < Kd; k0 += 16) {
        bf16x4 af, bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            af[i] = a[row * LDA + k0 + kb + i];
            bf[i] = b[row * LDB + k0 + kb + i];
        }
        acc = mfma_bf16(af, bf, acc);
    }
#endif
    return acc;
}

// Pack the NW4 plain scan's two K16 register-state fragments into one x32
// operand.  We intentionally concatenate the two local bf16x4 fragments
// instead of shuffling them into monotonically increasing K order.  Loading A
// with the identical K permutation preserves the dot product:
//
//   lane group g: [K(4g..4g+3), K(16+4g..16+4g+3)].
//
// Across four lane groups this is a permutation of K0..31, and MFMA reduction
// is invariant when both operands use it.  This avoids all cross-lane shuffles.
__device__ __forceinline__ bf16x8 pack_regb_k32_x32(
        const float (&state)[2][4], int lane) {
#if defined(__gfx950__)
    bf16x8 packed;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed[i] = f32_to_bf16(state[0][i]);
        packed[i + 4] = f32_to_bf16(state[1][i]);
    }
    (void)lane;
    return packed;
#else
    (void)state;
    (void)lane;
    return bf16x8{};
#endif
}

// Contract one K32 tile with the original pair of K16 MFMAs.  The packed
// fragment is split back into its two bf16x4 halves without changing their
// order, and the second instruction inherits the first instruction's FP32
// accumulator.  This is the exact reduction tree used by gemm_regB<LD, 2>,
// but accepts a fragment that has passed through the compact LDS exchange.
template <int LD>
__device__ __forceinline__ f32x4 gemm_packed_k32_x16(
        const __bf16* __restrict__ a,
        bf16x8 packed_state,
        int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        bf16x4 af, bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            af[i] = a[row * LD + kt * 16 + kb + i];
            bf[i] = packed_state[kt * 4 + i];
        }
        acc = mfma_bf16(af, bf, acc);
    }
    return acc;
}

// Contract one K32 tile with an already packed register-B fragment.  The
// fragment order is deliberately the same non-monotonic order produced by
// pack_regb_k32_x32, so callers may move that fragment through LDS without
// changing the MFMA reduction tree.
template <int LD>
__device__ __forceinline__ f32x4 gemm_packed_k32_x32(
        const __bf16* __restrict__ a,
        bf16x8 packed_state,
        int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
#if defined(__gfx950__)
    bf16x8 af;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        af[i] = a[row * LD + kb + i];
        af[i + 4] = a[row * LD + 16 + kb + i];
    }
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(
        af, packed_state, zero, 0, 0, 0);
#else
    // Complete semantic fallback for a gfx950-private kernel carried through
    // a gfx942 device pass.  Runtime policy never selects that kernel there.
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
        bf16x4 af, bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            af[i] = a[row * LD + kt * 16 + kb + i];
            bf[i] = packed_state[kt * 4 + i];
        }
        acc = mfma_bf16(af, bf, acc);
    }
    return acc;
#endif
}

// NW4 register-state contraction over exactly K32.  On gfx950, four callers
// (one per BT16 row tile) share the prepacked B fragment and each issue one
// native x32 MFMA.  The x16 fallback preserves fat-binary compilation for a
// gfx942 device pass, although runtime policy never selects this operator there.
template <int LD>
__device__ __forceinline__ f32x4 gemm_regb_k32_x32(
        const __bf16* __restrict__ a,
        const float (&state)[2][4],
        bf16x8 packed_state,
        int lane) {
#if defined(__gfx950__)
    (void)state;
    return gemm_packed_k32_x32<LD>(a, packed_state, lane);
#else
    (void)packed_state;
    return gemm_regB<LD, 2>(a, state, lane);
#endif
}

// Even-NKB form used by the segment output replay.  Pair adjacent K16 state
// fragments with the same simultaneous A/B permutation as the K32 helper and
// retain one FP32 accumulator across pairs.  For K128 this replaces eight x16
// MFMAs with four x32 MFMAs without changing the state register layout.
template <int LD, int NKB>
__device__ __forceinline__ f32x4 gemm_regb_even_x32(
        const __bf16* __restrict__ a,
        const float (&state)[NKB][4],
        int lane) {
    static_assert(NKB > 0 && NKB % 2 == 0,
                  "gfx950 register-state x32 helper requires even NKB");
#if defined(__gfx950__)
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int kp = 0; kp < NKB; kp += 2) {
        bf16x8 af, bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            af[i] = a[row * LD + kp * 16 + kb + i];
            af[i + 4] = a[row * LD + (kp + 1) * 16 + kb + i];
            bf[i] = f32_to_bf16(state[kp][i]);
            bf[i + 4] = f32_to_bf16(state[kp + 1][i]);
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            af, bf, acc, 0, 0, 0);
    }
    return acc;
#else
    return gemm_regB<LD, NKB>(a, state, lane);
#endif
}

// Two independent register-state contractions sharing one FP32->BF16 state
// pack.  Keeping the Kd and Qd accumulators separate preserves each original
// x32 reduction chain, while alternating their instructions exposes twice the
// independent MFMA work to the scheduler.  The state fragment is rounded at
// the same per-K32 boundary as gemm_regb_even_x32, but only once for the pair.
struct RegBPairX32 {
    f32x4 first;
    f32x4 second;
};

template <int LD, int NKB>
__device__ __forceinline__ RegBPairX32 gemm_regb_even_x32_pair(
        const __bf16* __restrict__ a0,
        const __bf16* __restrict__ a1,
        const float (&state)[NKB][4],
        int lane) {
    static_assert(NKB > 0 && NKB % 2 == 0,
                  "gfx950 paired register-state helper requires even NKB");
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    RegBPairX32 out{{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#if defined(__gfx950__)
    #pragma unroll
    for (int kp = 0; kp < NKB; kp += 2) {
        bf16x8 af0, af1, sf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col0 = kp * 16 + kb + i;
            const int col1 = (kp + 1) * 16 + kb + i;
            af0[i] = a0[row * LD + col0];
            af0[i + 4] = a0[row * LD + col1];
            af1[i] = a1[row * LD + col0];
            af1[i + 4] = a1[row * LD + col1];
            sf[i] = f32_to_bf16(state[kp][i]);
            sf[i + 4] = f32_to_bf16(state[kp + 1][i]);
        }
        out.first = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            af0, sf, out.first, 0, 0, 0);
        out.second = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            af1, sf, out.second, 0, 0, 0);
    }
#else
    // Semantic fat-binary fallback: the two K16 chains are interleaved, but
    // each accumulator sees exactly the same instruction order as gemm_regB.
    #pragma unroll
    for (int kt = 0; kt < NKB; ++kt) {
        bf16x4 af0, af1, sf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = kt * 16 + kb + i;
            af0[i] = a0[row * LD + col];
            af1[i] = a1[row * LD + col];
            sf[i] = f32_to_bf16(state[kt][i]);
        }
        out.first = mfma_bf16(af0, sf, out.first);
        out.second = mfma_bf16(af1, sf, out.second);
    }
#endif
    return out;
}

}  // namespace flashkda_hip::gfx950
