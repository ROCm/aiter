// Self-contained MFMA / math primitives for the FlashKDA HIP backend (CDNA).
// No dependency on aiter's opus library. Fragment layout for
// v_mfma_f32_16x16x16 (verified on gfx950):
//   A[m,k]: a[i] = A[lane&15][(lane>>4)*4 + i]
//   B[k,n]: b[i] = B[(lane>>4)*4 + i][lane&15]
//   D[m,n]: d[i] = D[(lane>>4)*4 + i][lane&15]
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace flashkda_hip {

inline constexpr float KDA_LOG2E = 1.4426950408889634f;

using f32x2  = float    __attribute__((ext_vector_type(2)));
using f32x4  = float    __attribute__((ext_vector_type(4)));
using bf16x4 = __bf16   __attribute__((ext_vector_type(4)));
using bf16x8 = __bf16   __attribute__((ext_vector_type(8)));
using f16x4  = _Float16 __attribute__((ext_vector_type(4)));

// Vectorized contiguous copy of `n` bf16 (n a multiple of 8, base 16B-aligned):
// each lane moves 128-bit chunks so HBM reads lower to global_load_dwordx4 and
// LDS writes to ds_write_b128 — 8x fewer memory instructions than the scalar
// `for(idx=lane;...) dst[idx]=src[idx]` pattern (which lowered to ushort/b16).
// Used for the flat workspace tiles (kd/qd/kr/INV/Mqk/wbar/Sin). Bit-identical.
#ifndef FK_VEC_COPY
#define FK_VEC_COPY 1          // set 0 to A/B against the scalar ushort/b16 path
#endif
__device__ __forceinline__ void copy_bf16_vec(
        __bf16* __restrict__ dst, const __bf16* __restrict__ src, int n, int lane) {
#if FK_VEC_COPY
    const int nv = n >> 3;                       // #128-bit chunks
    auto* d = reinterpret_cast<bf16x8*>(dst);
    auto* s = reinterpret_cast<const bf16x8*>(src);
    for (int g = lane; g < nv; g += 64) d[g] = s[g];
    for (int r = (nv << 3) + lane; r < n; r += 64) dst[r] = src[r];  // tail (unused: all tiles %8==0)
#else
    for (int r = lane; r < n; r += 64) dst[r] = src[r];
#endif
}

// Row-wise vectorized copy: nrows x W bf16 from row-major src (stride ldsrc) to
// dst with a possibly-padded stride lddst (W a multiple of 8). Used to load the
// flat workspace tiles into bank-conflict-free padded LDS (lddst = D+4).
__device__ __forceinline__ void copy_bf16_rows(
        __bf16* __restrict__ dst, int lddst,
        const __bf16* __restrict__ src, int ldsrc, int nrows, int W, int lane) {
    const int wv = W >> 3;
    for (int g = lane; g < nrows * wv; g += 64) {
        int r = g / wv, c = g % wv;
        reinterpret_cast<bf16x8*>(dst + r*lddst)[c] =
            reinterpret_cast<const bf16x8*>(src + r*ldsrc)[c];
    }
}

// float variant (contiguous, n a multiple of 4) — global_load_dwordx4 for g_total.
__device__ __forceinline__ void copy_f32_vec(
        float* __restrict__ dst, const float* __restrict__ src, int n, int lane) {
#if FK_VEC_COPY
    const int nv = n >> 2;
    auto* d = reinterpret_cast<f32x4*>(dst);
    auto* s = reinterpret_cast<const f32x4*>(src);
    for (int g = lane; g < nv; g += 64) d[g] = s[g];
    for (int r = (nv << 2) + lane; r < n; r += 64) dst[r] = src[r];
#else
    for (int r = lane; r < n; r += 64) dst[r] = src[r];
#endif
}

// ---- Direct global->LDS DMA (gfx950 `global_load_lds`, AMD's cp.async) --------
// Step E. The builtin __builtin_amdgcn_global_load_lds(gptr, ldsptr, size, off, aux)
// streams `size` bytes per lane straight from HBM into LDS, bypassing VGPRs (the
// data never enters the register file). Completion is tracked by vmcnt, exactly
// like a global load. This (a) removes the global->reg->LDS round trip (no
// ds_write, no staging VGPRs -> higher occupancy) and (b) lets a multi-buffer
// LDS prefetch pipeline run without the VGPR blowup register-staging would cost.
//
// HARDWARE SEMANTICS (verified by microbench, /tmp/gll_*.hip):
//  * The per-lane GLOBAL SOURCE address is honored (true gather on the source).
//  * The LDS DESTINATION is NOT per-lane: the hardware packs the wave's lanes
//    contiguously as m0 + lane*size, where m0 is the wave-uniform base taken from
//    lane0's ldsptr. So a single instruction fills `active_lanes*size` contiguous
//    LDS bytes; non-contiguous (padded-row) targets need one instruction per row.
//  * Valid `size` (bytes): 4 (dword), 12 (dwordx3), 16 (dwordx4). size=8 is a
//    COMPILE ERROR ("invalid size value") — dwordx2 is not supported.
#ifndef FK_GLL
#define FK_GLL 1               // set 0 to fall back to copy_* (reg round-trip)
#endif
#define FK_LDSP __attribute__((address_space(3)))

// Contiguous n bf16 (n%8==0, 16B-aligned) via dwordx4. lane c writes lds[c*8..].
__device__ __forceinline__ void gll_bf16_vec(
        __bf16* __restrict__ lds, const __bf16* __restrict__ g, int n, int lane) {
#if defined(__gfx950__)
    const int nv = n >> 3;                       // #dwordx4 (8 bf16) chunks
    for (int c = lane; c < nv; c += 64)
        __builtin_amdgcn_global_load_lds(
            reinterpret_cast<void*>(const_cast<__bf16*>(g + c*8)),
            (FK_LDSP int*)(lds + c*8), 16, 0, 0);
#else
    // global_load_lds is a gfx950-only fast path in this backend.  On CDNA3
    // (gfx942) use the vectorized global->VGPR->LDS copy; keeping the fallback
    // here lets callers retain the same software-pipeline structure.
    copy_bf16_vec(lds, g, n, lane);
#endif
}

// Contiguous n f32 (n%4==0, 16B-aligned) via dwordx4. Used for g_total.
__device__ __forceinline__ void gll_f32_vec(
        float* __restrict__ lds, const float* __restrict__ g, int n, int lane) {
#if defined(__gfx950__)
    const int nv = n >> 2;                        // #dwordx4 (4 f32) chunks
    for (int c = lane; c < nv; c += 64)
        __builtin_amdgcn_global_load_lds(
            reinterpret_cast<void*>(const_cast<float*>(g + c*4)),
            (FK_LDSP int*)(lds + c*4), 16, 0, 0);
#else
    copy_f32_vec(lds, g, n, lane);
#endif
}

// nrows x W bf16 from flat row-major src (stride W) into LDS with padded row pitch
// `ld` (bf16). One dword (2 bf16) per lane per row -> 4B-aligned, so padded pitches
// like SD=132 (byte stride 264, only 8B-aligned) are legal. W must be even.
__device__ __forceinline__ void gll_rows_pad(
        __bf16* __restrict__ lds, int ld,
        const __bf16* __restrict__ g, int W, int nrows, int lane) {
#if defined(__gfx950__)
    const int wv = W >> 1;                         // #dwords per row
    #pragma unroll 4
    for (int r = 0; r < nrows; r++)
        if (lane < wv)
            __builtin_amdgcn_global_load_lds(
                reinterpret_cast<void*>(const_cast<__bf16*>(g + r*W + lane*2)),
                (FK_LDSP int*)(lds + r*ld + lane*2), 4, 0, 0);
#else
    copy_bf16_rows(lds, ld, g, W, nrows, W, lane);
#endif
}

__device__ __forceinline__ f32x4 mfma_bf16(bf16x4 a, bf16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// gfx950 transposed LDS read (Step C). For a 16x16 bf16 tile stored row-major
// (stride 16) contiguously in LDS, returns the fragment
//     f[i] = tile[(lane>>4)*4 + i][lane&15]
// in ONE conflict-free `ds_read_b64_tr_b16`, replacing 4 strided `ds_read_u16`.
// This is exactly the MFMA B-fragment (D[m,n]=sum_k A[m,k]*B[k,n]) AND the
// contract-first A/B fragment (D[.,.]=sum_c X[c,.]*Y[c,.]) layout. Lane->element
// mapping verified by microbenchmark: per-lane base = tile + lane*4, the 16-lane
// group reads its 64 contiguous elements and delivers them transposed. The
// compiler emits ZERO of these on its own, so we call the builtin directly.
#ifndef FK_TR_READ
#define FK_TR_READ 1           // set 0 to A/B against the strided ds_read_u16 path
#endif
__device__ __forceinline__ bf16x4 ds_read_tr16(const __bf16* tile16x16, int lane) {
#if FK_TR_READ && defined(__gfx950__)
    auto p = reinterpret_cast<__attribute__((address_space(3))) bf16x4*>(
                 (__attribute__((address_space(3))) __bf16*)(tile16x16 + lane*4));
    return __builtin_amdgcn_ds_read_tr16_b64_v4bf16(p);
#else
    const int r = lane & 15, cb = (lane >> 4) * 4;   // same fragment, strided
    bf16x4 v;
    #pragma unroll
    for (int i = 0; i < 4; i++) v[i] = tile16x16[(cb + i) * 16 + r];
    return v;
#endif
}
__device__ __forceinline__ f32x4 mfma_f16(f16x4 a, f16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

// base-2 exp (ftz via hardware); matches reference fp32_ex2_ftz closely.
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }

// sigmoid(x) = tanh(x*0.5)*0.5 + 0.5  (reference formulation)
__device__ __forceinline__ float sigmoid_tanh(float x) {
    return tanhf(x * 0.5f) * 0.5f + 0.5f;
}

__device__ __forceinline__ float bf16_to_f32(__bf16 x)   { return static_cast<float>(x); }
__device__ __forceinline__ __bf16 f32_to_bf16(float x)   { return static_cast<__bf16>(x); }
__device__ __forceinline__ _Float16 f32_to_f16(float x)  { return static_cast<_Float16>(x); }
__device__ __forceinline__ float f16_to_f32(_Float16 x)  { return static_cast<float>(x); }

// Wavefront-64 reduce-sum over all 64 lanes; result broadcast to every lane.
__device__ __forceinline__ float wave_reduce_sum(float v) {
    for (int o = 32; o >= 1; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

// ----- 16x16x16 GEMM building blocks (single wavefront = 64 lanes) -----
// D[m,n] = sum_d A[m,d]*B[n,d], A,B row-major [16, Kd] in LDS ("contract last
// dim"): both fragments are read with row = lane&15. Accumulates over Kd/16.
// LD = storage row pitch (>= Kd), so tiles can be padded to a bank-conflict-free
// stride while the logical contract length stays Kd. Defaults to Kd (unpadded).
template <typename T, int Kd, int LD = Kd>
__device__ __forceinline__ f32x4 gemm_contract_last(
        const T* __restrict__ A, const T* __restrict__ B, int lane) {
    f32x4 c = {0, 0, 0, 0};
    const int row = lane & 15;
    const int kb  = (lane >> 4) * 4;
    #pragma unroll
    for (int k0 = 0; k0 < Kd; k0 += 16) {
        if constexpr (sizeof(T) == 2 && __is_same(T, __bf16)) {
            bf16x4 a, b;
            #pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = A[row*LD + k0+kb+i]; b[i] = B[row*LD + k0+kb+i]; }
            c = mfma_bf16(a, b, c);
        }
    }
    return c;
}

// D[m,n] = sum_k A[m,k]*B[k,n], A,B row-major [16,16] in LDS ("standard"),
// fp16 operands. Single k-block.
__device__ __forceinline__ f32x4 gemm_std_f16(
        const _Float16* __restrict__ A, const _Float16* __restrict__ B, int lane) {
    const int row = lane & 15;
    const int kb  = (lane >> 4) * 4;
    f16x4 a, b;
    #pragma unroll
    for (int i = 0; i < 4; i++) { a[i] = A[row*16 + kb+i]; b[i] = B[(kb+i)*16 + row]; }
    f32x4 c = {0, 0, 0, 0};
    return mfma_f16(a, b, c);
}

// D[m,n] = sum_k A[m,k]*B[k, n0+n], A is [16,16] row-major, B is [16, ldb]
// row-major with the output tile taken at column offset n0. Single k-block
// (K=16), bf16 operands. Used for INV@v and Mqk@U (v/U are [16,128]).
__device__ __forceinline__ f32x4 mm_std_tile_bf16(
        const __bf16* __restrict__ A, const __bf16* __restrict__ B,
        int n0, int ldb, int lane) {
    const int row = lane & 15;
    const int kb  = (lane >> 4) * 4;
    bf16x4 a, b;
    #pragma unroll
    for (int i = 0; i < 4; i++) { a[i] = A[row*16 + kb+i]; b[i] = B[(kb+i)*ldb + n0 + row]; }
    f32x4 c = {0, 0, 0, 0};
    return mfma_bf16(a, b, c);
}

// D[m, n0+n] = sum_k A[m,k]*B[k, n0+n] over the FULL inner dim Kd (accumulated
// across Kd/16 k-blocks). A is [16, Kd] row-major, B is [Kd, ldb] row-major,
// output tile at column offset n0. bf16 operands. Used for [16,K]@[K,K] products.
__device__ __forceinline__ f32x4 mm_std_bigK_bf16(
        const __bf16* __restrict__ A, const __bf16* __restrict__ B,
        int Kd, int n0, int ldb, int lane) {
    const int row = lane & 15;
    const int kb  = (lane >> 4) * 4;
    f32x4 c = {0, 0, 0, 0};
    for (int k0 = 0; k0 < Kd; k0 += 16) {
        bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) { a[i] = A[row*Kd + k0+kb+i]; b[i] = B[(k0+kb+i)*ldb + n0 + row]; }
        c = mfma_bf16(a, b, c);
    }
    return c;
}

// Step C tr-read variant of mm_std_tile: A,B both contiguous [16,16] row-major
// (the BV==16 case, n0=0, ldb=16). A-fragment is a natural contiguous read; the
// strided B-fragment becomes one ds_read_b64_tr_b16. Bit-identical to the strided
// path — same values into the same fragment lanes, same MFMA.
__device__ __forceinline__ f32x4 mm_std_16_tr(
        const __bf16* __restrict__ A, const __bf16* __restrict__ B, int lane) {
    const int row = lane & 15;
    const int kb  = (lane >> 4) * 4;
    bf16x4 a;
    #pragma unroll
    for (int i = 0; i < 4; i++) a[i] = A[row*16 + kb+i];
    bf16x4 b = ds_read_tr16(B, lane);
    f32x4 c = {0, 0, 0, 0};
    return mfma_bf16(a, b, c);
}

// Step C tr-read variant of mm_contract_first for the BV==16 state update:
// D[k0+k, v] = sum_c A[c, k0+k] * B[c, v], B contiguous [16,16] (U/vnew). The A
// operand (kr, stride ldA=D) stays a strided read; B becomes one transpose read.
// Bt16 must point at the contiguous [16,16] B tile.
__device__ __forceinline__ f32x4 mm_cf_trB(
        const __bf16* __restrict__ A, int ldA, int k0,
        const __bf16* __restrict__ Bt16, int lane) {
    const int r  = lane & 15;
    const int cb = (lane >> 4) * 4;
    bf16x4 a;
    #pragma unroll
    for (int i = 0; i < 4; i++) a[i] = A[(cb+i)*ldA + k0 + r];
    bf16x4 b = ds_read_tr16(Bt16, lane);
    f32x4 c = {0, 0, 0, 0};
    return mfma_bf16(a, b, c);
}

// D[k,v] = sum_c A[c, k0+k]*B[c, v0+v]  — contract over the FIRST dim (c, C=16).
// A is [16, ldA] (k_restored: rows=chunk c, cols=K), B is [16, ldB] (U: rows=c,
// cols=V). Output tile at (k0,v0). d[i] = D[k0 + (lane>>4)*4+i][v0 + lane&15].
__device__ __forceinline__ f32x4 mm_contract_first_bf16(
        const __bf16* __restrict__ A, const __bf16* __restrict__ B,
        int k0, int v0, int ldA, int ldB, int lane) {
    const int r  = lane & 15;
    const int cb = (lane >> 4) * 4;
    bf16x4 a, b;
    #pragma unroll
    for (int i = 0; i < 4; i++) { a[i] = A[(cb+i)*ldA + k0 + r]; b[i] = B[(cb+i)*ldB + v0 + r]; }
    f32x4 c = {0, 0, 0, 0};
    return mfma_bf16(a, b, c);
}

// Register-resident-state GEMM (M2b). Computes D[m,n] = sum_d A[m,d]*B[n,d] where
// A is a row-major [16,Kd] LDS tile (pitch LD) and B is the recurrence STATE held
// in per-lane fp32 REGISTERS instead of LDS: Svt[kt][i] is the state element for
// V-row (lane&15), K-col (kt*16 + (lane>>4)*4 + i) — exactly the B-fragment
// gemm_contract_last would read from Sv[(lane&15)*LD + kt*16 + (lane>>4)*4 + i].
// The fp32 state is cast to bf16 only for the MFMA operand (hardware MFMA is bf16),
// so the CARRY stays fp32 across chunks (the accuracy win) while the read matches
// the bf16-operand math. NKB = Kd/16 (=8 for D=128). Svt must be indexed only by
// the compile-time (kt,i) here so it stays in the register file (no movrel).
template <int LD, int NKB>
__device__ __forceinline__ f32x4 gemm_regB(
        const __bf16* __restrict__ A, const float (&Svt)[NKB][4], int lane) {
    const int row = lane & 15, kb = (lane >> 4) * 4;
    f32x4 c = {0, 0, 0, 0};
    #pragma unroll
    for (int kt = 0; kt < NKB; kt++) {
        bf16x4 a, b;
        #pragma unroll
        for (int i = 0; i < 4; i++) { a[i] = A[row*LD + kt*16 + kb + i]; b[i] = (__bf16)Svt[kt][i]; }
        c = mfma_bf16(a, b, c);
    }
    return c;
}

// Fused K2 state decay+accumulate, vectorized. In the state update the MFMA
// accumulator `c` holds d[i] at CONSECUTIVE k (k = kbase+i) for a fixed state row
// (sd_off = vloc*SD), so the four state elements Sv[sd_off+kbase .. +3] are
// contiguous and 8B-aligned (SD%4==0 AND kbase = kt*16 + {0,4,8,12} %4==0), and
// gtot[kbase..+3] is 16B-aligned. This lets one `ds_read_b64` + one `ds_write_b64`
// (+ `ds_read_b128` for gtot) replace the 4x scalar `ds_read_u16`/`ds_write_b16`
// the elementwise loop lowered to — the largest per-chunk LDS store in the
// recurrence. Bit-identical: same values read, same fp32 math, same rounding.
#ifndef FK_VEC_STATE
#define FK_VEC_STATE 1          // set 0 to A/B against the scalar ds_write_b16 path
#endif
__device__ __forceinline__ void state_decay_acc(
        __bf16* __restrict__ Sv, int sd_off,
        const float* __restrict__ gtot, int kbase, f32x4 c) {
#if FK_VEC_STATE
    __bf16* sp = Sv + sd_off + kbase;
    bf16x4 svo = *reinterpret_cast<bf16x4*>(sp);
    f32x4  gt  = *reinterpret_cast<const f32x4*>(gtot + kbase);
    bf16x4 svn;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        svn[i] = f32_to_bf16(bf16_to_f32(svo[i]) * ex2(gt[i]) + c[i]);
    *reinterpret_cast<bf16x4*>(sp) = svn;
#else
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int k = kbase + i;
        float sv = bf16_to_f32(Sv[sd_off + k]) * ex2(gtot[k]) + c[i];
        Sv[sd_off + k] = f32_to_bf16(sv);
    }
#endif
}

// Store an MFMA accumulator (d layout) into a row-major [16,16] LDS tile.
template <typename T>
__device__ __forceinline__ void store_acc_16x16(T* __restrict__ dst, f32x4 c, int lane) {
    const int n  = lane & 15;
    const int mb = (lane >> 4) * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) dst[(mb+i)*16 + n] = static_cast<T>(c[i]);
}

}  // namespace flashkda_hip
