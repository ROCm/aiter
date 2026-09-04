// SPDX-License-Identifier: MIT
// gfx950-private, strict dense N=1,H=12 direct-replay prototype.
//
// This symbol deliberately bypasses the context policy and consumes the K1
// global workspace in place.  It is intended only for an explicit A/B launch
// with grid=(96,1,1), block=(64,1,1), and NT in {16,32} (T in {256,512}).
// One wave owns one V16 state slice, exactly as the established NW1-flat
// replay.  No LDS object, workgroup barrier, or cross-wave communication is
// present.
//
// The arithmetic contract is intentionally identical to the established
// cached/U-forward/V-forward/P0 RegBX32/TiledKr path:
//
//   Kd@state : four ordered K32 MFMAs
//   INV@vnew : one K16 MFMA
//   Qd@state : four ordered K32 MFMAs
//   MQK@u    : one K16 MFMA
//   Kr@u     : eight ordered K16 MFMAs, one per K16 state tile
//
// Every FP32->BF16 conversion and output rounding remains at the same source
// point.  The only intended change is that each MFMA A fragment is assembled
// directly from the immutable global workspace instead of first publishing
// the same values through LDS.
#pragma once

#include <hip/hip_runtime.h>

#include "mfma_gfx950.hpp"

namespace flashkda_hip::gfx950 {

namespace direct_global_n1_h12_detail {

constexpr int kChunk = 16;
constexpr int kDim = 128;
constexpr int kVTile = 16;
constexpr int kKTiles = kDim / kChunk;
constexpr int kHeads = 12;
constexpr int kTokenStride = kHeads * kDim;
constexpr int kBlocks = kHeads * (kDim / kVTile);

static_assert(kChunk == 16 && kVTile == 16,
              "global direct replay requires C16 and V16 MFMA tiles");
static_assert(kDim == 128 && kKTiles == 8,
              "global direct replay requires the K128 recurrence");
static_assert(kHeads == 12 && kTokenStride == 1536 && kBlocks == 96,
              "global direct replay geometry must remain N1/H12/grid96");
static_assert(sizeof(bf16x4) == 4 * sizeof(__bf16),
              "BF16 MFMA fragment ABI changed");
static_assert(sizeof(f32x4) == 4 * sizeof(float),
              "FP32 MFMA accumulator ABI changed");

// Row-major global A fragment for INV@vnew and MQK@u.
//
// The established helper loads
//
//   A[(lane&15), (lane>>4)*4 + i], i=0..3,
//
// from its row-major LDS publication.  The workspace is already row-major
// C16xC16, so these four global values form the identical MFMA A fragment.
__device__ __forceinline__ f32x4 row_major_global_a_reg_b(
        const __bf16* __restrict__ matrix,
        bf16x4 b,
        int lane) {
    const int row = lane & (kChunk - 1);
    const int col4 = (lane >> 4) * 4;
    bf16x4 a;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        a[i] = matrix[row * kChunk + col4 + i];
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_bf16(a, b, zero);
}

// Global equivalent of
//
//   mm_cf_trB(kr, 128, ktile*16, umat, lane)
//
// and of context_mfma_tiled_kr_reg_b after its row-major->K16 LDS
// publication.  For lane r=(lane&15), group cb=(lane>>4)*4, the LDS
// transpose read returns kr_tile[(cb+i), r].  The source workspace element is
// exactly kr[(cb+i), ktile*16+r].  u_bf already is the register fragment that
// the corresponding umat transpose read would return:
// u_bf[i] == U[(lane>>4)*4+i, lane&15].
__device__ __forceinline__ f32x4 kr_global_a_reg_b(
        const __bf16* __restrict__ kr,
        bf16x4 u_bf,
        int ktile,
        int lane) {
    const int r = lane & (kChunk - 1);
    const int cb = (lane >> 4) * 4;
    bf16x4 a;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = kr[(cb + i) * kDim + ktile * kChunk + r];
    }
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_bf16(a, u_bf, zero);
}

// Low lanes cooperatively load exactly one activated-beta C16 vector.  Each
// consumer requests m=(lane>>4)*4+i from source lane m.  All lanes execute the
// shuffle, and every source is in the active low-16 set.
__device__ __forceinline__ bf16x4 vnew_fragment(
        bf16x4 v,
        f32x4 residual,
        const float* __restrict__ beta,
        int lane) {
    const float beta_lane = lane < kChunk ? beta[lane] : 0.0f;
    const int m0 = (lane >> 4) * 4;
    bf16x4 vnew;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float beta_m = __shfl(beta_lane, m0 + i);
        const float value =
            (bf16_to_f32(v[i]) - residual[i]) * beta_m;
        vnew[i] = f32_to_bf16(value);
    }
    return vnew;
}

// Every lane directly loads the aligned activated-decay fragment consumed by
// its state lane group.  For g=(lane>>4), component i is K index
// ktile*16 + 4*g + i.  This is value-identical to both the old source-lane
// shuffle and decay[ktile*16+4*g+i] in the established NW1 LDS arena.
__device__ __forceinline__ f32x4 load_decay_fragment(
        const float* __restrict__ decay,
        int ktile,
        int lane) {
    const int lane_group = lane >> 4;
    const int vector_index = ktile * (kChunk / 4) + lane_group;
    return reinterpret_cast<const f32x4*>(decay)[vector_index];
}

inline constexpr int kKrGllLdsBytes =
    kChunk * kDim * int(sizeof(__bf16));
static_assert(kKrGllLdsBytes == 4096,
              "Kr-only GLL candidate must use exactly 4 KiB LDS");

inline constexpr int kOneStateProductGllLdsBytes =
    kChunk * kDim * int(sizeof(__bf16));
inline constexpr int kKqGllLdsBytes =
    2 * kOneStateProductGllLdsBytes;
static_assert(kOneStateProductGllLdsBytes == 4096 &&
              kKqGllLdsBytes == 8192,
              "Kd/Qd GLL candidate must use exactly 8 KiB LDS");

// Publish one row-major C16xK128 state-product matrix without changing its
// layout.  Four issues each transfer 64 lanes x 16 bytes = 1 KiB.  Because
// lane L owns the aligned bf16x8 fragment starting at 8*L, the resulting LDS
// object is byte-identical to the global source and can feed the established
// gemm_regb_even_x32 helper directly.
__device__ __forceinline__ void stage_state_product_gll(
        __bf16* __restrict__ matrix_lds,
        const __bf16* __restrict__ matrix_global,
        int lane) {
    #pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        constexpr int kIssueBf16 = 1024 / int(sizeof(__bf16));
        constexpr int kLaneBf16 = 16 / int(sizeof(__bf16));
        global_to_lds_async<16>(
            matrix_lds + issue * kIssueBf16,
            matrix_global + issue * kIssueBf16 + lane * kLaneBf16);
    }
}

// Each issue transfers 64 lanes x 16 bytes = 1 KiB.  Lanes 0..31 cover
// K tile 2*issue and lanes 32..63 cover K tile 2*issue+1.  Pairs of lanes
// cover the two contiguous bf16x8 halves of one C16 row, so the fixed GLL
// destination pattern is exactly [Ktile][C-row][K-half].
__device__ __forceinline__ void stage_kr_tiled_gll(
        __bf16* __restrict__ kr_lds,
        const __bf16* __restrict__ kr_global,
        int lane) {
    #pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        const int ktile = issue * 2 + (lane >> 5);
        const int lane32 = lane & 31;
        const int row = lane32 >> 1;
        const int half8 = lane32 & 1;
        global_to_lds_async<16>(
            kr_lds + issue * 512,
            kr_global + row * kDim + ktile * kChunk + half8 * 8);
    }
}

__device__ __forceinline__ f32x4 kr_lds_a_reg_b(
        const __bf16* __restrict__ kr_lds,
        bf16x4 u_bf,
        int ktile,
        int lane) {
    const bf16x4 a =
        ds_read_tr16(kr_lds + ktile * kChunk * kChunk, lane);
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_bf16(a, u_bf, zero);
}

}  // namespace direct_global_n1_h12_detail

// External state remains [N,H,V,K] with N=1.  init_state retains its runtime
// nullptr semantics; SFP32 selects the public state storage type, while the
// recurrence state is FP32 in registers.  HO controls only the final publish.
template <bool HO = false, bool SFP32 = false, bool KR_GLL = false,
          bool KQ_GLL = false>
__global__ void __launch_bounds__(64)
k2_kda_context_direct_global_n1_h12_kernel(
        const __bf16* __restrict__ v_g,       // dense [T,12,128]
        const float* __restrict__ beta_g,     // activated [12*NT,16]
        __bf16* __restrict__ out_g,           // dense [T,12,128]
        const __bf16* __restrict__ ws_kd,     // [12*NT,16,128]
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,      // activated [12*NT,128]
        const __bf16* __restrict__ ws_inv,    // [12*NT,16,16]
        const __bf16* __restrict__ ws_mqk,
        const void* __restrict__ init_state,  // [1,12,128,128]
        void* __restrict__ final_state,       // [1,12,128,128], HO only
        int NT) {
    using namespace direct_global_n1_h12_detail;
    static_assert(!KR_GLL || HO,
                  "Kr-only GLL is valid only when final state is requested");

    // This is a prototype-only runtime backstop.  The intended launcher has
    // already proved T=NT*C in {256,512}; no metadata or tail predicate exists
    // in this all-full-C16 symbol.
    if (NT != 16 && NT != 32)
        return;

    const int lane = int(threadIdx.x) & 63;
    const int flat = int(blockIdx.x);
    const int h = flat >> 3;
    const int v_group = flat & 7;
    const int v0 = v_group * kVTile;
    const int vv = v0 + (lane & (kVTile - 1));
    const int64_t state_slab = int64_t(h) * kDim * kDim;

    // Dynamic LDS keeps both opt-in transports at zero LDS in the control.
    // Kd and Qd own distinct 4 KiB row-major arenas; an optional Kr arena is
    // placed after them so its asynchronous publication may overlap the Kd,
    // INV, Qd, and Mqk arithmetic without aliasing either state-product read.
    extern __shared__ __attribute__((aligned(16)))
        unsigned char gll_lds_raw[];
    __bf16* const kd_gll_lds =
        reinterpret_cast<__bf16*>(gll_lds_raw);
    __bf16* const qd_gll_lds = kd_gll_lds + kChunk * kDim;
    __bf16* const kr_gll_lds =
        kd_gll_lds + (KQ_GLL ? 2 * kChunk * kDim : 0);

    // Per-lane state mapping is unchanged:
    // state[ktile][i] == S[V=v0+(lane&15),
    //                       K=ktile*16+(lane>>4)*4+i].
    float state[kKTiles][4];
    #pragma unroll
    for (int ktile = 0; ktile < kKTiles; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = ktile * kChunk + (lane >> 4) * 4 + i;
            const int64_t index =
                state_slab + int64_t(vv) * kDim + kk;
            if (init_state != nullptr) {
                if constexpr (SFP32) {
                    state[ktile][i] =
                        reinterpret_cast<const float*>(init_state)[index];
                } else {
                    state[ktile][i] = bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[index]);
                }
            } else {
                state[ktile][i] = 0.0f;
            }
        }
    }

    for (int chunk = 0; chunk < NT; ++chunk) {
        const int ht = h * NT + chunk;
        const int t0 = chunk * kChunk;
        const int m0 = (lane >> 4) * 4;
        const int local_v = lane & (kVTile - 1);
        const int64_t v_base =
            int64_t(t0 + m0) * kTokenStride +
            int64_t(h) * kDim + v0 + local_v;

        const __bf16* const kd =
            ws_kd + int64_t(ht) * kChunk * kDim;
        const __bf16* const qd =
            ws_qd + int64_t(ht) * kChunk * kDim;
        const __bf16* const kr =
            ws_kr + int64_t(ht) * kChunk * kDim;
        const __bf16* const inv =
            ws_inv + int64_t(ht) * kChunk * kChunk;
        const __bf16* const mqk =
            ws_mqk + int64_t(ht) * kChunk * kChunk;
        const float* const beta = beta_g + int64_t(ht) * kChunk;
        const float* const decay = ws_gt + int64_t(ht) * kDim;

        // Put the two bulk publications at the front of the chunk, then
        // issue the independent strided V fragment while they are in flight.
        // The following vmcnt fence publishes both operands at once.
        if constexpr (KQ_GLL) {
            stage_state_product_gll(kd_gll_lds, kd, lane);
            stage_state_product_gll(qd_gll_lds, qd, lane);
        }

        bf16x4 v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = v_g[v_base + int64_t(i) * kTokenStride];
        }

        if constexpr (KQ_GLL) {
#if defined(__gfx950__)
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
            __syncwarp();
#endif
        }

        // Start the independent Kr publication only after the Kd/Qd fence.
        // Its wait remains below the four state-product MFMAs and the two
        // local C16 MFMAs, retaining the useful overlap of the Kr-only path.
        if constexpr (KR_GLL)
            stage_kr_tiled_gll(kr_gll_lds, kr, lane);

        // Separate Kd and Qd calls preserve the established non-paired x32
        // accumulator chains and their independent state BF16 conversions.
        const f32x4 residual = gemm_regb_even_x32<kDim, kKTiles>(
            KQ_GLL ? kd_gll_lds : kd, state, lane);
        const bf16x4 vnew_bf =
            vnew_fragment(v, residual, beta, lane);

        const f32x4 u = row_major_global_a_reg_b(inv, vnew_bf, lane);
        bf16x4 u_bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            u_bf[i] = f32_to_bf16(u[i]);

        const f32x4 from_state = gemm_regb_even_x32<kDim, kKTiles>(
            KQ_GLL ? qd_gll_lds : qd, state, lane);
        const f32x4 from_local =
            row_major_global_a_reg_b(mqk, u_bf, lane);
        if constexpr (KR_GLL) {
#if defined(__gfx950__)
            // Publish Kr before issuing this chunk's output stores so
            // vmcnt(0) does not unnecessarily drain those stores.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
            // The fat-binary fallback implements GLL as ordinary per-lane
            // global->VGPR->LDS stores.
            __syncwarp();
#endif
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __bf16 a = f32_to_bf16(from_state[i]);
            const __bf16 b = f32_to_bf16(from_local[i]);
            out_g[v_base + int64_t(i) * kTokenStride] =
                f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
        }

        #pragma unroll
        for (int ktile = 0; ktile < kKTiles; ++ktile) {
            // Issue the direct VMEM read before the carry MFMA so the load can
            // overlap with the unchanged Kr fragment assembly/MFMA chain.
            const f32x4 decay_fragment =
                load_decay_fragment(decay, ktile, lane);
            f32x4 carry;
            if constexpr (KR_GLL) {
                carry = kr_lds_a_reg_b(
                    kr_gll_lds, u_bf, ktile, lane);
            } else {
                carry = kr_global_a_reg_b(
                    kr, u_bf, ktile, lane);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                state[ktile][i] =
                    state[ktile][i] * decay_fragment[i] + carry[i];
            }
        }
    }

    if constexpr (HO) {
        #pragma unroll
        for (int ktile = 0; ktile < kKTiles; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kk =
                    ktile * kChunk + (lane >> 4) * 4 + i;
                const int64_t index =
                    state_slab + int64_t(vv) * kDim + kk;
                if constexpr (SFP32) {
                    reinterpret_cast<float*>(final_state)[index] =
                        state[ktile][i];
                } else {
                    reinterpret_cast<__bf16*>(final_state)[index] =
                        f32_to_bf16(state[ktile][i]);
                }
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950
