// gfx950-private two-dimensional affine scan experiment.
//
// The established NW2 scan assigns one wave to every V16 state slice.  A
// wave therefore carries the complete K128 state and serially contracts all
// eight K16 output tiles.  That leaves a dense N=1 launch with only 96 waves.
// This strict-opt-in variant doubles the workgroup to four waves and splits K
// as well as V:
//
//   wave = 2 * khalf + vhalf,   state = K64 x V16.
//
// Every wave forms the K64 contribution to all K128 outputs.  It retains the
// four output tiles that it owns and publishes the other four as lane-major
// MFMA fragments in LDS.  The owner combines low-K and high-K partials in a
// fixed Plo + Phi order, adds b, and keeps only its K64 state for the next
// group.  A pitch-132 K128 matrix plus the fragment exchange consume 50,176
// bytes of LDS.  Actual residency is also constrained by the compiled VGPR
// allocation and is audited from AMDHSA metadata rather than assumed here.
//
// Splitting the dot product changes the FP32 reduction tree from the
// established four-MFMA chain to (Plo + Phi).  The launcher therefore keeps
// this kernel behind an exact-"1" experiment and correctness uses a numerical
// tolerance rather than a bitwise comparison. As with the other affine
// kernels, G8/G16 instantiations are valid only behind the launcher's dense-N1
// workspace guard.
#pragma once

#include <hip/hip_runtime.h>

#include "k2_kda_vsplit_rs_x32_kernel.hpp"

namespace flashkda_hip::gfx950 {

template <
    int GROUP_CHUNKS,
    bool HI = false,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false>
__global__ void __launch_bounds__(256)
k2_kda_context_affine_scan_ksplit_wg4_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // [N,H,V,K], HI only
        void* __restrict__ final_state,       // empty packed sequences, HO
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        int T_seq,
        int H,
        int NT) {
    static_assert(
        GROUP_CHUNKS == 8 || GROUP_CHUNKS == 16 || GROUP_CHUNKS == 32 ||
            GROUP_CHUNKS == 64 || GROUP_CHUNKS == 128,
        "K-split affine scan supports G8/G16/G32/G64/G128");

    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int KHALF = 64;
    constexpr int LOCAL_NKB = KHALF / C;
    constexpr int AD = D + 4;
    constexpr int NTHREADS = 256;
    constexpr int A_ROW_VECS = D / 8;
    constexpr int A_VECS = D * A_ROW_VECS;
    constexpr int AMAT_ELEMENTS = D * AD;
    constexpr int AMAT_BYTES = AMAT_ELEMENTS * int(sizeof(__bf16));
    // [vhalf][destination khalf][local output K16][lane][fragment element]
    constexpr int XCHG_ELEMENTS = 2 * 2 * LOCAL_NKB * 64 * 4;
    constexpr int XCHG_BYTES = XCHG_ELEMENTS * int(sizeof(float));
    constexpr int SMEM_BYTES = AMAT_BYTES + XCHG_BYTES;
    static_assert(AMAT_BYTES == 33792 && XCHG_BYTES == 16384 &&
                  SMEM_BYTES == 50176,
                  "K-split affine scan LDS contract changed");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int vhalf = wave & 1;
    const int khalf = wave >> 1;
    const int bh = int(blockIdx.x);
    const int seq = bh / H;
    const int h = bh - seq * H;
    const int v0 = (int(blockIdx.y) * 2 + vhalf) * BV;
    const int k0 = khalf * KHALF;

    int context_base;
    int context_count;
    if constexpr (VL) {
        context_base = context_prefix[seq];
        context_count = context_prefix[seq + 1] - context_base;
        // A filtered hybrid prefix omits non-empty short sequences because
        // the preceding direct pass has already completed them.  Empty
        // sequences continue so the HO path can preserve the state contract.
        if (context_count == 0 &&
            cu_seqlens[seq + 1] != cu_seqlens[seq])
            return;
    } else {
        (void)T_seq;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        context_base = seq * groups_per_sequence;
        context_count = groups_per_sequence;
    }

    float hreg[LOCAL_NKB][4];
    const int64_t state_slab = (int64_t(seq) * H + h) * D * D;
    #pragma unroll
    for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = k0 + ktile * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_slab + int64_t(vv) * D + kk;
            if constexpr (HI) {
                if constexpr (SFP32) {
                    hreg[ktile][i] =
                        reinterpret_cast<const float*>(init_state)[idx];
                } else {
                    hreg[ktile][i] = bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
                }
            } else {
                hreg[ktile][i] = 0.0f;
            }
        }
    }

    __shared__ __align__(16) unsigned char smem[SMEM_BYTES];
    auto* const amat = reinterpret_cast<__bf16*>(smem);
    auto* const xchg = reinterpret_cast<float*>(smem + AMAT_BYTES);

    for (int local_group = 0; local_group < context_count; ++local_group) {
        const int global_context = context_base + local_group;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        // All prior-group readers reached the exchange barrier before any
        // wave could enter this iteration.  Publish the next pitch-132 A tile;
        // this barrier also prevents exchange reuse until every destination
        // wave has consumed its previous fragment.
        const auto* const a_src = reinterpret_cast<const bf16x8*>(
            affine_a + context_slab);
        #pragma unroll
        for (int j = 0; j < A_VECS / NTHREADS; ++j) {
            const int idx = tid + j * NTHREADS;
            const int row = idx / A_ROW_VECS;
            const int col8 = idx - row * A_ROW_VECS;
            reinterpret_cast<bf16x8*>(amat + row * AD)[col8] = a_src[idx];
        }
        __syncthreads();

        // Retain only the partials for this wave's output K64.  Spell the two
        // halves as separate loops so the compiler need not predicate eight
        // live fragments on a runtime destination comparison.
        float owned[LOCAL_NKB][4];
        #pragma unroll
        for (int output_local = 0;
             output_local < LOCAL_NKB;
             ++output_local) {
            const int output_ktile = khalf * LOCAL_NKB + output_local;
            const f32x4 partial = gemm_regb_even_x32<AD, LOCAL_NKB>(
                amat + output_ktile * C * AD + k0, hreg, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                owned[output_local][i] = partial[i];
        }
        const int remote_khalf = khalf ^ 1;
        #pragma unroll
        for (int output_local = 0;
             output_local < LOCAL_NKB;
             ++output_local) {
            const int output_ktile =
                remote_khalf * LOCAL_NKB + output_local;
            const f32x4 partial = gemm_regb_even_x32<AD, LOCAL_NKB>(
                amat + output_ktile * C * AD + k0, hreg, lane);
            const int exchange_base =
                ((((vhalf * 2 + remote_khalf) * LOCAL_NKB +
                   output_local) * 64 + lane) * 4);
            *reinterpret_cast<f32x4*>(xchg + exchange_base) = partial;
        }

        // Every source has finished reading A and publishing the remote K64
        // contribution.  The next iteration may overwrite A after this point;
        // its publication barrier protects the remaining exchange reads.
        __syncthreads();

        #pragma unroll
        for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
            const int exchange_base =
                ((((vhalf * 2 + khalf) * LOCAL_NKB + ktile) * 64 +
                   lane) * 4);
            const f32x4 remote =
                *reinterpret_cast<const f32x4*>(xchg + exchange_base);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = k0 + ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                const float b = affine_b[idx];
                affine_b[idx] = hreg[ktile][i];
                const float plo = khalf == 0 ? owned[ktile][i] : remote[i];
                const float phi = khalf == 0 ? remote[i] : owned[ktile][i];
                hreg[ktile][i] = (plo + phi) + b;
            }
        }
    }

    // Non-empty affine sequences publish final state from their replay's last
    // group.  Only an empty packed sequence is owned by the scan itself.
    if constexpr (HO && VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] == cu_seqlens[seq]) {
            #pragma unroll
            for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk =
                        k0 + ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            hreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(hreg[ktile][i]);
                    }
                }
            }
        }
    }
}

// G64-only latency-hiding candidate.  Keep this as an independent symbol so
// the established K-split template, mangling, and generated device body stay
// untouched when the experiment is disabled.  Each wave fetches its sixteen
// FP32 b values and publishes h_in before staging A.  Those independent global
// transactions can then overlap the A load, LDS publication, and two K64 MFMA
// chains.  The final arithmetic remains exactly (Plo + Phi) + b.
template <
    bool HI = false,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false>
__global__ void __launch_bounds__(256)
k2_kda_context_affine_scan_ksplit_prefetch_b_g64_wg4_kernel(
        const __bf16* __restrict__ affine_a,  // [G,H,K,K]
        float* __restrict__ affine_b,         // b -> h_in, [G,H,K,V]
        const void* __restrict__ init_state,  // [N,H,V,K], HI only
        void* __restrict__ final_state,       // empty packed sequences, HO
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ context_prefix,
        int T_seq,
        int H,
        int NT) {
    constexpr int GROUP_CHUNKS = 64;
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int KHALF = 64;
    constexpr int LOCAL_NKB = KHALF / C;
    constexpr int AD = D + 4;
    constexpr int NTHREADS = 256;
    constexpr int A_ROW_VECS = D / 8;
    constexpr int A_VECS = D * A_ROW_VECS;
    constexpr int AMAT_ELEMENTS = D * AD;
    constexpr int AMAT_BYTES = AMAT_ELEMENTS * int(sizeof(__bf16));
    constexpr int XCHG_ELEMENTS = 2 * 2 * LOCAL_NKB * 64 * 4;
    constexpr int XCHG_BYTES = XCHG_ELEMENTS * int(sizeof(float));
    constexpr int SMEM_BYTES = AMAT_BYTES + XCHG_BYTES;
    static_assert(AMAT_BYTES == 33792 && XCHG_BYTES == 16384 &&
                  SMEM_BYTES == 50176,
                  "G64 K-split prefetch-b LDS contract changed");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int vhalf = wave & 1;
    const int khalf = wave >> 1;
    const int bh = int(blockIdx.x);
    const int seq = bh / H;
    const int h = bh - seq * H;
    const int v0 = (int(blockIdx.y) * 2 + vhalf) * BV;
    const int k0 = khalf * KHALF;

    int context_base;
    int context_count;
    if constexpr (VL) {
        context_base = context_prefix[seq];
        context_count = context_prefix[seq + 1] - context_base;
        if (context_count == 0 &&
            cu_seqlens[seq + 1] != cu_seqlens[seq])
            return;
    } else {
        (void)T_seq;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        context_base = seq * groups_per_sequence;
        context_count = groups_per_sequence;
    }

    float hreg[LOCAL_NKB][4];
    const int64_t state_slab = (int64_t(seq) * H + h) * D * D;
    #pragma unroll
    for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = k0 + ktile * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_slab + int64_t(vv) * D + kk;
            if constexpr (HI) {
                if constexpr (SFP32) {
                    hreg[ktile][i] =
                        reinterpret_cast<const float*>(init_state)[idx];
                } else {
                    hreg[ktile][i] = bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
                }
            } else {
                hreg[ktile][i] = 0.0f;
            }
        }
    }

    __shared__ __align__(16) unsigned char smem[SMEM_BYTES];
    auto* const amat = reinterpret_cast<__bf16*>(smem);
    auto* const xchg = reinterpret_cast<float*>(smem + AMAT_BYTES);

    for (int local_group = 0; local_group < context_count; ++local_group) {
        const int global_context = context_base + local_group;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        // Every wave owns a disjoint K64xV16 slice.  Fetch b and publish the
        // old state before the A work so the load latency is off the final
        // exchange critical path.  breg intentionally remains live across
        // both barriers; metadata checks must reject spills before graduation.
        float breg[LOCAL_NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = k0 + ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                breg[ktile][i] = affine_b[idx];
                affine_b[idx] = hreg[ktile][i];
            }
        }

        const auto* const a_src = reinterpret_cast<const bf16x8*>(
            affine_a + context_slab);
        #pragma unroll
        for (int j = 0; j < A_VECS / NTHREADS; ++j) {
            const int idx = tid + j * NTHREADS;
            const int row = idx / A_ROW_VECS;
            const int col8 = idx - row * A_ROW_VECS;
            reinterpret_cast<bf16x8*>(amat + row * AD)[col8] = a_src[idx];
        }
        __syncthreads();

        float owned[LOCAL_NKB][4];
        #pragma unroll
        for (int output_local = 0;
             output_local < LOCAL_NKB;
             ++output_local) {
            const int output_ktile = khalf * LOCAL_NKB + output_local;
            const f32x4 partial = gemm_regb_even_x32<AD, LOCAL_NKB>(
                amat + output_ktile * C * AD + k0, hreg, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                owned[output_local][i] = partial[i];
        }
        const int remote_khalf = khalf ^ 1;
        #pragma unroll
        for (int output_local = 0;
             output_local < LOCAL_NKB;
             ++output_local) {
            const int output_ktile =
                remote_khalf * LOCAL_NKB + output_local;
            const f32x4 partial = gemm_regb_even_x32<AD, LOCAL_NKB>(
                amat + output_ktile * C * AD + k0, hreg, lane);
            const int exchange_base =
                ((((vhalf * 2 + remote_khalf) * LOCAL_NKB +
                   output_local) * 64 + lane) * 4);
            *reinterpret_cast<f32x4*>(xchg + exchange_base) = partial;
        }

        __syncthreads();

        #pragma unroll
        for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
            const int exchange_base =
                ((((vhalf * 2 + khalf) * LOCAL_NKB + ktile) * 64 +
                   lane) * 4);
            const f32x4 remote =
                *reinterpret_cast<const f32x4*>(xchg + exchange_base);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float plo =
                    khalf == 0 ? owned[ktile][i] : remote[i];
                const float phi =
                    khalf == 0 ? remote[i] : owned[ktile][i];
                hreg[ktile][i] =
                    (plo + phi) + breg[ktile][i];
            }
        }
    }

    if constexpr (HO && VL) {
        if (context_count == 0 &&
            cu_seqlens[seq + 1] == cu_seqlens[seq]) {
            #pragma unroll
            for (int ktile = 0; ktile < LOCAL_NKB; ++ktile) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + (lane & 15);
                    const int kk =
                        k0 + ktile * C + (lane >> 4) * 4 + i;
                    const int64_t idx =
                        state_slab + int64_t(vv) * D + kk;
                    if constexpr (SFP32) {
                        reinterpret_cast<float*>(final_state)[idx] =
                            hreg[ktile][i];
                    } else {
                        reinterpret_cast<__bf16*>(final_state)[idx] =
                            f32_to_bf16(hreg[ktile][i]);
                    }
                }
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950
