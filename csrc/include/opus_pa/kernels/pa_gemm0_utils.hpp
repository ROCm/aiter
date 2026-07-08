// sp3 cl_gemm0 — full MFMA GEMM0 register layout (Phase 4).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"

namespace pa_decode {

static constexpr int kVsAb = 2;   // vs_AB
static constexpr int kVsCd = 4;   // vs_CD
static constexpr int kSzVk = 32;  // sz_vK dwords per fch/pi

#if defined(__gfx942__) || defined(__gfx950__)
using mfma_acc4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ mfma_acc4 mfma_fp8_fp8_step(mfma_acc4 acc,
                                                       uint64_t a_packed,
                                                       uint64_t b_packed,
                                                       bool init) {
    const long a = __builtin_bit_cast(long, a_packed);
    const long b = __builtin_bit_cast(long, b_packed);
    if (init) {
        mfma_acc4 zero{};
        return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, zero, 0, 0, 0);
    }
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
}

// sp3 cl_gemm1 asm: v_mfma v_R, v_V, v_S. Default (P,V) matches HIP R=P@V packing;
// PA_GEMM1_SWAP_AB=1 uses asm operand order (V,P).
__device__ __forceinline__ mfma_acc4 mfma_fp8_fp8_gemm1_step(mfma_acc4 acc,
                                                             uint64_t p_packed,
                                                             uint64_t v_packed,
                                                             bool init) {
#if PA_GEMM1_SWAP_AB
    return mfma_fp8_fp8_step(acc, v_packed, p_packed, init);
#else
    return mfma_fp8_fp8_step(acc, p_packed, v_packed, init);
#endif
}
#endif

__device__ __forceinline__ uint64_t pack_u64(uint32_t lo, uint32_t hi) {
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// sp3 cl_gemm0(cl_p, pi): j in [0, SUB_KV/64), k in [0, HEAD_DIM/32)
template<int SUB_KV, int HEAD_DIM, int KV_REG_DWORDS>
__device__ __forceinline__ void cl_gemm0_fp8(const uint32_t* k_regs,
                                             const uint32_t* q_regs,
                                             mfma_acc4 s_out[SUB_KV / 64]) {
    static_assert(KV_REG_DWORDS >= kSzVk, "k_regs must hold sz_vK dwords");
    static_assert(HEAD_DIM / 32 == 4, "expected HEAD_DIM=128");

    constexpr int kNumJ = SUB_KV / 64;
    constexpr int kNumK = HEAD_DIM / 32;

#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        const int vK_off = j * (kSzVk / 4);
        const int vS_idx = j * kVsCd;
        (void)vS_idx;

        mfma_acc4 acc{};
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
        for (int k = 0; k < kNumK; ++k) {
            const uint64_t k_pk =
                pack_u64(k_regs[vK_off + k * kVsAb + 0], k_regs[vK_off + k * kVsAb + 1]);
            const uint64_t q_pk = pack_u64(q_regs[k * kVsAb + 0], q_regs[k * kVsAb + 1]);
            acc = mfma_fp8_fp8_step(acc, k_pk, q_pk, k == 0);
        }
#endif
        s_out[j] = acc;
    }
}

__device__ __forceinline__ float mfma_acc_sum(const mfma_acc4& acc) {
    return acc[0] + acc[1] + acc[2] + acc[3];
}

template<int SUB_KV>
__device__ __forceinline__ float cl_gemm0_fingerprint(const mfma_acc4 s_out[SUB_KV / 64],
                                                    float q_deq_scale,
                                                    float k_scale) {
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < SUB_KV / 64; ++j) {
        sum += mfma_acc_sum(s_out[j]);
    }
    return sum * q_deq_scale * k_scale;
}

}  // namespace pa_decode
