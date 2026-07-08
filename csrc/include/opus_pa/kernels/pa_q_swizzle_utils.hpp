// sp3 Q_lds_rd / Q_scale / Q_dyn_qt / Q_reshape — Phase 4.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"
#include "opus_pa/kernels/pa_fp8_utils.hpp"

namespace pa_decode {

static constexpr float kMaxDynBasisFp8 = 240.0f;
static constexpr int kQRegDwords = 8;  // _v_Q_reg_size
static constexpr int kDynRspDwords = 768;  // Q reshape uses <=704 dwords per wave set

__device__ __forceinline__ float bf16_to_float(bf16_t v) { return static_cast<float>(v); }

// sp3 Q_lds_rd_addr_gen
__device__ __forceinline__ uint32_t q_lds_rd_byte_offset(int lane, int wave) {
    const int h_id = lane >> 4;
    const int r_id = lane & 0xf;
    const int dw_off = h_id * 2 + r_id * 66;
    return static_cast<uint32_t>(dw_off * 4 + wave * 32);
}

// sp3 Q_lds_rd(0, 2): two ds_read_b64 per lane
__device__ __forceinline__ void q_lds_rd(const bf16_t* q_lds,
                                         int lane,
                                         int wave,
                                         uint32_t q_regs[kQRegDwords]) {
    const auto* lds_bytes = reinterpret_cast<const uint8_t*>(q_lds);
    const uint32_t base = q_lds_rd_byte_offset(lane, wave);
    std::memcpy(&q_regs[0], lds_bytes + base, 8);
    std::memcpy(&q_regs[4], lds_bytes + base + 128, 8);
    q_regs[2] = 0;
    q_regs[3] = 0;
    q_regs[6] = 0;
    q_regs[7] = 0;
}

// sp3 Q_scale for BF16 (Q_FP16 == 0): unpack packed bf16 dwords to fp32 lanes
__device__ __forceinline__ void q_scale_bf16(uint32_t q_regs[kQRegDwords]) {
    for (int k = 0; k < kQRegDwords; k += 4) {
        const uint32_t w0 = q_regs[k + 0];
        const uint32_t w1 = q_regs[k + 1];
        const uint16_t b0 = static_cast<uint16_t>(w0 & 0xffffu);
        const uint16_t b1 = static_cast<uint16_t>(w0 >> 16);
        const uint16_t b2 = static_cast<uint16_t>(w1 & 0xffffu);
        const uint16_t b3 = static_cast<uint16_t>(w1 >> 16);
        bf16_t bf0, bf1, bf2, bf3;
        std::memcpy(&bf0, &b0, 2);
        std::memcpy(&bf1, &b1, 2);
        std::memcpy(&bf2, &b2, 2);
        std::memcpy(&bf3, &b3, 2);
        q_regs[k + 0] = __float_as_uint(bf16_to_float(bf0));
        q_regs[k + 1] = __float_as_uint(bf16_to_float(bf1));
        q_regs[k + 2] = __float_as_uint(bf16_to_float(bf2));
        q_regs[k + 3] = __float_as_uint(bf16_to_float(bf3));
    }
}

__device__ __forceinline__ float uint_bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

__device__ __forceinline__ void dyn_qt_max_lds_wr(float lane_max, float* dyn_max_lds, int lane) {
    dyn_max_lds[lane] = lane_max;
}

__device__ __forceinline__ float dyn_qt_block_absmax(float* dyn_max_lds, int nthreads) {
    __syncthreads();
    float block_max = 0.f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < nthreads; ++i) {
            block_max = fmaxf(block_max, fabsf(dyn_max_lds[i]));
        }
        dyn_max_lds[0] = block_max;
    }
    __syncthreads();
    return dyn_max_lds[0] + 1e-6f;
}

__device__ __forceinline__ float dyn_qt_wave_absmax(float* dyn_max_lds, int lane, int wave) {
    __syncthreads();
    float wave_max = 0.f;
    const int wave_base = wave * 64;
    if (lane == 0) {
        for (int i = 0; i < 64; ++i) {
            wave_max = fmaxf(wave_max, fabsf(dyn_max_lds[wave_base + i]));
        }
        dyn_max_lds[wave_base] = wave_max;
    }
    __syncthreads();
    return dyn_max_lds[wave_base] + 1e-6f;
}

// sp3 Q_dyn_qt — per-lane fp32 regs -> fp8-packed dwords in q_regs[0..1]
__device__ __forceinline__ void q_dyn_qt_lane(uint32_t q_regs[kQRegDwords],
                                            float& q_deq_scale,
                                            int lane,
                                            int wave) {
    float lane_max = 0.f;
    for (int k = 0; k < kQRegDwords; k += 2) {
        lane_max = fmaxf(lane_max, fabsf(uint_bits_to_float(q_regs[k])));
        lane_max = fmaxf(lane_max, fabsf(uint_bits_to_float(q_regs[k + 1])));
    }
    __shared__ float dyn_max_lds[256];
    dyn_max_lds[threadIdx.x] = lane_max;
    const float absmax = dyn_qt_wave_absmax(dyn_max_lds, lane, wave);

    const float scale = kMaxDynBasisFp8 / absmax;
    q_deq_scale = absmax / kMaxDynBasisFp8;

    float scaled[kQRegDwords];
    for (int k = 0; k < kQRegDwords; ++k) {
        scaled[k] = uint_bits_to_float(q_regs[k]) * scale;
    }

    uint32_t packed[2] = {0, 0};
    for (int k = 0; k < kQRegDwords; k += 4) {
        packed[k / 4] =
            float4_to_fp8_pk(scaled[k + 0], scaled[k + 1], scaled[k + 2], scaled[k + 3]);
    }
    q_regs[0] = packed[0];
    q_regs[1] = packed[1];
    for (int k = 2; k < kQRegDwords; ++k) {
        q_regs[k] = 0;
    }
}

// sp3 Dyn_qnt_resp_lds_wr_addr_gen
__device__ __forceinline__ uint32_t q_reshape_wr_dword_offset(int lane, int wave) {
    const int h_id = lane >> 5;
    const int r_id = lane & 0x1f;
    const int p_id = r_id >> 4;
    const int q_id = r_id & 0xf;
    return static_cast<uint32_t>(h_id * 32 + p_id + q_id * 2 + wave * 64);
}

// sp3 Q_reshape_wr + Q_reshape_rd
__device__ __forceinline__ void q_reshape(uint32_t q_regs[kQRegDwords], uint32_t* dyn_resp_lds, int lane, int wave) {
    const uint32_t wr_base = q_reshape_wr_dword_offset(lane, wave);
    dyn_resp_lds[wr_base + 0] = q_regs[0];
    dyn_resp_lds[wr_base + 256] = q_regs[1];
    __syncthreads();

    // sp3 Q_reshape_rd / Dyn_qnt_resp_lds_rd_addr_gen
    const int h_id = lane >> 4;
    const int q_id = lane & 0xf;
    const uint32_t rd_base = static_cast<uint32_t>(h_id * 64 + q_id * 2);

    for (int k = 0; k < kQRegDwords; k += 2) {
        const uint32_t lds_off = static_cast<uint32_t>(((k & 3) / 2) * 32 + (k / 4) * 256);
        const uint32_t idx = rd_base + lds_off;
        const uint32_t lo = dyn_resp_lds[idx];
        const uint32_t hi = dyn_resp_lds[idx + 1];
        q_regs[k + 0] = lo;
        q_regs[k + 1] = hi;
    }
}

// Full sp3 Q path: lds_rd -> scale -> dyn_qt -> reshape
template<int HEAD_DIM>
__device__ __forceinline__ void q_swizzle_pipeline(const bf16_t* q_lds,
                                                   int lane,
                                                   int wave,
                                                   uint32_t q_mfma_regs[kQRegDwords],
                                                   uint32_t* dyn_resp_lds,
                                                   float& q_deq_scale) {
    (void)HEAD_DIM;
    q_lds_rd(q_lds, lane, wave, q_mfma_regs);
    q_scale_bf16(q_mfma_regs);
    q_dyn_qt_lane(q_mfma_regs, q_deq_scale, lane, wave);
    q_reshape(q_mfma_regs, dyn_resp_lds, lane, wave);
}

// MFMA Q path using precomputed per-query q_deq_scales (matches pa_ref quant).
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void q_swizzle_pipeline_paref_deq(const bf16_t* q_lds,
                                                            int lane,
                                                            int wave,
                                                            const float q_deq_scales[GQA],
                                                            uint32_t q_mfma_regs[kQRegDwords],
                                                            uint32_t* dyn_resp_lds) {
    (void)HEAD_DIM;
    q_lds_rd(q_lds, lane, wave, q_mfma_regs);
    q_scale_bf16(q_mfma_regs);

    const int qi = lane & 15;
    const float absmax =
        (qi < GQA) ? fmaxf(q_deq_scales[qi] * kMaxDynBasisFp8, 1e-6f) : 1e-6f;
    const float scale = kMaxDynBasisFp8 / absmax;

    float scaled[kQRegDwords];
#pragma unroll
    for (int k = 0; k < kQRegDwords; ++k) {
        scaled[k] = uint_bits_to_float(q_mfma_regs[k]) * scale;
    }

    uint32_t packed[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < kQRegDwords; k += 4) {
        packed[k / 4] =
            float4_to_fp8_pk(scaled[k + 0], scaled[k + 1], scaled[k + 2], scaled[k + 3]);
    }
    q_mfma_regs[0] = packed[0];
    q_mfma_regs[1] = packed[1];
#pragma unroll
    for (int k = 2; k < kQRegDwords; ++k) {
        q_mfma_regs[k] = 0;
    }
    q_reshape(q_mfma_regs, dyn_resp_lds, lane, wave);
}

// Legacy alias kept for callers that pass q_deq_scale out-param only.
template<int HEAD_DIM>
__device__ __forceinline__ void q_swizzle_pipeline_legacy(const bf16_t* q_lds,
                                                          int lane,
                                                          int wave,
                                                          uint32_t q_mfma_regs[kQRegDwords],
                                                          uint32_t* dyn_resp_lds,
                                                          float& q_deq_scale) {
    q_swizzle_pipeline<HEAD_DIM>(q_lds, lane, wave, q_mfma_regs, dyn_resp_lds, q_deq_scale);
}

// sp3 _v_unPack: row_shl:[8] bound_ctrl on 0xffffffff -> lanes (r*16+0..7) keep, (+8..15) zero.
__device__ __forceinline__ float q_deq_unpack_lane_mask(int lane) {
    return ((lane & 15) < 8) ? 1.f : 0.f;
}

// sp3 row_shr:[8] broadcast on _v_Q_deQ + collect per-query scales (lane groups 0..3).
template<int GQA>
__device__ __forceinline__ void q_deq_broadcast_and_collect(float lane_deq,
                                                            float q_deq_scales[GQA]) {
    __shared__ float deq_smem[256];
    const int lane = lane_id();
    const int wave = wave_id();
    const int wave_base = wave * 64;
    const int local = lane;
    lane_deq *= q_deq_unpack_lane_mask(lane);
    deq_smem[threadIdx.x] = lane_deq;
    __syncthreads();
    const int bro = (local >= 8) ? local - 8 : local;
    const float bcast = deq_smem[wave_base + bro];
    deq_smem[threadIdx.x] = bcast;
    __syncthreads();
    // Per-query deQ lives at global lane q*16 (threads 0,16,32,...,112 for GQA=8).
    if ((threadIdx.x & 15) == 0) {
        const int q = threadIdx.x >> 4;
        if (q < GQA) {
            q_deq_scales[q] = deq_smem[threadIdx.x];
        }
    }
    __syncthreads();
}

}  // namespace pa_decode
