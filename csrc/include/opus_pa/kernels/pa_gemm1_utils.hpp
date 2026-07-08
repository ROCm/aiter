// sp3 cl_gemm1 + R_div_L (Phase 5 smoke).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"
#include "opus_pa/kernels/pa_gemm0_utils.hpp"
#include "opus_pa/kernels/pa_q_gemm_utils.hpp"
#include "opus_pa/kernels/pa_fp8_utils.hpp"
#include "opus_pa/kernels/pa_mfma_layout_utils.hpp"
#include "opus_pa/kernels/pa_decode_device_utils.hpp"

namespace pa_decode {

static constexpr int kSzVv = 32;  // sz_vV
static constexpr int kSzVvHalf = kSzVv / 2;  // sp3: j*sz_vV/2 — 16 dwords per 64-dim head slice
static constexpr int kPiKSteps = 128 / 32;   // sp3 per-pi: 128/32 = 4 MFMA K steps

// sp3 cl_gemm1: R = P @ V, P is FP8 [16 x SUB_KV], V is FP8 [SUB_KV x HEAD_DIM].
template<int SUB_KV, int HEAD_DIM, int KV_REG_DWORDS>
__device__ __forceinline__ void cl_gemm1_fp8(const uint32_t* v_regs,
                                             const uint32_t* p_regs,
                                             mfma_acc4 r_out[HEAD_DIM / 64]) {
    static_assert(KV_REG_DWORDS >= kSzVv, "v_regs must hold sz_vV dwords");
    static_assert(SUB_KV / 32 == 8, "expected SUB_KV=256");
    static_assert(HEAD_DIM / 64 == 2, "expected HEAD_DIM=128");

    constexpr int kNumJ = HEAD_DIM / 64;
    constexpr int kNumK = SUB_KV / 32;

#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        mfma_acc4 acc{};
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
        for (int k = 0; k < kNumK; ++k) {
            const int pi = k / kPiKSteps;
            const int kk = k % kPiKSteps;
            const int vV_off = pi * kSzVv + j * kSzVvHalf + kk * kVsAb;
            const uint64_t v_pk =
                pack_u64(v_regs[vV_off + 0], v_regs[vV_off + 1]);
            const uint64_t p_pk = pack_u64(p_regs[k * kVsAb + 0], p_regs[k * kVsAb + 1]);
            acc = mfma_fp8_fp8_gemm1_step(acc, p_pk, v_pk, k == 0);
        }
#endif
        r_out[j] = acc;
    }
}

// sp3 R_div_L (single-tile smoke): normalize MFMA R accumulator by softmax row sum L.
template<int GQA>
__device__ __forceinline__ float r_div_l_smoke(const mfma_acc4 r_mfma[2],
                                               const float L_acc[GQA],
                                               int query_row) {
    const float inv_l = 1.f / fmaxf(L_acc[query_row], 1e-6f);
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        sum += mfma_acc_sum(r_mfma[j]);
    }
    return sum * inv_l;
}

// Reference GEMM1: P row is float (post-softmax * VQ), matching CPU matmul before P requant error.
template<int GQA, int SUB_KV, int HEAD_DIM>
__device__ __forceinline__ void gemm1_pv_float_reference(const float (*p_f32)[SUB_KV],
                                                         const uint8_t* v_pool,
                                                         const uint32_t* page_ids,
                                                         int valid_blks,
                                                         uint32_t stride_blk,
                                                         uint32_t stride_kvhead,
                                                         int kv_head_idx,
                                                         int block_size,
                                                         int tile_kv,
                                                         float (*o_out)[HEAD_DIM]) {
    const size_t kv_head_off =
        static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int qi = idx / HEAD_DIM;
        const int di = idx % HEAD_DIM;
        float acc = 0.f;
        for (int kv = 0; kv < tile_kv; ++kv) {
            const int blk = kv / block_size;
            const int in_blk = kv % block_size;
            const uint32_t page = (blk < valid_blks) ? page_ids[blk] : 0u;
            const uint8_t* page_base =
                v_pool + static_cast<size_t>(page) * stride_blk + kv_head_off;
            const uint8_t vv = page_base[v_shuffled_page_offset(in_blk, di, block_size)];
            acc += p_f32[qi][kv] * fp8_e4m3_to_float(vv);
        }
        o_out[qi][di] = acc;
    }
    __syncthreads();
}

// Reference GEMM1: accumulate one pi KV slice [kv_base, kv_base+slice_kv).
template<int GQA, int SUB_KV, int HEAD_DIM>
__device__ __forceinline__ void gemm1_pv_reference_pi_slice(
    const uint8_t (*p_fp8)[SUB_KV],
    const uint8_t* v_pool,
    const uint32_t* page_ids,
    int valid_blks,
    uint32_t stride_blk,
    int block_size,
    int kv_base,
    int slice_kv,
    int tile_kv,
    uint32_t stride_kvhead,
    int kv_head_idx,
    const float* p_deq_scales,
    float (*o_out)[HEAD_DIM]) {
    const size_t kv_head_off = static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int qi = idx / HEAD_DIM;
        const int di = idx % HEAD_DIM;
        double acc64 = 0.0;
        for (int kl = 0; kl < slice_kv; ++kl) {
            const int kv = kv_base + kl;
            if (kv >= tile_kv) {
                break;
            }
            const int blk = kv / block_size;
            const int in_blk = kv % block_size;
            const uint32_t page = (blk < valid_blks) ? page_ids[blk] : 0u;
            const uint8_t* page_base =
                v_pool + static_cast<size_t>(page) * stride_blk + kv_head_off;
            const uint8_t vv = page_base[v_shuffled_page_offset(in_blk, di, block_size)];
            const float pv = fp8_e4m3_to_float(p_fp8[qi][kv]) * p_deq_scales[qi];
            acc64 += static_cast<double>(pv) * static_cast<double>(fp8_e4m3_to_float(vv));
        }
        o_out[qi][di] += static_cast<float>(acc64);
    }
    __syncthreads();
}

// Gather cl_gemm1_fp8 MFMA accumulators into O with per-query p_deq.
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void gemm1_clgemm1_gather_o(const mfma_acc4 r_mfma[HEAD_DIM / 64],
                                                      int query_base,
                                                      const float* p_deq_scales,
                                                      float o_out[GQA][HEAD_DIM]) {
    const int lane = lane_id();
    const int wave = wave_id();
#pragma unroll
    for (int j = 0; j < HEAD_DIM / 64; ++j) {
        const int head_base = j * 64;
        mfma_acc4 scaled{};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = 0;
            int col = 0;
            mfma16_lane_to_mn(lane, k, row, col);
            (void)col;
            const int qi = query_base + row;
            const float deq = (qi < GQA) ? p_deq_scales[qi] : 0.f;
            scaled[k] = r_mfma[j][k] * deq;
        }
        mfma_gather_o_slice_accum<GQA, HEAD_DIM>(scaled, lane, wave, query_base, head_base, o_out);
    }
    __syncthreads();
}

// Device-side max |a-b| for GEMM1 bisect (writes smem[0]).
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void gemm1_max_abs_diff(const float a[GQA][HEAD_DIM],
                                                   const float b[GQA][HEAD_DIM],
                                                   float* smem_reduce) {
    float local = 0.f;
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int g = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        local = fmaxf(local, fabsf(a[g][d] - b[g][d]));
    }
    block_reduce_max_float(smem_reduce, blockDim.x, local);
}

// Reference GEMM1: P includes VQ; p_deq per query row; V is FP8 in shuffled pages.
template<int GQA, int SUB_KV, int HEAD_DIM>
__device__ __forceinline__ void gemm1_pv_reference(const uint8_t (*p_fp8)[SUB_KV],
                                                   const uint8_t* v_pool,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   uint32_t stride_blk,
                                                   uint32_t stride_kvhead,
                                                   int kv_head_idx,
                                                   int block_size,
                                                   int tile_kv,
                                                   const float* p_deq_scales,
                                                   const float* v_scale_pool,
                                                   float (*o_out)[HEAD_DIM]) {
    (void)v_scale_pool;
    const size_t kv_head_off =
        static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int qi = idx / HEAD_DIM;
        const int di = idx % HEAD_DIM;
        float acc = 0.f;
        double acc64 = 0.0;
        for (int kv = 0; kv < tile_kv; ++kv) {
            const int blk = kv / block_size;
            const int in_blk = kv % block_size;
            const uint32_t page = (blk < valid_blks) ? page_ids[blk] : 0u;
            const uint8_t* page_base =
                v_pool + static_cast<size_t>(page) * stride_blk + kv_head_off;
            const uint8_t vv = page_base[v_shuffled_page_offset(in_blk, di, block_size)];
            const float p_deq = p_deq_scales[qi];
            const float pv = fp8_e4m3_to_float(p_fp8[qi][kv]) * p_deq;
            const float vv_f = fp8_e4m3_to_float(vv);
            acc64 += static_cast<double>(pv) * static_cast<double>(vv_f);
        }
        acc = static_cast<float>(acc64);
        o_out[qi][di] = acc;
    }
    __syncthreads();
}

}  // namespace pa_decode