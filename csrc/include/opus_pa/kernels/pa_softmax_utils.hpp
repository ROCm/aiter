// asm pa_fuse_alu — scale(S) + online softmax + dyn_quant(P) for GEMM1.
#pragma once

#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"
#include "opus_pa/kernels/pa_fp8_utils.hpp"
#include "opus_pa/kernels/pa_output_utils.hpp"
#include "opus_pa/kernels/pa_q_swizzle_utils.hpp"
#include "opus_pa/kernels/pa_scores_compact_utils.hpp"

namespace pa_decode {

static constexpr float kNegInf = -1.0e30f;

// AMD v_exp_f32 computes 2^x (log2 domain softmax).
__device__ __forceinline__ float pa_exp2_scaled(float x) { return exp2f(x); }

// asm LDS_DYN_MAX layout: slot = wave*64 + threadIdx, 16 rows x 16 slots.
static constexpr int kLdsDynMaxSlots = 256;

__device__ __forceinline__ uint32_t lds_dyn_max_wr_byte_offset(int thread_idx, int wave) {
    return static_cast<uint32_t>(thread_idx * 4 + wave * 256);
}

__device__ __forceinline__ uint32_t lds_dyn_max_rd_byte_offset(int lane, int slot_i) {
    return static_cast<uint32_t>((lane & 15) * 4 + slot_i * 64);
}

// Tail tile: mask KV columns beyond ctx_len.
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_tail_mask_dense(float (*s_scores)[SUB_KV],
                                                   int kv_offset,
                                                   int tile_kv,
                                                   uint32_t ctx_len) {
    for (int idx = threadIdx.x; idx < GQA * SUB_KV; idx += blockDim.x) {
        const int g = idx / SUB_KV;
        const int ki = idx % SUB_KV;
        if (ki >= tile_kv || kv_offset + ki >= static_cast<int>(ctx_len)) {
            s_scores[g][ki] = kNegInf;
        }
    }
    __syncthreads();
}

// Tail tile: mask one pi slice of KV columns beyond ctx_len.
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_tail_mask_pi_slice(float (*s_scores)[SUB_KV],
                                                      int kv_offset,
                                                      int pi,
                                                      int tile_kv,
                                                      uint32_t ctx_len) {
    constexpr int kHalf = SUB_KV / 2;
    const int kv_base = pi * kHalf;
    for (int idx = threadIdx.x; idx < GQA * kHalf; idx += blockDim.x) {
        const int g = idx / kHalf;
        const int kl = idx % kHalf;
        const int ki = kv_base + kl;
        if (ki >= tile_kv || kv_offset + ki >= static_cast<int>(ctx_len)) {
            s_scores[g][ki] = kNegInf;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void block_reduce_max_float(float* smem, int nthreads, float val) {
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void block_reduce_add_float(float* smem, int nthreads, float val) {
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

// Per-pi slice dequant on MFMA GEMM0 sparse scores.
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_apply_qk_dequant_pi(float (*s_scores)[SUB_KV],
                                                           int pi,
                                                           int tile_kv,
                                                           const float q_deq_scales[GQA],
                                                           const float k_scale_pi[SUB_KV / 2]) {
    constexpr int kHalf = SUB_KV / 2;
    const int kv_base = pi * kHalf;
    for (int idx = threadIdx.x; idx < GQA * kHalf; idx += blockDim.x) {
        const int g = idx / kHalf;
        const int kl = idx % kHalf;
        const int gi = kv_base + kl;
        if (gi < tile_kv) {
            s_scores[g][gi] *= q_deq_scales[g] * k_scale_pi[kl];
        }
    }
    __syncthreads();
}

// Per-pi slice absmax for P quant (asm pa_fuse step6.6 on pi slice only).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_compute_row_dyn_scales_slice(float (*s_vq)[SUB_KV],
                                                                     int kv_base,
                                                                     int slice_kv,
                                                                     int tile_kv,
                                                                     float* p_deq_scales,
                                                                     float row_dyn_scale[GQA]) {
    __shared__ float absmax_smem[256];

    for (int g = 0; g < GQA; ++g) {
        float row_absmax = 1e-6f;
        for (int k = threadIdx.x; k < slice_kv; k += blockDim.x) {
            const int gi = kv_base + k;
            if (gi < tile_kv) {
                row_absmax = fmaxf(row_absmax, fabsf(s_vq[g][gi]));
            }
        }
        block_reduce_max_float(absmax_smem, blockDim.x, row_absmax);
        if (threadIdx.x == 0) {
            const float absmax = absmax_smem[0];
            p_deq_scales[g] = absmax / kMaxDynBasisFp8;
            row_dyn_scale[g] = kMaxDynBasisFp8 / absmax;
        }
        __syncthreads();
    }
}

// Compute per-query P quant scales from VQ-weighted softmax (full tile, CPU-aligned).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_compute_row_dyn_scales(float (*s_vq)[SUB_KV],
                                                               int tile_kv,
                                                               float* p_deq_scales,
                                                               float row_dyn_scale[GQA]) {
    __shared__ float absmax_smem[256];

    for (int g = 0; g < GQA; ++g) {
        float row_absmax = 1e-6f;
        for (int kv = threadIdx.x; kv < tile_kv; kv += blockDim.x) {
            row_absmax = fmaxf(row_absmax, fabsf(s_vq[g][kv]));
        }
        block_reduce_max_float(absmax_smem, blockDim.x, row_absmax);
        if (threadIdx.x == 0) {
            const float absmax = absmax_smem[0];
            p_deq_scales[g] = absmax / kMaxDynBasisFp8;
            row_dyn_scale[g] = kMaxDynBasisFp8 / absmax;
        }
        __syncthreads();
    }
}

// Quant one pi slice of P using precomputed row_dyn_scale (full-row absmax).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_quant_p_slice(float (*s_vq)[SUB_KV],
                                                      int kv_base,
                                                      int slice_kv,
                                                      int tile_kv,
                                                      uint8_t (*p_fp8)[SUB_KV],
                                                      const float row_dyn_scale[GQA]) {
    for (int idx = threadIdx.x; idx < GQA * slice_kv; idx += blockDim.x) {
        const int g = idx / slice_kv;
        const int kl = idx % slice_kv;
        const int gi = kv_base + kl;
        if (gi < tile_kv) {
            const float pn = s_vq[g][gi] * row_dyn_scale[g];
            p_fp8[g][gi] = float_to_fp8_e4m3_bias8(pn);
        }
    }
    __syncthreads();
}

// Apply asm step2/2.5 dequant on raw MFMA GEMM0 scores (full tile).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_apply_qk_dequant(float (*s_scores)[SUB_KV],
                                                        int tile_kv,
                                                        const float q_deq_scales[GQA],
                                                        const float k_scale_pi0[SUB_KV / 2],
                                                        const float k_scale_pi1[SUB_KV / 2]) {
    constexpr int kHalf = SUB_KV / 2;
    for (int idx = threadIdx.x; idx < GQA * tile_kv; idx += blockDim.x) {
        const int g = idx / tile_kv;
        const int gi = idx % tile_kv;
        const float ks = (gi < kHalf) ? k_scale_pi0[gi] : k_scale_pi1[gi - kHalf];
        s_scores[g][gi] *= q_deq_scales[g] * ks;
    }
    __syncthreads();
}

// asm pa_fuse step3: per-lane partial row max -> LDS -> read 16 slots -> full row max.
template<int GQA>
__device__ __forceinline__ void pa_fuse_row_max_lds(const float s_lane_partial[kSHalfRegSize],
                                                  float* lds_dyn_max,
                                                  int query_base,
                                                  int row_group,
                                                  float row_max_out[GQA]) {
    constexpr int kRowsPerGroup = 4;
    float lane_max = kNegInf;
    for (int k = 0; k < kSHalfRegSize; ++k) {
        lane_max = fmaxf(lane_max, s_lane_partial[k]);
    }

    const int wr_slot = threadIdx.x;
    lds_dyn_max[wr_slot] = lane_max;
    __syncthreads();

    float full_max = kNegInf;
    if ((lane_id() & 15) == 0) {
        for (int i = 0; i < 16; ++i) {
            const int rd_slot = (lane_id() & 15) + i * 16;
            full_max = fmaxf(full_max, lds_dyn_max[rd_slot]);
        }
        for (int inner_k = 0; inner_k < kRowsPerGroup; ++inner_k) {
            const int qi = query_base + row_group * kRowsPerGroup + inner_k;
            if (qi < GQA) {
                row_max_out[qi] = full_max;
            }
        }
    }
    __syncthreads();
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_alu_slice(float (*s_scores)[SUB_KV],
                                                  int kv_base,
                                                  int slice_kv,
                                                  int tile_kv,
                                                  const float* q_deq_scales,
                                                  const float* k_scale_slice,
                                                  const float* v_scale_slice,
                                                  float scale_log2e,
                                                  float* fa_max,
                                                  float* L_acc,
                                                  float* delta_scale) {
    __shared__ float absmax_smem[256];
    __shared__ float row_max_smem[GQA];
    __shared__ float fa_new_smem[GQA];

    (void)q_deq_scales;
    (void)k_scale_slice;
    __syncthreads();

    for (int g = 0; g < GQA; ++g) {
        float partial_max = kNegInf;
        for (int k = threadIdx.x; k < slice_kv; k += blockDim.x) {
            const int gi = kv_base + k;
            if (gi < tile_kv) {
                partial_max = fmaxf(partial_max, s_scores[g][gi]);
            }
        }
        block_reduce_max_float(absmax_smem, blockDim.x, partial_max);
        if (threadIdx.x == 0) {
            row_max_smem[g] = absmax_smem[0];
        }
        __syncthreads();
    }

    for (int g = threadIdx.x; g < GQA; g += blockDim.x) {
        const float fa_old = fa_max[g];
        const float fa_new = fmaxf(fa_old, row_max_smem[g]);
        fa_new_smem[g] = fa_new;
        float rescale = 1.f;
        if (fa_old > -1.0e20f) {
            rescale = pa_exp2_scaled((fa_old - fa_new) * scale_log2e);
            L_acc[g] *= rescale;
        }
        delta_scale[g] = rescale;
        fa_max[g] = fa_new;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < GQA * slice_kv; idx += blockDim.x) {
        const int g = idx / slice_kv;
        const int k = idx % slice_kv;
        const int gi = kv_base + k;
        if (gi < tile_kv) {
            const float max_log2e = fa_new_smem[g] * scale_log2e;
            const float sm = pa_exp2_scaled(s_scores[g][gi] * scale_log2e - max_log2e);
            s_scores[g][gi] = sm;
        }
    }
    __syncthreads();

    for (int g = 0; g < GQA; ++g) {
        float partial_sum = 0.f;
        for (int k = threadIdx.x; k < slice_kv; k += blockDim.x) {
            const int gi = kv_base + k;
            if (gi < tile_kv) {
                partial_sum += s_scores[g][gi];
            }
        }
        block_reduce_add_float(absmax_smem, blockDim.x, partial_sum);
        if (threadIdx.x == 0) {
            L_acc[g] += absmax_smem[0];
        }
        __syncthreads();
    }

    // CPU ref: multiply softmax P by VQ before per-row FP8 quant.
    for (int idx = threadIdx.x; idx < GQA * slice_kv; idx += blockDim.x) {
        const int g = idx / slice_kv;
        const int k = idx % slice_kv;
        const int gi = kv_base + k;
        if (gi < tile_kv) {
            s_scores[g][gi] *= (v_scale_slice ? v_scale_slice[k] : 1.f);
        }
    }
    __syncthreads();
}

// asm core_loop per-pi: fuse slice + per-slice P quant + p_deq for cl_gemm1(pi).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_alu_pi_tile(float (*s_scores)[SUB_KV],
                                                    int pi,
                                                    int tile_kv,
                                                    const float v_scale_pi[SUB_KV / 2],
                                                    float scale_log2e,
                                                    float* fa_max,
                                                    float* L_acc,
                                                    float* delta_scale,
                                                    uint8_t (*p_fp8)[SUB_KV],
                                                    float* p_deq_scales) {
    constexpr int kHalf = SUB_KV / 2;
    const int kv_base = pi * kHalf;
    __shared__ float row_dyn_scale[GQA];

    pa_fuse_alu_slice<GQA, SUB_KV>(s_scores, kv_base, kHalf, tile_kv, nullptr, nullptr,
                                   v_scale_pi, scale_log2e, fa_max, L_acc, delta_scale);
    pa_fuse_compute_row_dyn_scales_slice<GQA, SUB_KV>(s_scores, kv_base, kHalf, tile_kv,
                                                      p_deq_scales, row_dyn_scale);
    pa_fuse_quant_p_slice<GQA, SUB_KV>(s_scores, kv_base, kHalf, tile_kv, p_fp8, row_dyn_scale);
}

// Rescale unnormalized O accumulator with the same online-softmax delta as L (asm step6).
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void pa_fuse_rescale_o_acc(float (*o_acc)[HEAD_DIM],
                                                      const float* delta_scale) {
    pa_r_procss_rescale<GQA, HEAD_DIM>(o_acc, delta_scale);
}

// Per-query row FP8 quant on VQ-weighted softmax (matches CPU quant() per row).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_quant_p_rows(float (*s_vq)[SUB_KV],
                                                     int tile_kv,
                                                     uint8_t (*p_fp8)[SUB_KV],
                                                     float* p_deq_scales) {
    __shared__ float absmax_smem[256];
    __shared__ float row_dyn_scale[GQA];

    for (int g = 0; g < GQA; ++g) {
        float row_absmax = 1e-6f;
        for (int kv = threadIdx.x; kv < tile_kv; kv += blockDim.x) {
            row_absmax = fmaxf(row_absmax, fabsf(s_vq[g][kv]));
        }
        block_reduce_max_float(absmax_smem, blockDim.x, row_absmax);
        if (threadIdx.x == 0) {
            const float absmax = absmax_smem[0];
            p_deq_scales[g] = absmax / kMaxDynBasisFp8;
            row_dyn_scale[g] = kMaxDynBasisFp8 / absmax;
        }
        __syncthreads();

        for (int kv = threadIdx.x; kv < tile_kv; kv += blockDim.x) {
            const float pn = s_vq[g][kv] * row_dyn_scale[g];
            p_fp8[g][kv] = float_to_fp8_e4m3_bias8(pn);
        }
        // Pad the tail [tile_kv, SUB_KV) with fp8 zero. The MFMA P-gather reads the
        // full SUB_KV width; stale smem here would decode as fp8-NaN and poison MFMA.
        for (int kv = tile_kv + threadIdx.x; kv < SUB_KV; kv += blockDim.x) {
            p_fp8[g][kv] = static_cast<uint8_t>(0);
        }
        __syncthreads();
    }
}

// Full-tile softmax + VQ + P quant (matches CPU fmha_softmax_dev one-shot per tile).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_alu_tile_scales(float (*s_scores)[SUB_KV],
                                                        int tile_kv,
                                                        const float* q_deq_scales,
                                                        const float /*k_scale_pi0*/[SUB_KV / 2],
                                                        const float /*k_scale_pi1*/[SUB_KV / 2],
                                                        const float v_scale_pi0[SUB_KV / 2],
                                                        const float v_scale_pi1[SUB_KV / 2],
                                                        float scale_log2e,
                                                        float* fa_max,
                                                        float* L_acc,
                                                        float* delta_scale,
                                                        uint8_t (*p_fp8)[SUB_KV],
                                                        float* p_deq_scales) {
    constexpr int kHalf = SUB_KV / 2;
    (void)q_deq_scales;

    __shared__ float absmax_smem[256];
    __shared__ float row_max_smem[GQA];
    __shared__ float fa_new_smem[GQA];

    // Step 3-5: row max over full tile + online FA max + exp2 softmax.
    for (int g = 0; g < GQA; ++g) {
        float partial_max = kNegInf;
        for (int k = threadIdx.x; k < tile_kv; k += blockDim.x) {
            partial_max = fmaxf(partial_max, s_scores[g][k]);
        }
        block_reduce_max_float(absmax_smem, blockDim.x, partial_max);
        if (threadIdx.x == 0) {
            row_max_smem[g] = absmax_smem[0];
        }
        __syncthreads();
    }

    for (int g = threadIdx.x; g < GQA; g += blockDim.x) {
        const float fa_old = fa_max[g];
        const float fa_new = fmaxf(fa_old, row_max_smem[g]);
        fa_new_smem[g] = fa_new;
        float rescale = 1.f;
        if (fa_old > -1.0e20f) {
            rescale = pa_exp2_scaled((fa_old - fa_new) * scale_log2e);
            L_acc[g] *= rescale;
        }
        delta_scale[g] = rescale;
        fa_max[g] = fa_new;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < GQA * tile_kv; idx += blockDim.x) {
        const int g = idx / tile_kv;
        const int gi = idx % tile_kv;
        const float max_log2e = fa_new_smem[g] * scale_log2e;
        const float sm = pa_exp2_scaled(s_scores[g][gi] * scale_log2e - max_log2e);
        s_scores[g][gi] = sm;
    }
    __syncthreads();

    for (int g = 0; g < GQA; ++g) {
        float partial_sum = 0.f;
        for (int k = threadIdx.x; k < tile_kv; k += blockDim.x) {
            partial_sum += s_scores[g][k];
        }
        block_reduce_add_float(absmax_smem, blockDim.x, partial_sum);
        if (threadIdx.x == 0) {
            L_acc[g] += absmax_smem[0];
        }
        __syncthreads();
    }

    // CPU ref: multiply softmax by VQ before P quant.
    for (int idx = threadIdx.x; idx < GQA * tile_kv; idx += blockDim.x) {
        const int g = idx / tile_kv;
        const int gi = idx % tile_kv;
        const float vs = (gi < kHalf) ? v_scale_pi0[gi] : v_scale_pi1[gi - kHalf];
        s_scores[g][gi] *= vs;
    }
    __syncthreads();

    pa_fuse_quant_p_rows<GQA, SUB_KV>(s_scores, tile_kv, p_fp8, p_deq_scales);
    __syncthreads();
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void pa_fuse_alu_tile(float (*s_scores)[SUB_KV],
                                                 int tile_kv,
                                                 const float* q_deq_scales,
                                                 const float k_scale_pi0[SUB_KV / 2],
                                                 const float k_scale_pi1[SUB_KV / 2],
                                                 const float v_scale_pi0[SUB_KV / 2],
                                                 const float v_scale_pi1[SUB_KV / 2],
                                                 float scale_log2e,
                                                 float* fa_max,
                                                 float* L_acc,
                                                 float* delta_scale,
                                                 uint8_t (*p_fp8)[SUB_KV],
                                                 float* p_deq_scales) {
    pa_fuse_alu_tile_scales<GQA, SUB_KV>(s_scores, tile_kv, q_deq_scales, k_scale_pi0, k_scale_pi1,
                                        v_scale_pi0, v_scale_pi1, scale_log2e, fa_max, L_acc,
                                        delta_scale, p_fp8, p_deq_scales);
    __syncthreads();
}

// MFMA A-operand pack for P@V GEMM1: A[m][k], m = lane%16 (query), k = 8*(lane/16)+byte_i (kv).
template<int GQA, int SUB_KV, int P_SLICE>
__device__ __forceinline__ void p_fp8_gather_lane_packed(const uint8_t p_fp8[GQA][SUB_KV],
                                                         int kv_base,
                                                         int lane,
                                                         int wave,
                                                         uint32_t packed[2]) {
    (void)wave;
    const int query = lane & 15;
    const int g = lane >> 4;

    uint8_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ki = kv_base + g * 8 + i;
        bytes[i] = (query < GQA && (g * 8 + i) < P_SLICE) ? p_fp8[query][ki]
                                                          : static_cast<uint8_t>(0);
    }
    packed[0] = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    packed[1] = static_cast<uint32_t>(bytes[4]) | (static_cast<uint32_t>(bytes[5]) << 8) |
                (static_cast<uint32_t>(bytes[6]) << 16) | (static_cast<uint32_t>(bytes[7]) << 24);
}

// Gather FP8 P for GEMM1 MFMA A-operand (asm cl_gemm1): kv index has no wave offset.
template<int GQA, int SUB_KV, int P_SLICE>
__device__ __forceinline__ void p_fp8_gather_lane_packed_gemm1(const uint8_t p_fp8[GQA][SUB_KV],
                                                               int kv_base,
                                                               int lane,
                                                               uint32_t packed[2]) {
    constexpr int kRowsPerGroup = 4;
    const int col = lane & 15;
    const int row_group = lane >> 4;

    uint8_t bytes[8];
#pragma unroll
    for (int reg_k = 0; reg_k < 8; ++reg_k) {
        const int j = reg_k >> 2;
        const int inner_k = reg_k & 3;
        const int qi = row_group * kRowsPerGroup + inner_k;
        const int ki = kv_base + j * 64 + col;
        bytes[reg_k] = (qi < GQA && ki < kv_base + P_SLICE) ? p_fp8[qi][ki]
                                                            : static_cast<uint8_t>(0);
    }
    packed[0] = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    packed[1] = static_cast<uint32_t>(bytes[4]) | (static_cast<uint32_t>(bytes[5]) << 8) |
                (static_cast<uint32_t>(bytes[6]) << 16) | (static_cast<uint32_t>(bytes[7]) << 24);
}

// Gather one MFMA A-operand pair (P[qi,kv] tile) for a single 32-wide KV step.
template<int GQA, int SUB_KV, int P_SLICE>
__device__ __forceinline__ void p_mfma_gather_a_pair(const uint8_t p_fp8[GQA][SUB_KV],
                                                     int kv_base,
                                                     int lane,
                                                     uint32_t& lo,
                                                     uint32_t& hi) {
    uint32_t packed[2] = {};
    p_fp8_gather_lane_packed_gemm1<GQA, SUB_KV, P_SLICE>(p_fp8, kv_base, lane, packed);
    lo = packed[0];
    hi = packed[1];
}

// Legacy GEMM0-style gather (includes wave offset in kv); kept for bisect.
template<int GQA, int SUB_KV, int P_SLICE>
__device__ __forceinline__ void p_mfma_gather_a_pair_wave(const uint8_t p_fp8[GQA][SUB_KV],
                                                          int kv_base,
                                                          int lane,
                                                          int wave,
                                                          uint32_t& lo,
                                                          uint32_t& hi) {
    uint32_t packed[2] = {};
    p_fp8_gather_lane_packed<GQA, SUB_KV, P_SLICE>(p_fp8, kv_base, lane, wave, packed);
    lo = packed[0];
    hi = packed[1];
}

template<int P_SLICE>
__device__ __forceinline__ void p_fp8_to_compact_regs_slice(const uint8_t* p_fp8_row,
                                                            int kv_offset,
                                                            int lane,
                                                            uint32_t p_compact[kQRegDwords]) {
    const int base = (lane & 0xf) * 8;
    for (int i = 0; i < kQRegDwords / 2; ++i) {
        const int off = base + i * 4;
        uint32_t pk = 0;
        for (int b = 0; b < 4; ++b) {
            const int kv_local = off + b;
            const uint8_t v =
                (kv_local < P_SLICE) ? p_fp8_row[kv_offset + kv_local] : static_cast<uint8_t>(0);
            pk |= static_cast<uint32_t>(v) << (8 * b);
        }
        p_compact[i] = pk;
    }
    for (int i = kQRegDwords / 2; i < kQRegDwords; ++i) {
        p_compact[i] = 0;
    }
}

template<int SUB_KV>
__device__ __forceinline__ void p_fp8_to_compact_regs(const uint8_t* p_fp8_row,
                                                      int lane,
                                                      uint32_t p_compact[kQRegDwords]) {
    p_fp8_to_compact_regs_slice<SUB_KV>(p_fp8_row, 0, lane, p_compact);
}

// asm step9: DPP row_shl:[8] on 32-bit P fp8 packs.
__device__ __forceinline__ uint32_t dpp_row_shl8_u32(uint32_t v) {
#if defined(__gfx942__) || defined(__gfx950__)
    const int bits = __builtin_amdgcn_mov_dpp(static_cast<int>(v), 0x114, 0xf, 0xf, true);
    return static_cast<uint32_t>(bits);
#else
    (void)v;
    return 0u;
#endif
}

// asm prologue: row_shl:[8] bound_ctrl on 0xffffffff -> _v_unPack (lanes 0..7 keep, 8..15 zero).
__device__ __forceinline__ uint32_t p_unpack_mask_u32(int lane) {
#if defined(__gfx942__) || defined(__gfx950__)
    (void)lane;
    return dpp_row_shl8_u32(0xffffffffu);
#else
    return ((lane & 15) < 8) ? 0xffffffffu : 0u;
#endif
}

// asm pa_fuse step8 (S-reshape wr/rd) + step9 (row_shl decompact) -> _v_S[0..15] for cl_gemm1.
__device__ __forceinline__ void p_reshape_step8_9(const uint32_t p_compact[2],
                                                      uint32_t* dyn_resp_lds,
                                                      int lane,
                                                      int wave,
                                                      uint32_t p_regs[16]) {
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        p_regs[k] = 0;
    }

    // step8 write: same as Q_reshape_wr / Dyn_qnt_resp_lds (v_S[0..1] at +0, +256 dwords).
    const uint32_t wr_base = q_reshape_wr_dword_offset(lane, wave);
    dyn_resp_lds[wr_base + 0] = p_compact[0];
    dyn_resp_lds[wr_base + 256] = p_compact[1];
    __syncthreads();

    // step8 read: ds_read_b64 into s_lane[0..7] (same addressing as Q_reshape_rd).
    const int h_id = lane >> 4;
    const int q_id = lane & 0xf;
    const uint32_t rd_base = static_cast<uint32_t>(h_id * 64 + q_id * 2);
    uint32_t s_lane[kSHalfRegSize];
#pragma unroll
    for (int k = 0; k < kSHalfRegSize; k += 2) {
        const uint32_t lds_off = static_cast<uint32_t>(((k & 3) / 2) * 32 + (k / 4) * 256);
        const uint32_t idx = rd_base + lds_off;
        s_lane[k + 0] = dyn_resp_lds[idx];
        s_lane[k + 1] = dyn_resp_lds[idx + 1];
    }

    // step9: v_S[8+k] = row_shl v_S[k]; v_S[k] &= _v_unPack.
    const uint32_t unpack = p_unpack_mask_u32(lane);
#pragma unroll
    for (int k = 0; k < kSHalfRegSize; ++k) {
        p_regs[k + kSHalfRegSize] = dpp_row_shl8_u32(s_lane[k]);
        p_regs[k] = s_lane[k] & unpack;
    }
}

__device__ __forceinline__ void p_reshape_for_gemm1(const uint32_t p_compact[2],
                                                    uint32_t* dyn_resp_lds,
                                                    int lane,
                                                    int wave,
                                                    uint32_t p_regs[16]) {
    p_reshape_step8_9(p_compact, dyn_resp_lds, lane, wave, p_regs);
}

template<int GQA, int SUB_KV, int P_SLICE>
__device__ __forceinline__ void p_f32_gather_lane_packed(const float p_f32[GQA][SUB_KV],
                                                         int kv_base,
                                                         int tile_kv,
                                                         const float* p_deq_scales,
                                                         int lane,
                                                         int wave,
                                                         uint32_t packed[2]) {
    constexpr int kRowsPerGroup = 4;
    const int col = lane & 15;
    const int row_group = lane >> 4;

    uint8_t bytes[8];
#pragma unroll
    for (int reg_k = 0; reg_k < 8; ++reg_k) {
        const int j = reg_k >> 2;
        const int inner_k = reg_k & 3;
        const int qi = row_group * kRowsPerGroup + inner_k;
        const int ki = kv_base + j * 64 + mfma_wave_n_offset(wave) + col;
        if (qi < GQA && ki < tile_kv) {
            const float inv_deq = 1.f / fmaxf(p_deq_scales[qi], 1e-30f);
            const float pn = p_f32[qi][ki] * inv_deq;
            bytes[reg_k] = float_to_fp8_e4m3_bias8(pn);
        } else {
            bytes[reg_k] = 0;
        }
    }
    packed[0] = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    packed[1] = static_cast<uint32_t>(bytes[4]) | (static_cast<uint32_t>(bytes[5]) << 8) |
                (static_cast<uint32_t>(bytes[6]) << 16) | (static_cast<uint32_t>(bytes[7]) << 24);
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void p_prepare_mfma_regs_from_scores_slice(
    const float p_f32[GQA][SUB_KV],
    int kv_offset,
    int tile_kv,
    const float* p_deq_scales,
    uint32_t* dyn_resp_lds,
    int lane,
    int wave,
    uint32_t p_regs[16]) {
    uint32_t compact[2] = {};
    p_f32_gather_lane_packed<GQA, SUB_KV, SUB_KV / 2>(p_f32, kv_offset, tile_kv, p_deq_scales,
                                                      lane, wave, compact);
    p_reshape_for_gemm1(compact, dyn_resp_lds, lane, wave, p_regs);
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void p_prepare_mfma_regs_slice(const uint8_t p_fp8[GQA][SUB_KV],
                                                          int kv_offset,
                                                          uint32_t* dyn_resp_lds,
                                                          int lane,
                                                          int wave,
                                                          uint32_t p_regs[16]) {
    uint32_t compact[2] = {};
    p_fp8_gather_lane_packed<GQA, SUB_KV, SUB_KV / 2>(p_fp8, kv_offset, lane, wave, compact);
    p_reshape_for_gemm1(compact, dyn_resp_lds, lane, wave, p_regs);
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void p_prepare_mfma_regs(const uint8_t p_fp8[GQA][SUB_KV],
                                                    uint32_t* dyn_resp_lds,
                                                    int lane,
                                                    int wave,
                                                    uint32_t p_regs[16]) {
    p_prepare_mfma_regs_slice<GQA, SUB_KV>(p_fp8, 0, dyn_resp_lds, lane, wave, p_regs);
}

}  // namespace pa_decode
