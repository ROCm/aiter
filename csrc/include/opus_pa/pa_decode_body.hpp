// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus PA decode kernel body — single-file consolidation of the former
// opus_pa/kernels/*.hpp helpers (fp8 quant, device/layout utils, GEMM0/GEMM1
// MFMA tiles, online-softmax fuse, output write-back) plus the structural
// core_loop_tile + pa_decode_kernel_body. Included by opus_pa/pa_decode_kernel.hpp.
//
// LIVE build (module_pa_opus): PA_MFMA_MAIN_PATH, PA_MFMA_GEMM0, PA_MFMA_GEMM1,
// PA_GEMM1_VGATHER. fp8 16x16x32 MFMA on gfx942/gfx950.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <hip/hip_runtime.h>

#include "opus/opus.hpp"
#include "opus_pa/pa_decode_defs.h"

namespace pa_decode {

// ===========================================================================
// Q dynamic-quant shared constant
// ===========================================================================
static constexpr float kMaxDynBasisFp8 = 240.0f;

// ===========================================================================
// FP8 E4M3 (bias=8) encode helpers
// ===========================================================================
// Per-row dynamic-quant absmax for Q (fp8 MFMA path).
__device__ __forceinline__ float poc_kl_quant_row_abs(float x) {
    // Per-row dynamic-quant absmax for Q. Must be the true floating-point |x|:
    // truncating to int (static_cast<int>) collapses all |q|<1 values to 0, so the
    // row absmax degenerates to ~1e-6, the fp8 scale explodes (240/1e-6) and Q
    // overflows fp8 to NaN/saturation -> NaN scores -> softmax collapse -> zero/NaN
    // output. Only the fp8-Q MFMA path uses this, which is why the bf16 reference
    // path was unaffected.
    return fabsf(x);
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3_bias8(float x) {
#if defined(__gfx942__) || defined(__gfx950__)
    // Direct intrinsic (by-value, force-inlined) — opus::fp32_to_fp8 takes a
    // const-ref and is only `inline`, which spills x per element in the hot
    // quant loops and regresses the kernel ~2.5x. Keep the fnuz NaN/-0 flush.
    uint32_t w = 0;
    w = __builtin_amdgcn_cvt_pk_fp8_f32(x, x, w, false);
    const uint8_t out = static_cast<uint8_t>(w & 0xffu);
    return (out == 0x80u) ? static_cast<uint8_t>(0) : out;
#else
    (void)x;
    return 0;
#endif
}

// ===========================================================================
// Device helpers: lane/wave id, block-table + Q LDS load
// ===========================================================================
__device__ __forceinline__ int lane_id() { return threadIdx.x & 63; }

__device__ __forceinline__ int wave_id() { return threadIdx.x >> 6; }

// asm core_loop BT update: load page ids for KV range [kv_offset, kv_offset+SUB_KV).
template<int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ void load_block_table_tile_offset(const uint32_t* block_table,
                                                             int batch,
                                                             int max_blks,
                                                             uint32_t kv_seq,
                                                             int kv_offset,
                                                             uint32_t* out_page_ids,
                                                             int& valid_blks) {
    const int bt_dwords = SUB_KV / BLOCK_SIZE;
    const int bt_start = kv_offset / BLOCK_SIZE;
    const int total_blks = static_cast<int>((kv_seq + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const uint32_t tile_end_u = static_cast<uint32_t>(kv_offset + SUB_KV);
    const int tile_end = static_cast<int>(kv_seq < tile_end_u ? kv_seq : tile_end_u);
    valid_blks = (tile_end - kv_offset + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint32_t* bt_batch = block_table + batch * max_blks;
    for (int i = threadIdx.x; i < bt_dwords; i += blockDim.x) {
        const int global_blk = bt_start + i;
        out_page_ids[i] = (global_blk < total_blks) ? bt_batch[global_blk] : 0u;
    }
    __syncthreads();
}

template<int SUB_Q, int HEAD_DIM, int GQA_RATIO, int Q_LDS_ROWS, int Q_LDS_ROW_ELEMS>
__device__ __forceinline__ void load_q_tile_to_shared(const bf16_t* q_global,
                                                      bf16_t* q_lds,
                                                      int lds_row_stride_elems) {
    static_assert(Q_LDS_ROWS >= SUB_Q, "Q LDS rows must cover MFMA M");
    (void)lds_row_stride_elems;

    for (int row = 0; row < GQA_RATIO; ++row) {
        const bf16_t* src_row = q_global + row * HEAD_DIM;
        bf16_t* dst_row = q_lds + row * Q_LDS_ROW_ELEMS;
        for (int col = threadIdx.x; col < HEAD_DIM; col += blockDim.x) {
            dst_row[col] = src_row[col];
        }
        for (int col = HEAD_DIM + threadIdx.x; col < Q_LDS_ROW_ELEMS; col += blockDim.x) {
            dst_row[col] = bf16_t(0);
        }
    }

    for (int row = GQA_RATIO; row < Q_LDS_ROWS; ++row) {
        bf16_t* dst_row = q_lds + row * Q_LDS_ROW_ELEMS;
        for (int col = threadIdx.x; col < Q_LDS_ROW_ELEMS; col += blockDim.x) {
            dst_row[col] = bf16_t(0);
        }
    }
    __syncthreads();
}

// ===========================================================================
// MFMA accumulator type + u64 pack
// ===========================================================================
#if defined(__gfx942__) || defined(__gfx950__)
using mfma_acc4 = float __attribute__((ext_vector_type(4)));
#endif

__device__ __forceinline__ uint64_t pack_u64(uint32_t lo, uint32_t hi) {
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// ===========================================================================
// O rescale / accumulate / R_div_L write-out
// ===========================================================================
// asm pa_R_procss(part=0): rescale accumulated O by online softmax delta before new tile GEMM1.
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void pa_r_procss_rescale(float (*o_acc)[HEAD_DIM],
                                                    const float* delta_scale) {
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int g = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        o_acc[g][d] *= delta_scale[g];
    }
    __syncthreads();
}

template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void o_acc_add_tile(float (*o_acc)[HEAD_DIM],
                                               const float (*o_tile)[HEAD_DIM]) {
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int g = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        o_acc[g][d] += o_tile[g][d];
    }
    __syncthreads();
}

// asm R_div_L + R_write_out: normalize by L and store BF16 O (row-major GQA x HEAD_DIM).
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void r_write_out_bf16(const float (*o_acc)[HEAD_DIM],
                                                 const float* L_acc,
                                                 bf16_t* out_global,
                                                 int out_row_stride) {
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        const int g = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const float inv_l =
            static_cast<float>(1.0 / static_cast<double>(fmaxf(static_cast<double>(L_acc[g]), 1e-6)));
        out_global[g * out_row_stride + d] = static_cast<bf16_t>(o_acc[g][d] * inv_l);
    }
    __syncthreads();
}

// ===========================================================================
// Shuffled-page offsets, KV scale gather, Q fp8 quant
// ===========================================================================
// Shuffled K page layout (poc_kl fmha_shuffle_one): [dim/16, block_size, 16] within each page.
template<int HEAD_DIM, int BLOCK_SIZE, int TILE_MAJOR = 16>
__device__ __forceinline__ int k_shuffled_page_offset(int in_blk, int d) {
    constexpr int kTileStride = TILE_MAJOR * BLOCK_SIZE;  // 256 for 16x16 tiles
    return (d / TILE_MAJOR) * kTileStride + in_blk * TILE_MAJOR + (d % TILE_MAJOR);
}

// ASM V_shuffle layout: within each page, bytes are row-major [head_dim, block_size].
__device__ __forceinline__ int v_shuffled_page_offset(int in_blk, int d, int block_size) {
    return d * block_size + in_blk;
}

// asm scale layout: [num_blocks, num_kv_heads, block_size] (see KVQ_load_addr_offs_gen).
__device__ __forceinline__ float kv_scale_at(const float* scale_pool,
                                             uint32_t page_id,
                                             int in_blk,
                                             int block_size,
                                             int kv_nheads,
                                             int kv_head_idx) {
    return scale_pool[static_cast<size_t>(page_id) * static_cast<size_t>(kv_nheads) *
                          static_cast<size_t>(block_size) +
                      static_cast<size_t>(kv_head_idx) * static_cast<size_t>(block_size) +
                      static_cast<size_t>(in_blk)];
}

// Per-query row FP8 from LDS; writes matching q_deq_scales[g]=absmax/240.
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void q_rowmajor_fp8_per_query_from_lds(const bf16_t* q_lds,
                                                                  int q_row_stride_elems,
                                                                  uint8_t* q_fp8_out,
                                                                  float* q_deq_scales) {
    for (int g = 0; g < GQA; ++g) {
        float row_max = 1e-6f;
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            const float abs_val =
                poc_kl_quant_row_abs(static_cast<float>(q_lds[g * q_row_stride_elems + d]));
            row_max = fmaxf(row_max, abs_val);
        }
        __shared__ float row_max_smem[256];
        row_max_smem[threadIdx.x] = row_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                row_max_smem[threadIdx.x] =
                    fmaxf(row_max_smem[threadIdx.x], row_max_smem[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        const float absmax = row_max_smem[0];
        const float scale = kMaxDynBasisFp8 / absmax;
        if (threadIdx.x == 0) {
            q_deq_scales[g] = absmax / kMaxDynBasisFp8;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            const float qv = static_cast<float>(q_lds[g * q_row_stride_elems + d]);
            const float qn = qv * scale;
            q_fp8_out[g * HEAD_DIM + d] = float_to_fp8_e4m3_bias8(qn);
        }
        __syncthreads();
    }
}

// ===========================================================================
// MFMA 16x16 lane mapping + scatter/gather
// ===========================================================================
// acc[k] at MFMA 16x16 tile: row = 4*(lane/16)+k, col = lane%16 (within one 64-lane wave).
__device__ __forceinline__ void mfma16_lane_to_mn(int lane, int k, int& row, int& col) {
    col = lane & 15;
    row = ((lane >> 4) << 2) + k;
}

// Scatter one wave's MFMA GEMM0 slice (16x16) into S[query_base + row][kv_base + wave*16 + col].
template<int GQA, int SUB_KV>
__device__ __forceinline__ void mfma_scatter_scores_slice(const mfma_acc4& acc,
                                                          int lane,
                                                          int wave,
                                                          int query_base,
                                                          int kv_base,
                                                          int tile_kv,
                                                          float s_out[GQA][SUB_KV]) {
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int row = 0;
        int col = 0;
        mfma16_lane_to_mn(lane, k, row, col);
        const int qi = query_base + row;
        const int ki = kv_base + (wave << 4) + col;
        if (qi < GQA && ki < tile_kv) {
            s_out[qi][ki] = acc[k];
        }
    }
#else
    (void)acc;
    (void)lane;
    (void)wave;
    (void)query_base;
    (void)kv_base;
    (void)tile_kv;
    (void)s_out;
#endif
}

// Gather MFMA GEMM1 16x16 head slice into O[query_base + row][head_base + wave*16 + col].
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void mfma_gather_o_slice_accum(const mfma_acc4& acc,
                                                          int lane,
                                                          int wave,
                                                          int query_base,
                                                          int head_base,
                                                          float o_out[GQA][HEAD_DIM]) {
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int row = 0;
        int col = 0;
        mfma16_lane_to_mn(lane, k, row, col);
        const int qi = query_base + row;
        const int di = head_base + (wave << 4) + col;
        if (qi < GQA && di < HEAD_DIM) {
            o_out[qi][di] += acc[k];
        }
    }
#else
    (void)acc;
    (void)lane;
    (void)wave;
    (void)query_base;
    (void)head_base;
    (void)o_out;
#endif
}

// ===========================================================================
// Online softmax fuse + P MFMA A-operand gather
// ===========================================================================
static constexpr float kNegInf = -1.0e30f;

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
            rescale = exp2f((fa_old - fa_new) * scale_log2e);
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
        const float sm = exp2f(s_scores[g][gi] * scale_log2e - max_log2e);
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

// ===========================================================================
// V shuffled-page byte addressing + MFMA B gather
// ===========================================================================
// Expected HBM byte for V[kv,d] in tile (shuffled page layout).
__device__ __forceinline__ uint8_t v_hbm_tile_byte(const uint8_t* v_pool,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   uint32_t stride_blk,
                                                   uint32_t stride_kvhead,
                                                   int kv_head_idx,
                                                   int block_size,
                                                   int kv,
                                                   int d) {
    const int blk = kv / block_size;
    const int in_blk = kv % block_size;
    if (blk >= valid_blks) {
        return 0;
    }
    const uint32_t page = page_ids[blk];
    const size_t kv_head_off =
        static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    const uint8_t* page_base =
        v_pool + static_cast<size_t>(page) * stride_blk + kv_head_off;
    return page_base[v_shuffled_page_offset(in_blk, d, block_size)];
}

// Gather one MFMA B-operand pair (V[kv,d]) for P@V GEMM1.
// fp8 16x16x32 B[k][n]: n = lane%16, k = 8*(lane/16) + byte_i. Here n = head column
// (head = head_base + wave*16 + lane%16), k = kv contraction (kv = kv_base + 8*(lane/16) + i).
template<int SUB_KV, int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ void v_mfma_gather_b_pair(const uint8_t* v_pool,
                                                     const uint32_t* page_ids,
                                                     int valid_blks,
                                                     uint32_t stride_blk,
                                                     uint32_t stride_kvhead,
                                                     int kv_head_idx,
                                                     int block_size,
                                                     int tile_kv,
                                                     int kv_base,
                                                     int head_base,
                                                     int lane,
                                                     int wave,
                                                     uint32_t& lo,
                                                     uint32_t& hi) {
    const int col = lane & 15;
    const int g = lane >> 4;
    const int d = head_base + (wave << 4) + col;

    uint8_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int kv = kv_base + g * 8 + i;
        bytes[i] = (kv < tile_kv && d < HEAD_DIM)
                       ? v_hbm_tile_byte(v_pool, page_ids, valid_blks, stride_blk, stride_kvhead,
                                         kv_head_idx, block_size, kv, d)
                       : static_cast<uint8_t>(0);
    }
    lo = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    hi = static_cast<uint32_t>(bytes[4]) | (static_cast<uint32_t>(bytes[5]) << 8) |
         (static_cast<uint32_t>(bytes[6]) << 16) | (static_cast<uint32_t>(bytes[7]) << 24);
}

// ===========================================================================
// GEMM0 (Q@K) / GEMM1 (P@V) tiled MFMA main path
// ===========================================================================
static constexpr int kPiCount = 2;

template<int GQA, int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ void load_kv_scale_pi(const float* kq_base,
                                                   const float* vq_base,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   int kv_nheads,
                                                   int kv_head_idx,
                                                   int pi,
                                                   float k_scale_pi[SUB_KV / 2],
                                                   float v_scale_pi[SUB_KV / 2]) {
    constexpr int kHalf = SUB_KV / 2;
    constexpr int kBlksPerPi = SUB_KV / BLOCK_SIZE / 2;
    for (int i = threadIdx.x; i < kHalf; i += blockDim.x) {
        const int blk_local = i / BLOCK_SIZE;
        const int in_blk = i % BLOCK_SIZE;
        const int bt_slot = blk_local + pi * kBlksPerPi;
        float ks = 1.f;
        float vs = 1.f;
        if (kq_base != nullptr && bt_slot < valid_blks) {
            const uint32_t page = page_ids[bt_slot];
            ks = kv_scale_at(kq_base, page, in_blk, BLOCK_SIZE, kv_nheads, kv_head_idx);
            vs = kv_scale_at(vq_base, page, in_blk, BLOCK_SIZE, kv_nheads, kv_head_idx);
        }
        k_scale_pi[i] = ks;
        v_scale_pi[i] = vs;
    }
    __syncthreads();
}

// GEMM0 S = Q @ K^T via fp8 MFMA, built from gathered operands (probe-verified layout):
//   A = Q  -> A[m][k]: m = query (lane%16),        k = head = kk*32 + 8*(lane/16) + i
//   B = K  -> B[k][n]: n = kv-within-16 (lane%16), k = head
// D[m=query][n=kv] matches mfma_scatter_scores_slice (row=query, col=kv). Raw fp8 dot;
// q_deq/k_scale applied afterwards by scores_apply_qk_dequant.
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ void build_scores_mfma_pi(const uint8_t* q_fp8_tile,
                                                     const uint8_t* k_pool,
                                                     const uint32_t* page_ids,
                                                     int valid_blks,
                                                     uint32_t stride_blk,
                                                     uint32_t stride_kvhead,
                                                     int kv_head_idx,
                                                     int pi,
                                                     int kv_offset,
                                                     int tile_kv,
                                                     float s_out[GQA][SUB_KV]) {
    constexpr int kNumJ = SUB_KV / 64;   // 4 MFMA N-blocks per pi
    constexpr int kNumK = HEAD_DIM / 32; // 4 K-steps (head chunks of 32)

    const int lane = lane_id();
    const int wave = wave_id();
    const int query = lane & 15;
    const int g = lane >> 4;
    // page_ids and s_out are per-tile; use LOCAL kv indices (kv_offset is the global
    // tile base, only needed by the tail mask for ctx_len comparison).
    (void)kv_offset;
    const int kv_pi_base = pi * (SUB_KV / 2);
    const int bound = tile_kv;

    mfma_acc4 accs[kNumJ]{};
#if defined(__gfx942__) || defined(__gfx950__)
    using fp8_t = opus::fp8_t;
    // Whole per-wave GEMM0 as ONE tiled MFMA (E = M,N,K):
    //   A = Q [M=16, K=128]               -> E_M=1, E_K=kNumK
    //   B = K [N=64=kNumJ*16, K=128]      -> E_N=kNumJ, E_K=kNumK
    //   C = S [M=16 query, N=64 kv]       -> kNumJ blocks of mfma_acc4
    auto mma = opus::make_tiled_mma<fp8_t, fp8_t, opus::fp32_t>(
        opus::seq<1, kNumJ, kNumK>{}, opus::seq<1, 1, 1>{}, opus::seq<16, 16, 32>{});

    opus::vector_t<fp8_t, 8 * kNumK> qa;              // A operand (Q), K-step major
    opus::vector_t<fp8_t, 8 * kNumJ * kNumK> kb_all;  // B operand (K), (kv-block, K-step) major
    // A = Q (independent of kv-block j). Bytes go straight into the fp8 operand vector.
#pragma unroll
    for (int kk = 0; kk < kNumK; ++kk) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int head = kk * 32 + g * 8 + i;
            const uint8_t b = (query < GQA) ? q_fp8_tile[query * HEAD_DIM + head] : 0;
            qa[kk * 8 + i] = __builtin_bit_cast(fp8_t, static_cast<signed char>(b));
        }
    }
    // B = K, one 16-wide kv-block per j at kv_base = kv_pi_base + j*64.
    // The 8 fp8 bytes for one (j,kk) [head d0..d0+7] are contiguous in the shuffled
    // page, so gather each as ONE opus buffer_load_b64 (was 8x global_load_ubyte +
    // per-byte page re-resolve). Page base is resolved once per j.
    using fp8x8 = opus::vector_t<fp8_t, 8>;
#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        const int kv_lane = kv_pi_base + j * 64 + (wave << 4) + query;  // n = lane%16
        const int blk = kv_lane / BLOCK_SIZE;
        const int in_blk = kv_lane % BLOCK_SIZE;
        const bool valid = (kv_lane < bound) && (blk < valid_blks);
        const uint32_t page = valid ? page_ids[blk] : 0u;
        const uint8_t* page_base =
            k_pool + static_cast<size_t>(page) * stride_blk +
            static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
        auto gk = opus::make_gmem(reinterpret_cast<const fp8_t*>(page_base));
#pragma unroll
        for (int kk = 0; kk < kNumK; ++kk) {
            const int d0 = kk * 32 + g * 8;
            const int off = k_shuffled_page_offset<HEAD_DIM, BLOCK_SIZE>(in_blk, d0);
            fp8x8 kv8{};
            if (valid) kv8 = gk.template load<8>(off);
#pragma unroll
            for (int i = 0; i < 8; ++i) kb_all[(j * kNumK + kk) * 8 + i] = kv8[i];
        }
    }

    const auto c = mma(qa, kb_all);  // vtype_c = vector_t<fp32_t, 4 * kNumJ>
#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) accs[j][e] = c[j * 4 + e];
    }
#endif
#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        const int kv_base = kv_pi_base + j * 64;
        mfma_scatter_scores_slice<GQA, SUB_KV>(accs[j], lane, wave, 0, kv_base, bound, s_out);
    }
    __syncthreads();
}

// Accumulate MFMA GEMM1 head slice into O (multiply by per-query p_deq after MFMA).
// asm cl_gemm1(pi): vV_off = pi*sz_vV + j*sz_vV/2 + k*vs_AB, k in [0, 128/32).
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ void gemm1_mfma_pi(const uint8_t (*p_fp8)[SUB_KV],
                                              int pi,
                                              int query_base,
                                              const float* p_deq_scales,
                                              float o_out[GQA][HEAD_DIM],
                                              const uint8_t* v_pool,
                                              const uint32_t* page_ids,
                                              int valid_blks,
                                              uint32_t stride_blk,
                                              uint32_t stride_kvhead,
                                              int kv_head_idx,
                                              int block_size,
                                              int tile_kv,
                                              int lane,
                                              int wave) {
    constexpr int kHalf = SUB_KV / 2;
    constexpr int kNumJ = HEAD_DIM / 64;
    constexpr int kNumK = kHalf / 32;  // asm: 128/32 per pi
    constexpr int kPiKvSlice = 32;

    const int kv_pi_base = pi * kHalf;

    mfma_acc4 accs[kNumJ]{};
#if defined(__gfx942__) || defined(__gfx950__)
    // Whole per-wave GEMM1 as ONE tiled MFMA (E = M,N,K), A=P / B=V:
    //   A = P [M=16 query, K=128]            -> E_M=1, E_K=kNumK
    //   B = V [N=kNumJ*16 head, K=128]       -> E_N=kNumJ, E_K=kNumK
    //   C = R [M=16 query, N=head]           -> kNumJ blocks of mfma_acc4
    using fp8_t = opus::fp8_t;
    using fp8x8 = opus::vector_t<fp8_t, 8>;
    auto mma = opus::make_tiled_mma<fp8_t, fp8_t, opus::fp32_t>(
        opus::seq<1, kNumJ, kNumK>{}, opus::seq<1, 1, 1>{}, opus::seq<16, 16, 32>{});

    opus::vector_t<fp8_t, 8 * kNumK> pa;              // A operand (P), K-step major
    opus::vector_t<fp8_t, 8 * kNumJ * kNumK> vb;      // B operand (V), (head-block, K-step) major
    // A = P (shared across head-block j)
#pragma unroll
    for (int kk = 0; kk < kNumK; ++kk) {
        const int kv_base = kv_pi_base + kk * kPiKvSlice;
        uint32_t packed[2] = {};
        p_fp8_gather_lane_packed<GQA, SUB_KV, kPiKvSlice>(p_fp8, kv_base, lane, wave, packed);
        const fp8x8 pv = __builtin_bit_cast(fp8x8, pack_u64(packed[0], packed[1]));
#pragma unroll
        for (int i = 0; i < 8; ++i) pa[kk * 8 + i] = pv[i];
    }
    // B = V (per head-block j, K-step kk)
#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        const int head_base = j * 64;
#pragma unroll
        for (int kk = 0; kk < kNumK; ++kk) {
            const int kv_base = kv_pi_base + kk * kPiKvSlice;
            uint32_t v_lo = 0, v_hi = 0;
            v_mfma_gather_b_pair<SUB_KV, HEAD_DIM, BLOCK_SIZE>(
                v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, kv_head_idx, block_size,
                tile_kv, kv_base, head_base, lane, wave, v_lo, v_hi);
            const fp8x8 vv = __builtin_bit_cast(fp8x8, pack_u64(v_lo, v_hi));
#pragma unroll
            for (int i = 0; i < 8; ++i) vb[(j * kNumK + kk) * 8 + i] = vv[i];
        }
    }

    const auto c = mma(pa, vb);  // vtype_c = vector_t<fp32_t, 4 * kNumJ>
#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) accs[j][e] = c[j * 4 + e];
    }
#endif  // gfx942 || gfx950

    // Per-query p_deq scale + accumulate into O (unchanged), per head-block.
#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
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
            scaled[k] = accs[j][k] * deq;
        }
        mfma_gather_o_slice_accum<GQA, HEAD_DIM>(scaled, lane, wave, query_base, head_base, o_out);
    }

    __syncthreads();
}

// ===========================================================================
// Structural core loop (per-tile GEMM0 -> fuse -> GEMM1)
// ===========================================================================
namespace detail {

template<typename Traits>
__device__ __forceinline__ void core_loop_tile(int tile,
                               int kv_offset,
                               int tile_kv,
                               uint32_t ctx_len,
                               int valid_blks,
                               const pa_decode_kargs& kargs,
                               const uint32_t* page_ids,
                               uint32_t stride_blk,
                               float scale_log2e,
                               int kv_head_idx,
                               int lane,
                               int wave,
                               float* q_deq_scales,
                               float* fa_max,
                               float* L_acc,
                               float* delta_scale,
                               float (*o_acc)[Traits::HEAD_DIM],
                               float* k_scale_pi0,
                               float* k_scale_pi1,
                               float* v_scale_pi0,
                               float* v_scale_pi1,
                               float (*s_dense)[Traits::SUB_KV],
                               const uint8_t* q_fp8_tile,
                               uint8_t (*p_fp8)[Traits::SUB_KV],
                               float* p_deq_scales,
                               float (*o_tile)[Traits::HEAD_DIM]) {
    const int kv_nheads = static_cast<int>(kargs.kv_nheads);
    const float* kq_base = static_cast<const float*>(kargs.ptr_KQ);
    const float* vq_base = static_cast<const float*>(kargs.ptr_VQ);

    if (tile > 0) {
        pa_r_procss_rescale<Traits::GQA_RATIO, Traits::HEAD_DIM>(o_acc, delta_scale);
    }

    load_kv_scale_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::BLOCK_SIZE>(
        kq_base, vq_base, page_ids, valid_blks, kv_nheads, kv_head_idx, 0, k_scale_pi0,
        v_scale_pi0);
    load_kv_scale_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::BLOCK_SIZE>(
        kq_base, vq_base, page_ids, valid_blks, kv_nheads, kv_head_idx, 1, k_scale_pi1,
        v_scale_pi1);

    // GEMM0 (Q@Kᵀ) via asm MFMA gather. D[query][kv] is already the dense score
    // layout, so scatter straight into s_dense.
    for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * Traits::SUB_KV; idx += blockDim.x) {
        s_dense[idx / Traits::SUB_KV][idx % Traits::SUB_KV] = 0.f;
    }
    __syncthreads();

    build_scores_mfma_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::HEAD_DIM,
                         Traits::BLOCK_SIZE>(
        q_fp8_tile, static_cast<const uint8_t*>(kargs.ptr_K), page_ids, valid_blks, stride_blk,
        kargs.stride_kvhead, kv_head_idx, 0, kv_offset, tile_kv, s_dense);

    if (tile_kv < Traits::SUB_KV) {
        pa_tail_mask_dense<Traits::GQA_RATIO, Traits::SUB_KV>(s_dense, kv_offset, tile_kv,
                                                              ctx_len);
    }

    scores_apply_qk_dequant<Traits::GQA_RATIO, Traits::SUB_KV>(
        s_dense, tile_kv, q_deq_scales, k_scale_pi0, k_scale_pi1);

    if (tile_kv < Traits::SUB_KV) {
        pa_tail_mask_dense<Traits::GQA_RATIO, Traits::SUB_KV>(s_dense, kv_offset, tile_kv,
                                                              ctx_len);
    }

    pa_fuse_alu_tile_scales<Traits::GQA_RATIO, Traits::SUB_KV>(
        s_dense, tile_kv, q_deq_scales, k_scale_pi0, k_scale_pi1, v_scale_pi0, v_scale_pi1,
        scale_log2e, fa_max, L_acc, delta_scale, p_fp8, p_deq_scales);

    for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * Traits::HEAD_DIM; idx += blockDim.x) {
        o_tile[idx / Traits::HEAD_DIM][idx % Traits::HEAD_DIM] = 0.f;
    }
    __syncthreads();

    // GEMM1 (P@V) via MFMA, V gathered from HBM per pi (kPiCount halves of SUB_KV).
    for (int pi = 0; pi < kPiCount; ++pi) {
        gemm1_mfma_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::HEAD_DIM, Traits::BLOCK_SIZE>(
            p_fp8, pi, 0, p_deq_scales, o_tile, static_cast<const uint8_t*>(kargs.ptr_V),
            page_ids, valid_blks, stride_blk, kargs.stride_kvhead, kv_head_idx,
            Traits::BLOCK_SIZE, tile_kv, lane, wave);
    }

    o_acc_add_tile<Traits::GQA_RATIO, Traits::HEAD_DIM>(o_acc, o_tile);
}

}  // namespace detail

}  // namespace pa_decode

// ===========================================================================
// Global kernel body entry (instantiated by pa_decode_kernel.hpp)
// ===========================================================================
template<typename Traits>
__device__ void pa_decode_kernel_body(const pa_decode_kargs& kargs) {
    using namespace pa_decode;

    constexpr int kBtSlots = Traits::SUB_KV / Traits::BLOCK_SIZE;

    const int batch_id = static_cast<int>(blockIdx.y);
    const int kv_head_idx = static_cast<int>(blockIdx.x);
    const int lane = lane_id();
    const int wave = wave_id();

    const uint32_t stride_blk = kargs.stride_blk;
    const float scale_log2e = kargs.scale_log2e;

    const uint32_t* context_lens = static_cast<const uint32_t*>(kargs.ptr_CL);
    const uint32_t kv_seq = context_lens[batch_id];
    const int aligned_kv = static_cast<int>(kv_seq & ~static_cast<uint32_t>(Traits::SUB_KV - 1));
    const int num_aligned_tiles = aligned_kv / Traits::SUB_KV;
    const int tail_kv = static_cast<int>(kv_seq - static_cast<uint32_t>(aligned_kv));

    const uint32_t q_batch_stride = kargs.stride_Q_batch;
    const uint8_t* q_byte_base = static_cast<const uint8_t*>(kargs.ptr_Q);
    q_byte_base += kv_head_idx * kargs.stride_Q + batch_id * q_batch_stride;

    bf16_t* out_base = static_cast<bf16_t*>(kargs.ptr_O);
    out_base += kv_head_idx * kargs.stride_Q / static_cast<uint32_t>(sizeof(bf16_t)) +
                batch_id * q_batch_stride / static_cast<uint32_t>(sizeof(bf16_t));

    constexpr int kQRowStrideElems = Traits::Q_LDS_ROW_ELEMS;
    __shared__ bf16_t q_lds[Traits::Q_LDS_ROWS * kQRowStrideElems];

    load_q_tile_to_shared<Traits::SUB_Q, Traits::HEAD_DIM, Traits::GQA_RATIO,
                          Traits::Q_LDS_ROWS, Traits::Q_LDS_ROW_ELEMS>(
        reinterpret_cast<const bf16_t*>(q_byte_base), q_lds, kQRowStrideElems);

    __shared__ float q_deq_scales[Traits::GQA_RATIO];
    __shared__ uint8_t q_fp8_tile[Traits::GQA_RATIO * Traits::HEAD_DIM];

    q_rowmajor_fp8_per_query_from_lds<Traits::GQA_RATIO, Traits::HEAD_DIM>(
        q_lds, kQRowStrideElems, q_fp8_tile, q_deq_scales);

    __shared__ float fa_max[Traits::GQA_RATIO];
    __shared__ float L_acc[Traits::GQA_RATIO];
    __shared__ float delta_scale[Traits::GQA_RATIO];
    __shared__ uint8_t p_fp8[Traits::GQA_RATIO][Traits::SUB_KV];
    __shared__ float p_deq_scales[Traits::GQA_RATIO];
    __shared__ float o_tile[Traits::GQA_RATIO][Traits::HEAD_DIM];
    __shared__ float o_acc[Traits::GQA_RATIO][Traits::HEAD_DIM];
    __shared__ float k_scale_pi0[Traits::SUB_KV / 2];
    __shared__ float k_scale_pi1[Traits::SUB_KV / 2];
    __shared__ float v_scale_pi0[Traits::SUB_KV / 2];
    __shared__ float v_scale_pi1[Traits::SUB_KV / 2];
    __shared__ uint32_t page_ids[kBtSlots];
    __shared__ uint32_t pages_snap[kBtSlots];
    __shared__ float s_dense[Traits::GQA_RATIO][Traits::SUB_KV];

    for (int g = threadIdx.x; g < Traits::GQA_RATIO; g += blockDim.x) {
        fa_max[g] = kNegInf;
        L_acc[g] = 0.f;
        delta_scale[g] = 1.f;
    }
    for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * Traits::HEAD_DIM; idx += blockDim.x) {
        o_acc[idx / Traits::HEAD_DIM][idx % Traits::HEAD_DIM] = 0.f;
    }
    __syncthreads();

    for (int tile = 0; tile < num_aligned_tiles; ++tile) {
        const int kv_offset = tile * Traits::SUB_KV;
        int valid_blks = 0;
        load_block_table_tile_offset<Traits::SUB_KV, Traits::BLOCK_SIZE>(
            static_cast<const uint32_t*>(kargs.ptr_BT), batch_id, kargs.max_blks, kv_seq, kv_offset,
            page_ids, valid_blks);
        for (int i = threadIdx.x; i < kBtSlots; i += blockDim.x) {
            pages_snap[i] = page_ids[i];
        }
        __syncthreads();

        detail::core_loop_tile<Traits>(tile, kv_offset, Traits::SUB_KV, kv_seq, valid_blks,
                                    kargs, pages_snap, stride_blk, scale_log2e, kv_head_idx,
                                    lane, wave, &q_deq_scales[0], &fa_max[0], &L_acc[0],
                                    &delta_scale[0], &o_acc[0], &k_scale_pi0[0], &k_scale_pi1[0], &v_scale_pi0[0],
                                    &v_scale_pi1[0], &s_dense[0], q_fp8_tile, &p_fp8[0], &p_deq_scales[0],
                                    &o_tile[0]);
    }

    if (tail_kv > 0) {
        const int kv_offset = aligned_kv;
        int valid_blks = 0;
        load_block_table_tile_offset<Traits::SUB_KV, Traits::BLOCK_SIZE>(
            static_cast<const uint32_t*>(kargs.ptr_BT), batch_id, kargs.max_blks, kv_seq, kv_offset,
            page_ids, valid_blks);
        for (int i = threadIdx.x; i < kBtSlots; i += blockDim.x) {
            pages_snap[i] = page_ids[i];
        }
        __syncthreads();

        detail::core_loop_tile<Traits>(num_aligned_tiles, kv_offset, tail_kv, kv_seq, valid_blks,
                                    kargs, pages_snap, stride_blk, scale_log2e, kv_head_idx,
                                    lane, wave, &q_deq_scales[0], &fa_max[0], &L_acc[0],
                                    &delta_scale[0],
                                    &o_acc[0], &k_scale_pi0[0], &k_scale_pi1[0], &v_scale_pi0[0],
                                    &v_scale_pi1[0], &s_dense[0], q_fp8_tile, &p_fp8[0], &p_deq_scales[0],
                                    &o_tile[0]);
    }

    r_write_out_bf16<Traits::GQA_RATIO, Traits::HEAD_DIM>(&o_acc[0], &L_acc[0], out_base,
                                                          Traits::HEAD_DIM);
}
