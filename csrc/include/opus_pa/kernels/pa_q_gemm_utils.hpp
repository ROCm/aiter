// Reference GEMM0 + validation helpers (Phase 3/4).
#pragma once

#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"
#include "opus_pa/kernels/pa_fp8_utils.hpp"
#include "opus_pa/kernels/pa_q_swizzle_utils.hpp"

namespace pa_decode {

__device__ __forceinline__ uint8_t fp8_from_dword(uint32_t w, int byte_idx) {
    return static_cast<uint8_t>((w >> (8 * byte_idx)) & 0xffu);
}

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

// Raw FP8 K byte at [kv, d] in the shuffled paged cache (mirrors gather_k_tile_linear_offset).
template<int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ uint8_t k_hbm_tile_byte(const uint8_t* k_pool,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   uint32_t stride_blk,
                                                   uint32_t stride_kvhead,
                                                   int kv_head_idx,
                                                   int kv,
                                                   int d) {
    const int blk = kv / BLOCK_SIZE;
    const int in_blk = kv % BLOCK_SIZE;
    if (blk >= valid_blks) {
        return 0;
    }
    const uint32_t page = page_ids[blk];
    const uint8_t* page_base = k_pool + static_cast<size_t>(page) * stride_blk +
                               static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    return page_base[k_shuffled_page_offset<HEAD_DIM, BLOCK_SIZE>(in_blk, d)];
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

// Gather K-slice (tile-local page_ids) at global KV offset kv_base.
template<int KV_SLICE, int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ void gather_k_tile_linear_offset(const uint8_t* k_pool,
                                                            const uint32_t* page_ids,
                                                            int valid_blks,
                                                            uint32_t stride_blk,
                                                            uint32_t stride_kvhead,
                                                            int kv_head_idx,
                                                            int kv_base,
                                                            uint8_t* k_tile_out) {
    const size_t kv_head_off =
        static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    for (int kv = threadIdx.x; kv < KV_SLICE; kv += blockDim.x) {
        const int global_kv = kv_base + kv;
        const int blk_idx = global_kv / BLOCK_SIZE;
        const int in_blk = global_kv % BLOCK_SIZE;
        const uint32_t page = (blk_idx < valid_blks) ? page_ids[blk_idx] : 0u;
        const uint8_t* page_base =
            k_pool + static_cast<size_t>(page) * stride_blk + kv_head_off;
        for (int d = 0; d < HEAD_DIM; ++d) {
            k_tile_out[kv * HEAD_DIM + d] =
                page_base[k_shuffled_page_offset<HEAD_DIM, BLOCK_SIZE>(in_blk, d)];
        }
    }
    __syncthreads();
}

// Reference GEMM0 into a slice of full S[qi, kv_base+ki].
template<int GQA, int KV_SLICE, int HEAD_DIM, int SUB_KV>
__device__ __forceinline__ void gemm0_qk_reference_offset(const uint8_t* q_fp8,
                                                          float q_deq_scale,
                                                          const uint8_t* k_tile,
                                                          float k_scale,
                                                          int kv_base,
                                                          float s_out[GQA][SUB_KV]) {
    for (int idx = threadIdx.x; idx < GQA * KV_SLICE; idx += blockDim.x) {
        const int qi = idx / KV_SLICE;
        const int ki = idx % KV_SLICE;
        float acc = 0.f;
#pragma unroll 4
        for (int d = 0; d < HEAD_DIM; ++d) {
            const float qv = fp8_e4m3_to_float(q_fp8[qi * HEAD_DIM + d]) * q_deq_scale;
            const float kv = fp8_e4m3_to_float(k_tile[ki * HEAD_DIM + d]) * k_scale;
            acc += qv * kv;
        }
        s_out[qi][kv_base + ki] = acc;
    }
    __syncthreads();
}

// Build S tile from BF16 Q in LDS (no Q FP8 round-trip) — closer to dequantized matmul.
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_SLICE>
__device__ __forceinline__ void build_scores_tile_bf16_lds(const bf16_t* q_lds,
                                                            int q_row_stride_elems,
                                                            const uint8_t* k_pool,
                                                            const uint32_t* page_ids,
                                                            int valid_blks,
                                                            uint32_t stride_blk,
                                                            uint32_t stride_kvhead,
                                                            int kv_head_idx,
                                                            int kv_nheads,
                                                            const float* k_scale_pool,
                                                            int block_size,
                                                            int tile_kv,
                                                            uint8_t* k_tile,
                                                            float (*s_tile)[SUB_KV]) {
    const int num_slices = (tile_kv + KV_SLICE - 1) / KV_SLICE;
    for (int slice = 0; slice < num_slices; ++slice) {
        const int kv_local_base = slice * KV_SLICE;
        const int slice_kv =
            (tile_kv - kv_local_base) < KV_SLICE ? (tile_kv - kv_local_base) : KV_SLICE;
        if (slice_kv <= 0) {
            break;
        }
        gather_k_tile_linear_offset<KV_SLICE, HEAD_DIM, BLOCK_SIZE>(
            k_pool, page_ids, valid_blks, stride_blk, stride_kvhead, kv_head_idx, kv_local_base,
            k_tile);
        for (int idx = threadIdx.x; idx < GQA * slice_kv; idx += blockDim.x) {
            const int qi = idx / slice_kv;
            const int ki = idx % slice_kv;
            const int global_kv = kv_local_base + ki;
            const int blk_idx = global_kv / BLOCK_SIZE;
            const int in_blk = global_kv % BLOCK_SIZE;
            const uint32_t page = (blk_idx < valid_blks) ? page_ids[blk_idx] : 0u;
            const float k_scale = k_scale_pool
                                      ? kv_scale_at(k_scale_pool, page, in_blk, block_size,
                                                    kv_nheads, kv_head_idx)
                                      : 1.f;
            float acc = 0.f;
            double acc64 = 0.0;
#pragma unroll 4
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float qv =
                    static_cast<float>(q_lds[qi * q_row_stride_elems + d]);
                const float kv = fp8_e4m3_to_float(k_tile[ki * HEAD_DIM + d]) * k_scale;
                acc64 += static_cast<double>(qv) * static_cast<double>(kv);
            }
            acc = static_cast<float>(acc64);
            s_tile[qi][kv_local_base + ki] = acc;
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < GQA * SUB_KV; idx += blockDim.x) {
        const int g = idx / SUB_KV;
        const int k = idx % SUB_KV;
        if (k >= tile_kv) {
            s_tile[g][k] = -1.0e30f;
        }
    }
    __syncthreads();
}

// Build S tile — fully dequantized Q/K dot products (reference GEMM0).
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_SLICE>
__device__ __forceinline__ void build_scores_tile_reference(const uint8_t* q_fp8,
                                                            const float* q_deq_scales,
                                                            const uint8_t* k_pool,
                                                            const uint32_t* page_ids,
                                                            int valid_blks,
                                                            uint32_t stride_blk,
                                                            uint32_t stride_kvhead,
                                                            int kv_head_idx,
                                                            int kv_nheads,
                                                            const float* k_scale_pool,
                                                            int block_size,
                                                            int tile_kv,
                                                            uint8_t* k_tile,
                                                            float (*s_tile)[SUB_KV]) {
    const int num_slices = (tile_kv + KV_SLICE - 1) / KV_SLICE;
    for (int slice = 0; slice < num_slices; ++slice) {
        const int kv_local_base = slice * KV_SLICE;
        const int slice_kv = (tile_kv - kv_local_base) < KV_SLICE ? (tile_kv - kv_local_base) : KV_SLICE;
        if (slice_kv <= 0) {
            break;
        }
        gather_k_tile_linear_offset<KV_SLICE, HEAD_DIM, BLOCK_SIZE>(
            k_pool, page_ids, valid_blks, stride_blk, stride_kvhead, kv_head_idx, kv_local_base,
            k_tile);
        for (int idx = threadIdx.x; idx < GQA * slice_kv; idx += blockDim.x) {
            const int qi = idx / slice_kv;
            const int ki = idx % slice_kv;
            const int global_kv = kv_local_base + ki;
            const int blk_idx = global_kv / BLOCK_SIZE;
            const int in_blk = global_kv % BLOCK_SIZE;
            const uint32_t page =
                (blk_idx < valid_blks) ? page_ids[blk_idx] : 0u;
            const float k_scale = k_scale_pool
                                      ? kv_scale_at(k_scale_pool, page, in_blk, block_size,
                                                    kv_nheads, kv_head_idx)
                                      : 1.f;
            const float q_deq = q_deq_scales[qi];
            float acc = 0.f;
            double acc64 = 0.0;
#pragma unroll 4
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float qv = fp8_e4m3_to_float(q_fp8[qi * HEAD_DIM + d]) * q_deq;
                const float kv = fp8_e4m3_to_float(k_tile[ki * HEAD_DIM + d]) * k_scale;
                acc64 += static_cast<double>(qv) * static_cast<double>(kv);
            }
            acc = static_cast<float>(acc64);
            s_tile[qi][kv_local_base + ki] = acc;
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < GQA * SUB_KV; idx += blockDim.x) {
        const int g = idx / SUB_KV;
        const int k = idx % SUB_KV;
        if (k >= tile_kv) {
            s_tile[g][k] = -1.0e30f;
        }
    }
    __syncthreads();
}

// Row-major Q fp8 (from swizzle pipeline unpacked) for reference check.
template<int GQA, int KV_SLICE, int HEAD_DIM>
__device__ __forceinline__ void unpack_q_fp8_rowmajor(const uint32_t q_regs[8],
                                                      uint8_t* q_row_out) {
    if (threadIdx.x != 0) {
        return;
    }
    for (int i = 0; i < 4; ++i) {
        q_row_out[i] = fp8_from_dword(q_regs[0], i);
    }
    for (int i = 0; i < 4; ++i) {
        q_row_out[4 + i] = fp8_from_dword(q_regs[1], i);
    }
    (void)GQA;
    (void)KV_SLICE;
    (void)HEAD_DIM;
}

// Reference GEMM0: S[qi, ki] = dot(Q[qi,:], K[ki,:]) with FP8 + scales.
template<int GQA, int KV_SLICE, int HEAD_DIM>
__device__ __forceinline__ void gemm0_qk_reference(const uint8_t* q_fp8,
                                                   float q_deq_scale,
                                                   const uint8_t* k_tile,
                                                   float k_scale,
                                                   float* s_out) {
    for (int idx = threadIdx.x; idx < GQA * KV_SLICE; idx += blockDim.x) {
        const int qi = idx / KV_SLICE;
        const int ki = idx % KV_SLICE;
        float acc = 0.f;
#pragma unroll 4
        for (int d = 0; d < HEAD_DIM; ++d) {
            const float qv = fp8_e4m3_to_float(q_fp8[qi * HEAD_DIM + d]) * q_deq_scale;
            const float kv = fp8_e4m3_to_float(k_tile[ki * HEAD_DIM + d]) * k_scale;
            acc += qv * kv;
        }
        s_out[idx] = acc;
    }
    __syncthreads();
}

// Gather row-major Q fp8 from post-reshape MFMA Q registers (asm v0-v7 layout).
// Each lane holds 32 fp8 (q_regs[0..7]); lanes qi, qi+16, qi+32, qi+48 cover 128 dims.
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void q_fp8_gather_from_mfma_regs(const uint32_t q_regs[8],
                                                            int lane,
                                                            uint8_t q_fp8_out[GQA * HEAD_DIM]) {
    const int qi = lane & 15;
    const int h_id = lane >> 4;
    if (qi >= GQA || h_id >= 4) {
        return;
    }
    for (int dw = 0; dw < kQRegDwords; ++dw) {
        const uint32_t w = q_regs[dw];
        for (int fp = 0; fp < 4; ++fp) {
            const int d = h_id * 32 + dw * 4 + fp;
            if (d < HEAD_DIM) {
                q_fp8_out[qi * HEAD_DIM + d] = fp8_from_dword(w, fp);
            }
        }
    }
}

// Inverse of q_fp8_gather_from_mfma_regs — pack row-major Q fp8 into MFMA q_regs per lane.
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void q_fp8_tile_to_mfma_regs(const uint8_t* q_fp8_tile,
                                                        int lane,
                                                        uint32_t q_regs[8]) {
    const int qi = lane & 15;
    const int h_id = lane >> 4;
#pragma unroll
    for (int dw = 0; dw < kQRegDwords; ++dw) {
        q_regs[dw] = 0;
    }
    if (qi >= GQA || h_id >= HEAD_DIM / 32) {
        return;
    }
#pragma unroll
    for (int dw = 0; dw < kQRegDwords; ++dw) {
        uint32_t w = 0;
#pragma unroll
        for (int fp = 0; fp < 4; ++fp) {
            const int d = h_id * 32 + dw * 4 + fp;
            if (d < HEAD_DIM) {
                w |= static_cast<uint32_t>(q_fp8_tile[qi * HEAD_DIM + d]) << (8 * fp);
            }
        }
        q_regs[dw] = w;
    }
}

template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void q_fp8_tile_from_mfma_regs(const uint32_t q_regs[8],
                                                          int lane,
                                                          uint8_t q_fp8_tile[GQA * HEAD_DIM]) {
    for (int i = threadIdx.x; i < GQA * HEAD_DIM; i += blockDim.x) {
        q_fp8_tile[i] = 0;
    }
    __syncthreads();
    q_fp8_gather_from_mfma_regs<GQA, HEAD_DIM>(q_regs, lane, q_fp8_tile);
    __syncthreads();
}

template<int GQA, int HEAD_DIM>
__device__ __forceinline__ float q_fp8_tile_max_abs_diff(const uint8_t* a,
                                                         const uint8_t* b) {
    float local = 0.f;
    for (int i = threadIdx.x; i < GQA * HEAD_DIM; i += blockDim.x) {
        local = fmaxf(local, fabsf(static_cast<float>(a[i]) - static_cast<float>(b[i])));
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return smem[0];
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

}  // namespace pa_decode
