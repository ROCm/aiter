// Debug: per-slice compare reference vs MFMA S scores.
#pragma once

#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/kernels/pa_decode_device_utils.hpp"

namespace pa_decode {

// asm MFMA scatter indices: kv = pi*(SUB_KV/2) + j*64 + col, j in [0, SUB_KV/64).
template<int SUB_KV>
__device__ __forceinline__ int mfma_sparse_kv_index(int pi, int j, int col) {
    return pi * (SUB_KV / 2) + j * 64 + col;
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compare_mfma_sparse(const float s_ref[GQA][SUB_KV],
                                                            const float s_mfma[GQA][SUB_KV],
                                                            int tile_kv,
                                                            float& max_abs,
                                                            float& max_rel,
                                                            int& worst_kv) {
    constexpr int kNumJ = (SUB_KV / 2) / 64;
    float local_max_abs = 0.f;
    float local_max_rel = 0.f;
    int local_worst = -1;

    for (int idx = threadIdx.x; idx < GQA * kPiCount * kNumJ * 16; idx += blockDim.x) {
        const int g = idx / (kPiCount * kNumJ * 16);
        const int rem = idx % (kPiCount * kNumJ * 16);
        const int pi = rem / (kNumJ * 16);
        const int rem2 = rem % (kNumJ * 16);
        const int j = rem2 / 16;
        const int col = rem2 % 16;
        const int gi = mfma_sparse_kv_index<SUB_KV>(pi, j, col);
        if (gi >= tile_kv) {
            continue;
        }
        const float ref = s_ref[g][gi];
        const float got = s_mfma[g][gi];
        const float d = fabsf(got - ref);
        if (d > local_max_abs) {
            local_max_abs = d;
            local_max_rel = d / fmaxf(fabsf(ref), 1e-6f);
            local_worst = gi;
        }
    }

    __shared__ float smem_abs[256];
    __shared__ float smem_rel[256];
    __shared__ int smem_kv[256];
    smem_abs[threadIdx.x] = local_max_abs;
    smem_rel[threadIdx.x] = local_max_rel;
    smem_kv[threadIdx.x] = local_worst;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem_abs[threadIdx.x + stride] > smem_abs[threadIdx.x]) {
                smem_abs[threadIdx.x] = smem_abs[threadIdx.x + stride];
                smem_rel[threadIdx.x] = smem_rel[threadIdx.x + stride];
                smem_kv[threadIdx.x] = smem_kv[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        max_abs = smem_abs[0];
        max_rel = smem_rel[0];
        worst_kv = smem_kv[0];
    }
    __syncthreads();
}

// Per KV-slice dense compare at MFMA sparse write sites only.
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compare_sparse_slice(const float s_ref[GQA][SUB_KV],
                                                            const float s_mfma[GQA][SUB_KV],
                                                            int tile_kv,
                                                            int kv_lo,
                                                            int kv_hi,
                                                            float& slice_max_abs,
                                                            float& slice_max_rel) {
    constexpr int kNumJ = (SUB_KV / 2) / 64;
    float local_max_abs = 0.f;
    float local_max_rel = 0.f;
    for (int idx = threadIdx.x; idx < GQA * kPiCount * kNumJ * 16; idx += blockDim.x) {
        const int g = idx / (kPiCount * kNumJ * 16);
        const int rem = idx % (kPiCount * kNumJ * 16);
        const int pi = rem / (kNumJ * 16);
        const int rem2 = rem % (kNumJ * 16);
        const int j = rem2 / 16;
        const int col = rem2 % 16;
        const int gi = mfma_sparse_kv_index<SUB_KV>(pi, j, col);
        if (gi < kv_lo || gi >= kv_hi || gi >= tile_kv) {
            continue;
        }
        const float d = fabsf(s_mfma[g][gi] - s_ref[g][gi]);
        local_max_abs = fmaxf(local_max_abs, d);
        local_max_rel = fmaxf(local_max_rel, d / fmaxf(fabsf(s_ref[g][gi]), 1e-6f));
    }
    __shared__ float smem_abs[256];
    __shared__ float smem_rel[256];
    smem_abs[threadIdx.x] = local_max_abs;
    smem_rel[threadIdx.x] = local_max_rel;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_abs[threadIdx.x] = fmaxf(smem_abs[threadIdx.x], smem_abs[threadIdx.x + stride]);
            smem_rel[threadIdx.x] = fmaxf(smem_rel[threadIdx.x], smem_rel[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        slice_max_abs = smem_abs[0];
        slice_max_rel = smem_rel[0];
    }
    __syncthreads();
}

// Per KV-slice (width=SUB_KV/num_slices) dense compare.
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compare_dense_slice(const float s_ref[GQA][SUB_KV],
                                                           const float s_dense[GQA][SUB_KV],
                                                           int tile_kv,
                                                           int kv_base,
                                                           int slice_kv,
                                                           float& slice_max_abs,
                                                           float& slice_max_rel) {
    float local_max_abs = 0.f;
    float local_max_rel = 0.f;
    for (int idx = threadIdx.x; idx < GQA * slice_kv; idx += blockDim.x) {
        const int g = idx / slice_kv;
        const int k = idx % slice_kv;
        const int gi = kv_base + k;
        if (gi >= tile_kv) {
            continue;
        }
        const float d = fabsf(s_dense[g][gi] - s_ref[g][gi]);
        local_max_abs = fmaxf(local_max_abs, d);
        local_max_rel = fmaxf(local_max_rel, d / fmaxf(fabsf(s_ref[g][gi]), 1e-6f));
    }
    __shared__ float smem_abs[256];
    __shared__ float smem_rel[256];
    smem_abs[threadIdx.x] = local_max_abs;
    smem_rel[threadIdx.x] = local_max_rel;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_abs[threadIdx.x] = fmaxf(smem_abs[threadIdx.x], smem_abs[threadIdx.x + stride]);
            smem_rel[threadIdx.x] = fmaxf(smem_rel[threadIdx.x], smem_rel[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        slice_max_abs = smem_abs[0];
        slice_max_rel = smem_rel[0];
    }
    __syncthreads();
}

// Debug output: per-slice dense compare (compact MFMA vs ref).
template<int GQA, int SUB_KV, int NUM_SLICES>
__device__ __forceinline__ void scores_dump_dense_slice_diffs(const float s_ref[GQA][SUB_KV],
                                                              const float s_dense[GQA][SUB_KV],
                                                              int tile_kv,
                                                              float* dbg_out) {
    constexpr int kSliceW = SUB_KV / NUM_SLICES;
    for (int s = 0; s < NUM_SLICES; ++s) {
        const int kv_base = s * kSliceW;
        const int slice_kv = min(kSliceW, tile_kv - kv_base);
        float abs_d = 0.f;
        float rel_d = 0.f;
        if (slice_kv > 0) {
            scores_compare_dense_slice<GQA, SUB_KV>(s_ref, s_dense, tile_kv, kv_base, slice_kv,
                                                    abs_d, rel_d);
        }
        if (threadIdx.x == 0) {
            dbg_out[s * 2 + 0] = abs_d;
            dbg_out[s * 2 + 1] = rel_d;
        }
        __syncthreads();
    }
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compare_dense_all(const float s_ref[GQA][SUB_KV],
                                                         const float s_dense[GQA][SUB_KV],
                                                         int tile_kv,
                                                         float& max_abs,
                                                         float& max_rel,
                                                         int& worst_kv) {
    float local_max_abs = 0.f;
    float local_max_rel = 0.f;
    int local_worst = -1;
    for (int idx = threadIdx.x; idx < GQA * tile_kv; idx += blockDim.x) {
        const int g = idx / tile_kv;
        const int gi = idx % tile_kv;
        const float d = fabsf(s_dense[g][gi] - s_ref[g][gi]);
        if (d > local_max_abs) {
            local_max_abs = d;
            local_max_rel = d / fmaxf(fabsf(s_ref[g][gi]), 1e-6f);
            local_worst = gi;
        }
    }
    __shared__ float smem_abs[256];
    __shared__ float smem_rel[256];
    __shared__ int smem_kv[256];
    smem_abs[threadIdx.x] = local_max_abs;
    smem_rel[threadIdx.x] = local_max_rel;
    smem_kv[threadIdx.x] = local_worst;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (smem_abs[threadIdx.x + stride] > smem_abs[threadIdx.x]) {
                smem_abs[threadIdx.x] = smem_abs[threadIdx.x + stride];
                smem_rel[threadIdx.x] = smem_rel[threadIdx.x + stride];
                smem_kv[threadIdx.x] = smem_kv[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        max_abs = smem_abs[0];
        max_rel = smem_rel[0];
        worst_kv = smem_kv[0];
    }
    __syncthreads();
}

// Debug output: per-slice stats packed as [abs0, rel0, abs1, rel1, ...].
template<int GQA, int SUB_KV, int NUM_SLICES>
__device__ __forceinline__ void scores_dump_slice_diffs(const float s_ref[GQA][SUB_KV],
                                                        const float s_mfma[GQA][SUB_KV],
                                                        int tile_kv,
                                                        float* dbg_out) {
    constexpr int kSliceW = SUB_KV / NUM_SLICES;
    for (int s = 0; s < NUM_SLICES; ++s) {
        const int kv_base = s * kSliceW;
        const int slice_kv = min(kSliceW, tile_kv - kv_base);
        float abs_d = 0.f;
        float rel_d = 0.f;
        if (slice_kv > 0) {
            scores_compare_sparse_slice<GQA, SUB_KV>(s_ref, s_mfma, tile_kv, kv_base,
                                                     kv_base + slice_kv, abs_d, rel_d);
        }
        if (threadIdx.x == 0) {
            dbg_out[s * 2 + 0] = abs_d;
            dbg_out[s * 2 + 1] = rel_d;
        }
        __syncthreads();
    }
}

}  // namespace pa_decode
