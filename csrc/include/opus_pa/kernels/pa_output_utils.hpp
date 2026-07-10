// asm pa_R_procss + R_write_out (Phase 6 smoke).
#pragma once

#include <cstdint>
#include <cstring>

#include <hip/hip_runtime.h>

#include "opus_pa/pa_decode_defs.h"

namespace pa_decode {

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

}  // namespace pa_decode
