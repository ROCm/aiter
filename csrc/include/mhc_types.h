#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>
#include "aiter_hip_common.h"

namespace cooperative_groups {
template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T, typename Op>
__device__ __forceinline__ T reduce(::cooperative_groups::thread_block_tile<32> tile, T val, Op op)
{
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = op(val, other);
    }
    return val;
}
}  // namespace cooperative_groups

namespace mhc {

using floatX = __hip_bfloat16;
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
using floatN = float;

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

// Use aiter's common HIP/CHECK macros (HIP_CALL, CHECK_COND) instead of local wrappers.
} // namespace mhc
