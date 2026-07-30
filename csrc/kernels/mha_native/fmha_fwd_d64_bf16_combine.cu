#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fused/op_combine.hpp"
#include "mha_native_launch.h"

__global__ void __launch_bounds__(kBlockSize)
fmha_fwd_d64_bf16_combine(FmhaFwdCombineParams params) {
    combine_split(params, blockIdx.z, blockIdx.x, blockIdx.y);
}

void launch_combine(const FmhaFwdCombineParams& cp, dim3 grid, hipStream_t stream) {
    fmha_fwd_d64_bf16_combine<<<grid, kBlockSize, 0, stream>>>(cp);
}
