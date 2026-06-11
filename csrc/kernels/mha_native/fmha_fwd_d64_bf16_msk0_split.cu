#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fused/pipeline.hpp"
#include "mha_native_launch.h"

__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk0_split(FmhaFwdSplitParams sp) {
    __shared__ char lds[kLdsBytes];
    const int head_idx   = blockIdx.x;
    const int m_tile_idx = blockIdx.y;                  // natural order (no mask)
    const int b          = blockIdx.z / sp.num_splits;  // batch lives in z / G
    const int split_idx  = blockIdx.z % sp.num_splits;  // split lives in z % G
    fmha_fwd_d64_device<false, false, true>(
        sp.base, lds, b, head_idx, m_tile_idx,
        sp.scratch_o, sp.scratch_lse, sp.num_splits, split_idx);
}

void launch_msk0_split(const FmhaFwdSplitParams& sp, dim3 grid, hipStream_t stream) {
    fmha_fwd_d64_bf16_msk0_split<<<grid, kBlockSize, 0, stream>>>(sp);
}
