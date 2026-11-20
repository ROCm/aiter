#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fused/pipeline.hpp"
#include "mha_native_launch.h"

template <bool Wide>
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk1_split(FmhaFwdSplitParams sp) {
    __shared__ char lds[kLdsBytes];
    const int head_idx   = blockIdx.x;
    // Causal load balance: launch heavy high-index tiles first (matches CK).
    const int m_tile_idx = gridDim.y - 1 - blockIdx.y;
    const int b          = blockIdx.z / sp.num_splits;  // batch lives in z / G
    const int split_idx  = blockIdx.z % sp.num_splits;  // split lives in z % G
    fmha_fwd_d64_device<true, false, true, Wide>(
        sp.base, lds, b, head_idx, m_tile_idx,
        sp.scratch_o, sp.scratch_lse, sp.num_splits, split_idx);
}

void launch_msk1_split(const FmhaFwdSplitParams& sp, dim3 grid, hipStream_t stream) {
    if (aiter::fmha::requires_wide_scale(sp.base.scale))
        fmha_fwd_d64_bf16_msk1_split<true><<<grid, kBlockSize, 0, stream>>>(sp);
    else
        fmha_fwd_d64_bf16_msk1_split<false><<<grid, kBlockSize, 0, stream>>>(sp);
}
