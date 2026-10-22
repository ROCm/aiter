// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "kernel.cuh"
#include "kernel.h"

template <int CacheMode, bool RawQ>
__global__ __launch_bounds__(256, 1) void fused_norm_rope_attn_rope_cast_fwd_kernel(
    FusedNormRopeAttnRopeCastParams args)
{
    using Config = FusedNormRopeAttnRopeCastConfig<CacheMode, RawQ>;
    __shared__ __align__(16) char smem[Config::smem_bytes() + 2 * 68 * sizeof(int)];
    if(indices_valid<Config>(args))
        fused_norm_rope_attn_rope_cast_fwd_impl<Config, false>(args, smem);
    else
        fused_norm_rope_attn_rope_cast_fwd_impl<Config, true>(args, smem);
}

hipError_t run_fused_norm_rope_attn_rope_cast_fwd(const FusedNormRopeAttnRopeCastParams& args,
                                                  bool raw_q,
                                                  hipStream_t stream)
{
    const dim3 grid(args.N, 2);
    if(args.cache_mode == 1)
    {
        if(raw_q)
            fused_norm_rope_attn_rope_cast_fwd_kernel<1, true><<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_fwd_kernel<1, false><<<grid, 256, 0, stream>>>(args);
    }
    else
    {
        if(raw_q)
            fused_norm_rope_attn_rope_cast_fwd_kernel<2, true><<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_fwd_kernel<2, false><<<grid, 256, 0, stream>>>(args);
    }
    return hipGetLastError();
}
