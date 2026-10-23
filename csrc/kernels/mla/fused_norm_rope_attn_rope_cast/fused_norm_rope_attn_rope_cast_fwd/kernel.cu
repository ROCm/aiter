// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "decode.cuh"
#include "kernel.cuh"
#include "kernel.h"

template <int CacheMode, bool RawQ, bool Continuous = false, bool DirectQ = false>
__global__ __launch_bounds__(256, 1) void fused_norm_rope_attn_rope_cast_fwd_kernel(
    FusedNormRopeAttnRopeCastParams args)
{
    using Config = FusedNormRopeAttnRopeCastConfig<CacheMode, RawQ, DirectQ>;
    __shared__ __align__(16) char smem[Config::smem_bytes() + 2 * 68 * sizeof(int)];
    if(indices_valid<Config>(args))
        fused_norm_rope_attn_rope_cast_fwd_impl<Config, false, Continuous>(args, smem);
    else
        fused_norm_rope_attn_rope_cast_fwd_impl<Config, true>(args, smem);
}

hipError_t run_fused_norm_rope_attn_rope_cast_fwd(const FusedNormRopeAttnRopeCastParams& args,
                                                  bool raw_q,
                                                  hipStream_t stream)
{
    const dim3 grid(args.N, args.H / 64);
    if(args.num_splits > 1)
    {
        const dim3 split_grid(args.N, args.H / 64, args.num_splits);
        if(raw_q && args.H == 64)
            fused_norm_rope_attn_rope_cast_decode_kernel<true, true>
                <<<split_grid, 256, 0, stream>>>(args);
        else if(raw_q)
            fused_norm_rope_attn_rope_cast_decode_kernel<true>
                <<<split_grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_decode_kernel<false>
                <<<split_grid, 256, 0, stream>>>(args);
        const auto status = hipGetLastError();
        if(status != hipSuccess)
            return status;
        fused_norm_rope_attn_rope_cast_decode_reduce_kernel<<<dim3(args.N, args.H / 16),
                                                              256,
                                                              0,
                                                              stream>>>(args);
        return hipGetLastError();
    }
    if(args.cache_mode == 0)
    {
        // Buffer capacities are known on the host; a single-source request keeps
        // the original kernel without a device-side dispatch or extra scan.
        if(args.prefix_capacity > 0 && args.extend_capacity > 0)
        {
            if(raw_q && args.H == 64)
                fused_norm_rope_attn_rope_cast_fwd_kernel<0, true, true, true>
                    <<<grid, 256, 0, stream>>>(args);
            else if(raw_q)
                fused_norm_rope_attn_rope_cast_fwd_kernel<0, true, true>
                    <<<grid, 256, 0, stream>>>(args);
            else
                fused_norm_rope_attn_rope_cast_fwd_kernel<0, false, true>
                    <<<grid, 256, 0, stream>>>(args);
        }
        else if(raw_q && args.H == 64)
            fused_norm_rope_attn_rope_cast_fwd_kernel<0, true, false, true>
                <<<grid, 256, 0, stream>>>(args);
        else if(raw_q)
            fused_norm_rope_attn_rope_cast_fwd_kernel<0, true><<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_fwd_kernel<0, false><<<grid, 256, 0, stream>>>(args);
    }
    else if(args.cache_mode == 1)
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
