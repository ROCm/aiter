// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// H64 instantiation and dispatch. H128 is compiled separately to shorten cold builds.
#include "attention_64rows.cuh"

extern template hipError_t
run_fused_norm_rope_attn_rope_cast_heads<128>(const FusedNormRopeAttnRopeCastParams& args,
                                              hipStream_t stream);

namespace {

template <int Splits>
hipError_t launch_h64_split_head_tile32(const FusedNormRopeAttnRopeCastParams& args,
                                        hipStream_t stream)
{
    fused_norm_rope_attn_rope_cast_decode_kernel<64, true, Splits, 32>
        <<<dim3(args.N, 2, Splits), 256, 0, stream>>>(args);
    const auto status = hipGetLastError();
    if(status != hipSuccess)
        return status;
    return launch_h64_head_tile32_reduce(args, stream);
}

} // namespace

hipError_t run_fused_norm_rope_attn_rope_cast_fwd(const FusedNormRopeAttnRopeCastParams& args,
                                                  hipStream_t stream)
{
    if(args.H == 16 || args.H == 32)
        return run_fused_norm_rope_attn_rope_cast_fwd_small_heads(args, stream);
    if(args.H == 64)
    {
        const int64_t capacity = int64_t(args.prefix_capacity) + args.extend_capacity;
        const int64_t blocks   = int64_t(args.N) * 2 * args.num_splits;
        // Short lists benefit from smaller accumulator tiles while the split grid
        // fits one CTA per CU. Keep the existing path for larger grids/lists.
        if(!args.debug_ptr && !args.max_ptr && !args.out_ptr && capacity >= int64_t(args.N) * 512 &&
           capacity < int64_t(args.N) * 768 && blocks >= 128 && blocks <= 256)
        {
            if(args.num_splits == 4)
                return launch_h64_split_head_tile32<4>(args, stream);
            if(args.num_splits == 5)
                return launch_h64_split_head_tile32<5>(args, stream);
        }
        return run_fused_norm_rope_attn_rope_cast_heads<64>(args, stream);
    }
    if(args.H == 128)
        return run_fused_norm_rope_attn_rope_cast_heads<128>(args, stream);
    return hipErrorInvalidValue;
}
