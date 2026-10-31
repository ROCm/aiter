// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "attention_64rows.cuh"

hipError_t run_fused_norm_rope_attn_rope_cast_fwd_h16(const FusedNormRopeAttnRopeCastParams& args,
                                                      bool raw_q,
                                                      hipStream_t stream);

hipError_t run_fused_norm_rope_attn_rope_cast_h32_tile16_attention(
    const FusedNormRopeAttnRopeCastParams& args, bool raw_q, hipStream_t stream);

hipError_t run_fused_norm_rope_attn_rope_cast_fwd_small_heads(
    const FusedNormRopeAttnRopeCastParams& args, bool raw_q, hipStream_t stream)
{
    if(args.H == 16)
        return run_fused_norm_rope_attn_rope_cast_fwd_h16(args, raw_q, stream);
    if(args.H == 32)
    {
        if(fused_norm_rope_attn_rope_cast_policy::prefer_h32_split_head_tile16(
               args.N, args.num_splits))
        {
            const auto status =
                run_fused_norm_rope_attn_rope_cast_h32_tile16_attention(args, raw_q, stream);
            if(status != hipSuccess)
                return status;
            if(!args.debug_ptr && !args.max_ptr && !args.out_ptr &&
               (!args.lse_ptr || use_fused_split_reduce(args)))
                return launch_small_head_output_reduce(args, stream);
            fused_norm_rope_attn_rope_cast_decode_reduce_kernel<32, 0, false, 32>
                <<<dim3(args.N, 2), 256, 0, stream>>>(args);
            return hipGetLastError();
        }
        return run_fused_norm_rope_attn_rope_cast_small_heads<32>(args, raw_q, stream);
    }
    return hipErrorInvalidValue;
}
