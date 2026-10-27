// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "q256_producer_args.hpp"

AITER_CTYPES_ERROR_DEF

// The fused expert kernel has its own kernarg layout, separate from the shared fmoe ABI.
// Performs gate/up projection, SiLU, in-register hidden MXFP4 quantization,
// down projection and routing-weight multiplication. The compact contributions
// are reduced separately in FP32 to produce the final output tensor.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_q256_fused_experts,
    (aiter_tensor_t* out, aiter_tensor_t* partials, aiter_tensor_t* input,
     aiter_tensor_t* gate, aiter_tensor_t* down, aiter_tensor_t* sorted_ids,
     aiter_tensor_t* sorted_weights, aiter_tensor_t* sorted_experts,
     aiter_tensor_t* counts,
     aiter_tensor_t* input_scale, aiter_tensor_t* gate_scale,
     aiter_tensor_t* down_scale, int topk, aiter_tensor_t* job_map, hipStream_t stream),
    (out, partials, input, gate, down, sorted_ids, sorted_weights,
     sorted_experts, counts, input_scale, gate_scale,
     down_scale, topk, job_map, stream))
{
    AITER_CHECK(input, "Q256 requires an input tensor");
    const HipDeviceGuard guard(input->device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", "Q256 requires gfx950");
    for(auto* tensor : {out, partials, input, gate, down, sorted_ids,
                        sorted_weights, sorted_experts, counts,
                        input_scale, gate_scale, down_scale})
        AITER_CHECK(tensor && tensor->is_contiguous() &&
                    tensor->device_id == input->device_id,
                    "Q256 requires contiguous tensors on the same GPU");
    AITER_CHECK(out->dim() == 2 && input->dim() == 2 && gate->dim() == 3 &&
                down->dim() == 3 && partials->dim() == 3,
                "Q256 requires rank-2 input/output and rank-3 weights/contributions");
    const int64_t tokens = out->size(0), model = out->size(1);
    const int64_t experts = gate->size(0), capacity = sorted_ids->numel();
    AITER_CHECK(partials->size(0) == tokens &&
                partials->size(1) == topk && partials->size(2) == model,
                "Q256 compact contributions must be [tokens, topk, model]");
    AITER_CHECK(tokens > 0 && tokens < (1 << 24) && model >= 512 && model % 256 == 0 &&
                topk > 0 && topk <= 127 && topk <= experts && capacity > 0 && capacity % 256 == 0,
                "Unsupported Q256 geometry");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16 && partials->dtype() == AITER_DTYPE_bf16 &&
                input->element_size() == 1 && gate->element_size() == 1 && down->element_size() == 1 &&
                input_scale->element_size() == 1 && gate_scale->element_size() == 1 &&
                down_scale->element_size() == 1 && sorted_ids->dtype() == AITER_DTYPE_i32 &&
                sorted_experts->dtype() == AITER_DTYPE_i32 && counts->dtype() == AITER_DTYPE_i32 &&
                sorted_weights->dtype() == AITER_DTYPE_fp32,
                "Unsupported Q256 tensor dtype");
    AITER_CHECK(input->size(0) == tokens && input->size(1) == model / 2 &&
                gate->size(1) == 512 && gate->size(2) == model / 2 &&
                down->size(0) == experts && down->size(1) == model && down->size(2) == 128 &&
                partials->numel() == tokens * topk * model && sorted_weights->numel() == capacity &&
                sorted_experts->numel() >= capacity / 256 && counts->numel() == 2 &&
                input_scale->numel() >= capacity * model / 32 &&
                gate_scale->numel() == experts * 512 * model / 32 &&
                down_scale->numel() == experts * model * 8,
                "Q256 tensor extent mismatch");
    for(uint64_t bytes : {uint64_t(tokens * model / 2), uint64_t(experts * 256 * model),
                          uint64_t(capacity * model / 32), uint64_t(capacity * 4),
                          uint64_t(model * 512)})
        AITER_CHECK(bytes < (uint64_t(1) << 32), "Q256 input/scale offsets exceed 32 bits");

    if(job_map) {
        AITER_CHECK(tokens == 8192 && model == 8192 && experts == 513 && topk == 11,
                    "Q256 weighted scheduling requires compact Qwen 8K geometry");
        AITER_CHECK(job_map->is_contiguous() && job_map->device_id == input->device_id &&
                    job_map->dtype() == AITER_DTYPE_i32 && job_map->numel() == 1024,
                    "Q256 job map must be 1024 contiguous int32 entries on the input GPU");
    }
    q256::ProducerArgs args{};
    args.job_map = job_map ? job_map->ptr : nullptr;
    args.output = partials->ptr;
    args.activations = input->ptr;
    args.gate_up_weights = gate->ptr;
    args.persistent_counts = counts->ptr;
    args.down_weights = down->ptr;
    args.activation_scales = input_scale->ptr;
    args.gate_up_scales = gate_scale->ptr;
    args.down_scales = down_scale->ptr;
    args.sorted_token_ids = sorted_ids->ptr;
    args.routing_weights = sorted_weights->ptr;
    args.sorted_expert_ids = sorted_experts->ptr;
    args.model_width = model;
    args.intermediate_width = 256;
    args.token_count = tokens;
    args.expert_count = experts;
    args.activation_row_stride = model / 2;
    args.projection_row_stride = model / 2;
    args.down_row_stride = 128;
    args.output_row_stride = model * 2;
    args.projection_expert_stride = 256 * model;
    args.down_expert_stride = model * 128;
    args.projection_scale_expert_stride = 512 * model / 32;
    args.down_scale_expert_stride = model * 8;
    args.smoothing_scale_expert_stride = 1024;
    args.top_k = topk;
    args.total_workgroups = 256;
    args.hidden_tiles = 1;
    const auto select_kernel = [job_map]() -> AiterAsmKernel& {
        if(job_map) {
            static AiterAsmKernel weighted_kernel(
                "fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route_weighted",
                "fmoe/silu/fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route_weighted.co");
            return weighted_kernel;
        }
        static AiterAsmKernel compact_kernel(
            "fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route",
            "fmoe/silu/fmoe_bf16_pertokenMXfp4_g1u1_silu_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route.co");
        return compact_kernel;
    };
    size_t arg_size = sizeof(args);
    auto& kernel = select_kernel();
    kernel.launch_kernel({&args, &arg_size, 256, 1, 1, 256, 1, 1, stream});
}
