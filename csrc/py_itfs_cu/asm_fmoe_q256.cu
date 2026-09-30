// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "q256_producer_args.hpp"

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_q256_producer,
    (aiter_tensor_t* out, aiter_tensor_t* partials, aiter_tensor_t* input,
     aiter_tensor_t* gate, aiter_tensor_t* down, aiter_tensor_t* sorted_ids,
     aiter_tensor_t* sorted_weights, aiter_tensor_t* sorted_experts,
     aiter_tensor_t* counts, aiter_tensor_t* reverse_sorted,
     aiter_tensor_t* input_scale, aiter_tensor_t* gate_scale,
     aiter_tensor_t* down_scale, int topk, hipStream_t stream),
    (out, partials, input, gate, down, sorted_ids, sorted_weights,
     sorted_experts, counts, reverse_sorted, input_scale, gate_scale,
     down_scale, topk, stream))
{
    const HipDeviceGuard guard(input->device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", "Q256 requires gfx950");
    for(auto* tensor : {out, partials, input, gate, down, sorted_ids,
                        sorted_weights, sorted_experts, counts, reverse_sorted,
                        input_scale, gate_scale, down_scale})
        AITER_CHECK(tensor && tensor->is_contiguous() &&
                    tensor->device_id == input->device_id,
                    "Q256 requires contiguous tensors on the same GPU");
    const int64_t tokens = out->size(0), model = out->size(1);
    const int64_t experts = gate->size(0), capacity = sorted_ids->numel();
    AITER_CHECK(tokens > 0 && tokens < (1 << 24) && model >= 512 && model % 256 == 0 &&
                topk > 0 && topk <= 127 && topk <= experts && capacity > 0 && capacity % 256 == 0,
                "Unsupported Q256 geometry");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16 && partials->dtype() == AITER_DTYPE_bf16 &&
                input->element_size() == 1 && gate->element_size() == 1 && down->element_size() == 1 &&
                input_scale->element_size() == 1 && gate_scale->element_size() == 1 &&
                down_scale->element_size() == 1 && sorted_ids->dtype() == AITER_DTYPE_i32 &&
                sorted_experts->dtype() == AITER_DTYPE_i32 && counts->dtype() == AITER_DTYPE_i32 &&
                reverse_sorted->dtype() == AITER_DTYPE_i32 && sorted_weights->dtype() == AITER_DTYPE_fp32,
                "Unsupported Q256 tensor dtype");
    AITER_CHECK(input->numel() == tokens * model / 2 &&
                gate->size(1) == 512 && gate->size(2) == model / 2 &&
                down->size(0) == experts && down->size(1) == model && down->size(2) == 128 &&
                partials->numel() == capacity * model && sorted_weights->numel() == capacity &&
                sorted_experts->numel() >= capacity / 256 && counts->numel() == 2 &&
                reverse_sorted->numel() == tokens * topk &&
                input_scale->numel() >= capacity * model / 32 &&
                gate_scale->numel() == experts * 512 * model / 32 &&
                down_scale->numel() == experts * model * 8,
                "Q256 tensor extent mismatch");
    for(uint64_t bytes : {uint64_t(tokens * model / 2), uint64_t(experts * 256 * model),
                          uint64_t(capacity * model / 32), uint64_t(capacity * 4),
                          uint64_t(model * 512)})
        AITER_CHECK(bytes < (uint64_t(1) << 32), "Q256 input/scale offsets exceed 32 bits");

    q256::ProducerArgs args{};
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
    static AiterAsmKernel producer(
        "fused_moe_mxfp4_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate",
        "fmoe/silu/fused_moe_mxfp4_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate.co");
    size_t arg_size = sizeof(args);
    producer.launch_kernel({&args, &arg_size, 256, 1, 1, 256, 1, 1, stream});
}
