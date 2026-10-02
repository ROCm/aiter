// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Router GEMM, softmax top-k and the shared expert sigmoid in one launch, for
// decode batches of 1 to 8 tokens on gfx950.
//
// The kernel computes logits = hidden_states @ gate_weight.T in fp32 and rounds
// them to BF16. Rows 0 to num_experts - 1 of gate_weight are the routed experts
// and the last row is the shared expert gate. Every workgroup computes a few
// rows for all tokens and writes them to the workspace. The last workgroup to
// finish, found with an atomic counter, then selects the top-k routed experts
// of each token, writes the softmax weights renormalized over the top-k, and
// writes sigmoid(shared expert logit) after them. The outputs match
// topk_softmax(..., need_renorm=true, num_shared_experts=1, "sigmoid") on the
// BF16 logits, including the lower expert id winning a tie.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_router_topk_configs.hpp"
#include <hip/hip_runtime.h>

// The kernel reads seven 64-bit pointers packed back to back. Unlike most asm
// kernels in AITER, there is no padding between the arguments.
struct __attribute__((packed)) RouterTopkKernelArgs
{
    const void* hidden_states;
    const void* gate_weight;
    void* logits;
    void* counter;
    void* topk_weights;
    void* topk_ids;
    void* token_expert_indices;
};
static_assert(sizeof(RouterTopkKernelArgs) == 56, "the kernel reads a 56 byte argument buffer");

// Every kernel of this family keeps the logits of up to 8 tokens in the
// workspace, 520 floats per token, and puts the 32-bit workgroup counter right
// after them. The counter must be zero before a launch. The last workgroup of
// every launch sets it back to zero, so the same workspace can be reused by the
// next launch on the same stream.
static constexpr int kMaxTokens         = 8;
static constexpr size_t kLogitStride    = 520;
static constexpr size_t kCounterOffset  = kMaxTokens * kLogitStride * sizeof(float);
static constexpr size_t kWorkspaceBytes = kCounterOffset + sizeof(int);

static const router_topkConfig& get_router_topk_config(
    const std::string& arch_id, int m, int dim, int num_experts, int num_shared_experts, int topk)
{
    for(const auto& el : cfg_router_topk)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.m == m && cfg.dim == dim && cfg.num_experts == num_experts &&
           cfg.num_shared_experts == num_shared_experts && cfg.topk == topk)
            return cfg;
    }
    AITER_CHECK(false,
                "router_gemm_topk_softmax_asm: no kernel for arch ",
                arch_id,
                " with tokens=",
                m,
                " dim=",
                dim,
                " num_experts=",
                num_experts,
                " num_shared_experts=",
                num_shared_experts,
                " topk=",
                topk,
                ". The kernels cover gfx950, 1 to 8 tokens, dim 8192, 512 routed experts, 1 "
                "shared expert and top 10.");
    __builtin_unreachable();
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    router_gemm_topk_softmax_asm,
    (aiter_tensor_t* hidden_states,        // [M, dim] bf16
     aiter_tensor_t* gate_weight,          // [num_experts + 1, dim] bf16
     aiter_tensor_t* topk_weights,         // [M, topk + 1] fp32
     aiter_tensor_t* topk_ids,             // [M, topk] or [M, topk + 1] int32, row stride topk + 1
     aiter_tensor_t* token_expert_indices, // [M, topk] int32
     aiter_tensor_t* workspace,            // zero-initialized, at least kWorkspaceBytes
     hipStream_t stream),
    (hidden_states, gate_weight, topk_weights, topk_ids, token_expert_indices, workspace, stream))
{
    const char* fn = "router_gemm_topk_softmax_asm";
    AITER_CHECK(hidden_states->dtype() == AITER_DTYPE_bf16,
                fn,
                ": hidden_states must be bf16, got ",
                AiterDtype_to_str(hidden_states->dtype()));
    AITER_CHECK(gate_weight->dtype() == AITER_DTYPE_bf16,
                fn,
                ": gate_weight must be bf16, got ",
                AiterDtype_to_str(gate_weight->dtype()));
    AITER_CHECK(topk_weights->dtype() == AITER_DTYPE_fp32,
                fn,
                ": topk_weights must be float32, got ",
                AiterDtype_to_str(topk_weights->dtype()));
    AITER_CHECK(topk_ids->dtype() == AITER_DTYPE_i32,
                fn,
                ": topk_ids must be int32, got ",
                AiterDtype_to_str(topk_ids->dtype()));
    AITER_CHECK(token_expert_indices->dtype() == AITER_DTYPE_i32,
                fn,
                ": token_expert_indices must be int32, got ",
                AiterDtype_to_str(token_expert_indices->dtype()));
    AITER_CHECK(hidden_states->dim() == 2 && gate_weight->dim() == 2 && topk_weights->dim() == 2 &&
                    topk_ids->dim() == 2 && token_expert_indices->dim() == 2,
                fn,
                ": all tensors must be 2-D");

    const int m                  = hidden_states->size(0);
    const int dim                = hidden_states->size(1);
    const int topk               = token_expert_indices->size(1);
    const int num_shared_experts = topk_weights->size(1) - topk;
    const int num_experts        = gate_weight->size(0) - num_shared_experts;
    if(m == 0)
        return;

    // The kernel addresses every tensor with fixed row strides, so the checks
    // below reject views that it would read or write in the wrong place.
    AITER_CHECK(hidden_states->stride(1) == 1 && hidden_states->stride(0) == dim,
                fn,
                ": hidden_states must be contiguous");
    AITER_CHECK(gate_weight->size(1) == dim && gate_weight->stride(1) == 1 &&
                    gate_weight->stride(0) == dim,
                fn,
                ": gate_weight must be a contiguous [num_experts + 1, ",
                dim,
                "] tensor");
    AITER_CHECK(topk_weights->size(0) == m && topk_weights->stride(1) == 1 &&
                    topk_weights->stride(0) == topk_weights->size(1),
                fn,
                ": topk_weights must be a contiguous [",
                m,
                ", topk + 1] tensor");
    AITER_CHECK(topk_ids->size(0) == m && topk_ids->stride(1) == 1 &&
                    (topk_ids->size(1) == topk || topk_ids->size(1) == topk + num_shared_experts) &&
                    topk_ids->stride(0) == topk + num_shared_experts,
                fn,
                ": topk_ids must be [",
                m,
                ", topk] or [",
                m,
                ", topk + 1] with row stride topk + 1");
    AITER_CHECK(token_expert_indices->size(0) == m && token_expert_indices->stride(1) == 1 &&
                    token_expert_indices->stride(0) == topk,
                fn,
                ": token_expert_indices must be a contiguous [",
                m,
                ", topk] tensor");
    AITER_CHECK(workspace->numel() * workspace->element_size() >= kWorkspaceBytes,
                fn,
                ": workspace must hold at least ",
                kWorkspaceBytes,
                " bytes");
    const int dev = hidden_states->device_id;
    AITER_CHECK(gate_weight->device_id == dev && topk_weights->device_id == dev &&
                    topk_ids->device_id == dev && token_expert_indices->device_id == dev &&
                    workspace->device_id == dev,
                fn,
                ": all tensors must be on the same GPU");

    const HipDeviceGuard device_guard(dev);
    const std::string arch_id = get_gpu_arch();
    const auto& cfg =
        get_router_topk_config(arch_id, m, dim, num_experts, num_shared_experts, topk);

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });

    RouterTopkKernelArgs args;
    args.hidden_states        = hidden_states->ptr;
    args.gate_weight          = gate_weight->ptr;
    args.logits               = workspace->ptr;
    args.counter              = static_cast<char*>(workspace->ptr) + kCounterOffset;
    args.topk_weights         = topk_weights->ptr;
    args.topk_ids             = topk_ids->ptr;
    args.token_expert_indices = token_expert_indices->ptr;
    size_t arg_size           = sizeof(args);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             cfg.grid,    // gdx
                             1,           // gdy
                             1,           // gdz
                             cfg.threads, // bdx
                             1,           // bdy
                             1,           // bdz
                             stream});
}
