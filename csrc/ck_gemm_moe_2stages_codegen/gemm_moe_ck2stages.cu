// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "gemm_moe_ck2stages_lookup.h"
#include "gemm_moe_ck2stages_nkpad_lookup.h"
#include "gemm_moe_ck2stages.h"
#include "ck2stages_moe_stage1_heuristic_dispatch.hpp"
#include "ck2stages_moe_stage2_heuristic_dispatch.hpp"
#include "moe_ck.h"
#include "aiter_logger.h"
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

using MoeKernelMap = std::unordered_map<std::string, MoeKernel>;

// Map aiter ActivationType (stage1) to the CK stage1 activation op value.
// CK act values: gelu=0, silu=1, swiglu(OAI swiglu_oai)=3, gelu_tanh_and_mul=4.
// Value 2 is reserved.
static inline int map_activation_to_ck_stage1(int activation)
{
    switch (static_cast<ActivationType>(activation))
    {
    case ActivationType::Silu:
        return 1;
    case ActivationType::Gelu:
        return 0;
    case ActivationType::Swiglu:
        return 3;
    case ActivationType::GeluTanh:
        return 4;
    default:
        TORCH_CHECK(false, "Unsupported activation for ck_moe_stage1: ", activation);
        return -1;
    }
}

// API for user aiter.ck_moe_stage1(...)

template <int stage = 1>
MoeKernel moe_dispatch(std::string &kernelName, int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dtype, at::ScalarType y_dtype, int act_op, int quant_type, bool mul_routed_weight, bool is_shuffled, bool require_exact = false)
{
    static const auto lookup = []
    {
        return MoeKernelMap{GENERATE_LOOKUP_TABLE()};
    }();

    if (kernelName != "")
    {
        auto it = lookup.find(kernelName);
        if (it != lookup.end())
        {
            auto kernel = it->second;
            return kernel;
        }
        TORCH_CHECK(!require_exact, "Unknown or incompatible CK MoE kernel: ", kernelName,
                    "; regenerate instances or retune this configuration");
        AITER_LOG_WARNING("ck kernel not found: " << kernelName);
    }
    if constexpr (stage == 1)
    {
        return moe_stage1_heuristic_dispatch(block_m, inter_dim, x_dtype, w_dtype, y_dtype, act_op, quant_type, mul_routed_weight, is_shuffled);
    }
    else
    {
        return moe_stage2_heuristic_dispatch(block_m, inter_dim, x_dtype, w_dtype, y_dtype, 0, quant_type, mul_routed_weight, is_shuffled);
    }
}

struct MoeNKPadTwin
{
    int n_per_block;
    int k_per_block;
    MoeKernelSig *kernel;
};

// Replaces a Default kernel with its NKPadding twin when N or K is not a multiple of its block
// tile. `padded` tells whether the returned kernel is an NKPadding one.
static MoeKernel moe_nkpad_dispatch(const MoeKernel &kernel, int N, int K, bool &padded)
{
    static const std::unordered_map<MoeKernelSig *, MoeNKPadTwin> twins = GENERATE_NKPAD_LOOKUP_TABLE();
    static const auto padded_kernels = []
    {
        std::unordered_set<MoeKernelSig *> kernels;
        for (const auto &entry : twins)
            kernels.insert(entry.second.kernel);
        return kernels;
    }();

    padded = false;
    auto fn = kernel.target<MoeKernelSig *>();
    if (fn == nullptr)
        return kernel;
    if (padded_kernels.count(*fn))
    {
        padded = true;
        return kernel;
    }
    auto it = twins.find(*fn);
    if (it == twins.end() || (N % it->second.n_per_block == 0 && K % it->second.k_per_block == 0))
        return kernel;
    padded = true;
    return it->second.kernel;
}

static int64_t nkpad_round_up(int64_t x, int64_t tile) { return (x + tile - 1) / tile * tile; }

// NKPadding kernels address each expert's preshuffled B in whole tiles of 16 rows by 64 bytes, so
// B must be zero-padded to exactly that and no further.
static void check_nkpad_weight_cols(const torch::Tensor &w, int K, const char *name)
{
    const int64_t k_tile = 64 / w.element_size();
    TORCH_CHECK(w.size(-1) == nkpad_round_up(K, k_tile),
                name, " must have its K dimension zero-padded to ", nkpad_round_up(K, k_tile),
                " (a multiple of ", k_tile, ") for K=", K, ", got ", w.size(-1));
}

void ck_moe_stage1(torch::Tensor &hidden_states,     // [m, k], input token
                   torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   torch::Tensor &out,               // [m * topk, inter_dim]
                   int topk,
                   std::string &kernelName,
                   std::optional<torch::Tensor> w1_scale = std::nullopt, // [e, 1, n], gate(up) scale
                   std::optional<torch::Tensor> a1_scale = std::nullopt, // [m, 1], token scale
                   std::optional<int> block_m = 32,
                   std::optional<torch::Tensor> sorted_weights = std::nullopt,
                   int quant_type = 0,
                   int activation = 0,
                   std::optional<int> splitk = 1,
                   bool nt = false,
                   std::optional<std::string> dst_type = std::nullopt,
                   bool is_shuffled = true,
                   double swiglu_limit)
{
    TORCH_CHECK(!std::isnan(swiglu_limit) && swiglu_limit >= 0.0,
                "swiglu_limit must be non-negative and not NaN");
    const bool clamp_enabled = swiglu_limit > 0.0 && std::isfinite(swiglu_limit);
    TORCH_CHECK(!clamp_enabled ||
                    (activation == static_cast<int>(ActivationType::Silu) &&
                     (quant_type == static_cast<int>(QuantType::per_1x128) ||
                      quant_type == static_cast<int>(QuantType::per_Token) ||
                      quant_type == static_cast<int>(QuantType::per_Tensor)) &&
                     splitk.value_or(1) <= 1 &&
                     (kernelName.empty() ||
                      (kernelName.find("_silu_") != std::string::npos &&
                       kernelName.find("Splitk") == std::string::npos))),
                "Runtime SiLU clamp requires non-split per_1x128 / per_Token / per_Tensor "
                "CK SiLU");
    // std::cerr << __FILE__ << ":" << __LINE__ << " ck_moe_stage1 called!" << nt << " " << block_m.value() << std::endl;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(out));
    at::hip::getCurrentHIPStream();
    int32_t splitk_local = splitk.has_value() ? splitk.value() : 1;

    if (splitk_local > 1)
    {
        TORCH_CHECK(out.dtype() == at::ScalarType::Float,
                    "Out dtype only support Float when splitk_local > 1!")
    }
    else
    {
        TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
                    "Out dtype only support BFloat16/Float16!")
    }

    int tokens = hidden_states.size(0);
    int sorted_size = std::min(int64_t(tokens * topk * block_m.value()), sorted_token_ids.size(0));
    int E = w1.size(0);
    int N = w1.size(1) / 2;
    int K = hidden_states.size(-1);
    int MPerBlock = block_m.value();

    void *hidden_states_ptr = hidden_states.data_ptr();
    void *w1_ptr = w1.transpose(1, 2).data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *sorted_weights_ptr = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;
    void *out_ptr = out.data_ptr();
    void *w1_scale_ptr = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
    void *a1_scale_ptr = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;
    bool MulRoutedWeight = sorted_weights.has_value();
    if (!hidden_states_ptr || !w1_ptr || !w2_ptr || !sorted_token_ids_ptr || !sorted_expert_ids_ptr || !num_valid_ids_ptr || !out_ptr)
    {
        std::cerr << "detect null ptr !" << std::endl;
        return;
    }

    if (hidden_states.dtype() == torch_fp4x2 && w1.dtype() == torch_fp4x2)
    {
        K *= 2;
    }

    activation = map_activation_to_ck_stage1(activation);

    auto kernel = moe_dispatch<1>(kernelName, MPerBlock, N, hidden_states.dtype().toScalarType(), w1.dtype().toScalarType(), out.dtype().toScalarType(), activation, quant_type, MulRoutedWeight, is_shuffled, clamp_enabled);
    bool padded = false;
    kernel = moe_nkpad_dispatch(kernel, N, K, padded);
    if (padded)
    {
        // The gate and up halves of w1 are padded to 16 rows each, which a stacked
        // [e, 2 * n, k] tensor only matches when n itself is a multiple of 16.
        TORCH_CHECK(N % 16 == 0, "ck_moe_stage1: inter_dim ", N, " must be a multiple of 16");
        check_nkpad_weight_cols(w1, K, "w1");
    }

    // Narrow a finite positive limit to float without it becoming 0 (disabled)
    // or inf (rejected by CK).
    const float ck_limit = clamp_enabled
        ? std::max(
              static_cast<float>(std::min(
                  swiglu_limit,
                  static_cast<double>(std::numeric_limits<float>::max()))),
              std::numeric_limits<float>::denorm_min())
        : static_cast<float>(swiglu_limit);
    kernel(at::hip::getCurrentHIPStream(),
           tokens, sorted_size, N, K, topk,
           hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr, splitk_local, nt, ck_limit);
}

void ck_moe_stage2(torch::Tensor &inter_states,      // [m, k], input token
                   torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   torch::Tensor &out,               // [max_num_tokens_padded, inter_dim]
                   int topk,
                   std::string &kernelName,
                   std::optional<torch::Tensor> w2_scale = std::nullopt, // [e, 1, n], gate(up) scale
                   std::optional<torch::Tensor> a2_scale = std::nullopt, // [m, 1], token scale
                   std::optional<int> block_m = 32,
                   std::optional<torch::Tensor> sorted_weights = std::nullopt,
                   int quant_type = 0,
                   int activation = 0,
                   std::optional<int> splitk = 1,
                   bool nt = false,
                   std::optional<std::string> dst_type = std::nullopt,
                   bool is_shuffled = true)
{
    // std::cerr << __FILE__ << ":" << __LINE__ << " ck_moe_stage2 called!" << nt << " " << block_m.value() << std::endl;
    TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
                "Out dtype only support BFloat16/Float16!")

    int32_t splitk_local = splitk.has_value() ? splitk.value() : 1;

    int tokens = inter_states.size(0);
    int sorted_size = std::min(int64_t(tokens * topk * block_m.value()), sorted_token_ids.size(0));
    int E = w1.size(0);
    int N = out.size(-1);
    int K = inter_states.size(-1);
    int MPerBlock = block_m.value();

    void *inter_states_ptr = inter_states.data_ptr();
    void *w1_ptr = w1.data_ptr();
    void *w2_ptr = w2.data_ptr();
    void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
    void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
    void *sorted_weights_ptr = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;
    void *num_valid_ids_ptr = num_valid_ids.data_ptr();
    void *out_ptr = out.data_ptr();
    void *w2_scale_ptr = w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr;
    void *a2_scale_ptr = a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr;
    bool MulRoutedWeight = sorted_weights.has_value();

    if (!inter_states_ptr || !w1_ptr || !w2_ptr || !sorted_token_ids_ptr || !sorted_expert_ids_ptr || !num_valid_ids_ptr || !out_ptr)
    {
        std::cerr << "detect null ptr !" << std::endl;
        return;
    }
    if (inter_states.dtype() == torch_fp4x2 && w2.dtype() == torch_fp4x2)
    {
        K *= 2;
    }

    // stage2 does not apply an activation.
    auto kernel = moe_dispatch<2>(kernelName, MPerBlock, K, inter_states.dtype().toScalarType(), w1.dtype().toScalarType(), out.dtype().toScalarType(), activation, quant_type, MulRoutedWeight, is_shuffled);
    bool padded = false;
    kernel = moe_nkpad_dispatch(kernel, N, K, padded);
    const int64_t w2_rows = padded ? nkpad_round_up(N, 16) : N;
    TORCH_CHECK(w2.size(1) == w2_rows, "ck_moe_stage2: w2 has ", w2.size(1), " rows for ", N,
                " output columns, expected ", w2_rows);
    if (padded)
    {
        check_nkpad_weight_cols(w2, K, "w2");
    }

    kernel(at::hip::getCurrentHIPStream(),
           tokens, sorted_size, N, K, topk,
           inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr, splitk_local, nt, 0.f);
}
