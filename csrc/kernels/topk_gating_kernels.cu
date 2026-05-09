// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused topk gating kernel for MoE routing.
//
// Scoring functions (selected by string at the C++ entry):
//   "sqrtsoftplus"  → sqrt(softplus(x))   — DeepSeek V4-Pro default
//   "sigmoid"       → sigmoid(x)          — Llama4
//   "softmax"       → softmax(x)          — DeepSeek V3 / classic MoE
//
// Kernel variants:
//   topk_softplus_kernel_opt  — register-only, sort+merge (64/128/256/384 experts)
//   topk_softplus_kernel      — shared-memory fallback (any expert count)

#include "aiter_hip_common.h"
#include "hip_reduce.h"
#include "py_itfs_common.h"
#include "aiter_opus_plus.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <torch/all.h>
#include <type_traits>

namespace aiter {

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

enum { SCORE_SQRTSOFTPLUS = 0, SCORE_SIGMOID = 1, SCORE_SOFTMAX = 2 };

// Fused DPP warp argmax: 6× v_max_f32+DPP + ballot + ctzll + readlane ≈ 9 instr.
// NaN-safe: if all lanes have NaN (val_o == max_val is always false), ballot is 0
// and ctzll(0) is UB.  Detect this via the ballot result and fall back to lane 0.
__device__ __forceinline__ void warpReduceMax_softplus(float& val_o, int& idx)
{
    float max_val   = multithread_reduce_max_dpp<WARP_SIZE>(val_o);
    uint64_t mask   = __ballot(val_o == max_val);
    int win_lane    = (mask != 0) ? __builtin_ctzll(mask) : 0;
    idx             = __builtin_amdgcn_readlane(idx, win_lane);
    val_o           = max_val;
}

template <int SCORE_FUNC>
__device__ __forceinline__ float compute_score(float x)
{
    if constexpr(SCORE_FUNC == SCORE_SIGMOID)
    {
        // sigmoid(x) = rcp(1 + 2^(-x·log₂e))  →  v_exp_f32 + v_rcp_f32
        return __builtin_amdgcn_rcpf(1.0f + exp2f(-x * 1.4426950408889634f));
    }
    else if constexpr(SCORE_FUNC == SCORE_SOFTMAX)
    {
        // softmax: per-element score is identity; normalization done separately
        return x;
    }
    else
    {
        // sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
        // Highest-precision path: pure libm (expf + log1pf), ≤1 ULP.
        // Faster alternatives (commented out, ~0.5-1 ULP extra error):
        //   float sp = x > 20.0f ? x : log1pf(exp2f(x * 1.4426950408889634f));   // exp2f HW
        float sp = x > 20.0f ? x : log2f(1.0f + exp2f(x * 1.4426950408889634f)) * 0.6931471805599453f;  // both HW
        return sqrtf(sp);
    }
}

// ---------------------------------------------------------------------------
// Sorting network (descending, 3 arrays co-permuted: vals, orig, idxs)
// ---------------------------------------------------------------------------

#define _CAS_DESC(v, o, id, i, j)                                    \
    do                                                               \
    {                                                                \
        if((v)[i] < (v)[j])                                          \
        {                                                            \
            float _tv = (v)[i]; (v)[i] = (v)[j]; (v)[j] = _tv;      \
            float _to = (o)[i]; (o)[i] = (o)[j]; (o)[j] = _to;      \
            int _ti   = (id)[i]; (id)[i] = (id)[j]; (id)[j] = _ti;  \
        }                                                            \
    } while(0)

template <int N>
__device__ __forceinline__ void sort_network_desc(float* vals, float* orig, int* idxs)
{
    if constexpr(N <= 1)
        return;
    else if constexpr(N == 2)
    {
        _CAS_DESC(vals, orig, idxs, 0, 1);
    }
    else if constexpr(N == 3)
    {
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 0, 2);
        _CAS_DESC(vals, orig, idxs, 1, 2);
    }
    else if constexpr(N == 4)
    {   // 5-comparator optimal network
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 2, 3);
        _CAS_DESC(vals, orig, idxs, 0, 2);
        _CAS_DESC(vals, orig, idxs, 1, 3);
        _CAS_DESC(vals, orig, idxs, 1, 2);
    }
    else if constexpr(N == 6)
    {   // 12-comparator optimal network
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 2, 3);
        _CAS_DESC(vals, orig, idxs, 4, 5);
        _CAS_DESC(vals, orig, idxs, 0, 2);
        _CAS_DESC(vals, orig, idxs, 1, 4);
        _CAS_DESC(vals, orig, idxs, 3, 5);
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 2, 3);
        _CAS_DESC(vals, orig, idxs, 4, 5);
        _CAS_DESC(vals, orig, idxs, 1, 2);
        _CAS_DESC(vals, orig, idxs, 3, 4);
        _CAS_DESC(vals, orig, idxs, 2, 3);
    }
    else
    {   // generic unrolled bubble sort fallback
#pragma unroll
        for(int i = 0; i < N - 1; i++)
        {
#pragma unroll
            for(int j = 0; j < N - 1 - i; j++)
            {
                _CAS_DESC(vals, orig, idxs, j, j + 1);
            }
        }
    }
}

#undef _CAS_DESC

// ---------------------------------------------------------------------------
// Register-only kernel (for expert counts divisible by WARP_SIZE)
//
// Each thread loads EPT = NUM_EXPERTS/WARP_SIZE elements, sorts them locally
// via an optimal sorting network, then participates in a warp-level k-way
// merge (iterative argmax) to extract the global top-K.
// No shared memory, no __syncthreads.
//
// 1 warp = 1 token = 1 block.  Multi-warp-per-block was tried (WPB=2,4) and
// regressed K≥4 cases (extra register pressure / wave-scheduling overhead),
// while only marginally helping K=1~2.  K-merge serial chain is the actual
// bottleneck, not block-launch overhead.
// ---------------------------------------------------------------------------

template <typename DTYPE_I, typename DTYPE_B, int NUM_EXPERTS,
          bool need_renorm, int SCORE_FUNC = SCORE_SQRTSOFTPLUS>
__global__ void topk_softplus_kernel_opt(
    const DTYPE_I* __restrict__ gating_output,
    const DTYPE_B* __restrict__ correction_bias,
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids,
    const size_t stride_tk,
    const int topk,
    const int num_tokens,
    const float routed_scaling_factor)
{
    static constexpr int EPT = NUM_EXPERTS / WARP_SIZE;
    static_assert(NUM_EXPERTS % WARP_SIZE == 0);

    const int token_idx = blockIdx.x;
    auto const* input_ptr = gating_output + token_idx * NUM_EXPERTS;

    float vals[EPT];
    float orig[EPT];
    int   idxs[EPT];

    // Step 1: load → score → bias  (all in registers, strided access)
    // orig[] caches unbiased scores; sorted alongside vals[]/idxs[] so all
    // three arrays share one cursor index for the merge phase.
#pragma unroll
    for(int i = 0; i < EPT; i++)
    {
        int   e     = threadIdx.x + i * static_cast<int>(WARP_SIZE);
        float score = compute_score<SCORE_FUNC>(static_cast<float>(input_ptr[e]));
        orig[i]     = score;
        vals[i]     = score;
        idxs[i]     = e;
        if(correction_bias != nullptr)
            vals[i] += static_cast<float>(correction_bias[e]);
    }

    // Step 2: sort thread-local partition descending
    sort_network_desc<EPT>(vals, orig, idxs);

    // Step 3: warp-level k-way merge
    // Winning lane = expert_idx & (WARP_SIZE-1) → readlane broadcasts
    // the pre-cached unbiased score (no per-round global memory access).
    int   cursor      = 0;
    float sum         = 0.0f;
    int   topk_indice = 0;
    float topk_value  = 0.0f;

    for(int k = 0; k < topk; ++k)
    {
        float my_val = (cursor < EPT) ? vals[cursor] : -INFINITY;
        int   my_idx = (cursor < EPT) ? idxs[cursor] : 0;

        warpReduceMax_softplus(my_val, my_idx);

        bool  i_won   = (cursor < EPT && idxs[cursor] == my_idx);
        float my_orig = i_won ? orig[cursor] : 0.0f;
        if(i_won) cursor++;

        int   win_lane = my_idx & (static_cast<int>(WARP_SIZE) - 1);
        float weight   = __builtin_bit_cast(
            float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, my_orig), win_lane));

        if(static_cast<int>(threadIdx.x) == k)
        {
            topk_indice = my_idx;
            topk_value  = weight;
        }
        if constexpr(need_renorm) sum += weight;
    }

    // Step 4: renorm + scale + write
    if constexpr(need_renorm)
        sum = routed_scaling_factor / fmaxf(sum, 1e-20f);
    else
        sum = routed_scaling_factor;

    if(static_cast<int>(threadIdx.x) < topk)
    {
        topk_weights[token_idx * stride_tk + threadIdx.x] = topk_value * sum;
        topk_ids[token_idx * stride_tk + threadIdx.x]     = topk_indice;
    }
}

// ---------------------------------------------------------------------------
// Generic fallback kernel (shared-memory based, any expert count)
// ---------------------------------------------------------------------------

template <typename DTYPE_I, typename DTYPE_B, typename f32vec, bool need_renorm,
          int SCORE_FUNC = SCORE_SQRTSOFTPLUS>
__global__ void topk_softplus_kernel(
    const DTYPE_I* __restrict__ gating_output,
    const DTYPE_B* __restrict__ correction_bias,
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids,
    const size_t stride_tk,
    const int num_experts,
    const int topk,
    const int num_tokens,
    const float routed_scaling_factor)
{
    extern __shared__ char shared_mem[];
    const int token_idx = blockIdx.x;
    float* scores = reinterpret_cast<float*>(shared_mem);

    using cktype_i                = typename hip2opus<DTYPE_I>::type;
    f32vec* scores_vec            = reinterpret_cast<f32vec*>(scores);
    static constexpr int vec_size = opus::vector_traits<f32vec>::size();
    using vec_i                   = opus::vector_t<cktype_i, vec_size>;
    const int num_experts_vec     = num_experts / vec_size;

    // Step 1: load + score function
    // For softmax, bias is NOT added here — it's added AFTER normalization
    // (bias only shifts scores for topk selection, not for softmax computation).
    auto const* input_ptr = gating_output + token_idx * num_experts;
    for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
    {
        vec_i tmp = reinterpret_cast<vec_i const*>(input_ptr)[e];
        f32vec gating;
#pragma unroll
        for(size_t i = 0; i < vec_size; i++)
        {
            gating[i] = compute_score<SCORE_FUNC>(static_cast<float>(tmp[i]));
            if constexpr(SCORE_FUNC != SCORE_SOFTMAX)
            {
                if(correction_bias != nullptr)
                    gating[i] += static_cast<float>(correction_bias[e * vec_size + i]);
            }
        }
        scores_vec[e] = gating;
    }
    for(int e = num_experts_vec * vec_size + threadIdx.x; e < num_experts; e += blockDim.x)
    {
        scores[e] = compute_score<SCORE_FUNC>(static_cast<float>(input_ptr[e]));
        if constexpr(SCORE_FUNC != SCORE_SOFTMAX)
        {
            if(correction_bias != nullptr)
                scores[e] += static_cast<float>(correction_bias[e]);
        }
    }
    __syncthreads();

    // Softmax: normalize first, then add bias for topk selection.
    // scores[] after this block = softmax(x) + bias (biased for selection).
    // The topk loop subtracts bias back to get unbiased softmax weights.
    if constexpr(SCORE_FUNC == SCORE_SOFTMAX)
    {
        float local_max = -INFINITY;
        for(int e = threadIdx.x; e < num_experts; e += blockDim.x)
            local_max = fmaxf(local_max, scores[e]);
        local_max = multithread_reduce_max_dpp<WARP_SIZE>(local_max);

        float local_sum = 0.0f;
        for(int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            scores[e] = exp2f((scores[e] - local_max) * 1.4426950408889634f);
            local_sum += scores[e];
        }
        local_sum = wave_reduce(local_sum, [](float a, float b) { return a + b; });

        float inv_sum = __builtin_amdgcn_rcpf(local_sum);
        for(int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            scores[e] *= inv_sum;
            if(correction_bias != nullptr)
                scores[e] += static_cast<float>(correction_bias[e]);
        }
        __syncthreads();
    }

    float sum         = 0.0f;
    int   topk_indice = 0;
    float topk_value  = 0.0f;
    for(int k = 0; k < topk; ++k)
    {
        float max_val = -INFINITY;
        int max_idx   = k;
        for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        {
            f32vec tmp = scores_vec[e];
#pragma unroll
            for(size_t i = 0; i < vec_size; i++)
            {
                if(tmp[i] > max_val) { max_val = tmp[i]; max_idx = e * vec_size + i; }
            }
        }
        warpReduceMax_softplus(max_val, max_idx);
        if(correction_bias != nullptr)
            max_val -= static_cast<float>(correction_bias[max_idx]);
        scores[max_idx] = -INFINITY;
        if(static_cast<int>(threadIdx.x) == k)
        {
            topk_indice = max_idx;
            topk_value  = max_val;
        }
        if(need_renorm) sum += max_val;
    }

    if(need_renorm)
        sum = routed_scaling_factor / fmaxf(sum, 1e-20f);
    else
        sum = routed_scaling_factor;

    for(int k = threadIdx.x; k < topk; k += blockDim.x)
    {
        topk_weights[token_idx * stride_tk + k] = topk_value * sum;
        topk_ids[token_idx * stride_tk + k]     = topk_indice;
    }
}

// ---------------------------------------------------------------------------
// Launch macros
// ---------------------------------------------------------------------------

#define LAUNCH_TOPK_KERNEL(VEC_F, RENORM, SF)                                                    \
    hipLaunchKernelGGL(                                                                          \
        (aiter::topk_softplus_kernel<scalar_t, bias_scalar_t, VEC_F, RENORM, SF>),               \
        dim3(grid), dim3(block), shared_mem_size, stream,                                        \
        reinterpret_cast<const scalar_t*>(gating_output.data_ptr()),                              \
        has_bias ? reinterpret_cast<const bias_scalar_t*>(correction_bias.data_ptr()) : nullptr,  \
        topk_weights.data_ptr<float>(), topk_indices.data_ptr<int>(),                            \
        stride_tk, num_experts, topk, num_tokens, routed_scaling_factor);

#define LAUNCH_TOPK_KERNEL_OPT(NE, RENORM, SF)                                                  \
    hipLaunchKernelGGL(                                                                          \
        (aiter::topk_softplus_kernel_opt<scalar_t, bias_scalar_t, NE, RENORM, SF>),              \
        dim3(grid), dim3(block), 0, stream,                                                      \
        reinterpret_cast<const scalar_t*>(gating_output.data_ptr()),                              \
        has_bias ? reinterpret_cast<const bias_scalar_t*>(correction_bias.data_ptr()) : nullptr,  \
        topk_weights.data_ptr<float>(), topk_indices.data_ptr<int>(),                            \
        stride_tk, topk, num_tokens, routed_scaling_factor);

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

// Resolve "sqrtsoftplus"/"sigmoid"/"softmax" → SCORE_* enum, or AITER_CHECK fail.
static inline int parse_score_func(const std::string& s)
{
    if(s == "sqrtsoftplus") return SCORE_SQRTSOFTPLUS;
    if(s == "sigmoid")      return SCORE_SIGMOID;
    if(s == "softmax")      return SCORE_SOFTMAX;
    AITER_CHECK(false, "unknown score_func: ", s,
                " (expected sqrtsoftplus|sigmoid|softmax)");
    return SCORE_SQRTSOFTPLUS;  // unreachable
}

void topk_softplus(torch::Tensor& topk_weights,
                   torch::Tensor& topk_indices,
                   torch::Tensor& gating_output,
                   torch::Tensor& correction_bias,
                   bool need_renorm,
                   float routed_scaling_factor,
                   const std::string& score_func)
{
    const int sf_code      = parse_score_func(score_func);
    const int num_tokens   = gating_output.size(0);
    const int num_experts  = gating_output.size(1);
    const int topk         = topk_indices.size(1);
    const size_t stride_tk = topk_indices.stride(0);
    const bool has_bias    = correction_bias.numel() > 0;

    // Both kernels assign one lane per top-K winner during writeout
    // (`if (lane == k) topk_value = ...`), so topk must fit in a single warp
    // and cannot exceed the number of routable experts.  Fail fast with a
    // clear error rather than silently producing partial / wrong output.
    AITER_CHECK(topk <= static_cast<int>(WARP_SIZE),
                "topk (", topk, ") exceeds WARP_SIZE (", WARP_SIZE, ")");
    AITER_CHECK(topk <= num_experts,
                "topk (", topk, ") exceeds num_experts (", num_experts, ")");

    // Softmax outputs are already a probability distribution that sums to 1
    // across the routed top-K (post-selection); a second renorm would distort
    // those weights.  Enforce here so direct C++ callers behave the same as
    // the Python topk_gating() wrapper (which already forces this).
    if(sf_code == SCORE_SOFTMAX)
    {
        need_renorm = false;
    }

    dim3 grid(num_tokens);
    dim3 block(get_warp_size_func());

    // Use PyTorch's current stream so that the kernel runs on the same stream
    // as the surrounding torch ops (avoids race conditions and works with CUDA
    // graph capture).
    // TODO: when this op is migrated to aiter_tensor_t (and @compile_ops uses
    //       develop=True), switch to aiter::getCurrentHIPStream() — the wrapper
    //       will then sync torch.cuda.current_stream() before each call.
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const auto gating_st = gating_output.scalar_type();
    const auto bias_st   = has_bias ? correction_bias.scalar_type() : gating_st;

    // Three-level compile-time dispatch: gating dtype → bias dtype → score_func.
    auto dispatch = [&](auto gating_tag, auto bias_tag, auto sf_tag) {
        using scalar_t      = decltype(gating_tag);
        using bias_scalar_t = decltype(bias_tag);
        constexpr int SF    = decltype(sf_tag)::value;

        // Register-only opt kernel (NOT supported for softmax: needs global reduce).
        if constexpr(SF != SCORE_SOFTMAX)
        {
#define _DISPATCH_REG_KERNEL(NE)                                          \
    if(num_experts == NE) {                                                \
        if(need_renorm) { LAUNCH_TOPK_KERNEL_OPT(NE, true,  SF) }         \
        else            { LAUNCH_TOPK_KERNEL_OPT(NE, false, SF) }         \
        return;                                                            \
    }
            _DISPATCH_REG_KERNEL(64)
            _DISPATCH_REG_KERNEL(128)
            _DISPATCH_REG_KERNEL(256)
            _DISPATCH_REG_KERNEL(384)
#undef _DISPATCH_REG_KERNEL
        }

        // Shared-memory fallback kernel
        const size_t shared_mem_size = num_experts * sizeof(float);
#define _DISPATCH_SMEM_KERNEL(VEC_LANES)                                  \
    {                                                                      \
        using VT = opus::vector_t<float, VEC_LANES>;                       \
        if(need_renorm) { LAUNCH_TOPK_KERNEL(VT, true,  SF) }             \
        else            { LAUNCH_TOPK_KERNEL(VT, false, SF) }             \
    }
        switch(num_experts % 4)
        {
        case 0:  _DISPATCH_SMEM_KERNEL(4) break;
        case 2:  _DISPATCH_SMEM_KERNEL(2) break;
        default: _DISPATCH_SMEM_KERNEL(1) break;
        }
#undef _DISPATCH_SMEM_KERNEL
    };

    auto dispatch_sf = [&](auto gating_tag, auto bias_tag) {
        switch(sf_code)
        {
        case SCORE_SIGMOID:
            dispatch(gating_tag, bias_tag, std::integral_constant<int, SCORE_SIGMOID>{}); break;
        case SCORE_SOFTMAX:
            dispatch(gating_tag, bias_tag, std::integral_constant<int, SCORE_SOFTMAX>{}); break;
        default:
            dispatch(gating_tag, bias_tag, std::integral_constant<int, SCORE_SQRTSOFTPLUS>{}); break;
        }
    };

    auto dispatch_bias = [&](auto gating_tag) {
        switch(bias_st)
        {
        case at::kFloat:    dispatch_sf(gating_tag, float{});        break;
        case at::kHalf:     dispatch_sf(gating_tag, __half{});       break;
        case at::kBFloat16: dispatch_sf(gating_tag, hip_bfloat16{}); break;
        default: AITER_CHECK(false, "unsupported correction_bias dtype"); break;
        }
    };

    switch(gating_st)
    {
    case at::kFloat:    dispatch_bias(float{});        break;
    case at::kHalf:     dispatch_bias(__half{});       break;
    case at::kBFloat16: dispatch_bias(hip_bfloat16{}); break;
    default: AITER_CHECK(false, "unsupported gating_output dtype"); break;
    }
}

} // namespace aiter
