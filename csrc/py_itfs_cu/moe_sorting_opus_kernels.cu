// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting torch-free binding.
// Self-contained: no CK header dependency.

#define MOE_SORTING_OPUS_IMPL
#include "moe_sorting_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "mx_quant_utils.h"

#include <string>
#include <vector>

void moe_sorting_opus_fwd(aiter_tensor_t& topk_ids,
                          aiter_tensor_t& topk_weights,
                          aiter_tensor_t& sorted_token_ids,
                          aiter_tensor_t& sorted_weights,
                          aiter_tensor_t& sorted_expert_ids,
                          aiter_tensor_t& num_valid_ids,
                          aiter_tensor_t& moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<aiter_tensor_t> local_expert_mask,
                          std::optional<aiter_tensor_t> num_local_tokens,
                          std::optional<aiter_tensor_t> workspace,
                          int dispatch_policy,
                          std::optional<aiter_tensor_t> local_topk_ids,
                          std::optional<aiter_tensor_t> m_indices,
                          std::optional<aiter_tensor_t> reverse_sorted)
{
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32,
                "topk_weights must be FP32 (float32)");

    auto dtype_str = AiterDtype_to_str(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);

    if(local_topk_ids.has_value())
    {
        auto& ids_out = local_topk_ids.value();
        AITER_CHECK(ids_out.dim() == 2 && ids_out.size(0) == topk_ids.size(0) &&
                        ids_out.size(1) == topk_ids.size(1),
                    "local_topk_ids must have the same [tokens, topk] shape as topk_ids");
        AITER_CHECK(ids_out.dtype() == topk_ids.dtype(),
                    "local_topk_ids dtype must match topk_ids");
        AITER_CHECK(ids_out.device_id == topk_ids.device_id,
                    "local_topk_ids must be on the same device as topk_ids");
        AITER_CHECK(ids_out.is_contiguous(), "local_topk_ids must be contiguous");
    }

    HipDeviceGuard device_guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    void* ws_ptr = workspace.has_value() ? workspace.value().data_ptr() : nullptr;

    moe_sorting_opus(
        {
            dtype_str,
            "fp32",
            local_expert_mask.has_value(),
            true,
            dispatch_policy
        },
        {topk_ids.data_ptr(),
         topk_weights.data_ptr(),
         local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
         num_local_tokens.has_value() ? num_local_tokens.value().data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),
         sorted_weights.data_ptr(),
         sorted_expert_ids.data_ptr(),
         num_valid_ids.data_ptr(),
         moe_buf.data_ptr(),
         ws_ptr,
         local_topk_ids.has_value() ? local_topk_ids.value().data_ptr() : nullptr,
         m_indices.has_value() ? m_indices.value().data_ptr() : nullptr,
         reverse_sorted.has_value() ? reverse_sorted.value().data_ptr() : nullptr,
         num_tokens,
         unit_size,
         num_experts,
         topk,
         static_cast<int>(moe_buf.size(-1)),
         static_cast<int>(moe_buf.element_size())},
        {stream});
}

namespace aiter::mxfp4_moe {

constexpr int kHiddenSize    = 7168;
constexpr int kBlockM        = 32;
constexpr int kThreads       = 1024;
constexpr int kSortQuantCtas = 16;
constexpr int kScaleCols     = kHiddenSize / 32;

// Routed-MoE shapes the sort kernel is instantiated for. NumExperts and TopK
// are template parameters (the expert histogram lives in LDS), so widening the
// support surface means adding a row here.
#define AITER_MXFP4_SORT_QUANT_ROUTES(_) \
    _(256, 8)                            \
    _(384, 8)                            \
    _(385, 9)

__device__ __forceinline__ int pack_token_topk_id(int token_id, int topk_id)
{
    return (token_id & 0x00ffffff) | ((topk_id & 0xff) << 24);
}

__device__ __forceinline__ int round_up_to_block(int value)
{
    return (value + kBlockM - 1) & ~(kBlockM - 1);
}

__device__ __forceinline__ int dpp_inclusive_scan_wave(int value)
{
    int tmp = __builtin_amdgcn_mov_dpp(value, 0x111, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x112, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x114, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x118, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_update_dpp(0, value, 0x142, 0xA, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_update_dpp(0, value, 0x143, 0xC, 0xF, true);
    return value + tmp;
}

template <int NumCtas>
__device__ __forceinline__ void zero_moe_output(void* output, int tokens)
{
    if(output == nullptr)
        return;

    constexpr long long row_bytes = static_cast<long long>(kHiddenSize) * 2;
    const long long total_vecs = (static_cast<long long>(tokens) * row_bytes) / sizeof(int4);
    const long long global_thread =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr long long stride = static_cast<long long>(NumCtas) * kThreads;
    auto* output_vec = reinterpret_cast<int4*>(output);
    const int4 zero = {0, 0, 0, 0};

    for(long long i = global_thread; i < total_vecs; i += stride)
        output_vec[i] = zero;
}

template <int NumExperts, int TopK>
__device__ __forceinline__ void build_sorted_metadata(
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* sorted_token_ids,
    float* sorted_weights,
    int32_t* sorted_expert_ids,
    int32_t* num_valid_ids,
    int tokens)
{
    // Experts are scanned as complete 64-lane waves; a trailing partial wave
    // (E=385) is covered by the running prefix of the last full wave.
    constexpr int kFullExpertWaves = NumExperts / 64;
    __shared__ int expert_counts[NumExperts];
    __shared__ int expert_offsets[NumExperts + 1];
    __shared__ int expert_cursors[NumExperts];
    __shared__ int wave_totals[16];

    const int tid = threadIdx.x;
    for(int expert = tid; expert < NumExperts; expert += kThreads)
    {
        expert_counts[expert] = 0;
        expert_cursors[expert] = 0;
    }
    __syncthreads();

    const int total_pairs = tokens * TopK;
    for(int index = tid; index < total_pairs; index += kThreads)
    {
        const int expert = topk_ids[index];
        if(expert >= 0 && expert < NumExperts)
            atomicAdd(expert_counts + expert, 1);
    }
    __syncthreads();

    // At decode expert counts a small shared-memory prefix scan is cheaper
    // than the generic Opus one-shot bookkeeping.
    const int lane = tid & 63;
    const int wave = tid >> 6;
    int value = tid < NumExperts ? round_up_to_block(expert_counts[tid]) : 0;
    value = dpp_inclusive_scan_wave(value);
    if(lane == 63 && wave < kFullExpertWaves)
        wave_totals[wave] = value;
    __syncthreads();
    if(wave == 0)
    {
        int wave_value = lane < kFullExpertWaves ? wave_totals[lane] : 0;
        wave_value = dpp_inclusive_scan_wave(wave_value);
        if(lane < kFullExpertWaves)
            wave_totals[lane] = wave_value;
    }
    __syncthreads();

    const int wave_prefix = wave == 0 ? 0 : wave_totals[wave - 1];
    if(tid < NumExperts)
    {
        expert_offsets[tid + 1] = value + wave_prefix;
        expert_cursors[tid] = 0;
    }
    if(tid == 0)
        expert_offsets[0] = 0;
    __syncthreads();

    if(tid == 0)
    {
        num_valid_ids[0] = expert_offsets[NumExperts];
        num_valid_ids[1] = tokens;
    }
    __syncthreads();

    for(int index = tid; index < total_pairs; index += kThreads)
    {
        const int expert = topk_ids[index];
        if(expert < 0 || expert >= NumExperts)
            continue;

        const int token = index / TopK;
        const int topk = index - token * TopK;
        const int position = expert_offsets[expert] + atomicAdd(expert_cursors + expert, 1);
        sorted_token_ids[position] = pack_token_topk_id(token, topk);
        sorted_weights[position] = topk_weights[index];
    }
    __syncthreads();

    const int sentinel = pack_token_topk_id(tokens, 0);
    for(int expert = tid; expert < NumExperts; expert += kThreads)
    {
        const int count = expert_counts[expert];
        const int start = expert_offsets[expert];
        const int end = expert_offsets[expert + 1];

        for(int position = start + count; position < end; ++position)
        {
            sorted_token_ids[position] = sentinel;
            sorted_weights[position] = 0.0f;
        }
        for(int block = start / kBlockM; block < end / kBlockM; ++block)
            sorted_expert_ids[block] = expert;
    }
}

template <int QuantCtas>
__device__ __forceinline__ void quantize_activations_compact(
    const hip_bfloat16* hidden_states,
    uint8_t* activation_quant,
    uint8_t* activation_scale_token,
    int tokens)
{
    using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));
    constexpr int groups_per_token = kHiddenSize / 32;
    constexpr int groups_per_wave = 16;
    constexpr int groups_per_cta = (kThreads / 64) * groups_per_wave;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int group_in_wave = lane >> 2;
    const int lane_in_group = lane & 3;
    const int total_groups = tokens * groups_per_token;
    const int total_batches = (total_groups + groups_per_cta - 1) / groups_per_cta;
    const int quant_cta = blockIdx.x - 1;
    const int batches_per_cta = (total_batches + QuantCtas - 1) / QuantCtas;
    const int batch_begin = quant_cta * batches_per_cta;
    const int batch_end = min(batch_begin + batches_per_cta, total_batches);

    for(int batch = batch_begin; batch < batch_end; ++batch)
    {
        const int group = batch * groups_per_cta + wave * groups_per_wave + group_in_wave;
        if(group >= total_groups)
            continue;

        const int element_offset = group * 32 + lane_in_group * 8;
        uint32_t raw[4];
        *reinterpret_cast<int4*>(raw) =
            *reinterpret_cast<const int4*>(hidden_states + element_offset);

        uint16_t local_amax = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const uint16_t lo = static_cast<uint16_t>(raw[i] & 0xffffu) & 0x7fffu;
            const uint16_t hi = static_cast<uint16_t>(raw[i] >> 16) & 0x7fffu;
            local_amax = max(local_amax, max(lo, hi));
        }

        uint32_t max_bits = local_amax;
        max_bits = max(
            max_bits,
            static_cast<uint32_t>(
                __builtin_amdgcn_mov_dpp(static_cast<int>(max_bits), 0xB1, 0xF, 0xF, true)));
        max_bits = max(
            max_bits,
            static_cast<uint32_t>(
                __builtin_amdgcn_mov_dpp(static_cast<int>(max_bits), 0x4E, 0xF, 0xF, true)));
        const float amax =
            __uint_as_float(static_cast<uint32_t>(max_bits & 0xffffu) << 16);
        const float dequant_scale = aiter::fp4_f32_to_e8m0_scale(amax);
        const uint8_t scale =
            (static_cast<uint32_t>(__builtin_bit_cast(uint32_t, dequant_scale)) >> 23) & 0xff;

        uint32_t packed = 0;
#if defined(__gfx950__)
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[0]), dequant_scale, 0);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[1]), dequant_scale, 1);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[2]), dequant_scale, 2);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[3]), dequant_scale, 3);
#else
        __builtin_trap();
#endif

        *reinterpret_cast<uint32_t*>(
            activation_quant + static_cast<size_t>(group) * 16 + lane_in_group * 4) = packed;
        if(lane_in_group == 0)
            activation_scale_token[group] = scale;
    }
}

template <int NumExperts, int TopK>
__global__ void __launch_bounds__(kThreads) moe_sort_quant_kernel(
    const hip_bfloat16* hidden_states,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* sorted_token_ids,
    float* sorted_weights,
    int32_t* sorted_expert_ids,
    int32_t* num_valid_ids,
    void* moe_buf,
    uint8_t* activation_quant,
    uint8_t* activation_scale_token,
    int tokens)
{
    zero_moe_output<kSortQuantCtas>(moe_buf, tokens);
    if(blockIdx.x == 0)
    {
        build_sorted_metadata<NumExperts, TopK>(
            topk_ids,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            tokens);
    }
    else
    {
        quantize_activations_compact<kSortQuantCtas - 1>(
            hidden_states, activation_quant, activation_scale_token, tokens);
    }
}

// The activation quantization above lowers to v_cvt_scalef32_pk_fp4_bf16, which
// is part of the fp4-cvt-scale instruction set introduced with gfx950 (CDNA4).
// Earlier CDNA parts have no FP4 convert instruction at all, so there is no
// gfx942 equivalent to substitute -- and the MXFP4 MoE GEMM that consumes this
// kernel's output is gfx950-only for the same reason, so a software-emulated
// fallback would produce bytes nothing downstream could use. Reject other
// architectures on the host instead, so an unsupported call reports an error
// rather than executing the __builtin_trap() the device path compiles to.
//
// Cached per device: this sits on the decode path and hipGetDeviceProperties is
// not cheap enough to call per launch.
inline bool is_fp4_cvt_capable(int device_id)
{
    static const std::vector<char> capable = [] {
        int count = 0;
        HIP_CALL(hipGetDeviceCount(&count));
        std::vector<char> caps(static_cast<size_t>(count), 0);
        for(int i = 0; i < count; ++i)
        {
            hipDeviceProp_t prop;
            HIP_CALL(hipGetDeviceProperties(&prop, i));
            caps[static_cast<size_t>(i)] =
                std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
        }
        return caps;
    }();
    return device_id >= 0 && device_id < static_cast<int>(capable.size()) &&
           capable[static_cast<size_t>(device_id)] != 0;
}

} // namespace aiter::mxfp4_moe

void mxfp4_moe_sort_quant_fwd(aiter_tensor_t& hidden_states,
                              aiter_tensor_t& topk_ids,
                              aiter_tensor_t& topk_weights,
                              aiter_tensor_t& sorted_token_ids,
                              aiter_tensor_t& sorted_weights,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& num_valid_ids,
                              aiter_tensor_t& moe_buf,
                              aiter_tensor_t& activation_quant,
                              aiter_tensor_t& activation_scale_token,
                              int num_experts)
{
    using namespace aiter::mxfp4_moe;

    const int device_id = hidden_states.device_id;
    AITER_CHECK(is_fp4_cvt_capable(device_id),
                __func__,
                ": requires gfx950; the fp4 activation quantization uses the "
                "scalef32 fp4-cvt instructions, which no earlier architecture has");
    // An off-device tensor would otherwise be dereferenced by the kernel under
    // this call's HipDeviceGuard and fault, or silently read another device's
    // memory. moe_buf is exempt when empty: the zeroing pass skips it entirely.
    AITER_CHECK(topk_ids.device_id == device_id && topk_weights.device_id == device_id &&
                    sorted_token_ids.device_id == device_id &&
                    sorted_weights.device_id == device_id &&
                    sorted_expert_ids.device_id == device_id &&
                    num_valid_ids.device_id == device_id &&
                    (moe_buf.numel() == 0 || moe_buf.device_id == device_id) &&
                    activation_quant.device_id == device_id &&
                    activation_scale_token.device_id == device_id,
                __func__,
                ": all tensors must be on the same device");

    AITER_CHECK(hidden_states.dtype() == AITER_DTYPE_bf16,
                __func__,
                ": hidden_states must be BF16");
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 &&
                    topk_weights.dtype() == AITER_DTYPE_fp32,
                __func__,
                ": expected int32 topk_ids and fp32 topk_weights");
    AITER_CHECK(sorted_token_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_weights.dtype() == AITER_DTYPE_fp32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    num_valid_ids.dtype() == AITER_DTYPE_i32,
                __func__,
                ": expected int32 sorted_token_ids, fp32 sorted_weights, int32 "
                "sorted_expert_ids, and int32 num_valid_ids");
    const int topk = topk_ids.dim() == 2 ? static_cast<int>(topk_ids.size(1)) : 0;

#define AITER_MXFP4_SORT_QUANT_MATCH(NUM_EXPERTS, TOPK) \
    || (num_experts == (NUM_EXPERTS) && topk == (TOPK))
    const bool route_supported =
        false AITER_MXFP4_SORT_QUANT_ROUTES(AITER_MXFP4_SORT_QUANT_MATCH);
#undef AITER_MXFP4_SORT_QUANT_MATCH

    AITER_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) == kHiddenSize &&
                    topk_ids.dim() == 2 && topk_ids.size(0) == hidden_states.size(0) &&
                    route_supported,
                __func__,
                ": only H=7168 with (num_experts, topk) in "
                "{(256,8), (384,8), (385,9)} is supported");
    AITER_CHECK(hidden_states.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_weights.is_contiguous() && sorted_token_ids.is_contiguous() &&
                    sorted_weights.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    num_valid_ids.is_contiguous() && moe_buf.is_contiguous() &&
                    activation_quant.is_contiguous() &&
                    activation_scale_token.is_contiguous(),
                __func__,
                ": all tensors must be contiguous");
    AITER_CHECK(activation_quant.dtype() == AITER_DTYPE_fp4x2 ||
                    activation_quant.dtype() == AITER_DTYPE_u8,
                __func__,
                ": activation_quant must be packed FP4");
    AITER_CHECK(activation_scale_token.dtype() == AITER_DTYPE_fp8_e8m0 ||
                    activation_scale_token.dtype() == AITER_DTYPE_u8,
                __func__,
                ": activation_scale_token must hold e8m0 bytes");

    const int tokens = static_cast<int>(hidden_states.size(0));
    AITER_CHECK(tokens > 0 && tokens <= 128,
                __func__,
                ": only decode M in [1, 128] is supported");
    const int64_t route_count = topk_ids.numel();
    const int64_t active_experts =
        route_count < static_cast<int64_t>(num_experts) ? route_count : num_experts;
    const int64_t required_sorted =
        ((route_count + active_experts * (kBlockM - 1) + kBlockM - 1) /
         kBlockM) *
        kBlockM;
    AITER_CHECK(topk_weights.numel() == topk_ids.numel() &&
                    activation_quant.numel() == hidden_states.numel() / 2 &&
                    activation_scale_token.numel() ==
                        hidden_states.size(0) * kScaleCols &&
                    num_valid_ids.numel() >= 2,
                __func__,
                ": incompatible fused sort/quant tensor sizes");
    AITER_CHECK(moe_buf.numel() == 0 ||
                    ((moe_buf.dtype() == AITER_DTYPE_bf16 ||
                      moe_buf.dtype() == AITER_DTYPE_fp16) &&
                     moe_buf.numel() >=
                         static_cast<int64_t>(tokens) * kHiddenSize),
                __func__,
                ": moe_buf must be empty or contain at least M*7168 BF16/FP16 elements");

    AITER_CHECK(sorted_token_ids.numel() >= required_sorted &&
                    sorted_weights.numel() >= required_sorted &&
                    sorted_expert_ids.numel() >= required_sorted / kBlockM,
                __func__,
                ": output routing buffers are too small");

    HipDeviceGuard device_guard(hidden_states.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const hip_bfloat16* hidden_ptr =
        static_cast<const hip_bfloat16*>(hidden_states.data_ptr());
    const int32_t* topk_ids_ptr = static_cast<const int32_t*>(topk_ids.data_ptr());
    const float* topk_weights_ptr = static_cast<const float*>(topk_weights.data_ptr());
    int32_t* sorted_ids_ptr = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    float* sorted_weights_ptr = static_cast<float*>(sorted_weights.data_ptr());
    int32_t* sorted_experts_ptr = static_cast<int32_t*>(sorted_expert_ids.data_ptr());
    int32_t* num_valid_ptr = static_cast<int32_t*>(num_valid_ids.data_ptr());
    void* moe_buf_ptr = moe_buf.numel() == 0 ? nullptr : moe_buf.data_ptr();
    uint8_t* activation_quant_ptr = static_cast<uint8_t*>(activation_quant.data_ptr());
    uint8_t* activation_scale_token_ptr =
        static_cast<uint8_t*>(activation_scale_token.data_ptr());
#define AITER_MXFP4_SORT_QUANT_LAUNCH(NUM_EXPERTS, TOPK)                            \
    if(num_experts == (NUM_EXPERTS) && topk == (TOPK))                              \
    {                                                                               \
        aiter::mxfp4_moe::moe_sort_quant_kernel<(NUM_EXPERTS), (TOPK)>              \
            <<<dim3(kSortQuantCtas), dim3(kThreads), 0, stream>>>(                  \
                hidden_ptr,                                                         \
                topk_ids_ptr,                                                       \
                topk_weights_ptr,                                                   \
                sorted_ids_ptr,                                                     \
                sorted_weights_ptr,                                                 \
                sorted_experts_ptr,                                                 \
                num_valid_ptr,                                                      \
                moe_buf_ptr,                                                        \
                activation_quant_ptr,                                               \
                activation_scale_token_ptr,                                         \
                tokens);                                                            \
    }
    AITER_MXFP4_SORT_QUANT_ROUTES(AITER_MXFP4_SORT_QUANT_LAUNCH)
#undef AITER_MXFP4_SORT_QUANT_LAUNCH

    const hipError_t launch_status = hipGetLastError();
    AITER_CHECK(
        launch_status == hipSuccess,
        __func__,
        ": fused sort+quant launch failed");
}
