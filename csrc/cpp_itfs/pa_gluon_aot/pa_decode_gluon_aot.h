// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <string>

namespace aiter {

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = call;                                             \
        if (err != hipSuccess) {                                           \
            const char* errstr = hipGetErrorString(err);                   \
            fprintf(stderr, "HIP Error: %s at %s:%d\n",                   \
                    errstr, __FILE__, __LINE__);                           \
            throw std::runtime_error(errstr);                              \
        }                                                                  \
    } while (0)

struct CachedKernel {
    hipFunction_t function = nullptr;
    hipModule_t   module   = nullptr;
    int shared_mem  = 0;
    int num_warps   = 0;
};

struct PaDecodeCacheEntry {
    CachedKernel attention;
    CachedKernel reduce;
};

inline std::string scalar_type_to_str(at::ScalarType dtype) {
    switch (dtype) {
        case at::ScalarType::BFloat16:          return "torch.bfloat16";
        case at::ScalarType::Half:              return "torch.float16";
        case at::ScalarType::Float8_e4m3fnuz:   return "torch.float8_e4m3fnuz";
        case at::ScalarType::Float8_e4m3fn:     return "torch.float8_e4m3fn";
        default:
            throw std::runtime_error(
                std::string("pa_decode_gluon_aot: unsupported compute type: ")
                + toString(dtype));
    }
}

inline int get_cdna_version() {
    static int version = []() {
        hipDeviceProp_t props;
        int device = 0;
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipGetDeviceProperties(&props, device));
        std::string arch(props.gcnArchName);
        if (arch.find("gfx950") != std::string::npos) return 4;
        if (arch.find("gfx942") != std::string::npos) return 3;
        return -1;
    }();
    return version;
}

inline int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

inline float fp8_max_value(at::ScalarType dtype) {
    if (dtype == at::ScalarType::Float8_e4m3fnuz)
        return static_cast<float>(std::numeric_limits<c10::Float8_e4m3fnuz>::max());
    if (dtype == at::ScalarType::Float8_e4m3fn)
        return static_cast<float>(std::numeric_limits<c10::Float8_e4m3fn>::max());
    return 1.0f;
}

inline std::string pa_decode_cache_key(
    const std::string& compute_type_str,
    int query_seq_len,
    int one_query_group_size,
    int head_size_pow2,
    int kv_block_size,
    int context_partition_size,
    int query_quant_mode,
    int kv_quant_mode,
    float fp8_max_val,
    int value_transposed,
    int is_causal,
    int use_sinks,
    int cdna_version)
{
    return compute_type_str + "_" +
        std::to_string(query_seq_len) + "_" +
        std::to_string(one_query_group_size) + "_" +
        std::to_string(head_size_pow2) + "_" +
        std::to_string(kv_block_size) + "_" +
        std::to_string(context_partition_size) + "_" +
        std::to_string(query_quant_mode) + "_" +
        std::to_string(kv_quant_mode) + "_" +
        std::to_string(fp8_max_val) + "_" +
        std::to_string(value_transposed) + "_" +
        std::to_string(is_causal) + "_" +
        std::to_string(use_sinks) + "_" +
        std::to_string(cdna_version);
}

void pa_decode_gluon_aot(
    torch::Tensor& output,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& context_lengths,
    torch::Tensor& block_tables,
    float softmax_scale,
    int query_length,
    int max_context_partition_num,
    int context_partition_size,
    at::ScalarType compute_type,
    const torch::Tensor& query_scale,
    const torch::Tensor& key_scale,
    const torch::Tensor& value_scale,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& temporary_output,
    const torch::Tensor& sinks = {},
    void* stream = nullptr);

}  // namespace aiter
