// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "pa_decode_gluon_aot.h"
#include "utils.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <c10/hip/HIPStream.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace aiter {

static std::unique_ptr<LRUCache<std::string, PaDecodeCacheEntry>> g_kernel_cache;
static std::once_flag init_kernel_cache_flag;
static std::mutex g_cache_mutex;

static CachedKernel load_cached_kernel(
    const py::bytes& hsaco_bytes,
    const std::string& kernel_name,
    int shared_mem,
    int num_warps)
{
    CachedKernel cached;
    cached.shared_mem = shared_mem;
    cached.num_warps  = num_warps;

    std::string hsaco_data = static_cast<std::string>(hsaco_bytes);

    HIP_CHECK(hipModuleLoadData(&cached.module,
              reinterpret_cast<const void*>(hsaco_data.data())));
    HIP_CHECK(hipModuleGetFunction(&cached.function, cached.module,
              kernel_name.c_str()));
    return cached;
}

// Cold-path: call Python warmup, load HSACO
static PaDecodeCacheEntry warmup_and_load(
    const std::string& compute_type,
    int query_seq_len,
    int one_query_group_size,
    int head_size,
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
    py::gil_scoped_acquire gil;

    py::module_ warmup_mod =
        py::module_::import("csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot_warmup");

    py::dict result = warmup_mod.attr("warmup_pa_decode")(
        compute_type,
        query_seq_len,
        one_query_group_size,
        head_size,
        kv_block_size,
        context_partition_size,
        query_quant_mode,
        kv_quant_mode,
        fp8_max_val,
        value_transposed,
        is_causal,
        use_sinks,
        cdna_version
    ).cast<py::dict>();

    PaDecodeCacheEntry entry;

    entry.attention = load_cached_kernel(
        result["attention_hsaco"].cast<py::bytes>(),
        result["attention_name"].cast<std::string>(),
        result["attention_shared_mem"].cast<int>(),
        result["attention_num_warps"].cast<int>());

    entry.reduce = load_cached_kernel(
        result["reduce_hsaco"].cast<py::bytes>(),
        result["reduce_name"].cast<std::string>(),
        result["reduce_shared_mem"].cast<int>(),
        result["reduce_num_warps"].cast<int>());

    return entry;
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
    const torch::Tensor& sinks,
    void* stream)
{
    const int num_query_heads  = query.size(1);
    const int head_size        = query.size(-1);
    const int batch_size       = query.size(0) / query_length;
    const int num_kv_heads     = key_cache.size(1);
    const int query_group_size = num_query_heads / num_kv_heads;
    const int kv_block_size    = key_cache.size(-2);

    const bool is_causal       = (query_length > 1);
    const bool value_transposed = (value_cache.dim() == 5);
    const bool use_sinks_flag   = sinks.defined();

    int query_quant_mode = -1;
    int kv_quant_mode    = -1;
    if (query_scale.defined())
        query_quant_mode = (query_scale.numel() == 1) ? 0 : 1;
    if (key_scale.defined() && value_scale.defined())
        kv_quant_mode = (key_scale.numel() == 1) ? 0 : 1;

    const float fp8_max_val    = fp8_max_value(value_cache.scalar_type());
    const int cdna_ver         = get_cdna_version();
    const int head_size_pow2   = next_pow2(head_size);
    const std::string compute_type_str = scalar_type_to_str(compute_type);

    auto query_5d  = query.reshape(
        {batch_size, query_length, num_kv_heads, query_group_size, head_size});
    auto output_5d = output.reshape(
        {batch_size, query_length, num_kv_heads, query_group_size, head_size});

    torch::Tensor query_scale_5d;
    int stride_query_scale_bs      = 0;
    int stride_query_scale_qlen    = 0;
    int stride_query_scale_kv_head = 0;

    if (query_scale.defined()) {
        if (query_scale.numel() == 1) {
            query_scale_5d = query_scale;
        } else {
            query_scale_5d = query_scale.reshape(
                {batch_size, query_length, num_kv_heads, query_group_size, 1});
            stride_query_scale_bs      = query_scale_5d.stride(0);
            stride_query_scale_qlen    = query_scale_5d.stride(1);
            stride_query_scale_kv_head = query_scale_5d.stride(2);
        }
    }

    int kv_scale_stride_0 = 0;
    int kv_scale_stride_1 = 0;
    if (key_scale.defined() && key_scale.numel() > 1) {
        kv_scale_stride_0 = key_scale.stride(0);
        kv_scale_stride_1 = key_scale.stride(1);
    }

    std::string key = pa_decode_cache_key(
        compute_type_str, query_length, query_group_size, head_size_pow2,
        kv_block_size, context_partition_size, query_quant_mode, kv_quant_mode,
        fp8_max_val, static_cast<int>(value_transposed),
        static_cast<int>(is_causal), static_cast<int>(use_sinks_flag),
        cdna_ver);

    std::call_once(init_kernel_cache_flag,
        init_lru_cache<std::string, PaDecodeCacheEntry>, g_kernel_cache);

    PaDecodeCacheEntry cache_entry;
    {
        PaDecodeCacheEntry* entry_ptr = g_kernel_cache->get(key);
        if (entry_ptr != nullptr) {
            cache_entry = *entry_ptr;
        } else {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            entry_ptr = g_kernel_cache->get(key);
            if (entry_ptr != nullptr) {
                cache_entry = *entry_ptr;
            } else {
                cache_entry = warmup_and_load(
                    compute_type_str, query_length, query_group_size, head_size,
                    kv_block_size, context_partition_size, query_quant_mode,
                    kv_quant_mode, fp8_max_val, static_cast<int>(value_transposed),
                    static_cast<int>(is_causal), static_cast<int>(use_sinks_flag),
                    cdna_ver);
                g_kernel_cache->put(key, cache_entry);
            }
        }
    }

    hipStream_t hip_stream;
    if (stream != nullptr) {
        hip_stream = reinterpret_cast<hipStream_t>(stream);
    } else {
        hip_stream = c10::hip::getCurrentHIPStream(
                         output.device().index()).stream();
    }

    {
        float softmax_scale_f32 = softmax_scale;

        hipDeviceptr_t p_exp_sums    = reinterpret_cast<hipDeviceptr_t>(exp_sums.data_ptr());
        hipDeviceptr_t p_max_logits  = reinterpret_cast<hipDeviceptr_t>(max_logits.data_ptr());
        hipDeviceptr_t p_tmp_output  = reinterpret_cast<hipDeviceptr_t>(temporary_output.data_ptr());
        hipDeviceptr_t p_query       = reinterpret_cast<hipDeviceptr_t>(query_5d.data_ptr());
        hipDeviceptr_t p_key_cache   = reinterpret_cast<hipDeviceptr_t>(key_cache.data_ptr());
        hipDeviceptr_t p_value_cache = reinterpret_cast<hipDeviceptr_t>(value_cache.data_ptr());
        hipDeviceptr_t p_block_tbl   = reinterpret_cast<hipDeviceptr_t>(block_tables.data_ptr());
        hipDeviceptr_t p_ctx_lens    = reinterpret_cast<hipDeviceptr_t>(context_lengths.data_ptr());

        hipDeviceptr_t p_q_scale = query_scale_5d.defined()
            ? reinterpret_cast<hipDeviceptr_t>(query_scale_5d.data_ptr())
            : hipDeviceptr_t(0);
        hipDeviceptr_t p_k_scale = key_scale.defined()
            ? reinterpret_cast<hipDeviceptr_t>(key_scale.data_ptr())
            : hipDeviceptr_t(0);
        hipDeviceptr_t p_v_scale = value_scale.defined()
            ? reinterpret_cast<hipDeviceptr_t>(value_scale.data_ptr())
            : hipDeviceptr_t(0);

        hipDeviceptr_t global_scratch  = 0;
        hipDeviceptr_t profile_scratch = 0;

        int32_t s_es_0 = exp_sums.stride(0);
        int32_t s_es_1 = exp_sums.stride(1);
        int32_t s_es_2 = exp_sums.stride(2);

        int32_t s_to_0 = temporary_output.stride(0);
        int32_t s_to_1 = temporary_output.stride(1);
        int32_t s_to_2 = temporary_output.stride(2);
        int32_t s_to_3 = temporary_output.stride(3);

        int32_t s_q5_0 = query_5d.stride(0);
        int32_t s_q5_1 = query_5d.stride(1);
        int32_t s_q5_2 = query_5d.stride(2);
        int32_t s_q5_3 = query_5d.stride(3);

        int32_t s_kc_0 = key_cache.stride(0);
        int32_t s_kc_1 = key_cache.stride(1);
        int32_t s_kc_2 = key_cache.stride(2);
        int32_t s_kc_3 = key_cache.stride(3);

        int32_t s_vc_0 = value_cache.stride(0);
        int32_t s_vc_1 = value_cache.stride(1);
        int32_t s_vc_2 = value_cache.stride(2);

        int32_t s_bt_0 = block_tables.stride(0);

        int32_t qs_bs  = stride_query_scale_bs;
        int32_t qs_ql  = stride_query_scale_qlen;
        int32_t qs_kh  = stride_query_scale_kv_head;
        int32_t kvs0   = kv_scale_stride_0;
        int32_t kvs1   = kv_scale_stride_1;

        int32_t a_hs   = head_size;
        int32_t a_ns   = batch_size;
        int32_t a_nkh  = num_kv_heads;
        int32_t a_mcp  = max_context_partition_num;

        void* attn_args[] = {
            &p_exp_sums, &p_max_logits, &p_tmp_output, &p_query,
            &p_key_cache, &p_value_cache, &p_block_tbl, &p_ctx_lens,
            &softmax_scale_f32,
            &p_q_scale, &p_k_scale, &p_v_scale,
            &s_es_0, &s_es_1, &s_es_2,
            &s_to_0, &s_to_1, &s_to_2, &s_to_3,
            &s_q5_0, &s_q5_1, &s_q5_2, &s_q5_3,
            &s_kc_0, &s_kc_1, &s_kc_2, &s_kc_3,
            &s_vc_0, &s_vc_1, &s_vc_2,
            &s_bt_0,
            &qs_bs, &qs_ql, &qs_kh,
            &kvs0, &kvs1,
            &a_hs, &a_ns, &a_nkh, &a_mcp,
            &global_scratch, &profile_scratch,
        };

        const int attn_block_x = cache_entry.attention.num_warps * 64;
        const unsigned int gX = static_cast<unsigned int>(batch_size);
        const unsigned int gY = static_cast<unsigned int>(num_kv_heads);
        const unsigned int gZ = static_cast<unsigned int>(max_context_partition_num);

        if (gX * gY * gZ > 0) {
            HIP_CHECK(hipModuleLaunchKernel(
                cache_entry.attention.function,
                gX, gY, gZ,
                attn_block_x, 1, 1,
                cache_entry.attention.shared_mem,
                hip_stream,
                attn_args,
                nullptr));
        }
    }

    {
        hipDeviceptr_t r_output   = reinterpret_cast<hipDeviceptr_t>(output_5d.data_ptr());
        hipDeviceptr_t r_exp_sums = reinterpret_cast<hipDeviceptr_t>(exp_sums.data_ptr());
        hipDeviceptr_t r_max_log  = reinterpret_cast<hipDeviceptr_t>(max_logits.data_ptr());
        hipDeviceptr_t r_logits   = reinterpret_cast<hipDeviceptr_t>(temporary_output.data_ptr());
        hipDeviceptr_t r_ctx_lens = reinterpret_cast<hipDeviceptr_t>(context_lengths.data_ptr());
        hipDeviceptr_t r_sinks    = sinks.defined()
            ? reinterpret_cast<hipDeviceptr_t>(sinks.data_ptr())
            : hipDeviceptr_t(0);

        hipDeviceptr_t global_scratch  = 0;
        hipDeviceptr_t profile_scratch = 0;

        int32_t r_so_0 = output_5d.stride(0);
        int32_t r_so_1 = output_5d.stride(1);
        int32_t r_so_2 = output_5d.stride(2);
        int32_t r_so_3 = output_5d.stride(3);

        int32_t r_se_0 = exp_sums.stride(0);
        int32_t r_se_1 = exp_sums.stride(1);
        int32_t r_se_2 = exp_sums.stride(2);

        int32_t r_sl_0 = temporary_output.stride(0);
        int32_t r_sl_1 = temporary_output.stride(1);
        int32_t r_sl_2 = temporary_output.stride(2);
        int32_t r_sl_3 = temporary_output.stride(3);

        int32_t r_hs   = head_size;
        int32_t r_ns   = batch_size;
        int32_t r_nkh  = num_kv_heads;

        void* reduce_args[] = {
            &r_output, &r_exp_sums, &r_max_log, &r_logits,
            &r_ctx_lens, &r_sinks,
            &r_so_0, &r_so_1, &r_so_2, &r_so_3,
            &r_se_0, &r_se_1, &r_se_2,
            &r_sl_0, &r_sl_1, &r_sl_2, &r_sl_3,
            &r_hs, &r_ns, &r_nkh,
            &global_scratch, &profile_scratch,
        };

        const int reduce_block_x = cache_entry.reduce.num_warps * 64;
        const unsigned int gX = static_cast<unsigned int>(batch_size);
        const unsigned int gY = static_cast<unsigned int>(num_kv_heads);

        if (gX * gY > 0) {
            HIP_CHECK(hipModuleLaunchKernel(
                cache_entry.reduce.function,
                gX, gY, 1,
                reduce_block_x, 1, 1,
                cache_entry.reduce.shared_mem,
                hip_stream,
                reduce_args,
                nullptr));
        }
    }
}

}  // namespace aiter
