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
#include <unordered_map>

namespace py = pybind11;

namespace aiter {

static std::unique_ptr<LRUCache<std::string, PaDecodeCacheEntry>> g_kernel_cache;
static std::once_flag init_kernel_cache_flag;
static std::mutex g_cache_mutex;
static std::unordered_map<std::string, std::shared_ptr<std::mutex>> g_key_mutexes;

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

    auto guard = std::make_shared<HipModuleGuard>();
    HIP_CHECK(hipModuleLoadData(&guard->module,
              reinterpret_cast<const void*>(hsaco_data.data())));
    HIP_CHECK(hipModuleGetFunction(&cached.function, guard->module,
              kernel_name.c_str()));
    cached.module_guard = guard;
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
    // ==================== DEVICE VALIDATION ====================
    TORCH_CHECK(query.is_cuda(), "pa_decode_gluon_aot: query must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_cuda(), "pa_decode_gluon_aot: key_cache must be a CUDA tensor");
    TORCH_CHECK(value_cache.is_cuda(), "pa_decode_gluon_aot: value_cache must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "pa_decode_gluon_aot: output must be a CUDA tensor");
    TORCH_CHECK(context_lengths.is_cuda(), "pa_decode_gluon_aot: context_lengths must be a CUDA tensor");
    TORCH_CHECK(block_tables.is_cuda(), "pa_decode_gluon_aot: block_tables must be a CUDA tensor");
    TORCH_CHECK(exp_sums.is_cuda(), "pa_decode_gluon_aot: exp_sums must be a CUDA tensor");
    TORCH_CHECK(max_logits.is_cuda(), "pa_decode_gluon_aot: max_logits must be a CUDA tensor");
    TORCH_CHECK(temporary_output.is_cuda(), "pa_decode_gluon_aot: temporary_output must be a CUDA tensor");
    if (query_scale.defined())
        TORCH_CHECK(query_scale.is_cuda(), "pa_decode_gluon_aot: query_scale must be a CUDA tensor");
    if (key_scale.defined())
        TORCH_CHECK(key_scale.is_cuda(), "pa_decode_gluon_aot: key_scale must be a CUDA tensor");
    if (value_scale.defined())
        TORCH_CHECK(value_scale.is_cuda(), "pa_decode_gluon_aot: value_scale must be a CUDA tensor");
    if (sinks.defined())
        TORCH_CHECK(sinks.is_cuda(), "pa_decode_gluon_aot: sinks must be a CUDA tensor");

    // ==================== ARCHITECTURE VALIDATION ====================
    const int cdna_ver = get_cdna_version();
    TORCH_CHECK(cdna_ver == 3 || cdna_ver == 4,
        "pa_decode_gluon_aot: unsupported GPU architecture (cdna_version=",
        cdna_ver, "). Only gfx942 (CDNA3) and gfx950 (CDNA4) are supported.");

    // gfx942 (CDNA3) -> Float8_e4m3fnuz, gfx950 (CDNA4) -> Float8_e4m3fn
    const at::ScalarType arch_fp8_type = (cdna_ver == 3)
        ? at::ScalarType::Float8_e4m3fnuz
        : at::ScalarType::Float8_e4m3fn;

    // ==================== DTYPE VALIDATION ====================
    auto is_supported_data_dtype = [arch_fp8_type](at::ScalarType dt) {
        return dt == arch_fp8_type ||
               dt == at::ScalarType::BFloat16 ||
               dt == at::ScalarType::Half;
    };
    TORCH_CHECK(is_supported_data_dtype(query.scalar_type()),
        "pa_decode_gluon_aot: query dtype must be fp8/bf16/fp16, got ", query.scalar_type());
    TORCH_CHECK(is_supported_data_dtype(key_cache.scalar_type()),
        "pa_decode_gluon_aot: key_cache dtype must be fp8/bf16/fp16, got ", key_cache.scalar_type());
    TORCH_CHECK(is_supported_data_dtype(value_cache.scalar_type()),
        "pa_decode_gluon_aot: value_cache dtype must be fp8/bf16/fp16, got ", value_cache.scalar_type());
    TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16 ||
                output.scalar_type() == at::ScalarType::Half,
        "pa_decode_gluon_aot: output dtype must be bf16/fp16, got ", output.scalar_type());
    TORCH_CHECK(context_lengths.scalar_type() == at::ScalarType::Int,
        "pa_decode_gluon_aot: context_lengths dtype must be int32, got ", context_lengths.scalar_type());
    TORCH_CHECK(block_tables.scalar_type() == at::ScalarType::Int,
        "pa_decode_gluon_aot: block_tables dtype must be int32, got ", block_tables.scalar_type());
    TORCH_CHECK(exp_sums.scalar_type() == at::ScalarType::Float,
        "pa_decode_gluon_aot: exp_sums dtype must be float32, got ", exp_sums.scalar_type());
    TORCH_CHECK(max_logits.scalar_type() == at::ScalarType::Float,
        "pa_decode_gluon_aot: max_logits dtype must be float32, got ", max_logits.scalar_type());

    // ==================== COMPUTE TYPE VALIDATION ====================
    TORCH_CHECK(compute_type == at::ScalarType::BFloat16 ||
                compute_type == at::ScalarType::Half ||
                compute_type == arch_fp8_type,
        "pa_decode_gluon_aot: compute_type must be bf16/fp16/fp8, got ", compute_type);

    // ==================== SHAPE VALIDATION ====================
    TORCH_CHECK(query.dim() == 3,
        "pa_decode_gluon_aot: expected 3D query tensor, but got shape with dim=", query.dim());
    TORCH_CHECK(output.dim() == 3,
        "pa_decode_gluon_aot: expected 3D output tensor, but got shape with dim=", output.dim());
    TORCH_CHECK(key_cache.dim() == 5,
        "pa_decode_gluon_aot: expected 5D key_cache tensor, but got shape with dim=", key_cache.dim());

    TORCH_CHECK(query_length > 0 && query_length <= 4,
        "pa_decode_gluon_aot: query_length must be in [1, 4], but got ", query_length);
    TORCH_CHECK(query.size(0) % query_length == 0,
        "pa_decode_gluon_aot: query.size(0) (", query.size(0),
        ") must be divisible by query_length (", query_length, ")");

    const int num_query_heads  = query.size(1);
    const int num_kv_heads     = key_cache.size(1);
    TORCH_CHECK(num_kv_heads > 0,
        "pa_decode_gluon_aot: num_kv_heads must be positive");
    TORCH_CHECK(num_query_heads % num_kv_heads == 0,
        "pa_decode_gluon_aot: num_query_heads (", num_query_heads,
        ") must be divisible by num_kv_heads (", num_kv_heads, ")");

    const int head_size        = query.size(-1);
    const int batch_size       = query.size(0) / query_length;
    const int query_group_size = num_query_heads / num_kv_heads;
    const int kv_block_size    = key_cache.size(-2);
    const int equivalent_query_group_size = query_length * query_group_size;

    TORCH_CHECK(equivalent_query_group_size <= 64,
        "pa_decode_gluon_aot: query_length * query_group_size (", equivalent_query_group_size,
        ") exceeds maximum of 64");
    TORCH_CHECK(kv_block_size == 16 || kv_block_size == 64 || kv_block_size == 1024,
        "pa_decode_gluon_aot: kv_block_size (", kv_block_size, ") must be one of [16, 64, 1024]");

    TORCH_CHECK(value_cache.dim() == 4 || value_cache.dim() == 5,
        "pa_decode_gluon_aot: expected 4D or 5D value_cache tensor, but got dim=", value_cache.dim());
    const bool is_causal       = (query_length > 1);
    const bool value_transposed = (value_cache.dim() == 5);
    const bool use_sinks_flag   = sinks.defined();

    // ==================== QUANTIZATION MODE CONFIGURATION ====================
    int query_quant_mode = -1;
    int kv_quant_mode    = -1;

    if (query_scale.defined()) {
        TORCH_CHECK(query_scale.scalar_type() == at::ScalarType::Float,
            "pa_decode_gluon_aot: query_scale dtype must be float32, got ",
            query_scale.scalar_type());
        if (query_scale.numel() == 1) {
            query_quant_mode = 0;
        } else {
            TORCH_CHECK(query_scale.dim() == 3,
                "pa_decode_gluon_aot: expected 3D query_scale tensor for per-token mode, "
                "but got dim=", query_scale.dim());
            TORCH_CHECK(query_scale.size(-1) == 1,
                "pa_decode_gluon_aot: query_scale.size(-1) must be 1, but got ",
                query_scale.size(-1));
            query_quant_mode = 1;
        }
    }

    if (key_scale.defined() && value_scale.defined()) {
        TORCH_CHECK(key_scale.scalar_type() == at::ScalarType::Float,
            "pa_decode_gluon_aot: key_scale dtype must be float32, got ",
            key_scale.scalar_type());
        TORCH_CHECK(value_scale.scalar_type() == at::ScalarType::Float,
            "pa_decode_gluon_aot: value_scale dtype must be float32, got ",
            value_scale.scalar_type());
        if (key_scale.numel() == 1) {
            kv_quant_mode = 0;
        } else {
            TORCH_CHECK(key_scale.dim() == 4,
                "pa_decode_gluon_aot: expected 4D key_scale tensor for per-token mode, "
                "but got dim=", key_scale.dim());
            TORCH_CHECK(key_scale.size(-1) == 1,
                "pa_decode_gluon_aot: key_scale.size(-1) must be 1, but got ",
                key_scale.size(-1));
            kv_quant_mode = 1;
        }
        TORCH_CHECK(key_scale.sizes() == value_scale.sizes(),
            "pa_decode_gluon_aot: key_scale and value_scale must have the same shape, "
            "but got key_scale: ", key_scale.sizes(), ", value_scale: ", value_scale.sizes());
    }

    const float fp8_max_val    = fp8_max_value(value_cache.scalar_type());
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
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        PaDecodeCacheEntry* entry_ptr = g_kernel_cache->get(key);
        if (entry_ptr != nullptr) {
            cache_entry = *entry_ptr;
            goto kernel_ready;
        }
    }

    {
        std::shared_ptr<std::mutex> key_mtx;
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            auto& m = g_key_mutexes[key];
            if (!m) m = std::make_shared<std::mutex>();
            key_mtx = m;
        }

        std::lock_guard<std::mutex> key_lock(*key_mtx);

        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            PaDecodeCacheEntry* entry_ptr = g_kernel_cache->get(key);
            if (entry_ptr != nullptr) {
                cache_entry = *entry_ptr;
                goto kernel_ready;
            }
        }

        cache_entry = warmup_and_load(
            compute_type_str, query_length, query_group_size, head_size,
            kv_block_size, context_partition_size, query_quant_mode,
            kv_quant_mode, fp8_max_val, static_cast<int>(value_transposed),
            static_cast<int>(is_causal), static_cast<int>(use_sinks_flag),
            cdna_ver);

        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_kernel_cache->put(key, cache_entry);
        }
    }

kernel_ready:

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
