// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "pa_decode_gluon_aot.h"

#include <pybind11/embed.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace py = pybind11;

static void test_head_padding_cache_reuse() {
    const int num_seqs = 2;
    const int query_length = 4;
    const int num_query_heads = 4;
    const int num_kv_heads = 1;
    const int query_group_size = num_query_heads / num_kv_heads;
    const int context_length = 1025;
    const int partition_size = 256;
    const int page_size = 64;
    const int num_parts = (context_length + partition_size - 1) / partition_size;
    const int blocks_per_seq = (context_length + page_size - 1) / page_size;
    const int num_blocks = num_seqs * blocks_per_seq;
    const int eq_query_group_size = query_length * query_group_size;
    const int vec = 16 / static_cast<int>(torch::elementSize(torch::kBFloat16));
    auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto f32 = bf16.dtype(torch::kFloat32);
    auto i32 = bf16.dtype(torch::kInt32);
    auto lengths = torch::full({num_seqs}, context_length, i32);
    auto tables = torch::arange(num_blocks, i32).reshape({num_seqs, blocks_per_seq});
    torch::Tensor no_scale;

    // Count native C++ cache misses through its Python warmup entry point.
    auto warmup_mod = py::module_::import(
        "csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot_warmup");
    py::object original_warmup = warmup_mod.attr("warmup_pa_decode");
    std::vector<int> warmup_heads;
    warmup_mod.attr("warmup_pa_decode") = py::cpp_function(
        [&warmup_heads, original_warmup](py::args args) {
            warmup_heads.push_back(args[3].cast<int>());
            return original_warmup(*args);
        });

    try {
        for (int head_size : {256, 192, 256}) {
            auto query = torch::zeros(
                {num_seqs * query_length, num_query_heads, head_size}, bf16);
            auto output = torch::empty_like(query);
            // The extra NaN block catches an unmasked D192 read without leaving
            // allocated storage. Valid K=0 and V=1 give the exact reference 1.
            auto key_storage = torch::full(
                {num_blocks + 1, num_kv_heads, head_size / vec, page_size, vec},
                std::numeric_limits<float>::quiet_NaN(), bf16);
            auto value_storage = torch::full(
                {num_blocks + 1, num_kv_heads, page_size / vec, head_size, vec},
                std::numeric_limits<float>::quiet_NaN(), bf16);
            auto key = key_storage.narrow(0, 0, num_blocks);
            auto value = value_storage.narrow(0, 0, num_blocks);
            key.zero_();
            value.fill_(1);
            auto exp_sums = torch::zeros(
                {num_seqs, num_kv_heads, num_parts, eq_query_group_size}, f32);
            auto max_logits = torch::full(
                {num_seqs, num_kv_heads, num_parts, eq_query_group_size},
                -std::numeric_limits<float>::infinity(), f32);
            auto temporary_output = torch::zeros(
                {num_seqs, num_kv_heads, num_parts, eq_query_group_size, head_size},
                bf16);

            aiter::pa_decode_gluon_aot(
                output, query, key, value, lengths, tables,
                1.0f / std::sqrt(static_cast<float>(head_size)),
                query_length, num_parts, partition_size, at::ScalarType::BFloat16,
                {}, no_scale, no_scale, exp_sums, max_logits, temporary_output);
            torch::cuda::synchronize();
            TORCH_CHECK(torch::equal(output, torch::ones_like(output)),
                "AOT padding cache regression: incorrect output for D", head_size,
                ", page ", page_size, ", dtype BFloat16");
        }
    } catch (...) {
        warmup_mod.attr("warmup_pa_decode") = original_warmup;
        throw;
    }
    warmup_mod.attr("warmup_pa_decode") = original_warmup;
    TORCH_CHECK(warmup_heads == std::vector<int>({256, 192}),
        "Expected separate D256/D192 warmups followed by a D256 cache hit");
    std::cout << "[PASS] D256 -> D192 -> D256, page " << page_size
              << ", BFloat16: two warmups, exact outputs"
              << std::endl;
}

int main() {
    py::scoped_interpreter guard{};
    py::module_::import("sys").attr("path").cast<py::list>().insert(
        0, TEST_REPO_ROOT);

    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA/HIP not available." << std::endl;
        return 1;
    }

    int cdna_ver = aiter::get_cdna_version();
    if (cdna_ver < 0) {
        std::cerr << "Unsupported GPU architecture." << std::endl;
        return 1;
    }

    const int num_seqs                 = 2;
    const int query_length             = 3;
    const int num_query_heads          = 8;
    const int num_kv_heads             = 1;
    const int head_size                = 128;
    const int query_group_size         = num_query_heads / num_kv_heads;
    const int kv_block_size            = 64;
    const int max_num_blocks_per_seq   = 4;
    const int num_blocks               = num_seqs * max_num_blocks_per_seq;
    const int max_context_partition_num = 4;
    const int context_partition_size   = 256;
    const int kv_elem_per_vec          = 16 / static_cast<int>(
                                             torch::elementSize(torch::kBFloat16));
    const int eq_query_group_size      = query_length * query_group_size;

    torch::manual_seed(42);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16)
                         .device(torch::kCUDA);
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32)
                         .device(torch::kCUDA);
    auto opts_i32  = torch::TensorOptions().dtype(torch::kInt32)
                         .device(torch::kCUDA);

    auto query = torch::randn(
        {num_seqs * query_length, num_query_heads, head_size}, opts_bf16);
    auto key_cache = torch::randn(
        {num_blocks, num_kv_heads,
         head_size / kv_elem_per_vec, kv_block_size, kv_elem_per_vec},
        opts_bf16);
    auto value_cache = torch::randn(
        {num_blocks, num_kv_heads, head_size, kv_block_size}, opts_bf16);

    auto output = torch::zeros(
        {num_seqs * query_length, num_query_heads, head_size}, opts_bf16);

    auto context_lengths = torch::full({num_seqs}, 64, opts_i32);

    auto block_tables = torch::arange(
        num_seqs * max_num_blocks_per_seq, opts_i32)
        .reshape({num_seqs, max_num_blocks_per_seq});

    auto exp_sums = torch::zeros(
        {num_seqs, num_kv_heads, max_context_partition_num,
         eq_query_group_size}, opts_f32);
    auto max_logits = torch::full(
        {num_seqs, num_kv_heads, max_context_partition_num,
         eq_query_group_size},
        -std::numeric_limits<float>::infinity(), opts_f32);
    auto temporary_output = torch::zeros(
        {num_seqs, num_kv_heads, max_context_partition_num,
         eq_query_group_size, head_size}, opts_bf16);

    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_size));

    std::cout << "--- Cold-path call (warmup compile) ---" << std::endl;
    try {
        aiter::pa_decode_gluon_aot(
            output, query, key_cache, value_cache,
            context_lengths, block_tables,
            softmax_scale,
            query_length,
            max_context_partition_num,
            context_partition_size,
            at::ScalarType::BFloat16,
            {}, {}, {},
            exp_sums, max_logits, temporary_output,
            {}, nullptr);

        torch::cuda::synchronize();
        auto cold_sum = output.to(torch::kFloat32).sum().item<float>();
        std::cout << "Cold-path output sum = " << cold_sum << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Cold-path error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "--- Hot-path call (cache hit) ---" << std::endl;
    try {
        output.zero_();
        exp_sums.zero_();
        max_logits.fill_(-std::numeric_limits<float>::infinity());
        temporary_output.zero_();

        aiter::pa_decode_gluon_aot(
            output, query, key_cache, value_cache,
            context_lengths, block_tables,
            softmax_scale,
            query_length,
            max_context_partition_num,
            context_partition_size,
            at::ScalarType::BFloat16,
            {}, {}, {},
            exp_sums, max_logits, temporary_output,
            {}, nullptr);

        torch::cuda::synchronize();
        auto hot_sum = output.to(torch::kFloat32).sum().item<float>();
        std::cout << "Hot-path  output sum = " << hot_sum << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Hot-path error: " << e.what() << std::endl;
        return 1;
    }

    try {
        test_head_padding_cache_reuse();
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Padding cache regression: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "All tests passed!" << std::endl;

    // Use _exit() to avoid double-free from conflicting destruction order
    // between pybind11's scoped_interpreter and HIP/CUDA runtime static cleanup.
    _exit(0);
}
