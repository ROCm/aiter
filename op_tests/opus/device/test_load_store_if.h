// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_LOAD_STORE_IF_H
#define OP_TESTS_OPUS_DEVICE_TEST_LOAD_STORE_IF_H

#ifdef __cplusplus
extern "C" {
#endif

// Predicated copy using gmem::load_if / store_if (via free function wrappers).
// Copies src[0..n-1] to dst, skipping out-of-bounds accesses.
void run_predicated_copy(const void* d_src, void* d_dst, int n);

// Vector add using opus::load / opus::store free function wrappers.
// Result = A + B, element-wise.
void run_free_func_add(const void* d_a, const void* d_b, void* d_result, int n);

// Predicated async load using gmem::async_load_if (via free function wrapper).
// Copies src[0..n-1] to dst via LDS, zeros for out-of-bounds.
// n_padded must be a multiple of BLOCK_SIZE (256).
void run_predicated_async_load(const void* d_src, void* d_dst, int n, int n_padded);

#ifdef __cplusplus
}
#endif

#endif
