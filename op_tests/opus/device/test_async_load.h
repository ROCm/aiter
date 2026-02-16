// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_ASYNC_LOAD_H
#define OP_TESTS_OPUS_DEVICE_TEST_ASYNC_LOAD_H

#ifdef __cplusplus
extern "C" {
#endif

// Copy src -> dst through LDS using opus gmem::async_load + s_waitcnt_vmcnt.
// Verifies the global-memory-to-LDS async path.
// d_src, d_dst are device pointers to float data; n is the number of elements.
// n should be a multiple of BLOCK_SIZE (256) for best results.
void run_async_load(
    const void* d_src,
    void* d_dst,
    int n);

#ifdef __cplusplus
}
#endif

#endif
