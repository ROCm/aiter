// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_WORKGROUP_BARRIER_H
#define OP_TESTS_OPUS_DEVICE_TEST_WORKGROUP_BARRIER_H

#ifdef __cplusplus
extern "C" {
#endif

// Test 1: wait_lt + inc (cumulative barrier)
//   Launch N workgroups. Workgroup i waits until sem >= i, atomically adds (i+1)
//   to accumulator, then increments sem. Host verifies accumulator == N*(N+1)/2.
void run_workgroup_barrier_cumulative(void* d_accumulator, int n_workgroups);

// Test 2: stream-K style reduce
//   Launch N+1 workgroups with block_size=256. Input has 256*N float elements.
//   Workgroups 0..N-1 each reduce a contiguous 256-element chunk into workspace[i],
//   then inc() a semaphore. Workgroup N waits via wait_eq(N), sums workspace[0..N-1],
//   and writes the final scalar to result. Host verifies result == sum(input).
void run_workgroup_barrier_streamk_reduce(
    const void* d_input,
    void* d_workspace,
    void* d_result,
    int n_chunks);

#ifdef __cplusplus
}
#endif

#endif
