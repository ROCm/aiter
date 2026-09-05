#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Internal launcher surface for the a16w4 GEMM kernels.
//
// Each variant lives in its OWN translation unit. That is not a style choice:
// the four kernel headers under gfx1201/ define BM / BN / BK / THREADS /
// PIPE_STEP and friends to different values, and although each header undefs
// them on the way out, keeping one kernel per TU also keeps the four ninja
// jobs parallel and stops a change to one tile from rebuilding the others.
//
// aiter::gemm_a16w4() in gemm_a16w4.cu is the only caller.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace aiter {
namespace a16w4 {

// int4 quantisation group along K. Fixed by the checkpoint format, not a
// tuning knob: every kernel static_asserts that GROUP_SIZE % BK == 0 so that
// one K-step never straddles two groups.
constexpr int kGroupSize = 128;
// int4 nibbles packed per uint32 of the weight tensor.
constexpr int kPackK = 8;

// Shape predicates. On failure `why` is set to a static string explaining
// which constraint was violated; it is never freed. Each predicate is derived
// from the constexpr tile geometry of its own kernel, so it cannot drift out
// of sync with the kernel it guards.
bool prefill_bf16_supported(int M, int N, int K, const char** why);
bool prefill_fp16_supported(int M, int N, int K, const char** why);
bool decode_bf16_supported(int M, int N, int K, const char** why);
bool decode_fp16_supported(int M, int N, int K, const char** why);

// fp32 workspace elements the decode path needs: SPLIT_K * M * N.
int64_t decode_bf16_workspace_elems(int64_t M, int64_t N);
int64_t decode_fp16_workspace_elems(int64_t M, int64_t N);

// A/scales/zeros/C are bf16 or fp16 raw bit patterns (unsigned short), W_q is
// packed int4. void* rather than the raw typedefs so this header does not
// have to pull in the device-side common header.
void launch_prefill_bf16(const void* A,
                         const void* W_q,
                         const void* scales,
                         const void* zeros,
                         void* C,
                         int M,
                         int N,
                         int K,
                         hipStream_t stream);

void launch_prefill_fp16(const void* A,
                         const void* W_q,
                         const void* scales,
                         const void* zeros_biased,
                         void* C,
                         int M,
                         int N,
                         int K,
                         hipStream_t stream);

void launch_decode_bf16(const void* A,
                        const void* W_q,
                        const void* scales,
                        const void* zeros,
                        float* workspace,
                        void* C,
                        int M,
                        int N,
                        int K,
                        hipStream_t stream);

void launch_decode_fp16(const void* A,
                        const void* W_q,
                        const void* scales,
                        const void* zeros_biased,
                        float* workspace,
                        void* C,
                        int M,
                        int N,
                        int K,
                        hipStream_t stream);

} // namespace a16w4
} // namespace aiter
