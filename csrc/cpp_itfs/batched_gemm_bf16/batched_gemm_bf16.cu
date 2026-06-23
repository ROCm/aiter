// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free batched bf16 GEMM dispatch.
//
// Heuristic-only shape dispatch over a fixed set of CK kernel instances -- the
// same instances batched_heuristic_dispatch() selects in the upstream torch
// build (csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu). The tuned CSV lookup
// table is intentionally omitted: it is a performance optimization, not a
// correctness requirement, and keeping it out makes this a self-contained,
// torch-free proof of concept with no codegen step.
#include <climits>
#include <cstdio>
#include <exception>

#include "batched_gemm_bf16_ck.cuh"

#ifdef USE_ROCM

namespace {

using BatchedKernel = void (*)(const batched_gemm_bf16_args&);

// Each instance's template parameters are copied verbatim from the matching
// entry in csrc/ck_batched_gemm_bf16/batched_gemm_bf16_common.py (the kernel
// names below encode those same parameters).
#define DEFINE_BGEMM_INSTANCE(fn, ...)              \
    void fn(const batched_gemm_bf16_args& a)        \
    {                                               \
        using Inst = DeviceGemmHelper<__VA_ARGS__>; \
        batched_gemm_bf16_impl<Inst>(a);            \
    }

// clang-format off
DEFINE_BGEMM_INSTANCE(bgemm_64x16x16x64_interwave_v2,
    64, 16, 16, 64, 16, 16, 1, 1, S<8, 8, 1>, S<8, 8, 1>, S<1, 16, 1, 4>, S<4, 4, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v2)

DEFINE_BGEMM_INSTANCE(bgemm_128x16x32x64_intrawave_v2,
    128, 16, 32, 64, 16, 16, 1, 1, S<8, 16, 1>, S<8, 16, 1>, S<1, 16, 1, 8>, S<4, 4, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2)

DEFINE_BGEMM_INSTANCE(bgemm_128x32x16x64_interwave_v2,
    128, 32, 16, 64, 16, 16, 1, 1, S<8, 16, 1>, S<8, 16, 1>, S<1, 16, 1, 8>, S<2, 2, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v2)

DEFINE_BGEMM_INSTANCE(bgemm_64x16x16x128_intrawave_v1,
    64, 16, 16, 128, 16, 16, 1, 1, S<16, 4, 1>, S<16, 4, 1>, S<1, 16, 1, 4>, S<4, 4, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1)

DEFINE_BGEMM_INSTANCE(bgemm_256x128x128x64_interwave_v1,
    256, 128, 128, 64, 32, 32, 2, 2, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1)

DEFINE_BGEMM_INSTANCE(bgemm_256x128x128x64_intrawave_v3,
    256, 128, 128, 64, 32, 32, 2, 2, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3)

DEFINE_BGEMM_INSTANCE(bgemm_256x256x128x32_interwave_v1,
    256, 256, 128, 32, 32, 32, 4, 2, S<4, 64, 1>, S<4, 64, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 1,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1)

DEFINE_BGEMM_INSTANCE(bgemm_256x224x256x32_intrawave_v3,
    256, 224, 256, 32, 16, 16, 7, 8, S<8, 32, 1>, S<8, 32, 1>, S<1, 32, 1, 8>, S<8, 8, 1>, 1, 2,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3)
// clang-format on

#undef DEFINE_BGEMM_INSTANCE

// Mirror of batched_heuristic_dispatch() from the upstream torch build. B is
// not consulted by the heuristic (it only depends on M, N, K).
BatchedKernel batched_heuristic_dispatch(int M, int N, int K)
{
    if(M < 64 && N < 2048 && K < 2048)
    {
        return bgemm_64x16x16x64_interwave_v2;
    }
    else if(M < 64 && K < 2048)
    {
        return bgemm_128x16x32x64_intrawave_v2;
    }
    else if(M < 64 && N < 2048)
    {
        return bgemm_128x32x16x64_interwave_v2;
    }
    else if(M < 64 && N > 2048 && K > 2048)
    {
        return bgemm_64x16x16x128_intrawave_v1;
    }
    else if(M < 64)
    {
        return bgemm_64x16x16x64_interwave_v2;
    }
    else if(K < 1024)
    {
        return bgemm_256x128x128x64_interwave_v1;
    }
    else if(M < 1024)
    {
        return bgemm_256x128x128x64_intrawave_v3;
    }
    else if(M >= 1024 && N >= 1024 && K >= 1024)
    {
        return bgemm_256x256x128x32_interwave_v1;
    }
    else
    {
        return bgemm_256x224x256x32_intrawave_v3;
    }
}

} // namespace

extern "C" __attribute__((visibility("default"))) int
aiter_batched_gemm_bf16(const batched_gemm_bf16_args* args)
{
    try
    {
        AITER_CHECK(args != nullptr, "aiter_batched_gemm_bf16: args is null");
        batched_heuristic_dispatch(args->M, args->N, args->K)(*args);
        return 0;
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "aiter_batched_gemm_bf16 error: %s\n", e.what());
        return 1;
    }
    catch(...)
    {
        fprintf(stderr, "aiter_batched_gemm_bf16 error: unknown\n");
        return 2;
    }
}

#endif // USE_ROCM
