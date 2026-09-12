/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "custom.h"

namespace aiter {

// declare templates for front (cpp) and back (cuda) sides of function:
// template <typename T>

// void LLGemm_Silu(void* in_a, void* in_b, void* out_c, const int M, const int
// K,
//                  hipStream_t stream, const int rows_per_block);
// void LLMM_Silu(aiter_tensor_t& in_a, aiter_tensor_t& in_b, aiter_tensor_t& out_c,
//                const int64_t rows_per_block) {
//   auto M = in_a.size(0);
//   auto K = in_a.size(1);
//   LLGemm_Silu(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
//               aiter::getCurrentHIPStream(), rows_per_block);
// }

void LLGemm1(void* in_a,
             void* in_b,
             void* out_c,
             const int M,
             const int K,
             hipStream_t stream,
             const int rows_per_block          = 4,
             const AiterDtype scalar_type      = AITER_DTYPE_fp16);
// template <typename T>
void LLMM1(aiter_tensor_t& in_a, aiter_tensor_t& in_b, aiter_tensor_t& out_c, const int64_t rows_per_block)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    auto N = in_b.size(0);
    // if (N != in_b.numel())
    //         throw std::invalid_argument("Size mismatch A.numel(): " +
    //         std::to_string(in_a.numel())
    //                           + ", B.numel(): " +
    //                           std::to_string(in_b.numel()));

    // out_c.resize_({N});
    AITER_CHECK(N == 1, "Row number of activation tensor must be 1.");
    AITER_CHECK(in_a.dtype() == in_b.dtype());
    AITER_CHECK(in_b.dtype() == AITER_DTYPE_fp16 || in_b.dtype() == AITER_DTYPE_bf16);

    // call the kernel function...
    HipDeviceGuard device_guard(in_a.device_id);
    LLGemm1(in_a.data_ptr(),
            in_b.data_ptr(),
            out_c.data_ptr(),
            M,
            K,
            aiter::getCurrentHIPStream(),
            rows_per_block,
            in_b.dtype());
}

void wvSplitK_(void* in_a,
               void* in_b,
               void* out_c,
               const int M,
               const int K,
               const int N,
               hipStream_t stream,
               const int CuCount                 = 1,
               const AiterDtype scalar_type      = AITER_DTYPE_fp16);
void wvSpltK(aiter_tensor_t& in_a,
             aiter_tensor_t& in_b,
             aiter_tensor_t& out_c,
             const int64_t N_in,
             const int64_t CuCount)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    int N  = N_in;
    AITER_CHECK(in_a.dtype() == in_b.dtype());
    AITER_CHECK(K % 8 == 0, "k % 8 == 0");
    AITER_CHECK(in_a.dtype() == AITER_DTYPE_fp16 || in_a.dtype() == AITER_DTYPE_bf16);

    HipDeviceGuard device_guard(in_a.device_id);
    wvSplitK_(in_a.data_ptr(),
              in_b.data_ptr(),
              out_c.data_ptr(),
              M,
              K,
              N,
              aiter::getCurrentHIPStream(),
              CuCount,
              in_b.dtype());
}

void wv_splitk_small_fp16_bf16(void* in_a,
                               void* in_b,
                               void* out_c,
                               const int M,
                               const int K,
                               const int N,
                               hipStream_t stream,
                               const int CuCount                 = 1,
                               const AiterDtype scalar_type      = AITER_DTYPE_fp16);
void wv_splitk_small_fp16_bf16_wrapper(aiter_tensor_t& in_a,
                                       aiter_tensor_t& in_b,
                                       aiter_tensor_t& out_c,
                                       const int64_t N_in,
                                       const int64_t CuCount)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    int N  = N_in;
    AITER_CHECK(in_a.dtype() == in_b.dtype());
    AITER_CHECK(K % 8 == 0, "k % 8 == 0");
    AITER_CHECK(in_a.dtype() == AITER_DTYPE_fp16 || in_a.dtype() == AITER_DTYPE_bf16);

    HipDeviceGuard device_guard(in_a.device_id);
    wv_splitk_small_fp16_bf16(in_a.data_ptr(),
                              in_b.data_ptr(),
                              out_c.data_ptr(),
                              M,
                              K,
                              N,
                              aiter::getCurrentHIPStream(),
                              CuCount,
                              in_b.dtype());
}

void wvSplitKQ_(void* in_a,
                void* in_b,
                void* out_c,
                const float* scale_a,
                const float* scale_b,
                const int M,
                const int K,
                const int Kp,
                const int N,
                hipStream_t stream,
                const int CuCount                   = 1,
                const AiterDtype a_scalar_type       = AITER_DTYPE_fp8,
                const AiterDtype c_scalar_type       = AITER_DTYPE_fp16);
void wvSplitKQ(aiter_tensor_t& in_a,
               aiter_tensor_t& in_b,
               aiter_tensor_t& out_c,
               aiter_tensor_t& scale_a,
               aiter_tensor_t& scale_b,
               const int64_t CuCount)
{
    auto M  = in_a.size(0);
    auto K  = in_a.size(1);
    auto N  = in_b.size(0);
    auto Kp = in_a.stride(0);
    AITER_CHECK(K % 16 == 0, "k % 16 == 0");
    AITER_CHECK(in_a.dtype() == in_b.dtype() && in_a.dtype() == AITER_DTYPE_fp8);
    AITER_CHECK(out_c.dtype() == AITER_DTYPE_fp16 || out_c.dtype() == AITER_DTYPE_bf16);
    auto scale_a_ptr = reinterpret_cast<float*>(scale_a.data_ptr());
    auto scale_b_ptr = reinterpret_cast<float*>(scale_b.data_ptr());

    HipDeviceGuard device_guard(in_a.device_id);
    wvSplitKQ_(in_a.data_ptr(),
               in_b.data_ptr(),
               out_c.data_ptr(),
               scale_a_ptr,
               scale_b_ptr,
               M,
               K,
               Kp,
               N,
               aiter::getCurrentHIPStream(),
               CuCount,
               in_a.dtype(),
               out_c.dtype());
}

void LLGemmZZ(void* in_a,
              void* in_b,
              void* out_c,
              const int M,
              const int K,
              hipStream_t stream,
              const int solidx);

void LLZZ(aiter_tensor_t& in_a, aiter_tensor_t& in_b, aiter_tensor_t& out_c, const int64_t solidx = 0)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    HipDeviceGuard device_guard(in_a.device_id);
    LLGemmZZ(in_a.data_ptr(),
             in_b.data_ptr(),
             out_c.data_ptr(),
             M,
             K,
             aiter::getCurrentHIPStream(),
             solidx);
}
// instantiate the CPP template for T=float:
// template void AddGPU<float>(aiter_tensor_t in_a, aiter_tensor_t in_b, aiter_tensor_t
// out_c);

void MMGPUKernel(float* in_a,
                 float* in_b,
                 float* out_c,
                 int numARows,
                 int numAColumns,
                 int numBRows,
                 int numBColumns,
                 int numCRows,
                 int numCColumns,
                 hipStream_t stream);

void MMCustomGPU(aiter_tensor_t& in_a, aiter_tensor_t& in_b, aiter_tensor_t& out_c)
{
    auto numARows    = in_a.size(0);
    auto numAColumns = in_a.size(1);
    auto numBRows    = in_b.size(0);
    auto numBColumns = in_b.size(1);
    auto numCRows    = out_c.size(0);
    auto numCColumns = out_c.size(1);
    HipDeviceGuard device_guard(in_a.device_id);
    MMGPUKernel(reinterpret_cast<float*>(in_a.data_ptr()),
                reinterpret_cast<float*>(in_b.data_ptr()),
                reinterpret_cast<float*>(out_c.data_ptr()),
                numARows,
                numAColumns,
                numBRows,
                numBColumns,
                numCRows,
                numCColumns,
                aiter::getCurrentHIPStream());
}
} // namespace aiter
