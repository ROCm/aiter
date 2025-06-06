/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
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
#include <torch/all.h>
#include <c10/core/ScalarType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "py_itfs_common.h"

// declare templates for front (cpp) and back (cuda) sides of function:
// template <typename T>

// void LLGemm_Silu(void* in_a, void* in_b, void* out_c, const int M, const int
// K,
//                  cudaStream_t stream, const int rows_per_block);
// void LLMM_Silu(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c,
//                const int64_t rows_per_block) {
//   auto M = in_a.size(0);
//   auto K = in_a.size(1);
//   LLGemm_Silu(in_a.data_ptr(), in_b.data_ptr(), out_c.data_ptr(), M, K,
//               at::cuda::getCurrentCUDAStream(), rows_per_block);
// }

void LLGemm1(void* in_a,
             void* in_b,
             void* out_c,
             const int M,
             const int K,
             cudaStream_t stream,
             const int rows_per_block          = 4,
             const c10::ScalarType scalar_type = c10::ScalarType::Half);
// template <typename T>
void LLMM1(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c, const int64_t rows_per_block)
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
    TORCH_CHECK(N == 1, "Row number of activation tensor must be 1.");
    TORCH_CHECK(in_a.dtype() == in_b.dtype());
    TORCH_CHECK(in_b.dtype() == torch::kFloat16 || in_b.dtype() == torch::kBFloat16);

    // call the kernel function...
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
    LLGemm1(in_a.data_ptr(),
            in_b.data_ptr(),
            out_c.data_ptr(),
            M,
            K,
            at::cuda::getCurrentCUDAStream(),
            rows_per_block,
            in_b.scalar_type());
}

void wvSplitK_(void* in_a,
               void* in_b,
               void* out_c,
               const int M,
               const int K,
               const int N,
               cudaStream_t stream,
               const int CuCount                 = 1,
               const c10::ScalarType scalar_type = c10::ScalarType::Half);
void wvSpltK(at::Tensor& in_a,
             at::Tensor& in_b,
             at::Tensor& out_c,
             const int64_t N_in,
             const int64_t CuCount)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    int N  = N_in;
    TORCH_CHECK(in_a.dtype() == in_b.dtype());
    TORCH_CHECK(K % 8 == 0, "k % 8 == 0");
    TORCH_CHECK(in_a.dtype() == torch::kFloat16 || in_a.dtype() == torch::kBFloat16);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
    wvSplitK_(in_a.data_ptr(),
              in_b.data_ptr(),
              out_c.data_ptr(),
              M,
              K,
              N,
              at::cuda::getCurrentCUDAStream(),
              CuCount,
              in_b.scalar_type());
}

void wv_splitk_small_fp16_bf16(void* in_a,
                               void* in_b,
                               void* out_c,
                               const int M,
                               const int K,
                               const int N,
                               cudaStream_t stream,
                               const int CuCount                 = 1,
                               const c10::ScalarType scalar_type = c10::ScalarType::Half);
void wv_splitk_small_fp16_bf16_wrapper(at::Tensor& in_a,
                                       at::Tensor& in_b,
                                       at::Tensor& out_c,
                                       const int64_t N_in,
                                       const int64_t CuCount)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    int N  = N_in;
    TORCH_CHECK(in_a.dtype() == in_b.dtype());
    TORCH_CHECK(K % 8 == 0, "k % 8 == 0");
    TORCH_CHECK(in_a.dtype() == torch::kFloat16 || in_a.dtype() == torch::kBFloat16);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
    wv_splitk_small_fp16_bf16(in_a.data_ptr(),
                              in_b.data_ptr(),
                              out_c.data_ptr(),
                              M,
                              K,
                              N,
                              at::cuda::getCurrentCUDAStream(),
                              CuCount,
                              in_b.scalar_type());
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
                cudaStream_t stream,
                const int CuCount                   = 1,
                const c10::ScalarType a_scalar_type = c10::ScalarType::Float8_e4m3fnuz,
                const c10::ScalarType c_scalar_type = c10::ScalarType::Half);
void wvSplitKQ(at::Tensor& in_a,
               at::Tensor& in_b,
               at::Tensor& out_c,
               at::Tensor& scale_a,
               at::Tensor& scale_b,
               const int64_t CuCount)
{
    auto M  = in_a.size(0);
    auto K  = in_a.size(1);
    auto N  = in_b.size(0);
    auto Kp = in_a.stride(0);
    TORCH_CHECK(K % 16 == 0, "k % 16 == 0");
    TORCH_CHECK(in_a.dtype() == in_b.dtype() && in_a.dtype() == torch_fp8);
    TORCH_CHECK(out_c.dtype() == torch::kFloat16 || out_c.dtype() == torch::kBFloat16);
    auto scale_a_ptr = scale_a.data_ptr<float>();
    auto scale_b_ptr = scale_b.data_ptr<float>();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
    wvSplitKQ_(in_a.data_ptr(),
               in_b.data_ptr(),
               out_c.data_ptr(),
               scale_a_ptr,
               scale_b_ptr,
               M,
               K,
               Kp,
               N,
               at::cuda::getCurrentCUDAStream(),
               CuCount,
               in_a.scalar_type(),
               out_c.scalar_type());
}

void LLGemmZZ(void* in_a,
              void* in_b,
              void* out_c,
              const int M,
              const int K,
              cudaStream_t stream,
              const int solidx);

void LLZZ(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c, const int64_t solidx = 0)
{
    auto M = in_a.size(0);
    auto K = in_a.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
    LLGemmZZ(in_a.data_ptr(),
             in_b.data_ptr(),
             out_c.data_ptr(),
             M,
             K,
             at::cuda::getCurrentCUDAStream(),
             solidx);
}
// instantiate the CPP template for T=float:
// template void AddGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor
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
                 cudaStream_t stream);

void MMCustomGPU(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c)
{
    auto matA_sizes{in_a.sizes()};
    auto matB_sizes{in_b.sizes()};
    auto matO_sizes{out_c.sizes()};
    const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
    MMGPUKernel(in_a.data_ptr<float>(),
                in_b.data_ptr<float>(),
                out_c.data_ptr<float>(),
                matA_sizes[0],
                matA_sizes[1],
                matB_sizes[0],
                matB_sizes[1],
                matO_sizes[0],
                matO_sizes[1],
                at::cuda::getCurrentCUDAStream());
}
