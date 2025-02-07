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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "hip_compat.h"
#include "dispatch_utils.h"
#include <torch/torch.h>

#ifdef USE_ROCM
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 nv_bfloat16;
#else
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

namespace vllm
{
  template <typename T, typename Operation>
  inline __device__ T performOperation(T a, T b);

  template <typename Operation>
  torch::Tensor aten_compute(torch::Tensor &input, torch::Tensor &other);

  struct AddOp
  {
    template <typename T>
    inline __device__ static T apply(T a, T b) { return a + b; }

    void static print(
        int M, int N, int K, int in_stride0, int in_stride1, int in_stride2,
        int o_stride0, int o_stride1, int o_stride2)
    {
      printf("AddOp input shape: [%d %d %d], stride0: [%d %d %d], stride1: [%d %d %d]\n",
             M, N, K, in_stride0, in_stride1, in_stride2, o_stride0, o_stride1, o_stride2);
    }

    static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
    {
      return torch::add(input, other);
    }
  };

  struct SubOp
  {
    template <typename T>
    inline __device__ static T apply(T a, T b)
    {
      return a - b;
    }

    void static print(
        int M, int N, int K, int in_stride0, int in_stride1, int in_stride2,
        int o_stride0, int o_stride1, int o_stride2)
    {
      printf("SubOp input shape: [%d %d %d], stride0: [%d %d %d], stride1: [%d %d %d]\n",
             M, N, K, in_stride0, in_stride1, in_stride2, o_stride0, o_stride1, o_stride2);
    }

    static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
    {
      return torch::sub(input, other);
    }
  };

  struct MulOp
  {
    template <typename T>
    inline __device__ static T apply(T a, T b) { return a * b; }

    void static print(
        int M, int N, int K, int in_stride0, int in_stride1, int in_stride2,
        int o_stride0, int o_stride1, int o_stride2)
    {
      printf("MulOp input shape: [%d %d %d], stride0: [%d %d %d], stride1: [%d %d %d]\n",
             M, N, K, in_stride0, in_stride1, in_stride2, o_stride0, o_stride1, o_stride2);
    }

    static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
    {
      return torch::mul(input, other);
    }
  };

  struct DivOp
  {
    template <typename T>
    inline __device__ static T apply(T a, T b)
    {
      // assert(b == static_cast<T>(0));
      return a / b;
    }

    void static print(
        int M, int N, int K, int in_stride0, int in_stride1, int in_stride2,
        int o_stride0, int o_stride1, int o_stride2)
    {
      printf("DivOp input shape: [%d %d %d], stride0: [%d %d %d], stride1: [%d %d %d]\n",
             M, N, K, in_stride0, in_stride1, in_stride2, o_stride0, o_stride1, o_stride2);
    }

    static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
    {
      return torch::div(input, other);
    }
  };

  template <typename T, typename Operation, bool order_flag>
  inline __device__ T performOperation(T a, T b)
  {
    if constexpr (std::is_same_v<Operation, AddOp>)
    {
      return Operation::apply(a, b);
    }
    else if constexpr (std::is_same_v<Operation, SubOp>)
    {
      if constexpr (!order_flag)
      {
        return Operation::apply(b, a);
      }
      else
      {
        return Operation::apply(a, b);
      }
    }
    else if constexpr (std::is_same_v<Operation, MulOp>)
    {
      return Operation::apply(a, b);
    }
    else if constexpr (std::is_same_v<Operation, DivOp>)
    {
      if constexpr (!order_flag)
      {
        return Operation::apply(b, a);
      }
      else
      {
        return Operation::apply(a, b);
      }
    }
    else
    {
      static_assert(false, "Unsupported operation");
    }
  }

  template <typename Operation>
  torch::Tensor aten_compute(torch::Tensor &input, torch::Tensor &other)
  {
    if constexpr (std::is_same_v<Operation, AddOp>)
    {
      return Operation::compute(input, other);
    }
    else if constexpr (std::is_same_v<Operation, SubOp>)
    {
      return Operation::compute(input, other);
    }
    else if constexpr (std::is_same_v<Operation, MulOp>)
    {
      return Operation::compute(input, other);
    }
    else if constexpr (std::is_same_v<Operation, DivOp>)
    {
      return Operation::compute(input, other);
    }
    else
    {
      static_assert(false, "Unsupported operation");
    }
  }

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag>
  __global__ void operator_tn_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K, int stride0, int stride2)
  {
    // pad LDS row by dword
    constexpr uint32_t LDS_PAD = 4 / sizeof(_T);
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;

    union BLOCK_16B
    {
      _T e[elements_in_16B];
      __uint128_t ow;
    };

    // Round up processing to next full tile
    const uint32_t n_tiles = (N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N;
    const uint32_t k_tiles = (K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K;
    const uint32_t nk_tiles = n_tiles * k_tiles;
    const uint32_t m_tiles = gridDim.x / nk_tiles;
    const uint32_t m_tile_swizzle = blockIdx.x / nk_tiles / M_SWIZZLE * M_SWIZZLE;
    /// do m_swizzle when there are enough m_tiles
    const bool swizzle_m = m_tile_swizzle + M_SWIZZLE <= m_tiles;
    const uint32_t current_m = swizzle_m ? m_tile_swizzle + blockIdx.x % M_SWIZZLE : blockIdx.x / nk_tiles;

    const uint64_t stride_k = N * sizeof(_T);
    const uint64_t out_stride_nk = N * K * sizeof(_T);

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    __shared__ _T sa[BIG_TILE_SIZE_N][BIG_TILE_SIZE_K + LDS_PAD];

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;
    // use 128bit load&store whenever possible
    if (current_n_size % 8 == 0 && current_k_size % 8 == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes = BIG_TILE_SIZE_K * sizeof(_T);
      constexpr uint32_t ld_per_row = row_bytes / sizeof(__uint128_t);
      constexpr uint32_t rows_per_wg = _WG / ld_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE_N / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const uint8_t *pat = (const uint8_t *)a + tj * row_bytes + ti * BIG_TILE_SIZE_N * stride2 + current_m * stride0;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % ld_per_row;
        uint32_t row = threadIdx.x / ld_per_row + t * rows_per_wg;
        uint64_t offset = (col * 8 < current_k_size && row < current_n_size) ? row * stride2 + col * sizeof(__uint128_t) : 0;
        const __uint128_t *pfa = (const __uint128_t *)(pat + offset);
        BLOCK_16B d;
        d.ow = *pfa;
#pragma unroll
        for (uint32_t i = 0; i < elements_in_16B; i++)
        {
          sa[row][col * elements_in_16B + i] = d.e[i];
        }
      }
      __syncthreads();
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N * sizeof(_T);
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / sizeof(__uint128_t);
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const uint8_t *pb = (const uint8_t *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const uint8_t *pc = (const uint8_t *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * 8 < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
          BLOCK_16B d;
          const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
          d.ow = *pfb;
// Transpose tile on read from LDS
#pragma unroll
          for (uint32_t i = 0; i < elements_in_16B; i++)
          {
            d.e[i] = performOperation<_T, Operation, order_flag>(sa[col * elements_in_16B + i][row], d.e[i]);
          }
          __uint128_t *pfc = (__uint128_t *)(pc + offset);
          *pfc = d.ow;
        }
      }
    }
    else
    {
      // Copy partial tiles with element accesses
      constexpr uint32_t row_bytes = BIG_TILE_SIZE_K * sizeof(_T);
      constexpr uint32_t ld_per_row = BIG_TILE_SIZE_K;
      constexpr uint32_t rows_per_wg = _WG / ld_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE_N / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const uint8_t *pat = (const uint8_t *)a + ti * BIG_TILE_SIZE_N * stride2 + tj * row_bytes + current_m * stride0;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % ld_per_row;
        uint32_t row = threadIdx.x / ld_per_row + t * rows_per_wg;
        uint64_t offset = (col < current_k_size && row < current_n_size) ? row * stride2 + col * sizeof(_T) : 0;
        const _T *pfa = (const _T *)(pat + offset);
        sa[row][col] = *pfa;
      }
      __syncthreads();

      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N * sizeof(_T);
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const uint8_t *pb = (const uint8_t *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const uint8_t *pc = (const uint8_t *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col * sizeof(_T);
          const _T *pfb = (const _T *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          *pfc = performOperation<_T, Operation, order_flag>(sa[col][row], *pfb);
        }
      }
    }
  }

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag>
  __global__ void operator_bcast_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K)
  {
    // pad LDS row by dword
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;

    union BLOCK_16B
    {
      _T e[elements_in_16B];
      __uint128_t ow;
    };

    // Round up processing to next full tile
    const uint32_t n_tiles = (N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N;
    const uint32_t k_tiles = (K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K;
    const uint32_t nk_tiles = n_tiles * k_tiles;
    const uint32_t m_tiles = gridDim.x / nk_tiles;
    const uint32_t m_tile_swizzle = blockIdx.x / nk_tiles / M_SWIZZLE * M_SWIZZLE;
    /// do m_swizzle when there are enough m_tiles
    const bool swizzle_m = m_tile_swizzle + M_SWIZZLE <= m_tiles;
    const uint32_t current_m = swizzle_m ? m_tile_swizzle + blockIdx.x % M_SWIZZLE : blockIdx.x / nk_tiles;

    const uint64_t stride_k = N * sizeof(_T);
    const uint64_t out_stride_nk = N * K * sizeof(_T);

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;

    // use 128bit load&store whenever possible
    if (current_n_size % 8 == 0 && current_k_size % 8 == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N * sizeof(_T);
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / sizeof(__uint128_t);
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const uint8_t *pa = (const uint8_t *)a + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const uint8_t *pb = (const uint8_t *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr;
      const uint8_t *pc = (const uint8_t *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * 8 < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
          BLOCK_16B d, f;
          const __uint128_t *pfa = (const __uint128_t *)(pa + offset);
          const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
          f.ow = *pfa;
          d.ow = *pfb;
// Transpose tile on read from LDS
#pragma unroll
          for (uint32_t i = 0; i < elements_in_16B; i++)
          {
            d.e[i] = performOperation<_T, Operation, order_flag>(f.e[i], d.e[i]);
          }
          __uint128_t *pfc = (__uint128_t *)(pc + offset);
          *pfc = d.ow;
        }
      }
    }
    else
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N * sizeof(_T);
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const uint8_t *pa = (const uint8_t *)a + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const uint8_t *pb = (const uint8_t *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr;
      const uint8_t *pc = (const uint8_t *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col * sizeof(_T);
          const _T *pfa = (const _T *)(pa + offset);
          const _T *pfb = (const _T *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          *pfc = performOperation<_T, Operation, order_flag>(*pfa, *pfb);
        }
      }
    }
  }

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag>
  __global__ void operator_bcast1_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K)
  {
    // pad LDS row by dword
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;

    union BLOCK_16B
    {
      _T e[elements_in_16B];
      __uint128_t ow;
    };

    // Round up processing to next full tile
    const uint32_t n_tiles = (N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N;
    const uint32_t k_tiles = (K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K;
    const uint32_t nk_tiles = n_tiles * k_tiles;
    const uint32_t m_tiles = gridDim.x / nk_tiles;
    const uint32_t m_tile_swizzle = blockIdx.x / nk_tiles / M_SWIZZLE * M_SWIZZLE;
    /// do m_swizzle when there are enough m_tiles
    const bool swizzle_m = m_tile_swizzle + M_SWIZZLE <= m_tiles;
    const uint32_t current_m = swizzle_m ? m_tile_swizzle + blockIdx.x % M_SWIZZLE : blockIdx.x / nk_tiles;

    const uint64_t stride_k = N * sizeof(_T);
    const uint64_t out_stride_nk = N * K * sizeof(_T);

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;

    // use 128bit load&store whenever possible
    if (current_n_size % 8 == 0 && current_k_size % 8 == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N * sizeof(_T);
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / sizeof(__uint128_t);
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const uint8_t *pa = (const uint8_t *)a + ti * row_bytes_wr + current_m * stride_k;
      const uint8_t *pb = (const uint8_t *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const uint8_t *pc = (const uint8_t *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * 8 < current_n_size && row < current_k_size)
        {
          uint64_t offset_a = col * sizeof(__uint128_t);
          uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
          BLOCK_16B d, f;
          const __uint128_t *pfa = (const __uint128_t *)(pa + offset_a);
          const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
          f.ow = *pfa;
          d.ow = *pfb;
// Transpose tile on read from LDS
#pragma unroll
          for (uint32_t i = 0; i < elements_in_16B; i++)
          {
            d.e[i] = performOperation<_T, Operation, order_flag>(f.e[i], d.e[i]);
          }
          __uint128_t *pfc = (__uint128_t *)(pc + offset);
          *pfc = d.ow;
        }
      }
    }
    else
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N * sizeof(_T);
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const uint8_t *pa = (const uint8_t *)a + ti * row_bytes_wr + current_m * stride_k;
      const uint8_t *pb = (const uint8_t *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const uint8_t *pc = (const uint8_t *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset_a = col * sizeof(_T);
          uint64_t offset = row * stride_k + col * sizeof(_T);
          const _T *pfa = (const _T *)(pa + offset_a);
          const _T *pfb = (const _T *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);

          *pfc = performOperation<_T, Operation, order_flag>(*pfa, *pfb);
        }
      }
    }
  }

  template <class _T, int _rows, int _vec, typename Operation, bool order_flag>
  __global__ void operator_bcast_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int M, const int N, const int K)
  {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_tiles = N / _rows;
    uint32_t k_tiles = K / _vec;
    if (idx < (uint64_t)M * n_tiles * k_tiles)
    {
      uint32_t ti = idx / (k_tiles * n_tiles);
      uint64_t idx_block = idx % (k_tiles * n_tiles);
      uint32_t tj = (idx_block / k_tiles) % n_tiles;
      uint32_t tk = idx_block % k_tiles;
      for (int row = 0; row < _rows; row++)
      {
        uint64_t offset_b = (uint64_t)(tj + row * n_tiles) * K + tk * _vec;
        uint64_t offset_ac = (uint64_t)(tj + row * n_tiles) * K + tk * _vec + (uint64_t)ti * N * K;
        const _T *pa = (const _T *)a + offset_ac;
        const _T *pb = (const _T *)b + offset_b;
        _T *pc = (_T *)c + offset_ac;
        for (int col = 0; col < _vec; col++)
        {
          const _T *pfa = (const _T *)(pa + col);
          const _T *pfb = (const _T *)(pb + col);
          _T *pfc = (_T *)(pc + col);
          *pfc = performOperation<_T, Operation, order_flag>(*pfa, *pfb);
        }
      }
    }
  }
}

template <typename Operation>
torch::Tensor aiter_operation(torch::Tensor &input, torch::Tensor &other)
{
  int dim = input.dim();
  bool is_same_dtype = input.dtype() == other.dtype();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  if (dim == 2)
  {
    bool is_support = true;
    is_support &= is_same_dtype == true;
    is_support &= input.dim() == other.dim();
    is_support &= input.size(0) == other.size(0);
    is_support &= input.size(1) == other.size(1);
    is_support &= input.size(1) > 512;
    is_support &= input.is_contiguous() == other.is_contiguous();
    is_support &= input.is_contiguous() == true;
    int M = 1;
    int N = input.size(0);
    int K = input.size(1);
    const uint32_t rows = 8;
    const uint32_t vec = 16 / sizeof(input.dtype());
    is_support &= N % rows == 0;
    is_support &= K % vec == 0;
    if (is_support)
    {
      auto options = torch::TensorOptions().dtype(input.dtype()).device("cuda");
      auto output = torch::empty({N, K}, options);
      void *buf_c = reinterpret_cast<void *>(output.data_ptr());

      void *buf_a = reinterpret_cast<void *>(input.data_ptr());
      void *buf_b = reinterpret_cast<void *>(other.data_ptr());
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      int elements = N * K;

      constexpr uint32_t wg = 256;
      int grid_x = (elements / (rows * vec) + wg - 1) / wg;
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);

      VLLM_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "operator_bcast_tile_kernel", [&]
          { vllm::operator_bcast_tile_kernel<scalar_t, rows, vec, Operation, true>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K); });
      return output;
    }
    else
    {
      return vllm::aten_compute<Operation>(input, other);
    }
  }
  else if (dim == 3)
  {
    constexpr uint32_t PATTERN_TRANSPOSE = 1;
    constexpr uint32_t PATTERN_BROADCAST_0 = 2;
    constexpr uint32_t PATTERN_BROADCAST_1 = 3;

    // if (dim == 3 && (!input.is_contiguous() || !other.is_contiguous()))
    // {
    //   Operation::print(input.size(0), input.size(1), input.size(2),
    //                    input.stride(0), input.stride(1), input.stride(2),
    //                    other.stride(0), other.stride(1), other.stride(2));
    // }
    auto tensor_not_conti = input.is_contiguous() ? other : input;
    bool order_flag;
    int M, N, K;
    int stride0, stride1, stride2;
    int pattern = 0;
    if (input.is_contiguous() != other.is_contiguous())
    {
      order_flag = !input.is_contiguous() ? true : false;
      bool is_support = true;
      M = input.size(0);
      N = input.size(1);
      K = input.size(2);
      // avoid broadcast
      is_support &= is_same_dtype == true;
      is_support &= input.dim() == other.dim();
      is_support &= dim == 3;
      is_support &= input.size(0) == other.size(0);
      is_support &= input.size(1) == other.size(1);
      is_support &= input.size(2) == other.size(2);
      stride0 = tensor_not_conti.stride(0);
      stride1 = tensor_not_conti.stride(1);
      stride2 = tensor_not_conti.stride(2);
      is_support &= stride1 == 1;
      stride0 *= input.element_size();
      stride1 *= input.element_size();
      stride2 *= input.element_size();
      pattern = is_support ? PATTERN_TRANSPOSE : 0;
    }
    else if (input.is_contiguous() && other.is_contiguous())
    {
      bool is_support = false;
      if (!is_support && other.size(0) == 1)
      {
        is_support = true;
        is_support &= is_same_dtype == true;
        is_support &= input.dim() == other.dim();
        is_support &= input.size(0) > 1;
        is_support &= other.size(0) == 1;
        is_support &= input.size(1) == other.size(1);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_0 : 0;
        order_flag = true;
      }

      if (!is_support && input.size(0) == 1)
      {
        is_support = true;
        is_support &= is_same_dtype == true;
        is_support &= input.dim() == other.dim();
        is_support &= other.size(0) > 1;
        is_support &= input.size(0) == 1;
        is_support &= input.size(1) == other.size(1);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_0 : 0;
        order_flag = false;
      }

      if (!is_support && input.size(1) == 1)
      {
        is_support = true;
        is_support &= is_same_dtype == true;
        is_support &= input.dim() == other.dim();
        is_support &= other.size(1) > 1;
        is_support &= input.size(0) == other.size(0);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_1 : 0;
        order_flag = true;
      }

      if (!is_support && other.size(1) == 1)
      {
        is_support = true;
        is_support &= is_same_dtype == true;
        is_support &= input.dim() == other.dim();
        is_support &= input.size(1) > 1;
        is_support &= input.size(0) == other.size(0);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_1 : 0;
        order_flag = false;
      }
    }

    if (pattern == PATTERN_TRANSPOSE)
    {
      constexpr uint32_t BIG_TILE_SIZE_N = 64;
      constexpr uint32_t BIG_TILE_SIZE_K = 64;
      constexpr uint32_t M_SWIZZLE = 8;
      const int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(256, 1, 1);

      auto options =
          torch::TensorOptions().dtype(input.dtype()).device("cuda");
      auto output =
          torch::empty(input.sizes(), options);
      void *buf_c = reinterpret_cast<void *>(output.data_ptr());

      void *buf_a = reinterpret_cast<void *>(input.data_ptr());
      void *buf_b = reinterpret_cast<void *>(other.data_ptr());

      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "operator_tn_big_tile_kernel", [&]
            { vllm::operator_tn_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, stride0, stride2); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "operator_tn_big_tile_kernel", [&]
            { vllm::operator_tn_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N, stride0, stride2); });
      }

      return output;
    }
    else if (pattern == PATTERN_BROADCAST_0)
    {
      M = order_flag ? input.size(0) : other.size(0);
      N = order_flag ? input.size(1) : other.size(1);
      K = order_flag ? input.size(2) : other.size(2);
      auto options = torch::TensorOptions().dtype(input.dtype()).device("cuda");
      auto output = torch::empty({M, N, K}, options);
      void *buf_c = reinterpret_cast<void *>(output.data_ptr());

      void *buf_a = reinterpret_cast<void *>(input.data_ptr());
      void *buf_b = reinterpret_cast<void *>(other.data_ptr());
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      int elements = M * N * K;
      const uint32_t rows = 8;
      const uint32_t vec = 16 / sizeof(input.dtype());;
      if (N % rows == 0 && K % vec == 0)
      {
        constexpr uint32_t wg = 64;
        int grid_x = (elements / (rows * vec) + wg - 1) / wg;
        const dim3 grid_dim(grid_x, 1, 1);
        const dim3 block_dim(wg, 1, 1);

        if (order_flag)
        {
          VLLM_DISPATCH_FLOATING_TYPES(
              input.scalar_type(), "operator_bcast_tile_kernel", [&]
              { vllm::operator_bcast_tile_kernel<scalar_t, rows, vec, Operation, true>
                    <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K); });
        }
        else
        {
          VLLM_DISPATCH_FLOATING_TYPES(
              input.scalar_type(), "operator_bcast_tile_kernel", [&]
              { vllm::operator_bcast_tile_kernel<scalar_t, rows, vec, Operation, false>
                    <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, M, N, K); });
        }
      }
      else
      {
        constexpr uint32_t BIG_TILE_SIZE_N = 64;
        constexpr uint32_t BIG_TILE_SIZE_K = 64;
        constexpr uint32_t M_SWIZZLE = 8;
        const int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
        const dim3 grid_dim(grid_x, 1, 1);
        const dim3 block_dim(256, 1, 1);

        if (order_flag)
        {
          VLLM_DISPATCH_FLOATING_TYPES(
              input.scalar_type(), "operator_bcast_big_tile_kernel", [&]
              { vllm::operator_bcast_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                    <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N); });
        }
        else
        {
          VLLM_DISPATCH_FLOATING_TYPES(
              input.scalar_type(), "operator_bcast_big_tile_kernel", [&]
              { vllm::operator_bcast_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                    <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N); });
        }
      }

      return output;
    }
    else if (pattern == PATTERN_BROADCAST_1)
    {
      M = order_flag ? other.size(0) : input.size(0);
      N = order_flag ? other.size(1) : input.size(1);
      K = order_flag ? other.size(2) : input.size(2);
      constexpr uint32_t BIG_TILE_SIZE_N = 64;
      constexpr uint32_t BIG_TILE_SIZE_K = 64;
      constexpr uint32_t M_SWIZZLE = 8;
      const int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(256, 1, 1);

      auto options = torch::TensorOptions().dtype(input.dtype()).device("cuda");
      auto output = torch::empty({M, N, K}, options);
      void *buf_c = reinterpret_cast<void *>(output.data_ptr());

      void *buf_a = reinterpret_cast<void *>(input.data_ptr());
      void *buf_b = reinterpret_cast<void *>(other.data_ptr());

      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "operator_bcast1_big_tile_kernel", [&]
            { vllm::operator_bcast1_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "operator_bcast1_big_tile_kernel", [&]
            { vllm::operator_bcast1_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N); });
      }

      return output;
    }
    else
    {
      return vllm::aten_compute<Operation>(input, other);
    }
  }
}

torch::Tensor aiter_add(torch::Tensor &input, torch::Tensor &other)
{
  return aiter_operation<vllm::AddOp>(input, other);
}

torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other)
{
  return aiter_operation<vllm::SubOp>(input, other);
}

torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other)
{
  return aiter_operation<vllm::MulOp>(input, other);
}

torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other)
{
  return aiter_operation<vllm::DivOp>(input, other);
}
