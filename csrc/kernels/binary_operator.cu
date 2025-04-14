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

namespace aiter
{
  template <typename T, typename Operation>
  inline __device__ T performOperation(T a, T b);

  template <typename Operation>
  torch::Tensor aten_compute(torch::Tensor &input, torch::Tensor &other);

  struct AddOp
  {
    template <typename T>
    inline __device__ static T apply(T a, T b) { return a + b; }

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

    static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
    {
      return torch::sub(input, other);
    }
  };

  struct MulOp
  {
    template <typename T>
    inline __device__ static T apply(T a, T b) { return a * b; }

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

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_tn_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                              const int N, const int K, int stride0, int stride2, bool types_match)
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

    const uint64_t stride_k = N;
    const uint64_t out_stride_nk = N * K;

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    __shared__ _T0 sa[BIG_TILE_SIZE_N][BIG_TILE_SIZE_K + LDS_PAD];

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;
    // use 128bit load&store whenever possible
    if (current_n_size % elements_in_16B == 0 && current_k_size % 8 == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes = BIG_TILE_SIZE_K;
      constexpr uint32_t ld_per_row = row_bytes / elements_in_16B;
      constexpr uint32_t rows_per_wg = _WG / ld_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE_N / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const _T0 *pat = (const _T0 *)a + tj * row_bytes + ti * BIG_TILE_SIZE_N * stride2 + current_m * stride0;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % ld_per_row;
        uint32_t row = threadIdx.x / ld_per_row + t * rows_per_wg;
        uint64_t offset = (col * elements_in_16B < current_k_size && row < current_n_size) ? row * stride2 + col * elements_in_16B : 0;
        const _T0 *pfa = (const _T0 *)(pat + offset);
        // BLOCK_16B d;
        // d.ow = *pfa;
#pragma unroll
        for (uint32_t i = 0; i < elements_in_16B; i++)
        {
          sa[row][col * elements_in_16B + i] = pfa[i];
        }
      }
      __syncthreads();
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / elements_in_16B;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      _T *pc = (_T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * elements_in_16B < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col * elements_in_16B;
          BLOCK_16B d;
          if (types_match)
          {
            const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
            d.ow = *pfb;
// Transpose tile on read from LDS
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              d.e[i] = performOperation<_T, Operation, order_flag>(static_cast<_T>(sa[col * elements_in_16B + i][row]), d.e[i]);
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
          else
          {
            const _T1 *pfb = (const _T1 *)(pb + offset);
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              float a = static_cast<float>(sa[col * elements_in_16B + i][row]);
              float b = static_cast<float>(pfb[i]);
              float c = performOperation<float, Operation, order_flag>(a, b);
              d.e[i] = static_cast<_T>(c);
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
        }
      }
    }
    else
    {
      // Copy partial tiles with element accesses
      constexpr uint32_t row_bytes = BIG_TILE_SIZE_K;
      constexpr uint32_t ld_per_row = BIG_TILE_SIZE_K;
      constexpr uint32_t rows_per_wg = _WG / ld_per_row;
      constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE_N / rows_per_wg;
      // Make sure WG isn't too large
      static_assert(vmem_per_thread >= 1);

      const _T0 *pat = (const _T0 *)a + ti * BIG_TILE_SIZE_N * stride2 + tj * row_bytes + current_m * stride0;
#pragma unroll
      for (uint32_t t = 0; t < vmem_per_thread; t++)
      {
        uint32_t col = threadIdx.x % ld_per_row;
        uint32_t row = threadIdx.x / ld_per_row + t * rows_per_wg;
        uint64_t offset = (col < current_k_size && row < current_n_size) ? row * stride2 + col : 0;
        const _T0 *pfa = (const _T0 *)(pat + offset);
        sa[row][col] = *pfa;
      }
      __syncthreads();

      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      _T *pc = (_T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col;
          const _T1 *pfb = (const _T1 *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          if (types_match)
          {
            *pfc = performOperation<_T, Operation, order_flag>(static_cast<_T>(sa[col][row]), static_cast<_T>(*pfb));
          }
          else
          {
            float a = static_cast<float>(sa[col][row]);
            float b = static_cast<float>(*pfb);
            float c = performOperation<float, Operation, order_flag>(a, b);
            *pfc = static_cast<_T>(c);
          }
        }
      }
    }
  }

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcast_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                                 const int N, const int K, bool types_match)
  {
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

    const uint64_t stride_k = N;
    const uint64_t out_stride_nk = N * K;

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;

    // use 128bit load&store whenever possible
    if (current_n_size % 8 == 0 && current_k_size % elements_in_16B == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / elements_in_16B;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const _T0 *pa = (const _T0 *)a + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr;
      const _T *pc = (const _T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * elements_in_16B < current_n_size && row < current_k_size)
        {
          BLOCK_16B d, f;
          uint64_t offset = row * stride_k + col * elements_in_16B;
          if (types_match)
          {
            const __uint128_t *pfa = (const __uint128_t *)(pa + offset);
            const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
            f.ow = *pfa;
            d.ow = *pfb;
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              d.e[i] = performOperation<_T, Operation, order_flag>(static_cast<_T>(f.e[i]), static_cast<_T>(d.e[i]));
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
          else
          {
            const _T0 *pfa = (const _T0 *)(pa + offset);
            const _T1 *pfb = (const _T1 *)(pb + offset);
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              float a = static_cast<float>(pfa[i]);
              float b = static_cast<float>(pfb[i]);
              float c = performOperation<float, Operation, order_flag>(a, b);
              d.e[i] = static_cast<_T>(c);
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
        }
      }
    }
    else
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const _T0 *pa = (const _T0 *)a + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr;
      const _T *pc = (const _T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col;
          const _T0 *pfa = (const _T0 *)(pa + offset);
          const _T1 *pfb = (const _T1 *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          if (types_match)
          {
            *pfc = performOperation<_T, Operation, order_flag>(static_cast<_T>(*pfa), static_cast<_T>(*pfb));
          }
          else
          {
            float a = static_cast<float>(*pfa);
            float b = static_cast<float>(*pfb);
            float c = performOperation<float, Operation, order_flag>(a, b);
            *pfc = static_cast<_T>(c);
          }
        }
      }
    }
  }

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcast1_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                                  const int N, const int K, bool types_match)
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

    const uint64_t stride_k = N;
    const uint64_t out_stride_nk = N * K;

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;

    // use 128bit load&store whenever possible
    if (current_n_size % 8 == 0 && current_k_size % elements_in_16B == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / elements_in_16B;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const _T0 *pa = (const _T0 *)a + ti * row_bytes_wr + current_m * stride_k;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T *pc = (const _T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * elements_in_16B < current_n_size && row < current_k_size)
        {
          uint64_t offset_a = col * elements_in_16B;
          uint64_t offset = row * stride_k + col * elements_in_16B;
          BLOCK_16B d, f;
          if (types_match)
          {
            const __uint128_t *pfa = (const __uint128_t *)(pa + offset_a);
            const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
            f.ow = *pfa;
            d.ow = *pfb;
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              d.e[i] = performOperation<_T, Operation, order_flag>(static_cast<_T>(f.e[i]), static_cast<_T>(d.e[i]));
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
          else
          {
            const _T0 *pfa = (const _T0 *)(pa + offset_a);
            const _T1 *pfb = (const _T1 *)(pb + offset);
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              float a = static_cast<float>(pfa[i]);
              float b = static_cast<float>(pfb[i]);
              float c = performOperation<float, Operation, order_flag>(a, b);
              d.e[i] = static_cast<_T>(c);
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
        }
      }
    }
    else
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const _T0 *pa = (const _T0 *)a + ti * row_bytes_wr + current_m * stride_k;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T *pc = (const _T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset_a = col;
          uint64_t offset = row * stride_k + col;
          const _T0 *pfa = (const _T0 *)(pa + offset_a);
          const _T1 *pfb = (const _T1 *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          if (types_match)
          {
            *pfc = performOperation<_T, Operation, order_flag>(static_cast<_T>(*pfa), static_cast<_T>(*pfb));
          }
          else
          {
            float a = static_cast<float>(*pfa);
            float b = static_cast<float>(*pfb);
            float c = performOperation<float, Operation, order_flag>(a, b);
            *pfc = static_cast<_T>(c);
          }
        }
      }
    }
  }

  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcast_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int M, const int N, const int K, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_tiles = N / _rows;
    uint32_t k_tiles = K / elements_in_16B;
    if (idx < (uint64_t)M * n_tiles * k_tiles)
    {
      uint32_t ti = idx / (k_tiles * n_tiles);
      uint64_t idx_block = idx % (k_tiles * n_tiles);
      uint32_t tj = (idx_block / k_tiles) % n_tiles;
      uint32_t tk = idx_block % k_tiles;
      for (int row = 0; row < _rows; row++)
      {
        uint64_t offset_b = (uint64_t)(tj + row * n_tiles) * K + tk * elements_in_16B;
        uint64_t offset_ac = (uint64_t)(tj + row * n_tiles) * K + tk * elements_in_16B + (uint64_t)ti * N * K;
        const _T0 *pa = reinterpret_cast<const _T0 *>(a) + offset_ac;
        const _T1 *pb = reinterpret_cast<const _T1 *>(b) + offset_b;
        _T *pc = reinterpret_cast<_T *>(c) + offset_ac;
        for (int col = 0; col < elements_in_16B; col++)
        {
          const _T0 *pfa = pa + col;
          const _T1 *pfb = pb + col;
          _T *pfc = pc + col;
          if (types_match)
          {
            *pfc = performOperation<_T, Operation, order_flag>(static_cast<_T>(*pfa), static_cast<_T>(*pfb));
          }
          else
          {
            float t0 = static_cast<float>(*pfa);
            float t1 = static_cast<float>(*pfb);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *pfc = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  // (m, n, k), (k,)
  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcastK_unroll_vectorize_naive(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int m, const int n, const int k, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T);
    constexpr uint32_t vec_size = 16 / element_size;
    uint64_t idx = (uint64_t)(blockIdx.x * blockDim.x * _rows + threadIdx.x) * vec_size;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index + (uint64_t)((_rows - 1) * blockDim.x * vec_size) < (m * n * k); index += gridDim.x * blockDim.x * _rows * vec_size)
    {
#pragma unroll
      for (int unroll_index = 0; unroll_index < _rows; ++unroll_index)
      {
        uint64_t block_offset = (uint64_t)blockDim.x * vec_size * unroll_index;
        uint64_t other_start = (index + block_offset) % k;
#pragma unroll
        for (int vec_index = 0; vec_index < vec_size; ++vec_index)
        {
          uint64_t input_offset = index + block_offset + vec_index;
          uint64_t other_offset = other_start + vec_index;
          if (types_match)
          {
            _T t0 = static_cast<_T>(a_ptr[input_offset]);
            _T t1 = static_cast<_T>(b_ptr[other_offset]);
            *(c_ptr + input_offset) = performOperation<_T, Operation, order_flag>(t0, t1);
          }
          else
          {
            float t0 = static_cast<float>(a_ptr[input_offset]);
            float t1 = static_cast<float>(b_ptr[other_offset]);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *(c_ptr + input_offset) = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  // (m, n, k), (1)
  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcast_scalar_unroll_vectorize_naive(const void *__restrict a, const void *__restrict b, void *__restrict c, const int n)
  {
    constexpr uint32_t element_size = sizeof(_T);
    constexpr uint32_t vec_size = 16 / element_size;
    uint64_t idx = (uint64_t)(blockIdx.x * blockDim.x * _rows + threadIdx.x) * vec_size;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1 b_val = *reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    _T b_T = static_cast<_T>(b_val);
    for (uint64_t index = idx; index + (uint64_t)((_rows - 1) * blockDim.x * vec_size) < n; index += gridDim.x * blockDim.x * _rows * vec_size)
    {
#pragma unroll
      for (int unroll_index = 0; unroll_index < _rows; ++unroll_index)
      {
        uint64_t block_offset = (uint64_t)blockDim.x * vec_size * unroll_index;
#pragma unroll
        for (int vec_index = 0; vec_index < vec_size; ++vec_index)
        {
          uint64_t input_offset = index + block_offset + vec_index;
          _T t0 = static_cast<_T>(a_ptr[input_offset]);
          *(c_ptr + input_offset) = performOperation<_T, Operation, order_flag>(t0, b_T);
        }
      }
    }
  }

  // (m, n, k), (m, 1, k)
  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcastM1K_unroll_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int m, const int n, const int k, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T);
    constexpr uint32_t vec_size = 16 / element_size;
    uint64_t idx = (uint64_t)(blockIdx.x * blockDim.x * _rows + threadIdx.x) * vec_size;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index + (uint64_t)((_rows - 1) * blockDim.x * vec_size) < (m * n * k); index += gridDim.x * blockDim.x * _rows * vec_size)
    {
#pragma unroll
      for (int unroll_index = 0; unroll_index < _rows; ++unroll_index)
      {
        uint64_t block_offset = (uint64_t)blockDim.x * vec_size * unroll_index;
        uint64_t other_m_index = (index + block_offset) / (n * k);
        uint64_t other_k_index = (index + block_offset) % k;
#pragma unroll
        for (int vec_index = 0; vec_index < vec_size; ++vec_index)
        {
          uint64_t input_offset = index + block_offset + vec_index;
          uint64_t other_offset = other_m_index * k + other_k_index + vec_index;
          if (types_match)
          {
            _T t0 = static_cast<_T>(a_ptr[input_offset]);
            _T t1 = static_cast<_T>(b_ptr[other_offset]);
            *(c_ptr + input_offset) = performOperation<_T, Operation, order_flag>(t0, t1);
          }
          else
          {
            float t0 = static_cast<float>(a_ptr[input_offset]);
            float t1 = static_cast<float>(b_ptr[other_offset]);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *(c_ptr + input_offset) = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  // (m, n, k), (m, n, 1)
  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcastMN1_unroll_vec_naive(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int forward_dim, const int bcast_dim, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T);
    constexpr uint32_t vec_size = 16 / element_size;
    uint64_t idx = (uint64_t)(blockIdx.x * blockDim.x * _rows + threadIdx.x) * vec_size;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index + (uint64_t)((_rows - 1) * blockDim.x * vec_size) < (forward_dim * bcast_dim); index += gridDim.x * blockDim.x * _rows * vec_size)
    {
#pragma unroll
      for (int unroll_index = 0; unroll_index < _rows; ++unroll_index)
      {
        uint64_t block_offset = (uint64_t)blockDim.x * vec_size * unroll_index;
        uint64_t other_offset = (index + block_offset) / bcast_dim;
#pragma unroll
        for (int vec_index = 0; vec_index < vec_size; ++vec_index)
        {
          uint64_t input_offset = index + block_offset + vec_index;
          if (types_match)
          {
            _T t0 = static_cast<_T>(a_ptr[input_offset]);
            _T t1 = static_cast<_T>(b_ptr[other_offset]);
            *(c_ptr + input_offset) = performOperation<_T, Operation, order_flag>(t0, t1);
          }
          else
          {
            float t0 = static_cast<float>(a_ptr[input_offset]);
            float t1 = static_cast<float>(b_ptr[other_offset]);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *(c_ptr + input_offset) = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcastMN1_naive(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int forward_dim, const int bcast_dim, bool types_match)
  {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index < forward_dim * bcast_dim; index += blockDim.x * gridDim.x)
    {
      if (types_match)
      {
        _T t0 = static_cast<_T>(a_ptr[index]);
        _T t1 = static_cast<_T>(b_ptr[index / bcast_dim]);
        *(c_ptr + index) = performOperation<_T, Operation, order_flag>(t0, t1);
      }
      else
      {
        float t0 = static_cast<float>(a_ptr[index]);
        float t1 = static_cast<float>(b_ptr[index / bcast_dim]);
        float t2 = performOperation<float, Operation, order_flag>(t0, t1);
        *(c_ptr + index) = static_cast<_T>(t2);
      }
    }
  }

  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcast1N1_unroll_vec_naive(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int m, const int n, const int k, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T);
    constexpr uint32_t vec_size = 16 / element_size;
    uint64_t idx = (uint64_t)(blockIdx.x * blockDim.x * _rows + threadIdx.x) * vec_size;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index + (uint64_t)((_rows - 1) * blockDim.x * vec_size) < (m * n * k); index += gridDim.x * blockDim.x * _rows * vec_size)
    {
#pragma unroll
      for (int unroll_index = 0; unroll_index < _rows; ++unroll_index)
      {
        uint64_t block_offset = (uint64_t)blockDim.x * vec_size * unroll_index;
        uint64_t other_offset = (index + block_offset) % (n * k) / k;
#pragma unroll
        for (int vec_index = 0; vec_index < vec_size; ++vec_index)
        {
          uint64_t input_offset = index + block_offset + vec_index;
          if (types_match)
          {
            _T t0 = static_cast<_T>(a_ptr[input_offset]);
            _T t1 = static_cast<_T>(b_ptr[other_offset]);
            *(c_ptr + input_offset) = performOperation<_T, Operation, order_flag>(t0, t1);
          }
          else
          {
            float t0 = static_cast<float>(a_ptr[input_offset]);
            float t1 = static_cast<float>(b_ptr[other_offset]);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *(c_ptr + input_offset) = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_bcast1N1_naive(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int m, const int n, const int k, bool types_match)
  {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index < m * n * k; index += blockDim.x * gridDim.x)
    {
      if (types_match)
      {
        _T t0 = static_cast<_T>(a_ptr[index]);
        _T t1 = static_cast<_T>(b_ptr[index % (n * k) / k]);
        *(c_ptr + index) = performOperation<_T, Operation, order_flag>(t0, t1);
      }
      else
      {
        float t0 = static_cast<float>(a_ptr[index]);
        float t1 = static_cast<float>(b_ptr[index % (n * k) / k]);
        float t2 = performOperation<float, Operation, order_flag>(t0, t1);
        *(c_ptr + index) = static_cast<_T>(t2);
      }
    }
  }

  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_contiguous_kernel_naive(const void* __restrict a, const void* __restrict b, void* __restrict c,
                                                   const int n, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T);
    constexpr uint32_t vec_size = 16 / element_size;
    uint64_t idx = (uint64_t)(blockIdx.x * blockDim.x * _rows + threadIdx.x) * vec_size;
    const _T0* a_ptr = reinterpret_cast<const _T0*>(a);
    const _T1* b_ptr = reinterpret_cast<const _T1*>(b);
    _T* c_ptr = reinterpret_cast<_T*>(c);
    for (uint64_t index = idx; index + (uint64_t)(_rows - 1) * blockDim.x * vec_size < n; index += gridDim.x * blockDim.x * _rows * vec_size)
    {
#pragma unroll
      for (int unroll_index = 0; unroll_index < _rows; ++unroll_index)
      {
        uint64_t block_offset = (uint64_t)blockDim.x * vec_size * unroll_index;
#pragma unroll
        for (int vec_index = 0; vec_index < vec_size; ++vec_index)
        {
          uint64_t offset = index + block_offset + vec_index;
          if (types_match)
          {
            _T t0 = static_cast<_T>(a_ptr[offset]);
            _T t1 = static_cast<_T>(b_ptr[offset]);
            *(c_ptr + offset) = performOperation<_T, Operation, order_flag>(t0, t1);
          }
          else
          {
            float t0 = static_cast<float>(a_ptr[offset]);
            float t1 = static_cast<float>(b_ptr[offset]);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *(c_ptr + offset) = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  template <class _T, int _rows, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_contiguous_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int M, const int N, const int K, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_tiles = N / _rows;
    uint32_t k_tiles = K / elements_in_16B;
    if (idx < (uint64_t)M * n_tiles * k_tiles)
    {
      uint32_t ti = idx / (k_tiles * n_tiles);
      uint64_t idx_block = idx % (k_tiles * n_tiles);
      uint32_t tj = (idx_block / k_tiles) % n_tiles;
      uint32_t tk = idx_block % k_tiles;
      for (int row = 0; row < _rows; row++)
      {
        uint64_t offset = (uint64_t)(tj + row * n_tiles) * K + tk * elements_in_16B + (uint64_t)ti * N * K;
        const _T0 *pa = reinterpret_cast<const _T0 *>(a) + offset;
        const _T1 *pb = reinterpret_cast<const _T1 *>(b) + offset;
        _T *pc = reinterpret_cast<_T *>(c) + offset;
        for (int col = 0; col < elements_in_16B; col++)
        {
          const _T0 *pfa = pa + col;
          const _T1 *pfb = pb + col;
          _T *pfc = pc + col;
          if (types_match)
          {
            *pfc = performOperation<_T, Operation, order_flag>(static_cast<_T>(*pfa), static_cast<_T>(*pfb));
          }
          else
          {
            float t0 = static_cast<float>(*pfa);
            float t1 = static_cast<float>(*pfb);
            float t2 = performOperation<float, Operation, order_flag>(t0, t1);
            *pfc = static_cast<_T>(t2);
          }
        }
      }
    }
  }

  template <class _T, typename Operation, class _T0, class _T1>
  __global__ void operator_element_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                          const int size, bool types_match)
  {
    constexpr uint32_t element_size = sizeof(_T); // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;
    uint64_t idx = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (idx * elements_in_16B < size)
    {
      int offset = idx * elements_in_16B;
      const _T0 *pa = reinterpret_cast<const _T0 *>(a) + offset;
      const _T1 *pb = reinterpret_cast<const _T1 *>(b) + offset;
      _T *pc = reinterpret_cast<_T *>(c) + offset;
#pragma unroll
      for (uint32_t v = 0; v < elements_in_16B; v++)
      {
        if (types_match)
        {
          pc[v] = performOperation<_T, Operation, true>(static_cast<_T>(pa[v]), static_cast<_T>(pb[v]));
        }
        else
        {
          float t0 = static_cast<float>(pa[v]);
          float t1 = static_cast<float>(pb[v]);
          float t2 = performOperation<float, Operation, true>(t0, t1);
          pc[v] = static_cast<_T>(t2);
        }
      }
    }
  }

  template <class _T, int _WG, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag, class _T0, class _T1>
  __global__ void operator_contiguous_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                                      const int N, const int K, bool types_match)
  {
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

    const uint64_t stride_k = N;
    const uint64_t out_stride_nk = N * K;

    const uint32_t current_nk = swizzle_m ? blockIdx.x / M_SWIZZLE % nk_tiles : blockIdx.x % nk_tiles;
    const uint32_t ti = current_nk / k_tiles;
    const uint32_t tj = current_nk % k_tiles;

    const uint32_t current_n_size = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE_N) != 0) ? (N % BIG_TILE_SIZE_N) : BIG_TILE_SIZE_N;
    const uint32_t current_k_size = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE_K) != 0) ? (K % BIG_TILE_SIZE_K) : BIG_TILE_SIZE_K;

    // use 128bit load&store whenever possible
    if (current_n_size % 8 == 0 && current_k_size % elements_in_16B == 0)
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = row_bytes_wr / elements_in_16B;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      // Make sure WG isn't too large
      static_assert(wr_per_row >= 1);

      const _T0 *pa = (const _T0 *)a + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T *pc = (const _T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col * elements_in_16B < current_n_size && row < current_k_size)
        {
          BLOCK_16B d, f;
          uint64_t offset = row * stride_k + col * elements_in_16B;
          if (types_match)
          {
            const __uint128_t *pfa = (const __uint128_t *)(pa + offset);
            const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
            f.ow = *pfa;
            d.ow = *pfb;
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              d.e[i] = performOperation<_T, Operation, order_flag>(static_cast<_T>(f.e[i]), static_cast<_T>(d.e[i]));
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
          else
          {
            const _T0 *pfa = (const _T0 *)(pa + offset);
            const _T1 *pfb = (const _T1 *)(pb + offset);
#pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
              float a = static_cast<float>(pfa[i]);
              float b = static_cast<float>(pfb[i]);
              float c = performOperation<float, Operation, order_flag>(a, b);
              d.e[i] = static_cast<_T>(c);
            }
            __uint128_t *pfc = (__uint128_t *)(pc + offset);
            *pfc = d.ow;
          }
        }
      }
    }
    else
    {
      // Copy full tile with large loads
      constexpr uint32_t row_bytes_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t vmem_per_row_wr = BIG_TILE_SIZE_N;
      constexpr uint32_t rows_per_wg_wr = _WG / vmem_per_row_wr;
      constexpr uint32_t wr_per_row = BIG_TILE_SIZE_K / rows_per_wg_wr;
      const _T0 *pa = (const _T0 *)a + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T1 *pb = (const _T1 *)b + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
      const _T *pc = (const _T *)c + tj * BIG_TILE_SIZE_K * stride_k + ti * row_bytes_wr + current_m * out_stride_nk;
#pragma unroll
      for (uint32_t t = 0; t < wr_per_row; t++)
      {
        uint32_t col = threadIdx.x % vmem_per_row_wr;
        uint32_t row = threadIdx.x / vmem_per_row_wr + t * rows_per_wg_wr;
        if (col < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col;
          const _T0 *pfa = (const _T0 *)(pa + offset);
          const _T1 *pfb = (const _T1 *)(pb + offset);
          _T *pfc = (_T *)(pc + offset);
          if (types_match)
          {
            *pfc = performOperation<_T, Operation, order_flag>(static_cast<_T>(*pfa), static_cast<_T>(*pfb));
          }
          else
          {
            float a = static_cast<float>(*pfa);
            float b = static_cast<float>(*pfb);
            float c = performOperation<float, Operation, order_flag>(a, b);
            *pfc = static_cast<_T>(c);
          }
        }
      }
    }
  }
} // namespace aiter

std::vector<int64_t> broadcastShapes(const torch::Tensor &tensor1, const torch::Tensor &tensor2)
{
  auto shape1 = tensor1.sizes().vec();
  auto shape2 = tensor2.sizes().vec();

  int64_t max_dim = std::max(shape1.size(), shape2.size());
  shape1.insert(shape1.begin(), max_dim - shape1.size(), 1);
  shape2.insert(shape2.begin(), max_dim - shape2.size(), 1);

  std::vector<int64_t> result_shape(max_dim, 1);
  for (int64_t i = 0; i < max_dim; ++i)
  {
    if (shape1[i] == 1)
    {
      result_shape[i] = shape2[i];
    }
    else if (shape2[i] == 1)
    {
      result_shape[i] = shape1[i];
    }
    else if (shape1[i] == shape2[i])
    {
      result_shape[i] = shape1[i];
    }
    else
    {
      throw std::invalid_argument("Incompatible shapes for binary operator.");
    }
  }

  return result_shape;
}

template <int pattern, typename Operation, class _T0, class _T1>
struct BinaryOperationPattern;

// PATTERN_TRANSPOSE
template <typename Operation, class _T0, class _T1>
struct BinaryOperationPattern<1, Operation, _T0, _T1>
{
  static void apply(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
  {
    int dim = input.dim();
    auto shape = output.sizes().vec();
    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());

    int num_elements = output.numel();
    int rem_dim_size = num_elements / (shape[dim - 2] * shape[dim - 1]);
    int M = dim == 2 ? 1 : rem_dim_size;
    int N = shape[dim - 2];
    int K = shape[dim - 1];

    auto tensor_not_conti = input.is_contiguous() ? other : input;
    int stride0 = tensor_not_conti.stride(0);
    int stride2 = tensor_not_conti.stride(2);
    constexpr uint32_t BIG_TILE_SIZE_N = 64;
    constexpr uint32_t BIG_TILE_SIZE_K = 64;
    constexpr uint32_t M_SWIZZLE = 8;
    const int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
    const dim3 grid_dim(grid_x, 1, 1);
    const dim3 block_dim(256, 1, 1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool types_match = typeid(_T0) == typeid(_T1);

    if (order_flag)
    {
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_tn_big_tile_kernel", [&]
          { aiter::operator_tn_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true, _T0, _T1>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, stride0, stride2, types_match); });
    }
    else
    {
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_tn_big_tile_kernel", [&]
          { aiter::operator_tn_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false, _T1, _T0>
                <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N, stride0, stride2, types_match); });
    }
  }
};

// PATTERN_BROADCAST_0
template <typename Operation, class _T0, class _T1>
struct BinaryOperationPattern<2, Operation, _T0, _T1>
{
  static void apply(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
  {
    int dim = input.dim();
    auto shape = output.sizes().vec();

    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());
    int num_elements = output.numel();
    int rem_dim_size = num_elements / (shape[dim - 2] * shape[dim - 1]);
    int M = dim != 3 ? 1 : rem_dim_size;
    int N = shape[dim - 2];
    int K = shape[dim - 1];
    if (dim == 4)
    {
      N = shape[0] * shape[1] * shape[2];
      K = shape[3];
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool types_match = typeid(_T0) == typeid(_T1);

    const uint32_t rows = 8;
    int vec = 16 / output.element_size();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);
    uint32_t num_cu = dev_prop.multiProcessorCount;

    bool bcast_k_dim = (input.dim() == 1 && input.size(0) == K) || (other.dim() == 1 && other.size(0) == K);
    bool bcast_scalar = (input.dim() == 1 && input.size(0) == 1) || (other.dim() == 1 && other.size(0) == 1);
    bool vec_unroll_able = num_elements % (rows * vec * 256) == 0 && K % vec == 0;

    //  (m,n,k), (k,)
    if (bcast_k_dim && vec_unroll_able)
    {
      int grid_x = (num_elements / (rows * vec) + 256 - 1) / 256;
      int occupancy;
      auto kernel_ptr = aiter::operator_bcastK_unroll_vectorize_naive<_T0, rows, Operation, true, _T0, _T1>;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, reinterpret_cast<const void*>(kernel_ptr), 256, 0);
      grid_x = grid_x < num_cu * occupancy ? grid_x : num_cu * occupancy;
      const dim3 block_dim(256, 1, 1);
      const dim3 grid_dim(grid_x, 1, 1);
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastK_unroll_vectorize_naive", [&]
            { aiter::operator_bcastK_unroll_vectorize_naive<scalar_t, rows, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastK_unroll_vectorize_naive", [&]
            { aiter::operator_bcastK_unroll_vectorize_naive<scalar_t, rows, Operation, true, _T1, _T0>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, M, N, K, types_match); });
      }
    }
    // (m, n, k), (1)
    else if (bcast_scalar && num_elements % (rows * vec * 256) == 0)
    {
      int grid_x = (num_elements / (rows * vec) + 256 - 1) / 256;
      int occupancy;
      auto kernel_ptr = aiter::operator_bcast_scalar_unroll_vectorize_naive<_T0, rows, Operation, true, _T0, _T1>;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, reinterpret_cast<const void*>(kernel_ptr), 256, 0);
      grid_x = grid_x < num_cu * occupancy ? grid_x : num_cu * occupancy;
      const dim3 block_dim(256, 1, 1);
      const dim3 grid_dim(grid_x, 1, 1);
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_scalar_unroll_vectorize_naive", [&]
            { aiter::operator_bcast_scalar_unroll_vectorize_naive<scalar_t, rows, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, num_elements); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_scalar_unroll_vectorize_naive", [&]
            { aiter::operator_bcast_scalar_unroll_vectorize_naive<scalar_t, rows, Operation, true, _T1, _T0>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, num_elements); });
      }
    }
    else if (N % rows == 0 && K % vec == 0)
    {
      constexpr uint32_t wg = 64;
      int grid_x = (num_elements / (rows * vec) + wg - 1) / wg;
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);

      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_tile_kernel", [&]
            { aiter::operator_bcast_tile_kernel<scalar_t, rows, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_tile_kernel", [&]
            { aiter::operator_bcast_tile_kernel<scalar_t, rows, Operation, false, _T1, _T0>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, M, N, K, types_match); });
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
            output.scalar_type(), "operator_bcast_big_tile_kernel", [&]
            { aiter::operator_bcast_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_big_tile_kernel", [&]
            { aiter::operator_bcast_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false, _T1, _T0>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N, types_match); });
      }
    }
  }
};

// PATTERN_BROADCAST_1
template <typename Operation, class _T0, class _T1>
struct BinaryOperationPattern<3, Operation, _T0, _T1>
{
  static void apply(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
  {
    int dim = input.dim();
    auto shape = output.sizes().vec();
    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());

    int num_elements = output.numel();
    int rem_dim_size = num_elements / (shape[dim - 2] * shape[dim - 1]);
    int M = dim == 2 ? 1 : rem_dim_size;
    int N = shape[dim - 2];
    int K = shape[dim - 1];

    // (m, n, p, q), (m, 1, p, q)
    if (dim == 4 && input.size(1) != other.size(1) && (input.size(1) == 1 || other.size(1)))
    {
      M = shape[0];
      N = shape[1];
      K = shape[2] * shape[3];
    }

    constexpr uint32_t BIG_TILE_SIZE_N = 64;
    constexpr uint32_t BIG_TILE_SIZE_K = 64;
    constexpr uint32_t M_SWIZZLE = 8;
    int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
    const dim3 grid_dim(grid_x, 1, 1);
    const dim3 block_dim(256, 1, 1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool types_match = typeid(_T0) == typeid(_T1);
    constexpr int rows = 8;
    int vec_size = 16 / output.element_size();
    if (K % vec_size == 0 && num_elements % (256 * rows * vec_size) == 0)
    {
      hipDevice_t dev;
      hipDeviceProp_t dev_prop;
      hipGetDevice(&dev);
      hipGetDeviceProperties(&dev_prop, dev);
      uint32_t num_cu = dev_prop.multiProcessorCount;
      grid_x = (num_elements / (rows * vec_size) + 256 - 1) / 256;
      int occupancy;
      auto kernel_ptr = aiter::operator_bcastM1K_unroll_kernel<_T0, rows, Operation, true, _T0, _T1>;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, reinterpret_cast<const void*>(kernel_ptr), 256, 0);
      grid_x = grid_x < num_cu * occupancy ? grid_x : num_cu * occupancy;

      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(256, 1, 1);
      if (!order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastM1K_unroll_kernel", [&]
            { aiter::operator_bcastM1K_unroll_kernel<scalar_t, rows, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastM1K_unroll_kernel", [&]
            { aiter::operator_bcastM1K_unroll_kernel<scalar_t, rows, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, M, N, K, types_match); });
      }
    }
    else
    {
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast1_big_tile_kernel", [&]
            { aiter::operator_bcast1_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast1_big_tile_kernel", [&]
            { aiter::operator_bcast1_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false, _T1, _T0>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N, types_match); });
      }
    }
  }
};

// PATTERN_BROADCAST_2
template <typename Operation, class _T0, class _T1>
struct BinaryOperationPattern<5, Operation, _T0, _T1>
{
  static void apply(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
  {
    int dim = input.dim();
    auto shape = output.sizes().vec();
    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());

    int bcast_dim = order_flag ? input.numel() / other.numel() : other.numel() / input.numel();
    int forward_dim = order_flag ? other.numel() : input.numel();

    int num_elements = output.numel();
    int vec_size = 16 / output.element_size();
    constexpr uint32_t row = 8;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool types_match = typeid(_T0) == typeid(_T1);

    // optimize kernel
    if (bcast_dim % vec_size == 0 && forward_dim % row == 0 && num_elements % (256 * vec_size * row) == 0)
    {
      hipDevice_t dev;
      hipDeviceProp_t dev_prop;
      hipGetDevice(&dev);
      hipGetDeviceProperties(&dev_prop, dev);
      uint32_t num_cu = dev_prop.multiProcessorCount;

      constexpr uint32_t wg = 256;
      int grid_x = (num_elements / (row * vec_size) + wg - 1) / wg;
      int occupancy;
      auto kernel_ptr = aiter::operator_bcastMN1_unroll_vec_naive<_T0, row, Operation, true, _T0, _T1>;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, reinterpret_cast<const void*>(kernel_ptr), wg, 0);
      grid_x = grid_x < num_cu * occupancy ? grid_x : num_cu * occupancy;

      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastMN1_unroll_vec_naive", [&]
            { aiter::operator_bcastMN1_unroll_vec_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, forward_dim, bcast_dim, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastMN1_unroll_vec_naive", [&]
            { aiter::operator_bcastMN1_unroll_vec_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, forward_dim, bcast_dim, types_match); });
      }
    }
    // fallback
    else
    {
      const dim3 block_dim(256, 1, 1);
      const dim3 grid_dim((num_elements + 256 - 1) / 256, 1, 1);
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastMN1_naive", [&]
            { aiter::operator_bcastMN1_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, forward_dim, bcast_dim, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcastMN1_naive", [&]
            { aiter::operator_bcastMN1_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, forward_dim, bcast_dim, types_match); });
      }
    }
  }
};

// PATTERN_BROADCAST_3
template <typename Operation, class _T0, class _T1>
struct BinaryOperationPattern<6, Operation, _T0, _T1>
{
  static void apply(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
  {
    int dim = output.dim();
    auto shape = output.sizes().vec();
    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());

    int m = output.size(0);
    int n = output.size(1);
    int k = output.size(2);

    int num_elements = output.numel();
    int vec_size = 16 / output.element_size();
    constexpr uint32_t row = 8;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool types_match = typeid(_T0) == typeid(_T1);

    // optimize kernel
    if (k % vec_size == 0 && num_elements % (256 * vec_size * row) == 0)
    {
      hipDevice_t dev;
      hipDeviceProp_t dev_prop;
      hipGetDevice(&dev);
      hipGetDeviceProperties(&dev_prop, dev);
      uint32_t num_cu = dev_prop.multiProcessorCount;

      constexpr uint32_t wg = 256;
      int grid_x = (num_elements / (row * vec_size) + wg - 1) / wg;
      int occupancy;
      auto kernel_ptr = aiter::operator_bcast1N1_unroll_vec_naive<_T0, row, Operation, true, _T0, _T1>;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, reinterpret_cast<const void*>(kernel_ptr), wg, 0);
      grid_x = grid_x < num_cu * occupancy ? grid_x : num_cu * occupancy;

      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast1N1_unroll_vec_naive", [&]
            { aiter::operator_bcast1N1_unroll_vec_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, m, n, k, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast1N1_unroll_vec_naive", [&]
            { aiter::operator_bcast1N1_unroll_vec_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, m, n, k, types_match); });
      }
    }
    // fallback
    else
    {
      const dim3 block_dim(256, 1, 1);
      const dim3 grid_dim((num_elements + 256 - 1) / 256, 1, 1);
      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast1N1_naive", [&]
            { aiter::operator_bcast1N1_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, m, n, k, types_match); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast1N1_naive", [&]
            { aiter::operator_bcast1N1_naive<scalar_t, row, Operation, true, _T0, _T1>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, m, n, k, types_match); });
      }
    }
  }
};

// PATTERN_CONTIGUOUS
template <typename Operation, class _T0, class _T1>
struct BinaryOperationPattern<4, Operation, _T0, _T1>
{
  static void apply(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
  {
    int dim = input.dim();
    auto shape = output.sizes().vec();

    const uint32_t rows = 8;
    void *buf_a = reinterpret_cast<void *>(input.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());
    int num_elements = output.numel();
    int rem_dim_size = 1;
    int M, N, K;
    if (dim == 1)
    {
      M = 1;
      N = input.numel() / 128;
      K = 128;
    }
    else
    {
      for (int i = 0; i < dim - 2; ++i)
      {
        rem_dim_size *= shape[i];
      }
      M = dim == 3 ? shape[0] : rem_dim_size;
      N = shape[dim - 2];
      K = shape[dim - 1];
      if (N < rows)
      {
        K = N * K;
        N = M;
        M = 1;
      }
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool types_match = typeid(_T0) == typeid(_T1);
    int vec = 16 / output.element_size();
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);
    uint32_t num_cu = dev_prop.multiProcessorCount;

    if (num_elements % vec == 0 && num_elements < num_cu * 256 * vec)
    {
      constexpr uint32_t wg = 256;
      const int grid_x = (num_elements / vec + wg - 1) / wg;
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_element_kernel", [&]
          { aiter::operator_element_kernel<scalar_t, Operation, _T0, _T1>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, num_elements, types_match); });
    }
    else if (num_elements % (rows * vec * 256) == 0)
    {
      constexpr uint32_t wg = 256;
      int grid_x = (num_elements / (rows * vec) + wg - 1) / wg;
      int occupancy;
      auto kernel_ptr = aiter::operator_contiguous_kernel_naive<_T0, rows, Operation, true, _T0, _T1>;
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, reinterpret_cast<const void*>(kernel_ptr), wg, 0);
      grid_x = grid_x < num_cu * occupancy ? grid_x : num_cu * occupancy;
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_contiguous_kernel_naive", [&]
          { aiter::operator_contiguous_kernel_naive<scalar_t, rows, Operation, true, _T0, _T1>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, num_elements, types_match); });
    }
    else if (N % rows == 0 && K % vec == 0)
    {
      constexpr uint32_t wg = 256;
      const int grid_x = (num_elements / (rows * vec) + wg - 1) / wg;
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);

      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_contiguous_kernel", [&]
          { aiter::operator_contiguous_kernel<scalar_t, rows, Operation, true, _T0, _T1>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K, types_match); });
    }
    else
    {
      constexpr uint32_t wg = 256;
      constexpr uint32_t BIG_TILE_SIZE_N = 64;
      constexpr uint32_t BIG_TILE_SIZE_K = 64;
      constexpr uint32_t M_SWIZZLE = 8;
      const int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);

      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_contiguous_big_tile_kernel", [&]
          { aiter::operator_contiguous_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true, _T0, _T1>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, types_match); });
    }
  }
};

template <int pattern, typename Operation, class _T0, class _T1>
void binary_operation_process(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
{
  BinaryOperationPattern<pattern, Operation, _T0, _T1>::apply(input, other, output, order_flag);
}

#define DISPATCH_SECOND(pattern, Operation, _T0, scalar_type, cpp_type)                            \
  case scalar_type:                                                                                \
    binary_operation_process<pattern, Operation, _T0, cpp_type>(input, other, output, order_flag); \
    break

#define DISPATCH_FIRST(pattern, Operation, scalar_type, cpp_type)                    \
  case scalar_type:                                                                  \
    dispatch_second<pattern, Operation, cpp_type>(input, other, output, order_flag); \
    break

template <int pattern, typename Operation, typename _T0>
void dispatch_second(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
{
  switch (other.scalar_type())
  {
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kFloat32, float);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kFloat64, double);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kInt32, int);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kInt64, long long);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kBool, bool);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kHalf, torch::Half);
    DISPATCH_SECOND(pattern, Operation, _T0, torch::kBFloat16, torch::BFloat16);
  default:
    break;
  }
}

template <int pattern, typename Operation>
void dispatch_first(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, bool order_flag)
{
  switch (input.scalar_type())
  {
    DISPATCH_FIRST(pattern, Operation, torch::kFloat32, float);
    DISPATCH_FIRST(pattern, Operation, torch::kFloat64, double);
    DISPATCH_FIRST(pattern, Operation, torch::kInt32, int);
    DISPATCH_FIRST(pattern, Operation, torch::kInt64, long long);
    DISPATCH_FIRST(pattern, Operation, torch::kBool, bool);
    DISPATCH_FIRST(pattern, Operation, torch::kHalf, torch::Half);
    DISPATCH_FIRST(pattern, Operation, torch::kBFloat16, torch::BFloat16);
  default:
    break;
  }
}

#undef DISPATCH_SECOND
#undef DISPATCH_FIRST

template <typename Operation, bool Inplace = false>
torch::Tensor binary_operation(torch::Tensor &input, torch::Tensor &other)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  int dim = input.dim();

  bool is_support = false;
  bool order_flag = true;
  int pattern = 0;
  constexpr uint32_t PATTERN_TRANSPOSE = 1;
  constexpr uint32_t PATTERN_BROADCAST_0 = 2;   // (m, n, k), (1, n, k)
  constexpr uint32_t PATTERN_BROADCAST_1 = 3;   // (m, n, k), (m, 1, k)
  constexpr uint32_t PATTERN_CONTIGUOUS = 4;
  constexpr uint32_t PATTERN_BROADCAST_2 = 5;   // (m, n, k), (m, n, 1)
  constexpr uint32_t PATTERN_BROADCAST_3 = 6;   // (m, n, k), (   n, 1)

  // contiguous case
  if (!is_support)
  {
    is_support = true;
    is_support &= (input.dim() == other.dim());
    is_support &= input.is_contiguous() == other.is_contiguous();
    is_support &= input.is_contiguous() == true;
    if (input.dim() == 1)
    {
      is_support &= input.numel() % 128 == 0;
    }
    for (int i = 0; i < input.dim() && is_support; ++i)
    {
      is_support &= (input.size(i) == other.size(i));
    }
    pattern = is_support ? PATTERN_CONTIGUOUS : 0;
  }

  if (!is_support && (dim == 3 || other.dim() == 3))
  {
    // transpose case
    if (input.is_contiguous() != other.is_contiguous())
    {
      auto tensor_not_conti = input.is_contiguous() ? other : input;
      order_flag = !input.is_contiguous() ? true : false;
      is_support = true;
      // avoid broadcast
      is_support &= input.dim() == other.dim();
      is_support &= input.size(0) == other.size(0);
      is_support &= input.size(1) == other.size(1);
      is_support &= input.size(2) == other.size(2);
      is_support &= tensor_not_conti.stride(1) == 1;
      pattern = is_support ? PATTERN_TRANSPOSE : 0;
    }
    // broadcast case
    else if (input.is_contiguous() && other.is_contiguous())
    {
      is_support = false;
      // input tensor dim and other tensor dim both equal to 3
      if (input.dim() == other.dim())
      {
        // broadcast at dim0 or dim1 or dim2
        auto broadcast_3d_case = [&] (int bcast_dim)
        {
          constexpr int bcast_pattern[3] = {PATTERN_BROADCAST_0, PATTERN_BROADCAST_1, PATTERN_BROADCAST_2};
          if (!is_support && (input.size(bcast_dim) == 1 || other.size(bcast_dim)) && input.size(bcast_dim) != other.size(bcast_dim))
          {
            is_support = true;
            for (int i = 0; i < 3; ++i)
            {
              if (bcast_dim != i) is_support &= input.size(i) == other.size(i);
            }
            is_support &= input.size(bcast_dim) == 1 ? other.size(bcast_dim) != 1 : true;
            pattern = is_support ? bcast_pattern[bcast_dim] : 0;
            order_flag = input.size(bcast_dim) != 1 ? true : false;
            if (bcast_dim == 1) order_flag = !order_flag;
          }
        };
        // (m, n, k), (1, n, k) or (1, n, k), (m, n, k)
        broadcast_3d_case(0);
        // (m, n, k), (m, 1, k) or (m, 1, k), (m, n, k)
        broadcast_3d_case(1);
        // (m, n, k), (m, n, 1) or (m, n, 1), (m, n, k)
        broadcast_3d_case(2);
      }
      // (m, n, k), (n, 1) or (n, 1), (m, n, k)
      else if (input.dim() == 2 || other.dim() == 2)
      {
        is_support = true;
        if (input.dim() == 2)
        {
          is_support &= input.size(0) == other.size(1);
          is_support &= input.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = false;
        }
        else
        {
          is_support &= other.size(0) == input.size(1);
          is_support &= other.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = true;
        }
      }
      // (m, n, k), (k) or (k), (m, n, k)
      // (m, n, k), (1) or (1), (m, n, k)
      else if (input.dim() == 1 || other.dim() == 1)
      {
        if (other.dim() == 1)
        {
          if (other.size(0) == 1 || other.size(0) == input.size(2))
          {
            is_support = true;
            pattern = PATTERN_BROADCAST_0;
            order_flag = true;
          }
        }
        else
        {
          if (input.size(0) == 1 || input.size(0) == other.size(2))
          {
            is_support = true;
            pattern = PATTERN_BROADCAST_0;
            order_flag = false;
          }
        }
      }
    }
  }

  if (!is_support && input.dim() != 3 && other.dim() != 3)
  {
    if (input.dim() == other.dim())
    {
      std::vector<int> bcast_dim_index = {};
      for (int i = 0; i < input.dim(); ++i)
      {
        // broadcast condition
        if (input.size(i) != other.size(i) && (input.size(i) == 1 || other.size(i) == 1))
        {
          bcast_dim_index.push_back(i);
        }
      }
      if (bcast_dim_index.size() == 1 && bcast_dim_index[0] != 0 && bcast_dim_index[0] != input.dim() - 1)
      {
        is_support = true;
        pattern = PATTERN_BROADCAST_1;
        order_flag = other.size(bcast_dim_index[0]) == 1 ? true : false;
      }
    }
  }

  // hip does not support double
  if (input.dtype() == torch::kDouble || other.dtype() == torch::kDouble)
  {
    is_support = false;
  }

  if (is_support)
  {
    auto in0_dtype = input.dtype();
    auto in1_dtype = other.dtype();
    torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
    std::vector<int64_t> out_shape = broadcastShapes(input, other);
    auto device = input.device();
    auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());

    torch::Tensor output;
    if constexpr(Inplace)
    {
      input = input.to(out_dtype);
      output = input;
    }
    else
    {
      output = torch::empty(out_shape, options);
    }

    switch (pattern)
    {
      case PATTERN_TRANSPOSE:
        dispatch_first<1, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_0:
        dispatch_first<2, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_1:
        dispatch_first<3, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_CONTIGUOUS:
        dispatch_first<4, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_2:
        dispatch_first<5, Operation>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_3:
        dispatch_first<6, Operation>(input, other, output, order_flag);
        break;
      default:
        printf("[aiter/csrc/kernels/%s]: line %d break, unsupported type\n", __FILE__, __LINE__);
    }
    return output;
  }
  else
  {
    return aiter::aten_compute<Operation>(input, other);
  }
}

torch::Tensor aiter_add(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::AddOp, false>(input, other);
}

torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::SubOp, false>(input, other);
}

torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::MulOp, false>(input, other);
}

torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::DivOp, false>(input, other);
}

// inp interface
torch::Tensor aiter_add_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::AddOp, true>(input, other);
}

torch::Tensor aiter_sub_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::SubOp, true>(input, other);
}

torch::Tensor aiter_mul_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::MulOp, true>(input, other);
}

torch::Tensor aiter_div_(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::DivOp, true>(input, other);
}
