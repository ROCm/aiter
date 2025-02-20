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
#include <binary_operator.h>
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
  template <class _T, int _WG, int _vec, int BIG_TILE_SIZE_N, int BIG_TILE_SIZE_K, int M_SWIZZLE, typename Operation, bool order_flag>
  __global__ void operator_tn_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K, int stride0, int stride2)
  {
    // pad LDS row by dword
    constexpr uint32_t LDS_PAD = 4 / sizeof(_T);

    union BLOCK_16B
    {
      _T e[_vec];
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
    if (current_n_size % _vec == 0 && current_k_size % 8 == 0)
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
        uint64_t offset = (col * _vec < current_k_size && row < current_n_size) ? row * stride2 + col * sizeof(__uint128_t) : 0;
        const __uint128_t *pfa = (const __uint128_t *)(pat + offset);
        BLOCK_16B d;
        d.ow = *pfa;
#pragma unroll
        for (uint32_t i = 0; i < _vec; i++)
        {
          sa[row][col * _vec + i] = d.e[i];
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
        if (col * _vec < current_n_size && row < current_k_size)
        {
          uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
          BLOCK_16B d;
          const __uint128_t *pfb = (const __uint128_t *)(pb + offset);
          d.ow = *pfb;
// Transpose tile on read from LDS
#pragma unroll
          for (uint32_t i = 0; i < _vec; i++)
          {
            d.e[i] = aiter::performOperation<_T, Operation, order_flag>(sa[col * _vec + i][row], d.e[i]);
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
          *pfc = aiter::performOperation<_T, Operation, order_flag>(sa[col][row], *pfb);
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
            d.e[i] = aiter::performOperation<_T, Operation, order_flag>(f.e[i], d.e[i]);
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
          *pfc = aiter::performOperation<_T, Operation, order_flag>(*pfa, *pfb);
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
            d.e[i] = aiter::performOperation<_T, Operation, order_flag>(f.e[i], d.e[i]);
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

          *pfc = aiter::performOperation<_T, Operation, order_flag>(*pfa, *pfb);
        }
      }
    }
  }

  template <class _T, int _rows, int _vec, typename Operation, bool order_flag>
  __global__ void operator_bcast_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c,
                                             const int M, const int N, const int K)
  {
    int bpe = sizeof(_T);
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
        const uint8_t *pa = (const uint8_t *)a + offset_ac * bpe;
        const uint8_t *pb = (const uint8_t *)b + offset_b * bpe;
        const uint8_t *pc = (const uint8_t *)c + offset_ac * bpe;
        for (int col = 0; col < _vec; col++)
        {
          const _T *pfa = (const _T *)(pa + col * bpe);
          const _T *pfb = (const _T *)(pb + col * bpe);
          _T *pfc = (_T *)(pc + col * bpe);
          *pfc = aiter::performOperation<_T, Operation, order_flag>(*pfa, *pfb);
        }
      }
    }
  }
}

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

template <typename Operation, int vec>
void binary_operation_vec(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output, int pattern, bool order_flag)
{
  int dim = input.dim();
  constexpr uint32_t PATTERN_TRANSPOSE = 1;
  constexpr uint32_t PATTERN_BROADCAST_0 = 2;
  constexpr uint32_t PATTERN_BROADCAST_1 = 3;

  auto shape = output.sizes().vec();
  void *buf_c = reinterpret_cast<void *>(output.data_ptr());

  void *buf_a = reinterpret_cast<void *>(input.data_ptr());
  void *buf_b = reinterpret_cast<void *>(other.data_ptr());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int num_elements = output.numel();
  int rem_dim_size = num_elements / (shape[dim - 2] * shape[dim - 1]);
  int M = dim == 2 ? 1 : rem_dim_size;
  int N = shape[dim - 2];
  int K = shape[dim - 1];

  if (pattern == PATTERN_TRANSPOSE)
  {
    auto tensor_not_conti = input.is_contiguous() ? other : input;
    int stride0 = tensor_not_conti.stride(0);
    int stride2 = tensor_not_conti.stride(2);
    stride0 *= tensor_not_conti.element_size();
    stride2 *= tensor_not_conti.element_size();
    constexpr uint32_t BIG_TILE_SIZE_N = 64;
    constexpr uint32_t BIG_TILE_SIZE_K = 64;
    constexpr uint32_t M_SWIZZLE = 8;
    const int grid_x = M * ((N + BIG_TILE_SIZE_N - 1) / BIG_TILE_SIZE_N) * ((K + BIG_TILE_SIZE_K - 1) / BIG_TILE_SIZE_K);
    const dim3 grid_dim(grid_x, 1, 1);
    const dim3 block_dim(256, 1, 1);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (order_flag)
    {
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_tn_big_tile_kernel", [&]
          { aiter::operator_tn_big_tile_kernel<scalar_t, 256, vec, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, stride0, stride2); });
    }
    else
    {
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_tn_big_tile_kernel", [&]
          { aiter::operator_tn_big_tile_kernel<scalar_t, 256, vec, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N, stride0, stride2); });
    }
  }
  else if (pattern == PATTERN_BROADCAST_0)
  {
    const uint32_t rows = 8;
    // printf("PATTERN:%d\n", PATTERN_BROADCAST_0);
    if (N % rows == 0 && K % vec == 0)
    {
      constexpr uint32_t wg = 64;
      int grid_x = (num_elements / (rows * vec) + wg - 1) / wg;
      const dim3 grid_dim(grid_x, 1, 1);
      const dim3 block_dim(wg, 1, 1);

      if (order_flag)
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_tile_kernel", [&]
            { aiter::operator_bcast_tile_kernel<scalar_t, rows, vec, Operation, true>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, M, N, K); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_tile_kernel", [&]
            { aiter::operator_bcast_tile_kernel<scalar_t, rows, vec, Operation, false>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, M, N, K); });
      }
    }
    else
    {
      printf("in else.\n");
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
            { aiter::operator_bcast_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N); });
      }
      else
      {
        VLLM_DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "operator_bcast_big_tile_kernel", [&]
            { aiter::operator_bcast_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N); });
      }
    }
  }
  else if (pattern == PATTERN_BROADCAST_1)
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
          output.scalar_type(), "operator_bcast1_big_tile_kernel", [&]
          { aiter::operator_bcast1_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N); });
    }
    else
    {
      VLLM_DISPATCH_FLOATING_TYPES(
          output.scalar_type(), "operator_bcast1_big_tile_kernel", [&]
          { aiter::operator_bcast1_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N); });
    }
  }
}

template <typename Operation>
torch::Tensor binary_operation(torch::Tensor &input, torch::Tensor &other)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  int dim = input.dim();

  bool is_support = false;
  bool order_flag = true;
  int pattern = 0;
  constexpr uint32_t PATTERN_TRANSPOSE = 1;
  constexpr uint32_t PATTERN_BROADCAST_0 = 2;
  constexpr uint32_t PATTERN_BROADCAST_1 = 3;
  if (!is_support && dim == 2)
  {
    is_support = true;
    is_support &= input.dim() == other.dim();
    is_support &= input.size(0) == other.size(0);
    is_support &= input.size(1) == other.size(1);
    is_support &= input.size(1) > 512;
    is_support &= input.is_contiguous() == other.is_contiguous();
    is_support &= input.is_contiguous() == true;
    pattern = is_support ? PATTERN_BROADCAST_0 : 0;
  }

  if (!is_support && dim == 3)
  {
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
    else if (input.is_contiguous() && other.is_contiguous())
    {
      is_support = false;
      if (!is_support && other.size(0) == 1)
      {
        is_support = true;
        is_support &= input.dim() == other.dim();
        is_support &= input.size(0) > 1;
        is_support &= input.size(1) == other.size(1);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_0 : 0;
        order_flag = true;
      }

      if (!is_support && input.size(0) == 1)
      {
        is_support = true;
        is_support &= input.dim() == other.dim();
        is_support &= other.size(0) > 1;
        is_support &= input.size(1) == other.size(1);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_0 : 0;
        order_flag = false;
      }

      if (!is_support && input.size(1) == 1)
      {
        is_support = true;
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
        is_support &= input.dim() == other.dim();
        is_support &= input.size(1) > 1;
        is_support &= input.size(0) == other.size(0);
        is_support &= input.size(2) == other.size(2);
        pattern = is_support ? PATTERN_BROADCAST_1 : 0;
        order_flag = false;
      }
    }
  }

  if (is_support)
  {
    auto in0_dtype = input.dtype();
    auto in1_dtype = other.dtype();
    torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
    std::vector<int64_t> out_shape = broadcastShapes(input, other);
    auto device = input.device();
    auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());
    auto output = torch::empty(out_shape, options);

    torch::Tensor input0 = in0_dtype == out_dtype ? input : input.to(out_dtype);
    torch::Tensor other0 = in1_dtype == out_dtype ? other : other.to(out_dtype);

    void *buf_a = reinterpret_cast<void *>(input0.data_ptr());
    void *buf_b = reinterpret_cast<void *>(other0.data_ptr());
    void *buf_c = reinterpret_cast<void *>(output.data_ptr());

    int elementSize = output.element_size();
    if (elementSize == 1)
    {
      binary_operation_vec<Operation, 16>(input0, other0, output, pattern, order_flag);
    }
    else if (elementSize == 2)
    {
      binary_operation_vec<Operation, 8>(input0, other0, output, pattern, order_flag);
    }
    else if (elementSize == 4)
    {
      binary_operation_vec<Operation, 4>(input0, other0, output, pattern, order_flag);
    }
    else if (elementSize == 8)
    {
      binary_operation_vec<Operation, 2>(input0, other0, output, pattern, order_flag);
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
  return binary_operation<aiter::AddOp>(input, other);
}

torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::SubOp>(input, other);
}

torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::MulOp>(input, other);
}

torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other)
{
  return binary_operation<aiter::DivOp>(input, other);
}

// inline at::Tensor add(const at::Tensor & self, const at::Scalar & other) {
// }
