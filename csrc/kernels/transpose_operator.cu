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
  __global__ void add_tn_big_tile_kernel(const void *__restrict a, const void *__restrict b, void *__restrict c, const int N, const int K, int stride0, int stride2)
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
}

template <typename Operation>
torch::Tensor transpose_operation(torch::Tensor &input, torch::Tensor &other)
{
  int dim = input.dim();

  // if (dim == 3 && (!input.is_contiguous() || !other.is_contiguous()))
  // {
  //   Operation::print(input.size(0), input.size(1), input.size(2),
  //                    input.stride(0), input.stride(1), input.stride(2),
  //                    other.stride(0), other.stride(1), other.stride(2));
  // }
  bool is_support = input.is_contiguous() != other.is_contiguous();
  auto tensor_not_conti = input.is_contiguous() ? other : input;
  bool order_flag = !input.is_contiguous() ? true : false;
  is_support &= input.dim() == other.dim();
  is_support &= dim == 3;
  int M = input.size(0), N = input.size(1), K = input.size(2);
  int stride0, stride1, stride2;
  if (is_support)
  {
    // avoid broadcast
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
  }

  if (is_support)
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
          input.scalar_type(), "add_tn_big_tile_kernel", [&]
          { vllm::add_tn_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, true>
                <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_b, buf_c, K, N, stride0, stride2); });
    }
    else
    {
      VLLM_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "add_tn_big_tile_kernel", [&]
          { vllm::add_tn_big_tile_kernel<scalar_t, 256, BIG_TILE_SIZE_N, BIG_TILE_SIZE_K, M_SWIZZLE, Operation, false>
                <<<grid_dim, block_dim, 0, stream>>>(buf_b, buf_a, buf_c, K, N, stride0, stride2); });
    }

    return output;
  }
  else
  {
    return vllm::aten_compute<Operation>(input, other);
  }
}

torch::Tensor transpose_add(torch::Tensor &input, torch::Tensor &other)
{
  return transpose_operation<vllm::AddOp>(input, other);
}

torch::Tensor transpose_sub(torch::Tensor &input, torch::Tensor &other)
{
  return transpose_operation<vllm::SubOp>(input, other);
}

torch::Tensor transpose_mul(torch::Tensor &input, torch::Tensor &other)
{
  return transpose_operation<vllm::MulOp>(input, other);
}

torch::Tensor transpose_div(torch::Tensor &input, torch::Tensor &other)
{
  return transpose_operation<vllm::DivOp>(input, other);
}
