#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>
#include "ck_tile/core.hpp"
#include "aiter_hip_common.h"

struct RMSNormParameter
{
    void* p_out;
    p2 _p0;
    void* p_input;
    p2 _p1;
    void* p_residual_in;
    p2 _p2;
    void* p_residual_out;
    p2 _p3;
    void* p_x_scale;
    p2 _p4;
    void* p_y_scale;
    p2 _p5;
    void* p_weight;
    p2 _p6;
    void* p_out_before_quant;
    p2 _p7;

    float epsilon;
};

struct __attribute__((packed)) buffer_resource
{
    const void* ptr;
    uint32_t range;
    uint32_t config;
};

// using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));
// using bf16x8_t  = __hip_bfloat16 __attribute__((ext_vector_type(8)));

__device__ ck_tile::int32x4_t make_wave_buffer_resource(const void* ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
    ck_tile::int32x4_t r = __builtin_bit_cast(ck_tile::int32x4_t, res);
    r.x         = __builtin_amdgcn_readfirstlane(r.x);
    r.y         = __builtin_amdgcn_readfirstlane(r.y);
    r.z         = __builtin_amdgcn_readfirstlane(r.z);
    r.w         = __builtin_amdgcn_readfirstlane(r.w);
    return r;
}

// CK_TILE_DEVICE __amdgpu_buffer_rsrc_t cast_to_amdgpu_buffer_rsrc_t(ck_tile::int32x4_t res)
// {
//     __amdgpu_buffer_rsrc_t as_rsrc;
//     memcpy(&as_rsrc, &res, sizeof(res));
//     return as_rsrc;
// }

__device__ __inline__ ck_tile::bf16x8_t amd_buffer_load_raw(void *buffer, int v_offset, int i_offset)
{
    auto res = make_wave_buffer_resource(buffer);
    ck_tile::fp32x4_t tmp;
    asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                 : "+v"(tmp)
                 : "v"(v_offset), "s"(res), "n"(i_offset)
                 : "memory");
	return ck_tile::bit_cast<ck_tile::bf16x8_t>(tmp);
}



template <typename dtype>
struct _typeConvert
{
    static constexpr bool exists = false;
};

template <>
struct _typeConvert<__hip_bfloat16>
{
	static constexpr bool exists = true;
	using hip_type = __nv_bfloat16;
	using packed_hip_type = __nv_bfloat162;

	__device__ static inline float convert(hip_type x)
	{
	  return __bfloat162float(x);
	}
	__device__ static inline float2 convert(packed_hip_type x)
	{
	  return __bfloat1622float2(x);
	}
	__device__ static inline hip_type convert(float x)
	{
	  return __float2bfloat16(x);
	}
	__device__ static inline packed_hip_type convert(float2 x)
	{
	  return __float22bfloat162_rn(x);
	}
};

template <>
struct _typeConvert<float>
{
	static constexpr bool exists = true;
	using hip_type = float;
	using packed_hip_type = float2;
	__device__ static inline float convert(float x)
	{
	  return x;
	}
	__device__ static inline float2 convert(float2 x)
	{
	  return x;
	}
};

template <typename scalar_t, int width>
struct alignas(16) arr_t
{
    /* Not theoretically necessary that width is a power of 2 but should
       almost always be the case for optimization purposes */
    static_assert(width > 0 && (width & (width - 1)) == 0,
                  "Width is not a positive power of 2!");
    using Converter = _typeConvert<scalar_t>;
    using T1 = typename Converter::hip_type;
    using T2 = typename Converter::packed_hip_type;
    T1 data[width];

    __device__ arr_t &operator+=(const arr_t<scalar_t, width> &other)
    {
      if constexpr (width % 2 == 0)
      {
#pragma unroll
        for (int i = 0; i < width; i += 2)
        {
          T2 temp{data[i], data[i + 1]};
          temp += T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < width; ++i)
          data[i] += other.data[i];
      }
      return *this;
    }

    __device__ arr_t &operator*=(const arr_t<scalar_t, width> &other)
    {
      if constexpr (width % 2 == 0)
      {
#pragma unroll
        for (int i = 0; i < width; i += 2)
        {
          T2 temp{data[i], data[i + 1]};
          temp *= T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < width; ++i)
          data[i] *= other.data[i];
      }
      return *this;
    }

    __device__ arr_t &operator*=(const float scale)
    {
      if constexpr (width % 2 == 0)
      {
#pragma unroll
        for (int i = 0; i < width; i += 2)
        {
          float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
          temp_f.x *= scale;
          temp_f.y *= scale;
          T2 temp = Converter::convert(temp_f);
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < width; ++i)
        {
          float temp = Converter::convert(data[i]) * scale;
          data[i] = Converter::convert(temp);
        }
      }
      return *this;
    }

    __device__ void to_float(arr_t<float, width>& ret)
    {
      if constexpr (width % 2 == 0)
      {
#pragma unroll
        for (int i = 0; i < width; i += 2)
        {
          float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
          ret.data[i] = temp_f.x;
          ret.data[i + 1] = temp_f.y;
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < width; ++i)
        {
          ret.data[i] = Converter::convert(data[i]);
        }
      }
    }

    __device__ float sum_squares() const
    {
      float result = 0.0f;
      if constexpr (width % 2 == 0)
      {
#pragma unroll
        for (int i = 0; i < width; i += 2)
        {
          float2 z = Converter::convert(T2{data[i], data[i + 1]});
          result += z.x * z.x + z.y * z.y;
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < width; ++i)
        {
          float x = Converter::convert(data[i]);
          result += x * x;
        }
      }
      return result;
    }

    __device__ float max() const
    {

      const auto f_max3 = [](auto acc_, auto v_0_, auto v_1_) {
        float rtn;
        asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                     : "=v"(rtn)
                     : "v"(acc_), "v"(v_0_), "v"(v_1_));
        return rtn;
      };
      float result = 0.0f;
      if constexpr (width % 2 == 0)
      {
#pragma unroll
        for (int i = 0; i < width; i += 2)
        {
          float2 z = Converter::convert(T2{data[i], data[i + 1]});
          result = f_max3(result, z.x, z.y);
        }
      }
      else
      {
#pragma unroll
        for (int i = 0; i < width; ++i)
        {
          float x = Converter::convert(data[i]);
          result = max(result, x);
        }
      }
      return result;
    }
};



// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t
{
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct ReduceFunctor
{
  __device__ __inline__ T operator() (T a, T b)
  {
    return a + b;
  }
};

template <typename T>
struct MaxFunctor
{
  __device__ __inline__ T operator() (T a, T b)
  {
    T zero_t = ck_tile::type_convert<T>(0.0f);
    a = a > zero_t ? a : zero_t - a;
    b = b > zero_t ? b : zero_t - b;
    return max(a, b);
  }
};

template <>
struct ReduceFunctor<float2>
{
  __device__ __inline__ float2 operator() (float2 a, float2 b)
  {
    // return max(a, b);
    float zero_t = ck_tile::type_convert<float>(0.0f);
    a.y = a.y > zero_t ? a.y : zero_t - a.y;
    b.y = b.y > zero_t ? b.y : zero_t - b.y;
    return {a.x + b.x, max(a.y, b.y)};
  }
};

template <template <typename> class functor, typename T, int reduce_range>
__device__ __inline__ T warpReduce(T val)
{
    auto op = functor<T>();
#pragma unroll
    for (int stride = reduce_range / 2; stride > 0; stride >>= 1)
    {
        T tmp = __shfl_xor(val, stride, reduce_range);
        val = op(val, tmp);
    }
    return val;
}

template <typename scalar_t,
          int hidden_size,
          int width,
          int blockDim,
          typename acc_dtype,
          typename quant_dtype>
__global__ void fused_add_smooth_quant_rms_norm_kernel(RMSNormParameter params)
{
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  constexpr int vec_hidden_size = hidden_size / width;                   // 5120 / 8 = 640
  constexpr int vec_hidden_size_loc = vec_hidden_size / blockDim;        // 640 / 128 = 5

  __shared__ float s_variance;
  __shared__ float s_quant_scale;

  // float2 reduce_data = {0.f, 0.f};
  float variance = 0.0f;
  float max_local = 0.0f;

  void* input_v = reinterpret_cast<void *>(params.p_input);
  void* residual_in = reinterpret_cast<void *>(params.p_residual_in);
  // void* residual_out = reinterpret_cast<void *>(params.p_residual_out);
  // auto *__restrict__ out_before_quant = reinterpret_cast<int32x4_t *>(params.p_out_before_quant);
  auto *weight_v = reinterpret_cast<ck_tile::fp32x8_t *>(params.p_weight);

  arr_t<scalar_t, width> residual_out_local[vec_hidden_size_loc];
  arr_t<scalar_t, width> residual_in_local[2];
  arr_t<acc_dtype, width> acc_local[vec_hidden_size_loc];

  int id = blockIdx.x * vec_hidden_size + threadIdx.x;
  reinterpret_cast<ck_tile::bf16x8_t*>(&(residual_out_local))[0] = amd_buffer_load_raw(input_v, id, 0);
  reinterpret_cast<ck_tile::bf16x8_t*>(&(residual_in_local))[0] = amd_buffer_load_raw(residual_in, id, 0); //load_ntmprl(&residual_in[id]);

  // reduce_data.x += residual_out_local[0].sum_squares();
  // reduce_data.y = residual_out_local[0].max();

  residual_out_local[0] += residual_in_local[0];

#pragma unroll
  for (int idx = 1; idx < vec_hidden_size_loc; idx++)
  {
    id += blockDim;

    reinterpret_cast<ck_tile::bf16x8_t*>(&(residual_out_local))[idx] = amd_buffer_load_raw(input_v, id, 0);// load_ntmprl(&input_v[id]);
	variance += residual_out_local[idx - 1].sum_squares();
    reinterpret_cast<ck_tile::bf16x8_t*>(&(residual_in_local))[idx & 1] = amd_buffer_load_raw(residual_in, id, 0); //load_ntmprl(&residual_in[id]);
    // reduce_data.y = residual_out_local[idx].max();

    residual_out_local[idx] += residual_in_local[idx & 1];
  }

  variance += residual_out_local[vec_hidden_size_loc - 1].sum_squares();

  // reduce_data = warpReduce<ReduceFunctor, float2, blockDim>(reduce_data);

  // variance = warpReduce<ReduceFunctor, float, blockDim>(variance);
//   __shared__ float s_sm_scaler;

  // if (threadIdx.x == 0)
  // {
  //   auto epsilon_local  = params.epsilon;
  //   variance = __builtin_amdgcn_rcpf(variance + epsilon_local);
  //   s_variance = __builtin_amdgcn_sqrtf(variance * dim_scale );
  // }
  // else if (threadIdx.x == 1)
  // {
  //    s_sm_scaler = reinterpret_cast<float*>(params.p_x_scale)[blockIdx.x];
  // }
  __syncthreads();

#pragma unroll
  for (int idx = 0; idx < vec_hidden_size_loc; idx++)
  {
    int id = blockIdx.x * vec_hidden_size + idx;
    reinterpret_cast<ck_tile::bf16x8_t*>(&(params.p_residual_out))[id] = reinterpret_cast<ck_tile::bf16x8_t*>(&(residual_out_local))[idx];
  }

  // float variance_local = s_variance;
  float variance_local = 1.0f; //s_variance;
#pragma unroll
  for (int idx = 0; idx < vec_hidden_size_loc; idx++)
  {
    int id = blockIdx.x * vec_hidden_size + idx;
    residual_out_local[idx].to_float(acc_local[idx]);

    acc_local[idx] *= variance_local;
    // residual_out_local[idx] *= weight_v[id];
    reinterpret_cast<ck_tile::fp32x8_t*>(&(params.p_out_before_quant))[id] = reinterpret_cast<ck_tile::fp32x8_t*>(&(acc_local))[idx];
  }

// #pragma unroll
//   for (int idx = 1; idx < vec_hidden_size_loc; idx++)
//   {
//     id += blockDim;
//     max_local = max(max_local, residual_out_local[idx].max());
//   }
//
//   max_local = warpReduce<MaxFunctor, float, blockDim>(max_local);
//
//
//   if (threadIdx.x == 0)
//   {
//     acc_dtype scale = max_local / ck_tile::numeric<acc_dtype>::max();
//     s_quant_scale = __builtin_amdgcn_rcpf(scale);
//     reinterpret_cast<float*>(params.p_y_scale)[blockIdx.x] = scale;
//   }
//   __syncthreads();


// #pragma unroll
  // for (int idx = 0; idx < vec_hidden_size_loc; idx++)
  // {
  //   int id = blockIdx.x * vec_hidden_size + idx;
  //   acc_local[idx] *= s_sm_scaler; 
  //   acc_local[idx] *= s_quant_scale; 
  //   for(int y = 0; y < width; ++y)
  //   {
  //     int index = id * width + y;
  //     reinterpret_cast<quant_dtype*>(params.p_out)[index] = ck_tile::type_convert<quant_dtype>(acc_local[id].data[y]);
  //   }
  // }
}

void rmsnorm2d_with_add_smoothquant_hip(
    torch::Tensor& out,          // [m ,n]
    torch::Tensor& input,        // [m ,n]
    torch::Tensor& residual_in,  // [m ,n]
    torch::Tensor& residual_out, // [m ,n]
    torch::Tensor& xscale,       // [1 ,n]
    torch::Tensor& yscale,       // [m ,1]
    torch::Tensor& weight,       // [1 ,n]
    double epsilon,
    std::optional<torch::Tensor> out_before_quant,
    int use_model_sensitive_rmsnorm = 0)
{
  int hidden_size = input.size(-1);
  int num_tokens = input.size(0);

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = 128;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */

  RMSNormParameter params;
  params.p_out = out.data_ptr();
  params.p_input = input.data_ptr();
  params.p_residual_in = residual_in.data_ptr();
  params.p_residual_out = residual_out.data_ptr();
  params.p_x_scale = xscale.data_ptr();
  params.p_y_scale = yscale.data_ptr();
  params.p_weight = weight.data_ptr();
  params.p_out_before_quant = out_before_quant.value().data_ptr();
  params.epsilon = epsilon;

  fused_add_smooth_quant_rms_norm_kernel<__hip_bfloat16, 5120, 8, 128, float, int8_t><<<grid, block, 0, stream>>>(params);
}
