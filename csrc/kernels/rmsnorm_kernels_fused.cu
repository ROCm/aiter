#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include "hip_reduce.h"
#include <hipcub/hipcub.hpp>
#include "aiter_hip_common.h"
#include "vec_convert.h"

struct RMSNormParameter
{
    void* p_out;
    p2 _p0;
    void* p_input;
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

struct NoFusedRMSNormParameter
{
    void* p_out;
    void* p_input;
    void* p_weight;

    int32_t stride;
    float epsilon;
};


template <typename DTYPE,
          int HIDDEN_SIZE,
          int WIDTH,
          int blockDim,
          // bool RESIDUAL_OUT,
          // bool DO_SMOOTH_QUANT,
          // bool NO_QUANT_OUT,
          typename ACC_DTYPE,
          typename QUANT_DTYPE>
__global__ void fused_add_smooth_quant_rms_norm_kernel(RMSNormParameter params)
{
  // Sanity checks on our vector struct and type-punned pointer arithmetic
    static constexpr int LANE_HIDDEN_SIZE = HIDDEN_SIZE / blockDim;                   // 5120 / 128 = 80 
    static constexpr int VEC_HIDDEN_SIZE_LOC = LANE_HIDDEN_SIZE / WIDTH;        // 80 / 8 = 10

    static constexpr int WARP_GROUP = blockDim / 64;        // 80 / 8 = 10

    auto arg_sum = [](const float& a, const float& b) {
        return a + b;
    };

    auto arg_max = [](const float& a, const float& b) {
        return ck_tile::max(a, b);
    };

    using AccessType = ck_tile::vec_t<DTYPE, WIDTH>;
    using AccVecType = ck_tile::vec_t<float, WIDTH>;
    using StoreType  = ck_tile::vec_t<ck_tile::bf16_t, WIDTH>;
    using StoreQuantType = ck_tile::vec_t<QUANT_DTYPE, WIDTH>;
    using QuantVecType = ck_tile::vec_t<QUANT_DTYPE, WIDTH>;

    float variance = 0.0f;
    float max_local = 0.0f;

    __shared__ float s_variance[WARP_GROUP];
    __shared__ float s_max_local[WARP_GROUP];

    // const int cta_base_row = blockIdx.x * HIDDEN_SIZE * BlockDim.y;
    // const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int warp_base_row = blockIdx.x;

    constexpr int THREADS_PER_ROW = 128;
    // constexpr int THREADS_PER_ROW = 64;
    // const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    // const int thread_row         = warp_base_row + thread_row_in_warp;
    // const int row_offset = thread_row * HIDDEN_SIZE;

    const int row_offset = warp_base_row * HIDDEN_SIZE;

    const DTYPE* warp_in_ptr = reinterpret_cast<DTYPE*>(params.p_input) + row_offset;
    const DTYPE* warp_residual_in_ptr = reinterpret_cast<DTYPE*>(params.p_residual_in) + row_offset;

    const int first_elt_read_by_thread = threadIdx.x * WIDTH;

    // Input init
    const DTYPE* thread_in_ptr = warp_in_ptr + first_elt_read_by_thread;
    const DTYPE* thread_residual_in_ptr = warp_residual_in_ptr + first_elt_read_by_thread;

    float in_local[LANE_HIDDEN_SIZE];
    float residual_in_local[LANE_HIDDEN_SIZE];
    AccVecType* row_in_ptr           = reinterpret_cast<AccVecType*>(&in_local);
    const AccessType* vec_in_read_ptr = reinterpret_cast<const AccessType*>(thread_in_ptr);

    AccVecType* row_resdidual_in_ptr = reinterpret_cast<AccVecType*>(&residual_in_local);
    const AccessType* vec_residual_in_read_ptr = reinterpret_cast<const AccessType*>(thread_residual_in_ptr);

#pragma unroll
    for(int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        row_in_ptr[ii] = ck_tile::vec_convert<ACC_DTYPE, DTYPE, WIDTH>(
            vec_in_read_ptr[ii * THREADS_PER_ROW]);

        row_resdidual_in_ptr[ii] = ck_tile::vec_convert<ACC_DTYPE, DTYPE, WIDTH>(
            vec_residual_in_read_ptr[ii * THREADS_PER_ROW]);
    }

    // Weight init
    const DTYPE* thread_gamma_ptr = reinterpret_cast<DTYPE*>(params.p_weight) + first_elt_read_by_thread;
    DTYPE gamma_local[LANE_HIDDEN_SIZE];
    AccessType* gamma_in_ptr = reinterpret_cast<AccessType*>(&gamma_local);
    const AccessType* vec_gamma_read_ptr = reinterpret_cast<const AccessType*>(thread_gamma_ptr);


    // sm_scale init
    const ACC_DTYPE* thread_sm_scale_ptr = reinterpret_cast<ACC_DTYPE*>(params.p_x_scale) + first_elt_read_by_thread;
    float sm_scale_local[LANE_HIDDEN_SIZE];
    AccVecType* sm_scale_in_ptr = reinterpret_cast<AccVecType*>(&sm_scale_local);
    const AccVecType* vec_sm_scale_read_ptr = reinterpret_cast<const AccVecType*>(thread_sm_scale_ptr);


    auto * thread_residual_add_ptr = reinterpret_cast<ck_tile::bf16_t*>(params.p_residual_out) + row_offset + first_elt_read_by_thread;
    auto * thread_no_quant_ptr = reinterpret_cast<ck_tile::bf16_t*>(params.p_out_before_quant) + row_offset + first_elt_read_by_thread;

    StoreType* vec_residual_add_st_ptr = reinterpret_cast<StoreType*>(thread_residual_add_ptr);
    StoreType* vec_no_quant_st_ptr = reinterpret_cast<StoreType*>(thread_no_quant_ptr);

    static constexpr auto WIDTH_SUB = WIDTH - 1;

    // Issue gamma with add.
#pragma unroll
    for(int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        // if(threadIdx.x == 0 && blockIdx.x == 1)
            // printf("ii %d in_local %f, in_res_local %f \n", ii, in_local[ii], residual_in_local[ii]);
        gamma_in_ptr[ii] = vec_gamma_read_ptr[ii * THREADS_PER_ROW];
#pragma unroll
        for (int j = 0; j < WIDTH; ++j)
        {
            auto idx = ii * WIDTH + j;
            in_local[idx] += residual_in_local[idx];
            variance += in_local[idx] * in_local[idx];
        }
        // vec_residual_add_st_ptr[ii * THREADS_PER_ROW] = ck_tile::vec_convert<ck_tile::bf16_t, ACC_DTYPE, WIDTH>(row_in_ptr[ii]);
    }

    variance = multithread_reduce(variance, arg_sum, 64);

    auto warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);

    s_variance[warp_id] = variance;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        auto variance = s_variance[0] + s_variance[1];
        s_variance[0] = variance;
        s_variance[1] = variance;
    }

    __syncthreads();

    variance = s_variance[warp_id];

#pragma unroll
    for (int ii= 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        vec_residual_add_st_ptr[ii * THREADS_PER_ROW] = ck_tile::vec_convert<ck_tile::bf16_t, ACC_DTYPE, WIDTH>(row_in_ptr[ii]);
        sm_scale_in_ptr[ii] = vec_sm_scale_read_ptr[ii * THREADS_PER_ROW];
    }

    float scale_rms = rsqrtf(variance / HIDDEN_SIZE + params.epsilon);
#pragma unroll
    for (int ii = 0; ii < LANE_HIDDEN_SIZE; ++ii)
    {
        residual_in_local[ii] = in_local[ii] * scale_rms * ck_tile::type_convert<ACC_DTYPE>(gamma_local[ii]);
    }

#pragma unroll
    for (int ii= 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        vec_no_quant_st_ptr[ii * THREADS_PER_ROW] = ck_tile::vec_convert<ck_tile::bf16_t, ACC_DTYPE, WIDTH>(row_resdidual_in_ptr[ii]);
    }

    const auto f_max3 = [](auto acc_, auto v_0_, auto v_1_) {
        float rtn;
        asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                     : "=v"(rtn)
                     : "v"(acc_), "v"(v_0_), "v"(v_1_));
        return rtn;
    };

    const auto max_scale = __builtin_amdgcn_rcpf(ck_tile::type_convert<ACC_DTYPE>(ck_tile::numeric<QUANT_DTYPE>::max()));

#pragma unroll
    for (int ii = 0; ii < LANE_HIDDEN_SIZE; ii += 2)
    {
        in_local[ii] = residual_in_local[ii] * sm_scale_local[ii];
        in_local[ii + 1] = residual_in_local[ii + 1] * sm_scale_local[ii + 1];

        max_local = f_max3(max_local, in_local[ii], in_local[ii + 1]);
    }

    max_local = multithread_reduce(max_local, arg_max, 64);

    s_max_local[warp_id] = max_local;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        auto max_local = ck_tile::max(s_max_local[0], s_max_local[1]);
        s_max_local[0] = max_local;
        s_max_local[1] = max_local;
    }
    __syncthreads();

    max_local = s_max_local[warp_id];

    auto * thread_out_ptr = reinterpret_cast<QUANT_DTYPE*>(params.p_out) + row_offset + first_elt_read_by_thread;
    StoreQuantType* vec_out_st_ptr = reinterpret_cast<StoreQuantType*>(thread_out_ptr);

    auto y_scale = max_scale * max_local;


    reinterpret_cast<float *>(params.p_y_scale)[warp_base_row] = y_scale;

    auto r_y_scale = __builtin_amdgcn_rcpf(y_scale);

#pragma unroll
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
#pragma unroll
        for (int j = 0; j < WIDTH; ++j)
        {
            in_local[ii * WIDTH + j] *= r_y_scale;
        }
        vec_out_st_ptr[ii * THREADS_PER_ROW] = ck_tile::vec_convert<QUANT_DTYPE, ACC_DTYPE, WIDTH>(row_in_ptr[ii]);
    }
}

template <typename DTYPE,
          int HIDDEN_SIZE,
          int WIDTH,
          int ROW_WIDTH,
          int blockDim,
          // bool RESIDUAL_OUT,
          // bool DO_SMOOTH_QUANT,
          // bool NO_QUANT_OUT,
          typename ACC_DTYPE,
          typename QUANT_DTYPE>
__global__ void no_fused_rms_norm_kernel(NoFusedRMSNormParameter params)
{
  // Sanity checks on our vector struct and type-punned pointer arithmetic
    static constexpr int LANE_HIDDEN_SIZE = HIDDEN_SIZE / blockDim;                   // 5120 / 128 = 80 
    static constexpr int VEC_HIDDEN_SIZE_LOC = LANE_HIDDEN_SIZE / WIDTH;        // 80 / 8 = 10

    static constexpr int WARP_GROUP = blockDim / 64;        // 80 / 8 = 10

    auto arg_sum = [](const float& a, const float& b) {
        return a + b;
    };

    auto arg_max = [](const float& a, const float& b) {
        return ck_tile::max(a, b);
    };

    using AccessType = ck_tile::vec_t<DTYPE, WIDTH>;
    using AccVecType = ck_tile::vec_t<float, WIDTH>;
    using StoreType  = ck_tile::vec_t<ck_tile::bf16_t, WIDTH>;
    using StoreQuantType = ck_tile::vec_t<QUANT_DTYPE, WIDTH>;
    using QuantVecType = ck_tile::vec_t<QUANT_DTYPE, WIDTH>;

    __shared__ float s_variance[WARP_GROUP];
    __shared__ float s_max_local[WARP_GROUP];

    // const int cta_base_row = blockIdx.x * HIDDEN_SIZE * BlockDim.y;
    // const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int warp_base_row = blockIdx.x * ROW_WIDTH;

    constexpr int THREADS_PER_ROW = 128;
    // constexpr int THREADS_PER_ROW = 64;
    // const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    // const int thread_row         = warp_base_row + thread_row_in_warp;
    // const int row_offset = thread_row * HIDDEN_SIZE;

    const DTYPE* warp_in_ptr = reinterpret_cast<DTYPE*>(params.p_input) + warp_base_row * params.stride;

    const int first_elt_read_by_thread = threadIdx.x * WIDTH;

    // Input init
    const DTYPE* thread_in_ptr = warp_in_ptr + first_elt_read_by_thread;

    DTYPE in_local_b16[LANE_HIDDEN_SIZE];
    float in_local[LANE_HIDDEN_SIZE];
    AccVecType* row_in_ptr           = reinterpret_cast<AccVecType*>(&in_local);
    const AccessType* vec_in_read_ptr = reinterpret_cast<const AccessType*>(thread_in_ptr);

    AccessType* row_in_b16_ptr = reinterpret_cast<AccessType*>(&row_in_b16_ptr);

    auto * thread_out_ptr = reinterpret_cast<QUANT_DTYPE*>(params.p_out) + warp_base_row * HIDDEN_SIZE + first_elt_read_by_thread;
    StoreQuantType* vec_out_st_ptr = reinterpret_cast<StoreQuantType*>(thread_out_ptr);

    // Weight init
    const DTYPE* thread_gamma_ptr = reinterpret_cast<DTYPE*>(params.p_weight) + first_elt_read_by_thread;
    DTYPE gamma_local[LANE_HIDDEN_SIZE];
    AccessType* gamma_in_ptr = reinterpret_cast<AccessType*>(&gamma_local);
    const AccessType* vec_gamma_read_ptr = reinterpret_cast<const AccessType*>(thread_gamma_ptr);

    float r_dim_scale = __builtin_amdgcn_rcpf(HIDDEN_SIZE);
#pragma unroll  // Unroll loop for better instruction pipelining
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
#pragma unroll
        for (int j = 0; j < WIDTH; ++j)
        {
            in_local_b16[ii * WIDTH + j] = __builtin_nontemporal_load(thread_in_ptr + ii * blockDim * WIDTH + j);
        }
    }

    // Issue gamma with add.
#pragma unroll
    for(int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        gamma_in_ptr[ii] = vec_gamma_read_ptr[ii * THREADS_PER_ROW];
    }

#pragma unroll
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        row_in_ptr[ii] = ck_tile::vec_convert<ACC_DTYPE, DTYPE, WIDTH>(row_in_b16_ptr[ii]);
    }

#pragma unroll  // Unroll loop for better instruction pipelining
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
#pragma unroll
        for (int j = 0; j < WIDTH; ++j)
        {
            in_local_b16[ii * WIDTH + j] = __builtin_nontemporal_load(thread_in_ptr + ii * blockDim * WIDTH + j + params.stride);
        }
    }

#pragma unroll
    for (int r = 0; r < ROW_WIDTH; ++r)
    {
        float variance = 0.0f;
        float max_local = 0.0f;

#pragma unroll
        for(int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
        {
#pragma unroll
            for (int j = 0; j < WIDTH; ++j)
            {
                auto idx = ii * WIDTH + j;
                variance += in_local[idx + r * LANE_HIDDEN_SIZE] * in_local[idx + r * LANE_HIDDEN_SIZE];
            }
        }

        variance = multithread_reduce(variance, arg_sum, 64);

        auto warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);

        s_variance[warp_id] = variance;
        __syncthreads();

        if (threadIdx.x == 0)
        {
            auto variance_local = s_variance[0];
#pragma unroll
            for (int i = 1; i < WARP_GROUP; ++i)
            {
                variance_local += s_variance[i];
            }
#pragma unroll
            for (int i = 0; i < WARP_GROUP; ++i)
            {
                s_variance[i] = variance_local;
            }
        }

        __syncthreads();

        variance = s_variance[warp_id];

        float scale_rms = rsqrtf(variance * r_dim_scale + params.epsilon);

        if (r == 0)
        {
#pragma unroll
            for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
            {
                row_in_ptr[ii + VEC_HIDDEN_SIZE_LOC] = ck_tile::vec_convert<ACC_DTYPE, DTYPE, WIDTH>(row_in_b16_ptr[ii]);
            }
        }

#pragma unroll
        for(int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
        {
#pragma unroll
            for (int j = 0; j < WIDTH; ++j)
            {
                auto idx = ii * WIDTH + j;
                in_local[idx] = in_local[idx] * scale_rms * ck_tile::type_convert<ACC_DTYPE>(gamma_local[idx]);
            }
            row_in_b16_ptr[ii] = ck_tile::vec_convert<QUANT_DTYPE, ACC_DTYPE, WIDTH>(row_in_ptr[ii + r * VEC_HIDDEN_SIZE_LOC]);
        }

#pragma unroll  // Unroll loop for better instruction pipelining
        for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
        {
#pragma unroll
            for (int j = 0; j < WIDTH; ++j)
            {
                __builtin_nontemporal_store(in_local_b16[ii * WIDTH + j], thread_out_ptr + ii * blockDim * WIDTH + j + r * HIDDEN_SIZE);
            }
        }
    }
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

  fused_add_smooth_quant_rms_norm_kernel<ck_tile::bf16_t, 5120, 8, 128, float, int8_t><<<grid, block, 0, stream>>>(params);
  // fused_add_smooth_quant_rms_norm_kernel<__hip_bfloat16, 5120, 8, 128, float, int8_t><<<grid, block, 0, stream>>>(params);
}


torch::Tensor
rmsnorm2d_hip(torch::Tensor& input,
              torch::Tensor& weight,
              double epsilon,
              int use_model_sensitive_rmsnorm = 0)
{
    torch::Tensor out = torch::empty_like(input);

    int hidden_size = input.size(-1);
    int num_tokens = input.size(0);

    constexpr int row_block = 2;

    dim3 grid(num_tokens / row_block);
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

    NoFusedRMSNormParameter params;
    params.p_out = out.data_ptr();
    params.p_input = input.data_ptr();
    params.p_weight = weight.data_ptr();
    params.epsilon = epsilon;
    params.stride = input.stride(0);

    no_fused_rms_norm_kernel<ck_tile::bf16_t, 8192, 8, row_block, 256, float, ck_tile::bf16_t><<<grid, block, 0, stream>>>(params);

    return out;
}
