#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipcub/util_type.hpp>
#include "hip_reduce.h"
#include <hipcub/hipcub.hpp>
#include "aiter_hip_common.h"
#include "vec_convert.h"

struct SoftmaxParameter
{
    void* p_out;
    p2 _p0;
    void* p_input;
    p2 _p1;

    int32_t stride;
};


template <typename DTYPE,       // Input data type (e.g., float, bf16_t)
          int HIDDEN_SIZE,     // Size of the dimension over which softmax is computed (e.g., 768, 1024)
          int WIDTH,           // Vector width for memory access (e.g., 8, 16; improves memory throughput)
          int blockDim,        // Number of threads per block (e.g., 128, 256)
          typename ACC_DTYPE,  // Accumulation type for intermediate calculations (typically float for precision)
          typename QUANT_DTYPE>// Output quantization type (e.g., bf16_t, fp16_t)
__global__ void no_fused_softmax_kernel(SoftmaxParameter params)
{
    // Thread-to-data mapping constants
    // Number of elements processed per thread in the hidden dimension
    static constexpr int LANE_HIDDEN_SIZE = HIDDEN_SIZE / blockDim;
    // Number of vectorized chunks per thread (derived from vector width WIDTH)
    static constexpr int VEC_HIDDEN_SIZE_LOC = LANE_HIDDEN_SIZE / WIDTH;

    // Warp configuration (AMD GPUs typically use 64-thread warps)
    static constexpr int WARP_SIZE = 64;
    // Number of warps in the thread block
    static constexpr int WARP_GROUP = blockDim / WARP_SIZE;

    static constexpr float LOG2E = 1.4426950408889634;

    // Reduction operators for max and sum (used in parallel reduction)
    auto arg_max = [](const ACC_DTYPE& a, const ACC_DTYPE& b) { return ck_tile::max(a, b); };
    auto arg_sum = [](const ACC_DTYPE& a, const ACC_DTYPE& b) { return a + b; };

    // Vector types for efficient memory access and computation
    using AccessType = ck_tile::vec_t<DTYPE, WIDTH>;          // Vector type for loading input data
    using AccVecType = ck_tile::vec_t<ACC_DTYPE, WIDTH>;      // Vector type for intermediate calculations
    using StoreType = ck_tile::vec_t<QUANT_DTYPE, WIDTH>;     // Vector type for storing output data

    // Shared memory to hold intermediate max and sum values across warps
    __shared__ ACC_DTYPE s_max[WARP_GROUP];   // Max values per warp
    __shared__ ACC_DTYPE s_sum[WARP_GROUP];   // Sum values per warp

    // Calculate row index and memory offsets for input/output
    const int row_idx = blockIdx.x;                       // Current row processed by the block
    const int row_offset_in = row_idx * params.stride;    // Input memory offset for the current row (accounting for stride)
    const int row_offset_out = row_idx * HIDDEN_SIZE;     // Output memory offset for the current row

    // Input data pointers: map global memory to thread-local vector pointers
    const DTYPE* row_in_ptr = reinterpret_cast<const DTYPE*>(params.p_input) + row_offset_in;
    const int first_elt = threadIdx.x * WIDTH;            // First element index processed by this thread
    const AccessType* vec_in_ptr = reinterpret_cast<const AccessType*>(row_in_ptr + first_elt);

    // Local memory to hold input elements for this thread (faster than global memory)
    DTYPE in_local_b16[LANE_HIDDEN_SIZE];
    ACC_DTYPE in_local[LANE_HIDDEN_SIZE];
    AccVecType* acc_in_ptr = reinterpret_cast<AccVecType*>(in_local);  // Vector view of local memory
    AccessType* acc_in_b16_ptr = reinterpret_cast<AccessType*>(in_local_b16);  // Vector view of local memory

    // 1. Load input data from global memory to thread-local memory (vectorized load)
#pragma unroll  // Unroll loop for better instruction pipelining
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        // Load vector from global memory and convert to accumulation type
        // acc_in_ptr[ii] = ck_tile::vec_convert<ACC_DTYPE, DTYPE, WIDTH>(
        //     vec_in_ptr[ii * blockDim])  // Offset by thread count to avoid overlap
        // );
#pragma unroll
        for (int j = 0; j < WIDTH; ++j)
        {
            in_local_b16[ii * WIDTH + j] = __builtin_nontemporal_load(row_in_ptr + first_elt + ii * blockDim * WIDTH + j);
        }
    }

#pragma unroll
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        acc_in_ptr[ii] = ck_tile::vec_convert<ACC_DTYPE, DTYPE, WIDTH>(acc_in_b16_ptr[ii]);
    }

    // 2. Compute per-thread maximum value (for numerical stability in softmax)
    ACC_DTYPE thread_max = -std::numeric_limits<ACC_DTYPE>::max();  // Initialize to negative infinity
#pragma unroll
    for (int i = 0; i < LANE_HIDDEN_SIZE; ++i)
    {
        thread_max = ck_tile::max(thread_max, in_local[i]);  // Track max in thread's local elements
    }

    // Reduce max across threads in the warp
    ACC_DTYPE warp_max = multithread_reduce(thread_max, arg_max, WARP_SIZE);
    const int warp_id = threadIdx.x / WARP_SIZE;  // Current warp index in the block

    // Store warp-level max to shared memory (only first thread in warp does this)
    if (threadIdx.x % WARP_SIZE == 0)
    {
        s_max[warp_id] = warp_max;
    }
    __syncthreads();  // Sync to ensure all warps have written their max

    // Compute global max across all warps in the block (handled by thread 0)
    ACC_DTYPE global_max = s_max[0];
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int i = 1; i < WARP_GROUP; ++i)
        {
            global_max = ck_tile::max(global_max, s_max[i]);
        }
        // Broadcast global max to all shared memory slots for easy access
#pragma unroll
        for (int i = 0; i < WARP_GROUP; ++i)
        {
            s_max[i] = global_max;
        }
    }
    __syncthreads();  // Sync to ensure global max is broadcast
    global_max = s_max[warp_id];  // Each warp loads the global max

    // 3. Compute exp(x - global_max) (avoids overflow) and accumulate sums
    ACC_DTYPE thread_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < LANE_HIDDEN_SIZE; ++i)
    {
        in_local[i] = ck_tile::exp2((in_local[i] - global_max) * LOG2E);  // Stabilized exp calculation
        thread_sum += in_local[i];  // Accumulate sum of exponentials
    }

    // 4. Reduce sum across threads in the warp
    ACC_DTYPE warp_sum = multithread_reduce(thread_sum, arg_sum, WARP_SIZE);

    // Store warp-level sum to shared memory (only first thread in warp does this)
    if (threadIdx.x % WARP_SIZE == 0)
    {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();  // Sync to ensure all warps have written their sum

    // Compute global sum across all warps in the block (handled by thread 0)
    ACC_DTYPE global_sum = s_sum[0];
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int i = 1; i < WARP_GROUP; ++i)
        {
            global_sum += s_sum[i];
        }
        // Broadcast global sum to all shared memory slots
#pragma unroll
        for (int i = 0; i < WARP_GROUP; ++i)
        {
            s_sum[i] = global_sum;
        }
    }
    __syncthreads();  // Sync to ensure global sum is broadcast
    global_sum = s_sum[warp_id];  // Each warp loads the global sum

    // 5. Compute final softmax: exp(x - max) / sum(exp(x - max))
    ACC_DTYPE inv_sum = 1.0f / global_sum;  // Precompute inverse sum to avoid repeated division
#pragma unroll
    for (int i = 0; i < LANE_HIDDEN_SIZE; ++i)
    {
        in_local[i] *= inv_sum;  // Softmax normalization
    }

    // 6. Store results to global memory (vectorized store)
    QUANT_DTYPE* row_out_ptr = reinterpret_cast<QUANT_DTYPE*>(params.p_out) + row_offset_out;
    StoreType* vec_out_ptr = reinterpret_cast<StoreType*>(row_out_ptr + first_elt);

#pragma unroll
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
        acc_in_b16_ptr[ii] = ck_tile::vec_convert<QUANT_DTYPE, ACC_DTYPE, WIDTH>(acc_in_ptr[ii]);
    }

// #pragma unroll
//     for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
//     {
//         // Convert to output type and store vector to global memory
//         // vec_out_ptr[ii * blockDim] = ck_tile::vec_convert<QUANT_DTYPE, ACC_DTYPE, WIDTH>(
//         //     acc_in_ptr[ii]
//         // );
//         __builtin_nontemporal_store(ck_tile::vec_convert<QUANT_DTYPE, ACC_DTYPE, WIDTH>(acc_in_ptr[ii]), vec_out_ptr + ii * blockDim);
//     }
#pragma unroll  // Unroll loop for better instruction pipelining
    for (int ii = 0; ii < VEC_HIDDEN_SIZE_LOC; ++ii)
    {
#pragma unroll
        for (int j = 0; j < WIDTH; ++j)
        {
            __builtin_nontemporal_store(in_local_b16[ii * WIDTH + j], row_out_ptr + first_elt + ii * blockDim * WIDTH + j);
        }
    }
}

// void softmax2d_hip(
//     torch::Tensor& out,          // [m ,n]
//     torch::Tensor& input,        // [m ,n]
//     std::optional<torch::Tensor> out_before_quant,
//     int use_model_sensitive_rmsnorm = 0)
void softmax2d(torch::Tensor &out,    // [m, n]
               torch::Tensor &input)  // [m, n]
{
  int hidden_size = input.size(-1);
  int num_tokens = input.size(0);

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  // const int max_block_size = 128;
  const int max_block_size = 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
  const hipStream_t stream = at::hip::getCurrentHIPStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */

  SoftmaxParameter params;
  params.p_out = out.data_ptr();
  params.p_input = input.data_ptr();
  params.stride = input.stride(0);

  no_fused_softmax_kernel<ck_tile::bf16_t, 8192, 8, 256, float, ck_tile::bf16_t><<<grid, block, 0, stream>>>(params);
}

torch::Tensor softmax2d_hip(torch::Tensor &input) // [m, n]
{
    torch::Tensor out = torch::empty_like(input);
    softmax2d(out, input);

    return out;
}

