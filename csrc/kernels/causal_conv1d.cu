// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// Based on Tri Dao's optimized causal_conv1d implementation

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <hipcub/hipcub.hpp>

#include "aiter_hip_common.h"
#include "ck_tile/core.hpp"
#include "dispatch_utils.h"
#include "hip_compat.h"
#include "py_itfs_common.h"

// Helper macros
#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")

#define HIP_CHECK(err)                                                      \
    do {                                                                    \
        hipError_t err_ = (err);                                            \
        if (err_ != hipSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("HIP error: ") + hipGetErrorString(err_) +      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
        }                                                                   \
    } while (0)

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                  \
    [&] {                                                                   \
        if (COND) {                                                         \
            static constexpr bool CONST_NAME = true;                        \
            return __VA_ARGS__();                                           \
        } else {                                                            \
            static constexpr bool CONST_NAME = false;                       \
            return __VA_ARGS__();                                           \
        }                                                                   \
    }()

namespace aiter {

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
constexpr T custom_max(std::initializer_list<T> ilist) {
    T result = *ilist.begin();
    for (auto it = ilist.begin() + 1; it != ilist.end(); ++it) {
        if (*it > result) result = *it;
    }
    return result;
}

// BytesToType: maps byte count to vector type
template<int N> struct BytesToType {};
template<> struct BytesToType<2> { using Type = __half2; };
template<> struct BytesToType<4> { using Type = float; };
template<> struct BytesToType<8> { using Type = float2; };
template<> struct BytesToType<16> { using Type = float4; };
template<> struct BytesToType<32> { using Type = struct { float4 x; float4 y; }; };

// Custom half8_t for FP16
struct half8_t {
    __half2 x, y, z, w;
};

// SiLU activation
template <typename T>
__device__ __forceinline__ float silu(const T& x) {
    float fx = ck_tile::type_convert<float>(x);
    return fx / (1.0f + expf(-fx));
}

// ============================================================================
// Parameter Structures
// ============================================================================

struct ConvParamsBase {
    using index_t = int32_t;

    int32_t batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    
    index_t weight_c_stride;
    index_t weight_width_stride;
    
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    // Common pointers
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;
};

// ============================================================================
// Kernel Traits
// ============================================================================

template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4, "Only FP16 and FP32 are supported");
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts, "Width must be <= kNElts");
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    
    // Vector type for vectorized loads
    using vec_t = typename std::conditional<
        std::is_same<input_t, __half>::value,
        half8_t,
        typename BytesToType<kNBytes * kNElts>::Type
    >::type;
    
    // BlockLoad and BlockStore for coalesced memory access
    using BlockLoadT = hipcub::BlockLoad<input_t, kNThreads, kNElts, hipcub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = hipcub::BlockLoad<vec_t, kNThreads, 1, hipcub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = hipcub::BlockStore<input_t, kNThreads, kNElts, hipcub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = hipcub::BlockStore<vec_t, kNThreads, 1, hipcub::BLOCK_STORE_DIRECT>;
    
    // Shared memory size calculation
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

// ============================================================================
// Optimized Causal Conv1D Forward Kernel
// ============================================================================

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    
    // Pointer arithmetic
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    // Initialize boundary (thread 0 sets up initial exchange buffer for causality)
    if (tidx == 0) {
        input_t zeros[kNElts] = {0};
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(zeros)[0];
    }

    // Load weights into registers (all threads load the same weights for this channel)
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { 
        weight_vals[i] = float(weight[i * params.weight_width_stride]); 
    }

    // Process sequence in chunks
    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        
        // Load current chunk using BlockLoad or BlockLoadVec
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(
                reinterpret_cast<vec_t*>(x), 
                *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), 
                (params.seqlen - chunk * kChunkSize) / kNElts
            );
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(
                x, 
                *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), 
                params.seqlen - chunk * kChunkSize
            );
        }
        x += kChunkSize;
        
        __syncthreads();
        
        // Exchange boundary data for causal convolution
        // Thread i writes its first kNElts elements for thread i+1 to read
        if (tidx < kNThreads - 1) { 
            smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; 
        }
        
        __syncthreads();
        
        // Thread i reads from thread i-1 (causality: need previous elements)
        reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        
        __syncthreads();
        
        // Last thread updates the exchange buffer for next chunk
        if (tidx == kNThreads - 1) { 
            smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; 
        }

        // Convert to float for computation
        float x_vals[2 * kNElts];
        #pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i) { 
            x_vals[i] = float(x_vals_load[i]); 
        }

        // Perform causal convolution
        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                // Causal access pattern: output[i] depends on input[i-(kWidth-1), ..., i]
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
            }
        }

        // Apply SiLU activation if requested
        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        // Convert back and store
        input_t out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { 
            out_vals_store[i] = input_t(out_vals[i]); 
        }
        
        // Store results using BlockStore or BlockStoreVec
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(
                reinterpret_cast<vec_t*>(out), 
                reinterpret_cast<vec_t (&)[1]>(out_vals_store), 
                (params.seqlen - chunk * kChunkSize) / kNElts
            );
        } else {
            __syncthreads();
            typename Ktraits::BlockStoreT(smem_store).Store(
                out, 
                out_vals_store, 
                params.seqlen - chunk * kChunkSize
            );
        }
        out += kChunkSize;
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase &params, hipStream_t stream) {
    // Determine if we can use vectorized load (seqlen must be multiple of kNElts)
    const int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    const bool can_use_vec_load = params.seqlen % kNElts == 0;

    // Grid and block dimensions
    dim3 grid(params.batch, params.dim);
    constexpr int kNThreads = 128;  // Standard block size

    // Dispatch based on width and vectorized load capability
    BOOL_SWITCH(can_use_vec_load, kIsVecLoad, [&] {
        switch (params.width) {
            case 2: {
                using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, 2, kIsVecLoad, input_t, weight_t>;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(causal_conv1d_fwd_kernel<Ktraits>),
                    grid, kNThreads, Ktraits::kSmemSize, stream,
                    params
                );
                break;
            }
            case 3: {
                using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, 3, kIsVecLoad, input_t, weight_t>;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(causal_conv1d_fwd_kernel<Ktraits>),
                    grid, kNThreads, Ktraits::kSmemSize, stream,
                    params
                );
                break;
            }
            case 4: {
                using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, 4, kIsVecLoad, input_t, weight_t>;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(causal_conv1d_fwd_kernel<Ktraits>),
                    grid, kNThreads, Ktraits::kSmemSize, stream,
                    params
                );
                break;
            }
            default:
                TORCH_CHECK(false, "Unsupported width. Only 2, 3, 4 are supported.");
        }
    });
}

// ============================================================================
// PyTorch Interface
// ============================================================================

void causal_conv1d_fwd(
    torch::Tensor& out,           // [batch, dim, seqlen]
    const torch::Tensor& x,       // [batch, dim, seqlen]
    const torch::Tensor& weight,  // [dim, width]
    const torch::Tensor& bias,    // [dim] or empty
    bool use_silu)
{
    CHECK_INPUT(out);
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    
    const int32_t batch = x.size(0);
    const int32_t dim = x.size(1);
    const int32_t seqlen = x.size(2);
    const int32_t width = weight.size(1);
    
    TORCH_CHECK(out.size(0) == batch && out.size(1) == dim && out.size(2) == seqlen,
                "Output shape mismatch");
    TORCH_CHECK(weight.size(0) == dim, "Weight shape mismatch");
    TORCH_CHECK(width >= 2 && width <= 4, "Width must be 2, 3, or 4");
    
    // Setup parameters
    ConvParamsBase params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;
    
    // Strides (assuming contiguous tensors with channel-first layout)
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(2);
    
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(2);
    
    // Pointers
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.out_ptr = out.data_ptr();
    
    if(bias.defined() && bias.numel() > 0)
    {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.size(0) == dim, "Bias shape mismatch");
        params.bias_ptr = bias.data_ptr();
    } else {
        params.bias_ptr = nullptr;
    }
    
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Type dispatching
    VLLM_DISPATCH_FLOATING_TYPES(x.scalar_type(), "causal_conv1d_fwd", [&] {
        using dtype = typename t2ck<scalar_t>::type;
        causal_conv1d_fwd_launch<dtype, dtype>(params, stream);
    });
    
    HIP_CHECK(hipGetLastError());
}

} // namespace aiter
