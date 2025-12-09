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

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define HIP_CHECK(err)                                                      \
    do {                                                                    \
        hipError_t err_ = (err);                                            \
        if (err_ != hipSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("HIP error: ") + hipGetErrorString(err_) +      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
        }                                                                   \
    } while (0)

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half) {                                             \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::Float)  {                                    \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

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

// Custom bfloat16_8_t for BF16
struct bfloat16_8_t {
    __hip_bfloat162 x, y, z, w;
};

// SiLU activation
template <typename T>
__device__ __forceinline__ float silu(const T& x) {
    float fx = ck_tile::type_convert<float>(x);
    return fx / (1.0f + expf(-fx));
}

// Helper for constexpr min
constexpr int constexpr_min(int a, int b) {
    return (a < b) ? a : b;
}

// ============================================================================
// Parameter Structures
// ============================================================================

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    int32_t *__restrict__ cache_seqlens;

    // Only used if the elements of the batch are gathered from a larger buffer,
    // which may happen for continuous batching.
    int32_t *__restrict__ conv_state_indices_ptr;

    void *__restrict__ seq_idx_ptr;

    // No __restrict__ since initial_states could be the same as final_states.
    void * initial_states_ptr;
    index_t initial_states_batch_stride;
    index_t initial_states_l_stride;
    index_t initial_states_c_stride;

    void * final_states_ptr;
    index_t final_states_batch_stride;
    index_t final_states_l_stride;
    index_t final_states_c_stride;
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
        std::is_same<input_t, at::Half>::value,
        half8_t,
        typename std::conditional<
            std::is_same<input_t, at::BFloat16>::value,
            bfloat16_8_t,
            typename BytesToType<kNBytes * kNElts>::Type
        >::type
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
// Causal Conv1D Forward Kernel
// ============================================================================

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_fn_kernel(ConvParamsBase params) {
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

void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const torch::Tensor x,
                         const torch::Tensor weight,
                         const torch::Tensor out,
                         void* bias_ptr,
                         bool silu_activation) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias_ptr;
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(-1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(-1);
}

template<typename input_t, typename weight_t>
void causal_conv1d_fn_launch(ConvParamsBase &params, hipStream_t stream) {
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
                    HIP_KERNEL_NAME(causal_conv1d_fn_kernel<Ktraits>),
                    grid, kNThreads, Ktraits::kSmemSize, stream,
                    params
                );
                break;
            }
            case 3: {
                using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, 3, kIsVecLoad, input_t, weight_t>;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(causal_conv1d_fn_kernel<Ktraits>),
                    grid, kNThreads, Ktraits::kSmemSize, stream,
                    params
                );
                break;
            }
            case 4: {
                using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, 4, kIsVecLoad, input_t, weight_t>;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(causal_conv1d_fn_kernel<Ktraits>),
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

// ==================== Kernel Traits ====================

template<int kNThreads_, int kWidth_, int kChunkSizeL_, typename input_t_, typename weight_t_>
struct Causal_conv1d_channellast_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 64 == 0, "MI308 uses 64-wide wavefronts");
    static constexpr int kNWarps = kNThreads / 64;  // AMD MI308 wavefront size
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4, "Only FP16/BF16 and FP32 are supported");
    
    // Vectorization: 128-byte cache line
    // For FP32: 4 elements (16 bytes per load), 32 elements per cache line
    // For FP16/BF16: 8 elements (16 bytes per load), 64 elements per cache line
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static constexpr int kNEltsPerRow = 128 / kNBytes;
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128, "Cache line alignment");
    
    static constexpr int kNColsPerWarp = 64 / kNThreadsPerRow;  // MI308: 64-wide wavefront
    static_assert(kNColsPerWarp * kNThreadsPerRow == 64, "Wavefront coverage");
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL, "Chunk coverage");
    
    using vec_t = typename std::conditional<
        std::is_same<input_t, at::Half>::value,
        half8_t,
        typename std::conditional<
            std::is_same<input_t, at::BFloat16>::value,
            bfloat16_8_t,
            typename BytesToType<kNBytes * kNElts>::Type
        >::type
    >::type;
};

// ==================== Channel-Last Forward Kernel ====================

template<typename Ktraits, bool kHasSeqIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_channellast_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory with padding to avoid LDS bank conflicts
    __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL][kChunkSizeC + kNElts];

    // Block and thread indices
    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    
    // Thread mapping for loading (channel-wise)
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    
    // Global memory pointers
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;
    
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.out_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    
    // Sequence index pointer (for handling sub-sequences)
    int *seq_idx = !kHasSeqIdx ? nullptr : reinterpret_cast<int *>(params.seq_idx_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;
    
    // Initial states pointer (for first L-chunk only)
    input_t *initial_states = params.initial_states_ptr == nullptr || chunk_l_id > 0 ? nullptr
        : reinterpret_cast<input_t *>(params.initial_states_ptr) + batch_id * params.initial_states_batch_stride 
        + l_idx * params.initial_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    
    // Final states pointer (for last L-chunk only)
    // The last L-chunk will have enough info to write to final states, since it also contains
    // a few x values from the previous L-chunk.
    input_t *final_states = params.final_states_ptr == nullptr || chunk_l_id < gridDim.y - 1 ? nullptr
        : reinterpret_cast<input_t *>(params.final_states_ptr) + batch_id * params.final_states_batch_stride 
        + l_idx * params.final_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;

    // ==================== Load input data into shared memory ====================
    
    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t x_vals_load[kNElts] = {0};
        
        // Vectorized load (128-byte aligned)
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        
        reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = 
            reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    
    // Load elements from previous chunk (for causal dependency)
    if (l_idx < kWidth - 1) {
        input_t x_vals_load[kNElts] = {0};
        
        // Try to load from previous positions in the sequence
        if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
            && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            // Load from global memory (previous part of sequence)
            reinterpret_cast<vec_t *>(x_vals_load)[0] = 
                *reinterpret_cast<vec_t *>(x - (kWidth - 1) * params.x_l_stride);
        } else if (initial_states != nullptr
                   && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < 0
                   && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            // Load from initial_states (for first chunk)
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(initial_states);
        }
        
        reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    __syncthreads();
    
    // Write final states (for last L-chunk only)
    if (final_states != nullptr
        && l_idx < kWidth - 1
        && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
        // x_smem[0] contains element at index chunk_l_id * kChunkSizeL - (kWidth - 1)
        // So last few elements (index params.seqlen - kWidth + 1 + l_idx) are stored in
        // x_smem[params.seqlen + l_idx - chunk_l_id * kChunkSizeL]
        *reinterpret_cast<vec_t *>(final_states) = 
            reinterpret_cast<vec_t *>(x_smem[params.seqlen + l_idx - chunk_l_id * kChunkSizeL])[c_idx];
    }

    // ==================== Compute convolution ====================
    
    constexpr int kLPerThread = constexpr_min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    
    // Power of 2 checks for efficiency
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 64);

    // Thread remapping for computation (time-wise)
    const int row_idx = tid / kNThreadsPerRow;  // Channel index
    const int col_idx = tid % kNThreadsPerRow;  // Time index

    // Load bias
    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim 
        ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    
    // Load weights into registers
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = float(weight[row_idx * params.weight_c_stride + w * params.weight_width_stride]);
        }
    }
    
    // Load input values from shared memory
    float x_vals[kWidth - 1 + kLPerThread];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }
    
    // Load sequence indices (for handling sub-sequences)
    int seq_idx_thread[kWidth - 1 + kLPerThread];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            // seq_idx is -1 for positions before the start of the sequence
            seq_idx_thread[i] = chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i - (kWidth - 1) >= 0 
                ? seq_idx[col_idx * kLPerThread + i - (kWidth - 1)] : -1;
        }
    }

    // Perform causal convolution
    float out_vals[kLPerThread];
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        out_vals[i] = bias_val;
        
        const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
        
        // For padding tokens (seq_idx < 0), skip computation and set output to 0
        if (seq_idx_cur < 0) {
            out_vals[i] = 0.f;
            continue;
        }
        
        // Convolve
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            if constexpr (!kHasSeqIdx) {
                // Normal case: no sequence boundaries
                out_vals[i] += weight_vals[w] * x_vals[i + w];
            } else {
                // With seq_idx: only accumulate if within same sub-sequence
                out_vals[i] += seq_idx_thread[i + w] == seq_idx_cur ? weight_vals[w] * x_vals[i + w] : 0.f;
            }
        }
        
        // Apply SiLU activation: x / (1 + exp(-x))
        if (params.silu_activation) {
            out_vals[i] = out_vals[i] / (1.f + expf(-out_vals[i]));
        }
    }

    // ==================== Write output ====================
    
    __syncthreads();
    
    // Store to shared memory (transposed for coalesced writes)
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        x_smem[col_idx * kLPerThread + i][row_idx] = static_cast<input_t>(out_vals[i]);
    }
    
    __syncthreads();
    
    // Vectorized write to global memory
    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t out_vals_store[kNElts];
        reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[l * kLPerLoad + l_idx])[c_idx];
        
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(out + l * kLPerLoad * params.out_l_stride) = 
                reinterpret_cast<vec_t *>(out_vals_store)[0];
        }
    }
}

// ==================== Launch Helper ====================

// Internal launch function with explicit width template parameter
template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_launch_impl(ConvParamsBase &params, hipStream_t stream) {
    using Ktraits = Causal_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, 64, input_t, weight_t>;
    
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    
    const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
    const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
    
    dim3 grid(params.batch, n_chunks_L, n_chunks_C);
    dim3 block(Ktraits::kNThreads);
    
    // Dispatch based on whether seq_idx is present
    if (params.seq_idx_ptr != nullptr) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_fwd_kernel<Ktraits, true>),
            grid, block, 0, stream,
            params
        );
    } else {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_fwd_kernel<Ktraits, false>),
            grid, block, 0, stream,
            params
        );
    }
}

// Overloaded launch function that dispatches based on width (like causal_conv1d_fn_launch)
template<typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_launch(ConvParamsBase &params, hipStream_t stream) {
    constexpr int kNThreads = 128;  // Standard block size
    
    // Dispatch based on width
    switch (params.width) {
        case 2:
            causal_conv1d_channellast_fwd_launch_impl<kNThreads, 2, input_t, weight_t>(params, stream);
            break;
        case 3:
            causal_conv1d_channellast_fwd_launch_impl<kNThreads, 3, input_t, weight_t>(params, stream);
            break;
        case 4:
            causal_conv1d_channellast_fwd_launch_impl<kNThreads, 4, input_t, weight_t>(params, stream);
            break;
        default:
            TORCH_CHECK(false, "Unsupported width. Only 2, 3, 4 are supported.");
    }
}

// ============================================================================
// PyTorch Interface
// ============================================================================

void causal_conv1d_fn(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias_,
    const c10::optional<torch::Tensor> &seq_idx_,
    const c10::optional<torch::Tensor> &initial_states_,
    torch::Tensor &out,
    c10::optional<torch::Tensor> &final_states_out_,
    bool silu_activation)
{
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
    const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;

    if (is_channel_last) {
        TORCH_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        TORCH_CHECK(x.stride(2) % 8 == 0 && x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");
    }
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        TORCH_CHECK(is_channel_last, "seq_idx is only supported for channel last layout");
        auto seq_idx = seq_idx_.value();
        TORCH_CHECK(seq_idx.scalar_type() == torch::kInt32);
        TORCH_CHECK(seq_idx.is_cuda());
        TORCH_CHECK(seq_idx.is_contiguous());
        CHECK_SHAPE(seq_idx, batch_size, seqlen);
    }

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        params.seq_idx_ptr = seq_idx_.value().data_ptr();
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        TORCH_CHECK(is_channel_last, "initial_states is only supported for channel last layout");
        auto initial_states = initial_states_.value();
        TORCH_CHECK(initial_states.scalar_type() == input_type);
        TORCH_CHECK(initial_states.is_cuda());
        CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
        TORCH_CHECK(initial_states.stride(1) == 1);
        params.initial_states_ptr = initial_states.data_ptr();
        params.initial_states_batch_stride = initial_states.stride(0);
        params.initial_states_c_stride = initial_states.stride(1);
        params.initial_states_l_stride = initial_states.stride(2);
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (final_states_out_.has_value()) {
        TORCH_CHECK(is_channel_last, "final_states is only supported for channel last layout");
        auto final_states = final_states_out_.value();
        TORCH_CHECK(final_states.scalar_type() == input_type);
        TORCH_CHECK(final_states.is_cuda());
        CHECK_SHAPE(final_states, batch_size, dim, width - 1);
        TORCH_CHECK(final_states.stride(1) == 1);
        params.final_states_ptr = final_states.data_ptr();
        params.final_states_batch_stride = final_states.stride(0);
        params.final_states_c_stride = final_states.stride(1);
        params.final_states_l_stride = final_states.stride(2);
    } else {
        params.final_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(x));
    auto stream = at::hip::getCurrentHIPStream();
    
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fn", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_fn", [&] {
            if (!is_channel_last) {
                causal_conv1d_fn_launch<input_t, weight_t>(params, stream);
            } else {
                causal_conv1d_channellast_fwd_launch<input_t, weight_t>(params, stream);
            }
        });
    });
    
    HIP_CHECK(hipGetLastError());
}

} // namespace aiter
