// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <algorithm>
#include "hip_compat.h"

#define AT_DISPATCH_FP8_CASE(enum_type, ...) \
    AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

#define AITER_DISPATCH_CASE_FP8_TYPES(...)                           \
    AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
    AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

#define AITER_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, AITER_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx942__))
#define __HIP__MI300_MI250__
#endif

#if defined(__HIPCC__) && defined(__gfx942__)
#define __HIP__MI300__
#endif

#if defined(NDEBUG)
#undef NDEBUG
#include <assert.h>
#define UNREACHABLE_CODE assert(false);
#define NDEBUG
#else
#define UNREACHABLE_CODE assert(false);
#endif

namespace aiter {

template <typename T>
struct scalar
{
};
template <>
struct scalar<c10::Half>
{
    using type = half;
};
template <>
struct scalar<c10::BFloat16>
{
    using type = __hip_bfloat16;
};

template <typename T>
struct scalar2
{
};
template <>
struct scalar2<c10::Half>
{
    using type = __half2;
};
template <>
struct scalar2<c10::BFloat16>
{
    using type = __hip_bfloat162;
};

template <typename T>
struct fmul2_out
{
};
template <>
struct fmul2_out<c10::Half>
{
    // using type = __half2;
    using type = float2;
};
template <>
struct fmul2_out<c10::BFloat16>
{
    using type = float2;
    // using type = __hip_bfloat162;
};

template <typename T>
struct type_to_type2
{
};
template <>
struct type_to_type2<__half>
{
    using type = __half2;
};
template <>
struct type_to_type2<__hip_bfloat16>
{
    using type = __hip_bfloat162;
};

template <typename T>
__device__ __forceinline__ float2 __s22float2(T v);

template <typename T>
__device__ __forceinline__ T __float2s(float v);

template <typename T>
__device__ __forceinline__ T __float22s2_rn(float2 v);

template <>
__device__ __forceinline__ half __float2s(float v)
{
    return __float2half(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__half2 v)
{
    return __half22float2(v);
}

template <>
__device__ __forceinline__ __half2 __float22s2_rn(float2 v)
{
    return __float22half2_rn(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s(float v)
{
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float2 __s22float2(__hip_bfloat162 v)
{
    return __bfloat1622float2(v);
}
template <>
__device__ __forceinline__ float2 __s22float2(float2 v)
{
    return v;
}

template <>
__device__ __forceinline__ __hip_bfloat162 __float22s2_rn(float2 v)
{
    return __float22bfloat162_rn(v);
}

__device__ __forceinline__ float __hmul_fp32(const __hip_bfloat16 a, const __hip_bfloat16 b)
{
    return __bfloat162float(a) * __bfloat162float(b);
}
// __device__ __forceinline__ __hip_bfloat162 __hmul2_fp32(const __hip_bfloat162
// a, const __hip_bfloat162 b) {
//   return __hmul2(a, b);
// }
__device__ __forceinline__ float2 __hmul2_fp32(const __hip_bfloat162 a, const __hip_bfloat162 b)
{
    return float2(__hmul_fp32(a.x, b.x), __hmul_fp32(a.y, b.y));
}

__device__ __forceinline__ float __hmul_fp32(const __half a, const __half b)
{
    return __half2float(a) * __half2float(b);
}
// __device__ __forceinline__ __half2 __hmul2_fp32(const __half2 a, const __half2 b) { return
// __hmul2(a, b); }
__device__ __forceinline__ float2 __hmul2_fp32(const __half2 a, const __half2 b)
{
    return float2(__hmul_fp32(a.x, b.x), __hmul_fp32(a.y, b.y));
}

__device__ __forceinline__ float
__hfma_fp32(const __hip_bfloat16 a, const __hip_bfloat16 b, const float c)
{
    return __ocml_fma_f32(__bfloat162float(a), __bfloat162float(b), c);
}
__device__ __forceinline__ float2 __hfma2_fp32(const __hip_bfloat162 a,
                                               const __hip_bfloat162 b,
                                               const float2 c)
{
    return float2(__hfma_fp32(a.x, b.x, c.x), __hfma_fp32(a.y, b.y, c.y));
}
__device__ __forceinline__ __hip_bfloat162 __hfma2_fp32(const __hip_bfloat162 a,
                                                        const __hip_bfloat162 b,
                                                        const __hip_bfloat162 c)
{
    return __hfma2(a, b, c);
}

__device__ __forceinline__ float __hfma_fp32(const __half a, const __half b, const float c)
{
    return __ocml_fma_f32(__half2float(a), __half2float(b), c);
}
// __device__ __forceinline__ __half2 __hfma2_fp32(const __half2 a, const __half2 b, const __half2
// c) {
//   return __hfma2(a, b, c);
// }
__device__ __forceinline__ float2 __hfma2_fp32(const __half2 a, const __half2 b, const float2 c)
{
    return float2(__hfma_fp32(a.x, b.x, c.x), __hfma_fp32(a.y, b.y, c.y));
}

template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr)
{
    auto addr_alias = reinterpret_cast<const float*>(addr);
    auto dat0       = loadnt(addr_alias);
    auto dat1       = loadnt(addr_alias + 1);
    auto dat2       = loadnt(addr_alias + 2);
    auto dat3       = loadnt(addr_alias + 3);
    return make_float4(dat0, dat1, dat2, dat3);
}

// Asynchronously load 128-bit data from global memory to VGPR (non-temporal)
template <typename T>
__device__ __forceinline__ void global_load_dwordx4_nontemporal_async(T& reg, const T* addr)
{
    asm volatile("global_load_dwordx4 %0, %1, off nt" : "=v"(reg) : "v"(addr) : "memory");
}

// Asynchronously load 128-bit data from global memory to VGPR
template <typename T>
__device__ __forceinline__ void global_load_dwordx4_async(T& reg, const T* addr)
{
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(reg) : "v"(addr) : "memory");
}

// Asynchronously load 128-bit data from LDS to VGPR
template <typename T>
__device__ __forceinline__ void lds_read_dwordx4_async(T& reg, const T* addr)
{
    asm volatile("ds_read_b128 %0, %1" : "=v"(reg) : "v"(*((int*)(&addr))) : "memory");
}

// Wait for global memory load operations to complete (vmcnt)
// max_pending must be a constant that can be determined at compile time
__device__ __forceinline__ void wait_global_loads(uint32_t max_pending)
{
    asm volatile("s_waitcnt vmcnt(%0)" ::"i"(max_pending));
}

// Wait for LDS-related operations to complete (lgkmcnt)
// max_pending must be a constant that can be determined at compile time
__device__ __forceinline__ void wait_lds_ops(uint32_t max_pending)
{
    asm volatile("s_waitcnt lgkmcnt(%0)" ::"i"(max_pending));
}

#define DOT2C(V0, V2, V3)                                                                    \
    if constexpr(std::is_same_v<scalar_t, half>)                                             \
    {                                                                                        \
        asm("v_dot2c_f32_f16 %0, %2, %3" : "=v"(V0) : "0"(V0), "v"(V2), "v"(V3));            \
    }                                                                                        \
    else if constexpr(std::is_same_v<scalar_t, __hip_bfloat16>)                              \
    {                                                                                        \
        float2 s = __hmul2_fp32(*((__hip_bfloat162*)(&(V2))), *((__hip_bfloat162*)(&(V3)))); \
        V0 += (s.x + s.y);                                                                   \
    }

// TBlock fetches entire rows of A, and entire col of B (K dimension); assume
// N=1 for time being grid is M/A_NUM_ROWS blocks
template <typename scalar_t, int NUM_A_ROWS_PER_BLOCK>
__global__ void
LLGemm1_kernel(const scalar_t* in_a, const scalar_t* in_b, scalar_t* out_c, const int K)
{
    using scalar2_t   = typename scalar2<scalar_t>::type;
    using fmul2_out_t = typename fmul2_out<scalar_t>::type;

    auto af4 = reinterpret_cast<const float4*>(in_a);
    auto bf4 = reinterpret_cast<const scalar2_t*>(in_b);
    auto c   = reinterpret_cast<scalar2_t*>(out_c);
    __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
    const int row_addr  = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
    const int threadid  = threadIdx.x;
    const int warp      = threadIdx.x / WARP_SIZE;
    const int lane      = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int qwarpid   = threadid / num_warps;
    const int qthreadid = threadid % num_warps;
    float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
    scalar2_t colB_elem4x, colB_elem4y, colB_elem4z, colB_elem4w;
    float acc[NUM_A_ROWS_PER_BLOCK];
    fmul2_out_t acch2;
    scalar2_t oval;

    // As we later use warp shuffle operations, we may have more threads in the
    // block than the actual available data, hence the if guard here.
    if(threadid * 8 < K)
    {
#pragma unroll
        for(int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++)
        {
            // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
            rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
        }
    }

    colB_elem4x = bf4[threadid * 4 + 0];
    colB_elem4y = bf4[threadid * 4 + 1];
    colB_elem4z = bf4[threadid * 4 + 2];
    colB_elem4w = bf4[threadid * 4 + 3];

    scalar2_t Af2;
    [[maybe_unused]] scalar2_t Bf2;
    float2 S;

    auto Ah2ptr = reinterpret_cast<scalar2_t*>(&rowA_elem4);
    scalar2_t* ah2lptr;

#pragma unroll
    for(int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++)
    {
        // Multiply-add on 8 scalar_t.
        ah2lptr = Ah2ptr + i * 4;
        Af2     = *(ah2lptr);
        acch2   = __hmul2_fp32(Af2, colB_elem4x);
        Af2     = *(ah2lptr + 1);
        acch2   = __hfma2_fp32(Af2, colB_elem4y, acch2);
        Af2     = *(ah2lptr + 2);
        acch2   = __hfma2_fp32(Af2, colB_elem4z, acch2);
        Af2     = *(ah2lptr + 3);
        acch2   = __hfma2_fp32(Af2, colB_elem4w, acch2);

        // See comment above concerning the if guard.
        acc[i] = (threadid * 8 < K ? acch2.x + acch2.y : 0.f);
    }

// all reduce across warp.
#pragma unroll
    for(int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
    {
#pragma unroll
        for(int i = 0; i < NUM_A_ROWS_PER_BLOCK; i++)
        {
            acc[i] += __shfl_xor(acc[i], mask);
        }
    }

    // Warp leaders store the data to shared memory.
    if(lane < NUM_A_ROWS_PER_BLOCK)
    {
        red_smem[lane][warp] = acc[lane];
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    if(qwarpid < NUM_A_ROWS_PER_BLOCK)
    {
        acc[qwarpid] = qthreadid < num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
        for(int mask = num_warps / 2; mask >= 1; mask /= 2)
        {
            acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
        }
        float oval2 = __shfl_xor(acc[qwarpid], num_warps);

        if(lane % (num_warps * 2) == 0)
        {
            oval = __float22s2_rn<scalar2_t>(make_float2(acc[qwarpid], oval2));
            c[blockIdx.x * NUM_A_ROWS_PER_BLOCK / 2 + qwarpid / 2] = oval;
        }
    }
}

// template <typename T>
void LLGemm1(void* in_a,
             void* in_b,
             void* out_c,
             const int M,
             const int K,
             cudaStream_t stream,
             const int rows_per_block,
             const c10::ScalarType scalar_type)
{
    // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle
    // operations.
    const int NUM_THREADS = K * 2 / 16 % WARP_SIZE == 0
                                ? K * 2 / 16
                                : K * 2 / 16 + (WARP_SIZE - K * 2 / 16 % WARP_SIZE);

    int NUM_BLOCKS = M / rows_per_block;

    // call the kernel function...
    AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "LLGemm1", [&] {
        scalar_t* a_ptr = reinterpret_cast<scalar_t*>(in_a);
        scalar_t* b_ptr = reinterpret_cast<scalar_t*>(in_b);
        scalar_t* c_ptr = reinterpret_cast<scalar_t*>(out_c);
        if(rows_per_block == 2)
        {
            LLGemm1_kernel<scalar_t, 2>
                <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
        }
        else if(rows_per_block == 4)
        {
            LLGemm1_kernel<scalar_t, 4>
                <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
        }
        else if(rows_per_block == 8)
        {
            LLGemm1_kernel<scalar_t, 8>
                <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
        }
        else if(rows_per_block == 16)
        {
            LLGemm1_kernel<scalar_t, 16>
                <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
        }
        else
        {
            NUM_BLOCKS = M / 4;
            LLGemm1_kernel<scalar_t, 4>
                <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(a_ptr, b_ptr, c_ptr, K);
        }
    });
}

#if defined(__HIP__MI300_MI250__) // TODO: Add NAVI support

// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wv_splitk_small_fp16_bf16_kernel(const int K,
                                     const int M,
                                     const scalar_t* B,
                                     const scalar_t* __restrict__ A,
                                     scalar_t* C,
                                     const int _WvPrGrp,
                                     const int CuCount)
{
    // static_assert(UNRL * N <= 16);
    using scalar2_t = typename type_to_type2<scalar_t>::type;
    using scalar8   = __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
    union bigType
    {
        scalar_t h[A_CHUNK];
        float f[A_CHUNK / 2];
        float2 f2[A_CHUNK / 4];
        double d[A_CHUNK / 4];
        scalar8 h8;
    };
    float sum[N][YTILE];
    uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

    //----------------------------------------------------
    // Each wave works on a single column of weight matrix.
    // There are 16 waves per WG, and hence, each WG is
    // working on 16 columns of weight matrix. Moreover,
    // we tile in column direction by YTILE, so when YTILE=1
    // the above math is right, however, when YTILE=2 then
    // each wave  will be working on 2 columns and WG will
    // be working on 32 columns.
    //
    // Top level loop that makes WGs persistent!
    // - WGs iterates across columns of weight matrix
    // - Each wave within WG works on a given column(s)
    // - After completing first set of columns, WGs start
    //   working on the next set of available columns
    //----------------------------------------------------
    while(m < M)
    {
        //----------------------------------------------------
        // 'sum' accumulates the matrix A x B computation
        // split across 64 lanes.
        //
        // YTILE represents how many column of weight matrix
        // are being worked on by each wave.
        //----------------------------------------------------
        for(int i = 0; i < YTILE; i++)
            for(int n = 0; n < N; n++)
                sum[n][i] = 0;

        bigType bigA[N][UNRL];
        bigType bigB[YTILE][UNRL];
        //----------------------------------------------------
        // Fetch weight matrix B in interleaved K-split!
        // - Each thread (lane) is fetching 8 elements (A_Chunk)
        // - Each wave will fetch 64*8=> 512 elements (1024B)
        // - YTILE represents the number of column being serviced
        //   by wave
        // - Loop for fetching weight matrix (B) are unrolled
        //
        // Fetch activation matrix A from LDS
        // - Loop for fetching activation matrix (A) are unrolled
        //
        // Finally, do the matrix multiplication in an unrolled
        // fashion. This provides lot of food for compiler
        // scheduling.
        //
        // TODO: Logic below will only work when K is multiple of 8
        //----------------------------------------------------
        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
            // Fetch weight matrix B from memory
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const scalar_t* B_ = &B[(m + 0) * K + k_];
                bigB[0][k2].h8     = (loadnt((scalar8*)(&B_[0 * K])));
                //----------------------------------------------------
                // The following code with YTILE > 1 has to be deleted
                //----------------------------------------------------
                if constexpr(YTILE >= 2)
                    bigB[1][k2].h8 = (loadnt((scalar8*)(&B_[1 * K])));
                if constexpr(YTILE >= 3)
                    bigB[2][k2].h8 = (loadnt((scalar8*)(&B_[2 * K])));
                if constexpr(YTILE >= 4)
                    bigB[3][k2].h8 = (loadnt((scalar8*)(&B_[3 * K])));

                // global_load_dwordx4_nontemporal_async(bigB[0][k2].h8, (scalar8*)(&B_[0 * K]));
                // if constexpr (YTILE >= 2) global_load_dwordx4_nontemporal_async(bigB[1][k2].h8,
                // (scalar8*)(&B_[1 * K]));

                // Fetch activation matrix A from memory
                for(int n = 0; n < N; n++)
                {
                    const scalar_t* a_addr = &(A[k_ + n * K]);
                    bigA[n][k2]            = *((const bigType*)a_addr);
                    // global_load_dwordx4_async(bigA[n][k2].h8, (scalar8*)a_addr);
                }
            }

            // Do the matrix multiplication in interleaved manner
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;
                // wait_global_loads((UNRL - k2 - 1) * (N + YTILE));

                // Do the matrix multiplication of activation and weight matrix
                if constexpr(std::is_same_v<scalar_t, __hip_bfloat16>)
                {
#pragma unroll
                    for(uint32_t n = 0; n < N; n++)
                    {
                        float2 acc2[YTILE];
                        acc2[0] = __hmul2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[0]))),
                                               *((scalar2_t*)(&(bigB[0][k2].f[0]))));
                        if constexpr(YTILE >= 2)
                            acc2[1] = __hmul2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[0]))),
                                                   *((scalar2_t*)(&(bigB[1][k2].f[0]))));
                        if constexpr(YTILE >= 3)
                            acc2[2] = __hmul2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[0]))),
                                                   *((scalar2_t*)(&(bigB[2][k2].f[0]))));
                        if constexpr(YTILE >= 4)
                            acc2[3] = __hmul2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[0]))),
                                                   *((scalar2_t*)(&(bigB[3][k2].f[0]))));

#pragma unroll
                        for(uint32_t b = 1; b < A_CHUNK / 2; b++)
                        {
                            acc2[0] = __hfma2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[b]))),
                                                   *((scalar2_t*)(&(bigB[0][k2].f[b]))),
                                                   acc2[0]);
                            if constexpr(YTILE >= 2)
                                acc2[1] = __hfma2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[b]))),
                                                       *((scalar2_t*)(&(bigB[1][k2].f[b]))),
                                                       acc2[1]);
                            if constexpr(YTILE >= 3)
                                acc2[2] = __hfma2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[b]))),
                                                       *((scalar2_t*)(&(bigB[2][k2].f[b]))),
                                                       acc2[2]);
                            if constexpr(YTILE >= 4)
                                acc2[3] = __hfma2_fp32(*((scalar2_t*)(&(bigA[n][k2].f[b]))),
                                                       *((scalar2_t*)(&(bigB[3][k2].f[b]))),
                                                       acc2[3]);
                        }
#pragma unroll
                        for(uint32_t i = 0; i < YTILE; ++i)
                        {
                            sum[n][i] += acc2[i].x + acc2[i].y;
                        }
                    }
                }
                else
                {
#pragma unroll
                    for(uint32_t n = 0; n < N; n++)
                    {
#pragma unroll
                        for(uint32_t b = 0; b < A_CHUNK / 2; b++)
                        {
                            DOT2C(sum[n][0], bigA[n][k2].f[b], bigB[0][k2].f[b])
                            //----------------------------------------------------
                            // The following code with YTILE > 1
                            //----------------------------------------------------
                            if constexpr(YTILE >= 2)
                            {
                                DOT2C(sum[n][1], bigA[n][k2].f[b], bigB[1][k2].f[b]);
                            }
                            if constexpr(YTILE >= 3)
                            {
                                DOT2C(sum[n][2], bigA[n][k2].f[b], bigB[2][k2].f[b]);
                            }
                            if constexpr(YTILE >= 4)
                            {
                                DOT2C(sum[n][3], bigA[n][k2].f[b], bigB[3][k2].f[b]);
                            }
                        }
                    }
                }
            }
        }

        //----------------------------------------------------
        // Final reduction step using shuffle
        //----------------------------------------------------
        for(int n = 0; n < N; n++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            }
        }
        if(threadIdx.x == 63)
        {
            for(int n = 0; n < N; n++)
            {
                for(int i = 0; i < YTILE; i++)
                {
                    C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
                }
            }
        }

        m += CuCount * _WvPrGrp * YTILE;
    }
}

#else // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void wv_splitk_small_fp16_bf16_kernel(const int K,
                                                 const int M,
                                                 const scalar_t* B,
                                                 const scalar_t* __restrict__ A,
                                                 scalar_t* C,
                                                 const int _WvPrGrp,
                                                 const int CuCount)
{
    UNREACHABLE_CODE
}

#endif // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#if defined(__HIP__MI300_MI250__) // TODO: Add NAVI support
// This version targets cases where A[] fits LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS) wvSplitK_hf_sml_(const int K,
                                                                   const int M,
                                                                   const scalar_t* B,
                                                                   const scalar_t* __restrict__ A,
                                                                   scalar_t* C,
                                                                   const int _WvPrGrp,
                                                                   const int CuCount)
{
    using scalar8 = __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
    union bigType
    {
        scalar_t h[A_CHUNK];
        float f[A_CHUNK / 2];
        float2 f2[A_CHUNK / 4];
        double d[A_CHUNK / 4];
        scalar8 h8;
    };

    //----------------------------------------------------
    // Reserving 64 KB of LDS to have 1 WG / CU
    // Goal is to bring the activation matrix A to the LDS
    // and use it across the lifetime of the work group
    // TODO: When activation matrix is larger than 64 KB
    //	     then this is not goint to work!
    //----------------------------------------------------
    __shared__ scalar_t s[1024 * 32];

    //----------------------------------------------------
    // Fetch the activation matrix to LDS
    // Loop iteration:
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements
    // - Each WG will fetch 512 * 16 => 8K elements
    // - Then the WG will move to another 8 K elements
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for(uint32_t k = 0; k < min(K * N, 32 * 1024); k += THRDS * WvPrGrp * A_CHUNK)
    {
        uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

        if(k_in >= min(K * N, 32 * 1024))
            break;

        *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
    }
    __syncthreads();

    if(threadIdx.y >= _WvPrGrp)
        return;

    uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

    float sum[N][YTILE];

    //----------------------------------------------------
    // Each wave works on a single column of weight matrix.
    // There are 16 waves per WG, and hence, each WG is
    // working on 16 columns of weight matrix. Moreover,
    // we tile in column direction by YTILE, so when YTILE=1
    // the above math is right, however, when YTILE=2 then
    // each wave  will be working on 2 columns and WG will
    // be working on 32 columns.
    //
    // Top level loop that makes WGs persistent!
    // - WGs iterates across columns of weight matrix
    // - Each wave within WG works on a given column(s)
    // - After completing first set of columns, WGs start
    //   working on the next set of available columns
    //----------------------------------------------------
    while(m < M)
    {
        //----------------------------------------------------
        // 'sum' accumulates the matrix A x B computation
        // split across 64 lanes.
        //
        // YTILE represents how many column of weight matrix
        // are being worked on by each wave.
        //----------------------------------------------------
        for(int i = 0; i < YTILE; i++)
            for(int n = 0; n < N; n++)
                sum[n][i] = 0;

        bigType bigA[N][UNRL];
        bigType bigB[YTILE][UNRL];
        //----------------------------------------------------
        // Fetch weight matrix B in interleaved K-split!
        // - Each thread (lane) is fetching 8 elements (A_Chunk)
        // - Each wave will fetch 64*8=> 512 elements (1024B)
        // - YTILE represents the number of column being serviced
        //   by wave
        // - Loop for fetching weight matrix (B) are unrolled
        //
        // Fetch activation matrix A from LDS
        // - Loop for fetching activation matrix (A) are unrolled
        //
        // Finally, do the matrix multiplication in an unrolled
        // fashion. This provides lot of food for compiler
        // scheduling.
        //
        // TODO: Logic below will only work when K is multiple of 8
        //----------------------------------------------------
        // for (uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL) {
        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
            // Fetch the weight matrix from memory!
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const scalar_t* B_ = &B[(m + 0) * K + k_];
                bigB[0][k2].h8     = (loadnt((scalar8*)(&B_[0 * K])));
                //----------------------------------------------------
                // The following code with YTILE > 1 has to be deleted
                //----------------------------------------------------
                if constexpr(YTILE >= 2)
                    bigB[1][k2].h8 = (loadnt((scalar8*)(&B_[1 * K])));
                if constexpr(YTILE >= 3)
                    bigB[2][k2].h8 = (loadnt((scalar8*)(&B_[2 * K])));
                if constexpr(YTILE >= 4)
                    bigB[3][k2].h8 = (loadnt((scalar8*)(&B_[3 * K])));
                if constexpr(YTILE >= 5)
                    bigB[4][k2].h8 = (loadnt((scalar8*)(&B_[4 * K])));
                if constexpr(YTILE >= 6)
                    bigB[5][k2].h8 = (loadnt((scalar8*)(&B_[5 * K])));
                if constexpr(YTILE >= 7)
                    bigB[6][k2].h8 = (loadnt((scalar8*)(&B_[6 * K])));
                if constexpr(YTILE >= 8)
                    bigB[7][k2].h8 = (loadnt((scalar8*)(&B_[7 * K])));
            }

            // Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                // Fetch A activation matrix in interleaved fashion from LDS or memory

                for(int n = 0; n < N; n++)
                {
                    bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
                }
            }

            // Do the matrix multiplication in interleaved manner
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;
                // Do the matrix multiplication of activation and weight matrix
                // - Remember the accumulation is happening for K-split of 64!
#pragma unroll
                for(uint32_t n = 0; n < N; n++)
                {
#pragma unroll
                    for(uint32_t b = 0; b < A_CHUNK / 2; b++)
                    {
                        DOT2C(sum[n][0], bigA[n][k2].f[b], bigB[0][k2].f[b])
                        //----------------------------------------------------
                        // The following code with YTILE > 1
                        //----------------------------------------------------
                        if constexpr(YTILE >= 2)
                        {
                            DOT2C(sum[n][1], bigA[n][k2].f[b], bigB[1][k2].f[b]);
                        }
                        if constexpr(YTILE >= 3)
                        {
                            DOT2C(sum[n][2], bigA[n][k2].f[b], bigB[2][k2].f[b]);
                        }
                        if constexpr(YTILE >= 4)
                        {
                            DOT2C(sum[n][3], bigA[n][k2].f[b], bigB[3][k2].f[b]);
                        }
                        if constexpr(YTILE >= 5)
                        {
                            DOT2C(sum[n][4], bigA[n][k2].f[b], bigB[4][k2].f[b]);
                        }
                        if constexpr(YTILE >= 6)
                        {
                            DOT2C(sum[n][5], bigA[n][k2].f[b], bigB[5][k2].f[b]);
                        }
                        if constexpr(YTILE >= 7)
                        {
                            DOT2C(sum[n][6], bigA[n][k2].f[b], bigB[6][k2].f[b]);
                        }
                        if constexpr(YTILE >= 8)
                        {
                            DOT2C(sum[n][7], bigA[n][k2].f[b], bigB[7][k2].f[b]);
                        }
                    }
                }
            }
        }

        //----------------------------------------------------
        // Final reduction step using shuffle
        //----------------------------------------------------
        for(int n = 0; n < N; n++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            }
        }
        if(threadIdx.x == 63)
        {
            for(int n = 0; n < N; n++)
            {
                for(int i = 0; i < YTILE; i++)
                {
                    // if (commitColumn[i]) C[m + i + n * M] = __float2half(sum[n][i]);
                    C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
                }
            }
        }

        m += CuCount * _WvPrGrp * YTILE;
    }
}
#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void wvSplitK_hf_sml_(const int K,
                                 const int M,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 scalar_t* C,
                                 const int _WvPrGrp,
                                 const int CuCount)
{
    UNREACHABLE_CODE
}
#endif // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#if defined(__HIP__MI300_MI250__) // TODO: Add NAVI support
// This version targets cases where A[] marginally exceeds LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS) wvSplitK_hf_(const int K,
                                                               const int M,
                                                               const scalar_t* B,
                                                               const scalar_t* __restrict__ A,
                                                               scalar_t* C,
                                                               const int _WvPrGrp,
                                                               const int CuCount)
{
    using scalar8 = __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
    union bigType
    {
        scalar_t h[A_CHUNK];
        float f[A_CHUNK / 2];
        float2 f2[A_CHUNK / 4];
        double d[A_CHUNK / 4];
        scalar8 h8;
    };

    //----------------------------------------------------
    // Reserving 64 KB of LDS to have 1 WG / CU
    // Goal is to bring the activation matrix A to the LDS
    // and use it across the lifetime of the work group
    // TODO: When activation matrix is larger than 64 KB
    //	     then this is not goint to work!
    //----------------------------------------------------
    __shared__ scalar_t s[1024 * 32];

    //----------------------------------------------------
    // Computation of columns that need to be committed to memory!
    //----------------------------------------------------
    uint32_t commitColumn[YTILE];
    for(uint32_t i = 0; i < YTILE; i++)
    {
        commitColumn[i] = 1;
    }

    //----------------------------------------------------
    // Indexing function into the column of weight matrix B
    // Algorithm does 64 lane k-splitting / wave and uses
    // WG ID and Thread ID to find the index.
    //----------------------------------------------------
    // int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);
    uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if(m < M && (m + YTILE) >= M)
    {
        uint32_t startColumn = M - YTILE;
        for(uint32_t i = 0; i < (m - startColumn); i++)
        {
            commitColumn[i] = 0;
        }
        m = startColumn;
    }

    //----------------------------------------------------
    // Fetch the activation matrix to LDS
    // Loop iteration:
    // - Each thread (lane) is fetching 8 elements (A_Chunk)
    // - Each wave will fetch 64*8=> 512 elements
    // - Each WG will fetch 512 * 16 => 8K elements
    // - Then the WG will move to another 8 K elements
    // TODO: Logic below will only work when K is multiple of 8
    //----------------------------------------------------
    for(uint32_t k = 0; k < min(K * N, 32 * 1024); k += THRDS * WvPrGrp * A_CHUNK)
    {
        uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

        if(k_in >= min(K * N, 32 * 1024))
            break;

        *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
    }

    __syncthreads();

    if(threadIdx.y >= _WvPrGrp)
        return;

    float sum[N][YTILE];

    //----------------------------------------------------
    // Each wave works on a single column of weight matrix.
    // There are 16 waves per WG, and hence, each WG is
    // working on 16 columns of weight matrix. Moreover,
    // we tile in column direction by YTILE, so when YTILE=1
    // the above math is right, however, when YTILE=2 then
    // each wave  will be working on 2 columns and WG will
    // be working on 32 columns.
    //
    // Top level loop that makes WGs persistent!
    // - WGs iterates across columns of weight matrix
    // - Each wave within WG works on a given column(s)
    // - After completing first set of columns, WGs start
    //   working on the next set of available columns
    //----------------------------------------------------
    while(m < M)
    {
        //----------------------------------------------------
        // 'sum' accumulates the matrix A x B computation
        // split across 64 lanes.
        //
        // YTILE represents how many column of weight matrix
        // are being worked on by each wave.
        //----------------------------------------------------
        for(int i = 0; i < YTILE; i++)
            for(int n = 0; n < N; n++)
                sum[n][i] = 0;

        bigType bigA[N][UNRL];
        bigType bigB[YTILE][UNRL];
        //----------------------------------------------------
        // Fetch weight matrix B in interleaved K-split!
        // - Each thread (lane) is fetching 8 elements (A_Chunk)
        // - Each wave will fetch 64*8=> 512 elements (1024B)
        // - YTILE represents the number of column being serviced
        //   by wave
        // - Loop for fetching weight matrix (B) are unrolled
        //
        // Fetch activation matrix A from LDS
        // - Loop for fetching activation matrix (A) are unrolled
        //
        // Finally, do the matrix multiplication in an unrolled
        // fashion. This provides lot of food for compiler
        // scheduling.
        //
        // TODO: Logic below will only work when K is multiple of 8
        //----------------------------------------------------
        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
            // Fetch the weight matrix from memory!
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const scalar_t* B_ = &B[(m + 0) * K + k_];
                bigB[0][k2].h8     = (loadnt((scalar8*)(&B_[0 * K])));
                //----------------------------------------------------
                // The following code with YTILE > 1 has to be deleted
                //----------------------------------------------------
                if constexpr(YTILE >= 2)
                    bigB[1][k2].h8 = (loadnt((scalar8*)(&B_[1 * K])));
                if constexpr(YTILE >= 3)
                    bigB[2][k2].h8 = (loadnt((scalar8*)(&B_[2 * K])));
                if constexpr(YTILE >= 4)
                    bigB[3][k2].h8 = (loadnt((scalar8*)(&B_[3 * K])));
                if constexpr(YTILE >= 5)
                    bigB[4][k2].h8 = (loadnt((scalar8*)(&B_[4 * K])));
                if constexpr(YTILE >= 6)
                    bigB[5][k2].h8 = (loadnt((scalar8*)(&B_[5 * K])));
                if constexpr(YTILE >= 7)
                    bigB[6][k2].h8 = (loadnt((scalar8*)(&B_[6 * K])));
                if constexpr(YTILE >= 8)
                    bigB[7][k2].h8 = (loadnt((scalar8*)(&B_[7 * K])));
            }

            // Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                // Fetch A activation matrix in interleaved fashion from LDS or memory

                for(int n = 0; n < N; n++)
                {
                    if(k_ + K * n < 32 * 1024)
                        bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
                    else
                        bigA[n][k2] = *((const bigType*)(&(A[k_ + K * n])));
                }
            }

            // Do the matrix multiplication in interleaved manner
#pragma unroll
            for(uint32_t n = 0; n < N; n++)
            {
#pragma unroll
                for(uint32_t k2 = 0; k2 < UNRL; k2++)
                {
                    uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                    uint32_t k_ = k + threadIdx.x * A_CHUNK;
                    if(k_ >= K)
                        break;
                    // Do the matrix multiplication of activation and weight matrix
                    // - Remember the accumulation is happening for K-split of 64!
#pragma unroll
                    for(uint32_t b = 0; b < A_CHUNK / 2; b++)
                    {
                        DOT2C(sum[n][0], bigA[n][k2].f[b], bigB[0][k2].f[b]);
                        //----------------------------------------------------
                        // The following code with YTILE > 1
                        //----------------------------------------------------
                        if constexpr(YTILE >= 2)
                        {
                            DOT2C(sum[n][1], bigA[n][k2].f[b], bigB[1][k2].f[b]);
                        }
                        if constexpr(YTILE >= 3)
                        {
                            DOT2C(sum[n][2], bigA[n][k2].f[b], bigB[2][k2].f[b]);
                        }
                        if constexpr(YTILE >= 4)
                        {
                            DOT2C(sum[n][3], bigA[n][k2].f[b], bigB[3][k2].f[b]);
                        }
                        if constexpr(YTILE >= 5)
                        {
                            DOT2C(sum[n][4], bigA[n][k2].f[b], bigB[4][k2].f[b]);
                        }
                        if constexpr(YTILE >= 6)
                        {
                            DOT2C(sum[n][5], bigA[n][k2].f[b], bigB[5][k2].f[b]);
                        }
                        if constexpr(YTILE >= 7)
                        {
                            DOT2C(sum[n][6], bigA[n][k2].f[b], bigB[6][k2].f[b]);
                        }
                        if constexpr(YTILE >= 8)
                        {
                            DOT2C(sum[n][7], bigA[n][k2].f[b], bigB[7][k2].f[b]);
                        }
                    }
                }
            }
        }

        //----------------------------------------------------
        // Final reduction step using shuffle
        //----------------------------------------------------
        for(int n = 0; n < N; n++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            }
        }

        if(threadIdx.x == 63)
        {
            for(int n = 0; n < N; n++)
            {
                for(int i = 0; i < YTILE; i++)
                {
                    if(commitColumn[i])
                        C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
                }
            }
        }

        m += CuCount * _WvPrGrp * YTILE;

        // Check whether there will be fragmenation!
        // This will happen only for the last wave!
        if(m < M && (m + YTILE) >= M)
        {
            uint32_t startColumn = M - YTILE;
            for(uint32_t i = 0; i < (m - startColumn); i++)
            {
                commitColumn[i] = 0;
            }
            m = startColumn;
        }
    }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void wvSplitK_hf_(const int K,
                             const int M,
                             const scalar_t* B,
                             const scalar_t* __restrict__ A,
                             scalar_t* C,
                             const int _WvPrGrp,
                             const int CuCount)
{
    UNREACHABLE_CODE
}
#endif // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#if defined(__HIP__MI300_MI250__) // TODO: Add NAVI support
// This version targets big A[] cases, where it is much larger than LDS capacity
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS) wvSplitK_hf_big_(const int K,
                                                                   const int M,
                                                                   const scalar_t* B,
                                                                   const scalar_t* __restrict__ A,
                                                                   scalar_t* C,
                                                                   const int _WvPrGrp,
                                                                   const int CuCount)
{
    using scalar8 = __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;

    union bigType
    {
        scalar_t h[A_CHUNK];
        float f[A_CHUNK / 2];
        float2 f2[A_CHUNK / 4];
        double d[A_CHUNK / 4];
        scalar8 h8;
    };

    //----------------------------------------------------
    // Reserving 64 KB of LDS to have 1 WG / CU
    // Goal is to bring the activation matrix A to the LDS
    // and use it across the lifetime of the work group
    // TODO: When activation matrix is larger than 64 KB
    //	     then this is not goint to work!
    //----------------------------------------------------
    __shared__ scalar_t s[1024 * 32];

    //----------------------------------------------------
    // Computation of columns that need to be committed to memory!
    //----------------------------------------------------
    uint32_t commitColumn[YTILE];
    for(uint32_t i = 0; i < YTILE; i++)
    {
        commitColumn[i] = 1;
    }

    // int _WvPrGrp = mindiv(N, CuCount * YTILE, WvPrGrp);
    if(threadIdx.y >= _WvPrGrp)
        return;

    //----------------------------------------------------
    // Indexing function into the column of weight matrix B
    // Algorithm does 64 lane k-splitting / wave and uses
    // WG ID and Thread ID to find the index.
    //----------------------------------------------------
    uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;

    // Check whether there will be fragmenation!
    // This will happen only for the last wave!
    if(m < M && (m + YTILE) >= M)
    {
        uint32_t startColumn = M - YTILE;
        for(uint32_t i = 0; i < (m - startColumn); i++)
        {
            commitColumn[i] = 0;
        }
        m = startColumn;
    }

//----------------------------------------------------
// Fetch the activation matrix to LDS
// Loop iteration:
// - Each thread (lane) is fetching 8 elements (A_Chunk)
// - Each wave will fetch 64*8=> 512 elements
// - Each WG will fetch 512 * 16 => 8K elements
// - Then the WG will move to another 8 K elements
// TODO: Logic below will only work when K is multiple of 8
//----------------------------------------------------
#define PCML
#ifndef PCML
    for(uint32_t k = 0; k < min(K * N, 32 * 1024); k += THRDS * WvPrGrp * A_CHUNK)
    {
        uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

        if(k_in >= min(K * N, 32 * 1024))
            break;

        *((bigType*)(&s[k_in])) = *((bigType*)(&A[k_in]));
    }
    __syncthreads();
#endif

#define TUC (THRDS * UNRL * A_CHUNK)
    uint32_t kBase = 0;
    // find biggest k size that fits in LDS
    uint32_t kFit = (32 * 1024) / N;
    // kFit = (kFit%TWC==0) ? kFit : (kFit-kFit%TWC+TWC); //round up to multiple
    // of TUC
    kFit = (kFit % TUC == 0) ? kFit : (kFit - kFit % TUC); // round up to multiple of TUC
    // if (kFit == 0) kFit = TUC;
    kFit = min(kFit, K);

    float sum[N][YTILE];

//----------------------------------------------------
// Each wave works on a single column of weight matrix.
// There are 16 waves per WG, and hence, each WG is
// working on 16 columns of weight matrix. Moreover,
// we tile in column direction by YTILE, so when YTILE=1
// the above math is right, however, when YTILE=2 then
// each wave  will be working on 2 columns and WG will
// be working on 32 columns.
//
// Top level loop that makes WGs persistent!
// - WGs iterates across columns of weight matrix
// - Each wave within WG works on a given column(s)
// - After completing first set of columns, WGs start
//   working on the next set of available columns
//----------------------------------------------------
#ifdef PCML
    int YW         = (YTILE * _WvPrGrp);
    uint32_t Mrndp = (M % YW == 0) ? M : (M - M % YW + YW);
    while(m < Mrndp)
    {
#else
    while(m < M)
    {
#endif
        //----------------------------------------------------
        // 'sum' accumulates the matrix A x B computation
        // split across 64 lanes.
        //
        // YTILE represents how many column of weight matrix
        // are being worked on by each wave.
        //----------------------------------------------------
        for(int i = 0; i < YTILE; i++)
            for(int n = 0; n < N; n++)
                sum[n][i] = 0;

        bigType bigA[N][UNRL];
        bigType bigB[YTILE][UNRL];
        //----------------------------------------------------
        // Fetch weight matrix B in interleaved K-split!
        // - Each thread (lane) is fetching 8 elements (A_Chunk)
        // - Each wave will fetch 64*8=> 512 elements (1024B)
        // - YTILE represents the number of column being serviced
        //   by wave
        // - Loop for fetching weight matrix (B) are unrolled
        //
        // Fetch activation matrix A from LDS
        // - Loop for fetching activation matrix (A) are unrolled
        //
        // Finally, do the matrix multiplication in an unrolled
        // fashion. This provides lot of food for compiler
        // scheduling.
        //
        // TODO: Logic below will only work when K is multiple of 8
        //----------------------------------------------------
        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
#ifdef PCML
            if((k1 == 0) || (k1 == kBase + kFit))
            { // load next chunk of A[] to LDS
                if(k1 != 0)
                    kBase += kFit;
                __syncthreads();
                for(uint32_t k = 0; k < kFit; k += THRDS * _WvPrGrp * A_CHUNK)
                {
                    uint32_t kOff = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
                    if(kBase + kOff >= K)
                        break;
                    if(kOff >= kFit)
                        break;
                    for(uint32_t n = 0; n < N; n++)
                    {
                        uint32_t k_in           = kBase + n * K + kOff;
                        uint32_t k_ot           = n * kFit + kOff;
                        *((bigType*)(&s[k_ot])) = *((bigType*)(&A[k_in]));
                    }
                }
                __syncthreads();
            }
            if(m >= M)
                continue;
#endif

            // Fetch the weight matrix from memory!
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const scalar_t* B_ = &B[(m + 0) * K + k_];
                bigB[0][k2].h8     = (loadnt((scalar8*)(&B_[0 * K])));
                //----------------------------------------------------
                // The following code with YTILE > 1 has to be deleted
                //----------------------------------------------------
                if constexpr(YTILE >= 2)
                    bigB[1][k2].h8 = (loadnt((scalar8*)(&B_[1 * K])));
                if constexpr(YTILE >= 3)
                    bigB[2][k2].h8 = (loadnt((scalar8*)(&B_[2 * K])));
                if constexpr(YTILE >= 4)
                    bigB[3][k2].h8 = (loadnt((scalar8*)(&B_[3 * K])));
                if constexpr(YTILE >= 5)
                    bigB[4][k2].h8 = (loadnt((scalar8*)(&B_[4 * K])));
                if constexpr(YTILE >= 6)
                    bigB[5][k2].h8 = (loadnt((scalar8*)(&B_[5 * K])));
                if constexpr(YTILE >= 7)
                    bigB[6][k2].h8 = (loadnt((scalar8*)(&B_[6 * K])));
                if constexpr(YTILE >= 8)
                    bigB[7][k2].h8 = (loadnt((scalar8*)(&B_[7 * K])));
            }

            // Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                // Fetch A activation matrix in interleaved fashion from LDS or memory

                for(int n = 0; n < N; n++)
                {
#ifdef PCML
                    bigA[n][k2] = *((const bigType*)(&(s[k_ - kBase + kFit * n])));
#else
                    if(k_ + K * n < 32 * 1024)
                        bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
                    else
                        bigA[n][k2] = *((const bigType*)(&(A[k_ + K * n])));
#endif
                }
            }

            // Do the matrix multiplication in interleaved manner
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;
#pragma unroll
                for(uint32_t n = 0; n < N; n++)
                {
                    // Do the matrix multiplication of activation and weight matrix
                    // - Remember the accumulation is happening for K-split of 64!
#pragma unroll
                    for(uint32_t b = 0; b < A_CHUNK / 2; b++)
                    {
                        DOT2C(sum[n][0], bigA[n][k2].f[b], bigB[0][k2].f[b]);
                        //----------------------------------------------------
                        // The following code with YTILE > 1
                        //----------------------------------------------------
                        if constexpr(YTILE >= 2)
                        {
                            DOT2C(sum[n][1], bigA[n][k2].f[b], bigB[1][k2].f[b]);
                        }
                        if constexpr(YTILE >= 3)
                        {
                            DOT2C(sum[n][2], bigA[n][k2].f[b], bigB[2][k2].f[b]);
                        }
                        if constexpr(YTILE >= 4)
                        {
                            DOT2C(sum[n][3], bigA[n][k2].f[b], bigB[3][k2].f[b]);
                        }
                        if constexpr(YTILE >= 5)
                        {
                            DOT2C(sum[n][4], bigA[n][k2].f[b], bigB[4][k2].f[b]);
                        }
                        if constexpr(YTILE >= 6)
                        {
                            DOT2C(sum[n][5], bigA[n][k2].f[b], bigB[5][k2].f[b]);
                        }
                        if constexpr(YTILE >= 7)
                        {
                            DOT2C(sum[n][6], bigA[n][k2].f[b], bigB[6][k2].f[b]);
                        }
                        if constexpr(YTILE >= 8)
                        {
                            DOT2C(sum[n][7], bigA[n][k2].f[b], bigB[7][k2].f[b]);
                        }
                    }
                }
            }
        }

#ifdef PCML
        if(m >= M)
        {
            m += CuCount * _WvPrGrp * YTILE;
            kBase = 0;
            continue;
        }
#endif

        //----------------------------------------------------
        // Final reduction step using shuffle
        //----------------------------------------------------
        for(int n = 0; n < N; n++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
                asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0"
                    : "=v"(sum[n][y])
                    : "0"(sum[n][y]), "v"(sum[n][y]), "v"(sum[n][y]));
            }
        }

        if(threadIdx.x == 63)
        {
            for(int n = 0; n < N; n++)
            {
                for(int i = 0; i < YTILE; i++)
                {
                    if(commitColumn[i])
                        C[m + i + n * M] = __float2s<scalar_t>(sum[n][i]);
                }
            }
        }

        m += CuCount * _WvPrGrp * YTILE;
        kBase = 0;

        // Check whether there will be fragmenation!
        // This will happen only for the last wave!
        if(m < M && (m + YTILE) >= M)
        {
            uint32_t startColumn = M - YTILE;
            for(uint32_t i = 0; i < (m - startColumn); i++)
            {
                commitColumn[i] = 0;
            }
            m = startColumn;
        }
    }
}
#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void wvSplitK_hf_big_(const int K,
                                 const int M,
                                 const scalar_t* B,
                                 const scalar_t* __restrict__ A,
                                 scalar_t* C,
                                 const int _WvPrGrp,
                                 const int CuCount)
{
    UNREACHABLE_CODE
}
#endif // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

int mindiv(int N, int div1, int div2)
{
    int nPrRnd = div1 * div2;
    int rnds0  = N / nPrRnd;
    nPrRnd -= div1 * 3;
    int rnds3 = N / nPrRnd;
    nPrRnd -= div1;
    int rnds4 = N / nPrRnd;
    nPrRnd -= div1;
    int rnds5 = N / nPrRnd;
    nPrRnd -= div1;
    int rnds6 = N / nPrRnd;
    nPrRnd -= div1;
    int rnds7 = N / nPrRnd;
    nPrRnd -= div1;
    int rnds8 = N / nPrRnd;
    nPrRnd -= div1;
    int rnds9 = N / nPrRnd;
    nPrRnd -= div1;
    int rtn = div2;
    if(rnds0 == rnds3)
        rtn = div2 - 3;
    if(rnds0 == rnds4)
        rtn = div2 - 4;
    if(rnds0 == rnds5)
        rtn = div2 - 5;
    if(rnds0 == rnds6)
        rtn = div2 - 6;
    if(rnds0 == rnds7)
        rtn = div2 - 7;
    if(rnds0 == rnds8)
        rtn = div2 - 8;
    if(rnds0 == rnds9)
        rtn = div2 - 9;
    return rtn;
}

constexpr int MAX_N = 16;
template <typename fptype, int N>
void launch_wv_splitk_small_fp16_bf16_kernel(
    cudaStream_t stream, int K_in, int M_in, fptype* af4, const fptype* bf4, fptype* c, int CuCount)
{
    dim3 grid(CuCount);
    dim3 block(64, 1);
    // hipLaunchKernelGGL((wv_splitk_small_fp16_bf16_kernel<fptype, 64, 1, 1, 8, 4, N>),
    //                    dim3(grid),
    //                    dim3(block),
    //                    0,
    //                    stream,
    //                    K_in,
    //                    M_in,
    //                    af4,
    //                    bf4,
    //                    c,
    //                    1,
    //                    CuCount);
    wv_splitk_small_fp16_bf16_kernel<fptype, 64, 1, 1, 8, 4, N>
        <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, 1, CuCount);
}

template <typename fptype>
using KernelFuncPtr = void (*)(cudaStream_t, int, int, fptype*, const fptype*, fptype*, int);

// generate jump table during compilation (1~MAX_N)
template <typename fptype, int... Is>
static constexpr std::array<KernelFuncPtr<fptype>, MAX_N + 1>
make_jump_table(std::integer_sequence<int, Is...>)
{
    return {{nullptr, &launch_wv_splitk_small_fp16_bf16_kernel<fptype, Is + 1>...}};
}

void wv_splitk_small_fp16_bf16(void* in_a,
                               void* in_b,
                               void* out_c,
                               const int M_in,
                               const int K_in,
                               const int N_in,
                               cudaStream_t stream,
                               const int CuCount,
                               const c10::ScalarType scalar_type)
{
    dim3 grid(CuCount);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "wv_splitk_small_fp16_bf16", [&] {
        using fptype      = typename scalar<scalar_t>::type;
        fptype* af4       = reinterpret_cast<fptype*>(in_a);
        const fptype* bf4 = reinterpret_cast<const fptype*>(in_b);
        fptype* c         = reinterpret_cast<fptype*>(out_c);
        static constexpr auto jump_table =
            make_jump_table<fptype>(std::make_integer_sequence<int, MAX_N>{});

        if(N_in < 1 || N_in > MAX_N)
        {
            throw std::runtime_error("Unsupported N value: " + std::to_string(M_in) + "," +
                                     std::to_string(K_in) + "," + std::to_string(N_in));
        }

        jump_table[N_in](stream, K_in, M_in, af4, bf4, c, CuCount);
    });
}

void wvSplitK_(void* in_a,
               void* in_b,
               void* out_c,
               const int M_in,
               const int K_in,
               const int N_in,
               cudaStream_t stream,
               const int CuCount,
               const c10::ScalarType scalar_type)
{
    dim3 grid(CuCount);

#define WVSPLITK(_WvPrGrp, _YTILEs, _YTILEm, _YTILEb, _UNRLs, _UNRLm, _UNRLb, _N)          \
    {                                                                                      \
        dim3 block(64, _WvPrGrp);                                                          \
        if((K_in * N_in <= 32 * 1024) && (M_in % _YTILEs == 0))                            \
        {                                                                                  \
            int __wvPrGrp = mindiv(M_in, CuCount * _YTILEs, _WvPrGrp);                     \
            wvSplitK_hf_sml_<fptype, 64, _YTILEs, _WvPrGrp, 8, _UNRLs, _N>                 \
                <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, __wvPrGrp, CuCount); \
        }                                                                                  \
        else if(K_in * N_in <= 32 * 1024 * 1.2)                                            \
        {                                                                                  \
            int __wvPrGrp = mindiv(M_in, CuCount * _YTILEm, _WvPrGrp);                     \
            wvSplitK_hf_<fptype, 64, _YTILEm, _WvPrGrp, 8, _UNRLm, _N>                     \
                <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, __wvPrGrp, CuCount); \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            int __wvPrGrp = mindiv(M_in, CuCount * _YTILEb, _WvPrGrp);                     \
            wvSplitK_hf_big_<fptype, 64, _YTILEb, _WvPrGrp, 8, _UNRLb, _N>                 \
                <<<grid, block, 0, stream>>>(K_in, M_in, af4, bf4, c, __wvPrGrp, CuCount); \
        }                                                                                  \
    }

    AT_DISPATCH_REDUCED_FLOATING_TYPES(scalar_type, "wvSplitK", [&] {
        using fptype      = typename scalar<scalar_t>::type;
        fptype* af4       = reinterpret_cast<fptype*>(in_a);
        const fptype* bf4 = reinterpret_cast<const fptype*>(in_b);
        fptype* c         = reinterpret_cast<fptype*>(out_c);
        switch(N_in)
        {
        case 1: WVSPLITK(16, 2, 2, 2, 2, 2, 2, 1) break;
        case 2: WVSPLITK(16, 2, 2, 2, 2, 2, 2, 2) break;
        case 3: WVSPLITK(16, 4, 7, 7, 1, 1, 1, 3) break;
        case 4: WVSPLITK(16, 4, 7, 7, 1, 1, 1, 4) break;
        default:
            throw std::runtime_error("Unsupported N value: " + std::to_string(M_in) + "," +
                                     std::to_string(K_in) + "," + std::to_string(N_in));
        }
    });
}

#if defined(__HIP__MI300__) // TODO: Add NAVI support
template <typename scalar_t,
          typename fp8_t,
          int THRDS,
          int YTILE,
          int WvPrGrp,
          int A_CHUNK,
          int UNRL,
          int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS) wvSplitKQ_hf_sml_(const int K,
                                                                    const int Kp,
                                                                    const int M,
                                                                    const fp8_t* B,
                                                                    const fp8_t* __restrict__ A,
                                                                    scalar_t* C,
                                                                    const float* __restrict__ s_A,
                                                                    const float* __restrict__ s_B,
                                                                    const int _WvPrGrp,
                                                                    const int CuCount)
{
    using scalar8 = __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
    using intx2   = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using intx4   = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    union bigType
    {
        char f8[A_CHUNK];
        char2 c2[A_CHUNK / 2];
        scalar_t h[A_CHUNK / 2];
        float f[A_CHUNK / 4];
        int i[A_CHUNK / 4];
        long l[A_CHUNK / 8];
        intx4 l2[A_CHUNK / 16];
        scalar8 h8;
    };

    __shared__ fp8_t s[1024 * 64];

    for(uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK; k < min(K * N, 64 * 1024);
        k += THRDS * WvPrGrp * A_CHUNK)
    {
        *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
    }
    __syncthreads();

    if(threadIdx.y >= _WvPrGrp)
        return;

    uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    floatx16 sum[N][YTILE];
    float sA = *s_A;
    float sB = *s_B;

    while(m < M)
    {
        for(int i = 0; i < YTILE; i++)
            for(int n = 0; n < N; n++)
                sum[n][i] = {0.f};

        bigType bigA[N][UNRL];
        bigType bigB[YTILE][UNRL];

        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
#pragma unroll
                for(uint32_t n = 0; n < N; ++n)
                    bigA[n][k2].h8 = {0.f};
#pragma unroll
                for(uint32_t y = 0; y < YTILE; ++y)
                    bigB[y][k2].h8 = {0.f};
            }

            // Fetch the weight matrix from memory!
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const fp8_t* B_ = &B[(m + 0) * Kp + k_];
#pragma unroll
                for(uint32_t y = 0; y < YTILE; ++y)
                {
                    bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[y * Kp])));
                }
            }

// Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;
                for(int n = 0; n < N; n++)
                {
                    bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
                }
            }

// Do the matrix multiplication in interleaved manner
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k = k1 + k2 * THRDS * A_CHUNK;
                if(k >= K)
                    break;

                for(uint32_t n = 0; n < N; n++)
                {
                    for(int i = 0; i < A_CHUNK; i += 8)
                    {
                        for(int y = 0; y < YTILE; ++y)
                        {
                            sum[n][y] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                                bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0, 0);
                        }
                    }
                }
            }
        }

        // Final reduction
        for(int n = 0; n < N; n++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                float accm0  = sum[n][y][0];
                float accm16 = sum[n][y][8];
                asm("v_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][1]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][9]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][2]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][10]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][3]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][11]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][4]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][12]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:9 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][5]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:9 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][13]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:10 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][6]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:10 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][14]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:11 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][7]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:11 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][15]), "v"(accm16));
                accm0 += __shfl(accm0, 36);
                accm16 += __shfl(accm16, 52);
                sum[n][y][0] = accm0 + __shfl(accm16, 16);
            }
        }

        if(threadIdx.x == 0)
        {
            for(int n = 0; n < N; n++)
            {
                for(int y = 0; y < YTILE; y++)
                {
                    C[m + y + n * M] = __float2s<scalar_t>(sum[n][y][0] * sA * sB);
                }
            }
        }

        m += CuCount * _WvPrGrp * YTILE;
    }
}
#else  // !defined(__HIP__MI300__) TODO: Add NAVI support
template <typename scalar_t,
          typename fp8_t,
          int THRDS,
          int YTILE,
          int WvPrGrp,
          int A_CHUNK,
          int UNRL,
          int N>
__global__ void wvSplitKQ_hf_sml_(const int K,
                                  const int Kp,
                                  const int M,
                                  const fp8_t* B,
                                  const fp8_t* __restrict__ A,
                                  scalar_t* C,
                                  const float* __restrict__ s_A,
                                  const float* __restrict__ s_B,
                                  const int _WvPrGrp,
                                  const int CuCount)
{
    UNREACHABLE_CODE
}
#endif // defined(__HIP__MI300__) TODO: Add NAVI support

#if defined(__HIP__MI300__) // TODO: Add NAVI support
template <typename scalar_t,
          typename fp8_t,
          int THRDS,
          int YTILE,
          int WvPrGrp,
          int A_CHUNK,
          int UNRL,
          int N>
__global__ void __launch_bounds__(WvPrGrp* THRDS) wvSplitKQ_hf_(const int K,
                                                                const int Kp,
                                                                const int M,
                                                                const fp8_t* B,
                                                                const fp8_t* __restrict__ A,
                                                                scalar_t* C,
                                                                const float* __restrict__ s_A,
                                                                const float* __restrict__ s_B,
                                                                const int _WvPrGrp,
                                                                const int CuCount)
{
    using scalar8 = __attribute__((__vector_size__((A_CHUNK / 4) * sizeof(float)))) float;
    using intx2   = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using intx4   = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    union bigType
    {
        char f8[A_CHUNK];
        char2 c2[A_CHUNK / 2];
        scalar_t h[A_CHUNK / 2];
        float f[A_CHUNK / 4];
        int i[A_CHUNK / 4];
        long l[A_CHUNK / 8];
        intx4 l2[A_CHUNK / 16];
        scalar8 h8;
    };

    __shared__ fp8_t s[1024 * 64];

    for(uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK; k < min(K * N, 64 * 1024);
        k += THRDS * WvPrGrp * A_CHUNK)
    {
        *((bigType*)(&s[k])) = *((bigType*)(&A[k]));
    }
    __syncthreads();

    if(threadIdx.y >= _WvPrGrp)
        return;

    uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

    using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
    floatx16 sum[N][YTILE];
    float sA = *s_A;
    float sB = *s_B;

    while(m < M)
    {
        for(int i = 0; i < YTILE; i++)
            for(int n = 0; n < N; n++)
                sum[n][i] = {0};

        bigType bigA[N][UNRL];
        bigType bigB[YTILE][UNRL];

        for(uint32_t k1 = 0; k1 < K; k1 += THRDS * A_CHUNK * UNRL)
        {
            // Fetch the weight matrix from memory!
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                const fp8_t* B_ = &B[(m + 0) * Kp + k_];
                for(int y = 0; y < YTILE; ++y)
                {
                    if(y + m >= M)
                        break; // To avoid mem access fault.
                    bigB[y][k2].h8 = (loadnt((scalar8*)(&B_[y * Kp])));
                }
            }

// Fetch activation matrix from either just LDS or from both LDS / memory
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;
                for(int n = 0; n < N; n++)
                {
                    if(k_ + K * n < 64 * 1024)
                        bigA[n][k2] = *((const bigType*)(&(s[k_ + K * n])));
                    else
                        bigA[n][k2] = *((const bigType*)(&(A[k_ + K * n])));
                }
            }

// Do the matrix multiplication in interleaved manner
#pragma unroll
            for(uint32_t k2 = 0; k2 < UNRL; k2++)
            {
                uint32_t k  = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if(k_ >= K)
                    break;

                for(uint32_t n = 0; n < N; n++)
                {
                    for(int i = 0; i < A_CHUNK; i += 8)
                    {
                        for(int y = 0; y < YTILE; ++y)
                        {
                            sum[n][y] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                                bigA[n][k2].l[i / 8], bigB[y][k2].l[i / 8], sum[n][y], 0, 0, 0);
                        }
                    }
                }
            }
        }

        // Final reduction
        for(int n = 0; n < N; n++)
        {
            for(int y = 0; y < YTILE; y++)
            {
                float accm0  = sum[n][y][0];
                float accm16 = sum[n][y][8];
                asm("v_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][1]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:1 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][9]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][2]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:2 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][10]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][3]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:3 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][11]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][4]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:8 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][12]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:9 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][5]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:9 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][13]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:10 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][6]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:10 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][14]), "v"(accm16));
                asm("v_add_f32 %0, %2, %3 row_shl:11 bound_ctrl:0 "
                    : "=v"(accm0)
                    : "0"(accm0), "v"(sum[n][y][7]), "v"(accm0));
                asm("v_add_f32 %0, %2, %3 row_shl:11 bound_ctrl:0 "
                    : "=v"(accm16)
                    : "0"(accm16), "v"(sum[n][y][15]), "v"(accm16));
                accm0 += __shfl(accm0, 36);
                accm16 += __shfl(accm16, 52);
                sum[n][y][0] = accm0 + __shfl(accm16, 16);
            }
        }

        if(threadIdx.x == 0)
        {
            for(int n = 0; n < N; n++)
            {
                for(int y = 0; y < YTILE; y++)
                {
                    if(y + m >= M)
                        break; // To avoid mem access fault.
                    C[m + y + n * M] = __float2s<scalar_t>(sum[n][y][0] * sA * sB);
                }
            }
        }

        m += CuCount * _WvPrGrp * YTILE;
    }
}
#else  // !defined(__HIP__MI300__) TODO: Add NAVI support
template <typename scalar_t,
          typename fp8_t,
          int THRDS,
          int YTILE,
          int WvPrGrp,
          int A_CHUNK,
          int UNRL,
          int N>
__global__ void wvSplitKQ_hf_(const int K,
                              const int Kp,
                              const int M,
                              const fp8_t* B,
                              const fp8_t* __restrict__ A,
                              scalar_t* C,
                              const float* __restrict__ s_A,
                              const float* __restrict__ s_B,
                              const int _WvPrGrp,
                              const int CuCount)
{
    UNREACHABLE_CODE
}
#endif // defined(__HIP__MI300__) TODO: Add NAVI support

void wvSplitKQ_(void* in_a,
                void* in_b,
                void* out_c,
                const float* scale_a,
                const float* scale_b,
                const int M_in,
                const int K_in,
                const int Kp_in,
                const int N_in,
                cudaStream_t stream,
                const int CuCount,
                const c10::ScalarType a_scalar_type,
                const c10::ScalarType c_scalar_type)
{
#define WVSPLITKQ(_WvPrGrp, _YTILEs, _YTILEm, _YTILEb, _UNRLs, _UNRLm, _UNRLb, _N)                 \
    {                                                                                              \
        dim3 block(64, _WvPrGrp);                                                                  \
        if((K_in * N_in <= 64 * 1024) && (M_in % _YTILEs == 0))                                    \
        {                                                                                          \
            int __wvPrGrp = mindiv(M_in, CuCount * _YTILEs, _WvPrGrp);                             \
            wvSplitKQ_hf_sml_<fptype, fp8_t, 64, _YTILEs, _WvPrGrp, 16, _UNRLs, _N>                \
                <<<grid, block, 0, stream>>>(                                                      \
                    K_in, Kp_in, M_in, a_ptr, b_ptr, c_ptr, scale_a, scale_b, __wvPrGrp, CuCount); \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            int __wvPrGrp = mindiv(M_in, CuCount * _YTILEm, _WvPrGrp);                             \
            wvSplitKQ_hf_<fptype, fp8_t, 64, _YTILEm, _WvPrGrp, 16, _UNRLm, _N>                    \
                <<<grid, block, 0, stream>>>(                                                      \
                    K_in, Kp_in, M_in, a_ptr, b_ptr, c_ptr, scale_a, scale_b, __wvPrGrp, CuCount); \
        }                                                                                          \
    }

    dim3 grid(CuCount);
    AT_DISPATCH_REDUCED_FLOATING_TYPES(c_scalar_type, "wvSplitKQ", [&] {
        using fptype = typename scalar<scalar_t>::type;
        auto c_ptr   = reinterpret_cast<fptype*>(out_c);
        AITER_DISPATCH_FP8_TYPES(a_scalar_type, "wvSplitKQ", [&] {
            auto a_ptr = reinterpret_cast<fp8_t*>(in_a);
            auto b_ptr = reinterpret_cast<fp8_t*>(in_b);
            switch(N_in)
            {
            case 1: WVSPLITKQ(16, 2, 2, 2, 2, 2, 2, 1) break;
            case 2: WVSPLITKQ(16, 2, 2, 2, 2, 2, 2, 2) break;
            case 3: WVSPLITKQ(16, 4, 7, 7, 1, 1, 1, 3) break;
            case 4: WVSPLITKQ(16, 4, 7, 7, 1, 1, 1, 4) break;
            default:
                throw std::runtime_error("Unsupported N value: " + std::to_string(M_in) + "," +
                                         std::to_string(K_in) + "," + std::to_string(N_in));
            }
        });
    });
}

template <int nThreads_per_row, int CTA, int MT0, int MT1>
__global__ __launch_bounds__(512) void HGEMV_WFPerRow(
    int m, int n, const _Float16* A, int lda, const _Float16* x, _Float16* y)
{
    int num_row_per_block = CTA / nThreads_per_row;
    int row_id            = (blockIdx.x * num_row_per_block + threadIdx.y) * MT0;
    int inc               = (gridDim.x * num_row_per_block) * MT0;

    while(row_id < m)
    {
        float2 sum2[MT0];

#pragma unroll
        for(int i = 0; i < MT0; ++i)
        {
            sum2[i] = {0.0, 0.0};
        }

        for(int j = threadIdx.x; j < n; j += (nThreads_per_row * MT1))
        {
            bool is_active = j < n;
            if(is_active)
            {
                float2 x2[MT1 >> 1];
#pragma unroll
                for(int offset = 0; offset < MT1; offset += 2)
                {
                    x2[offset >> 1] = {x[j + nThreads_per_row * offset],
                                       x[j + nThreads_per_row * (offset + 1)]};
                }
                float2 a2[MT0][MT1 >> 1];
#pragma unroll
                for(int i = 0; i < MT0; i++)
                {
#pragma unroll
                    for(int offset = 0; offset < MT1; offset += 2)
                    {
                        a2[i][offset >> 1] = {
                            A[(row_id + i) * n + j + nThreads_per_row * offset],
                            A[(row_id + i) * n + j + nThreads_per_row * (offset + 1)]};
                    }
                }

#pragma unroll
                for(int i = 0; i < MT0; i++)
                {
#pragma unroll
                    for(int offset = 0; offset < (MT1 >> 1); offset++)
                    {
                        sum2[i] += a2[i][offset] * x2[offset];
                    }
                }
            }
        }
        float sum[MT0];
#pragma unroll
        for(int i = 0; i < MT0; i++)
        {
            sum[i] = sum2[i].x + sum2[i].y;
        }

#pragma unroll
        for(int i = 0; i < MT0; i++)
        {
#pragma unroll
            for(int offset = nThreads_per_row >> 1; offset >= 1; offset = offset >> 1)
            {
                sum[i] += __shfl_down(sum[i], offset, nThreads_per_row);
            }
        }
        if(threadIdx.x == 0)
        {
#pragma unroll
            for(int i = 0; i < MT0; i++)
            {
                y[row_id + i] = sum[i];
            }
        }
        row_id += inc;
    }
}

void LLGemmZZ(void* in_a,
              void* in_b,
              void* out_c,
              const int M,
              const int K,
              cudaStream_t stream,
              const int solidx = 0)
{
    // m -> M, n-> K
    dim3 grid(1024);
    dim3 block(64, 8);
    if(solidx == 0)
    {
        HGEMV_WFPerRow<64, 512, 4, 8>
            <<<grid, block, 0, stream>>>(M,
                                         K,
                                         reinterpret_cast<const _Float16*>(in_a),
                                         K,
                                         reinterpret_cast<const _Float16*>(in_b),
                                         reinterpret_cast<_Float16*>(out_c));
    }
    else if(solidx == 1)
    {
        HGEMV_WFPerRow<64, 512, 2, 8>
            <<<grid, block, 0, stream>>>(M,
                                         K,
                                         reinterpret_cast<const _Float16*>(in_a),
                                         K,
                                         reinterpret_cast<const _Float16*>(in_b),
                                         reinterpret_cast<_Float16*>(out_c));
    }
    else if(solidx == 2)
    {
        HGEMV_WFPerRow<64, 512, 1, 8>
            <<<grid, block, 0, stream>>>(M,
                                         K,
                                         reinterpret_cast<const _Float16*>(in_a),
                                         K,
                                         reinterpret_cast<const _Float16*>(in_b),
                                         reinterpret_cast<_Float16*>(out_c));
    }
    else
    {
        HGEMV_WFPerRow<64, 512, 4, 8>
            <<<grid, block, 0, stream>>>(M,
                                         K,
                                         reinterpret_cast<const _Float16*>(in_a),
                                         K,
                                         reinterpret_cast<const _Float16*>(in_b),
                                         reinterpret_cast<_Float16*>(out_c));
    }
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
        throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

// instantiate the kernel template for T=float:
// template void AddGPUKernel<float>(float *in_a, float *in_b, float *out_c,
// const int M, const int K, cudaStream_t stream);
const unsigned int TILE_WIDTH = 32;
// Compute C = A * B
__global__ void matrixMultiplyShared(float* A,
                                     float* B,
                                     float* C,
                                     int numARows,
                                     int numAColumns,
                                     int numBRows,
                                     int numBColumns,
                                     int numCRows,
                                     int numCColumns)
{
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH]; // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int Row                      = blockDim.y * blockIdx.y + threadIdx.y;
    int Col                      = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue                 = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for(int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++)
    {
        if((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if(Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
        for(int j = 0; j < TILE_WIDTH; ++j)
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if(Row < numCRows && Col < numCColumns)
    {
        C[Row * numCColumns + Col] = Cvalue;
    }
}

void MMGPUKernel(float* in_a,
                 float* in_b,
                 float* out_c,
                 int numARows,
                 int numAColumns,
                 int numBRows,
                 int numBColumns,
                 int numCRows,
                 int numCColumns,
                 cudaStream_t stream)
{
    // Initialize the grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((numCColumns / TILE_WIDTH) + 1, (numCRows / TILE_WIDTH) + 1, 1);
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(
        in_a, in_b, out_c, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
        throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}
} // namespace aiter
