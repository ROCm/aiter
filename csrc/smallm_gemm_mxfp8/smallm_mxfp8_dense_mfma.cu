// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Decode small-M MXFP8 MFMA crossover GEMM (fp8 e4m3 + e8m0 1x32 scales),
// M in {8,16,32,64}, gfx950. BLOCK_M=16, BLOCK_N=N_SUB*16, 1 warp/WG; 4
// scaled-MFMA calls per K-iter (one OPSEL per 32-K scale byte). Split-K via
// gridDim.y (K_SPLITS) and M-tiling via gridDim.z (ceil(M/16) tiles, each an
// independent 16-row GEMM). K_SPLITS==1 writes bf16 to Out; K_SPLITS>1 writes
// fp32 partials reduced to bf16 by a follow-up kernel.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

using bf16 = __hip_bfloat16;
using int32x4_t = int  __attribute__((ext_vector_type(4)));
using int32x8_t = int  __attribute__((ext_vector_type(8)));
using float32x4_t = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int32x4_t make_srd(const void* base_ptr, uint32_t num_bytes) {
    struct __attribute__((packed)) {
        const void* p;
        uint32_t r, c;
    } res{base_ptr, num_bytes, 0x00020000u};
    int32x4_t srd = __builtin_bit_cast(int32x4_t, res);
    #pragma unroll
    for (int i = 0; i < 4; ++i) srd[i] = __builtin_amdgcn_readfirstlane(srd[i]);
    return srd;
}

__device__ int32x4_t llvm_amdgcn_raw_buffer_load_v4i32(
    int32x4_t srd, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ int32_t llvm_amdgcn_raw_buffer_load_dword(
    int32x4_t srd, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ __forceinline__ int32x4_t buffer_load_v4i32(int32x4_t srd, int32_t voffset_bytes) {
    return llvm_amdgcn_raw_buffer_load_v4i32(srd, voffset_bytes, 0, 0);
}
__device__ __forceinline__ uint32_t buffer_load_dword(int32x4_t srd, int32_t voffset_bytes) {
    return (uint32_t)llvm_amdgcn_raw_buffer_load_dword(srd, voffset_bytes, 0, 0);
}

template<int N>
__device__ __forceinline__ void s_wait_vmcnt() {
    constexpr int CN = (N > 63) ? 63 : N;
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(CN) : "memory");
}

template<int OPSEL>
__device__ __forceinline__ float32x4_t mfma_scale_opsel(int32x8_t a, int32x8_t b, float32x4_t c,
                                                       uint32_t scale_a, uint32_t scale_b) {
#if defined(__gfx950__)
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, OPSEL, scale_a, OPSEL, scale_b);
#else
    (void)a; (void)b; (void)scale_a; (void)scale_b;
    return c;
#endif
}

__device__ __forceinline__ int32x8_t mask_and(int32x8_t v, uint32_t mask) {
    int32x8_t r;
    r[0] = v[0] & (int32_t)mask;
    r[1] = v[1] & (int32_t)mask;
    r[2] = v[2] & (int32_t)mask;
    r[3] = v[3] & (int32_t)mask;
    r[4] = v[4] & (int32_t)mask;
    r[5] = v[5] & (int32_t)mask;
    r[6] = v[6] & (int32_t)mask;
    r[7] = v[7] & (int32_t)mask;
    return r;
}

template<int M_TILE, int N_SUB, int K_SPLITS>
__global__ __launch_bounds__(64, 4)
void smallm_mxfp8_mfma_kernel_db(
    const unsigned char* __restrict__ Xq,
    const unsigned char* __restrict__ Xs,
    const unsigned char* __restrict__ Wq,
    const unsigned char* __restrict__ Ws,
    bf16*                __restrict__ Out,
    float*               __restrict__ Partial,
    int M, int N, int K)
{
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;
    constexpr int BLOCK_N = N_SUB * MFMA_N;

    static_assert(M_TILE == 16, "MFMA kernel: M_TILE fixed at 16 (one MFMA tile/WG); M>16 handled by gridDim.z m-tiling");
    static_assert(K_SPLITS >= 1 && K_SPLITS <= 16, "K_SPLITS in [1, 16]");

    const int lane_id  = threadIdx.x;
    const int wg_n     = blockIdx.x * BLOCK_N;
    const int split_id = (K_SPLITS == 1) ? 0 : (int)blockIdx.y;
    // Grid M-tiling: each WG owns one 16-row tile starting at m_base.
    const int m_base   = (int)blockIdx.z * MFMA_M;

    if (wg_n >= N) return;
    if (m_base >= M) return;

    const int m_row   = lane_id % 16;       // local row within this 16-row tile
    const int g_row   = m_base + m_row;      // global A/output row
    const int k_chunk = lane_id / 16;

    const uint32_t mask_kb0 = (k_chunk == 0) ? 0xFFFFFFFFu : 0u;
    const uint32_t mask_kb1 = (k_chunk == 1) ? 0xFFFFFFFFu : 0u;
    const uint32_t mask_kb2 = (k_chunk == 2) ? 0xFFFFFFFFu : 0u;
    const uint32_t mask_kb3 = (k_chunk == 3) ? 0xFFFFFFFFu : 0u;

    const uint32_t xq_bytes = (uint32_t)((size_t)M * K);
    const uint32_t xs_bytes = (uint32_t)((size_t)M * (K / 32));
    const uint32_t wq_bytes = (uint32_t)((size_t)N * K);
    const uint32_t ws_bytes = (uint32_t)((size_t)N * (K / 32));
    int32x4_t srd_Xq = make_srd(Xq, xq_bytes);
    int32x4_t srd_Xs = make_srd(Xs, xs_bytes);
    int32x4_t srd_Wq = make_srd(Wq, wq_bytes);
    int32x4_t srd_Ws = make_srd(Ws, ws_bytes);

    float32x4_t acc[N_SUB];
    #pragma unroll
    for (int s = 0; s < N_SUB; ++s) acc[s] = (float32x4_t){0.f, 0.f, 0.f, 0.f};

    const uint32_t a_row_off   = (uint32_t)(g_row * K);
    const uint32_t a_chunk_off = (uint32_t)(k_chunk * 32);

    const int K_ITERS_TOTAL = K / MFMA_K;
    const int K_ITERS_PER_SPLIT = K_ITERS_TOTAL / K_SPLITS;
    const int k_it_start = split_id * K_ITERS_PER_SPLIT;
    const int k_it_end   = (split_id == K_SPLITS - 1) ? K_ITERS_TOTAL : (k_it_start + K_ITERS_PER_SPLIT);

    auto load_A = [&](int k_iter) -> int32x8_t {
        uint32_t base_off = a_row_off + (uint32_t)(k_iter * MFMA_K) + a_chunk_off;
        int32x4_t lo = buffer_load_v4i32(srd_Xq, (int32_t)base_off);
        int32x4_t hi = buffer_load_v4i32(srd_Xq, (int32_t)(base_off + 16));
        int32x8_t r;
        r[0]=lo[0]; r[1]=lo[1]; r[2]=lo[2]; r[3]=lo[3];
        r[4]=hi[0]; r[5]=hi[1]; r[6]=hi[2]; r[7]=hi[3];
        return r;
    };

    auto load_B_sub = [&](int k_iter, int n_sub) -> int32x8_t {
        int w_row = wg_n + n_sub * MFMA_N + m_row;
        uint32_t base_off = (uint32_t)(w_row * K) + (uint32_t)(k_iter * MFMA_K) + a_chunk_off;
        int32x4_t lo = buffer_load_v4i32(srd_Wq, (int32_t)base_off);
        int32x4_t hi = buffer_load_v4i32(srd_Wq, (int32_t)(base_off + 16));
        int32x8_t r;
        r[0]=lo[0]; r[1]=lo[1]; r[2]=lo[2]; r[3]=lo[3];
        r[4]=hi[0]; r[5]=hi[1]; r[6]=hi[2]; r[7]=hi[3];
        return r;
    };

    auto load_SA = [&](int k_iter) -> uint32_t {
        uint32_t off = (uint32_t)(g_row * (K / 32)) + (uint32_t)(k_iter * 4);
        return buffer_load_dword(srd_Xs, (int32_t)off);
    };
    auto load_SB_sub = [&](int k_iter, int n_sub) -> uint32_t {
        int w_row = wg_n + n_sub * MFMA_N + m_row;
        uint32_t off = (uint32_t)(w_row * (K / 32)) + (uint32_t)(k_iter * 4);
        return buffer_load_dword(srd_Ws, (int32_t)off);
    };

    int32x8_t a_a, a_b;
    int32x8_t b_a[N_SUB], b_b[N_SUB];
    uint32_t  sa_a, sa_b;
    uint32_t  sb_a[N_SUB], sb_b[N_SUB];

    if (k_it_end > k_it_start) {
        a_a = load_A(k_it_start);
        sa_a = load_SA(k_it_start);
        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            b_a[s]  = load_B_sub(k_it_start, s);
            sb_a[s] = load_SB_sub(k_it_start, s);
        }
    }

    int k_it = k_it_start;
    for (; k_it + 1 < k_it_end; k_it += 2) {
        a_b = load_A(k_it + 1);
        sa_b = load_SA(k_it + 1);
        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            b_b[s]  = load_B_sub(k_it + 1, s);
            sb_b[s] = load_SB_sub(k_it + 1, s);
        }

        s_wait_vmcnt<2 * N_SUB + 1>();

        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            acc[s] = mfma_scale_opsel<0>(mask_and(a_a, mask_kb0), mask_and(b_a[s], mask_kb0), acc[s], sa_a, sb_a[s]);
            acc[s] = mfma_scale_opsel<1>(mask_and(a_a, mask_kb1), mask_and(b_a[s], mask_kb1), acc[s], sa_a, sb_a[s]);
            acc[s] = mfma_scale_opsel<2>(mask_and(a_a, mask_kb2), mask_and(b_a[s], mask_kb2), acc[s], sa_a, sb_a[s]);
            acc[s] = mfma_scale_opsel<3>(mask_and(a_a, mask_kb3), mask_and(b_a[s], mask_kb3), acc[s], sa_a, sb_a[s]);
        }

        int next2 = k_it + 2;
        if (next2 < k_it_end) {
            a_a = load_A(next2);
            sa_a = load_SA(next2);
            #pragma unroll
            for (int s = 0; s < N_SUB; ++s) {
                b_a[s]  = load_B_sub(next2, s);
                sb_a[s] = load_SB_sub(next2, s);
            }
        }

        s_wait_vmcnt<2 * N_SUB + 1>();

        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            acc[s] = mfma_scale_opsel<0>(mask_and(a_b, mask_kb0), mask_and(b_b[s], mask_kb0), acc[s], sa_b, sb_b[s]);
            acc[s] = mfma_scale_opsel<1>(mask_and(a_b, mask_kb1), mask_and(b_b[s], mask_kb1), acc[s], sa_b, sb_b[s]);
            acc[s] = mfma_scale_opsel<2>(mask_and(a_b, mask_kb2), mask_and(b_b[s], mask_kb2), acc[s], sa_b, sb_b[s]);
            acc[s] = mfma_scale_opsel<3>(mask_and(a_b, mask_kb3), mask_and(b_b[s], mask_kb3), acc[s], sa_b, sb_b[s]);
        }
    }
    if (k_it < k_it_end) {
        s_wait_vmcnt<0>();
        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            acc[s] = mfma_scale_opsel<0>(mask_and(a_a, mask_kb0), mask_and(b_a[s], mask_kb0), acc[s], sa_a, sb_a[s]);
            acc[s] = mfma_scale_opsel<1>(mask_and(a_a, mask_kb1), mask_and(b_a[s], mask_kb1), acc[s], sa_a, sb_a[s]);
            acc[s] = mfma_scale_opsel<2>(mask_and(a_a, mask_kb2), mask_and(b_a[s], mask_kb2), acc[s], sa_a, sb_a[s]);
            acc[s] = mfma_scale_opsel<3>(mask_and(a_a, mask_kb3), mask_and(b_a[s], mask_kb3), acc[s], sa_a, sb_a[s]);
        }
    }

    const int out_col_base = lane_id % 16;
    const int row_base     = (lane_id / 16) * 4;

    if (K_SPLITS == 1) {
        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            int n_idx = wg_n + s * MFMA_N + out_col_base;
            if (n_idx < N) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row = m_base + row_base + i;   // global output row
                    if (row < M) {
                        Out[row * (size_t)N + n_idx] = (bf16)acc[s][i];
                    }
                }
            }
        }
    } else {
        const size_t split_stride = (size_t)M * (size_t)N;
        float* split_base = Partial + (size_t)split_id * split_stride;
        #pragma unroll
        for (int s = 0; s < N_SUB; ++s) {
            int n_idx = wg_n + s * MFMA_N + out_col_base;
            if (n_idx < N) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row = m_base + row_base + i;   // global output row
                    if (row < M) {
                        split_base[row * (size_t)N + n_idx] = acc[s][i];
                    }
                }
            }
        }
    }
}

template<int K_SPLITS>
__global__ void smallm_mxfp8_mfma_reduce_kernel(
    const float* __restrict__ Partial,
    bf16*        __restrict__ Out,
    int M, int N)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = M * N;
    if (idx >= total) return;
    const size_t split_stride = (size_t)M * (size_t)N;
    float acc = 0.f;
    #pragma unroll
    for (int s = 0; s < K_SPLITS; ++s) {
        acc += Partial[(size_t)s * split_stride + (size_t)idx];
    }
    Out[idx] = (bf16)acc;
}

#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#include "aiter_hip_common.h"  // get_gpu_arch()

#define CHECK_U8(t)  TORCH_CHECK(t.scalar_type() == at::kByte, #t " must be uint8 (e8m0)")
#define CHECK_CUDA(t) TORCH_CHECK(t.is_cuda(), #t " must be CUDA")

template<int N_SUB, int K_SPLITS>
static void launch_kernel_specialized(const at::Tensor& Xq, const at::Tensor& Xs,
                                       const at::Tensor& Wq, const at::Tensor& Ws,
                                       at::Tensor& Out,
                                       int M, int N, int K) {
    constexpr int M_TILE = 16;
    auto stream = c10::hip::getCurrentHIPStream();
    const unsigned char* xqp = reinterpret_cast<const unsigned char*>(Xq.data_ptr());
    const unsigned char* xsp = reinterpret_cast<const unsigned char*>(Xs.data_ptr());
    const unsigned char* wqp = reinterpret_cast<const unsigned char*>(Wq.data_ptr());
    const unsigned char* wsp = reinterpret_cast<const unsigned char*>(Ws.data_ptr());
    bf16* op = reinterpret_cast<bf16*>(Out.data_ptr());

    constexpr int BLOCK_N = N_SUB * 16;
    // gridDim.z = number of 16-row M-tiles. Each WG owns one tile. This is the
    // iter-2 occupancy lever: M=64 launches 4x the WGs of M=16.
    const int num_m_tiles = (M + M_TILE - 1) / M_TILE;

    if constexpr (K_SPLITS == 1) {
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, 1, num_m_tiles);
        dim3 block(64);
        smallm_mxfp8_mfma_kernel_db<M_TILE, N_SUB, 1><<<grid, block, 0, stream>>>(
            xqp, xsp, wqp, wsp, op, nullptr, M, N, K);
    } else {
        auto partial = at::empty({K_SPLITS, M, N},
            at::TensorOptions().dtype(at::kFloat).device(Xq.device()));
        float* pp = reinterpret_cast<float*>(partial.data_ptr());
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, K_SPLITS, num_m_tiles);
        dim3 block(64);
        smallm_mxfp8_mfma_kernel_db<M_TILE, N_SUB, K_SPLITS><<<grid, block, 0, stream>>>(
            xqp, xsp, wqp, wsp, nullptr, pp, M, N, K);
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        smallm_mxfp8_mfma_reduce_kernel<K_SPLITS><<<blocks, threads, 0, stream>>>(
            pp, op, M, N);
    }
}

template<int N_SUB>
static void launch_kernel_ks(const at::Tensor& Xq, const at::Tensor& Xs,
                              const at::Tensor& Wq, const at::Tensor& Ws,
                              at::Tensor& Out,
                              int M, int N, int K, int k_splits) {
    switch (k_splits) {
        case 1: launch_kernel_specialized<N_SUB, 1>(Xq, Xs, Wq, Ws, Out, M, N, K); break;
        case 2: launch_kernel_specialized<N_SUB, 2>(Xq, Xs, Wq, Ws, Out, M, N, K); break;
        case 4: launch_kernel_specialized<N_SUB, 4>(Xq, Xs, Wq, Ws, Out, M, N, K); break;
        case 8: launch_kernel_specialized<N_SUB, 8>(Xq, Xs, Wq, Ws, Out, M, N, K); break;
        default: TORCH_CHECK(false, "Unsupported K_SPLITS=", k_splits);
    }
}

static void launch_kernel_m(const at::Tensor& Xq, const at::Tensor& Xs,
                            const at::Tensor& Wq, const at::Tensor& Ws,
                            at::Tensor& Out,
                            int M, int N, int K, int n_sub, int k_splits) {
    switch (n_sub) {
        case 1: launch_kernel_ks<1>(Xq, Xs, Wq, Ws, Out, M, N, K, k_splits); break;
        case 2: launch_kernel_ks<2>(Xq, Xs, Wq, Ws, Out, M, N, K, k_splits); break;
        case 4: launch_kernel_ks<4>(Xq, Xs, Wq, Ws, Out, M, N, K, k_splits); break;
        default: TORCH_CHECK(false, "Unsupported N_SUB=", n_sub);
    }
}

at::Tensor smallm_mxfp8_mfma(
    at::Tensor Xq,
    at::Tensor Xs,
    at::Tensor Wq,
    at::Tensor Ws,
    c10::ScalarType out_dtype,
    int64_t n_sub,
    int64_t k_splits)
{
    TORCH_CHECK(get_gpu_arch() == "gfx950",
                "smallm_mxfp8_mfma requires a CDNA4 (gfx950) device; got ", get_gpu_arch());
    CHECK_CUDA(Xq); CHECK_CUDA(Xs); CHECK_CUDA(Wq); CHECK_CUDA(Ws);
    CHECK_U8(Xs); CHECK_U8(Ws);
    TORCH_CHECK(out_dtype == at::kBFloat16, "only bf16 out supported");
    int M = Xq.size(0);
    int K = Xq.size(1);
    int N = Wq.size(0);
    // Each tensor is addressed via one raw-buffer SRD whose voffset is a SIGNED
    // int32; a byte offset >= 2^31 wraps negative -> OOB fault. Reject >2GB
    // tensors before launch (a GPU fault is uncatchable).
    TORCH_CHECK((size_t)N * K <= 0x7FFFFFFFull && (size_t)M * K <= 0x7FFFFFFFull,
                "smallm_mxfp8_mfma: a tensor exceeds the 2GB raw-buffer offset limit");
    TORCH_CHECK(Wq.size(1) == K, "K mismatch");
    TORCH_CHECK(Xs.size(0) == M && Xs.size(1) == K / 32, "Xs shape mismatch");
    TORCH_CHECK(Ws.size(0) == N && Ws.size(1) == K / 32, "Ws shape mismatch");
    TORCH_CHECK(K % 128 == 0, "K must be multiple of 128");
    int K_ITERS_TOTAL = K / 128;
    TORCH_CHECK(K_ITERS_TOTAL % (int)k_splits == 0,
        "K_ITERS_TOTAL=", K_ITERS_TOTAL, " must be divisible by k_splits=", k_splits);
    auto Out = at::empty({M, N}, at::TensorOptions().dtype(out_dtype).device(Xq.device()));
    int NSUB = (int)n_sub;
    int KS   = (int)k_splits;
    // M-tiling (gridDim.z) supports any M that is a multiple of an MFMA tile's
    // grid coverage; we expose the padded tiles {8,16,32,64}. M=8 uses one 16-row
    // tile (top 8 rows valid, bottom 8 masked by the `row < M` store guard).
    TORCH_CHECK(M == 8 || M == 16 || M == 32 || M == 64,
                "Unsupported M=", M, " for MFMA kernel (use {8, 16, 32, 64})");
    launch_kernel_m(Xq, Xs, Wq, Ws, Out, M, N, K, NSUB, KS);
    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallm_mxfp8_mfma", &smallm_mxfp8_mfma,
          "MXFP8 MFMA crossover kernel for M in {8, 16, 32, 64}",
          pybind11::arg("Xq"), pybind11::arg("Xs"),
          pybind11::arg("Wq"), pybind11::arg("Ws"),
          pybind11::arg("out_dtype") = at::kBFloat16,
          pybind11::arg("n_sub") = 1,
          pybind11::arg("k_splits") = 1);
}
