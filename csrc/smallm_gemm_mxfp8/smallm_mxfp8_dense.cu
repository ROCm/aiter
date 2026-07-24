// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Decode small-M MXFP8 GEMV (fp8 e4m3 + e8m0 1x32 block scales), gfx950.
// Tile BLOCK_N=8, 4 warps, WARP_N=2, K_PER_LANE=16 (1024 K/warp-step,
// 32 K-blocks); per-K-block scale via __builtin_amdgcn_ldexpf, scales
// register-resident, named double-buffers (w/ws/xs, never arrays).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

using bf16 = __hip_bfloat16;
using int32x4_t = int __attribute__((ext_vector_type(4)));
using fp8x16_t  = unsigned int __attribute__((ext_vector_type(4)));  // 4 dwords = 16 fp8 bytes
using float2_t  = float __attribute__((ext_vector_type(2)));

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

__device__ uint8_t llvm_amdgcn_raw_buffer_load_ubyte(
    int32x4_t srd, int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.u8");

__device__ __forceinline__ fp8x16_t buffer_load_fp8x16(int32x4_t srd, int32_t voffset_bytes) {
    int32x4_t r = llvm_amdgcn_raw_buffer_load_v4i32(srd, voffset_bytes, 0, 0);
    return __builtin_bit_cast(fp8x16_t, r);
}

__device__ __forceinline__ uint32_t buffer_load_ubyte(int32x4_t srd, int32_t voffset_bytes) {
    return (uint32_t)llvm_amdgcn_raw_buffer_load_ubyte(srd, voffset_bytes, 0, 0);
}

template<int N>
__device__ __forceinline__ void s_wait_vmcnt() {
    constexpr int CN = (N > 63) ? 63 : N;
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(CN) : "memory");
}

// e8m0 scale: 2^(ea + eb - 254).
__device__ __forceinline__ float apply_scale(float val, uint32_t ea, uint32_t eb) {
    int exp = (int)ea + (int)eb - 254;
    return __builtin_amdgcn_ldexpf(val, exp);
}

// Unpack one dwordx4 (16 fp8 bytes) into 16 floats via packed cvt.
__device__ __forceinline__ void unpack_fp8x16_to_f32x16(fp8x16_t v, float* out) {
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        uint32_t dw = v[d];
        float2_t p0 = __builtin_amdgcn_cvt_pk_f32_fp8(dw, false);  // bytes 0,1
        float2_t p1 = __builtin_amdgcn_cvt_pk_f32_fp8(dw, true);   // bytes 2,3
        out[d * 4 + 0] = p0[0];
        out[d * 4 + 1] = p0[1];
        out[d * 4 + 2] = p1[0];
        out[d * 4 + 3] = p1[1];
    }
}

template<int M_TILE, int BLOCK_N>
__global__ __launch_bounds__(256, 2)
void smallm_mxfp8_gemv_kernel_db(
    const unsigned char* __restrict__ Xq,    // [M, K] fp8 e4m3 (raw bytes)
    const unsigned char* __restrict__ Xs,    // [M, K//32] uint8 e8m0
    const unsigned char* __restrict__ Wq,    // [N, K] fp8 e4m3
    const unsigned char* __restrict__ Ws,    // [N, K//32] uint8 e8m0
    bf16*                __restrict__ Out,
    int M, int N, int K)
{
    constexpr int WARPS_PER_WG    = 4;
    constexpr int LANES           = 64;
    constexpr int WARP_N          = BLOCK_N / WARPS_PER_WG;
    constexpr int K_PER_LANE      = 16;
    constexpr int K_PER_WARP_STEP = LANES * K_PER_LANE;
    constexpr int KB_PER_WARP_STEP = K_PER_WARP_STEP / 32;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 6;
    const int lane   = tid & 63;
    const int wg_n   = blockIdx.x * BLOCK_N;
    const int warp_n = wg_n + warp * WARP_N;

    if (warp_n >= N) return;

    float acc[WARP_N][M_TILE];
    #pragma unroll
    for (int c = 0; c < WARP_N; ++c)
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) acc[c][m] = 0.0f;

    // 64-bit intermediate: M*K / N*K can exceed INT_MAX (int*int overflow is UB).
    // The value still feeds a raw-buffer SRD addressed by a signed-int32 voffset,
    // so the host guards each tensor < 2GB (the 2^31 offset limit).
    const uint32_t wq_bytes = (uint32_t)((size_t)N * K);
    const uint32_t xq_bytes = (uint32_t)((size_t)M * K);
    const uint32_t ws_bytes = (uint32_t)((size_t)N * (K / 32));
    const uint32_t xs_bytes = (uint32_t)((size_t)M * (K / 32));
    int32x4_t srd_Wq = make_srd(Wq, wq_bytes);
    int32x4_t srd_Xq = make_srd(Xq, xq_bytes);
    int32x4_t srd_Ws = make_srd(Ws, ws_bytes);
    int32x4_t srd_Xs = make_srd(Xs, xs_bytes);

    const uint32_t wq_warp_base = (uint32_t)(warp_n * K);
    const uint32_t wq_lane_off  = (uint32_t)(lane * K_PER_LANE);
    constexpr uint32_t K_STEP_BYTES_FP8 = K_PER_WARP_STEP;
    constexpr uint32_t KB_STEP = KB_PER_WARP_STEP;
    const uint32_t lane_kb_off  = (uint32_t)(lane >> 1);

    fp8x16_t w_a[WARP_N];
    fp8x16_t w_b[WARP_N];
    fp8x16_t x_reg[M_TILE];

    uint32_t ws_a[WARP_N];
    uint32_t ws_b[WARP_N];
    uint32_t xs_a[M_TILE];
    uint32_t xs_b[M_TILE];

    const int K_STEPS = K / K_PER_WARP_STEP;

    auto load_Wq = [&](fp8x16_t* dst, int k_step) {
        uint32_t k_off = (uint32_t)k_step * K_STEP_BYTES_FP8;
        #pragma unroll
        for (int c = 0; c < WARP_N; ++c) {
            uint32_t off = wq_warp_base + (uint32_t)(c * K) + k_off + wq_lane_off;
            dst[c] = buffer_load_fp8x16(srd_Wq, (int32_t)off);
        }
    };
    auto load_Ws = [&](uint32_t* dst, int k_step) {
        uint32_t kb_off = (uint32_t)k_step * KB_STEP + lane_kb_off;
        #pragma unroll
        for (int c = 0; c < WARP_N; ++c) {
            uint32_t off = (uint32_t)((warp_n + c) * (K / 32)) + kb_off;
            dst[c] = buffer_load_ubyte(srd_Ws, (int32_t)off);
        }
    };
    auto load_Xq = [&](int k_step) {
        uint32_t k_off = (uint32_t)k_step * K_STEP_BYTES_FP8;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < M) {
                uint32_t off = (uint32_t)(m * K) + k_off + wq_lane_off;
                x_reg[m] = buffer_load_fp8x16(srd_Xq, (int32_t)off);
            }
        }
    };
    auto load_Xs = [&](uint32_t* dst, int k_step) {
        uint32_t kb_off = (uint32_t)k_step * KB_STEP + lane_kb_off;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < M) {
                uint32_t off = (uint32_t)(m * (K / 32)) + kb_off;
                dst[m] = buffer_load_ubyte(srd_Xs, (int32_t)off);
            }
        }
    };

    auto fma_block = [&](fp8x16_t* w_buf, uint32_t* ws_buf, uint32_t* xs_buf) {
        float xf[M_TILE][16];
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < M) unpack_fp8x16_to_f32x16(x_reg[m], xf[m]);
        }
        #pragma unroll
        for (int c = 0; c < WARP_N; ++c) {
            float wf[16];
            unpack_fp8x16_to_f32x16(w_buf[c], wf);
            uint32_t ew = ws_buf[c];
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (m < M) {
                    float partial = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < 16; ++e) {
                        partial += xf[m][e] * wf[e];
                    }
                    acc[c][m] += apply_scale(partial, xs_buf[m], ew);
                }
            }
        }
    };

    if (K_STEPS == 0) goto epilogue;

    load_Wq(w_a, 0);
    load_Ws(ws_a, 0);

    {
        int k_step = 0;
        for (; k_step + 1 < K_STEPS; k_step += 2) {
            load_Xq(k_step);
            load_Xs(xs_a, k_step);
            load_Wq(w_b, k_step + 1);
            load_Ws(ws_b, k_step + 1);
            s_wait_vmcnt<2 * WARP_N>();
            fma_block(w_a, ws_a, xs_a);

            int next2 = k_step + 2;
            load_Xq(k_step + 1);
            load_Xs(xs_b, k_step + 1);
            if (next2 < K_STEPS) {
                load_Wq(w_a, next2);
                load_Ws(ws_a, next2);
            }
            s_wait_vmcnt<2 * WARP_N>();
            fma_block(w_b, ws_b, xs_b);
        }
        if (k_step < K_STEPS) {
            load_Xq(k_step);
            load_Xs(xs_a, k_step);
            s_wait_vmcnt<0>();
            fma_block(w_a, ws_a, xs_a);
        }
    }

epilogue:
    const int k_handled = K_STEPS * K_PER_WARP_STEP;
    if (k_handled < K) {
        int k_base = k_handled + lane * K_PER_LANE;
        if (k_base + K_PER_LANE <= K) {
            int kb = k_base / 32;
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                uint32_t ew_local = (uint32_t)Ws[(warp_n + c) * (K / 32) + kb];
                const unsigned int* p = reinterpret_cast<const unsigned int*>(
                    Wq + (warp_n + c) * (size_t)K + k_base);
                int32x4_t r;
                r[0] = (int32_t)p[0]; r[1] = (int32_t)p[1];
                r[2] = (int32_t)p[2]; r[3] = (int32_t)p[3];
                fp8x16_t wv = __builtin_bit_cast(fp8x16_t, r);
                float wf[16]; unpack_fp8x16_to_f32x16(wv, wf);
                #pragma unroll
                for (int m = 0; m < M_TILE; ++m) {
                    if (m < M) {
                        uint32_t ex = (uint32_t)Xs[m * (K / 32) + kb];
                        const unsigned int* xp = reinterpret_cast<const unsigned int*>(
                            Xq + m * (size_t)K + k_base);
                        int32x4_t xr;
                        xr[0] = (int32_t)xp[0]; xr[1] = (int32_t)xp[1];
                        xr[2] = (int32_t)xp[2]; xr[3] = (int32_t)xp[3];
                        fp8x16_t xv = __builtin_bit_cast(fp8x16_t, xr);
                        float xf[16]; unpack_fp8x16_to_f32x16(xv, xf);
                        float partial = 0.0f;
                        #pragma unroll
                        for (int e = 0; e < 16; ++e) partial += xf[e] * wf[e];
                        acc[c][m] += apply_scale(partial, ex, ew_local);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < WARP_N; ++c) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < M) {
                float v = acc[c][m];
                v += __shfl_xor(v, 1);
                v += __shfl_xor(v, 2);
                v += __shfl_xor(v, 4);
                v += __shfl_xor(v, 8);
                v += __shfl_xor(v, 16);
                v += __shfl_xor(v, 32);
                acc[c][m] = v;
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            if (m < M) {
                #pragma unroll
                for (int c = 0; c < WARP_N; ++c) {
                    int n_idx = warp_n + c;
                    if (n_idx < N) {
                        Out[m * (size_t)N + n_idx] = (bf16)acc[c][m];
                    }
                }
            }
        }
    }
}

#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#include "aiter_hip_common.h"  // get_gpu_arch()

#define CHECK_U8(t)  TORCH_CHECK(t.scalar_type() == at::kByte, #t " must be uint8 (e8m0)")
#define CHECK_CUDA(t) TORCH_CHECK(t.is_cuda(), #t " must be CUDA")

template<int M_TILE>
static void launch_kernel_m(const at::Tensor& Xq, const at::Tensor& Xs,
                            const at::Tensor& Wq, const at::Tensor& Ws,
                            at::Tensor& Out,
                            int M, int N, int K, int BLOCK_N) {
    auto stream = c10::hip::getCurrentHIPStream();
    const unsigned char* xqp = reinterpret_cast<const unsigned char*>(Xq.data_ptr());
    const unsigned char* xsp = reinterpret_cast<const unsigned char*>(Xs.data_ptr());
    const unsigned char* wqp = reinterpret_cast<const unsigned char*>(Wq.data_ptr());
    const unsigned char* wsp = reinterpret_cast<const unsigned char*>(Ws.data_ptr());
    bf16* op = reinterpret_cast<bf16*>(Out.data_ptr());

    // K>=6144 small-M: BN=8 -> BN=4 (WARP_N=1) halves accumulator/W VGPR
    // footprint, crossing the 4-waves/SIMD occupancy threshold at M_TILE=4.
    // K<=2048 is X-bandwidth bound, keep BN=8.
    if (BLOCK_N == 8 && K >= 6144 && M_TILE <= 8) {
        BLOCK_N = 4;
    }

    #define DISP(BN) do { \
        if (BLOCK_N == BN) { \
            dim3 grid((N + BN - 1) / BN); \
            dim3 block(256); \
            smallm_mxfp8_gemv_kernel_db<M_TILE, BN><<<grid, block, 0, stream>>>(xqp, xsp, wqp, wsp, op, M, N, K); \
            return; \
        } \
    } while(0)
    DISP(4);
    DISP(8);
    DISP(16);
    DISP(32);
    DISP(64);
    TORCH_CHECK(false, "Unsupported BLOCK_N=", BLOCK_N);
    #undef DISP
}

at::Tensor smallm_mxfp8_gemv(
    at::Tensor Xq,
    at::Tensor Xs,
    at::Tensor Wq,
    at::Tensor Ws,
    c10::ScalarType out_dtype,
    int64_t block_n)
{
    TORCH_CHECK(get_gpu_arch() == "gfx950",
                "smallm_mxfp8_gemv requires a CDNA4 (gfx950) device; got ", get_gpu_arch());
    CHECK_CUDA(Xq); CHECK_CUDA(Xs); CHECK_CUDA(Wq); CHECK_CUDA(Ws);
    CHECK_U8(Xs); CHECK_U8(Ws);
    TORCH_CHECK(out_dtype == at::kBFloat16, "only bf16 out supported");
    int M = Xq.size(0);
    int K = Xq.size(1);
    int N = Wq.size(0);
    TORCH_CHECK(Wq.size(1) == K, "K mismatch");
    // MX-FP8 carries one e8m0 scale per 1x32 K-block; a non-multiple-of-32 K
    // makes K/32 truncate (silently dropping the tail block and mis-sizing the
    // scale stride), so reject it instead of computing a wrong result.
    TORCH_CHECK(K % 32 == 0,
                "K must be a multiple of 32 (1x32 MX scale block); got ", K);
    // Each tensor is addressed via one raw-buffer SRD whose voffset is a SIGNED
    // int32; a byte offset >= 2^31 wraps negative -> OOB fault. Reject >2GB
    // tensors before launch (a GPU fault is uncatchable). (>2GB would need a
    // per-row/per-tile base-pointer advance.)
    TORCH_CHECK((size_t)N * K <= 0x7FFFFFFFull && (size_t)M * K <= 0x7FFFFFFFull,
                "smallm_mxfp8_gemv: a tensor exceeds the 2GB raw-buffer offset limit");
    TORCH_CHECK(Xs.size(0) == M && Xs.size(1) == K / 32, "Xs shape mismatch");
    TORCH_CHECK(Ws.size(0) == N && Ws.size(1) == K / 32, "Ws shape mismatch");
    auto Out = at::empty({M, N}, at::TensorOptions().dtype(out_dtype).device(Xq.device()));
    int BLOCK_N = (int)block_n;
    switch (M) {
        case 1:  launch_kernel_m<1> (Xq, Xs, Wq, Ws, Out, M, N, K, BLOCK_N); break;
        case 2:  launch_kernel_m<2> (Xq, Xs, Wq, Ws, Out, M, N, K, BLOCK_N); break;
        case 4:  launch_kernel_m<4> (Xq, Xs, Wq, Ws, Out, M, N, K, BLOCK_N); break;
        case 8:  launch_kernel_m<8> (Xq, Xs, Wq, Ws, Out, M, N, K, BLOCK_N); break;
        case 16: launch_kernel_m<16>(Xq, Xs, Wq, Ws, Out, M, N, K, BLOCK_N); break;
        default: TORCH_CHECK(false, "Unsupported M=", M);
    }
    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallm_mxfp8_gemv", &smallm_mxfp8_gemv,
          "small-M MXFP8 GEMV (X@W.T) with E8M0 1x32 scales",
          pybind11::arg("Xq"), pybind11::arg("Xs"),
          pybind11::arg("Wq"), pybind11::arg("Ws"),
          pybind11::arg("out_dtype") = at::kBFloat16,
          pybind11::arg("block_n") = 8);
}
