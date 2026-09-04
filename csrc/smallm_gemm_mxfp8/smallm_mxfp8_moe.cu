// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Decode small-M MXFP8 MoE grouped GEMM (fp8 e4m3 + e8m0 1x32 scales), gfx950.
// Dense-GEMV tile (BLOCK_N=8, 4 warps, WARP_N=2, K_PER_LANE=16) plus MoE
// plumbing: per-tile expert dispatch (expert_ids[pid_m], skip if invalid),
// per-row gather (sorted_token_ids, a_row=offs_token/a_div), masked per-token
// store with optional mul_weight, inner row-stride loop over BLOCK_M.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

using bf16 = __hip_bfloat16;
using int32x4_t = int __attribute__((ext_vector_type(4)));
using fp8x16_t  = unsigned int __attribute__((ext_vector_type(4)));
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

__device__ __forceinline__ float apply_scale(float val, uint32_t ea, uint32_t eb) {
    int exp = (int)ea + (int)eb - 254;
    return __builtin_amdgcn_ldexpf(val, exp);
}

__device__ __forceinline__ void unpack_fp8x16_to_f32x16(fp8x16_t v, float* out) {
    #pragma unroll
    for (int d = 0; d < 4; ++d) {
        uint32_t dw = v[d];
        float2_t p0 = __builtin_amdgcn_cvt_pk_f32_fp8(dw, false);
        float2_t p1 = __builtin_amdgcn_cvt_pk_f32_fp8(dw, true);
        out[d * 4 + 0] = p0[0];
        out[d * 4 + 1] = p0[1];
        out[d * 4 + 2] = p1[0];
        out[d * 4 + 3] = p1[1];
    }
}

template<int M_TILE, int BLOCK_N, int A_DIV, bool MUL_WEIGHT>
__global__ __launch_bounds__(256, 2)
void smallm_mxfp8_moe_gemm_kernel(
    const unsigned char* __restrict__ Aq,
    const unsigned char* __restrict__ As,
    const unsigned char* __restrict__ Bq,
    const unsigned char* __restrict__ Bs,
    bf16*                __restrict__ Out,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float*   __restrict__ topk_weights,
    int E, int N, int K, int num_valid_tokens, int M_act, int block_m)
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
    const int pid_m  = blockIdx.x;
    const int pid_n  = blockIdx.y;
    const int wg_n   = pid_n * BLOCK_N;
    const int warp_n = wg_n + warp * WARP_N;

    if (warp_n >= N) return;

    int num_post = num_tokens_post_padded[0];
    int tile_base = pid_m * block_m;
    if (tile_base >= num_post) return;
    int off_e = expert_ids[pid_m];
    if (off_e < 0 || off_e >= E) return;

    const uint32_t bq_bytes = (uint32_t)((size_t)E * N * K);
    const uint32_t aq_bytes = (uint32_t)((size_t)M_act * K);
    const uint32_t bs_bytes = (uint32_t)((size_t)E * N * (K / 32));
    const uint32_t as_bytes = (uint32_t)((size_t)M_act * (K / 32));
    int32x4_t srd_Bq = make_srd(Bq, bq_bytes);
    int32x4_t srd_Aq = make_srd(Aq, aq_bytes);
    int32x4_t srd_Bs = make_srd(Bs, bs_bytes);
    int32x4_t srd_As = make_srd(As, as_bytes);

    const uint32_t bq_expert_base = (uint32_t)((size_t)off_e * N * K);
    const uint32_t bq_warp_base   = bq_expert_base + (uint32_t)(warp_n * K);
    const uint32_t bq_lane_off    = (uint32_t)(lane * K_PER_LANE);
    constexpr uint32_t K_STEP_BYTES_FP8 = K_PER_WARP_STEP;
    constexpr uint32_t KB_STEP = KB_PER_WARP_STEP;
    const uint32_t lane_kb_off  = (uint32_t)(lane >> 1);
    const uint32_t bs_expert_base = (uint32_t)((size_t)off_e * N * (K / 32));

    const int K_STEPS = K / K_PER_WARP_STEP;
    const int K_TAIL_START = K_STEPS * K_PER_WARP_STEP;

    int num_row_groups = block_m / M_TILE;

    for (int rg = 0; rg < num_row_groups; ++rg) {
        int row_base = tile_base + rg * M_TILE;

        int32_t offs_token_raw[M_TILE];
        int32_t a_row[M_TILE];
        bool    valid[M_TILE];
        bool    any_valid = false;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            int32_t t = sorted_token_ids[row_base + m];
            offs_token_raw[m] = t;
            bool v = (t < num_valid_tokens);
            valid[m] = v;
            a_row[m] = v ? (t / A_DIV) : 0;
            any_valid = any_valid || v;
        }
        if (!any_valid) break;  // real tokens contiguous at tile start

        float wgt[M_TILE];
        if (MUL_WEIGHT) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                wgt[m] = valid[m] ? topk_weights[offs_token_raw[m]] : 0.0f;
            }
        }

        float acc[WARP_N][M_TILE];
        #pragma unroll
        for (int c = 0; c < WARP_N; ++c)
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) acc[c][m] = 0.0f;

        fp8x16_t w_a[WARP_N];
        fp8x16_t w_b[WARP_N];
        fp8x16_t x_reg[M_TILE];
        uint32_t ws_a[WARP_N];
        uint32_t ws_b[WARP_N];
        uint32_t xs_a[M_TILE];
        uint32_t xs_b[M_TILE];

        auto load_Bq = [&](fp8x16_t* dst, int k_step) {
            uint32_t k_off = (uint32_t)k_step * K_STEP_BYTES_FP8;
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                uint32_t off = bq_warp_base + (uint32_t)(c * K) + k_off + bq_lane_off;
                dst[c] = buffer_load_fp8x16(srd_Bq, (int32_t)off);
            }
        };
        auto load_Bs = [&](uint32_t* dst, int k_step) {
            uint32_t kb_off = (uint32_t)k_step * KB_STEP + lane_kb_off;
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                uint32_t off = bs_expert_base + (uint32_t)((warp_n + c) * (K / 32)) + kb_off;
                dst[c] = buffer_load_ubyte(srd_Bs, (int32_t)off);
            }
        };
        auto load_Aq = [&](int k_step) {
            uint32_t k_off = (uint32_t)k_step * K_STEP_BYTES_FP8;
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) {
                    uint32_t off = (uint32_t)(a_row[m] * K) + k_off + bq_lane_off;
                    x_reg[m] = buffer_load_fp8x16(srd_Aq, (int32_t)off);
                } else {
                    x_reg[m] = fp8x16_t{0, 0, 0, 0};
                }
            }
        };
        auto load_As = [&](uint32_t* dst, int k_step) {
            uint32_t kb_off = (uint32_t)k_step * KB_STEP + lane_kb_off;
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) {
                    uint32_t off = (uint32_t)(a_row[m] * (K / 32)) + kb_off;
                    dst[m] = buffer_load_ubyte(srd_As, (int32_t)off);
                } else {
                    dst[m] = 127u;
                }
            }
        };

        auto fma_block = [&](fp8x16_t* w_buf, uint32_t* ws_buf, uint32_t* xs_buf) {
            float xf[M_TILE][16];
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) unpack_fp8x16_to_f32x16(x_reg[m], xf[m]);
            }
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                float wf[16];
                unpack_fp8x16_to_f32x16(w_buf[c], wf);
                uint32_t ew = ws_buf[c];
                #pragma unroll
                for (int m = 0; m < M_TILE; ++m) {
                    if (valid[m]) {
                        float partial = 0.0f;
                        #pragma unroll
                        for (int e = 0; e < 16; ++e) partial += xf[m][e] * wf[e];
                        acc[c][m] += apply_scale(partial, xs_buf[m], ew);
                    }
                }
            }
        };

        if (K_STEPS > 0) {
            load_Bq(w_a, 0);
            load_Bs(ws_a, 0);

            int k_step = 0;
            for (; k_step + 1 < K_STEPS; k_step += 2) {
                load_Aq(k_step);
                load_As(xs_a, k_step);
                load_Bq(w_b, k_step + 1);
                load_Bs(ws_b, k_step + 1);
                s_wait_vmcnt<2 * WARP_N>();
                fma_block(w_a, ws_a, xs_a);

                int next2 = k_step + 2;
                load_Aq(k_step + 1);
                load_As(xs_b, k_step + 1);
                if (next2 < K_STEPS) {
                    load_Bq(w_a, next2);
                    load_Bs(ws_a, next2);
                }
                s_wait_vmcnt<2 * WARP_N>();
                fma_block(w_b, ws_b, xs_b);
            }
            if (k_step < K_STEPS) {
                load_Aq(k_step);
                load_As(xs_a, k_step);
                s_wait_vmcnt<0>();
                fma_block(w_a, ws_a, xs_a);
            }
        }

        if (K_TAIL_START < K) {
            int k_base = K_TAIL_START + lane * K_PER_LANE;
            if (k_base + K_PER_LANE <= K) {
                int kb = k_base / 32;
                #pragma unroll
                for (int c = 0; c < WARP_N; ++c) {
                    uint32_t ew_local = (uint32_t)Bs[bs_expert_base + (warp_n + c) * (K / 32) + kb];
                    const unsigned int* p = reinterpret_cast<const unsigned int*>(
                        Bq + (size_t)bq_expert_base + (warp_n + c) * (size_t)K + k_base);
                    int32x4_t r;
                    r[0] = (int32_t)p[0]; r[1] = (int32_t)p[1];
                    r[2] = (int32_t)p[2]; r[3] = (int32_t)p[3];
                    fp8x16_t wv = __builtin_bit_cast(fp8x16_t, r);
                    float wf[16]; unpack_fp8x16_to_f32x16(wv, wf);
                    #pragma unroll
                    for (int m = 0; m < M_TILE; ++m) {
                        if (valid[m]) {
                            uint32_t ex = (uint32_t)As[a_row[m] * (K / 32) + kb];
                            const unsigned int* xp = reinterpret_cast<const unsigned int*>(
                                Aq + a_row[m] * (size_t)K + k_base);
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
                if (valid[m]) {
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
                if (valid[m]) {
                    int out_row = offs_token_raw[m];
                    #pragma unroll
                    for (int c = 0; c < WARP_N; ++c) {
                        int n_idx = warp_n + c;
                        if (n_idx < N) {
                            float v = acc[c][m];
                            if (MUL_WEIGHT) v *= wgt[m];
                            Out[(size_t)out_row * N + n_idx] = (bf16)v;
                        }
                    }
                }
            }
        }
    }
}


// =====================================================================
// K=768 specialization: single warp-step (48 lanes * K_PER_LANE=16 = 768).
// Lanes 48-63 skip load+FMA; their `acc` stays 0 so the 64-way cross-lane
// reduction shfl_xor sums zeros from the upper 16 lanes and gives the
// correct result. No tail-K branch, no scalar-pointer reads.
// =====================================================================
template<int M_TILE, int BLOCK_N, int A_DIV, bool MUL_WEIGHT>
__global__ __launch_bounds__(256, 2)
void smallm_mxfp8_moe_gemm_kernel_k768(
    const unsigned char* __restrict__ Aq,
    const unsigned char* __restrict__ As,
    const unsigned char* __restrict__ Bq,
    const unsigned char* __restrict__ Bs,
    bf16*                __restrict__ Out,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const float*   __restrict__ topk_weights,
    int E, int N, int K, int num_valid_tokens, int M_act, int block_m)
{
    constexpr int WARPS_PER_WG    = 4;
    constexpr int WARP_N          = BLOCK_N / WARPS_PER_WG;
    constexpr int K_PER_LANE      = 16;
    constexpr int LANES_USED      = 48;       // 48 * 16 = 768
    constexpr int K_FIXED         = 768;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 6;
    const int lane   = tid & 63;
    const int pid_m  = blockIdx.x;
    const int pid_n  = blockIdx.y;
    const int wg_n   = pid_n * BLOCK_N;
    const int warp_n = wg_n + warp * WARP_N;

    if (warp_n >= N) return;

    int num_post = num_tokens_post_padded[0];
    int tile_base = pid_m * block_m;
    if (tile_base >= num_post) return;
    int off_e = expert_ids[pid_m];
    if (off_e < 0 || off_e >= E) return;

    const uint32_t bq_bytes = (uint32_t)((size_t)E * N * K_FIXED);
    const uint32_t aq_bytes = (uint32_t)((size_t)M_act * K_FIXED);
    const uint32_t bs_bytes = (uint32_t)((size_t)E * N * (K_FIXED / 32));
    const uint32_t as_bytes = (uint32_t)((size_t)M_act * (K_FIXED / 32));
    int32x4_t srd_Bq = make_srd(Bq, bq_bytes);
    int32x4_t srd_Aq = make_srd(Aq, aq_bytes);
    int32x4_t srd_Bs = make_srd(Bs, bs_bytes);
    int32x4_t srd_As = make_srd(As, as_bytes);

    const uint32_t bq_expert_base = (uint32_t)((size_t)off_e * N * K_FIXED);
    const uint32_t bq_warp_base   = bq_expert_base + (uint32_t)(warp_n * K_FIXED);
    const uint32_t bq_lane_off    = (uint32_t)(lane * K_PER_LANE);
    const uint32_t lane_kb_off    = (uint32_t)(lane >> 1);
    const uint32_t bs_expert_base = (uint32_t)((size_t)off_e * N * (K_FIXED / 32));

    const bool lane_active = (lane < LANES_USED);

    int num_row_groups = block_m / M_TILE;

    for (int rg = 0; rg < num_row_groups; ++rg) {
        int row_base = tile_base + rg * M_TILE;

        int32_t offs_token_raw[M_TILE];
        int32_t a_row[M_TILE];
        bool    valid[M_TILE];
        bool    any_valid = false;
        #pragma unroll
        for (int m = 0; m < M_TILE; ++m) {
            int32_t t = sorted_token_ids[row_base + m];
            offs_token_raw[m] = t;
            bool v = (t < num_valid_tokens);
            valid[m] = v;
            a_row[m] = v ? (t / A_DIV) : 0;
            any_valid = any_valid || v;
        }
        if (!any_valid) break;

        float wgt[M_TILE];
        if (MUL_WEIGHT) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                wgt[m] = valid[m] ? topk_weights[offs_token_raw[m]] : 0.0f;
            }
        }

        float acc[WARP_N][M_TILE];
        #pragma unroll
        for (int c = 0; c < WARP_N; ++c)
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) acc[c][m] = 0.0f;

        // Single warp-step: lanes 0..47 load B (per (warp_n+c) column), A (per
        // valid m row), and execute the FMA. Lanes 48..63 skip everything;
        // their `acc` stays 0 (initialised above) so the 64-way reduction is
        // correct.
        if (lane_active) {
            fp8x16_t w_a[WARP_N];
            fp8x16_t x_reg[M_TILE];
            uint32_t ws_a[WARP_N];
            uint32_t xs_a[M_TILE];

            // Loads -- single step (k_step=0).
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                uint32_t off = bq_warp_base + (uint32_t)(c * K_FIXED) + bq_lane_off;
                w_a[c] = buffer_load_fp8x16(srd_Bq, (int32_t)off);
            }
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                uint32_t off = bs_expert_base + (uint32_t)((warp_n + c) * (K_FIXED / 32)) + lane_kb_off;
                ws_a[c] = buffer_load_ubyte(srd_Bs, (int32_t)off);
            }
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) {
                    uint32_t off = (uint32_t)(a_row[m] * K_FIXED) + bq_lane_off;
                    x_reg[m] = buffer_load_fp8x16(srd_Aq, (int32_t)off);
                } else {
                    x_reg[m] = fp8x16_t{0, 0, 0, 0};
                }
            }
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) {
                    uint32_t off = (uint32_t)(a_row[m] * (K_FIXED / 32)) + lane_kb_off;
                    xs_a[m] = buffer_load_ubyte(srd_As, (int32_t)off);
                } else {
                    xs_a[m] = 127u;
                }
            }
            s_wait_vmcnt<0>();

            // FMA (single step).
            float xf[M_TILE][16];
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) unpack_fp8x16_to_f32x16(x_reg[m], xf[m]);
            }
            #pragma unroll
            for (int c = 0; c < WARP_N; ++c) {
                float wf[16];
                unpack_fp8x16_to_f32x16(w_a[c], wf);
                uint32_t ew = ws_a[c];
                #pragma unroll
                for (int m = 0; m < M_TILE; ++m) {
                    if (valid[m]) {
                        float partial = 0.0f;
                        #pragma unroll
                        for (int e = 0; e < 16; ++e) partial += xf[m][e] * wf[e];
                        acc[c][m] += apply_scale(partial, xs_a[m], ew);
                    }
                }
            }
        }
        // lanes 48-63: acc stays zero -- contributes 0 to reduction.

        // 64-way cross-lane reduction (unchanged from main kernel).
        #pragma unroll
        for (int c = 0; c < WARP_N; ++c) {
            #pragma unroll
            for (int m = 0; m < M_TILE; ++m) {
                if (valid[m]) {
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
                if (valid[m]) {
                    int out_row = offs_token_raw[m];
                    #pragma unroll
                    for (int c = 0; c < WARP_N; ++c) {
                        int n_idx = warp_n + c;
                        if (n_idx < N) {
                            float v = acc[c][m];
                            if (MUL_WEIGHT) v *= wgt[m];
                            Out[(size_t)out_row * N + n_idx] = (bf16)v;
                        }
                    }
                }
            }
        }
    }
}

// =====================================================================
// Host launcher
// =====================================================================
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#include "aiter_hip_common.h"  // get_gpu_arch()

#define CHECK_U8(t)  TORCH_CHECK(t.scalar_type() == at::kByte, #t " must be uint8 (fp8 payload bytes or e8m0 scales)")
#define CHECK_CUDA(t) TORCH_CHECK(t.is_cuda(), #t " must be CUDA")
#define CHECK_I32(t) TORCH_CHECK(t.scalar_type() == at::kInt, #t " must be int32")

template<int M_TILE_INNER, int BLOCK_N, int A_DIV, bool MUL_WEIGHT>
static void launch_one(
    const unsigned char* aqp, const unsigned char* asp,
    const unsigned char* bqp, const unsigned char* bsp,
    bf16* op, const int32_t* sip, const int32_t* eip, const int32_t* npp,
    const float* twp,
    int E, int N, int K, int num_valid_tokens, int M_act, int block_m,
    int num_m_tiles, hipStream_t stream)
{
    dim3 grid((unsigned)num_m_tiles, (unsigned)((N + BLOCK_N - 1) / BLOCK_N));
    dim3 block(256);
    if (K == 768) {
        // K=768 specialization (gemm2 down-proj): single warp-step, lanes
        // 48-63 idle.
        smallm_mxfp8_moe_gemm_kernel_k768<M_TILE_INNER, BLOCK_N, A_DIV, MUL_WEIGHT>
            <<<grid, block, 0, stream>>>(
                aqp, asp, bqp, bsp, op, sip, eip, npp, twp,
                E, N, K, num_valid_tokens, M_act, block_m);
    } else {
        smallm_mxfp8_moe_gemm_kernel<M_TILE_INNER, BLOCK_N, A_DIV, MUL_WEIGHT>
            <<<grid, block, 0, stream>>>(
                aqp, asp, bqp, bsp, op, sip, eip, npp, twp,
                E, N, K, num_valid_tokens, M_act, block_m);
    }
}

#define LAUNCH_BN(M_TI, BN, ADV, MUL)                                                                            \
    launch_one<M_TI, BN, ADV, MUL>(aqp, asp, bqp, bsp, op, sip, eip, npp, twp,                                    \
                                   E, N, K, num_valid_tokens, M_act, block_m, num_m_tiles, stream)

// BLOCK_N is tuned per shape (see _MOE_BLOCK_N on the Python side): the narrow
// dense-GEMV tile (8) wins on deep-K gemm1; the wider tile (16) wins on
// shallow-K / wide-N gemm2 in the decode regime.
#define DISP_AD(M_TI, ADV)                                                                                       \
    do {                                                                                                          \
        if (has_weight) {                                                                                         \
            if (block_n == 16) LAUNCH_BN(M_TI, 16, ADV, true);                                                    \
            else               LAUNCH_BN(M_TI, 8,  ADV, true);                                                    \
        } else {                                                                                                  \
            if (block_n == 16) LAUNCH_BN(M_TI, 16, ADV, false);                                                   \
            else               LAUNCH_BN(M_TI, 8,  ADV, false);                                                   \
        }                                                                                                         \
    } while (0)

#define DISP_KERNEL(M_TI)                                                                                         \
    do {                                                                                                          \
        if (a_div == 1)        DISP_AD(M_TI, 1);                                                                  \
        else if (a_div == 4)   DISP_AD(M_TI, 4);                                                                  \
        else if (a_div == 8)   DISP_AD(M_TI, 8);                                                                  \
        else TORCH_CHECK(false, "Unsupported a_div=", a_div);                                                     \
    } while (0)

at::Tensor smallm_mxfp8_moe_grouped_gemm(
    at::Tensor a_q,
    at::Tensor a_scale,
    at::Tensor w,
    at::Tensor w_scale,
    at::Tensor sorted_token_ids,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    at::Tensor out,
    int64_t E_i, int64_t N_i, int64_t K_i,
    int64_t num_valid_tokens, int64_t M_act,
    int64_t a_div, int64_t block_m,
    c10::optional<at::Tensor> mul_weight_by,
    int64_t block_n)
{
    TORCH_CHECK(block_n == 8 || block_n == 16,
                "block_n must be 8 or 16; got ", block_n);
    TORCH_CHECK(get_gpu_arch() == "gfx950",
                "smallm_mxfp8_moe_grouped_gemm requires a CDNA4 (gfx950) device; got ",
                get_gpu_arch());
    CHECK_CUDA(a_q); CHECK_CUDA(a_scale); CHECK_CUDA(w); CHECK_CUDA(w_scale);
    CHECK_CUDA(sorted_token_ids); CHECK_CUDA(expert_ids); CHECK_CUDA(num_tokens_post_padded);
    CHECK_CUDA(out);
    CHECK_U8(a_q); CHECK_U8(a_scale); CHECK_U8(w); CHECK_U8(w_scale);
    CHECK_I32(sorted_token_ids); CHECK_I32(expert_ids); CHECK_I32(num_tokens_post_padded);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bf16");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK((size_t)out.numel() >= (size_t)num_valid_tokens * N_i,
                "out has ", out.numel(), " elements but needs num_valid_tokens*N = ",
                (size_t)num_valid_tokens * N_i);

    const int E = (int)E_i;
    const int N = (int)N_i;
    const int K = (int)K_i;
    // Weights [E,N,K] and activations [M_act,K] are each addressed via one
    // raw-buffer SRD whose voffset is a SIGNED int32 (buffer_load voffset_bytes),
    // so any byte offset >= 2^31 wraps negative -> OOB fault. Reject >2GB tensors
    // here (the host runs before launch; a GPU fault is uncatchable). With M3's
    // 256 experts this caps weights at ~EP2 (E=128, 1.2GB); no-EP (E=256, 2.4GB)
    // is correctly rejected and falls back to Triton. (A per-expert base-pointer
    // advance would lift the E*N*K term -- tracked as a follow-up.)
    TORCH_CHECK((size_t)E * N * K <= 0x7FFFFFFFull &&
                    (size_t)M_act * K <= 0x7FFFFFFFull,
                "smallm_mxfp8_moe_grouped_gemm: a tensor exceeds the 2GB "
                "signed-32-bit raw-buffer offset limit");
    TORCH_CHECK(block_m % 4 == 0, "block_m must be multiple of 4");

    // sorted_token_ids is the block_m-padded routing layout; if its length is
    // not a whole number of m-tiles the integer division below would silently
    // drop the tail tile. expert_ids carries one expert per m-tile, so it must
    // be at least as long as the tile count we are about to index.
    TORCH_CHECK(sorted_token_ids.size(0) % (int64_t)block_m == 0,
                "sorted_token_ids length (", sorted_token_ids.size(0),
                ") must be a multiple of block_m (", block_m, ")");
    int num_m_tiles = (int)sorted_token_ids.size(0) / (int)block_m;
    if (num_m_tiles <= 0) return out;
    TORCH_CHECK(expert_ids.size(0) >= num_m_tiles,
                "expert_ids has ", expert_ids.size(0), " entries but ",
                num_m_tiles, " m-tiles are indexed");

    const unsigned char* aqp = reinterpret_cast<const unsigned char*>(a_q.data_ptr());
    const unsigned char* asp = reinterpret_cast<const unsigned char*>(a_scale.data_ptr());
    const unsigned char* bqp = reinterpret_cast<const unsigned char*>(w.data_ptr());
    const unsigned char* bsp = reinterpret_cast<const unsigned char*>(w_scale.data_ptr());
    bf16* op = reinterpret_cast<bf16*>(out.data_ptr());
    const int32_t* sip = sorted_token_ids.data_ptr<int32_t>();
    const int32_t* eip = expert_ids.data_ptr<int32_t>();
    const int32_t* npp = num_tokens_post_padded.data_ptr<int32_t>();
    bool has_weight = mul_weight_by.has_value() && mul_weight_by->defined();
    if (has_weight) {
        // The kernel reads mul_weight_by[token] as raw float32 on-device for
        // every valid token; validate before taking the pointer so a CPU /
        // wrong-dtype / too-short tensor fails loudly instead of reading garbage.
        const at::Tensor& mw = mul_weight_by.value();
        TORCH_CHECK(mw.is_cuda(), "mul_weight_by must be a CUDA tensor");
        TORCH_CHECK(mw.scalar_type() == at::kFloat, "mul_weight_by must be float32");
        TORCH_CHECK(mw.is_contiguous(), "mul_weight_by must be contiguous");
        TORCH_CHECK(mw.numel() >= num_valid_tokens,
                    "mul_weight_by has ", mw.numel(), " elements but ",
                    num_valid_tokens, " valid tokens are indexed");
    }
    const float* twp = has_weight ? mul_weight_by->data_ptr<float>() : nullptr;

    auto stream = c10::hip::getCurrentHIPStream().stream();

    DISP_KERNEL(4);  // M_TILE_INNER=4; block_m % 4 == 0 enforced above
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallm_mxfp8_moe_grouped_gemm", &smallm_mxfp8_moe_grouped_gemm,
          "small-M MXFP8 MoE grouped GEMM (E8M0 1x32 scales, decode regime)",
          pybind11::arg("a_q"), pybind11::arg("a_scale"),
          pybind11::arg("w"),   pybind11::arg("w_scale"),
          pybind11::arg("sorted_token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("num_tokens_post_padded"),
          pybind11::arg("out"),
          pybind11::arg("E"), pybind11::arg("N"), pybind11::arg("K"),
          pybind11::arg("num_valid_tokens"), pybind11::arg("M_act"),
          pybind11::arg("a_div"), pybind11::arg("block_m"),
          pybind11::arg("mul_weight_by") = c10::optional<at::Tensor>(),
          pybind11::arg("block_n") = 8);
}
