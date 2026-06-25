// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250-ONLY split-precision (NoPE fp8 / RoPE bf16) sparse paged prefill
// attention for DeepSeek-V4 DSA. Parallel, self-contained implementation,
// derived from the gfx950 fp8 kernel in pa_sparse_prefill_opus.h but rewritten
// for wave32 WMMA (gfx1250). First-correctness path: dequantize NoPE fp8 -> bf16
// and run the NoPE QK^T as a bf16 WMMA (sidestepping scaled-WMMA). RoPE QK and
// PV are bf16. Native fp8 scaled-WMMA for NoPE QK is a perf TODO.

#pragma once
#include "aiter_tensor.h"

// Public API: gfx1250 split-precision prefill attention.
void pa_sparse_prefill_fp8_opus_gfx1250_fwd(aiter_tensor_t& q_nope,
                                            aiter_tensor_t& q_rope,
                                            aiter_tensor_t& unified_kv_nope,
                                            aiter_tensor_t& unified_kv_rope,
                                            aiter_tensor_t& kv_indices_prefix,
                                            aiter_tensor_t& kv_indptr_prefix,
                                            aiter_tensor_t& kv_nope,
                                            aiter_tensor_t& kv_rope,
                                            aiter_tensor_t& kv_indices_extend,
                                            aiter_tensor_t& kv_indptr_extend,
                                            aiter_tensor_t& attn_sink,
                                            aiter_tensor_t& out,
                                            float softmax_scale);

#ifdef PA_SPARSE_PREFILL_FP8_OPUS_GFX1250_IMPL
// ============================================================================
// Implementation section - only compiled in the .cu translation unit
// ============================================================================
#include <opus/opus.hpp>
#include <bit>

using bf16_t = __bf16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

// Kernel arguments for the split-precision (NoPE fp8 / RoPE bf16) DSA prefill.
// (Fields copied from pa_fp8_kargs in pa_sparse_prefill_opus.h.)
struct pa_fp8_gfx1250_kargs
{
    const void* __restrict__ q_nope_ptr;          // [N, H, D_NOPE_PADDED] fp8
    const void* __restrict__ q_rope_ptr;          // [N, H, D_ROPE]        bf16
    const void* __restrict__ unified_kv_nope_ptr; // [total_pages, D_NOPE_PADDED] fp8
    const void* __restrict__ unified_kv_rope_ptr; // [total_pages, D_ROPE]        bf16
    const void* __restrict__ kv_nope_ptr;         // [total_tokens, D_NOPE_PADDED] fp8
    const void* __restrict__ kv_rope_ptr;         // [total_tokens, D_ROPE]        bf16
    const void* __restrict__ attn_sink_ptr;       // [H]
    void* __restrict__ out_ptr;                   // [N, H, D_HEAD] bf16
    const int* __restrict__ kv_indptr_prefix;     // [N+1]
    const int* __restrict__ kv_indices_prefix;    // [nnz_prefix]
    const int* __restrict__ kv_indptr_extend;     // [N+1]
    const int* __restrict__ kv_indices_extend;    // [nnz_extend]
    int N;
    int H;
    int total_pages;
    int total_tokens;
    int stride_q_nope_n;
    int stride_q_nope_h;
    int stride_q_rope_n;
    int stride_q_rope_h;
    int stride_o_n;
    int stride_o_h;
    int stride_kv_nope_page;
    int stride_kv_rope_page;
    float softmax_scale;
};

// Compile-time tile config for the gfx1250 wave32 fp8 variant.
// Single wave per query tile (T_M=T_N=1) -> intra-wave online softmax, no
// cross-warp reduction needed (simplest correct path). KV_TILE=16.
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 1,
          typename D_NOPE_  = fp8_t,
          typename D_ROPE_  = bf16_t,
          typename D_OUT_   = bf16_t>
struct pa_fp8_gfx1250_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 32; // gfx1250 wave32
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Packed DSA hdim split
    static constexpr int D_NOPE_SIZE        = 448; // NoPE fp8 elements
    static constexpr int D_NOPE_PADDED_SIZE = 512; // NoPE padded row (incl scales)
    static constexpr int D_ROPE_SIZE        = 64;  // RoPE bf16 elements
    static constexpr int D_HEAD_SIZE        = D_NOPE_SIZE + D_ROPE_SIZE; // 512

    static constexpr int NBLK = D_NOPE_SIZE / 32; // 14 E8M0 block scales

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    // WMMA base tile (wave32): bf16 16x16x32.
    static constexpr int W_M   = 16;
    static constexpr int W_N   = 16;
    static constexpr int W_K   = 32;

    // GEMM0 S = Q @ K^T  (Q is [Qtile, Dhead], K is [KVtile, Dhead])
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;            // 1
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;           // 2
    static constexpr int GEMM0_E_K = D_HEAD_SIZE / W_K;            // 16 (full dhead bf16)

    // GEMM1 O = P @ V  (P is [Qtile, KVtile], V is [KVtile, Dhead])
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;            // 1
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / W_N;            // 32
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;           // 1 (KV_TILE=32=W_K)

    static constexpr int VEC_O = 4;

    // smem KV row stride (bf16 staging of full head dim).
    static constexpr int SMEM_KV_PAD = 8;
    static constexpr int SMEM_KV_ROW = D_HEAD_SIZE + SMEM_KV_PAD;
    static constexpr int SMEM_Q_ROW  = D_HEAD_SIZE + SMEM_KV_PAD;
    // P staging: [Q_TILE rows, KV_TILE cols] bf16.
    static constexpr int SMEM_P_ROW  = KV_TILE_SIZE + SMEM_KV_PAD;

    static constexpr size_t smem_kv_bytes() { return (size_t)KV_TILE_SIZE * SMEM_KV_ROW * sizeof(D_ROPE); }
    static constexpr size_t smem_q_bytes()  { return (size_t)Q_TILE_SIZE  * SMEM_Q_ROW  * sizeof(D_ROPE); }
    static constexpr size_t smem_p_bytes()  { return (size_t)Q_TILE_SIZE  * SMEM_P_ROW  * sizeof(D_ROPE); }

    static constexpr size_t smem_size_bytes()
    {
        return smem_kv_bytes() + smem_q_bytes() + smem_p_bytes();
    }
};

__host__ __device__ inline int ceil_div_g(int a, int b) { return (a + b - 1) / b; }

template <class Traits>
__global__ void pa_prefill_fp8_gfx1250_kernel(pa_fp8_gfx1250_kargs kargs);

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx1250__)
// Host pass / other arches: empty stub so symbols resolve.
template <class Traits>
__global__ void pa_prefill_fp8_gfx1250_kernel(pa_fp8_gfx1250_kargs)
{
}
#else
// =============================================================================
// Device-side kernel (gfx1250 wave32 WMMA).
// =============================================================================
using opus::operator""_I;

namespace pa_fp8_gfx1250 {

using opus::number;
using opus::tuple;
using opus::index_t;

// Dequantize a packed fp8 dword (4 fp8 vals) -> 4 floats with an E8M0 scale.
__device__ inline opus::vector_t<float, 4> deq_fp8x4(opus::u32_t w, float scale) {
    int bitwise = static_cast<int>(w);
    auto x = __builtin_amdgcn_cvt_pk_f32_fp8(bitwise, 0); // elems 0,1
    auto y = __builtin_amdgcn_cvt_pk_f32_fp8(bitwise, 1); // elems 2,3
    return opus::vector_t<float, 4>{x[0] * scale, x[1] * scale, y[0] * scale, y[1] * scale};
}

// Stage one packed row (NoPE fp8 + E8M0 scales + RoPE bf16) into a bf16 LDS row
// [d in 0..D_HEAD_SIZE). Lane-strided over the 112 NoPE dwords + 64 RoPE elems.
template <class T>
__device__ inline void stage_row_bf16(typename T::D_ROPE* dst,
                                      const typename T::D_NOPE* nope_row,
                                      const typename T::D_ROPE* rope_row,
                                      int lane_id) {
    using D_ROPE = typename T::D_ROPE;
    const opus::u32_t* nope_w   = reinterpret_cast<const opus::u32_t*>(nope_row);
    const unsigned char* scl    = reinterpret_cast<const unsigned char*>(nope_row) + T::D_NOPE_SIZE;
    for (int dw = lane_id; dw < T::D_NOPE_SIZE / 4; dw += T::WARP_SIZE) {
        const int blk = (dw * 4) / 32;
        const unsigned e8m0 = scl[blk];
        const float scale = std::bit_cast<float>((opus::u32_t)e8m0 << 23);
        auto f4 = deq_fp8x4(nope_w[dw], scale);
        const int d0 = dw * 4;
        dst[d0 + 0] = static_cast<D_ROPE>(f4[0]);
        dst[d0 + 1] = static_cast<D_ROPE>(f4[1]);
        dst[d0 + 2] = static_cast<D_ROPE>(f4[2]);
        dst[d0 + 3] = static_cast<D_ROPE>(f4[3]);
    }
    for (int d = lane_id; d < T::D_ROPE_SIZE; d += T::WARP_SIZE) {
        dst[T::D_NOPE_SIZE + d] = rope_row[d];
    }
}

// Accumulate one KV segment (prefix or extend) into the LDS O accumulator
// (o_lds, f32 [Q_TILE, D_HEAD]) and per-row online-softmax state (m_lds, l_lds).
//   s_q  : [Q_TILE, SMEM_Q_ROW]   bf16  (Q, staged once, read-only here)
//   s_kv : [KV_TILE, SMEM_KV_ROW] bf16  (K/V tile; K and V are the same data)
//   s_p  : [Q_TILE, SMEM_P_ROW]   f32   (scores/probs scratch) then bf16 P region
//   o_lds: [Q_TILE, D_HEAD]       f32   (running output accumulator)
//   m_lds,l_lds: [Q_TILE]         f32
template <class Traits, class MMA0, class MMA1, class VQ>
__device__ void accum_segment(
        const pa_fp8_gfx1250_kargs& kargs,
        const void* kv_nope_ptr, const void* kv_rope_ptr,
        const int* kv_indices, int page_idx_begin, int valid_kv_len, int num_kv_tiles,
        MMA0& mma0, MMA1& mma1,
        typename Traits::D_ROPE* s_q, typename Traits::D_ROPE* s_kv,
        float* s_p, typename Traits::D_ROPE* s_pb,
        float* o_lds, float* m_lds, float* l_lds,
        const VQ& v_q, float temperature_scale, int lane_id) {
    using namespace opus;
    using T = Traits;
    using D_ROPE = typename T::D_ROPE;
    using D_NOPE = typename T::D_NOPE;

    auto s_kv_mem = make_smem(s_kv);
    auto s_p_mem  = make_smem(s_p);
    auto s_pb_mem = make_smem(s_pb);

    typename MMA0::vtype_c v_s;

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        const int tile_base = tile_idx * T::KV_TILE_SIZE;

        // ---- Stage K/V tile into LDS (bf16), zero OOB tokens ----
        for (int idx = lane_id; idx < T::KV_TILE_SIZE * T::SMEM_KV_ROW; idx += T::WARP_SIZE)
            s_kv[idx] = static_cast<D_ROPE>(0);
        __builtin_amdgcn_s_barrier();
        for (int tok = 0; tok < T::KV_TILE_SIZE; ++tok) {
            const int kv_pos = tile_base + tok;
            if (kv_pos >= valid_kv_len) break;
            const int page = kv_indices[page_idx_begin + kv_pos];
            const D_NOPE* nope_row = reinterpret_cast<const D_NOPE*>(kv_nope_ptr)
                                     + (int64_t)page * kargs.stride_kv_nope_page;
            const D_ROPE* rope_row = reinterpret_cast<const D_ROPE*>(kv_rope_ptr)
                                     + (int64_t)page * kargs.stride_kv_rope_page;
            stage_row_bf16<T>(s_kv + tok * T::SMEM_KV_ROW, nope_row, rope_row, lane_id);
        }
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        // ---- GEMM0: S = Q @ K^T ----
        // A = Q [M=16, K=512] (s_q row stride SMEM_Q_ROW), B = K [N=32, K=512].
        auto lb = mma0.template layout_b<0>(
            tuple{number<T::SMEM_KV_ROW>{}, 1_I},
            tuple{0, lane_id % T::W_N, 0, lane_id / T::W_N});
        auto v_k = load(s_kv_mem, lb);
        s_wait_dscnt(0_I);
        clear(v_s);
        v_s = mma0(v_q, v_k, v_s);

        // ---- Scale + store S to s_p (C layout -> [m, n]) ----
        constexpr index_t s_len = vector_traits<decltype(v_s)>::size();
        static_for<s_len>([&](auto i){ v_s[i.value] *= temperature_scale; });
        auto lc = mma0.template layout_c<0>(
            tuple{number<T::SMEM_P_ROW>{}, 1_I},
            tuple{0, lane_id % T::W_N, 0, lane_id / T::W_N});
        store(s_p_mem, v_s, lc);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        // ---- Per-row online softmax (lane = query row, lanes 0..Q_TILE-1) ----
        if (lane_id < T::Q_TILE_SIZE) {
            const int qr = lane_id;
            float* prow = s_p + qr * T::SMEM_P_ROW;
            float tmax = -opus::numeric_limits<float>::infinity();
            for (int n = 0; n < T::KV_TILE_SIZE; ++n) {
                const int tok = tile_base + n;
                float v = (tok < valid_kv_len) ? prow[n] : -opus::numeric_limits<float>::infinity();
                tmax = max(tmax, v);
            }
            float new_m   = max(m_lds[qr], tmax);
            float rescale = __builtin_amdgcn_exp2f(m_lds[qr] - new_m);
            float psum    = 0.0f;
            for (int n = 0; n < T::KV_TILE_SIZE; ++n) {
                const int tok = tile_base + n;
                float v = (tok < valid_kv_len) ? prow[n] : -opus::numeric_limits<float>::infinity();
                float p = __builtin_amdgcn_exp2f(v - new_m);
                prow[n] = p;
                psum += p;
            }
            m_lds[qr] = new_m;
            l_lds[qr] = l_lds[qr] * rescale + psum;
            // rescale the running O accumulator row in LDS.
            float* orow = o_lds + qr * T::D_HEAD_SIZE;
            for (int d = 0; d < T::D_HEAD_SIZE; ++d) orow[d] *= rescale;
        }
        __builtin_amdgcn_s_barrier();

        // ---- Repack P (f32) -> bf16 contiguous for A fragment ----
        for (int idx = lane_id; idx < T::Q_TILE_SIZE * T::SMEM_P_ROW; idx += T::WARP_SIZE)
            s_pb[idx] = static_cast<D_ROPE>(0);
        __builtin_amdgcn_s_barrier();
        for (int idx = lane_id; idx < T::Q_TILE_SIZE * T::KV_TILE_SIZE; idx += T::WARP_SIZE) {
            int m = idx / T::KV_TILE_SIZE, n = idx % T::KV_TILE_SIZE;
            s_pb[m * T::SMEM_P_ROW + n] = static_cast<D_ROPE>(s_p[m * T::SMEM_P_ROW + n]);
        }
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();

        // ---- GEMM1: dO = P @ V (V = same s_kv, transposed via strides) ----
        auto la = mma1.template layout_a<0>(
            tuple{number<T::SMEM_P_ROW>{}, 1_I},
            tuple{0, lane_id % T::W_M, 0, lane_id / T::W_M});
        auto v_p = load(s_pb_mem, la);
        // V[token, d] -> B fragment [N=d, K=token]: N-stride=1, K-stride=SMEM_KV_ROW.
        auto lvb = mma1.template layout_b<0>(
            tuple{1_I, number<T::SMEM_KV_ROW>{}},
            tuple{0, lane_id % T::W_N, 0, lane_id / T::W_N});
        auto v_v = load(s_kv_mem, lvb);
        s_wait_dscnt(0_I);
        typename MMA1::vtype_c v_do;
        clear(v_do);
        v_do = mma1(v_p, v_v, v_do);

        // ---- Accumulate dO into o_lds (C layout [m, d]) ----
        // Store dO to s_p (reused f32 scratch) in C layout, then add to o_lds.
        __builtin_amdgcn_s_barrier();
        auto lc_do = mma1.template layout_c<0>(
            tuple{number<T::D_HEAD_SIZE>{}, 1_I},
            tuple{0, lane_id % T::W_N, 0, lane_id / T::W_N});
        // o_lds is [Q_TILE, D_HEAD] contiguous (stride D_HEAD). Use a fresh smem
        // view to store dO additively: store to a temp region then add. We add via
        // a per-lane store to a scratch o2, then accumulate. Simplest: store dO to
        // s_p f32 (large enough: Q_TILE*D_HEAD floats <= s_p capacity), add to o_lds.
        auto o2_mem = make_smem(s_p);
        store(o2_mem, v_do, lc_do);
        s_wait_dscnt(0_I);
        __builtin_amdgcn_s_barrier();
        for (int idx = lane_id; idx < T::Q_TILE_SIZE * T::D_HEAD_SIZE; idx += T::WARP_SIZE)
            o_lds[idx] += s_p[idx];
        __builtin_amdgcn_s_barrier();
    }
}

} // namespace pa_fp8_gfx1250

template <class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2)
void pa_prefill_fp8_gfx1250_kernel(pa_fp8_gfx1250_kargs kargs) {
    using namespace opus;
    using namespace pa_fp8_gfx1250;
    using T = Traits;
    using D_NOPE = typename T::D_NOPE;
    using D_ROPE = typename T::D_ROPE;
    using D_OUT  = typename T::D_OUT;

    const int q_token_idx = block_id_x();
    const int h_block_idx = block_id_y();
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int h_block_start = h_block_idx * T::T_M * T::Q_TILE_SIZE;

    constexpr float LOG2_E = 1.44269504089f;
    const float temperature_scale = kargs.softmax_scale * LOG2_E;

    // ---- Shared memory ----
    __shared__ D_ROPE s_q [T::Q_TILE_SIZE  * T::SMEM_Q_ROW];    // Q bf16
    __shared__ D_ROPE s_kv[T::KV_TILE_SIZE * T::SMEM_KV_ROW];   // K/V bf16
    __shared__ D_ROPE s_pb[T::Q_TILE_SIZE  * T::SMEM_P_ROW];    // P bf16
    __shared__ float  s_p [T::Q_TILE_SIZE  * T::D_HEAD_SIZE];   // scores/dO f32 scratch (32KB)
    __shared__ float  o_lds[T::Q_TILE_SIZE * T::D_HEAD_SIZE];   // O accumulator f32 (32KB)
    __shared__ float  m_lds[T::Q_TILE_SIZE];
    __shared__ float  l_lds[T::Q_TILE_SIZE];

    // ---- Stage Q tile (dequant NoPE + RoPE) into LDS bf16 [Q_TILE, dhead] ----
    for (int h = 0; h < T::Q_TILE_SIZE; ++h) {
        const int head = h_block_start + h;
        if (head >= kargs.H) {
            for (int d = lane_id; d < T::SMEM_Q_ROW; d += T::WARP_SIZE)
                s_q[h * T::SMEM_Q_ROW + d] = static_cast<D_ROPE>(0);
            continue;
        }
        const D_NOPE* qn = reinterpret_cast<const D_NOPE*>(kargs.q_nope_ptr)
                           + (int64_t)q_token_idx * kargs.stride_q_nope_n
                           + (int64_t)head * kargs.stride_q_nope_h;
        const D_ROPE* qr = reinterpret_cast<const D_ROPE*>(kargs.q_rope_ptr)
                           + (int64_t)q_token_idx * kargs.stride_q_rope_n
                           + (int64_t)head * kargs.stride_q_rope_h;
        stage_row_bf16<T>(s_q + h * T::SMEM_Q_ROW, qn, qr, lane_id);
    }
    s_wait_dscnt(0_I);
    __builtin_amdgcn_s_barrier();

    // ---- MMA operators ----
    auto mma0 = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        wmma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<bf16_t, bf16_t, float>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        wmma_adaptor_swap_ab{});

    auto s_q_mem = make_smem(s_q);
    auto la_q = mma0.template layout_a<0>(
        tuple{number<T::SMEM_Q_ROW>{}, 1_I},
        tuple{0, lane_id % T::W_M, 0, lane_id / T::W_M});
    auto v_q = load(s_q_mem, la_q);
    s_wait_dscnt(0_I);

    // ---- Init online state + O accumulator in LDS ----
    for (int i = lane_id; i < T::Q_TILE_SIZE; i += T::WARP_SIZE) {
        m_lds[i] = -opus::numeric_limits<float>::infinity();
        l_lds[i] = 0.0f;
    }
    for (int i = lane_id; i < T::Q_TILE_SIZE * T::D_HEAD_SIZE; i += T::WARP_SIZE)
        o_lds[i] = 0.0f;
    __builtin_amdgcn_s_barrier();

    // ---- Prefix segment ----
    {
        const int pb = kargs.kv_indptr_prefix[q_token_idx];
        const int pe = kargs.kv_indptr_prefix[q_token_idx + 1];
        const int valid = pe - pb;
        const int ntiles = ceil_div_g(valid, T::KV_TILE_SIZE);
        accum_segment<T>(kargs, kargs.unified_kv_nope_ptr, kargs.unified_kv_rope_ptr,
                         kargs.kv_indices_prefix, pb, valid, ntiles,
                         mma0, mma1, s_q, s_kv, s_p, s_pb, o_lds, m_lds, l_lds,
                         v_q, temperature_scale, lane_id);
    }
    // ---- Extend segment ----
    {
        const int pb = kargs.kv_indptr_extend[q_token_idx];
        const int pe = kargs.kv_indptr_extend[q_token_idx + 1];
        const int valid = pe - pb;
        const int ntiles = ceil_div_g(valid, T::KV_TILE_SIZE);
        accum_segment<T>(kargs, kargs.kv_nope_ptr, kargs.kv_rope_ptr,
                         kargs.kv_indices_extend, pb, valid, ntiles,
                         mma0, mma1, s_q, s_kv, s_p, s_pb, o_lds, m_lds, l_lds,
                         v_q, temperature_scale, lane_id);
    }

    // ---- Sink finalization + normalize + store ----
    const int64_t o_base = (int64_t)q_token_idx * kargs.stride_o_n
                           + (int64_t)h_block_start * kargs.stride_o_h;
    // Per query row finalize + store to gmem.
    for (int m = lane_id; m < T::Q_TILE_SIZE; m += T::WARP_SIZE) {
        const int head = h_block_start + m;
        if (head >= kargs.H) continue;
        float sink_log2 = reinterpret_cast<const float*>(kargs.attn_sink_ptr)[head] * LOG2_E;
        float m_final = max(m_lds[m], sink_log2);
        float alpha   = __builtin_amdgcn_exp2f(m_lds[m] - m_final);
        float l_final = l_lds[m] * alpha + __builtin_amdgcn_exp2f(sink_log2 - m_final);
        float o_scale = (l_final > 0.0f) ? (alpha / l_final) : 0.0f;
        for (int d = 0; d < T::D_HEAD_SIZE; ++d) {
            float ov = o_lds[m * T::D_HEAD_SIZE + d] * o_scale;
            reinterpret_cast<D_OUT*>(kargs.out_ptr)[o_base + (int64_t)m * kargs.stride_o_h + d]
                = static_cast<D_OUT>(ov);
        }
    }
}
#endif

#endif // PA_SPARSE_PREFILL_FP8_OPUS_GFX1250_IMPL
