// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 wave-K-cooperative pipeline family.
//
// Deterministic WKC kids 10300-10303: one WG owns a small (M, N) output tile;
// all waves within the WG split across K, then LDS-reduce partials and store Y.
// Accum WKC kids 10310-10314: split K across 8 WGs and atomic-add bf16x2 Y.
//
// Geometry: B=(B_M, B_N, B_K), T=(1, 1, BLOCK_SIZE/wave_size),
// W=(16, 16, 16), LDS_DEPTH=1.
//   E_M = B_M / W_M, E_N = B_N / W_N, E_K = B_K / W_K
//   Each wave runs K_LOOPS = K / (B_K * T_K) K-tiles and emits E_M*E_N
//   fp32 acc groups (4 fp32 per lane per group).
//
// Determinism: each wave's accumulator is a fixed K-loop order; LDS reduce
// of T_K partials uses fixed wave-id order.
#pragma once

#include "opus_gemm_traits_a16w16.cuh"

#ifdef __HIP_DEVICE_COMPILE__

namespace opus_wkc_gfx942 {

using opus::operator""_I;

// ---- Layout ----

template<typename T>
OPUS_D auto make_layout_wkc_ga(int lane_id, int stride_a)
{
    constexpr int threads_k = T::B_K / T::VEC_A;

    return opus::make_tuple(
        lane_id / threads_k,
        (lane_id % threads_k) * T::VEC_A,
        stride_a);
}

template<typename T>
OPUS_D auto make_layout_wkc_gb(int lane_id, int stride_b)
{
    constexpr int threads_k = T::B_K / T::VEC_B;

    return opus::make_tuple(
        lane_id / threads_k,
        (lane_id % threads_k) * T::VEC_B,
        stride_b);
}

template<typename T>
OPUS_D auto make_layout_wkc_sa(int lane_id)
{
    constexpr int threads_k = T::B_K / T::VEC_A;

    return opus::make_tuple(
        lane_id / threads_k,
        (lane_id % threads_k) * T::VEC_A);
}

template<typename T>
OPUS_D auto make_layout_wkc_sb(int lane_id)
{
    constexpr int threads_k = T::B_K / T::VEC_B;

    return opus::make_tuple(
        lane_id / threads_k,
        (lane_id % threads_k) * T::VEC_B);
}

template<typename T>
OPUS_D auto make_layout_wkc_ua(int lane_id)
{
    return opus::make_tuple(
        lane_id % T::W_M,
        (lane_id / T::W_M) * 4);
}

template<typename T>
OPUS_D auto make_layout_wkc_ub(int lane_id)
{
    return opus::make_tuple(
        lane_id % T::W_N,
        (lane_id / T::W_N) * 4);
}

// ---- Prologue / Memory ----

// Lane layout for v_mfma_f32_16x16x16_bf16:
//   A operand: lane (m, k4) where m = lane%16, k4 = lane/16
//              short4 per lane = A[m, k4*4 + 0..3]
//   B operand: same shape -- short4 per lane = B[n, k4*4 + 0..3]
//   C accumulator: 4 fp32 per lane, lane (m_sub, n) where m_sub = lane/16, n = lane%16
//                  acc[i] = C[m_sub*4 + i, n]

using short4_ab = opus::vector_t<__bf16, 4>;
using float4_acc = float __attribute__((ext_vector_type(4)));
using bf16x2_accum = __bf16 __attribute__((ext_vector_type(2)));

template<typename T, typename G, typename U>
OPUS_D auto load_a_wave_tile(G& g_a, const U& u_ga,
                             int wave_k_tile_offset)
{
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_row = opus::get_warp_size() / threads_k;
    constexpr int loads = T::B_M / threads_row;

    opus::vector_t<typename T::D_A, T::VEC_A * loads> out;
    #pragma unroll
    for (int i = 0; i < loads; ++i) {
        int row = i * threads_row + opus::get<0>(u_ga);
        int g_off = row * opus::get<2>(u_ga) + wave_k_tile_offset
                    + opus::get<1>(u_ga);
        auto v = g_a.template load<T::VEC_A>(g_off);
        #pragma unroll
        for (int j = 0; j < T::VEC_A; ++j) out[i * T::VEC_A + j] = v[j];
    }
    return out;
}

template<typename T, typename G, typename U>
OPUS_D auto load_b_wave_tile(G& g_b, const U& u_gb,
                             int wave_k_tile_offset)
{
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_row = opus::get_warp_size() / threads_k;
    constexpr int loads = T::B_N / threads_row;

    opus::vector_t<typename T::D_B, T::VEC_B * loads> out;
    #pragma unroll
    for (int i = 0; i < loads; ++i) {
        int row = i * threads_row + opus::get<0>(u_gb);
        int g_off = row * opus::get<2>(u_gb) + wave_k_tile_offset
                    + opus::get<1>(u_gb);
        auto v = g_b.template load<T::VEC_B>(g_off);
        #pragma unroll
        for (int j = 0; j < T::VEC_B; ++j) out[i * T::VEC_B + j] = v[j];
    }
    return out;
}

template<typename T, typename V, typename U>
OPUS_D void store_a_wave_to_lds(char* smem_a_base, int wave_slab_id,
                                const V& va, const U& u_sa)
{
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_row = opus::get_warp_size() / threads_k;
    constexpr int loads = T::B_M / threads_row;
    constexpr int a_stride = T::B_K + 4;
    constexpr int a_wave_bytes = T::B_M * a_stride * sizeof(typename T::D_A);

    typename T::D_A* s = reinterpret_cast<typename T::D_A*>(
        smem_a_base + wave_slab_id * a_wave_bytes);
    #pragma unroll
    for (int i = 0; i < loads; ++i) {
        int row = i * threads_row + opus::get<0>(u_sa);
        int s_off = row * a_stride + opus::get<1>(u_sa);
        #pragma unroll
        for (int j = 0; j < T::VEC_A; ++j) {
            s[s_off + j] = va[i * T::VEC_A + j];
        }
    }
}

template<typename T, typename V, typename U>
OPUS_D void store_b_wave_to_lds(char* smem_b_base, int wave_slab_id,
                                const V& vb, const U& u_sb)
{
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_row = opus::get_warp_size() / threads_k;
    constexpr int loads = T::B_N / threads_row;
    constexpr int b_stride = T::B_K + 4;
    constexpr int b_wave_bytes = T::B_N * b_stride * sizeof(typename T::D_B);

    typename T::D_B* s = reinterpret_cast<typename T::D_B*>(
        smem_b_base + wave_slab_id * b_wave_bytes);
    #pragma unroll
    for (int i = 0; i < loads; ++i) {
        int row = i * threads_row + opus::get<0>(u_sb);
        int s_off = row * b_stride + opus::get<1>(u_sb);
        #pragma unroll
        for (int j = 0; j < T::VEC_B; ++j) {
            s[s_off + j] = vb[i * T::VEC_B + j];
        }
    }
}

template<typename T, typename U>
__device__ __forceinline__
short4_ab load_a_mfma_operand(const char* smem_a_base, int wave_slab_id,
                              int e_m, int e_k, const U& u_ua)
{
    constexpr int a_stride = T::B_K + 4;
    constexpr int a_wave_bytes = T::B_M * a_stride * sizeof(typename T::D_A);
    const typename T::D_A* s = reinterpret_cast<const typename T::D_A*>(
        smem_a_base + wave_slab_id * a_wave_bytes);
    int row = e_m * T::W_M + opus::get<0>(u_ua);
    int k = e_k * T::W_K + opus::get<1>(u_ua);
    return *reinterpret_cast<const short4_ab*>(s + row * a_stride + k);
}

template<typename T, typename U>
__device__ __forceinline__
short4_ab load_b_mfma_operand(const char* smem_b_base, int wave_slab_id,
                              int e_n, int e_k, const U& u_ub)
{
    constexpr int b_stride = T::B_K + 4;
    constexpr int b_wave_bytes = T::B_N * b_stride * sizeof(typename T::D_B);
    const typename T::D_B* s = reinterpret_cast<const typename T::D_B*>(
        smem_b_base + wave_slab_id * b_wave_bytes);
    int row = e_n * T::W_N + opus::get<0>(u_ub);
    int k = e_k * T::W_K + opus::get<1>(u_ub);
    return *reinterpret_cast<const short4_ab*>(s + row * b_stride + k);
}

// ---- Mainloop / Compute ----

template<typename T, typename U_A, typename U_B>
__device__ __forceinline__
void wave_compute_one_k_tile(const char* smem_a_base,
                             const char* smem_b_base,
                             int wave_slab_id,
                             const U_A& u_ua,
                             const U_B& u_ub,
                             float4_acc* acc)
{
    constexpr int OPERANDS_PER_K = T::E_M + T::E_N;
    constexpr int MFMAS_PER_K = T::E_M * T::E_N;
    // Keep enough K stages for one stage's MFMAs to cover its operand reads.
    constexpr int PIPELINE_STAGES =
        (OPERANDS_PER_K + MFMAS_PER_K - 1) / MFMAS_PER_K;
    constexpr int K_STAGES =
        PIPELINE_STAGES < T::E_K ? PIPELINE_STAGES : T::E_K;

    opus::array<short4_ab, K_STAGES * T::E_M> v_a;
    opus::array<short4_ab, K_STAGES * T::E_N> v_b;
    opus::array<float4_acc, T::E_M * T::E_N> v_c;

    opus::static_for<K_STAGES>([&](auto stage) {
        constexpr int stage_idx = decltype(stage)::value;

        opus::static_for<T::E_M>([&](auto e_m) {
            constexpr int e_m_idx = decltype(e_m)::value;
            v_a[stage_idx * T::E_M + e_m_idx] =
                load_a_mfma_operand<T>(
                    smem_a_base, wave_slab_id,
                    e_m_idx, stage_idx, u_ua);
        });
        opus::static_for<T::E_N>([&](auto e_n) {
            constexpr int e_n_idx = decltype(e_n)::value;
            v_b[stage_idx * T::E_N + e_n_idx] =
                load_b_mfma_operand<T>(
                    smem_b_base, wave_slab_id,
                    e_n_idx, stage_idx, u_ub);
        });
    });

    opus::static_for<T::E_M * T::E_N>([&](auto c) {
        constexpr int c_idx = decltype(c)::value;
        v_c[c_idx] = acc[c_idx];
    });

    opus::static_for<T::E_K>([&](auto e_k) {
        constexpr int e_k_idx = decltype(e_k)::value;
        constexpr int stage_idx = e_k_idx % K_STAGES;
        constexpr bool has_future_k = e_k_idx + K_STAGES < T::E_K;
        opus::array<short4_ab, T::E_M> next_a;
        opus::array<short4_ab, T::E_N> next_b;

        opus::static_for<T::E_N>([&](auto e_n) {
            constexpr int e_n_idx = decltype(e_n)::value;

            if constexpr (K_STAGES == 1) {
                // Constrain LLVM while the single operand slot is recycled.
                constexpr int pending_b = T::E_N - 1 - e_n_idx;
                s_waitcnt_lgkmcnt(opus::number<pending_b>{});
            }

            if constexpr (has_future_k && e_n_idx + 1 == T::E_N) {
                opus::static_for<T::E_M>([&](auto e_m) {
                    constexpr int e_m_idx = decltype(e_m)::value;
                    next_a[e_m_idx] = load_a_mfma_operand<T>(
                        smem_a_base, wave_slab_id,
                        e_m_idx, e_k_idx + K_STAGES, u_ua);
                });
                opus::static_for<T::E_N>([&](auto next_e_n) {
                    constexpr int next_e_n_idx = decltype(next_e_n)::value;
                    next_b[next_e_n_idx] = load_b_mfma_operand<T>(
                        smem_b_base, wave_slab_id,
                        next_e_n_idx, e_k_idx + K_STAGES, u_ub);
                });
            }

            opus::static_for<T::E_M>([&](auto e_m) {
                constexpr int e_m_idx = decltype(e_m)::value;
                constexpr int c_idx = e_m_idx * T::E_N + e_n_idx;
                v_c[c_idx] = opus::mfma_f32_16x16x16_bf16{}(
                    v_a[stage_idx * T::E_M + e_m_idx],
                    v_b[stage_idx * T::E_N + e_n_idx],
                    v_c[c_idx]);
            });
        });

        if constexpr (has_future_k) {
            opus::static_for<T::E_M>([&](auto e_m) {
                constexpr int e_m_idx = decltype(e_m)::value;
                v_a[stage_idx * T::E_M + e_m_idx] = next_a[e_m_idx];
            });
            opus::static_for<T::E_N>([&](auto e_n) {
                constexpr int e_n_idx = decltype(e_n)::value;
                v_b[stage_idx * T::E_N + e_n_idx] = next_b[e_n_idx];
            });
        }
    });

    opus::static_for<T::E_M * T::E_N>([&](auto c) {
        constexpr int c_idx = decltype(c)::value;
        acc[c_idx] = v_c[c_idx];
    });
}

// ---- Epilogue ----

template<typename T>
OPUS_D void store_wave_acc_to_lds_partial(char* lds_partial_base, int wave_id_k,
                                          const float4_acc* acc, int lane_id)
{
    constexpr int E_M = T::E_M;
    constexpr int E_N = T::E_N;
    constexpr int partial_stride = T::B_M * T::B_N;
    float* s = reinterpret_cast<float*>(lds_partial_base) +
               wave_id_k * partial_stride;

    int m_sub = lane_id / T::W_N;
    int n     = lane_id % T::W_N;
    #pragma unroll
    for (int e_m = 0; e_m < E_M; ++e_m) {
        #pragma unroll
        for (int e_n = 0; e_n < E_N; ++e_n) {
            int m_base = e_m * T::W_M + m_sub * 4;
            int n_pos  = e_n * T::W_N + n;
            float4_acc a = acc[e_m * E_N + e_n];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[(m_base + i) * T::B_N + n_pos] = a[i];
            }
        }
    }
}

template<typename T, typename Y_T>
OPUS_D void reduce_partials_and_store_y(const char* lds_partial_base,
                                        Y_T* ptr_y, int row, int col,
                                        int m_total, int n_total,
                                        int stride_y, int tid_in_wg)
{
    constexpr int TILE_CELLS = T::B_M * T::B_N;
    const float* partials = reinterpret_cast<const float*>(lds_partial_base);
    const bool full_tile = (row + T::B_M <= m_total) && (col + T::B_N <= n_total);
    for (int idx = tid_in_wg; idx < TILE_CELLS; idx += T::BLOCK_SIZE) {
        int m = idx / T::B_N;
        int n = idx - m * T::B_N;
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < T::T_K; ++w) {
            sum += partials[w * TILE_CELLS + idx];
        }
        int gm = row + m;
        int gn = col + n;
        if (full_tile || (gm < m_total && gn < n_total)) {
            ptr_y[gm * stride_y + gn] = static_cast<Y_T>(sum);
        }
    }
}

OPUS_D void atomic_add_bf16x2(__bf16* ptr, float x0, float x1)
{
    bf16x2_accum v;
    v[0] = static_cast<__bf16>(x0);
    v[1] = static_cast<__bf16>(x1);
    __builtin_amdgcn_global_atomic_fadd_v2bf16(
        reinterpret_cast<bf16x2_accum*>(ptr), v);
}

template<typename T, typename Y_T>
OPUS_D void zero_output_tile_bf16x2(Y_T* ptr_y, int row, int col,
                                    int m_total, int n_total,
                                    int stride_y, int tid_in_wg)
{
    constexpr int TILE_CELLS = T::B_M * T::B_N;
    static_assert(std::is_same<Y_T, __bf16>::value,
                  "WKC atomic accumulate supports bf16 output only");
    const bool full_tile = (row + T::B_M <= m_total) && (col + T::B_N <= n_total);
    bf16x2_accum zero;
    zero[0] = static_cast<__bf16>(0.0f);
    zero[1] = static_cast<__bf16>(0.0f);

    for (int idx = tid_in_wg * 2; idx < TILE_CELLS; idx += T::BLOCK_SIZE * 2) {
        int m = idx / T::B_N;
        int n = idx - m * T::B_N;
        int gm = row + m;
        int gn = col + n;
        if (full_tile || (gm < m_total && gn + 1 < n_total)) {
            __bf16* out = reinterpret_cast<__bf16*>(
                ptr_y + gm * stride_y + gn);
            *reinterpret_cast<bf16x2_accum*>(out) = zero;
        } else if (gm < m_total && gn < n_total) {
            ptr_y[gm * stride_y + gn] = static_cast<Y_T>(0.0f);
        }
    }
}

template<typename T, typename Y_T>
OPUS_D void reduce_partials_and_atomic_add_y(
    const char* lds_partial_base, Y_T* ptr_y, int row, int col,
    int m_total, int n_total, int stride_y, int tid_in_wg)
{
    constexpr int TILE_CELLS = T::B_M * T::B_N;
    const float* partials = reinterpret_cast<const float*>(lds_partial_base);
    static_assert(std::is_same<Y_T, __bf16>::value,
                  "WKC atomic accumulate supports bf16 output only");

    const bool full_tile = (row + T::B_M <= m_total) && (col + T::B_N <= n_total);
    for (int idx = tid_in_wg * 2; idx < TILE_CELLS; idx += T::BLOCK_SIZE * 2) {
        int m = idx / T::B_N;
        int n = idx - m * T::B_N;
        int gm = row + m;
        int gn = col + n;
        if (!(full_tile || (gm < m_total && gn + 1 < n_total))) {
            continue;
        }

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < T::T_K; ++w) {
            sum0 += partials[w * TILE_CELLS + idx];
            sum1 += partials[w * TILE_CELLS + idx + 1];
        }
        __bf16* out = reinterpret_cast<__bf16*>(
            ptr_y + gm * stride_y + gn);
        atomic_add_bf16x2(out, sum0, sum1);
    }
}

} // namespace opus_wkc_gfx942

#endif // __HIP_DEVICE_COMPILE__

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 3)
void gemm_a16w16_wave_k_coop_kernel(opus_gemm_noscale_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    using namespace opus_wkc_gfx942;
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;

    constexpr int K_TILE_FULL = T::B_K * T::T_K;  // full WG K-tile = T_K wave-tiles
    constexpr int A_BYTES = T::B_M * (T::B_K + 4) * sizeof(D_A) * T::T_K;
    constexpr int B_BYTES = T::B_N * (T::B_K + 4) * sizeof(D_B) * T::T_K;
    constexpr int REDUCE_BYTES = T::T_K * T::B_M * T::B_N * sizeof(float);
    constexpr int AB_BYTES = A_BYTES + B_BYTES;
    constexpr bool ALIAS_PARTIAL =
        (AB_BYTES + REDUCE_BYTES > 64 * 1024) &&
        (AB_BYTES <= 64 * 1024);
    constexpr int LDS_BYTES =
        ALIAS_PARTIAL
            ? (AB_BYTES > REDUCE_BYTES ? AB_BYTES : REDUCE_BYTES)
            : AB_BYTES + REDUCE_BYTES;
    static_assert(LDS_BYTES <= 64 * 1024, "wave-K-coop: LDS budget exceeded");

    const int tile_n = opus::block_id_x();
    const int tile_m = opus::block_id_y();
    const int batch_id = opus::block_id_z();
    const int row = tile_m * T::B_M;
    const int col = tile_n * T::B_N;

    const int tid = opus::thread_id_x();
    const int wave_id =
        __builtin_amdgcn_readfirstlane(tid / opus::get_warp_size());
    const int wave_id_k = wave_id;
    const int lane_id = tid % opus::get_warp_size();

    const D_A* ptr_a = reinterpret_cast<const D_A*>(kargs.ptr_a)
                       + batch_id * kargs.stride_a_batch + row * kargs.stride_a;
    const D_B* ptr_b = reinterpret_cast<const D_B*>(kargs.ptr_b)
                       + batch_id * kargs.stride_b_batch + col * kargs.stride_b;
    D_C* ptr_y = reinterpret_cast<D_C*>(kargs.ptr_c)
                 + batch_id * kargs.stride_c_batch;

    const bool full_tile = row + T::B_M <= kargs.m && col + T::B_N <= kargs.n;
    auto g_a = make_gmem(
        ptr_a,
        full_tile ? 0xffffffffu
                  : (unsigned int)(((kargs.m - row) * kargs.stride_a) * sizeof(D_A)));
    auto g_b = make_gmem(
        ptr_b,
        full_tile ? 0xffffffffu
                  : (unsigned int)(((kargs.n - col) * kargs.stride_b) * sizeof(D_B)));

    // LDS buffer (reused for A/B during K-loop, partials after)
    __shared__ char smem[LDS_BYTES];
    char* smem_a = smem;
    char* smem_b = smem + A_BYTES;
    char* smem_partial = smem + A_BYTES + B_BYTES;
    if constexpr (ALIAS_PARTIAL) {
        smem_partial = smem;
    }

    // Per-wave accumulators
    float4_acc acc[T::E_M * T::E_N] = {};

    const int wg_k_loops = ceil_div(kargs.k, K_TILE_FULL);
    const int wave_k_offset = wave_id_k * T::B_K;

    auto u_ga = make_layout_wkc_ga<T>(lane_id, kargs.stride_a);
    auto u_gb = make_layout_wkc_gb<T>(lane_id, kargs.stride_b);
    auto u_sa = make_layout_wkc_sa<T>(lane_id);
    auto u_sb = make_layout_wkc_sb<T>(lane_id);
    auto u_ua = make_layout_wkc_ua<T>(lane_id);
    auto u_ub = make_layout_wkc_ub<T>(lane_id);

    // PROLOGUE: prefetch the first per-wave K slab into VGPR.
    auto va = load_a_wave_tile<T>(
        g_a, u_ga, wave_k_offset);
    auto vb = load_b_wave_tile<T>(
        g_b, u_gb, wave_k_offset);

    // MAIN LOOP: stage current VGPR slab, prefetch next slab, then compute.
    #pragma unroll 4
    for (int t = 0; t < wg_k_loops; ++t) {
        store_b_wave_to_lds<T>(smem_b, wave_id_k, vb, u_sb);
        store_a_wave_to_lds<T>(smem_a, wave_id_k, va, u_sa);
        const bool has_next = (t + 1) < wg_k_loops;
        auto va_next = va;
        auto vb_next = vb;
        if (has_next) {
            int next_k_base = (t + 1) * K_TILE_FULL;
            vb_next = load_b_wave_tile<T>(
                g_b, u_gb, next_k_base + wave_k_offset);
            va_next = load_a_wave_tile<T>(
                g_a, u_ga, next_k_base + wave_k_offset);
        }
        wave_compute_one_k_tile<T>(
            smem_a, smem_b, wave_id_k, u_ua, u_ub,
            acc);
        va = va_next;
        vb = vb_next;
    }

    // EPILOGUE: deterministic in-WG K reduce and final bf16 store.
    if constexpr (ALIAS_PARTIAL) {
        __builtin_amdgcn_s_barrier();
    }
    store_wave_acc_to_lds_partial<T>(
        smem_partial, wave_id_k, acc, lane_id);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    reduce_partials_and_store_y<T, D_C>(
        smem_partial, ptr_y, row, col, kargs.m, kargs.n, kargs.stride_c, tid);

#endif // __gfx942__
#endif // __HIP_DEVICE_COMPILE__
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a16w16_wave_k_coop_accum_kernel(opus_gemm_noscale_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    using namespace opus_wkc_gfx942;
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;

    constexpr int SPLIT_K = 8;
    constexpr int K_PER_SPLIT = T::B_K * T::T_K;
    constexpr int A_BYTES = T::B_M * (T::B_K + 4) * sizeof(D_A) * T::T_K;
    constexpr int B_BYTES = T::B_N * (T::B_K + 4) * sizeof(D_B) * T::T_K;
    constexpr int REDUCE_BYTES = T::T_K * T::B_M * T::B_N * sizeof(float);
    constexpr int AB_BYTES = A_BYTES + B_BYTES;
    constexpr bool ALIAS_PARTIAL =
        (AB_BYTES + REDUCE_BYTES > 64 * 1024) &&
        (AB_BYTES <= 64 * 1024);
    constexpr int LDS_BYTES =
        ALIAS_PARTIAL
            ? (AB_BYTES > REDUCE_BYTES ? AB_BYTES : REDUCE_BYTES)
            : AB_BYTES + REDUCE_BYTES;
    static_assert(LDS_BYTES <= 64 * 1024,
                  "WKC accumulate LDS budget exceeded");

    const int tile_split = opus::block_id_x();
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int split_id = tile_split / num_tiles_n;
    const int tile_n = tile_split - split_id * num_tiles_n;
    const int tile_m = opus::block_id_y();
    const int batch_id = opus::block_id_z();
    const int row = tile_m * T::B_M;
    const int col = tile_n * T::B_N;

    const int tid = opus::thread_id_x();
    const int wave_id =
        __builtin_amdgcn_readfirstlane(tid / opus::get_warp_size());
    const int wave_id_k = wave_id;
    const int lane_id = tid % opus::get_warp_size();

    const D_A* ptr_a = reinterpret_cast<const D_A*>(kargs.ptr_a)
                       + batch_id * kargs.stride_a_batch + row * kargs.stride_a;
    const D_B* ptr_b = reinterpret_cast<const D_B*>(kargs.ptr_b)
                       + batch_id * kargs.stride_b_batch + col * kargs.stride_b;
    D_C* ptr_y = reinterpret_cast<D_C*>(kargs.ptr_c)
                 + batch_id * kargs.stride_c_batch;

    const bool full_tile = row + T::B_M <= kargs.m && col + T::B_N <= kargs.n;
    auto g_a = make_gmem(
        ptr_a,
        full_tile ? 0xffffffffu
                  : (unsigned int)(((kargs.m - row) * kargs.stride_a) * sizeof(D_A)));
    auto g_b = make_gmem(
        ptr_b,
        full_tile ? 0xffffffffu
                  : (unsigned int)(((kargs.n - col) * kargs.stride_b) * sizeof(D_B)));

    if (split_id == 0) {
        zero_output_tile_bf16x2<T, D_C>(
            ptr_y, row, col, kargs.m, kargs.n, kargs.stride_c, tid);
        __builtin_amdgcn_s_barrier();
    }

    __shared__ char smem[LDS_BYTES];
    char* smem_a = smem;
    char* smem_b = smem + A_BYTES;
    char* smem_partial = smem + A_BYTES + B_BYTES;
    if constexpr (ALIAS_PARTIAL) {
        smem_partial = smem;
    }

    float4_acc acc[T::E_M * T::E_N] = {};

    auto u_ga = make_layout_wkc_ga<T>(lane_id, kargs.stride_a);
    auto u_gb = make_layout_wkc_gb<T>(lane_id, kargs.stride_b);
    auto u_sa = make_layout_wkc_sa<T>(lane_id);
    auto u_sb = make_layout_wkc_sb<T>(lane_id);
    auto u_ua = make_layout_wkc_ua<T>(lane_id);
    auto u_ub = make_layout_wkc_ub<T>(lane_id);

    #pragma unroll 1
    for (int k_base = split_id * K_PER_SPLIT + wave_id_k * T::B_K;
         k_base < kargs.k;
         k_base += SPLIT_K * K_PER_SPLIT) {
        auto vb = load_b_wave_tile<T>(g_b, u_gb, k_base);
        auto va = load_a_wave_tile<T>(g_a, u_ga, k_base);
        store_b_wave_to_lds<T>(smem_b, wave_id_k, vb, u_sb);
        store_a_wave_to_lds<T>(smem_a, wave_id_k, va, u_sa);
        wave_compute_one_k_tile<T>(
            smem_a, smem_b, wave_id_k, u_ua, u_ub,
            acc);
    }

    if constexpr (ALIAS_PARTIAL) {
        __builtin_amdgcn_s_barrier();
    }
    store_wave_acc_to_lds_partial<T>(
        smem_partial, wave_id_k, acc, lane_id);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    reduce_partials_and_atomic_add_y<T, D_C>(
        smem_partial, ptr_y, row, col, kargs.m, kargs.n, kargs.stride_c, tid);

#endif // __gfx942__
#endif // __HIP_DEVICE_COMPILE__
}
