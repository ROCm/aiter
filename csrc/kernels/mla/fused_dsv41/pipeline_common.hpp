// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Adapted from AITER cbda2884f5b54aa10f32ca4ab43dafac08c9d8db.
// Four KV buffers, direct global->LDS, register P, two-slice LDS prefetch,
// and interleaved next-QK / previous-softmax / previous-PV.
#pragma once
#include <bit>
#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
#include <utility>
using bf16_t = opus::dtypes::bf16;
// Invalid sparse slots load this immutable zero row, never an unselected KV
// row. Keep one VMEM issue per slice so partial wait counts remain valid.
__device__ __constant__ bf16_t fused_zero_kv[512] = {};
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int D_TILE_SIZE_  = 512,
          int NUM_WARPS_    = 8,
          typename D_ATTN_  = bf16_t>
struct opus_mla_v4_prefill_a16w16_16mx8_32nx1_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE  = D_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_TILE_SIZE / SLICE_D;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D / W_K;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = WARP_SIZE * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave  = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt       = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt       = D_TILE_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_kv_tile_elems =
        smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_padding_32B);

    static constexpr int kv_buffer_load_insts =
        (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int k_ds_read_insts =
        (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);

    static constexpr size_t smem_size_bytes() { return 4 * smem_kv_tile_elems * sizeof(D_ATTN); }
};

using opus::operator""_I;

// global -> LDS async copy (GLOBAL_LOAD_LDS). Same call shape as opus::async_load; completion is
// tracked by vmcnt only (ISA 10.4). u_smem must be wave-uniform: it feeds M0.
template <opus::index_t VEC, class D, class LayoutG, class LayoutS>
__device__ inline void
global_load(const D* g_base, void* smem_base, const LayoutG& u_gmem, const LayoutS& u_smem)
{
    constexpr opus::index_t BYTES = VEC * static_cast<opus::index_t>(sizeof(D));
    constexpr opus::index_t N     = opus::layout_load_traits<LayoutG, VEC>::r_elem.value;
    constexpr auto g_imm          = opus::layout_imm_offsets_v<LayoutG, VEC>;
    constexpr auto s_imm          = opus::layout_imm_offsets_v<LayoutS, VEC>;

    const auto g_os = opus::layout_to_offsets<VEC>(u_gmem);
    const auto s_os = opus::layout_to_offsets<VEC>(u_smem);
    auto* s_ptr =
        reinterpret_cast<OPUS_LDS_ADDR char*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_base));
    const char* src = reinterpret_cast<const char*>(g_base) +
                      static_cast<int64_t>(g_os[0]) * static_cast<int64_t>(sizeof(D));
    const unsigned int m0_base = __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
        reinterpret_cast<__UINTPTR_TYPE__>(s_ptr + s_os[0] * static_cast<int>(sizeof(D)))));

    opus::static_for<N>([&](auto i) {
        constexpr int g_delta = (g_imm[i.value] - g_imm[0]) * static_cast<int>(sizeof(D));
        constexpr int m0_fix  = (s_imm[i.value] - s_imm[0]) * static_cast<int>(sizeof(D)) - g_delta;
        static_assert(
            m0_fix >= 0 && m0_fix % 4 == 0,
            "offset: shifts the LDS address too, so M0 must absorb it (ISA 9.1.9 / 10.3)");
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        const unsigned int m0_val = m0_base + m0_fix;
        const char* addr = src; // asm operands do not odr-use, so name it inside the lambda
#define OPUS_MLA_V4_GLOBAL_LOAD_LDS(mnemonic)                                                   \
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\t" mnemonic " %1, off offset:%2" ::"s"(m0_val), \
                 "v"(addr),                                                                     \
                 "n"(g_delta)                                                                   \
                 : "memory")
        if constexpr(BYTES == 16)
            OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_dwordx4");
        else if constexpr(BYTES == 12)
            OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_dwordx3");
        else if constexpr(BYTES == 4)
            OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_dword");
        else if constexpr(BYTES == 2)
            OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_ushort");
        else
            OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_ubyte");
#undef OPUS_MLA_V4_GLOBAL_LOAD_LDS
#else
        *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<D, VEC>*>(
            s_ptr + s_os[i.value] * static_cast<int>(sizeof(D))) =
            *reinterpret_cast<const opus::vector_t<D, VEC>*>(src + g_delta);
#endif
    });
}

// global -> VGPR, one issue per y-position of the layout (drop-in for opus::load(gmem, layout)).
template <opus::index_t VEC,
          class D,
          class Layout,
          std::enable_if_t<opus::is_layout_v<Layout>, bool> = true>
__device__ inline auto global_load(const D* g_base, const Layout& u)
{
    constexpr opus::index_t N = opus::layout_load_traits<Layout, VEC>::r_elem.value;
    const auto os             = opus::layout_to_offsets<VEC>(u);
    opus::vector_t<D, VEC * N> r;
    opus::static_for<N>([&](auto i) {
        opus::set_slice(r,
                        *reinterpret_cast<const opus::vector_t<D, VEC>*>(
                            reinterpret_cast<const char*>(g_base) +
                            static_cast<int64_t>(os[i.value]) * static_cast<int64_t>(sizeof(D))),
                        opus::number<i.value * VEC>{},
                        opus::number<(i.value + 1) * VEC>{});
    });
    return r;
}

// global -> VGPR at a plain element offset.
template <opus::index_t VEC, class D>
__device__ inline auto global_load(const D* g_base, opus::index_t os)
{
    return *reinterpret_cast<const opus::vector_t<D, VEC>*>(reinterpret_cast<const char*>(g_base) +
                                                            static_cast<int64_t>(os) *
                                                                static_cast<int64_t>(sizeof(D)));
}

// =============================================================================
// Variant 16mx8_32nx1 (T_M=NUM_WARPS, T_N=1) -- used when H > 32.
// =============================================================================
namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1 {

constexpr int MFMA_MASK    = 0x08;
constexpr int VALU_MASK    = 0x02;
constexpr int SALU_MASK    = 0x04;
constexpr int EXP_MASK     = 0x400;
constexpr int DS_READ_MASK = 0x100;

template <int Group>
__device__ inline void sched_compute_qk()
{
    opus::static_for<4>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(EXP_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(SALU_MASK, 1, Group);
    });
    opus::static_for<10>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(VALU_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(SALU_MASK, 1, Group);
    });
    opus::static_for<4>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);
        __builtin_amdgcn_sched_group_barrier(VALU_MASK, 2, Group);
    });
}

// Create layout for loading Q matrix from global memory
template <class T>
__device__ inline auto make_layout_q(int warp_id, int lane_id, int stride_q_h)
{
    constexpr auto q_block_shape = opus::make_tuple(opus::number<T::GEMM0_E_M>{},
                                                    opus::number<T::T_M>{},
                                                    opus::number<T::W_M>{},
                                                    opus::number<T::D_TILE_SIZE / T::W_K>{},
                                                    opus::number<T::WARP_SIZE / T::W_M>{},
                                                    opus::number<T::VEC_Q>{});

    constexpr auto q_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        q_block_shape,
        opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_h, 1_I}),
        opus::unfold_p_coord(q_block_dim,
                             opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Create layout for storing O matrix to global memory
template <class T>
__device__ inline auto make_layout_o(int warp_id, int lane_id, int stride_o_h)
{
    constexpr auto o_block_shape =
        opus::make_tuple(opus::number<T::GEMM1_E_M>{},
                         opus::number<T::T_M>{},
                         opus::number<T::W_M>{},
                         opus::number<T::D_TILE_SIZE / T::W_N>{},
                         opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{},
                         opus::number<T::WARP_SIZE / T::W_M>{},
                         opus::number<T::VEC_O>{});

    constexpr auto o_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        o_block_shape,
        opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_h, 1_I}),
        opus::unfold_p_coord(o_block_dim,
                             opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
}

// Create layout for loading K matrix from global memory
template <typename T>
__device__ inline auto make_layout_gkv(int warp_id, int lane_id)
{
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;
    constexpr int warps_d   = T::NUM_WARPS / T::smem_n_rpt;

    constexpr auto gk_block_shape = opus::make_tuple(opus::number<T::smem_d_rpt / warps_d>{},
                                                     opus::number<warps_d>{},
                                                     opus::number<threads_d>{},
                                                     opus::number<T::VEC_KV>{});

    constexpr auto gk_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                                                   opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        gk_block_shape,
        opus::unfold_x_stride(
            gk_block_dim, gk_block_shape, opus::tuple{opus::number<T::D_128B_SIZE>{}, 1_I}),
        opus::unfold_p_coord(gk_block_dim,
                             opus::tuple{warp_id / T::smem_n_rpt, lane_id % threads_d}));
}

// Create layout for storing K matrix to shared memory
template <typename T>
__device__ inline auto make_layout_skv(int warp_id)
{
    constexpr auto sk_block_shape =
        opus::make_tuple(opus::number<T::smem_d_rpt * T::smem_n_rpt / T::NUM_WARPS>{},
                         opus::number<T::NUM_WARPS>{},
                         opus::number<T::VEC_KV>{});

    constexpr auto sk_block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sk_block_shape,
        opus::unfold_x_stride(
            sk_block_dim,
            sk_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding_32B>{}, 1_I}),
        opus::unfold_p_coord(sk_block_dim, opus::tuple{warp_id}));
}

// Create layout for reading K matrix from shared memory to registers
template <typename T>
__device__ inline auto make_layout_rk(int lane_id)
{
    constexpr auto rk_block_shape = opus::make_tuple(opus::number<T::smem_n_rpt>{},
                                                     opus::number<T::GEMM0_E_N>{},
                                                     opus::number<T::W_N / T::smem_n_rpt>{},
                                                     opus::number<T::GEMM0_E_K>{},
                                                     opus::number<opus::get_warp_size() / T::W_N>{},
                                                     opus::number<T::VEC_KV>{});

    constexpr auto rk_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout(
        rk_block_shape,
        opus::unfold_x_stride(rk_block_dim,
                              rk_block_shape,
                              opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding_32B>{},
                                          opus::number<T::D_128B_SIZE>{},
                                          1_I}),
        opus::unfold_p_coord(
            rk_block_dim,
            opus::tuple{lane_id_n % T::smem_n_rpt, lane_id_n / T::smem_n_rpt, lane_id / T::W_N}));
}

template <class T>
__device__ inline auto make_layout_rv(int lane_id)
{
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo      = 4;
    constexpr int lane_hi      = lane_per_grp / lane_lo;

    constexpr int num_grps = T::WARP_SIZE / lane_per_grp;
    constexpr int grp_n    = T::W_N / (lane_lo * T::VEC_TR_V);
    constexpr int grp_k    = num_grps / grp_n;

    constexpr auto rv_block_shape = opus::make_tuple(opus::number<T::GEMM1_E_N>{},
                                                     opus::number<T::GEMM1_E_K>{},
                                                     opus::number<lane_hi>{},
                                                     opus::number<T::W_K / (lane_hi * grp_k)>{},
                                                     opus::number<grp_k>{},
                                                     opus::number<grp_n>{},
                                                     opus::number<lane_lo>{},
                                                     opus::number<T::VEC_TR_V>{});

    constexpr auto rv_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    int grp_id      = lane_id / lane_per_grp;
    int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout(
        rv_block_shape,
        opus::unfold_x_stride(rv_block_dim,
                              rv_block_shape,
                              opus::tuple{opus::number<grp_n * lane_lo * T::VEC_TR_V>{},
                                          opus::number<T::smem_linear_wave + T::smem_padding_32B>{},
                                          opus::number<T::D_128B_SIZE>{},
                                          1_I}),
        opus::unfold_p_coord(
            rv_block_dim,
            opus::tuple{
                lane_in_grp / lane_lo, grp_id / grp_n, grp_id % grp_n, lane_in_grp % lane_lo}));
}

template <class T>
__device__ inline auto make_layout_kv_indices(int warp_id, int lane_id)
{
    constexpr int threads_d = T::D_128B_SIZE / T::VEC_KV;

    constexpr auto kv_indices_shape =
        opus::make_tuple(opus::number<T::smem_n_per_wave>{}, opus::number<T::smem_n_rpt>{}, 1_I);

    constexpr auto kv_indices_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout(
        kv_indices_shape,
        opus::unfold_x_stride(kv_indices_dim, kv_indices_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(kv_indices_dim,
                             opus::tuple{lane_id / threads_d, warp_id % T::smem_n_rpt}));
}

template <typename T, typename V>
__device__ inline typename T::D_ACC attn_row_max(const V& v_s)
{
    using D_ACC                   = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_max                 = -INFINITY;
    opus::static_for<s_len>([&](auto i) { row_max = max(row_max, v_s[i.value]); });
    // swap lanes 32 apart (i <-> i+32).
    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    row_max = max(std::bit_cast<float>(res32.x), std::bit_cast<float>(res32.y));
    // swap lanes 16 apart (i <-> i+16).
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(row_max), std::bit_cast<opus::u32_t>(row_max), false, true);
    return max(std::bit_cast<float>(res16.x), std::bit_cast<float>(res16.y));
}

template <typename T, typename V>
__device__ inline float track_max(const V& s, float temperature, float& actual)
{
    float x = attn_row_max<T>(s) * temperature;
    actual  = fmaxf(actual, x);
    return fmaxf(x, -__FLT_MAX__);
}

template <typename T, typename V>
__device__ inline void
attn_row_scale_sub(V& v_s, typename T::D_ACC scale, typename T::D_ACC row_max)
{
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    opus::static_for<s_len>([&](auto i) { v_s[i.value] = v_s[i.value] * scale - row_max; });
}

template <typename T, opus::index_t Offset, opus::index_t Count, typename V>
__device__ inline void attn_exp2_slice(V& v_s)
{
    opus::static_for<Count>([&](auto i) {
        constexpr opus::index_t idx = Offset + i.value;
        v_s[idx]                    = __builtin_amdgcn_exp2f(v_s[idx]);
    });
}

template <typename T, typename V>
__device__ inline typename T::D_ACC attn_row_sum(const V& v_s)
{
    using D_ACC                   = typename T::D_ACC;
    constexpr opus::index_t s_len = opus::vector_traits<V>::size();
    D_ACC row_sum                 = 0.0f;
    opus::static_for<s_len>([&](auto i) { row_sum += v_s[i.value]; });
    // swap lanes 32 apart (i <-> i+32).
    opus::vector_t<opus::u32_t, 2> res32 = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    row_sum = std::bit_cast<float>(res32.x) + std::bit_cast<float>(res32.y);
    // swap lanes 16 apart (i <-> i+16).
    opus::vector_t<opus::u32_t, 2> res16 = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(row_sum), std::bit_cast<opus::u32_t>(row_sum), false, true);
    return std::bit_cast<float>(res16.x) + std::bit_cast<float>(res16.y);
}

template <typename T, typename V, size_t N>
__device__ inline void scale_output_tile(V (&v_o)[N], typename T::D_ACC scale)
{
    opus::static_for<N>([&](auto i) { v_o[i.value] *= scale; });
}

template <int THR_X, int THR_Y>
__device__ inline void attn_mask_vec2_imm(opus::u32_t rel_vgpr,
                                          opus::u32_t neg_inf_vgpr,
                                          opus::u32_t& x_ref,
                                          opus::u32_t& y_ref)
{
    uint64_t x_mask, y_mask;
    asm volatile("v_cmp_lt_i32_e64 %0, %6, %7\n\t"
                 "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
                 "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
                 "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
                 : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
                 : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr), "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
                 : "vcc");
}

template <typename T, typename V>
__device__ inline void
attn_mask_oob_score(V& v_s, int valid_kv_len, int kv_tile_idx, opus::u32_t neg_inf_v)
{
    using D_ACC    = typename T::D_ACC;
    using D_ACC_X2 = opus::vector_t<D_ACC, 2>;
    using U32_X2   = opus::vector_t<opus::u32_t, 2>;

    if((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len)
        return;

    constexpr int elems_per_wave_tile = (T::W_M * T::W_N) / T::WARP_SIZE;
    constexpr int c_pack              = 4;
    constexpr int c_rept              = elems_per_wave_tile / c_pack;
    constexpr int c_rept_stride       = (T::WARP_SIZE / T::W_M) * c_pack;

    const int last_valid_kv_pos = valid_kv_len - 1;
    const int k_start_pos       = kv_tile_idx * T::KV_TILE_SIZE;
    int lane_id                 = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    const int lane_group = lane_id / T::W_M;

    opus::static_for<T::GEMM0_E_N>([&](auto i_n) {
        constexpr int base_idx = i_n.value * elems_per_wave_tile;
        const int k_pos        = k_start_pos + i_n.value * T::W_N + lane_group * c_pack;
        const opus::u32_t rel  = static_cast<opus::u32_t>(last_valid_kv_pos - k_pos);

        opus::static_for<c_rept>([&](auto i_rept) {
            constexpr int rept_base_idx = base_idx + i_rept.value * c_pack;
            constexpr int thr_base      = i_rept.value * c_rept_stride;
            opus::static_for<c_pack / 2>([&](auto i_pair) {
                constexpr int idx   = rept_base_idx + i_pair.value * 2;
                constexpr int thr_x = thr_base + i_pair.value * 2;
                constexpr int thr_y = thr_x + 1;

                auto pair_acc     = opus::slice(v_s, opus::number<idx>{}, opus::number<idx + 2>{});
                auto pair_bits    = __builtin_bit_cast(U32_X2, pair_acc);
                opus::u32_t x_ref = pair_bits[0];
                opus::u32_t y_ref = pair_bits[1];
                attn_mask_vec2_imm<thr_x, thr_y>(rel, neg_inf_v, x_ref, y_ref);
                pair_bits[0] = x_ref;
                pair_bits[1] = y_ref;
                opus::set_slice(v_s,
                                __builtin_bit_cast(D_ACC_X2, pair_bits),
                                opus::number<idx>{},
                                opus::number<idx + 2>{});
            });
        });
    });
}

template <class T, class V>
__device__ inline void attn_mask_oob_value(V& v_v, int valid_kv_len, int kv_tile_idx)
{
    using D_ATTN = typename T::D_ATTN;

    if((kv_tile_idx + 1) * T::KV_TILE_SIZE <= valid_kv_len)
        return;

    int lane_id = opus::thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id)); // break CSE
    const int base = (lane_id / T::W_N) * T::VEC_TR_V;
    const int rel  = (valid_kv_len - 1) - kv_tile_idx * T::KV_TILE_SIZE - base;

    constexpr int en_stride = opus::vector_traits<V>::size() / T::GEMM1_E_N;
    opus::static_for<en_stride>([&](auto ik) {
        constexpr int k   = ik.value;
        constexpr int thr = (k / T::VEC_TR_V) * (T::W_K / 2) + (k % T::VEC_TR_V);
        if(thr > rel)
        {
            opus::static_for<T::GEMM1_E_N>(
                [&](auto e) { v_v[e.value * en_stride + k] = static_cast<D_ATTN>(0); });
        }
    });
}

} // namespace opus_mla_v4_prefill_a16w16_16mx8_32nx1
