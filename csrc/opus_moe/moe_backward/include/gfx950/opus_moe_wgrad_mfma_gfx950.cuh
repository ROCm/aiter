// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused opus-MFMA grouped wgrad (BF16 -> FP32). dW[e] = dy_e^T @ a_e reducing
// over routes. Inputs are FEATURE-MAJOR transposed & route-padded activations
// (build_padded_transposed): dyT [P, Mp], aT [Q, Mp], so the contraction dim
// (routes) is contiguous -> this is an NT GEMM C[P,Q]=dyT_e @ aT_e^T, and the
// a16w16 BF16 NT pipeline applies directly. Per-block expert = grid.z; each
// block contracts over its expert's padded route columns [pad_offs[e]:e+1].
// Output is FP32 (D_C=D_ACC). Correctness-first.
#pragma once

#include "gfx950/opus_gemm_pipeline_a16w16_gfx950.cuh" // layout fns + CPOL + traits + mma

struct opus_moe_wgrad_mfma_kargs
{
    const void* __restrict__ ptr_dyT;   // [P, Mp] feature-major (A: rows=P, K=routes contiguous)
    const void* __restrict__ ptr_aT;    // [Q, Mp] feature-major (B: rows=Q, K=routes contiguous)
    void* __restrict__ ptr_dW;          // [E, P, Q] fp32
    const int32_t* __restrict__ pad_offs; // [E+1] per-expert padded route cumulative
    int P, Q, Mp;
};

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void opus_moe_wgrad_mfma_kernel(
    opus_moe_wgrad_mfma_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using T     = opus::remove_cvref_t<Traits>;
    using D_A   = typename T::D_A;
    using D_B   = typename T::D_B;
    using D_C   = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    const int e          = opus::block_id_z();
    const int route_base = kargs.pad_offs[e];
    const int k          = kargs.pad_offs[e + 1] - route_base; // per-expert padded routes
    if(k <= 0)
        return;

    const int grid_dim_x  = opus::grid_size_x() / opus::block_size_x();
    int wgid              = (opus::block_id_y() * grid_dim_x) + opus::block_id_x();
    const int num_tiles_m = ceil_div(kargs.P, T::B_M);      // over P (dy features)
    const int num_tiles_n = ceil_div_constexpr(kargs.Q, T::B_N); // over Q (a features)

    int tile_m_id = wgid / num_tiles_n;
    int tile_n_id = wgid % num_tiles_n;
    if(tile_m_id >= num_tiles_m)
        return;

    int row = tile_m_id * T::B_M;
    int col = tile_n_id * T::B_N;

    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    const int stride_a = kargs.Mp; // dyT row stride
    const int stride_b = kargs.Mp; // aT  row stride
    const int stride_c = kargs.Q;  // dW[e] row stride

    // A = dyT[row : row+B_M, route_base : route_base+k]; K (routes) contiguous.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_dyT) +
                             static_cast<int64_t>(row) * stride_a + route_base,
                         (kargs.P - row) * static_cast<int64_t>(stride_a) * sizeof(D_A));
    // B = aT[col : col+B_N, route_base : route_base+k].
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_aT) +
                             static_cast<int64_t>(col) * stride_b + route_base,
                         (kargs.Q - col) * static_cast<int64_t>(stride_b) * sizeof(D_B));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_dW) +
                         static_cast<int64_t>(e) * kargs.P * kargs.Q +
                         static_cast<int64_t>(row) * stride_c + col);

    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    auto u_ga = make_layout_ga_noscale<T>(lane_id, wave_id_m, wave_id_n, stride_a);
    auto u_sa = make_layout_sa_noscale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra_noscale<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_noscale<T>(lane_id, wave_id_m, wave_id_n, stride_b);
    auto u_sb = make_layout_sb_noscale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_noscale<T>(lane_id, wave_id_n);

    constexpr int smem_a_byte = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 4];
    smem<D_A> s_a[2][2] = {
        {make_smem(reinterpret_cast<D_A*>(smem_a)), make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))},
        {make_smem(reinterpret_cast<D_A*>(smem_a + 2 * smem_a_byte)),
         make_smem(reinterpret_cast<D_A*>(smem_a + 3 * smem_a_byte))}};
    constexpr int smem_b_byte = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 4];
    smem<D_B> s_b[2][2] = {
        {make_smem(reinterpret_cast<D_B*>(smem_b)), make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte))},
        {make_smem(reinterpret_cast<D_B*>(smem_b + 2 * smem_b_byte)),
         make_smem(reinterpret_cast<D_B*>(smem_b + 3 * smem_b_byte))}};

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{}, seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);

    auto a_offset = [&](int half_tile_m, int tile_k) {
        return half_tile_m * T::HALF_B_M * stride_a + tile_k * T::B_K;
    };
    auto b_offset = [&](int half_tile_n, int tile_k) {
        return half_tile_n * T::HALF_B_N * stride_b + tile_k * T::B_K;
    };

    const int loops = ceil_div(k, T::B_K);
    int tic = 0, toc = 1;

    async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, 0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, 0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
    async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, 0), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, 0), opus::number<0>{}, opus::number<T::CACHECTL_A>{});

    if(wave_id_m == 1)
        __builtin_amdgcn_s_barrier();

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
    async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
    async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, 1), opus::number<0>{}, opus::number<T::CACHECTL_B>{});

    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
    __builtin_amdgcn_s_barrier();

    v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
    __builtin_amdgcn_s_barrier();

    for(int tile = 0; tile < loops - 2; tile += 2)
    {
        v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[tic][0].ptr, u_gb, u_sb, b_offset(0, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][0].ptr, u_ga, u_sa, a_offset(0, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        async_load<T::VEC_B>(g_b, s_b[tic][1].ptr, u_gb, u_sb, b_offset(1, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        v_b[0] = load<T::VEC_B>(s_b[toc][0], u_rb);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[toc][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[tic][1].ptr, u_ga, u_sa, a_offset(1, tile + 2), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[toc][1], u_rb);
        async_load<T::VEC_B>(g_b, s_b[toc][0].ptr, u_gb, u_sb, b_offset(0, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[toc][1], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][0].ptr, u_ga, u_sa, a_offset(0, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        async_load<T::VEC_B>(g_b, s_b[toc][1].ptr, u_gb, u_sb, b_offset(1, tile + 3), opus::number<0>{}, opus::number<T::CACHECTL_B>{});
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    {
        int tile = loops - 2;
        v_a = load<T::VEC_A>(s_a[tic][0], u_ra);
        async_load<T::VEC_A>(g_a, s_a[toc][1].ptr, u_ga, u_sa, a_offset(1, tile + 1), opus::number<0>{}, opus::number<T::CACHECTL_A>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        tic ^= 1;
        toc ^= 1;
    }
    {
        v_b[0] = load<T::VEC_B>(s_b[tic][0], u_rb);
        v_a   = load<T::VEC_A>(s_a[tic][0], u_ra);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = load<T::VEC_B>(s_b[tic][1], u_rb);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_a = load<T::VEC_A>(s_a[tic][1], u_ra);
        __builtin_amdgcn_s_barrier();
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if(wave_id_m == 0)
        __builtin_amdgcn_s_barrier();

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_gc      = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(stride_c, 1_I), p_coord_c);
    auto u_gc_m    = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
    auto u_gc_n    = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * stride_c + half_tile_n * T::HALF_B_N;
    };

    auto store_c = [&](auto& vc, int half_tile_m, int half_tile_n) {
        int g_c_offset = c_offset(half_tile_m, half_tile_n);
        int m_base     = row + half_tile_m * T::HALF_B_M;
        int n_base     = col + half_tile_n * T::HALF_B_N;
        auto pred      = [&](auto... ids) {
            return (m_base + u_gc_m(ids...)) < kargs.P && (n_base + u_gc_n(ids...)) < kargs.Q;
        };
        if constexpr(std::is_same_v<D_C, D_ACC>)
        {
            store_if<T::VEC_C>(g_c, pred, vc, u_gc, g_c_offset, opus::number<CPOL_NT>{});
        }
        else
        {
            auto vc_out = cast<D_C>(vc);
            store_if<T::VEC_C>(g_c, pred, vc_out, u_gc, g_c_offset, opus::number<CPOL_NT>{});
        }
    };

    store_c(v_c[0][0], 0, 0);
    store_c(v_c[0][1], 0, 1);
    store_c(v_c[1][0], 1, 0);
    store_c(v_c[1][1], 1, 1);
#else
    (void)kargs;
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// Traits: same tile as dgrad (256, 128x256x32, T=(2,2,1), W=16x16x32) but D_C=FP32.
using OpusMoeWgradMfmaTraits = opus_gemm_a16w16_traits_gfx950<
    256,
    opus::seq<128, 256, 32>,
    opus::tuple<opus::bf16_t, opus::bf16_t, opus::fp32_t, opus::fp32_t>,
    opus::seq<8, 8, 4>,
    opus::seq<2, 2, 1>,
    opus::seq<16, 16, 32>,
    /*HAS_BIAS*/ false,
    /*D_BIAS*/ void,
    /*HAS_OOB*/ true>;
static constexpr int OPUS_MOE_WGRAD_MFMA_B_K = 32; // pad routes to 2*B_K mult

inline void opus_moe_wgrad_mfma_launch_gfx950(const opus_moe_wgrad_mfma_kargs& k,
                                              int num_experts,
                                              hipStream_t stream)
{
    using T               = OpusMoeWgradMfmaTraits;
    const int num_tiles_m = (k.P + T::B_M - 1) / T::B_M;
    const int num_tiles_n = (k.Q + T::B_N - 1) / T::B_N;
    dim3 grid(num_tiles_n, num_tiles_m, num_experts);
    dim3 block(T::BLOCK_SIZE);
    opus_moe_wgrad_mfma_kernel<T><<<grid, block, 0, stream>>>(k);
}
