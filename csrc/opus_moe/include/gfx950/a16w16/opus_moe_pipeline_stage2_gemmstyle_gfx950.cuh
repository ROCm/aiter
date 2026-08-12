// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../../opus_moe_common.cuh"
#include "opus_moe_traits_stage2_gfx950.cuh"

#include "opus/opus.hpp"

#ifdef __HIP_DEVICE_COMPILE__

template<typename T>
inline __device__ void opus_moe_stage2_tile_ids(int wgid,
                                                int num_tiles_m,
                                                int num_tiles_n,
                                                int& tile_m_id,
                                                int& tile_n_id)
{
    if constexpr(T::NUM_XCD > 1)
    {
        constexpr int nXCD = T::NUM_XCD;
        constexpr int W = T::SWIZZLE_W;
        constexpr int C = T::SWIZZLE_C;
        const int total_wgs = num_tiles_m * num_tiles_n;
        const int blocks_per_cycle = nXCD * C;
        const int tiles_per_group = W * num_tiles_n;
        const int limit = (total_wgs / blocks_per_cycle) * blocks_per_cycle;
        if(wgid >= limit)
        {
            const int full_groups = limit / tiles_per_group;
            const int covered_cols = (limit - full_groups * tiles_per_group) / W;
            const int partial_first_row = full_groups * W;
            int partial_row_extent = num_tiles_m - partial_first_row;
            if(partial_row_extent > W)
                partial_row_extent = W;
            const int tail = wgid - limit;
            const int partial_tiles =
                (partial_row_extent > 0) ? (num_tiles_n - covered_cols) * partial_row_extent
                                         : 0;
            if(tail < partial_tiles)
            {
                tile_m_id = partial_first_row + (tail % partial_row_extent);
                tile_n_id = covered_cols + (tail / partial_row_extent);
                return;
            }

            const int rest = tail - partial_tiles;
            tile_m_id = partial_first_row + partial_row_extent + rest / num_tiles_n;
            tile_n_id = rest % num_tiles_n;
            return;
        }

        const int xcd = wgid % nXCD;
        const int local = wgid / nXCD;
        const int chunk_idx = local / C;
        const int pos = local % C;
        const int swizzled = xcd * C + chunk_idx * blocks_per_cycle + pos;
        const int group_id = swizzled / tiles_per_group;
        const int first_row = group_id * W;
        int win_h = num_tiles_m - first_row;
        if(win_h > W)
            win_h = W;
        const int in_group = swizzled % tiles_per_group;
        tile_m_id = first_row + (in_group % win_h);
        tile_n_id = in_group / win_h;
        if(tile_n_id < num_tiles_n)
            return;
    }

    tile_m_id = wgid / num_tiles_n;
    tile_n_id = wgid % num_tiles_n;
}

template<typename T>
inline __device__ auto opus_moe_stage2_layout_ga(int lane_id,
                                                 int wave_id_m,
                                                 int wave_id_n,
                                                 int stride_a)
{
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto block_shape = opus::make_tuple(
        opus::number<(T::HALF_B_M + threads_m_per_block - 1) / threads_m_per_block>{},
        opus::number<T::T_M>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_N>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_a, opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}

template<typename T>
inline __device__ auto opus_moe_stage2_layout_sa(int lane_id,
                                                 int wave_id_m,
                                                 int wave_id_n)
{
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto block_shape = opus::make_tuple(
        opus::number<(T::smem_m_rep + num_waves - 1) / num_waves>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{T::smem_linear_wave + T::smem_padding, opus::number<1>{}}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<typename T>
inline __device__ auto opus_moe_stage2_layout_ra(int lane_id, int wave_id_m)
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M / T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<T::W_M * T::W_K / opus::get_warp_size() / T::VEC_A>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{},
                         opus::p_dim{},
                         opus::y_dim{},
                         opus::y_dim{},
                         opus::p_dim{},
                         opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{T::smem_linear_wave + T::smem_padding, opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{lane_id_m % T::T_N,
                        wave_id_m,
                        lane_id_m / T::T_N,
                        lane_id / T::W_M}));
}

template<typename T>
inline __device__ auto opus_moe_stage2_layout_gb(int lane_id,
                                                 int wave_id_m,
                                                 int wave_id_n,
                                                 int stride_b)
{
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / threads_n_per_block>{},
        opus::number<T::T_M>{},
        opus::number<threads_n_per_wave>{},
        opus::number<T::T_N>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_B>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_b, opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}

template<typename T>
inline __device__ auto opus_moe_stage2_layout_sb(int lane_id,
                                                 int wave_id_m,
                                                 int wave_id_n)
{
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{T::smem_linear_wave + T::smem_padding, opus::number<1>{}}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<typename T>
inline __device__ auto opus_moe_stage2_layout_rb(int lane_id, int wave_id_n)
{
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N / T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::W_N / T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<T::W_N * T::W_K / opus::get_warp_size() / T::VEC_B>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{},
                         opus::p_dim{},
                         opus::y_dim{},
                         opus::y_dim{},
                         opus::p_dim{},
                         opus::y_dim{}));

    const int lane_id_n = lane_id % T::W_N;

    return opus::make_layout<T::VEC_B>(
        block_shape,
        opus::unfold_x_stride(
            block_dim,
            block_shape,
            opus::tuple{T::smem_linear_wave + T::smem_padding, opus::number<1>{}}),
        opus::unfold_p_coord(
            block_dim,
            opus::tuple{wave_id_n / T::T_M,
                        lane_id_n % T::T_N,
                        wave_id_n % T::T_M,
                        lane_id_n / T::T_N,
                        lane_id / T::W_N}));
}

#endif

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void
opus_moe_stage2_gemmstyle_kernel_gfx950(opus_moe_stage2_bf16_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;
    using opus::operator""_I;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;

    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int BK = T::B_K;
    constexpr int HALF_BM = T::HALF_B_M;
    constexpr int HALF_BN = T::HALF_B_N;

    const int valid_rows = kargs.num_valid_ids[0];
    const int grid_dim_x = static_cast<int>(gridDim.x);
    const int wgid = static_cast<int>(blockIdx.y) * grid_dim_x +
                     static_cast<int>(blockIdx.x);

    int tile_m_id;
    int tile_n_id;
    opus_moe_stage2_tile_ids<T>(
        wgid, static_cast<int>(gridDim.y), grid_dim_x, tile_m_id, tile_n_id);
    const int route_base = tile_m_id * BM;
    const int col_base = tile_n_id * BN;
    if(route_base >= valid_rows)
        return;

    const int sorted_block_id =
        (kargs.block_m == BM) ? tile_m_id : (route_base / kargs.block_m);
    const int expert_id = kargs.sorted_expert_ids[sorted_block_id];
    if(expert_id < 0 || expert_id >= kargs.num_experts)
        return;

    const int tid = static_cast<int>(thread_id_x());
    const int lane_id = tid % get_warp_size();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / get_warp_size());
    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    constexpr int smem_a_byte =
        T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    constexpr int smem_b_byte =
        T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_a_storage[smem_a_byte * 4];
    __shared__ char smem_b_storage[smem_b_byte * 4];
    __shared__ int32_t smem_a_base[BM];
    __shared__ int32_t smem_route_base[BM];
    __shared__ float smem_weight[BM];

    smem<D_A> s_a[2][2] = {
        {make_smem(reinterpret_cast<D_A*>(smem_a_storage)),
         make_smem(reinterpret_cast<D_A*>(smem_a_storage + smem_a_byte))},
        {make_smem(reinterpret_cast<D_A*>(smem_a_storage + 2 * smem_a_byte)),
         make_smem(reinterpret_cast<D_A*>(smem_a_storage + 3 * smem_a_byte))},
    };
    smem<D_B> s_b[2][2] = {
        {make_smem(reinterpret_cast<D_B*>(smem_b_storage)),
         make_smem(reinterpret_cast<D_B*>(smem_b_storage + smem_b_byte))},
        {make_smem(reinterpret_cast<D_B*>(smem_b_storage + 2 * smem_b_byte)),
         make_smem(reinterpret_cast<D_B*>(smem_b_storage + 3 * smem_b_byte))},
    };

    const D_A* __restrict__ inter_states = reinterpret_cast<const D_A*>(kargs.inter_states);
    const D_B* __restrict__ w2 = reinterpret_cast<const D_B*>(kargs.w2);

    const unsigned int a_size_bytes =
        static_cast<unsigned int>(static_cast<unsigned long long>(kargs.token_num) *
                                  static_cast<unsigned long long>(kargs.stride_a_t) *
                                  sizeof(D_A));
    auto g_a = make_gmem(inter_states, a_size_bytes);

    const int64_t w2_expert_base = static_cast<int64_t>(expert_id) * kargs.stride_w_e;
    const int b_rows_remaining = (kargs.model_dim > col_base) ? (kargs.model_dim - col_base) : 0;
    unsigned int b_size_bytes = 0;
    if(b_rows_remaining > 0)
    {
        b_size_bytes = static_cast<unsigned int>(
            (static_cast<unsigned long long>(b_rows_remaining - 1) *
                 static_cast<unsigned long long>(kargs.stride_w_h) +
             static_cast<unsigned long long>(kargs.inter_dim)) *
            sizeof(D_B));
    }
    auto g_b = make_gmem(w2 + w2_expert_base +
                static_cast<int64_t>(col_base) * kargs.stride_w_h,
                         b_size_bytes);

    for(int local_m = tid; local_m < BM; local_m += T::BLOCK_SIZE)
    {
        const int row = route_base + local_m;
        const int32_t packed = kargs.sorted_token_ids[row];
        const int token = opus_moe_token_id(packed);
        const int slot = opus_moe_topk_slot(packed);
        smem_a_base[local_m] = static_cast<int32_t>(
            static_cast<int64_t>(token) * kargs.stride_a_t +
            static_cast<int64_t>(slot) * kargs.stride_a_k);
        smem_route_base[local_m] = token * kargs.topk + slot;
        smem_weight[local_m] =
            (kargs.sorted_weights == nullptr) ? 1.0f : kargs.sorted_weights[row];
    }
    __syncthreads();

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    auto u_ga = opus_moe_stage2_layout_ga<T>(lane_id, wave_id_m, wave_id_n, BK);
    auto u_sa = opus_moe_stage2_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = opus_moe_stage2_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = opus_moe_stage2_layout_gb<T>(lane_id, wave_id_m, wave_id_n, BK);
    auto u_sb = opus_moe_stage2_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = opus_moe_stage2_layout_rb<T>(lane_id, wave_id_n);

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b[2];
    typename decltype(mma)::vtype_c v_c[2][2];
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);

    auto issue_a_half = [&](int buf, int half_m, int tile_k) {
        OPUS_LDS_ADDR D_A* smem_ptr =
            reinterpret_cast<OPUS_LDS_ADDR D_A*>(s_a[buf][half_m].ptr);
        auto g_offsets = layout_to_offsets<T::VEC_A>(u_ga);
        auto s_offsets = layout_to_offsets<T::VEC_A>(u_sa);
        using LT_A = layout_load_traits<decltype(u_ga), T::VEC_A>;
        for(index_t idx = 0; idx < LT_A::r_elem.value; ++idx)
        {
            const int logical = static_cast<int>(g_offsets[idx]);
            const int local_m = logical / BK;
            const int local_k = logical - local_m * BK;
            const int route_m = half_m * HALF_BM + local_m;
            const int k = tile_k * BK + local_k;
            const int a_base = smem_a_base[route_m];
            OPUS_LDS_ADDR D_A* dst = smem_ptr + s_offsets[idx];
            g_a.template async_load<T::VEC_A>(reinterpret_cast<void*>(dst),
                                              a_base + k,
                                              0,
                                              opus::number<T::CACHECTL_A>{});
        }
    };

    auto issue_b_half = [&](int buf, int half_n, int tile_k) {
        OPUS_LDS_ADDR D_B* smem_ptr =
            reinterpret_cast<OPUS_LDS_ADDR D_B*>(s_b[buf][half_n].ptr);
        auto g_offsets = layout_to_offsets<T::VEC_B>(u_gb);
        auto s_offsets = layout_to_offsets<T::VEC_B>(u_sb);
        using LT_B = layout_load_traits<decltype(u_gb), T::VEC_B>;
        for(index_t idx = 0; idx < LT_B::r_elem.value; ++idx)
        {
            const int logical = static_cast<int>(g_offsets[idx]);
            const int local_n = logical / BK;
            const int local_k = logical - local_n * BK;
            const int rel_col = half_n * HALF_BN + local_n;
            const int k = tile_k * BK + local_k;
            OPUS_LDS_ADDR D_B* dst = smem_ptr + s_offsets[idx];
            g_b.template async_load<T::VEC_B>(reinterpret_cast<void*>(dst),
                                              rel_col * kargs.stride_w_h + k,
                                              0,
                                              opus::number<T::CACHECTL_B>{});
        }
    };

    auto issue_tile = [&](int buf, int tile_k) {
        issue_b_half(buf, 0, tile_k);
        issue_a_half(buf, 0, tile_k);
        issue_b_half(buf, 1, tile_k);
        issue_a_half(buf, 1, tile_k);
    };

    auto wait_for_tile = [&]() {
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    };

    auto compute_tile = [&](int buf) {
        v_a = s_a[buf][0].template load<T::VEC_A>(u_ra);
        v_b[0] = s_b[buf][0].template load<T::VEC_B>(u_rb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);

        v_b[1] = s_b[buf][1].template load<T::VEC_B>(u_rb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
        __builtin_amdgcn_s_setprio(0);

        v_a = s_a[buf][1].template load<T::VEC_A>(u_ra);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
        v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
    };

    auto run_simple_pipeline = [&](int loops) {
        int cur = 0;
        int next = 1;
        issue_tile(cur, 0);
        wait_for_tile();
        if(loops > 1)
            issue_tile(next, 1);

        for(int tile = 0; tile < loops; ++tile)
        {
            compute_tile(cur);
            if(tile + 1 < loops)
            {
                wait_for_tile();
                cur ^= 1;
                next ^= 1;
                if(tile + 2 < loops)
                    issue_tile(next, tile + 2);
            }
        }
    };

    auto run_gemmstyle_pipeline = [&](int loops) {
        int tic = 0;
        int toc = 1;

        issue_tile(tic, 0);
        if(wave_id_m == 1)
            __builtin_amdgcn_s_barrier();

        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        issue_b_half(toc, 0, 1);
        issue_a_half(toc, 0, 1);
        issue_b_half(toc, 1, 1);

        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
        __builtin_amdgcn_s_barrier();

        v_b[0] = s_b[tic][0].template load<T::VEC_B>(u_rb);
        __builtin_amdgcn_s_barrier();

        for(int tile = 0; tile < loops - 2; tile += 2)
        {
            v_a = s_a[tic][0].template load<T::VEC_A>(u_ra);
            issue_a_half(toc, 1, tile + 1);
            s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = s_b[tic][1].template load<T::VEC_B>(u_rb);
            issue_b_half(tic, 0, tile + 2);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_a = s_a[tic][1].template load<T::VEC_A>(u_ra);
            issue_a_half(tic, 0, tile + 2);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            issue_b_half(tic, 1, tile + 2);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();
            v_b[0] = s_b[toc][0].template load<T::VEC_B>(u_rb);

            __builtin_amdgcn_s_setprio(1);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_a = s_a[toc][0].template load<T::VEC_A>(u_ra);
            issue_a_half(tic, 1, tile + 2);
            s_waitcnt_lgkmcnt(number<T::a_ds_read_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = s_b[toc][1].template load<T::VEC_B>(u_rb);
            issue_b_half(toc, 0, tile + 3);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_a = s_a[toc][1].template load<T::VEC_A>(u_ra);
            issue_a_half(toc, 0, tile + 3);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            issue_b_half(toc, 1, tile + 3);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();
            v_b[0] = s_b[tic][0].template load<T::VEC_B>(u_rb);

            __builtin_amdgcn_s_setprio(1);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        {
            int tile = loops - 2;

            v_a = s_a[tic][0].template load<T::VEC_A>(u_ra);
            issue_a_half(toc, 1, tile + 1);
            __builtin_amdgcn_s_barrier();
            s_waitcnt_lgkmcnt(0_I);

            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = s_b[tic][1].template load<T::VEC_B>(u_rb);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_a = s_a[tic][1].template load<T::VEC_A>(u_ra);
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
            v_b[0] = s_b[tic][0].template load<T::VEC_B>(u_rb);
            v_a = s_a[tic][0].template load<T::VEC_A>(u_ra);
            s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][0] = mma(v_a, v_b[0], v_c[0][0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_b[1] = s_b[tic][1].template load<T::VEC_B>(u_rb);
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[0][1] = mma(v_a, v_b[1], v_c[0][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            v_a = s_a[tic][1].template load<T::VEC_A>(u_ra);
            __builtin_amdgcn_s_barrier();

            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_setprio(1);
            v_c[1][0] = mma(v_a, v_b[0], v_c[1][0]);
            v_c[1][1] = mma(v_a, v_b[1], v_c[1][1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
    };

    const int loops = (kargs.inter_dim + BK - 1) / BK;
    if(loops >= 2 && (loops % 2) == 0)
        run_gemmstyle_pipeline(loops);
    else
        run_simple_pipeline(loops);

    if(wave_id_m == 0)
        __builtin_amdgcn_s_barrier();

    auto p_coord_c =
        opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_c_m = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(1_I, 0_I), p_coord_c);
    auto u_c_n = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(0_I, 1_I), p_coord_c);

    using LT_C = layout_load_traits<decltype(u_c_n), T::VEC_C>;
    constexpr auto issue_space_vec_c = LT_C::issue_space_vec;
    constexpr auto u_r_c = make_layout<-1>(issue_space_vec_c);
    static_assert(T::VEC_C == 4, "route-output store packs exactly four bf16 values");

    auto store_c = [&](auto& vc, int half_m, int half_n) {
        static_ford(issue_space_vec_c, [&](auto... ids) {
            constexpr index_t idx = u_r_c(ids...);
            const int local_m = half_m * HALF_BM + static_cast<int>(u_c_m(ids...));
            const int local_n_base = half_n * HALF_BN + static_cast<int>(u_c_n(ids...));
            const int col = col_base + local_n_base;
            const float weight = smem_weight[local_m];
            auto values = slice(vc,
                                number<idx * T::VEC_C>{},
                                number<(idx + 1) * T::VEC_C>{});
            const int route_row = smem_route_base[local_m];
            const hip_bfloat16 v0(static_cast<float>(values[0]) * weight);
            const hip_bfloat16 v1(static_cast<float>(values[1]) * weight);
            const hip_bfloat16 v2(static_cast<float>(values[2]) * weight);
            const hip_bfloat16 v3(static_cast<float>(values[3]) * weight);
            const uint64_t packed =
                static_cast<uint64_t>(static_cast<uint16_t>(v0.data)) |
                (static_cast<uint64_t>(static_cast<uint16_t>(v1.data)) << 16) |
                (static_cast<uint64_t>(static_cast<uint16_t>(v2.data)) << 32) |
                (static_cast<uint64_t>(static_cast<uint16_t>(v3.data)) << 48);
            *reinterpret_cast<uint64_t*>(
                kargs.route_out_bf16 +
                static_cast<int64_t>(route_row) * kargs.stride_route_out_t + col) =
                packed;
        });
    };

    store_c(v_c[0][0], 0, 0);
    store_c(v_c[0][1], 0, 1);
    store_c(v_c[1][0], 1, 0);
    store_c(v_c[1][1], 1, 1);
#endif
#endif
}
