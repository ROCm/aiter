// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh>

// Equal-routes stage-2 backward specialized for the Sonic-parity shape family.
// The mainloop is the mature 192x256x64, eight-wave mono GEMM, but its epilogue
// consumes the pre-activation and writes standard SwiGLU gradients directly:
//
//   dh       = dy @ W2
//   dgate    = dh * up * silu'(gate)
//   dup      = dh * silu(gate)
//
// This avoids materializing/reloading dh and removes a separate activation
// kernel.  The accumulator is rounded to BF16 before applying the Jacobian, so
// the numerical contract matches the former GEMM(BF16 store) + act-bwd path.
struct opus_moe_dgrad_swiglu_kargs
{
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    const void* __restrict__ ptr_act_input;
    void* __restrict__ ptr_dact;
    void* __restrict__ ptr_dscore_partials;
    const int32_t* __restrict__ route_to_flat;
    const int32_t* __restrict__ expert_offsets;
    const int32_t* __restrict__ tile_offsets;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_act_input;
    int stride_dact;
    int stride_a_batch;
    int stride_b_batch;
    int stride_act_input_batch;
    int stride_dact_batch;
    int stride_dscore;
    int ragged;
    int compact_tiles;
    int num_tiles;
};

__device__ __forceinline__ void opus_moe_dgrad_tile_map(
    int wgid,
    int num_tiles_m,
    int num_tiles_n,
    int& tile_m,
    int& tile_n)
{
    tile_m = wgid % num_tiles_m;
    tile_n = wgid / num_tiles_m;
    constexpr int GROUP_M = 4;
    constexpr int GROUP_N = 2;
    const int grouped_tiles_m = (num_tiles_m / GROUP_M) * GROUP_M;
    if(grouped_tiles_m > 0 && (num_tiles_n % GROUP_N) == 0)
    {
        const int grouped_blocks = grouped_tiles_m * num_tiles_n;
        if(wgid < grouped_blocks)
        {
            const int group_id = wgid / (GROUP_M * GROUP_N);
            const int in_group = wgid % (GROUP_M * GROUP_N);
            const int groups_m = grouped_tiles_m / GROUP_M;
            tile_m = (group_id % groups_m) * GROUP_M + in_group % GROUP_M;
            tile_n = (group_id / groups_m) * GROUP_N + in_group / GROUP_M;
        }
        else
        {
            const int remainder_m = num_tiles_m - grouped_tiles_m;
            const int remainder_wgid = wgid - grouped_blocks;
            tile_m = grouped_tiles_m + remainder_wgid % remainder_m;
            tile_n = remainder_wgid / remainder_m;
        }
    }
}

template<typename UserTraits,
         bool WRITE_DSCORE = false,
         bool APPLY_SWIGLU = true,
         bool RAGGED = false,
         bool COMPACT_TILES = false,
         int FIXED_K = 0,
         bool TARGET_EXACT = false,
         bool PADDED_GRID = false,
         bool FLAT_DSCORE = false>
__global__ __launch_bounds__(UserTraits::BLOCK_SIZE, 2)
void opus_moe_dgrad_swiglu_kernel_gfx950(opus_moe_dgrad_swiglu_kargs kargs)
{
#if defined(__gfx950__)
    using namespace opus;
    using namespace opus_mono_tile_gfx950;
    using opus::operator""_I;
    using T = kernel_traits<opus::remove_cvref_t<UserTraits>>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    constexpr int TARGET_N = APPLY_SWIGLU ? 1024 : 2048;
    const int problem_n = TARGET_EXACT ? TARGET_N : kargs.n;
    const int problem_batch = TARGET_EXACT ? 64 : kargs.batch;
    const int stride_a = TARGET_EXACT ? 2048 : kargs.stride_a;
    const int stride_b = TARGET_EXACT ? 2048 : kargs.stride_b;
    const int stride_act_input =
        TARGET_EXACT ? 2048 : kargs.stride_act_input;
    const int stride_dact = TARGET_EXACT ? 2048 : kargs.stride_dact;
    const int stride_b_batch = TARGET_EXACT
        ? TARGET_N * 2048
        : kargs.stride_b_batch;
    const int stride_dscore = TARGET_EXACT
        ? TARGET_N / T::B_N
        : kargs.stride_dscore;

    const int wgid = block_id_x();
    const int num_tiles_n = problem_n / T::B_N;
    int batch_id = block_id_z();
    int local_wgid = wgid;
    int num_tiles_m = (kargs.m + T::B_M - 1) / T::B_M;
    if constexpr(COMPACT_TILES)
    {
        if constexpr(PADDED_GRID)
        {
            const int total_tiles_m = kargs.tile_offsets[kargs.batch];
            if(wgid >= total_tiles_m * num_tiles_n)
                return;
        }
        if constexpr(TARGET_EXACT)
        {
            // Prefix sums and one descriptor per 192-row M tile share the
            // existing tile_offsets allocation.  Exact N turns both div/mod
            // operations into shifts/masks; one scalar load replaces the
            // six-step expert-prefix search in every CTA.
            const int global_tile_m = wgid / num_tiles_n;
            const uint32_t tile_desc = static_cast<uint32_t>(
                kargs.tile_offsets[problem_batch + 1 + global_tile_m]);
            batch_id = tile_desc & 0xffu;
            const int local_tile_m = (tile_desc >> 8) & 0x7ffu;
            num_tiles_m = tile_desc >> 19;
            local_wgid =
                local_tile_m * num_tiles_n + wgid % num_tiles_n;
        }
        else
        {
            // Generic compact routing retains the prefix-search fallback.
            int lo = 0;
            int hi = kargs.batch;
            while(lo + 1 < hi)
            {
                const int mid = (lo + hi) / 2;
                if(wgid >= kargs.tile_offsets[mid] * num_tiles_n)
                    lo = mid;
                else
                    hi = mid;
            }
            batch_id = lo;
            const int first_tile = kargs.tile_offsets[batch_id];
            num_tiles_m = kargs.tile_offsets[batch_id + 1] - first_tile;
            local_wgid = wgid - first_tile * num_tiles_n;
        }
    }
    int tile_m;
    int tile_n;
    if constexpr(TARGET_EXACT)
    {
        // Natural and balanced target routing has 20--23 M tiles per
        // expert, hence exactly five complete four-M groups.  Specialize the
        // hot grouped region so division/modulo by dynamic groups_m becomes
        // compile-time division by five; unusual skew keeps the generic map.
        if(num_tiles_m >= 20 && num_tiles_m < 24)
        {
            constexpr int GROUP_M = 4;
            constexpr int GROUP_N = 2;
            constexpr int GROUPS_M = 5;
            constexpr int GROUPED_TILES_M = GROUP_M * GROUPS_M;
            const int grouped_blocks = GROUPED_TILES_M * num_tiles_n;
            if(local_wgid < grouped_blocks)
            {
                const int group_id = local_wgid / (GROUP_M * GROUP_N);
                const int in_group = local_wgid % (GROUP_M * GROUP_N);
                tile_m = (group_id % GROUPS_M) * GROUP_M +
                    in_group % GROUP_M;
                tile_n = (group_id / GROUPS_M) * GROUP_N +
                    in_group / GROUP_M;
            }
            else
            {
                const int remainder_m = num_tiles_m - GROUPED_TILES_M;
                const int remainder_wgid = local_wgid - grouped_blocks;
                tile_m = GROUPED_TILES_M + remainder_wgid % remainder_m;
                tile_n = remainder_wgid / remainder_m;
            }
        }
        else
            opus_moe_dgrad_tile_map(
                local_wgid, num_tiles_m, num_tiles_n, tile_m, tile_n);
    }
    else
        opus_moe_dgrad_tile_map(
            local_wgid, num_tiles_m, num_tiles_n, tile_m, tile_n);
    const int row = tile_m * T::B_M;
    const int col = tile_n * T::B_N;

    int batch_row_start = 0;
    int batch_m = kargs.m;
    if constexpr(RAGGED)
    {
        batch_row_start = kargs.expert_offsets[batch_id];
        batch_m = kargs.expert_offsets[batch_id + 1] - batch_row_start;
        if(row >= batch_m)
            return;
    }
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / get_warp_size());
    const int lane_id = thread_id_x() % get_warp_size();

    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a) +
            (RAGGED ? batch_row_start * stride_a
                    : batch_id * kargs.stride_a_batch) +
            row * stride_a,
        (batch_m - row) * stride_a * sizeof(D_A));
    auto g_b = make_gmem(
        reinterpret_cast<const D_B*>(kargs.ptr_b) +
            batch_id * stride_b_batch + col * stride_b,
        (problem_n - col) * stride_b * sizeof(D_B));
    auto g_act_input = make_gmem(
        reinterpret_cast<const D_C*>(kargs.ptr_act_input) +
            (RAGGED ? batch_row_start * stride_act_input
                    : batch_id * kargs.stride_act_input_batch) +
            row * stride_act_input,
        (batch_m - row) * stride_act_input * sizeof(D_C));
    auto g_dact = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_dact) +
            (RAGGED ? batch_row_start * stride_dact
                    : batch_id * kargs.stride_dact_batch) +
            row * stride_dact,
        (batch_m - row) * stride_dact * sizeof(D_C));

    const int wave_id_m = wave_id / T::T_N;
    const int wave_id_n = wave_id % T::T_N;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, stride_a);
    auto u_sa = make_layout_sa<T>(lane_id, wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, stride_b);
    auto u_sb = make_layout_sb<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);

    constexpr int smem_a_byte =
        T::smem_m_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_A);
    __shared__ char smem_a[smem_a_byte * 2];
    smem<D_A> s_a[2] = {
        make_smem(reinterpret_cast<D_A*>(smem_a)),
        make_smem(reinterpret_cast<D_A*>(smem_a + smem_a_byte))};
    constexpr int smem_b_byte =
        T::smem_n_rep * (T::smem_linear_wave + T::smem_padding) * sizeof(D_B);
    __shared__ char smem_b[smem_b_byte * 3];
    smem<D_B> sb_r0 = make_smem(reinterpret_cast<D_B*>(smem_b));
    smem<D_B> sb_r1 = make_smem(reinterpret_cast<D_B*>(smem_b + smem_b_byte));
    smem<D_B> sb_w = make_smem(reinterpret_cast<D_B*>(smem_b + 2 * smem_b_byte));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<1_I, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);

    auto k_offset = [&](int tile_k) { return tile_k * T::B_K; };
    const int loops = FIXED_K > 0
        ? FIXED_K / T::B_K
        : (kargs.k + T::B_K - 1) / T::B_K;
    int tic = 0;
    int toc = 1;

    async_load<T::VEC_A>(g_a, s_a[tic].ptr, u_ga, u_sa, k_offset(0));
    async_load<T::VEC_B>(g_b, sb_r0.ptr, u_gb, u_sb, k_offset(0));
    __builtin_amdgcn_sched_barrier(0);
    async_load<T::VEC_B>(g_b, sb_r1.ptr, u_gb, u_sb, k_offset(1));
    s_waitcnt_vmcnt(number<T::b_buffer_load_insts>{});
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    if(wave_id_m == 1)
        __builtin_amdgcn_s_barrier();

    if constexpr(TARGET_EXACT)
        __builtin_amdgcn_s_setprio(1);
#pragma unroll 4
    for(int tile = 0; tile < loops - 2; tile += 2)
    {
        async_load<T::VEC_A>(g_a, s_a[toc].ptr, u_ga, u_sa, k_offset(tile + 1));
        v_a = load<T::VEC_A>(s_a[tic], u_ra);
        v_b = load<T::VEC_B>(sb_r0, u_rb);
        async_load<T::VEC_B>(g_b, sb_w.ptr, u_gb, u_sb, k_offset(tile + 2));
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(!TARGET_EXACT)
            __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts>{});
        if constexpr(!TARGET_EXACT)
            __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        async_load<T::VEC_A>(g_a, s_a[tic].ptr, u_ga, u_sa, k_offset(tile + 2));
        v_a = load<T::VEC_A>(s_a[toc], u_ra);
        v_b = load<T::VEC_B>(sb_r1, u_rb);
        async_load<T::VEC_B>(g_b, sb_r0.ptr, u_gb, u_sb, k_offset(tile + 3));
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + T::b_buffer_load_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(!TARGET_EXACT)
            __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::b_buffer_load_insts>{});
        if constexpr(!TARGET_EXACT)
            __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        smem<D_B> tmp = sb_w;
        sb_w = sb_r1;
        sb_r1 = sb_r0;
        sb_r0 = tmp;
    }

    {
        const int tile = loops - 2;
        v_a = load<T::VEC_A>(s_a[tic], u_ra);
        v_b = load<T::VEC_B>(sb_r0, u_rb);
        async_load<T::VEC_A>(g_a, s_a[toc].ptr, u_ga, u_sa, k_offset(tile + 1));
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts>{});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(!TARGET_EXACT)
            __builtin_amdgcn_s_setprio(1);
        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(0_I);
        if constexpr(!TARGET_EXACT)
            __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    {
        v_a = load<T::VEC_A>(s_a[toc], u_ra);
        v_b = load<T::VEC_B>(sb_r1, u_rb);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        v_c = mma(v_a, v_b, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if(wave_id_m == 0)
        __builtin_amdgcn_s_barrier();

    auto u_gc = make_layout_gc<T>(lane_id, 0, wave_id_n, stride_dact);
    auto v_dh = cast<D_C>(v_c);
    static_assert(sizeof(D_C) * 8 % sizeof(u32_t) == 0);
    constexpr int u32_per_chunk = sizeof(D_C) * 8 / sizeof(u32_t);
    constexpr int num_chunks = sizeof(v_dh) / (sizeof(u32_t) * u32_per_chunk);
    auto* p_u32 = reinterpret_cast<u32_t*>(&v_dh);
    static_for<num_chunks>([&](auto c) {
        auto* p = p_u32 + c.value * u32_per_chunk;
        auto r0 = __builtin_amdgcn_permlane16_swap(p[0], p[2], false, true);
        auto r1 = __builtin_amdgcn_permlane16_swap(p[1], p[3], false, true);
        p[0] = r0[0];
        p[2] = r0[1];
        p[1] = r1[0];
        p[3] = r1[1];
    });

    using GCLoad = layout_load_traits<decltype(u_gc), T::VEC_C>;
    constexpr auto r_elem = GCLoad::r_elem;
    auto offsets = layout_to_offsets<T::VEC_C>(u_gc);
    const int output_offset =
        wave_id_m * (T::B_M / T::T_M) * stride_dact + col;
    if constexpr(!APPLY_SWIGLU)
    {
        store<T::VEC_C>(g_dact, v_dh, u_gc, output_offset);
        return;
    }
    constexpr int rows_per_lane =
        (T::B_M / T::T_M) / T::W_M;
    constexpr int vecs_per_row = r_elem.value / rows_per_lane;
    // Reduce one route row at a time so six route-dot accumulators do not stay
    // live across the whole epilogue. Dedicated score LDS permits early writes
    // without aliasing a mainloop A buffer still being consumed by another wave.
    __shared__ float score_smem[
        WRITE_DSCORE ? T::B_M * T::T_N : 1];
    auto apply_swiglu = [&](auto i, const auto& gate, const auto& up) {
        vector_t<D_C, T::VEC_C> dgate;
        vector_t<D_C, T::VEC_C> dup;
        float vec_dot = 0.0f;
        auto pk_mul = [](opus::fp32x2_t a, opus::fp32x2_t b) {
            opus::fp32x2_t out;
            asm volatile(
                "v_pk_mul_f32 %0, %1, %2"
                : "=v"(out)
                : "v"(a), "v"(b));
            return out;
        };
        auto pk_fma = [](opus::fp32x2_t a,
                         opus::fp32x2_t b,
                         opus::fp32x2_t c) {
            opus::fp32x2_t out;
            asm volatile(
                "v_pk_fma_f32 %0, %1, %2, %3"
                : "=v"(out)
                : "v"(a), "v"(b), "v"(c));
            return out;
        };
        static_assert((T::VEC_C % 2) == 0);
#pragma unroll
        for(index_t j = 0; j < T::VEC_C; j += 2)
        {
            const opus::fp32x2_t dh = {
                static_cast<float>(v_dh[i.value * T::VEC_C + j]),
                static_cast<float>(v_dh[i.value * T::VEC_C + j + 1])};
            const opus::fp32x2_t g = {
                static_cast<float>(gate[j]),
                static_cast<float>(gate[j + 1])};
            const opus::fp32x2_t u = {
                static_cast<float>(up[j]),
                static_cast<float>(up[j + 1])};
            constexpr float log2e = 1.4426950408889634f;
            const opus::fp32x2_t sig = {
                __builtin_amdgcn_rcpf(
                    1.0f + __builtin_amdgcn_exp2f(-g[0] * log2e)),
                __builtin_amdgcn_rcpf(
                    1.0f + __builtin_amdgcn_exp2f(-g[1] * log2e))};
            // Pair the algebraically shared terms for dSwiGLU and dscore:
            // dup=dh*g*sig, dot=dup*u, dgate=dh*sig*u+dot*(1-sig).
            const opus::fp32x2_t base = pk_mul(dh, sig);
            const opus::fp32x2_t dup_f = pk_mul(base, g);
            const opus::fp32x2_t dot = pk_mul(dup_f, u);
            const opus::fp32x2_t base_u = pk_mul(base, u);
            const opus::fp32x2_t one_minus_sig = {
                1.0f - sig[0], 1.0f - sig[1]};
            const opus::fp32x2_t dgate_f =
                pk_fma(dot, one_minus_sig, base_u);
            dgate[j] = static_cast<D_C>(dgate_f[0]);
            dgate[j + 1] = static_cast<D_C>(dgate_f[1]);
            dup[j] = static_cast<D_C>(dup_f[0]);
            dup[j + 1] = static_cast<D_C>(dup_f[1]);
            if constexpr(WRITE_DSCORE)
                vec_dot += dot[0] + dot[1];
        }
        g_dact.template store<T::VEC_C>(
            dgate, offsets[i.value], output_offset);
        g_dact.template store<T::VEC_C>(
            dup, offsets[i.value], output_offset + problem_n);
        return vec_dot;
    };
    if constexpr(WRITE_DSCORE)
    {
        static_assert(vecs_per_row == 2);
        static_for<rows_per_lane>([&](auto r) {
            constexpr auto i0 = number<r.value * vecs_per_row>{};
            constexpr auto i1 = number<r.value * vecs_per_row + 1>{};
            auto gate0 = g_act_input.template load<T::VEC_C>(
                offsets[i0.value], output_offset);
            auto up0 = g_act_input.template load<T::VEC_C>(
                offsets[i0.value], output_offset + problem_n);
            auto gate1 = g_act_input.template load<T::VEC_C>(
                offsets[i1.value], output_offset);
            float route_dot = apply_swiglu(i0, gate0, up0);
            auto up1 = g_act_input.template load<T::VEC_C>(
                offsets[i1.value], output_offset + problem_n);
            route_dot += apply_swiglu(i1, gate1, up1);
            route_dot += __shfl_down(route_dot, 16, 64);
            route_dot += __shfl_down(route_dot, 32, 64);
            if(lane_id < T::W_M)
            {
                const int local_row =
                    wave_id_m * (T::B_M / T::T_M) +
                    r.value * T::W_M + lane_id;
                score_smem[wave_id_n * T::B_M + local_row] = route_dot;
            }
        });
        __syncthreads();
        constexpr int SCORE_REDUCE_WAVES = 4;
        constexpr int SCORE_ROWS_PER_WAVE =
            T::B_M / SCORE_REDUCE_WAVES;
        if(wave_id < SCORE_REDUCE_WAVES &&
           lane_id < SCORE_ROWS_PER_WAVE)
        {
            const int local_row =
                wave_id * SCORE_ROWS_PER_WAVE + lane_id;
            const int route_row = row + local_row;
            if(route_row < batch_m)
            {
                float partial = 0.0f;
#pragma unroll
                for(int wn = 0; wn < T::T_N; ++wn)
                    partial += score_smem[wn * T::B_M + local_row];
                const int n_tile = col / T::B_N;
                const int compact_row = RAGGED
                    ? batch_row_start + route_row
                    : batch_id * kargs.m + route_row;
                const int score_row = FLAT_DSCORE
                    ? kargs.route_to_flat[compact_row]
                    : compact_row;
                reinterpret_cast<float*>(kargs.ptr_dscore_partials)[
                    score_row * stride_dscore +
                    n_tile] = partial;
            }
        }
    }
    else
    {
        static_for<r_elem.value>([&](auto i) {
            auto gate = g_act_input.template load<T::VEC_C>(
                offsets[i.value], output_offset);
            auto up = g_act_input.template load<T::VEC_C>(
                offsets[i.value], output_offset + problem_n);
            (void)apply_swiglu(i, gate, up);
        });
    }
#else
    (void)kargs;
#endif
}

using opus_moe_dgrad_swiglu_traits_gfx950 =
    opus_gemm_a16w16_mono_tile_traits_gfx950<
        512,
        opus::seq<192, 256, 64>,
        opus::tuple<opus::bf16_t, opus::bf16_t, opus::bf16_t, opus::fp32_t>,
        opus::seq<8, 8, 8>>;

inline void opus_moe_dgrad_swiglu_launch_gfx950(
    const opus_moe_dgrad_swiglu_kargs& kargs,
    hipStream_t stream)
{
    const int num_tiles_m = (kargs.m + 191) / 192;
    const int num_tiles_n = kargs.n / 256;
    dim3 grid(num_tiles_m * num_tiles_n, 1, kargs.batch);
    dim3 block(512);
    opus_moe_dgrad_swiglu_kernel_gfx950<opus_moe_dgrad_swiglu_traits_gfx950>
        <<<grid, block, 0, stream>>>(kargs);
}

inline void opus_moe_dgrad_swiglu_dscore_launch_gfx950(
    const opus_moe_dgrad_swiglu_kargs& kargs,
    hipStream_t stream)
{
    const int num_tiles_m = (kargs.m + 191) / 192;
    const int num_tiles_n = kargs.n / 256;
    dim3 grid(num_tiles_m * num_tiles_n, 1, kargs.batch);
    dim3 block(512);
    opus_moe_dgrad_swiglu_kernel_gfx950<
        opus_moe_dgrad_swiglu_traits_gfx950, true>
        <<<grid, block, 0, stream>>>(kargs);
}

inline void opus_moe_dgrad_swiglu_dscore_ragged_launch_gfx950(
    const opus_moe_dgrad_swiglu_kargs& kargs,
    hipStream_t stream)
{
    const int num_tiles_n = kargs.n / 256;
    dim3 grid(kargs.num_tiles * num_tiles_n, 1, 1);
    dim3 block(512);
    const bool target_exact =
        (kargs.compact_tiles == 2 || kargs.compact_tiles == 4) &&
        kargs.batch == 64 &&
        kargs.n == 1024 && kargs.k == 2048 &&
        kargs.stride_a == 2048 && kargs.stride_b == 2048 &&
        kargs.stride_act_input == 2048 && kargs.stride_dact == 2048 &&
        kargs.stride_b_batch == 1024 * 2048 && kargs.stride_dscore == 4;
    if(target_exact && kargs.compact_tiles == 4)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            true, true, true, true, 2048, true, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.compact_tiles == 4)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            true, true, true, true, 0, false, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.compact_tiles == 3)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            true, true, true, true, 2048, false, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(target_exact)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            true, true, true, true, 2048, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.k == 2048)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            true, true, true, true, 2048>
            <<<grid, block, 0, stream>>>(kargs);
    else
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            true, true, true, true>
            <<<grid, block, 0, stream>>>(kargs);
}

inline void opus_moe_dgrad_plain_ragged_launch_gfx950(
    const opus_moe_dgrad_swiglu_kargs& kargs,
    hipStream_t stream)
{
    const int num_tiles_n = kargs.n / 256;
    dim3 grid(kargs.num_tiles * num_tiles_n, 1, 1);
    dim3 block(512);
    const bool target_exact =
        (kargs.compact_tiles == 2 || kargs.compact_tiles == 4) &&
        kargs.batch == 64 &&
        kargs.n == 2048 && kargs.k == 2048 &&
        kargs.stride_a == 2048 && kargs.stride_b == 2048 &&
        kargs.stride_dact == 2048 &&
        kargs.stride_b_batch == 2048 * 2048;
    if(target_exact && kargs.compact_tiles == 4)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 2048, true, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.compact_tiles == 4 && kargs.k == 2048)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 2048, false, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.compact_tiles == 4)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 0, false, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.compact_tiles == 3 && kargs.k == 2048)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 2048, false, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.compact_tiles == 3)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 0, false, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(target_exact)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 2048, true>
            <<<grid, block, 0, stream>>>(kargs);
    else if(kargs.k == 2048)
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true, 2048>
            <<<grid, block, 0, stream>>>(kargs);
    else
        opus_moe_dgrad_swiglu_kernel_gfx950<
            opus_moe_dgrad_swiglu_traits_gfx950,
            false, false, true, true>
            <<<grid, block, 0, stream>>>(kargs);
}

// The exact full-MoE variant writes the small dscore partial tensor in original
// (token, top-k rank) order.  The final router tail then avoids reverse-map
// loads while the 1-GiB route gradient retains its efficient grouped stores.
inline void opus_moe_dgrad_swiglu_dscore_ragged_flat_launch_gfx950(
    const opus_moe_dgrad_swiglu_kargs& kargs,
    hipStream_t stream)
{
    const int num_tiles_n = kargs.n / 256;
    dim3 grid(kargs.num_tiles * num_tiles_n, 1, 1);
    dim3 block(512);
    opus_moe_dgrad_swiglu_kernel_gfx950<
        opus_moe_dgrad_swiglu_traits_gfx950,
        true, true, true, true, 2048, true, true, true>
        <<<grid, block, 0, stream>>>(kargs);
}
