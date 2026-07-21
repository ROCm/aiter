// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_pipeline_common_gfx950.cuh"

namespace opus_moe
{
namespace stage1_a8w4
{
namespace pipeline_pair_kwave
{

#ifdef __HIP_DEVICE_COMPILE__

using namespace pipeline_common;

template<typename T>
static __device__ void process_tile(
    const OpusMoeStage1A8W4Kargs& kargs,
    uint8_t* __restrict__ smem_scratch,
    int* __restrict__ smem_route)
{
    static_assert(!T::GATE_UP_GROUP_SPLIT,
                  "pair-kwave stage1 pipeline requires pair gate/up K-wave policy");
    static_assert(T::K_WAVE > 1,
                  "pair-kwave stage1 pipeline requires K-wave reduction");

    const int route_tile = static_cast<int>(opus::block_id_y());
    const int col_tile = static_cast<int>(opus::block_id_x());
    const int tid = static_cast<int>(opus::thread_id_x());
    const int wave_id =
        __builtin_amdgcn_readfirstlane(tid / opus::get_warp_size());
    const int lane_id = tid & (opus::get_warp_size() - 1);
    const int route_base = route_tile * T::B_M;
    const int out_col_base = col_tile * T::OUTPUT_COLS_PER_TILE;
    const int valid_rows = kargs.num_valid_ids[0];
    const int expert_id = kargs.sorted_expert_ids[route_tile];
    const int out_half = wave_id % T::KWAVE_BASE_WAVES;
    const int group_base = 0;
    const int stride_a = static_cast<int>(kargs.stride_hidden_t);
    const int k_steps = kargs.k_steps;
    const int scale_stride_n0 =
        scale_layout_stride_n0_stage1<T>(kargs.hidden_scale_cols);

    if(route_base >= valid_rows || expert_id < 0 ||
       expert_id >= kargs.num_experts)
        return;

    load_routes_stage1<T>(
        kargs, route_base, valid_rows, tid, smem_route);
    opus::sync_threads();

    auto g_a = opus::make_gmem(kargs.hidden_fp8);
    auto g_b = opus::make_gmem(kargs.w1_fp4);
    auto g_as = opus::make_gmem(kargs.hidden_scale_e8m0);
    auto g_bs = opus::make_gmem(kargs.w1_scale_e8m0);
    auto g_out =
        opus::make_gmem(reinterpret_cast<uint32_t*>(kargs.inter_states_fp8));
    auto g_out_scale = opus::make_gmem(kargs.inter_states_scale_e8m0);

    auto mma = opus::make_tiled_mma<typename T::D_MFMA_A,
                                    typename T::D_MFMA_B,
                                    typename T::D_ACC>(
        opus::seq<1, 1, 1>{},
        opus::seq<1, 1, 1>{},
        opus::seq<T::MMA_M, T::MMA_N, T::MMA_K>{});
    auto u_ga = make_layout_ga_stage1<T>(
        mma, lane_id, stride_a);
    auto u_gb = make_layout_gb_stage1<T>(mma, lane_id);
    auto u_c_m = make_layout_c_m_stage1<T>(mma, lane_id);
    auto u_c_n = make_layout_c_n_stage1<T>(mma, lane_id);

    typename decltype(mma)::vtype_c
        v_c[T::M_MFMA_PER_WAVE][T::ACC_SCALE_GROUPS_PER_TILE]{};

    const Tile tile{
        tid,
        wave_id,
        route_base,
        out_col_base,
        out_half,
        group_base};

    const int b_lane_byte = static_cast<int>(u_gb(0, 0, 0, 0));
    auto u_sf = make_layout_scale_word_stage1<T>(
        b_lane_byte, scale_stride_n0);

    bool a_route_valid[T::M_MFMA_PER_WAVE]{};
    int64_t a_payload_base[T::M_MFMA_PER_WAVE]{};
    int a_scale_base_word[T::M_SCALE_PACKS]{};
    #pragma unroll
    for(int mi = 0; mi < T::M_MFMA_PER_WAVE; ++mi)
    {
        const int a_lane_byte = a_lane_byte_stage1<T>(
            u_ga, mi, stride_a);
        const int local_m = a_lane_local_m_stage1(
            a_lane_byte, stride_a);
        const int k_byte = a_lane_byte - local_m * stride_a;
        const auto route = route_from_smem(local_m, smem_route);
        a_route_valid[mi] = route.valid;
        a_payload_base[mi] =
            a_payload_base_stage1(kargs, route, k_byte);
    }
    #pragma unroll
    for(int mp = 0; mp < T::M_SCALE_PACKS; ++mp)
        a_scale_base_word[mp] =
            a_scale_base_word_stage1<T>(u_sf, route_base, mp);

    int64_t b_group_payload_base[T::B_GROUPS_PER_WAVE]{};
    int b_scale_base_word[T::B_GROUPS_PER_WAVE]{};
    #pragma unroll
    for(int local_group = 0; local_group < T::B_GROUPS_PER_WAVE;
        ++local_group)
    {
        const int w1_n0 = w1_n0_stage1<T>(
            out_col_base, out_half, group_base, local_group);
        b_group_payload_base[local_group] =
            b_payload_base_stage1<T>(
                kargs, expert_id, w1_n0, b_lane_byte, k_steps);
        b_scale_base_word[local_group] =
            b_scale_base_word_stage1<T>(u_sf, kargs, expert_id, w1_n0);
    }

    using V_A = typename decltype(mma)::mfma_type::vtype_a;
    using V_B = typename decltype(mma)::mfma_type::vtype_b;

    const int mainloop_wave_k = tile.wave_id / T::KWAVE_BASE_WAVES;
    const int k_steps_per_wave = k_steps / T::K_WAVE;
    const int k_begin = mainloop_wave_k * k_steps_per_wave;
    const int k_end = k_begin + k_steps_per_wave;

    for(int k_pair = k_begin; k_pair < k_end;
        k_pair += T::SCALE_K_PACK)
    {
        int a_scale[T::M_SCALE_PACKS];
        load_sfa_frag_stage1<T>(
            g_as, a_scale_base_word, k_pair, a_scale);

        int b_scale[T::B_GROUPS_PER_WAVE];
        stage1_u32x4_t b_kk0[T::B_ITEMS_PER_WAVE];
        load_sfb_frag_stage1<T>(
            g_bs, b_scale_base_word, k_pair, b_scale);
        load_b_kpair_stage1<T>(
            g_b, b_group_payload_base, k_pair, k_steps, b_kk0);

        opus::static_for<T::SCALE_K_PACK>([&](auto kk_id) {
            constexpr int kk = kk_id.value;
            const int k_step = k_pair + kk;

            V_A ra[T::M_MFMA_PER_WAVE]{};
            opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi_id) {
                constexpr int mi = mi_id.value;
                if(!a_route_valid[mi])
                    return;

                ra[mi] = load_a_reg_stage1<V_A, T>(
                    g_a, a_payload_base[mi], k_step);
            });

            opus::static_for<T::B_ITEMS_PER_WAVE>([&](auto flat_id) {
                constexpr int flat_item = flat_id.value;
                constexpr int local_group =
                    flat_item / T::B_ITEMS_PER_GROUP;
                constexpr int item =
                    flat_item - local_group * T::B_ITEMS_PER_GROUP;
                stage1_u32x4_t b_raw = b_kk0[flat_item];
                if constexpr(kk != 0)
                {
                    b_raw = load_b_raw_stage1<T>(
                        g_b, b_group_payload_base[local_group],
                        item, k_step, k_steps);
                }
                const V_B rb = make_b_reg<V_B>(b_raw);

                constexpr int acc_group = item;
                const int sfb = b_scale[local_group];
                const auto selector_b_id = selector_b<T, item, kk>();

                opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi_id) {
                    constexpr int mi = mi_id.value;
                    const int sfa =
                        select_sfa_stage1<T, mi>(
                            a_route_valid, a_scale);

                    with_selector_a<T, mi, kk>(
                        route_base, [&](auto selector_a_id) {
                            v_c[mi][acc_group] = mma(
                                ra[mi], rb, v_c[mi][acc_group],
                                sfa, sfb,
                                selector_a_id, selector_b_id);
                        });
                });
            });
        });
    }

    reduce_single_group_kwave<T>(tile.wave_id, lane_id,
                                 smem_scratch, v_c);

    quant_epilogue<T>(
        kargs, g_out, g_out_scale, tile, expert_id, u_c_m, u_c_n,
        v_c, smem_route,
        reinterpret_cast<float*>(smem_scratch));
}

#endif // __HIP_DEVICE_COMPILE__

template<typename T>
__global__ __launch_bounds__(T::BLOCK_SIZE, T::MIN_BLOCKS_PER_CU) void
opus_moe_stage1_a8w4_kernel_pair_kwave_gfx950(OpusMoeStage1A8W4Kargs kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    __shared__ __align__(T::BYTES_PER_VEC) uint8_t
        smem_scratch[T::SHARED_SCRATCH_BYTES];
    __shared__ int smem_route[T::B_M];

    process_tile<T>(kargs, smem_scratch, smem_route);
#else
    (void)kargs;
#endif
}

} // namespace pipeline_pair_kwave
} // namespace stage1_a8w4
} // namespace opus_moe
