// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_pipeline_common_gfx950.cuh"

namespace opus_moe
{
namespace stage1_a8w4
{
namespace pipeline_group_split
{

#ifdef __HIP_DEVICE_COMPILE__

using namespace pipeline_common;

// M-major tile remap keeping contiguous work on one of 8 XCDs (flydsl xcd_remap), for HBM channel balance.
template<typename T>
__device__ __forceinline__ void xcd_remap_tile(int& route_tile, int& col_tile)
{
    constexpr int num_xcds = 8;
    constexpr int group = T::XCD_SWIZZLE;
    const int n_col = static_cast<int>(gridDim.x);
    const int n_route = static_cast<int>(gridDim.y);
    const int total = n_col * n_route;
    const int per_xcd = total / num_xcds;
    const int rem = total % num_xcds;
    const int linear = route_tile * n_col + col_tile;
    const int xcd = linear % num_xcds;
    const int wgid = xcd * per_xcd + (xcd < rem ? xcd : rem) + linear / num_xcds;
    const int per_group = group * n_col;
    const int first_route = (wgid / per_group) * group;
    const int group_rows = n_route - first_route < group ? n_route - first_route : group;
    const int in_group = wgid % per_group;
    route_tile = first_route + in_group % group_rows;
    col_tile = in_group / group_rows;
}

template<typename T>
static __device__ void process_tile(
    const OpusMoeStage1A8W4Kargs& kargs,
    uint8_t* __restrict__ smem_scratch,
    int* __restrict__ smem_route)
{
    static_assert(T::GATE_UP_GROUP_SPLIT,
                  "group-split stage1 pipeline requires group split policy");
    static_assert(T::K_WAVE == 1,
                  "group-split stage1 pipeline does not split K across waves");

    int route_tile = static_cast<int>(opus::block_id_y());
    int col_tile = static_cast<int>(opus::block_id_x());
    if constexpr(T::XCD_SWIZZLE > 0)
        xcd_remap_tile<T>(route_tile, col_tile);
    const int tid = static_cast<int>(opus::thread_id_x());
    const int wave_id =
        __builtin_amdgcn_readfirstlane(tid / opus::get_warp_size());
    const int lane_id = tid & (opus::get_warp_size() - 1);
    const int route_base = route_tile * T::B_M;
    const int out_col_base = col_tile * T::OUTPUT_COLS_PER_TILE;
    const int valid_rows = kargs.num_valid_ids[0];
    const int expert_id = kargs.sorted_expert_ids[route_tile];
    const int out_half = wave_id & 1;
    const int group_base = (wave_id / 2) * T::B_GROUPS_PER_WAVE;
    const int wave_id_m = 0;
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

    auto s_a = opus::make_smem(reinterpret_cast<uint8_t*>(smem_scratch));

    auto mma = opus::make_tiled_mma<typename T::D_MFMA_A,
                                    typename T::D_MFMA_B,
                                    typename T::D_ACC>(
        opus::seq<1, 1, 1>{},
        opus::seq<1, 1, 1>{},
        opus::seq<T::MMA_M, T::MMA_N, T::MMA_K>{});
    auto u_ga = make_layout_ga_stage1<T>(
        mma, lane_id, stride_a);
    auto u_sa = make_layout_sa_stage1<T>(lane_id, wave_id_m);
    auto u_ra = make_layout_ra_stage1<T>(lane_id);
    auto u_gb = make_layout_gb_stage1<T>(mma, lane_id);
    auto u_c_m = make_layout_c_m_stage1<T>(mma, lane_id);
    auto u_c_n = make_layout_c_n_stage1<T>(mma, lane_id);

    typename decltype(mma)::vtype_c
        v_c[T::M_MFMA_PER_WAVE][T::ACC_SCALE_GROUPS_PER_TILE]{};
    using V_A = typename decltype(mma)::mfma_type::vtype_a;
    using V_B = typename decltype(mma)::mfma_type::vtype_b;

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
    int64_t b_group_payload_base[T::B_GROUPS_PER_WAVE]{};
    int b_scale_base_word[T::B_GROUPS_PER_WAVE]{};

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

    stage_a_reg_kpair_to_lds<T>(
        g_a, s_a, u_sa, tile, 0, a_payload_base);
    opus::s_waitcnt_vmcnt(opus::number<0>{});
    opus::sync_threads();

    for(int k_pair = 0; k_pair < k_steps;
        k_pair += T::SCALE_K_PACK)
    {
        const int next_k_pair = k_pair + T::SCALE_K_PACK;

        int a_scale[T::M_SCALE_PACKS];
        load_sfa_frag_stage1<T>(
            g_as, a_scale_base_word, k_pair, a_scale);

        int b_scale[T::B_GROUPS_PER_WAVE];
        stage1_u32x4_t
            b_kpair[T::SCALE_K_PACK][T::B_ITEMS_PER_WAVE];
        load_sfb_frag_stage1<T>(
            g_bs, b_scale_base_word, k_pair, b_scale);
        load_b_full_kpair_stage1<T>(
            g_b, b_group_payload_base, k_pair, k_steps, b_kpair);

        if(next_k_pair < k_steps)
            stage_a_reg_kpair_to_lds<T>(
                g_a, s_a, u_sa, tile, next_k_pair, a_payload_base);

        opus::static_for<T::SCALE_K_PACK>([&](auto kk_id) {
            constexpr int kk = kk_id.value;
            const int k_step = k_pair + kk;

            V_A ra[T::M_MFMA_PER_WAVE]{};
            opus::static_for<T::M_MFMA_PER_WAVE>([&](auto mi_id) {
                constexpr int mi = mi_id.value;
                ra[mi] = load_a_reg_lds_stage1<V_A, T>(
                    s_a, u_ra, k_step, mi);
            });

            opus::static_for<T::B_ITEMS_PER_WAVE>([&](auto flat_id) {
                constexpr int flat_item = flat_id.value;
                constexpr int local_group =
                    flat_item / T::B_ITEMS_PER_GROUP;
                constexpr int item =
                    flat_item - local_group * T::B_ITEMS_PER_GROUP;

                const stage1_u32x4_t b_raw = b_kpair[kk][flat_item];
                const V_B rb = make_b_reg<V_B>(b_raw);

                constexpr int acc_group =
                    local_group + item * T::B_GROUPS_PER_WAVE;
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

        if(next_k_pair < k_steps)
        {
            opus::s_waitcnt_vmcnt(opus::number<0>{});
            opus::sync_threads();
        }
    }

    if constexpr(!T::DEDICATED_EPILOGUE_SCRATCH)
    {
        // Protect the final A-register LDS buffer before epilogue reuses scratch.
        opus::sync_threads();
    }

    quant_epilogue<T>(
        kargs, g_out, g_out_scale, tile, expert_id, u_c_m, u_c_n,
        v_c, smem_route,
        reinterpret_cast<float*>(smem_scratch + T::EPILOGUE_SCRATCH_OFFSET));
}

#endif // __HIP_DEVICE_COMPILE__

template<typename T>
__global__ __launch_bounds__(T::BLOCK_SIZE, T::MIN_BLOCKS_PER_CU) void
opus_moe_stage1_a8w4_kernel_group_split_gfx950(OpusMoeStage1A8W4Kargs kargs)
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

} // namespace pipeline_group_split
} // namespace stage1_a8w4
} // namespace opus_moe
