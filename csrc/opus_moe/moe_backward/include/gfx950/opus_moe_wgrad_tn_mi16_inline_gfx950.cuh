// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Experimental exact-shape route-major wgrad microkernel.  The complete K64
// Direct-to-LDS/local-read/permute/MI16 schedule lives in one inline-asm block
// so clang cannot fragment the producer/consumer ordering with address code.
#pragma once

template<int GPR>
__device__ inline uint32_t opus_wgtn_mi16_inline_read_acc()
{
    uint32_t value;
    asm volatile("v_accvgpr_read_b32 %0, a[%1]"
                 : "=v"(value)
                 : "n"(GPR));
    return value;
}

template<int Q, bool INTERLEAVE_EXPERTS = false>
__global__ __launch_bounds__(256, 1)
__attribute__((amdgpu_num_vgpr(256)))
void opus_moe_wgrad_tn_mi16_inline_kernel(
    const __bf16* __restrict__ left,
    const __bf16* __restrict__ right,
    const int32_t* __restrict__ offs,
    __bf16* __restrict__ dW)
{
    constexpr int P = 2048;
    static_assert(Q == 1024 || Q == 2048);
    constexpr int BM = 256;
    constexpr int BN = 256;
    constexpr int BK = 64;
    constexpr int RIGHT_STAGE_BYTES = BK * Q * sizeof(__bf16);
    constexpr int OPERAND_BYTES = BK * BM * sizeof(__bf16);
    constexpr int BUFFER_BYTES = 2 * OPERAND_BYTES;
    __shared__ __align__(16) uint8_t lds[2][BUFFER_BYTES];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 63;
    const int warp = tid >> 6;
    const int scalar_warp = __builtin_amdgcn_readfirstlane(warp);
    const int wm = warp & 1;
    const int wn = warp >> 1;
    constexpr int GROUP_M = 8;
    constexpr int GROUP_N = 2;
    constexpr int TILES_N = Q / BN;
    constexpr int GROUPS_N = TILES_N / GROUP_N;
    int e;
    int linear;
    if constexpr(INTERLEAVE_EXPERTS)
    {
        static_assert(Q == 2048);
        constexpr int TILES_M = P / BM;
        constexpr int TILES_PER_EXPERT = TILES_M * TILES_N;
        constexpr int EXPERT_GROUP = 2;
        constexpr int TILES_PER_TURN = 2;
        const int schedule = static_cast<int>(blockIdx.x);
        const int expert_group =
            schedule / (TILES_PER_EXPERT * EXPERT_GROUP);
        const int group_work =
            schedule % (TILES_PER_EXPERT * EXPERT_GROUP);
        const int turn = group_work / (EXPERT_GROUP * TILES_PER_TURN);
        const int in_turn = group_work % (EXPERT_GROUP * TILES_PER_TURN);
        e = expert_group * EXPERT_GROUP + in_turn / TILES_PER_TURN;
        linear = turn * TILES_PER_TURN + in_turn % TILES_PER_TURN;
    }
    else
    {
        e = static_cast<int>(blockIdx.z);
        linear = static_cast<int>(blockIdx.y) * TILES_N +
                 static_cast<int>(blockIdx.x);
    }
    const int group_id = linear / (GROUP_M * GROUP_N);
    const int in_group = linear % (GROUP_M * GROUP_N);
    const int tile_m = (group_id / GROUPS_N) * GROUP_M +
                       in_group / GROUP_N;
    const int tile_n = (group_id % GROUPS_N) * GROUP_N +
                       in_group % GROUP_N;
    const int tile_m0 = tile_m * BM;
    const int tile_n0 = tile_n * BN;
    const int r0 = offs[e];
    const int r1 = offs[e + 1];
    const int nroute = r1 - r0;
    if(nroute <= 0)
        return;

    const unsigned int left_resource_bytes = static_cast<unsigned int>(
        (static_cast<int64_t>(nroute) * P - tile_m0) * sizeof(__bf16));
    const unsigned int right_resource_bytes = static_cast<unsigned int>(
        (static_cast<int64_t>(nroute) * Q - tile_n0) * sizeof(__bf16));
    auto g_left = opus::make_gmem(
        reinterpret_cast<const opus::bf16_t*>(left) +
            static_cast<int64_t>(r0) * P + tile_m0,
        left_resource_bytes);
    auto g_right = opus::make_gmem(
        reinterpret_cast<const opus::bf16_t*>(right) +
            static_cast<int64_t>(r0) * Q + tile_n0,
        right_resource_bytes);

    // Keep mutable buffer descriptors in fixed SGPR quartets. Advancing the
    // descriptor base/size once per K64 stage avoids sixteen dependent VGPR
    // address increments while preserving descriptor OOB zero-fill.
    opus::i32x4_t left_rsrc;
    opus::i32x4_t right_rsrc;
    __builtin_memcpy(&left_rsrc, &g_left.cached_rsrc, sizeof(left_rsrc));
    __builtin_memcpy(&right_rsrc, &g_right.cached_rsrc, sizeof(right_rsrc));

    const int load_route = tid >> 5;
    const int load_feature = (tid & 31) * 8;
    int right_vaddr[8];
    int left_vaddr[8];
#pragma unroll
    for(int group = 0; group < 8; ++group)
    {
        right_vaddr[group] =
            ((group * 8 + load_route) * Q + load_feature) * sizeof(__bf16);
        left_vaddr[group] =
            ((group * 8 + load_route) * P + load_feature) * sizeof(__bf16);
    }

    const uint32_t base0 = static_cast<uint32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(&lds[0][0]));
    const uint32_t base1 = static_cast<uint32_t>(
        reinterpret_cast<__UINTPTR_TYPE__>(&lds[1][0]));
    uint32_t right_lds = base0 + 16 * (lane & 15) +
                         4096 * (lane >> 4) + 256 * wn;
    uint32_t left_lds = base0 + OPERAND_BYTES + 16 * (lane & 15) +
                        4096 * (lane >> 4) + 256 * wm;
    const uint32_t right_m0_current = base0 + scalar_warp * 1024;
    const uint32_t left_m0_current =
        base0 + OPERAND_BYTES + scalar_warp * 1024;
    uint32_t right_m0_next = base1 + scalar_warp * 1024;
    uint32_t left_m0_next =
        base1 + OPERAND_BYTES + scalar_warp * 1024;
    uint32_t loop_count = (nroute + BK - 1) / BK - 1;
    const uint32_t perm_even = 0x05040100;
    const uint32_t perm_odd = 0x07060302;

    asm volatile(
R"MI16(
        s_mov_b32 s64, %[right_rsrc0]
        s_mov_b32 s65, %[right_rsrc1]
        s_mov_b32 s66, %[right_rsrc2]
        s_mov_b32 s67, %[right_rsrc3]
        s_mov_b32 s68, %[left_rsrc0]
        s_mov_b32 s69, %[left_rsrc1]
        s_mov_b32 s70, %[left_rsrc2]
        s_mov_b32 s71, %[left_rsrc3]
        v_mov_b32 v0, %[right_v0]
        v_mov_b32 v1, %[right_v1]
        v_mov_b32 v2, %[right_v2]
        v_mov_b32 v3, %[right_v3]
        v_mov_b32 v4, %[right_v4]
        v_mov_b32 v5, %[right_v5]
        v_mov_b32 v6, %[right_v6]
        v_mov_b32 v7, %[right_v7]
        v_mov_b32 v8, %[left_v0]
        v_mov_b32 v9, %[left_v1]
        v_mov_b32 v10, %[left_v2]
        v_mov_b32 v11, %[left_v3]
        v_mov_b32 v12, %[left_v4]
        v_mov_b32 v13, %[left_v5]
        v_mov_b32 v14, %[left_v6]
        v_mov_b32 v15, %[left_v7]
        s_mov_b32 m0, %[right_m0_current]
        s_nop 0
        buffer_load_dwordx4 v0, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v1, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v2, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v3, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v4, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v5, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v6, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v7, s[64:67], 0 offen lds
        s_mov_b32 m0, %[left_m0_current]
        s_nop 0
        buffer_load_dwordx4 v8, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v9, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v10, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v11, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v12, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v13, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v14, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v15, s[68:71], 0 offen lds
        v_mov_b64 v[210:211], 0
        v_accvgpr_write_b32 a0, v210
        v_accvgpr_write_b32 a1, v210
        v_accvgpr_write_b32 a2, v210
        v_accvgpr_write_b32 a3, v210
        v_accvgpr_write_b32 a4, v210
        v_accvgpr_write_b32 a5, v210
        v_accvgpr_write_b32 a6, v210
        v_accvgpr_write_b32 a7, v210
        v_accvgpr_write_b32 a8, v210
        v_accvgpr_write_b32 a9, v210
        v_accvgpr_write_b32 a10, v210
        v_accvgpr_write_b32 a11, v210
        v_accvgpr_write_b32 a12, v210
        v_accvgpr_write_b32 a13, v210
        v_accvgpr_write_b32 a14, v210
        v_accvgpr_write_b32 a15, v210
        v_mfma_i32_32x32x16_i8 a[16:31], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[32:47], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[48:63], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[64:79], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[80:95], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[96:111], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[112:127], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[128:143], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[144:159], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[160:175], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[176:191], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[192:207], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[208:223], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[224:239], v[210:211], v[210:211], a[0:15]
        v_mfma_i32_32x32x16_i8 a[240:255], v[210:211], v[210:211], a[0:15]
        s_waitcnt vmcnt(0)
        s_waitcnt lgkmcnt(0)
        s_barrier
        s_cmp_eq_u32 %[loop_count], 0
        s_cbranch_scc1 2f
        s_add_u32 s64, s64, %[right_stage_bytes]
        s_addc_u32 s65, s65, 0
        s_sub_u32 s66, s66, %[right_stage_bytes]
        s_add_u32 s68, s68, 0x40000
        s_addc_u32 s69, s69, 0
        s_sub_u32 s70, s70, 0x40000
        s_mov_b32 m0, %[right_m0_next]
        buffer_load_dwordx4 v0, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v1, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v2, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v3, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v4, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v5, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v6, s[64:67], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v7, s[64:67], 0 offen lds
        s_waitcnt lgkmcnt(0)
        s_barrier
        s_mov_b32 m0, %[left_m0_next]
        buffer_load_dwordx4 v8, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v9, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v10, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v11, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v12, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v13, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v14, s[68:71], 0 offen lds
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v15, s[68:71], 0 offen lds
        s_xor_b32 %[right_m0_next], 0x10000, %[right_m0_next]
        s_xor_b32 %[left_m0_next], 0x10000, %[left_m0_next]
        s_barrier
        2:
        ds_read_b128 v[82:85], %[right_lds] offset:0
        ds_read_b128 v[86:89], %[right_lds] offset:512
        ds_read_b128 v[90:93], %[right_lds] offset:1024
        ds_read_b128 v[94:97], %[right_lds] offset:1536
        ds_read_b128 v[98:101], %[right_lds] offset:2048
        ds_read_b128 v[102:105], %[right_lds] offset:2560
        ds_read_b128 v[106:109], %[right_lds] offset:3072
        ds_read_b128 v[110:113], %[right_lds] offset:3584
        ds_read_b128 v[178:181], %[left_lds] offset:0
        ds_read_b128 v[182:185], %[left_lds] offset:512
        ds_read_b128 v[186:189], %[left_lds] offset:1024
        ds_read_b128 v[190:193], %[left_lds] offset:1536
        ds_read_b128 v[194:197], %[left_lds] offset:2048
        ds_read_b128 v[198:201], %[left_lds] offset:2560
        ds_read_b128 v[202:205], %[left_lds] offset:3072
        ds_read_b128 v[206:209], %[left_lds] offset:3584
        s_waitcnt lgkmcnt(0)
        v_perm_b32 v18, v86, v82, %[perm_even]
        v_perm_b32 v19, v94, v90, %[perm_even]
        v_perm_b32 v20, v102, v98, %[perm_even]
        v_perm_b32 v21, v110, v106, %[perm_even]
        v_perm_b32 v22, v86, v82, %[perm_odd]
        v_perm_b32 v23, v94, v90, %[perm_odd]
        v_perm_b32 v24, v102, v98, %[perm_odd]
        v_perm_b32 v25, v110, v106, %[perm_odd]
        v_perm_b32 v26, v87, v83, %[perm_even]
        v_perm_b32 v27, v95, v91, %[perm_even]
        v_perm_b32 v28, v103, v99, %[perm_even]
        v_perm_b32 v29, v111, v107, %[perm_even]
        v_perm_b32 v30, v87, v83, %[perm_odd]
        v_perm_b32 v31, v95, v91, %[perm_odd]
        v_perm_b32 v32, v103, v99, %[perm_odd]
        v_perm_b32 v33, v111, v107, %[perm_odd]
        v_perm_b32 v34, v88, v84, %[perm_even]
        v_perm_b32 v35, v96, v92, %[perm_even]
        v_perm_b32 v36, v104, v100, %[perm_even]
        v_perm_b32 v37, v112, v108, %[perm_even]
        v_perm_b32 v38, v88, v84, %[perm_odd]
        v_perm_b32 v39, v96, v92, %[perm_odd]
        v_perm_b32 v40, v104, v100, %[perm_odd]
        v_perm_b32 v41, v112, v108, %[perm_odd]
        v_perm_b32 v42, v89, v85, %[perm_even]
        v_perm_b32 v43, v97, v93, %[perm_even]
        v_perm_b32 v44, v105, v101, %[perm_even]
        v_perm_b32 v45, v113, v109, %[perm_even]
        v_perm_b32 v46, v89, v85, %[perm_odd]
        v_perm_b32 v47, v97, v93, %[perm_odd]
        v_perm_b32 v48, v105, v101, %[perm_odd]
        v_perm_b32 v49, v113, v109, %[perm_odd]
        v_perm_b32 v114, v182, v178, %[perm_even]
        v_perm_b32 v115, v190, v186, %[perm_even]
        v_perm_b32 v116, v198, v194, %[perm_even]
        v_perm_b32 v117, v206, v202, %[perm_even]
        v_perm_b32 v118, v182, v178, %[perm_odd]
        v_perm_b32 v119, v190, v186, %[perm_odd]
        v_perm_b32 v120, v198, v194, %[perm_odd]
        v_perm_b32 v121, v206, v202, %[perm_odd]
        v_perm_b32 v122, v183, v179, %[perm_even]
        v_perm_b32 v123, v191, v187, %[perm_even]
        v_perm_b32 v124, v199, v195, %[perm_even]
        v_perm_b32 v125, v207, v203, %[perm_even]
        v_perm_b32 v126, v183, v179, %[perm_odd]
        v_perm_b32 v127, v191, v187, %[perm_odd]
        v_perm_b32 v128, v199, v195, %[perm_odd]
        v_perm_b32 v129, v207, v203, %[perm_odd]
        v_perm_b32 v130, v184, v180, %[perm_even]
        v_perm_b32 v131, v192, v188, %[perm_even]
        v_perm_b32 v132, v200, v196, %[perm_even]
        v_perm_b32 v133, v208, v204, %[perm_even]
        v_perm_b32 v134, v184, v180, %[perm_odd]
        v_perm_b32 v135, v192, v188, %[perm_odd]
        v_perm_b32 v136, v200, v196, %[perm_odd]
        v_perm_b32 v137, v208, v204, %[perm_odd]
        v_perm_b32 v138, v185, v181, %[perm_even]
        v_perm_b32 v139, v193, v189, %[perm_even]
        v_perm_b32 v140, v201, v197, %[perm_even]
        v_perm_b32 v141, v209, v205, %[perm_even]
        v_perm_b32 v142, v185, v181, %[perm_odd]
        v_perm_b32 v143, v193, v189, %[perm_odd]
        v_perm_b32 v144, v201, v197, %[perm_odd]
        v_perm_b32 v145, v209, v205, %[perm_odd]
        s_setprio 1
        s_cmp_eq_u32 %[loop_count], 0
        s_cbranch_scc1 3f
        1:
        v_mfma_f32_16x16x32_bf16 a[0:3], v[114:117], v[18:21], a[0:3]
        ds_read_b128 v[82:85], %[right_lds] offset:16384
        ds_read_b128 v[86:89], %[right_lds] offset:16896
        v_mfma_f32_16x16x32_bf16 a[4:7], v[114:117], v[22:25], a[4:7]
        v_mfma_f32_16x16x32_bf16 a[8:11], v[114:117], v[26:29], a[8:11]
        ds_read_b128 v[90:93], %[right_lds] offset:17408
        ds_read_b128 v[94:97], %[right_lds] offset:17920
        s_add_u32 s64, s64, %[right_stage_bytes]
        v_mfma_f32_16x16x32_bf16 a[12:15], v[114:117], v[30:33], a[12:15]
        s_addc_u32 s65, s65, 0
        v_mfma_f32_16x16x32_bf16 a[16:19], v[114:117], v[34:37], a[16:19]
        ds_read_b128 v[98:101], %[right_lds] offset:18432
        ds_read_b128 v[102:105], %[right_lds] offset:18944
        s_sub_u32 s66, s66, %[right_stage_bytes]
        s_cmp_eq_u32 %[loop_count], 1
        s_cselect_b32 s66, 0, s66
        v_mfma_f32_16x16x32_bf16 a[20:23], v[114:117], v[38:41], a[20:23]
        v_mfma_f32_16x16x32_bf16 a[24:27], v[114:117], v[42:45], a[24:27]
        ds_read_b128 v[106:109], %[right_lds] offset:19456
        ds_read_b128 v[110:113], %[right_lds] offset:19968
        v_mfma_f32_16x16x32_bf16 a[28:31], v[114:117], v[46:49], a[28:31]
        v_mfma_f32_16x16x32_bf16 a[32:35], v[118:121], v[18:21], a[32:35]
        ds_read_b128 v[178:181], %[left_lds] offset:16384
        ds_read_b128 v[182:185], %[left_lds] offset:16896
        v_mfma_f32_16x16x32_bf16 a[36:39], v[118:121], v[22:25], a[36:39]
        v_mfma_f32_16x16x32_bf16 a[40:43], v[118:121], v[26:29], a[40:43]
        ds_read_b128 v[186:189], %[left_lds] offset:17408
        ds_read_b128 v[190:193], %[left_lds] offset:17920
        v_mfma_f32_16x16x32_bf16 a[44:47], v[118:121], v[30:33], a[44:47]
        v_mfma_f32_16x16x32_bf16 a[48:51], v[118:121], v[34:37], a[48:51]
        s_waitcnt lgkmcnt(4)
        v_mfma_f32_16x16x32_bf16 a[52:55], v[118:121], v[38:41], a[52:55]
        s_barrier
        v_mfma_f32_16x16x32_bf16 a[56:59], v[118:121], v[42:45], a[56:59]
        s_mov_b32 m0, %[right_m0_next]
        buffer_load_dwordx4 v0, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[60:63], v[118:121], v[46:49], a[60:63]
        ds_read_b128 v[194:197], %[left_lds] offset:18432
        ds_read_b128 v[198:201], %[left_lds] offset:18944
        v_mfma_f32_16x16x32_bf16 a[64:67], v[122:125], v[18:21], a[64:67]
        v_perm_b32 v50, v86, v82, %[perm_even]
        v_perm_b32 v51, v94, v90, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[68:71], v[122:125], v[22:25], a[68:71]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v1, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[72:75], v[122:125], v[26:29], a[72:75]
        ds_read_b128 v[202:205], %[left_lds] offset:19456
        ds_read_b128 v[206:209], %[left_lds] offset:19968
        v_mfma_f32_16x16x32_bf16 a[76:79], v[122:125], v[30:33], a[76:79]
        v_perm_b32 v52, v102, v98, %[perm_even]
        v_perm_b32 v53, v110, v106, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[80:83], v[122:125], v[34:37], a[80:83]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v2, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[84:87], v[122:125], v[38:41], a[84:87]
        v_perm_b32 v54, v86, v82, %[perm_odd]
        v_perm_b32 v55, v94, v90, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[88:91], v[122:125], v[42:45], a[88:91]
        v_perm_b32 v56, v102, v98, %[perm_odd]
        v_perm_b32 v57, v110, v106, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[92:95], v[122:125], v[46:49], a[92:95]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v3, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[96:99], v[126:129], v[18:21], a[96:99]
        v_perm_b32 v58, v87, v83, %[perm_even]
        v_perm_b32 v59, v95, v91, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[100:103], v[126:129], v[22:25], a[100:103]
        v_perm_b32 v60, v103, v99, %[perm_even]
        v_perm_b32 v61, v111, v107, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[104:107], v[126:129], v[26:29], a[104:107]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v4, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[108:111], v[126:129], v[30:33], a[108:111]
        v_perm_b32 v62, v87, v83, %[perm_odd]
        v_perm_b32 v63, v95, v91, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[112:115], v[126:129], v[34:37], a[112:115]
        v_perm_b32 v64, v103, v99, %[perm_odd]
        v_perm_b32 v65, v111, v107, %[perm_odd]
        s_add_u32 s68, s68, 0x40000
        v_mfma_f32_16x16x32_bf16 a[116:119], v[126:129], v[38:41], a[116:119]
        v_perm_b32 v66, v88, v84, %[perm_even]
        v_perm_b32 v67, v96, v92, %[perm_even]
        s_addc_u32 s69, s69, 0
        v_mfma_f32_16x16x32_bf16 a[120:123], v[126:129], v[42:45], a[120:123]
        v_perm_b32 v68, v104, v100, %[perm_even]
        v_perm_b32 v69, v112, v108, %[perm_even]
        s_sub_u32 s70, s70, 0x40000
        s_cmp_eq_u32 %[loop_count], 1
        s_cselect_b32 s70, 0, s70
        v_mfma_f32_16x16x32_bf16 a[124:127], v[126:129], v[46:49], a[124:127]
        v_perm_b32 v70, v88, v84, %[perm_odd]
        v_perm_b32 v71, v96, v92, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[128:131], v[130:133], v[18:21], a[128:131]
        v_perm_b32 v72, v104, v100, %[perm_odd]
        v_perm_b32 v73, v112, v108, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[132:135], v[130:133], v[22:25], a[132:135]
        v_perm_b32 v74, v89, v85, %[perm_even]
        v_perm_b32 v75, v97, v93, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[136:139], v[130:133], v[26:29], a[136:139]
        v_perm_b32 v76, v105, v101, %[perm_even]
        v_perm_b32 v77, v113, v109, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[140:143], v[130:133], v[30:33], a[140:143]
        v_perm_b32 v78, v89, v85, %[perm_odd]
        v_perm_b32 v79, v97, v93, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[144:147], v[130:133], v[34:37], a[144:147]
        s_waitcnt lgkmcnt(0)
        v_perm_b32 v80, v105, v101, %[perm_odd]
        v_perm_b32 v81, v113, v109, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[148:151], v[130:133], v[38:41], a[148:151]
        v_perm_b32 v146, v182, v178, %[perm_even]
        v_perm_b32 v147, v190, v186, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[152:155], v[130:133], v[42:45], a[152:155]
        v_perm_b32 v148, v198, v194, %[perm_even]
        v_perm_b32 v149, v206, v202, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[156:159], v[130:133], v[46:49], a[156:159]
        v_perm_b32 v150, v182, v178, %[perm_odd]
        v_perm_b32 v151, v190, v186, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[160:163], v[134:137], v[18:21], a[160:163]
        v_perm_b32 v152, v198, v194, %[perm_odd]
        v_perm_b32 v153, v206, v202, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[164:167], v[134:137], v[22:25], a[164:167]
        v_perm_b32 v154, v183, v179, %[perm_even]
        v_perm_b32 v155, v191, v187, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[168:171], v[134:137], v[26:29], a[168:171]
        v_perm_b32 v156, v199, v195, %[perm_even]
        v_perm_b32 v157, v207, v203, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[172:175], v[134:137], v[30:33], a[172:175]
        v_perm_b32 v158, v183, v179, %[perm_odd]
        v_perm_b32 v159, v191, v187, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[176:179], v[134:137], v[34:37], a[176:179]
        s_barrier
        v_mfma_f32_16x16x32_bf16 a[180:183], v[134:137], v[38:41], a[180:183]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v5, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[184:187], v[134:137], v[42:45], a[184:187]
        v_perm_b32 v160, v199, v195, %[perm_odd]
        v_perm_b32 v161, v207, v203, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[188:191], v[134:137], v[46:49], a[188:191]
        v_perm_b32 v162, v184, v180, %[perm_even]
        v_perm_b32 v163, v192, v188, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[192:195], v[138:141], v[18:21], a[192:195]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v6, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[196:199], v[138:141], v[22:25], a[196:199]
        v_perm_b32 v164, v200, v196, %[perm_even]
        v_perm_b32 v165, v208, v204, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[200:203], v[138:141], v[26:29], a[200:203]
        v_perm_b32 v166, v184, v180, %[perm_odd]
        v_perm_b32 v167, v192, v188, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[204:207], v[138:141], v[30:33], a[204:207]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v7, s[64:67], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[208:211], v[138:141], v[34:37], a[208:211]
        v_perm_b32 v168, v200, v196, %[perm_odd]
        v_perm_b32 v169, v208, v204, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[212:215], v[138:141], v[38:41], a[212:215]
        v_perm_b32 v170, v185, v181, %[perm_even]
        v_perm_b32 v171, v193, v189, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[216:219], v[138:141], v[42:45], a[216:219]
        s_mov_b32 m0, %[left_m0_next]
        buffer_load_dwordx4 v8, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[220:223], v[138:141], v[46:49], a[220:223]
        v_perm_b32 v172, v201, v197, %[perm_even]
        v_perm_b32 v173, v209, v205, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[224:227], v[142:145], v[18:21], a[224:227]
        s_waitcnt vmcnt(17)
        v_perm_b32 v174, v185, v181, %[perm_odd]
        v_perm_b32 v175, v193, v189, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[228:231], v[142:145], v[22:25], a[228:231]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v9, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[232:235], v[142:145], v[26:29], a[232:235]
        v_perm_b32 v176, v201, v197, %[perm_odd]
        v_perm_b32 v177, v209, v205, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[236:239], v[142:145], v[30:33], a[236:239]
        s_barrier
        v_xor_b32 %[right_lds], 0x10000, %[right_lds]
        v_xor_b32 %[left_lds], 0x10000, %[left_lds]
        v_mfma_f32_16x16x32_bf16 a[240:243], v[142:145], v[34:37], a[240:243]
        ds_read_b128 v[82:85], %[right_lds]
        ds_read_b128 v[86:89], %[right_lds] offset:512
        v_mfma_f32_16x16x32_bf16 a[244:247], v[142:145], v[38:41], a[244:247]
        v_mfma_f32_16x16x32_bf16 a[248:251], v[142:145], v[42:45], a[248:251]
        ds_read_b128 v[90:93], %[right_lds] offset:1024
        ds_read_b128 v[94:97], %[right_lds] offset:1536
        v_mfma_f32_16x16x32_bf16 a[252:255], v[142:145], v[46:49], a[252:255]
        v_mfma_f32_16x16x32_bf16 a[0:3], v[146:149], v[50:53], a[0:3]
        ds_read_b128 v[98:101], %[right_lds] offset:2048
        ds_read_b128 v[102:105], %[right_lds] offset:2560
        v_mfma_f32_16x16x32_bf16 a[4:7], v[146:149], v[54:57], a[4:7]
        v_mfma_f32_16x16x32_bf16 a[8:11], v[146:149], v[58:61], a[8:11]
        s_waitcnt vmcnt(9)
        ds_read_b128 v[106:109], %[right_lds] offset:3072
        ds_read_b128 v[110:113], %[right_lds] offset:3584
        v_mfma_f32_16x16x32_bf16 a[12:15], v[146:149], v[62:65], a[12:15]
        v_mfma_f32_16x16x32_bf16 a[16:19], v[146:149], v[66:69], a[16:19]
        s_barrier
        v_mfma_f32_16x16x32_bf16 a[20:23], v[146:149], v[70:73], a[20:23]
        ds_read_b128 v[178:181], %[left_lds]
        ds_read_b128 v[182:185], %[left_lds] offset:512
        v_mfma_f32_16x16x32_bf16 a[24:27], v[146:149], v[74:77], a[24:27]
        v_mfma_f32_16x16x32_bf16 a[28:31], v[146:149], v[78:81], a[28:31]
        ds_read_b128 v[186:189], %[left_lds] offset:1024
        ds_read_b128 v[190:193], %[left_lds] offset:1536
        v_mfma_f32_16x16x32_bf16 a[32:35], v[150:153], v[50:53], a[32:35]
        v_mfma_f32_16x16x32_bf16 a[36:39], v[150:153], v[54:57], a[36:39]
        s_waitcnt lgkmcnt(4)
        ds_read_b128 v[194:197], %[left_lds] offset:2048
        ds_read_b128 v[198:201], %[left_lds] offset:2560
        v_mfma_f32_16x16x32_bf16 a[40:43], v[150:153], v[58:61], a[40:43]
        v_perm_b32 v18, v86, v82, %[perm_even]
        v_perm_b32 v19, v94, v90, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[44:47], v[150:153], v[62:65], a[44:47]
        ds_read_b128 v[202:205], %[left_lds] offset:3072
        ds_read_b128 v[206:209], %[left_lds] offset:3584
        v_mfma_f32_16x16x32_bf16 a[48:51], v[150:153], v[66:69], a[48:51]
        v_perm_b32 v20, v102, v98, %[perm_even]
        v_perm_b32 v21, v110, v106, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[52:55], v[150:153], v[70:73], a[52:55]
        v_perm_b32 v22, v86, v82, %[perm_odd]
        v_perm_b32 v23, v94, v90, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[56:59], v[150:153], v[74:77], a[56:59]
        v_perm_b32 v24, v102, v98, %[perm_odd]
        v_perm_b32 v25, v110, v106, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[60:63], v[150:153], v[78:81], a[60:63]
        v_perm_b32 v26, v87, v83, %[perm_even]
        v_perm_b32 v27, v95, v91, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[64:67], v[154:157], v[50:53], a[64:67]
        v_perm_b32 v28, v103, v99, %[perm_even]
        v_perm_b32 v29, v111, v107, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[68:71], v[154:157], v[54:57], a[68:71]
        v_perm_b32 v30, v87, v83, %[perm_odd]
        v_perm_b32 v31, v95, v91, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[72:75], v[154:157], v[58:61], a[72:75]
        v_perm_b32 v32, v103, v99, %[perm_odd]
        v_perm_b32 v33, v111, v107, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[76:79], v[154:157], v[62:65], a[76:79]
        v_perm_b32 v34, v88, v84, %[perm_even]
        v_perm_b32 v35, v96, v92, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[80:83], v[154:157], v[66:69], a[80:83]
        v_perm_b32 v36, v104, v100, %[perm_even]
        v_perm_b32 v37, v112, v108, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[84:87], v[154:157], v[70:73], a[84:87]
        v_perm_b32 v38, v88, v84, %[perm_odd]
        v_perm_b32 v39, v96, v92, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[88:91], v[154:157], v[74:77], a[88:91]
        v_perm_b32 v40, v104, v100, %[perm_odd]
        v_perm_b32 v41, v112, v108, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[92:95], v[154:157], v[78:81], a[92:95]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v10, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[96:99], v[158:161], v[50:53], a[96:99]
        v_perm_b32 v42, v89, v85, %[perm_even]
        v_perm_b32 v43, v97, v93, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[100:103], v[158:161], v[54:57], a[100:103]
        v_perm_b32 v44, v105, v101, %[perm_even]
        v_perm_b32 v45, v113, v109, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[104:107], v[158:161], v[58:61], a[104:107]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v11, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[108:111], v[158:161], v[62:65], a[108:111]
        v_perm_b32 v46, v89, v85, %[perm_odd]
        v_perm_b32 v47, v97, v93, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[112:115], v[158:161], v[66:69], a[112:115]
        s_waitcnt lgkmcnt(0)
        v_perm_b32 v48, v105, v101, %[perm_odd]
        v_perm_b32 v49, v113, v109, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[116:119], v[158:161], v[70:73], a[116:119]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v12, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[120:123], v[158:161], v[74:77], a[120:123]
        v_perm_b32 v114, v182, v178, %[perm_even]
        v_perm_b32 v115, v190, v186, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[124:127], v[158:161], v[78:81], a[124:127]
        v_perm_b32 v116, v198, v194, %[perm_even]
        v_perm_b32 v117, v206, v202, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[128:131], v[162:165], v[50:53], a[128:131]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v13, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[132:135], v[162:165], v[54:57], a[132:135]
        v_perm_b32 v118, v182, v178, %[perm_odd]
        v_perm_b32 v119, v190, v186, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[136:139], v[162:165], v[58:61], a[136:139]
        v_perm_b32 v120, v198, v194, %[perm_odd]
        v_perm_b32 v121, v206, v202, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[140:143], v[162:165], v[62:65], a[140:143]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v14, s[68:71], 0 offen lds
        v_mfma_f32_16x16x32_bf16 a[144:147], v[162:165], v[66:69], a[144:147]
        v_perm_b32 v122, v183, v179, %[perm_even]
        v_perm_b32 v123, v191, v187, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[148:151], v[162:165], v[70:73], a[148:151]
        v_perm_b32 v124, v199, v195, %[perm_even]
        v_perm_b32 v125, v207, v203, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[152:155], v[162:165], v[74:77], a[152:155]
        v_perm_b32 v126, v183, v179, %[perm_odd]
        v_perm_b32 v127, v191, v187, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[156:159], v[162:165], v[78:81], a[156:159]
        v_perm_b32 v128, v199, v195, %[perm_odd]
        v_perm_b32 v129, v207, v203, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[160:163], v[166:169], v[50:53], a[160:163]
        v_perm_b32 v130, v184, v180, %[perm_even]
        v_perm_b32 v131, v192, v188, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[164:167], v[166:169], v[54:57], a[164:167]
        v_perm_b32 v132, v200, v196, %[perm_even]
        v_perm_b32 v133, v208, v204, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[168:171], v[166:169], v[58:61], a[168:171]
        v_perm_b32 v134, v184, v180, %[perm_odd]
        v_perm_b32 v135, v192, v188, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[172:175], v[166:169], v[62:65], a[172:175]
        v_perm_b32 v136, v200, v196, %[perm_odd]
        v_perm_b32 v137, v208, v204, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[176:179], v[166:169], v[66:69], a[176:179]
        v_perm_b32 v138, v185, v181, %[perm_even]
        v_perm_b32 v139, v193, v189, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[180:183], v[166:169], v[70:73], a[180:183]
        v_perm_b32 v140, v201, v197, %[perm_even]
        v_perm_b32 v141, v209, v205, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[184:187], v[166:169], v[74:77], a[184:187]
        v_perm_b32 v142, v185, v181, %[perm_odd]
        v_perm_b32 v143, v193, v189, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[188:191], v[166:169], v[78:81], a[188:191]
        v_perm_b32 v144, v201, v197, %[perm_odd]
        v_perm_b32 v145, v209, v205, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[192:195], v[170:173], v[50:53], a[192:195]
        v_mfma_f32_16x16x32_bf16 a[196:199], v[170:173], v[54:57], a[196:199]
        v_mfma_f32_16x16x32_bf16 a[200:203], v[170:173], v[58:61], a[200:203]
        v_mfma_f32_16x16x32_bf16 a[204:207], v[170:173], v[62:65], a[204:207]
        v_mfma_f32_16x16x32_bf16 a[208:211], v[170:173], v[66:69], a[208:211]
        v_mfma_f32_16x16x32_bf16 a[212:215], v[170:173], v[70:73], a[212:215]
        v_mfma_f32_16x16x32_bf16 a[216:219], v[170:173], v[74:77], a[216:219]
        v_mfma_f32_16x16x32_bf16 a[220:223], v[170:173], v[78:81], a[220:223]
        v_mfma_f32_16x16x32_bf16 a[224:227], v[174:177], v[50:53], a[224:227]
        v_mfma_f32_16x16x32_bf16 a[228:231], v[174:177], v[54:57], a[228:231]
        v_mfma_f32_16x16x32_bf16 a[232:235], v[174:177], v[58:61], a[232:235]
        v_mfma_f32_16x16x32_bf16 a[236:239], v[174:177], v[62:65], a[236:239]
        s_add_u32 m0, m0, 0x1000
        buffer_load_dwordx4 v15, s[68:71], 0 offen lds
        s_xor_b32 %[right_m0_next], 0x10000, %[right_m0_next]
        s_xor_b32 %[left_m0_next], 0x10000, %[left_m0_next]
        v_mfma_f32_16x16x32_bf16 a[240:243], v[174:177], v[66:69], a[240:243]
        v_mfma_f32_16x16x32_bf16 a[244:247], v[174:177], v[70:73], a[244:247]
        v_mfma_f32_16x16x32_bf16 a[248:251], v[174:177], v[74:77], a[248:251]
        v_mfma_f32_16x16x32_bf16 a[252:255], v[174:177], v[78:81], a[252:255]
        s_sub_u32 %[loop_count], %[loop_count], 1
        s_cmp_lg_u32 %[loop_count], 0
        s_cbranch_scc1 1b
        3:
        s_waitcnt vmcnt(0)
        s_waitcnt lgkmcnt(0)
        s_barrier
        ds_read_b128 v[82:85], %[right_lds] offset:16384
        v_mfma_f32_16x16x32_bf16 a[0:3], v[114:117], v[18:21], a[0:3]
        ds_read_b128 v[86:89], %[right_lds] offset:16896
        v_mfma_f32_16x16x32_bf16 a[4:7], v[114:117], v[22:25], a[4:7]
        ds_read_b128 v[90:93], %[right_lds] offset:17408
        v_mfma_f32_16x16x32_bf16 a[8:11], v[114:117], v[26:29], a[8:11]
        ds_read_b128 v[94:97], %[right_lds] offset:17920
        v_mfma_f32_16x16x32_bf16 a[12:15], v[114:117], v[30:33], a[12:15]
        ds_read_b128 v[98:101], %[right_lds] offset:18432
        v_mfma_f32_16x16x32_bf16 a[16:19], v[114:117], v[34:37], a[16:19]
        ds_read_b128 v[102:105], %[right_lds] offset:18944
        v_mfma_f32_16x16x32_bf16 a[20:23], v[114:117], v[38:41], a[20:23]
        ds_read_b128 v[106:109], %[right_lds] offset:19456
        v_mfma_f32_16x16x32_bf16 a[24:27], v[114:117], v[42:45], a[24:27]
        ds_read_b128 v[110:113], %[right_lds] offset:19968
        v_mfma_f32_16x16x32_bf16 a[28:31], v[114:117], v[46:49], a[28:31]
        ds_read_b128 v[178:181], %[left_lds] offset:16384
        v_mfma_f32_16x16x32_bf16 a[32:35], v[118:121], v[18:21], a[32:35]
        ds_read_b128 v[182:185], %[left_lds] offset:16896
        v_mfma_f32_16x16x32_bf16 a[36:39], v[118:121], v[22:25], a[36:39]
        ds_read_b128 v[186:189], %[left_lds] offset:17408
        v_mfma_f32_16x16x32_bf16 a[40:43], v[118:121], v[26:29], a[40:43]
        ds_read_b128 v[190:193], %[left_lds] offset:17920
        v_mfma_f32_16x16x32_bf16 a[44:47], v[118:121], v[30:33], a[44:47]
        ds_read_b128 v[194:197], %[left_lds] offset:18432
        v_mfma_f32_16x16x32_bf16 a[48:51], v[118:121], v[34:37], a[48:51]
        ds_read_b128 v[198:201], %[left_lds] offset:18944
        v_mfma_f32_16x16x32_bf16 a[52:55], v[118:121], v[38:41], a[52:55]
        ds_read_b128 v[202:205], %[left_lds] offset:19456
        v_mfma_f32_16x16x32_bf16 a[56:59], v[118:121], v[42:45], a[56:59]
        ds_read_b128 v[206:209], %[left_lds] offset:19968
        v_mfma_f32_16x16x32_bf16 a[60:63], v[118:121], v[46:49], a[60:63]
        s_waitcnt lgkmcnt(0)
        v_perm_b32 v50, v86, v82, %[perm_even]
        v_perm_b32 v146, v182, v178, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[64:67], v[122:125], v[18:21], a[64:67]
        v_perm_b32 v51, v94, v90, %[perm_even]
        v_perm_b32 v147, v190, v186, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[68:71], v[122:125], v[22:25], a[68:71]
        v_perm_b32 v52, v102, v98, %[perm_even]
        v_perm_b32 v148, v198, v194, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[72:75], v[122:125], v[26:29], a[72:75]
        v_perm_b32 v53, v110, v106, %[perm_even]
        v_perm_b32 v149, v206, v202, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[76:79], v[122:125], v[30:33], a[76:79]
        v_perm_b32 v54, v86, v82, %[perm_odd]
        v_perm_b32 v150, v182, v178, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[80:83], v[122:125], v[34:37], a[80:83]
        v_perm_b32 v55, v94, v90, %[perm_odd]
        v_perm_b32 v151, v190, v186, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[84:87], v[122:125], v[38:41], a[84:87]
        v_perm_b32 v56, v102, v98, %[perm_odd]
        v_perm_b32 v152, v198, v194, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[88:91], v[122:125], v[42:45], a[88:91]
        v_perm_b32 v57, v110, v106, %[perm_odd]
        v_perm_b32 v153, v206, v202, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[92:95], v[122:125], v[46:49], a[92:95]
        v_perm_b32 v58, v87, v83, %[perm_even]
        v_perm_b32 v154, v183, v179, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[96:99], v[126:129], v[18:21], a[96:99]
        v_perm_b32 v59, v95, v91, %[perm_even]
        v_perm_b32 v155, v191, v187, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[100:103], v[126:129], v[22:25], a[100:103]
        v_perm_b32 v60, v103, v99, %[perm_even]
        v_perm_b32 v156, v199, v195, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[104:107], v[126:129], v[26:29], a[104:107]
        v_perm_b32 v61, v111, v107, %[perm_even]
        v_perm_b32 v157, v207, v203, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[108:111], v[126:129], v[30:33], a[108:111]
        v_perm_b32 v62, v87, v83, %[perm_odd]
        v_perm_b32 v158, v183, v179, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[112:115], v[126:129], v[34:37], a[112:115]
        v_perm_b32 v63, v95, v91, %[perm_odd]
        v_perm_b32 v159, v191, v187, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[116:119], v[126:129], v[38:41], a[116:119]
        v_perm_b32 v64, v103, v99, %[perm_odd]
        v_perm_b32 v160, v199, v195, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[120:123], v[126:129], v[42:45], a[120:123]
        v_perm_b32 v65, v111, v107, %[perm_odd]
        v_perm_b32 v161, v207, v203, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[124:127], v[126:129], v[46:49], a[124:127]
        v_perm_b32 v66, v88, v84, %[perm_even]
        v_perm_b32 v162, v184, v180, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[128:131], v[130:133], v[18:21], a[128:131]
        v_perm_b32 v67, v96, v92, %[perm_even]
        v_perm_b32 v163, v192, v188, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[132:135], v[130:133], v[22:25], a[132:135]
        v_perm_b32 v68, v104, v100, %[perm_even]
        v_perm_b32 v164, v200, v196, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[136:139], v[130:133], v[26:29], a[136:139]
        v_perm_b32 v69, v112, v108, %[perm_even]
        v_perm_b32 v165, v208, v204, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[140:143], v[130:133], v[30:33], a[140:143]
        v_perm_b32 v70, v88, v84, %[perm_odd]
        v_perm_b32 v166, v184, v180, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[144:147], v[130:133], v[34:37], a[144:147]
        v_perm_b32 v71, v96, v92, %[perm_odd]
        v_perm_b32 v167, v192, v188, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[148:151], v[130:133], v[38:41], a[148:151]
        v_perm_b32 v72, v104, v100, %[perm_odd]
        v_perm_b32 v168, v200, v196, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[152:155], v[130:133], v[42:45], a[152:155]
        v_perm_b32 v73, v112, v108, %[perm_odd]
        v_perm_b32 v169, v208, v204, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[156:159], v[130:133], v[46:49], a[156:159]
        v_perm_b32 v74, v89, v85, %[perm_even]
        v_perm_b32 v170, v185, v181, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[160:163], v[134:137], v[18:21], a[160:163]
        v_perm_b32 v75, v97, v93, %[perm_even]
        v_perm_b32 v171, v193, v189, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[164:167], v[134:137], v[22:25], a[164:167]
        v_perm_b32 v76, v105, v101, %[perm_even]
        v_perm_b32 v172, v201, v197, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[168:171], v[134:137], v[26:29], a[168:171]
        v_perm_b32 v77, v113, v109, %[perm_even]
        v_perm_b32 v173, v209, v205, %[perm_even]
        v_mfma_f32_16x16x32_bf16 a[172:175], v[134:137], v[30:33], a[172:175]
        v_perm_b32 v78, v89, v85, %[perm_odd]
        v_perm_b32 v174, v185, v181, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[176:179], v[134:137], v[34:37], a[176:179]
        v_perm_b32 v79, v97, v93, %[perm_odd]
        v_perm_b32 v175, v193, v189, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[180:183], v[134:137], v[38:41], a[180:183]
        v_perm_b32 v80, v105, v101, %[perm_odd]
        v_perm_b32 v176, v201, v197, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[184:187], v[134:137], v[42:45], a[184:187]
        v_perm_b32 v81, v113, v109, %[perm_odd]
        v_perm_b32 v177, v209, v205, %[perm_odd]
        v_mfma_f32_16x16x32_bf16 a[188:191], v[134:137], v[46:49], a[188:191]
        v_mfma_f32_16x16x32_bf16 a[192:195], v[138:141], v[18:21], a[192:195]
        v_mfma_f32_16x16x32_bf16 a[196:199], v[138:141], v[22:25], a[196:199]
        v_mfma_f32_16x16x32_bf16 a[200:203], v[138:141], v[26:29], a[200:203]
        v_mfma_f32_16x16x32_bf16 a[204:207], v[138:141], v[30:33], a[204:207]
        v_mfma_f32_16x16x32_bf16 a[208:211], v[138:141], v[34:37], a[208:211]
        v_mfma_f32_16x16x32_bf16 a[212:215], v[138:141], v[38:41], a[212:215]
        v_mfma_f32_16x16x32_bf16 a[216:219], v[138:141], v[42:45], a[216:219]
        v_mfma_f32_16x16x32_bf16 a[220:223], v[138:141], v[46:49], a[220:223]
        v_mfma_f32_16x16x32_bf16 a[224:227], v[142:145], v[18:21], a[224:227]
        v_mfma_f32_16x16x32_bf16 a[228:231], v[142:145], v[22:25], a[228:231]
        v_mfma_f32_16x16x32_bf16 a[232:235], v[142:145], v[26:29], a[232:235]
        v_mfma_f32_16x16x32_bf16 a[236:239], v[142:145], v[30:33], a[236:239]
        v_mfma_f32_16x16x32_bf16 a[240:243], v[142:145], v[34:37], a[240:243]
        v_mfma_f32_16x16x32_bf16 a[244:247], v[142:145], v[38:41], a[244:247]
        v_mfma_f32_16x16x32_bf16 a[248:251], v[142:145], v[42:45], a[248:251]
        v_mfma_f32_16x16x32_bf16 a[252:255], v[142:145], v[46:49], a[252:255]
        v_mfma_f32_16x16x32_bf16 a[0:3], v[146:149], v[50:53], a[0:3]
        v_mfma_f32_16x16x32_bf16 a[4:7], v[146:149], v[54:57], a[4:7]
        v_mfma_f32_16x16x32_bf16 a[8:11], v[146:149], v[58:61], a[8:11]
        v_mfma_f32_16x16x32_bf16 a[12:15], v[146:149], v[62:65], a[12:15]
        v_mfma_f32_16x16x32_bf16 a[16:19], v[146:149], v[66:69], a[16:19]
        v_mfma_f32_16x16x32_bf16 a[20:23], v[146:149], v[70:73], a[20:23]
        v_mfma_f32_16x16x32_bf16 a[24:27], v[146:149], v[74:77], a[24:27]
        v_mfma_f32_16x16x32_bf16 a[28:31], v[146:149], v[78:81], a[28:31]
        v_mfma_f32_16x16x32_bf16 a[32:35], v[150:153], v[50:53], a[32:35]
        v_mfma_f32_16x16x32_bf16 a[36:39], v[150:153], v[54:57], a[36:39]
        v_mfma_f32_16x16x32_bf16 a[40:43], v[150:153], v[58:61], a[40:43]
        v_mfma_f32_16x16x32_bf16 a[44:47], v[150:153], v[62:65], a[44:47]
        v_mfma_f32_16x16x32_bf16 a[48:51], v[150:153], v[66:69], a[48:51]
        v_mfma_f32_16x16x32_bf16 a[52:55], v[150:153], v[70:73], a[52:55]
        v_mfma_f32_16x16x32_bf16 a[56:59], v[150:153], v[74:77], a[56:59]
        v_mfma_f32_16x16x32_bf16 a[60:63], v[150:153], v[78:81], a[60:63]
        v_mfma_f32_16x16x32_bf16 a[64:67], v[154:157], v[50:53], a[64:67]
        v_mfma_f32_16x16x32_bf16 a[68:71], v[154:157], v[54:57], a[68:71]
        v_mfma_f32_16x16x32_bf16 a[72:75], v[154:157], v[58:61], a[72:75]
        v_mfma_f32_16x16x32_bf16 a[76:79], v[154:157], v[62:65], a[76:79]
        v_mfma_f32_16x16x32_bf16 a[80:83], v[154:157], v[66:69], a[80:83]
        v_mfma_f32_16x16x32_bf16 a[84:87], v[154:157], v[70:73], a[84:87]
        v_mfma_f32_16x16x32_bf16 a[88:91], v[154:157], v[74:77], a[88:91]
        v_mfma_f32_16x16x32_bf16 a[92:95], v[154:157], v[78:81], a[92:95]
        v_mfma_f32_16x16x32_bf16 a[96:99], v[158:161], v[50:53], a[96:99]
        v_mfma_f32_16x16x32_bf16 a[100:103], v[158:161], v[54:57], a[100:103]
        v_mfma_f32_16x16x32_bf16 a[104:107], v[158:161], v[58:61], a[104:107]
        v_mfma_f32_16x16x32_bf16 a[108:111], v[158:161], v[62:65], a[108:111]
        v_mfma_f32_16x16x32_bf16 a[112:115], v[158:161], v[66:69], a[112:115]
        v_mfma_f32_16x16x32_bf16 a[116:119], v[158:161], v[70:73], a[116:119]
        v_mfma_f32_16x16x32_bf16 a[120:123], v[158:161], v[74:77], a[120:123]
        v_mfma_f32_16x16x32_bf16 a[124:127], v[158:161], v[78:81], a[124:127]
        v_mfma_f32_16x16x32_bf16 a[128:131], v[162:165], v[50:53], a[128:131]
        v_mfma_f32_16x16x32_bf16 a[132:135], v[162:165], v[54:57], a[132:135]
        v_mfma_f32_16x16x32_bf16 a[136:139], v[162:165], v[58:61], a[136:139]
        v_mfma_f32_16x16x32_bf16 a[140:143], v[162:165], v[62:65], a[140:143]
        v_mfma_f32_16x16x32_bf16 a[144:147], v[162:165], v[66:69], a[144:147]
        v_mfma_f32_16x16x32_bf16 a[148:151], v[162:165], v[70:73], a[148:151]
        v_mfma_f32_16x16x32_bf16 a[152:155], v[162:165], v[74:77], a[152:155]
        v_mfma_f32_16x16x32_bf16 a[156:159], v[162:165], v[78:81], a[156:159]
        v_mfma_f32_16x16x32_bf16 a[160:163], v[166:169], v[50:53], a[160:163]
        v_mfma_f32_16x16x32_bf16 a[164:167], v[166:169], v[54:57], a[164:167]
        v_mfma_f32_16x16x32_bf16 a[168:171], v[166:169], v[58:61], a[168:171]
        v_mfma_f32_16x16x32_bf16 a[172:175], v[166:169], v[62:65], a[172:175]
        v_mfma_f32_16x16x32_bf16 a[176:179], v[166:169], v[66:69], a[176:179]
        v_mfma_f32_16x16x32_bf16 a[180:183], v[166:169], v[70:73], a[180:183]
        v_mfma_f32_16x16x32_bf16 a[184:187], v[166:169], v[74:77], a[184:187]
        v_mfma_f32_16x16x32_bf16 a[188:191], v[166:169], v[78:81], a[188:191]
        v_mfma_f32_16x16x32_bf16 a[192:195], v[170:173], v[50:53], a[192:195]
        v_mfma_f32_16x16x32_bf16 a[196:199], v[170:173], v[54:57], a[196:199]
        v_mfma_f32_16x16x32_bf16 a[200:203], v[170:173], v[58:61], a[200:203]
        v_mfma_f32_16x16x32_bf16 a[204:207], v[170:173], v[62:65], a[204:207]
        v_mfma_f32_16x16x32_bf16 a[208:211], v[170:173], v[66:69], a[208:211]
        v_mfma_f32_16x16x32_bf16 a[212:215], v[170:173], v[70:73], a[212:215]
        v_mfma_f32_16x16x32_bf16 a[216:219], v[170:173], v[74:77], a[216:219]
        v_mfma_f32_16x16x32_bf16 a[220:223], v[170:173], v[78:81], a[220:223]
        v_mfma_f32_16x16x32_bf16 a[224:227], v[174:177], v[50:53], a[224:227]
        v_mfma_f32_16x16x32_bf16 a[228:231], v[174:177], v[54:57], a[228:231]
        v_mfma_f32_16x16x32_bf16 a[232:235], v[174:177], v[58:61], a[232:235]
        v_mfma_f32_16x16x32_bf16 a[236:239], v[174:177], v[62:65], a[236:239]
        v_mfma_f32_16x16x32_bf16 a[240:243], v[174:177], v[66:69], a[240:243]
        v_mfma_f32_16x16x32_bf16 a[244:247], v[174:177], v[70:73], a[244:247]
        v_mfma_f32_16x16x32_bf16 a[248:251], v[174:177], v[74:77], a[248:251]
        v_mfma_f32_16x16x32_bf16 a[252:255], v[174:177], v[78:81], a[252:255]
        s_setprio 0
)MI16"
        : [right_lds] "+&v"(right_lds),
          [left_lds] "+&v"(left_lds),
          [right_m0_next] "+&s"(right_m0_next),
          [left_m0_next] "+&s"(left_m0_next),
          [loop_count] "+&s"(loop_count)
        :
        [right_rsrc0] "s"(right_rsrc[0]),
        [right_rsrc1] "s"(right_rsrc[1]),
        [right_rsrc2] "s"(right_rsrc[2]),
        [right_rsrc3] "s"(right_rsrc[3]),
        [left_rsrc0] "s"(left_rsrc[0]),
        [left_rsrc1] "s"(left_rsrc[1]),
        [left_rsrc2] "s"(left_rsrc[2]),
        [left_rsrc3] "s"(left_rsrc[3]),
        [right_stage_bytes] "n"(RIGHT_STAGE_BYTES),
        [right_m0_current] "s"(right_m0_current),
        [left_m0_current] "s"(left_m0_current),
        [perm_even] "s"(perm_even),
        [perm_odd] "s"(perm_odd),
        [right_v0] "v"(right_vaddr[0]),
        [right_v1] "v"(right_vaddr[1]),
        [right_v2] "v"(right_vaddr[2]),
        [right_v3] "v"(right_vaddr[3]),
        [right_v4] "v"(right_vaddr[4]),
        [right_v5] "v"(right_vaddr[5]),
        [right_v6] "v"(right_vaddr[6]),
        [right_v7] "v"(right_vaddr[7]),
        [left_v0] "v"(left_vaddr[0]),
        [left_v1] "v"(left_vaddr[1]),
        [left_v2] "v"(left_vaddr[2]),
        [left_v3] "v"(left_vaddr[3]),
        [left_v4] "v"(left_vaddr[4]),
        [left_v5] "v"(left_vaddr[5]),
        [left_v6] "v"(left_vaddr[6]),
        [left_v7] "v"(left_vaddr[7])
        :
            "s64", "s65", "s66", "s67", "s68", "s69", "s70", "s71",
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
            "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
            "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
            "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39",
            "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47",
            "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55",
            "v56", "v57", "v58", "v59", "v60", "v61", "v62", "v63",
            "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71",
            "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79",
            "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87",
            "v88", "v89", "v90", "v91", "v92", "v93", "v94", "v95",
            "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103",
            "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111",
            "v112", "v113", "v114", "v115", "v116", "v117", "v118", "v119",
            "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127",
            "v128", "v129", "v130", "v131", "v132", "v133", "v134", "v135",
            "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143",
            "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151",
            "v152", "v153", "v154", "v155", "v156", "v157", "v158", "v159",
            "v160", "v161", "v162", "v163", "v164", "v165", "v166", "v167",
            "v168", "v169", "v170", "v171", "v172", "v173", "v174", "v175",
            "v176", "v177", "v178", "v179", "v180", "v181", "v182", "v183",
            "v184", "v185", "v186", "v187", "v188", "v189", "v190", "v191",
            "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199",
            "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207",
            "v208", "v209", "v210", "v211", "v212", "v213", "a0", "a1",
            "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
            "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17",
            "a18", "a19", "a20", "a21", "a22", "a23", "a24", "a25",
            "a26", "a27", "a28", "a29", "a30", "a31", "a32", "a33",
            "a34", "a35", "a36", "a37", "a38", "a39", "a40", "a41",
            "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",
            "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57",
            "a58", "a59", "a60", "a61", "a62", "a63", "a64", "a65",
            "a66", "a67", "a68", "a69", "a70", "a71", "a72", "a73",
            "a74", "a75", "a76", "a77", "a78", "a79", "a80", "a81",
            "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
            "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97",
            "a98", "a99", "a100", "a101", "a102", "a103", "a104", "a105",
            "a106", "a107", "a108", "a109", "a110", "a111", "a112", "a113",
            "a114", "a115", "a116", "a117", "a118", "a119", "a120", "a121",
            "a122", "a123", "a124", "a125", "a126", "a127", "a128", "a129",
            "a130", "a131", "a132", "a133", "a134", "a135", "a136", "a137",
            "a138", "a139", "a140", "a141", "a142", "a143", "a144", "a145",
            "a146", "a147", "a148", "a149", "a150", "a151", "a152", "a153",
            "a154", "a155", "a156", "a157", "a158", "a159", "a160", "a161",
            "a162", "a163", "a164", "a165", "a166", "a167", "a168", "a169",
            "a170", "a171", "a172", "a173", "a174", "a175", "a176", "a177",
            "a178", "a179", "a180", "a181", "a182", "a183", "a184", "a185",
            "a186", "a187", "a188", "a189", "a190", "a191", "a192", "a193",
            "a194", "a195", "a196", "a197", "a198", "a199", "a200", "a201",
            "a202", "a203", "a204", "a205", "a206", "a207", "a208", "a209",
            "a210", "a211", "a212", "a213", "a214", "a215", "a216", "a217",
            "a218", "a219", "a220", "a221", "a222", "a223", "a224", "a225",
            "a226", "a227", "a228", "a229", "a230", "a231", "a232", "a233",
            "a234", "a235", "a236", "a237", "a238", "a239", "a240", "a241",
            "a242", "a243", "a244", "a245", "a246", "a247", "a248", "a249",
            "a250", "a251", "a252", "a253", "a254", "a255", "memory"    );

    constexpr int INTERLEAVED_EXPERTS = INTERLEAVE_EXPERTS ? 64 : 0;
    const int output_experts = INTERLEAVED_EXPERTS > 0
                                   ? INTERLEAVED_EXPERTS
                                   : static_cast<int>(gridDim.z);
    const unsigned int dW_resource_bytes = static_cast<unsigned int>(
        static_cast<uint64_t>(output_experts) * P * Q * sizeof(__bf16));
    auto g_dW = opus::make_gmem(
        reinterpret_cast<opus::bf16_t*>(dW), dW_resource_bytes);
    const int dW_e = e * P * Q;
    const int ln = lane & 15;
    const int q0 = tile_n0 + wn * 128 + ln * 8;
    opus::static_for<8>([&](auto sm) {
        opus::static_for<4>([&](auto i) {
            opus::vector_t<opus::bf16_t, 8> values;
            opus::static_for<8>([&](auto sn) {
                constexpr int c =
                    (sm.value * 8 + sn.value) * 4 + i.value;
                const uint32_t bits =
                    opus_wgtn_mi16_inline_read_acc<c>();
                const float value = __builtin_bit_cast(float, bits);
                values[sn.value] = opus::fp32_to_bf16(value);
            });
            const int p_perm = sm.value * 16 +
                               (lane >> 4) * 4 + i.value;
            const int p = tile_m0 + wm * 128 +
                          (p_perm & 15) * 8 + (p_perm >> 4);
            g_dW.template store<8>(values, dW_e + p * Q + q0);
        });
    });
}

inline bool opus_moe_wgrad_tn_mi16_inline_try_launch_gfx950(
    const __bf16* left,
    const __bf16* right,
    const int32_t* offs,
    __bf16* dW,
    int E,
    int P,
    int Q,
    int uniform_m,
    hipStream_t stream)
{
    // This experimental schedule assumes the target workload's long K loop.
    // Keep all smaller/general grouped shapes on the production fallback.
    if((E != 8 && E != 64) || P != 2048 ||
       (Q != 1024 && Q != 2048))
        return false;
    const bool interleave_experts =
        Q == 2048 && E == 64 && uniform_m == 0;
    dim3 grid = interleave_experts
                    ? dim3(E * (P / 256) * (Q / 256))
                    : dim3(Q / 256, P / 256, E);
    if(Q == 1024)
        opus_moe_wgrad_tn_mi16_inline_kernel<1024, false>
            <<<grid, 256, 0, stream>>>(left, right, offs, dW);
    else if(interleave_experts)
        opus_moe_wgrad_tn_mi16_inline_kernel<2048, true>
            <<<grid, 256, 0, stream>>>(left, right, offs, dW);
    else
        opus_moe_wgrad_tn_mi16_inline_kernel<2048, false>
            <<<grid, 256, 0, stream>>>(left, right, offs, dW);
    return true;
}
