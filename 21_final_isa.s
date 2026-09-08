	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1
	.p2align	8
	.type	a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1,@function
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b64 s[12:13], s[0:1], 0xcc nv
	s_load_b64 s[4:5], s[0:1], 0x70 nv
	s_set_vgpr_msb 0x80
	v_mov_b32_e32 v194 /*v706*/, v0
	s_set_vgpr_msb 0x8000
	v_mov_b32_e32 v0, 0
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[4:5] offset:192
	s_add_co_i32 s6, s13, 0xff
	s_and_b32 s8, ttmp6, 15
	s_ashr_i32 s7, s6, 31
	s_lshl2_add_u32 s10, ttmp9, s8
	s_lshr_b32 s7, s7, 24
	s_add_co_i32 s7, s6, s7
	s_and_b32 s8, s7, 0xffffff00
	s_ashr_i32 s7, s7, 8
	s_cmp_lg_u32 s6, s8
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_and_b32 s6, s6, s8
	s_sub_co_ci_u32 s6, s7, 0
	s_add_co_i32 s7, s12, 0xff
	s_ashr_i32 s8, s7, 31
	s_lshr_b32 s8, s8, 24
	s_add_co_i32 s8, s7, s8
	s_and_b32 s9, s8, 0xffffff00
	s_ashr_i32 s8, s8, 8
	s_cmp_lg_u32 s7, s9
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_ashr_i32 s11, s10, 31
	s_and_b32 s7, s7, s9
	s_lshr_b32 s11, s11, 30
	s_add_co_i32 s11, s10, s11
	s_and_b32 s9, s11, -4
	s_ashr_i32 s12, s11, 2
	s_cmp_lg_u32 s10, s9
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s10, 0
	s_cselect_b32 s11, -1, 0
	s_and_b32 s9, s11, s9
	s_sub_co_ci_u32 s11, s12, 0
	s_ashr_i32 s14, s6, 31
	s_lshr_b32 s14, s14, 30
	s_add_co_i32 s14, s6, s14
	s_and_b32 s15, s14, -4
	s_ashr_i32 s14, s14, 2
	s_cmp_lg_u32 s6, s15
	s_cselect_b32 s15, -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_and_b32 s6, s6, s15
	s_sub_co_ci_u32 s6, s14, 0
	s_abs_i32 s18, s11
	s_lshl_b32 s14, s6, 4
	s_abs_i32 s15, s14
	s_cvt_f32_u32 s16, s15
	s_sub_co_i32 s17, 0, s15
	v_s_rcp_f32 s16, s16
	s_mul_f32 s16, s16, 0x4f7ffffe
	s_cvt_u32_f32 s16, s16
	s_mul_i32 s17, s17, s16
	s_mul_hi_u32 s17, s16, s17
	s_add_co_i32 s16, s16, s17
	s_xor_b32 s17, s11, s14
	s_mul_hi_u32 s16, s18, s16
	s_ashr_i32 s17, s17, 31
	s_mul_i32 s19, s16, s15
	s_sub_co_i32 s18, s18, s19
	s_add_co_i32 s19, s16, 1
	s_sub_co_i32 s21, s18, s15
	s_cmp_ge_u32 s18, s15
	s_cselect_b32 s16, s19, s16
	s_cselect_b32 s18, s21, s18
	s_add_co_i32 s19, s16, 1
	s_cmp_ge_u32 s18, s15
	s_cselect_b32 s15, s19, s16
	s_xor_b32 s15, s15, s17
	s_sub_co_i32 s16, s15, s17
	s_mul_i32 s16, s16, s14
	s_cmp_lg_u32 s11, s16
	s_cselect_b32 s16, -1, 0
	s_xor_b32 s6, s6, s11
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_and_b32 s6, s6, s16
	s_sub_co_ci_u32 s6, s15, s17
	s_lshl_b32 s15, s6, 4
	s_mul_i32 s6, s6, s14
	s_cmp_lg_u32 s9, 0
	s_sub_co_ci_u32 s6, s12, s6
	s_cmp_lg_u32 s7, 0
	s_sub_co_ci_u32 s7, s8, s15
	s_abs_i32 s16, s6
	s_min_i32 s8, s7, 16
	s_abs_i32 s9, s8
	s_cvt_f32_u32 s12, s9
	s_sub_co_i32 s14, 0, s9
	v_s_rcp_f32 s12, s12
	s_mul_f32 s12, s12, 0x4f7ffffe
	s_cvt_u32_f32 s12, s12
	s_mul_i32 s14, s14, s12
	s_mul_hi_u32 s14, s12, s14
	s_add_co_i32 s12, s12, s14
	s_xor_b32 s14, s6, s8
	s_mul_hi_u32 s12, s16, s12
	s_ashr_i32 s14, s14, 31
	s_mul_i32 s17, s12, s9
	s_sub_co_i32 s16, s16, s17
	s_add_co_i32 s17, s12, 1
	s_sub_co_i32 s18, s16, s9
	s_cmp_ge_u32 s16, s9
	s_cselect_b32 s12, s17, s12
	s_cselect_b32 s16, s18, s16
	s_add_co_i32 s17, s12, 1
	s_cmp_ge_u32 s16, s9
	s_cselect_b32 s9, s17, s12
	s_xor_b32 s9, s9, s14
	s_sub_co_i32 s12, s9, s14
	s_mul_i32 s12, s8, s12
	s_cmp_lg_u32 s6, s12
	s_cselect_b32 s12, -1, 0
	s_xor_b32 s7, s6, s7
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_and_b32 s7, s7, s12
	s_sub_co_ci_u32 s12, s9, s14
	s_add_co_i32 s6, s6, s15
	s_mul_i32 s7, s12, s8
	s_sub_co_i32 s6, s6, s7
	s_lshl_b32 s22, s6, 8
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s8, v1
	s_cmp_gt_i32 s8, s22
	s_cselect_b32 s6, 24, 0x48
	s_cselect_b32 s8, 0, 49
	v_mov_b32_e32 v1, s6
	s_cselect_b32 s9, 48, 0x60
	s_or_b32 s14, s6, 1
	global_load_b32 v1, v1, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s7, v1
	s_cmp_gt_i32 s7, s22
	s_cselect_b32 s7, s8, s14
	s_cselect_b32 s6, s6, s9
	s_add_co_i32 s8, s7, s6
	s_lshr_b32 s8, s8, 1
	v_mov_b32_e32 v2, s8
	s_or_b32 s14, s8, 1
	global_load_b32 v1, v2, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v1
	s_cmp_gt_i32 s9, s22
	s_cselect_b32 s14, s7, s14
	s_cselect_b32 s6, s8, s6
	s_add_co_i32 s7, s14, s6
	s_lshr_b32 s15, s7, 1
	s_mov_b32 s7, 0
	v_mov_b32_e32 v3, s15
	s_add_co_i32 s16, s15, 1
	s_mov_b32 s9, s7
	global_load_b32 v1, v3, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s8, v1
	s_cmp_gt_i32 s8, s22
	s_cselect_b32 s8, s14, s16
	s_cselect_b32 s6, s15, s6
	s_add_nc_u64 s[14:15], s[8:9], s[6:7]
	s_lshr_b64 s[14:15], s[14:15], 1
	s_lshl_b64 s[16:17], s[14:15], 2
	s_add_co_i32 s9, s14, 1
	s_add_nc_u64 s[16:17], s[4:5], s[16:17]
	global_load_b32 v0, v0, s[16:17]
	s_set_vgpr_msb 2
	v_readfirstlane_b32 s16, v194 /*v706*/
	s_set_vgpr_msb 0x200
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s7, v0
	s_cmp_gt_i32 s7, s22
	s_cselect_b32 s7, s8, s9
	s_cselect_b32 s6, s14, s6
	s_add_co_i32 s8, s7, s6
	s_lshr_b32 s8, s8, 1
	v_mov_b32_e32 v4, s8
	s_add_co_i32 s14, s8, 1
	global_load_b32 v0, v4, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v0
	s_cmp_gt_i32 s9, s22
	s_cselect_b32 s7, s7, s14
	s_cselect_b32 s6, s8, s6
	s_add_co_i32 s8, s7, s6
	s_lshr_b32 s8, s8, 1
	s_min_u32 s9, s8, 0x5f
	s_add_co_i32 s14, s8, 1
	v_mov_b32_e32 v5, s9
	global_load_b32 v0, v5, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v0
	s_cmp_gt_i32 s9, s22
	s_cselect_b32 s7, s7, s14
	s_cselect_b32 s6, s8, s6
	s_add_co_i32 s6, s7, s6
	s_lshr_b32 s6, s6, 1
	s_min_u32 s8, s6, 0x5f
	s_add_co_i32 s6, s6, 1
	v_mov_b32_e32 v6, s8
	global_load_b32 v0, v6, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s8, v0
	s_cmp_gt_i32 s8, s22
	s_cselect_b32 s14, s7, s6
	s_cmp_lt_u32 s14, 0x60
	s_cselect_b32 s6, -1, 0
	s_cmp_gt_u32 s14, 0x5f
	s_cbranch_scc1 .LBB0_47
	s_lshr_b32 s17, s16, 5
	s_ashr_i32 s23, s22, 31
	s_and_b32 s6, s6, exec_lo
	s_cselect_b32 s6, s14, 0x5f
	s_mul_u64 s[18:19], s[22:23], 0x600
	v_mov_b32_e32 v0, s6
	s_mov_b32 s20, s13
	global_load_b32 v0, v0, s[4:5] scale_offset
	s_wait_xcnt 0x0
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28 nv
	s_load_b64 s[8:9], s[0:1], 0x38 nv
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s15, v0
	s_sub_co_i32 s33, s15, s22
	s_cmp_eq_u32 s17, 0
	s_cselect_b32 s39, -1, 0
	s_cmp_lg_u32 s17, 0
	s_cbranch_scc1 .LBB0_3
	s_max_i32 s4, s33, 0
	s_mov_b32 s41, 0
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_or_b32 s43, s19, 0x80000000
	s_mov_b32 s40, 1
	s_mov_b32 s42, s18
	s_or_b32 s26, s5, 0x7fff
	s_or_b32 s27, s4, 0x800000
	s_movk_i32 s29, 0x600
	s_movk_i32 s28, 0x100
	s_mov_b32 s25, 0xffff0000
	s_mov_b32 s24, 0x730000f
	s_mov_b32 s30, s41
	s_mov_b32 s31, s41
	tensor_load_to_lds s[40:43], s[24:31]
.LBB0_3:
	s_ashr_i32 s21, s13, 31
	s_lshl_b32 s5, s10, 8
	s_lshr_b32 s4, s21, 28
	s_lshl_b32 s10, s11, 10
	s_add_co_i32 s4, s13, s4
	s_lshl_b32 s12, s12, 10
	s_ashr_i32 s4, s4, 4
	s_sub_co_i32 s10, s5, s10
	s_ashr_i32 s5, s4, 31
	s_add_co_i32 s24, s10, s12
	s_lshl_b64 s[10:11], s[4:5], 4
	s_ashr_i32 s25, s24, 31
	s_cmp_lg_u64 s[10:11], s[20:21]
	s_mov_b32 s15, 0
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s13, 0
	s_mov_b32 s11, s15
	s_cselect_b32 s38, -1, 0
	s_and_b32 s10, s38, s10
	v_cndmask_b32_e64 v0, 0, 1, s10
	v_readfirstlane_b32 s10, v0
	s_sub_nc_u64 s[4:5], s[4:5], s[10:11]
	s_lshr_b64 s[10:11], s[24:25], 4
	s_mul_u64 s[4:5], s[4:5], s[14:15]
	s_cmp_eq_u32 s17, 1
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	s_cselect_b32 s40, -1, 0
	s_mul_u64 s[4:5], s[4:5], 0x6000
	s_cmp_lg_u32 s17, 1
	s_add_nc_u64 s[36:37], s[6:7], s[4:5]
	s_mov_b32 s4, 1
	s_cbranch_scc1 .LBB0_5
	s_or_b32 s7, s37, 0x80000000
	s_mov_b32 s5, 0x9000
	s_mov_b32 s6, s36
	s_movk_i32 s49, 0x6000
	s_mov_b32 s48, 16
	s_mov_b32 s47, 0x8007fff
	s_mov_b32 s46, 0xffff7fff
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, s15
	s_mov_b32 s50, s15
	s_mov_b32 s51, s15
	tensor_load_to_lds s[4:7], s[44:51]
.LBB0_5:
	s_lshr_b64 s[4:5], s[22:23], 7
	s_cmp_eq_u32 s17, 2
	s_mul_u64 s[4:5], s[4:5], 0x3000
	s_cselect_b32 s41, -1, 0
	s_add_nc_u64 s[34:35], s[8:9], s[4:5]
	s_cmp_lg_u32 s17, 2
	s_mov_b32 s8, 2
	s_cbranch_scc1 .LBB0_7
	s_mov_b32 s10, 0
	s_or_b32 s31, s35, 0x80000000
	s_mov_b32 s29, 0x11000
	s_mov_b32 s28, 1
	s_mov_b32 s30, s34
	s_movk_i32 s9, 0xc00
	s_mov_b32 s7, 0x1007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x20000
	s_mov_b32 s11, s10
	tensor_load_to_lds s[28:31], s[4:11]
.LBB0_7:
	s_add_nc_u64 s[4:5], s[20:21], 31
	s_mov_b32 s11, 0
	s_lshr_b32 s10, s5, 27
	s_mov_b64 s[6:7], 0xffffffffffffffe0
	s_add_nc_u64 s[8:9], s[4:5], s[10:11]
	s_load_b64 s[26:27], s[0:1], 0x60 nv
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_ashr_i64 s[8:9], s[8:9], 5
	s_cmp_lg_u64 s[4:5], s[6:7]
	s_wait_xcnt 0x0
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s13, 0xffffffe1
	s_cselect_b32 s1, -1, 0
	s_and_b32 s0, s1, s0
	v_cndmask_b32_e64 v0, 0, 1, s0
	s_lshr_b64 s[0:1], s[24:25], 5
	s_cmp_eq_u32 s17, 3
	s_mul_u64 s[28:29], s[0:1], 0xc00
	v_readfirstlane_b32 s10, v0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[26:27], s[28:29]
	s_sub_nc_u64 s[0:1], s[8:9], s[10:11]
	s_mul_u64 s[0:1], s[0:1], s[14:15]
	s_cselect_b32 s14, -1, 0
	s_mul_u64 s[30:31], s[0:1], 0xc00
	s_cmp_lg_u32 s17, 3
	s_add_nc_u64 s[12:13], s[4:5], s[30:31]
	s_cbranch_scc1 .LBB0_9
	s_or_b32 s47, s13, 0x80000000
	s_mov_b32 s45, 0x11800
	s_mov_b32 s44, 1
	s_mov_b32 s46, s12
	s_movk_i32 s9, 0x300
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x407fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x20000
	s_mov_b32 s10, s11
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_9:
	v_cndmask_b32_e64 v0, 0, 1, s39
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_mov_b32 s8, 1
	v_cmp_ne_u32_e64 s5, 1, v0
	s_cbranch_vccnz .LBB0_11
	s_max_i32 s0, s33, 0
	s_add_nc_u64 s[10:11], s[18:19], 0x80
	s_mov_b32 s50, 0
	s_lshl_b32 s1, s0, 16
	s_lshr_b32 s0, s0, 16
	s_mov_b32 s9, 0x12000
	s_bitset1_b32 s11, 31
	s_or_b32 s46, s1, 0x7fff
	s_or_b32 s47, s0, 0x800000
	s_movk_i32 s49, 0x600
	s_movk_i32 s48, 0x100
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x730000f
	s_mov_b32 s51, s50
	tensor_load_to_lds s[8:11], s[44:51]
.LBB0_11:
	v_cndmask_b32_e64 v0, 0, 1, s40
	s_and_not1_b32 vcc_lo, exec_lo, s40
	v_cmp_ne_u32_e64 s4, 1, v0
	s_cbranch_vccnz .LBB0_13
	s_add_nc_u64 s[10:11], s[36:37], 0x800
	s_mov_b32 s44, 0
	s_mov_b32 s9, 0x1b000
	s_bitset1_b32 s11, 31
	s_movk_i32 s49, 0x6000
	s_mov_b32 s48, 16
	s_mov_b32 s47, 0x8007fff
	s_mov_b32 s46, 0xffff7fff
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s50, s44
	s_mov_b32 s51, s44
	tensor_load_to_lds s[8:11], s[44:51]
.LBB0_13:
	v_cndmask_b32_e64 v0, 0, 1, s41
	s_and_not1_b32 vcc_lo, exec_lo, s41
	v_cmp_ne_u32_e64 s1, 1, v0
	s_cbranch_vccnz .LBB0_15
	s_add_nc_u64 s[10:11], s[34:35], 0x400
	s_mov_b32 s46, 0
	s_mov_b32 s9, 0x23000
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0xc00
	s_mov_b32 s44, 2
	s_mov_b32 s43, 0x1007fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x20000
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
.LBB0_15:
	v_cndmask_b32_e64 v0, 0, 1, s14
	s_and_not1_b32 vcc_lo, exec_lo, s14
	v_cmp_ne_u32_e64 s0, 1, v0
	s_cbranch_vccz .LBB0_27
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB0_28
.LBB0_17:
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccz .LBB0_29
.LBB0_18:
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccz .LBB0_30
.LBB0_19:
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB0_31
.LBB0_20:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB0_32
.LBB0_21:
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccz .LBB0_33
.LBB0_22:
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccz .LBB0_34
.LBB0_23:
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_25
.LBB0_24:
	s_add_nc_u64 s[42:43], s[12:13], 0x300
	s_mov_b32 s14, 0
	s_mov_b32 s41, 0x47800
	s_bitset1_b32 s43, 31
	s_mov_b32 s40, 1
	s_movk_i32 s13, 0x300
	s_mov_b32 s12, 8
	s_mov_b32 s11, 0x407fff
	s_mov_b32 s10, 0xffff7fff
	s_mov_b32 s9, 0xffff0000
	s_mov_b32 s8, 0x20000
	s_mov_b32 s15, s14
	tensor_load_to_lds s[40:43], s[8:15]
.LBB0_25:
	s_wait_tensorcnt 0x3
	s_set_vgpr_msb 8
	s_barrier_signal -1
	v_and_b32_e32 v1, 15, v194 /*v706*/
	s_set_vgpr_msb 0x882
	v_bfe_u32 v232 /*v744*/, v194 /*v706*/, 4, 1
	s_set_vgpr_msb 0x8208
	v_and_b32_e32 v0, 31, v194 /*v706*/
	s_lshr_b32 s7, s16, 6
	s_lshl_b32 s6, s17, 7
	s_lshl_b32 s39, s7, 7
	v_lshlrev_b32_e32 v3, 8, v232 /*v744*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v4, 4, v1 :: v_dual_lshlrev_b32 v0, 2, v0
	s_and_b32 s40, s6, 0x80
	s_set_vgpr_msb 8
	v_and_b32_e32 v2, 16, v194 /*v706*/
	s_set_vgpr_msb 0x880
	v_or_b32_e32 v233 /*v745*/, s39, v1
	s_lshl_b32 s6, s40, 7
	s_set_vgpr_msb 0x8000
	v_lshl_or_b32 v1, s7, 10, v0
	s_set_vgpr_msb 0x80
	v_or3_b32 v213 /*v725*/, s6, v3, v4
	s_set_vgpr_msb 0x8000
	v_lshl_or_b32 v0, s40, 3, v0
	s_set_vgpr_msb 0x88
	v_mad_u32 v224 /*v736*/, 0x90, v233 /*v745*/, v2
	s_barrier_wait -1
	v_or_b32_e32 v225 /*v737*/, 0x9000, v213 /*v725*/
	s_set_vgpr_msb 0x8880
	v_add_nc_u32_e32 v195 /*v707*/, 0x11000, v1
	v_or_b32_e32 v212 /*v724*/, 0x11800, v0
	s_set_vgpr_msb 0x8002
	v_or_b32_e32 v4, 0x11a00, v0
	v_or_b32_e32 v1, 0x11900, v0
	v_or_b32_e32 v0, 0x11b00, v0
	ds_load_b32 v2, v212 /*v724*/
	ds_load_2addr_b32 v[6:7], v195 /*v707*/ offset1:32
	ds_load_2addr_b32 v[18:19], v195 /*v707*/ offset0:64 offset1:96
	s_set_vgpr_msb 0x282
	ds_load_b128 v[2:5] /*v[514:517]*/, v213 /*v725*/ offset:36864
	ds_load_b128 v[6:9] /*v[518:521]*/, v213 /*v725*/ offset:37376
	ds_load_b128 v[10:13] /*v[522:525]*/, v213 /*v725*/ offset:38912
	ds_load_b128 v[14:17] /*v[526:529]*/, v213 /*v725*/ offset:39424
	ds_load_b128 v[18:21] /*v[530:533]*/, v213 /*v725*/ offset:40960
	ds_load_b128 v[22:25] /*v[534:537]*/, v213 /*v725*/ offset:41472
	ds_load_b128 v[26:29] /*v[538:541]*/, v213 /*v725*/ offset:43008
	ds_load_b128 v[30:33] /*v[542:545]*/, v213 /*v725*/ offset:43520
	s_set_vgpr_msb 0x8200
	ds_load_b32 v3, v1
	ds_load_b32 v4, v4
	ds_load_b32 v5, v0
	s_set_vgpr_msb 0x82
	ds_load_b128 v[34:37] /*v[546:549]*/, v213 /*v725*/ offset:45056
	ds_load_b128 v[38:41] /*v[550:553]*/, v213 /*v725*/ offset:45568
	ds_load_b128 v[42:45] /*v[554:557]*/, v213 /*v725*/ offset:47104
	ds_load_b128 v[46:49] /*v[558:561]*/, v213 /*v725*/ offset:47616
	ds_load_b128 v[50:53] /*v[562:565]*/, v213 /*v725*/ offset:49152
	ds_load_b128 v[54:57] /*v[566:569]*/, v213 /*v725*/ offset:49664
	ds_load_b128 v[58:61] /*v[570:573]*/, v213 /*v725*/ offset:51200
	ds_load_b128 v[62:65] /*v[574:577]*/, v213 /*v725*/ offset:51712
	ds_load_b128 v[186:189] /*v[698:701]*/, v224 /*v736*/
	ds_load_b128 v[190:193] /*v[702:705]*/, v224 /*v736*/ offset:32
	ds_load_b128 v[178:181] /*v[690:693]*/, v224 /*v736*/ offset:2304
	ds_load_b128 v[182:185] /*v[694:697]*/, v224 /*v736*/ offset:2336
	ds_load_b128 v[138:141] /*v[650:653]*/, v224 /*v736*/ offset:4608
	ds_load_b128 v[142:145] /*v[654:657]*/, v224 /*v736*/ offset:4640
	ds_load_b128 v[130:133] /*v[642:645]*/, v224 /*v736*/ offset:6912
	ds_load_b128 v[134:137] /*v[646:649]*/, v224 /*v736*/ offset:6944
	ds_load_b128 v[170:173] /*v[682:685]*/, v224 /*v736*/ offset:9216
	ds_load_b128 v[174:177] /*v[686:689]*/, v224 /*v736*/ offset:9248
	ds_load_b128 v[162:165] /*v[674:677]*/, v224 /*v736*/ offset:11520
	ds_load_b128 v[166:169] /*v[678:681]*/, v224 /*v736*/ offset:11552
	ds_load_b128 v[154:157] /*v[666:669]*/, v224 /*v736*/ offset:13824
	ds_load_b128 v[158:161] /*v[670:673]*/, v224 /*v736*/ offset:13856
	ds_load_b128 v[146:149] /*v[658:661]*/, v224 /*v736*/ offset:16128
	ds_load_b128 v[150:153] /*v[662:665]*/, v224 /*v736*/ offset:16160
	s_mov_b32 s14, 0
	s_and_b32 vcc_lo, exec_lo, s5
	s_mov_b32 s16, 1
	s_set_vgpr_msb 0x8200
	s_cbranch_vccz .LBB0_35
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v196 /*v708*/, 0
	v_dual_mov_b32 v197 /*v709*/, v196 /*v708*/ :: v_dual_mov_b32 v198 /*v710*/, v196 /*v708*/
	v_dual_mov_b32 v209 /*v721*/, v196 /*v708*/ :: v_dual_mov_b32 v210 /*v722*/, v196 /*v708*/
	v_dual_mov_b32 v211 /*v723*/, v196 /*v708*/ :: v_dual_mov_b32 v199 /*v711*/, v196 /*v708*/
	v_dual_mov_b32 v200 /*v712*/, v196 /*v708*/ :: v_dual_mov_b32 v201 /*v713*/, v196 /*v708*/
	v_dual_mov_b32 v202 /*v714*/, v196 /*v708*/ :: v_dual_mov_b32 v203 /*v715*/, v196 /*v708*/
	v_dual_mov_b32 v204 /*v716*/, v196 /*v708*/ :: v_dual_mov_b32 v205 /*v717*/, v196 /*v708*/
	v_dual_mov_b32 v206 /*v718*/, v196 /*v708*/ :: v_dual_mov_b32 v207 /*v719*/, v196 /*v708*/
	v_mov_b32_e32 v208 /*v720*/, v196 /*v708*/
	v_mov_b64_e32 v[248:249] /*v[760:761]*/, v[210:211] /*v[722:723]*/
	s_set_vgpr_msb 0x8202
	v_mov_b64_e32 v[50:51], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[66:67], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[34:35], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[82:83], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[98:99], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[114:115], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[130:131], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[146:147], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[162:163], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[178:179], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[194:195], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[210:211], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[226:227], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[242:243], v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x242
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[18:19] /*v[274:275]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x4282
	v_mov_b64_e32 v[246:247] /*v[758:759]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[244:245] /*v[756:757]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[242:243] /*v[754:755]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[240:241] /*v[752:753]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[238:239] /*v[750:751]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[236:237] /*v[748:749]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[234:235] /*v[746:747]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x8202
	v_mov_b64_e32 v[52:53], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[54:55], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[56:57], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[58:59], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[60:61], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[62:63], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[64:65], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[68:69], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[70:71], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[72:73], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[74:75], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[76:77], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[78:79], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[80:81], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[36:37], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[38:39], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[40:41], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[42:43], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[44:45], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[46:47], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[48:49], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[84:85], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[86:87], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[88:89], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[90:91], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[92:93], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[94:95], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[96:97], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[100:101], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[102:103], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[104:105], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[106:107], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[108:109], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[110:111], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[112:113], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[116:117], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[118:119], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[120:121], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[122:123], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[124:125], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[126:127], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[128:129], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[132:133], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[134:135], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[136:137], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[138:139], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[140:141], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[142:143], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[144:145], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[148:149], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[150:151], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[152:153], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[154:155], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[156:157], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[158:159], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[160:161], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[164:165], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[166:167], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[168:169], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[170:171], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[172:173], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[174:175], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[176:177], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[180:181], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[182:183], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[184:185], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[186:187], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[188:189], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[190:191], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[192:193], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[196:197], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[198:199], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[200:201], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[202:203], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[204:205], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[206:207], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[208:209], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[212:213], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[214:215], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[216:217], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[218:219], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[220:221], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[222:223], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[224:225], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[228:229], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[230:231], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[232:233], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[234:235], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[236:237], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[238:239], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[240:241], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[244:245], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[246:247], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[248:249], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[250:251], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[252:253], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[254:255], v[208:209] /*v[720:721]*/
	s_set_vgpr_msb 0x242
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[208:209] /*v[720:721]*/
	s_set_vgpr_msb 0x4282
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[210:211] /*v[722:723]*/
	s_set_vgpr_msb 0x8242
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[210:211] /*v[722:723]*/
	s_set_vgpr_msb 0x4200
	s_branch .LBB0_37
.LBB0_27:
	s_add_nc_u64 s[10:11], s[12:13], 0x100
	s_mov_b32 s46, 0
	s_mov_b32 s9, 0x23800
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0x300
	s_mov_b32 s44, 8
	s_mov_b32 s43, 0x407fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x20000
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_17
.LBB0_28:
	s_max_i32 s6, s33, 0
	s_add_nc_u64 s[10:11], s[18:19], 0x100
	s_mov_b32 s46, 0
	s_lshl_b32 s7, s6, 16
	s_lshr_b32 s6, s6, 16
	s_mov_b32 s9, 0x24000
	s_bitset1_b32 s11, 31
	s_or_b32 s42, s7, 0x7fff
	s_or_b32 s43, s6, 0x800000
	s_movk_i32 s45, 0x600
	s_movk_i32 s44, 0x100
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x730000f
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB0_18
.LBB0_29:
	s_add_nc_u64 s[10:11], s[36:37], 0x1000
	s_mov_b32 s40, 0
	s_mov_b32 s9, 0x2d000
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0x6000
	s_mov_b32 s44, 16
	s_mov_b32 s43, 0x8007fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s46, s40
	s_mov_b32 s47, s40
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_19
.LBB0_30:
	s_add_nc_u64 s[10:11], s[34:35], 0x800
	s_mov_b32 s46, 0
	s_mov_b32 s9, 0x35000
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0xc00
	s_mov_b32 s44, 2
	s_mov_b32 s43, 0x1007fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x20000
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_20
.LBB0_31:
	s_add_nc_u64 s[10:11], s[12:13], 0x200
	s_mov_b32 s46, 0
	s_mov_b32 s9, 0x35800
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0x300
	s_mov_b32 s44, 8
	s_mov_b32 s43, 0x407fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x20000
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_21
.LBB0_32:
	s_max_i32 s6, s33, 0
	s_add_nc_u64 s[10:11], s[18:19], 0x180
	s_mov_b32 s46, 0
	s_lshl_b32 s7, s6, 16
	s_lshr_b32 s6, s6, 16
	s_mov_b32 s9, 0x36000
	s_bitset1_b32 s11, 31
	s_or_b32 s42, s7, 0x7fff
	s_or_b32 s43, s6, 0x800000
	s_movk_i32 s45, 0x600
	s_movk_i32 s44, 0x100
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x730000f
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB0_22
.LBB0_33:
	s_add_nc_u64 s[10:11], s[36:37], 0x1800
	s_mov_b32 s40, 0
	s_mov_b32 s9, 0x3f000
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0x6000
	s_mov_b32 s44, 16
	s_mov_b32 s43, 0x8007fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s46, s40
	s_mov_b32 s47, s40
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_23
.LBB0_34:
	s_add_nc_u64 s[10:11], s[34:35], 0xc00
	s_mov_b32 s46, 0
	s_mov_b32 s9, 0x47000
	s_bitset1_b32 s11, 31
	s_movk_i32 s45, 0xc00
	s_mov_b32 s44, 2
	s_mov_b32 s43, 0x1007fff
	s_mov_b32 s42, 0xffff7fff
	s_mov_b32 s41, 0xffff0000
	s_mov_b32 s40, 0x20000
	s_mov_b32 s47, s46
	tensor_load_to_lds s[8:11], s[40:47]
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB0_24
	s_branch .LBB0_25
.LBB0_35:
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v194 /*v450*/, 0
	s_max_i32 s5, s33, 0
	s_movk_i32 s13, 0x600
	s_lshl_b32 s6, s5, 16
	s_movk_i32 s12, 0x100
	v_mov_b32_e32 v195 /*v451*/, v194 /*v450*/
	s_set_vgpr_msb 0x4101
	v_mov_b32_e32 v247, v194 /*v450*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v196 /*v452*/, v194 /*v450*/ :: v_dual_mov_b32 v197 /*v453*/, v194 /*v450*/
	v_dual_mov_b32 v198 /*v454*/, v194 /*v450*/ :: v_dual_mov_b32 v199 /*v455*/, v194 /*v450*/
	v_dual_mov_b32 v200 /*v456*/, v194 /*v450*/ :: v_dual_mov_b32 v201 /*v457*/, v194 /*v450*/
	v_dual_mov_b32 v202 /*v458*/, v194 /*v450*/ :: v_dual_mov_b32 v203 /*v459*/, v194 /*v450*/
	v_dual_mov_b32 v204 /*v460*/, v194 /*v450*/ :: v_dual_mov_b32 v205 /*v461*/, v194 /*v450*/
	v_dual_mov_b32 v206 /*v462*/, v194 /*v450*/ :: v_dual_mov_b32 v207 /*v463*/, v194 /*v450*/
	v_dual_mov_b32 v208 /*v464*/, v194 /*v450*/ :: v_dual_mov_b32 v209 /*v465*/, v194 /*v450*/
	v_dual_mov_b32 v242 /*v498*/, v194 /*v450*/ :: v_dual_mov_b32 v243 /*v499*/, v194 /*v450*/
	v_dual_mov_b32 v244 /*v500*/, v194 /*v450*/ :: v_dual_mov_b32 v245 /*v501*/, v194 /*v450*/
	v_dual_mov_b32 v246 /*v502*/, v194 /*v450*/ :: v_dual_mov_b32 v247 /*v503*/, v194 /*v450*/
	v_dual_mov_b32 v248 /*v504*/, v194 /*v450*/ :: v_dual_mov_b32 v249 /*v505*/, v194 /*v450*/
	v_dual_mov_b32 v250 /*v506*/, v194 /*v450*/ :: v_dual_mov_b32 v251 /*v507*/, v194 /*v450*/
	v_dual_mov_b32 v252 /*v508*/, v194 /*v450*/ :: v_dual_mov_b32 v253 /*v509*/, v194 /*v450*/
	v_dual_mov_b32 v254 /*v510*/, v194 /*v450*/ :: v_dual_mov_b32 v255 /*v511*/, v194 /*v450*/
	s_set_vgpr_msb 0x4181
	v_dual_mov_b32 v0 /*v512*/, v194 /*v450*/ :: v_dual_mov_b32 v1 /*v513*/, v194 /*v450*/
	s_set_vgpr_msb 0x8141
	v_dual_mov_b32 v226 /*v482*/, v194 /*v450*/ :: v_dual_mov_b32 v227 /*v483*/, v194 /*v450*/
	v_dual_mov_b32 v228 /*v484*/, v194 /*v450*/ :: v_dual_mov_b32 v229 /*v485*/, v194 /*v450*/
	v_dual_mov_b32 v230 /*v486*/, v194 /*v450*/ :: v_dual_mov_b32 v231 /*v487*/, v194 /*v450*/
	v_dual_mov_b32 v232 /*v488*/, v194 /*v450*/ :: v_dual_mov_b32 v233 /*v489*/, v194 /*v450*/
	v_dual_mov_b32 v234 /*v490*/, v194 /*v450*/ :: v_dual_mov_b32 v235 /*v491*/, v194 /*v450*/
	v_dual_mov_b32 v236 /*v492*/, v194 /*v450*/ :: v_dual_mov_b32 v237 /*v493*/, v194 /*v450*/
	v_dual_mov_b32 v238 /*v494*/, v194 /*v450*/ :: v_dual_mov_b32 v239 /*v495*/, v194 /*v450*/
	v_dual_mov_b32 v240 /*v496*/, v194 /*v450*/ :: v_dual_mov_b32 v241 /*v497*/, v194 /*v450*/
	v_dual_mov_b32 v210 /*v466*/, v194 /*v450*/ :: v_dual_mov_b32 v211 /*v467*/, v194 /*v450*/
	v_dual_mov_b32 v212 /*v468*/, v194 /*v450*/ :: v_dual_mov_b32 v213 /*v469*/, v194 /*v450*/
	v_dual_mov_b32 v214 /*v470*/, v194 /*v450*/ :: v_dual_mov_b32 v215 /*v471*/, v194 /*v450*/
	v_dual_mov_b32 v216 /*v472*/, v194 /*v450*/ :: v_dual_mov_b32 v217 /*v473*/, v194 /*v450*/
	v_dual_mov_b32 v218 /*v474*/, v194 /*v450*/ :: v_dual_mov_b32 v219 /*v475*/, v194 /*v450*/
	v_dual_mov_b32 v220 /*v476*/, v194 /*v450*/ :: v_dual_mov_b32 v221 /*v477*/, v194 /*v450*/
	v_dual_mov_b32 v222 /*v478*/, v194 /*v450*/ :: v_dual_mov_b32 v223 /*v479*/, v194 /*v450*/
	v_dual_mov_b32 v224 /*v480*/, v194 /*v450*/ :: v_dual_mov_b32 v225 /*v481*/, v194 /*v450*/
	v_dual_mov_b32 v178 /*v434*/, v194 /*v450*/ :: v_dual_mov_b32 v179 /*v435*/, v194 /*v450*/
	v_dual_mov_b32 v180 /*v436*/, v194 /*v450*/ :: v_dual_mov_b32 v181 /*v437*/, v194 /*v450*/
	v_dual_mov_b32 v182 /*v438*/, v194 /*v450*/ :: v_dual_mov_b32 v183 /*v439*/, v194 /*v450*/
	v_dual_mov_b32 v184 /*v440*/, v194 /*v450*/ :: v_dual_mov_b32 v185 /*v441*/, v194 /*v450*/
	v_dual_mov_b32 v186 /*v442*/, v194 /*v450*/ :: v_dual_mov_b32 v187 /*v443*/, v194 /*v450*/
	v_dual_mov_b32 v188 /*v444*/, v194 /*v450*/ :: v_dual_mov_b32 v189 /*v445*/, v194 /*v450*/
	v_dual_mov_b32 v190 /*v446*/, v194 /*v450*/ :: v_dual_mov_b32 v191 /*v447*/, v194 /*v450*/
	v_dual_mov_b32 v192 /*v448*/, v194 /*v450*/ :: v_dual_mov_b32 v193 /*v449*/, v194 /*v450*/
	v_dual_mov_b32 v162 /*v418*/, v194 /*v450*/ :: v_dual_mov_b32 v163 /*v419*/, v194 /*v450*/
	v_dual_mov_b32 v164 /*v420*/, v194 /*v450*/ :: v_dual_mov_b32 v165 /*v421*/, v194 /*v450*/
	v_dual_mov_b32 v166 /*v422*/, v194 /*v450*/ :: v_dual_mov_b32 v167 /*v423*/, v194 /*v450*/
	v_dual_mov_b32 v168 /*v424*/, v194 /*v450*/ :: v_dual_mov_b32 v169 /*v425*/, v194 /*v450*/
	v_dual_mov_b32 v170 /*v426*/, v194 /*v450*/ :: v_dual_mov_b32 v171 /*v427*/, v194 /*v450*/
	v_dual_mov_b32 v172 /*v428*/, v194 /*v450*/ :: v_dual_mov_b32 v173 /*v429*/, v194 /*v450*/
	v_dual_mov_b32 v174 /*v430*/, v194 /*v450*/ :: v_dual_mov_b32 v175 /*v431*/, v194 /*v450*/
	v_dual_mov_b32 v176 /*v432*/, v194 /*v450*/ :: v_dual_mov_b32 v177 /*v433*/, v194 /*v450*/
	v_dual_mov_b32 v146 /*v402*/, v194 /*v450*/ :: v_dual_mov_b32 v147 /*v403*/, v194 /*v450*/
	v_dual_mov_b32 v148 /*v404*/, v194 /*v450*/ :: v_dual_mov_b32 v149 /*v405*/, v194 /*v450*/
	v_dual_mov_b32 v150 /*v406*/, v194 /*v450*/ :: v_dual_mov_b32 v151 /*v407*/, v194 /*v450*/
	v_dual_mov_b32 v152 /*v408*/, v194 /*v450*/ :: v_dual_mov_b32 v153 /*v409*/, v194 /*v450*/
	v_dual_mov_b32 v154 /*v410*/, v194 /*v450*/ :: v_dual_mov_b32 v155 /*v411*/, v194 /*v450*/
	v_dual_mov_b32 v156 /*v412*/, v194 /*v450*/ :: v_dual_mov_b32 v157 /*v413*/, v194 /*v450*/
	v_dual_mov_b32 v158 /*v414*/, v194 /*v450*/ :: v_dual_mov_b32 v159 /*v415*/, v194 /*v450*/
	v_dual_mov_b32 v160 /*v416*/, v194 /*v450*/ :: v_dual_mov_b32 v161 /*v417*/, v194 /*v450*/
	v_dual_mov_b32 v130 /*v386*/, v194 /*v450*/ :: v_dual_mov_b32 v131 /*v387*/, v194 /*v450*/
	v_dual_mov_b32 v132 /*v388*/, v194 /*v450*/ :: v_dual_mov_b32 v133 /*v389*/, v194 /*v450*/
	v_dual_mov_b32 v134 /*v390*/, v194 /*v450*/ :: v_dual_mov_b32 v135 /*v391*/, v194 /*v450*/
	v_dual_mov_b32 v136 /*v392*/, v194 /*v450*/ :: v_dual_mov_b32 v137 /*v393*/, v194 /*v450*/
	v_dual_mov_b32 v138 /*v394*/, v194 /*v450*/ :: v_dual_mov_b32 v139 /*v395*/, v194 /*v450*/
	v_dual_mov_b32 v140 /*v396*/, v194 /*v450*/ :: v_dual_mov_b32 v141 /*v397*/, v194 /*v450*/
	v_dual_mov_b32 v142 /*v398*/, v194 /*v450*/ :: v_dual_mov_b32 v143 /*v399*/, v194 /*v450*/
	v_dual_mov_b32 v144 /*v400*/, v194 /*v450*/ :: v_dual_mov_b32 v145 /*v401*/, v194 /*v450*/
	v_dual_mov_b32 v114 /*v370*/, v194 /*v450*/ :: v_dual_mov_b32 v115 /*v371*/, v194 /*v450*/
	v_dual_mov_b32 v116 /*v372*/, v194 /*v450*/ :: v_dual_mov_b32 v117 /*v373*/, v194 /*v450*/
	v_dual_mov_b32 v118 /*v374*/, v194 /*v450*/ :: v_dual_mov_b32 v119 /*v375*/, v194 /*v450*/
	v_dual_mov_b32 v120 /*v376*/, v194 /*v450*/ :: v_dual_mov_b32 v121 /*v377*/, v194 /*v450*/
	v_dual_mov_b32 v122 /*v378*/, v194 /*v450*/ :: v_dual_mov_b32 v123 /*v379*/, v194 /*v450*/
	v_dual_mov_b32 v124 /*v380*/, v194 /*v450*/ :: v_dual_mov_b32 v125 /*v381*/, v194 /*v450*/
	v_dual_mov_b32 v126 /*v382*/, v194 /*v450*/ :: v_dual_mov_b32 v127 /*v383*/, v194 /*v450*/
	v_dual_mov_b32 v128 /*v384*/, v194 /*v450*/ :: v_dual_mov_b32 v129 /*v385*/, v194 /*v450*/
	v_dual_mov_b32 v98 /*v354*/, v194 /*v450*/ :: v_dual_mov_b32 v99 /*v355*/, v194 /*v450*/
	v_dual_mov_b32 v100 /*v356*/, v194 /*v450*/ :: v_dual_mov_b32 v101 /*v357*/, v194 /*v450*/
	v_dual_mov_b32 v102 /*v358*/, v194 /*v450*/ :: v_dual_mov_b32 v103 /*v359*/, v194 /*v450*/
	v_dual_mov_b32 v104 /*v360*/, v194 /*v450*/ :: v_dual_mov_b32 v105 /*v361*/, v194 /*v450*/
	v_dual_mov_b32 v106 /*v362*/, v194 /*v450*/ :: v_dual_mov_b32 v107 /*v363*/, v194 /*v450*/
	v_dual_mov_b32 v108 /*v364*/, v194 /*v450*/ :: v_dual_mov_b32 v109 /*v365*/, v194 /*v450*/
	v_dual_mov_b32 v110 /*v366*/, v194 /*v450*/ :: v_dual_mov_b32 v111 /*v367*/, v194 /*v450*/
	v_dual_mov_b32 v112 /*v368*/, v194 /*v450*/ :: v_dual_mov_b32 v113 /*v369*/, v194 /*v450*/
	v_dual_mov_b32 v82 /*v338*/, v194 /*v450*/ :: v_dual_mov_b32 v83 /*v339*/, v194 /*v450*/
	v_dual_mov_b32 v84 /*v340*/, v194 /*v450*/ :: v_dual_mov_b32 v85 /*v341*/, v194 /*v450*/
	v_dual_mov_b32 v86 /*v342*/, v194 /*v450*/ :: v_dual_mov_b32 v87 /*v343*/, v194 /*v450*/
	v_dual_mov_b32 v88 /*v344*/, v194 /*v450*/ :: v_dual_mov_b32 v89 /*v345*/, v194 /*v450*/
	v_dual_mov_b32 v90 /*v346*/, v194 /*v450*/ :: v_dual_mov_b32 v91 /*v347*/, v194 /*v450*/
	v_dual_mov_b32 v92 /*v348*/, v194 /*v450*/ :: v_dual_mov_b32 v93 /*v349*/, v194 /*v450*/
	v_dual_mov_b32 v94 /*v350*/, v194 /*v450*/ :: v_dual_mov_b32 v95 /*v351*/, v194 /*v450*/
	v_dual_mov_b32 v96 /*v352*/, v194 /*v450*/ :: v_dual_mov_b32 v97 /*v353*/, v194 /*v450*/
	v_dual_mov_b32 v66 /*v322*/, v194 /*v450*/ :: v_dual_mov_b32 v67 /*v323*/, v194 /*v450*/
	v_dual_mov_b32 v68 /*v324*/, v194 /*v450*/ :: v_dual_mov_b32 v69 /*v325*/, v194 /*v450*/
	v_dual_mov_b32 v70 /*v326*/, v194 /*v450*/ :: v_dual_mov_b32 v71 /*v327*/, v194 /*v450*/
	v_dual_mov_b32 v72 /*v328*/, v194 /*v450*/ :: v_dual_mov_b32 v73 /*v329*/, v194 /*v450*/
	v_dual_mov_b32 v74 /*v330*/, v194 /*v450*/ :: v_dual_mov_b32 v75 /*v331*/, v194 /*v450*/
	v_dual_mov_b32 v76 /*v332*/, v194 /*v450*/ :: v_dual_mov_b32 v77 /*v333*/, v194 /*v450*/
	v_dual_mov_b32 v78 /*v334*/, v194 /*v450*/ :: v_dual_mov_b32 v79 /*v335*/, v194 /*v450*/
	v_dual_mov_b32 v80 /*v336*/, v194 /*v450*/ :: v_dual_mov_b32 v81 /*v337*/, v194 /*v450*/
	v_dual_mov_b32 v50 /*v306*/, v194 /*v450*/ :: v_dual_mov_b32 v51 /*v307*/, v194 /*v450*/
	v_dual_mov_b32 v52 /*v308*/, v194 /*v450*/ :: v_dual_mov_b32 v53 /*v309*/, v194 /*v450*/
	v_dual_mov_b32 v54 /*v310*/, v194 /*v450*/ :: v_dual_mov_b32 v55 /*v311*/, v194 /*v450*/
	v_dual_mov_b32 v56 /*v312*/, v194 /*v450*/ :: v_dual_mov_b32 v57 /*v313*/, v194 /*v450*/
	v_dual_mov_b32 v58 /*v314*/, v194 /*v450*/ :: v_dual_mov_b32 v59 /*v315*/, v194 /*v450*/
	v_dual_mov_b32 v60 /*v316*/, v194 /*v450*/ :: v_dual_mov_b32 v61 /*v317*/, v194 /*v450*/
	v_dual_mov_b32 v62 /*v318*/, v194 /*v450*/ :: v_dual_mov_b32 v63 /*v319*/, v194 /*v450*/
	v_dual_mov_b32 v64 /*v320*/, v194 /*v450*/ :: v_dual_mov_b32 v65 /*v321*/, v194 /*v450*/
	v_dual_mov_b32 v34 /*v290*/, v194 /*v450*/ :: v_dual_mov_b32 v35 /*v291*/, v194 /*v450*/
	v_dual_mov_b32 v36 /*v292*/, v194 /*v450*/ :: v_dual_mov_b32 v37 /*v293*/, v194 /*v450*/
	v_dual_mov_b32 v38 /*v294*/, v194 /*v450*/ :: v_dual_mov_b32 v39 /*v295*/, v194 /*v450*/
	v_dual_mov_b32 v40 /*v296*/, v194 /*v450*/ :: v_dual_mov_b32 v41 /*v297*/, v194 /*v450*/
	v_dual_mov_b32 v42 /*v298*/, v194 /*v450*/ :: v_dual_mov_b32 v43 /*v299*/, v194 /*v450*/
	v_dual_mov_b32 v44 /*v300*/, v194 /*v450*/ :: v_dual_mov_b32 v45 /*v301*/, v194 /*v450*/
	v_dual_mov_b32 v46 /*v302*/, v194 /*v450*/ :: v_dual_mov_b32 v47 /*v303*/, v194 /*v450*/
	v_dual_mov_b32 v48 /*v304*/, v194 /*v450*/ :: v_dual_mov_b32 v49 /*v305*/, v194 /*v450*/
	v_dual_mov_b32 v18 /*v274*/, v194 /*v450*/ :: v_dual_mov_b32 v19 /*v275*/, v194 /*v450*/
	v_dual_mov_b32 v20 /*v276*/, v194 /*v450*/ :: v_dual_mov_b32 v21 /*v277*/, v194 /*v450*/
	v_dual_mov_b32 v22 /*v278*/, v194 /*v450*/ :: v_dual_mov_b32 v23 /*v279*/, v194 /*v450*/
	v_dual_mov_b32 v24 /*v280*/, v194 /*v450*/ :: v_dual_mov_b32 v25 /*v281*/, v194 /*v450*/
	v_dual_mov_b32 v26 /*v282*/, v194 /*v450*/ :: v_dual_mov_b32 v27 /*v283*/, v194 /*v450*/
	v_dual_mov_b32 v28 /*v284*/, v194 /*v450*/ :: v_dual_mov_b32 v29 /*v285*/, v194 /*v450*/
	v_dual_mov_b32 v30 /*v286*/, v194 /*v450*/ :: v_dual_mov_b32 v31 /*v287*/, v194 /*v450*/
	v_dual_mov_b32 v32 /*v288*/, v194 /*v450*/ :: v_dual_mov_b32 v33 /*v289*/, v194 /*v450*/
	v_dual_mov_b32 v2 /*v258*/, v194 /*v450*/ :: v_dual_mov_b32 v3 /*v259*/, v194 /*v450*/
	v_dual_mov_b32 v4 /*v260*/, v194 /*v450*/ :: v_dual_mov_b32 v5 /*v261*/, v194 /*v450*/
	v_dual_mov_b32 v6 /*v262*/, v194 /*v450*/ :: v_dual_mov_b32 v7 /*v263*/, v194 /*v450*/
	v_dual_mov_b32 v8 /*v264*/, v194 /*v450*/ :: v_dual_mov_b32 v16 /*v272*/, v194 /*v450*/
	v_dual_mov_b32 v15 /*v271*/, v194 /*v450*/ :: v_dual_mov_b32 v14 /*v270*/, v194 /*v450*/
	v_dual_mov_b32 v13 /*v269*/, v194 /*v450*/ :: v_dual_mov_b32 v12 /*v268*/, v194 /*v450*/
	v_dual_mov_b32 v11 /*v267*/, v194 /*v450*/ :: v_dual_mov_b32 v10 /*v266*/, v194 /*v450*/
	v_dual_mov_b32 v9 /*v265*/, v194 /*v450*/ :: v_dual_mov_b32 v17 /*v273*/, v194 /*v450*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v242, v194 /*v450*/ :: v_dual_mov_b32 v243, v194 /*v450*/
	v_dual_mov_b32 v244, v194 /*v450*/ :: v_dual_mov_b32 v245, v194 /*v450*/
	v_dual_mov_b32 v246, v194 /*v450*/ :: v_dual_mov_b32 v248, v194 /*v450*/
	v_dual_mov_b32 v249, v194 /*v450*/ :: v_dual_mov_b32 v250, v194 /*v450*/
	v_dual_mov_b32 v251, v194 /*v450*/ :: v_dual_mov_b32 v252, v194 /*v450*/
	v_dual_mov_b32 v253, v194 /*v450*/ :: v_dual_mov_b32 v254, v194 /*v450*/
	v_mov_b32_e32 v255, v194 /*v450*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v0 /*v256*/, v194 /*v450*/ :: v_dual_mov_b32 v1 /*v257*/, v194 /*v450*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v226, v194 /*v450*/ :: v_dual_mov_b32 v227, v194 /*v450*/
	v_dual_mov_b32 v228, v194 /*v450*/ :: v_dual_mov_b32 v229, v194 /*v450*/
	v_dual_mov_b32 v230, v194 /*v450*/ :: v_dual_mov_b32 v231, v194 /*v450*/
	v_dual_mov_b32 v232, v194 /*v450*/ :: v_dual_mov_b32 v233, v194 /*v450*/
	v_dual_mov_b32 v234, v194 /*v450*/ :: v_dual_mov_b32 v235, v194 /*v450*/
	v_dual_mov_b32 v236, v194 /*v450*/ :: v_dual_mov_b32 v237, v194 /*v450*/
	v_dual_mov_b32 v238, v194 /*v450*/ :: v_dual_mov_b32 v239, v194 /*v450*/
	v_dual_mov_b32 v240, v194 /*v450*/ :: v_dual_mov_b32 v241, v194 /*v450*/
	v_dual_mov_b32 v210, v194 /*v450*/ :: v_dual_mov_b32 v211, v194 /*v450*/
	v_dual_mov_b32 v212, v194 /*v450*/ :: v_dual_mov_b32 v213, v194 /*v450*/
	v_dual_mov_b32 v214, v194 /*v450*/ :: v_dual_mov_b32 v215, v194 /*v450*/
	v_dual_mov_b32 v216, v194 /*v450*/ :: v_dual_mov_b32 v217, v194 /*v450*/
	v_dual_mov_b32 v218, v194 /*v450*/ :: v_dual_mov_b32 v219, v194 /*v450*/
	v_dual_mov_b32 v220, v194 /*v450*/ :: v_dual_mov_b32 v221, v194 /*v450*/
	v_dual_mov_b32 v222, v194 /*v450*/ :: v_dual_mov_b32 v223, v194 /*v450*/
	v_dual_mov_b32 v224, v194 /*v450*/ :: v_dual_mov_b32 v225, v194 /*v450*/
	v_dual_mov_b32 v194, v194 /*v450*/ :: v_dual_mov_b32 v195, v194 /*v450*/
	v_dual_mov_b32 v196, v194 /*v450*/ :: v_dual_mov_b32 v197, v194 /*v450*/
	v_dual_mov_b32 v198, v194 /*v450*/ :: v_dual_mov_b32 v199, v194 /*v450*/
	v_dual_mov_b32 v200, v194 /*v450*/ :: v_dual_mov_b32 v201, v194 /*v450*/
	v_dual_mov_b32 v202, v194 /*v450*/ :: v_dual_mov_b32 v203, v194 /*v450*/
	v_dual_mov_b32 v204, v194 /*v450*/ :: v_dual_mov_b32 v205, v194 /*v450*/
	v_dual_mov_b32 v206, v194 /*v450*/ :: v_dual_mov_b32 v207, v194 /*v450*/
	v_dual_mov_b32 v208, v194 /*v450*/ :: v_dual_mov_b32 v209, v194 /*v450*/
	v_dual_mov_b32 v178, v194 /*v450*/ :: v_dual_mov_b32 v179, v194 /*v450*/
	v_dual_mov_b32 v180, v194 /*v450*/ :: v_dual_mov_b32 v181, v194 /*v450*/
	v_dual_mov_b32 v182, v194 /*v450*/ :: v_dual_mov_b32 v183, v194 /*v450*/
	v_dual_mov_b32 v184, v194 /*v450*/ :: v_dual_mov_b32 v185, v194 /*v450*/
	v_dual_mov_b32 v186, v194 /*v450*/ :: v_dual_mov_b32 v187, v194 /*v450*/
	v_dual_mov_b32 v188, v194 /*v450*/ :: v_dual_mov_b32 v189, v194 /*v450*/
	v_dual_mov_b32 v190, v194 /*v450*/ :: v_dual_mov_b32 v191, v194 /*v450*/
	v_dual_mov_b32 v192, v194 /*v450*/ :: v_dual_mov_b32 v193, v194 /*v450*/
	v_dual_mov_b32 v162, v194 /*v450*/ :: v_dual_mov_b32 v163, v194 /*v450*/
	v_dual_mov_b32 v164, v194 /*v450*/ :: v_dual_mov_b32 v165, v194 /*v450*/
	v_dual_mov_b32 v166, v194 /*v450*/ :: v_dual_mov_b32 v167, v194 /*v450*/
	v_dual_mov_b32 v168, v194 /*v450*/ :: v_dual_mov_b32 v169, v194 /*v450*/
	v_dual_mov_b32 v170, v194 /*v450*/ :: v_dual_mov_b32 v171, v194 /*v450*/
	v_dual_mov_b32 v172, v194 /*v450*/ :: v_dual_mov_b32 v173, v194 /*v450*/
	v_dual_mov_b32 v174, v194 /*v450*/ :: v_dual_mov_b32 v175, v194 /*v450*/
	v_dual_mov_b32 v176, v194 /*v450*/ :: v_dual_mov_b32 v177, v194 /*v450*/
	v_dual_mov_b32 v146, v194 /*v450*/ :: v_dual_mov_b32 v147, v194 /*v450*/
	v_dual_mov_b32 v148, v194 /*v450*/ :: v_dual_mov_b32 v149, v194 /*v450*/
	v_dual_mov_b32 v150, v194 /*v450*/ :: v_dual_mov_b32 v151, v194 /*v450*/
	v_dual_mov_b32 v152, v194 /*v450*/ :: v_dual_mov_b32 v153, v194 /*v450*/
	v_dual_mov_b32 v154, v194 /*v450*/ :: v_dual_mov_b32 v155, v194 /*v450*/
	v_dual_mov_b32 v156, v194 /*v450*/ :: v_dual_mov_b32 v157, v194 /*v450*/
	v_dual_mov_b32 v158, v194 /*v450*/ :: v_dual_mov_b32 v159, v194 /*v450*/
	v_dual_mov_b32 v160, v194 /*v450*/ :: v_dual_mov_b32 v161, v194 /*v450*/
	v_dual_mov_b32 v130, v194 /*v450*/ :: v_dual_mov_b32 v131, v194 /*v450*/
	v_dual_mov_b32 v132, v194 /*v450*/ :: v_dual_mov_b32 v133, v194 /*v450*/
	v_dual_mov_b32 v134, v194 /*v450*/ :: v_dual_mov_b32 v135, v194 /*v450*/
	v_dual_mov_b32 v136, v194 /*v450*/ :: v_dual_mov_b32 v137, v194 /*v450*/
	v_dual_mov_b32 v138, v194 /*v450*/ :: v_dual_mov_b32 v139, v194 /*v450*/
	v_dual_mov_b32 v140, v194 /*v450*/ :: v_dual_mov_b32 v141, v194 /*v450*/
	v_dual_mov_b32 v142, v194 /*v450*/ :: v_dual_mov_b32 v143, v194 /*v450*/
	v_dual_mov_b32 v144, v194 /*v450*/ :: v_dual_mov_b32 v145, v194 /*v450*/
	v_dual_mov_b32 v114, v194 /*v450*/ :: v_dual_mov_b32 v115, v194 /*v450*/
	v_dual_mov_b32 v116, v194 /*v450*/ :: v_dual_mov_b32 v117, v194 /*v450*/
	v_dual_mov_b32 v118, v194 /*v450*/ :: v_dual_mov_b32 v119, v194 /*v450*/
	v_dual_mov_b32 v120, v194 /*v450*/ :: v_dual_mov_b32 v121, v194 /*v450*/
	v_dual_mov_b32 v122, v194 /*v450*/ :: v_dual_mov_b32 v123, v194 /*v450*/
	v_dual_mov_b32 v124, v194 /*v450*/ :: v_dual_mov_b32 v125, v194 /*v450*/
	v_dual_mov_b32 v126, v194 /*v450*/ :: v_dual_mov_b32 v127, v194 /*v450*/
	v_dual_mov_b32 v128, v194 /*v450*/ :: v_dual_mov_b32 v129, v194 /*v450*/
	v_dual_mov_b32 v98, v194 /*v450*/ :: v_dual_mov_b32 v99, v194 /*v450*/
	v_dual_mov_b32 v100, v194 /*v450*/ :: v_dual_mov_b32 v101, v194 /*v450*/
	v_dual_mov_b32 v102, v194 /*v450*/ :: v_dual_mov_b32 v103, v194 /*v450*/
	v_dual_mov_b32 v104, v194 /*v450*/ :: v_dual_mov_b32 v105, v194 /*v450*/
	v_dual_mov_b32 v106, v194 /*v450*/ :: v_dual_mov_b32 v107, v194 /*v450*/
	v_dual_mov_b32 v108, v194 /*v450*/ :: v_dual_mov_b32 v109, v194 /*v450*/
	v_dual_mov_b32 v110, v194 /*v450*/ :: v_dual_mov_b32 v111, v194 /*v450*/
	v_dual_mov_b32 v112, v194 /*v450*/ :: v_dual_mov_b32 v113, v194 /*v450*/
	v_dual_mov_b32 v82, v194 /*v450*/ :: v_dual_mov_b32 v83, v194 /*v450*/
	v_dual_mov_b32 v84, v194 /*v450*/ :: v_dual_mov_b32 v85, v194 /*v450*/
	v_dual_mov_b32 v86, v194 /*v450*/ :: v_dual_mov_b32 v87, v194 /*v450*/
	v_dual_mov_b32 v88, v194 /*v450*/ :: v_dual_mov_b32 v89, v194 /*v450*/
	v_dual_mov_b32 v90, v194 /*v450*/ :: v_dual_mov_b32 v91, v194 /*v450*/
	v_dual_mov_b32 v92, v194 /*v450*/ :: v_dual_mov_b32 v93, v194 /*v450*/
	v_dual_mov_b32 v94, v194 /*v450*/ :: v_dual_mov_b32 v95, v194 /*v450*/
	v_dual_mov_b32 v96, v194 /*v450*/ :: v_dual_mov_b32 v97, v194 /*v450*/
	v_dual_mov_b32 v34, v194 /*v450*/ :: v_dual_mov_b32 v35, v194 /*v450*/
	v_dual_mov_b32 v36, v194 /*v450*/ :: v_dual_mov_b32 v37, v194 /*v450*/
	v_dual_mov_b32 v38, v194 /*v450*/ :: v_dual_mov_b32 v39, v194 /*v450*/
	v_dual_mov_b32 v40, v194 /*v450*/ :: v_dual_mov_b32 v41, v194 /*v450*/
	v_dual_mov_b32 v42, v194 /*v450*/ :: v_dual_mov_b32 v43, v194 /*v450*/
	v_dual_mov_b32 v44, v194 /*v450*/ :: v_dual_mov_b32 v45, v194 /*v450*/
	v_dual_mov_b32 v46, v194 /*v450*/ :: v_dual_mov_b32 v47, v194 /*v450*/
	v_dual_mov_b32 v48, v194 /*v450*/ :: v_dual_mov_b32 v49, v194 /*v450*/
	v_dual_mov_b32 v66, v194 /*v450*/ :: v_dual_mov_b32 v67, v194 /*v450*/
	v_dual_mov_b32 v68, v194 /*v450*/ :: v_dual_mov_b32 v69, v194 /*v450*/
	v_dual_mov_b32 v70, v194 /*v450*/ :: v_dual_mov_b32 v71, v194 /*v450*/
	v_dual_mov_b32 v72, v194 /*v450*/ :: v_dual_mov_b32 v73, v194 /*v450*/
	v_dual_mov_b32 v74, v194 /*v450*/ :: v_dual_mov_b32 v75, v194 /*v450*/
	v_dual_mov_b32 v76, v194 /*v450*/ :: v_dual_mov_b32 v77, v194 /*v450*/
	v_dual_mov_b32 v78, v194 /*v450*/ :: v_dual_mov_b32 v79, v194 /*v450*/
	v_dual_mov_b32 v80, v194 /*v450*/ :: v_dual_mov_b32 v81, v194 /*v450*/
	v_dual_mov_b32 v50, v194 /*v450*/ :: v_dual_mov_b32 v51, v194 /*v450*/
	v_dual_mov_b32 v52, v194 /*v450*/ :: v_dual_mov_b32 v53, v194 /*v450*/
	v_dual_mov_b32 v54, v194 /*v450*/ :: v_dual_mov_b32 v55, v194 /*v450*/
	v_dual_mov_b32 v56, v194 /*v450*/ :: v_dual_mov_b32 v57, v194 /*v450*/
	v_dual_mov_b32 v58, v194 /*v450*/ :: v_dual_mov_b32 v59, v194 /*v450*/
	v_dual_mov_b32 v60, v194 /*v450*/ :: v_dual_mov_b32 v61, v194 /*v450*/
	v_dual_mov_b32 v62, v194 /*v450*/ :: v_dual_mov_b32 v63, v194 /*v450*/
	v_dual_mov_b32 v64, v194 /*v450*/ :: v_dual_mov_b32 v65, v194 /*v450*/
	s_set_vgpr_msb 0x181
	v_dual_mov_b32 v234 /*v746*/, v194 /*v450*/ :: v_dual_mov_b32 v235 /*v747*/, v194 /*v450*/
	v_dual_mov_b32 v236 /*v748*/, v194 /*v450*/ :: v_dual_mov_b32 v237 /*v749*/, v194 /*v450*/
	v_dual_mov_b32 v238 /*v750*/, v194 /*v450*/ :: v_dual_mov_b32 v239 /*v751*/, v194 /*v450*/
	v_dual_mov_b32 v240 /*v752*/, v194 /*v450*/ :: v_dual_mov_b32 v241 /*v753*/, v194 /*v450*/
	v_dual_mov_b32 v242 /*v754*/, v194 /*v450*/ :: v_dual_mov_b32 v243 /*v755*/, v194 /*v450*/
	v_dual_mov_b32 v244 /*v756*/, v194 /*v450*/ :: v_dual_mov_b32 v245 /*v757*/, v194 /*v450*/
	v_dual_mov_b32 v246 /*v758*/, v194 /*v450*/ :: v_dual_mov_b32 v247 /*v759*/, v194 /*v450*/
	v_dual_mov_b32 v248 /*v760*/, v194 /*v450*/ :: v_dual_mov_b32 v249 /*v761*/, v194 /*v450*/
	v_dual_mov_b32 v196 /*v708*/, v194 /*v450*/ :: v_dual_mov_b32 v197 /*v709*/, v194 /*v450*/
	v_dual_mov_b32 v198 /*v710*/, v194 /*v450*/ :: v_dual_mov_b32 v199 /*v711*/, v194 /*v450*/
	v_dual_mov_b32 v200 /*v712*/, v194 /*v450*/ :: v_dual_mov_b32 v201 /*v713*/, v194 /*v450*/
	v_dual_mov_b32 v202 /*v714*/, v194 /*v450*/ :: v_dual_mov_b32 v203 /*v715*/, v194 /*v450*/
	v_dual_mov_b32 v204 /*v716*/, v194 /*v450*/ :: v_dual_mov_b32 v205 /*v717*/, v194 /*v450*/
	v_dual_mov_b32 v206 /*v718*/, v194 /*v450*/ :: v_dual_mov_b32 v207 /*v719*/, v194 /*v450*/
	v_dual_mov_b32 v208 /*v720*/, v194 /*v450*/ :: v_dual_mov_b32 v209 /*v721*/, v194 /*v450*/
	v_dual_mov_b32 v210 /*v722*/, v194 /*v450*/ :: v_dual_mov_b32 v211 /*v723*/, v194 /*v450*/
	s_lshr_b32 s5, s5, 16
	s_or_b32 s10, s6, 0x7fff
	s_or_b32 s11, s5, 0x800000
	s_mov_b32 s9, 0xffff0000
	s_mov_b32 s8, 0x730000f
	s_mov_b32 s15, s14
	s_add_nc_u64 s[6:7], s[18:19], 0x200
	s_mov_b32 s5, s14
	s_set_vgpr_msb 0x8100
	s_wait_dscnt 0x0
.LBB0_36:
	s_and_b32 s17, s5, 3
	s_add_co_i32 s5, s5, 1
	s_mul_i32 s17, s17, 0x12000
	s_and_b32 s18, s5, 3
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v0, s17, v224 /*v736*/
	s_mul_i32 s18, s18, 0x12000
	v_dual_add_nc_u32 v1, s17, v225 /*v737*/ :: v_dual_add_nc_u32 v10, s17, v195 /*v707*/
	v_add_nc_u32_e32 v11, s17, v212 /*v724*/
	v_dual_add_nc_u32 v14, s18, v224 /*v736*/ :: v_dual_add_nc_u32 v15, s18, v225 /*v737*/
	v_dual_add_nc_u32 v20, s18, v195 /*v707*/ :: v_dual_add_nc_u32 v21, s18, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v2, v6
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v3, v6
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v4, v6
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v5, v6
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_b32 v[8:9], v11 offset0:32 offset1:96
	ds_load_2addr_b32 v[12:13], v11 offset0:160 offset1:224
	ds_load_2addr_b32 v[16:17], v10 offset0:128 offset1:160
	ds_load_2addr_b32 v[10:11], v10 offset0:192 offset1:224
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v1 offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v1 offset:1536
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x10
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v5, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v4, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v3, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v2, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[74:77] /*v[586:589]*/, v1 offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v1 offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v1 offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v1 offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v1 offset:7168
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x13
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v2, v7
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v3, v7
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v4, v7
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v5, v7
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[94:97] /*v[606:609]*/, v1 offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v1 offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v1 offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v1 offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v1 offset:11776
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x16
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17] /*v[258:273]*/, v5, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[18:33] /*v[274:289]*/, v4, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[34:49] /*v[290:305]*/, v3, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65] /*v[306:321]*/, v2, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[114:117] /*v[626:629]*/, v1 offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v1 offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v1 offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v1 offset:15872
	ds_load_b128 v[138:141] /*v[650:653]*/, v0 offset:64
	ds_load_b128 v[142:145] /*v[654:657]*/, v0 offset:96
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x18
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[242:257], v2, v18
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[226:241], v3, v18
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[210:225], v4, v18
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[194:209], v5, v18
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[130:133] /*v[642:645]*/, v0 offset:2368
	ds_load_b128 v[134:137] /*v[646:649]*/, v0 offset:2400
	ds_load_b128 v[178:181] /*v[690:693]*/, v0 offset:4672
	ds_load_b128 v[182:185] /*v[694:697]*/, v0 offset:4704
	ds_load_b128 v[186:189] /*v[698:701]*/, v0 offset:6976
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x1b
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v5, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v4, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v3, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v2, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[190:193] /*v[702:705]*/, v0 offset:7008
	ds_load_b128 v[170:173] /*v[682:685]*/, v0 offset:9280
	ds_load_b128 v[174:177] /*v[686:689]*/, v0 offset:9312
	ds_load_b128 v[214:217] /*v[726:729]*/, v0 offset:11584
	ds_load_b128 v[218:221] /*v[730:733]*/, v0 offset:11616
	ds_load_b128 v[162:165] /*v[674:677]*/, v0 offset:16192
	ds_load_b128 v[166:169] /*v[678:681]*/, v0 offset:16224
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[114:129], v2, v19
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[98:113], v3, v19
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[82:97], v4, v19
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[34:49], v5, v19
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[154:157] /*v[666:669]*/, v0 offset:13888
	ds_load_b128 v[158:161] /*v[670:673]*/, v0 offset:13920
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[196:211] /*v[708:723]*/, v5, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[234:249] /*v[746:761]*/, v4, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[50:65], v3, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[66:81], v2, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_or_b32 s19, s7, 0x80000000
	s_mov_b32 s18, s6
	s_wait_tensorcnt 0x2
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	tensor_load_to_lds s[16:19], s[8:15]
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[194:209] /*v[450:465]*/, v8, v16
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[146:149] /*v[658:661]*/, v14 offset:16128
	ds_load_b128 v[150:153] /*v[662:665]*/, v14 offset:16160
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[242:257] /*v[498:513]*/, v9, v16
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[226:241] /*v[482:497]*/, v12, v16
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[210:225] /*v[466:481]*/, v13, v16
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_stride64_b32 v[2:3], v21 offset1:1
	ds_load_2addr_stride64_b32 v[4:5], v21 offset0:2 offset1:3
	ds_load_2addr_b32 v[6:7], v20 offset1:32
	ds_load_2addr_b32 v[18:19], v20 offset0:64 offset1:96
	s_set_vgpr_msb 0x5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v13, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v12, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v9, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v8, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[2:5] /*v[514:517]*/, v15
	ds_load_b128 v[6:9] /*v[518:521]*/, v15 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v15 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v15 offset:2560
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[178:185] /*v[690:697]*/, v[114:129] /*v[370:385]*/, v8, v17
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[178:185] /*v[690:697]*/, v[98:113] /*v[354:369]*/, v9, v17
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[178:185] /*v[690:697]*/, v[82:97] /*v[338:353]*/, v12, v17
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[178:185] /*v[690:697]*/, v[66:81] /*v[322:337]*/, v13, v17
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[18:21] /*v[530:533]*/, v15 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v15 offset:4608
	ds_load_b128 v[26:29] /*v[538:541]*/, v15 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v15 offset:6656
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[2:17] /*v[258:273]*/, v13, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[18:33] /*v[274:289]*/, v12, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[34:49] /*v[290:305]*/, v9, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[50:65] /*v[306:321]*/, v8, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[34:37] /*v[546:549]*/, v15 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v15 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v15 offset:10240
	ds_load_b128 v[46:49] /*v[558:561]*/, v15 offset:10752
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[242:257], v8, v10
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[226:241], v9, v10
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[170:177] /*v[682:689]*/, v[210:225], v12, v10
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[170:177] /*v[682:689]*/, v[194:209], v13, v10
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[50:53] /*v[562:565]*/, v15 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v15 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v15 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v15 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[214:221] /*v[726:733]*/, v[130:145], v13, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[214:221] /*v[726:733]*/, v[146:161], v12, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[214:221] /*v[726:733]*/, v[162:177], v9, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[214:221] /*v[726:733]*/, v[178:193], v8, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[186:189] /*v[698:701]*/, v14
	ds_load_b128 v[190:193] /*v[702:705]*/, v14 offset:32
	ds_load_b128 v[178:181] /*v[690:693]*/, v14 offset:2304
	ds_load_b128 v[182:185] /*v[694:697]*/, v14 offset:2336
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[154:161] /*v[666:673]*/, v[114:129], v8, v11
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[154:161] /*v[666:673]*/, v[98:113], v9, v11
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[82:97], v12, v11
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[34:49], v13, v11
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[138:141] /*v[650:653]*/, v14 offset:4608
	ds_load_b128 v[142:145] /*v[654:657]*/, v14 offset:4640
	ds_load_b128 v[130:133] /*v[642:645]*/, v14 offset:6912
	ds_load_b128 v[134:137] /*v[646:649]*/, v14 offset:6944
	ds_load_b128 v[154:157] /*v[666:669]*/, v14 offset:13824
	ds_load_b128 v[158:161] /*v[670:673]*/, v14 offset:13856
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[114:129] /*v[626:641]*/, v[162:169] /*v[674:681]*/, v[196:211] /*v[708:723]*/, v13, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[162:169] /*v[674:681]*/, v[234:249] /*v[746:761]*/, v12, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[50:65], v9, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[66:81], v8, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[170:173] /*v[682:685]*/, v14 offset:9216
	ds_load_b128 v[174:177] /*v[686:689]*/, v14 offset:9248
	ds_load_b128 v[162:165] /*v[674:677]*/, v14 offset:11520
	ds_load_b128 v[166:169] /*v[678:681]*/, v14 offset:11552
	s_cmp_lg_u32 s5, 8
	s_add_nc_u64 s[6:7], s[6:7], 0x80
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_36
.LBB0_37:
	s_set_vgpr_msb 0x88
	v_or_b32_e32 v214 /*v726*/, 0x9200, v213 /*v725*/
	v_add_nc_u32_e32 v213 /*v725*/, 32, v224 /*v736*/
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s12, 1
	s_set_vgpr_msb 0x8800
	s_cbranch_vccnz .LBB0_40
	s_mov_b32 s4, 0
	s_add_nc_u64 s[16:17], s[36:37], 0x2000
	s_movk_i32 s9, 0x6000
	s_mov_b32 s8, 16
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s4
	s_mov_b32 s11, s4
	s_mov_b32 s18, s4
	s_wait_dscnt 0x0
.LBB0_39:
	s_and_b32 s13, s18, 3
	s_add_co_i32 s18, s18, 1
	s_mul_i32 s13, s13, 0x12000
	s_and_b32 s14, s18, 3
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v0, s13, v224 /*v736*/
	s_mul_i32 s14, s14, 0x12000
	v_dual_add_nc_u32 v1, s13, v225 /*v737*/ :: v_dual_add_nc_u32 v10, s13, v195 /*v707*/
	v_add_nc_u32_e32 v11, s13, v212 /*v724*/
	v_dual_add_nc_u32 v14, s14, v224 /*v736*/ :: v_dual_add_nc_u32 v15, s14, v225 /*v737*/
	v_dual_add_nc_u32 v20, s14, v195 /*v707*/ :: v_dual_add_nc_u32 v21, s14, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v2, v6
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v3, v6
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v4, v6
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v5, v6
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_b32 v[8:9], v11 offset0:32 offset1:96
	ds_load_2addr_b32 v[12:13], v11 offset0:160 offset1:224
	ds_load_2addr_b32 v[16:17], v10 offset0:128 offset1:160
	ds_load_2addr_b32 v[10:11], v10 offset0:192 offset1:224
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v1 offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v1 offset:1536
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x10
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v5, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v4, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v3, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v2, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[74:77] /*v[586:589]*/, v1 offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v1 offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v1 offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v1 offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v1 offset:7168
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x13
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v2, v7
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v3, v7
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v4, v7
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v5, v7
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[94:97] /*v[606:609]*/, v1 offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v1 offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v1 offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v1 offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v1 offset:11776
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x16
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17] /*v[258:273]*/, v5, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[18:33] /*v[274:289]*/, v4, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[34:49] /*v[290:305]*/, v3, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65] /*v[306:321]*/, v2, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[114:117] /*v[626:629]*/, v1 offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v1 offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v1 offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v1 offset:15872
	ds_load_b128 v[138:141] /*v[650:653]*/, v0 offset:64
	ds_load_b128 v[142:145] /*v[654:657]*/, v0 offset:96
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x18
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[242:257], v2, v18
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[226:241], v3, v18
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[210:225], v4, v18
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[194:209], v5, v18
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[130:133] /*v[642:645]*/, v0 offset:2368
	ds_load_b128 v[134:137] /*v[646:649]*/, v0 offset:2400
	ds_load_b128 v[178:181] /*v[690:693]*/, v0 offset:4672
	ds_load_b128 v[182:185] /*v[694:697]*/, v0 offset:4704
	ds_load_b128 v[186:189] /*v[698:701]*/, v0 offset:6976
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x1b
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v5, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v4, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v3, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v2, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[190:193] /*v[702:705]*/, v0 offset:7008
	ds_load_b128 v[170:173] /*v[682:685]*/, v0 offset:9280
	ds_load_b128 v[174:177] /*v[686:689]*/, v0 offset:9312
	ds_load_b128 v[216:219] /*v[728:731]*/, v0 offset:11584
	ds_load_b128 v[220:223] /*v[732:735]*/, v0 offset:11616
	ds_load_b128 v[162:165] /*v[674:677]*/, v0 offset:16192
	ds_load_b128 v[166:169] /*v[678:681]*/, v0 offset:16224
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[114:129], v2, v19
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[98:113], v3, v19
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[82:97], v4, v19
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[34:49], v5, v19
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[154:157] /*v[666:669]*/, v0 offset:13888
	ds_load_b128 v[158:161] /*v[670:673]*/, v0 offset:13920
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[196:211] /*v[708:723]*/, v5, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[234:249] /*v[746:761]*/, v4, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[50:65], v3, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[66:81], v2, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_add_co_i32 s13, s13, 0x9000
	s_or_b32 s15, s17, 0x80000000
	s_mov_b32 s14, s16
	s_wait_tensorcnt 0x2
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	tensor_load_to_lds s[12:15], s[4:11]
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[194:209] /*v[450:465]*/, v8, v16
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[146:149] /*v[658:661]*/, v14 offset:16128
	ds_load_b128 v[150:153] /*v[662:665]*/, v14 offset:16160
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[242:257] /*v[498:513]*/, v9, v16
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[226:241] /*v[482:497]*/, v12, v16
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[210:225] /*v[466:481]*/, v13, v16
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_stride64_b32 v[2:3], v21 offset1:1
	ds_load_2addr_stride64_b32 v[4:5], v21 offset0:2 offset1:3
	ds_load_2addr_b32 v[6:7], v20 offset1:32
	ds_load_2addr_b32 v[18:19], v20 offset0:64 offset1:96
	s_set_vgpr_msb 0x5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v13, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v12, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v9, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v8, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[2:5] /*v[514:517]*/, v15
	ds_load_b128 v[6:9] /*v[518:521]*/, v15 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v15 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v15 offset:2560
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[178:185] /*v[690:697]*/, v[114:129] /*v[370:385]*/, v8, v17
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[178:185] /*v[690:697]*/, v[98:113] /*v[354:369]*/, v9, v17
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[178:185] /*v[690:697]*/, v[82:97] /*v[338:353]*/, v12, v17
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[178:185] /*v[690:697]*/, v[66:81] /*v[322:337]*/, v13, v17
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[18:21] /*v[530:533]*/, v15 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v15 offset:4608
	ds_load_b128 v[26:29] /*v[538:541]*/, v15 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v15 offset:6656
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[2:17] /*v[258:273]*/, v13, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[18:33] /*v[274:289]*/, v12, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[34:49] /*v[290:305]*/, v9, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[50:65] /*v[306:321]*/, v8, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[34:37] /*v[546:549]*/, v15 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v15 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v15 offset:10240
	ds_load_b128 v[46:49] /*v[558:561]*/, v15 offset:10752
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[242:257], v8, v10
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[226:241], v9, v10
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[170:177] /*v[682:689]*/, v[210:225], v12, v10
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[170:177] /*v[682:689]*/, v[194:209], v13, v10
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[50:53] /*v[562:565]*/, v15 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v15 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v15 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v15 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[216:223] /*v[728:735]*/, v[130:145], v13, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[216:223] /*v[728:735]*/, v[146:161], v12, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[216:223] /*v[728:735]*/, v[162:177], v9, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[216:223] /*v[728:735]*/, v[178:193], v8, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[186:189] /*v[698:701]*/, v14
	ds_load_b128 v[190:193] /*v[702:705]*/, v14 offset:32
	ds_load_b128 v[178:181] /*v[690:693]*/, v14 offset:2304
	ds_load_b128 v[182:185] /*v[694:697]*/, v14 offset:2336
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[154:161] /*v[666:673]*/, v[114:129], v8, v11
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[154:161] /*v[666:673]*/, v[98:113], v9, v11
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[82:97], v12, v11
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[34:49], v13, v11
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[138:141] /*v[650:653]*/, v14 offset:4608
	ds_load_b128 v[142:145] /*v[654:657]*/, v14 offset:4640
	ds_load_b128 v[130:133] /*v[642:645]*/, v14 offset:6912
	ds_load_b128 v[134:137] /*v[646:649]*/, v14 offset:6944
	ds_load_b128 v[154:157] /*v[666:669]*/, v14 offset:13824
	ds_load_b128 v[158:161] /*v[670:673]*/, v14 offset:13856
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[114:129] /*v[626:641]*/, v[162:169] /*v[674:681]*/, v[196:211] /*v[708:723]*/, v13, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[162:169] /*v[674:681]*/, v[234:249] /*v[746:761]*/, v12, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[50:65], v9, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[66:81], v8, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[170:173] /*v[682:685]*/, v14 offset:9216
	ds_load_b128 v[174:177] /*v[686:689]*/, v14 offset:9248
	ds_load_b128 v[162:165] /*v[674:677]*/, v14 offset:11520
	ds_load_b128 v[166:169] /*v[678:681]*/, v14 offset:11552
	s_cmp_lg_u32 s18, 8
	s_add_nc_u64 s[16:17], s[16:17], 0x800
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_39
.LBB0_40:
	s_and_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s12, 1
	s_cbranch_vccnz .LBB0_43
	s_mov_b32 s10, 0
	s_add_nc_u64 s[16:17], s[34:35], 0x1000
	s_movk_i32 s9, 0xc00
	s_mov_b32 s8, 2
	s_mov_b32 s7, 0x1007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x20000
	s_mov_b32 s11, s10
	s_mov_b32 s1, s10
	s_wait_dscnt 0x0
.LBB0_42:
	s_and_b32 s13, s1, 3
	s_add_co_i32 s1, s1, 1
	s_mul_i32 s13, s13, 0x12000
	s_and_b32 s14, s1, 3
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v0, s13, v224 /*v736*/
	s_mul_i32 s14, s14, 0x12000
	v_dual_add_nc_u32 v1, s13, v225 /*v737*/ :: v_dual_add_nc_u32 v10, s13, v195 /*v707*/
	v_add_nc_u32_e32 v11, s13, v212 /*v724*/
	v_dual_add_nc_u32 v14, s14, v224 /*v736*/ :: v_dual_add_nc_u32 v15, s14, v225 /*v737*/
	v_dual_add_nc_u32 v20, s14, v195 /*v707*/ :: v_dual_add_nc_u32 v21, s14, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v2, v6
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v3, v6
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v4, v6
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v5, v6
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_b32 v[8:9], v11 offset0:32 offset1:96
	ds_load_2addr_b32 v[12:13], v11 offset0:160 offset1:224
	ds_load_2addr_b32 v[16:17], v10 offset0:128 offset1:160
	ds_load_2addr_b32 v[10:11], v10 offset0:192 offset1:224
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v1 offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v1 offset:1536
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x10
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v5, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v4, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v3, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v2, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[74:77] /*v[586:589]*/, v1 offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v1 offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v1 offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v1 offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v1 offset:7168
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x13
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v2, v7
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v3, v7
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v4, v7
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v5, v7
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[94:97] /*v[606:609]*/, v1 offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v1 offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v1 offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v1 offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v1 offset:11776
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x16
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17] /*v[258:273]*/, v5, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[18:33] /*v[274:289]*/, v4, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[34:49] /*v[290:305]*/, v3, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65] /*v[306:321]*/, v2, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[114:117] /*v[626:629]*/, v1 offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v1 offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v1 offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v1 offset:15872
	ds_load_b128 v[138:141] /*v[650:653]*/, v0 offset:64
	ds_load_b128 v[142:145] /*v[654:657]*/, v0 offset:96
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x18
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[242:257], v2, v18
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[226:241], v3, v18
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[210:225], v4, v18
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[194:209], v5, v18
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[130:133] /*v[642:645]*/, v0 offset:2368
	ds_load_b128 v[134:137] /*v[646:649]*/, v0 offset:2400
	ds_load_b128 v[178:181] /*v[690:693]*/, v0 offset:4672
	ds_load_b128 v[182:185] /*v[694:697]*/, v0 offset:4704
	ds_load_b128 v[186:189] /*v[698:701]*/, v0 offset:6976
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x1b
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v5, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v4, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v3, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v2, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[190:193] /*v[702:705]*/, v0 offset:7008
	ds_load_b128 v[170:173] /*v[682:685]*/, v0 offset:9280
	ds_load_b128 v[174:177] /*v[686:689]*/, v0 offset:9312
	ds_load_b128 v[216:219] /*v[728:731]*/, v0 offset:11584
	ds_load_b128 v[220:223] /*v[732:735]*/, v0 offset:11616
	ds_load_b128 v[162:165] /*v[674:677]*/, v0 offset:16192
	ds_load_b128 v[166:169] /*v[678:681]*/, v0 offset:16224
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[114:129], v2, v19
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[98:113], v3, v19
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[82:97], v4, v19
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[34:49], v5, v19
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[154:157] /*v[666:669]*/, v0 offset:13888
	ds_load_b128 v[158:161] /*v[670:673]*/, v0 offset:13920
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[196:211] /*v[708:723]*/, v5, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[234:249] /*v[746:761]*/, v4, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[50:65], v3, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[66:81], v2, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_add_co_i32 s13, s13, 0x11000
	s_or_b32 s15, s17, 0x80000000
	s_mov_b32 s14, s16
	s_wait_tensorcnt 0x2
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	tensor_load_to_lds s[12:15], s[4:11]
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[194:209] /*v[450:465]*/, v8, v16
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[146:149] /*v[658:661]*/, v14 offset:16128
	ds_load_b128 v[150:153] /*v[662:665]*/, v14 offset:16160
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[242:257] /*v[498:513]*/, v9, v16
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[226:241] /*v[482:497]*/, v12, v16
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[210:225] /*v[466:481]*/, v13, v16
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_stride64_b32 v[2:3], v21 offset1:1
	ds_load_2addr_stride64_b32 v[4:5], v21 offset0:2 offset1:3
	ds_load_2addr_b32 v[6:7], v20 offset1:32
	ds_load_2addr_b32 v[18:19], v20 offset0:64 offset1:96
	s_set_vgpr_msb 0x5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v13, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v12, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v9, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v8, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[2:5] /*v[514:517]*/, v15
	ds_load_b128 v[6:9] /*v[518:521]*/, v15 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v15 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v15 offset:2560
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[178:185] /*v[690:697]*/, v[114:129] /*v[370:385]*/, v8, v17
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[178:185] /*v[690:697]*/, v[98:113] /*v[354:369]*/, v9, v17
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[178:185] /*v[690:697]*/, v[82:97] /*v[338:353]*/, v12, v17
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[178:185] /*v[690:697]*/, v[66:81] /*v[322:337]*/, v13, v17
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[18:21] /*v[530:533]*/, v15 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v15 offset:4608
	ds_load_b128 v[26:29] /*v[538:541]*/, v15 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v15 offset:6656
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[2:17] /*v[258:273]*/, v13, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[18:33] /*v[274:289]*/, v12, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[34:49] /*v[290:305]*/, v9, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[50:65] /*v[306:321]*/, v8, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[34:37] /*v[546:549]*/, v15 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v15 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v15 offset:10240
	ds_load_b128 v[46:49] /*v[558:561]*/, v15 offset:10752
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[242:257], v8, v10
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[226:241], v9, v10
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[170:177] /*v[682:689]*/, v[210:225], v12, v10
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[170:177] /*v[682:689]*/, v[194:209], v13, v10
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[50:53] /*v[562:565]*/, v15 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v15 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v15 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v15 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[216:223] /*v[728:735]*/, v[130:145], v13, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[216:223] /*v[728:735]*/, v[146:161], v12, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[216:223] /*v[728:735]*/, v[162:177], v9, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[216:223] /*v[728:735]*/, v[178:193], v8, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[186:189] /*v[698:701]*/, v14
	ds_load_b128 v[190:193] /*v[702:705]*/, v14 offset:32
	ds_load_b128 v[178:181] /*v[690:693]*/, v14 offset:2304
	ds_load_b128 v[182:185] /*v[694:697]*/, v14 offset:2336
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[154:161] /*v[666:673]*/, v[114:129], v8, v11
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[154:161] /*v[666:673]*/, v[98:113], v9, v11
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[82:97], v12, v11
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[34:49], v13, v11
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[138:141] /*v[650:653]*/, v14 offset:4608
	ds_load_b128 v[142:145] /*v[654:657]*/, v14 offset:4640
	ds_load_b128 v[130:133] /*v[642:645]*/, v14 offset:6912
	ds_load_b128 v[134:137] /*v[646:649]*/, v14 offset:6944
	ds_load_b128 v[154:157] /*v[666:669]*/, v14 offset:13824
	ds_load_b128 v[158:161] /*v[670:673]*/, v14 offset:13856
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[114:129] /*v[626:641]*/, v[162:169] /*v[674:681]*/, v[196:211] /*v[708:723]*/, v13, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[162:169] /*v[674:681]*/, v[234:249] /*v[746:761]*/, v12, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[50:65], v9, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[66:81], v8, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[170:173] /*v[682:685]*/, v14 offset:9216
	ds_load_b128 v[174:177] /*v[686:689]*/, v14 offset:9248
	ds_load_b128 v[162:165] /*v[674:677]*/, v14 offset:11520
	ds_load_b128 v[166:169] /*v[678:681]*/, v14 offset:11552
	s_cmp_lg_u32 s1, 8
	s_add_nc_u64 s[16:17], s[16:17], 0x400
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_42
.LBB0_43:
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s12, 1
	s_cbranch_vccnz .LBB0_46
	s_add_nc_u64 s[0:1], s[26:27], s[30:31]
	s_mov_b32 s10, 0
	s_add_nc_u64 s[0:1], s[0:1], s[28:29]
	s_movk_i32 s9, 0x300
	s_add_nc_u64 s[0:1], s[0:1], 0x400
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x407fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x20000
	s_mov_b32 s11, s10
	s_mov_b32 s16, s10
	s_wait_dscnt 0x0
.LBB0_45:
	s_and_b32 s13, s16, 3
	s_add_co_i32 s16, s16, 1
	s_mul_i32 s13, s13, 0x12000
	s_and_b32 s14, s16, 3
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v22, s13, v224 /*v736*/
	s_mul_i32 s14, s14, 0x12000
	v_dual_add_nc_u32 v8, s13, v225 /*v737*/ :: v_dual_add_nc_u32 v9, s13, v195 /*v707*/
	v_add_nc_u32_e32 v0, s13, v212 /*v724*/
	v_dual_add_nc_u32 v23, s14, v224 /*v736*/ :: v_dual_add_nc_u32 v24, s14, v225 /*v737*/
	v_dual_add_nc_u32 v25, s14, v195 /*v707*/ :: v_dual_add_nc_u32 v26, s14, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v2, v6
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v3, v6
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v4, v6
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v5, v6
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_b32 v[16:17], v0 offset0:32 offset1:96
	ds_load_2addr_b32 v[14:15], v0 offset0:160 offset1:224
	ds_load_2addr_b32 v[0:1], v9 offset0:128 offset1:160
	ds_load_2addr_b32 v[20:21], v9 offset0:192 offset1:224
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v8 offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v8 offset:1536
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x10
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v5, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v4, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v3, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v2, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[74:77] /*v[586:589]*/, v8 offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v8 offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v8 offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v8 offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v8 offset:7168
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x13
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v2, v7
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v3, v7
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v4, v7
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v5, v7
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[94:97] /*v[606:609]*/, v8 offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v8 offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v8 offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v8 offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v8 offset:11776
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x16
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17] /*v[258:273]*/, v5, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[18:33] /*v[274:289]*/, v4, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[34:49] /*v[290:305]*/, v3, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65] /*v[306:321]*/, v2, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[114:117] /*v[626:629]*/, v8 offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v8 offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v8 offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v8 offset:15872
	s_set_vgpr_msb 0x8000
	ds_load_b128 v[6:9], v22 offset:64
	ds_load_b128 v[10:13], v22 offset:96
	s_set_vgpr_msb 10
	s_wait_dscnt 0x18
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[242:257], v2, v18
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[226:241], v3, v18
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[210:225], v4, v18
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[194:209], v5, v18
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[130:133] /*v[642:645]*/, v22 offset:2368
	ds_load_b128 v[134:137] /*v[646:649]*/, v22 offset:2400
	ds_load_b128 v[138:141] /*v[650:653]*/, v22 offset:4672
	ds_load_b128 v[142:145] /*v[654:657]*/, v22 offset:4704
	ds_load_b128 v[178:181] /*v[690:693]*/, v22 offset:6976
	s_set_vgpr_msb 0x800a
	s_wait_dscnt 0x1b
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v5, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v4, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v3, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v2, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[182:185] /*v[694:697]*/, v22 offset:7008
	ds_load_b128 v[170:173] /*v[682:685]*/, v22 offset:9280
	ds_load_b128 v[174:177] /*v[686:689]*/, v22 offset:9312
	ds_load_b128 v[186:189] /*v[698:701]*/, v22 offset:11584
	ds_load_b128 v[190:193] /*v[702:705]*/, v22 offset:11616
	ds_load_b128 v[162:165] /*v[674:677]*/, v22 offset:16192
	ds_load_b128 v[166:169] /*v[678:681]*/, v22 offset:16224
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[114:129], v2, v19
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[98:113], v3, v19
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[82:97], v4, v19
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[34:49], v5, v19
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[154:157] /*v[666:669]*/, v22 offset:13888
	ds_load_b128 v[158:161] /*v[670:673]*/, v22 offset:13920
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[196:211] /*v[708:723]*/, v5, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[234:249] /*v[746:761]*/, v4, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[50:65], v3, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[66:81], v2, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_add_co_i32 s13, s13, 0x11800
	s_or_b32 s15, s1, 0x80000000
	s_mov_b32 s14, s0
	s_wait_tensorcnt 0x2
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	tensor_load_to_lds s[12:15], s[4:11]
	s_set_vgpr_msb 0xa52
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[66:81] /*v[578:593]*/, v[6:13], v[194:209] /*v[450:465]*/, v16, v0
	s_set_vgpr_msb 0x5280
	ds_load_b128 v[146:149] /*v[658:661]*/, v23 offset:16128
	ds_load_b128 v[150:153] /*v[662:665]*/, v23 offset:16160
	s_set_vgpr_msb 0x8052
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[82:97] /*v[594:609]*/, v[6:13], v[242:257] /*v[498:513]*/, v17, v0
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[6:13], v[226:241] /*v[482:497]*/, v14, v0
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[114:129] /*v[626:641]*/, v[6:13], v[210:225] /*v[466:481]*/, v15, v0
	s_set_vgpr_msb 0x5200
	ds_load_2addr_stride64_b32 v[2:3], v26 offset1:1
	ds_load_2addr_stride64_b32 v[4:5], v26 offset0:2 offset1:3
	ds_load_2addr_b32 v[6:7], v25 offset1:32
	ds_load_2addr_b32 v[18:19], v25 offset0:64 offset1:96
	s_set_vgpr_msb 0x5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v15, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v14, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v17, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v16, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[2:5] /*v[514:517]*/, v24
	ds_load_b128 v[6:9] /*v[518:521]*/, v24 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v24 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v24 offset:2560
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v16, v1
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v17, v1
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v14, v1
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v15, v1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[18:21] /*v[530:533]*/, v24 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v24 offset:4608
	ds_load_b128 v[26:29] /*v[538:541]*/, v24 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v24 offset:6656
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[178:185] /*v[690:697]*/, v[2:17] /*v[258:273]*/, v15, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[178:185] /*v[690:697]*/, v[18:33] /*v[274:289]*/, v14, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[178:185] /*v[690:697]*/, v[34:49] /*v[290:305]*/, v17, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[178:185] /*v[690:697]*/, v[50:65] /*v[306:321]*/, v16, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[34:37] /*v[546:549]*/, v24 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v24 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v24 offset:10240
	ds_load_b128 v[46:49] /*v[558:561]*/, v24 offset:10752
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[242:257], v16, v20
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[226:241], v17, v20
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[170:177] /*v[682:689]*/, v[210:225], v14, v20
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[170:177] /*v[682:689]*/, v[194:209], v15, v20
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[50:53] /*v[562:565]*/, v24 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v24 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v24 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v24 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[130:145], v15, v20 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[146:161], v14, v20 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[162:177], v17, v20 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[178:193], v16, v20 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[186:189] /*v[698:701]*/, v23
	ds_load_b128 v[190:193] /*v[702:705]*/, v23 offset:32
	ds_load_b128 v[178:181] /*v[690:693]*/, v23 offset:2304
	ds_load_b128 v[182:185] /*v[694:697]*/, v23 offset:2336
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[154:161] /*v[666:673]*/, v[114:129], v16, v21
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[154:161] /*v[666:673]*/, v[98:113], v17, v21
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[82:97], v14, v21
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[34:49], v15, v21
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[138:141] /*v[650:653]*/, v23 offset:4608
	ds_load_b128 v[142:145] /*v[654:657]*/, v23 offset:4640
	ds_load_b128 v[130:133] /*v[642:645]*/, v23 offset:6912
	ds_load_b128 v[134:137] /*v[646:649]*/, v23 offset:6944
	ds_load_b128 v[154:157] /*v[666:669]*/, v23 offset:13824
	ds_load_b128 v[158:161] /*v[670:673]*/, v23 offset:13856
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[114:129] /*v[626:641]*/, v[162:169] /*v[674:681]*/, v[196:211] /*v[708:723]*/, v15, v21 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[162:169] /*v[674:681]*/, v[234:249] /*v[746:761]*/, v14, v21 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[50:65], v17, v21 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[66:81], v16, v21 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[170:173] /*v[682:685]*/, v23 offset:9216
	ds_load_b128 v[174:177] /*v[686:689]*/, v23 offset:9248
	ds_load_b128 v[162:165] /*v[674:677]*/, v23 offset:11520
	ds_load_b128 v[166:169] /*v[678:681]*/, v23 offset:11552
	s_cmp_lg_u32 s16, 8
	s_add_nc_u64 s[0:1], s[0:1], 0x100
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_45
.LBB0_46:
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v251 /*v763*/, 0x11fe0, v213 /*v725*/
	s_set_vgpr_msb 0x8808
	v_add_nc_u32_e32 v17, 0x11e00, v214 /*v726*/
	v_add_nc_u32_e32 v0, 0x12000, v195 /*v707*/
	v_add_nc_u32_e32 v1, 0x12000, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v2, v6
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[224:227] /*v[736:739]*/, v213 /*v725*/ offset:16160
	ds_load_b128 v[228:231] /*v[740:743]*/, v213 /*v725*/ offset:16192
	s_set_vgpr_msb 0x825a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v3, v6
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v4, v6
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v5, v6
	s_set_vgpr_msb 0x5a02
	ds_load_2addr_b32 v[8:9], v212 /*v724*/ offset0:32 offset1:96
	ds_load_2addr_b32 v[12:13], v212 /*v724*/ offset0:160 offset1:224
	ds_load_2addr_b32 v[14:15], v195 /*v707*/ offset0:128 offset1:160
	ds_load_2addr_b32 v[10:11], v195 /*v707*/ offset0:192 offset1:224
	s_set_vgpr_msb 0x282
	ds_load_b128 v[66:69] /*v[578:581]*/, v214 /*v726*/ offset:512
	ds_load_b128 v[70:73] /*v[582:585]*/, v214 /*v726*/ offset:1024
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x12
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v5, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v4, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v3, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v2, v6 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[74:77] /*v[586:589]*/, v214 /*v726*/ offset:2560
	ds_load_b128 v[78:81] /*v[590:593]*/, v214 /*v726*/ offset:3072
	ds_load_b128 v[82:85] /*v[594:597]*/, v214 /*v726*/ offset:4608
	ds_load_b128 v[86:89] /*v[598:601]*/, v214 /*v726*/ offset:5120
	ds_load_b128 v[90:93] /*v[602:605]*/, v214 /*v726*/ offset:6656
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x15
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v2, v7
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v3, v7
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v4, v7
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v5, v7
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[94:97] /*v[606:609]*/, v214 /*v726*/ offset:7168
	ds_load_b128 v[98:101] /*v[610:613]*/, v214 /*v726*/ offset:8704
	ds_load_b128 v[102:105] /*v[614:617]*/, v214 /*v726*/ offset:9216
	ds_load_b128 v[106:109] /*v[618:621]*/, v214 /*v726*/ offset:10752
	ds_load_b128 v[110:113] /*v[622:625]*/, v214 /*v726*/ offset:11264
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x18
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17] /*v[258:273]*/, v5, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[18:33] /*v[274:289]*/, v4, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[34:49] /*v[290:305]*/, v3, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65] /*v[306:321]*/, v2, v7 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[130:133] /*v[642:645]*/, v214 /*v726*/ offset:12800
	ds_load_b128 v[134:137] /*v[646:649]*/, v214 /*v726*/ offset:13312
	ds_load_b128 v[138:141] /*v[650:653]*/, v214 /*v726*/ offset:14848
	ds_load_b128 v[142:145] /*v[654:657]*/, v214 /*v726*/ offset:15360
	ds_load_b128 v[114:117] /*v[626:629]*/, v213 /*v725*/ offset:32
	ds_load_b128 v[118:121] /*v[630:633]*/, v213 /*v725*/ offset:64
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x1a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[242:257], v2, v18
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[226:241], v3, v18
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[210:225], v4, v18
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[194:209], v5, v18
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[122:125] /*v[634:637]*/, v213 /*v725*/ offset:2336
	ds_load_b128 v[126:129] /*v[638:641]*/, v213 /*v725*/ offset:2368
	ds_load_b128 v[178:181] /*v[690:693]*/, v213 /*v725*/ offset:4640
	ds_load_b128 v[182:185] /*v[694:697]*/, v213 /*v725*/ offset:4672
	ds_load_b128 v[186:189] /*v[698:701]*/, v213 /*v725*/ offset:6944
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x1d
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v5, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v4, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v3, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v2, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[190:193] /*v[702:705]*/, v213 /*v725*/ offset:6976
	ds_load_b128 v[170:173] /*v[682:685]*/, v213 /*v725*/ offset:9248
	ds_load_b128 v[174:177] /*v[686:689]*/, v213 /*v725*/ offset:9280
	ds_load_b128 v[216:219] /*v[728:731]*/, v213 /*v725*/ offset:11552
	ds_load_b128 v[220:223] /*v[732:735]*/, v213 /*v725*/ offset:11584
	ds_load_b128 v[162:165] /*v[674:677]*/, v213 /*v725*/ offset:13856
	ds_load_b128 v[166:169] /*v[678:681]*/, v213 /*v725*/ offset:13888
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[114:129], v2, v19
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[98:113], v3, v19
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[82:97], v4, v19
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[34:49], v5, v19
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[196:211] /*v[708:723]*/, v5, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[234:249] /*v[746:761]*/, v4, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[50:65], v3, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[66:81], v2, v19 matrix_b_scale:MATRIX_SCALE_ROW1
	v_add_nc_u32_e32 v2, 0x12100, v212 /*v724*/
	v_add_nc_u32_e32 v3, 0x12200, v212 /*v724*/
	v_add_nc_u32_e32 v4, 0x12300, v212 /*v724*/
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[66:81] /*v[578:593]*/, v[114:121] /*v[626:633]*/, v[194:209] /*v[450:465]*/, v8, v14
	s_wait_tensorcnt 0x2
	s_set_vgpr_msb 0x5a08
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_add_nc_u32_e32 v5, 0x12080, v195 /*v707*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[82:97] /*v[594:609]*/, v[114:121] /*v[626:633]*/, v[242:257] /*v[498:513]*/, v9, v14
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v6, 0x12100, v195 /*v707*/
	v_add_nc_u32_e32 v7, 0x12180, v195 /*v707*/
	v_add_nc_u32_e32 v18, 0x12000, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[114:121] /*v[626:633]*/, v[226:241] /*v[482:497]*/, v12, v14
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v19, 0x12600, v214 /*v726*/
	v_add_nc_u32_e32 v20, 0x12800, v214 /*v726*/
	v_add_nc_u32_e32 v21, 0x12e00, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[130:145] /*v[642:657]*/, v[114:121] /*v[626:633]*/, v[210:225] /*v[466:481]*/, v13, v14
	s_set_vgpr_msb 0x5a08
	ds_load_b32 v33, v2
	ds_load_b32 v30, v3
	v_add_nc_u32_e32 v22, 0x13000, v214 /*v726*/
	ds_load_b32 v31, v4
	v_add_nc_u32_e32 v23, 0x13600, v214 /*v726*/
	ds_load_b32 v16, v1
	v_add_nc_u32_e32 v24, 0x13800, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[130:145] /*v[642:657]*/, v[122:129] /*v[634:641]*/, v[130:145] /*v[386:401]*/, v13, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v25, 0x13e00, v214 /*v726*/
	v_add_nc_u32_e32 v26, 0x14000, v214 /*v726*/
	v_add_nc_u32_e32 v27, 0x14600, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[122:129] /*v[634:641]*/, v[146:161] /*v[402:417]*/, v12, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v28, 0x14800, v214 /*v726*/
	v_add_nc_u32_e32 v2, 0x14e00, v214 /*v726*/
	v_add_nc_u32_e32 v3, 0x15000, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[122:129] /*v[634:641]*/, v[162:177] /*v[418:433]*/, v9, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v4, 0x15600, v214 /*v726*/
	v_add_nc_u32_e32 v29, 0x15800, v214 /*v726*/
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v6 /*v518*/, 0x12000, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[122:129] /*v[634:641]*/, v[178:193] /*v[434:449]*/, v8, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a00
	ds_load_b32 v0, v0
	ds_load_b32 v1, v5
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v10 /*v522*/, 0x128e0, v213 /*v725*/
	s_set_vgpr_msb 0x8800
	ds_load_b32 v32, v6
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v14 /*v526*/, 0x12900, v213 /*v725*/
	s_set_vgpr_msb 0x8800
	ds_load_b32 v14, v7
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v34 /*v546*/, 0x131e0, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[178:185] /*v[690:697]*/, v[114:129] /*v[370:385]*/, v8, v15
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v38 /*v550*/, 0x13200, v213 /*v725*/
	v_add_nc_u32_e32 v42 /*v554*/, 0x13ae0, v213 /*v725*/
	v_add_nc_u32_e32 v43 /*v555*/, 0x13b00, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[178:185] /*v[690:697]*/, v[98:113] /*v[354:369]*/, v9, v15
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v44 /*v556*/, 0x143e0, v213 /*v725*/
	v_add_nc_u32_e32 v45 /*v557*/, 0x14400, v213 /*v725*/
	v_add_nc_u32_e32 v46 /*v558*/, 0x14ce0, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[178:185] /*v[690:697]*/, v[82:97] /*v[338:353]*/, v12, v15
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v47 /*v559*/, 0x14d00, v213 /*v725*/
	v_add_nc_u32_e32 v48 /*v560*/, 0x155e0, v213 /*v725*/
	v_add_nc_u32_e32 v49 /*v561*/, 0x15600, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[130:145] /*v[642:657]*/, v[178:185] /*v[690:697]*/, v[66:81] /*v[322:337]*/, v13, v15
	s_set_vgpr_msb 0x5a88
	ds_load_b128 v[18:21] /*v[530:533]*/, v17
	ds_load_b128 v[22:25] /*v[534:537]*/, v18
	v_add_nc_u32_e32 v215 /*v727*/, 0x15ee0, v213 /*v725*/
	ds_load_b128 v[26:29] /*v[538:541]*/, v19
	v_add_nc_u32_e32 v250 /*v762*/, 0x15f00, v213 /*v725*/
	ds_load_b128 v[30:33] /*v[542:545]*/, v20
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[130:145] /*v[642:657]*/, v[186:193] /*v[698:705]*/, v[2:17] /*v[258:273]*/, v13, v15 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[18:33] /*v[274:289]*/, v12, v15 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[34:49] /*v[290:305]*/, v9, v15 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[50:65] /*v[306:321]*/, v8, v15 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[50:53] /*v[562:565]*/, v21
	ds_load_b128 v[54:57] /*v[566:569]*/, v22
	ds_load_b128 v[58:61] /*v[570:573]*/, v23
	ds_load_b128 v[62:65] /*v[574:577]*/, v24
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[242:257], v8, v10
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[226:241], v9, v10
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[170:177] /*v[682:689]*/, v[210:225], v12, v10
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[130:145] /*v[642:657]*/, v[170:177] /*v[682:689]*/, v[194:209], v13, v10
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[114:117] /*v[626:629]*/, v25
	ds_load_b128 v[118:121] /*v[630:633]*/, v26
	ds_load_b128 v[122:125] /*v[634:637]*/, v27
	ds_load_b128 v[126:129] /*v[638:641]*/, v28
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[130:145] /*v[642:657]*/, v[216:223] /*v[728:735]*/, v[130:145], v13, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[216:223] /*v[728:735]*/, v[146:161], v12, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[216:223] /*v[728:735]*/, v[162:177], v9, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[216:223] /*v[728:735]*/, v[178:193], v8, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[146:149] /*v[658:661]*/, v2
	ds_load_b128 v[150:153] /*v[662:665]*/, v3
	ds_load_b128 v[154:157] /*v[666:669]*/, v4
	ds_load_b128 v[158:161] /*v[670:673]*/, v29
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[114:129], v8, v11
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[98:113], v9, v11
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[162:169] /*v[674:681]*/, v[82:97], v12, v11
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[130:145] /*v[642:657]*/, v[162:169] /*v[674:681]*/, v[34:49], v13, v11
	s_set_vgpr_msb 0xaaa
	ds_load_b128 v[2:5] /*v[514:517]*/, v251 /*v763*/
	ds_load_b128 v[6:9] /*v[518:521]*/, v6 /*v518*/
	ds_load_b128 v[10:13] /*v[522:525]*/, v10 /*v522*/
	ds_load_b128 v[14:17] /*v[526:529]*/, v14 /*v526*/
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[130:145] /*v[642:657]*/, v[224:231] /*v[736:743]*/, v[196:211] /*v[708:723]*/, v13, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[130:133] /*v[642:645]*/, v215 /*v727*/
	ds_load_b128 v[134:137] /*v[646:649]*/, v250 /*v762*/
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[224:231] /*v[736:743]*/, v[234:249] /*v[746:761]*/, v12, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[98:101] /*v[610:613]*/, v46 /*v558*/
	ds_load_b128 v[102:105] /*v[614:617]*/, v47 /*v559*/
	ds_load_b128 v[106:109] /*v[618:621]*/, v48 /*v560*/
	ds_load_b128 v[110:113] /*v[622:625]*/, v49 /*v561*/
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[82:97] /*v[594:609]*/, v[224:231] /*v[736:743]*/, v[50:65], v9, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[90:93] /*v[602:605]*/, v44 /*v556*/
	ds_load_b128 v[94:97] /*v[606:609]*/, v45 /*v557*/
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[66:81] /*v[578:593]*/, v[224:231] /*v[736:743]*/, v[66:81], v8, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa8a
	ds_load_b128 v[34:37] /*v[546:549]*/, v34 /*v546*/
	ds_load_b128 v[38:41] /*v[550:553]*/, v38 /*v550*/
	ds_load_b128 v[82:85] /*v[594:597]*/, v42 /*v554*/
	ds_load_b128 v[86:89] /*v[598:601]*/, v43 /*v555*/
	v_add_nc_u32_e32 v219 /*v731*/, 0x23fe0, v213 /*v725*/
	v_add_nc_u32_e32 v220 /*v732*/, 0x23e00, v214 /*v726*/
	v_add_nc_u32_e32 v221 /*v733*/, 0x24000, v195 /*v707*/
	s_set_vgpr_msb 0x8a08
	v_add_nc_u32_e32 v15, 0x24000, v212 /*v724*/
	v_add_nc_u32_e32 v2, 0x12080, v212 /*v724*/
	v_add_nc_u32_e32 v3, 0x12180, v212 /*v724*/
	v_add_nc_u32_e32 v4, 0x12280, v212 /*v724*/
	v_add_nc_u32_e32 v5, 0x12380, v212 /*v724*/
	v_add_nc_u32_e32 v6, 0x12200, v195 /*v707*/
	s_wait_dscnt 0x24
	v_dual_mov_b32 v18, v16 :: v_dual_add_nc_u32 v7, 0x12280, v195 /*v707*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[18:33] /*v[530:545]*/, v[2:9] /*v[514:521]*/, v[194:209] /*v[450:465]*/, v18, v0
	s_set_vgpr_msb 0x5a08
	v_dual_mov_b32 v26, v18 :: v_dual_add_nc_u32 v19, 0x13220, v213 /*v725*/
	v_add_nc_u32_e32 v20, 0x13240, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[50:65] /*v[562:577]*/, v[2:9] /*v[514:521]*/, v[242:257] /*v[498:513]*/, v33, v0
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v21, 0x13b20, v213 /*v725*/
	v_add_nc_u32_e32 v22, 0x13b40, v213 /*v725*/
	v_add_nc_u32_e32 v23, 0x15640, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[114:129] /*v[626:641]*/, v[2:9] /*v[514:521]*/, v[226:241] /*v[482:497]*/, v30, v0
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v24, 0x15f20, v213 /*v725*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[170:173] /*v[682:685]*/, v24
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[146:161] /*v[658:673]*/, v[2:9] /*v[514:521]*/, v[210:225] /*v[466:481]*/, v31, v0
	s_set_vgpr_msb 0x5a08
	ds_load_b32 v27, v2
	ds_load_b32 v28, v3
	ds_load_b32 v17, v4
	ds_load_b32 v16, v5
	ds_load_b32 v8, v6
	ds_load_b32 v9, v7
	v_add_nc_u32_e32 v2, 0x12300, v195 /*v707*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0x13
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[146:161] /*v[658:673]*/, v[10:17] /*v[522:529]*/, v[130:145] /*v[386:401]*/, v31, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v3, 0x12380, v195 /*v707*/
	v_add_nc_u32_e32 v4, 0x12200, v214 /*v726*/
	v_add_nc_u32_e32 v5, 0x12400, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[114:129] /*v[626:641]*/, v[10:17] /*v[522:529]*/, v[146:161] /*v[402:417]*/, v30, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v25, 0x15f40, v213 /*v725*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[174:177] /*v[686:689]*/, v25
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[50:65] /*v[562:577]*/, v[10:17] /*v[522:529]*/, v[162:177] /*v[418:433]*/, v33, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[18:33] /*v[530:545]*/, v[10:17] /*v[522:529]*/, v[178:193] /*v[434:449]*/, v18, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	ds_load_b32 v10, v2
	ds_load_b32 v11, v3
	v_add_nc_u32_e32 v0, 0x12a00, v214 /*v726*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[2:5] /*v[514:517]*/, v4
	ds_load_b128 v[6:9] /*v[518:521]*/, v5
	ds_load_b128 v[10:13] /*v[522:525]*/, v0
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x12c00, v214 /*v726*/
	v_add_nc_u32_e32 v2, 0x13200, v214 /*v726*/
	v_add_nc_u32_e32 v3, 0x13400, v214 /*v726*/
	v_add_nc_u32_e32 v4, 0x13a00, v214 /*v726*/
	v_add_nc_u32_e32 v5, 0x13c00, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xf
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[18:33] /*v[530:545]*/, v[34:41] /*v[546:553]*/, v[114:129] /*v[370:385]*/, v18, v1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v18, 0x12920, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[50:65] /*v[562:577]*/, v[34:41] /*v[546:553]*/, v[98:113] /*v[354:369]*/, v33, v1
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[114:129] /*v[626:641]*/, v[34:41] /*v[546:553]*/, v[82:97] /*v[338:353]*/, v30, v1
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[146:161] /*v[658:673]*/, v[34:41] /*v[546:553]*/, v[66:81] /*v[322:337]*/, v31, v1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[14:17] /*v[526:529]*/, v0
	ds_load_b128 v[34:37] /*v[546:549]*/, v2
	ds_load_b128 v[38:41] /*v[550:553]*/, v3
	ds_load_b128 v[42:45] /*v[554:557]*/, v4
	ds_load_b128 v[46:49] /*v[558:561]*/, v5
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x14200, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0x12
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[146:161] /*v[658:673]*/, v[82:89] /*v[594:601]*/, v[2:17] /*v[258:273]*/, v31, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v2, 0x14400, v214 /*v726*/
	v_add_nc_u32_e32 v3, 0x14a00, v214 /*v726*/
	v_add_nc_u32_e32 v4, 0x14c00, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[114:129] /*v[626:641]*/, v[82:89] /*v[594:601]*/, v[18:33] /*v[274:289]*/, v30, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v5, 0x15200, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[50:65] /*v[562:577]*/, v[82:89] /*v[594:601]*/, v[34:49] /*v[290:305]*/, v33, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[18:33] /*v[530:545]*/, v[82:89] /*v[594:601]*/, v[50:65] /*v[306:321]*/, v26, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[66:69] /*v[578:581]*/, v0
	ds_load_b128 v[70:73] /*v[582:585]*/, v2
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v1, 0x15400, v214 /*v726*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[74:77] /*v[586:589]*/, v3
	ds_load_b128 v[78:81] /*v[590:593]*/, v4
	ds_load_b128 v[82:85] /*v[594:597]*/, v5
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x15a00, v214 /*v726*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[86:89] /*v[598:601]*/, v1
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[18:33] /*v[530:545]*/, v[90:97] /*v[602:609]*/, v[242:257], v26, v32
	v_add_nc_u32_e32 v1, 0x15c00, v214 /*v726*/
	v_add_nc_u32_e32 v2, 0x12020, v213 /*v725*/
	v_add_nc_u32_e32 v4, 0x12040, v213 /*v725*/
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[50:65] /*v[562:577]*/, v[90:97] /*v[602:609]*/, v[226:241], v33, v32
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[114:129] /*v[626:641]*/, v[90:97] /*v[602:609]*/, v[210:225], v30, v32
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[146:161] /*v[658:673]*/, v[90:97] /*v[602:609]*/, v[194:209], v31, v32
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[178:181] /*v[690:693]*/, v18
	ds_load_b128 v[90:93] /*v[602:605]*/, v0
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v18, 0x12940, v213 /*v725*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[94:97] /*v[606:609]*/, v1
	s_set_vgpr_msb 0x8000
	ds_load_b128 v[0:3], v2
	ds_load_b128 v[4:7], v4
	s_set_vgpr_msb 10
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[146:161] /*v[658:673]*/, v[98:105] /*v[610:617]*/, v[130:145], v31, v32 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[114:129] /*v[626:641]*/, v[98:105] /*v[610:617]*/, v[146:161], v30, v32 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[50:65] /*v[562:577]*/, v[98:105] /*v[610:617]*/, v[162:177], v33, v32 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[18:33] /*v[530:545]*/, v[98:105] /*v[610:617]*/, v[178:193], v26, v32 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[182:185] /*v[694:697]*/, v18
	ds_load_b128 v[138:141] /*v[650:653]*/, v19
	ds_load_b128 v[142:145] /*v[654:657]*/, v20
	ds_load_b128 v[162:165] /*v[674:677]*/, v21
	ds_load_b128 v[166:169] /*v[678:681]*/, v22
	s_set_vgpr_msb 0x800a
	v_add_nc_u32_e32 v18, 0x14420, v213 /*v725*/
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[146:161] /*v[658:673]*/, v[106:113] /*v[618:625]*/, v[34:49], v31, v14
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[98:101] /*v[610:613]*/, v18
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v19, 0x14440, v213 /*v725*/
	v_add_nc_u32_e32 v20, 0x14d20, v213 /*v725*/
	v_add_nc_u32_e32 v21, 0x14d40, v213 /*v725*/
	v_add_nc_u32_e32 v22, 0x15620, v213 /*v725*/
	s_set_vgpr_msb 0x8aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[146:161] /*v[658:673]*/, v[130:137] /*v[642:649]*/, v[196:211] /*v[708:723]*/, v31, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa80
	ds_load_b128 v[158:161] /*v[670:673]*/, v23
	ds_load_b128 v[102:105] /*v[614:617]*/, v19
	ds_load_b128 v[146:149] /*v[658:661]*/, v20
	ds_load_b128 v[150:153] /*v[662:665]*/, v21
	ds_load_b128 v[154:157] /*v[666:669]*/, v22
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[18:33] /*v[530:545]*/, v[106:113] /*v[618:625]*/, v[114:129], v26, v14
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[50:65] /*v[562:577]*/, v[106:113] /*v[618:625]*/, v[98:113], v33, v14
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[114:129] /*v[626:641]*/, v[106:113] /*v[618:625]*/, v[82:97], v30, v14
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[234:249] /*v[746:761]*/, v30, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[50:65], v33, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[66:81], v26, v14 matrix_b_scale:MATRIX_SCALE_ROW1
	v_add_nc_u32_e32 v18, 0x24100, v212 /*v724*/
	v_add_nc_u32_e32 v19, 0x24200, v212 /*v724*/
	v_add_nc_u32_e32 v20, 0x24300, v212 /*v724*/
	s_set_vgpr_msb 0xa52
	s_wait_dscnt 0xb
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[0:7], v[194:209] /*v[450:465]*/, v27, v8
	s_wait_tensorcnt 0x1
	s_set_vgpr_msb 0x5208
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_add_nc_u32_e32 v21, 0x24080, v195 /*v707*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[34:49] /*v[546:561]*/, v[0:7], v[242:257] /*v[498:513]*/, v28, v8
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v22, 0x24100, v195 /*v707*/
	v_add_nc_u32_e32 v23, 0x24180, v195 /*v707*/
	v_add_nc_u32_e32 v24, 0x24000, v214 /*v726*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[66:81] /*v[578:593]*/, v[0:7], v[226:241] /*v[482:497]*/, v17, v8
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v25, 0x24600, v214 /*v726*/
	v_add_nc_u32_e32 v26, 0x24800, v214 /*v726*/
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v50 /*v562*/, 0x24e00, v214 /*v726*/
	s_set_vgpr_msb 0x8852
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[82:97] /*v[594:609]*/, v[0:7], v[210:225] /*v[466:481]*/, v16, v8
	s_set_vgpr_msb 0x5200
	ds_load_b32 v12, v15
	ds_load_b32 v13, v18
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v54 /*v566*/, 0x25000, v214 /*v726*/
	s_set_vgpr_msb 0x8808
	ds_load_b32 v14, v19
	v_add_nc_u32_e32 v29, 0x25600, v214 /*v726*/
	ds_load_b32 v15, v20
	v_add_nc_u32_e32 v30, 0x25800, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[82:97] /*v[594:609]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v16, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v31, 0x25e00, v214 /*v726*/
	v_add_nc_u32_e32 v32, 0x26000, v214 /*v726*/
	v_add_nc_u32_e32 v33, 0x26600, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[66:81] /*v[578:593]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v17, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v110 /*v622*/, 0x26800, v214 /*v726*/
	v_add_nc_u32_e32 v114 /*v626*/, 0x26e00, v214 /*v726*/
	s_set_vgpr_msb 0x8808
	v_add_nc_u32_e32 v18, 0x27000, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v28, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v19, 0x27600, v214 /*v726*/
	v_add_nc_u32_e32 v2, 0x27800, v214 /*v726*/
	v_add_nc_u32_e32 v6, 0x24000, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v27, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a02
	ds_load_b32 v0, v221 /*v733*/
	s_set_vgpr_msb 0x208
	ds_load_b32 v1, v21
	v_add_nc_u32_e32 v20, 0x248e0, v213 /*v725*/
	ds_load_b32 v22, v22
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v134 /*v646*/, 0x24900, v213 /*v725*/
	s_set_vgpr_msb 0x8800
	ds_load_b32 v23, v23
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v186 /*v698*/, 0x251e0, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v27, v9
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v187 /*v699*/, 0x25200, v213 /*v725*/
	v_add_nc_u32_e32 v188 /*v700*/, 0x25ae0, v213 /*v725*/
	v_add_nc_u32_e32 v189 /*v701*/, 0x25b00, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v28, v9
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v190 /*v702*/, 0x263e0, v213 /*v725*/
	v_add_nc_u32_e32 v191 /*v703*/, 0x26400, v213 /*v725*/
	v_add_nc_u32_e32 v192 /*v704*/, 0x26ce0, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v17, v9
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v193 /*v705*/, 0x26d00, v213 /*v725*/
	v_add_nc_u32_e32 v215 /*v727*/, 0x275e0, v213 /*v725*/
	v_add_nc_u32_e32 v216 /*v728*/, 0x27600, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v16, v9
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[18:21] /*v[530:533]*/, v220 /*v732*/
	s_set_vgpr_msb 0x8288
	ds_load_b128 v[22:25] /*v[534:537]*/, v24
	v_add_nc_u32_e32 v217 /*v729*/, 0x27ee0, v213 /*v725*/
	ds_load_b128 v[26:29] /*v[538:541]*/, v25
	v_add_nc_u32_e32 v218 /*v730*/, 0x27f00, v213 /*v725*/
	ds_load_b128 v[30:33] /*v[542:545]*/, v26
	s_set_vgpr_msb 0x8882
	ds_load_b128 v[138:141] /*v[650:653]*/, v192 /*v704*/
	s_set_vgpr_msb 0x825a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[2:17] /*v[258:273]*/, v16, v9 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[142:145] /*v[654:657]*/, v193 /*v705*/
	s_set_vgpr_msb 0x825a
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[18:33] /*v[274:289]*/, v17, v9 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[34:49] /*v[290:305]*/, v28, v9 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[50:65] /*v[306:321]*/, v27, v9 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[50:53] /*v[562:565]*/, v50 /*v562*/
	ds_load_b128 v[54:57] /*v[566:569]*/, v54 /*v566*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[58:61] /*v[570:573]*/, v29
	ds_load_b128 v[62:65] /*v[574:577]*/, v30
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[98:105] /*v[610:617]*/, v[242:257], v27, v10
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[34:49] /*v[546:561]*/, v[98:105] /*v[610:617]*/, v[226:241], v28, v10
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[66:81] /*v[578:593]*/, v[98:105] /*v[610:617]*/, v[210:225], v17, v10
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[82:97] /*v[594:609]*/, v[98:105] /*v[610:617]*/, v[194:209], v16, v10
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[98:101] /*v[610:613]*/, v31
	ds_load_b128 v[102:105] /*v[614:617]*/, v32
	ds_load_b128 v[106:109] /*v[618:621]*/, v33
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[110:113] /*v[622:625]*/, v110 /*v622*/
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[82:97] /*v[594:609]*/, v[146:153] /*v[658:665]*/, v[130:145], v16, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[66:81] /*v[578:593]*/, v[146:153] /*v[658:665]*/, v[146:161], v17, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[162:177], v28, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[178:193], v27, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[114:117] /*v[626:629]*/, v114 /*v626*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[118:121] /*v[630:633]*/, v18
	ds_load_b128 v[122:125] /*v[634:637]*/, v19
	ds_load_b128 v[126:129] /*v[638:641]*/, v2
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[146:149] /*v[658:661]*/, v215 /*v727*/
	ds_load_b128 v[150:153] /*v[662:665]*/, v216 /*v728*/
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[114:129], v27, v11
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[98:113], v28, v11
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[66:81] /*v[578:593]*/, v[154:161] /*v[666:673]*/, v[82:97], v17, v11
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[82:97] /*v[594:609]*/, v[154:161] /*v[666:673]*/, v[34:49], v16, v11
	ds_load_b128 v[2:5], v219 /*v731*/
	s_set_vgpr_msb 0xa00
	ds_load_b128 v[6:9], v6
	s_set_vgpr_msb 0x80
	ds_load_b128 v[130:133] /*v[642:645]*/, v20
	s_set_vgpr_msb 0x80aa
	ds_load_b128 v[134:137] /*v[646:649]*/, v134 /*v646*/
	ds_load_b128 v[154:157] /*v[666:669]*/, v217 /*v729*/
	ds_load_b128 v[158:161] /*v[670:673]*/, v218 /*v730*/
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[196:211] /*v[708:723]*/, v16, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[90:93] /*v[602:605]*/, v190 /*v702*/
	ds_load_b128 v[94:97] /*v[606:609]*/, v191 /*v703*/
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[234:249] /*v[746:761]*/, v17, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[50:65], v28, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[66:81], v27, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa8a
	ds_load_b128 v[34:37] /*v[546:549]*/, v186 /*v698*/
	ds_load_b128 v[38:41] /*v[550:553]*/, v187 /*v699*/
	ds_load_b128 v[66:69] /*v[578:581]*/, v188 /*v700*/
	ds_load_b128 v[70:73] /*v[582:585]*/, v189 /*v701*/
	v_add_nc_u32_e32 v183 /*v695*/, 0x35fe0, v213 /*v725*/
	v_add_nc_u32_e32 v184 /*v696*/, 0x35e00, v214 /*v726*/
	v_add_nc_u32_e32 v185 /*v697*/, 0x36000, v195 /*v707*/
	s_set_vgpr_msb 0x8a08
	v_add_nc_u32_e32 v28, 0x36000, v212 /*v724*/
	s_wait_dscnt 0x27
	v_dual_mov_b32 v21, v12 :: v_dual_add_nc_u32 v10, 0x24080, v212 /*v724*/
	v_add_nc_u32_e32 v11, 0x24180, v212 /*v724*/
	v_add_nc_u32_e32 v12, 0x24280, v212 /*v724*/
	v_add_nc_u32_e32 v16, 0x24380, v212 /*v724*/
	v_add_nc_u32_e32 v19, 0x24200, v195 /*v707*/
	v_add_nc_u32_e32 v20, 0x24280, v195 /*v707*/
	s_set_vgpr_msb 0x852
	s_wait_dscnt 0xa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[18:33] /*v[530:545]*/, v[2:9], v[194:209] /*v[450:465]*/, v21, v0
	s_set_vgpr_msb 0x5208
	v_dual_mov_b32 v26, v21 :: v_dual_mov_b32 v27, v23
	v_add_nc_u32_e32 v24, 0x27f20, v213 /*v725*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[50:65] /*v[562:577]*/, v[2:9], v[242:257] /*v[498:513]*/, v13, v0
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v25, 0x27f40, v213 /*v725*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[2:9], v[226:241] /*v[482:497]*/, v14, v0
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[114:129] /*v[626:641]*/, v[2:9], v[210:225] /*v[466:481]*/, v15, v0
	s_set_vgpr_msb 0x5208
	ds_load_b32 v18, v10
	ds_load_b32 v31, v11
	ds_load_b32 v17, v12
	ds_load_b32 v16, v16
	ds_load_b32 v10, v19
	ds_load_b32 v11, v20
	v_add_nc_u32_e32 v19, 0x25220, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v15, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v20, 0x25240, v213 /*v725*/
	v_nop
	v_add_nc_u32_e32 v2, 0x24300, v195 /*v707*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v14, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v3, 0x24380, v195 /*v707*/
	v_add_nc_u32_e32 v4, 0x24200, v214 /*v726*/
	v_add_nc_u32_e32 v5, 0x24400, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v13, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v9, 0x24920, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v21, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a00
	ds_load_b32 v8, v2
	ds_load_b32 v12, v3
	s_set_vgpr_msb 0x80
	ds_load_b128 v[2:5] /*v[514:517]*/, v4
	ds_load_b128 v[6:9] /*v[518:521]*/, v5
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x24a00, v214 /*v726*/
	v_add_nc_u32_e32 v2, 0x25200, v214 /*v726*/
	v_add_nc_u32_e32 v3, 0x25400, v214 /*v726*/
	v_add_nc_u32_e32 v4, 0x25a00, v214 /*v726*/
	v_add_nc_u32_e32 v5, 0x25c00, v214 /*v726*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[10:13] /*v[522:525]*/, v0
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x24c00, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xd
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[18:33] /*v[530:545]*/, v[34:41] /*v[546:553]*/, v[114:129] /*v[370:385]*/, v21, v1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v21, 0x25b20, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[50:65] /*v[562:577]*/, v[34:41] /*v[546:553]*/, v[98:113] /*v[354:369]*/, v13, v1
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[34:41] /*v[546:553]*/, v[82:97] /*v[338:353]*/, v14, v1
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[34:41] /*v[546:553]*/, v[66:81] /*v[322:337]*/, v15, v1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[14:17] /*v[526:529]*/, v0
	ds_load_b128 v[34:37] /*v[546:549]*/, v2
	ds_load_b128 v[38:41] /*v[550:553]*/, v3
	ds_load_b128 v[42:45] /*v[554:557]*/, v4
	ds_load_b128 v[46:49] /*v[558:561]*/, v5
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x26200, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0x10
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[66:73] /*v[578:585]*/, v[2:17] /*v[258:273]*/, v15, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v2, 0x26400, v214 /*v726*/
	v_add_nc_u32_e32 v4, 0x26c00, v214 /*v726*/
	v_add_nc_u32_e32 v3, 0x26a00, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[66:73] /*v[578:585]*/, v[18:33] /*v[274:289]*/, v14, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v5, 0x27200, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[50:65] /*v[562:577]*/, v[66:73] /*v[578:585]*/, v[34:49] /*v[290:305]*/, v13, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[18:33] /*v[530:545]*/, v[66:73] /*v[578:585]*/, v[50:65] /*v[306:321]*/, v26, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[66:69] /*v[578:581]*/, v0
	ds_load_b128 v[70:73] /*v[582:585]*/, v2
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v1, 0x27400, v214 /*v726*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[78:81] /*v[590:593]*/, v4
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v0, 0x27a00, v214 /*v726*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[74:77] /*v[586:589]*/, v3
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v2, 0x24020, v213 /*v725*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[82:85] /*v[594:597]*/, v5
	ds_load_b128 v[86:89] /*v[598:601]*/, v1
	s_set_vgpr_msb 0x800a
	v_add_nc_u32_e32 v1, 0x27c00, v214 /*v726*/
	v_add_nc_u32_e32 v4, 0x24040, v213 /*v725*/
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[18:33] /*v[530:545]*/, v[90:97] /*v[602:609]*/, v[242:257], v26, v22
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[50:65] /*v[562:577]*/, v[90:97] /*v[602:609]*/, v[226:241], v13, v22
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[90:97] /*v[602:609]*/, v[210:225], v14, v22
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[90:97] /*v[602:609]*/, v[194:209], v15, v22
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[90:93] /*v[602:605]*/, v0
	ds_load_b128 v[94:97] /*v[606:609]*/, v1
	s_set_vgpr_msb 0x8000
	ds_load_b128 v[0:3], v2
	ds_load_b128 v[4:7], v4
	s_set_vgpr_msb 0x80
	ds_load_b128 v[170:173] /*v[682:685]*/, v9
	s_set_vgpr_msb 0x800a
	v_add_nc_u32_e32 v9, 0x24940, v213 /*v725*/
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[130:145], v15, v22 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[146:161], v14, v22 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[162:177], v13, v22 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[178:193], v26, v22 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[174:177] /*v[686:689]*/, v9
	ds_load_b128 v[130:133] /*v[642:645]*/, v19
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v22, 0x25b40, v213 /*v725*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[134:137] /*v[646:649]*/, v20
	ds_load_b128 v[162:165] /*v[674:677]*/, v21
	ds_load_b128 v[166:169] /*v[678:681]*/, v22
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[114:129], v26, v23
	v_add_nc_u32_e32 v9, 0x26420, v213 /*v725*/
	v_add_nc_u32_e32 v19, 0x26440, v213 /*v725*/
	v_add_nc_u32_e32 v20, 0x26d20, v213 /*v725*/
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[98:113], v13, v23
	v_add_nc_u32_e32 v21, 0x26d40, v213 /*v725*/
	v_add_nc_u32_e32 v22, 0x27620, v213 /*v725*/
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[138:141] /*v[650:653]*/, v22
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[146:153] /*v[658:665]*/, v[82:97], v14, v23
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[114:129] /*v[626:641]*/, v[146:153] /*v[658:665]*/, v[34:49], v15, v23
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[146:149] /*v[658:661]*/, v24
	ds_load_b128 v[150:153] /*v[662:665]*/, v25
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v23, 0x27640, v213 /*v725*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[142:145] /*v[654:657]*/, v23
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[196:211] /*v[708:723]*/, v15, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa80
	ds_load_b128 v[114:117] /*v[626:629]*/, v9
	ds_load_b128 v[118:121] /*v[630:633]*/, v19
	ds_load_b128 v[122:125] /*v[634:637]*/, v20
	ds_load_b128 v[126:129] /*v[638:641]*/, v21
	s_set_vgpr_msb 0x80aa
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[234:249] /*v[746:761]*/, v14, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[50:65], v13, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[66:81], v26, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	v_add_nc_u32_e32 v9, 0x36100, v212 /*v724*/
	v_add_nc_u32_e32 v19, 0x36200, v212 /*v724*/
	v_add_nc_u32_e32 v20, 0x36300, v212 /*v724*/
	v_add_nc_u32_e32 v23, 0x36180, v195 /*v707*/
	s_set_vgpr_msb 0xa52
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[0:7], v[194:209] /*v[450:465]*/, v18, v10
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x5208
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_add_nc_u32_e32 v21, 0x36080, v195 /*v707*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[34:49] /*v[546:561]*/, v[0:7], v[242:257] /*v[498:513]*/, v31, v10
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v22, 0x36100, v195 /*v707*/
	v_add_nc_u32_e32 v24, 0x36000, v214 /*v726*/
	v_add_nc_u32_e32 v25, 0x36600, v214 /*v726*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[66:81] /*v[578:593]*/, v[0:7], v[226:241] /*v[482:497]*/, v17, v10
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v26, 0x36800, v214 /*v726*/
	v_add_nc_u32_e32 v27, 0x36e00, v214 /*v726*/
	v_add_nc_u32_e32 v15, 0x37000, v214 /*v726*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[82:97] /*v[594:609]*/, v[0:7], v[210:225] /*v[466:481]*/, v16, v10
	s_set_vgpr_msb 0x5200
	ds_load_b32 v32, v28
	ds_load_b32 v33, v9
	ds_load_b32 v28, v19
	ds_load_b32 v29, v20
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v58 /*v570*/, 0x37600, v214 /*v726*/
	s_set_vgpr_msb 0x8808
	v_add_nc_u32_e32 v30, 0x37800, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[82:97] /*v[594:609]*/, v[170:177] /*v[682:689]*/, v[130:145] /*v[386:401]*/, v16, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v98 /*v610*/, 0x37e00, v214 /*v726*/
	s_set_vgpr_msb 0x8808
	v_add_nc_u32_e32 v13, 0x38000, v214 /*v726*/
	v_add_nc_u32_e32 v14, 0x38600, v214 /*v726*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[66:81] /*v[578:593]*/, v[170:177] /*v[682:689]*/, v[146:161] /*v[402:417]*/, v17, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v110 /*v622*/, 0x38800, v214 /*v726*/
	s_set_vgpr_msb 0x8808
	v_add_nc_u32_e32 v9, 0x38e00, v214 /*v726*/
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v186 /*v698*/, 0x39000, v214 /*v726*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[162:177] /*v[418:433]*/, v31, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v19, 0x39600, v214 /*v726*/
	v_add_nc_u32_e32 v2, 0x39800, v214 /*v726*/
	v_add_nc_u32_e32 v6, 0x36000, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[178:193] /*v[434:449]*/, v18, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a00
	ds_load_b32 v3, v23
	s_set_vgpr_msb 10
	ds_load_b32 v10, v185 /*v697*/
	v_add_nc_u32_e32 v20, 0x368e0, v213 /*v725*/
	s_set_vgpr_msb 0xa00
	ds_load_b32 v0, v21
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v154 /*v666*/, 0x36900, v213 /*v725*/
	s_set_vgpr_msb 0x8800
	ds_load_b32 v1, v22
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v155 /*v667*/, 0x371e0, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[114:129] /*v[370:385]*/, v18, v11
	s_set_vgpr_msb 0x5a00
	s_wait_dscnt 0x3
	scratch_store_b32 off, v3, off nv
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v156 /*v668*/, 0x37200, v213 /*v725*/
	v_add_nc_u32_e32 v157 /*v669*/, 0x37ae0, v213 /*v725*/
	v_add_nc_u32_e32 v158 /*v670*/, 0x37b00, v213 /*v725*/
	v_add_nc_u32_e32 v159 /*v671*/, 0x383e0, v213 /*v725*/
	v_add_nc_u32_e32 v160 /*v672*/, 0x38400, v213 /*v725*/
	v_add_nc_u32_e32 v161 /*v673*/, 0x38ce0, v213 /*v725*/
	v_add_nc_u32_e32 v178 /*v690*/, 0x38d00, v213 /*v725*/
	v_add_nc_u32_e32 v179 /*v691*/, 0x395e0, v213 /*v725*/
	v_add_nc_u32_e32 v180 /*v692*/, 0x39600, v213 /*v725*/
	v_add_nc_u32_e32 v181 /*v693*/, 0x39ee0, v213 /*v725*/
	v_add_nc_u32_e32 v182 /*v694*/, 0x39f00, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[98:113] /*v[354:369]*/, v31, v11
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[82:97] /*v[338:353]*/, v17, v11
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[66:81] /*v[322:337]*/, v16, v11
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[18:21] /*v[530:533]*/, v184 /*v696*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[22:25] /*v[534:537]*/, v24
	ds_load_b128 v[26:29] /*v[538:541]*/, v25
	ds_load_b128 v[30:33] /*v[542:545]*/, v26
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[2:17] /*v[258:273]*/, v16, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[18:33] /*v[274:289]*/, v17, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[34:49] /*v[290:305]*/, v31, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[50:65] /*v[306:321]*/, v18, v11 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[50:53] /*v[562:565]*/, v27
	ds_load_b128 v[54:57] /*v[566:569]*/, v15
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[58:61] /*v[570:573]*/, v58 /*v570*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[62:65] /*v[574:577]*/, v30
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[114:121] /*v[626:633]*/, v[242:257], v18, v8
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[34:49] /*v[546:561]*/, v[114:121] /*v[626:633]*/, v[226:241], v31, v8
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[66:81] /*v[578:593]*/, v[114:121] /*v[626:633]*/, v[210:225], v17, v8
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[82:97] /*v[594:609]*/, v[114:121] /*v[626:633]*/, v[194:209], v16, v8
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[98:101] /*v[610:613]*/, v98 /*v610*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[102:105] /*v[614:617]*/, v13
	ds_load_b128 v[106:109] /*v[618:621]*/, v14
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[110:113] /*v[622:625]*/, v110 /*v622*/
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[82:97] /*v[594:609]*/, v[122:129] /*v[634:641]*/, v[130:145], v16, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[66:81] /*v[578:593]*/, v[122:129] /*v[634:641]*/, v[146:161], v17, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[34:49] /*v[546:561]*/, v[122:129] /*v[634:641]*/, v[162:177], v31, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[122:129] /*v[634:641]*/, v[178:193], v18, v8 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[114:117] /*v[626:629]*/, v9
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[118:121] /*v[630:633]*/, v186 /*v698*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[122:125] /*v[634:637]*/, v19
	ds_load_b128 v[126:129] /*v[638:641]*/, v2
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129], v18, v12
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[98:113], v31, v12
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[82:97], v17, v12
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[34:49], v16, v12
	ds_load_b128 v[2:5], v183 /*v695*/
	s_set_vgpr_msb 0xa00
	ds_load_b128 v[6:9], v6
	s_set_vgpr_msb 0x80
	ds_load_b128 v[130:133] /*v[642:645]*/, v20
	s_set_vgpr_msb 0x80aa
	ds_load_b128 v[134:137] /*v[646:649]*/, v154 /*v666*/
	ds_load_b128 v[138:141] /*v[650:653]*/, v179 /*v691*/
	ds_load_b128 v[142:145] /*v[654:657]*/, v180 /*v692*/
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[82:97] /*v[594:609]*/, v[146:153] /*v[658:665]*/, v[196:211] /*v[708:723]*/, v16, v12 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[82:85] /*v[594:597]*/, v161 /*v673*/
	ds_load_b128 v[86:89] /*v[598:601]*/, v178 /*v690*/
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[66:81] /*v[578:593]*/, v[146:153] /*v[658:665]*/, v[234:249] /*v[746:761]*/, v17, v12 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[66:69] /*v[578:581]*/, v159 /*v671*/
	ds_load_b128 v[70:73] /*v[582:585]*/, v160 /*v672*/
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[50:65], v31, v12 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[66:81], v18, v12 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[34:37] /*v[546:549]*/, v155 /*v667*/
	ds_load_b128 v[38:41] /*v[550:553]*/, v156 /*v668*/
	ds_load_b128 v[42:45] /*v[554:557]*/, v157 /*v669*/
	ds_load_b128 v[46:49] /*v[558:561]*/, v158 /*v670*/
	ds_load_b128 v[154:157] /*v[666:669]*/, v181 /*v693*/
	ds_load_b128 v[158:161] /*v[670:673]*/, v182 /*v694*/
	s_set_vgpr_msb 0x8208
	v_add_nc_u32_e32 v11, 0x36080, v212 /*v724*/
	v_add_nc_u32_e32 v12, 0x36180, v212 /*v724*/
	v_add_nc_u32_e32 v13, 0x36280, v212 /*v724*/
	v_add_nc_u32_e32 v14, 0x36380, v212 /*v724*/
	v_add_nc_u32_e32 v18, 0x36300, v195 /*v707*/
	s_set_vgpr_msb 0x852
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[18:33] /*v[530:545]*/, v[2:9], v[194:209] /*v[450:465]*/, v32, v10
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v15, 0x36200, v195 /*v707*/
	v_add_nc_u32_e32 v17, 0x36280, v195 /*v707*/
	v_add_nc_u32_e32 v19, 0x36380, v195 /*v707*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[50:65] /*v[562:577]*/, v[2:9], v[242:257] /*v[498:513]*/, v33, v10
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v20, 0x36200, v214 /*v726*/
	v_add_nc_u32_e32 v21, 0x36400, v214 /*v726*/
	v_add_nc_u32_e32 v22, 0x36a00, v214 /*v726*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[98:113] /*v[610:625]*/, v[2:9], v[226:241] /*v[482:497]*/, v28, v10
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v23, 0x36c00, v214 /*v726*/
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v77 /*v589*/, 0x37200, v214 /*v726*/
	v_add_nc_u32_e32 v79 /*v591*/, 0x37400, v214 /*v726*/
	s_set_vgpr_msb 0x8852
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[114:129] /*v[626:641]*/, v[2:9], v[210:225] /*v[466:481]*/, v29, v10
	s_set_vgpr_msb 0x5200
	ds_load_b32 v30, v11
	ds_load_b32 v31, v12
	ds_load_b32 v24, v13
	ds_load_b32 v25, v14
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v80 /*v592*/, 0x37a00, v214 /*v726*/
	v_add_nc_u32_e32 v81 /*v593*/, 0x37c00, v214 /*v726*/
	s_set_vgpr_msb 0x885a
	s_wait_dscnt 0x10
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v29, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v74 /*v586*/, 0x38200, v214 /*v726*/
	v_add_nc_u32_e32 v75 /*v587*/, 0x38400, v214 /*v726*/
	v_add_nc_u32_e32 v76 /*v588*/, 0x38a00, v214 /*v726*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v28, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v78 /*v590*/, 0x38c00, v214 /*v726*/
	v_add_nc_u32_e32 v90 /*v602*/, 0x39200, v214 /*v726*/
	v_add_nc_u32_e32 v91 /*v603*/, 0x39400, v214 /*v726*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v33, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v11, 0x39a00, v214 /*v726*/
	v_add_nc_u32_e32 v12, 0x39c00, v214 /*v726*/
	v_add_nc_u32_e32 v13, 0x36020, v213 /*v725*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v32, v10 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a08
	ds_load_b32 v27, v18
	scratch_load_b32 v18, off, off th:TH_LOAD_LU nv
	v_add_nc_u32_e32 v4, 0x36040, v213 /*v725*/
	ds_load_b32 v16, v15
	v_add_nc_u32_e32 v8, 0x36920, v213 /*v725*/
	ds_load_b32 v17, v17
	v_add_nc_u32_e32 v14, 0x36940, v213 /*v725*/
	ds_load_b32 v26, v19
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[18:33] /*v[530:545]*/, v[34:41] /*v[546:553]*/, v[114:129] /*v[370:385]*/, v32, v0
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v146 /*v658*/, 0x37220, v213 /*v725*/
	v_add_nc_u32_e32 v147 /*v659*/, 0x37240, v213 /*v725*/
	v_add_nc_u32_e32 v148 /*v660*/, 0x37b20, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[50:65] /*v[562:577]*/, v[34:41] /*v[546:553]*/, v[98:113] /*v[354:369]*/, v33, v0
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v149 /*v661*/, 0x37b40, v213 /*v725*/
	v_add_nc_u32_e32 v150 /*v662*/, 0x38420, v213 /*v725*/
	v_add_nc_u32_e32 v151 /*v663*/, 0x38440, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[34:41] /*v[546:553]*/, v[82:97] /*v[338:353]*/, v28, v0
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v152 /*v664*/, 0x38d20, v213 /*v725*/
	v_add_nc_u32_e32 v153 /*v665*/, 0x38d40, v213 /*v725*/
	v_add_nc_u32_e32 v162 /*v674*/, 0x39620, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[34:41] /*v[546:553]*/, v[66:81] /*v[322:337]*/, v29, v0
	s_set_vgpr_msb 0x5a88
	ds_load_b128 v[2:5] /*v[514:517]*/, v20
	ds_load_b128 v[6:9] /*v[518:521]*/, v21
	v_add_nc_u32_e32 v163 /*v675*/, 0x39640, v213 /*v725*/
	ds_load_b128 v[10:13] /*v[522:525]*/, v22
	v_add_nc_u32_e32 v164 /*v676*/, 0x39f20, v213 /*v725*/
	ds_load_b128 v[14:17] /*v[526:529]*/, v23
	v_add_nc_u32_e32 v165 /*v677*/, 0x39f40, v213 /*v725*/
	s_set_vgpr_msb 0x885a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[42:49] /*v[554:561]*/, v[2:17] /*v[258:273]*/, v29, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[42:49] /*v[554:561]*/, v[18:33] /*v[274:289]*/, v28, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[50:65] /*v[562:577]*/, v[42:49] /*v[554:561]*/, v[34:49] /*v[290:305]*/, v33, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[18:33] /*v[530:545]*/, v[42:49] /*v[554:561]*/, v[50:65] /*v[306:321]*/, v32, v0 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a82
	ds_load_b128 v[34:37] /*v[546:549]*/, v77 /*v589*/
	ds_load_b128 v[38:41] /*v[550:553]*/, v79 /*v591*/
	ds_load_b128 v[42:45] /*v[554:557]*/, v80 /*v592*/
	ds_load_b128 v[46:49] /*v[558:561]*/, v81 /*v593*/
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[18:33] /*v[530:545]*/, v[66:73] /*v[578:585]*/, v[242:257], v32, v1
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[50:65] /*v[562:577]*/, v[66:73] /*v[578:585]*/, v[226:241], v33, v1
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[66:73] /*v[578:585]*/, v[210:225], v28, v1
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[66:73] /*v[578:585]*/, v[194:209], v29, v1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[66:69] /*v[578:581]*/, v74 /*v586*/
	ds_load_b128 v[70:73] /*v[582:585]*/, v75 /*v587*/
	ds_load_b128 v[74:77] /*v[586:589]*/, v76 /*v588*/
	ds_load_b128 v[78:81] /*v[590:593]*/, v78 /*v590*/
	s_set_vgpr_msb 0x820a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[82:89] /*v[594:601]*/, v[130:145], v29, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[82:89] /*v[594:601]*/, v[146:161], v28, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[50:65] /*v[562:577]*/, v[82:89] /*v[594:601]*/, v[162:177], v33, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[18:33] /*v[530:545]*/, v[82:89] /*v[594:601]*/, v[178:193], v32, v1 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[82:85] /*v[594:597]*/, v90 /*v602*/
	ds_load_b128 v[86:89] /*v[598:601]*/, v91 /*v603*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[90:93] /*v[602:605]*/, v11
	ds_load_b128 v[94:97] /*v[606:609]*/, v12
	s_set_vgpr_msb 0x800a
	s_wait_loadcnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[114:129], v32, v18
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[98:113], v33, v18
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[82:97], v28, v18
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[34:49], v29, v18
	s_set_vgpr_msb 0xa00
	ds_load_b128 v[0:3], v13
	ds_load_b128 v[4:7], v4
	ds_load_b128 v[8:11], v8
	ds_load_b128 v[12:15], v14
	s_set_vgpr_msb 0xaa
	s_wait_dscnt 0x1c
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[196:211] /*v[708:723]*/, v29, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[114:117] /*v[626:629]*/, v162 /*v674*/
	ds_load_b128 v[118:121] /*v[630:633]*/, v163 /*v675*/
	ds_load_b128 v[122:125] /*v[634:637]*/, v164 /*v676*/
	ds_load_b128 v[126:129] /*v[638:641]*/, v165 /*v677*/
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[234:249] /*v[746:761]*/, v28, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[98:101] /*v[610:613]*/, v150 /*v662*/
	ds_load_b128 v[102:105] /*v[614:617]*/, v151 /*v663*/
	ds_load_b128 v[106:109] /*v[618:621]*/, v152 /*v664*/
	ds_load_b128 v[110:113] /*v[622:625]*/, v153 /*v665*/
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[50:65], v33, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[66:81], v32, v18 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[50:53] /*v[562:565]*/, v146 /*v658*/
	ds_load_b128 v[54:57] /*v[566:569]*/, v147 /*v659*/
	ds_load_b128 v[58:61] /*v[570:573]*/, v148 /*v660*/
	ds_load_b128 v[62:65] /*v[574:577]*/, v149 /*v661*/
	s_set_vgpr_msb 0x8252
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[2:17] /*v[514:529]*/, v[0:7], v[194:209] /*v[450:465]*/, v30, v16
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[34:49] /*v[546:561]*/, v[0:7], v[242:257] /*v[498:513]*/, v31, v16
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[66:81] /*v[578:593]*/, v[0:7], v[226:241] /*v[482:497]*/, v24, v16
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[82:97] /*v[594:609]*/, v[0:7], v[210:225] /*v[466:481]*/, v25, v16
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[82:97] /*v[594:609]*/, v[8:15], v[130:145] /*v[386:401]*/, v25, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[66:81] /*v[578:593]*/, v[8:15], v[146:161] /*v[402:417]*/, v24, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[34:49] /*v[546:561]*/, v[8:15], v[162:177] /*v[418:433]*/, v31, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[8:15], v[178:193] /*v[434:449]*/, v30, v16 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x525a
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[50:57] /*v[562:569]*/, v[114:129] /*v[370:385]*/, v30, v17
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[34:49] /*v[546:561]*/, v[50:57] /*v[562:569]*/, v[98:113] /*v[354:369]*/, v31, v17
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[66:81] /*v[578:593]*/, v[50:57] /*v[562:569]*/, v[82:97] /*v[338:353]*/, v24, v17
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[82:97] /*v[594:609]*/, v[50:57] /*v[562:569]*/, v[66:81] /*v[322:337]*/, v25, v17
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[82:97] /*v[594:609]*/, v[58:65] /*v[570:577]*/, v[2:17] /*v[258:273]*/, v25, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[66:81] /*v[578:593]*/, v[58:65] /*v[570:577]*/, v[18:33] /*v[274:289]*/, v24, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[34:49] /*v[546:561]*/, v[58:65] /*v[570:577]*/, v[34:49] /*v[290:305]*/, v31, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[58:65] /*v[570:577]*/, v[50:65] /*v[306:321]*/, v30, v17 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0x5a0a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[98:105] /*v[610:617]*/, v[242:257], v30, v27
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[34:49] /*v[546:561]*/, v[98:105] /*v[610:617]*/, v[226:241], v31, v27
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[66:81] /*v[578:593]*/, v[98:105] /*v[610:617]*/, v[210:225], v24, v27
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[82:97] /*v[594:609]*/, v[98:105] /*v[610:617]*/, v[194:209], v25, v27
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[82:97] /*v[594:609]*/, v[106:113] /*v[618:625]*/, v[130:145], v25, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[66:81] /*v[578:593]*/, v[106:113] /*v[618:625]*/, v[146:161], v24, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[34:49] /*v[546:561]*/, v[106:113] /*v[618:625]*/, v[162:177], v31, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[106:113] /*v[618:625]*/, v[178:193], v30, v27 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[114:121] /*v[626:633]*/, v[114:129], v30, v26
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[34:49] /*v[546:561]*/, v[114:121] /*v[626:633]*/, v[98:113], v31, v26
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[66:81] /*v[578:593]*/, v[114:121] /*v[626:633]*/, v[82:97], v24, v26
	v_wmma_scale_f32_32x16x128_f4 v[34:49], v[82:97] /*v[594:609]*/, v[114:121] /*v[626:633]*/, v[34:49], v25, v26
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[196:211] /*v[708:723]*/, v[82:97] /*v[594:609]*/, v[122:129] /*v[634:641]*/, v[196:211] /*v[708:723]*/, v25, v26 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[234:249] /*v[746:761]*/, v[66:81] /*v[578:593]*/, v[122:129] /*v[634:641]*/, v[234:249] /*v[746:761]*/, v24, v26 matrix_b_scale:MATRIX_SCALE_ROW1
	s_set_vgpr_msb 0xaa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[34:49] /*v[546:561]*/, v[122:129] /*v[634:641]*/, v[50:65], v31, v26 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[2:17] /*v[514:529]*/, v[122:129] /*v[634:641]*/, v[66:81], v30, v26 matrix_b_scale:MATRIX_SCALE_ROW1
	v_mul_lo_u32 v17, 0x110, v233 /*v745*/
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_lshl_or_b32 v16, v232 /*v744*/, 3, s40
	s_set_vgpr_msb 0xa05
	v_cvt_pk_bf16_f32 v3, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v2, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v1, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v0, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v7, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v6, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v5, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v4, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v11, v248 /*v504*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v10, v246 /*v502*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v9, v244 /*v500*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v8, v242 /*v498*/, v243 /*v499*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v15, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0xa05
	v_cvt_pk_bf16_f32 v14, v254 /*v510*/, v255 /*v511*/
	v_cvt_pk_bf16_f32 v13, v252 /*v508*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v12, v250 /*v506*/, v251 /*v507*/
	s_set_vgpr_msb 0x500
	v_add_lshl_u32 v20, v17, v16, 1
	s_set_vgpr_msb 2
	s_barrier_wait -1
	v_or3_b32 v17, v194 /*v706*/, s39, 0x70
	s_mul_u64 s[0:1], s[22:23], s[20:21]
	s_bfe_u32 s4, ttmp8, 0x50019
	s_set_vgpr_msb 0x245
	v_cvt_pk_bf16_f32 v197 /*v453*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v230 /*v486*/, v231 /*v487*/
	s_set_vgpr_msb 0x4500
	v_mul_lo_u32 v17, 0x110, v17
	s_lshl_b64 s[0:1], s[0:1], 1
	s_and_b32 s4, s4, 3
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v195 /*v451*/, v228 /*v484*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v226 /*v482*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v201 /*v457*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v238 /*v494*/, v239 /*v495*/
	s_set_vgpr_msb 0x4500
	ds_store_b128 v20, v[0:3]
	ds_store_b128 v20, v[4:7] offset:32
	ds_store_b128 v20, v[8:11] offset:64
	ds_store_b128 v20, v[12:15] offset:96
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v3, v120 /*v376*/, v121 /*v377*/
	v_cvt_pk_bf16_f32 v2, v118 /*v374*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v1, v116 /*v372*/, v117 /*v373*/
	v_cvt_pk_bf16_f32 v0, v114 /*v370*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v7, v128 /*v384*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v6, v126 /*v382*/, v127 /*v383*/
	v_cvt_pk_bf16_f32 v5, v124 /*v380*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v4, v122 /*v378*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v11, v104 /*v360*/, v105 /*v361*/
	v_cvt_pk_bf16_f32 v10, v102 /*v358*/, v103 /*v359*/
	v_cvt_pk_bf16_f32 v9, v100 /*v356*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v8, v98 /*v354*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v15, v112 /*v368*/, v113 /*v369*/
	v_cvt_pk_bf16_f32 v14, v110 /*v366*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v13, v108 /*v364*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v12, v106 /*v362*/, v107 /*v363*/
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v199 /*v455*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v216 /*v472*/, v217 /*v473*/
	s_set_vgpr_msb 0x4500
	ds_store_b128 v20, v[0:3] offset:17408
	v_cvt_pk_bf16_f32 v3, v248, v249
	v_cvt_pk_bf16_f32 v2, v246, v247
	v_cvt_pk_bf16_f32 v1, v244, v245
	ds_store_b128 v20, v[4:7] offset:17440
	v_cvt_pk_bf16_f32 v0, v242, v243
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v7, v0 /*v256*/, v1 /*v257*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v6, v254, v255
	ds_store_b128 v20, v[8:11] offset:17472
	v_cvt_pk_bf16_f32 v5, v252, v253
	v_cvt_pk_bf16_f32 v4, v250, v251
	v_cvt_pk_bf16_f32 v11, v232, v233
	ds_store_b128 v20, v[12:15] offset:17504
	v_cvt_pk_bf16_f32 v10, v230, v231
	v_cvt_pk_bf16_f32 v9, v228, v229
	v_cvt_pk_bf16_f32 v8, v226, v227
	v_cvt_pk_bf16_f32 v15, v240, v241
	v_cvt_pk_bf16_f32 v14, v238, v239
	v_cvt_pk_bf16_f32 v13, v236, v237
	v_cvt_pk_bf16_f32 v12, v234, v235
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v204 /*v460*/, v214 /*v470*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v203 /*v459*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v209 /*v465*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v183 /*v439*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v180 /*v436*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v178 /*v434*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v153 /*v409*/, v152 /*v408*/, v153 /*v409*/
	v_cvt_pk_bf16_f32 v152 /*v408*/, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v151 /*v407*/, v148 /*v404*/, v149 /*v405*/
	v_cvt_pk_bf16_f32 v150 /*v406*/, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v149 /*v405*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v148 /*v404*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v147 /*v403*/, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v146 /*v402*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v137 /*v393*/, v136 /*v392*/, v137 /*v393*/
	v_cvt_pk_bf16_f32 v136 /*v392*/, v134 /*v390*/, v135 /*v391*/
	v_cvt_pk_bf16_f32 v135 /*v391*/, v132 /*v388*/, v133 /*v389*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v130 /*v386*/, v131 /*v387*/
	v_cvt_pk_bf16_f32 v133 /*v389*/, v144 /*v400*/, v145 /*v401*/
	v_cvt_pk_bf16_f32 v132 /*v388*/, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v131 /*v387*/, v140 /*v396*/, v141 /*v397*/
	v_cvt_pk_bf16_f32 v130 /*v386*/, v138 /*v394*/, v139 /*v395*/
	v_cvt_pk_bf16_f32 v89 /*v345*/, v88 /*v344*/, v89 /*v345*/
	v_cvt_pk_bf16_f32 v88 /*v344*/, v86 /*v342*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v87 /*v343*/, v84 /*v340*/, v85 /*v341*/
	v_cvt_pk_bf16_f32 v86 /*v342*/, v82 /*v338*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v85 /*v341*/, v96 /*v352*/, v97 /*v353*/
	v_cvt_pk_bf16_f32 v84 /*v340*/, v94 /*v350*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v83 /*v339*/, v92 /*v348*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v82 /*v338*/, v90 /*v346*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v73 /*v329*/, v72 /*v328*/, v73 /*v329*/
	v_cvt_pk_bf16_f32 v72 /*v328*/, v70 /*v326*/, v71 /*v327*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v68 /*v324*/, v69 /*v325*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v66 /*v322*/, v67 /*v323*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v80 /*v336*/, v81 /*v337*/
	v_cvt_pk_bf16_f32 v68 /*v324*/, v78 /*v334*/, v79 /*v335*/
	v_cvt_pk_bf16_f32 v67 /*v323*/, v76 /*v332*/, v77 /*v333*/
	v_cvt_pk_bf16_f32 v66 /*v322*/, v74 /*v330*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v56 /*v312*/, v57 /*v313*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v54 /*v310*/, v55 /*v311*/
	v_cvt_pk_bf16_f32 v55 /*v311*/, v52 /*v308*/, v53 /*v309*/
	v_cvt_pk_bf16_f32 v54 /*v310*/, v50 /*v306*/, v51 /*v307*/
	v_cvt_pk_bf16_f32 v53 /*v309*/, v64 /*v320*/, v65 /*v321*/
	v_cvt_pk_bf16_f32 v52 /*v308*/, v62 /*v318*/, v63 /*v319*/
	v_cvt_pk_bf16_f32 v51 /*v307*/, v60 /*v316*/, v61 /*v317*/
	v_cvt_pk_bf16_f32 v50 /*v306*/, v58 /*v314*/, v59 /*v315*/
	v_cvt_pk_bf16_f32 v41 /*v297*/, v40 /*v296*/, v41 /*v297*/
	v_cvt_pk_bf16_f32 v40 /*v296*/, v38 /*v294*/, v39 /*v295*/
	v_cvt_pk_bf16_f32 v39 /*v295*/, v36 /*v292*/, v37 /*v293*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v34 /*v290*/, v35 /*v291*/
	v_cvt_pk_bf16_f32 v37 /*v293*/, v48 /*v304*/, v49 /*v305*/
	v_cvt_pk_bf16_f32 v36 /*v292*/, v46 /*v302*/, v47 /*v303*/
	v_cvt_pk_bf16_f32 v35 /*v291*/, v44 /*v300*/, v45 /*v301*/
	v_cvt_pk_bf16_f32 v34 /*v290*/, v42 /*v298*/, v43 /*v299*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v24 /*v280*/, v25 /*v281*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v22 /*v278*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v23 /*v279*/, v20 /*v276*/, v21 /*v277*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v18 /*v274*/, v19 /*v275*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v32 /*v288*/, v33 /*v289*/
	v_cvt_pk_bf16_f32 v20 /*v276*/, v30 /*v286*/, v31 /*v287*/
	v_cvt_pk_bf16_f32 v19 /*v275*/, v28 /*v284*/, v29 /*v285*/
	v_cvt_pk_bf16_f32 v18 /*v274*/, v26 /*v282*/, v27 /*v283*/
	v_cvt_pk_bf16_f32 v9 /*v265*/, v8 /*v264*/, v9 /*v265*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v6 /*v262*/, v7 /*v263*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v4 /*v260*/, v5 /*v261*/
	v_cvt_pk_bf16_f32 v6 /*v262*/, v2 /*v258*/, v3 /*v259*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v16 /*v272*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v4 /*v260*/, v14 /*v270*/, v15 /*v271*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v12 /*v268*/, v13 /*v269*/
	v_cvt_pk_bf16_f32 v2 /*v258*/, v10 /*v266*/, v11 /*v267*/
	s_set_vgpr_msb 0x4500
	v_cvt_pk_bf16_f32 v217, v216, v217
	v_cvt_pk_bf16_f32 v216, v214, v215
	v_cvt_pk_bf16_f32 v215, v212, v213
	v_cvt_pk_bf16_f32 v214, v210, v211
	v_cvt_pk_bf16_f32 v213, v224, v225
	v_cvt_pk_bf16_f32 v212, v222, v223
	v_cvt_pk_bf16_f32 v211, v220, v221
	v_cvt_pk_bf16_f32 v210, v218, v219
	v_cvt_pk_bf16_f32 v201, v200, v201
	v_cvt_pk_bf16_f32 v200, v198, v199
	v_cvt_pk_bf16_f32 v199, v196, v197
	v_cvt_pk_bf16_f32 v198, v194, v195
	v_cvt_pk_bf16_f32 v197, v208, v209
	v_cvt_pk_bf16_f32 v196, v206, v207
	v_cvt_pk_bf16_f32 v195, v204, v205
	v_cvt_pk_bf16_f32 v194, v202, v203
	v_cvt_pk_bf16_f32 v185, v184, v185
	v_cvt_pk_bf16_f32 v184, v182, v183
	v_cvt_pk_bf16_f32 v183, v180, v181
	v_cvt_pk_bf16_f32 v182, v178, v179
	v_cvt_pk_bf16_f32 v181, v192, v193
	v_cvt_pk_bf16_f32 v180, v190, v191
	v_cvt_pk_bf16_f32 v179, v188, v189
	v_cvt_pk_bf16_f32 v178, v186, v187
	v_cvt_pk_bf16_f32 v169, v168, v169
	v_cvt_pk_bf16_f32 v168, v166, v167
	v_cvt_pk_bf16_f32 v167, v164, v165
	v_cvt_pk_bf16_f32 v166, v162, v163
	v_cvt_pk_bf16_f32 v165, v176, v177
	v_cvt_pk_bf16_f32 v164, v174, v175
	v_cvt_pk_bf16_f32 v163, v172, v173
	v_cvt_pk_bf16_f32 v162, v170, v171
	v_cvt_pk_bf16_f32 v153, v152, v153
	v_cvt_pk_bf16_f32 v152, v150, v151
	v_cvt_pk_bf16_f32 v151, v148, v149
	v_cvt_pk_bf16_f32 v150, v146, v147
	v_cvt_pk_bf16_f32 v149, v160, v161
	v_cvt_pk_bf16_f32 v148, v158, v159
	v_cvt_pk_bf16_f32 v147, v156, v157
	v_cvt_pk_bf16_f32 v146, v154, v155
	v_cvt_pk_bf16_f32 v137, v136, v137
	v_cvt_pk_bf16_f32 v136, v134, v135
	v_cvt_pk_bf16_f32 v135, v132, v133
	v_cvt_pk_bf16_f32 v134, v130, v131
	v_cvt_pk_bf16_f32 v133, v144, v145
	v_cvt_pk_bf16_f32 v132, v142, v143
	v_cvt_pk_bf16_f32 v131, v140, v141
	v_cvt_pk_bf16_f32 v130, v138, v139
	ds_store_b128 v20, v[0:3] offset:34816
	ds_store_b128 v20, v[4:7] offset:34848
	ds_store_b128 v20, v[8:11] offset:34880
	ds_store_b128 v20, v[12:15] offset:34912
	v_cvt_pk_bf16_f32 v3, v120, v121
	v_cvt_pk_bf16_f32 v2, v118, v119
	v_cvt_pk_bf16_f32 v1, v116, v117
	v_cvt_pk_bf16_f32 v0, v114, v115
	v_cvt_pk_bf16_f32 v7, v128, v129
	v_cvt_pk_bf16_f32 v6, v126, v127
	v_cvt_pk_bf16_f32 v5, v124, v125
	v_cvt_pk_bf16_f32 v4, v122, v123
	v_cvt_pk_bf16_f32 v11, v104, v105
	v_cvt_pk_bf16_f32 v10, v102, v103
	v_cvt_pk_bf16_f32 v9, v100, v101
	v_cvt_pk_bf16_f32 v8, v98, v99
	v_cvt_pk_bf16_f32 v15, v112, v113
	v_cvt_pk_bf16_f32 v14, v110, v111
	v_cvt_pk_bf16_f32 v13, v108, v109
	v_cvt_pk_bf16_f32 v12, v106, v107
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	v_cvt_pk_bf16_f32 v87, v84, v85
	v_cvt_pk_bf16_f32 v86, v82, v83
	v_cvt_pk_bf16_f32 v85, v96, v97
	v_cvt_pk_bf16_f32 v84, v94, v95
	v_cvt_pk_bf16_f32 v83, v92, v93
	v_cvt_pk_bf16_f32 v82, v90, v91
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_cvt_pk_bf16_f32 v38, v34, v35
	v_cvt_pk_bf16_f32 v37, v48, v49
	v_cvt_pk_bf16_f32 v36, v46, v47
	v_cvt_pk_bf16_f32 v35, v44, v45
	v_cvt_pk_bf16_f32 v34, v42, v43
	v_cvt_pk_bf16_f32 v45, v72, v73
	v_cvt_pk_bf16_f32 v44, v70, v71
	v_cvt_pk_bf16_f32 v43, v68, v69
	v_cvt_pk_bf16_f32 v42, v66, v67
	v_add_lshl_u32 v21, v17, v16, 1
	v_cvt_pk_bf16_f32 v49, v80, v81
	v_cvt_pk_bf16_f32 v48, v78, v79
	v_cvt_pk_bf16_f32 v47, v76, v77
	v_cvt_pk_bf16_f32 v46, v74, v75
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v55, v52, v53
	v_cvt_pk_bf16_f32 v54, v50, v51
	v_cvt_pk_bf16_f32 v53, v64, v65
	v_cvt_pk_bf16_f32 v52, v62, v63
	v_cvt_pk_bf16_f32 v51, v60, v61
	v_cvt_pk_bf16_f32 v50, v58, v59
	s_set_vgpr_msb 10
	v_cvt_pk_bf16_f32 v25, v240 /*v752*/, v241 /*v753*/
	v_cvt_pk_bf16_f32 v24, v238 /*v750*/, v239 /*v751*/
	v_cvt_pk_bf16_f32 v23, v236 /*v748*/, v237 /*v749*/
	v_cvt_pk_bf16_f32 v22, v234 /*v746*/, v235 /*v747*/
	v_cvt_pk_bf16_f32 v19, v248 /*v760*/, v249 /*v761*/
	v_cvt_pk_bf16_f32 v18, v246 /*v758*/, v247 /*v759*/
	v_cvt_pk_bf16_f32 v17, v244 /*v756*/, v245 /*v757*/
	v_cvt_pk_bf16_f32 v16, v242 /*v754*/, v243 /*v755*/
	v_cvt_pk_bf16_f32 v29, v202 /*v714*/, v203 /*v715*/
	v_cvt_pk_bf16_f32 v28, v200 /*v712*/, v201 /*v713*/
	v_cvt_pk_bf16_f32 v27, v198 /*v710*/, v199 /*v711*/
	v_cvt_pk_bf16_f32 v26, v196 /*v708*/, v197 /*v709*/
	v_cvt_pk_bf16_f32 v33, v210 /*v722*/, v211 /*v723*/
	v_cvt_pk_bf16_f32 v32, v208 /*v720*/, v209 /*v721*/
	v_cvt_pk_bf16_f32 v31, v206 /*v718*/, v207 /*v719*/
	v_cvt_pk_bf16_f32 v30, v204 /*v716*/, v205 /*v717*/
	s_lshl_b64 s[2:3], s[24:25], 1
	s_lshl_b32 s6, s4, 7
	s_mov_b32 s7, 0
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mul_u64 s[2:3], s[6:7], s[20:21]
	s_lshl_b32 s5, s4, 6
	s_add_nc_u64 s[10:11], s[2:3], s[0:1]
	s_sub_co_i32 s0, s33, s5
	s_mov_b32 s8, 1
	s_max_i32 s0, s0, 0
	s_mul_i32 s9, s4, 0x8800
	s_lshr_b32 s1, s0, 16
	s_lshl_b32 s2, s0, 16
	s_or_b32 s3, s1, 0x1100000
	s_and_b32 s0, s38, exec_lo
	s_bitset1_b32 s11, 31
	s_cselect_b32 s6, 0xffff, 0
	s_mov_b32 s4, 64
	s_mov_b32 s1, 0x1000000
	s_mov_b32 s0, 0x10000
	s_mov_b32 s5, s20
	s_set_vgpr_msb 0xa04
	ds_store_b128 v20, v[194:197] /*v[450:453]*/ offset:128
	ds_store_b128 v20, v[198:201] /*v[454:457]*/ offset:160
	ds_store_b128 v20, v[202:205] /*v[458:461]*/ offset:192
	ds_store_b128 v20, v[206:209] /*v[462:465]*/ offset:224
	ds_store_b128 v20, v[182:185] /*v[438:441]*/ offset:8704
	ds_store_b128 v20, v[178:181] /*v[434:437]*/ offset:8736
	ds_store_b128 v20, v[166:169] /*v[422:425]*/ offset:8768
	ds_store_b128 v20, v[162:165] /*v[418:421]*/ offset:8800
	ds_store_b128 v20, v[150:153] /*v[406:409]*/ offset:8832
	ds_store_b128 v20, v[146:149] /*v[402:405]*/ offset:8864
	ds_store_b128 v20, v[134:137] /*v[390:393]*/ offset:8896
	ds_store_b128 v20, v[130:133] /*v[386:389]*/ offset:8928
	ds_store_b128 v20, v[86:89] /*v[342:345]*/ offset:17536
	ds_store_b128 v20, v[82:85] /*v[338:341]*/ offset:17568
	ds_store_b128 v20, v[70:73] /*v[326:329]*/ offset:17600
	ds_store_b128 v20, v[66:69] /*v[322:325]*/ offset:17632
	ds_store_b128 v20, v[54:57] /*v[310:313]*/ offset:26112
	ds_store_b128 v20, v[50:53] /*v[306:309]*/ offset:26144
	ds_store_b128 v20, v[38:41] /*v[294:297]*/ offset:26176
	ds_store_b128 v20, v[34:37] /*v[290:293]*/ offset:26208
	ds_store_b128 v20, v[22:25] /*v[278:281]*/ offset:26240
	ds_store_b128 v20, v[18:21] /*v[274:277]*/ offset:26272
	ds_store_b128 v20, v[6:9] /*v[262:265]*/ offset:26304
	ds_store_b128 v20, v[2:5] /*v[258:261]*/ offset:26336
	s_set_vgpr_msb 0x400
	ds_store_b128 v20, v[214:217] offset:34944
	ds_store_b128 v20, v[210:213] offset:34976
	ds_store_b128 v20, v[198:201] offset:35008
	ds_store_b128 v20, v[194:197] offset:35040
	ds_store_b128 v20, v[182:185] offset:43520
	ds_store_b128 v20, v[178:181] offset:43552
	ds_store_b128 v20, v[166:169] offset:43584
	ds_store_b128 v20, v[162:165] offset:43616
	ds_store_b128 v20, v[150:153] offset:43648
	ds_store_b128 v20, v[146:149] offset:43680
	ds_store_b128 v20, v[134:137] offset:43712
	ds_store_b128 v20, v[130:133] offset:43744
	ds_store_b128 v20, v[0:3] offset:52224
	ds_store_b128 v20, v[4:7] offset:52256
	ds_store_b128 v20, v[8:11] offset:52288
	ds_store_b128 v20, v[12:15] offset:52320
	ds_store_b128 v20, v[86:89] offset:52352
	ds_store_b128 v20, v[82:85] offset:52384
	ds_store_b128 v20, v[38:41] offset:52416
	ds_store_b128 v20, v[34:37] offset:52448
	ds_store_b128 v21, v[42:45]
	ds_store_b128 v21, v[46:49] offset:32
	ds_store_b128 v21, v[54:57] offset:64
	ds_store_b128 v21, v[50:53] offset:96
	ds_store_b128 v21, v[22:25] offset:128
	ds_store_b128 v21, v[16:19] offset:160
	ds_store_b128 v21, v[26:29] offset:192
	ds_store_b128 v21, v[30:33] offset:224
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_store_from_lds s[8:11], s[0:7]
	s_wait_tensorcnt 0x0
.LBB0_47:
	s_endpgm
.Lfunc_end0:
	.size	a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1, .Lfunc_end0-a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1
		.amdhsa_group_segment_fixed_size 294912
		.amdhsa_private_segment_fixed_size 8
		.amdhsa_kernarg_size 224
		.amdhsa_user_sgpr_count 4
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 2
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 764
		.amdhsa_next_free_sgpr 52
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.num_vgpr, 764
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.num_agpr, 0
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.numbered_sgpr, 52
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.num_named_barrier, 0
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.private_seg_size, 8
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.uses_vcc, 1
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.uses_flat_scratch, 0
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.has_dyn_sized_stack, 0
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.has_recursion, 0
	.set .La8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .offset:         136
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         168
        .size:           8
        .value_kind:     global_buffer
      - .offset:         176
        .size:           28
        .value_kind:     by_value
      - .offset:         204
        .size:           4
        .value_kind:     by_value
      - .offset:         208
        .size:           4
        .value_kind:     by_value
      - .offset:         212
        .size:           4
        .value_kind:     by_value
      - .offset:         216
        .size:           4
        .value_kind:     by_value
      - .offset:         220
        .size:           4
        .value_kind:     by_value
    .cluster_dims:
      - 4
      - 1
      - 1
    .group_segment_fixed_size: 294912
    .kernarg_segment_align: 8
    .kernarg_segment_size: 224
    .max_flat_workgroup_size: 128
    .name:           a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1
    .private_segment_fixed_size: 8
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     764
    .vgpr_spill_count: 1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
