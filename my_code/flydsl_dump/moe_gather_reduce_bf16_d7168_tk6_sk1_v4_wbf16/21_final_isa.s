	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
	.p2align	8
	.type	moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16,@function
moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_mov_b64 s[16:17], s[6:7]
	s_add_co_i32 s0, s0, 1
	s_and_b32 s1, ttmp6, 15
	s_mul_i32 s6, ttmp9, s0
	s_getreg_b32 s0, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s1, s1, s6
	s_cmp_eq_u32 s0, 0
	s_mov_b32 s15, 0
	s_cselect_b32 s21, ttmp9, s1
	s_or_b64 s[12:13], s[12:13], 0xfe00000000000000
	s_mov_b32 s14, 0x1ffffff
	buffer_load_b32 v1, off, s[12:15], null
	s_wait_loadcnt 0x0
	v_cmp_ge_u32_e32 vcc_lo, s21, v1
	s_cbranch_vccnz .LBB0_66
	s_bfe_u32 s1, ttmp6, 0x40010
	s_bfe_u32 s6, ttmp6, 0x40004
	s_add_co_i32 s1, s1, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s1, ttmp7, s1
	s_add_co_i32 s6, s6, s1
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s0, ttmp7, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s0, s0, 10
	v_lshl_or_b32 v33, v0, 2, s0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e32 0xe00, v33
	s_cbranch_execz .LBB0_66
	s_mul_i32 s23, s21, 24
	s_or_b64 s[4:5], s[4:5], 0xfe00000000000000
	v_dual_mov_b32 v0, s23 :: v_dual_mov_b32 v7, 0
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mul_i32 s24, s21, 12
	buffer_load_b32 v6, v0, s[12:15], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s24
	s_or_b32 s17, s17, 0xfe000000
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	v_lshlrev_b32_e32 v31, 2, v33
	buffer_load_u16 v2, v0, s[16:19], null offen
	s_mov_b64 s[10:11], 0x1ffffff
	s_mov_b64 s[6:7], 0x1ffffff
	s_or_b32 s9, s9, 0xfe000000
	s_mul_i32 s20, s21, 6
	s_mul_i32 s1, s21, 0xe00
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[6:7]
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v28, 16, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[4:5], s[2:3], v[0:1]
	v_cmpx_lt_u32_e32 0xdfc, v33
	s_xor_b32 s22, exec_lo, s0
	s_cbranch_execz .LBB0_52
	v_mov_b32_e32 v6, 0x70
	s_mov_b32 s18, exec_lo
.LBB0_4:
	v_readfirstlane_b32 s12, v4
	v_readfirstlane_b32 s13, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v6
	v_readfirstlane_b32 s15, v7
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[4:5]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[6:7]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v8, v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_4
	s_mov_b32 exec_lo, s18
	s_or_b32 s0, s20, 1
	s_mov_b32 s18, s6
	s_lshl_b32 s25, s0, 2
	s_lshl_b32 s26, s0, 1
	v_dual_mov_b32 v0, s25 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s19, s7
	s_mov_b32 s27, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s26
	buffer_load_u16 v4, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v2, 0x70 :: v_dual_lshlrev_b32 v29, 16, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_6:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v5, v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_6
	s_mov_b32 exec_lo, s27
	s_add_co_i32 s0, s20, 2
	v_lshlrev_b32_e32 v6, 16, v8
	s_lshl_b32 s27, s0, 2
	s_lshl_b32 s28, s0, 1
	v_dual_mov_b32 v0, s27 :: v_dual_mov_b32 v3, 0
	s_wait_loadcnt 0x0
	v_and_b32_e32 v9, 0xffff0000, v5
	v_and_b32_e32 v8, 0xffff0000, v8
	s_mov_b32 s29, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s28 :: v_dual_lshlrev_b32 v7, 16, v5
	v_pk_add_f32 v[8:9], v[8:9], 0 op_sel_hi:[1,0]
	buffer_load_u16 v4, v0, s[16:19], null offen
	v_pk_add_f32 v[6:7], v[6:7], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[28:29]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[6:7], v[6:7], v[28:29]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_dual_add_f32 v2, 0, v6 :: v_dual_add_f32 v6, 0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v2, v7
	v_add_f32_e32 v6, v6, v9
	v_mov_b32_e32 v2, 0x70
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_8:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v8, v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_8
	s_mov_b32 exec_lo, s29
	s_add_co_i32 s0, s20, 3
	s_mov_b32 s31, exec_lo
	s_lshl_b32 s29, s0, 2
	s_lshl_b32 s30, s0, 1
	v_dual_mov_b32 v0, s29 :: v_dual_mov_b32 v3, 0
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s30
	buffer_load_u16 v9, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_10:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v10, v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_10
	s_mov_b32 exec_lo, s31
	s_add_co_i32 s0, s20, 4
	s_wait_loadcnt 0x1
	v_dual_lshlrev_b32 v12, 16, v8 :: v_dual_lshlrev_b32 v9, 16, v9
	s_lshl_b32 s31, s0, 2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v0, s31
	s_lshl_b32 s33, s0, 1
	s_wait_loadcnt 0x0
	v_and_b32_e32 v11, 0xffff0000, v10
	s_mov_b32 s34, exec_lo
	v_lshlrev_b32_e32 v13, 16, v10
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s33
	v_and_b32_e32 v10, 0xffff0000, v8
	v_lshlrev_b32_e32 v8, 16, v4
	v_pk_add_f32 v[12:13], v[12:13], 0 op_sel_hi:[1,0]
	buffer_load_u16 v7, v0, s[16:19], null offen
	v_pk_add_f32 v[10:11], v[10:11], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[8:9]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v5, v12
	v_add_f32_e32 v4, v2, v13
	v_pk_mul_f32 v[8:9], v[10:11], v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v2, 0x70 :: v_dual_add_f32 v5, v6, v8
	v_add_f32_e32 v5, v5, v9
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_12:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v6, v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_12
	s_mov_b32 exec_lo, s34
	s_add_co_i32 s0, s20, 5
	s_mov_b32 s36, exec_lo
	s_lshl_b32 s34, s0, 2
	s_lshl_b32 s35, s0, 1
	v_dual_mov_b32 v0, s34 :: v_dual_mov_b32 v3, 0
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s35
	buffer_load_u16 v8, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_14:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v9, v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_14
	s_mov_b32 exec_lo, s36
	s_wait_loadcnt 0x0
	v_dual_lshlrev_b32 v1, 16, v9 :: v_dual_lshlrev_b32 v0, 16, v6
	v_and_b32_e32 v3, 0xffff0000, v9
	v_and_b32_e32 v2, 0xffff0000, v6
	v_dual_lshlrev_b32 v9, 16, v8 :: v_dual_lshlrev_b32 v8, 16, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	s_mov_b32 s36, exec_lo
	v_pk_add_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[0:1], v[0:1], v[8:9]
	v_pk_mul_f32 v[2:3], v[2:3], v[8:9]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v0, v4, v0
	v_add_lshl_u32 v4, v33, s1, 2
	v_dual_add_f32 v2, v5, v2 :: v_dual_add_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v1, v2, v3 :: v_dual_mov_b32 v3, 0
	v_cvt_pk_bf16_f32 v0, v0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_and_b32_e32 v0, 0xffff, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	v_or_b32_e32 v0, v1, v0
	v_mov_b32_e32 v1, s23
	buffer_store_b32 v0, v4, s[8:11], null offen
	buffer_load_b32 v2, v1, s[4:7], null offen
	s_wait_xcnt 0x1
	v_mov_b32_e32 v0, s24
	buffer_load_u16 v6, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_16:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v7, v31, s[12:15], null offen offset:4
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_16
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s25 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s36, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s26
	buffer_load_u16 v8, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_18:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v9, v31, s[12:15], null offen offset:4
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_18
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s27 :: v_dual_mov_b32 v3, 0
	v_lshlrev_b32_e32 v10, 16, v7
	s_wait_loadcnt 0x0
	v_and_b32_e32 v13, 0xffff0000, v9
	v_and_b32_e32 v12, 0xffff0000, v7
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s28 :: v_dual_lshlrev_b32 v11, 16, v9
	v_dual_lshlrev_b32 v7, 16, v8 :: v_dual_lshlrev_b32 v6, 16, v6
	s_mov_b32 s36, exec_lo
	buffer_load_u16 v5, v0, s[16:19], null offen
	v_pk_add_f32 v[8:9], v[10:11], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[12:13], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[8:9], v[8:9], v[6:7]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_add_f32_e32 v2, 0, v8
	v_pk_mul_f32 v[6:7], v[10:11], v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v8, 0, v6 :: v_dual_add_f32 v6, v2, v9
	v_dual_mov_b32 v2, 0x70 :: v_dual_add_f32 v7, v8, v7
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_20:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v9, v31, s[12:15], null offen offset:4
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_20
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s29 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s36, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s30
	buffer_load_u16 v10, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_22:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v11, v31, s[12:15], null offen offset:4
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_22
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s31 :: v_dual_mov_b32 v3, 0
	v_lshlrev_b32_e32 v12, 16, v9
	s_wait_loadcnt 0x0
	v_and_b32_e32 v15, 0xffff0000, v11
	v_and_b32_e32 v14, 0xffff0000, v9
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s33 :: v_dual_lshlrev_b32 v13, 16, v11
	v_dual_lshlrev_b32 v11, 16, v10 :: v_dual_lshlrev_b32 v10, 16, v5
	v_pk_add_f32 v[14:15], v[14:15], 0 op_sel_hi:[1,0]
	buffer_load_u16 v8, v0, s[16:19], null offen
	v_pk_add_f32 v[12:13], v[12:13], 0 op_sel_hi:[1,0]
	s_mov_b32 s36, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[12:13], v[12:13], v[10:11]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_add_f32_e32 v2, v6, v12
	v_pk_mul_f32 v[10:11], v[14:15], v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v5, v2, v13 :: v_dual_add_f32 v6, v7, v10
	v_mov_b32_e32 v2, 0x70
	v_add_f32_e32 v6, v6, v11
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_24:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v7, v31, s[12:15], null offen offset:4
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_24
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s34 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s36, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s35
	buffer_load_u16 v9, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_26:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v10, v31, s[12:15], null offen offset:4
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_26
	s_mov_b32 exec_lo, s36
	s_wait_loadcnt 0x0
	v_dual_lshlrev_b32 v1, 16, v10 :: v_dual_lshlrev_b32 v0, 16, v7
	v_and_b32_e32 v3, 0xffff0000, v10
	v_and_b32_e32 v2, 0xffff0000, v7
	v_dual_lshlrev_b32 v9, 16, v9 :: v_dual_lshlrev_b32 v8, 16, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	s_mov_b32 s36, exec_lo
	v_pk_add_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[0:1], v[0:1], v[8:9]
	v_pk_mul_f32 v[2:3], v[2:3], v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v0, v5, v0 :: v_dual_add_f32 v2, v6, v2
	v_dual_add_f32 v0, v0, v1 :: v_dual_add_f32 v1, v2, v3
	v_dual_mov_b32 v2, s23 :: v_dual_mov_b32 v3, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v0, v0, s0
	v_cvt_pk_bf16_f32 v1, v1, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_b32_e32 v0, 0xffff, v0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, v1, v0
	v_add_nc_u32_e32 v1, 4, v4
	buffer_store_b32 v0, v1, s[8:11], null offen
	buffer_load_b32 v2, v2, s[4:7], null offen
	s_wait_xcnt 0x1
	v_mov_b32_e32 v0, s24
	buffer_load_u16 v6, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_28:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v7, v31, s[12:15], null offen offset:8
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_28
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s25 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s36, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s26
	buffer_load_u16 v8, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_30:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v9, v31, s[12:15], null offen offset:8
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_30
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s27 :: v_dual_mov_b32 v3, 0
	v_lshlrev_b32_e32 v10, 16, v7
	s_wait_loadcnt 0x0
	v_and_b32_e32 v13, 0xffff0000, v9
	v_and_b32_e32 v12, 0xffff0000, v7
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s28 :: v_dual_lshlrev_b32 v11, 16, v9
	v_dual_lshlrev_b32 v7, 16, v8 :: v_dual_lshlrev_b32 v6, 16, v6
	s_mov_b32 s36, exec_lo
	buffer_load_u16 v5, v0, s[16:19], null offen
	v_pk_add_f32 v[8:9], v[10:11], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[12:13], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[8:9], v[8:9], v[6:7]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_add_f32_e32 v2, 0, v8
	v_pk_mul_f32 v[6:7], v[10:11], v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v8, 0, v6 :: v_dual_add_f32 v6, v2, v9
	v_dual_mov_b32 v2, 0x70 :: v_dual_add_f32 v7, v8, v7
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_32:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v9, v31, s[12:15], null offen offset:8
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_32
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s29 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s36, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s30
	buffer_load_u16 v10, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_34:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v11, v31, s[12:15], null offen offset:8
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_34
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s31 :: v_dual_mov_b32 v3, 0
	v_lshlrev_b32_e32 v12, 16, v9
	s_wait_loadcnt 0x0
	v_and_b32_e32 v15, 0xffff0000, v11
	v_and_b32_e32 v14, 0xffff0000, v9
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s33 :: v_dual_lshlrev_b32 v13, 16, v11
	v_dual_lshlrev_b32 v11, 16, v10 :: v_dual_lshlrev_b32 v10, 16, v5
	v_pk_add_f32 v[14:15], v[14:15], 0 op_sel_hi:[1,0]
	buffer_load_u16 v8, v0, s[16:19], null offen
	v_pk_add_f32 v[12:13], v[12:13], 0 op_sel_hi:[1,0]
	s_mov_b32 s36, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[12:13], v[12:13], v[10:11]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_add_f32_e32 v2, v6, v12
	v_pk_mul_f32 v[10:11], v[14:15], v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v5, v2, v13 :: v_dual_add_f32 v6, v7, v10
	v_mov_b32_e32 v2, 0x70
	v_add_f32_e32 v6, v6, v11
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_36:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v7, v31, s[12:15], null offen offset:8
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_36
	s_mov_b32 exec_lo, s36
	v_dual_mov_b32 v0, s34 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s36, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s35
	buffer_load_u16 v9, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_38:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v10, v31, s[12:15], null offen offset:8
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_38
	s_mov_b32 exec_lo, s36
	s_wait_loadcnt 0x0
	v_dual_lshlrev_b32 v1, 16, v10 :: v_dual_lshlrev_b32 v0, 16, v7
	v_and_b32_e32 v3, 0xffff0000, v10
	v_and_b32_e32 v2, 0xffff0000, v7
	v_dual_lshlrev_b32 v9, 16, v9 :: v_dual_lshlrev_b32 v8, 16, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[0:1], v[0:1], v[8:9]
	v_pk_mul_f32 v[2:3], v[2:3], v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v0, v5, v0 :: v_dual_add_f32 v2, v6, v2
	v_dual_add_f32 v0, v0, v1 :: v_dual_add_f32 v1, v2, v3
	v_mov_b32_e32 v2, s23
	s_mov_b32 s23, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v0, v0, s0
	v_mov_b32_e32 v3, 0
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_and_b32_e32 v0, 0xffff, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 16, v1
	v_or_b32_e32 v0, v1, v0
	v_add_nc_u32_e32 v1, 8, v4
	buffer_store_b32 v0, v1, s[8:11], null offen
	buffer_load_b32 v2, v2, s[4:7], null offen
	s_wait_xcnt 0x1
	v_mov_b32_e32 v0, s24
	buffer_load_u16 v5, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_40:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v6, off, s[12:15], null offset:14332
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_40
	s_mov_b32 exec_lo, s23
	v_dual_mov_b32 v0, s25 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s23, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s26
	buffer_load_u16 v7, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_42:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v8, off, s[12:15], null offset:14332
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_42
	s_mov_b32 exec_lo, s23
	v_dual_mov_b32 v0, s27 :: v_dual_mov_b32 v3, 0
	s_wait_loadcnt 0x1
	v_dual_lshlrev_b32 v10, 16, v6 :: v_dual_lshlrev_b32 v7, 16, v7
	v_and_b32_e32 v12, 0xffff0000, v6
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s28 :: v_dual_lshlrev_b32 v11, 16, v8
	v_and_b32_e32 v13, 0xffff0000, v8
	v_lshlrev_b32_e32 v6, 16, v5
	s_mov_b32 s23, exec_lo
	buffer_load_u16 v4, v0, s[16:19], null offen
	v_pk_add_f32 v[8:9], v[10:11], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[12:13], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[8:9], v[8:9], v[6:7]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_add_f32_e32 v2, 0, v8
	v_pk_mul_f32 v[6:7], v[10:11], v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v5, 0, v6 :: v_dual_add_f32 v6, v2, v9
	v_dual_mov_b32 v2, 0x70 :: v_dual_add_f32 v5, v5, v7
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_44:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v8, off, s[12:15], null offset:14332
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_44
	s_mov_b32 exec_lo, s23
	v_dual_mov_b32 v0, s29 :: v_dual_mov_b32 v3, 0
	s_mov_b32 s23, exec_lo
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s30
	buffer_load_u16 v9, v0, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_46:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v10, off, s[12:15], null offset:14332
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_46
	s_mov_b32 exec_lo, s23
	v_dual_mov_b32 v0, s31 :: v_dual_mov_b32 v3, 0
	s_wait_loadcnt 0x1
	v_dual_lshlrev_b32 v12, 16, v8 :: v_dual_lshlrev_b32 v9, 16, v9
	v_and_b32_e32 v14, 0xffff0000, v8
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_dual_mov_b32 v0, s33 :: v_dual_lshlrev_b32 v13, 16, v10
	v_and_b32_e32 v15, 0xffff0000, v10
	v_lshlrev_b32_e32 v8, 16, v4
	s_mov_b32 s23, exec_lo
	buffer_load_u16 v7, v0, s[16:19], null offen
	v_pk_add_f32 v[10:11], v[12:13], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[14:15], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[10:11], v[10:11], v[8:9]
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_add_f32_e32 v2, v6, v10
	v_pk_mul_f32 v[8:9], v[12:13], v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v4, v5, v8 :: v_dual_add_f32 v5, v2, v11
	v_mov_b32_e32 v2, 0x70
	v_add_f32_e32 v4, v4, v9
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_48:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v6, off, s[12:15], null offset:14332
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_48
	s_mov_b32 exec_lo, s23
	v_dual_mov_b32 v0, s34 :: v_dual_mov_b32 v3, 0
	buffer_load_b32 v2, v0, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, s35
	buffer_load_u16 v8, v0, s[16:19], null offen
	s_wait_xcnt 0x0
	s_mov_b32 s18, exec_lo
	s_wait_loadcnt 0x1
	v_mul_u64_e32 v[0:1], 0x3800, v[2:3]
	v_mov_b32_e32 v2, 0x70
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u64_e32 v[0:1], s[2:3], v[0:1]
.LBB0_50:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v2
	v_readfirstlane_b32 s15, v3
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[2:3]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b32 v9, off, s[12:15], null offset:14332
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_50
	s_mov_b32 exec_lo, s18
	s_wait_loadcnt 0x0
	v_dual_lshlrev_b32 v1, 16, v9 :: v_dual_lshlrev_b32 v0, 16, v6
	v_and_b32_e32 v2, 0xffff0000, v6
	v_and_b32_e32 v3, 0xffff0000, v9
	v_dual_lshlrev_b32 v9, 16, v8 :: v_dual_lshlrev_b32 v8, 16, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[0:1], v[0:1], v[8:9]
	v_pk_mul_f32 v[2:3], v[2:3], v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v0, v5, v0 :: v_dual_add_f32 v2, v4, v2
	v_dual_add_f32 v0, v0, v1 :: v_dual_add_f32 v1, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v0, v0, s0
	v_cvt_pk_bf16_f32 v1, v1, s0
	s_mul_i32 s0, s21, 0x3800
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_and_b32_e32 v0, 0xffff, v0
	s_addk_co_i32 s0, 0x37fc
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v0, v1, v0
	v_mov_b32_e32 v1, s0
	buffer_store_b32 v0, v1, s[8:11], null offen
.LBB0_52:
	s_wait_xcnt 0x0
	s_and_not1_saveexec_b32 s0, s22
	s_cbranch_execz .LBB0_66
	v_dual_mov_b32 v6, 0x70 :: v_dual_mov_b32 v7, 0
	s_mov_b32 s18, exec_lo
.LBB0_54:
	v_readfirstlane_b32 s12, v4
	v_readfirstlane_b32 s13, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s14, v6
	v_readfirstlane_b32 s15, v7
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[4:5]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s0, s[14:15], v[6:7]
	s_and_b32 s0, vcc_lo, s0
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[0:3], v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_54
	s_mov_b32 exec_lo, s18
	s_or_b32 s0, s20, 1
	s_mov_b32 s18, s6
	s_lshl_b32 s12, s0, 2
	s_lshl_b32 s0, s0, 1
	v_dual_mov_b32 v4, s12 :: v_dual_mov_b32 v11, 0
	s_mov_b32 s19, s7
	s_mov_b32 s21, exec_lo
	buffer_load_b32 v10, v4, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v4, s0
	buffer_load_u16 v6, v4, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[4:5], 0x3800, v[10:11]
	v_mov_b32_e32 v10, 0x70
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v30, 16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[8:9], s[2:3], v[4:5]
.LBB0_56:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v8
	v_readfirstlane_b32 s13, v9
	v_readfirstlane_b32 s14, v10
	v_readfirstlane_b32 s15, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[8:9]
	v_cmp_eq_u64_e64 s0, s[14:15], v[10:11]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[4:7], v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_56
	s_mov_b32 exec_lo, s21
	s_add_co_i32 s0, s20, 2
	s_mov_b32 s21, exec_lo
	s_lshl_b32 s12, s0, 2
	s_lshl_b32 s0, s0, 1
	v_dual_mov_b32 v8, s12 :: v_dual_mov_b32 v15, 0
	buffer_load_b32 v14, v8, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v8, s0
	buffer_load_u16 v10, v8, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[8:9], 0x3800, v[14:15]
	v_mov_b32_e32 v14, 0x70
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v32, 16, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[12:13], s[2:3], v[8:9]
.LBB0_58:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v12
	v_readfirstlane_b32 s13, v13
	v_readfirstlane_b32 s14, v14
	v_readfirstlane_b32 s15, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[12:13]
	v_cmp_eq_u64_e64 s0, s[14:15], v[14:15]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[8:11], v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_58
	s_mov_b32 exec_lo, s21
	s_add_co_i32 s0, s20, 3
	s_mov_b32 s21, exec_lo
	s_lshl_b32 s12, s0, 2
	s_lshl_b32 s0, s0, 1
	v_dual_mov_b32 v12, s12 :: v_dual_mov_b32 v19, 0
	buffer_load_b32 v18, v12, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v12, s0
	buffer_load_u16 v14, v12, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[12:13], 0x3800, v[18:19]
	v_mov_b32_e32 v18, 0x70
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v34, 16, v14
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[16:17], s[2:3], v[12:13]
.LBB0_60:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v16
	v_readfirstlane_b32 s13, v17
	v_readfirstlane_b32 s14, v18
	v_readfirstlane_b32 s15, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[16:17]
	v_cmp_eq_u64_e64 s0, s[14:15], v[18:19]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[12:15], v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_60
	s_mov_b32 exec_lo, s21
	s_add_co_i32 s0, s20, 4
	s_mov_b32 s21, exec_lo
	s_lshl_b32 s12, s0, 2
	s_lshl_b32 s0, s0, 1
	v_dual_mov_b32 v16, s12 :: v_dual_mov_b32 v23, 0
	buffer_load_b32 v22, v16, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v16, s0
	buffer_load_u16 v18, v16, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[16:17], 0x3800, v[22:23]
	v_mov_b32_e32 v22, 0x70
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v36, 16, v18
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[20:21], s[2:3], v[16:17]
.LBB0_62:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s12, v20
	v_readfirstlane_b32 s13, v21
	v_readfirstlane_b32 s14, v22
	v_readfirstlane_b32 s15, v23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[12:13], v[20:21]
	v_cmp_eq_u64_e64 s0, s[14:15], v[22:23]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[16:19], v31, s[12:15], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_62
	s_mov_b32 exec_lo, s21
	s_add_co_i32 s20, s20, 5
	v_mov_b32_e32 v27, 0
	s_lshl_b32 s0, s20, 2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mov_b32_e32 v20, s0
	s_lshl_b32 s0, s20, 1
	buffer_load_b32 v26, v20, s[4:7], null offen
	s_wait_xcnt 0x0
	v_mov_b32_e32 v20, s0
	buffer_load_u16 v22, v20, s[16:19], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mul_u64_e32 v[20:21], 0x3800, v[26:27]
	v_mov_b32_e32 v26, 0x70
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v38, 16, v22
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u64_e32 v[24:25], s[2:3], v[20:21]
	s_mov_b32 s2, exec_lo
.LBB0_64:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s4, v24
	v_readfirstlane_b32 s5, v25
	v_readfirstlane_b32 s6, v26
	v_readfirstlane_b32 s7, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[24:25]
	v_cmp_eq_u64_e64 s0, s[6:7], v[26:27]
	s_and_b32 s0, vcc_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[20:23], v31, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_64
	s_mov_b32 exec_lo, s2
	v_dual_lshlrev_b32 v25, 16, v1 :: v_dual_lshlrev_b32 v24, 16, v0
	v_and_b32_e32 v1, 0xffff0000, v1
	v_and_b32_e32 v0, 0xffff0000, v0
	v_dual_lshlrev_b32 v41, 16, v5 :: v_dual_lshlrev_b32 v40, 16, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f32 v[24:25], v[24:25], 0 op_sel_hi:[1,0]
	v_and_b32_e32 v27, 0xffff0000, v5
	v_and_b32_e32 v26, 0xffff0000, v4
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[40:41], v[40:41], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[24:25], v[28:29] op_sel_hi:[1,0]
	v_dual_lshlrev_b32 v25, 16, v9 :: v_dual_lshlrev_b32 v24, 16, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[0:1], v[0:1], v[28:29] op_sel_hi:[1,0]
	v_pk_add_f32 v[26:27], v[26:27], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[4:5], v[4:5], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[30:31] op_sel_hi:[1,0]
	v_and_b32_e32 v9, 0xffff0000, v9
	v_and_b32_e32 v8, 0xffff0000, v8
	v_pk_add_f32 v[24:25], v[24:25], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[4:5], v[4:5], v[40:41]
	v_pk_mul_f32 v[26:27], v[26:27], v[30:31] op_sel_hi:[1,0]
	v_pk_add_f32 v[8:9], v[8:9], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[32:33] op_sel_hi:[1,0]
	v_and_b32_e32 v41, 0xffff0000, v13
	v_dual_lshlrev_b32 v43, 16, v13 :: v_dual_lshlrev_b32 v42, 16, v12
	v_and_b32_e32 v40, 0xffff0000, v12
	v_pk_add_f32 v[0:1], v[0:1], v[26:27]
	v_pk_add_f32 v[4:5], v[4:5], v[24:25]
	v_pk_mul_f32 v[8:9], v[8:9], v[32:33] op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[42:43], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[24:25], v[40:41], 0 op_sel_hi:[1,0]
	v_dual_lshlrev_b32 v41, 16, v17 :: v_dual_lshlrev_b32 v40, 16, v16
	v_and_b32_e32 v27, 0xffff0000, v17
	v_and_b32_e32 v26, 0xffff0000, v16
	v_pk_add_f32 v[0:1], v[0:1], v[8:9]
	v_pk_mul_f32 v[8:9], v[12:13], v[34:35] op_sel_hi:[1,0]
	v_pk_add_f32 v[16:17], v[40:41], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[24:25], v[34:35] op_sel_hi:[1,0]
	v_pk_add_f32 v[24:25], v[26:27], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[4:5], v[4:5], v[8:9]
	v_pk_mul_f32 v[8:9], v[16:17], v[36:37] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[0:1], v[0:1], v[12:13]
	v_pk_mul_f32 v[12:13], v[24:25], v[36:37] op_sel_hi:[1,0]
	v_dual_lshlrev_b32 v25, 16, v7 :: v_dual_lshlrev_b32 v24, 16, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[4:5], v[4:5], v[8:9]
	v_dual_lshlrev_b32 v9, 16, v3 :: v_dual_lshlrev_b32 v8, 16, v2
	v_pk_add_f32 v[0:1], v[0:1], v[12:13]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v13, 0xffff0000, v21
	v_dual_lshlrev_b32 v17, 16, v21 :: v_dual_lshlrev_b32 v16, 16, v20
	v_pk_add_f32 v[8:9], v[8:9], 0 op_sel_hi:[1,0]
	v_and_b32_e32 v3, 0xffff0000, v3
	v_and_b32_e32 v2, 0xffff0000, v2
	v_and_b32_e32 v12, 0xffff0000, v20
	v_pk_add_f32 v[20:21], v[24:25], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[28:29] op_sel_hi:[1,0]
	v_and_b32_e32 v7, 0xffff0000, v7
	v_pk_add_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	v_and_b32_e32 v6, 0xffff0000, v6
	v_pk_mul_f32 v[20:21], v[20:21], v[30:31] op_sel_hi:[1,0]
	v_pk_add_f32 v[8:9], v[8:9], 0 op_sel_hi:[1,0]
	v_dual_lshlrev_b32 v25, 16, v11 :: v_dual_lshlrev_b32 v24, 16, v10
	v_pk_mul_f32 v[2:3], v[2:3], v[28:29] op_sel_hi:[1,0]
	v_pk_add_f32 v[6:7], v[6:7], 0 op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_add_f32 v[8:9], v[8:9], v[20:21]
	v_and_b32_e32 v11, 0xffff0000, v11
	v_and_b32_e32 v10, 0xffff0000, v10
	v_pk_add_f32 v[20:21], v[24:25], 0 op_sel_hi:[1,0]
	v_dual_lshlrev_b32 v25, 16, v15 :: v_dual_lshlrev_b32 v24, 16, v14
	v_pk_add_f32 v[2:3], v[2:3], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[30:31] op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[10:11], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[32:33] op_sel_hi:[1,0]
	v_and_b32_e32 v15, 0xffff0000, v15
	v_and_b32_e32 v14, 0xffff0000, v14
	v_pk_add_f32 v[24:25], v[24:25], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3], v[2:3], v[6:7]
	v_pk_add_f32 v[6:7], v[8:9], v[20:21]
	v_pk_mul_f32 v[8:9], v[10:11], v[32:33] op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[14:15], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[24:25], v[34:35] op_sel_hi:[1,0]
	v_and_b32_e32 v21, 0xffff0000, v19
	v_dual_lshlrev_b32 v25, 16, v19 :: v_dual_lshlrev_b32 v24, 16, v18
	v_and_b32_e32 v20, 0xffff0000, v18
	v_pk_add_f32 v[2:3], v[2:3], v[8:9]
	v_pk_add_f32 v[6:7], v[6:7], v[14:15]
	v_pk_mul_f32 v[8:9], v[10:11], v[34:35] op_sel_hi:[1,0]
	v_pk_add_f32 v[10:11], v[24:25], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[14:15], v[20:21], 0 op_sel_hi:[1,0]
	v_and_b32_e32 v19, 0xffff0000, v23
	v_dual_lshlrev_b32 v21, 16, v23 :: v_dual_lshlrev_b32 v20, 16, v22
	v_and_b32_e32 v18, 0xffff0000, v22
	v_pk_add_f32 v[16:17], v[16:17], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3], v[2:3], v[8:9]
	v_pk_mul_f32 v[8:9], v[10:11], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[14:15], v[36:37] op_sel_hi:[1,0]
	v_pk_add_f32 v[14:15], v[20:21], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[18:19], v[18:19], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[12:13], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[6:7], v[6:7], v[8:9]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	v_pk_mul_f32 v[8:9], v[14:15], v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[18:19], v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[16:17], v[38:39] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[38:39] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[6:7], v[6:7], v[8:9]
	v_pk_add_f32 v[2:3], v[2:3], v[10:11]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[4:5], v[4:5], v[14:15]
	v_pk_add_f32 v[0:1], v[0:1], v[12:13]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_pk_bf16_f32 v6, v6, v7
	v_cvt_pk_bf16_f32 v2, v2, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_pk_bf16_f32 v4, v4, v5
	v_cvt_pk_bf16_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_perm_b32 v3, v6, v2, 0x3020706
	v_perm_b32 v2, v6, v2, 0x1000504
	v_perm_b32 v1, v4, v0, 0x3020706
	v_perm_b32 v0, v4, v0, 0x1000504
	v_add_lshl_u32 v4, v33, s1, 2
	buffer_store_b128 v[0:3], v4, s[8:11], null offen
.LBB0_66:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 12
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 44
		.amdhsa_next_free_sgpr 37
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 60
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
.Lfunc_end0:
	.size	moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16, .Lfunc_end0-moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16

	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.num_vgpr, 44
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.num_agpr, 0
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.numbered_sgpr, 37
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.num_named_barrier, 0
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.private_seg_size, 0
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.uses_vcc, 1
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.uses_flat_scratch, 0
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.has_dyn_sized_stack, 0
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.has_recursion, 0
	.set moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.has_indirect_call, 0
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 256
    .name:           moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     44
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
