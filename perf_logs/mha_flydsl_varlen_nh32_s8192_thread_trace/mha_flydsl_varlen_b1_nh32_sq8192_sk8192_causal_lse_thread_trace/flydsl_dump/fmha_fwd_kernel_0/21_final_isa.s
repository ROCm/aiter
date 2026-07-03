	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	fmha_fwd_kernel_0
	.p2align	8
	.type	fmha_fwd_kernel_0,@function
fmha_fwd_kernel_0:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0x1001
	s_clause 0x1
	s_load_b64 s[24:25], s[0:1], 0x0 nv
	s_load_b64 s[2:3], s[0:1], 0xa0 nv
	s_wait_kmcnt 0x0
	v_writelane_b32 v0 /*v256*/, s2, 0
	v_writelane_b32 v0 /*v256*/, s3, 1
	s_clause 0x3
	s_load_b64 s[2:3], s[0:1], 0xb8 nv
	s_load_b64 s[4:5], s[0:1], 0xc8 nv
	s_load_b64 s[8:9], s[0:1], 0xf4 nv
	s_load_b256 s[16:23], s[0:1], 0xd4 nv
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_load_b96 s[28:30], s[0:1], 0x108 nv
	s_bfe_u32 s6, ttmp6, 0x4000c
	s_bfe_u32 s10, ttmp6, 0x40014
	s_add_co_i32 s6, s6, 1
	s_and_b32 s7, ttmp6, 15
	s_mul_i32 s6, ttmp9, s6
	s_lshr_b32 s11, ttmp7, 16
	s_add_co_i32 s10, s10, 1
	s_bfe_u32 s12, ttmp6, 0x40010
	s_add_co_i32 s7, s7, s6
	s_mul_i32 s6, s11, s10
	s_bfe_u32 s10, ttmp6, 0x40008
	s_and_b32 s13, ttmp7, 0xffff
	s_add_co_i32 s12, s12, 1
	s_add_co_i32 s10, s10, s6
	s_mul_i32 s6, s13, s12
	s_bfe_u32 s12, ttmp6, 0x40004
	s_getreg_b32 s14, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s12, s12, s6
	s_cmp_eq_u32 s14, 0
	s_nop 0
	s_cselect_b32 s6, s13, s12
	s_wait_kmcnt 0x0
	s_mul_i32 s12, s29, s28
	s_mul_i32 s6, s28, s6
	s_cselect_b32 s7, ttmp9, s7
	s_cselect_b32 s10, s11, s10
	s_add_co_i32 s6, s6, s7
	s_mul_i32 s7, s12, s10
	s_cvt_f32_u32 s10, s28
	s_mul_i32 s12, s12, s30
	s_add_co_i32 s6, s6, s7
	s_lshr_b32 s7, s12, 3
	s_set_vgpr_msb 0x4000
	v_rcp_iflag_f32_e32 v1, s10
	s_and_b32 s10, s12, 7
	s_cmp_gt_u32 s12, 8
	s_cselect_b32 s11, -1, 0
	s_cmp_eq_u32 s10, 0
	s_cselect_b32 s10, -1, 0
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_readfirstlane_b32 s12, v1
	s_and_b32 s13, s6, 7
	s_and_b32 s10, s11, s10
	s_mul_i32 s7, s13, s7
	s_lshr_b32 s13, s6, 3
	s_mul_f32 s12, s12, 0x4f7ffffe
	s_add_co_i32 s7, s7, s13
	s_and_b32 s10, s10, exec_lo
	s_cselect_b32 s6, s7, s6
	s_cvt_u32_f32 s10, s12
	s_sub_co_i32 s7, 0, s28
	s_cvt_f32_u32 s11, s29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s7, s7, s10
	s_mul_hi_u32 s7, s10, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_rcp_iflag_f32_e32 v1, s11
	s_add_co_i32 s10, s10, s7
	s_mul_hi_u32 s7, s6, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_1)
	s_mul_i32 s10, s7, s28
	s_add_co_i32 s11, s7, 1
	s_sub_co_i32 s10, s6, s10
	v_nop
	v_readfirstlane_b32 s12, v1
	s_sub_co_i32 s13, s10, s28
	s_cmp_ge_u32 s10, s28
	s_cselect_b32 s7, s11, s7
	s_mul_f32 s11, s12, 0x4f7ffffe
	s_cselect_b32 s10, s13, s10
	s_add_co_i32 s12, s7, 1
	s_cmp_ge_u32 s10, s28
	s_cvt_u32_f32 s10, s11
	s_cselect_b32 s13, s12, s7
	s_sub_co_i32 s7, 0, s29
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s7, s7, s10
	s_mul_hi_u32 s7, s10, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s10, s10, s7
	s_mul_hi_u32 s7, s13, s10
	s_mul_i32 s10, s13, s28
	s_mul_i32 s11, s7, s29
	s_sub_co_i32 s6, s6, s10
	s_sub_co_i32 s10, s13, s11
	s_add_co_i32 s11, s7, 1
	s_sub_co_i32 s12, s10, s29
	s_cmp_ge_u32 s10, s29
	s_cselect_b32 s7, s11, s7
	s_cselect_b32 s10, s12, s10
	s_add_co_i32 s11, s7, 1
	s_cmp_ge_u32 s10, s29
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s28, 2
	s_cselect_b32 s14, s11, s7
	s_lshl2_add_u32 s7, s6, 4
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v1, s6 :: v_dual_mov_b32 v2, s7
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v3, v1, s[2:3] scale_offset
	global_load_b32 v4, v2, s[2:3]
	global_load_b32 v5, v1, s[4:5] scale_offset
	global_load_b32 v6, v2, s[4:5]
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s29, 3
	s_wait_xcnt 0x2
	s_mul_i32 s2, s14, s29
	s_wait_xcnt 0x0
	s_mov_b32 s4, 0
	s_sub_co_i32 s102, s13, s2
	v_writelane_b32 v0 /*v256*/, s30, 4
	s_lshl_b32 s63, s102, 7
	v_writelane_b32 v0 /*v256*/, s13, 5
	v_writelane_b32 v0 /*v256*/, s2, 6
	s_wait_loadcnt 0x3
	v_readfirstlane_b32 s12, v3
	s_wait_loadcnt 0x2
	v_readfirstlane_b32 s15, v4
	s_wait_loadcnt 0x1
	v_readfirstlane_b32 s33, v5
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s10, v6
	s_sub_co_i32 s26, s15, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s2, s26, s63
	s_cmp_gt_i32 s26, s63
	v_cmp_gt_i32_e32 vcc_lo, s2, v0
	v_writelane_b32 v0 /*v256*/, s2, 7
	s_cselect_b32 s2, -1, 0
	s_cmp_eq_u32 s10, s33
	s_cselect_b32 s3, -1, 0
	s_and_b32 s5, vcc_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s5, s3
	s_mul_i32 s3, s14, s8
	v_writelane_b32 v0 /*v256*/, s3, 8
	s_and_saveexec_b32 s3, s5
	s_set_vgpr_msb 0x4000
	s_cbranch_execz .LBB0_2
	s_wait_alu depctr_vm_vsrc(1)
	v_or_b32_e32 v1, s63, v0
	s_mov_b32 s5, s4
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b64_e32 v[2:3], s[4:5]
	v_add_nc_u32_e32 v1, s12, v1
	v_mov_b64_e32 v[4:5], s[6:7]
	s_set_vgpr_msb 1
	v_readlane_b32 s4, v0 /*v256*/, 8
	v_mov_b32_e32 v7, 0xff800000
	v_mul_lo_u32 v6, s20, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v6, s4, v6, 1
	s_load_b96 s[4:6], s[0:1], 0x108 nv
	s_wait_kmcnt 0x0
	s_load_b64 s[4:5], s[0:1], 0xa0 nv
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x100
	s_clause 0xf
	global_store_b128 v6, v[2:5], s[24:25]
	global_store_b128 v6, v[2:5], s[24:25] offset:16
	global_store_b128 v6, v[2:5], s[24:25] offset:32
	global_store_b128 v6, v[2:5], s[24:25] offset:48
	global_store_b128 v6, v[2:5], s[24:25] offset:64
	global_store_b128 v6, v[2:5], s[24:25] offset:80
	global_store_b128 v6, v[2:5], s[24:25] offset:96
	global_store_b128 v6, v[2:5], s[24:25] offset:112
	global_store_b128 v6, v[2:5], s[24:25] offset:128
	global_store_b128 v6, v[2:5], s[24:25] offset:144
	global_store_b128 v6, v[2:5], s[24:25] offset:160
	global_store_b128 v6, v[2:5], s[24:25] offset:176
	global_store_b128 v6, v[2:5], s[24:25] offset:192
	global_store_b128 v6, v[2:5], s[24:25] offset:208
	global_store_b128 v6, v[2:5], s[24:25] offset:224
	global_store_b128 v6, v[2:5], s[24:25] offset:240
	v_mad_u32 v1, v1, s6, s14
	s_wait_kmcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	global_store_b32 v1, v7, s[4:5] scale_offset
.LBB0_2:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s3
	s_sub_co_i32 s98, s10, s33
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s10, 9
	s_cmp_lt_i32 s98, 1
	s_cselect_b32 s3, -1, 0
	s_xor_b32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x4000
	s_cbranch_vccnz .LBB0_25
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s24, 10
	s_set_vgpr_msb 0x4080
	v_dual_lshrrev_b32 v75 /*v587*/, 5, v0 :: v_dual_bitop2_b32 v73 /*v585*/, 16, v0 bitop3:0x40
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8000
	v_and_b32_e32 v1, 0x6f, v0
	s_load_b64 s[4:5], s[0:1], 0x28 nv
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s25, 11
	s_clause 0x1
	s_load_b64 s[10:11], s[0:1], 0x50 nv
	s_load_b64 s[2:3], s[0:1], 0x78 nv
	s_wait_xcnt 0x0
	s_add_co_i32 s1, s12, s63
	s_set_vgpr_msb 0x4020
	v_mad_u32 v1, s17, v1, v73 /*v585*/
	s_mul_i32 s6, s14, s21
	s_set_vgpr_msb 0x2040
	v_writelane_b32 v0 /*v256*/, s12, 12
	s_mul_i32 s7, s1, s17
	s_mov_b32 s47, 0
	s_mul_i32 s0, s15, s17
	s_mov_b32 s12, s47
	v_writelane_b32 v0 /*v256*/, s15, 13
	s_set_vgpr_msb 0x4000
	v_and_b32_e32 v1, -4, v1
	s_lshl_b32 s13, s0, 25
	s_set_vgpr_msb 8
	v_lshlrev_b32_e32 v252, 3, v75 /*v587*/
	s_mov_b32 s99, 0xc80000
	s_set_vgpr_msb 0x840
	v_writelane_b32 v0 /*v256*/, s1, 14
	s_set_vgpr_msb 0x4000
	v_add3_u32 v18, s7, s6, v1
	s_ashr_i32 s1, s0, 31
	v_sub_nc_u32_e32 v1, s98, v252
	s_lshr_b64 s[6:7], s[0:1], 7
	s_wait_kmcnt 0x0
	s_or_b64 s[4:5], s[4:5], s[12:13]
	s_wait_alu depctr_va_vdst(1)
	s_clause 0x5
	buffer_load_b128 v[66:69], v18, s[4:7], null offen
	buffer_load_b128 v[70:73], v18, s[4:7], null offen offset:32
	buffer_load_b128 v[50:53], v18, s[4:7], null offen offset:64
	buffer_load_b128 v[54:57], v18, s[4:7], null offen offset:96
	buffer_load_b128 v[34:37], v18, s[4:7], null offen offset:128
	buffer_load_b128 v[38:41], v18, s[4:7], null offen offset:160
	v_subrev_nc_u32_e32 v3, 32, v1
	s_mov_b32 s28, 8
	s_mov_b32 s97, 0xc00000
	s_mov_b32 s46, s47
	s_mov_b32 s43, s99
	v_med3_i32 v3, v3, 0, 8
	s_mov_b32 s40, 0x10000
	s_mov_b32 s41, s97
	s_mov_b32 s44, s28
	v_mov_b64_e32 v[112:113], s[46:47]
	v_lshlrev_b32_e32 v3, 16, v3
	v_med3_i32 v2, v1, 0, 8
	v_subrev_nc_u32_e32 v4, 64, v1
	v_add_nc_u32_e32 v5, 0xffffffa0, v1
	v_mov_b64_e32 v[128:129], s[46:47]
	v_mov_b64_e32 v[108:109], s[42:43]
	v_lshlrev_b32_e32 v2, 16, v2
	v_mov_b64_e32 v[124:125], s[42:43]
	v_med3_i32 v4, v4, 0, 8
	v_mov_b32_e32 v124, v3
	v_med3_i32 v5, v5, 0, 8
	v_mov_b32_e32 v108, v2
	v_readfirstlane_b32 s0, v252
	s_ashr_i32 s85, s18, 1
	v_mov_b64_e32 v[110:111], s[44:45]
	v_mov_b64_e32 v[126:127], s[44:45]
	v_readfirstlane_b32 vcc_hi, v1
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s0, 15
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v111, s85 :: v_dual_lshlrev_b32 v4, 16, v4
	v_dual_mov_b32 v127, s85 :: v_dual_lshlrev_b32 v5, 16, v5
	s_mov_b32 s25, 0x800000
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v149 /*v661*/, 0x60, v0
	v_and_b32_e32 v148 /*v660*/, 15, v0
	s_mov_b32 s30, s47
	s_mov_b32 s31, s47
	s_mov_b32 s27, s25
	s_ashr_i32 s93, s19, 1
	s_set_vgpr_msb 0x8000
	v_mov_b64_e32 v[136:137], s[46:47]
	v_mov_b64_e32 v[152:153], s[46:47]
	s_mov_b32 s24, 0xf510000
	v_mov_b64_e32 v[176:177], s[30:31]
	v_mov_b64_e32 v[144:145], s[30:31]
	v_mov_b64_e32 v[120:121], s[30:31]
	v_mov_b64_e32 v[104:105], s[30:31]
	s_mov_b32 s0, 1
	v_mov_b64_e32 v[134:135], s[44:45]
	v_mov_b64_e32 v[132:133], s[42:43]
	v_mov_b64_e32 v[150:151], s[44:45]
	v_mov_b64_e32 v[148:149], s[42:43]
	v_mov_b64_e32 v[174:175], s[28:29]
	v_mov_b64_e32 v[172:173], s[26:27]
	v_mov_b64_e32 v[142:143], s[28:29]
	v_mov_b64_e32 v[140:141], s[26:27]
	v_mov_b64_e32 v[118:119], s[28:29]
	v_mov_b64_e32 v[116:117], s[26:27]
	v_mov_b64_e32 v[102:103], s[28:29]
	v_mov_b64_e32 v[100:101], s[26:27]
	v_mov_b64_e32 v[106:107], s[40:41]
	v_mov_b64_e32 v[122:123], s[40:41]
	v_mov_b64_e32 v[130:131], s[40:41]
	v_dual_mov_b32 v132, v4 :: v_dual_mov_b32 v135, s85
	v_mov_b64_e32 v[146:147], s[40:41]
	v_dual_mov_b32 v148, v5 :: v_dual_mov_b32 v151, s85
	v_mov_b64_e32 v[170:171], s[24:25]
	v_dual_mov_b32 v172, v2 :: v_dual_mov_b32 v175, s93
	v_mov_b64_e32 v[138:139], s[24:25]
	v_dual_mov_b32 v140, v3 :: v_dual_mov_b32 v143, s93
	v_mov_b64_e32 v[114:115], s[24:25]
	v_dual_mov_b32 v116, v4 :: v_dual_mov_b32 v119, s93
	v_mov_b64_e32 v[98:99], s[24:25]
	v_dual_mov_b32 v100, v5 :: v_dual_mov_b32 v103, s93
	s_clause 0x3
	buffer_load_b128 v[42:45], v18, s[4:7], null offen offset:192
	buffer_load_b128 v[46:49], v18, s[4:7], null offen offset:224
	buffer_load_b128 v[10:13], v18, s[4:7], null offen offset:256
	buffer_load_b128 v[14:17], v18, s[4:7], null offen offset:288
	s_wait_alu depctr_va_vdst(14)
	s_clause 0x1
	buffer_load_b128 v[2:5], v18, s[4:7], null offen offset:320
	buffer_load_b128 v[6:9], v18, s[4:7], null offen offset:352
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_add_u32 v18, s17, 4, v18
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x5
	buffer_load_b128 v[90:93], v18, s[4:7], null offen
	buffer_load_b128 v[94:97], v18, s[4:7], null offen offset:32
	buffer_load_b128 v[82:85], v18, s[4:7], null offen offset:64
	buffer_load_b128 v[86:89], v18, s[4:7], null offen offset:96
	buffer_load_b128 v[74:77], v18, s[4:7], null offen offset:128
	buffer_load_b128 v[78:81], v18, s[4:7], null offen offset:160
	v_add_nc_u32_e32 v154, 0xc0, v18
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x3
	buffer_load_b128 v[58:61], v154, s[4:7], null offen
	buffer_load_b128 v[62:65], v154, s[4:7], null offen offset:32
	buffer_load_b128 v[26:29], v154, s[4:7], null offen offset:64
	buffer_load_b128 v[30:33], v154, s[4:7], null offen offset:96
	s_wait_alu depctr_vm_vsrc(4)
	s_clause 0x1
	buffer_load_b128 v[18:21], v154, s[4:7], null offen offset:128
	buffer_load_b128 v[22:25], v154, s[4:7], null offen offset:160
	s_cvt_f32_u32 s1, s9
	s_wait_xcnt 0x0
	s_sub_co_i32 s4, 0, s9
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s14, 16
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_rcp_iflag_f32_e32 v154, s1
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s1, v154
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_cvt_u32_f32 s1, s1
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s4, s4, s1
	s_mul_hi_u32 s4, s1, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s4
	s_mul_hi_u32 s1, s14, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_mul_i32 s4, s1, s9
	s_add_co_i32 s5, s1, 1
	s_sub_co_i32 s4, s14, s4
	s_sub_co_i32 s6, s4, s9
	s_cmp_ge_u32 s4, s9
	s_cselect_b32 s1, s5, s1
	s_cselect_b32 s4, s6, s4
	s_add_co_i32 s5, s1, 1
	s_cmp_ge_u32 s4, s9
	s_mul_i32 s4, s33, s18
	s_cselect_b32 s1, s5, s1
	s_mul_i32 s5, s33, s19
	s_mul_i32 s6, s1, s22
	s_mul_i32 s1, s1, s23
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s6, 17
	s_add_co_i32 s22, s1, s5
	s_add_co_i32 s4, s6, s4
	v_writelane_b32 v0 /*v256*/, s1, 18
	s_set_vgpr_msb 0x4028
	v_mad_u32_u24 v255, 0x190, v148 /*v660*/, v73 /*v585*/
	s_set_vgpr_msb 0x2840
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v11 /*v267*/, 0x10000, v255
	v_add_nc_u32_e32 v10 /*v266*/, 0x11900, v255
	v_add_nc_u32_e32 v26 /*v282*/, 0x100c0, v255
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v253, 0x119c0, v255
	s_mul_f32 s6, s16, 0x3fb8aa3b
	s_lshl_b32 s66, s19, 5
	s_lshl_b32 s100, s18, 5
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s6, 19
	v_writelane_b32 v0 /*v256*/, s7, 20
	s_set_vgpr_msb 0x4000
	v_mul_lo_u32 v154, v252, s18
	s_set_vgpr_msb 0x48
	v_mul_u32_u24_e32 v12 /*v268*/, 0xc80, v75 /*v587*/
	s_ashr_i32 s5, s4, 31
	v_writelane_b32 v0 /*v256*/, s4, 21
	s_ashr_i32 s101, s100, 31
	v_readfirstlane_b32 s7, v109
	v_readfirstlane_b32 s8, v110
	v_readfirstlane_b32 s9, v111
	s_set_vgpr_msb 0x4800
	v_ashrrev_i32_e32 v155, 31, v154
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s5, 22
	v_readfirstlane_b32 s6, v108
	s_set_vgpr_msb 0x4001
	v_readfirstlane_b32 s104, v12 /*v268*/
	v_add_nc_u64_e32 v[250:251], s[10:11], v[154:155]
	v_mov_b64_e32 v[156:157], s[2:3]
	v_mov_b64_e32 v[154:155], s[0:1]
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s10, v112
	v_readfirstlane_b32 s11, v113
	s_movk_i32 s1, 0x3200
	v_add_nc_u64_e32 v[156:157], s[4:5], v[250:251]
	v_readfirstlane_b32 s4, v106
	v_readfirstlane_b32 s5, v107
	s_set_vgpr_msb 1
	v_mov_b32_e32 v107, v12 /*v268*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s12, v154
	v_or_b32_e32 v109, 0x80000000, v157
	v_add_nc_u64_e32 v[110:111], s[100:101], v[156:157]
	v_mov_b32_e32 v106, v156
	v_readfirstlane_b32 s13, v107
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s15, v109
	v_readfirstlane_b32 s14, v106
	v_mov_b64_e32 v[108:109], s[2:3]
	v_mov_b64_e32 v[106:107], s[0:1]
	s_set_vgpr_msb 8
	v_mad_u32_u24 v107, 0xc80, v75 /*v587*/, s1
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v109, 0x80000000, v111
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_mov_b32_e32 v108, v110
	v_readfirstlane_b32 s4, v122
	v_readfirstlane_b32 s5, v123
	v_readfirstlane_b32 s6, v124
	v_readfirstlane_b32 s7, v125
	v_readfirstlane_b32 s8, v126
	v_readfirstlane_b32 s9, v127
	v_readfirstlane_b32 s10, v128
	v_readfirstlane_b32 s11, v129
	v_readfirstlane_b32 s12, v106
	v_readfirstlane_b32 s13, v107
	v_readfirstlane_b32 s14, v108
	v_readfirstlane_b32 s15, v109
	v_add_nc_u64_e32 v[110:111], s[100:101], v[110:111]
	s_movk_i32 s1, 0x6400
	s_barrier_wait -1
	v_mov_b64_e32 v[108:109], s[2:3]
	v_mov_b64_e32 v[106:107], s[0:1]
	s_set_vgpr_msb 8
	v_mad_u32_u24 v107, 0xc80, v75 /*v587*/, s1
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v109, 0x80000000, v111
	v_mov_b32_e32 v108, v110
	s_mov_b32 s1, 0x9600
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_readfirstlane_b32 s4, v130
	v_readfirstlane_b32 s5, v131
	v_readfirstlane_b32 s6, v132
	v_readfirstlane_b32 s7, v133
	v_readfirstlane_b32 s8, v134
	v_readfirstlane_b32 s9, v135
	v_readfirstlane_b32 s10, v136
	v_readfirstlane_b32 s11, v137
	v_readfirstlane_b32 s12, v106
	v_readfirstlane_b32 s13, v107
	v_readfirstlane_b32 s14, v108
	v_readfirstlane_b32 s15, v109
	v_mov_b64_e32 v[108:109], s[2:3]
	v_add_nc_u64_e32 v[108:109], s[100:101], v[110:111]
	v_mov_b64_e32 v[106:107], s[0:1]
	s_set_vgpr_msb 8
	s_barrier_wait -1
	v_mad_u32_u24 v107, 0xc80, v75 /*v587*/, s1
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v109, 0x80000000, v109
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_readfirstlane_b32 s4, v146
	v_readfirstlane_b32 s5, v147
	v_readfirstlane_b32 s6, v148
	v_readfirstlane_b32 s7, v149
	v_readfirstlane_b32 s8, v150
	v_readfirstlane_b32 s9, v151
	v_readfirstlane_b32 s10, v152
	v_readfirstlane_b32 s11, v153
	v_readfirstlane_b32 s12, v106
	v_readfirstlane_b32 s13, v107
	v_readfirstlane_b32 s14, v108
	v_readfirstlane_b32 s15, v109
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_readfirstlane_b32 s4, v250
	v_readfirstlane_b32 s5, v251
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s4, 23
	v_writelane_b32 v0 /*v256*/, s5, 24
	s_barrier_wait -1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[122:125], v255
	ds_load_b128 v[126:129], v255 offset:32
	ds_load_b128 v[146:149], v255 offset:64
	ds_load_b128 v[150:153], v255 offset:96
	ds_load_b128 v[154:157], v255 offset:128
	ds_load_b128 v[158:161], v255 offset:160
	ds_load_b128 v[162:165], v255 offset:6400
	ds_load_b128 v[166:169], v255 offset:6432
	ds_load_b128 v[178:181], v255 offset:6464
	ds_load_b128 v[182:185], v255 offset:6496
	ds_load_b128 v[186:189], v255 offset:6528
	ds_load_b128 v[190:193], v255 offset:6560
	ds_load_b128 v[194:197], v255 offset:192
	ds_load_b128 v[198:201], v255 offset:224
	ds_load_b128 v[202:205], v255 offset:256
	ds_load_b128 v[206:209], v255 offset:288
	ds_load_b128 v[210:213], v255 offset:320
	ds_load_b128 v[214:217], v255 offset:352
	ds_load_b128 v[218:221], v255 offset:6592
	ds_load_b128 v[222:225], v255 offset:6624
	ds_load_b128 v[226:229], v255 offset:6656
	ds_load_b128 v[230:233], v255 offset:6688
	ds_load_b128 v[234:237], v255 offset:6720
	ds_load_b128 v[238:241], v255 offset:6752
	s_wait_loadcnt_dscnt 0x1600
	v_wmma_f32_16x16x32_bf16 v[130:137], v[122:129], v[66:73], 0
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x32_bf16 v[106:113], v[122:129], v[90:97], 0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[146:153], v[50:57], v[130:137]
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[106:113], v[146:153], v[82:89], v[106:113]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[154:161], v[34:41], v[130:137]
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[106:113], v[154:161], v[74:81], v[106:113]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[194:201], v[42:49], v[130:137]
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[106:113], v[194:201], v[58:65], v[106:113]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[202:209], v[10:17], v[130:137]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[106:113], v[202:209], v[26:33], v[106:113]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[130:137], v[210:217], v[2:9], v[130:137]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[210:217], v[18:25], v[106:113]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[162:169], v[66:73], 0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[162:169], v[90:97], 0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[122:129], v[178:185], v[50:57], v[122:129]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[178:185], v[82:89], v[154:161]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[122:129], v[186:193], v[34:41], v[122:129]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[186:193], v[74:81], v[154:161]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[122:129], v[218:225], v[42:49], v[122:129]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[218:225], v[58:65], v[154:161]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[122:129], v[226:233], v[10:17], v[122:129]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[226:233], v[26:33], v[154:161]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[122:129], v[234:241], v[2:9], v[122:129]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[234:241], v[18:25], v[154:161]
	ds_load_b128 v[162:165], v255 offset:12800
	ds_load_b128 v[166:169], v255 offset:12832
	ds_load_b128 v[178:181], v255 offset:12864
	ds_load_b128 v[182:185], v255 offset:12896
	ds_load_b128 v[194:197], v255 offset:12928
	ds_load_b128 v[198:201], v255 offset:12960
	ds_load_b128 v[210:213], v255 offset:19200
	ds_load_b128 v[214:217], v255 offset:19232
	ds_load_b128 v[218:221], v255 offset:19264
	ds_load_b128 v[222:225], v255 offset:19296
	ds_load_b128 v[226:229], v255 offset:19328
	ds_load_b128 v[230:233], v255 offset:19360
	ds_load_b128 v[202:205], v255 offset:12992
	ds_load_b128 v[206:209], v255 offset:13024
	ds_load_b128 v[234:237], v255 offset:13056
	ds_load_b128 v[238:241], v255 offset:13088
	ds_load_b128 v[242:245], v255 offset:13120
	ds_load_b128 v[246:249], v255 offset:13152
	s_set_vgpr_msb 64
	ds_load_b128 v[2:5] /*v[258:261]*/, v255 offset:19392
	ds_load_b128 v[6:9] /*v[262:265]*/, v255 offset:19424
	ds_load_b128 v[14:17] /*v[270:273]*/, v255 offset:19456
	ds_load_b128 v[18:21] /*v[274:277]*/, v255 offset:19488
	ds_load_b128 v[28:31] /*v[284:287]*/, v255 offset:19520
	ds_load_b128 v[32:35] /*v[288:291]*/, v255 offset:19552
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[186:193], v[162:169], v[66:73], 0
	v_wmma_f32_16x16x32_bf16 v[146:153], v[162:169], v[90:97], 0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[178:185], v[50:57], v[186:193]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[178:185], v[82:89], v[146:153]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[194:201], v[34:41], v[186:193]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[194:201], v[74:81], v[146:153]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[202:209], v[42:49], v[186:193]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[202:209], v[58:65], v[146:153]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[234:241], v[10:17], v[186:193]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[234:241], v[26:33], v[146:153]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[242:249], v[2:9], v[186:193]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[242:249], v[18:25], v[146:153]
	v_wmma_f32_16x16x32_bf16 v[202:209], v[210:217], v[66:73], 0
	v_wmma_f32_16x16x32_bf16 v[162:169], v[210:217], v[90:97], 0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[218:225], v[50:57], v[202:209]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[218:225], v[82:89], v[162:169]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[226:233], v[34:41], v[202:209]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[226:233], v[74:81], v[162:169]
	s_set_vgpr_msb 1
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[2:9] /*v[258:265]*/, v[42:49], v[202:209]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[2:9] /*v[258:265]*/, v[58:65], v[162:169]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[14:21] /*v[270:277]*/, v[10:17], v[202:209]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[14:21] /*v[270:277]*/, v[26:33], v[162:169]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[28:35] /*v[284:291]*/, v[2:9], v[202:209]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[28:35] /*v[284:291]*/, v[18:25], v[162:169]
	s_set_vgpr_msb 0x100
	ds_load_b128 v[178:181], v255 offset:25600
	ds_load_b128 v[182:185], v255 offset:25632
	ds_load_b128 v[210:213], v255 offset:25664
	ds_load_b128 v[214:217], v255 offset:25696
	ds_load_b128 v[226:229], v255 offset:25728
	ds_load_b128 v[230:233], v255 offset:25760
	ds_load_b128 v[234:237], v255 offset:32000
	ds_load_b128 v[238:241], v255 offset:32032
	ds_load_b128 v[242:245], v255 offset:32064
	ds_load_b128 v[246:249], v255 offset:32096
	s_set_vgpr_msb 64
	ds_load_b128 v[2:5] /*v[258:261]*/, v255 offset:32128
	ds_load_b128 v[6:9] /*v[262:265]*/, v255 offset:32160
	ds_load_b128 v[14:17] /*v[270:273]*/, v255 offset:25792
	ds_load_b128 v[18:21] /*v[274:277]*/, v255 offset:25824
	ds_load_b128 v[28:31] /*v[284:287]*/, v255 offset:25856
	ds_load_b128 v[32:35] /*v[288:291]*/, v255 offset:25888
	ds_load_b128 v[36:39] /*v[292:295]*/, v255 offset:25920
	ds_load_b128 v[40:43] /*v[296:299]*/, v255 offset:25952
	ds_load_b128 v[44:47] /*v[300:303]*/, v255 offset:32192
	ds_load_b128 v[48:51] /*v[304:307]*/, v255 offset:32224
	ds_load_b128 v[52:55] /*v[308:311]*/, v255 offset:32256
	ds_load_b128 v[56:59] /*v[312:315]*/, v255 offset:32288
	ds_load_b128 v[60:63] /*v[316:319]*/, v255 offset:32320
	ds_load_b128 v[64:67] /*v[320:323]*/, v255 offset:32352
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[218:225], v[178:185], v[66:73], 0
	v_wmma_f32_16x16x32_bf16 v[194:201], v[178:185], v[90:97], 0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[210:217], v[50:57], v[218:225]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[210:217], v[82:89], v[194:201]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[226:233], v[34:41], v[218:225]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[226:233], v[74:81], v[194:201]
	s_set_vgpr_msb 1
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[14:21] /*v[270:277]*/, v[42:49], v[218:225]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[14:21] /*v[270:277]*/, v[58:65], v[194:201]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[28:35] /*v[284:291]*/, v[10:17], v[218:225]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[28:35] /*v[284:291]*/, v[26:33], v[194:201]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[218:225], v[36:43] /*v[292:299]*/, v[2:9], v[218:225]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[36:43] /*v[292:299]*/, v[18:25], v[194:201]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[226:233], v[234:241], v[66:73], 0
	v_wmma_f32_16x16x32_bf16 v[210:217], v[234:241], v[90:97], 0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[226:233], v[242:249], v[50:57], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[242:249], v[82:89], v[210:217]
	s_set_vgpr_msb 1
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[226:233], v[2:9] /*v[258:265]*/, v[34:41], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[2:9] /*v[258:265]*/, v[74:81], v[210:217]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[226:233], v[44:51] /*v[300:307]*/, v[42:49], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[44:51] /*v[300:307]*/, v[58:65], v[210:217]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[226:233], v[52:59] /*v[308:315]*/, v[10:17], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[52:59] /*v[308:315]*/, v[26:33], v[210:217]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[226:233], v[60:67] /*v[316:323]*/, v[2:9], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[60:67] /*v[316:323]*/, v[18:25], v[210:217]
	s_set_vgpr_msb 0x100
	ds_load_b128 v[178:181], v255 offset:38400
	ds_load_b128 v[182:185], v255 offset:38432
	s_set_vgpr_msb 64
	ds_load_b128 v[2:5] /*v[258:261]*/, v255 offset:38464
	ds_load_b128 v[6:9] /*v[262:265]*/, v255 offset:38496
	ds_load_b128 v[14:17] /*v[270:273]*/, v255 offset:38528
	ds_load_b128 v[18:21] /*v[274:277]*/, v255 offset:38560
	ds_load_b128 v[28:31] /*v[284:287]*/, v255 offset:44800
	ds_load_b128 v[32:35] /*v[288:291]*/, v255 offset:44832
	ds_load_b128 v[36:39] /*v[292:295]*/, v255 offset:44864
	ds_load_b128 v[40:43] /*v[296:299]*/, v255 offset:44896
	ds_load_b128 v[44:47] /*v[300:303]*/, v255 offset:44928
	ds_load_b128 v[48:51] /*v[304:307]*/, v255 offset:44960
	ds_load_b128 v[52:55] /*v[308:311]*/, v255 offset:38592
	ds_load_b128 v[56:59] /*v[312:315]*/, v255 offset:38624
	ds_load_b128 v[60:63] /*v[316:319]*/, v255 offset:38656
	ds_load_b128 v[64:67] /*v[320:323]*/, v255 offset:38688
	ds_load_b128 v[68:71] /*v[324:327]*/, v255 offset:38720
	ds_load_b128 v[72:75] /*v[328:331]*/, v255 offset:38752
	ds_load_b128 v[76:79] /*v[332:335]*/, v255 offset:44992
	ds_load_b128 v[80:83] /*v[336:339]*/, v255 offset:45024
	ds_load_b128 v[84:87] /*v[340:343]*/, v255 offset:45056
	ds_load_b128 v[88:91] /*v[344:347]*/, v255 offset:45088
	ds_load_b128 v[92:95] /*v[348:351]*/, v255 offset:45120
	ds_load_b128 v[96:99] /*v[352:355]*/, v255 offset:45152
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[242:249], v[178:185], v[66:73], 0
	v_wmma_f32_16x16x32_bf16 v[234:241], v[178:185], v[90:97], 0
	s_set_vgpr_msb 1
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[2:9] /*v[258:265]*/, v[50:57], v[242:249]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[2:9] /*v[258:265]*/, v[82:89], v[234:241]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[14:21] /*v[270:277]*/, v[34:41], v[242:249]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[14:21] /*v[270:277]*/, v[74:81], v[234:241]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[52:59] /*v[308:315]*/, v[42:49], v[242:249]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[52:59] /*v[308:315]*/, v[58:65], v[234:241]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[60:67] /*v[316:323]*/, v[10:17], v[242:249]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[60:67] /*v[316:323]*/, v[26:33], v[234:241]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[242:249], v[68:75] /*v[324:331]*/, v[2:9], v[242:249]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[68:75] /*v[324:331]*/, v[18:25], v[234:241]
	s_set_vgpr_msb 0x141
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[28:35] /*v[284:291]*/, v[66:73], 0
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[178:185], v[28:35] /*v[284:291]*/, v[90:97], 0
	s_set_vgpr_msb 0x151
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[36:43] /*v[292:299]*/, v[50:57], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5101
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[36:43] /*v[292:299]*/, v[82:89], v[178:185]
	s_set_vgpr_msb 0x151
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[44:51] /*v[300:307]*/, v[34:41], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5101
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[44:51] /*v[300:307]*/, v[74:81], v[178:185]
	s_set_vgpr_msb 0x151
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[76:83] /*v[332:339]*/, v[42:49], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5101
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[76:83] /*v[332:339]*/, v[58:65], v[178:185]
	s_set_vgpr_msb 0x151
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[84:91] /*v[340:347]*/, v[10:17], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5101
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[84:91] /*v[340:347]*/, v[26:33], v[178:185]
	s_set_vgpr_msb 0x151
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[92:99] /*v[348:355]*/, v[2:9], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5101
	s_delay_alu instid0(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[178:185], v[92:99] /*v[348:355]*/, v[18:25], v[178:185]
	s_set_vgpr_msb 0x140
	v_mul_lo_u32 v14 /*v270*/, v252, s19
	s_ashr_i32 s23, s22, 31
	s_set_vgpr_msb 0x4008
	v_mul_u32_u24_e32 v252, 0x900, v75 /*v587*/
	v_readfirstlane_b32 s4, v170
	v_readfirstlane_b32 s5, v171
	v_readfirstlane_b32 s7, v173
	v_readfirstlane_b32 s6, v172
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v171, 0x20000, v252
	s_set_vgpr_msb 0x44
	v_ashrrev_i32_e32 v15 /*v271*/, 31, v14 /*v270*/
	v_readfirstlane_b32 s8, v174
	v_readfirstlane_b32 s9, v175
	v_readfirstlane_b32 s10, v176
	v_readfirstlane_b32 s11, v177
	v_add_nc_u64_e32 v[18:19] /*v[274:275]*/, s[2:3], v[14:15] /*v[270:271]*/
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, s[2:3]
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, s[0:1]
	v_readfirstlane_b32 s13, v171
	v_readfirstlane_b32 s68, v98
	s_mov_b32 s2, s22
	v_add_nc_u64_e32 v[16:17] /*v[272:273]*/, s[22:23], v[18:19] /*v[274:275]*/
	s_ashr_i32 s67, s66, 31
	s_set_vgpr_msb 0x4441
	v_readfirstlane_b32 s12, v14 /*v270*/
	v_writelane_b32 v0 /*v256*/, s2, 25
	s_mov_b32 s1, 0x22400
	s_set_vgpr_msb 0x4104
	v_readfirstlane_b32 s69, v99
	v_or_b32_e32 v173, 0x80000000, v17 /*v273*/
	s_set_vgpr_msb 0x401
	v_dual_mov_b32 v170, v16 /*v272*/ :: v_dual_lshrrev_b32 v98, 1, v0
	s_set_vgpr_msb 0x140
	v_writelane_b32 v0 /*v256*/, s3, 26
	v_readfirstlane_b32 s70, v100
	v_readfirstlane_b32 s15, v173
	v_readfirstlane_b32 s14, v170
	s_set_vgpr_msb 0x4004
	v_mov_b64_e32 v[172:173], s[2:3]
	v_add_nc_u64_e32 v[172:173], s[66:67], v[16:17] /*v[272:273]*/
	v_mov_b64_e32 v[170:171], s[0:1]
	s_set_vgpr_msb 0x440
	v_writelane_b32 v0 /*v256*/, s16, 27
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_readfirstlane_b32 s4, v138
	v_readfirstlane_b32 s5, v139
	v_readfirstlane_b32 s7, v141
	s_set_vgpr_msb 0x4008
	v_mad_u32_u24 v139, 0x900, v75 /*v587*/, s1
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v141, 0x80000000, v173
	v_mov_b32_e32 v138, v172
	v_readfirstlane_b32 s6, v140
	v_readfirstlane_b32 s8, v142
	v_readfirstlane_b32 s9, v143
	v_readfirstlane_b32 s10, v144
	v_readfirstlane_b32 s11, v145
	v_readfirstlane_b32 s12, v170
	v_readfirstlane_b32 s13, v139
	v_readfirstlane_b32 s14, v138
	v_readfirstlane_b32 s15, v141
	s_barrier_wait -1
	s_mov_b32 s1, 0x24800
	v_mov_b64_e32 v[140:141], s[2:3]
	v_add_nc_u64_e32 v[140:141], s[66:67], v[172:173]
	v_mov_b64_e32 v[138:139], s[0:1]
	s_set_vgpr_msb 64
	v_writelane_b32 v1 /*v257*/, s21, 0
	s_set_vgpr_msb 0x4080
	v_and_b32_e32 v146 /*v658*/, 8, v98
	v_bitop3_b32 v150 /*v662*/, v98, 8, v98 bitop3:0x3f
	v_readfirstlane_b32 s71, v101
	v_readfirstlane_b32 s72, v102
	s_set_vgpr_msb 0x8040
	v_writelane_b32 v1 /*v257*/, s22, 1
	s_set_vgpr_msb 0x400a
	v_sub_nc_u32_e32 v98, v148 /*v660*/, v146 /*v658*/
	s_set_vgpr_msb 0xa40
	v_readfirstlane_b32 s73, v103
	v_readfirstlane_b32 s74, v104
	v_readfirstlane_b32 s75, v105
	v_writelane_b32 v1 /*v257*/, s23, 2
	s_sub_co_i32 s103, s98, s26
	v_writelane_b32 v0 /*v256*/, s17, 28
	s_set_vgpr_msb 0x4008
	v_add_nc_u32_e32 v100, v98, v149 /*v661*/
	s_set_vgpr_msb 0x840
	v_writelane_b32 v1 /*v257*/, s26, 3
	v_writelane_b32 v0 /*v256*/, s18, 29
	s_set_vgpr_msb 0x4080
	v_add3_u32 v151 /*v663*/, s103, s63, v100
	s_set_vgpr_msb 0x8048
	v_writelane_b32 v0 /*v256*/, s19, 30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s64, 1, v151 /*v663*/
	s_set_vgpr_msb 0x4808
	v_add_nc_u32_e32 v100, -16, v151 /*v663*/
	v_cmp_gt_i32_e64 s65, 2, v151 /*v663*/
	v_subrev_nc_u32_e32 v142, 64, v151 /*v663*/
	v_add_nc_u32_e32 v143, 0xffffffb0, v151 /*v663*/
	s_set_vgpr_msb 0x840
	v_writelane_b32 v0 /*v256*/, s20, 31
	s_set_vgpr_msb 0x4008
	v_add_nc_u32_e32 v176, 0xffffffa0, v151 /*v663*/
	v_add_nc_u32_e32 v177, 0xffffff90, v151 /*v663*/
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_readfirstlane_b32 s4, v114
	v_readfirstlane_b32 s5, v115
	v_readfirstlane_b32 s7, v117
	v_mad_u32_u24 v115, 0x900, v75 /*v587*/, s1
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v117, 0x80000000, v141
	v_mov_b32_e32 v114, v140
	v_readfirstlane_b32 s6, v116
	v_readfirstlane_b32 s8, v118
	v_readfirstlane_b32 s9, v119
	v_readfirstlane_b32 s10, v120
	v_readfirstlane_b32 s11, v121
	v_readfirstlane_b32 s12, v138
	v_readfirstlane_b32 s13, v115
	v_readfirstlane_b32 s14, v114
	v_readfirstlane_b32 s15, v117
	s_barrier_wait -1
	s_mov_b32 s1, 0x26c00
	v_mov_b64_e32 v[116:117], s[2:3]
	v_mov_b64_e32 v[114:115], s[0:1]
	s_set_vgpr_msb 8
	v_mad_u32_u24 v99, 0x900, v75 /*v587*/, s1
	v_readfirstlane_b32 s1, v252
	v_dual_add_nc_u32 v138, s98, v150 /*v662*/ :: v_dual_add_nc_u32 v139, 16, v151 /*v663*/
	s_delay_alu instid0(VALU_DEP_3)
	v_readfirstlane_b32 s77, v99
	s_set_vgpr_msb 0x840
	v_writelane_b32 v1 /*v257*/, s1, 4
	v_readfirstlane_b32 s76, v114
	v_cmp_gt_i32_e64 s54, 1, v138
	v_cmp_gt_i32_e64 s53, 2, v138
	v_cmp_gt_i32_e64 s52, 3, v138
	v_writelane_b32 v1 /*v257*/, s33, 5
	v_cmp_gt_i32_e64 s62, 16, v138
	v_cmp_gt_i32_e64 s51, 4, v138
	v_cmp_gt_i32_e64 s61, 17, v138
	v_cmp_gt_i32_e64 s50, 5, v138
	v_writelane_b32 v1 /*v257*/, s66, 6
	s_set_vgpr_msb 0x4000
	v_add_nc_u64_e32 v[98:99], s[66:67], v[140:141]
	v_cmp_gt_i32_e64 s60, 18, v138
	v_cmp_gt_i32_e64 s58, 20, v138
	v_cmp_gt_i32_e64 s55, 22, v138
	s_set_vgpr_msb 0x41
	v_writelane_b32 v1 /*v257*/, s67, 7
	v_readfirstlane_b32 s66, v18 /*v274*/
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v99, 0x80000000, v99
	v_readfirstlane_b32 s78, v98
	s_set_vgpr_msb 8
	v_or_b32_e32 v98, v138, v151 /*v663*/
	s_set_vgpr_msb 0x841
	v_writelane_b32 v1 /*v257*/, s63, 8
	v_readfirstlane_b32 s67, v19 /*v275*/
	s_set_vgpr_msb 0x4108
	v_readfirstlane_b32 s79, v99
	v_cmp_gt_i32_e64 s63, 0, v151 /*v663*/
	v_subrev_nc_u32_e32 v140, 32, v151 /*v663*/
	s_set_vgpr_msb 0x840
	v_writelane_b32 v1 /*v257*/, s66, 9
	v_cmp_gt_i32_e64 s49, 6, v138
	v_cmp_gt_i32_e64 s56, 23, v138
	s_or_b32 s63, s62, s63
	v_cmp_gt_i32_e64 s48, 7, v138
	v_writelane_b32 v1 /*v257*/, s67, 10
	s_set_vgpr_msb 0x4008
	v_cmp_gt_i32_e64 s66, 3, v151 /*v663*/
	v_cmp_lt_i32_e64 s95, v138, 32
	v_cmp_lt_i32_e64 s81, v140, 0
	v_cmp_lt_i32_e64 s59, v138, 19
	v_cmp_lt_i32_e64 s57, v138, 21
	v_cmp_lt_i32_e64 s94, v138, 33
	v_cmp_lt_i32_e64 s84, v140, 1
	v_cmp_lt_i32_e64 s92, v138, 34
	v_cmp_lt_i32_e64 s45, v138, 35
	v_cmp_lt_i32_e64 s42, v138, 36
	v_cmp_lt_i32_e64 s67, v100, 4
	v_cmp_lt_i32_e64 s31, v138, 37
	v_cmp_lt_i32_e64 s80, v140, 5
	v_subrev_nc_u32_e32 v141, 48, v151 /*v663*/
	v_cmp_lt_i32_e64 s30, v138, 38
	v_cmp_lt_i32_e64 s82, v140, 6
	v_cmp_lt_i32_e64 s38, v138, 39
	v_cmp_lt_i32_e64 s83, v140, 7
	v_cmp_lt_i32_e64 s29, v138, 48
	v_cmp_lt_i32_e64 s39, v138, 49
	v_cmp_lt_i32_e64 s37, v138, 50
	v_cmp_lt_i32_e64 s36, v138, 51
	v_cmp_lt_i32_e64 s35, v138, 52
	v_cmp_lt_i32_e64 s34, v138, 53
	v_cmp_lt_i32_e64 s33, v138, 54
	v_cmp_lt_i32_e64 s91, v138, 55
	v_cmp_lt_i32_e64 s23, v138, 64
	s_set_vgpr_msb 0x800
	v_cmp_gt_i32_e64 s26, 0x41, v138
	v_cmp_gt_i32_e64 s86, 0x42, v138
	v_cmp_gt_i32_e64 s88, 0x43, v138
	v_cmp_gt_i32_e64 s90, 0x44, v138
	v_cmp_gt_i32_e64 s89, 0x45, v138
	v_cmp_gt_i32_e64 s87, 0x46, v138
	v_cmp_gt_i32_e64 s27, 0x47, v138
	v_cmp_gt_i32_e64 s16, 0x51, v138
	v_cmp_gt_i32_e64 s18, 0x52, v138
	v_cmp_gt_i32_e64 s20, 0x53, v138
	v_cmp_gt_i32_e64 s22, 0x54, v138
	v_cmp_gt_i32_e64 s21, 0x55, v138
	v_cmp_gt_i32_e64 s19, 0x56, v138
	v_cmp_gt_i32_e64 s17, 0x57, v138
	v_cmp_gt_i32_e64 s96, 0x73, v138
	v_cmp_gt_i32_e64 s3, 0x74, v138
	v_cmp_gt_i32_e64 s2, 0x75, v138
	v_cmp_gt_i32_e64 s1, 0x76, v138
	v_cmp_gt_i32_e32 vcc_lo, 0x77, v138
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_cmp_gt_i32_e64 s15, 0x50, v138
	v_cmp_gt_i32_e64 s7, 0x60, v138
	v_cmp_gt_i32_e64 s8, 0x61, v138
	v_cmp_gt_i32_e64 s10, 0x62, v138
	v_cmp_gt_i32_e64 s12, 0x63, v138
	v_cmp_gt_i32_e64 s14, 0x64, v138
	v_cmp_gt_i32_e64 s13, 0x65, v138
	v_cmp_gt_i32_e64 s11, 0x66, v138
	v_cmp_gt_i32_e64 s9, 0x67, v138
	v_cmp_gt_i32_e64 s6, 0x70, v138
	v_cmp_gt_i32_e64 s5, 0x71, v138
	v_cmp_gt_i32_e64 s4, 0x72, v138
	s_barrier_wait -1
	tensor_load_to_lds s[76:79], s[68:75]
	v_cmp_lt_i32_e64 s70, -1, v98
	v_cmp_gt_i32_e64 s73, 0, v100
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e64 s75, 4, v151 /*v663*/
	v_cmp_gt_i32_e64 s76, 5, v151 /*v663*/
	v_cmp_gt_i32_e64 s77, 6, v151 /*v663*/
	s_set_vgpr_msb 0x800
	v_cndmask_b32_e64 v120, 0xff800000, v130, s70
	s_or_b32 s70, s54, s64
	s_or_b32 s62, s62, s73
	v_cndmask_b32_e64 v121, v131, 0xff800000, s70
	s_or_b32 s70, s53, s65
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e64 s78, 7, v151 /*v663*/
	v_cndmask_b32_e64 v130, v132, 0xff800000, s70
	s_or_b32 s70, s52, s66
	v_cndmask_b32_e64 v98, v154, 0xff800000, s63
	v_cndmask_b32_e64 v131, v133, 0xff800000, s70
	s_or_b32 s70, s51, s75
	s_or_b32 s63, s61, s64
	v_cndmask_b32_e64 v114, v122, 0xff800000, s62
	v_cmp_lt_i32_e64 s62, v139, 7
	v_cndmask_b32_e64 v132, v134, 0xff800000, s70
	s_or_b32 s70, s50, s76
	v_cndmask_b32_e64 v99, v155, 0xff800000, s63
	s_or_b32 s63, s60, s65
	s_or_b32 s65, s58, s75
	s_or_b32 s75, s55, s77
	v_cmp_lt_i32_e64 s69, v100, 2
	v_cndmask_b32_e64 v133, v135, 0xff800000, s70
	s_or_b32 s70, s49, s77
	v_cndmask_b32_e64 v104, v160, 0xff800000, s75
	s_or_b32 s75, s56, s78
	v_cndmask_b32_e64 v134, v136, 0xff800000, s70
	s_or_b32 s70, s48, s78
	s_or_b32 s48, s48, s62
	v_cmp_lt_i32_e64 s71, v100, 6
	v_cndmask_b32_e64 v105, v161, 0xff800000, s75
	v_cmp_lt_i32_e64 s75, v140, 2
	v_cndmask_b32_e64 v113, v113, 0xff800000, s48
	s_or_b32 s48, s95, s81
	v_cmp_lt_i32_e64 s72, v100, 1
	v_cndmask_b32_e64 v135, v137, 0xff800000, s70
	v_cmp_lt_i32_e64 s70, v100, 7
	s_or_b32 s64, s59, s66
	s_or_b32 s66, s57, s76
	s_or_b32 s60, s60, s69
	v_cmp_lt_i32_e64 s76, v140, 3
	v_cndmask_b32_e64 v160, v186, 0xff800000, s48
	s_or_b32 s48, s94, s84
	v_cndmask_b32_e64 v116, v124, 0xff800000, s60
	v_cmp_lt_i32_e64 s79, v140, 4
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v124, v138, v139
	v_cmp_gt_i32_e64 s68, 3, v100
	v_cmp_gt_i32_e64 s74, 5, v100
	v_cndmask_b32_e64 v100, v156, 0xff800000, s63
	v_cmp_gt_i32_e64 s63, 1, v139
	s_or_b32 s55, s55, s71
	v_cndmask_b32_e64 v161, v187, 0xff800000, s48
	s_or_b32 s48, s92, s75
	s_or_b32 s61, s61, s72
	v_cndmask_b32_e64 v122, v128, 0xff800000, s55
	s_or_b32 s55, s56, s70
	v_cndmask_b32_e64 v170, v188, 0xff800000, s48
	s_or_b32 s48, s45, s76
	v_cndmask_b32_e64 v101, v157, 0xff800000, s64
	v_cmp_gt_i32_e64 s64, 2, v139
	v_cndmask_b32_e64 v115, v123, 0xff800000, s61
	v_cndmask_b32_e64 v123, v129, 0xff800000, s55
	v_cmp_lt_i32_e64 s55, -1, v124
	v_cndmask_b32_e64 v171, v189, 0xff800000, s48
	s_or_b32 s48, s42, s79
	s_or_b32 s54, s54, s63
	v_cndmask_b32_e64 v172, v190, 0xff800000, s48
	s_or_b32 s48, s31, s80
	v_cndmask_b32_e64 v107, v107, 0xff800000, s54
	v_cmp_gt_i32_e64 s54, 0, v141
	v_cndmask_b32_e64 v173, v191, 0xff800000, s48
	s_or_b32 s48, s30, s82
	v_cndmask_b32_e64 v106, 0xff800000, v106, s55
	s_or_b32 s55, s53, s64
	v_cmp_gt_i32_e64 s53, 1, v141
	v_cndmask_b32_e64 v174, v192, 0xff800000, s48
	s_or_b32 s48, s38, s83
	v_cndmask_b32_e64 v108, v108, 0xff800000, s55
	v_cmp_gt_i32_e64 s55, 2, v141
	v_cndmask_b32_e64 v175, v193, 0xff800000, s48
	s_or_b32 s48, s29, s54
	v_cmp_gt_i32_e64 s61, 3, v141
	v_cndmask_b32_e64 v102, v158, 0xff800000, s65
	v_cmp_gt_i32_e64 s65, 3, v139
	v_cndmask_b32_e64 v144, v202, 0xff800000, s48
	s_or_b32 s48, s39, s53
	v_cmp_gt_i32_e64 s63, 4, v141
	v_cmp_gt_i32_e64 s77, 5, v139
	v_cndmask_b32_e64 v145, v203, 0xff800000, s48
	s_or_b32 s48, s37, s55
	v_cmp_gt_i32_e64 s64, 5, v141
	v_cndmask_b32_e64 v103, v159, 0xff800000, s66
	v_cmp_gt_i32_e64 s66, 4, v139
	v_cndmask_b32_e64 v154, v204, 0xff800000, s48
	s_or_b32 s48, s36, s61
	s_or_b32 s52, s52, s65
	v_cmp_gt_i32_e64 s65, 6, v141
	v_cmp_gt_i32_e64 s78, 6, v139
	v_cndmask_b32_e64 v155, v205, 0xff800000, s48
	s_or_b32 s48, s35, s63
	s_or_b32 s50, s50, s77
	v_cndmask_b32_e64 v156, v206, 0xff800000, s48
	s_or_b32 s48, s34, s64
	s_or_b32 s51, s51, s66
	v_cndmask_b32_e64 v111, v111, 0xff800000, s50
	v_cmp_gt_i32_e64 s50, 0, v142
	v_cndmask_b32_e64 v157, v207, 0xff800000, s48
	s_or_b32 s48, s33, s65
	s_or_b32 s33, s33, s82
	v_cndmask_b32_e64 v110, v110, 0xff800000, s51
	s_or_b32 s51, s49, s78
	v_cmp_gt_i32_e64 s49, 1, v142
	v_cmp_gt_i32_e64 s66, 7, v141
	v_cndmask_b32_e64 v136, v168, 0xff800000, s33
	s_or_b32 s33, s91, s83
	v_cndmask_b32_e64 v109, v109, 0xff800000, s52
	v_cmp_gt_i32_e64 s52, 2, v142
	v_cndmask_b32_e64 v137, v169, 0xff800000, s33
	s_or_b32 s33, s23, s50
	v_cmp_gt_i32_e64 s62, 3, v142
	v_cndmask_b32_e64 v202, v218, 0xff800000, s33
	s_or_b32 s33, s26, s49
	v_cndmask_b32_e64 v158, v208, 0xff800000, s48
	s_or_b32 s48, s91, s66
	v_cmp_gt_i32_e64 s77, 4, v142
	v_cndmask_b32_e64 v203, v219, 0xff800000, s33
	s_or_b32 s33, s86, s52
	v_cndmask_b32_e64 v159, v209, 0xff800000, s48
	s_or_b32 s48, s95, s73
	v_cmp_gt_i32_e64 s73, 5, v142
	v_cndmask_b32_e64 v204, v220, 0xff800000, s33
	s_or_b32 s33, s88, s62
	v_cmp_gt_i32_e64 s78, 6, v142
	v_cndmask_b32_e64 v205, v221, 0xff800000, s33
	s_or_b32 s33, s90, s77
	v_cndmask_b32_e64 v138, v146, 0xff800000, s48
	s_or_b32 s48, s94, s72
	v_cmp_gt_i32_e64 s72, 7, v142
	v_cndmask_b32_e64 v206, v222, 0xff800000, s33
	s_or_b32 s33, s89, s73
	v_cmp_gt_i32_e64 s94, 0, v143
	v_cndmask_b32_e64 v207, v223, 0xff800000, s33
	s_or_b32 s33, s87, s78
	v_cndmask_b32_e64 v112, v112, 0xff800000, s51
	v_cndmask_b32_e64 v208, v224, 0xff800000, s33
	s_or_b32 s33, s27, s72
	s_or_b32 s51, s92, s69
	v_cndmask_b32_e64 v209, v225, 0xff800000, s33
	s_or_b32 s33, s15, s94
	s_or_b32 s15, s15, s50
	v_cndmask_b32_e64 v140, v148, 0xff800000, s51
	s_or_b32 s45, s45, s68
	v_cndmask_b32_e64 v148, v210, 0xff800000, s15
	s_or_b32 s15, s16, s49
	v_cndmask_b32_e64 v141, v149, 0xff800000, s45
	s_or_b32 s42, s42, s67
	v_cndmask_b32_e64 v149, v211, 0xff800000, s15
	s_or_b32 s15, s18, s52
	s_or_b32 s59, s59, s68
	s_or_b32 s58, s58, s67
	s_or_b32 s57, s57, s74
	v_cndmask_b32_e64 v142, v150, 0xff800000, s42
	s_or_b32 s31, s31, s74
	v_cndmask_b32_e64 v150, v212, 0xff800000, s15
	s_or_b32 s15, s20, s62
	v_cndmask_b32_e64 v117, v125, 0xff800000, s59
	v_cndmask_b32_e64 v118, v126, 0xff800000, s58
	v_cndmask_b32_e64 v119, v127, 0xff800000, s57
	v_cndmask_b32_e64 v139, v147, 0xff800000, s48
	v_cmp_gt_i32_e64 s48, 1, v143
	v_cmp_gt_i32_e64 s51, 2, v143
	v_cmp_gt_i32_e64 s56, 3, v143
	v_cmp_gt_i32_e64 s57, 4, v143
	v_cmp_gt_i32_e64 s58, 5, v143
	v_cmp_gt_i32_e64 s59, 6, v143
	v_cmp_gt_i32_e64 s60, 7, v143
	v_cndmask_b32_e64 v143, v151, 0xff800000, s31
	s_or_b32 s30, s30, s71
	v_cndmask_b32_e64 v151, v213, 0xff800000, s15
	s_or_b32 s15, s22, s77
	v_cndmask_b32_e64 v146, v152, 0xff800000, s30
	s_or_b32 s30, s38, s70
	v_cndmask_b32_e64 v152, v214, 0xff800000, s15
	s_or_b32 s15, s21, s73
	v_cmp_gt_i32_e64 s74, 0, v176
	v_cndmask_b32_e64 v147, v153, 0xff800000, s30
	s_or_b32 s29, s29, s81
	v_cndmask_b32_e64 v153, v215, 0xff800000, s15
	s_or_b32 s15, s19, s78
	v_cmp_gt_i32_e64 s71, 1, v176
	v_cndmask_b32_e64 v124, v162, 0xff800000, s29
	s_or_b32 s39, s39, s84
	v_cndmask_b32_e64 v162, v216, 0xff800000, s15
	s_or_b32 s15, s17, s72
	v_cmp_gt_i32_e64 s45, 2, v176
	v_cndmask_b32_e64 v125, v163, 0xff800000, s39
	s_or_b32 s67, s37, s75
	s_or_b32 s23, s23, s54
	v_cndmask_b32_e64 v163, v217, 0xff800000, s15
	s_or_b32 s15, s7, s74
	s_barrier_signal -1
	v_cmp_gt_i32_e64 s42, 3, v176
	v_cndmask_b32_e64 v126, v164, 0xff800000, s67
	s_or_b32 s68, s36, s76
	v_cndmask_b32_e64 v164, v194, 0xff800000, s23
	s_or_b32 s23, s26, s53
	v_cndmask_b32_e64 v194, v242, 0xff800000, s15
	s_or_b32 s15, s8, s71
	v_cmp_gt_i32_e64 s31, 4, v176
	v_cndmask_b32_e64 v127, v165, 0xff800000, s68
	s_or_b32 s69, s35, s79
	v_cndmask_b32_e64 v165, v195, 0xff800000, s23
	s_or_b32 s23, s86, s55
	v_cndmask_b32_e64 v195, v243, 0xff800000, s15
	s_or_b32 s15, s10, s45
	v_cmp_gt_i32_e64 s30, 5, v176
	v_cndmask_b32_e64 v128, v166, 0xff800000, s69
	s_or_b32 s70, s34, s80
	v_cndmask_b32_e64 v166, v196, 0xff800000, s23
	s_or_b32 s23, s88, s61
	v_cndmask_b32_e64 v196, v244, 0xff800000, s15
	s_or_b32 s15, s12, s42
	v_cmp_gt_i32_e64 s29, 6, v176
	v_cndmask_b32_e64 v129, v167, 0xff800000, s70
	v_cndmask_b32_e64 v167, v197, 0xff800000, s23
	s_or_b32 s23, s90, s63
	v_cndmask_b32_e64 v197, v245, 0xff800000, s15
	s_or_b32 s15, s14, s31
	v_cmp_gt_i32_e64 s38, 7, v176
	v_cndmask_b32_e64 v168, v198, 0xff800000, s23
	s_or_b32 s23, s89, s64
	v_cndmask_b32_e64 v198, v246, 0xff800000, s15
	s_or_b32 s15, s13, s30
	v_cmp_gt_i32_e64 s39, 0, v177
	v_cndmask_b32_e64 v169, v199, 0xff800000, s23
	s_or_b32 s23, s87, s65
	v_cndmask_b32_e64 v199, v247, 0xff800000, s15
	s_or_b32 s15, s11, s29
	v_cmp_gt_i32_e64 s37, 1, v177
	v_cndmask_b32_e64 v176, v200, 0xff800000, s23
	s_or_b32 s23, s27, s66
	v_cndmask_b32_e64 v200, v248, 0xff800000, s15
	s_or_b32 s15, s9, s38
	v_cmp_gt_i32_e64 s67, 2, v177
	v_cmp_gt_i32_e64 s36, 3, v177
	v_cmp_gt_i32_e64 s68, 4, v177
	v_cmp_gt_i32_e64 s35, 5, v177
	v_cmp_gt_i32_e64 s69, 6, v177
	v_cmp_gt_i32_e64 s34, 7, v177
	v_cndmask_b32_e64 v186, v226, 0xff800000, s33
	s_or_b32 s33, s16, s48
	v_cndmask_b32_e64 v177, v201, 0xff800000, s23
	v_cndmask_b32_e64 v201, v249, 0xff800000, s15
	s_or_b32 s15, s6, s39
	s_or_b32 s7, s7, s94
	s_barrier_wait -1
	v_cndmask_b32_e64 v187, v227, 0xff800000, s33
	s_or_b32 s33, s18, s51
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v210, v2 /*v258*/, 0xff800000, s15
	s_or_b32 s15, s5, s37
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v218, v234, 0xff800000, s7
	s_or_b32 s7, s8, s48
	v_cndmask_b32_e64 v188, v228, 0xff800000, s33
	s_or_b32 s33, s20, s56
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v211, v3 /*v259*/, 0xff800000, s15
	s_or_b32 s15, s4, s67
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v219, v235, 0xff800000, s7
	s_or_b32 s7, s10, s51
	v_cndmask_b32_e64 v189, v229, 0xff800000, s33
	s_or_b32 s33, s22, s57
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v212, v4 /*v260*/, 0xff800000, s15
	s_or_b32 s15, s96, s36
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v220, v236, 0xff800000, s7
	s_or_b32 s7, s12, s56
	v_cndmask_b32_e64 v190, v230, 0xff800000, s33
	s_or_b32 s33, s21, s58
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v213, v5 /*v261*/, 0xff800000, s15
	s_or_b32 s15, s3, s68
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v221, v237, 0xff800000, s7
	s_or_b32 s7, s14, s57
	v_cndmask_b32_e64 v191, v231, 0xff800000, s33
	s_or_b32 s33, s19, s59
	s_set_vgpr_msb 1
	v_readlane_b32 s66, v0 /*v256*/, 19
	v_cndmask_b32_e64 v214, v6 /*v262*/, 0xff800000, s15
	s_or_b32 s15, s2, s35
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v222, v238, 0xff800000, s7
	s_or_b32 s7, s13, s58
	s_or_b32 s6, s6, s74
	s_set_vgpr_msb 1
	v_readlane_b32 s72, v0 /*v256*/, 27
	v_readlane_b32 s67, v0 /*v256*/, 20
	v_readlane_b32 s74, v0 /*v256*/, 29
	v_readlane_b32 s75, v0 /*v256*/, 30
	v_readlane_b32 s76, v0 /*v256*/, 31
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v192, v232, 0xff800000, s33
	s_or_b32 s33, s17, s60
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v215, v7 /*v263*/, 0xff800000, s15
	s_or_b32 s15, s1, s69
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v223, v239, 0xff800000, s7
	s_or_b32 s7, s11, s59
	s_or_b32 s4, s4, s45
	s_or_b32 s1, s1, s29
	s_set_vgpr_msb 1
	v_readlane_b32 s78, v1 /*v257*/, 1
	v_readlane_b32 s79, v1 /*v257*/, 2
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v193, v233, 0xff800000, s33
	s_mov_b32 s65, s98
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v216, v8 /*v264*/, 0xff800000, s15
	s_or_b32 s15, vcc_lo, s34
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v224, v240, 0xff800000, s7
	s_or_b32 s7, s9, s60
	s_mov_b64 s[78:79], s[100:101]
	s_or_b32 s5, s5, s71
	v_cndmask_b32_e64 v180, v180, 0xff800000, s4
	s_or_b32 s4, s96, s42
	s_or_b32 s3, s3, s31
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_or_b32 s2, s2, s30
	v_cndmask_b32_e64 v184, v184, 0xff800000, s1
	s_or_b32 s1, vcc_lo, s38
	s_mov_b32 s33, 0x20000
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v217, v9 /*v265*/, 0xff800000, s15
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v225, v241, 0xff800000, s7
	s_set_vgpr_msb 1
	v_readlane_b32 s73, v0 /*v256*/, 28
	v_readlane_b32 s77, v1 /*v257*/, 0
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v178, v178, 0xff800000, s6
	v_cndmask_b32_e64 v179, v179, 0xff800000, s5
	v_cndmask_b32_e64 v181, v181, 0xff800000, s4
	v_cndmask_b32_e64 v182, v182, 0xff800000, s3
	v_cndmask_b32_e64 v183, v183, 0xff800000, s2
	v_cndmask_b32_e64 v185, v185, 0xff800000, s1
	s_barrier_wait -1
	v_max3_num_f32 v226, v120, v121, v130
	v_max3_num_f32 v227, v160, v161, v170
	v_max3_num_f32 v228, v202, v203, v204
	v_max3_num_f32 v229, v194, v195, v196
	v_max3_num_f32 v230, v114, v115, v116
	v_max3_num_f32 v231, v144, v145, v154
	v_max3_num_f32 v232, v186, v187, v188
	v_max3_num_f32 v233, v210, v211, v212
	v_max3_num_f32 v234, v106, v107, v108
	v_max3_num_f32 v235, v138, v139, v140
	v_max3_num_f32 v236, v164, v165, v166
	v_max3_num_f32 v237, v218, v219, v220
	v_max3_num_f32 v238, v98, v99, v100
	v_max3_num_f32 v239, v124, v125, v126
	v_max3_num_f32 v240, v148, v149, v150
	v_max3_num_f32 v241, v178, v179, v180
	v_max3_num_f32 v226, v131, v132, v226
	v_max3_num_f32 v227, v171, v172, v227
	v_max3_num_f32 v228, v205, v206, v228
	v_max3_num_f32 v229, v197, v198, v229
	v_max3_num_f32 v230, v117, v118, v230
	v_max3_num_f32 v231, v155, v156, v231
	v_max3_num_f32 v232, v189, v190, v232
	v_max3_num_f32 v233, v213, v214, v233
	v_max3_num_f32 v234, v109, v110, v234
	v_max3_num_f32 v235, v141, v142, v235
	v_max3_num_f32 v236, v167, v168, v236
	v_max3_num_f32 v237, v221, v222, v237
	v_max3_num_f32 v238, v101, v102, v238
	v_max3_num_f32 v239, v127, v128, v239
	v_max3_num_f32 v240, v151, v152, v240
	v_max3_num_f32 v241, v181, v182, v241
	v_max3_num_f32 v226, v133, v134, v226
	v_max3_num_f32 v227, v173, v174, v227
	v_max3_num_f32 v228, v207, v208, v228
	v_max3_num_f32 v229, v199, v200, v229
	v_max3_num_f32 v230, v119, v122, v230
	v_max3_num_f32 v231, v157, v158, v231
	v_max3_num_f32 v232, v191, v192, v232
	v_max3_num_f32 v233, v215, v216, v233
	v_max3_num_f32 v234, v111, v112, v234
	v_max3_num_f32 v235, v143, v146, v235
	v_max3_num_f32 v236, v169, v176, v236
	v_max3_num_f32 v237, v223, v224, v237
	v_max3_num_f32 v238, v103, v104, v238
	v_max3_num_f32 v239, v129, v136, v239
	v_max3_num_f32 v240, v153, v162, v240
	v_max3_num_f32 v241, v183, v184, v241
	v_max3_num_f32 v226, v135, v226, v120
	v_max3_num_f32 v227, v175, v227, v160
	v_max3_num_f32 v228, v209, v228, v202
	v_max3_num_f32 v229, v201, v229, v194
	v_max3_num_f32 v230, v123, v230, v114
	v_max3_num_f32 v231, v159, v231, v144
	v_max3_num_f32 v232, v193, v232, v186
	v_max3_num_f32 v233, v217, v233, v210
	v_max3_num_f32 v234, v113, v234, v106
	v_max3_num_f32 v235, v147, v235, v138
	v_max3_num_f32 v236, v177, v236, v164
	v_max3_num_f32 v237, v225, v237, v218
	v_max3_num_f32 v238, v105, v238, v98
	v_max3_num_f32 v239, v137, v239, v124
	v_max3_num_f32 v240, v163, v240, v148
	v_max3_num_f32 v241, v185, v241, v178
	v_max3_num_f32 v226, v226, v227, v228
	v_max3_num_f32 v228, v230, v231, v232
	v_max3_num_f32 v230, v234, v235, v236
	v_max3_num_f32 v232, v238, v239, v240
	s_mul_f32 s1, s66, 0xff800000
	v_max3_num_f32 v226, v226, v229, v227
	v_max3_num_f32 v227, v228, v233, v231
	v_max3_num_f32 v228, v230, v237, v235
	v_max3_num_f32 v229, v232, v241, v239
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v230, 0, v226
	v_add_f32_e32 v231, 0, v227
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v232, 0, v228
	v_add_f32_e32 v233, 0, v229
	s_mov_b32 s35, 0x76543210
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_permlanex16_b32 v230, v230, s35, 0xfedcba98
	v_permlanex16_b32 v231, v231, s35, 0xfedcba98
	v_permlanex16_b32 v232, v232, s35, 0xfedcba98
	v_permlanex16_b32 v233, v233, s35, 0xfedcba98
	v_max3_num_f32 v226, v226, v230, 0xff800000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max3_num_f32 v227, v227, v231, 0xff800000
	v_max3_num_f32 v228, v228, v232, 0xff800000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_max3_num_f32 v229, v229, v233, 0xff800000
	v_max3_num_f32 v254, v226, v227, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v252, v228, v229, s1
	v_fma_f32 v226, -v254, s66, s1
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v227, -v252, s66, s1
	s_mov_b32 s67, s66
	s_set_vgpr_msb 0x88
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v74 /*v586*/, v226
	v_nop
	v_mul_f32_e32 v70 /*v582*/, 0, v74 /*v586*/
	s_set_vgpr_msb 0x8800
	v_mul_f32_e64 v226, s66, -v254
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[120:121], v[120:121], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131], v[130:131], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133], v[132:133], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[134:135], v[134:135], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x80
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[160:161], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[122:123] /*v[634:635]*/, v[170:171], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131] /*v[642:643]*/, v[172:173], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[128:129] /*v[640:641]*/, v[174:175], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[116:117] /*v[628:629]*/, v[202:203], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[114:115] /*v[626:627]*/, v[204:205], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[112:113] /*v[624:625]*/, v[206:207], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[110:111] /*v[622:623]*/, v[208:209], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[108:109] /*v[620:621]*/, v[194:195], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[106:107] /*v[618:619]*/, v[196:197], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[10:11] /*v[522:523]*/, v[198:199], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[532:533]*/, v[200:201], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v40 /*v552*/, v120
	v_exp_f32_e32 v41 /*v553*/, v121
	v_exp_f32_e32 v42 /*v554*/, v130
	v_exp_f32_e32 v43 /*v555*/, v131
	v_exp_f32_e32 v52 /*v564*/, v132
	v_exp_f32_e32 v53 /*v565*/, v133
	v_exp_f32_e32 v54 /*v566*/, v134
	v_exp_f32_e32 v55 /*v567*/, v135
	s_set_vgpr_msb 0x8000
	v_pk_fma_f32 v[114:115], v[114:115], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[116:117], v[116:117], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[118:119], v[118:119], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[120:121], v[122:123], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x88
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[144:145], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[154:155], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[134:135] /*v[646:647]*/, v[156:157], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133] /*v[644:645]*/, v[158:159], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[126:127] /*v[638:639]*/, v[186:187], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[100:101] /*v[612:613]*/, v[188:189], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[190:191], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[598:599]*/, v[192:193], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[84:85] /*v[596:597]*/, v[210:211], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[80:81] /*v[592:593]*/, v[212:213], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[50:51] /*v[562:563]*/, v[214:215], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[216:217], s[66:67], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v36 /*v548*/, v114
	v_exp_f32_e32 v37 /*v549*/, v115
	v_exp_f32_e32 v38 /*v550*/, v116
	v_exp_f32_e32 v39 /*v551*/, v117
	v_exp_f32_e32 v44 /*v556*/, v118
	v_exp_f32_e32 v45 /*v557*/, v119
	v_exp_f32_e32 v46 /*v558*/, v120
	v_exp_f32_e32 v47 /*v559*/, v121
	v_exp_f32_e32 v72 /*v584*/, v227
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mul_f32_e32 v68 /*v580*/, 0, v72 /*v584*/
	s_set_vgpr_msb 0x8800
	v_mul_f32_e64 v114, s66, -v252
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[106:107], v[106:107], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[108:109], v[108:109], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[110:111], v[110:111], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[112:113], v[112:113], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x80
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[138:139], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[118:119] /*v[630:631]*/, v[140:141], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[124:125] /*v[636:637]*/, v[142:143], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[104:105] /*v[616:617]*/, v[146:147], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[102:103] /*v[614:615]*/, v[164:165], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[98:99] /*v[610:611]*/, v[166:167], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[82:83] /*v[594:595]*/, v[168:169], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[176:177], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[60:61] /*v[572:573]*/, v[218:219], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[22:23] /*v[534:535]*/, v[220:221], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[222:223], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[224:225], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v14 /*v526*/, v106
	v_exp_f32_e32 v15 /*v527*/, v107
	v_exp_f32_e32 v30 /*v542*/, v108
	v_exp_f32_e32 v31 /*v543*/, v109
	v_exp_f32_e32 v32 /*v544*/, v110
	v_exp_f32_e32 v33 /*v545*/, v111
	v_exp_f32_e32 v34 /*v546*/, v112
	v_exp_f32_e32 v35 /*v547*/, v113
	s_set_vgpr_msb 0x8000
	v_pk_fma_f32 v[98:99], v[98:99], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[100:101], v[100:101], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[102:103], v[102:103], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[104:105], v[104:105], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x80
	v_pk_fma_f32 v[142:143] /*v[654:655]*/, v[124:125], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[120:121] /*v[632:633]*/, v[126:127], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[128:129], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[136:137], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[148:149], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[150:151], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[78:79] /*v[590:591]*/, v[152:153], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[162:163], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[26:27] /*v[538:539]*/, v[178:179], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[180:181], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[588:589]*/, v[182:183], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[58:59] /*v[570:571]*/, v[184:185], s[66:67], v[114:115] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v12 /*v524*/, v98
	v_exp_f32_e32 v13 /*v525*/, v99
	v_exp_f32_e32 v18 /*v530*/, v100
	v_exp_f32_e32 v19 /*v531*/, v101
	v_exp_f32_e32 v16 /*v528*/, v102
	v_exp_f32_e32 v17 /*v529*/, v103
	v_exp_f32_e32 v28 /*v540*/, v104
	v_exp_f32_e32 v29 /*v541*/, v105
	s_add_co_i32 s1, s103, 0x7f
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s23, s1, 7
	s_add_co_i32 s1, s98, 0x7f
	s_add_co_i32 s2, s102, s23
	s_lshr_b32 s22, s1, 7
	s_add_co_i32 s2, s2, 1
	s_min_u32 s60, s2, s22
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s34, s60, -1
	s_set_vgpr_msb 0x8001
	v_readlane_b32 s2, v0 /*v256*/, 21
	v_add_nc_u32_e32 v98, 0xffffff80, v1
	s_lshl_b32 s61, s74, 7
	v_readlane_b32 s3, v0 /*v256*/, 22
	s_set_vgpr_msb 0x104
	v_or_b32_e32 v112, 0x10000, v12 /*v268*/
	s_add_co_i32 s2, s2, s61
	v_med3_i32 v113, v98, 0, 8
	s_ashr_i32 s3, s2, 31
	v_mov_b64_e32 v[104:105], s[46:47]
	v_mov_b64_e32 v[108:109], s[2:3]
	s_set_vgpr_msb 0x400
	v_add_nc_u64_e32 v[110:111], s[2:3], v[250:251]
	v_mov_b64_e32 v[98:99], s[40:41]
	v_mov_b64_e32 v[106:107], s[0:1]
	v_dual_mov_b32 v107, v112 :: v_dual_lshlrev_b32 v108, 16, v113
	v_mov_b64_e32 v[102:103], s[44:45]
	v_mov_b64_e32 v[100:101], s[42:43]
	v_mov_b32_e32 v103, s85
	v_or_b32_e32 v109, 0x80000000, v111
	v_readfirstlane_b32 s4, v98
	v_dual_mov_b32 v100, v110 :: v_dual_mov_b32 v98, v108
	v_readfirstlane_b32 s12, v106
	v_readfirstlane_b32 s5, v99
	v_readfirstlane_b32 s13, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_readfirstlane_b32 s14, v100
	v_readfirstlane_b32 s6, v98
	v_readfirstlane_b32 s15, v109
	v_readfirstlane_b32 s7, v101
	v_readfirstlane_b32 s8, v102
	v_readfirstlane_b32 s9, v103
	v_readfirstlane_b32 s10, v104
	v_readfirstlane_b32 s11, v105
	v_add_nc_u32_e32 v112, 0xffffff60, v1
	v_mov_b64_e32 v[108:109], s[2:3]
	v_mov_b64_e32 v[104:105], s[46:47]
	v_mov_b64_e32 v[106:107], s[0:1]
	tensor_load_to_lds s[12:15], s[4:11]
	s_mov_b32 s1, 0x13200
	v_mov_b64_e32 v[102:103], s[44:45]
	v_mov_b64_e32 v[100:101], s[42:43]
	v_mov_b64_e32 v[98:99], s[40:41]
	v_add_nc_u64_e32 v[110:111], s[78:79], v[110:111]
	s_barrier_signal -1
	v_mov_b32_e32 v103, s85
	s_set_vgpr_msb 8
	v_mad_u32_u24 v100, 0xc80, v75 /*v587*/, s1
	v_med3_i32 v107, v112, 0, 8
	v_readfirstlane_b32 s4, v98
	v_readfirstlane_b32 s5, v99
	v_readfirstlane_b32 s7, v101
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v101, 0x80000000, v111
	v_dual_mov_b32 v99, v100 :: v_dual_lshlrev_b32 v98, 16, v107
	v_mov_b32_e32 v100, v110
	v_readfirstlane_b32 s12, v106
	v_readfirstlane_b32 s8, v102
	v_readfirstlane_b32 s9, v103
	v_readfirstlane_b32 s10, v104
	v_readfirstlane_b32 s11, v105
	v_readfirstlane_b32 s13, v99
	v_readfirstlane_b32 s6, v98
	v_readfirstlane_b32 s14, v100
	v_readfirstlane_b32 s15, v101
	s_barrier_wait -1
	v_mov_b64_e32 v[108:109], s[2:3]
	v_mov_b64_e32 v[104:105], s[46:47]
	v_mov_b64_e32 v[106:107], s[0:1]
	s_mov_b32 s1, 0x16400
	v_mov_b64_e32 v[100:101], s[42:43]
	v_mov_b64_e32 v[98:99], s[40:41]
	s_set_vgpr_msb 8
	v_mad_u32_u24 v100, 0xc80, v75 /*v587*/, s1
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v112, 0xffffff40, v1
	v_add_nc_u64_e32 v[110:111], s[78:79], v[110:111]
	v_mov_b64_e32 v[102:103], s[44:45]
	v_mov_b32_e32 v103, s85
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	v_readfirstlane_b32 s5, v99
	v_mov_b32_e32 v99, v100
	v_med3_i32 v107, v112, 0, 8
	v_readfirstlane_b32 s4, v98
	v_readfirstlane_b32 s7, v101
	v_or_b32_e32 v101, 0x80000000, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v100, v110 :: v_dual_lshlrev_b32 v98, 16, v107
	v_readfirstlane_b32 s12, v106
	v_readfirstlane_b32 s8, v102
	v_readfirstlane_b32 s9, v103
	v_readfirstlane_b32 s10, v104
	v_readfirstlane_b32 s11, v105
	v_readfirstlane_b32 s13, v99
	v_readfirstlane_b32 s6, v98
	v_readfirstlane_b32 s14, v100
	v_readfirstlane_b32 s15, v101
	s_barrier_wait -1
	v_mov_b64_e32 v[108:109], s[2:3]
	v_mov_b64_e32 v[106:107], s[0:1]
	v_mov_b64_e32 v[104:105], s[46:47]
	s_mov_b32 s1, 0x19600
	v_mov_b64_e32 v[100:101], s[42:43]
	v_mov_b64_e32 v[98:99], s[40:41]
	s_set_vgpr_msb 8
	v_mad_u32_u24 v100, 0xc80, v75 /*v587*/, s1
	v_mov_b64_e32 v[102:103], s[44:45]
	tensor_load_to_lds s[12:15], s[4:11]
	v_readfirstlane_b32 s12, v106
	s_set_vgpr_msb 0x800
	v_add_nc_u64_e32 v[106:107], s[78:79], v[110:111]
	v_add_nc_u32_e32 v1, 0xffffff20, v1
	s_barrier_signal -1
	v_readfirstlane_b32 s4, v98
	v_readfirstlane_b32 s5, v99
	v_readfirstlane_b32 s7, v101
	v_med3_i32 v1, v1, 0, 8
	v_mov_b32_e32 v103, s85
	v_or_b32_e32 v99, 0x80000000, v107
	v_readfirstlane_b32 s8, v102
	v_readfirstlane_b32 s10, v104
	v_dual_lshlrev_b32 v98, 16, v1 :: v_dual_mov_b32 v1, v100
	v_mov_b32_e32 v100, v106
	v_readfirstlane_b32 s9, v103
	v_readfirstlane_b32 s11, v105
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s6, v98
	v_readfirstlane_b32 s13, v1
	v_readfirstlane_b32 s14, v100
	v_readfirstlane_b32 s15, v99
	s_barrier_wait -1
	tensor_load_to_lds s[12:15], s[4:11]
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x41
	ds_load_b128 v[66:69] /*v[322:325]*/, v11 /*v267*/
	s_set_vgpr_msb 0x4100
	v_add_nc_u32_e32 v1, 0x10020, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[70:73] /*v[326:329]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10040, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[58:61] /*v[314:317]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10060, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[62:65] /*v[318:321]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10080, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[42:45] /*v[298:301]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x100a0, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[46:49] /*v[302:305]*/, v1
	s_set_vgpr_msb 0x4041
	ds_load_b128 v[18:21] /*v[274:277]*/, v10 /*v266*/
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4100
	v_add_nc_u32_e32 v1, 0x11920, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[22:25] /*v[278:281]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x11940, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[10:13] /*v[266:269]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x11960, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[14:17] /*v[270:273]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x11980, v255
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[242:245], v1
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0x119a0, v255
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[246:249], v1
	s_set_vgpr_msb 0x41
	ds_load_b128 v[50:53] /*v[306:309]*/, v26 /*v282*/
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4100
	v_add_nc_u32_e32 v1, 0x100e0, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[54:57] /*v[310:313]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10100, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[34:37] /*v[290:293]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10120, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[38:41] /*v[294:297]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10140, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[26:29] /*v[282:285]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x10160, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[30:33] /*v[286:289]*/, v1
	ds_load_b128 v[2:5] /*v[258:261]*/, v253
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x119e0, v255
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 64
	ds_load_b128 v[6:9] /*v[262:265]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0x11a00, v255
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[234:237], v1
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0x11a20, v255
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[238:241], v1
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0x11a40, v255
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[226:229], v1
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0x11a60, v255
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[230:233], v1
	s_lshr_b32 s1, s103, 7
	s_set_vgpr_msb 0xa0
	v_and_or_b32 v152 /*v664*/, v0, 7, v146 /*v658*/
	s_add_co_i32 s1, s1, s102
	v_lshlrev_b32_e32 v147 /*v659*/, 1, v0
	s_max_i32 s94, s1, 1
	s_wait_dscnt 0x0
	s_min_u32 s46, s94, s34
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_u32 s46, 2
	s_set_vgpr_msb 0xa000
	s_cbranch_scc1 .LBB0_6
	s_set_vgpr_msb 9
	v_readlane_b32 s37, v1 /*v257*/, 5
	v_mul_u32_u24_e32 v0, 0x120, v152 /*v664*/
	v_mov_b32_e32 v218, 0
	v_readlane_b32 s98, v0 /*v256*/, 15
	v_readlane_b32 s43, v0 /*v256*/, 17
	v_readlane_b32 s42, v0 /*v256*/, 18
	s_add_co_i32 s2, s37, 0x100
	s_add_co_i32 s3, s37, 0x80
	v_readlane_b32 s56, v1 /*v257*/, 6
	v_readlane_b32 s58, v1 /*v257*/, 9
	v_readlane_b32 s38, v0 /*v256*/, 23
	s_set_vgpr_msb 0x982
	v_and_or_b32 v153 /*v665*/, v147 /*v659*/, 16, v0
	v_dual_mov_b32 v71 /*v583*/, v70 /*v582*/ :: v_dual_mov_b32 v69 /*v581*/, v68 /*v580*/
	s_set_vgpr_msb 0x8200
	v_dual_mov_b32 v219, v218 :: v_dual_mov_b32 v220, v218
	v_dual_mov_b32 v221, v218 :: v_dual_mov_b32 v222, v218
	v_dual_mov_b32 v223, v218 :: v_dual_mov_b32 v224, v218
	v_dual_mov_b32 v225, v218 :: v_dual_mov_b32 v186, v218
	v_dual_mov_b32 v187, v218 :: v_dual_mov_b32 v188, v218
	v_dual_mov_b32 v189, v218 :: v_dual_mov_b32 v190, v218
	v_dual_mov_b32 v191, v218 :: v_dual_mov_b32 v192, v218
	v_dual_mov_b32 v193, v218 :: v_dual_mov_b32 v154, v218
	v_dual_mov_b32 v155, v218 :: v_dual_mov_b32 v156, v218
	v_dual_mov_b32 v157, v218 :: v_dual_mov_b32 v158, v218
	v_dual_mov_b32 v159, v218 :: v_dual_mov_b32 v160, v218
	v_dual_mov_b32 v161, v218 :: v_dual_mov_b32 v122, v218
	v_dual_mov_b32 v123, v218 :: v_dual_mov_b32 v124, v218
	v_dual_mov_b32 v125, v218 :: v_dual_mov_b32 v126, v218
	v_dual_mov_b32 v127, v218 :: v_dual_mov_b32 v128, v218
	v_dual_mov_b32 v129, v218 :: v_dual_mov_b32 v210, v218
	v_dual_mov_b32 v211, v218 :: v_dual_mov_b32 v212, v218
	v_dual_mov_b32 v213, v218 :: v_dual_mov_b32 v214, v218
	v_dual_mov_b32 v215, v218 :: v_dual_mov_b32 v216, v218
	v_dual_mov_b32 v217, v218 :: v_dual_mov_b32 v178, v218
	v_dual_mov_b32 v179, v218 :: v_dual_mov_b32 v180, v218
	v_dual_mov_b32 v181, v218 :: v_dual_mov_b32 v182, v218
	v_dual_mov_b32 v183, v218 :: v_dual_mov_b32 v184, v218
	v_dual_mov_b32 v185, v218 :: v_dual_mov_b32 v146, v218
	v_dual_mov_b32 v147, v218 :: v_dual_mov_b32 v148, v218
	v_dual_mov_b32 v149, v218 :: v_dual_mov_b32 v150, v218
	v_dual_mov_b32 v151, v218 :: v_dual_mov_b32 v152, v218
	v_dual_mov_b32 v153, v218 :: v_dual_mov_b32 v114, v218
	v_dual_mov_b32 v115, v218 :: v_dual_mov_b32 v116, v218
	v_dual_mov_b32 v117, v218 :: v_dual_mov_b32 v118, v218
	v_dual_mov_b32 v119, v218 :: v_dual_mov_b32 v120, v218
	v_dual_mov_b32 v121, v218 :: v_dual_mov_b32 v202, v218
	v_dual_mov_b32 v203, v218 :: v_dual_mov_b32 v204, v218
	v_dual_mov_b32 v205, v218 :: v_dual_mov_b32 v206, v218
	v_dual_mov_b32 v207, v218 :: v_dual_mov_b32 v208, v218
	v_dual_mov_b32 v209, v218 :: v_dual_mov_b32 v170, v218
	v_dual_mov_b32 v171, v218 :: v_dual_mov_b32 v172, v218
	v_dual_mov_b32 v173, v218 :: v_dual_mov_b32 v174, v218
	v_dual_mov_b32 v175, v218 :: v_dual_mov_b32 v176, v218
	v_dual_mov_b32 v177, v218 :: v_dual_mov_b32 v138, v218
	v_dual_mov_b32 v139, v218 :: v_dual_mov_b32 v140, v218
	v_dual_mov_b32 v141, v218 :: v_dual_mov_b32 v142, v218
	v_dual_mov_b32 v143, v218 :: v_dual_mov_b32 v144, v218
	v_dual_mov_b32 v145, v218 :: v_dual_mov_b32 v106, v218
	v_dual_mov_b32 v107, v218 :: v_dual_mov_b32 v108, v218
	v_dual_mov_b32 v109, v218 :: v_dual_mov_b32 v110, v218
	v_dual_mov_b32 v111, v218 :: v_dual_mov_b32 v112, v218
	v_dual_mov_b32 v113, v218 :: v_dual_mov_b32 v194, v218
	v_dual_mov_b32 v195, v218 :: v_dual_mov_b32 v196, v218
	v_dual_mov_b32 v197, v218 :: v_dual_mov_b32 v198, v218
	v_dual_mov_b32 v199, v218 :: v_dual_mov_b32 v200, v218
	v_dual_mov_b32 v201, v218 :: v_dual_mov_b32 v162, v218
	v_dual_mov_b32 v163, v218 :: v_dual_mov_b32 v164, v218
	v_dual_mov_b32 v165, v218 :: v_dual_mov_b32 v166, v218
	v_dual_mov_b32 v167, v218 :: v_dual_mov_b32 v168, v218
	v_dual_mov_b32 v169, v218 :: v_dual_mov_b32 v130, v218
	v_dual_mov_b32 v131, v218 :: v_dual_mov_b32 v132, v218
	v_dual_mov_b32 v133, v218 :: v_dual_mov_b32 v134, v218
	v_dual_mov_b32 v135, v218 :: v_dual_mov_b32 v136, v218
	v_dual_mov_b32 v137, v218 :: v_dual_mov_b32 v98, v218
	v_dual_mov_b32 v99, v218 :: v_dual_mov_b32 v100, v218
	v_dual_mov_b32 v101, v218 :: v_dual_mov_b32 v102, v218
	v_dual_mov_b32 v103, v218 :: v_dual_mov_b32 v104, v218
	v_mov_b32_e32 v105, v218
	s_set_vgpr_msb 1
	v_readlane_b32 s57, v1 /*v257*/, 7
	v_readlane_b32 s59, v1 /*v257*/, 10
	v_readlane_b32 s101, v1 /*v257*/, 4
	v_readlane_b32 s39, v0 /*v256*/, 24
	s_sub_co_i32 s1, s65, s98
	s_mul_i32 s2, s74, s2
	s_mul_i32 s3, s75, s3
	s_mov_b32 s96, 0x10000
	s_mov_b32 s30, 0
	s_lshl_b32 s36, s75, 7
	s_addk_co_i32 s1, 0xff60
	s_add_co_i32 s16, s43, s2
	s_add_co_i32 s18, s42, s3
	s_mov_b64 s[20:21], 1
	s_mov_b32 s2, 0x30000
	s_mov_b32 s3, s30
	s_mov_b32 s40, s96
	s_set_vgpr_msb 0x100
.LBB0_5:
	s_mov_b32 s29, s40
	s_mov_b32 s26, s33
	s_mov_b32 s40, s3
	s_mov_b32 s33, s2
	s_set_vgpr_msb 64
	v_add_nc_u32_e32 v164 /*v420*/, s29, v255
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x4088
	v_add_nc_u32_e32 v155 /*v667*/, s26, v153 /*v665*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8880
	v_add_nc_u32_e32 v154 /*v666*/, s40, v255
	s_add_co_i32 s2, s1, 32
	s_mov_b32 s31, s30
	s_set_vgpr_msb 0x8000
	v_med3_i32 v0, s2, 0, 8
	v_med3_i32 v1, s1, 0, 8
	s_sub_co_i32 s2, s1, 32
	s_mov_b32 s27, s25
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, s[30:31]
	s_set_vgpr_msb 0x4000
	v_med3_i32 v250, s2, 0, 8
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, s[26:27]
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v1, 16, v1
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, s[28:29]
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, s[24:25]
	v_dual_mov_b32 v115 /*v371*/, s93 :: v_dual_lshlrev_b32 v114 /*v370*/, 16, v0
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v0, 16, v250
	s_set_vgpr_msb 0x41
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[96:97] /*v[352:353]*/
	s_sub_co_i32 s2, s1, 64
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x4140
	v_mov_b32_e32 v108 /*v364*/, v1
	s_set_vgpr_msb 0x4000
	v_med3_i32 v1, s2, 0, 8
	s_add_co_i32 s2, s1, 0xffffffa0
	s_set_vgpr_msb 0x41
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[96:97] /*v[352:353]*/
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[96:97] /*v[352:353]*/
	s_mov_b32 s102, s30
	s_mov_b32 s103, s30
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[94:95] /*v[350:351]*/
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[92:93] /*v[348:349]*/
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[90:91] /*v[346:347]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[94:95] /*v[350:351]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[90:91] /*v[346:347]*/
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[94:95] /*v[350:351]*/
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[92:93] /*v[348:349]*/
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[90:91] /*v[346:347]*/
	s_set_vgpr_msb 0x4140
	v_mov_b32_e32 v92 /*v348*/, v0
	s_set_vgpr_msb 0x4000
	v_med3_i32 v0, s2, 0, 8
	s_mov_b32 s100, s28
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, s[102:103]
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, s[96:97]
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, s[98:99]
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v251, s93 :: v_dual_lshlrev_b32 v0, 16, v0
	s_set_vgpr_msb 0x41
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, s[100:101]
	s_add_co_i32 s27, s1, 0xffffff80
	s_add_co_i32 s2, s1, 0xffffff60
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[132:133] /*v[388:389]*/
	s_addk_co_i32 s1, 0xff40
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[134:135] /*v[390:391]*/
	s_set_vgpr_msb 0x4100
	v_dual_mov_b32 v253, s93 :: v_dual_lshlrev_b32 v250, 16, v1
	v_med3_i32 v1, s27, 0, 8
	s_set_vgpr_msb 64
	v_med3_i32 v82 /*v338*/, s2, 0, 8
	v_mov_b32_e32 v76 /*v332*/, v0
	s_set_vgpr_msb 0x4000
	v_med3_i32 v0, s1, 0, 8
	s_set_vgpr_msb 0x41
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[136:137] /*v[392:393]*/
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[132:133] /*v[388:389]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[138:139] /*v[394:395]*/
	v_mov_b32_e32 v111 /*v367*/, s93
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[138:139] /*v[394:395]*/
	v_dual_mov_b32 v79 /*v335*/, s85 :: v_dual_lshlrev_b32 v100 /*v356*/, 16, v1
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[134:135] /*v[390:391]*/
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[136:137] /*v[392:393]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[138:139] /*v[394:395]*/
	s_set_vgpr_msb 0x4145
	v_dual_mov_b32 v95 /*v351*/, s85 :: v_dual_lshlrev_b32 v118 /*v374*/, 16, v82 /*v338*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[136:137] /*v[392:393]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[134:135] /*v[390:391]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x4541
	v_dual_mov_b32 v103 /*v359*/, s85 :: v_dual_lshlrev_b32 v134 /*v390*/, 16, v0
	v_mov_b32_e32 v121 /*v377*/, s85
	s_wait_dscnt 0xc
	s_ashr_i32 s17, s16, 31
	s_add_co_i32 s1, s40, s104
	s_add_nc_u64 s[2:3], s[38:39], s[16:17]
	s_add_co_i32 s17, s1, 0x6400
	s_mov_b64 s[14:15], s[2:3]
	s_or_b32 s4, s3, 0x80000000
	s_mov_b64 s[12:13], s[0:1]
	s_add_nc_u64 s[2:3], s[2:3], s[78:79]
	s_mov_b32 s15, s4
	s_mov_b64 s[10:11], s[2:3]
	s_add_co_i32 s4, s1, 0x3200
	s_or_b32 s5, s3, 0x80000000
	s_mov_b64 s[8:9], s[0:1]
	s_mov_b32 s10, s2
	s_add_nc_u64 s[2:3], s[2:3], s[78:79]
	s_mov_b32 s9, s4
	s_mov_b32 s11, s5
	s_mov_b64 s[6:7], s[2:3]
	s_or_b32 s19, s3, 0x80000000
	s_mov_b64 s[4:5], s[0:1]
	s_mov_b32 s6, s2
	s_add_nc_u64 s[2:3], s[2:3], s[78:79]
	s_add_co_i32 s1, s1, 0x9600
	s_bitset1_b32 s3, 31
	s_mov_b32 s5, s17
	s_mov_b32 s7, s19
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[66:73] /*v[322:329]*/, v[66:73], 0
	v_readfirstlane_b32 s48, v74 /*v330*/
	v_readfirstlane_b32 s49, v75 /*v331*/
	v_readfirstlane_b32 s50, v76 /*v332*/
	v_readfirstlane_b32 s51, v77 /*v333*/
	v_readfirstlane_b32 s52, v78 /*v334*/
	v_readfirstlane_b32 s53, v79 /*v335*/
	v_readfirstlane_b32 s54, v80 /*v336*/
	v_readfirstlane_b32 s55, v81 /*v337*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[12:15], s[48:55]
	v_readfirstlane_b32 s48, v124 /*v380*/
	v_readfirstlane_b32 s49, v125 /*v381*/
	v_readfirstlane_b32 s50, v100 /*v356*/
	v_readfirstlane_b32 s51, v127 /*v383*/
	v_readfirstlane_b32 s52, v128 /*v384*/
	v_readfirstlane_b32 s53, v95 /*v351*/
	v_readfirstlane_b32 s54, v130 /*v386*/
	v_readfirstlane_b32 s55, v131 /*v387*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[8:11], s[48:55]
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v168 /*v424*/, v136 /*v648*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[66:73] /*v[322:329]*/, v[90:97], 0
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[66:69] /*v[322:325]*/, v164 /*v420*/ offset:12800
	ds_load_b128 v[70:73] /*v[326:329]*/, v164 /*v420*/ offset:12832
	ds_load_b128 v[148:151] /*v[404:407]*/, v164 /*v420*/ offset:12864
	s_set_vgpr_msb 0x4108
	v_pk_mul_f32 v[224:225], v[224:225], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[58:65] /*v[314:321]*/, v[50:57], v[82:89] /*v[338:345]*/
	ds_load_b128 v[152:155] /*v[408:411]*/, v164 /*v420*/ offset:12896
	ds_load_b128 v[180:183] /*v[436:439]*/, v164 /*v420*/ offset:12928
	ds_load_b128 v[184:187] /*v[440:443]*/, v164 /*v420*/ offset:12960
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v169 /*v425*/, v137 /*v649*/
	v_exp_f32_e32 v170 /*v426*/, v122 /*v634*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[58:65] /*v[314:321]*/, v[82:89], v[74:81] /*v[330:337]*/
	ds_load_b128 v[204:207] /*v[460:463]*/, v164 /*v420*/ offset:19200
	ds_load_b128 v[208:211] /*v[464:467]*/, v164 /*v420*/ offset:19232
	ds_load_b128 v[212:215] /*v[468:471]*/, v164 /*v420*/ offset:19264
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v188 /*v444*/, v144 /*v656*/
	v_exp_f32_e32 v189 /*v445*/, v145 /*v657*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[42:49] /*v[298:305]*/, v[34:41], v[82:89] /*v[338:345]*/
	ds_load_b128 v[216:219] /*v[472:475]*/, v164 /*v420*/ offset:19296
	ds_load_b128 v[58:61] /*v[314:317]*/, v164 /*v420*/ offset:19328
	ds_load_b128 v[62:65] /*v[318:321]*/, v164 /*v420*/ offset:19360
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v190 /*v446*/, v140 /*v652*/
	v_exp_f32_e32 v191 /*v447*/, v141 /*v653*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[74:81], v[74:81] /*v[330:337]*/
	s_wait_dscnt 0xc
	ds_load_b128 v[220:223] /*v[476:479]*/, v164 /*v420*/ offset:12992
	ds_load_b128 v[224:227] /*v[480:483]*/, v164 /*v420*/ offset:13024
	ds_load_b128 v[244:247] /*v[500:503]*/, v164 /*v420*/ offset:13056
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v194 /*v450*/, v138 /*v650*/
	v_exp_f32_e32 v195 /*v451*/, v139 /*v651*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[50:57] /*v[306:313]*/, v[42:49], v[82:89] /*v[338:345]*/
	ds_load_b128 v[248:251] /*v[504:507]*/, v164 /*v420*/ offset:13088
	ds_load_b128 v[252:255] /*v[508:511]*/, v164 /*v420*/ offset:13120
	s_set_vgpr_msb 0x5181
	ds_load_b128 v[0:3] /*v[512:515]*/, v164 /*v420*/ offset:13152
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v230 /*v486*/, v118 /*v630*/
	v_exp_f32_e32 v171 /*v427*/, v123 /*v635*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[50:57] /*v[306:313]*/, v[58:65], v[74:81] /*v[330:337]*/
	ds_load_b128 v[124:127] /*v[380:383]*/, v164 /*v420*/ offset:19392
	ds_load_b128 v[128:131] /*v[384:387]*/, v164 /*v420*/ offset:19424
	ds_load_b128 v[50:53] /*v[306:309]*/, v164 /*v420*/ offset:19456
	s_set_vgpr_msb 0x5102
	v_exp_f32_e32 v0, v142 /*v654*/
	v_exp_f32_e32 v1, v143 /*v655*/
	s_set_vgpr_msb 0x251
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[34:41] /*v[290:297]*/, v[10:17], v[82:89] /*v[338:345]*/
	ds_load_b128 v[54:57] /*v[310:313]*/, v164 /*v420*/ offset:19488
	ds_load_b128 v[42:45] /*v[298:301]*/, v164 /*v420*/ offset:19520
	ds_load_b128 v[46:49] /*v[302:305]*/, v164 /*v420*/ offset:19552
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v228 /*v484*/, v120 /*v632*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[34:41] /*v[290:297]*/, v[26:33], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v200 /*v456*/, v130 /*v642*/
	v_exp_f32_e32 v201 /*v457*/, v131 /*v643*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[26:33] /*v[282:289]*/, v[2:9], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v202 /*v458*/, v128 /*v640*/
	v_exp_f32_e32 v203 /*v459*/, v129 /*v641*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[26:33] /*v[282:289]*/, v[18:25], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v192 /*v448*/, v134 /*v646*/
	v_exp_f32_e32 v193 /*v449*/, v135 /*v647*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[66:73], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v122 /*v634*/, v132 /*v644*/
	v_exp_f32_e32 v123 /*v635*/, v133 /*v645*/
	s_set_vgpr_msb 0x8241
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[18:25] /*v[274:281]*/, v[90:97], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v126 /*v638*/, v126 /*v638*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[216:217], v[216:217], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215], v[214:215], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213], v[212:213], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[10:17] /*v[266:273]*/, v[50:57], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v127 /*v639*/, v127 /*v639*/
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v232 /*v488*/, v116 /*v628*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[10:17] /*v[266:273]*/, v[82:89], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v231 /*v487*/, v119 /*v631*/
	v_exp_f32_e32 v229 /*v485*/, v121 /*v633*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[242:249], v[34:41], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v238 /*v494*/, v124 /*v636*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[208:209], v[208:209], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[242:249], v[74:81], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v239 /*v495*/, v125 /*v637*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[200:201], v[200:201], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199], v[198:199], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[196:197], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195], v[194:195], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[42:49], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v233 /*v489*/, v117 /*v629*/
	v_exp_f32_e32 v234 /*v490*/, v114 /*v626*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[58:65], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v235 /*v491*/, v115 /*v627*/
	v_exp_f32_e32 v236 /*v492*/, v112 /*v624*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[234:241], v[10:17], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v237 /*v493*/, v113 /*v625*/
	v_exp_f32_e32 v240 /*v496*/, v110 /*v622*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[234:241], v[26:33], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v241 /*v497*/, v111 /*v623*/
	v_exp_f32_e32 v242 /*v498*/, v108 /*v620*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[226:233], v[2:9], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v243 /*v499*/, v109 /*v621*/
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v106 /*v618*/, v106 /*v618*/
	s_set_vgpr_msb 0x8250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[226:233], v[18:25], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5082
	v_exp_f32_e32 v107 /*v619*/, v107 /*v619*/
	s_wait_dscnt 0xc
	v_exp_f32_e32 v104 /*v616*/, v104 /*v616*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[234:241], v[66:73] /*v[322:329]*/, v[66:73], 0
	v_readfirstlane_b32 s8, v140 /*v396*/
	v_readfirstlane_b32 s9, v141 /*v397*/
	v_readfirstlane_b32 s10, v118 /*v374*/
	v_readfirstlane_b32 s11, v143 /*v399*/
	v_readfirstlane_b32 s12, v144 /*v400*/
	v_readfirstlane_b32 s13, v103 /*v359*/
	v_readfirstlane_b32 s14, v146 /*v402*/
	v_readfirstlane_b32 s15, v147 /*v403*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[4:7], s[8:15]
	v_readfirstlane_b32 s4, v132 /*v388*/
	v_readfirstlane_b32 s5, v133 /*v389*/
	v_readfirstlane_b32 s6, v134 /*v390*/
	v_readfirstlane_b32 s7, v135 /*v391*/
	v_readfirstlane_b32 s8, v136 /*v392*/
	v_readfirstlane_b32 s9, v121 /*v377*/
	v_readfirstlane_b32 s10, v138 /*v394*/
	v_readfirstlane_b32 s11, v139 /*v395*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[0:3], s[4:11]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v105 /*v617*/, v105 /*v617*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[226:233], v[66:73] /*v[322:329]*/, v[90:97], 0
	s_set_vgpr_msb 0x181
	ds_load_b128 v[128:131] /*v[640:643]*/, v164 /*v420*/ offset:25600
	ds_load_b128 v[132:135] /*v[644:647]*/, v164 /*v420*/ offset:25632
	ds_load_b128 v[136:139] /*v[648:651]*/, v164 /*v420*/ offset:25664
	s_set_vgpr_msb 0x8108
	v_pk_mul_f32 v[192:193], v[192:193], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191], v[190:191], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189], v[188:189], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187], v[186:187], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x801
	v_wmma_f32_16x16x32_bf16 v[234:241], v[148:155] /*v[404:411]*/, v[50:57], v[234:241]
	s_set_vgpr_msb 0x181
	ds_load_b128 v[140:143] /*v[652:655]*/, v164 /*v420*/ offset:25696
	ds_load_b128 v[156:159] /*v[668:671]*/, v164 /*v420*/ offset:25728
	ds_load_b128 v[160:163] /*v[672:675]*/, v164 /*v420*/ offset:25760
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v108 /*v620*/, v10 /*v522*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[226:233], v[148:155] /*v[404:411]*/, v[82:89], v[226:233]
	s_set_vgpr_msb 0x141
	ds_load_b128 v[172:175] /*v[428:431]*/, v164 /*v420*/ offset:32000
	ds_load_b128 v[176:179] /*v[432:435]*/, v164 /*v420*/ offset:32032
	ds_load_b128 v[156:159] /*v[412:415]*/, v164 /*v420*/ offset:32064
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v120 /*v632*/, v100 /*v612*/
	v_exp_f32_e32 v121 /*v633*/, v101 /*v613*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[234:241], v[180:187] /*v[436:443]*/, v[34:41], v[234:241]
	s_set_vgpr_msb 0x141
	ds_load_b128 v[160:163] /*v[416:419]*/, v164 /*v420*/ offset:32096
	ds_load_b128 v[66:69] /*v[322:325]*/, v164 /*v420*/ offset:32128
	ds_load_b128 v[70:73] /*v[326:329]*/, v164 /*v420*/ offset:32160
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v96 /*v608*/, v96 /*v608*/
	v_exp_f32_e32 v97 /*v609*/, v97 /*v609*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[226:233], v[180:187] /*v[436:443]*/, v[74:81], v[226:233]
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x181
	ds_load_b128 v[164:167] /*v[676:679]*/, v164 /*v420*/ offset:25792
	ds_load_b128 v[168:171] /*v[680:683]*/, v164 /*v420*/ offset:25824
	ds_load_b128 v[172:175] /*v[684:687]*/, v164 /*v420*/ offset:25856
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v114 /*v626*/, v102 /*v614*/
	v_exp_f32_e32 v115 /*v627*/, v103 /*v615*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[234:241], v[220:227] /*v[476:483]*/, v[42:49], v[234:241]
	s_set_vgpr_msb 0x181
	ds_load_b128 v[176:179] /*v[688:691]*/, v164 /*v420*/ offset:25888
	ds_load_b128 v[180:183] /*v[692:695]*/, v164 /*v420*/ offset:25920
	ds_load_b128 v[184:187] /*v[696:699]*/, v164 /*v420*/ offset:25952
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v118 /*v630*/, v98 /*v610*/
	v_exp_f32_e32 v119 /*v631*/, v99 /*v611*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[226:233], v[220:227] /*v[476:483]*/, v[58:65], v[226:233]
	s_set_vgpr_msb 0x141
	ds_load_b128 v[132:135] /*v[388:391]*/, v164 /*v420*/ offset:32192
	ds_load_b128 v[136:139] /*v[392:395]*/, v164 /*v420*/ offset:32224
	ds_load_b128 v[148:151] /*v[404:407]*/, v164 /*v420*/ offset:32256
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v88 /*v600*/, v88 /*v600*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[234:241], v[244:251] /*v[500:507]*/, v[10:17], v[234:241]
	s_set_vgpr_msb 0x141
	ds_load_b128 v[152:155] /*v[408:411]*/, v164 /*v420*/ offset:32288
	ds_load_b128 v[140:143] /*v[396:399]*/, v164 /*v420*/ offset:32320
	ds_load_b128 v[144:147] /*v[400:403]*/, v164 /*v420*/ offset:32352
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v89 /*v601*/, v89 /*v601*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[226:233], v[244:251] /*v[500:507]*/, v[26:33], v[226:233]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v94 /*v606*/, v94 /*v606*/
	v_exp_f32_e32 v95 /*v607*/, v95 /*v607*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[234:241], v[252:259] /*v[508:515]*/, v[2:9], v[234:241]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v92 /*v604*/, v92 /*v604*/
	v_exp_f32_e32 v93 /*v605*/, v93 /*v605*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[226:233], v[252:259] /*v[508:515]*/, v[18:25], v[226:233]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v90 /*v602*/, v90 /*v602*/
	v_exp_f32_e32 v91 /*v603*/, v91 /*v603*/
	s_set_vgpr_msb 0x8241
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[204:211] /*v[460:467]*/, v[66:73], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v86 /*v598*/, v86 /*v598*/
	v_exp_f32_e32 v87 /*v599*/, v87 /*v599*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[242:249], v[204:211] /*v[460:467]*/, v[90:97], 0
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v84 /*v596*/, v84 /*v596*/
	v_exp_f32_e32 v85 /*v597*/, v85 /*v597*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[212:219] /*v[468:475]*/, v[50:57], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v98 /*v610*/, v80 /*v592*/
	v_exp_f32_e32 v80 /*v592*/, v82 /*v594*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[242:249], v[212:219] /*v[468:475]*/, v[82:89], v[242:249]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v99 /*v611*/, v81 /*v593*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[184:185], v[184:185], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183], v[182:183], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181], v[180:181], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179], v[178:179], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[58:65] /*v[314:321]*/, v[34:41], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v81 /*v593*/, v83 /*v595*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[176:177], v[176:177], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175], v[174:175], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[172:173], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171], v[170:171], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x801
	v_wmma_f32_16x16x32_bf16 v[242:249], v[58:65] /*v[314:321]*/, v[74:81], v[242:249]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v100 /*v612*/, v66 /*v578*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[168:169], v[168:169], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[124:131] /*v[380:387]*/, v[42:49], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v101 /*v613*/, v67 /*v579*/
	v_exp_f32_e32 v112 /*v624*/, v60 /*v572*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[242:249], v[124:131] /*v[380:387]*/, v[58:65], v[242:249]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v113 /*v625*/, v61 /*v573*/
	v_exp_f32_e32 v116 /*v628*/, v22 /*v534*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[50:57] /*v[306:313]*/, v[10:17], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v78 /*v590*/, v78 /*v590*/
	v_exp_f32_e32 v79 /*v591*/, v79 /*v591*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[242:249], v[50:57] /*v[306:313]*/, v[26:33], v[242:249]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v102 /*v614*/, v62 /*v574*/
	v_exp_f32_e32 v103 /*v615*/, v63 /*v575*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[42:49] /*v[298:305]*/, v[2:9], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v82 /*v594*/, v26 /*v538*/
	v_exp_f32_e32 v83 /*v595*/, v27 /*v539*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[242:249], v[42:49] /*v[298:305]*/, v[18:25], v[242:249]
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v110 /*v622*/, v24 /*v536*/
	s_wait_dscnt 0xc
	v_exp_f32_e32 v109 /*v621*/, v11 /*v523*/
	s_set_vgpr_msb 0x8242
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[128:135] /*v[640:647]*/, v[66:73], 0
	s_set_vgpr_msb 0x4281
	ds_load_b128 v[188:191] /*v[700:703]*/, v164 /*v420*/ offset:38400
	ds_load_b128 v[192:195] /*v[704:707]*/, v164 /*v420*/ offset:38432
	ds_load_b128 v[196:199] /*v[708:711]*/, v164 /*v420*/ offset:38464
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v124 /*v636*/, v20 /*v532*/
	v_exp_f32_e32 v111 /*v623*/, v25 /*v537*/
	s_set_vgpr_msb 0x8242
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[128:135] /*v[640:647]*/, v[90:97], 0
	s_set_vgpr_msb 0x4281
	ds_load_b128 v[200:203] /*v[712:715]*/, v164 /*v420*/ offset:38496
	ds_load_b128 v[4:7] /*v[516:519]*/, v164 /*v420*/ offset:38528
	ds_load_b128 v[8:11] /*v[520:523]*/, v164 /*v420*/ offset:38560
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v125 /*v637*/, v21 /*v533*/
	v_exp_f32_e32 v117 /*v629*/, v23 /*v535*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[136:143] /*v[648:655]*/, v[50:57], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5281
	ds_load_b128 v[20:23] /*v[532:535]*/, v164 /*v420*/ offset:44800
	ds_load_b128 v[24:27] /*v[536:539]*/, v164 /*v420*/ offset:44832
	s_set_vgpr_msb 0x8141
	ds_load_b128 v[252:255] /*v[508:511]*/, v164 /*v420*/ offset:44864
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v50 /*v562*/, v50 /*v562*/
	v_exp_f32_e32 v51 /*v563*/, v51 /*v563*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[136:143] /*v[648:655]*/, v[82:89], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5281
	ds_load_b128 v[0:3] /*v[512:515]*/, v164 /*v420*/ offset:44896
	s_set_vgpr_msb 0x8141
	ds_load_b128 v[220:223] /*v[476:479]*/, v164 /*v420*/ offset:44928
	ds_load_b128 v[224:227] /*v[480:483]*/, v164 /*v420*/ offset:44960
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v49 /*v561*/, v49 /*v561*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[156:163] /*v[668:675]*/, v[34:41], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5281
	ds_load_b128 v[128:131] /*v[640:643]*/, v164 /*v420*/ offset:38592
	ds_load_b128 v[132:135] /*v[644:647]*/, v164 /*v420*/ offset:38624
	ds_load_b128 v[60:63] /*v[572:575]*/, v164 /*v420*/ offset:38656
	v_nop
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v136 /*v648*/, v64 /*v576*/
	v_exp_f32_e32 v137 /*v649*/, v65 /*v577*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[156:163] /*v[668:675]*/, v[74:81], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0xf
	s_set_vgpr_msb 0x5281
	ds_load_b128 v[64:67] /*v[576:579]*/, v164 /*v420*/ offset:38688
	s_set_vgpr_msb 0x8141
	ds_load_b128 v[180:183] /*v[436:439]*/, v164 /*v420*/ offset:38720
	ds_load_b128 v[184:187] /*v[440:443]*/, v164 /*v420*/ offset:38752
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v57 /*v569*/, v57 /*v569*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[164:171] /*v[676:683]*/, v[42:49], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5241
	ds_load_b128 v[244:247] /*v[500:503]*/, v164 /*v420*/ offset:44992
	ds_load_b128 v[248:251] /*v[504:507]*/, v164 /*v420*/ offset:45024
	ds_load_b128 v[212:215] /*v[468:471]*/, v164 /*v420*/ offset:45056
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v76 /*v588*/, v76 /*v588*/
	v_exp_f32_e32 v77 /*v589*/, v77 /*v589*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[164:171] /*v[676:683]*/, v[58:65], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5241
	ds_load_b128 v[216:219] /*v[472:475]*/, v164 /*v420*/ offset:45088
	ds_load_b128 v[204:207] /*v[460:463]*/, v164 /*v420*/ offset:45120
	ds_load_b128 v[208:211] /*v[464:467]*/, v164 /*v420*/ offset:45152
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v138 /*v650*/, v58 /*v570*/
	v_exp_f32_e32 v139 /*v651*/, v59 /*v571*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[172:179] /*v[684:691]*/, v[10:17], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x524a
	v_cvt_pk_bf16_f32 v196 /*v452*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v54 /*v566*/, v55 /*v567*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v164 /*v420*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v170 /*v426*/, v171 /*v427*/
	s_set_vgpr_msb 0x4552
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[172:179] /*v[684:691]*/, v[26:33], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5245
	v_cvt_pk_bf16_f32 v166 /*v422*/, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v124 /*v380*/, v242 /*v498*/, v243 /*v499*/
	s_set_vgpr_msb 0x4552
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[180:187] /*v[692:699]*/, v[2:9], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x524a
	v_cvt_pk_bf16_f32 v125 /*v381*/, v106 /*v618*/, v107 /*v619*/
	v_cvt_pk_bf16_f32 v126 /*v382*/, v108 /*v620*/, v109 /*v621*/
	v_cvt_pk_bf16_f32 v127 /*v383*/, v124 /*v636*/, v125 /*v637*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[168:169] /*v[424:425]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[140:141] /*v[652:653]*/, v[200:201] /*v[456:457]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x8552
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[180:187] /*v[692:699]*/, v[18:25], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x524a
	v_cvt_pk_bf16_f32 v200 /*v456*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v201 /*v457*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v203 /*v459*/, v46 /*v558*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v168 /*v424*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v192 /*v448*/, v193 /*v449*/
	s_set_vgpr_msb 0x4541
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[172:179] /*v[428:435]*/, v[66:73], 0
	s_set_vgpr_msb 0x414a
	v_cvt_pk_bf16_f32 v171 /*v427*/, v122 /*v634*/, v123 /*v635*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v126 /*v638*/, v127 /*v639*/
	v_cvt_pk_bf16_f32 v63 /*v319*/, v120 /*v632*/, v121 /*v633*/
	v_cvt_pk_bf16_f32 v64 /*v320*/, v96 /*v608*/, v97 /*v609*/
	v_cvt_pk_bf16_f32 v65 /*v321*/, v86 /*v598*/, v87 /*v599*/
	v_cvt_pk_bf16_f32 v128 /*v384*/, v84 /*v596*/, v85 /*v597*/
	v_cvt_pk_bf16_f32 v129 /*v385*/, v98 /*v610*/, v99 /*v611*/
	s_set_vgpr_msb 0x4a41
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[172:179] /*v[428:435]*/, v[90:97], 0
	s_set_vgpr_msb 0x414a
	v_cvt_pk_bf16_f32 v130 /*v386*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v131 /*v387*/, v48 /*v560*/, v49 /*v561*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[142:143] /*v[654:655]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[144:145] /*v[656:657]*/, v[188:189] /*v[444:445]*/, v[190:191] /*v[446:447]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[192:193] /*v[448:449]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[126:127] /*v[638:639]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[156:163] /*v[412:419]*/, v[50:57], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v188 /*v444*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v30 /*v542*/, v31 /*v543*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[160:161], v[160:161], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[156:163] /*v[412:419]*/, v[82:89], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v190 /*v446*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v34 /*v546*/, v35 /*v547*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[152:153], v[152:153], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[66:73] /*v[322:329]*/, v[34:41], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5145
	v_cvt_pk_bf16_f32 v172 /*v428*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v230 /*v486*/, v231 /*v487*/
	s_set_vgpr_msb 0x4508
	v_pk_mul_f32 v[144:145], v[144:145], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[66:73] /*v[322:329]*/, v[74:81], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v192 /*v448*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v18 /*v530*/, v19 /*v531*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[136:137], v[136:137], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[134:135], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[132:133], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[132:139] /*v[388:395]*/, v[42:49], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5145
	v_cvt_pk_bf16_f32 v174 /*v430*/, v238 /*v494*/, v239 /*v495*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v104 /*v616*/, v105 /*v617*/
	v_cvt_pk_bf16_f32 v66 /*v322*/, v114 /*v626*/, v115 /*v627*/
	v_cvt_pk_bf16_f32 v67 /*v323*/, v118 /*v630*/, v119 /*v631*/
	v_cvt_pk_bf16_f32 v68 /*v324*/, v80 /*v592*/, v81 /*v593*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v100 /*v612*/, v101 /*v613*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[132:139] /*v[388:395]*/, v[58:65], v[42:49] /*v[298:305]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v132 /*v388*/, v112 /*v624*/, v113 /*v625*/
	v_cvt_pk_bf16_f32 v133 /*v389*/, v116 /*v628*/, v117 /*v629*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v136 /*v648*/, v137 /*v649*/
	v_cvt_pk_bf16_f32 v135 /*v391*/, v56 /*v568*/, v57 /*v569*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[14:15] /*v[526:527]*/, v[30:31] /*v[542:543]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[32:33] /*v[544:545]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[148:155] /*v[404:411]*/, v[10:17], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[194:195] /*v[450:451]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[238:239] /*v[494:495]*/, v[104:105] /*v[616:617]*/
	s_set_vgpr_msb 0x498a
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[114:115] /*v[626:627]*/, v[118:119] /*v[630:631]*/
	s_set_vgpr_msb 0x8a4a
	v_cvt_pk_bf16_f32 v194 /*v450*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v28 /*v540*/, v29 /*v541*/
	s_set_vgpr_msb 0x4a40
	v_cvt_pk_bf16_f32 v176 /*v432*/, v0, v1
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[148:155] /*v[404:411]*/, v[26:33], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5145
	v_cvt_pk_bf16_f32 v177 /*v433*/, v228 /*v484*/, v229 /*v485*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v178 /*v434*/, v88 /*v600*/, v89 /*v601*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v94 /*v606*/, v95 /*v607*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v92 /*v604*/, v93 /*v605*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v90 /*v602*/, v91 /*v603*/
	v_cvt_pk_bf16_f32 v72 /*v328*/, v78 /*v590*/, v79 /*v591*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[140:147] /*v[396:403]*/, v[2:9], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v73 /*v329*/, v102 /*v614*/, v103 /*v615*/
	v_cvt_pk_bf16_f32 v136 /*v392*/, v82 /*v594*/, v83 /*v595*/
	v_cvt_pk_bf16_f32 v137 /*v393*/, v110 /*v622*/, v111 /*v623*/
	v_cvt_pk_bf16_f32 v138 /*v394*/, v76 /*v588*/, v77 /*v589*/
	v_cvt_pk_bf16_f32 v139 /*v395*/, v138 /*v650*/, v139 /*v651*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[12:13] /*v[524:525]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[140:147] /*v[396:403]*/, v[18:25], v[42:49] /*v[298:305]*/
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[232:233] /*v[488:489]*/, v[234:235] /*v[490:491]*/
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[236:237] /*v[492:493]*/, v[240:241] /*v[496:497]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[242:243] /*v[498:499]*/, v[106:107] /*v[618:619]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[108:109] /*v[620:621]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[16:17] /*v[528:529]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x8a04
	v_pk_add_f32 v[0:1], v[0:1], v[228:229] /*v[484:485]*/
	s_barrier_wait -1
	s_set_vgpr_msb 0x442
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[188:195] /*v[700:707]*/, v[66:73], 0
	s_set_vgpr_msb 0x428a
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v155 /*v667*/
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v155 /*v667*/ offset:4608
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[58:59] /*v[570:571]*/, v[52:53] /*v[564:565]*/
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[54:55] /*v[566:567]*/, v[140:141] /*v[652:653]*/
	s_set_vgpr_msb 0x8a42
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[188:195] /*v[700:707]*/, v[90:97], 0
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v155 /*v667*/ offset:32
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v155 /*v667*/ offset:4640
	s_set_vgpr_msb 0x8285
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[230:231] /*v[486:487]*/, v[232:233] /*v[488:489]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[234:235] /*v[490:491]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8552
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[196:203] /*v[708:715]*/, v[50:57], v[148:155] /*v[404:411]*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v155 /*v667*/ offset:128
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v155 /*v667*/ offset:4736
	s_set_vgpr_msb 0x528a
	v_pk_add_f32 v[86:87] /*v[598:599]*/, v[96:97] /*v[608:609]*/, v[86:87] /*v[598:599]*/
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[196:203] /*v[708:715]*/, v[82:89], v[140:147] /*v[396:403]*/
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v155 /*v667*/ offset:160
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v155 /*v667*/ offset:4768
	s_set_vgpr_msb 0x528a
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[50:51] /*v[562:563]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[142:143] /*v[654:655]*/, v[44:45] /*v[556:557]*/
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[4:11] /*v[516:523]*/, v[34:41], v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x528a
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v155 /*v667*/ offset:64
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v155 /*v667*/ offset:4672
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[112:113] /*v[624:625]*/, v[116:117] /*v[628:629]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[136:137] /*v[648:649]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[4:11] /*v[516:523]*/, v[74:81], v[140:147] /*v[396:403]*/
	s_wait_dscnt 0xa
	s_set_vgpr_msb 0x5282
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v155 /*v667*/ offset:96
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v155 /*v667*/ offset:4704
	s_set_vgpr_msb 0x8285
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[156:157] /*v[412:413]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[136:137] /*v[648:649]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	s_set_vgpr_msb 0x8552
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[128:135] /*v[640:647]*/, v[42:49], v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x5282
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v155 /*v667*/ offset:192
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v155 /*v667*/ offset:4800
	s_set_vgpr_msb 0x824a
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[88:89] /*v[600:601]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[92:93] /*v[604:605]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[128:135] /*v[640:647]*/, v[58:65], v[140:147] /*v[396:403]*/
	s_set_vgpr_msb 0x528a
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v155 /*v667*/ offset:224
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v155 /*v667*/ offset:4832
	v_pk_add_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, v[102:103] /*v[614:615]*/
	s_set_vgpr_msb 0x8a4a
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[144:145] /*v[656:657]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x4a52
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[60:67] /*v[572:579]*/, v[10:17], v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x524a
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[120:121] /*v[632:633]*/, v[86:87] /*v[598:599]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[84:85] /*v[596:597]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[60:67] /*v[572:579]*/, v[26:33], v[140:147] /*v[396:403]*/
	s_set_vgpr_msb 0x5246
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[98:99] /*v[610:611]*/, v[158:159] /*v[414:415]*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x468a
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[108:109] /*v[620:621]*/, v[118:119] /*v[630:631]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[180:187] /*v[436:443]*/, v[2:9], v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	s_set_vgpr_msb 0x458a
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[76:77] /*v[588:589]*/, v[138:139] /*v[650:651]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[180:187] /*v[436:443]*/, v[18:25], v[140:147] /*v[396:403]*/
	s_set_vgpr_msb 0x5185
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[158:159] /*v[414:415]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[124:125] /*v[636:637]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[76:77] /*v[588:589]*/, v[114:115] /*v[626:627]*/, v[106:107] /*v[618:619]*/
	s_set_vgpr_msb 0x8a42
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[20:27] /*v[532:539]*/, v[66:73], 0
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[104:105] /*v[616:617]*/, v[80:81] /*v[592:593]*/
	s_set_vgpr_msb 0x8a04
	v_pk_add_f32 v[0:1], v[0:1], v[156:157] /*v[412:413]*/
	s_set_vgpr_msb 0x442
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[20:27] /*v[532:539]*/, v[90:97], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[100:101] /*v[612:613]*/, v[112:113] /*v[624:625]*/
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[116:117] /*v[628:629]*/, v[136:137] /*v[648:649]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[252:259] /*v[508:515]*/, v[50:57], v[180:187] /*v[436:443]*/
	s_set_vgpr_msb 0x518a
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[88:89] /*v[600:601]*/, v[78:79] /*v[590:591]*/
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[82:83] /*v[594:595]*/, v[62:63] /*v[574:575]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[252:259] /*v[508:515]*/, v[82:89], v[156:163] /*v[412:419]*/
	s_set_vgpr_msb 0x5102
	v_pk_add_f32 v[0:1], v[76:77] /*v[588:589]*/, v[0:1]
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x24a
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[24:25] /*v[536:537]*/, v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[220:227] /*v[476:483]*/, v[34:41], v[180:187] /*v[436:443]*/
	s_set_vgpr_msb 0x5108
	v_pk_mul_f32 v[128:129], v[128:129], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x84a
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[80:81] /*v[592:593]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[220:227] /*v[476:483]*/, v[74:81], v[156:163] /*v[412:419]*/
	s_set_vgpr_msb 0x5108
	v_pk_mul_f32 v[120:121], v[120:121], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x846
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[22:23] /*v[534:535]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x4651
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[244:251] /*v[500:507]*/, v[42:49], v[180:187] /*v[436:443]*/
	s_set_vgpr_msb 0x5108
	v_pk_mul_f32 v[112:113], v[112:113], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x84a
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[60:61] /*v[572:573]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[244:251] /*v[500:507]*/, v[58:65], v[156:163] /*v[412:419]*/
	s_set_vgpr_msb 0x5108
	v_pk_mul_f32 v[104:105], v[104:105], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x844
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[0:1], v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x4451
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[212:219] /*v[468:475]*/, v[10:17], v[180:187] /*v[436:443]*/
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[212:219] /*v[468:475]*/, v[26:33], v[156:163] /*v[412:419]*/
	s_set_vgpr_msb 0x5101
	v_mov_b32_e32 v0, v222 /*v478*/
	s_set_vgpr_msb 0x102
	v_mov_b32_e32 v1, v64 /*v576*/
	s_set_vgpr_msb 0x281
	v_mov_b32_e32 v64 /*v576*/, v223 /*v479*/
	s_set_vgpr_msb 0x8108
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[0:1], v[0:1], v[64:65] /*v[576:577]*/
	v_pk_add_f32 v[0:1], v[0:1], v[70:71] /*v[582:583]*/
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[204:211] /*v[460:467]*/, v[2:9], v[180:187] /*v[436:443]*/
	v_dual_mov_b32 v212 /*v468*/, v220 /*v476*/ :: v_dual_mov_b32 v213 /*v469*/, v224 /*v480*/
	v_mov_b32_e32 v224 /*v480*/, v221 /*v477*/
	s_set_vgpr_msb 0x5145
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x4546
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[68:69] /*v[580:581]*/, v[212:213] /*v[468:469]*/
	s_set_vgpr_msb 0x4651
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[204:211] /*v[460:467]*/, v[18:25], v[156:163] /*v[412:419]*/
	s_wait_dscnt 0xc
	s_ashr_i32 s19, s18, 31
	s_add_co_i32 s1, s33, s101
	s_add_nc_u64 s[2:3], s[58:59], s[18:19]
	s_add_co_i32 s17, s1, 0x4800
	s_mov_b64 s[14:15], s[2:3]
	s_or_b32 s4, s3, 0x80000000
	s_mov_b64 s[12:13], s[0:1]
	s_add_nc_u64 s[2:3], s[2:3], s[56:57]
	s_mov_b32 s15, s4
	s_mov_b64 s[10:11], s[2:3]
	s_add_co_i32 s4, s1, 0x2400
	s_or_b32 s5, s3, 0x80000000
	s_mov_b64 s[8:9], s[0:1]
	s_mov_b32 s10, s2
	s_add_nc_u64 s[2:3], s[2:3], s[56:57]
	s_mov_b32 s9, s4
	s_mov_b32 s11, s5
	s_mov_b64 s[6:7], s[2:3]
	s_or_b32 s19, s3, 0x80000000
	s_mov_b64 s[4:5], s[0:1]
	s_mov_b32 s6, s2
	s_add_nc_u64 s[2:3], s[2:3], s[56:57]
	s_addk_co_i32 s1, 0x6c00
	s_bitset1_b32 s3, 31
	s_mov_b32 s5, s17
	s_mov_b32 s7, s19
	s_set_vgpr_msb 0x5106
	v_wmma_f32_16x16x32_bf16 v[218:225], v[36:43] /*v[548:555]*/, v[196:203] /*v[452:459]*/, v[218:225]
	s_set_vgpr_msb 0x655
	v_readfirstlane_b32 s48, v116 /*v372*/
	v_readfirstlane_b32 s49, v117 /*v373*/
	v_readfirstlane_b32 s50, v114 /*v370*/
	v_readfirstlane_b32 s51, v119 /*v375*/
	v_readfirstlane_b32 s52, v120 /*v376*/
	v_readfirstlane_b32 s53, v115 /*v371*/
	v_readfirstlane_b32 s54, v122 /*v378*/
	v_readfirstlane_b32 s55, v123 /*v379*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[12:15], s[48:55]
	v_readfirstlane_b32 s48, v106 /*v362*/
	v_readfirstlane_b32 s49, v107 /*v363*/
	v_readfirstlane_b32 s50, v108 /*v364*/
	v_readfirstlane_b32 s51, v109 /*v365*/
	v_readfirstlane_b32 s52, v110 /*v366*/
	v_readfirstlane_b32 s53, v111 /*v367*/
	v_readfirstlane_b32 s54, v112 /*v368*/
	v_readfirstlane_b32 s55, v113 /*v369*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[8:11], s[48:55]
	v_max3_num_f32 v95 /*v351*/, v82 /*v338*/, v83 /*v339*/, v84 /*v340*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v100 /*v356*/, v234, v235, v236
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v103 /*v359*/, v18 /*v274*/, v19 /*v275*/, v20 /*v276*/
	s_set_vgpr_msb 0x5506
	v_wmma_f32_16x16x32_bf16 v[186:193], v[28:35] /*v[540:547]*/, v[196:203] /*v[452:459]*/, v[186:193]
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[106:109] /*v[362:365]*/, v155 /*v667*/ offset:9216
	ds_load_tr16_b128 v[110:113] /*v[366:369]*/, v155 /*v667*/ offset:13824
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v122 /*v378*/, v148 /*v404*/, v149 /*v405*/, v150 /*v406*/
	v_max3_num_f32 v95 /*v351*/, v85 /*v341*/, v86 /*v342*/, v95 /*v351*/
	s_set_vgpr_msb 0x5550
	v_max3_num_f32 v100 /*v356*/, v237, v238, v100 /*v356*/
	s_set_vgpr_msb 0x5006
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[44:51] /*v[556:563]*/, v[196:203] /*v[452:459]*/, v[154:161]
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[114:117] /*v[370:373]*/, v155 /*v667*/ offset:9248
	ds_load_tr16_b128 v[118:121] /*v[374:377]*/, v155 /*v667*/ offset:13856
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v103 /*v359*/, v21 /*v277*/, v22 /*v278*/, v103 /*v359*/
	v_max3_num_f32 v122 /*v378*/, v151 /*v407*/, v152 /*v408*/, v122 /*v378*/
	v_max3_num_f32 v95 /*v351*/, v87 /*v343*/, v88 /*v344*/, v95 /*v351*/
	s_set_vgpr_msb 0x5506
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[122:129], v[52:59] /*v[564:571]*/, v[196:203] /*v[452:459]*/, v[122:129]
	s_wait_dscnt 0x4
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v155 /*v667*/ offset:9344
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v155 /*v667*/ offset:13952
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v123 /*v379*/, v34 /*v290*/, v35 /*v291*/, v36 /*v292*/
	v_max3_num_f32 v222 /*v478*/, v2 /*v258*/, v3 /*v259*/, v4 /*v260*/
	v_max3_num_f32 v223 /*v479*/, v50 /*v306*/, v51 /*v307*/, v52 /*v308*/
	v_max3_num_f32 v224 /*v480*/, v180 /*v436*/, v181 /*v437*/, v182 /*v438*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[210:217], v[236:243] /*v[492:499]*/, v[196:203] /*v[452:459]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v155 /*v667*/ offset:9376
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v155 /*v667*/ offset:13984
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v123 /*v379*/, v37 /*v293*/, v38 /*v294*/, v123 /*v379*/
	v_max3_num_f32 v222 /*v478*/, v5 /*v261*/, v6 /*v262*/, v222 /*v478*/
	v_max3_num_f32 v223 /*v479*/, v53 /*v309*/, v54 /*v310*/, v223 /*v479*/
	v_max3_num_f32 v224 /*v480*/, v183 /*v439*/, v184 /*v440*/, v224 /*v480*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[178:185], v[228:235] /*v[484:491]*/, v[196:203] /*v[452:459]*/, v[178:185]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v155 /*v667*/ offset:9280
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v155 /*v667*/ offset:13888
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v225 /*v481*/, v74 /*v330*/, v75 /*v331*/, v76 /*v332*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v226 /*v482*/, v226, v227, v228
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v227 /*v483*/, v10 /*v266*/, v11 /*v267*/, v12 /*v268*/
	s_set_vgpr_msb 0x5595
	v_max3_num_f32 v20 /*v532*/, v140 /*v396*/, v141 /*v397*/, v142 /*v398*/
	s_set_vgpr_msb 0x9506
	v_wmma_f32_16x16x32_bf16 v[146:153], v[12:19] /*v[524:531]*/, v[196:203] /*v[452:459]*/, v[146:153]
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v155 /*v667*/ offset:9312
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v155 /*v667*/ offset:13920
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v225 /*v481*/, v77 /*v333*/, v78 /*v334*/, v225 /*v481*/
	s_set_vgpr_msb 0x5550
	v_max3_num_f32 v226 /*v482*/, v229, v230, v226 /*v482*/
	s_set_vgpr_msb 0x5055
	v_max3_num_f32 v227 /*v483*/, v13 /*v269*/, v14 /*v270*/, v227 /*v483*/
	s_set_vgpr_msb 0x55a5
	v_max3_num_f32 v60 /*v572*/, v143 /*v399*/, v144 /*v400*/, v20 /*v532*/
	s_set_vgpr_msb 0xa506
	v_wmma_f32_16x16x32_bf16 v[114:121], v[4:11] /*v[516:523]*/, v[196:203] /*v[452:459]*/, v[114:121]
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v155 /*v667*/ offset:9408
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v155 /*v667*/ offset:14016
	s_set_vgpr_msb 0x4295
	v_max3_num_f32 v61 /*v573*/, v26 /*v282*/, v27 /*v283*/, v28 /*v284*/
	s_set_vgpr_msb 0x9580
	v_max3_num_f32 v62 /*v574*/, v242, v243, v244
	s_set_vgpr_msb 0x8095
	v_max3_num_f32 v63 /*v575*/, v42 /*v298*/, v43 /*v299*/, v44 /*v300*/
	v_max3_num_f32 v64 /*v576*/, v156 /*v412*/, v157 /*v413*/, v158 /*v414*/
	s_set_vgpr_msb 0x9506
	v_wmma_f32_16x16x32_bf16 v[202:209], v[36:43] /*v[548:555]*/, v[188:195] /*v[444:451]*/, v[202:209]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x682
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v155 /*v667*/ offset:9440
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v155 /*v667*/ offset:14048
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x82a5
	v_max3_num_f32 v36 /*v548*/, v29 /*v285*/, v30 /*v286*/, v61 /*v573*/
	s_set_vgpr_msb 0xa5a0
	v_max3_num_f32 v37 /*v549*/, v245, v246, v62 /*v574*/
	s_set_vgpr_msb 0xa0a5
	v_max3_num_f32 v38 /*v550*/, v45 /*v301*/, v46 /*v302*/, v63 /*v575*/
	v_max3_num_f32 v39 /*v551*/, v159 /*v415*/, v160 /*v416*/, v64 /*v576*/
	s_set_vgpr_msb 0xa506
	v_wmma_f32_16x16x32_bf16 v[170:177], v[28:35] /*v[540:547]*/, v[188:195] /*v[444:451]*/, v[170:177]
	s_set_vgpr_msb 0x650
	v_max3_num_f32 v100 /*v356*/, v239, v240, v100 /*v356*/
	s_set_vgpr_msb 0x5055
	v_max3_num_f32 v225 /*v481*/, v79 /*v335*/, v80 /*v336*/, v225 /*v481*/
	s_set_vgpr_msb 0x5550
	v_max3_num_f32 v226 /*v482*/, v231, v232, v226 /*v482*/
	s_set_vgpr_msb 0x5055
	v_max3_num_f32 v227 /*v483*/, v15 /*v271*/, v16 /*v272*/, v227 /*v483*/
	s_set_vgpr_msb 0x55a5
	v_max3_num_f32 v28 /*v540*/, v31 /*v287*/, v32 /*v288*/, v36 /*v548*/
	s_set_vgpr_msb 0xa506
	v_wmma_f32_16x16x32_bf16 v[138:145], v[44:51] /*v[556:563]*/, v[188:195] /*v[444:451]*/, v[138:145]
	s_set_vgpr_msb 0x655
	v_max3_num_f32 v103 /*v359*/, v23 /*v279*/, v24 /*v280*/, v103 /*v359*/
	v_max3_num_f32 v123 /*v379*/, v39 /*v295*/, v40 /*v296*/, v123 /*v379*/
	v_max3_num_f32 v222 /*v478*/, v7 /*v263*/, v8 /*v264*/, v222 /*v478*/
	s_set_vgpr_msb 0x55a0
	v_max3_num_f32 v29 /*v541*/, v247, v248, v37 /*v549*/
	s_set_vgpr_msb 0xa006
	v_wmma_f32_16x16x32_bf16 v[106:113], v[52:59] /*v[564:571]*/, v[188:195] /*v[444:451]*/, v[106:113]
	s_set_vgpr_msb 0x655
	v_max3_num_f32 v122 /*v378*/, v153 /*v409*/, v154 /*v410*/, v122 /*v378*/
	v_max3_num_f32 v95 /*v351*/, v89 /*v345*/, v95 /*v351*/, v82 /*v338*/
	s_set_vgpr_msb 0x5544
	v_max3_num_f32 v100 /*v356*/, v241, v100 /*v356*/, v234
	s_set_vgpr_msb 0x4455
	v_max3_num_f32 v103 /*v359*/, v25 /*v281*/, v103 /*v359*/, v18 /*v274*/
	v_max3_num_f32 v223 /*v479*/, v55 /*v311*/, v56 /*v312*/, v223 /*v479*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[194:201], v[236:243] /*v[492:499]*/, v[188:195] /*v[444:451]*/, v[194:201]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v122 /*v378*/, v155 /*v411*/, v122 /*v378*/, v148 /*v404*/
	v_max3_num_f32 v224 /*v480*/, v185 /*v441*/, v186 /*v442*/, v224 /*v480*/
	v_max3_num_f32 v123 /*v379*/, v41 /*v297*/, v123 /*v379*/, v34 /*v290*/
	v_nop
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v236 /*v492*/, v145 /*v401*/, v146 /*v402*/, v60 /*v572*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v225 /*v481*/, v81 /*v337*/, v225 /*v481*/, v74 /*v330*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[162:169], v[228:235] /*v[484:491]*/, v[188:195] /*v[444:451]*/, v[162:169]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v222 /*v478*/, v9 /*v265*/, v222 /*v478*/, v2 /*v258*/
	s_set_vgpr_msb 0x5544
	v_max3_num_f32 v226 /*v482*/, v233, v226 /*v482*/, v226
	s_set_vgpr_msb 0x4455
	v_max3_num_f32 v227 /*v483*/, v17 /*v273*/, v227 /*v483*/, v10 /*v266*/
	v_nop
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v228 /*v484*/, v47 /*v303*/, v48 /*v304*/, v38 /*v550*/
	v_max3_num_f32 v229 /*v485*/, v161 /*v417*/, v162 /*v418*/, v39 /*v551*/
	s_set_vgpr_msb 0x6506
	v_wmma_f32_16x16x32_bf16 v[130:137], v[12:19] /*v[524:531]*/, v[188:195] /*v[444:451]*/, v[130:137]
	s_set_vgpr_msb 0x655
	v_max3_num_f32 v230 /*v486*/, v147 /*v403*/, v236 /*v492*/, v140 /*v396*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v231 /*v487*/, v33 /*v289*/, v28 /*v540*/, v26 /*v282*/
	s_set_vgpr_msb 0x5948
	v_max3_num_f32 v232 /*v488*/, v249, v29 /*v541*/, v242
	s_set_vgpr_msb 0x4855
	v_max3_num_f32 v223 /*v479*/, v57 /*v313*/, v223 /*v479*/, v50 /*v306*/
	v_max3_num_f32 v224 /*v480*/, v187 /*v443*/, v224 /*v480*/, v180 /*v436*/
	s_set_vgpr_msb 0x5506
	v_wmma_f32_16x16x32_bf16 v[98:105], v[4:11] /*v[516:523]*/, v[188:195] /*v[444:451]*/, v[98:105]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x655
	v_max3_num_f32 v188 /*v444*/, v49 /*v305*/, v228 /*v484*/, v42 /*v298*/
	s_wait_dscnt 0xc
	v_max3_num_f32 v189 /*v445*/, v163 /*v419*/, v229 /*v485*/, v156 /*v412*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[218:225], v[106:113] /*v[362:369]*/, v[164:171] /*v[420:427]*/, v[218:225]
	v_readfirstlane_b32 s8, v98 /*v354*/
	v_readfirstlane_b32 s9, v99 /*v355*/
	v_readfirstlane_b32 s10, v92 /*v348*/
	v_readfirstlane_b32 s11, v101 /*v357*/
	v_readfirstlane_b32 s12, v102 /*v358*/
	s_set_vgpr_msb 0x500
	v_readfirstlane_b32 s13, v253
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s14, v104 /*v360*/
	v_readfirstlane_b32 s15, v105 /*v361*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[4:7], s[8:15]
	v_readfirstlane_b32 s4, v90 /*v346*/
	v_readfirstlane_b32 s5, v91 /*v347*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s6, v250
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s7, v93 /*v349*/
	v_readfirstlane_b32 s8, v94 /*v350*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s9, v251
	s_set_vgpr_msb 21
	v_readfirstlane_b32 s10, v96 /*v352*/
	v_readfirstlane_b32 s11, v97 /*v353*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[0:3], s[4:11]
	v_max3_num_f32 v250, v95 /*v351*/, v100 /*v356*/, v103 /*v359*/
	s_set_vgpr_msb 0x1505
	v_wmma_f32_16x16x32_bf16 v[186:193], v[114:121] /*v[370:377]*/, v[164:171] /*v[420:427]*/, v[186:193]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[90:93] /*v[346:349]*/, v155 /*v667*/ offset:18432
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[94:97] /*v[350:353]*/, v155 /*v667*/ offset:23040
	s_set_vgpr_msb 0x4214
	v_max3_num_f32 v250, v250, v122 /*v378*/, v100 /*v356*/
	s_set_vgpr_msb 0x1415
	v_max3_num_f32 v251, v123 /*v379*/, v222 /*v478*/, v223 /*v479*/
	v_max3_num_f32 v253, v225 /*v481*/, v226 /*v482*/, v227 /*v483*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v122 /*v378*/, v231 /*v487*/, v232 /*v488*/, v188 /*v444*/
	s_set_vgpr_msb 0x5505
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[244:251] /*v[500:507]*/, v[164:171] /*v[420:427]*/, v[154:161]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[98:101] /*v[354:357]*/, v155 /*v667*/ offset:18464
	ds_load_tr16_b128 v[102:105] /*v[358:361]*/, v155 /*v667*/ offset:23072
	v_add_f32_e32 v123 /*v379*/, 0, v250
	s_set_vgpr_msb 0x4214
	v_max3_num_f32 v251, v251, v224 /*v480*/, v222 /*v478*/
	v_max3_num_f32 v253, v253, v230 /*v486*/, v226 /*v482*/
	s_set_vgpr_msb 0x1455
	v_max3_num_f32 v122 /*v378*/, v122 /*v378*/, v189 /*v445*/, v232 /*v488*/
	s_set_vgpr_msb 0x5505
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[122:129], v[252:259] /*v[508:515]*/, v[164:171] /*v[420:427]*/, v[122:129]
	s_wait_dscnt 0x4
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v155 /*v667*/ offset:18560
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v155 /*v667*/ offset:23168
	v_add_f32_e32 v238 /*v494*/, 0, v251
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v123 /*v379*/, v123 /*v379*/, s35, 0xfedcba98
	v_add_f32_e32 v239 /*v495*/, 0, v253
	s_set_vgpr_msb 0x4144
	v_add_f32_e32 v240 /*v496*/, 0, v122 /*v378*/
	s_set_vgpr_msb 0x4405
	v_wmma_f32_16x16x32_bf16 v[210:217], v[204:211] /*v[460:467]*/, v[164:171] /*v[420:427]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v155 /*v667*/ offset:18592
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v155 /*v667*/ offset:23200
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v238 /*v494*/, v238 /*v494*/, s35, 0xfedcba98
	v_mul_f32_e32 v241 /*v497*/, s66, v254
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[178:185], v[214:221] /*v[470:477]*/, v[164:171] /*v[420:427]*/, v[178:185]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v155 /*v667*/ offset:18496
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v155 /*v667*/ offset:23104
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v239 /*v495*/, v239 /*v495*/, s35, 0xfedcba98
	v_mul_f32_e32 v242 /*v498*/, s66, v252
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[146:153], v[196:203] /*v[452:459]*/, v[164:171] /*v[420:427]*/, v[146:153]
	s_set_vgpr_msb 0x582
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v155 /*v667*/ offset:18528
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v155 /*v667*/ offset:23136
	s_set_vgpr_msb 0x8204
	v_max3_num_f32 v253, v253, v239 /*v495*/, v252
	v_max3_num_f32 v250, v250, v123 /*v379*/, v254
	v_max3_num_f32 v251, v251, v238 /*v494*/, v254
	s_set_vgpr_msb 0x406
	v_wmma_f32_16x16x32_bf16 v[114:121], v[20:27] /*v[532:539]*/, v[164:171] /*v[420:427]*/, v[114:121]
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v155 /*v667*/ offset:18624
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v155 /*v667*/ offset:23232
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v240 /*v496*/, v240 /*v496*/, s35, 0xfedcba98
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[202:209], v[106:113] /*v[362:369]*/, v[172:179] /*v[428:435]*/, v[202:209]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[106:109] /*v[362:365]*/, v155 /*v667*/ offset:18656
	ds_load_tr16_b128 v[110:113] /*v[366:369]*/, v155 /*v667*/ offset:23264
	s_set_vgpr_msb 0x4205
	v_max3_num_f32 v252, v122 /*v378*/, v240 /*v496*/, v252
	v_wmma_f32_16x16x32_bf16 v[170:177], v[114:121] /*v[370:377]*/, v[172:179] /*v[428:435]*/, v[170:177]
	s_set_vgpr_msb 0x510
	v_max3_num_f32 v254, v250, v251, v241 /*v497*/
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v252, v253, v252, v242 /*v498*/
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[138:145], v[244:251] /*v[500:507]*/, v[172:179] /*v[428:435]*/, v[138:145]
	s_set_vgpr_msb 0x510
	v_fma_f32 v250, -v254, s66, v241 /*v497*/
	v_fma_f32 v251, -v252, s66, v242 /*v498*/
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[106:113], v[252:259] /*v[508:515]*/, v[172:179] /*v[428:435]*/, v[106:113]
	s_set_vgpr_msb 0x580
	v_exp_f32_e32 v74 /*v586*/, v250
	v_exp_f32_e32 v72 /*v584*/, v251
	s_set_vgpr_msb 0x8005
	v_wmma_f32_16x16x32_bf16 v[194:201], v[204:211] /*v[460:467]*/, v[172:179] /*v[428:435]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[214:221] /*v[470:477]*/, v[172:179] /*v[428:435]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[196:203] /*v[452:459]*/, v[172:179] /*v[428:435]*/, v[130:137]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[98:105], v[20:27] /*v[532:539]*/, v[172:179] /*v[428:435]*/, v[98:105]
	s_set_vgpr_msb 0x688
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[0:1], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x8889
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[212:213] /*v[468:469]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8905
	v_wmma_f32_16x16x32_bf16 v[218:225], v[90:97] /*v[346:353]*/, v[58:65] /*v[314:321]*/, v[218:225]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[114:117] /*v[370:373]*/, v155 /*v667*/ offset:27648
	ds_load_tr16_b128 v[118:121] /*v[374:377]*/, v155 /*v667*/ offset:32256
	s_set_vgpr_msb 0x4200
	v_mov_b64_e32 v[250:251], s[66:67]
	v_mul_f32_e64 v0, s66, -v254
	s_set_vgpr_msb 0x41
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[122:123] /*v[378:379]*/, v[82:83] /*v[338:339]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[84:85] /*v[340:341]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[86:87] /*v[342:343]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[186:193], v[98:105] /*v[354:361]*/, v[58:65] /*v[314:321]*/, v[186:193]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v155 /*v667*/ offset:27680
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v155 /*v667*/ offset:32288
	s_set_vgpr_msb 0x4241
	v_pk_fma_f32 v[248:249] /*v[504:505]*/, v[88:89] /*v[344:345]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4180
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[234:235], v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[122:123] /*v[634:635]*/, v[236:237], v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8005
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[154:161], v[230:237] /*v[486:493]*/, v[58:65] /*v[314:321]*/, v[154:161]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v155 /*v667*/ offset:27776
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v155 /*v667*/ offset:32384
	s_set_vgpr_msb 0x4201
	v_pk_fma_f32 v[234:235], v[34:35] /*v[290:291]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x140
	v_mul_f32_e64 v250 /*v506*/, s66, -v252
	s_set_vgpr_msb 0x4011
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[236:237], v[74:75] /*v[330:331]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[252:253] /*v[508:509]*/, v[76:77] /*v[332:333]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5106
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[122:129], v[12:19] /*v[524:531]*/, v[58:65] /*v[314:321]*/, v[122:129]
	s_wait_dscnt 0x6
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v155 /*v667*/ offset:27808
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v155 /*v667*/ offset:32416
	s_set_vgpr_msb 0x4241
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[36:37] /*v[292:293]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4151
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[78:79] /*v[334:335]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5191
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[80:81] /*v[336:337]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9105
	v_wmma_f32_16x16x32_bf16 v[210:217], v[188:195] /*v[444:451]*/, v[58:65] /*v[314:321]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v155 /*v667*/ offset:27712
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v155 /*v667*/ offset:32320
	s_set_vgpr_msb 0x4290
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[226:227], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9081
	v_pk_fma_f32 v[2:3] /*v[514:515]*/, v[38:39] /*v[294:295]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[40:41] /*v[296:297]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[178:185], v[222:229] /*v[478:485]*/, v[58:65] /*v[314:321]*/, v[178:185]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v155 /*v667*/ offset:27744
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v155 /*v667*/ offset:32352
	s_set_vgpr_msb 0x4290
	v_pk_fma_f32 v[118:119] /*v[630:631]*/, v[228:229], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9081
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[2:3] /*v[258:259]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[4:5] /*v[260:261]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[146:153], v[164:171] /*v[420:427]*/, v[58:65] /*v[314:321]*/, v[146:153]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v155 /*v667*/ offset:27840
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v155 /*v667*/ offset:32448
	s_set_vgpr_msb 0x4291
	v_pk_fma_f32 v[6:7] /*v[518:519]*/, v[26:27] /*v[282:283]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[28:29] /*v[284:285]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[28:29] /*v[540:541]*/, v[30:31] /*v[286:287]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[156:157] /*v[668:669]*/, v[32:33] /*v[288:289]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9105
	v_wmma_f32_16x16x32_bf16 v[114:121], v[106:113] /*v[362:369]*/, v[58:65] /*v[314:321]*/, v[114:121]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v155 /*v667*/ offset:27872
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v155 /*v667*/ offset:32480
	s_set_vgpr_msb 0x4290
	v_pk_fma_f32 v[142:143] /*v[654:655]*/, v[242:243], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[120:121] /*v[632:633]*/, v[244:245], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[246:247], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9005
	v_wmma_f32_16x16x32_bf16 v[202:209], v[90:97] /*v[346:353]*/, v[66:73] /*v[322:329]*/, v[202:209]
	s_set_vgpr_msb 0x580
	v_pk_fma_f32 v[130:131] /*v[642:643]*/, v[238:239], v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[128:129] /*v[640:641]*/, v[240:241], v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8081
	v_pk_fma_f32 v[116:117] /*v[628:629]*/, v[18:19] /*v[274:275]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[114:115] /*v[626:627]*/, v[20:21] /*v[276:277]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[112:113] /*v[624:625]*/, v[22:23] /*v[278:279]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[110:111] /*v[622:623]*/, v[24:25] /*v[280:281]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[170:177], v[98:105] /*v[354:361]*/, v[66:73] /*v[322:329]*/, v[170:177]
	s_set_vgpr_msb 0x581
	v_pk_fma_f32 v[108:109] /*v[620:621]*/, v[148:149] /*v[404:405]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[106:107] /*v[618:619]*/, v[150:151] /*v[406:407]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[10:11] /*v[522:523]*/, v[152:153] /*v[408:409]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[134:135] /*v[646:647]*/, v[6:7] /*v[262:263]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[138:145], v[230:237] /*v[486:493]*/, v[66:73] /*v[322:329]*/, v[138:145]
	s_set_vgpr_msb 0x581
	v_pk_fma_f32 v[132:133] /*v[644:645]*/, v[8:9] /*v[264:265]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[126:127] /*v[638:639]*/, v[50:51] /*v[306:307]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[100:101] /*v[612:613]*/, v[52:53] /*v[308:309]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[54:55] /*v[310:311]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[86:87] /*v[598:599]*/, v[56:57] /*v[312:313]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[84:85] /*v[596:597]*/, v[180:181] /*v[436:437]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8106
	v_wmma_f32_16x16x32_bf16 v[106:113], v[12:19] /*v[524:531]*/, v[66:73] /*v[322:329]*/, v[106:113]
	s_set_vgpr_msb 0x681
	v_pk_fma_f32 v[80:81] /*v[592:593]*/, v[182:183] /*v[438:439]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[50:51] /*v[562:563]*/, v[184:185] /*v[440:441]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[186:187] /*v[442:443]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8190
	v_pk_fma_f32 v[124:125] /*v[636:637]*/, v[230:231], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[104:105] /*v[616:617]*/, v[232:233], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9005
	v_wmma_f32_16x16x32_bf16 v[194:201], v[188:195] /*v[444:451]*/, v[66:73] /*v[322:329]*/, v[194:201]
	s_set_vgpr_msb 0x591
	v_pk_fma_f32 v[102:103] /*v[614:615]*/, v[10:11] /*v[266:267]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[98:99] /*v[610:611]*/, v[12:13] /*v[268:269]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[82:83] /*v[594:595]*/, v[14:15] /*v[270:271]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[16:17] /*v[272:273]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[60:61] /*v[572:573]*/, v[140:141] /*v[396:397]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[22:23] /*v[534:535]*/, v[142:143] /*v[398:399]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9105
	v_wmma_f32_16x16x32_bf16 v[162:169], v[222:229] /*v[478:485]*/, v[66:73] /*v[322:329]*/, v[162:169]
	s_set_vgpr_msb 0x590
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[248:249], v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9091
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[42:43] /*v[298:299]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[44:45] /*v[300:301]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9105
	v_wmma_f32_16x16x32_bf16 v[130:137], v[164:171] /*v[420:427]*/, v[66:73] /*v[322:329]*/, v[130:137]
	s_set_vgpr_msb 0x591
	v_pk_fma_f32 v[78:79] /*v[590:591]*/, v[46:47] /*v[302:303]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[48:49] /*v[304:305]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[26:27] /*v[538:539]*/, v[156:157] /*v[412:413]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[158:159] /*v[414:415]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9105
	v_wmma_f32_16x16x32_bf16 v[98:105], v[106:113] /*v[362:369]*/, v[66:73] /*v[322:329]*/, v[98:105]
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
	s_set_vgpr_msb 0x581
	v_pk_fma_f32 v[20:21] /*v[532:533]*/, v[154:155] /*v[410:411]*/, v[250:251], v[0:1] op_sel_hi:[1,1,0]
	s_wait_dscnt 0xc
	s_barrier_wait -1
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[218:225], v[114:121] /*v[370:377]*/, v[124:131] /*v[380:387]*/, v[218:225]
	s_set_vgpr_msb 0x542
	ds_load_b128 v[66:69] /*v[322:325]*/, v154 /*v666*/
	ds_load_b128 v[70:73] /*v[326:329]*/, v154 /*v666*/ offset:32
	ds_load_b128 v[58:61] /*v[314:317]*/, v154 /*v666*/ offset:64
	s_set_vgpr_msb 0x4281
	v_exp_f32_e32 v40 /*v552*/, v122 /*v378*/
	v_exp_f32_e32 v41 /*v553*/, v123 /*v379*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[186:193], v[172:179] /*v[428:435]*/, v[124:131] /*v[380:387]*/, v[186:193]
	s_set_vgpr_msb 0x542
	ds_load_b128 v[62:65] /*v[318:321]*/, v154 /*v666*/ offset:96
	ds_load_b128 v[42:45] /*v[298:301]*/, v154 /*v666*/ offset:128
	ds_load_b128 v[46:49] /*v[302:305]*/, v154 /*v666*/ offset:160
	s_set_vgpr_msb 0x4281
	v_exp_f32_e32 v42 /*v554*/, v220 /*v476*/
	s_set_vgpr_msb 0x8105
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[154:161], v[74:81] /*v[330:337]*/, v[124:131] /*v[380:387]*/, v[154:161]
	s_set_vgpr_msb 0x542
	ds_load_b128 v[18:21] /*v[274:277]*/, v154 /*v666*/ offset:6400
	ds_load_b128 v[22:25] /*v[278:281]*/, v154 /*v666*/ offset:6432
	ds_load_b128 v[10:13] /*v[266:269]*/, v154 /*v666*/ offset:6464
	s_set_vgpr_msb 0x4280
	v_exp_f32_e32 v36 /*v548*/, v234
	v_exp_f32_e32 v37 /*v549*/, v235
	s_set_vgpr_msb 0x8005
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[122:129], v[204:211] /*v[460:467]*/, v[124:131] /*v[380:387]*/, v[122:129]
	s_wait_dscnt 0x9
	s_set_vgpr_msb 0x542
	ds_load_b128 v[14:17] /*v[270:273]*/, v154 /*v666*/ offset:6496
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[242:245], v154 /*v666*/ offset:6528
	ds_load_b128 v[246:249], v154 /*v666*/ offset:6560
	s_set_vgpr_msb 0x281
	v_exp_f32_e32 v38 /*v550*/, v34 /*v290*/
	v_exp_f32_e32 v39 /*v551*/, v35 /*v291*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[210:217], v[82:89] /*v[338:345]*/, v[124:131] /*v[380:387]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_b128 v[50:53] /*v[306:309]*/, v154 /*v666*/ offset:192
	ds_load_b128 v[54:57] /*v[310:313]*/, v154 /*v666*/ offset:224
	ds_load_b128 v[34:37] /*v[290:293]*/, v154 /*v666*/ offset:256
	s_set_vgpr_msb 0x4291
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[144:145] /*v[400:401]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9180
	v_exp_f32_e32 v14 /*v526*/, v236
	s_set_vgpr_msb 0x8005
	v_wmma_f32_16x16x32_bf16 v[178:185], v[196:203] /*v[452:459]*/, v[124:131] /*v[380:387]*/, v[178:185]
	s_set_vgpr_msb 0x542
	ds_load_b128 v[38:41] /*v[294:297]*/, v154 /*v666*/ offset:288
	ds_load_b128 v[26:29] /*v[282:285]*/, v154 /*v666*/ offset:320
	ds_load_b128 v[30:33] /*v[286:289]*/, v154 /*v666*/ offset:352
	s_set_vgpr_msb 0x4291
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[146:147] /*v[402:403]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9180
	v_exp_f32_e32 v15 /*v527*/, v237
	s_set_vgpr_msb 0x8005
	v_wmma_f32_16x16x32_bf16 v[146:153], v[212:219] /*v[468:475]*/, v[124:131] /*v[380:387]*/, v[146:153]
	s_set_vgpr_msb 0x542
	ds_load_b128 v[2:5] /*v[258:261]*/, v154 /*v666*/ offset:6592
	ds_load_b128 v[6:9] /*v[262:265]*/, v154 /*v666*/ offset:6624
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[234:237], v154 /*v666*/ offset:6656
	s_set_vgpr_msb 0x291
	v_pk_fma_f32 v[76:77] /*v[588:589]*/, v[160:161] /*v[416:417]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9182
	v_exp_f32_e32 v12 /*v524*/, v6 /*v518*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[114:121], v[238:245] /*v[494:501]*/, v[124:131] /*v[380:387]*/, v[114:121]
	s_set_vgpr_msb 0x502
	ds_load_b128 v[238:241], v154 /*v666*/ offset:6688
	ds_load_b128 v[226:229], v154 /*v666*/ offset:6720
	ds_load_b128 v[230:233], v154 /*v666*/ offset:6752
	s_set_vgpr_msb 0x291
	v_pk_fma_f32 v[58:59] /*v[570:571]*/, v[162:163] /*v[418:419]*/, v[250:251], v[250:251] /*v[506:507]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x9182
	v_exp_f32_e32 v13 /*v525*/, v7 /*v519*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[202:209], v[114:121] /*v[370:377]*/, v[132:139] /*v[388:395]*/, v[202:209]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v43 /*v555*/, v221 /*v477*/
	v_exp_f32_e32 v52 /*v564*/, v246 /*v502*/
	v_exp_f32_e32 v53 /*v565*/, v247 /*v503*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[170:177], v[172:179] /*v[428:435]*/, v[132:139] /*v[388:395]*/, v[170:177]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v54 /*v566*/, v248 /*v504*/
	v_exp_f32_e32 v55 /*v567*/, v249 /*v505*/
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v44 /*v556*/, v2 /*v514*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[138:145], v[74:81] /*v[330:337]*/, v[132:139] /*v[388:395]*/, v[138:145]
	s_set_vgpr_msb 0x582
	v_exp_f32_e32 v45 /*v557*/, v3 /*v515*/
	v_exp_f32_e32 v46 /*v558*/, v4 /*v516*/
	v_exp_f32_e32 v47 /*v559*/, v5 /*v517*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[106:113], v[204:211] /*v[460:467]*/, v[132:139] /*v[388:395]*/, v[106:113]
	s_set_vgpr_msb 0x582
	v_exp_f32_e32 v18 /*v530*/, v8 /*v520*/
	v_exp_f32_e32 v19 /*v531*/, v9 /*v521*/
	v_exp_f32_e32 v16 /*v528*/, v28 /*v540*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[194:201], v[82:89] /*v[338:345]*/, v[132:139] /*v[388:395]*/, v[194:201]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v30 /*v542*/, v252 /*v508*/
	v_exp_f32_e32 v31 /*v543*/, v253 /*v509*/
	v_exp_f32_e32 v32 /*v544*/, v254 /*v510*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[162:169], v[196:203] /*v[452:459]*/, v[132:139] /*v[388:395]*/, v[162:169]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v33 /*v545*/, v255 /*v511*/
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v34 /*v546*/, v0 /*v512*/
	v_exp_f32_e32 v35 /*v547*/, v1 /*v513*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[130:137], v[212:219] /*v[468:475]*/, v[132:139] /*v[388:395]*/, v[130:137]
	s_set_vgpr_msb 0x582
	v_exp_f32_e32 v17 /*v529*/, v29 /*v541*/
	v_exp_f32_e32 v28 /*v540*/, v156 /*v668*/
	v_exp_f32_e32 v29 /*v541*/, v157 /*v669*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[98:105], v[238:245] /*v[494:501]*/, v[132:139] /*v[388:395]*/, v[98:105]
	s_wait_dscnt 0xc
	s_add_nc_u64 s[20:21], s[20:21], 1
	s_add_co_i32 s16, s16, s61
	v_cmp_ge_u64_e64 s2, s[20:21], s[46:47]
	s_add_co_i32 s18, s18, s36
	s_mov_b32 s1, s27
	s_mov_b32 s3, s29
	s_and_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s2, s26
	s_set_vgpr_msb 0x500
	s_cbranch_vccz .LBB0_5
	s_branch .LBB0_7
.LBB0_6:
	v_mov_b32_e32 v98, 0
	s_set_vgpr_msb 1
	v_readlane_b32 s56, v1 /*v257*/, 6
	v_readlane_b32 s58, v1 /*v257*/, 9
	s_set_vgpr_msb 0x182
	v_dual_mov_b32 v69 /*v581*/, v68 /*v580*/ :: v_dual_mov_b32 v71 /*v583*/, v70 /*v582*/
	s_set_vgpr_msb 0x8200
	v_dual_mov_b32 v103, v98 :: v_dual_mov_b32 v104, v98
	v_dual_mov_b32 v105, v98 :: v_dual_mov_b32 v99, v98
	v_dual_mov_b32 v100, v98 :: v_dual_mov_b32 v101, v98
	v_mov_b32_e32 v102, v98
	s_delay_alu instid0(VALU_DEP_3)
	v_mov_b64_e32 v[136:137], v[104:105]
	v_mov_b64_e32 v[168:169], v[104:105]
	v_mov_b64_e32 v[200:201], v[104:105]
	v_mov_b64_e32 v[112:113], v[104:105]
	v_mov_b64_e32 v[144:145], v[104:105]
	v_mov_b64_e32 v[176:177], v[104:105]
	v_mov_b64_e32 v[208:209], v[104:105]
	v_mov_b64_e32 v[120:121], v[104:105]
	v_mov_b64_e32 v[152:153], v[104:105]
	v_mov_b64_e32 v[184:185], v[104:105]
	v_mov_b64_e32 v[216:217], v[104:105]
	v_mov_b64_e32 v[128:129], v[104:105]
	v_mov_b64_e32 v[160:161], v[104:105]
	v_mov_b64_e32 v[192:193], v[104:105]
	v_mov_b64_e32 v[224:225], v[104:105]
	v_mov_b64_e32 v[134:135], v[102:103]
	v_mov_b64_e32 v[132:133], v[100:101]
	v_mov_b64_e32 v[130:131], v[98:99]
	v_mov_b64_e32 v[166:167], v[102:103]
	v_mov_b64_e32 v[164:165], v[100:101]
	v_mov_b64_e32 v[162:163], v[98:99]
	v_mov_b64_e32 v[198:199], v[102:103]
	v_mov_b64_e32 v[196:197], v[100:101]
	v_mov_b64_e32 v[194:195], v[98:99]
	v_mov_b64_e32 v[110:111], v[102:103]
	v_mov_b64_e32 v[108:109], v[100:101]
	v_mov_b64_e32 v[106:107], v[98:99]
	v_mov_b64_e32 v[142:143], v[102:103]
	v_mov_b64_e32 v[140:141], v[100:101]
	v_mov_b64_e32 v[138:139], v[98:99]
	v_mov_b64_e32 v[174:175], v[102:103]
	v_mov_b64_e32 v[172:173], v[100:101]
	v_mov_b64_e32 v[170:171], v[98:99]
	v_mov_b64_e32 v[206:207], v[102:103]
	v_mov_b64_e32 v[204:205], v[100:101]
	v_mov_b64_e32 v[202:203], v[98:99]
	v_mov_b64_e32 v[118:119], v[102:103]
	v_mov_b64_e32 v[116:117], v[100:101]
	v_mov_b64_e32 v[114:115], v[98:99]
	v_mov_b64_e32 v[150:151], v[102:103]
	v_mov_b64_e32 v[148:149], v[100:101]
	v_mov_b64_e32 v[146:147], v[98:99]
	v_mov_b64_e32 v[182:183], v[102:103]
	v_mov_b64_e32 v[180:181], v[100:101]
	v_mov_b64_e32 v[178:179], v[98:99]
	v_mov_b64_e32 v[214:215], v[102:103]
	v_mov_b64_e32 v[212:213], v[100:101]
	v_mov_b64_e32 v[210:211], v[98:99]
	v_mov_b64_e32 v[126:127], v[102:103]
	v_mov_b64_e32 v[124:125], v[100:101]
	v_mov_b64_e32 v[122:123], v[98:99]
	v_mov_b64_e32 v[158:159], v[102:103]
	v_mov_b64_e32 v[156:157], v[100:101]
	v_mov_b64_e32 v[154:155], v[98:99]
	v_mov_b64_e32 v[190:191], v[102:103]
	v_mov_b64_e32 v[188:189], v[100:101]
	v_mov_b64_e32 v[186:187], v[98:99]
	v_mov_b64_e32 v[222:223], v[102:103]
	v_mov_b64_e32 v[220:221], v[100:101]
	v_mov_b64_e32 v[218:219], v[98:99]
	s_set_vgpr_msb 1
	v_readlane_b32 s98, v0 /*v256*/, 15
	v_readlane_b32 s57, v1 /*v257*/, 7
	v_readlane_b32 s59, v1 /*v257*/, 10
	v_readlane_b32 s101, v1 /*v257*/, 4
	v_readlane_b32 s37, v1 /*v257*/, 5
	v_readlane_b32 s43, v0 /*v256*/, 17
	v_readlane_b32 s42, v0 /*v256*/, 18
	s_mov_b32 s26, 0x30000
	s_mov_b32 s29, s47
	s_set_vgpr_msb 0x100
.LBB0_7:
	s_set_vgpr_msb 9
	v_readlane_b32 s97, v0 /*v256*/, 12
	v_readlane_b32 s99, v1 /*v257*/, 3
	v_readlane_b32 s100, v1 /*v257*/, 8
	v_mul_u32_u24_e32 v0, 0x120, v152 /*v664*/
	s_ashr_i32 s35, s34, 31
	s_cmp_lt_i32 s46, s34
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_9
	s_set_vgpr_msb 0x80
	v_add_nc_u32_e32 v153 /*v665*/, s40, v255
	s_mov_b32 s0, 0
	s_set_vgpr_msb 0x8000
	s_branch .LBB0_10
.LBB0_9:
	s_mov_b32 s0, -1
.LBB0_10:
	s_set_vgpr_msb 0x82
	v_and_or_b32 v152 /*v664*/, v147 /*v659*/, 16, v0
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_set_vgpr_msb 0x8200
	s_cbranch_vccnz .LBB0_14
	s_set_vgpr_msb 41
	v_readlane_b32 s2, v0 /*v256*/, 5
	v_readlane_b32 s3, v0 /*v256*/, 6
	s_mov_b32 s95, 0
	v_readlane_b32 s76, v0 /*v256*/, 23
	s_mov_b32 s1, s95
	s_add_co_i32 s0, s2, s23
	v_readlane_b32 s77, v0 /*v256*/, 24
	s_sub_co_i32 s0, s0, s3
	s_mov_b32 s92, 8
	s_add_co_i32 s0, s0, 1
	s_mov_b32 s89, 0x800000
	s_min_u32 s0, s22, s0
	s_lshl_b32 s41, s75, 7
	s_add_co_i32 s0, s0, -1
	s_mov_b32 s36, 1
	v_min_u64 v[250:251], s[94:95], s[0:1]
	v_readlane_b32 s0, v0 /*v256*/, 9
	s_lshl_b32 s1, s2, 7
	s_mov_b32 s88, 0xf510000
	s_mov_b32 s83, 0xc80000
	s_mov_b32 s81, 0xc00000
	s_add_co_i32 s0, s97, s0
	s_mov_b32 s80, 0x10000
	s_add_co_i32 s0, s0, s1
	s_mov_b32 s91, s89
	v_add3_u32 v0, s0, v149 /*v661*/, v148 /*v660*/
	s_lshl_b32 s0, s3, 7
	s_mov_b32 s84, s92
	s_mov_b32 s86, s95
	s_mov_b32 s87, s95
	s_set_vgpr_msb 0x2901
	v_subrev_nc_u32_e32 v0, s0, v0
	v_readlane_b32 s0, v0 /*v256*/, 13
	s_mov_b32 s94, s95
	s_mov_b32 s63, 0x76543210
	v_subrev_nc_u32_e32 v251, s0, v0
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b64_e32 v[0:1], s[66:67]
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s0, v250
	v_subrev_nc_u32_e32 v250, s37, v251
	s_lshl_b32 s0, s0, 7
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v253, v250, v146 /*v658*/
	s_add_co_i32 s1, s37, s0
	s_sub_co_i32 s62, 0, s0
	s_add_co_i32 s0, s1, 0x80
	s_mul_i32 s1, s75, s1
	s_mul_i32 s0, s74, s0
	s_add_co_i32 s42, s42, s1
	s_add_co_i32 s44, s43, s0
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x800
.LBB0_12:
	s_mov_b32 s96, s33
	s_mov_b32 s33, s26
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x80
	v_add_nc_u32_e32 v147 /*v659*/, s40, v255
	s_set_vgpr_msb 0x8088
	v_add_nc_u32_e32 v146 /*v658*/, s96, v152 /*v664*/
	s_mov_b32 s64, s29
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8880
	v_add_nc_u32_e32 v153 /*v665*/, s64, v255
	s_add_co_i32 s0, s62, vcc_hi
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, s[94:95]
	s_set_vgpr_msb 0x4000
	v_med3_i32 v250, s0, 0, 8
	s_max_i32 s1, s0, 32
	s_max_i32 s2, s0, 64
	s_max_i32 s3, s0, 0x60
	s_addk_co_i32 s0, 0xff80
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, s[90:91]
	s_sub_co_i32 s1, s1, 32
	s_sub_co_i32 s2, s2, 64
	s_addk_co_i32 s3, 0xffa0
	s_max_i32 s4, s0, 32
	v_med3_i32 v76 /*v332*/, s0, 0, 8
	s_max_i32 s5, s0, 64
	s_max_i32 s0, s0, 0x60
	s_min_u32 s1, s1, 8
	s_min_u32 s2, s2, 8
	s_min_u32 s3, s3, 8
	s_sub_co_i32 s4, s4, 32
	s_sub_co_i32 s5, s5, 64
	s_addk_co_i32 s0, 0xffa0
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, s[92:93]
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, s[88:89]
	s_lshl_b32 s90, s1, 16
	s_lshl_b32 s1, s2, 16
	s_lshl_b32 s2, s3, 16
	s_min_u32 s3, s4, 8
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, s[86:87]
	s_min_u32 s4, s5, 8
	s_min_u32 s0, s0, 8
	s_mov_b64 s[48:49], s[88:89]
	s_mov_b64 s[24:25], s[88:89]
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, s[84:85]
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, s[82:83]
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, s[80:81]
	s_mov_b64 s[50:51], s[90:91]
	s_mov_b64 s[26:27], s[90:91]
	s_mov_b64 s[28:29], s[92:93]
	s_lshl_b32 s82, s3, 16
	s_lshl_b32 s16, s4, 16
	s_lshl_b32 s17, s0, 16
	s_mov_b32 s50, s1
	s_mov_b32 s26, s2
	s_mov_b32 s29, s93
	s_mov_b64 s[8:9], s[80:81]
	s_mov_b64 s[0:1], s[80:81]
	s_mov_b64 s[52:53], s[92:93]
	s_mov_b64 s[10:11], s[82:83]
	s_mov_b64 s[12:13], s[84:85]
	s_mov_b64 s[2:3], s[82:83]
	s_mov_b64 s[4:5], s[84:85]
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v251, s93 :: v_dual_lshlrev_b32 v250, 16, v250
	s_set_vgpr_msb 0x44
	v_dual_mov_b32 v79 /*v335*/, s85 :: v_dual_lshlrev_b32 v76 /*v332*/, 16, v76 /*v332*/
	s_mov_b64 s[54:55], s[94:95]
	s_mov_b64 s[30:31], s[94:95]
	s_mov_b32 s53, s93
	s_mov_b64 s[14:15], s[86:87]
	s_mov_b64 s[6:7], s[86:87]
	s_mov_b32 s10, s16
	s_mov_b32 s13, s85
	s_mov_b32 s2, s17
	s_mov_b32 s5, s85
	s_wait_dscnt 0xc
	s_ashr_i32 s45, s44, 31
	s_add_co_i32 s37, s64, s104
	s_add_nc_u64 s[38:39], s[76:77], s[44:45]
	s_mov_b64 s[20:21], s[36:37]
	s_or_b32 s18, s39, 0x80000000
	s_add_nc_u64 s[16:17], s[38:39], s[78:79]
	s_mov_b64 s[22:23], s[38:39]
	s_mov_b32 s23, s18
	s_add_co_i32 s18, s37, 0x3200
	s_or_b32 s19, s17, 0x80000000
	s_mov_b64 s[58:59], s[38:39]
	s_mov_b64 s[56:57], s[36:37]
	s_add_nc_u64 s[38:39], s[16:17], s[78:79]
	s_mov_b32 s57, s18
	s_mov_b32 s58, s16
	s_mov_b32 s59, s19
	s_mov_b64 s[16:17], s[36:37]
	s_mov_b64 s[18:19], s[38:39]
	s_add_co_i32 s43, s37, 0x6400
	s_or_b32 s45, s39, 0x80000000
	s_mov_b32 s18, s38
	s_add_nc_u64 s[38:39], s[38:39], s[78:79]
	s_mov_b32 s17, s43
	s_mov_b32 s19, s45
	s_add_co_i32 s37, s37, 0x9600
	s_bitset1_b32 s39, 31
	s_set_vgpr_msb 0x4441
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[66:73] /*v[322:329]*/, v[66:73], 0
	v_readfirstlane_b32 s68, v82 /*v338*/
	v_readfirstlane_b32 s69, v83 /*v339*/
	v_readfirstlane_b32 s70, v76 /*v332*/
	v_readfirstlane_b32 s71, v85 /*v341*/
	v_readfirstlane_b32 s72, v86 /*v342*/
	v_readfirstlane_b32 s73, v79 /*v335*/
	v_readfirstlane_b32 s74, v88 /*v344*/
	v_readfirstlane_b32 s75, v89 /*v345*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[20:23], s[68:75]
	tensor_load_to_lds s[56:59], s[80:87]
	v_readlane_b32 s58, v1 /*v257*/, 9
	v_readlane_b32 s56, v1 /*v257*/, 6
	v_readlane_b32 s59, v1 /*v257*/, 10
	v_readlane_b32 s57, v1 /*v257*/, 7
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v166 /*v422*/, v136 /*v648*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[66:73] /*v[322:329]*/, v[90:97], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4142
	ds_load_b128 v[66:69] /*v[322:325]*/, v147 /*v659*/ offset:12800
	ds_load_b128 v[70:73] /*v[326:329]*/, v147 /*v659*/ offset:12832
	ds_load_b128 v[82:85] /*v[338:341]*/, v147 /*v659*/ offset:12864
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[224:225], v[224:225], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[58:65] /*v[314:321]*/, v[50:57], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[86:89] /*v[342:345]*/, v147 /*v659*/ offset:12896
	ds_load_b128 v[106:109] /*v[362:365]*/, v147 /*v659*/ offset:12928
	ds_load_b128 v[110:113] /*v[366:369]*/, v147 /*v659*/ offset:12960
	v_exp_f32_e32 v167 /*v423*/, v137 /*v649*/
	v_exp_f32_e32 v178 /*v434*/, v122 /*v634*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[58:65] /*v[314:321]*/, v[82:89], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[122:125] /*v[378:381]*/, v147 /*v659*/ offset:19200
	ds_load_b128 v[126:129] /*v[382:385]*/, v147 /*v659*/ offset:19232
	ds_load_b128 v[198:201] /*v[454:457]*/, v147 /*v659*/ offset:19264
	v_exp_f32_e32 v164 /*v420*/, v144 /*v656*/
	v_exp_f32_e32 v165 /*v421*/, v145 /*v657*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[42:49] /*v[298:305]*/, v[34:41], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[202:205] /*v[458:461]*/, v147 /*v659*/ offset:19296
	ds_load_b128 v[58:61] /*v[314:317]*/, v147 /*v659*/ offset:19328
	ds_load_b128 v[62:65] /*v[318:321]*/, v147 /*v659*/ offset:19360
	v_exp_f32_e32 v168 /*v424*/, v140 /*v652*/
	v_exp_f32_e32 v169 /*v425*/, v141 /*v653*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[42:49] /*v[298:305]*/, v[74:81], v[98:105] /*v[354:361]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[130:133] /*v[386:389]*/, v147 /*v659*/ offset:12992
	ds_load_b128 v[134:137] /*v[390:393]*/, v147 /*v659*/ offset:13024
	ds_load_b128 v[206:209] /*v[462:465]*/, v147 /*v659*/ offset:13056
	v_exp_f32_e32 v158 /*v414*/, v138 /*v650*/
	v_exp_f32_e32 v159 /*v415*/, v139 /*v651*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[50:57] /*v[306:313]*/, v[42:49], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[210:213] /*v[466:469]*/, v147 /*v659*/ offset:13088
	ds_load_b128 v[214:217] /*v[470:473]*/, v147 /*v659*/ offset:13120
	ds_load_b128 v[218:221] /*v[474:477]*/, v147 /*v659*/ offset:13152
	v_exp_f32_e32 v160 /*v416*/, v118 /*v630*/
	v_exp_f32_e32 v179 /*v435*/, v123 /*v635*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[50:57] /*v[306:313]*/, v[58:65], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[222:225] /*v[478:481]*/, v147 /*v659*/ offset:19392
	ds_load_b128 v[226:229] /*v[482:485]*/, v147 /*v659*/ offset:19424
	ds_load_b128 v[114:117] /*v[370:373]*/, v147 /*v659*/ offset:19456
	v_exp_f32_e32 v154 /*v410*/, v142 /*v654*/
	v_exp_f32_e32 v155 /*v411*/, v143 /*v655*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[34:41] /*v[290:297]*/, v[10:17], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[118:121] /*v[374:377]*/, v147 /*v659*/ offset:19488
	ds_load_b128 v[50:53] /*v[306:309]*/, v147 /*v659*/ offset:19520
	ds_load_b128 v[54:57] /*v[310:313]*/, v147 /*v659*/ offset:19552
	v_exp_f32_e32 v156 /*v412*/, v120 /*v632*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[34:41] /*v[290:297]*/, v[26:33], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v194 /*v450*/, v130 /*v642*/
	v_exp_f32_e32 v195 /*v451*/, v131 /*v643*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[26:33] /*v[282:289]*/, v[2:9], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v196 /*v452*/, v128 /*v640*/
	v_exp_f32_e32 v197 /*v453*/, v129 /*v641*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[26:33] /*v[282:289]*/, v[18:25], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v186 /*v442*/, v134 /*v646*/
	v_exp_f32_e32 v187 /*v443*/, v135 /*v647*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[66:73], 0
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v188 /*v444*/, v132 /*v644*/
	v_exp_f32_e32 v189 /*v445*/, v133 /*v645*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[18:25] /*v[274:281]*/, v[90:97], 0
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v190 /*v446*/, v126 /*v638*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[216:217], v[216:217], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215], v[214:215], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213], v[212:213], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[10:17] /*v[266:273]*/, v[50:57], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v191 /*v447*/, v127 /*v639*/
	v_exp_f32_e32 v162 /*v418*/, v116 /*v628*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[10:17] /*v[266:273]*/, v[82:89], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v161 /*v417*/, v119 /*v631*/
	v_exp_f32_e32 v157 /*v413*/, v121 /*v633*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[242:249], v[34:41], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v174 /*v430*/, v124 /*v636*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[208:209], v[208:209], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[242:249], v[74:81], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v175 /*v431*/, v125 /*v637*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[200:201], v[200:201], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199], v[198:199], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[196:197], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195], v[194:195], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[42:49], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v163 /*v419*/, v117 /*v629*/
	v_exp_f32_e32 v170 /*v426*/, v114 /*v626*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[58:65], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v171 /*v427*/, v115 /*v627*/
	v_exp_f32_e32 v172 /*v428*/, v112 /*v624*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[234:241], v[10:17], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v173 /*v429*/, v113 /*v625*/
	v_exp_f32_e32 v176 /*v432*/, v110 /*v622*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[234:241], v[26:33], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v177 /*v433*/, v111 /*v623*/
	v_exp_f32_e32 v180 /*v436*/, v108 /*v620*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[226:233], v[2:9], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v181 /*v437*/, v109 /*v621*/
	v_exp_f32_e32 v182 /*v438*/, v106 /*v618*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[226:233], v[18:25], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v183 /*v439*/, v107 /*v619*/
	s_wait_dscnt 0xc
	v_exp_f32_e32 v192 /*v448*/, v104 /*v616*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[66:73] /*v[322:329]*/, v[66:73], 0
	tensor_load_to_lds s[16:19], s[8:15]
	tensor_load_to_lds s[36:39], s[0:7]
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v193 /*v449*/, v105 /*v617*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[66:73] /*v[322:329]*/, v[90:97], 0
	s_set_vgpr_msb 0x4102
	ds_load_b128 v[226:229], v147 /*v659*/ offset:25600
	ds_load_b128 v[230:233], v147 /*v659*/ offset:25632
	ds_load_b128 v[234:237], v147 /*v659*/ offset:25664
	v_pk_mul_f32 v[192:193], v[74:75] /*v[586:587]*/, v[192:193] op_sel_hi:[0,1]
	v_pk_mul_f32 v[190:191], v[74:75] /*v[586:587]*/, v[190:191] op_sel_hi:[0,1]
	v_pk_mul_f32 v[188:189], v[74:75] /*v[586:587]*/, v[188:189] op_sel_hi:[0,1]
	v_pk_mul_f32 v[186:187], v[74:75] /*v[586:587]*/, v[186:187] op_sel_hi:[0,1]
	s_set_vgpr_msb 0x251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[82:89] /*v[338:345]*/, v[50:57], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5102
	ds_load_b128 v[238:241], v147 /*v659*/ offset:25696
	s_set_vgpr_msb 0x242
	ds_load_b128 v[2:5] /*v[258:261]*/, v147 /*v659*/ offset:25728
	ds_load_b128 v[6:9] /*v[262:265]*/, v147 /*v659*/ offset:25760
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v10 /*v522*/, v10 /*v522*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[82:89] /*v[338:345]*/, v[82:89], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[146:149] /*v[402:405]*/, v147 /*v659*/ offset:32000
	ds_load_b128 v[150:153] /*v[406:409]*/, v147 /*v659*/ offset:32032
	ds_load_b128 v[10:13] /*v[266:269]*/, v147 /*v659*/ offset:32064
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v104 /*v616*/, v100 /*v612*/
	v_exp_f32_e32 v105 /*v617*/, v101 /*v613*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[106:113] /*v[362:369]*/, v[34:41], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[14:17] /*v[270:273]*/, v147 /*v659*/ offset:32096
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[242:245], v147 /*v659*/ offset:32128
	ds_load_b128 v[246:249], v147 /*v659*/ offset:32160
	s_set_vgpr_msb 0x242
	v_exp_f32_e32 v184 /*v440*/, v96 /*v608*/
	v_exp_f32_e32 v185 /*v441*/, v97 /*v609*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[106:113] /*v[362:369]*/, v[74:81], v[18:25] /*v[274:281]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[106:109] /*v[618:621]*/, v147 /*v659*/ offset:25792
	ds_load_b128 v[110:113] /*v[622:625]*/, v147 /*v659*/ offset:25824
	ds_load_b128 v[114:117] /*v[626:629]*/, v147 /*v659*/ offset:25856
	v_exp_f32_e32 v100 /*v612*/, v102 /*v614*/
	v_exp_f32_e32 v101 /*v613*/, v103 /*v615*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[130:137] /*v[386:393]*/, v[42:49], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[118:121] /*v[630:633]*/, v147 /*v659*/ offset:25888
	ds_load_b128 v[122:125] /*v[634:637]*/, v147 /*v659*/ offset:25920
	ds_load_b128 v[126:129] /*v[638:641]*/, v147 /*v659*/ offset:25952
	v_exp_f32_e32 v98 /*v610*/, v98 /*v610*/
	v_exp_f32_e32 v99 /*v611*/, v99 /*v611*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[130:137] /*v[386:393]*/, v[58:65], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[82:85] /*v[338:341]*/, v147 /*v659*/ offset:32192
	ds_load_b128 v[86:89] /*v[342:345]*/, v147 /*v659*/ offset:32224
	ds_load_b128 v[138:141] /*v[394:397]*/, v147 /*v659*/ offset:32256
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v88 /*v600*/, v88 /*v600*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[206:213] /*v[462:469]*/, v[10:17], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[142:145] /*v[398:401]*/, v147 /*v659*/ offset:32288
	ds_load_b128 v[130:133] /*v[386:389]*/, v147 /*v659*/ offset:32320
	ds_load_b128 v[134:137] /*v[390:393]*/, v147 /*v659*/ offset:32352
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v89 /*v601*/, v89 /*v601*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[206:213] /*v[462:469]*/, v[26:33], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v94 /*v606*/, v94 /*v606*/
	v_exp_f32_e32 v95 /*v607*/, v95 /*v607*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[214:221] /*v[470:477]*/, v[2:9], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v92 /*v604*/, v92 /*v604*/
	v_exp_f32_e32 v93 /*v605*/, v93 /*v605*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[214:221] /*v[470:477]*/, v[18:25], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v90 /*v602*/, v90 /*v602*/
	v_exp_f32_e32 v91 /*v603*/, v91 /*v603*/
	s_set_vgpr_msb 0x8241
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[122:129] /*v[378:385]*/, v[66:73], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v86 /*v598*/, v86 /*v598*/
	v_exp_f32_e32 v87 /*v599*/, v87 /*v599*/
	s_set_vgpr_msb 0x8241
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[122:129] /*v[378:385]*/, v[90:97], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v84 /*v596*/, v84 /*v596*/
	v_exp_f32_e32 v85 /*v597*/, v85 /*v597*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[198:205] /*v[454:461]*/, v[50:57], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v96 /*v608*/, v80 /*v592*/
	v_exp_f32_e32 v80 /*v592*/, v82 /*v594*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[198:205] /*v[454:461]*/, v[82:89], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v97 /*v609*/, v81 /*v593*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[184:185], v[184:185], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183], v[182:183], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181], v[180:181], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179], v[178:179], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[58:65] /*v[314:321]*/, v[34:41], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v81 /*v593*/, v83 /*v595*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[176:177], v[176:177], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175], v[174:175], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[172:173], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171], v[170:171], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[58:65] /*v[314:321]*/, v[74:81], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v66 /*v578*/, v66 /*v578*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[168:169], v[168:169], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[222:229] /*v[478:485]*/, v[42:49], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v67 /*v579*/, v67 /*v579*/
	v_exp_f32_e32 v82 /*v594*/, v60 /*v572*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[222:229] /*v[478:485]*/, v[58:65], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v83 /*v595*/, v61 /*v573*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[114:121] /*v[370:377]*/, v[10:17], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v60 /*v572*/, v78 /*v590*/
	v_exp_f32_e32 v61 /*v573*/, v79 /*v591*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[114:121] /*v[370:377]*/, v[26:33], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v62 /*v574*/, v62 /*v574*/
	v_exp_f32_e32 v63 /*v575*/, v63 /*v575*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[50:57] /*v[306:313]*/, v[2:9], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v26 /*v538*/, v26 /*v538*/
	v_exp_f32_e32 v27 /*v539*/, v27 /*v539*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[50:57] /*v[306:313]*/, v[18:25], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	s_wait_dscnt 0xc
	v_exp_f32_e32 v11 /*v523*/, v11 /*v523*/
	s_set_vgpr_msb 0x8240
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[226:233], v[66:73], 0
	s_set_vgpr_msb 0x4082
	ds_load_b128 v[130:133] /*v[642:645]*/, v147 /*v659*/ offset:38400
	ds_load_b128 v[134:137] /*v[646:649]*/, v147 /*v659*/ offset:38432
	ds_load_b128 v[138:141] /*v[650:653]*/, v147 /*v659*/ offset:38464
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v25 /*v537*/, v25 /*v537*/
	s_set_vgpr_msb 0x8240
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[226:233], v[90:97], 0
	s_set_vgpr_msb 0x4082
	ds_load_b128 v[142:145] /*v[654:657]*/, v147 /*v659*/ offset:38496
	ds_load_b128 v[154:157] /*v[666:669]*/, v147 /*v659*/ offset:38528
	ds_load_b128 v[158:161] /*v[670:673]*/, v147 /*v659*/ offset:38560
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	s_set_vgpr_msb 0x8250
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[234:241], v[50:57], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5042
	ds_load_b128 v[250:253] /*v[506:509]*/, v147 /*v659*/ offset:44800
	ds_load_b128 v[254:257] /*v[510:513]*/, v147 /*v659*/ offset:44832
	ds_load_b128 v[242:245] /*v[498:501]*/, v147 /*v659*/ offset:44864
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v50 /*v562*/, v50 /*v562*/
	v_exp_f32_e32 v51 /*v563*/, v51 /*v563*/
	s_set_vgpr_msb 0x8250
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[234:241], v[82:89], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5042
	ds_load_b128 v[246:249] /*v[502:505]*/, v147 /*v659*/ offset:44896
	ds_load_b128 v[226:229] /*v[482:485]*/, v147 /*v659*/ offset:44928
	ds_load_b128 v[230:233] /*v[486:489]*/, v147 /*v659*/ offset:44960
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v49 /*v561*/, v49 /*v561*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[2:9] /*v[258:265]*/, v[34:41], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[162:165] /*v[674:677]*/, v147 /*v659*/ offset:38592
	ds_load_b128 v[166:169] /*v[678:681]*/, v147 /*v659*/ offset:38624
	ds_load_b128 v[170:173] /*v[682:685]*/, v147 /*v659*/ offset:38656
	v_exp_f32_e32 v64 /*v576*/, v64 /*v576*/
	v_exp_f32_e32 v65 /*v577*/, v65 /*v577*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[2:9] /*v[258:265]*/, v[74:81], v[122:129] /*v[378:385]*/
	s_wait_dscnt 0xf
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[174:177] /*v[686:689]*/, v147 /*v659*/ offset:38688
	ds_load_b128 v[2:5] /*v[514:517]*/, v147 /*v659*/ offset:38720
	ds_load_b128 v[6:9] /*v[518:521]*/, v147 /*v659*/ offset:38752
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v57 /*v569*/, v57 /*v569*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[106:113] /*v[618:625]*/, v[42:49], v[114:121] /*v[370:377]*/
	ds_load_b128 v[234:237] /*v[490:493]*/, v147 /*v659*/ offset:44992
	ds_load_b128 v[238:241] /*v[494:497]*/, v147 /*v659*/ offset:45024
	ds_load_b128 v[218:221] /*v[474:477]*/, v147 /*v659*/ offset:45056
	s_set_vgpr_msb 0x5282
	v_exp_f32_e32 v76 /*v588*/, v76 /*v588*/
	v_exp_f32_e32 v77 /*v589*/, v77 /*v589*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[106:113] /*v[618:625]*/, v[58:65], v[122:129] /*v[378:385]*/
	ds_load_b128 v[222:225] /*v[478:481]*/, v147 /*v659*/ offset:45088
	ds_load_b128 v[202:205] /*v[458:461]*/, v147 /*v659*/ offset:45120
	ds_load_b128 v[206:209] /*v[462:465]*/, v147 /*v659*/ offset:45152
	s_set_vgpr_msb 0x5282
	v_exp_f32_e32 v58 /*v570*/, v58 /*v570*/
	v_exp_f32_e32 v59 /*v571*/, v59 /*v571*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[114:121] /*v[626:633]*/, v[10:17], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x524a
	v_cvt_pk_bf16_f32 v58 /*v314*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v54 /*v566*/, v55 /*v567*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v2 /*v258*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v178 /*v434*/, v179 /*v435*/
	s_set_vgpr_msb 0x4552
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[114:121] /*v[626:633]*/, v[26:33], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5245
	v_cvt_pk_bf16_f32 v4 /*v260*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v196 /*v452*/, v197 /*v453*/
	s_set_vgpr_msb 0x4505
	v_cvt_pk_bf16_f32 v234, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v235, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v236, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v237, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v226, v180 /*v436*/, v181 /*v437*/
	s_set_vgpr_msb 0x552
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[122:129] /*v[634:641]*/, v[2:9], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5205
	v_cvt_pk_bf16_f32 v227, v182 /*v438*/, v183 /*v439*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v228, v10 /*v522*/, v11 /*v523*/
	v_cvt_pk_bf16_f32 v229, v20 /*v532*/, v21 /*v533*/
	s_set_vgpr_msb 0xa8a
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[52:53] /*v[564:565]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8a45
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, v[178:179] /*v[434:435]*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x4552
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[122:129] /*v[634:641]*/, v[18:25], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x524a
	v_cvt_pk_bf16_f32 v62 /*v318*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v63 /*v319*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v64 /*v320*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v65 /*v321*/, v46 /*v558*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v6 /*v262*/, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v186 /*v442*/, v187 /*v443*/
	s_set_vgpr_msb 0x4541
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[146:153] /*v[402:409]*/, v[66:73], 0
	s_set_vgpr_msb 0x4145
	v_cvt_pk_bf16_f32 v9 /*v265*/, v188 /*v444*/, v189 /*v445*/
	s_set_vgpr_msb 0x4505
	v_cvt_pk_bf16_f32 v238, v190 /*v446*/, v191 /*v447*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v239, v104 /*v616*/, v105 /*v617*/
	s_set_vgpr_msb 0xa05
	v_cvt_pk_bf16_f32 v240, v184 /*v440*/, v185 /*v441*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v241, v86 /*v598*/, v87 /*v599*/
	v_cvt_pk_bf16_f32 v230, v84 /*v596*/, v85 /*v597*/
	v_cvt_pk_bf16_f32 v231, v96 /*v608*/, v97 /*v609*/
	s_set_vgpr_msb 0xa41
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[146:153] /*v[402:409]*/, v[90:97], 0
	s_set_vgpr_msb 0x410a
	v_cvt_pk_bf16_f32 v232, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v233, v48 /*v560*/, v49 /*v561*/
	v_nop
	v_nop
	s_set_vgpr_msb 0xa4a
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[44:45] /*v[556:557]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a85
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[164:165] /*v[420:421]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[186:187] /*v[442:443]*/, v[188:189] /*v[444:445]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[190:191] /*v[446:447]*/, v[104:105] /*v[616:617]*/
	s_set_vgpr_msb 0x8951
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[10:17] /*v[266:273]*/, v[50:57], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v50 /*v306*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v51 /*v307*/, v30 /*v542*/, v31 /*v543*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[160:161], v[160:161], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[10:17] /*v[266:273]*/, v[82:89], v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v52 /*v308*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v53 /*v309*/, v34 /*v546*/, v35 /*v547*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[152:153], v[152:153], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[242:249], v[34:41], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5045
	v_cvt_pk_bf16_f32 v10 /*v266*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v11 /*v267*/, v160 /*v416*/, v161 /*v417*/
	s_set_vgpr_msb 0x4508
	v_pk_mul_f32 v[144:145], v[144:145], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[242:249], v[74:81], v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x504a
	v_cvt_pk_bf16_f32 v54 /*v310*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v55 /*v311*/, v18 /*v530*/, v19 /*v531*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[136:137], v[136:137], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[134:135], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[132:133], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[82:89] /*v[338:345]*/, v[42:49], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5145
	v_cvt_pk_bf16_f32 v12 /*v268*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v13 /*v269*/, v192 /*v448*/, v193 /*v449*/
	s_set_vgpr_msb 0x450a
	v_cvt_pk_bf16_f32 v242, v100 /*v612*/, v101 /*v613*/
	v_cvt_pk_bf16_f32 v243, v98 /*v610*/, v99 /*v611*/
	v_cvt_pk_bf16_f32 v244, v80 /*v592*/, v81 /*v593*/
	v_cvt_pk_bf16_f32 v245, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0xa51
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[82:89] /*v[338:345]*/, v[58:65], v[210:217] /*v[466:473]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v82 /*v338*/, v82 /*v594*/, v83 /*v595*/
	v_cvt_pk_bf16_f32 v83 /*v339*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v84 /*v340*/, v64 /*v576*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v85 /*v341*/, v56 /*v568*/, v57 /*v569*/
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[14:15] /*v[526:527]*/, v[30:31] /*v[542:543]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[32:33] /*v[544:545]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[138:145] /*v[394:401]*/, v[10:17], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[174:175] /*v[430:431]*/, v[192:193] /*v[448:449]*/
	s_set_vgpr_msb 0x458a
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[100:101] /*v[612:613]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8a4a
	v_cvt_pk_bf16_f32 v56 /*v312*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v28 /*v540*/, v29 /*v541*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v14 /*v270*/, v154 /*v410*/, v155 /*v411*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[138:145] /*v[394:401]*/, v[26:33], v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5145
	v_cvt_pk_bf16_f32 v15 /*v271*/, v156 /*v412*/, v157 /*v413*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v16 /*v272*/, v88 /*v600*/, v89 /*v601*/
	v_cvt_pk_bf16_f32 v17 /*v273*/, v94 /*v606*/, v95 /*v607*/
	s_set_vgpr_msb 0x4a0a
	v_cvt_pk_bf16_f32 v246, v92 /*v604*/, v93 /*v605*/
	v_cvt_pk_bf16_f32 v247, v90 /*v602*/, v91 /*v603*/
	v_cvt_pk_bf16_f32 v248, v60 /*v572*/, v61 /*v573*/
	s_set_vgpr_msb 0xa51
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[130:137] /*v[386:393]*/, v[2:9], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x510a
	v_cvt_pk_bf16_f32 v249, v62 /*v574*/, v63 /*v575*/
	s_set_vgpr_msb 0xa4a
	v_cvt_pk_bf16_f32 v86 /*v342*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v87 /*v343*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v88 /*v344*/, v76 /*v588*/, v77 /*v589*/
	v_cvt_pk_bf16_f32 v89 /*v345*/, v58 /*v570*/, v59 /*v571*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[12:13] /*v[524:525]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[130:137] /*v[386:393]*/, v[18:25], v[210:217] /*v[466:473]*/
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[172:173] /*v[428:429]*/, v[176:177] /*v[432:433]*/
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[180:181] /*v[436:437]*/, v[182:183] /*v[438:439]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[10:11] /*v[522:523]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[16:17] /*v[528:529]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[154:155] /*v[410:411]*/, v[156:157] /*v[412:413]*/
	s_barrier_wait -1
	s_set_vgpr_msb 0x8582
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[130:137] /*v[642:649]*/, v[66:73], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v146 /*v658*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v146 /*v658*/ offset:4608
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[166:167] /*v[422:423]*/, v[178:179] /*v[434:435]*/
	s_set_vgpr_msb 0x8582
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[130:137] /*v[642:649]*/, v[90:97], 0
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v146 /*v658*/ offset:32
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v146 /*v658*/ offset:4640
	s_set_vgpr_msb 0x4285
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[130:131] /*v[386:387]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[78:79] /*v[590:591]*/, v[134:135] /*v[390:391]*/, v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x85a2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[138:145] /*v[650:657]*/, v[50:57], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v146 /*v658*/ offset:128
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v146 /*v658*/ offset:4736
	s_set_vgpr_msb 0x4289
	v_pk_add_f32 v[86:87] /*v[598:599]*/, v[184:185] /*v[440:441]*/, v[86:87] /*v[598:599]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[138:145] /*v[650:657]*/, v[82:89], v[28:35] /*v[540:547]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v146 /*v658*/ offset:160
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v146 /*v658*/ offset:4768
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[50:51] /*v[562:563]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[146:147] /*v[402:403]*/, v[148:149] /*v[404:405]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[80:81] /*v[592:593]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[154:161] /*v[666:673]*/, v[34:41], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v146 /*v658*/ offset:64
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v146 /*v658*/ offset:4672
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[82:83] /*v[594:595]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[154:161] /*v[666:673]*/, v[74:81], v[28:35] /*v[540:547]*/
	s_wait_dscnt 0xa
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v146 /*v658*/ offset:96
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v146 /*v658*/ offset:4704
	s_set_vgpr_msb 0x4285
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[150:151] /*v[406:407]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[82:83] /*v[594:595]*/, v[158:159] /*v[414:415]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x85a2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[162:169] /*v[674:681]*/, v[42:49], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v146 /*v658*/ offset:192
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v146 /*v658*/ offset:4800
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[88:89] /*v[600:601]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[92:93] /*v[604:605]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[162:169] /*v[674:681]*/, v[58:65], v[28:35] /*v[540:547]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v146 /*v658*/ offset:224
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v146 /*v658*/ offset:4832
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, v[62:63] /*v[574:575]*/
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[170:177] /*v[682:689]*/, v[10:17], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa28a
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/, v[86:87] /*v[598:599]*/
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[84:85] /*v[596:597]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[24:25] /*v[536:537]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[170:177] /*v[682:689]*/, v[26:33], v[28:35] /*v[540:547]*/
	s_set_vgpr_msb 0xa28a
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[50:51] /*v[562:563]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[2:9] /*v[514:521]*/, v[2:9], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa28a
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[76:77] /*v[588:589]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[2:9] /*v[514:521]*/, v[18:25], v[28:35] /*v[540:547]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa28a
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[22:23] /*v[534:535]*/, v[24:25] /*v[536:537]*/
	v_pk_add_f32 v[4:5] /*v[516:517]*/, v[54:55] /*v[566:567]*/, v[78:79] /*v[590:591]*/
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x8a81
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[250:257] /*v[506:513]*/, v[66:73], 0
	s_set_vgpr_msb 0x818a
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[46:47] /*v[558:559]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[52:53] /*v[564:565]*/, v[88:89] /*v[600:601]*/
	s_set_vgpr_msb 0x8a81
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[250:257] /*v[506:513]*/, v[90:97], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x814a
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[80:81] /*v[592:593]*/, v[56:57] /*v[568:569]*/
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[64:65] /*v[576:577]*/, v[82:83] /*v[594:595]*/
	s_set_vgpr_msb 0x4aa1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[242:249] /*v[498:505]*/, v[50:57], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa14a
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[90:91] /*v[602:603]*/, v[60:61] /*v[572:573]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[26:27] /*v[538:539]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x8aa1
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[242:249] /*v[498:505]*/, v[82:89], v[36:43] /*v[548:555]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa14a
	v_pk_add_f32 v[242:243] /*v[498:499]*/, v[6:7] /*v[518:519]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[244:245] /*v[500:501]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x49a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[226:233] /*v[482:489]*/, v[34:41], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[128:129], v[128:129], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x846
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[8:9] /*v[520:521]*/, v[250:251] /*v[506:507]*/
	s_set_vgpr_msb 0x46a1
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[226:233] /*v[482:489]*/, v[74:81], v[36:43] /*v[548:555]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[120:121], v[120:121], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x845
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[252:253] /*v[508:509]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x45a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[234:241] /*v[490:497]*/, v[42:49], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[112:113], v[112:113], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x84a
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[44:45] /*v[556:557]*/, v[4:5] /*v[516:517]*/
	s_set_vgpr_msb 0x4aa1
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[234:241] /*v[490:497]*/, v[58:65], v[36:43] /*v[548:555]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[104:105], v[104:105], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x845
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[242:243] /*v[498:499]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x45a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[218:225] /*v[474:481]*/, v[10:17], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[218:225] /*v[474:481]*/, v[26:33], v[36:43] /*v[548:555]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa141
	v_mov_b32_e32 v218 /*v474*/, v228 /*v484*/
	s_set_vgpr_msb 0x4142
	v_mov_b32_e32 v219 /*v475*/, v2 /*v514*/
	s_set_vgpr_msb 0x4281
	v_mov_b32_e32 v2 /*v514*/, v229 /*v485*/
	s_set_vgpr_msb 0x8149
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x4986
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[70:71] /*v[582:583]*/, v[218:219] /*v[474:475]*/
	s_set_vgpr_msb 0x86a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[202:209] /*v[458:465]*/, v[2:9], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa145
	v_dual_mov_b32 v218 /*v474*/, v226 /*v482*/ :: v_dual_mov_b32 v219 /*v475*/, v230 /*v486*/
	v_mov_b32_e32 v230 /*v486*/, v227 /*v483*/
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x4586
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[68:69] /*v[580:581]*/, v[218:219] /*v[474:475]*/
	s_set_vgpr_msb 0x86a1
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[202:209] /*v[458:465]*/, v[18:25], v[36:43] /*v[548:555]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0xa140
	v_add_nc_u32_e32 v76 /*v332*/, s62, v253
	s_ashr_i32 s43, s42, 31
	s_add_co_i32 s37, s33, s101
	s_add_nc_u64 s[38:39], s[58:59], s[42:43]
	s_set_vgpr_msb 0x4045
	v_add_nc_u32_e32 v79 /*v335*/, -16, v76 /*v332*/
	v_cmp_gt_i32_e32 vcc_lo, 0, v76 /*v332*/
	v_cmp_gt_i32_e64 s0, 1, v76 /*v332*/
	v_cmp_gt_i32_e64 s1, 2, v76 /*v332*/
	v_cmp_gt_i32_e64 s2, 3, v76 /*v332*/
	v_cmp_gt_i32_e64 s7, 0, v79 /*v335*/
	v_cmp_gt_i32_e64 s3, 4, v76 /*v332*/
	v_cmp_gt_i32_e64 s4, 5, v76 /*v332*/
	v_cmp_gt_i32_e64 s5, 6, v76 /*v332*/
	v_cmp_gt_i32_e64 s6, 7, v76 /*v332*/
	v_cndmask_b32_e64 v242 /*v498*/, v34 /*v290*/, 0xff800000, s7
	v_add_nc_u32_e32 v34 /*v290*/, 16, v76 /*v332*/
	v_cmp_gt_i32_e64 s9, 2, v79 /*v335*/
	v_cndmask_b32_e64 v208 /*v464*/, v26 /*v282*/, 0xff800000, vcc_lo
	v_cndmask_b32_e64 v209 /*v465*/, v27 /*v283*/, 0xff800000, s0
	v_subrev_nc_u32_e32 v26 /*v282*/, 32, v76 /*v332*/
	v_cmp_lt_i32_e64 s15, -1, v34 /*v290*/
	v_subrev_nc_u32_e32 v27 /*v283*/, 48, v76 /*v332*/
	v_cndmask_b32_e64 v252 /*v508*/, v90 /*v346*/, 0xff800000, vcc_lo
	v_cndmask_b32_e64 v253 /*v509*/, v91 /*v347*/, 0xff800000, s0
	v_cndmask_b32_e64 v254 /*v510*/, v92 /*v348*/, 0xff800000, s1
	v_cndmask_b32_e64 v244 /*v500*/, 0xff800000, v98 /*v354*/, s15
	v_cmp_lt_i32_e64 s15, 0, v34 /*v290*/
	v_cndmask_b32_e64 v255 /*v511*/, v93 /*v349*/, 0xff800000, s2
	s_set_vgpr_msb 0x4581
	v_cndmask_b32_e64 v0 /*v512*/, v94 /*v350*/, 0xff800000, s3
	v_cndmask_b32_e64 v1 /*v513*/, v95 /*v351*/, 0xff800000, s4
	s_set_vgpr_msb 0x8145
	v_cndmask_b32_e64 v248 /*v504*/, v96 /*v352*/, 0xff800000, s5
	v_cndmask_b32_e64 v245 /*v501*/, 0xff800000, v99 /*v355*/, s15
	v_cmp_lt_i32_e64 s15, 1, v34 /*v290*/
	v_cndmask_b32_e64 v249 /*v505*/, v97 /*v353*/, 0xff800000, s6
	v_cndmask_b32_e64 v202 /*v458*/, v28 /*v284*/, 0xff800000, s1
	v_cndmask_b32_e64 v203 /*v459*/, v29 /*v285*/, 0xff800000, s2
	v_cndmask_b32_e64 v206 /*v462*/, v30 /*v286*/, 0xff800000, s3
	v_cndmask_b32_e64 v246 /*v502*/, 0xff800000, v100 /*v356*/, s15
	v_cmp_lt_i32_e64 s15, 2, v34 /*v290*/
	v_cndmask_b32_e64 v207 /*v463*/, v31 /*v287*/, 0xff800000, s4
	v_cndmask_b32_e64 v204 /*v460*/, v32 /*v288*/, 0xff800000, s5
	v_cndmask_b32_e64 v205 /*v461*/, v33 /*v289*/, 0xff800000, s6
	v_cmp_gt_i32_e32 vcc_lo, 0, v26 /*v282*/
	v_cndmask_b32_e64 v247 /*v503*/, 0xff800000, v101 /*v357*/, s15
	v_cmp_lt_i32_e64 s15, 3, v34 /*v290*/
	v_cmp_gt_i32_e64 s0, 1, v26 /*v282*/
	v_cmp_gt_i32_e64 s1, 2, v26 /*v282*/
	v_cmp_gt_i32_e64 s2, 3, v26 /*v282*/
	v_cmp_gt_i32_e64 s3, 4, v26 /*v282*/
	v_cndmask_b32_e64 v236 /*v492*/, 0xff800000, v102 /*v358*/, s15
	v_cmp_lt_i32_e64 s15, 4, v34 /*v290*/
	v_cmp_gt_i32_e64 s4, 5, v26 /*v282*/
	v_cmp_gt_i32_e64 s5, 6, v26 /*v282*/
	v_cmp_gt_i32_e64 s6, 7, v26 /*v282*/
	v_cndmask_b32_e64 v240 /*v496*/, v20 /*v276*/, 0xff800000, s9
	v_cndmask_b32_e64 v237 /*v493*/, 0xff800000, v103 /*v359*/, s15
	v_cmp_lt_i32_e64 s15, 5, v34 /*v290*/
	v_subrev_nc_u32_e32 v20 /*v276*/, 64, v76 /*v332*/
	v_add_nc_u32_e32 v26 /*v282*/, 0xffffff90, v76 /*v332*/
	s_set_vgpr_msb 0x4581
	v_cndmask_b32_e64 v2 /*v514*/, v42 /*v298*/, 0xff800000, vcc_lo
	v_cndmask_b32_e64 v4 /*v516*/, v44 /*v300*/, 0xff800000, s1
	s_set_vgpr_msb 0x8145
	v_cndmask_b32_e64 v238 /*v494*/, 0xff800000, v104 /*v360*/, s15
	v_cmp_lt_i32_e64 s15, 6, v34 /*v290*/
	v_cndmask_b32_e64 v98 /*v354*/, v46 /*v302*/, 0xff800000, s3
	v_cndmask_b32_e64 v224 /*v480*/, v66 /*v322*/, 0xff800000, vcc_lo
	v_cndmask_b32_e64 v226 /*v482*/, v70 /*v326*/, 0xff800000, s3
	v_cmp_gt_i32_e32 vcc_lo, 0, v20 /*v276*/
	v_cndmask_b32_e64 v239 /*v495*/, 0xff800000, v105 /*v361*/, s15
	v_cmp_gt_i32_e64 s15, 0, v27 /*v283*/
	v_cmp_gt_i32_e64 s3, 4, v20 /*v276*/
	s_set_vgpr_msb 0x4581
	v_cndmask_b32_e64 v3 /*v515*/, v43 /*v299*/, 0xff800000, s0
	s_set_vgpr_msb 0x8149
	v_cndmask_b32_e64 v99 /*v355*/, v47 /*v303*/, 0xff800000, s4
	v_cndmask_b32_e64 v225 /*v481*/, v67 /*v323*/, 0xff800000, s0
	v_cndmask_b32_e64 v232 /*v488*/, v106 /*v362*/, 0xff800000, s15
	v_cndmask_b32_e64 v44 /*v300*/, v122 /*v378*/, 0xff800000, s15
	v_cmp_gt_i32_e64 s15, v26 /*v282*/, -1
	v_cndmask_b32_e64 v227 /*v483*/, v71 /*v327*/, 0xff800000, s4
	v_cndmask_b32_e64 v218 /*v474*/, v114 /*v370*/, 0xff800000, vcc_lo
	v_cmp_lt_i32_e64 s0, v20 /*v276*/, 1
	v_cndmask_b32_e64 v114 /*v370*/, v118 /*v374*/, 0xff800000, s3
	v_cmp_lt_i32_e64 s4, v20 /*v276*/, 5
	v_cndmask_b32_e64 v118 /*v374*/, 0xff800000, v18 /*v530*/, s15
	v_cmp_gt_i32_e64 s15, v26 /*v282*/, 0
	v_cmp_lt_i32_e64 s17, v27 /*v283*/, 2
	v_cndmask_b32_e64 v219 /*v475*/, v115 /*v371*/, 0xff800000, s0
	v_cndmask_b32_e64 v115 /*v371*/, v119 /*v375*/, 0xff800000, s4
	v_cmp_lt_i32_e64 s10, v79 /*v335*/, 3
	v_cndmask_b32_e64 v119 /*v375*/, 0xff800000, v19 /*v531*/, s15
	v_cmp_gt_i32_e64 s15, v26 /*v282*/, 1
	v_cndmask_b32_e64 v234 /*v490*/, v108 /*v364*/, 0xff800000, s17
	v_cmp_lt_i32_e64 s18, v27 /*v283*/, 3
	v_cmp_lt_i32_e64 s8, v79 /*v335*/, 1
	v_cmp_lt_i32_e64 s11, v79 /*v335*/, 4
	v_cndmask_b32_e64 v108 /*v364*/, 0xff800000, v20 /*v532*/, s15
	v_cmp_gt_i32_e64 s15, v26 /*v282*/, 2
	v_cmp_lt_i32_e64 s12, v79 /*v335*/, 5
	v_cmp_lt_i32_e64 s13, v79 /*v335*/, 6
	v_cmp_lt_i32_e64 s14, v79 /*v335*/, 7
	v_cndmask_b32_e64 v241 /*v497*/, v21 /*v277*/, 0xff800000, s10
	s_set_vgpr_msb 0x4945
	v_add_nc_u32_e32 v21 /*v277*/, 0xffffffb0, v76 /*v332*/
	v_cndmask_b32_e64 v235 /*v491*/, v109 /*v365*/, 0xff800000, s18
	v_cmp_gt_i32_e64 s19, 4, v27 /*v283*/
	s_set_vgpr_msb 0x4548
	v_cndmask_b32_e64 v109 /*v365*/, 0xff800000, v21 /*v533*/, s15
	s_set_vgpr_msb 0x4845
	v_cmp_lt_i32_e64 s15, 3, v26 /*v282*/
	v_cndmask_b32_e64 v243 /*v499*/, v35 /*v291*/, 0xff800000, s8
	v_cndmask_b32_e64 v230 /*v486*/, v36 /*v292*/, 0xff800000, s9
	v_cndmask_b32_e64 v231 /*v487*/, v37 /*v293*/, 0xff800000, s10
	v_cndmask_b32_e64 v220 /*v476*/, v38 /*v294*/, 0xff800000, s11
	v_cndmask_b32_e64 v221 /*v477*/, v39 /*v295*/, 0xff800000, s12
	v_cndmask_b32_e64 v228 /*v484*/, v40 /*v296*/, 0xff800000, s13
	v_cndmask_b32_e64 v229 /*v485*/, v41 /*v297*/, 0xff800000, s14
	s_set_vgpr_msb 0x4581
	v_cndmask_b32_e64 v5 /*v517*/, v45 /*v301*/, 0xff800000, s2
	s_set_vgpr_msb 0x8145
	v_cndmask_b32_e64 v100 /*v356*/, v48 /*v304*/, 0xff800000, s5
	v_cndmask_b32_e64 v101 /*v357*/, v49 /*v305*/, 0xff800000, s6
	v_cmp_lt_i32_e64 s21, v27 /*v283*/, 6
	v_cmp_lt_i32_e64 s22, v27 /*v283*/, 7
	v_cndmask_b32_e64 v250 /*v506*/, v18 /*v274*/, 0xff800000, s7
	v_cndmask_b32_e64 v251 /*v507*/, v19 /*v275*/, 0xff800000, s8
	v_cndmask_b32_e64 v30 /*v286*/, v22 /*v278*/, 0xff800000, s11
	v_cndmask_b32_e64 v31 /*v287*/, v23 /*v279*/, 0xff800000, s12
	v_cndmask_b32_e64 v24 /*v280*/, v24 /*v280*/, 0xff800000, s13
	v_cndmask_b32_e64 v25 /*v281*/, v25 /*v281*/, 0xff800000, s14
	v_cndmask_b32_e64 v222 /*v478*/, v68 /*v324*/, 0xff800000, s1
	v_cndmask_b32_e64 v223 /*v479*/, v69 /*v325*/, 0xff800000, s2
	v_cndmask_b32_e64 v18 /*v274*/, v72 /*v328*/, 0xff800000, s5
	v_cndmask_b32_e64 v19 /*v275*/, v73 /*v329*/, 0xff800000, s6
	v_cmp_lt_i32_e64 s1, v20 /*v276*/, 2
	v_cmp_lt_i32_e64 s2, v20 /*v276*/, 3
	v_cmp_lt_i32_e64 s5, v20 /*v276*/, 6
	v_cmp_lt_i32_e64 s6, v20 /*v276*/, 7
	v_cmp_lt_i32_e64 s7, v21 /*v277*/, 0
	v_cmp_lt_i32_e64 s8, v21 /*v277*/, 1
	v_cmp_lt_i32_e64 s9, v21 /*v277*/, 2
	v_cmp_lt_i32_e64 s10, v21 /*v277*/, 3
	v_cmp_lt_i32_e64 s11, v21 /*v277*/, 4
	v_cmp_lt_i32_e64 s12, v21 /*v277*/, 5
	v_cmp_lt_i32_e64 s13, v21 /*v277*/, 6
	v_cmp_lt_i32_e64 s14, v21 /*v277*/, 7
	v_add_nc_u32_e32 v21 /*v277*/, 0xffffffa0, v76 /*v332*/
	v_cndmask_b32_e64 v96 /*v352*/, v110 /*v366*/, 0xff800000, s19
	v_cmp_gt_i32_e64 s20, 5, v27 /*v283*/
	s_set_vgpr_msb 0x4548
	v_cndmask_b32_e64 v110 /*v366*/, 0xff800000, v22 /*v534*/, s15
	s_set_vgpr_msb 0x4845
	v_cmp_lt_i32_e64 s15, 4, v26 /*v282*/
	v_cndmask_b32_e64 v48 /*v304*/, v112 /*v368*/, 0xff800000, s21
	v_cndmask_b32_e64 v49 /*v305*/, v113 /*v369*/, 0xff800000, s22
	v_cndmask_b32_e64 v112 /*v368*/, v116 /*v372*/, 0xff800000, s1
	v_cndmask_b32_e64 v113 /*v369*/, v117 /*v373*/, 0xff800000, s2
	v_cndmask_b32_e64 v116 /*v372*/, v120 /*v376*/, 0xff800000, s5
	v_cndmask_b32_e64 v117 /*v373*/, v121 /*v377*/, 0xff800000, s6
	v_cndmask_b32_e64 v35 /*v291*/, v211 /*v467*/, 0xff800000, s0
	v_cndmask_b32_e64 v29 /*v285*/, v215 /*v471*/, 0xff800000, s4
	v_cndmask_b32_e64 v22 /*v278*/, v216 /*v472*/, 0xff800000, s5
	v_cndmask_b32_e64 v23 /*v279*/, v217 /*v473*/, 0xff800000, s6
	v_cmp_gt_i32_e64 s0, 0, v21 /*v277*/
	v_cmp_gt_i32_e64 s4, 4, v21 /*v277*/
	v_cmp_gt_i32_e64 s5, 5, v21 /*v277*/
	v_cmp_gt_i32_e64 s6, 6, v21 /*v277*/
	v_cmp_gt_i32_e64 s16, 1, v27 /*v283*/
	v_cndmask_b32_e64 v97 /*v353*/, v111 /*v367*/, 0xff800000, s20
	v_cndmask_b32_e64 v32 /*v288*/, v212 /*v468*/, 0xff800000, s1
	v_cmp_gt_i32_e64 s1, 1, v21 /*v277*/
	s_set_vgpr_msb 0x4548
	v_cndmask_b32_e64 v111 /*v367*/, 0xff800000, v23 /*v535*/, s15
	s_set_vgpr_msb 0x4845
	v_cmp_lt_i32_e64 s15, 5, v26 /*v282*/
	v_cndmask_b32_e64 v34 /*v290*/, v210 /*v466*/, 0xff800000, vcc_lo
	v_cndmask_b32_e64 v33 /*v289*/, v213 /*v469*/, 0xff800000, s2
	v_cndmask_b32_e64 v28 /*v284*/, v214 /*v470*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s2, 2, v21 /*v277*/
	v_cmp_gt_i32_e64 s3, 3, v21 /*v277*/
	v_cmp_gt_i32_e32 vcc_lo, 7, v21 /*v277*/
	v_cndmask_b32_e64 v102 /*v358*/, v194 /*v450*/, 0xff800000, s7
	s_set_vgpr_msb 0x4542
	v_cndmask_b32_e64 v122 /*v378*/, v10 /*v522*/, 0xff800000, s0
	v_cndmask_b32_e64 v120 /*v376*/, v14 /*v526*/, 0xff800000, s4
	v_cndmask_b32_e64 v121 /*v377*/, v15 /*v527*/, 0xff800000, s5
	v_cndmask_b32_e64 v20 /*v276*/, v16 /*v528*/, 0xff800000, s6
	v_cndmask_b32_e64 v68 /*v324*/, v28 /*v540*/, 0xff800000, s7
	v_cndmask_b32_e64 v46 /*v302*/, v36 /*v548*/, 0xff800000, s0
	v_cndmask_b32_e64 v92 /*v348*/, v40 /*v552*/, 0xff800000, s4
	v_cndmask_b32_e64 v93 /*v349*/, v41 /*v553*/, 0xff800000, s5
	v_cndmask_b32_e64 v90 /*v346*/, v42 /*v554*/, 0xff800000, s6
	s_or_b32 s0, s39, 0x80000000
	s_mov_b64 s[4:5], s[36:37]
	s_mov_b64 s[6:7], s[38:39]
	s_set_vgpr_msb 0x4241
	v_cndmask_b32_e64 v45 /*v301*/, v123 /*v379*/, 0xff800000, s16
	s_set_vgpr_msb 0x414a
	v_cndmask_b32_e64 v123 /*v379*/, v11 /*v523*/, 0xff800000, s1
	v_cndmask_b32_e64 v106 /*v362*/, 0xff800000, v24 /*v536*/, s15
	s_set_vgpr_msb 0x4a46
	v_cmp_lt_i32_e64 s15, 6, v26 /*v282*/
	v_cndmask_b32_e64 v47 /*v303*/, v37 /*v549*/, 0xff800000, s1
	s_mov_b32 s7, s0
	s_add_nc_u64 s[0:1], s[38:39], s[56:57]
	s_set_vgpr_msb 0x4641
	v_cndmask_b32_e64 v103 /*v359*/, v195 /*v451*/, 0xff800000, s8
	v_cndmask_b32_e64 v104 /*v360*/, v196 /*v452*/, 0xff800000, s9
	v_cndmask_b32_e64 v105 /*v361*/, v197 /*v453*/, 0xff800000, s10
	v_cndmask_b32_e64 v70 /*v326*/, v198 /*v454*/, 0xff800000, s11
	v_cndmask_b32_e64 v71 /*v327*/, v199 /*v455*/, 0xff800000, s12
	v_cndmask_b32_e64 v72 /*v328*/, v200 /*v456*/, 0xff800000, s13
	v_cndmask_b32_e64 v73 /*v329*/, v201 /*v457*/, 0xff800000, s14
	v_cndmask_b32_e64 v40 /*v296*/, v124 /*v380*/, 0xff800000, s17
	v_cndmask_b32_e64 v41 /*v297*/, v125 /*v381*/, 0xff800000, s18
	v_cndmask_b32_e64 v38 /*v294*/, v126 /*v382*/, 0xff800000, s19
	v_cndmask_b32_e64 v39 /*v295*/, v127 /*v383*/, 0xff800000, s20
	v_cndmask_b32_e64 v36 /*v292*/, v128 /*v384*/, 0xff800000, s21
	v_cndmask_b32_e64 v37 /*v293*/, v129 /*v385*/, 0xff800000, s22
	s_set_vgpr_msb 0x4142
	v_cndmask_b32_e64 v124 /*v380*/, v12 /*v524*/, 0xff800000, s2
	v_cndmask_b32_e64 v125 /*v381*/, v13 /*v525*/, 0xff800000, s3
	v_cndmask_b32_e64 v21 /*v277*/, v17 /*v529*/, 0xff800000, vcc_lo
	v_cndmask_b32_e64 v69 /*v325*/, v29 /*v541*/, 0xff800000, s8
	v_cndmask_b32_e64 v66 /*v322*/, v30 /*v542*/, 0xff800000, s9
	v_cndmask_b32_e64 v67 /*v323*/, v31 /*v543*/, 0xff800000, s10
	v_cndmask_b32_e64 v26 /*v282*/, v32 /*v544*/, 0xff800000, s11
	v_cndmask_b32_e64 v27 /*v283*/, v33 /*v545*/, 0xff800000, s12
	v_cndmask_b32_e64 v42 /*v298*/, v38 /*v550*/, 0xff800000, s2
	v_cndmask_b32_e64 v43 /*v299*/, v39 /*v551*/, 0xff800000, s3
	s_add_co_i32 s2, s37, 0x2400
	s_or_b32 s3, s1, 0x80000000
	s_mov_b64 s[8:9], s[36:37]
	v_cndmask_b32_e64 v94 /*v350*/, v34 /*v546*/, 0xff800000, s13
	s_mov_b64 s[10:11], s[38:39]
	s_add_nc_u64 s[12:13], s[0:1], s[56:57]
	s_mov_b32 s9, s2
	s_mov_b32 s10, s0
	s_mov_b32 s11, s3
	s_mov_b64 s[0:1], s[36:37]
	s_set_vgpr_msb 0x4249
	v_cndmask_b32_e64 v233 /*v489*/, v107 /*v363*/, 0xff800000, s16
	v_cndmask_b32_e64 v107 /*v363*/, 0xff800000, v25 /*v537*/, s15
	s_set_vgpr_msb 0x4942
	v_cndmask_b32_e64 v95 /*v351*/, v35 /*v547*/, 0xff800000, s14
	s_add_co_i32 s14, s37, 0x4800
	s_or_b32 s15, s13, 0x80000000
	s_mov_b64 s[2:3], s[38:39]
	s_add_nc_u64 s[38:39], s[12:13], s[56:57]
	v_cndmask_b32_e64 v91 /*v347*/, v43 /*v555*/, 0xff800000, vcc_lo
	s_mov_b32 s1, s14
	s_mov_b32 s2, s12
	s_mov_b32 s3, s15
	s_addk_co_i32 s37, 0x6c00
	s_bitset1_b32 s39, 31
	s_set_vgpr_msb 0x4205
	v_wmma_f32_16x16x32_bf16 v[218:225], v[170:177] /*v[426:433]*/, v[58:65] /*v[314:321]*/, v[218:225]
	v_readfirstlane_b32 s12, v74 /*v330*/
	v_readfirstlane_b32 s13, v75 /*v331*/
	s_set_vgpr_msb 0x500
	v_readfirstlane_b32 s14, v250
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s15, v77 /*v333*/
	v_readfirstlane_b32 s16, v78 /*v334*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s17, v251
	s_set_vgpr_msb 21
	v_readfirstlane_b32 s18, v80 /*v336*/
	v_readfirstlane_b32 s19, v81 /*v337*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[4:7], s[12:19]
	tensor_load_to_lds s[8:11], s[88:95]
	v_max3_num_f32 v250, v252 /*v508*/, v253 /*v509*/, v254 /*v510*/
	s_set_vgpr_msb 0x152a
	v_max3_num_f32 v251, v2 /*v514*/, v3 /*v515*/, v4 /*v516*/
	s_set_vgpr_msb 0x2a55
	v_max3_num_f32 v126 /*v382*/, v218 /*v474*/, v219 /*v475*/, v112 /*v368*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[186:193], v[162:169] /*v[418:425]*/, v[58:65] /*v[314:321]*/, v[186:193]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v146 /*v658*/ offset:9216
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v146 /*v658*/ offset:13824
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v127 /*v383*/, v122 /*v378*/, v123 /*v379*/, v124 /*v380*/
	s_set_vgpr_msb 0x5509
	v_max3_num_f32 v250, v255 /*v511*/, v0 /*v512*/, v250
	s_set_vgpr_msb 0x906
	v_max3_num_f32 v251, v5 /*v517*/, v98 /*v354*/, v251
	s_set_vgpr_msb 0x605
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[178:185] /*v[434:441]*/, v[58:65] /*v[314:321]*/, v[154:161]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v146 /*v658*/ offset:9248
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v146 /*v658*/ offset:13856
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v126 /*v382*/, v113 /*v369*/, v114 /*v370*/, v126 /*v382*/
	v_max3_num_f32 v127 /*v383*/, v125 /*v381*/, v120 /*v376*/, v127 /*v383*/
	s_set_vgpr_msb 0x5506
	v_max3_num_f32 v250, v1 /*v513*/, v248 /*v504*/, v250
	s_set_vgpr_msb 0x605
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[122:129], v[186:193] /*v[442:449]*/, v[58:65] /*v[314:321]*/, v[122:129]
	s_wait_dscnt 0x4
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v146 /*v658*/ offset:9344
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v146 /*v658*/ offset:13952
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v128 /*v384*/, v242 /*v498*/, v243 /*v499*/, v230 /*v486*/
	v_max3_num_f32 v129 /*v385*/, v232 /*v488*/, v233 /*v489*/, v234 /*v490*/
	s_set_vgpr_msb 0x5595
	v_max3_num_f32 v18 /*v530*/, v102 /*v358*/, v103 /*v359*/, v104 /*v360*/
	v_max3_num_f32 v19 /*v531*/, v118 /*v374*/, v119 /*v375*/, v108 /*v364*/
	s_set_vgpr_msb 0x9505
	v_wmma_f32_16x16x32_bf16 v[210:217], v[138:145] /*v[394:401]*/, v[58:65] /*v[314:321]*/, v[210:217]
	s_set_vgpr_msb 0x582
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v146 /*v658*/ offset:9376
	ds_load_tr16_b128 v[14:17] /*v[526:529]*/, v146 /*v658*/ offset:13984
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v128 /*v384*/, v231 /*v487*/, v220 /*v476*/, v128 /*v384*/
	v_max3_num_f32 v129 /*v385*/, v235 /*v491*/, v96 /*v352*/, v129 /*v385*/
	s_set_vgpr_msb 0x55a5
	v_max3_num_f32 v34 /*v546*/, v105 /*v361*/, v70 /*v326*/, v18 /*v530*/
	v_max3_num_f32 v35 /*v547*/, v109 /*v365*/, v110 /*v366*/, v19 /*v531*/
	s_set_vgpr_msb 0xa505
	v_wmma_f32_16x16x32_bf16 v[178:185], v[130:137] /*v[386:393]*/, v[58:65] /*v[314:321]*/, v[178:185]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x582
	ds_load_tr16_b128 v[18:21] /*v[530:533]*/, v146 /*v658*/ offset:9280
	ds_load_tr16_b128 v[22:25] /*v[534:537]*/, v146 /*v658*/ offset:13888
	s_set_vgpr_msb 0x8295
	v_max3_num_f32 v36 /*v548*/, v244 /*v500*/, v245 /*v501*/, v246 /*v502*/
	v_max3_num_f32 v37 /*v549*/, v250 /*v506*/, v251 /*v507*/, v240 /*v496*/
	v_max3_num_f32 v38 /*v550*/, v44 /*v300*/, v45 /*v301*/, v40 /*v296*/
	v_max3_num_f32 v39 /*v551*/, v68 /*v324*/, v69 /*v325*/, v66 /*v322*/
	s_set_vgpr_msb 0x9505
	v_wmma_f32_16x16x32_bf16 v[146:153], v[154:161] /*v[410:417]*/, v[58:65] /*v[314:321]*/, v[146:153]
	s_set_vgpr_msb 0x582
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v146 /*v658*/ offset:9312
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v146 /*v658*/ offset:13920
	s_set_vgpr_msb 0x82a5
	v_max3_num_f32 v36 /*v548*/, v247 /*v503*/, v236 /*v492*/, v36 /*v548*/
	v_max3_num_f32 v37 /*v549*/, v241 /*v497*/, v30 /*v286*/, v37 /*v549*/
	v_max3_num_f32 v38 /*v550*/, v41 /*v297*/, v38 /*v294*/, v38 /*v550*/
	v_max3_num_f32 v39 /*v551*/, v67 /*v323*/, v26 /*v282*/, v39 /*v551*/
	s_set_vgpr_msb 0xa505
	v_wmma_f32_16x16x32_bf16 v[114:121], v[146:153] /*v[402:409]*/, v[58:65] /*v[314:321]*/, v[114:121]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v146 /*v658*/ offset:9408
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v146 /*v658*/ offset:14016
	s_set_vgpr_msb 0x4295
	v_max3_num_f32 v40 /*v552*/, v208 /*v464*/, v209 /*v465*/, v202 /*v458*/
	v_max3_num_f32 v41 /*v553*/, v224 /*v480*/, v225 /*v481*/, v222 /*v478*/
	v_max3_num_f32 v42 /*v554*/, v34 /*v290*/, v35 /*v291*/, v32 /*v288*/
	v_max3_num_f32 v43 /*v555*/, v46 /*v302*/, v47 /*v303*/, v42 /*v298*/
	s_set_vgpr_msb 0x9505
	v_wmma_f32_16x16x32_bf16 v[202:209], v[170:177] /*v[426:433]*/, v[50:57] /*v[306:313]*/, v[202:209]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v146 /*v658*/ offset:9440
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v146 /*v658*/ offset:14048
	s_set_vgpr_msb 0x42a5
	v_max3_num_f32 v40 /*v552*/, v203 /*v459*/, v206 /*v462*/, v40 /*v552*/
	v_max3_num_f32 v41 /*v553*/, v223 /*v479*/, v226 /*v482*/, v41 /*v553*/
	v_max3_num_f32 v42 /*v554*/, v33 /*v289*/, v28 /*v284*/, v42 /*v554*/
	v_max3_num_f32 v43 /*v555*/, v43 /*v299*/, v92 /*v348*/, v43 /*v555*/
	s_set_vgpr_msb 0xa505
	v_wmma_f32_16x16x32_bf16 v[170:177], v[162:169] /*v[418:425]*/, v[50:57] /*v[306:313]*/, v[170:177]
	v_max3_num_f32 v251, v99 /*v355*/, v100 /*v356*/, v251
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x565
	v_max3_num_f32 v162 /*v418*/, v237 /*v493*/, v238 /*v494*/, v36 /*v548*/
	v_max3_num_f32 v163 /*v419*/, v31 /*v287*/, v24 /*v280*/, v37 /*v549*/
	v_max3_num_f32 v164 /*v420*/, v39 /*v295*/, v36 /*v292*/, v38 /*v550*/
	v_max3_num_f32 v165 /*v421*/, v207 /*v463*/, v204 /*v460*/, v40 /*v552*/
	s_set_vgpr_msb 0x6505
	v_wmma_f32_16x16x32_bf16 v[138:145], v[178:185] /*v[434:441]*/, v[50:57] /*v[306:313]*/, v[138:145]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v126 /*v382*/, v115 /*v371*/, v116 /*v372*/, v126 /*v382*/
	v_max3_num_f32 v128 /*v384*/, v221 /*v477*/, v228 /*v484*/, v128 /*v384*/
	v_max3_num_f32 v129 /*v385*/, v97 /*v353*/, v48 /*v304*/, v129 /*v385*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v166 /*v422*/, v227 /*v483*/, v18 /*v274*/, v41 /*v553*/
	s_set_vgpr_msb 0x6505
	v_wmma_f32_16x16x32_bf16 v[106:113], v[186:193] /*v[442:449]*/, v[50:57] /*v[306:313]*/, v[106:113]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v127 /*v383*/, v121 /*v377*/, v20 /*v276*/, v127 /*v383*/
	s_set_vgpr_msb 0x5511
	v_max3_num_f32 v250, v249 /*v505*/, v250, v252 /*v508*/
	s_set_vgpr_msb 0x1121
	v_max3_num_f32 v251, v101 /*v357*/, v251, v2 /*v514*/
	s_set_vgpr_msb 0x2155
	v_max3_num_f32 v126 /*v382*/, v117 /*v373*/, v126 /*v382*/, v218 /*v474*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v167 /*v423*/, v71 /*v327*/, v72 /*v328*/, v34 /*v546*/
	s_set_vgpr_msb 0x6505
	v_wmma_f32_16x16x32_bf16 v[194:201], v[138:145] /*v[394:401]*/, v[50:57] /*v[306:313]*/, v[194:201]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v127 /*v383*/, v21 /*v277*/, v127 /*v383*/, v122 /*v378*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v138 /*v394*/, v111 /*v367*/, v106 /*v362*/, v35 /*v547*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v128 /*v384*/, v229 /*v485*/, v128 /*v384*/, v242 /*v498*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v139 /*v395*/, v27 /*v283*/, v94 /*v350*/, v39 /*v551*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v140 /*v396*/, v239 /*v495*/, v162 /*v418*/, v244 /*v500*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[162:169], v[130:137] /*v[386:393]*/, v[50:57] /*v[306:313]*/, v[162:169]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v134 /*v390*/, v49 /*v305*/, v129 /*v385*/, v232 /*v488*/
	v_max3_num_f32 v135 /*v391*/, v25 /*v281*/, v163 /*v419*/, v250 /*v506*/
	v_max3_num_f32 v129 /*v385*/, v37 /*v293*/, v164 /*v420*/, v44 /*v300*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v130 /*v386*/, v29 /*v285*/, v22 /*v278*/, v42 /*v554*/
	v_max3_num_f32 v131 /*v387*/, v93 /*v349*/, v90 /*v346*/, v43 /*v555*/
	s_set_vgpr_msb 0x6505
	v_wmma_f32_16x16x32_bf16 v[130:137], v[154:161] /*v[410:417]*/, v[50:57] /*v[306:313]*/, v[130:137]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v136 /*v392*/, v95 /*v351*/, v139 /*v395*/, v68 /*v324*/
	v_max3_num_f32 v132 /*v388*/, v205 /*v461*/, v165 /*v421*/, v208 /*v464*/
	v_max3_num_f32 v137 /*v393*/, v19 /*v275*/, v166 /*v422*/, v224 /*v480*/
	v_max3_num_f32 v133 /*v389*/, v73 /*v329*/, v167 /*v423*/, v102 /*v358*/
	v_max3_num_f32 v138 /*v394*/, v107 /*v363*/, v138 /*v394*/, v118 /*v374*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[98:105], v[146:153] /*v[402:409]*/, v[50:57] /*v[306:313]*/, v[98:105]
	s_set_vgpr_msb 0x555
	v_max3_num_f32 v130 /*v386*/, v23 /*v279*/, v130 /*v386*/, v34 /*v290*/
	s_wait_dscnt 0xc
	v_max3_num_f32 v139 /*v395*/, v91 /*v347*/, v131 /*v387*/, v46 /*v302*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[218:225], v[74:81] /*v[330:337]*/, v[2:9] /*v[258:265]*/, v[218:225]
	tensor_load_to_lds s[0:3], s[48:55]
	tensor_load_to_lds s[36:39], s[24:31]
	s_set_vgpr_msb 0x510
	v_max3_num_f32 v250, v250, v251, v126 /*v382*/
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[186:193], v[194:201] /*v[450:457]*/, v[2:9] /*v[258:265]*/, v[186:193]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v146 /*v658*/ offset:18432
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v146 /*v658*/ offset:23040
	s_set_vgpr_msb 0x4204
	v_max3_num_f32 v250, v250, v127 /*v383*/, v251
	s_set_vgpr_msb 0x415
	v_max3_num_f32 v251, v128 /*v384*/, v134 /*v390*/, v133 /*v389*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v140 /*v396*/, v140 /*v396*/, v135 /*v391*/, v129 /*v385*/
	v_max3_num_f32 v141 /*v397*/, v132 /*v388*/, v137 /*v393*/, v130 /*v386*/
	s_set_vgpr_msb 0x5506
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[18:25] /*v[530:537]*/, v[2:9] /*v[258:265]*/, v[154:161]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[126:129] /*v[382:385]*/, v146 /*v658*/ offset:18464
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v146 /*v658*/ offset:23072
	v_add_f32_e32 v166 /*v422*/, 0, v250
	s_set_vgpr_msb 0x4214
	v_max3_num_f32 v251, v251, v138 /*v394*/, v134 /*v390*/
	s_set_vgpr_msb 0x1455
	v_max3_num_f32 v167 /*v423*/, v140 /*v396*/, v136 /*v392*/, v135 /*v391*/
	v_max3_num_f32 v168 /*v424*/, v141 /*v397*/, v139 /*v395*/, v137 /*v393*/
	s_set_vgpr_msb 0x5506
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[122:129], v[26:33] /*v[538:545]*/, v[2:9] /*v[258:265]*/, v[122:129]
	s_wait_dscnt 0x4
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v146 /*v658*/ offset:18560
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v146 /*v658*/ offset:23168
	v_add_f32_e32 v169 /*v425*/, 0, v251
	s_set_vgpr_msb 0x4245
	v_permlanex16_b32 v166 /*v422*/, v166 /*v422*/, s63, 0xfedcba98
	v_add_f32_e32 v178 /*v434*/, 0, v167 /*v423*/
	v_add_f32_e32 v179 /*v435*/, 0, v168 /*v424*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[210:217], v[210:217] /*v[466:473]*/, v[2:9] /*v[258:265]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v146 /*v658*/ offset:18592
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v146 /*v658*/ offset:23200
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v169 /*v425*/, v169 /*v425*/, s63, 0xfedcba98
	v_mul_f32_e32 v180 /*v436*/, s66, v254
	s_set_vgpr_msb 0x4106
	v_wmma_f32_16x16x32_bf16 v[178:185], v[10:17] /*v[522:529]*/, v[2:9] /*v[258:265]*/, v[178:185]
	s_set_vgpr_msb 0x642
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v146 /*v658*/ offset:18496
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v146 /*v658*/ offset:23104
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v178 /*v434*/, v178 /*v434*/, s63, 0xfedcba98
	v_mul_f32_e32 v181 /*v437*/, s66, v252
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[146:153], v[58:65] /*v[314:321]*/, v[2:9] /*v[258:265]*/, v[146:153]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v146 /*v658*/ offset:18528
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v146 /*v658*/ offset:23136
	s_set_vgpr_msb 0x4245
	v_max3_num_f32 v167 /*v423*/, v167 /*v423*/, v178 /*v434*/, v252
	s_set_vgpr_msb 0x4504
	v_max3_num_f32 v250, v250, v166 /*v422*/, v254
	v_max3_num_f32 v251, v251, v169 /*v425*/, v254
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[114:121], v[170:177] /*v[426:433]*/, v[2:9] /*v[258:265]*/, v[114:121]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v146 /*v658*/ offset:18624
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v146 /*v658*/ offset:23232
	s_set_vgpr_msb 0x4241
	v_permlanex16_b32 v179 /*v435*/, v179 /*v435*/, s63, 0xfedcba98
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[202:209], v[74:81] /*v[330:337]*/, v[10:17] /*v[266:273]*/, v[202:209]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v146 /*v658*/ offset:18656
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v146 /*v658*/ offset:23264
	s_set_vgpr_msb 0x4205
	v_max3_num_f32 v252, v168 /*v424*/, v179 /*v435*/, v252
	v_wmma_f32_16x16x32_bf16 v[170:177], v[194:201] /*v[450:457]*/, v[10:17] /*v[266:273]*/, v[170:177]
	s_set_vgpr_msb 0x510
	v_max3_num_f32 v254, v250, v251, v180 /*v436*/
	s_set_vgpr_msb 0x1011
	v_max3_num_f32 v252, v167 /*v423*/, v252, v181 /*v437*/
	s_set_vgpr_msb 0x1106
	v_wmma_f32_16x16x32_bf16 v[138:145], v[18:25] /*v[530:537]*/, v[10:17] /*v[266:273]*/, v[138:145]
	s_set_vgpr_msb 0x610
	v_fma_f32 v250, -v254, s66, v180 /*v436*/
	v_fma_f32 v251, -v252, s66, v181 /*v437*/
	s_set_vgpr_msb 0x1006
	v_wmma_f32_16x16x32_bf16 v[106:113], v[26:33] /*v[538:545]*/, v[10:17] /*v[266:273]*/, v[106:113]
	s_set_vgpr_msb 0x680
	v_exp_f32_e32 v74 /*v586*/, v250
	v_exp_f32_e32 v72 /*v584*/, v251
	s_set_vgpr_msb 0x8005
	v_wmma_f32_16x16x32_bf16 v[194:201], v[210:217] /*v[466:473]*/, v[10:17] /*v[266:273]*/, v[194:201]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[162:169], v[10:17] /*v[522:529]*/, v[10:17] /*v[266:273]*/, v[162:169]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[130:137], v[58:65] /*v[314:321]*/, v[10:17] /*v[266:273]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[170:177] /*v[426:433]*/, v[10:17] /*v[266:273]*/, v[98:105]
	s_set_vgpr_msb 0x58a
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[6:7] /*v[518:519]*/, v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_wait_dscnt 0xc
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[8:9] /*v[520:521]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8a01
	v_wmma_f32_16x16x32_bf16 v[218:225], v[50:57] /*v[306:313]*/, v[234:241], v[218:225]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v146 /*v658*/ offset:27648
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v146 /*v658*/ offset:32256
	s_set_vgpr_msb 0x4200
	v_mul_f32_e64 v250, s66, -v254
	s_set_vgpr_msb 0x41
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[252:253] /*v[508:509]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[252:253] /*v[508:509]*/, v[254:255] /*v[510:511]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4142
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4201
	v_wmma_f32_16x16x32_bf16 v[186:193], v[126:133] /*v[382:389]*/, v[234:241], v[186:193]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v146 /*v658*/ offset:27680
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v146 /*v658*/ offset:32288
	s_set_vgpr_msb 0x4281
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[248:249] /*v[504:505]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8182
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[2:3] /*v[514:515]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[122:123] /*v[634:635]*/, v[4:5] /*v[516:517]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8201
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[154:161], v[150:157] /*v[406:413]*/, v[234:241], v[154:161]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v146 /*v658*/ offset:27776
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v146 /*v658*/ offset:32384
	s_set_vgpr_msb 0x4241
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[242:243] /*v[498:499]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x41a1
	v_mul_f32_e64 v2 /*v514*/, s66, -v252
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[244:245] /*v[500:501]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[6:7] /*v[518:519]*/, v[246:247] /*v[502:503]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[122:129], v[158:165] /*v[414:421]*/, v[234:241], v[122:129]
	s_wait_dscnt 0x6
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v146 /*v658*/ offset:27808
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v146 /*v658*/ offset:32416
	s_set_vgpr_msb 0x4281
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[230:231] /*v[486:487]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[210:217], v[134:141] /*v[390:397]*/, v[234:241], v[210:217]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v146 /*v658*/ offset:27712
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v146 /*v658*/ offset:32320
	s_set_vgpr_msb 0x42a1
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[250:251] /*v[506:507]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa141
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[250:251] /*v[506:507]*/, v[228:229] /*v[484:485]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[178:185], v[142:149] /*v[398:405]*/, v[234:241], v[178:185]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v146 /*v658*/ offset:27744
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v146 /*v658*/ offset:32352
	s_set_vgpr_msb 0x42a1
	v_pk_fma_f32 v[118:119] /*v[630:631]*/, v[240:241] /*v[496:497]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa181
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[232:233] /*v[488:489]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[234:235] /*v[490:491]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[146:153], v[2:9] /*v[258:265]*/, v[234:241], v[146:153]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v146 /*v658*/ offset:27840
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v146 /*v658*/ offset:32448
	s_set_vgpr_msb 0x4261
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[202:203] /*v[458:459]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x61a1
	v_pk_fma_f32 v[28:29] /*v[540:541]*/, v[204:205] /*v[460:461]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[114:121], v[74:81] /*v[330:337]*/, v[234:241], v[114:121]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v146 /*v658*/ offset:27872
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v146 /*v658*/ offset:32480
	s_set_vgpr_msb 0x42a1
	v_pk_fma_f32 v[142:143] /*v[654:655]*/, v[224:225] /*v[480:481]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[120:121] /*v[632:633]*/, v[222:223] /*v[478:479]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[226:227] /*v[482:483]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[202:209], v[50:57] /*v[306:313]*/, v[242:249], v[202:209]
	s_set_vgpr_msb 0x181
	v_pk_fma_f32 v[130:131] /*v[642:643]*/, v[98:99] /*v[354:355]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[128:129] /*v[640:641]*/, v[100:101] /*v[356:357]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[116:117] /*v[628:629]*/, v[218:219] /*v[474:475]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[114:115] /*v[626:627]*/, v[112:113] /*v[368:369]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[112:113] /*v[624:625]*/, v[114:115] /*v[370:371]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[110:111] /*v[622:623]*/, v[116:117] /*v[372:373]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[170:177], v[126:133] /*v[382:389]*/, v[242:249], v[170:177]
	s_set_vgpr_msb 0x181
	v_pk_fma_f32 v[108:109] /*v[620:621]*/, v[122:123] /*v[378:379]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[106:107] /*v[618:619]*/, v[124:125] /*v[380:381]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[10:11] /*v[522:523]*/, v[120:121] /*v[376:377]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[134:135] /*v[646:647]*/, v[96:97] /*v[352:353]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[138:145], v[150:157] /*v[406:413]*/, v[242:249], v[138:145]
	s_set_vgpr_msb 0x181
	v_pk_fma_f32 v[132:133] /*v[644:645]*/, v[48:49] /*v[304:305]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[126:127] /*v[638:639]*/, v[102:103] /*v[358:359]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[100:101] /*v[612:613]*/, v[104:105] /*v[360:361]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[70:71] /*v[326:327]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[86:87] /*v[598:599]*/, v[72:73] /*v[328:329]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[84:85] /*v[596:597]*/, v[118:119] /*v[374:375]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[106:113], v[158:165] /*v[414:421]*/, v[242:249], v[106:113]
	s_set_vgpr_msb 0x181
	v_pk_fma_f32 v[80:81] /*v[592:593]*/, v[108:109] /*v[364:365]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[50:51] /*v[562:563]*/, v[110:111] /*v[366:367]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[106:107] /*v[362:363]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x81a1
	v_pk_fma_f32 v[124:125] /*v[636:637]*/, v[30:31] /*v[286:287]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[104:105] /*v[616:617]*/, v[24:25] /*v[280:281]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[194:201], v[134:141] /*v[390:397]*/, v[242:249], v[194:201]
	s_set_vgpr_msb 0x1a1
	v_pk_fma_f32 v[102:103] /*v[614:615]*/, v[44:45] /*v[300:301]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[98:99] /*v[610:611]*/, v[40:41] /*v[296:297]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[82:83] /*v[594:595]*/, v[38:39] /*v[294:295]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[36:37] /*v[292:293]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[60:61] /*v[572:573]*/, v[68:69] /*v[324:325]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[22:23] /*v[534:535]*/, v[66:67] /*v[322:323]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[162:169], v[142:149] /*v[398:405]*/, v[242:249], v[162:169]
	s_set_vgpr_msb 0x1a1
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[18:19] /*v[274:275]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[34:35] /*v[290:291]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[32:33] /*v[288:289]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[130:137], v[2:9] /*v[258:265]*/, v[242:249], v[130:137]
	s_set_vgpr_msb 0x1a1
	v_pk_fma_f32 v[78:79] /*v[590:591]*/, v[28:29] /*v[284:285]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[22:23] /*v[278:279]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[26:27] /*v[538:539]*/, v[46:47] /*v[302:303]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[42:43] /*v[298:299]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[98:105], v[74:81] /*v[330:337]*/, v[242:249], v[98:105]
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
	s_set_vgpr_msb 0x181
	v_pk_fma_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[276:277]*/, v[0:1], v[250:251] op_sel_hi:[1,1,0]
	s_wait_dscnt 0xc
	s_barrier_wait -1
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[218:225], v[166:173] /*v[422:429]*/, v[226:233], v[218:225]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[66:69] /*v[322:325]*/, v153 /*v665*/
	ds_load_b128 v[70:73] /*v[326:329]*/, v153 /*v665*/ offset:32
	ds_load_b128 v[58:61] /*v[314:317]*/, v153 /*v665*/ offset:64
	s_set_vgpr_msb 0x4281
	v_exp_f32_e32 v40 /*v552*/, v10 /*v266*/
	v_exp_f32_e32 v41 /*v553*/, v11 /*v267*/
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[186:193], v[174:181] /*v[430:437]*/, v[226:233], v[186:193]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[62:65] /*v[318:321]*/, v153 /*v665*/ offset:96
	ds_load_b128 v[42:45] /*v[298:301]*/, v153 /*v665*/ offset:128
	ds_load_b128 v[46:49] /*v[302:305]*/, v153 /*v665*/ offset:160
	s_set_vgpr_msb 0x4281
	v_exp_f32_e32 v42 /*v554*/, v252 /*v508*/
	s_set_vgpr_msb 0x8101
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[154:161], v[210:217] /*v[466:473]*/, v[226:233], v[154:161]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[18:21] /*v[274:277]*/, v153 /*v665*/ offset:6400
	ds_load_b128 v[22:25] /*v[278:281]*/, v153 /*v665*/ offset:6432
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[10:13] /*v[266:269]*/, v153 /*v665*/ offset:6464
	s_set_vgpr_msb 0x4281
	v_exp_f32_e32 v36 /*v548*/, v14 /*v270*/
	v_exp_f32_e32 v37 /*v549*/, v15 /*v271*/
	s_set_vgpr_msb 0x8101
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[122:129], v[242:249] /*v[498:505]*/, v[226:233], v[122:129]
	s_wait_dscnt 0x9
	s_set_vgpr_msb 0x142
	ds_load_b128 v[14:17] /*v[270:273]*/, v153 /*v665*/ offset:6496
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[242:245], v153 /*v665*/ offset:6528
	ds_load_b128 v[246:249], v153 /*v665*/ offset:6560
	s_set_vgpr_msb 0x282
	v_exp_f32_e32 v38 /*v550*/, v8 /*v520*/
	v_exp_f32_e32 v39 /*v551*/, v9 /*v521*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[210:217], v[182:189] /*v[438:445]*/, v[226:233], v[210:217]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[50:53] /*v[306:309]*/, v153 /*v665*/ offset:192
	ds_load_b128 v[54:57] /*v[310:313]*/, v153 /*v665*/ offset:224
	ds_load_b128 v[34:37] /*v[290:293]*/, v153 /*v665*/ offset:256
	s_set_vgpr_msb 0x42a1
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[26:27] /*v[282:283]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa182
	v_exp_f32_e32 v14 /*v526*/, v4 /*v516*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[178:185], v[190:197] /*v[446:453]*/, v[226:233], v[178:185]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[38:41] /*v[294:297]*/, v153 /*v665*/ offset:288
	ds_load_b128 v[26:29] /*v[282:285]*/, v153 /*v665*/ offset:320
	ds_load_b128 v[30:33] /*v[286:289]*/, v153 /*v665*/ offset:352
	s_set_vgpr_msb 0x42a1
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[94:95] /*v[350:351]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0xa182
	v_exp_f32_e32 v15 /*v527*/, v5 /*v517*/
	s_set_vgpr_msb 0x8201
	v_wmma_f32_16x16x32_bf16 v[146:153], v[228:235] /*v[484:491]*/, v[226:233], v[146:153]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[2:5] /*v[258:261]*/, v153 /*v665*/ offset:6592
	ds_load_b128 v[6:9] /*v[262:265]*/, v153 /*v665*/ offset:6624
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[234:237], v153 /*v665*/ offset:6656
	s_set_vgpr_msb 0x2a1
	v_pk_fma_f32 v[76:77] /*v[588:589]*/, v[92:93] /*v[348:349]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_exp_f32_e32 v12 /*v524*/, v208 /*v464*/
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[114:121], v[198:205] /*v[454:461]*/, v[226:233], v[114:121]
	s_set_vgpr_msb 0x102
	ds_load_b128 v[238:241], v153 /*v665*/ offset:6688
	ds_load_b128 v[226:229], v153 /*v665*/ offset:6720
	ds_load_b128 v[230:233], v153 /*v665*/ offset:6752
	s_set_vgpr_msb 0x2a1
	v_pk_fma_f32 v[58:59] /*v[570:571]*/, v[90:91] /*v[346:347]*/, v[0:1], v[2:3] /*v[514:515]*/ op_sel_hi:[1,1,0]
	v_exp_f32_e32 v13 /*v525*/, v209 /*v465*/
	s_set_vgpr_msb 0xa105
	v_wmma_f32_16x16x32_bf16 v[202:209], v[166:173] /*v[422:429]*/, v[82:89] /*v[338:345]*/, v[202:209]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v43 /*v555*/, v253 /*v509*/
	v_exp_f32_e32 v52 /*v564*/, v254 /*v510*/
	v_exp_f32_e32 v53 /*v565*/, v255 /*v511*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[170:177], v[174:181] /*v[430:437]*/, v[82:89] /*v[338:345]*/, v[170:177]
	s_set_vgpr_msb 0x582
	v_exp_f32_e32 v54 /*v566*/, v0 /*v512*/
	v_exp_f32_e32 v55 /*v567*/, v1 /*v513*/
	s_set_vgpr_msb 0x8281
	v_exp_f32_e32 v44 /*v556*/, v220 /*v476*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[138:145], v[210:217] /*v[466:473]*/, v[82:89] /*v[338:345]*/, v[138:145]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v45 /*v557*/, v221 /*v477*/
	v_exp_f32_e32 v46 /*v558*/, v250 /*v506*/
	v_exp_f32_e32 v47 /*v559*/, v251 /*v507*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[106:113], v[242:249] /*v[498:505]*/, v[82:89] /*v[338:345]*/, v[106:113]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v18 /*v530*/, v240 /*v496*/
	v_exp_f32_e32 v19 /*v531*/, v241 /*v497*/
	v_exp_f32_e32 v16 /*v528*/, v206 /*v462*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[194:201], v[182:189] /*v[438:445]*/, v[82:89] /*v[338:345]*/, v[194:201]
	s_set_vgpr_msb 0x582
	v_exp_f32_e32 v30 /*v542*/, v6 /*v518*/
	v_exp_f32_e32 v31 /*v543*/, v7 /*v519*/
	s_set_vgpr_msb 0x8281
	v_exp_f32_e32 v32 /*v544*/, v236 /*v492*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[162:169], v[190:197] /*v[446:453]*/, v[82:89] /*v[338:345]*/, v[162:169]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v33 /*v545*/, v237 /*v493*/
	v_exp_f32_e32 v34 /*v546*/, v238 /*v494*/
	v_exp_f32_e32 v35 /*v547*/, v239 /*v495*/
	s_set_vgpr_msb 0x8105
	v_wmma_f32_16x16x32_bf16 v[130:137], v[228:235] /*v[484:491]*/, v[82:89] /*v[338:345]*/, v[130:137]
	s_set_vgpr_msb 0x581
	v_exp_f32_e32 v17 /*v529*/, v207 /*v463*/
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v28 /*v540*/, v28 /*v540*/
	v_exp_f32_e32 v29 /*v541*/, v29 /*v541*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[98:105], v[198:205] /*v[454:461]*/, v[82:89] /*v[338:345]*/, v[98:105]
	s_wait_dscnt 0xc
	s_add_nc_u64 s[46:47], s[46:47], 1
	s_set_vgpr_msb 0x500
	v_add_nc_u32_e32 v253, 0xffffff80, v253
	v_cmp_lt_i64_e64 s0, s[46:47], s[34:35]
	s_addk_co_i32 vcc_hi, 0xff80
	s_add_co_i32 s44, s44, s61
	s_add_co_i32 s42, s42, s41
	s_mov_b32 s26, s96
	s_mov_b32 s29, s40
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s40, s64
	s_cbranch_vccnz .LBB0_12
	s_set_vgpr_msb 1
	v_readlane_b32 s72, v0 /*v256*/, 27
	v_readlane_b32 s75, v0 /*v256*/, 30
	v_readlane_b32 s76, v0 /*v256*/, 31
	v_readlane_b32 s73, v0 /*v256*/, 28
	v_readlane_b32 s74, v0 /*v256*/, 29
	v_readlane_b32 s77, v1 /*v257*/, 0
	v_readlane_b32 s78, v1 /*v257*/, 1
	v_readlane_b32 s79, v1 /*v257*/, 2
	s_set_vgpr_msb 0x100
	s_branch .LBB0_15
.LBB0_14:
	s_mov_b32 s96, s26
.LBB0_15:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v154 /*v666*/, s33, v152 /*v664*/
	;;#ASMSTART
	s_wait_idle
	;;#ASMEND
	s_barrier_signal -1
	v_readfirstlane_b32 s0, v0
	s_set_vgpr_msb 0x8802
	v_exp_f32_e32 v0, v136 /*v648*/
	v_exp_f32_e32 v1, v137 /*v649*/
	s_set_vgpr_msb 0x242
	v_exp_f32_e32 v138 /*v394*/, v122 /*v634*/
	v_exp_f32_e32 v139 /*v395*/, v123 /*v635*/
	v_exp_f32_e32 v136 /*v392*/, v130 /*v642*/
	v_exp_f32_e32 v137 /*v393*/, v131 /*v643*/
	v_exp_f32_e32 v142 /*v398*/, v128 /*v640*/
	v_exp_f32_e32 v143 /*v399*/, v129 /*v641*/
	v_exp_f32_e32 v140 /*v396*/, v116 /*v628*/
	v_exp_f32_e32 v141 /*v397*/, v117 /*v629*/
	v_exp_f32_e32 v146 /*v402*/, v114 /*v626*/
	v_exp_f32_e32 v147 /*v403*/, v115 /*v627*/
	v_exp_f32_e32 v144 /*v400*/, v112 /*v624*/
	v_exp_f32_e32 v145 /*v401*/, v113 /*v625*/
	v_exp_f32_e32 v150 /*v406*/, v110 /*v622*/
	v_exp_f32_e32 v151 /*v407*/, v111 /*v623*/
	v_exp_f32_e32 v148 /*v404*/, v108 /*v620*/
	v_exp_f32_e32 v149 /*v405*/, v109 /*v621*/
	v_exp_f32_e32 v154 /*v410*/, v106 /*v618*/
	v_exp_f32_e32 v155 /*v411*/, v107 /*v619*/
	v_exp_f32_e32 v152 /*v408*/, v10 /*v522*/
	v_exp_f32_e32 v153 /*v409*/, v11 /*v523*/
	v_exp_f32_e32 v158 /*v414*/, v20 /*v532*/
	v_exp_f32_e32 v159 /*v415*/, v21 /*v533*/
	s_set_vgpr_msb 0x4202
	v_exp_f32_e32 v250, v144 /*v656*/
	v_exp_f32_e32 v251, v145 /*v657*/
	s_set_vgpr_msb 0x24a
	v_exp_f32_e32 v112 /*v368*/, v140 /*v652*/
	v_exp_f32_e32 v113 /*v369*/, v141 /*v653*/
	v_exp_f32_e32 v110 /*v366*/, v134 /*v646*/
	v_exp_f32_e32 v111 /*v367*/, v135 /*v647*/
	v_exp_f32_e32 v116 /*v372*/, v132 /*v644*/
	v_exp_f32_e32 v117 /*v373*/, v133 /*v645*/
	v_exp_f32_e32 v114 /*v370*/, v126 /*v638*/
	v_exp_f32_e32 v115 /*v371*/, v127 /*v639*/
	v_exp_f32_e32 v120 /*v376*/, v100 /*v612*/
	v_exp_f32_e32 v121 /*v377*/, v101 /*v613*/
	v_exp_f32_e32 v118 /*v374*/, v96 /*v608*/
	v_exp_f32_e32 v119 /*v375*/, v97 /*v609*/
	v_exp_f32_e32 v124 /*v380*/, v86 /*v598*/
	v_exp_f32_e32 v125 /*v381*/, v87 /*v599*/
	v_exp_f32_e32 v122 /*v378*/, v84 /*v596*/
	v_exp_f32_e32 v123 /*v379*/, v85 /*v597*/
	v_exp_f32_e32 v128 /*v384*/, v80 /*v592*/
	v_exp_f32_e32 v129 /*v385*/, v81 /*v593*/
	v_exp_f32_e32 v126 /*v382*/, v50 /*v562*/
	v_exp_f32_e32 v127 /*v383*/, v51 /*v563*/
	v_exp_f32_e32 v132 /*v388*/, v48 /*v560*/
	v_exp_f32_e32 v133 /*v389*/, v49 /*v561*/
	v_exp_f32_e32 v78 /*v334*/, v138 /*v650*/
	v_exp_f32_e32 v79 /*v335*/, v139 /*v651*/
	v_exp_f32_e32 v90 /*v346*/, v118 /*v630*/
	v_exp_f32_e32 v91 /*v347*/, v119 /*v631*/
	v_exp_f32_e32 v80 /*v336*/, v124 /*v636*/
	v_exp_f32_e32 v81 /*v337*/, v125 /*v637*/
	v_exp_f32_e32 v94 /*v350*/, v104 /*v616*/
	v_exp_f32_e32 v95 /*v351*/, v105 /*v617*/
	v_exp_f32_e32 v92 /*v348*/, v102 /*v614*/
	v_exp_f32_e32 v93 /*v349*/, v103 /*v615*/
	v_exp_f32_e32 v98 /*v354*/, v98 /*v610*/
	v_exp_f32_e32 v99 /*v355*/, v99 /*v611*/
	v_exp_f32_e32 v96 /*v352*/, v82 /*v594*/
	v_exp_f32_e32 v97 /*v353*/, v83 /*v595*/
	v_exp_f32_e32 v102 /*v358*/, v66 /*v578*/
	v_exp_f32_e32 v103 /*v359*/, v67 /*v579*/
	v_exp_f32_e32 v100 /*v356*/, v60 /*v572*/
	v_exp_f32_e32 v101 /*v357*/, v61 /*v573*/
	v_exp_f32_e32 v106 /*v362*/, v22 /*v534*/
	v_exp_f32_e32 v107 /*v363*/, v23 /*v535*/
	v_exp_f32_e32 v104 /*v360*/, v64 /*v576*/
	v_exp_f32_e32 v105 /*v361*/, v65 /*v577*/
	v_exp_f32_e32 v108 /*v364*/, v56 /*v568*/
	v_exp_f32_e32 v109 /*v365*/, v57 /*v569*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[52:53] /*v[564:565]*/, v[54:55] /*v[566:567]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[44:45] /*v[556:557]*/, v[46:47] /*v[558:559]*/
	v_cvt_pk_bf16_f32 v82 /*v338*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v83 /*v339*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v84 /*v340*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v85 /*v341*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v86 /*v342*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v87 /*v343*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v88 /*v344*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v89 /*v345*/, v46 /*v558*/, v47 /*v559*/
	v_cvt_pk_bf16_f32 v74 /*v330*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v75 /*v331*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v76 /*v332*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v77 /*v333*/, v34 /*v546*/, v35 /*v547*/
	s_cmp_lt_u32 s60, 2
	s_set_vgpr_msb 0x4a00
	s_barrier_wait -1
	s_cbranch_scc0 .LBB0_17
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v242 /*v498*/, v0, v1
	s_set_vgpr_msb 0x4045
	v_cvt_pk_bf16_f32 v243 /*v499*/, v138 /*v394*/, v139 /*v395*/
	v_cvt_pk_bf16_f32 v244 /*v500*/, v136 /*v392*/, v137 /*v393*/
	v_cvt_pk_bf16_f32 v245 /*v501*/, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v226 /*v482*/, v140 /*v396*/, v141 /*v397*/
	v_cvt_pk_bf16_f32 v227 /*v483*/, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v228 /*v484*/, v144 /*v400*/, v145 /*v401*/
	v_cvt_pk_bf16_f32 v229 /*v485*/, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v218 /*v474*/, v148 /*v404*/, v149 /*v405*/
	v_cvt_pk_bf16_f32 v219 /*v475*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v220 /*v476*/, v152 /*v408*/, v153 /*v409*/
	v_cvt_pk_bf16_f32 v221 /*v477*/, v158 /*v414*/, v159 /*v415*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[0:1], v[0:1], v[138:139] /*v[394:395]*/
	s_set_vgpr_msb 0x445
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[142:143] /*v[398:399]*/
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[140:141] /*v[396:397]*/, v[146:147] /*v[402:403]*/
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[144:145] /*v[400:401]*/, v[150:151] /*v[406:407]*/
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[148:149] /*v[404:405]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[152:153] /*v[408:409]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[156:157] /*v[412:413]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[0:1], v[0:1], v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x445
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[138:139] /*v[394:395]*/, v[140:141] /*v[396:397]*/
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[142:143] /*v[398:399]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4501
	v_pk_add_f32 v[0:1], v[146:147] /*v[402:403]*/, v[0:1]
	s_set_vgpr_msb 0x145
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[138:139] /*v[394:395]*/
	s_set_vgpr_msb 0x4504
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[0:1], v[0:1], v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x440
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v136 /*v392*/, v1
	s_set_vgpr_msb 0x4004
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[0:1], v[0:1], v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x402
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[0:1], v[70:71] /*v[582:583]*/, v[0:1]
	s_set_vgpr_msb 0x240
	v_cvt_pk_bf16_f32 v246 /*v502*/, v250, v251
	s_set_vgpr_msb 0x4045
	v_cvt_pk_bf16_f32 v247 /*v503*/, v112 /*v368*/, v113 /*v369*/
	v_cvt_pk_bf16_f32 v248 /*v504*/, v110 /*v366*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v249 /*v505*/, v116 /*v372*/, v117 /*v373*/
	v_cvt_pk_bf16_f32 v230 /*v486*/, v114 /*v370*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v231 /*v487*/, v120 /*v376*/, v121 /*v377*/
	v_cvt_pk_bf16_f32 v232 /*v488*/, v118 /*v374*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v233 /*v489*/, v124 /*v380*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v222 /*v478*/, v122 /*v378*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v223 /*v479*/, v128 /*v384*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v224 /*v480*/, v126 /*v382*/, v127 /*v383*/
	v_cvt_pk_bf16_f32 v225 /*v481*/, v132 /*v388*/, v133 /*v389*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[250:251], v[250:251], v[112:113] /*v[368:369]*/
	s_set_vgpr_msb 0x445
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[110:111] /*v[366:367]*/, v[116:117] /*v[372:373]*/
	v_pk_add_f32 v[112:113] /*v[368:369]*/, v[114:115] /*v[370:371]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[114:115] /*v[370:371]*/, v[118:119] /*v[374:375]*/, v[124:125] /*v[380:381]*/
	v_pk_add_f32 v[116:117] /*v[372:373]*/, v[122:123] /*v[378:379]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[118:119] /*v[374:375]*/, v[126:127] /*v[382:383]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[120:121] /*v[376:377]*/, v[130:131] /*v[386:387]*/, v[134:135] /*v[390:391]*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[250:251], v[250:251], v[110:111] /*v[366:367]*/
	s_set_vgpr_msb 0x445
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[112:113] /*v[368:369]*/, v[114:115] /*v[370:371]*/
	v_pk_add_f32 v[112:113] /*v[368:369]*/, v[116:117] /*v[372:373]*/, v[118:119] /*v[374:375]*/
	s_set_vgpr_msb 0x4501
	v_pk_add_f32 v[250:251], v[120:121] /*v[376:377]*/, v[250:251]
	s_set_vgpr_msb 0x145
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[110:111] /*v[366:367]*/, v[112:113] /*v[368:369]*/
	s_set_vgpr_msb 0x4504
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[250:251], v[250:251], v[110:111] /*v[366:367]*/
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v250, v250, v251
	s_set_vgpr_msb 2
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[250:251], v[70:71] /*v[582:583]*/, v[250:251] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x285
	v_cvt_pk_bf16_f32 v2 /*v514*/, v78 /*v334*/, v79 /*v335*/
	v_cvt_pk_bf16_f32 v3 /*v515*/, v90 /*v346*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v4 /*v516*/, v80 /*v336*/, v81 /*v337*/
	v_cvt_pk_bf16_f32 v5 /*v517*/, v94 /*v350*/, v95 /*v351*/
	s_set_vgpr_msb 0x8545
	v_cvt_pk_bf16_f32 v250 /*v506*/, v92 /*v348*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v251 /*v507*/, v98 /*v354*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v252 /*v508*/, v96 /*v352*/, v97 /*v353*/
	v_cvt_pk_bf16_f32 v253 /*v509*/, v102 /*v358*/, v103 /*v359*/
	v_cvt_pk_bf16_f32 v234 /*v490*/, v100 /*v356*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v235 /*v491*/, v106 /*v362*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v236 /*v492*/, v104 /*v360*/, v105 /*v361*/
	v_cvt_pk_bf16_f32 v237 /*v493*/, v108 /*v364*/, v109 /*v365*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[14:15] /*v[526:527]*/, v[30:31] /*v[542:543]*/
	v_pk_add_f32 v[112:113] /*v[368:369]*/, v[32:33] /*v[544:545]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[78:79] /*v[334:335]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, v[94:95] /*v[350:351]*/
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[92:93] /*v[348:349]*/, v[98:99] /*v[354:355]*/
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[96:97] /*v[352:353]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[100:101] /*v[356:357]*/, v[106:107] /*v[362:363]*/
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[110:111] /*v[366:367]*/, v[112:113] /*v[368:369]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[78:79] /*v[334:335]*/, v[80:81] /*v[336:337]*/
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[90:91] /*v[346:347]*/, v[92:93] /*v[348:349]*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[98:99] /*v[354:355]*/, v[78:79] /*v[334:335]*/
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[78:79] /*v[334:335]*/, v[80:81] /*v[336:337]*/
	s_set_vgpr_msb 0x454a
	v_exp_f32_e32 v92 /*v348*/, v142 /*v654*/
	v_exp_f32_e32 v93 /*v349*/, v143 /*v655*/
	v_exp_f32_e32 v94 /*v350*/, v120 /*v632*/
	v_exp_f32_e32 v95 /*v351*/, v121 /*v633*/
	v_exp_f32_e32 v96 /*v352*/, v88 /*v600*/
	v_exp_f32_e32 v97 /*v353*/, v89 /*v601*/
	v_exp_f32_e32 v98 /*v354*/, v94 /*v606*/
	v_exp_f32_e32 v99 /*v355*/, v95 /*v607*/
	v_exp_f32_e32 v100 /*v356*/, v92 /*v604*/
	v_exp_f32_e32 v101 /*v357*/, v93 /*v605*/
	v_exp_f32_e32 v102 /*v358*/, v90 /*v602*/
	v_exp_f32_e32 v103 /*v359*/, v91 /*v603*/
	v_exp_f32_e32 v104 /*v360*/, v78 /*v590*/
	v_exp_f32_e32 v105 /*v361*/, v79 /*v591*/
	v_exp_f32_e32 v106 /*v362*/, v62 /*v574*/
	v_exp_f32_e32 v107 /*v363*/, v63 /*v575*/
	v_exp_f32_e32 v108 /*v364*/, v26 /*v538*/
	v_exp_f32_e32 v109 /*v365*/, v27 /*v539*/
	v_exp_f32_e32 v110 /*v366*/, v24 /*v536*/
	v_exp_f32_e32 v111 /*v367*/, v25 /*v537*/
	v_exp_f32_e32 v112 /*v368*/, v76 /*v588*/
	v_exp_f32_e32 v113 /*v369*/, v77 /*v589*/
	v_exp_f32_e32 v114 /*v370*/, v58 /*v570*/
	v_exp_f32_e32 v115 /*v371*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v78 /*v334*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v79 /*v335*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v80 /*v336*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v81 /*v337*/, v28 /*v540*/, v29 /*v541*/
	s_set_vgpr_msb 0x4a85
	v_cvt_pk_bf16_f32 v6 /*v518*/, v92 /*v348*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v7 /*v519*/, v94 /*v350*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v8 /*v520*/, v96 /*v352*/, v97 /*v353*/
	v_cvt_pk_bf16_f32 v9 /*v521*/, v98 /*v354*/, v99 /*v355*/
	s_set_vgpr_msb 0x8545
	v_cvt_pk_bf16_f32 v254 /*v510*/, v100 /*v356*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v255 /*v511*/, v102 /*v358*/, v103 /*v359*/
	s_set_vgpr_msb 0x4585
	v_cvt_pk_bf16_f32 v0 /*v512*/, v104 /*v360*/, v105 /*v361*/
	v_cvt_pk_bf16_f32 v1 /*v513*/, v106 /*v362*/, v107 /*v363*/
	s_set_vgpr_msb 0x8545
	v_cvt_pk_bf16_f32 v238 /*v494*/, v108 /*v364*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v239 /*v495*/, v110 /*v366*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v240 /*v496*/, v112 /*v368*/, v113 /*v369*/
	v_cvt_pk_bf16_f32 v241 /*v497*/, v114 /*v370*/, v115 /*v371*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[116:117] /*v[372:373]*/, v[12:13] /*v[524:525]*/, v[18:19] /*v[530:531]*/
	v_pk_add_f32 v[118:119] /*v[374:375]*/, v[16:17] /*v[528:529]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[92:93] /*v[348:349]*/, v[94:95] /*v[350:351]*/
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/, v[98:99] /*v[354:355]*/
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[100:101] /*v[356:357]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[104:105] /*v[360:361]*/, v[106:107] /*v[362:363]*/
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[108:109] /*v[364:365]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[112:113] /*v[368:369]*/, v[114:115] /*v[370:371]*/
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[116:117] /*v[372:373]*/, v[118:119] /*v[374:375]*/
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[92:93] /*v[348:349]*/, v[94:95] /*v[350:351]*/
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/, v[98:99] /*v[354:355]*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[100:101] /*v[356:357]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[104:105] /*v[360:361]*/, v[92:93] /*v[348:349]*/
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[92:93] /*v[348:349]*/, v[94:95] /*v[350:351]*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v94 /*v350*/, v90 /*v346*/ :: v_dual_mov_b32 v95 /*v351*/, v92 /*v348*/
	v_mov_b32_e32 v92 /*v348*/, v91 /*v347*/
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[94:95] /*v[350:351]*/, v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x4586
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[146:147] /*v[658:659]*/, v[68:69] /*v[580:581]*/, v[90:91] /*v[346:347]*/
	s_set_vgpr_msb 0x8648
	v_pk_mul_f32 v[96:97] /*v[352:353]*/, v[224:225], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95] /*v[350:351]*/, v[222:223], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93] /*v[348:349]*/, v[220:221], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91] /*v[346:347]*/, v[218:219], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105] /*v[360:361]*/, v[192:193], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103] /*v[358:359]*/, v[190:191], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101] /*v[356:357]*/, v[188:189], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99] /*v[354:355]*/, v[186:187], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113] /*v[368:369]*/, v[160:161], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111] /*v[366:367]*/, v[158:159], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109] /*v[364:365]*/, v[156:157], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107] /*v[362:363]*/, v[154:155], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121] /*v[376:377]*/, v[128:129], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119] /*v[374:375]*/, v[126:127], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117] /*v[372:373]*/, v[124:125], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115] /*v[370:371]*/, v[122:123], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129] /*v[384:385]*/, v[216:217], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127] /*v[382:383]*/, v[214:215], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125] /*v[380:381]*/, v[212:213], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123] /*v[378:379]*/, v[210:211], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[184:185], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[182:183], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[180:181], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[178:179], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[152:153], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[150:151], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[148:149], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[146:147], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[120:121], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[118:119], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[116:117], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[114:115], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[208:209], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[206:207], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[204:205], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[202:203], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[176:177], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[174:175], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[172:173], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[170:171], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[144:145], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[142:143], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[140:141], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[138:139], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[112:113], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[110:111], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[108:109], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[106:107], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[168:169], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[166:167], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[164:165], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[162:163], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[136:137], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[134:135], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[132:133], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[130:131], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[104:105], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[102:103], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[100:101], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[98:99], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4882
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v154 /*v666*/
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v154 /*v666*/ offset:4608
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v154 /*v666*/ offset:32
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v154 /*v666*/ offset:4640
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v154 /*v666*/ offset:9216
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v154 /*v666*/ offset:13824
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v154 /*v666*/ offset:9248
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v154 /*v666*/ offset:13856
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v154 /*v666*/ offset:128
	ds_load_tr16_b128 v[192:195] /*v[704:707]*/, v154 /*v666*/ offset:4736
	ds_load_tr16_b128 v[196:199] /*v[708:711]*/, v154 /*v666*/ offset:160
	ds_load_tr16_b128 v[200:203] /*v[712:715]*/, v154 /*v666*/ offset:4768
	ds_load_tr16_b128 v[204:207] /*v[716:719]*/, v154 /*v666*/ offset:9344
	ds_load_tr16_b128 v[208:211] /*v[720:723]*/, v154 /*v666*/ offset:13952
	ds_load_tr16_b128 v[212:215] /*v[724:727]*/, v154 /*v666*/ offset:9376
	ds_load_tr16_b128 v[216:219] /*v[728:731]*/, v154 /*v666*/ offset:13984
	ds_load_tr16_b128 v[220:223] /*v[732:735]*/, v154 /*v666*/ offset:64
	ds_load_tr16_b128 v[224:227] /*v[736:739]*/, v154 /*v666*/ offset:4672
	ds_load_tr16_b128 v[228:231] /*v[740:743]*/, v154 /*v666*/ offset:96
	ds_load_tr16_b128 v[232:235] /*v[744:747]*/, v154 /*v666*/ offset:4704
	ds_load_tr16_b128 v[236:239] /*v[748:751]*/, v154 /*v666*/ offset:9280
	ds_load_tr16_b128 v[240:243] /*v[752:755]*/, v154 /*v666*/ offset:13888
	ds_load_tr16_b128 v[244:247] /*v[756:759]*/, v154 /*v666*/ offset:9312
	ds_load_tr16_b128 v[248:251] /*v[760:763]*/, v154 /*v666*/ offset:13920
	ds_load_tr16_b128 v[252:255] /*v[764:767]*/, v154 /*v666*/ offset:192
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[0:3] /*v[768:771]*/, v154 /*v666*/ offset:4800
	ds_load_tr16_b128 v[4:7] /*v[772:775]*/, v154 /*v666*/ offset:224
	ds_load_tr16_b128 v[8:11] /*v[776:779]*/, v154 /*v666*/ offset:4832
	ds_load_tr16_b128 v[12:15] /*v[780:783]*/, v154 /*v666*/ offset:9408
	ds_load_tr16_b128 v[16:19] /*v[784:787]*/, v154 /*v666*/ offset:14016
	ds_load_tr16_b128 v[20:23] /*v[788:791]*/, v154 /*v666*/ offset:9440
	ds_load_tr16_b128 v[24:27] /*v[792:795]*/, v154 /*v666*/ offset:14048
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0xc256
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[156:163] /*v[668:675]*/, v[82:89] /*v[338:345]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[164:171] /*v[676:683]*/, v[82:89] /*v[338:345]*/, v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[220:227] /*v[732:739]*/, v[82:89] /*v[338:345]*/, v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[228:235] /*v[740:747]*/, v[82:89] /*v[338:345]*/, v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[188:195] /*v[700:707]*/, v[82:89] /*v[338:345]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[196:203] /*v[708:715]*/, v[82:89] /*v[338:345]*/, v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[252:259] /*v[764:771]*/, v[82:89] /*v[338:345]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5657
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[4:11] /*v[772:779]*/, v[82:89] /*v[338:345]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5756
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[156:163] /*v[668:675]*/, v[74:81] /*v[330:337]*/, v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[164:171] /*v[676:683]*/, v[74:81] /*v[330:337]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[220:227] /*v[732:739]*/, v[74:81] /*v[330:337]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[228:235] /*v[740:747]*/, v[74:81] /*v[330:337]*/, v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[188:195] /*v[700:707]*/, v[74:81] /*v[330:337]*/, v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[196:203] /*v[708:715]*/, v[74:81] /*v[330:337]*/, v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[252:259] /*v[764:771]*/, v[74:81] /*v[330:337]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5657
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[4:11] /*v[772:779]*/, v[74:81] /*v[330:337]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5756
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[172:179] /*v[684:691]*/, v[242:249] /*v[498:505]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[180:187] /*v[692:699]*/, v[242:249] /*v[498:505]*/, v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[236:243] /*v[748:755]*/, v[242:249] /*v[498:505]*/, v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[244:251] /*v[756:763]*/, v[242:249] /*v[498:505]*/, v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[204:211] /*v[716:723]*/, v[242:249] /*v[498:505]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[212:219] /*v[724:731]*/, v[242:249] /*v[498:505]*/, v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5657
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[12:19] /*v[780:787]*/, v[242:249] /*v[498:505]*/, v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[20:27] /*v[788:795]*/, v[242:249] /*v[498:505]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x575a
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[172:179] /*v[684:691]*/, v[2:9] /*v[514:521]*/, v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[180:187] /*v[692:699]*/, v[2:9] /*v[514:521]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[236:243] /*v[748:755]*/, v[2:9] /*v[514:521]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[244:251] /*v[756:763]*/, v[2:9] /*v[514:521]*/, v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[204:211] /*v[716:723]*/, v[2:9] /*v[514:521]*/, v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[212:219] /*v[724:731]*/, v[2:9] /*v[514:521]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5a5b
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[12:19] /*v[780:787]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[20:27] /*v[788:795]*/, v[2:9] /*v[514:521]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5b42
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v154 /*v666*/ offset:18432
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v154 /*v666*/ offset:23040
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v154 /*v666*/ offset:18464
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v154 /*v666*/ offset:23072
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v154 /*v666*/ offset:27648
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v154 /*v666*/ offset:32256
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v154 /*v666*/ offset:27680
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v154 /*v666*/ offset:32288
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v154 /*v666*/ offset:18560
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v154 /*v666*/ offset:23168
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v154 /*v666*/ offset:18592
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v154 /*v666*/ offset:23200
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v154 /*v666*/ offset:27776
	ds_load_tr16_b128 v[192:195] /*v[704:707]*/, v154 /*v666*/ offset:32384
	ds_load_tr16_b128 v[196:199] /*v[708:711]*/, v154 /*v666*/ offset:27808
	ds_load_tr16_b128 v[200:203] /*v[712:715]*/, v154 /*v666*/ offset:32416
	ds_load_tr16_b128 v[204:207] /*v[716:719]*/, v154 /*v666*/ offset:18496
	ds_load_tr16_b128 v[208:211] /*v[720:723]*/, v154 /*v666*/ offset:23104
	ds_load_tr16_b128 v[212:215] /*v[724:727]*/, v154 /*v666*/ offset:18528
	ds_load_tr16_b128 v[216:219] /*v[728:731]*/, v154 /*v666*/ offset:23136
	ds_load_tr16_b128 v[220:223] /*v[732:735]*/, v154 /*v666*/ offset:27712
	ds_load_tr16_b128 v[224:227] /*v[736:739]*/, v154 /*v666*/ offset:32320
	ds_load_tr16_b128 v[228:231] /*v[740:743]*/, v154 /*v666*/ offset:27744
	ds_load_tr16_b128 v[232:235] /*v[744:747]*/, v154 /*v666*/ offset:32352
	ds_load_tr16_b128 v[236:239] /*v[748:751]*/, v154 /*v666*/ offset:18624
	ds_load_tr16_b128 v[240:243] /*v[752:755]*/, v154 /*v666*/ offset:23232
	ds_load_tr16_b128 v[244:247] /*v[756:759]*/, v154 /*v666*/ offset:18656
	ds_load_tr16_b128 v[248:251] /*v[760:763]*/, v154 /*v666*/ offset:23264
	ds_load_tr16_b128 v[252:255] /*v[764:767]*/, v154 /*v666*/ offset:27840
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[0:3] /*v[768:771]*/, v154 /*v666*/ offset:32448
	ds_load_tr16_b128 v[4:7] /*v[772:775]*/, v154 /*v666*/ offset:27872
	ds_load_tr16_b128 v[8:11] /*v[776:779]*/, v154 /*v666*/ offset:32480
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0xc255
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[242:249] /*v[498:505]*/, v[226:233] /*v[482:489]*/, v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5556
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[2:9] /*v[514:521]*/, v[226:233] /*v[482:489]*/, v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[204:211] /*v[716:723]*/, v[226:233] /*v[482:489]*/, v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[212:219] /*v[724:731]*/, v[226:233] /*v[482:489]*/, v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[172:179] /*v[684:691]*/, v[226:233] /*v[482:489]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[180:187] /*v[692:699]*/, v[226:233] /*v[482:489]*/, v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[236:243] /*v[748:755]*/, v[226:233] /*v[482:489]*/, v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[244:251] /*v[756:763]*/, v[226:233] /*v[482:489]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5655
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[242:249] /*v[498:505]*/, v[250:257] /*v[506:513]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5556
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[2:9] /*v[514:521]*/, v[250:257] /*v[506:513]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[204:211] /*v[716:723]*/, v[250:257] /*v[506:513]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[212:219] /*v[724:731]*/, v[250:257] /*v[506:513]*/, v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[172:179] /*v[684:691]*/, v[250:257] /*v[506:513]*/, v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[180:187] /*v[692:699]*/, v[250:257] /*v[506:513]*/, v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[236:243] /*v[748:755]*/, v[250:257] /*v[506:513]*/, v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[244:251] /*v[756:763]*/, v[250:257] /*v[506:513]*/, v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[156:163] /*v[668:675]*/, v[218:225] /*v[474:481]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[164:171] /*v[676:683]*/, v[218:225] /*v[474:481]*/, v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[220:227] /*v[732:739]*/, v[218:225] /*v[474:481]*/, v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[228:235] /*v[740:747]*/, v[218:225] /*v[474:481]*/, v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[188:195] /*v[700:707]*/, v[218:225] /*v[474:481]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[196:203] /*v[708:715]*/, v[218:225] /*v[474:481]*/, v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[252:259] /*v[764:771]*/, v[218:225] /*v[474:481]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5657
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[4:11] /*v[772:779]*/, v[218:225] /*v[474:481]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5756
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[156:163] /*v[668:675]*/, v[234:241] /*v[490:497]*/, v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[164:171] /*v[676:683]*/, v[234:241] /*v[490:497]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[220:227] /*v[732:739]*/, v[234:241] /*v[490:497]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[228:235] /*v[740:747]*/, v[234:241] /*v[490:497]*/, v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[188:195] /*v[700:707]*/, v[234:241] /*v[490:497]*/, v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[196:203] /*v[708:715]*/, v[234:241] /*v[490:497]*/, v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[252:259] /*v[764:771]*/, v[234:241] /*v[490:497]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5657
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[4:11] /*v[772:779]*/, v[234:241] /*v[490:497]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x570a
	v_add_f32_e32 v255, v146 /*v658*/, v147 /*v659*/
	s_set_vgpr_msb 0xa40
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, s[72:73]
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v253, v251
	s_mov_b32 s0, 0x76543210
	s_set_vgpr_msb 64
	v_mov_b32_e32 v78 /*v334*/, v255
	s_bfe_u32 s1, ttmp8, 0x50019
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[0:1], v[0:1], v[250:251]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[250:251], v[80:81] /*v[336:337]*/, v[252:253]
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v80 /*v336*/, s72
	v_permlanex16_b32 v78 /*v334*/, v78 /*v334*/, s0, 0xfedcba98
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78 /*v334*/, v78 /*v334*/, v255
	s_set_vgpr_msb 0x4101
	v_mov_b32_e32 v255, 0x3f317218
	s_delay_alu instid0(VALU_DEP_2)
	v_log_f32_e32 v0, v78 /*v334*/
	s_set_vgpr_msb 0x100
	v_mov_b32_e32 v251, v1
	v_permlanex16_b32 v1, v1, s0, 0xfedcba98
	s_lshl_b32 s0, s1, 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	s_add_co_i32 s2, s100, s0
	v_mul_f32_e32 v0, 0x3f317218, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[0:1], v[250:251], v[0:1]
	s_set_vgpr_msb 8
	v_dual_mov_b32 v250, s1 :: v_dual_bitop2_b32 v251, s2, v148 /*v660*/ bitop3:0x54
	s_mov_b32 s1, 0
	s_set_vgpr_msb 0x840
	v_log_f32_e32 v81 /*v337*/, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s99, v251
	s_and_b32 s87, vcc_lo, exec_lo
	s_set_vgpr_msb 0x4041
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, v[254:255]
	s_set_vgpr_msb 0x4105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v255, s2 :: v_dual_add_f32 v253, v80 /*v336*/, v81 /*v337*/
	s_set_vgpr_msb 0x500
	s_branch .LBB0_18
.LBB0_17:
	s_mov_b32 s1, -1
	s_mov_b32 s87, 0
.LBB0_18:
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s16, 1
	s_cbranch_vccnz .LBB0_20
	s_lshl_b32 s33, s34, 7
	s_set_vgpr_msb 1
	v_readlane_b32 s2, v0 /*v256*/, 25
	s_sub_co_i32 s71, s65, s33
	s_mul_i32 s1, s33, s75
	s_sub_co_i32 s0, s71, s98
	s_mov_b32 s94, 0
	s_add_co_i32 s18, s2, s1
	s_max_i32 s1, s0, 32
	s_max_i32 s2, s0, 64
	s_mov_b32 s89, 0x800000
	s_sub_co_i32 s1, s1, 32
	v_med3_i32 v0, s0, 0, 8
	s_mov_b32 s95, s94
	s_sub_co_i32 s2, s2, 64
	s_mov_b32 s92, 8
	s_mov_b32 s88, 0xf510000
	s_mov_b32 s91, s89
	s_min_u32 s1, s1, 8
	s_set_vgpr_msb 0x141
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, s[94:95]
	s_min_u32 s2, s2, 8
	s_max_i32 s9, s0, 0x60
	v_readlane_b32 s3, v0 /*v256*/, 26
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, s[92:93]
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, s[90:91]
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, s[88:89]
	s_lshl_b32 s90, s1, 16
	s_lshl_b32 s8, s2, 16
	s_mov_b64 s[0:1], s[88:89]
	s_addk_co_i32 s9, 0xffa0
	s_set_vgpr_msb 0x4100
	v_dual_mov_b32 v251, s93 :: v_dual_lshlrev_b32 v250, 16, v0
	s_mov_b64 s[2:3], s[90:91]
	s_mov_b32 s2, s8
	s_min_u32 s8, s9, 8
	s_mov_b64 s[4:5], s[92:93]
	s_lshl_b32 s17, s8, 16
	s_mov_b64 s[8:9], s[88:89]
	s_mov_b64 s[10:11], s[90:91]
	s_mov_b64 s[12:13], s[92:93]
	s_mov_b64 s[6:7], s[94:95]
	s_mov_b32 s5, s93
	s_mov_b64 s[14:15], s[94:95]
	s_mov_b32 s10, s17
	s_mov_b32 s13, s93
	s_wait_dscnt 0xc
	s_ashr_i32 s19, s18, 31
	s_add_co_i32 s17, s96, s101
	s_add_nc_u64 s[18:19], s[58:59], s[18:19]
	s_add_co_i32 s34, s17, 0x4800
	s_mov_b64 s[26:27], s[18:19]
	s_or_b32 s20, s19, 0x80000000
	s_mov_b64 s[24:25], s[16:17]
	s_add_nc_u64 s[18:19], s[18:19], s[56:57]
	s_mov_b32 s27, s20
	s_mov_b64 s[30:31], s[18:19]
	s_add_co_i32 s20, s17, 0x2400
	s_or_b32 s21, s19, 0x80000000
	s_mov_b64 s[28:29], s[16:17]
	s_mov_b32 s30, s18
	s_add_nc_u64 s[18:19], s[18:19], s[56:57]
	s_mov_b32 s29, s20
	s_mov_b32 s31, s21
	s_mov_b64 s[22:23], s[18:19]
	s_or_b32 s35, s19, 0x80000000
	s_mov_b64 s[20:21], s[16:17]
	s_mov_b32 s22, s18
	s_add_nc_u64 s[18:19], s[18:19], s[56:57]
	s_mov_b32 s21, s34
	s_mov_b32 s23, s35
	s_addk_co_i32 s17, 0x6c00
	s_bitset1_b32 s19, 31
	s_set_vgpr_msb 0x41
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[66:73] /*v[322:329]*/, v[66:73], 0
	v_readfirstlane_b32 s36, v90 /*v346*/
	v_readfirstlane_b32 s37, v91 /*v347*/
	s_set_vgpr_msb 0x4100
	v_readfirstlane_b32 s38, v250
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s39, v93 /*v349*/
	v_readfirstlane_b32 s40, v94 /*v350*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s41, v251
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s42, v96 /*v352*/
	v_readfirstlane_b32 s43, v97 /*v353*/
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[24:27], s[36:43]
	tensor_load_to_lds s[28:31], s[88:95]
	s_set_vgpr_msb 0x142
	v_exp_f32_e32 v158 /*v414*/, v136 /*v648*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[66:73] /*v[322:329]*/, v[90:97], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4142
	ds_load_b128 v[66:69] /*v[322:325]*/, v153 /*v665*/ offset:12800
	ds_load_b128 v[70:73] /*v[326:329]*/, v153 /*v665*/ offset:12832
	ds_load_b128 v[122:125] /*v[378:381]*/, v153 /*v665*/ offset:12864
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[224:225], v[224:225], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[58:65] /*v[314:321]*/, v[50:57], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[126:129] /*v[382:385]*/, v153 /*v665*/ offset:12896
	ds_load_b128 v[130:133] /*v[386:389]*/, v153 /*v665*/ offset:12928
	ds_load_b128 v[134:137] /*v[390:393]*/, v153 /*v665*/ offset:12960
	v_exp_f32_e32 v159 /*v415*/, v137 /*v649*/
	v_exp_f32_e32 v168 /*v424*/, v122 /*v634*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[58:65] /*v[314:321]*/, v[82:89], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[190:193] /*v[446:449]*/, v153 /*v665*/ offset:19200
	ds_load_b128 v[194:197] /*v[450:453]*/, v153 /*v665*/ offset:19232
	ds_load_b128 v[198:201] /*v[454:457]*/, v153 /*v665*/ offset:19264
	v_exp_f32_e32 v78 /*v334*/, v144 /*v656*/
	v_exp_f32_e32 v79 /*v335*/, v145 /*v657*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[42:49] /*v[298:305]*/, v[34:41], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[202:205] /*v[458:461]*/, v153 /*v665*/ offset:19296
	ds_load_b128 v[114:117] /*v[370:373]*/, v153 /*v665*/ offset:19328
	ds_load_b128 v[118:121] /*v[374:377]*/, v153 /*v665*/ offset:19360
	v_exp_f32_e32 v160 /*v416*/, v140 /*v652*/
	v_exp_f32_e32 v161 /*v417*/, v141 /*v653*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[42:49] /*v[298:305]*/, v[74:81], v[90:97] /*v[346:353]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[58:61] /*v[314:317]*/, v153 /*v665*/ offset:12992
	ds_load_b128 v[62:65] /*v[318:321]*/, v153 /*v665*/ offset:13024
	ds_load_b128 v[206:209] /*v[462:465]*/, v153 /*v665*/ offset:13056
	v_exp_f32_e32 v80 /*v336*/, v138 /*v650*/
	v_exp_f32_e32 v81 /*v337*/, v139 /*v651*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[50:57] /*v[306:313]*/, v[42:49], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[210:213] /*v[466:469]*/, v153 /*v665*/ offset:13088
	ds_load_b128 v[214:217] /*v[470:473]*/, v153 /*v665*/ offset:13120
	ds_load_b128 v[218:221] /*v[474:477]*/, v153 /*v665*/ offset:13152
	v_exp_f32_e32 v154 /*v410*/, v118 /*v630*/
	v_exp_f32_e32 v169 /*v425*/, v123 /*v635*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[50:57] /*v[306:313]*/, v[58:65], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[222:225] /*v[478:481]*/, v153 /*v665*/ offset:19392
	ds_load_b128 v[226:229] /*v[482:485]*/, v153 /*v665*/ offset:19424
	ds_load_b128 v[146:149] /*v[402:405]*/, v153 /*v665*/ offset:19456
	s_set_vgpr_msb 0x4202
	v_exp_f32_e32 v0, v142 /*v654*/
	v_exp_f32_e32 v1, v143 /*v655*/
	s_set_vgpr_msb 0x251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[34:41] /*v[290:297]*/, v[10:17], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[150:153] /*v[406:409]*/, v153 /*v665*/ offset:19488
	ds_load_b128 v[106:109] /*v[362:365]*/, v153 /*v665*/ offset:19520
	ds_load_b128 v[110:113] /*v[366:369]*/, v153 /*v665*/ offset:19552
	s_set_vgpr_msb 0x4202
	v_exp_f32_e32 v250, v120 /*v632*/
	s_set_vgpr_msb 0x251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[34:41] /*v[290:297]*/, v[26:33], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v186 /*v442*/, v130 /*v642*/
	v_exp_f32_e32 v187 /*v443*/, v131 /*v643*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[26:33] /*v[282:289]*/, v[2:9], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v188 /*v444*/, v128 /*v640*/
	v_exp_f32_e32 v189 /*v445*/, v129 /*v641*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[26:33] /*v[282:289]*/, v[18:25], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v176 /*v432*/, v134 /*v646*/
	v_exp_f32_e32 v177 /*v433*/, v135 /*v647*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[66:73], 0
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v180 /*v436*/, v132 /*v644*/
	v_exp_f32_e32 v181 /*v437*/, v133 /*v645*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[18:25] /*v[274:281]*/, v[90:97], 0
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v182 /*v438*/, v126 /*v638*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[216:217], v[216:217], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215], v[214:215], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213], v[212:213], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[10:17] /*v[266:273]*/, v[50:57], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v183 /*v439*/, v127 /*v639*/
	v_exp_f32_e32 v156 /*v412*/, v116 /*v628*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[10:17] /*v[266:273]*/, v[82:89], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v155 /*v411*/, v119 /*v631*/
	s_set_vgpr_msb 0x4202
	v_exp_f32_e32 v251, v121 /*v633*/
	s_set_vgpr_msb 0x250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[242:249], v[34:41], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v166 /*v422*/, v124 /*v636*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[208:209], v[208:209], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[242:249], v[74:81], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v167 /*v423*/, v125 /*v637*/
	s_set_vgpr_msb 0x4208
	v_pk_mul_f32 v[200:201], v[200:201], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199], v[198:199], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[196:197], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195], v[194:195], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[2:9] /*v[258:265]*/, v[42:49], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v157 /*v413*/, v117 /*v629*/
	v_exp_f32_e32 v162 /*v418*/, v114 /*v626*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[2:9] /*v[258:265]*/, v[58:65], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5142
	v_exp_f32_e32 v163 /*v419*/, v115 /*v627*/
	v_exp_f32_e32 v164 /*v420*/, v112 /*v624*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[234:241], v[10:17], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v165 /*v421*/, v113 /*v625*/
	v_exp_f32_e32 v170 /*v426*/, v110 /*v622*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[234:241], v[26:33], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v171 /*v427*/, v111 /*v623*/
	v_exp_f32_e32 v172 /*v428*/, v108 /*v620*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[226:233], v[2:9], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v173 /*v429*/, v109 /*v621*/
	v_exp_f32_e32 v174 /*v430*/, v106 /*v618*/
	s_set_vgpr_msb 0x4250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[226:233], v[18:25], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5042
	v_exp_f32_e32 v175 /*v431*/, v107 /*v619*/
	s_wait_dscnt 0xc
	v_exp_f32_e32 v184 /*v440*/, v104 /*v616*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[66:73] /*v[322:329]*/, v[66:73], 0
	tensor_load_to_lds s[20:23], s[0:7]
	tensor_load_to_lds s[16:19], s[8:15]
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v185 /*v441*/, v105 /*v617*/
	s_set_vgpr_msb 0x4241
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[66:73] /*v[322:329]*/, v[90:97], 0
	s_set_vgpr_msb 0x4102
	ds_load_b128 v[226:229], v153 /*v665*/ offset:25600
	ds_load_b128 v[230:233], v153 /*v665*/ offset:25632
	ds_load_b128 v[242:245], v153 /*v665*/ offset:25664
	v_pk_mul_f32 v[192:193], v[74:75] /*v[586:587]*/, v[192:193] op_sel_hi:[0,1]
	v_pk_mul_f32 v[190:191], v[74:75] /*v[586:587]*/, v[190:191] op_sel_hi:[0,1]
	v_pk_mul_f32 v[188:189], v[74:75] /*v[586:587]*/, v[188:189] op_sel_hi:[0,1]
	v_pk_mul_f32 v[186:187], v[74:75] /*v[586:587]*/, v[186:187] op_sel_hi:[0,1]
	s_set_vgpr_msb 0x251
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[122:129] /*v[378:385]*/, v[50:57], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5102
	ds_load_b128 v[246:249], v153 /*v665*/ offset:25696
	s_set_vgpr_msb 0x242
	ds_load_b128 v[10:13] /*v[266:269]*/, v153 /*v665*/ offset:25728
	ds_load_b128 v[14:17] /*v[270:273]*/, v153 /*v665*/ offset:25760
	v_exp_f32_e32 v250 /*v506*/, v10 /*v522*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[122:129] /*v[378:385]*/, v[82:89], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[18:21] /*v[274:277]*/, v153 /*v665*/ offset:32000
	ds_load_b128 v[22:25] /*v[278:281]*/, v153 /*v665*/ offset:32032
	ds_load_b128 v[2:5] /*v[258:261]*/, v153 /*v665*/ offset:32064
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v0 /*v512*/, v100 /*v612*/
	v_exp_f32_e32 v1 /*v513*/, v101 /*v613*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[34:41], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[6:9] /*v[262:265]*/, v153 /*v665*/ offset:32096
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[234:237], v153 /*v665*/ offset:32128
	ds_load_b128 v[238:241], v153 /*v665*/ offset:32160
	s_set_vgpr_msb 0x242
	v_exp_f32_e32 v178 /*v434*/, v96 /*v608*/
	v_exp_f32_e32 v179 /*v435*/, v97 /*v609*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[130:137] /*v[386:393]*/, v[74:81], v[42:49] /*v[298:305]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[2:5] /*v[514:517]*/, v153 /*v665*/ offset:25792
	ds_load_b128 v[6:9] /*v[518:521]*/, v153 /*v665*/ offset:25824
	ds_load_b128 v[104:107] /*v[616:619]*/, v153 /*v665*/ offset:25856
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v252 /*v508*/, v102 /*v614*/
	v_exp_f32_e32 v253 /*v509*/, v103 /*v615*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[58:65] /*v[314:321]*/, v[42:49], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[108:111] /*v[620:623]*/, v153 /*v665*/ offset:25888
	ds_load_b128 v[112:115] /*v[624:627]*/, v153 /*v665*/ offset:25920
	ds_load_b128 v[116:119] /*v[628:631]*/, v153 /*v665*/ offset:25952
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v254 /*v510*/, v98 /*v610*/
	v_exp_f32_e32 v255 /*v511*/, v99 /*v611*/
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[58:65] /*v[314:321]*/, v[58:65], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[138:141] /*v[394:397]*/, v153 /*v665*/ offset:32192
	ds_load_b128 v[142:145] /*v[398:401]*/, v153 /*v665*/ offset:32224
	ds_load_b128 v[130:133] /*v[386:389]*/, v153 /*v665*/ offset:32256
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v88 /*v600*/, v88 /*v600*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[206:213] /*v[462:469]*/, v[10:17], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5142
	ds_load_b128 v[134:137] /*v[390:393]*/, v153 /*v665*/ offset:32288
	ds_load_b128 v[122:125] /*v[378:381]*/, v153 /*v665*/ offset:32320
	ds_load_b128 v[126:129] /*v[382:385]*/, v153 /*v665*/ offset:32352
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v89 /*v601*/, v89 /*v601*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[206:213] /*v[462:469]*/, v[26:33], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v94 /*v606*/, v94 /*v606*/
	v_exp_f32_e32 v95 /*v607*/, v95 /*v607*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[214:221] /*v[470:477]*/, v[2:9], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v92 /*v604*/, v92 /*v604*/
	v_exp_f32_e32 v93 /*v605*/, v93 /*v605*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[214:221] /*v[470:477]*/, v[18:25], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v90 /*v602*/, v90 /*v602*/
	v_exp_f32_e32 v91 /*v603*/, v91 /*v603*/
	s_set_vgpr_msb 0x8241
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[190:197] /*v[446:453]*/, v[66:73], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v86 /*v598*/, v86 /*v598*/
	v_exp_f32_e32 v87 /*v599*/, v87 /*v599*/
	s_set_vgpr_msb 0x8241
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[190:197] /*v[446:453]*/, v[90:97], 0
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v84 /*v596*/, v84 /*v596*/
	v_exp_f32_e32 v85 /*v597*/, v85 /*v597*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[198:205] /*v[454:461]*/, v[50:57], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v96 /*v608*/, v80 /*v592*/
	v_exp_f32_e32 v80 /*v592*/, v82 /*v594*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[198:205] /*v[454:461]*/, v[82:89], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v97 /*v609*/, v81 /*v593*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[184:185], v[184:185], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183], v[182:183], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181], v[180:181], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179], v[178:179], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[114:121] /*v[370:377]*/, v[34:41], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v81 /*v593*/, v83 /*v595*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[176:177], v[176:177], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175], v[174:175], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[172:173], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171], v[170:171], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[114:121] /*v[370:377]*/, v[74:81], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v66 /*v578*/, v66 /*v578*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[168:169], v[168:169], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[222:229] /*v[478:485]*/, v[42:49], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v67 /*v579*/, v67 /*v579*/
	v_exp_f32_e32 v82 /*v594*/, v60 /*v572*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[222:229] /*v[478:485]*/, v[58:65], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v83 /*v595*/, v61 /*v573*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[146:153] /*v[402:409]*/, v[10:17], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v60 /*v572*/, v78 /*v590*/
	v_exp_f32_e32 v61 /*v573*/, v79 /*v591*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[146:153] /*v[402:409]*/, v[26:33], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v62 /*v574*/, v62 /*v574*/
	v_exp_f32_e32 v63 /*v575*/, v63 /*v575*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[106:113] /*v[362:369]*/, v[2:9], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v26 /*v538*/, v26 /*v538*/
	v_exp_f32_e32 v27 /*v539*/, v27 /*v539*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[106:113] /*v[362:369]*/, v[18:25], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v10 /*v522*/, v24 /*v536*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v251 /*v507*/, v11 /*v523*/
	s_set_vgpr_msb 0x4240
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[226:233], v[66:73], 0
	s_set_vgpr_msb 0x4042
	ds_load_b128 v[146:149] /*v[402:405]*/, v153 /*v665*/ offset:38400
	ds_load_b128 v[150:153] /*v[406:409]*/, v153 /*v665*/ offset:38432
	s_set_vgpr_msb 0x4282
	ds_load_b128 v[120:123] /*v[632:635]*/, v153 /*v665*/ offset:38464
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v11 /*v523*/, v25 /*v537*/
	s_set_vgpr_msb 0x8240
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[226:233], v[90:97], 0
	s_set_vgpr_msb 0x4082
	ds_load_b128 v[124:127] /*v[636:639]*/, v153 /*v665*/ offset:38496
	ds_load_b128 v[128:131] /*v[640:643]*/, v153 /*v665*/ offset:38528
	ds_load_b128 v[132:135] /*v[644:647]*/, v153 /*v665*/ offset:38560
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	s_set_vgpr_msb 0x8250
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[242:249], v[50:57], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5042
	ds_load_b128 v[242:245] /*v[498:501]*/, v153 /*v665*/ offset:44800
	ds_load_b128 v[246:249] /*v[502:505]*/, v153 /*v665*/ offset:44832
	ds_load_b128 v[234:237] /*v[490:493]*/, v153 /*v665*/ offset:44864
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v24 /*v536*/, v50 /*v562*/
	v_exp_f32_e32 v25 /*v537*/, v51 /*v563*/
	s_set_vgpr_msb 0x8250
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[242:249], v[82:89], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5042
	ds_load_b128 v[238:241] /*v[494:497]*/, v153 /*v665*/ offset:44896
	ds_load_b128 v[218:221] /*v[474:477]*/, v153 /*v665*/ offset:44928
	ds_load_b128 v[222:225] /*v[478:481]*/, v153 /*v665*/ offset:44960
	s_set_vgpr_msb 0x4282
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v49 /*v561*/, v49 /*v561*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[10:17] /*v[266:273]*/, v[34:41], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[136:139] /*v[648:651]*/, v153 /*v665*/ offset:38592
	ds_load_b128 v[140:143] /*v[652:655]*/, v153 /*v665*/ offset:38624
	ds_load_b128 v[156:159] /*v[668:671]*/, v153 /*v665*/ offset:38656
	v_exp_f32_e32 v50 /*v562*/, v64 /*v576*/
	v_exp_f32_e32 v51 /*v563*/, v65 /*v577*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[10:17] /*v[266:273]*/, v[74:81], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0xf
	s_set_vgpr_msb 0x5182
	ds_load_b128 v[160:163] /*v[672:675]*/, v153 /*v665*/ offset:38688
	ds_load_b128 v[164:167] /*v[676:679]*/, v153 /*v665*/ offset:38720
	ds_load_b128 v[168:171] /*v[680:683]*/, v153 /*v665*/ offset:38752
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v57 /*v569*/, v57 /*v569*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[2:9] /*v[514:521]*/, v[42:49], v[114:121] /*v[370:377]*/
	ds_load_b128 v[226:229] /*v[482:485]*/, v153 /*v665*/ offset:44992
	ds_load_b128 v[230:233] /*v[486:489]*/, v153 /*v665*/ offset:45024
	ds_load_b128 v[210:213] /*v[466:469]*/, v153 /*v665*/ offset:45056
	s_set_vgpr_msb 0x5282
	v_exp_f32_e32 v64 /*v576*/, v76 /*v588*/
	v_exp_f32_e32 v65 /*v577*/, v77 /*v589*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[2:9] /*v[514:521]*/, v[58:65], v[106:113] /*v[362:369]*/
	ds_load_b128 v[214:217] /*v[470:473]*/, v153 /*v665*/ offset:45088
	ds_load_b128 v[202:205] /*v[458:461]*/, v153 /*v665*/ offset:45120
	ds_load_b128 v[206:209] /*v[462:465]*/, v153 /*v665*/ offset:45152
	s_set_vgpr_msb 0x5282
	v_exp_f32_e32 v58 /*v570*/, v58 /*v570*/
	v_exp_f32_e32 v59 /*v571*/, v59 /*v571*/
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[104:111] /*v[616:623]*/, v[10:17], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5245
	v_cvt_pk_bf16_f32 v10 /*v266*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v11 /*v267*/, v168 /*v424*/, v169 /*v425*/
	s_set_vgpr_msb 0x4552
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[104:111] /*v[616:623]*/, v[26:33], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5245
	v_cvt_pk_bf16_f32 v12 /*v268*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v13 /*v269*/, v188 /*v444*/, v189 /*v445*/
	s_set_vgpr_msb 0x4505
	v_cvt_pk_bf16_f32 v242, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v243, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v244, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v245, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v226, v172 /*v428*/, v173 /*v429*/
	s_set_vgpr_msb 0x552
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[112:119] /*v[624:631]*/, v[2:9], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5205
	v_cvt_pk_bf16_f32 v227, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v228, v250 /*v506*/, v251 /*v507*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v229, v20 /*v532*/, v21 /*v533*/
	s_set_vgpr_msb 0xa8a
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[52:53] /*v[564:565]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8a45
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/
	s_set_vgpr_msb 0x4585
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[186:187] /*v[442:443]*/, v[188:189] /*v[444:445]*/
	s_set_vgpr_msb 0x8552
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[112:119] /*v[624:631]*/, v[18:25], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5245
	v_cvt_pk_bf16_f32 v14 /*v270*/, v78 /*v334*/, v79 /*v335*/
	v_cvt_pk_bf16_f32 v15 /*v271*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v176 /*v432*/, v177 /*v433*/
	s_set_vgpr_msb 0x4541
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[18:25] /*v[274:281]*/, v[66:73], 0
	s_set_vgpr_msb 0x4145
	v_cvt_pk_bf16_f32 v17 /*v273*/, v180 /*v436*/, v181 /*v437*/
	s_set_vgpr_msb 0x4505
	v_cvt_pk_bf16_f32 v246, v182 /*v438*/, v183 /*v439*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v247, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0xa05
	v_cvt_pk_bf16_f32 v248, v178 /*v434*/, v179 /*v435*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v249, v86 /*v598*/, v87 /*v599*/
	v_cvt_pk_bf16_f32 v230, v84 /*v596*/, v85 /*v597*/
	v_cvt_pk_bf16_f32 v231, v96 /*v608*/, v97 /*v609*/
	s_set_vgpr_msb 0xa41
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[18:25] /*v[274:281]*/, v[90:97], 0
	s_set_vgpr_msb 0x410a
	v_cvt_pk_bf16_f32 v232, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v233, v48 /*v560*/, v49 /*v561*/
	s_set_vgpr_msb 0xa8a
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[44:45] /*v[556:557]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[78:79] /*v[334:335]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[176:177] /*v[432:433]*/, v[180:181] /*v[436:437]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[182:183] /*v[438:439]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x8951
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[2:9] /*v[258:265]*/, v[50:57], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5108
	v_pk_mul_f32 v[160:161], v[160:161], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[2:9] /*v[258:265]*/, v[82:89], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5108
	v_pk_mul_f32 v[152:153], v[152:153], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[234:241], v[34:41], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5045
	v_cvt_pk_bf16_f32 v18 /*v274*/, v80 /*v336*/, v81 /*v337*/
	v_cvt_pk_bf16_f32 v19 /*v275*/, v154 /*v410*/, v155 /*v411*/
	s_set_vgpr_msb 0x4508
	v_pk_mul_f32 v[144:145], v[144:145], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[234:241], v[74:81], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x504a
	v_cvt_pk_bf16_f32 v78 /*v334*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v79 /*v335*/, v18 /*v530*/, v19 /*v531*/
	s_set_vgpr_msb 0x4a08
	v_pk_mul_f32 v[136:137], v[136:137], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[134:135], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[132:133], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x851
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[138:145] /*v[394:401]*/, v[42:49], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5145
	v_cvt_pk_bf16_f32 v20 /*v276*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v2 /*v258*/, v252 /*v508*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v4 /*v260*/, v80 /*v592*/, v81 /*v593*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[138:145] /*v[394:401]*/, v[58:65], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x510a
	v_cvt_pk_bf16_f32 v234, v82 /*v594*/, v83 /*v595*/
	v_cvt_pk_bf16_f32 v235, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v236, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v237, v56 /*v568*/, v57 /*v569*/
	s_set_vgpr_msb 0xa4a
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[14:15] /*v[526:527]*/, v[30:31] /*v[542:543]*/
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[32:33] /*v[544:545]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[130:137] /*v[386:393]*/, v[10:17], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[80:81] /*v[336:337]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[166:167] /*v[422:423]*/, v[184:185] /*v[440:441]*/
	s_set_vgpr_msb 0x4585
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[252:253] /*v[508:509]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v80 /*v336*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v81 /*v337*/, v28 /*v540*/, v29 /*v541*/
	s_set_vgpr_msb 0x4a40
	v_cvt_pk_bf16_f32 v22 /*v278*/, v0, v1
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[130:137] /*v[386:393]*/, v[26:33], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5140
	v_cvt_pk_bf16_f32 v23 /*v279*/, v250, v251
	s_set_vgpr_msb 0x404a
	v_cvt_pk_bf16_f32 v24 /*v280*/, v88 /*v600*/, v89 /*v601*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v94 /*v606*/, v95 /*v607*/
	v_cvt_pk_bf16_f32 v6 /*v262*/, v92 /*v604*/, v93 /*v605*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v90 /*v602*/, v91 /*v603*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v60 /*v572*/, v61 /*v573*/
	s_set_vgpr_msb 0x4a51
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[122:129] /*v[378:385]*/, v[2:9], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x514a
	v_cvt_pk_bf16_f32 v9 /*v265*/, v62 /*v574*/, v63 /*v575*/
	s_set_vgpr_msb 0x4a0a
	v_cvt_pk_bf16_f32 v238, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v239, v10 /*v522*/, v11 /*v523*/
	v_cvt_pk_bf16_f32 v240, v64 /*v576*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v241, v58 /*v570*/, v59 /*v571*/
	s_set_vgpr_msb 0xa8a
	v_pk_add_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x8a51
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[122:129] /*v[378:385]*/, v[18:25], v[194:201] /*v[450:457]*/
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[122:123] /*v[378:379]*/, v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[124:125] /*v[380:381]*/, v[164:165] /*v[420:421]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[126:127] /*v[382:383]*/, v[172:173] /*v[428:429]*/, v[174:175] /*v[430:431]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[128:129] /*v[384:385]*/, v[250:251] /*v[506:507]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x498a
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[16:17] /*v[528:529]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x8a00
	v_pk_add_f32 v[0:1], v[0:1], v[250:251]
	s_barrier_wait -1
	s_set_vgpr_msb 0x81
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[146:153] /*v[402:409]*/, v[66:73], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8142
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v154 /*v666*/
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v154 /*v666*/ offset:4608
	s_set_vgpr_msb 0x420a
	v_pk_add_f32 v[250:251], v[40:41] /*v[552:553]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0xa89
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[158:159] /*v[414:415]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8941
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[146:153] /*v[402:409]*/, v[90:97], 0
	s_set_vgpr_msb 0x4142
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v154 /*v666*/ offset:32
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v154 /*v666*/ offset:4640
	s_set_vgpr_msb 0x4285
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[122:123] /*v[378:379]*/, v[124:125] /*v[380:381]*/
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	s_set_vgpr_msb 0x85a2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[120:127] /*v[632:639]*/, v[50:57], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v154 /*v666*/ offset:128
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v154 /*v666*/ offset:4736
	s_set_vgpr_msb 0x4289
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[178:179] /*v[434:435]*/, v[86:87] /*v[598:599]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[84:85] /*v[596:597]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[120:127] /*v[632:639]*/, v[82:89], v[250:257] /*v[506:513]*/
	ds_load_tr16_b128 v[122:125] /*v[378:381]*/, v154 /*v666*/ offset:160
	ds_load_tr16_b128 v[126:129] /*v[382:385]*/, v154 /*v666*/ offset:4768
	s_set_vgpr_msb 0x528a
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[550:551]*/
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[80:81] /*v[592:593]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[128:135] /*v[640:647]*/, v[34:41], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v154 /*v666*/ offset:64
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v154 /*v666*/ offset:4672
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[82:83] /*v[594:595]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[50:51] /*v[562:563]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[128:135] /*v[640:647]*/, v[74:81], v[250:257] /*v[506:513]*/
	s_wait_dscnt 0xa
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v154 /*v666*/ offset:96
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v154 /*v666*/ offset:4704
	s_set_vgpr_msb 0x5285
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[138:139] /*v[394:395]*/, v[140:141] /*v[396:397]*/
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[142:143] /*v[398:399]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x85a2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[136:143] /*v[648:655]*/, v[42:49], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa242
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v154 /*v666*/ offset:192
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v154 /*v666*/ offset:4800
	s_set_vgpr_msb 0x428a
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[88:89] /*v[600:601]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[92:93] /*v[604:605]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[136:143] /*v[648:655]*/, v[58:65], v[250:257] /*v[506:513]*/
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v154 /*v666*/ offset:224
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v154 /*v666*/ offset:4832
	s_set_vgpr_msb 0x528a
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[60:61] /*v[572:573]*/, v[62:63] /*v[574:575]*/
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[156:163] /*v[668:675]*/, v[10:17], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa28a
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[54:55] /*v[566:567]*/, v[28:29] /*v[540:541]*/
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[32:33] /*v[544:545]*/, v[24:25] /*v[536:537]*/
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[156:163] /*v[668:675]*/, v[26:33], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x528a
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[34:35] /*v[546:547]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a88
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[250:251], v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x88a2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[164:171] /*v[676:683]*/, v[2:9], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa20a
	v_pk_add_f32 v[250:251], v[28:29] /*v[540:541]*/, v[24:25] /*v[536:537]*/
	s_set_vgpr_msb 0xa8a
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[64:65] /*v[576:577]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x8a52
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[164:171] /*v[676:683]*/, v[18:25], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5202
	v_pk_add_f32 v[250:251], v[10:11] /*v[522:523]*/, v[250:251]
	s_set_vgpr_msb 0x28a
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[12:13] /*v[524:525]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x8a81
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[242:249] /*v[498:505]*/, v[66:73], 0
	s_set_vgpr_msb 0x818a
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[36:37] /*v[548:549]*/
	s_set_vgpr_msb 0x8a08
	v_pk_add_f32 v[0:1], v[0:1], v[22:23] /*v[534:535]*/
	s_set_vgpr_msb 0x881
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[242:249] /*v[498:505]*/, v[90:97], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x810a
	v_pk_add_f32 v[90:91], v[38:39] /*v[550:551]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[92:93], v[42:43] /*v[554:555]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0xaa1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[234:241] /*v[490:497]*/, v[50:57], v[10:17] /*v[522:529]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa10a
	v_pk_add_f32 v[50:51], v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	v_pk_add_f32 v[52:53], v[26:27] /*v[538:539]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0xaa1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[234:241] /*v[490:497]*/, v[82:89], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa102
	v_pk_add_f32 v[0:1], v[44:45] /*v[556:557]*/, v[0:1]
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[82:83], v[50:51], v[52:53]
	s_set_vgpr_msb 0xa1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[218:225] /*v[474:481]*/, v[34:41], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[72:73], v[128:129], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[126:127], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[124:125], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[122:123], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_add_f32 v[34:35], v[90:91], v[30:31] /*v[542:543]*/
	s_set_vgpr_msb 0x8a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[218:225] /*v[474:481]*/, v[74:81], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[56:57], v[120:121], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[118:119], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[116:117], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[114:115], v[74:75] /*v[586:587]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	v_pk_add_f32 v[90:91], v[92:93], v[34:35]
	s_set_vgpr_msb 0xa1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[226:233] /*v[482:489]*/, v[42:49], v[10:17] /*v[522:529]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[48:49], v[112:113], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[110:111], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[108:109], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[106:107], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x80a
	v_pk_add_f32 v[74:75], v[32:33] /*v[544:545]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0xaa1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[226:233] /*v[482:489]*/, v[58:65], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa108
	v_pk_mul_f32 v[40:41], v[104:105], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[102:103], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[100:101], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[98:99], v[72:73] /*v[584:585]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	v_pk_add_f32 v[92:93], v[0:1], v[82:83]
	s_set_vgpr_msb 0xa1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[210:217] /*v[466:473]*/, v[10:17], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[210:217] /*v[466:473]*/, v[26:33], v[18:25] /*v[530:537]*/
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[202:209] /*v[458:465]*/, v[2:9], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[202:209] /*v[458:465]*/, v[18:25], v[18:25] /*v[530:537]*/
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0xa108
	v_subrev_nc_u32_e32 v0, s33, v151 /*v663*/
	s_mov_b64 s[88:89], s[66:67]
	v_nop
	v_nop
	s_set_vgpr_msb 0x800
	v_dual_add_nc_u32 v1, -16, v0 :: v_dual_add_nc_u32 v2, 16, v0
	v_subrev_nc_u32_e32 v3, 48, v0
	v_cmp_gt_i32_e64 s61, 0, v0
	v_cmp_gt_i32_e64 s59, 1, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_i32_e64 s45, 0, v1
	v_cmp_gt_i32_e64 s43, 1, v1
	v_cmp_gt_i32_e64 s41, 2, v1
	v_cmp_gt_i32_e64 s40, 3, v1
	v_cmp_gt_i32_e64 s47, 4, v1
	v_cmp_gt_i32_e64 s46, 5, v1
	v_cmp_gt_i32_e64 s44, 6, v1
	v_cmp_gt_i32_e64 s42, 7, v1
	v_subrev_nc_u32_e32 v1, 32, v0
	v_cmp_gt_i32_e64 s37, 0, v3
	v_cmp_gt_i32_e64 s35, 1, v3
	v_cmp_gt_i32_e64 s31, 2, v3
	v_cmp_gt_i32_e64 s28, 3, v3
	v_cmp_gt_i32_e64 s53, 0, v1
	v_cmp_gt_i32_e64 s51, 1, v1
	v_cmp_gt_i32_e64 s50, 2, v1
	v_cmp_gt_i32_e64 s49, 3, v1
	v_cmp_gt_i32_e64 s55, 4, v1
	v_cmp_gt_i32_e64 s54, 5, v1
	v_cmp_gt_i32_e64 s52, 6, v1
	v_cmp_gt_i32_e64 s48, 7, v1
	v_subrev_nc_u32_e32 v1, 64, v0
	v_cmp_gt_i32_e64 s39, 4, v3
	v_cmp_gt_i32_e64 s38, 5, v3
	v_cmp_gt_i32_e64 s36, 6, v3
	v_cmp_gt_i32_e64 s33, 7, v3
	v_cmp_gt_i32_e64 s29, 0, v1
	v_cmp_gt_i32_e64 s26, 1, v1
	v_cmp_gt_i32_e64 s25, 2, v1
	v_cmp_gt_i32_e64 s24, 3, v1
	v_cmp_gt_i32_e64 s34, 4, v1
	v_cmp_gt_i32_e64 s30, 5, v1
	v_cmp_gt_i32_e64 s27, 6, v1
	v_cmp_gt_i32_e64 s23, 7, v1
	v_add_nc_u32_e32 v1, 0xffffffa0, v0
	v_add_nc_u32_e32 v3, 0xffffffb0, v0
	v_cmp_gt_i32_e64 s58, 2, v0
	v_cmp_gt_i32_e64 s57, 3, v0
	v_cmp_gt_i32_e64 s63, 4, v0
	v_cmp_gt_i32_e64 s5, 0, v1
	v_cmp_gt_i32_e64 s2, 1, v1
	v_cmp_gt_i32_e64 s0, 2, v1
	v_cmp_gt_i32_e32 vcc_lo, 3, v1
	v_cmp_gt_i32_e64 s9, 4, v1
	v_cmp_gt_i32_e64 s6, 5, v1
	v_cmp_gt_i32_e64 s3, 6, v1
	v_cmp_gt_i32_e64 s1, 7, v1
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v1, s71, v150 /*v662*/
	v_cmp_lt_i32_e64 s62, v0, 5
	v_cmp_lt_i32_e64 s60, v0, 6
	v_cmp_lt_i32_e64 s56, v0, 7
	v_cmp_lt_i32_e64 s12, v3, 0
	v_cmp_lt_i32_e64 s10, v3, 1
	v_cmp_lt_i32_e64 s7, v3, 2
	v_cmp_lt_i32_e64 s4, v3, 3
	v_cmp_lt_i32_e64 s14, v3, 4
	v_cmp_lt_i32_e64 s13, v3, 5
	v_cmp_lt_i32_e64 s11, v3, 6
	v_cmp_lt_i32_e64 s8, v3, 7
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v3, 0xffffff90, v0
	v_or_b32_e32 v0, v1, v0
	v_cmp_gt_i32_e64 s71, 1, v1
	v_cmp_gt_i32_e64 s72, 2, v1
	v_cmp_gt_i32_e64 s76, 6, v1
	v_cmp_gt_i32_e64 s80, 18, v1
	v_cmp_lt_i32_e64 s73, -1, v0
	s_or_b32 s74, s71, s59
	v_cmp_gt_i32_e64 s84, 22, v1
	s_set_vgpr_msb 5
	v_cndmask_b32_e64 v115, v99 /*v355*/, 0xff800000, s74
	s_or_b32 s74, s72, s58
	v_cndmask_b32_e64 v114, 0xff800000, v98 /*v354*/, s73
	s_set_vgpr_msb 0x501
	v_cmp_gt_i32_e64 s73, 3, v1
	v_cndmask_b32_e64 v120, v100 /*v356*/, 0xff800000, s74
	v_cmp_gt_i32_e64 s74, 4, v1
	v_cmp_gt_i32_e64 s64, 1, v2
	v_cmp_gt_i32_e64 s65, 2, v2
	s_or_b32 s75, s73, s57
	v_cmp_gt_i32_e64 s66, 3, v2
	v_cndmask_b32_e64 v121, v101 /*v357*/, 0xff800000, s75
	v_cmp_gt_i32_e64 s75, 5, v1
	s_or_b32 s77, s74, s63
	v_cmp_gt_i32_e64 s67, 4, v2
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v98 /*v354*/, v102 /*v358*/, 0xff800000, s77
	v_cmp_gt_i32_e64 s77, 7, v1
	s_or_b32 s78, s75, s62
	v_cmp_gt_i32_e64 s68, 5, v2
	v_cndmask_b32_e64 v99 /*v355*/, v103 /*v359*/, 0xff800000, s78
	s_or_b32 s78, s76, s60
	s_or_b32 s79, s77, s56
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v112, v104 /*v360*/, 0xff800000, s78
	v_cmp_gt_i32_e64 s78, 16, v1
	v_cndmask_b32_e64 v113, v105 /*v361*/, 0xff800000, s79
	v_cmp_gt_i32_e64 s79, 17, v1
	v_cmp_gt_i32_e64 s69, 6, v2
	v_cmp_gt_i32_e64 s70, 7, v2
	s_or_b32 s81, s78, s45
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v2, v1, v2
	s_set_vgpr_msb 1
	v_cndmask_b32_e64 v98, v34 /*v290*/, 0xff800000, s81
	v_cmp_gt_i32_e64 s81, 19, v1
	s_or_b32 s82, s79, s43
	s_or_b32 s59, s79, s59
	v_cndmask_b32_e64 v99, v35 /*v291*/, 0xff800000, s82
	s_or_b32 s82, s80, s41
	s_or_b32 s83, s81, s40
	v_cndmask_b32_e64 v88, v36 /*v292*/, 0xff800000, s82
	v_cmp_gt_i32_e64 s82, 20, v1
	v_cndmask_b32_e64 v89, v37 /*v293*/, 0xff800000, s83
	v_cmp_gt_i32_e64 s83, 21, v1
	s_or_b32 s57, s81, s57
	s_or_b32 s58, s80, s58
	s_or_b32 s85, s82, s47
	v_cndmask_b32_e64 v23, v29 /*v285*/, 0xff800000, s57
	s_or_b32 s57, s82, s63
	v_cndmask_b32_e64 v64, v38 /*v294*/, 0xff800000, s85
	v_cmp_gt_i32_e64 s85, 23, v1
	s_or_b32 s86, s83, s46
	v_cndmask_b32_e64 v24, v30 /*v286*/, 0xff800000, s57
	s_or_b32 s57, s83, s62
	v_cndmask_b32_e64 v65, v39 /*v295*/, 0xff800000, s86
	s_or_b32 s86, s84, s44
	v_cndmask_b32_e64 v25, v31 /*v287*/, 0xff800000, s57
	s_or_b32 s57, s84, s60
	v_cndmask_b32_e64 v78, v40 /*v296*/, 0xff800000, s86
	s_or_b32 s86, s85, s42
	v_cndmask_b32_e64 v26, v32 /*v288*/, 0xff800000, s57
	v_cmp_gt_i32_e64 s57, 32, v1
	s_or_b32 s56, s85, s56
	v_cndmask_b32_e64 v79, v41 /*v297*/, 0xff800000, s86
	v_cmp_lt_i32_e64 s86, -1, v2
	v_cndmask_b32_e64 v27, v33 /*v289*/, 0xff800000, s56
	v_cmp_gt_i32_e64 s56, 33, v1
	s_or_b32 s64, s71, s64
	v_cndmask_b32_e64 v29, v27 /*v283*/, 0xff800000, s59
	v_cndmask_b32_e64 v22, v28 /*v284*/, 0xff800000, s58
	v_cmp_gt_i32_e64 s58, 34, v1
	s_or_b32 s59, s57, s53
	s_set_vgpr_msb 0x145
	v_cndmask_b32_e64 v34 /*v290*/, 0xff800000, v90 /*v346*/, s86
	v_cndmask_b32_e64 v35 /*v291*/, v91 /*v347*/, 0xff800000, s64
	s_or_b32 s64, s72, s65
	v_cndmask_b32_e64 v90 /*v346*/, v50 /*v306*/, 0xff800000, s59
	s_set_vgpr_msb 0x4541
	v_cmp_gt_i32_e64 s59, 35, v1
	s_or_b32 s60, s56, s51
	v_cndmask_b32_e64 v36 /*v292*/, v92 /*v348*/, 0xff800000, s64
	s_or_b32 s64, s73, s66
	s_or_b32 s61, s78, s61
	v_cndmask_b32_e64 v91 /*v347*/, v51 /*v307*/, 0xff800000, s60
	s_or_b32 s60, s58, s50
	v_cndmask_b32_e64 v37 /*v293*/, v93 /*v349*/, 0xff800000, s64
	s_or_b32 s64, s74, s67
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v28, v26 /*v282*/, 0xff800000, s61
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v92 /*v348*/, v52 /*v308*/, 0xff800000, s60
	v_cmp_gt_i32_e64 s60, 36, v1
	s_or_b32 s61, s59, s49
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v102, v94 /*v350*/, 0xff800000, s64
	s_or_b32 s64, s75, s68
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v93 /*v349*/, v53 /*v309*/, 0xff800000, s61
	v_cmp_gt_i32_e64 s61, 37, v1
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v103, v95 /*v351*/, 0xff800000, s64
	s_or_b32 s64, s76, s69
	v_cmp_gt_i32_e64 s62, 38, v1
	v_cndmask_b32_e64 v104, v96 /*v352*/, 0xff800000, s64
	s_or_b32 s64, s77, s70
	s_or_b32 s63, s60, s55
	v_cndmask_b32_e64 v105, v97 /*v353*/, 0xff800000, s64
	v_cndmask_b32_e64 v18, v54 /*v310*/, 0xff800000, s63
	v_cmp_gt_i32_e64 s63, 39, v1
	s_or_b32 s64, s61, s54
	s_or_b32 s40, s59, s40
	v_cndmask_b32_e64 v19, v55 /*v311*/, 0xff800000, s64
	s_or_b32 s64, s62, s52
	s_or_b32 s65, s63, s48
	v_cndmask_b32_e64 v20, v56 /*v312*/, 0xff800000, s64
	v_cmp_gt_i32_e64 s64, 48, v1
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v45 /*v301*/, v45 /*v301*/, 0xff800000, s40
	s_or_b32 s40, s60, s47
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v21, v57 /*v313*/, 0xff800000, s65
	v_cmp_gt_i32_e64 s65, 49, v1
	v_cndmask_b32_e64 v10, v46 /*v302*/, 0xff800000, s40
	s_or_b32 s40, s61, s46
	v_cmp_gt_i32_e64 s66, 50, v1
	s_or_b32 s67, s64, s37
	v_cndmask_b32_e64 v11, v47 /*v303*/, 0xff800000, s40
	s_or_b32 s40, s62, s44
	v_cndmask_b32_e64 v122, v66 /*v322*/, 0xff800000, s67
	v_cmp_gt_i32_e64 s67, 51, v1
	v_cndmask_b32_e64 v12, v48 /*v304*/, 0xff800000, s40
	s_or_b32 s40, s63, s42
	s_or_b32 s68, s65, s35
	v_cndmask_b32_e64 v13, v49 /*v305*/, 0xff800000, s40
	s_or_b32 s40, s64, s53
	v_cndmask_b32_e64 v123, v67 /*v323*/, 0xff800000, s68
	s_or_b32 s68, s66, s31
	v_cndmask_b32_e64 v124, v58 /*v314*/, 0xff800000, s40
	s_or_b32 s40, s65, s51
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v26 /*v282*/, v68 /*v324*/, 0xff800000, s68
	v_cmp_gt_i32_e64 s68, 52, v1
	s_or_b32 s69, s67, s28
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v125, v59 /*v315*/, 0xff800000, s40
	s_or_b32 s40, s66, s50
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v27 /*v283*/, v69 /*v325*/, 0xff800000, s69
	v_cmp_gt_i32_e64 s69, 53, v1
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v126, v60 /*v316*/, 0xff800000, s40
	s_or_b32 s40, s67, s49
	v_cmp_gt_i32_e64 s70, 54, v1
	s_or_b32 s71, s68, s39
	v_cndmask_b32_e64 v127, v61 /*v317*/, 0xff800000, s40
	s_or_b32 s40, s68, s55
	v_cndmask_b32_e64 v16, v70 /*v326*/, 0xff800000, s71
	v_cmp_gt_i32_e64 s71, 55, v1
	v_cndmask_b32_e64 v128, v62 /*v318*/, 0xff800000, s40
	s_or_b32 s40, s69, s54
	s_or_b32 s41, s58, s41
	v_cndmask_b32_e64 v129, v63 /*v319*/, 0xff800000, s40
	s_or_b32 s40, s70, s52
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v44 /*v300*/, v44 /*v300*/, 0xff800000, s41
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v8, v64 /*v320*/, 0xff800000, s40
	v_cmp_gt_i32_e64 s40, 64, v1
	s_or_b32 s41, s71, s48
	s_or_b32 s43, s56, s43
	v_cndmask_b32_e64 v9, v65 /*v321*/, 0xff800000, s41
	v_cmp_gt_i32_e64 s41, 0x41, v1
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v51 /*v307*/, v43 /*v299*/, 0xff800000, s43
	v_cmp_gt_i32_e64 s42, 0x42, v1
	s_or_b32 s43, s40, s29
	s_or_b32 s45, s57, s45
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v116, v114 /*v370*/, 0xff800000, s43
	v_cmp_gt_i32_e64 s43, 0x43, v1
	s_or_b32 s44, s41, s26
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v50 /*v306*/, v42 /*v298*/, 0xff800000, s45
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v117, v115 /*v371*/, 0xff800000, s44
	s_or_b32 s44, s42, s25
	s_or_b32 s45, s43, s24
	v_cndmask_b32_e64 v118, v116 /*v372*/, 0xff800000, s44
	v_cmp_gt_i32_e64 s44, 0x44, v1
	v_cndmask_b32_e64 v119, v117 /*v373*/, 0xff800000, s45
	v_cmp_gt_i32_e64 s45, 0x45, v1
	v_cmp_gt_i32_e64 s46, 0x46, v1
	v_cmp_gt_i32_e64 s50, 0x52, v1
	s_or_b32 s47, s44, s34
	v_cmp_gt_i32_e64 s54, 0x56, v1
	s_set_vgpr_msb 0x141
	v_cndmask_b32_e64 v28 /*v284*/, v118 /*v374*/, 0xff800000, s47
	v_cmp_gt_i32_e64 s47, 0x47, v1
	s_or_b32 s48, s45, s30
	s_or_b32 s28, s43, s28
	v_cndmask_b32_e64 v29 /*v285*/, v119 /*v375*/, 0xff800000, s48
	s_or_b32 s48, s46, s27
	s_or_b32 s49, s47, s23
	v_cndmask_b32_e64 v30 /*v286*/, v120 /*v376*/, 0xff800000, s48
	v_cmp_gt_i32_e64 s48, 0x50, v1
	v_cndmask_b32_e64 v31 /*v287*/, v121 /*v377*/, 0xff800000, s49
	v_cmp_gt_i32_e64 s49, 0x51, v1
	s_set_vgpr_msb 0x4101
	v_cndmask_b32_e64 v85, v109 /*v365*/, 0xff800000, s28
	s_or_b32 s28, s44, s39
	s_or_b32 s51, s48, s12
	v_cndmask_b32_e64 v86, v110 /*v366*/, 0xff800000, s28
	v_cndmask_b32_e64 v100, v186 /*v442*/, 0xff800000, s51
	v_cmp_gt_i32_e64 s51, 0x53, v1
	s_or_b32 s52, s49, s10
	s_or_b32 s26, s49, s26
	v_cndmask_b32_e64 v101, v187 /*v443*/, 0xff800000, s52
	s_or_b32 s52, s50, s7
	s_or_b32 s53, s51, s4
	v_cndmask_b32_e64 v106, v188 /*v444*/, 0xff800000, s52
	v_cmp_gt_i32_e64 s52, 0x54, v1
	v_cndmask_b32_e64 v107, v189 /*v445*/, 0xff800000, s53
	v_cmp_gt_i32_e64 s53, 0x55, v1
	s_or_b32 s24, s51, s24
	s_or_b32 s28, s45, s38
	v_cndmask_b32_e64 v63, v197 /*v453*/, 0xff800000, s24
	s_or_b32 s24, s52, s34
	s_or_b32 s55, s52, s14
	v_cndmask_b32_e64 v30, v198 /*v454*/, 0xff800000, s24
	s_or_b32 s24, s53, s30
	v_cndmask_b32_e64 v108, v190 /*v446*/, 0xff800000, s55
	v_cmp_gt_i32_e64 s55, 0x57, v1
	v_cndmask_b32_e64 v31, v199 /*v455*/, 0xff800000, s24
	s_or_b32 s24, s54, s27
	v_cndmask_b32_e64 v61, v195 /*v451*/, 0xff800000, s26
	v_cndmask_b32_e64 v58, v200 /*v456*/, 0xff800000, s24
	v_cmp_gt_i32_e64 s24, 0x60, v1
	s_or_b32 s23, s55, s23
	s_or_b32 s25, s50, s25
	v_cndmask_b32_e64 v59, v201 /*v457*/, 0xff800000, s23
	v_cmp_gt_i32_e64 s23, 0x61, v1
	s_or_b32 s26, s24, s5
	v_cndmask_b32_e64 v87, v111 /*v367*/, 0xff800000, s28
	s_or_b32 s28, s46, s36
	v_cndmask_b32_e64 v62, v196 /*v452*/, 0xff800000, s25
	v_cmp_gt_i32_e64 s25, 0x62, v1
	s_set_vgpr_msb 0x142
	v_cndmask_b32_e64 v56 /*v312*/, v2 /*v514*/, 0xff800000, s26
	v_cmp_gt_i32_e64 s26, 0x63, v1
	s_set_vgpr_msb 0x4201
	v_cndmask_b32_e64 v80, v112 /*v368*/, 0xff800000, s28
	s_or_b32 s28, s47, s33
	s_or_b32 s27, s23, s2
	v_cndmask_b32_e64 v81, v113 /*v369*/, 0xff800000, s28
	s_or_b32 s28, s48, s29
	s_set_vgpr_msb 0x142
	v_cndmask_b32_e64 v57 /*v313*/, v3 /*v515*/, 0xff800000, s27
	s_set_vgpr_msb 0x4201
	v_cndmask_b32_e64 v60, v194 /*v450*/, 0xff800000, s28
	s_or_b32 s27, s25, s0
	s_or_b32 s28, s26, vcc_lo
	s_set_vgpr_msb 0x142
	v_cndmask_b32_e64 v58 /*v314*/, v4 /*v516*/, 0xff800000, s27
	v_cmp_gt_i32_e64 s27, 0x64, v1
	v_cndmask_b32_e64 v59 /*v315*/, v5 /*v517*/, 0xff800000, s28
	v_cmp_gt_i32_e64 s28, 0x65, v1
	v_cmp_gt_i32_e64 s29, 0x66, v1
	s_or_b32 s31, s42, s31
	s_or_b32 s30, s27, s9
	s_set_vgpr_msb 0x4201
	v_cndmask_b32_e64 v84, v108 /*v364*/, 0xff800000, s31
	s_or_b32 s31, s28, s6
	s_set_vgpr_msb 0x142
	v_cndmask_b32_e64 v60 /*v316*/, v6 /*v518*/, 0xff800000, s30
	v_cmp_gt_i32_e64 s30, 0x67, v1
	v_cndmask_b32_e64 v61 /*v317*/, v7 /*v519*/, 0xff800000, s31
	s_or_b32 s31, s29, s3
	s_or_b32 s72, s69, s38
	v_cmp_gt_i32_e64 s22, 0, v3
	s_set_vgpr_msb 0x4202
	v_cndmask_b32_e64 v76, v8 /*v520*/, 0xff800000, s31
	v_cmp_gt_i32_e64 s31, 0x70, v1
	s_set_vgpr_msb 0x201
	v_cndmask_b32_e64 v17, v71 /*v327*/, 0xff800000, s72
	s_or_b32 s72, s70, s36
	v_cmp_gt_i32_e64 s19, 2, v3
	v_cndmask_b32_e64 v14, v72 /*v328*/, 0xff800000, s72
	s_or_b32 s72, s71, s33
	s_or_b32 s33, s30, s1
	v_cmp_gt_i32_e64 s34, 0x72, v1
	v_cmp_gt_i32_e64 s20, 1, v3
	s_set_vgpr_msb 0x102
	v_cndmask_b32_e64 v77, v9 /*v521*/, 0xff800000, s33
	v_cmp_gt_i32_e64 s33, 0x71, v1
	s_or_b32 s22, s31, s22
	v_cmp_gt_i32_e64 s21, 3, v3
	s_set_vgpr_msb 0x242
	v_cndmask_b32_e64 v48 /*v304*/, v10 /*v522*/, 0xff800000, s22
	v_cmp_gt_i32_e64 s22, 0x73, v1
	s_or_b32 s19, s34, s19
	s_or_b32 s20, s33, s20
	v_cmp_gt_i32_e64 s18, 4, v3
	v_cndmask_b32_e64 v46 /*v302*/, v12 /*v524*/, 0xff800000, s19
	v_cmp_gt_i32_e64 s19, 0x74, v1
	v_cndmask_b32_e64 v49 /*v305*/, v11 /*v523*/, 0xff800000, s20
	s_or_b32 s20, s22, s21
	s_or_b32 s0, s34, s0
	v_cndmask_b32_e64 v47 /*v303*/, v13 /*v525*/, 0xff800000, s20
	v_cmp_gt_i32_e64 s20, 0x75, v1
	v_cndmask_b32_e64 v38 /*v294*/, v20 /*v532*/, 0xff800000, s0
	s_or_b32 s0, s22, vcc_lo
	v_cmp_gt_i32_e64 s21, 0x76, v1
	s_or_b32 s18, s19, s18
	v_cndmask_b32_e64 v39 /*v295*/, v21 /*v533*/, 0xff800000, s0
	s_or_b32 s0, s19, s9
	v_cndmask_b32_e64 v52 /*v308*/, v14 /*v526*/, 0xff800000, s18
	v_cmp_gt_i32_e64 s18, 0x77, v1
	s_or_b32 s4, s26, s4
	s_set_vgpr_msb 0x4202
	v_cndmask_b32_e64 v2, v22 /*v534*/, 0xff800000, s0
	s_or_b32 s0, s20, s6
	s_set_vgpr_msb 0x241
	v_cndmask_b32_e64 v43 /*v299*/, v253 /*v509*/, 0xff800000, s4
	s_or_b32 s4, s27, s14
	v_cmp_gt_i32_e64 s16, 5, v3
	v_cmp_gt_i32_e64 s15, 6, v3
	v_cmp_gt_i32_e64 s17, 7, v3
	s_set_vgpr_msb 0x4102
	v_cndmask_b32_e64 v3, v23 /*v535*/, 0xff800000, s0
	s_or_b32 s0, s21, s3
	s_set_vgpr_msb 0x201
	v_cndmask_b32_e64 v6, v254 /*v510*/, 0xff800000, s4
	s_or_b32 s4, s28, s13
	s_set_vgpr_msb 0x102
	v_cndmask_b32_e64 v0, v24 /*v536*/, 0xff800000, s0
	s_or_b32 s0, s18, s1
	s_set_vgpr_msb 0x201
	v_cndmask_b32_e64 v15, v73 /*v329*/, 0xff800000, s72
	v_readlane_b32 s72, v0 /*v256*/, 27
	s_or_b32 s56, s53, s13
	v_cndmask_b32_e64 v7, v255 /*v511*/, 0xff800000, s4
	s_or_b32 s4, s29, s11
	v_readlane_b32 s73, v0 /*v256*/, 28
	v_readlane_b32 s74, v0 /*v256*/, 29
	v_readlane_b32 s75, v0 /*v256*/, 30
	v_readlane_b32 s76, v0 /*v256*/, 31
	v_readlane_b32 s77, v1 /*v257*/, 0
	v_readlane_b32 s78, v1 /*v257*/, 1
	v_readlane_b32 s79, v1 /*v257*/, 2
	s_set_vgpr_msb 0x102
	v_cndmask_b32_e64 v1, v25 /*v537*/, 0xff800000, s0
	s_set_vgpr_msb 0x201
	v_cndmask_b32_e64 v109, v191 /*v447*/, 0xff800000, s56
	s_or_b32 s56, s54, s11
	s_or_b32 s15, s21, s15
	s_set_vgpr_msb 0x102
	v_cndmask_b32_e64 v4, v0 /*v512*/, 0xff800000, s4
	s_or_b32 s4, s30, s8
	s_set_vgpr_msb 0x201
	v_cndmask_b32_e64 v110, v192 /*v448*/, 0xff800000, s56
	s_or_b32 s56, s55, s8
	s_or_b32 s37, s40, s37
	s_or_b32 s35, s41, s35
	s_or_b32 s16, s20, s16
	s_set_vgpr_msb 0x142
	v_cndmask_b32_e64 v54 /*v310*/, v16 /*v528*/, 0xff800000, s15
	s_or_b32 s15, s18, s17
	s_or_b32 s12, s24, s12
	s_or_b32 s10, s23, s10
	s_or_b32 s7, s25, s7
	s_set_vgpr_msb 0x4202
	v_cndmask_b32_e64 v5, v1 /*v513*/, 0xff800000, s4
	s_or_b32 s4, s31, s5
	s_or_b32 s2, s33, s2
	s_set_vgpr_msb 0x201
	v_cndmask_b32_e64 v111, v193 /*v449*/, 0xff800000, s56
	v_cndmask_b32_e64 v82, v106 /*v362*/, 0xff800000, s37
	v_cndmask_b32_e64 v83, v107 /*v363*/, 0xff800000, s35
	s_set_vgpr_msb 0x142
	v_cndmask_b32_e64 v53 /*v309*/, v15 /*v527*/, 0xff800000, s16
	v_cndmask_b32_e64 v55 /*v311*/, v17 /*v529*/, 0xff800000, s15
	s_set_vgpr_msb 0x4241
	v_cndmask_b32_e64 v40 /*v296*/, v250 /*v506*/, 0xff800000, s12
	v_cndmask_b32_e64 v41 /*v297*/, v251 /*v507*/, 0xff800000, s10
	v_cndmask_b32_e64 v42 /*v298*/, v252 /*v508*/, 0xff800000, s7
	s_set_vgpr_msb 0x4142
	v_cndmask_b32_e64 v32 /*v288*/, v18 /*v530*/, 0xff800000, s4
	v_cndmask_b32_e64 v33 /*v289*/, v19 /*v531*/, 0xff800000, s2
	s_set_vgpr_msb 0x4205
	v_wmma_f32_16x16x32_bf16 v[218:225], v[162:169] /*v[418:425]*/, v[82:89] /*v[338:345]*/, v[218:225]
	s_set_vgpr_msb 0x500
	v_max3_num_f32 v32, v114, v115, v120
	s_set_vgpr_msb 21
	v_max3_num_f32 v33, v90 /*v346*/, v91 /*v347*/, v92 /*v348*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v94, v116, v117, v118
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[186:193], v[154:161] /*v[410:417]*/, v[82:89] /*v[338:345]*/, v[186:193]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v154 /*v666*/ offset:9216
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v154 /*v666*/ offset:13824
	s_set_vgpr_msb 0x4215
	v_max3_num_f32 v95, v56 /*v312*/, v57 /*v313*/, v58 /*v314*/
	s_set_vgpr_msb 0x1504
	v_max3_num_f32 v32, v121, v98 /*v354*/, v32
	s_set_vgpr_msb 0x401
	v_max3_num_f32 v33, v93 /*v349*/, v18, v33
	s_set_vgpr_msb 0x105
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[170:177] /*v[426:433]*/, v[82:89] /*v[338:345]*/, v[154:161]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v154 /*v666*/ offset:9248
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v154 /*v666*/ offset:13856
	s_set_vgpr_msb 0x4204
	v_max3_num_f32 v94, v119, v28 /*v284*/, v94
	s_set_vgpr_msb 0x405
	v_max3_num_f32 v95, v59 /*v315*/, v60 /*v316*/, v95
	s_set_vgpr_msb 0x501
	v_max3_num_f32 v32, v99 /*v355*/, v112, v32
	s_set_vgpr_msb 0x105
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[66:73], v[178:185] /*v[434:441]*/, v[82:89] /*v[338:345]*/, v[66:73]
	s_wait_dscnt 0x4
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v154 /*v666*/ offset:9344
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v154 /*v666*/ offset:13952
	s_set_vgpr_msb 0x4200
	v_max3_num_f32 v96, v98, v99, v88
	s_set_vgpr_msb 16
	v_max3_num_f32 v97, v122, v123, v26 /*v282*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v253, v100, v101, v106
	s_set_vgpr_msb 21
	v_max3_num_f32 v255, v48 /*v304*/, v49 /*v305*/, v46 /*v302*/
	s_set_vgpr_msb 0x1505
	v_wmma_f32_16x16x32_bf16 v[210:217], v[130:137] /*v[386:393]*/, v[82:89] /*v[338:345]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v154 /*v666*/ offset:9376
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v154 /*v666*/ offset:13984
	s_set_vgpr_msb 0x4200
	v_max3_num_f32 v96, v89, v64, v96
	s_set_vgpr_msb 1
	v_max3_num_f32 v97, v27 /*v283*/, v16, v97
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v253, v107, v108, v253
	s_set_vgpr_msb 5
	v_max3_num_f32 v255, v47 /*v303*/, v52 /*v308*/, v255
	v_wmma_f32_16x16x32_bf16 v[178:185], v[122:129] /*v[378:385]*/, v[82:89] /*v[338:345]*/, v[178:185]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v154 /*v666*/ offset:9280
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v154 /*v666*/ offset:13888
	s_set_vgpr_msb 0x4255
	v_max3_num_f32 v70 /*v326*/, v34 /*v290*/, v35 /*v291*/, v36 /*v292*/
	v_max3_num_f32 v71 /*v327*/, v50 /*v306*/, v51 /*v307*/, v44 /*v300*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v72 /*v328*/, v82, v83, v84
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v73 /*v329*/, v40 /*v296*/, v41 /*v297*/, v42 /*v298*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[146:153], v[146:153] /*v[402:409]*/, v[82:89] /*v[338:345]*/, v[146:153]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v154 /*v666*/ offset:9312
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v154 /*v666*/ offset:13920
	s_set_vgpr_msb 0x4251
	v_max3_num_f32 v70 /*v326*/, v37 /*v293*/, v102, v70 /*v326*/
	v_max3_num_f32 v71 /*v327*/, v45 /*v301*/, v10, v71 /*v327*/
	s_set_vgpr_msb 0x5150
	v_max3_num_f32 v72 /*v328*/, v85, v86, v72 /*v328*/
	s_set_vgpr_msb 0x5051
	v_max3_num_f32 v73 /*v329*/, v43 /*v299*/, v6, v73 /*v329*/
	s_set_vgpr_msb 0x5105
	v_wmma_f32_16x16x32_bf16 v[50:57], v[138:145] /*v[394:401]*/, v[82:89] /*v[338:345]*/, v[50:57]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v154 /*v666*/ offset:9408
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v154 /*v666*/ offset:14016
	s_set_vgpr_msb 0x4240
	v_max3_num_f32 v94 /*v350*/, v28, v29, v22
	v_max3_num_f32 v95 /*v351*/, v124, v125, v126
	v_max3_num_f32 v96 /*v352*/, v60, v61, v62
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v97 /*v353*/, v32 /*v288*/, v33 /*v289*/, v38 /*v294*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[202:209], v[162:169] /*v[418:425]*/, v[74:81] /*v[330:337]*/, v[202:209]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v154 /*v666*/ offset:9440
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v154 /*v666*/ offset:14048
	s_set_vgpr_msb 0x4250
	v_max3_num_f32 v94 /*v350*/, v23, v24, v94 /*v350*/
	v_max3_num_f32 v95 /*v351*/, v127, v128, v95 /*v351*/
	v_max3_num_f32 v96 /*v352*/, v63, v30, v96 /*v352*/
	s_set_vgpr_msb 0x5051
	v_max3_num_f32 v97 /*v353*/, v39 /*v295*/, v2, v97 /*v353*/
	s_set_vgpr_msb 0x5105
	v_wmma_f32_16x16x32_bf16 v[170:177], v[154:161] /*v[410:417]*/, v[74:81] /*v[330:337]*/, v[170:177]
	s_set_vgpr_msb 0x500
	v_max3_num_f32 v33, v19, v20, v33
	s_set_vgpr_msb 0x50
	v_max3_num_f32 v70 /*v326*/, v103, v104, v70 /*v326*/
	v_max3_num_f32 v71 /*v327*/, v11, v12, v71 /*v327*/
	v_max3_num_f32 v72 /*v328*/, v87, v80, v72 /*v328*/
	v_max3_num_f32 v94 /*v350*/, v25, v26, v94 /*v350*/
	s_set_vgpr_msb 0x5005
	v_wmma_f32_16x16x32_bf16 v[138:145], v[170:177] /*v[426:433]*/, v[74:81] /*v[330:337]*/, v[138:145]
	v_max3_num_f32 v94, v29 /*v285*/, v30 /*v286*/, v94
	s_set_vgpr_msb 0x500
	v_max3_num_f32 v96, v65, v78, v96
	v_max3_num_f32 v97, v17, v14, v97
	s_set_vgpr_msb 0x50
	v_max3_num_f32 v95 /*v351*/, v129, v8, v95 /*v351*/
	s_set_vgpr_msb 0x5005
	v_wmma_f32_16x16x32_bf16 v[42:49], v[178:185] /*v[434:441]*/, v[74:81] /*v[330:337]*/, v[42:49]
	s_set_vgpr_msb 0x501
	v_max3_num_f32 v95, v61 /*v317*/, v76, v95
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v32, v113, v32, v114
	s_set_vgpr_msb 16
	v_max3_num_f32 v33, v21, v33, v90 /*v346*/
	s_set_vgpr_msb 0x1001
	v_max3_num_f32 v94, v31 /*v287*/, v94, v116
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v253, v109, v110, v253
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[194:201], v[130:137] /*v[386:393]*/, v[74:81] /*v[330:337]*/, v[194:201]
	s_set_vgpr_msb 0x510
	v_max3_num_f32 v95, v77, v95, v56 /*v312*/
	s_set_vgpr_msb 0x1005
	v_max3_num_f32 v255, v53 /*v309*/, v54 /*v310*/, v255
	s_set_vgpr_msb 0x500
	v_max3_num_f32 v96, v79, v96, v98
	s_set_vgpr_msb 0x50
	v_max3_num_f32 v73 /*v329*/, v7, v4, v73 /*v329*/
	s_set_vgpr_msb 0x5054
	v_max3_num_f32 v116 /*v372*/, v105, v70 /*v326*/, v34 /*v290*/
	s_set_vgpr_msb 0x5405
	v_wmma_f32_16x16x32_bf16 v[162:169], v[122:129] /*v[378:385]*/, v[74:81] /*v[330:337]*/, v[162:169]
	s_set_vgpr_msb 0x500
	v_max3_num_f32 v97, v15, v97, v122
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54
	v_max3_num_f32 v124 /*v380*/, v13, v71 /*v327*/, v50 /*v306*/
	s_set_vgpr_msb 0x5444
	v_max3_num_f32 v117 /*v373*/, v81, v72 /*v328*/, v82
	s_set_vgpr_msb 0x4450
	v_max3_num_f32 v70 /*v326*/, v31, v58, v96 /*v352*/
	v_max3_num_f32 v71 /*v327*/, v3, v0, v97 /*v353*/
	s_set_vgpr_msb 0x5005
	v_wmma_f32_16x16x32_bf16 v[130:137], v[146:153] /*v[402:409]*/, v[74:81] /*v[330:337]*/, v[130:137]
	s_set_vgpr_msb 0x554
	v_max3_num_f32 v96 /*v352*/, v5, v73 /*v329*/, v40 /*v296*/
	s_set_vgpr_msb 0x5444
	v_max3_num_f32 v94 /*v350*/, v27, v94 /*v350*/, v28
	v_max3_num_f32 v95 /*v351*/, v9, v95 /*v351*/, v124
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v253, v111, v253, v100
	s_set_vgpr_msb 17
	v_max3_num_f32 v255, v55 /*v311*/, v255, v48 /*v304*/
	s_set_vgpr_msb 0x1105
	v_wmma_f32_16x16x32_bf16 v[34:41], v[138:145] /*v[394:401]*/, v[74:81] /*v[330:337]*/, v[34:41]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x544
	v_max3_num_f32 v78 /*v334*/, v59, v70 /*v326*/, v60
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x4454
	v_max3_num_f32 v79 /*v335*/, v1, v71 /*v327*/, v32 /*v288*/
	s_set_vgpr_msb 0x5405
	v_wmma_f32_16x16x32_bf16 v[218:225], v[62:69] /*v[318:325]*/, v[10:17] /*v[266:273]*/, v[218:225]
	s_set_vgpr_msb 0x500
	v_max3_num_f32 v32, v32, v33, v94
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[186:193], v[100:107] /*v[356:363]*/, v[10:17] /*v[266:273]*/, v[186:193]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[70:73] /*v[326:329]*/, v154 /*v666*/ offset:18432
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v154 /*v666*/ offset:23040
	s_set_vgpr_msb 0x4200
	v_max3_num_f32 v32, v32, v95, v33
	v_max3_num_f32 v33, v96, v97, v253
	s_set_vgpr_msb 21
	v_max3_num_f32 v94, v116 /*v372*/, v124 /*v380*/, v117 /*v373*/
	v_max3_num_f32 v95, v94 /*v350*/, v95 /*v351*/, v78 /*v334*/
	s_set_vgpr_msb 0x1505
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[194:201] /*v[450:457]*/, v[10:17] /*v[266:273]*/, v[154:161]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v154 /*v666*/ offset:18464
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v154 /*v666*/ offset:23072
	s_set_vgpr_msb 0x4200
	v_add_f32_e32 v96, 0, v32
	v_max3_num_f32 v33, v33, v255, v97
	s_set_vgpr_msb 20
	v_max3_num_f32 v94, v94, v96 /*v352*/, v124 /*v380*/
	v_max3_num_f32 v95, v95, v79 /*v335*/, v95 /*v351*/
	s_set_vgpr_msb 0x1405
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[66:73], v[202:209] /*v[458:465]*/, v[10:17] /*v[266:273]*/, v[66:73]
	s_wait_dscnt 0x4
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v154 /*v666*/ offset:18560
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v154 /*v666*/ offset:23168
	s_set_vgpr_msb 0x4200
	v_add_f32_e32 v97, 0, v33
	s_mov_b32 s0, 0x76543210
	s_delay_alu instid0(SALU_CYCLE_1)
	v_permlanex16_b32 v96, v96, s0, 0xfedcba98
	v_add_f32_e32 v253, 0, v94
	v_add_f32_e32 v255, 0, v95
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[210:217], v[108:115] /*v[364:371]*/, v[10:17] /*v[266:273]*/, v[210:217]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v154 /*v666*/ offset:18592
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v154 /*v666*/ offset:23200
	s_set_vgpr_msb 0x4200
	v_permlanex16_b32 v97, v97, s0, 0xfedcba98
	s_set_vgpr_msb 64
	v_mul_f32_e32 v78 /*v334*/, s88, v254
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[178:185], v[186:193] /*v[442:449]*/, v[10:17] /*v[266:273]*/, v[178:185]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v154 /*v666*/ offset:18496
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v154 /*v666*/ offset:23104
	s_set_vgpr_msb 0x4200
	v_permlanex16_b32 v253, v253, s0, 0xfedcba98
	s_set_vgpr_msb 64
	v_mul_f32_e32 v79 /*v335*/, s88, v252
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[146:153], v[82:89] /*v[338:345]*/, v[10:17] /*v[266:273]*/, v[146:153]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v154 /*v666*/ offset:18528
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v154 /*v666*/ offset:23136
	s_set_vgpr_msb 0x4200
	v_max3_num_f32 v94, v94, v253, v252
	v_max3_num_f32 v32, v32, v96, v254
	v_max3_num_f32 v33, v33, v97, v254
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[50:57], v[162:169] /*v[418:425]*/, v[10:17] /*v[266:273]*/, v[50:57]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[10:13] /*v[266:269]*/, v154 /*v666*/ offset:18624
	ds_load_tr16_b128 v[14:17] /*v[270:273]*/, v154 /*v666*/ offset:23232
	s_set_vgpr_msb 0x4200
	v_permlanex16_b32 v255, v255, s0, 0xfedcba98
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[202:209], v[62:69] /*v[318:325]*/, v[18:25] /*v[274:281]*/, v[202:209]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v154 /*v666*/ offset:18656
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v154 /*v666*/ offset:23264
	s_set_vgpr_msb 0x4200
	v_max3_num_f32 v95, v95, v255, v252
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[170:177], v[100:107] /*v[356:363]*/, v[18:25] /*v[274:281]*/, v[170:177]
	s_set_vgpr_msb 0x510
	v_max3_num_f32 v32, v32, v33, v78 /*v334*/
	v_max3_num_f32 v97, v94, v95, v79 /*v335*/
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[138:145], v[194:201] /*v[450:457]*/, v[18:25] /*v[274:281]*/, v[138:145]
	s_set_vgpr_msb 0x510
	v_fma_f32 v33, -v32, s88, v78 /*v334*/
	v_fma_f32 v94, -v97, s88, v79 /*v335*/
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[42:49], v[202:209] /*v[458:465]*/, v[18:25] /*v[274:281]*/, v[42:49]
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v96, v33
	v_exp_f32_e32 v94, v94
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[194:201], v[108:115] /*v[364:371]*/, v[18:25] /*v[274:281]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[186:193] /*v[442:449]*/, v[18:25] /*v[274:281]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[82:89] /*v[338:345]*/, v[18:25] /*v[274:281]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[162:169] /*v[418:425]*/, v[18:25] /*v[274:281]*/, v[34:41]
	s_wait_dscnt 0xc
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[218:225], v[70:77] /*v[326:333]*/, v[242:249], v[218:225]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v154 /*v666*/ offset:27648
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v154 /*v666*/ offset:32256
	s_set_vgpr_msb 0x4200
	v_mov_b64_e32 v[254:255], s[88:89]
	v_mul_f32_e64 v252, s88, -v32
	s_set_vgpr_msb 64
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[110:111] /*v[366:367]*/, v[114:115], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4000
	v_pk_fma_f32 v[120:121], v[120:121], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x41
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[98:99] /*v[354:355]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[186:193], v[116:123] /*v[372:379]*/, v[242:249], v[186:193]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v154 /*v666*/ offset:27680
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v154 /*v666*/ offset:32288
	s_set_vgpr_msb 0x4240
	v_pk_fma_f32 v[114:115] /*v[370:371]*/, v[112:113], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4041
	v_pk_fma_f32 v[172:173] /*v[428:429]*/, v[90:91] /*v[346:347]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[174:175] /*v[430:431]*/, v[92:93] /*v[348:349]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4101
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[154:161], v[140:147] /*v[396:403]*/, v[242:249], v[154:161]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v154 /*v666*/ offset:27776
	ds_load_tr16_b128 v[90:93] /*v[346:349]*/, v154 /*v666*/ offset:32384
	s_set_vgpr_msb 0x4240
	v_pk_fma_f32 v[176:177] /*v[432:433]*/, v[98:99], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_mul_f32_e64 v178 /*v434*/, s88, -v97
	s_set_vgpr_msb 0x4051
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[34:35] /*v[290:291]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[36:37] /*v[292:293]*/, v[36:37] /*v[292:293]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[66:73], v[148:155] /*v[404:411]*/, v[242:249], v[66:73]
	s_wait_dscnt 0x6
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[94:97] /*v[350:353]*/, v154 /*v666*/ offset:27808
	ds_load_tr16_b128 v[98:101] /*v[354:357]*/, v154 /*v666*/ offset:32416
	s_set_vgpr_msb 0x4240
	v_pk_fma_f32 v[180:181] /*v[436:437]*/, v[88:89], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4050
	v_pk_fma_f32 v[182:183] /*v[438:439]*/, v[102:103], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[184:185] /*v[440:441]*/, v[104:105], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5001
	v_wmma_f32_16x16x32_bf16 v[210:217], v[124:131] /*v[380:387]*/, v[242:249], v[210:217]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[102:105] /*v[358:361]*/, v154 /*v666*/ offset:27712
	ds_load_tr16_b128 v[106:109] /*v[362:365]*/, v154 /*v666*/ offset:32320
	s_set_vgpr_msb 0x4211
	v_pk_fma_f32 v[98:99], v[50:51] /*v[306:307]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1140
	v_pk_fma_f32 v[50:51] /*v[306:307]*/, v[64:65], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[186:187] /*v[442:443]*/, v[78:79], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[178:185], v[132:139] /*v[388:395]*/, v[242:249], v[178:185]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v154 /*v666*/ offset:27744
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v154 /*v666*/ offset:32352
	s_set_vgpr_msb 0x4211
	v_pk_fma_f32 v[102:103], v[44:45] /*v[300:301]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1100
	v_pk_fma_f32 v[122:123], v[122:123], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[64:65], v[26:27] /*v[282:283]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[10:17] /*v[266:273]*/, v[242:249], v[146:153]
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v154 /*v666*/ offset:27840
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v154 /*v666*/ offset:32448
	s_set_vgpr_msb 0x4250
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[28:29], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5010
	v_pk_fma_f32 v[22:23], v[22:23], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1050
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[24:25], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[188:189] /*v[444:445]*/, v[26:27], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5001
	v_wmma_f32_16x16x32_bf16 v[50:57], v[62:69] /*v[318:325]*/, v[242:249], v[50:57]
	s_set_vgpr_msb 0x102
	ds_load_tr16_b128 v[242:245], v154 /*v666*/ offset:27872
	ds_load_tr16_b128 v[246:249], v154 /*v666*/ offset:32480
	s_set_vgpr_msb 0x210
	v_pk_fma_f32 v[88:89], v[124:125], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[78:79], v[126:127], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[28:29], v[128:129], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[202:209], v[70:77] /*v[326:333]*/, v[2:9] /*v[258:265]*/, v[202:209]
	s_set_vgpr_msb 0x500
	v_pk_fma_f32 v[18:19], v[18:19], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[20:21], v[20:21], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[124:125], v[116:117], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[126:127], v[118:119], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[128:129], v[28:29] /*v[284:285]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x141
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[30:31] /*v[286:287]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[170:177], v[116:123] /*v[372:379]*/, v[2:9] /*v[258:265]*/, v[170:177]
	s_set_vgpr_msb 0x541
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[56:57] /*v[312:313]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[56:57] /*v[312:313]*/, v[58:59] /*v[314:315]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[58:59] /*v[314:315]*/, v[60:61] /*v[316:317]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4140
	v_pk_fma_f32 v[60:61] /*v[316:317]*/, v[16:17], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[138:145], v[140:147] /*v[396:403]*/, v[2:9] /*v[258:265]*/, v[138:145]
	s_set_vgpr_msb 0x500
	v_pk_fma_f32 v[14:15], v[14:15], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 64
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[100:101], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[106:107], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[108:109], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[110:111], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4041
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[48:49] /*v[304:305]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[42:49], v[148:155] /*v[404:411]*/, v[2:9] /*v[258:265]*/, v[42:49]
	s_set_vgpr_msb 0x541
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[46:47] /*v[302:303]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[52:53] /*v[308:309]*/, v[52:53] /*v[308:309]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[54:55] /*v[310:311]*/, v[54:55] /*v[310:311]*/, v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x4110
	v_pk_fma_f32 v[114:115], v[10:11], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[110:111], v[12:13], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[194:201], v[124:131] /*v[380:387]*/, v[2:9] /*v[258:265]*/, v[194:201]
	s_set_vgpr_msb 0x510
	v_pk_fma_f32 v[116:117], v[82:83], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[112:113], v[84:85], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[84:85], v[86:87], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[82:83], v[80:81], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[26:27], v[40:41] /*v[296:297]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[24:25], v[42:43] /*v[298:299]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1105
	v_wmma_f32_16x16x32_bf16 v[162:169], v[132:139] /*v[388:395]*/, v[2:9] /*v[258:265]*/, v[162:169]
	s_set_vgpr_msb 0x510
	v_pk_fma_f32 v[108:109], v[8:9], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[104:105], v[60:61], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[86:87], v[62:63], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1005
	v_wmma_f32_16x16x32_bf16 v[130:137], v[10:17] /*v[266:273]*/, v[2:9] /*v[258:265]*/, v[130:137]
	s_set_vgpr_msb 0x510
	v_pk_fma_f32 v[106:107], v[30:31], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[100:101], v[58:59], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[80:81], v[32:33] /*v[288:289]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	v_pk_fma_f32 v[30:31], v[38:39] /*v[294:295]*/, v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1105
	v_wmma_f32_16x16x32_bf16 v[34:41], v[62:69] /*v[318:325]*/, v[2:9] /*v[258:265]*/, v[34:41]
	s_set_vgpr_msb 0x500
	v_pk_fma_f32 v[8:9], v[76:77], v[254:255], v[252:253] op_sel_hi:[1,1,0]
	s_wait_dscnt 0xc
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[218:225], v[18:25] /*v[274:281]*/, v[226:233], v[218:225]
	v_exp_f32_e32 v12, v110 /*v366*/
	v_exp_f32_e32 v13, v111 /*v367*/
	v_wmma_f32_16x16x32_bf16 v[186:193], v[78:85] /*v[334:341]*/, v[226:233], v[186:193]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v62, v120
	s_set_vgpr_msb 1
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[154:161], v[102:109] /*v[358:365]*/, v[226:233], v[154:161]
	v_exp_f32_e32 v76, v176 /*v432*/
	v_exp_f32_e32 v77, v177 /*v433*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[66:73], v[156:163] /*v[412:419]*/, v[226:233], v[66:73]
	s_wait_dscnt 0x9
	v_exp_f32_e32 v252, v180 /*v436*/
	v_exp_f32_e32 v253, v181 /*v437*/
	v_wmma_f32_16x16x32_bf16 v[210:217], v[86:93] /*v[342:349]*/, v[226:233], v[210:217]
	s_set_vgpr_msb 0x150
	v_pk_fma_f32 v[2:3] /*v[258:259]*/, v[6:7], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5041
	v_exp_f32_e32 v4 /*v260*/, v34 /*v290*/
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[178:185], v[94:101] /*v[350:357]*/, v[226:233], v[178:185]
	s_set_vgpr_msb 0x150
	v_pk_fma_f32 v[6:7] /*v[262:263]*/, v[4:5], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5041
	v_exp_f32_e32 v5 /*v261*/, v35 /*v291*/
	s_set_vgpr_msb 0x4101
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[146:153], v[164:171] /*v[420:427]*/, v[226:233], v[146:153]
	s_set_vgpr_msb 0x150
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[2:3], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x5001
	v_exp_f32_e32 v118, v26 /*v282*/
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57], v[242:249], v[226:233], v[50:57]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 16
	v_pk_fma_f32 v[226:227], v[0:1], v[254:255], v[178:179] /*v[434:435]*/ op_sel_hi:[1,1,0]
	s_set_vgpr_msb 0x1001
	v_exp_f32_e32 v119, v27 /*v283*/
	v_wmma_f32_16x16x32_bf16 v[202:209], v[18:25] /*v[274:281]*/, v[234:241], v[202:209]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v63, v121
	s_set_vgpr_msb 1
	v_exp_f32_e32 v4, v112 /*v368*/
	v_exp_f32_e32 v5, v113 /*v369*/
	v_wmma_f32_16x16x32_bf16 v[170:177], v[78:85] /*v[334:341]*/, v[234:241], v[170:177]
	v_exp_f32_e32 v6, v114 /*v370*/
	v_exp_f32_e32 v7, v115 /*v371*/
	v_exp_f32_e32 v120, v50 /*v306*/
	v_wmma_f32_16x16x32_bf16 v[138:145], v[102:109] /*v[358:365]*/, v[234:241], v[138:145]
	v_exp_f32_e32 v121, v51 /*v307*/
	v_exp_f32_e32 v228, v186 /*v442*/
	v_exp_f32_e32 v229, v187 /*v443*/
	v_wmma_f32_16x16x32_bf16 v[42:49], v[156:163] /*v[412:419]*/, v[234:241], v[42:49]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v230, v22
	v_exp_f32_e32 v231, v23
	s_set_vgpr_msb 1
	v_exp_f32_e32 v232, v44 /*v300*/
	v_wmma_f32_16x16x32_bf16 v[194:201], v[86:93] /*v[342:349]*/, v[234:241], v[194:201]
	v_exp_f32_e32 v254, v36 /*v292*/
	v_exp_f32_e32 v255, v37 /*v293*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v10 /*v266*/, v182 /*v438*/
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[162:169], v[94:101] /*v[350:357]*/, v[234:241], v[162:169]
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v11 /*v267*/, v183 /*v439*/
	v_exp_f32_e32 v12 /*v268*/, v184 /*v440*/
	v_exp_f32_e32 v13 /*v269*/, v185 /*v441*/
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[130:137], v[164:171] /*v[420:427]*/, v[234:241], v[130:137]
	v_exp_f32_e32 v233, v45 /*v301*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v14 /*v270*/, v188 /*v444*/
	v_exp_f32_e32 v15 /*v271*/, v189 /*v445*/
	s_set_vgpr_msb 0x4100
	v_wmma_f32_16x16x32_bf16 v[34:41], v[242:249], v[234:241], v[34:41]
	s_wait_dscnt 0xc
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 1
	v_exp_f32_e32 v22, v172 /*v428*/
	v_exp_f32_e32 v23, v173 /*v429*/
	v_nop
	v_nop
	v_exp_f32_e32 v234, v174 /*v430*/
	v_exp_f32_e32 v235, v175 /*v431*/
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v236, v18
	v_exp_f32_e32 v237, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	v_exp_f32_e32 v128, v128
	v_exp_f32_e32 v129, v129
	s_set_vgpr_msb 1
	v_exp_f32_e32 v238, v28 /*v284*/
	v_exp_f32_e32 v239, v29 /*v285*/
	v_exp_f32_e32 v240, v30 /*v286*/
	v_exp_f32_e32 v241, v31 /*v287*/
	v_exp_f32_e32 v242, v56 /*v312*/
	v_exp_f32_e32 v243, v57 /*v313*/
	v_exp_f32_e32 v244, v58 /*v314*/
	v_exp_f32_e32 v245, v59 /*v315*/
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v246, v8
	v_exp_f32_e32 v247, v9
	v_cvt_pk_bf16_f32 v58, v12, v13
	v_cvt_pk_bf16_f32 v59, v62, v63
	v_cvt_pk_bf16_f32 v60, v4, v5
	v_cvt_pk_bf16_f32 v61, v6, v7
	v_cvt_pk_bf16_f32 v16, v22, v23
	v_cvt_pk_bf16_f32 v17, v234, v235
	v_cvt_pk_bf16_f32 v18, v236, v237
	v_cvt_pk_bf16_f32 v19, v20, v21
	v_cvt_pk_bf16_f32 v8, v124, v125
	v_cvt_pk_bf16_f32 v9, v126, v127
	v_cvt_pk_bf16_f32 v10, v128, v129
	v_cvt_pk_bf16_f32 v11, v238, v239
	v_cvt_pk_bf16_f32 v0, v240, v241
	v_cvt_pk_bf16_f32 v1, v242, v243
	v_cvt_pk_bf16_f32 v2, v244, v245
	v_cvt_pk_bf16_f32 v3, v246, v247
	v_pk_add_f32 v[12:13], v[12:13], v[62:63]
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]
	v_pk_add_f32 v[6:7], v[22:23], v[234:235]
	v_pk_add_f32 v[20:21], v[236:237], v[20:21]
	v_pk_add_f32 v[22:23], v[124:125], v[126:127]
	v_pk_add_f32 v[62:63], v[128:129], v[238:239]
	v_pk_add_f32 v[124:125], v[240:241], v[242:243]
	v_pk_add_f32 v[126:127], v[244:245], v[246:247]
	v_pk_add_f32 v[4:5], v[12:13], v[4:5]
	v_pk_add_f32 v[6:7], v[6:7], v[20:21]
	v_pk_add_f32 v[12:13], v[22:23], v[62:63]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[20:21], v[124:125], v[126:127]
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[6:7], v[12:13], v[20:21]
	v_pk_add_f32 v[12:13], v[4:5], v[6:7]
	v_dual_mov_b32 v4, v74 :: v_dual_mov_b32 v6, v75
	s_set_vgpr_msb 2
	v_mov_b32_e32 v20, v70 /*v582*/
	s_set_vgpr_msb 0x200
	v_mov_b32_e32 v21, v12
	v_pk_add_f32 v[4:5], v[4:5], v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[4:5], v[20:21], v[4:5]
	v_mul_f32_e32 v4, v4, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v95, v4, v5
	v_exp_f32_e32 v74, v122
	v_exp_f32_e32 v75, v123
	v_exp_f32_e32 v122, v64
	v_exp_f32_e32 v123, v65
	s_set_vgpr_msb 1
	v_exp_f32_e32 v124, v60 /*v316*/
	v_exp_f32_e32 v125, v61 /*v317*/
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v126, v14
	v_exp_f32_e32 v127, v15
	s_set_vgpr_msb 1
	v_exp_f32_e32 v128, v70 /*v326*/
	v_exp_f32_e32 v129, v71 /*v327*/
	v_exp_f32_e32 v234, v72 /*v328*/
	v_exp_f32_e32 v235, v73 /*v329*/
	v_exp_f32_e32 v236, v74 /*v330*/
	v_exp_f32_e32 v237, v75 /*v331*/
	v_exp_f32_e32 v238, v76 /*v332*/
	v_exp_f32_e32 v239, v77 /*v333*/
	v_exp_f32_e32 v240, v48 /*v304*/
	v_exp_f32_e32 v241, v49 /*v305*/
	v_exp_f32_e32 v242, v46 /*v302*/
	v_exp_f32_e32 v243, v47 /*v303*/
	v_exp_f32_e32 v244, v52 /*v308*/
	v_exp_f32_e32 v245, v53 /*v309*/
	v_exp_f32_e32 v246, v54 /*v310*/
	v_exp_f32_e32 v247, v55 /*v311*/
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v62, v76, v77
	v_cvt_pk_bf16_f32 v63, v252, v253
	v_cvt_pk_bf16_f32 v64, v120, v121
	v_cvt_pk_bf16_f32 v65, v228, v229
	v_cvt_pk_bf16_f32 v20, v74, v75
	v_cvt_pk_bf16_f32 v21, v122, v123
	v_cvt_pk_bf16_f32 v22, v124, v125
	v_cvt_pk_bf16_f32 v23, v126, v127
	v_cvt_pk_bf16_f32 v12, v128, v129
	v_cvt_pk_bf16_f32 v13, v234, v235
	v_cvt_pk_bf16_f32 v14, v236, v237
	v_cvt_pk_bf16_f32 v15, v238, v239
	v_cvt_pk_bf16_f32 v4, v240, v241
	v_cvt_pk_bf16_f32 v5, v242, v243
	v_cvt_pk_bf16_f32 v6, v244, v245
	v_cvt_pk_bf16_f32 v7, v246, v247
	v_pk_add_f32 v[76:77], v[76:77], v[252:253]
	v_pk_add_f32 v[120:121], v[120:121], v[228:229]
	v_pk_add_f32 v[74:75], v[74:75], v[122:123]
	v_pk_add_f32 v[122:123], v[124:125], v[126:127]
	v_pk_add_f32 v[124:125], v[128:129], v[234:235]
	v_pk_add_f32 v[126:127], v[236:237], v[238:239]
	v_pk_add_f32 v[128:129], v[240:241], v[242:243]
	v_pk_add_f32 v[228:229], v[244:245], v[246:247]
	v_pk_add_f32 v[76:77], v[76:77], v[120:121]
	v_pk_add_f32 v[74:75], v[74:75], v[122:123]
	v_pk_add_f32 v[120:121], v[124:125], v[126:127]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[122:123], v[128:129], v[228:229]
	v_pk_add_f32 v[74:75], v[76:77], v[74:75]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[76:77], v[120:121], v[122:123]
	v_pk_add_f32 v[120:121], v[74:75], v[76:77]
	v_dual_mov_b32 v74, v250 :: v_dual_mov_b32 v76, v251
	s_set_vgpr_msb 2
	v_mov_b32_e32 v122, v71 /*v583*/
	s_set_vgpr_msb 0x200
	v_mov_b32_e32 v123, v120
	v_pk_add_f32 v[74:75], v[74:75], v[76:77]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[74:75], v[122:123], v[74:75]
	v_mul_f32_e32 v33, v74, v96
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v33, v33, v75
	v_exp_f32_e32 v98, v98
	v_exp_f32_e32 v99, v99
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v114, v114
	v_exp_f32_e32 v115, v115
	v_exp_f32_e32 v120, v110
	v_exp_f32_e32 v121, v111
	v_exp_f32_e32 v116, v116
	v_exp_f32_e32 v117, v117
	v_exp_f32_e32 v122, v112
	v_exp_f32_e32 v123, v113
	v_exp_f32_e32 v124, v84
	v_exp_f32_e32 v125, v85
	v_exp_f32_e32 v126, v82
	v_exp_f32_e32 v127, v83
	v_exp_f32_e32 v128, v26
	v_exp_f32_e32 v129, v27
	v_exp_f32_e32 v228, v24
	v_exp_f32_e32 v229, v25
	s_set_vgpr_msb 5
	v_exp_f32_e32 v234, v2 /*v258*/
	v_exp_f32_e32 v235, v3 /*v259*/
	v_exp_f32_e32 v236, v6 /*v262*/
	v_exp_f32_e32 v237, v7 /*v263*/
	v_cvt_pk_bf16_f32 v110, v4 /*v260*/, v5 /*v261*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v111, v254, v255
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v112, v10 /*v266*/, v11 /*v267*/
	v_cvt_pk_bf16_f32 v113, v12 /*v268*/, v13 /*v269*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v82, v98, v99
	v_cvt_pk_bf16_f32 v83, v102, v103
	v_cvt_pk_bf16_f32 v84, v114, v115
	v_cvt_pk_bf16_f32 v85, v120, v121
	v_cvt_pk_bf16_f32 v74, v116, v117
	v_cvt_pk_bf16_f32 v75, v122, v123
	v_cvt_pk_bf16_f32 v76, v124, v125
	v_cvt_pk_bf16_f32 v77, v126, v127
	v_cvt_pk_bf16_f32 v24, v128, v129
	v_cvt_pk_bf16_f32 v25, v228, v229
	v_cvt_pk_bf16_f32 v26, v234, v235
	v_cvt_pk_bf16_f32 v27, v236, v237
	s_set_vgpr_msb 1
	v_pk_add_f32 v[238:239], v[4:5] /*v[260:261]*/, v[254:255]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[240:241], v[10:11] /*v[266:267]*/, v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[98:99], v[98:99], v[102:103]
	v_pk_add_f32 v[102:103], v[114:115], v[120:121]
	v_pk_add_f32 v[114:115], v[116:117], v[122:123]
	v_pk_add_f32 v[116:117], v[124:125], v[126:127]
	v_pk_add_f32 v[120:121], v[128:129], v[228:229]
	v_pk_add_f32 v[122:123], v[234:235], v[236:237]
	v_pk_add_f32 v[124:125], v[238:239], v[240:241]
	v_pk_add_f32 v[98:99], v[98:99], v[102:103]
	v_pk_add_f32 v[102:103], v[114:115], v[116:117]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[114:115], v[120:121], v[122:123]
	v_pk_add_f32 v[98:99], v[124:125], v[98:99]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[102:103], v[102:103], v[114:115]
	v_pk_add_f32 v[98:99], v[98:99], v[102:103]
	v_exp_f32_e32 v102, v88
	v_exp_f32_e32 v103, v89
	v_exp_f32_e32 v120, v78
	v_exp_f32_e32 v121, v79
	v_exp_f32_e32 v122, v28
	v_exp_f32_e32 v123, v29
	v_exp_f32_e32 v108, v108
	v_exp_f32_e32 v109, v109
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v124, v86
	v_exp_f32_e32 v125, v87
	v_exp_f32_e32 v106, v106
	v_exp_f32_e32 v107, v107
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v126, v80
	v_exp_f32_e32 v127, v81
	v_exp_f32_e32 v128, v30
	v_exp_f32_e32 v129, v31
	s_set_vgpr_msb 1
	v_exp_f32_e32 v228, v8 /*v264*/
	v_exp_f32_e32 v229, v9 /*v265*/
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v226, v226
	v_exp_f32_e32 v227, v227
	v_cvt_pk_bf16_f32 v114, v118, v119
	v_cvt_pk_bf16_f32 v115, v230, v231
	v_cvt_pk_bf16_f32 v116, v232, v233
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v117, v14 /*v270*/, v15 /*v271*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v86, v102, v103
	v_cvt_pk_bf16_f32 v87, v120, v121
	v_cvt_pk_bf16_f32 v88, v122, v123
	v_cvt_pk_bf16_f32 v89, v108, v109
	v_cvt_pk_bf16_f32 v78, v104, v105
	v_cvt_pk_bf16_f32 v79, v124, v125
	v_cvt_pk_bf16_f32 v80, v106, v107
	v_cvt_pk_bf16_f32 v81, v100, v101
	v_cvt_pk_bf16_f32 v28, v126, v127
	v_cvt_pk_bf16_f32 v29, v128, v129
	v_cvt_pk_bf16_f32 v30, v228, v229
	v_cvt_pk_bf16_f32 v31, v226, v227
	v_pk_add_f32 v[118:119], v[118:119], v[230:231]
	s_set_vgpr_msb 4
	v_pk_add_f32 v[230:231], v[232:233], v[14:15] /*v[270:271]*/
	s_set_vgpr_msb 0x400
	v_pk_add_f32 v[102:103], v[102:103], v[120:121]
	v_pk_add_f32 v[108:109], v[122:123], v[108:109]
	v_pk_add_f32 v[104:105], v[104:105], v[124:125]
	v_pk_add_f32 v[100:101], v[106:107], v[100:101]
	v_pk_add_f32 v[106:107], v[126:127], v[128:129]
	v_pk_add_f32 v[120:121], v[228:229], v[226:227]
	v_pk_add_f32 v[118:119], v[118:119], v[230:231]
	v_pk_add_f32 v[102:103], v[102:103], v[108:109]
	v_pk_add_f32 v[100:101], v[104:105], v[100:101]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[104:105], v[106:107], v[120:121]
	v_pk_add_f32 v[102:103], v[118:119], v[102:103]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_add_f32 v[100:101], v[100:101], v[104:105]
	v_pk_add_f32 v[100:101], v[102:103], v[100:101]
	v_dual_mov_b32 v102, v90 :: v_dual_mov_b32 v103, v92
	v_mov_b32_e32 v92, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[90:91], v[102:103], v[92:93]
	v_dual_mov_b32 v92, v98 :: v_dual_mov_b32 v93, v100
	v_mov_b32_e32 v100, v99
	s_set_vgpr_msb 2
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_add_f32 v[90:91], v[68:69] /*v[580:581]*/, v[90:91]
	s_set_vgpr_msb 0x200
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[92:93], v[92:93], v[100:101]
	v_pk_mul_f32 v[90:91], v[90:91], v[94:95] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_add_f32 v[90:91], v[90:91], v[92:93]
	s_set_vgpr_msb 64
	v_pk_mul_f32 v[90:91] /*v[346:347]*/, v[96:97], v[218:219] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97] /*v[352:353]*/, v[96:97], v[224:225] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95] /*v[350:351]*/, v[96:97], v[222:223] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93] /*v[348:349]*/, v[96:97], v[220:221] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99] /*v[354:355]*/, v[96:97], v[186:187] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105] /*v[360:361]*/, v[96:97], v[192:193] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103] /*v[358:359]*/, v[96:97], v[190:191] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101] /*v[356:357]*/, v[96:97], v[188:189] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107] /*v[362:363]*/, v[96:97], v[154:155] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113] /*v[368:369]*/, v[96:97], v[160:161] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111] /*v[366:367]*/, v[96:97], v[158:159] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109] /*v[364:365]*/, v[96:97], v[156:157] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115] /*v[370:371]*/, v[96:97], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121] /*v[376:377]*/, v[96:97], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[118:119] /*v[374:375]*/, v[96:97], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[116:117] /*v[372:373]*/, v[96:97], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[122:123] /*v[378:379]*/, v[96:97], v[210:211] op_sel_hi:[0,1]
	v_pk_mul_f32 v[128:129] /*v[384:385]*/, v[96:97], v[216:217] op_sel_hi:[0,1]
	v_pk_mul_f32 v[126:127] /*v[382:383]*/, v[96:97], v[214:215] op_sel_hi:[0,1]
	v_pk_mul_f32 v[124:125] /*v[380:381]*/, v[96:97], v[212:213] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[96:97], v[178:179] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[96:97], v[184:185] op_sel_hi:[0,1]
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[96:97], v[182:183] op_sel_hi:[0,1]
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[96:97], v[180:181] op_sel_hi:[0,1]
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[96:97], v[146:147] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[96:97], v[152:153] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[96:97], v[150:151] op_sel_hi:[0,1]
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[96:97], v[148:149] op_sel_hi:[0,1]
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[96:97], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[96:97], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[96:97], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[96:97], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[94:95], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[94:95], v[208:209] op_sel_hi:[0,1]
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[94:95], v[206:207] op_sel_hi:[0,1]
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[94:95], v[204:205] op_sel_hi:[0,1]
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[94:95], v[170:171] op_sel_hi:[0,1]
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[94:95], v[176:177] op_sel_hi:[0,1]
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[94:95], v[174:175] op_sel_hi:[0,1]
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[94:95], v[172:173] op_sel_hi:[0,1]
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[94:95], v[138:139] op_sel_hi:[0,1]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[94:95], v[144:145] op_sel_hi:[0,1]
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[94:95], v[142:143] op_sel_hi:[0,1]
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[94:95], v[140:141] op_sel_hi:[0,1]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[94:95], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[94:95], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[94:95], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[94:95], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[94:95], v[194:195] op_sel_hi:[0,1]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[94:95], v[200:201] op_sel_hi:[0,1]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[94:95], v[198:199] op_sel_hi:[0,1]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[94:95], v[196:197] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[94:95], v[162:163] op_sel_hi:[0,1]
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[94:95], v[168:169] op_sel_hi:[0,1]
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[94:95], v[166:167] op_sel_hi:[0,1]
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[94:95], v[164:165] op_sel_hi:[0,1]
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[94:95], v[130:131] op_sel_hi:[0,1]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[94:95], v[136:137] op_sel_hi:[0,1]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[94:95], v[134:135] op_sel_hi:[0,1]
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[94:95], v[132:133] op_sel_hi:[0,1]
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[94:95], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[94:95], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[94:95], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[94:95], v[36:37] op_sel_hi:[0,1]
	s_set_vgpr_msb 0x4008
	v_add_nc_u32_e32 v92, s96, v152 /*v664*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[34:37], v92
	ds_load_tr16_b128 v[38:41], v92 offset:4608
	ds_load_tr16_b128 v[42:45], v92 offset:32
	ds_load_tr16_b128 v[46:49], v92 offset:4640
	ds_load_tr16_b128 v[50:53], v92 offset:9216
	ds_load_tr16_b128 v[54:57], v92 offset:13824
	ds_load_tr16_b128 v[66:69], v92 offset:9248
	ds_load_tr16_b128 v[70:73], v92 offset:13856
	ds_load_tr16_b128 v[98:101], v92 offset:128
	ds_load_tr16_b128 v[102:105], v92 offset:4736
	ds_load_tr16_b128 v[118:121], v92 offset:160
	ds_load_tr16_b128 v[122:125], v92 offset:4768
	ds_load_tr16_b128 v[126:129], v92 offset:9344
	ds_load_tr16_b128 v[130:133], v92 offset:13952
	ds_load_tr16_b128 v[134:137], v92 offset:9376
	ds_load_tr16_b128 v[138:141], v92 offset:13984
	ds_load_tr16_b128 v[142:145], v92 offset:64
	ds_load_tr16_b128 v[146:149], v92 offset:4672
	ds_load_tr16_b128 v[150:153], v92 offset:96
	ds_load_tr16_b128 v[154:157], v92 offset:4704
	ds_load_tr16_b128 v[158:161], v92 offset:9280
	ds_load_tr16_b128 v[162:165], v92 offset:13888
	ds_load_tr16_b128 v[166:169], v92 offset:9312
	ds_load_tr16_b128 v[170:173], v92 offset:13920
	ds_load_tr16_b128 v[174:177], v92 offset:192
	ds_load_tr16_b128 v[178:181], v92 offset:4800
	ds_load_tr16_b128 v[182:185], v92 offset:224
	ds_load_tr16_b128 v[186:189], v92 offset:4832
	ds_load_tr16_b128 v[190:193], v92 offset:9408
	ds_load_tr16_b128 v[194:197], v92 offset:14016
	ds_load_tr16_b128 v[198:201], v92 offset:9440
	ds_load_tr16_b128 v[202:205], v92 offset:14048
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[34:41], v[58:65], v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[42:49], v[58:65], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[142:149], v[58:65], v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[150:157], v[58:65], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[98:105], v[58:65], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[118:125], v[58:65], v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[174:181], v[58:65], v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[182:189], v[58:65], v[146:153] /*v[402:409]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[34:41], v[110:117], v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[42:49], v[110:117], v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[142:149], v[110:117], v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[150:157], v[110:117], v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[98:105], v[110:117], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[118:125], v[110:117], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[174:181], v[110:117], v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[182:189], v[110:117], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[50:57], v[16:23], v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[66:73], v[16:23], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[158:165], v[16:23], v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[166:173], v[16:23], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[126:133], v[16:23], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[134:141], v[16:23], v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[190:197], v[16:23], v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[198:205], v[16:23], v[146:153] /*v[402:409]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[50:57], v[82:89], v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[66:73], v[82:89], v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[158:165], v[82:89], v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[166:173], v[82:89], v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[126:133], v[82:89], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[134:141], v[82:89], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[190:197], v[82:89], v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[198:205], v[82:89], v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5000
	ds_load_tr16_b128 v[16:19], v92 offset:18432
	ds_load_tr16_b128 v[20:23], v92 offset:23040
	ds_load_tr16_b128 v[34:37], v92 offset:18464
	ds_load_tr16_b128 v[38:41], v92 offset:23072
	ds_load_tr16_b128 v[42:45], v92 offset:27648
	ds_load_tr16_b128 v[46:49], v92 offset:32256
	ds_load_tr16_b128 v[50:53], v92 offset:27680
	ds_load_tr16_b128 v[54:57], v92 offset:32288
	ds_load_tr16_b128 v[58:61], v92 offset:18560
	ds_load_tr16_b128 v[62:65], v92 offset:23168
	ds_load_tr16_b128 v[66:69], v92 offset:18592
	ds_load_tr16_b128 v[70:73], v92 offset:23200
	ds_load_tr16_b128 v[82:85], v92 offset:27776
	ds_load_tr16_b128 v[86:89], v92 offset:32384
	ds_load_tr16_b128 v[98:101], v92 offset:27808
	ds_load_tr16_b128 v[102:105], v92 offset:32416
	ds_load_tr16_b128 v[106:109], v92 offset:18496
	ds_load_tr16_b128 v[110:113], v92 offset:23104
	ds_load_tr16_b128 v[114:117], v92 offset:18528
	ds_load_tr16_b128 v[118:121], v92 offset:23136
	ds_load_tr16_b128 v[122:125], v92 offset:27712
	ds_load_tr16_b128 v[126:129], v92 offset:32320
	ds_load_tr16_b128 v[130:133], v92 offset:27744
	ds_load_tr16_b128 v[134:137], v92 offset:32352
	ds_load_tr16_b128 v[138:141], v92 offset:18624
	ds_load_tr16_b128 v[142:145], v92 offset:23232
	ds_load_tr16_b128 v[146:149], v92 offset:18656
	ds_load_tr16_b128 v[150:153], v92 offset:23264
	ds_load_tr16_b128 v[154:157], v92 offset:27840
	ds_load_tr16_b128 v[158:161], v92 offset:32448
	ds_load_tr16_b128 v[162:165], v92 offset:27872
	ds_load_tr16_b128 v[166:169], v92 offset:32480
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[16:23], v[8:15], v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[34:41], v[8:15], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[106:113], v[8:15], v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[114:121], v[8:15], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[58:65], v[8:15], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[66:73], v[8:15], v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[138:145], v[8:15], v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[146:153], v[8:15], v[146:153] /*v[402:409]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[16:23], v[74:81], v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[34:41], v[74:81], v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[106:113], v[74:81], v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[114:121], v[74:81], v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[58:65], v[74:81], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[66:73], v[74:81], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[138:145], v[74:81], v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[146:153], v[74:81], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[42:49], v[0:7], v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[50:57], v[0:7], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[122:129], v[0:7], v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[130:137], v[0:7], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[82:89], v[0:7], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[98:105], v[0:7], v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[154:161], v[0:7], v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[162:169], v[0:7], v[146:153] /*v[402:409]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[42:49], v[24:31], v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[50:57], v[24:31], v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[122:129], v[24:31], v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[130:137], v[24:31], v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[82:89], v[24:31], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[98:105], v[24:31], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[154:161], v[24:31], v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[162:169], v[24:31], v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5000
	v_dual_add_f32 v0, v90, v91 :: v_dual_mov_b32 v94, s72
	s_bfe_u32 s1, ttmp8, 0x50019
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v2, s72, v97 :: v_dual_mov_b32 v250, s1
	v_mov_b32_e32 v1, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_permlanex16_b32 v1, v1, s0, 0xfedcba98
	s_set_vgpr_msb 64
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v78 /*v334*/, v0, v1
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[0:1], v[94:95], v[32:33]
	v_mov_b32_e32 v33, 0x3f317218
	s_set_vgpr_msb 1
	v_log_f32_e32 v0, v78 /*v334*/
	s_set_vgpr_msb 0x108
	v_mov_b32_e32 v3, v1
	v_permlanex16_b32 v1, v1, s0, 0xfedcba98
	s_lshl_b32 s0, s1, 5
	s_and_not1_b32 s1, s87, exec_lo
	s_add_co_i32 s2, s100, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v255, s2 :: v_dual_bitop2_b32 v251, s2, v148 /*v660*/ bitop3:0x54
	s_set_vgpr_msb 0x800
	v_mul_f32_e32 v0, 0x3f317218, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s99, v251
	v_pk_add_f32 v[0:1], v[2:3], v[0:1]
	s_and_b32 s2, vcc_lo, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_log_f32_e32 v95, v1
	s_or_b32 s87, s1, s2
	v_nop
	v_pk_mul_f32 v[2:3], v[94:95], v[32:33]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v253, v2, v3
.LBB0_20:
	s_set_vgpr_msb 1
	v_readlane_b32 s4, v0 /*v256*/, 10
	v_readlane_b32 s5, v0 /*v256*/, 11
	v_readlane_b32 s6, v0 /*v256*/, 16
	s_and_saveexec_b32 s1, s87
	s_set_vgpr_msb 0x100
	s_cbranch_execz .LBB0_22
	s_set_vgpr_msb 1
	v_readlane_b32 s8, v0 /*v256*/, 2
	v_add_nc_u32_e32 v2, s97, v251
	v_readlane_b32 s10, v0 /*v256*/, 4
	v_readlane_b32 s2, v0 /*v256*/, 0
	v_readlane_b32 s3, v0 /*v256*/, 1
	v_readlane_b32 s9, v0 /*v256*/, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_u32 v2, s10, v2, s6
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x100
	global_store_b32 v2, v253, s[2:3] scale_offset
.LBB0_22:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 8
	v_add3_u32 v2, v255, v148 /*v660*/, 16
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_i32_e64 v2, s99
	s_set_vgpr_msb 0x800
	s_cbranch_execz .LBB0_24
	s_set_vgpr_msb 1
	v_readlane_b32 s8, v0 /*v256*/, 2
	v_add_nc_u32_e32 v2, s97, v2
	v_readlane_b32 s10, v0 /*v256*/, 4
	v_readlane_b32 s2, v0 /*v256*/, 0
	v_readlane_b32 s3, v0 /*v256*/, 1
	v_readlane_b32 s9, v0 /*v256*/, 3
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_u32 v2, s10, v2, s6
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x100
	global_store_b32 v2, v0, s[2:3] scale_offset
.LBB0_24:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s1
	v_rcp_f32_e32 v28, v1
	s_set_vgpr_msb 9
	v_readlane_b32 s1, v0 /*v256*/, 7
	v_rcp_f32_e32 v60, v78 /*v334*/
	s_barrier_signal -1
	s_mov_b32 s6, 0
	s_mov_b32 s8, 1
	s_max_i32 s1, s1, 0
	s_mov_b32 s7, s6
	s_wait_alu depctr_vm_vsrc(0)
	v_sub_nc_u32_e32 v0, s1, v149 /*v661*/
	s_set_vgpr_msb 0x901
	v_pk_mul_f32 v[4:5], v[94:95] /*v[350:351]*/, v[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[96:97] /*v[352:353]*/, v[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[100:101] /*v[356:357]*/, v[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[90:91] /*v[346:347]*/, v[28:29] op_sel_hi:[1,0]
	v_med3_i32 v70, 0, v0, 32
	v_pk_mul_f32 v[0:1], v[92:93] /*v[348:349]*/, v[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[102:103] /*v[358:359]*/, v[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[104:105] /*v[360:361]*/, v[28:29] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v3, v2, v3
	v_cvt_pk_bf16_f32 v2, v4, v5
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[14:15], v[98:99] /*v[354:355]*/, v[28:29] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v5, v8, v9
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[8:9], v[112:113] /*v[368:369]*/, v[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[108:109] /*v[364:365]*/, v[28:29] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v1, v0, v1
	v_cvt_pk_bf16_f32 v0, v6, v7
	v_cvt_pk_bf16_f32 v7, v12, v13
	v_cvt_pk_bf16_f32 v6, v10, v11
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[12:13], v[110:111] /*v[366:367]*/, v[28:29] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v4, v14, v15
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[14:15], v[106:107] /*v[362:363]*/, v[28:29] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v11, v8, v9
	v_cvt_pk_bf16_f32 v9, v16, v17
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[16:17], v[28:29], v[118:119] /*v[374:375]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v10, v12, v13
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[12:13], v[28:29], v[120:121] /*v[376:377]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[28:29], v[116:117] /*v[372:373]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v8, v14, v15
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[20:21], v[28:29], v[114:115] /*v[370:371]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v14, v16, v17
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[16:17], v[28:29], v[128:129] /*v[384:385]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[28:29], v[126:127] /*v[382:383]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[28:29], v[124:125] /*v[380:381]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v13, v18, v19
	v_cvt_pk_bf16_f32 v12, v20, v21
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[20:21], v[28:29], v[122:123] /*v[378:379]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v19, v16, v17
	v_cvt_pk_bf16_f32 v18, v22, v23
	v_cvt_pk_bf16_f32 v17, v24, v25
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[22:23], v[28:29], v[136:137] /*v[392:393]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[28:29], v[130:131] /*v[386:387]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[28:29], v[142:143] /*v[398:399]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v16, v20, v21
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v24, v25
	v_cvt_pk_bf16_f32 v21, v26, v27
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[24:25], v[28:29], v[144:145] /*v[400:401]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[28:29], v[140:141] /*v[396:397]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v20, v30, v31
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[30:31], v[28:29], v[138:139] /*v[394:395]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v26, v32, v33
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[32:33], v[28:29], v[152:153] /*v[408:409]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v27, v24, v25
	v_cvt_pk_bf16_f32 v25, v34, v35
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[34:35], v[28:29], v[150:151] /*v[406:407]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[28:29], v[148:149] /*v[404:405]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v24, v30, v31
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[38:39], v[28:29], v[146:147] /*v[402:403]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v31, v32, v33
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[32:33], v[160:161] /*v[416:417]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[156:157] /*v[412:413]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v30, v34, v35
	v_cvt_pk_bf16_f32 v29, v36, v37
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[36:37], v[158:159] /*v[414:415]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v28, v38, v39
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[38:39], v[154:155] /*v[410:411]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v35, v32, v33
	v_cvt_pk_bf16_f32 v33, v40, v41
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[40:41], v[166:167] /*v[422:423]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v34, v36, v37
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[36:37], v[168:169] /*v[424:425]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[164:165] /*v[420:421]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v32, v38, v39
	v_cvt_pk_bf16_f32 v38, v40, v41
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[40:41], v[176:177] /*v[432:433]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[174:175] /*v[430:431]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[172:173] /*v[428:429]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[162:163] /*v[418:419]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v39, v36, v37
	v_cvt_pk_bf16_f32 v37, v42, v43
	v_cvt_pk_bf16_f32 v43, v40, v41
	v_cvt_pk_bf16_f32 v42, v46, v47
	v_cvt_pk_bf16_f32 v41, v48, v49
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[46:47], v[60:61], v[184:185] /*v[440:441]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[60:61], v[182:183] /*v[438:439]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v36, v44, v45
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[44:45], v[170:171] /*v[426:427]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[180:181] /*v[436:437]*/, v[60:61] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[178:179] /*v[434:435]*/, v[60:61] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v47, v46, v47
	v_cvt_pk_bf16_f32 v46, v48, v49
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[48:49], v[60:61], v[200:201] /*v[456:457]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[60:61], v[198:199] /*v[454:455]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[60:61], v[196:197] /*v[452:453]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x401
	v_readlane_b32 s1, v0 /*v256*/, 14
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v40, v44, v45
	v_cvt_pk_bf16_f32 v45, v50, v51
	v_cvt_pk_bf16_f32 v44, v52, v53
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[52:53], v[60:61], v[194:195] /*v[450:451]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v51, v48, v49
	v_cvt_pk_bf16_f32 v50, v54, v55
	v_cvt_pk_bf16_f32 v49, v56, v57
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[54:55], v[60:61], v[208:209] /*v[464:465]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[60:61], v[206:207] /*v[462:463]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[60:61], v[204:205] /*v[460:461]*/ op_sel_hi:[0,1]
	s_add_co_i32 s0, s0, s1
	s_set_vgpr_msb 0x401
	v_readlane_b32 s1, v0 /*v256*/, 8
	s_mul_i32 s0, s0, s76
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v48, v52, v53
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[60:61], v[202:203] /*v[458:459]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v55, v54, v55
	v_cvt_pk_bf16_f32 v54, v56, v57
	v_cvt_pk_bf16_f32 v53, v58, v59
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[56:57], v[60:61], v[188:189] /*v[444:445]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[60:61], v[192:193] /*v[448:449]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[60:61], v[190:191] /*v[446:447]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x408
	v_mul_lo_u32 v61, 0x2400, v75 /*v587*/
	s_add_co_i32 s0, s0, s1
	s_set_vgpr_msb 0x800
	v_cvt_pk_bf16_f32 v52, v62, v63
	s_lshl_b32 s0, s0, 1
	v_cvt_pk_bf16_f32 v59, v58, v59
	s_ashr_i32 s1, s0, 31
	v_cvt_pk_bf16_f32 v58, v64, v65
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[60:61], v[186:187] /*v[442:443]*/ op_sel_hi:[0,1]
	s_bitset1_b32 s1, 31
	s_set_vgpr_msb 0x408
	v_or_b32_e32 v61, v61, v73 /*v585*/
	s_set_vgpr_msb 0x800
	v_cvt_pk_bf16_f32 v57, v56, v57
	s_barrier_wait -1
	v_cvt_pk_bf16_f32 v56, v62, v63
	s_mov_b32 s4, 32
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[60:61], v[216:217] /*v[472:473]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[60:61], v[214:215] /*v[470:471]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x402
	v_lshl_add_u32 v71, v148 /*v660*/, 8, v61
	s_set_vgpr_msb 0x204
	v_pk_mul_f32 v[66:67], v[60:61], v[212:213] /*v[468:469]*/ op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[60:61], v[210:211] /*v[466:467]*/ op_sel_hi:[0,1]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v63, v62, v63
	v_cvt_pk_bf16_f32 v62, v64, v65
	v_or_b32_e32 v64, 0x20000, v71
	v_or_b32_e32 v65, 0x20020, v71
	v_cvt_pk_bf16_f32 v61, v66, v67
	v_or_b32_e32 v66, 0x20040, v71
	v_or_b32_e32 v67, 0x20060, v71
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v64, v[0:3]
	ds_store_b128 v65, v[4:7]
	ds_store_b128 v66, v[8:11]
	ds_store_b128 v67, v[12:15]
	s_wait_alu depctr_vm_vsrc(3)
	v_or_b32_e32 v0, 0x20080, v71
	v_or_b32_e32 v1, 0x200a0, v71
	v_or_b32_e32 v2, 0x200c0, v71
	v_or_b32_e32 v3, 0x200e0, v71
	s_wait_alu depctr_vm_vsrc(2)
	v_dual_mov_b32 v5, s76 :: v_dual_add_nc_u32 v4, 0x21000, v71
	s_wait_alu depctr_va_vdst(4)
	ds_store_b128 v0, v[16:19]
	s_wait_alu depctr_va_vdst(3)
	ds_store_b128 v1, v[20:23]
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v2, v[24:27]
	s_wait_alu depctr_va_vdst(1)
	ds_store_b128 v3, v[28:31]
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v4, v[32:35]
	s_wait_alu depctr_vm_vsrc(4)
	v_add_nc_u32_e32 v0, 0x21020, v71
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v1, 0x21040, v71
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v2, 0x21060, v71
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v3, 0x21080, v71
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v4, 0x210a0, v71
	s_wait_alu depctr_va_vdst(4)
	ds_store_b128 v0, v[36:39]
	s_wait_alu depctr_va_vdst(3)
	ds_store_b128 v1, v[40:43]
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v2, v[44:47]
	s_wait_alu depctr_va_vdst(1)
	ds_store_b128 v3, v[48:51]
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v4, v[52:55]
	s_wait_alu depctr_vm_vsrc(2)
	v_mul_lo_u32 v2, 0x2400, v250
	v_cvt_pk_bf16_f32 v60, v68, v69
	v_add_nc_u32_e32 v0, 0x210c0, v71
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v4, s0 :: v_dual_add_nc_u32 v1, 0x210e0, v71
	v_mov_b32_e32 v3, s1
	v_readfirstlane_b32 s5, v5
	s_wait_alu depctr_va_vdst(3)
	ds_store_b128 v0, v[56:59]
	s_wait_alu depctr_va_vdst(2)
	ds_store_b128 v1, v[60:63]
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0x20000, v2
	v_lshlrev_b32_e32 v2, 16, v70
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v3
	s_mov_b32 s1, 0x800000
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s2, v2
	s_mov_b32 s0, 0x10000
	s_mov_b32 s3, s1
	;;#ASMSTART
	s_wait_dscnt 0x0
	;;#ASMEND
	tensor_store_from_lds s[8:11], s[0:7]
	s_wait_tensorcnt 0x0
.LBB0_25:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fmha_fwd_kernel_0
		.amdhsa_group_segment_fixed_size 233472
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 520
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 796
		.amdhsa_next_free_sgpr 105
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 255
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
	.size	fmha_fwd_kernel_0, .Lfunc_end0-fmha_fwd_kernel_0

	.set fmha_fwd_kernel_0.num_vgpr, 796
	.set fmha_fwd_kernel_0.num_agpr, 0
	.set fmha_fwd_kernel_0.numbered_sgpr, 105
	.set fmha_fwd_kernel_0.num_named_barrier, 0
	.set fmha_fwd_kernel_0.private_seg_size, 0
	.set fmha_fwd_kernel_0.uses_vcc, 1
	.set fmha_fwd_kernel_0.uses_flat_scratch, 0
	.set fmha_fwd_kernel_0.has_dyn_sized_stack, 0
	.set fmha_fwd_kernel_0.has_recursion, 0
	.set fmha_fwd_kernel_0.has_indirect_call, 0
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
      - .offset:         48
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     global_buffer
      - .offset:         168
        .size:           16
        .value_kind:     by_value
      - .address_space:  global
        .offset:         184
        .size:           8
        .value_kind:     global_buffer
      - .offset:         192
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         200
        .size:           8
        .value_kind:     global_buffer
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
      - .offset:         224
        .size:           4
        .value_kind:     by_value
      - .offset:         228
        .size:           4
        .value_kind:     by_value
      - .offset:         232
        .size:           4
        .value_kind:     by_value
      - .offset:         236
        .size:           4
        .value_kind:     by_value
      - .offset:         240
        .size:           4
        .value_kind:     by_value
      - .offset:         244
        .size:           4
        .value_kind:     by_value
      - .offset:         248
        .size:           4
        .value_kind:     by_value
      - .offset:         252
        .size:           4
        .value_kind:     by_value
      - .offset:         256
        .size:           4
        .value_kind:     by_value
      - .offset:         264
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         268
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         272
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         276
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         278
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         280
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         282
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         284
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         286
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         304
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         312
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         320
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         328
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 233472
    .kernarg_segment_align: 8
    .kernarg_segment_size: 520
    .max_flat_workgroup_size: 128
    .name:           fmha_fwd_kernel_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     107
    .sgpr_spill_count: 43
    .symbol:         fmha_fwd_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     796
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
