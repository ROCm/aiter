	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1
	.p2align	8
	.type	gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1,@function
gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x70 nv
	s_load_b64 s[34:35], s[0:1], 0xa4 nv
	s_mov_b32 s44, 1
	v_mov_b32_e32 v1, 0
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
	s_wait_kmcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v1, s[4:5] offset:768
	s_add_co_i32 s2, s35, 0x1ff
	s_movk_i32 s21, 0xc0
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s49, 0
	s_lshr_b32 s3, s3, 23
	v_lshl_add_u32 v6, v0, 4, 0
	s_add_co_i32 s3, s2, s3
	s_mov_b32 s50, s49
	s_and_b32 s8, s3, 0xfffffe00
	s_ashr_i32 s3, s3, 9
	s_cmp_lg_u32 s2, s8
	s_mov_b32 s51, s49
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s2, 0
	v_add_nc_u32_e32 v7, 0x10000, v6
	s_cselect_b32 s2, -1, 0
	v_add_nc_u32_e32 v9, 0x11000, v6
	s_and_b32 s2, s2, s8
	s_sub_co_ci_u32 s2, s3, 0
	s_add_co_i32 s3, s34, 15
	v_add_nc_u32_e32 v11, 0x12000, v6
	s_ashr_i32 s8, s3, 31
	v_add_nc_u32_e32 v8, 0x10800, v6
	s_lshr_b32 s8, s8, 28
	v_add_nc_u32_e32 v10, 0x11800, v6
	s_add_co_i32 s8, s3, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s9, s8, -16
	s_ashr_i32 s8, s8, 4
	s_cmp_lg_u32 s3, s9
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s3, 0
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_cselect_b32 s10, -1, 0
	s_bfe_u32 s11, ttmp6, 0x4000c
	s_and_b32 s13, ttmp6, 15
	s_add_co_i32 s11, s11, 1
	s_lshl_b32 s12, s2, 4
	s_mul_i32 s11, ttmp9, s11
	s_and_b32 s9, s10, s9
	s_add_co_i32 s13, s13, s11
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s3, ttmp9, s13
	s_abs_i32 s10, s12
	s_abs_i32 s14, s3
	s_cvt_f32_u32 s11, s10
	s_sub_co_i32 s13, 0, s10
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v3, s11
	v_nop
	v_readfirstlane_b32 s11, v3
	s_mul_f32 s11, s11, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s11, s11
	s_mul_i32 s13, s13, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s13, s11, s13
	s_add_co_i32 s11, s11, s13
	s_xor_b32 s13, s3, s12
	s_mul_hi_u32 s11, s14, s11
	s_ashr_i32 s13, s13, 31
	s_mul_i32 s15, s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s14, s14, s15
	s_add_co_i32 s15, s11, 1
	s_sub_co_i32 s16, s14, s10
	s_cmp_ge_u32 s14, s10
	s_cselect_b32 s11, s15, s11
	s_cselect_b32 s14, s16, s14
	s_add_co_i32 s15, s11, 1
	s_cmp_ge_u32 s14, s10
	s_cselect_b32 s10, s15, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s10, s10, s13
	s_sub_co_i32 s11, s10, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s11, s11, s12
	s_cmp_lg_u32 s3, s11
	s_cselect_b32 s11, -1, 0
	s_xor_b32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s11
	s_sub_co_ci_u32 s2, s10, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s10, s2, s12
	s_lshl_b32 s11, s2, 4
	s_sub_co_i32 s3, s3, s10
	s_cmp_lg_u32 s9, 0
	s_sub_co_ci_u32 s2, s8, s11
	s_abs_i32 s13, s3
	s_min_i32 s8, s2, 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_abs_i32 s9, s8
	s_cvt_f32_u32 s10, s9
	s_sub_co_i32 s12, 0, s9
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v3, s10
	v_nop
	v_readfirstlane_b32 s10, v3
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s10, s10
	s_mul_i32 s12, s12, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s12, s10, s12
	s_add_co_i32 s10, s10, s12
	s_xor_b32 s12, s3, s8
	s_mul_hi_u32 s10, s13, s10
	s_ashr_i32 s12, s12, 31
	s_mul_i32 s14, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s13, s13, s14
	s_add_co_i32 s14, s10, 1
	s_sub_co_i32 s15, s13, s9
	s_cmp_ge_u32 s13, s9
	s_cselect_b32 s10, s14, s10
	s_cselect_b32 s13, s15, s13
	s_add_co_i32 s14, s10, 1
	s_cmp_ge_u32 s13, s9
	s_cselect_b32 s9, s14, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s9, s9, s12
	s_sub_co_i32 s10, s9, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s10, s8, s10
	s_cmp_lg_u32 s3, s10
	s_cselect_b32 s10, -1, 0
	s_xor_b32 s2, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s10
	s_sub_co_ci_u32 s2, s9, s12
	s_add_co_i32 s3, s3, s11
	s_mul_i32 s8, s2, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s3, s3, s8
	s_lshl_b32 s68, s3, 4
	s_movk_i32 s3, 0x60
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v2
	s_cmp_gt_i32 s9, s68
	s_cselect_b32 s3, s3, 0x120
	s_cselect_b32 s9, 0, 0xc1
	v_mov_b32_e32 v2, s3
	s_cselect_b32 s10, s21, 0x180
	s_or_b32 s11, s3, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v2, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s8, v2
	s_cmp_gt_i32 s8, s68
	s_cselect_b32 s8, s9, s11
	s_cselect_b32 s3, s3, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s8, s3
	s_lshr_b32 s9, s9, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v2, s9
	s_or_b32 s11, s9, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v2, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s10, v2
	s_cmp_gt_i32 s10, s68
	s_cselect_b32 s8, s8, s11
	s_cselect_b32 s3, s9, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s8, s3
	s_lshr_b32 s10, s9, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v2, s10
	s_add_co_i32 s11, s10, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v2, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v2
	s_cmp_gt_i32 s9, s68
	s_mov_b32 s9, s49
	s_cselect_b32 s8, s8, s11
	s_cselect_b32 s48, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[10:11], s[8:9], s[48:49]
	s_lshr_b64 s[10:11], s[10:11], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[12:13], s[10:11], 2
	s_add_co_i32 s9, s10, 1
	s_add_nc_u64 s[12:13], s[4:5], s[12:13]
	global_load_b32 v1, v1, s[12:13]
	s_wait_xcnt 0x0
	s_mov_b32 s12, 4
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s3, v1
	s_cmp_gt_i32 s3, s68
	s_cselect_b32 s3, s8, s9
	s_cselect_b32 s8, s10, s48
	s_mov_b32 s48, s49
	s_add_co_i32 s9, s3, s8
	s_wait_alu depctr_vm_vsrc(1)
	v_mov_b64_e32 v[2:3], s[48:49]
	s_lshr_b32 s9, s9, 1
	v_mov_b64_e32 v[4:5], s[50:51]
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s9
	s_add_co_i32 s11, s9, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s10, v1
	s_cmp_gt_i32 s10, s68
	s_cselect_b32 s3, s3, s11
	s_cselect_b32 s8, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s3, s8
	s_lshr_b32 s9, s9, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s9
	s_add_co_i32 s11, s9, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s10, v1
	s_cmp_gt_i32 s10, s68
	s_cselect_b32 s3, s3, s11
	s_cselect_b32 s8, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s3, s8
	s_lshr_b32 s9, s9, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s9
	s_add_co_i32 s11, s9, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s10, v1
	s_cmp_gt_i32 s10, s68
	s_cselect_b32 s3, s3, s11
	s_cselect_b32 s8, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s3, s8
	s_lshr_b32 s9, s9, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s10, s9, 0x17f
	s_add_co_i32 s11, s9, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s10
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s10, v1
	s_cmp_gt_i32 s10, s68
	s_cselect_b32 s3, s3, s11
	s_cselect_b32 s8, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s8, s3, s8
	s_lshr_b32 s8, s8, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s9, s8, 0x17f
	s_add_co_i32 s8, s8, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s9
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[4:5] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s9, v1
	s_cmp_gt_i32 s9, s68
	s_cselect_b32 s74, s3, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_u32 s74, 0x180
	s_cselect_b32 s3, s74, 0x17f
	s_cmp_gt_u32 s74, 0x17f
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s3
	v_readfirstlane_b32 s3, v0
	s_wait_alu depctr_va_vdst(1)
	global_load_b32 v1, v1, s[4:5] scale_offset
	ds_store_b128 v6, v[2:5]
	ds_store_b128 v6, v[2:5] offset:2048
	ds_store_b128 v6, v[2:5] offset:4096
	ds_store_b128 v6, v[2:5] offset:6144
	ds_store_b128 v6, v[2:5] offset:8192
	ds_store_b128 v6, v[2:5] offset:10240
	ds_store_b128 v6, v[2:5] offset:12288
	ds_store_b128 v6, v[2:5] offset:14336
	ds_store_b128 v6, v[2:5] offset:16384
	ds_store_b128 v6, v[2:5] offset:18432
	ds_store_b128 v6, v[2:5] offset:20480
	ds_store_b128 v6, v[2:5] offset:22528
	ds_store_b128 v6, v[2:5] offset:24576
	ds_store_b128 v6, v[2:5] offset:26624
	ds_store_b128 v6, v[2:5] offset:28672
	ds_store_b128 v6, v[2:5] offset:30720
	ds_store_b128 v6, v[2:5] offset:32768
	ds_store_b128 v6, v[2:5] offset:34816
	ds_store_b128 v6, v[2:5] offset:36864
	ds_store_b128 v6, v[2:5] offset:38912
	ds_store_b128 v6, v[2:5] offset:40960
	ds_store_b128 v6, v[2:5] offset:43008
	ds_store_b128 v6, v[2:5] offset:45056
	ds_store_b128 v6, v[2:5] offset:47104
	ds_store_b128 v6, v[2:5] offset:49152
	ds_store_b128 v6, v[2:5] offset:51200
	ds_store_b128 v6, v[2:5] offset:53248
	ds_store_b128 v6, v[2:5] offset:55296
	ds_store_b128 v6, v[2:5] offset:57344
	ds_store_b128 v6, v[2:5] offset:59392
	ds_store_b128 v6, v[2:5] offset:61440
	ds_store_b128 v6, v[2:5] offset:63488
	ds_store_b128 v7, v[2:5]
	ds_store_b128 v8, v[2:5]
	ds_store_b128 v9, v[2:5]
	ds_store_b128 v10, v[2:5]
	ds_store_b128 v11, v[2:5]
	s_wait_xcnt 0x0
	s_mov_b32 s4, 0x10000
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s8, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s51, s35, 31
	s_mov_b32 s50, s35
	s_lshl_b32 s5, s3, 2
	s_add_nc_u64 s[10:11], s[50:51], 31
	s_lshr_b32 s60, s3, 3
	s_lshr_b32 s48, s11, 27
	s_lshl_b32 s70, s2, 9
	s_add_nc_u64 s[2:3], s[10:11], s[48:49]
	s_mov_b64 s[14:15], 0xffffffffffffffe0
	s_and_b32 s5, s5, 0x180
	s_and_b64 s[14:15], s[2:3], s[14:15]
	s_ashr_i32 s71, s70, 31
	s_ashr_i32 s69, s68, 31
	s_lshr_b64 s[2:3], s[2:3], 5
	s_cmp_lg_u64 s[10:11], s[14:15]
	s_load_b128 s[16:19], s[0:1], 0x28 nv
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s35, 0xffffffe1
	s_mov_b32 s75, s49
	s_cselect_b32 s13, -1, 0
	s_lshr_b32 s10, s51, 28
	s_and_b32 s20, s13, s9
	s_add_co_i32 s10, s35, s10
	s_lshr_b64 s[22:23], s[70:71], 5
	s_ashr_i32 s10, s10, 4
	s_mov_b32 s37, s49
	s_ashr_i32 s11, s10, 31
	s_movk_i32 s29, 0x1800
	s_lshl_b64 s[14:15], s[10:11], 4
	s_mov_b32 s28, 8
	s_cmp_lg_u64 s[14:15], s[50:51]
	s_mul_u64 s[14:15], s[68:69], 0x300
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s35, 0
	s_mov_b32 s30, s49
	s_cselect_b32 s13, -1, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[14:15], s[16:17], s[14:15]
	s_and_b32 s9, s13, s9
	s_movk_i32 s13, 0x300
	s_wait_alu depctr_vm_vsrc(0)
	v_cndmask_b32_e64 v1, 0, 1, s9
	s_bfe_u32 s9, ttmp8, 0x50019
	s_lshr_b64 s[38:39], s[70:71], 4
	s_and_b32 s33, s9, 3
	s_add_co_i32 s62, 0, 0x900
	v_readfirstlane_b32 s48, v1
	s_lshl_b32 s9, s33, 2
	s_mul_i32 s36, s33, 0xc000
	s_or_b32 s9, s9, s68
	s_lshl_b32 s61, s33, 13
	s_sub_nc_u64 s[16:17], s[10:11], s[48:49]
	s_mul_i32 s48, s33, 0xc00
	s_mul_i32 s10, s33, 0x240
	s_sub_co_i32 s8, s8, s9
	s_add_nc_u64 s[46:47], s[48:49], s[14:15]
	s_add_co_i32 s45, s10, 0
	s_max_i32 s8, s8, 0
	s_or_b32 s10, s47, 0x80000000
	s_mov_b64 s[24:25], s[44:45]
	s_lshl_b32 s9, s8, 16
	s_lshr_b32 s34, s8, 16
	s_mov_b64 s[26:27], s[46:47]
	s_mov_b32 s27, s10
	s_or_b32 s10, s9, 0x7fff
	s_or_b32 s11, s34, 0x800000
	s_mov_b32 s9, 0xffff0000
	s_mov_b32 s8, 0x7100000
	s_mov_b32 s14, s49
	s_mov_b32 s15, s49
	s_mul_u64 s[16:17], s[16:17], s[74:75]
	tensor_load_to_lds s[24:27], s[8:15]
	s_add_nc_u64 s[16:17], s[16:17], s[38:39]
	s_lshl_b32 s26, s33, 19
	s_mul_u64 s[16:17], s[16:17], 0x1800
	s_add_co_i32 s61, s61, s62
	s_add_nc_u64 s[16:17], s[18:19], s[16:17]
	s_mov_b64 s[54:55], s[46:47]
	s_add_nc_u64 s[76:77], s[16:17], s[36:37]
	s_mov_b64 s[52:53], s[44:45]
	s_mov_b32 s27, 0x4007fff
	s_mov_b32 s24, s49
	s_mov_b32 s31, s49
	s_mov_b32 s25, s9
	s_xor_b32 s26, s26, 0xffff7fff
	s_mov_b32 s53, s61
	s_or_b32 s55, s77, 0x80000000
	s_mov_b32 s54, s76
	s_load_b64 s[16:17], s[0:1], 0x38 nv
	s_mov_b32 s57, s49
	s_mul_i32 s56, s33, 0x60
	s_lshl_b32 s38, s33, 18
	s_mov_b32 s41, 6
	s_mov_b32 s39, 0x17fff
	s_mov_b32 s36, 0x20000
	s_mov_b32 s40, s12
	s_mov_b32 s42, s49
	s_mov_b32 s43, s49
	s_mov_b32 s37, s9
	s_xor_b32 s38, s38, 0xffff7fff
	v_mov_b32_e32 v3, s49
	v_cndmask_b32_e64 v2, 0, 1, s20
	s_load_b64 s[18:19], s[0:1], 0x60 nv
	s_mov_b64 s[66:67], s[46:47]
	s_mov_b64 s[64:65], s[44:45]
	s_mov_b32 s20, s12
	v_sub_nc_u64_e32 v[2:3], s[2:3], v[2:3]
	s_mul_u64 s[2:3], s[22:23], 0x300
	s_mov_b32 s22, s49
	s_mov_b32 s23, s49
	v_and_b32_e32 v1, 15, v0
	v_bfe_u32 v160, v0, 4, 1
	v_mul_u64_e32 v[2:3], s[74:75], v[2:3]
	s_load_b64 s[72:73], s[0:1], 0x0 nv
	s_wait_xcnt 0x0
	s_lshl_b32 s0, s5, 6
	v_and_or_b32 v161, 0x1ffffff0, s60, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v5, v161, v160
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]
	s_mov_b32 s19, 0x207fff
	s_mov_b32 s18, s38
	v_mul_u64_e32 v[2:3], 0x300, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u64_e32 v[2:3], s[2:3], v[2:3]
	s_add_co_i32 s2, 0, 0x8940
	v_add_nc_u64_e32 v[154:155], s[48:49], v[2:3]
	s_lshl_b32 s48, s33, 9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s48, s48, s2
	s_mov_b32 s65, s48
	v_mov_b64_e32 v[146:147], s[66:67]
	v_mov_b64_e32 v[144:145], s[64:65]
	v_or_b32_e32 v3, 0x80000000, v155
	v_mov_b32_e32 v2, v154
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s64, v144
	v_readfirstlane_b32 s65, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_readfirstlane_b32 s66, v2
	v_readfirstlane_b32 s67, v3
	v_or_b32_e32 v2, s5, v0
	v_or_b32_e32 v3, s5, v1
	v_dual_lshlrev_b32 v1, 4, v1 :: v_dual_lshlrev_b32 v0, 2, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_lshlrev_b32 v2, 2, v3 :: v_dual_lshlrev_b32 v3, 8, v160
	v_or_b32_e32 v4, 0x89c0, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_nc_u32_e32 v162, 0, v2
	v_or3_b32 v1, s0, v3, v1
	v_or_b32_e32 v0, 0x1c0, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_nc_u32 v165, s2, v2 :: v_dual_add_nc_u32 v163, 0, v4
	v_add_nc_u32_e32 v164, 0x8800, v162
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_nc_u32 v166, s62, v1 :: v_dual_add_nc_u32 v167, s2, v0
	v_mul_lo_u32 v4, 0x90, v161
	v_add_nc_u32_e32 v153, 0x8940, v162
	v_add_nc_u32_e32 v169, 0x8a00, v162
	tensor_load_to_lds s[52:55], s[24:31]
	s_mul_u64 s[54:55], s[68:69], 24
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[16:17], s[16:17], s[54:55]
	s_add_co_i32 s55, 0, 0x8900
	s_lshl_b32 s54, s33, 4
	s_add_nc_u64 s[78:79], s[56:57], s[16:17]
	s_or_b32 s16, s54, s55
	s_or_b32 s17, s79, 0x80000000
	s_mov_b64 s[58:59], s[46:47]
	s_mov_b64 s[56:57], s[44:45]
	s_mov_b32 s57, s16
	s_mov_b32 s58, s78
	s_mov_b32 s59, s17
	s_mov_b32 s16, s36
	s_mov_b32 s17, s9
	v_lshl_add_u32 v168, v5, 2, s55
	v_lshlrev_b32_e32 v5, 4, v160
	tensor_load_to_lds s[56:59], s[36:43]
	tensor_load_to_lds s[64:67], s[16:23]
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[16:19], v166
	ds_load_b128 v[20:23], v166 offset:512
	ds_load_b128 v[24:27], v166 offset:1024
	ds_load_b128 v[28:31], v166 offset:1536
	ds_load_b128 v[32:35], v166 offset:2048
	ds_load_b128 v[36:39], v166 offset:2560
	ds_load_b128 v[40:43], v166 offset:3072
	ds_load_b128 v[44:47], v166 offset:3584
	ds_load_b128 v[80:83], v166 offset:4096
	ds_load_b128 v[84:87], v166 offset:4608
	ds_load_b128 v[88:91], v166 offset:5120
	ds_load_b128 v[92:95], v166 offset:5632
	ds_load_b128 v[96:99], v166 offset:6144
	ds_load_b128 v[100:103], v166 offset:6656
	ds_load_b128 v[104:107], v166 offset:7168
	ds_load_b128 v[108:111], v166 offset:7680
	ds_load_2addr_b32 v[116:117], v164 offset0:80 offset1:128
	ds_load_b32 v48, v165 offset:64
	ds_load_b32 v49, v163
	ds_load_2addr_b32 v[118:119], v164 offset0:144 offset1:160
	ds_load_b32 v112, v162 offset:35520
	ds_load_b32 v113, v167
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v114, v168
	s_add_nc_u64 s[58:59], s[46:47], 0x80
	s_add_co_i32 s55, s45, 0x9200
	s_bitset1_b32 s59, 31
	s_mov_b64 s[0:1], s[44:45]
	s_mov_b64 s[2:3], s[46:47]
	s_mov_b32 s1, s55
	s_mov_b32 s2, s58
	s_mov_b32 s3, s59
	s_add_co_i32 s55, s61, 0x9200
	tensor_load_to_lds s[0:3], s[8:15]
	s_add_nc_u64 s[2:3], s[76:77], 0x400
	s_mov_b64 s[62:63], s[46:47]
	s_bitset1_b32 s3, 31
	s_mov_b64 s[60:61], s[44:45]
	s_mov_b32 s61, s55
	s_mov_b32 s62, s2
	s_mov_b32 s63, s3
	s_add_co_i32 s54, s54, 0
	s_add_nc_u64 s[2:3], s[78:79], 4
	s_add_co_i32 s54, s54, 0x11b00
	s_bitset1_b32 s3, 31
	s_mov_b64 s[66:67], s[46:47]
	s_mov_b64 s[64:65], s[44:45]
	s_mov_b32 s65, s54
	s_mov_b32 s66, s2
	s_mov_b32 s67, s3
	v_add_nc_u64_e32 v[0:1], 0x80, v[154:155]
	s_add_co_i32 s48, s48, 0x9200
	s_mov_b64 s[82:83], s[46:47]
	s_mov_b64 s[80:81], s[44:45]
	s_mov_b32 s81, s48
	v_mov_b64_e32 v[148:149], s[82:83]
	v_mov_b64_e32 v[146:147], s[80:81]
	v_or_b32_e32 v3, 0x80000000, v1
	v_mov_b32_e32 v2, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s80, v146
	v_readfirstlane_b32 s81, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s82, v2
	v_readfirstlane_b32 s83, v3
	tensor_load_to_lds s[60:63], s[24:31]
	tensor_load_to_lds s[64:67], s[36:43]
	tensor_load_to_lds s[80:83], s[16:23]
	v_add3_u32 v170, v4, v5, 0
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v170
	ds_load_b128 v[4:7], v170 offset:32
	ds_load_b128 v[8:11], v170 offset:64
	ds_load_b128 v[12:15], v170 offset:96
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[16:23], v[0:15], 0, v116, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[24:31], v[0:15], 0, v48, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63], v[32:39], v[0:15], 0, v49, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55], v[40:47], v[0:15], 0, v117, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[80:87], v[0:15], 0, v118, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[88:95], v[0:15], 0, v119, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[96:103], v[0:15], 0, v112, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[104:111], v[0:15], 0, v113, v114 matrix_a_fmt:MATRIX_FMT_FP4
	v_add_nc_u32_e32 v171, 0x9000, v169
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[80:83], v166 offset:37376
	ds_load_b128 v[84:87], v166 offset:37888
	ds_load_b128 v[88:91], v166 offset:38400
	ds_load_b128 v[92:95], v166 offset:38912
	ds_load_b128 v[96:99], v166 offset:39424
	ds_load_b128 v[100:103], v166 offset:39936
	ds_load_b128 v[104:107], v166 offset:40448
	ds_load_b128 v[108:111], v166 offset:40960
	ds_load_b128 v[112:115], v166 offset:41472
	ds_load_b128 v[116:119], v166 offset:41984
	ds_load_b128 v[120:123], v166 offset:42496
	ds_load_b128 v[124:127], v166 offset:43008
	ds_load_b128 v[128:131], v166 offset:43520
	ds_load_b128 v[132:135], v166 offset:44032
	ds_load_b128 v[136:139], v166 offset:44544
	ds_load_b128 v[140:143], v166 offset:45056
	ds_load_b32 v148, v153 offset:37376
	ds_load_b32 v149, v165 offset:37440
	ds_load_b32 v150, v163 offset:37376
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[156:157], v171 offset0:128 offset1:144
	ds_load_2addr_b32 v[158:159], v171 offset0:160 offset1:176
	ds_load_b32 v151, v167 offset:37376
	ds_load_b32 v152, v168 offset:37376
	s_add_nc_u64 s[2:3], s[46:47], 0x100
	s_mov_b64 s[82:83], s[46:47]
	s_bitset1_b32 s3, 31
	s_mov_b64 s[80:81], s[44:45]
	s_mov_b32 s82, s2
	s_mov_b32 s83, s3
	s_add_nc_u64 s[54:55], s[76:77], 0x800
	tensor_load_to_lds s[80:83], s[8:15]
	s_bitset1_b32 s55, 31
	s_add_nc_u64 s[58:59], s[78:79], 8
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u64_e32 v[0:1], 0x100, v[154:155]
	s_bitset1_b32 s59, 31
	v_readfirstlane_b32 s80, v144
	v_readfirstlane_b32 s81, v145
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_or_b32_e32 v1, 0x80000000, v1
	v_readfirstlane_b32 s82, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s83, v1
	tensor_load_to_lds s[52:55], s[24:31]
	tensor_load_to_lds s[56:59], s[36:43]
	tensor_load_to_lds s[80:83], s[16:23]
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[0:3], v170 offset:37376
	ds_load_b128 v[4:7], v170 offset:37408
	ds_load_b128 v[8:11], v170 offset:37440
	ds_load_b128 v[12:15], v170 offset:37472
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[80:87], v[0:15], v[72:79], v148, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[88:95], v[0:15], v[64:71], v149, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63], v[96:103], v[0:15], v[56:63], v150, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55], v[104:111], v[0:15], v[48:55], v156, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[112:119], v[0:15], v[40:47], v157, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[120:127], v[0:15], v[32:39], v158, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[128:135], v[0:15], v[16:23], v159, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[136:143], v[0:15], v[24:31], v151, v152 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[80:83], v166
	ds_load_b128 v[84:87], v166 offset:512
	ds_load_b128 v[88:91], v166 offset:1024
	ds_load_b128 v[92:95], v166 offset:1536
	ds_load_b128 v[96:99], v166 offset:2048
	ds_load_b128 v[100:103], v166 offset:2560
	ds_load_b128 v[104:107], v166 offset:3072
	ds_load_b128 v[108:111], v166 offset:3584
	ds_load_b128 v[112:115], v166 offset:4096
	ds_load_b128 v[116:119], v166 offset:4608
	ds_load_b128 v[120:123], v166 offset:5120
	ds_load_b128 v[124:127], v166 offset:5632
	ds_load_b128 v[128:131], v166 offset:6144
	ds_load_b128 v[132:135], v166 offset:6656
	ds_load_b128 v[136:139], v166 offset:7168
	ds_load_b128 v[140:143], v166 offset:7680
	ds_load_2addr_b32 v[156:157], v164 offset0:80 offset1:128
	ds_load_b32 v148, v165 offset:64
	ds_load_b32 v149, v163
	ds_load_2addr_b32 v[158:159], v164 offset0:144 offset1:160
	ds_load_b32 v150, v162 offset:35520
	ds_load_b32 v151, v167
	ds_load_b32 v152, v168
	s_add_nc_u64 s[2:3], s[46:47], 0x180
	s_add_nc_u64 s[62:63], s[76:77], 0xc00
	s_bitset1_b32 s3, 31
	s_bitset1_b32 s63, 31
	tensor_load_to_lds s[0:3], s[8:15]
	s_add_nc_u64 s[66:67], s[78:79], 12
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u64_e32 v[0:1], 0x180, v[154:155]
	s_bitset1_b32 s67, 31
	v_readfirstlane_b32 s80, v146
	v_readfirstlane_b32 s81, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_or_b32_e32 v1, 0x80000000, v1
	v_readfirstlane_b32 s82, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s83, v1
	tensor_load_to_lds s[60:63], s[24:31]
	tensor_load_to_lds s[64:67], s[36:43]
	tensor_load_to_lds s[80:83], s[16:23]
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v170
	ds_load_b128 v[4:7], v170 offset:32
	ds_load_b128 v[8:11], v170 offset:64
	ds_load_b128 v[12:15], v170 offset:96
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[80:87], v[0:15], v[72:79], v156, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[88:95], v[0:15], v[64:71], v148, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63], v[96:103], v[0:15], v[56:63], v149, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55], v[104:111], v[0:15], v[48:55], v157, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[112:119], v[0:15], v[40:47], v158, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[120:127], v[0:15], v[32:39], v159, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[128:135], v[0:15], v[16:23], v150, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[136:143], v[0:15], v[24:31], v151, v152 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[80:83], v166 offset:37376
	ds_load_b128 v[84:87], v166 offset:37888
	ds_load_b128 v[88:91], v166 offset:38400
	ds_load_b128 v[92:95], v166 offset:38912
	ds_load_b128 v[96:99], v166 offset:39424
	ds_load_b128 v[100:103], v166 offset:39936
	ds_load_b128 v[104:107], v166 offset:40448
	ds_load_b128 v[108:111], v166 offset:40960
	ds_load_b128 v[112:115], v166 offset:41472
	ds_load_b128 v[116:119], v166 offset:41984
	ds_load_b128 v[120:123], v166 offset:42496
	ds_load_b128 v[124:127], v166 offset:43008
	ds_load_b128 v[128:131], v166 offset:43520
	ds_load_b128 v[132:135], v166 offset:44032
	ds_load_b128 v[136:139], v166 offset:44544
	ds_load_b128 v[140:143], v166 offset:45056
	ds_load_b32 v148, v153 offset:37376
	ds_load_b32 v149, v165 offset:37440
	ds_load_b32 v150, v163 offset:37376
	ds_load_2addr_b32 v[156:157], v171 offset0:128 offset1:144
	ds_load_2addr_b32 v[158:159], v171 offset0:160 offset1:176
	ds_load_b32 v151, v167 offset:37376
	ds_load_b32 v152, v168 offset:37376
	s_add_nc_u64 s[2:3], s[46:47], 0x200
	s_mov_b64 s[82:83], s[46:47]
	s_bitset1_b32 s3, 31
	s_mov_b64 s[80:81], s[44:45]
	s_mov_b32 s82, s2
	s_mov_b32 s83, s3
	s_add_nc_u64 s[54:55], s[76:77], 0x1000
	tensor_load_to_lds s[80:83], s[8:15]
	s_bitset1_b32 s55, 31
	s_add_nc_u64 s[58:59], s[78:79], 16
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u64_e32 v[0:1], 0x200, v[154:155]
	s_bitset1_b32 s59, 31
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v1, 0x80000000, v1
	tensor_load_to_lds s[52:55], s[24:31]
	v_readfirstlane_b32 s52, v144
	v_readfirstlane_b32 s53, v145
	v_readfirstlane_b32 s54, v0
	v_readfirstlane_b32 s55, v1
	tensor_load_to_lds s[56:59], s[36:43]
	s_delay_alu instid0(VALU_DEP_2)
	tensor_load_to_lds s[52:55], s[16:23]
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v170 offset:37376
	ds_load_b128 v[4:7], v170 offset:37408
	ds_load_b128 v[8:11], v170 offset:37440
	ds_load_b128 v[12:15], v170 offset:37472
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[80:87], v[0:15], v[72:79], v148, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[88:95], v[0:15], v[64:71], v149, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63], v[96:103], v[0:15], v[56:63], v150, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55], v[104:111], v[0:15], v[48:55], v156, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[112:119], v[0:15], v[40:47], v157, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[120:127], v[0:15], v[32:39], v158, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[128:135], v[0:15], v[16:23], v159, v152 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[136:143], v[0:15], v[24:31], v151, v152 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[80:83], v166
	ds_load_b128 v[84:87], v166 offset:512
	ds_load_b128 v[88:91], v166 offset:1024
	ds_load_b128 v[92:95], v166 offset:1536
	ds_load_b128 v[96:99], v166 offset:2048
	ds_load_b128 v[100:103], v166 offset:2560
	ds_load_b128 v[104:107], v166 offset:3072
	ds_load_b128 v[108:111], v166 offset:3584
	ds_load_b128 v[112:115], v166 offset:4096
	ds_load_b128 v[116:119], v166 offset:4608
	ds_load_b128 v[120:123], v166 offset:5120
	ds_load_b128 v[124:127], v166 offset:5632
	ds_load_b128 v[128:131], v166 offset:6144
	ds_load_b128 v[132:135], v166 offset:6656
	ds_load_b128 v[136:139], v166 offset:7168
	ds_load_b128 v[140:143], v166 offset:7680
	s_wait_alu depctr_vm_vsrc(6)
	ds_load_2addr_b32 v[152:153], v164 offset0:80 offset1:128
	ds_load_b32 v144, v165 offset:64
	ds_load_b32 v145, v163
	ds_load_2addr_b32 v[156:157], v164 offset0:144 offset1:160
	ds_load_b32 v148, v162 offset:35520
	ds_load_b32 v149, v167
	ds_load_b32 v150, v168
	s_add_nc_u64 s[2:3], s[46:47], 0x280
	s_add_nc_u64 s[62:63], s[76:77], 0x1400
	s_bitset1_b32 s3, 31
	s_bitset1_b32 s63, 31
	tensor_load_to_lds s[0:3], s[8:15]
	s_add_nc_u64 s[66:67], s[78:79], 20
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u64_e32 v[0:1], 0x280, v[154:155]
	s_bitset1_b32 s67, 31
	v_readfirstlane_b32 s0, v146
	v_readfirstlane_b32 s1, v147
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_or_b32_e32 v1, 0x80000000, v1
	v_readfirstlane_b32 s2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s3, v1
	tensor_load_to_lds s[60:63], s[24:31]
	tensor_load_to_lds s[64:67], s[36:43]
	tensor_load_to_lds s[0:3], s[16:23]
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v170
	ds_load_b128 v[4:7], v170 offset:32
	ds_load_b128 v[8:11], v170 offset:64
	ds_load_b128 v[12:15], v170 offset:96
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[80:87], v[0:15], v[72:79], v152, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[88:95], v[0:15], v[64:71], v144, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63], v[96:103], v[0:15], v[56:63], v145, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55], v[104:111], v[0:15], v[48:55], v153, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[112:119], v[0:15], v[40:47], v156, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[120:127], v[0:15], v[32:39], v157, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[128:135], v[0:15], v[16:23], v148, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[136:143], v[0:15], v[24:31], v149, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_add_nc_u32_e32 v147, 0x9000, v165
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[80:83], v166 offset:38400
	ds_load_b128 v[84:87], v166 offset:38912
	ds_load_b128 v[0:3], v170 offset:37376
	ds_load_b128 v[4:7], v170 offset:37408
	ds_load_b128 v[8:11], v170 offset:37440
	ds_load_b128 v[12:15], v170 offset:37472
	ds_load_b32 v144, v168 offset:37376
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[148:149], v147 offset0:144 offset1:160
	ds_load_b128 v[88:91], v166 offset:37376
	ds_load_b128 v[92:95], v166 offset:37888
	ds_load_b128 v[96:99], v166 offset:39424
	ds_load_b128 v[100:103], v166 offset:39936
	ds_load_b128 v[104:107], v166 offset:40448
	ds_load_b128 v[108:111], v166 offset:40960
	ds_load_b128 v[112:115], v166 offset:41472
	ds_load_b128 v[116:119], v166 offset:41984
	ds_load_b128 v[120:123], v166 offset:42496
	ds_load_b128 v[124:127], v166 offset:43008
	ds_load_b128 v[128:131], v166 offset:43520
	ds_load_b128 v[132:135], v166 offset:44032
	ds_load_b128 v[136:139], v166 offset:44544
	ds_load_b128 v[140:143], v166 offset:45056
	ds_load_b32 v145, v169 offset:37504
	ds_load_b32 v146, v167 offset:37376
	ds_load_2addr_b32 v[150:151], v171 offset0:80 offset1:128
	ds_load_2addr_b32 v[152:153], v147 offset0:192 offset1:224
	s_wait_dscnt 0x12
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[80:87], v[0:15], v[64:71], v148, v144 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63], v[96:103], v[0:15], v[56:63], v149, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[88:95], v[0:15], v[72:79], v150, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55], v[104:111], v[0:15], v[48:55], v151, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47], v[112:119], v[0:15], v[40:47], v152, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[128:135], v[0:15], v[16:23], v153, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[120:127], v[0:15], v[32:39], v145, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[136:143], v[0:15], v[24:31], v146, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_lshl_or_b32 v98, v160, 3, s5
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_mul_u64 s[2:3], s[68:69], s[50:51]
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_mad_u32 v0, s74, s35, v98
	s_lshl_b64 s[0:1], s[70:71], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_cmp_lg_u32 s35, 0x80000000
	s_mov_b32 s8, s12
	s_add_nc_u64 s[2:3], s[72:73], s[2:3]
	s_cselect_b32 s13, s51, 0
	s_cselect_b32 s12, s35, 0x200
	v_ashrrev_i32_e32 v1, 31, v0
	s_lshl_b32 s48, s33, 3
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	s_mul_u64 s[2:3], s[48:49], s[12:13]
	s_mov_b32 s5, s9
	v_lshl_add_u64 v[96:97], v[0:1], 1, s[6:7]
	s_barrier_wait -1
	s_mul_i32 s7, s33, 0xdc0
	s_add_nc_u64 s[46:47], s[2:3], s[0:1]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x7
	global_load_b128 v[0:3], v[96:97], off
	global_load_b128 v[4:7], v[96:97], off offset:32
	global_load_b128 v[8:11], v[96:97], off offset:64
	global_load_b128 v[12:15], v[96:97], off offset:96
	global_load_b128 v[80:83], v[96:97], off offset:128
	global_load_b128 v[84:87], v[96:97], off offset:160
	global_load_b128 v[88:91], v[96:97], off offset:192
	global_load_b128 v[92:95], v[96:97], off offset:224
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_lshlrev_b32_e32 v96, 1, v98
	s_mov_b32 s6, s10
	s_mov_b32 s11, s49
	s_add_co_i32 s45, s45, s7
	s_or_b32 s7, s34, 0x2000000
	v_lshl_or_b32 v96, v161, 10, v96
	s_and_b32 s10, s13, 0xffff
	s_mov_b32 s9, s12
	s_bitset1_b32 s47, 31
	s_wait_loadcnt 0x7
	v_dual_add_nc_u32 v130, 0, v96 :: v_dual_lshlrev_b32 v98, 16, v3
	v_and_b32_e32 v97, 0xffff0000, v0
	v_lshlrev_b32_e32 v96, 16, v0
	v_and_b32_e32 v99, 0xffff0000, v3
	v_and_b32_e32 v3, 0xffff0000, v2
	v_lshlrev_b32_e32 v2, 16, v2
	v_and_b32_e32 v101, 0xffff0000, v1
	s_wait_loadcnt 0x6
	v_dual_lshlrev_b32 v100, 16, v1 :: v_dual_lshlrev_b32 v0, 16, v4
	v_and_b32_e32 v1, 0xffff0000, v4
	v_and_b32_e32 v103, 0xffff0000, v7
	v_lshlrev_b32_e32 v102, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v6
	v_lshlrev_b32_e32 v6, 16, v6
	v_and_b32_e32 v105, 0xffff0000, v5
	s_wait_loadcnt 0x5
	v_dual_lshlrev_b32 v104, 16, v5 :: v_dual_lshlrev_b32 v4, 16, v8
	v_and_b32_e32 v5, 0xffff0000, v8
	v_and_b32_e32 v107, 0xffff0000, v11
	v_lshlrev_b32_e32 v106, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v10
	v_lshlrev_b32_e32 v10, 16, v10
	v_and_b32_e32 v109, 0xffff0000, v9
	s_wait_loadcnt 0x4
	v_dual_lshlrev_b32 v108, 16, v9 :: v_dual_lshlrev_b32 v8, 16, v12
	v_and_b32_e32 v9, 0xffff0000, v12
	v_and_b32_e32 v111, 0xffff0000, v15
	v_lshlrev_b32_e32 v110, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v14
	v_lshlrev_b32_e32 v14, 16, v14
	v_and_b32_e32 v113, 0xffff0000, v13
	s_wait_loadcnt 0x3
	v_dual_lshlrev_b32 v112, 16, v13 :: v_dual_lshlrev_b32 v12, 16, v80
	v_and_b32_e32 v13, 0xffff0000, v80
	v_and_b32_e32 v115, 0xffff0000, v83
	v_lshlrev_b32_e32 v114, 16, v83
	v_and_b32_e32 v83, 0xffff0000, v82
	v_lshlrev_b32_e32 v82, 16, v82
	v_and_b32_e32 v117, 0xffff0000, v81
	s_wait_loadcnt 0x2
	v_dual_lshlrev_b32 v116, 16, v81 :: v_dual_lshlrev_b32 v80, 16, v84
	v_and_b32_e32 v81, 0xffff0000, v84
	v_and_b32_e32 v119, 0xffff0000, v87
	v_lshlrev_b32_e32 v118, 16, v87
	v_and_b32_e32 v87, 0xffff0000, v86
	v_lshlrev_b32_e32 v86, 16, v86
	v_and_b32_e32 v121, 0xffff0000, v85
	s_wait_loadcnt 0x1
	v_dual_lshlrev_b32 v120, 16, v85 :: v_dual_lshlrev_b32 v84, 16, v88
	v_and_b32_e32 v85, 0xffff0000, v88
	v_and_b32_e32 v123, 0xffff0000, v91
	v_lshlrev_b32_e32 v122, 16, v91
	v_and_b32_e32 v91, 0xffff0000, v90
	v_lshlrev_b32_e32 v90, 16, v90
	v_and_b32_e32 v125, 0xffff0000, v89
	s_wait_loadcnt 0x0
	v_dual_lshlrev_b32 v124, 16, v89 :: v_dual_lshlrev_b32 v88, 16, v92
	v_and_b32_e32 v89, 0xffff0000, v92
	v_and_b32_e32 v127, 0xffff0000, v95
	v_lshlrev_b32_e32 v126, 16, v95
	v_and_b32_e32 v95, 0xffff0000, v94
	v_lshlrev_b32_e32 v94, 16, v94
	v_and_b32_e32 v129, 0xffff0000, v93
	v_lshlrev_b32_e32 v128, 16, v93
	v_pk_add_f32 v[74:75], v[74:75], v[100:101]
	v_pk_add_f32 v[76:77], v[76:77], v[2:3]
	v_pk_add_f32 v[2:3], v[78:79], v[98:99]
	v_pk_add_f32 v[72:73], v[72:73], v[96:97]
	v_pk_add_f32 v[66:67], v[66:67], v[104:105]
	v_pk_add_f32 v[68:69], v[68:69], v[6:7]
	v_pk_add_f32 v[6:7], v[70:71], v[102:103]
	v_pk_add_f32 v[64:65], v[64:65], v[0:1]
	v_pk_add_f32 v[58:59], v[58:59], v[108:109]
	v_pk_add_f32 v[60:61], v[60:61], v[10:11]
	v_pk_add_f32 v[10:11], v[62:63], v[106:107]
	v_pk_add_f32 v[56:57], v[56:57], v[4:5]
	v_pk_add_f32 v[50:51], v[50:51], v[112:113]
	v_pk_add_f32 v[52:53], v[52:53], v[14:15]
	v_pk_add_f32 v[14:15], v[54:55], v[110:111]
	v_pk_add_f32 v[48:49], v[48:49], v[8:9]
	v_pk_add_f32 v[42:43], v[42:43], v[116:117]
	v_pk_add_f32 v[44:45], v[44:45], v[82:83]
	v_pk_add_f32 v[46:47], v[46:47], v[114:115]
	v_pk_add_f32 v[40:41], v[40:41], v[12:13]
	v_pk_add_f32 v[34:35], v[34:35], v[120:121]
	v_pk_add_f32 v[36:37], v[36:37], v[86:87]
	v_pk_add_f32 v[38:39], v[38:39], v[118:119]
	v_pk_add_f32 v[32:33], v[32:33], v[80:81]
	v_pk_add_f32 v[54:55], v[18:19], v[124:125]
	v_pk_add_f32 v[62:63], v[20:21], v[90:91]
	v_pk_add_f32 v[70:71], v[22:23], v[122:123]
	v_pk_add_f32 v[78:79], v[16:17], v[84:85]
	v_pk_add_f32 v[80:81], v[26:27], v[128:129]
	v_pk_add_f32 v[28:29], v[28:29], v[94:95]
	v_pk_add_f32 v[30:31], v[30:31], v[126:127]
	v_pk_add_f32 v[82:83], v[24:25], v[88:89]
	v_cvt_pk_bf16_f32 v3, v2, v3
	v_cvt_pk_bf16_f32 v2, v76, v77
	v_cvt_pk_bf16_f32 v1, v74, v75
	v_cvt_pk_bf16_f32 v0, v72, v73
	v_cvt_pk_bf16_f32 v7, v6, v7
	v_cvt_pk_bf16_f32 v6, v68, v69
	v_cvt_pk_bf16_f32 v5, v66, v67
	v_cvt_pk_bf16_f32 v4, v64, v65
	v_cvt_pk_bf16_f32 v11, v10, v11
	v_cvt_pk_bf16_f32 v10, v60, v61
	v_cvt_pk_bf16_f32 v9, v58, v59
	v_cvt_pk_bf16_f32 v8, v56, v57
	v_cvt_pk_bf16_f32 v15, v14, v15
	v_cvt_pk_bf16_f32 v14, v52, v53
	v_cvt_pk_bf16_f32 v13, v50, v51
	v_cvt_pk_bf16_f32 v12, v48, v49
	v_cvt_pk_bf16_f32 v19, v46, v47
	v_cvt_pk_bf16_f32 v18, v44, v45
	v_cvt_pk_bf16_f32 v17, v42, v43
	v_cvt_pk_bf16_f32 v16, v40, v41
	v_cvt_pk_bf16_f32 v23, v38, v39
	v_cvt_pk_bf16_f32 v22, v36, v37
	v_cvt_pk_bf16_f32 v21, v34, v35
	v_cvt_pk_bf16_f32 v20, v32, v33
	v_cvt_pk_bf16_f32 v27, v70, v71
	v_cvt_pk_bf16_f32 v26, v62, v63
	v_cvt_pk_bf16_f32 v25, v54, v55
	v_cvt_pk_bf16_f32 v24, v78, v79
	v_cvt_pk_bf16_f32 v31, v30, v31
	v_cvt_pk_bf16_f32 v30, v28, v29
	v_cvt_pk_bf16_f32 v29, v80, v81
	v_cvt_pk_bf16_f32 v28, v82, v83
	s_wait_alu depctr_va_vdst(14)
	ds_store_b128 v130, v[0:3]
	ds_store_b128 v130, v[4:7] offset:32
	ds_store_b128 v130, v[8:11] offset:64
	ds_store_b128 v130, v[12:15] offset:96
	s_wait_alu depctr_va_vdst(12)
	ds_store_b128 v130, v[16:19] offset:128
	s_wait_alu depctr_va_vdst(8)
	ds_store_b128 v130, v[20:23] offset:160
	s_wait_alu depctr_va_vdst(4)
	ds_store_b128 v130, v[24:27] offset:192
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v130, v[28:31] offset:224
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_store_from_lds s[44:47], s[4:11]
	s_wait_tensorcnt 0x0
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 176
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 172
		.amdhsa_next_free_sgpr 84
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 56
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
	.size	gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1, .Lfunc_end0-gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1

	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.num_vgpr, 172
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.num_agpr, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.numbered_sgpr, 84
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.num_named_barrier, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.private_seg_size, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.uses_vcc, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.uses_flat_scratch, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.has_dyn_sized_stack, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.has_recursion, 0
	.set gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.has_indirect_call, 0
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
      - .offset:         164
        .size:           4
        .value_kind:     by_value
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .offset:         172
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 176
    .max_flat_workgroup_size: 128
    .name:           gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     84
    .sgpr_spill_count: 0
    .symbol:         gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     172
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
