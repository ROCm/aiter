	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_gemm1_a8w4
	.p2align	8
	.type	moe_gemm1_a8w4,@function
moe_gemm1_a8w4:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b96 s[72:74], s[0:1], 0x5c nv
	s_load_b128 s[52:55], s[0:1], 0x40 nv
	v_mov_b32_e32 v1, 0
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
	s_wait_kmcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v1, s[52:53] offset:768
	s_add_co_i32 s2, s73, 0xff
	v_lshl_add_u32 v6, v0, 4, 0
	s_ashr_i32 s3, s2, 31
	v_readfirstlane_b32 s41, v0
	s_lshr_b32 s3, s3, 24
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s2, s3
	v_add_nc_u32_e32 v7, 0x10000, v6
	s_and_b32 s4, s3, 0xffffff00
	s_ashr_i32 s3, s3, 8
	s_cmp_lg_u32 s2, s4
	v_add_nc_u32_e32 v9, 0x11000, v6
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s2, 0
	v_add_nc_u32_e32 v11, 0x12000, v6
	s_cselect_b32 s2, -1, 0
	v_add_nc_u32_e32 v13, 0x13000, v6
	s_and_b32 s2, s2, s4
	s_sub_co_ci_u32 s2, s3, 0
	s_add_co_i32 s3, s72, 63
	v_add_nc_u32_e32 v14, 0x13800, v6
	s_ashr_i32 s4, s3, 31
	v_add_nc_u32_e32 v15, 0x14000, v6
	s_lshr_b32 s4, s4, 26
	v_add_nc_u32_e32 v16, 0x14800, v6
	s_add_co_i32 s4, s3, s4
	v_add_nc_u32_e32 v17, 0x15000, v6
	s_and_b32 s5, s4, 0xffffffc0
	s_ashr_i32 s4, s4, 6
	s_cmp_lg_u32 s3, s5
	v_add_nc_u32_e32 v18, 0x15800, v6
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s3, 0
	s_getreg_b32 s3, hwreg(HW_REG_IB_STS2, 6, 4)
	s_cselect_b32 s6, -1, 0
	s_bfe_u32 s7, ttmp6, 0x4000c
	s_and_b32 s9, ttmp6, 15
	s_add_co_i32 s7, s7, 1
	s_lshl_b32 s8, s2, 4
	s_mul_i32 s7, ttmp9, s7
	s_and_b32 s5, s6, s5
	s_add_co_i32 s9, s9, s7
	s_cmp_eq_u32 s3, 0
	v_add_nc_u32_e32 v19, 0x16000, v6
	s_cselect_b32 s3, ttmp9, s9
	s_abs_i32 s6, s8
	s_abs_i32 s10, s3
	s_cvt_f32_u32 s7, s6
	s_sub_co_i32 s9, 0, s6
	v_add_nc_u32_e32 v20, 0x16800, v6
	v_add_nc_u32_e32 v21, 0x17000, v6
	v_rcp_iflag_f32_e32 v3, s7
	v_add_nc_u32_e32 v22, 0x17800, v6
	v_add_nc_u32_e32 v23, 0x18000, v6
	v_add_nc_u32_e32 v24, 0x18800, v6
	v_add_nc_u32_e32 v25, 0x19000, v6
	v_add_nc_u32_e32 v26, 0x19800, v6
	v_add_nc_u32_e32 v27, 0x1a000, v6
	v_add_nc_u32_e32 v28, 0x1a800, v6
	v_readfirstlane_b32 s7, v3
	v_add_nc_u32_e32 v29, 0x1b000, v6
	v_add_nc_u32_e32 v30, 0x1b800, v6
	v_add_nc_u32_e32 v31, 0x1c000, v6
	v_add_nc_u32_e32 v32, 0x1c800, v6
	s_mul_f32 s7, s7, 0x4f7ffffe
	v_add_nc_u32_e32 v33, 0x1d000, v6
	v_add_nc_u32_e32 v34, 0x1d800, v6
	v_add_nc_u32_e32 v35, 0x1e000, v6
	s_cvt_u32_f32 s7, s7
	v_add_nc_u32_e32 v36, 0x1e800, v6
	v_add_nc_u32_e32 v37, 0x1f000, v6
	v_add_nc_u32_e32 v38, 0x1f800, v6
	s_mul_i32 s9, s9, s7
	v_add_nc_u32_e32 v39, 0x20000, v6
	s_mul_hi_u32 s9, s7, s9
	v_add_nc_u32_e32 v40, 0x20800, v6
	s_add_co_i32 s7, s7, s9
	s_xor_b32 s9, s3, s8
	s_mul_hi_u32 s7, s10, s7
	s_ashr_i32 s9, s9, 31
	s_mul_i32 s11, s7, s6
	v_add_nc_u32_e32 v41, 0x21000, v6
	s_sub_co_i32 s10, s10, s11
	s_add_co_i32 s11, s7, 1
	s_sub_co_i32 s13, s10, s6
	s_cmp_ge_u32 s10, s6
	v_add_nc_u32_e32 v42, 0x21800, v6
	s_cselect_b32 s7, s11, s7
	s_cselect_b32 s10, s13, s10
	s_add_co_i32 s11, s7, 1
	s_cmp_ge_u32 s10, s6
	v_add_nc_u32_e32 v43, 0x22000, v6
	s_cselect_b32 s6, s11, s7
	v_add_nc_u32_e32 v44, 0x22800, v6
	s_xor_b32 s6, s6, s9
	v_add_nc_u32_e32 v45, 0x23000, v6
	s_sub_co_i32 s7, s6, s9
	v_add_nc_u32_e32 v46, 0x23800, v6
	s_mul_i32 s7, s7, s8
	v_add_nc_u32_e32 v47, 0x24000, v6
	s_cmp_lg_u32 s3, s7
	v_add_nc_u32_e32 v48, 0x24800, v6
	s_cselect_b32 s7, -1, 0
	s_xor_b32 s2, s2, s3
	v_add_nc_u32_e32 v49, 0x25000, v6
	s_cmp_lt_i32 s2, 0
	v_add_nc_u32_e32 v50, 0x25800, v6
	s_cselect_b32 s2, -1, 0
	v_add_nc_u32_e32 v51, 0x26000, v6
	s_and_b32 s2, s2, s7
	s_sub_co_ci_u32 s2, s6, s9
	v_add_nc_u32_e32 v52, 0x26800, v6
	s_mul_i32 s6, s2, s8
	s_lshl_b32 s7, s2, 4
	s_sub_co_i32 s3, s3, s6
	s_cmp_lg_u32 s5, 0
	v_add_nc_u32_e32 v8, 0x10800, v6
	s_sub_co_ci_u32 s2, s4, s7
	s_abs_i32 s9, s3
	s_min_i32 s4, s2, 16
	v_add_nc_u32_e32 v10, 0x11800, v6
	s_abs_i32 s5, s4
	v_add_nc_u32_e32 v12, 0x12800, v6
	s_cvt_f32_u32 s6, s5
	s_sub_co_i32 s8, 0, s5
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v3, s6
	v_nop
	v_readfirstlane_b32 s6, v3
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s6, s6
	s_mul_i32 s8, s8, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s8, s6, s8
	s_add_co_i32 s6, s6, s8
	s_xor_b32 s8, s3, s4
	s_mul_hi_u32 s6, s9, s6
	s_ashr_i32 s8, s8, 31
	s_mul_i32 s10, s6, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s9, s9, s10
	s_add_co_i32 s10, s6, 1
	s_sub_co_i32 s11, s9, s5
	s_cmp_ge_u32 s9, s5
	s_cselect_b32 s6, s10, s6
	s_cselect_b32 s9, s11, s9
	s_add_co_i32 s10, s6, 1
	s_cmp_ge_u32 s9, s5
	s_cselect_b32 s5, s10, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s5, s5, s8
	s_sub_co_i32 s6, s5, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s6, s4, s6
	s_cmp_lg_u32 s3, s6
	s_cselect_b32 s6, -1, 0
	s_xor_b32 s2, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s6
	s_sub_co_ci_u32 s2, s5, s8
	s_add_co_i32 s3, s3, s7
	s_mul_i32 s4, s2, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s48, s3, s4
	s_movk_i32 s3, 0x60
	s_lshl_b32 s66, s48, 6
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s5, v2
	s_cmp_gt_i32 s5, s66
	s_movk_i32 s5, 0xc0
	s_cselect_b32 s3, s3, 0x120
	s_cselect_b32 s6, 0, 0xc1
	v_mov_b32_e32 v2, s3
	s_cselect_b32 s5, s5, 0x180
	s_or_b32 s7, s3, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v2, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s4, v2
	s_cmp_gt_i32 s4, s66
	s_cselect_b32 s4, s6, s7
	s_cselect_b32 s3, s3, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s4, s3
	s_lshr_b32 s5, s5, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v2, s5
	s_or_b32 s7, s5, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v2, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s6, v2
	s_cmp_gt_i32 s6, s66
	s_cselect_b32 s4, s4, s7
	s_cselect_b32 s3, s5, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s4, s3
	s_lshr_b32 s8, s5, 1
	s_mov_b32 s5, 0
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v2, s8
	s_add_co_i32 s9, s8, 1
	s_mov_b32 s7, s5
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v2, v2, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s6, v2
	s_cmp_gt_i32 s6, s66
	s_cselect_b32 s6, s4, s9
	s_cselect_b32 s4, s8, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[8:9], s[6:7], s[4:5]
	s_lshr_b64 s[8:9], s[8:9], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[10:11], s[8:9], 2
	s_add_co_i32 s7, s8, 1
	s_add_nc_u64 s[10:11], s[52:53], s[10:11]
	global_load_b32 v1, v1, s[10:11]
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s3, v1
	s_cmp_gt_i32 s3, s66
	s_cselect_b32 s3, s6, s7
	s_cselect_b32 s4, s8, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s3, s4
	s_lshr_b32 s6, s6, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s6
	s_add_co_i32 s8, s6, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s7, v1
	s_cmp_gt_i32 s7, s66
	s_cselect_b32 s3, s3, s8
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s3, s4
	s_lshr_b32 s6, s6, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s6
	s_add_co_i32 s8, s6, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s7, v1
	s_cmp_gt_i32 s7, s66
	s_cselect_b32 s3, s3, s8
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s3, s4
	s_lshr_b32 s6, s6, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s6
	s_add_co_i32 s8, s6, 1
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s7, v1
	s_cmp_gt_i32 s7, s66
	s_cselect_b32 s3, s3, s8
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s3, s4
	s_lshr_b32 s6, s6, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s7, s6, 0x17f
	s_add_co_i32 s8, s6, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s7
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s7, v1
	s_cmp_gt_i32 s7, s66
	s_mov_b32 s7, s5
	s_cselect_b32 s3, s3, s8
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s3, s4
	s_lshr_b32 s4, s4, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s6, s4, 0x17f
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s6
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[52:53] scale_offset
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s6, v1
	s_cmp_gt_i32 s6, s66
	s_mov_b32 s6, s5
	s_cselect_b32 s34, s3, s4
	s_mov_b32 s4, s5
	s_cmp_lt_u32 s34, 0x180
	v_mov_b64_e32 v[2:3], s[4:5]
	s_cselect_b32 s3, s34, 0x17f
	v_mov_b64_e32 v[4:5], s[6:7]
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v1, s3
	s_cmp_gt_u32 s34, 0x17f
	s_wait_alu depctr_va_vdst(0)
	global_load_b32 v1, v1, s[52:53] scale_offset
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
	ds_store_b128 v12, v[2:5]
	ds_store_b128 v13, v[2:5]
	ds_store_b128 v14, v[2:5]
	ds_store_b128 v15, v[2:5]
	ds_store_b128 v16, v[2:5]
	ds_store_b128 v17, v[2:5]
	ds_store_b128 v18, v[2:5]
	ds_store_b128 v19, v[2:5]
	ds_store_b128 v20, v[2:5]
	ds_store_b128 v21, v[2:5]
	ds_store_b128 v22, v[2:5]
	ds_store_b128 v23, v[2:5]
	ds_store_b128 v24, v[2:5]
	ds_store_b128 v25, v[2:5]
	ds_store_b128 v26, v[2:5]
	ds_store_b128 v27, v[2:5]
	ds_store_b128 v28, v[2:5]
	ds_store_b128 v29, v[2:5]
	ds_store_b128 v30, v[2:5]
	ds_store_b128 v31, v[2:5]
	ds_store_b128 v32, v[2:5]
	ds_store_b128 v33, v[2:5]
	ds_store_b128 v34, v[2:5]
	ds_store_b128 v35, v[2:5]
	ds_store_b128 v36, v[2:5]
	ds_store_b128 v37, v[2:5]
	ds_store_b128 v38, v[2:5]
	ds_store_b128 v39, v[2:5]
	ds_store_b128 v40, v[2:5]
	ds_store_b128 v41, v[2:5]
	ds_store_b128 v42, v[2:5]
	ds_store_b128 v43, v[2:5]
	ds_store_b128 v44, v[2:5]
	ds_store_b128 v45, v[2:5]
	ds_store_b128 v46, v[2:5]
	ds_store_b128 v47, v[2:5]
	ds_store_b128 v48, v[2:5]
	ds_store_b128 v49, v[2:5]
	ds_store_b128 v50, v[2:5]
	ds_store_b128 v51, v[2:5]
	ds_store_b128 v52, v[2:5]
	s_wait_loadcnt 0x0
	v_readfirstlane_b32 s3, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_45
	s_clause 0x1
	s_load_b128 s[56:59], s[0:1], 0x10 nv
	s_load_b64 s[52:53], s[0:1], 0x20 nv
	s_lshr_b32 s42, s41, 5
	s_ashr_i32 s67, s66, 31
	s_sub_co_i32 s33, s3, s66
	s_cmp_eq_u32 s42, 0
	s_mul_u64 s[4:5], s[66:67], 0x1c00
	s_mov_b32 s12, s73
	s_cselect_b32 s3, -1, 0
	s_cmp_lg_u32 s42, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[64:65], s[56:57], s[4:5]
	s_cbranch_scc1 .LBB0_3
	s_max_i32 s4, s33, 0
	s_mov_b32 s17, 0
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_or_b32 s19, s65, 0x80000000
	s_mov_b32 s16, 1
	s_mov_b32 s18, s64
	s_or_b32 s6, s5, 0x7fff
	s_or_b32 s7, s4, 0x1000000
	s_movk_i32 s9, 0x1c00
	s_mov_b32 s8, 32
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x7500000
	s_mov_b32 s10, s17
	s_mov_b32 s11, s17
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[4:11]
.LBB0_3:
	s_sub_co_i32 s40, s33, 32
	s_cmp_eq_u32 s42, 1
	s_add_nc_u64 s[20:21], s[64:65], 0x38000
	s_cselect_b32 s14, -1, 0
	s_cmp_lg_u32 s42, 1
	s_mov_b32 s4, 1
	s_cbranch_scc1 .LBB0_5
	s_max_i32 s8, s40, 0
	s_mov_b32 s30, 0
	s_lshl_b32 s9, s8, 16
	s_lshr_b32 s8, s8, 16
	s_or_b32 s7, s21, 0x80000000
	s_add_co_i32 s5, 0, 0x2200
	s_mov_b32 s6, s20
	s_or_b32 s26, s9, 0x7fff
	s_or_b32 s27, s8, 0x1000000
	s_movk_i32 s29, 0x1c00
	s_mov_b32 s28, 32
	s_mov_b32 s25, 0xffff0000
	s_mov_b32 s24, 0x7500000
	s_mov_b32 s31, s30
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[4:7], s[24:31]
.LBB0_5:
	s_ashr_i32 s13, s73, 31
	s_lshl_b32 s76, s2, 8
	s_lshr_b32 s4, s13, 28
	s_ashr_i32 s77, s76, 31
	s_add_co_i32 s4, s73, s4
	s_mov_b32 s35, 0
	s_ashr_i32 s22, s4, 4
	s_mov_b32 s8, 8
	s_ashr_i32 s23, s22, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[4:5], s[22:23], 4
	s_cmp_lg_u64 s[4:5], s[12:13]
	s_mov_b32 s5, s35
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s73, 0
	s_cselect_b32 s75, -1, 0
	s_lshr_b64 s[6:7], s[76:77], 4
	s_and_b32 s2, s75, s2
	s_cmp_eq_u32 s42, 2
	s_wait_alu depctr_vm_vsrc(0)
	v_cndmask_b32_e64 v1, 0, 1, s2
	s_cselect_b32 s16, -1, 0
	s_cmp_lg_u32 s42, 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_readfirstlane_b32 s4, v1
	s_sub_nc_u64 s[4:5], s[22:23], s[4:5]
	s_mul_u64 s[4:5], s[4:5], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[4:5], s[4:5], s[6:7]
	s_mul_u64 s[68:69], s[4:5], 0xe000
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[24:25], s[58:59], s[68:69]
	s_cbranch_scc1 .LBB0_7
	s_add_co_i32 s29, 0, 0x4400
	s_or_b32 s31, s25, 0x80000000
	s_mov_b32 s28, 1
	s_mov_b32 s30, s24
	s_mov_b32 s9, 0xe000
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, s35
	s_mov_b32 s10, s35
	s_mov_b32 s11, s35
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[28:31], s[4:11]
.LBB0_7:
	s_cmp_eq_u32 s42, 3
	s_add_nc_u64 s[26:27], s[24:25], 0x70000
	s_cselect_b32 s23, -1, 0
	s_cmp_lg_u32 s42, 3
	s_cbranch_scc1 .LBB0_9
	s_mov_b32 s4, 0
	s_add_co_i32 s29, 0, 0x8400
	s_or_b32 s31, s27, 0x80000000
	s_mov_b32 s28, 1
	s_mov_b32 s30, s26
	s_mov_b32 s9, 0xe000
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s4
	s_mov_b32 s11, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[28:31], s[4:11]
.LBB0_9:
	v_cndmask_b32_e64 v2, 0, 1, s3
	s_lshr_b64 s[4:5], s[66:67], 2
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_mul_u64 s[70:71], s[4:5], 0x380
	s_mov_b32 s4, 1
	v_cmp_ne_u32_e64 s2, 1, v2
	s_add_nc_u64 s[28:29], s[52:53], s[70:71]
	s_cbranch_vccnz .LBB0_11
	s_mov_b32 s86, 0
	s_add_co_i32 s5, 0, 0xc400
	s_or_b32 s7, s29, 0x80000000
	s_mov_b32 s6, s28
	s_movk_i32 s85, 0xe0
	s_mov_b32 s84, 8
	s_mov_b32 s83, 0x87fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x20000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[4:7], s[80:87]
.LBB0_11:
	v_cndmask_b32_e64 v2, 0, 1, s14
	s_and_not1_b32 vcc_lo, exec_lo, s14
	s_add_nc_u64 s[36:37], s[28:29], 0x1c00
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e64 s3, 1, v2
	s_cbranch_vccnz .LBB0_13
	s_mov_b32 s10, 0
	s_add_co_i32 s45, 0, 0xc500
	s_or_b32 s47, s37, 0x80000000
	s_mov_b32 s44, 1
	s_mov_b32 s46, s36
	s_movk_i32 s9, 0xe0
	s_mov_b32 s7, 0x87fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x20000
	s_mov_b32 s11, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[44:47], s[4:11]
.LBB0_13:
	s_add_nc_u64 s[4:5], s[12:13], 31
	s_mov_b32 s15, 0
	s_lshr_b32 s14, s5, 27
	s_mov_b64 s[6:7], 0xffffffffffffffe0
	s_add_nc_u64 s[8:9], s[4:5], s[14:15]
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_and_b64 s[6:7], s[8:9], s[6:7]
	s_ashr_i64 s[30:31], s[8:9], 5
	s_cmp_lg_u64 s[4:5], s[6:7]
	v_cndmask_b32_e64 v3, 0, 1, s16
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s73, 0xffffffe1
	s_cselect_b32 s5, -1, 0
	s_lshr_b64 s[6:7], s[76:77], 5
	s_and_b32 s4, s5, s4
	s_mul_u64 s[6:7], s[6:7], 0x1c00
	v_cndmask_b32_e64 v2, 0, 1, s4
	v_cmp_ne_u32_e64 s4, 1, v3
	s_and_not1_b32 vcc_lo, exec_lo, s16
	s_mov_b32 s16, 1
	v_readfirstlane_b32 s14, v2
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[10:11], s[44:45], s[6:7]
	s_sub_nc_u64 s[8:9], s[30:31], s[14:15]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[8:9], s[8:9], s[34:35]
	s_mul_u64 s[46:47], s[8:9], 0x1c00
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[38:39], s[10:11], s[46:47]
	s_cbranch_vccnz .LBB0_15
	s_add_co_i32 s17, 0, 0xc600
	s_or_b32 s19, s39, 0x80000000
	s_mov_b32 s18, s38
	s_movk_i32 s13, 0x700
	s_mov_b32 s12, 4
	s_mov_b32 s11, 0x407fff
	s_mov_b32 s10, 0xffff7fff
	s_mov_b32 s9, 0xffff0000
	s_mov_b32 s8, 0x20000
	s_mov_b32 s14, s15
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[8:15]
.LBB0_15:
	v_cndmask_b32_e64 v3, 0, 1, s23
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_add_nc_u64 s[12:13], s[38:39], 0x7000
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e64 s5, 1, v3
	s_cbranch_vccnz .LBB0_17
	s_mov_b32 s86, 0
	s_add_co_i32 s9, 0, 0xca00
	s_or_b32 s11, s13, 0x80000000
	s_mov_b32 s8, 1
	s_mov_b32 s10, s12
	s_movk_i32 s85, 0x700
	s_mov_b32 s84, 4
	s_mov_b32 s83, 0x407fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x20000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
.LBB0_17:
	s_and_b32 vcc_lo, exec_lo, s2
	s_mov_b32 s8, 1
	s_cbranch_vccz .LBB0_46
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB0_47
.LBB0_19:
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccz .LBB0_48
.LBB0_20:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB0_49
.LBB0_21:
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccz .LBB0_50
.LBB0_22:
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB0_51
.LBB0_23:
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccz .LBB0_52
.LBB0_24:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_26
.LBB0_25:
	s_add_nc_u64 s[18:19], s[12:13], 0x100
	s_mov_b32 s14, 0
	s_add_co_i32 s17, 0, 0x19800
	s_bitset1_b32 s19, 31
	s_movk_i32 s13, 0x700
	s_mov_b32 s12, 4
	s_mov_b32 s11, 0x407fff
	s_mov_b32 s10, 0xffff7fff
	s_mov_b32 s9, 0xffff0000
	s_mov_b32 s8, 0x20000
	s_mov_b32 s15, s14
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[8:15]
.LBB0_26:
	s_set_vgpr_msb 64
	v_bfe_u32 v82 /*v338*/, v0, 4, 1
	v_and_b32_e32 v83 /*v339*/, 15, v0
	s_lshr_b32 s8, s41, 1
	s_lshr_b32 s13, s76, 4
	s_and_b32 s8, s8, 0x7fffffc0
	s_set_vgpr_msb 0x4004
	v_dual_lshlrev_b32 v0, 4, v82 /*v338*/ :: v_dual_lshlrev_b32 v3, 8, v82 /*v338*/
	s_set_vgpr_msb 0x444
	v_dual_mov_b32 v64 /*v320*/, 1 :: v_dual_bitop2_b32 v65 /*v321*/, s8, v83 /*v339*/ bitop3:0x54
	s_lshr_b32 s10, s76, 5
	s_lshl_b32 s9, s42, 6
	s_mulk_i32 s10, 0x1c00
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32 v85 /*v341*/, 0x110, v65 /*v321*/, v0
	s_set_vgpr_msb 0x4400
	v_dual_sub_nc_u32 v0, s22, v1 :: v_dual_sub_nc_u32 v1, s30, v2
	s_and_b32 s35, s9, 0xc0
	s_lshr_b32 s14, s8, 2
	s_lshl_b32 s11, s35, 7
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v0, s34, v0
	v_mul_lo_u32 v1, s34, v1
	s_addk_co_i32 s11, 0x4400
	s_max_i32 s8, s33, 0
	s_mov_b32 s16, 0
	v_mov_b32_e32 v128, 0
	s_mov_b32 s17, 0xffff0000
	s_mov_b32 s30, s16
	v_add_nc_u32_e32 v0, s13, v0
	v_mul_lo_u32 v1, 0x1c00, v1
	s_mov_b32 s31, s16
	s_lshr_b32 s9, s66, 2
	s_movk_i32 s29, 0x1c00
	s_set_vgpr_msb 64
	v_mul_lo_u32 v88 /*v344*/, 0xe000, v0
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v129, v128
	s_mov_b32 s28, 32
	s_mov_b32 s24, 0x7500000
	v_add_nc_u32_e32 v0, s10, v1
	s_set_vgpr_msb 4
	v_lshlrev_b32_e32 v7, 2, v82 /*v338*/
	s_mov_b32 s25, s17
	v_dual_mov_b32 v130, v128 :: v_dual_mov_b32 v131, v128
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v89 /*v345*/, s44, v0
	s_set_vgpr_msb 0x4004
	v_dual_lshlrev_b32 v4, 4, v83 /*v339*/ :: v_dual_lshlrev_b32 v5, 2, v83 /*v339*/
	v_dual_mov_b32 v132, v128 :: v_dual_bitop2_b32 v6, s14, v83 /*v339*/ bitop3:0x54
	v_mov_b32_e32 v133, v128
	s_set_vgpr_msb 0x440
	s_delay_alu instid0(VALU_DEP_3)
	v_or3_b32 v87 /*v343*/, s11, v3, v4
	s_lshl_b32 s11, s8, 16
	s_lshr_b32 s8, s8, 16
	s_or_b32 s26, s11, 0x7fff
	s_or_b32 s27, s8, 0x1000000
	s_max_i32 s8, s40, 0
	s_mov_b64 s[42:43], s[30:31]
	s_lshl_b32 s11, s8, 16
	s_lshr_b32 s8, s8, 16
	v_lshl_or_b32 v86 /*v342*/, s35, 3, v5
	s_bitset1_b32 s8, 24
	v_lshl_or_b32 v84 /*v340*/, v6, 5, v7
	s_mov_b64 s[38:39], s[26:27]
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v134, v128 :: v_dual_mov_b32 v135, v128
	s_set_vgpr_msb 64
	v_dual_mov_b32 v66 /*v322*/, v128 :: v_dual_mov_b32 v67 /*v323*/, v128
	v_dual_mov_b32 v68 /*v324*/, v128 :: v_dual_mov_b32 v69 /*v325*/, v128
	v_dual_mov_b32 v70 /*v326*/, v128 :: v_dual_mov_b32 v71 /*v327*/, v128
	v_dual_mov_b32 v72 /*v328*/, v128 :: v_dual_mov_b32 v73 /*v329*/, v128
	v_dual_mov_b32 v74 /*v330*/, v128 :: v_dual_mov_b32 v75 /*v331*/, v128
	v_dual_mov_b32 v76 /*v332*/, v128 :: v_dual_mov_b32 v77 /*v333*/, v128
	v_dual_mov_b32 v78 /*v334*/, v128 :: v_dual_mov_b32 v79 /*v335*/, v128
	v_dual_mov_b32 v80 /*v336*/, v128 :: v_dual_mov_b32 v81 /*v337*/, v128
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v152, v128 :: v_dual_mov_b32 v153, v128
	v_dual_mov_b32 v154, v128 :: v_dual_mov_b32 v155, v128
	v_dual_mov_b32 v156, v128 :: v_dual_mov_b32 v157, v128
	v_dual_mov_b32 v158, v128 :: v_dual_mov_b32 v159, v128
	v_dual_mov_b32 v160, v128 :: v_dual_mov_b32 v161, v128
	v_dual_mov_b32 v162, v128 :: v_dual_mov_b32 v163, v128
	v_dual_mov_b32 v164, v128 :: v_dual_mov_b32 v165, v128
	v_dual_mov_b32 v166, v128 :: v_dual_mov_b32 v167, v128
	v_dual_mov_b32 v168, v128 :: v_dual_mov_b32 v169, v128
	v_dual_mov_b32 v170, v128 :: v_dual_mov_b32 v171, v128
	v_dual_mov_b32 v172, v128 :: v_dual_mov_b32 v173, v128
	v_dual_mov_b32 v174, v128 :: v_dual_mov_b32 v175, v128
	v_dual_mov_b32 v176, v128 :: v_dual_mov_b32 v177, v128
	v_dual_mov_b32 v178, v128 :: v_dual_mov_b32 v179, v128
	v_dual_mov_b32 v180, v128 :: v_dual_mov_b32 v181, v128
	v_dual_mov_b32 v182, v128 :: v_dual_mov_b32 v183, v128
	v_dual_mov_b32 v184, v128 :: v_dual_mov_b32 v185, v128
	v_dual_mov_b32 v186, v128 :: v_dual_mov_b32 v187, v128
	v_dual_mov_b32 v188, v128 :: v_dual_mov_b32 v189, v128
	v_dual_mov_b32 v190, v128 :: v_dual_mov_b32 v191, v128
	v_dual_mov_b32 v192, v128 :: v_dual_mov_b32 v193, v128
	v_dual_mov_b32 v194, v128 :: v_dual_mov_b32 v195, v128
	v_dual_mov_b32 v196, v128 :: v_dual_mov_b32 v197, v128
	v_dual_mov_b32 v198, v128 :: v_dual_mov_b32 v199, v128
	v_dual_mov_b32 v200, v128 :: v_dual_mov_b32 v201, v128
	v_dual_mov_b32 v202, v128 :: v_dual_mov_b32 v203, v128
	v_dual_mov_b32 v204, v128 :: v_dual_mov_b32 v205, v128
	v_dual_mov_b32 v206, v128 :: v_dual_mov_b32 v207, v128
	v_dual_mov_b32 v208, v128 :: v_dual_mov_b32 v209, v128
	v_dual_mov_b32 v210, v128 :: v_dual_mov_b32 v211, v128
	v_dual_mov_b32 v212, v128 :: v_dual_mov_b32 v213, v128
	v_dual_mov_b32 v214, v128 :: v_dual_mov_b32 v215, v128
	v_dual_mov_b32 v216, v128 :: v_dual_mov_b32 v217, v128
	v_dual_mov_b32 v218, v128 :: v_dual_mov_b32 v219, v128
	v_dual_mov_b32 v220, v128 :: v_dual_mov_b32 v221, v128
	v_dual_mov_b32 v222, v128 :: v_dual_mov_b32 v223, v128
	v_dual_mov_b32 v224, v128 :: v_dual_mov_b32 v225, v128
	v_dual_mov_b32 v226, v128 :: v_dual_mov_b32 v227, v128
	v_dual_mov_b32 v228, v128 :: v_dual_mov_b32 v229, v128
	v_dual_mov_b32 v230, v128 :: v_dual_mov_b32 v231, v128
	v_dual_mov_b32 v232, v128 :: v_dual_mov_b32 v233, v128
	v_dual_mov_b32 v234, v128 :: v_dual_mov_b32 v235, v128
	v_dual_mov_b32 v236, v128 :: v_dual_mov_b32 v237, v128
	v_dual_mov_b32 v238, v128 :: v_dual_mov_b32 v239, v128
	v_dual_mov_b32 v240, v128 :: v_dual_mov_b32 v241, v128
	v_dual_mov_b32 v242, v128 :: v_dual_mov_b32 v243, v128
	v_dual_mov_b32 v244, v128 :: v_dual_mov_b32 v245, v128
	v_dual_mov_b32 v246, v128 :: v_dual_mov_b32 v247, v128
	v_dual_mov_b32 v248, v128 :: v_dual_mov_b32 v249, v128
	v_dual_mov_b32 v250, v128 :: v_dual_mov_b32 v251, v128
	v_dual_mov_b32 v252, v128 :: v_dual_mov_b32 v253, v128
	v_dual_mov_b32 v254, v128 :: v_dual_mov_b32 v255, v128
	s_addk_co_i32 s11, 0x7fff
	s_mov_b32 s39, s8
	s_mul_i32 s78, s9, 0x380
	s_mul_i32 s10, s48, 0x70000
	s_add_nc_u64 s[8:9], s[44:45], s[46:47]
	s_mov_b32 s12, 4
	s_mov_b32 s60, 1
	s_mov_b32 s72, 2
	s_mov_b32 s20, 8
	s_mov_b64 s[40:41], s[28:29]
	s_mov_b64 s[36:37], s[24:25]
	s_mov_b32 s38, s11
	s_mov_b32 s21, 0xe000
	s_add_co_i32 s79, s10, s56
	s_add_nc_u64 s[6:7], s[8:9], s[6:7]
	s_mov_b64 s[56:57], 0
	s_mov_b32 s19, 0x8007fff
	s_mov_b32 s18, 0xffff7fff
	s_movk_i32 s49, 0xe0
	s_mov_b32 s47, 0x87fff
	s_mov_b32 s44, 0x20000
	s_movk_i32 s13, 0x700
	s_mov_b32 s11, 0x407fff
	s_branch .LBB0_28
.LBB0_27:
	s_wait_dscnt 0xc
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
	ds_load_b128 v[64:67], v124 offset:128
	ds_load_b128 v[68:71], v124 offset:160
	ds_load_b128 v[72:75], v124 offset:192
	ds_load_b128 v[76:79], v124 offset:224
	ds_load_b128 v[80:83], v124 offset:4480
	ds_load_b128 v[84:87], v124 offset:4512
	ds_load_b128 v[88:91], v124 offset:4544
	ds_load_b128 v[92:95], v124 offset:4576
	ds_load_b128 v[96:99], v124 offset:8832
	ds_load_b128 v[100:103], v124 offset:8864
	ds_load_b128 v[104:107], v124 offset:8896
	ds_load_b128 v[108:111], v124 offset:8928
	ds_load_b128 v[112:115], v124 offset:13184
	ds_load_b128 v[116:119], v124 offset:13216
	ds_load_b128 v[120:123], v124 offset:13248
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[124:127], v124 offset:13280
	s_set_vgpr_msb 1
	s_wait_dscnt 0x13
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[8:15] /*v[264:271]*/, v[48:63], v[128:135], v140, v150 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[24:31] /*v[280:287]*/, v[48:63], v[66:73] /*v[322:329]*/, v141, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[48:55] /*v[304:311]*/, v[48:63], v[74:81] /*v[330:337]*/, v144, v150 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[56:63] /*v[312:319]*/, v[48:63], v[152:159], v145, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[56:63] /*v[312:319]*/, v[32:47], v[184:191], v145, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[48:55] /*v[304:311]*/, v[32:47], v[176:183], v144, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[24:31] /*v[280:287]*/, v[32:47], v[168:175], v141, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[8:15] /*v[264:271]*/, v[32:47], v[160:167], v140, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[8:15] /*v[264:271]*/, v[16:31], v[192:199], v140, v148 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[24:31] /*v[280:287]*/, v[16:31], v[200:207], v141, v148 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[48:55] /*v[304:311]*/, v[16:31], v[208:215], v144, v148 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[56:63] /*v[312:319]*/, v[16:31], v[216:223], v145, v148 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[56:63] /*v[312:319]*/, v[0:15], v[248:255], v145, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[48:55] /*v[304:311]*/, v[0:15], v[240:247], v144, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[24:31] /*v[280:287]*/, v[0:15], v[232:239], v141, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[8:15] /*v[264:271]*/, v[0:15], v[224:231], v140, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[0:7] /*v[256:263]*/, v[64:79], v[128:135], v136, v146 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[16:23] /*v[272:279]*/, v[64:79], v[66:73] /*v[322:329]*/, v137, v146 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[32:39] /*v[288:295]*/, v[64:79], v[74:81] /*v[330:337]*/, v142, v146 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[40:47] /*v[296:303]*/, v[64:79], v[152:159], v143, v146 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[40:47] /*v[296:303]*/, v[80:95], v[184:191], v143, v147 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[32:39] /*v[288:295]*/, v[80:95], v[176:183], v142, v147 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[16:23] /*v[272:279]*/, v[80:95], v[168:175], v137, v147 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[0:7] /*v[256:263]*/, v[80:95], v[160:167], v136, v147 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[0:7] /*v[256:263]*/, v[96:111], v[192:199], v136, v138 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[16:23] /*v[272:279]*/, v[96:111], v[200:207], v137, v138 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[32:39] /*v[288:295]*/, v[96:111], v[208:215], v142, v138 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[40:47] /*v[296:303]*/, v[96:111], v[216:223], v143, v138 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[40:47] /*v[296:303]*/, v[112:127], v[248:255], v143, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[32:39] /*v[288:295]*/, v[112:127], v[240:247], v142, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[16:23] /*v[272:279]*/, v[112:127], v[232:239], v137, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[0:7] /*v[256:263]*/, v[112:127], v[224:231], v136, v139 matrix_a_fmt:MATRIX_FMT_FP4
	s_add_nc_u64 s[56:57], s[56:57], 0x100
	s_add_co_i32 s72, s72, 1
	s_add_nc_u64 s[52:53], s[52:53], 32
	s_cmp_lg_u32 s56, 0x1a00
	s_add_nc_u64 s[58:59], s[58:59], 0x800
	s_set_vgpr_msb 0x100
	s_cbranch_scc0 .LBB0_44
.LBB0_28:
	s_mul_i32 s8, s72, 0xab
	s_wait_tensorcnt 0x2
	s_add_co_i32 s9, s8, 0xfeaa
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_bfe_u32 s9, s9, 0x70009
	s_barrier_signal -1
	s_mul_i32 s9, s9, 3
	s_sub_co_i32 s9, s72, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s9, 0xfffe
	s_and_b32 s9, s9, 0xff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s9, s9, 0xce00
	s_add_co_i32 s9, s9, 0
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v64, s9, v87 /*v343*/ :: v_dual_add_nc_u32 v0, s9, v86 /*v342*/
	s_wait_alu depctr_vm_vsrc(0)
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_dual_add_nc_u32 v65, s9, v84 /*v340*/ :: v_dual_add_nc_u32 v124, s9, v85 /*v341*/
	s_set_vgpr_msb 0x400
	s_barrier_wait -1
	v_add_nc_u32_e32 v66, 0xc400, v0
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[48:51], v124
	ds_load_b128 v[52:55], v124 offset:32
	ds_load_b128 v[56:59], v124 offset:64
	ds_load_b128 v[60:63], v124 offset:96
	ds_load_b128 v[32:35], v124 offset:4352
	ds_load_b128 v[36:39], v124 offset:4384
	ds_load_b128 v[40:43], v124 offset:4416
	ds_load_b128 v[44:47], v124 offset:4448
	ds_load_b128 v[16:19], v124 offset:8704
	ds_load_b128 v[20:23], v124 offset:8736
	ds_load_b128 v[24:27], v124 offset:8768
	ds_load_b128 v[28:31], v124 offset:8800
	ds_load_b128 v[0:3], v124 offset:13056
	ds_load_b128 v[4:7], v124 offset:13088
	ds_load_b128 v[8:11], v124 offset:13120
	ds_load_b128 v[12:15], v124 offset:13152
	s_set_vgpr_msb 64
	ds_load_b128 v[0:3] /*v[256:259]*/, v64 offset:1024
	ds_load_b128 v[4:7] /*v[260:263]*/, v64 offset:1536
	ds_load_b128 v[16:19] /*v[272:275]*/, v64 offset:3072
	ds_load_b128 v[20:23] /*v[276:279]*/, v64 offset:3584
	ds_load_b128 v[32:35] /*v[288:291]*/, v64 offset:5120
	ds_load_b128 v[36:39] /*v[292:295]*/, v64 offset:5632
	ds_load_b128 v[40:43] /*v[296:299]*/, v64 offset:7168
	ds_load_b128 v[44:47] /*v[300:303]*/, v64 offset:7680
	s_set_vgpr_msb 0x4000
	ds_load_2addr_b32 v[136:137], v66 offset0:160 offset1:176
	v_add_nc_u32_e32 v67, 0xc400, v65
	v_add_nc_u32_e32 v68, 0xc408, v65
	s_set_vgpr_msb 64
	ds_load_b128 v[8:11] /*v[264:267]*/, v64
	ds_load_b128 v[12:15] /*v[268:271]*/, v64 offset:512
	ds_load_b128 v[24:27] /*v[280:283]*/, v64 offset:2048
	ds_load_b128 v[28:31] /*v[284:287]*/, v64 offset:2560
	ds_load_b128 v[48:51] /*v[304:307]*/, v64 offset:4096
	ds_load_b128 v[52:55] /*v[308:311]*/, v64 offset:4608
	ds_load_b128 v[56:59] /*v[312:315]*/, v64 offset:6144
	ds_load_b128 v[60:63] /*v[316:319]*/, v64 offset:6656
	s_set_vgpr_msb 0x4000
	ds_load_2addr_b32 v[140:141], v66 offset0:128 offset1:144
	ds_load_2addr_b32 v[144:145], v66 offset0:192 offset1:208
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v64, 0xc410, v65
	v_add_nc_u32_e32 v65, 0xc418, v65
	ds_load_2addr_b32 v[142:143], v66 offset0:224 offset1:240
	s_wait_alu depctr_va_vdst(3)
	ds_load_2addr_b32 v[150:151], v67 offset1:1
	s_wait_alu depctr_va_vdst(2)
	ds_load_2addr_b32 v[148:149], v68 offset1:1
	s_wait_alu depctr_va_vdst(1)
	ds_load_2addr_b32 v[146:147], v64 offset1:1
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[138:139], v65 offset1:1
	s_bfe_u32 s8, s8, 0x70009
	s_and_b32 vcc_lo, exec_lo, s2
	s_mul_i32 s8, s8, 3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s8, s72, s8
	s_and_b32 s10, s8, 0xff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s8, s10, 0xce00
	s_add_co_i32 s61, s8, 0
	s_cbranch_vccz .LBB0_42
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_31
.LBB0_30:
	s_add_nc_u64 s[8:9], s[64:65], s[56:57]
	s_add_co_i32 s14, s79, s56
	s_add_nc_u64 s[8:9], s[8:9], 0x38200
	s_add_co_i32 s81, s61, 0x2200
	s_add_co_i32 s82, s14, 0x38200
	s_or_b32 s83, s9, 0x80000000
	s_mov_b32 s80, s60
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[80:83], s[36:43]
.LBB0_31:
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v64, s58, v88 /*v344*/
	s_and_b32 vcc_lo, exec_lo, s4
	s_add_nc_u64 s[8:9], s[68:69], s[58:59]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_33
	s_add_nc_u64 s[14:15], s[8:9], 0x1000
	s_add_co_i32 s14, s61, 0x4400
	s_bitset1_b32 s15, 31
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v65, s14 :: v_dual_add_nc_u32 v66, 0x1000, v64
	v_mov_b32_e32 v67, s15
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s80, v64 /*v320*/
	s_mov_b32 s22, s16
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s81, v65
	v_readfirstlane_b32 s82, v66
	v_readfirstlane_b32 s83, v67
	s_mov_b32 s23, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[80:83], s[16:23]
.LBB0_33:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_35
	s_add_nc_u64 s[8:9], s[8:9], 0x71000
	s_add_co_i32 s8, s61, 0x8400
	s_bitset1_b32 s9, 31
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v65, s8 :: v_dual_add_nc_u32 v64, 0x71000, v64
	v_mov_b32_e32 v67, s9
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s80, v64 /*v320*/
	s_mov_b32 s22, s16
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s81, v65
	v_readfirstlane_b32 s82, v64
	v_readfirstlane_b32 s83, v67
	s_mov_b32 s23, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[80:83], s[16:23]
.LBB0_35:
	s_and_b32 vcc_lo, exec_lo, s2
	s_mul_i32 s80, s10, 0x3380
	s_cbranch_vccz .LBB0_43
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_38
.LBB0_37:
	s_lshl2_add_u32 s10, s80, 0
	s_add_nc_u64 s[8:9], s[70:71], s[52:53]
	s_add_co_i32 s61, s10, 0xc500
	s_add_co_i32 s10, s78, s52
	s_add_nc_u64 s[8:9], s[8:9], 0x1c40
	s_add_co_i32 s62, s10, 0x1c40
	s_or_b32 s63, s9, 0x80000000
	s_mov_b32 s45, s17
	s_mov_b32 s46, s18
	s_mov_b32 s48, s20
	s_mov_b32 s50, s16
	s_mov_b32 s51, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[60:63], s[44:51]
.LBB0_38:
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v64, s56, v89 /*v345*/
	s_and_b32 vcc_lo, exec_lo, s4
	s_add_nc_u64 s[22:23], s[6:7], s[56:57]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_40
	s_lshl2_add_u32 s10, s80, 0
	s_add_nc_u64 s[8:9], s[22:23], 0x200
	s_add_co_i32 s10, s10, 0xc600
	s_or_b32 s8, s9, 0x80000000
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v65, s10 :: v_dual_add_nc_u32 v66, 0x200, v64
	v_mov_b32_e32 v67, s8
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s84, v64 /*v320*/
	s_mov_b32 s8, s44
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s86, v66
	v_readfirstlane_b32 s85, v65
	v_readfirstlane_b32 s87, v67
	s_mov_b32 s9, s17
	s_mov_b32 s10, s18
	s_mov_b32 s14, s16
	s_mov_b32 s15, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[8:15]
.LBB0_40:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_27
	s_lshl2_add_u32 s10, s80, 0
	s_add_nc_u64 s[8:9], s[22:23], 0x7200
	s_add_co_i32 s10, s10, 0xca00
	s_or_b32 s8, s9, 0x80000000
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v65, s10 :: v_dual_add_nc_u32 v64, 0x7200, v64
	v_mov_b32_e32 v67, s8
	s_set_vgpr_msb 1
	v_readfirstlane_b32 s80, v64 /*v320*/
	s_mov_b32 s8, s44
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s82, v64
	v_readfirstlane_b32 s81, v65
	v_readfirstlane_b32 s83, v67
	s_mov_b32 s9, s17
	s_mov_b32 s10, s18
	s_mov_b32 s14, s16
	s_mov_b32 s15, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[80:83], s[8:15]
	s_branch .LBB0_27
.LBB0_42:
	s_add_nc_u64 s[8:9], s[64:65], s[56:57]
	s_add_co_i32 s14, s79, s56
	s_add_nc_u64 s[8:9], s[8:9], 0x200
	s_add_co_i32 s62, s14, 0x200
	s_or_b32 s63, s9, 0x80000000
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[60:63], s[24:31]
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB0_30
	s_branch .LBB0_31
.LBB0_43:
	s_lshl2_add_u32 s10, s80, 0
	s_add_nc_u64 s[8:9], s[70:71], s[52:53]
	s_add_co_i32 s61, s10, 0xc400
	s_add_co_i32 s10, s78, s52
	s_add_nc_u64 s[8:9], s[8:9], 64
	s_add_co_i32 s62, s10, 64
	s_or_b32 s63, s9, 0x80000000
	s_mov_b32 s45, s17
	s_mov_b32 s46, s18
	s_mov_b32 s48, s20
	s_mov_b32 s50, s16
	s_mov_b32 s51, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[60:63], s[44:51]
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB0_37
	s_branch .LBB0_38
.LBB0_44:
	s_lshl_b32 s3, s35, 1
	s_add_co_i32 s2, 0, 0x19c00
	s_or_b32 s3, s3, 64
	s_set_vgpr_msb 0x44
	v_add_nc_u32_e32 v84 /*v340*/, s2, v84 /*v340*/
	s_set_vgpr_msb 0x4410
	v_and_or_b32 v0, 0x1c0, s3, v83 /*v339*/
	s_add_co_i32 s3, 0, 0x26200
	s_set_vgpr_msb 0x1044
	v_dual_add_nc_u32 v64 /*v320*/, s2, v87 /*v343*/ :: v_dual_add_nc_u32 v83 /*v339*/, s3, v86 /*v342*/
	v_add_nc_u32_e32 v85 /*v341*/, s2, v85 /*v341*/
	v_lshl_add_u32 v86 /*v342*/, v0, 2, s3
	s_set_vgpr_msb 0x4404
	v_add_nc_u32_e32 v64, 0xc400, v84 /*v340*/
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x441
	ds_load_b128 v[56:59] /*v[312:315]*/, v64 /*v320*/
	ds_load_b128 v[60:63] /*v[316:319]*/, v64 /*v320*/ offset:512
	ds_load_b128 v[0:3] /*v[256:259]*/, v64 /*v320*/ offset:2048
	ds_load_b128 v[4:7] /*v[260:263]*/, v64 /*v320*/ offset:2560
	ds_load_b128 v[8:11] /*v[264:267]*/, v64 /*v320*/ offset:4096
	ds_load_b128 v[12:15] /*v[268:271]*/, v64 /*v320*/ offset:4608
	ds_load_b128 v[16:19] /*v[272:275]*/, v64 /*v320*/ offset:6144
	ds_load_b128 v[20:23] /*v[276:279]*/, v64 /*v320*/ offset:6656
	s_set_vgpr_msb 0x4101
	ds_load_2addr_b32 v[148:149], v83 /*v339*/ offset1:16
	ds_load_2addr_b32 v[136:137], v83 /*v339*/ offset0:64 offset1:96
	ds_load_b32 v146, v86 /*v342*/ offset:64
	ds_load_b128 v[0:3], v85 /*v341*/
	ds_load_b128 v[4:7], v85 /*v341*/ offset:32
	ds_load_b128 v[8:11], v85 /*v341*/ offset:64
	ds_load_b128 v[12:15], v85 /*v341*/ offset:96
	ds_load_b128 v[16:19], v85 /*v341*/ offset:4352
	ds_load_b128 v[20:23], v85 /*v341*/ offset:4384
	ds_load_b128 v[24:27], v85 /*v341*/ offset:4416
	ds_load_b128 v[28:31], v85 /*v341*/ offset:4448
	ds_load_b128 v[32:35], v85 /*v341*/ offset:8704
	ds_load_b128 v[36:39], v85 /*v341*/ offset:8736
	ds_load_b128 v[40:43], v85 /*v341*/ offset:8768
	ds_load_b128 v[44:47], v85 /*v341*/ offset:8800
	ds_load_b128 v[48:51], v85 /*v341*/ offset:13056
	ds_load_b128 v[52:55], v85 /*v341*/ offset:13088
	ds_load_b128 v[56:59], v85 /*v341*/ offset:13120
	ds_load_b128 v[60:63], v85 /*v341*/ offset:13152
	s_set_vgpr_msb 0x100
	ds_load_2addr_b32 v[138:139], v64 offset1:1
	ds_load_2addr_b32 v[140:141], v64 offset0:2 offset1:3
	s_set_vgpr_msb 0x41
	ds_load_b128 v[24:27] /*v[280:283]*/, v64 /*v320*/ offset:1024
	ds_load_b128 v[28:31] /*v[284:287]*/, v64 /*v320*/ offset:1536
	ds_load_b128 v[32:35] /*v[288:291]*/, v64 /*v320*/ offset:3072
	ds_load_b128 v[36:39] /*v[292:295]*/, v64 /*v320*/ offset:3584
	ds_load_b128 v[40:43] /*v[296:299]*/, v64 /*v320*/ offset:5120
	ds_load_b128 v[44:47] /*v[300:303]*/, v64 /*v320*/ offset:5632
	ds_load_b128 v[48:51] /*v[304:307]*/, v64 /*v320*/ offset:7168
	ds_load_b128 v[52:55] /*v[308:311]*/, v64 /*v320*/ offset:7680
	s_set_vgpr_msb 0x4101
	ds_load_2addr_b32 v[150:151], v83 /*v339*/ offset0:32 offset1:48
	ds_load_b32 v147, v83 /*v339*/ offset:448
	s_set_vgpr_msb 0x100
	ds_load_2addr_b32 v[142:143], v64 offset0:4 offset1:5
	ds_load_2addr_b32 v[144:145], v64 offset0:6 offset1:7
	s_wait_dscnt 0xc
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[64:67], v85 /*v341*/ offset:128
	ds_load_b128 v[68:71], v85 /*v341*/ offset:160
	ds_load_b128 v[72:75], v85 /*v341*/ offset:192
	ds_load_b128 v[76:79], v85 /*v341*/ offset:224
	ds_load_b128 v[80:83], v85 /*v341*/ offset:4480
	ds_load_b128 v[84:87], v85 /*v341*/ offset:4512
	ds_load_b128 v[88:91], v85 /*v341*/ offset:4544
	ds_load_b128 v[92:95], v85 /*v341*/ offset:4576
	ds_load_b128 v[96:99], v85 /*v341*/ offset:8832
	ds_load_b128 v[100:103], v85 /*v341*/ offset:8864
	ds_load_b128 v[104:107], v85 /*v341*/ offset:8896
	ds_load_b128 v[108:111], v85 /*v341*/ offset:8928
	ds_load_b128 v[112:115], v85 /*v341*/ offset:13184
	ds_load_b128 v[116:119], v85 /*v341*/ offset:13216
	ds_load_b128 v[120:123], v85 /*v341*/ offset:13248
	ds_load_b128 v[124:127], v85 /*v341*/ offset:13280
	s_load_b64 s[78:79], s[0:1], 0x0 nv
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[56:63] /*v[312:319]*/, v[0:15], v[128:135], v148, v138 matrix_a_fmt:MATRIX_FMT_FP4
	s_mov_b32 s71, 0
	s_mov_b32 s72, 1
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[0:7] /*v[256:263]*/, v[0:15], v[66:73] /*v[322:329]*/, v149, v138 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[8:15] /*v[264:271]*/, v[0:15], v[74:81] /*v[330:337]*/, v136, v138 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[16:23] /*v[272:279]*/, v[0:15], v[152:159], v146, v138 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[16:23] /*v[272:279]*/, v[16:31], v[184:191], v146, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[8:15] /*v[264:271]*/, v[16:31], v[176:183], v136, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[0:7] /*v[256:263]*/, v[16:31], v[168:175], v149, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[56:63] /*v[312:319]*/, v[16:31], v[160:167], v148, v139 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[56:63] /*v[312:319]*/, v[32:47], v[192:199], v148, v140 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[0:7] /*v[256:263]*/, v[32:47], v[200:207], v149, v140 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[8:15] /*v[264:271]*/, v[32:47], v[208:215], v136, v140 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[16:23] /*v[272:279]*/, v[32:47], v[216:223], v146, v140 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[16:23] /*v[272:279]*/, v[48:63], v[248:255], v146, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[8:15] /*v[264:271]*/, v[48:63], v[240:247], v136, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[0:7] /*v[256:263]*/, v[48:63], v[232:239], v149, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[56:63] /*v[312:319]*/, v[48:63], v[224:231], v148, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[24:31] /*v[280:287]*/, v[64:79], v[128:135], v150, v142 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[32:39] /*v[288:295]*/, v[64:79], v[66:73] /*v[322:329]*/, v151, v142 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[40:47] /*v[296:303]*/, v[64:79], v[74:81] /*v[330:337]*/, v137, v142 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[48:55] /*v[304:311]*/, v[64:79], v[152:159], v147, v142 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[48:55] /*v[304:311]*/, v[80:95], v[184:191], v147, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[40:47] /*v[296:303]*/, v[80:95], v[176:183], v137, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[32:39] /*v[288:295]*/, v[80:95], v[168:175], v151, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[24:31] /*v[280:287]*/, v[80:95], v[160:167], v150, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[24:31] /*v[280:287]*/, v[96:111], v[192:199], v150, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[32:39] /*v[288:295]*/, v[96:111], v[200:207], v151, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[40:47] /*v[296:303]*/, v[96:111], v[208:215], v137, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[48:55] /*v[304:311]*/, v[96:111], v[216:223], v147, v144 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[48:55] /*v[304:311]*/, v[112:127], v[248:255], v147, v145 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[40:47] /*v[296:303]*/, v[112:127], v[240:247], v137, v145 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[32:39] /*v[288:295]*/, v[112:127], v[232:239], v151, v145 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[24:31] /*v[280:287]*/, v[112:127], v[224:231], v150, v145 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x104
	v_add_nc_u32_e32 v0, 0xfffe6400, v64 /*v320*/
	v_add_nc_u32_e32 v1, 0xfffe6600, v64 /*v320*/
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_add_nc_u32_e32 v2, 0xfffe6c00, v64 /*v320*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[48:51] /*v[304:307]*/, v0
	ds_load_b128 v[52:55] /*v[308:311]*/, v1
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v0, 0xfffe7400, v64 /*v320*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0xfffe7600, v64 /*v320*/
	v_add_nc_u32_e32 v3, 0xfffe6e00, v64 /*v320*/
	s_set_vgpr_msb 0x440
	ds_load_b128 v[56:59] /*v[312:315]*/, v2
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v2, 0xfffe7c00, v64 /*v320*/
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[0:3] /*v[256:259]*/, v0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v0, 0xfffe6400, v83 /*v339*/
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[4:7] /*v[260:263]*/, v1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v1, 0xfffe6440, v83 /*v339*/
	v_add_nc_u32_e32 v64, 0xfffe6800, v64 /*v320*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[60:63] /*v[316:319]*/, v3
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v3, 0xfffe7e00, v64 /*v320*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[8:11] /*v[264:267]*/, v2
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x4004
	ds_load_b32 v136, v0
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v0, 0xfffe6500, v83 /*v339*/
	v_add_nc_u32_e32 v2, 0xfffe6440, v86 /*v342*/
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v137, v1
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0xffff2804, v84 /*v340*/
	v_add_nc_u32_e32 v65, 0xfffe6a00, v64 /*v320*/
	s_wait_alu depctr_va_vdst(5)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[16:19] /*v[272:275]*/, v64
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v64, 0xfffe7000, v64 /*v320*/
	s_wait_alu depctr_va_vdst(5)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[12:15] /*v[268:271]*/, v3
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v3, 0xffff2800, v84 /*v340*/
	s_wait_alu depctr_va_vdst(5)
	ds_load_b32 v138, v0
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v139, v2
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v0, 0xffff2808, v84 /*v340*/
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v141, v1
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0xffff280c, v84 /*v340*/
	v_add_nc_u32_e32 v2, 0xfffe6400, v85 /*v341*/
	v_add_nc_u32_e32 v4, 0xfffe6420, v85 /*v341*/
	v_add_nc_u32_e32 v8, 0xfffe6440, v85 /*v341*/
	v_add_nc_u32_e32 v12, 0xfffe6460, v85 /*v341*/
	v_add_nc_u32_e32 v16, 0xfffe7500, v85 /*v341*/
	v_add_nc_u32_e32 v20, 0xfffe7520, v85 /*v341*/
	v_add_nc_u32_e32 v24, 0xfffe7540, v85 /*v341*/
	v_add_nc_u32_e32 v28, 0xfffe7560, v85 /*v341*/
	v_add_nc_u32_e32 v32, 0xfffe8600, v85 /*v341*/
	v_add_nc_u32_e32 v36, 0xfffe8620, v85 /*v341*/
	v_add_nc_u32_e32 v40, 0xfffe8640, v85 /*v341*/
	v_add_nc_u32_e32 v44, 0xfffe8660, v85 /*v341*/
	v_add_nc_u32_e32 v48, 0xfffe9700, v85 /*v341*/
	v_add_nc_u32_e32 v52, 0xfffe9720, v85 /*v341*/
	v_add_nc_u32_e32 v56, 0xfffe9740, v85 /*v341*/
	v_add_nc_u32_e32 v60, 0xfffe9760, v85 /*v341*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[20:23] /*v[276:279]*/, v65
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v65, 0xfffe7200, v64 /*v320*/
	s_set_vgpr_msb 0x440
	ds_load_b128 v[24:27] /*v[280:283]*/, v64
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v64, 0xfffe8000, v64 /*v320*/
	ds_load_b32 v140, v3
	ds_load_b32 v142, v0
	ds_load_b32 v143, v1
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[0:3], v2
	ds_load_b128 v[4:7], v4
	s_wait_alu depctr_va_vdst(14)
	ds_load_b128 v[8:11], v8
	ds_load_b128 v[12:15], v12
	s_wait_alu depctr_va_vdst(13)
	ds_load_b128 v[16:19], v16
	s_wait_alu depctr_va_vdst(12)
	ds_load_b128 v[20:23], v20
	s_wait_alu depctr_va_vdst(11)
	ds_load_b128 v[24:27], v24
	s_wait_alu depctr_va_vdst(10)
	ds_load_b128 v[28:31], v28
	s_wait_alu depctr_va_vdst(9)
	ds_load_b128 v[32:35], v32
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[36:39], v36
	s_wait_alu depctr_va_vdst(7)
	ds_load_b128 v[40:43], v40
	s_wait_alu depctr_va_vdst(6)
	ds_load_b128 v[44:47], v44
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[48:51], v48
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[52:55], v52
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[56:59], v56
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[60:63], v60
	v_add_nc_u32_e32 v66, 0xfffe7800, v64 /*v320*/
	v_add_nc_u32_e32 v67, 0xfffe7a00, v64 /*v320*/
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[28:31] /*v[284:287]*/, v65
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v65, 0xfffe8200, v64 /*v320*/
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[40:43] /*v[296:299]*/, v64
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v64, 0xfffe6480, v83 /*v339*/
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[32:35] /*v[288:291]*/, v66
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[36:39] /*v[292:295]*/, v67
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v66, 0xfffe64c0, v83 /*v339*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v67, 0xfffe6580, v83 /*v339*/
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x440
	ds_load_b128 v[44:47] /*v[300:303]*/, v65
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v65, 0xfffe65c0, v83 /*v339*/
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v144, v64
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v64, 0xffff2810, v84 /*v340*/
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v145, v66
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v146, v67
	s_wait_alu depctr_va_vdst(1)
	ds_load_b32 v147, v65
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v65, 0xffff2814, v84 /*v340*/
	v_add_nc_u32_e32 v66, 0xffff2818, v84 /*v340*/
	v_add_nc_u32_e32 v67, 0xffff281c, v84 /*v340*/
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v148, v64
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v64, 0xfffe6480, v85 /*v341*/
	v_add_nc_u32_e32 v68, 0xfffe64a0, v85 /*v341*/
	v_add_nc_u32_e32 v72, 0xfffe64c0, v85 /*v341*/
	v_add_nc_u32_e32 v76, 0xfffe64e0, v85 /*v341*/
	v_add_nc_u32_e32 v80, 0xfffe7580, v85 /*v341*/
	v_add_nc_u32_e32 v84, 0xfffe75a0, v85 /*v341*/
	v_add_nc_u32_e32 v88, 0xfffe75c0, v85 /*v341*/
	v_add_nc_u32_e32 v92, 0xfffe75e0, v85 /*v341*/
	v_add_nc_u32_e32 v96, 0xfffe8680, v85 /*v341*/
	v_add_nc_u32_e32 v100, 0xfffe86a0, v85 /*v341*/
	v_add_nc_u32_e32 v104, 0xfffe86c0, v85 /*v341*/
	v_add_nc_u32_e32 v108, 0xfffe86e0, v85 /*v341*/
	v_add_nc_u32_e32 v112, 0xfffe9780, v85 /*v341*/
	v_add_nc_u32_e32 v116, 0xfffe97a0, v85 /*v341*/
	v_add_nc_u32_e32 v120, 0xfffe97c0, v85 /*v341*/
	v_add_nc_u32_e32 v124, 0xfffe97e0, v85 /*v341*/
	s_wait_alu depctr_va_vdst(14)
	ds_load_b32 v149, v65
	ds_load_b32 v150, v66
	ds_load_b32 v151, v67
	s_wait_dscnt 0xc
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[64:67], v64
	ds_load_b128 v[68:71], v68
	s_wait_alu depctr_va_vdst(13)
	ds_load_b128 v[72:75], v72
	s_wait_alu depctr_va_vdst(12)
	ds_load_b128 v[76:79], v76
	s_wait_alu depctr_va_vdst(11)
	ds_load_b128 v[80:83], v80
	s_wait_alu depctr_va_vdst(10)
	ds_load_b128 v[84:87], v84
	s_wait_alu depctr_va_vdst(9)
	ds_load_b128 v[88:91], v88
	s_wait_alu depctr_va_vdst(8)
	ds_load_b128 v[92:95], v92
	s_wait_alu depctr_va_vdst(7)
	ds_load_b128 v[96:99], v96
	s_wait_alu depctr_va_vdst(6)
	ds_load_b128 v[100:103], v100
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[104:107], v104
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[108:111], v108
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[112:115], v112
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[116:119], v116
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[120:123], v120
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[124:127], v124
	s_set_vgpr_msb 0x401
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[48:55] /*v[304:311]*/, v[0:15], v[128:135], v136, v140 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[56:63] /*v[312:319]*/, v[0:15], v[66:73] /*v[322:329]*/, v137, v140 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[0:7] /*v[256:263]*/, v[0:15], v[74:81] /*v[330:337]*/, v138, v140 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[8:15] /*v[264:271]*/, v[0:15], v[152:159], v139, v140 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[8:15] /*v[264:271]*/, v[16:31], v[184:191], v139, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[0:7] /*v[256:263]*/, v[16:31], v[176:183], v138, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[56:63] /*v[312:319]*/, v[16:31], v[168:175], v137, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[48:55] /*v[304:311]*/, v[16:31], v[160:167], v136, v141 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[48:55] /*v[304:311]*/, v[32:47], v[192:199], v136, v142 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[56:63] /*v[312:319]*/, v[32:47], v[200:207], v137, v142 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[0:7] /*v[256:263]*/, v[32:47], v[208:215], v138, v142 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[8:15] /*v[264:271]*/, v[32:47], v[216:223], v139, v142 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[8:15] /*v[264:271]*/, v[48:63], v[248:255], v139, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[0:7] /*v[256:263]*/, v[48:63], v[240:247], v138, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[56:63] /*v[312:319]*/, v[48:63], v[232:239], v137, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[48:55] /*v[304:311]*/, v[48:63], v[224:231], v136, v143 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[16:23] /*v[272:279]*/, v[64:79], v[128:135], v144, v148 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[24:31] /*v[280:287]*/, v[64:79], v[66:73] /*v[322:329]*/, v145, v148 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[32:39] /*v[288:295]*/, v[64:79], v[74:81] /*v[330:337]*/, v146, v148 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[40:47] /*v[296:303]*/, v[64:79], v[152:159], v147, v148 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[40:47] /*v[296:303]*/, v[80:95], v[184:191], v147, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[32:39] /*v[288:295]*/, v[80:95], v[176:183], v146, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[24:31] /*v[280:287]*/, v[80:95], v[168:175], v145, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[16:23] /*v[272:279]*/, v[80:95], v[160:167], v144, v149 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[16:23] /*v[272:279]*/, v[96:111], v[192:199], v144, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[24:31] /*v[280:287]*/, v[96:111], v[200:207], v145, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[32:39] /*v[288:295]*/, v[96:111], v[208:215], v146, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[40:47] /*v[296:303]*/, v[96:111], v[216:223], v147, v150 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[40:47] /*v[296:303]*/, v[112:127], v[248:255], v147, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[32:39] /*v[288:295]*/, v[112:127], v[240:247], v146, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[24:31] /*v[280:287]*/, v[112:127], v[232:239], v145, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[16:23] /*v[272:279]*/, v[112:127], v[224:231], v144, v151 matrix_a_fmt:MATRIX_FMT_FP4
	v_lshl_or_b32 v18, v82 /*v338*/, 3, s35
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_u32 v0, s34, s73, v18
	s_wait_xcnt 0x0
	s_lshr_b32 s1, s73, 31
	s_sub_f32 s0, 0, s74
	s_add_co_i32 s1, s73, s1
	s_mov_b32 s65, 0xffff0000
	s_ashr_i32 s69, s1, 1
	s_and_b32 s1, s1, -2
	v_ashrrev_i32_e32 v1, 31, v0
	s_cmp_lg_u32 s73, s1
	s_mov_b32 s68, 16
	s_cselect_b32 s1, -1, 0
	s_bfe_u32 s2, ttmp8, 0x50019
	s_set_vgpr_msb 0x100
	v_lshl_add_u64 v[16:17], v[0:1], 1, s[54:55]
	s_and_b32 s1, s75, s1
	s_and_b32 s2, s2, 3
	s_mov_b32 s64, 0x10000
	s_lshl_b32 s3, s2, 12
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x3
	global_load_b128 v[0:3], v[16:17], off
	global_load_b128 v[4:7], v[16:17], off offset:32
	global_load_b128 v[8:11], v[16:17], off offset:64
	global_load_b128 v[12:15], v[16:17], off offset:96
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 1
	v_lshl_or_b32 v16, v65 /*v321*/, 8, v18
	s_lshl_b32 s70, s2, 5
	s_add_co_i32 s73, s3, 0
	s_set_vgpr_msb 0x140
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v0 /*v256*/, 0, v16
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e64 v16, 0, 1, s1
	s_lshl_b32 s1, s2, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s1, s33, s1
	s_set_vgpr_msb 0x44
	v_add_nc_u32_e32 v1 /*v257*/, 0x1000, v0 /*v256*/
	v_readfirstlane_b32 s80, v16
	s_max_i32 s75, s1, 0
	v_add_nc_u32_e32 v2 /*v258*/, 0x2000, v0 /*v256*/
	v_add_nc_u32_e32 v3 /*v259*/, 0x3000, v0 /*v256*/
	s_set_vgpr_msb 0x4400
	s_wait_loadcnt 0x3
	v_and_b32_e32 v19, 0xffff0000, v1
	v_lshlrev_b32_e32 v18, 16, v1
	v_and_b32_e32 v17, 0xffff0000, v0
	v_lshlrev_b32_e32 v16, 16, v0
	v_and_b32_e32 v1, 0xffff0000, v3
	v_lshlrev_b32_e32 v0, 16, v3
	v_pk_add_f32 v[36:37], v[130:131], v[18:19]
	v_and_b32_e32 v3, 0xffff0000, v2
	v_pk_add_f32 v[38:39], v[128:129], v[16:17]
	v_lshlrev_b32_e32 v2, 16, v2
	v_pk_add_f32 v[34:35], v[134:135], v[0:1]
	v_cmp_gt_f32_e32 vcc_lo, s74, v36
	s_wait_loadcnt 0x2
	v_and_b32_e32 v23, 0xffff0000, v5
	v_lshlrev_b32_e32 v22, 16, v5
	v_pk_add_f32 v[32:33], v[132:133], v[2:3]
	v_and_b32_e32 v21, 0xffff0000, v4
	v_cndmask_b32_e32 v129, s74, v36, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v38
	v_lshlrev_b32_e32 v20, 16, v4
	s_set_vgpr_msb 1
	v_pk_add_f32 v[44:45], v[68:69] /*v[324:325]*/, v[22:23]
	v_and_b32_e32 v5, 0xffff0000, v7
	v_dual_lshlrev_b32 v4, 16, v7 :: v_dual_cndmask_b32 v128, s74, v38, vcc_lo
	v_cmp_lt_f32_e64 vcc_lo, -s74, v37
	v_pk_add_f32 v[46:47], v[66:67] /*v[322:323]*/, v[20:21]
	v_and_b32_e32 v7, 0xffff0000, v6
	v_lshlrev_b32_e32 v6, 16, v6
	v_pk_add_f32 v[42:43], v[72:73] /*v[328:329]*/, v[4:5]
	v_cndmask_b32_e32 v130, s0, v37, vcc_lo
	v_cmp_lt_f32_e64 vcc_lo, -s74, v39
	s_wait_loadcnt 0x1
	v_and_b32_e32 v27, 0xffff0000, v9
	v_pk_add_f32 v[40:41], v[70:71] /*v[326:327]*/, v[6:7]
	v_lshlrev_b32_e32 v26, 16, v9
	v_and_b32_e32 v25, 0xffff0000, v8
	v_cndmask_b32_e32 v131, s0, v39, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v34
	v_lshlrev_b32_e32 v24, 16, v8
	v_pk_add_f32 v[52:53], v[76:77] /*v[332:333]*/, v[26:27]
	v_and_b32_e32 v9, 0xffff0000, v11
	v_dual_cndmask_b32 v37, s74, v34 :: v_dual_lshlrev_b32 v8, 16, v11
	v_cmp_gt_f32_e32 vcc_lo, s74, v32
	v_pk_add_f32 v[54:55], v[74:75] /*v[330:331]*/, v[24:25]
	v_and_b32_e32 v11, 0xffff0000, v10
	v_lshlrev_b32_e32 v10, 16, v10
	v_pk_add_f32 v[50:51], v[80:81] /*v[336:337]*/, v[8:9]
	v_cndmask_b32_e32 v36, s74, v32, vcc_lo
	v_cmp_lt_f32_e64 vcc_lo, -s74, v35
	s_wait_loadcnt 0x0
	v_and_b32_e32 v31, 0xffff0000, v13
	v_pk_add_f32 v[48:49], v[78:79] /*v[334:335]*/, v[10:11]
	v_lshlrev_b32_e32 v30, 16, v13
	v_and_b32_e32 v29, 0xffff0000, v12
	v_cndmask_b32_e32 v132, s0, v35, vcc_lo
	v_cmp_lt_f32_e64 vcc_lo, -s74, v33
	v_lshlrev_b32_e32 v28, 16, v12
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[60:61], v[154:155], v[30:31]
	v_and_b32_e32 v13, 0xffff0000, v15
	v_dual_cndmask_b32 v133, s0, v33 :: v_dual_lshlrev_b32 v12, 16, v15
	v_cmp_gt_f32_e32 vcc_lo, s74, v44
	v_pk_add_f32 v[62:63], v[152:153], v[28:29]
	v_and_b32_e32 v15, 0xffff0000, v14
	v_lshlrev_b32_e32 v14, 16, v14
	v_pk_add_f32 v[58:59], v[158:159], v[12:13]
	v_cndmask_b32_e32 v33, s74, v44, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v46
	v_pk_add_f32 v[68:69], v[162:163], v[18:19]
	v_pk_add_f32 v[56:57], v[156:157], v[14:15]
	v_pk_add_f32 v[70:71], v[160:161], v[16:17]
	v_pk_add_f32 v[66:67], v[166:167], v[0:1]
	v_cndmask_b32_e32 v32, s74, v46, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v45, -s74
	v_pk_add_f32 v[64:65], v[164:165], v[2:3]
	v_pk_add_f32 v[76:77], v[170:171], v[22:23]
	v_pk_add_f32 v[78:79], v[168:169], v[20:21]
	v_pk_add_f32 v[74:75], v[174:175], v[4:5]
	v_cndmask_b32_e32 v134, s0, v45, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v47, -s74
	v_pk_add_f32 v[72:73], v[172:173], v[6:7]
	v_pk_add_f32 v[84:85], v[178:179], v[26:27]
	v_pk_add_f32 v[86:87], v[176:177], v[24:25]
	v_pk_add_f32 v[82:83], v[182:183], v[8:9]
	v_cndmask_b32_e32 v135, s0, v47, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v42
	v_pk_add_f32 v[80:81], v[180:181], v[10:11]
	v_pk_add_f32 v[92:93], v[186:187], v[30:31]
	v_pk_add_f32 v[94:95], v[184:185], v[28:29]
	v_pk_add_f32 v[90:91], v[190:191], v[12:13]
	v_cndmask_b32_e32 v35, s74, v42, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v40
	v_pk_add_f32 v[88:89], v[188:189], v[14:15]
	v_pk_add_f32 v[100:101], v[194:195], v[18:19]
	v_pk_add_f32 v[102:103], v[192:193], v[16:17]
	v_pk_add_f32 v[98:99], v[198:199], v[0:1]
	v_cndmask_b32_e32 v34, s74, v40, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v43, -s74
	v_pk_add_f32 v[96:97], v[196:197], v[2:3]
	v_pk_add_f32 v[108:109], v[202:203], v[22:23]
	v_pk_add_f32 v[110:111], v[200:201], v[20:21]
	v_pk_add_f32 v[106:107], v[206:207], v[4:5]
	v_cndmask_b32_e32 v136, s0, v43, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v41, -s74
	v_pk_add_f32 v[104:105], v[204:205], v[6:7]
	v_pk_add_f32 v[116:117], v[210:211], v[26:27]
	v_pk_add_f32 v[118:119], v[208:209], v[24:25]
	v_pk_add_f32 v[114:115], v[214:215], v[8:9]
	v_cndmask_b32_e32 v137, s0, v41, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v52
	v_pk_add_f32 v[112:113], v[212:213], v[10:11]
	v_pk_add_f32 v[124:125], v[218:219], v[30:31]
	v_pk_add_f32 v[126:127], v[216:217], v[28:29]
	v_pk_add_f32 v[122:123], v[222:223], v[12:13]
	v_cndmask_b32_e32 v39, s74, v52, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v54
	v_pk_add_f32 v[120:121], v[220:221], v[14:15]
	v_pk_add_f32 v[18:19], v[226:227], v[18:19]
	v_pk_add_f32 v[16:17], v[224:225], v[16:17]
	v_pk_add_f32 v[0:1], v[230:231], v[0:1]
	v_cndmask_b32_e32 v38, s74, v54, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v53, -s74
	v_pk_add_f32 v[2:3], v[228:229], v[2:3]
	v_pk_add_f32 v[22:23], v[234:235], v[22:23]
	v_pk_add_f32 v[20:21], v[232:233], v[20:21]
	v_pk_add_f32 v[4:5], v[238:239], v[4:5]
	v_cndmask_b32_e32 v138, s0, v53, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v55, -s74
	v_pk_add_f32 v[6:7], v[236:237], v[6:7]
	v_pk_add_f32 v[26:27], v[242:243], v[26:27]
	v_pk_add_f32 v[24:25], v[240:241], v[24:25]
	v_pk_add_f32 v[8:9], v[246:247], v[8:9]
	v_cndmask_b32_e32 v139, s0, v55, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v50
	v_pk_add_f32 v[10:11], v[244:245], v[10:11]
	v_pk_add_f32 v[30:31], v[250:251], v[30:31]
	v_pk_add_f32 v[28:29], v[248:249], v[28:29]
	v_pk_add_f32 v[12:13], v[254:255], v[12:13]
	v_cndmask_b32_e32 v41, s74, v50, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v48
	v_pk_add_f32 v[14:15], v[252:253], v[14:15]
	v_dual_mul_f32 v158, 0xbfb8aa3b, v128 :: v_dual_mul_f32 v159, 0xbfb8aa3b, v129
	v_mul_f32_e32 v160, 0xbfb8aa3b, v36
	v_cndmask_b32_e32 v40, s74, v48, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v51, -s74
	v_mul_f32_e32 v161, 0xbfb8aa3b, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_cmp_gt_f32_e64 s1, 0xc2fc0000, v160
	v_cndmask_b32_e32 v140, s0, v51, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v49, -s74
	v_cmp_gt_f32_e64 s2, 0xc2fc0000, v161
	v_cndmask_b32_e32 v141, s0, v49, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v60
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v195, 0, 0x42800000, s2
	v_cndmask_b32_e32 v43, s74, v60, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v62
	v_add_f32_e32 v161, v161, v195
	v_cndmask_b32_e64 v195, 0, 0xffffffc0, s2
	v_cndmask_b32_e32 v42, s74, v62, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v61, -s74
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v161, v161
	v_cndmask_b32_e32 v142, s0, v61, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v63, -s74
	v_cndmask_b32_e32 v143, s0, v63, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v58
	v_cndmask_b32_e32 v45, s74, v58, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v56
	v_cndmask_b32_e32 v44, s74, v56, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v59, -s74
	v_cndmask_b32_e32 v144, s0, v59, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v57, -s74
	v_cndmask_b32_e32 v145, s0, v57, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v68
	v_cndmask_b32_e32 v47, s74, v68, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v70
	v_cndmask_b32_e32 v46, s74, v70, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v69, -s74
	v_cndmask_b32_e32 v146, s0, v69, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v71, -s74
	v_cndmask_b32_e32 v147, s0, v71, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v66
	v_cndmask_b32_e32 v49, s74, v66, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v64
	v_cndmask_b32_e32 v48, s74, v64, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v67, -s74
	v_cndmask_b32_e32 v148, s0, v67, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v65, -s74
	v_cndmask_b32_e32 v149, s0, v65, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v76
	v_cndmask_b32_e32 v51, s74, v76, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v78
	v_cndmask_b32_e32 v50, s74, v78, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v77, -s74
	v_cndmask_b32_e32 v150, s0, v77, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v79, -s74
	v_cndmask_b32_e32 v151, s0, v79, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v74
	v_cndmask_b32_e32 v53, s74, v74, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v72
	v_cndmask_b32_e32 v52, s74, v72, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v75, -s74
	v_cndmask_b32_e32 v152, s0, v75, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v73, -s74
	v_cndmask_b32_e32 v153, s0, v73, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v84
	v_cndmask_b32_e32 v55, s74, v84, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v86
	v_cndmask_b32_e32 v54, s74, v86, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v85, -s74
	v_cndmask_b32_e32 v86, s0, v85, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v87, -s74
	v_cndmask_b32_e32 v154, s0, v87, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v82
	v_cndmask_b32_e32 v57, s74, v82, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v80
	v_cndmask_b32_e32 v56, s74, v80, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v83, -s74
	v_cndmask_b32_e32 v155, s0, v83, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v81, -s74
	v_cndmask_b32_e32 v156, s0, v81, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v92
	v_cndmask_b32_e32 v59, s74, v92, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v94
	v_cndmask_b32_e32 v58, s74, v94, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v93, -s74
	v_cndmask_b32_e32 v92, s0, v93, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v95, -s74
	v_cndmask_b32_e32 v94, s0, v95, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v90
	v_cndmask_b32_e32 v61, s74, v90, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v88
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v162, 0xbfb8aa3b, v61
	v_cndmask_b32_e32 v60, s74, v88, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v91, -s74
	v_cmp_gt_f32_e64 s30, 0xc2fc0000, v162
	v_cndmask_b32_e32 v90, s0, v91, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v89, -s74
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v223, 0, 0x42800000, s30
	v_cndmask_b32_e32 v95, s0, v89, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v100
	v_add_f32_e32 v162, v162, v223
	v_cndmask_b32_e64 v223, 0, 0xffffffc0, s30
	v_cndmask_b32_e32 v63, s74, v100, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v102
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v162, v162
	v_mul_f32_e32 v164, 0xbfb8aa3b, v63
	v_cndmask_b32_e32 v62, s74, v102, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v101, -s74
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s33, 0xc2fc0000, v164
	v_mul_f32_e32 v163, 0xbfb8aa3b, v62
	v_cndmask_b32_e32 v100, s0, v101, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v103, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v225, 0, 0x42800000, s33
	v_cmp_gt_f32_e64 s31, 0xc2fc0000, v163
	v_cndmask_b32_e32 v101, s0, v103, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v164, v164, v225
	v_cndmask_b32_e64 v224, 0, 0x42800000, s31
	v_cndmask_b32_e64 v225, 0, 0xffffffc0, s33
	v_cndmask_b32_e32 v65, s74, v98, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v163, v163, v224
	v_exp_f32_e32 v164, v164
	v_cndmask_b32_e64 v224, 0, 0xffffffc0, s31
	v_mul_f32_e32 v166, 0xbfb8aa3b, v65
	v_cndmask_b32_e32 v64, s74, v96, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v99, -s74
	v_exp_f32_e32 v163, v163
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s35, 0xc2fc0000, v166
	v_mul_f32_e32 v165, 0xbfb8aa3b, v64
	v_cndmask_b32_e32 v96, s0, v99, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v97, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v227, 0, 0x42800000, s35
	v_cmp_gt_f32_e64 s34, 0xc2fc0000, v165
	v_cndmask_b32_e32 v98, s0, v97, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v108
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v166, v166, v227
	v_cndmask_b32_e64 v226, 0, 0x42800000, s34
	v_cndmask_b32_e64 v227, 0, 0xffffffc0, s35
	v_cndmask_b32_e32 v67, s74, v108, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v110
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v165, v165, v226
	v_exp_f32_e32 v166, v166
	v_cndmask_b32_e64 v226, 0, 0xffffffc0, s34
	v_mul_f32_e32 v168, 0xbfb8aa3b, v67
	v_cndmask_b32_e32 v66, s74, v110, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v109, -s74
	v_exp_f32_e32 v165, v165
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s37, 0xc2fc0000, v168
	v_mul_f32_e32 v167, 0xbfb8aa3b, v66
	v_cndmask_b32_e32 v99, s0, v109, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v111, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v229, 0, 0x42800000, s37
	v_cmp_gt_f32_e64 s36, 0xc2fc0000, v167
	v_cndmask_b32_e32 v102, s0, v111, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v106
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v168, v168, v229
	v_cndmask_b32_e64 v228, 0, 0x42800000, s36
	v_cndmask_b32_e64 v229, 0, 0xffffffc0, s37
	v_cndmask_b32_e32 v69, s74, v106, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v167, v167, v228
	v_exp_f32_e32 v168, v168
	v_cndmask_b32_e64 v228, 0, 0xffffffc0, s36
	v_mul_f32_e32 v170, 0xbfb8aa3b, v69
	v_cndmask_b32_e32 v68, s74, v104, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v107, -s74
	v_exp_f32_e32 v167, v167
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s39, 0xc2fc0000, v170
	v_mul_f32_e32 v169, 0xbfb8aa3b, v68
	v_cndmask_b32_e32 v103, s0, v107, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v105, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v231, 0, 0x42800000, s39
	v_cmp_gt_f32_e64 s38, 0xc2fc0000, v169
	v_cndmask_b32_e32 v104, s0, v105, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v116
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v170, v170, v231
	v_cndmask_b32_e64 v230, 0, 0x42800000, s38
	v_cndmask_b32_e64 v231, 0, 0xffffffc0, s39
	v_cndmask_b32_e32 v71, s74, v116, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v169, v169, v230
	v_exp_f32_e32 v170, v170
	v_cndmask_b32_e64 v230, 0, 0xffffffc0, s38
	v_mul_f32_e32 v172, 0xbfb8aa3b, v71
	v_cndmask_b32_e32 v70, s74, v118, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v117, -s74
	v_exp_f32_e32 v169, v169
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s41, 0xc2fc0000, v172
	v_mul_f32_e32 v171, 0xbfb8aa3b, v70
	v_cndmask_b32_e32 v105, s0, v117, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v119, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v233, 0, 0x42800000, s41
	v_cmp_gt_f32_e64 s40, 0xc2fc0000, v171
	v_cndmask_b32_e32 v106, s0, v119, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v114
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v172, v172, v233
	v_cndmask_b32_e64 v232, 0, 0x42800000, s40
	v_cndmask_b32_e64 v233, 0, 0xffffffc0, s41
	v_cndmask_b32_e32 v73, s74, v114, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v171, v171, v232
	v_exp_f32_e32 v172, v172
	v_cndmask_b32_e64 v232, 0, 0xffffffc0, s40
	v_mul_f32_e32 v174, 0xbfb8aa3b, v73
	v_cndmask_b32_e32 v72, s74, v112, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v115, -s74
	v_exp_f32_e32 v171, v171
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s43, 0xc2fc0000, v174
	v_mul_f32_e32 v173, 0xbfb8aa3b, v72
	v_cndmask_b32_e32 v107, s0, v115, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v113, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v235, 0, 0x42800000, s43
	v_cmp_gt_f32_e64 s42, 0xc2fc0000, v173
	v_cndmask_b32_e32 v108, s0, v113, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v124
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v174, v174, v235
	v_cndmask_b32_e64 v234, 0, 0x42800000, s42
	v_cndmask_b32_e64 v235, 0, 0xffffffc0, s43
	v_cndmask_b32_e32 v75, s74, v124, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v173, v173, v234
	v_exp_f32_e32 v174, v174
	v_cndmask_b32_e64 v234, 0, 0xffffffc0, s42
	v_mul_f32_e32 v176, 0xbfb8aa3b, v75
	v_cndmask_b32_e32 v74, s74, v126, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v125, -s74
	v_exp_f32_e32 v173, v173
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s45, 0xc2fc0000, v176
	v_mul_f32_e32 v175, 0xbfb8aa3b, v74
	v_cndmask_b32_e32 v109, s0, v125, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v127, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v237, 0, 0x42800000, s45
	v_cmp_gt_f32_e64 s44, 0xc2fc0000, v175
	v_cndmask_b32_e32 v110, s0, v127, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v122
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v176, v176, v237
	v_cndmask_b32_e64 v236, 0, 0x42800000, s44
	v_cndmask_b32_e64 v237, 0, 0xffffffc0, s45
	v_cndmask_b32_e32 v77, s74, v122, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v175, v175, v236
	v_exp_f32_e32 v176, v176
	v_cndmask_b32_e64 v236, 0, 0xffffffc0, s44
	v_mul_f32_e32 v178, 0xbfb8aa3b, v77
	v_cndmask_b32_e32 v76, s74, v120, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v123, -s74
	v_exp_f32_e32 v175, v175
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s47, 0xc2fc0000, v178
	v_mul_f32_e32 v177, 0xbfb8aa3b, v76
	v_cndmask_b32_e32 v111, s0, v123, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v121, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v239, 0, 0x42800000, s47
	v_cmp_gt_f32_e64 s46, 0xc2fc0000, v177
	v_cndmask_b32_e32 v112, s0, v121, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v178, v178, v239
	v_cndmask_b32_e64 v238, 0, 0x42800000, s46
	v_cndmask_b32_e64 v239, 0, 0xffffffc0, s47
	v_cndmask_b32_e32 v79, s74, v18, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v177, v177, v238
	v_exp_f32_e32 v178, v178
	v_cndmask_b32_e64 v238, 0, 0xffffffc0, s46
	v_mul_f32_e32 v180, 0xbfb8aa3b, v79
	v_cndmask_b32_e32 v78, s74, v16, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v19, -s74
	v_exp_f32_e32 v177, v177
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s49, 0xc2fc0000, v180
	v_mul_f32_e32 v179, 0xbfb8aa3b, v78
	v_cndmask_b32_e32 v113, s0, v19, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v17, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v241, 0, 0x42800000, s49
	v_cmp_gt_f32_e64 s48, 0xc2fc0000, v179
	v_cndmask_b32_e32 v114, s0, v17, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v180, v180, v241
	v_cndmask_b32_e64 v240, 0, 0x42800000, s48
	v_cndmask_b32_e64 v241, 0, 0xffffffc0, s49
	v_cndmask_b32_e32 v17, s74, v0, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v179, v179, v240
	v_exp_f32_e32 v180, v180
	v_cndmask_b32_e64 v240, 0, 0xffffffc0, s48
	v_mul_f32_e32 v182, 0xbfb8aa3b, v17
	v_cndmask_b32_e32 v16, s74, v2, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v1, -s74
	v_exp_f32_e32 v179, v179
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s51, 0xc2fc0000, v182
	v_mul_f32_e32 v181, 0xbfb8aa3b, v16
	v_cndmask_b32_e32 v115, s0, v1, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v3, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v243, 0, 0x42800000, s51
	v_cmp_gt_f32_e64 s50, 0xc2fc0000, v181
	v_cndmask_b32_e32 v116, s0, v3, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v182, v182, v243
	v_cndmask_b32_e64 v242, 0, 0x42800000, s50
	v_cndmask_b32_e64 v243, 0, 0xffffffc0, s51
	v_cndmask_b32_e32 v1, s74, v22, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v181, v181, v242
	v_exp_f32_e32 v182, v182
	v_cndmask_b32_e64 v242, 0, 0xffffffc0, s50
	v_mul_f32_e32 v184, 0xbfb8aa3b, v1
	v_cndmask_b32_e32 v0, s74, v20, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v23, -s74
	v_exp_f32_e32 v181, v181
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s53, 0xc2fc0000, v184
	v_mul_f32_e32 v183, 0xbfb8aa3b, v0
	v_cndmask_b32_e32 v117, s0, v23, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v21, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v245, 0, 0x42800000, s53
	v_cmp_gt_f32_e64 s52, 0xc2fc0000, v183
	v_cndmask_b32_e32 v118, s0, v21, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v184, v184, v245
	v_cndmask_b32_e64 v244, 0, 0x42800000, s52
	v_cndmask_b32_e64 v245, 0, 0xffffffc0, s53
	v_cndmask_b32_e32 v3, s74, v4, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v183, v183, v244
	v_exp_f32_e32 v184, v184
	v_cndmask_b32_e64 v244, 0, 0xffffffc0, s52
	v_mul_f32_e32 v186, 0xbfb8aa3b, v3
	v_cndmask_b32_e32 v2, s74, v6, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v5, -s74
	v_exp_f32_e32 v183, v183
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s55, 0xc2fc0000, v186
	v_mul_f32_e32 v185, 0xbfb8aa3b, v2
	v_cndmask_b32_e32 v119, s0, v5, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v7, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v247, 0, 0x42800000, s55
	v_cmp_gt_f32_e64 s54, 0xc2fc0000, v185
	v_cndmask_b32_e32 v120, s0, v7, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v186, v186, v247
	v_cndmask_b32_e64 v246, 0, 0x42800000, s54
	v_cndmask_b32_e64 v247, 0, 0xffffffc0, s55
	v_cndmask_b32_e32 v5, s74, v26, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v185, v185, v246
	v_exp_f32_e32 v186, v186
	v_cndmask_b32_e64 v246, 0, 0xffffffc0, s54
	v_mul_f32_e32 v188, 0xbfb8aa3b, v5
	v_cndmask_b32_e32 v4, s74, v24, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v27, -s74
	v_exp_f32_e32 v185, v185
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s57, 0xc2fc0000, v188
	v_mul_f32_e32 v187, 0xbfb8aa3b, v4
	v_cndmask_b32_e32 v121, s0, v27, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v25, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v249, 0, 0x42800000, s57
	v_cmp_gt_f32_e64 s56, 0xc2fc0000, v187
	v_cndmask_b32_e32 v122, s0, v25, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v188, v188, v249
	v_cndmask_b32_e64 v248, 0, 0x42800000, s56
	v_cndmask_b32_e64 v249, 0, 0xffffffc0, s57
	v_cndmask_b32_e32 v7, s74, v8, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v187, v187, v248
	v_exp_f32_e32 v188, v188
	v_cndmask_b32_e64 v248, 0, 0xffffffc0, s56
	v_mul_f32_e32 v190, 0xbfb8aa3b, v7
	v_cndmask_b32_e32 v6, s74, v10, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v9, -s74
	v_exp_f32_e32 v187, v187
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s59, 0xc2fc0000, v190
	v_mul_f32_e32 v189, 0xbfb8aa3b, v6
	v_cndmask_b32_e32 v123, s0, v9, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v11, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v251, 0, 0x42800000, s59
	v_cmp_gt_f32_e64 s58, 0xc2fc0000, v189
	v_cndmask_b32_e32 v124, s0, v11, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v190, v190, v251
	v_cndmask_b32_e64 v250, 0, 0x42800000, s58
	v_cndmask_b32_e64 v251, 0, 0xffffffc0, s59
	v_cndmask_b32_e32 v9, s74, v30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v189, v189, v250
	v_exp_f32_e32 v190, v190
	v_cndmask_b32_e64 v250, 0, 0xffffffc0, s58
	v_mul_f32_e32 v192, 0xbfb8aa3b, v9
	v_cndmask_b32_e32 v8, s74, v28, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v31, -s74
	v_exp_f32_e32 v189, v189
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s61, 0xc2fc0000, v192
	v_mul_f32_e32 v191, 0xbfb8aa3b, v8
	v_cndmask_b32_e32 v125, s0, v31, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v29, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v253, 0, 0x42800000, s61
	v_cmp_gt_f32_e64 s60, 0xc2fc0000, v191
	v_cndmask_b32_e32 v126, s0, v29, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v192, v192, v253
	v_cndmask_b32_e64 v252, 0, 0x42800000, s60
	v_cndmask_b32_e64 v253, 0, 0xffffffc0, s61
	v_cndmask_b32_e32 v11, s74, v12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v191, v191, v252
	v_exp_f32_e32 v192, v192
	v_cndmask_b32_e64 v252, 0, 0xffffffc0, s60
	v_mul_f32_e32 v194, 0xbfb8aa3b, v11
	v_cndmask_b32_e32 v10, s74, v14, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v13, -s74
	v_exp_f32_e32 v191, v191
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s63, 0xc2fc0000, v194
	v_mul_f32_e32 v193, 0xbfb8aa3b, v10
	v_cndmask_b32_e32 v127, s0, v13, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v15, -s74
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v255, 0, 0x42800000, s63
	v_cmp_gt_f32_e64 s62, 0xc2fc0000, v193
	v_cndmask_b32_e32 v157, s0, v15, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v130
	v_cmp_gt_f32_e64 s0, 0xc2fc0000, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v254, 0, 0x42800000, s62
	v_add_f32_e32 v194, v194, v255
	v_cndmask_b32_e64 v255, 0, 0xffffffc0, s63
	v_cndmask_b32_e32 v13, s74, v130, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v132
	v_dual_mul_f32 v130, 0xbfb8aa3b, v32 :: v_dual_add_f32 v193, v193, v254
	v_exp_f32_e32 v194, v194
	v_cndmask_b32_e64 v254, 0, 0xffffffc0, s62
	v_cndmask_b32_e32 v15, s74, v132, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v133
	v_mul_f32_e32 v132, 0xbfb8aa3b, v34
	v_cmp_gt_f32_e64 s3, 0xc2fc0000, v130
	v_exp_f32_e32 v193, v193
	v_cndmask_b32_e32 v14, s74, v133, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v131
	v_mul_f32_e32 v133, 0xbfb8aa3b, v35
	v_cmp_gt_f32_e64 s5, 0xc2fc0000, v132
	v_cndmask_b32_e64 v196, 0, 0x42800000, s3
	v_cndmask_b32_e32 v12, s74, v131, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v134
	v_mul_f32_e32 v131, 0xbfb8aa3b, v33
	v_cmp_gt_f32_e64 s6, 0xc2fc0000, v133
	v_cndmask_b32_e64 v198, 0, 0x42800000, s5
	v_dual_add_f32 v130, v130, v196 :: v_dual_cndmask_b32 v19, s74, v134
	v_cmp_gt_f32_e32 vcc_lo, s74, v136
	v_mul_f32_e32 v134, 0xbfb8aa3b, v38
	v_cmp_gt_f32_e64 s4, 0xc2fc0000, v131
	v_cndmask_b32_e64 v199, 0, 0x42800000, s6
	v_dual_add_f32 v132, v132, v198 :: v_dual_cndmask_b32 v21, s74, v136
	v_cmp_gt_f32_e32 vcc_lo, s74, v137
	v_mul_f32_e32 v136, 0xbfb8aa3b, v40
	v_cmp_gt_f32_e64 s7, 0xc2fc0000, v134
	v_cndmask_b32_e64 v197, 0, 0x42800000, s4
	v_dual_add_f32 v133, v133, v199 :: v_dual_cndmask_b32 v20, s74, v137
	v_cmp_gt_f32_e32 vcc_lo, s74, v135
	v_mul_f32_e32 v137, 0xbfb8aa3b, v41
	v_cmp_gt_f32_e64 s9, 0xc2fc0000, v136
	v_cndmask_b32_e64 v200, 0, 0x42800000, s7
	v_dual_add_f32 v131, v131, v197 :: v_dual_cndmask_b32 v18, s74, v135
	v_cmp_gt_f32_e32 vcc_lo, s74, v138
	v_mul_f32_e32 v135, 0xbfb8aa3b, v39
	v_cmp_gt_f32_e64 s10, 0xc2fc0000, v137
	v_cndmask_b32_e64 v202, 0, 0x42800000, s9
	v_dual_add_f32 v134, v134, v200 :: v_dual_cndmask_b32 v23, s74, v138
	v_cmp_gt_f32_e32 vcc_lo, s74, v140
	v_mul_f32_e32 v138, 0xbfb8aa3b, v42
	v_cmp_gt_f32_e64 s8, 0xc2fc0000, v135
	v_cndmask_b32_e64 v203, 0, 0x42800000, s10
	v_dual_add_f32 v136, v136, v202 :: v_dual_cndmask_b32 v25, s74, v140
	v_cmp_gt_f32_e32 vcc_lo, s74, v141
	v_mul_f32_e32 v140, 0xbfb8aa3b, v44
	v_cmp_gt_f32_e64 s11, 0xc2fc0000, v138
	v_cndmask_b32_e64 v201, 0, 0x42800000, s8
	v_dual_add_f32 v137, v137, v203 :: v_dual_cndmask_b32 v24, s74, v141
	v_cmp_gt_f32_e32 vcc_lo, s74, v139
	v_mul_f32_e32 v141, 0xbfb8aa3b, v45
	v_cmp_gt_f32_e64 s13, 0xc2fc0000, v140
	v_cndmask_b32_e64 v204, 0, 0x42800000, s11
	v_dual_add_f32 v135, v135, v201 :: v_dual_cndmask_b32 v22, s74, v139
	v_cmp_gt_f32_e32 vcc_lo, s74, v142
	v_mul_f32_e32 v139, 0xbfb8aa3b, v43
	v_cmp_gt_f32_e64 s14, 0xc2fc0000, v141
	v_cndmask_b32_e64 v206, 0, 0x42800000, s13
	v_dual_add_f32 v138, v138, v204 :: v_dual_cndmask_b32 v27, s74, v142
	v_cmp_gt_f32_e32 vcc_lo, s74, v144
	v_mul_f32_e32 v142, 0xbfb8aa3b, v46
	v_cmp_gt_f32_e64 s12, 0xc2fc0000, v139
	v_cndmask_b32_e64 v207, 0, 0x42800000, s14
	v_dual_add_f32 v140, v140, v206 :: v_dual_cndmask_b32 v29, s74, v144
	v_cmp_gt_f32_e32 vcc_lo, s74, v145
	v_mul_f32_e32 v144, 0xbfb8aa3b, v48
	v_cmp_gt_f32_e64 s15, 0xc2fc0000, v142
	v_cndmask_b32_e64 v205, 0, 0x42800000, s12
	v_dual_add_f32 v141, v141, v207 :: v_dual_cndmask_b32 v28, s74, v145
	v_cmp_gt_f32_e32 vcc_lo, s74, v143
	v_mul_f32_e32 v145, 0xbfb8aa3b, v49
	v_cmp_gt_f32_e64 s17, 0xc2fc0000, v144
	v_cndmask_b32_e64 v208, 0, 0x42800000, s15
	v_dual_add_f32 v139, v139, v205 :: v_dual_cndmask_b32 v26, s74, v143
	v_cmp_gt_f32_e32 vcc_lo, s74, v146
	v_mul_f32_e32 v143, 0xbfb8aa3b, v47
	v_cmp_gt_f32_e64 s18, 0xc2fc0000, v145
	v_cndmask_b32_e64 v210, 0, 0x42800000, s17
	v_dual_add_f32 v142, v142, v208 :: v_dual_cndmask_b32 v31, s74, v146
	v_cmp_gt_f32_e32 vcc_lo, s74, v148
	v_mul_f32_e32 v146, 0xbfb8aa3b, v50
	v_cmp_gt_f32_e64 s16, 0xc2fc0000, v143
	v_cndmask_b32_e64 v211, 0, 0x42800000, s18
	v_dual_add_f32 v144, v144, v210 :: v_dual_cndmask_b32 v81, s74, v148
	v_cmp_gt_f32_e32 vcc_lo, s74, v149
	v_mul_f32_e32 v148, 0xbfb8aa3b, v52
	v_cmp_gt_f32_e64 s19, 0xc2fc0000, v146
	v_cndmask_b32_e64 v209, 0, 0x42800000, s16
	v_dual_add_f32 v145, v145, v211 :: v_dual_cndmask_b32 v80, s74, v149
	v_cmp_gt_f32_e32 vcc_lo, s74, v147
	v_mul_f32_e32 v149, 0xbfb8aa3b, v53
	v_cmp_gt_f32_e64 s21, 0xc2fc0000, v148
	v_cndmask_b32_e64 v212, 0, 0x42800000, s19
	v_dual_add_f32 v143, v143, v209 :: v_dual_cndmask_b32 v30, s74, v147
	v_cmp_gt_f32_e32 vcc_lo, s74, v150
	v_mul_f32_e32 v147, 0xbfb8aa3b, v51
	v_cmp_gt_f32_e64 s22, 0xc2fc0000, v149
	v_cndmask_b32_e64 v214, 0, 0x42800000, s21
	v_dual_add_f32 v146, v146, v212 :: v_dual_cndmask_b32 v83, s74, v150
	v_cmp_gt_f32_e32 vcc_lo, s74, v152
	v_mul_f32_e32 v150, 0xbfb8aa3b, v54
	v_cmp_gt_f32_e64 s20, 0xc2fc0000, v147
	v_cndmask_b32_e64 v215, 0, 0x42800000, s22
	v_dual_add_f32 v148, v148, v214 :: v_dual_cndmask_b32 v85, s74, v152
	v_cmp_gt_f32_e32 vcc_lo, s74, v153
	v_mul_f32_e32 v152, 0xbfb8aa3b, v56
	v_cmp_gt_f32_e64 s23, 0xc2fc0000, v150
	v_cndmask_b32_e64 v213, 0, 0x42800000, s20
	v_dual_add_f32 v149, v149, v215 :: v_dual_cndmask_b32 v84, s74, v153
	v_cmp_gt_f32_e32 vcc_lo, s74, v151
	v_mul_f32_e32 v153, 0xbfb8aa3b, v57
	v_cmp_gt_f32_e64 s25, 0xc2fc0000, v152
	v_cndmask_b32_e64 v216, 0, 0x42800000, s23
	v_dual_add_f32 v147, v147, v213 :: v_dual_cndmask_b32 v82, s74, v151
	v_cmp_gt_f32_e32 vcc_lo, s74, v86
	v_mul_f32_e32 v151, 0xbfb8aa3b, v55
	v_cmp_gt_f32_e64 s26, 0xc2fc0000, v153
	v_cndmask_b32_e64 v218, 0, 0x42800000, s25
	v_dual_add_f32 v150, v150, v216 :: v_dual_cndmask_b32 v87, s74, v86
	v_cmp_gt_f32_e32 vcc_lo, s74, v155
	v_cmp_gt_f32_e64 s24, 0xc2fc0000, v151
	v_cndmask_b32_e64 v219, 0, 0x42800000, s26
	v_add_f32_e32 v152, v152, v218
	v_exp_f32_e32 v130, v130
	v_cndmask_b32_e32 v89, s74, v155, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v156
	v_mul_f32_e32 v155, 0xbfb8aa3b, v59
	v_cndmask_b32_e64 v217, 0, 0x42800000, s24
	v_add_f32_e32 v153, v153, v219
	v_exp_f32_e32 v131, v131
	v_cndmask_b32_e32 v88, s74, v156, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v154
	v_mul_f32_e32 v156, 0xbfb8aa3b, v60
	v_cmp_gt_f32_e64 s28, 0xc2fc0000, v155
	v_add_f32_e32 v151, v151, v217
	v_exp_f32_e32 v132, v132
	v_cndmask_b32_e32 v86, s74, v154, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v92
	v_mul_f32_e32 v154, 0xbfb8aa3b, v58
	v_cmp_gt_f32_e64 s29, 0xc2fc0000, v156
	v_cndmask_b32_e64 v221, 0, 0x42800000, s28
	v_exp_f32_e32 v133, v133
	v_cndmask_b32_e32 v91, s74, v92, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v90
	v_cmp_gt_f32_e64 s27, 0xc2fc0000, v154
	v_cndmask_b32_e64 v222, 0, 0x42800000, s29
	v_add_f32_e32 v155, v155, v221
	v_exp_f32_e32 v134, v134
	v_cndmask_b32_e32 v93, s74, v90, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v95
	v_cndmask_b32_e64 v220, 0, 0x42800000, s27
	v_add_f32_e32 v156, v156, v222
	v_exp_f32_e32 v135, v135
	v_exp_f32_e32 v136, v136
	v_cndmask_b32_e32 v92, s74, v95, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v94
	v_add_f32_e32 v154, v154, v220
	v_exp_f32_e32 v137, v137
	v_exp_f32_e32 v138, v138
	v_exp_f32_e32 v139, v139
	v_cndmask_b32_e32 v90, s74, v94, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v100
	v_exp_f32_e32 v140, v140
	v_exp_f32_e32 v141, v141
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v143, v143
	v_cndmask_b32_e32 v95, s74, v100, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v96
	v_exp_f32_e32 v144, v144
	v_exp_f32_e32 v145, v145
	v_exp_f32_e32 v146, v146
	v_exp_f32_e32 v147, v147
	v_cndmask_b32_e32 v97, s74, v96, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v98
	v_exp_f32_e32 v148, v148
	v_exp_f32_e32 v149, v149
	v_exp_f32_e32 v150, v150
	v_exp_f32_e32 v151, v151
	v_cndmask_b32_e32 v96, s74, v98, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v101
	v_exp_f32_e32 v152, v152
	v_exp_f32_e32 v153, v153
	v_exp_f32_e32 v154, v154
	v_exp_f32_e32 v155, v155
	v_cndmask_b32_e32 v94, s74, v101, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v99
	v_exp_f32_e32 v156, v156
	v_cndmask_b32_e64 v196, 0, 0xffffffc0, s3
	v_cndmask_b32_e64 v197, 0, 0xffffffc0, s4
	v_cndmask_b32_e64 v198, 0, 0xffffffc0, s5
	v_cndmask_b32_e32 v99, s74, v99, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v103
	v_cndmask_b32_e64 v199, 0, 0xffffffc0, s6
	v_cndmask_b32_e64 v200, 0, 0xffffffc0, s7
	v_cndmask_b32_e64 v201, 0, 0xffffffc0, s8
	v_cndmask_b32_e64 v202, 0, 0xffffffc0, s9
	v_cndmask_b32_e32 v101, s74, v103, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v104
	v_cndmask_b32_e64 v203, 0, 0xffffffc0, s10
	v_cndmask_b32_e64 v204, 0, 0xffffffc0, s11
	v_cndmask_b32_e64 v205, 0, 0xffffffc0, s12
	v_cndmask_b32_e64 v206, 0, 0xffffffc0, s13
	v_cndmask_b32_e32 v100, s74, v104, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v102
	v_cndmask_b32_e64 v207, 0, 0xffffffc0, s14
	v_cndmask_b32_e64 v208, 0, 0xffffffc0, s15
	v_cndmask_b32_e64 v209, 0, 0xffffffc0, s16
	v_cndmask_b32_e64 v210, 0, 0xffffffc0, s17
	v_cndmask_b32_e32 v98, s74, v102, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v105
	v_cndmask_b32_e64 v211, 0, 0xffffffc0, s18
	v_cndmask_b32_e64 v212, 0, 0xffffffc0, s19
	v_cndmask_b32_e64 v213, 0, 0xffffffc0, s20
	v_cndmask_b32_e64 v214, 0, 0xffffffc0, s21
	v_cndmask_b32_e32 v103, s74, v105, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v107
	v_cndmask_b32_e64 v215, 0, 0xffffffc0, s22
	v_cndmask_b32_e64 v216, 0, 0xffffffc0, s23
	v_cndmask_b32_e64 v217, 0, 0xffffffc0, s24
	v_cndmask_b32_e64 v218, 0, 0xffffffc0, s25
	v_cndmask_b32_e32 v105, s74, v107, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v108
	v_cndmask_b32_e64 v219, 0, 0xffffffc0, s26
	v_cndmask_b32_e64 v220, 0, 0xffffffc0, s27
	v_cndmask_b32_e64 v221, 0, 0xffffffc0, s28
	v_cndmask_b32_e64 v222, 0, 0xffffffc0, s29
	v_cndmask_b32_e32 v104, s74, v108, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v106
	v_ldexp_f32 v130, v130, v196
	v_ldexp_f32 v131, v131, v197
	v_ldexp_f32 v132, v132, v198
	v_ldexp_f32 v133, v133, v199
	v_cndmask_b32_e32 v102, s74, v106, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v109
	v_ldexp_f32 v134, v134, v200
	v_ldexp_f32 v135, v135, v201
	v_ldexp_f32 v136, v136, v202
	v_ldexp_f32 v137, v137, v203
	v_cndmask_b32_e32 v107, s74, v109, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v111
	v_ldexp_f32 v138, v138, v204
	v_ldexp_f32 v139, v139, v205
	v_ldexp_f32 v140, v140, v206
	v_ldexp_f32 v141, v141, v207
	v_cndmask_b32_e32 v109, s74, v111, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v112
	v_ldexp_f32 v142, v142, v208
	v_ldexp_f32 v143, v143, v209
	v_ldexp_f32 v144, v144, v210
	v_ldexp_f32 v145, v145, v211
	v_cndmask_b32_e32 v108, s74, v112, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v110
	v_ldexp_f32 v146, v146, v212
	v_ldexp_f32 v147, v147, v213
	v_ldexp_f32 v148, v148, v214
	v_ldexp_f32 v149, v149, v215
	v_cndmask_b32_e32 v106, s74, v110, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v113
	v_ldexp_f32 v150, v150, v216
	v_ldexp_f32 v151, v151, v217
	v_ldexp_f32 v152, v152, v218
	v_ldexp_f32 v153, v153, v219
	v_cndmask_b32_e32 v111, s74, v113, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v115
	v_ldexp_f32 v154, v154, v220
	v_ldexp_f32 v155, v155, v221
	v_ldexp_f32 v156, v156, v222
	v_dual_add_f32 v196, 1.0, v134 :: v_dual_cndmask_b32 v113, s74, v115
	v_cmp_gt_f32_e32 vcc_lo, s74, v116
	v_dual_add_f32 v197, 1.0, v135 :: v_dual_add_f32 v198, 1.0, v136
	v_dual_add_f32 v199, 1.0, v137 :: v_dual_add_f32 v200, 1.0, v138
	v_cndmask_b32_e32 v112, s74, v116, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v114
	v_dual_add_f32 v201, 1.0, v139 :: v_dual_add_f32 v202, 1.0, v140
	v_dual_add_f32 v203, 1.0, v141 :: v_dual_add_f32 v204, 1.0, v142
	v_cndmask_b32_e32 v110, s74, v114, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v117
	v_dual_add_f32 v205, 1.0, v143 :: v_dual_add_f32 v206, 1.0, v144
	v_dual_add_f32 v207, 1.0, v145 :: v_dual_add_f32 v208, 1.0, v146
	v_cndmask_b32_e32 v115, s74, v117, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v119
	v_dual_add_f32 v209, 1.0, v147 :: v_dual_add_f32 v210, 1.0, v148
	v_dual_add_f32 v211, 1.0, v149 :: v_dual_add_f32 v212, 1.0, v150
	v_cndmask_b32_e32 v117, s74, v119, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v120
	v_dual_add_f32 v213, 1.0, v151 :: v_dual_add_f32 v214, 1.0, v152
	v_dual_add_f32 v215, 1.0, v153 :: v_dual_add_f32 v216, 1.0, v154
	v_cndmask_b32_e32 v116, s74, v120, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v118
	v_dual_add_f32 v217, 1.0, v155 :: v_dual_add_f32 v218, 1.0, v156
	v_rcp_f32_e32 v136, v196
	v_rcp_f32_e32 v137, v197
	v_cndmask_b32_e32 v114, s74, v118, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v121
	v_rcp_f32_e32 v138, v198
	v_rcp_f32_e32 v139, v199
	v_rcp_f32_e32 v140, v200
	v_rcp_f32_e32 v141, v201
	v_cndmask_b32_e32 v119, s74, v121, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v123
	v_rcp_f32_e32 v142, v202
	v_rcp_f32_e32 v143, v203
	v_rcp_f32_e32 v144, v204
	v_rcp_f32_e32 v145, v205
	v_cndmask_b32_e32 v121, s74, v123, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v124
	v_rcp_f32_e32 v146, v206
	v_rcp_f32_e32 v147, v207
	v_rcp_f32_e32 v148, v208
	v_rcp_f32_e32 v149, v209
	v_cndmask_b32_e32 v120, s74, v124, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v122
	v_rcp_f32_e32 v150, v210
	v_rcp_f32_e32 v151, v211
	v_rcp_f32_e32 v152, v212
	v_rcp_f32_e32 v153, v213
	v_cndmask_b32_e32 v118, s74, v122, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v125
	v_rcp_f32_e32 v154, v214
	v_rcp_f32_e32 v155, v215
	v_rcp_f32_e32 v156, v216
	v_pk_mul_f32 v[38:39], v[38:39], v[136:137]
	v_cndmask_b32_e32 v123, s74, v125, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v127
	v_pk_mul_f32 v[40:41], v[40:41], v[138:139]
	v_pk_mul_f32 v[42:43], v[42:43], v[140:141]
	v_pk_mul_f32 v[44:45], v[44:45], v[142:143]
	v_pk_mul_f32 v[46:47], v[46:47], v[144:145]
	v_cndmask_b32_e32 v125, s74, v127, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v157
	v_cndmask_b32_e64 v127, 0, 0x42800000, s0
	v_pk_mul_f32 v[48:49], v[48:49], v[146:147]
	v_pk_mul_f32 v[50:51], v[50:51], v[148:149]
	v_pk_mul_f32 v[52:53], v[52:53], v[150:151]
	v_cndmask_b32_e32 v124, s74, v157, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v126
	v_cndmask_b32_e64 v157, 0, 0x42800000, s1
	v_add_f32_e32 v127, v159, v127
	v_cndmask_b32_e64 v159, 0, 0xffffffc0, s0
	s_sub_co_i32 s0, s69, s80
	v_cndmask_b32_e32 v122, s74, v126, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, 0xc2fc0000, v158
	v_add_f32_e32 v157, v160, v157
	v_exp_f32_e32 v127, v127
	v_cndmask_b32_e64 v160, 0, 0xffffffc0, s1
	v_pk_mul_f32 v[54:55], v[54:55], v[152:153]
	v_cndmask_b32_e64 v126, 0, 0x42800000, vcc_lo
	v_exp_f32_e32 v157, v157
	v_pk_mul_f32 v[56:57], v[56:57], v[154:155]
	s_ashr_i32 s1, s0, 31
	v_pk_mul_f32 v[24:25], v[24:25], v[40:41]
	v_add_f32_e32 v126, v158, v126
	v_cndmask_b32_e64 v158, 0, 0xffffffc0, vcc_lo
	v_ldexp_f32 v127, v127, v159
	v_ldexp_f32 v159, v162, v223
	v_ldexp_f32 v157, v157, v160
	v_exp_f32_e32 v126, v126
	v_ldexp_f32 v160, v163, v224
	v_ldexp_f32 v162, v165, v226
	v_ldexp_f32 v163, v166, v227
	v_ldexp_f32 v165, v168, v229
	v_ldexp_f32 v166, v169, v230
	v_ldexp_f32 v168, v171, v232
	v_ldexp_f32 v169, v172, v233
	v_ldexp_f32 v126, v126, v158
	v_ldexp_f32 v158, v161, v195
	v_ldexp_f32 v161, v164, v225
	v_ldexp_f32 v164, v167, v228
	v_ldexp_f32 v167, v170, v231
	v_ldexp_f32 v170, v173, v234
	v_ldexp_f32 v171, v174, v235
	v_ldexp_f32 v172, v175, v236
	v_ldexp_f32 v173, v176, v237
	v_ldexp_f32 v174, v177, v238
	v_ldexp_f32 v175, v178, v239
	v_ldexp_f32 v176, v179, v240
	v_ldexp_f32 v177, v180, v241
	v_ldexp_f32 v178, v181, v242
	v_ldexp_f32 v179, v182, v243
	v_ldexp_f32 v180, v183, v244
	v_ldexp_f32 v181, v184, v245
	v_ldexp_f32 v182, v185, v246
	v_ldexp_f32 v183, v186, v247
	v_ldexp_f32 v184, v187, v248
	v_ldexp_f32 v185, v188, v249
	v_ldexp_f32 v186, v189, v250
	v_ldexp_f32 v187, v190, v251
	v_ldexp_f32 v188, v191, v252
	v_ldexp_f32 v189, v192, v253
	v_ldexp_f32 v190, v193, v254
	v_ldexp_f32 v191, v194, v255
	v_dual_add_f32 v126, 1.0, v126 :: v_dual_add_f32 v127, 1.0, v127
	v_dual_add_f32 v157, 1.0, v157 :: v_dual_add_f32 v158, 1.0, v158
	v_dual_add_f32 v192, 1.0, v130 :: v_dual_add_f32 v193, 1.0, v131
	v_dual_add_f32 v194, 1.0, v132 :: v_dual_add_f32 v195, 1.0, v133
	v_dual_add_f32 v159, 1.0, v159 :: v_dual_add_f32 v160, 1.0, v160
	v_dual_add_f32 v161, 1.0, v161 :: v_dual_add_f32 v162, 1.0, v162
	v_dual_add_f32 v163, 1.0, v163 :: v_dual_add_f32 v164, 1.0, v164
	v_dual_add_f32 v165, 1.0, v165 :: v_dual_add_f32 v166, 1.0, v166
	v_dual_add_f32 v167, 1.0, v167 :: v_dual_add_f32 v168, 1.0, v168
	v_dual_add_f32 v169, 1.0, v169 :: v_dual_add_f32 v170, 1.0, v170
	v_dual_add_f32 v171, 1.0, v171 :: v_dual_add_f32 v172, 1.0, v172
	v_dual_add_f32 v173, 1.0, v173 :: v_dual_add_f32 v174, 1.0, v174
	v_dual_add_f32 v175, 1.0, v175 :: v_dual_add_f32 v176, 1.0, v176
	v_dual_add_f32 v177, 1.0, v177 :: v_dual_add_f32 v178, 1.0, v178
	v_dual_add_f32 v179, 1.0, v179 :: v_dual_add_f32 v180, 1.0, v180
	v_dual_add_f32 v181, 1.0, v181 :: v_dual_add_f32 v182, 1.0, v182
	v_dual_add_f32 v183, 1.0, v183 :: v_dual_add_f32 v184, 1.0, v184
	v_dual_add_f32 v185, 1.0, v185 :: v_dual_add_f32 v186, 1.0, v186
	v_dual_add_f32 v187, 1.0, v187 :: v_dual_add_f32 v188, 1.0, v188
	v_dual_add_f32 v189, 1.0, v189 :: v_dual_add_f32 v190, 1.0, v190
	v_add_f32_e32 v191, 1.0, v191
	v_rcp_f32_e32 v126, v126
	v_rcp_f32_e32 v127, v127
	v_rcp_f32_e32 v130, v157
	v_rcp_f32_e32 v131, v158
	v_rcp_f32_e32 v132, v192
	v_rcp_f32_e32 v133, v193
	v_rcp_f32_e32 v134, v194
	v_rcp_f32_e32 v135, v195
	v_rcp_f32_e32 v157, v217
	v_rcp_f32_e32 v158, v218
	v_rcp_f32_e32 v159, v159
	v_rcp_f32_e32 v160, v160
	v_rcp_f32_e32 v161, v161
	v_rcp_f32_e32 v162, v162
	v_rcp_f32_e32 v163, v163
	v_rcp_f32_e32 v164, v164
	v_rcp_f32_e32 v165, v165
	v_rcp_f32_e32 v166, v166
	v_rcp_f32_e32 v167, v167
	v_rcp_f32_e32 v168, v168
	v_rcp_f32_e32 v169, v169
	v_rcp_f32_e32 v170, v170
	v_rcp_f32_e32 v171, v171
	v_rcp_f32_e32 v172, v172
	v_rcp_f32_e32 v173, v173
	v_rcp_f32_e32 v174, v174
	v_rcp_f32_e32 v175, v175
	v_rcp_f32_e32 v176, v176
	v_rcp_f32_e32 v177, v177
	v_rcp_f32_e32 v178, v178
	v_rcp_f32_e32 v179, v179
	v_rcp_f32_e32 v180, v180
	v_rcp_f32_e32 v181, v181
	v_rcp_f32_e32 v182, v182
	v_rcp_f32_e32 v183, v183
	v_rcp_f32_e32 v184, v184
	v_rcp_f32_e32 v185, v185
	v_rcp_f32_e32 v186, v186
	v_rcp_f32_e32 v187, v187
	v_rcp_f32_e32 v188, v188
	v_rcp_f32_e32 v189, v189
	v_rcp_f32_e32 v190, v190
	v_rcp_f32_e32 v191, v191
	v_pk_mul_f32 v[126:127], v[128:129], v[126:127]
	v_pk_mul_f32 v[36:37], v[36:37], v[130:131]
	v_pk_mul_f32 v[32:33], v[32:33], v[132:133]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135]
	v_pk_mul_f32 v[58:59], v[58:59], v[156:157]
	v_pk_mul_f32 v[60:61], v[60:61], v[158:159]
	v_pk_mul_f32 v[62:63], v[62:63], v[160:161]
	v_pk_mul_f32 v[64:65], v[64:65], v[162:163]
	v_pk_mul_f32 v[66:67], v[66:67], v[164:165]
	v_pk_mul_f32 v[68:69], v[68:69], v[166:167]
	v_pk_mul_f32 v[70:71], v[70:71], v[168:169]
	v_pk_mul_f32 v[72:73], v[72:73], v[170:171]
	v_pk_mul_f32 v[74:75], v[74:75], v[172:173]
	v_pk_mul_f32 v[76:77], v[76:77], v[174:175]
	v_pk_mul_f32 v[78:79], v[78:79], v[176:177]
	v_pk_mul_f32 v[16:17], v[16:17], v[178:179]
	v_pk_mul_f32 v[0:1], v[0:1], v[180:181]
	v_pk_mul_f32 v[2:3], v[2:3], v[182:183]
	v_pk_mul_f32 v[4:5], v[4:5], v[184:185]
	v_pk_mul_f32 v[6:7], v[6:7], v[186:187]
	v_pk_mul_f32 v[8:9], v[8:9], v[188:189]
	v_pk_mul_f32 v[10:11], v[10:11], v[190:191]
	s_mul_u64 s[2:3], s[66:67], s[0:1]
	v_pk_mul_f32 v[14:15], v[14:15], v[36:37]
	v_pk_mul_f32 v[12:13], v[12:13], v[126:127]
	v_pk_mul_f32 v[20:21], v[20:21], v[34:35]
	v_pk_mul_f32 v[18:19], v[18:19], v[32:33]
	v_pk_mul_f32 v[22:23], v[22:23], v[38:39]
	v_pk_mul_f32 v[28:29], v[28:29], v[44:45]
	v_pk_mul_f32 v[26:27], v[26:27], v[42:43]
	v_pk_mul_f32 v[32:33], v[80:81], v[48:49]
	v_pk_mul_f32 v[30:31], v[30:31], v[46:47]
	v_pk_mul_f32 v[34:35], v[84:85], v[52:53]
	v_pk_mul_f32 v[36:37], v[82:83], v[50:51]
	v_pk_mul_f32 v[38:39], v[88:89], v[56:57]
	v_pk_mul_f32 v[40:41], v[86:87], v[54:55]
	v_pk_mul_f32 v[42:43], v[92:93], v[60:61]
	v_pk_mul_f32 v[44:45], v[90:91], v[58:59]
	v_pk_mul_f32 v[46:47], v[96:97], v[64:65]
	v_pk_mul_f32 v[48:49], v[94:95], v[62:63]
	v_pk_mul_f32 v[50:51], v[100:101], v[68:69]
	v_pk_mul_f32 v[52:53], v[98:99], v[66:67]
	v_pk_mul_f32 v[54:55], v[104:105], v[72:73]
	v_pk_mul_f32 v[56:57], v[102:103], v[70:71]
	v_pk_mul_f32 v[58:59], v[108:109], v[76:77]
	v_pk_mul_f32 v[60:61], v[106:107], v[74:75]
	v_pk_mul_f32 v[16:17], v[112:113], v[16:17]
	v_pk_mul_f32 v[62:63], v[110:111], v[78:79]
	v_pk_mul_f32 v[2:3], v[116:117], v[2:3]
	v_pk_mul_f32 v[0:1], v[114:115], v[0:1]
	v_pk_mul_f32 v[6:7], v[120:121], v[6:7]
	v_pk_mul_f32 v[4:5], v[118:119], v[4:5]
	v_pk_mul_f32 v[10:11], v[124:125], v[10:11]
	v_pk_mul_f32 v[8:9], v[122:123], v[8:9]
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b32 s4, s75, 16
	s_lshr_b32 s5, s75, 16
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[78:79], s[2:3]
	s_or_b32 s66, s4, 0x7fff
	s_or_b32 s67, s5, 0x800000
	s_mul_u64 s[4:5], s[70:71], s[0:1]
	s_add_nc_u64 s[2:3], s[2:3], s[76:77]
	v_cvt_pk_bf16_f32 v15, v14, v15
	v_cvt_pk_bf16_f32 v14, v12, v13
	v_cvt_pk_bf16_f32 v13, v20, v21
	v_cvt_pk_bf16_f32 v12, v18, v19
	v_cvt_pk_bf16_f32 v19, v24, v25
	v_cvt_pk_bf16_f32 v18, v22, v23
	v_cvt_pk_bf16_f32 v21, v28, v29
	v_cvt_pk_bf16_f32 v20, v26, v27
	v_cvt_pk_bf16_f32 v23, v32, v33
	v_cvt_pk_bf16_f32 v22, v30, v31
	v_cvt_pk_bf16_f32 v25, v34, v35
	v_cvt_pk_bf16_f32 v24, v36, v37
	v_cvt_pk_bf16_f32 v27, v38, v39
	v_cvt_pk_bf16_f32 v26, v40, v41
	v_cvt_pk_bf16_f32 v29, v42, v43
	v_cvt_pk_bf16_f32 v28, v44, v45
	v_cvt_pk_bf16_f32 v31, v46, v47
	v_cvt_pk_bf16_f32 v30, v48, v49
	v_cvt_pk_bf16_f32 v33, v50, v51
	v_cvt_pk_bf16_f32 v32, v52, v53
	v_cvt_pk_bf16_f32 v35, v54, v55
	v_cvt_pk_bf16_f32 v34, v56, v57
	v_cvt_pk_bf16_f32 v37, v58, v59
	v_cvt_pk_bf16_f32 v36, v60, v61
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v62, v63
	v_cvt_pk_bf16_f32 v3, v2, v3
	v_cvt_pk_bf16_f32 v2, v0, v1
	v_cvt_pk_bf16_f32 v1, v6, v7
	v_cvt_pk_bf16_f32 v0, v4, v5
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_cvt_pk_bf16_f32 v4, v8, v9
	s_add_nc_u64 s[74:75], s[4:5], s[2:3]
	s_mov_b32 s69, s0
	s_and_b32 s70, s1, 0xffff
	s_bitset1_b32 s75, 31
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_store_2addr_b64 v0 /*v256*/, v[14:15], v[12:13] offset1:2
	ds_store_2addr_b64 v0 /*v256*/, v[18:19], v[20:21] offset0:4 offset1:6
	ds_store_2addr_b64 v1 /*v257*/, v[22:23], v[24:25] offset1:2
	ds_store_2addr_b64 v1 /*v257*/, v[26:27], v[28:29] offset0:4 offset1:6
	ds_store_2addr_b64 v2 /*v258*/, v[30:31], v[32:33] offset1:2
	ds_store_2addr_b64 v2 /*v258*/, v[34:35], v[36:37] offset0:4 offset1:6
	ds_store_2addr_b64 v3 /*v259*/, v[16:17], v[2:3] offset1:2
	ds_store_2addr_b64 v3 /*v259*/, v[0:1], v[4:5] offset0:4 offset1:6
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_store_from_lds s[72:75], s[64:71]
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x100
.LBB0_45:
	s_endpgm
.LBB0_46:
	s_max_i32 s14, s33, 0
	s_add_nc_u64 s[10:11], s[64:65], 0x100
	s_mov_b32 s86, 0
	s_lshl_b32 s15, s14, 16
	s_lshr_b32 s14, s14, 16
	s_bitset1_b32 s11, 31
	s_add_co_i32 s9, 0, 0xce00
	s_or_b32 s82, s15, 0x7fff
	s_or_b32 s83, s14, 0x1000000
	s_movk_i32 s85, 0x1c00
	s_mov_b32 s84, 32
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x7500000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_19
.LBB0_47:
	s_max_i32 s14, s40, 0
	s_add_nc_u64 s[10:11], s[20:21], 0x100
	s_mov_b32 s86, 0
	s_lshl_b32 s15, s14, 16
	s_lshr_b32 s14, s14, 16
	s_bitset1_b32 s11, 31
	s_add_co_i32 s9, 0, 0xf000
	s_or_b32 s82, s15, 0x7fff
	s_or_b32 s83, s14, 0x1000000
	s_movk_i32 s85, 0x1c00
	s_mov_b32 s84, 32
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x7500000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB0_20
.LBB0_48:
	s_add_nc_u64 s[10:11], s[24:25], 0x800
	s_mov_b32 s80, 0
	s_add_co_i32 s9, 0, 0x11200
	s_bitset1_b32 s11, 31
	s_mov_b32 s85, 0xe000
	s_mov_b32 s84, 8
	s_mov_b32 s83, 0x8007fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s86, s80
	s_mov_b32 s87, s80
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_21
.LBB0_49:
	s_add_nc_u64 s[10:11], s[26:27], 0x800
	s_mov_b32 s80, 0
	s_add_co_i32 s9, 0, 0x15200
	s_bitset1_b32 s11, 31
	s_mov_b32 s85, 0xe000
	s_mov_b32 s84, 8
	s_mov_b32 s83, 0x8007fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s86, s80
	s_mov_b32 s87, s80
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_22
.LBB0_50:
	s_add_nc_u64 s[10:11], s[28:29], 32
	s_mov_b32 s86, 0
	s_add_co_i32 s9, 0, 0x19200
	s_bitset1_b32 s11, 31
	s_movk_i32 s85, 0xe0
	s_mov_b32 s84, 8
	s_mov_b32 s83, 0x87fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x20000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_23
.LBB0_51:
	s_add_nc_u64 s[10:11], s[36:37], 32
	s_mov_b32 s86, 0
	s_add_co_i32 s9, 0, 0x19300
	s_bitset1_b32 s11, 31
	s_movk_i32 s85, 0xe0
	s_mov_b32 s84, 8
	s_mov_b32 s83, 0x87fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x20000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB0_24
.LBB0_52:
	s_add_nc_u64 s[10:11], s[38:39], 0x100
	s_mov_b32 s86, 0
	s_add_co_i32 s9, 0, 0x19400
	s_bitset1_b32 s11, 31
	s_movk_i32 s85, 0x700
	s_mov_b32 s84, 4
	s_mov_b32 s83, 0x407fff
	s_mov_b32 s82, 0xffff7fff
	s_mov_b32 s81, 0xffff0000
	s_mov_b32 s80, 0x20000
	s_mov_b32 s87, s86
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[8:11], s[80:87]
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB0_25
	s_branch .LBB0_26
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_gemm1_a8w4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 104
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
		.amdhsa_next_free_vgpr 513
		.amdhsa_next_free_sgpr 88
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 148
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
	.size	moe_gemm1_a8w4, .Lfunc_end0-moe_gemm1_a8w4

	.set moe_gemm1_a8w4.num_vgpr, 346
	.set moe_gemm1_a8w4.num_agpr, 0
	.set moe_gemm1_a8w4.numbered_sgpr, 88
	.set moe_gemm1_a8w4.num_named_barrier, 0
	.set moe_gemm1_a8w4.private_seg_size, 0
	.set moe_gemm1_a8w4.uses_vcc, 1
	.set moe_gemm1_a8w4.uses_flat_scratch, 0
	.set moe_gemm1_a8w4.has_dyn_sized_stack, 0
	.set moe_gemm1_a8w4.has_recursion, 0
	.set moe_gemm1_a8w4.has_indirect_call, 0
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
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 128
    .name:           moe_gemm1_a8w4
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     90
    .sgpr_spill_count: 0
    .symbol:         moe_gemm1_a8w4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     346
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
