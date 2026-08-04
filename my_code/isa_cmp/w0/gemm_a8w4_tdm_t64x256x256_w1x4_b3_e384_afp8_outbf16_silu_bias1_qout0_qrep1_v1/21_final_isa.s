	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
	.p2align	8
	.type	gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1,@function
gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1:
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
	v_bfe_u32 v22 /*v278*/, v0, 4, 1
	v_and_b32_e32 v23 /*v279*/, 15, v0
	s_lshr_b32 s8, s41, 1
	s_lshl_b32 s9, s42, 6
	s_and_b32 s8, s8, 0x7fffffc0
	s_set_vgpr_msb 0x4004
	v_dual_lshlrev_b32 v0, 4, v22 /*v278*/ :: v_dual_lshlrev_b32 v3, 8, v22 /*v278*/
	v_dual_mov_b32 v64, 0 :: v_dual_bitop2_b32 v225, s8, v23 /*v279*/ bitop3:0x54
	s_and_b32 s35, s9, 0xc0
	s_lshr_b32 s13, s76, 4
	v_dual_lshlrev_b32 v4, 4, v23 /*v279*/ :: v_dual_lshlrev_b32 v5, 2, v23 /*v279*/
	s_set_vgpr_msb 0x400
	v_mad_u32 v236, 0x110, v225, v0
	v_dual_sub_nc_u32 v0, s22, v1 :: v_dual_sub_nc_u32 v1, s30, v2
	s_lshl_b32 s11, s35, 7
	s_lshr_b32 s10, s76, 5
	s_addk_co_i32 s11, 0x4400
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v0, s34, v0
	v_mul_lo_u32 v1, s34, v1
	s_lshr_b32 s14, s8, 2
	s_max_i32 s8, s33, 0
	v_or3_b32 v239, s11, v3, v4
	s_lshl_b32 s11, s8, 16
	s_lshr_b32 s8, s8, 16
	s_mulk_i32 s10, 0x1c00
	v_dual_add_nc_u32 v0, s13, v0 :: v_dual_mov_b32 v224, 1
	v_mul_lo_u32 v1, 0x1c00, v1
	s_set_vgpr_msb 4
	v_dual_lshlrev_b32 v7, 2, v22 /*v278*/ :: v_dual_bitop2_b32 v6, s14, v23 /*v279*/ bitop3:0x54
	s_mov_b32 s16, 0
	s_or_b32 s27, s8, 0x1000000
	s_max_i32 s8, s40, 0
	s_set_vgpr_msb 0x400
	v_mul_lo_u32 v240, 0xe000, v0
	s_or_b32 s26, s11, 0x7fff
	v_dual_mov_b32 v65, v64 :: v_dual_add_nc_u32 v0, s10, v1
	s_mov_b32 s17, 0xffff0000
	s_mov_b32 s30, s16
	s_mov_b32 s31, s16
	s_lshl_b32 s11, s8, 16
	s_lshr_b32 s8, s8, 16
	s_lshr_b32 s9, s66, 2
	s_movk_i32 s29, 0x1c00
	s_mov_b32 s28, 32
	s_mov_b32 s24, 0x7500000
	s_mov_b32 s25, s17
	s_bitset1_b32 s8, 24
	s_mov_b64 s[42:43], s[30:31]
	v_lshl_or_b32 v237, s35, 3, v5
	v_lshl_or_b32 v238, v6, 5, v7
	s_mov_b64 s[38:39], s[26:27]
	v_dual_mov_b32 v66, v64 :: v_dual_add_nc_u32 v241, s44, v0
	v_dual_mov_b32 v67, v64 :: v_dual_mov_b32 v68, v64
	v_dual_mov_b32 v69, v64 :: v_dual_mov_b32 v70, v64
	v_dual_mov_b32 v71, v64 :: v_dual_mov_b32 v72, v64
	v_dual_mov_b32 v73, v64 :: v_dual_mov_b32 v74, v64
	v_dual_mov_b32 v75, v64 :: v_dual_mov_b32 v76, v64
	v_dual_mov_b32 v77, v64 :: v_dual_mov_b32 v78, v64
	v_dual_mov_b32 v79, v64 :: v_dual_mov_b32 v80, v64
	v_dual_mov_b32 v81, v64 :: v_dual_mov_b32 v82, v64
	v_dual_mov_b32 v83, v64 :: v_dual_mov_b32 v84, v64
	v_dual_mov_b32 v85, v64 :: v_dual_mov_b32 v86, v64
	v_dual_mov_b32 v87, v64 :: v_dual_mov_b32 v88, v64
	v_dual_mov_b32 v89, v64 :: v_dual_mov_b32 v90, v64
	v_dual_mov_b32 v91, v64 :: v_dual_mov_b32 v92, v64
	v_dual_mov_b32 v93, v64 :: v_dual_mov_b32 v94, v64
	v_dual_mov_b32 v95, v64 :: v_dual_mov_b32 v96, v64
	v_dual_mov_b32 v97, v64 :: v_dual_mov_b32 v98, v64
	v_dual_mov_b32 v99, v64 :: v_dual_mov_b32 v100, v64
	v_dual_mov_b32 v101, v64 :: v_dual_mov_b32 v102, v64
	v_dual_mov_b32 v103, v64 :: v_dual_mov_b32 v104, v64
	v_dual_mov_b32 v105, v64 :: v_dual_mov_b32 v106, v64
	v_dual_mov_b32 v107, v64 :: v_dual_mov_b32 v108, v64
	v_dual_mov_b32 v109, v64 :: v_dual_mov_b32 v110, v64
	v_dual_mov_b32 v111, v64 :: v_dual_mov_b32 v112, v64
	v_dual_mov_b32 v113, v64 :: v_dual_mov_b32 v114, v64
	v_dual_mov_b32 v115, v64 :: v_dual_mov_b32 v116, v64
	v_dual_mov_b32 v117, v64 :: v_dual_mov_b32 v118, v64
	v_dual_mov_b32 v119, v64 :: v_dual_mov_b32 v120, v64
	v_dual_mov_b32 v121, v64 :: v_dual_mov_b32 v122, v64
	v_dual_mov_b32 v123, v64 :: v_dual_mov_b32 v124, v64
	v_dual_mov_b32 v125, v64 :: v_dual_mov_b32 v126, v64
	v_dual_mov_b32 v127, v64 :: v_dual_mov_b32 v128, v64
	v_dual_mov_b32 v129, v64 :: v_dual_mov_b32 v130, v64
	v_dual_mov_b32 v131, v64 :: v_dual_mov_b32 v132, v64
	v_dual_mov_b32 v133, v64 :: v_dual_mov_b32 v134, v64
	v_dual_mov_b32 v135, v64 :: v_dual_mov_b32 v136, v64
	v_dual_mov_b32 v137, v64 :: v_dual_mov_b32 v138, v64
	v_dual_mov_b32 v139, v64 :: v_dual_mov_b32 v140, v64
	v_dual_mov_b32 v141, v64 :: v_dual_mov_b32 v142, v64
	v_dual_mov_b32 v143, v64 :: v_dual_mov_b32 v144, v64
	v_dual_mov_b32 v145, v64 :: v_dual_mov_b32 v146, v64
	v_dual_mov_b32 v147, v64 :: v_dual_mov_b32 v148, v64
	v_dual_mov_b32 v149, v64 :: v_dual_mov_b32 v150, v64
	v_dual_mov_b32 v151, v64 :: v_dual_mov_b32 v152, v64
	v_dual_mov_b32 v153, v64 :: v_dual_mov_b32 v154, v64
	v_dual_mov_b32 v155, v64 :: v_dual_mov_b32 v156, v64
	v_dual_mov_b32 v157, v64 :: v_dual_mov_b32 v158, v64
	v_dual_mov_b32 v159, v64 :: v_dual_mov_b32 v160, v64
	v_dual_mov_b32 v161, v64 :: v_dual_mov_b32 v162, v64
	v_dual_mov_b32 v163, v64 :: v_dual_mov_b32 v164, v64
	v_dual_mov_b32 v165, v64 :: v_dual_mov_b32 v166, v64
	v_dual_mov_b32 v167, v64 :: v_dual_mov_b32 v168, v64
	v_dual_mov_b32 v169, v64 :: v_dual_mov_b32 v170, v64
	v_dual_mov_b32 v171, v64 :: v_dual_mov_b32 v172, v64
	v_dual_mov_b32 v173, v64 :: v_dual_mov_b32 v174, v64
	v_dual_mov_b32 v175, v64 :: v_dual_mov_b32 v176, v64
	v_dual_mov_b32 v177, v64 :: v_dual_mov_b32 v178, v64
	v_dual_mov_b32 v179, v64 :: v_dual_mov_b32 v180, v64
	v_dual_mov_b32 v181, v64 :: v_dual_mov_b32 v182, v64
	v_dual_mov_b32 v183, v64 :: v_dual_mov_b32 v184, v64
	v_dual_mov_b32 v185, v64 :: v_dual_mov_b32 v186, v64
	v_dual_mov_b32 v187, v64 :: v_dual_mov_b32 v188, v64
	v_dual_mov_b32 v189, v64 :: v_dual_mov_b32 v190, v64
	v_mov_b32_e32 v191, v64
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
	s_set_vgpr_msb 64
	v_add_nc_u32_e32 v24 /*v280*/, s80, v236
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4001
	ds_load_b128 v[0:3], v24 /*v280*/
	ds_load_b128 v[4:7], v24 /*v280*/ offset:32
	ds_load_b128 v[8:11], v24 /*v280*/ offset:64
	ds_load_b128 v[12:15], v24 /*v280*/ offset:96
	ds_load_b128 v[32:35], v24 /*v280*/ offset:4352
	ds_load_b128 v[36:39], v24 /*v280*/ offset:4384
	ds_load_b128 v[40:43], v24 /*v280*/ offset:4416
	ds_load_b128 v[44:47], v24 /*v280*/ offset:4448
	ds_load_b128 v[16:19], v24 /*v280*/ offset:8704
	ds_load_b128 v[20:23], v24 /*v280*/ offset:8736
	ds_load_b128 v[24:27], v24 /*v280*/ offset:8768
	ds_load_b128 v[28:31], v24 /*v280*/ offset:8800
	ds_load_b128 v[48:51], v24 /*v280*/ offset:13056
	ds_load_b128 v[52:55], v24 /*v280*/ offset:13088
	ds_load_b128 v[56:59], v24 /*v280*/ offset:13120
	ds_load_b128 v[60:63], v24 /*v280*/ offset:13152
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x0
	ds_load_b128 v[246:249], v244 offset:1024
	ds_load_b128 v[250:253], v244 offset:1536
	ds_load_b128 v[254:257], v244 offset:3072
	s_set_vgpr_msb 64
	ds_load_b128 v[2:5] /*v[258:261]*/, v244 offset:3584
	ds_load_b128 v[6:9] /*v[262:265]*/, v244 offset:5120
	ds_load_b128 v[10:13] /*v[266:269]*/, v244 offset:5632
	ds_load_b128 v[14:17] /*v[270:273]*/, v244 offset:7168
	ds_load_b128 v[18:21] /*v[274:277]*/, v244 offset:7680
	s_set_vgpr_msb 0x4000
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[192:199], v[0:15], v[64:71], v226, v232 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[200:207], v[0:15], v[72:79], v227, v232 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[208:215], v[0:15], v[80:87], v230, v232 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[216:223], v[0:15], v[88:95], v231, v232 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v0, 0xc410, v242
	v_add_nc_u32_e32 v1, 0xc418, v242
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[216:223], v[32:47], v[120:127], v231, v233 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[208:215], v[32:47], v[112:119], v230, v233 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[200:207], v[32:47], v[104:111], v227, v233 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[192:199], v[32:47], v[96:103], v226, v233 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_2addr_b32 v[232:233], v243 offset0:160 offset1:176
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_2addr_b32 v[242:243], v243 offset0:224 offset1:240
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[244:245], v0 offset1:1
	ds_load_2addr_b32 v[234:235], v1 offset1:1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[0:3], v24 /*v280*/ offset:128
	ds_load_b128 v[4:7], v24 /*v280*/ offset:160
	ds_load_b128 v[8:11], v24 /*v280*/ offset:192
	ds_load_b128 v[12:15], v24 /*v280*/ offset:224
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[192:199], v[16:31], v[128:135], v226, v228 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[200:207], v[16:31], v[136:143], v227, v228 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[208:215], v[16:31], v[144:151], v230, v228 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[216:223], v[16:31], v[152:159], v231, v228 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[216:223], v[48:63], v[184:191], v231, v229 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[208:215], v[48:63], v[176:183], v230, v229 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[200:207], v[48:63], v[168:175], v227, v229 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[192:199], v[48:63], v[160:167], v226, v229 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[16:19], v24 /*v280*/ offset:4480
	ds_load_b128 v[20:23], v24 /*v280*/ offset:4512
	ds_load_b128 v[24:27], v24 /*v280*/ offset:4544
	ds_load_b128 v[28:31], v24 /*v280*/ offset:4576
	ds_load_b128 v[32:35], v24 /*v280*/ offset:8832
	ds_load_b128 v[36:39], v24 /*v280*/ offset:8864
	ds_load_b128 v[40:43], v24 /*v280*/ offset:8896
	ds_load_b128 v[44:47], v24 /*v280*/ offset:8928
	ds_load_b128 v[48:51], v24 /*v280*/ offset:13184
	ds_load_b128 v[52:55], v24 /*v280*/ offset:13216
	ds_load_b128 v[56:59], v24 /*v280*/ offset:13248
	ds_load_b128 v[60:63], v24 /*v280*/ offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[246:253], v[0:15], v[64:71], v232, v244 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[254:261], v[0:15], v[72:79], v233, v244 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[6:13] /*v[262:269]*/, v[0:15], v[80:87], v242, v244 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[14:21] /*v[270:277]*/, v[0:15], v[88:95], v243, v244 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[14:21] /*v[270:277]*/, v[16:31], v[120:127], v243, v245 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[6:13] /*v[262:269]*/, v[16:31], v[112:119], v242, v245 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[254:261], v[16:31], v[104:111], v233, v245 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[246:253], v[16:31], v[96:103], v232, v245 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[246:253], v[32:47], v[128:135], v232, v234 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[254:261], v[32:47], v[136:143], v233, v234 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[6:13] /*v[262:269]*/, v[32:47], v[144:151], v242, v234 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[14:21] /*v[270:277]*/, v[32:47], v[152:159], v243, v234 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[14:21] /*v[270:277]*/, v[48:63], v[184:191], v243, v235 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[6:13] /*v[262:269]*/, v[48:63], v[176:183], v242, v235 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[254:261], v[48:63], v[168:175], v233, v235 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[246:253], v[48:63], v[160:167], v232, v235 matrix_a_fmt:MATRIX_FMT_FP4
	s_add_nc_u64 s[56:57], s[56:57], 0x100
	s_add_co_i32 s72, s72, 1
	s_add_nc_u64 s[52:53], s[52:53], 32
	s_cmp_lg_u32 s56, 0x1a00
	s_add_nc_u64 s[58:59], s[58:59], 0x800
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
	s_add_co_i32 s80, s9, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_add_nc_u32 v244, s80, v239 :: v_dual_add_nc_u32 v0, s80, v237
	v_add_nc_u32_e32 v242, s80, v238
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[192:195], v244
	ds_load_b128 v[196:199], v244 offset:512
	ds_load_b128 v[200:203], v244 offset:2048
	ds_load_b128 v[204:207], v244 offset:2560
	ds_load_b128 v[208:211], v244 offset:4096
	ds_load_b128 v[212:215], v244 offset:4608
	v_add_nc_u32_e32 v243, 0xc400, v0
	v_add_nc_u32_e32 v0, 0xc400, v242
	v_add_nc_u32_e32 v1, 0xc408, v242
	ds_load_b128 v[216:219], v244 offset:6144
	ds_load_b128 v[220:223], v244 offset:6656
	s_wait_alu depctr_va_vdst(2)
	ds_load_2addr_b32 v[226:227], v243 offset0:128 offset1:144
	ds_load_2addr_b32 v[230:231], v243 offset0:192 offset1:208
	s_wait_alu depctr_va_vdst(1)
	ds_load_2addr_b32 v[232:233], v0 offset1:1
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[228:229], v1 offset1:1
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
	s_add_co_i32 s85, s61, 0x2200
	s_add_co_i32 s86, s14, 0x38200
	s_or_b32 s87, s9, 0x80000000
	s_mov_b32 s84, s60
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[36:43]
.LBB0_31:
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v0, s58, v240
	s_and_b32 vcc_lo, exec_lo, s4
	s_add_nc_u64 s[8:9], s[68:69], s[58:59]
	s_cbranch_vccnz .LBB0_33
	s_add_nc_u64 s[14:15], s[8:9], 0x1000
	s_add_co_i32 s14, s61, 0x4400
	s_bitset1_b32 s15, 31
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v1, s14 :: v_dual_add_nc_u32 v2, 0x1000, v0
	v_mov_b32_e32 v3, s15
	v_readfirstlane_b32 s84, v224
	s_mov_b32 s22, s16
	v_readfirstlane_b32 s85, v1
	v_readfirstlane_b32 s86, v2
	v_readfirstlane_b32 s87, v3
	s_mov_b32 s23, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[16:23]
.LBB0_33:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_35
	s_add_nc_u64 s[8:9], s[8:9], 0x71000
	s_add_co_i32 s8, s61, 0x8400
	s_bitset1_b32 s9, 31
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v1, s8 :: v_dual_add_nc_u32 v2, 0x71000, v0
	v_mov_b32_e32 v3, s9
	v_readfirstlane_b32 s84, v224
	s_mov_b32 s22, s16
	v_readfirstlane_b32 s85, v1
	v_readfirstlane_b32 s86, v2
	v_readfirstlane_b32 s87, v3
	s_mov_b32 s23, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[16:23]
.LBB0_35:
	s_and_b32 vcc_lo, exec_lo, s2
	s_mul_i32 s81, s10, 0x3380
	s_cbranch_vccz .LBB0_43
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_38
.LBB0_37:
	s_lshl2_add_u32 s10, s81, 0
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
	v_add_nc_u32_e32 v0, s56, v241
	s_and_b32 vcc_lo, exec_lo, s4
	s_add_nc_u64 s[22:23], s[6:7], s[56:57]
	s_cbranch_vccnz .LBB0_40
	s_lshl2_add_u32 s10, s81, 0
	s_add_nc_u64 s[8:9], s[22:23], 0x200
	s_add_co_i32 s10, s10, 0xc600
	s_or_b32 s8, s9, 0x80000000
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v1, s10 :: v_dual_add_nc_u32 v2, 0x200, v0
	v_mov_b32_e32 v3, s8
	v_readfirstlane_b32 s84, v224
	s_mov_b32 s8, s44
	v_readfirstlane_b32 s86, v2
	v_readfirstlane_b32 s85, v1
	v_readfirstlane_b32 s87, v3
	s_mov_b32 s9, s17
	s_mov_b32 s10, s18
	s_mov_b32 s14, s16
	s_mov_b32 s15, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[8:15]
.LBB0_40:
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB0_27
	s_lshl2_add_u32 s10, s81, 0
	s_add_nc_u64 s[8:9], s[22:23], 0x7200
	s_add_co_i32 s10, s10, 0xca00
	s_or_b32 s8, s9, 0x80000000
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v1, s10 :: v_dual_add_nc_u32 v2, 0x7200, v0
	v_mov_b32_e32 v3, s8
	v_readfirstlane_b32 s84, v224
	s_mov_b32 s8, s44
	v_readfirstlane_b32 s86, v2
	v_readfirstlane_b32 s85, v1
	v_readfirstlane_b32 s87, v3
	s_mov_b32 s9, s17
	s_mov_b32 s10, s18
	s_mov_b32 s14, s16
	s_mov_b32 s15, s16
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[8:15]
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
	s_lshl2_add_u32 s10, s81, 0
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
	s_add_co_i32 s2, 0, 0x19c00
	s_add_co_i32 s3, 0, 0x26200
	v_dual_add_nc_u32 v224, s2, v239 :: v_dual_add_nc_u32 v232, s2, v238
	v_add_nc_u32_e32 v234, s2, v236
	s_lshl_b32 s2, s35, 1
	v_add_nc_u32_e32 v233, s3, v237
	s_or_b32 s2, s2, 64
	v_add_nc_u32_e32 v230, 0xc400, v232
	s_set_vgpr_msb 16
	v_and_or_b32 v16, 0x1c0, s2, v23 /*v279*/
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_add_u32 v235, v16, 2, s3
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[44:47], v224
	ds_load_b128 v[48:51], v224 offset:512
	ds_load_2addr_b32 v[62:63], v233 offset1:16
	ds_load_2addr_b32 v[208:209], v230 offset1:1
	ds_load_b128 v[0:3], v234
	ds_load_b128 v[4:7], v234 offset:32
	ds_load_b128 v[8:11], v234 offset:64
	ds_load_b128 v[12:15], v234 offset:96
	ds_load_b128 v[52:55], v224 offset:2048
	ds_load_b128 v[56:59], v224 offset:2560
	ds_load_b128 v[192:195], v224 offset:4096
	ds_load_b128 v[196:199], v224 offset:4608
	ds_load_b128 v[200:203], v224 offset:6144
	ds_load_b128 v[204:207], v224 offset:6656
	ds_load_b32 v60, v233 offset:256
	ds_load_b32 v61, v235 offset:64
	ds_load_b128 v[28:31], v234 offset:4352
	ds_load_b128 v[32:35], v234 offset:4384
	ds_load_b128 v[36:39], v234 offset:4416
	ds_load_b128 v[40:43], v234 offset:4448
	ds_load_2addr_b32 v[220:221], v230 offset0:2 offset1:3
	ds_load_b128 v[16:19], v234 offset:8704
	ds_load_b128 v[20:23], v234 offset:8736
	ds_load_b128 v[24:27], v234 offset:8768
	s_set_vgpr_msb 0x1000
	s_wait_dscnt 0x10
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[44:51], v[0:15], v[64:71], v62, v208 matrix_a_fmt:MATRIX_FMT_FP4
	s_load_b64 s[78:79], s[0:1], 0x0 nv
	s_mov_b32 s71, 0
	s_mov_b32 s72, 1
	s_wait_dscnt 0xe
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[52:59], v[0:15], v[72:79], v63, v208 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x9
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[192:199], v[0:15], v[80:87], v60, v208 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[200:207], v[0:15], v[88:95], v61, v208 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[200:207], v[28:43], v[120:127], v61, v209 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[192:199], v[28:43], v[112:119], v60, v209 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[52:59], v[28:43], v[104:111], v63, v209 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[44:51], v[28:43], v[96:103], v62, v209 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[28:31], v234 offset:8800
	ds_load_b128 v[0:3], v234 offset:13056
	ds_load_b128 v[4:7], v234 offset:13088
	ds_load_b128 v[8:11], v234 offset:13120
	ds_load_b128 v[12:15], v234 offset:13152
	s_wait_dscnt 0x0
	ds_load_b128 v[208:211], v224 offset:1024
	ds_load_b128 v[212:215], v224 offset:1536
	ds_load_b128 v[216:219], v224 offset:3072
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[44:51], v[16:31], v[128:135], v62, v220 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[52:59], v[16:31], v[136:143], v63, v220 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[192:199], v[16:31], v[144:151], v60, v220 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[200:207], v[16:31], v[152:159], v61, v220 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[200:207], v[0:15], v[184:191], v61, v221 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[192:199], v[0:15], v[176:183], v60, v221 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[52:59], v[0:15], v[168:175], v63, v221 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[44:51], v[0:15], v[160:167], v62, v221 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_2addr_b32 v[226:227], v233 offset0:32 offset1:48
	ds_load_2addr_b32 v[44:45], v230 offset0:4 offset1:5
	ds_load_b128 v[0:3], v234 offset:128
	ds_load_b128 v[4:7], v234 offset:160
	ds_load_b128 v[8:11], v234 offset:192
	ds_load_b128 v[12:15], v234 offset:224
	ds_load_b128 v[220:223], v224 offset:3584
	ds_load_b128 v[192:195], v224 offset:5120
	ds_load_b128 v[196:199], v224 offset:5632
	ds_load_b128 v[200:203], v224 offset:7168
	ds_load_b128 v[204:207], v224 offset:7680
	ds_load_2addr_b32 v[228:229], v233 offset0:96 offset1:112
	s_wait_alu depctr_vm_vsrc(6)
	ds_load_2addr_b32 v[230:231], v230 offset0:6 offset1:7
	ds_load_b128 v[16:19], v234 offset:4480
	ds_load_b128 v[20:23], v234 offset:4512
	ds_load_b128 v[24:27], v234 offset:4544
	ds_load_b128 v[28:31], v234 offset:4576
	ds_load_b128 v[48:51], v234 offset:8832
	ds_load_b128 v[52:55], v234 offset:8864
	ds_load_b128 v[56:59], v234 offset:8896
	ds_load_b128 v[60:63], v234 offset:8928
	ds_load_b128 v[32:35], v234 offset:13184
	ds_load_b128 v[36:39], v234 offset:13216
	ds_load_b128 v[40:43], v234 offset:13248
	s_wait_dscnt 0x12
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[208:215], v[0:15], v[64:71], v226, v44 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x11
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[216:223], v[0:15], v[72:79], v227, v44 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xc
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[192:199], v[0:15], v[80:87], v228, v44 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[200:207], v[0:15], v[88:95], v229, v44 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x7
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[200:207], v[16:31], v[120:127], v229, v45 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[192:199], v[16:31], v[112:119], v228, v45 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[216:223], v[16:31], v[104:111], v227, v45 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[208:215], v[16:31], v[96:103], v226, v45 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[44:47], v234 offset:13280
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[208:215], v[48:63], v[128:135], v226, v230 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[216:223], v[48:63], v[136:143], v227, v230 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[192:199], v[48:63], v[144:151], v228, v230 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[200:207], v[48:63], v[152:159], v229, v230 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[200:207], v[32:47], v[184:191], v229, v231 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[192:199], v[32:47], v[176:183], v228, v231 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[216:223], v[32:47], v[168:175], v227, v231 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[208:215], v[32:47], v[160:167], v226, v231 matrix_a_fmt:MATRIX_FMT_FP4
	v_add_nc_u32_e32 v0, 0xfffe6400, v224
	v_add_nc_u32_e32 v1, 0xfffe6600, v224
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_add_nc_u32_e32 v2, 0xfffe6c00, v224
	v_add_nc_u32_e32 v3, 0xfffe6e00, v224
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[32:35], v0
	ds_load_b128 v[36:39], v1
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v0, 0xfffe7400, v224
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0xfffe7600, v224
	v_add_nc_u32_e32 v4, 0xfffe6400, v233
	ds_load_b128 v[40:43], v2
	ds_load_b128 v[44:47], v3
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v2, 0xfffe7c00, v224
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, 0xfffe7e00, v224
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[48:51], v0
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[52:55], v1
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v192, v4
	v_add_nc_u32_e32 v16, 0xfffe6440, v233
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v0, 0xffff2800, v232
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v1, 0xfffe6400, v234
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v4, 0xfffe6420, v234
	v_add_nc_u32_e32 v8, 0xfffe6440, v234
	v_add_nc_u32_e32 v12, 0xfffe6460, v234
	v_add_nc_u32_e32 v17, 0xfffe6500, v233
	v_add_nc_u32_e32 v18, 0xfffe6440, v235
	v_add_nc_u32_e32 v20, 0xffff2808, v232
	s_wait_alu depctr_va_vdst(10)
	ds_load_b128 v[56:59], v2
	s_wait_alu depctr_va_vdst(9)
	ds_load_b128 v[60:63], v3
	s_wait_alu depctr_va_vdst(7)
	ds_load_b32 v193, v0
	s_wait_alu depctr_va_vdst(6) depctr_vm_vsrc(0)
	ds_load_b128 v[0:3], v1
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[4:7], v4
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[8:11], v8
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[12:15], v12
	v_add_nc_u32_e32 v19, 0xffff2804, v232
	ds_load_b32 v194, v16
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v195, v17
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v196, v18
	s_wait_alu depctr_va_vdst(1)
	ds_load_b32 v198, v20
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v16, 0xffff280c, v232
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v17, 0xfffe7500, v234
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v20, 0xfffe7520, v234
	v_add_nc_u32_e32 v24, 0xfffe7540, v234
	v_add_nc_u32_e32 v28, 0xfffe7560, v234
	s_wait_alu depctr_va_vdst(5)
	ds_load_b32 v197, v19
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v199, v16
	s_wait_alu depctr_va_vdst(3) depctr_vm_vsrc(0)
	ds_load_b128 v[16:19], v17
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[20:23], v20
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[24:27], v24
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[28:31], v28
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[32:39], v[0:15], v[64:71], v192, v193 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x9
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[40:47], v[0:15], v[72:79], v194, v193 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[48:55], v[0:15], v[80:87], v195, v193 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x7
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[56:63], v[0:15], v[88:95], v196, v193 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v0, 0xfffe8600, v234
	v_add_nc_u32_e32 v1, 0xfffe8620, v234
	v_add_nc_u32_e32 v2, 0xfffe8640, v234
	v_add_nc_u32_e32 v3, 0xfffe8660, v234
	v_add_nc_u32_e32 v4, 0xfffe9720, v234
	v_add_nc_u32_e32 v8, 0xfffe9740, v234
	v_add_nc_u32_e32 v12, 0xfffe9760, v234
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[56:63], v[16:31], v[120:127], v196, v197 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[48:55], v[16:31], v[112:119], v195, v197 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[40:47], v[16:31], v[104:111], v194, v197 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[32:39], v[16:31], v[96:103], v192, v197 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(3)
	ds_load_b128 v[16:19], v0
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v0, 0xfffe9700, v234
	ds_load_b128 v[20:23], v1
	ds_load_b128 v[24:27], v2
	ds_load_b128 v[28:31], v3
	ds_load_b128 v[4:7], v4
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(1)
	ds_load_b128 v[0:3], v0
	ds_load_b128 v[8:11], v8
	ds_load_b128 v[12:15], v12
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[56:63], v[0:15], v[184:191], v196, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[48:55], v[0:15], v[176:183], v195, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[40:47], v[0:15], v[168:175], v194, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[32:39], v[0:15], v[160:167], v192, v199 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_alu depctr_vm_vsrc(2)
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v0, 0xfffe6800, v224
	v_add_nc_u32_e32 v1, 0xfffe6a00, v224
	v_add_nc_u32_e32 v4, 0xfffe7800, v224
	v_add_nc_u32_e32 v2, 0xfffe7000, v224
	v_add_nc_u32_e32 v3, 0xfffe7200, v224
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v8, 0xfffe64c0, v234
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v12, 0xfffe64e0, v234
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[32:39], v[16:31], v[128:135], v192, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[40:47], v[16:31], v[136:143], v194, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[48:55], v[16:31], v[144:151], v195, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[56:63], v[16:31], v[152:159], v196, v198 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[32:35], v0
	ds_load_b128 v[36:39], v1
	ds_load_b128 v[48:51], v4
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v0, 0xfffe7a00, v224
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v1, 0xfffe8000, v224
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v4, 0xfffe64c0, v233
	ds_load_b128 v[40:43], v2
	ds_load_b128 v[44:47], v3
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v2, 0xfffe8200, v224
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, 0xfffe6480, v233
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[52:55], v0
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[56:59], v1
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v193, v4
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v0, 0xfffe6580, v233
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v1, 0xfffe65c0, v233
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v4, 0xffff2818, v232
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[60:63], v2
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v192, v3
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v2, 0xffff2810, v232
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v3, 0xffff2814, v232
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v194, v0
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v195, v1
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v198, v4
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v0, 0xffff281c, v232
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v1, 0xfffe6480, v234
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v4, 0xfffe64a0, v234
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v196, v2
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v197, v3
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v199, v0
	s_wait_alu depctr_va_vdst(1) depctr_vm_vsrc(0)
	ds_load_b128 v[0:3], v1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[4:7], v4
	ds_load_b128 v[8:11], v8
	ds_load_b128 v[12:15], v12
	v_add_nc_u32_e32 v16, 0xfffe7580, v234
	v_add_nc_u32_e32 v20, 0xfffe75a0, v234
	v_add_nc_u32_e32 v24, 0xfffe75c0, v234
	v_add_nc_u32_e32 v28, 0xfffe75e0, v234
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[16:19], v16
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[20:23], v20
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[24:27], v24
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[28:31], v28
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[32:39], v[0:15], v[64:71], v192, v196 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[40:47], v[0:15], v[72:79], v193, v196 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[48:55], v[0:15], v[80:87], v194, v196 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[56:63], v[0:15], v[88:95], v195, v196 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v0, 0xfffe8680, v234
	s_wait_alu depctr_vm_vsrc(6)
	v_add_nc_u32_e32 v1, 0xfffe86a0, v234
	v_add_nc_u32_e32 v2, 0xfffe86c0, v234
	v_add_nc_u32_e32 v3, 0xfffe86e0, v234
	v_add_nc_u32_e32 v4, 0xfffe97a0, v234
	s_wait_alu depctr_vm_vsrc(5)
	v_add_nc_u32_e32 v8, 0xfffe97c0, v234
	s_wait_alu depctr_vm_vsrc(4)
	v_add_nc_u32_e32 v12, 0xfffe97e0, v234
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[56:63], v[16:31], v[120:127], v195, v197 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[48:55], v[16:31], v[112:119], v194, v197 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[40:47], v[16:31], v[104:111], v193, v197 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[32:39], v[16:31], v[96:103], v192, v197 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(3)
	ds_load_b128 v[16:19], v0
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v0, 0xfffe9780, v234
	ds_load_b128 v[20:23], v1
	ds_load_b128 v[24:27], v2
	ds_load_b128 v[28:31], v3
	ds_load_b128 v[4:7], v4
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(1)
	ds_load_b128 v[0:3], v0
	ds_load_b128 v[8:11], v8
	ds_load_b128 v[12:15], v12
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[32:39], v[16:31], v[128:135], v192, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[40:47], v[16:31], v[136:143], v193, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[48:55], v[16:31], v[144:151], v194, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[56:63], v[16:31], v[152:159], v195, v198 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[56:63], v[0:15], v[184:191], v195, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[48:55], v[0:15], v[176:183], v194, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[40:47], v[0:15], v[168:175], v193, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[32:39], v[0:15], v[160:167], v192, v199 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 1
	v_lshl_or_b32 v18, v22 /*v278*/, 3, s35
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_vm_vsrc(2)
	v_nop
	v_nop
	v_nop
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
	s_clause 0x1
	global_load_b128 v[0:3], v[16:17], off
	global_load_b128 v[4:7], v[16:17], off offset:32
	s_wait_alu depctr_vm_vsrc(0)
	s_clause 0x1
	global_load_b128 v[8:11], v[16:17], off offset:64
	global_load_b128 v[12:15], v[16:17], off offset:96
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_or_b32 v16, v225, 8, v18
	s_lshl_b32 s70, s2, 5
	s_add_co_i32 s73, s3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v192, 0, v16
	v_cndmask_b32_e64 v16, 0, 1, s1
	s_lshl_b32 s1, s2, 4
	s_sub_co_i32 s1, s33, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v193, 0x1000, v192
	v_readfirstlane_b32 s80, v16
	s_max_i32 s75, s1, 0
	v_add_nc_u32_e32 v194, 0x2000, v192
	v_add_nc_u32_e32 v195, 0x3000, v192
	s_wait_loadcnt 0x3
	v_and_b32_e32 v19, 0xffff0000, v1
	v_lshlrev_b32_e32 v18, 16, v1
	v_and_b32_e32 v17, 0xffff0000, v0
	v_lshlrev_b32_e32 v16, 16, v0
	v_and_b32_e32 v1, 0xffff0000, v3
	v_lshlrev_b32_e32 v0, 16, v3
	v_pk_add_f32 v[36:37], v[66:67], v[18:19]
	v_and_b32_e32 v3, 0xffff0000, v2
	v_pk_add_f32 v[38:39], v[64:65], v[16:17]
	v_lshlrev_b32_e32 v2, 16, v2
	v_pk_add_f32 v[66:67], v[102:103], v[0:1]
	v_cmp_gt_f32_e32 vcc_lo, s74, v36
	v_pk_add_f32 v[102:103], v[128:129], v[16:17]
	v_pk_add_f32 v[34:35], v[70:71], v[0:1]
	v_pk_add_f32 v[64:65], v[100:101], v[2:3]
	v_pk_add_f32 v[100:101], v[130:131], v[18:19]
	v_cndmask_b32_e32 v129, s74, v36, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v38
	v_pk_add_f32 v[32:33], v[68:69], v[2:3]
	s_wait_loadcnt 0x2
	v_and_b32_e32 v23, 0xffff0000, v5
	v_lshlrev_b32_e32 v22, 16, v5
	v_and_b32_e32 v21, 0xffff0000, v4
	v_cndmask_b32_e32 v128, s74, v38, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v37, -s74
	v_lshlrev_b32_e32 v20, 16, v4
	v_pk_add_f32 v[44:45], v[74:75], v[22:23]
	v_pk_add_f32 v[70:71], v[96:97], v[16:17]
	v_pk_add_f32 v[96:97], v[132:133], v[2:3]
	v_cndmask_b32_e32 v130, s0, v37, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v39, -s74
	v_pk_add_f32 v[46:47], v[72:73], v[20:21]
	v_and_b32_e32 v5, 0xffff0000, v7
	v_lshlrev_b32_e32 v4, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v6
	v_cndmask_b32_e32 v131, s0, v39, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v34
	v_lshlrev_b32_e32 v6, 16, v6
	v_pk_add_f32 v[42:43], v[78:79], v[4:5]
	v_pk_add_f32 v[68:69], v[98:99], v[18:19]
	v_pk_add_f32 v[98:99], v[134:135], v[0:1]
	v_cndmask_b32_e32 v37, s74, v34, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v32
	v_pk_add_f32 v[40:41], v[76:77], v[6:7]
	s_wait_loadcnt 0x1
	v_and_b32_e32 v27, 0xffff0000, v9
	v_lshlrev_b32_e32 v26, 16, v9
	v_and_b32_e32 v25, 0xffff0000, v8
	v_cndmask_b32_e32 v36, s74, v32, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v35, -s74
	v_lshlrev_b32_e32 v24, 16, v8
	v_pk_add_f32 v[52:53], v[82:83], v[26:27]
	v_pk_add_f32 v[74:75], v[110:111], v[4:5]
	v_pk_add_f32 v[110:111], v[136:137], v[20:21]
	v_cndmask_b32_e32 v132, s0, v35, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v33, -s74
	v_pk_add_f32 v[54:55], v[80:81], v[24:25]
	v_and_b32_e32 v9, 0xffff0000, v11
	v_lshlrev_b32_e32 v8, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v10
	v_cndmask_b32_e32 v133, s0, v33, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v44
	v_lshlrev_b32_e32 v10, 16, v10
	v_pk_add_f32 v[50:51], v[86:87], v[8:9]
	v_pk_add_f32 v[72:73], v[108:109], v[6:7]
	v_pk_add_f32 v[108:109], v[138:139], v[22:23]
	v_cndmask_b32_e32 v33, s74, v44, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v46
	v_pk_add_f32 v[48:49], v[84:85], v[10:11]
	s_wait_loadcnt 0x0
	v_and_b32_e32 v31, 0xffff0000, v13
	v_lshlrev_b32_e32 v30, 16, v13
	v_and_b32_e32 v29, 0xffff0000, v12
	v_cndmask_b32_e32 v32, s74, v46, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v45, -s74
	v_lshlrev_b32_e32 v28, 16, v12
	v_pk_add_f32 v[60:61], v[90:91], v[30:31]
	v_pk_add_f32 v[78:79], v[104:105], v[20:21]
	v_pk_add_f32 v[104:105], v[140:141], v[6:7]
	v_cndmask_b32_e32 v134, s0, v45, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v47, -s74
	v_pk_add_f32 v[62:63], v[88:89], v[28:29]
	v_and_b32_e32 v13, 0xffff0000, v15
	v_lshlrev_b32_e32 v12, 16, v15
	v_and_b32_e32 v15, 0xffff0000, v14
	v_cndmask_b32_e32 v135, s0, v47, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v42
	v_lshlrev_b32_e32 v14, 16, v14
	v_pk_add_f32 v[58:59], v[94:95], v[12:13]
	v_pk_add_f32 v[76:77], v[106:107], v[22:23]
	v_pk_add_f32 v[106:107], v[142:143], v[4:5]
	v_cndmask_b32_e32 v35, s74, v42, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v40
	v_pk_add_f32 v[56:57], v[92:93], v[14:15]
	v_pk_add_f32 v[82:83], v[118:119], v[8:9]
	v_pk_add_f32 v[118:119], v[144:145], v[24:25]
	v_pk_add_f32 v[80:81], v[116:117], v[10:11]
	v_cndmask_b32_e32 v34, s74, v40, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v43, -s74
	v_pk_add_f32 v[116:117], v[146:147], v[26:27]
	v_pk_add_f32 v[86:87], v[112:113], v[24:25]
	v_pk_add_f32 v[112:113], v[148:149], v[10:11]
	v_pk_add_f32 v[84:85], v[114:115], v[26:27]
	v_cndmask_b32_e32 v136, s0, v43, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v41, -s74
	v_pk_add_f32 v[114:115], v[150:151], v[8:9]
	v_pk_add_f32 v[90:91], v[126:127], v[12:13]
	v_pk_add_f32 v[126:127], v[152:153], v[28:29]
	v_pk_add_f32 v[88:89], v[124:125], v[14:15]
	v_cndmask_b32_e32 v137, s0, v41, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v52
	v_pk_add_f32 v[124:125], v[154:155], v[30:31]
	v_pk_add_f32 v[92:93], v[122:123], v[30:31]
	v_pk_add_f32 v[94:95], v[120:121], v[28:29]
	v_pk_add_f32 v[120:121], v[156:157], v[14:15]
	v_cndmask_b32_e32 v39, s74, v52, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v54
	v_pk_add_f32 v[122:123], v[158:159], v[12:13]
	v_pk_add_f32 v[18:19], v[162:163], v[18:19]
	v_pk_add_f32 v[16:17], v[160:161], v[16:17]
	v_pk_add_f32 v[0:1], v[166:167], v[0:1]
	v_cndmask_b32_e32 v38, s74, v54, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v53, -s74
	v_pk_add_f32 v[2:3], v[164:165], v[2:3]
	v_pk_add_f32 v[22:23], v[170:171], v[22:23]
	v_pk_add_f32 v[20:21], v[168:169], v[20:21]
	v_pk_add_f32 v[4:5], v[174:175], v[4:5]
	v_cndmask_b32_e32 v138, s0, v53, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v55, -s74
	v_pk_add_f32 v[6:7], v[172:173], v[6:7]
	v_pk_add_f32 v[26:27], v[178:179], v[26:27]
	v_pk_add_f32 v[24:25], v[176:177], v[24:25]
	v_pk_add_f32 v[8:9], v[182:183], v[8:9]
	v_cndmask_b32_e32 v139, s0, v55, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v50
	v_pk_add_f32 v[10:11], v[180:181], v[10:11]
	v_pk_add_f32 v[30:31], v[186:187], v[30:31]
	v_pk_add_f32 v[28:29], v[184:185], v[28:29]
	v_pk_add_f32 v[12:13], v[190:191], v[12:13]
	v_cndmask_b32_e32 v41, s74, v50, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v48
	v_pk_add_f32 v[14:15], v[188:189], v[14:15]
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
	v_cndmask_b32_e64 v199, 0, 0x42800000, s2
	v_cndmask_b32_e32 v43, s74, v60, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v62
	v_add_f32_e32 v161, v161, v199
	v_cndmask_b32_e64 v199, 0, 0xffffffc0, s2
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
	v_cndmask_b32_e64 v227, 0, 0x42800000, s30
	v_cndmask_b32_e32 v95, s0, v89, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v100
	v_add_f32_e32 v162, v162, v227
	v_cndmask_b32_e64 v227, 0, 0xffffffc0, s30
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
	v_cndmask_b32_e64 v229, 0, 0x42800000, s33
	v_cmp_gt_f32_e64 s31, 0xc2fc0000, v163
	v_cndmask_b32_e32 v101, s0, v103, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v98
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v164, v164, v229
	v_cndmask_b32_e64 v228, 0, 0x42800000, s31
	v_cndmask_b32_e64 v229, 0, 0xffffffc0, s33
	v_cndmask_b32_e32 v65, s74, v98, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v163, v163, v228
	v_exp_f32_e32 v164, v164
	v_cndmask_b32_e64 v228, 0, 0xffffffc0, s31
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
	v_cndmask_b32_e64 v231, 0, 0x42800000, s35
	v_cmp_gt_f32_e64 s34, 0xc2fc0000, v165
	v_cndmask_b32_e32 v98, s0, v97, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v108
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v166, v166, v231
	v_cndmask_b32_e64 v230, 0, 0x42800000, s34
	v_cndmask_b32_e64 v231, 0, 0xffffffc0, s35
	v_cndmask_b32_e32 v67, s74, v108, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v110
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v165, v165, v230
	v_exp_f32_e32 v166, v166
	v_cndmask_b32_e64 v230, 0, 0xffffffc0, s34
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
	v_cndmask_b32_e64 v233, 0, 0x42800000, s37
	v_cmp_gt_f32_e64 s36, 0xc2fc0000, v167
	v_cndmask_b32_e32 v102, s0, v111, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v106
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v168, v168, v233
	v_cndmask_b32_e64 v232, 0, 0x42800000, s36
	v_cndmask_b32_e64 v233, 0, 0xffffffc0, s37
	v_cndmask_b32_e32 v69, s74, v106, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v167, v167, v232
	v_exp_f32_e32 v168, v168
	v_cndmask_b32_e64 v232, 0, 0xffffffc0, s36
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
	v_cndmask_b32_e64 v235, 0, 0x42800000, s39
	v_cmp_gt_f32_e64 s38, 0xc2fc0000, v169
	v_cndmask_b32_e32 v104, s0, v105, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v116
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v170, v170, v235
	v_cndmask_b32_e64 v234, 0, 0x42800000, s38
	v_cndmask_b32_e64 v235, 0, 0xffffffc0, s39
	v_cndmask_b32_e32 v71, s74, v116, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v169, v169, v234
	v_exp_f32_e32 v170, v170
	v_cndmask_b32_e64 v234, 0, 0xffffffc0, s38
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
	v_cndmask_b32_e64 v237, 0, 0x42800000, s41
	v_cmp_gt_f32_e64 s40, 0xc2fc0000, v171
	v_cndmask_b32_e32 v106, s0, v119, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v114
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v172, v172, v237
	v_cndmask_b32_e64 v236, 0, 0x42800000, s40
	v_cndmask_b32_e64 v237, 0, 0xffffffc0, s41
	v_cndmask_b32_e32 v73, s74, v114, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v171, v171, v236
	v_exp_f32_e32 v172, v172
	v_cndmask_b32_e64 v236, 0, 0xffffffc0, s40
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
	v_cndmask_b32_e64 v239, 0, 0x42800000, s43
	v_cmp_gt_f32_e64 s42, 0xc2fc0000, v173
	v_cndmask_b32_e32 v108, s0, v113, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v124
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v174, v174, v239
	v_cndmask_b32_e64 v238, 0, 0x42800000, s42
	v_cndmask_b32_e64 v239, 0, 0xffffffc0, s43
	v_cndmask_b32_e32 v75, s74, v124, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v173, v173, v238
	v_exp_f32_e32 v174, v174
	v_cndmask_b32_e64 v238, 0, 0xffffffc0, s42
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
	v_cndmask_b32_e64 v241, 0, 0x42800000, s45
	v_cmp_gt_f32_e64 s44, 0xc2fc0000, v175
	v_cndmask_b32_e32 v110, s0, v127, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v122
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v176, v176, v241
	v_cndmask_b32_e64 v240, 0, 0x42800000, s44
	v_cndmask_b32_e64 v241, 0, 0xffffffc0, s45
	v_cndmask_b32_e32 v77, s74, v122, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v175, v175, v240
	v_exp_f32_e32 v176, v176
	v_cndmask_b32_e64 v240, 0, 0xffffffc0, s44
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
	v_cndmask_b32_e64 v243, 0, 0x42800000, s47
	v_cmp_gt_f32_e64 s46, 0xc2fc0000, v177
	v_cndmask_b32_e32 v112, s0, v121, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v178, v178, v243
	v_cndmask_b32_e64 v242, 0, 0x42800000, s46
	v_cndmask_b32_e64 v243, 0, 0xffffffc0, s47
	v_cndmask_b32_e32 v79, s74, v18, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v16
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v177, v177, v242
	v_exp_f32_e32 v178, v178
	v_cndmask_b32_e64 v242, 0, 0xffffffc0, s46
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
	v_cndmask_b32_e64 v245, 0, 0x42800000, s49
	v_cmp_gt_f32_e64 s48, 0xc2fc0000, v179
	v_cndmask_b32_e32 v114, s0, v17, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v180, v180, v245
	v_cndmask_b32_e64 v244, 0, 0x42800000, s48
	v_cndmask_b32_e64 v245, 0, 0xffffffc0, s49
	v_cndmask_b32_e32 v17, s74, v0, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v179, v179, v244
	v_exp_f32_e32 v180, v180
	v_cndmask_b32_e64 v244, 0, 0xffffffc0, s48
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
	v_cndmask_b32_e64 v247, 0, 0x42800000, s51
	v_cmp_gt_f32_e64 s50, 0xc2fc0000, v181
	v_cndmask_b32_e32 v116, s0, v3, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v182, v182, v247
	v_cndmask_b32_e64 v246, 0, 0x42800000, s50
	v_cndmask_b32_e64 v247, 0, 0xffffffc0, s51
	v_cndmask_b32_e32 v1, s74, v22, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v181, v181, v246
	v_exp_f32_e32 v182, v182
	v_cndmask_b32_e64 v246, 0, 0xffffffc0, s50
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
	v_cndmask_b32_e64 v249, 0, 0x42800000, s53
	v_cmp_gt_f32_e64 s52, 0xc2fc0000, v183
	v_cndmask_b32_e32 v118, s0, v21, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v184, v184, v249
	v_cndmask_b32_e64 v248, 0, 0x42800000, s52
	v_cndmask_b32_e64 v249, 0, 0xffffffc0, s53
	v_cndmask_b32_e32 v3, s74, v4, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v6
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v183, v183, v248
	v_exp_f32_e32 v184, v184
	v_cndmask_b32_e64 v248, 0, 0xffffffc0, s52
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
	v_cndmask_b32_e64 v251, 0, 0x42800000, s55
	v_cmp_gt_f32_e64 s54, 0xc2fc0000, v185
	v_cndmask_b32_e32 v120, s0, v7, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v186, v186, v251
	v_cndmask_b32_e64 v250, 0, 0x42800000, s54
	v_cndmask_b32_e64 v251, 0, 0xffffffc0, s55
	v_cndmask_b32_e32 v5, s74, v26, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v185, v185, v250
	v_exp_f32_e32 v186, v186
	v_cndmask_b32_e64 v250, 0, 0xffffffc0, s54
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
	v_cndmask_b32_e64 v253, 0, 0x42800000, s57
	v_cmp_gt_f32_e64 s56, 0xc2fc0000, v187
	v_cndmask_b32_e32 v122, s0, v25, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v188, v188, v253
	v_cndmask_b32_e64 v252, 0, 0x42800000, s56
	v_cndmask_b32_e64 v253, 0, 0xffffffc0, s57
	v_cndmask_b32_e32 v7, s74, v8, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v187, v187, v252
	v_exp_f32_e32 v188, v188
	v_cndmask_b32_e64 v252, 0, 0xffffffc0, s56
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
	v_cndmask_b32_e64 v255, 0, 0x42800000, s59
	v_cmp_gt_f32_e64 s58, 0xc2fc0000, v189
	v_cndmask_b32_e32 v124, s0, v11, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v190, v190, v255
	v_cndmask_b32_e64 v254, 0, 0x42800000, s58
	v_cndmask_b32_e64 v255, 0, 0xffffffc0, s59
	v_cndmask_b32_e32 v9, s74, v30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v189, v189, v254
	v_exp_f32_e32 v190, v190
	v_cndmask_b32_e64 v254, 0, 0xffffffc0, s58
	v_mul_f32_e32 v196, 0xbfb8aa3b, v9
	v_cndmask_b32_e32 v8, s74, v28, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v31, -s74
	v_exp_f32_e32 v189, v189
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s61, 0xc2fc0000, v196
	v_mul_f32_e32 v191, 0xbfb8aa3b, v8
	v_cndmask_b32_e32 v125, s0, v31, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v29, -s74
	s_set_vgpr_msb 64
	v_cndmask_b32_e64 v1 /*v257*/, 0, 0x42800000, s61
	v_cmp_gt_f32_e64 s60, 0xc2fc0000, v191
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v126, s0, v29, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v12
	s_set_vgpr_msb 4
	v_add_f32_e32 v196, v196, v1 /*v257*/
	s_set_vgpr_msb 0x440
	v_cndmask_b32_e64 v0 /*v256*/, 0, 0x42800000, s60
	v_cndmask_b32_e64 v1 /*v257*/, 0, 0xffffffc0, s61
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v11, s74, v12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v14
	s_set_vgpr_msb 4
	v_add_f32_e32 v191, v191, v0 /*v256*/
	v_exp_f32_e32 v196, v196
	s_set_vgpr_msb 0x440
	v_cndmask_b32_e64 v0 /*v256*/, 0, 0xffffffc0, s60
	s_set_vgpr_msb 0x4000
	v_mul_f32_e32 v198, 0xbfb8aa3b, v11
	v_cndmask_b32_e32 v10, s74, v14, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v13, -s74
	v_exp_f32_e32 v191, v191
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_f32_e64 s63, 0xc2fc0000, v198
	v_mul_f32_e32 v197, 0xbfb8aa3b, v10
	v_cndmask_b32_e32 v127, s0, v13, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, v15, -s74
	s_set_vgpr_msb 64
	v_cndmask_b32_e64 v3 /*v259*/, 0, 0x42800000, s63
	v_cmp_gt_f32_e64 s62, 0xc2fc0000, v197
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v157, s0, v15, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v130
	v_cmp_gt_f32_e64 s0, 0xc2fc0000, v159
	s_set_vgpr_msb 64
	v_cndmask_b32_e64 v2 /*v258*/, 0, 0x42800000, s62
	s_set_vgpr_msb 0x4004
	v_add_f32_e32 v198, v198, v3 /*v259*/
	s_set_vgpr_msb 0x440
	v_cndmask_b32_e64 v3 /*v259*/, 0, 0xffffffc0, s63
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v13, s74, v130, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v132
	v_mul_f32_e32 v130, 0xbfb8aa3b, v32
	s_set_vgpr_msb 4
	v_add_f32_e32 v197, v197, v2 /*v258*/
	v_exp_f32_e32 v198, v198
	s_set_vgpr_msb 0x440
	v_cndmask_b32_e64 v2 /*v258*/, 0, 0xffffffc0, s62
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v15, s74, v132, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v133
	v_mul_f32_e32 v132, 0xbfb8aa3b, v34
	v_cmp_gt_f32_e64 s3, 0xc2fc0000, v130
	v_exp_f32_e32 v197, v197
	v_cndmask_b32_e32 v14, s74, v133, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v131
	v_mul_f32_e32 v133, 0xbfb8aa3b, v35
	v_cmp_gt_f32_e64 s5, 0xc2fc0000, v132
	v_cndmask_b32_e64 v200, 0, 0x42800000, s3
	v_cndmask_b32_e32 v12, s74, v131, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v134
	v_mul_f32_e32 v131, 0xbfb8aa3b, v33
	v_cmp_gt_f32_e64 s6, 0xc2fc0000, v133
	v_cndmask_b32_e64 v202, 0, 0x42800000, s5
	v_dual_add_f32 v130, v130, v200 :: v_dual_cndmask_b32 v19, s74, v134
	v_cmp_gt_f32_e32 vcc_lo, s74, v136
	v_mul_f32_e32 v134, 0xbfb8aa3b, v38
	v_cmp_gt_f32_e64 s4, 0xc2fc0000, v131
	v_cndmask_b32_e64 v203, 0, 0x42800000, s6
	v_dual_add_f32 v132, v132, v202 :: v_dual_cndmask_b32 v21, s74, v136
	v_cmp_gt_f32_e32 vcc_lo, s74, v137
	v_mul_f32_e32 v136, 0xbfb8aa3b, v40
	v_cmp_gt_f32_e64 s7, 0xc2fc0000, v134
	v_cndmask_b32_e64 v201, 0, 0x42800000, s4
	v_dual_add_f32 v133, v133, v203 :: v_dual_cndmask_b32 v20, s74, v137
	v_cmp_gt_f32_e32 vcc_lo, s74, v135
	v_mul_f32_e32 v137, 0xbfb8aa3b, v41
	v_cmp_gt_f32_e64 s9, 0xc2fc0000, v136
	v_cndmask_b32_e64 v204, 0, 0x42800000, s7
	v_dual_add_f32 v131, v131, v201 :: v_dual_cndmask_b32 v18, s74, v135
	v_cmp_gt_f32_e32 vcc_lo, s74, v138
	v_mul_f32_e32 v135, 0xbfb8aa3b, v39
	v_cmp_gt_f32_e64 s10, 0xc2fc0000, v137
	v_cndmask_b32_e64 v206, 0, 0x42800000, s9
	v_dual_add_f32 v134, v134, v204 :: v_dual_cndmask_b32 v23, s74, v138
	v_cmp_gt_f32_e32 vcc_lo, s74, v140
	v_mul_f32_e32 v138, 0xbfb8aa3b, v42
	v_cmp_gt_f32_e64 s8, 0xc2fc0000, v135
	v_cndmask_b32_e64 v207, 0, 0x42800000, s10
	v_dual_add_f32 v136, v136, v206 :: v_dual_cndmask_b32 v25, s74, v140
	v_cmp_gt_f32_e32 vcc_lo, s74, v141
	v_mul_f32_e32 v140, 0xbfb8aa3b, v44
	v_cmp_gt_f32_e64 s11, 0xc2fc0000, v138
	v_cndmask_b32_e64 v205, 0, 0x42800000, s8
	v_dual_add_f32 v137, v137, v207 :: v_dual_cndmask_b32 v24, s74, v141
	v_cmp_gt_f32_e32 vcc_lo, s74, v139
	v_mul_f32_e32 v141, 0xbfb8aa3b, v45
	v_cmp_gt_f32_e64 s13, 0xc2fc0000, v140
	v_cndmask_b32_e64 v208, 0, 0x42800000, s11
	v_dual_add_f32 v135, v135, v205 :: v_dual_cndmask_b32 v22, s74, v139
	v_cmp_gt_f32_e32 vcc_lo, s74, v142
	v_mul_f32_e32 v139, 0xbfb8aa3b, v43
	v_cmp_gt_f32_e64 s14, 0xc2fc0000, v141
	v_cndmask_b32_e64 v210, 0, 0x42800000, s13
	v_dual_add_f32 v138, v138, v208 :: v_dual_cndmask_b32 v27, s74, v142
	v_cmp_gt_f32_e32 vcc_lo, s74, v144
	v_mul_f32_e32 v142, 0xbfb8aa3b, v46
	v_cmp_gt_f32_e64 s12, 0xc2fc0000, v139
	v_cndmask_b32_e64 v211, 0, 0x42800000, s14
	v_dual_add_f32 v140, v140, v210 :: v_dual_cndmask_b32 v29, s74, v144
	v_cmp_gt_f32_e32 vcc_lo, s74, v145
	v_mul_f32_e32 v144, 0xbfb8aa3b, v48
	v_cmp_gt_f32_e64 s15, 0xc2fc0000, v142
	v_cndmask_b32_e64 v209, 0, 0x42800000, s12
	v_dual_add_f32 v141, v141, v211 :: v_dual_cndmask_b32 v28, s74, v145
	v_cmp_gt_f32_e32 vcc_lo, s74, v143
	v_mul_f32_e32 v145, 0xbfb8aa3b, v49
	v_cmp_gt_f32_e64 s17, 0xc2fc0000, v144
	v_cndmask_b32_e64 v212, 0, 0x42800000, s15
	v_dual_add_f32 v139, v139, v209 :: v_dual_cndmask_b32 v26, s74, v143
	v_cmp_gt_f32_e32 vcc_lo, s74, v146
	v_mul_f32_e32 v143, 0xbfb8aa3b, v47
	v_cmp_gt_f32_e64 s18, 0xc2fc0000, v145
	v_cndmask_b32_e64 v214, 0, 0x42800000, s17
	v_dual_add_f32 v142, v142, v212 :: v_dual_cndmask_b32 v31, s74, v146
	v_cmp_gt_f32_e32 vcc_lo, s74, v148
	v_mul_f32_e32 v146, 0xbfb8aa3b, v50
	v_cmp_gt_f32_e64 s16, 0xc2fc0000, v143
	v_cndmask_b32_e64 v215, 0, 0x42800000, s18
	v_dual_add_f32 v144, v144, v214 :: v_dual_cndmask_b32 v81, s74, v148
	v_cmp_gt_f32_e32 vcc_lo, s74, v149
	v_mul_f32_e32 v148, 0xbfb8aa3b, v52
	v_cmp_gt_f32_e64 s19, 0xc2fc0000, v146
	v_cndmask_b32_e64 v213, 0, 0x42800000, s16
	v_dual_add_f32 v145, v145, v215 :: v_dual_cndmask_b32 v80, s74, v149
	v_cmp_gt_f32_e32 vcc_lo, s74, v147
	v_mul_f32_e32 v149, 0xbfb8aa3b, v53
	v_cmp_gt_f32_e64 s21, 0xc2fc0000, v148
	v_cndmask_b32_e64 v216, 0, 0x42800000, s19
	v_dual_add_f32 v143, v143, v213 :: v_dual_cndmask_b32 v30, s74, v147
	v_cmp_gt_f32_e32 vcc_lo, s74, v150
	v_mul_f32_e32 v147, 0xbfb8aa3b, v51
	v_cmp_gt_f32_e64 s22, 0xc2fc0000, v149
	v_cndmask_b32_e64 v218, 0, 0x42800000, s21
	v_dual_add_f32 v146, v146, v216 :: v_dual_cndmask_b32 v83, s74, v150
	v_cmp_gt_f32_e32 vcc_lo, s74, v152
	v_mul_f32_e32 v150, 0xbfb8aa3b, v54
	v_cmp_gt_f32_e64 s20, 0xc2fc0000, v147
	v_cndmask_b32_e64 v219, 0, 0x42800000, s22
	v_dual_add_f32 v148, v148, v218 :: v_dual_cndmask_b32 v85, s74, v152
	v_cmp_gt_f32_e32 vcc_lo, s74, v153
	v_mul_f32_e32 v152, 0xbfb8aa3b, v56
	v_cmp_gt_f32_e64 s23, 0xc2fc0000, v150
	v_cndmask_b32_e64 v217, 0, 0x42800000, s20
	v_dual_add_f32 v149, v149, v219 :: v_dual_cndmask_b32 v84, s74, v153
	v_cmp_gt_f32_e32 vcc_lo, s74, v151
	v_mul_f32_e32 v153, 0xbfb8aa3b, v57
	v_cmp_gt_f32_e64 s25, 0xc2fc0000, v152
	v_cndmask_b32_e64 v220, 0, 0x42800000, s23
	v_dual_add_f32 v147, v147, v217 :: v_dual_cndmask_b32 v82, s74, v151
	v_cmp_gt_f32_e32 vcc_lo, s74, v86
	v_mul_f32_e32 v151, 0xbfb8aa3b, v55
	v_cmp_gt_f32_e64 s26, 0xc2fc0000, v153
	v_cndmask_b32_e64 v222, 0, 0x42800000, s25
	v_dual_add_f32 v150, v150, v220 :: v_dual_cndmask_b32 v87, s74, v86
	v_cmp_gt_f32_e32 vcc_lo, s74, v155
	v_cmp_gt_f32_e64 s24, 0xc2fc0000, v151
	v_cndmask_b32_e64 v223, 0, 0x42800000, s26
	v_add_f32_e32 v152, v152, v222
	v_exp_f32_e32 v130, v130
	v_cndmask_b32_e32 v89, s74, v155, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v156
	v_mul_f32_e32 v155, 0xbfb8aa3b, v59
	v_cndmask_b32_e64 v221, 0, 0x42800000, s24
	v_add_f32_e32 v153, v153, v223
	v_exp_f32_e32 v131, v131
	v_cndmask_b32_e32 v88, s74, v156, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v154
	v_mul_f32_e32 v156, 0xbfb8aa3b, v60
	v_cmp_gt_f32_e64 s28, 0xc2fc0000, v155
	v_add_f32_e32 v151, v151, v221
	v_exp_f32_e32 v132, v132
	v_cndmask_b32_e32 v86, s74, v154, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v92
	v_mul_f32_e32 v154, 0xbfb8aa3b, v58
	v_cmp_gt_f32_e64 s29, 0xc2fc0000, v156
	v_cndmask_b32_e64 v225, 0, 0x42800000, s28
	v_exp_f32_e32 v133, v133
	v_cndmask_b32_e32 v91, s74, v92, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v90
	v_cmp_gt_f32_e64 s27, 0xc2fc0000, v154
	v_cndmask_b32_e64 v226, 0, 0x42800000, s29
	v_add_f32_e32 v155, v155, v225
	v_exp_f32_e32 v134, v134
	v_cndmask_b32_e32 v93, s74, v90, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v95
	v_cndmask_b32_e64 v224, 0, 0x42800000, s27
	v_add_f32_e32 v156, v156, v226
	v_exp_f32_e32 v135, v135
	v_exp_f32_e32 v136, v136
	v_cndmask_b32_e32 v92, s74, v95, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v94
	v_add_f32_e32 v154, v154, v224
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
	v_cndmask_b32_e64 v200, 0, 0xffffffc0, s3
	v_cndmask_b32_e64 v201, 0, 0xffffffc0, s4
	v_cndmask_b32_e64 v202, 0, 0xffffffc0, s5
	v_cndmask_b32_e32 v99, s74, v99, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v103
	v_cndmask_b32_e64 v203, 0, 0xffffffc0, s6
	v_cndmask_b32_e64 v204, 0, 0xffffffc0, s7
	v_cndmask_b32_e64 v205, 0, 0xffffffc0, s8
	v_cndmask_b32_e64 v206, 0, 0xffffffc0, s9
	v_cndmask_b32_e32 v101, s74, v103, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v104
	v_cndmask_b32_e64 v207, 0, 0xffffffc0, s10
	v_cndmask_b32_e64 v208, 0, 0xffffffc0, s11
	v_cndmask_b32_e64 v209, 0, 0xffffffc0, s12
	v_cndmask_b32_e64 v210, 0, 0xffffffc0, s13
	v_cndmask_b32_e32 v100, s74, v104, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v102
	v_cndmask_b32_e64 v211, 0, 0xffffffc0, s14
	v_cndmask_b32_e64 v212, 0, 0xffffffc0, s15
	v_cndmask_b32_e64 v213, 0, 0xffffffc0, s16
	v_cndmask_b32_e64 v214, 0, 0xffffffc0, s17
	v_cndmask_b32_e32 v98, s74, v102, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v105
	v_cndmask_b32_e64 v215, 0, 0xffffffc0, s18
	v_cndmask_b32_e64 v216, 0, 0xffffffc0, s19
	v_cndmask_b32_e64 v217, 0, 0xffffffc0, s20
	v_cndmask_b32_e64 v218, 0, 0xffffffc0, s21
	v_cndmask_b32_e32 v103, s74, v105, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v107
	v_cndmask_b32_e64 v219, 0, 0xffffffc0, s22
	v_cndmask_b32_e64 v220, 0, 0xffffffc0, s23
	v_cndmask_b32_e64 v221, 0, 0xffffffc0, s24
	v_cndmask_b32_e64 v222, 0, 0xffffffc0, s25
	v_cndmask_b32_e32 v105, s74, v107, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v108
	v_cndmask_b32_e64 v223, 0, 0xffffffc0, s26
	v_cndmask_b32_e64 v224, 0, 0xffffffc0, s27
	v_cndmask_b32_e64 v225, 0, 0xffffffc0, s28
	v_cndmask_b32_e64 v226, 0, 0xffffffc0, s29
	v_cndmask_b32_e32 v104, s74, v108, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v106
	v_ldexp_f32 v130, v130, v200
	v_ldexp_f32 v131, v131, v201
	v_ldexp_f32 v132, v132, v202
	v_ldexp_f32 v133, v133, v203
	v_cndmask_b32_e32 v102, s74, v106, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v109
	v_ldexp_f32 v134, v134, v204
	v_ldexp_f32 v135, v135, v205
	v_ldexp_f32 v136, v136, v206
	v_ldexp_f32 v137, v137, v207
	v_cndmask_b32_e32 v107, s74, v109, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v111
	v_ldexp_f32 v138, v138, v208
	v_ldexp_f32 v139, v139, v209
	v_ldexp_f32 v140, v140, v210
	v_ldexp_f32 v141, v141, v211
	v_cndmask_b32_e32 v109, s74, v111, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v112
	v_ldexp_f32 v142, v142, v212
	v_ldexp_f32 v143, v143, v213
	v_ldexp_f32 v144, v144, v214
	v_ldexp_f32 v145, v145, v215
	v_cndmask_b32_e32 v108, s74, v112, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v110
	v_ldexp_f32 v146, v146, v216
	v_ldexp_f32 v147, v147, v217
	v_ldexp_f32 v148, v148, v218
	v_ldexp_f32 v149, v149, v219
	v_cndmask_b32_e32 v106, s74, v110, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v113
	v_ldexp_f32 v150, v150, v220
	v_ldexp_f32 v151, v151, v221
	v_ldexp_f32 v152, v152, v222
	v_ldexp_f32 v153, v153, v223
	v_cndmask_b32_e32 v111, s74, v113, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v115
	v_ldexp_f32 v154, v154, v224
	v_ldexp_f32 v155, v155, v225
	v_ldexp_f32 v156, v156, v226
	v_dual_add_f32 v200, 1.0, v134 :: v_dual_cndmask_b32 v113, s74, v115
	v_cmp_gt_f32_e32 vcc_lo, s74, v116
	v_dual_add_f32 v201, 1.0, v135 :: v_dual_add_f32 v202, 1.0, v136
	v_dual_add_f32 v203, 1.0, v137 :: v_dual_add_f32 v204, 1.0, v138
	v_cndmask_b32_e32 v112, s74, v116, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v114
	v_dual_add_f32 v205, 1.0, v139 :: v_dual_add_f32 v206, 1.0, v140
	v_dual_add_f32 v207, 1.0, v141 :: v_dual_add_f32 v208, 1.0, v142
	v_cndmask_b32_e32 v110, s74, v114, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v117
	v_dual_add_f32 v209, 1.0, v143 :: v_dual_add_f32 v210, 1.0, v144
	v_dual_add_f32 v211, 1.0, v145 :: v_dual_add_f32 v212, 1.0, v146
	v_cndmask_b32_e32 v115, s74, v117, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v119
	v_dual_add_f32 v213, 1.0, v147 :: v_dual_add_f32 v214, 1.0, v148
	v_dual_add_f32 v215, 1.0, v149 :: v_dual_add_f32 v216, 1.0, v150
	v_cndmask_b32_e32 v117, s74, v119, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v120
	v_dual_add_f32 v217, 1.0, v151 :: v_dual_add_f32 v218, 1.0, v152
	v_dual_add_f32 v219, 1.0, v153 :: v_dual_add_f32 v220, 1.0, v154
	v_cndmask_b32_e32 v116, s74, v120, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v118
	v_dual_add_f32 v221, 1.0, v155 :: v_dual_add_f32 v222, 1.0, v156
	v_rcp_f32_e32 v136, v200
	v_rcp_f32_e32 v137, v201
	v_cndmask_b32_e32 v114, s74, v118, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v121
	v_rcp_f32_e32 v138, v202
	v_rcp_f32_e32 v139, v203
	v_rcp_f32_e32 v140, v204
	v_rcp_f32_e32 v141, v205
	v_cndmask_b32_e32 v119, s74, v121, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v123
	v_rcp_f32_e32 v142, v206
	v_rcp_f32_e32 v143, v207
	v_rcp_f32_e32 v144, v208
	v_rcp_f32_e32 v145, v209
	v_cndmask_b32_e32 v121, s74, v123, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v124
	v_rcp_f32_e32 v146, v210
	v_rcp_f32_e32 v147, v211
	v_rcp_f32_e32 v148, v212
	v_rcp_f32_e32 v149, v213
	v_cndmask_b32_e32 v120, s74, v124, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v122
	v_rcp_f32_e32 v150, v214
	v_rcp_f32_e32 v151, v215
	v_rcp_f32_e32 v152, v216
	v_rcp_f32_e32 v153, v217
	v_cndmask_b32_e32 v118, s74, v122, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, s74, v125
	v_rcp_f32_e32 v154, v218
	v_rcp_f32_e32 v155, v219
	v_rcp_f32_e32 v156, v220
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
	v_ldexp_f32 v159, v162, v227
	v_ldexp_f32 v157, v157, v160
	v_exp_f32_e32 v126, v126
	v_ldexp_f32 v160, v163, v228
	v_ldexp_f32 v162, v165, v230
	v_ldexp_f32 v163, v166, v231
	v_ldexp_f32 v165, v168, v233
	v_ldexp_f32 v166, v169, v234
	v_ldexp_f32 v168, v171, v236
	v_ldexp_f32 v169, v172, v237
	v_ldexp_f32 v126, v126, v158
	v_ldexp_f32 v158, v161, v199
	v_ldexp_f32 v161, v164, v229
	v_ldexp_f32 v164, v167, v232
	v_ldexp_f32 v167, v170, v235
	v_ldexp_f32 v170, v173, v238
	v_ldexp_f32 v171, v174, v239
	v_ldexp_f32 v172, v175, v240
	v_ldexp_f32 v173, v176, v241
	v_ldexp_f32 v174, v177, v242
	v_ldexp_f32 v175, v178, v243
	v_ldexp_f32 v176, v179, v244
	v_ldexp_f32 v177, v180, v245
	v_ldexp_f32 v178, v181, v246
	v_ldexp_f32 v179, v182, v247
	v_ldexp_f32 v180, v183, v248
	v_ldexp_f32 v181, v184, v249
	v_ldexp_f32 v182, v185, v250
	v_ldexp_f32 v183, v186, v251
	v_ldexp_f32 v184, v187, v252
	v_ldexp_f32 v185, v188, v253
	v_ldexp_f32 v186, v189, v254
	v_ldexp_f32 v187, v190, v255
	s_set_vgpr_msb 4
	v_ldexp_f32 v188, v191, v0 /*v256*/
	v_ldexp_f32 v189, v196, v1 /*v257*/
	v_ldexp_f32 v190, v197, v2 /*v258*/
	v_ldexp_f32 v191, v198, v3 /*v259*/
	s_set_vgpr_msb 0x400
	v_dual_add_f32 v126, 1.0, v126 :: v_dual_add_f32 v127, 1.0, v127
	v_dual_add_f32 v157, 1.0, v157 :: v_dual_add_f32 v158, 1.0, v158
	v_dual_add_f32 v196, 1.0, v130 :: v_dual_add_f32 v197, 1.0, v131
	v_dual_add_f32 v198, 1.0, v132 :: v_dual_add_f32 v199, 1.0, v133
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
	v_rcp_f32_e32 v132, v196
	v_rcp_f32_e32 v133, v197
	v_rcp_f32_e32 v134, v198
	v_rcp_f32_e32 v135, v199
	v_rcp_f32_e32 v157, v221
	v_rcp_f32_e32 v158, v222
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
	ds_store_2addr_b64 v192, v[14:15], v[12:13] offset1:2
	ds_store_2addr_b64 v192, v[18:19], v[20:21] offset0:4 offset1:6
	ds_store_2addr_b64 v193, v[22:23], v[24:25] offset1:2
	ds_store_2addr_b64 v193, v[26:27], v[28:29] offset0:4 offset1:6
	ds_store_2addr_b64 v194, v[30:31], v[32:33] offset1:2
	ds_store_2addr_b64 v194, v[34:35], v[36:37] offset0:4 offset1:6
	ds_store_2addr_b64 v195, v[16:17], v[2:3] offset1:2
	ds_store_2addr_b64 v195, v[0:1], v[4:5] offset0:4 offset1:6
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_store_from_lds s[72:75], s[64:71]
	s_wait_tensorcnt 0x0
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
	.amdhsa_kernel gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
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
	.size	gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1, .Lfunc_end0-gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1

	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.num_vgpr, 281
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.num_agpr, 0
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.numbered_sgpr, 88
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.num_named_barrier, 0
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.private_seg_size, 0
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.uses_vcc, 1
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.uses_flat_scratch, 0
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.has_dyn_sized_stack, 0
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.has_recursion, 0
	.set gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.has_indirect_call, 0
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
    .name:           gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     90
    .sgpr_spill_count: 0
    .symbol:         gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     281
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
