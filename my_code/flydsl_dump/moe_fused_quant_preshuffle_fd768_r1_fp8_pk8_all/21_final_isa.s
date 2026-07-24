	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all
	.p2align	8
	.type	moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all,@function
moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s8, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	v_lshrrev_b32_e32 v1, 5, v0
	s_add_co_i32 s1, s1, s0
	s_cmp_eq_u32 s8, 0
	s_cselect_b32 s0, ttmp9, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_lshl_or_b32 v20, s0, 3, v1
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s10, v20
	s_cbranch_execz .LBB0_19
	v_dual_mov_b32 v19, 0 :: v_dual_bitop2_b32 v2, 3, v0 bitop3:0x40
	v_bfe_u32 v1, v0, 2, 3
	v_mad_nc_u64_u32 v[4:5], 0x600, v20, s[2:3]
	v_mov_b64_e32 v[16:17], 6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v18, 6 :: v_dual_lshlrev_b32 v21, 3, v2
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	v_mad_nc_u64_u32 v[2:3], 0x300, v20, s[4:5]
	v_mov_b64_e32 v[6:7], 12
	v_lshl_or_b32 v22, v1, 5, v21
	v_dual_mov_b32 v14, 12 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v15, v19
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v23, 1, v22
	s_mov_b64 s[10:11], 0x1ffffff
	s_or_b64 s[8:9], s[6:7], 0xfe00000000000000
	s_mov_b32 s2, exec_lo
.LBB0_2:
	v_readfirstlane_b32 s4, v12
	v_readfirstlane_b32 s5, v13
	v_readfirstlane_b32 s6, v14
	v_readfirstlane_b32 s7, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e64 s0, s[4:5], v[12:13]
	v_cmp_eq_u64_e64 s1, s[6:7], v[14:15]
	s_and_b32 s0, s0, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[8:11], v23, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_2
	s_mov_b32 exec_lo, s2
	s_wait_loadcnt 0x0
	v_and_b32_e32 v12, 0xffff0000, v8
	v_dual_lshlrev_b32 v13, 16, v8 :: v_dual_lshlrev_b32 v15, 16, v9
	v_and_b32_e32 v14, 0xffff0000, v9
	v_dual_lshlrev_b32 v23, 16, v10 :: v_dual_lshlrev_b32 v25, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_maximum3_f32 v12, |v13|, 0, |v12|
	v_and_b32_e32 v13, 0xffff0000, v10
	v_mbcnt_lo_u32_b32 v24, -1, 0
	s_mov_b32 s2, exec_lo
	v_maximum3_f32 v12, v12, |v15|, |v14|
	v_and_b32_e32 v15, 0xffff0000, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v24, 2, v24
	v_maximum3_f32 v12, v12, |v23|, |v13|
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v14, 4, v24
	v_maximum3_f32 v12, v12, |v25|, |v15|
	v_xor_b32_e32 v15, 8, v24
	ds_bpermute_b32 v13, v14, v12
	s_wait_dscnt 0x0
	v_maximum_f32 v12, v12, v13
	ds_bpermute_b32 v13, v15, v12
	s_wait_dscnt 0x0
	v_maximum_f32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v12, 0x3b124925, v12
	v_and_b32_e32 v13, 0x7fffff, v12
	v_bfe_u32 v12, v12, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v13
	v_add_co_ci_u32_e64 v12, null, 0, v12, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v12, 0xff, v12
	v_lshlrev_b32_e32 v13, 23, v12
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp8_bf16 v[8:9], v[8:11], v13
	;;#ASMEND
.LBB0_4:
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s6, v18
	v_readfirstlane_b32 s7, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e64 s0, s[4:5], v[2:3]
	v_cmp_eq_u64_e64 s1, s[6:7], v[18:19]
	s_and_b32 s0, s0, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	buffer_store_b64 v[8:9], v22, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_4
	s_mov_b32 exec_lo, s2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_7
	v_mul_lo_u32 v8, v20, 24
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v8, v8, v1
	buffer_store_b8 v12, v8, s[8:11], null offen
.LBB0_7:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 8, v1
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v19, v18, 5, v21
	v_lshlrev_b32_e32 v12, 1, v19
.LBB0_8:
	v_readfirstlane_b32 s4, v4
	v_readfirstlane_b32 s5, v5
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e64 s0, s[4:5], v[4:5]
	v_cmp_eq_u64_e64 s1, s[6:7], v[6:7]
	s_and_b32 s0, s0, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[8:11], v12, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_8
	s_mov_b32 exec_lo, s2
	s_wait_loadcnt 0x0
	v_and_b32_e32 v12, 0xffff0000, v8
	v_dual_lshlrev_b32 v13, 16, v8 :: v_dual_lshlrev_b32 v23, 16, v9
	v_and_b32_e32 v22, 0xffff0000, v9
	v_lshlrev_b32_e32 v24, 16, v10
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v12, |v13|, 0, |v12|
	v_and_b32_e32 v13, 0xffff0000, v10
	v_maximum3_f32 v12, v12, |v23|, |v22|
	v_and_b32_e32 v22, 0xffff0000, v11
	v_lshlrev_b32_e32 v23, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v12, v12, |v24|, |v13|
	v_maximum3_f32 v12, v12, |v23|, |v22|
	ds_bpermute_b32 v13, v14, v12
	s_wait_dscnt 0x0
	v_maximum_f32 v12, v12, v13
	ds_bpermute_b32 v13, v15, v12
	s_wait_dscnt 0x0
	v_maximum_f32 v12, v12, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v12, 0x3b124925, v12
	v_and_b32_e32 v13, 0x7fffff, v12
	v_bfe_u32 v12, v12, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e64 s0, 0, v13
	v_mov_b32_e32 v13, v17
	v_add_co_ci_u32_e64 v12, null, 0, v12, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v22, 0xff, v12
	v_dual_mov_b32 v12, v16 :: v_dual_lshlrev_b32 v23, 23, v22
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp8_bf16 v[8:9], v[8:11], v23
	;;#ASMEND
.LBB0_10:
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s6, v12
	v_readfirstlane_b32 s7, v13
	v_cmp_eq_u64_e64 s0, s[4:5], v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s1, s[6:7], v[12:13]
	s_and_b32 s0, s0, s1
	s_and_saveexec_b32 s0, s0
	buffer_store_b64 v[8:9], v19, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_10
	s_mov_b32 exec_lo, s2
	v_lshrrev_b32_e32 v12, 2, v0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_13
	v_mad_u32 v0, v20, 24, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v0, -4, v0, v12
	buffer_store_b8 v22, v0, s[8:11], null offen
.LBB0_13:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v13, 16, v1
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v18, v13, 5, v21
	v_lshlrev_b32_e32 v0, 1, v18
.LBB0_14:
	v_readfirstlane_b32 s4, v4
	v_readfirstlane_b32 s5, v5
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e64 s0, s[4:5], v[4:5]
	v_cmp_eq_u64_e64 s1, s[6:7], v[6:7]
	s_and_b32 s0, s0, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	s_wait_loadcnt 0x0
	buffer_load_b128 v[8:11], v0, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_14
	s_mov_b32 exec_lo, s2
	s_wait_loadcnt 0x0
	v_and_b32_e32 v0, 0xffff0000, v8
	v_dual_lshlrev_b32 v1, 16, v8 :: v_dual_lshlrev_b32 v5, 16, v9
	v_and_b32_e32 v4, 0xffff0000, v9
	v_lshlrev_b32_e32 v6, 16, v10
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v0, |v1|, 0, |v0|
	v_and_b32_e32 v1, 0xffff0000, v10
	v_maximum3_f32 v0, v0, |v5|, |v4|
	v_and_b32_e32 v4, 0xffff0000, v11
	v_lshlrev_b32_e32 v5, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v0, v0, |v6|, |v1|
	v_maximum3_f32 v0, v0, |v5|, |v4|
	v_dual_mov_b32 v5, v17 :: v_dual_mov_b32 v4, v16
	ds_bpermute_b32 v1, v14, v0
	s_wait_dscnt 0x0
	v_maximum_f32 v0, v0, v1
	ds_bpermute_b32 v1, v15, v0
	s_wait_dscnt 0x0
	v_maximum_f32 v0, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v0, 0x3b124925, v0
	v_and_b32_e32 v1, 0x7fffff, v0
	v_bfe_u32 v0, v0, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v1
	v_add_co_ci_u32_e64 v0, null, 0, v0, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v6, 0xff, v0
	v_lshlrev_b32_e32 v0, 23, v6
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp8_bf16 v[0:1], v[8:11], v0
	;;#ASMEND
.LBB0_16:
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s6, v4
	v_readfirstlane_b32 s7, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u64_e64 s0, s[4:5], v[2:3]
	v_cmp_eq_u64_e64 s1, s[6:7], v[4:5]
	s_and_b32 s0, s0, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b32 s0, s0
	buffer_store_b64 v[0:1], v18, s[4:7], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_16
	s_mov_b32 exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_19
	v_mad_u32 v0, v20, 24, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_bfi_b32 v0, -4, v0, v12
	buffer_store_b8 v6, v0, s[8:11], null offen
.LBB0_19:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 10
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
		.amdhsa_next_free_vgpr 26
		.amdhsa_next_free_sgpr 12
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 12
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
	.size	moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all, .Lfunc_end0-moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all

	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.num_vgpr, 26
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.num_agpr, 0
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.numbered_sgpr, 12
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.num_named_barrier, 0
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.private_seg_size, 0
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.uses_vcc, 1
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.uses_flat_scratch, 0
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.has_dyn_sized_stack, 0
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.has_recursion, 0
	.set moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.has_indirect_call, 0
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name:           moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         moe_fused_quant_preshuffle_fd768_r1_fp8_pk8_all.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     26
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
