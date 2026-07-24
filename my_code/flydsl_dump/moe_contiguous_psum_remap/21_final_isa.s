	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_contiguous_psum_remap
	.p2align	8
	.type	moe_contiguous_psum_remap,@function
moe_contiguous_psum_remap:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v6, 2, v0
	v_cmp_gt_u32_e32 vcc_lo, s13, v0
	s_mov_b64 s[22:23], 0x1ffffff
	s_or_b64 s[20:21], s[2:3], 0xfe00000000000000
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_2
	v_lshlrev_b32_e32 v2, 2, v0
	s_cvt_f32_u32 s0, s15
	s_sub_co_i32 s2, 0, s15
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_4) | instid1(SALU_CYCLE_3)
	v_rcp_iflag_f32_e32 v4, s0
	buffer_load_b32 v3, v2, s[20:23], null offen
	v_nop
	v_readfirstlane_b32 s0, v4
	s_mul_f32 s0, s0, 0x4f7ffffe
	s_cvt_u32_f32 s0, s0
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s0
	s_mul_hi_u32 s2, s0, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_add_co_i32 s0, s0, s2
	s_wait_loadcnt 0x0
	v_add3_u32 v3, s15, -1, v3
	v_mul_hi_u32 v4, v3, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v4, v4, s15
	v_sub_nc_u32_e32 v4, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_subrev_nc_u32_e32 v5, s15, v4
	v_cmp_le_u32_e64 s0, s15, v4
	v_cndmask_b32_e64 v4, v4, v5, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_subrev_nc_u32_e32 v5, s15, v4
	v_cmp_le_u32_e64 s0, s15, v4
	v_cndmask_b32_e64 v4, v4, v5, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v3, v3, v4
	ds_store_b32 v2, v3
.LBB0_2:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_6
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2
	v_cmpx_ne_u32_e32 0, v0
	v_add_nc_u32_e32 v4, -4, v2
	ds_load_b32 v4, v4
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3 offset:2048
.LBB0_6:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_10
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2 offset:2048
	v_cmpx_lt_u32_e32 1, v0
	ds_load_b32 v4, v2 offset:2040
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3
.LBB0_10:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_14
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2
	v_cmpx_lt_u32_e32 3, v0
	v_add_nc_u32_e32 v4, -16, v2
	ds_load_b32 v4, v4
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3 offset:2048
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_18
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2 offset:2048
	v_cmpx_lt_u32_e32 7, v0
	ds_load_b32 v4, v2 offset:2016
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_22
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2
	v_cmpx_lt_u32_e32 15, v0
	v_subrev_nc_u32_e32 v4, 64, v2
	ds_load_b32 v4, v4
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3 offset:2048
.LBB0_22:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_26
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2 offset:2048
	v_cmpx_lt_u32_e32 31, v0
	ds_load_b32 v4, v2 offset:1920
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3
.LBB0_26:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_30
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2
	v_cmpx_lt_u32_e32 63, v0
	v_add_nc_u32_e32 v4, 0xffffff00, v2
	ds_load_b32 v4, v4
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3 offset:2048
.LBB0_30:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_34
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2 offset:2048
	v_cmpx_lt_u32_e32 0x7f, v0
	ds_load_b32 v4, v2 offset:1536
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3
.LBB0_34:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_38
	v_dual_lshlrev_b32 v2, 2, v0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s2, exec_lo
	ds_load_b32 v3, v2
	v_cmpx_lt_u32_e32 0xff, v0
	v_add_nc_u32_e32 v4, 0xfffffc00, v2
	ds_load_b32 v4, v4
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v4, v3
	ds_store_b32 v2, v3 offset:2048
.LBB0_38:
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_mov_b64 s[2:3], 0x1ffffff
	s_or_b64 s[0:1], s[6:7], 0xfe00000000000000
	s_barrier_wait -1
	s_and_saveexec_b32 s6, vcc_lo
	s_cbranch_execz .LBB0_43
	v_mov_b32_e32 v2, 0
	s_mov_b32 s7, exec_lo
	v_cmpx_ne_u32_e32 0, v0
	v_lshlrev_b32_e32 v2, 2, v0
	ds_load_b32 v2, v2 offset:2044
	s_or_b32 exec_lo, exec_lo, s7
	v_lshlrev_b32_e32 v3, 2, v0
	s_add_co_i32 s13, s13, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_eq_u32_e32 vcc_lo, s13, v0
	buffer_load_b32 v4, v3, s[20:23], null offen
	s_wait_xcnt 0x0
	s_mov_b32 s23, 0
	s_mov_b32 s22, 0x1ffffff
	s_or_b64 s[20:21], s[8:9], 0xfe00000000000000
	s_wait_loadcnt_dscnt 0x0
	v_add_nc_u32_e32 v4, v4, v2
	buffer_store_b32 v2, v3, s[0:3], null offen
	buffer_store_b32 v4, v3, s[20:23], null offen
	s_wait_xcnt 0x0
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_43
	ds_load_b32 v2, v3 offset:2048
	s_or_b64 s[20:21], s[10:11], 0xfe00000000000000
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, s15, v2
	buffer_store_b32 v2, off, s[20:23], null
.LBB0_43:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_or_b64 s[8:9], s[16:17], 0xfe00000000000000
	s_mov_b32 s11, 0
	s_mov_b32 s10, 0x1ffffff
	s_mov_b32 s6, exec_lo
	s_barrier_wait -1
	buffer_load_b32 v2, off, s[8:11], null
	s_wait_loadcnt 0x0
	v_cmpx_lt_i32_e64 v0, v2
	s_cbranch_execz .LBB0_46
	s_cvt_f32_u32 s6, s14
	s_sub_co_i32 s9, 0, s14
	s_or_b64 s[4:5], s[4:5], 0xfe00000000000000
	v_mov_b32_e32 v5, 0
	v_rcp_iflag_f32_e32 v3, s6
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s6, v3
	v_ashrrev_i32_e32 v3, 31, v2
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_cvt_u32_f32 s8, s6
	s_mov_b64 s[6:7], 0x1ffffff
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s9, s9, s8
	s_mul_hi_u32 s9, s8, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s10, s8, s9
	s_mov_b32 s8, s11
.LBB0_45:
	buffer_load_b32 v4, v6, s[4:7], null offen
	v_add_nc_u64_e32 v[0:1], 0x200, v[0:1]
	s_wait_loadcnt 0x0
	v_mul_u64_e32 v[8:9], s[10:11], v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v7, v9, s14
	v_dual_add_nc_u32 v8, 1, v9 :: v_dual_sub_nc_u32 v7, v4, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_subrev_nc_u32_e32 v10, s14, v7
	v_cmp_le_u32_e32 vcc_lo, s14, v7
	v_dual_cndmask_b32 v8, v9, v8 :: v_dual_cndmask_b32 v7, v7, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v9, 1, v8
	v_cmp_le_u32_e32 vcc_lo, s14, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v7, v8, v9, vcc_lo
	v_cmp_ge_i64_e32 vcc_lo, v[0:1], v[2:3]
	v_lshlrev_b32_e32 v8, 2, v7
	s_or_b32 s8, vcc_lo, s8
	buffer_load_b32 v8, v8, s[0:3], null offen
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v4, v8, v4
	v_mul_lo_u32 v7, s14, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v4, v4, v7
	buffer_store_b32 v4, v6, s[4:7], null offen
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v6, 0x800, v6
	s_and_not1_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_45
.LBB0_46:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_contiguous_psum_remap
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 18
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 16
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
		.amdhsa_next_free_vgpr 11
		.amdhsa_next_free_sgpr 24
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
	.size	moe_contiguous_psum_remap, .Lfunc_end0-moe_contiguous_psum_remap

	.set moe_contiguous_psum_remap.num_vgpr, 11
	.set moe_contiguous_psum_remap.num_agpr, 0
	.set moe_contiguous_psum_remap.numbered_sgpr, 24
	.set moe_contiguous_psum_remap.num_named_barrier, 0
	.set moe_contiguous_psum_remap.private_seg_size, 0
	.set moe_contiguous_psum_remap.uses_vcc, 1
	.set moe_contiguous_psum_remap.uses_flat_scratch, 0
	.set moe_contiguous_psum_remap.has_dyn_sized_stack, 0
	.set moe_contiguous_psum_remap.has_recursion, 0
	.set moe_contiguous_psum_remap.has_indirect_call, 0
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
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 512
    .name:           moe_contiguous_psum_remap
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         moe_contiguous_psum_remap.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     11
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
