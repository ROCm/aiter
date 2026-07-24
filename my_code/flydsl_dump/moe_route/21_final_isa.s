	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_route
	.p2align	8
	.type	moe_route,@function
moe_route:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_mov_b64 s[0:1], s[6:7]
	s_bfe_u32 s6, ttmp6, 0x4000c
	s_and_b32 s7, ttmp6, 15
	s_add_co_i32 s6, s6, 1
	s_getreg_b32 s10, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s6, ttmp9, s6
	s_mov_b32 s15, 0
	s_add_co_i32 s7, s7, s6
	s_cmp_eq_u32 s10, 0
	s_cselect_b32 s6, ttmp9, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v0, s6, 8, v0
	s_mov_b32 s6, exec_lo
	v_cmpx_gt_u32_e64 s8, v0
	s_cbranch_execz .LBB0_2
	v_dual_lshlrev_b32 v0, 2, v0 :: v_dual_mov_b32 v2, 1
	s_or_b64 s[12:13], s[2:3], 0xfe00000000000000
	s_mov_b32 s14, 0x1ffffff
	s_or_b32 s1, s1, 0xfe000000
	buffer_load_b32 v1, v0, s[12:15], null offen
	s_mov_b32 s2, s14
	s_mov_b32 s3, s15
	s_wait_loadcnt 0x0
	global_atomic_add_u32 v2, v1, v2, s[4:5] scale_offset th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_mad_u32 v1, v1, s9, v2
	buffer_store_b32 v1, v0, s[0:3], null offen
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_route
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 10
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 8
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
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 16
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 2
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
	.size	moe_route, .Lfunc_end0-moe_route

	.set moe_route.num_vgpr, 3
	.set moe_route.num_agpr, 0
	.set moe_route.numbered_sgpr, 16
	.set moe_route.num_named_barrier, 0
	.set moe_route.private_seg_size, 0
	.set moe_route.uses_vcc, 0
	.set moe_route.uses_flat_scratch, 0
	.set moe_route.has_dyn_sized_stack, 0
	.set moe_route.has_recursion, 0
	.set moe_route.has_indirect_call, 0
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
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name:           moe_route
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         moe_route.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
