// Minimal gfx1250 kernel used to prove the ISA runner really executes code.
// Each thread writes sentinel + global_tid to out[global_tid], so a correct run
// is distinguishable from a zeroed buffer, a partial launch and a stale buffer.
//
// kernarg layout (kernarg_size 16, no hidden args):
//   0x00  out        global u32*   8B
//   0x08  sentinel   u32           4B
//   0x0c  block_size u32           4B   (host passes blockDim.x)
//
// s[0:1] = kernarg segment ptr (user SGPR), s2 = workgroup_id_x (system SGPR),
// v0 = workitem_id_x. Registers follow from .amdhsa_user_sgpr_count 2.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	isa_smoke
	.p2align	8
	.type	isa_smoke,@function
isa_smoke:
	s_load_b64 s[4:5], s[0:1], 0x0          // out
	s_load_b64 s[6:7], s[0:1], 0x8          // s6=sentinel, s7=block_size
	s_wait_kmcnt 0x0
	s_mul_i32 s8, s2, s7                    // wg_id_x * block_size
	v_add_nc_u32_e32 v1, s8, v0             // gid
	v_add_nc_u32_e32 v2, s6, v1             // value = sentinel + gid
	v_lshlrev_b32_e32 v3, 2, v1             // byte offset = gid * 4
	global_store_b32 v3, v2, s[4:5]
	s_endpgm
.Lfunc_end0:
	.size	isa_smoke, .Lfunc_end0-isa_smoke

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel isa_smoke
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
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
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 9
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
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
      - .offset:         12
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           isa_smoke
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .sgpr_spill_count: 0
    .symbol:         isa_smoke.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
