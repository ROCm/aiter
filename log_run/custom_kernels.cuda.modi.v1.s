	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.protected	_Z20matrixMultiplySharedPfS_S_iiiiii ; -- Begin function _Z20matrixMultiplySharedPfS_S_iiiiii
	.globl	_Z20matrixMultiplySharedPfS_S_iiiiii
	.p2align	8
	.type	_Z20matrixMultiplySharedPfS_S_iiiiii,@function
_Z20matrixMultiplySharedPfS_S_iiiiii:   ; @_Z20matrixMultiplySharedPfS_S_iiiiii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_add_u32 s0, s0, 48
	s_addc_u32 s1, s1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_u32 s15, s17
	s_cselect_b32 s17, 14, 20
	v_mov_b32_e32 v1, s17
	s_cmp_lt_u32 s14, s16
	s_cselect_b32 s16, 12, 18
	global_load_ushort v2, v1, s[0:1]
	v_mov_b32_e32 v1, s16
	global_load_ushort v4, v1, s[0:1]
	v_mov_b32_e32 v1, 0
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v6, 0x3ff, v0
	v_lshlrev_b32_e32 v0, 2, v6
	v_lshlrev_b32_e32 v7, 7, v3
	v_mul_lo_u32 v5, v3, s11
	v_add_u32_e32 v9, v7, v0
	v_or_b32_e32 v10, 0x1000, v0
	v_add_u32_e32 v11, v10, v7
	v_add_u32_e32 v12, 0x400, v10
	v_add_u32_e32 v13, 0x800, v10
	v_add_u32_e32 v14, 0xc00, v10
	s_add_i32 s0, s9, -1
	s_lshl_b32 s20, s11, 5
	s_lshr_b32 s0, s0, 5
	s_add_i32 s21, s0, 1
	s_waitcnt vmcnt(1)
	v_mul_lo_u32 v0, s15, v2
	v_add_u32_e32 v8, v0, v3
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v0, s14, v4
	v_add_u32_e32 v2, v0, v6
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_mul_lo_u32 v15, v8, s9
	v_add3_u32 v4, v6, v5, v0
	v_cmp_gt_i32_e64 s[0:1], s11, v2
	v_mov_b32_e32 v16, 0
	s_branch .LBB0_3
.LBB0_1:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b32 v11, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[34:35], v10 offset1:32
	ds_read_b128 v[18:21], v7
	ds_read_b128 v[22:25], v7 offset:16
	ds_read2_b32 v[36:37], v10 offset0:64 offset1:96
	ds_read_b128 v[26:29], v7 offset:32
	ds_read_b128 v[30:33], v7 offset:48
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v16, v18, v34
	ds_read2_b32 v[38:39], v10 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v19, v35
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v16, v20, v36
	ds_read2_b32 v[18:19], v10 offset0:192 offset1:224
	v_fmac_f32_e32 v16, v21, v37
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v22, v38
	ds_read2_b32 v[20:21], v12 offset1:32
	v_fmac_f32_e32 v16, v23, v39
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v24, v18
	ds_read2_b32 v[22:23], v12 offset0:64 offset1:96
	v_fmac_f32_e32 v16, v25, v19
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v26, v20
	ds_read2_b32 v[18:19], v12 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v27, v21
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v28, v22
	v_fmac_f32_e32 v16, v29, v23
	ds_read2_b32 v[22:23], v12 offset0:192 offset1:224
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v30, v18
	v_fmac_f32_e32 v16, v31, v19
	ds_read2_b32 v[26:27], v13 offset1:32
	ds_read_b128 v[18:21], v7 offset:64
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v32, v22
	v_fmac_f32_e32 v16, v33, v23
	ds_read2_b32 v[28:29], v13 offset0:64 offset1:96
	ds_read_b128 v[22:25], v7 offset:80
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v18, v26
	ds_read2_b32 v[30:31], v13 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v19, v27
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v20, v28
	ds_read2_b32 v[18:19], v13 offset0:192 offset1:224
	v_fmac_f32_e32 v16, v21, v29
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[20:21], v[22:23], v[30:31]
	s_nop 0
	v_add_f32_e32 v0, v16, v20
	v_add_f32_e32 v0, v0, v21
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[20:21], v[24:25], v[18:19]
	ds_read2_b32 v[24:25], v14 offset1:32
	ds_read_b128 v[16:19], v7 offset:96
	v_add_f32_e32 v0, v0, v20
	v_add_f32_e32 v0, v0, v21
	ds_read2_b32 v[26:27], v14 offset0:64 offset1:96
	ds_read_b128 v[20:23], v7 offset:112
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[16:17], v[16:17], v[24:25]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v0, v0, v17
	ds_read2_b32 v[16:17], v14 offset0:128 offset1:160
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[18:19], v[18:19], v[26:27]
	s_nop 0
	v_add_f32_e32 v0, v0, v18
	ds_read2_b32 v[24:25], v14 offset0:192 offset1:224
	v_add_f32_e32 v0, v0, v19
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[16:17], v[20:21], v[16:17]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v0, v0, v17
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[16:17], v[22:23], v[24:25]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v16, v0, v17
	s_add_i32 s21, s21, -1
	v_add_u32_e32 v6, 32, v6
	v_add_u32_e32 v4, s20, v4
	s_cmp_eq_u32 s21, 0
	v_add_u32_e32 v3, 32, v3
	s_cbranch_scc1 .LBB0_10
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_7
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_gt_u32_e64 s[14:15], s9, v6
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[18:19], s[14:15]
	s_cbranch_execz .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_u32_e32 v0, v15, v6
	v_lshl_add_u64 v[18:19], v[0:1], 2, s[2:3]
	global_load_dword v0, v[18:19], off
.LBB0_6:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b32 v9, v0
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_gt_u32_e64 s[14:15], s10, v3
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[18:19], s[14:15]
	s_cbranch_execz .LBB0_1
; %bb.9:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v5, v1
	v_lshl_add_u64 v[18:19], v[4:5], 2, s[4:5]
	global_load_dword v0, v[18:19], off
	s_branch .LBB0_1
.LBB0_10:
	v_cmp_gt_i32_e32 vcc, s12, v8
	v_cmp_gt_i32_e64 s[0:1], s13, v2
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_mad_u64_u32 v[0:1], s[0:1], v8, s13, v[2:3]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	global_store_dword v[0:1], v16, off
.LBB0_12:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20matrixMultiplySharedPfS_S_iiiiii
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  12
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 22
		.amdhsa_accum_offset 40
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
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
	.size	_Z20matrixMultiplySharedPfS_S_iiiiii, .Lfunc_end0-_Z20matrixMultiplySharedPfS_S_iiiiii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 908
; NumSgprs: 28
; NumVgprs: 40
; NumAgprs: 0
; TotalNumVgprs: 40
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 8192 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 40
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_bfe_u32 v1, v0, 10, 10
	v_and_b32_e32 v4, 0x3ff, v0
	v_lshlrev_b32_e32 v0, 3, v4
	s_lshl_b32 s18, s2, 2
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB1_6
; %bb.1:
	s_min_i32 s19, s18, 0x8000
	v_lshlrev_b32_e32 v2, 4, v4
	v_lshl_add_u32 v5, v1, 10, v2
	v_lshl_add_u32 v6, v1, 9, v0
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v3, 0
                                        ; implicit-def: $sgpr14_sgpr15
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_b64 s[16:17], exec, s[14:15]
	s_or_b64 s[0:1], s[16:17], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB1_5
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v2, s13, v6
	v_cmp_gt_u32_e32 vcc, s19, v2
	s_or_b64 s[14:15], s[14:15], exec
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshl_add_u64 v[8:9], v[2:3], 1, s[6:7]
	global_load_dwordx4 v[8:11], v[8:9], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s19
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[8:11]
	v_add_u32_e32 v5, 0x4000, v5
	s_or_b64 s[14:15], s[14:15], s[20:21]
	s_branch .LBB1_2
.LBB1_5:
	s_or_b64 exec, exec, s[0:1]
.LBB1_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_22
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v2, s12, v1
	v_cmp_gt_u32_e32 vcc, s3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_22
; %bb.8:
	v_cmp_gt_u32_e64 s[0:1], s2, v0
	s_cmpk_gt_u32 s2, 0x200
	v_add_u32_e32 v1, 0xfffffe00, v0
	v_cmp_eq_u32_e64 s[12:13], 63, v4
	s_mul_i32 s16, s11, s10
	v_lshlrev_b32_e32 v8, 1, v0
	s_cselect_b64 s[6:7], -1, 0
	s_lshl_b32 s19, s2, 1
	v_add_u32_e32 v9, s19, v8
	v_mov_b32_e32 v3, 16
	v_add_u32_e32 v10, 32, v3
	v_add_u32_e32 v11, s19, v9
	v_add_u32_e32 v12, 64, v3
	v_add_u32_e32 v13, s19, v11
	v_add_u32_e32 v14, 0x60, v3
	s_mul_i32 s17, s2, 6
	s_addk_i32 s17, 0x400
	s_addk_i32 s18, 0x400
	s_addk_i32 s19, 0x400
	v_mad_u64_u32 v[4:5], s[10:11], s2, v2, v[0:1]
	s_mul_i32 s20, s16, s2
	s_mov_b64 s[10:11], 0
	v_mov_b32_e32 v7, 0
	v_cndmask_b32_e64 v3, 0, 1, s[6:7]
	v_cmp_ne_u32_e64 s[6:7], 1, v3
	s_branch .LBB1_10
.LBB1_9:                                ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v2, s16, v2
	v_cmp_le_u32_e32 vcc, s3, v2
	s_or_b64 s[10:11], vcc, s[10:11]
	v_add_u32_e32 v4, s20, v4
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB1_22
.LBB1_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_14 Depth 2
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execz .LBB1_12
; %bb.11:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mad_u64_u32 v[16:17], s[22:23], v2, s2, v[0:1]
	v_mov_b32_e32 v17, v7
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[4:5]
	global_load_dwordx4 v[16:19], v[16:17], off nt
	ds_read_b128 v[20:23], v8
	ds_read2_b64 v[24:27], v9 offset1:1
	ds_read2_b32 v[30:31], v11 offset0:2 offset1:3
	ds_read2_b32 v[28:29], v11 offset1:1
	ds_read2_b64 v[32:35], v13 offset1:1
	s_waitcnt lgkmcnt(4)
	scratch_store_dwordx4 off, v[20:23], off offset:16
	s_waitcnt lgkmcnt(3)
	scratch_store_dwordx2 v10, v[24:25], off
	scratch_store_dwordx2 v10, v[26:27], off offset:8
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 v12, v[28:31], off
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 v14, v[32:33], off
	s_waitcnt vmcnt(5)
	scratch_store_dwordx4 off, v[16:19], off offset:144
	scratch_store_dwordx2 v14, v[34:35], off offset:8
.LBB1_12:                               ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[14:15]
	s_mov_b32 s22, 0
	s_movk_i32 s21, 0x200
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v5, 0
	s_cbranch_vccnz .LBB1_18
; %bb.13:                               ;   in Loop: Header=BB1_10 Depth=1
	s_mov_b32 s23, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v17, v8
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v3, 0
.LBB1_14:                               ;   Parent Loop BB1_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s22, s23, 1
	v_add_u32_e32 v6, s21, v0
	v_cmp_gt_u32_e32 vcc, s2, v6
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_16
; %bb.15:                               ;   in Loop: Header=BB1_14 Depth=2
	v_add_u32_e32 v6, s21, v4
	v_lshl_add_u64 v[18:19], v[6:7], 1, s[4:5]
	global_load_dwordx4 v[18:21], v[18:19], off nt
	s_lshl_b32 s24, s22, 4
	ds_read_b128 v[22:25], v17 offset:1024
	v_add_u32_e32 v6, s19, v17
	v_add_u32_e32 v30, s18, v17
	v_add_u32_e32 v34, s17, v17
	s_lshl_b32 s23, s23, 4
	s_and_b32 s24, s24, 16
	ds_read2_b64 v[26:29], v6 offset1:1
	ds_read2_b32 v[32:33], v30 offset0:2 offset1:3
	ds_read2_b32 v[30:31], v30 offset1:1
	ds_read2_b64 v[34:37], v34 offset1:1
	s_and_b32 s23, s23, 16
	s_add_i32 s25, s24, 0x90
	s_add_i32 s24, s24, 16
	s_add_i32 s26, s23, 16
	s_addk_i32 s23, 0x90
	s_waitcnt lgkmcnt(4)
	scratch_store_dwordx4 off, v[22:25], s24
	s_waitcnt lgkmcnt(3)
	scratch_store_dwordx2 off, v[28:29], s24 offset:40
	scratch_store_dwordx2 off, v[26:27], s24 offset:32
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 off, v[30:33], s24 offset:64
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[34:35], s24 offset:96
	scratch_store_dwordx2 off, v[36:37], s24 offset:104
	s_waitcnt vmcnt(6)
	scratch_store_dwordx4 off, v[18:21], s25
	scratch_load_dword v6, off, s23
	s_nop 0
	scratch_load_dword v18, off, s26
	scratch_load_dword v19, off, s26 offset:4
	scratch_load_dword v20, off, s26 offset:8
	scratch_load_dword v21, off, s26 offset:12
	scratch_load_dword v22, off, s26 offset:32
	scratch_load_dword v23, off, s26 offset:64
	scratch_load_dword v24, off, s26 offset:96
	scratch_load_dword v25, off, s23 offset:4
	scratch_load_dword v26, off, s26 offset:36
	scratch_load_dword v27, off, s26 offset:68
	scratch_load_dword v28, off, s26 offset:100
	scratch_load_dword v29, off, s23 offset:8
	scratch_load_dword v30, off, s26 offset:40
	scratch_load_dword v31, off, s26 offset:72
	scratch_load_dword v32, off, s26 offset:104
	scratch_load_dword v33, off, s23 offset:12
	scratch_load_dword v34, off, s26 offset:44
	scratch_load_dword v35, off, s26 offset:76
	scratch_load_dword v36, off, s26 offset:108
	s_waitcnt vmcnt(18)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v18, v6
	;;#ASMEND
	s_waitcnt vmcnt(14)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v22, v6
	;;#ASMEND
	s_waitcnt vmcnt(13)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v23, v6
	;;#ASMEND
	s_waitcnt vmcnt(12)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v24, v6
	;;#ASMEND
	s_waitcnt vmcnt(11)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v19, v25
	;;#ASMEND
	s_waitcnt vmcnt(10)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v26, v25
	;;#ASMEND
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v27, v25
	;;#ASMEND
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v28, v25
	;;#ASMEND
	s_waitcnt vmcnt(7)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v20, v29
	;;#ASMEND
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v30, v29
	;;#ASMEND
	s_waitcnt vmcnt(5)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v31, v29
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v32, v29
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v21, v33
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v34, v33
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v35, v33
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v36, v33
	;;#ASMEND
.LBB1_16:                               ;   in Loop: Header=BB1_14 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_addk_i32 s21, 0x200
	s_cmp_lt_u32 s21, s2
	v_add_u32_e32 v17, 0x400, v17
	s_cbranch_scc0 .LBB1_18
; %bb.17:                               ;   in Loop: Header=BB1_14 Depth=2
	s_mov_b32 s23, s22
	s_branch .LBB1_14
.LBB1_18:                               ;   in Loop: Header=BB1_10 Depth=1
	v_add_u32_e32 v6, s21, v1
	v_cmp_gt_u32_e32 vcc, s2, v6
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_20
; %bb.19:                               ;   in Loop: Header=BB1_10 Depth=1
	s_lshl_b32 s21, s22, 4
	s_and_b32 s21, s21, 16
	s_add_i32 s22, s21, 16
	s_addk_i32 s21, 0x90
	scratch_load_dword v6, off, s21
	scratch_load_dword v17, off, s22
	scratch_load_dword v18, off, s22 offset:4
	scratch_load_dword v19, off, s22 offset:8
	scratch_load_dword v20, off, s22 offset:12
	scratch_load_dword v21, off, s22 offset:32
	scratch_load_dword v22, off, s22 offset:64
	scratch_load_dword v23, off, s22 offset:96
	scratch_load_dword v24, off, s21 offset:4
	scratch_load_dword v25, off, s22 offset:36
	scratch_load_dword v26, off, s22 offset:68
	scratch_load_dword v27, off, s22 offset:100
	scratch_load_dword v28, off, s21 offset:8
	scratch_load_dword v29, off, s22 offset:40
	scratch_load_dword v30, off, s22 offset:72
	scratch_load_dword v31, off, s22 offset:104
	scratch_load_dword v32, off, s21 offset:12
	scratch_load_dword v33, off, s22 offset:44
	scratch_load_dword v34, off, s22 offset:76
	scratch_load_dword v35, off, s22 offset:108
	s_waitcnt vmcnt(18)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v17, v6
	;;#ASMEND
	s_waitcnt vmcnt(14)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v21, v6
	;;#ASMEND
	s_waitcnt vmcnt(13)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v22, v6
	;;#ASMEND
	s_waitcnt vmcnt(12)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v23, v6
	;;#ASMEND
	s_waitcnt vmcnt(11)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v18, v24
	;;#ASMEND
	s_waitcnt vmcnt(10)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v25, v24
	;;#ASMEND
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v26, v24
	;;#ASMEND
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v27, v24
	;;#ASMEND
	s_waitcnt vmcnt(7)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v19, v28
	;;#ASMEND
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v29, v28
	;;#ASMEND
	s_waitcnt vmcnt(5)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v30, v28
	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v31, v28
	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v20, v32
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v16, v33, v32
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v15, v34, v32
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v35, v32
	;;#ASMEND
.LBB1_20:                               ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[14:15]
	;;#ASMSTART
	s_nop 0
	v_add_f32 v3, v3, v3 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v3, v3, v3 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v3, v3, v3 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v3, v3, v3 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v3, v3, v3 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v3, v3, v3 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v15, v15, v15 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v15, v15, v15 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v15, v15, v15 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v15, v15, v15 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v15, v15, v15 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v15, v15, v15 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v5, v5, v5 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v5, v5, v5 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v5, v5, v5 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v5, v5, v5 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v5, v5, v5 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v5, v5, v5 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[14:15], s[12:13]
	s_cbranch_execz .LBB1_9
; %bb.21:                               ;   in Loop: Header=BB1_10 Depth=1
	v_cvt_f16_f32_e32 v6, v3
	v_mov_b32_e32 v3, v7
	v_cvt_f16_f32_e32 v18, v16
	v_lshl_add_u64 v[16:17], v[2:3], 1, s[8:9]
	global_store_short v[16:17], v6, off
	v_add_u32_e32 v6, s3, v2
	v_lshl_add_u64 v[16:17], v[6:7], 1, s[8:9]
	global_store_short v[16:17], v18, off
	v_cvt_f16_f32_e32 v3, v15
	v_add_u32_e32 v6, s3, v6
	v_lshl_add_u64 v[16:17], v[6:7], 1, s[8:9]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[16:17], v3, off
	v_add_u32_e32 v6, s3, v6
	v_lshl_add_u64 v[16:17], v[6:7], 1, s[8:9]
	global_store_short v[16:17], v5, off
	s_branch .LBB1_9
.LBB1_22:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 176
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 38
		.amdhsa_next_free_sgpr 27
		.amdhsa_accum_offset 40
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end1:
	.size	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end1-_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2152
; NumSgprs: 33
; NumVgprs: 38
; NumAgprs: 0
; TotalNumVgprs: 38
; ScratchSize: 176
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 38
; AccumOffset: 40
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v8, 3, v2
	s_lshl_b32 s22, s2, 2
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB2_6
; %bb.1:
	s_min_i32 s18, s22, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v8
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr14_sgpr15
	s_branch .LBB2_3
.LBB2_2:                                ;   in Loop: Header=BB2_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_and_b64 s[16:17], exec, s[14:15]
	s_or_b64 s[0:1], s[16:17], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_5
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v5
	v_cmp_gt_u32_e32 vcc, s18, v0
	s_or_b64 s[14:15], s[14:15], exec
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[10:13], v[6:7], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s18
	s_cselect_b64 s[20:21], -1, 0
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[10:13]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[14:15], s[14:15], s[20:21]
	s_branch .LBB2_2
.LBB2_5:
	s_or_b64 exec, exec, s[0:1]
.LBB2_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_39
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v10, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v10
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_39
; %bb.8:
	v_cmp_gt_u32_e64 s[0:1], s2, v8
	s_cmpk_gt_u32 s2, 0x200
	v_add_u32_e32 v9, 0xfffffe00, v8
	v_cmp_eq_u32_e64 s[12:13], 63, v2
	s_mul_i32 s20, s11, s10
	v_lshlrev_b32_e32 v20, 1, v8
	s_cselect_b64 s[6:7], -1, 0
	s_lshl_b32 s23, s2, 1
	v_add_u32_e32 v21, s23, v20
	v_mov_b32_e32 v0, 16
	v_add_u32_e32 v22, 32, v0
	v_add_u32_e32 v23, s23, v21
	v_add_u32_e32 v24, 64, v0
	v_add_u32_e32 v25, s23, v23
	v_add_u32_e32 v26, 0x60, v0
	s_mul_i32 s21, s2, 6
	s_addk_i32 s21, 0x400
	s_addk_i32 s22, 0x400
	s_addk_i32 s23, 0x400
	v_mad_u64_u32 v[12:13], s[10:11], s2, v10, v[8:9]
	s_mul_i32 s24, s20, s2
	s_mov_b64 s[10:11], 0
	v_mov_b32_e32 v15, 0
	s_mov_b32 s14, 0
	v_cndmask_b32_e64 v0, 0, 1, s[6:7]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	s_mov_b32 s25, 0x7f800000
	s_movk_i32 s26, 0x7fff
	s_branch .LBB2_11
.LBB2_9:                                ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_add_u32_e32 v14, s3, v14
	v_lshl_add_u64 v[2:3], v[14:15], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_add_u32_e32 v10, s20, v10
	v_cmp_le_u32_e32 vcc, s3, v10
	s_or_b64 s[10:11], vcc, s[10:11]
	v_add_u32_e32 v12, s24, v12
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB2_39
.LBB2_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_15 Depth 2
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB2_13
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mad_u64_u32 v[0:1], s[18:19], v10, s2, v[8:9]
	v_mov_b32_e32 v1, v15
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	ds_read_b128 v[4:7], v20
	ds_read2_b64 v[16:19], v21 offset1:1
	ds_read2_b32 v[30:31], v23 offset0:2 offset1:3
	ds_read2_b32 v[28:29], v23 offset1:1
	ds_read2_b64 v[32:35], v25 offset1:1
	s_waitcnt lgkmcnt(4)
	scratch_store_dwordx4 off, v[4:7], off offset:16
	s_waitcnt lgkmcnt(3)
	scratch_store_dwordx2 v22, v[16:17], off
	scratch_store_dwordx2 v22, v[18:19], off offset:8
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 v24, v[28:31], off
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 v26, v[32:33], off
	s_waitcnt vmcnt(5)
	scratch_store_dwordx4 off, v[0:3], off offset:144
	scratch_store_dwordx2 v26, v[34:35], off offset:8
.LBB2_13:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_mov_b32 s15, s14
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB2_19
; %bb.14:                               ;   in Loop: Header=BB2_11 Depth=1
	s_mov_b32 s19, 0
	s_movk_i32 s18, 0x200
	v_mov_b64_e32 v[16:17], s[14:15]
	v_mov_b32_e32 v11, v20
	v_mov_b64_e32 v[18:19], s[14:15]
.LBB2_15:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s15, s19, 1
	v_add_u32_e32 v0, s18, v8
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_17
; %bb.16:                               ;   in Loop: Header=BB2_15 Depth=2
	v_add_u32_e32 v14, s18, v12
	v_lshl_add_u64 v[0:1], v[14:15], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	s_lshl_b32 s27, s15, 4
	ds_read_b128 v[4:7], v11 offset:1024
	v_add_u32_e32 v13, s23, v11
	s_lshl_b32 s19, s19, 4
	s_and_b32 s27, s27, 16
	s_and_b32 s19, s19, 16
	s_add_i32 s28, s27, 16
	s_add_i32 s29, s19, 16
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[4:7], s28
	scratch_load_dwordx4 v[4:7], off, s29
	ds_read2_b64 v[28:31], v13 offset1:1
	v_add_u32_e32 v13, s22, v11
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[30:31], s28 offset:40
	scratch_store_dwordx2 off, v[28:29], s28 offset:32
	scratch_load_dwordx4 v[28:31], off, s29 offset:32
	ds_read2_b32 v[34:35], v13 offset0:2 offset1:3
	ds_read2_b32 v[32:33], v13 offset1:1
	v_add_u32_e32 v13, s21, v11
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[32:35], s28 offset:64
	scratch_load_dwordx4 v[32:35], off, s29 offset:64
	ds_read2_b64 v[36:39], v13 offset1:1
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx2 off, v[38:39], s28 offset:104
	scratch_store_dwordx2 off, v[36:37], s28 offset:96
	scratch_load_dwordx4 v[36:39], off, s29 offset:96
	s_addk_i32 s27, 0x90
	s_addk_i32 s19, 0x90
	s_waitcnt vmcnt(10)
	scratch_store_dwordx4 off, v[0:3], s27
	scratch_load_dwordx4 v[0:3], off, s19
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v41, 0xffff0000, v4
	v_lshlrev_b32_e32 v40, 16, v4
	v_and_b32_e32 v43, 0xffff0000, v5
	v_lshlrev_b32_e32 v42, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v6
	v_lshlrev_b32_e32 v4, 16, v6
	v_and_b32_e32 v45, 0xffff0000, v7
	v_lshlrev_b32_e32 v44, 16, v7
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v7, 0xffff0000, v28
	v_lshlrev_b32_e32 v6, 16, v28
	v_and_b32_e32 v47, 0xffff0000, v29
	v_lshlrev_b32_e32 v46, 16, v29
	v_and_b32_e32 v29, 0xffff0000, v30
	v_lshlrev_b32_e32 v28, 16, v30
	v_and_b32_e32 v49, 0xffff0000, v31
	v_lshlrev_b32_e32 v48, 16, v31
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v31, 0xffff0000, v32
	v_lshlrev_b32_e32 v30, 16, v32
	v_and_b32_e32 v51, 0xffff0000, v33
	v_lshlrev_b32_e32 v50, 16, v33
	v_and_b32_e32 v33, 0xffff0000, v34
	v_lshlrev_b32_e32 v32, 16, v34
	v_and_b32_e32 v53, 0xffff0000, v35
	v_lshlrev_b32_e32 v52, 16, v35
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v35, 0xffff0000, v36
	v_lshlrev_b32_e32 v34, 16, v36
	v_and_b32_e32 v55, 0xffff0000, v37
	v_lshlrev_b32_e32 v54, 16, v37
	v_and_b32_e32 v37, 0xffff0000, v38
	v_lshlrev_b32_e32 v36, 16, v38
	v_and_b32_e32 v57, 0xffff0000, v39
	v_lshlrev_b32_e32 v56, 16, v39
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v39, 0xffff0000, v0
	v_lshlrev_b32_e32 v38, 16, v0
	v_and_b32_e32 v59, 0xffff0000, v1
	v_lshlrev_b32_e32 v58, 16, v1
	v_and_b32_e32 v1, 0xffff0000, v2
	v_lshlrev_b32_e32 v0, 16, v2
	v_and_b32_e32 v61, 0xffff0000, v3
	v_lshlrev_b32_e32 v60, 16, v3
	v_pk_mul_f32 v[2:3], v[40:41], v[38:39]
	v_pk_mul_f32 v[40:41], v[42:43], v[58:59]
	v_pk_mul_f32 v[4:5], v[4:5], v[0:1]
	v_pk_mul_f32 v[42:43], v[44:45], v[60:61]
	v_pk_mul_f32 v[6:7], v[6:7], v[38:39]
	v_pk_mul_f32 v[44:45], v[46:47], v[58:59]
	v_pk_mul_f32 v[28:29], v[28:29], v[0:1]
	v_pk_mul_f32 v[46:47], v[48:49], v[60:61]
	v_pk_mul_f32 v[30:31], v[30:31], v[38:39]
	v_pk_mul_f32 v[48:49], v[50:51], v[58:59]
	v_pk_mul_f32 v[32:33], v[32:33], v[0:1]
	v_pk_mul_f32 v[50:51], v[52:53], v[60:61]
	v_pk_mul_f32 v[34:35], v[34:35], v[38:39]
	v_pk_mul_f32 v[38:39], v[54:55], v[58:59]
	v_pk_mul_f32 v[0:1], v[36:37], v[0:1]
	v_pk_mul_f32 v[36:37], v[56:57], v[60:61]
	v_mov_b32_e32 v52, v6
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v2, v7
	v_mov_b32_e32 v6, v44
	v_mov_b32_e32 v7, v40
	v_mov_b32_e32 v40, v45
	v_mov_b32_e32 v44, v28
	v_mov_b32_e32 v45, v4
	v_mov_b32_e32 v4, v29
	v_mov_b32_e32 v28, v46
	v_mov_b32_e32 v29, v42
	v_mov_b32_e32 v42, v47
	v_mov_b32_e32 v46, v34
	v_mov_b32_e32 v47, v30
	v_mov_b32_e32 v30, v35
	v_mov_b32_e32 v34, v38
	v_mov_b32_e32 v35, v48
	v_mov_b32_e32 v48, v39
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v32
	v_mov_b32_e32 v32, v1
	v_mov_b32_e32 v0, v36
	v_mov_b32_e32 v1, v50
	v_mov_b32_e32 v50, v37
	v_pk_add_f32 v[2:3], v[52:53], v[2:3]
	v_pk_add_f32 v[6:7], v[6:7], v[40:41]
	v_pk_add_f32 v[4:5], v[44:45], v[4:5]
	v_pk_add_f32 v[28:29], v[28:29], v[42:43]
	v_pk_add_f32 v[30:31], v[46:47], v[30:31]
	v_pk_add_f32 v[34:35], v[34:35], v[48:49]
	v_pk_add_f32 v[32:33], v[38:39], v[32:33]
	v_pk_add_f32 v[0:1], v[0:1], v[50:51]
	v_pk_add_f32 v[2:3], v[18:19], v[2:3]
	v_pk_add_f32 v[16:17], v[16:17], v[30:31]
	v_pk_add_f32 v[2:3], v[2:3], v[6:7]
	v_pk_add_f32 v[6:7], v[16:17], v[34:35]
	v_pk_add_f32 v[2:3], v[2:3], v[4:5]
	v_pk_add_f32 v[4:5], v[6:7], v[32:33]
	v_pk_add_f32 v[18:19], v[2:3], v[28:29]
	v_pk_add_f32 v[16:17], v[4:5], v[0:1]
.LBB2_17:                               ;   in Loop: Header=BB2_15 Depth=2
	s_or_b64 exec, exec, s[16:17]
	s_addk_i32 s18, 0x200
	s_cmp_lt_u32 s18, s2
	v_add_u32_e32 v11, 0x400, v11
	s_cbranch_scc0 .LBB2_20
; %bb.18:                               ;   in Loop: Header=BB2_15 Depth=2
	s_mov_b32 s19, s15
	s_branch .LBB2_15
.LBB2_19:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mov_b64_e32 v[18:19], s[14:15]
	s_mov_b32 s15, 0
	s_movk_i32 s18, 0x200
	v_mov_b64_e32 v[16:17], v[18:19]
.LBB2_20:                               ;   in Loop: Header=BB2_11 Depth=1
	v_add_u32_e32 v0, s18, v9
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_22
; %bb.21:                               ;   in Loop: Header=BB2_11 Depth=1
	s_lshl_b32 s15, s15, 4
	s_and_b32 s15, s15, 16
	s_add_i32 s18, s15, 16
	scratch_load_dwordx4 v[0:3], off, s18
	s_addk_i32 s15, 0x90
	scratch_load_dwordx4 v[4:7], off, s15
	scratch_load_dwordx4 v[28:31], off, s18 offset:32
	scratch_load_dwordx4 v[32:35], off, s18 offset:64
	scratch_load_dwordx4 v[36:39], off, s18 offset:96
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v41, 0xffff0000, v0
	v_lshlrev_b32_e32 v40, 16, v0
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v43, 0xffff0000, v4
	v_lshlrev_b32_e32 v42, 16, v4
	v_and_b32_e32 v45, 0xffff0000, v1
	v_lshlrev_b32_e32 v44, 16, v1
	v_and_b32_e32 v1, 0xffff0000, v5
	v_lshlrev_b32_e32 v0, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v2
	v_lshlrev_b32_e32 v4, 16, v2
	v_and_b32_e32 v47, 0xffff0000, v6
	v_lshlrev_b32_e32 v46, 16, v6
	v_and_b32_e32 v49, 0xffff0000, v3
	v_lshlrev_b32_e32 v48, 16, v3
	v_and_b32_e32 v3, 0xffff0000, v7
	v_lshlrev_b32_e32 v2, 16, v7
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v7, 0xffff0000, v28
	v_lshlrev_b32_e32 v6, 16, v28
	v_and_b32_e32 v51, 0xffff0000, v29
	v_lshlrev_b32_e32 v50, 16, v29
	v_and_b32_e32 v29, 0xffff0000, v30
	v_lshlrev_b32_e32 v28, 16, v30
	v_and_b32_e32 v53, 0xffff0000, v31
	v_lshlrev_b32_e32 v52, 16, v31
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v31, 0xffff0000, v32
	v_lshlrev_b32_e32 v30, 16, v32
	v_and_b32_e32 v55, 0xffff0000, v33
	v_lshlrev_b32_e32 v54, 16, v33
	v_and_b32_e32 v33, 0xffff0000, v34
	v_lshlrev_b32_e32 v32, 16, v34
	v_and_b32_e32 v57, 0xffff0000, v35
	v_lshlrev_b32_e32 v56, 16, v35
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v35, 0xffff0000, v36
	v_lshlrev_b32_e32 v34, 16, v36
	v_and_b32_e32 v59, 0xffff0000, v37
	v_lshlrev_b32_e32 v58, 16, v37
	v_and_b32_e32 v37, 0xffff0000, v38
	v_lshlrev_b32_e32 v36, 16, v38
	v_and_b32_e32 v61, 0xffff0000, v39
	v_lshlrev_b32_e32 v60, 16, v39
	v_pk_mul_f32 v[38:39], v[40:41], v[42:43]
	v_pk_mul_f32 v[40:41], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[4:5], v[46:47]
	v_pk_mul_f32 v[44:45], v[48:49], v[2:3]
	v_pk_mul_f32 v[6:7], v[6:7], v[42:43]
	v_pk_mul_f32 v[48:49], v[50:51], v[0:1]
	v_pk_mul_f32 v[28:29], v[28:29], v[46:47]
	v_pk_mul_f32 v[50:51], v[52:53], v[2:3]
	v_pk_mul_f32 v[30:31], v[30:31], v[42:43]
	v_pk_mul_f32 v[52:53], v[54:55], v[0:1]
	v_pk_mul_f32 v[32:33], v[32:33], v[46:47]
	v_pk_mul_f32 v[54:55], v[56:57], v[2:3]
	v_pk_mul_f32 v[34:35], v[34:35], v[42:43]
	v_pk_mul_f32 v[0:1], v[58:59], v[0:1]
	v_pk_mul_f32 v[36:37], v[36:37], v[46:47]
	v_pk_mul_f32 v[2:3], v[60:61], v[2:3]
	v_mov_b32_e32 v42, v6
	v_mov_b32_e32 v43, v38
	v_mov_b32_e32 v38, v7
	v_mov_b32_e32 v6, v48
	v_mov_b32_e32 v7, v40
	v_mov_b32_e32 v40, v49
	v_mov_b32_e32 v46, v28
	v_mov_b32_e32 v47, v4
	v_mov_b32_e32 v4, v29
	v_mov_b32_e32 v28, v50
	v_mov_b32_e32 v29, v44
	v_mov_b32_e32 v44, v51
	v_mov_b32_e32 v48, v34
	v_mov_b32_e32 v49, v30
	v_mov_b32_e32 v30, v35
	v_mov_b32_e32 v34, v0
	v_pk_add_f32 v[38:39], v[42:43], v[38:39]
	v_pk_add_f32 v[6:7], v[6:7], v[40:41]
	v_pk_add_f32 v[4:5], v[46:47], v[4:5]
	v_pk_add_f32 v[28:29], v[28:29], v[44:45]
	v_pk_add_f32 v[30:31], v[48:49], v[30:31]
	v_pk_add_f32 v[18:19], v[18:19], v[38:39]
	v_pk_add_f32 v[16:17], v[16:17], v[30:31]
	v_pk_add_f32 v[6:7], v[18:19], v[6:7]
	s_nop 0
	v_pk_add_f32 v[4:5], v[6:7], v[4:5]
	s_nop 0
	v_pk_add_f32 v[18:19], v[4:5], v[28:29]
	v_mov_b32_e32 v35, v52
	v_mov_b32_e32 v52, v1
	v_pk_add_f32 v[0:1], v[34:35], v[52:53]
	s_nop 0
	v_pk_add_f32 v[0:1], v[16:17], v[0:1]
	v_mov_b32_e32 v4, v36
	v_mov_b32_e32 v5, v32
	v_mov_b32_e32 v32, v37
	v_pk_add_f32 v[4:5], v[4:5], v[32:33]
	s_nop 0
	v_pk_add_f32 v[0:1], v[0:1], v[4:5]
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v54
	v_mov_b32_e32 v54, v3
	v_pk_add_f32 v[2:3], v[4:5], v[54:55]
	s_nop 0
	v_pk_add_f32 v[16:17], v[0:1], v[2:3]
.LBB2_22:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[16:17]
	;;#ASMSTART
	s_nop 0
	v_add_f32 v19, v19, v19 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v19, v19, v19 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v19, v19, v19 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v19, v19, v19 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v19, v19, v19 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v19, v19, v19 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v18, v18, v18 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v18, v18, v18 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v18, v18, v18 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v18, v18, v18 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v18, v18, v18 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v18, v18, v18 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v17, v17, v17 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v17, v17, v17 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v17, v17, v17 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v17, v17, v17 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v17, v17, v17 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v17, v17, v17 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v16, v16, v16 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[16:17], s[12:13]
	s_cbranch_execz .LBB2_10
; %bb.23:                               ;   in Loop: Header=BB2_11 Depth=1
	v_and_b32_e32 v0, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc, s25, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
; %bb.24:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v19, 16, 1
	v_add3_u32 v0, v19, v0, s26
; %bb.25:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.26:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v19
	v_cmp_eq_u32_sdwa vcc, v19, v15 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v19, vcc
; %bb.27:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_mov_b32_e32 v11, v15
	v_lshl_add_u64 v[2:3], v[10:11], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
	v_and_b32_e32 v0, 0x7f800000, v18
	v_cmp_ne_u32_e32 vcc, s25, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
; %bb.28:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v18, 16, 1
	v_add3_u32 v0, v18, v0, s26
                                        ; implicit-def: $vgpr18
; %bb.29:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.30:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v18
	v_cmp_eq_u32_sdwa vcc, v18, v15 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v18, vcc
; %bb.31:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_add_u32_e32 v14, s3, v10
	v_lshl_add_u64 v[2:3], v[14:15], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
	v_and_b32_e32 v0, 0x7f800000, v17
	v_cmp_ne_u32_e32 vcc, s25, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
; %bb.32:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v17, 16, 1
	v_add3_u32 v0, v17, v0, s26
; %bb.33:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[18:19], s[18:19]
; %bb.34:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v17
	v_cmp_eq_u32_sdwa vcc, v17, v15 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v17, vcc
; %bb.35:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[18:19]
	v_add_u32_e32 v14, s3, v14
	v_lshl_add_u64 v[2:3], v[14:15], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
	v_and_b32_e32 v0, 0x7f800000, v16
	v_cmp_ne_u32_e32 vcc, s25, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[18:19], vcc
	s_xor_b64 s[18:19], exec, s[18:19]
; %bb.36:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v16, 16, 1
	v_add3_u32 v0, v16, v0, s26
                                        ; implicit-def: $vgpr16
; %bb.37:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[18:19], s[18:19]
	s_cbranch_execz .LBB2_9
; %bb.38:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v16
	v_cmp_eq_u32_sdwa vcc, v16, v15 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v16, vcc
	s_branch .LBB2_9
.LBB2_39:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 176
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 62
		.amdhsa_next_free_sgpr 30
		.amdhsa_accum_offset 64
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end2:
	.size	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end2-_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3032
; NumSgprs: 36
; NumVgprs: 62
; NumAgprs: 0
; TotalNumVgprs: 62
; ScratchSize: 176
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 36
; NumVGPRsForWavesPerEU: 62
; AccumOffset: 64
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 15
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_46fa8fceecced319,@object ; @__hip_cuid_46fa8fceecced319
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_46fa8fceecced319
__hip_cuid_46fa8fceecced319:
	.byte	0                               ; 0x0
	.size	__hip_cuid_46fa8fceecced319, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_46fa8fceecced319
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
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
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20matrixMultiplySharedPfS_S_iiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         _Z20matrixMultiplySharedPfS_S_iiiiii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     40
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 176
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     38
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 176
    .sgpr_count:     36
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     62
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
