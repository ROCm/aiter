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
	v_bfe_u32 v4, v0, 10, 10
	v_and_b32_e32 v1, 0x3ff, v0
	v_lshlrev_b32_e32 v0, 3, v1
	s_lshl_b32 s20, s2, 2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB1_6
; %bb.1:
	s_min_i32 s21, s20, 0x8000
	v_lshlrev_b32_e32 v2, 4, v1
	v_lshl_add_u32 v5, v4, 10, v2
	v_lshl_add_u32 v6, v4, 9, v0
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v3, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB1_5
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v2, s13, v6
	v_cmp_gt_u32_e32 vcc, s21, v2
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshl_add_u64 v[8:9], v[2:3], 1, s[6:7]
	global_load_dwordx4 v[8:11], v[8:9], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s21
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[8:11]
	v_add_u32_e32 v5, 0x4000, v5
	s_or_b64 s[16:17], s[16:17], s[22:23]
	s_branch .LBB1_2
.LBB1_5:
	s_or_b64 exec, exec, s[0:1]
.LBB1_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_17
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v2, s12, v4
	v_cmp_gt_u32_e32 vcc, s3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_17
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v1
	s_mul_i32 s16, s11, s10
	s_mul_i32 s17, s2, 6
	v_lshlrev_b32_e32 v1, 4, v1
	s_lshl_b32 s18, s2, 1
	v_mad_u64_u32 v[4:5], s[6:7], s2, v2, v[0:1]
	s_mul_i32 s19, s16, s2
	s_mov_b64 s[10:11], 0
	v_cndmask_b32_e64 v3, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v3
	v_mov_b32_e32 v7, 0
	s_branch .LBB1_10
.LBB1_9:                                ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v2, s16, v2
	v_cmp_le_u32_e32 vcc, s3, v2
	s_or_b64 s[10:11], vcc, s[10:11]
	v_add_u32_e32 v4, s19, v4
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB1_17
.LBB1_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_13 Depth 2
	s_mov_b32 s14, 0
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v5, v7
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v9, v7
	v_mov_b32_e32 v3, v7
	s_cbranch_vccnz .LBB1_15
; %bb.11:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v10, v1
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v5, 0
	s_branch .LBB1_13
.LBB1_12:                               ;   in Loop: Header=BB1_13 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_addk_i32 s14, 0x200
	s_cmp_ge_u32 s14, s2
	v_add_u32_e32 v10, 0x400, v10
	s_cbranch_scc1 .LBB1_15
.LBB1_13:                               ;   Parent Loop BB1_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v6, s14, v0
	v_cmp_gt_u32_e32 vcc, s2, v6
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB1_12
; %bb.14:                               ;   in Loop: Header=BB1_13 Depth=2
	v_add_u32_e32 v6, s14, v4
	v_lshl_add_u64 v[12:13], v[6:7], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	ds_read_b128 v[16:19], v10
	v_add_u32_e32 v6, s18, v10
	v_add_u32_e32 v11, s20, v10
	v_add_u32_e32 v24, s17, v10
	ds_read_b128 v[20:23], v6
	ds_read2_b32 v[28:29], v11 offset1:1
	ds_read_b128 v[24:27], v24
	ds_read2_b32 v[30:31], v11 offset0:2 offset1:3
	s_waitcnt vmcnt(0) lgkmcnt(4)
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v16, v12
	;;#ASMEND
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	v_dot2c_f32_f16 v9, v20, v12
	;;#ASMEND
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_dot2c_f32_f16 v8, v28, v12
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v24, v12
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v17, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v9, v21, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v8, v29, v13
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v25, v13
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v18, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v9, v22, v14
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_dot2c_f32_f16 v8, v30, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v26, v14
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v3, v19, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v9, v23, v15
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_dot2c_f32_f16 v8, v31, v15
	;;#ASMEND
	;;#ASMSTART
	v_dot2c_f32_f16 v5, v27, v15
	;;#ASMEND
	s_branch .LBB1_12
.LBB1_15:                               ;   in Loop: Header=BB1_10 Depth=1
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
	v_add_f32 v9, v9, v9 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v9, v9, v9 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v9, v9, v9 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v9, v9, v9 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v9, v9, v9 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v9, v9, v9 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v8, v8, v8 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v8, v8, v8 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v8, v8, v8 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v8, v8, v8 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v8, v8, v8 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v8, v8, v8 row_bcast:31 bound_ctrl:0
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
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB1_9
; %bb.16:                               ;   in Loop: Header=BB1_10 Depth=1
	v_cvt_f16_f32_e32 v6, v3
	v_mov_b32_e32 v3, v7
	v_cvt_f16_f32_e32 v9, v9
	v_lshl_add_u64 v[10:11], v[2:3], 1, s[8:9]
	global_store_short v[10:11], v6, off
	v_add_u32_e32 v6, s3, v2
	v_lshl_add_u64 v[10:11], v[6:7], 1, s[8:9]
	global_store_short v[10:11], v9, off
	v_cvt_f16_f32_e32 v3, v8
	v_add_u32_e32 v6, s3, v6
	v_lshl_add_u64 v[8:9], v[6:7], 1, s[8:9]
	v_cvt_f16_f32_e32 v5, v5
	global_store_short v[8:9], v3, off
	v_add_u32_e32 v6, s3, v6
	v_lshl_add_u64 v[8:9], v[6:7], 1, s[8:9]
	global_store_short v[8:9], v5, off
	s_branch .LBB1_9
.LBB1_17:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 32
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 32
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
; codeLenInByte = 1224
; NumSgprs: 30
; NumVgprs: 32
; NumAgprs: 0
; TotalNumVgprs: 32
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 32
; AccumOffset: 32
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
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
	v_lshlrev_b32_e32 v4, 3, v2
	s_lshl_b32 s20, s2, 2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB2_6
; %bb.1:
	s_min_i32 s21, s20, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v5, v3, 10, v0
	v_lshl_add_u32 v6, v3, 9, v4
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB2_3
.LBB2_2:                                ;   in Loop: Header=BB2_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_5
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v6
	v_cmp_gt_u32_e32 vcc, s21, v0
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	v_lshl_add_u64 v[8:9], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[8:11], v[8:9], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s21
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[8:11]
	v_add_u32_e32 v5, 0x4000, v5
	s_or_b64 s[16:17], s[16:17], s[22:23]
	s_branch .LBB2_2
.LBB2_5:
	s_or_b64 exec, exec, s[0:1]
.LBB2_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_33
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v6, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v6
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_33
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s16, s11, s10
	s_mul_i32 s17, s2, 6
	v_lshlrev_b32_e32 v5, 4, v2
	s_lshl_b32 s18, s2, 1
	v_mad_u64_u32 v[8:9], s[6:7], s2, v6, v[4:5]
	s_mul_i32 s19, s16, s2
	s_mov_b64 s[10:11], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v11, 0
	s_mov_b32 s21, 0x7f800000
	s_movk_i32 s22, 0x7fff
	s_branch .LBB2_11
.LBB2_9:                                ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v10, s3, v10
	v_lshl_add_u64 v[2:3], v[10:11], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[12:13]
	v_add_u32_e32 v6, s16, v6
	v_cmp_le_u32_e32 vcc, s3, v6
	s_or_b64 s[10:11], vcc, s[10:11]
	v_add_u32_e32 v8, s19, v8
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB2_33
.LBB2_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_14 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	v_mov_b32_e32 v13, v11
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v15, v11
	v_mov_b32_e32 v14, v11
	s_cbranch_vccnz .LBB2_16
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=1
	s_mov_b32 s14, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v15, v14
	v_mov_b32_e32 v12, v14
	v_mov_b32_e32 v13, v14
	s_branch .LBB2_14
.LBB2_13:                               ;   in Loop: Header=BB2_14 Depth=2
	s_or_b64 exec, exec, s[12:13]
	s_addk_i32 s14, 0x200
	s_cmp_ge_u32 s14, s2
	v_add_u32_e32 v7, 0x400, v7
	s_cbranch_scc1 .LBB2_16
.LBB2_14:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v0, s14, v4
	v_cmp_gt_u32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB2_13
; %bb.15:                               ;   in Loop: Header=BB2_14 Depth=2
	v_add_u32_e32 v10, s14, v8
	v_lshl_add_u64 v[0:1], v[10:11], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	v_add_u32_e32 v9, s18, v7
	v_add_u32_e32 v10, s20, v7
	v_add_u32_e32 v24, s17, v7
	ds_read_b128 v[16:19], v7
	ds_read_b128 v[20:23], v9
	ds_read2_b32 v[28:29], v10 offset1:1
	ds_read2_b32 v[30:31], v10 offset0:2 offset1:3
	ds_read_b128 v[24:27], v24
	s_waitcnt lgkmcnt(4)
	v_and_b32_e32 v33, 0xffff0000, v16
	v_lshlrev_b32_e32 v32, 16, v16
	v_and_b32_e32 v35, 0xffff0000, v17
	v_lshlrev_b32_e32 v34, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v18
	v_lshlrev_b32_e32 v16, 16, v18
	v_and_b32_e32 v37, 0xffff0000, v19
	v_lshlrev_b32_e32 v36, 16, v19
	s_waitcnt lgkmcnt(3)
	v_and_b32_e32 v19, 0xffff0000, v20
	v_lshlrev_b32_e32 v18, 16, v20
	v_and_b32_e32 v39, 0xffff0000, v21
	v_lshlrev_b32_e32 v38, 16, v21
	v_and_b32_e32 v21, 0xffff0000, v22
	v_lshlrev_b32_e32 v20, 16, v22
	v_and_b32_e32 v41, 0xffff0000, v23
	v_lshlrev_b32_e32 v40, 16, v23
	s_waitcnt lgkmcnt(2)
	v_and_b32_e32 v23, 0xffff0000, v28
	v_lshlrev_b32_e32 v22, 16, v28
	v_and_b32_e32 v43, 0xffff0000, v29
	v_lshlrev_b32_e32 v42, 16, v29
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v29, 0xffff0000, v30
	v_lshlrev_b32_e32 v28, 16, v30
	v_and_b32_e32 v45, 0xffff0000, v31
	v_lshlrev_b32_e32 v44, 16, v31
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v31, 0xffff0000, v24
	v_lshlrev_b32_e32 v30, 16, v24
	v_and_b32_e32 v47, 0xffff0000, v25
	v_lshlrev_b32_e32 v46, 16, v25
	v_and_b32_e32 v25, 0xffff0000, v26
	v_lshlrev_b32_e32 v24, 16, v26
	v_and_b32_e32 v49, 0xffff0000, v27
	v_lshlrev_b32_e32 v48, 16, v27
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v27, 0xffff0000, v0
	v_lshlrev_b32_e32 v26, 16, v0
	v_and_b32_e32 v51, 0xffff0000, v1
	v_lshlrev_b32_e32 v50, 16, v1
	v_and_b32_e32 v1, 0xffff0000, v2
	v_lshlrev_b32_e32 v0, 16, v2
	v_and_b32_e32 v53, 0xffff0000, v3
	v_lshlrev_b32_e32 v52, 16, v3
	v_pk_mul_f32 v[2:3], v[32:33], v[26:27]
	v_pk_mul_f32 v[32:33], v[34:35], v[50:51]
	v_pk_mul_f32 v[16:17], v[16:17], v[0:1]
	v_pk_mul_f32 v[34:35], v[36:37], v[52:53]
	v_pk_mul_f32 v[18:19], v[18:19], v[26:27]
	v_pk_mul_f32 v[36:37], v[38:39], v[50:51]
	v_pk_mul_f32 v[20:21], v[20:21], v[0:1]
	v_pk_mul_f32 v[38:39], v[40:41], v[52:53]
	v_pk_mul_f32 v[22:23], v[22:23], v[26:27]
	v_pk_mul_f32 v[40:41], v[42:43], v[50:51]
	v_pk_mul_f32 v[28:29], v[28:29], v[0:1]
	v_pk_mul_f32 v[42:43], v[44:45], v[52:53]
	v_pk_mul_f32 v[26:27], v[30:31], v[26:27]
	v_pk_mul_f32 v[30:31], v[46:47], v[50:51]
	v_pk_mul_f32 v[0:1], v[24:25], v[0:1]
	v_pk_mul_f32 v[24:25], v[48:49], v[52:53]
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v18
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v2, v32
	v_mov_b32_e32 v3, v36
	v_mov_b32_e32 v36, v33
	v_mov_b32_e32 v32, v16
	v_mov_b32_e32 v33, v20
	v_mov_b32_e32 v20, v17
	v_mov_b32_e32 v16, v34
	v_mov_b32_e32 v17, v38
	v_mov_b32_e32 v38, v35
	v_mov_b32_e32 v34, v22
	v_mov_b32_e32 v35, v26
	v_mov_b32_e32 v26, v23
	v_pk_add_f32 v[18:19], v[44:45], v[18:19]
	v_pk_add_f32 v[2:3], v[2:3], v[36:37]
	v_pk_add_f32 v[20:21], v[32:33], v[20:21]
	v_pk_add_f32 v[16:17], v[16:17], v[38:39]
	v_pk_add_f32 v[14:15], v[14:15], v[18:19]
	s_nop 0
	v_pk_add_f32 v[2:3], v[14:15], v[2:3]
	s_nop 0
	v_pk_add_f32 v[2:3], v[2:3], v[20:21]
	s_nop 0
	v_pk_add_f32 v[14:15], v[2:3], v[16:17]
	v_pk_add_f32 v[2:3], v[34:35], v[26:27]
	s_nop 0
	v_pk_add_f32 v[2:3], v[12:13], v[2:3]
	v_mov_b32_e32 v12, v40
	v_mov_b32_e32 v13, v30
	v_mov_b32_e32 v30, v41
	v_pk_add_f32 v[12:13], v[12:13], v[30:31]
	s_nop 0
	v_pk_add_f32 v[2:3], v[2:3], v[12:13]
	v_mov_b32_e32 v12, v28
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v0, v29
	v_pk_add_f32 v[0:1], v[12:13], v[0:1]
	s_nop 0
	v_pk_add_f32 v[0:1], v[2:3], v[0:1]
	v_mov_b32_e32 v2, v42
	v_mov_b32_e32 v3, v24
	v_mov_b32_e32 v24, v43
	v_pk_add_f32 v[2:3], v[2:3], v[24:25]
	s_nop 0
	v_pk_add_f32 v[12:13], v[0:1], v[2:3]
	s_branch .LBB2_13
.LBB2_16:                               ;   in Loop: Header=BB2_11 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v14, v14, v14 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v14, v14, v14 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v14, v14, v14 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v14, v14, v14 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v14, v14, v14 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v14, v14, v14 row_bcast:31 bound_ctrl:0
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
	v_add_f32 v12, v12, v12 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v12, v12, v12 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v12, v12, v12 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v12, v12, v12 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v12, v12, v12 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v12, v12, v12 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	;;#ASMSTART
	s_nop 0
	v_add_f32 v13, v13, v13 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v13, v13, v13 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v13, v13, v13 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v13, v13, v13 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v13, v13, v13 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v13, v13, v13 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[12:13], s[0:1]
	s_cbranch_execz .LBB2_10
; %bb.17:                               ;   in Loop: Header=BB2_11 Depth=1
	v_and_b32_e32 v0, 0x7f800000, v14
	v_cmp_ne_u32_e32 vcc, s21, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.18:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v14, 16, 1
	v_add3_u32 v0, v14, v0, s22
; %bb.19:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.20:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v14
	v_cmp_eq_u32_sdwa vcc, v14, v11 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v14, vcc
; %bb.21:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_mov_b32_e32 v7, v11
	v_lshl_add_u64 v[2:3], v[6:7], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
	v_and_b32_e32 v0, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc, s21, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.22:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v15, 16, 1
	v_add3_u32 v0, v15, v0, s22
                                        ; implicit-def: $vgpr15
; %bb.23:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.24:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v15
	v_cmp_eq_u32_sdwa vcc, v15, v11 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v15, vcc
; %bb.25:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v10, s3, v6
	v_lshl_add_u64 v[2:3], v[10:11], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
	v_and_b32_e32 v0, 0x7f800000, v12
	v_cmp_ne_u32_e32 vcc, s21, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.26:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v12, 16, 1
	v_add3_u32 v0, v12, v0, s22
; %bb.27:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
; %bb.28:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v12
	v_cmp_eq_u32_sdwa vcc, v12, v11 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v12, vcc
; %bb.29:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v10, s3, v10
	v_lshl_add_u64 v[2:3], v[10:11], 1, s[8:9]
	global_store_short_d16_hi v[2:3], v0, off
	v_and_b32_e32 v0, 0x7f800000, v13
	v_cmp_ne_u32_e32 vcc, s21, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[14:15], exec, s[14:15]
; %bb.30:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v0, v13, 16, 1
	v_add3_u32 v0, v13, v0, s22
                                        ; implicit-def: $vgpr13
; %bb.31:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[14:15], s[14:15]
	s_cbranch_execz .LBB2_9
; %bb.32:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v0, 0x10000, v13
	v_cmp_eq_u32_sdwa vcc, v13, v11 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v13, vcc
	s_branch .LBB2_9
.LBB2_33:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 54
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 56
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
; codeLenInByte = 1984
; NumSgprs: 30
; NumVgprs: 54
; NumAgprs: 0
; TotalNumVgprs: 54
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 54
; AccumOffset: 56
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 13
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_54af5210437cd3a5,@object ; @__hip_cuid_54af5210437cd3a5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_54af5210437cd3a5
__hip_cuid_54af5210437cd3a5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_54af5210437cd3a5, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_54af5210437cd3a5
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
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     32
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
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi1ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     54
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
