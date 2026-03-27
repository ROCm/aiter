	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_gemm2_0
	.p2align	8
	.type	moe_gemm2_0,@function
moe_gemm2_0:
	s_load_dwordx2 s[8:9], s[0:1], 0x40
	s_load_dword s33, s[0:1], 0x50
	s_load_dwordx2 s[6:7], s[0:1], 0x58
	s_mov_b32 s27, 0x27000
	s_mov_b32 s10, 4
	s_waitcnt lgkmcnt(0)
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, s27
	buffer_load_dword v37, off, s[8:11], 0
	s_load_dwordx16 s[8:23], s[0:1], 0x0
	s_mov_b32 s4, s3
	s_ashr_i32 s3, s2, 31
	v_lshlrev_b32_e32 v1, 4, v0
	v_bfe_u32 v3, v0, 4, 2
	v_and_b32_e32 v47, 15, v0
	s_movk_i32 s5, 0xf0
	v_lshrrev_b32_e32 v36, 6, v0
	v_lshlrev_b32_e32 v4, 1, v0
	v_and_b32_e32 v5, 0x70, v1
	v_lshlrev_b32_e32 v6, 4, v3
	s_lshl_b64 s[34:35], s[2:3], 7
	v_and_b32_e32 v1, 0xf0, v1
	v_lshlrev_b32_e32 v7, 4, v47
	v_lshlrev_b32_e32 v8, 8, v47
	v_lshl_or_b32 v38, v36, 5, s34
	v_mov_b32_e32 v39, s35
	v_bitop3_b32 v42, v4, v5, s5 bitop3:0x6c
	v_lshl_or_b32 v51, v3, 8, v1
	v_xor_b32_e32 v1, v6, v7
	v_bitop3_b32 v52, v6, v8, v7 bitop3:0xde
	v_or_b32_e32 v4, 0x1000, v8
	v_or_b32_e32 v5, 0x2000, v8
	v_or_b32_e32 v10, 64, v6
	v_bitop3_b32 v6, v6, v7, 64 bitop3:0x36
	s_ashr_i32 s3, s6, 31
	s_mov_b32 s2, s6
	v_cmp_gt_u32_e64 s[0:1], 64, v0
	v_lshlrev_b32_e32 v43, 2, v0
	v_and_b32_e32 v2, 63, v0
	v_mov_b32_e32 v41, s35
	v_or_b32_e32 v54, v1, v4
	v_or_b32_e32 v55, v1, v5
	v_or_b32_e32 v58, v6, v4
	v_or_b32_e32 v59, v6, v5
	v_lshlrev_b64 v[4:5], 1, v[38:39]
	s_lshl_b32 s5, s6, 3
	s_mul_i32 s35, s33, s6
	s_lshr_b64 s[2:3], s[2:3], 1
	v_lshrrev_b32_e32 v0, 1, v0
	s_mul_hi_i32 s26, s33, s6
	s_waitcnt lgkmcnt(0)
	s_and_b32 s29, s13, 0xffff
	s_mov_b32 s28, s12
	s_and_b32 s12, s5, 0xfffffc00
	s_lshr_b32 s5, s35, 1
	s_and_b32 s13, s2, -4
	v_lshl_or_b32 v4, v3, 3, v4
	s_lshl_b32 s2, s4, 10
	v_and_b32_e32 v0, 0x7c, v0
	v_lshlrev_b32_e32 v49, 2, v47
	v_or_b32_e32 v9, 0x3000, v8
	s_ashr_i32 s6, s6, 5
	s_mov_b32 s36, s16
	s_lshl_b32 s42, s7, 8
	s_mov_b32 s16, s22
	s_lshl_b32 s22, s7, 2
	s_lshl_b32 s7, s26, 31
	s_lshl2_add_u32 s3, s35, s5
	v_lshl_add_u64 v[44:45], s[8:9], 0, v[4:5]
	v_or_b32_e32 v48, s2, v0
	v_lshlrev_b32_e32 v0, 2, v2
	s_movk_i32 s8, 0xfc00
	s_mov_b32 s30, -1
	v_mov_b32_e32 v33, 0x27000
	v_mov_b32_e32 v35, 0
	s_mov_b32 s31, s27
	s_mov_b32 s39, s27
	s_mov_b32 s43, s27
	v_or_b32_e32 v40, 16, v38
	v_or_b32_e32 v53, 0x400, v51
	v_or_b32_e32 v56, v1, v9
	v_bitop3_b32 v57, v10, v8, v7 bitop3:0xde
	v_or_b32_e32 v60, v6, v9
	s_and_b32 s25, s11, 0xffff
	s_mov_b32 s24, s10
	s_and_b32 s15, s15, 0xffff
	s_and_b32 s37, s17, 0xffff
	s_and_b32 s41, s19, 0xffff
	s_mov_b32 s40, s18
	s_and_b32 s17, s23, 0xffff
	s_mov_b32 s23, s27
	s_and_b32 s21, s21, 0xffff
	s_mul_i32 s38, s6, 0x1c1c00
	s_add_i32 s26, s3, s7
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v32, v37, s6
	s_lshl_b32 s35, s4, 4
	v_or_b32_e32 v46, s2, v49
	s_lshl_b32 s44, s4, 8
	v_or_b32_e32 v50, s2, v43
	v_lshl_or_b32 v61, s4, 11, v0
	s_mov_b32 s9, -1
	s_movk_i32 s45, 0x100
	s_movk_i32 s46, 0x1c00
	s_mov_b32 s47, 0x9000000
	s_mov_b32 s48, 0x94900247
	s_mov_b32 s49, 0x246ddb4
	s_mov_b32 s50, 0x1c1c0
	s_movk_i32 s51, 0x3800
	v_lshlrev_b32_e32 v62, 2, v2
	s_branch .LBB0_3
.LBB0_1:
	s_or_b64 exec, exec, s[4:5]
.LBB0_2:
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s9, s9, 0
	s_add_i32 s35, s35, 4
	s_add_i32 s44, s44, 64
	s_cmp_lg_u64 s[8:9], 0
	v_add_u32_e32 v61, 0x200, v61
	s_barrier
	s_cbranch_scc0 .LBB0_22
.LBB0_3:
	v_mov_b32_e32 v0, s35
	buffer_load_dword v0, v0, s[20:23], 0 offen
	v_cmp_ge_u32_e32 vcc, s44, v37
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[2:3], s45, v0
	s_or_b64 s[2:3], vcc, s[2:3]
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_2
	v_add_u32_e32 v3, s8, v48
	v_add_u32_e32 v1, 0x400, v3
	v_xor_b32_e32 v4, 0x80, v1
	buffer_load_dword v2, v3, s[40:43], 0 offen offset:1024
	buffer_load_dword v1, v4, s[40:43], 0 offen
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_6
	v_add_u32_e32 v3, s8, v50
	buffer_load_dword v3, v3, s[40:43], 0 offen offset:1024
	s_waitcnt vmcnt(0)
	ds_write_b32 v43, v3 offset:32768
.LBB0_6:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, 0xffffff, v2
	v_lshrrev_b32_e32 v4, 24, v2
	v_cmp_gt_u32_e32 vcc, s33, v3
	v_cmp_gt_u32_e64 s[2:3], s47, v2
	v_mad_u32_u24 v2, v2, 9, v4
	s_and_b64 vcc, s[2:3], vcc
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffffff, v1
	v_cndmask_b32_e32 v2, 0, v2, vcc
	v_cmp_gt_u32_e64 s[4:5], s33, v3
	v_mad_u64_u32 v[2:3], s[2:3], s13, v2, v[42:43]
	v_lshrrev_b32_e32 v4, 24, v1
	v_cmp_gt_u32_e64 s[6:7], s47, v1
	v_readfirstlane_b32 s2, v36
	v_mad_u32_u24 v1, v1, 9, v4
	s_lshl_b32 s10, s2, 10
	s_and_b64 vcc, s[6:7], s[4:5]
	s_mov_b32 m0, s10
	v_cndmask_b32_e32 v1, 0, v1, vcc
	buffer_load_dwordx4 v2, s[24:27], 0 offen lds
	v_mad_u64_u32 v[4:5], s[2:3], s13, v1, v[42:43]
	s_add_i32 m0, s10, 0x1000
	v_mul_lo_u32 v34, v0, s46
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	s_add_i32 m0, s10, 0x2000
	v_lshl_add_u64 v[0:1], v[38:39], 0, v[34:35]
	buffer_load_dwordx4 v2, s[24:27], 0 offen lds
	s_add_i32 m0, s10, 0x3000
	v_mov_b32_e32 v3, v35
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_alignbit_b32 v4, v1, v0, 4
	v_mul_hi_u32 v2, v4, s48
	v_lshrrev_b32_e32 v5, 4, v1
	v_mad_u64_u32 v[0:1], s[2:3], v5, s48, v[2:3]
	v_mov_b32_e32 v2, v0
	v_mad_u64_u32 v[2:3], s[2:3], v4, s49, v[2:3]
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v3, v35
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v1, v35
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	v_mad_u64_u32 v[0:1], s[2:3], v5, s49, v[0:1]
	v_alignbit_b32 v0, v1, v0, 10
	v_mul_lo_u32 v0, v0, s50
	v_sub_u32_e32 v3, v4, v0
	v_lshl_add_u64 v[0:1], v[40:41], 0, v[34:35]
	v_alignbit_b32 v2, v1, v0, 4
	v_mul_hi_u32 v4, v2, s48
	v_mov_b32_e32 v5, v35
	v_lshrrev_b32_e32 v6, 4, v1
	v_mad_u64_u32 v[0:1], s[2:3], v6, s48, v[4:5]
	v_mov_b32_e32 v4, v0
	v_mad_u64_u32 v[4:5], s[2:3], v2, s49, v[4:5]
	v_mov_b32_e32 v4, v5
	v_mov_b32_e32 v5, v35
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v1, v35
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[4:5]
	v_mad_u64_u32 v[0:1], s[2:3], v6, s49, v[0:1]
	v_alignbit_b32 v0, v1, v0, 10
	v_mul_lo_u32 v0, v0, s50
	v_sub_u32_e32 v5, v2, v0
	v_mov_b32_e32 v0, s14
	v_mov_b32_e32 v1, s15
	s_mov_b64 s[10:11], exec
.LBB0_7:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v32
	v_readfirstlane_b32 s7, v33
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[2:3], s[6:7], v[32:33]
	s_and_b64 s[2:3], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[2:3]
	buffer_load_dword v2, v61, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_7
	s_mov_b64 exec, s[10:11]
	s_mov_b64 s[10:11], exec
.LBB0_9:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v32
	v_readfirstlane_b32 s7, v33
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[2:3], s[6:7], v[32:33]
	s_and_b64 s[2:3], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[2:3]
	buffer_load_dword v4, v61, s[4:7], 0 offen offset:256
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_9
	s_mov_b64 exec, s[10:11]
	v_add_u32_e32 v0, s34, v34
	v_lshrrev_b32_e32 v0, 5, v0
	v_or_b32_e32 v0, v0, v36
	v_lshl_or_b32 v0, v0, 8, v62
	buffer_load_dword v76, v0, s[36:39], 0 offen
	v_mul_lo_u32 v0, s12, v3
	v_or_b32_e32 v1, v51, v0
	v_mul_lo_u32 v3, s12, v5
	v_or_b32_e32 v5, v51, v3
	buffer_load_dwordx4 v[68:71], v1, s[28:31], 0 offen
	v_add_u32_e32 v0, v0, v53
	v_add_u32_e32 v1, v3, v53
	buffer_load_dwordx4 v[72:75], v5, s[28:31], 0 offen
	buffer_load_dwordx4 v[78:81], v0, s[28:31], 0 offen
	buffer_load_dwordx4 v[82:85], v1, s[28:31], 0 offen
	s_waitcnt vmcnt(7) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[10:13], v52
	ds_read_b128 v[18:21], v54
	v_add_u32_e32 v5, s8, v46
	s_mov_b32 s18, s42
	s_mov_b32 s19, s43
	s_waitcnt vmcnt(3) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[68:71], v[10:13], 0, v76, v2 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v63, s44, v47
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[72:75], v[10:13], 0, v76, v2 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[86:89], v55
	ds_read_b128 v[90:93], v56
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[68:71], v[18:21], 0, v76, v2 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[72:75], v[18:21], 0, v76, v2 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[18:21], v57
	ds_read_b128 v[94:97], v58
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[78:81], v[18:21], v[6:9], v76, v2 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[82:85], v[18:21], v[10:13], v76, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[98:101], v59
	ds_read_b128 v[102:105], v60
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[78:81], v[94:97], v[14:17], v76, v2 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[82:85], v[94:97], v[64:67], v76, v2 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 2
	buffer_load_dword v66, v5, s[16:19], 0 offen offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[68:71], v[86:89], 0, v76, v4 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[72:75], v[86:89], 0, v76, v4 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v65, v5, s[16:19], 0 offen offset:1088
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[68:71], v[90:93], 0, v76, v4 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[72:75], v[90:93], 0, v76, v4 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dword v64, v5, s[16:19], 0 offen offset:1152
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[78:81], v[98:101], v[0:3], v76, v4 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[82:85], v[98:101], v[6:9], v76, v4 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v34, v5, s[16:19], 0 offen offset:1216
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[78:81], v[102:105], v[68:71], v76, v4 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[82:85], v[102:105], v[72:75], v76, v4 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_gt_u32_e32 vcc, v37, v63
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_13
	ds_read_b32 v68, v49 offset:32768
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v67, 0xffffff, v68
	v_cmp_gt_u32_e32 vcc, s33, v67
	v_cmp_gt_u32_e64 s[2:3], s47, v68
	s_and_b64 s[2:3], s[2:3], vcc
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_13
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v28, v66, v28
	v_mul_f32_e32 v29, v66, v29
	v_mul_f32_e32 v30, v66, v30
	v_mul_f32_e32 v31, v66, v31
	v_cvt_pk_bf16_f32 v68, v28, v29
	v_mad_u64_u32 v[28:29], s[2:3], v67, s51, v[44:45]
	global_atomic_pk_add_bf16 v[28:29], v68, off
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_mul_f32_e32 v24, v66, v24
	v_mul_f32_e32 v25, v66, v25
	global_atomic_pk_add_bf16 v[28:29], v30, off offset:4
	v_mul_f32_e32 v26, v66, v26
	v_mul_f32_e32 v27, v66, v27
	v_cvt_pk_bf16_f32 v24, v24, v25
	global_atomic_pk_add_bf16 v[28:29], v24, off offset:32
	v_cvt_pk_bf16_f32 v24, v26, v27
	global_atomic_pk_add_bf16 v[28:29], v24, off offset:36
.LBB0_13:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v24, 16, v63
	v_cmp_gt_u32_e32 vcc, v37, v24
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_16
	ds_read_b32 v25, v49 offset:32832
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v24, 0xffffff, v25
	v_cmp_gt_u32_e32 vcc, s33, v24
	v_cmp_gt_u32_e64 s[2:3], s47, v25
	s_and_b64 s[2:3], s[2:3], vcc
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v20, v65, v20
	v_mul_f32_e32 v21, v65, v21
	v_mul_f32_e32 v22, v65, v22
	v_mul_f32_e32 v23, v65, v23
	v_cvt_pk_bf16_f32 v25, v20, v21
	v_mad_u64_u32 v[20:21], s[2:3], v24, s51, v[44:45]
	global_atomic_pk_add_bf16 v[20:21], v25, off
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_mul_f32_e32 v16, v65, v16
	v_mul_f32_e32 v17, v65, v17
	global_atomic_pk_add_bf16 v[20:21], v22, off offset:4
	v_mul_f32_e32 v18, v65, v18
	v_mul_f32_e32 v19, v65, v19
	v_cvt_pk_bf16_f32 v16, v16, v17
	global_atomic_pk_add_bf16 v[20:21], v16, off offset:32
	v_cvt_pk_bf16_f32 v16, v18, v19
	global_atomic_pk_add_bf16 v[20:21], v16, off offset:36
.LBB0_16:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v16, 32, v63
	v_cmp_gt_u32_e32 vcc, v37, v16
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_19
	ds_read_b32 v17, v49 offset:32896
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v16, 0xffffff, v17
	v_cmp_gt_u32_e32 vcc, s33, v16
	v_cmp_gt_u32_e64 s[2:3], s47, v17
	s_and_b64 s[2:3], s[2:3], vcc
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_19
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v12, v64, v12
	v_mul_f32_e32 v13, v64, v13
	v_mul_f32_e32 v14, v64, v14
	v_mul_f32_e32 v15, v64, v15
	v_cvt_pk_bf16_f32 v17, v12, v13
	v_mad_u64_u32 v[12:13], s[2:3], v16, s51, v[44:45]
	global_atomic_pk_add_bf16 v[12:13], v17, off
	v_cvt_pk_bf16_f32 v14, v14, v15
	v_mul_f32_e32 v8, v64, v8
	v_mul_f32_e32 v9, v64, v9
	global_atomic_pk_add_bf16 v[12:13], v14, off offset:4
	v_mul_f32_e32 v10, v64, v10
	v_mul_f32_e32 v11, v64, v11
	v_cvt_pk_bf16_f32 v8, v8, v9
	global_atomic_pk_add_bf16 v[12:13], v8, off offset:32
	v_cvt_pk_bf16_f32 v8, v10, v11
	global_atomic_pk_add_bf16 v[12:13], v8, off offset:36
.LBB0_19:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v8, 48, v63
	v_cmp_gt_u32_e32 vcc, v37, v8
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_1
	ds_read_b32 v9, v49 offset:32960
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v9
	v_cmp_gt_u32_e32 vcc, s33, v8
	v_cmp_gt_u32_e64 s[2:3], s47, v9
	s_and_b64 s[2:3], s[2:3], vcc
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_1
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v0, v34, v0
	v_mul_f32_e32 v1, v34, v1
	v_mul_f32_e32 v2, v34, v2
	v_mul_f32_e32 v3, v34, v3
	v_cvt_pk_bf16_f32 v9, v0, v1
	v_mad_u64_u32 v[0:1], s[2:3], v8, s51, v[44:45]
	global_atomic_pk_add_bf16 v[0:1], v9, off
	v_cvt_pk_bf16_f32 v2, v2, v3
	global_atomic_pk_add_bf16 v[0:1], v2, off offset:4
	v_mul_f32_e32 v2, v34, v4
	v_mul_f32_e32 v3, v34, v5
	v_mul_f32_e32 v4, v34, v6
	v_mul_f32_e32 v5, v34, v7
	v_cvt_pk_bf16_f32 v2, v2, v3
	global_atomic_pk_add_bf16 v[0:1], v2, off offset:32
	v_cvt_pk_bf16_f32 v2, v4, v5
	global_atomic_pk_add_bf16 v[0:1], v2, off offset:36
	s_branch .LBB0_1
.LBB0_22:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_gemm2_0
		.amdhsa_group_segment_fixed_size 33024
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 106
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 108
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
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
	.size	moe_gemm2_0, .Lfunc_end0-moe_gemm2_0

	.set moe_gemm2_0.num_vgpr, 106
	.set moe_gemm2_0.num_agpr, 0
	.set moe_gemm2_0.numbered_sgpr, 52
	.set moe_gemm2_0.num_named_barrier, 0
	.set moe_gemm2_0.private_seg_size, 0
	.set moe_gemm2_0.uses_vcc, 1
	.set moe_gemm2_0.uses_flat_scratch, 0
	.set moe_gemm2_0.has_dyn_sized_stack, 0
	.set moe_gemm2_0.has_recursion, 0
	.set moe_gemm2_0.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
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
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
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
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 33024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name:           moe_gemm2_0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         moe_gemm2_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     106
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
