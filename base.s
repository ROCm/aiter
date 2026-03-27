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
	buffer_load_dword v5, off, s[8:11], 0
	s_load_dwordx16 s[8:23], s[0:1], 0x0
	v_bfe_u32 v12, v0, 4, 2
	v_and_b32_e32 v2, 15, v0
	v_lshrrev_b32_e32 v6, 1, v0
	v_lshlrev_b32_e32 v10, 4, v12
	v_and_b32_e32 v11, 0x60, v6
	v_lshlrev_b32_e32 v21, 4, v2
	v_lshlrev_b32_e32 v24, 8, v2
	s_movk_i32 s5, 0xf0
	v_lshrrev_b32_e32 v16, 3, v0
	v_lshlrev_b32_e32 v13, 2, v0
	v_lshlrev_b32_e32 v8, 4, v0
	v_lshlrev_b32_e32 v17, 8, v12
	v_bitop3_b32 v19, v10, v24, v21 bitop3:0xde
	v_or_b32_e32 v10, 64, v10
	v_or_b32_e32 v27, v11, v2
	s_mov_b32 s4, s3
	s_ashr_i32 s3, s2, 31
	v_lshlrev_b32_e32 v14, 1, v0
	v_and_b32_e32 v4, 28, v13
	v_or_b32_e32 v18, 32, v16
	v_and_or_b32 v15, v8, s5, v17
	v_or_b32_e32 v17, v17, v21
	v_bitop3_b32 v32, v10, v24, v21 bitop3:0xde
	v_lshlrev_b32_e32 v21, 1, v27
	s_lshl_b64 s[2:3], s[2:3], 7
	v_and_b32_e32 v22, 0xf0, v14
	v_lshlrev_b32_e32 v23, 8, v16
	v_lshlrev_b32_e32 v25, 2, v4
	v_lshlrev_b32_e32 v26, 8, v18
	s_lshl_b32 s5, s6, 3
	s_mul_i32 s35, s33, s6
	v_lshl_or_b32 v33, v12, 10, v21
	v_and_b32_e32 v12, 62, v14
	v_and_b32_e32 v20, 63, v0
	v_or_b32_e32 v6, s2, v11
	v_bitop3_b32 v30, v25, v23, v22 bitop3:0xde
	v_bitop3_b32 v31, v26, v25, v22 bitop3:0xf6
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v10, s8
	s_ashr_i32 s45, s6, 31
	s_mov_b32 s44, s6
	s_mul_hi_i32 s26, s33, s6
	s_mov_b32 s8, s18
	s_and_b32 s18, s5, 0xfffffc00
	s_lshr_b32 s5, s35, 1
	v_lshrrev_b32_e32 v49, 5, v0
	v_or_b32_e32 v22, s2, v12
	s_lshl_b32 s2, s4, 10
	v_cmp_gt_u32_e64 s[0:1], 64, v0
	v_mov_b32_e32 v11, s9
	s_and_b32 s29, s13, 0xffff
	s_mov_b32 s28, s12
	s_ashr_i32 s6, s6, 5
	s_and_b32 s37, s17, 0xffff
	s_mov_b32 s36, s16
	s_lshl_b32 s42, s7, 8
	s_mov_b32 s40, s22
	s_lshl_b32 s22, s7, 2
	s_lshl_b32 s7, s26, 31
	s_lshr_b64 s[12:13], s[44:45], 3
	s_lshl2_add_u32 s5, s35, s5
	v_lshlrev_b32_e32 v21, 2, v49
	v_lshlrev_b32_e32 v14, 1, v12
	v_mov_b32_e32 v23, s3
	v_and_or_b32 v12, v0, 48, s2
	v_lshlrev_b32_e32 v0, 2, v20
	s_movk_i32 s16, 0xfc00
	s_mov_b32 s30, -1
	s_mov_b32 s34, 0
	v_mov_b32_e32 v3, 0x27000
	v_mov_b32_e32 v1, 0
	s_mov_b32 s31, s27
	s_mov_b32 s39, s27
	s_mov_b32 s43, s27
	v_mov_b32_e32 v7, s3
	v_mov_b32_e32 v9, s3
	v_or_b32_e32 v8, 16, v6
	v_or_b32_e32 v28, 0x400, v15
	v_or_b32_e32 v29, 0x400, v17
	s_and_b32 s25, s11, 0xffff
	s_mov_b32 s24, s10
	s_and_b32 s15, s15, 0xffff
	s_and_b32 s9, s19, 0xffff
	s_and_b32 s41, s23, 0xffff
	s_mov_b32 s23, s27
	s_and_b32 s21, s21, 0xffff
	s_mul_i32 s38, s6, 0x1c1c00
	s_add_i32 s26, s5, s7
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v2, v5, s6
	s_movk_i32 s13, 0x100
	v_or_b32_e32 v34, 0x100, v33
	v_or_b32_e32 v35, 0x200, v33
	v_or_b32_e32 v36, 0x300, v33
	v_or_b32_e32 v37, 0x1000, v33
	v_or_b32_e32 v38, 0x1100, v33
	v_or_b32_e32 v39, 0x1200, v33
	v_or_b32_e32 v40, 0x1300, v33
	v_or_b32_e32 v41, 0x2000, v33
	v_or_b32_e32 v42, 0x2100, v33
	v_or_b32_e32 v43, 0x2200, v33
	v_or_b32_e32 v44, 0x2300, v33
	v_or_b32_e32 v45, 0x3000, v33
	v_or_b32_e32 v46, 0x3100, v33
	v_or_b32_e32 v47, 0x3200, v33
	v_or_b32_e32 v48, 0x3300, v33
	v_lshl_or_b32 v50, v49, 8, v14
	v_lshl_add_u64 v[10:11], v[22:23], 1, v[10:11]
	s_lshl_b32 s19, s4, 4
	s_lshl_b32 s35, s4, 8
	v_lshl_or_b32 v14, v18, 2, s2
	v_lshl_or_b32 v16, v16, 2, s2
	v_or_b32_e32 v18, s2, v13
	v_lshl_or_b32 v51, s4, 11, v0
	s_mov_b32 s17, -1
	s_movk_i32 s44, 0x1c00
	s_mov_b32 s45, 0x9000000
	s_mov_b32 s46, 0x94900247
	s_mov_b32 s47, 0x246ddb4
	s_mov_b32 s48, 0x1c1c0
	s_movk_i32 s49, 0x3800
	v_lshlrev_b32_e32 v52, 2, v20
	v_add_u32_e32 v53, 0x8000, v21
	s_branch .LBB0_3
.LBB0_1:
	s_or_b64 exec, exec, s[2:3]
.LBB0_2:
	s_add_u32 s16, s16, 0x100
	s_addc_u32 s17, s17, 0
	s_add_i32 s19, s19, 4
	s_add_i32 s35, s35, 64
	s_cmp_lg_u64 s[16:17], 0
	v_add_u32_e32 v51, 0x200, v51
	s_barrier
	s_cbranch_scc0 .LBB0_26
.LBB0_3:
	v_mov_b32_e32 v0, s19
	buffer_load_dword v0, v0, s[20:23], 0 offen
	v_cmp_ge_u32_e32 vcc, s35, v5
	s_waitcnt vmcnt(0)
	v_cmp_lt_u32_e64 s[2:3], s13, v0
	s_or_b64 s[2:3], vcc, s[2:3]
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_2
	v_add_u32_e32 v20, s16, v16
	s_mov_b32 s10, s42
	s_mov_b32 s11, s43
	buffer_load_dword v66, v20, s[8:11], 0 offen offset:1024
	v_add_u32_e32 v20, s16, v14
	buffer_load_dword v67, v20, s[8:11], 0 offen offset:1024
	v_mul_lo_u32 v0, v0, s44
	v_lshl_add_u64 v[20:21], v[6:7], 0, v[0:1]
	v_lshl_add_u64 v[64:65], v[8:9], 0, v[0:1]
	v_alignbit_b32 v0, v21, v20, 4
	v_mov_b32_e32 v23, s34
	v_lshrrev_b32_e32 v21, 4, v21
	v_mul_hi_u32 v22, v0, s46
	v_mad_u64_u32 v[22:23], s[2:3], v21, s46, v[22:23]
	v_mov_b32_e32 v25, s34
	v_alignbit_b32 v64, v65, v64, 4
	v_mov_b32_e32 v24, v22
	v_mov_b32_e32 v57, s34
	v_lshrrev_b32_e32 v65, 4, v65
	v_mul_hi_u32 v56, v64, s46
	v_mov_b32_e32 v54, v23
	v_mad_u64_u32 v[22:23], s[2:3], v0, s47, v[24:25]
	v_mov_b32_e32 v27, s34
	v_mov_b32_e32 v55, s34
	v_mad_u64_u32 v[56:57], s[2:3], v65, s46, v[56:57]
	v_mov_b32_e32 v26, v23
	v_mov_b32_e32 v59, s34
	v_mov_b32_e32 v58, v56
	v_lshl_add_u64 v[22:23], v[54:55], 0, v[26:27]
	v_mad_u64_u32 v[24:25], s[2:3], v64, s47, v[58:59]
	v_mad_u64_u32 v[22:23], s[2:3], v21, s47, v[22:23]
	v_mov_b32_e32 v61, s34
	v_mov_b32_e32 v63, s34
	v_mov_b32_e32 v62, v57
	v_mov_b32_e32 v60, v25
	v_alignbit_b32 v21, v23, v22, 10
	v_lshl_add_u64 v[24:25], v[62:63], 0, v[60:61]
	v_mul_lo_u32 v21, v21, s48
	v_mad_u64_u32 v[24:25], s[2:3], v65, s47, v[24:25]
	v_sub_u32_e32 v0, v0, v21
	v_alignbit_b32 v22, v25, v24, 10
	v_mul_lo_u32 v22, v22, s48
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v21, 0xffffff, v66
	v_lshrrev_b32_e32 v23, 24, v66
	v_cmp_gt_u32_e32 vcc, s45, v66
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v24, 0xffffff, v67
	v_cmp_gt_u32_e64 s[4:5], s33, v21
	v_lshrrev_b32_e32 v25, 24, v67
	v_cmp_gt_u32_e64 s[2:3], s45, v67
	v_mad_u32_u24 v21, v66, 9, v23
	v_cmp_gt_u32_e64 s[6:7], s33, v24
	s_and_b64 vcc, vcc, s[4:5]
	v_mad_u32_u24 v23, v67, 9, v25
	v_cndmask_b32_e32 v21, 0, v21, vcc
	s_and_b64 vcc, s[2:3], s[6:7]
	v_cndmask_b32_e32 v23, 0, v23, vcc
	v_mul_lo_u32 v25, v21, s12
	v_mul_lo_u32 v24, v23, s12
	v_sub_u32_e32 v21, v64, v22
	v_mul_lo_u32 v0, s18, v0
	v_or_b32_e32 v22, v15, v0
	v_mul_lo_u32 v21, s18, v21
	v_add_u32_e32 v0, v0, v28
	v_or_b32_e32 v23, v17, v21
	buffer_load_dwordx4 v[66:69], v22, s[28:31], 0 offen
	buffer_load_dwordx4 v[62:65], v23, s[28:31], 0 offen
	v_add_u32_e32 v21, v21, v29
	buffer_load_dwordx4 v[58:61], v0, s[28:31], 0 offen
	buffer_load_dwordx4 v[54:57], v21, s[28:31], 0 offen
	v_mov_b32_e32 v22, s14
	v_mov_b32_e32 v23, s15
	s_mov_b64 s[10:11], exec
.LBB0_5:
	v_readfirstlane_b32 s4, v22
	v_readfirstlane_b32 s5, v23
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[22:23]
	s_nop 0
	v_cmp_eq_u64_e64 s[2:3], s[6:7], v[2:3]
	s_and_b64 s[2:3], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[2:3]
	buffer_load_dword v0, v51, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_5
	s_mov_b64 exec, s[10:11]
	s_mov_b64 s[10:11], exec
.LBB0_7:
	v_readfirstlane_b32 s4, v22
	v_readfirstlane_b32 s5, v23
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[22:23]
	s_nop 0
	v_cmp_eq_u64_e64 s[2:3], s[6:7], v[2:3]
	s_and_b64 s[2:3], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[2:3]
	buffer_load_dword v21, v51, s[4:7], 0 offen offset:256
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_7
	s_mov_b64 exec, s[10:11]
	v_add_lshl_u32 v23, v25, v4, 2
	v_lshl_or_b32 v22, v20, 3, v52
	v_add_lshl_u32 v24, v24, v4, 2
	buffer_load_dwordx4 v[70:73], v23, s[24:27], 0 offen
	buffer_load_dwordx4 v[74:77], v24, s[24:27], 0 offen
	buffer_load_dword v20, v22, s[36:39], 0 offen
	s_waitcnt vmcnt(2)
	ds_write_b128 v30, v[70:73]
	s_waitcnt vmcnt(1)
	ds_write_b128 v31, v[74:77]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_10
	v_add_u32_e32 v22, s16, v18
	s_mov_b32 s10, s42
	s_mov_b32 s11, s43
	buffer_load_dword v22, v22, s[8:11], 0 offen offset:1024
	s_waitcnt vmcnt(0)
	ds_write_b32 v13, v22 offset:32768
.LBB0_10:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v26, s16, v12
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[22:25], v26, s[40:43], 0 offen offset:1024
	buffer_load_dwordx4 v[70:73], v26, s[40:43], 0 offen offset:1088
	buffer_load_dwordx4 v[74:77], v26, s[40:43], 0 offen offset:1152
	buffer_load_dwordx4 v[78:81], v26, s[40:43], 0 offen offset:1216
	ds_read_b128 v[86:89], v19
	s_waitcnt vmcnt(4) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[86:89], v[66:69], 0, v0, v20 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[94:97], v19 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[86:89], v[62:65], 0, v0, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[94:97], v[66:69], 0, v0, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[98:101], v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[94:97], v[62:65], 0, v0, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[98:101], v[58:61], v[82:85], v0, v20 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[102:105], v32 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[98:101], v[54:57], v[86:89], v0, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[102:105], v[58:61], v[90:93], v0, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[106:109], v19 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[102:105], v[54:57], v[94:97], v0, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[106:109], v[66:69], 0, v21, v20 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[110:113], v19 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:109], v[62:65], 0, v21, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[110:113], v[66:69], 0, v21, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[106:109], v32 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[110:113], v[62:65], 0, v21, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[106:109], v[58:61], v[98:101], v21, v20 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[110:113], v32 offset:12288
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:109], v[54:57], v[102:105], v21, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[110:113], v[58:61], v[66:69], v21, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v0, v22, v82
	v_cvt_pk_bf16_f32 v0, v0, s0
	s_barrier
	ds_write_b16 v33, v0
	v_mul_f32_e32 v0, v22, v86
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v33, v0 offset:32
	v_mul_f32_e32 v0, v23, v83
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v34, v0
	v_mul_f32_e32 v0, v23, v87
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v34, v0 offset:32
	v_mul_f32_e32 v0, v24, v84
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v35, v0
	v_mul_f32_e32 v0, v24, v88
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v35, v0 offset:32
	v_mul_f32_e32 v0, v25, v85
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v36, v0
	v_mul_f32_e32 v0, v25, v89
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v36, v0 offset:32
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v0, v70, v90
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v37, v0
	v_mul_f32_e32 v0, v70, v94
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v37, v0 offset:32
	v_mul_f32_e32 v0, v71, v91
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v38, v0
	v_mul_f32_e32 v0, v71, v95
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v38, v0 offset:32
	v_mul_f32_e32 v0, v72, v92
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v39, v0
	v_mul_f32_e32 v0, v72, v96
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v39, v0 offset:32
	v_mul_f32_e32 v0, v73, v93
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v40, v0
	v_mul_f32_e32 v0, v73, v97
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v40, v0 offset:32
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v0, v74, v98
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v41, v0
	v_mul_f32_e32 v0, v74, v102
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v41, v0 offset:32
	v_mul_f32_e32 v0, v75, v99
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v42, v0
	v_mul_f32_e32 v0, v75, v103
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v42, v0 offset:32
	v_mul_f32_e32 v0, v76, v100
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v43, v0
	v_mul_f32_e32 v0, v76, v104
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v43, v0 offset:32
	v_mul_f32_e32 v0, v77, v101
	v_cvt_pk_bf16_f32 v0, v0, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[110:113], v[54:57], v[62:65], v21, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b16 v44, v0
	v_mul_f32_e32 v0, v77, v105
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v44, v0 offset:32
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v0, v78, v58
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v45, v0
	v_mul_f32_e32 v0, v78, v54
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v45, v0 offset:32
	v_mul_f32_e32 v0, v79, v59
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v46, v0
	v_mul_f32_e32 v0, v79, v55
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v46, v0 offset:32
	v_mul_f32_e32 v0, v80, v60
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v47, v0
	v_mul_f32_e32 v0, v80, v56
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v47, v0 offset:32
	v_mul_f32_e32 v0, v81, v61
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v48, v0
	v_mul_f32_e32 v0, v81, v57
	v_cvt_pk_bf16_f32 v0, v0, s0
	ds_write_b16 v48, v0 offset:32
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[26:27], v53 offset1:8
	ds_read2_b32 v[24:25], v53 offset0:16 offset1:24
	ds_read2_b32 v[22:23], v53 offset0:32 offset1:40
	ds_read2_b32 v[20:21], v53 offset0:48 offset1:56
	v_add_u32_e32 v0, s35, v49
	v_cmp_gt_u32_e32 vcc, v5, v0
	s_waitcnt lgkmcnt(3)
	v_and_b32_e32 v54, 0xffffff, v26
	v_cmp_gt_u32_e64 s[2:3], s33, v54
	v_cmp_gt_u32_e64 s[4:5], s45, v26
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_12
	ds_read_b32 v26, v50
	v_mad_u64_u32 v[54:55], s[4:5], v54, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[54:55], v26, off
	ds_read_b32 v26, v50 offset:128
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[54:55], v26, off offset:128
.LBB0_12:
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v26, 0xffffff, v27
	v_add_u32_e32 v54, 8, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v26
	v_cmp_gt_u32_e64 s[4:5], s45, v27
	v_cmp_gt_u32_e32 vcc, v5, v54
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_14
	ds_read_b32 v54, v50 offset:2048
	v_mad_u64_u32 v[26:27], s[4:5], v26, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[26:27], v54, off
	ds_read_b32 v54, v50 offset:2176
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[26:27], v54, off offset:128
.LBB0_14:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(2)
	v_and_b32_e32 v26, 0xffffff, v24
	v_add_u32_e32 v27, 16, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v26
	v_cmp_gt_u32_e64 s[4:5], s45, v24
	v_cmp_gt_u32_e32 vcc, v5, v27
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_16
	ds_read_b32 v24, v50 offset:4096
	v_mad_u64_u32 v[26:27], s[4:5], v26, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[26:27], v24, off
	ds_read_b32 v24, v50 offset:4224
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[26:27], v24, off offset:128
.LBB0_16:
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v24, 0xffffff, v25
	v_add_u32_e32 v26, 24, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v24
	v_cmp_gt_u32_e64 s[4:5], s45, v25
	v_cmp_gt_u32_e32 vcc, v5, v26
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_18
	ds_read_b32 v26, v50 offset:6144
	v_mad_u64_u32 v[24:25], s[4:5], v24, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[24:25], v26, off
	ds_read_b32 v26, v50 offset:6272
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[24:25], v26, off offset:128
.LBB0_18:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v24, 0xffffff, v22
	v_add_u32_e32 v25, 32, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v24
	v_cmp_gt_u32_e64 s[4:5], s45, v22
	v_cmp_gt_u32_e32 vcc, v5, v25
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_20
	ds_read_b32 v22, v50 offset:8192
	v_mad_u64_u32 v[24:25], s[4:5], v24, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[24:25], v22, off
	ds_read_b32 v22, v50 offset:8320
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[24:25], v22, off offset:128
.LBB0_20:
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v22, 0xffffff, v23
	v_add_u32_e32 v24, 40, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v22
	v_cmp_gt_u32_e64 s[4:5], s45, v23
	v_cmp_gt_u32_e32 vcc, v5, v24
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_22
	ds_read_b32 v24, v50 offset:10240
	v_mad_u64_u32 v[22:23], s[4:5], v22, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[22:23], v24, off
	ds_read_b32 v24, v50 offset:10368
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[22:23], v24, off offset:128
.LBB0_22:
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v22, 0xffffff, v20
	v_add_u32_e32 v23, 48, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v22
	v_cmp_gt_u32_e64 s[4:5], s45, v20
	v_cmp_gt_u32_e32 vcc, v5, v23
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_24
	ds_read_b32 v20, v50 offset:12288
	v_mad_u64_u32 v[22:23], s[4:5], v22, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[22:23], v20, off
	ds_read_b32 v20, v50 offset:12416
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[22:23], v20, off offset:128
.LBB0_24:
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v20, 0xffffff, v21
	v_add_u32_e32 v0, 56, v0
	v_cmp_gt_u32_e64 s[2:3], s33, v20
	v_cmp_gt_u32_e64 s[4:5], s45, v21
	v_cmp_gt_u32_e32 vcc, v5, v0
	s_and_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], vcc, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_1
	ds_read_b32 v0, v50 offset:14336
	v_mad_u64_u32 v[20:21], s[4:5], v20, s49, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[20:21], v0, off
	ds_read_b32 v0, v50 offset:14464
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[20:21], v0, off offset:128
	s_branch .LBB0_1
.LBB0_26:
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
		.amdhsa_next_free_vgpr 114
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 116
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

	.set moe_gemm2_0.num_vgpr, 114
	.set moe_gemm2_0.num_agpr, 0
	.set moe_gemm2_0.numbered_sgpr, 50
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
    .sgpr_count:     56
    .sgpr_spill_count: 0
    .symbol:         moe_gemm2_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     114
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
