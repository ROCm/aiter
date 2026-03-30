	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_gemm2_small_0
	.p2align	8
	.type	moe_gemm2_small_0,@function
moe_gemm2_small_0:
	s_load_dwordx2 s[28:29], s[0:1], 0x40
	s_load_dwordx16 s[12:27], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x58
	s_mov_b32 s34, s3
	s_mov_b32 s31, 0x27000
	s_waitcnt lgkmcnt(0)
	s_and_b32 s29, s29, 0xffff
	s_mov_b32 s30, 4
	s_lshl_b32 s3, s3, 2
	buffer_load_dword v3, off, s[28:31], 0
	s_lshl_b32 s30, s9, 2
	s_and_b32 s29, s25, 0xffff
	s_mov_b32 s28, s24
	v_mov_b32_e32 v1, s3
	buffer_load_dword v1, v1, s[28:31], 0 offen
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[24:25], s[34:35], 5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s10, v1
	s_movk_i32 s3, 0x100
	v_cmp_ge_u32_e32 vcc, s24, v3
	v_cmp_lt_u32_e64 s[4:5], s3, v1
	s_or_b64 s[4:5], vcc, s[4:5]
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_33
	v_lshlrev_b32_e32 v1, 2, v0
	v_add_u32_e32 v5, 0x100, v1
	v_lshrrev_b32_e32 v2, 3, v0
	v_bfe_u32 v6, v5, 5, 5
	s_lshl_b32 s30, s9, 7
	v_or_b32_e32 v4, s24, v2
	v_or_b32_e32 v7, s24, v6
	s_and_b32 s5, s23, 0xffff
	s_mov_b32 s4, s22
	s_mov_b32 s6, s30
	s_mov_b32 s7, s31
	v_lshlrev_b32_e32 v4, 2, v4
	v_lshlrev_b32_e32 v7, 2, v7
	v_bitop3_b32 v8, v2, s24, 16 bitop3:0xde
	buffer_load_dword v4, v4, s[4:7], 0 offen
	v_lshlrev_b32_e32 v8, 2, v8
	buffer_load_dword v9, v7, s[4:7], 0 offen
	buffer_load_dword v10, v8, s[4:7], 0 offen
	v_add_u32_e32 v7, 0x300, v1
	v_bfe_u32 v8, v7, 5, 5
	v_or_b32_e32 v11, s24, v8
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_load_dword v11, v11, s[4:7], 0 offen
	s_load_dword s33, s[0:1], 0x50
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s4, s8, 5
	s_ashr_i32 s1, s8, 31
	s_mov_b32 s0, s8
	s_and_b32 s41, s15, 0xffff
	s_mul_i32 s15, s10, 0x1c00
	s_lshl_b64 s[10:11], s[2:3], 6
	s_mul_i32 s2, s9, s4
	s_lshr_b64 s[44:45], s[0:1], 3
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s1, s33, s8
	s_lshl_b32 s38, s2, 5
	s_mul_hi_i32 s0, s33, s8
	s_lshr_b32 s2, s1, 1
	s_mov_b32 s40, s14
	s_mov_b32 s14, 0x9000000
	s_lshl_b32 s0, s0, 31
	s_lshl2_add_u32 s1, s1, s2
	s_add_i32 s42, s1, s0
	s_lshl_b32 s5, s8, 3
	s_and_b32 s29, s27, 0xffff
	s_mul_i32 s22, s4, 0x1c1c00
	s_and_b32 s27, s5, 0xfffffc00
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s37, s19, 0xffff
	s_and_b32 s17, s17, 0xffff
	v_and_b32_e32 v1, 28, v1
	s_mov_b32 s43, s31
	v_bfe_u32 v12, v0, 4, 2
	v_and_b32_e32 v36, 15, v0
	v_lshlrev_b32_e32 v37, 4, v12
	s_mov_b32 s36, s18
	s_mov_b32 s39, s31
	v_lshlrev_b32_e32 v40, 8, v12
	v_lshlrev_b32_e32 v12, 4, v0
	s_mov_b32 s23, s31
	s_mov_b32 s18, -1
	s_mov_b32 s19, s31
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v13, 0xffffff, v4
	v_lshrrev_b32_e32 v14, 24, v4
	v_cmp_gt_u32_e32 vcc, s14, v4
	v_cmp_gt_u32_e64 s[0:1], s33, v13
	v_mad_u32_u24 v13, v4, 9, v14
	s_and_b64 vcc, vcc, s[0:1]
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v14, 0xffffff, v9
	v_cndmask_b32_e32 v13, 0, v13, vcc
	v_lshrrev_b32_e32 v15, 24, v9
	v_cmp_gt_u32_e32 vcc, s14, v9
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v16, 0xffffff, v10
	v_cmp_gt_u32_e64 s[4:5], s33, v14
	v_lshrrev_b32_e32 v17, 24, v10
	v_cmp_gt_u32_e64 s[0:1], s14, v10
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v18, 0xffffff, v11
	v_mad_u32_u24 v14, v9, 9, v15
	v_cmp_gt_u32_e64 s[6:7], s33, v16
	s_and_b64 vcc, vcc, s[4:5]
	v_cmp_gt_u32_e64 s[2:3], s14, v11
	v_mad_u32_u24 v15, v10, 9, v17
	v_cmp_gt_u32_e64 s[8:9], s33, v18
	v_cndmask_b32_e32 v14, 0, v14, vcc
	s_and_b64 vcc, s[0:1], s[6:7]
	v_cndmask_b32_e32 v15, 0, v15, vcc
	s_and_b64 vcc, s[2:3], s[8:9]
	s_add_u32 s0, s10, s15
	s_addc_u32 s1, s11, 0
	s_lshr_b64 s[2:3], s[0:1], 4
	s_lshr_b32 s1, s1, 4
	s_mul_i32 s7, s1, 0x94900247
	s_mul_hi_u32 s5, s2, 0x94900247
	s_mul_hi_u32 s6, s1, 0x94900247
	s_add_u32 s5, s7, s5
	s_mul_i32 s4, s2, 0x246ddb4
	s_addc_u32 s8, s6, 0
	s_mul_hi_u32 s3, s2, 0x246ddb4
	s_add_u32 s4, s4, s5
	s_addc_u32 s3, s3, 0
	s_add_u32 s3, s8, s3
	s_addc_u32 s5, 0, 0
	s_mul_hi_u32 s8, s1, 0x246ddb4
	s_mul_i32 s1, s1, 0x246ddb4
	s_add_u32 s4, s1, s3
	s_addc_u32 s5, s8, s5
	s_lshr_b64 s[4:5], s[4:5], 10
	s_mul_i32 s3, s4, 0x1c1c0
	s_sub_i32 s9, s2, s3
	s_or_b32 s3, s2, 1
	s_mul_hi_u32 s15, s3, 0x94900247
	s_add_u32 s15, s7, s15
	s_mul_i32 s5, s3, 0x246ddb4
	s_addc_u32 s28, s6, 0
	s_mul_hi_u32 s4, s3, 0x246ddb4
	s_add_u32 s5, s5, s15
	s_addc_u32 s4, s4, 0
	s_add_u32 s4, s28, s4
	s_addc_u32 s5, 0, 0
	s_add_u32 s4, s1, s4
	s_addc_u32 s5, s8, s5
	s_lshr_b64 s[4:5], s[4:5], 10
	s_mul_i32 s4, s4, 0x1c1c0
	s_sub_i32 s15, s3, s4
	s_or_b32 s3, s2, 2
	s_mul_hi_u32 s28, s3, 0x94900247
	s_add_u32 s28, s7, s28
	s_mul_i32 s5, s3, 0x246ddb4
	s_addc_u32 s35, s6, 0
	s_mul_hi_u32 s4, s3, 0x246ddb4
	s_add_u32 s5, s5, s28
	s_addc_u32 s4, s4, 0
	s_add_u32 s4, s35, s4
	s_addc_u32 s5, 0, 0
	s_add_u32 s4, s1, s4
	s_addc_u32 s5, s8, s5
	s_lshr_b64 s[4:5], s[4:5], 10
	s_mul_i32 s4, s4, 0x1c1c0
	s_or_b32 s5, s2, 3
	s_sub_i32 s4, s3, s4
	s_mul_hi_u32 s28, s5, 0x94900247
	s_add_u32 s7, s7, s28
	s_mul_i32 s3, s5, 0x246ddb4
	s_addc_u32 s6, s6, 0
	v_mul_lo_u32 v13, v13, s44
	s_mul_hi_u32 s2, s5, 0x246ddb4
	s_add_u32 s3, s3, s7
	v_mul_lo_u32 v14, v14, s44
	s_addc_u32 s2, s2, 0
	v_add_lshl_u32 v13, v13, v1, 2
	s_add_u32 s2, s6, s2
	v_add_lshl_u32 v14, v14, v1, 2
	buffer_load_dwordx4 v[22:25], v13, s[40:43], 0 offen
	buffer_load_dwordx4 v[26:29], v14, s[40:43], 0 offen
	v_lshrrev_b32_e32 v19, 24, v11
	s_addc_u32 s3, 0, 0
	v_mad_u32_u24 v16, v11, 9, v19
	s_add_u32 s2, s1, s2
	v_cndmask_b32_e32 v16, 0, v16, vcc
	v_mul_lo_u32 v15, v15, s44
	s_addc_u32 s3, s8, s3
	v_mul_lo_u32 v16, v16, s44
	s_lshr_b64 s[2:3], s[2:3], 10
	v_add_lshl_u32 v13, v15, v1, 2
	s_mul_i32 s1, s2, 0x1c1c0
	v_add_lshl_u32 v14, v16, v1, 2
	buffer_load_dwordx4 v[30:33], v13, s[40:43], 0 offen
	buffer_load_dwordx4 v[42:45], v14, s[40:43], 0 offen
	s_lshl_b32 s2, s34, 6
	v_or3_b32 v13, v37, s2, v36
	v_lshlrev_b32_e32 v13, 2, v13
	s_lshl_b32 s0, s0, 1
	buffer_load_dword v38, v13, s[36:39], 0 offen
	v_or3_b32 v13, s0, v37, v36
	v_or_b32_e32 v16, 64, v37
	s_movk_i32 s2, 0xf0
	v_lshlrev_b32_e32 v13, 2, v13
	v_or3_b32 v14, v16, s0, v36
	s_mul_i32 s0, s27, s9
	v_and_or_b32 v12, v12, s2, v40
	s_sub_i32 s1, s5, s1
	v_lshlrev_b32_e32 v14, 2, v14
	buffer_load_dword v20, v13, s[20:23], 0 offen
	buffer_load_dword v39, v14, s[20:23], 0 offen
	v_or_b32_e32 v13, s0, v12
	s_mul_i32 s2, s27, s15
	s_mul_i32 s3, s27, s4
	v_or_b32_e32 v14, s2, v12
	buffer_load_dwordx4 v[46:49], v13, s[16:19], 0 offen
	buffer_load_dwordx4 v[50:53], v14, s[16:19], 0 offen
	v_or_b32_e32 v13, s3, v12
	s_mul_i32 s1, s27, s1
	s_addk_i32 s0, 0x400
	v_or_b32_e32 v14, s1, v12
	buffer_load_dwordx4 v[54:57], v13, s[16:19], 0 offen
	buffer_load_dwordx4 v[58:61], v14, s[16:19], 0 offen
	v_or_b32_e32 v13, s0, v12
	s_addk_i32 s2, 0x400
	s_addk_i32 s3, 0x400
	v_or_b32_e32 v14, s2, v12
	buffer_load_dwordx4 v[62:65], v13, s[16:19], 0 offen
	buffer_load_dwordx4 v[66:69], v14, s[16:19], 0 offen
	v_or_b32_e32 v13, s3, v12
	s_addk_i32 s1, 0x400
	v_or_b32_e32 v12, s1, v12
	buffer_load_dwordx4 v[70:73], v13, s[16:19], 0 offen
	buffer_load_dwordx4 v[74:77], v12, s[16:19], 0 offen
	v_lshlrev_b32_e32 v12, 2, v2
	ds_write_b32 v12, v4 offset:16384
	v_lshlrev_b32_e32 v4, 2, v6
	ds_write_b32 v4, v9 offset:16384
	v_lshlrev_b32_e32 v9, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_lshlrev_b32_e32 v1, 2, v1
	v_and_b32_e32 v9, 0xf0, v9
	v_lshlrev_b32_e32 v2, 8, v2
	v_bitop3_b32 v2, v1, v2, v9 bitop3:0xde
	s_mov_b32 s28, s26
	s_waitcnt vmcnt(14)
	ds_write_b128 v2, v[22:25]
	v_lshrrev_b32_e32 v2, 1, v5
	v_and_b32_e32 v2, 0xf0, v2
	v_lshlrev_b32_e32 v5, 8, v6
	v_bitop3_b32 v2, v2, v5, v1 bitop3:0xde
	s_waitcnt vmcnt(13)
	ds_write_b128 v2, v[26:29]
	v_lshlrev_b32_e32 v2, 8, v4
	v_bitop3_b32 v2, v2, v1, v9 bitop3:0xf6
	v_lshlrev_b32_e32 v5, 8, v8
	v_lshlrev_b32_e32 v4, 2, v4
	ds_write_b32 v4, v10 offset:16384
	v_lshlrev_b32_e32 v4, 2, v8
	ds_write_b32 v4, v11 offset:16384
	s_waitcnt vmcnt(12)
	ds_write_b128 v2, v[30:33]
	v_lshrrev_b32_e32 v2, 1, v7
	v_and_b32_e32 v2, 0xf0, v2
	v_bitop3_b32 v1, v2, v5, v1 bitop3:0xde
	s_waitcnt vmcnt(11)
	ds_write_b128 v1, v[42:45]
	v_lshlrev_b32_e32 v1, 4, v36
	v_lshlrev_b32_e32 v2, 8, v36
	v_bitop3_b32 v24, v37, v2, v1 bitop3:0xde
	ds_read_b128 v[12:15], v24
	s_waitcnt vmcnt(7) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[12:15], v[46:49], 0, v38, v20 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[26:29], v24 offset:4096
	s_waitcnt vmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[12:15], v[50:53], 0, v38, v20 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[26:29], v[46:49], 0, v38, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bitop3_b32 v1, v16, v2, v1 bitop3:0xde
	ds_read_b128 v[30:33], v1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[26:29], v[50:53], 0, v38, v20 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[30:33], v[62:65], v[4:7], v38, v20 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[26:29], v1 offset:4096
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[30:33], v[66:69], v[8:11], v38, v20 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[26:29], v[62:65], v[12:15], v38, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[30:33], v24
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[26:29], v[66:69], v[16:19], v38, v20 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[30:33], v[54:57], 0, v38, v39 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[42:45], v24 offset:4096
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[30:33], v[58:61], 0, v38, v39 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[42:45], v[54:57], 0, v38, v39 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[46:49], v1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[42:45], v[58:61], 0, v38, v39 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[46:49], v[70:73], v[20:23], v38, v39 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[50:53], v1 offset:4096
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[46:49], v[74:77], v[24:27], v38, v39 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[50:53], v[70:73], v[28:31], v38, v39 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshl_or_b32 v1, s24, 2, v37
	s_barrier
	buffer_load_dword v2, v1, s[28:31], 0 offen
	v_or_b32_e32 v37, 4, v1
	buffer_load_dword v37, v37, s[28:31], 0 offen
	v_or_b32_e32 v41, 8, v1
	v_or_b32_e32 v42, 12, v1
	v_or_b32_e32 v43, 64, v1
	v_or_b32_e32 v44, 0x44, v1
	buffer_load_dword v41, v41, s[28:31], 0 offen
	v_or_b32_e32 v45, 0x48, v1
	buffer_load_dword v42, v42, s[28:31], 0 offen
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[50:53], v[74:77], v[32:35], v38, v39 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dword v43, v43, s[28:31], 0 offen
	v_or_b32_e32 v36, v40, v36
	buffer_load_dword v44, v44, s[28:31], 0 offen
	v_or_b32_e32 v1, 0x4c, v1
	buffer_load_dword v45, v45, s[28:31], 0 offen
	v_lshlrev_b32_e32 v36, 1, v36
	buffer_load_dword v1, v1, s[28:31], 0 offen
	s_waitcnt vmcnt(7)
	v_mul_f32_e32 v4, v4, v2
	v_mul_f32_e32 v8, v8, v2
	v_mul_f32_e32 v20, v20, v2
	v_mul_f32_e32 v2, v24, v2
	v_cvt_pk_bf16_f32 v4, v4, s0
	v_cvt_pk_bf16_f32 v2, v2, s0
	s_waitcnt vmcnt(6)
	v_mul_f32_e32 v5, v5, v37
	v_cvt_pk_bf16_f32 v8, v8, s0
	v_cvt_pk_bf16_f32 v20, v20, s0
	v_mul_f32_e32 v9, v9, v37
	v_mul_f32_e32 v21, v21, v37
	v_mul_f32_e32 v24, v25, v37
	s_waitcnt vmcnt(5)
	v_mul_f32_e32 v6, v6, v41
	v_mul_f32_e32 v10, v10, v41
	v_mul_f32_e32 v22, v22, v41
	v_mul_f32_e32 v25, v26, v41
	s_waitcnt vmcnt(4)
	v_mul_f32_e32 v7, v7, v42
	v_mul_f32_e32 v11, v11, v42
	v_mul_f32_e32 v23, v23, v42
	v_mul_f32_e32 v26, v27, v42
	s_waitcnt vmcnt(3)
	v_mul_f32_e32 v12, v12, v43
	v_mul_f32_e32 v16, v16, v43
	v_mul_f32_e32 v27, v28, v43
	v_mul_f32_e32 v28, v32, v43
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v13, v13, v44
	ds_write_b16 v36, v4
	ds_write_b16 v36, v8 offset:32
	ds_write_b16 v36, v20 offset:64
	ds_write_b16 v36, v2 offset:96
	v_cvt_pk_bf16_f32 v2, v5, s0
	v_cvt_pk_bf16_f32 v4, v9, s0
	v_cvt_pk_bf16_f32 v5, v21, s0
	v_cvt_pk_bf16_f32 v8, v24, s0
	v_cvt_pk_bf16_f32 v6, v6, s0
	v_cvt_pk_bf16_f32 v9, v10, s0
	v_cvt_pk_bf16_f32 v10, v22, s0
	v_cvt_pk_bf16_f32 v20, v25, s0
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v21, v23, s0
	v_cvt_pk_bf16_f32 v22, v26, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	v_cvt_pk_bf16_f32 v16, v16, s0
	v_cvt_pk_bf16_f32 v23, v27, s0
	v_cvt_pk_bf16_f32 v24, v28, s0
	v_cvt_pk_bf16_f32 v13, v13, s0
	ds_write_b16 v36, v2 offset:128
	ds_write_b16 v36, v4 offset:160
	ds_write_b16 v36, v5 offset:192
	ds_write_b16 v36, v8 offset:224
	ds_write_b16 v36, v6 offset:256
	ds_write_b16 v36, v9 offset:288
	ds_write_b16 v36, v10 offset:320
	ds_write_b16 v36, v20 offset:352
	ds_write_b16 v36, v7 offset:384
	ds_write_b16 v36, v11 offset:416
	ds_write_b16 v36, v21 offset:448
	ds_write_b16 v36, v22 offset:480
	ds_write_b16 v36, v12 offset:2048
	ds_write_b16 v36, v16 offset:2080
	ds_write_b16 v36, v23 offset:2112
	ds_write_b16 v36, v24 offset:2144
	ds_write_b16 v36, v13 offset:2176
	v_mul_f32_e32 v2, v17, v44
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2208
	v_mul_f32_e32 v2, v29, v44
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2240
	v_mul_f32_e32 v2, v33, v44
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2272
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v2, v14, v45
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2304
	v_mul_f32_e32 v2, v18, v45
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2336
	v_mul_f32_e32 v2, v30, v45
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2368
	v_mul_f32_e32 v2, v34, v45
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2400
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v2, v15, v1
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2432
	v_mul_f32_e32 v2, v19, v1
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2464
	v_mul_f32_e32 v2, v31, v1
	v_cvt_pk_bf16_f32 v2, v2, s0
	ds_write_b16 v36, v2 offset:2496
	v_lshrrev_b32_e32 v2, 5, v0
	v_mul_f32_e32 v1, v35, v1
	v_lshlrev_b32_e32 v4, 2, v2
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_add_u32_e32 v4, 0x4000, v4
	ds_write_b16 v36, v1 offset:2528
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[22:23], v4 offset1:2
	ds_read2_b32 v[18:19], v4 offset0:4 offset1:6
	ds_read2_b32 v[16:17], v4 offset0:8 offset1:10
	ds_read2_b32 v[14:15], v4 offset0:12 offset1:14
	ds_read2_b32 v[12:13], v4 offset0:16 offset1:18
	ds_read2_b32 v[10:11], v4 offset0:20 offset1:22
	ds_read2_b32 v[8:9], v4 offset0:24 offset1:26
	ds_read2_b32 v[4:5], v4 offset0:28 offset1:30
	v_or_b32_e32 v1, s24, v2
	v_and_b32_e32 v0, 31, v0
	s_waitcnt lgkmcnt(7)
	v_and_b32_e32 v6, 0xffffff, v22
	v_cmp_gt_u32_e64 s[0:1], s33, v6
	v_cmp_gt_u32_e64 s[2:3], s14, v22
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_mov_b32_e32 v21, 0
	s_and_b64 s[2:3], vcc, s[0:1]
	v_and_b32_e32 v25, 0xffffff, v23
	v_lshlrev_b32_e32 v22, 2, v0
	v_lshl_or_b32 v0, v0, 1, s10
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_3
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v20, v1
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[26:27], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[6:7], s[2:3], v6, s2, v[26:27]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[6:7], v20, off
.LBB0_3:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v20, 2, v2
	v_lshl_add_u64 v[6:7], v[20:21], 0, s[24:25]
	v_cmp_gt_u32_e64 s[0:1], s33, v25
	v_cmp_gt_u32_e64 s[2:3], s14, v23
	v_cmp_gt_u32_e32 vcc, v3, v6
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(6)
	v_and_b32_e32 v24, 0xffffff, v18
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_5
	v_lshl_or_b32 v1, v20, 7, v22
	ds_read_b32 v7, v1
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[20:21], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[20:21], s[2:3], v25, s2, v[20:21]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[20:21], v7, off
.LBB0_5:
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s4, 0x9000000
	v_add_u32_e32 v1, 2, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v24
	v_cmp_gt_u32_e64 s[2:3], s4, v18
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v19
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_7
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v18, v1 offset:512
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[20:21], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[20:21], s[2:3], v24, s2, v[20:21]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[20:21], v18, off
.LBB0_7:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 4, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v19
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(5)
	v_and_b32_e32 v18, 0xffffff, v16
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_9
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v19, v1 offset:768
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[20:21], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[20:21], s[2:3], v7, s2, v[20:21]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[20:21], v19, off
.LBB0_9:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 6, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v18
	v_cmp_gt_u32_e64 s[2:3], s4, v16
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v17
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_11
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v16, v1 offset:1024
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[20:21], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[18:19], s[2:3], v18, s2, v[20:21]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[18:19], v16, off
.LBB0_11:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 8, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v17
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(4)
	v_and_b32_e32 v16, 0xffffff, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_13
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v17, v1 offset:1280
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[18:19], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[18:19], s[2:3], v7, s2, v[18:19]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[18:19], v17, off
.LBB0_13:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 10, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v16
	v_cmp_gt_u32_e64 s[2:3], s4, v14
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_15
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v14, v1 offset:1536
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[18:19], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[16:17], s[2:3], v16, s2, v[18:19]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[16:17], v14, off
.LBB0_15:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 12, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v15
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(3)
	v_and_b32_e32 v14, 0xffffff, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_17
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v15, v1 offset:1792
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[16:17], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[16:17], s[2:3], v7, s2, v[16:17]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[16:17], v15, off
.LBB0_17:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 14, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v14
	v_cmp_gt_u32_e64 s[2:3], s4, v12
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v13
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_19
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v12, v1 offset:2048
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[16:17], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[14:15], s[2:3], v14, s2, v[16:17]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[14:15], v12, off
.LBB0_19:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 16, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v13
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(2)
	v_and_b32_e32 v12, 0xffffff, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_21
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v13, v1 offset:2304
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[14:15], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[14:15], s[2:3], v7, s2, v[14:15]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[14:15], v13, off
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 18, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v12
	v_cmp_gt_u32_e64 s[2:3], s4, v10
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v11
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_23
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v10, v1 offset:2560
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[14:15], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[12:13], s[2:3], v12, s2, v[14:15]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[12:13], v10, off
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 20, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v11
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(1)
	v_and_b32_e32 v10, 0xffffff, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_25
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v11, v1 offset:2816
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[12:13], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[12:13], s[2:3], v7, s2, v[12:13]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[12:13], v11, off
.LBB0_25:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 22, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v10
	v_cmp_gt_u32_e64 s[2:3], s4, v8
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v9
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_27
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v8, v1 offset:3072
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[12:13], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[10:11], s[2:3], v10, s2, v[12:13]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[10:11], v8, off
.LBB0_27:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 24, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v9
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v4
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_29
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v9, v1 offset:3328
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[10:11], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[10:11], s[2:3], v7, s2, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[10:11], v9, off
.LBB0_29:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 26, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v8
	v_cmp_gt_u32_e64 s[2:3], s4, v4
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	v_and_b32_e32 v7, 0xffffff, v5
	s_and_b64 s[2:3], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_31
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v4, v1 offset:3584
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[10:11], v[0:1], 1, s[12:13]
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[8:9], s[2:3], v8, s2, v[10:11]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[8:9], v4, off
.LBB0_31:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v1, 28, v6
	v_cmp_gt_u32_e64 s[0:1], s33, v7
	v_cmp_gt_u32_e64 s[2:3], s4, v5
	v_cmp_gt_u32_e32 vcc, v3, v1
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_33
	v_lshl_or_b32 v1, v2, 7, v22
	ds_read_b32 v2, v1 offset:3840
	v_mov_b32_e32 v1, s11
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[12:13]
	s_movk_i32 s0, 0x3800
	v_mad_u64_u32 v[0:1], s[0:1], v7, s0, v[0:1]
	s_waitcnt lgkmcnt(0)
	global_atomic_pk_add_bf16 v[0:1], v2, off
.LBB0_33:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_gemm2_small_0
		.amdhsa_group_segment_fixed_size 16512
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
		.amdhsa_next_free_vgpr 78
		.amdhsa_next_free_sgpr 46
		.amdhsa_accum_offset 80
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
	.size	moe_gemm2_small_0, .Lfunc_end0-moe_gemm2_small_0

	.set moe_gemm2_small_0.num_vgpr, 78
	.set moe_gemm2_small_0.num_agpr, 0
	.set moe_gemm2_small_0.numbered_sgpr, 46
	.set moe_gemm2_small_0.num_named_barrier, 0
	.set moe_gemm2_small_0.private_seg_size, 0
	.set moe_gemm2_small_0.uses_vcc, 1
	.set moe_gemm2_small_0.uses_flat_scratch, 0
	.set moe_gemm2_small_0.has_dyn_sized_stack, 0
	.set moe_gemm2_small_0.has_recursion, 0
	.set moe_gemm2_small_0.has_indirect_call, 0
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
    .group_segment_fixed_size: 16512
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 256
    .name:           moe_gemm2_small_0
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .sgpr_spill_count: 0
    .symbol:         moe_gemm2_small_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     78
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
