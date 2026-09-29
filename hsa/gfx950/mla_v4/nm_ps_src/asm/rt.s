.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.amdhsa_code_object_version 5
.text
.globl _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE
.protected _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE
.p2align 8
.type _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE,@function
_ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE:
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s70, 0
	s_mov_b32 s73, 0x7060302
	s_mov_b32 s74, 0x5040100
	s_mov_b32 s78, 0
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx2 s[12:13], s[0:1], 0x10
	s_load_dwordx2 s[16:17], s[0:1], 0x20
	s_load_dwordx2 s[20:21], s[0:1], 0x30
	s_load_dwordx2 s[28:29], s[0:1], 0x40
	s_load_dwordx2 s[24:25], s[0:1], 0x50
	s_load_dword s79, s[0:1], 0x70
	s_load_dword s80, s[0:1], 0x80
	s_load_dword s81, s[0:1], 0x90
	s_load_dword s86, s[0:1], 0xc0
	s_load_dwordx2 s[36:37], s[0:1], 0xd0
	s_load_dwordx2 s[40:41], s[0:1], 0xe0
	s_load_dword s83, s[0:1], 0xf0
	s_load_dwordx2 s[44:45], s[0:1], 0x100
	s_load_dwordx2 s[48:49], s[0:1], 0x110
	s_load_dwordx2 s[52:53], s[0:1], 0x120
	s_load_dwordx2 s[32:33], s[0:1], 0x130
	s_load_dword s99, s[0:1], 0x140
	v_lshrrev_b32_e32 v1, 10, v0
	v_lshrrev_b32_e32 v2, 10, v1
	v_and_b32_e32 v2, 0x3ff, v2
	v_and_b32_e32 v1, 0x3ff, v1
	v_and_b32_e32 v0, 0x3ff, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_and_b32_e32 v0, 63, v0
	s_mov_b32 s2, s2
	s_mov_b32 s3, s3
	s_mov_b32 s4, s4
	v_readfirstlane_b32 s7, v3
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s75, s3, 4
	s_and_b32 s29, s29, 0xffff
	s_add_u32 s28, s75, s28
	s_addc_u32 s29, 0, s29
	s_load_dword s68, s[28:29], 0x0
	s_load_dword s69, s[28:29], 0x4
	s_mul_i32 s75, s3, 4
	s_and_b32 s41, s41, 0xffff
	s_add_u32 s40, s75, s40
	s_addc_u32 s41, 0, s41
	s_and_b32 s37, s37, 0xffff
	s_add_u32 s36, s75, s36
	s_addc_u32 s37, 0, s37
	s_load_dword s93, s[40:41], 0x0
	s_load_dword s92, s[40:41], 0x4
	s_load_dword s89, s[36:37], 0x0
	s_load_dword s90, s[36:37], 0x4
	s_waitcnt lgkmcnt(0)
	s_sub_u32 s98, s92, s93
	s_cmp_le_u32 s98, s4
	s_cbranch_scc1 label_8EF4
	s_mov_b32 s82, s98
	s_sub_u32 s91, s90, s89
	s_cmp_eq_u32 s91, 0
	s_cbranch_scc1 label_8EF4
	s_mul_i32 s72, s91, s80
	s_mul_i32 s75, 0x800, s72
	s_mul_i32 s76, 0x400, s72
	s_cmp_eq_u32 s82, 1
	s_cselect_b32 s95, s76, s75
	s_mul_i32 s76, 0x200, s72
	s_mul_i32 s77, 0x80, s72
	s_mul_i32 s75, 4, s72
	s_mov_b32 s10, s95
	s_mov_b32 s18, s76
	s_mov_b32 s46, s77
	s_mov_b32 s14, s75
	s_mov_b32 s22, -16
	s_mov_b32 s50, -16
	s_mov_b32 s26, -16
	s_mov_b32 s11, 0x20000
	s_mov_b32 s19, 0x20000
	s_mov_b32 s47, 0x20000
	s_mov_b32 s15, 0x20000
	s_mov_b32 s23, 0x20000
	s_mov_b32 s51, 0x20000
	s_mov_b32 s27, 0x20000
	s_and_b32 s9, s9, 0xffff
	s_and_b32 s17, s17, 0xffff
	s_and_b32 s45, s45, 0xffff
	s_and_b32 s13, s13, 0xffff
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s49, s49, 0xffff
	s_and_b32 s25, s25, 0xffff
	s_or_b32 s9, s9, 0x40000
	s_or_b32 s17, s17, 0x40000
	s_or_b32 s45, s45, 0x40000
	s_or_b32 s13, s13, 0x40000
	s_or_b32 s21, s21, 0x40000
	s_or_b32 s49, s49, 0x40000
	s_or_b32 s25, s25, 0x40000
	s_and_b32 s53, s53, 0xffff
	s_or_b32 s53, s53, 0x40000
	s_lshl_b32 s75, s80, 2
	s_mov_b32 s54, s75
	s_mov_b32 s55, 0x20000
	s_and_b32 s33, s33, 0xffff
	s_or_b32 s33, s33, 0x40000
	s_mov_b32 s34, -16
	s_mov_b32 s35, 0x20000
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s86, 0
	s_lshr_b32 s67, 32, s86
	s_sub_u32 s71, s69, s68
	s_cmp_eq_u32 s71, 0
	s_cbranch_scc1 label_8EF4
	s_mul_i32 s100, s4, s67
	s_cmp_eq_u32 s99, 1
	s_cbranch_scc0 label_030C
	s_sub_u32 s76, 5, s86
	s_sub_u32 s77, s67, 1
	s_add_u32 s77, s71, s77
	s_lshr_b32 s77, s77, s76
	s_cmp_lt_u32 s77, s82
	s_cselect_b32 s77, s77, s82
	s_cmp_eq_u32 s4, 0
	s_cbranch_scc0 label_030C
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc0 label_030C
	s_cmp_eq_u32 s7, 0
	s_cbranch_scc0 label_030C
	v_mov_b32_e32 v26, s77
	v_mov_b32_e32 v27, s3
	v_lshlrev_b32_e32 v27, 2, v27
	buffer_store_dword v26, v27, s[32:35], 0 offen
label_030C:
	s_cmp_le_u32 s71, s100
	s_cbranch_scc1 label_8DAC
	s_mov_b32 s87, 0
	s_sub_u32 s88, s71, s100
	s_mul_i32 s65, s82, s67
	s_mov_b32 s64, s88
	s_mov_b32 s66, 0
	s_mov_b32 s66, 0
	v_cvt_f32_u32_e32 v26, s65
	s_sub_i32 s75, 0, s65
	v_rcp_iflag_f32_e32 v26, v26
	s_nop 0
	v_mul_f32_e32 v26, 0x4f7ffffe, v26
	v_cvt_u32_f32_e32 v26, v26
	v_mul_lo_u32 v27, s75, v26
	v_mul_hi_u32 v27, v26, v27
	v_add_u32_e32 v26, v26, v27
	v_mul_hi_u32 v26, s64, v26
	v_mul_lo_u32 v27, v26, s65
	v_sub_u32_e32 v29, s64, v27
	v_add_u32_e32 v28, 1, v26
	v_cmp_le_u32_e32 vcc, s65, v29
	v_subrev_u32_e32 v27, s65, v29
	s_nop 0
	v_cndmask_b32_e32 v26, v26, v28, vcc
	v_cndmask_b32_e32 v29, v29, v27, vcc
	v_add_u32_e32 v27, 1, v26
	v_cmp_le_u32_e32 vcc, s65, v29
	s_nop 1
	v_cndmask_b32_e32 v29, v26, v27, vcc
	s_nop 3
	v_readfirstlane_b32 s66, v29
	s_nop 3
	s_mov_b32 s88, s66
	s_mul_i32 s75, s88, s65
	s_sub_u32 s75, s64, s75
	s_mov_b32 s76, 0
	s_cmp_lt_u32 s75, s67
	s_cselect_b32 s76, s76, 1
	s_add_u32 s88, s76, s88
	s_cmpk_eq_u32 s76, 0x1
	s_cselect_b32 s70, 0, s75
	s_mul_i32 s94, s67, 4
	s_mul_i32 s94, s94, s82
	s_mul_i32 s75, s71, 4
	s_mov_b32 s26, s75
	s_mul_i32 s75, s68, 4
	s_add_u32 s24, s75, s24
	s_addc_u32 s25, 0, s25
	v_lshlrev_b32_e32 v26, 4, v0
	v_and_b32_e32 v27, 32, v26
	v_lshlrev_b32_e32 v26, 3, v0
	v_and_or_b32 v27, v26, 64, v27
	v_and_or_b32 v27, v0, 4, v27
	s_mul_i32 s75, 8, s7
	s_mul_i32 s76, 4, s100
	s_add_u32 s75, s76, s75
	v_add_u32_e32 v38, s75, v27
	v_bfe_u32 v44, v0, 1, 3
	v_lshlrev_b32_e32 v44, 2, v44
	s_mul_i32 s75, 32, s7
	v_add_u32_e32 v44, s75, v44
	s_mul_i32 s76, 4, s100
	v_add_u32_e32 v44, s76, v44
	buffer_load_dword v39, v38, s[24:27], 0 offen
	buffer_load_dword v242, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	buffer_load_dword v40, v38, s[24:27], 0 offen
	buffer_load_dword v243, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_cmp_eq_u32 s80, 16
	s_cselect_b32 s69, 16, 64
	s_cmp_eq_u32 s80, 32
	s_cselect_b32 s69, 32, s69
	s_cmp_eq_u32 s91, 1
	s_cselect_b32 s69, s69, 64
	s_lshr_b32 s72, s69, 4
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s72, s72, 4
	s_cmp_gt_u32 s88, 0
	s_cselect_b32 s72, s72, 4
	s_mul_i32 s75, s89, s80
	s_mul_i32 s75, s75, 0x200
	s_add_u32 s16, s75, s16
	s_addc_u32 s17, 0, s17
	s_mov_b32 s75, 0x8000
	s_mul_i32 s75, s75, s2
	s_mov_b32 s76, 0x200
	s_mul_i32 s76, s7, s76
	s_add_u32 s75, s75, s76
	v_and_b32_e32 v1, 31, v0
	v_lshlrev_b32_e32 v1, 4, v1
	v_bfe_u32 v26, v0, 5, 1
	v_lshlrev_b32_e32 v26, 11, v26
	v_add_u32_e32 v1, v26, v1
	v_add_u32_e32 v1, s75, v1
	s_mul_i32 s75, s7, 0x820
	s_add_u32 s56, 0x2000, s75
	s_add_u32 s57, 0x2080, s56
	s_add_u32 s58, 0x2080, s57
	s_add_u32 s59, 0x2080, s58
	s_mov_b32 m0, s56
	v_add_u32_e32 v2, 0, v1
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	s_mov_b32 m0, s57
	v_add_u32_e32 v2, 0x2000, v1
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	s_mov_b32 s6, 0x3fb8aa3b
	v_mov_b32_e32 v27, s6
	v_mov_b32_e32 v26, s79
	v_mul_f32_e32 v26, s6, v26
	v_rcp_f32_e32 v27, v27
	v_mov_b32_e32 v45, 0xff7fffff
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v46, 0
	s_mov_b32 m0, s58
	v_add_u32_e32 v2, 0x4000, v1
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	s_mov_b32 m0, s59
	v_add_u32_e32 v2, 0x6000, v1
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	buffer_load_dwordx4 v2, s[16:19], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x1000, v2
	v_readfirstlane_b32 s5, v26
	v_readfirstlane_b32 s6, v27
	s_mul_i32 s75, s89, s80
	s_mul_i32 s75, s75, 0x80
	s_add_u32 s44, s75, s44
	s_addc_u32 s45, 0, s45
	s_mov_b32 s75, 0x2000
	s_mul_i32 s75, s75, s2
	v_bfe_u32 v26, v0, 1, 3
	v_lshlrev_b32_e32 v12, 7, v26
	v_lshrrev_b32_e32 v26, 4, v0
	v_lshl_add_u32 v12, v26, 5, v12
	v_and_b32_e32 v26, 1, v0
	v_lshl_add_u32 v12, v26, 4, v12
	v_add_u32_e32 v12, s75, v12
	s_add_u32 s60, 0x12800, 0
	s_add_u32 s61, 0x8c0, s60
	s_add_u32 s62, 0x8c0, s61
	s_add_u32 s63, 0x8c0, s62
	v_and_b32_e32 v26, 7, v0
	v_lshlrev_b32_e32 v26, 5, v26
	v_lshrrev_b32_e32 v29, 4, v0
	v_lshl_add_u32 v26, v29, 8, v26
	v_bfe_u32 v29, v0, 3, 1
	v_mul_i32_i24_e32 v29, 0x410, v29
	v_add_u32_e32 v16, v29, v26
	v_add_u32_e32 v16, 0x12800, v16
	s_mul_i32 s75, s7, 0x8c0
	s_add_u32 m0, 0x12800, s75
	s_mul_i32 s75, s7, 0x800
	v_add_u32_e32 v2, s75, v12
	buffer_load_dwordx4 v2, s[44:47], 0 offen lds
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v2, 0x400, v2
	buffer_load_dwordx4 v2, s[44:47], 0 offen lds
	v_bfe_u32 v26, v0, 3, 1
	v_and_b32_e32 v27, 3, v0
	v_lshl_or_b32 v26, v27, 1, v26
	v_mul_i32_i24_e32 v27, 0x410, v26
	v_bfe_u32 v26, v0, 2, 1
	v_mul_i32_i24_e32 v26, 0x200, v26
	v_add_u32_e32 v27, v26, v27
	v_bfe_u32 v26, v0, 4, 2
	v_lshl_add_u32 v10, v26, 5, v27
	v_lshrrev_b32_e32 v28, 4, v0
	v_mul_i32_i24_e32 v29, 32, v28
	v_sub_u32_e32 v34, v10, v29
	v_add_u32_e32 v34, 0x1c0, v34
	v_lshlrev_b32_e32 v29, 1, v28
	v_lshrrev_b32_e32 v28, 1, v28
	v_or_b32_e32 v28, v28, v29
	v_and_b32_e32 v28, 3, v28
	v_add_u32_e32 v30, v28, v34
	v_sub_u32_e32 v25, v30, v34
	v_lshlrev_b32_e32 v25, 3, v25
	v_lshlrev_b32_e32 v43, 2, v0
	v_lshlrev_b32_e32 v17, 2, v0
	s_mul_i32 s75, 0x100, s7
	v_add_u32_e32 v17, s75, v17
	v_and_b32_e32 v26, 31, v0
	v_lshlrev_b32_e32 v27, 3, v26
	v_bfe_u32 v26, v0, 4, 1
	v_lshl_add_u32 v27, v26, 8, v27
	v_lshrrev_b32_e32 v26, 5, v0
	v_lshl_add_u32 v18, v26, 11, v27
	v_xor_b32_e32 v19, 0x80, v18
	v_lshlrev_b32_e32 v26, 2, v0
	v_and_b32_e32 v26, 0x80, v26
	s_mul_i32 s75, 0x200, s7
	v_lshl_add_u32 v20, v0, 3, s75
	v_xor_b32_e32 v20, v20, v26
	s_waitcnt vmcnt(2)
	s_barrier
	s_cmp_ge_u32 s7, s72
	s_cbranch_scc1 label_09EC
	s_cmp_eq_i32 s7, 0
	s_cbranch_scc0 label_0854
	ds_read_b128 a[0:3], v10 offset:8192
	ds_read_b128 a[4:7], v10 offset:8208
	ds_read_b128 a[8:11], v10 offset:8320
	ds_read_b128 a[12:15], v10 offset:8336
	ds_read_b128 a[16:19], v10 offset:8448
	ds_read_b128 a[20:23], v10 offset:8464
	ds_read_b128 a[24:27], v10 offset:8576
	ds_read_b128 a[28:31], v10 offset:8592
	ds_read_b128 v[26:29], v34 offset:8192
	s_waitcnt lgkmcnt(0)
	v_bfe_u32 v224, v26, v25, 8
	v_bfe_u32 v225, v27, v25, 8
	v_bfe_u32 v226, v28, v25, 8
	v_bfe_u32 v227, v29, v25, 8
	v_cmp_ge_u32_e64 vcc, v25, 16
	v_cndmask_b32_e64 v227, v227, 0, vcc
	s_branch label_09EC
label_0854:
	s_cmp_eq_i32 s7, 1
	s_cbranch_scc0 label_08DC
	ds_read_b128 a[0:3], v10 offset:16640
	ds_read_b128 a[4:7], v10 offset:16656
	ds_read_b128 a[8:11], v10 offset:16512
	ds_read_b128 a[12:15], v10 offset:16528
	ds_read_b128 a[16:19], v10 offset:16768
	ds_read_b128 a[20:23], v10 offset:16784
	ds_read_b128 a[24:27], v10 offset:16896
	ds_read_b128 a[28:31], v10 offset:16912
	ds_read_b128 v[26:29], v34 offset:16512
	s_waitcnt lgkmcnt(0)
	v_bfe_u32 v224, v27, v25, 8
	v_bfe_u32 v225, v26, v25, 8
	v_bfe_u32 v226, v28, v25, 8
	v_bfe_u32 v227, v29, v25, 8
	v_cmp_ge_u32_e64 vcc, v25, 16
	v_cndmask_b32_e64 v227, v227, 0, vcc
	s_branch label_09EC
label_08DC:
	s_cmp_eq_i32 s7, 2
	s_cbranch_scc0 label_0964
	ds_read_b128 a[0:3], v10 offset:24832
	ds_read_b128 a[4:7], v10 offset:24848
	ds_read_b128 a[8:11], v10 offset:24960
	ds_read_b128 a[12:15], v10 offset:24976
	ds_read_b128 a[16:19], v10 offset:25088
	ds_read_b128 a[20:23], v10 offset:25104
	ds_read_b128 a[24:27], v10 offset:25216
	ds_read_b128 a[28:31], v10 offset:25232
	ds_read_b128 v[26:29], v34 offset:24832
	s_waitcnt lgkmcnt(0)
	v_bfe_u32 v224, v26, v25, 8
	v_bfe_u32 v225, v27, v25, 8
	v_bfe_u32 v226, v28, v25, 8
	v_bfe_u32 v227, v29, v25, 8
	v_cmp_ge_u32_e64 vcc, v25, 16
	v_cndmask_b32_e64 v227, v227, 0, vcc
	s_branch label_09EC
label_0964:
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc0 label_09EC
	ds_read_b128 a[0:3], v10 offset:33152
	ds_read_b128 a[4:7], v10 offset:33168
	ds_read_b128 a[8:11], v10 offset:33280
	ds_read_b128 a[12:15], v10 offset:33296
	ds_read_b128 a[16:19], v10 offset:33408
	ds_read_b128 a[20:23], v10 offset:33424
	ds_read_b128 a[24:27], v10 offset:33536
	ds_read_b128 a[28:31], v10 offset:33552
	ds_read_b128 v[26:29], v34 offset:33152
	s_waitcnt lgkmcnt(0)
	v_bfe_u32 v224, v26, v25, 8
	v_bfe_u32 v225, v27, v25, 8
	v_bfe_u32 v226, v28, v25, 8
	v_bfe_u32 v227, v29, v25, 8
	v_cmp_ge_u32_e64 vcc, v25, 16
	v_cndmask_b32_e64 v227, v227, 0, vcc
	s_branch label_09EC
label_09EC:
	s_waitcnt vmcnt(0)
	s_barrier
	s_cmp_ge_u32 s7, s72
	s_cbranch_scc1 label_0A6C
	s_cmp_eq_i32 s7, 0
	s_cbranch_scc0 label_0A18
	ds_read_b128 a[32:35], v16
	ds_read_b128 a[36:39], v16 offset:16
	s_branch label_0A6C
label_0A18:
	s_cmp_eq_i32 s7, 1
	s_cbranch_scc0 label_0A34
	ds_read_b128 a[32:35], v16 offset:2240
	ds_read_b128 a[36:39], v16 offset:2256
	s_branch label_0A6C
label_0A34:
	s_cmp_eq_i32 s7, 2
	s_cbranch_scc0 label_0A50
	ds_read_b128 a[32:35], v16 offset:4480
	ds_read_b128 a[36:39], v16 offset:4496
	s_branch label_0A6C
label_0A50:
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc0 label_0A6C
	ds_read_b128 a[32:35], v16 offset:6720
	ds_read_b128 a[36:39], v16 offset:6736
	s_branch label_0A6C
label_0A6C:
	v_bfe_u32 v26, v0, 2, 2
	v_lshlrev_b32_e32 v1, 7, v26
	v_lshrrev_b32_e32 v26, 4, v0
	v_lshl_add_u32 v1, v26, 5, v1
	v_and_b32_e32 v26, 1, v0
	v_lshl_add_u32 v1, v26, 4, v1
	s_mov_b32 s92, 0xcccccccc
	s_mov_b32 s93, 0xcccccccc
	s_mov_b32 s74, 0xff7fffff
	s_mul_i32 s75, s7, 0x1040
	s_add_u32 s56, 0x2000, s75
	s_add_u32 s57, 0x4100, s56
	s_add_u32 s58, 0x4100, s57
	s_add_u32 s59, 0x4100, s58
	s_movk_i32 s73, 0x200
	s_and_b32 s75, s21, 0xffff
	v_add_co_u32_e32 v14, vcc, s20, v1
	v_mov_b32_e32 v15, s75
	v_addc_co_u32_e64 v15, vcc, v15, 0, vcc
	v_max_i32_e32 v39, 0, v39
	s_nop 1
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	buffer_load_dword v41, v38, s[24:27], 0 offen
	buffer_load_dword v244, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_ge_u32 s7, s72
	s_cbranch_scc1 label_0C34
	v_lshrrev_b32_e32 v32, 4, v0
	v_cmp_ge_u32_e64 vcc, v32, 2
	s_mov_b64 exec, vcc
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	s_mov_b64 exec, -1
label_0C34:
	v_and_b32_e32 v26, 48, v0
	v_lshlrev_b32_e32 v12, 1, v26
	v_and_b32_e32 v26, 1, v0
	v_lshl_or_b32 v12, v26, 4, v12
	s_mul_i32 s75, s7, 0x410
	s_add_u32 s60, 0x12800, s75
	s_add_u32 s61, 0x1040, s60
	s_add_u32 s62, 0x1040, s61
	s_add_u32 s63, 0x1040, s62
	s_movk_i32 s86, 0x80
	s_and_b32 s75, s49, 0xffff
	v_add_co_u32_e32 v246, vcc, s48, v12
	v_mov_b32_e32 v247, s75
	v_addc_co_u32_e64 v247, vcc, v247, 0, vcc
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_lshrrev_b32_e32 v26, 4, v0
	v_lshlrev_b32_e32 v27, 8, v26
	v_and_b32_e32 v26, 7, v0
	v_lshlrev_b32_e32 v28, 5, v26
	v_add_u32_e32 v27, v28, v27
	v_bfe_u32 v26, v0, 3, 1
	v_mul_i32_i24_e32 v28, 0x410, v26
	v_add_u32_e32 v27, v28, v27
	v_add_u32_e32 v16, 0x12800, v27
	v_and_b32_e32 v26, 6, v0
	v_lshlrev_b32_e32 v26, 1, v26
	v_and_or_b32 v26, v0, 1, v26
	v_mul_i32_i24_e32 v27, 0x410, v26
	v_bfe_u32 v26, v0, 3, 1
	v_mul_i32_i24_e32 v26, 32, v26
	v_add_u32_e32 v27, v26, v27
	v_bfe_u32 v26, v0, 4, 2
	v_lshl_add_u32 v13, v26, 8, v27
	v_max_i32_e32 v40, 0, v40
	s_nop 1
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_lshrrev_b32_e32 v28, 4, v0
	v_mul_i32_i24_e32 v29, 0x100, v28
	v_sub_u32_e32 v1, v13, v29
	v_add_u32_e32 v1, 0x2c0, v1
	v_lshlrev_b32_e32 v29, 1, v28
	v_lshrrev_b32_e32 v28, 1, v28
	v_or_b32_e32 v28, v28, v29
	v_and_b32_e32 v28, 3, v28
	v_add_u32_e32 v30, v28, v1
	v_sub_u32_e32 v25, v30, v1
	v_lshlrev_b32_e32 v25, 3, v25
	s_mul_i32 s75, s7, 64
	s_lshl_b32 s76, s7, 2
	v_lshrrev_b32_e32 v32, 4, v0
	s_cmp_eq_i32 s7, 3
	s_cselect_b64 vcc, -1, 0
	v_and_b32_e32 v33, 32, v0
	v_lshlrev_b32_e32 v33, 4, v33
	v_sub_u32_e32 v241, v13, v33
	v_cndmask_b32_e32 v33, v13, v241, vcc
	v_add_u32_e32 v10, s75, v33
	v_add3_u32 v238, s76, v1, v32
	s_xor_b32 s78, s75, 64
	s_cmp_lt_u32 s7, 2
	s_cselect_b32 s76, s75, 0
	s_cselect_b32 s78, s78, 64
	v_add_u32_e32 v239, s76, v13
	v_add_u32_e32 v240, s78, v13
	s_cmp_eq_i32 s7, 1
	s_cselect_b32 s78, 4, 0
	s_or_b32 s78, s78, 0xc0c0c00
	v_lshrrev_b32_e32 v12, 3, v25
	v_or_b32_e32 v12, s78, v12
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_and_b32_e32 v26, 15, v0
	v_mul_i32_i24_e32 v26, 0x390, v26
	v_lshrrev_b32_e32 v27, 4, v0
	v_lshl_add_u32 v26, v27, 6, v26
	s_mul_i32 s75, 0x100, s7
	s_add_u32 s75, 0x17000, s75
	v_add_u32_e32 v11, s75, v26
	v_cmp_ge_u32_e64 vcc, v27, 2
	s_cmp_lg_u32 s7, 3
	s_nop 4
	s_cselect_b64 vcc, 0, vcc
	v_and_b32_e32 v28, 31, v0
	v_lshlrev_b32_e32 v28, 4, v28
	v_add_u32_e32 v28, 0x1e200, v28
	v_cndmask_b32_e32 v11, v11, v28, vcc
	v_bfe_u32 v26, v0, 2, 2
	v_bfe_u32 v27, v0, 4, 2
	v_and_b32_e32 v28, 3, v0
	v_lshlrev_b32_e32 v28, 3, v28
	v_lshl_add_u32 v29, v27, 3, v26
	v_mul_i32_i24_e32 v29, 0x390, v29
	v_add_u32_e32 v29, v28, v29
	s_mul_i32 s75, 0x100, s7
	s_add_u32 s75, 0x17000, s75
	v_add_u32_e32 v21, s75, v29
	v_add_u32_e32 v22, 32, v21
	v_mul_i32_i24_e32 v29, 0x410, v27
	v_add_u32_e32 v29, v28, v29
	v_lshl_add_u32 v29, v26, 5, v29
	v_add_u32_e32 v23, 0x12800, v29
	v_add_u32_e32 v24, 0x100, v23
	buffer_load_dword v42, v38, s[24:27], 0 offen
	buffer_load_dword v245, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mov_b64_e32 v[96:97], 0
	v_mov_b64_e32 v[98:99], 0
	v_mov_b64_e32 v[100:101], 0
	v_mov_b64_e32 v[102:103], 0
	v_mov_b64_e32 v[104:105], 0
	v_mov_b64_e32 v[106:107], 0
	v_mov_b64_e32 v[108:109], 0
	v_mov_b64_e32 v[110:111], 0
	v_mov_b64_e32 v[112:113], 0
	v_mov_b64_e32 v[114:115], 0
	v_mov_b64_e32 v[116:117], 0
	v_mov_b64_e32 v[118:119], 0
	v_mov_b64_e32 v[120:121], 0
	v_mov_b64_e32 v[122:123], 0
	v_mov_b64_e32 v[124:125], 0
	v_mov_b64_e32 v[126:127], 0
	s_cmp_le_u32 s72, 1
	s_cbranch_scc1 label_1068
	v_mov_b64_e32 v[128:129], 0
	v_mov_b64_e32 v[130:131], 0
	v_mov_b64_e32 v[132:133], 0
	v_mov_b64_e32 v[134:135], 0
	v_mov_b64_e32 v[136:137], 0
	v_mov_b64_e32 v[138:139], 0
	v_mov_b64_e32 v[140:141], 0
	v_mov_b64_e32 v[142:143], 0
	v_mov_b64_e32 v[144:145], 0
	v_mov_b64_e32 v[146:147], 0
	v_mov_b64_e32 v[148:149], 0
	v_mov_b64_e32 v[150:151], 0
	v_mov_b64_e32 v[152:153], 0
	v_mov_b64_e32 v[154:155], 0
	v_mov_b64_e32 v[156:157], 0
	v_mov_b64_e32 v[158:159], 0
	s_cmp_le_u32 s72, 2
	s_cbranch_scc1 label_1068
	v_mov_b64_e32 v[160:161], 0
	v_mov_b64_e32 v[162:163], 0
	v_mov_b64_e32 v[164:165], 0
	v_mov_b64_e32 v[166:167], 0
	v_mov_b64_e32 v[168:169], 0
	v_mov_b64_e32 v[170:171], 0
	v_mov_b64_e32 v[172:173], 0
	v_mov_b64_e32 v[174:175], 0
	v_mov_b64_e32 v[176:177], 0
	v_mov_b64_e32 v[178:179], 0
	v_mov_b64_e32 v[180:181], 0
	v_mov_b64_e32 v[182:183], 0
	v_mov_b64_e32 v[184:185], 0
	v_mov_b64_e32 v[186:187], 0
	v_mov_b64_e32 v[188:189], 0
	v_mov_b64_e32 v[190:191], 0
	v_mov_b64_e32 v[192:193], 0
	v_mov_b64_e32 v[194:195], 0
	v_mov_b64_e32 v[196:197], 0
	v_mov_b64_e32 v[198:199], 0
	v_mov_b64_e32 v[200:201], 0
	v_mov_b64_e32 v[202:203], 0
	v_mov_b64_e32 v[204:205], 0
	v_mov_b64_e32 v[206:207], 0
	v_mov_b64_e32 v[208:209], 0
	v_mov_b64_e32 v[210:211], 0
	v_mov_b64_e32 v[212:213], 0
	v_mov_b64_e32 v[214:215], 0
	v_mov_b64_e32 v[216:217], 0
	v_mov_b64_e32 v[218:219], 0
	v_mov_b64_e32 v[220:221], 0
	v_mov_b64_e32 v[222:223], 0
label_1068:
	s_waitcnt vmcnt(6)
	s_barrier
	s_cmp_ge_u32 s7, s72
	s_cbranch_scc1 label_1098
	ds_read_b128 a[104:107], v16
	ds_read_b128 a[108:111], v16 offset:16
	ds_read_b128 a[112:115], v16 offset:2080
	ds_read_b128 a[116:119], v16 offset:2096
label_1098:
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	s_cmp_ge_u32 s7, s72
	s_cbranch_scc1 label_11FC
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_lshrrev_b32_e32 v32, 4, v0
	v_cmp_eq_u32_e32 vcc, 3, v32
	v_mov_b32_e32 v47, 0xff
	v_cndmask_b32_e64 v47, v47, 0, vcc
	ds_read_b128 a[40:43], v239 offset:8192
	ds_read_b128 a[44:47], v239 offset:8208
	ds_read_b128 a[72:75], v239 offset:10272
	ds_read_b128 a[76:79], v239 offset:10288
	ds_read_b128 a[48:51], v240 offset:8192
	ds_read_b128 a[52:55], v240 offset:8208
	ds_read_b128 a[80:83], v240 offset:10272
	ds_read_b128 a[84:87], v240 offset:10288
	ds_read_b128 a[56:59], v13 offset:8320
	ds_read_b128 a[60:63], v13 offset:8336
	ds_read_b128 a[88:91], v13 offset:10400
	ds_read_b128 a[92:95], v13 offset:10416
	ds_read_b128 a[64:67], v241 offset:8384
	ds_read_b128 a[68:71], v241 offset:8400
	ds_read_b128 a[96:99], v241 offset:10464
	ds_read_b128 a[100:103], v241 offset:10480
	ds_read_b128 v[228:231], v1 offset:8192
	ds_read_u8 v236, v238 offset:8192
	ds_read_b128 v[232:235], v1 offset:10272
	ds_read_u8 v237, v238 offset:10272
	s_branch label_120C
label_11FC:
	ds_read_u8 v236, v238 offset:8192
	ds_read_u8 v237, v238 offset:10272
label_120C:
	s_cmp_eq_u32 s88, 0
	s_cbranch_scc1 label_1F14
	s_cmp_ge_u32 s7, s72
	s_cbranch_scc1 label_12E14
	s_cmp_lt_u32 s7, 2
	s_cbranch_scc0 label_18A0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	buffer_load_dword v39, v38, s[24:27], 0 offen
	buffer_load_dword v242, v44, s[24:27], 0 offen
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	ds_write_b128 v11, v[34:37]
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	ds_write_b128 v11, v[34:37] offset:16
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_cvt_scalef32_pk_bf16_fp8 v34, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v85, v33 op_sel:[1,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_write_b128 v11, v[34:37] offset:32
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_bfe_u32 v234, v234, v25, 8
	v_cvt_scalef32_pk_bf16_fp8 v34, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v87, v33 op_sel:[1,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_write_b128 v11, v[34:37] offset:48
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_and_b32_e32 v235, v47, v235
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v34, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v89, v33 op_sel:[1,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_write_b128 v11, v[34:37] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v34, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_write_b128 v11, v[34:37] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v34, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	ds_write_b128 v11, v[34:37] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v34, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v95, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[34:37] offset:14640
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[228:231], v1 offset:24832
	ds_read_u8 v236, v238 offset:24832
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	ds_read_b128 v[232:235], v1 offset:26912
	ds_read_u8 v237, v238 offset:26912
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	ds_read_b128 a[40:43], v239 offset:24832
	ds_read_b128 a[44:47], v239 offset:24848
	v_exp_f32_e32 v33, v33
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	ds_read_b128 a[72:75], v239 offset:26912
	ds_read_b128 a[76:79], v239 offset:26928
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	ds_read_b128 a[56:59], v13 offset:24960
	ds_read_b128 a[60:63], v13 offset:24976
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	ds_read_b128 a[88:91], v13 offset:27040
	ds_read_b128 a[92:95], v13 offset:27056
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	ds_read_b128 a[64:67], v241 offset:25024
	ds_read_b128 a[68:71], v241 offset:25040
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	ds_read_b128 a[96:99], v241 offset:27104
	ds_read_b128 a[100:103], v241 offset:27120
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	v_max_i32_e32 v42, 0, v42
	s_nop 1
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	buffer_load_dword v40, v38, s[24:27], 0 offen
	buffer_load_dword v243, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_branch label_1F1C
label_18A0:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	buffer_load_dword v39, v38, s[24:27], 0 offen
	buffer_load_dword v242, v44, s[24:27], 0 offen
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	ds_write_b128 v11, v[34:37]
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	ds_write_b128 v11, v[34:37] offset:16
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_cvt_scalef32_pk_bf16_fp8 v34, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v85, v33 op_sel:[1,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_write_b128 v11, v[34:37] offset:32
	v_bfe_u32 v234, v234, v25, 8
	v_cvt_scalef32_pk_bf16_fp8 v34, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v87, v33 op_sel:[1,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_write_b128 v11, v[34:37] offset:48
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_and_b32_e32 v235, v47, v235
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_cvt_scalef32_pk_bf16_fp8 v34, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v89, v33 op_sel:[1,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_write_b128 v11, v[34:37] offset:14592
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_cvt_scalef32_pk_bf16_fp8 v34, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_write_b128 v11, v[34:37] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v34, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	ds_write_b128 v11, v[34:37] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v34, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v95, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[34:37] offset:14640
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[228:231], v1 offset:24832
	ds_read_u8 v236, v238 offset:24832
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	ds_read_b128 v[232:235], v1 offset:26912
	ds_read_u8 v237, v238 offset:26912
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	ds_read_b128 a[40:43], v239 offset:24832
	ds_read_b128 a[44:47], v239 offset:24848
	v_exp_f32_e32 v33, v33
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	ds_read_b128 a[72:75], v239 offset:26912
	ds_read_b128 a[76:79], v239 offset:26928
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	ds_read_b128 a[56:59], v13 offset:24960
	ds_read_b128 a[60:63], v13 offset:24976
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	ds_read_b128 a[88:91], v13 offset:27040
	ds_read_b128 a[92:95], v13 offset:27056
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	ds_read_b128 a[64:67], v241 offset:25024
	ds_read_b128 a[68:71], v241 offset:25040
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	ds_read_b128 a[96:99], v241 offset:27104
	ds_read_b128 a[100:103], v241 offset:27120
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	v_max_i32_e32 v42, 0, v42
	s_nop 1
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	buffer_load_dword v40, v38, s[24:27], 0 offen
	buffer_load_dword v243, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_branch label_1F1C
label_1F14:
	s_mov_b32 s72, s69
	s_branch label_8EFC
label_1F1C:
	s_mov_b32 s72, s69
	s_cmp_eq_u32 s88, 1
	s_cbranch_scc1 label_795C
	s_mov_b32 s87, 1
	s_cmp_eq_u32 s72, 64
	s_cbranch_scc1 label_1F44
	s_cmp_eq_u32 s72, 16
	s_cbranch_scc1 label_9FB8
	s_cmp_eq_u32 s72, 32
	s_cbranch_scc1 label_EDDC
label_1F44:
	s_cmp_eq_i32 s7, 2
	s_cbranch_scc1 label_3D74
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc1 label_5B68
label_1F54:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[54:55], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[56:57], v18 offset:1024
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[58:59], v19 offset:1024
	ds_read_b64 v[60:61], v18 offset:1536
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	ds_read_b64 v[62:63], v19 offset:1536
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_cbranch_vccz label_221C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_221C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:41472
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	ds_read_u8 v236, v238 offset:41472
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:43552
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	ds_read_b128 a[56:59], v13 offset:41600
	ds_read_b128 a[60:63], v13 offset:41616
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:43552
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	ds_read_b128 a[88:91], v13 offset:43680
	ds_read_b128 a[92:95], v13 offset:43696
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	ds_read_b128 a[64:67], v241 offset:41664
	ds_read_b128 a[68:71], v241 offset:41680
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	ds_read_b128 a[96:99], v241 offset:43744
	ds_read_b128 a[100:103], v241 offset:43760
	ds_write_b64 v20, v[64:65]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[70:71], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[72:73], v18 offset:1024
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[74:75], v19 offset:1024
	ds_read_b64 v[76:77], v18 offset:1536
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_read_b64 v[78:79], v19 offset:1536
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_cbranch_vccz label_29A0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_29A0:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:58112
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[72:75], v[160:163]
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	ds_read_u8 v236, v238 offset:58112
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[72:75], v[164:167]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[72:75], v[168:171]
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:60192
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[72:75], v[172:175]
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[72:75], v[176:179]
	ds_read_b128 a[56:59], v13 offset:58240
	ds_read_b128 a[60:63], v13 offset:58256
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:60192
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[72:75], v[180:183]
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[72:75], v[184:187]
	ds_read_b128 a[88:91], v13 offset:60320
	ds_read_b128 a[92:95], v13 offset:60336
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[72:75], v[188:191]
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[76:79], v[192:195]
	ds_read_b128 a[64:67], v241 offset:58304
	ds_read_b128 a[68:71], v241 offset:58320
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[76:79], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[76:79], v[200:203]
	ds_read_b128 a[96:99], v241 offset:60384
	ds_read_b128 a[100:103], v241 offset:60400
	ds_write_b64 v20, v[48:49]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[76:79], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[76:79], v[208:211]
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[76:79], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[76:79], v[216:219]
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[76:79], v[220:223]
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v39, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v242, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[54:55], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[56:57], v18 offset:1024
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[58:59], v19 offset:1024
	ds_read_b64 v[60:61], v18 offset:1536
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	ds_read_b64 v[62:63], v19 offset:1536
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_cbranch_vccz label_3124
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_3124:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:8192
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	ds_read_b128 a[48:51], v240 offset:8192
	ds_read_b128 a[52:55], v240 offset:8208
	ds_read_u8 v236, v238 offset:8192
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	ds_read_b128 a[80:83], v240 offset:10272
	ds_read_b128 a[84:87], v240 offset:10288
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:10272
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	ds_read_b128 a[56:59], v13 offset:8320
	ds_read_b128 a[60:63], v13 offset:8336
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:10272
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	ds_read_b128 a[88:91], v13 offset:10400
	ds_read_b128 a[92:95], v13 offset:10416
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	ds_read_b128 a[64:67], v241 offset:8384
	ds_read_b128 a[68:71], v241 offset:8400
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	ds_read_b128 a[96:99], v241 offset:10464
	ds_read_b128 a[100:103], v241 offset:10480
	ds_write_b64 v20, v[64:65]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	ds_read_b128 a[104:107], v16
	ds_read_b128 a[108:111], v16 offset:16
	ds_write_b64 v20, v[66:67] offset:2048
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	ds_read_b128 a[112:115], v16 offset:2080
	ds_read_b128 a[116:119], v16 offset:2096
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v40, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v243, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[70:71], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[72:73], v18 offset:1024
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[74:75], v19 offset:1024
	ds_read_b64 v[76:77], v18 offset:1536
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_read_b64 v[78:79], v19 offset:1536
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_cbranch_vccz label_38A8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_38A8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:24832
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[72:75], v[160:163]
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	ds_read_u8 v236, v238 offset:24832
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[72:75], v[164:167]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[72:75], v[168:171]
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:26912
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[72:75], v[172:175]
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[72:75], v[176:179]
	ds_read_b128 a[56:59], v13 offset:24960
	ds_read_b128 a[60:63], v13 offset:24976
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:26912
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[72:75], v[180:183]
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[72:75], v[184:187]
	ds_read_b128 a[88:91], v13 offset:27040
	ds_read_b128 a[92:95], v13 offset:27056
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[72:75], v[188:191]
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[76:79], v[192:195]
	ds_read_b128 a[64:67], v241 offset:25024
	ds_read_b128 a[68:71], v241 offset:25040
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[76:79], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[76:79], v[200:203]
	ds_read_b128 a[96:99], v241 offset:27104
	ds_read_b128 a[100:103], v241 offset:27120
	ds_write_b64 v20, v[48:49]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[76:79], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[76:79], v[208:211]
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	ds_write_b64 v20, v[50:51] offset:2048
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[76:79], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[76:79], v[216:219]
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[76:79], v[220:223]
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_1F54
label_3D68:
	s_nop 0
	s_nop 0
	s_branch label_795C
label_3D74:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[60:61], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	ds_read_b64 v[62:63], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_4034
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_4034:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:41472
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	ds_read_b128 a[40:43], v239 offset:41472
	ds_read_b128 a[44:47], v239 offset:41488
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:41472
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	ds_read_b128 a[72:75], v239 offset:43552
	ds_read_b128 a[76:79], v239 offset:43568
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	ds_read_b128 v[232:235], v1 offset:43552
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:43552
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	ds_read_b128 a[64:67], v241 offset:41664
	ds_read_b128 a[68:71], v241 offset:41680
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	ds_read_b128 a[96:99], v241 offset:43744
	ds_read_b128 a[100:103], v241 offset:43760
	ds_write_b64 v20, v[64:65]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[72:73], v18 offset:1024
	ds_read_b64 v[74:75], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[76:77], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_read_b64 v[78:79], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_47B0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_47B0:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:58112
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[72:75], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[72:75], v[164:167]
	ds_read_b128 a[40:43], v239 offset:58112
	ds_read_b128 a[44:47], v239 offset:58128
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:58112
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[72:75], v[168:171]
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[72:75], v[172:175]
	ds_read_b128 a[72:75], v239 offset:60192
	ds_read_b128 a[76:79], v239 offset:60208
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	ds_read_b128 v[232:235], v1 offset:60192
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[72:75], v[176:179]
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[72:75], v[180:183]
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:60192
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[72:75], v[184:187]
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[72:75], v[188:191]
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[76:79], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[76:79], v[196:199]
	ds_read_b128 a[64:67], v241 offset:58304
	ds_read_b128 a[68:71], v241 offset:58320
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[76:79], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[76:79], v[204:207]
	ds_read_b128 a[96:99], v241 offset:60384
	ds_read_b128 a[100:103], v241 offset:60400
	ds_write_b64 v20, v[48:49]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[76:79], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[76:79], v[212:215]
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[76:79], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[76:79], v[220:223]
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v39, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v242, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[60:61], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	ds_read_b64 v[62:63], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_4F2C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_4F2C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:8192
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	ds_read_b128 a[40:43], v239 offset:8192
	ds_read_b128 a[44:47], v239 offset:8208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:8192
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	ds_read_b128 a[72:75], v239 offset:10272
	ds_read_b128 a[76:79], v239 offset:10288
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	ds_read_b128 v[232:235], v1 offset:10272
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	ds_read_b128 a[48:51], v240 offset:8192
	ds_read_b128 a[52:55], v240 offset:8208
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:10272
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	ds_read_b128 a[80:83], v240 offset:10272
	ds_read_b128 a[84:87], v240 offset:10288
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	ds_read_b128 a[64:67], v241 offset:8384
	ds_read_b128 a[68:71], v241 offset:8400
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	ds_read_b128 a[96:99], v241 offset:10464
	ds_read_b128 a[100:103], v241 offset:10480
	ds_write_b64 v20, v[64:65]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	ds_read_b128 a[104:107], v16
	ds_read_b128 a[108:111], v16 offset:16
	ds_write_b64 v20, v[66:67] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
	ds_read_b128 a[112:115], v16 offset:2080
	ds_read_b128 a[116:119], v16 offset:2096
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v40, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v243, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64 v[72:73], v18 offset:1024
	ds_read_b64 v[74:75], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	ds_read_b64 v[76:77], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_read_b64 v[78:79], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_56A8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_56A8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:24832
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[72:75], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[72:75], v[164:167]
	ds_read_b128 a[40:43], v239 offset:24832
	ds_read_b128 a[44:47], v239 offset:24848
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:24832
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[72:75], v[168:171]
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[72:75], v[172:175]
	ds_read_b128 a[72:75], v239 offset:26912
	ds_read_b128 a[76:79], v239 offset:26928
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	ds_read_b128 v[232:235], v1 offset:26912
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[72:75], v[176:179]
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[72:75], v[180:183]
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:26912
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[72:75], v[184:187]
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[72:75], v[188:191]
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[76:79], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[76:79], v[196:199]
	ds_read_b128 a[64:67], v241 offset:25024
	ds_read_b128 a[68:71], v241 offset:25040
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[76:79], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[76:79], v[204:207]
	ds_read_b128 a[96:99], v241 offset:27104
	ds_read_b128 a[100:103], v241 offset:27120
	ds_write_b64 v20, v[48:49]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[76:79], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[76:79], v[212:215]
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	ds_write_b64 v20, v[50:51] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[76:79], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[76:79], v[220:223]
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D74
label_5B68:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64 v[52:53], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
	ds_read_b64 v[60:61], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	ds_read_b64 v[62:63], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_5E28
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_5E28:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:41472
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	ds_read_b128 a[40:43], v239 offset:41472
	ds_read_b128 a[44:47], v239 offset:41488
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:41472
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	ds_read_b128 a[72:75], v239 offset:43552
	ds_read_b128 a[76:79], v239 offset:43568
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	ds_read_b128 v[232:235], v1 offset:43552
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:43552
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	ds_read_b128 a[56:59], v13 offset:41600
	ds_read_b128 a[60:63], v13 offset:41616
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	ds_read_b128 a[88:91], v13 offset:43680
	ds_read_b128 a[92:95], v13 offset:43696
	ds_write_b64 v20, v[64:65]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:4160
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:4288
	ds_read_b64 v[68:69], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v24 offset:4160
	ds_read_b64_tr_b16 a[142:143], v24 offset:4288
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v23 offset:4672
	ds_read_b64_tr_b16 a[146:147], v23 offset:4800
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v24 offset:4672
	ds_read_b64 v[72:73], v18 offset:1024
	ds_read_b64 v[74:75], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v24 offset:4800
	ds_read_b64 v[76:77], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_read_b64 v[78:79], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_65A4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_65A4:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:58112
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[72:75], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[72:75], v[164:167]
	ds_read_b128 a[40:43], v239 offset:58112
	ds_read_b128 a[44:47], v239 offset:58128
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:58112
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[72:75], v[168:171]
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[72:75], v[172:175]
	ds_read_b128 a[72:75], v239 offset:60192
	ds_read_b128 a[76:79], v239 offset:60208
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	ds_read_b128 v[232:235], v1 offset:60192
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[72:75], v[176:179]
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[72:75], v[180:183]
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:60192
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[72:75], v[184:187]
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[72:75], v[188:191]
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[76:79], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[76:79], v[196:199]
	ds_read_b128 a[56:59], v13 offset:58240
	ds_read_b128 a[60:63], v13 offset:58256
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[76:79], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[76:79], v[204:207]
	ds_read_b128 a[88:91], v13 offset:60320
	ds_read_b128 a[92:95], v13 offset:60336
	ds_write_b64 v20, v[48:49]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[76:79], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[76:79], v[212:215]
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[76:79], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[76:79], v[220:223]
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v39, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:8320
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v242, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:8448
	ds_read_b64 v[52:53], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v24 offset:8320
	ds_read_b64_tr_b16 a[142:143], v24 offset:8448
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v23 offset:8832
	ds_read_b64_tr_b16 a[146:147], v23 offset:8960
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v24 offset:8832
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v24 offset:8960
	ds_read_b64 v[60:61], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	ds_read_b64 v[62:63], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_6D20
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_6D20:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:8192
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	ds_read_b128 a[40:43], v239 offset:8192
	ds_read_b128 a[44:47], v239 offset:8208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:8192
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	ds_read_b128 a[72:75], v239 offset:10272
	ds_read_b128 a[76:79], v239 offset:10288
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	ds_read_b128 v[232:235], v1 offset:10272
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	ds_read_b128 a[48:51], v240 offset:8192
	ds_read_b128 a[52:55], v240 offset:8208
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:10272
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	ds_read_b128 a[80:83], v240 offset:10272
	ds_read_b128 a[84:87], v240 offset:10288
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	ds_read_b128 a[56:59], v13 offset:8320
	ds_read_b128 a[60:63], v13 offset:8336
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	ds_read_b128 a[88:91], v13 offset:10400
	ds_read_b128 a[92:95], v13 offset:10416
	ds_write_b64 v20, v[64:65]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	ds_read_b128 a[104:107], v16
	ds_read_b128 a[108:111], v16 offset:16
	ds_write_b64 v20, v[66:67] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
	ds_read_b128 a[112:115], v16 offset:2080
	ds_read_b128 a[116:119], v16 offset:2096
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_bfe_u32 v228, v228, v25, 8
	v_bfe_u32 v229, v229, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v40, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:12480
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v243, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:12608
	ds_read_b64 v[68:69], v18 offset:512
	v_bfe_u32 v232, v232, v25, 8
	v_bfe_u32 v233, v233, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v24 offset:12480
	ds_read_b64_tr_b16 a[142:143], v24 offset:12608
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v23 offset:12992
	ds_read_b64_tr_b16 a[146:147], v23 offset:13120
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v24 offset:12992
	ds_read_b64 v[72:73], v18 offset:1024
	ds_read_b64 v[74:75], v19 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v24 offset:13120
	ds_read_b64 v[76:77], v18 offset:1536
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	ds_read_b64 v[78:79], v19 offset:1536
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cbranch_vccz label_749C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_749C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	ds_read_b128 v[228:231], v1 offset:24832
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[72:75], v[160:163]
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[72:75], v[164:167]
	ds_read_b128 a[40:43], v239 offset:24832
	ds_read_b128 a[44:47], v239 offset:24848
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_u8 v236, v238 offset:24832
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[72:75], v[168:171]
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[72:75], v[172:175]
	ds_read_b128 a[72:75], v239 offset:26912
	ds_read_b128 a[76:79], v239 offset:26928
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	ds_read_b128 v[232:235], v1 offset:26912
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[72:75], v[176:179]
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[72:75], v[180:183]
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	ds_read_u8 v237, v238 offset:26912
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[72:75], v[184:187]
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[72:75], v[188:191]
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[76:79], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[76:79], v[196:199]
	ds_read_b128 a[56:59], v13 offset:24960
	ds_read_b128 a[60:63], v13 offset:24976
	ds_write_b32 v17, v33 offset:4096
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[76:79], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[76:79], v[204:207]
	ds_read_b128 a[88:91], v13 offset:27040
	ds_read_b128 a[92:95], v13 offset:27056
	ds_write_b64 v20, v[48:49]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[76:79], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[76:79], v[212:215]
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	ds_write_b64 v20, v[50:51] offset:2048
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[76:79], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[76:79], v[220:223]
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_5B68
label_795C:
	s_add_i32 s75, s88, -1
	s_and_b32 s75, s75, 3
	s_mul_i32 s75, s75, 0x1040
	v_add_u32_e32 v23, s75, v23
	v_add_u32_e32 v24, s75, v24
	s_cmp_eq_u32 s72, 64
	s_cbranch_scc1 label_7A20
	s_cmp_eq_u32 s72, 16
	s_cbranch_scc0 label_79C4
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b32 v34, v43 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cbranch_vccz label_79C0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_79C0:
	s_branch label_7AB0
label_79C4:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cbranch_vccz label_7A1C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_7A1C:
	s_branch label_7AB0
label_7A20:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	ds_read_b64 v[60:61], v18 offset:1536
	ds_read_b64 v[62:63], v19 offset:1536
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cbranch_vccz label_7AB0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_7AB0:
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc1 label_7B3C
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	s_branch label_7BBC
label_7B3C:
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
label_7BBC:
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	s_cmp_le_u32 s72, 16
	s_cbranch_scc1 label_7CD0
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	s_cmp_le_u32 s72, 32
	s_cbranch_scc1 label_7CD0
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
label_7CD0:
	s_add_i32 s75, s88, -1
	s_and_b32 s75, s75, 3
	s_mul_i32 s75, s75, 0xffffefc0
	v_add_u32_e32 v23, s75, v23
	v_add_u32_e32 v24, s75, v24
	s_cmp_eq_i32 s70, 0
	s_cbranch_scc1 label_7CF4
	s_branch label_8EFC
label_7CF4:
	s_cmp_eq_u32 s4, 0
	s_cbranch_scc0 label_7D78
	v_and_b32_e64 v33, v0, 15
	v_lshlrev_b32_e32 v33, 2, v33
	s_lshl_b32 s75, s7, 4
	s_sub_u32 s76, s80, 1
	s_min_u32 s76, s76, 63
	s_and_b32 s75, s75, s76
	s_lshl_b32 s76, s2, 6
	s_add_u32 s75, s75, s76
	s_lshl_b32 s75, s75, 2
	v_add_u32_e32 v33, s75, v33
	buffer_load_dword v30, v33, s[52:55], 0 offen
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v30, 0x41b504f3, v30
	v_max_f32_e32 v31, v45, v30
	v_sub_f32_e32 v33, v45, v31
	v_mul_f32_e64 v33, v33, s5
	v_exp_f32_e32 v32, v33
	v_sub_f32_e32 v30, v30, v31
	v_mul_f32_e64 v30, v30, s5
	v_exp_f32_e32 v30, v30
	s_nop 0
	v_mul_f32_e32 v30, 0x3e800000, v30
	v_fma_f32 v46, v46, v32, v30
	v_mov_b32_e32 v45, v31
label_7D78:
	v_mov_b32_e32 v26, v46
	v_mov_b32_e32 v27, v46
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_mov_b32_e32 v46, 0
	v_add_f32_e32 v46, v26, v46
	v_add_f32_e32 v46, v27, v46
	v_add_f32_e32 v46, v28, v46
	v_add_f32_e32 v46, v29, v46
	v_mov_b32_e32 v26, 0
	v_cmp_eq_u32_e64 s[64:65], v26, v46
	v_mul_f32_e64 v26, v45, s79
	v_log_f32_e32 v27, v46
	v_cndmask_b32_e64 v46, v46, 1.0, s[64:65]
	s_nop 1
	v_rcp_f32_e32 v46, v46
	s_nop 1
	v_fma_f32 v33, v27, s6, v26
	s_cmp_eq_u32 s4, 0
	s_cbranch_scc0 label_7DF0
	v_mul_f32_e32 v46, v46, v32
label_7DF0:
	s_mov_b32 s87, s72
	v_lshlrev_b32_e32 v30, 2, v0
	s_mul_i32 s75, 0x100, s7
	v_add_u32_e32 v30, s75, v30
	ds_write_b32 v30, v46 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v26, v43 offset:4096
	s_cmp_le_u32 s87, 16
	s_cbranch_scc1 label_7E44
	ds_read_b32 v27, v43 offset:4352
	s_cmp_le_u32 s87, 32
	s_cbranch_scc1 label_7E44
	ds_read_b32 v28, v43 offset:4608
	ds_read_b32 v29, v43 offset:4864
label_7E44:
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[96:97], v[26:27], v[96:97] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[26:27], v[98:99] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[26:27], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[26:27], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[26:27], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[26:27], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[26:27], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[26:27], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[26:27], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115], v[26:27], v[114:115] op_sel_hi:[0,1]
	v_pk_mul_f32 v[116:117], v[26:27], v[116:117] op_sel_hi:[0,1]
	v_pk_mul_f32 v[118:119], v[26:27], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121], v[26:27], v[120:121] op_sel_hi:[0,1]
	v_pk_mul_f32 v[122:123], v[26:27], v[122:123] op_sel_hi:[0,1]
	v_pk_mul_f32 v[124:125], v[26:27], v[124:125] op_sel_hi:[0,1]
	v_pk_mul_f32 v[126:127], v[26:27], v[126:127] op_sel_hi:[0,1]
	s_cmp_le_u32 s87, 16
	s_cbranch_scc1 label_8058
	v_pk_mul_f32 v[128:129], v[26:27], v[128:129] op_sel:[1,0]
	v_pk_mul_f32 v[130:131], v[26:27], v[130:131] op_sel:[1,0]
	v_pk_mul_f32 v[132:133], v[26:27], v[132:133] op_sel:[1,0]
	v_pk_mul_f32 v[134:135], v[26:27], v[134:135] op_sel:[1,0]
	v_pk_mul_f32 v[136:137], v[26:27], v[136:137] op_sel:[1,0]
	v_pk_mul_f32 v[138:139], v[26:27], v[138:139] op_sel:[1,0]
	v_pk_mul_f32 v[140:141], v[26:27], v[140:141] op_sel:[1,0]
	v_pk_mul_f32 v[142:143], v[26:27], v[142:143] op_sel:[1,0]
	v_pk_mul_f32 v[144:145], v[26:27], v[144:145] op_sel:[1,0]
	v_pk_mul_f32 v[146:147], v[26:27], v[146:147] op_sel:[1,0]
	v_pk_mul_f32 v[148:149], v[26:27], v[148:149] op_sel:[1,0]
	v_pk_mul_f32 v[150:151], v[26:27], v[150:151] op_sel:[1,0]
	v_pk_mul_f32 v[152:153], v[26:27], v[152:153] op_sel:[1,0]
	v_pk_mul_f32 v[154:155], v[26:27], v[154:155] op_sel:[1,0]
	v_pk_mul_f32 v[156:157], v[26:27], v[156:157] op_sel:[1,0]
	v_pk_mul_f32 v[158:159], v[26:27], v[158:159] op_sel:[1,0]
	s_cmp_le_u32 s87, 32
	s_cbranch_scc1 label_8058
	v_pk_mul_f32 v[160:161], v[28:29], v[160:161] op_sel_hi:[0,1]
	v_pk_mul_f32 v[162:163], v[28:29], v[162:163] op_sel_hi:[0,1]
	v_pk_mul_f32 v[164:165], v[28:29], v[164:165] op_sel_hi:[0,1]
	v_pk_mul_f32 v[166:167], v[28:29], v[166:167] op_sel_hi:[0,1]
	v_pk_mul_f32 v[168:169], v[28:29], v[168:169] op_sel_hi:[0,1]
	v_pk_mul_f32 v[170:171], v[28:29], v[170:171] op_sel_hi:[0,1]
	v_pk_mul_f32 v[172:173], v[28:29], v[172:173] op_sel_hi:[0,1]
	v_pk_mul_f32 v[174:175], v[28:29], v[174:175] op_sel_hi:[0,1]
	v_pk_mul_f32 v[176:177], v[28:29], v[176:177] op_sel_hi:[0,1]
	v_pk_mul_f32 v[178:179], v[28:29], v[178:179] op_sel_hi:[0,1]
	v_pk_mul_f32 v[180:181], v[28:29], v[180:181] op_sel_hi:[0,1]
	v_pk_mul_f32 v[182:183], v[28:29], v[182:183] op_sel_hi:[0,1]
	v_pk_mul_f32 v[184:185], v[28:29], v[184:185] op_sel_hi:[0,1]
	v_pk_mul_f32 v[186:187], v[28:29], v[186:187] op_sel_hi:[0,1]
	v_pk_mul_f32 v[188:189], v[28:29], v[188:189] op_sel_hi:[0,1]
	v_pk_mul_f32 v[190:191], v[28:29], v[190:191] op_sel_hi:[0,1]
	v_pk_mul_f32 v[192:193], v[28:29], v[192:193] op_sel:[1,0]
	v_pk_mul_f32 v[194:195], v[28:29], v[194:195] op_sel:[1,0]
	v_pk_mul_f32 v[196:197], v[28:29], v[196:197] op_sel:[1,0]
	v_pk_mul_f32 v[198:199], v[28:29], v[198:199] op_sel:[1,0]
	v_pk_mul_f32 v[200:201], v[28:29], v[200:201] op_sel:[1,0]
	v_pk_mul_f32 v[202:203], v[28:29], v[202:203] op_sel:[1,0]
	v_pk_mul_f32 v[204:205], v[28:29], v[204:205] op_sel:[1,0]
	v_pk_mul_f32 v[206:207], v[28:29], v[206:207] op_sel:[1,0]
	v_pk_mul_f32 v[208:209], v[28:29], v[208:209] op_sel:[1,0]
	v_pk_mul_f32 v[210:211], v[28:29], v[210:211] op_sel:[1,0]
	v_pk_mul_f32 v[212:213], v[28:29], v[212:213] op_sel:[1,0]
	v_pk_mul_f32 v[214:215], v[28:29], v[214:215] op_sel:[1,0]
	v_pk_mul_f32 v[216:217], v[28:29], v[216:217] op_sel:[1,0]
	v_pk_mul_f32 v[218:219], v[28:29], v[218:219] op_sel:[1,0]
	v_pk_mul_f32 v[220:221], v[28:29], v[220:221] op_sel:[1,0]
	v_pk_mul_f32 v[222:223], v[28:29], v[222:223] op_sel:[1,0]
label_8058:
	s_cmp_gt_u32 s82, 1
	s_cbranch_scc1 label_8068
	s_cmp_eq_u32 s83, 1
	s_cbranch_scc1 label_86D0
label_8068:
	s_mul_i32 s95, 0x800, s80
	s_mul_i32 s96, s81, s95
	s_mul_i32 s75, s89, s96
	s_add_u32 s8, s75, s8
	s_addc_u32 s9, 0, s9
	s_mov_b32 s75, 0x20000
	s_mul_i32 s75, s75, s2
	s_add_u32 s8, s75, s8
	s_addc_u32 s9, 0, s9
	s_mul_i32 s75, s96, s91
	s_mov_b32 s10, s75
	v_and_b32_e32 v26, 15, v0
	v_lshlrev_b32_e32 v2, 4, v26
	v_lshrrev_b32_e32 v26, 4, v0
	v_mul_i32_i24_e32 v26, 0x800, v26
	v_add_u32_e32 v2, v2, v26
	s_mul_i32 s75, s4, s95
	v_add_u32_e64 v2, v2, s75
	s_mul_i32 s75, s7, 0x200
	v_add_u32_e64 v2, v2, s75
	v_mov_b32_e32 v1, v2
	s_lshl_b32 s64, s7, 4
	s_ff1_i32_b32 s78, s80
	s_lshr_b32 s66, s64, s78
	s_sub_u32 s78, s80, 1
	s_and_b32 s76, s64, s78
	s_mul_i32 s97, 4, s80
	s_mul_i32 s77, s81, s97
	s_mul_i32 s75, s89, s77
	s_mul_i32 s78, s4, s97
	s_add_u32 s75, s75, s78
	s_mov_b32 s78, 0x100
	s_mul_i32 s78, s78, s2
	s_add_u32 s75, s75, s78
	s_add_u32 s12, s75, s12
	s_addc_u32 s13, 0, s13
	s_mul_i32 s75, s91, s77
	s_mov_b32 s14, s75
	v_and_b32_e32 v26, 15, v0
	v_lshlrev_b32_e32 v26, 2, v26
	s_lshl_b32 s76, s76, 2
	v_add_u32_e64 v26, v26, s76
	s_mul_i32 s76, s66, s77
	v_add_u32_e64 v26, v26, s76
	buffer_store_dword v33, v26, s[12:15], 0 offen
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_and_b32_e32 v26, 3, v0
	v_lshlrev_b32_e32 v16, 2, v26
	v_bfe_u32 v26, v0, 2, 2
	v_lshl_add_u32 v16, v26, 4, v16
	v_lshrrev_b32_e32 v26, 4, v0
	v_mul_i32_i24_e32 v26, 0x44, v26
	v_add_u32_e32 v16, v26, v16
	s_mul_i32 s75, s7, 0x880
	v_add_u32_e32 v16, s75, v16
	v_lshlrev_b32_e32 v16, 2, v16
	v_lshrrev_b32_e32 v26, 4, v0
	v_mul_i32_i24_e32 v10, 4, v26
	v_and_b32_e32 v26, 3, v0
	v_mul_i32_i24_e32 v26, 0x44, v26
	v_add_u32_e32 v10, v26, v10
	v_and_b32_e32 v26, 15, v0
	v_lshrrev_b32_e32 v26, 2, v26
	v_mul_i32_i24_e32 v26, 0x110, v26
	v_add_u32_e32 v10, v26, v10
	s_mul_i32 s75, s7, 0x880
	v_add_u32_e32 v10, s75, v10
	v_lshlrev_b32_e32 v10, 2, v10
	v_mov_b32_e32 v2, v1
	v_mov_b64_e32 v[26:27], v[96:97]
	v_mov_b64_e32 v[28:29], v[98:99]
	ds_write_b128 v16, v[26:29]
	v_mov_b64_e32 v[26:27], v[100:101]
	v_mov_b64_e32 v[28:29], v[102:103]
	ds_write_b128 v16, v[26:29] offset:1088
	v_mov_b64_e32 v[26:27], v[104:105]
	v_mov_b64_e32 v[28:29], v[106:107]
	ds_write_b128 v16, v[26:29] offset:2176
	v_mov_b64_e32 v[26:27], v[108:109]
	v_mov_b64_e32 v[28:29], v[110:111]
	ds_write_b128 v16, v[26:29] offset:3264
	v_mov_b64_e32 v[26:27], v[112:113]
	v_mov_b64_e32 v[28:29], v[114:115]
	ds_write_b128 v16, v[26:29] offset:4352
	v_mov_b64_e32 v[26:27], v[116:117]
	v_mov_b64_e32 v[28:29], v[118:119]
	ds_write_b128 v16, v[26:29] offset:5440
	v_mov_b64_e32 v[26:27], v[120:121]
	v_mov_b64_e32 v[28:29], v[122:123]
	ds_write_b128 v16, v[26:29] offset:6528
	v_mov_b64_e32 v[26:27], v[124:125]
	v_mov_b64_e32 v[28:29], v[126:127]
	ds_write_b128 v16, v[26:29] offset:7616
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[96:99], v10
	ds_read_b128 v[100:103], v10 offset:64
	ds_read_b128 v[104:107], v10 offset:128
	ds_read_b128 v[108:111], v10 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[112:115], v10 offset:4352
	ds_read_b128 v[116:119], v10 offset:4416
	ds_read_b128 v[120:123], v10 offset:4480
	ds_read_b128 v[124:127], v10 offset:4544
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[112:115], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[116:119], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[120:123], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[124:127], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	s_cmp_le_u32 s87, 16
	s_cbranch_scc1 label_86CC
	s_sub_u32 s76, s96, s95
	s_sub_u32 s77, s80, 1
	s_and_b32 s75, s77, 16
	s_cmp_eq_u32 s75, 0
	s_cselect_b32 s75, s76, 0
	v_add_u32_e32 v2, s75, v2
	v_mov_b64_e32 v[26:27], v[128:129]
	v_mov_b64_e32 v[28:29], v[130:131]
	ds_write_b128 v16, v[26:29]
	v_mov_b64_e32 v[26:27], v[132:133]
	v_mov_b64_e32 v[28:29], v[134:135]
	ds_write_b128 v16, v[26:29] offset:1088
	v_mov_b64_e32 v[26:27], v[136:137]
	v_mov_b64_e32 v[28:29], v[138:139]
	ds_write_b128 v16, v[26:29] offset:2176
	v_mov_b64_e32 v[26:27], v[140:141]
	v_mov_b64_e32 v[28:29], v[142:143]
	ds_write_b128 v16, v[26:29] offset:3264
	v_mov_b64_e32 v[26:27], v[144:145]
	v_mov_b64_e32 v[28:29], v[146:147]
	ds_write_b128 v16, v[26:29] offset:4352
	v_mov_b64_e32 v[26:27], v[148:149]
	v_mov_b64_e32 v[28:29], v[150:151]
	ds_write_b128 v16, v[26:29] offset:5440
	v_mov_b64_e32 v[26:27], v[152:153]
	v_mov_b64_e32 v[28:29], v[154:155]
	ds_write_b128 v16, v[26:29] offset:6528
	v_mov_b64_e32 v[26:27], v[156:157]
	v_mov_b64_e32 v[28:29], v[158:159]
	ds_write_b128 v16, v[26:29] offset:7616
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[96:99], v10
	ds_read_b128 v[100:103], v10 offset:64
	ds_read_b128 v[104:107], v10 offset:128
	ds_read_b128 v[108:111], v10 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[112:115], v10 offset:4352
	ds_read_b128 v[116:119], v10 offset:4416
	ds_read_b128 v[120:123], v10 offset:4480
	ds_read_b128 v[124:127], v10 offset:4544
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[112:115], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[116:119], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[120:123], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[124:127], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	s_cmp_le_u32 s87, 32
	s_cbranch_scc1 label_86CC
	s_sub_u32 s76, s96, s95
	s_sub_u32 s77, s80, 1
	s_and_b32 s75, s77, 32
	s_cmp_eq_u32 s75, 0
	s_cselect_b32 s75, s76, 0
	v_add_u32_e32 v2, s75, v2
	v_mov_b64_e32 v[26:27], v[160:161]
	v_mov_b64_e32 v[28:29], v[162:163]
	ds_write_b128 v16, v[26:29]
	v_mov_b64_e32 v[26:27], v[164:165]
	v_mov_b64_e32 v[28:29], v[166:167]
	ds_write_b128 v16, v[26:29] offset:1088
	v_mov_b64_e32 v[26:27], v[168:169]
	v_mov_b64_e32 v[28:29], v[170:171]
	ds_write_b128 v16, v[26:29] offset:2176
	v_mov_b64_e32 v[26:27], v[172:173]
	v_mov_b64_e32 v[28:29], v[174:175]
	ds_write_b128 v16, v[26:29] offset:3264
	v_mov_b64_e32 v[26:27], v[176:177]
	v_mov_b64_e32 v[28:29], v[178:179]
	ds_write_b128 v16, v[26:29] offset:4352
	v_mov_b64_e32 v[26:27], v[180:181]
	v_mov_b64_e32 v[28:29], v[182:183]
	ds_write_b128 v16, v[26:29] offset:5440
	v_mov_b64_e32 v[26:27], v[184:185]
	v_mov_b64_e32 v[28:29], v[186:187]
	ds_write_b128 v16, v[26:29] offset:6528
	v_mov_b64_e32 v[26:27], v[188:189]
	v_mov_b64_e32 v[28:29], v[190:191]
	ds_write_b128 v16, v[26:29] offset:7616
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[96:99], v10
	ds_read_b128 v[100:103], v10 offset:64
	ds_read_b128 v[104:107], v10 offset:128
	ds_read_b128 v[108:111], v10 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[112:115], v10 offset:4352
	ds_read_b128 v[116:119], v10 offset:4416
	ds_read_b128 v[120:123], v10 offset:4480
	ds_read_b128 v[124:127], v10 offset:4544
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[112:115], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[116:119], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[120:123], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[124:127], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	s_sub_u32 s76, s96, s95
	s_sub_u32 s77, s80, 1
	s_and_b32 s75, s77, 48
	s_cmp_eq_u32 s75, 0
	s_cselect_b32 s75, s76, 0
	v_add_u32_e32 v2, s75, v2
	v_mov_b64_e32 v[26:27], v[192:193]
	v_mov_b64_e32 v[28:29], v[194:195]
	ds_write_b128 v16, v[26:29]
	v_mov_b64_e32 v[26:27], v[196:197]
	v_mov_b64_e32 v[28:29], v[198:199]
	ds_write_b128 v16, v[26:29] offset:1088
	v_mov_b64_e32 v[26:27], v[200:201]
	v_mov_b64_e32 v[28:29], v[202:203]
	ds_write_b128 v16, v[26:29] offset:2176
	v_mov_b64_e32 v[26:27], v[204:205]
	v_mov_b64_e32 v[28:29], v[206:207]
	ds_write_b128 v16, v[26:29] offset:3264
	v_mov_b64_e32 v[26:27], v[208:209]
	v_mov_b64_e32 v[28:29], v[210:211]
	ds_write_b128 v16, v[26:29] offset:4352
	v_mov_b64_e32 v[26:27], v[212:213]
	v_mov_b64_e32 v[28:29], v[214:215]
	ds_write_b128 v16, v[26:29] offset:5440
	v_mov_b64_e32 v[26:27], v[216:217]
	v_mov_b64_e32 v[28:29], v[218:219]
	ds_write_b128 v16, v[26:29] offset:6528
	v_mov_b64_e32 v[26:27], v[220:221]
	v_mov_b64_e32 v[28:29], v[222:223]
	ds_write_b128 v16, v[26:29] offset:7616
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[96:99], v10
	ds_read_b128 v[100:103], v10 offset:64
	ds_read_b128 v[104:107], v10 offset:128
	ds_read_b128 v[108:111], v10 offset:192
	s_waitcnt lgkmcnt(4)
	ds_read_b128 v[112:115], v10 offset:4352
	ds_read_b128 v[116:119], v10 offset:4416
	ds_read_b128 v[120:123], v10 offset:4480
	ds_read_b128 v[124:127], v10 offset:4544
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[112:115], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[116:119], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[120:123], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[124:127], v2, s[8:11], 0 offen offset:256
	v_add_u32_e32 v2, 0x2000, v2
label_86CC:
	s_branch label_8EF4
label_86D0:
	s_mul_i32 s95, 0x400, s80
	s_mul_i32 s96, s82, s95
	s_mul_i32 s75, s89, s96
	s_add_u32 s8, s75, s8
	s_addc_u32 s9, 0, s9
	s_mov_b32 s75, 0x10000
	s_mul_i32 s75, s75, s2
	s_add_u32 s8, s75, s8
	s_addc_u32 s9, 0, s9
	s_mul_i32 s75, s96, s91
	s_mov_b32 s10, s75
	v_and_b32_e32 v26, 7, v0
	v_lshlrev_b32_e32 v2, 4, v26
	v_lshrrev_b32_e32 v26, 3, v0
	v_mul_i32_i24_e32 v26, 0x400, v26
	v_add_u32_e32 v2, v2, v26
	s_mul_i32 s75, s4, s95
	v_add_u32_e64 v2, v2, s75
	s_mul_i32 s75, s7, 0x100
	v_add_u32_e64 v2, v2, s75
	v_mov_b32_e32 v1, v2
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_lshrrev_b32_e32 v26, 4, v0
	v_mul_i32_i24_e32 v16, 0x48, v26
	v_and_b32_e32 v26, 15, v0
	v_mul_i32_i24_e32 v26, 2, v26
	v_add_u32_e32 v16, v26, v16
	s_mul_i32 s75, s7, 0x480
	v_add_u32_e32 v16, s75, v16
	v_lshlrev_b32_e32 v16, 2, v16
	v_lshrrev_b32_e32 v26, 3, v0
	v_mul_i32_i24_e32 v10, 2, v26
	v_and_b32_e32 v26, 3, v0
	v_mul_i32_i24_e32 v26, 0x90, v26
	v_add_u32_e32 v10, v26, v10
	v_lshrrev_b32_e32 v26, 2, v0
	v_and_b32_e32 v26, 1, v26
	v_mul_i32_i24_e32 v26, 36, v26
	v_add_u32_e32 v10, v26, v10
	s_mul_i32 s75, s7, 0x480
	v_add_u32_e32 v10, s75, v10
	v_lshlrev_b32_e32 v10, 2, v10
	v_mov_b32_e32 v2, v1
	v_mov_b64_e32 v[30:31], v[96:97]
	v_mov_b64_e32 v[32:33], v[98:99]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31]
	v_mov_b64_e32 v[30:31], v[100:101]
	v_mov_b64_e32 v[32:33], v[102:103]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1152
	v_mov_b64_e32 v[30:31], v[104:105]
	v_mov_b64_e32 v[32:33], v[106:107]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:144
	v_mov_b64_e32 v[30:31], v[108:109]
	v_mov_b64_e32 v[32:33], v[110:111]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1296
	v_mov_b64_e32 v[30:31], v[112:113]
	v_mov_b64_e32 v[32:33], v[114:115]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2304
	v_mov_b64_e32 v[30:31], v[116:117]
	v_mov_b64_e32 v[32:33], v[118:119]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3456
	v_mov_b64_e32 v[30:31], v[120:121]
	v_mov_b64_e32 v[32:33], v[122:123]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2448
	v_mov_b64_e32 v[30:31], v[124:125]
	v_mov_b64_e32 v[32:33], v[126:127]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[96:97], v10
	ds_read_b64 v[100:101], v10 offset:64
	ds_read_b64 v[98:99], v10 offset:288
	ds_read_b64 v[102:103], v10 offset:352
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[104:105], v10 offset:2304
	ds_read_b64 v[108:109], v10 offset:2368
	ds_read_b64 v[106:107], v10 offset:2592
	ds_read_b64 v[110:111], v10 offset:2656
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	s_cmp_le_u32 s87, 16
	s_cbranch_scc1 label_8DA8
	v_mov_b64_e32 v[30:31], v[128:129]
	v_mov_b64_e32 v[32:33], v[130:131]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31]
	v_mov_b64_e32 v[30:31], v[132:133]
	v_mov_b64_e32 v[32:33], v[134:135]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1152
	v_mov_b64_e32 v[30:31], v[136:137]
	v_mov_b64_e32 v[32:33], v[138:139]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:144
	v_mov_b64_e32 v[30:31], v[140:141]
	v_mov_b64_e32 v[32:33], v[142:143]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1296
	v_mov_b64_e32 v[30:31], v[144:145]
	v_mov_b64_e32 v[32:33], v[146:147]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2304
	v_mov_b64_e32 v[30:31], v[148:149]
	v_mov_b64_e32 v[32:33], v[150:151]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3456
	v_mov_b64_e32 v[30:31], v[152:153]
	v_mov_b64_e32 v[32:33], v[154:155]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2448
	v_mov_b64_e32 v[30:31], v[156:157]
	v_mov_b64_e32 v[32:33], v[158:159]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[96:97], v10
	ds_read_b64 v[100:101], v10 offset:64
	ds_read_b64 v[98:99], v10 offset:288
	ds_read_b64 v[102:103], v10 offset:352
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[104:105], v10 offset:2304
	ds_read_b64 v[108:109], v10 offset:2368
	ds_read_b64 v[106:107], v10 offset:2592
	ds_read_b64 v[110:111], v10 offset:2656
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	s_cmp_le_u32 s87, 32
	s_cbranch_scc1 label_8DA8
	v_mov_b64_e32 v[30:31], v[160:161]
	v_mov_b64_e32 v[32:33], v[162:163]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31]
	v_mov_b64_e32 v[30:31], v[164:165]
	v_mov_b64_e32 v[32:33], v[166:167]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1152
	v_mov_b64_e32 v[30:31], v[168:169]
	v_mov_b64_e32 v[32:33], v[170:171]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:144
	v_mov_b64_e32 v[30:31], v[172:173]
	v_mov_b64_e32 v[32:33], v[174:175]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1296
	v_mov_b64_e32 v[30:31], v[176:177]
	v_mov_b64_e32 v[32:33], v[178:179]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2304
	v_mov_b64_e32 v[30:31], v[180:181]
	v_mov_b64_e32 v[32:33], v[182:183]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3456
	v_mov_b64_e32 v[30:31], v[184:185]
	v_mov_b64_e32 v[32:33], v[186:187]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2448
	v_mov_b64_e32 v[30:31], v[188:189]
	v_mov_b64_e32 v[32:33], v[190:191]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[96:97], v10
	ds_read_b64 v[100:101], v10 offset:64
	ds_read_b64 v[98:99], v10 offset:288
	ds_read_b64 v[102:103], v10 offset:352
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[104:105], v10 offset:2304
	ds_read_b64 v[108:109], v10 offset:2368
	ds_read_b64 v[106:107], v10 offset:2592
	ds_read_b64 v[110:111], v10 offset:2656
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	v_mov_b64_e32 v[30:31], v[192:193]
	v_mov_b64_e32 v[32:33], v[194:195]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31]
	v_mov_b64_e32 v[30:31], v[196:197]
	v_mov_b64_e32 v[32:33], v[198:199]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1152
	v_mov_b64_e32 v[30:31], v[200:201]
	v_mov_b64_e32 v[32:33], v[202:203]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:144
	v_mov_b64_e32 v[30:31], v[204:205]
	v_mov_b64_e32 v[32:33], v[206:207]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:1296
	v_mov_b64_e32 v[30:31], v[208:209]
	v_mov_b64_e32 v[32:33], v[210:211]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2304
	v_mov_b64_e32 v[30:31], v[212:213]
	v_mov_b64_e32 v[32:33], v[214:215]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3456
	v_mov_b64_e32 v[30:31], v[216:217]
	v_mov_b64_e32 v[32:33], v[218:219]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:2448
	v_mov_b64_e32 v[30:31], v[220:221]
	v_mov_b64_e32 v[32:33], v[222:223]
	v_cvt_pk_bf16_f32 v30, v30, v31
	v_cvt_pk_bf16_f32 v31, v32, v33
	ds_write_b64 v16, v[30:31] offset:3600
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[96:97], v10
	ds_read_b64 v[100:101], v10 offset:64
	ds_read_b64 v[98:99], v10 offset:288
	ds_read_b64 v[102:103], v10 offset:352
	s_waitcnt lgkmcnt(4)
	ds_read_b64 v[104:105], v10 offset:2304
	ds_read_b64 v[108:109], v10 offset:2368
	ds_read_b64 v[106:107], v10 offset:2592
	ds_read_b64 v[110:111], v10 offset:2656
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[96:99], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[104:107], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
	buffer_store_dwordx4 v[100:103], v2, s[8:11], 0 offen
	buffer_store_dwordx4 v[108:111], v2, s[8:11], 0 offen offset:128
	v_add_u32_e32 v2, 0x2000, v2
label_8DA8:
	s_branch label_8EF4
label_8DAC:
	s_cmp_eq_u32 s99, 1
	s_cbranch_scc0 label_8DC4
	s_waitcnt vmcnt(0)
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_branch label_8EF4
label_8DC4:
	s_cmp_eq_u32 s80, 16
	s_cselect_b32 s69, 16, 64
	s_cmp_eq_u32 s80, 32
	s_cselect_b32 s69, 32, s69
	s_cmp_eq_u32 s91, 1
	s_cselect_b32 s69, s69, 64
	s_mov_b32 s72, s69
	v_mov_b32_e32 v33, 0xe0ad78ec
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v43, 0
	v_mov_b64_e32 v[96:97], 0
	v_mov_b64_e32 v[98:99], 0
	v_mov_b64_e32 v[100:101], 0
	v_mov_b64_e32 v[102:103], 0
	v_mov_b64_e32 v[104:105], 0
	v_mov_b64_e32 v[106:107], 0
	v_mov_b64_e32 v[108:109], 0
	v_mov_b64_e32 v[110:111], 0
	v_mov_b64_e32 v[112:113], 0
	v_mov_b64_e32 v[114:115], 0
	v_mov_b64_e32 v[116:117], 0
	v_mov_b64_e32 v[118:119], 0
	v_mov_b64_e32 v[120:121], 0
	v_mov_b64_e32 v[122:123], 0
	v_mov_b64_e32 v[124:125], 0
	v_mov_b64_e32 v[126:127], 0
	v_mov_b64_e32 v[128:129], 0
	v_mov_b64_e32 v[130:131], 0
	v_mov_b64_e32 v[132:133], 0
	v_mov_b64_e32 v[134:135], 0
	v_mov_b64_e32 v[136:137], 0
	v_mov_b64_e32 v[138:139], 0
	v_mov_b64_e32 v[140:141], 0
	v_mov_b64_e32 v[142:143], 0
	v_mov_b64_e32 v[144:145], 0
	v_mov_b64_e32 v[146:147], 0
	v_mov_b64_e32 v[148:149], 0
	v_mov_b64_e32 v[150:151], 0
	v_mov_b64_e32 v[152:153], 0
	v_mov_b64_e32 v[154:155], 0
	v_mov_b64_e32 v[156:157], 0
	v_mov_b64_e32 v[158:159], 0
	v_mov_b64_e32 v[160:161], 0
	v_mov_b64_e32 v[162:163], 0
	v_mov_b64_e32 v[164:165], 0
	v_mov_b64_e32 v[166:167], 0
	v_mov_b64_e32 v[168:169], 0
	v_mov_b64_e32 v[170:171], 0
	v_mov_b64_e32 v[172:173], 0
	v_mov_b64_e32 v[174:175], 0
	v_mov_b64_e32 v[176:177], 0
	v_mov_b64_e32 v[178:179], 0
	v_mov_b64_e32 v[180:181], 0
	v_mov_b64_e32 v[182:183], 0
	v_mov_b64_e32 v[184:185], 0
	v_mov_b64_e32 v[186:187], 0
	v_mov_b64_e32 v[188:189], 0
	v_mov_b64_e32 v[190:191], 0
	v_mov_b64_e32 v[192:193], 0
	v_mov_b64_e32 v[194:195], 0
	v_mov_b64_e32 v[196:197], 0
	v_mov_b64_e32 v[198:199], 0
	v_mov_b64_e32 v[200:201], 0
	v_mov_b64_e32 v[202:203], 0
	v_mov_b64_e32 v[204:205], 0
	v_mov_b64_e32 v[206:207], 0
	v_mov_b64_e32 v[208:209], 0
	v_mov_b64_e32 v[210:211], 0
	v_mov_b64_e32 v[212:213], 0
	v_mov_b64_e32 v[214:215], 0
	v_mov_b64_e32 v[216:217], 0
	v_mov_b64_e32 v[218:219], 0
	v_mov_b64_e32 v[220:221], 0
	v_mov_b64_e32 v[222:223], 0
	s_branch label_7DF0
label_8EF4:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
label_8EFC:
	s_waitcnt vmcnt(0)
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_bfe_u32 v230, v230, v25, 8
	v_and_b32_e32 v231, v47, v231
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_bfe_u32 v234, v234, v25, 8
	v_and_b32_e32 v235, v47, v235
	s_and_b32 s77, s88, 3
	s_mul_i32 s75, s77, 0x4100
	v_add_u32_e32 v30, s75, v13
	ds_read_b128 a[56:59], v30 offset:8320
	ds_read_b128 a[60:63], v30 offset:8336
	ds_read_b128 a[88:91], v30 offset:10400
	ds_read_b128 a[92:95], v30 offset:10416
	v_add_u32_e32 v30, s75, v241
	ds_read_b128 a[64:67], v30 offset:8384
	ds_read_b128 a[68:71], v30 offset:8400
	ds_read_b128 a[96:99], v30 offset:10464
	ds_read_b128 a[100:103], v30 offset:10480
	v_add_u32_e32 v31, s75, v239
	ds_read_b128 a[40:43], v31 offset:8192
	ds_read_b128 a[44:47], v31 offset:8208
	ds_read_b128 a[72:75], v31 offset:10272
	ds_read_b128 a[76:79], v31 offset:10288
	v_add_u32_e32 v31, s75, v238
	v_add_u32_e32 v32, s75, v1
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[26:29] offset:14640
	s_mul_i32 s76, s77, 0x1040
	v_add_u32_e32 v31, s76, v16
	v_mov_b64_e32 v[26:27], 0
	v_mov_b64_e32 v[28:29], 0
	v_and_b32_e32 v30, 15, v0
	s_and_b32 s75, s70, 0xff
	s_sub_i32 s76, s75, 0
	v_cmp_le_i32_e32 vcc, s76, v30
	s_nop 4
	s_mov_b64 exec, vcc
	ds_write_b128 v11, v[26:29]
	ds_write_b128 v11, v[26:29] offset:16
	ds_write_b128 v11, v[26:29] offset:32
	ds_write_b128 v11, v[26:29] offset:48
	ds_write_b128 v31, v[26:29]
	ds_write_b128 v31, v[26:29] offset:16
	s_mov_b64 exec, -1
	s_sub_i32 s76, s75, 16
	v_cmp_le_i32_e32 vcc, s76, v30
	s_nop 4
	s_mov_b64 exec, vcc
	ds_write_b128 v11, v[26:29] offset:14592
	ds_write_b128 v11, v[26:29] offset:14608
	ds_write_b128 v11, v[26:29] offset:14624
	ds_write_b128 v11, v[26:29] offset:14640
	ds_write_b128 v31, v[26:29] offset:2080
	ds_write_b128 v31, v[26:29] offset:2096
	s_mov_b64 exec, -1
	s_cmp_eq_i32 s7, 3
	s_cbranch_scc1 label_97C0
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	v_mov_b32_e32 v31, 0xff800000
	s_and_b32 s75, s70, 0xff
	v_mov_b32_e32 v30, s75
	v_lshrrev_b32_e32 v26, 4, v0
	v_mul_i32_i24_e32 v26, 4, v26
	v_add_u32_e32 v27, 1, v26
	v_add_u32_e32 v28, 2, v26
	v_add_u32_e32 v29, 3, v26
	v_cmp_lt_u32_e64 s[64:65], v26, v30
	v_add_u32_e32 v26, 16, v26
	s_nop 0
	v_cndmask_b32_e64 v48, v31, v48, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v27, v30
	v_add_u32_e32 v27, 16, v27
	s_nop 0
	v_cndmask_b32_e64 v49, v31, v49, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v28, v30
	v_add_u32_e32 v28, 16, v28
	s_nop 0
	v_cndmask_b32_e64 v50, v31, v50, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v29, v30
	v_add_u32_e32 v29, 16, v29
	s_nop 0
	v_cndmask_b32_e64 v51, v31, v51, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v26, v30
	v_add_u32_e32 v26, 16, v26
	s_nop 0
	v_cndmask_b32_e64 v52, v31, v52, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v27, v30
	v_add_u32_e32 v27, 16, v27
	s_nop 0
	v_cndmask_b32_e64 v53, v31, v53, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v28, v30
	v_add_u32_e32 v28, 16, v28
	s_nop 0
	v_cndmask_b32_e64 v54, v31, v54, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v29, v30
	v_add_u32_e32 v29, 16, v29
	s_nop 0
	v_cndmask_b32_e64 v55, v31, v55, s[64:65]
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_cmp_eq_u32 s72, 64
	s_cbranch_scc1 label_9604
	s_cmp_eq_u32 s72, 16
	s_cbranch_scc0 label_9590
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b32 v34, v43 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cbranch_vccz label_958C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_958C:
	s_branch label_96AC
label_9590:
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cbranch_vccz label_9600
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_9600:
	s_branch label_96AC
label_9604:
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	ds_read_b64 v[60:61], v18 offset:1536
	ds_read_b64 v[62:63], v19 offset:1536
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cbranch_vccz label_96AC
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_96AC:
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	s_cmp_le_u32 s72, 16
	s_cbranch_scc1 label_97BC
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	s_cmp_le_u32 s72, 32
	s_cbranch_scc1 label_97BC
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
label_97BC:
	s_branch label_7CF4
label_97C0:
	s_and_b32 s75, s88, 3
	s_mul_i32 s75, s75, 0x1040
	v_add_u32_e32 v23, s75, v23
	v_add_u32_e32 v24, s75, v24
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[40:47], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[72:79], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	v_mov_b32_e32 v31, 0xff800000
	s_and_b32 s75, s70, 0xff
	v_mov_b32_e32 v30, s75
	v_lshrrev_b32_e32 v26, 4, v0
	v_mul_i32_i24_e32 v26, 4, v26
	v_add_u32_e32 v27, 1, v26
	v_add_u32_e32 v28, 2, v26
	v_add_u32_e32 v29, 3, v26
	v_cmp_lt_u32_e64 s[64:65], v26, v30
	v_add_u32_e32 v26, 16, v26
	s_nop 0
	v_cndmask_b32_e64 v48, v31, v48, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v27, v30
	v_add_u32_e32 v27, 16, v27
	s_nop 0
	v_cndmask_b32_e64 v49, v31, v49, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v28, v30
	v_add_u32_e32 v28, 16, v28
	s_nop 0
	v_cndmask_b32_e64 v50, v31, v50, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v29, v30
	v_add_u32_e32 v29, 16, v29
	s_nop 0
	v_cndmask_b32_e64 v51, v31, v51, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v26, v30
	v_add_u32_e32 v26, 16, v26
	s_nop 0
	v_cndmask_b32_e64 v52, v31, v52, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v27, v30
	v_add_u32_e32 v27, 16, v27
	s_nop 0
	v_cndmask_b32_e64 v53, v31, v53, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v28, v30
	v_add_u32_e32 v28, 16, v28
	s_nop 0
	v_cndmask_b32_e64 v54, v31, v54, s[64:65]
	v_cmp_lt_u32_e64 s[64:65], v29, v30
	v_add_u32_e32 v29, 16, v29
	s_nop 0
	v_cndmask_b32_e64 v55, v31, v55, s[64:65]
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_cmp_eq_u32 s72, 64
	s_cbranch_scc1 label_9BF8
	s_cmp_eq_u32 s72, 16
	s_cbranch_scc0 label_9B84
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b32 v34, v43 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cbranch_vccz label_9B80
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_9B80:
	s_branch label_9CA0
label_9B84:
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cbranch_vccz label_9BF4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_9BF4:
	s_branch label_9CA0
label_9BF8:
	ds_write_b32 v17, v33 offset:4096
	ds_write_b64 v20, v[48:49]
	ds_write_b64 v20, v[50:51] offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[48:49], v18
	ds_read_b64 v[50:51], v19
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64 v[56:57], v18 offset:1024
	ds_read_b64 v[58:59], v19 offset:1024
	ds_read_b64 v[60:61], v18 offset:1536
	ds_read_b64 v[62:63], v19 offset:1536
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b32 v36, v43 offset:4608
	ds_read_b32 v37, v43 offset:4864
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_min_f32_e32 v26, v34, v35
	v_min_f32_e32 v26, v26, v36
	v_min_f32_e32 v26, v26, v37
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cbranch_vccz label_9CA0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9DB4
label_9CA0:
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	s_cmp_le_u32 s72, 16
	s_cbranch_scc1 label_9DB0
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	s_cmp_le_u32 s72, 32
	s_cbranch_scc1 label_9DB0
	v_mfma_f32_16x16x32_bf16 v[160:163], a[120:123], v[56:59], v[160:163]
	v_mfma_f32_16x16x32_bf16 v[164:167], a[124:127], v[56:59], v[164:167]
	v_mfma_f32_16x16x32_bf16 v[168:171], a[128:131], v[56:59], v[168:171]
	v_mfma_f32_16x16x32_bf16 v[172:175], a[132:135], v[56:59], v[172:175]
	v_mfma_f32_16x16x32_bf16 v[176:179], a[136:139], v[56:59], v[176:179]
	v_mfma_f32_16x16x32_bf16 v[180:183], a[140:143], v[56:59], v[180:183]
	v_mfma_f32_16x16x32_bf16 v[184:187], a[144:147], v[56:59], v[184:187]
	v_mfma_f32_16x16x32_bf16 v[188:191], a[148:151], v[56:59], v[188:191]
	v_mfma_f32_16x16x32_bf16 v[192:195], a[120:123], v[60:63], v[192:195]
	v_mfma_f32_16x16x32_bf16 v[196:199], a[124:127], v[60:63], v[196:199]
	v_mfma_f32_16x16x32_bf16 v[200:203], a[128:131], v[60:63], v[200:203]
	v_mfma_f32_16x16x32_bf16 v[204:207], a[132:135], v[60:63], v[204:207]
	v_mfma_f32_16x16x32_bf16 v[208:211], a[136:139], v[60:63], v[208:211]
	v_mfma_f32_16x16x32_bf16 v[212:215], a[140:143], v[60:63], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[216:219], a[144:147], v[60:63], v[216:219]
	v_mfma_f32_16x16x32_bf16 v[220:223], a[148:151], v[60:63], v[220:223]
label_9DB0:
	s_branch label_7CF4
label_9DB4:
	v_pk_mul_f32 v[192:193], v[36:37], v[192:193] op_sel:[1,0]
	v_pk_mul_f32 v[194:195], v[36:37], v[194:195] op_sel:[1,0]
	v_pk_mul_f32 v[196:197], v[36:37], v[196:197] op_sel:[1,0]
	v_pk_mul_f32 v[198:199], v[36:37], v[198:199] op_sel:[1,0]
	v_pk_mul_f32 v[200:201], v[36:37], v[200:201] op_sel:[1,0]
	v_pk_mul_f32 v[202:203], v[36:37], v[202:203] op_sel:[1,0]
	v_pk_mul_f32 v[204:205], v[36:37], v[204:205] op_sel:[1,0]
	v_pk_mul_f32 v[206:207], v[36:37], v[206:207] op_sel:[1,0]
	v_pk_mul_f32 v[208:209], v[36:37], v[208:209] op_sel:[1,0]
	v_pk_mul_f32 v[210:211], v[36:37], v[210:211] op_sel:[1,0]
	v_pk_mul_f32 v[212:213], v[36:37], v[212:213] op_sel:[1,0]
	v_pk_mul_f32 v[214:215], v[36:37], v[214:215] op_sel:[1,0]
	v_pk_mul_f32 v[216:217], v[36:37], v[216:217] op_sel:[1,0]
	v_pk_mul_f32 v[218:219], v[36:37], v[218:219] op_sel:[1,0]
	v_pk_mul_f32 v[220:221], v[36:37], v[220:221] op_sel:[1,0]
	v_pk_mul_f32 v[222:223], v[36:37], v[222:223] op_sel:[1,0]
	v_pk_mul_f32 v[160:161], v[36:37], v[160:161] op_sel_hi:[0,1]
	v_pk_mul_f32 v[162:163], v[36:37], v[162:163] op_sel_hi:[0,1]
	v_pk_mul_f32 v[164:165], v[36:37], v[164:165] op_sel_hi:[0,1]
	v_pk_mul_f32 v[166:167], v[36:37], v[166:167] op_sel_hi:[0,1]
	v_pk_mul_f32 v[168:169], v[36:37], v[168:169] op_sel_hi:[0,1]
	v_pk_mul_f32 v[170:171], v[36:37], v[170:171] op_sel_hi:[0,1]
	v_pk_mul_f32 v[172:173], v[36:37], v[172:173] op_sel_hi:[0,1]
	v_pk_mul_f32 v[174:175], v[36:37], v[174:175] op_sel_hi:[0,1]
	v_pk_mul_f32 v[176:177], v[36:37], v[176:177] op_sel_hi:[0,1]
	v_pk_mul_f32 v[178:179], v[36:37], v[178:179] op_sel_hi:[0,1]
	v_pk_mul_f32 v[180:181], v[36:37], v[180:181] op_sel_hi:[0,1]
	v_pk_mul_f32 v[182:183], v[36:37], v[182:183] op_sel_hi:[0,1]
	v_pk_mul_f32 v[184:185], v[36:37], v[184:185] op_sel_hi:[0,1]
	v_pk_mul_f32 v[186:187], v[36:37], v[186:187] op_sel_hi:[0,1]
	v_pk_mul_f32 v[188:189], v[36:37], v[188:189] op_sel_hi:[0,1]
	v_pk_mul_f32 v[190:191], v[36:37], v[190:191] op_sel_hi:[0,1]
label_9EB4:
	v_pk_mul_f32 v[128:129], v[34:35], v[128:129] op_sel:[1,0]
	v_pk_mul_f32 v[130:131], v[34:35], v[130:131] op_sel:[1,0]
	v_pk_mul_f32 v[132:133], v[34:35], v[132:133] op_sel:[1,0]
	v_pk_mul_f32 v[134:135], v[34:35], v[134:135] op_sel:[1,0]
	v_pk_mul_f32 v[136:137], v[34:35], v[136:137] op_sel:[1,0]
	v_pk_mul_f32 v[138:139], v[34:35], v[138:139] op_sel:[1,0]
	v_pk_mul_f32 v[140:141], v[34:35], v[140:141] op_sel:[1,0]
	v_pk_mul_f32 v[142:143], v[34:35], v[142:143] op_sel:[1,0]
	v_pk_mul_f32 v[144:145], v[34:35], v[144:145] op_sel:[1,0]
	v_pk_mul_f32 v[146:147], v[34:35], v[146:147] op_sel:[1,0]
	v_pk_mul_f32 v[148:149], v[34:35], v[148:149] op_sel:[1,0]
	v_pk_mul_f32 v[150:151], v[34:35], v[150:151] op_sel:[1,0]
	v_pk_mul_f32 v[152:153], v[34:35], v[152:153] op_sel:[1,0]
	v_pk_mul_f32 v[154:155], v[34:35], v[154:155] op_sel:[1,0]
	v_pk_mul_f32 v[156:157], v[34:35], v[156:157] op_sel:[1,0]
	v_pk_mul_f32 v[158:159], v[34:35], v[158:159] op_sel:[1,0]
label_9F34:
	v_pk_mul_f32 v[96:97], v[34:35], v[96:97] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[34:35], v[98:99] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[34:35], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[34:35], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[34:35], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[34:35], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[34:35], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[34:35], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[34:35], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115], v[34:35], v[114:115] op_sel_hi:[0,1]
	v_pk_mul_f32 v[116:117], v[34:35], v[116:117] op_sel_hi:[0,1]
	v_pk_mul_f32 v[118:119], v[34:35], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[120:121], v[34:35], v[120:121] op_sel_hi:[0,1]
	v_pk_mul_f32 v[122:123], v[34:35], v[122:123] op_sel_hi:[0,1]
	v_pk_mul_f32 v[124:125], v[34:35], v[124:125] op_sel_hi:[0,1]
	v_pk_mul_f32 v[126:127], v[34:35], v[126:127] op_sel_hi:[0,1]
	s_setpc_b64 vcc
label_9FB8:
	s_cmp_eq_u32 s88, 8
	s_cselect_b32 s69, 1, 0
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s69, s69, 0
	s_cmp_lt_u32 s7, 1
	s_cbranch_scc1 label_A02C
	s_cmp_eq_u32 s70, 0
	s_cbranch_scc1 label_A018
	v_mov_b64_e32 v[48:49], 0
	v_mov_b64_e32 v[50:51], 0
	v_mov_b64_e32 v[52:53], 0
	v_mov_b64_e32 v[54:55], 0
	v_mov_b64_e32 v[56:57], 0
	v_mov_b64_e32 v[58:59], 0
	v_mov_b64_e32 v[60:61], 0
	v_mov_b64_e32 v[62:63], 0
	v_mov_b64_e32 v[64:65], 0
	v_mov_b64_e32 v[66:67], 0
	v_mov_b64_e32 v[68:69], 0
	v_mov_b64_e32 v[70:71], 0
	v_mov_b64_e32 v[72:73], 0
	v_mov_b64_e32 v[74:75], 0
	v_mov_b64_e32 v[76:77], 0
	v_mov_b64_e32 v[78:79], 0
label_A018:
	s_cmp_eq_u32 s7, 1
	s_cbranch_scc1 label_BBC0
	s_cmp_eq_u32 s7, 2
	s_cbranch_scc1 label_CC74
	s_branch label_DD28
label_A02C:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_cbranch_vccz label_A2A0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_A2A0:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:41472
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:43552
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:41600
	ds_read_b128 a[60:63], v13 offset:41616
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:43680
	ds_read_b128 a[92:95], v13 offset:43696
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:41664
	ds_read_b128 a[68:71], v241 offset:41680
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:43744
	ds_read_b128 a[100:103], v241 offset:43760
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_cbranch_vccz label_A970
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_A970:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(11)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:58112
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:60192
	s_nop 1
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:58240
	ds_read_b128 a[60:63], v13 offset:58256
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:60320
	ds_read_b128 a[92:95], v13 offset:60336
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	s_nop 1
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:58304
	ds_read_b128 a[68:71], v241 offset:58320
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:60384
	ds_read_b128 a[100:103], v241 offset:60400
	ds_write_b64 v20, v[48:49]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_AEF4
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_AEF4:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_AF1C
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_AF1C:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_B040
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_B040:
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_cbranch_vccz label_B058
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_B058:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_B26C
	s_waitcnt vmcnt(9)
label_B26C:
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:8192
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:8192
	ds_read_b128 a[52:55], v240 offset:8208
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:10272
	ds_read_b128 a[84:87], v240 offset:10288
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:10272
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:8320
	ds_read_b128 a[60:63], v13 offset:8336
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:10400
	ds_read_b128 a[92:95], v13 offset:10416
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:8384
	ds_read_b128 a[68:71], v241 offset:8400
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:10464
	ds_read_b128 a[100:103], v241 offset:10480
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16
	ds_read_b128 a[108:111], v16 offset:16
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:2080
	ds_read_b128 a[116:119], v16 offset:2096
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_B5E8
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_B5E8:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_B610
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_B610:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_B734
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_B734:
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_cbranch_vccz label_B74C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_B74C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_B960
	s_waitcnt vmcnt(10)
label_B960:
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:24832
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:26912
	s_nop 1
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:24960
	ds_read_b128 a[60:63], v13 offset:24976
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:27040
	ds_read_b128 a[92:95], v13 offset:27056
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	s_nop 1
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:25024
	ds_read_b128 a[68:71], v241 offset:25040
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:27104
	ds_read_b128 a[100:103], v241 offset:27120
	ds_write_b64 v20, v[48:49]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	ds_write_b64 v20, v[50:51] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_13644
	s_branch label_A02C
label_BBC0:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_nop 0
	s_cbranch_vccz label_BD58
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_BD58:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_nop 0
	s_cbranch_vccz label_C170
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_C170:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(11)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_C4B0
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_C4B0:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_C4D0
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_C4D0:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_C58C
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_C58C:
	s_nop 0
	s_cbranch_vccz label_C5A0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_C5A0:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_C794
	s_waitcnt vmcnt(9)
label_C794:
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_C8EC
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_C8EC:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_C90C
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_C90C:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_C9C8
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_C9C8:
	s_nop 0
	s_cbranch_vccz label_C9DC
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_C9DC:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_CBD0
	s_waitcnt vmcnt(10)
label_CBD0:
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_14710
	s_branch label_BBC0
label_CC74:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_CE0C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_CE0C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(12)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_D224
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_D224:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(11)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_D564
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_D564:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_D584
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_D584:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_D630
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_D630:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_D654
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_D654:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_D848
	s_waitcnt vmcnt(9)
label_D848:
	s_waitcnt vmcnt(12)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_D9A0
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_D9A0:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_D9C0
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_D9C0:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_DA6C
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_DA6C:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_DA90
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_DA90:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_DC84
	s_waitcnt vmcnt(10)
label_DC84:
	s_waitcnt vmcnt(12)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_15038
	s_branch label_CC74
label_DD28:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_DEC0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_DEC0:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(12)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:4160
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:4288
	ds_read_b64_tr_b16 a[140:141], v24 offset:4160
	ds_read_b64_tr_b16 a[142:143], v24 offset:4288
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:4672
	ds_read_b64_tr_b16 a[146:147], v23 offset:4800
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:4672
	ds_read_b64_tr_b16 a[150:151], v24 offset:4800
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_E2D8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_E2D8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_waitcnt vmcnt(11)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_E618
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_E618:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:8320
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_E638
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_E638:
	ds_read_b64_tr_b16 a[138:139], v23 offset:8448
	ds_read_b64_tr_b16 a[140:141], v24 offset:8320
	ds_read_b64_tr_b16 a[142:143], v24 offset:8448
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:8832
	ds_read_b64_tr_b16 a[146:147], v23 offset:8960
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:8832
	ds_read_b64_tr_b16 a[150:151], v24 offset:8960
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_E6E4
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_E6E4:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_E708
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_E708:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_E8FC
	s_waitcnt vmcnt(9)
label_E8FC:
	s_waitcnt vmcnt(12)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_EA54
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_EA54:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:12480
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_EA74
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_EA74:
	ds_read_b64_tr_b16 a[138:139], v23 offset:12608
	ds_read_b64_tr_b16 a[140:141], v24 offset:12480
	ds_read_b64_tr_b16 a[142:143], v24 offset:12608
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:12992
	ds_read_b64_tr_b16 a[146:147], v23 offset:13120
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:12992
	ds_read_b64_tr_b16 a[150:151], v24 offset:13120
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_EB20
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_EB20:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_EB44
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_EB44:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_nop 1
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_nop 1
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_ED38
	s_waitcnt vmcnt(10)
label_ED38:
	s_waitcnt vmcnt(12)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_15960
	s_branch label_DD28
label_EDDC:
	s_cmp_eq_u32 s88, 8
	s_cselect_b32 s69, 1, 0
	s_cmp_eq_u32 s70, 0
	s_cselect_b32 s69, s69, 0
	s_cmp_lt_u32 s7, 2
	s_cbranch_scc1 label_EE48
	s_cmp_eq_u32 s70, 0
	s_cbranch_scc1 label_EE3C
	v_mov_b64_e32 v[48:49], 0
	v_mov_b64_e32 v[50:51], 0
	v_mov_b64_e32 v[52:53], 0
	v_mov_b64_e32 v[54:55], 0
	v_mov_b64_e32 v[56:57], 0
	v_mov_b64_e32 v[58:59], 0
	v_mov_b64_e32 v[60:61], 0
	v_mov_b64_e32 v[62:63], 0
	v_mov_b64_e32 v[64:65], 0
	v_mov_b64_e32 v[66:67], 0
	v_mov_b64_e32 v[68:69], 0
	v_mov_b64_e32 v[70:71], 0
	v_mov_b64_e32 v[72:73], 0
	v_mov_b64_e32 v[74:75], 0
	v_mov_b64_e32 v[76:77], 0
	v_mov_b64_e32 v[78:79], 0
label_EE3C:
	s_cmp_eq_u32 s7, 2
	s_cbranch_scc1 label_10ACC
	s_branch label_11C70
label_EE48:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[54:55], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_cbranch_vccz label_F0D8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_F0D8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:41472
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:43552
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:41600
	ds_read_b128 a[60:63], v13 offset:41616
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:43680
	ds_read_b128 a[92:95], v13 offset:43696
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:41664
	ds_read_b128 a[68:71], v241 offset:41680
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:43744
	ds_read_b128 a[100:103], v241 offset:43760
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[70:71], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_cbranch_vccz label_F7E4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_F7E4:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(11)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:58112
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:60192
	s_nop 1
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:58240
	ds_read_b128 a[60:63], v13 offset:58256
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:60320
	ds_read_b128 a[92:95], v13 offset:60336
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	s_nop 1
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:58304
	ds_read_b128 a[68:71], v241 offset:58320
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:60384
	ds_read_b128 a[100:103], v241 offset:60400
	ds_write_b64 v20, v[48:49]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_FD90
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_FD90:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_FDB8
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_FDB8:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[54:55], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_FEF0
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_FEF0:
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_cbranch_vccz label_FF08
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_FF08:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_10130
	s_waitcnt vmcnt(9)
label_10130:
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:8192
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:8192
	ds_read_b128 a[52:55], v240 offset:8208
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:10272
	ds_read_b128 a[84:87], v240 offset:10288
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:10272
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:8320
	ds_read_b128 a[60:63], v13 offset:8336
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:10400
	ds_read_b128 a[92:95], v13 offset:10416
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:8384
	ds_read_b128 a[68:71], v241 offset:8400
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:10464
	ds_read_b128 a[100:103], v241 offset:10480
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16
	ds_read_b128 a[108:111], v16 offset:16
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:2080
	ds_read_b128 a[116:119], v16 offset:2096
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_104C0
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_104C0:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_104E8
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_104E8:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[70:71], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_10620
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_10620:
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_cbranch_vccz label_10638
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_10638:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_10860
	s_waitcnt vmcnt(10)
label_10860:
	s_waitcnt vmcnt(12)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:24832
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:24832
	ds_read_b128 a[52:55], v240 offset:24848
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:26912
	ds_read_b128 a[84:87], v240 offset:26928
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:26912
	s_nop 1
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:24960
	ds_read_b128 a[60:63], v13 offset:24976
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:27040
	ds_read_b128 a[92:95], v13 offset:27056
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	s_nop 1
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:25024
	ds_read_b128 a[68:71], v241 offset:25040
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:27104
	ds_read_b128 a[100:103], v241 offset:27120
	ds_write_b64 v20, v[48:49]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:4160
	ds_read_b128 a[108:111], v16 offset:4176
	ds_write_b64 v20, v[50:51] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:6240
	ds_read_b128 a[116:119], v16 offset:6256
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_16288
	s_branch label_EE48
label_10ACC:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_10C80
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_10C80:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_110D4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_110D4:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(11)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_1143C
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_1143C:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_1145C
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_1145C:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_1151C
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_1151C:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_11540
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_11540:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_11748
	s_waitcnt vmcnt(9)
label_11748:
	s_waitcnt vmcnt(12)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_118B4
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_118B4:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_118D4
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_118D4:
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_11994
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_11994:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_119B8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_119B8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_11BC0
	s_waitcnt vmcnt(10)
label_11BC0:
	s_waitcnt vmcnt(12)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_17408
	s_branch label_10ACC
label_11C70:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v39, 0, v39
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v39 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s56
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v41, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	buffer_load_dword v244, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v39 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v39 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v39 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_11E24
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_11E24:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v242, 0, v242
	v_mad_u64_u32 v[2:3], vcc, v242, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s60
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(12)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v40, 0, v40
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v40 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s57
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	buffer_load_dword v42, v38, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:4160
	buffer_load_dword v245, v44, s[24:27], 0 offen
	ds_read_b64_tr_b16 a[138:139], v23 offset:4288
	ds_read_b64 v[68:69], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24 offset:4160
	ds_read_b64_tr_b16 a[142:143], v24 offset:4288
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v40 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:4672
	ds_read_b64_tr_b16 a[146:147], v23 offset:4800
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:4672
	ds_read_b64_tr_b16 a[150:151], v24 offset:4800
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v40 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v40 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_12278
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_12278:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v243, 0, v243
	v_mad_u64_u32 v[2:3], vcc, v243, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s61
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(11)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v41, 0, v41
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_125E0
	buffer_load_dword v39, v38, s[24:27], 0 offen
label_125E0:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:8320
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_12600
	buffer_load_dword v242, v44, s[24:27], 0 offen
label_12600:
	ds_read_b64_tr_b16 a[138:139], v23 offset:8448
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24 offset:8320
	ds_read_b64_tr_b16 a[142:143], v24 offset:8448
	ds_read_b64 v[54:55], v19 offset:512
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:8832
	ds_read_b64_tr_b16 a[146:147], v23 offset:8960
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:8832
	ds_read_b64_tr_b16 a[150:151], v24 offset:8960
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_126C0
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_126C0:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_126E4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_126E4:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_128EC
	s_waitcnt vmcnt(9)
label_128EC:
	s_waitcnt vmcnt(12)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:8192
	ds_read_b128 v[84:87], v10 offset:8208
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:10272
	ds_read_b128 v[92:95], v10 offset:10288
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:8192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:10272
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_max_i32_e32 v42, 0, v42
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_12A58
	buffer_load_dword v40, v38, s[24:27], 0 offen
label_12A58:
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:12480
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_12A78
	buffer_load_dword v243, v44, s[24:27], 0 offen
label_12A78:
	ds_read_b64_tr_b16 a[138:139], v23 offset:12608
	ds_read_b64 v[68:69], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24 offset:12480
	ds_read_b64_tr_b16 a[142:143], v24 offset:12608
	ds_read_b64 v[70:71], v19 offset:512
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	ds_read_b64_tr_b16 a[144:145], v23 offset:12992
	ds_read_b64_tr_b16 a[146:147], v23 offset:13120
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	ds_read_b64_tr_b16 a[148:149], v24 offset:12992
	ds_read_b64_tr_b16 a[150:151], v24 offset:13120
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_12B38
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
label_12B38:
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	s_nop 0
	s_cbranch_vccz label_12B5C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_12B5C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	s_waitcnt vmcnt(17)
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_cmp_eq_u32 s69, 0
	s_cbranch_scc1 label_12D64
	s_waitcnt vmcnt(10)
label_12D64:
	s_waitcnt vmcnt(12)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:24832
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:26912
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_cmp_eq_u32 s69, 1
	s_cbranch_scc1 label_17DE4
	s_branch label_11C70
label_12E14:
	s_cmp_ge_u32 s7, 2
	s_cbranch_scc1 label_13230
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v33, 1.0
	v_mov_b64_e32 v[48:49], 0
	v_mov_b64_e32 v[50:51], 0
	v_mov_b64_e32 v[52:53], 0
	v_mov_b64_e32 v[54:55], 0
	v_mov_b64_e32 v[56:57], 0
	v_mov_b64_e32 v[58:59], 0
	v_mov_b64_e32 v[60:61], 0
	v_mov_b64_e32 v[62:63], 0
	v_mov_b64_e32 v[64:65], 0
	v_mov_b64_e32 v[66:67], 0
	v_mov_b64_e32 v[68:69], 0
	v_mov_b64_e32 v[70:71], 0
	v_mov_b64_e32 v[72:73], 0
	v_mov_b64_e32 v[74:75], 0
	v_mov_b64_e32 v[76:77], 0
	v_mov_b64_e32 v[78:79], 0
	s_nop 1
	buffer_load_dword v39, v38, s[24:27], 0 offen
	buffer_load_dword v242, v44, s[24:27], 0 offen
	v_max_i32_e32 v41, 0, v41
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	s_nop 1
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	s_nop 1
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_nop 1
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v81, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37]
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v83, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v34, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v85, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:32
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_cvt_scalef32_pk_bf16_fp8 v34, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v87, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:48
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	s_nop 1
	v_cvt_scalef32_pk_bf16_fp8 v34, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v89, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v34, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v91, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v34, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v34, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v95, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[34:37] offset:14640
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_u8 v236, v238 offset:24832
	ds_read_u8 v237, v238 offset:26912
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_max_i32_e32 v42, 0, v42
	s_nop 1
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	buffer_load_dword v40, v38, s[24:27], 0 offen
	buffer_load_dword v243, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_branch label_1F1C
label_13230:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v33, 1.0
	v_mov_b64_e32 v[48:49], 0
	v_mov_b64_e32 v[50:51], 0
	v_mov_b64_e32 v[52:53], 0
	v_mov_b64_e32 v[54:55], 0
	v_mov_b64_e32 v[56:57], 0
	v_mov_b64_e32 v[58:59], 0
	v_mov_b64_e32 v[60:61], 0
	v_mov_b64_e32 v[62:63], 0
	v_mov_b64_e32 v[64:65], 0
	v_mov_b64_e32 v[66:67], 0
	v_mov_b64_e32 v[68:69], 0
	v_mov_b64_e32 v[70:71], 0
	v_mov_b64_e32 v[72:73], 0
	v_mov_b64_e32 v[74:75], 0
	v_mov_b64_e32 v[76:77], 0
	v_mov_b64_e32 v[78:79], 0
	s_nop 1
	buffer_load_dword v39, v38, s[24:27], 0 offen
	buffer_load_dword v242, v44, s[24:27], 0 offen
	v_max_i32_e32 v41, 0, v41
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	s_nop 1
	v_mov_b32_dpp v26, v41 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	s_nop 1
	s_mov_b32 m0, s58
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	s_nop 1
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	v_mov_b32_dpp v26, v41 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v81, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37]
	v_mov_b32_dpp v26, v41 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v41 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	v_cvt_scalef32_pk_bf16_fp8 v34, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v83, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v34, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v85, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v34, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v87, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:48
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	s_nop 1
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	v_cvt_scalef32_pk_bf16_fp8 v34, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v89, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:14592
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_cvt_scalef32_pk_bf16_fp8 v34, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v91, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v34, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[34:37] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v34, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v35, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v36, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v37, v95, v33 op_sel:[1,0,0]
	ds_write_b128 v11, v[34:37] offset:14640
	v_max_i32_e32 v244, 0, v244
	v_mad_u64_u32 v[2:3], vcc, v244, s86, v[246:247]
	s_mov_b32 m0, s62
	global_load_lds_dwordx4 v[2:3], off
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_u8 v236, v238 offset:24832
	ds_read_u8 v237, v238 offset:26912
	ds_read_b128 v[80:83], v10 offset:24832
	ds_read_b128 v[84:87], v10 offset:24848
	ds_read_b128 v[88:91], v10 offset:26912
	ds_read_b128 v[92:95], v10 offset:26928
	v_max_i32_e32 v42, 0, v42
	s_nop 1
	v_mov_b32_dpp v26, v42 row_newbcast:0 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:2 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[2:3], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:4 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:6 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[4:5], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:8 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:10 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[6:7], vcc, v26, s73, v[14:15]
	v_mov_b32_dpp v26, v42 row_newbcast:12 row_mask:0xf bank_mask:0xf
	v_mov_b32_dpp v27, v42 row_newbcast:14 row_mask:0xf bank_mask:0xf
	v_cndmask_b32_e64 v26, v26, v27, s[92:93]
	v_mad_u64_u32 v[8:9], vcc, v26, s73, v[14:15]
	buffer_load_dword v40, v38, s[24:27], 0 offen
	buffer_load_dword v243, v44, s[24:27], 0 offen
	v_add_u32_e32 v38, s94, v38
	v_add_u32_e32 v44, s94, v44
	s_mov_b32 m0, s59
	global_load_lds_dwordx4 v[2:3], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[4:5], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[6:7], off
	s_add_u32 m0, m0, 0x410
	global_load_lds_dwordx4 v[8:9], off
	s_add_u32 m0, m0, 0x410
	v_max_i32_e32 v245, 0, v245
	v_mad_u64_u32 v[2:3], vcc, v245, s86, v[246:247]
	s_mov_b32 m0, s63
	global_load_lds_dwordx4 v[2:3], off
	s_branch label_1F1C
label_13644:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_nop 2
	s_cbranch_vccz label_137DC
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_137DC:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:41472
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:43552
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:41600
	ds_read_b128 a[60:63], v13 offset:41616
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:43680
	ds_read_b128 a[92:95], v13 offset:43696
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:41664
	ds_read_b128 a[68:71], v241 offset:41680
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:43744
	ds_read_b128 a[100:103], v241 offset:43760
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_nop 2
	s_cbranch_vccz label_13DB4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_13DB4:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:58112
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:60192
	s_nop 1
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:58240
	ds_read_b128 a[60:63], v13 offset:58256
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:60320
	ds_read_b128 a[92:95], v13 offset:60336
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	s_nop 1
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:58304
	ds_read_b128 a[68:71], v241 offset:58320
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:60384
	ds_read_b128 a[100:103], v241 offset:60400
	ds_write_b64 v20, v[48:49]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_nop 2
	s_cbranch_vccz label_1438C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_1438C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	s_nop 1
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	s_nop 1
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	s_nop 1
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
label_14710:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_147C8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_147C8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_14AE4
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_14AE4:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_14E00
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_14E00:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
label_15038:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_150F0
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_150F0:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(5)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_1540C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_1540C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_15728
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_15728:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
label_15960:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_15A18
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_15A18:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(5)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:4160
	ds_read_b64_tr_b16 a[138:139], v23 offset:4288
	ds_read_b64_tr_b16 a[140:141], v24 offset:4160
	ds_read_b64_tr_b16 a[142:143], v24 offset:4288
	ds_read_b64_tr_b16 a[144:145], v23 offset:4672
	ds_read_b64_tr_b16 a[146:147], v23 offset:4800
	ds_read_b64_tr_b16 a[148:149], v24 offset:4672
	ds_read_b64_tr_b16 a[150:151], v24 offset:4800
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_15D34
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_15D34:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	s_nop 1
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	s_nop 1
	s_nop 1
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:8320
	ds_read_b64_tr_b16 a[138:139], v23 offset:8448
	ds_read_b64_tr_b16 a[140:141], v24 offset:8320
	ds_read_b64_tr_b16 a[142:143], v24 offset:8448
	ds_read_b64_tr_b16 a[144:145], v23 offset:8832
	ds_read_b64_tr_b16 a[146:147], v23 offset:8960
	ds_read_b64_tr_b16 a[148:149], v24 offset:8832
	ds_read_b64_tr_b16 a[150:151], v24 offset:8960
	v_cmp_neq_f32_e32 vcc, 1.0, v34
	s_nop 4
	s_cbranch_vccz label_16050
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9F34
label_16050:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	s_nop 1
	ds_write_b128 v11, v[26:29] offset:14640
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_waitcnt vmcnt(0)
	s_barrier
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
label_16288:
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[54:55], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_nop 2
	s_cbranch_vccz label_1643C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_1643C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(5)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:41472
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:41472
	ds_read_b128 a[52:55], v240 offset:41488
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:43552
	ds_read_b128 a[84:87], v240 offset:43568
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:43552
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:41600
	ds_read_b128 a[60:63], v13 offset:41616
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:43680
	ds_read_b128 a[92:95], v13 offset:43696
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:41664
	ds_read_b128 a[68:71], v241 offset:41680
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:43744
	ds_read_b128 a[100:103], v241 offset:43760
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:8320
	ds_read_b128 a[108:111], v16 offset:8336
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:10400
	ds_read_b128 a[116:119], v16 offset:10416
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[48:55], a[8:15], v[48:51], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[56:63], a[16:23], v[48:51], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], a[64:71], a[24:31], v[48:51], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[66:67], v19
	v_mfma_f32_16x16x32_bf16 v[48:51], a[104:107], a[32:35], v[48:51]
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[48:51], a[108:111], a[36:39], v[48:51]
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[80:87], a[8:15], v[52:55], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[70:71], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[88:95], a[16:23], v[52:55], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], a[96:103], a[24:31], v[52:55], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[52:55], a[112:115], a[32:35], v[52:55]
	v_mfma_f32_16x16x32_bf16 v[52:55], a[116:119], a[36:39], v[52:55]
	s_nop 2
	s_cbranch_vccz label_16A50
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_16A50:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v48, v49, v48
	v_max3_f32 v30, v50, v51, v30
	v_max3_f32 v30, v52, v53, v30
	v_max3_f32 v30, v54, v55, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	ds_read_b128 v[228:231], v1 offset:58112
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	ds_read_b128 a[48:51], v240 offset:58112
	ds_read_b128 a[52:55], v240 offset:58128
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	ds_read_b128 a[80:83], v240 offset:60192
	ds_read_b128 a[84:87], v240 offset:60208
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	ds_read_b128 v[232:235], v1 offset:60192
	s_nop 1
	v_fma_f32 v48, v48, s5, -v32
	v_fma_f32 v49, v49, s5, -v32
	v_fma_f32 v50, v50, s5, -v32
	v_fma_f32 v51, v51, s5, -v32
	v_fma_f32 v52, v52, s5, -v32
	v_fma_f32 v53, v53, s5, -v32
	v_fma_f32 v54, v54, s5, -v32
	v_fma_f32 v55, v55, s5, -v32
	s_nop 1
	ds_read_b128 a[56:59], v13 offset:58240
	ds_read_b128 a[60:63], v13 offset:58256
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v50, v50
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v52, v52
	v_exp_f32_e32 v53, v53
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	ds_read_b128 a[88:91], v13 offset:60320
	ds_read_b128 a[92:95], v13 offset:60336
	v_add_f32_e32 v26, v49, v48
	v_add_f32_e32 v26, v50, v26
	v_add_f32_e32 v26, v51, v26
	v_add_f32_e32 v26, v52, v26
	s_nop 1
	v_add_f32_e32 v26, v53, v26
	v_add_f32_e32 v26, v54, v26
	v_add_f32_e32 v26, v55, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_cvt_pk_bf16_f32 v49, v50, v51
	v_cvt_pk_bf16_f32 v50, v52, v53
	v_cvt_pk_bf16_f32 v51, v54, v55
	s_nop 1
	ds_read_b128 a[64:67], v241 offset:58304
	ds_read_b128 a[68:71], v241 offset:58320
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_read_b128 a[96:99], v241 offset:60384
	ds_read_b128 a[100:103], v241 offset:60400
	ds_write_b64 v20, v[48:49]
	s_nop 1
	s_nop 1
	ds_read_b128 a[104:107], v16 offset:12480
	ds_read_b128 a[108:111], v16 offset:12496
	ds_write_b64 v20, v[50:51] offset:2048
	s_nop 1
	s_nop 1
	ds_read_b128 a[112:115], v16 offset:14560
	ds_read_b128 a[116:119], v16 offset:14576
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(2)
	s_barrier
	v_perm_b32 v32, v228, v229, v12
	v_perm_b32 v228, v229, v228, v12
	v_mov_b32_e32 v229, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[80:87], a[0:7], 0, v228, v224 op_sel_hi:[0,0,0]
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	v_bfe_u32 v230, v230, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[48:55], a[8:15], v[64:67], v229, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	v_and_b32_e32 v231, v47, v231
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[56:63], a[16:23], v[64:67], v230, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], a[64:71], a[24:31], v[64:67], v231, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64 v[50:51], v19
	v_mfma_f32_16x16x32_bf16 v[64:67], a[104:107], a[32:35], v[64:67]
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	v_mfma_f32_16x16x32_bf16 v[64:67], a[108:111], a[36:39], v[64:67]
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	v_perm_b32 v32, v232, v233, v12
	v_perm_b32 v232, v233, v232, v12
	v_mov_b32_e32 v233, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[88:95], a[0:7], 0, v232, v224 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	v_bfe_u32 v234, v234, v25, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[80:87], a[8:15], v[68:71], v233, v225 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64 v[54:55], v19 offset:512
	v_and_b32_e32 v235, v47, v235
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[88:95], a[16:23], v[68:71], v234, v226 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], a[96:103], a[24:31], v[68:71], v235, v227 op_sel_hi:[0,0,0]
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	v_mfma_f32_16x16x32_bf16 v[68:71], a[112:115], a[32:35], v[68:71]
	v_mfma_f32_16x16x32_bf16 v[68:71], a[116:119], a[36:39], v[68:71]
	s_nop 2
	s_cbranch_vccz label_17064
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_17064:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	ds_write_b128 v11, v[26:29] offset:14640
	v_max3_f32 v30, v64, v65, v64
	v_max3_f32 v30, v66, v67, v30
	v_max3_f32 v30, v68, v69, v30
	v_max3_f32 v30, v70, v71, v30
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	s_nop 1
	v_permlane16_swap_b32_e32 v26, v27
	v_mov_b32_e32 v29, v26
	v_mov_b32_e32 v28, v27
	s_nop 1
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_permlane32_swap_b32_e32 v26, v27
	v_permlane32_swap_b32_e32 v28, v29
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	v_max3_f32 v30, v26, v27, v30
	v_max3_f32 v30, v28, v29, v30
	v_cmp_eq_u32_e64 s[64:65], s74, v45
	v_max_f32_e32 v31, v30, v45
	s_nop 1
	s_nop 1
	v_sub_f32_e32 v27, v31, v45
	v_mul_f32_e32 v27, s5, v27
	v_cmp_lt_f32_e32 vcc, 0x41000000, v27
	v_sub_f32_e32 v33, v45, v31
	v_cndmask_b32_e64 v33, v33, 0, s[64:65]
	v_cndmask_b32_e32 v33, 0, v33, vcc
	v_cndmask_b32_e32 v45, v45, v31, vcc
	s_nop 1
	v_mul_f32_e32 v32, s5, v45
	v_mul_f32_e32 v33, s5, v33
	v_exp_f32_e32 v33, v33
	s_nop 1
	v_fma_f32 v64, v64, s5, -v32
	v_fma_f32 v65, v65, s5, -v32
	v_fma_f32 v66, v66, s5, -v32
	v_fma_f32 v67, v67, s5, -v32
	v_fma_f32 v68, v68, s5, -v32
	v_fma_f32 v69, v69, s5, -v32
	v_fma_f32 v70, v70, s5, -v32
	v_fma_f32 v71, v71, s5, -v32
	s_nop 1
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v66, v66
	s_nop 1
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mul_f32_e32 v46, v33, v46
	s_nop 1
	v_add_f32_e32 v26, v65, v64
	v_add_f32_e32 v26, v66, v26
	v_add_f32_e32 v26, v67, v26
	v_add_f32_e32 v26, v68, v26
	s_nop 1
	v_add_f32_e32 v26, v69, v26
	v_add_f32_e32 v26, v70, v26
	v_add_f32_e32 v26, v71, v26
	v_add_f32_e32 v46, v26, v46
	v_cvt_pk_bf16_f32 v64, v64, v65
	v_cvt_pk_bf16_f32 v65, v66, v67
	v_cvt_pk_bf16_f32 v66, v68, v69
	v_cvt_pk_bf16_f32 v67, v70, v71
	s_nop 1
	ds_write_b32 v17, v33 offset:4096
	s_nop 1
	s_nop 1
	ds_write_b64 v20, v[64:65]
	s_nop 1
	s_nop 1
	ds_write_b64 v20, v[66:67] offset:2048
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
label_17408:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_nop 4
	s_cbranch_vccz label_174DC
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_174DC:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(5)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[68:69], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[70:71], v19 offset:512
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_nop 4
	s_cbranch_vccz label_17834
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_17834:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v21 offset:128
	ds_read_b64_tr_b16 a[138:139], v21 offset:3776
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v22 offset:128
	ds_read_b64_tr_b16 a[142:143], v22 offset:3776
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64_tr_b16 a[144:145], v21 offset:192
	ds_read_b64_tr_b16 a[146:147], v21 offset:3840
	ds_read_b64_tr_b16 a[148:149], v22 offset:192
	ds_read_b64_tr_b16 a[150:151], v22 offset:3840
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_nop 4
	s_cbranch_vccz label_17B8C
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_17B8C:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
label_17DE4:
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23
	ds_read_b64_tr_b16 a[138:139], v23 offset:128
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24
	ds_read_b64_tr_b16 a[142:143], v24 offset:128
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64_tr_b16 a[144:145], v23 offset:512
	ds_read_b64_tr_b16 a[146:147], v23 offset:640
	ds_read_b64_tr_b16 a[148:149], v24 offset:512
	ds_read_b64_tr_b16 a[150:151], v24 offset:640
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_nop 4
	s_cbranch_vccz label_17EB8
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_17EB8:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(5)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	ds_read_b128 v[80:83], v10 offset:41472
	ds_read_b128 v[84:87], v10 offset:41488
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	ds_read_b128 v[88:91], v10 offset:43552
	ds_read_b128 v[92:95], v10 offset:43568
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:41472
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:43552
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[64:65], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[66:67], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:4160
	ds_read_b64_tr_b16 a[138:139], v23 offset:4288
	ds_read_b64 v[68:69], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24 offset:4160
	ds_read_b64_tr_b16 a[142:143], v24 offset:4288
	ds_read_b64 v[70:71], v19 offset:512
	ds_read_b64_tr_b16 a[144:145], v23 offset:4672
	ds_read_b64_tr_b16 a[146:147], v23 offset:4800
	ds_read_b64_tr_b16 a[148:149], v24 offset:4672
	ds_read_b64_tr_b16 a[150:151], v24 offset:4800
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_nop 4
	s_cbranch_vccz label_18210
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_18210:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[64:67], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[64:67], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[64:67], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[64:67], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[64:67], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[64:67], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[64:67], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[64:67], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[68:71], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[68:71], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[68:71], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[68:71], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[68:71], v[144:147]
	s_waitcnt vmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[68:71], v[148:151]
	ds_read_b128 v[80:83], v10 offset:58112
	ds_read_b128 v[84:87], v10 offset:58128
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[68:71], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[68:71], v[156:159]
	ds_read_b128 v[88:91], v10 offset:60192
	ds_read_b128 v[92:95], v10 offset:60208
	s_nop 1
	s_nop 1
	ds_read_u8 v236, v238 offset:58112
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	ds_read_u8 v237, v238 offset:60192
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v34, v43 offset:4096
	ds_read_b32 v35, v43 offset:4352
	ds_read_b64_tr_b16 a[120:121], v21
	ds_read_b64_tr_b16 a[122:123], v21 offset:3648
	ds_read_b64_tr_b16 a[124:125], v22
	ds_read_b64 v[48:49], v18
	ds_read_b64_tr_b16 a[126:127], v22 offset:3648
	ds_read_b64_tr_b16 a[128:129], v21 offset:64
	ds_read_b64 v[50:51], v19
	ds_read_b64_tr_b16 a[130:131], v21 offset:3712
	ds_read_b64_tr_b16 a[132:133], v22 offset:64
	ds_read_b64_tr_b16 a[134:135], v22 offset:3712
	ds_read_b64_tr_b16 a[136:137], v23 offset:8320
	ds_read_b64_tr_b16 a[138:139], v23 offset:8448
	ds_read_b64 v[52:53], v18 offset:512
	ds_read_b64_tr_b16 a[140:141], v24 offset:8320
	ds_read_b64_tr_b16 a[142:143], v24 offset:8448
	ds_read_b64 v[54:55], v19 offset:512
	ds_read_b64_tr_b16 a[144:145], v23 offset:8832
	ds_read_b64_tr_b16 a[146:147], v23 offset:8960
	ds_read_b64_tr_b16 a[148:149], v24 offset:8832
	ds_read_b64_tr_b16 a[150:151], v24 offset:8960
	v_min_f32_e32 v26, v34, v35
	v_cmp_neq_f32_e32 vcc, 1.0, v26
	s_nop 4
	s_cbranch_vccz label_18568
	s_getpc_b64 vcc
	s_add_i32 vcc_lo, vcc_lo, 8
	s_branch label_9EB4
label_18568:
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[96:99], a[120:123], v[48:51], v[96:99]
	v_lshlrev_b32_e32 v33, 23, v236
	v_cmp_eq_u32_e64 vcc, v236, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v80, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v80, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v81, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v81, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[100:103], a[124:127], v[48:51], v[100:103]
	ds_write_b128 v11, v[26:29]
	v_cvt_scalef32_pk_bf16_fp8 v26, v82, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v82, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v83, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v83, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[104:107], a[128:131], v[48:51], v[104:107]
	ds_write_b128 v11, v[26:29] offset:16
	v_cvt_scalef32_pk_bf16_fp8 v26, v84, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v84, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v85, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v85, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[108:111], a[132:135], v[48:51], v[108:111]
	ds_write_b128 v11, v[26:29] offset:32
	v_cvt_scalef32_pk_bf16_fp8 v26, v86, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v86, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v87, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v87, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[112:115], a[136:139], v[48:51], v[112:115]
	ds_write_b128 v11, v[26:29] offset:48
	v_lshlrev_b32_e32 v33, 23, v237
	v_cmp_eq_u32_e64 vcc, v237, 0
	v_cndmask_b32_e64 v33, v33, 0, vcc
	v_cvt_scalef32_pk_bf16_fp8 v26, v88, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v88, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v89, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v89, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[116:119], a[140:143], v[48:51], v[116:119]
	ds_write_b128 v11, v[26:29] offset:14592
	v_cvt_scalef32_pk_bf16_fp8 v26, v90, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v90, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v91, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v91, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[120:123], a[144:147], v[48:51], v[120:123]
	ds_write_b128 v11, v[26:29] offset:14608
	v_cvt_scalef32_pk_bf16_fp8 v26, v92, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v92, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v93, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v93, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[124:127], a[148:151], v[48:51], v[124:127]
	ds_write_b128 v11, v[26:29] offset:14624
	v_cvt_scalef32_pk_bf16_fp8 v26, v94, v33
	v_cvt_scalef32_pk_bf16_fp8 v27, v94, v33 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v28, v95, v33
	v_cvt_scalef32_pk_bf16_fp8 v29, v95, v33 op_sel:[1,0,0]
	v_mfma_f32_16x16x32_bf16 v[128:131], a[120:123], v[52:55], v[128:131]
	ds_write_b128 v11, v[26:29] offset:14640
	v_mfma_f32_16x16x32_bf16 v[132:135], a[124:127], v[52:55], v[132:135]
	v_mfma_f32_16x16x32_bf16 v[136:139], a[128:131], v[52:55], v[136:139]
	v_mfma_f32_16x16x32_bf16 v[140:143], a[132:135], v[52:55], v[140:143]
	v_mfma_f32_16x16x32_bf16 v[144:147], a[136:139], v[52:55], v[144:147]
	s_waitcnt vmcnt(0)
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[148:151], a[140:143], v[52:55], v[148:151]
	v_mfma_f32_16x16x32_bf16 v[152:155], a[144:147], v[52:55], v[152:155]
	v_mfma_f32_16x16x32_bf16 v[156:159], a[148:151], v[52:55], v[156:159]
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_nop 1
	s_addk_i32 s87, 0x1
	s_cmp_lt_i32 s87, s88
	s_cbranch_scc0 label_3D68
	v_permlane32_swap_b32_e32 v26, v26
	v_permlane32_swap_b32_e32 v26, v26
	s_branch label_3D68
.rodata
.p2align 6
.globl _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE.kd
.type _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE.kd,@object
.size _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE.kd, 64
_ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE.kd:
.byte 0, 128, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
.quad _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE - _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE.kd
.byte 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 0, 63, 3, 12, 0, 132, 3, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_write
        .address_space:  global
        .name:           O
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         8
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_write
        .address_space:  global
        .name:           Q
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         24
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           K
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         40
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           V
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         56
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           BT
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         72
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           CL
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         88
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           KQ
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         104
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           sclg
        .offset:         112
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         116
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         120
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         124
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           mblk
        .offset:         128
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         132
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         136
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         140
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           batch
        .offset:         144
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         148
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         152
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         156
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Qs
        .offset:         160
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         164
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         168
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         172
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Bs
        .offset:         176
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         180
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         184
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         188
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           Cs
        .offset:         192
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         196
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         200
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         204
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           QT
        .offset:         208
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         216
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           ST
        .offset:         224
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         232
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_write
        .address_space:  global
        .name:           RP
        .offset:         240
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         248
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           DQ
        .offset:         256
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         264
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_only
        .address_space:  global
        .name:           DK
        .offset:         272
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         280
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           out16
        .offset:         288
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         292
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         296
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         300
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .actual_access:  read_write
        .address_space:  global
        .name:           VALID_SPLIT
        .offset:         304
        .size:           8
        .value_kind:     global_buffer
      - .name:           pad
        .offset:         312
        .size:           8
        .value_kind:     by_value
        .value_type:     i32
      - .name:           use_valid_split
        .offset:         320
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         324
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         328
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
      - .name:           pad
        .offset:         332
        .size:           4
        .value_kind:     by_value
        .value_type:     i32
    .group_segment_fixed_size: 163840
    .kernarg_segment_align: 4
    .kernarg_segment_size: 336
    .max_flat_workgroup_size: 256
    .name:           _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     96
    .symbol:         _ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE.kd
    .vgpr_count:     512
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
