	.text
	.globl	flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32
	.p2align	8
	.type	flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32 @function

flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32:

	s_load_dwordx4 s[4:7], s[0:1], 0x58
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dwordx4 s[8:11], s[0:1], 0x0
	s_load_dwordx4 s[12:15], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s33, s6, s4
	s_lshl_b32 s0, s3, 7
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v2, 0x7c, v1
	s_ashr_i32 s16, s5, 31
	s_lshr_b32 s1, s16, 28
	s_add_i32 s1, s5, s1
	s_ashr_i32 s1, s1, 4
	s_ashr_i32 s17, s4, 31
	s_lshr_b32 s7, s17, 27
	s_add_i32 s7, s4, s7
	s_ashr_i32 s18, s7, 5
	s_lshl_b32 s7, s2, 4
	s_lshl_b32 s19, s18, 9
	s_mul_i32 s7, s7, s19
	s_ashr_i32 s20, s7, 31
	s_add_u32 s36, s10, s7
	s_addc_u32 s37, s11, s20
	s_add_i32 s1, s1, -1
	s_mul_i32 s1, s19, s1
	s_add_i32 s38, s19, s1
	v_lshrrev_b32_e32 v1, 6, v0
	s_nop 0
	v_readfirstlane_b32 s10, v1
	s_mov_b32 s7, 0
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	s_mul_i32 s1, s10, s19
	v_lshl_add_u32 v4, v3, 4, s1
	s_lshl_b32 s1, s18, 11
	v_add_u32_e32 v5, s1, v4
	v_add_u32_e32 v6, s1, v5
	v_add_u32_e32 v7, s1, v6
	v_lshrrev_b32_e32 v8, 2, v0
	s_lshl_b32 s1, s2, 8
	s_mul_i32 s39, s10, 0x110
	s_lshr_b32 s10, s17, 25
	s_add_i32 s10, s4, s10
	s_ashr_i32 s40, s10, 7
	v_lshlrev_b32_e32 v9, 4, v0
	v_and_b32_e32 v10, 48, v0
	v_lshrrev_b32_e32 v3, 3, v0
	v_and_b32_e32 v3, 4, v3
	v_or3_b32 v3, v3, v1, s0
	v_mad_u64_u32 v[2:3], s[10:11], s4, v3, v[2:3]
	s_lshl_b32 s4, s4, 3
	v_add_u32_e32 v3, s4, v2
	v_add_u32_e32 v11, s4, v3
	v_add_u32_e32 v12, s4, v11
	v_add_u32_e32 v13, s4, v12
	v_add_u32_e32 v14, s4, v13
	v_add_u32_e32 v15, s4, v14
	v_add_u32_e32 v16, s4, v15
	v_add_u32_e32 v17, s4, v16
	v_add_u32_e32 v18, s4, v17
	v_add_u32_e32 v19, s4, v18
	v_add_u32_e32 v20, s4, v19
	v_add_u32_e32 v21, s4, v20
	v_add_u32_e32 v22, s4, v21
	v_add_u32_e32 v23, s4, v22
	v_add_u32_e32 v24, s4, v23
	s_movk_i32 s4, 0x120
	v_mul_u32_u24_e32 v1, 0x120, v1
	s_nop 0
	v_readfirstlane_b32 s10, v1
	v_and_b32_e32 v1, 3, v0
	v_bfe_i32 v25, v0, 2, 1
	s_movk_i32 s11, 0x470
	v_and_b32_e32 v25, 0x470, v25
	s_movk_i32 s41, 0x80
	v_and_b32_e32 v26, 0x80, v9
	v_mad_u32_u24 v10, v1, s4, v10
	v_add3_u32 v10, v10, v26, v25
	v_bfe_u32 v25, v0, 3, 1
	v_lshrrev_b32_e32 v0, 1, v0
	v_and_or_b32 v0, v0, 2, v25
	v_lshl_or_b32 v0, v0, 2, v1
	s_lshl_b32 s3, s3, 9
	v_lshl_or_b32 v1, v0, 2, s3
	s_ashr_i32 s3, s33, 31
	s_lshr_b32 s3, s3, 25
	s_add_i32 s3, s33, s3
	s_ashr_i32 s3, s3, 7
	s_lshl_b32 s4, s3, 2
	s_lshl_b32 s42, s6, 2
	s_lshl_b32 s6, s2, 1
	s_lshl_b64 s[2:3], s[6:7], 2
	s_add_u32 s6, s14, s2
	s_addc_u32 s14, s15, s3
	s_lshr_b32 s2, s16, 25
	s_add_i32 s2, s5, s2
	s_ashr_i32 s2, s2, 7
	s_lshl_b32 s15, s2, 2
	s_mov_b64 s[2:3], src_shared_base
	s_mov_b32 s2, s7
	s_mov_b32 s7, 0x20000
	s_movk_i32 s43, 0x800
	s_movk_i32 s44, 0x440
	s_movk_i32 s45, 0x46f0
	;;#ASMSTART
	s_mov_b32 s16, s8 
s_mov_b32 s17, s9 
s_mov_b32 s18, s33 
s_mov_b32 s19, s7 
s_mov_b32 s20, s36 
s_mov_b32 s21, s37 
s_mov_b32 s22, s38 
s_mov_b32 s23, s7 
s_mov_b32 s24, s12   
s_mov_b32 s25, s13   
s_mov_b32 s26, s4   
s_mov_b32 s27, s7   
s_mov_b32 s28, s6 
s_mov_b32 s29, s14 
 s_add_u32     m0,  s10, 0x46f0 * 0   
 buffer_load_dword  v2,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v3,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v11,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v12,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v13,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v14,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v15,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v16,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v17,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v18,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v19, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v20, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v21, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v22, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v23, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v24, s[16:19], 0 offen lds 
 buffer_load_dwordx4  acc[0:3],   v4, s[20:23], 0 offen offset: 0x400 * 0 
 buffer_load_dwordx4  acc[4:7],   v4, s[20:23], 0 offen offset: 0x400 * 1 
 buffer_load_dwordx4  acc[8:11],  v5, s[20:23], 0 offen offset: 0x400 * 0 
 buffer_load_dwordx4  acc[12:15], v5, s[20:23], 0 offen offset: 0x400 * 1 
 buffer_load_dwordx4  acc[32:35], v6, s[20:23], 0 offen offset: 0x400 * 0 
 buffer_load_dwordx4  acc[36:39], v6, s[20:23], 0 offen offset: 0x400 * 1 
 buffer_load_dwordx4  acc[40:43], v7, s[20:23], 0 offen offset: 0x400 * 0 
 buffer_load_dwordx4  acc[44:47], v7, s[20:23], 0 offen offset: 0x400 * 1 
 buffer_load_dword  v224, v1,  s[24:27], 0 offen offset: 64 * 0 
 buffer_load_dword  v225, v1,  s[24:27], 0 offen offset: 64 * 1 
 buffer_load_dword  v226, v1,  s[24:27], 0 offen offset: 64 * 2 
 buffer_load_dword  v227, v1,  s[24:27], 0 offen offset: 64 * 3 
 buffer_load_dword  v228, v1,  s[24:27], 0 offen offset: 64 * 4 
 buffer_load_dword  v229, v1,  s[24:27], 0 offen offset: 64 * 5 
 buffer_load_dword  v230, v1,  s[24:27], 0 offen offset: 64 * 6 
 buffer_load_dword  v231, v1,  s[24:27], 0 offen offset: 64 * 7 
 s_load_dwordx2 s[30:31], s[28:29],  0 
s_cmp_gt_i32  s40 1        
s_cselect_b32 s86, s41, 0 
s_add_u32     s16, s86, s16          
s_addc_u32    s17, 0, s17            
s_cmp_gt_i32  s40 1        
s_cselect_b32 s86, s43, 0 
s_add_u32     s20, s86, s20          
s_addc_u32    s21, 0, s21            
s_cmp_gt_i32  s40 1        
s_cselect_b32 s86, s42, 0   
s_add_u32     s24, s86, s24          
s_addc_u32    s25, 0, s25            
s_cmp_gt_i32  s40 1        
s_cselect_b32 s86, s15, 0   
s_add_u32     s28, s86, s28          
s_addc_u32    s29, 0, s29            
 s_add_u32     m0,  s10, 0x46f0 * 1   
 buffer_load_dword  v2,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v3,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v11,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v12,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v13,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v14,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v15,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v16,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v17,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v18,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v19, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v20, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v21, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v22, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v23, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 buffer_load_dword  v24, s[16:19], 0 offen lds 
 s_waitcnt vmcnt(16) 
 s_barrier          
 ds_read_b128  acc[64:67],   v10  offset: 0x8e0 * 0 + 64 * 0 
 ds_read_b128  acc[68:71],   v10  offset: 0x8e0 * 0 + 64 * 1 
 ds_read_b128  acc[72:75],   v10  offset: 0x8e0 * 1 + 64 * 0 
 ds_read_b128  acc[76:79],   v10  offset: 0x8e0 * 1 + 64 * 1 
 ds_read_b128  acc[80:83],   v10  offset: 0x8e0 * 2 + 64 * 0 
 ds_read_b128  acc[84:87],   v10  offset: 0x8e0 * 2 + 64 * 1 
 ds_read_b128  acc[88:91],   v10  offset: 0x8e0 * 3 + 64 * 0 
 ds_read_b128  acc[92:95],   v10  offset: 0x8e0 * 3 + 64 * 1 
 ds_read_b128  acc[96:99],   v10  offset: 0x8e0 * 4 + 64 * 0 
 ds_read_b128  acc[100:103], v10  offset: 0x8e0 * 4 + 64 * 1 
 ds_read_b128  acc[104:107], v10  offset: 0x8e0 * 5 + 64 * 0 
 ds_read_b128  acc[108:111], v10  offset: 0x8e0 * 5 + 64 * 1 
 ds_read_b128  acc[112:115], v10  offset: 0x8e0 * 6 + 64 * 0 
 ds_read_b128  acc[116:119], v10  offset: 0x8e0 * 6 + 64 * 1 
 ds_read_b128  acc[120:123], v10  offset: 0x8e0 * 7 + 64 * 0 
 ds_read_b128  acc[124:127], v10  offset: 0x8e0 * 7 + 64 * 1 
 v_mov_b32 v64,  0 
 v_mov_b32 v65,  0 
 v_mov_b32 v66,  0 
 v_mov_b32 v67,  0 
 v_mov_b32 v68,  0 
 v_mov_b32 v69,  0 
 v_mov_b32 v70,  0 
 v_mov_b32 v71,  0 
 v_mov_b32 v72,  0 
 v_mov_b32 v73,  0 
 v_mov_b32 v74,  0 
 v_mov_b32 v75,  0 
 v_mov_b32 v76,  0 
 v_mov_b32 v77,  0 
 v_mov_b32 v78,  0 
 v_mov_b32 v79,  0 
 v_mov_b32 v80,  0 
 v_mov_b32 v81,  0 
 v_mov_b32 v82,  0 
 v_mov_b32 v83,  0 
 v_mov_b32 v84,  0 
 v_mov_b32 v85,  0 
 v_mov_b32 v86,  0 
 v_mov_b32 v87,  0 
 v_mov_b32 v88,  0 
 v_mov_b32 v89,  0 
 v_mov_b32 v90,  0 
 v_mov_b32 v91,  0 
 v_mov_b32 v92,  0 
 v_mov_b32 v93,  0 
 v_mov_b32 v94,  0 
 v_mov_b32 v95,  0 
 v_mov_b32 v96,  0 
 v_mov_b32 v97,  0 
 v_mov_b32 v98,  0 
 v_mov_b32 v99,  0 
 v_mov_b32 v100,  0 
 v_mov_b32 v101,  0 
 v_mov_b32 v102,  0 
 v_mov_b32 v103,  0 
 v_mov_b32 v104, 0 
 v_mov_b32 v105, 0 
 v_mov_b32 v106, 0 
 v_mov_b32 v107, 0 
 v_mov_b32 v108, 0 
 v_mov_b32 v109, 0 
 v_mov_b32 v110, 0 
 v_mov_b32 v111, 0 
 v_mov_b32 v112, 0 
 v_mov_b32 v113, 0 
 v_mov_b32 v114, 0 
 v_mov_b32 v115, 0 
 v_mov_b32 v116, 0 
 v_mov_b32 v117, 0 
 v_mov_b32 v118, 0 
 v_mov_b32 v119, 0 
 v_mov_b32 v120, 0 
 v_mov_b32 v121, 0 
 v_mov_b32 v122, 0 
 v_mov_b32 v123, 0 
 v_mov_b32 v124, 0 
 v_mov_b32 v125, 0 
 v_mov_b32 v126, 0 
 v_mov_b32 v127, 0 
 v_mov_b32 v128, 0 
 v_mov_b32 v129, 0 
 v_mov_b32 v130, 0 
 v_mov_b32 v131, 0 
 v_mov_b32 v132, 0 
 v_mov_b32 v133, 0 
 v_mov_b32 v134, 0 
 v_mov_b32 v135, 0 
 v_mov_b32 v136, 0 
 v_mov_b32 v137, 0 
 v_mov_b32 v138, 0 
 v_mov_b32 v139, 0 
 v_mov_b32 v140, 0 
 v_mov_b32 v141, 0 
 v_mov_b32 v142, 0 
 v_mov_b32 v143, 0 
 v_mov_b32 v144, 0 
 v_mov_b32 v145, 0 
 v_mov_b32 v146, 0 
 v_mov_b32 v147, 0 
 v_mov_b32 v148, 0 
 v_mov_b32 v149, 0 
 v_mov_b32 v150, 0 
 v_mov_b32 v151, 0 
 v_mov_b32 v152, 0 
 v_mov_b32 v153, 0 
 v_mov_b32 v154, 0 
 v_mov_b32 v155, 0 
 v_mov_b32 v156, 0 
 v_mov_b32 v157, 0 
 v_mov_b32 v158, 0 
 v_mov_b32 v159, 0 
 v_mov_b32 v160, 0 
 v_mov_b32 v161, 0 
 v_mov_b32 v162, 0 
 v_mov_b32 v163, 0 
 v_mov_b32 v164, 0 
 v_mov_b32 v165, 0 
 v_mov_b32 v166, 0 
 v_mov_b32 v167, 0 
 v_mov_b32 v168, 0 
 v_mov_b32 v169, 0 
 v_mov_b32 v170, 0 
 v_mov_b32 v171, 0 
 v_mov_b32 v172, 0 
 v_mov_b32 v173, 0 
 v_mov_b32 v174, 0 
 v_mov_b32 v175, 0 
 v_mov_b32 v176, 0 
 v_mov_b32 v177, 0 
 v_mov_b32 v178, 0 
 v_mov_b32 v179, 0 
 v_mov_b32 v180, 0 
 v_mov_b32 v181, 0 
 v_mov_b32 v182, 0 
 v_mov_b32 v183, 0 
 v_mov_b32 v184, 0 
 v_mov_b32 v185, 0 
 v_mov_b32 v186, 0 
 v_mov_b32 v187, 0 
 v_mov_b32 v188, 0 
 v_mov_b32 v189, 0 
 v_mov_b32 v190, 0 
 v_mov_b32 v191, 0 
s_cmp_gt_i32  s40 2        
s_cselect_b32 s86, s41, 0 
s_add_u32     s16, s86, s16          
s_addc_u32    s17, 0, s17            
L_start0: 
 s_waitcnt lgkmcnt(0) 
 s_waitcnt vmcnt(16)  
 s_add_u32 m0,  s10, 0x46f0 * 0 
 s_load_dwordx2 s[84:85], s[28:29], 0 
 v_mov_b32  v50, s31        
 v_mul_f32  v240, v224, v50 
 v_mul_f32  v241, v225, v50 
 v_mul_f32  v242, v226, v50 
 v_mul_f32  v243, v227, v50 
 v_mul_f32  v244, v228, v50 
 v_mul_f32  v245, v229, v50 
 v_mul_f32  v246, v230, v50 
 v_mul_f32  v247, v231, v50 
 v_mov_b32  v50, s30        
 v_mul_f32  v224, v224, v50 
 v_mul_f32  v225, v225, v50 
 v_mul_f32  v226, v226, v50 
 v_mul_f32  v227, v227, v50 
 v_mul_f32  v228, v228, v50 
 v_mul_f32  v229, v229, v50 
 v_mul_f32  v230, v230, v50 
 v_mul_f32  v231, v231, v50 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[64:65], 0        
 buffer_load_dwordx4  acc[16:19], v4, s[20:23], 0 offen offset: 0x400 * 0 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[66:67], v[56:59] 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[68:69], v[56:59] 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[70:71], v[56:59] 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[64:65], 0        
 buffer_load_dwordx4  acc[20:23], v4, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v64,  v56, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[66:67], v[60:63] 
 v_fmac_f32 v65,  v57, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[68:69], v[60:63] 
 v_fmac_f32 v66,  v58, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[70:71], v[60:63] 
 v_fmac_f32 v67,  v59, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[72:73], 0        
 buffer_load_dwordx4  acc[24:27], v5, s[20:23], 0 offen offset: 0x400 * 0 
 v_fmac_f32 v68,  v60, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[74:75], v[56:59] 
 v_fmac_f32 v69,  v61, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[76:77], v[56:59] 
 v_fmac_f32 v70,  v62, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[78:79], v[56:59] 
 v_fmac_f32 v71,  v63, v224 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[72:73], 0        
 buffer_load_dwordx4  acc[28:31], v5, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v72,  v56, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[74:75], v[60:63] 
 v_fmac_f32 v73,  v57, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[76:77], v[60:63] 
 v_fmac_f32 v74,  v58, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[78:79], v[60:63] 
 v_fmac_f32 v75,  v59, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[80:81], 0        
 buffer_load_dwordx4  acc[48:51], v6, s[20:23], 0 offen offset: 0x400 * 0 
 v_fmac_f32 v76,  v60, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[82:83], v[56:59] 
 v_fmac_f32 v77,  v61, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[84:85], v[56:59] 
 v_fmac_f32 v78,  v62, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[86:87], v[56:59] 
 v_fmac_f32 v79,  v63, v225 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[80:81], 0        
 buffer_load_dwordx4  acc[52:55], v6, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v80,  v56,v226 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[82:83], v[60:63] 
 v_fmac_f32 v81,  v57,v226 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[84:85], v[60:63] 
 v_fmac_f32 v82,  v58,v226 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[86:87], v[60:63] 
 v_fmac_f32 v83,  v59,v226 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[88:89], 0        
 buffer_load_dwordx4  acc[56:59], v7, s[20:23], 0 offen offset: 0x400 * 0 
 v_fmac_f32 v84,  v60, v226 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[90:91], v[56:59] 
 v_fmac_f32 v85,  v61, v226 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[92:93], v[56:59] 
 v_fmac_f32 v86,  v62, v226 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[94:95], v[56:59] 
 v_fmac_f32 v87,  v63, v226 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[88:89], 0        
 buffer_load_dwordx4  acc[60:63], v7, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v88,  v56, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[90:91], v[60:63] 
 v_fmac_f32 v89,  v57, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[92:93], v[60:63] 
 v_fmac_f32 v90,  v58, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[94:95], v[60:63] 
 v_fmac_f32 v91,  v59, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[96:97],   0        
 buffer_load_dword    v232, v1,  s[24:27], 0 offen offset: 64 * 0 
 v_fmac_f32 v92,  v60, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[98:99],   v[56:59] 
 v_fmac_f32 v93,  v61, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[100:101], v[56:59] 
 v_fmac_f32 v94,  v62, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[102:103], v[56:59] 
 v_fmac_f32 v95,  v63, v227 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[96:97],   0        
 buffer_load_dword    v233, v1,  s[24:27], 0 offen offset: 64 * 1 
 v_fmac_f32 v96,  v56, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[98:99],   v[60:63] 
 v_fmac_f32 v97,  v57, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[100:101], v[60:63] 
 v_fmac_f32 v98,  v58, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[102:103], v[60:63] 
 v_fmac_f32 v99,  v59, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[104:105], 0        
 buffer_load_dword    v234, v1,  s[24:27], 0 offen offset: 64 * 2 
 v_fmac_f32 v100,  v60, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[106:107], v[56:59] 
 v_fmac_f32 v101,  v61, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[108:109], v[56:59] 
 v_fmac_f32 v102,  v62, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[110:111], v[56:59] 
 v_fmac_f32 v103,  v63, v228 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[104:105], 0        
 buffer_load_dword    v235, v1,  s[24:27], 0 offen offset: 64 * 3 
 v_fmac_f32 v104, v56, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[106:107], v[60:63] 
 v_fmac_f32 v105, v57, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[108:109], v[60:63] 
 v_fmac_f32 v106, v58, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[110:111], v[60:63] 
 v_fmac_f32 v107, v59, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[112:113], 0        
 buffer_load_dword    v236, v1,  s[24:27], 0 offen offset: 64 * 4 
 v_fmac_f32 v108, v60, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[114:115], v[56:59] 
 v_fmac_f32 v109, v61, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[116:117], v[56:59] 
 v_fmac_f32 v110, v62, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[118:119], v[56:59] 
 v_fmac_f32 v111, v63, v229 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[112:113], 0        
 buffer_load_dword    v237, v1,  s[24:27], 0 offen offset: 64 * 5 
 v_fmac_f32 v112, v56, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[114:115], v[60:63] 
 v_fmac_f32 v113, v57, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[116:117], v[60:63] 
 v_fmac_f32 v114, v58, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[118:119], v[60:63] 
 v_fmac_f32 v115, v59, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[0:1],   acc[120:121], 0        
 buffer_load_dword    v238, v1,  s[24:27], 0 offen offset: 64 * 6 
 v_fmac_f32 v116, v60, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[2:3],   acc[122:123], v[56:59] 
 v_fmac_f32 v117, v61, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[4:5],   acc[124:125], v[56:59] 
 v_fmac_f32 v118, v62, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[6:7],   acc[126:127], v[56:59] 
 v_fmac_f32 v119, v63, v230 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[8:9],   acc[120:121], 0        
 buffer_load_dword    v239, v1,  s[24:27], 0 offen offset: 64 * 7 
 v_fmac_f32 v120, v56, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[10:11], acc[122:123], v[60:63] 
 v_fmac_f32 v121, v57, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[12:13], acc[124:125], v[60:63] 
 v_fmac_f32 v122, v58, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[14:15], acc[126:127], v[60:63] 
 s_waitcnt vmcnt(16) 
 s_barrier           
 v_fmac_f32 v123, v59, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[64:65], 0        
 ds_read_b128  acc[128:131], v10  offset: 0x8e0 * 0 + 64 * 0 + 0x46f0  
 v_fmac_f32 v124, v60, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[66:67], v[56:59] 
 buffer_load_dword  v2,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v125, v61, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[68:69], v[56:59] 
 v_fmac_f32 v126, v62, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[70:71], v[56:59] 
 v_fmac_f32 v127, v63, v231 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[64:65], 0        
 ds_read_b128  acc[132:135], v10  offset: 0x8e0 * 0 + 64 * 1 + 0x46f0  
 v_fmac_f32 v128, v56, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[66:67], v[60:63] 
 buffer_load_dword  v3,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v129, v57, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[68:69], v[60:63] 
 v_fmac_f32 v130, v58, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[70:71], v[60:63] 
 v_fmac_f32 v131, v59, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[72:73], 0        
 ds_read_b128  acc[136:139], v10  offset: 0x8e0 * 1 + 64 * 0 + 0x46f0  
 v_fmac_f32 v132, v60, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[74:75], v[56:59] 
 buffer_load_dword  v11,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v133, v61, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[76:77], v[56:59] 
 v_fmac_f32 v134, v62, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[78:79], v[56:59] 
 v_fmac_f32 v135, v63, v240 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[72:73], 0        
 ds_read_b128  acc[140:143], v10  offset: 0x8e0 * 1 + 64 * 1 + 0x46f0  
 v_fmac_f32 v136, v56, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[74:75], v[60:63] 
 buffer_load_dword  v12,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v137, v57, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[76:77], v[60:63] 
 v_fmac_f32 v138, v58, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[78:79], v[60:63] 
 v_fmac_f32 v139, v59, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[80:81], 0        
 ds_read_b128  acc[144:147], v10  offset: 0x8e0 * 2 + 64 * 0 + 0x46f0  
 v_fmac_f32 v140, v60, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[82:83], v[56:59] 
 buffer_load_dword  v13,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v141, v61, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[84:85], v[56:59] 
 v_fmac_f32 v142, v62, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[86:87], v[56:59] 
 v_fmac_f32 v143, v63, v241 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[80:81], 0        
 ds_read_b128  acc[148:151], v10  offset: 0x8e0 * 2 + 64 * 1 + 0x46f0  
 v_fmac_f32 v144, v56, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[82:83], v[60:63] 
 buffer_load_dword  v14,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v145, v57, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[84:85], v[60:63] 
 v_fmac_f32 v146, v58, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[86:87], v[60:63] 
 v_fmac_f32 v147, v59, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[88:89], 0        
 ds_read_b128  acc[152:155], v10  offset: 0x8e0 * 3 + 64 * 0 + 0x46f0  
 v_fmac_f32 v148, v60, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[90:91], v[56:59] 
 buffer_load_dword  v15,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v149, v61, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[92:93], v[56:59] 
 v_fmac_f32 v150, v62, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[94:95], v[56:59] 
 v_fmac_f32 v151, v63, v242 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[88:89], 0        
 ds_read_b128  acc[156:159], v10  offset: 0x8e0 * 3 + 64 * 1 + 0x46f0  
 v_fmac_f32 v152, v56, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[90:91], v[60:63] 
 buffer_load_dword  v16,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v153, v57, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[92:93], v[60:63] 
 v_fmac_f32 v154, v58, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[94:95], v[60:63] 
 v_fmac_f32 v155, v59, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[96:97],   0        
 ds_read_b128  acc[160:163], v10  offset: 0x8e0 * 4 + 64 * 0 + 0x46f0  
 v_fmac_f32 v156, v60, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[98:99],   v[56:59] 
 buffer_load_dword  v17,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v157, v61, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[100:101], v[56:59] 
 v_fmac_f32 v158, v62, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[102:103], v[56:59] 
 v_fmac_f32 v159, v63, v243 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[96:97],   0        
 ds_read_b128  acc[164:167], v10  offset: 0x8e0 * 4 + 64 * 1 + 0x46f0  
 v_fmac_f32 v160, v56, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[98:99],   v[60:63] 
 buffer_load_dword  v18,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v161, v57, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[100:101], v[60:63] 
 v_fmac_f32 v162, v58, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[102:103], v[60:63] 
 v_fmac_f32 v163, v59, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[104:105], 0        
 ds_read_b128  acc[168:171], v10  offset: 0x8e0 * 5 + 64 * 0 + 0x46f0  
 v_fmac_f32 v164, v60, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[106:107], v[56:59] 
 buffer_load_dword  v19, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v165, v61, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[108:109], v[56:59] 
 v_fmac_f32 v166, v62, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[110:111], v[56:59] 
 v_fmac_f32 v167, v63, v244 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[104:105], 0        
 ds_read_b128  acc[172:175], v10  offset: 0x8e0 * 5 + 64 * 1 + 0x46f0  
 v_fmac_f32 v168, v56, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[106:107], v[60:63] 
 buffer_load_dword  v20, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v169, v57, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[108:109], v[60:63] 
 v_fmac_f32 v170, v58, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[110:111], v[60:63] 
 v_fmac_f32 v171, v59, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[112:113], 0        
 ds_read_b128  acc[176:179], v10  offset: 0x8e0 * 6 + 64 * 0 + 0x46f0  
 v_fmac_f32 v172, v60, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[114:115], v[56:59] 
 buffer_load_dword  v21, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v173, v61, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[116:117], v[56:59] 
 v_fmac_f32 v174, v62, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[118:119], v[56:59] 
 v_fmac_f32 v175, v63, v245 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[112:113], 0        
 ds_read_b128  acc[180:183], v10  offset: 0x8e0 * 6 + 64 * 1 + 0x46f0  
 v_fmac_f32 v176, v56, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[114:115], v[60:63] 
 buffer_load_dword  v22, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v177, v57, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[116:117], v[60:63] 
 v_fmac_f32 v178, v58, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[118:119], v[60:63] 
 v_fmac_f32 v179, v59, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[32:33], acc[120:121], 0        
 ds_read_b128  acc[184:187], v10  offset: 0x8e0 * 7 + 64 * 0 + 0x46f0  
 v_fmac_f32 v180, v60, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[34:35], acc[122:123], v[56:59] 
 buffer_load_dword  v23, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v181, v61, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[36:37], acc[124:125], v[56:59] 
 v_fmac_f32 v182, v62, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[38:39], acc[126:127], v[56:59] 
 v_fmac_f32 v183, v63, v246 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[40:41], acc[120:121], 0        
 ds_read_b128  acc[188:191], v10  offset: 0x8e0 * 7 + 64 * 1 + 0x46f0  
 v_fmac_f32 v184, v56, v247 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[42:43], acc[122:123], v[60:63] 
 buffer_load_dword  v24, s[16:19], 0 offen lds 
 v_fmac_f32 v185, v57, v247 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[44:45], acc[124:125], v[60:63] 
 v_fmac_f32 v186, v58, v247 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[46:47], acc[126:127], v[60:63] 
 v_fmac_f32 v187, v59, v247 
 s_cmp_gt_i32  s40 2        
 s_cselect_b32 s86, s41, 0 
 v_fmac_f32 v188, v60, v247 
 v_fmac_f32 v189, v61, v247 
 v_fmac_f32 v190, v62, v247 
 v_fmac_f32 v191, v63, v247 
 s_sub_i32     s40, s40, 1 
 s_cmp_gt_i32  s40 0                 
 s_cbranch_scc0 L_end0                        
 s_add_u32     s16, s86, s16          
 s_addc_u32    s17, 0, s17            
 s_cmp_gt_i32  s40 1        
 s_cselect_b32 s86, s43, 0 
 s_add_u32     s20, s86, s20          
 s_addc_u32    s21, 0, s21            
 s_cmp_gt_i32  s40 1        
 s_cselect_b32 s86, s42, 0   
 s_add_u32     s24, s86, s24          
 s_addc_u32    s25, 0, s25            
 s_cmp_gt_i32  s40 1        
 s_cselect_b32 s86, s15, 0   
 s_add_u32     s28, s86, s28          
 s_addc_u32    s29, 0, s29            
 s_waitcnt lgkmcnt(0) 
 s_waitcnt vmcnt(16)  
 s_add_u32 m0,  s10, 0x46f0 * 1 
 s_load_dwordx2 s[30:31], s[28:29],  0 
 v_mov_b32  v50, s85        
 v_mul_f32  v248, v232, v50 
 v_mul_f32  v249, v233, v50 
 v_mul_f32  v250, v234, v50 
 v_mul_f32  v251, v235, v50 
 v_mul_f32  v252, v236, v50 
 v_mul_f32  v253, v237, v50 
 v_mul_f32  v254, v238, v50 
 v_mul_f32  v255, v239, v50 
 v_mov_b32  v50, s84        
 v_mul_f32  v232, v232, v50 
 v_mul_f32  v233, v233, v50 
 v_mul_f32  v234, v234, v50 
 v_mul_f32  v235, v235, v50 
 v_mul_f32  v236, v236, v50 
 v_mul_f32  v237, v237, v50 
 v_mul_f32  v238, v238, v50 
 v_mul_f32  v239, v239, v50 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[128:129], 0        
 buffer_load_dwordx4  acc[0:3],   v4, s[20:23], 0 offen offset: 0x400 * 0 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[130:131], v[56:59] 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[132:133], v[56:59] 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[134:135], v[56:59] 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[128:129], 0        
 buffer_load_dwordx4  acc[4:7],   v4, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v64,  v56, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[130:131], v[60:63] 
 v_fmac_f32 v65,  v57, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[132:133], v[60:63] 
 v_fmac_f32 v66,  v58, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[134:135], v[60:63] 
 v_fmac_f32 v67,  v59, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[136:137], 0        
 buffer_load_dwordx4  acc[8:11],  v5, s[20:23], 0 offen offset: 0x400 * 0 
 v_fmac_f32 v68,  v60, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[138:139], v[56:59] 
 v_fmac_f32 v69,  v61, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[140:141], v[56:59] 
 v_fmac_f32 v70,  v62, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[142:143], v[56:59] 
 v_fmac_f32 v71,  v63, v232 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[136:137], 0        
 buffer_load_dwordx4  acc[12:15], v5, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v72,  v56, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[138:139], v[60:63] 
 v_fmac_f32 v73,  v57, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[140:141], v[60:63] 
 v_fmac_f32 v74,  v58, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[142:143], v[60:63] 
 v_fmac_f32 v75,  v59, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[144:145], 0        
 buffer_load_dwordx4  acc[32:35], v6, s[20:23], 0 offen offset: 0x400 * 0 
 v_fmac_f32 v76,  v60, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[146:147], v[56:59] 
 v_fmac_f32 v77,  v61, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[148:149], v[56:59] 
 v_fmac_f32 v78,  v62, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[150:151], v[56:59] 
 v_fmac_f32 v79,  v63, v233 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[144:145], 0        
 buffer_load_dwordx4  acc[36:39], v6, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v80,  v56, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[146:147], v[60:63] 
 v_fmac_f32 v81,  v57, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[148:149], v[60:63] 
 v_fmac_f32 v82,  v58, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[150:151], v[60:63] 
 v_fmac_f32 v83,  v59, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[152:153], 0 
 buffer_load_dwordx4  acc[40:43], v7, s[20:23], 0 offen offset: 0x400 * 0 
 v_fmac_f32 v84,  v60, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[154:155], v[56:59] 
 v_fmac_f32 v85,  v61, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[156:157], v[56:59] 
 v_fmac_f32 v86,  v62, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[158:159], v[56:59] 
 v_fmac_f32 v87,  v63, v234 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[152:153], 0 
 buffer_load_dwordx4  acc[44:47], v7, s[20:23], 0 offen offset: 0x400 * 1 
 v_fmac_f32 v88,  v56, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[154:155], v[60:63] 
 v_fmac_f32 v89,  v57, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[156:157], v[60:63] 
 v_fmac_f32 v90,  v58, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[158:159], v[60:63] 
 v_fmac_f32 v91,  v59, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[160:161], 0 
 buffer_load_dword v224, v1,  s[24:27], 0 offen offset: 64 * 0 
 v_fmac_f32 v92,  v60, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[162:163], v[56:59] 
 v_fmac_f32 v93,  v61, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[164:165], v[56:59] 
 v_fmac_f32 v94,  v62, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[166:167], v[56:59] 
 v_fmac_f32 v95,  v63, v235 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[160:161], 0 
 buffer_load_dword v225, v1,  s[24:27], 0 offen offset: 64 * 1 
 v_fmac_f32 v96,  v56, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[162:163], v[60:63] 
 v_fmac_f32 v97,  v57, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[164:165], v[60:63] 
 v_fmac_f32 v98,  v58, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[166:167], v[60:63] 
 v_fmac_f32 v99,  v59, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[168:169], 0 
 buffer_load_dword v226, v1,  s[24:27], 0 offen offset: 64 * 2 
 v_fmac_f32 v100,  v60, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[170:171], v[56:59] 
 v_fmac_f32 v101,  v61, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[172:173], v[56:59] 
 v_fmac_f32 v102,  v62, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[174:175], v[56:59] 
 v_fmac_f32 v103,  v63, v236 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[168:169], 0 
 buffer_load_dword v227, v1,  s[24:27], 0 offen offset: 64 * 3 
 v_fmac_f32 v104, v56, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[170:171], v[60:63] 
 v_fmac_f32 v105, v57, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[172:173], v[60:63] 
 v_fmac_f32 v106, v58, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[174:175], v[60:63] 
 v_fmac_f32 v107, v59, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[176:177], 0 
 buffer_load_dword v228, v1,  s[24:27], 0 offen offset: 64 * 4 
 v_fmac_f32 v108, v60, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[178:179], v[56:59] 
 v_fmac_f32 v109, v61, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[180:181], v[56:59] 
 v_fmac_f32 v110, v62, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[182:183], v[56:59] 
 v_fmac_f32 v111, v63, v237 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[176:177], 0 
 buffer_load_dword v229, v1,  s[24:27], 0 offen offset: 64 * 5 
 v_fmac_f32 v112, v56, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[178:179], v[60:63] 
 v_fmac_f32 v113, v57, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[180:181], v[60:63] 
 v_fmac_f32 v114, v58, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[182:183], v[60:63] 
 v_fmac_f32 v115, v59, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[16:17], acc[184:185], 0 
 buffer_load_dword v230, v1,  s[24:27], 0 offen offset: 64 * 6 
 v_fmac_f32 v116, v60, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[18:19], acc[186:187], v[56:59] 
 v_fmac_f32 v117, v61, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[20:21], acc[188:189], v[56:59] 
 v_fmac_f32 v118, v62, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[22:23], acc[190:191], v[56:59] 
 v_fmac_f32 v119, v63, v238 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[24:25], acc[184:185], 0 
 buffer_load_dword v231, v1,  s[24:27], 0 offen offset: 64 * 7 
 v_fmac_f32 v120, v56, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[26:27], acc[186:187], v[60:63] 
 v_fmac_f32 v121, v57, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[28:29], acc[188:189], v[60:63] 
 v_fmac_f32 v122, v58, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[30:31], acc[190:191], v[60:63] 
 s_waitcnt vmcnt(16) 
 s_barrier           
 v_fmac_f32 v123, v59, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[128:129], 0 
 ds_read_b128  acc[64:67],   v10  offset: 0x8e0 * 0 + 64 * 0 
 v_fmac_f32 v124, v60, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[130:131], v[56:59] 
 buffer_load_dword  v2,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v125, v61, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[132:133], v[56:59] 
 v_fmac_f32 v126, v62, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[134:135], v[56:59] 
 v_fmac_f32 v127, v63, v239 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[128:129], 0 
 ds_read_b128  acc[68:71],   v10  offset: 0x8e0 * 0 + 64 * 1 
 v_fmac_f32 v128, v56, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[130:131], v[60:63] 
 buffer_load_dword  v3,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v129, v57, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[132:133], v[60:63] 
 v_fmac_f32 v130, v58, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[134:135], v[60:63] 
 v_fmac_f32 v131, v59, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[136:137], 0 
 ds_read_b128  acc[72:75],   v10  offset: 0x8e0 * 1 + 64 * 0 
 v_fmac_f32 v132, v60, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[138:139], v[56:59] 
 buffer_load_dword  v11,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v133, v61, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[140:141], v[56:59] 
 v_fmac_f32 v134, v62, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[142:143], v[56:59] 
 v_fmac_f32 v135, v63, v248 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[136:137], 0 
 ds_read_b128  acc[76:79],   v10  offset: 0x8e0 * 1 + 64 * 1 
 v_fmac_f32 v136, v56, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[138:139], v[60:63] 
 buffer_load_dword  v12,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v137, v57, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[140:141], v[60:63] 
 v_fmac_f32 v138, v58, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[142:143], v[60:63] 
 v_fmac_f32 v139, v59, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[144:145], 0 
 ds_read_b128  acc[80:83],   v10  offset: 0x8e0 * 2 + 64 * 0 
 v_fmac_f32 v140, v60, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[146:147], v[56:59] 
 buffer_load_dword  v13,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v141, v61, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[148:149], v[56:59] 
 v_fmac_f32 v142, v62, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[150:151], v[56:59] 
 v_fmac_f32 v143, v63, v249 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[144:145], 0 
 ds_read_b128  acc[84:87],   v10  offset: 0x8e0 * 2 + 64 * 1 
 v_fmac_f32 v144, v56, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[146:147], v[60:63] 
 buffer_load_dword  v14,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v145, v57, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[148:149], v[60:63] 
 v_fmac_f32 v146, v58, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[150:151], v[60:63] 
 v_fmac_f32 v147, v59, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[152:153], 0 
 ds_read_b128  acc[88:91],   v10  offset: 0x8e0 * 3 + 64 * 0 
 v_fmac_f32 v148, v60, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[154:155], v[56:59] 
 buffer_load_dword  v15,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v149, v61, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[156:157], v[56:59] 
 v_fmac_f32 v150, v62, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[158:159], v[56:59] 
 v_fmac_f32 v151, v63, v250 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[152:153], 0 
 ds_read_b128  acc[92:95],   v10  offset: 0x8e0 * 3 + 64 * 1 
 v_fmac_f32 v152, v56, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[154:155], v[60:63] 
 buffer_load_dword  v16,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v153, v57, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[156:157], v[60:63] 
 v_fmac_f32 v154, v58, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[158:159], v[60:63] 
 v_fmac_f32 v155, v59, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[160:161], 0 
 ds_read_b128  acc[96:99],   v10  offset: 0x8e0 * 4 + 64 * 0 
 v_fmac_f32 v156, v60, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[162:163], v[56:59] 
 buffer_load_dword  v17,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v157, v61, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[164:165], v[56:59] 
 v_fmac_f32 v158, v62, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[166:167], v[56:59] 
 v_fmac_f32 v159, v63, v251 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[160:161], 0 
 ds_read_b128  acc[100:103], v10  offset: 0x8e0 * 4 + 64 * 1 
 v_fmac_f32 v160, v56, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[162:163], v[60:63] 
 buffer_load_dword  v18,  s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v161, v57, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[164:165], v[60:63] 
 v_fmac_f32 v162, v58, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[166:167], v[60:63] 
 v_fmac_f32 v163, v59, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[168:169], 0 
 ds_read_b128  acc[104:107], v10  offset: 0x8e0 * 5 + 64 * 0 
 v_fmac_f32 v164, v60, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[170:171], v[56:59] 
 buffer_load_dword  v19, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v165, v61, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[172:173], v[56:59] 
 v_fmac_f32 v166, v62, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[174:175], v[56:59] 
 v_fmac_f32 v167, v63, v252 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[168:169], 0 
 ds_read_b128  acc[108:111], v10  offset: 0x8e0 * 5 + 64 * 1 
 v_fmac_f32 v168, v56, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[170:171], v[60:63] 
 buffer_load_dword  v20, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v169, v57, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[172:173], v[60:63] 
 v_fmac_f32 v170, v58, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[174:175], v[60:63] 
 v_fmac_f32 v171, v59, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[176:177], 0 
 ds_read_b128  acc[112:115], v10  offset: 0x8e0 * 6 + 64 * 0 
 v_fmac_f32 v172, v60, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[178:179], v[56:59] 
 buffer_load_dword  v21, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v173, v61, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[180:181], v[56:59] 
 v_fmac_f32 v174, v62, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[182:183], v[56:59] 
 v_fmac_f32 v175, v63, v253 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[176:177], 0 
 ds_read_b128  acc[116:119], v10  offset: 0x8e0 * 6 + 64 * 1 
 v_fmac_f32 v176, v56, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[178:179], v[60:63] 
 buffer_load_dword  v22, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v177, v57, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[180:181], v[60:63] 
 v_fmac_f32 v178, v58, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[182:183], v[60:63] 
 v_fmac_f32 v179, v59, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[48:49], acc[184:185], 0 
 ds_read_b128  acc[120:123], v10  offset: 0x8e0 * 7 + 64 * 0 
 v_fmac_f32 v180, v60, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[50:51], acc[186:187], v[56:59] 
 buffer_load_dword  v23, s[16:19], 0 offen lds 
 s_add_u32     m0,  s11, m0 
 v_fmac_f32 v181, v61, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[52:53], acc[188:189], v[56:59] 
 v_fmac_f32 v182, v62, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], acc[54:55], acc[190:191], v[56:59] 
 v_fmac_f32 v183, v63, v254 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[56:57], acc[184:185], 0 
 ds_read_b128  acc[124:127], v10  offset: 0x8e0 * 7 + 64 * 1 
 v_fmac_f32 v184, v56, v255 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[58:59], acc[186:187], v[60:63] 
 buffer_load_dword  v24, s[16:19], 0 offen lds 
 v_fmac_f32 v185, v57, v255 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[60:61], acc[188:189], v[60:63] 
 v_fmac_f32 v186, v58, v255 
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], acc[62:63], acc[190:191], v[60:63] 
 v_fmac_f32 v187, v59, v255 
 s_cmp_gt_i32  s40 3        
 s_cselect_b32 s86, s41, 0 
 v_fmac_f32 v188, v60, v255 
 v_fmac_f32 v189, v61, v255 
 v_fmac_f32 v190, v62, v255 
 v_fmac_f32 v191, v63, v255 
 s_add_u32     s16, s86, s16          
 s_addc_u32    s17, 0, s17            
 s_cmp_gt_i32  s40 2        
 s_cselect_b32 s86, s43, 0 
 s_add_u32     s20, s86, s20          
 s_addc_u32    s21, 0, s21            
 s_cmp_gt_i32  s40 2        
 s_cselect_b32 s86, s42, 0   
 s_add_u32     s24, s86, s24          
 s_addc_u32    s25, 0, s25            
 s_cmp_gt_i32  s40 2        
 s_cselect_b32 s86, s15, 0   
 s_add_u32     s28, s86, s28          
 s_addc_u32    s29, 0, s29            
 s_sub_i32    s40, s40, 1 
 s_cmp_gt_i32 s40 0                 
 s_cbranch_scc0 L_end0   
 s_branch       L_start0 
L_end0: 
  s_nop 2 

	;;#ASMEND
	v_or_b32_e32 v1, s0, v0
	v_and_or_b32 v0, v8, 60, s1
	v_mad_u64_u32 v[2:3], s[0:1], s5, v1, v[0:1]
	v_cvt_f16_f32_e32 v4, v64
	v_ashrrev_i32_e32 v3, 31, v2
	v_cvt_f16_f32_e32 v5, v66
	v_cvt_f16_f32_e32 v6, v67
	v_lshl_add_u64 v[0:1], v[2:3], 1, s[34:35]
	v_cvt_f16_f32_e32 v3, v65
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v68
	v_cvt_f16_f32_e32 v7, v70
	v_cvt_f16_f32_e32 v8, v71
	v_cvt_f16_f32_e32 v9, v69
	v_pack_b32_f16 v4, v4, v3
	global_store_dwordx2 v[0:1], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[0:1], v[4:5], off offset:128
	s_lshl_b32 s0, s5, 4
	v_add_u32_e32 v4, s0, v2
	v_cvt_f16_f32_e32 v6, v72
	v_ashrrev_i32_e32 v5, 31, v4
	v_cvt_f16_f32_e32 v7, v74
	v_cvt_f16_f32_e32 v8, v75
	v_lshl_add_u64 v[2:3], v[4:5], 1, s[34:35]
	v_cvt_f16_f32_e32 v5, v73
	v_pack_b32_f16 v7, v7, v8
	v_cvt_f16_f32_e32 v8, v76
	v_cvt_f16_f32_e32 v9, v78
	v_cvt_f16_f32_e32 v10, v79
	v_cvt_f16_f32_e32 v11, v77
	v_pack_b32_f16 v6, v6, v5
	global_store_dwordx2 v[2:3], v[6:7], off
	v_pack_b32_f16 v7, v9, v10
	v_pack_b32_f16 v6, v8, v11
	global_store_dwordx2 v[2:3], v[6:7], off offset:128
	v_add_u32_e32 v6, s0, v4
	v_cvt_f16_f32_e32 v8, v80
	v_ashrrev_i32_e32 v7, 31, v6
	v_cvt_f16_f32_e32 v9, v82
	v_cvt_f16_f32_e32 v10, v83
	v_lshl_add_u64 v[4:5], v[6:7], 1, s[34:35]
	v_cvt_f16_f32_e32 v7, v81
	v_pack_b32_f16 v9, v9, v10
	v_cvt_f16_f32_e32 v10, v84
	v_cvt_f16_f32_e32 v11, v86
	v_cvt_f16_f32_e32 v12, v87
	v_cvt_f16_f32_e32 v13, v85
	v_pack_b32_f16 v8, v8, v7
	global_store_dwordx2 v[4:5], v[8:9], off
	v_pack_b32_f16 v9, v11, v12
	v_pack_b32_f16 v8, v10, v13
	global_store_dwordx2 v[4:5], v[8:9], off offset:128
	v_add_u32_e32 v8, s0, v6
	v_cvt_f16_f32_e32 v10, v88
	v_ashrrev_i32_e32 v9, 31, v8
	v_cvt_f16_f32_e32 v11, v90
	v_cvt_f16_f32_e32 v12, v91
	v_lshl_add_u64 v[6:7], v[8:9], 1, s[34:35]
	v_cvt_f16_f32_e32 v9, v89
	v_pack_b32_f16 v11, v11, v12
	v_cvt_f16_f32_e32 v12, v92
	v_cvt_f16_f32_e32 v13, v94
	v_cvt_f16_f32_e32 v14, v95
	v_cvt_f16_f32_e32 v15, v93
	v_pack_b32_f16 v10, v10, v9
	global_store_dwordx2 v[6:7], v[10:11], off
	v_pack_b32_f16 v11, v13, v14
	v_pack_b32_f16 v10, v12, v15
	global_store_dwordx2 v[6:7], v[10:11], off offset:128
	v_add_u32_e32 v10, s0, v8
	v_cvt_f16_f32_e32 v12, v96
	v_ashrrev_i32_e32 v11, 31, v10
	v_cvt_f16_f32_e32 v13, v98
	v_cvt_f16_f32_e32 v14, v99
	v_lshl_add_u64 v[8:9], v[10:11], 1, s[34:35]
	v_cvt_f16_f32_e32 v11, v97
	v_pack_b32_f16 v13, v13, v14
	v_cvt_f16_f32_e32 v14, v100
	v_cvt_f16_f32_e32 v15, v102
	v_cvt_f16_f32_e32 v16, v103
	v_cvt_f16_f32_e32 v17, v101
	v_pack_b32_f16 v12, v12, v11
	global_store_dwordx2 v[8:9], v[12:13], off
	v_pack_b32_f16 v13, v15, v16
	v_pack_b32_f16 v12, v14, v17
	global_store_dwordx2 v[8:9], v[12:13], off offset:128
	v_add_u32_e32 v12, s0, v10
	v_cvt_f16_f32_e32 v14, v104
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_f16_f32_e32 v15, v106
	v_cvt_f16_f32_e32 v16, v107
	v_lshl_add_u64 v[10:11], v[12:13], 1, s[34:35]
	v_cvt_f16_f32_e32 v13, v105
	v_pack_b32_f16 v15, v15, v16
	v_cvt_f16_f32_e32 v16, v108
	v_cvt_f16_f32_e32 v17, v110
	v_cvt_f16_f32_e32 v18, v111
	v_cvt_f16_f32_e32 v19, v109
	v_pack_b32_f16 v14, v14, v13
	global_store_dwordx2 v[10:11], v[14:15], off
	v_pack_b32_f16 v15, v17, v18
	v_pack_b32_f16 v14, v16, v19
	global_store_dwordx2 v[10:11], v[14:15], off offset:128
	v_add_u32_e32 v14, s0, v12
	v_cvt_f16_f32_e32 v16, v112
	v_ashrrev_i32_e32 v15, 31, v14
	v_cvt_f16_f32_e32 v17, v114
	v_cvt_f16_f32_e32 v18, v115
	v_lshl_add_u64 v[12:13], v[14:15], 1, s[34:35]
	v_cvt_f16_f32_e32 v15, v113
	v_pack_b32_f16 v17, v17, v18
	v_cvt_f16_f32_e32 v18, v116
	v_cvt_f16_f32_e32 v19, v118
	v_cvt_f16_f32_e32 v20, v119
	v_cvt_f16_f32_e32 v21, v117
	v_pack_b32_f16 v16, v16, v15
	global_store_dwordx2 v[12:13], v[16:17], off
	v_pack_b32_f16 v17, v19, v20
	v_pack_b32_f16 v16, v18, v21
	global_store_dwordx2 v[12:13], v[16:17], off offset:128
	v_add_u32_e32 v14, s0, v14
	v_cvt_f16_f32_e32 v16, v120
	v_ashrrev_i32_e32 v15, 31, v14
	v_cvt_f16_f32_e32 v17, v122
	v_cvt_f16_f32_e32 v18, v123
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[34:35]
	v_cvt_f16_f32_e32 v19, v121
	v_pack_b32_f16 v17, v17, v18
	v_cvt_f16_f32_e32 v18, v124
	v_cvt_f16_f32_e32 v20, v126
	v_cvt_f16_f32_e32 v21, v127
	v_cvt_f16_f32_e32 v22, v125
	v_pack_b32_f16 v16, v16, v19
	global_store_dwordx2 v[14:15], v[16:17], off
	v_pack_b32_f16 v17, v20, v21
	v_pack_b32_f16 v16, v18, v22
	global_store_dwordx2 v[14:15], v[16:17], off offset:128
	v_cvt_f16_f32_e32 v16, v130
	v_cvt_f16_f32_e32 v17, v131
	v_cvt_f16_f32_e32 v18, v128
	v_cvt_f16_f32_e32 v19, v129
	v_pack_b32_f16 v17, v16, v17
	v_cvt_f16_f32_e32 v20, v132
	v_cvt_f16_f32_e32 v21, v134
	v_cvt_f16_f32_e32 v22, v135
	v_cvt_f16_f32_e32 v23, v133
	v_pack_b32_f16 v16, v18, v19
	global_store_dwordx2 v[0:1], v[16:17], off offset:256
	v_pack_b32_f16 v17, v21, v22
	v_pack_b32_f16 v16, v20, v23
	global_store_dwordx2 v[0:1], v[16:17], off offset:384
	v_cvt_f16_f32_e32 v0, v138
	v_cvt_f16_f32_e32 v1, v139
	v_cvt_f16_f32_e32 v16, v136
	v_cvt_f16_f32_e32 v17, v137
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v18, v140
	v_cvt_f16_f32_e32 v19, v142
	v_cvt_f16_f32_e32 v20, v143
	v_cvt_f16_f32_e32 v21, v141
	v_pack_b32_f16 v0, v16, v17
	global_store_dwordx2 v[2:3], v[0:1], off offset:256
	v_pack_b32_f16 v1, v19, v20
	v_pack_b32_f16 v0, v18, v21
	global_store_dwordx2 v[2:3], v[0:1], off offset:384
	v_cvt_f16_f32_e32 v0, v146
	v_cvt_f16_f32_e32 v1, v147
	v_cvt_f16_f32_e32 v2, v144
	v_cvt_f16_f32_e32 v3, v145
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v16, v148
	v_cvt_f16_f32_e32 v17, v150
	v_cvt_f16_f32_e32 v18, v151
	v_cvt_f16_f32_e32 v19, v149
	v_pack_b32_f16 v0, v2, v3
	global_store_dwordx2 v[4:5], v[0:1], off offset:256
	v_pack_b32_f16 v1, v17, v18
	v_pack_b32_f16 v0, v16, v19
	global_store_dwordx2 v[4:5], v[0:1], off offset:384
	v_cvt_f16_f32_e32 v0, v154
	v_cvt_f16_f32_e32 v1, v155
	v_cvt_f16_f32_e32 v2, v152
	v_cvt_f16_f32_e32 v3, v153
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v4, v156
	v_cvt_f16_f32_e32 v5, v158
	v_cvt_f16_f32_e32 v16, v159
	v_cvt_f16_f32_e32 v17, v157
	v_pack_b32_f16 v0, v2, v3
	global_store_dwordx2 v[6:7], v[0:1], off offset:256
	v_pack_b32_f16 v1, v5, v16
	v_pack_b32_f16 v0, v4, v17
	global_store_dwordx2 v[6:7], v[0:1], off offset:384
	v_cvt_f16_f32_e32 v0, v162
	v_cvt_f16_f32_e32 v1, v163
	v_cvt_f16_f32_e32 v2, v160
	v_cvt_f16_f32_e32 v3, v161
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v4, v164
	v_cvt_f16_f32_e32 v5, v166
	v_cvt_f16_f32_e32 v6, v167
	v_cvt_f16_f32_e32 v7, v165
	v_pack_b32_f16 v0, v2, v3
	global_store_dwordx2 v[8:9], v[0:1], off offset:256
	v_pack_b32_f16 v1, v5, v6
	v_pack_b32_f16 v0, v4, v7
	global_store_dwordx2 v[8:9], v[0:1], off offset:384
	v_cvt_f16_f32_e32 v0, v170
	v_cvt_f16_f32_e32 v1, v171
	v_cvt_f16_f32_e32 v2, v168
	v_cvt_f16_f32_e32 v3, v169
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v4, v172
	v_cvt_f16_f32_e32 v5, v174
	v_cvt_f16_f32_e32 v6, v175
	v_cvt_f16_f32_e32 v7, v173
	v_pack_b32_f16 v0, v2, v3
	global_store_dwordx2 v[10:11], v[0:1], off offset:256
	v_pack_b32_f16 v1, v5, v6
	v_pack_b32_f16 v0, v4, v7
	global_store_dwordx2 v[10:11], v[0:1], off offset:384
	v_cvt_f16_f32_e32 v0, v178
	v_cvt_f16_f32_e32 v1, v179
	v_cvt_f16_f32_e32 v2, v176
	v_cvt_f16_f32_e32 v3, v177
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v4, v180
	v_cvt_f16_f32_e32 v5, v182
	v_cvt_f16_f32_e32 v6, v183
	v_cvt_f16_f32_e32 v7, v181
	v_pack_b32_f16 v0, v2, v3
	global_store_dwordx2 v[12:13], v[0:1], off offset:256
	v_pack_b32_f16 v1, v5, v6
	v_pack_b32_f16 v0, v4, v7
	global_store_dwordx2 v[12:13], v[0:1], off offset:384
	v_cvt_f16_f32_e32 v0, v186
	v_cvt_f16_f32_e32 v1, v187
	v_cvt_f16_f32_e32 v2, v184
	v_cvt_f16_f32_e32 v3, v185
	v_pack_b32_f16 v1, v0, v1
	v_cvt_f16_f32_e32 v4, v188
	v_cvt_f16_f32_e32 v5, v190
	v_cvt_f16_f32_e32 v6, v191
	v_cvt_f16_f32_e32 v7, v189
	v_pack_b32_f16 v0, v2, v3
	global_store_dwordx2 v[14:15], v[0:1], off offset:256
	v_pack_b32_f16 v1, v5, v6
	v_pack_b32_f16 v0, v4, v7
	global_store_dwordx2 v[14:15], v[0:1], off offset:384
	s_endpgm




	.rodata
	.p2align	6
	.amdhsa_kernel flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 112
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  0
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 87
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_xnack_mask 1
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


	.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name:           flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32
    .symbol:         flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.kd
    .sgpr_count:     93
    .vgpr_count:     512
    .agpr_count:     256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    ; .args:
    ;   - .offset:         0
    ;     .size:           112
    ;     .value_kind:     by_value
    .args:
    	- {.name: a_ptr, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    	- {.name: b_ptr, .size: 8, .offset: 8, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    	- {.name: c_ptr, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: sa_ptr, .size: 8, .offset: 24, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    	- {.name: sb_ptr, .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    	- {.name: d_ptr, .size: 8, .offset: 40, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: d_f16_ptr, .size: 8, .offset: 48, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: dbg_int_ptr, .size: 8, .offset: 56, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: dbg_fp8_ptr, .size: 8, .offset: 64, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: dbg_f16_ptr, .size: 8, .offset: 72, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: dbg_fp32_ptr, .size: 8, .offset: 80, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    	- {.name: hidden_size, .size: 4, .offset: 88, .value_kind: by_value, .value_type: i32}
    	- {.name: intermediate_size, .size: 4, .offset: 92, .value_kind: by_value, .value_type: i32}
    	- {.name: num_tokens, .size: 4, .offset: 96, .value_kind: by_value, .value_type: i32}
    	- {.name: num_experts, .size: 4, .offset: 100, .value_kind: by_value, .value_type: i32}
    	- {.name: topk, .size: 4, .offset: 104, .value_kind: by_value, .value_type: i32}
    	- {.name: stride_token, .size: 4, .offset: 108, .value_kind: by_value, .value_type: i32}
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
...
	.end_amdgpu_metadata

