	.text
	.globl	flatmm_uk_gfx9_f16f8_128x128x128_1x4x1_16x16x32
	.p2align	8
	.type	flatmm_uk_gfx9_f16f8_128x128x128_1x4x1_16x16x32 @function

flatmm_uk_gfx9_f16f8_128x128x128_1x4x1_16x16x32:



	s_mov_b32 s4, s3
	s_load_dwordx4 s[8:11], s[0:1], 0x58
	s_load_dwordx2 s[6:7], s[0:1], 0x30
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_load_dwordx4 s[36:39], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s11, s10, s8
	s_lshl_b32 s0, s3, 7
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v2, 0x7c, v1
	s_ashr_i32 s16, s9, 31
	s_lshr_b32 s1, s16, 28
	s_add_i32 s1, s9, s1
	s_ashr_i32 s1, s1, 4
	s_ashr_i32 s5, s8, 31
	s_lshr_b32 s3, s5, 27
	s_add_i32 s3, s8, s3
	s_ashr_i32 s17, s3, 5
	s_lshl_b32 s3, s2, 3
	s_lshl_b32 s18, s17, 9
	s_mul_i32 s3, s3, s18
	s_ashr_i32 s19, s3, 31
	s_add_u32 s33, s14, s3
	s_addc_u32 s34, s15, s19
	s_add_i32 s1, s1, -1
	s_mul_i32 s1, s18, s1
	s_add_i32 s35, s18, s1
	v_lshrrev_b32_e32 v1, 6, v0
	s_nop 0
	v_readfirstlane_b32 s14, v1
	s_mov_b32 s3, 0
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v3, -1, v3
	s_mul_i32 s1, s14, s18
	v_lshl_add_u32 v4, v3, 4, s1
	v_lshl_add_u32 v5, s17, 11, v4
	v_lshrrev_b32_e32 v6, 2, v0
	s_lshl_b32 s1, s2, 7
	s_mul_i32 s40, s14, 0x110
	s_lshr_b32 s5, s5, 25
	s_add_i32 s5, s8, s5
	s_ashr_i32 s41, s5, 7
	v_lshlrev_b32_e32 v7, 4, v0
	v_and_b32_e32 v8, 48, v0
	v_lshlrev_b32_e32 v3, 7, v0
	s_movk_i32 s5, 0x780
	v_and_or_b32 v9, v3, s5, v8
	v_lshrrev_b32_e32 v3, 3, v0
	v_and_b32_e32 v3, 4, v3
	v_or3_b32 v3, v3, v1, s0
	v_mad_u64_u32 v[2:3], s[14:15], s8, v3, v[2:3]
	s_lshl_b32 s5, s8, 3
	v_add_u32_e32 v3, s5, v2
	v_add_u32_e32 v10, s5, v3
	v_add_u32_e32 v11, s5, v10
	v_add_u32_e32 v12, s5, v11
	v_add_u32_e32 v13, s5, v12
	v_add_u32_e32 v14, s5, v13
	v_add_u32_e32 v15, s5, v14
	v_add_u32_e32 v16, s5, v15
	v_add_u32_e32 v17, s5, v16
	v_add_u32_e32 v18, s5, v17
	v_add_u32_e32 v19, s5, v18
	v_add_u32_e32 v20, s5, v19
	v_add_u32_e32 v21, s5, v20
	v_add_u32_e32 v22, s5, v21
	v_add_u32_e32 v23, s5, v22
	s_movk_i32 s5, 0x120
	v_mul_u32_u24_e32 v1, 0x120, v1
	s_nop 0
	v_readfirstlane_b32 s8, v1
	v_and_b32_e32 v1, 3, v0
	v_bfe_i32 v24, v0, 2, 1
	s_movk_i32 s14, 0x470
	v_and_b32_e32 v24, 0x470, v24
	s_movk_i32 s15, 0x80
	v_and_b32_e32 v25, 0x80, v7
	v_mad_u32_u24 v8, v1, s5, v8
	v_add3_u32 v8, v8, v25, v24
	v_bfe_u32 v24, v0, 3, 1
	v_lshrrev_b32_e32 v0, 1, v0
	v_and_or_b32 v0, v0, 2, v24
	v_lshl_or_b32 v0, v0, 2, v1
	s_lshl_b32 s4, s4, 9
	v_lshl_or_b32 v1, v0, 2, s4
	s_ashr_i32 s4, s11, 31
	s_lshr_b32 s4, s4, 25
	s_add_i32 s4, s11, s4
	s_ashr_i32 s4, s4, 7
	s_lshl_b32 s42, s4, 2
	s_lshl_b32 s10, s10, 2
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s2, s38, s4
	s_addc_u32 s38, s39, s5
	s_lshr_b32 s4, s16, 25
	s_add_i32 s4, s9, s4
	s_ashr_i32 s4, s4, 7
	s_lshl_b32 s39, s4, 2
	s_mov_b64 s[4:5], src_shared_base
	s_mov_b32 s4, s3
	s_mov_b32 s3, 0x20000
	s_movk_i32 s43, 0x800
	s_movk_i32 s44, 0x440
	s_movk_i32 s45, 0x46f0
	;;#ASMSTART
	s_mov_b32 s16,    s12      
s_mov_b32 s17,    s13      
s_mov_b32 s18,    s11      
s_mov_b32 s19,    s3      
s_mov_b32 s20,    s33      
s_mov_b32 s21,    s34      
s_mov_b32 s22,    s35      
s_mov_b32 s23,    s3      
s_mov_b32 s24,    s36     
s_mov_b32 s25,    s37     
s_mov_b32 s26,    s42     
s_mov_b32 s27,    s3     
s_mov_b32 s28,    s2   
s_mov_b32 s29,    s38   
 v_mov_b32 v64,   0              
 v_mov_b32 v65,   0              
 v_mov_b32 v66,   0              
 v_mov_b32 v67,   0              
 v_mov_b32 v68,   0              
 v_mov_b32 v69,   0              
 v_mov_b32 v70,   0              
 v_mov_b32 v71,   0              
 v_mov_b32 v72,   0              
 v_mov_b32 v73,   0              
 v_mov_b32 v74,   0              
 v_mov_b32 v75,   0              
 v_mov_b32 v76,   0              
 v_mov_b32 v77,   0              
 v_mov_b32 v78,   0              
 v_mov_b32 v79,   0              
 v_mov_b32 v80,   0              
 v_mov_b32 v81,   0              
 v_mov_b32 v82,   0              
 v_mov_b32 v83,   0              
 v_mov_b32 v84,   0              
 v_mov_b32 v85,   0              
 v_mov_b32 v86,   0              
 v_mov_b32 v87,   0              
 v_mov_b32 v88,   0              
 v_mov_b32 v89,   0              
 v_mov_b32 v90,   0              
 v_mov_b32 v91,   0              
 v_mov_b32 v92,   0              
 v_mov_b32 v93,   0              
 v_mov_b32 v94,   0              
 v_mov_b32 v95,   0              
 v_mov_b32 v96,   0              
 v_mov_b32 v97,   0              
 v_mov_b32 v98,   0              
 v_mov_b32 v99,   0              
 v_mov_b32 v100,  0              
 v_mov_b32 v101,  0              
 v_mov_b32 v102,  0              
 v_mov_b32 v103,  0              
 v_mov_b32 v104,  0              
 v_mov_b32 v105,  0              
 v_mov_b32 v106,  0              
 v_mov_b32 v107,  0              
 v_mov_b32 v108,  0              
 v_mov_b32 v109,  0              
 v_mov_b32 v110,  0              
 v_mov_b32 v111,  0              
 v_mov_b32 v112,  0              
 v_mov_b32 v113,  0              
 v_mov_b32 v114,  0              
 v_mov_b32 v115,  0              
 v_mov_b32 v116,  0              
 v_mov_b32 v117,  0              
 v_mov_b32 v118,  0              
 v_mov_b32 v119,  0              
 v_mov_b32 v120,  0              
 v_mov_b32 v121,  0              
 v_mov_b32 v122,  0              
 v_mov_b32 v123,  0              
 v_mov_b32 v124,  0              
 v_mov_b32 v125,  0              
 v_mov_b32 v126,  0              
 v_mov_b32 v127,  0              
 s_add_u32     m0,  s8, 0x46f0 * 0   
 buffer_load_dword  v2,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v3,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v10,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v11,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v12,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v13,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v14,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v15,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v16,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v17,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v18, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v19, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v20, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v21, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v22, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v23, s[16:19], 0 offen lds     
 s_waitcnt vmcnt(0)                                                
 s_barrier                                                         
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s15, 0                              
 s_add_u32     s16, s86, s16                                       
 s_addc_u32    s17, 0, s17                                         
 s_add_u32     m0,  s8, 0x46f0 * 1   
 buffer_load_dword  v2,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v3,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v10,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v11,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v12,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v13,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v14,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v15,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v16,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v17,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v18, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v19, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v20, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v21, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v22, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 buffer_load_dword  v23, s[16:19], 0 offen lds     
 s_waitcnt vmcnt(0)                                                
 s_barrier                                                         
 s_cmp_gt_i32  s41 2                                     
 s_cselect_b32 s86, s15, 0                              
 s_add_u32     s16, s86, s16                                       
 s_addc_u32    s17, 0, s17                                         
 ds_read_b128  acc[64:67],   v8  offset: 0x8e0 * 0 + 64 * 0   
 ds_read_b128  acc[68:71],   v8  offset: 0x8e0 * 0 + 64 * 1   
 ds_read_b128  acc[72:75],   v8  offset: 0x8e0 * 1 + 64 * 0   
 ds_read_b128  acc[76:79],   v8  offset: 0x8e0 * 1 + 64 * 1   
 ds_read_b128  acc[80:83],   v8  offset: 0x8e0 * 2 + 64 * 0   
 ds_read_b128  acc[84:87],   v8  offset: 0x8e0 * 2 + 64 * 1   
 ds_read_b128  acc[88:91],   v8  offset: 0x8e0 * 3 + 64 * 0   
 ds_read_b128  acc[92:95],   v8  offset: 0x8e0 * 3 + 64 * 1   
 ds_read_b128  acc[96:99],   v8  offset: 0x8e0 * 4 + 64 * 0   
 ds_read_b128  acc[100:103], v8  offset: 0x8e0 * 4 + 64 * 1   
 ds_read_b128  acc[104:107], v8  offset: 0x8e0 * 5 + 64 * 0   
 ds_read_b128  acc[108:111], v8  offset: 0x8e0 * 5 + 64 * 1   
 ds_read_b128  acc[112:115], v8  offset: 0x8e0 * 6 + 64 * 0   
 ds_read_b128  acc[116:119], v8  offset: 0x8e0 * 6 + 64 * 1   
 ds_read_b128  acc[120:123], v8  offset: 0x8e0 * 7 + 64 * 0   
 ds_read_b128  acc[124:127], v8  offset: 0x8e0 * 7 + 64 * 1   
 s_waitcnt lgkmcnt(0)                                              
 s_barrier                                                         
 buffer_load_dwordx4  acc[0:3],   v4, s[20:23], 0 offen offset: 0x400 * 0       
 buffer_load_dwordx4  acc[4:7],   v4, s[20:23], 0 offen offset: 0x400 * 1       
 buffer_load_dwordx4  acc[8:11],  v5, s[20:23], 0 offen offset: 0x400 * 0       
 buffer_load_dwordx4  acc[12:15], v5, s[20:23], 0 offen offset: 0x400 * 1       
 s_waitcnt vmcnt(0)                                                
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s43, 0                              
 s_add_u32     s20, s86, s20                                       
 s_addc_u32    s21, 0, s21                                         
 buffer_load_dword    v160, v1,  s[24:27], 0 offen offset: 64 * 0    
 buffer_load_dword    v161, v1,  s[24:27], 0 offen offset: 64 * 1    
 buffer_load_dword    v162, v1,  s[24:27], 0 offen offset: 64 * 2    
 buffer_load_dword    v163, v1,  s[24:27], 0 offen offset: 64 * 3    
 buffer_load_dword    v164, v1,  s[24:27], 0 offen offset: 64 * 4    
 buffer_load_dword    v165, v1,  s[24:27], 0 offen offset: 64 * 5    
 buffer_load_dword    v166, v1,  s[24:27], 0 offen offset: 64 * 6    
 buffer_load_dword    v167, v1,  s[24:27], 0 offen offset: 64 * 7    
 s_waitcnt vmcnt(0)                                                
 s_load_dword         s30, s[28:29],  0                            
 s_waitcnt lgkmcnt(0)                                              
 s_add_u32     m0,  s8, 0x46f0 * 0   
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s10, 0                                
 s_add_u32     s24, s86, s24                                       
 s_addc_u32    s25, 0, s25                                         
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s39, 0                                
 s_add_u32     s28, s86, s28                                       
 s_addc_u32    s29, 0, s29                                         
L_start0:                                                             
 s_waitcnt vmcnt(0)                                                
 s_barrier                                                         
 s_waitcnt lgkmcnt(0)                                              
 v_mov_b32  v176, s30                                              
 v_mul_f32  v160, v160, v176                                       
 v_mul_f32  v161, v161, v176                                       
 v_mul_f32  v162, v162, v176                                       
 v_mul_f32  v163, v163, v176                                       
 v_mul_f32  v164, v164, v176                                       
 v_mul_f32  v165, v165, v176                                       
 v_mul_f32  v166, v166, v176                                       
 v_mul_f32  v167, v167, v176
 s_load_dword         s31, s[28:29],  0                            
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[64:65], 0                 
 buffer_load_dwordx4  acc[16:19], v4, s[20:23], 0 offen offset: 0x400 * 0       
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[66:67], v[192:195]        
 buffer_load_dwordx4  acc[20:23], v4, s[20:23], 0 offen offset: 0x400 * 1       
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[68:69], v[192:195]        
 buffer_load_dwordx4  acc[24:27], v5, s[20:23], 0 offen offset: 0x400 * 0       
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[70:71], v[192:195]        
 buffer_load_dwordx4  acc[28:31], v5, s[20:23], 0 offen offset: 0x400 * 1       
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[64:65], 0           
 v_fmac_f32 v64,  v192, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[66:67], v[196:199]        
 v_fmac_f32 v65,  v193, v160                                 
 
;;v_accvgpr_read v64, acc1
;;s_nop 4
;;v_cvt_pk_f32_fp8 v[64:65], v64
;;v_mov_b32 v64, v176

v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[68:69], v[196:199]        
 v_fmac_f32 v66,  v194, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[70:71], v[196:199]        
 buffer_load_dword    v169, v1,  s[24:27], 0 offen offset: 64 * 1    
 v_fmac_f32 v67,  v195, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[72:73], 0                 
 v_fmac_f32 v68,  v196, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[74:75], v[192:195]        
 v_fmac_f32 v69,  v197, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[76:77], v[192:195]        
 buffer_load_dword    v170, v1,  s[24:27], 0 offen offset: 64 * 2    
 v_fmac_f32 v70,  v198, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[78:79], v[192:195]        
 v_fmac_f32 v71,  v199, v160                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[72:73], 0                 
 v_fmac_f32 v72,  v192, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[74:75], v[196:199]        
 buffer_load_dword    v171, v1,  s[24:27], 0 offen offset: 64 * 3    
 ds_read_b128  acc[128:131], v8  offset: 0x8e0 * 0 + 64 * 0 + 0x46f0  
 v_fmac_f32 v73,  v193, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[76:77], v[196:199]        
 v_fmac_f32 v74,  v194, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[78:79], v[196:199]        
 v_fmac_f32 v75,  v195, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[80:81], 0                 
 buffer_load_dword  v2,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                                         
 v_fmac_f32 v76,  v196, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[82:83], v[192:195]        
 buffer_load_dword    v173, v1,  s[24:27], 0 offen offset: 64 * 5    
 ds_read_b128  acc[132:135], v8  offset: 0x8e0 * 0 + 64 * 1 + 0x46f0  
 v_fmac_f32 v77,  v197, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[84:85], v[192:195]        
 buffer_load_dword  v3,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v78,  v198, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[86:87], v[192:195]        
 buffer_load_dword  v10,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v79,  v199, v161                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[80:81], 0                 
 buffer_load_dword  v11,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                                             
 v_fmac_f32 v80,  v192,v162                                  
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[82:83], v[196:199]        
 buffer_load_dword    v174, v1,  s[24:27], 0 offen offset: 64 * 6    
 ds_read_b128  acc[136:139], v8  offset: 0x8e0 * 1 + 64 * 0 + 0x46f0  
 v_fmac_f32 v81,  v193,v162                                  
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[84:85], v[196:199]        
 buffer_load_dword  v12,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v82,  v194,v162                                  
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[86:87], v[196:199]        
 buffer_load_dword  v13,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v83,  v195,v162                                  
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[88:89], 0                 
 buffer_load_dword  v14,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v84,  v196, v162                                              
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[90:91], v[192:195]        
 buffer_load_dword    v175, v1,  s[24:27], 0 offen offset: 64 * 7    
 ds_read_b128  acc[140:143], v8  offset: 0x8e0 * 1 + 64 * 1 + 0x46f0  
 v_fmac_f32 v85,  v197, v162                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[92:93], v[192:195]        
 buffer_load_dword  v15,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v86,  v198, v162                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[94:95], v[192:195]        
 buffer_load_dword  v16,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v87,  v199, v162                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[88:89], 0                 
 buffer_load_dword  v17,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                                     
 v_fmac_f32 v88,  v192, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[90:91], v[196:199]        
 ds_read_b128  acc[144:147], v8  offset: 0x8e0 * 2 + 64 * 0 + 0x46f0  
 v_fmac_f32 v89,  v193, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[92:93], v[196:199]        
 buffer_load_dword  v18, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v90,  v194, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[94:95], v[196:199]        
 buffer_load_dword  v19, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v91,  v195, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[96:97],   0               
 buffer_load_dword  v20, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                                     
 v_fmac_f32 v92,  v196, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[98:99],   v[192:195]      
 buffer_load_dword    v168, v1,  s[24:27], 0 offen offset: 64 * 0    
 ds_read_b128  acc[148:151], v8  offset: 0x8e0 * 2 + 64 * 1 + 0x46f0  
 v_fmac_f32 v93,  v197, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[100:101], v[192:195]      
 buffer_load_dword  v21, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v94,  v198, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[102:103], v[192:195]      
 buffer_load_dword  v22, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v95,  v199, v163                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[96:97],   0               
 buffer_load_dword  v23, s[16:19], 0 offen lds     
 v_fmac_f32 v96,  v192, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[98:99],   v[196:199]      
 ds_read_b128  acc[152:155], v8  offset: 0x8e0 * 3 + 64 * 0 + 0x46f0  
 v_fmac_f32 v97,  v193, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[100:101], v[196:199]      
 buffer_load_dword    v172, v1,  s[24:27], 0 offen offset: 64 * 4    
 v_fmac_f32 v98,  v194, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[102:103], v[196:199]      
 v_fmac_f32 v99,  v195, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[104:105], 0               
 v_fmac_f32 v100,  v196, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[106:107], v[192:195]      
 ds_read_b128  acc[156:159], v8  offset: 0x8e0 * 3 + 64 * 1 + 0x46f0  
 v_fmac_f32 v101,  v197, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[108:109], v[192:195]      
 v_fmac_f32 v102,  v198, v164                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[110:111], v[192:195]      
 v_fmac_f32 v103,  v199, v164                                 
 s_waitcnt lgkmcnt(8)                                              
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[104:105], 0               
 v_fmac_f32 v104, v192, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[106:107], v[196:199]      
 ds_read_b128  acc[160:163], v8  offset: 0x8e0 * 4 + 64 * 0 + 0x46f0  
 v_fmac_f32 v105, v193, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[108:109], v[196:199]      
 v_fmac_f32 v106, v194, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[110:111], v[196:199]      
 ds_read_b128  acc[164:167], v8  offset: 0x8e0 * 4 + 64 * 1 + 0x46f0  
 v_fmac_f32 v107, v195, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[112:113], 0               
 v_fmac_f32 v108, v196, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[114:115], v[192:195]      
 ds_read_b128  acc[168:171], v8  offset: 0x8e0 * 5 + 64 * 0 + 0x46f0  
 v_fmac_f32 v109, v197, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[116:117], v[192:195]      
 v_fmac_f32 v110, v198, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[118:119], v[192:195]      
 ds_read_b128  acc[172:175], v8  offset: 0x8e0 * 5 + 64 * 1 + 0x46f0  
 v_fmac_f32 v111, v199, v165                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[112:113], 0               
 v_fmac_f32 v112, v192, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[114:115], v[196:199]      
 ds_read_b128  acc[176:179], v8  offset: 0x8e0 * 6 + 64 * 0 + 0x46f0  
 v_fmac_f32 v113, v193, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[116:117], v[196:199]      
 v_fmac_f32 v114, v194, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[118:119], v[196:199]      
 ds_read_b128  acc[180:183], v8  offset: 0x8e0 * 6 + 64 * 1 + 0x46f0  
 v_fmac_f32 v115, v195, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[0:1],   acc[120:121], 0               
 v_fmac_f32 v116, v196, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[2:3],   acc[122:123], v[192:195]      
 ds_read_b128  acc[184:187], v8  offset: 0x8e0 * 7 + 64 * 0 + 0x46f0  
 v_fmac_f32 v117, v197, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[4:5],   acc[124:125], v[192:195]      
 v_fmac_f32 v118, v198, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[6:7],   acc[126:127], v[192:195]      
 ds_read_b128  acc[188:191], v8  offset: 0x8e0 * 7 + 64 * 1 + 0x46f0  
 v_fmac_f32 v119, v199, v166                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[8:9],   acc[120:121], 0               
 v_fmac_f32 v120, v192, v167                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[10:11], acc[122:123], v[196:199]      
 v_fmac_f32 v121, v193, v167                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[12:13], acc[124:125], v[196:199]      
 v_fmac_f32 v122, v194, v167                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[14:15], acc[126:127], v[196:199]      
 v_fmac_f32 v123, v195, v167                                 
 s_add_u32     m0,  s8, 0x46f0 * 1   
 v_fmac_f32 v124, v196, v167                                 
 v_fmac_f32 v125, v197, v167                                 
 v_fmac_f32 v126, v198, v167                                 
 v_fmac_f32 v127, v199, v167                                 
  s_sub_i32     s41, s41, 1                    
  s_cmp_gt_i32  s41 0                                    
  s_cbranch_scc0 L_end0                                           
 s_cmp_gt_i32  s41 2                                     
 s_cselect_b32 s86, s15, 0                              
 s_add_u32     s16, s86, s16                                       
 s_addc_u32    s17, 0, s17                                         
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s43, 0                              
 s_add_u32     s20, s86, s20                                       
 s_addc_u32    s21, 0, s21                                         
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s10, 0                                
 s_add_u32     s24, s86, s24                                       
 s_addc_u32    s25, 0, s25                                         
 s_cmp_gt_i32  s41 1                                     
 s_cselect_b32 s86, s39, 0                                
 s_add_u32     s28, s86, s28                                       
 s_addc_u32    s29, 0, s29                                         
 s_waitcnt vmcnt(0)                                                
 s_barrier                                                         
 s_waitcnt lgkmcnt(0)                                              
 v_mov_b32  v176, s31                                              
 v_mul_f32  v168, v168, v176                                       
 v_mul_f32  v169, v169, v176                                       
 v_mul_f32  v170, v170, v176                                       
 v_mul_f32  v171, v171, v176                                       
 v_mul_f32  v172, v172, v176                                       
 v_mul_f32  v173, v173, v176                                       
 v_mul_f32  v174, v174, v176                                       
 v_mul_f32  v175, v175, v176                                       
 s_load_dword         s30, s[28:29],  0                            
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[128:129], 0             
 buffer_load_dwordx4  acc[0:3],   v4, s[20:23], 0 offen offset: 0x400 * 0       
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[130:131], v[192:195]    
 buffer_load_dwordx4  acc[4:7],   v4, s[20:23], 0 offen offset: 0x400 * 1       
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[132:133], v[192:195]    
 buffer_load_dwordx4  acc[8:11],  v5, s[20:23], 0 offen offset: 0x400 * 0       
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[134:135], v[192:195]    
 buffer_load_dwordx4  acc[12:15], v5, s[20:23], 0 offen offset: 0x400 * 1       
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[128:129], 0             
 v_fmac_f32 v64,  v192, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[130:131], v[196:199]    
 v_fmac_f32 v65,  v193, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[132:133], v[196:199]    
 v_fmac_f32 v66,  v194, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[134:135], v[196:199]    
 buffer_load_dword    v160, v1,  s[24:27], 0 offen offset: 64 * 0    
 v_fmac_f32 v67,  v195, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[136:137], 0             
 v_fmac_f32 v68,  v196, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[138:139], v[192:195]    
 v_fmac_f32 v69,  v197, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[140:141], v[192:195]    
 v_fmac_f32 v70,  v198, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[142:143], v[192:195]    
 buffer_load_dword    v161, v1,  s[24:27], 0 offen offset: 64 * 1    
 v_fmac_f32 v71,  v199, v168                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[136:137], 0    
 v_fmac_f32 v72,  v192, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[138:139], v[196:199]    
 buffer_load_dword    v162, v1,  s[24:27], 0 offen offset: 64 * 2    
 ds_read_b128  acc[64:67],   v8  offset: 0x8e0 * 0 + 64 * 0   
 v_fmac_f32 v73,  v193, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[140:141], v[196:199]    
 v_fmac_f32 v74,  v194, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[142:143], v[196:199]    
 v_fmac_f32 v75,  v195, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[144:145], 0             
 buffer_load_dword  v2,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v76,  v196, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[146:147], v[192:195]    
 buffer_load_dword    v163, v1,  s[24:27], 0 offen offset: 64 * 3    
 ds_read_b128  acc[68:71],   v8  offset: 0x8e0 * 0 + 64 * 1   
 v_fmac_f32 v77,  v197, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[148:149], v[192:195]    
 buffer_load_dword  v3,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v78,  v198, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[150:151], v[192:195]    
 buffer_load_dword  v10,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v79,  v199, v169                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[144:145], 0             
 buffer_load_dword  v11,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v80,  v192, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[146:147], v[196:199]    
 buffer_load_dword    v164, v1,  s[24:27], 0 offen offset: 64 * 4    
 ds_read_b128  acc[72:75],   v8  offset: 0x8e0 * 1 + 64 * 0   
 v_fmac_f32 v81,  v193, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[148:149], v[196:199]    
 buffer_load_dword  v12,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v82,  v194, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[150:151], v[196:199]    
 buffer_load_dword  v13,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v83,  v195, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[152:153], 0             
 buffer_load_dword  v14,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v84,  v196, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[154:155], v[192:195]    
 buffer_load_dword    v165, v1,  s[24:27], 0 offen offset: 64 * 5    
 ds_read_b128  acc[76:79],   v8  offset: 0x8e0 * 1 + 64 * 1   
 v_fmac_f32 v85,  v197, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[156:157], v[192:195]    
 buffer_load_dword  v15,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v86,  v198, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[158:159], v[192:195]    
 buffer_load_dword  v16,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v87,  v199, v170                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[152:153], 0             
 buffer_load_dword  v17,  s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v88,  v192, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[154:155], v[196:199]    
 ds_read_b128  acc[80:83],   v8  offset: 0x8e0 * 2 + 64 * 0   
 v_fmac_f32 v89,  v193, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[156:157], v[196:199]    
 buffer_load_dword  v18, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v90,  v194, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[158:159], v[196:199]    
 buffer_load_dword  v19, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v91,  v195, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[160:161], 0             
 buffer_load_dword  v20, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v92,  v196, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[162:163], v[192:195]    
 buffer_load_dword    v166, v1,  s[24:27], 0 offen offset: 64 * 6    
 ds_read_b128  acc[84:87],   v8  offset: 0x8e0 * 2 + 64 * 1   
 v_fmac_f32 v93,  v197, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[164:165], v[192:195]    
 buffer_load_dword  v21, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v94,  v198, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[166:167], v[192:195]    
 buffer_load_dword  v22, s[16:19], 0 offen lds     
 s_add_u32     m0,  s14, m0                           
 v_fmac_f32 v95,  v199, v171                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[160:161], 0             
 buffer_load_dword  v23, s[16:19], 0 offen lds     
 v_fmac_f32 v96,  v192, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[162:163], v[196:199]    
 ds_read_b128  acc[88:91],   v8  offset: 0x8e0 * 3 + 64 * 0   
 v_fmac_f32 v97,  v193, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[164:165], v[196:199]    
 v_fmac_f32 v98,  v194, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[166:167], v[196:199]    
 buffer_load_dword    v167, v1,  s[24:27], 0 offen offset: 64 * 7    
 v_fmac_f32 v99,  v195, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[168:169], 0             
 v_fmac_f32 v100,  v196, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[170:171], v[192:195]    
 ds_read_b128  acc[92:95],   v8  offset: 0x8e0 * 3 + 64 * 1   
 v_fmac_f32 v101,  v197, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[172:173], v[192:195]    
 v_fmac_f32 v102,  v198, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[174:175], v[192:195]    
 s_waitcnt lgkmcnt(8)                                              
 v_fmac_f32 v103,  v199, v172                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[168:169], 0             
 v_fmac_f32 v104, v192, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[170:171], v[196:199]    
 ds_read_b128  acc[96:99],   v8  offset: 0x8e0 * 4 + 64 * 0   
 v_fmac_f32 v105, v193, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[172:173], v[196:199]    
 v_fmac_f32 v106, v194, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[174:175], v[196:199]    
 ds_read_b128  acc[100:103], v8  offset: 0x8e0 * 4 + 64 * 1   
 v_fmac_f32 v107, v195, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[176:177], 0             
 v_fmac_f32 v108, v196, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[178:179], v[192:195]    
 ds_read_b128  acc[104:107], v8  offset: 0x8e0 * 5 + 64 * 0   
 v_fmac_f32 v109, v197, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[180:181], v[192:195]    
 v_fmac_f32 v110, v198, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[182:183], v[192:195]    
 ds_read_b128  acc[108:111], v8  offset: 0x8e0 * 5 + 64 * 1   
 v_fmac_f32 v111, v199, v173                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[176:177], 0             
 v_fmac_f32 v112, v192, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[178:179], v[196:199]    
 ds_read_b128  acc[112:115], v8  offset: 0x8e0 * 6 + 64 * 0   
 v_fmac_f32 v113, v193, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[180:181], v[196:199]    
 v_fmac_f32 v114, v194, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[182:183], v[196:199]    
 ds_read_b128  acc[116:119], v8  offset: 0x8e0 * 6 + 64 * 1   
 v_fmac_f32 v115, v195, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[16:17],   acc[184:185], 0             
 v_fmac_f32 v116, v196, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[18:19],   acc[186:187], v[192:195]    
 ds_read_b128  acc[120:123], v8  offset: 0x8e0 * 7 + 64 * 0   
 v_fmac_f32 v117, v197, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[20:21],   acc[188:189], v[192:195]    
 v_fmac_f32 v118, v198, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[192:195], acc[22:23],   acc[190:191], v[192:195]    
 ds_read_b128  acc[124:127], v8  offset: 0x8e0 * 7 + 64 * 1   
 v_fmac_f32 v119, v199, v174                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[24:25],   acc[184:185], 0             
 v_fmac_f32 v120, v192, v175                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[26:27],   acc[186:187], v[196:199]    
 v_fmac_f32 v121, v193, v175                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[28:29],   acc[188:189], v[196:199]    
 v_fmac_f32 v122, v194, v175                                 
v_mfma_f32_16x16x32_fp8_fp8 v[196:199], acc[30:31],   acc[190:191], v[196:199]    
 v_fmac_f32 v123, v195, v175                                 
 s_add_u32     m0,  s8, 0x46f0 * 0   
 v_fmac_f32 v124, v196, v175                                 
 v_fmac_f32 v125, v197, v175                                 
 v_fmac_f32 v126, v198, v175                                 
 v_fmac_f32 v127, v199, v175                                 
 s_cmp_gt_i32  s41 3                                     
 s_cselect_b32 s86, s15, 0                              
 s_add_u32     s16, s86, s16                                       
 s_addc_u32    s17, 0, s17                                         
 s_cmp_gt_i32  s41 2                                     
 s_cselect_b32 s86, s43, 0                              
 s_add_u32     s20, s86, s20                                       
 s_addc_u32    s21, 0, s21                                         
 s_cmp_gt_i32  s41 2                                     
 s_cselect_b32 s86, s10, 0                                
 s_add_u32     s24, s86, s24                                       
 s_addc_u32    s25, 0, s25                                         
 s_cmp_gt_i32  s41 2                                     
 s_cselect_b32 s86, s39, 0                                
 s_add_u32     s28, s86, s28                                       
 s_addc_u32    s29, 0, s29                                         
  s_sub_i32     s41, s41, 1                    
  s_cmp_gt_i32  s41 0                                    
  s_cbranch_scc0 L_end0                                           
  s_branch     L_start0                                           
L_end0:                                                               
  s_nop 2 

	;;#ASMEND
	v_or_b32_e32 v1, s0, v0
	v_and_or_b32 v0, v6, 60, s1
	v_mad_u64_u32 v[0:1], s[0:1], s9, v1, v[0:1]
	v_cvt_f16_f32_e32 v4, v64
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v66
	v_cvt_f16_f32_e32 v6, v67
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v65
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v68
	v_cvt_f16_f32_e32 v7, v70
	v_cvt_f16_f32_e32 v8, v71
	v_cvt_f16_f32_e32 v9, v69
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	s_lshl_b32 s0, s9, 4
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v4, v72
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v74
	v_cvt_f16_f32_e32 v6, v75
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v73
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v76
	v_cvt_f16_f32_e32 v7, v78
	v_cvt_f16_f32_e32 v8, v79
	v_cvt_f16_f32_e32 v9, v77
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v4, v80
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v82
	v_cvt_f16_f32_e32 v6, v83
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v81
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v84
	v_cvt_f16_f32_e32 v7, v86
	v_cvt_f16_f32_e32 v8, v87
	v_cvt_f16_f32_e32 v9, v85
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v4, v88
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v90
	v_cvt_f16_f32_e32 v6, v91
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v89
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v92
	v_cvt_f16_f32_e32 v7, v94
	v_cvt_f16_f32_e32 v8, v95
	v_cvt_f16_f32_e32 v9, v93
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v4, v96
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v98
	v_cvt_f16_f32_e32 v6, v99
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v97
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v100
	v_cvt_f16_f32_e32 v7, v102
	v_cvt_f16_f32_e32 v8, v103
	v_cvt_f16_f32_e32 v9, v101
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v4, v104
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v106
	v_cvt_f16_f32_e32 v6, v107
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v105
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v108
	v_cvt_f16_f32_e32 v7, v110
	v_cvt_f16_f32_e32 v8, v111
	v_cvt_f16_f32_e32 v9, v109
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v4, v112
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v5, v114
	v_cvt_f16_f32_e32 v6, v115
	v_lshl_add_u64 v[2:3], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v1, v113
	v_pack_b32_f16 v5, v5, v6
	v_cvt_f16_f32_e32 v6, v116
	v_cvt_f16_f32_e32 v7, v118
	v_cvt_f16_f32_e32 v8, v119
	v_cvt_f16_f32_e32 v9, v117
	v_pack_b32_f16 v4, v4, v1
	global_store_dwordx2 v[2:3], v[4:5], off
	v_pack_b32_f16 v5, v7, v8
	v_pack_b32_f16 v4, v6, v9
	global_store_dwordx2 v[2:3], v[4:5], off offset:128
	v_add_u32_e32 v0, s0, v0
	v_cvt_f16_f32_e32 v2, v120
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_f16_f32_e32 v3, v122
	v_cvt_f16_f32_e32 v4, v123
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	v_cvt_f16_f32_e32 v5, v121
	v_pack_b32_f16 v3, v3, v4
	v_cvt_f16_f32_e32 v4, v124
	v_cvt_f16_f32_e32 v6, v126
	v_cvt_f16_f32_e32 v7, v127
	v_cvt_f16_f32_e32 v8, v125
	v_pack_b32_f16 v2, v2, v5
	global_store_dwordx2 v[0:1], v[2:3], off
	v_pack_b32_f16 v3, v6, v7
	v_pack_b32_f16 v2, v4, v8
	global_store_dwordx2 v[0:1], v[2:3], off offset:128
	s_endpgm


	.rodata
	.p2align	6
	.amdhsa_kernel flatmm_uk_gfx9_f16f8_128x128x128_1x4x1_16x16x32
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
  - .name:           flatmm_uk_gfx9_f16f8_128x128x128_1x4x1_16x16x32
    .symbol:         flatmm_uk_gfx9_f16f8_128x128x128_1x4x1_16x16x32.kd
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
