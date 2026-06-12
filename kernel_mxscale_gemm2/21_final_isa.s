	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel_mxscale_gemm2
	.p2align	8
	.type	kernel_mxscale_gemm2,@function
kernel_mxscale_gemm2:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[2:3], s[0:1], 0xf0 nv
	s_mov_b32 s28, 1
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
	s_bfe_u32 s4, ttmp6, 0x4000c
	s_and_b32 s5, ttmp6, 15
	s_add_co_i32 s4, s4, 1
	s_mov_b32 s47, 0
	s_mul_i32 s6, ttmp9, s4
	s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s5, s5, s6
	s_cmp_eq_u32 s4, 0
	s_mov_b32 s46, 0x1ffffff
	s_cselect_b32 s56, ttmp9, s5
	s_wait_kmcnt 0x0
	s_or_b64 s[44:45], s[2:3], 0xfe00000000000000
	s_lshl_b32 s5, s56, 2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mov_b32_e32 v1, s5
	buffer_load_b32 v1, v1, s[44:47], null offen
	s_wait_loadcnt 0x0
	v_cmp_gt_i32_e32 vcc_lo, 1, v1
	s_cbranch_vccnz .LBB0_4
	s_bfe_u32 s5, ttmp6, 0x40010
	s_load_b64 s[2:3], s[0:1], 0x28 nv
	s_add_co_i32 s5, s5, 1
	s_bfe_u32 s6, ttmp6, 0x40004
	s_mul_i32 s5, ttmp7, s5
	s_ashr_i32 s57, s56, 31
	s_add_co_i32 s6, s6, s5
	s_cmp_eq_u32 s4, 0
	s_movk_i32 s9, 0xc00
	s_cselect_b32 s50, ttmp7, s6
	s_bfe_u32 s6, ttmp8, 0x50019
	s_lshl_b64 s[4:5], s[56:57], 6
	s_and_b32 s69, s6, 3
	s_lshr_b32 s40, s6, 2
	s_lshl_b32 s33, s69, 4
	s_lshl_b32 s70, s40, 8
	s_or_b32 s4, s4, s33
	s_mul_i32 s66, s69, 0x1100
	s_mul_u64 s[4:5], s[4:5], 0xc00
	s_add_co_i32 s66, s66, s70
	s_or_b32 s4, s4, s70
	s_add_co_i32 s29, s66, 0xd000
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[34:35], s[4:5], s[2:3]
	s_mov_b64 s[12:13], s[28:29]
	s_or_b32 s71, s35, 0x80000000
	s_mov_b32 s5, 0x1000000
	s_mov_b64 s[14:15], s[30:31]
	s_mov_b32 s13, s66
	s_mov_b32 s14, s34
	s_mov_b32 s15, s71
	s_mov_b32 s8, 16
	s_mov_b32 s6, 0x100000
	s_mov_b32 s4, 0x7500000
	s_mov_b32 s7, s5
	s_mov_b32 s10, s47
	s_mov_b32 s11, s47
	s_clause 0x2
	s_load_b64 s[52:53], s[0:1], 0x50 nv
	s_load_b64 s[36:37], s[0:1], 0x78 nv
	s_load_b64 s[58:59], s[0:1], 0xa0 nv
	tensor_load_to_lds s[12:15], s[4:11]
	s_ashr_i32 s51, s50, 31
	s_mul_u64 s[20:21], s[56:57], 0xc0
	s_lshl_b64 s[54:55], s[50:51], 8
	s_lshl_b32 s73, s69, 2
	s_lshr_b64 s[30:31], s[54:55], 4
	s_lshl_b32 s72, s40, 11
	s_add_nc_u64 s[60:61], s[30:31], s[20:21]
	s_lshl_b32 s75, s69, 13
	s_or_b32 s20, s60, s73
	s_mov_b32 s21, s61
	s_add_co_i32 s75, s75, s72
	s_mul_u64 s[24:25], s[20:21], 0x6000
	s_add_co_i32 s31, s75, 0x4400
	s_or_b32 s24, s24, s72
	s_brev_b32 s13, 16
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[44:45], s[24:25], s[52:53]
	s_mov_b64 s[20:21], s[28:29]
	s_or_b32 s74, s45, 0x80000000
	s_mov_b64 s[22:23], s[30:31]
	s_mov_b32 s16, 4
	s_movk_i32 s17, 0x6000
	s_mov_b32 s14, 0x40000
	s_mov_b32 s12, s47
	s_mov_b32 s18, s47
	s_mov_b32 s19, s47
	s_mov_b32 s15, s13
	s_mov_b32 s21, s31
	s_mov_b32 s22, s44
	s_mov_b32 s23, s74
	s_lshl_b64 s[38:39], s[56:57], 4
	s_lshl_b32 s46, s40, 5
	s_or_b32 s38, s38, s73
	s_lshl_b32 s77, s69, 7
	s_mul_u64 s[38:39], s[38:39], 0x180
	s_add_co_i32 s77, s77, s46
	s_add_nc_u64 s[48:49], s[38:39], s[36:37]
	s_or_b32 s67, s77, 0xc400
	s_add_nc_u64 s[62:63], s[48:49], s[46:47]
	s_mov_b64 s[38:39], s[30:31]
	s_or_b32 s76, s63, 0x80000000
	s_mov_b64 s[36:37], s[28:29]
	s_movk_i32 s25, 0x180
	s_mov_b32 s26, s47
	s_mov_b32 s27, s47
	s_mov_b32 s24, s16
	s_mov_b32 s37, s67
	s_mov_b32 s38, s62
	s_mov_b32 s39, s76
	s_mul_u64 s[64:65], s[56:57], 0x300
	s_lshl_b64 s[50:51], s[50:51], 6
	s_mul_i32 s63, s69, 0x180
	s_add_nc_u64 s[50:51], s[50:51], s[64:65]
	s_add_co_i32 s79, s77, s63
	s_or_b32 s50, s50, s33
	s_add_co_i32 s68, s79, 0xc610
	s_mul_u64 s[50:51], s[50:51], 0x180
	s_mov_b64 s[82:83], s[30:31]
	s_add_nc_u64 s[50:51], s[50:51], s[58:59]
	s_mov_b64 s[80:81], s[28:29]
	s_add_nc_u64 s[64:65], s[50:51], s[46:47]
	s_mov_b32 s42, s47
	s_or_b32 s78, s65, 0x80000000
	s_mov_b32 s43, s47
	s_mov_b32 s40, s8
	s_mov_b32 s41, s25
	s_mov_b32 s81, s68
	s_mov_b32 s82, s64
	s_mov_b32 s83, s78
	v_dual_lshrrev_b32 v1, 5, v0 :: v_dual_bitop2_b32 v220, 15, v0 bitop3:0x40
	v_bfe_u32 v2, v0, 4, 1
	v_dual_mov_b32 v80, 0 :: v_dual_bitop2_b32 v3, 63, v0 bitop3:0x40
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshlrev_b32 v228, 13, v1 :: v_dual_lshlrev_b32 v0, 4, v220
	v_dual_lshlrev_b32 v4, 8, v2 :: v_dual_lshlrev_b32 v221, 4, v2
	v_dual_lshlrev_b32 v5, 5, v220 :: v_dual_lshlrev_b32 v2, 2, v2
	v_readfirstlane_b32 s33, v1
	v_dual_mov_b32 v81, v80 :: v_dual_lshlrev_b32 v6, 9, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_or3_b32 v227, v4, v0, v228
	v_dual_mov_b32 v1, s61 :: v_dual_bitop2_b32 v4, v2, v5 bitop3:0x54
	v_or_b32_e32 v0, s60, v220
	s_add_co_i32 s63, s75, 0x11400
	s_or_b32 s61, s77, 0x19400
	s_add_co_i32 s60, s79, 0x19610
	s_cmp_gt_u32 s34, 0xfffffeff
	s_mul_i32 s84, s69, 0xc000
	v_mad_u32_u24 v223, 0x110, v220, v221
	v_or3_b32 v229, v6, v5, v2
	v_dual_mov_b32 v82, v80 :: v_dual_mov_b32 v83, v80
	v_dual_mov_b32 v84, v80 :: v_dual_mov_b32 v85, v80
	v_dual_mov_b32 v86, v80 :: v_dual_mov_b32 v87, v80
	v_dual_mov_b32 v200, v80 :: v_dual_mov_b32 v201, v80
	v_dual_mov_b32 v202, v80 :: v_dual_mov_b32 v203, v80
	v_dual_mov_b32 v204, v80 :: v_dual_mov_b32 v205, v80
	v_dual_mov_b32 v206, v80 :: v_dual_mov_b32 v207, v80
	v_dual_mov_b32 v192, v80 :: v_dual_mov_b32 v193, v80
	v_dual_mov_b32 v194, v80 :: v_dual_mov_b32 v195, v80
	v_dual_mov_b32 v196, v80 :: v_dual_mov_b32 v197, v80
	v_dual_mov_b32 v198, v80 :: v_dual_mov_b32 v199, v80
	v_dual_mov_b32 v184, v80 :: v_dual_mov_b32 v185, v80
	v_dual_mov_b32 v186, v80 :: v_dual_mov_b32 v187, v80
	v_dual_mov_b32 v188, v80 :: v_dual_mov_b32 v189, v80
	v_dual_mov_b32 v190, v80 :: v_dual_mov_b32 v191, v80
	v_dual_mov_b32 v176, v80 :: v_dual_mov_b32 v177, v80
	v_dual_mov_b32 v178, v80 :: v_dual_mov_b32 v179, v80
	v_dual_mov_b32 v180, v80 :: v_dual_mov_b32 v181, v80
	v_dual_mov_b32 v182, v80 :: v_dual_mov_b32 v183, v80
	v_dual_mov_b32 v168, v80 :: v_dual_mov_b32 v169, v80
	v_dual_mov_b32 v170, v80 :: v_dual_mov_b32 v171, v80
	v_dual_mov_b32 v172, v80 :: v_dual_mov_b32 v173, v80
	v_dual_mov_b32 v174, v80 :: v_dual_mov_b32 v175, v80
	v_dual_mov_b32 v160, v80 :: v_dual_mov_b32 v161, v80
	v_dual_mov_b32 v162, v80 :: v_dual_mov_b32 v163, v80
	v_dual_mov_b32 v164, v80 :: v_dual_mov_b32 v165, v80
	v_dual_mov_b32 v166, v80 :: v_dual_mov_b32 v167, v80
	v_dual_mov_b32 v152, v80 :: v_dual_mov_b32 v153, v80
	v_dual_mov_b32 v154, v80 :: v_dual_mov_b32 v155, v80
	v_dual_mov_b32 v156, v80 :: v_dual_mov_b32 v157, v80
	v_dual_mov_b32 v158, v80 :: v_dual_mov_b32 v159, v80
	v_dual_mov_b32 v144, v80 :: v_dual_mov_b32 v145, v80
	v_dual_mov_b32 v146, v80 :: v_dual_mov_b32 v147, v80
	v_dual_mov_b32 v148, v80 :: v_dual_mov_b32 v149, v80
	v_dual_mov_b32 v150, v80 :: v_dual_mov_b32 v151, v80
	v_dual_mov_b32 v136, v80 :: v_dual_mov_b32 v137, v80
	v_dual_mov_b32 v138, v80 :: v_dual_mov_b32 v139, v80
	v_dual_mov_b32 v140, v80 :: v_dual_mov_b32 v141, v80
	v_dual_mov_b32 v142, v80 :: v_dual_mov_b32 v143, v80
	v_dual_mov_b32 v128, v80 :: v_dual_mov_b32 v129, v80
	v_dual_mov_b32 v130, v80 :: v_dual_mov_b32 v131, v80
	v_dual_mov_b32 v132, v80 :: v_dual_mov_b32 v133, v80
	v_dual_mov_b32 v134, v80 :: v_dual_mov_b32 v135, v80
	v_dual_mov_b32 v120, v80 :: v_dual_mov_b32 v121, v80
	v_dual_mov_b32 v122, v80 :: v_dual_mov_b32 v123, v80
	v_dual_mov_b32 v124, v80 :: v_dual_mov_b32 v125, v80
	v_dual_mov_b32 v126, v80 :: v_dual_mov_b32 v127, v80
	v_dual_mov_b32 v112, v80 :: v_dual_mov_b32 v113, v80
	v_dual_mov_b32 v114, v80 :: v_dual_mov_b32 v115, v80
	v_dual_mov_b32 v116, v80 :: v_dual_mov_b32 v117, v80
	v_dual_mov_b32 v118, v80 :: v_dual_mov_b32 v119, v80
	v_dual_mov_b32 v104, v80 :: v_dual_mov_b32 v105, v80
	v_dual_mov_b32 v106, v80 :: v_dual_mov_b32 v107, v80
	v_dual_mov_b32 v108, v80 :: v_dual_mov_b32 v109, v80
	v_dual_mov_b32 v110, v80 :: v_dual_mov_b32 v111, v80
	v_dual_mov_b32 v96, v80 :: v_dual_mov_b32 v97, v80
	v_dual_mov_b32 v98, v80 :: v_dual_mov_b32 v99, v80
	v_dual_mov_b32 v100, v80 :: v_dual_mov_b32 v101, v80
	v_dual_mov_b32 v102, v80 :: v_dual_mov_b32 v103, v80
	v_dual_mov_b32 v88, v80 :: v_dual_mov_b32 v89, v80
	v_dual_mov_b32 v90, v80 :: v_dual_mov_b32 v91, v80
	v_dual_mov_b32 v92, v80 :: v_dual_mov_b32 v93, v80
	v_dual_mov_b32 v94, v80 :: v_dual_mov_b32 v95, v80
	v_or_b32_e32 v230, 0xc400, v4
	v_or_b32_e32 v225, 0x19400, v4
	v_add_nc_u32_e32 v226, 0x19610, v229
	v_add_nc_u32_e32 v222, 0xd000, v223
	s_mov_b64 s[58:59], 0
	tensor_load_to_lds s[20:23], s[12:19]
	s_mov_b32 s21, 0x200000
	s_mov_b32 s20, s47
	s_mov_b32 s22, s14
	s_mov_b32 s23, s21
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[36:39], s[20:27]
	s_mov_b32 s36, s47
	s_mov_b32 s37, s21
	s_mov_b32 s38, s6
	s_mov_b32 s39, s21
	tensor_load_to_lds s[80:83], s[36:43]
	s_mul_i32 s82, s56, 0xc0
	s_mul_i32 s83, s56, 0x30000
	s_mul_u64 s[56:57], s[56:57], 0x30000
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[80:81], s[2:3], s[56:57]
	s_add_co_ci_u32 s3, s71, 0
	s_cmp_gt_u32 s44, 0xfffff7ff
	v_mov_b32_e32 v231, s3
	v_mul_u64_e32 v[216:217], 0x6000, v[0:1]
	s_add_co_ci_u32 s3, s74, 0
	s_add_co_i32 s69, s62, 32
	s_cmp_gt_u32 s62, 0xffffffdf
	v_mov_b32_e32 v232, s3
	s_add_co_ci_u32 s3, s76, 0
	s_add_co_i32 s65, s64, 32
	s_cmp_gt_u32 s64, 0xffffffdf
	v_dual_mov_b32 v233, s3 :: v_dual_add_nc_u32 v224, 0x11400, v227
	s_add_co_ci_u32 s3, s78, 0
	s_or_b32 s62, s82, s73
	v_mad_nc_u64_u32 v[218:219], 0xc00, v3, s[80:81]
	s_add_co_i32 s64, s83, s84
	s_add_co_i32 s30, s62, s30
	s_or_b32 s62, s64, s70
	v_mov_b32_e32 v234, s3
	s_mul_i32 s3, s30, 0x6000
	s_add_co_i32 s2, s62, s2
	s_or_b32 s30, s3, s72
	s_add_co_i32 s3, s2, 0x100
	s_add_co_i32 s62, s30, 0x800
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB0_2:
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s64, s62, s52
	s_add_co_i32 s30, s3, s58
	s_barrier_wait -1
	ds_load_b128 v[236:239], v227 offset:17408
	ds_load_b128 v[240:243], v227 offset:17920
	ds_load_b128 v[70:73], v230
	ds_load_b128 v[0:3], v223
	ds_load_b128 v[4:7], v223 offset:32
	ds_load_b128 v[8:11], v223 offset:64
	ds_load_b128 v[12:15], v223 offset:96
	ds_load_b128 v[74:77], v229 offset:50704
	ds_load_b128 v[244:247], v227 offset:19456
	ds_load_b128 v[248:251], v227 offset:19968
	ds_load_b128 v[252:255], v227 offset:21504
	s_set_vgpr_msb 64
	ds_load_b128 v[0:3] /*v[256:259]*/, v227 offset:22016
	ds_load_b128 v[4:7] /*v[260:263]*/, v227 offset:23552
	ds_load_b128 v[8:11] /*v[264:267]*/, v227 offset:24064
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[16:19], v223 offset:4352
	ds_load_b128 v[20:23], v223 offset:4384
	ds_load_b128 v[24:27], v223 offset:4416
	ds_load_b128 v[28:31], v223 offset:4448
	ds_load_b128 v[208:211], v229 offset:50720
	ds_load_b128 v[212:215], v230 offset:16
	s_set_vgpr_msb 64
	ds_load_b128 v[12:15] /*v[268:271]*/, v227 offset:18432
	ds_load_b128 v[16:19] /*v[272:275]*/, v227 offset:18944
	ds_load_b128 v[20:23] /*v[276:279]*/, v227 offset:20480
	ds_load_b128 v[24:27] /*v[280:283]*/, v227 offset:20992
	ds_load_b128 v[28:31] /*v[284:287]*/, v227 offset:22528
	ds_load_b128 v[32:35] /*v[288:291]*/, v227 offset:23040
	ds_load_b128 v[36:39] /*v[292:295]*/, v227 offset:24576
	ds_load_b128 v[40:43] /*v[296:299]*/, v227 offset:25088
	s_set_vgpr_msb 0x4000
	s_wait_dscnt 0x14
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[236:243], v[0:15], v[80:87], v74, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[244:251], v[0:15], v[200:207], v75, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[252:259], v[0:15], v[192:199], v76, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[4:11] /*v[260:267]*/, v[0:15], v[184:191], v77, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[4:11] /*v[260:267]*/, v[16:31], v[152:159], v77, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[252:259], v[16:31], v[160:167], v76, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[244:251], v[16:31], v[168:175], v75, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[236:243], v[16:31], v[176:183], v74, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[0:1], s[28:29]
	v_mov_b64_e32 v[2:3], s[30:31]
	v_mov_b32_e32 v3, v231
	v_add_nc_u64_e32 v[8:9], s[52:53], v[216:217]
	s_cmp_gt_u32 s30, 0xfffffeff
	s_cselect_b32 vcc_lo, -1, 0
	v_readfirstlane_b32 s72, v0
	v_readfirstlane_b32 s73, v1
	v_readfirstlane_b32 s74, v2
	v_readfirstlane_b32 s75, v3
	v_mov_b32_e32 v3, v232
	v_add_nc_u64_e32 v[0:1], s[58:59], v[218:219]
	s_set_vgpr_msb 64
	v_add_nc_u64_e32 v[46:47] /*v[302:303]*/, 0x2000, v[8:9]
	s_cmp_gt_u32 s64, 0xfffff7ff
	tensor_load_to_lds s[72:75], s[4:11]
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s63
	s_mov_b32 s74, s64
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_readfirstlane_b32 s75, v3
	v_mov_b32_e32 v3, v233
	s_set_vgpr_msb 64
	v_add_nc_u64_e32 v[44:45] /*v[300:301]*/, 0x400, v[0:1]
	s_cselect_b32 s2, -1, 0
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s73, v5
	v_readfirstlane_b32 s74, v6
	s_cmp_gt_u32 s69, 0xffffffdf
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e64 v235, 0, 1, vcc_lo
	s_set_vgpr_msb 64
	v_cndmask_b32_e64 v48 /*v304*/, 0, 1, s2
	tensor_load_to_lds s[72:75], s[12:19]
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s61
	s_mov_b32 s74, s69
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_readfirstlane_b32 s75, v3
	v_mov_b32_e32 v3, v234
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s73, v5
	v_readfirstlane_b32 s74, v6
	tensor_load_to_lds s[72:75], s[20:27]
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s60
	s_mov_b32 s74, s65
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_readfirstlane_b32 s75, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s73, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s74, v6
	tensor_load_to_lds s[72:75], s[36:43]
	s_set_vgpr_msb 1
	global_prefetch_b8 v[44:45] /*v[300:301]*/, off offset:-256 scope:SCOPE_SE
	global_prefetch_b8 v[46:47] /*v[302:303]*/, off offset:-2048 scope:SCOPE_SE
	s_set_vgpr_msb 0x100
	ds_load_b128 v[0:3], v223 offset:8704
	ds_load_b128 v[4:7], v223 offset:8736
	ds_load_b128 v[8:11], v223 offset:8768
	ds_load_b128 v[12:15], v223 offset:8800
	ds_load_b128 v[16:19], v223 offset:13056
	ds_load_b128 v[20:23], v223 offset:13088
	ds_load_b128 v[24:27], v223 offset:13120
	ds_load_b128 v[28:31], v223 offset:13152
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v223 offset:128
	ds_load_b128 v[36:39], v223 offset:160
	ds_load_b128 v[40:43], v223 offset:192
	ds_load_b128 v[44:47], v223 offset:224
	ds_load_b128 v[48:51], v223 offset:4480
	ds_load_b128 v[52:55], v223 offset:4512
	ds_load_b128 v[56:59], v223 offset:4544
	ds_load_b128 v[60:63], v223 offset:4576
	s_wait_dscnt 0x0
	ds_load_b128 v[64:67], v223 offset:8832
	ds_load_b128 v[68:71], v223 offset:8864
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[236:243], v[0:15], v[144:151], v74, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[244:251], v[0:15], v[136:143], v75, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[252:259], v[0:15], v[128:135], v76, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[4:11] /*v[260:267]*/, v[0:15], v[120:127], v77, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[4:11] /*v[260:267]*/, v[16:31], v[88:95], v77, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[252:259], v[16:31], v[96:103], v76, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[244:251], v[16:31], v[104:111], v75, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[236:243], v[16:31], v[112:119], v74, v73 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[72:75], v223 offset:8896
	ds_load_b128 v[76:79], v223 offset:8928
	ds_load_b128 v[0:3], v223 offset:13184
	ds_load_b128 v[4:7], v223 offset:13216
	ds_load_b128 v[8:11], v223 offset:13248
	ds_load_b128 v[12:15], v223 offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[12:19] /*v[268:275]*/, v[32:47], v[80:87], v208, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[20:27] /*v[276:283]*/, v[32:47], v[200:207], v209, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[28:35] /*v[284:291]*/, v[32:47], v[192:199], v210, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[36:43] /*v[292:299]*/, v[32:47], v[184:191], v211, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_co_ci_u32_e64 v32, null, 0, v231, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_gt_u32 s65, 0xffffffdf
	v_add_co_ci_u32_e64 v33, null, 0, v232, s2
	s_cselect_b32 s2, -1, 0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[36:43] /*v[292:299]*/, v[48:63], v[152:159], v211, v213 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e64 v49 /*v305*/, 0, 1, vcc_lo
	s_set_vgpr_msb 0x4000
	v_add_co_ci_u32_e64 v34, null, 0, v233, vcc_lo
	s_set_vgpr_msb 64
	v_cndmask_b32_e64 v50 /*v306*/, 0, 1, s2
	s_set_vgpr_msb 0x4001
	v_add_co_ci_u32_e64 v35, null, 0, v234, s2
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[28:35] /*v[284:291]*/, v[48:63], v[160:167], v210, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[20:27] /*v[276:283]*/, v[48:63], v[168:175], v209, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[12:19] /*v[268:275]*/, v[48:63], v[176:183], v208, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[12:19] /*v[268:275]*/, v[64:79], v[144:151], v208, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[20:27] /*v[276:283]*/, v[64:79], v[136:143], v209, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[28:35] /*v[284:291]*/, v[64:79], v[128:135], v210, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[36:43] /*v[292:299]*/, v[64:79], v[120:127], v211, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[36:43] /*v[292:299]*/, v[0:15], v[88:95], v211, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[28:35] /*v[284:291]*/, v[0:15], v[96:103], v210, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[20:27] /*v[276:283]*/, v[0:15], v[104:111], v209, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[12:19] /*v[268:275]*/, v[0:15], v[112:119], v208, v215 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x100
	ds_load_b128 v[236:239], v224
	ds_load_b128 v[240:243], v224 offset:512
	ds_load_b128 v[70:73], v225
	ds_load_b128 v[0:3], v223 offset:53248
	ds_load_b128 v[4:7], v223 offset:53280
	ds_load_b128 v[8:11], v223 offset:53312
	ds_load_b128 v[12:15], v223 offset:53344
	ds_load_b128 v[74:77], v226
	ds_load_b128 v[244:247], v224 offset:2048
	ds_load_b128 v[248:251], v224 offset:2560
	ds_load_b128 v[252:255], v224 offset:4096
	s_set_vgpr_msb 64
	ds_load_b128 v[0:3] /*v[256:259]*/, v224 offset:4608
	ds_load_b128 v[4:7] /*v[260:263]*/, v224 offset:6144
	ds_load_b128 v[8:11] /*v[264:267]*/, v224 offset:6656
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[16:19], v223 offset:57600
	ds_load_b128 v[20:23], v223 offset:57632
	ds_load_b128 v[24:27], v223 offset:57664
	ds_load_b128 v[28:31], v223 offset:57696
	ds_load_b128 v[208:211], v226 offset:16
	ds_load_b128 v[212:215], v225 offset:16
	s_set_vgpr_msb 64
	ds_load_b128 v[12:15] /*v[268:271]*/, v224 offset:1024
	ds_load_b128 v[16:19] /*v[272:275]*/, v224 offset:1536
	ds_load_b128 v[20:23] /*v[276:279]*/, v224 offset:3072
	ds_load_b128 v[24:27] /*v[280:283]*/, v224 offset:3584
	ds_load_b128 v[28:31] /*v[284:287]*/, v224 offset:5120
	ds_load_b128 v[32:35] /*v[288:291]*/, v224 offset:5632
	ds_load_b128 v[36:39] /*v[292:295]*/, v224 offset:7168
	ds_load_b128 v[40:43] /*v[296:299]*/, v224 offset:7680
	s_set_vgpr_msb 0x4000
	s_wait_dscnt 0x14
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[236:243], v[0:15], v[80:87], v74, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[244:251], v[0:15], v[200:207], v75, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[252:259], v[0:15], v[192:199], v76, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[4:11] /*v[260:267]*/, v[0:15], v[184:191], v77, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[4:11] /*v[260:267]*/, v[16:31], v[152:159], v77, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[252:259], v[16:31], v[160:167], v76, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[244:251], v[16:31], v[168:175], v75, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[236:243], v[16:31], v[176:183], v74, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_mov_b64 s[74:75], s[30:31]
	s_add_co_i32 s2, s30, 0x100
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s66
	s_mov_b32 s74, s2
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_mov_b32_e32 v3, v32
	s_add_co_i32 s70, s64, 0x800
	s_add_co_i32 s69, s69, 32
	s_add_co_i32 s65, s65, 32
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s75, v3
	v_readfirstlane_b32 s73, v5
	v_readfirstlane_b32 s74, v6
	v_mov_b32_e32 v3, v33
	s_delay_alu instid0(VALU_DEP_2)
	tensor_load_to_lds s[72:75], s[4:11]
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s31
	s_mov_b32 s74, s70
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_readfirstlane_b32 s75, v3
	v_mov_b32_e32 v3, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s73, v5
	v_readfirstlane_b32 s74, v6
	tensor_load_to_lds s[72:75], s[12:19]
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s67
	s_mov_b32 s74, s69
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_readfirstlane_b32 s75, v3
	v_mov_b32_e32 v3, v35
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s73, v5
	v_readfirstlane_b32 s74, v6
	tensor_load_to_lds s[72:75], s[20:27]
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_mov_b32 s73, s68
	s_mov_b32 s74, s65
	v_mov_b64_e32 v[4:5], s[72:73]
	v_mov_b64_e32 v[6:7], s[74:75]
	v_readfirstlane_b32 s75, v3
	s_addk_co_i32 s30, 0x200
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_u32 s30, s2
	v_readfirstlane_b32 s72, v4
	v_readfirstlane_b32 s73, v5
	v_readfirstlane_b32 s74, v6
	s_cselect_b32 vcc_lo, -1, 0
	s_addk_co_i32 s64, 0x1000
	v_add_co_ci_u32_e64 v231, null, v231, v235, vcc_lo
	s_cmp_lt_u32 s64, s70
	s_cselect_b32 vcc_lo, -1, 0
	s_add_co_u32 s69, s69, 32
	s_cselect_b32 s2, -1, 0
	s_add_co_u32 s65, s65, 32
	s_set_vgpr_msb 4
	v_add_co_ci_u32_e64 v232, null, v232, v48 /*v304*/, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	v_add_co_ci_u32_e64 v233, null, v233, v49 /*v305*/, s2
	v_add_co_ci_u32_e64 v234, null, v234, v50 /*v306*/, vcc_lo
	tensor_load_to_lds s[72:75], s[36:43]
	s_set_vgpr_msb 0x401
	global_prefetch_b8 v[44:45] /*v[300:301]*/, off scope:SCOPE_SE
	global_prefetch_b8 v[46:47] /*v[302:303]*/, off scope:SCOPE_SE
	s_set_vgpr_msb 0x100
	ds_load_b128 v[0:3], v223 offset:61952
	ds_load_b128 v[4:7], v223 offset:61984
	ds_load_b128 v[8:11], v223 offset:62016
	ds_load_b128 v[12:15], v223 offset:62048
	ds_load_b128 v[16:19], v222 offset:13056
	ds_load_b128 v[20:23], v222 offset:13088
	ds_load_b128 v[24:27], v222 offset:13120
	ds_load_b128 v[28:31], v222 offset:13152
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v223 offset:53376
	ds_load_b128 v[36:39], v223 offset:53408
	ds_load_b128 v[40:43], v223 offset:53440
	ds_load_b128 v[44:47], v223 offset:53472
	ds_load_b128 v[48:51], v223 offset:57728
	ds_load_b128 v[52:55], v223 offset:57760
	ds_load_b128 v[56:59], v223 offset:57792
	ds_load_b128 v[60:63], v223 offset:57824
	s_wait_dscnt 0x0
	ds_load_b128 v[64:67], v223 offset:62080
	ds_load_b128 v[68:71], v223 offset:62112
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[236:243], v[0:15], v[144:151], v74, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[244:251], v[0:15], v[136:143], v75, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[252:259], v[0:15], v[128:135], v76, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[4:11] /*v[260:267]*/, v[0:15], v[120:127], v77, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[4:11] /*v[260:267]*/, v[16:31], v[88:95], v77, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[252:259], v[16:31], v[96:103], v76, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[244:251], v[16:31], v[104:111], v75, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[236:243], v[16:31], v[112:119], v74, v73 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[72:75], v223 offset:62144
	ds_load_b128 v[76:79], v223 offset:62176
	ds_load_b128 v[0:3], v222 offset:13184
	ds_load_b128 v[4:7], v222 offset:13216
	ds_load_b128 v[8:11], v222 offset:13248
	ds_load_b128 v[12:15], v222 offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[12:19] /*v[268:275]*/, v[32:47], v[80:87], v208, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[20:27] /*v[276:283]*/, v[32:47], v[200:207], v209, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[28:35] /*v[284:291]*/, v[32:47], v[192:199], v210, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[36:43] /*v[292:299]*/, v[32:47], v[184:191], v211, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[36:43] /*v[292:299]*/, v[48:63], v[152:159], v211, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[28:35] /*v[284:291]*/, v[48:63], v[160:167], v210, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[20:27] /*v[276:283]*/, v[48:63], v[168:175], v209, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[12:19] /*v[268:275]*/, v[48:63], v[176:183], v208, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[12:19] /*v[268:275]*/, v[64:79], v[144:151], v208, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[20:27] /*v[276:283]*/, v[64:79], v[136:143], v209, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[28:35] /*v[284:291]*/, v[64:79], v[128:135], v210, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[36:43] /*v[292:299]*/, v[64:79], v[120:127], v211, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[36:43] /*v[292:299]*/, v[0:15], v[88:95], v211, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[28:35] /*v[284:291]*/, v[0:15], v[96:103], v210, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[20:27] /*v[276:283]*/, v[0:15], v[104:111], v209, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[12:19] /*v[268:275]*/, v[0:15], v[112:119], v208, v215 matrix_a_fmt:MATRIX_FMT_FP4
	s_add_nc_u64 s[58:59], s[58:59], 0x200
	s_add_nc_u64 s[52:53], s[52:53], 0x1000
	s_cmp_lg_u64 s[58:59], 0xa00
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_2
	s_load_b64 s[0:1], s[0:1], 0x0 nv
	s_add_nc_u64 s[2:3], s[56:57], s[54:55]
	s_lshl_b32 s4, s33, 7
	s_lshl_b64 s[2:3], s[2:3], 1
	s_wait_tensorcnt 0x0
	s_or_b32 s2, s2, s4
	s_mov_b32 s28, 1
	v_mov_b64_e32 v[218:219], s[30:31]
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_tensorcnt 0x0
	v_mov_b64_e32 v[216:217], s[28:29]
	s_barrier_signal -1
	v_mov_b32_e32 v217, v228
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_bitset1_b32 s1, 31
	v_dual_mov_b32 v218, s0 :: v_dual_mov_b32 v219, s1
	ds_load_b128 v[232:235], v227 offset:17408
	ds_load_b128 v[236:239], v227 offset:17920
	ds_load_b128 v[70:73], v230
	ds_load_b128 v[0:3], v223
	ds_load_b128 v[4:7], v223 offset:32
	ds_load_b128 v[8:11], v223 offset:64
	ds_load_b128 v[12:15], v223 offset:96
	ds_load_b128 v[74:77], v229 offset:50704
	ds_load_b128 v[240:243], v227 offset:19456
	ds_load_b128 v[244:247], v227 offset:19968
	ds_load_b128 v[248:251], v227 offset:21504
	ds_load_b128 v[252:255], v227 offset:22016
	s_set_vgpr_msb 64
	ds_load_b128 v[0:3] /*v[256:259]*/, v227 offset:23552
	ds_load_b128 v[4:7] /*v[260:263]*/, v227 offset:24064
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[16:19], v223 offset:4352
	ds_load_b128 v[20:23], v223 offset:4384
	ds_load_b128 v[24:27], v223 offset:4416
	ds_load_b128 v[28:31], v223 offset:4448
	ds_load_b128 v[208:211], v229 offset:50720
	ds_load_b128 v[212:215], v230 offset:16
	s_set_vgpr_msb 64
	ds_load_b128 v[8:11] /*v[264:267]*/, v227 offset:18432
	ds_load_b128 v[12:15] /*v[268:271]*/, v227 offset:18944
	ds_load_b128 v[16:19] /*v[272:275]*/, v227 offset:20480
	ds_load_b128 v[20:23] /*v[276:279]*/, v227 offset:20992
	ds_load_b128 v[24:27] /*v[280:283]*/, v227 offset:22528
	ds_load_b128 v[28:31] /*v[284:287]*/, v227 offset:23040
	ds_load_b128 v[32:35] /*v[288:291]*/, v227 offset:24576
	ds_load_b128 v[36:39] /*v[292:295]*/, v227 offset:25088
	s_set_vgpr_msb 0x4000
	s_wait_dscnt 0x14
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[232:239], v[0:15], v[80:87], v74, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[240:247], v[0:15], v[200:207], v75, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[248:255], v[0:15], v[192:199], v76, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[0:7] /*v[256:263]*/, v[0:15], v[184:191], v77, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[0:7] /*v[256:263]*/, v[16:31], v[152:159], v77, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[248:255], v[16:31], v[160:167], v76, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[240:247], v[16:31], v[168:175], v75, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[232:239], v[16:31], v[176:183], v74, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_add_nc_u64 s[30:31], s[34:35], 0xb00
	s_mov_b32 s6, 0
	s_mov_b32 s1, 0x1000000
	s_bitset1_b32 s31, 31
	s_movk_i32 s5, 0xc00
	s_mov_b32 s4, 16
	s_mov_b32 s2, 0x100000
	s_mov_b32 s0, 0x7500000
	s_mov_b32 s3, s1
	s_mov_b32 s7, s6
	s_brev_b32 s9, 16
	tensor_load_to_lds s[28:31], s[0:7]
	s_add_nc_u64 s[0:1], s[44:45], 0x5800
	s_mov_b64 s[16:17], s[28:29]
	s_bitset1_b32 s1, 31
	s_mov_b64 s[18:19], s[30:31]
	s_movk_i32 s13, 0x6000
	s_mov_b32 s12, 4
	s_mov_b32 s10, 0x40000
	s_mov_b32 s8, s6
	s_mov_b32 s11, s9
	s_mov_b32 s14, s6
	s_mov_b32 s15, s6
	s_mov_b32 s17, s63
	s_mov_b32 s18, s0
	s_mov_b32 s19, s1
	s_add_nc_u64 s[0:1], s[46:47], 0x160
	tensor_load_to_lds s[16:19], s[8:15]
	s_add_nc_u64 s[8:9], s[48:49], s[0:1]
	s_mov_b64 s[16:17], s[28:29]
	s_or_b32 s3, s9, 0x80000000
	s_mov_b32 s9, 0x200000
	s_mov_b64 s[18:19], s[30:31]
	s_mov_b32 s17, s61
	s_mov_b32 s18, s8
	s_mov_b32 s19, s3
	s_movk_i32 s13, 0x180
	s_mov_b32 s8, s6
	s_mov_b32 s11, s9
	s_add_nc_u64 s[0:1], s[50:51], s[0:1]
	s_mov_b32 s29, s60
	s_or_b32 s31, s1, 0x80000000
	s_mov_b32 s30, s0
	tensor_load_to_lds s[16:19], s[8:15]
	s_mov_b32 s10, s2
	s_mov_b32 s12, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[28:31], s[8:15]
	ds_load_b128 v[0:3], v223 offset:8704
	ds_load_b128 v[4:7], v223 offset:8736
	ds_load_b128 v[8:11], v223 offset:8768
	ds_load_b128 v[12:15], v223 offset:8800
	ds_load_b128 v[16:19], v223 offset:13056
	ds_load_b128 v[20:23], v223 offset:13088
	ds_load_b128 v[24:27], v223 offset:13120
	ds_load_b128 v[28:31], v223 offset:13152
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v223 offset:128
	ds_load_b128 v[36:39], v223 offset:160
	ds_load_b128 v[40:43], v223 offset:192
	ds_load_b128 v[44:47], v223 offset:224
	ds_load_b128 v[48:51], v223 offset:4480
	ds_load_b128 v[52:55], v223 offset:4512
	ds_load_b128 v[56:59], v223 offset:4544
	ds_load_b128 v[60:63], v223 offset:4576
	s_wait_dscnt 0x0
	ds_load_b128 v[64:67], v223 offset:8832
	ds_load_b128 v[68:71], v223 offset:8864
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[232:239], v[0:15], v[144:151], v74, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[240:247], v[0:15], v[136:143], v75, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[248:255], v[0:15], v[128:135], v76, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[0:7] /*v[256:263]*/, v[0:15], v[120:127], v77, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[0:7] /*v[256:263]*/, v[16:31], v[88:95], v77, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[248:255], v[16:31], v[96:103], v76, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[240:247], v[16:31], v[104:111], v75, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[232:239], v[16:31], v[112:119], v74, v73 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[72:75], v223 offset:8896
	ds_load_b128 v[76:79], v223 offset:8928
	ds_load_b128 v[0:3], v223 offset:13184
	ds_load_b128 v[4:7], v223 offset:13216
	ds_load_b128 v[8:11], v223 offset:13248
	ds_load_b128 v[12:15], v223 offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[8:15] /*v[264:271]*/, v[32:47], v[80:87], v208, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[16:23] /*v[272:279]*/, v[32:47], v[200:207], v209, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[24:31] /*v[280:287]*/, v[32:47], v[192:199], v210, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[32:39] /*v[288:295]*/, v[32:47], v[184:191], v211, v212 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[32:39] /*v[288:295]*/, v[48:63], v[152:159], v211, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[24:31] /*v[280:287]*/, v[48:63], v[160:167], v210, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[16:23] /*v[272:279]*/, v[48:63], v[168:175], v209, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[8:15] /*v[264:271]*/, v[48:63], v[176:183], v208, v213 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[8:15] /*v[264:271]*/, v[64:79], v[144:151], v208, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[16:23] /*v[272:279]*/, v[64:79], v[136:143], v209, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[24:31] /*v[280:287]*/, v[64:79], v[128:135], v210, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[32:39] /*v[288:295]*/, v[64:79], v[120:127], v211, v214 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[32:39] /*v[288:295]*/, v[0:15], v[88:95], v211, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[24:31] /*v[280:287]*/, v[0:15], v[96:103], v210, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[16:23] /*v[272:279]*/, v[0:15], v[104:111], v209, v215 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[8:15] /*v[264:271]*/, v[0:15], v[112:119], v208, v215 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x100
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[40:43], v224
	ds_load_b128 v[44:47], v224 offset:512
	ds_load_b128 v[242:245], v225
	ds_load_b128 v[0:3], v223 offset:53248
	ds_load_b128 v[4:7], v223 offset:53280
	ds_load_b128 v[8:11], v223 offset:53312
	ds_load_b128 v[12:15], v223 offset:53344
	ds_load_b128 v[246:249], v226
	ds_load_b128 v[48:51], v224 offset:2048
	ds_load_b128 v[52:55], v224 offset:2560
	ds_load_b128 v[56:59], v224 offset:4096
	ds_load_b128 v[60:63], v224 offset:4608
	ds_load_b128 v[64:67], v224 offset:6144
	ds_load_b128 v[68:71], v224 offset:6656
	ds_load_b128 v[16:19], v223 offset:57600
	ds_load_b128 v[20:23], v223 offset:57632
	ds_load_b128 v[24:27], v223 offset:57664
	ds_load_b128 v[28:31], v223 offset:57696
	ds_load_b128 v[32:35], v226 offset:16
	ds_load_b128 v[36:39], v225 offset:16
	s_wait_dscnt 0xc
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[40:47], v[0:15], v[80:87], v246, v242 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[72:75], v224 offset:1024
	ds_load_b128 v[76:79], v224 offset:1536
	ds_load_b128 v[208:211], v224 offset:3072
	ds_load_b128 v[212:215], v224 offset:3584
	ds_load_b128 v[226:229], v224 offset:5120
	ds_load_b128 v[230:233], v224 offset:5632
	ds_load_b128 v[234:237], v224 offset:7168
	ds_load_b128 v[238:241], v224 offset:7680
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[48:55], v[0:15], v[200:207], v247, v242 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[56:63], v[0:15], v[192:199], v248, v242 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[64:71], v[0:15], v[184:191], v249, v242 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[0:3], v223 offset:61952
	ds_load_b128 v[4:7], v223 offset:61984
	ds_load_b128 v[8:11], v223 offset:62016
	ds_load_b128 v[12:15], v223 offset:62048
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[64:71], v[16:31], v[152:159], v249, v243 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[56:63], v[16:31], v[160:167], v248, v243 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[48:55], v[16:31], v[168:175], v247, v243 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[40:47], v[16:31], v[176:183], v246, v243 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[16:19], v222 offset:13056
	ds_load_b128 v[20:23], v222 offset:13088
	ds_load_b128 v[24:27], v222 offset:13120
	ds_load_b128 v[28:31], v222 offset:13152
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[40:47], v[0:15], v[144:151], v246, v244 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[48:55], v[0:15], v[136:143], v247, v244 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[56:63], v[0:15], v[128:135], v248, v244 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[64:71], v[0:15], v[120:127], v249, v244 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[0:3], v223 offset:53376
	ds_load_b128 v[4:7], v223 offset:53408
	ds_load_b128 v[8:11], v223 offset:53440
	ds_load_b128 v[12:15], v223 offset:53472
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[64:71], v[16:31], v[88:95], v249, v245 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[56:63], v[16:31], v[96:103], v248, v245 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[48:55], v[16:31], v[104:111], v247, v245 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[40:47], v[16:31], v[112:119], v246, v245 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[16:19], v223 offset:57728
	ds_load_b128 v[20:23], v223 offset:57760
	ds_load_b128 v[24:27], v223 offset:57792
	ds_load_b128 v[28:31], v223 offset:57824
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[72:79], v[0:15], v[80:87], v32, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[208:215], v[0:15], v[200:207], v33, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[226:233], v[0:15], v[192:199], v34, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[234:241], v[0:15], v[184:191], v35, v36 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[0:3], v223 offset:62080
	ds_load_b128 v[4:7], v223 offset:62112
	ds_load_b128 v[8:11], v223 offset:62144
	ds_load_b128 v[12:15], v223 offset:62176
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[234:241], v[16:31], v[152:159], v35, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[226:233], v[16:31], v[160:167], v34, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[208:215], v[16:31], v[168:175], v33, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[72:79], v[16:31], v[176:183], v32, v37 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[16:19], v222 offset:13184
	ds_load_b128 v[20:23], v222 offset:13216
	ds_load_b128 v[24:27], v222 offset:13248
	ds_load_b128 v[28:31], v222 offset:13280
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[72:79], v[0:15], v[144:151], v32, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[208:215], v[0:15], v[136:143], v33, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[226:233], v[0:15], v[128:135], v34, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[234:241], v[0:15], v[120:127], v35, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[234:241], v[16:31], v[88:95], v35, v39 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[226:233], v[16:31], v[96:103], v34, v39 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[208:215], v[16:31], v[104:111], v33, v39 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[72:79], v[16:31], v[112:119], v32, v39 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_lshlrev_b32_e32 v4, 7, v220
	s_lshl_b32 s0, s33, 13
	v_cvt_pk_bf16_f32 v3, v86, v87
	v_cvt_pk_bf16_f32 v2, v84, v85
	v_cvt_pk_bf16_f32 v1, v82, v83
	v_cvt_pk_bf16_f32 v0, v80, v81
	v_or3_b32 v24, s0, v4, v221
	v_cvt_pk_bf16_f32 v7, v206, v207
	v_cvt_pk_bf16_f32 v6, v204, v205
	v_cvt_pk_bf16_f32 v5, v202, v203
	v_cvt_pk_bf16_f32 v4, v200, v201
	v_cvt_pk_bf16_f32 v11, v198, v199
	v_cvt_pk_bf16_f32 v10, v196, v197
	v_cvt_pk_bf16_f32 v9, v194, v195
	v_cvt_pk_bf16_f32 v8, v192, v193
	v_cvt_pk_bf16_f32 v15, v190, v191
	v_cvt_pk_bf16_f32 v14, v188, v189
	v_cvt_pk_bf16_f32 v13, v186, v187
	v_cvt_pk_bf16_f32 v12, v184, v185
	ds_store_b128 v24, v[0:3]
	ds_store_b128 v24, v[4:7] offset:32
	ds_store_b128 v24, v[8:11] offset:64
	v_cvt_pk_bf16_f32 v3, v182, v183
	ds_store_b128 v24, v[12:15] offset:96
	v_cvt_pk_bf16_f32 v2, v180, v181
	v_cvt_pk_bf16_f32 v1, v178, v179
	v_cvt_pk_bf16_f32 v0, v176, v177
	v_cvt_pk_bf16_f32 v7, v174, v175
	v_cvt_pk_bf16_f32 v6, v172, v173
	v_cvt_pk_bf16_f32 v5, v170, v171
	v_cvt_pk_bf16_f32 v4, v168, v169
	v_cvt_pk_bf16_f32 v11, v166, v167
	v_cvt_pk_bf16_f32 v10, v164, v165
	v_cvt_pk_bf16_f32 v9, v162, v163
	v_cvt_pk_bf16_f32 v8, v160, v161
	v_cvt_pk_bf16_f32 v15, v158, v159
	v_cvt_pk_bf16_f32 v14, v156, v157
	v_cvt_pk_bf16_f32 v13, v154, v155
	v_cvt_pk_bf16_f32 v12, v152, v153
	v_cvt_pk_bf16_f32 v19, v150, v151
	v_cvt_pk_bf16_f32 v18, v148, v149
	v_cvt_pk_bf16_f32 v17, v146, v147
	v_cvt_pk_bf16_f32 v16, v144, v145
	v_cvt_pk_bf16_f32 v23, v142, v143
	v_cvt_pk_bf16_f32 v22, v140, v141
	v_cvt_pk_bf16_f32 v21, v138, v139
	v_cvt_pk_bf16_f32 v20, v136, v137
	ds_store_b128 v24, v[0:3] offset:2048
	ds_store_b128 v24, v[4:7] offset:2080
	ds_store_b128 v24, v[8:11] offset:2112
	ds_store_b128 v24, v[12:15] offset:2144
	ds_store_b128 v24, v[16:19] offset:4096
	ds_store_b128 v24, v[20:23] offset:4128
	v_cvt_pk_bf16_f32 v3, v134, v135
	v_cvt_pk_bf16_f32 v2, v132, v133
	v_cvt_pk_bf16_f32 v1, v130, v131
	v_cvt_pk_bf16_f32 v0, v128, v129
	v_cvt_pk_bf16_f32 v7, v126, v127
	v_cvt_pk_bf16_f32 v6, v124, v125
	v_cvt_pk_bf16_f32 v5, v122, v123
	v_cvt_pk_bf16_f32 v4, v120, v121
	v_cvt_pk_bf16_f32 v11, v118, v119
	v_cvt_pk_bf16_f32 v10, v116, v117
	v_cvt_pk_bf16_f32 v9, v114, v115
	v_cvt_pk_bf16_f32 v8, v112, v113
	v_cvt_pk_bf16_f32 v15, v110, v111
	v_cvt_pk_bf16_f32 v14, v108, v109
	v_cvt_pk_bf16_f32 v13, v106, v107
	v_cvt_pk_bf16_f32 v12, v104, v105
	v_cvt_pk_bf16_f32 v19, v102, v103
	v_cvt_pk_bf16_f32 v18, v100, v101
	v_cvt_pk_bf16_f32 v17, v98, v99
	v_cvt_pk_bf16_f32 v16, v96, v97
	v_cvt_pk_bf16_f32 v23, v94, v95
	v_cvt_pk_bf16_f32 v22, v92, v93
	v_cvt_pk_bf16_f32 v21, v90, v91
	v_cvt_pk_bf16_f32 v20, v88, v89
	v_readfirstlane_b32 s8, v216
	v_readfirstlane_b32 s9, v217
	v_readfirstlane_b32 s10, v218
	v_readfirstlane_b32 s11, v219
	s_mov_b32 s1, 0x400000
	s_mov_b32 s4, 64
	s_mov_b32 s0, 0x10000
	s_mov_b32 s2, s1
	s_mov_b32 s3, s1
	ds_store_b128 v24, v[0:3] offset:4160
	ds_store_b128 v24, v[4:7] offset:4192
	ds_store_b128 v24, v[8:11] offset:6144
	ds_store_b128 v24, v[12:15] offset:6176
	ds_store_b128 v24, v[16:19] offset:6208
	ds_store_b128 v24, v[20:23] offset:6240
	s_wait_dscnt 0x0
	tensor_store_from_lds s[8:11], s[0:7]
	s_wait_tensorcnt 0x0
.LBB0_4:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernel_mxscale_gemm2
		.amdhsa_group_segment_fixed_size 106496
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 307
		.amdhsa_next_free_sgpr 85
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 60
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
	.size	kernel_mxscale_gemm2, .Lfunc_end0-kernel_mxscale_gemm2

	.set kernel_mxscale_gemm2.num_vgpr, 307
	.set kernel_mxscale_gemm2.num_agpr, 0
	.set kernel_mxscale_gemm2.numbered_sgpr, 85
	.set kernel_mxscale_gemm2.num_named_barrier, 0
	.set kernel_mxscale_gemm2.private_seg_size, 0
	.set kernel_mxscale_gemm2.uses_vcc, 1
	.set kernel_mxscale_gemm2.uses_flat_scratch, 0
	.set kernel_mxscale_gemm2.has_dyn_sized_stack, 0
	.set kernel_mxscale_gemm2.has_recursion, 0
	.set kernel_mxscale_gemm2.has_indirect_call, 0
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
      - .offset:         8
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .offset:         128
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         160
        .size:           8
        .value_kind:     global_buffer
      - .offset:         168
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         200
        .size:           8
        .value_kind:     global_buffer
      - .offset:         208
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         240
        .size:           8
        .value_kind:     global_buffer
      - .offset:         248
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         256
        .size:           8
        .value_kind:     global_buffer
      - .offset:         264
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         272
        .size:           8
        .value_kind:     global_buffer
      - .offset:         280
        .size:           4
        .value_kind:     by_value
      - .offset:         284
        .size:           4
        .value_kind:     by_value
      - .offset:         288
        .size:           4
        .value_kind:     by_value
      - .offset:         292
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 106496
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .max_flat_workgroup_size: 128
    .name:           kernel_mxscale_gemm2
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     87
    .sgpr_spill_count: 0
    .symbol:         kernel_mxscale_gemm2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     307
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
