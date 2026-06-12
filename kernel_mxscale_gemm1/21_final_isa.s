	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel_mxscale_gemm1
	.p2align	8
	.type	kernel_mxscale_gemm1,@function
kernel_mxscale_gemm1:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[4:5], s[0:1], 0xf0 nv
	s_mov_b32 s28, 1
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_mov_b32 s10, 0
	s_mul_i32 s6, ttmp9, s2
	s_getreg_b32 s2, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s3, s3, s6
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s6, 0x1ffffff
	s_cselect_b32 s64, ttmp9, s3
	s_wait_kmcnt 0x0
	s_or_b64 s[4:5], s[4:5], 0xfe00000000000000
	s_lshl_b32 s3, s64, 2
	s_mov_b32 s7, s10
	v_mov_b32_e32 v1, s3
	buffer_load_b32 v1, v1, s[4:7], null offen
	s_wait_loadcnt 0x0
	v_cmp_gt_i32_e32 vcc_lo, 1, v1
	s_cbranch_vccnz .LBB0_4
	s_bfe_u32 s3, ttmp6, 0x40010
	s_load_b64 s[66:67], s[0:1], 0x28 nv
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s4, ttmp6, 0x40004
	s_mul_i32 s3, ttmp7, s3
	s_ashr_i32 s65, s64, 31
	s_add_co_i32 s4, s4, s3
	s_cmp_eq_u32 s2, 0
	s_movk_i32 s9, 0xc00
	s_cselect_b32 s20, ttmp7, s4
	s_bfe_u32 s4, ttmp8, 0x50019
	s_lshl_b64 s[34:35], s[64:65], 6
	s_and_b32 s86, s4, 3
	s_lshr_b32 s30, s4, 2
	s_lshl_b32 s87, s86, 4
	s_mov_b32 s5, s35
	s_or_b32 s4, s34, s87
	s_lshl_b32 s6, s30, 8
	s_mul_u64 s[4:5], s[4:5], 0xc00
	s_mul_i32 s31, s86, 0x1100
	s_or_b32 s4, s4, s6
	s_add_co_i32 s31, s31, s6
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[46:47], s[4:5], s[66:67]
	s_add_co_i32 s29, s31, 0x15800
	s_or_b32 s89, s47, 0x80000000
	s_mov_b64 s[12:13], s[28:29]
	s_mov_b32 s5, 0x1000000
	s_mov_b64 s[14:15], s[30:31]
	s_mov_b32 s13, s31
	s_mov_b32 s14, s46
	s_mov_b32 s15, s89
	s_mov_b32 s8, 16
	s_mov_b32 s6, 0x100000
	s_mov_b32 s4, 0x7500000
	s_mov_b32 s7, s5
	s_mov_b32 s11, s10
	s_clause 0x2
	s_load_b64 s[70:71], s[0:1], 0x50 nv
	s_load_b64 s[68:69], s[0:1], 0x78 nv
	s_load_b64 s[2:3], s[0:1], 0xa0 nv
	tensor_load_to_lds s[12:15], s[4:11]
	s_ashr_i32 s21, s20, 31
	s_mul_u64 s[54:55], s[64:65], 0x180
	s_lshl_b64 s[44:45], s[20:21], 8
	s_lshl_b32 s72, s30, 11
	s_lshr_b64 s[74:75], s[44:45], 4
	s_lshl_b32 s20, s86, 13
	s_lshl_b32 s88, s86, 2
	s_add_nc_u64 s[76:77], s[74:75], s[54:55]
	s_mov_b32 s73, s10
	s_add_co_i32 s35, s20, s72
	s_or_b32 s20, s76, s88
	s_mov_b32 s21, s77
	s_brev_b32 s13, 16
	s_mul_u64 s[48:49], s[20:21], 0x6000
	s_add_co_i32 s75, s35, 0x4400
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[50:51], s[72:73], s[70:71]
	s_mov_b64 s[20:21], s[28:29]
	s_add_nc_u64 s[90:91], s[50:51], s[48:49]
	s_mov_b64 s[22:23], s[30:31]
	s_or_b32 s100, s91, 0x80000000
	s_mov_b32 s16, 4
	s_movk_i32 s17, 0x6000
	s_mov_b32 s14, 0x40000
	s_mov_b32 s12, s10
	s_mov_b32 s18, s10
	s_mov_b32 s19, s10
	s_mov_b32 s15, s13
	s_mov_b32 s21, s75
	s_mov_b32 s22, s90
	s_mov_b32 s23, s100
	s_lshl_b64 s[36:37], s[64:65], 4
	s_mov_b32 s63, s10
	s_or_b32 s36, s36, s88
	s_lshl_b32 s62, s30, 5
	s_mul_u64 s[36:37], s[36:37], 0x180
	s_lshl_b32 s30, s86, 7
	s_add_nc_u64 s[40:41], s[36:37], s[68:69]
	s_add_co_i32 s30, s30, s62
	s_add_nc_u64 s[52:53], s[40:41], s[62:63]
	s_or_b32 s73, s30, 0x14400
	s_mov_b64 s[38:39], s[30:31]
	s_or_b32 s101, s53, 0x80000000
	s_mov_b64 s[36:37], s[28:29]
	s_movk_i32 s25, 0x180
	s_mov_b32 s26, s10
	s_mov_b32 s27, s10
	s_mov_b32 s24, s16
	s_mov_b32 s37, s73
	s_mov_b32 s38, s52
	s_mov_b32 s39, s101
	s_mul_u64 s[60:61], s[64:65], 0x600
	s_lshr_b64 s[78:79], s[44:45], 2
	s_mul_i32 s33, s86, 0x180
	s_add_nc_u64 s[56:57], s[78:79], s[60:61]
	s_add_nc_u64 s[58:59], s[62:63], s[2:3]
	s_or_b32 s56, s56, s87
	s_add_co_i32 s33, s30, s33
	s_mul_u64 s[56:57], s[56:57], 0x180
	s_add_co_i32 s79, s33, 0x14610
	s_add_nc_u64 s[84:85], s[58:59], s[56:57]
	s_mov_b64 s[82:83], s[30:31]
	s_or_b32 s102, s85, 0x80000000
	s_mov_b64 s[80:81], s[28:29]
	s_mov_b32 s42, s10
	s_mov_b32 s43, s10
	s_mov_b32 s40, s8
	s_mov_b32 s41, s25
	s_mov_b32 s81, s79
	s_mov_b32 s82, s84
	s_mov_b32 s83, s102
	s_add_co_i32 s3, s35, 0xc400
	s_mov_b64 s[94:95], s[30:31]
	s_mov_b64 s[92:93], s[28:29]
	s_mov_b32 s93, s3
	s_add_co_i32 s69, s35, 0x19c00
	s_add_co_i32 s35, s35, 0x21c00
	s_or_b32 s63, s30, 0x29c00
	s_set_vgpr_msb 64
	v_dual_mov_b32 v80 /*v336*/, 0 :: v_dual_bitop2_b32 v112 /*v368*/, 15, v0 bitop3:0x40
	v_bfe_u32 v114 /*v370*/, v0, 4, 1
	s_mulk_i32 s86, 0x600
	v_lshrrev_b32_e32 v128 /*v384*/, 5, v0
	s_set_vgpr_msb 0x4005
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v240, v80 /*v336*/ :: v_dual_lshlrev_b32 v4, 8, v114 /*v370*/
	v_dual_mov_b32 v241, v80 /*v336*/ :: v_dual_lshlrev_b32 v6, 5, v112 /*v368*/
	s_set_vgpr_msb 0x500
	v_and_b32_e32 v1, 63, v0
	s_set_vgpr_msb 4
	v_mul_u32_u24_e32 v0, 0x110, v112 /*v368*/
	s_set_vgpr_msb 0x441
	v_dual_mov_b32 v81 /*v337*/, v80 /*v336*/ :: v_dual_mov_b32 v82 /*v338*/, v80 /*v336*/
	v_dual_mov_b32 v83 /*v339*/, v80 /*v336*/ :: v_dual_mov_b32 v84 /*v340*/, v80 /*v336*/
	v_dual_mov_b32 v85 /*v341*/, v80 /*v336*/ :: v_dual_mov_b32 v86 /*v342*/, v80 /*v336*/
	v_dual_mov_b32 v87 /*v343*/, v80 /*v336*/ :: v_dual_mov_b32 v64 /*v320*/, v80 /*v336*/
	v_dual_mov_b32 v65 /*v321*/, v80 /*v336*/ :: v_dual_mov_b32 v66 /*v322*/, v80 /*v336*/
	v_dual_mov_b32 v67 /*v323*/, v80 /*v336*/ :: v_dual_mov_b32 v68 /*v324*/, v80 /*v336*/
	v_dual_mov_b32 v69 /*v325*/, v80 /*v336*/ :: v_dual_mov_b32 v70 /*v326*/, v80 /*v336*/
	v_dual_mov_b32 v71 /*v327*/, v80 /*v336*/ :: v_dual_mov_b32 v48 /*v304*/, v80 /*v336*/
	v_dual_mov_b32 v49 /*v305*/, v80 /*v336*/ :: v_dual_mov_b32 v50 /*v306*/, v80 /*v336*/
	v_dual_mov_b32 v51 /*v307*/, v80 /*v336*/ :: v_dual_mov_b32 v52 /*v308*/, v80 /*v336*/
	v_dual_mov_b32 v53 /*v309*/, v80 /*v336*/ :: v_dual_mov_b32 v54 /*v310*/, v80 /*v336*/
	v_dual_mov_b32 v55 /*v311*/, v80 /*v336*/ :: v_dual_mov_b32 v32 /*v288*/, v80 /*v336*/
	v_dual_mov_b32 v33 /*v289*/, v80 /*v336*/ :: v_dual_mov_b32 v34 /*v290*/, v80 /*v336*/
	v_dual_mov_b32 v35 /*v291*/, v80 /*v336*/ :: v_dual_mov_b32 v36 /*v292*/, v80 /*v336*/
	v_dual_mov_b32 v37 /*v293*/, v80 /*v336*/ :: v_dual_mov_b32 v38 /*v294*/, v80 /*v336*/
	v_dual_mov_b32 v39 /*v295*/, v80 /*v336*/ :: v_dual_mov_b32 v16 /*v272*/, v80 /*v336*/
	v_dual_mov_b32 v17 /*v273*/, v80 /*v336*/ :: v_dual_mov_b32 v18 /*v274*/, v80 /*v336*/
	v_dual_mov_b32 v19 /*v275*/, v80 /*v336*/ :: v_dual_mov_b32 v20 /*v276*/, v80 /*v336*/
	v_dual_mov_b32 v21 /*v277*/, v80 /*v336*/ :: v_dual_mov_b32 v22 /*v278*/, v80 /*v336*/
	v_dual_mov_b32 v23 /*v279*/, v80 /*v336*/ :: v_dual_mov_b32 v0 /*v256*/, v80 /*v336*/
	v_dual_mov_b32 v1 /*v257*/, v80 /*v336*/ :: v_dual_mov_b32 v2 /*v258*/, v80 /*v336*/
	v_dual_mov_b32 v3 /*v259*/, v80 /*v336*/ :: v_dual_mov_b32 v4 /*v260*/, v80 /*v336*/
	v_dual_mov_b32 v5 /*v261*/, v80 /*v336*/ :: v_dual_mov_b32 v6 /*v262*/, v80 /*v336*/
	v_dual_mov_b32 v7 /*v263*/, v80 /*v336*/ :: v_dual_mov_b32 v96 /*v352*/, v80 /*v336*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v243, v80 /*v336*/ :: v_dual_mov_b32 v244, v80 /*v336*/
	v_dual_mov_b32 v245, v80 /*v336*/ :: v_dual_mov_b32 v246, v80 /*v336*/
	v_dual_mov_b32 v247, v80 /*v336*/ :: v_dual_mov_b32 v224, v80 /*v336*/
	v_dual_mov_b32 v225, v80 /*v336*/ :: v_dual_mov_b32 v226, v80 /*v336*/
	v_dual_mov_b32 v227, v80 /*v336*/ :: v_dual_mov_b32 v228, v80 /*v336*/
	v_dual_mov_b32 v229, v80 /*v336*/ :: v_dual_mov_b32 v230, v80 /*v336*/
	v_dual_mov_b32 v231, v80 /*v336*/ :: v_dual_mov_b32 v208, v80 /*v336*/
	v_dual_mov_b32 v209, v80 /*v336*/ :: v_dual_mov_b32 v210, v80 /*v336*/
	v_dual_mov_b32 v211, v80 /*v336*/ :: v_dual_mov_b32 v212, v80 /*v336*/
	v_dual_mov_b32 v213, v80 /*v336*/ :: v_dual_mov_b32 v214, v80 /*v336*/
	v_dual_mov_b32 v215, v80 /*v336*/ :: v_dual_mov_b32 v192, v80 /*v336*/
	v_dual_mov_b32 v193, v80 /*v336*/ :: v_dual_mov_b32 v194, v80 /*v336*/
	v_dual_mov_b32 v195, v80 /*v336*/ :: v_dual_mov_b32 v196, v80 /*v336*/
	v_dual_mov_b32 v197, v80 /*v336*/ :: v_dual_mov_b32 v198, v80 /*v336*/
	v_dual_mov_b32 v199, v80 /*v336*/ :: v_dual_mov_b32 v176, v80 /*v336*/
	v_dual_mov_b32 v177, v80 /*v336*/ :: v_dual_mov_b32 v178, v80 /*v336*/
	v_dual_mov_b32 v179, v80 /*v336*/ :: v_dual_mov_b32 v180, v80 /*v336*/
	v_dual_mov_b32 v181, v80 /*v336*/ :: v_dual_mov_b32 v182, v80 /*v336*/
	v_dual_mov_b32 v183, v80 /*v336*/ :: v_dual_mov_b32 v160, v80 /*v336*/
	v_dual_mov_b32 v161, v80 /*v336*/ :: v_dual_mov_b32 v162, v80 /*v336*/
	v_dual_mov_b32 v163, v80 /*v336*/ :: v_dual_mov_b32 v164, v80 /*v336*/
	v_dual_mov_b32 v165, v80 /*v336*/ :: v_dual_mov_b32 v166, v80 /*v336*/
	v_dual_mov_b32 v167, v80 /*v336*/ :: v_dual_mov_b32 v144, v80 /*v336*/
	v_dual_mov_b32 v145, v80 /*v336*/ :: v_dual_mov_b32 v146, v80 /*v336*/
	v_dual_mov_b32 v147, v80 /*v336*/ :: v_dual_mov_b32 v148, v80 /*v336*/
	v_dual_mov_b32 v149, v80 /*v336*/ :: v_dual_mov_b32 v150, v80 /*v336*/
	v_dual_mov_b32 v151, v80 /*v336*/ :: v_dual_mov_b32 v128, v80 /*v336*/
	v_dual_mov_b32 v129, v80 /*v336*/ :: v_dual_mov_b32 v130, v80 /*v336*/
	v_dual_mov_b32 v131, v80 /*v336*/ :: v_dual_mov_b32 v132, v80 /*v336*/
	v_dual_mov_b32 v133, v80 /*v336*/ :: v_dual_mov_b32 v134, v80 /*v336*/
	v_dual_mov_b32 v135, v80 /*v336*/ :: v_dual_mov_b32 v112, v80 /*v336*/
	v_dual_mov_b32 v113, v80 /*v336*/ :: v_dual_mov_b32 v114, v80 /*v336*/
	v_dual_mov_b32 v115, v80 /*v336*/ :: v_dual_mov_b32 v116, v80 /*v336*/
	v_dual_mov_b32 v117, v80 /*v336*/ :: v_dual_mov_b32 v118, v80 /*v336*/
	v_dual_mov_b32 v119, v80 /*v336*/ :: v_dual_mov_b32 v248, v80 /*v336*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v97 /*v353*/, v80 /*v336*/ :: v_dual_mov_b32 v98 /*v354*/, v80 /*v336*/
	v_dual_mov_b32 v99 /*v355*/, v80 /*v336*/ :: v_dual_mov_b32 v100 /*v356*/, v80 /*v336*/
	v_dual_mov_b32 v101 /*v357*/, v80 /*v336*/ :: v_dual_mov_b32 v102 /*v358*/, v80 /*v336*/
	v_dual_mov_b32 v103 /*v359*/, v80 /*v336*/ :: v_dual_mov_b32 v88 /*v344*/, v80 /*v336*/
	v_dual_mov_b32 v89 /*v345*/, v80 /*v336*/ :: v_dual_mov_b32 v90 /*v346*/, v80 /*v336*/
	v_dual_mov_b32 v91 /*v347*/, v80 /*v336*/ :: v_dual_mov_b32 v92 /*v348*/, v80 /*v336*/
	v_dual_mov_b32 v93 /*v349*/, v80 /*v336*/ :: v_dual_mov_b32 v94 /*v350*/, v80 /*v336*/
	v_dual_mov_b32 v95 /*v351*/, v80 /*v336*/ :: v_dual_mov_b32 v72 /*v328*/, v80 /*v336*/
	v_dual_mov_b32 v73 /*v329*/, v80 /*v336*/ :: v_dual_mov_b32 v74 /*v330*/, v80 /*v336*/
	v_dual_mov_b32 v75 /*v331*/, v80 /*v336*/ :: v_dual_mov_b32 v76 /*v332*/, v80 /*v336*/
	v_dual_mov_b32 v77 /*v333*/, v80 /*v336*/ :: v_dual_mov_b32 v78 /*v334*/, v80 /*v336*/
	v_dual_mov_b32 v79 /*v335*/, v80 /*v336*/ :: v_dual_mov_b32 v56 /*v312*/, v80 /*v336*/
	v_dual_mov_b32 v57 /*v313*/, v80 /*v336*/ :: v_dual_mov_b32 v58 /*v314*/, v80 /*v336*/
	v_dual_mov_b32 v59 /*v315*/, v80 /*v336*/ :: v_dual_mov_b32 v60 /*v316*/, v80 /*v336*/
	v_dual_mov_b32 v61 /*v317*/, v80 /*v336*/ :: v_dual_mov_b32 v62 /*v318*/, v80 /*v336*/
	v_dual_mov_b32 v63 /*v319*/, v80 /*v336*/ :: v_dual_mov_b32 v40 /*v296*/, v80 /*v336*/
	v_dual_mov_b32 v41 /*v297*/, v80 /*v336*/ :: v_dual_mov_b32 v42 /*v298*/, v80 /*v336*/
	v_dual_mov_b32 v43 /*v299*/, v80 /*v336*/ :: v_dual_mov_b32 v44 /*v300*/, v80 /*v336*/
	v_dual_mov_b32 v45 /*v301*/, v80 /*v336*/ :: v_dual_mov_b32 v46 /*v302*/, v80 /*v336*/
	v_dual_mov_b32 v47 /*v303*/, v80 /*v336*/ :: v_dual_mov_b32 v24 /*v280*/, v80 /*v336*/
	v_dual_mov_b32 v25 /*v281*/, v80 /*v336*/ :: v_dual_mov_b32 v26 /*v282*/, v80 /*v336*/
	v_dual_mov_b32 v27 /*v283*/, v80 /*v336*/ :: v_dual_mov_b32 v28 /*v284*/, v80 /*v336*/
	v_dual_mov_b32 v29 /*v285*/, v80 /*v336*/ :: v_dual_mov_b32 v30 /*v286*/, v80 /*v336*/
	v_dual_mov_b32 v31 /*v287*/, v80 /*v336*/ :: v_dual_mov_b32 v8 /*v264*/, v80 /*v336*/
	v_dual_mov_b32 v9 /*v265*/, v80 /*v336*/ :: v_dual_mov_b32 v10 /*v266*/, v80 /*v336*/
	v_dual_mov_b32 v11 /*v267*/, v80 /*v336*/ :: v_dual_mov_b32 v12 /*v268*/, v80 /*v336*/
	v_dual_mov_b32 v13 /*v269*/, v80 /*v336*/ :: v_dual_mov_b32 v14 /*v270*/, v80 /*v336*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v249, v80 /*v336*/ :: v_dual_mov_b32 v250, v80 /*v336*/
	v_dual_mov_b32 v251, v80 /*v336*/ :: v_dual_mov_b32 v252, v80 /*v336*/
	v_dual_mov_b32 v253, v80 /*v336*/ :: v_dual_mov_b32 v254, v80 /*v336*/
	v_dual_mov_b32 v255, v80 /*v336*/ :: v_dual_mov_b32 v232, v80 /*v336*/
	v_dual_mov_b32 v233, v80 /*v336*/ :: v_dual_mov_b32 v234, v80 /*v336*/
	v_dual_mov_b32 v235, v80 /*v336*/ :: v_dual_mov_b32 v236, v80 /*v336*/
	v_dual_mov_b32 v237, v80 /*v336*/ :: v_dual_mov_b32 v238, v80 /*v336*/
	v_dual_mov_b32 v239, v80 /*v336*/ :: v_dual_mov_b32 v216, v80 /*v336*/
	v_dual_mov_b32 v217, v80 /*v336*/ :: v_dual_mov_b32 v218, v80 /*v336*/
	v_dual_mov_b32 v219, v80 /*v336*/ :: v_dual_mov_b32 v220, v80 /*v336*/
	v_dual_mov_b32 v221, v80 /*v336*/ :: v_dual_mov_b32 v222, v80 /*v336*/
	v_dual_mov_b32 v223, v80 /*v336*/ :: v_dual_mov_b32 v200, v80 /*v336*/
	v_dual_mov_b32 v201, v80 /*v336*/ :: v_dual_mov_b32 v202, v80 /*v336*/
	v_dual_mov_b32 v203, v80 /*v336*/ :: v_dual_mov_b32 v204, v80 /*v336*/
	v_dual_mov_b32 v205, v80 /*v336*/ :: v_dual_mov_b32 v206, v80 /*v336*/
	v_dual_mov_b32 v207, v80 /*v336*/ :: v_dual_mov_b32 v184, v80 /*v336*/
	v_dual_mov_b32 v185, v80 /*v336*/ :: v_dual_mov_b32 v186, v80 /*v336*/
	v_dual_mov_b32 v187, v80 /*v336*/ :: v_dual_mov_b32 v188, v80 /*v336*/
	v_dual_mov_b32 v189, v80 /*v336*/ :: v_dual_mov_b32 v190, v80 /*v336*/
	v_dual_mov_b32 v191, v80 /*v336*/ :: v_dual_mov_b32 v168, v80 /*v336*/
	v_dual_mov_b32 v169, v80 /*v336*/ :: v_dual_mov_b32 v170, v80 /*v336*/
	v_dual_mov_b32 v171, v80 /*v336*/ :: v_dual_mov_b32 v172, v80 /*v336*/
	v_dual_mov_b32 v173, v80 /*v336*/ :: v_dual_mov_b32 v174, v80 /*v336*/
	v_dual_mov_b32 v175, v80 /*v336*/ :: v_dual_mov_b32 v152, v80 /*v336*/
	v_dual_mov_b32 v153, v80 /*v336*/ :: v_dual_mov_b32 v154, v80 /*v336*/
	v_dual_mov_b32 v155, v80 /*v336*/ :: v_dual_mov_b32 v156, v80 /*v336*/
	v_dual_mov_b32 v157, v80 /*v336*/ :: v_dual_mov_b32 v158, v80 /*v336*/
	v_dual_mov_b32 v159, v80 /*v336*/ :: v_dual_mov_b32 v136, v80 /*v336*/
	v_dual_mov_b32 v137, v80 /*v336*/ :: v_dual_mov_b32 v138, v80 /*v336*/
	tensor_load_to_lds s[20:23], s[12:19]
	s_mov_b32 s21, 0x200000
	s_mov_b32 s20, s10
	s_mov_b32 s22, s14
	s_mov_b32 s23, s21
	v_dual_mov_b32 v139, v80 /*v336*/ :: v_dual_mov_b32 v140, v80 /*v336*/
	v_dual_mov_b32 v141, v80 /*v336*/ :: v_dual_mov_b32 v142, v80 /*v336*/
	v_dual_mov_b32 v143, v80 /*v336*/ :: v_dual_mov_b32 v120, v80 /*v336*/
	v_dual_mov_b32 v121, v80 /*v336*/ :: v_dual_mov_b32 v122, v80 /*v336*/
	v_dual_mov_b32 v124, v80 /*v336*/ :: v_dual_mov_b32 v125, v80 /*v336*/
	v_dual_mov_b32 v126, v80 /*v336*/ :: v_dual_mov_b32 v127, v80 /*v336*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v107 /*v363*/, v80 /*v336*/ :: v_dual_mov_b32 v109 /*v365*/, v80 /*v336*/
	v_mov_b32_e32 v111 /*v367*/, v80 /*v336*/
	s_set_vgpr_msb 0x4105
	v_dual_lshlrev_b32 v5, 13, v128 /*v384*/ :: v_dual_lshlrev_b32 v7, 2, v114 /*v370*/
	v_dual_lshlrev_b32 v8, 9, v128 /*v384*/ :: v_dual_mov_b32 v242, v80 /*v336*/
	s_set_vgpr_msb 0x541
	v_lshl_add_u32 v124 /*v380*/, v114 /*v370*/, 4, v0
	s_set_vgpr_msb 0x4101
	v_mov_b32_e32 v123, v80 /*v336*/
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v106 /*v362*/, v80 /*v336*/
	s_set_vgpr_msb 0x4100
	v_or3_b32 v0, v8, v6, v7
	s_set_vgpr_msb 0x45
	v_dual_mov_b32 v110 /*v366*/, v80 /*v336*/ :: v_dual_add_nc_u32 v113 /*v369*/, 0x15800, v124 /*v380*/
	v_dual_mov_b32 v104 /*v360*/, v80 /*v336*/ :: v_dual_mov_b32 v105 /*v361*/, v80 /*v336*/
	s_set_vgpr_msb 0x4541
	v_add_nc_u32_e32 v134 /*v390*/, 0x14610, v0
	v_dual_mov_b32 v108 /*v364*/, v80 /*v336*/ :: v_dual_add_nc_u32 v123 /*v379*/, 0x29e10, v0
	v_add_nc_u32_e32 v121 /*v377*/, 0x2a620, v0
	tensor_load_to_lds s[36:39], s[20:27]
	s_mov_b32 s36, s10
	s_mov_b32 s37, s21
	s_mov_b32 s38, s6
	s_mov_b32 s39, s21
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[80:83], s[36:43]
	s_add_nc_u64 s[80:81], s[44:45], 0xc00
	s_add_co_i32 s45, s33, 0x29e10
	s_lshr_b64 s[82:83], s[80:81], 4
	s_lshr_b64 s[80:81], s[80:81], 2
	s_add_nc_u64 s[54:55], s[82:83], s[54:55]
	s_add_nc_u64 s[60:61], s[80:81], s[60:61]
	s_or_b32 s54, s54, s88
	s_or_b32 s60, s60, s87
	s_mul_u64 s[54:55], s[54:55], 0x6000
	s_mul_u64 s[60:61], s[60:61], 0x180
	s_add_nc_u64 s[96:97], s[50:51], s[54:55]
	s_add_nc_u64 s[98:99], s[58:59], s[60:61]
	s_or_b32 s85, s97, 0x80000000
	s_mov_b32 s94, s96
	s_mov_b32 s95, s85
	s_add_co_i32 s81, s33, 0x14e20
	s_or_b32 s83, s99, 0x80000000
	s_add_co_i32 s33, s33, 0x2a620
	tensor_load_to_lds s[92:95], s[12:19]
	s_mov_b64 s[94:95], s[30:31]
	s_mov_b64 s[92:93], s[28:29]
	s_mov_b32 s93, s81
	s_mov_b32 s94, s98
	s_mov_b32 s95, s83
	s_add_co_i32 s30, s46, 0x100
	s_cmp_gt_u32 s46, 0xfffffeff
	tensor_load_to_lds s[92:95], s[36:43]
	s_add_co_ci_u32 s92, s89, 0
	s_cmp_gt_u32 s90, 0xfffff7ff
	s_mul_u64 s[90:91], s[64:65], 0x30000
	s_add_co_ci_u32 s65, s100, 0
	s_add_nc_u64 s[66:67], s[66:67], s[90:91]
	s_cmp_gt_u32 s96, 0xfffff7ff
	s_set_vgpr_msb 0x4100
	v_mad_nc_u64_u32 v[2:3], 0xc00, v1, s[66:67]
	s_add_co_ci_u32 s66, s85, 0
	s_set_vgpr_msb 4
	v_lshlrev_b32_e32 v1, 4, v112 /*v368*/
	s_set_vgpr_msb 0x440
	v_mov_b32_e32 v131 /*v387*/, s66
	s_cmp_gt_u32 s52, 0xffffffdf
	s_mul_i32 s89, s64, 0x180
	s_add_co_ci_u32 s67, s101, 0
	s_cmp_gt_u32 s84, 0xffffffdf
	v_or3_b32 v125 /*v381*/, v5, v1, v4
	s_set_vgpr_msb 0x4004
	v_or_b32_e32 v1, s76, v112 /*v368*/
	s_add_co_ci_u32 s66, s102, 0
	s_cmp_gt_u32 s98, 0xffffffdf
	s_mul_i32 s84, s64, 0x600
	s_mul_i32 s85, s64, 0x1800
	s_set_vgpr_msb 0x440
	v_mad_nc_u64_u32 v[116:117] /*v[372:373]*/, 0x6000, v1, s[70:71]
	s_add_co_ci_u32 s71, s83, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v135 /*v391*/, s71 :: v_dual_add_nc_u32 v126 /*v382*/, 0x14e20, v0
	s_or_b32 s71, s89, s88
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v1, v7, v6
	s_add_co_i32 s74, s71, s74
	s_add_co_i32 s71, s71, s82
	s_mulk_i32 s74, 0x6000
	s_mulk_i32 s71, 0x6000
	s_or_b32 s74, s74, s72
	s_or_b32 s72, s71, s72
	s_add_co_i32 s74, s74, s70
	s_add_co_i32 s70, s72, s70
	s_add_co_i32 s71, s74, 0x800
	s_add_co_i32 s72, s85, s86
	s_or_b32 s74, s84, s87
	s_set_vgpr_msb 0x50
	v_mad_u32 v117 /*v373*/, 0x6000, s77, v117 /*v373*/
	s_add_co_i32 s68, s72, s68
	s_add_co_i32 s72, s74, s78
	s_add_co_i32 s74, s74, s80
	v_add_nc_u64_e32 v[118:119] /*v[374:375]*/, 0x400, v[2:3]
	s_mulk_i32 s72, 0x180
	s_mulk_i32 s74, 0x180
	v_dual_mov_b32 v129 /*v385*/, s92 :: v_dual_mov_b32 v130 /*v386*/, s65
	v_dual_mov_b32 v132 /*v388*/, s67 :: v_dual_mov_b32 v133 /*v389*/, s66
	s_set_vgpr_msb 0x5045
	v_dual_mov_b32 v15 /*v271*/, v80 /*v336*/ :: v_dual_add_nc_u32 v122 /*v378*/, 0x19c00, v125 /*v381*/
	v_add_nc_u32_e32 v115 /*v371*/, 0x21c00, v125 /*v381*/
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v120 /*v376*/, 0x29c00, v1
	v_or_b32_e32 v127 /*v383*/, 0x14400, v1
	s_add_co_i32 s72, s72, s2
	s_add_co_i32 s2, s74, s2
	s_mov_b64 s[64:65], 0
	s_mov_b64 s[66:67], 0xffffffffffffff00
	s_addk_co_i32 s70, 0x800
	s_add_co_i32 s68, s68, 32
	s_add_co_i32 s72, s72, 32
	s_add_co_i32 s74, s2, 32
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x4000
.LBB0_2:
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s78, s62, s74
	s_add_co_i32 s80, s62, s72
	s_add_co_i32 s82, s62, s68
	s_add_co_i32 s76, s70, s64
	s_add_co_i32 s77, s71, s64
	s_barrier_wait -1
	s_set_vgpr_msb 1
	ds_load_b128 v[88:91], v125 /*v381*/ offset:17408
	ds_load_b128 v[92:95], v125 /*v381*/ offset:17920
	ds_load_b128 v[70:73], v127 /*v383*/
	ds_load_b128 v[0:3], v124 /*v380*/
	ds_load_b128 v[4:7], v124 /*v380*/ offset:32
	ds_load_b128 v[8:11], v124 /*v380*/ offset:64
	ds_load_b128 v[12:15], v124 /*v380*/ offset:96
	ds_load_b128 v[74:77], v134 /*v390*/
	ds_load_b128 v[96:99], v125 /*v381*/ offset:19456
	ds_load_b128 v[100:103], v125 /*v381*/ offset:19968
	ds_load_b128 v[104:107], v125 /*v381*/ offset:21504
	ds_load_b128 v[108:111], v125 /*v381*/ offset:22016
	s_set_vgpr_msb 0x141
	ds_load_b128 v[136:139] /*v[392:395]*/, v125 /*v381*/ offset:23552
	ds_load_b128 v[140:143] /*v[396:399]*/, v125 /*v381*/ offset:24064
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[16:19], v124 /*v380*/ offset:4352
	ds_load_b128 v[20:23], v124 /*v380*/ offset:4384
	ds_load_b128 v[24:27], v124 /*v380*/ offset:4416
	ds_load_b128 v[28:31], v124 /*v380*/ offset:4448
	ds_load_b128 v[80:83], v134 /*v390*/ offset:16
	ds_load_b128 v[84:87], v127 /*v383*/ offset:16
	s_set_vgpr_msb 0x141
	ds_load_b128 v[144:147] /*v[400:403]*/, v125 /*v381*/ offset:18432
	ds_load_b128 v[148:151] /*v[404:407]*/, v125 /*v381*/ offset:18944
	ds_load_b128 v[152:155] /*v[408:411]*/, v125 /*v381*/ offset:20480
	ds_load_b128 v[156:159] /*v[412:415]*/, v125 /*v381*/ offset:20992
	ds_load_b128 v[160:163] /*v[416:419]*/, v125 /*v381*/ offset:22528
	ds_load_b128 v[164:167] /*v[420:423]*/, v125 /*v381*/ offset:23040
	ds_load_b128 v[168:171] /*v[424:427]*/, v125 /*v381*/ offset:24576
	ds_load_b128 v[172:175] /*v[428:431]*/, v125 /*v381*/ offset:25088
	s_set_vgpr_msb 0x4150
	s_wait_dscnt 0x14
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[88:95], v[0:15], v[80:87] /*v[336:343]*/, v74, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[96:103], v[0:15], v[64:71] /*v[320:327]*/, v75, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[104:111], v[0:15], v[48:55] /*v[304:311]*/, v76, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5051
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[136:143] /*v[392:399]*/, v[0:15], v[32:39] /*v[288:295]*/, v77, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[136:143] /*v[392:399]*/, v[16:31], v[224:231], v77, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[104:111], v[16:31], v[240:247], v76, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[96:103], v[16:31], v[0:7] /*v[256:263]*/, v75, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[88:95], v[16:31], v[16:23] /*v[272:279]*/, v74, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5005
	v_mov_b64_e32 v[0:1], s[28:29]
	v_mov_b64_e32 v[2:3], s[30:31]
	v_mov_b32_e32 v3, v129 /*v385*/
	v_nop
	v_add_nc_u64_e32 v[16:17], s[66:67], v[118:119] /*v[374:375]*/
	s_cmp_gt_u32 s30, 0xfffffeff
	s_cselect_b32 vcc_lo, -1, 0
	s_set_vgpr_msb 0x500
	v_readfirstlane_b32 s84, v0
	v_readfirstlane_b32 s85, v1
	v_readfirstlane_b32 s86, v2
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 5
	v_mov_b32_e32 v3, v130 /*v386*/
	v_add_nc_u64_e32 v[0:1], s[64:65], v[116:117] /*v[372:373]*/
	s_cmp_gt_u32 s77, 0xfffff7ff
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v202 /*v458*/, 0, 1, vcc_lo
	tensor_load_to_lds s[84:87], s[4:11]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s69
	s_mov_b32 s86, s77
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v132 /*v388*/
	s_set_vgpr_msb 0x140
	v_add_nc_u64_e32 v[200:201] /*v[456:457]*/, 0x2000, v[0:1]
	s_cselect_b32 s2, -1, 0
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_cmp_gt_u32 s76, 0xfffff7ff
	s_set_vgpr_msb 0x4044
	v_add_co_ci_u32_e64 v203 /*v459*/, null, 0, v129 /*v385*/, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_gt_u32 s82, 0xffffffdf
	v_cndmask_b32_e64 v204 /*v460*/, 0, 1, s2
	v_add_co_ci_u32_e64 v205 /*v461*/, null, 0, v130 /*v386*/, s2
	s_cselect_b32 s2, -1, 0
	s_cmp_gt_u32 s80, 0xffffffdf
	v_cndmask_b32_e64 v206 /*v462*/, 0, 1, vcc_lo
	v_add_co_ci_u32_e64 v207 /*v463*/, null, 0, v131 /*v387*/, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_gt_u32 s78, 0xffffffdf
	v_cndmask_b32_e64 v208 /*v464*/, 0, 1, s2
	v_add_co_ci_u32_e64 v209 /*v465*/, null, 0, v132 /*v388*/, s2
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v210 /*v466*/, 0, 1, vcc_lo
	v_add_co_ci_u32_e64 v211 /*v467*/, null, 0, v133 /*v389*/, vcc_lo
	v_cndmask_b32_e64 v212 /*v468*/, 0, 1, s2
	v_add_co_ci_u32_e64 v213 /*v469*/, null, 0, v135 /*v391*/, s2
	tensor_load_to_lds s[84:87], s[12:19]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s63
	s_mov_b32 s86, s82
	s_set_vgpr_msb 0x4400
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v133 /*v389*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[84:87], s[20:27]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s45
	s_mov_b32 s86, s80
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v131 /*v387*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[84:87], s[36:43]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s35
	s_mov_b32 s86, s76
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v135 /*v391*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[84:87], s[12:19]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s33
	s_mov_b32 s86, s78
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_readfirstlane_b32 s86, v6
	tensor_load_to_lds s[84:87], s[36:43]
	global_prefetch_b8 v[16:17], off scope:SCOPE_SE
	s_set_vgpr_msb 1
	global_prefetch_b8 v[200:201] /*v[456:457]*/, off offset:-2048 scope:SCOPE_SE
	ds_load_b128 v[0:3], v124 /*v380*/ offset:8704
	ds_load_b128 v[4:7], v124 /*v380*/ offset:8736
	ds_load_b128 v[8:11], v124 /*v380*/ offset:8768
	ds_load_b128 v[12:15], v124 /*v380*/ offset:8800
	ds_load_b128 v[16:19], v124 /*v380*/ offset:13056
	ds_load_b128 v[20:23], v124 /*v380*/ offset:13088
	ds_load_b128 v[24:27], v124 /*v380*/ offset:13120
	ds_load_b128 v[28:31], v124 /*v380*/ offset:13152
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v124 /*v380*/ offset:128
	ds_load_b128 v[36:39], v124 /*v380*/ offset:160
	ds_load_b128 v[40:43], v124 /*v380*/ offset:192
	ds_load_b128 v[44:47], v124 /*v380*/ offset:224
	ds_load_b128 v[48:51], v124 /*v380*/ offset:4480
	ds_load_b128 v[52:55], v124 /*v380*/ offset:4512
	ds_load_b128 v[56:59], v124 /*v380*/ offset:4544
	ds_load_b128 v[60:63], v124 /*v380*/ offset:4576
	s_wait_dscnt 0x0
	ds_load_b128 v[64:67], v124 /*v380*/ offset:8832
	ds_load_b128 v[68:71], v124 /*v380*/ offset:8864
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[88:95], v[0:15], v[208:215], v74, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[96:103], v[0:15], v[192:199], v75, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[104:111], v[0:15], v[176:183], v76, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[136:143] /*v[392:399]*/, v[0:15], v[160:167], v77, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[136:143] /*v[392:399]*/, v[16:31], v[96:103] /*v[352:359]*/, v77, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[104:111], v[16:31], v[112:119], v76, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[96:103], v[16:31], v[128:135], v75, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[88:95], v[16:31], v[144:151], v74, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[72:75], v124 /*v380*/ offset:8896
	ds_load_b128 v[76:79], v124 /*v380*/ offset:8928
	ds_load_b128 v[0:3], v124 /*v380*/ offset:13184
	ds_load_b128 v[4:7], v124 /*v380*/ offset:13216
	ds_load_b128 v[8:11], v124 /*v380*/ offset:13248
	ds_load_b128 v[12:15], v124 /*v380*/ offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[144:151] /*v[400:407]*/, v[32:47], v[80:87] /*v[336:343]*/, v80, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[152:159] /*v[408:415]*/, v[32:47], v[64:71] /*v[320:327]*/, v81, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[160:167] /*v[416:423]*/, v[32:47], v[48:55] /*v[304:311]*/, v82, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[32:47], v[32:39] /*v[288:295]*/, v83, v84 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[168:175] /*v[424:431]*/, v[48:63], v[224:231], v83, v85 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[160:167] /*v[416:423]*/, v[48:63], v[240:247], v82, v85 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[152:159] /*v[408:415]*/, v[48:63], v[0:7] /*v[256:263]*/, v81, v85 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[144:151] /*v[400:407]*/, v[48:63], v[16:23] /*v[272:279]*/, v80, v85 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[144:151] /*v[400:407]*/, v[64:79], v[208:215], v80, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[152:159] /*v[408:415]*/, v[64:79], v[192:199], v81, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[160:167] /*v[416:423]*/, v[64:79], v[176:183], v82, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[168:175] /*v[424:431]*/, v[64:79], v[160:167], v83, v86 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[0:15], v[96:103] /*v[352:359]*/, v83, v87 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[160:167] /*v[416:423]*/, v[0:15], v[112:119], v82, v87 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[152:159] /*v[408:415]*/, v[0:15], v[128:135], v81, v87 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[144:151] /*v[400:407]*/, v[0:15], v[144:151], v80, v87 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x141
	ds_load_b128 v[136:139] /*v[392:395]*/, v125 /*v381*/ offset:50176
	ds_load_b128 v[140:143] /*v[396:399]*/, v125 /*v381*/ offset:50688
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[104:107], v126 /*v382*/
	ds_load_b128 v[108:111], v127 /*v383*/
	ds_load_b128 v[8:11], v124 /*v380*/
	ds_load_b128 v[12:15], v124 /*v380*/ offset:32
	ds_load_b128 v[16:19], v124 /*v380*/ offset:64
	ds_load_b128 v[20:23], v124 /*v380*/ offset:96
	s_set_vgpr_msb 0x141
	ds_load_b128 v[144:147] /*v[400:403]*/, v125 /*v381*/ offset:52224
	ds_load_b128 v[148:151] /*v[404:407]*/, v125 /*v381*/ offset:52736
	ds_load_b128 v[152:155] /*v[408:411]*/, v125 /*v381*/ offset:54272
	ds_load_b128 v[156:159] /*v[412:415]*/, v125 /*v381*/ offset:54784
	ds_load_b128 v[160:163] /*v[416:419]*/, v125 /*v381*/ offset:56320
	ds_load_b128 v[164:167] /*v[420:423]*/, v125 /*v381*/ offset:56832
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[96:99], v126 /*v382*/ offset:16
	ds_load_b128 v[100:103], v127 /*v383*/ offset:16
	ds_load_b128 v[0:3], v124 /*v380*/ offset:4352
	ds_load_b128 v[4:7], v124 /*v380*/ offset:4384
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[136:143] /*v[392:399]*/, v[8:23], v[88:95] /*v[344:351]*/, v104, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[144:151] /*v[400:407]*/, v[8:23], v[72:79] /*v[328:335]*/, v105, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x6
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[152:159] /*v[408:415]*/, v[8:23], v[56:63] /*v[312:319]*/, v106, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[160:167] /*v[416:423]*/, v[8:23], v[40:47] /*v[296:303]*/, v107, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[8:11], v124 /*v380*/ offset:4416
	ds_load_b128 v[12:15], v124 /*v380*/ offset:4448
	s_set_vgpr_msb 0x151
	ds_load_b128 v[168:171] /*v[424:427]*/, v125 /*v381*/ offset:51200
	ds_load_b128 v[172:175] /*v[428:431]*/, v125 /*v381*/ offset:51712
	ds_load_b128 v[176:179] /*v[432:435]*/, v125 /*v381*/ offset:53248
	ds_load_b128 v[180:183] /*v[436:439]*/, v125 /*v381*/ offset:53760
	ds_load_b128 v[184:187] /*v[440:443]*/, v125 /*v381*/ offset:55296
	ds_load_b128 v[188:191] /*v[444:447]*/, v125 /*v381*/ offset:55808
	s_wait_dscnt 0x6
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[144:151] /*v[400:407]*/, v[0:15], v[8:15] /*v[264:271]*/, v105, v109 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[192:195] /*v[448:451]*/, v125 /*v381*/ offset:57344
	ds_load_b128 v[196:199] /*v[452:455]*/, v125 /*v381*/ offset:57856
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v124 /*v380*/ offset:13056
	ds_load_b128 v[36:39], v124 /*v380*/ offset:13088
	ds_load_b128 v[40:43], v124 /*v380*/ offset:13120
	ds_load_b128 v[44:47], v124 /*v380*/ offset:13152
	ds_load_b128 v[16:19], v124 /*v380*/ offset:8704
	ds_load_b128 v[20:23], v124 /*v380*/ offset:8736
	ds_load_b128 v[24:27], v124 /*v380*/ offset:8768
	ds_load_b128 v[28:31], v124 /*v380*/ offset:8800
	s_wait_dscnt 0xa
	ds_load_b128 v[64:67], v124 /*v380*/ offset:4480
	ds_load_b128 v[68:71], v124 /*v380*/ offset:4512
	ds_load_b128 v[72:75], v124 /*v380*/ offset:4544
	ds_load_b128 v[76:79], v124 /*v380*/ offset:4576
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[64:79], v[8:15] /*v[264:271]*/, v97, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[152:159] /*v[408:415]*/, v[0:15], v[248:255], v106, v109 matrix_a_fmt:MATRIX_FMT_FP4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[184:191] /*v[440:447]*/, v[64:79], v[248:255], v98, v101 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[160:167] /*v[416:423]*/, v[0:15], v[232:239], v107, v109 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[192:199] /*v[448:455]*/, v[64:79], v[232:239], v99, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[136:143] /*v[392:399]*/, v[0:15], v[24:31] /*v[280:287]*/, v104, v109 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[48:51], v124 /*v380*/ offset:128
	ds_load_b128 v[52:55], v124 /*v380*/ offset:160
	ds_load_b128 v[56:59], v124 /*v380*/ offset:192
	ds_load_b128 v[60:63], v124 /*v380*/ offset:224
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[168:175] /*v[424:431]*/, v[48:63], v[88:95] /*v[344:351]*/, v96, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[48:63], v[72:79] /*v[328:335]*/, v97, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[184:191] /*v[440:447]*/, v[48:63], v[56:63] /*v[312:319]*/, v98, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[48:63], v[40:47] /*v[296:303]*/, v99, v100 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[136:143] /*v[392:399]*/, v[32:47], v[152:159], v104, v111 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[144:151] /*v[400:407]*/, v[32:47], v[136:143], v105, v111 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[152:159] /*v[408:415]*/, v[32:47], v[120:127], v106, v111 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[160:167] /*v[416:423]*/, v[32:47], v[104:111] /*v[360:367]*/, v107, v111 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[160:167] /*v[416:423]*/, v[16:31], v[168:175], v107, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[152:159] /*v[408:415]*/, v[16:31], v[184:191], v106, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[144:151] /*v[400:407]*/, v[16:31], v[200:207], v105, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[136:143] /*v[392:399]*/, v[16:31], v[216:223], v104, v110 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[80:83], v124 /*v380*/ offset:13184
	ds_load_b128 v[84:87], v124 /*v380*/ offset:13216
	ds_load_b128 v[88:91], v124 /*v380*/ offset:13248
	ds_load_b128 v[92:95], v124 /*v380*/ offset:13280
	ds_load_b128 v[0:3], v124 /*v380*/ offset:8832
	ds_load_b128 v[4:7], v124 /*v380*/ offset:8864
	ds_load_b128 v[8:11], v124 /*v380*/ offset:8896
	ds_load_b128 v[12:15], v124 /*v380*/ offset:8928
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[168:175] /*v[424:431]*/, v[80:95], v[152:159], v96, v103 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[176:183] /*v[432:439]*/, v[80:95], v[136:143], v97, v103 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[184:191] /*v[440:447]*/, v[80:95], v[120:127], v98, v103 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[80:95], v[104:111] /*v[360:367]*/, v99, v103 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[192:199] /*v[448:455]*/, v[0:15], v[168:175], v99, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[184:191] /*v[440:447]*/, v[0:15], v[184:191], v98, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[176:183] /*v[432:439]*/, v[0:15], v[200:207], v97, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[168:175] /*v[424:431]*/, v[0:15], v[216:223], v96, v102 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, v[64:79], v[24:31] /*v[280:287]*/, v96, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[88:91], v122 /*v378*/
	ds_load_b128 v[92:95], v122 /*v378*/ offset:512
	ds_load_b128 v[70:73], v120 /*v376*/
	ds_load_b128 v[0:3], v113 /*v369*/
	ds_load_b128 v[4:7], v113 /*v369*/ offset:32
	ds_load_b128 v[8:11], v113 /*v369*/ offset:64
	ds_load_b128 v[12:15], v113 /*v369*/ offset:96
	ds_load_b128 v[74:77], v123 /*v379*/
	ds_load_b128 v[96:99], v122 /*v378*/ offset:2048
	ds_load_b128 v[100:103], v122 /*v378*/ offset:2560
	ds_load_b128 v[104:107], v122 /*v378*/ offset:4096
	ds_load_b128 v[108:111], v122 /*v378*/ offset:4608
	s_set_vgpr_msb 0x141
	ds_load_b128 v[136:139] /*v[392:395]*/, v122 /*v378*/ offset:6144
	ds_load_b128 v[140:143] /*v[396:399]*/, v122 /*v378*/ offset:6656
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[16:19], v113 /*v369*/ offset:4352
	ds_load_b128 v[20:23], v113 /*v369*/ offset:4384
	ds_load_b128 v[24:27], v113 /*v369*/ offset:4416
	ds_load_b128 v[28:31], v113 /*v369*/ offset:4448
	ds_load_b128 v[80:83], v123 /*v379*/ offset:16
	ds_load_b128 v[84:87], v120 /*v376*/ offset:16
	s_set_vgpr_msb 0x141
	ds_load_b128 v[144:147] /*v[400:403]*/, v122 /*v378*/ offset:1024
	ds_load_b128 v[148:151] /*v[404:407]*/, v122 /*v378*/ offset:1536
	ds_load_b128 v[152:155] /*v[408:411]*/, v122 /*v378*/ offset:3072
	ds_load_b128 v[156:159] /*v[412:415]*/, v122 /*v378*/ offset:3584
	ds_load_b128 v[160:163] /*v[416:419]*/, v122 /*v378*/ offset:5120
	ds_load_b128 v[164:167] /*v[420:423]*/, v122 /*v378*/ offset:5632
	ds_load_b128 v[168:171] /*v[424:427]*/, v122 /*v378*/ offset:7168
	ds_load_b128 v[172:175] /*v[428:431]*/, v122 /*v378*/ offset:7680
	s_set_vgpr_msb 0x4150
	s_wait_dscnt 0x14
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[88:95], v[0:15], v[80:87] /*v[336:343]*/, v74, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[96:103], v[0:15], v[64:71] /*v[320:327]*/, v75, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[104:111], v[0:15], v[48:55] /*v[304:311]*/, v76, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5051
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[136:143] /*v[392:399]*/, v[0:15], v[32:39] /*v[288:295]*/, v77, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[136:143] /*v[392:399]*/, v[16:31], v[224:231], v77, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[104:111], v[16:31], v[240:247], v76, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[96:103], v[16:31], v[0:7] /*v[256:263]*/, v75, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[88:95], v[16:31], v[16:23] /*v[272:279]*/, v74, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_mov_b64 s[86:87], s[30:31]
	s_add_co_i32 s2, s30, 0x100
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s31
	s_mov_b32 s86, s2
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5001
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_mov_b32_e32 v3, v203 /*v459*/
	s_add_co_i32 s83, s77, 0x800
	s_add_co_i32 s82, s82, 32
	s_add_co_i32 s80, s80, 32
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s87, v3
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v205 /*v461*/
	s_mov_b64 s[90:91], s[30:31]
	s_mov_b64 s[88:89], s[28:29]
	s_mov_b32 s89, s3
	tensor_load_to_lds s[84:87], s[4:11]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s75
	s_mov_b32 s86, s83
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v209 /*v465*/
	s_add_co_i32 s78, s78, 32
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[84:87], s[12:19]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s73
	s_mov_b32 s86, s82
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v211 /*v467*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[84:87], s[20:27]
	s_mov_b64 s[86:87], s[30:31]
	s_mov_b64 s[84:85], s[28:29]
	s_mov_b32 s85, s79
	s_mov_b32 s86, s80
	v_mov_b64_e32 v[4:5], s[84:85]
	v_mov_b64_e32 v[6:7], s[86:87]
	v_readfirstlane_b32 s87, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v207 /*v463*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s84, v4
	v_readfirstlane_b32 s85, v5
	v_readfirstlane_b32 s86, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[84:87], s[36:43]
	s_add_co_i32 s84, s76, 0x800
	s_mov_b32 s90, s84
	v_mov_b64_e32 v[4:5], s[88:89]
	v_mov_b64_e32 v[6:7], s[90:91]
	v_readfirstlane_b32 s91, v3
	s_set_vgpr_msb 1
	v_mov_b32_e32 v3, v213 /*v469*/
	s_set_vgpr_msb 0x100
	v_readfirstlane_b32 s88, v4
	v_readfirstlane_b32 s89, v5
	v_readfirstlane_b32 s90, v6
	s_delay_alu instid0(VALU_DEP_1)
	tensor_load_to_lds s[88:91], s[12:19]
	s_mov_b64 s[90:91], s[30:31]
	s_mov_b64 s[88:89], s[28:29]
	s_mov_b32 s89, s81
	s_mov_b32 s90, s78
	v_mov_b64_e32 v[4:5], s[88:89]
	v_mov_b64_e32 v[6:7], s[90:91]
	v_readfirstlane_b32 s91, v3
	s_add_co_u32 s30, s2, 0x100
	s_cselect_b32 vcc_lo, -1, 0
	s_addk_co_i32 s77, 0x1000
	v_readfirstlane_b32 s88, v4
	v_readfirstlane_b32 s89, v5
	v_readfirstlane_b32 s90, v6
	s_cmp_lt_u32 s77, s83
	s_set_vgpr_msb 0x45
	v_add_co_ci_u32_e64 v129 /*v385*/, null, v129 /*v385*/, v202 /*v458*/, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_addk_co_i32 s76, 0x1000
	v_add_co_ci_u32_e64 v130 /*v386*/, null, v130 /*v386*/, v204 /*v460*/, vcc_lo
	s_cmp_lt_u32 s76, s84
	s_cselect_b32 vcc_lo, -1, 0
	s_add_co_u32 s2, s82, 32
	s_cselect_b32 s2, -1, 0
	s_add_co_u32 s76, s80, 32
	v_add_co_ci_u32_e64 v131 /*v387*/, null, v131 /*v387*/, v206 /*v462*/, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	s_add_co_u32 s76, s78, 32
	v_add_co_ci_u32_e64 v132 /*v388*/, null, v132 /*v388*/, v208 /*v464*/, s2
	s_cselect_b32 s2, -1, 0
	v_add_co_ci_u32_e64 v133 /*v389*/, null, v133 /*v389*/, v210 /*v466*/, vcc_lo
	v_add_co_ci_u32_e64 v135 /*v391*/, null, v135 /*v391*/, v212 /*v468*/, s2
	tensor_load_to_lds s[88:91], s[36:43]
	global_prefetch_b8 v[118:119] /*v[374:375]*/, off scope:SCOPE_SE
	global_prefetch_b8 v[200:201] /*v[456:457]*/, off scope:SCOPE_SE
	s_set_vgpr_msb 0x4501
	ds_load_b128 v[0:3], v113 /*v369*/ offset:8704
	ds_load_b128 v[4:7], v113 /*v369*/ offset:8736
	ds_load_b128 v[8:11], v113 /*v369*/ offset:8768
	ds_load_b128 v[12:15], v113 /*v369*/ offset:8800
	ds_load_b128 v[16:19], v113 /*v369*/ offset:13056
	ds_load_b128 v[20:23], v113 /*v369*/ offset:13088
	ds_load_b128 v[24:27], v113 /*v369*/ offset:13120
	ds_load_b128 v[28:31], v113 /*v369*/ offset:13152
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v113 /*v369*/ offset:128
	ds_load_b128 v[36:39], v113 /*v369*/ offset:160
	ds_load_b128 v[40:43], v113 /*v369*/ offset:192
	ds_load_b128 v[44:47], v113 /*v369*/ offset:224
	ds_load_b128 v[48:51], v113 /*v369*/ offset:4480
	ds_load_b128 v[52:55], v113 /*v369*/ offset:4512
	ds_load_b128 v[56:59], v113 /*v369*/ offset:4544
	ds_load_b128 v[60:63], v113 /*v369*/ offset:4576
	s_wait_dscnt 0x0
	ds_load_b128 v[64:67], v113 /*v369*/ offset:8832
	ds_load_b128 v[68:71], v113 /*v369*/ offset:8864
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[88:95], v[0:15], v[208:215], v74, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[96:103], v[0:15], v[192:199], v75, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[104:111], v[0:15], v[176:183], v76, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[136:143] /*v[392:399]*/, v[0:15], v[160:167], v77, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[136:143] /*v[392:399]*/, v[16:31], v[96:103] /*v[352:359]*/, v77, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[104:111], v[16:31], v[112:119], v76, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[96:103], v[16:31], v[128:135], v75, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[88:95], v[16:31], v[144:151], v74, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[72:75], v113 /*v369*/ offset:8896
	ds_load_b128 v[76:79], v113 /*v369*/ offset:8928
	ds_load_b128 v[0:3], v113 /*v369*/ offset:13184
	ds_load_b128 v[4:7], v113 /*v369*/ offset:13216
	ds_load_b128 v[8:11], v113 /*v369*/ offset:13248
	ds_load_b128 v[12:15], v113 /*v369*/ offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[144:151] /*v[400:407]*/, v[32:47], v[80:87] /*v[336:343]*/, v80, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[152:159] /*v[408:415]*/, v[32:47], v[64:71] /*v[320:327]*/, v81, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[160:167] /*v[416:423]*/, v[32:47], v[48:55] /*v[304:311]*/, v82, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[32:47], v[32:39] /*v[288:295]*/, v83, v84 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[168:175] /*v[424:431]*/, v[48:63], v[224:231], v83, v85 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[160:167] /*v[416:423]*/, v[48:63], v[240:247], v82, v85 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[152:159] /*v[408:415]*/, v[48:63], v[0:7] /*v[256:263]*/, v81, v85 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[144:151] /*v[400:407]*/, v[48:63], v[16:23] /*v[272:279]*/, v80, v85 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[144:151] /*v[400:407]*/, v[64:79], v[208:215], v80, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[152:159] /*v[408:415]*/, v[64:79], v[192:199], v81, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[160:167] /*v[416:423]*/, v[64:79], v[176:183], v82, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[168:175] /*v[424:431]*/, v[64:79], v[160:167], v83, v86 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[0:15], v[96:103] /*v[352:359]*/, v83, v87 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[160:167] /*v[416:423]*/, v[0:15], v[112:119], v82, v87 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[152:159] /*v[408:415]*/, v[0:15], v[128:135], v81, v87 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[144:151] /*v[400:407]*/, v[0:15], v[144:151], v80, v87 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x141
	ds_load_b128 v[136:139] /*v[392:395]*/, v115 /*v371*/
	ds_load_b128 v[140:143] /*v[396:399]*/, v115 /*v371*/ offset:512
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[104:107], v121 /*v377*/
	ds_load_b128 v[108:111], v120 /*v376*/
	ds_load_b128 v[8:11], v113 /*v369*/
	ds_load_b128 v[12:15], v113 /*v369*/ offset:32
	ds_load_b128 v[16:19], v113 /*v369*/ offset:64
	ds_load_b128 v[20:23], v113 /*v369*/ offset:96
	s_set_vgpr_msb 0x141
	ds_load_b128 v[144:147] /*v[400:403]*/, v115 /*v371*/ offset:2048
	ds_load_b128 v[148:151] /*v[404:407]*/, v115 /*v371*/ offset:2560
	ds_load_b128 v[152:155] /*v[408:411]*/, v115 /*v371*/ offset:4096
	ds_load_b128 v[156:159] /*v[412:415]*/, v115 /*v371*/ offset:4608
	ds_load_b128 v[160:163] /*v[416:419]*/, v115 /*v371*/ offset:6144
	ds_load_b128 v[164:167] /*v[420:423]*/, v115 /*v371*/ offset:6656
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[96:99], v121 /*v377*/ offset:16
	ds_load_b128 v[100:103], v120 /*v376*/ offset:16
	ds_load_b128 v[0:3], v113 /*v369*/ offset:4352
	ds_load_b128 v[4:7], v113 /*v369*/ offset:4384
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[136:143] /*v[392:399]*/, v[8:23], v[88:95] /*v[344:351]*/, v104, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[144:151] /*v[400:407]*/, v[8:23], v[72:79] /*v[328:335]*/, v105, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x6
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[152:159] /*v[408:415]*/, v[8:23], v[56:63] /*v[312:319]*/, v106, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[160:167] /*v[416:423]*/, v[8:23], v[40:47] /*v[296:303]*/, v107, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[8:11], v113 /*v369*/ offset:4416
	ds_load_b128 v[12:15], v113 /*v369*/ offset:4448
	s_set_vgpr_msb 0x151
	ds_load_b128 v[168:171] /*v[424:427]*/, v115 /*v371*/ offset:1024
	ds_load_b128 v[172:175] /*v[428:431]*/, v115 /*v371*/ offset:1536
	ds_load_b128 v[176:179] /*v[432:435]*/, v115 /*v371*/ offset:3072
	ds_load_b128 v[180:183] /*v[436:439]*/, v115 /*v371*/ offset:3584
	ds_load_b128 v[184:187] /*v[440:443]*/, v115 /*v371*/ offset:5120
	ds_load_b128 v[188:191] /*v[444:447]*/, v115 /*v371*/ offset:5632
	s_wait_dscnt 0x6
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[144:151] /*v[400:407]*/, v[0:15], v[8:15] /*v[264:271]*/, v105, v109 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[192:195] /*v[448:451]*/, v115 /*v371*/ offset:7168
	ds_load_b128 v[196:199] /*v[452:455]*/, v115 /*v371*/ offset:7680
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v113 /*v369*/ offset:13056
	ds_load_b128 v[36:39], v113 /*v369*/ offset:13088
	ds_load_b128 v[40:43], v113 /*v369*/ offset:13120
	ds_load_b128 v[44:47], v113 /*v369*/ offset:13152
	ds_load_b128 v[16:19], v113 /*v369*/ offset:8704
	ds_load_b128 v[20:23], v113 /*v369*/ offset:8736
	ds_load_b128 v[24:27], v113 /*v369*/ offset:8768
	ds_load_b128 v[28:31], v113 /*v369*/ offset:8800
	s_wait_dscnt 0xa
	ds_load_b128 v[64:67], v113 /*v369*/ offset:4480
	ds_load_b128 v[68:71], v113 /*v369*/ offset:4512
	ds_load_b128 v[72:75], v113 /*v369*/ offset:4544
	ds_load_b128 v[76:79], v113 /*v369*/ offset:4576
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[64:79], v[8:15] /*v[264:271]*/, v97, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[152:159] /*v[408:415]*/, v[0:15], v[248:255], v106, v109 matrix_a_fmt:MATRIX_FMT_FP4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[184:191] /*v[440:447]*/, v[64:79], v[248:255], v98, v101 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[160:167] /*v[416:423]*/, v[0:15], v[232:239], v107, v109 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[192:199] /*v[448:455]*/, v[64:79], v[232:239], v99, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[136:143] /*v[392:399]*/, v[0:15], v[24:31] /*v[280:287]*/, v104, v109 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[48:51], v113 /*v369*/ offset:128
	ds_load_b128 v[52:55], v113 /*v369*/ offset:160
	ds_load_b128 v[56:59], v113 /*v369*/ offset:192
	ds_load_b128 v[60:63], v113 /*v369*/ offset:224
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[168:175] /*v[424:431]*/, v[48:63], v[88:95] /*v[344:351]*/, v96, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[48:63], v[72:79] /*v[328:335]*/, v97, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[184:191] /*v[440:447]*/, v[48:63], v[56:63] /*v[312:319]*/, v98, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[48:63], v[40:47] /*v[296:303]*/, v99, v100 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[136:143] /*v[392:399]*/, v[32:47], v[152:159], v104, v111 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[144:151] /*v[400:407]*/, v[32:47], v[136:143], v105, v111 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[152:159] /*v[408:415]*/, v[32:47], v[120:127], v106, v111 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[160:167] /*v[416:423]*/, v[32:47], v[104:111] /*v[360:367]*/, v107, v111 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[160:167] /*v[416:423]*/, v[16:31], v[168:175], v107, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[152:159] /*v[408:415]*/, v[16:31], v[184:191], v106, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[144:151] /*v[400:407]*/, v[16:31], v[200:207], v105, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[136:143] /*v[392:399]*/, v[16:31], v[216:223], v104, v110 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[80:83], v113 /*v369*/ offset:13184
	ds_load_b128 v[84:87], v113 /*v369*/ offset:13216
	ds_load_b128 v[88:91], v113 /*v369*/ offset:13248
	ds_load_b128 v[92:95], v113 /*v369*/ offset:13280
	ds_load_b128 v[0:3], v113 /*v369*/ offset:8832
	ds_load_b128 v[4:7], v113 /*v369*/ offset:8864
	ds_load_b128 v[8:11], v113 /*v369*/ offset:8896
	ds_load_b128 v[12:15], v113 /*v369*/ offset:8928
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[168:175] /*v[424:431]*/, v[80:95], v[152:159], v96, v103 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[176:183] /*v[432:439]*/, v[80:95], v[136:143], v97, v103 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[184:191] /*v[440:447]*/, v[80:95], v[120:127], v98, v103 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[80:95], v[104:111] /*v[360:367]*/, v99, v103 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[192:199] /*v[448:455]*/, v[0:15], v[168:175], v99, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[184:191] /*v[440:447]*/, v[0:15], v[184:191], v98, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[176:183] /*v[432:439]*/, v[0:15], v[200:207], v97, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[168:175] /*v[424:431]*/, v[0:15], v[216:223], v96, v102 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, v[64:79], v[24:31] /*v[280:287]*/, v96, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5144
	v_add_nc_u64_e32 v[118:119] /*v[374:375]*/, 0x200, v[118:119] /*v[374:375]*/
	s_add_nc_u64 s[64:65], s[64:65], 0x1000
	s_add_co_i32 s74, s74, 64
	s_add_co_i32 s72, s72, 64
	s_add_co_i32 s68, s68, 64
	s_cmp_lg_u64 s[64:65], 0x5000
	s_set_vgpr_msb 0x4400
	s_cbranch_scc1 .LBB0_2
	s_load_b64 s[24:25], s[0:1], 0x0 nv
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x44
	s_barrier_signal -1
	v_lshlrev_b32_e32 v116 /*v372*/, 6, v128 /*v384*/
	s_mov_b32 s27, 0
	s_mov_b32 s26, 0x60000
	s_barrier_wait -1
	s_set_vgpr_msb 0x4401
	ds_load_b128 v[88:91], v125 /*v381*/ offset:17408
	ds_load_b128 v[92:95], v125 /*v381*/ offset:17920
	ds_load_b128 v[70:73], v127 /*v383*/
	ds_load_b128 v[0:3], v124 /*v380*/
	ds_load_b128 v[4:7], v124 /*v380*/ offset:32
	ds_load_b128 v[8:11], v124 /*v380*/ offset:64
	ds_load_b128 v[12:15], v124 /*v380*/ offset:96
	ds_load_b128 v[74:77], v134 /*v390*/
	ds_load_b128 v[80:83], v134 /*v390*/ offset:16
	ds_load_b128 v[96:99], v125 /*v381*/ offset:19456
	ds_load_b128 v[100:103], v125 /*v381*/ offset:19968
	ds_load_b128 v[104:107], v125 /*v381*/ offset:21504
	ds_load_b128 v[108:111], v125 /*v381*/ offset:22016
	s_set_vgpr_msb 0x141
	ds_load_b128 v[128:131] /*v[384:387]*/, v125 /*v381*/ offset:23552
	ds_load_b128 v[132:135] /*v[388:391]*/, v125 /*v381*/ offset:24064
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[16:19], v124 /*v380*/ offset:4352
	ds_load_b128 v[20:23], v124 /*v380*/ offset:4384
	ds_load_b128 v[24:27], v124 /*v380*/ offset:4416
	ds_load_b128 v[28:31], v124 /*v380*/ offset:4448
	ds_load_b128 v[84:87], v127 /*v383*/ offset:16
	s_set_vgpr_msb 0x141
	ds_load_b128 v[136:139] /*v[392:395]*/, v125 /*v381*/ offset:18432
	ds_load_b128 v[140:143] /*v[396:399]*/, v125 /*v381*/ offset:18944
	ds_load_b128 v[144:147] /*v[400:403]*/, v125 /*v381*/ offset:20480
	ds_load_b128 v[148:151] /*v[404:407]*/, v125 /*v381*/ offset:20992
	ds_load_b128 v[152:155] /*v[408:411]*/, v125 /*v381*/ offset:22528
	ds_load_b128 v[156:159] /*v[412:415]*/, v125 /*v381*/ offset:23040
	ds_load_b128 v[160:163] /*v[416:419]*/, v125 /*v381*/ offset:24576
	ds_load_b128 v[164:167] /*v[420:423]*/, v125 /*v381*/ offset:25088
	s_set_vgpr_msb 0x4150
	s_wait_dscnt 0x14
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[88:95], v[0:15], v[80:87] /*v[336:343]*/, v74, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[96:103], v[0:15], v[64:71] /*v[320:327]*/, v75, v70 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[104:111], v[0:15], v[48:55] /*v[304:311]*/, v76, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5051
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[128:135] /*v[384:391]*/, v[0:15], v[32:39] /*v[288:295]*/, v77, v70 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0x9
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[128:135] /*v[384:391]*/, v[16:31], v[224:231], v77, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[104:111], v[16:31], v[240:247], v76, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[96:103], v[16:31], v[0:7] /*v[256:263]*/, v75, v71 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[88:95], v[16:31], v[16:23] /*v[272:279]*/, v74, v71 matrix_a_fmt:MATRIX_FMT_FP4
	s_add_nc_u64 s[30:31], s[46:47], 0xb00
	s_wait_xcnt 0x0
	s_mov_b32 s1, 0x1000000
	s_mov_b32 s28, 1
	s_bitset1_b32 s31, 31
	s_movk_i32 s5, 0xc00
	s_mov_b32 s4, 16
	s_mov_b32 s2, 0x100000
	s_mov_b32 s0, 0x7500000
	s_mov_b32 s3, s1
	s_mov_b32 s6, s27
	s_mov_b32 s7, s27
	s_brev_b32 s9, 16
	tensor_load_to_lds s[28:31], s[0:7]
	s_add_nc_u64 s[0:1], s[50:51], 0x5800
	s_mov_b64 s[16:17], s[28:29]
	s_add_nc_u64 s[6:7], s[0:1], s[48:49]
	s_mov_b64 s[18:19], s[30:31]
	s_or_b32 s3, s7, 0x80000000
	s_movk_i32 s13, 0x6000
	s_mov_b32 s12, 4
	s_mov_b32 s10, 0x40000
	s_mov_b32 s8, s27
	s_mov_b32 s14, s27
	s_mov_b32 s15, s27
	s_mov_b32 s11, s9
	s_mov_b32 s17, s69
	s_mov_b32 s18, s6
	s_mov_b32 s19, s3
	s_add_nc_u64 s[6:7], s[52:53], 0x160
	s_mov_b64 s[38:39], s[30:31]
	s_or_b32 s3, s7, 0x80000000
	s_mov_b64 s[36:37], s[28:29]
	s_movk_i32 s21, 0x180
	s_mov_b32 s22, s27
	s_mov_b32 s23, s27
	s_mov_b32 s20, s12
	s_mov_b32 s37, s63
	s_mov_b32 s38, s6
	s_mov_b32 s39, s3
	s_add_nc_u64 s[6:7], s[58:59], 0x160
	tensor_load_to_lds s[16:19], s[8:15]
	s_mov_b32 s17, 0x200000
	s_mov_b32 s16, s27
	s_mov_b32 s18, s10
	s_mov_b32 s19, s17
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[36:39], s[16:23]
	s_add_nc_u64 s[18:19], s[6:7], s[56:57]
	s_mov_b64 s[38:39], s[30:31]
	s_or_b32 s3, s19, 0x80000000
	s_mov_b64 s[36:37], s[28:29]
	s_mov_b32 s37, s45
	s_mov_b32 s38, s18
	s_mov_b32 s39, s3
	s_mov_b32 s18, s2
	s_mov_b32 s19, s17
	s_mov_b32 s20, s4
	s_add_nc_u64 s[4:5], s[0:1], s[54:55]
	s_mov_b64 s[0:1], s[28:29]
	s_bitset1_b32 s5, 31
	s_mov_b64 s[2:3], s[30:31]
	s_mov_b32 s1, s35
	s_mov_b32 s2, s4
	s_mov_b32 s3, s5
	s_mov_b32 s29, s33
	tensor_load_to_lds s[36:39], s[16:23]
	tensor_load_to_lds s[0:3], s[8:15]
	s_add_nc_u64 s[0:1], s[6:7], s[60:61]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_or_b32 s31, s1, 0x80000000
	s_mov_b32 s30, s0
	tensor_load_to_lds s[28:31], s[16:23]
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[0:3], v124 /*v380*/ offset:8704
	ds_load_b128 v[4:7], v124 /*v380*/ offset:8736
	ds_load_b128 v[8:11], v124 /*v380*/ offset:8768
	ds_load_b128 v[12:15], v124 /*v380*/ offset:8800
	ds_load_b128 v[16:19], v124 /*v380*/ offset:13056
	ds_load_b128 v[20:23], v124 /*v380*/ offset:13088
	ds_load_b128 v[24:27], v124 /*v380*/ offset:13120
	ds_load_b128 v[28:31], v124 /*v380*/ offset:13152
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v124 /*v380*/ offset:128
	ds_load_b128 v[36:39], v124 /*v380*/ offset:160
	ds_load_b128 v[40:43], v124 /*v380*/ offset:192
	ds_load_b128 v[44:47], v124 /*v380*/ offset:224
	ds_load_b128 v[48:51], v124 /*v380*/ offset:4480
	ds_load_b128 v[52:55], v124 /*v380*/ offset:4512
	ds_load_b128 v[56:59], v124 /*v380*/ offset:4544
	ds_load_b128 v[60:63], v124 /*v380*/ offset:4576
	s_wait_dscnt 0x0
	ds_load_b128 v[64:67], v124 /*v380*/ offset:8832
	ds_load_b128 v[68:71], v124 /*v380*/ offset:8864
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[88:95], v[0:15], v[208:215], v74, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[96:103], v[0:15], v[192:199], v75, v72 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[104:111], v[0:15], v[176:183], v76, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[128:135] /*v[384:391]*/, v[0:15], v[160:167], v77, v72 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[128:135] /*v[384:391]*/, v[16:31], v[96:103] /*v[352:359]*/, v77, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[104:111], v[16:31], v[112:119], v76, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[96:103], v[16:31], v[128:135], v75, v73 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[88:95], v[16:31], v[144:151], v74, v73 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[72:75], v124 /*v380*/ offset:8896
	ds_load_b128 v[76:79], v124 /*v380*/ offset:8928
	ds_load_b128 v[0:3], v124 /*v380*/ offset:13184
	ds_load_b128 v[4:7], v124 /*v380*/ offset:13216
	ds_load_b128 v[8:11], v124 /*v380*/ offset:13248
	ds_load_b128 v[12:15], v124 /*v380*/ offset:13280
	s_wait_dscnt 0x0
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[136:143] /*v[392:399]*/, v[32:47], v[80:87] /*v[336:343]*/, v80, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[144:151] /*v[400:407]*/, v[32:47], v[64:71] /*v[320:327]*/, v81, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[152:159] /*v[408:415]*/, v[32:47], v[48:55] /*v[304:311]*/, v82, v84 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[160:167] /*v[416:423]*/, v[32:47], v[32:39] /*v[288:295]*/, v83, v84 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[160:167] /*v[416:423]*/, v[48:63], v[224:231], v83, v85 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[152:159] /*v[408:415]*/, v[48:63], v[240:247], v82, v85 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[144:151] /*v[400:407]*/, v[48:63], v[0:7] /*v[256:263]*/, v81, v85 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[136:143] /*v[392:399]*/, v[48:63], v[16:23] /*v[272:279]*/, v80, v85 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[136:143] /*v[392:399]*/, v[64:79], v[208:215], v80, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[144:151] /*v[400:407]*/, v[64:79], v[192:199], v81, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[152:159] /*v[408:415]*/, v[64:79], v[176:183], v82, v86 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[160:167] /*v[416:423]*/, v[64:79], v[160:167], v83, v86 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[160:167] /*v[416:423]*/, v[0:15], v[96:103] /*v[352:359]*/, v83, v87 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[152:159] /*v[408:415]*/, v[0:15], v[112:119], v82, v87 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[144:151] /*v[400:407]*/, v[0:15], v[128:135], v81, v87 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[136:143] /*v[392:399]*/, v[0:15], v[144:151], v80, v87 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x141
	ds_load_b128 v[128:131] /*v[384:387]*/, v125 /*v381*/ offset:50176
	ds_load_b128 v[132:135] /*v[388:391]*/, v125 /*v381*/ offset:50688
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[104:107], v126 /*v382*/
	ds_load_b128 v[108:111], v127 /*v383*/
	ds_load_b128 v[8:11], v124 /*v380*/
	ds_load_b128 v[12:15], v124 /*v380*/ offset:32
	ds_load_b128 v[16:19], v124 /*v380*/ offset:64
	ds_load_b128 v[20:23], v124 /*v380*/ offset:96
	s_set_vgpr_msb 0x141
	ds_load_b128 v[136:139] /*v[392:395]*/, v125 /*v381*/ offset:52224
	ds_load_b128 v[140:143] /*v[396:399]*/, v125 /*v381*/ offset:52736
	ds_load_b128 v[144:147] /*v[400:403]*/, v125 /*v381*/ offset:54272
	ds_load_b128 v[148:151] /*v[404:407]*/, v125 /*v381*/ offset:54784
	ds_load_b128 v[152:155] /*v[408:411]*/, v125 /*v381*/ offset:56320
	ds_load_b128 v[156:159] /*v[412:415]*/, v125 /*v381*/ offset:56832
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[96:99], v126 /*v382*/ offset:16
	ds_load_b128 v[100:103], v127 /*v383*/ offset:16
	ds_load_b128 v[0:3], v124 /*v380*/ offset:4352
	ds_load_b128 v[4:7], v124 /*v380*/ offset:4384
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[128:135] /*v[384:391]*/, v[8:23], v[88:95] /*v[344:351]*/, v104, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[136:143] /*v[392:399]*/, v[8:23], v[72:79] /*v[328:335]*/, v105, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x6
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[144:151] /*v[400:407]*/, v[8:23], v[56:63] /*v[312:319]*/, v106, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[152:159] /*v[408:415]*/, v[8:23], v[40:47] /*v[296:303]*/, v107, v108 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[8:11], v124 /*v380*/ offset:4416
	ds_load_b128 v[12:15], v124 /*v380*/ offset:4448
	s_set_vgpr_msb 0x151
	ds_load_b128 v[160:163] /*v[416:419]*/, v125 /*v381*/ offset:51200
	ds_load_b128 v[164:167] /*v[420:423]*/, v125 /*v381*/ offset:51712
	ds_load_b128 v[168:171] /*v[424:427]*/, v125 /*v381*/ offset:53248
	ds_load_b128 v[172:175] /*v[428:431]*/, v125 /*v381*/ offset:53760
	ds_load_b128 v[176:179] /*v[432:435]*/, v125 /*v381*/ offset:55296
	ds_load_b128 v[180:183] /*v[436:439]*/, v125 /*v381*/ offset:55808
	s_wait_dscnt 0x6
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[136:143] /*v[392:399]*/, v[0:15], v[8:15] /*v[264:271]*/, v105, v109 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[184:187] /*v[440:443]*/, v125 /*v381*/ offset:57344
	ds_load_b128 v[188:191] /*v[444:447]*/, v125 /*v381*/ offset:57856
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0xa
	ds_load_b128 v[32:35], v124 /*v380*/ offset:13056
	ds_load_b128 v[36:39], v124 /*v380*/ offset:13088
	ds_load_b128 v[40:43], v124 /*v380*/ offset:13120
	ds_load_b128 v[44:47], v124 /*v380*/ offset:13152
	ds_load_b128 v[16:19], v124 /*v380*/ offset:8704
	ds_load_b128 v[20:23], v124 /*v380*/ offset:8736
	ds_load_b128 v[24:27], v124 /*v380*/ offset:8768
	ds_load_b128 v[28:31], v124 /*v380*/ offset:8800
	s_wait_dscnt 0xa
	ds_load_b128 v[64:67], v124 /*v380*/ offset:4480
	ds_load_b128 v[68:71], v124 /*v380*/ offset:4512
	ds_load_b128 v[72:75], v124 /*v380*/ offset:4544
	ds_load_b128 v[76:79], v124 /*v380*/ offset:4576
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[168:175] /*v[424:431]*/, v[64:79], v[8:15] /*v[264:271]*/, v97, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[144:151] /*v[400:407]*/, v[0:15], v[248:255], v106, v109 matrix_a_fmt:MATRIX_FMT_FP4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[176:183] /*v[432:439]*/, v[64:79], v[248:255], v98, v101 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[152:159] /*v[408:415]*/, v[0:15], v[232:239], v107, v109 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[184:191] /*v[440:447]*/, v[64:79], v[232:239], v99, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[128:135] /*v[384:391]*/, v[0:15], v[24:31] /*v[280:287]*/, v104, v109 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	ds_load_b128 v[48:51], v124 /*v380*/ offset:128
	ds_load_b128 v[52:55], v124 /*v380*/ offset:160
	ds_load_b128 v[56:59], v124 /*v380*/ offset:192
	ds_load_b128 v[60:63], v124 /*v380*/ offset:224
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[160:167] /*v[416:423]*/, v[48:63], v[88:95] /*v[344:351]*/, v96, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[168:175] /*v[424:431]*/, v[48:63], v[72:79] /*v[328:335]*/, v97, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[176:183] /*v[432:439]*/, v[48:63], v[56:63] /*v[312:319]*/, v98, v100 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[184:191] /*v[440:447]*/, v[48:63], v[40:47] /*v[296:303]*/, v99, v100 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[128:135] /*v[384:391]*/, v[32:47], v[152:159], v104, v111 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[136:143] /*v[392:399]*/, v[32:47], v[136:143], v105, v111 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[144:151] /*v[400:407]*/, v[32:47], v[120:127], v106, v111 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[152:159] /*v[408:415]*/, v[32:47], v[104:111] /*v[360:367]*/, v107, v111 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[152:159] /*v[408:415]*/, v[16:31], v[168:175], v107, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[144:151] /*v[400:407]*/, v[16:31], v[184:191], v106, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[136:143] /*v[392:399]*/, v[16:31], v[200:207], v105, v110 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[128:135] /*v[384:391]*/, v[16:31], v[216:223], v104, v110 matrix_a_fmt:MATRIX_FMT_FP4
	ds_load_b128 v[80:83], v124 /*v380*/ offset:13184
	ds_load_b128 v[84:87], v124 /*v380*/ offset:13216
	ds_load_b128 v[88:91], v124 /*v380*/ offset:13248
	ds_load_b128 v[92:95], v124 /*v380*/ offset:13280
	ds_load_b128 v[0:3], v124 /*v380*/ offset:8832
	ds_load_b128 v[4:7], v124 /*v380*/ offset:8864
	ds_load_b128 v[8:11], v124 /*v380*/ offset:8896
	ds_load_b128 v[12:15], v124 /*v380*/ offset:8928
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[160:167] /*v[416:423]*/, v[80:95], v[152:159], v96, v103 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[168:175] /*v[424:431]*/, v[80:95], v[136:143], v97, v103 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[176:183] /*v[432:439]*/, v[80:95], v[120:127], v98, v103 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[184:191] /*v[440:447]*/, v[80:95], v[104:111] /*v[360:367]*/, v99, v103 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5101
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[184:191] /*v[440:447]*/, v[0:15], v[168:175], v99, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[176:183] /*v[432:439]*/, v[0:15], v[184:191], v98, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[168:175] /*v[424:431]*/, v[0:15], v[200:207], v97, v102 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[160:167] /*v[416:423]*/, v[0:15], v[216:223], v96, v102 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x151
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[160:167] /*v[416:423]*/, v[64:79], v[24:31] /*v[280:287]*/, v96, v101 matrix_a_fmt:MATRIX_FMT_FP4
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x5101
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[40:43], v122 /*v378*/
	ds_load_b128 v[44:47], v122 /*v378*/ offset:512
	ds_load_b128 v[104:107], v120 /*v376*/
	ds_load_b128 v[0:3], v113 /*v369*/
	ds_load_b128 v[4:7], v113 /*v369*/ offset:32
	ds_load_b128 v[8:11], v113 /*v369*/ offset:64
	ds_load_b128 v[12:15], v113 /*v369*/ offset:96
	ds_load_b128 v[108:111], v123 /*v379*/
	ds_load_b128 v[48:51], v122 /*v378*/ offset:2048
	ds_load_b128 v[52:55], v122 /*v378*/ offset:2560
	ds_load_b128 v[56:59], v122 /*v378*/ offset:4096
	ds_load_b128 v[60:63], v122 /*v378*/ offset:4608
	ds_load_b128 v[64:67], v122 /*v378*/ offset:6144
	ds_load_b128 v[68:71], v122 /*v378*/ offset:6656
	ds_load_b128 v[16:19], v113 /*v369*/ offset:4352
	ds_load_b128 v[20:23], v113 /*v369*/ offset:4384
	ds_load_b128 v[24:27], v113 /*v369*/ offset:4416
	ds_load_b128 v[28:31], v113 /*v369*/ offset:4448
	ds_load_b128 v[32:35], v123 /*v379*/ offset:16
	ds_load_b128 v[36:39], v120 /*v376*/ offset:16
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0xc
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[40:47], v[0:15], v[80:87] /*v[336:343]*/, v108, v104 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[72:75], v122 /*v378*/ offset:1024
	ds_load_b128 v[76:79], v122 /*v378*/ offset:1536
	ds_load_b128 v[80:83], v122 /*v378*/ offset:3072
	ds_load_b128 v[84:87], v122 /*v378*/ offset:3584
	ds_load_b128 v[88:91], v122 /*v378*/ offset:5120
	ds_load_b128 v[92:95], v122 /*v378*/ offset:5632
	ds_load_b128 v[96:99], v122 /*v378*/ offset:7168
	ds_load_b128 v[100:103], v122 /*v378*/ offset:7680
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[48:55], v[0:15], v[64:71] /*v[320:327]*/, v109, v104 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[56:63], v[0:15], v[48:55] /*v[304:311]*/, v110, v104 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[64:71], v[0:15], v[32:39] /*v[288:295]*/, v111, v104 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[0:3], v113 /*v369*/ offset:8704
	ds_load_b128 v[4:7], v113 /*v369*/ offset:8736
	ds_load_b128 v[8:11], v113 /*v369*/ offset:8768
	ds_load_b128 v[12:15], v113 /*v369*/ offset:8800
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[64:71], v[16:31], v[224:231], v111, v105 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[56:63], v[16:31], v[240:247], v110, v105 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[48:55], v[16:31], v[0:7] /*v[256:263]*/, v109, v105 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[40:47], v[16:31], v[16:23] /*v[272:279]*/, v108, v105 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[16:19], v113 /*v369*/ offset:13056
	ds_load_b128 v[20:23], v113 /*v369*/ offset:13088
	ds_load_b128 v[24:27], v113 /*v369*/ offset:13120
	ds_load_b128 v[28:31], v113 /*v369*/ offset:13152
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[40:47], v[0:15], v[208:215], v108, v106 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[48:55], v[0:15], v[192:199], v109, v106 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[56:63], v[0:15], v[176:183], v110, v106 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[64:71], v[0:15], v[160:167], v111, v106 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[0:3], v113 /*v369*/ offset:128
	ds_load_b128 v[4:7], v113 /*v369*/ offset:160
	ds_load_b128 v[8:11], v113 /*v369*/ offset:192
	ds_load_b128 v[12:15], v113 /*v369*/ offset:224
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[64:71], v[16:31], v[96:103] /*v[352:359]*/, v111, v107 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5000
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[56:63], v[16:31], v[112:119], v110, v107 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[48:55], v[16:31], v[128:135], v109, v107 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[40:47], v[16:31], v[144:151], v108, v107 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[16:19], v113 /*v369*/ offset:4480
	ds_load_b128 v[20:23], v113 /*v369*/ offset:4512
	ds_load_b128 v[24:27], v113 /*v369*/ offset:4544
	ds_load_b128 v[28:31], v113 /*v369*/ offset:4576
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87] /*v[336:343]*/, v[72:79], v[0:15], v[80:87] /*v[336:343]*/, v32, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71] /*v[320:327]*/, v[80:87], v[0:15], v[64:71] /*v[320:327]*/, v33, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[48:55] /*v[304:311]*/, v[88:95], v[0:15], v[48:55] /*v[304:311]*/, v34, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39] /*v[288:295]*/, v[96:103], v[0:15], v[32:39] /*v[288:295]*/, v35, v36 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[0:3], v113 /*v369*/ offset:8832
	ds_load_b128 v[4:7], v113 /*v369*/ offset:8864
	ds_load_b128 v[8:11], v113 /*v369*/ offset:8896
	ds_load_b128 v[12:15], v113 /*v369*/ offset:8928
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[96:103], v[16:31], v[224:231], v35, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[88:95], v[16:31], v[240:247], v34, v37 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7] /*v[256:263]*/, v[80:87], v[16:31], v[0:7] /*v[256:263]*/, v33, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23] /*v[272:279]*/, v[72:79], v[16:31], v[16:23] /*v[272:279]*/, v32, v37 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[16:19], v113 /*v369*/ offset:13184
	ds_load_b128 v[20:23], v113 /*v369*/ offset:13216
	ds_load_b128 v[24:27], v113 /*v369*/ offset:13248
	ds_load_b128 v[28:31], v113 /*v369*/ offset:13280
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[72:79], v[0:15], v[208:215], v32, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[80:87], v[0:15], v[192:199], v33, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[88:95], v[0:15], v[176:183], v34, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[96:103], v[0:15], v[160:167], v35, v38 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[40:43], v115 /*v371*/
	ds_load_b128 v[44:47], v115 /*v371*/ offset:512
	ds_load_b128 v[104:107], v120 /*v376*/
	ds_load_b128 v[0:3], v113 /*v369*/
	ds_load_b128 v[4:7], v113 /*v369*/ offset:32
	ds_load_b128 v[8:11], v113 /*v369*/ offset:64
	ds_load_b128 v[12:15], v113 /*v369*/ offset:96
	ds_load_b128 v[108:111], v121 /*v377*/
	ds_load_b128 v[48:51], v115 /*v371*/ offset:2048
	ds_load_b128 v[52:55], v115 /*v371*/ offset:2560
	ds_load_b128 v[56:59], v115 /*v371*/ offset:4096
	ds_load_b128 v[60:63], v115 /*v371*/ offset:4608
	ds_load_b128 v[64:67], v115 /*v371*/ offset:6144
	ds_load_b128 v[68:71], v115 /*v371*/ offset:6656
	s_set_vgpr_msb 0x150
	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103] /*v[352:359]*/, v[96:103], v[16:31], v[96:103] /*v[352:359]*/, v35, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[96:99], v115 /*v371*/ offset:7168
	ds_load_b128 v[100:103], v115 /*v371*/ offset:7680
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[88:95], v[16:31], v[112:119], v34, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[88:91], v115 /*v371*/ offset:5120
	ds_load_b128 v[92:95], v115 /*v371*/ offset:5632
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[80:87], v[16:31], v[128:135], v33, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[80:83], v115 /*v371*/ offset:3072
	ds_load_b128 v[84:87], v115 /*v371*/ offset:3584
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[72:79], v[16:31], v[144:151], v32, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[16:19], v113 /*v369*/ offset:4352
	ds_load_b128 v[20:23], v113 /*v369*/ offset:4384
	ds_load_b128 v[24:27], v113 /*v369*/ offset:4416
	ds_load_b128 v[28:31], v113 /*v369*/ offset:4448
	ds_load_b128 v[32:35], v121 /*v377*/ offset:16
	ds_load_b128 v[36:39], v120 /*v376*/ offset:16
	ds_load_b128 v[72:75], v115 /*v371*/ offset:1024
	ds_load_b128 v[76:79], v115 /*v371*/ offset:1536
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0xa
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[40:47], v[0:15], v[88:95] /*v[344:351]*/, v108, v104 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[48:55], v[0:15], v[72:79] /*v[328:335]*/, v109, v104 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[56:63], v[0:15], v[56:63] /*v[312:319]*/, v110, v104 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[64:71], v[0:15], v[40:47] /*v[296:303]*/, v111, v104 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[0:3], v113 /*v369*/ offset:8704
	ds_load_b128 v[4:7], v113 /*v369*/ offset:8736
	ds_load_b128 v[8:11], v113 /*v369*/ offset:8768
	ds_load_b128 v[12:15], v113 /*v369*/ offset:8800
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x8
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[64:71], v[16:31], v[232:239], v111, v105 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[56:63], v[16:31], v[248:255], v110, v105 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[48:55], v[16:31], v[8:15] /*v[264:271]*/, v109, v105 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[40:47], v[16:31], v[24:31] /*v[280:287]*/, v108, v105 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[16:19], v113 /*v369*/ offset:13056
	ds_load_b128 v[20:23], v113 /*v369*/ offset:13088
	ds_load_b128 v[24:27], v113 /*v369*/ offset:13120
	ds_load_b128 v[28:31], v113 /*v369*/ offset:13152
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[40:47], v[0:15], v[216:223], v108, v106 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[48:55], v[0:15], v[200:207], v109, v106 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[56:63], v[0:15], v[184:191], v110, v106 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[64:71], v[0:15], v[168:175], v111, v106 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[0:3], v113 /*v369*/ offset:128
	ds_load_b128 v[4:7], v113 /*v369*/ offset:160
	ds_load_b128 v[8:11], v113 /*v369*/ offset:192
	ds_load_b128 v[12:15], v113 /*v369*/ offset:224
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0x4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[64:71], v[16:31], v[104:111] /*v[360:367]*/, v111, v107 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5000
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[56:63], v[16:31], v[120:127], v110, v107 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[48:55], v[16:31], v[136:143], v109, v107 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[40:47], v[16:31], v[152:159], v108, v107 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 1
	ds_load_b128 v[16:19], v113 /*v369*/ offset:4480
	ds_load_b128 v[20:23], v113 /*v369*/ offset:4512
	ds_load_b128 v[24:27], v113 /*v369*/ offset:4544
	ds_load_b128 v[28:31], v113 /*v369*/ offset:4576
	s_set_vgpr_msb 0x150
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95] /*v[344:351]*/, v[72:79], v[0:15], v[88:95] /*v[344:351]*/, v32, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79] /*v[328:335]*/, v[80:87], v[0:15], v[72:79] /*v[328:335]*/, v33, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[56:63] /*v[312:319]*/, v[88:95], v[0:15], v[56:63] /*v[312:319]*/, v34, v36 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[40:47] /*v[296:303]*/, v[96:103], v[0:15], v[40:47] /*v[296:303]*/, v35, v36 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[0:3], v113 /*v369*/ offset:8832
	ds_load_b128 v[4:7], v113 /*v369*/ offset:8864
	ds_load_b128 v[8:11], v113 /*v369*/ offset:8896
	ds_load_b128 v[12:15], v113 /*v369*/ offset:8928
	s_set_vgpr_msb 0x100
	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[96:103], v[16:31], v[232:239], v35, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[88:95], v[16:31], v[248:255], v34, v37 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15] /*v[264:271]*/, v[80:87], v[16:31], v[8:15] /*v[264:271]*/, v33, v37 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31] /*v[280:287]*/, v[72:79], v[16:31], v[24:31] /*v[280:287]*/, v32, v37 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5001
	ds_load_b128 v[16:19], v113 /*v369*/ offset:13184
	ds_load_b128 v[20:23], v113 /*v369*/ offset:13216
	ds_load_b128 v[24:27], v113 /*v369*/ offset:13248
	ds_load_b128 v[28:31], v113 /*v369*/ offset:13280
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x0
	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[72:79], v[0:15], v[216:223], v32, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[80:87], v[0:15], v[200:207], v33, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[88:95], v[0:15], v[184:191], v34, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[96:103], v[0:15], v[168:175], v35, v38 matrix_a_fmt:MATRIX_FMT_FP4
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4
	v_dual_lshlrev_b32 v0, 3, v114 /*v370*/ :: v_dual_bitop2_b32 v1, s34, v112 /*v368*/ bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_or3_b32 v36, v0, v116 /*v372*/, s44
	s_set_vgpr_msb 0x400
	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[80:87], v[16:31], v[136:143], v33, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v33, 0xc00, v1
	v_or_b32_e32 v3, 16, v36
	v_or_b32_e32 v2, 32, v36
	v_or_b32_e32 v0, 48, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v37, 0xc000, v33
	v_add_nc_u32_e32 v38, 0x18000, v33
	v_add_nc_u32_e32 v1, 0x24000, v33
	s_set_vgpr_msb 0x50
	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111] /*v[360:367]*/, v[96:103], v[16:31], v[104:111] /*v[360:367]*/, v35, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 0x5000
	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[88:95], v[16:31], v[120:127], v34, v39 matrix_a_fmt:MATRIX_FMT_FP4
	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[72:79], v[16:31], v[152:159], v32, v39 matrix_a_fmt:MATRIX_FMT_FP4
	s_set_vgpr_msb 4
	v_minimum_f32 v4, 0x40e00000, v80 /*v336*/
	v_minimum_f32 v5, 0x40e00000, v81 /*v337*/
	v_minimum_f32 v6, 0x40e00000, v82 /*v338*/
	v_minimum_f32 v7, 0x40e00000, v83 /*v339*/
	v_minimum_f32 v8, 0x40e00000, v84 /*v340*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v9, 0x3fd9db23, v4 :: v_dual_mul_f32 v10, 0x3fd9db23, v5
	v_mul_f32_e32 v11, 0x3fd9db23, v6
	v_nop
	v_dual_mul_f32 v17, 0x3fd9db23, v7 :: v_dual_mul_f32 v18, 0x3fd9db23, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v12, 0xbfb8aa3b, v9 :: v_dual_mul_f32 v13, 0xbfb8aa3b, v10
	v_mul_f32_e32 v14, 0xbfb8aa3b, v11
	s_set_vgpr_msb 4
	v_minimum_f32 v9, 0x40e00000, v85 /*v341*/
	v_minimum_f32 v10, 0x40e00000, v86 /*v342*/
	v_exp_f32_e32 v15, v12
	v_exp_f32_e32 v16, v13
	v_exp_f32_e32 v14, v14
	v_minimum_f32 v11, 0x40e00000, v87 /*v343*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v19, 0x3fd9db23, v9 :: v_dual_mul_f32 v20, 0x3fd9db23, v10
	v_dual_mul_f32 v17, 0xbfb8aa3b, v17 :: v_dual_mul_f32 v18, 0xbfb8aa3b, v18
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_dual_add_f32 v15, 1.0, v15 :: v_dual_add_f32 v16, 1.0, v16
	v_add_f32_e32 v21, 1.0, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v19, 0xbfb8aa3b, v19 :: v_dual_mul_f32 v20, 0xbfb8aa3b, v20
	v_rcp_f32_e32 v14, v15
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v15, v16
	v_nop
	v_mul_f32_e32 v16, 0x3fd9db23, v11
	v_exp_f32_e32 v22, v17
	v_exp_f32_e32 v23, v18
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_mul_f32_e32 v16, 0xbfb8aa3b, v16
	v_rcp_f32_e32 v18, v21
	s_mov_b32 s0, 0xc0e00000
	v_pk_mul_f32 v[4:5], v[4:5], v[14:15]
	v_add_f32_e32 v21, 1.0, v22
	v_exp_f32_e32 v24, v16
	v_add_f32_e32 v22, 1.0, v23
	v_dual_add_f32 v23, 1.0, v19 :: v_dual_add_f32 v25, 1.0, v20
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v19, v21
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v12, v88 /*v344*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v20, v22
	v_rcp_f32_e32 v21, v23
	v_add_f32_e32 v24, 1.0, v24
	v_rcp_f32_e32 v22, v25
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v13, v89 /*v345*/, 0x40e00000, s0
	v_minimummaximum_f32 v25, v93 /*v349*/, 0x40e00000, s0
	v_minimummaximum_f32 v26, v94 /*v350*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v23, v24
	v_nop
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v24, v92 /*v348*/, 0x40e00000, s0
	v_minimummaximum_f32 v27, v95 /*v351*/, 0x40e00000, s0
	v_minimummaximum_f32 v16, v90 /*v346*/, 0x40e00000, s0
	v_minimummaximum_f32 v17, v91 /*v347*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[6:7], v[6:7], v[18:19]
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[22:23]
	v_pk_add_f32 v[14:15], v[26:27], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[18:19], v[24:25], 1.0 op_sel_hi:[1,0]
	s_set_vgpr_msb 4
	v_minimum_f32 v20, 0x40e00000, v64 /*v320*/
	v_pk_add_f32 v[16:17], v[16:17], 1.0 op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[12:13], v[4:5], v[12:13]
	s_set_vgpr_msb 4
	v_minimum_f32 v21, 0x40e00000, v65 /*v321*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[4:5], v[10:11], v[14:15]
	v_pk_mul_f32 v[8:9], v[8:9], v[18:19]
	v_mul_f32_e32 v14, 0x3fd9db23, v20
	v_pk_mul_f32 v[10:11], v[6:7], v[16:17]
	v_mul_f32_e32 v15, 0x3fd9db23, v21
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_mul_f32_e32 v8, 0xbfb8aa3b, v14
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_mul_f32_e32 v9, 0xbfb8aa3b, v15
	v_cvt_pk_bf16_f32 v4, v12, v13
	v_add_lshl_u32 v10, v36, v33, 1
	v_exp_f32_e32 v11, v8
	v_nop
	s_set_vgpr_msb 4
	v_minimum_f32 v8, 0x40e00000, v66 /*v322*/
	v_exp_f32_e32 v12, v9
	v_nop
	v_minimum_f32 v9, 0x40e00000, v67 /*v323*/
	s_wait_kmcnt 0x0
	buffer_store_b128 v[4:7], v10, s[24:27], null offen
	v_minimum_f32 v13, 0x40e00000, v71 /*v327*/
	s_wait_xcnt 0x0
	s_set_vgpr_msb 0x401
	v_mul_f32_e32 v5, 0x3fd9db23, v8
	v_minimummaximum_f32 v4, v72 /*v328*/, 0x40e00000, s0
	v_dual_add_f32 v6, 1.0, v11 :: v_dual_mul_f32 v11, 0x3fd9db23, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v7, 1.0, v12 :: v_dual_mul_f32 v10, 0xbfb8aa3b, v5
	s_set_vgpr_msb 0x104
	v_minimum_f32 v12, 0x40e00000, v70 /*v326*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v19, 0x3fd9db23, v13
	v_mul_f32_e32 v15, 0xbfb8aa3b, v11
	s_set_vgpr_msb 4
	v_minimum_f32 v11, 0x40e00000, v69 /*v325*/
	v_exp_f32_e32 v14, v10
	v_nop
	v_minimum_f32 v10, 0x40e00000, v68 /*v324*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v18, 0x3fd9db23, v12
	v_exp_f32_e32 v15, v15
	v_mul_f32_e32 v17, 0x3fd9db23, v11
	v_rcp_f32_e32 v6, v6
	v_mul_f32_e32 v16, 0x3fd9db23, v10
	v_rcp_f32_e32 v7, v7
	v_add_f32_e32 v22, 1.0, v14
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v5, v73 /*v329*/, 0x40e00000, s0
	v_minimummaximum_f32 v26, v78 /*v334*/, 0x40e00000, s0
	v_mul_f32_e32 v14, 0xbfb8aa3b, v16
	v_dual_mul_f32 v16, 0xbfb8aa3b, v17 :: v_dual_mul_f32 v17, 0xbfb8aa3b, v18
	v_mul_f32_e32 v18, 0xbfb8aa3b, v19
	v_add_f32_e32 v19, 1.0, v15
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v23, v14
	v_exp_f32_e32 v24, v16
	v_exp_f32_e32 v25, v17
	v_exp_f32_e32 v18, v18
	v_rcp_f32_e32 v16, v22
	v_rcp_f32_e32 v17, v19
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v27, v79 /*v335*/, 0x40e00000, s0
	v_minimummaximum_f32 v14, v74 /*v330*/, 0x40e00000, s0
	v_dual_add_f32 v19, 1.0, v23 :: v_dual_add_f32 v23, 1.0, v24
	s_delay_alu instid0(TRANS32_DEP_3)
	v_dual_add_f32 v24, 1.0, v25 :: v_dual_add_f32 v25, 1.0, v18
	v_minimummaximum_f32 v15, v75 /*v331*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v22, v19
	v_rcp_f32_e32 v23, v23
	v_pk_mul_f32 v[8:9], v[8:9], v[16:17]
	s_set_vgpr_msb 4
	v_minimum_f32 v16, 0x40e00000, v48 /*v304*/
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[6:7], v[20:21], v[6:7]
	s_set_vgpr_msb 4
	v_minimum_f32 v17, 0x40e00000, v49 /*v305*/
	v_pk_add_f32 v[4:5], v[4:5], 1.0 op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[10:11], v[10:11], v[22:23]
	v_mul_f32_e32 v23, 0x3fd9db23, v16
	s_set_vgpr_msb 4
	v_minimum_f32 v22, 0x40e00000, v50 /*v306*/
	v_pk_add_f32 v[20:21], v[26:27], 1.0 op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_pk_add_f32 v[14:15], v[14:15], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v23, 0xbfb8aa3b, v23
	v_mul_f32_e32 v26, 0x3fd9db23, v17
	v_pk_mul_f32 v[24:25], v[6:7], v[4:5]
	v_mul_f32_e32 v7, 0x3fd9db23, v22
	v_pk_mul_f32 v[4:5], v[12:13], v[20:21]
	v_exp_f32_e32 v12, v23
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v18, v76 /*v332*/, 0x40e00000, s0
	v_minimummaximum_f32 v19, v77 /*v333*/, 0x40e00000, s0
	v_mul_f32_e32 v6, 0xbfb8aa3b, v26
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[8:9], v[8:9], v[14:15]
	v_mul_f32_e32 v14, 0xbfb8aa3b, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_pk_add_f32 v[18:19], v[18:19], 1.0 op_sel_hi:[1,0]
	v_exp_f32_e32 v13, v6
	v_cvt_pk_bf16_f32 v5, v8, v9
	v_add_f32_e32 v9, 1.0, v12
	v_exp_f32_e32 v12, v14
	v_pk_mul_f32 v[10:11], v[10:11], v[18:19]
	s_set_vgpr_msb 4
	v_minimum_f32 v23, 0x40e00000, v51 /*v307*/
	v_minimum_f32 v14, 0x40e00000, v54 /*v310*/
	v_minimum_f32 v15, 0x40e00000, v55 /*v311*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v4, v24, v25
	v_cvt_pk_bf16_f32 v6, v10, v11
	v_dual_add_f32 v11, 1.0, v13 :: v_dual_add_f32 v19, 1.0, v12
	s_set_vgpr_msb 4
	v_minimum_f32 v12, 0x40e00000, v52 /*v308*/
	v_minimum_f32 v13, 0x40e00000, v53 /*v309*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v18, 0x3fd9db23, v23
	v_dual_mul_f32 v24, 0x3fd9db23, v14 :: v_dual_mul_f32 v25, 0x3fd9db23, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, 0x3fd9db23, v12 :: v_dual_mul_f32 v21, 0x3fd9db23, v13
	v_mul_f32_e32 v18, 0xbfb8aa3b, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v24, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v25
	v_dual_mul_f32 v20, 0xbfb8aa3b, v20 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v26, v18
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v27, v20
	v_exp_f32_e32 v21, v21
	v_rcp_f32_e32 v10, v9
	v_rcp_f32_e32 v11, v11
	v_rcp_f32_e32 v20, v19
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v29, 1.0, v24
	v_dual_add_f32 v30, 1.0, v25 :: v_dual_add_f32 v27, 1.0, v27
	v_add_f32_e32 v28, 1.0, v21
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v21, v26
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v8, v56 /*v312*/, 0x40e00000, s0
	v_minimummaximum_f32 v9, v57 /*v313*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v24, v27
	v_rcp_f32_e32 v25, v28
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v18, v58 /*v314*/, 0x40e00000, s0
	v_minimummaximum_f32 v19, v59 /*v315*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v26, v29
	v_rcp_f32_e32 v27, v30
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v28, v60 /*v316*/, 0x40e00000, s0
	v_minimummaximum_f32 v29, v61 /*v317*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[10:11], v[16:17], v[10:11]
	v_pk_mul_f32 v[16:17], v[22:23], v[20:21]
	s_set_vgpr_msb 5
	v_minimum_f32 v22, 0x40e00000, v32 /*v288*/
	v_minimummaximum_f32 v30, v62 /*v318*/, 0x40e00000, s0
	v_minimummaximum_f32 v31, v63 /*v319*/, 0x40e00000, s0
	s_set_vgpr_msb 0x504
	v_pk_add_f32 v[8:9], v[8:9], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[18:19], v[18:19], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v23, 0x40e00000, v33 /*v289*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_pk_add_f32 v[24:25], v[28:29], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v28, 0x3fd9db23, v22
	v_pk_mul_f32 v[14:15], v[14:15], v[26:27]
	v_pk_add_f32 v[20:21], v[30:31], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[10:11], v[8:9]
	v_pk_mul_f32 v[8:9], v[16:17], v[18:19]
	v_mul_f32_e32 v17, 0x3fd9db23, v23
	v_mul_f32_e32 v18, 0xbfb8aa3b, v28
	s_set_vgpr_msb 4
	v_minimum_f32 v16, 0x40e00000, v34 /*v290*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[10:11], v[14:15], v[20:21]
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_mul_f32_e32 v14, 0xbfb8aa3b, v17
	v_exp_f32_e32 v17, v18
	v_add_lshl_u32 v32, v3, v33, 1
	v_mul_f32_e32 v15, 0x3fd9db23, v16
	v_cvt_pk_bf16_f32 v11, v10, v11
	v_cvt_pk_bf16_f32 v10, v12, v13
	v_exp_f32_e32 v12, v14
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v26, v27
	v_add_lshl_u32 v14, v2, v33, 1
	v_mul_f32_e32 v13, 0xbfb8aa3b, v15
	v_add_f32_e32 v15, 1.0, v17
	buffer_store_b128 v[4:7], v32, s[24:27], null offen
	s_set_vgpr_msb 4
	v_minimum_f32 v17, 0x40e00000, v35 /*v291*/
	buffer_store_b128 v[8:11], v14, s[24:27], null offen
	s_wait_xcnt 0x0
	v_minimum_f32 v8, 0x40e00000, v36 /*v292*/
	v_minimum_f32 v9, 0x40e00000, v37 /*v293*/
	v_minimum_f32 v10, 0x40e00000, v38 /*v294*/
	v_minimum_f32 v11, 0x40e00000, v39 /*v295*/
	v_rcp_f32_e32 v6, v15
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v14, 0x3fd9db23, v8 :: v_dual_mul_f32 v15, 0x3fd9db23, v9
	v_dual_add_f32 v7, 1.0, v12 :: v_dual_mul_f32 v12, 0x3fd9db23, v17
	v_dual_mul_f32 v18, 0x3fd9db23, v10 :: v_dual_mul_f32 v19, 0x3fd9db23, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v14, 0xbfb8aa3b, v14 :: v_dual_mul_f32 v15, 0xbfb8aa3b, v15
	v_mul_f32_e32 v12, 0xbfb8aa3b, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v18, 0xbfb8aa3b, v18 :: v_dual_mul_f32 v19, 0xbfb8aa3b, v19
	v_exp_f32_e32 v21, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v15, v15
	v_exp_f32_e32 v20, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v18, v18
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v13, v13
	v_rcp_f32_e32 v7, v7
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v4, v40 /*v296*/, 0x40e00000, s0
	v_dual_add_f32 v21, 1.0, v21 :: v_dual_add_f32 v24, 1.0, v15
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v25, 1.0, v18
	s_delay_alu instid0(TRANS32_DEP_2)
	v_dual_add_f32 v26, 1.0, v19 :: v_dual_add_f32 v13, 1.0, v13
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v18, v21
	v_rcp_f32_e32 v19, v24
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v5, v41 /*v297*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v15, v20
	v_rcp_f32_e32 v20, v25
	v_rcp_f32_e32 v21, v26
	v_rcp_f32_e32 v14, v13
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v24, v44 /*v300*/, 0x40e00000, s0
	v_minimummaximum_f32 v25, v45 /*v301*/, 0x40e00000, s0
	v_minimummaximum_f32 v12, v42 /*v298*/, 0x40e00000, s0
	v_minimummaximum_f32 v13, v43 /*v299*/, 0x40e00000, s0
	v_minimummaximum_f32 v26, v46 /*v302*/, 0x40e00000, s0
	v_minimummaximum_f32 v27, v47 /*v303*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[6:7], v[22:23], v[6:7]
	v_pk_mul_f32 v[8:9], v[8:9], v[18:19]
	s_set_vgpr_msb 4
	v_minimum_f32 v18, 0x40e00000, v16 /*v272*/
	v_pk_add_f32 v[4:5], v[4:5], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v19, 0x40e00000, v17 /*v273*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[10:11], v[10:11], v[20:21]
	v_pk_add_f32 v[20:21], v[24:25], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[16:17], v[14:15]
	v_pk_add_f32 v[16:17], v[26:27], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v24, 0x3fd9db23, v18
	v_pk_mul_f32 v[22:23], v[6:7], v[4:5]
	v_mul_f32_e32 v6, 0x3fd9db23, v19
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	v_pk_mul_f32 v[4:5], v[10:11], v[16:17]
	v_mul_f32_e32 v16, 0xbfb8aa3b, v24
	v_pk_mul_f32 v[10:11], v[14:15], v[12:13]
	v_mul_f32_e32 v12, 0xbfb8aa3b, v6
	v_cvt_pk_bf16_f32 v6, v8, v9
	s_set_vgpr_msb 4
	v_minimum_f32 v8, 0x40e00000, v18 /*v274*/
	v_exp_f32_e32 v13, v16
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_exp_f32_e32 v12, v12
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_mul_f32_e32 v11, 0x3fd9db23, v8
	s_set_vgpr_msb 4
	v_minimum_f32 v9, 0x40e00000, v19 /*v275*/
	v_minimum_f32 v17, 0x40e00000, v23 /*v279*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v4, v22, v23
	v_add_f32_e32 v13, 1.0, v13
	v_dual_mul_f32 v15, 0xbfb8aa3b, v11 :: v_dual_add_f32 v14, 1.0, v12
	v_mul_f32_e32 v16, 0x3fd9db23, v9
	v_mul_f32_e32 v25, 0x3fd9db23, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v12, v13
	v_exp_f32_e32 v20, v15
	v_rcp_f32_e32 v13, v14
	v_mul_f32_e32 v21, 0xbfb8aa3b, v16
	s_set_vgpr_msb 5
	v_minimum_f32 v14, 0x40e00000, v20 /*v276*/
	v_minimum_f32 v15, 0x40e00000, v21 /*v277*/
	v_minimum_f32 v16, 0x40e00000, v22 /*v278*/
	v_minimummaximum_f32 v10, v24 /*v280*/, 0x40e00000, s0
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v21, v21
	v_dual_mul_f32 v22, 0x3fd9db23, v14 :: v_dual_mul_f32 v23, 0x3fd9db23, v15
	v_mul_f32_e32 v24, 0x3fd9db23, v16
	v_add_f32_e32 v26, 1.0, v20
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v11, v25 /*v281*/, 0x40e00000, s0
	v_mul_f32_e32 v20, 0xbfb8aa3b, v22
	v_dual_mul_f32 v22, 0xbfb8aa3b, v23 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v24
	v_mul_f32_e32 v24, 0xbfb8aa3b, v25
	v_add_f32_e32 v25, 1.0, v21
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v27, v20
	v_exp_f32_e32 v28, v22
	v_exp_f32_e32 v29, v23
	v_exp_f32_e32 v24, v24
	v_rcp_f32_e32 v23, v25
	v_rcp_f32_e32 v22, v26
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v30, v30 /*v286*/, 0x40e00000, s0
	v_minimummaximum_f32 v31, v31 /*v287*/, 0x40e00000, s0
	v_dual_add_f32 v25, 1.0, v27 :: v_dual_add_f32 v27, 1.0, v28
	s_delay_alu instid0(TRANS32_DEP_3)
	v_dual_add_f32 v28, 1.0, v29 :: v_dual_add_f32 v29, 1.0, v24
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[12:13], v[18:19], v[12:13]
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v26, v25
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_pk_mul_f32 v[8:9], v[8:9], v[22:23]
	v_pk_add_f32 v[22:23], v[30:31], 1.0 op_sel_hi:[1,0]
	s_set_vgpr_msb 4
	v_minimum_f32 v19, 0x40e00000, v1 /*v257*/
	v_pk_add_f32 v[10:11], v[10:11], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v18, 0x40e00000, v0 /*v256*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[14:15], v[14:15], v[26:27]
	s_set_vgpr_msb 4
	v_minimum_f32 v26, 0x40e00000, v2 /*v258*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[16:17], v[16:17], v[28:29]
	v_mul_f32_e32 v28, 0x3fd9db23, v19
	v_pk_mul_f32 v[12:13], v[12:13], v[10:11]
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v24, v28 /*v284*/, 0x40e00000, s0
	v_minimummaximum_f32 v25, v29 /*v285*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[10:11], v[16:17], v[22:23]
	v_mul_f32_e32 v17, 0x3fd9db23, v26
	v_mul_f32_e32 v27, 0x3fd9db23, v18
	v_mul_f32_e32 v16, 0xbfb8aa3b, v28
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v20, v26 /*v282*/, 0x40e00000, s0
	v_minimummaximum_f32 v21, v27 /*v283*/, 0x40e00000, s0
	v_mul_f32_e32 v17, 0xbfb8aa3b, v17
	v_pk_add_f32 v[24:25], 1.0, v[24:25] op_sel_hi:[0,1]
	v_mul_f32_e32 v27, 0xbfb8aa3b, v27
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v16, v16
	v_pk_add_f32 v[20:21], v[20:21], 1.0 op_sel_hi:[1,0]
	v_exp_f32_e32 v17, v17
	v_pk_mul_f32 v[14:15], v[14:15], v[24:25]
	v_exp_f32_e32 v22, v27
	v_cvt_pk_bf16_f32 v11, v10, v11
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	s_set_vgpr_msb 4
	v_minimum_f32 v27, 0x40e00000, v3 /*v259*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v10, v14, v15
	v_dual_add_f32 v15, 1.0, v16 :: v_dual_add_f32 v23, 1.0, v17
	s_set_vgpr_msb 4
	v_minimum_f32 v16, 0x40e00000, v4 /*v260*/
	v_minimum_f32 v17, 0x40e00000, v5 /*v261*/
	v_minimum_f32 v20, 0x40e00000, v6 /*v262*/
	v_minimum_f32 v21, 0x40e00000, v7 /*v263*/
	s_set_vgpr_msb 0x400
	v_add_f32_e32 v14, 1.0, v22
	v_mul_f32_e32 v22, 0x3fd9db23, v27
	v_dual_mul_f32 v24, 0x3fd9db23, v16 :: v_dual_mul_f32 v25, 0x3fd9db23, v17
	v_dual_mul_f32 v28, 0x3fd9db23, v20 :: v_dual_mul_f32 v29, 0x3fd9db23, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v22, 0xbfb8aa3b, v22
	v_dual_mul_f32 v24, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v28, 0xbfb8aa3b, v28 :: v_dual_mul_f32 v29, 0xbfb8aa3b, v29
	v_exp_f32_e32 v30, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v31, v24
	v_exp_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_add_lshl_u32 v39, v0, v33, 1
	v_rcp_f32_e32 v14, v14
	v_rcp_f32_e32 v15, v15
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v25 :: v_dual_add_f32 v33, 1.0, v28
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v34, 1.0, v29
	v_rcp_f32_e32 v24, v23
	v_rcp_f32_e32 v25, v30
	v_rcp_f32_e32 v28, v31
	v_rcp_f32_e32 v29, v32
	v_rcp_f32_e32 v30, v33
	v_rcp_f32_e32 v31, v34
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v12, v13
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v12, v8 /*v264*/, 0x40e00000, s0
	v_minimummaximum_f32 v13, v9 /*v265*/, 0x40e00000, s0
	v_minimummaximum_f32 v32, v12 /*v268*/, 0x40e00000, s0
	v_minimummaximum_f32 v33, v13 /*v269*/, 0x40e00000, s0
	v_minimummaximum_f32 v34, v14 /*v270*/, 0x40e00000, s0
	v_minimummaximum_f32 v35, v15 /*v271*/, 0x40e00000, s0
	v_minimummaximum_f32 v22, v10 /*v266*/, 0x40e00000, s0
	v_minimummaximum_f32 v23, v11 /*v267*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[14:15], v[18:19], v[14:15]
	v_pk_mul_f32 v[18:19], v[26:27], v[24:25]
	v_pk_mul_f32 v[16:17], v[16:17], v[28:29]
	v_pk_mul_f32 v[20:21], v[20:21], v[30:31]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[24:25], v[34:35], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[26:27], v[32:33], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v28, 0x40e00000, v240
	v_pk_add_f32 v[22:23], v[22:23], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[14:15], v[12:13]
	v_minimum_f32 v29, 0x40e00000, v241
	v_pk_mul_f32 v[12:13], v[20:21], v[24:25]
	v_pk_mul_f32 v[16:17], v[16:17], v[26:27]
	v_mul_f32_e32 v20, 0x3fd9db23, v28
	v_pk_mul_f32 v[18:19], v[18:19], v[22:23]
	v_mul_f32_e32 v21, 0x3fd9db23, v29
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v16, v17
	v_mul_f32_e32 v16, 0xbfb8aa3b, v20
	v_cvt_pk_bf16_f32 v13, v18, v19
	v_mul_f32_e32 v17, 0xbfb8aa3b, v21
	v_add_lshl_u32 v40, v36, v37, 1
	buffer_store_b128 v[4:7], v39, s[24:27], null offen
	v_exp_f32_e32 v19, v16
	v_nop
	v_minimum_f32 v16, 0x40e00000, v242
	v_exp_f32_e32 v20, v17
	v_nop
	v_minimum_f32 v17, 0x40e00000, v243
	v_cvt_pk_bf16_f32 v12, v30, v31
	v_add_lshl_u32 v18, v3, v37, 1
	s_wait_xcnt 0x0
	v_mul_f32_e32 v5, 0x3fd9db23, v16
	buffer_store_b128 v[8:11], v40, s[24:27], null offen
	s_wait_xcnt 0x0
	v_mul_f32_e32 v9, 0x3fd9db23, v17
	v_minimum_f32 v10, 0x40e00000, v246
	buffer_store_b128 v[12:15], v18, s[24:27], null offen
	v_mul_f32_e32 v8, 0xbfb8aa3b, v5
	v_minimum_f32 v11, 0x40e00000, v247
	s_wait_xcnt 0x0
	v_mul_f32_e32 v13, 0xbfb8aa3b, v9
	v_minimum_f32 v9, 0x40e00000, v245
	v_add_f32_e32 v6, 1.0, v19
	v_exp_f32_e32 v12, v8
	v_nop
	v_minimum_f32 v8, 0x40e00000, v244
	v_dual_mul_f32 v18, 0x3fd9db23, v10 :: v_dual_mul_f32 v15, 0x3fd9db23, v9
	v_mul_f32_e32 v19, 0x3fd9db23, v11
	v_add_f32_e32 v7, 1.0, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v14, 0x3fd9db23, v8
	v_exp_f32_e32 v13, v13
	v_add_f32_e32 v20, 1.0, v12
	v_rcp_f32_e32 v6, v6
	v_rcp_f32_e32 v7, v7
	v_mul_f32_e32 v12, 0xbfb8aa3b, v14
	v_dual_mul_f32 v14, 0xbfb8aa3b, v15 :: v_dual_mul_f32 v15, 0xbfb8aa3b, v18
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v18, 0xbfb8aa3b, v19 :: v_dual_add_f32 v19, 1.0, v13
	v_exp_f32_e32 v21, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v22, v14
	v_exp_f32_e32 v23, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v18, v18
	v_rcp_f32_e32 v15, v19
	v_rcp_f32_e32 v14, v20
	v_minimummaximum_f32 v4, v248, 0x40e00000, s0
	v_minimummaximum_f32 v5, v249, 0x40e00000, s0
	v_dual_add_f32 v19, 1.0, v21 :: v_dual_add_f32 v21, 1.0, v22
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v22, 1.0, v23 :: v_dual_add_f32 v23, 1.0, v18
	v_minimummaximum_f32 v12, v250, 0x40e00000, s0
	v_rcp_f32_e32 v20, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_minimummaximum_f32 v13, v251, 0x40e00000, s0
	v_minimummaximum_f32 v24, v254, 0x40e00000, s0
	v_minimummaximum_f32 v25, v255, 0x40e00000, s0
	v_pk_mul_f32 v[6:7], v[28:29], v[6:7]
	v_pk_mul_f32 v[14:15], v[16:17], v[14:15]
	v_minimum_f32 v17, 0x40e00000, v225
	v_pk_add_f32 v[4:5], v[4:5], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[22:23]
	v_minimum_f32 v22, 0x40e00000, v226
	v_minimum_f32 v16, 0x40e00000, v224
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	v_pk_add_f32 v[20:21], v[24:25], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v26, 0x3fd9db23, v17
	v_pk_mul_f32 v[24:25], v[6:7], v[4:5]
	v_mul_f32_e32 v7, 0x3fd9db23, v22
	v_mul_f32_e32 v23, 0x3fd9db23, v16
	v_pk_mul_f32 v[4:5], v[10:11], v[20:21]
	v_mul_f32_e32 v6, 0xbfb8aa3b, v26
	v_pk_mul_f32 v[10:11], v[14:15], v[12:13]
	v_mul_f32_e32 v13, 0xbfb8aa3b, v7
	v_minimummaximum_f32 v18, v252, 0x40e00000, s0
	v_minimummaximum_f32 v19, v253, 0x40e00000, s0
	v_mul_f32_e32 v23, 0xbfb8aa3b, v23
	v_exp_f32_e32 v12, v6
	v_exp_f32_e32 v13, v13
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_pk_add_f32 v[18:19], v[18:19], 1.0 op_sel_hi:[1,0]
	v_exp_f32_e32 v20, v23
	v_nop
	v_minimum_f32 v23, 0x40e00000, v227
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_minimum_f32 v14, 0x40e00000, v230
	v_pk_mul_f32 v[8:9], v[8:9], v[18:19]
	v_dual_add_f32 v11, 1.0, v12 :: v_dual_add_f32 v19, 1.0, v13
	v_minimum_f32 v12, 0x40e00000, v228
	v_minimum_f32 v13, 0x40e00000, v229
	v_mul_f32_e32 v18, 0x3fd9db23, v23
	v_minimum_f32 v15, 0x40e00000, v231
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_add_f32_e32 v9, 1.0, v20
	v_dual_mul_f32 v20, 0x3fd9db23, v12 :: v_dual_mul_f32 v21, 0x3fd9db23, v13
	v_mul_f32_e32 v18, 0xbfb8aa3b, v18
	v_cvt_pk_bf16_f32 v4, v24, v25
	v_dual_mul_f32 v24, 0x3fd9db23, v14 :: v_dual_mul_f32 v25, 0x3fd9db23, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v20, 0xbfb8aa3b, v20 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v21
	v_exp_f32_e32 v26, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v24, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v25
	v_exp_f32_e32 v27, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	v_rcp_f32_e32 v10, v9
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_add_f32_e32 v26, 1.0, v26
	v_rcp_f32_e32 v11, v11
	v_rcp_f32_e32 v20, v19
	v_dual_add_f32 v27, 1.0, v27 :: v_dual_add_f32 v28, 1.0, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v21, v26
	v_dual_add_f32 v29, 1.0, v24 :: v_dual_add_f32 v30, 1.0, v25
	v_rcp_f32_e32 v24, v27
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v25, v28
	v_minimummaximum_f32 v8, v232, 0x40e00000, s0
	v_minimummaximum_f32 v9, v233, 0x40e00000, s0
	v_minimummaximum_f32 v18, v234, 0x40e00000, s0
	v_minimummaximum_f32 v19, v235, 0x40e00000, s0
	v_rcp_f32_e32 v26, v29
	v_rcp_f32_e32 v27, v30
	v_minimummaximum_f32 v28, v236, 0x40e00000, s0
	v_minimummaximum_f32 v29, v237, 0x40e00000, s0
	v_pk_mul_f32 v[10:11], v[16:17], v[10:11]
	v_pk_mul_f32 v[16:17], v[22:23], v[20:21]
	v_minimum_f32 v22, 0x40e00000, v208
	v_minimummaximum_f32 v30, v238, 0x40e00000, s0
	v_minimummaximum_f32 v31, v239, 0x40e00000, s0
	v_pk_add_f32 v[8:9], v[8:9], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[18:19], v[18:19], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v23, 0x40e00000, v209
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_pk_add_f32 v[24:25], v[28:29], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v28, 0x3fd9db23, v22
	v_pk_mul_f32 v[14:15], v[14:15], v[26:27]
	v_pk_add_f32 v[20:21], v[30:31], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[10:11], v[8:9]
	v_pk_mul_f32 v[8:9], v[16:17], v[18:19]
	v_mul_f32_e32 v17, 0x3fd9db23, v23
	v_mul_f32_e32 v18, 0xbfb8aa3b, v28
	v_minimum_f32 v16, 0x40e00000, v210
	v_pk_mul_f32 v[10:11], v[14:15], v[20:21]
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_mul_f32_e32 v14, 0xbfb8aa3b, v17
	v_exp_f32_e32 v17, v18
	v_add_lshl_u32 v32, v2, v37, 1
	v_mul_f32_e32 v15, 0x3fd9db23, v16
	v_cvt_pk_bf16_f32 v11, v10, v11
	v_cvt_pk_bf16_f32 v10, v12, v13
	v_exp_f32_e32 v12, v14
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v26, v27
	v_add_lshl_u32 v14, v0, v37, 1
	v_mul_f32_e32 v13, 0xbfb8aa3b, v15
	v_add_f32_e32 v15, 1.0, v17
	buffer_store_b128 v[4:7], v32, s[24:27], null offen
	v_minimum_f32 v17, 0x40e00000, v211
	buffer_store_b128 v[8:11], v14, s[24:27], null offen
	s_wait_xcnt 0x0
	v_minimum_f32 v8, 0x40e00000, v212
	v_minimum_f32 v9, 0x40e00000, v213
	v_minimum_f32 v10, 0x40e00000, v214
	v_minimum_f32 v11, 0x40e00000, v215
	v_rcp_f32_e32 v6, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v14, 0x3fd9db23, v8 :: v_dual_mul_f32 v15, 0x3fd9db23, v9
	v_dual_add_f32 v7, 1.0, v12 :: v_dual_mul_f32 v12, 0x3fd9db23, v17
	v_dual_mul_f32 v18, 0x3fd9db23, v10 :: v_dual_mul_f32 v19, 0x3fd9db23, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v14, 0xbfb8aa3b, v14 :: v_dual_mul_f32 v15, 0xbfb8aa3b, v15
	v_mul_f32_e32 v12, 0xbfb8aa3b, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v18, 0xbfb8aa3b, v18 :: v_dual_mul_f32 v19, 0xbfb8aa3b, v19
	v_exp_f32_e32 v21, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v15, v15
	v_exp_f32_e32 v20, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v18, v18
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v13, v13
	v_rcp_f32_e32 v7, v7
	v_minimummaximum_f32 v4, v216, 0x40e00000, s0
	v_dual_add_f32 v21, 1.0, v21 :: v_dual_add_f32 v24, 1.0, v15
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v25, 1.0, v18
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v26, 1.0, v19 :: v_dual_add_f32 v13, 1.0, v13
	v_rcp_f32_e32 v18, v21
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v19, v24
	v_minimummaximum_f32 v5, v217, 0x40e00000, s0
	v_rcp_f32_e32 v15, v20
	v_rcp_f32_e32 v20, v25
	v_rcp_f32_e32 v21, v26
	v_rcp_f32_e32 v14, v13
	v_minimummaximum_f32 v24, v220, 0x40e00000, s0
	v_minimummaximum_f32 v25, v221, 0x40e00000, s0
	v_minimummaximum_f32 v12, v218, 0x40e00000, s0
	v_minimummaximum_f32 v13, v219, 0x40e00000, s0
	v_minimummaximum_f32 v26, v222, 0x40e00000, s0
	v_minimummaximum_f32 v27, v223, 0x40e00000, s0
	v_pk_mul_f32 v[6:7], v[22:23], v[6:7]
	v_pk_mul_f32 v[8:9], v[8:9], v[18:19]
	v_minimum_f32 v18, 0x40e00000, v192
	v_pk_add_f32 v[4:5], v[4:5], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v19, 0x40e00000, v193
	v_pk_mul_f32 v[10:11], v[10:11], v[20:21]
	v_pk_add_f32 v[20:21], v[24:25], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[16:17], v[14:15]
	v_pk_add_f32 v[16:17], v[26:27], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v24, 0x3fd9db23, v18
	v_pk_mul_f32 v[22:23], v[6:7], v[4:5]
	v_mul_f32_e32 v6, 0x3fd9db23, v19
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	v_pk_mul_f32 v[4:5], v[10:11], v[16:17]
	v_mul_f32_e32 v16, 0xbfb8aa3b, v24
	v_pk_mul_f32 v[10:11], v[14:15], v[12:13]
	v_mul_f32_e32 v12, 0xbfb8aa3b, v6
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_minimum_f32 v8, 0x40e00000, v194
	v_exp_f32_e32 v13, v16
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_exp_f32_e32 v12, v12
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_mul_f32_e32 v11, 0x3fd9db23, v8
	v_minimum_f32 v9, 0x40e00000, v195
	v_minimum_f32 v17, 0x40e00000, v199
	v_cvt_pk_bf16_f32 v4, v22, v23
	v_add_f32_e32 v13, 1.0, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v15, 0xbfb8aa3b, v11 :: v_dual_add_f32 v14, 1.0, v12
	v_mul_f32_e32 v16, 0x3fd9db23, v9
	v_mul_f32_e32 v25, 0x3fd9db23, v17
	v_rcp_f32_e32 v12, v13
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v20, v15
	v_rcp_f32_e32 v13, v14
	v_mul_f32_e32 v21, 0xbfb8aa3b, v16
	v_minimum_f32 v14, 0x40e00000, v196
	v_minimum_f32 v15, 0x40e00000, v197
	v_minimum_f32 v16, 0x40e00000, v198
	v_minimummaximum_f32 v10, v200, 0x40e00000, s0
	v_exp_f32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v22, 0x3fd9db23, v14 :: v_dual_mul_f32 v23, 0x3fd9db23, v15
	v_mul_f32_e32 v24, 0x3fd9db23, v16
	v_add_f32_e32 v26, 1.0, v20
	v_minimummaximum_f32 v11, v201, 0x40e00000, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v20, 0xbfb8aa3b, v22
	v_dual_mul_f32 v22, 0xbfb8aa3b, v23 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v24
	v_mul_f32_e32 v24, 0xbfb8aa3b, v25
	v_add_f32_e32 v25, 1.0, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v27, v20
	v_exp_f32_e32 v28, v22
	v_exp_f32_e32 v29, v23
	v_exp_f32_e32 v24, v24
	v_rcp_f32_e32 v23, v25
	v_rcp_f32_e32 v22, v26
	v_minimummaximum_f32 v30, v206, 0x40e00000, s0
	v_minimummaximum_f32 v31, v207, 0x40e00000, s0
	v_dual_add_f32 v25, 1.0, v27 :: v_dual_add_f32 v27, 1.0, v28
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v28, 1.0, v29 :: v_dual_add_f32 v29, 1.0, v24
	v_pk_mul_f32 v[12:13], v[18:19], v[12:13]
	v_rcp_f32_e32 v26, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_pk_mul_f32 v[8:9], v[8:9], v[22:23]
	v_pk_add_f32 v[22:23], v[30:31], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v19, 0x40e00000, v177
	v_pk_add_f32 v[10:11], v[10:11], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v18, 0x40e00000, v176
	v_pk_mul_f32 v[14:15], v[14:15], v[26:27]
	v_minimum_f32 v26, 0x40e00000, v178
	v_pk_mul_f32 v[16:17], v[16:17], v[28:29]
	v_mul_f32_e32 v28, 0x3fd9db23, v19
	v_pk_mul_f32 v[12:13], v[12:13], v[10:11]
	v_minimummaximum_f32 v24, v204, 0x40e00000, s0
	v_minimummaximum_f32 v25, v205, 0x40e00000, s0
	v_pk_mul_f32 v[10:11], v[16:17], v[22:23]
	v_mul_f32_e32 v17, 0x3fd9db23, v26
	v_mul_f32_e32 v27, 0x3fd9db23, v18
	v_mul_f32_e32 v16, 0xbfb8aa3b, v28
	v_minimummaximum_f32 v20, v202, 0x40e00000, s0
	v_minimummaximum_f32 v21, v203, 0x40e00000, s0
	v_mul_f32_e32 v17, 0xbfb8aa3b, v17
	v_pk_add_f32 v[24:25], v[24:25], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v27, 0xbfb8aa3b, v27
	v_exp_f32_e32 v16, v16
	v_pk_add_f32 v[20:21], v[20:21], 1.0 op_sel_hi:[1,0]
	v_exp_f32_e32 v17, v17
	v_pk_mul_f32 v[14:15], v[14:15], v[24:25]
	v_exp_f32_e32 v22, v27
	v_cvt_pk_bf16_f32 v11, v10, v11
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	v_minimum_f32 v27, 0x40e00000, v179
	v_cvt_pk_bf16_f32 v10, v14, v15
	s_delay_alu instid0(TRANS32_DEP_2)
	v_dual_add_f32 v15, 1.0, v16 :: v_dual_add_f32 v23, 1.0, v17
	v_minimum_f32 v16, 0x40e00000, v180
	v_minimum_f32 v17, 0x40e00000, v181
	v_minimum_f32 v20, 0x40e00000, v182
	v_minimum_f32 v21, 0x40e00000, v183
	v_add_f32_e32 v14, 1.0, v22
	v_mul_f32_e32 v22, 0x3fd9db23, v27
	v_dual_mul_f32 v24, 0x3fd9db23, v16 :: v_dual_mul_f32 v25, 0x3fd9db23, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v28, 0x3fd9db23, v20 :: v_dual_mul_f32 v29, 0x3fd9db23, v21
	v_mul_f32_e32 v22, 0xbfb8aa3b, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v24, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v25
	v_dual_mul_f32 v28, 0xbfb8aa3b, v28 :: v_dual_mul_f32 v29, 0xbfb8aa3b, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v30, v22
	v_exp_f32_e32 v31, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_rcp_f32_e32 v14, v14
	v_rcp_f32_e32 v15, v15
	v_rcp_f32_e32 v24, v23
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v25 :: v_dual_add_f32 v33, 1.0, v28
	v_add_f32_e32 v34, 1.0, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v25, v30
	v_rcp_f32_e32 v28, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v29, v32
	v_rcp_f32_e32 v30, v33
	v_rcp_f32_e32 v31, v34
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v12, v13
	v_minimummaximum_f32 v12, v184, 0x40e00000, s0
	v_minimummaximum_f32 v13, v185, 0x40e00000, s0
	v_minimummaximum_f32 v32, v188, 0x40e00000, s0
	v_minimummaximum_f32 v33, v189, 0x40e00000, s0
	v_minimummaximum_f32 v34, v190, 0x40e00000, s0
	v_minimummaximum_f32 v35, v191, 0x40e00000, s0
	v_minimummaximum_f32 v22, v186, 0x40e00000, s0
	v_minimummaximum_f32 v23, v187, 0x40e00000, s0
	v_pk_mul_f32 v[14:15], v[18:19], v[14:15]
	v_pk_mul_f32 v[18:19], v[26:27], v[24:25]
	v_pk_mul_f32 v[16:17], v[16:17], v[28:29]
	v_pk_mul_f32 v[20:21], v[20:21], v[30:31]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[24:25], v[34:35], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[26:27], v[32:33], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v28, 0x40e00000, v160
	v_pk_add_f32 v[22:23], v[22:23], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[14:15], v[12:13]
	v_minimum_f32 v29, 0x40e00000, v161
	v_pk_mul_f32 v[12:13], v[20:21], v[24:25]
	v_pk_mul_f32 v[16:17], v[16:17], v[26:27]
	v_mul_f32_e32 v20, 0x3fd9db23, v28
	v_pk_mul_f32 v[18:19], v[18:19], v[22:23]
	v_mul_f32_e32 v21, 0x3fd9db23, v29
	v_add_lshl_u32 v37, v36, v38, 1
	v_cvt_pk_bf16_f32 v14, v16, v17
	v_mul_f32_e32 v16, 0xbfb8aa3b, v20
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v13, v18, v19
	v_mul_f32_e32 v17, 0xbfb8aa3b, v21
	v_add_lshl_u32 v39, v3, v38, 1
	v_exp_f32_e32 v19, v16
	v_nop
	v_minimum_f32 v16, 0x40e00000, v162
	buffer_store_b128 v[4:7], v37, s[24:27], null offen
	v_exp_f32_e32 v20, v17
	v_nop
	v_minimum_f32 v17, 0x40e00000, v163
	v_cvt_pk_bf16_f32 v12, v30, v31
	s_wait_xcnt 0x0
	v_mul_f32_e32 v5, 0x3fd9db23, v16
	v_add_lshl_u32 v18, v2, v38, 1
	buffer_store_b128 v[8:11], v39, s[24:27], null offen
	s_wait_xcnt 0x0
	v_mul_f32_e32 v9, 0x3fd9db23, v17
	v_minimum_f32 v10, 0x40e00000, v166
	v_mul_f32_e32 v8, 0xbfb8aa3b, v5
	buffer_store_b128 v[12:15], v18, s[24:27], null offen
	v_minimum_f32 v11, 0x40e00000, v167
	s_wait_xcnt 0x0
	v_mul_f32_e32 v13, 0xbfb8aa3b, v9
	v_minimum_f32 v9, 0x40e00000, v165
	v_exp_f32_e32 v12, v8
	v_nop
	v_minimum_f32 v8, 0x40e00000, v164
	v_add_f32_e32 v6, 1.0, v19
	v_dual_mul_f32 v18, 0x3fd9db23, v10 :: v_dual_mul_f32 v15, 0x3fd9db23, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v19, 0x3fd9db23, v11 :: v_dual_mul_f32 v14, 0x3fd9db23, v8
	v_add_f32_e32 v7, 1.0, v20
	v_exp_f32_e32 v13, v13
	v_add_f32_e32 v20, 1.0, v12
	v_rcp_f32_e32 v6, v6
	v_mul_f32_e32 v12, 0xbfb8aa3b, v14
	v_dual_mul_f32 v14, 0xbfb8aa3b, v15 :: v_dual_mul_f32 v15, 0xbfb8aa3b, v18
	v_mul_f32_e32 v18, 0xbfb8aa3b, v19
	v_rcp_f32_e32 v7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v21, v12
	v_exp_f32_e32 v22, v14
	v_exp_f32_e32 v23, v15
	v_exp_f32_e32 v18, v18
	v_add_f32_e32 v19, 1.0, v13
	v_rcp_f32_e32 v14, v20
	v_minimummaximum_f32 v4, v168, 0x40e00000, s0
	v_minimummaximum_f32 v5, v169, 0x40e00000, s0
	v_minimummaximum_f32 v12, v170, 0x40e00000, s0
	v_rcp_f32_e32 v15, v19
	v_dual_add_f32 v19, 1.0, v21 :: v_dual_add_f32 v21, 1.0, v22
	v_dual_add_f32 v22, 1.0, v23 :: v_dual_add_f32 v23, 1.0, v18
	v_minimummaximum_f32 v13, v171, 0x40e00000, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v20, v19
	v_rcp_f32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_minimummaximum_f32 v24, v174, 0x40e00000, s0
	v_minimummaximum_f32 v25, v175, 0x40e00000, s0
	v_pk_mul_f32 v[6:7], v[28:29], v[6:7]
	v_pk_mul_f32 v[14:15], v[16:17], v[14:15]
	v_minimum_f32 v17, 0x40e00000, v145
	v_pk_add_f32 v[4:5], v[4:5], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[20:21]
	v_pk_mul_f32 v[10:11], v[10:11], v[22:23]
	v_minimum_f32 v22, 0x40e00000, v146
	v_minimum_f32 v16, 0x40e00000, v144
	v_pk_add_f32 v[20:21], v[24:25], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[12:13], v[12:13], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v26, 0x3fd9db23, v17
	v_pk_mul_f32 v[24:25], v[6:7], v[4:5]
	v_mul_f32_e32 v7, 0x3fd9db23, v22
	v_mul_f32_e32 v23, 0x3fd9db23, v16
	v_pk_mul_f32 v[4:5], v[10:11], v[20:21]
	v_mul_f32_e32 v6, 0xbfb8aa3b, v26
	v_pk_mul_f32 v[10:11], v[14:15], v[12:13]
	v_mul_f32_e32 v13, 0xbfb8aa3b, v7
	v_minimummaximum_f32 v18, v172, 0x40e00000, s0
	v_minimummaximum_f32 v19, v173, 0x40e00000, s0
	v_mul_f32_e32 v23, 0xbfb8aa3b, v23
	v_exp_f32_e32 v12, v6
	v_exp_f32_e32 v13, v13
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_pk_add_f32 v[18:19], v[18:19], 1.0 op_sel_hi:[1,0]
	v_exp_f32_e32 v20, v23
	v_nop
	v_minimum_f32 v23, 0x40e00000, v147
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_minimum_f32 v14, 0x40e00000, v150
	v_pk_mul_f32 v[8:9], v[8:9], v[18:19]
	v_dual_add_f32 v11, 1.0, v12 :: v_dual_add_f32 v19, 1.0, v13
	v_minimum_f32 v12, 0x40e00000, v148
	v_minimum_f32 v13, 0x40e00000, v149
	v_minimum_f32 v15, 0x40e00000, v151
	v_mul_f32_e32 v18, 0x3fd9db23, v23
	v_cvt_pk_bf16_f32 v6, v8, v9
	v_add_f32_e32 v9, 1.0, v20
	v_dual_mul_f32 v20, 0x3fd9db23, v12 :: v_dual_mul_f32 v21, 0x3fd9db23, v13
	v_cvt_pk_bf16_f32 v4, v24, v25
	v_dual_mul_f32 v24, 0x3fd9db23, v14 :: v_dual_mul_f32 v25, 0x3fd9db23, v15
	v_mul_f32_e32 v18, 0xbfb8aa3b, v18
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, 0xbfb8aa3b, v20 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v21
	v_dual_mul_f32 v24, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v26, v18
	v_exp_f32_e32 v27, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_rcp_f32_e32 v10, v9
	v_rcp_f32_e32 v11, v11
	v_rcp_f32_e32 v20, v19
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v21 :: v_dual_add_f32 v29, 1.0, v24
	v_add_f32_e32 v30, 1.0, v25
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v21, v26
	v_minimummaximum_f32 v8, v152, 0x40e00000, s0
	v_minimummaximum_f32 v9, v153, 0x40e00000, s0
	v_rcp_f32_e32 v24, v27
	v_rcp_f32_e32 v25, v28
	v_rcp_f32_e32 v26, v29
	v_rcp_f32_e32 v27, v30
	v_minimummaximum_f32 v28, v156, 0x40e00000, s0
	v_minimummaximum_f32 v29, v157, 0x40e00000, s0
	v_minimummaximum_f32 v18, v154, 0x40e00000, s0
	v_minimummaximum_f32 v19, v155, 0x40e00000, s0
	v_minimummaximum_f32 v30, v158, 0x40e00000, s0
	v_minimummaximum_f32 v31, v159, 0x40e00000, s0
	v_pk_mul_f32 v[10:11], v[16:17], v[10:11]
	v_pk_mul_f32 v[16:17], v[22:23], v[20:21]
	v_minimum_f32 v22, 0x40e00000, v128
	v_pk_add_f32 v[8:9], v[8:9], 1.0 op_sel_hi:[1,0]
	v_minimum_f32 v23, 0x40e00000, v129
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_pk_add_f32 v[24:25], v[28:29], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[26:27]
	v_pk_add_f32 v[20:21], v[30:31], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[18:19], v[18:19], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v28, 0x3fd9db23, v22
	v_pk_mul_f32 v[26:27], v[10:11], v[8:9]
	v_mul_f32_e32 v10, 0x3fd9db23, v23
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_pk_mul_f32 v[8:9], v[14:15], v[20:21]
	v_mul_f32_e32 v20, 0xbfb8aa3b, v28
	v_pk_mul_f32 v[14:15], v[16:17], v[18:19]
	v_mul_f32_e32 v16, 0xbfb8aa3b, v10
	v_cvt_pk_bf16_f32 v10, v12, v13
	v_minimum_f32 v12, 0x40e00000, v130
	v_exp_f32_e32 v17, v20
	v_cvt_pk_bf16_f32 v11, v8, v9
	v_exp_f32_e32 v16, v16
	v_cvt_pk_bf16_f32 v9, v14, v15
	v_mul_f32_e32 v15, 0x3fd9db23, v12
	v_minimum_f32 v13, 0x40e00000, v131
	v_minimum_f32 v21, 0x40e00000, v135
	v_cvt_pk_bf16_f32 v8, v26, v27
	v_add_f32_e32 v17, 1.0, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v19, 0xbfb8aa3b, v15 :: v_dual_add_f32 v18, 1.0, v16
	v_mul_f32_e32 v20, 0x3fd9db23, v13
	v_mul_f32_e32 v29, 0x3fd9db23, v21
	v_rcp_f32_e32 v16, v17
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v24, v19
	v_rcp_f32_e32 v17, v18
	v_mul_f32_e32 v25, 0xbfb8aa3b, v20
	v_minimum_f32 v18, 0x40e00000, v132
	v_minimum_f32 v19, 0x40e00000, v133
	v_minimum_f32 v20, 0x40e00000, v134
	v_minimummaximum_f32 v14, v136, 0x40e00000, s0
	v_exp_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v26, 0x3fd9db23, v18 :: v_dual_mul_f32 v27, 0x3fd9db23, v19
	v_mul_f32_e32 v28, 0x3fd9db23, v20
	v_add_f32_e32 v30, 1.0, v24
	v_minimummaximum_f32 v15, v137, 0x40e00000, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v26
	v_dual_mul_f32 v26, 0xbfb8aa3b, v27 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v28
	v_mul_f32_e32 v28, 0xbfb8aa3b, v29
	v_add_f32_e32 v29, 1.0, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v31, v24
	v_exp_f32_e32 v32, v26
	v_exp_f32_e32 v33, v27
	v_exp_f32_e32 v28, v28
	v_rcp_f32_e32 v27, v29
	v_rcp_f32_e32 v26, v30
	v_minimummaximum_f32 v34, v142, 0x40e00000, s0
	v_minimummaximum_f32 v35, v143, 0x40e00000, s0
	v_dual_add_f32 v29, 1.0, v31 :: v_dual_add_f32 v31, 1.0, v32
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v32, 1.0, v33 :: v_dual_add_f32 v33, 1.0, v28
	v_minimummaximum_f32 v28, v140, 0x40e00000, s0
	v_rcp_f32_e32 v30, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_minimummaximum_f32 v29, v141, 0x40e00000, s0
	v_pk_mul_f32 v[16:17], v[22:23], v[16:17]
	v_pk_mul_f32 v[12:13], v[12:13], v[26:27]
	v_pk_add_f32 v[26:27], v[34:35], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[14:15], v[14:15], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[30:31]
	v_pk_add_f32 v[28:29], v[28:29], 1.0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[32:33]
	v_minimum_f32 v22, 0x40e00000, v112
	v_pk_mul_f32 v[16:17], v[16:17], v[14:15]
	v_minimummaximum_f32 v24, v138, 0x40e00000, s0
	v_pk_mul_f32 v[18:19], v[18:19], v[28:29]
	v_pk_mul_f32 v[14:15], v[20:21], v[26:27]
	v_mul_f32_e32 v30, 0x3fd9db23, v22
	v_minimummaximum_f32 v25, v139, 0x40e00000, s0
	v_minimum_f32 v23, 0x40e00000, v113
	v_minimum_f32 v26, 0x40e00000, v116
	v_cvt_pk_bf16_f32 v15, v14, v15
	v_cvt_pk_bf16_f32 v14, v18, v19
	v_minimum_f32 v19, 0x40e00000, v115
	v_mul_f32_e32 v30, 0xbfb8aa3b, v30
	v_pk_add_f32 v[24:25], v[24:25], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v31, 0x3fd9db23, v23
	v_minimum_f32 v18, 0x40e00000, v114
	v_mul_f32_e32 v27, 0x3fd9db23, v19
	v_exp_f32_e32 v21, v30
	v_pk_mul_f32 v[12:13], v[12:13], v[24:25]
	v_mul_f32_e32 v20, 0xbfb8aa3b, v31
	v_minimum_f32 v28, 0x40e00000, v118
	v_mul_f32_e32 v30, 0xbfb8aa3b, v27
	v_minimum_f32 v27, 0x40e00000, v117
	v_minimum_f32 v29, 0x40e00000, v119
	v_mul_f32_e32 v31, 0x3fd9db23, v26
	v_exp_f32_e32 v20, v20
	v_cvt_pk_bf16_f32 v13, v12, v13
	v_mul_f32_e32 v32, 0x3fd9db23, v27
	v_cvt_pk_bf16_f32 v12, v16, v17
	v_add_f32_e32 v17, 1.0, v21
	v_mul_f32_e32 v21, 0x3fd9db23, v18
	v_dual_mul_f32 v33, 0x3fd9db23, v28 :: v_dual_mul_f32 v34, 0x3fd9db23, v29
	v_dual_mul_f32 v31, 0xbfb8aa3b, v31 :: v_dual_mul_f32 v32, 0xbfb8aa3b, v32
	v_add_lshl_u32 v47, v36, v1, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v25, 0xbfb8aa3b, v21
	v_dual_mul_f32 v33, 0xbfb8aa3b, v33 :: v_dual_mul_f32 v34, 0xbfb8aa3b, v34
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v35, v31
	v_exp_f32_e32 v36, v32
	v_add_f32_e32 v24, 1.0, v20
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_rcp_f32_e32 v20, v17
	v_rcp_f32_e32 v21, v24
	v_dual_add_f32 v35, 1.0, v35 :: v_dual_add_f32 v36, 1.0, v36
	v_add_lshl_u32 v46, v0, v38, 1
	v_dual_add_f32 v25, 1.0, v25 :: v_dual_add_f32 v31, 1.0, v30
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v37, 1.0, v33 :: v_dual_add_f32 v38, 1.0, v34
	v_rcp_f32_e32 v34, v35
	v_rcp_f32_e32 v35, v36
	s_delay_alu instid0(TRANS32_DEP_3)
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	s_set_vgpr_msb 4
	v_minimum_f32 v22, 0x40e00000, v96 /*v352*/
	v_rcp_f32_e32 v30, v25
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v36, v37
	v_rcp_f32_e32 v37, v38
	v_minimummaximum_f32 v16, v120, 0x40e00000, s0
	v_minimummaximum_f32 v17, v121, 0x40e00000, s0
	v_minimummaximum_f32 v38, v126, 0x40e00000, s0
	v_minimummaximum_f32 v39, v127, 0x40e00000, s0
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[26:27], v[26:27], v[34:35]
	s_set_vgpr_msb 4
	v_minimum_f32 v23, 0x40e00000, v97 /*v353*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v35, 0x3fd9db23, v22
	s_set_vgpr_msb 4
	v_minimum_f32 v34, 0x40e00000, v98 /*v354*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[18:19], v[18:19], v[30:31]
	v_pk_mul_f32 v[28:29], v[28:29], v[36:37]
	v_pk_add_f32 v[30:31], v[38:39], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v36, 0x3fd9db23, v23
	v_mul_f32_e32 v35, 0xbfb8aa3b, v35
	v_pk_add_f32 v[16:17], v[16:17], 1.0 op_sel_hi:[1,0]
	v_mul_f32_e32 v37, 0x3fd9db23, v34
	v_minimummaximum_f32 v24, v122, 0x40e00000, s0
	v_minimummaximum_f32 v25, v123, 0x40e00000, s0
	v_minimummaximum_f32 v32, v124, 0x40e00000, s0
	v_minimummaximum_f32 v33, v125, 0x40e00000, s0
	v_mul_f32_e32 v36, 0xbfb8aa3b, v36
	v_exp_f32_e32 v35, v35
	v_pk_mul_f32 v[20:21], v[20:21], v[16:17]
	v_pk_mul_f32 v[16:17], v[28:29], v[30:31]
	v_mul_f32_e32 v29, 0xbfb8aa3b, v37
	v_pk_add_f32 v[32:33], v[32:33], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[24:25], v[24:25], 1.0 op_sel_hi:[1,0]
	v_exp_f32_e32 v28, v36
	s_set_vgpr_msb 4
	v_minimum_f32 v30, 0x40e00000, v102 /*v358*/
	v_exp_f32_e32 v29, v29
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[26:27], v[26:27], v[32:33]
	v_pk_mul_f32 v[24:25], v[18:19], v[24:25]
	v_cvt_pk_bf16_f32 v19, v16, v17
	v_add_f32_e32 v16, 1.0, v35
	s_set_vgpr_msb 4
	v_minimum_f32 v35, 0x40e00000, v99 /*v355*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v18, v26, v27
	v_add_f32_e32 v27, 1.0, v28
	s_set_vgpr_msb 4
	v_minimum_f32 v28, 0x40e00000, v100 /*v356*/
	v_rcp_f32_e32 v26, v16
	v_nop
	s_set_vgpr_msb 0x400
	v_add_f32_e32 v16, 1.0, v29
	s_set_vgpr_msb 4
	v_minimum_f32 v29, 0x40e00000, v101 /*v357*/
	v_minimum_f32 v31, 0x40e00000, v103 /*v359*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v32, 0x3fd9db23, v35 :: v_dual_mul_f32 v33, 0x3fd9db23, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v37, 0x3fd9db23, v30 :: v_dual_mul_f32 v36, 0x3fd9db23, v29
	v_mul_f32_e32 v38, 0x3fd9db23, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v32, 0xbfb8aa3b, v32 :: v_dual_mul_f32 v33, 0xbfb8aa3b, v33
	v_dual_mul_f32 v37, 0xbfb8aa3b, v37 :: v_dual_mul_f32 v36, 0xbfb8aa3b, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v38, 0xbfb8aa3b, v38
	v_exp_f32_e32 v39, v32
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v40, v33
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v41, v36
	v_exp_f32_e32 v38, v38
	v_rcp_f32_e32 v36, v16
	v_rcp_f32_e32 v27, v27
	v_cvt_pk_bf16_f32 v17, v24, v25
	v_dual_add_f32 v16, 1.0, v39 :: v_dual_add_f32 v39, 1.0, v40
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v24, v104 /*v360*/, 0x40e00000, s0
	v_add_f32_e32 v40, 1.0, v41
	v_dual_add_f32 v41, 1.0, v37 :: v_dual_add_f32 v42, 1.0, v38
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v37, v16
	v_rcp_f32_e32 v38, v39
	v_rcp_f32_e32 v39, v40
	v_rcp_f32_e32 v40, v41
	v_rcp_f32_e32 v41, v42
	s_set_vgpr_msb 1
	v_minimummaximum_f32 v25, v105 /*v361*/, 0x40e00000, s0
	v_minimummaximum_f32 v32, v106 /*v362*/, 0x40e00000, s0
	v_minimummaximum_f32 v33, v107 /*v363*/, 0x40e00000, s0
	v_minimummaximum_f32 v42, v108 /*v364*/, 0x40e00000, s0
	v_minimummaximum_f32 v43, v109 /*v365*/, 0x40e00000, s0
	v_minimummaximum_f32 v44, v110 /*v366*/, 0x40e00000, s0
	v_minimummaximum_f32 v45, v111 /*v367*/, 0x40e00000, s0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[22:23], v[26:27]
	v_pk_mul_f32 v[26:27], v[34:35], v[36:37]
	v_pk_mul_f32 v[28:29], v[28:29], v[38:39]
	v_pk_mul_f32 v[30:31], v[30:31], v[40:41]
	v_pk_add_f32 v[34:35], v[44:45], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[36:37], v[42:43], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[32:33], v[32:33], 1.0 op_sel_hi:[1,0]
	v_pk_add_f32 v[24:25], v[24:25], 1.0 op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v16, v20, v21
	v_pk_mul_f32 v[20:21], v[30:31], v[34:35]
	v_pk_mul_f32 v[28:29], v[28:29], v[36:37]
	v_pk_mul_f32 v[26:27], v[26:27], v[32:33]
	v_pk_mul_f32 v[24:25], v[22:23], v[24:25]
	v_add_lshl_u32 v3, v3, v1, 1
	v_add_lshl_u32 v2, v2, v1, 1
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_cvt_pk_bf16_f32 v22, v28, v29
	v_cvt_pk_bf16_f32 v21, v26, v27
	v_cvt_pk_bf16_f32 v20, v24, v25
	v_add_lshl_u32 v0, v0, v1, 1
	s_clause 0x4
	buffer_store_b128 v[4:7], v46, s[24:27], null offen
	buffer_store_b128 v[8:11], v47, s[24:27], null offen
	buffer_store_b128 v[12:15], v3, s[24:27], null offen
	buffer_store_b128 v[16:19], v2, s[24:27], null offen
	buffer_store_b128 v[20:23], v0, s[24:27], null offen
.LBB0_4:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernel_mxscale_gemm1
		.amdhsa_group_segment_fixed_size 176128
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
		.amdhsa_next_free_vgpr 513
		.amdhsa_next_free_sgpr 103
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 176
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
	.size	kernel_mxscale_gemm1, .Lfunc_end0-kernel_mxscale_gemm1

	.set kernel_mxscale_gemm1.num_vgpr, 470
	.set kernel_mxscale_gemm1.num_agpr, 0
	.set kernel_mxscale_gemm1.numbered_sgpr, 103
	.set kernel_mxscale_gemm1.num_named_barrier, 0
	.set kernel_mxscale_gemm1.private_seg_size, 0
	.set kernel_mxscale_gemm1.uses_vcc, 1
	.set kernel_mxscale_gemm1.uses_flat_scratch, 0
	.set kernel_mxscale_gemm1.has_dyn_sized_stack, 0
	.set kernel_mxscale_gemm1.has_recursion, 0
	.set kernel_mxscale_gemm1.has_indirect_call, 0
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
    .group_segment_fixed_size: 176128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .max_flat_workgroup_size: 128
    .name:           kernel_mxscale_gemm1
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     105
    .sgpr_spill_count: 0
    .symbol:         kernel_mxscale_gemm1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     470
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
