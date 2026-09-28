	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	sage_hip_attn_bm128_bn32 ; -- Begin function sage_hip_attn_bm128_bn32
	.globl	sage_hip_attn_bm128_bn32
	.p2align	8
	.type	sage_hip_attn_bm128_bn32,@function
sage_hip_attn_bm128_bn32:               ; @sage_hip_attn_bm128_bn32
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[24:26], s[0:1], 0x3c
	s_wait_kmcnt 0x0
	s_add_co_i32 s2, s25, 0x1ff
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s2, s2, 9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s2, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s3
	s_sub_co_i32 s7, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s6, s6
	s_delay_alu instid0(TRANS32_DEP_1)
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s6, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s7, s6, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s6, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s4, ttmp9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s5, ttmp9, s7
	s_add_co_i32 s7, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s8, s5, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s4, s7, s4
	s_cselect_b32 s5, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s4, 1
	s_sub_co_i32 s8, s5, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s4, s7, s4
	s_cselect_b32 s5, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s6, s2
	s_sub_co_i32 s7, 0, s2
	s_wait_alu depctr_sa_sdst(0)
	v_s_rcp_f32 s6, s6
	s_delay_alu instid0(TRANS32_DEP_1)
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s6, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s7, s6, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s6, s7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s33, s5, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s7, s33, s2
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s9, s5, s7
	s_add_co_i32 s7, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s8, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s9, s2
	s_cselect_b32 s33, s7, s33
	s_cselect_b32 s9, s8, s9
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s33, 1
	s_sub_co_i32 s8, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_u32 s9, s2
	s_cselect_b32 s33, s7, s33
	s_cselect_b32 s9, s8, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s4, s4, s3
	s_mul_i32 s9, s9, s26
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s32, s4, s9
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s32, s32, s33
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s4, s32
	v_lshrrev_b32_e32 v197, 1, v0
	v_and_b32_e32 v195, 15, v0
	v_mov_b32_e32 v163, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v169, 0 :: v_dual_and_b32 v194, 8, v197
	v_dual_mov_b32 v170, v163 :: v_dual_and_b32 v1, 0x1e0, v0
	s_wait_kmcnt 0x0
	s_abs_i32 s2, s26
	s_add_co_i32 s5, s25, 0x1ff
	s_cvt_f32_u32 s3, s2
	s_sub_co_i32 s6, 0, s2
	s_ashr_i32 s8, s5, 31
	s_xor_b32 s7, s32, s26
	v_s_rcp_f32 s3, s3
	s_ashr_i32 s7, s7, 31
	s_mov_b32 s12, s24
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s3, s3
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s6, s6, s3
	s_mul_hi_u32 s6, s3, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s3, s6
	s_lshr_b32 s6, s8, 23
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s3, s4, s3
	s_add_co_i32 s5, s5, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s3, s2
	s_ashr_i32 s5, s5, 9
	s_sub_co_i32 s4, s4, s6
	s_add_co_i32 s6, s3, 1
	s_sub_co_i32 s8, s4, s2
	s_cmp_ge_u32 s4, s2
	s_cselect_b32 s3, s6, s3
	s_cselect_b32 s4, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s3, 1
	s_cmp_ge_u32 s4, s2
	s_cselect_b32 s2, s6, s3
	s_abs_i32 s3, s5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, s2, s7
	s_cvt_f32_u32 s4, s3
	s_sub_co_i32 s6, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s2, s2, s7
	v_s_rcp_f32 s4, s4
	s_wait_alu depctr_sa_sdst(0)
	s_abs_i32 s7, s2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_u32_f32 s4, s4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s6, s6, s4
	s_mul_hi_u32 s6, s4, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s4, s4, s6
	s_xor_b32 s6, s2, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s4, s7, s4
	s_ashr_i32 s6, s6, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s8, s4, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s7, s7, s8
	s_add_co_i32 s8, s4, 1
	s_sub_co_i32 s9, s7, s3
	s_cmp_ge_u32 s7, s3
	s_cselect_b32 s4, s8, s4
	s_cselect_b32 s7, s9, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s4, 1
	s_cmp_ge_u32 s7, s3
	s_cselect_b32 s3, s8, s4
	s_ashr_i32 s13, s24, 31
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, s3, s6
	s_ashr_i32 s27, s26, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s30, s3, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s3, s30, s5
	s_ashr_i32 s31, s30, 31
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s3, s2, s3
	s_mul_u64 s[14:15], s[30:31], s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s3, s3, 9
	s_mul_i32 s2, s2, s26
	s_wait_alu depctr_sa_sdst(0)
	v_or3_b32 v1, s3, v1, v195
	s_sub_co_i32 s28, s32, s2
	s_load_b256 s[4:11], s[0:1], 0x0
	s_ashr_i32 s29, s28, 31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_add_co_u32 v3, vcc_lo, s14, v1
	v_add_co_ci_u32_e64 v4, null, s15, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v5, v3, s27
	v_mad_co_u64_u32 v[155:156], null, v3, s26, s[28:29]
	v_cmp_gt_i32_e32 vcc_lo, s25, v1
	v_mul_lo_u32 v3, v4, s26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v156, v3, v156, v5
	v_lshlrev_b64_e32 v[3:4], 7, v[155:156]
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, s2, s4, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s5, v4, s2
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[169:170], v[5:6], off
.LBB0_2:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v172, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.3:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[171:172], v[5:6], off offset:16
.LBB0_4:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v164, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_6
; %bb.5:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[163:164], v[5:6], off offset:32
.LBB0_6:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v180, 0
	v_mov_b32_e32 v179, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[179:180], v[5:6], off offset:48
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v174, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[173:174], v[5:6], off offset:64
.LBB0_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v183, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[183:184], v[5:6], off offset:80
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_mov_b32_e32 v182, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_add_co_u32 v5, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s2
	global_load_b64 v[181:182], v[5:6], off offset:96
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v188, 0
	v_mov_b32_e32 v187, 0
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_16
; %bb.15:
	v_add_co_u32 v3, s2, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, s2
	global_load_b64 v[187:188], v[3:4], off offset:112
.LBB0_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v5, 16, v1
	v_mov_b32_e32 v158, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_ashrrev_i32_e32 v3, 31, v5
	v_add_co_u32 v4, s2, s14, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s15, v3, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v6, v4, s27
	v_mad_co_u64_u32 v[153:154], null, v4, s26, s[28:29]
	v_cmp_gt_i32_e64 s2, s25, v5
	v_mul_lo_u32 v3, v3, s26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add3_u32 v154, v3, v154, v6
	v_lshlrev_b64_e32 v[3:4], 7, v[153:154]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, s3, s4, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s5, v4, s3
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_add_co_u32 v5, s3, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s3
	global_load_b64 v[157:158], v[5:6], off
.LBB0_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v162, 0
	v_mov_b32_e32 v161, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_add_co_u32 v5, s3, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s3
	global_load_b64 v[161:162], v[5:6], off offset:16
.LBB0_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_mov_b32_e32 v160, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_add_co_u32 v5, s3, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s3
	global_load_b64 v[159:160], v[5:6], off offset:32
.LBB0_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v168, 0
	v_mov_b32_e32 v167, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_add_co_u32 v5, s3, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s3
	global_load_b64 v[167:168], v[5:6], off offset:48
.LBB0_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_mov_b32_e32 v166, 0
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_add_co_u32 v5, s3, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s3
	global_load_b64 v[165:166], v[5:6], off offset:64
.LBB0_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b256 s[16:23], s[0:1], 0x20
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v177, 0
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_add_co_u32 v5, s0, v3, v194
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s0
	global_load_b64 v[177:178], v[5:6], off offset:80
.LBB0_28:
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v176, 0
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_add_co_u32 v5, s0, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, 0, v4, s0
	global_load_b64 v[175:176], v[5:6], off offset:96
.LBB0_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v185, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v186, v185
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_add_co_u32 v3, s0, v3, v194
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, s0
	global_load_b64 v[185:186], v[3:4], off offset:112
.LBB0_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_lshrrev_b32_e32 v2, 27, v2
	s_ashr_i32 s0, s12, 31
	s_mul_i32 s1, s30, s26
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s0, s0, 27
	s_add_co_i32 s3, s1, s28
	v_add_nc_u32_e32 v1, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s12, s0
	s_mul_u64 s[4:5], s[30:31], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s0, s0, 5
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[22:23], s[4:5], s[28:29]
	v_ashrrev_i32_e32 v1, 5, v1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s3, s0
                                        ; implicit-def: $vgpr145_vgpr146_vgpr147_vgpr148
                                        ; implicit-def: $vgpr149_vgpr150_vgpr151_vgpr152
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v1, s3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s10, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s11, v2, s0
	v_cmp_gt_u32_e64 s0, 0x100, v0
	global_load_b32 v198, v[1:2], off
	s_and_saveexec_b32 s4, s0
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_lshrrev_b32_e32 v13, 3, v0
	s_lshl_b64 s[10:11], s[22:23], 7
	s_cmp_gt_i32 s12, 32
	v_lshlrev_b32_e32 v6, 4, v0
	s_cselect_b32 s5, 32, 0
	v_add_co_u32 v1, s1, s14, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s15, 0, s1
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v3, s5, v13
	v_mul_lo_u32 v4, v1, s27
	v_or_b32_e32 v7, s10, v197
	v_mul_lo_u32 v5, v2, s26
	v_mad_co_u64_u32 v[1:2], null, v1, s26, s[28:29]
	v_add_co_u32 v3, s1, s14, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s15, 0, s1
	v_and_b32_e32 v14, 0x70, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v9, v3, s27
	v_and_b32_e32 v15, 16, v6
	v_add3_u32 v2, v5, v2, v4
	v_mul_lo_u32 v8, v8, s26
	v_mad_co_u64_u32 v[3:4], null, v3, s26, s[28:29]
	v_mul_lo_u32 v10, v7, s13
	v_mad_co_u64_u32 v[5:6], null, v7, s12, s[8:9]
	v_lshlrev_b64_e32 v[1:2], 7, v[1:2]
	s_mul_i32 s10, s11, s12
	v_or_b32_e32 v11, s5, v15
	v_add3_u32 v4, v8, v4, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v1, s1, s6, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s7, v2, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v6, s10, v6, v10
	v_add_co_u32 v7, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, 0, v2, s1
	v_lshlrev_b64_e32 v[1:2], 7, v[3:4]
	v_add_co_u32 v3, s1, v5, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, v6, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s6, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s7, v2, s1
	v_add_co_u32 v9, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, 0, v2, s1
	v_add_co_u32 v11, s1, v5, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, 0, v6, s1
	global_load_b128 v[1:4], v[3:4], off
	s_clause 0x1
	global_load_b128 v[5:8], v[7:8], off
	global_load_b128 v[145:148], v[9:10], off
	global_load_b128 v[149:152], v[11:12], off
	v_mul_u32_u24_e32 v9, 40, v197
	v_mad_u32_u24 v10, 0x88, v13, v14
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v9, v9, v15, 0x3300
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v10, v[5:6], v[7:8] offset1:1
	ds_store_2addr_b64 v9, v[1:2], v[3:4] offset1:1
.LBB0_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_ashr_i32 s1, s25, 31
	v_mbcnt_lo_u32_b32 v199, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s1, s1, 27
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s5, s25, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s5, 0xffffffe0
	s_cmp_lt_i32 s25, 32
	s_cbranch_scc1 .LBB0_43
; %bb.35:
	v_dual_mov_b32 v204, 0 :: v_dual_lshlrev_b32 v1, 4, v0
	s_lshl_b64 s[10:11], s[22:23], 7
	v_lshrrev_b32_e32 v201, 3, v0
	v_mad_u32_u24 v192, v195, 40, v194
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v211, 0 :: v_dual_and_b32 v2, 0x70, v1
	v_dual_mov_b32 v213, 0xff800000 :: v_dual_and_b32 v202, 16, v1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s1, s10, v197
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s11, 0, s1
	v_add_co_u32 v205, s1, s6, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v4, v1, s13
	v_mad_co_u64_u32 v[190:191], null, v1, s12, s[8:9]
	v_xor_b32_e32 v1, 16, v199
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v206, null, s7, 0, s1
	v_mul_lo_u32 v3, v3, s12
	v_mad_u32_u24 v189, 0x88, v201, v2
	v_cmp_gt_u32_e64 s1, 32, v1
	v_mad_u32_u24 v203, v197, 40, v202
	v_mad_u32_u24 v193, 0x88, v195, v194
	v_mov_b32_e32 v208, 0
	v_dual_mov_b32 v214, 0xff800000 :: v_dual_mov_b32 v121, 0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v1, v199, v1, s1
	v_add3_u32 v191, v3, v191, v4
	v_dual_mov_b32 v212, 0 :: v_dual_mov_b32 v123, v204
	v_dual_mov_b32 v122, v204 :: v_dual_mov_b32 v125, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v124, v204 :: v_dual_lshlrev_b32 v207, 2, v1
	v_dual_mov_b32 v127, v204 :: v_dual_mov_b32 v126, v204
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v128, v204
	v_dual_mov_b32 v115, v204 :: v_dual_mov_b32 v114, v204
	v_dual_mov_b32 v117, v204 :: v_dual_mov_b32 v116, v204
	v_dual_mov_b32 v119, v204 :: v_dual_mov_b32 v118, v204
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v120, v204
	v_dual_mov_b32 v107, v204 :: v_dual_mov_b32 v106, v204
	v_dual_mov_b32 v109, v204 :: v_dual_mov_b32 v108, v204
	v_dual_mov_b32 v111, v204 :: v_dual_mov_b32 v110, v204
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v112, v204
	v_dual_mov_b32 v99, v204 :: v_dual_mov_b32 v98, v204
	v_dual_mov_b32 v101, v204 :: v_dual_mov_b32 v100, v204
	v_dual_mov_b32 v103, v204 :: v_dual_mov_b32 v102, v204
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v104, v204
	v_dual_mov_b32 v91, v204 :: v_dual_mov_b32 v90, v204
	v_dual_mov_b32 v93, v204 :: v_dual_mov_b32 v92, v204
	v_dual_mov_b32 v95, v204 :: v_dual_mov_b32 v94, v204
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v96, v204
	v_dual_mov_b32 v83, v204 :: v_dual_mov_b32 v82, v204
	v_dual_mov_b32 v85, v204 :: v_dual_mov_b32 v84, v204
	v_dual_mov_b32 v87, v204 :: v_dual_mov_b32 v86, v204
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v88, v204
	v_dual_mov_b32 v75, v204 :: v_dual_mov_b32 v74, v204
	v_dual_mov_b32 v77, v204 :: v_dual_mov_b32 v76, v204
	v_dual_mov_b32 v79, v204 :: v_dual_mov_b32 v78, v204
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v80, v204
	v_dual_mov_b32 v67, v204 :: v_dual_mov_b32 v66, v204
	v_dual_mov_b32 v69, v204 :: v_dual_mov_b32 v68, v204
	v_dual_mov_b32 v71, v204 :: v_dual_mov_b32 v70, v204
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v72, v204
	v_dual_mov_b32 v59, v204 :: v_dual_mov_b32 v58, v204
	v_dual_mov_b32 v61, v204 :: v_dual_mov_b32 v60, v204
	v_dual_mov_b32 v63, v204 :: v_dual_mov_b32 v62, v204
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v64, v204
	v_dual_mov_b32 v51, v204 :: v_dual_mov_b32 v50, v204
	v_dual_mov_b32 v53, v204 :: v_dual_mov_b32 v52, v204
	v_dual_mov_b32 v55, v204 :: v_dual_mov_b32 v54, v204
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v56, v204
	v_dual_mov_b32 v43, v204 :: v_dual_mov_b32 v42, v204
	v_dual_mov_b32 v45, v204 :: v_dual_mov_b32 v44, v204
	v_dual_mov_b32 v47, v204 :: v_dual_mov_b32 v46, v204
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v48, v204
	v_dual_mov_b32 v35, v204 :: v_dual_mov_b32 v34, v204
	v_dual_mov_b32 v37, v204 :: v_dual_mov_b32 v36, v204
	v_dual_mov_b32 v39, v204 :: v_dual_mov_b32 v38, v204
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v40, v204
	v_dual_mov_b32 v27, v204 :: v_dual_mov_b32 v26, v204
	v_dual_mov_b32 v29, v204 :: v_dual_mov_b32 v28, v204
	v_dual_mov_b32 v31, v204 :: v_dual_mov_b32 v30, v204
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v32, v204
	v_dual_mov_b32 v19, v204 :: v_dual_mov_b32 v18, v204
	v_dual_mov_b32 v21, v204 :: v_dual_mov_b32 v20, v204
	v_dual_mov_b32 v23, v204 :: v_dual_mov_b32 v22, v204
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v24, v204
	v_dual_mov_b32 v11, v204 :: v_dual_mov_b32 v10, v204
	v_dual_mov_b32 v13, v204 :: v_dual_mov_b32 v12, v204
	v_dual_mov_b32 v15, v204 :: v_dual_mov_b32 v14, v204
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v16, v204
	v_dual_mov_b32 v3, v204 :: v_dual_mov_b32 v2, v204
	v_dual_mov_b32 v5, v204 :: v_dual_mov_b32 v4, v204
	v_dual_mov_b32 v7, v204 :: v_dual_mov_b32 v6, v204
	v_mov_b32_e32 v8, v204
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0
	v_dual_mov_b32 v244, 0x4b400000 :: v_dual_mov_b32 v245, 0x4b400000
	v_dual_mov_b32 v246, 0x4b400000 :: v_dual_mov_b32 v247, 0x4b400000
	v_dual_mov_b32 v248, 0x4b400000 :: v_dual_mov_b32 v249, 0x4b400000
	v_dual_mov_b32 v250, 0x4b400000 :: v_dual_mov_b32 v251, 0x4b400000
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_branch .LBB0_37
.LBB0_36:                               ;   in Loop: Header=BB0_37 Depth=1
	s_setprio 1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_wait_dscnt 0x0
	v_add_f32_e32 v200, v200, v231
	v_add_f32_e32 v196, v196, v232
	s_add_co_i32 s7, s7, 32
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s7, s4
	v_fmac_f32_e32 v200, v212, v255
	v_cvt_pk_fp8_f32 v232.h, v228, v229
	v_cvt_pk_fp8_f32 v233.l, v129, v130
	v_cvt_pk_fp8_f32 v233.h, v131, v132
	v_cvt_pk_fp8_f32 v228.l, v218, v219
	v_cvt_pk_fp8_f32 v228.h, v220, v221
	v_mad_co_u64_u32 v[219:220], null, 0x1400, s8, v[192:193]
	v_cvt_pk_fp8_f32 v229.l, v137, v138
	v_cvt_pk_fp8_f32 v232.l, v226, v227
	v_cvt_pk_fp8_f32 v234.l, v133, v134
	v_add_nc_u32_e32 v137, 0x3100, v219
	ds_load_2addr_b64 v[129:132], v137 offset0:64 offset1:66
	v_cvt_pk_fp8_f32 v229.h, v139, v140
	v_cvt_pk_fp8_f32 v227.l, v214, v215
	v_cvt_pk_fp8_f32 v227.h, v216, v217
	v_cvt_pk_fp8_f32 v234.h, v135, v136
	ds_load_2addr_b64 v[133:136], v137 offset0:144 offset1:146
	ds_load_2addr_b64 v[137:140], v137 offset0:224 offset1:226
	v_fmac_f32_e32 v196, v211, v213
	v_cvt_pk_fp8_f32 v230.l, v141, v142
	v_cvt_pk_fp8_f32 v230.h, v143, v144
	v_cvt_pk_fp8_f32 v231.l, v222, v223
	v_cvt_pk_fp8_f32 v231.h, v224, v225
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[129:130], v[227:228], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[129:130], v[233:234], v[57:64]
	v_add_nc_u32_e32 v129, 0x3900, v219
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[131:132], v[231:232], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[131:132], v[229:230], v[57:64]
	ds_load_2addr_b64 v[141:144], v129 offset0:48 offset1:50
	ds_load_2addr_b64 v[211:214], v129 offset0:128 offset1:130
	ds_load_2addr_b64 v[215:218], v129 offset0:208 offset1:210
	v_add_nc_u32_e32 v129, 0x4100, v219
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[133:134], v[227:228], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[133:134], v[233:234], v[49:56]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[137:138], v[227:228], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[137:138], v[233:234], v[41:48]
	ds_load_2addr_b64 v[219:222], v129 offset0:32 offset1:34
	ds_load_2addr_b64 v[223:226], v129 offset0:112 offset1:114
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[135:136], v[231:232], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[135:136], v[229:230], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[139:140], v[231:232], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[139:140], v[229:230], v[41:48]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[141:142], v[227:228], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[141:142], v[233:234], v[33:40]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[211:212], v[227:228], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[211:212], v[233:234], v[25:32]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[215:216], v[227:228], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[215:216], v[233:234], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[143:144], v[231:232], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[143:144], v[229:230], v[33:40]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[219:220], v[227:228], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[219:220], v[233:234], v[9:16]
	s_wait_dscnt 0x0
	s_setprio 0
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[223:224], v[227:228], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[223:224], v[233:234], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[213:214], v[231:232], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[213:214], v[229:230], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[217:218], v[231:232], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[217:218], v[229:230], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[221:222], v[231:232], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[221:222], v[229:230], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[225:226], v[231:232], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[225:226], v[229:230], v[1:8]
	v_dual_mov_b32 v213, v210 :: v_dual_mov_b32 v214, v209
	v_dual_mov_b32 v211, v196 :: v_dual_mov_b32 v212, v200
	s_cbranch_scc1 .LBB0_44
.LBB0_37:                               ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_hi_u32 s33, s6, 0xaaaaaaab
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s33, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s33, s33, 3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s8, s6, s33
	s_wait_alu depctr_sa_sdst(0)
	s_barrier_wait -1
	s_setprio 3
	s_wait_loadcnt 0x0
	s_and_saveexec_b32 s9, s0
	s_cbranch_execz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_37 Depth=1
	s_add_co_i32 s1, s7, 64
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s8, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s10, 3
	s_cselect_b32 s10, 0, s10
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s1, s12
	s_cselect_b32 s1, s1, 0
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v129, s1, v201
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v129, s11, s14, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, s15, 0, s11
	s_mul_i32 s11, s10, 0x1400
	v_mul_lo_u32 v131, v129, s27
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v134, v203, s11, 0x3300
	v_mul_lo_u32 v132, v130, s26
	v_mad_co_u64_u32 v[129:130], null, v129, s26, s[28:29]
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v130, v132, v130, v131
	v_mad_co_u64_u32 v[131:132], null, 0x1100, s10, v[189:190]
	v_or_b32_e32 v132, s1, v202
	ds_store_2addr_b64 v131, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v134, v[149:150], v[151:152] offset1:1
	v_lshlrev_b64_e32 v[129:130], 7, v[129:130]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v129, s1, v205, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v130, null, v206, v130, s1
	v_add_co_u32 v132, s1, v190, v132
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v133, null, 0, v191, s1
	global_load_b128 v[145:148], v[129:130], off
	global_load_b128 v[149:152], v[132:133], off
.LBB0_39:                               ;   in Loop: Header=BB0_37 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_mad_co_u64_u32 v[209:210], null, 0x1100, s8, v[193:194]
	v_add_nc_u32_e32 v196, 0x800, v209
	ds_load_2addr_b64 v[231:234], v196 offset0:16 offset1:18
	ds_load_2addr_b64 v[235:238], v209 offset1:2
	ds_load_2addr_b64 v[239:242], v209 offset0:4 offset1:6
	s_add_co_i32 s10, s3, s6
	s_mov_b32 s9, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s11, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[10:11], s[10:11], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[10:11], s[16:17], s[10:11]
	s_wait_dscnt 0x2
	s_barrier_signal -1
	v_wmma_i32_16x16x16_iu8 v[223:230], v[231:232], v[169:170], v[244:251] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[231:232], v[157:158], v[244:251] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x16_iu8 v[215:222], v[235:236], v[169:170], v[244:251] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_i32_16x16x16_iu8 v[223:230], v[233:234], v[171:172], v[223:230] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[233:234], v[161:162], v[129:136] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[231:234], v196 offset0:20 offset1:22
	v_wmma_i32_16x16x16_iu8 v[137:144], v[235:236], v[157:158], v[244:251] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[215:222], v[237:238], v[171:172], v[215:222] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[237:238], v[161:162], v[137:144] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x16_iu8 v[215:222], v[239:240], v[163:164], v[215:222] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[239:240], v[159:160], v[137:144] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[215:222], v[241:242], v[179:180], v[215:222] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[241:242], v[167:168], v[137:144] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x16_iu8 v[223:230], v[231:232], v[163:164], v[223:230] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[231:232], v[159:160], v[129:136] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[223:230], v[233:234], v[179:180], v[223:230] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_wmma_i32_16x16x16_iu8 v[129:136], v[233:234], v[167:168], v[129:136] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[231:234], v209 offset0:8 offset1:10
	ds_load_2addr_b64 v[235:238], v196 offset0:24 offset1:26
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x16_iu8 v[215:222], v[231:232], v[173:174], v[215:222] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x16_iu8 v[223:230], v[235:236], v[173:174], v[223:230] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[137:144], v[231:232], v[165:166], v[137:144] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[235:236], v[165:166], v[129:136] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[215:222], v[233:234], v[183:184], v[215:222] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_i32_16x16x16_iu8 v[223:230], v[237:238], v[183:184], v[223:230] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[137:144], v[233:234], v[177:178], v[137:144] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_i32_16x16x16_iu8 v[129:136], v[237:238], v[177:178], v[129:136] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[231:234], v209 offset0:12 offset1:14
	ds_load_2addr_b64 v[235:238], v196 offset0:28 offset1:30
	global_load_b32 v196, v204, s[10:11]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x16_iu8 v[215:222], v[231:232], v[181:182], v[215:222] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x16_iu8 v[223:230], v[235:236], v[181:182], v[223:230] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[137:144], v[231:232], v[175:176], v[137:144] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[235:236], v[175:176], v[129:136] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[215:222], v[233:234], v[187:188], v[215:222] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_i32_16x16x16_iu8 v[223:230], v[237:238], v[187:188], v[223:230] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[137:144], v[233:234], v[185:186], v[137:144] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_i32_16x16x16_iu8 v[129:136], v[237:238], v[185:186], v[129:136] neg_lo:[1,1,0]
	v_max_num_f32_e32 v209, v215, v216
	v_max_num_f32_e32 v239, v137, v138
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v217, v218
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v139, v140
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v219, v220
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v141, v142
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v221, v222
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v143, v144
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v223, v224
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v129, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v225, v226
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v131, v132
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v227, v228
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v133, v134
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v209, v209, v229, v230
	s_delay_alu instid0(VALU_DEP_2)
	v_max3_num_f32 v239, v239, v135, v136
	ds_bpermute_b32 v240, v207, v209
	ds_bpermute_b32 v241, v207, v239
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v196, v198, v196
	v_max_num_f32_e32 v242, v214, v214
	v_max_num_f32_e32 v243, v213, v213
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v240, v240, v240
	v_max_num_f32_e32 v241, v241, v241
	s_delay_alu instid0(VALU_DEP_2)
	v_max_num_f32_e32 v209, v209, v240
	s_delay_alu instid0(VALU_DEP_2)
	v_max_num_f32_e32 v239, v239, v241
	s_mov_b32 s34, 0x4b400000
	v_cmp_lt_f32_e64 s35, 0x3b2aaaab, v196
	s_wait_alu depctr_va_sdst(0)
	s_setprio 2
	s_cmp_eq_u32 s35, 0
	s_cbranch_scc1 .Lnocvt_fast
	s_mov_b32 s34, 0
	v_sub_f32_e32 v215, v215, v244
	v_sub_f32_e32 v216, v216, v244
	v_sub_f32_e32 v217, v217, v244
	v_sub_f32_e32 v218, v218, v244
	v_sub_f32_e32 v219, v219, v244
	v_sub_f32_e32 v220, v220, v244
	v_sub_f32_e32 v221, v221, v244
	v_sub_f32_e32 v222, v222, v244
	v_sub_f32_e32 v223, v223, v244
	v_sub_f32_e32 v224, v224, v244
	v_sub_f32_e32 v225, v225, v244
	v_sub_f32_e32 v226, v226, v244
	v_sub_f32_e32 v227, v227, v244
	v_sub_f32_e32 v228, v228, v244
	v_sub_f32_e32 v229, v229, v244
	v_sub_f32_e32 v230, v230, v244
	v_sub_f32_e32 v137, v137, v244
	v_sub_f32_e32 v138, v138, v244
	v_sub_f32_e32 v139, v139, v244
	v_sub_f32_e32 v140, v140, v244
	v_sub_f32_e32 v141, v141, v244
	v_sub_f32_e32 v142, v142, v244
	v_sub_f32_e32 v143, v143, v244
	v_sub_f32_e32 v144, v144, v244
	v_sub_f32_e32 v129, v129, v244
	v_sub_f32_e32 v130, v130, v244
	v_sub_f32_e32 v131, v131, v244
	v_sub_f32_e32 v132, v132, v244
	v_sub_f32_e32 v133, v133, v244
	v_sub_f32_e32 v134, v134, v244
	v_sub_f32_e32 v135, v135, v244
	v_sub_f32_e32 v136, v136, v244
	v_sub_f32_e32 v209, v209, v244
	v_sub_f32_e32 v239, v239, v244
.Lnocvt_fast:
	s_wait_alu depctr_sa_sdst(0)
	v_fmaak_f32 v254, s34, v196, 0x410ce979
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v209, v209, v196, -v254
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v239, v239, v196, -v254
	s_delay_alu instid0(VALU_DEP_2)
	v_max_num_f32_e32 v209, v242, v209
	s_delay_alu instid0(VALU_DEP_2)
	v_max_num_f32_e32 v210, v243, v239
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v252, s34, v196, v209
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v253, s34, v196, v210
	s_delay_alu instid0(VALU_DEP_4)
	v_sub_f32_e32 v255, v214, v209
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v233, v215, v196, -v252
	s_delay_alu instid0(VALU_DEP_3)
	v_fma_f32 v235, v137, v196, -v253
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v236, v129, v196, -v253
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v255, v255
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v214, v233
	v_fma_f32 v234, v216, v196, -v252
	v_exp_f32_e32 v129, v235
	v_exp_f32_e32 v137, v236
	v_fma_f32 v237, v138, v196, -v253
	v_fma_f32 v238, v130, v196, -v253
	v_exp_f32_e32 v215, v234
	v_fma_f32 v233, v217, v196, -v252
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v130, v237
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v138, v238
	v_fma_f32 v235, v139, v196, -v253
	v_fma_f32 v236, v131, v196, -v253
	v_exp_f32_e32 v216, v233
	v_fma_f32 v234, v218, v196, -v252
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v131, v235
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v139, v236
	v_fma_f32 v237, v140, v196, -v253
	v_fma_f32 v238, v132, v196, -v253
	v_exp_f32_e32 v217, v234
	v_fma_f32 v233, v219, v196, -v252
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v132, v237
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v140, v238
	v_fma_f32 v235, v141, v196, -v253
	v_fma_f32 v236, v133, v196, -v253
	v_dual_add_f32 v200, v214, v215 :: v_dual_add_f32 v243, v129, v130
	v_exp_f32_e32 v218, v233
	v_fma_f32 v234, v220, v196, -v252
	v_exp_f32_e32 v133, v235
	v_exp_f32_e32 v141, v236
	v_fma_f32 v237, v142, v196, -v253
	v_fma_f32 v238, v134, v196, -v253
	v_dual_add_f32 v200, v216, v200 :: v_dual_add_f32 v243, v131, v243
	v_exp_f32_e32 v219, v234
	v_fma_f32 v233, v221, v196, -v252
	v_exp_f32_e32 v134, v237
	v_exp_f32_e32 v142, v238
	v_fma_f32 v235, v143, v196, -v253
	v_fma_f32 v236, v135, v196, -v253
	v_dual_add_f32 v200, v217, v200 :: v_dual_add_f32 v243, v132, v243
	v_exp_f32_e32 v220, v233
	v_fma_f32 v234, v222, v196, -v252
	v_exp_f32_e32 v135, v235
	v_exp_f32_e32 v143, v236
	v_fma_f32 v237, v144, v196, -v253
	v_fma_f32 v238, v136, v196, -v253
	v_dual_add_f32 v200, v218, v200 :: v_dual_add_f32 v243, v133, v243
	v_exp_f32_e32 v221, v234
	v_fma_f32 v233, v223, v196, -v252
	v_exp_f32_e32 v136, v237
	v_exp_f32_e32 v144, v238
	v_dual_add_f32 v200, v219, v200 :: v_dual_add_f32 v243, v134, v243
	s_delay_alu instid0(VALU_DEP_4)
	v_exp_f32_e32 v222, v233
	v_fma_f32 v234, v224, v196, -v252
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v200, v220, v200 :: v_dual_add_f32 v243, v135, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v223, v234
	v_fma_f32 v233, v225, v196, -v252
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v200, v221, v200 :: v_dual_add_f32 v243, v136, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v224, v233
	v_fma_f32 v234, v226, v196, -v252
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v200, v222, v200 :: v_dual_add_f32 v243, v137, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v225, v234
	v_fma_f32 v233, v227, v196, -v252
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v200, v223, v200 :: v_dual_add_f32 v243, v138, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v226, v233
	v_fma_f32 v234, v228, v196, -v252
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v200, v224, v200 :: v_dual_add_f32 v243, v139, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v227, v234
	v_fma_f32 v233, v229, v196, -v252
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v200, v225, v200 :: v_dual_add_f32 v243, v140, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v228, v233
	v_fma_f32 v234, v230, v196, -v252
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v200, v226, v200 :: v_dual_add_f32 v243, v141, v243
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v229, v234
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_2)
	v_dual_add_f32 v200, v227, v200 :: v_dual_add_f32 v243, v142, v243
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_dual_add_f32 v200, v228, v200 :: v_dual_add_f32 v243, v143, v243
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v200, v229, v200
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v196, v144, v243
	v_sub_f32_e32 v213, v213, v210
	ds_bpermute_b32 v231, v207, v200
	ds_bpermute_b32 v232, v207, v196
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v213, v213
	v_cmpx_neq_f32_e32 1.0, v255
	s_cbranch_execz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_37 Depth=1
	v_dual_mul_f32 v121, v255, v121 :: v_dual_mul_f32 v122, v255, v122
	v_dual_mul_f32 v123, v255, v123 :: v_dual_mul_f32 v124, v255, v124
	v_dual_mul_f32 v125, v255, v125 :: v_dual_mul_f32 v126, v255, v126
	v_dual_mul_f32 v127, v255, v127 :: v_dual_mul_f32 v128, v255, v128
	v_dual_mul_f32 v113, v255, v113 :: v_dual_mul_f32 v114, v255, v114
	v_dual_mul_f32 v115, v255, v115 :: v_dual_mul_f32 v116, v255, v116
	v_dual_mul_f32 v117, v255, v117 :: v_dual_mul_f32 v118, v255, v118
	v_dual_mul_f32 v119, v255, v119 :: v_dual_mul_f32 v120, v255, v120
	v_dual_mul_f32 v105, v255, v105 :: v_dual_mul_f32 v106, v255, v106
	v_dual_mul_f32 v107, v255, v107 :: v_dual_mul_f32 v108, v255, v108
	v_dual_mul_f32 v109, v255, v109 :: v_dual_mul_f32 v110, v255, v110
	v_dual_mul_f32 v111, v255, v111 :: v_dual_mul_f32 v112, v255, v112
	v_dual_mul_f32 v97, v255, v97 :: v_dual_mul_f32 v98, v255, v98
	v_dual_mul_f32 v99, v255, v99 :: v_dual_mul_f32 v100, v255, v100
	v_dual_mul_f32 v101, v255, v101 :: v_dual_mul_f32 v102, v255, v102
	v_dual_mul_f32 v103, v255, v103 :: v_dual_mul_f32 v104, v255, v104
	v_dual_mul_f32 v89, v255, v89 :: v_dual_mul_f32 v90, v255, v90
	v_dual_mul_f32 v91, v255, v91 :: v_dual_mul_f32 v92, v255, v92
	v_dual_mul_f32 v93, v255, v93 :: v_dual_mul_f32 v94, v255, v94
	v_dual_mul_f32 v95, v255, v95 :: v_dual_mul_f32 v96, v255, v96
	v_dual_mul_f32 v81, v255, v81 :: v_dual_mul_f32 v82, v255, v82
	v_dual_mul_f32 v83, v255, v83 :: v_dual_mul_f32 v84, v255, v84
	v_dual_mul_f32 v85, v255, v85 :: v_dual_mul_f32 v86, v255, v86
	v_dual_mul_f32 v87, v255, v87 :: v_dual_mul_f32 v88, v255, v88
	v_dual_mul_f32 v73, v255, v73 :: v_dual_mul_f32 v74, v255, v74
	v_dual_mul_f32 v75, v255, v75 :: v_dual_mul_f32 v76, v255, v76
	v_dual_mul_f32 v77, v255, v77 :: v_dual_mul_f32 v78, v255, v78
	v_dual_mul_f32 v79, v255, v79 :: v_dual_mul_f32 v80, v255, v80
	v_dual_mul_f32 v65, v255, v65 :: v_dual_mul_f32 v66, v255, v66
	v_dual_mul_f32 v67, v255, v67 :: v_dual_mul_f32 v68, v255, v68
	v_dual_mul_f32 v69, v255, v69 :: v_dual_mul_f32 v70, v255, v70
	v_dual_mul_f32 v71, v255, v71 :: v_dual_mul_f32 v72, v255, v72
.LBB0_41:                               ;   in Loop: Header=BB0_37 Depth=1
	s_or_b32 exec_lo, exec_lo, s9
	s_mov_b32 s9, exec_lo
	v_cmpx_neq_f32_e32 1.0, v213
	s_cbranch_execz .LBB0_36
; %bb.42:                               ;   in Loop: Header=BB0_37 Depth=1
	v_dual_mul_f32 v57, v213, v57 :: v_dual_mul_f32 v58, v213, v58
	v_dual_mul_f32 v59, v213, v59 :: v_dual_mul_f32 v60, v213, v60
	v_dual_mul_f32 v61, v213, v61 :: v_dual_mul_f32 v62, v213, v62
	v_dual_mul_f32 v63, v213, v63 :: v_dual_mul_f32 v64, v213, v64
	v_dual_mul_f32 v49, v213, v49 :: v_dual_mul_f32 v50, v213, v50
	v_dual_mul_f32 v51, v213, v51 :: v_dual_mul_f32 v52, v213, v52
	v_dual_mul_f32 v53, v213, v53 :: v_dual_mul_f32 v54, v213, v54
	v_dual_mul_f32 v55, v213, v55 :: v_dual_mul_f32 v56, v213, v56
	v_dual_mul_f32 v41, v213, v41 :: v_dual_mul_f32 v42, v213, v42
	v_dual_mul_f32 v43, v213, v43 :: v_dual_mul_f32 v44, v213, v44
	v_dual_mul_f32 v45, v213, v45 :: v_dual_mul_f32 v46, v213, v46
	v_dual_mul_f32 v47, v213, v47 :: v_dual_mul_f32 v48, v213, v48
	v_dual_mul_f32 v33, v213, v33 :: v_dual_mul_f32 v34, v213, v34
	v_dual_mul_f32 v35, v213, v35 :: v_dual_mul_f32 v36, v213, v36
	v_dual_mul_f32 v37, v213, v37 :: v_dual_mul_f32 v38, v213, v38
	v_dual_mul_f32 v39, v213, v39 :: v_dual_mul_f32 v40, v213, v40
	v_dual_mul_f32 v25, v213, v25 :: v_dual_mul_f32 v26, v213, v26
	v_dual_mul_f32 v27, v213, v27 :: v_dual_mul_f32 v28, v213, v28
	v_dual_mul_f32 v29, v213, v29 :: v_dual_mul_f32 v30, v213, v30
	v_dual_mul_f32 v31, v213, v31 :: v_dual_mul_f32 v32, v213, v32
	v_dual_mul_f32 v17, v213, v17 :: v_dual_mul_f32 v18, v213, v18
	v_dual_mul_f32 v19, v213, v19 :: v_dual_mul_f32 v20, v213, v20
	v_dual_mul_f32 v21, v213, v21 :: v_dual_mul_f32 v22, v213, v22
	v_dual_mul_f32 v23, v213, v23 :: v_dual_mul_f32 v24, v213, v24
	v_dual_mul_f32 v9, v213, v9 :: v_dual_mul_f32 v10, v213, v10
	v_dual_mul_f32 v11, v213, v11 :: v_dual_mul_f32 v12, v213, v12
	v_dual_mul_f32 v13, v213, v13 :: v_dual_mul_f32 v14, v213, v14
	v_dual_mul_f32 v15, v213, v15 :: v_dual_mul_f32 v16, v213, v16
	v_dual_mul_f32 v1, v213, v1 :: v_dual_mul_f32 v2, v213, v2
	v_dual_mul_f32 v3, v213, v3 :: v_dual_mul_f32 v4, v213, v4
	v_dual_mul_f32 v5, v213, v5 :: v_dual_mul_f32 v6, v213, v6
	v_dual_mul_f32 v7, v213, v7 :: v_dual_mul_f32 v8, v213, v8
	s_branch .LBB0_36
.LBB0_43:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v209, 0xff800000
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v210, 0xff800000 :: v_dual_mov_b32 v7, v8
	v_dual_mov_b32 v6, v8 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v4, v8 :: v_dual_mov_b32 v3, v8
	v_dual_mov_b32 v2, v8 :: v_dual_mov_b32 v1, v8
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v15, v8
	v_dual_mov_b32 v14, v8 :: v_dual_mov_b32 v13, v8
	v_dual_mov_b32 v12, v8 :: v_dual_mov_b32 v11, v8
	v_dual_mov_b32 v10, v8 :: v_dual_mov_b32 v9, v8
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v23, v8
	v_dual_mov_b32 v22, v8 :: v_dual_mov_b32 v21, v8
	v_dual_mov_b32 v20, v8 :: v_dual_mov_b32 v19, v8
	v_dual_mov_b32 v18, v8 :: v_dual_mov_b32 v17, v8
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v31, v8
	v_dual_mov_b32 v30, v8 :: v_dual_mov_b32 v29, v8
	v_dual_mov_b32 v28, v8 :: v_dual_mov_b32 v27, v8
	v_dual_mov_b32 v26, v8 :: v_dual_mov_b32 v25, v8
	v_dual_mov_b32 v40, v8 :: v_dual_mov_b32 v39, v8
	v_dual_mov_b32 v38, v8 :: v_dual_mov_b32 v37, v8
	v_dual_mov_b32 v36, v8 :: v_dual_mov_b32 v35, v8
	v_dual_mov_b32 v34, v8 :: v_dual_mov_b32 v33, v8
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v47, v8
	v_dual_mov_b32 v46, v8 :: v_dual_mov_b32 v45, v8
	v_dual_mov_b32 v44, v8 :: v_dual_mov_b32 v43, v8
	v_dual_mov_b32 v42, v8 :: v_dual_mov_b32 v41, v8
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v8
	v_dual_mov_b32 v54, v8 :: v_dual_mov_b32 v53, v8
	v_dual_mov_b32 v52, v8 :: v_dual_mov_b32 v51, v8
	v_dual_mov_b32 v50, v8 :: v_dual_mov_b32 v49, v8
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v63, v8
	v_dual_mov_b32 v62, v8 :: v_dual_mov_b32 v61, v8
	v_dual_mov_b32 v60, v8 :: v_dual_mov_b32 v59, v8
	v_dual_mov_b32 v58, v8 :: v_dual_mov_b32 v57, v8
	v_dual_mov_b32 v72, v8 :: v_dual_mov_b32 v71, v8
	v_dual_mov_b32 v70, v8 :: v_dual_mov_b32 v69, v8
	v_dual_mov_b32 v68, v8 :: v_dual_mov_b32 v67, v8
	v_dual_mov_b32 v66, v8 :: v_dual_mov_b32 v65, v8
	v_dual_mov_b32 v80, v8 :: v_dual_mov_b32 v79, v8
	v_dual_mov_b32 v78, v8 :: v_dual_mov_b32 v77, v8
	v_dual_mov_b32 v76, v8 :: v_dual_mov_b32 v75, v8
	v_dual_mov_b32 v74, v8 :: v_dual_mov_b32 v73, v8
	v_dual_mov_b32 v88, v8 :: v_dual_mov_b32 v87, v8
	v_dual_mov_b32 v86, v8 :: v_dual_mov_b32 v85, v8
	v_dual_mov_b32 v84, v8 :: v_dual_mov_b32 v83, v8
	v_dual_mov_b32 v82, v8 :: v_dual_mov_b32 v81, v8
	v_dual_mov_b32 v96, v8 :: v_dual_mov_b32 v95, v8
	v_dual_mov_b32 v94, v8 :: v_dual_mov_b32 v93, v8
	v_dual_mov_b32 v92, v8 :: v_dual_mov_b32 v91, v8
	v_dual_mov_b32 v90, v8 :: v_dual_mov_b32 v89, v8
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v103, v8
	v_dual_mov_b32 v102, v8 :: v_dual_mov_b32 v101, v8
	v_dual_mov_b32 v100, v8 :: v_dual_mov_b32 v99, v8
	v_dual_mov_b32 v98, v8 :: v_dual_mov_b32 v97, v8
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v111, v8
	v_dual_mov_b32 v110, v8 :: v_dual_mov_b32 v109, v8
	v_dual_mov_b32 v108, v8 :: v_dual_mov_b32 v107, v8
	v_dual_mov_b32 v106, v8 :: v_dual_mov_b32 v105, v8
	v_dual_mov_b32 v120, v8 :: v_dual_mov_b32 v119, v8
	v_dual_mov_b32 v118, v8 :: v_dual_mov_b32 v117, v8
	v_dual_mov_b32 v116, v8 :: v_dual_mov_b32 v115, v8
	v_dual_mov_b32 v114, v8 :: v_dual_mov_b32 v113, v8
	v_dual_mov_b32 v128, v8 :: v_dual_mov_b32 v127, v8
	v_dual_mov_b32 v126, v8 :: v_dual_mov_b32 v125, v8
	v_dual_mov_b32 v124, v8 :: v_dual_mov_b32 v123, v8
	v_dual_mov_b32 v122, v8 :: v_dual_mov_b32 v121, v8
	v_mov_b32_e32 v200, v8
	v_mov_b32_e32 v196, v8
	s_wait_dscnt 0x0
	s_barrier_signal -1
.LBB0_44:
	s_barrier_wait -1
	s_setprio 0
	s_ashr_i32 s1, s5, 5
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s4, s25
	s_cbranch_scc0 .LBB0_52
; %bb.45:
	s_wait_loadcnt_dscnt 0x0
	s_mul_hi_u32 s33, s1, 0xaaaaaaab
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s33, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s33, s33, 3
	s_wait_alu depctr_sa_sdst(0)
	s_sub_co_i32 s24, s1, s33
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s0
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_lshrrev_b32_e32 v129, 3, v0
	v_lshlrev_b32_e32 v0, 4, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s24, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s0, 3
	s_cselect_b32 s0, 0, s0
	s_wait_alu depctr_sa_sdst(0)
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s0, 0x1400
	v_mul_u32_u24_e32 v129, 0x88, v129
	v_and_b32_e32 v130, 0x70, v0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_u32_u24 v131, v197, 40, s6
	v_and_b32_e32 v0, 16, v0
	s_mulk_i32 s0, 0x1100
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v129, s0, v129, v130
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v0, v131, v0, 0x3300
	ds_store_2addr_b64 v129, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v0, v[149:150], v[151:152] offset1:1
.LBB0_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_add_co_i32 s0, s3, s1
	v_mov_b32_e32 v0, 0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	v_mul_u32_u24_e32 v129, 0x88, v195
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_xor_b32_e32 v147, 16, v199
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	v_or_b32_e32 v148, s4, v194
	global_load_b32 v0, v0, s[0:1]
	s_mul_i32 s0, s24, 0x1100
	s_wait_alu depctr_sa_sdst(0)
	v_add3_u32 v145, s0, v129, v194
	v_cmp_gt_u32_e64 s0, 32, v147
	v_or_b32_e32 v149, 3, v148
	v_cmp_gt_i32_e64 s15, s25, v148
	v_or_b32_e32 v150, 4, v148
	v_add_nc_u32_e32 v146, 0x800, v145
	ds_load_2addr_b64 v[223:226], v145 offset0:12 offset1:14
	v_or_b32_e32 v151, 5, v148
	ds_load_2addr_b64 v[189:192], v145 offset1:2
	ds_load_2addr_b64 v[201:204], v146 offset0:16 offset1:18
	ds_load_2addr_b64 v[205:208], v145 offset0:4 offset1:6
	ds_load_2addr_b64 v[211:214], v146 offset0:20 offset1:22
	ds_load_2addr_b64 v[215:218], v145 offset0:8 offset1:10
	ds_load_2addr_b64 v[219:222], v146 offset0:24 offset1:26
	ds_load_2addr_b64 v[227:230], v146 offset0:28 offset1:30
	v_or_b32_e32 v146, 1, v148
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v145, v199, v147, s0
	v_or_b32_e32 v147, 2, v148
	v_cmp_gt_i32_e64 s13, s25, v149
	v_or_b32_e32 v152, 6, v148
	v_cmp_gt_i32_e64 s16, s25, v146
	v_cmp_gt_i32_e64 s12, s25, v150
	v_cmp_gt_i32_e64 s14, s25, v147
	v_cmp_gt_i32_e64 s11, s25, v151
	v_cmp_gt_i32_e64 s10, s25, v152
	s_wait_dscnt 0x6
	v_wmma_i32_16x16x16_iu8 v[129:136], v[189:190], v[169:170], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x5
	v_wmma_i32_16x16x16_iu8 v[137:144], v[201:202], v[169:170], 0 neg_lo:[1,1,0]
	v_or_b32_e32 v169, 17, v148
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_i32_16x16x16_iu8 v[129:136], v[191:192], v[171:172], v[129:136] neg_lo:[1,1,0]
	v_or_b32_e32 v170, 18, v148
	v_wmma_i32_16x16x16_iu8 v[137:144], v[203:204], v[171:172], v[137:144] neg_lo:[1,1,0]
	v_or_b32_e32 v171, 19, v148
	v_cmp_gt_i32_e64 s7, s25, v169
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x16_iu8 v[129:136], v[205:206], v[163:164], v[129:136] neg_lo:[1,1,0]
	v_or_b32_e32 v172, 20, v148
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x16_iu8 v[137:144], v[211:212], v[163:164], v[137:144] neg_lo:[1,1,0]
	v_or_b32_e32 v163, 7, v148
	v_or_b32_e32 v164, 16, v148
	v_wmma_i32_16x16x16_iu8 v[129:136], v[207:208], v[179:180], v[129:136] neg_lo:[1,1,0]
	v_cmp_gt_i32_e64 s6, s25, v170
	v_wmma_i32_16x16x16_iu8 v[137:144], v[213:214], v[179:180], v[137:144] neg_lo:[1,1,0]
	v_cmp_gt_i32_e64 s9, s25, v163
	v_cmp_gt_i32_e64 s8, s25, v164
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x16_iu8 v[129:136], v[215:216], v[173:174], v[129:136] neg_lo:[1,1,0]
	v_cmp_gt_i32_e64 s5, s25, v171
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x16_iu8 v[137:144], v[219:220], v[173:174], v[137:144] neg_lo:[1,1,0]
	v_or_b32_e32 v173, 21, v148
	v_or_b32_e32 v174, 22, v148
	v_wmma_i32_16x16x16_iu8 v[129:136], v[217:218], v[183:184], v[129:136] neg_lo:[1,1,0]
	v_or_b32_e32 v179, 23, v148
	v_wmma_i32_16x16x16_iu8 v[137:144], v[221:222], v[183:184], v[137:144] neg_lo:[1,1,0]
	v_cmp_gt_i32_e64 s4, s25, v172
	v_cmp_gt_i32_e64 s1, s25, v173
	v_wmma_i32_16x16x16_iu8 v[129:136], v[223:224], v[181:182], v[129:136] neg_lo:[1,1,0]
	v_cmp_gt_i32_e64 s3, s25, v174
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x16_iu8 v[137:144], v[227:228], v[181:182], v[137:144] neg_lo:[1,1,0]
	v_cmp_gt_i32_e64 s0, s25, v179
	v_lshlrev_b32_e32 v171, 2, v145
	v_wmma_i32_16x16x16_iu8 v[129:136], v[225:226], v[187:188], v[129:136] neg_lo:[1,1,0]
	s_mov_b32 s25, exec_lo
	v_wmma_i32_16x16x16_iu8 v[137:144], v[229:230], v[187:188], v[137:144] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v130, v130
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v133, v133
	v_cndmask_b32_e64 v129, 0xff7fffff, v129, s15
	v_cndmask_b32_e64 v130, 0xff7fffff, v130, s16
	v_cvt_f32_i32_e32 v134, v134
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v131, 0xff7fffff, v131, s14
	v_cndmask_b32_e64 v132, 0xff7fffff, v132, s13
	v_cvt_f32_i32_e32 v135, v135
	v_max_num_f32_e32 v146, v129, v130
	v_cvt_f32_i32_e32 v136, v136
	v_cndmask_b32_e64 v133, 0xff7fffff, v133, s12
	v_cndmask_b32_e64 v134, 0xff7fffff, v134, s11
	v_cvt_f32_i32_e32 v137, v137
	v_max3_num_f32 v146, v146, v131, v132
	v_cvt_f32_i32_e32 v138, v138
	v_cndmask_b32_e64 v135, 0xff7fffff, v135, s10
	v_cndmask_b32_e64 v136, 0xff7fffff, v136, s9
	v_cvt_f32_i32_e32 v139, v139
	v_max3_num_f32 v146, v146, v133, v134
	v_cvt_f32_i32_e32 v140, v140
	v_cndmask_b32_e64 v137, 0xff7fffff, v137, s8
	v_cndmask_b32_e64 v138, 0xff7fffff, v138, s7
	v_cvt_f32_i32_e32 v141, v141
	v_max3_num_f32 v146, v146, v135, v136
	v_cvt_f32_i32_e32 v142, v142
	v_cndmask_b32_e64 v139, 0xff7fffff, v139, s6
	v_cndmask_b32_e64 v140, 0xff7fffff, v140, s5
	v_cvt_f32_i32_e32 v143, v143
	v_max3_num_f32 v146, v146, v137, v138
	v_cvt_f32_i32_e32 v144, v144
	v_cndmask_b32_e64 v141, 0xff7fffff, v141, s4
	v_cndmask_b32_e64 v142, 0xff7fffff, v142, s1
	v_cndmask_b32_e64 v143, 0xff7fffff, v143, s3
	v_max3_num_f32 v146, v146, v139, v140
	v_cndmask_b32_e64 v173, 0xff7fffff, v144, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v144, v146, v141, v142
	v_max3_num_f32 v144, v144, v143, v173
	ds_bpermute_b32 v145, v171, v144
	s_wait_loadcnt_dscnt 0x0
	v_dual_max_num_f32 v145, v145, v145 :: v_dual_mul_f32 v172, v198, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v144, v144, v145
	v_max_num_f32_e32 v0, v209, v209
	v_fmaak_f32 v144, v144, v172, 0xc10ce979
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v174, v0, v144
	v_fma_f32 v0, v129, v172, -v174
	v_fma_f32 v129, v130, v172, -v174
	v_fma_f32 v130, v131, v172, -v174
	v_fma_f32 v131, v132, v172, -v174
	v_fma_f32 v132, v133, v172, -v174
	v_exp_f32_e32 v0, v0
	v_exp_f32_e32 v129, v129
	v_exp_f32_e32 v130, v130
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v132, v132
	v_fma_f32 v133, v135, v172, -v174
	v_cndmask_b32_e64 v0, 0, v0, s15
	v_cndmask_b32_e64 v145, 0, v129, s16
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v146, 0, v130, s14
	v_fma_f32 v129, v134, v172, -v174
	v_cndmask_b32_e64 v147, 0, v131, s13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v148, 0, v132, s12
	v_add_f32_e32 v130, v0, v145
	v_fma_f32 v131, v136, v172, -v174
	v_exp_f32_e32 v129, v129
	v_exp_f32_e32 v133, v133
	v_add_f32_e32 v130, v146, v130
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v131, v131
	v_add_f32_e32 v130, v147, v130
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v149, 0, v129, s11
	v_cndmask_b32_e64 v150, 0, v133, s10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v129, v148, v130
	v_cndmask_b32_e64 v151, 0, v131, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v129, v149, v129
	v_add_f32_e32 v129, v150, v129
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v129, v151, v129
	v_fma_f32 v132, v137, v172, -v174
	v_fma_f32 v130, v138, v172, -v174
	v_fma_f32 v133, v139, v172, -v174
	v_fma_f32 v131, v140, v172, -v174
	v_fma_f32 v173, v173, v172, -v174
	v_exp_f32_e32 v132, v132
	v_exp_f32_e32 v130, v130
	v_exp_f32_e32 v133, v133
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v173, v173
	v_cndmask_b32_e64 v152, 0, v132, s8
	v_cndmask_b32_e64 v163, 0, v130, s7
	v_fma_f32 v132, v141, v172, -v174
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v164, 0, v133, s6
	v_fma_f32 v130, v142, v172, -v174
	v_add_f32_e32 v129, v152, v129
	v_cndmask_b32_e64 v170, 0, v131, s5
	v_exp_f32_e32 v132, v132
	v_fma_f32 v133, v143, v172, -v174
	v_exp_f32_e32 v179, v130
	v_add_f32_e32 v129, v163, v129
	v_wmma_i32_16x16x16_iu8 v[137:144], v[189:190], v[157:158], 0 neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v180, v133
	v_add_f32_e32 v129, v164, v129
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v169, 0, v132, s4
	v_wmma_i32_16x16x16_iu8 v[137:144], v[191:192], v[161:162], v[137:144] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v181, v170, v129
	v_wmma_i32_16x16x16_iu8 v[129:136], v[201:202], v[157:158], 0 neg_lo:[1,1,0]
	v_cndmask_b32_e64 v157, 0, v179, s1
	v_cndmask_b32_e64 v158, 0, v180, s3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v179, v169, v181
	v_wmma_i32_16x16x16_iu8 v[129:136], v[203:204], v[161:162], v[129:136] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[137:144], v[205:206], v[159:160], v[137:144] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v161, v157, v179
	v_wmma_i32_16x16x16_iu8 v[129:136], v[211:212], v[159:160], v[129:136] neg_lo:[1,1,0]
	v_cndmask_b32_e64 v159, 0, v173, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[207:208], v[167:168], v[137:144] neg_lo:[1,1,0]
	v_add_f32_e32 v160, v158, v161
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_i32_16x16x16_iu8 v[129:136], v[213:214], v[167:168], v[129:136] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[137:144], v[215:216], v[165:166], v[137:144] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v161, v159, v160
	v_wmma_i32_16x16x16_iu8 v[129:136], v[219:220], v[165:166], v[129:136] neg_lo:[1,1,0]
	v_sub_f32_e32 v160, v209, v174
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[217:218], v[177:178], v[137:144] neg_lo:[1,1,0]
	ds_bpermute_b32 v162, v171, v161
	v_wmma_i32_16x16x16_iu8 v[129:136], v[221:222], v[177:178], v[129:136] neg_lo:[1,1,0]
	v_exp_f32_e32 v160, v160
	v_wmma_i32_16x16x16_iu8 v[137:144], v[223:224], v[175:176], v[137:144] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[227:228], v[175:176], v[129:136] neg_lo:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[225:226], v[185:186], v[137:144] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[129:136], v[229:230], v[185:186], v[129:136] neg_lo:[1,1,0]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cmpx_neq_f32_e32 1.0, v160
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_dual_mul_f32 v121, v160, v121 :: v_dual_mul_f32 v122, v160, v122
	v_dual_mul_f32 v123, v160, v123 :: v_dual_mul_f32 v124, v160, v124
	v_dual_mul_f32 v125, v160, v125 :: v_dual_mul_f32 v126, v160, v126
	v_dual_mul_f32 v127, v160, v127 :: v_dual_mul_f32 v128, v160, v128
	v_dual_mul_f32 v113, v160, v113 :: v_dual_mul_f32 v114, v160, v114
	v_dual_mul_f32 v115, v160, v115 :: v_dual_mul_f32 v116, v160, v116
	v_dual_mul_f32 v117, v160, v117 :: v_dual_mul_f32 v118, v160, v118
	v_dual_mul_f32 v119, v160, v119 :: v_dual_mul_f32 v120, v160, v120
	v_dual_mul_f32 v105, v160, v105 :: v_dual_mul_f32 v106, v160, v106
	v_dual_mul_f32 v107, v160, v107 :: v_dual_mul_f32 v108, v160, v108
	v_dual_mul_f32 v109, v160, v109 :: v_dual_mul_f32 v110, v160, v110
	v_dual_mul_f32 v111, v160, v111 :: v_dual_mul_f32 v112, v160, v112
	v_dual_mul_f32 v97, v160, v97 :: v_dual_mul_f32 v98, v160, v98
	v_dual_mul_f32 v99, v160, v99 :: v_dual_mul_f32 v100, v160, v100
	v_dual_mul_f32 v101, v160, v101 :: v_dual_mul_f32 v102, v160, v102
	v_dual_mul_f32 v103, v160, v103 :: v_dual_mul_f32 v104, v160, v104
	v_dual_mul_f32 v89, v160, v89 :: v_dual_mul_f32 v90, v160, v90
	v_dual_mul_f32 v91, v160, v91 :: v_dual_mul_f32 v92, v160, v92
	v_dual_mul_f32 v93, v160, v93 :: v_dual_mul_f32 v94, v160, v94
	v_dual_mul_f32 v95, v160, v95 :: v_dual_mul_f32 v96, v160, v96
	v_dual_mul_f32 v81, v160, v81 :: v_dual_mul_f32 v82, v160, v82
	v_dual_mul_f32 v83, v160, v83 :: v_dual_mul_f32 v84, v160, v84
	v_dual_mul_f32 v85, v160, v85 :: v_dual_mul_f32 v86, v160, v86
	v_dual_mul_f32 v87, v160, v87 :: v_dual_mul_f32 v88, v160, v88
	v_dual_mul_f32 v73, v160, v73 :: v_dual_mul_f32 v74, v160, v74
	v_dual_mul_f32 v75, v160, v75 :: v_dual_mul_f32 v76, v160, v76
	v_dual_mul_f32 v77, v160, v77 :: v_dual_mul_f32 v78, v160, v78
	v_dual_mul_f32 v79, v160, v79 :: v_dual_mul_f32 v80, v160, v80
	v_dual_mul_f32 v65, v160, v65 :: v_dual_mul_f32 v66, v160, v66
	v_dual_mul_f32 v67, v160, v67 :: v_dual_mul_f32 v68, v160, v68
	v_dual_mul_f32 v69, v160, v69 :: v_dual_mul_f32 v70, v160, v70
	v_dual_mul_f32 v71, v160, v71 :: v_dual_mul_f32 v72, v160, v72
.LBB0_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s25
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v140, v140
	v_cvt_f32_i32_e32 v141, v141
	v_cndmask_b32_e64 v137, 0xff7fffff, v137, s15
	v_cndmask_b32_e64 v138, 0xff7fffff, v138, s16
	v_cvt_f32_i32_e32 v142, v142
	v_cndmask_b32_e64 v139, 0xff7fffff, v139, s14
	v_cndmask_b32_e64 v140, 0xff7fffff, v140, s13
	v_cvt_f32_i32_e32 v143, v143
	v_max_num_f32_e32 v165, v137, v138
	v_cvt_f32_i32_e32 v144, v144
	v_cndmask_b32_e64 v141, 0xff7fffff, v141, s12
	v_cndmask_b32_e64 v142, 0xff7fffff, v142, s11
	v_cvt_f32_i32_e32 v129, v129
	v_max3_num_f32 v165, v165, v139, v140
	v_cvt_f32_i32_e32 v130, v130
	v_cndmask_b32_e64 v143, 0xff7fffff, v143, s10
	v_cndmask_b32_e64 v144, 0xff7fffff, v144, s9
	v_cvt_f32_i32_e32 v131, v131
	v_max3_num_f32 v165, v165, v141, v142
	v_cvt_f32_i32_e32 v132, v132
	v_cndmask_b32_e64 v166, 0xff7fffff, v129, s8
	v_cndmask_b32_e64 v167, 0xff7fffff, v130, s7
	v_cvt_f32_i32_e32 v130, v133
	v_max3_num_f32 v129, v165, v143, v144
	v_cvt_f32_i32_e32 v133, v134
	v_cndmask_b32_e64 v165, 0xff7fffff, v131, s6
	v_cndmask_b32_e64 v168, 0xff7fffff, v132, s5
	v_cvt_f32_i32_e32 v131, v135
	v_max3_num_f32 v129, v129, v166, v167
	v_cvt_f32_i32_e32 v132, v136
	v_cndmask_b32_e64 v173, 0xff7fffff, v130, s4
	v_cndmask_b32_e64 v174, 0xff7fffff, v133, s1
	v_cndmask_b32_e64 v175, 0xff7fffff, v131, s3
	v_max3_num_f32 v129, v129, v165, v168
	v_cndmask_b32_e64 v176, 0xff7fffff, v132, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_num_f32 v129, v129, v173, v174
	v_max3_num_f32 v129, v129, v175, v176
	ds_bpermute_b32 v130, v171, v129
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v130, v130, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v129, v129, v130
	v_dual_max_num_f32 v130, v210, v210 :: v_dual_fmaak_f32 v129, v129, v172, 0xc10ce979
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v177, v130, v129
	v_fma_f32 v129, v137, v172, -v177
	v_fma_f32 v130, v138, v172, -v177
	v_fma_f32 v131, v139, v172, -v177
	v_fma_f32 v132, v140, v172, -v177
	v_fma_f32 v133, v141, v172, -v177
	v_exp_f32_e32 v129, v129
	v_exp_f32_e32 v130, v130
	v_exp_f32_e32 v131, v131
	v_exp_f32_e32 v132, v132
	v_cndmask_b32_e64 v134, 0, v129, s15
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v135, 0, v130, s16
	v_cndmask_b32_e64 v137, 0, v131, s14
	v_fma_f32 v129, v142, v172, -v177
	v_exp_f32_e32 v130, v133
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v139, 0, v132, s13
	v_add_f32_e32 v131, v134, v135
	v_exp_f32_e32 v136, v129
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_add_f32_e32 v131, v137, v131
	v_cndmask_b32_e64 v129, 0, v130, s12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v131, v139, v131
	v_cndmask_b32_e64 v130, 0, v136, s11
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v136, v129, v131
	v_fma_f32 v133, v143, v172, -v177
	v_fma_f32 v132, v144, v172, -v177
	v_fma_f32 v138, v166, v172, -v177
	v_fma_f32 v140, v167, v172, -v177
	v_fma_f32 v142, v168, v172, -v177
	v_exp_f32_e32 v133, v133
	v_exp_f32_e32 v132, v132
	v_exp_f32_e32 v138, v138
	v_exp_f32_e32 v140, v140
	v_fma_f32 v144, v174, v172, -v177
	v_fma_f32 v166, v176, v172, -v177
	v_cndmask_b32_e64 v131, 0, v133, s10
	v_add_f32_e32 v133, v130, v136
	s_delay_alu instid0(TRANS32_DEP_3)
	v_cndmask_b32_e64 v132, 0, v132, s9
	v_fma_f32 v136, v165, v172, -v177
	v_fma_f32 v165, v175, v172, -v177
	v_exp_f32_e32 v144, v144
	v_add_f32_e32 v141, v131, v133
	v_cndmask_b32_e64 v133, 0, v138, s8
	v_exp_f32_e32 v143, v136
	v_cndmask_b32_e64 v136, 0, v140, s7
	v_exp_f32_e32 v165, v165
	v_add_f32_e32 v138, v132, v141
	v_exp_f32_e32 v141, v142
	v_fma_f32 v142, v173, v172, -v177
	v_exp_f32_e32 v166, v166
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v140, v133, v138
	v_cndmask_b32_e64 v138, 0, v143, s6
	v_exp_f32_e32 v142, v142
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_add_f32_e32 v143, v136, v140
	v_cndmask_b32_e64 v140, 0, v141, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v143, v138, v143
	v_cndmask_b32_e64 v141, 0, v142, s4
	v_cndmask_b32_e64 v142, 0, v144, s1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v143, v140, v143
	v_add_f32_e32 v144, v141, v143
	v_cndmask_b32_e64 v143, 0, v165, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v144, v142, v144
	v_add_f32_e32 v165, v143, v144
	v_cndmask_b32_e64 v144, 0, v166, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v166, v144, v165
	v_sub_f32_e32 v165, v210, v177
	ds_bpermute_b32 v167, v171, v166
	v_exp_f32_e32 v165, v165
	v_cmpx_neq_f32_e32 1.0, v165
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_dual_mul_f32 v57, v165, v57 :: v_dual_mul_f32 v58, v165, v58
	v_dual_mul_f32 v59, v165, v59 :: v_dual_mul_f32 v60, v165, v60
	v_dual_mul_f32 v61, v165, v61 :: v_dual_mul_f32 v62, v165, v62
	v_dual_mul_f32 v63, v165, v63 :: v_dual_mul_f32 v64, v165, v64
	v_dual_mul_f32 v49, v165, v49 :: v_dual_mul_f32 v50, v165, v50
	v_dual_mul_f32 v51, v165, v51 :: v_dual_mul_f32 v52, v165, v52
	v_dual_mul_f32 v53, v165, v53 :: v_dual_mul_f32 v54, v165, v54
	v_dual_mul_f32 v55, v165, v55 :: v_dual_mul_f32 v56, v165, v56
	v_dual_mul_f32 v41, v165, v41 :: v_dual_mul_f32 v42, v165, v42
	v_dual_mul_f32 v43, v165, v43 :: v_dual_mul_f32 v44, v165, v44
	v_dual_mul_f32 v45, v165, v45 :: v_dual_mul_f32 v46, v165, v46
	v_dual_mul_f32 v47, v165, v47 :: v_dual_mul_f32 v48, v165, v48
	v_dual_mul_f32 v33, v165, v33 :: v_dual_mul_f32 v34, v165, v34
	v_dual_mul_f32 v35, v165, v35 :: v_dual_mul_f32 v36, v165, v36
	v_dual_mul_f32 v37, v165, v37 :: v_dual_mul_f32 v38, v165, v38
	v_dual_mul_f32 v39, v165, v39 :: v_dual_mul_f32 v40, v165, v40
	v_dual_mul_f32 v25, v165, v25 :: v_dual_mul_f32 v26, v165, v26
	v_dual_mul_f32 v27, v165, v27 :: v_dual_mul_f32 v28, v165, v28
	v_dual_mul_f32 v29, v165, v29 :: v_dual_mul_f32 v30, v165, v30
	v_dual_mul_f32 v31, v165, v31 :: v_dual_mul_f32 v32, v165, v32
	v_dual_mul_f32 v17, v165, v17 :: v_dual_mul_f32 v18, v165, v18
	v_dual_mul_f32 v19, v165, v19 :: v_dual_mul_f32 v20, v165, v20
	v_dual_mul_f32 v21, v165, v21 :: v_dual_mul_f32 v22, v165, v22
	v_dual_mul_f32 v23, v165, v23 :: v_dual_mul_f32 v24, v165, v24
	v_dual_mul_f32 v9, v165, v9 :: v_dual_mul_f32 v10, v165, v10
	v_dual_mul_f32 v11, v165, v11 :: v_dual_mul_f32 v12, v165, v12
	v_dual_mul_f32 v13, v165, v13 :: v_dual_mul_f32 v14, v165, v14
	v_dual_mul_f32 v15, v165, v15 :: v_dual_mul_f32 v16, v165, v16
	v_dual_mul_f32 v1, v165, v1 :: v_dual_mul_f32 v2, v165, v2
	v_dual_mul_f32 v3, v165, v3 :: v_dual_mul_f32 v4, v165, v4
	v_dual_mul_f32 v5, v165, v5 :: v_dual_mul_f32 v6, v165, v6
	v_dual_mul_f32 v7, v165, v7 :: v_dual_mul_f32 v8, v165, v8
.LBB0_51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	v_dual_mov_b32 v168, 0 :: v_dual_add_f32 v189, v166, v167
	v_add_f32_e32 v190, v161, v162
	v_mul_u32_u24_e32 v161, 40, v195
	s_mulk_i32 s24, 0x1400
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_mov_b16_e64 v182.l, v168.l
	v_mov_b16_e64 v182.h, 0
	v_fmac_f32_e32 v189, v196, v165
	v_fmac_f32_e32 v190, v200, v160
	v_mov_b16_e64 v183.l, v182.l
	v_mov_b16_e64 v185.l, v182.l
	v_mov_b16_e64 v184.l, v182.l
	v_mov_b16_e64 v187.l, v182.l
	v_mov_b16_e64 v183.h, v182.h
	v_cvt_pk_fp8_f32 v183.l, v152, v163
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v152, s24, v161
	v_cvt_pk_fp8_f32 v185.l, v0, v145
	v_mov_b16_e64 v184.h, v182.h
	v_mov_b16_e64 v185.h, v182.h
	v_cvt_pk_fp8_f32 v184.l, v169, v157
	v_add_nc_u32_e32 v0, v152, v194
	v_mov_b16_e64 v186.l, v182.l
	v_mov_b16_e64 v186.h, v182.h
	v_cvt_pk_fp8_f32 v187.l, v134, v135
	v_cvt_pk_fp8_f32 v183.h, v164, v170
	v_dual_mov_b32 v200, v190 :: v_dual_add_nc_u32 v157, 0x3100, v0
	v_add_nc_u32_e32 v134, 0x3900, v0
	v_add_nc_u32_e32 v0, 0x4100, v0
	v_cvt_pk_fp8_f32 v184.h, v158, v159
	v_cvt_pk_fp8_f32 v185.h, v146, v147
	v_cvt_pk_fp8_f32 v186.l, v148, v149
	v_cvt_pk_fp8_f32 v186.h, v150, v151
	ds_load_2addr_b64 v[145:148], v157 offset0:64 offset1:66
	ds_load_2addr_b64 v[149:152], v157 offset0:144 offset1:146
	ds_load_2addr_b64 v[157:160], v157 offset0:224 offset1:226
	ds_load_2addr_b64 v[161:164], v134 offset0:48 offset1:50
	ds_load_2addr_b64 v[165:168], v134 offset0:128 offset1:130
	ds_load_2addr_b64 v[169:172], v134 offset0:208 offset1:210
	ds_load_2addr_b64 v[173:176], v0 offset0:32 offset1:34
	ds_load_2addr_b64 v[177:180], v0 offset0:112 offset1:114
	v_mov_b16_e64 v187.h, v182.h
	v_mov_b16_e64 v188.l, v182.l
	v_mov_b16_e64 v188.h, v182.h
	v_mov_b16_e64 v181.l, v182.l
	v_mov_b16_e64 v181.h, v182.h
	v_cvt_pk_fp8_f32 v187.h, v137, v139
	v_cvt_pk_fp8_f32 v188.l, v129, v130
	v_cvt_pk_fp8_f32 v188.h, v131, v132
	v_cvt_pk_fp8_f32 v181.l, v133, v136
	v_cvt_pk_fp8_f32 v181.h, v138, v140
	v_cvt_pk_fp8_f32 v182.l, v141, v142
	v_cvt_pk_fp8_f32 v182.h, v143, v144
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[145:146], v[185:186], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[145:146], v[187:188], v[57:64]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[149:150], v[185:186], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[149:150], v[187:188], v[49:56]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[157:158], v[185:186], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[157:158], v[187:188], v[41:48]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[161:162], v[185:186], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[161:162], v[187:188], v[33:40]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[165:166], v[185:186], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[165:166], v[187:188], v[25:32]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[169:170], v[185:186], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[169:170], v[187:188], v[17:24]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[173:174], v[185:186], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[173:174], v[187:188], v[9:16]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[177:178], v[185:186], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[177:178], v[187:188], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[147:148], v[183:184], v[121:128]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[147:148], v[181:182], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[113:120], v[151:152], v[183:184], v[113:120]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[151:152], v[181:182], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[105:112], v[159:160], v[183:184], v[105:112]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[159:160], v[181:182], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[97:104], v[163:164], v[183:184], v[97:104]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[163:164], v[181:182], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[89:96], v[167:168], v[183:184], v[89:96]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[167:168], v[181:182], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[81:88], v[171:172], v[183:184], v[81:88]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[171:172], v[181:182], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[73:80], v[175:176], v[183:184], v[73:80]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[175:176], v[181:182], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[65:72], v[179:180], v[183:184], v[65:72]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[179:180], v[181:182], v[1:8]
	v_mov_b32_e32 v196, v189
.LBB0_52:
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v157, 2, v194
	v_lshlrev_b32_e32 v0, 1, v194
	s_lshl_b64 s[0:1], s[22:23], 9
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[18:19], s[0:1]
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execnz .LBB0_55
; %bb.53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s3, s2
	s_cbranch_execnz .LBB0_56
.LBB0_54:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.LBB0_55:
	s_clause 0xa
	global_load_b128 v[133:136], v157, s[0:1]
	global_load_b128 v[129:132], v157, s[0:1] offset:16
	global_load_b128 v[141:144], v157, s[0:1] offset:64
	global_load_b128 v[137:140], v157, s[0:1] offset:80
	global_load_b128 v[158:161], v157, s[0:1] offset:128
	global_load_b128 v[162:165], v157, s[0:1] offset:144
	global_load_b128 v[166:169], v157, s[0:1] offset:192
	global_load_b128 v[170:173], v157, s[0:1] offset:208
	global_load_b128 v[174:177], v157, s[0:1] offset:256
	global_load_b128 v[178:181], v157, s[0:1] offset:272
	global_load_b128 v[182:185], v157, s[0:1] offset:320
	s_wait_loadcnt 0xb
	s_clause 0x2
	global_load_b128 v[149:152], v157, s[0:1] offset:336
	global_load_b128 v[186:189], v157, s[0:1] offset:400
	global_load_b128 v[190:193], v157, s[0:1] offset:384
	v_div_scale_f32 v145, null, v200, v200, 1.0
	v_div_scale_f32 v147, vcc_lo, 1.0, v200, 1.0
	v_lshlrev_b64_e32 v[155:156], 8, v[155:156]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v194, v145
	v_fma_f32 v146, -v145, v194, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v194, v146, v194
	v_mul_f32_e32 v195, v147, v194
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v146, -v145, v195, v147
	v_fmac_f32_e32 v195, v146, v194
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v197, -v145, v195, v147
	s_clause 0x1
	global_load_b128 v[145:148], v157, s[0:1] offset:464
	global_load_b128 v[201:204], v157, s[0:1] offset:448
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v194, v197, v194, v195
	v_add_co_u32 v155, vcc_lo, s20, v155
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v156, null, s21, v156, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v194, v194, v200, 1.0
	v_add_co_u32 v155, vcc_lo, v155, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v156, null, 0, v156, vcc_lo
	v_dual_mul_f32 v121, v194, v121 :: v_dual_mul_f32 v122, v194, v122
	v_dual_mul_f32 v123, v194, v123 :: v_dual_mul_f32 v126, v194, v126
	v_dual_mul_f32 v124, v194, v124 :: v_dual_mul_f32 v125, v194, v125
	v_dual_mul_f32 v128, v194, v128 :: v_dual_mul_f32 v103, v194, v103
	v_dual_mul_f32 v90, v194, v90 :: v_dual_mul_f32 v89, v194, v89
	v_dual_mul_f32 v92, v194, v92 :: v_dual_mul_f32 v113, v194, v113
	v_dual_mul_f32 v116, v194, v116 :: v_dual_mul_f32 v115, v194, v115
	v_dual_mul_f32 v118, v194, v118 :: v_dual_mul_f32 v117, v194, v117
	v_dual_mul_f32 v120, v194, v120 :: v_dual_mul_f32 v101, v194, v101
	v_dual_mul_f32 v104, v194, v104 :: v_dual_mul_f32 v91, v194, v91
	v_dual_mul_f32 v94, v194, v94 :: v_dual_mul_f32 v81, v194, v81
	v_dual_mul_f32 v119, v194, v119 :: v_dual_mul_f32 v106, v194, v106
	v_dual_mul_f32 v109, v194, v109 :: v_dual_mul_f32 v112, v194, v112
	v_dual_mul_f32 v93, v194, v93 :: v_dual_mul_f32 v96, v194, v96
	v_dual_mul_f32 v127, v194, v127 :: v_dual_mul_f32 v114, v194, v114
	v_dual_mul_f32 v105, v194, v105 :: v_dual_mul_f32 v108, v194, v108
	v_dual_mul_f32 v107, v194, v107 :: v_dual_mul_f32 v110, v194, v110
	v_dual_mul_f32 v111, v194, v111 :: v_dual_mul_f32 v98, v194, v98
	v_dual_mul_f32 v95, v194, v95 :: v_dual_mul_f32 v82, v194, v82
	v_mul_f32_e32 v83, v194, v83
	v_dual_mul_f32 v97, v194, v97 :: v_dual_mul_f32 v100, v194, v100
	v_dual_mul_f32 v99, v194, v99 :: v_dual_mul_f32 v102, v194, v102
	v_mul_f32_e32 v87, v194, v87
	v_dual_mul_f32 v73, v194, v73 :: v_dual_mul_f32 v74, v194, v74
	v_dual_mul_f32 v75, v194, v75 :: v_dual_mul_f32 v76, v194, v76
	v_dual_mul_f32 v77, v194, v77 :: v_dual_mul_f32 v78, v194, v78
	v_mul_f32_e32 v79, v194, v79
	v_mul_f32_e32 v71, v194, v71
	v_dual_mul_f32 v69, v194, v69 :: v_dual_mul_f32 v86, v194, v86
	v_dual_mul_f32 v84, v194, v84 :: v_dual_mul_f32 v85, v194, v85
	v_dual_mul_f32 v66, v194, v66 :: v_dual_mul_f32 v65, v194, v65
	v_dual_mul_f32 v68, v194, v68 :: v_dual_mul_f32 v67, v194, v67
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v70, v194, v70 :: v_dual_mul_f32 v121, v133, v121
	v_dual_mul_f32 v122, v134, v122 :: v_dual_mul_f32 v123, v135, v123
	v_mul_f32_e32 v124, v136, v124
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v128, v132, v128 :: v_dual_mul_f32 v113, v141, v113
	s_wait_loadcnt 0x8
	v_dual_mul_f32 v118, v138, v118 :: v_dual_mul_f32 v135, v172, v103
	s_wait_loadcnt 0x7
	v_mul_f32_e32 v136, v174, v89
	v_bfe_u32 v89, v121, 16, 1
	v_dual_mul_f32 v115, v143, v115 :: v_dual_mul_f32 v120, v140, v120
	v_dual_mul_f32 v117, v137, v117 :: v_dual_mul_f32 v106, v159, v106
	v_dual_mul_f32 v133, v170, v101 :: v_dual_mul_f32 v138, v176, v91
	s_wait_loadcnt 0x6
	v_dual_mul_f32 v137, v175, v90 :: v_dual_mul_f32 v140, v178, v93
	v_or_b32_e32 v90, 0x400000, v121
	v_bfe_u32 v91, v122, 16, 1
	v_bfe_u32 v103, v128, 16, 1
	v_add3_u32 v89, v89, v121, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v121, v121
	v_dual_mul_f32 v116, v144, v116 :: v_dual_mul_f32 v119, v139, v119
	v_dual_mul_f32 v108, v161, v108 :: v_dual_mul_f32 v109, v162, v109
	v_mul_f32_e32 v139, v177, v92
	v_or_b32_e32 v92, 0x400000, v122
	v_bfe_u32 v93, v123, 16, 1
	v_or_b32_e32 v144, 0x400000, v128
	v_bfe_u32 v162, v115, 16, 1
	v_add3_u32 v91, v91, v122, 0x7fff
	v_add3_u32 v103, v103, v128, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v121, v89, v90, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v122, v122
	v_dual_mul_f32 v125, v129, v125 :: v_dual_mul_f32 v126, v130, v126
	v_dual_mul_f32 v127, v131, v127 :: v_dual_mul_f32 v114, v142, v114
	v_dual_mul_f32 v105, v158, v105 :: v_dual_mul_f32 v110, v163, v110
	v_mul_f32_e32 v142, v180, v95
	v_bfe_u32 v95, v124, 16, 1
	v_or_b32_e32 v163, 0x400000, v115
	v_add3_u32 v93, v93, v123, 0x7fff
	v_add3_u32 v162, v162, v115, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v89, v91, v92, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v123, v123
	v_mul_f32_e32 v141, v179, v94
	v_or_b32_e32 v94, 0x400000, v123
	v_dual_mul_f32 v129, v166, v97 :: v_dual_mul_f32 v134, v171, v102
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v143, v181, v96 :: v_dual_mul_f32 v82, v183, v82
	v_or_b32_e32 v96, 0x400000, v124
	v_bfe_u32 v97, v125, 16, 1
	v_add3_u32 v95, v95, v124, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v122, v93, v94, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v124, v124
	v_dual_mul_f32 v111, v164, v111 :: v_dual_mul_f32 v132, v169, v100
	v_dual_mul_f32 v130, v167, v98 :: v_dual_mul_f32 v131, v168, v99
	v_dual_mul_f32 v104, v173, v104 :: v_dual_mul_f32 v81, v182, v81
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v90, v95, v96, vcc_lo
	v_or_b32_e32 v98, 0x400000, v125
	v_bfe_u32 v99, v126, 16, 1
	v_bfe_u32 v164, v116, 16, 1
	v_add3_u32 v97, v97, v125, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v125, v125
	v_dual_mul_f32 v107, v160, v107 :: v_dual_mul_f32 v112, v165, v112
	v_or_b32_e32 v100, 0x400000, v126
	v_bfe_u32 v101, v127, 16, 1
	v_or_b32_e32 v165, 0x400000, v116
	v_bfe_u32 v170, v119, 16, 1
	v_add3_u32 v99, v99, v126, 0x7fff
	v_add3_u32 v164, v164, v116, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v123, v97, v98, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v126, v126
	v_or_b32_e32 v102, 0x400000, v127
	v_or_b32_e32 v171, 0x400000, v119
	v_add3_u32 v101, v101, v127, 0x7fff
	v_add3_u32 v170, v170, v119, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v91, v99, v100, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v127, v127
	v_mul_f32_e32 v83, v184, v83
	v_bfe_u32 v158, v113, 16, 1
	v_or_b32_e32 v159, 0x400000, v113
	v_bfe_u32 v160, v114, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v124, v101, v102, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v128, v128
	v_bfe_u32 v172, v120, 16, 1
	v_add3_u32 v158, v158, v113, 0x7fff
	v_or_b32_e32 v161, 0x400000, v114
	v_or_b32_e32 v173, 0x400000, v120
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v92, v103, v144, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v113, v113
	v_bfe_u32 v178, v107, 16, 1
	v_add3_u32 v160, v160, v114, 0x7fff
	v_add3_u32 v172, v172, v120, 0x7fff
	v_or_b32_e32 v179, 0x400000, v107
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v113, v158, v159, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v114, v114
	v_add3_u32 v178, v178, v107, 0x7fff
	v_bfe_u32 v166, v117, 16, 1
	v_or_b32_e32 v167, 0x400000, v117
	v_bfe_u32 v168, v118, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v93, v160, v161, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v115, v115
	v_bfe_u32 v180, v108, 16, 1
	v_add3_u32 v166, v166, v117, 0x7fff
	v_or_b32_e32 v169, 0x400000, v118
	v_or_b32_e32 v181, 0x400000, v108
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v114, v162, v163, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v116, v116
	v_bfe_u32 v197, v111, 16, 1
	v_add3_u32 v168, v168, v118, 0x7fff
	v_add3_u32 v180, v180, v108, 0x7fff
	v_or_b32_e32 v198, 0x400000, v111
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v94, v164, v165, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v117, v117
	v_add3_u32 v197, v197, v111, 0x7fff
	v_bfe_u32 v174, v105, 16, 1
	v_or_b32_e32 v175, 0x400000, v105
	v_bfe_u32 v176, v106, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v115, v166, v167, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v118, v118
	v_bfe_u32 v199, v112, 16, 1
	v_add3_u32 v174, v174, v105, 0x7fff
	v_or_b32_e32 v177, 0x400000, v106
	v_or_b32_e32 v200, 0x400000, v112
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v95, v168, v169, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v119, v119
	v_bfe_u32 v209, v131, 16, 1
	v_add3_u32 v176, v176, v106, 0x7fff
	v_add3_u32 v199, v199, v112, 0x7fff
	v_or_b32_e32 v210, 0x400000, v131
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v116, v170, v171, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v120, v120
	v_add3_u32 v209, v209, v131, 0x7fff
	v_bfe_u32 v182, v109, 16, 1
	v_or_b32_e32 v183, 0x400000, v109
	v_bfe_u32 v184, v110, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v96, v172, v173, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v105, v105
	v_bfe_u32 v211, v132, 16, 1
	v_add3_u32 v182, v182, v109, 0x7fff
	v_or_b32_e32 v195, 0x400000, v110
	v_or_b32_e32 v212, 0x400000, v132
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v117, v174, v175, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v106, v106
	v_bfe_u32 v217, v135, 16, 1
	v_add3_u32 v184, v184, v110, 0x7fff
	v_add3_u32 v211, v211, v132, 0x7fff
	v_or_b32_e32 v218, 0x400000, v135
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v97, v176, v177, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v107, v107
	v_add3_u32 v217, v217, v135, 0x7fff
	v_bfe_u32 v205, v129, 16, 1
	v_or_b32_e32 v206, 0x400000, v129
	v_bfe_u32 v207, v130, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v118, v178, v179, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v108, v108
	v_bfe_u32 v219, v104, 16, 1
	v_add3_u32 v205, v205, v129, 0x7fff
	v_or_b32_e32 v208, 0x400000, v130
	v_or_b32_e32 v220, 0x400000, v104
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v98, v180, v181, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v109, v109
	v_bfe_u32 v225, v138, 16, 1
	v_add3_u32 v207, v207, v130, 0x7fff
	v_add3_u32 v219, v219, v104, 0x7fff
	v_or_b32_e32 v226, 0x400000, v138
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v109, v182, v183, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v110, v110
	v_add3_u32 v225, v225, v138, 0x7fff
	v_bfe_u32 v213, v133, 16, 1
	v_or_b32_e32 v214, 0x400000, v133
	v_bfe_u32 v215, v134, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v99, v184, v195, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v111, v111
	v_bfe_u32 v227, v139, 16, 1
	v_add3_u32 v213, v213, v133, 0x7fff
	v_or_b32_e32 v216, 0x400000, v134
	v_or_b32_e32 v228, 0x400000, v139
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v110, v197, v198, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v112, v112
	v_bfe_u32 v233, v142, 16, 1
	v_add3_u32 v215, v215, v134, 0x7fff
	v_add3_u32 v227, v227, v139, 0x7fff
	v_or_b32_e32 v234, 0x400000, v142
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v100, v199, v200, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v129, v129
	v_add3_u32 v233, v233, v142, 0x7fff
	v_bfe_u32 v221, v136, 16, 1
	v_or_b32_e32 v222, 0x400000, v136
	v_bfe_u32 v223, v137, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v111, v205, v206, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v130, v130
	v_bfe_u32 v235, v143, 16, 1
	v_add3_u32 v221, v221, v136, 0x7fff
	v_or_b32_e32 v224, 0x400000, v137
	v_or_b32_e32 v236, 0x400000, v143
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v101, v207, v208, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v131, v131
	v_add3_u32 v223, v223, v137, 0x7fff
	v_add3_u32 v235, v235, v143, 0x7fff
	v_bfe_u32 v229, v140, 16, 1
	v_or_b32_e32 v230, 0x400000, v140
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v112, v209, v210, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v132, v132
	v_bfe_u32 v231, v141, 16, 1
	v_add3_u32 v229, v229, v140, 0x7fff
	v_or_b32_e32 v232, 0x400000, v141
	v_bfe_u32 v237, v81, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v102, v211, v212, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v133, v133
	v_add3_u32 v231, v231, v141, 0x7fff
	v_mov_b16_e32 v92.l, v124.h
	v_mov_b16_e32 v91.l, v123.h
	v_mov_b16_e32 v90.l, v122.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v119, v213, v214, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v134, v134
	v_mov_b16_e32 v89.l, v121.h
	v_mov_b16_e32 v96.l, v116.h
	v_mov_b16_e32 v95.l, v115.h
	v_mov_b16_e32 v94.l, v114.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v103, v215, v216, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v135, v135
	v_mov_b16_e32 v93.l, v113.h
	s_clause 0x1
	global_store_b128 v[155:156], v[89:92], off
	global_store_b128 v[155:156], v[93:96], off offset:32
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v120, v217, v218, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v104, v104
	v_add3_u32 v89, v237, v81, 0x7fff
	v_or_b32_e32 v90, 0x400000, v81
	v_bfe_u32 v91, v82, 16, 1
	v_bfe_u32 v92, v83, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v104, v219, v220, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v136, v136
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v84, v185, v84 :: v_dual_mul_f32 v73, v190, v73
	v_mul_f32_e32 v76, v193, v76
	v_mul_f32_e32 v74, v191, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v125, v221, v222, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v137, v137
	v_bfe_u32 v93, v84, 16, 1
	v_mul_f32_e32 v75, v192, v75
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v65, v201, v65 :: v_dual_mul_f32 v66, v202, v66
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v105, v223, v224, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v138, v138
	v_dual_mul_f32 v68, v204, v68 :: v_dual_mul_f32 v67, v203, v67
	v_mov_b16_e32 v100.l, v110.h
	v_mov_b16_e32 v99.l, v109.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v126, v225, v226, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v139, v139
	v_mov_b16_e32 v98.l, v118.h
	v_mov_b16_e32 v97.l, v117.h
	v_mov_b16_e32 v104.l, v120.h
	v_mov_b16_e32 v102.l, v112.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v106, v227, v228, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v140, v140
	v_mov_b16_e32 v103.l, v119.h
	v_mov_b16_e32 v101.l, v111.h
	v_mov_b16_e32 v106.l, v126.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v127, v229, v230, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v141, v141
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v107, v231, v232, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v142, v142
	v_mov_b16_e32 v107.l, v127.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v128, v233, v234, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v143, v143
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v108, v235, v236, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v81, v81
	v_add3_u32 v81, v91, v82, 0x7fff
	v_add3_u32 v91, v92, v83, 0x7fff
	v_or_b32_e32 v92, 0x400000, v83
	v_mov_b16_e64 v108.l, v128.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v89, v89, v90, vcc_lo
	v_or_b32_e32 v90, 0x400000, v82
	v_cmp_u_f32_e32 vcc_lo, v82, v82
	v_add3_u32 v82, v93, v84, 0x7fff
	v_mov_b16_e32 v105.l, v125.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v81, v81, v90, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v83, v83
	v_mul_f32_e32 v83, v149, v85
	v_or_b32_e32 v85, 0x400000, v84
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v90, v91, v92, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v84, v84
	v_mov_b16_e32 v81.l, v89.h
	v_bfe_u32 v89, v76, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v82, v82, v85, vcc_lo
	v_mul_f32_e32 v85, v150, v86
	v_mul_f32_e32 v86, v194, v88
	v_bfe_u32 v91, v83, 16, 1
	v_or_b32_e32 v88, 0x400000, v83
	v_cmp_u_f32_e32 vcc_lo, v83, v83
	v_mov_b16_e32 v82.l, v90.h
	v_mul_f32_e32 v86, v152, v86
	v_mul_f32_e32 v84, v151, v87
	v_add3_u32 v87, v91, v83, 0x7fff
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_bfe_u32 v91, v84, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v87, v87, v88, vcc_lo
	v_or_b32_e32 v88, 0x400000, v84
	v_cmp_u_f32_e32 vcc_lo, v84, v84
	v_add3_u32 v83, v91, v84, 0x7fff
	v_bfe_u32 v91, v86, 16, 1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v88, v83, v88, vcc_lo
	v_bfe_u32 v92, v85, 16, 1
	v_or_b32_e32 v93, 0x400000, v85
	v_cmp_u_f32_e32 vcc_lo, v85, v85
	v_add3_u32 v84, v91, v86, 0x7fff
	v_or_b32_e32 v91, 0x400000, v86
	v_add3_u32 v92, v92, v85, 0x7fff
	v_bfe_u32 v85, v73, 16, 1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v83, v92, v93, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v86, v86
	v_add3_u32 v85, v85, v73, 0x7fff
	v_or_b32_e32 v86, 0x400000, v73
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v84, v84, v91, vcc_lo
	v_mov_b16_e32 v83.l, v87.h
	v_bfe_u32 v87, v74, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v73, v73
	v_mov_b16_e32 v84.l, v88.h
	v_bfe_u32 v88, v75, 16, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v73, v87, v74, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v85, v85, v86, vcc_lo
	v_or_b32_e32 v86, 0x400000, v74
	v_cmp_u_f32_e32 vcc_lo, v74, v74
	v_add3_u32 v87, v88, v75, 0x7fff
	v_or_b32_e32 v88, 0x400000, v75
	v_add3_u32 v74, v89, v76, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v73, v73, v86, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v75, v75
	v_mul_f32_e32 v75, v186, v77
	v_or_b32_e32 v77, 0x400000, v76
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v86, v87, v88, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v76, v76
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v74, v74, v77 :: v_dual_mul_f32 v77, v187, v78
	v_mul_f32_e32 v78, v194, v80
	v_mul_f32_e32 v76, v188, v79
	v_mov_b16_e32 v73.l, v85.h
	v_bfe_u32 v85, v68, 16, 1
	v_bfe_u32 v88, v77, 16, 1
	v_mul_f32_e32 v78, v189, v78
	v_bfe_u32 v87, v75, 16, 1
	v_or_b32_e32 v80, 0x400000, v75
	v_cmp_u_f32_e32 vcc_lo, v75, v75
	v_add3_u32 v88, v88, v77, 0x7fff
	v_or_b32_e32 v89, 0x400000, v77
	v_add3_u32 v79, v87, v75, 0x7fff
	v_bfe_u32 v87, v76, 16, 1
	v_mov_b16_e32 v74.l, v86.h
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v79, v79, v80, vcc_lo
	v_add3_u32 v75, v87, v76, 0x7fff
	v_or_b32_e32 v80, 0x400000, v76
	v_cmp_u_f32_e32 vcc_lo, v76, v76
	v_bfe_u32 v87, v78, 16, 1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v80, v75, v80, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v77, v77
	v_add3_u32 v76, v87, v78, 0x7fff
	v_or_b32_e32 v87, 0x400000, v78
	v_bfe_u32 v77, v65, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v75, v88, v89, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v78, v78
	v_mov_b16_e32 v75.l, v79.h
	v_add3_u32 v77, v77, v65, 0x7fff
	v_or_b32_e32 v78, 0x400000, v65
	v_bfe_u32 v79, v66, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v76, v76, v87, vcc_lo
	v_mov_b16_e32 v76.l, v80.h
	v_bfe_u32 v80, v67, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v65, v65
	v_add3_u32 v65, v79, v66, 0x7fff
	s_delay_alu instid0(VALU_DEP_3)
	v_add3_u32 v79, v80, v67, 0x7fff
	v_or_b32_e32 v80, 0x400000, v67
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v77, v77, v78, vcc_lo
	v_or_b32_e32 v78, 0x400000, v66
	v_cmp_u_f32_e32 vcc_lo, v66, v66
	v_add3_u32 v66, v85, v68, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v65, v65, v78, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v67, v67
	v_or_b32_e32 v67, 0x400000, v68
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v78, v79, v80, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v68, v68
	v_mov_b16_e32 v65.l, v77.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v66, v66, v67, vcc_lo
	v_mul_f32_e32 v67, v194, v72
	v_mul_f32_e32 v68, v146, v70
	v_dual_mul_f32 v70, v147, v71 :: v_dual_mul_f32 v69, v145, v69
	v_mov_b16_e32 v66.l, v78.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_bfe_u32 v80, v70, 16, 1
	v_bfe_u32 v79, v69, 16, 1
	v_or_b32_e32 v72, 0x400000, v69
	v_cmp_u_f32_e32 vcc_lo, v69, v69
	s_delay_alu instid0(VALU_DEP_3)
	v_add3_u32 v71, v79, v69, 0x7fff
	v_mul_f32_e32 v79, v148, v67
	v_bfe_u32 v67, v68, 16, 1
	v_or_b32_e32 v85, 0x400000, v68
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v69, v71, v72, vcc_lo
	v_add3_u32 v72, v80, v70, 0x7fff
	v_or_b32_e32 v80, 0x400000, v70
	v_cmp_u_f32_e32 vcc_lo, v70, v70
	v_bfe_u32 v71, v79, 16, 1
	v_add3_u32 v67, v67, v68, 0x7fff
	v_or_b32_e32 v86, 0x400000, v79
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v70, v72, v80, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v68, v68
	v_add3_u32 v71, v71, v79, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v67, v67, v85, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v79, v79
	v_mov_b16_e32 v67.l, v69.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v68, v71, v86, vcc_lo
	v_mov_b16_e32 v68.l, v70.h
	s_clause 0x5
	global_store_b128 v[155:156], v[97:100], off offset:64
	global_store_b128 v[155:156], v[101:104], off offset:96
	global_store_b128 v[155:156], v[105:108], off offset:128
	global_store_b128 v[155:156], v[81:84], off offset:160
	global_store_b128 v[155:156], v[73:76], off offset:192
	global_store_b128 v[155:156], v[65:68], off offset:224
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB0_54
.LBB0_56:
	s_clause 0xd
	global_load_b128 v[69:72], v157, s[0:1]
	global_load_b128 v[65:68], v157, s[0:1] offset:16
	global_load_b128 v[77:80], v157, s[0:1] offset:64
	global_load_b128 v[73:76], v157, s[0:1] offset:80
	global_load_b128 v[91:94], v157, s[0:1] offset:128
	global_load_b128 v[95:98], v157, s[0:1] offset:144
	global_load_b128 v[99:102], v157, s[0:1] offset:192
	global_load_b128 v[103:106], v157, s[0:1] offset:208
	global_load_b128 v[107:110], v157, s[0:1] offset:256
	global_load_b128 v[111:114], v157, s[0:1] offset:272
	global_load_b128 v[115:118], v157, s[0:1] offset:320
	global_load_b128 v[85:88], v157, s[0:1] offset:336
	global_load_b128 v[119:122], v157, s[0:1] offset:400
	global_load_b128 v[123:126], v157, s[0:1] offset:384
	v_div_scale_f32 v81, null, v196, v196, 1.0
	v_div_scale_f32 v83, vcc_lo, 1.0, v196, 1.0
	v_lshlrev_b64_e32 v[89:90], 8, v[153:154]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v131, v81
	v_fma_f32 v82, -v81, v131, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v131, v82, v131
	v_mul_f32_e32 v132, v83, v131
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v82, -v81, v132, v83
	v_fmac_f32_e32 v132, v82, v131
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v133, -v81, v132, v83
	s_clause 0x1
	global_load_b128 v[81:84], v157, s[0:1] offset:464
	global_load_b128 v[127:130], v157, s[0:1] offset:448
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v131, v133, v131, v132
	v_add_co_u32 v89, vcc_lo, s20, v89
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v90, null, s21, v90, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v131, v131, v196, 1.0
	v_add_co_u32 v89, vcc_lo, v89, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v90, null, 0, v90, vcc_lo
	v_dual_mul_f32 v0, v131, v57 :: v_dual_mul_f32 v57, v131, v58
	v_dual_mul_f32 v58, v131, v59 :: v_dual_mul_f32 v59, v131, v60
	v_dual_mul_f32 v60, v131, v61 :: v_dual_mul_f32 v61, v131, v62
	v_dual_mul_f32 v38, v131, v38 :: v_dual_mul_f32 v39, v131, v39
	v_dual_mul_f32 v40, v131, v40 :: v_dual_mul_f32 v25, v131, v25
	v_dual_mul_f32 v62, v131, v63 :: v_dual_mul_f32 v49, v131, v49
	v_mul_f32_e32 v63, v131, v64
	v_dual_mul_f32 v52, v131, v52 :: v_dual_mul_f32 v53, v131, v53
	v_dual_mul_f32 v34, v131, v34 :: v_dual_mul_f32 v35, v131, v35
	v_dual_mul_f32 v26, v131, v26 :: v_dual_mul_f32 v27, v131, v27
	v_dual_mul_f32 v54, v131, v54 :: v_dual_mul_f32 v55, v131, v55
	v_dual_mul_f32 v36, v131, v36 :: v_dual_mul_f32 v37, v131, v37
	v_dual_mul_f32 v28, v131, v28 :: v_dual_mul_f32 v29, v131, v29
	v_dual_mul_f32 v56, v131, v56 :: v_dual_mul_f32 v41, v131, v41
	v_dual_mul_f32 v30, v131, v30 :: v_dual_mul_f32 v31, v131, v31
	v_dual_mul_f32 v18, v131, v18 :: v_dual_mul_f32 v19, v131, v19
	v_dual_mul_f32 v50, v131, v50 :: v_dual_mul_f32 v51, v131, v51
	v_dual_mul_f32 v48, v131, v48 :: v_dual_mul_f32 v33, v131, v33
	v_dual_mul_f32 v32, v131, v32 :: v_dual_mul_f32 v17, v131, v17
	v_dual_mul_f32 v42, v131, v42 :: v_dual_mul_f32 v43, v131, v43
	v_dual_mul_f32 v44, v131, v44 :: v_dual_mul_f32 v45, v131, v45
	v_dual_mul_f32 v46, v131, v46 :: v_dual_mul_f32 v47, v131, v47
	v_dual_mul_f32 v20, v131, v20 :: v_dual_mul_f32 v21, v131, v21
	v_dual_mul_f32 v23, v131, v23 :: v_dual_mul_f32 v22, v131, v22
	v_dual_mul_f32 v15, v131, v15 :: v_dual_mul_f32 v14, v131, v14
	v_dual_mul_f32 v10, v131, v10 :: v_dual_mul_f32 v11, v131, v11
	v_dual_mul_f32 v12, v131, v12 :: v_dual_mul_f32 v13, v131, v13
	v_mul_f32_e32 v4, v131, v4
	v_mul_f32_e32 v2, v131, v2
	v_dual_mul_f32 v6, v131, v6 :: v_dual_mul_f32 v9, v131, v9
	v_mul_f32_e32 v1, v131, v1
	v_mul_f32_e32 v3, v131, v3
	v_mul_f32_e32 v5, v131, v5
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v7, v131, v7 :: v_dual_mul_f32 v0, v69, v0
	v_dual_mul_f32 v57, v70, v57 :: v_dual_mul_f32 v58, v71, v58
	s_wait_loadcnt 0xe
	v_dual_mul_f32 v59, v72, v59 :: v_dual_mul_f32 v60, v65, v60
	v_mul_f32_e32 v61, v66, v61
	s_wait_loadcnt 0xc
	v_dual_mul_f32 v53, v73, v53 :: v_dual_mul_f32 v54, v74, v54
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v70, v105, v39 :: v_dual_mul_f32 v71, v107, v25
	v_bfe_u32 v25, v0, 16, 1
	v_mul_f32_e32 v66, v101, v35
	v_dual_mul_f32 v40, v106, v40 :: v_dual_mul_f32 v73, v109, v27
	v_mul_f32_e32 v72, v108, v26
	v_or_b32_e32 v26, 0x400000, v0
	v_bfe_u32 v27, v57, 16, 1
	v_bfe_u32 v35, v61, 16, 1
	v_add3_u32 v25, v25, v0, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_dual_mul_f32 v62, v67, v62 :: v_dual_mul_f32 v63, v68, v63
	v_dual_mul_f32 v55, v75, v55 :: v_dual_mul_f32 v56, v76, v56
	v_mul_f32_e32 v67, v102, v36
	s_wait_loadcnt 0x6
	v_dual_mul_f32 v74, v110, v28 :: v_dual_mul_f32 v75, v111, v29
	v_or_b32_e32 v28, 0x400000, v57
	v_bfe_u32 v29, v58, 16, 1
	v_or_b32_e32 v36, 0x400000, v61
	v_add3_u32 v27, v27, v57, 0x7fff
	v_add3_u32 v35, v35, v61, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v25, v26, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v57, v57
	v_dual_mul_f32 v49, v77, v49 :: v_dual_mul_f32 v50, v78, v50
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v76, v112, v30 :: v_dual_mul_f32 v17, v115, v17
	v_dual_mul_f32 v77, v113, v31 :: v_dual_mul_f32 v18, v116, v18
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v25, v27, v28, vcc_lo
	v_or_b32_e32 v30, 0x400000, v58
	v_bfe_u32 v31, v59, 16, 1
	v_add3_u32 v29, v29, v58, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v58, v58
	v_dual_mul_f32 v64, v99, v33 :: v_dual_mul_f32 v65, v100, v34
	v_dual_mul_f32 v78, v114, v32 :: v_dual_mul_f32 v19, v117, v19
	v_or_b32_e32 v32, 0x400000, v59
	v_bfe_u32 v33, v60, 16, 1
	v_add3_u32 v31, v31, v59, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, v29, v30, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v59, v59
	v_dual_mul_f32 v51, v79, v51 :: v_dual_mul_f32 v52, v80, v52
	v_or_b32_e32 v34, 0x400000, v60
	v_bfe_u32 v80, v49, 16, 1
	v_add3_u32 v33, v33, v60, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v26, v31, v32, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v60, v60
	v_dual_mul_f32 v41, v91, v41 :: v_dual_mul_f32 v42, v92, v42
	v_dual_mul_f32 v68, v103, v37 :: v_dual_mul_f32 v69, v104, v38
	v_bfe_u32 v37, v62, 16, 1
	v_or_b32_e32 v91, 0x400000, v49
	v_add3_u32 v80, v80, v49, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v58, v33, v34, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v61, v61
	v_or_b32_e32 v38, 0x400000, v62
	v_bfe_u32 v39, v63, 16, 1
	v_add3_u32 v37, v37, v62, 0x7fff
	v_or_b32_e32 v79, 0x400000, v63
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v27, v35, v36, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v62, v62
	v_add3_u32 v39, v39, v63, 0x7fff
	v_bfe_u32 v92, v50, 16, 1
	v_dual_mul_f32 v43, v93, v43 :: v_dual_mul_f32 v44, v94, v44
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v59, v37, v38, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v63, v63
	v_or_b32_e32 v93, 0x400000, v50
	v_bfe_u32 v94, v51, 16, 1
	v_add3_u32 v92, v92, v50, 0x7fff
	v_dual_mul_f32 v45, v95, v45 :: v_dual_mul_f32 v46, v96, v46
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v28, v39, v79, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v49, v49
	v_or_b32_e32 v95, 0x400000, v51
	v_bfe_u32 v96, v52, 16, 1
	v_add3_u32 v94, v94, v51, 0x7fff
	v_dual_mul_f32 v47, v97, v47 :: v_dual_mul_f32 v48, v98, v48
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, v80, v91, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v50, v50
	v_or_b32_e32 v97, 0x400000, v52
	v_bfe_u32 v98, v53, 16, 1
	v_add3_u32 v96, v96, v52, 0x7fff
	v_or_b32_e32 v99, 0x400000, v53
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v29, v92, v93, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v51, v51
	v_bfe_u32 v100, v54, 16, 1
	v_add3_u32 v98, v98, v53, 0x7fff
	v_or_b32_e32 v101, 0x400000, v54
	v_bfe_u32 v102, v55, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v50, v94, v95, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v52, v52
	v_add3_u32 v100, v100, v54, 0x7fff
	v_or_b32_e32 v103, 0x400000, v55
	v_bfe_u32 v104, v56, 16, 1
	v_add3_u32 v102, v102, v55, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v30, v96, v97, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v53, v53
	v_or_b32_e32 v105, 0x400000, v56
	v_bfe_u32 v106, v41, 16, 1
	v_add3_u32 v104, v104, v56, 0x7fff
	v_or_b32_e32 v107, 0x400000, v41
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v51, v98, v99, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v54, v54
	v_bfe_u32 v108, v42, 16, 1
	v_add3_u32 v106, v106, v41, 0x7fff
	v_or_b32_e32 v109, 0x400000, v42
	v_bfe_u32 v110, v43, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v31, v100, v101, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v55, v55
	v_add3_u32 v108, v108, v42, 0x7fff
	v_or_b32_e32 v111, 0x400000, v43
	v_bfe_u32 v112, v44, 16, 1
	v_add3_u32 v110, v110, v43, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v52, v102, v103, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v56, v56
	v_or_b32_e32 v113, 0x400000, v44
	v_bfe_u32 v114, v45, 16, 1
	v_add3_u32 v112, v112, v44, 0x7fff
	v_or_b32_e32 v115, 0x400000, v45
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v32, v104, v105, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v41, v41
	v_bfe_u32 v116, v46, 16, 1
	v_add3_u32 v114, v114, v45, 0x7fff
	v_or_b32_e32 v117, 0x400000, v46
	v_bfe_u32 v132, v47, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v53, v106, v107, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v42, v42
	v_bfe_u32 v134, v48, 16, 1
	v_add3_u32 v116, v116, v46, 0x7fff
	v_or_b32_e32 v133, 0x400000, v47
	v_or_b32_e32 v135, 0x400000, v48
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v33, v108, v109, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v43, v43
	v_bfe_u32 v138, v65, 16, 1
	v_add3_u32 v132, v132, v47, 0x7fff
	v_add3_u32 v134, v134, v48, 0x7fff
	v_bfe_u32 v136, v64, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v54, v110, v111, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v44, v44
	v_or_b32_e32 v139, 0x400000, v65
	v_add3_u32 v138, v138, v65, 0x7fff
	v_or_b32_e32 v137, 0x400000, v64
	v_bfe_u32 v142, v67, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v34, v112, v113, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v45, v45
	v_add3_u32 v136, v136, v64, 0x7fff
	v_bfe_u32 v140, v66, 16, 1
	v_or_b32_e32 v143, 0x400000, v67
	v_add3_u32 v142, v142, v67, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, v114, v115, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v46, v46
	v_or_b32_e32 v141, 0x400000, v66
	v_bfe_u32 v146, v69, 16, 1
	v_add3_u32 v140, v140, v66, 0x7fff
	v_bfe_u32 v144, v68, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v35, v116, v117, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v47, v47
	v_or_b32_e32 v147, 0x400000, v69
	v_add3_u32 v146, v146, v69, 0x7fff
	v_or_b32_e32 v145, 0x400000, v68
	v_bfe_u32 v150, v40, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v46, v132, v133, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v48, v48
	v_add3_u32 v144, v144, v68, 0x7fff
	v_bfe_u32 v148, v70, 16, 1
	v_or_b32_e32 v151, 0x400000, v40
	v_add3_u32 v150, v150, v40, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v36, v134, v135, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v64, v64
	v_or_b32_e32 v149, 0x400000, v70
	v_bfe_u32 v154, v72, 16, 1
	v_add3_u32 v148, v148, v70, 0x7fff
	v_bfe_u32 v152, v71, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v47, v136, v137, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v65, v65
	v_or_b32_e32 v155, 0x400000, v72
	v_add3_u32 v154, v154, v72, 0x7fff
	v_or_b32_e32 v153, 0x400000, v71
	v_bfe_u32 v162, v76, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v37, v138, v139, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v66, v66
	v_add3_u32 v152, v152, v71, 0x7fff
	v_bfe_u32 v156, v73, 16, 1
	v_or_b32_e32 v163, 0x400000, v76
	v_add3_u32 v162, v162, v76, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v48, v140, v141, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v67, v67
	v_or_b32_e32 v157, 0x400000, v73
	v_bfe_u32 v158, v74, 16, 1
	v_add3_u32 v156, v156, v73, 0x7fff
	v_or_b32_e32 v159, 0x400000, v74
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v38, v142, v143, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v68, v68
	v_bfe_u32 v160, v75, 16, 1
	v_add3_u32 v158, v158, v74, 0x7fff
	v_or_b32_e32 v161, 0x400000, v75
	v_bfe_u32 v164, v77, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v55, v144, v145, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v69, v69
	v_add3_u32 v160, v160, v75, 0x7fff
	v_or_b32_e32 v165, 0x400000, v77
	v_bfe_u32 v166, v78, 16, 1
	v_add3_u32 v164, v164, v77, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v39, v146, v147, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v70, v70
	v_or_b32_e32 v167, 0x400000, v78
	v_bfe_u32 v168, v17, 16, 1
	v_add3_u32 v166, v166, v78, 0x7fff
	v_mov_b16_e32 v28.l, v59.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v56, v148, v149, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v40, v40
	v_mov_b16_e32 v27.l, v58.h
	v_mov_b16_e32 v26.l, v57.h
	v_mov_b16_e32 v25.l, v0.h
	v_mov_b16_e32 v32.l, v52.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v40, v150, v151, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v71, v71
	v_mov_b16_e32 v31.l, v51.h
	v_mov_b16_e32 v30.l, v50.h
	v_mov_b16_e32 v29.l, v49.h
	s_clause 0x1
	global_store_b128 v[89:90], v[25:28], off
	global_store_b128 v[89:90], v[29:32], off offset:32
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v60, v152, v153, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v72, v72
	v_add3_u32 v0, v168, v17, 0x7fff
	v_or_b32_e32 v25, 0x400000, v17
	v_bfe_u32 v26, v18, 16, 1
	v_bfe_u32 v27, v19, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v41, v154, v155, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v73, v73
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v20, v118, v20 :: v_dual_mul_f32 v9, v123, v9
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v1, v127, v1
	v_mov_b16_e32 v36.l, v46.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v61, v156, v157, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v74, v74
	v_mov_b16_e32 v35.l, v45.h
	v_mov_b16_e32 v34.l, v54.h
	v_mov_b16_e32 v33.l, v53.h
	v_mov_b16_e32 v40.l, v56.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v42, v158, v159, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v75, v75
	v_mov_b16_e32 v39.l, v55.h
	v_mov_b16_e32 v38.l, v48.h
	v_mov_b16_e32 v37.l, v47.h
	v_mov_b16_e32 v42.l, v61.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v62, v160, v161, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v76, v76
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, v162, v163, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v77, v77
	v_mov_b16_e32 v41.l, v60.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v63, v164, v165, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v78, v78
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v44, v166, v167, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v17, v17
	v_add3_u32 v17, v26, v18, 0x7fff
	v_add3_u32 v26, v27, v19, 0x7fff
	v_or_b32_e32 v27, 0x400000, v19
	v_mov_b16_e32 v44.l, v63.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v0, v25, vcc_lo
	v_or_b32_e32 v25, 0x400000, v18
	v_cmp_u_f32_e32 vcc_lo, v18, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v17, v17, v25, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	v_mul_f32_e32 v19, v85, v21
	v_bfe_u32 v28, v20, 16, 1
	v_or_b32_e32 v21, 0x400000, v20
	v_mov_b16_e32 v17.l, v0.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v25, v26, v27, vcc_lo
	v_bfe_u32 v26, v19, 16, 1
	v_add3_u32 v18, v28, v20, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v20, v20
	v_mul_f32_e32 v20, v87, v23
	v_mov_b16_e32 v43.l, v62.h
	v_add3_u32 v23, v26, v19, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, v18, v21, vcc_lo
	v_dual_mul_f32 v21, v86, v22 :: v_dual_mul_f32 v22, v131, v24
	v_or_b32_e32 v24, 0x400000, v19
	v_bfe_u32 v26, v20, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_bfe_u32 v27, v21, 16, 1
	v_or_b32_e32 v28, 0x400000, v21
	v_mov_b16_e32 v18.l, v25.h
	v_add3_u32 v19, v26, v20, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v23, v23, v24, vcc_lo
	v_or_b32_e32 v24, 0x400000, v20
	v_cmp_u_f32_e32 vcc_lo, v20, v20
	v_add3_u32 v27, v27, v21, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v24, v19, v24, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v21, v21
	v_mul_f32_e32 v22, v88, v22
	v_bfe_u32 v21, v9, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, v27, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v26, v22, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v22, v22
	v_mov_b16_e32 v19.l, v23.h
	v_add3_u32 v20, v26, v22, 0x7fff
	v_or_b32_e32 v26, 0x400000, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v20, v20, v26, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v9, v9
	v_mul_f32_e32 v0, v124, v10
	v_mul_f32_e32 v10, v125, v11
	v_add3_u32 v11, v21, v9, 0x7fff
	v_or_b32_e32 v21, 0x400000, v9
	v_mov_b16_e32 v20.l, v24.h
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v21, v11, v21, vcc_lo
	v_bfe_u32 v22, v0, 16, 1
	v_mul_f32_e32 v11, v126, v12
	v_bfe_u32 v23, v10, 16, 1
	v_or_b32_e32 v12, 0x400000, v0
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v9, v22, v0, 0x7fff
	v_bfe_u32 v24, v11, 16, 1
	v_add3_u32 v22, v23, v10, 0x7fff
	v_or_b32_e32 v23, 0x400000, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v9, v12, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v10, v10
	v_mul_f32_e32 v12, v119, v13
	v_add3_u32 v10, v24, v11, 0x7fff
	v_or_b32_e32 v13, 0x400000, v11
	v_mov_b16_e32 v9.l, v21.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v22, v23, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v11, v11
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v11, v121, v15 :: v_dual_cndmask_b32 v10, v10, v13
	v_mul_f32_e32 v13, v120, v14
	v_bfe_u32 v22, v12, 16, 1
	v_mul_f32_e32 v14, v131, v16
	v_or_b32_e32 v16, 0x400000, v12
	v_cmp_u_f32_e32 vcc_lo, v12, v12
	v_bfe_u32 v23, v13, 16, 1
	v_add3_u32 v15, v22, v12, 0x7fff
	v_bfe_u32 v22, v11, 16, 1
	v_or_b32_e32 v24, 0x400000, v13
	v_mov_b16_e32 v10.l, v0.h
	v_add3_u32 v23, v23, v13, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v15, v16, vcc_lo
	v_add3_u32 v12, v22, v11, 0x7fff
	v_or_b32_e32 v16, 0x400000, v11
	v_cmp_u_f32_e32 vcc_lo, v11, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v16, v12, v16, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_mul_f32_e32 v14, v122, v14
	v_bfe_u32 v13, v1, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v23, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v22, v14, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v14, v14
	v_mov_b16_e32 v11.l, v15.h
	v_add3_u32 v12, v22, v14, 0x7fff
	v_or_b32_e32 v22, 0x400000, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v12, v12, v22, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v1, v1
	v_mul_f32_e32 v0, v128, v2
	v_mul_f32_e32 v2, v129, v3
	v_add3_u32 v3, v13, v1, 0x7fff
	v_or_b32_e32 v13, 0x400000, v1
	v_mov_b16_e32 v12.l, v16.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_bfe_u32 v15, v2, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v3, v13, vcc_lo
	v_bfe_u32 v14, v0, 16, 1
	v_mul_f32_e32 v3, v130, v4
	v_or_b32_e32 v4, 0x400000, v0
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v1, v14, v0, 0x7fff
	v_add3_u32 v14, v15, v2, 0x7fff
	v_or_b32_e32 v15, 0x400000, v2
	v_bfe_u32 v16, v3, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v1, v4, vcc_lo
	v_mul_f32_e32 v4, v81, v5
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	v_add3_u32 v1, v16, v3, 0x7fff
	v_or_b32_e32 v2, 0x400000, v3
	v_mov_b16_e32 v0.l, v13.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v14, v15, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v3, v3
	v_mul_f32_e32 v3, v82, v6
	v_bfe_u32 v14, v4, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v6, v83, v7 :: v_dual_cndmask_b32 v1, v1, v2
	v_mul_f32_e32 v2, v131, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add3_u32 v7, v14, v4, 0x7fff
	v_or_b32_e32 v8, 0x400000, v4
	v_bfe_u32 v15, v6, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v4, v4
	v_mul_f32_e32 v14, v84, v2
	v_bfe_u32 v2, v3, 16, 1
	v_or_b32_e32 v16, 0x400000, v3
	v_mov_b16_e32 v1.l, v5.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v4, v7, v8, vcc_lo
	v_add3_u32 v8, v15, v6, 0x7fff
	v_or_b32_e32 v15, 0x400000, v6
	v_cmp_u_f32_e32 vcc_lo, v6, v6
	v_bfe_u32 v7, v14, 16, 1
	v_add3_u32 v2, v2, v3, 0x7fff
	v_or_b32_e32 v21, 0x400000, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v8, v15, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v3, v3
	v_add3_u32 v7, v7, v14, 0x7fff
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v16, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v14, v14
	v_mov_b16_e32 v2.l, v4.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, v7, v21, vcc_lo
	v_mov_b16_e32 v3.l, v6.h
	s_clause 0x5
	global_store_b128 v[89:90], v[33:36], off offset:64
	global_store_b128 v[89:90], v[37:40], off offset:96
	global_store_b128 v[89:90], v[41:44], off offset:128
	global_store_b128 v[89:90], v[17:20], off offset:160
	global_store_b128 v[89:90], v[9:12], off offset:192
	global_store_b128 v[89:90], v[0:3], off offset:224
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	sage_hip_attn_bm128_bn32, .Lfunc_end0-sage_hip_attn_bm128_bn32
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sage_hip_attn_bm128_bn32
		.amdhsa_group_segment_fixed_size 28416
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 72
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 36
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-sage_hip_attn_bm128_bn32)<<4)&4080)>>4
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
                                        ; -- End function
	.set .Lsage_hip_attn_bm128_bn32.num_vgpr, 243
	.set .Lsage_hip_attn_bm128_bn32.num_agpr, 0
	.set .Lsage_hip_attn_bm128_bn32.numbered_sgpr, 32
	.set .Lsage_hip_attn_bm128_bn32.num_named_barrier, 0
	.set .Lsage_hip_attn_bm128_bn32.private_seg_size, 0
	.set .Lsage_hip_attn_bm128_bn32.uses_vcc, 1
	.set .Lsage_hip_attn_bm128_bn32.uses_flat_scratch, 0
	.set .Lsage_hip_attn_bm128_bn32.has_dyn_sized_stack, 0
	.set .Lsage_hip_attn_bm128_bn32.has_recursion, 0
	.set .Lsage_hip_attn_bm128_bn32.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 17780
; TotalNumSgprs: 34
; NumVgprs: 243
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 18944 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 30
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 243
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_5f12645180c82362,@object ; @__hip_cuid_5f12645180c82362
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5f12645180c82362
__hip_cuid_5f12645180c82362:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5f12645180c82362, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5f12645180c82362
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 28416
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           sage_hip_attn_bm128_bn32
    .private_segment_fixed_size: 0
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         sage_hip_attn_bm128_bn32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
