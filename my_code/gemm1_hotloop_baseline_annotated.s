; ============================================================================
; gemm1 hot loop -- BASELINE ISA with the observed pipeline inlined as comments
;
; kernel : gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
; build  : AITER_TDM_WIDE_KSL=0 (FRONT/BACK split), gfx1250, waves-per-eu=1,1.
;          NOT gemm1's default -- with the knob unset gemm1 takes the wide loop
;          in the companion file. Both are correct; this one is the fallback.
; source : 21_final_isa.s lines 815..1071 (steady-state K256 tile), verbatim
; state  : vgpr_count=281  vgpr_spill_count=0  sgpr_count=90
;          rel_l2 = 2.8725e-03 (correct)
;
; Timing context (b8-2, g2_m64_nb3 tiles): gemm1 204.3 / 203.1 us.
; The two s_wait_dscnt cost 92.9 + 73.1 = 166 cyc per K-tile (wv1 ATT trace),
; against 168 cyc of WMMA issue -- i.e. LDS waits and compute are about equal,
; which caps what any pure latency-hiding rewrite can win here.
;
; Companion file: gemm1_hotloop_annotated.s (the wide variant, gemm1's default).
; ============================================================================


        ; ==============================================================================
        ; STEADY-STATE K256 TILE  (.LBB0_27 .. s_cbranch_scc1 .LBB0_27)   -- BASELINE
        ;
        ; Software-pipelined: the plan's steps 1-2 sit at the END of the body
        ; (L948-L1068) and feed the NEXT iteration. Per-iteration execution order is
        ; therefore:  3, 4, 5, 6, 7, 8, 9, 10, then 1, 2.
        ;
        ; Register numbers: v256+ is reached via VGPR-MSB indexing (MI400 Shader
        ; Programming #65 sec 3.3.2.3). An operand printed as 'v[66:73] /*v[322:329]*/'
        ; really uses v322..v329; the bare form is only the 8-bit encoding.
        ; WMMA operand order: dst, srcA, srcB, acc, sA, sB -- srcA = FP4 weight (8 dw),
        ; srcB = FP8 activation (16 dw).
        ; ==============================================================================
  815  .LBB0_27:
  816  	s_set_vgpr_msb 64
  817  	v_add_nc_u32_e32 v24 /*v280*/, s80, v236
  818  	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
  819  	s_set_vgpr_msb 0x4001

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 3: load all A0                                        # 16 DS
        ; MATCHES: 16 x ds_load_b128 -> v0..v63.
        ; (Steps 1 and 2 for THIS tile ran at the end of the previous iteration.)
        ; ------------------------------------------------------------------------------
  820  	ds_load_b128 v[0:3], v24 /*v280*/
  821  	ds_load_b128 v[4:7], v24 /*v280*/ offset:32
  822  	ds_load_b128 v[8:11], v24 /*v280*/ offset:64
  823  	ds_load_b128 v[12:15], v24 /*v280*/ offset:96
  824  	ds_load_b128 v[32:35], v24 /*v280*/ offset:4352
  825  	ds_load_b128 v[36:39], v24 /*v280*/ offset:4384
  826  	ds_load_b128 v[40:43], v24 /*v280*/ offset:4416
  827  	ds_load_b128 v[44:47], v24 /*v280*/ offset:4448
  828  	ds_load_b128 v[16:19], v24 /*v280*/ offset:8704
  829  	ds_load_b128 v[20:23], v24 /*v280*/ offset:8736
  830  	ds_load_b128 v[24:27], v24 /*v280*/ offset:8768
  831  	ds_load_b128 v[28:31], v24 /*v280*/ offset:8800
  832  	ds_load_b128 v[48:51], v24 /*v280*/ offset:13056
  833  	ds_load_b128 v[52:55], v24 /*v280*/ offset:13088
  834  	ds_load_b128 v[56:59], v24 /*v280*/ offset:13120
  835  	ds_load_b128 v[60:63], v24 /*v280*/ offset:13152
  836  	s_set_vgpr_msb 0x100

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 4: s_wait_dscnt 0                          # avg 93.9 cycles
        ; MATCHES. Measured 92.9 cyc/hit in the wv1 ATT trace (code.json idx 813,
        ; hit=936, stall=86989).
        ; ------------------------------------------------------------------------------
  837  	s_wait_dscnt 0x0

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 5: execute first 8 KSL0 WMMA, interleaved with load B1  # 8 DS
        ; MATCHES: L838-847 are the 8 x ds_load_b128 for B1 -> v246..v277, and the
        ; 8 WMMA of the first half follow at L848-865 (activations v0..v15, v32..v47).
        ; ------------------------------------------------------------------------------
  838  	ds_load_b128 v[246:249], v244 offset:1024
  839  	ds_load_b128 v[250:253], v244 offset:1536
  840  	ds_load_b128 v[254:257], v244 offset:3072
  841  	s_set_vgpr_msb 64
  842  	ds_load_b128 v[2:5] /*v[258:261]*/, v244 offset:3584
  843  	ds_load_b128 v[6:9] /*v[262:265]*/, v244 offset:5120
  844  	ds_load_b128 v[10:13] /*v[266:269]*/, v244 offset:5632
  845  	ds_load_b128 v[14:17] /*v[270:273]*/, v244 offset:7168
  846  	ds_load_b128 v[18:21] /*v[274:277]*/, v244 offset:7680
  847  	s_set_vgpr_msb 0x4000
  848  	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[192:199], v[0:15], v[64:71], v226, v232 matrix_a_fmt:MATRIX_FMT_FP4
  849  	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[200:207], v[0:15], v[72:79], v227, v232 matrix_a_fmt:MATRIX_FMT_FP4
  850  	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[208:215], v[0:15], v[80:87], v230, v232 matrix_a_fmt:MATRIX_FMT_FP4
  851  	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[216:223], v[0:15], v[88:95], v231, v232 matrix_a_fmt:MATRIX_FMT_FP4
  852  	v_nop
  853  	v_nop
  854  	v_nop
  855  	v_nop
  856  	v_nop
  857  	v_nop
  858  	v_nop
  859  	v_nop
  860  	v_add_nc_u32_e32 v0, 0xc410, v242
  861  	v_add_nc_u32_e32 v1, 0xc418, v242
  862  	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[216:223], v[32:47], v[120:127], v231, v233 matrix_a_fmt:MATRIX_FMT_FP4
  863  	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[208:215], v[32:47], v[112:119], v230, v233 matrix_a_fmt:MATRIX_FMT_FP4
  864  	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[200:207], v[32:47], v[104:111], v227, v233 matrix_a_fmt:MATRIX_FMT_FP4
  865  	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[192:199], v[32:47], v[96:103], v226, v233 matrix_a_fmt:MATRIX_FMT_FP4

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 6: load SB1/SA1                                        # 4 DS
        ;            + load A1 wm0                                         # 4 DS
        ; MATCHES: L866-873 = 4 x ds_load_2addr_b32 (the 8 logical scale b32 loads
        ; paired by the backend) -> v232..v235; L874-878 = 4 x ds_load_b128 = A1 wm0
        ; -> v0..v15 (reusing A0's registers, which KSL0's first half has consumed).
        ; ------------------------------------------------------------------------------
  866  	ds_load_2addr_b32 v[232:233], v243 offset0:160 offset1:176
  867  	s_wait_alu depctr_vm_vsrc(0)
  868  	ds_load_2addr_b32 v[242:243], v243 offset0:224 offset1:240
  869  	s_wait_alu depctr_va_vdst(0)
  870  	ds_load_2addr_b32 v[244:245], v0 offset1:1
  871  	ds_load_2addr_b32 v[234:235], v1 offset1:1
  872  	s_wait_alu depctr_vm_vsrc(0)
  873  	s_set_vgpr_msb 1
  874  	ds_load_b128 v[0:3], v24 /*v280*/ offset:128
  875  	ds_load_b128 v[4:7], v24 /*v280*/ offset:160
  876  	ds_load_b128 v[8:11], v24 /*v280*/ offset:192
  877  	ds_load_b128 v[12:15], v24 /*v280*/ offset:224
  878  	s_set_vgpr_msb 0x100

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 7: execute remaining 8 KSL0 WMMA
        ; MATCHES: 8 WMMA at L879-887, activations v16..v31 and v48..v63.
        ; ------------------------------------------------------------------------------
  879  	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[192:199], v[16:31], v[128:135], v226, v228 matrix_a_fmt:MATRIX_FMT_FP4
  880  	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[200:207], v[16:31], v[136:143], v227, v228 matrix_a_fmt:MATRIX_FMT_FP4
  881  	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[208:215], v[16:31], v[144:151], v230, v228 matrix_a_fmt:MATRIX_FMT_FP4
  882  	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[216:223], v[16:31], v[152:159], v231, v228 matrix_a_fmt:MATRIX_FMT_FP4
  883  	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[216:223], v[48:63], v[184:191], v231, v229 matrix_a_fmt:MATRIX_FMT_FP4
  884  	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[208:215], v[48:63], v[176:183], v230, v229 matrix_a_fmt:MATRIX_FMT_FP4
  885  	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[200:207], v[48:63], v[168:175], v227, v229 matrix_a_fmt:MATRIX_FMT_FP4
  886  	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[192:199], v[48:63], v[160:167], v226, v229 matrix_a_fmt:MATRIX_FMT_FP4
  887  	s_set_vgpr_msb 1

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 8: load A1 wm1..wm3                                   # 12 DS
        ; MATCHES: 12 x ds_load_b128 -> v16..v63.
        ; ------------------------------------------------------------------------------
  888  	ds_load_b128 v[16:19], v24 /*v280*/ offset:4480
  889  	ds_load_b128 v[20:23], v24 /*v280*/ offset:4512
  890  	ds_load_b128 v[24:27], v24 /*v280*/ offset:4544
  891  	ds_load_b128 v[28:31], v24 /*v280*/ offset:4576
  892  	ds_load_b128 v[32:35], v24 /*v280*/ offset:8832
  893  	ds_load_b128 v[36:39], v24 /*v280*/ offset:8864
  894  	ds_load_b128 v[40:43], v24 /*v280*/ offset:8896
  895  	ds_load_b128 v[44:47], v24 /*v280*/ offset:8928
  896  	ds_load_b128 v[48:51], v24 /*v280*/ offset:13184
  897  	ds_load_b128 v[52:55], v24 /*v280*/ offset:13216
  898  	ds_load_b128 v[56:59], v24 /*v280*/ offset:13248
  899  	ds_load_b128 v[60:63], v24 /*v280*/ offset:13280

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 9: s_wait_dscnt 0                          # avg 74.1 cycles
        ; MATCHES. Measured 73.1 cyc/hit in the wv1 ATT trace (code.json idx 876,
        ; hit=936, stall=68459).
        ; ------------------------------------------------------------------------------
  900  	s_wait_dscnt 0x0
  901  	s_set_vgpr_msb 0x100

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 10: execute 16 KSL1 WMMA
        ; MATCHES: 16 WMMA at L902-926, activations span v0..v63 (all of A1).
        ; ------------------------------------------------------------------------------
  902  	v_wmma_scale_f32_16x16x128_f8f6f4 v[64:71], v[246:253], v[0:15], v[64:71], v232, v244 matrix_a_fmt:MATRIX_FMT_FP4
  903  	v_wmma_scale_f32_16x16x128_f8f6f4 v[72:79], v[254:261], v[0:15], v[72:79], v233, v244 matrix_a_fmt:MATRIX_FMT_FP4
  904  	s_set_vgpr_msb 1
  905  	v_wmma_scale_f32_16x16x128_f8f6f4 v[80:87], v[6:13] /*v[262:269]*/, v[0:15], v[80:87], v242, v244 matrix_a_fmt:MATRIX_FMT_FP4
  906  	v_wmma_scale_f32_16x16x128_f8f6f4 v[88:95], v[14:21] /*v[270:277]*/, v[0:15], v[88:95], v243, v244 matrix_a_fmt:MATRIX_FMT_FP4
  907  	v_wmma_scale_f32_16x16x128_f8f6f4 v[120:127], v[14:21] /*v[270:277]*/, v[16:31], v[120:127], v243, v245 matrix_a_fmt:MATRIX_FMT_FP4
  908  	v_wmma_scale_f32_16x16x128_f8f6f4 v[112:119], v[6:13] /*v[262:269]*/, v[16:31], v[112:119], v242, v245 matrix_a_fmt:MATRIX_FMT_FP4
  909  	s_set_vgpr_msb 0x100
  910  	v_wmma_scale_f32_16x16x128_f8f6f4 v[104:111], v[254:261], v[16:31], v[104:111], v233, v245 matrix_a_fmt:MATRIX_FMT_FP4
  911  	v_wmma_scale_f32_16x16x128_f8f6f4 v[96:103], v[246:253], v[16:31], v[96:103], v232, v245 matrix_a_fmt:MATRIX_FMT_FP4
  912  	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[246:253], v[32:47], v[128:135], v232, v234 matrix_a_fmt:MATRIX_FMT_FP4
  913  	v_wmma_scale_f32_16x16x128_f8f6f4 v[136:143], v[254:261], v[32:47], v[136:143], v233, v234 matrix_a_fmt:MATRIX_FMT_FP4
  914  	s_set_vgpr_msb 1
  915  	v_wmma_scale_f32_16x16x128_f8f6f4 v[144:151], v[6:13] /*v[262:269]*/, v[32:47], v[144:151], v242, v234 matrix_a_fmt:MATRIX_FMT_FP4
  916  	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[14:21] /*v[270:277]*/, v[32:47], v[152:159], v243, v234 matrix_a_fmt:MATRIX_FMT_FP4
  917  	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[14:21] /*v[270:277]*/, v[48:63], v[184:191], v243, v235 matrix_a_fmt:MATRIX_FMT_FP4
  918  	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[6:13] /*v[262:269]*/, v[48:63], v[176:183], v242, v235 matrix_a_fmt:MATRIX_FMT_FP4
  919  	s_set_vgpr_msb 0x100
  920  	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[254:261], v[48:63], v[168:175], v233, v235 matrix_a_fmt:MATRIX_FMT_FP4
  921  	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[246:253], v[48:63], v[160:167], v232, v235 matrix_a_fmt:MATRIX_FMT_FP4
  922  	s_add_nc_u64 s[56:57], s[56:57], 0x100
  923  	s_add_co_i32 s72, s72, 1
  924  	s_add_nc_u64 s[52:53], s[52:53], 32
  925  	s_cmp_lg_u32 s56, 0x1a00
  926  	s_add_nc_u64 s[58:59], s[58:59], 0x800
  927  	s_cbranch_scc0 .LBB0_44

        ; ==============================================================================
        ; End of compute. Everything below feeds the NEXT iteration.
        ; ==============================================================================
  928  .LBB0_28:
  929  	s_mul_i32 s8, s72, 0xab
  930  	s_wait_tensorcnt 0x2
  931  	s_add_co_i32 s9, s8, 0xfeaa
  932  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
  933  	s_bfe_u32 s9, s9, 0x70009
  934  	s_barrier_signal -1
  935  	s_mul_i32 s9, s9, 3
  936  	s_sub_co_i32 s9, s72, s9
  937  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  938  	s_add_co_i32 s9, s9, 0xfffe
  939  	s_and_b32 s9, s9, 0xff
  940  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  941  	s_mul_i32 s9, s9, 0xce00
  942  	s_add_co_i32 s80, s9, 0
  943  	s_delay_alu instid0(SALU_CYCLE_1)
  944  	v_dual_add_nc_u32 v244, s80, v239 :: v_dual_add_nc_u32 v0, s80, v237
  945  	v_add_nc_u32_e32 v242, s80, v238
  946  	s_barrier_wait -1
  947  	s_wait_alu depctr_va_vdst(0)

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 1: load B0/SB0/SA0                                    # 12 DS
        ; MATCHES: L948-959 = 8 x ds_load_b128 (B0) -> v192..v223, plus L960-974 =
        ; 4 x ds_load_2addr_b32 (SB0+SA0) -> v226..v229.  8 + 4 = 12 physical DS.
        ; ------------------------------------------------------------------------------
  948  	ds_load_b128 v[192:195], v244
  949  	ds_load_b128 v[196:199], v244 offset:512
  950  	ds_load_b128 v[200:203], v244 offset:2048
  951  	ds_load_b128 v[204:207], v244 offset:2560
  952  	ds_load_b128 v[208:211], v244 offset:4096
  953  	ds_load_b128 v[212:215], v244 offset:4608
  954  	v_add_nc_u32_e32 v243, 0xc400, v0
  955  	v_add_nc_u32_e32 v0, 0xc400, v242
  956  	v_add_nc_u32_e32 v1, 0xc408, v242
  957  	ds_load_b128 v[216:219], v244 offset:6144
  958  	ds_load_b128 v[220:223], v244 offset:6656
  959  	s_wait_alu depctr_va_vdst(2)
  960  	ds_load_2addr_b32 v[226:227], v243 offset0:128 offset1:144
  961  	ds_load_2addr_b32 v[230:231], v243 offset0:192 offset1:208
  962  	s_wait_alu depctr_va_vdst(1)
  963  	ds_load_2addr_b32 v[232:233], v0 offset1:1
  964  	s_wait_alu depctr_va_vdst(0)
  965  	ds_load_2addr_b32 v[228:229], v1 offset1:1
  966  	s_bfe_u32 s8, s8, 0x70009
  967  	s_and_b32 vcc_lo, exec_lo, s2
  968  	s_mul_i32 s8, s8, 3
  969  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  970  	s_sub_co_i32 s8, s72, s8
  971  	s_and_b32 s10, s8, 0xff
  972  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  973  	s_mul_i32 s8, s10, 0xce00
  974  	s_add_co_i32 s61, s8, 0

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 2: issue next K-tile TDM
        ; Uses TENSORcnt, so it does not add to this wave's DScnt.
        ; Emitted as 5 predicated tensor_load_to_lds (L987/1007/1024/1045/1068), each
        ; behind its own branch: issue() emits one copy per job and the wave-specialised
        ; path guards each on 'wave == j.wave'.
        ; ------------------------------------------------------------------------------
  975  	s_cbranch_vccz .LBB0_42
  976  	s_and_b32 vcc_lo, exec_lo, s3
  977  	s_cbranch_vccnz .LBB0_31
  978  .LBB0_30:
  979  	s_add_nc_u64 s[8:9], s[64:65], s[56:57]
  980  	s_add_co_i32 s14, s79, s56
  981  	s_add_nc_u64 s[8:9], s[8:9], 0x38200
  982  	s_add_co_i32 s85, s61, 0x2200
  983  	s_add_co_i32 s86, s14, 0x38200
  984  	s_or_b32 s87, s9, 0x80000000
  985  	s_mov_b32 s84, s60
  986  	s_delay_alu instid0(SALU_CYCLE_1)
  987  	tensor_load_to_lds s[84:87], s[36:43]
  988  .LBB0_31:
  989  	s_wait_alu depctr_vm_vsrc(1)
  990  	v_add_nc_u32_e32 v0, s58, v240
  991  	s_and_b32 vcc_lo, exec_lo, s4
  992  	s_add_nc_u64 s[8:9], s[68:69], s[58:59]
  993  	s_cbranch_vccnz .LBB0_33
  994  	s_add_nc_u64 s[14:15], s[8:9], 0x1000
  995  	s_add_co_i32 s14, s61, 0x4400
  996  	s_bitset1_b32 s15, 31
  997  	s_wait_alu depctr_vm_vsrc(0)
  998  	v_dual_mov_b32 v1, s14 :: v_dual_add_nc_u32 v2, 0x1000, v0
  999  	v_mov_b32_e32 v3, s15
 1000  	v_readfirstlane_b32 s84, v224
 1001  	s_mov_b32 s22, s16
 1002  	v_readfirstlane_b32 s85, v1
 1003  	v_readfirstlane_b32 s86, v2
 1004  	v_readfirstlane_b32 s87, v3
 1005  	s_mov_b32 s23, s16
 1006  	s_delay_alu instid0(SALU_CYCLE_1)
 1007  	tensor_load_to_lds s[84:87], s[16:23]
 1008  .LBB0_33:
 1009  	s_and_b32 vcc_lo, exec_lo, s5
 1010  	s_cbranch_vccnz .LBB0_35
 1011  	s_add_nc_u64 s[8:9], s[8:9], 0x71000
 1012  	s_add_co_i32 s8, s61, 0x8400
 1013  	s_bitset1_b32 s9, 31
 1014  	s_wait_alu depctr_vm_vsrc(0)
 1015  	v_dual_mov_b32 v1, s8 :: v_dual_add_nc_u32 v2, 0x71000, v0
 1016  	v_mov_b32_e32 v3, s9
 1017  	v_readfirstlane_b32 s84, v224
 1018  	s_mov_b32 s22, s16
 1019  	v_readfirstlane_b32 s85, v1
 1020  	v_readfirstlane_b32 s86, v2
 1021  	v_readfirstlane_b32 s87, v3
 1022  	s_mov_b32 s23, s16
 1023  	s_delay_alu instid0(SALU_CYCLE_1)
 1024  	tensor_load_to_lds s[84:87], s[16:23]
 1025  .LBB0_35:
 1026  	s_and_b32 vcc_lo, exec_lo, s2
 1027  	s_mul_i32 s81, s10, 0x3380
 1028  	s_cbranch_vccz .LBB0_43
 1029  	s_and_b32 vcc_lo, exec_lo, s3
 1030  	s_cbranch_vccnz .LBB0_38
 1031  .LBB0_37:
 1032  	s_lshl2_add_u32 s10, s81, 0
 1033  	s_add_nc_u64 s[8:9], s[70:71], s[52:53]
 1034  	s_add_co_i32 s61, s10, 0xc500
 1035  	s_add_co_i32 s10, s78, s52
 1036  	s_add_nc_u64 s[8:9], s[8:9], 0x1c40
 1037  	s_add_co_i32 s62, s10, 0x1c40
 1038  	s_or_b32 s63, s9, 0x80000000
 1039  	s_mov_b32 s45, s17
 1040  	s_mov_b32 s46, s18
 1041  	s_mov_b32 s48, s20
 1042  	s_mov_b32 s50, s16
 1043  	s_mov_b32 s51, s16
 1044  	s_delay_alu instid0(SALU_CYCLE_1)
 1045  	tensor_load_to_lds s[60:63], s[44:51]
 1046  .LBB0_38:
 1047  	v_add_nc_u32_e32 v0, s56, v241
 1048  	s_and_b32 vcc_lo, exec_lo, s4
 1049  	s_add_nc_u64 s[22:23], s[6:7], s[56:57]
 1050  	s_cbranch_vccnz .LBB0_40
 1051  	s_lshl2_add_u32 s10, s81, 0
 1052  	s_add_nc_u64 s[8:9], s[22:23], 0x200
 1053  	s_add_co_i32 s10, s10, 0xc600
 1054  	s_or_b32 s8, s9, 0x80000000
 1055  	s_wait_alu depctr_vm_vsrc(0)
 1056  	v_dual_mov_b32 v1, s10 :: v_dual_add_nc_u32 v2, 0x200, v0
 1057  	v_mov_b32_e32 v3, s8
 1058  	v_readfirstlane_b32 s84, v224
 1059  	s_mov_b32 s8, s44
 1060  	v_readfirstlane_b32 s86, v2
 1061  	v_readfirstlane_b32 s85, v1
 1062  	v_readfirstlane_b32 s87, v3
 1063  	s_mov_b32 s9, s17
 1064  	s_mov_b32 s10, s18
 1065  	s_mov_b32 s14, s16
 1066  	s_mov_b32 s15, s16
 1067  	s_delay_alu instid0(SALU_CYCLE_1)
 1068  	tensor_load_to_lds s[84:87], s[8:15]
 1069  .LBB0_40:
 1070  	s_and_b32 vcc_lo, exec_lo, s5
 1071  	s_cbranch_vccnz .LBB0_27
 1072  	s_lshl2_add_u32 s10, s81, 0

; ============================================================================
; SUMMARY -- every step of the observed pipeline maps 1:1 onto the ISA
;
;   step                                        emitted           DS  verdict
;   ------------------------------------------  ---------------  ---  -------
;   1.  load B0/SB0/SA0                         L948-974          12  match
;   2.  issue next K-tile TDM                   L987..L1068        0  match
;   3.  load all A0                             L820-836          16  match
;   4.  s_wait_dscnt 0        avg 93.9 cyc      L837               -  match
;   5.  first 8 KSL0 WMMA + load B1             L838-865           8  match
;   6.  load SB1/SA1 + A1 wm0                   L866-878         4+4  match
;   7.  remaining 8 KSL0 WMMA                   L879-887           -  match
;   8.  load A1 wm1..wm3                        L888-899          12  match
;   9.  s_wait_dscnt 0        avg 74.1 cyc      L900               -  match
;   10. 16 KSL1 WMMA                            L902-926           -  match
;
;   Physical DS per K256 tile: 12 + 16 + 8 + 8 + 12 = 56.
;
; Why the loads are interleaved with the WMMAs: A1 is written into v0..v63,
; the same registers A0 occupies. Step 6 can only start once KSL0's first half
; (step 5) has consumed v0..v15 / v32..v47, and step 8 once step 7 has consumed
; the rest. That is what keeps the register count at 281 instead of the 346 the
; wide variant needs, at the price of two full s_wait_dscnt 0 barriers.
; ============================================================================
