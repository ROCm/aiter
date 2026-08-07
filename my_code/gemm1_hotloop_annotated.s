; ============================================================================
; gemm1 hot loop -- manually corrected wide-KSL target pipeline
;
; kernel : gemm_a8w4_tdm_t64x256x256_w1x4_b3_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
; build  : AITER_TDM_WIDE_KSL=1, gfx1250, waves-per-eu=1,1
; basis  : 21_final_isa.s source IDs 828..1093 (loop back-edge at ID 1092)
; state  : vgpr_count=346  vgpr_spill_count=0  sgpr_count=90
;
; STATUS: HAND-CORRECTED TARGET. This file is no longer a verbatim compiler dump.
;         The DS groups and DScnt waits have been manually scheduled into the
;         requested B0/S0 -> A0 -> B1/S1 -> TDM -> wait12 -> A1 -> KSL0 ->
;         wait0 -> KSL1 pipeline. The left-column numbers are original compiler
;         source instruction IDs retained only as provenance tags; after manual
;         movement they are intentionally non-monotonic and are not line/order IDs.
; ============================================================================


        ; ==============================================================================
        ; STEADY-STATE K256 TILE  (.LBB0_27 .. s_cbranch_scc1 .LBB0_27)
        ;
        ; The loop is software-pipelined: plan steps 1-4 sit at the END of the body
        ; and feed the NEXT iteration; the textual body opens at step 5. Starting at
        ; the cyclic tile boundary, actual order is exactly steps 1 through 9.
        ;
        ; KSL boundary in this hand-corrected stream:
        ;   KSL0 block = IDs 850..868; its 16 WMMA are 850, 853..854, 856..868
        ;                (IDs 852 and 855 are retained s_set_vgpr_msb)
        ;   wait0 = source ID 851, manually moved to the boundary
        ;   KSL1 block = IDs 869..886; its 16 WMMA are 869, 871..872, 874..886
        ;                (IDs 870 and 873 are retained s_set_vgpr_msb)
        ;
        ; NOTE on register numbers: v256+ is reached through VGPR-MSB indexing
        ; (MI400 Shader Programming #65 sec 3.3.2.3). Where an operand prints as
        ; 'v[66:73] /*v[322:329]*/' the COMMENT is the real register; the bare
        ; v[66:73] is only the 8-bit encoding.
        ;
        ; WMMA operand order is  dst, srcA, srcB, acc, sA, sB  where srcA = FP4
        ; weight (8 dw) and srcB = FP8 activation (16 dw).
        ; ==============================================================================
  828  .LBB0_27:

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 5: s_wait_dscnt(12)
        ;   - B0/SB0/SA0 all ready
        ;   - A0 all ready
        ;   - B1/SB1/SA1 allowed to stay pending
        ; MATCHES: 0xc == 12.
        ; ------------------------------------------------------------------------------
  829  	s_wait_dscnt 0xc
  830  	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 6: load all A1
        ;   - 16 x ds_load_b128
        ;   - DScnt goes from 12 up to 28
        ; MATCHES: 16 loads, destinations v64..v127 (consumed by step 9 below).
        ; ------------------------------------------------------------------------------
  831  	ds_load_b128 v[64:67], v124 offset:128
  832  	ds_load_b128 v[68:71], v124 offset:160
  833  	ds_load_b128 v[72:75], v124 offset:192
  834  	ds_load_b128 v[76:79], v124 offset:224
  835  	ds_load_b128 v[80:83], v124 offset:4480
  836  	ds_load_b128 v[84:87], v124 offset:4512
  837  	ds_load_b128 v[88:91], v124 offset:4544
  838  	ds_load_b128 v[92:95], v124 offset:4576
  839  	ds_load_b128 v[96:99], v124 offset:8832
  840  	ds_load_b128 v[100:103], v124 offset:8864
  841  	ds_load_b128 v[104:107], v124 offset:8896
  842  	ds_load_b128 v[108:111], v124 offset:8928
  843  	ds_load_b128 v[112:115], v124 offset:13184
  844  	ds_load_b128 v[116:119], v124 offset:13216
  845  	ds_load_b128 v[120:123], v124 offset:13248
  846  	s_wait_alu depctr_vm_vsrc(0)
  847  	ds_load_b128 v[124:127], v124 offset:13280
  848  	s_set_vgpr_msb 1

        ; PLAN STEP 7 begins here (16 WMMA, activation = v0..v63 = A0).
        ;   4 wm x 4 wn = 16 x WMMAScale_16x16x128
        ;   All 16 execute before wait0 and hide B1/SB1/SA1 + A1 LDS latency.
  850  	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[8:15] /*v[264:271]*/, v[48:63], v[128:135], v140, v150 matrix_a_fmt:MATRIX_FMT_FP4
  852  	s_set_vgpr_msb 0x151
  853  	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[24:31] /*v[280:287]*/, v[48:63], v[66:73] /*v[322:329]*/, v141, v150 matrix_a_fmt:MATRIX_FMT_FP4
  854  	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[48:55] /*v[304:311]*/, v[48:63], v[74:81] /*v[330:337]*/, v144, v150 matrix_a_fmt:MATRIX_FMT_FP4
  855  	s_set_vgpr_msb 0x5101
  856  	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[56:63] /*v[312:319]*/, v[48:63], v[152:159], v145, v150 matrix_a_fmt:MATRIX_FMT_FP4
  857  	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[56:63] /*v[312:319]*/, v[32:47], v[184:191], v145, v151 matrix_a_fmt:MATRIX_FMT_FP4
  858  	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[48:55] /*v[304:311]*/, v[32:47], v[176:183], v144, v151 matrix_a_fmt:MATRIX_FMT_FP4
  859  	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[24:31] /*v[280:287]*/, v[32:47], v[168:175], v141, v151 matrix_a_fmt:MATRIX_FMT_FP4
  860  	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[8:15] /*v[264:271]*/, v[32:47], v[160:167], v140, v151 matrix_a_fmt:MATRIX_FMT_FP4
  861  	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[8:15] /*v[264:271]*/, v[16:31], v[192:199], v140, v148 matrix_a_fmt:MATRIX_FMT_FP4
  862  	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[24:31] /*v[280:287]*/, v[16:31], v[200:207], v141, v148 matrix_a_fmt:MATRIX_FMT_FP4
  863  	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[48:55] /*v[304:311]*/, v[16:31], v[208:215], v144, v148 matrix_a_fmt:MATRIX_FMT_FP4
  864  	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[56:63] /*v[312:319]*/, v[16:31], v[216:223], v145, v148 matrix_a_fmt:MATRIX_FMT_FP4
  865  	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[56:63] /*v[312:319]*/, v[0:15], v[248:255], v145, v149 matrix_a_fmt:MATRIX_FMT_FP4
  866  	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[48:55] /*v[304:311]*/, v[0:15], v[240:247], v144, v149 matrix_a_fmt:MATRIX_FMT_FP4
  867  	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[24:31] /*v[280:287]*/, v[0:15], v[232:239], v141, v149 matrix_a_fmt:MATRIX_FMT_FP4
  868  	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[8:15] /*v[264:271]*/, v[0:15], v[224:231], v140, v149 matrix_a_fmt:MATRIX_FMT_FP4

        ; PLAN STEP 8: full DS wait at the exact KSL0/KSL1 boundary.
  851  	s_wait_dscnt 0x0

        ; PLAN STEP 9 begins here (16 WMMA, activation = v64..v127 = A1 loaded at
        ; source ID 831 this iteration).  16 x WMMAScale_16x16x128.
  869  	v_wmma_scale_f32_16x16x128_f8f6f4 v[128:135], v[0:7] /*v[256:263]*/, v[64:79], v[128:135], v136, v146 matrix_a_fmt:MATRIX_FMT_FP4
  870  	s_set_vgpr_msb 0x151
  871  	v_wmma_scale_f32_16x16x128_f8f6f4 v[66:73] /*v[322:329]*/, v[16:23] /*v[272:279]*/, v[64:79], v[66:73] /*v[322:329]*/, v137, v146 matrix_a_fmt:MATRIX_FMT_FP4
  872  	v_wmma_scale_f32_16x16x128_f8f6f4 v[74:81] /*v[330:337]*/, v[32:39] /*v[288:295]*/, v[64:79], v[74:81] /*v[330:337]*/, v142, v146 matrix_a_fmt:MATRIX_FMT_FP4
  873  	s_set_vgpr_msb 0x5101
  874  	v_wmma_scale_f32_16x16x128_f8f6f4 v[152:159], v[40:47] /*v[296:303]*/, v[64:79], v[152:159], v143, v146 matrix_a_fmt:MATRIX_FMT_FP4
  875  	v_wmma_scale_f32_16x16x128_f8f6f4 v[184:191], v[40:47] /*v[296:303]*/, v[80:95], v[184:191], v143, v147 matrix_a_fmt:MATRIX_FMT_FP4
  876  	v_wmma_scale_f32_16x16x128_f8f6f4 v[176:183], v[32:39] /*v[288:295]*/, v[80:95], v[176:183], v142, v147 matrix_a_fmt:MATRIX_FMT_FP4
  877  	v_wmma_scale_f32_16x16x128_f8f6f4 v[168:175], v[16:23] /*v[272:279]*/, v[80:95], v[168:175], v137, v147 matrix_a_fmt:MATRIX_FMT_FP4
  878  	v_wmma_scale_f32_16x16x128_f8f6f4 v[160:167], v[0:7] /*v[256:263]*/, v[80:95], v[160:167], v136, v147 matrix_a_fmt:MATRIX_FMT_FP4
  879  	v_wmma_scale_f32_16x16x128_f8f6f4 v[192:199], v[0:7] /*v[256:263]*/, v[96:111], v[192:199], v136, v138 matrix_a_fmt:MATRIX_FMT_FP4
  880  	v_wmma_scale_f32_16x16x128_f8f6f4 v[200:207], v[16:23] /*v[272:279]*/, v[96:111], v[200:207], v137, v138 matrix_a_fmt:MATRIX_FMT_FP4
  881  	v_wmma_scale_f32_16x16x128_f8f6f4 v[208:215], v[32:39] /*v[288:295]*/, v[96:111], v[208:215], v142, v138 matrix_a_fmt:MATRIX_FMT_FP4
  882  	v_wmma_scale_f32_16x16x128_f8f6f4 v[216:223], v[40:47] /*v[296:303]*/, v[96:111], v[216:223], v143, v138 matrix_a_fmt:MATRIX_FMT_FP4
  883  	v_wmma_scale_f32_16x16x128_f8f6f4 v[248:255], v[40:47] /*v[296:303]*/, v[112:127], v[248:255], v143, v139 matrix_a_fmt:MATRIX_FMT_FP4
  884  	v_wmma_scale_f32_16x16x128_f8f6f4 v[240:247], v[32:39] /*v[288:295]*/, v[112:127], v[240:247], v142, v139 matrix_a_fmt:MATRIX_FMT_FP4
  885  	v_wmma_scale_f32_16x16x128_f8f6f4 v[232:239], v[16:23] /*v[272:279]*/, v[112:127], v[232:239], v137, v139 matrix_a_fmt:MATRIX_FMT_FP4
  886  	v_wmma_scale_f32_16x16x128_f8f6f4 v[224:231], v[0:7] /*v[256:263]*/, v[112:127], v[224:231], v136, v139 matrix_a_fmt:MATRIX_FMT_FP4
  887  	s_add_nc_u64 s[56:57], s[56:57], 0x100
  888  	s_add_co_i32 s72, s72, 1
  889  	s_add_nc_u64 s[52:53], s[52:53], 32
  890  	s_cmp_lg_u32 s56, 0x1a00
  891  	s_add_nc_u64 s[58:59], s[58:59], 0x800
  892  	s_set_vgpr_msb 0x100
  893  	s_cbranch_scc0 .LBB0_44

        ; ==============================================================================
        ; End of compute. Everything below feeds the NEXT iteration.
        ; ==============================================================================
  894  .LBB0_28:
  895  	s_mul_i32 s8, s72, 0xab
  896  	s_wait_tensorcnt 0x2
  897  	s_add_co_i32 s9, s8, 0xfeaa
  898  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
  899  	s_bfe_u32 s9, s9, 0x70009
  900  	s_barrier_signal -1
  901  	s_mul_i32 s9, s9, 3
  902  	s_sub_co_i32 s9, s72, s9
  903  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  904  	s_add_co_i32 s9, s9, 0xfffe
  905  	s_and_b32 s9, s9, 0xff
  906  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  907  	s_mul_i32 s9, s9, 0xce00
  908  	s_add_co_i32 s9, s9, 0
  909  	s_set_vgpr_msb 4
  910  	v_dual_add_nc_u32 v64, s9, v87 /*v343*/ :: v_dual_add_nc_u32 v0, s9, v86 /*v342*/
  911  	s_wait_alu depctr_vm_vsrc(0)
  912  	v_nop
  913  	v_nop
  914  	v_nop
  915  	v_nop
  916  	v_nop
  917  	v_nop
  918  	v_nop
  919  	v_dual_add_nc_u32 v65, s9, v84 /*v340*/ :: v_dual_add_nc_u32 v124, s9, v85 /*v341*/
  920  	s_set_vgpr_msb 0x400
  921  	s_barrier_wait -1
  922  	v_add_nc_u32_e32 v66, 0xc400, v0
  951  	v_add_nc_u32_e32 v67, 0xc400, v65
  952  	v_add_nc_u32_e32 v68, 0xc408, v65
  923  	s_wait_alu depctr_va_vdst(0)

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 1: load B0/SB0/SA0  (for the next tile)
        ;   - 8 x ds_load_b128: source IDs 954..961
        ;   - 4 x ds_load_2addr_b32: source IDs 963, 964, 970, 972
        ;   - destinations are exactly the B/scales consumed by KSL0
        ; PHYSICAL DS: 12.
        ; ------------------------------------------------------------------------------
  953  	s_set_vgpr_msb 64
  954  	ds_load_b128 v[8:11] /*v[264:267]*/, v64
  955  	ds_load_b128 v[12:15] /*v[268:271]*/, v64 offset:512
  956  	ds_load_b128 v[24:27] /*v[280:283]*/, v64 offset:2048
  957  	ds_load_b128 v[28:31] /*v[284:287]*/, v64 offset:2560
  958  	ds_load_b128 v[48:51] /*v[304:307]*/, v64 offset:4096
  959  	ds_load_b128 v[52:55] /*v[308:311]*/, v64 offset:4608
  960  	ds_load_b128 v[56:59] /*v[312:315]*/, v64 offset:6144
  961  	ds_load_b128 v[60:63] /*v[316:319]*/, v64 offset:6656
  962  	s_set_vgpr_msb 0x4000
  963  	ds_load_2addr_b32 v[140:141], v66 offset0:128 offset1:144
  964  	ds_load_2addr_b32 v[144:145], v66 offset0:192 offset1:208
  969  	s_wait_alu depctr_va_vdst(3)
  970  	ds_load_2addr_b32 v[150:151], v67 offset1:1
  971  	s_wait_alu depctr_va_vdst(2)
  972  	ds_load_2addr_b32 v[148:149], v68 offset1:1

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 2: load all A0  (for the next tile)
        ;   - wm0..wm3, each 4 x ds_load_b128 -> 16 total
        ;   - destinations v0..v63 are consumed by KSL0
        ; PHYSICAL DS: 16.
        ; ------------------------------------------------------------------------------
  924  	ds_load_b128 v[48:51], v124
  925  	ds_load_b128 v[52:55], v124 offset:32
  926  	ds_load_b128 v[56:59], v124 offset:64
  927  	ds_load_b128 v[60:63], v124 offset:96
  928  	ds_load_b128 v[32:35], v124 offset:4352
  929  	ds_load_b128 v[36:39], v124 offset:4384
  930  	ds_load_b128 v[40:43], v124 offset:4416
  931  	ds_load_b128 v[44:47], v124 offset:4448
  932  	ds_load_b128 v[16:19], v124 offset:8704
  933  	ds_load_b128 v[20:23], v124 offset:8736
  934  	ds_load_b128 v[24:27], v124 offset:8768
  935  	ds_load_b128 v[28:31], v124 offset:8800
  936  	ds_load_b128 v[0:3], v124 offset:13056
  937  	ds_load_b128 v[4:7], v124 offset:13088
  938  	ds_load_b128 v[8:11], v124 offset:13120
  939  	ds_load_b128 v[12:15], v124 offset:13152

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 3: load B1/SB1/SA1  (for the next tile)
        ;   - 8 x ds_load_b128: source IDs 941..948
        ;   - 4 x ds_load_2addr_b32: source IDs 950, 968, 974, 976
        ;   - destinations are exactly the B/scales consumed by KSL1
        ; PHYSICAL DS: 12. The two v66 scale loads immediately before the VM-source
        ; wait preserve the original safe point before v64/v65 are repurposed.
        ; ------------------------------------------------------------------------------
  940  	s_set_vgpr_msb 64
  941  	ds_load_b128 v[0:3] /*v[256:259]*/, v64 offset:1024
  942  	ds_load_b128 v[4:7] /*v[260:263]*/, v64 offset:1536
  943  	ds_load_b128 v[16:19] /*v[272:275]*/, v64 offset:3072
  944  	ds_load_b128 v[20:23] /*v[276:279]*/, v64 offset:3584
  945  	ds_load_b128 v[32:35] /*v[288:291]*/, v64 offset:5120
  946  	ds_load_b128 v[36:39] /*v[292:295]*/, v64 offset:5632
  947  	ds_load_b128 v[40:43] /*v[296:299]*/, v64 offset:7168
  948  	ds_load_b128 v[44:47] /*v[300:303]*/, v64 offset:7680
  949  	s_set_vgpr_msb 0x4000
  950  	ds_load_2addr_b32 v[136:137], v66 offset0:160 offset1:176
  968  	ds_load_2addr_b32 v[142:143], v66 offset0:224 offset1:240
  965  	s_wait_alu depctr_vm_vsrc(2)
  966  	v_add_nc_u32_e32 v64, 0xc410, v65
  967  	v_add_nc_u32_e32 v65, 0xc418, v65
  973  	s_wait_alu depctr_va_vdst(1)
  974  	ds_load_2addr_b32 v[146:147], v64 offset1:1
  975  	s_wait_alu depctr_va_vdst(0)
  976  	ds_load_2addr_b32 v[138:139], v65 offset1:1
  977  	s_bfe_u32 s8, s8, 0x70009
  978  	s_and_b32 vcc_lo, exec_lo, s2
  979  	s_mul_i32 s8, s8, 3
  980  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  981  	s_sub_co_i32 s8, s72, s8
  982  	s_and_b32 s10, s8, 0xff
  983  	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
  984  	s_mul_i32 s8, s10, 0xce00
  985  	s_add_co_i32 s61, s8, 0

        ; ------------------------------------------------------------------------------
        ; PLAN STEP 4: issue next K-tile TDM
        ;   - writes into another ring buffer
        ;   - uses TENSORcnt, does NOT add to this wave's DScnt
        ;
        ; The compiler's five predicated tensor_load_to_lds operations (source IDs
        ; 998/1022/1041/1062/1089) and their scalar guards are retained unchanged.
        ; Together they implement the one logical next-tile issue.
        ; ------------------------------------------------------------------------------
  986  	s_cbranch_vccz .LBB0_42
  987  	s_and_b32 vcc_lo, exec_lo, s3
  988  	s_cbranch_vccnz .LBB0_31
  989  .LBB0_30:
  990  	s_add_nc_u64 s[8:9], s[64:65], s[56:57]
  991  	s_add_co_i32 s14, s79, s56
  992  	s_add_nc_u64 s[8:9], s[8:9], 0x38200
  993  	s_add_co_i32 s81, s61, 0x2200
  994  	s_add_co_i32 s82, s14, 0x38200
  995  	s_or_b32 s83, s9, 0x80000000
  996  	s_mov_b32 s80, s60
  997  	s_delay_alu instid0(SALU_CYCLE_1)
  998  	tensor_load_to_lds s[80:83], s[36:43]
  999  .LBB0_31:
 1000  	s_wait_alu depctr_vm_vsrc(1)
 1001  	s_set_vgpr_msb 4
 1002  	v_add_nc_u32_e32 v64, s58, v88 /*v344*/
 1003  	s_and_b32 vcc_lo, exec_lo, s4
 1004  	s_add_nc_u64 s[8:9], s[68:69], s[58:59]
 1005  	s_set_vgpr_msb 0x400
 1006  	s_cbranch_vccnz .LBB0_33
 1007  	s_add_nc_u64 s[14:15], s[8:9], 0x1000
 1008  	s_add_co_i32 s14, s61, 0x4400
 1009  	s_bitset1_b32 s15, 31
 1010  	s_wait_alu depctr_vm_vsrc(0)
 1011  	v_dual_mov_b32 v65, s14 :: v_dual_add_nc_u32 v66, 0x1000, v64
 1012  	v_mov_b32_e32 v67, s15
 1013  	s_set_vgpr_msb 1
 1014  	v_readfirstlane_b32 s80, v64 /*v320*/
 1015  	s_mov_b32 s22, s16
 1016  	s_set_vgpr_msb 0x100
 1017  	v_readfirstlane_b32 s81, v65
 1018  	v_readfirstlane_b32 s82, v66
 1019  	v_readfirstlane_b32 s83, v67
 1020  	s_mov_b32 s23, s16
 1021  	s_delay_alu instid0(SALU_CYCLE_1)
 1022  	tensor_load_to_lds s[80:83], s[16:23]
 1023  .LBB0_33:
 1024  	s_and_b32 vcc_lo, exec_lo, s5
 1025  	s_cbranch_vccnz .LBB0_35
 1026  	s_add_nc_u64 s[8:9], s[8:9], 0x71000
 1027  	s_add_co_i32 s8, s61, 0x8400
 1028  	s_bitset1_b32 s9, 31
 1029  	s_wait_alu depctr_vm_vsrc(0)
 1030  	v_dual_mov_b32 v65, s8 :: v_dual_add_nc_u32 v64, 0x71000, v64
 1031  	v_mov_b32_e32 v67, s9
 1032  	s_set_vgpr_msb 1
 1033  	v_readfirstlane_b32 s80, v64 /*v320*/
 1034  	s_mov_b32 s22, s16
 1035  	s_set_vgpr_msb 0x100
 1036  	v_readfirstlane_b32 s81, v65
 1037  	v_readfirstlane_b32 s82, v64
 1038  	v_readfirstlane_b32 s83, v67
 1039  	s_mov_b32 s23, s16
 1040  	s_delay_alu instid0(SALU_CYCLE_1)
 1041  	tensor_load_to_lds s[80:83], s[16:23]
 1042  .LBB0_35:
 1043  	s_and_b32 vcc_lo, exec_lo, s2
 1044  	s_mul_i32 s80, s10, 0x3380
 1045  	s_cbranch_vccz .LBB0_43
 1046  	s_and_b32 vcc_lo, exec_lo, s3
 1047  	s_cbranch_vccnz .LBB0_38
 1048  .LBB0_37:
 1049  	s_lshl2_add_u32 s10, s80, 0
 1050  	s_add_nc_u64 s[8:9], s[70:71], s[52:53]
 1051  	s_add_co_i32 s61, s10, 0xc500
 1052  	s_add_co_i32 s10, s78, s52
 1053  	s_add_nc_u64 s[8:9], s[8:9], 0x1c40
 1054  	s_add_co_i32 s62, s10, 0x1c40
 1055  	s_or_b32 s63, s9, 0x80000000
 1056  	s_mov_b32 s45, s17
 1057  	s_mov_b32 s46, s18
 1058  	s_mov_b32 s48, s20
 1059  	s_mov_b32 s50, s16
 1060  	s_mov_b32 s51, s16
 1061  	s_delay_alu instid0(SALU_CYCLE_1)
 1062  	tensor_load_to_lds s[60:63], s[44:51]
 1063  .LBB0_38:
 1064  	s_set_vgpr_msb 4
 1065  	v_add_nc_u32_e32 v64, s56, v89 /*v345*/
 1066  	s_and_b32 vcc_lo, exec_lo, s4
 1067  	s_add_nc_u64 s[22:23], s[6:7], s[56:57]
 1068  	s_set_vgpr_msb 0x400
 1069  	s_cbranch_vccnz .LBB0_40
 1070  	s_lshl2_add_u32 s10, s80, 0
 1071  	s_add_nc_u64 s[8:9], s[22:23], 0x200
 1072  	s_add_co_i32 s10, s10, 0xc600
 1073  	s_or_b32 s8, s9, 0x80000000
 1074  	s_wait_alu depctr_vm_vsrc(0)
 1075  	v_dual_mov_b32 v65, s10 :: v_dual_add_nc_u32 v66, 0x200, v64
 1076  	v_mov_b32_e32 v67, s8
 1077  	s_set_vgpr_msb 1
 1078  	v_readfirstlane_b32 s84, v64 /*v320*/
 1079  	s_mov_b32 s8, s44
 1080  	s_set_vgpr_msb 0x100
 1081  	v_readfirstlane_b32 s86, v66
 1082  	v_readfirstlane_b32 s85, v65
 1083  	v_readfirstlane_b32 s87, v67
 1084  	s_mov_b32 s9, s17
 1085  	s_mov_b32 s10, s18
 1086  	s_mov_b32 s14, s16
 1087  	s_mov_b32 s15, s16
 1088  	s_delay_alu instid0(SALU_CYCLE_1)
 1089  	tensor_load_to_lds s[84:87], s[8:15]
 1090  .LBB0_40:
 1091  	s_and_b32 vcc_lo, exec_lo, s5
 1092  	s_cbranch_vccnz .LBB0_27
 1093  	s_lshl2_add_u32 s10, s80, 0

; ============================================================================
; SUMMARY
;
;   Cyclic steady-state order (numbers below are original source IDs):
;   1. B0/S0:  954..961 + 963/964/970/972       8 b128 + 4 2addr = 12 DS
;   2. A0:     924..939                          16 b128       = 16 DS
;   3. B1/S1:  941..948 + 950/968/974/976       8 b128 + 4 2addr = 12 DS
;   4. next K-tile TDM: 998/1022/1041/1062/1089 (TENSORcnt, not DScnt)
;   5. s_wait_dscnt(12): 829 (0xc)
;   6. A1:     831..845, 847                     16 b128       = 16 DS
;   7. KSL0:   850, 853..854, 856..868           exactly 16 WMMA
;   8. s_wait_dscnt(0): 851                      exact KSL boundary
;   9. KSL1:   869, 871..872, 874..886           exactly 16 WMMA
;
;   Physical DS per K256 tile: 12 + 16 + 12 + 16 = 56, exactly as planned.
;   Activation split confirms 7 vs 9: WMMA 1..16 read v0..v63 (A0),
;   WMMA 17..32 read v64..v127 (A1 loaded at source ID 831).
;
; STATIC TARGET CHECKS:
;   - no VGPR/SGPR spill; 346 VGPRs is inside the 1024 wave32 limit (doc :2250)
;   - A1 destinations v64..v127 are read by WMMA 17..32 -- not dead, not clobbered
;   - no WMMA reads a register that a later load in the same iteration overwrites
;   - accumulators (v128..v255, v322..v337) overlap no load destination
;   - after the final A1 load there is no DScnt wait before all 16 KSL0 WMMAs
;   - the only following DScnt wait is wait0, immediately before the 16 KSL1 WMMAs
;   - steady-state DScnt waits are exactly 12 and 0
;   Reading VGPR numbers here requires parsing the /*vNNN*/ comment: v256+ goes
;   through VGPR-MSB indexing (doc sec 3.3.2.3 :2249), so the bare 8-bit encoding
;   lies. A script that dropped the comment once invented a false overlap.
;
; This hand-edited target has not been runtime-benchmarked; measurements from the
; original compiler ordering do not validate this manually corrected schedule.
; ============================================================================
