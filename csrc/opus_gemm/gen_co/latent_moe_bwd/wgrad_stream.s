// SPDX-License-Identifier: MIT
    s_branch .Lrecord
    s_nop 0
.Lpipeline:
    .incbin "body.bin", 8, 13972
    s_mov_b64 exec, -1
    s_mov_b32 s16, 0x88
    s_lshl_b32 s14, s13, 5
    s_lshl_b32 s15, s13, 9
    v_and_b32_e32 v70, 15, v73
    v_and_b32_e32 v71, 64, v73
    v_lshrrev_b32_e32 v71, 2, v71
    v_add_u32_e32 v70, v70, v71
    v_mul_lo_u32 v70, s16, v70
    v_lshrrev_b32_e32 v71, 4, v73
    v_and_b32_e32 v71, 3, v71
    v_lshlrev_b32_e32 v71, 2, v71
    v_and_b32_e32 v72, 0x80, v73
    v_lshrrev_b32_e32 v72, 3, v72
    v_add_u32_e32 v71, v71, v72
    v_add_u32_e32 v70, v70, v71
    v_lshlrev_b32_e32 v70, 2, v70
    v_lshrrev_b32_e32 v71, 5, v73
    v_and_b32_e32 v72, 31, v73
    v_lshlrev_b32_e32 v72, 2, v72
    v_mul_lo_u32 v74, s16, v71
    v_add_u32_e32 v74, v74, v72
    v_lshlrev_b32_e32 v74, 2, v74
    v_mul_lo_u32 v75, s13, v71
    v_add_u32_e32 v75, v75, v72
    v_lshlrev_b32_e32 v75, 2, v75
    ds_write_b128 v70, a[128:131]
    ds_write_b128 v70, a[132:135] offset:128
    ds_write_b128 v70, a[136:139] offset:256
    ds_write_b128 v70, a[140:143] offset:384
    ds_write_b128 v70, a[144:147] offset:17408
    ds_write_b128 v70, a[148:151] offset:17536
    ds_write_b128 v70, a[152:155] offset:17664
    ds_write_b128 v70, a[156:159] offset:17792
    ds_write_b128 v70, a[160:163] offset:34816
    ds_write_b128 v70, a[164:167] offset:34944
    ds_write_b128 v70, a[168:171] offset:35072
    ds_write_b128 v70, a[172:175] offset:35200
    ds_write_b128 v70, a[176:179] offset:52224
    ds_write_b128 v70, a[180:183] offset:52352
    ds_write_b128 v70, a[184:187] offset:52480
    ds_write_b128 v70, a[188:191] offset:52608
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b128 v[0:3], v74
    ds_read_b128 v[4:7], v74 offset:4352
    ds_read_b128 v[8:11], v74 offset:8704
    ds_read_b128 v[12:15], v74 offset:13056
    ds_read_b128 v[16:19], v74 offset:17408
    ds_read_b128 v[20:23], v74 offset:21760
    ds_read_b128 v[24:27], v74 offset:26112
    ds_read_b128 v[28:31], v74 offset:30464
    ds_read_b128 v[32:35], v74 offset:34816
    ds_read_b128 v[36:39], v74 offset:39168
    ds_read_b128 v[40:43], v74 offset:43520
    ds_read_b128 v[44:47], v74 offset:47872
    ds_read_b128 v[48:51], v74 offset:52224
    ds_read_b128 v[52:55], v74 offset:56576
    ds_read_b128 v[56:59], v74 offset:60928
    ds_read_b128 v[60:63], v74 offset:65280
    s_mov_b32 s8, s36
    s_mov_b32 s9, s39
    v_mov_b32_e32 v76, v75
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v76, v[0:3], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[4:7], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[8:11], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[12:15], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[16:19], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[20:23], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[24:27], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[28:31], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[32:35], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[36:39], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[40:43], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[44:47], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[48:51], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[52:55], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[56:59], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[60:63], s[8:9] nt
    s_barrier
    ds_write_b128 v70, a[192:195]
    ds_write_b128 v70, a[196:199] offset:128
    ds_write_b128 v70, a[200:203] offset:256
    ds_write_b128 v70, a[204:207] offset:384
    ds_write_b128 v70, a[208:211] offset:17408
    ds_write_b128 v70, a[212:215] offset:17536
    ds_write_b128 v70, a[216:219] offset:17664
    ds_write_b128 v70, a[220:223] offset:17792
    ds_write_b128 v70, a[224:227] offset:34816
    ds_write_b128 v70, a[228:231] offset:34944
    ds_write_b128 v70, a[232:235] offset:35072
    ds_write_b128 v70, a[236:239] offset:35200
    ds_write_b128 v70, a[240:243] offset:52224
    ds_write_b128 v70, a[244:247] offset:52352
    ds_write_b128 v70, a[248:251] offset:52480
    ds_write_b128 v70, a[252:255] offset:52608
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b128 v[0:3], v74
    ds_read_b128 v[4:7], v74 offset:4352
    ds_read_b128 v[8:11], v74 offset:8704
    ds_read_b128 v[12:15], v74 offset:13056
    ds_read_b128 v[16:19], v74 offset:17408
    ds_read_b128 v[20:23], v74 offset:21760
    ds_read_b128 v[24:27], v74 offset:26112
    ds_read_b128 v[28:31], v74 offset:30464
    ds_read_b128 v[32:35], v74 offset:34816
    ds_read_b128 v[36:39], v74 offset:39168
    ds_read_b128 v[40:43], v74 offset:43520
    ds_read_b128 v[44:47], v74 offset:47872
    ds_read_b128 v[48:51], v74 offset:52224
    ds_read_b128 v[52:55], v74 offset:56576
    ds_read_b128 v[56:59], v74 offset:60928
    ds_read_b128 v[60:63], v74 offset:65280
    s_add_u32 s8, s36, s15
    s_addc_u32 s9, s39, 0
    v_mov_b32_e32 v76, v75
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v76, v[0:3], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[4:7], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[8:11], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[12:15], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[16:19], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[20:23], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[24:27], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[28:31], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[32:35], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[36:39], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[40:43], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[44:47], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[48:51], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[52:55], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[56:59], s[8:9] nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[60:63], s[8:9] nt
    s_barrier
    ds_write_b128 v70, v[128:131]
    ds_write_b128 v70, v[132:135] offset:128
    ds_write_b128 v70, v[136:139] offset:256
    ds_write_b128 v70, v[140:143] offset:384
    ds_write_b128 v70, v[144:147] offset:17408
    ds_write_b128 v70, v[148:151] offset:17536
    ds_write_b128 v70, v[152:155] offset:17664
    ds_write_b128 v70, v[156:159] offset:17792
    ds_write_b128 v70, v[160:163] offset:34816
    ds_write_b128 v70, v[164:167] offset:34944
    ds_write_b128 v70, v[168:171] offset:35072
    ds_write_b128 v70, v[172:175] offset:35200
    ds_write_b128 v70, v[176:179] offset:52224
    ds_write_b128 v70, v[180:183] offset:52352
    ds_write_b128 v70, v[184:187] offset:52480
    ds_write_b128 v70, v[188:191] offset:52608
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b128 v[0:3], v74
    ds_read_b128 v[4:7], v74 offset:4352
    ds_read_b128 v[8:11], v74 offset:8704
    ds_read_b128 v[12:15], v74 offset:13056
    ds_read_b128 v[16:19], v74 offset:17408
    ds_read_b128 v[20:23], v74 offset:21760
    ds_read_b128 v[24:27], v74 offset:26112
    ds_read_b128 v[28:31], v74 offset:30464
    ds_read_b128 v[32:35], v74 offset:34816
    ds_read_b128 v[36:39], v74 offset:39168
    ds_read_b128 v[40:43], v74 offset:43520
    ds_read_b128 v[44:47], v74 offset:47872
    ds_read_b128 v[48:51], v74 offset:52224
    ds_read_b128 v[52:55], v74 offset:56576
    ds_read_b128 v[56:59], v74 offset:60928
    ds_read_b128 v[60:63], v74 offset:65280
    s_mov_b32 s8, s36
    s_mov_b32 s9, s39
    v_mov_b32_e32 v76, v75
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v76, v[0:3], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[4:7], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[8:11], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[12:15], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[16:19], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[20:23], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[24:27], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[28:31], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[32:35], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[36:39], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[40:43], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[44:47], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[48:51], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[52:55], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[56:59], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[60:63], s[8:9] offset:512 nt
    s_barrier
    ds_write_b128 v70, a[0:3]
    ds_write_b128 v70, a[4:7] offset:128
    ds_write_b128 v70, a[8:11] offset:256
    ds_write_b128 v70, a[12:15] offset:384
    ds_write_b128 v70, a[16:19] offset:17408
    ds_write_b128 v70, a[20:23] offset:17536
    ds_write_b128 v70, a[24:27] offset:17664
    ds_write_b128 v70, a[28:31] offset:17792
    ds_write_b128 v70, a[32:35] offset:34816
    ds_write_b128 v70, a[36:39] offset:34944
    ds_write_b128 v70, a[40:43] offset:35072
    ds_write_b128 v70, a[44:47] offset:35200
    ds_write_b128 v70, v[240:243] offset:52224
    ds_write_b128 v70, v[244:247] offset:52352
    ds_write_b128 v70, v[248:251] offset:52480
    ds_write_b128 v70, v[252:255] offset:52608
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_read_b128 v[0:3], v74
    ds_read_b128 v[4:7], v74 offset:4352
    ds_read_b128 v[8:11], v74 offset:8704
    ds_read_b128 v[12:15], v74 offset:13056
    ds_read_b128 v[16:19], v74 offset:17408
    ds_read_b128 v[20:23], v74 offset:21760
    ds_read_b128 v[24:27], v74 offset:26112
    ds_read_b128 v[28:31], v74 offset:30464
    ds_read_b128 v[32:35], v74 offset:34816
    ds_read_b128 v[36:39], v74 offset:39168
    ds_read_b128 v[40:43], v74 offset:43520
    ds_read_b128 v[44:47], v74 offset:47872
    ds_read_b128 v[48:51], v74 offset:52224
    ds_read_b128 v[52:55], v74 offset:56576
    ds_read_b128 v[56:59], v74 offset:60928
    ds_read_b128 v[60:63], v74 offset:65280
    s_add_u32 s8, s36, s15
    s_addc_u32 s9, s39, 0
    v_mov_b32_e32 v76, v75
    s_waitcnt lgkmcnt(0)
    global_store_dwordx4 v76, v[0:3], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[4:7], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[8:11], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[12:15], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[16:19], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[20:23], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[24:27], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[28:31], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[32:35], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[36:39], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[40:43], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[44:47], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[48:51], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[52:55], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[56:59], s[8:9] offset:512 nt
    v_add_u32_e32 v76, s14, v76
    global_store_dwordx4 v76, v[60:63], s[8:9] offset:512 nt
    s_endpgm
    .incbin "body.bin", 15992, 4488
.Lrecord:
    // Adjacent workgroups visit the same output tile in a pair of experts.
    s_and_b32 s4, s2, 1
    s_lshr_b32 s5, s2, 1
    s_lshl_b32 s6, s3, 1
    s_add_u32 s3, s6, s4
    s_mov_b32 s2, s5
    s_load_dwordx2 s[4:5], s[0:1], 0x20
    s_lshl_b32 s6, s3, 4
    s_waitcnt lgkmcnt(0)
    s_load_dwordx4 s[8:11], s[4:5], s6
    s_waitcnt lgkmcnt(0)
    s_cmp_eq_u32 s10, 0
    s_cbranch_scc0 .Lactive
    s_cmp_lt_i32 s8, 0
    s_cbranch_scc1 .Ldone
    s_load_dwordx2 s[4:5], s[0:1], 0x10
    s_mul_i32 s6, s8, 0x2a00000
    s_mul_hi_u32 s7, s8, 0x2a00000
    s_waitcnt lgkmcnt(0)
    s_add_u32 s4, s4, s6
    s_addc_u32 s5, s5, s7
    s_lshl_b32 s6, s2, 18
    s_add_u32 s4, s4, s6
    s_addc_u32 s5, s5, 0
    v_lshlrev_b32_e32 v6, 4, v0
    v_mov_b32_e32 v5, 0
    v_mov_b32_e32 v2, 0
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, 0
    .rept 64
    global_store_dwordx4 v6, v[2:5], s[4:5] sc0 sc1
    v_add_u32_e32 v6, 0x1000, v6
    .endr
.Ldone:
    s_endpgm
.Lactive:
    s_mul_i32 s6, s3, 168
    s_add_u32 s2, s6, s2
    .incbin "body.bin", 0, 8
    s_branch .Lpipeline
