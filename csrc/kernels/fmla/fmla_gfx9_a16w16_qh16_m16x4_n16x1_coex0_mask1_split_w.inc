"  s_and_b32     s1, s1, 0x0000ffff                       \n" // 000000000000: 8601FF01 0000FFFF
"  ;s_load_dwordx2  [%[o_res_0], %[o_res_1]], s[0:1], 0x00                   \n" // 000000000008: C0060200 00000000
"  ;s_load_dwordx2  [%[lse_res_0], %[lse_res_1]], s[0:1], 0x10                 \n" // 000000000010: C0060300 00000010
"  ;s_load_dwordx2  [%[q_res_0], %[q_res_1]], s[0:1], 0x20                 \n" // 000000000018: C0060400 00000020
"  ;s_load_dwordx2  [%[kv_res_0], %[kv_res_1]], s[0:1], 0x30                 \n" // 000000000020: C0060500 00000030
"  ;s_load_dwordx2  [%[kv_indptr_res_0], %[kv_indptr_res_1]], s[0:1], 0x40                 \n" // 000000000028: C0060700 00000040
"  ;s_load_dwordx2  [%[kv_indices_res_0], %[kv_indices_res_1]], s[0:1], 0x50                 \n" // 000000000030: C0060600 00000050
"  ;s_load_dwordx2  s[30:31], s[0:1], 0x60                 \n" // 000000000038: C0060780 00000060
"  ;s_load_dword  s64, s[0:1], 0x70                        \n" // 000000000040: C0021000 00000070
"  ;s_load_dword  s65, s[0:1], 0x80                        \n" // 000000000048: C0021040 00000080
"  ;s_load_dword  s92, s[0:1], 0x90                        \n" // 000000000050: C0021700 00000090
"  ;s_load_dword  s66, s[0:1], 0xa0                        \n" // 000000000058: C0021080 000000A0
"  ;s_load_dword  %[s_page_block_size], s[0:1], 0xb0                        \n" // 000000000060: C0021100 000000B0
"  ;s_load_dword  s69, s[0:1], 0xc0                        \n" // 000000000068: C0021140 000000C0
"  ;s_load_dwordx2  [%[qo_res_0], %[qo_res_1]], s[0:1], 0xd0                 \n" // 000000000070: C0060800 000000D0
"  ;s_load_dwordx2  [%[kv_splits_res_0], %[kv_splits_res_1]], s[0:1], 0xe0                 \n" // 000000000078: C0061600 000000E0
"  v_lshrrev_b32  v1, 10, %[threadIdxx]                             \n" // 000000000080: 2002008A
"  v_lshrrev_b32  v2, 10, v1                              \n" // 000000000084: 2004028A
"  v_and_b32     v2, 0x000003ff, v2                       \n" // 000000000088: 260404FF 000003FF
"  v_and_b32     v1, 0x000003ff, v1                       \n" // 000000000090: 260202FF 000003FF
"  v_and_b32     %[threadIdxx], 0x000003ff, %[threadIdxx]                      \n" // 000000000098: 260000FF 000003FF
"  v_lshrrev_b32  v3, 6, %[threadIdxx]                              \n" // 0000000000A0: 20060086
"  v_and_b32     %[threadIdxx], 63, %[threadIdxx]                              \n" // 0000000000A4: 260000BF
"  s_mov_b32     %[blockIdxx], %[blockIdxx]                                   \n" // 0000000000A8: BE820002
"  s_mov_b32     %[blockIdxy], %[blockIdxy]                                   \n" // 0000000000AC: BE830003
"  s_mov_b32     %[blockIdxz], %[blockIdxz]                                   \n" // 0000000000B0: BE840004
" ;v_readfirstlane_b32  %[wave_id], v3                            \n" // 0000000000B4: 7E0E0503
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000000B8: BF8CC07F
"  s_mov_b32     s65, 16                                  \n" // 0000000000BC: BEC10090
"  s_mul_i32     s56, %[blockIdxy], 4                               \n" // 0000000000C0: 92388403
"  s_and_b32     %[kv_indptr_res_1], %[kv_indptr_res_1], 0x0000ffff                     \n" // 0000000000C4: 861DFF1D 0000FFFF
"  s_and_b32     s31, s31, 0x0000ffff                     \n" // 0000000000CC: 861FFF1F 0000FFFF
"  s_add_u32     %[kv_indptr_res_0], s56, %[kv_indptr_res_0]                            \n" // 0000000000D4: 801C1C38
"  s_addc_u32    %[kv_indptr_res_1], 0, %[kv_indptr_res_1]                              \n" // 0000000000D8: 821D1D80
"  s_load_dword  s47, [%[kv_indptr_res_0], %[kv_indptr_res_1]], 0x00                      \n" // 0000000000DC: C0020BCE 00000000
"  s_load_dword  s46, [%[kv_indptr_res_0], %[kv_indptr_res_1]], 0x04                      \n" // 0000000000E4: C0020B8E 00000004
"  s_mul_i32     s56, %[blockIdxy], 4                               \n" // 0000000000EC: 92388403
"  s_and_b32     %[kv_splits_res_1], %[kv_splits_res_1], 0x0000ffff                     \n" // 0000000000F0: 8659FF59 0000FFFF
"  s_add_u32     %[kv_splits_res_0], s56, %[kv_splits_res_0]                            \n" // 0000000000F8: 80585838
"  s_addc_u32    %[kv_splits_res_1], 0, %[kv_splits_res_1]                              \n" // 0000000000FC: 82595980
"  s_and_b32     %[qo_res_1], %[qo_res_1], 0x0000ffff                     \n" // 000000000100: 8621FF21 0000FFFF
"  s_add_u32     %[qo_res_0], s56, %[qo_res_0]                            \n" // 000000000108: 80202038
"  s_addc_u32    %[qo_res_1], 0, %[qo_res_1]                              \n" // 00000000010C: 82212180
"  s_load_dword  s86, [%[kv_splits_res_0], %[kv_splits_res_1]], 0x00                      \n" // 000000000110: C00215AC 00000000
"  s_load_dword  s85, [%[kv_splits_res_0], %[kv_splits_res_1]], 0x04                      \n" // 000000000118: C002156C 00000004
"  s_load_dword  s79, [%[qo_res_0], %[qo_res_1]], 0x00                      \n" // 000000000120: C00213D0 00000000
"  s_load_dword  s78, [%[qo_res_0], %[qo_res_1]], 0x04                      \n" // 000000000128: C0021390 00000004
"  s_mul_i32     s75, 0x00000800, s65                     \n" // 000000000130: 924B41FF 00000800
"  s_mul_i32     s74, 0x00000480, s65                     \n" // 000000000138: 924A41FF 00000480
"  s_mul_i32     s56, 4, s65                              \n" // 000000000140: 92384184
"  s_mov_b32     %[o_res_2], s75                                 \n" // 000000000144: BE8A004B
"  s_mov_b32     %[q_res_2], -16                                 \n" // 000000000148: BE9200D0
"  s_mov_b32     %[lse_res_2], -16                                 \n" // 00000000014C: BE8E00D0
"  s_mov_b32     %[kv_res_2], -16                                 \n" // 000000000150: BE9600D0
"  s_mov_b32     %[kv_indices_res_2], -16                                 \n" // 000000000154: BE9A00D0
"  s_mov_b32     %[o_res_3], 0x00020000                          \n" // 000000000158: BE8B00FF 00020000
"  s_mov_b32     %[q_res_3], 0x00020000                          \n" // 000000000160: BE9300FF 00020000
"  s_mov_b32     %[lse_res_3], 0x00020000                          \n" // 000000000168: BE8F00FF 00020000
"  s_mov_b32     %[kv_res_3], 0x00020000                          \n" // 000000000170: BE9700FF 00020000
"  s_mov_b32     %[kv_indices_res_3], 0x00020000                          \n" // 000000000178: BE9B00FF 00020000
"  s_and_b32     %[o_res_1], %[o_res_1], 0x0000ffff                       \n" // 000000000180: 8609FF09 0000FFFF
"  s_and_b32     %[q_res_1], %[q_res_1], 0x0000ffff                     \n" // 000000000188: 8611FF11 0000FFFF
"  s_and_b32     %[lse_res_1], %[lse_res_1], 0x0000ffff                     \n" // 000000000190: 860DFF0D 0000FFFF
"  s_and_b32     %[kv_res_1], %[kv_res_1], 0x0000ffff                     \n" // 000000000198: 8615FF15 0000FFFF
"  s_and_b32     %[kv_indices_res_1], %[kv_indices_res_1], 0x0000ffff                     \n" // 0000000001A0: 8619FF19 0000FFFF
"  s_or_b32      %[o_res_1], %[o_res_1], 0x00040000                       \n" // 0000000001A8: 8709FF09 00040000
"  s_or_b32      %[q_res_1], %[q_res_1], 0x00040000                     \n" // 0000000001B0: 8711FF11 00040000
"  s_or_b32      %[lse_res_1], %[lse_res_1], 0x00040000                     \n" // 0000000001B8: 870DFF0D 00040000
"  s_or_b32      %[kv_res_1], %[kv_res_1], 0x00040000                     \n" // 0000000001C0: 8715FF15 00040000
"  s_or_b32      %[kv_indices_res_1], %[kv_indices_res_1], 0x00040000                     \n" // 0000000001C8: 8719FF19 00040000
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000001D0: BF8CC07F
"  s_mul_i32     s80, %[blockIdxx], 8                               \n" // 0000000001D4: 92508802
"  s_sub_u32     s81, s78, s79                            \n" // 0000000001D8: 80D14F4E
"  s_sub_u32     s87, s85, s86                            \n" // 0000000001DC: 80D75655
"  s_cmp_le_u32  s81, s80                                 \n" // 0000000001E0: BF0B5051
"  s_cbranch_scc1  label_1971_%=                             \n" // 0000000001E4: BF8518F7
"  s_cmp_le_u32  s87, %[blockIdxz]                                  \n" // 0000000001E8: BF0B0457
"  s_cbranch_scc1  label_1971_%=                             \n" // 0000000001EC: BF8518F5
"  s_mov_b32     %[num_splits], s87                                 \n" // 0000000001F0: BEC30057
"  s_mov_b32     s69, 0                                   \n" // 0000000001F4: BEC50080
"  s_lshr_b32    s44, 16, s69                             \n" // 0000000001F8: 8F2C4590
"  s_mul_i32     s73, s44, 4                              \n" // 0000000001FC: 9249842C
"  s_mul_i32     s73, s73, %[num_splits]                            \n" // 000000000200: 92494349
"  s_mul_i32     s45, %[blockIdxz], s44                   \n" // 000000000204: 922D2C04
"  s_sub_u32     s50, s46, s47                            \n" // 000000000208: 80B22F2E
"  s_lshl_b32    s56, s50, s69                            \n" // 00000000020C: 8E384532
"  s_sub_u32     s82, s56, s81                            \n" // 000000000210: 80D25138
"  s_mul_i32     s58, %[blockIdxx], 8                     \n" // 000000000214: 923A8802
"  s_add_u32     s82, s82, s58                            \n" // 000000000218: 80523A52
"  s_add_u32     s57, s82, 8                              \n" // 00000000021C: 80398852
"  s_min_u32     s56, s56, s57                            \n" // 000000000220: 83B83938
"  s_lshr_b32    s50, s56, s69                            \n" // 000000000224: 8F324538
"  s_lshl_b32    s56, s45, s69                            \n" // 000000000228: 8E38452D
"  s_add_u32     s83, s56, 15                             \n" // 00000000022C: 80538F38
"  s_mul_i32     s84, %[num_splits], 16                             \n" // 000000000230: 92549043
"  s_cmp_le_u32  s50, s45                                 \n" // 000000000234: BF0B2D32
"  s_cbranch_scc1  label_1971_%=                             \n" // 000000000238: BF8518E2
"  s_mul_i32     s56, s50, 4                              \n" // 00000000023C: 92388432
"  s_mov_b32     %[kv_indices_res_2], s56                                 \n" // 000000000240: BE9A0038
"  s_mul_i32     s56, s47, 4                              \n" // 000000000244: 9238842F
"  s_add_u32     %[kv_indices_res_0], s56, %[kv_indices_res_0]                            \n" // 000000000248: 80181838
"  s_addc_u32    %[kv_indices_res_1], 0, %[kv_indices_res_1]                              \n" // 00000000024C: 82191980
"  s_mov_b32     s70, 0                                   \n" // 000000000250: BEC60080
"  s_sub_u32     s71, s50, s45                            \n" // 000000000254: 80C72D32
"  s_mul_i32     s39, %[num_splits], s44                            \n" // 000000000258: 92272C43
"  s_mov_b32     s38, s71                                 \n" // 00000000025C: BEA60047
"  v_cvt_f32_u32  v20, s39                                \n" // 000000000260: 7E280C27
"  s_sub_i32     s56, 0, s39                              \n" // 000000000264: 81B82780
"  v_rcp_iflag_f32  v20, v20                              \n" // 000000000268: 7E284714
"  s_nop         0x0000                                   \n" // 00000000026C: BF800000
"  v_mul_f32     v20, 0x4f7ffffe, v20                     \n" // 000000000270: 0A2828FF 4F7FFFFE
"  v_cvt_u32_f32  v20, v20                                \n" // 000000000278: 7E280F14
"  v_mul_lo_u32  v21, s56, v20                            \n" // 00000000027C: D2850015 00022838
"  v_mul_hi_u32  v21, v20, v21                            \n" // 000000000284: D2860015 00022B14
"  v_add_u32     v20, v20, v21                            \n" // 00000000028C: 68282B14
"  v_mul_hi_u32  v20, s38, v20                            \n" // 000000000290: D2860014 00022826
"  v_mul_lo_u32  v21, v20, s39                            \n" // 000000000298: D2850015 00004F14
"  v_sub_u32     v23, s38, v21                            \n" // 0000000002A0: 6A2E2A26
"  v_add_u32     v22, 1, v20                              \n" // 0000000002A4: 682C2881
"  v_cmp_le_u32  vcc, s39, v23                            \n" // 0000000002A8: 7D962E27
"  v_subrev_u32  v21, s39, v23                            \n" // 0000000002AC: 6C2A2E27
"  s_nop         0x0000                                   \n" // 0000000002B0: BF800000
"  v_cndmask_b32  v20, v20, v22, vcc                      \n" // 0000000002B4: 00282D14
"  v_cndmask_b32  v23, v23, v21, vcc                      \n" // 0000000002B8: 002E2B17
"  v_add_u32     v21, 1, v20                              \n" // 0000000002BC: 682A2881
"  v_cmp_le_u32  vcc, s39, v23                            \n" // 0000000002C0: 7D962E27
"  s_nop         0x0001                                   \n" // 0000000002C4: BF800001
"  v_cndmask_b32  v23, v20, v21, vcc                      \n" // 0000000002C8: 002E2B14
"  s_nop         0x0003                                   \n" // 0000000002CC: BF800003
"  v_readfirstlane_b32  s40, v23                          \n" // 0000000002D0: 7E500517
"  s_nop         0x0003                                   \n" // 0000000002D4: BF800003
"  s_mov_b32     s71, s40                                 \n" // 0000000002D8: BEC70028
"  s_mul_i32     s56, s71, s39                            \n" // 0000000002DC: 92382747
"  s_sub_u32     s56, s38, s56                            \n" // 0000000002E0: 80B83826
"  s_mov_b32     s57, 0                                   \n" // 0000000002E4: BEB90080
"  s_cmp_lt_u32  s56, s44                                 \n" // 0000000002E8: BF0A2C38
"  s_cselect_b32  s57, s57, 1                             \n" // 0000000002EC: 85398139
"  s_add_u32     s71, s57, s71                            \n" // 0000000002F0: 80474739
"  s_cmpk_eq_u32  s57, 0x0001                             \n" // 0000000002F4: B4390001
"  s_cselect_b32  s49, 0, s56                             \n" // 0000000002F8: 85313880
"  s_mov_b32     s48, s49                                 \n" // 0000000002FC: BEB00031
"  v_lshrrev_b32  v20, 3, %[threadIdxx]                             \n" // 000000000300: 20280083
"  v_and_b32     v21, 1, v20                              \n" // 000000000304: 262A2881
"  v_lshlrev_b32  v8, 3, v21                              \n" // 000000000308: 24102A83
"  v_and_b32     v20, 4, v20                              \n" // 00000000030C: 26282884
"  v_add_u32     v8, v8, v20                              \n" // 000000000310: 68102908
"  v_add_u32     v8, v8, %[wave_id]                               \n" // 000000000314: D1340008 00000F08
"  v_add_u32     v8, s45, v8                              \n" // 00000000031C: 6810102D
"  v_lshlrev_b32  v8, 2, v8                               \n" // 000000000320: 24101082
"  buffer_load_dword  v10, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 000000000324: E0501000 80060A08
"  v_add_u32     v8, s73, v8                              \n" // 00000000032C: 68101049
"  buffer_load_dword  v11, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 000000000330: E0501000 80060B08
"  v_add_u32     v8, s73, v8                              \n" // 000000000338: 68101049
"  s_add_u32     s56, s80, s79                            \n" // 00000000033C: 80384F50
"  v_mov_b32     v20, s56                                 \n" // 000000000340: 7E280238
"  v_mul_lo_u32  v21, s74, v20                            \n" // 000000000344: D2850015 0002284A
"  v_mul_hi_u32  v22, s74, v20                            \n" // 00000000034C: D2860016 0002284A
"  s_nop         0x0002                                   \n" // 000000000354: BF800002
"  v_readfirstlane_b32  s56, v21                          \n" // 000000000358: 7E700515
"  v_readfirstlane_b32  s57, v22                          \n" // 00000000035C: 7E720516
"  s_nop         0x0004                                   \n" // 000000000360: BF800004
"  s_add_u32     %[q_res_0], s56, %[q_res_0]                            \n" // 000000000364: 80101038
"  s_addc_u32    %[q_res_1], s57, %[q_res_1]                            \n" // 000000000368: 82111139
"  s_sub_u32     s56, s81, s80                            \n" // 00000000036C: 80B85051
"  s_mul_i32     s56, s56, s74                            \n" // 000000000370: 92384A38
"  s_mov_b32     %[q_res_2], s56                                 \n" // 000000000374: BE920038
"  s_mul_i32     s56, %[wave_id], 0x00000480                      \n" // 000000000378: 9238FF07 00000480
"  v_lshlrev_b32  v30, 2, %[threadIdxx]                             \n" // 000000000380: 243C0082
"  v_add_u32     v30, s56, v30                            \n" // 000000000384: 683C3C38
"  s_mul_i32     s56, %[wave_id], 0x00001420                      \n" // 000000000388: 9238FF07 00001420
"  s_add_u32     s34, 0, s56                              \n" // 000000000390: 80223880
"  s_add_u32     s35, 0x00005080, s34                     \n" // 000000000394: 802322FF 00005080
"  s_add_u32     s36, 0x00005080, s35                     \n" // 00000000039C: 802423FF 00005080
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 0000000003A4: 20280084
"  v_lshlrev_b32  v21, 2, v20                             \n" // 0000000003A8: 242A2882
"  v_and_b32     v20, 15, %[threadIdxx]                             \n" // 0000000003AC: 2628008F
"  v_lshrrev_b32  v22, 2, v20                             \n" // 0000000003B0: 202C2882
"  v_mul_i32_i24  v22, 0x00000140, v22                    \n" // 0000000003B4: 0C2C2CFF 00000140
"  v_add_u32     v21, v22, v21                            \n" // 0000000003BC: 682A2B16
"  v_and_b32     v20, 3, %[threadIdxx]                              \n" // 0000000003C0: 26280083
"  v_mul_i32_i24  v22, 0x00000508, v20                    \n" // 0000000003C4: 0C2C28FF 00000508
"  v_add_u32     v21, v22, v21                            \n" // 0000000003CC: 682A2B16
"  v_lshlrev_b32  v29, 2, v21                             \n" // 0000000003D0: 243A2A82
"  s_mov_b32     m0, s34                                  \n" // 0000000003D4: BEFC0022
"  ;v_add_u32     v28, 0, v30                              \n" // 0000000003D8: 68383C80
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 0000000003DC: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 0000000003E4: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 0000000003EC: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 0000000003F4: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 0000000003FC: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000404: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 00000000040C: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000414: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 00000000041C: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000424: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 00000000042C: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000434: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 00000000043C: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 000000000444: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 00000000044C: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 000000000454: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 00000000045C: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 000000000464: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 00000000046C: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000474: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 00000000047C: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000484: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 00000000048C: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000494: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 00000000049C: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 0000000004A4: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 0000000004AC: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 0000000004B4: 683838FF 00001200
"  ;s_mov_b32     m0, s35                                  \n" // 0000000004BC: BEFC0023
"  ;v_add_u32     v28, 0x00004800, v30                     \n" // 0000000004C0: 68383CFF 00004800
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 0000000004C8: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 0000000004D0: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 0000000004D8: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 0000000004E0: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 0000000004E8: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 0000000004F0: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 0000000004F8: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000500: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 000000000508: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000510: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 000000000518: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000520: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000528: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 000000000530: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000538: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 000000000540: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000548: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 000000000550: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000558: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000560: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 000000000568: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000570: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 000000000578: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000580: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 000000000588: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000590: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000598: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 0000000005A0: 683838FF 00001200
"  ;s_waitcnt     vmcnt(20)                                \n" // 0000000005A8: BF8C4F74
"  ;s_barrier                                              \n" // 0000000005AC: BF8A0000
"  ;s_mov_b32     m0, s36                                  \n" // 0000000005B0: BEFC0024
"  ;v_add_u32     v28, 0x00009000, v30                     \n" // 0000000005B4: 68383CFF 00009000
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 0000000005BC: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 0000000005C4: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 0000000005CC: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 0000000005D4: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 0000000005DC: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 0000000005E4: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 0000000005EC: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 0000000005F4: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 0000000005FC: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000604: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 00000000060C: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000614: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 00000000061C: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 000000000624: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 00000000062C: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 000000000634: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 00000000063C: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 000000000644: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 00000000064C: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000654: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 00000000065C: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000664: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 00000000066C: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000674: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 00000000067C: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000684: E0511400 8004001C
"  s_add_u32     m0, m0, 0x00000500                       \n" // 00000000068C: 807CFF7C 00000500
"  v_add_u32     v28, 0x00001200, v28                     \n" // 000000000694: 683838FF 00001200
"  s_cmp_eq_i32  %[wave_id], 0                                    \n" // 00000000069C: BF008007
"  s_cbranch_scc0  label_01CE_%=                             \n" // 0000000006A0: BF840025
"  ds_read_b128  acc[0:3], v29                            \n" // 0000000006A4: DBFE0000 0000001D
"  ds_read_b128  acc[4:7], v29 offset:64                  \n" // 0000000006AC: DBFE0040 0400001D
"  ds_read_b128  acc[8:11], v29 offset:128                \n" // 0000000006B4: DBFE0080 0800001D
"  ds_read_b128  acc[12:15], v29 offset:192               \n" // 0000000006BC: DBFE00C0 0C00001D
"  ds_read_b128  acc[16:19], v29 offset:256               \n" // 0000000006C4: DBFE0100 1000001D
"  ds_read_b128  acc[20:23], v29 offset:320               \n" // 0000000006CC: DBFE0140 1400001D
"  ds_read_b128  acc[24:27], v29 offset:384               \n" // 0000000006D4: DBFE0180 1800001D
"  ds_read_b128  acc[28:31], v29 offset:448               \n" // 0000000006DC: DBFE01C0 1C00001D
"  ds_read_b128  acc[32:35], v29 offset:512               \n" // 0000000006E4: DBFE0200 2000001D
"  ds_read_b128  acc[36:39], v29 offset:576               \n" // 0000000006EC: DBFE0240 2400001D
"  ds_read_b128  acc[40:43], v29 offset:640               \n" // 0000000006F4: DBFE0280 2800001D
"  ds_read_b128  acc[44:47], v29 offset:704               \n" // 0000000006FC: DBFE02C0 2C00001D
"  ds_read_b128  acc[48:51], v29 offset:768               \n" // 000000000704: DBFE0300 3000001D
"  ds_read_b128  acc[52:55], v29 offset:832               \n" // 00000000070C: DBFE0340 3400001D
"  ds_read_b128  acc[56:59], v29 offset:896               \n" // 000000000714: DBFE0380 3800001D
"  ds_read_b128  acc[60:63], v29 offset:960               \n" // 00000000071C: DBFE03C0 3C00001D
"  ds_read_b128  acc[64:67], v29 offset:1024              \n" // 000000000724: DBFE0400 4000001D
"  ds_read_b128  acc[68:71], v29 offset:1088              \n" // 00000000072C: DBFE0440 4400001D
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000000734: BF8CC07F
"label_01CE_%=: \n"
"  s_waitcnt     vmcnt(20)                                \n" // 000000000738: BF8C4F74
"  s_barrier                                              \n" // 00000000073C: BF8A0000
"  s_mov_b32     m0, s34                                  \n" // 000000000740: BEFC0022
"  ;v_add_u32     v28, 0x0000d800, v30                     \n" // 000000000744: 68383CFF 0000D800
"  ;v_add_u32     v28, 0, 0                                \n" // 0000000003D8: 68383C80
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 00000000074C: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 000000000754: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 00000000075C: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 000000000764: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 00000000076C: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 000000000774: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 00000000077C: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 000000000784: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 00000000078C: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000794: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 00000000079C: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 0000000007A4: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 0000000007AC: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 0000000007B4: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 0000000007BC: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 0000000007C4: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 0000000007CC: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 0000000007D4: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 0000000007DC: E0511400 8004001C
"  ;s_add_u32     m0, m0, 0x00000500                       \n" // 0000000007E4: 807CFF7C 00000500
"  ;v_add_u32     v28, 0x00001200, v28                     \n" // 0000000007EC: 683838FF 00001200
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen lds      \n" // 0000000007F4: E0511000 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:256 lds  \n" // 0000000007FC: E0511100 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:512 lds  \n" // 000000000804: E0511200 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:768 lds  \n" // 00000000080C: E0511300 8004001C
"  ;buffer_load_dword   v28, [%[q_res_0], %[q_res_1], %[q_res_2], %[q_res_3]], 0 offen offset:1024 lds  \n" // 000000000814: E0511400 8004001C
"  s_add_u32     m0, m0, 0x00000500                       \n" // 00000000081C: 807CFF7C 00000500
"  v_add_u32     v28, 0x00001200, v28                     \n" // 000000000824: 683838FF 00001200
"  s_cmp_eq_i32  %[wave_id], 1                                    \n" // 00000000082C: BF008107
"  s_cbranch_scc0  label_0232%=                             \n" // 000000000830: BF840025
"  ds_read_b128  acc[0:3], v29 offset:20608               \n" // 000000000834: DBFE5080 0000001D
"  ds_read_b128  acc[4:7], v29 offset:20672               \n" // 00000000083C: DBFE50C0 0400001D
"  ds_read_b128  acc[8:11], v29 offset:20736              \n" // 000000000844: DBFE5100 0800001D
"  ds_read_b128  acc[12:15], v29 offset:20800             \n" // 00000000084C: DBFE5140 0C00001D
"  ds_read_b128  acc[16:19], v29 offset:20864             \n" // 000000000854: DBFE5180 1000001D
"  ds_read_b128  acc[20:23], v29 offset:20928             \n" // 00000000085C: DBFE51C0 1400001D
"  ds_read_b128  acc[24:27], v29 offset:20992             \n" // 000000000864: DBFE5200 1800001D
"  ds_read_b128  acc[28:31], v29 offset:21056             \n" // 00000000086C: DBFE5240 1C00001D
"  ds_read_b128  acc[32:35], v29 offset:21120             \n" // 000000000874: DBFE5280 2000001D
"  ds_read_b128  acc[36:39], v29 offset:21184             \n" // 00000000087C: DBFE52C0 2400001D
"  ds_read_b128  acc[40:43], v29 offset:21248             \n" // 000000000884: DBFE5300 2800001D
"  ds_read_b128  acc[44:47], v29 offset:21312             \n" // 00000000088C: DBFE5340 2C00001D
"  ds_read_b128  acc[48:51], v29 offset:21376             \n" // 000000000894: DBFE5380 3000001D
"  ds_read_b128  acc[52:55], v29 offset:21440             \n" // 00000000089C: DBFE53C0 3400001D
"  ds_read_b128  acc[56:59], v29 offset:21504             \n" // 0000000008A4: DBFE5400 3800001D
"  ds_read_b128  acc[60:63], v29 offset:21568             \n" // 0000000008AC: DBFE5440 3C00001D
"  ds_read_b128  acc[64:67], v29 offset:21632             \n" // 0000000008B4: DBFE5480 4000001D
"  ds_read_b128  acc[68:71], v29 offset:21696             \n" // 0000000008BC: DBFE54C0 4400001D
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000008C4: BF8CC07F
"label_0232%=: \n"
"  s_waitcnt     vmcnt(20)                                \n" // 0000000008C8: BF8C4F74
"  s_barrier                                              \n" // 0000000008CC: BF8A0000
"  s_cmp_eq_i32  %[wave_id], 2                                    \n" // 0000000008D0: BF008207
"  s_cbranch_scc0  label_025B_%=                             \n" // 0000000008D4: BF840025
"  ds_read_b128  acc[0:3], v29 offset:41216               \n" // 0000000008D8: DBFEA100 0000001D
"  ds_read_b128  acc[4:7], v29 offset:41280               \n" // 0000000008E0: DBFEA140 0400001D
"  ds_read_b128  acc[8:11], v29 offset:41344              \n" // 0000000008E8: DBFEA180 0800001D
"  ds_read_b128  acc[12:15], v29 offset:41408             \n" // 0000000008F0: DBFEA1C0 0C00001D
"  ds_read_b128  acc[16:19], v29 offset:41472             \n" // 0000000008F8: DBFEA200 1000001D
"  ds_read_b128  acc[20:23], v29 offset:41536             \n" // 000000000900: DBFEA240 1400001D
"  ds_read_b128  acc[24:27], v29 offset:41600             \n" // 000000000908: DBFEA280 1800001D
"  ds_read_b128  acc[28:31], v29 offset:41664             \n" // 000000000910: DBFEA2C0 1C00001D
"  ds_read_b128  acc[32:35], v29 offset:41728             \n" // 000000000918: DBFEA300 2000001D
"  ds_read_b128  acc[36:39], v29 offset:41792             \n" // 000000000920: DBFEA340 2400001D
"  ds_read_b128  acc[40:43], v29 offset:41856             \n" // 000000000928: DBFEA380 2800001D
"  ds_read_b128  acc[44:47], v29 offset:41920             \n" // 000000000930: DBFEA3C0 2C00001D
"  ds_read_b128  acc[48:51], v29 offset:41984             \n" // 000000000938: DBFEA400 3000001D
"  ds_read_b128  acc[52:55], v29 offset:42048             \n" // 000000000940: DBFEA440 3400001D
"  ds_read_b128  acc[56:59], v29 offset:42112             \n" // 000000000948: DBFEA480 3800001D
"  ds_read_b128  acc[60:63], v29 offset:42176             \n" // 000000000950: DBFEA4C0 3C00001D
"  ds_read_b128  acc[64:67], v29 offset:42240             \n" // 000000000958: DBFEA500 4000001D
"  ds_read_b128  acc[68:71], v29 offset:42304             \n" // 000000000960: DBFEA540 4400001D
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000000968: BF8CC07F
"label_025B_%=: \n"
"  s_waitcnt     vmcnt(0)                                 \n" // 00000000096C: BF8C0F70
"  s_barrier                                              \n" // 000000000970: BF8A0000
"  s_cmp_eq_i32  %[wave_id], 3                                    \n" // 000000000974: BF008307
"  s_cbranch_scc0  label_0284_%=                             \n" // 000000000978: BF840025
"  ds_read_b128  acc[0:3], v29                            \n" // 00000000097C: DBFE0000 0000001D
"  ds_read_b128  acc[4:7], v29 offset:64                  \n" // 000000000984: DBFE0040 0400001D
"  ds_read_b128  acc[8:11], v29 offset:128                \n" // 00000000098C: DBFE0080 0800001D
"  ds_read_b128  acc[12:15], v29 offset:192               \n" // 000000000994: DBFE00C0 0C00001D
"  ds_read_b128  acc[16:19], v29 offset:256               \n" // 00000000099C: DBFE0100 1000001D
"  ds_read_b128  acc[20:23], v29 offset:320               \n" // 0000000009A4: DBFE0140 1400001D
"  ds_read_b128  acc[24:27], v29 offset:384               \n" // 0000000009AC: DBFE0180 1800001D
"  ds_read_b128  acc[28:31], v29 offset:448               \n" // 0000000009B4: DBFE01C0 1C00001D
"  ds_read_b128  acc[32:35], v29 offset:512               \n" // 0000000009BC: DBFE0200 2000001D
"  ds_read_b128  acc[36:39], v29 offset:576               \n" // 0000000009C4: DBFE0240 2400001D
"  ds_read_b128  acc[40:43], v29 offset:640               \n" // 0000000009CC: DBFE0280 2800001D
"  ds_read_b128  acc[44:47], v29 offset:704               \n" // 0000000009D4: DBFE02C0 2C00001D
"  ds_read_b128  acc[48:51], v29 offset:768               \n" // 0000000009DC: DBFE0300 3000001D
"  ds_read_b128  acc[52:55], v29 offset:832               \n" // 0000000009E4: DBFE0340 3400001D
"  ds_read_b128  acc[56:59], v29 offset:896               \n" // 0000000009EC: DBFE0380 3800001D
"  ds_read_b128  acc[60:63], v29 offset:960               \n" // 0000000009F4: DBFE03C0 3C00001D
"  ds_read_b128  acc[64:67], v29 offset:1024              \n" // 0000000009FC: DBFE0400 4000001D
"  ds_read_b128  acc[68:71], v29 offset:1088              \n" // 000000000A04: DBFE0440 4400001D
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000000A0C: BF8CC07F
"label_0284_%=: \n"
"  s_waitcnt     vmcnt(0)                                 \n" // 000000000A10: BF8C0F70
"  s_barrier                                              \n" // 000000000A14: BF8A0000
"  s_mov_b32     s52, 0x07060302                          \n" // 000000000A18: BEB400FF 07060302
"  s_mov_b32     s53, 0x05040100                          \n" // 000000000A20: BEB500FF 05040100
"  s_mov_b32     s6, 0x3fb8aa3b                           \n" // 000000000A28: BE8600FF 3FB8AA3B
"  v_mov_b32     v21, s6                                  \n" // 000000000A30: 7E2A0206
"  v_mov_b32     v20, s64                                 \n" // 000000000A34: 7E280240
"  v_mul_f32     v20, s6, v20                             \n" // 000000000A38: 0A282806
"  v_rcp_f32     v21, v21                                 \n" // 000000000A3C: 7E2A4515
"  v_mov_b32     v12, 0xff7fffff                          \n" // 000000000A40: 7E1802FF FF7FFFFF
"  v_mov_b32     v13, 0xff7fffff                          \n" // 000000000A48: 7E1A02FF FF7FFFFF
"  v_mov_b32     v16, 0                                   \n" // 000000000A50: 7E200280
"  v_mov_b32     v17, 0                                   \n" // 000000000A54: 7E220280
"  v_mov_b32     v14, 0                                   \n" // 000000000A58: 7E1C0280
"  v_mov_b32     v15, 0                                   \n" // 000000000A5C: 7E1E0280
"  v_mov_b32     v9, %[s_page_block_size]                 \n" // 000000000A60: 7E120244
"  v_mov_b32     v181, %[s_rope_page_block_size]          \n" // 000000000A60: 7E120244
"  v_readfirstlane_b32  s5, v20                           \n" // 000000000A64: 7E0A0514
"  v_readfirstlane_b32  s63, v21                          \n" // 000000000A68: 7E7E0515
"  v_and_b32     v2, 15, %[threadIdxx]                              \n" // 000000000A6C: 2604008F
"  v_lshlrev_b32  v2, 2, v2                               \n" // 000000000A70: 24040482
"  s_mul_i32     s56, 0x00000100, %[wave_id]                      \n" // 000000000A74: 923807FF 00000100
"  v_add_u32     v2, s56, v2                              \n" // 000000000A7C: 68040438
"  v_lshlrev_b32  v3, 2, %[threadIdxx]                              \n" // 000000000A80: 24060082
"  s_mul_i32     s56, 0x00000100, %[wave_id]                      \n" // 000000000A84: 923807FF 00000100
"  v_add_u32     v3, s56, v3                              \n" // 000000000A8C: 68060638
"  v_and_b32     v20, 31, %[threadIdxx]                             \n" // 000000000A90: 2628009F
"  v_lshlrev_b32  v1, 2, v20                              \n" // 000000000A94: 24022882
"  s_mul_i32     s34, %[wave_id], 0x00001220                      \n" // 000000000A98: 9222FF07 00001220
"  s_add_u32     s34, 0, s34                              \n" // 000000000AA0: 80222280
"  s_add_u32     s35, 0x00000900, s34                     \n" // 000000000AA4: 802322FF 00000900
"  s_add_u32     s36, 0x00004880, s34                     \n" // 000000000AAC: 802422FF 00004880
"  s_add_u32     s37, 0x00004880, s35                     \n" // 000000000AB4: 802523FF 00004880
"  s_waitcnt     vmcnt(0)                                 \n" // 000000000ABC: BF8C0F70
"  v_mul_u32_u24  v18, v10, v9 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v19, v10, v9 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v18, v18, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 000000000AD4: 68260313
"  v_mul_u32_u24  v182, v10, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v10, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s34                                  \n" // 000000000AD8: BEFC0022
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000000ADC: E0511000 80050012
"  s_add_u32     m0, 0, s35                               \n" // 000000000AE4: 807C2380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000000AE8: E0511000 80050013
"  s_add_u32     m0, 0x00000080, s34                      \n" // 000000000AF0: 807C22FF 00000080
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000000AF8: E0511080 80050012
"  s_add_u32     m0, 0x00000080, s35                      \n" // 000000000B00: 807C23FF 00000080
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000000B08: E0511080 80050013
"  s_add_u32     m0, 0x00000100, s34                      \n" // 000000000B10: 807C22FF 00000100
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000000B18: E0511100 80050012
"  s_add_u32     m0, 0x00000100, s35                      \n" // 000000000B20: 807C23FF 00000100
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000000B28: E0511100 80050013
"  s_add_u32     m0, 0x00000180, s34                      \n" // 000000000B30: 807C22FF 00000180
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 000000000B38: E0511180 80050012
"  s_add_u32     m0, 0x00000180, s35                      \n" // 000000000B40: 807C23FF 00000180
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 000000000B48: E0511180 80050013
"  s_add_u32     m0, 0x00000200, s34                      \n" // 000000000B50: 807C22FF 00000200
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000000B58: E0511200 80050012
"  s_add_u32     m0, 0x00000200, s35                      \n" // 000000000B60: 807C23FF 00000200
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000000B68: E0511200 80050013
"  s_add_u32     m0, 0x00000280, s34                      \n" // 000000000B70: 807C22FF 00000280
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000000B78: E0511280 80050012
"  s_add_u32     m0, 0x00000280, s35                      \n" // 000000000B80: 807C23FF 00000280
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000000B88: E0511280 80050013
"  s_add_u32     m0, 0x00000300, s34                      \n" // 000000000B90: 807C22FF 00000300
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000000B98: E0511300 80050012
"  s_add_u32     m0, 0x00000300, s35                      \n" // 000000000BA0: 807C23FF 00000300
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000000BA8: E0511300 80050013
"  s_add_u32     m0, 0x00000380, s34                      \n" // 000000000BB0: 807C22FF 00000380
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000000BB8: E0511380 80050012
"  s_add_u32     m0, 0x00000380, s35                      \n" // 000000000BC0: 807C23FF 00000380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000000BC8: E0511380 80050013
"  s_add_u32     m0, 0x00000400, s34                      \n" // 000000000BD0: 807C22FF 00000400
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v182, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000400, s35                      \n" // 000000000BE0: 807C23FF 00000400
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v183, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BE8: E0511400 80050013
"  ;s_add_u32     m0, 0x00000480, s34                      \n" // 000000000BF0: 807C22FF 00000480
"  buffer_load_dword  v10, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 000000000BF8: E0501000 80060A08
"  v_add_u32     v8, s73, v8                              \n" // 000000000C00: 68101049
"  v_mov_b32     %[o_regs_0], 0                                   \n" // 000000000C04: 7E500280
"  v_mov_b32     %[o_regs_1], 0                                   \n" // 000000000C08: 7E520280
"  v_mov_b32     %[o_regs_2], 0                                   \n" // 000000000C0C: 7E540280
"  v_mov_b32     %[o_regs_3], 0                                   \n" // 000000000C10: 7E560280
"  v_mov_b32     %[o_regs_4], 0                                   \n" // 000000000C14: 7E580280
"  v_mov_b32     %[o_regs_5], 0                                   \n" // 000000000C18: 7E5A0280
"  v_mov_b32     %[o_regs_6], 0                                   \n" // 000000000C1C: 7E5C0280
"  v_mov_b32     %[o_regs_7], 0                                   \n" // 000000000C20: 7E5E0280
"  v_mov_b32     %[o_regs_8], 0                                   \n" // 000000000C24: 7E600280
"  v_mov_b32     %[o_regs_9], 0                                   \n" // 000000000C28: 7E620280
"  v_mov_b32     %[o_regs_10], 0                                   \n" // 000000000C2C: 7E640280
"  v_mov_b32     %[o_regs_11], 0                                   \n" // 000000000C30: 7E660280
"  v_mov_b32     %[o_regs_12], 0                                   \n" // 000000000C34: 7E680280
"  v_mov_b32     %[o_regs_13], 0                                   \n" // 000000000C38: 7E6A0280
"  v_mov_b32     %[o_regs_14], 0                                   \n" // 000000000C3C: 7E6C0280
"  v_mov_b32     %[o_regs_15], 0                                   \n" // 000000000C40: 7E6E0280
"  v_mov_b32     %[o_regs_16], 0                                   \n" // 000000000C44: 7E700280
"  v_mov_b32     %[o_regs_17], 0                                   \n" // 000000000C48: 7E720280
"  v_mov_b32     %[o_regs_18], 0                                   \n" // 000000000C4C: 7E740280
"  v_mov_b32     %[o_regs_19], 0                                   \n" // 000000000C50: 7E760280
"  v_mov_b32     %[o_regs_20], 0                                   \n" // 000000000C54: 7E780280
"  v_mov_b32     %[o_regs_21], 0                                   \n" // 000000000C58: 7E7A0280
"  v_mov_b32     %[o_regs_22], 0                                   \n" // 000000000C5C: 7E7C0280
"  v_mov_b32     %[o_regs_23], 0                                   \n" // 000000000C60: 7E7E0280
"  v_mov_b32     %[o_regs_24], 0                                   \n" // 000000000C64: 7E800280
"  v_mov_b32     %[o_regs_25], 0                                   \n" // 000000000C68: 7E820280
"  v_mov_b32     %[o_regs_26], 0                                   \n" // 000000000C6C: 7E840280
"  v_mov_b32     %[o_regs_27], 0                                   \n" // 000000000C70: 7E860280
"  v_mov_b32     %[o_regs_28], 0                                   \n" // 000000000C74: 7E880280
"  v_mov_b32     %[o_regs_29], 0                                   \n" // 000000000C78: 7E8A0280
"  v_mov_b32     %[o_regs_30], 0                                   \n" // 000000000C7C: 7E8C0280
"  v_mov_b32     %[o_regs_31], 0                                   \n" // 000000000C80: 7E8E0280
"  v_mov_b32     %[o_regs_32], 0                                   \n" // 000000000C84: 7E900280
"  v_mov_b32     %[o_regs_33], 0                                   \n" // 000000000C88: 7E920280
"  v_mov_b32     %[o_regs_34], 0                                   \n" // 000000000C8C: 7E940280
"  v_mov_b32     %[o_regs_35], 0                                   \n" // 000000000C90: 7E960280
"  v_mov_b32     %[o_regs_36], 0                                   \n" // 000000000C94: 7E980280
"  v_mov_b32     %[o_regs_37], 0                                   \n" // 000000000C98: 7E9A0280
"  v_mov_b32     %[o_regs_38], 0                                   \n" // 000000000C9C: 7E9C0280
"  v_mov_b32     %[o_regs_39], 0                                   \n" // 000000000CA0: 7E9E0280
"  v_mov_b32     %[o_regs_40], 0                                   \n" // 000000000CA4: 7EA00280
"  v_mov_b32     %[o_regs_41], 0                                   \n" // 000000000CA8: 7EA20280
"  v_mov_b32     %[o_regs_42], 0                                   \n" // 000000000CAC: 7EA40280
"  v_mov_b32     %[o_regs_43], 0                                   \n" // 000000000CB0: 7EA60280
"  v_mov_b32     %[o_regs_44], 0                                   \n" // 000000000CB4: 7EA80280
"  v_mov_b32     %[o_regs_45], 0                                   \n" // 000000000CB8: 7EAA0280
"  v_mov_b32     %[o_regs_46], 0                                   \n" // 000000000CBC: 7EAC0280
"  v_mov_b32     %[o_regs_47], 0                                   \n" // 000000000CC0: 7EAE0280
"  v_mov_b32     %[o_regs_48], 0                                   \n" // 000000000CC4: 7EB00280
"  v_mov_b32     %[o_regs_49], 0                                   \n" // 000000000CC8: 7EB20280
"  v_mov_b32     %[o_regs_50], 0                                   \n" // 000000000CCC: 7EB40280
"  v_mov_b32     %[o_regs_51], 0                                   \n" // 000000000CD0: 7EB60280
"  v_mov_b32     %[o_regs_52], 0                                   \n" // 000000000CD4: 7EB80280
"  v_mov_b32     %[o_regs_53], 0                                   \n" // 000000000CD8: 7EBA0280
"  v_mov_b32     %[o_regs_54], 0                                   \n" // 000000000CDC: 7EBC0280
"  v_mov_b32     %[o_regs_55], 0                                   \n" // 000000000CE0: 7EBE0280
"  v_mov_b32     %[o_regs_56], 0                                   \n" // 000000000CE4: 7EC00280
"  v_mov_b32     %[o_regs_57], 0                                   \n" // 000000000CE8: 7EC20280
"  v_mov_b32     %[o_regs_58], 0                                   \n" // 000000000CEC: 7EC40280
"  v_mov_b32     %[o_regs_59], 0                                   \n" // 000000000CF0: 7EC60280
"  v_mov_b32     %[o_regs_60], 0                                  \n" // 000000000CF4: 7EC80280
"  v_mov_b32     %[o_regs_61], 0                                  \n" // 000000000CF8: 7ECA0280
"  v_mov_b32     %[o_regs_62], 0                                  \n" // 000000000CFC: 7ECC0280
"  v_mov_b32     %[o_regs_63], 0                                  \n" // 000000000D00: 7ECE0280
"  v_mov_b32     %[o_regs_64], 0                                  \n" // 000000000D04: 7ED00280
"  v_mov_b32     %[o_regs_65], 0                                  \n" // 000000000D08: 7ED20280
"  v_mov_b32     %[o_regs_66], 0                                  \n" // 000000000D0C: 7ED40280
"  v_mov_b32     %[o_regs_67], 0                                  \n" // 000000000D10: 7ED60280
"  v_mov_b32     %[o_regs_68], 0                                  \n" // 000000000D14: 7ED80280
"  v_mov_b32     %[o_regs_69], 0                                  \n" // 000000000D18: 7EDA0280
"  v_mov_b32     %[o_regs_70], 0                                  \n" // 000000000D1C: 7EDC0280
"  v_mov_b32     %[o_regs_71], 0                                  \n" // 000000000D20: 7EDE0280
"  v_mov_b32     %[o_regs_72], 0                                  \n" // 000000000D24: 7EE00280
"  v_mov_b32     %[o_regs_73], 0                                  \n" // 000000000D28: 7EE20280
"  v_mov_b32     %[o_regs_74], 0                                  \n" // 000000000D2C: 7EE40280
"  v_mov_b32     %[o_regs_75], 0                                  \n" // 000000000D30: 7EE60280
"  v_mov_b32     %[o_regs_76], 0                                  \n" // 000000000D34: 7EE80280
"  v_mov_b32     %[o_regs_77], 0                                  \n" // 000000000D38: 7EEA0280
"  v_mov_b32     %[o_regs_78], 0                                  \n" // 000000000D3C: 7EEC0280
"  v_mov_b32     %[o_regs_79], 0                                  \n" // 000000000D40: 7EEE0280
"  v_mov_b32     %[o_regs_80], 0                                  \n" // 000000000D44: 7EF00280
"  v_mov_b32     %[o_regs_81], 0                                  \n" // 000000000D48: 7EF20280
"  v_mov_b32     %[o_regs_82], 0                                  \n" // 000000000D4C: 7EF40280
"  v_mov_b32     %[o_regs_83], 0                                  \n" // 000000000D50: 7EF60280
"  v_mov_b32     %[o_regs_84], 0                                  \n" // 000000000D54: 7EF80280
"  v_mov_b32     %[o_regs_85], 0                                  \n" // 000000000D58: 7EFA0280
"  v_mov_b32     %[o_regs_86], 0                                  \n" // 000000000D5C: 7EFC0280
"  v_mov_b32     %[o_regs_87], 0                                  \n" // 000000000D60: 7EFE0280
"  v_mov_b32     %[o_regs_88], 0                                  \n" // 000000000D64: 7F000280
"  v_mov_b32     %[o_regs_89], 0                                  \n" // 000000000D68: 7F020280
"  v_mov_b32     %[o_regs_90], 0                                  \n" // 000000000D6C: 7F040280
"  v_mov_b32     %[o_regs_91], 0                                  \n" // 000000000D70: 7F060280
"  v_mov_b32     %[o_regs_92], 0                                  \n" // 000000000D74: 7F080280
"  v_mov_b32     %[o_regs_93], 0                                  \n" // 000000000D78: 7F0A0280
"  v_mov_b32     %[o_regs_94], 0                                  \n" // 000000000D7C: 7F0C0280
"  v_mov_b32     %[o_regs_95], 0                                  \n" // 000000000D80: 7F0E0280
"  v_mov_b32     %[o_regs_96], 0                                  \n" // 000000000D84: 7F100280
"  v_mov_b32     %[o_regs_97], 0                                  \n" // 000000000D88: 7F120280
"  v_mov_b32     %[o_regs_98], 0                                  \n" // 000000000D8C: 7F140280
"  v_mov_b32     %[o_regs_99], 0                                  \n" // 000000000D90: 7F160280
"  v_mov_b32     %[o_regs_100], 0                                  \n" // 000000000D94: 7F180280
"  v_mov_b32     %[o_regs_101], 0                                  \n" // 000000000D98: 7F1A0280
"  v_mov_b32     %[o_regs_102], 0                                  \n" // 000000000D9C: 7F1C0280
"  v_mov_b32     %[o_regs_103], 0                                  \n" // 000000000DA0: 7F1E0280
"  v_mov_b32     %[o_regs_104], 0                                  \n" // 000000000DA4: 7F200280
"  v_mov_b32     %[o_regs_105], 0                                  \n" // 000000000DA8: 7F220280
"  v_mov_b32     %[o_regs_106], 0                                  \n" // 000000000DAC: 7F240280
"  v_mov_b32     %[o_regs_107], 0                                  \n" // 000000000DB0: 7F260280
"  v_mov_b32     %[o_regs_108], 0                                  \n" // 000000000DB4: 7F280280
"  v_mov_b32     %[o_regs_109], 0                                  \n" // 000000000DB8: 7F2A0280
"  v_mov_b32     %[o_regs_110], 0                                  \n" // 000000000DBC: 7F2C0280
"  v_mov_b32     %[o_regs_111], 0                                  \n" // 000000000DC0: 7F2E0280
"  v_mov_b32     %[o_regs_112], 0                                  \n" // 000000000DC4: 7F300280
"  v_mov_b32     %[o_regs_113], 0                                  \n" // 000000000DC8: 7F320280
"  v_mov_b32     %[o_regs_114], 0                                  \n" // 000000000DCC: 7F340280
"  v_mov_b32     %[o_regs_115], 0                                  \n" // 000000000DD0: 7F360280
"  v_mov_b32     %[o_regs_116], 0                                  \n" // 000000000DD4: 7F380280
"  v_mov_b32     %[o_regs_117], 0                                  \n" // 000000000DD8: 7F3A0280
"  v_mov_b32     %[o_regs_118], 0                                  \n" // 000000000DDC: 7F3C0280
"  v_mov_b32     %[o_regs_119], 0                                  \n" // 000000000DE0: 7F3E0280
"  v_mov_b32     %[o_regs_120], 0                                  \n" // 000000000DE4: 7F400280
"  v_mov_b32     %[o_regs_121], 0                                  \n" // 000000000DE8: 7F420280
"  v_mov_b32     %[o_regs_122], 0                                  \n" // 000000000DEC: 7F440280
"  v_mov_b32     %[o_regs_123], 0                                  \n" // 000000000DF0: 7F460280
"  v_mov_b32     %[o_regs_124], 0                                  \n" // 000000000DF4: 7F480280
"  v_mov_b32     %[o_regs_125], 0                                  \n" // 000000000DF8: 7F4A0280
"  v_mov_b32     %[o_regs_126], 0                                  \n" // 000000000DFC: 7F4C0280
"  v_mov_b32     %[o_regs_127], 0                                  \n" // 000000000E00: 7F4E0280
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000000E04: 20280084
"  v_mul_i32_i24  v21, 4, v20                             \n" // 000000000E08: 0C2A2884
"  v_and_b32     v20, 15, %[threadIdxx]                             \n" // 000000000E0C: 2628008F
"  v_and_b32     v22, 3, v20                              \n" // 000000000E10: 262C2883
"  v_mul_i32_i24  v22, 0x00000488, v22                    \n" // 000000000E14: 0C2C2CFF 00000488
"  v_add_u32     v4, v22, v21                             \n" // 000000000E1C: 68082B16
"  v_lshrrev_b32  v20, 2, v20                             \n" // 000000000E20: 20282882
"  v_and_b32     v21, 1, v20                              \n" // 000000000E24: 262A2881
"  v_mul_i32_i24  v21, 32, v21                            \n" // 000000000E28: 0C2A2AA0
"  v_add_u32     v4, v4, v21                              \n" // 000000000E2C: 68082B04
"  v_and_b32     v21, 2, v20                              \n" // 000000000E30: 262A2882
"  v_mul_i32_i24  v21, 0x00000120, v21                    \n" // 000000000E34: 0C2A2AFF 00000120
"  v_add_u32     v4, v4, v21                              \n" // 000000000E3C: 68082B04
"  v_lshlrev_b32  v4, 2, v4                               \n" // 000000000E40: 24080882
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000000E44: 20280084
"  v_and_b32     v21, 1, v20                              \n" // 000000000E48: 262A2881
"  v_mul_i32_i24  v5, 32, v21                             \n" // 000000000E4C: 0C0A2AA0
"  v_and_b32     v21, 2, v20                              \n" // 000000000E50: 262A2882
"  v_mul_i32_i24  v21, 0x00000120, v21                    \n" // 000000000E54: 0C2A2AFF 00000120
"  v_add_u32     v5, v5, v21                              \n" // 000000000E5C: 680A2B05
"  v_and_b32     v20, 15, %[threadIdxx]                             \n" // 000000000E60: 2628008F
"  v_mul_i32_i24  v21, 2, v20                             \n" // 000000000E64: 0C2A2882
"  v_add_u32     v5, v5, v21                              \n" // 000000000E68: 680A2B05
"  s_mul_i32     s56, 64, %[wave_id]                              \n" // 000000000E6C: 923807C0
"  v_add_u32     v5, v5, s56                              \n" // 000000000E70: D1340005 00007105
"  v_lshlrev_b32  v5, 2, v5                               \n" // 000000000E78: 240A0A82
"  v_lshlrev_b32  v6, 2, %[threadIdxx]                              \n" // 000000000E7C: 240C0082
"  s_mul_i32     s56, 0x00000200, %[wave_id]                      \n" // 000000000E80: 923807FF 00000200
"  v_add_u32     v6, v6, s56                              \n" // 000000000E88: D1340006 00007106
"  v_lshlrev_b32  v6, 2, v6                               \n" // 000000000E90: 240C0C82
"  v_lshlrev_b32  v7, 4, %[threadIdxx]                              \n" // 000000000E94: 240E0084
"  v_mul_u32_u24  v18, v11, v9 row_newbcast:0             \n" // 000000000E98: 102412FA FF01500B
"  v_mul_u32_u24  v19, v11, v9 row_newbcast:8             \n" // 000000000EA0: 102612FA FF01580B
"  v_add_u32     v18, v18, v1                             \n" // 000000000EA8: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 000000000EAC: 68260313
"  v_mul_u32_u24  v182, v11, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v11, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s36                                  \n" // 000000000EB0: BEFC0024
"  buffer_load_dword  v11, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 000000000EB4: E0501000 80060B08
"  v_add_u32     v8, s73, v8                              \n" // 000000000EBC: 68101049
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000000EC0: E0511000 80050012
"  s_add_u32     m0, 0, s37                               \n" // 000000000EC8: 807C2580
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000000ECC: E0511000 80050013
"  s_add_u32     m0, 0x00000080, s36                      \n" // 000000000ED4: 807C24FF 00000080
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000000EDC: E0511080 80050012
"  s_add_u32     m0, 0x00000080, s37                      \n" // 000000000EE4: 807C25FF 00000080
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000000EEC: E0511080 80050013
"  s_add_u32     m0, 0x00000100, s36                      \n" // 000000000EF4: 807C24FF 00000100
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000000EFC: E0511100 80050012
"  s_add_u32     m0, 0x00000100, s37                      \n" // 000000000F04: 807C25FF 00000100
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000000F0C: E0511100 80050013
"  s_add_u32     m0, 0x00000180, s36                      \n" // 000000000F14: 807C24FF 00000180
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 000000000F1C: E0511180 80050012
"  s_add_u32     m0, 0x00000180, s37                      \n" // 000000000F24: 807C25FF 00000180
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 000000000F2C: E0511180 80050013
"  s_add_u32     m0, 0x00000200, s36                      \n" // 000000000F34: 807C24FF 00000200
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000000F3C: E0511200 80050012
"  s_add_u32     m0, 0x00000200, s37                      \n" // 000000000F44: 807C25FF 00000200
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000000F4C: E0511200 80050013
"  s_add_u32     m0, 0x00000280, s36                      \n" // 000000000F54: 807C24FF 00000280
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000000F5C: E0511280 80050012
"  s_add_u32     m0, 0x00000280, s37                      \n" // 000000000F64: 807C25FF 00000280
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000000F6C: E0511280 80050013
"  s_add_u32     m0, 0x00000300, s36                      \n" // 000000000F74: 807C24FF 00000300
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000000F7C: E0511300 80050012
"  s_add_u32     m0, 0x00000300, s37                      \n" // 000000000F84: 807C25FF 00000300
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000000F8C: E0511300 80050013
"  s_add_u32     m0, 0x00000380, s36                      \n" // 000000000F94: 807C24FF 00000380
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000000F9C: E0511380 80050012
"  s_add_u32     m0, 0x00000380, s37                      \n" // 000000000FA4: 807C25FF 00000380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000000FAC: E0511380 80050013
"  s_add_u32     m0, 0x00000400, s36                      \n" // 000000000FB4: 807C24FF 00000400
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v182, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000400, s37                      \n" // 000000000FC4: 807C25FF 00000400
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v183, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000480, s36                      \n" // 000000000FD4: 807C24FF 00000480
"  s_waitcnt     vmcnt(19) & lgkmcnt(0)                   \n" // 000000000FDC: BF8C4073
"  s_barrier                                              \n" // 000000000FE0: BF8A0000
"  v_mul_u32_u24  v18, v10, v9 row_newbcast:0             \n" // 000000000FE4: 102412FA FF01500A
"  v_mul_u32_u24  v19, v10, v9 row_newbcast:8             \n" // 000000000FEC: 102612FA FF01580A
"  v_add_u32     v18, v18, v1                             \n" // 000000000FF4: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 000000000FF8: 68260313
"  v_mul_u32_u24  v182, v10, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v10, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s34                                  \n" // 000000000FFC: BEFC0022
"  ds_read_b128  acc[144:147], v4                         \n" // 000000001000: DBFE0000 90000004
"  ds_read_b128  acc[148:151], v4 offset:64               \n" // 000000001008: DBFE0040 94000004
"  ds_read_b128  acc[152:155], v4 offset:256              \n" // 000000001010: DBFE0100 98000004
"  ds_read_b128  acc[156:159], v4 offset:320              \n" // 000000001018: DBFE0140 9C000004
"  ds_read_b128  acc[160:163], v4 offset:512              \n" // 000000001020: DBFE0200 A0000004
"  ds_read_b128  acc[164:167], v4 offset:576              \n" // 000000001028: DBFE0240 A4000004
"  ds_read_b128  acc[168:171], v4 offset:768              \n" // 000000001030: DBFE0300 A8000004
"  ds_read_b128  acc[172:175], v4 offset:832              \n" // 000000001038: DBFE0340 AC000004
"  ds_read_b128  acc[176:179], v4 offset:1024             \n" // 000000001040: DBFE0400 B0000004
"  ds_read_b128  acc[180:183], v4 offset:1088             \n" // 000000001048: DBFE0440 B4000004
"  ds_read_b128  acc[184:187], v4 offset:1280             \n" // 000000001050: DBFE0500 B8000004
"  ds_read_b128  acc[188:191], v4 offset:1344             \n" // 000000001058: DBFE0540 BC000004
"  ds_read_b128  acc[192:195], v4 offset:1536             \n" // 000000001060: DBFE0600 C0000004
"  ds_read_b128  acc[196:199], v4 offset:1600             \n" // 000000001068: DBFE0640 C4000004
"  ds_read_b128  acc[200:203], v4 offset:1792             \n" // 000000001070: DBFE0700 C8000004
"  ds_read_b128  acc[204:207], v4 offset:1856             \n" // 000000001078: DBFE0740 CC000004
"  ds_read_b128  acc[208:211], v4 offset:2048             \n" // 000000001080: DBFE0800 D0000004
"  ds_read_b128  acc[212:215], v4 offset:2112             \n" // 000000001088: DBFE0840 D4000004
"  ds_read_b64   v[20:21], v5                             \n" // 000000001090: D8EC0000 14000005
"  ds_read_b64   v[22:23], v5 offset:4640                 \n" // 000000001098: D8EC1220 16000005
"  ds_read_b64   v[24:25], v5 offset:9280                 \n" // 0000000010A0: D8EC2440 18000005
"  ds_read_b64   v[26:27], v5 offset:13920                \n" // 0000000010A8: D8EC3660 1A000005
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000010B0: BF8CC07F
"  v_perm_b32    v168, v22, v20, s53                      \n" // 0000000010B4: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 0000000010BC: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 0000000010C4: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 0000000010CC: D1ED00AB 00D2311A
"  ds_write_b128  v6, v[168:171] offset:37120             \n" // 0000000010D4: D9BE9100 0000A806
"  v_perm_b32    v168, v23, v21, s53                      \n" // 0000000010DC: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 0000000010E4: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 0000000010EC: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 0000000010F4: D1ED00AB 00D2331B
"  ds_write_b128  v6, v[168:171] offset:38144             \n" // 0000000010FC: D9BE9500 0000A806
"  ds_read_b64   v[20:21], v5 offset:1024                 \n" // 000000001104: D8EC0400 14000005
"  ds_read_b64   v[22:23], v5 offset:5664                 \n" // 00000000110C: D8EC1620 16000005
"  ds_read_b64   v[24:25], v5 offset:10304                \n" // 000000001114: D8EC2840 18000005
"  ds_read_b64   v[26:27], v5 offset:14944                \n" // 00000000111C: D8EC3A60 1A000005
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000001124: BF8CC07F
"  v_perm_b32    v168, v22, v20, s53                      \n" // 000000001128: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000001130: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 000000001138: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000001140: D1ED00AB 00D2311A
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 000000001148: D9BEB100 0000A806
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000001150: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000001158: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000001160: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000001168: D1ED00AB 00D2331B
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 000000001170: D9BEB500 0000A806
"  s_nop         0x0000                                   \n" // 000000001178: BF800000
"  s_cmp_lt_u32  s71, 1                                   \n" // 00000000117C: BF0A8147
"  s_cbranch_scc1  label_0DFA_%=                             \n" // 000000001180: BF850999
"  s_cmp_lt_i32  %[wave_id], 2                                    \n" // 000000001184: BF048207
"  s_cbranch_scc0  label_092F_%=                             \n" // 000000001188: BF8404CC
"label_0463_%=: \n"
"  s_waitcnt     lgkmcnt(4)                               \n" // 00000000118C: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[144:145], acc[0:1], 0  \n" // 000000001190: D3E10020 1A020190
"  ds_read_b128  acc[176:179], v4 offset:1024             \n" // 000000001198: DBFE0400 B0000004
"  ds_read_b128  acc[180:183], v4 offset:1088             \n" // 0000000011A0: DBFE0440 B4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[146:147], acc[2:3], v[32:35]  \n" // 0000000011A8: D3E10020 1C820592
"  buffer_load_dword  v10, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 0000000011B0: E0501000 80060A08
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[148:149], acc[4:5], v[32:35]  \n" // 0000000011B8: D3E10020 1C820994
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[150:151], acc[6:7], v[32:35]  \n" // 0000000011C0: D3E10020 1C820D96
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[152:153], acc[8:9], v[32:35]  \n" // 0000000011C8: D3E10020 1C821198
"  ds_read_b128  acc[184:187], v4 offset:1280             \n" // 0000000011D0: DBFE0500 B8000004
"  ds_read_b128  acc[188:191], v4 offset:1344             \n" // 0000000011D8: DBFE0540 BC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[154:155], acc[10:11], v[32:35]  \n" // 0000000011E0: D3E10020 1C82159A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[156:157], acc[12:13], v[32:35]  \n" // 0000000011E8: D3E10020 1C82199C
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[158:159], acc[14:15], v[32:35]  \n" // 0000000011F0: D3E10020 1C821D9E
"  s_waitcnt     lgkmcnt(4)                               \n" // 0000000011F8: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[160:161], acc[16:17], v[32:35]  \n" // 0000000011FC: D3E10020 1C8221A0
"  ds_read_b128  acc[192:195], v4 offset:1536             \n" // 000000001204: DBFE0600 C0000004
"  ds_read_b128  acc[196:199], v4 offset:1600             \n" // 00000000120C: DBFE0640 C4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[162:163], acc[18:19], v[32:35]  \n" // 000000001214: D3E10020 1C8225A2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[164:165], acc[20:21], v[32:35]  \n" // 00000000121C: D3E10020 1C8229A4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[166:167], acc[22:23], v[32:35]  \n" // 000000001224: D3E10020 1C822DA6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[168:169], acc[24:25], v[32:35]  \n" // 00000000122C: D3E10020 1C8231A8
"  ds_read_b128  acc[200:203], v4 offset:1792             \n" // 000000001234: DBFE0700 C8000004
"  ds_read_b128  acc[204:207], v4 offset:1856             \n" // 00000000123C: DBFE0740 CC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[170:171], acc[26:27], v[32:35]  \n" // 000000001244: D3E10020 1C8235AA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[172:173], acc[28:29], v[32:35]  \n" // 00000000124C: D3E10020 1C8239AC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[174:175], acc[30:31], v[32:35]  \n" // 000000001254: D3E10020 1C823DAE
"  s_waitcnt     lgkmcnt(4)                               \n" // 00000000125C: BF8CC47F
"  s_barrier                                              \n" // 000000001260: BF8A0000
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[176:177], acc[32:33], v[32:35]  \n" // 000000001264: D3E10020 1C8241B0
"  ds_read_b128  acc[208:211], v4 offset:2048             \n" // 00000000126C: DBFE0800 D0000004
"  ds_read_b128  acc[212:215], v4 offset:2112             \n" // 000000001274: DBFE0840 D4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[178:179], acc[34:35], v[32:35]  \n" // 00000000127C: D3E10020 1C8245B2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[180:181], acc[36:37], v[32:35]  \n" // 000000001284: D3E10020 1C8249B4
"  v_perm_b32    v168, v22, v20, s53                      \n" // 00000000128C: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000001294: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 00000000129C: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 0000000012A4: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[182:183], acc[38:39], v[32:35]  \n" // 0000000012AC: D3E10020 1C824DB6
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 0000000012B4: E0511000 80050012
"  s_add_u32     m0, 0, s35                               \n" // 0000000012BC: 807C2380
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[184:185], acc[40:41], v[32:35]  \n" // 0000000012C0: D3E10020 1C8251B8
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 0000000012C8: D9BEB100 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[186:187], acc[42:43], v[32:35]  \n" // 0000000012D0: D3E10020 1C8255BA
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 0000000012D8: E0511000 80050013
"  s_add_u32     m0, 0x00000080, s34                      \n" // 0000000012E0: 807C22FF 00000080
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[188:189], acc[44:45], v[32:35]  \n" // 0000000012E8: D3E10020 1C8259BC
"  v_perm_b32    v168, v23, v21, s53                      \n" // 0000000012F0: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 0000000012F8: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000001300: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000001308: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[190:191], acc[46:47], v[32:35]  \n" // 000000001310: D3E10020 1C825DBE
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000001318: E0511080 80050012
"  s_add_u32     m0, 0x00000080, s35                      \n" // 000000001320: 807C23FF 00000080
"  s_waitcnt     lgkmcnt(1)                               \n" // 000000001328: BF8CC17F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[192:193], acc[48:49], v[32:35]  \n" // 00000000132C: D3E10020 1C8261C0
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 000000001334: D9BEB500 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[194:195], acc[50:51], v[32:35]  \n" // 00000000133C: D3E10020 1C8265C2
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000001344: E0511080 80050013
"  s_add_u32     m0, 0x00000100, s34                      \n" // 00000000134C: 807C22FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[196:197], acc[52:53], v[32:35]  \n" // 000000001354: D3E10020 1C8269C4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[198:199], acc[54:55], v[32:35]  \n" // 00000000135C: D3E10020 1C826DC6
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000001364: E0511100 80050012
"  s_add_u32     m0, 0x00000100, s35                      \n" // 00000000136C: 807C23FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[200:201], acc[56:57], v[32:35]  \n" // 000000001374: D3E10020 1C8271C8
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[202:203], acc[58:59], v[32:35]  \n" // 00000000137C: D3E10020 1C8275CA
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000001384: E0511100 80050013
"  s_add_u32     m0, 0x00000180, s34                      \n" // 00000000138C: 807C22FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[204:205], acc[60:61], v[32:35]  \n" // 000000001394: D3E10020 1C8279CC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[206:207], acc[62:63], v[32:35]  \n" // 00000000139C: D3E10020 1C827DCE
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 0000000013A4: E0511180 80050012
"  s_add_u32     m0, 0x00000180, s35                      \n" // 0000000013AC: 807C23FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[208:209], acc[64:65], v[32:35]  \n" // 0000000013B4: D3E10020 1C8281D0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[210:211], acc[66:67], v[32:35]  \n" // 0000000013BC: D3E10020 1C8285D2
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 0000000013C4: E0511180 80050013
"  s_add_u32     m0, 0x00000200, s34                      \n" // 0000000013CC: 807C22FF 00000200
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[212:213], acc[68:69], v[32:35]  \n" // 0000000013D4: D3E10020 1C8289D4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[214:215], acc[70:71], v[32:35]  \n" // 0000000013DC: D3E10020 1C828DD6
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 0000000013E4: E0511200 80050012
"  s_add_u32     m0, 0x00000200, s35                      \n" // 0000000013EC: 807C23FF 00000200
"  v_add_u32     v8, s73, v8                              \n" // 0000000013F4: 68101049
"  s_cmp_le_i32  s83, s82                                 \n" // 0000000013F8: BF055253
"  s_cbranch_scc1  label_0524_%=                             \n" // 0000000013FC: BF850024
"  v_mov_b32     v25, 0xff800000                          \n" // 000000001400: 7E3202FF FF800000
"  s_add_u32     s57, s82, 0                              \n" // 000000001408: 80398052
"  v_mov_b32     v24, s57                                 \n" // 00000000140C: 7E300239
"  v_add_u32     v24, %[wave_id], v24                             \n" // 000000001410: 68303007
"  s_sub_u32     s56, s83, 15                             \n" // 000000001414: 80B88F53
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000001418: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 00000000141C: 0C282884
"  v_add_u32     v20, s56, v20                            \n" // 000000001420: 68282838
"  v_add_u32     v21, 1, v20                              \n" // 000000001424: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 000000001428: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 00000000142C: 682E2883
"  v_cmp_le_u32  s[38:39], v20, v24                       \n" // 000000001430: D0CB0026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000001438: 682828C0
"  s_nop         0x0000                                   \n" // 00000000143C: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000001440: D1000020 009A4119
"  v_cmp_le_u32  s[38:39], v21, v24                       \n" // 000000001448: D0CB0026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000001450: 682A2AC0
"  s_nop         0x0000                                   \n" // 000000001454: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000001458: D1000021 009A4319
"  v_cmp_le_u32  s[38:39], v22, v24                       \n" // 000000001460: D0CB0026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000001468: 682C2CC0
"  s_nop         0x0000                                   \n" // 00000000146C: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 000000001470: D1000022 009A4519
"  v_cmp_le_u32  s[38:39], v23, v24                       \n" // 000000001478: D0CB0026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 000000001480: 682E2EC0
"  s_nop         0x0000                                   \n" // 000000001484: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 000000001488: D1000023 009A4719
"label_0524_%=: \n"
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000001490: BF8CC07F
"  s_barrier                                              \n" // 000000001494: BF8A0000
"  v_max3_f32    v24, v32, v33, v32                       \n" // 000000001498: D1D30018 04824320
"  v_max3_f32    v24, v34, v35, v24                       \n" // 0000000014A0: D1D30018 04624722
"  ds_write_b32  v3, v24 offset:53504                     \n" // 0000000014A8: D81AD100 00001803
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 0000000014B0: E0511200 80050013
"  s_add_u32     m0, 0x00000280, s34                      \n" // 0000000014B8: 807C22FF 00000280
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 0000000014C0: E0511280 80050012
"  s_add_u32     m0, 0x00000280, s35                      \n" // 0000000014C8: 807C23FF 00000280
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000014D0: BF8CC07F
"  ds_read_b32   v20, v2 offset:53504                     \n" // 0000000014D4: D86CD100 14000002
"  ds_read_b32   v21, v2 offset:53568                     \n" // 0000000014DC: D86CD140 15000002
"  ds_read_b32   v22, v2 offset:53632                     \n" // 0000000014E4: D86CD180 16000002
"  ds_read_b32   v23, v2 offset:53696                     \n" // 0000000014EC: D86CD1C0 17000002
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 0000000014F4: E0511280 80050013
"  s_add_u32     m0, 0x00000300, s34                      \n" // 0000000014FC: 807C22FF 00000300
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000001504: E0511300 80050012
"  s_add_u32     m0, 0x00000300, s35                      \n" // 00000000150C: 807C23FF 00000300
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000001514: E0511300 80050013
"  s_add_u32     m0, 0x00000380, s34                      \n" // 00000000151C: 807C22FF 00000380
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000001524: BF8CC07F
"  v_max3_f32    v24, v20, v21, v24                       \n" // 000000001528: D1D30018 04622B14
"  v_max3_f32    v24, v22, v23, v24                       \n" // 000000001530: D1D30018 04622F16
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000001538: E0511380 80050012
"  s_add_u32     m0, 0x00000380, s35                      \n" // 000000001540: 807C23FF 00000380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000001548: E0511380 80050013
"  s_add_u32     m0, 0x00000400, s34                      \n" // 000000001550: 807C22FF 00000400
"  ds_read_b128  acc[144:147], v7 offset:37120            \n" // 000000001558: DBFE9100 90000007
"  ds_read_b128  acc[148:151], v7 offset:38144            \n" // 000000001560: DBFE9500 94000007
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v182, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000400, s35                      \n" // 000000001570: 807C23FF 00000400
"  ds_read_b128  acc[152:155], v7 offset:39168            \n" // 000000001578: DBFE9900 98000007
"  ds_read_b128  acc[156:159], v7 offset:40192            \n" // 000000001580: DBFE9D00 9C000007
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v183, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  ;s_add_u32     m0, 0x00000480, s34                      \n" // 000000001590: 807C22FF 00000480
"  ds_read_b128  acc[160:163], v7 offset:41216            \n" // 000000001598: DBFEA100 A0000007
"  ds_read_b128  acc[164:167], v7 offset:42240            \n" // 0000000015A0: DBFEA500 A4000007
"  ds_read_b128  acc[168:171], v7 offset:43264            \n" // 0000000015A8: DBFEA900 A8000007
"  ds_read_b128  acc[172:175], v7 offset:44288            \n" // 0000000015B0: DBFEAD00 AC000007
"  v_mov_b32     v25, 0xff7fffff                          \n" // 0000000015B8: 7E3202FF FF7FFFFF
"  v_cmp_eq_u32  s[38:39], v25, v12                       \n" // 0000000015C0: D0CA0026 00021919
"  v_max_f32     v20, v24, v12                            \n" // 0000000015C8: 16281918
"  v_sub_f32     v16, v12, v20                            \n" // 0000000015CC: 0420290C
"  v_cndmask_b32  v16, v16, 0, s[38:39]                   \n" // 0000000015D0: D1000010 00990110
"  v_mov_b32     v12, v20                                 \n" // 0000000015D8: 7E180314
"  v_mul_f32     v21, s5, v20                             \n" // 0000000015DC: 0A2A2805
"  v_mul_f32     v16, s5, v16                             \n" // 0000000015E0: 0A202005
"  v_exp_f32     v16, v16                                 \n" // 0000000015E4: 7E204110
"  v_fma_f32     v32, v32, s5, -v21                       \n" // 0000000015E8: D1CB0020 84540B20
"  v_fma_f32     v33, v33, s5, -v21                       \n" // 0000000015F0: D1CB0021 84540B21
"  v_fma_f32     v34, v34, s5, -v21                       \n" // 0000000015F8: D1CB0022 84540B22
"  v_fma_f32     v35, v35, s5, -v21                       \n" // 000000001600: D1CB0023 84540B23
"  v_exp_f32     v32, v32                                 \n" // 000000001608: 7E404120
"  v_exp_f32     v33, v33                                 \n" // 00000000160C: 7E424121
"  v_exp_f32     v34, v34                                 \n" // 000000001610: 7E444122
"  v_exp_f32     v35, v35                                 \n" // 000000001614: 7E464123
"  v_mul_f32     v14, v16, v14                            \n" // 000000001618: 0A1C1D10
"  v_mov_b32     v22, v32                                 \n" // 00000000161C: 7E2C0320
"  v_add_f32     v22, v33, v22                            \n" // 000000001620: 022C2D21
"  v_add_f32     v22, v34, v22                            \n" // 000000001624: 022C2D22
"  v_add_f32     v22, v35, v22                            \n" // 000000001628: 022C2D23
"  v_add_f32     v14, v22, v14                            \n" // 00000000162C: 021C1D16
"  v_mov_b32     v29, 0xffff0000                          \n" // 000000001630: 7E3A02FF FFFF0000
"  v_mov_b32     v30, 0x7fff0000                          \n" // 000000001638: 7E3C02FF 7FFF0000
"  v_mov_b32     v31, 0x00007fff                          \n" // 000000001640: 7E3E02FF 00007FFF
"  v_cmp_u_f32   s[38:39], v32, v32                       \n" // 000000001648: D0480026 00024120
"  v_add3_u32    v28, v32, v31, 1                         \n" // 000000001650: D1FF001C 02063F20
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000001658: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v33, v33                       \n" // 000000001660: D0480026 00024321
"  v_add3_u32    v28, v33, v31, 1                         \n" // 000000001668: D1FF001C 02063F21
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000001670: D1000015 009A3D1C
"  v_perm_b32    v32, v21, v20, s52                       \n" // 000000001678: D1ED0020 00D22915
"  v_cmp_u_f32   s[38:39], v34, v34                       \n" // 000000001680: D0480026 00024522
"  v_add3_u32    v28, v34, v31, 1                         \n" // 000000001688: D1FF001C 02063F22
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000001690: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v35, v35                       \n" // 000000001698: D0480026 00024723
"  v_add3_u32    v28, v35, v31, 1                         \n" // 0000000016A0: D1FF001C 02063F23
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 0000000016A8: D1000015 009A3D1C
"  v_perm_b32    v33, v21, v20, s52                       \n" // 0000000016B0: D1ED0021 00D22915
"  s_nop         0x0002                                   \n" // 0000000016B8: BF800002
"  s_add_u32     s83, s84, s83                            \n" // 0000000016BC: 80535354
"  s_nop         0x0000                                   \n" // 0000000016C0: BF800000
"  v_mul_u32_u24  v18, v11, v9 row_newbcast:0             \n" // 0000000016C4: 102412FA FF01500B
"  v_mul_u32_u24  v19, v11, v9 row_newbcast:8             \n" // 0000000016CC: 102612FA FF01580B
"  v_add_u32     v18, v18, v1                             \n" // 0000000016D4: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 0000000016D8: 68260313
"  v_mul_u32_u24  v182, v11, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v11, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s36                                  \n" // 0000000016DC: BEFC0024
"  v_mov_b32     v22, v16                                 \n" // 0000000016E0: 7E2C0310
"  v_mov_b32     v23, v16                                 \n" // 0000000016E4: 7E2E0310
"  v_pk_mul_f32  [%[o_regs_0], %[o_regs_1]], v[22:23], [%[o_regs_0], %[o_regs_1]]             \n" // 0000000016E8: D3B14028 18025116
"  v_pk_mul_f32  [%[o_regs_2], %[o_regs_3]], v[22:23], [%[o_regs_2], %[o_regs_3]]             \n" // 0000000016F0: D3B1402A 18025516
"  v_pk_mul_f32  [%[o_regs_4], %[o_regs_5]], v[22:23], [%[o_regs_4], %[o_regs_5]]             \n" // 0000000016F8: D3B1402C 18025916
"  v_pk_mul_f32  [%[o_regs_6], %[o_regs_7]], v[22:23], [%[o_regs_6], %[o_regs_7]]             \n" // 000000001700: D3B1402E 18025D16
"  v_pk_mul_f32  [%[o_regs_8], %[o_regs_9]], v[22:23], [%[o_regs_8], %[o_regs_9]]             \n" // 000000001708: D3B14030 18026116
"  v_pk_mul_f32  [%[o_regs_10], %[o_regs_11]], v[22:23], [%[o_regs_10], %[o_regs_11]]             \n" // 000000001710: D3B14032 18026516
"  v_pk_mul_f32  [%[o_regs_12], %[o_regs_13]], v[22:23], [%[o_regs_12], %[o_regs_13]]             \n" // 000000001718: D3B14034 18026916
"  v_pk_mul_f32  [%[o_regs_14], %[o_regs_15]], v[22:23], [%[o_regs_14], %[o_regs_15]]             \n" // 000000001720: D3B14036 18026D16
"  v_pk_mul_f32  [%[o_regs_16], %[o_regs_17]], v[22:23], [%[o_regs_16], %[o_regs_17]]             \n" // 000000001728: D3B14038 18027116
"  v_pk_mul_f32  [%[o_regs_18], %[o_regs_19]], v[22:23], [%[o_regs_18], %[o_regs_19]]             \n" // 000000001730: D3B1403A 18027516
"  v_pk_mul_f32  [%[o_regs_20], %[o_regs_21]], v[22:23], [%[o_regs_20], %[o_regs_21]]             \n" // 000000001738: D3B1403C 18027916
"  v_pk_mul_f32  [%[o_regs_22], %[o_regs_23]], v[22:23], [%[o_regs_22], %[o_regs_23]]             \n" // 000000001740: D3B1403E 18027D16
"  v_pk_mul_f32  [%[o_regs_24], %[o_regs_25]], v[22:23], [%[o_regs_24], %[o_regs_25]]             \n" // 000000001748: D3B14040 18028116
"  v_pk_mul_f32  [%[o_regs_26], %[o_regs_27]], v[22:23], [%[o_regs_26], %[o_regs_27]]             \n" // 000000001750: D3B14042 18028516
"  v_pk_mul_f32  [%[o_regs_28], %[o_regs_29]], v[22:23], [%[o_regs_28], %[o_regs_29]]             \n" // 000000001758: D3B14044 18028916
"  v_pk_mul_f32  [%[o_regs_30], %[o_regs_31]], v[22:23], [%[o_regs_30], %[o_regs_31]]             \n" // 000000001760: D3B14046 18028D16
"  v_pk_mul_f32  [%[o_regs_32], %[o_regs_33]], v[22:23], [%[o_regs_32], %[o_regs_33]]             \n" // 000000001768: D3B14048 18029116
"  v_pk_mul_f32  [%[o_regs_34], %[o_regs_35]], v[22:23], [%[o_regs_34], %[o_regs_35]]             \n" // 000000001770: D3B1404A 18029516
"  v_pk_mul_f32  [%[o_regs_36], %[o_regs_37]], v[22:23], [%[o_regs_36], %[o_regs_37]]             \n" // 000000001778: D3B1404C 18029916
"  v_pk_mul_f32  [%[o_regs_38], %[o_regs_39]], v[22:23], [%[o_regs_38], %[o_regs_39]]             \n" // 000000001780: D3B1404E 18029D16
"  v_pk_mul_f32  [%[o_regs_40], %[o_regs_41]], v[22:23], [%[o_regs_40], %[o_regs_41]]             \n" // 000000001788: D3B14050 1802A116
"  v_pk_mul_f32  [%[o_regs_42], %[o_regs_43]], v[22:23], [%[o_regs_42], %[o_regs_43]]             \n" // 000000001790: D3B14052 1802A516
"  v_pk_mul_f32  [%[o_regs_44], %[o_regs_45]], v[22:23], [%[o_regs_44], %[o_regs_45]]             \n" // 000000001798: D3B14054 1802A916
"  v_pk_mul_f32  [%[o_regs_46], %[o_regs_47]], v[22:23], [%[o_regs_46], %[o_regs_47]]             \n" // 0000000017A0: D3B14056 1802AD16
"  v_pk_mul_f32  [%[o_regs_48], %[o_regs_49]], v[22:23], [%[o_regs_48], %[o_regs_49]]             \n" // 0000000017A8: D3B14058 1802B116
"  v_pk_mul_f32  [%[o_regs_50], %[o_regs_51]], v[22:23], [%[o_regs_50], %[o_regs_51]]             \n" // 0000000017B0: D3B1405A 1802B516
"  v_pk_mul_f32  [%[o_regs_52], %[o_regs_53]], v[22:23], [%[o_regs_52], %[o_regs_53]]             \n" // 0000000017B8: D3B1405C 1802B916
"  v_pk_mul_f32  [%[o_regs_54], %[o_regs_55]], v[22:23], [%[o_regs_54], %[o_regs_55]]             \n" // 0000000017C0: D3B1405E 1802BD16
"  v_pk_mul_f32  [%[o_regs_56], %[o_regs_57]], v[22:23], [%[o_regs_56], %[o_regs_57]]             \n" // 0000000017C8: D3B14060 1802C116
"  v_pk_mul_f32  [%[o_regs_58], %[o_regs_59]], v[22:23], [%[o_regs_58], %[o_regs_59]]             \n" // 0000000017D0: D3B14062 1802C516
"  v_pk_mul_f32  [%[o_regs_60], %[o_regs_61]], v[22:23], [%[o_regs_60], %[o_regs_61]]         \n" // 0000000017D8: D3B14064 1802C916
"  v_pk_mul_f32  [%[o_regs_62], %[o_regs_63]], v[22:23], [%[o_regs_62], %[o_regs_63]]         \n" // 0000000017E0: D3B14066 1802CD16
"  v_pk_mul_f32  [%[o_regs_64], %[o_regs_65]], v[22:23], [%[o_regs_64], %[o_regs_65]]         \n" // 0000000017E8: D3B14068 1802D116
"  v_pk_mul_f32  [%[o_regs_66], %[o_regs_67]], v[22:23], [%[o_regs_66], %[o_regs_67]]         \n" // 0000000017F0: D3B1406A 1802D516
"  v_pk_mul_f32  [%[o_regs_68], %[o_regs_69]], v[22:23], [%[o_regs_68], %[o_regs_69]]         \n" // 0000000017F8: D3B1406C 1802D916
"  v_pk_mul_f32  [%[o_regs_70], %[o_regs_71]], v[22:23], [%[o_regs_70], %[o_regs_71]]         \n" // 000000001800: D3B1406E 1802DD16
"  v_pk_mul_f32  [%[o_regs_72], %[o_regs_73]], v[22:23], [%[o_regs_72], %[o_regs_73]]         \n" // 000000001808: D3B14070 1802E116
"  v_pk_mul_f32  [%[o_regs_74], %[o_regs_75]], v[22:23], [%[o_regs_74], %[o_regs_75]]         \n" // 000000001810: D3B14072 1802E516
"  v_pk_mul_f32  [%[o_regs_76], %[o_regs_77]], v[22:23], [%[o_regs_76], %[o_regs_77]]         \n" // 000000001818: D3B14074 1802E916
"  v_pk_mul_f32  [%[o_regs_78], %[o_regs_79]], v[22:23], [%[o_regs_78], %[o_regs_79]]         \n" // 000000001820: D3B14076 1802ED16
"  v_pk_mul_f32  [%[o_regs_80], %[o_regs_81]], v[22:23], [%[o_regs_80], %[o_regs_81]]         \n" // 000000001828: D3B14078 1802F116
"  v_pk_mul_f32  [%[o_regs_82], %[o_regs_83]], v[22:23], [%[o_regs_82], %[o_regs_83]]         \n" // 000000001830: D3B1407A 1802F516
"  v_pk_mul_f32  [%[o_regs_84], %[o_regs_85]], v[22:23], [%[o_regs_84], %[o_regs_85]]         \n" // 000000001838: D3B1407C 1802F916
"  v_pk_mul_f32  [%[o_regs_86], %[o_regs_87]], v[22:23], [%[o_regs_86], %[o_regs_87]]         \n" // 000000001840: D3B1407E 1802FD16
"  v_pk_mul_f32  [%[o_regs_88], %[o_regs_89]], v[22:23], [%[o_regs_88], %[o_regs_89]]         \n" // 000000001848: D3B14080 18030116
"  v_pk_mul_f32  [%[o_regs_90], %[o_regs_91]], v[22:23], [%[o_regs_90], %[o_regs_91]]         \n" // 000000001850: D3B14082 18030516
"  v_pk_mul_f32  [%[o_regs_92], %[o_regs_93]], v[22:23], [%[o_regs_92], %[o_regs_93]]         \n" // 000000001858: D3B14084 18030916
"  v_pk_mul_f32  [%[o_regs_94], %[o_regs_95]], v[22:23], [%[o_regs_94], %[o_regs_95]]         \n" // 000000001860: D3B14086 18030D16
"  v_pk_mul_f32  [%[o_regs_96], %[o_regs_97]], v[22:23], [%[o_regs_96], %[o_regs_97]]         \n" // 000000001868: D3B14088 18031116
"  v_pk_mul_f32  [%[o_regs_98], %[o_regs_99]], v[22:23], [%[o_regs_98], %[o_regs_99]]         \n" // 000000001870: D3B1408A 18031516
"  v_pk_mul_f32  [%[o_regs_100], %[o_regs_101]], v[22:23], [%[o_regs_100], %[o_regs_101]]         \n" // 000000001878: D3B1408C 18031916
"  v_pk_mul_f32  [%[o_regs_102], %[o_regs_103]], v[22:23], [%[o_regs_102], %[o_regs_103]]         \n" // 000000001880: D3B1408E 18031D16
"  v_pk_mul_f32  [%[o_regs_104], %[o_regs_105]], v[22:23], [%[o_regs_104], %[o_regs_105]]         \n" // 000000001888: D3B14090 18032116
"  v_pk_mul_f32  [%[o_regs_106], %[o_regs_107]], v[22:23], [%[o_regs_106], %[o_regs_107]]         \n" // 000000001890: D3B14092 18032516
"  v_pk_mul_f32  [%[o_regs_108], %[o_regs_109]], v[22:23], [%[o_regs_108], %[o_regs_109]]         \n" // 000000001898: D3B14094 18032916
"  v_pk_mul_f32  [%[o_regs_110], %[o_regs_111]], v[22:23], [%[o_regs_110], %[o_regs_111]]         \n" // 0000000018A0: D3B14096 18032D16
"  v_pk_mul_f32  [%[o_regs_112], %[o_regs_113]], v[22:23], [%[o_regs_112], %[o_regs_113]]         \n" // 0000000018A8: D3B14098 18033116
"  v_pk_mul_f32  [%[o_regs_114], %[o_regs_115]], v[22:23], [%[o_regs_114], %[o_regs_115]]         \n" // 0000000018B0: D3B1409A 18033516
"  v_pk_mul_f32  [%[o_regs_116], %[o_regs_117]], v[22:23], [%[o_regs_116], %[o_regs_117]]         \n" // 0000000018B8: D3B1409C 18033916
"  v_pk_mul_f32  [%[o_regs_118], %[o_regs_119]], v[22:23], [%[o_regs_118], %[o_regs_119]]         \n" // 0000000018C0: D3B1409E 18033D16
"  v_pk_mul_f32  [%[o_regs_120], %[o_regs_121]], v[22:23], [%[o_regs_120], %[o_regs_121]]         \n" // 0000000018C8: D3B140A0 18034116
"  v_pk_mul_f32  [%[o_regs_122], %[o_regs_123]], v[22:23], [%[o_regs_122], %[o_regs_123]]         \n" // 0000000018D0: D3B140A2 18034516
"  v_pk_mul_f32  [%[o_regs_124], %[o_regs_125]], v[22:23], [%[o_regs_124], %[o_regs_125]]         \n" // 0000000018D8: D3B140A4 18034916
"  v_pk_mul_f32  [%[o_regs_126], %[o_regs_127]], v[22:23], [%[o_regs_126], %[o_regs_127]]         \n" // 0000000018E0: D3B140A6 18034D16
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000018E8: BF8CC07F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]], acc[144:145], v[32:33], [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]]  \n" // 0000000018EC: D3E10028 0CA24190
"  ds_read_b128  acc[176:179], v7 offset:45312            \n" // 0000000018F4: DBFEB100 B0000007
"  ds_read_b128  acc[180:183], v7 offset:46336            \n" // 0000000018FC: DBFEB500 B4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]], acc[146:147], v[32:33], [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]]  \n" // 000000001904: D3E1002C 0CB24192
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]], acc[148:149], v[32:33], [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]]  \n" // 00000000190C: D3E10030 0CC24194
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]], acc[150:151], v[32:33], [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]]  \n" // 000000001914: D3E10034 0CD24196
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]], acc[152:153], v[32:33], [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]]  \n" // 00000000191C: D3E10038 0CE24198
"  ds_read_b128  acc[184:187], v7 offset:47360            \n" // 000000001924: DBFEB900 B8000007
"  ds_read_b128  acc[188:191], v7 offset:48384            \n" // 00000000192C: DBFEBD00 BC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]], acc[154:155], v[32:33], [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]]  \n" // 000000001934: D3E1003C 0CF2419A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]], acc[156:157], v[32:33], [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]]  \n" // 00000000193C: D3E10040 0D02419C
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]], acc[158:159], v[32:33], [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]]  \n" // 000000001944: D3E10044 0D12419E
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]], acc[160:161], v[32:33], [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]]  \n" // 00000000194C: D3E10048 0D2241A0
"  ds_read_b128  acc[192:195], v7 offset:49408            \n" // 000000001954: DBFEC100 C0000007
"  ds_read_b128  acc[196:199], v7 offset:50432            \n" // 00000000195C: DBFEC500 C4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]], acc[162:163], v[32:33], [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]]  \n" // 000000001964: D3E1004C 0D3241A2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]], acc[164:165], v[32:33], [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]]  \n" // 00000000196C: D3E10050 0D4241A4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]], acc[166:167], v[32:33], [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]]  \n" // 000000001974: D3E10054 0D5241A6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]], acc[168:169], v[32:33], [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]]  \n" // 00000000197C: D3E10058 0D6241A8
"  ds_read_b128  acc[200:203], v7 offset:51456            \n" // 000000001984: DBFEC900 C8000007
"  ds_read_b128  acc[204:207], v7 offset:52480            \n" // 00000000198C: DBFECD00 CC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]], acc[170:171], v[32:33], [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]]  \n" // 000000001994: D3E1005C 0D7241AA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]], acc[172:173], v[32:33], [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]]  \n" // 00000000199C: D3E10060 0D8241AC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]], acc[174:175], v[32:33], [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]]  \n" // 0000000019A4: D3E10064 0D9241AE
"  s_waitcnt     lgkmcnt(4)                               \n" // 0000000019AC: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]], acc[176:177], v[32:33], [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]]  \n" // 0000000019B0: D3E10068 0DA241B0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]], acc[178:179], v[32:33], [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]]  \n" // 0000000019B8: D3E1006C 0DB241B2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]], acc[180:181], v[32:33], [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]]  \n" // 0000000019C0: D3E10070 0DC241B4
"  s_waitcnt     vmcnt(18) & lgkmcnt(0)                   \n" // 0000000019C8: BF8C4072
"  s_barrier                                              \n" // 0000000019CC: BF8A0000
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]], acc[182:183], v[32:33], [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]]  \n" // 0000000019D0: D3E10074 0DD241B6
"  ds_read_b64   v[20:21], v5 offset:18560                \n" // 0000000019D8: D8EC4880 14000005
"  ds_read_b64   v[22:23], v5 offset:23200                \n" // 0000000019E0: D8EC5AA0 16000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]], acc[184:185], v[32:33], [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]]  \n" // 0000000019E8: D3E10078 0DE241B8
"  ds_read_b64   v[24:25], v5 offset:27840                \n" // 0000000019F0: D8EC6CC0 18000005
"  ds_read_b64   v[26:27], v5 offset:32480                \n" // 0000000019F8: D8EC7EE0 1A000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]], acc[186:187], v[32:33], [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]]  \n" // 000000001A00: D3E1007C 0DF241BA
"  ds_read_b128  acc[144:147], v4 offset:18560            \n" // 000000001A08: DBFE4880 90000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]], acc[188:189], v[32:33], [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]]  \n" // 000000001A10: D3E10080 0E0241BC
"  ds_read_b128  acc[148:151], v4 offset:18624            \n" // 000000001A18: DBFE48C0 94000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]], acc[190:191], v[32:33], [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]]  \n" // 000000001A20: D3E10084 0E1241BE
"  ds_read_b128  acc[152:155], v4 offset:18816            \n" // 000000001A28: DBFE4980 98000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]], acc[192:193], v[32:33], [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]]  \n" // 000000001A30: D3E10088 0E2241C0
"  ds_read_b128  acc[156:159], v4 offset:18880            \n" // 000000001A38: DBFE49C0 9C000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]], acc[194:195], v[32:33], [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]]  \n" // 000000001A40: D3E1008C 0E3241C2
"  ds_read_b128  acc[160:163], v4 offset:19072            \n" // 000000001A48: DBFE4A80 A0000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]], acc[196:197], v[32:33], [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]]  \n" // 000000001A50: D3E10090 0E4241C4
"  ds_read_b128  acc[164:167], v4 offset:19136            \n" // 000000001A58: DBFE4AC0 A4000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]], acc[198:199], v[32:33], [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]]  \n" // 000000001A60: D3E10094 0E5241C6
"  ds_read_b128  acc[168:171], v4 offset:19328            \n" // 000000001A68: DBFE4B80 A8000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]], acc[200:201], v[32:33], [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]]  \n" // 000000001A70: D3E10098 0E6241C8
"  ds_read_b128  acc[172:175], v4 offset:19392            \n" // 000000001A78: DBFE4BC0 AC000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]], acc[202:203], v[32:33], [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]]  \n" // 000000001A80: D3E1009C 0E7241CA
"  s_waitcnt     lgkmcnt(8)                               \n" // 000000001A88: BF8CC87F
"  v_perm_b32    v168, v22, v20, s53                      \n" // 000000001A8C: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000001A94: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 000000001A9C: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000001AA4: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]], acc[204:205], v[32:33], [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]]  \n" // 000000001AAC: D3E100A0 0E8241CC
"  ds_write_b128  v6, v[168:171] offset:37120             \n" // 000000001AB4: D9BE9100 0000A806
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000001ABC: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000001AC4: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000001ACC: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000001AD4: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]], acc[206:207], v[32:33], [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]]  \n" // 000000001ADC: D3E100A4 0E9241CE
"  ds_write_b128  v6, v[168:171] offset:38144             \n" // 000000001AE4: D9BE9500 0000A806
"  ds_read_b64   v[20:21], v5 offset:19584                \n" // 000000001AEC: D8EC4C80 14000005
"  ds_read_b64   v[22:23], v5 offset:24224                \n" // 000000001AF4: D8EC5EA0 16000005
"  ds_read_b64   v[24:25], v5 offset:28864                \n" // 000000001AFC: D8EC70C0 18000005
"  ds_read_b64   v[26:27], v5 offset:33504                \n" // 000000001B04: D8EC82E0 1A000005
"  s_nop         0x0000                                   \n" // 000000001B0C: BF800000
"  s_addk_i32    s70, 0x0001                              \n" // 000000001B10: B7460001
"  s_cmp_lt_i32  s70, s71                                 \n" // 000000001B14: BF044746
"  s_cbranch_scc0  label_092C_%=                             \n" // 000000001B18: BF840265
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000001B1C: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[144:145], acc[0:1], 0  \n" // 000000001B20: D3E10020 1A020190
"  ds_read_b128  acc[176:179], v4 offset:19584            \n" // 000000001B28: DBFE4C80 B0000004
"  ds_read_b128  acc[180:183], v4 offset:19648            \n" // 000000001B30: DBFE4CC0 B4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[146:147], acc[2:3], v[32:35]  \n" // 000000001B38: D3E10020 1C820592
"  buffer_load_dword  v11, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 000000001B40: E0501000 80060B08
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[148:149], acc[4:5], v[32:35]  \n" // 000000001B48: D3E10020 1C820994
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[150:151], acc[6:7], v[32:35]  \n" // 000000001B50: D3E10020 1C820D96
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[152:153], acc[8:9], v[32:35]  \n" // 000000001B58: D3E10020 1C821198
"  ds_read_b128  acc[184:187], v4 offset:19840            \n" // 000000001B60: DBFE4D80 B8000004
"  ds_read_b128  acc[188:191], v4 offset:19904            \n" // 000000001B68: DBFE4DC0 BC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[154:155], acc[10:11], v[32:35]  \n" // 000000001B70: D3E10020 1C82159A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[156:157], acc[12:13], v[32:35]  \n" // 000000001B78: D3E10020 1C82199C
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[158:159], acc[14:15], v[32:35]  \n" // 000000001B80: D3E10020 1C821D9E
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000001B88: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[160:161], acc[16:17], v[32:35]  \n" // 000000001B8C: D3E10020 1C8221A0
"  ds_read_b128  acc[192:195], v4 offset:20096            \n" // 000000001B94: DBFE4E80 C0000004
"  ds_read_b128  acc[196:199], v4 offset:20160            \n" // 000000001B9C: DBFE4EC0 C4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[162:163], acc[18:19], v[32:35]  \n" // 000000001BA4: D3E10020 1C8225A2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[164:165], acc[20:21], v[32:35]  \n" // 000000001BAC: D3E10020 1C8229A4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[166:167], acc[22:23], v[32:35]  \n" // 000000001BB4: D3E10020 1C822DA6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[168:169], acc[24:25], v[32:35]  \n" // 000000001BBC: D3E10020 1C8231A8
"  ds_read_b128  acc[200:203], v4 offset:20352            \n" // 000000001BC4: DBFE4F80 C8000004
"  ds_read_b128  acc[204:207], v4 offset:20416            \n" // 000000001BCC: DBFE4FC0 CC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[170:171], acc[26:27], v[32:35]  \n" // 000000001BD4: D3E10020 1C8235AA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[172:173], acc[28:29], v[32:35]  \n" // 000000001BDC: D3E10020 1C8239AC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[174:175], acc[30:31], v[32:35]  \n" // 000000001BE4: D3E10020 1C823DAE
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000001BEC: BF8CC47F
"  s_barrier                                              \n" // 000000001BF0: BF8A0000
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[176:177], acc[32:33], v[32:35]  \n" // 000000001BF4: D3E10020 1C8241B0
"  ds_read_b128  acc[208:211], v4 offset:20608            \n" // 000000001BFC: DBFE5080 D0000004
"  ds_read_b128  acc[212:215], v4 offset:20672            \n" // 000000001C04: DBFE50C0 D4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[178:179], acc[34:35], v[32:35]  \n" // 000000001C0C: D3E10020 1C8245B2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[180:181], acc[36:37], v[32:35]  \n" // 000000001C14: D3E10020 1C8249B4
"  v_perm_b32    v168, v22, v20, s53                      \n" // 000000001C1C: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000001C24: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 000000001C2C: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000001C34: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[182:183], acc[38:39], v[32:35]  \n" // 000000001C3C: D3E10020 1C824DB6
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000001C44: E0511000 80050012
"  s_add_u32     m0, 0, s37                               \n" // 000000001C4C: 807C2580
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[184:185], acc[40:41], v[32:35]  \n" // 000000001C50: D3E10020 1C8251B8
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 000000001C58: D9BEB100 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[186:187], acc[42:43], v[32:35]  \n" // 000000001C60: D3E10020 1C8255BA
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000001C68: E0511000 80050013
"  s_add_u32     m0, 0x00000080, s36                      \n" // 000000001C70: 807C24FF 00000080
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[188:189], acc[44:45], v[32:35]  \n" // 000000001C78: D3E10020 1C8259BC
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000001C80: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000001C88: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000001C90: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000001C98: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[190:191], acc[46:47], v[32:35]  \n" // 000000001CA0: D3E10020 1C825DBE
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000001CA8: E0511080 80050012
"  s_add_u32     m0, 0x00000080, s37                      \n" // 000000001CB0: 807C25FF 00000080
"  s_waitcnt     lgkmcnt(1)                               \n" // 000000001CB8: BF8CC17F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[192:193], acc[48:49], v[32:35]  \n" // 000000001CBC: D3E10020 1C8261C0
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 000000001CC4: D9BEB500 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[194:195], acc[50:51], v[32:35]  \n" // 000000001CCC: D3E10020 1C8265C2
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000001CD4: E0511080 80050013
"  s_add_u32     m0, 0x00000100, s36                      \n" // 000000001CDC: 807C24FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[196:197], acc[52:53], v[32:35]  \n" // 000000001CE4: D3E10020 1C8269C4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[198:199], acc[54:55], v[32:35]  \n" // 000000001CEC: D3E10020 1C826DC6
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000001CF4: E0511100 80050012
"  s_add_u32     m0, 0x00000100, s37                      \n" // 000000001CFC: 807C25FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[200:201], acc[56:57], v[32:35]  \n" // 000000001D04: D3E10020 1C8271C8
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[202:203], acc[58:59], v[32:35]  \n" // 000000001D0C: D3E10020 1C8275CA
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 000000001D14: E0511100 80050013
"  s_add_u32     m0, 0x00000180, s36                      \n" // 000000001D1C: 807C24FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[204:205], acc[60:61], v[32:35]  \n" // 000000001D24: D3E10020 1C8279CC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[206:207], acc[62:63], v[32:35]  \n" // 000000001D2C: D3E10020 1C827DCE
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 000000001D34: E0511180 80050012
"  s_add_u32     m0, 0x00000180, s37                      \n" // 000000001D3C: 807C25FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[208:209], acc[64:65], v[32:35]  \n" // 000000001D44: D3E10020 1C8281D0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[210:211], acc[66:67], v[32:35]  \n" // 000000001D4C: D3E10020 1C8285D2
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 000000001D54: E0511180 80050013
"  s_add_u32     m0, 0x00000200, s36                      \n" // 000000001D5C: 807C24FF 00000200
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[212:213], acc[68:69], v[32:35]  \n" // 000000001D64: D3E10020 1C8289D4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[214:215], acc[70:71], v[32:35]  \n" // 000000001D6C: D3E10020 1C828DD6
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000001D74: E0511200 80050012
"  s_add_u32     m0, 0x00000200, s37                      \n" // 000000001D7C: 807C25FF 00000200
"  v_add_u32     v8, s73, v8                              \n" // 000000001D84: 68101049
"  s_cmp_le_i32  s83, s82                                 \n" // 000000001D88: BF055253
"  s_cbranch_scc1  label_0788_%=                             \n" // 000000001D8C: BF850024
"  v_mov_b32     v25, 0xff800000                          \n" // 000000001D90: 7E3202FF FF800000
"  s_add_u32     s57, s82, 0                              \n" // 000000001D98: 80398052
"  v_mov_b32     v24, s57                                 \n" // 000000001D9C: 7E300239
"  v_add_u32     v24, %[wave_id], v24                             \n" // 000000001DA0: 68303007
"  s_sub_u32     s56, s83, 15                             \n" // 000000001DA4: 80B88F53
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000001DA8: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 000000001DAC: 0C282884
"  v_add_u32     v20, s56, v20                            \n" // 000000001DB0: 68282838
"  v_add_u32     v21, 1, v20                              \n" // 000000001DB4: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 000000001DB8: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 000000001DBC: 682E2883
"  v_cmp_le_u32  s[38:39], v20, v24                       \n" // 000000001DC0: D0CB0026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000001DC8: 682828C0
"  s_nop         0x0000                                   \n" // 000000001DCC: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000001DD0: D1000020 009A4119
"  v_cmp_le_u32  s[38:39], v21, v24                       \n" // 000000001DD8: D0CB0026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000001DE0: 682A2AC0
"  s_nop         0x0000                                   \n" // 000000001DE4: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000001DE8: D1000021 009A4319
"  v_cmp_le_u32  s[38:39], v22, v24                       \n" // 000000001DF0: D0CB0026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000001DF8: 682C2CC0
"  s_nop         0x0000                                   \n" // 000000001DFC: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 000000001E00: D1000022 009A4519
"  v_cmp_le_u32  s[38:39], v23, v24                       \n" // 000000001E08: D0CB0026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 000000001E10: 682E2EC0
"  s_nop         0x0000                                   \n" // 000000001E14: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 000000001E18: D1000023 009A4719
"label_0788_%=: \n"
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000001E20: BF8CC07F
"  s_barrier                                              \n" // 000000001E24: BF8A0000
"  v_max3_f32    v24, v32, v33, v32                       \n" // 000000001E28: D1D30018 04824320
"  v_max3_f32    v24, v34, v35, v24                       \n" // 000000001E30: D1D30018 04624722
"  ds_write_b32  v3, v24 offset:53504                     \n" // 000000001E38: D81AD100 00001803
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000001E40: E0511200 80050013
"  s_add_u32     m0, 0x00000280, s36                      \n" // 000000001E48: 807C24FF 00000280
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000001E50: E0511280 80050012
"  s_add_u32     m0, 0x00000280, s37                      \n" // 000000001E58: 807C25FF 00000280
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000001E60: BF8CC07F
"  ds_read_b32   v20, v2 offset:53504                     \n" // 000000001E64: D86CD100 14000002
"  ds_read_b32   v21, v2 offset:53568                     \n" // 000000001E6C: D86CD140 15000002
"  ds_read_b32   v22, v2 offset:53632                     \n" // 000000001E74: D86CD180 16000002
"  ds_read_b32   v23, v2 offset:53696                     \n" // 000000001E7C: D86CD1C0 17000002
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000001E84: E0511280 80050013
"  s_add_u32     m0, 0x00000300, s36                      \n" // 000000001E8C: 807C24FF 00000300
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000001E94: E0511300 80050012
"  s_add_u32     m0, 0x00000300, s37                      \n" // 000000001E9C: 807C25FF 00000300
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000001EA4: E0511300 80050013
"  s_add_u32     m0, 0x00000380, s36                      \n" // 000000001EAC: 807C24FF 00000380
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000001EB4: BF8CC07F
"  v_max3_f32    v24, v20, v21, v24                       \n" // 000000001EB8: D1D30018 04622B14
"  v_max3_f32    v24, v22, v23, v24                       \n" // 000000001EC0: D1D30018 04622F16
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000001EC8: E0511380 80050012
"  s_add_u32     m0, 0x00000380, s37                      \n" // 000000001ED0: 807C25FF 00000380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000001ED8: E0511380 80050013
"  s_add_u32     m0, 0x00000400, s36                      \n" // 000000001EE0: 807C24FF 00000400
"  ds_read_b128  acc[144:147], v7 offset:37120            \n" // 000000001EE8: DBFE9100 90000007
"  ds_read_b128  acc[148:151], v7 offset:38144            \n" // 000000001EF0: DBFE9500 94000007
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v182, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000400, s37                      \n" // 000000001F00: 807C25FF 00000400
"  ds_read_b128  acc[152:155], v7 offset:39168            \n" // 000000001F08: DBFE9900 98000007
"  ds_read_b128  acc[156:159], v7 offset:40192            \n" // 000000001F10: DBFE9D00 9C000007
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v183, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000480, s36                      \n" // 000000001F20: 807C24FF 00000480
"  ds_read_b128  acc[160:163], v7 offset:41216            \n" // 000000001F28: DBFEA100 A0000007
"  ds_read_b128  acc[164:167], v7 offset:42240            \n" // 000000001F30: DBFEA500 A4000007
"  ds_read_b128  acc[168:171], v7 offset:43264            \n" // 000000001F38: DBFEA900 A8000007
"  ds_read_b128  acc[172:175], v7 offset:44288            \n" // 000000001F40: DBFEAD00 AC000007
"  v_mov_b32     v25, 0xff7fffff                          \n" // 000000001F48: 7E3202FF FF7FFFFF
"  v_cmp_eq_u32  s[38:39], v25, v12                       \n" // 000000001F50: D0CA0026 00021919
"  v_max_f32     v20, v24, v12                            \n" // 000000001F58: 16281918
"  v_sub_f32     v16, v12, v20                            \n" // 000000001F5C: 0420290C
"  v_cndmask_b32  v16, v16, 0, s[38:39]                   \n" // 000000001F60: D1000010 00990110
"  v_mov_b32     v12, v20                                 \n" // 000000001F68: 7E180314
"  v_mul_f32     v21, s5, v20                             \n" // 000000001F6C: 0A2A2805
"  v_mul_f32     v16, s5, v16                             \n" // 000000001F70: 0A202005
"  v_exp_f32     v16, v16                                 \n" // 000000001F74: 7E204110
"  v_fma_f32     v32, v32, s5, -v21                       \n" // 000000001F78: D1CB0020 84540B20
"  v_fma_f32     v33, v33, s5, -v21                       \n" // 000000001F80: D1CB0021 84540B21
"  v_fma_f32     v34, v34, s5, -v21                       \n" // 000000001F88: D1CB0022 84540B22
"  v_fma_f32     v35, v35, s5, -v21                       \n" // 000000001F90: D1CB0023 84540B23
"  v_exp_f32     v32, v32                                 \n" // 000000001F98: 7E404120
"  v_exp_f32     v33, v33                                 \n" // 000000001F9C: 7E424121
"  v_exp_f32     v34, v34                                 \n" // 000000001FA0: 7E444122
"  v_exp_f32     v35, v35                                 \n" // 000000001FA4: 7E464123
"  v_mul_f32     v14, v16, v14                            \n" // 000000001FA8: 0A1C1D10
"  v_mov_b32     v22, v32                                 \n" // 000000001FAC: 7E2C0320
"  v_add_f32     v22, v33, v22                            \n" // 000000001FB0: 022C2D21
"  v_add_f32     v22, v34, v22                            \n" // 000000001FB4: 022C2D22
"  v_add_f32     v22, v35, v22                            \n" // 000000001FB8: 022C2D23
"  v_add_f32     v14, v22, v14                            \n" // 000000001FBC: 021C1D16
"  v_mov_b32     v29, 0xffff0000                          \n" // 000000001FC0: 7E3A02FF FFFF0000
"  v_mov_b32     v30, 0x7fff0000                          \n" // 000000001FC8: 7E3C02FF 7FFF0000
"  v_mov_b32     v31, 0x00007fff                          \n" // 000000001FD0: 7E3E02FF 00007FFF
"  v_cmp_u_f32   s[38:39], v32, v32                       \n" // 000000001FD8: D0480026 00024120
"  v_add3_u32    v28, v32, v31, 1                         \n" // 000000001FE0: D1FF001C 02063F20
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000001FE8: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v33, v33                       \n" // 000000001FF0: D0480026 00024321
"  v_add3_u32    v28, v33, v31, 1                         \n" // 000000001FF8: D1FF001C 02063F21
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000002000: D1000015 009A3D1C
"  v_perm_b32    v32, v21, v20, s52                       \n" // 000000002008: D1ED0020 00D22915
"  v_cmp_u_f32   s[38:39], v34, v34                       \n" // 000000002010: D0480026 00024522
"  v_add3_u32    v28, v34, v31, 1                         \n" // 000000002018: D1FF001C 02063F22
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000002020: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v35, v35                       \n" // 000000002028: D0480026 00024723
"  v_add3_u32    v28, v35, v31, 1                         \n" // 000000002030: D1FF001C 02063F23
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000002038: D1000015 009A3D1C
"  v_perm_b32    v33, v21, v20, s52                       \n" // 000000002040: D1ED0021 00D22915
"  s_nop         0x0002                                   \n" // 000000002048: BF800002
"  s_add_u32     s83, s84, s83                            \n" // 00000000204C: 80535354
"  s_nop         0x0000                                   \n" // 000000002050: BF800000
"  v_mul_u32_u24  v18, v10, v9 row_newbcast:0             \n" // 000000002054: 102412FA FF01500A
"  v_mul_u32_u24  v19, v10, v9 row_newbcast:8             \n" // 00000000205C: 102612FA FF01580A
"  v_add_u32     v18, v18, v1                             \n" // 000000002064: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 000000002068: 68260313
"  v_mul_u32_u24  v182, v10, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v10, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s34                                  \n" // 00000000206C: BEFC0022
"  v_mov_b32     v22, v16                                 \n" // 000000002070: 7E2C0310
"  v_mov_b32     v23, v16                                 \n" // 000000002074: 7E2E0310
"  v_pk_mul_f32  [%[o_regs_0], %[o_regs_1]], v[22:23], [%[o_regs_0], %[o_regs_1]]             \n" // 000000002078: D3B14028 18025116
"  v_pk_mul_f32  [%[o_regs_2], %[o_regs_3]], v[22:23], [%[o_regs_2], %[o_regs_3]]             \n" // 000000002080: D3B1402A 18025516
"  v_pk_mul_f32  [%[o_regs_4], %[o_regs_5]], v[22:23], [%[o_regs_4], %[o_regs_5]]             \n" // 000000002088: D3B1402C 18025916
"  v_pk_mul_f32  [%[o_regs_6], %[o_regs_7]], v[22:23], [%[o_regs_6], %[o_regs_7]]             \n" // 000000002090: D3B1402E 18025D16
"  v_pk_mul_f32  [%[o_regs_8], %[o_regs_9]], v[22:23], [%[o_regs_8], %[o_regs_9]]             \n" // 000000002098: D3B14030 18026116
"  v_pk_mul_f32  [%[o_regs_10], %[o_regs_11]], v[22:23], [%[o_regs_10], %[o_regs_11]]             \n" // 0000000020A0: D3B14032 18026516
"  v_pk_mul_f32  [%[o_regs_12], %[o_regs_13]], v[22:23], [%[o_regs_12], %[o_regs_13]]             \n" // 0000000020A8: D3B14034 18026916
"  v_pk_mul_f32  [%[o_regs_14], %[o_regs_15]], v[22:23], [%[o_regs_14], %[o_regs_15]]             \n" // 0000000020B0: D3B14036 18026D16
"  v_pk_mul_f32  [%[o_regs_16], %[o_regs_17]], v[22:23], [%[o_regs_16], %[o_regs_17]]             \n" // 0000000020B8: D3B14038 18027116
"  v_pk_mul_f32  [%[o_regs_18], %[o_regs_19]], v[22:23], [%[o_regs_18], %[o_regs_19]]             \n" // 0000000020C0: D3B1403A 18027516
"  v_pk_mul_f32  [%[o_regs_20], %[o_regs_21]], v[22:23], [%[o_regs_20], %[o_regs_21]]             \n" // 0000000020C8: D3B1403C 18027916
"  v_pk_mul_f32  [%[o_regs_22], %[o_regs_23]], v[22:23], [%[o_regs_22], %[o_regs_23]]             \n" // 0000000020D0: D3B1403E 18027D16
"  v_pk_mul_f32  [%[o_regs_24], %[o_regs_25]], v[22:23], [%[o_regs_24], %[o_regs_25]]             \n" // 0000000020D8: D3B14040 18028116
"  v_pk_mul_f32  [%[o_regs_26], %[o_regs_27]], v[22:23], [%[o_regs_26], %[o_regs_27]]             \n" // 0000000020E0: D3B14042 18028516
"  v_pk_mul_f32  [%[o_regs_28], %[o_regs_29]], v[22:23], [%[o_regs_28], %[o_regs_29]]             \n" // 0000000020E8: D3B14044 18028916
"  v_pk_mul_f32  [%[o_regs_30], %[o_regs_31]], v[22:23], [%[o_regs_30], %[o_regs_31]]             \n" // 0000000020F0: D3B14046 18028D16
"  v_pk_mul_f32  [%[o_regs_32], %[o_regs_33]], v[22:23], [%[o_regs_32], %[o_regs_33]]             \n" // 0000000020F8: D3B14048 18029116
"  v_pk_mul_f32  [%[o_regs_34], %[o_regs_35]], v[22:23], [%[o_regs_34], %[o_regs_35]]             \n" // 000000002100: D3B1404A 18029516
"  v_pk_mul_f32  [%[o_regs_36], %[o_regs_37]], v[22:23], [%[o_regs_36], %[o_regs_37]]             \n" // 000000002108: D3B1404C 18029916
"  v_pk_mul_f32  [%[o_regs_38], %[o_regs_39]], v[22:23], [%[o_regs_38], %[o_regs_39]]             \n" // 000000002110: D3B1404E 18029D16
"  v_pk_mul_f32  [%[o_regs_40], %[o_regs_41]], v[22:23], [%[o_regs_40], %[o_regs_41]]             \n" // 000000002118: D3B14050 1802A116
"  v_pk_mul_f32  [%[o_regs_42], %[o_regs_43]], v[22:23], [%[o_regs_42], %[o_regs_43]]             \n" // 000000002120: D3B14052 1802A516
"  v_pk_mul_f32  [%[o_regs_44], %[o_regs_45]], v[22:23], [%[o_regs_44], %[o_regs_45]]             \n" // 000000002128: D3B14054 1802A916
"  v_pk_mul_f32  [%[o_regs_46], %[o_regs_47]], v[22:23], [%[o_regs_46], %[o_regs_47]]             \n" // 000000002130: D3B14056 1802AD16
"  v_pk_mul_f32  [%[o_regs_48], %[o_regs_49]], v[22:23], [%[o_regs_48], %[o_regs_49]]             \n" // 000000002138: D3B14058 1802B116
"  v_pk_mul_f32  [%[o_regs_50], %[o_regs_51]], v[22:23], [%[o_regs_50], %[o_regs_51]]             \n" // 000000002140: D3B1405A 1802B516
"  v_pk_mul_f32  [%[o_regs_52], %[o_regs_53]], v[22:23], [%[o_regs_52], %[o_regs_53]]             \n" // 000000002148: D3B1405C 1802B916
"  v_pk_mul_f32  [%[o_regs_54], %[o_regs_55]], v[22:23], [%[o_regs_54], %[o_regs_55]]             \n" // 000000002150: D3B1405E 1802BD16
"  v_pk_mul_f32  [%[o_regs_56], %[o_regs_57]], v[22:23], [%[o_regs_56], %[o_regs_57]]             \n" // 000000002158: D3B14060 1802C116
"  v_pk_mul_f32  [%[o_regs_58], %[o_regs_59]], v[22:23], [%[o_regs_58], %[o_regs_59]]             \n" // 000000002160: D3B14062 1802C516
"  v_pk_mul_f32  [%[o_regs_60], %[o_regs_61]], v[22:23], [%[o_regs_60], %[o_regs_61]]         \n" // 000000002168: D3B14064 1802C916
"  v_pk_mul_f32  [%[o_regs_62], %[o_regs_63]], v[22:23], [%[o_regs_62], %[o_regs_63]]         \n" // 000000002170: D3B14066 1802CD16
"  v_pk_mul_f32  [%[o_regs_64], %[o_regs_65]], v[22:23], [%[o_regs_64], %[o_regs_65]]         \n" // 000000002178: D3B14068 1802D116
"  v_pk_mul_f32  [%[o_regs_66], %[o_regs_67]], v[22:23], [%[o_regs_66], %[o_regs_67]]         \n" // 000000002180: D3B1406A 1802D516
"  v_pk_mul_f32  [%[o_regs_68], %[o_regs_69]], v[22:23], [%[o_regs_68], %[o_regs_69]]         \n" // 000000002188: D3B1406C 1802D916
"  v_pk_mul_f32  [%[o_regs_70], %[o_regs_71]], v[22:23], [%[o_regs_70], %[o_regs_71]]         \n" // 000000002190: D3B1406E 1802DD16
"  v_pk_mul_f32  [%[o_regs_72], %[o_regs_73]], v[22:23], [%[o_regs_72], %[o_regs_73]]         \n" // 000000002198: D3B14070 1802E116
"  v_pk_mul_f32  [%[o_regs_74], %[o_regs_75]], v[22:23], [%[o_regs_74], %[o_regs_75]]         \n" // 0000000021A0: D3B14072 1802E516
"  v_pk_mul_f32  [%[o_regs_76], %[o_regs_77]], v[22:23], [%[o_regs_76], %[o_regs_77]]         \n" // 0000000021A8: D3B14074 1802E916
"  v_pk_mul_f32  [%[o_regs_78], %[o_regs_79]], v[22:23], [%[o_regs_78], %[o_regs_79]]         \n" // 0000000021B0: D3B14076 1802ED16
"  v_pk_mul_f32  [%[o_regs_80], %[o_regs_81]], v[22:23], [%[o_regs_80], %[o_regs_81]]         \n" // 0000000021B8: D3B14078 1802F116
"  v_pk_mul_f32  [%[o_regs_82], %[o_regs_83]], v[22:23], [%[o_regs_82], %[o_regs_83]]         \n" // 0000000021C0: D3B1407A 1802F516
"  v_pk_mul_f32  [%[o_regs_84], %[o_regs_85]], v[22:23], [%[o_regs_84], %[o_regs_85]]         \n" // 0000000021C8: D3B1407C 1802F916
"  v_pk_mul_f32  [%[o_regs_86], %[o_regs_87]], v[22:23], [%[o_regs_86], %[o_regs_87]]         \n" // 0000000021D0: D3B1407E 1802FD16
"  v_pk_mul_f32  [%[o_regs_88], %[o_regs_89]], v[22:23], [%[o_regs_88], %[o_regs_89]]         \n" // 0000000021D8: D3B14080 18030116
"  v_pk_mul_f32  [%[o_regs_90], %[o_regs_91]], v[22:23], [%[o_regs_90], %[o_regs_91]]         \n" // 0000000021E0: D3B14082 18030516
"  v_pk_mul_f32  [%[o_regs_92], %[o_regs_93]], v[22:23], [%[o_regs_92], %[o_regs_93]]         \n" // 0000000021E8: D3B14084 18030916
"  v_pk_mul_f32  [%[o_regs_94], %[o_regs_95]], v[22:23], [%[o_regs_94], %[o_regs_95]]         \n" // 0000000021F0: D3B14086 18030D16
"  v_pk_mul_f32  [%[o_regs_96], %[o_regs_97]], v[22:23], [%[o_regs_96], %[o_regs_97]]         \n" // 0000000021F8: D3B14088 18031116
"  v_pk_mul_f32  [%[o_regs_98], %[o_regs_99]], v[22:23], [%[o_regs_98], %[o_regs_99]]         \n" // 000000002200: D3B1408A 18031516
"  v_pk_mul_f32  [%[o_regs_100], %[o_regs_101]], v[22:23], [%[o_regs_100], %[o_regs_101]]         \n" // 000000002208: D3B1408C 18031916
"  v_pk_mul_f32  [%[o_regs_102], %[o_regs_103]], v[22:23], [%[o_regs_102], %[o_regs_103]]         \n" // 000000002210: D3B1408E 18031D16
"  v_pk_mul_f32  [%[o_regs_104], %[o_regs_105]], v[22:23], [%[o_regs_104], %[o_regs_105]]         \n" // 000000002218: D3B14090 18032116
"  v_pk_mul_f32  [%[o_regs_106], %[o_regs_107]], v[22:23], [%[o_regs_106], %[o_regs_107]]         \n" // 000000002220: D3B14092 18032516
"  v_pk_mul_f32  [%[o_regs_108], %[o_regs_109]], v[22:23], [%[o_regs_108], %[o_regs_109]]         \n" // 000000002228: D3B14094 18032916
"  v_pk_mul_f32  [%[o_regs_110], %[o_regs_111]], v[22:23], [%[o_regs_110], %[o_regs_111]]         \n" // 000000002230: D3B14096 18032D16
"  v_pk_mul_f32  [%[o_regs_112], %[o_regs_113]], v[22:23], [%[o_regs_112], %[o_regs_113]]         \n" // 000000002238: D3B14098 18033116
"  v_pk_mul_f32  [%[o_regs_114], %[o_regs_115]], v[22:23], [%[o_regs_114], %[o_regs_115]]         \n" // 000000002240: D3B1409A 18033516
"  v_pk_mul_f32  [%[o_regs_116], %[o_regs_117]], v[22:23], [%[o_regs_116], %[o_regs_117]]         \n" // 000000002248: D3B1409C 18033916
"  v_pk_mul_f32  [%[o_regs_118], %[o_regs_119]], v[22:23], [%[o_regs_118], %[o_regs_119]]         \n" // 000000002250: D3B1409E 18033D16
"  v_pk_mul_f32  [%[o_regs_120], %[o_regs_121]], v[22:23], [%[o_regs_120], %[o_regs_121]]         \n" // 000000002258: D3B140A0 18034116
"  v_pk_mul_f32  [%[o_regs_122], %[o_regs_123]], v[22:23], [%[o_regs_122], %[o_regs_123]]         \n" // 000000002260: D3B140A2 18034516
"  v_pk_mul_f32  [%[o_regs_124], %[o_regs_125]], v[22:23], [%[o_regs_124], %[o_regs_125]]         \n" // 000000002268: D3B140A4 18034916
"  v_pk_mul_f32  [%[o_regs_126], %[o_regs_127]], v[22:23], [%[o_regs_126], %[o_regs_127]]         \n" // 000000002270: D3B140A6 18034D16
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000002278: BF8CC07F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]], acc[144:145], v[32:33], [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]]  \n" // 00000000227C: D3E10028 0CA24190
"  ds_read_b128  acc[176:179], v7 offset:45312            \n" // 000000002284: DBFEB100 B0000007
"  ds_read_b128  acc[180:183], v7 offset:46336            \n" // 00000000228C: DBFEB500 B4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]], acc[146:147], v[32:33], [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]]  \n" // 000000002294: D3E1002C 0CB24192
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]], acc[148:149], v[32:33], [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]]  \n" // 00000000229C: D3E10030 0CC24194
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]], acc[150:151], v[32:33], [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]]  \n" // 0000000022A4: D3E10034 0CD24196
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]], acc[152:153], v[32:33], [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]]  \n" // 0000000022AC: D3E10038 0CE24198
"  ds_read_b128  acc[184:187], v7 offset:47360            \n" // 0000000022B4: DBFEB900 B8000007
"  ds_read_b128  acc[188:191], v7 offset:48384            \n" // 0000000022BC: DBFEBD00 BC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]], acc[154:155], v[32:33], [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]]  \n" // 0000000022C4: D3E1003C 0CF2419A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]], acc[156:157], v[32:33], [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]]  \n" // 0000000022CC: D3E10040 0D02419C
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]], acc[158:159], v[32:33], [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]]  \n" // 0000000022D4: D3E10044 0D12419E
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]], acc[160:161], v[32:33], [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]]  \n" // 0000000022DC: D3E10048 0D2241A0
"  ds_read_b128  acc[192:195], v7 offset:49408            \n" // 0000000022E4: DBFEC100 C0000007
"  ds_read_b128  acc[196:199], v7 offset:50432            \n" // 0000000022EC: DBFEC500 C4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]], acc[162:163], v[32:33], [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]]  \n" // 0000000022F4: D3E1004C 0D3241A2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]], acc[164:165], v[32:33], [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]]  \n" // 0000000022FC: D3E10050 0D4241A4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]], acc[166:167], v[32:33], [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]]  \n" // 000000002304: D3E10054 0D5241A6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]], acc[168:169], v[32:33], [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]]  \n" // 00000000230C: D3E10058 0D6241A8
"  ds_read_b128  acc[200:203], v7 offset:51456            \n" // 000000002314: DBFEC900 C8000007
"  ds_read_b128  acc[204:207], v7 offset:52480            \n" // 00000000231C: DBFECD00 CC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]], acc[170:171], v[32:33], [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]]  \n" // 000000002324: D3E1005C 0D7241AA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]], acc[172:173], v[32:33], [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]]  \n" // 00000000232C: D3E10060 0D8241AC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]], acc[174:175], v[32:33], [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]]  \n" // 000000002334: D3E10064 0D9241AE
"  s_waitcnt     lgkmcnt(4)                               \n" // 00000000233C: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]], acc[176:177], v[32:33], [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]]  \n" // 000000002340: D3E10068 0DA241B0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]], acc[178:179], v[32:33], [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]]  \n" // 000000002348: D3E1006C 0DB241B2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]], acc[180:181], v[32:33], [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]]  \n" // 000000002350: D3E10070 0DC241B4
"  s_waitcnt     vmcnt(18) & lgkmcnt(0)                   \n" // 000000002358: BF8C4072
"  s_barrier                                              \n" // 00000000235C: BF8A0000
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]], acc[182:183], v[32:33], [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]]  \n" // 000000002360: D3E10074 0DD241B6
"  ds_read_b64   v[20:21], v5                             \n" // 000000002368: D8EC0000 14000005
"  ds_read_b64   v[22:23], v5 offset:4640                 \n" // 000000002370: D8EC1220 16000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]], acc[184:185], v[32:33], [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]]  \n" // 000000002378: D3E10078 0DE241B8
"  ds_read_b64   v[24:25], v5 offset:9280                 \n" // 000000002380: D8EC2440 18000005
"  ds_read_b64   v[26:27], v5 offset:13920                \n" // 000000002388: D8EC3660 1A000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]], acc[186:187], v[32:33], [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]]  \n" // 000000002390: D3E1007C 0DF241BA
"  ds_read_b128  acc[144:147], v4                         \n" // 000000002398: DBFE0000 90000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]], acc[188:189], v[32:33], [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]]  \n" // 0000000023A0: D3E10080 0E0241BC
"  ds_read_b128  acc[148:151], v4 offset:64               \n" // 0000000023A8: DBFE0040 94000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]], acc[190:191], v[32:33], [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]]  \n" // 0000000023B0: D3E10084 0E1241BE
"  ds_read_b128  acc[152:155], v4 offset:256              \n" // 0000000023B8: DBFE0100 98000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]], acc[192:193], v[32:33], [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]]  \n" // 0000000023C0: D3E10088 0E2241C0
"  ds_read_b128  acc[156:159], v4 offset:320              \n" // 0000000023C8: DBFE0140 9C000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]], acc[194:195], v[32:33], [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]]  \n" // 0000000023D0: D3E1008C 0E3241C2
"  ds_read_b128  acc[160:163], v4 offset:512              \n" // 0000000023D8: DBFE0200 A0000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]], acc[196:197], v[32:33], [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]]  \n" // 0000000023E0: D3E10090 0E4241C4
"  ds_read_b128  acc[164:167], v4 offset:576              \n" // 0000000023E8: DBFE0240 A4000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]], acc[198:199], v[32:33], [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]]  \n" // 0000000023F0: D3E10094 0E5241C6
"  ds_read_b128  acc[168:171], v4 offset:768              \n" // 0000000023F8: DBFE0300 A8000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]], acc[200:201], v[32:33], [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]]  \n" // 000000002400: D3E10098 0E6241C8
"  ds_read_b128  acc[172:175], v4 offset:832              \n" // 000000002408: DBFE0340 AC000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]], acc[202:203], v[32:33], [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]]  \n" // 000000002410: D3E1009C 0E7241CA
"  s_waitcnt     lgkmcnt(8)                               \n" // 000000002418: BF8CC87F
"  v_perm_b32    v168, v22, v20, s53                      \n" // 00000000241C: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000002424: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 00000000242C: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000002434: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]], acc[204:205], v[32:33], [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]]  \n" // 00000000243C: D3E100A0 0E8241CC
"  ds_write_b128  v6, v[168:171] offset:37120             \n" // 000000002444: D9BE9100 0000A806
"  v_perm_b32    v168, v23, v21, s53                      \n" // 00000000244C: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000002454: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 00000000245C: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000002464: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]], acc[206:207], v[32:33], [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]]  \n" // 00000000246C: D3E100A4 0E9241CE
"  ds_write_b128  v6, v[168:171] offset:38144             \n" // 000000002474: D9BE9500 0000A806
"  ds_read_b64   v[20:21], v5 offset:1024                 \n" // 00000000247C: D8EC0400 14000005
"  ds_read_b64   v[22:23], v5 offset:5664                 \n" // 000000002484: D8EC1620 16000005
"  ds_read_b64   v[24:25], v5 offset:10304                \n" // 00000000248C: D8EC2840 18000005
"  ds_read_b64   v[26:27], v5 offset:14944                \n" // 000000002494: D8EC3A60 1A000005
"  s_nop         0x0000                                   \n" // 00000000249C: BF800000
"  s_addk_i32    s70, 0x0001                              \n" // 0000000024A0: B7460001
"  s_cmp_lt_i32  s70, s71                                 \n" // 0000000024A4: BF044746
"  s_cbranch_scc0  label_092C_%=                             \n" // 0000000024A8: BF840001
"  s_branch      label_0463_%=                               \n" // 0000000024AC: BF82FB37
"label_092C_%=: \n"
"  s_nop         0x0000                                   \n" // 0000000024B0: BF800000
"  s_nop         0x0000                                   \n" // 0000000024B4: BF800000
"  s_branch      label_0DF8_%=                               \n" // 0000000024B8: BF8204C9
"label_092F_%=: \n"
"  s_waitcnt     lgkmcnt(4)                               \n" // 0000000024BC: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[144:145], acc[0:1], 0  \n" // 0000000024C0: D3E10020 1A020190
"  buffer_load_dword  v10, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 0000000024C8: E0501000 80060A08
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[146:147], acc[2:3], v[32:35]  \n" // 0000000024D0: D3E10020 1C820592
"  ds_read_b128  acc[176:179], v4 offset:1024             \n" // 0000000024D8: DBFE0400 B0000004
"  ds_read_b128  acc[180:183], v4 offset:1088             \n" // 0000000024E0: DBFE0440 B4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[148:149], acc[4:5], v[32:35]  \n" // 0000000024E8: D3E10020 1C820994
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[150:151], acc[6:7], v[32:35]  \n" // 0000000024F0: D3E10020 1C820D96
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[152:153], acc[8:9], v[32:35]  \n" // 0000000024F8: D3E10020 1C821198
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[154:155], acc[10:11], v[32:35]  \n" // 000000002500: D3E10020 1C82159A
"  ds_read_b128  acc[184:187], v4 offset:1280             \n" // 000000002508: DBFE0500 B8000004
"  ds_read_b128  acc[188:191], v4 offset:1344             \n" // 000000002510: DBFE0540 BC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[156:157], acc[12:13], v[32:35]  \n" // 000000002518: D3E10020 1C82199C
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[158:159], acc[14:15], v[32:35]  \n" // 000000002520: D3E10020 1C821D9E
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000002528: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[160:161], acc[16:17], v[32:35]  \n" // 00000000252C: D3E10020 1C8221A0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[162:163], acc[18:19], v[32:35]  \n" // 000000002534: D3E10020 1C8225A2
"  ds_read_b128  acc[192:195], v4 offset:1536             \n" // 00000000253C: DBFE0600 C0000004
"  ds_read_b128  acc[196:199], v4 offset:1600             \n" // 000000002544: DBFE0640 C4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[164:165], acc[20:21], v[32:35]  \n" // 00000000254C: D3E10020 1C8229A4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[166:167], acc[22:23], v[32:35]  \n" // 000000002554: D3E10020 1C822DA6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[168:169], acc[24:25], v[32:35]  \n" // 00000000255C: D3E10020 1C8231A8
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[170:171], acc[26:27], v[32:35]  \n" // 000000002564: D3E10020 1C8235AA
"  ds_read_b128  acc[200:203], v4 offset:1792             \n" // 00000000256C: DBFE0700 C8000004
"  ds_read_b128  acc[204:207], v4 offset:1856             \n" // 000000002574: DBFE0740 CC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[172:173], acc[28:29], v[32:35]  \n" // 00000000257C: D3E10020 1C8239AC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[174:175], acc[30:31], v[32:35]  \n" // 000000002584: D3E10020 1C823DAE
"  s_waitcnt     lgkmcnt(4)                               \n" // 00000000258C: BF8CC47F
"  s_barrier                                              \n" // 000000002590: BF8A0000
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[176:177], acc[32:33], v[32:35]  \n" // 000000002594: D3E10020 1C8241B0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[178:179], acc[34:35], v[32:35]  \n" // 00000000259C: D3E10020 1C8245B2
"  ds_read_b128  acc[208:211], v4 offset:2048             \n" // 0000000025A4: DBFE0800 D0000004
"  ds_read_b128  acc[212:215], v4 offset:2112             \n" // 0000000025AC: DBFE0840 D4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[180:181], acc[36:37], v[32:35]  \n" // 0000000025B4: D3E10020 1C8249B4
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 0000000025BC: E0511000 80050012
"  s_add_u32     m0, 0, s35                               \n" // 0000000025C4: 807C2380
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[182:183], acc[38:39], v[32:35]  \n" // 0000000025C8: D3E10020 1C824DB6
"  v_perm_b32    v168, v22, v20, s53                      \n" // 0000000025D0: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 0000000025D8: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 0000000025E0: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 0000000025E8: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[184:185], acc[40:41], v[32:35]  \n" // 0000000025F0: D3E10020 1C8251B8
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 0000000025F8: E0511000 80050013
"  s_add_u32     m0, 0x00000080, s34                      \n" // 000000002600: 807C22FF 00000080
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[186:187], acc[42:43], v[32:35]  \n" // 000000002608: D3E10020 1C8255BA
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 000000002610: D9BEB100 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[188:189], acc[44:45], v[32:35]  \n" // 000000002618: D3E10020 1C8259BC
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000002620: E0511080 80050012
"  s_add_u32     m0, 0x00000080, s35                      \n" // 000000002628: 807C23FF 00000080
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[190:191], acc[46:47], v[32:35]  \n" // 000000002630: D3E10020 1C825DBE
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000002638: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000002640: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000002648: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000002650: D1ED00AB 00D2331B
"  s_waitcnt     lgkmcnt(1)                               \n" // 000000002658: BF8CC17F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[192:193], acc[48:49], v[32:35]  \n" // 00000000265C: D3E10020 1C8261C0
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000002664: E0511080 80050013
"  s_add_u32     m0, 0x00000100, s34                      \n" // 00000000266C: 807C22FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[194:195], acc[50:51], v[32:35]  \n" // 000000002674: D3E10020 1C8265C2
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 00000000267C: D9BEB500 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[196:197], acc[52:53], v[32:35]  \n" // 000000002684: D3E10020 1C8269C4
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 00000000268C: E0511100 80050012
"  s_add_u32     m0, 0x00000100, s35                      \n" // 000000002694: 807C23FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[198:199], acc[54:55], v[32:35]  \n" // 00000000269C: D3E10020 1C826DC6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[200:201], acc[56:57], v[32:35]  \n" // 0000000026A4: D3E10020 1C8271C8
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 0000000026AC: E0511100 80050013
"  s_add_u32     m0, 0x00000180, s34                      \n" // 0000000026B4: 807C22FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[202:203], acc[58:59], v[32:35]  \n" // 0000000026BC: D3E10020 1C8275CA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[204:205], acc[60:61], v[32:35]  \n" // 0000000026C4: D3E10020 1C8279CC
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 0000000026CC: E0511180 80050012
"  s_add_u32     m0, 0x00000180, s35                      \n" // 0000000026D4: 807C23FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[206:207], acc[62:63], v[32:35]  \n" // 0000000026DC: D3E10020 1C827DCE
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[208:209], acc[64:65], v[32:35]  \n" // 0000000026E4: D3E10020 1C8281D0
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 0000000026EC: E0511180 80050013
"  s_add_u32     m0, 0x00000200, s34                      \n" // 0000000026F4: 807C22FF 00000200
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[210:211], acc[66:67], v[32:35]  \n" // 0000000026FC: D3E10020 1C8285D2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[212:213], acc[68:69], v[32:35]  \n" // 000000002704: D3E10020 1C8289D4
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 00000000270C: E0511200 80050012
"  s_add_u32     m0, 0x00000200, s35                      \n" // 000000002714: 807C23FF 00000200
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[214:215], acc[70:71], v[32:35]  \n" // 00000000271C: D3E10020 1C828DD6
"  v_add_u32     v8, s73, v8                              \n" // 000000002724: 68101049
"  s_cmp_le_i32  s83, s82                                 \n" // 000000002728: BF055253
"  s_cbranch_scc1  label_09F0_%=                             \n" // 00000000272C: BF850024
"  v_mov_b32     v25, 0xff800000                          \n" // 000000002730: 7E3202FF FF800000
"  s_add_u32     s57, s82, 0                              \n" // 000000002738: 80398052
"  v_mov_b32     v24, s57                                 \n" // 00000000273C: 7E300239
"  v_add_u32     v24, %[wave_id], v24                             \n" // 000000002740: 68303007
"  s_sub_u32     s56, s83, 15                             \n" // 000000002744: 80B88F53
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000002748: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 00000000274C: 0C282884
"  v_add_u32     v20, s56, v20                            \n" // 000000002750: 68282838
"  v_add_u32     v21, 1, v20                              \n" // 000000002754: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 000000002758: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 00000000275C: 682E2883
"  v_cmp_le_u32  s[38:39], v20, v24                       \n" // 000000002760: D0CB0026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000002768: 682828C0
"  s_nop         0x0000                                   \n" // 00000000276C: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000002770: D1000020 009A4119
"  v_cmp_le_u32  s[38:39], v21, v24                       \n" // 000000002778: D0CB0026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000002780: 682A2AC0
"  s_nop         0x0000                                   \n" // 000000002784: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000002788: D1000021 009A4319
"  v_cmp_le_u32  s[38:39], v22, v24                       \n" // 000000002790: D0CB0026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000002798: 682C2CC0
"  s_nop         0x0000                                   \n" // 00000000279C: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 0000000027A0: D1000022 009A4519
"  v_cmp_le_u32  s[38:39], v23, v24                       \n" // 0000000027A8: D0CB0026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 0000000027B0: 682E2EC0
"  s_nop         0x0000                                   \n" // 0000000027B4: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 0000000027B8: D1000023 009A4719
"label_09F0_%=: \n"
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000027C0: BF8CC07F
"  s_barrier                                              \n" // 0000000027C4: BF8A0000
"  v_max3_f32    v24, v32, v33, v32                       \n" // 0000000027C8: D1D30018 04824320
"  v_max3_f32    v24, v34, v35, v24                       \n" // 0000000027D0: D1D30018 04624722
"  ds_write_b32  v3, v24 offset:53504                     \n" // 0000000027D8: D81AD100 00001803
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 0000000027E0: E0511200 80050013
"  s_add_u32     m0, 0x00000280, s34                      \n" // 0000000027E8: 807C22FF 00000280
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 0000000027F0: E0511280 80050012
"  s_add_u32     m0, 0x00000280, s35                      \n" // 0000000027F8: 807C23FF 00000280
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000002800: BF8CC07F
"  ds_read_b32   v20, v2 offset:53504                     \n" // 000000002804: D86CD100 14000002
"  ds_read_b32   v21, v2 offset:53568                     \n" // 00000000280C: D86CD140 15000002
"  ds_read_b32   v22, v2 offset:53632                     \n" // 000000002814: D86CD180 16000002
"  ds_read_b32   v23, v2 offset:53696                     \n" // 00000000281C: D86CD1C0 17000002
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000002824: E0511280 80050013
"  s_add_u32     m0, 0x00000300, s34                      \n" // 00000000282C: 807C22FF 00000300
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000002834: E0511300 80050012
"  s_add_u32     m0, 0x00000300, s35                      \n" // 00000000283C: 807C23FF 00000300
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 000000002844: E0511300 80050013
"  s_add_u32     m0, 0x00000380, s34                      \n" // 00000000284C: 807C22FF 00000380
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000002854: BF8CC07F
"  v_max3_f32    v24, v20, v21, v24                       \n" // 000000002858: D1D30018 04622B14
"  v_max3_f32    v24, v22, v23, v24                       \n" // 000000002860: D1D30018 04622F16
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000002868: E0511380 80050012
"  s_add_u32     m0, 0x00000380, s35                      \n" // 000000002870: 807C23FF 00000380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000002878: E0511380 80050013
"  s_add_u32     m0, 0x00000400, s34                      \n" // 000000002880: 807C22FF 00000400
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v182, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000400, s35                      \n" // 000000002890: 807C23FF 00000400
"  ds_read_b128  acc[144:147], v7 offset:37120            \n" // 000000002898: DBFE9100 90000007
"  ds_read_b128  acc[148:151], v7 offset:38144            \n" // 0000000028A0: DBFE9500 94000007
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v183, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  ;s_add_u32     m0, 0x00000480, s34                      \n" // 0000000028B0: 807C22FF 00000480
"  ds_read_b128  acc[152:155], v7 offset:39168            \n" // 0000000028B8: DBFE9900 98000007
"  ds_read_b128  acc[156:159], v7 offset:40192            \n" // 0000000028C0: DBFE9D00 9C000007
"  ds_read_b128  acc[160:163], v7 offset:41216            \n" // 0000000028C8: DBFEA100 A0000007
"  ds_read_b128  acc[164:167], v7 offset:42240            \n" // 0000000028D0: DBFEA500 A4000007
"  ds_read_b128  acc[168:171], v7 offset:43264            \n" // 0000000028D8: DBFEA900 A8000007
"  ds_read_b128  acc[172:175], v7 offset:44288            \n" // 0000000028E0: DBFEAD00 AC000007
"  v_mov_b32     v25, 0xff7fffff                          \n" // 0000000028E8: 7E3202FF FF7FFFFF
"  v_cmp_eq_u32  s[38:39], v25, v12                       \n" // 0000000028F0: D0CA0026 00021919
"  v_max_f32     v20, v24, v12                            \n" // 0000000028F8: 16281918
"  v_sub_f32     v16, v12, v20                            \n" // 0000000028FC: 0420290C
"  v_cndmask_b32  v16, v16, 0, s[38:39]                   \n" // 000000002900: D1000010 00990110
"  v_mov_b32     v12, v20                                 \n" // 000000002908: 7E180314
"  v_mul_f32     v21, s5, v20                             \n" // 00000000290C: 0A2A2805
"  v_mul_f32     v16, s5, v16                             \n" // 000000002910: 0A202005
"  v_exp_f32     v16, v16                                 \n" // 000000002914: 7E204110
"  v_fma_f32     v32, v32, s5, -v21                       \n" // 000000002918: D1CB0020 84540B20
"  v_fma_f32     v33, v33, s5, -v21                       \n" // 000000002920: D1CB0021 84540B21
"  v_fma_f32     v34, v34, s5, -v21                       \n" // 000000002928: D1CB0022 84540B22
"  v_fma_f32     v35, v35, s5, -v21                       \n" // 000000002930: D1CB0023 84540B23
"  v_exp_f32     v32, v32                                 \n" // 000000002938: 7E404120
"  v_exp_f32     v33, v33                                 \n" // 00000000293C: 7E424121
"  v_exp_f32     v34, v34                                 \n" // 000000002940: 7E444122
"  v_exp_f32     v35, v35                                 \n" // 000000002944: 7E464123
"  v_mul_f32     v14, v16, v14                            \n" // 000000002948: 0A1C1D10
"  v_mov_b32     v22, v32                                 \n" // 00000000294C: 7E2C0320
"  v_add_f32     v22, v33, v22                            \n" // 000000002950: 022C2D21
"  v_add_f32     v22, v34, v22                            \n" // 000000002954: 022C2D22
"  v_add_f32     v22, v35, v22                            \n" // 000000002958: 022C2D23
"  v_add_f32     v14, v22, v14                            \n" // 00000000295C: 021C1D16
"  v_mov_b32     v29, 0xffff0000                          \n" // 000000002960: 7E3A02FF FFFF0000
"  v_mov_b32     v30, 0x7fff0000                          \n" // 000000002968: 7E3C02FF 7FFF0000
"  v_mov_b32     v31, 0x00007fff                          \n" // 000000002970: 7E3E02FF 00007FFF
"  v_cmp_u_f32   s[38:39], v32, v32                       \n" // 000000002978: D0480026 00024120
"  v_add3_u32    v28, v32, v31, 1                         \n" // 000000002980: D1FF001C 02063F20
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000002988: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v33, v33                       \n" // 000000002990: D0480026 00024321
"  v_add3_u32    v28, v33, v31, 1                         \n" // 000000002998: D1FF001C 02063F21
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 0000000029A0: D1000015 009A3D1C
"  v_perm_b32    v32, v21, v20, s52                       \n" // 0000000029A8: D1ED0020 00D22915
"  v_cmp_u_f32   s[38:39], v34, v34                       \n" // 0000000029B0: D0480026 00024522
"  v_add3_u32    v28, v34, v31, 1                         \n" // 0000000029B8: D1FF001C 02063F22
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 0000000029C0: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v35, v35                       \n" // 0000000029C8: D0480026 00024723
"  v_add3_u32    v28, v35, v31, 1                         \n" // 0000000029D0: D1FF001C 02063F23
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 0000000029D8: D1000015 009A3D1C
"  v_perm_b32    v33, v21, v20, s52                       \n" // 0000000029E0: D1ED0021 00D22915
"  s_nop         0x0002                                   \n" // 0000000029E8: BF800002
"  s_add_u32     s83, s84, s83                            \n" // 0000000029EC: 80535354
"  s_nop         0x0000                                   \n" // 0000000029F0: BF800000
"  v_mul_u32_u24  v18, v11, v9 row_newbcast:0             \n" // 0000000029F4: 102412FA FF01500B
"  v_mul_u32_u24  v19, v11, v9 row_newbcast:8             \n" // 0000000029FC: 102612FA FF01580B
"  v_add_u32     v18, v18, v1                             \n" // 000000002A04: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 000000002A08: 68260313
"  v_mul_u32_u24  v182, v11, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v11, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s36                                  \n" // 000000002A0C: BEFC0024
"  v_mov_b32     v22, v16                                 \n" // 000000002A10: 7E2C0310
"  v_mov_b32     v23, v16                                 \n" // 000000002A14: 7E2E0310
"  v_pk_mul_f32  [%[o_regs_0], %[o_regs_1]], v[22:23], [%[o_regs_0], %[o_regs_1]]             \n" // 000000002A18: D3B14028 18025116
"  v_pk_mul_f32  [%[o_regs_2], %[o_regs_3]], v[22:23], [%[o_regs_2], %[o_regs_3]]             \n" // 000000002A20: D3B1402A 18025516
"  v_pk_mul_f32  [%[o_regs_4], %[o_regs_5]], v[22:23], [%[o_regs_4], %[o_regs_5]]             \n" // 000000002A28: D3B1402C 18025916
"  v_pk_mul_f32  [%[o_regs_6], %[o_regs_7]], v[22:23], [%[o_regs_6], %[o_regs_7]]             \n" // 000000002A30: D3B1402E 18025D16
"  v_pk_mul_f32  [%[o_regs_8], %[o_regs_9]], v[22:23], [%[o_regs_8], %[o_regs_9]]             \n" // 000000002A38: D3B14030 18026116
"  v_pk_mul_f32  [%[o_regs_10], %[o_regs_11]], v[22:23], [%[o_regs_10], %[o_regs_11]]             \n" // 000000002A40: D3B14032 18026516
"  v_pk_mul_f32  [%[o_regs_12], %[o_regs_13]], v[22:23], [%[o_regs_12], %[o_regs_13]]             \n" // 000000002A48: D3B14034 18026916
"  v_pk_mul_f32  [%[o_regs_14], %[o_regs_15]], v[22:23], [%[o_regs_14], %[o_regs_15]]             \n" // 000000002A50: D3B14036 18026D16
"  v_pk_mul_f32  [%[o_regs_16], %[o_regs_17]], v[22:23], [%[o_regs_16], %[o_regs_17]]             \n" // 000000002A58: D3B14038 18027116
"  v_pk_mul_f32  [%[o_regs_18], %[o_regs_19]], v[22:23], [%[o_regs_18], %[o_regs_19]]             \n" // 000000002A60: D3B1403A 18027516
"  v_pk_mul_f32  [%[o_regs_20], %[o_regs_21]], v[22:23], [%[o_regs_20], %[o_regs_21]]             \n" // 000000002A68: D3B1403C 18027916
"  v_pk_mul_f32  [%[o_regs_22], %[o_regs_23]], v[22:23], [%[o_regs_22], %[o_regs_23]]             \n" // 000000002A70: D3B1403E 18027D16
"  v_pk_mul_f32  [%[o_regs_24], %[o_regs_25]], v[22:23], [%[o_regs_24], %[o_regs_25]]             \n" // 000000002A78: D3B14040 18028116
"  v_pk_mul_f32  [%[o_regs_26], %[o_regs_27]], v[22:23], [%[o_regs_26], %[o_regs_27]]             \n" // 000000002A80: D3B14042 18028516
"  v_pk_mul_f32  [%[o_regs_28], %[o_regs_29]], v[22:23], [%[o_regs_28], %[o_regs_29]]             \n" // 000000002A88: D3B14044 18028916
"  v_pk_mul_f32  [%[o_regs_30], %[o_regs_31]], v[22:23], [%[o_regs_30], %[o_regs_31]]             \n" // 000000002A90: D3B14046 18028D16
"  v_pk_mul_f32  [%[o_regs_32], %[o_regs_33]], v[22:23], [%[o_regs_32], %[o_regs_33]]             \n" // 000000002A98: D3B14048 18029116
"  v_pk_mul_f32  [%[o_regs_34], %[o_regs_35]], v[22:23], [%[o_regs_34], %[o_regs_35]]             \n" // 000000002AA0: D3B1404A 18029516
"  v_pk_mul_f32  [%[o_regs_36], %[o_regs_37]], v[22:23], [%[o_regs_36], %[o_regs_37]]             \n" // 000000002AA8: D3B1404C 18029916
"  v_pk_mul_f32  [%[o_regs_38], %[o_regs_39]], v[22:23], [%[o_regs_38], %[o_regs_39]]             \n" // 000000002AB0: D3B1404E 18029D16
"  v_pk_mul_f32  [%[o_regs_40], %[o_regs_41]], v[22:23], [%[o_regs_40], %[o_regs_41]]             \n" // 000000002AB8: D3B14050 1802A116
"  v_pk_mul_f32  [%[o_regs_42], %[o_regs_43]], v[22:23], [%[o_regs_42], %[o_regs_43]]             \n" // 000000002AC0: D3B14052 1802A516
"  v_pk_mul_f32  [%[o_regs_44], %[o_regs_45]], v[22:23], [%[o_regs_44], %[o_regs_45]]             \n" // 000000002AC8: D3B14054 1802A916
"  v_pk_mul_f32  [%[o_regs_46], %[o_regs_47]], v[22:23], [%[o_regs_46], %[o_regs_47]]             \n" // 000000002AD0: D3B14056 1802AD16
"  v_pk_mul_f32  [%[o_regs_48], %[o_regs_49]], v[22:23], [%[o_regs_48], %[o_regs_49]]             \n" // 000000002AD8: D3B14058 1802B116
"  v_pk_mul_f32  [%[o_regs_50], %[o_regs_51]], v[22:23], [%[o_regs_50], %[o_regs_51]]             \n" // 000000002AE0: D3B1405A 1802B516
"  v_pk_mul_f32  [%[o_regs_52], %[o_regs_53]], v[22:23], [%[o_regs_52], %[o_regs_53]]             \n" // 000000002AE8: D3B1405C 1802B916
"  v_pk_mul_f32  [%[o_regs_54], %[o_regs_55]], v[22:23], [%[o_regs_54], %[o_regs_55]]             \n" // 000000002AF0: D3B1405E 1802BD16
"  v_pk_mul_f32  [%[o_regs_56], %[o_regs_57]], v[22:23], [%[o_regs_56], %[o_regs_57]]             \n" // 000000002AF8: D3B14060 1802C116
"  v_pk_mul_f32  [%[o_regs_58], %[o_regs_59]], v[22:23], [%[o_regs_58], %[o_regs_59]]             \n" // 000000002B00: D3B14062 1802C516
"  v_pk_mul_f32  [%[o_regs_60], %[o_regs_61]], v[22:23], [%[o_regs_60], %[o_regs_61]]         \n" // 000000002B08: D3B14064 1802C916
"  v_pk_mul_f32  [%[o_regs_62], %[o_regs_63]], v[22:23], [%[o_regs_62], %[o_regs_63]]         \n" // 000000002B10: D3B14066 1802CD16
"  v_pk_mul_f32  [%[o_regs_64], %[o_regs_65]], v[22:23], [%[o_regs_64], %[o_regs_65]]         \n" // 000000002B18: D3B14068 1802D116
"  v_pk_mul_f32  [%[o_regs_66], %[o_regs_67]], v[22:23], [%[o_regs_66], %[o_regs_67]]         \n" // 000000002B20: D3B1406A 1802D516
"  v_pk_mul_f32  [%[o_regs_68], %[o_regs_69]], v[22:23], [%[o_regs_68], %[o_regs_69]]         \n" // 000000002B28: D3B1406C 1802D916
"  v_pk_mul_f32  [%[o_regs_70], %[o_regs_71]], v[22:23], [%[o_regs_70], %[o_regs_71]]         \n" // 000000002B30: D3B1406E 1802DD16
"  v_pk_mul_f32  [%[o_regs_72], %[o_regs_73]], v[22:23], [%[o_regs_72], %[o_regs_73]]         \n" // 000000002B38: D3B14070 1802E116
"  v_pk_mul_f32  [%[o_regs_74], %[o_regs_75]], v[22:23], [%[o_regs_74], %[o_regs_75]]         \n" // 000000002B40: D3B14072 1802E516
"  v_pk_mul_f32  [%[o_regs_76], %[o_regs_77]], v[22:23], [%[o_regs_76], %[o_regs_77]]         \n" // 000000002B48: D3B14074 1802E916
"  v_pk_mul_f32  [%[o_regs_78], %[o_regs_79]], v[22:23], [%[o_regs_78], %[o_regs_79]]         \n" // 000000002B50: D3B14076 1802ED16
"  v_pk_mul_f32  [%[o_regs_80], %[o_regs_81]], v[22:23], [%[o_regs_80], %[o_regs_81]]         \n" // 000000002B58: D3B14078 1802F116
"  v_pk_mul_f32  [%[o_regs_82], %[o_regs_83]], v[22:23], [%[o_regs_82], %[o_regs_83]]         \n" // 000000002B60: D3B1407A 1802F516
"  v_pk_mul_f32  [%[o_regs_84], %[o_regs_85]], v[22:23], [%[o_regs_84], %[o_regs_85]]         \n" // 000000002B68: D3B1407C 1802F916
"  v_pk_mul_f32  [%[o_regs_86], %[o_regs_87]], v[22:23], [%[o_regs_86], %[o_regs_87]]         \n" // 000000002B70: D3B1407E 1802FD16
"  v_pk_mul_f32  [%[o_regs_88], %[o_regs_89]], v[22:23], [%[o_regs_88], %[o_regs_89]]         \n" // 000000002B78: D3B14080 18030116
"  v_pk_mul_f32  [%[o_regs_90], %[o_regs_91]], v[22:23], [%[o_regs_90], %[o_regs_91]]         \n" // 000000002B80: D3B14082 18030516
"  v_pk_mul_f32  [%[o_regs_92], %[o_regs_93]], v[22:23], [%[o_regs_92], %[o_regs_93]]         \n" // 000000002B88: D3B14084 18030916
"  v_pk_mul_f32  [%[o_regs_94], %[o_regs_95]], v[22:23], [%[o_regs_94], %[o_regs_95]]         \n" // 000000002B90: D3B14086 18030D16
"  v_pk_mul_f32  [%[o_regs_96], %[o_regs_97]], v[22:23], [%[o_regs_96], %[o_regs_97]]         \n" // 000000002B98: D3B14088 18031116
"  v_pk_mul_f32  [%[o_regs_98], %[o_regs_99]], v[22:23], [%[o_regs_98], %[o_regs_99]]         \n" // 000000002BA0: D3B1408A 18031516
"  v_pk_mul_f32  [%[o_regs_100], %[o_regs_101]], v[22:23], [%[o_regs_100], %[o_regs_101]]         \n" // 000000002BA8: D3B1408C 18031916
"  v_pk_mul_f32  [%[o_regs_102], %[o_regs_103]], v[22:23], [%[o_regs_102], %[o_regs_103]]         \n" // 000000002BB0: D3B1408E 18031D16
"  v_pk_mul_f32  [%[o_regs_104], %[o_regs_105]], v[22:23], [%[o_regs_104], %[o_regs_105]]         \n" // 000000002BB8: D3B14090 18032116
"  v_pk_mul_f32  [%[o_regs_106], %[o_regs_107]], v[22:23], [%[o_regs_106], %[o_regs_107]]         \n" // 000000002BC0: D3B14092 18032516
"  v_pk_mul_f32  [%[o_regs_108], %[o_regs_109]], v[22:23], [%[o_regs_108], %[o_regs_109]]         \n" // 000000002BC8: D3B14094 18032916
"  v_pk_mul_f32  [%[o_regs_110], %[o_regs_111]], v[22:23], [%[o_regs_110], %[o_regs_111]]         \n" // 000000002BD0: D3B14096 18032D16
"  v_pk_mul_f32  [%[o_regs_112], %[o_regs_113]], v[22:23], [%[o_regs_112], %[o_regs_113]]         \n" // 000000002BD8: D3B14098 18033116
"  v_pk_mul_f32  [%[o_regs_114], %[o_regs_115]], v[22:23], [%[o_regs_114], %[o_regs_115]]         \n" // 000000002BE0: D3B1409A 18033516
"  v_pk_mul_f32  [%[o_regs_116], %[o_regs_117]], v[22:23], [%[o_regs_116], %[o_regs_117]]         \n" // 000000002BE8: D3B1409C 18033916
"  v_pk_mul_f32  [%[o_regs_118], %[o_regs_119]], v[22:23], [%[o_regs_118], %[o_regs_119]]         \n" // 000000002BF0: D3B1409E 18033D16
"  v_pk_mul_f32  [%[o_regs_120], %[o_regs_121]], v[22:23], [%[o_regs_120], %[o_regs_121]]         \n" // 000000002BF8: D3B140A0 18034116
"  v_pk_mul_f32  [%[o_regs_122], %[o_regs_123]], v[22:23], [%[o_regs_122], %[o_regs_123]]         \n" // 000000002C00: D3B140A2 18034516
"  v_pk_mul_f32  [%[o_regs_124], %[o_regs_125]], v[22:23], [%[o_regs_124], %[o_regs_125]]         \n" // 000000002C08: D3B140A4 18034916
"  v_pk_mul_f32  [%[o_regs_126], %[o_regs_127]], v[22:23], [%[o_regs_126], %[o_regs_127]]         \n" // 000000002C10: D3B140A6 18034D16
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000002C18: BF8CC07F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]], acc[144:145], v[32:33], [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]]  \n" // 000000002C1C: D3E10028 0CA24190
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]], acc[146:147], v[32:33], [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]]  \n" // 000000002C24: D3E1002C 0CB24192
"  ds_read_b128  acc[176:179], v7 offset:45312            \n" // 000000002C2C: DBFEB100 B0000007
"  ds_read_b128  acc[180:183], v7 offset:46336            \n" // 000000002C34: DBFEB500 B4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]], acc[148:149], v[32:33], [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]]  \n" // 000000002C3C: D3E10030 0CC24194
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]], acc[150:151], v[32:33], [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]]  \n" // 000000002C44: D3E10034 0CD24196
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]], acc[152:153], v[32:33], [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]]  \n" // 000000002C4C: D3E10038 0CE24198
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]], acc[154:155], v[32:33], [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]]  \n" // 000000002C54: D3E1003C 0CF2419A
"  ds_read_b128  acc[184:187], v7 offset:47360            \n" // 000000002C5C: DBFEB900 B8000007
"  ds_read_b128  acc[188:191], v7 offset:48384            \n" // 000000002C64: DBFEBD00 BC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]], acc[156:157], v[32:33], [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]]  \n" // 000000002C6C: D3E10040 0D02419C
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]], acc[158:159], v[32:33], [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]]  \n" // 000000002C74: D3E10044 0D12419E
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]], acc[160:161], v[32:33], [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]]  \n" // 000000002C7C: D3E10048 0D2241A0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]], acc[162:163], v[32:33], [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]]  \n" // 000000002C84: D3E1004C 0D3241A2
"  ds_read_b128  acc[192:195], v7 offset:49408            \n" // 000000002C8C: DBFEC100 C0000007
"  ds_read_b128  acc[196:199], v7 offset:50432            \n" // 000000002C94: DBFEC500 C4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]], acc[164:165], v[32:33], [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]]  \n" // 000000002C9C: D3E10050 0D4241A4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]], acc[166:167], v[32:33], [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]]  \n" // 000000002CA4: D3E10054 0D5241A6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]], acc[168:169], v[32:33], [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]]  \n" // 000000002CAC: D3E10058 0D6241A8
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]], acc[170:171], v[32:33], [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]]  \n" // 000000002CB4: D3E1005C 0D7241AA
"  ds_read_b128  acc[200:203], v7 offset:51456            \n" // 000000002CBC: DBFEC900 C8000007
"  ds_read_b128  acc[204:207], v7 offset:52480            \n" // 000000002CC4: DBFECD00 CC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]], acc[172:173], v[32:33], [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]]  \n" // 000000002CCC: D3E10060 0D8241AC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]], acc[174:175], v[32:33], [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]]  \n" // 000000002CD4: D3E10064 0D9241AE
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000002CDC: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]], acc[176:177], v[32:33], [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]]  \n" // 000000002CE0: D3E10068 0DA241B0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]], acc[178:179], v[32:33], [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]]  \n" // 000000002CE8: D3E1006C 0DB241B2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]], acc[180:181], v[32:33], [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]]  \n" // 000000002CF0: D3E10070 0DC241B4
"  s_waitcnt     vmcnt(18) & lgkmcnt(0)                   \n" // 000000002CF8: BF8C4072
"  s_barrier                                              \n" // 000000002CFC: BF8A0000
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]], acc[182:183], v[32:33], [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]]  \n" // 000000002D00: D3E10074 0DD241B6
"  ds_read_b64   v[20:21], v5 offset:18560                \n" // 000000002D08: D8EC4880 14000005
"  ds_read_b64   v[22:23], v5 offset:23200                \n" // 000000002D10: D8EC5AA0 16000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]], acc[184:185], v[32:33], [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]]  \n" // 000000002D18: D3E10078 0DE241B8
"  ds_read_b64   v[24:25], v5 offset:27840                \n" // 000000002D20: D8EC6CC0 18000005
"  ds_read_b64   v[26:27], v5 offset:32480                \n" // 000000002D28: D8EC7EE0 1A000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]], acc[186:187], v[32:33], [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]]  \n" // 000000002D30: D3E1007C 0DF241BA
"  ds_read_b128  acc[144:147], v4 offset:18560            \n" // 000000002D38: DBFE4880 90000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]], acc[188:189], v[32:33], [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]]  \n" // 000000002D40: D3E10080 0E0241BC
"  ds_read_b128  acc[148:151], v4 offset:18624            \n" // 000000002D48: DBFE48C0 94000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]], acc[190:191], v[32:33], [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]]  \n" // 000000002D50: D3E10084 0E1241BE
"  ds_read_b128  acc[152:155], v4 offset:18816            \n" // 000000002D58: DBFE4980 98000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]], acc[192:193], v[32:33], [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]]  \n" // 000000002D60: D3E10088 0E2241C0
"  ds_read_b128  acc[156:159], v4 offset:18880            \n" // 000000002D68: DBFE49C0 9C000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]], acc[194:195], v[32:33], [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]]  \n" // 000000002D70: D3E1008C 0E3241C2
"  ds_read_b128  acc[160:163], v4 offset:19072            \n" // 000000002D78: DBFE4A80 A0000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]], acc[196:197], v[32:33], [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]]  \n" // 000000002D80: D3E10090 0E4241C4
"  ds_read_b128  acc[164:167], v4 offset:19136            \n" // 000000002D88: DBFE4AC0 A4000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]], acc[198:199], v[32:33], [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]]  \n" // 000000002D90: D3E10094 0E5241C6
"  ds_read_b128  acc[168:171], v4 offset:19328            \n" // 000000002D98: DBFE4B80 A8000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]], acc[200:201], v[32:33], [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]]  \n" // 000000002DA0: D3E10098 0E6241C8
"  ds_read_b128  acc[172:175], v4 offset:19392            \n" // 000000002DA8: DBFE4BC0 AC000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]], acc[202:203], v[32:33], [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]]  \n" // 000000002DB0: D3E1009C 0E7241CA
"  s_waitcnt     lgkmcnt(8)                               \n" // 000000002DB8: BF8CC87F
"  v_perm_b32    v168, v22, v20, s53                      \n" // 000000002DBC: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000002DC4: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 000000002DCC: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000002DD4: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]], acc[204:205], v[32:33], [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]]  \n" // 000000002DDC: D3E100A0 0E8241CC
"  ds_write_b128  v6, v[168:171] offset:37120             \n" // 000000002DE4: D9BE9100 0000A806
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000002DEC: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000002DF4: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000002DFC: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000002E04: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]], acc[206:207], v[32:33], [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]]  \n" // 000000002E0C: D3E100A4 0E9241CE
"  ds_write_b128  v6, v[168:171] offset:38144             \n" // 000000002E14: D9BE9500 0000A806
"  ds_read_b64   v[20:21], v5 offset:19584                \n" // 000000002E1C: D8EC4C80 14000005
"  ds_read_b64   v[22:23], v5 offset:24224                \n" // 000000002E24: D8EC5EA0 16000005
"  ds_read_b64   v[24:25], v5 offset:28864                \n" // 000000002E2C: D8EC70C0 18000005
"  ds_read_b64   v[26:27], v5 offset:33504                \n" // 000000002E34: D8EC82E0 1A000005
"  s_nop         0x0000                                   \n" // 000000002E3C: BF800000
"  s_addk_i32    s70, 0x0001                              \n" // 000000002E40: B7460001
"  s_cmp_lt_i32  s70, s71                                 \n" // 000000002E44: BF044746
"  s_cbranch_scc0  label_092C_%=                             \n" // 000000002E48: BF84FD99
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000002E4C: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[144:145], acc[0:1], 0  \n" // 000000002E50: D3E10020 1A020190
"  buffer_load_dword  v11, v8, [%[kv_indices_res_0], %[kv_indices_res_1], %[kv_indices_res_2], %[kv_indices_res_3]], 0 offen          \n" // 000000002E58: E0501000 80060B08
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[146:147], acc[2:3], v[32:35]  \n" // 000000002E60: D3E10020 1C820592
"  ds_read_b128  acc[176:179], v4 offset:19584            \n" // 000000002E68: DBFE4C80 B0000004
"  ds_read_b128  acc[180:183], v4 offset:19648            \n" // 000000002E70: DBFE4CC0 B4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[148:149], acc[4:5], v[32:35]  \n" // 000000002E78: D3E10020 1C820994
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[150:151], acc[6:7], v[32:35]  \n" // 000000002E80: D3E10020 1C820D96
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[152:153], acc[8:9], v[32:35]  \n" // 000000002E88: D3E10020 1C821198
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[154:155], acc[10:11], v[32:35]  \n" // 000000002E90: D3E10020 1C82159A
"  ds_read_b128  acc[184:187], v4 offset:19840            \n" // 000000002E98: DBFE4D80 B8000004
"  ds_read_b128  acc[188:191], v4 offset:19904            \n" // 000000002EA0: DBFE4DC0 BC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[156:157], acc[12:13], v[32:35]  \n" // 000000002EA8: D3E10020 1C82199C
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[158:159], acc[14:15], v[32:35]  \n" // 000000002EB0: D3E10020 1C821D9E
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000002EB8: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[160:161], acc[16:17], v[32:35]  \n" // 000000002EBC: D3E10020 1C8221A0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[162:163], acc[18:19], v[32:35]  \n" // 000000002EC4: D3E10020 1C8225A2
"  ds_read_b128  acc[192:195], v4 offset:20096            \n" // 000000002ECC: DBFE4E80 C0000004
"  ds_read_b128  acc[196:199], v4 offset:20160            \n" // 000000002ED4: DBFE4EC0 C4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[164:165], acc[20:21], v[32:35]  \n" // 000000002EDC: D3E10020 1C8229A4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[166:167], acc[22:23], v[32:35]  \n" // 000000002EE4: D3E10020 1C822DA6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[168:169], acc[24:25], v[32:35]  \n" // 000000002EEC: D3E10020 1C8231A8
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[170:171], acc[26:27], v[32:35]  \n" // 000000002EF4: D3E10020 1C8235AA
"  ds_read_b128  acc[200:203], v4 offset:20352            \n" // 000000002EFC: DBFE4F80 C8000004
"  ds_read_b128  acc[204:207], v4 offset:20416            \n" // 000000002F04: DBFE4FC0 CC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[172:173], acc[28:29], v[32:35]  \n" // 000000002F0C: D3E10020 1C8239AC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[174:175], acc[30:31], v[32:35]  \n" // 000000002F14: D3E10020 1C823DAE
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000002F1C: BF8CC47F
"  s_barrier                                              \n" // 000000002F20: BF8A0000
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[176:177], acc[32:33], v[32:35]  \n" // 000000002F24: D3E10020 1C8241B0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[178:179], acc[34:35], v[32:35]  \n" // 000000002F2C: D3E10020 1C8245B2
"  ds_read_b128  acc[208:211], v4 offset:20608            \n" // 000000002F34: DBFE5080 D0000004
"  ds_read_b128  acc[212:215], v4 offset:20672            \n" // 000000002F3C: DBFE50C0 D4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[180:181], acc[36:37], v[32:35]  \n" // 000000002F44: D3E10020 1C8249B4
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000002F4C: E0511000 80050012
"  s_add_u32     m0, 0, s37                               \n" // 000000002F54: 807C2580
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[182:183], acc[38:39], v[32:35]  \n" // 000000002F58: D3E10020 1C824DB6
"  v_perm_b32    v168, v22, v20, s53                      \n" // 000000002F60: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000002F68: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 000000002F70: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000002F78: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[184:185], acc[40:41], v[32:35]  \n" // 000000002F80: D3E10020 1C8251B8
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen lds      \n" // 000000002F88: E0511000 80050013
"  s_add_u32     m0, 0x00000080, s36                      \n" // 000000002F90: 807C24FF 00000080
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[186:187], acc[42:43], v[32:35]  \n" // 000000002F98: D3E10020 1C8255BA
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 000000002FA0: D9BEB100 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[188:189], acc[44:45], v[32:35]  \n" // 000000002FA8: D3E10020 1C8259BC
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000002FB0: E0511080 80050012
"  s_add_u32     m0, 0x00000080, s37                      \n" // 000000002FB8: 807C25FF 00000080
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[190:191], acc[46:47], v[32:35]  \n" // 000000002FC0: D3E10020 1C825DBE
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000002FC8: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000002FD0: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000002FD8: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000002FE0: D1ED00AB 00D2331B
"  s_waitcnt     lgkmcnt(1)                               \n" // 000000002FE8: BF8CC17F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[192:193], acc[48:49], v[32:35]  \n" // 000000002FEC: D3E10020 1C8261C0
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:128 lds  \n" // 000000002FF4: E0511080 80050013
"  s_add_u32     m0, 0x00000100, s36                      \n" // 000000002FFC: 807C24FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[194:195], acc[50:51], v[32:35]  \n" // 000000003004: D3E10020 1C8265C2
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 00000000300C: D9BEB500 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[196:197], acc[52:53], v[32:35]  \n" // 000000003014: D3E10020 1C8269C4
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 00000000301C: E0511100 80050012
"  s_add_u32     m0, 0x00000100, s37                      \n" // 000000003024: 807C25FF 00000100
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[198:199], acc[54:55], v[32:35]  \n" // 00000000302C: D3E10020 1C826DC6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[200:201], acc[56:57], v[32:35]  \n" // 000000003034: D3E10020 1C8271C8
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:256 lds  \n" // 00000000303C: E0511100 80050013
"  s_add_u32     m0, 0x00000180, s36                      \n" // 000000003044: 807C24FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[202:203], acc[58:59], v[32:35]  \n" // 00000000304C: D3E10020 1C8275CA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[204:205], acc[60:61], v[32:35]  \n" // 000000003054: D3E10020 1C8279CC
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 00000000305C: E0511180 80050012
"  s_add_u32     m0, 0x00000180, s37                      \n" // 000000003064: 807C25FF 00000180
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[206:207], acc[62:63], v[32:35]  \n" // 00000000306C: D3E10020 1C827DCE
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[208:209], acc[64:65], v[32:35]  \n" // 000000003074: D3E10020 1C8281D0
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:384 lds  \n" // 00000000307C: E0511180 80050013
"  s_add_u32     m0, 0x00000200, s36                      \n" // 000000003084: 807C24FF 00000200
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[210:211], acc[66:67], v[32:35]  \n" // 00000000308C: D3E10020 1C8285D2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[212:213], acc[68:69], v[32:35]  \n" // 000000003094: D3E10020 1C8289D4
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 00000000309C: E0511200 80050012
"  s_add_u32     m0, 0x00000200, s37                      \n" // 0000000030A4: 807C25FF 00000200
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[214:215], acc[70:71], v[32:35]  \n" // 0000000030AC: D3E10020 1C828DD6
"  v_add_u32     v8, s73, v8                              \n" // 0000000030B4: 68101049
"  s_cmp_le_i32  s83, s82                                 \n" // 0000000030B8: BF055253
"  s_cbranch_scc1  label_0C54_%=                             \n" // 0000000030BC: BF850024
"  v_mov_b32     v25, 0xff800000                          \n" // 0000000030C0: 7E3202FF FF800000
"  s_add_u32     s57, s82, 0                              \n" // 0000000030C8: 80398052
"  v_mov_b32     v24, s57                                 \n" // 0000000030CC: 7E300239
"  v_add_u32     v24, %[wave_id], v24                             \n" // 0000000030D0: 68303007
"  s_sub_u32     s56, s83, 15                             \n" // 0000000030D4: 80B88F53
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 0000000030D8: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 0000000030DC: 0C282884
"  v_add_u32     v20, s56, v20                            \n" // 0000000030E0: 68282838
"  v_add_u32     v21, 1, v20                              \n" // 0000000030E4: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 0000000030E8: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 0000000030EC: 682E2883
"  v_cmp_le_u32  s[38:39], v20, v24                       \n" // 0000000030F0: D0CB0026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 0000000030F8: 682828C0
"  s_nop         0x0000                                   \n" // 0000000030FC: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000003100: D1000020 009A4119
"  v_cmp_le_u32  s[38:39], v21, v24                       \n" // 000000003108: D0CB0026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000003110: 682A2AC0
"  s_nop         0x0000                                   \n" // 000000003114: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000003118: D1000021 009A4319
"  v_cmp_le_u32  s[38:39], v22, v24                       \n" // 000000003120: D0CB0026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000003128: 682C2CC0
"  s_nop         0x0000                                   \n" // 00000000312C: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 000000003130: D1000022 009A4519
"  v_cmp_le_u32  s[38:39], v23, v24                       \n" // 000000003138: D0CB0026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 000000003140: 682E2EC0
"  s_nop         0x0000                                   \n" // 000000003144: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 000000003148: D1000023 009A4719
"label_0C54_%=: \n"
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000003150: BF8CC07F
"  s_barrier                                              \n" // 000000003154: BF8A0000
"  v_max3_f32    v24, v32, v33, v32                       \n" // 000000003158: D1D30018 04824320
"  v_max3_f32    v24, v34, v35, v24                       \n" // 000000003160: D1D30018 04624722
"  ds_write_b32  v3, v24 offset:53504                     \n" // 000000003168: D81AD100 00001803
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:512 lds  \n" // 000000003170: E0511200 80050013
"  s_add_u32     m0, 0x00000280, s36                      \n" // 000000003178: 807C24FF 00000280
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 000000003180: E0511280 80050012
"  s_add_u32     m0, 0x00000280, s37                      \n" // 000000003188: 807C25FF 00000280
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000003190: BF8CC07F
"  ds_read_b32   v20, v2 offset:53504                     \n" // 000000003194: D86CD100 14000002
"  ds_read_b32   v21, v2 offset:53568                     \n" // 00000000319C: D86CD140 15000002
"  ds_read_b32   v22, v2 offset:53632                     \n" // 0000000031A4: D86CD180 16000002
"  ds_read_b32   v23, v2 offset:53696                     \n" // 0000000031AC: D86CD1C0 17000002
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:640 lds  \n" // 0000000031B4: E0511280 80050013
"  s_add_u32     m0, 0x00000300, s36                      \n" // 0000000031BC: 807C24FF 00000300
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 0000000031C4: E0511300 80050012
"  s_add_u32     m0, 0x00000300, s37                      \n" // 0000000031CC: 807C25FF 00000300
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:768 lds  \n" // 0000000031D4: E0511300 80050013
"  s_add_u32     m0, 0x00000380, s36                      \n" // 0000000031DC: 807C24FF 00000380
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000031E4: BF8CC07F
"  v_max3_f32    v24, v20, v21, v24                       \n" // 0000000031E8: D1D30018 04622B14
"  v_max3_f32    v24, v22, v23, v24                       \n" // 0000000031F0: D1D30018 04622F16
"  buffer_load_dword   v18, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 0000000031F8: E0511380 80050012
"  s_add_u32     m0, 0x00000380, s37                      \n" // 000000003200: 807C25FF 00000380
"  buffer_load_dword   v19, [%[kv_res_0], %[kv_res_1], %[kv_res_2], %[kv_res_3]], 0 offen offset:896 lds  \n" // 000000003208: E0511380 80050013
"  s_add_u32     m0, 0x00000400, s36                      \n" // 000000003210: 807C24FF 00000400
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v182, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000400, s37                      \n" // 000000003220: 807C25FF 00000400
"  ds_read_b128  acc[144:147], v7 offset:37120            \n" // 000000003228: DBFE9100 90000007
"  ds_read_b128  acc[148:151], v7 offset:38144            \n" // 000000003230: DBFE9500 94000007
"  s_add_u32     m0, m0, 1024 \n" // 000000000BE0: 807C23FF 00000400
"  buffer_load_dword   v183, [%[kv_rope_res_0], %[kv_rope_res_1], %[kv_rope_res_2], %[kv_rope_res_3]], 0 offen lds  \n" // 000000000BD8: E0511400 80050012
"  s_add_u32     m0, 0x00000480, s36                      \n" // 000000003240: 807C24FF 00000480
"  ds_read_b128  acc[152:155], v7 offset:39168            \n" // 000000003248: DBFE9900 98000007
"  ds_read_b128  acc[156:159], v7 offset:40192            \n" // 000000003250: DBFE9D00 9C000007
"  ds_read_b128  acc[160:163], v7 offset:41216            \n" // 000000003258: DBFEA100 A0000007
"  ds_read_b128  acc[164:167], v7 offset:42240            \n" // 000000003260: DBFEA500 A4000007
"  ds_read_b128  acc[168:171], v7 offset:43264            \n" // 000000003268: DBFEA900 A8000007
"  ds_read_b128  acc[172:175], v7 offset:44288            \n" // 000000003270: DBFEAD00 AC000007
"  v_mov_b32     v25, 0xff7fffff                          \n" // 000000003278: 7E3202FF FF7FFFFF
"  v_cmp_eq_u32  s[38:39], v25, v12                       \n" // 000000003280: D0CA0026 00021919
"  v_max_f32     v20, v24, v12                            \n" // 000000003288: 16281918
"  v_sub_f32     v16, v12, v20                            \n" // 00000000328C: 0420290C
"  v_cndmask_b32  v16, v16, 0, s[38:39]                   \n" // 000000003290: D1000010 00990110
"  v_mov_b32     v12, v20                                 \n" // 000000003298: 7E180314
"  v_mul_f32     v21, s5, v20                             \n" // 00000000329C: 0A2A2805
"  v_mul_f32     v16, s5, v16                             \n" // 0000000032A0: 0A202005
"  v_exp_f32     v16, v16                                 \n" // 0000000032A4: 7E204110
"  v_fma_f32     v32, v32, s5, -v21                       \n" // 0000000032A8: D1CB0020 84540B20
"  v_fma_f32     v33, v33, s5, -v21                       \n" // 0000000032B0: D1CB0021 84540B21
"  v_fma_f32     v34, v34, s5, -v21                       \n" // 0000000032B8: D1CB0022 84540B22
"  v_fma_f32     v35, v35, s5, -v21                       \n" // 0000000032C0: D1CB0023 84540B23
"  v_exp_f32     v32, v32                                 \n" // 0000000032C8: 7E404120
"  v_exp_f32     v33, v33                                 \n" // 0000000032CC: 7E424121
"  v_exp_f32     v34, v34                                 \n" // 0000000032D0: 7E444122
"  v_exp_f32     v35, v35                                 \n" // 0000000032D4: 7E464123
"  v_mul_f32     v14, v16, v14                            \n" // 0000000032D8: 0A1C1D10
"  v_mov_b32     v22, v32                                 \n" // 0000000032DC: 7E2C0320
"  v_add_f32     v22, v33, v22                            \n" // 0000000032E0: 022C2D21
"  v_add_f32     v22, v34, v22                            \n" // 0000000032E4: 022C2D22
"  v_add_f32     v22, v35, v22                            \n" // 0000000032E8: 022C2D23
"  v_add_f32     v14, v22, v14                            \n" // 0000000032EC: 021C1D16
"  v_mov_b32     v29, 0xffff0000                          \n" // 0000000032F0: 7E3A02FF FFFF0000
"  v_mov_b32     v30, 0x7fff0000                          \n" // 0000000032F8: 7E3C02FF 7FFF0000
"  v_mov_b32     v31, 0x00007fff                          \n" // 000000003300: 7E3E02FF 00007FFF
"  v_cmp_u_f32   s[38:39], v32, v32                       \n" // 000000003308: D0480026 00024120
"  v_add3_u32    v28, v32, v31, 1                         \n" // 000000003310: D1FF001C 02063F20
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000003318: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v33, v33                       \n" // 000000003320: D0480026 00024321
"  v_add3_u32    v28, v33, v31, 1                         \n" // 000000003328: D1FF001C 02063F21
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000003330: D1000015 009A3D1C
"  v_perm_b32    v32, v21, v20, s52                       \n" // 000000003338: D1ED0020 00D22915
"  v_cmp_u_f32   s[38:39], v34, v34                       \n" // 000000003340: D0480026 00024522
"  v_add3_u32    v28, v34, v31, 1                         \n" // 000000003348: D1FF001C 02063F22
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000003350: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v35, v35                       \n" // 000000003358: D0480026 00024723
"  v_add3_u32    v28, v35, v31, 1                         \n" // 000000003360: D1FF001C 02063F23
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000003368: D1000015 009A3D1C
"  v_perm_b32    v33, v21, v20, s52                       \n" // 000000003370: D1ED0021 00D22915
"  s_nop         0x0002                                   \n" // 000000003378: BF800002
"  s_add_u32     s83, s84, s83                            \n" // 00000000337C: 80535354
"  s_nop         0x0000                                   \n" // 000000003380: BF800000
"  v_mul_u32_u24  v18, v10, v9 row_newbcast:0             \n" // 000000003384: 102412FA FF01500A
"  v_mul_u32_u24  v19, v10, v9 row_newbcast:8             \n" // 00000000338C: 102612FA FF01580A
"  v_add_u32     v18, v18, v1                             \n" // 000000003394: 68240312
"  v_add_u32     v19, v19, v1                             \n" // 000000003398: 68260313
"  v_mul_u32_u24  v182, v10, v181 row_newbcast:0             \n" // 000000000AC0: 102412FA FF01500A
"  v_mul_u32_u24  v183, v10, v181 row_newbcast:8             \n" // 000000000AC8: 102612FA FF01580A
"  v_add_u32     v182, v182, v1                             \n" // 000000000AD0: 68240312
"  v_add_u32     v183, v183, v1                             \n" // 000000000AD4: 68260313
"  s_mov_b32     m0, s34                                  \n" // 00000000339C: BEFC0022
"  v_mov_b32     v22, v16                                 \n" // 0000000033A0: 7E2C0310
"  v_mov_b32     v23, v16                                 \n" // 0000000033A4: 7E2E0310
"  v_pk_mul_f32  [%[o_regs_0], %[o_regs_1]], v[22:23], [%[o_regs_0], %[o_regs_1]]             \n" // 0000000033A8: D3B14028 18025116
"  v_pk_mul_f32  [%[o_regs_2], %[o_regs_3]], v[22:23], [%[o_regs_2], %[o_regs_3]]             \n" // 0000000033B0: D3B1402A 18025516
"  v_pk_mul_f32  [%[o_regs_4], %[o_regs_5]], v[22:23], [%[o_regs_4], %[o_regs_5]]             \n" // 0000000033B8: D3B1402C 18025916
"  v_pk_mul_f32  [%[o_regs_6], %[o_regs_7]], v[22:23], [%[o_regs_6], %[o_regs_7]]             \n" // 0000000033C0: D3B1402E 18025D16
"  v_pk_mul_f32  [%[o_regs_8], %[o_regs_9]], v[22:23], [%[o_regs_8], %[o_regs_9]]             \n" // 0000000033C8: D3B14030 18026116
"  v_pk_mul_f32  [%[o_regs_10], %[o_regs_11]], v[22:23], [%[o_regs_10], %[o_regs_11]]             \n" // 0000000033D0: D3B14032 18026516
"  v_pk_mul_f32  [%[o_regs_12], %[o_regs_13]], v[22:23], [%[o_regs_12], %[o_regs_13]]             \n" // 0000000033D8: D3B14034 18026916
"  v_pk_mul_f32  [%[o_regs_14], %[o_regs_15]], v[22:23], [%[o_regs_14], %[o_regs_15]]             \n" // 0000000033E0: D3B14036 18026D16
"  v_pk_mul_f32  [%[o_regs_16], %[o_regs_17]], v[22:23], [%[o_regs_16], %[o_regs_17]]             \n" // 0000000033E8: D3B14038 18027116
"  v_pk_mul_f32  [%[o_regs_18], %[o_regs_19]], v[22:23], [%[o_regs_18], %[o_regs_19]]             \n" // 0000000033F0: D3B1403A 18027516
"  v_pk_mul_f32  [%[o_regs_20], %[o_regs_21]], v[22:23], [%[o_regs_20], %[o_regs_21]]             \n" // 0000000033F8: D3B1403C 18027916
"  v_pk_mul_f32  [%[o_regs_22], %[o_regs_23]], v[22:23], [%[o_regs_22], %[o_regs_23]]             \n" // 000000003400: D3B1403E 18027D16
"  v_pk_mul_f32  [%[o_regs_24], %[o_regs_25]], v[22:23], [%[o_regs_24], %[o_regs_25]]             \n" // 000000003408: D3B14040 18028116
"  v_pk_mul_f32  [%[o_regs_26], %[o_regs_27]], v[22:23], [%[o_regs_26], %[o_regs_27]]             \n" // 000000003410: D3B14042 18028516
"  v_pk_mul_f32  [%[o_regs_28], %[o_regs_29]], v[22:23], [%[o_regs_28], %[o_regs_29]]             \n" // 000000003418: D3B14044 18028916
"  v_pk_mul_f32  [%[o_regs_30], %[o_regs_31]], v[22:23], [%[o_regs_30], %[o_regs_31]]             \n" // 000000003420: D3B14046 18028D16
"  v_pk_mul_f32  [%[o_regs_32], %[o_regs_33]], v[22:23], [%[o_regs_32], %[o_regs_33]]             \n" // 000000003428: D3B14048 18029116
"  v_pk_mul_f32  [%[o_regs_34], %[o_regs_35]], v[22:23], [%[o_regs_34], %[o_regs_35]]             \n" // 000000003430: D3B1404A 18029516
"  v_pk_mul_f32  [%[o_regs_36], %[o_regs_37]], v[22:23], [%[o_regs_36], %[o_regs_37]]             \n" // 000000003438: D3B1404C 18029916
"  v_pk_mul_f32  [%[o_regs_38], %[o_regs_39]], v[22:23], [%[o_regs_38], %[o_regs_39]]             \n" // 000000003440: D3B1404E 18029D16
"  v_pk_mul_f32  [%[o_regs_40], %[o_regs_41]], v[22:23], [%[o_regs_40], %[o_regs_41]]             \n" // 000000003448: D3B14050 1802A116
"  v_pk_mul_f32  [%[o_regs_42], %[o_regs_43]], v[22:23], [%[o_regs_42], %[o_regs_43]]             \n" // 000000003450: D3B14052 1802A516
"  v_pk_mul_f32  [%[o_regs_44], %[o_regs_45]], v[22:23], [%[o_regs_44], %[o_regs_45]]             \n" // 000000003458: D3B14054 1802A916
"  v_pk_mul_f32  [%[o_regs_46], %[o_regs_47]], v[22:23], [%[o_regs_46], %[o_regs_47]]             \n" // 000000003460: D3B14056 1802AD16
"  v_pk_mul_f32  [%[o_regs_48], %[o_regs_49]], v[22:23], [%[o_regs_48], %[o_regs_49]]             \n" // 000000003468: D3B14058 1802B116
"  v_pk_mul_f32  [%[o_regs_50], %[o_regs_51]], v[22:23], [%[o_regs_50], %[o_regs_51]]             \n" // 000000003470: D3B1405A 1802B516
"  v_pk_mul_f32  [%[o_regs_52], %[o_regs_53]], v[22:23], [%[o_regs_52], %[o_regs_53]]             \n" // 000000003478: D3B1405C 1802B916
"  v_pk_mul_f32  [%[o_regs_54], %[o_regs_55]], v[22:23], [%[o_regs_54], %[o_regs_55]]             \n" // 000000003480: D3B1405E 1802BD16
"  v_pk_mul_f32  [%[o_regs_56], %[o_regs_57]], v[22:23], [%[o_regs_56], %[o_regs_57]]             \n" // 000000003488: D3B14060 1802C116
"  v_pk_mul_f32  [%[o_regs_58], %[o_regs_59]], v[22:23], [%[o_regs_58], %[o_regs_59]]             \n" // 000000003490: D3B14062 1802C516
"  v_pk_mul_f32  [%[o_regs_60], %[o_regs_61]], v[22:23], [%[o_regs_60], %[o_regs_61]]         \n" // 000000003498: D3B14064 1802C916
"  v_pk_mul_f32  [%[o_regs_62], %[o_regs_63]], v[22:23], [%[o_regs_62], %[o_regs_63]]         \n" // 0000000034A0: D3B14066 1802CD16
"  v_pk_mul_f32  [%[o_regs_64], %[o_regs_65]], v[22:23], [%[o_regs_64], %[o_regs_65]]         \n" // 0000000034A8: D3B14068 1802D116
"  v_pk_mul_f32  [%[o_regs_66], %[o_regs_67]], v[22:23], [%[o_regs_66], %[o_regs_67]]         \n" // 0000000034B0: D3B1406A 1802D516
"  v_pk_mul_f32  [%[o_regs_68], %[o_regs_69]], v[22:23], [%[o_regs_68], %[o_regs_69]]         \n" // 0000000034B8: D3B1406C 1802D916
"  v_pk_mul_f32  [%[o_regs_70], %[o_regs_71]], v[22:23], [%[o_regs_70], %[o_regs_71]]         \n" // 0000000034C0: D3B1406E 1802DD16
"  v_pk_mul_f32  [%[o_regs_72], %[o_regs_73]], v[22:23], [%[o_regs_72], %[o_regs_73]]         \n" // 0000000034C8: D3B14070 1802E116
"  v_pk_mul_f32  [%[o_regs_74], %[o_regs_75]], v[22:23], [%[o_regs_74], %[o_regs_75]]         \n" // 0000000034D0: D3B14072 1802E516
"  v_pk_mul_f32  [%[o_regs_76], %[o_regs_77]], v[22:23], [%[o_regs_76], %[o_regs_77]]         \n" // 0000000034D8: D3B14074 1802E916
"  v_pk_mul_f32  [%[o_regs_78], %[o_regs_79]], v[22:23], [%[o_regs_78], %[o_regs_79]]         \n" // 0000000034E0: D3B14076 1802ED16
"  v_pk_mul_f32  [%[o_regs_80], %[o_regs_81]], v[22:23], [%[o_regs_80], %[o_regs_81]]         \n" // 0000000034E8: D3B14078 1802F116
"  v_pk_mul_f32  [%[o_regs_82], %[o_regs_83]], v[22:23], [%[o_regs_82], %[o_regs_83]]         \n" // 0000000034F0: D3B1407A 1802F516
"  v_pk_mul_f32  [%[o_regs_84], %[o_regs_85]], v[22:23], [%[o_regs_84], %[o_regs_85]]         \n" // 0000000034F8: D3B1407C 1802F916
"  v_pk_mul_f32  [%[o_regs_86], %[o_regs_87]], v[22:23], [%[o_regs_86], %[o_regs_87]]         \n" // 000000003500: D3B1407E 1802FD16
"  v_pk_mul_f32  [%[o_regs_88], %[o_regs_89]], v[22:23], [%[o_regs_88], %[o_regs_89]]         \n" // 000000003508: D3B14080 18030116
"  v_pk_mul_f32  [%[o_regs_90], %[o_regs_91]], v[22:23], [%[o_regs_90], %[o_regs_91]]         \n" // 000000003510: D3B14082 18030516
"  v_pk_mul_f32  [%[o_regs_92], %[o_regs_93]], v[22:23], [%[o_regs_92], %[o_regs_93]]         \n" // 000000003518: D3B14084 18030916
"  v_pk_mul_f32  [%[o_regs_94], %[o_regs_95]], v[22:23], [%[o_regs_94], %[o_regs_95]]         \n" // 000000003520: D3B14086 18030D16
"  v_pk_mul_f32  [%[o_regs_96], %[o_regs_97]], v[22:23], [%[o_regs_96], %[o_regs_97]]         \n" // 000000003528: D3B14088 18031116
"  v_pk_mul_f32  [%[o_regs_98], %[o_regs_99]], v[22:23], [%[o_regs_98], %[o_regs_99]]         \n" // 000000003530: D3B1408A 18031516
"  v_pk_mul_f32  [%[o_regs_100], %[o_regs_101]], v[22:23], [%[o_regs_100], %[o_regs_101]]         \n" // 000000003538: D3B1408C 18031916
"  v_pk_mul_f32  [%[o_regs_102], %[o_regs_103]], v[22:23], [%[o_regs_102], %[o_regs_103]]         \n" // 000000003540: D3B1408E 18031D16
"  v_pk_mul_f32  [%[o_regs_104], %[o_regs_105]], v[22:23], [%[o_regs_104], %[o_regs_105]]         \n" // 000000003548: D3B14090 18032116
"  v_pk_mul_f32  [%[o_regs_106], %[o_regs_107]], v[22:23], [%[o_regs_106], %[o_regs_107]]         \n" // 000000003550: D3B14092 18032516
"  v_pk_mul_f32  [%[o_regs_108], %[o_regs_109]], v[22:23], [%[o_regs_108], %[o_regs_109]]         \n" // 000000003558: D3B14094 18032916
"  v_pk_mul_f32  [%[o_regs_110], %[o_regs_111]], v[22:23], [%[o_regs_110], %[o_regs_111]]         \n" // 000000003560: D3B14096 18032D16
"  v_pk_mul_f32  [%[o_regs_112], %[o_regs_113]], v[22:23], [%[o_regs_112], %[o_regs_113]]         \n" // 000000003568: D3B14098 18033116
"  v_pk_mul_f32  [%[o_regs_114], %[o_regs_115]], v[22:23], [%[o_regs_114], %[o_regs_115]]         \n" // 000000003570: D3B1409A 18033516
"  v_pk_mul_f32  [%[o_regs_116], %[o_regs_117]], v[22:23], [%[o_regs_116], %[o_regs_117]]         \n" // 000000003578: D3B1409C 18033916
"  v_pk_mul_f32  [%[o_regs_118], %[o_regs_119]], v[22:23], [%[o_regs_118], %[o_regs_119]]         \n" // 000000003580: D3B1409E 18033D16
"  v_pk_mul_f32  [%[o_regs_120], %[o_regs_121]], v[22:23], [%[o_regs_120], %[o_regs_121]]         \n" // 000000003588: D3B140A0 18034116
"  v_pk_mul_f32  [%[o_regs_122], %[o_regs_123]], v[22:23], [%[o_regs_122], %[o_regs_123]]         \n" // 000000003590: D3B140A2 18034516
"  v_pk_mul_f32  [%[o_regs_124], %[o_regs_125]], v[22:23], [%[o_regs_124], %[o_regs_125]]         \n" // 000000003598: D3B140A4 18034916
"  v_pk_mul_f32  [%[o_regs_126], %[o_regs_127]], v[22:23], [%[o_regs_126], %[o_regs_127]]         \n" // 0000000035A0: D3B140A6 18034D16
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000035A8: BF8CC07F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]], acc[144:145], v[32:33], [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]]  \n" // 0000000035AC: D3E10028 0CA24190
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]], acc[146:147], v[32:33], [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]]  \n" // 0000000035B4: D3E1002C 0CB24192
"  ds_read_b128  acc[176:179], v7 offset:45312            \n" // 0000000035BC: DBFEB100 B0000007
"  ds_read_b128  acc[180:183], v7 offset:46336            \n" // 0000000035C4: DBFEB500 B4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]], acc[148:149], v[32:33], [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]]  \n" // 0000000035CC: D3E10030 0CC24194
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]], acc[150:151], v[32:33], [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]]  \n" // 0000000035D4: D3E10034 0CD24196
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]], acc[152:153], v[32:33], [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]]  \n" // 0000000035DC: D3E10038 0CE24198
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]], acc[154:155], v[32:33], [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]]  \n" // 0000000035E4: D3E1003C 0CF2419A
"  ds_read_b128  acc[184:187], v7 offset:47360            \n" // 0000000035EC: DBFEB900 B8000007
"  ds_read_b128  acc[188:191], v7 offset:48384            \n" // 0000000035F4: DBFEBD00 BC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]], acc[156:157], v[32:33], [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]]  \n" // 0000000035FC: D3E10040 0D02419C
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]], acc[158:159], v[32:33], [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]]  \n" // 000000003604: D3E10044 0D12419E
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]], acc[160:161], v[32:33], [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]]  \n" // 00000000360C: D3E10048 0D2241A0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]], acc[162:163], v[32:33], [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]]  \n" // 000000003614: D3E1004C 0D3241A2
"  ds_read_b128  acc[192:195], v7 offset:49408            \n" // 00000000361C: DBFEC100 C0000007
"  ds_read_b128  acc[196:199], v7 offset:50432            \n" // 000000003624: DBFEC500 C4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]], acc[164:165], v[32:33], [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]]  \n" // 00000000362C: D3E10050 0D4241A4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]], acc[166:167], v[32:33], [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]]  \n" // 000000003634: D3E10054 0D5241A6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]], acc[168:169], v[32:33], [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]]  \n" // 00000000363C: D3E10058 0D6241A8
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]], acc[170:171], v[32:33], [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]]  \n" // 000000003644: D3E1005C 0D7241AA
"  ds_read_b128  acc[200:203], v7 offset:51456            \n" // 00000000364C: DBFEC900 C8000007
"  ds_read_b128  acc[204:207], v7 offset:52480            \n" // 000000003654: DBFECD00 CC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]], acc[172:173], v[32:33], [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]]  \n" // 00000000365C: D3E10060 0D8241AC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]], acc[174:175], v[32:33], [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]]  \n" // 000000003664: D3E10064 0D9241AE
"  s_waitcnt     lgkmcnt(4)                               \n" // 00000000366C: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]], acc[176:177], v[32:33], [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]]  \n" // 000000003670: D3E10068 0DA241B0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]], acc[178:179], v[32:33], [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]]  \n" // 000000003678: D3E1006C 0DB241B2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]], acc[180:181], v[32:33], [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]]  \n" // 000000003680: D3E10070 0DC241B4
"  s_waitcnt     vmcnt(18) & lgkmcnt(0)                   \n" // 000000003688: BF8C4072
"  s_barrier                                              \n" // 00000000368C: BF8A0000
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]], acc[182:183], v[32:33], [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]]  \n" // 000000003690: D3E10074 0DD241B6
"  ds_read_b64   v[20:21], v5                             \n" // 000000003698: D8EC0000 14000005
"  ds_read_b64   v[22:23], v5 offset:4640                 \n" // 0000000036A0: D8EC1220 16000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]], acc[184:185], v[32:33], [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]]  \n" // 0000000036A8: D3E10078 0DE241B8
"  ds_read_b64   v[24:25], v5 offset:9280                 \n" // 0000000036B0: D8EC2440 18000005
"  ds_read_b64   v[26:27], v5 offset:13920                \n" // 0000000036B8: D8EC3660 1A000005
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]], acc[186:187], v[32:33], [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]]  \n" // 0000000036C0: D3E1007C 0DF241BA
"  ds_read_b128  acc[144:147], v4                         \n" // 0000000036C8: DBFE0000 90000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]], acc[188:189], v[32:33], [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]]  \n" // 0000000036D0: D3E10080 0E0241BC
"  ds_read_b128  acc[148:151], v4 offset:64               \n" // 0000000036D8: DBFE0040 94000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]], acc[190:191], v[32:33], [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]]  \n" // 0000000036E0: D3E10084 0E1241BE
"  ds_read_b128  acc[152:155], v4 offset:256              \n" // 0000000036E8: DBFE0100 98000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]], acc[192:193], v[32:33], [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]]  \n" // 0000000036F0: D3E10088 0E2241C0
"  ds_read_b128  acc[156:159], v4 offset:320              \n" // 0000000036F8: DBFE0140 9C000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]], acc[194:195], v[32:33], [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]]  \n" // 000000003700: D3E1008C 0E3241C2
"  ds_read_b128  acc[160:163], v4 offset:512              \n" // 000000003708: DBFE0200 A0000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]], acc[196:197], v[32:33], [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]]  \n" // 000000003710: D3E10090 0E4241C4
"  ds_read_b128  acc[164:167], v4 offset:576              \n" // 000000003718: DBFE0240 A4000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]], acc[198:199], v[32:33], [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]]  \n" // 000000003720: D3E10094 0E5241C6
"  ds_read_b128  acc[168:171], v4 offset:768              \n" // 000000003728: DBFE0300 A8000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]], acc[200:201], v[32:33], [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]]  \n" // 000000003730: D3E10098 0E6241C8
"  ds_read_b128  acc[172:175], v4 offset:832              \n" // 000000003738: DBFE0340 AC000004
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]], acc[202:203], v[32:33], [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]]  \n" // 000000003740: D3E1009C 0E7241CA
"  s_waitcnt     lgkmcnt(8)                               \n" // 000000003748: BF8CC87F
"  v_perm_b32    v168, v22, v20, s53                      \n" // 00000000374C: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 000000003754: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 00000000375C: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000003764: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]], acc[204:205], v[32:33], [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]]  \n" // 00000000376C: D3E100A0 0E8241CC
"  ds_write_b128  v6, v[168:171] offset:37120             \n" // 000000003774: D9BE9100 0000A806
"  v_perm_b32    v168, v23, v21, s53                      \n" // 00000000377C: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000003784: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 00000000378C: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000003794: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]], acc[206:207], v[32:33], [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]]  \n" // 00000000379C: D3E100A4 0E9241CE
"  ds_write_b128  v6, v[168:171] offset:38144             \n" // 0000000037A4: D9BE9500 0000A806
"  ds_read_b64   v[20:21], v5 offset:1024                 \n" // 0000000037AC: D8EC0400 14000005
"  ds_read_b64   v[22:23], v5 offset:5664                 \n" // 0000000037B4: D8EC1620 16000005
"  ds_read_b64   v[24:25], v5 offset:10304                \n" // 0000000037BC: D8EC2840 18000005
"  ds_read_b64   v[26:27], v5 offset:14944                \n" // 0000000037C4: D8EC3A60 1A000005
"  s_nop         0x0000                                   \n" // 0000000037CC: BF800000
"  s_addk_i32    s70, 0x0001                              \n" // 0000000037D0: B7460001
"  s_cmp_lt_i32  s70, s71                                 \n" // 0000000037D4: BF044746
"  s_cbranch_scc0  label_092C_%=                             \n" // 0000000037D8: BF84FB35
"  s_branch      label_092F_%=                               \n" // 0000000037DC: BF82FB37
"label_0DF8_%=: \n"
"  s_cmp_eq_i32  s48, 0                                   \n" // 0000000037E0: BF008030
"  s_cbranch_scc1  label_11F9%=                             \n" // 0000000037E4: BF8503FF
"label_0DFA_%=: \n"
"  s_and_b32     s56, s71, 1                              \n" // 0000000037E8: 86388147
"  s_cmp_eq_i32  s56, 1                                   \n" // 0000000037EC: BF008138
"  s_cbranch_scc1  label_0FFB_%=                             \n" // 0000000037F0: BF8501FE
"  s_waitcnt     lgkmcnt(4)                               \n" // 0000000037F4: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[144:145], acc[0:1], 0  \n" // 0000000037F8: D3E10020 1A020190
"  ds_read_b128  acc[176:179], v4 offset:1024             \n" // 000000003800: DBFE0400 B0000004
"  ds_read_b128  acc[180:183], v4 offset:1088             \n" // 000000003808: DBFE0440 B4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[146:147], acc[2:3], v[32:35]  \n" // 000000003810: D3E10020 1C820592
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[148:149], acc[4:5], v[32:35]  \n" // 000000003818: D3E10020 1C820994
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[150:151], acc[6:7], v[32:35]  \n" // 000000003820: D3E10020 1C820D96
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[152:153], acc[8:9], v[32:35]  \n" // 000000003828: D3E10020 1C821198
"  ds_read_b128  acc[184:187], v4 offset:1280             \n" // 000000003830: DBFE0500 B8000004
"  ds_read_b128  acc[188:191], v4 offset:1344             \n" // 000000003838: DBFE0540 BC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[154:155], acc[10:11], v[32:35]  \n" // 000000003840: D3E10020 1C82159A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[156:157], acc[12:13], v[32:35]  \n" // 000000003848: D3E10020 1C82199C
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[158:159], acc[14:15], v[32:35]  \n" // 000000003850: D3E10020 1C821D9E
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000003858: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[160:161], acc[16:17], v[32:35]  \n" // 00000000385C: D3E10020 1C8221A0
"  ds_read_b128  acc[192:195], v4 offset:1536             \n" // 000000003864: DBFE0600 C0000004
"  ds_read_b128  acc[196:199], v4 offset:1600             \n" // 00000000386C: DBFE0640 C4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[162:163], acc[18:19], v[32:35]  \n" // 000000003874: D3E10020 1C8225A2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[164:165], acc[20:21], v[32:35]  \n" // 00000000387C: D3E10020 1C8229A4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[166:167], acc[22:23], v[32:35]  \n" // 000000003884: D3E10020 1C822DA6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[168:169], acc[24:25], v[32:35]  \n" // 00000000388C: D3E10020 1C8231A8
"  ds_read_b128  acc[200:203], v4 offset:1792             \n" // 000000003894: DBFE0700 C8000004
"  ds_read_b128  acc[204:207], v4 offset:1856             \n" // 00000000389C: DBFE0740 CC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[170:171], acc[26:27], v[32:35]  \n" // 0000000038A4: D3E10020 1C8235AA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[172:173], acc[28:29], v[32:35]  \n" // 0000000038AC: D3E10020 1C8239AC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[174:175], acc[30:31], v[32:35]  \n" // 0000000038B4: D3E10020 1C823DAE
"  s_waitcnt     lgkmcnt(4)                               \n" // 0000000038BC: BF8CC47F
"  s_barrier                                              \n" // 0000000038C0: BF8A0000
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[176:177], acc[32:33], v[32:35]  \n" // 0000000038C4: D3E10020 1C8241B0
"  ds_read_b128  acc[208:211], v4 offset:2048             \n" // 0000000038CC: DBFE0800 D0000004
"  ds_read_b128  acc[212:215], v4 offset:2112             \n" // 0000000038D4: DBFE0840 D4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[178:179], acc[34:35], v[32:35]  \n" // 0000000038DC: D3E10020 1C8245B2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[180:181], acc[36:37], v[32:35]  \n" // 0000000038E4: D3E10020 1C8249B4
"  v_perm_b32    v168, v22, v20, s53                      \n" // 0000000038EC: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 0000000038F4: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 0000000038FC: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 000000003904: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[182:183], acc[38:39], v[32:35]  \n" // 00000000390C: D3E10020 1C824DB6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[184:185], acc[40:41], v[32:35]  \n" // 000000003914: D3E10020 1C8251B8
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 00000000391C: D9BEB100 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[186:187], acc[42:43], v[32:35]  \n" // 000000003924: D3E10020 1C8255BA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[188:189], acc[44:45], v[32:35]  \n" // 00000000392C: D3E10020 1C8259BC
"  v_perm_b32    v168, v23, v21, s53                      \n" // 000000003934: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 00000000393C: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 000000003944: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 00000000394C: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[190:191], acc[46:47], v[32:35]  \n" // 000000003954: D3E10020 1C825DBE
"  s_waitcnt     lgkmcnt(1)                               \n" // 00000000395C: BF8CC17F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[192:193], acc[48:49], v[32:35]  \n" // 000000003960: D3E10020 1C8261C0
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 000000003968: D9BEB500 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[194:195], acc[50:51], v[32:35]  \n" // 000000003970: D3E10020 1C8265C2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[196:197], acc[52:53], v[32:35]  \n" // 000000003978: D3E10020 1C8269C4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[198:199], acc[54:55], v[32:35]  \n" // 000000003980: D3E10020 1C826DC6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[200:201], acc[56:57], v[32:35]  \n" // 000000003988: D3E10020 1C8271C8
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[202:203], acc[58:59], v[32:35]  \n" // 000000003990: D3E10020 1C8275CA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[204:205], acc[60:61], v[32:35]  \n" // 000000003998: D3E10020 1C8279CC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[206:207], acc[62:63], v[32:35]  \n" // 0000000039A0: D3E10020 1C827DCE
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[208:209], acc[64:65], v[32:35]  \n" // 0000000039A8: D3E10020 1C8281D0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[210:211], acc[66:67], v[32:35]  \n" // 0000000039B0: D3E10020 1C8285D2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[212:213], acc[68:69], v[32:35]  \n" // 0000000039B8: D3E10020 1C8289D4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[214:215], acc[70:71], v[32:35]  \n" // 0000000039C0: D3E10020 1C828DD6
"  s_cmp_le_i32  s83, s82                                 \n" // 0000000039C8: BF055253
"  s_cbranch_scc1  label_0E98_%=                             \n" // 0000000039CC: BF850024
"  v_mov_b32     v25, 0xff800000                          \n" // 0000000039D0: 7E3202FF FF800000
"  s_add_u32     s57, s82, 0                              \n" // 0000000039D8: 80398052
"  v_mov_b32     v24, s57                                 \n" // 0000000039DC: 7E300239
"  v_add_u32     v24, %[wave_id], v24                             \n" // 0000000039E0: 68303007
"  s_sub_u32     s56, s83, 15                             \n" // 0000000039E4: 80B88F53
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 0000000039E8: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 0000000039EC: 0C282884
"  v_add_u32     v20, s56, v20                            \n" // 0000000039F0: 68282838
"  v_add_u32     v21, 1, v20                              \n" // 0000000039F4: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 0000000039F8: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 0000000039FC: 682E2883
"  v_cmp_le_u32  s[38:39], v20, v24                       \n" // 000000003A00: D0CB0026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000003A08: 682828C0
"  s_nop         0x0000                                   \n" // 000000003A0C: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000003A10: D1000020 009A4119
"  v_cmp_le_u32  s[38:39], v21, v24                       \n" // 000000003A18: D0CB0026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000003A20: 682A2AC0
"  s_nop         0x0000                                   \n" // 000000003A24: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000003A28: D1000021 009A4319
"  v_cmp_le_u32  s[38:39], v22, v24                       \n" // 000000003A30: D0CB0026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000003A38: 682C2CC0
"  s_nop         0x0000                                   \n" // 000000003A3C: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 000000003A40: D1000022 009A4519
"  v_cmp_le_u32  s[38:39], v23, v24                       \n" // 000000003A48: D0CB0026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 000000003A50: 682E2EC0
"  s_nop         0x0000                                   \n" // 000000003A54: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 000000003A58: D1000023 009A4719
"label_0E98_%=: \n"
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000003A60: BF8CC07F
"  s_barrier                                              \n" // 000000003A64: BF8A0000
"  v_mov_b32     v25, 0xff800000                          \n" // 000000003A68: 7E3202FF FF800000
"  s_and_b32     s56, s48, 0x000000ff                     \n" // 000000003A70: 8638FF30 000000FF
"  v_mov_b32     v24, s56                                 \n" // 000000003A78: 7E300238
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000003A7C: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 000000003A80: 0C282884
"  v_add_u32     v21, 1, v20                              \n" // 000000003A84: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 000000003A88: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 000000003A8C: 682E2883
"  v_cmp_lt_u32  s[38:39], v20, v24                       \n" // 000000003A90: D0C90026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000003A98: 682828C0
"  s_nop         0x0000                                   \n" // 000000003A9C: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000003AA0: D1000020 009A4119
"  v_cmp_lt_u32  s[38:39], v21, v24                       \n" // 000000003AA8: D0C90026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000003AB0: 682A2AC0
"  s_nop         0x0000                                   \n" // 000000003AB4: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000003AB8: D1000021 009A4319
"  v_cmp_lt_u32  s[38:39], v22, v24                       \n" // 000000003AC0: D0C90026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000003AC8: 682C2CC0
"  s_nop         0x0000                                   \n" // 000000003ACC: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 000000003AD0: D1000022 009A4519
"  v_cmp_lt_u32  s[38:39], v23, v24                       \n" // 000000003AD8: D0C90026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 000000003AE0: 682E2EC0
"  s_nop         0x0000                                   \n" // 000000003AE4: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 000000003AE8: D1000023 009A4719
"  v_max3_f32    v24, v32, v33, v32                       \n" // 000000003AF0: D1D30018 04824320
"  v_max3_f32    v24, v34, v35, v24                       \n" // 000000003AF8: D1D30018 04624722
"  ds_write_b32  v3, v24 offset:53504                     \n" // 000000003B00: D81AD100 00001803
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000003B08: BF8CC07F
"  ds_read_b32   v20, v2 offset:53504                     \n" // 000000003B0C: D86CD100 14000002
"  ds_read_b32   v21, v2 offset:53568                     \n" // 000000003B14: D86CD140 15000002
"  ds_read_b32   v22, v2 offset:53632                     \n" // 000000003B1C: D86CD180 16000002
"  ds_read_b32   v23, v2 offset:53696                     \n" // 000000003B24: D86CD1C0 17000002
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000003B2C: BF8CC07F
"  v_max3_f32    v24, v20, v21, v24                       \n" // 000000003B30: D1D30018 04622B14
"  v_max3_f32    v24, v22, v23, v24                       \n" // 000000003B38: D1D30018 04622F16
"  ds_read_b128  acc[144:147], v7 offset:37120            \n" // 000000003B40: DBFE9100 90000007
"  ds_read_b128  acc[148:151], v7 offset:38144            \n" // 000000003B48: DBFE9500 94000007
"  ds_read_b128  acc[152:155], v7 offset:39168            \n" // 000000003B50: DBFE9900 98000007
"  ds_read_b128  acc[156:159], v7 offset:40192            \n" // 000000003B58: DBFE9D00 9C000007
"  ds_read_b128  acc[160:163], v7 offset:41216            \n" // 000000003B60: DBFEA100 A0000007
"  ds_read_b128  acc[164:167], v7 offset:42240            \n" // 000000003B68: DBFEA500 A4000007
"  ds_read_b128  acc[168:171], v7 offset:43264            \n" // 000000003B70: DBFEA900 A8000007
"  ds_read_b128  acc[172:175], v7 offset:44288            \n" // 000000003B78: DBFEAD00 AC000007
"  v_mov_b32     v25, 0xff7fffff                          \n" // 000000003B80: 7E3202FF FF7FFFFF
"  v_cmp_eq_u32  s[38:39], v25, v12                       \n" // 000000003B88: D0CA0026 00021919
"  v_max_f32     v20, v24, v12                            \n" // 000000003B90: 16281918
"  v_sub_f32     v16, v12, v20                            \n" // 000000003B94: 0420290C
"  v_cndmask_b32  v16, v16, 0, s[38:39]                   \n" // 000000003B98: D1000010 00990110
"  v_mov_b32     v12, v20                                 \n" // 000000003BA0: 7E180314
"  v_mul_f32     v21, s5, v20                             \n" // 000000003BA4: 0A2A2805
"  v_mul_f32     v16, s5, v16                             \n" // 000000003BA8: 0A202005
"  v_exp_f32     v16, v16                                 \n" // 000000003BAC: 7E204110
"  v_fma_f32     v32, v32, s5, -v21                       \n" // 000000003BB0: D1CB0020 84540B20
"  v_fma_f32     v33, v33, s5, -v21                       \n" // 000000003BB8: D1CB0021 84540B21
"  v_fma_f32     v34, v34, s5, -v21                       \n" // 000000003BC0: D1CB0022 84540B22
"  v_fma_f32     v35, v35, s5, -v21                       \n" // 000000003BC8: D1CB0023 84540B23
"  v_exp_f32     v32, v32                                 \n" // 000000003BD0: 7E404120
"  v_exp_f32     v33, v33                                 \n" // 000000003BD4: 7E424121
"  v_exp_f32     v34, v34                                 \n" // 000000003BD8: 7E444122
"  v_exp_f32     v35, v35                                 \n" // 000000003BDC: 7E464123
"  v_mul_f32     v14, v16, v14                            \n" // 000000003BE0: 0A1C1D10
"  v_mov_b32     v22, v32                                 \n" // 000000003BE4: 7E2C0320
"  v_add_f32     v22, v33, v22                            \n" // 000000003BE8: 022C2D21
"  v_add_f32     v22, v34, v22                            \n" // 000000003BEC: 022C2D22
"  v_add_f32     v22, v35, v22                            \n" // 000000003BF0: 022C2D23
"  v_add_f32     v14, v22, v14                            \n" // 000000003BF4: 021C1D16
"  v_mov_b32     v29, 0xffff0000                          \n" // 000000003BF8: 7E3A02FF FFFF0000
"  v_mov_b32     v30, 0x7fff0000                          \n" // 000000003C00: 7E3C02FF 7FFF0000
"  v_mov_b32     v31, 0x00007fff                          \n" // 000000003C08: 7E3E02FF 00007FFF
"  v_cmp_u_f32   s[38:39], v32, v32                       \n" // 000000003C10: D0480026 00024120
"  v_add3_u32    v28, v32, v31, 1                         \n" // 000000003C18: D1FF001C 02063F20
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000003C20: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v33, v33                       \n" // 000000003C28: D0480026 00024321
"  v_add3_u32    v28, v33, v31, 1                         \n" // 000000003C30: D1FF001C 02063F21
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000003C38: D1000015 009A3D1C
"  v_perm_b32    v32, v21, v20, s52                       \n" // 000000003C40: D1ED0020 00D22915
"  v_cmp_u_f32   s[38:39], v34, v34                       \n" // 000000003C48: D0480026 00024522
"  v_add3_u32    v28, v34, v31, 1                         \n" // 000000003C50: D1FF001C 02063F22
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000003C58: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v35, v35                       \n" // 000000003C60: D0480026 00024723
"  v_add3_u32    v28, v35, v31, 1                         \n" // 000000003C68: D1FF001C 02063F23
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000003C70: D1000015 009A3D1C
"  v_perm_b32    v33, v21, v20, s52                       \n" // 000000003C78: D1ED0021 00D22915
"  s_nop         0x0002                                   \n" // 000000003C80: BF800002
"  s_add_u32     s83, s84, s83                            \n" // 000000003C84: 80535354
"  s_nop         0x0000                                   \n" // 000000003C88: BF800000
"  v_mov_b32     v22, v16                                 \n" // 000000003C8C: 7E2C0310
"  v_mov_b32     v23, v16                                 \n" // 000000003C90: 7E2E0310
"  v_pk_mul_f32  [%[o_regs_0], %[o_regs_1]], v[22:23], [%[o_regs_0], %[o_regs_1]]             \n" // 000000003C94: D3B14028 18025116
"  v_pk_mul_f32  [%[o_regs_2], %[o_regs_3]], v[22:23], [%[o_regs_2], %[o_regs_3]]             \n" // 000000003C9C: D3B1402A 18025516
"  v_pk_mul_f32  [%[o_regs_4], %[o_regs_5]], v[22:23], [%[o_regs_4], %[o_regs_5]]             \n" // 000000003CA4: D3B1402C 18025916
"  v_pk_mul_f32  [%[o_regs_6], %[o_regs_7]], v[22:23], [%[o_regs_6], %[o_regs_7]]             \n" // 000000003CAC: D3B1402E 18025D16
"  v_pk_mul_f32  [%[o_regs_8], %[o_regs_9]], v[22:23], [%[o_regs_8], %[o_regs_9]]             \n" // 000000003CB4: D3B14030 18026116
"  v_pk_mul_f32  [%[o_regs_10], %[o_regs_11]], v[22:23], [%[o_regs_10], %[o_regs_11]]             \n" // 000000003CBC: D3B14032 18026516
"  v_pk_mul_f32  [%[o_regs_12], %[o_regs_13]], v[22:23], [%[o_regs_12], %[o_regs_13]]             \n" // 000000003CC4: D3B14034 18026916
"  v_pk_mul_f32  [%[o_regs_14], %[o_regs_15]], v[22:23], [%[o_regs_14], %[o_regs_15]]             \n" // 000000003CCC: D3B14036 18026D16
"  v_pk_mul_f32  [%[o_regs_16], %[o_regs_17]], v[22:23], [%[o_regs_16], %[o_regs_17]]             \n" // 000000003CD4: D3B14038 18027116
"  v_pk_mul_f32  [%[o_regs_18], %[o_regs_19]], v[22:23], [%[o_regs_18], %[o_regs_19]]             \n" // 000000003CDC: D3B1403A 18027516
"  v_pk_mul_f32  [%[o_regs_20], %[o_regs_21]], v[22:23], [%[o_regs_20], %[o_regs_21]]             \n" // 000000003CE4: D3B1403C 18027916
"  v_pk_mul_f32  [%[o_regs_22], %[o_regs_23]], v[22:23], [%[o_regs_22], %[o_regs_23]]             \n" // 000000003CEC: D3B1403E 18027D16
"  v_pk_mul_f32  [%[o_regs_24], %[o_regs_25]], v[22:23], [%[o_regs_24], %[o_regs_25]]             \n" // 000000003CF4: D3B14040 18028116
"  v_pk_mul_f32  [%[o_regs_26], %[o_regs_27]], v[22:23], [%[o_regs_26], %[o_regs_27]]             \n" // 000000003CFC: D3B14042 18028516
"  v_pk_mul_f32  [%[o_regs_28], %[o_regs_29]], v[22:23], [%[o_regs_28], %[o_regs_29]]             \n" // 000000003D04: D3B14044 18028916
"  v_pk_mul_f32  [%[o_regs_30], %[o_regs_31]], v[22:23], [%[o_regs_30], %[o_regs_31]]             \n" // 000000003D0C: D3B14046 18028D16
"  v_pk_mul_f32  [%[o_regs_32], %[o_regs_33]], v[22:23], [%[o_regs_32], %[o_regs_33]]             \n" // 000000003D14: D3B14048 18029116
"  v_pk_mul_f32  [%[o_regs_34], %[o_regs_35]], v[22:23], [%[o_regs_34], %[o_regs_35]]             \n" // 000000003D1C: D3B1404A 18029516
"  v_pk_mul_f32  [%[o_regs_36], %[o_regs_37]], v[22:23], [%[o_regs_36], %[o_regs_37]]             \n" // 000000003D24: D3B1404C 18029916
"  v_pk_mul_f32  [%[o_regs_38], %[o_regs_39]], v[22:23], [%[o_regs_38], %[o_regs_39]]             \n" // 000000003D2C: D3B1404E 18029D16
"  v_pk_mul_f32  [%[o_regs_40], %[o_regs_41]], v[22:23], [%[o_regs_40], %[o_regs_41]]             \n" // 000000003D34: D3B14050 1802A116
"  v_pk_mul_f32  [%[o_regs_42], %[o_regs_43]], v[22:23], [%[o_regs_42], %[o_regs_43]]             \n" // 000000003D3C: D3B14052 1802A516
"  v_pk_mul_f32  [%[o_regs_44], %[o_regs_45]], v[22:23], [%[o_regs_44], %[o_regs_45]]             \n" // 000000003D44: D3B14054 1802A916
"  v_pk_mul_f32  [%[o_regs_46], %[o_regs_47]], v[22:23], [%[o_regs_46], %[o_regs_47]]             \n" // 000000003D4C: D3B14056 1802AD16
"  v_pk_mul_f32  [%[o_regs_48], %[o_regs_49]], v[22:23], [%[o_regs_48], %[o_regs_49]]             \n" // 000000003D54: D3B14058 1802B116
"  v_pk_mul_f32  [%[o_regs_50], %[o_regs_51]], v[22:23], [%[o_regs_50], %[o_regs_51]]             \n" // 000000003D5C: D3B1405A 1802B516
"  v_pk_mul_f32  [%[o_regs_52], %[o_regs_53]], v[22:23], [%[o_regs_52], %[o_regs_53]]             \n" // 000000003D64: D3B1405C 1802B916
"  v_pk_mul_f32  [%[o_regs_54], %[o_regs_55]], v[22:23], [%[o_regs_54], %[o_regs_55]]             \n" // 000000003D6C: D3B1405E 1802BD16
"  v_pk_mul_f32  [%[o_regs_56], %[o_regs_57]], v[22:23], [%[o_regs_56], %[o_regs_57]]             \n" // 000000003D74: D3B14060 1802C116
"  v_pk_mul_f32  [%[o_regs_58], %[o_regs_59]], v[22:23], [%[o_regs_58], %[o_regs_59]]             \n" // 000000003D7C: D3B14062 1802C516
"  v_pk_mul_f32  [%[o_regs_60], %[o_regs_61]], v[22:23], [%[o_regs_60], %[o_regs_61]]         \n" // 000000003D84: D3B14064 1802C916
"  v_pk_mul_f32  [%[o_regs_62], %[o_regs_63]], v[22:23], [%[o_regs_62], %[o_regs_63]]         \n" // 000000003D8C: D3B14066 1802CD16
"  v_pk_mul_f32  [%[o_regs_64], %[o_regs_65]], v[22:23], [%[o_regs_64], %[o_regs_65]]         \n" // 000000003D94: D3B14068 1802D116
"  v_pk_mul_f32  [%[o_regs_66], %[o_regs_67]], v[22:23], [%[o_regs_66], %[o_regs_67]]         \n" // 000000003D9C: D3B1406A 1802D516
"  v_pk_mul_f32  [%[o_regs_68], %[o_regs_69]], v[22:23], [%[o_regs_68], %[o_regs_69]]         \n" // 000000003DA4: D3B1406C 1802D916
"  v_pk_mul_f32  [%[o_regs_70], %[o_regs_71]], v[22:23], [%[o_regs_70], %[o_regs_71]]         \n" // 000000003DAC: D3B1406E 1802DD16
"  v_pk_mul_f32  [%[o_regs_72], %[o_regs_73]], v[22:23], [%[o_regs_72], %[o_regs_73]]         \n" // 000000003DB4: D3B14070 1802E116
"  v_pk_mul_f32  [%[o_regs_74], %[o_regs_75]], v[22:23], [%[o_regs_74], %[o_regs_75]]         \n" // 000000003DBC: D3B14072 1802E516
"  v_pk_mul_f32  [%[o_regs_76], %[o_regs_77]], v[22:23], [%[o_regs_76], %[o_regs_77]]         \n" // 000000003DC4: D3B14074 1802E916
"  v_pk_mul_f32  [%[o_regs_78], %[o_regs_79]], v[22:23], [%[o_regs_78], %[o_regs_79]]         \n" // 000000003DCC: D3B14076 1802ED16
"  v_pk_mul_f32  [%[o_regs_80], %[o_regs_81]], v[22:23], [%[o_regs_80], %[o_regs_81]]         \n" // 000000003DD4: D3B14078 1802F116
"  v_pk_mul_f32  [%[o_regs_82], %[o_regs_83]], v[22:23], [%[o_regs_82], %[o_regs_83]]         \n" // 000000003DDC: D3B1407A 1802F516
"  v_pk_mul_f32  [%[o_regs_84], %[o_regs_85]], v[22:23], [%[o_regs_84], %[o_regs_85]]         \n" // 000000003DE4: D3B1407C 1802F916
"  v_pk_mul_f32  [%[o_regs_86], %[o_regs_87]], v[22:23], [%[o_regs_86], %[o_regs_87]]         \n" // 000000003DEC: D3B1407E 1802FD16
"  v_pk_mul_f32  [%[o_regs_88], %[o_regs_89]], v[22:23], [%[o_regs_88], %[o_regs_89]]         \n" // 000000003DF4: D3B14080 18030116
"  v_pk_mul_f32  [%[o_regs_90], %[o_regs_91]], v[22:23], [%[o_regs_90], %[o_regs_91]]         \n" // 000000003DFC: D3B14082 18030516
"  v_pk_mul_f32  [%[o_regs_92], %[o_regs_93]], v[22:23], [%[o_regs_92], %[o_regs_93]]         \n" // 000000003E04: D3B14084 18030916
"  v_pk_mul_f32  [%[o_regs_94], %[o_regs_95]], v[22:23], [%[o_regs_94], %[o_regs_95]]         \n" // 000000003E0C: D3B14086 18030D16
"  v_pk_mul_f32  [%[o_regs_96], %[o_regs_97]], v[22:23], [%[o_regs_96], %[o_regs_97]]         \n" // 000000003E14: D3B14088 18031116
"  v_pk_mul_f32  [%[o_regs_98], %[o_regs_99]], v[22:23], [%[o_regs_98], %[o_regs_99]]         \n" // 000000003E1C: D3B1408A 18031516
"  v_pk_mul_f32  [%[o_regs_100], %[o_regs_101]], v[22:23], [%[o_regs_100], %[o_regs_101]]         \n" // 000000003E24: D3B1408C 18031916
"  v_pk_mul_f32  [%[o_regs_102], %[o_regs_103]], v[22:23], [%[o_regs_102], %[o_regs_103]]         \n" // 000000003E2C: D3B1408E 18031D16
"  v_pk_mul_f32  [%[o_regs_104], %[o_regs_105]], v[22:23], [%[o_regs_104], %[o_regs_105]]         \n" // 000000003E34: D3B14090 18032116
"  v_pk_mul_f32  [%[o_regs_106], %[o_regs_107]], v[22:23], [%[o_regs_106], %[o_regs_107]]         \n" // 000000003E3C: D3B14092 18032516
"  v_pk_mul_f32  [%[o_regs_108], %[o_regs_109]], v[22:23], [%[o_regs_108], %[o_regs_109]]         \n" // 000000003E44: D3B14094 18032916
"  v_pk_mul_f32  [%[o_regs_110], %[o_regs_111]], v[22:23], [%[o_regs_110], %[o_regs_111]]         \n" // 000000003E4C: D3B14096 18032D16
"  v_pk_mul_f32  [%[o_regs_112], %[o_regs_113]], v[22:23], [%[o_regs_112], %[o_regs_113]]         \n" // 000000003E54: D3B14098 18033116
"  v_pk_mul_f32  [%[o_regs_114], %[o_regs_115]], v[22:23], [%[o_regs_114], %[o_regs_115]]         \n" // 000000003E5C: D3B1409A 18033516
"  v_pk_mul_f32  [%[o_regs_116], %[o_regs_117]], v[22:23], [%[o_regs_116], %[o_regs_117]]         \n" // 000000003E64: D3B1409C 18033916
"  v_pk_mul_f32  [%[o_regs_118], %[o_regs_119]], v[22:23], [%[o_regs_118], %[o_regs_119]]         \n" // 000000003E6C: D3B1409E 18033D16
"  v_pk_mul_f32  [%[o_regs_120], %[o_regs_121]], v[22:23], [%[o_regs_120], %[o_regs_121]]         \n" // 000000003E74: D3B140A0 18034116
"  v_pk_mul_f32  [%[o_regs_122], %[o_regs_123]], v[22:23], [%[o_regs_122], %[o_regs_123]]         \n" // 000000003E7C: D3B140A2 18034516
"  v_pk_mul_f32  [%[o_regs_124], %[o_regs_125]], v[22:23], [%[o_regs_124], %[o_regs_125]]         \n" // 000000003E84: D3B140A4 18034916
"  v_pk_mul_f32  [%[o_regs_126], %[o_regs_127]], v[22:23], [%[o_regs_126], %[o_regs_127]]         \n" // 000000003E8C: D3B140A6 18034D16
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000003E94: BF8CC07F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]], acc[144:145], v[32:33], [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]]  \n" // 000000003E98: D3E10028 0CA24190
"  ds_read_b128  acc[176:179], v7 offset:45312            \n" // 000000003EA0: DBFEB100 B0000007
"  ds_read_b128  acc[180:183], v7 offset:46336            \n" // 000000003EA8: DBFEB500 B4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]], acc[146:147], v[32:33], [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]]  \n" // 000000003EB0: D3E1002C 0CB24192
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]], acc[148:149], v[32:33], [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]]  \n" // 000000003EB8: D3E10030 0CC24194
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]], acc[150:151], v[32:33], [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]]  \n" // 000000003EC0: D3E10034 0CD24196
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]], acc[152:153], v[32:33], [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]]  \n" // 000000003EC8: D3E10038 0CE24198
"  ds_read_b128  acc[184:187], v7 offset:47360            \n" // 000000003ED0: DBFEB900 B8000007
"  ds_read_b128  acc[188:191], v7 offset:48384            \n" // 000000003ED8: DBFEBD00 BC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]], acc[154:155], v[32:33], [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]]  \n" // 000000003EE0: D3E1003C 0CF2419A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]], acc[156:157], v[32:33], [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]]  \n" // 000000003EE8: D3E10040 0D02419C
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]], acc[158:159], v[32:33], [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]]  \n" // 000000003EF0: D3E10044 0D12419E
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]], acc[160:161], v[32:33], [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]]  \n" // 000000003EF8: D3E10048 0D2241A0
"  ds_read_b128  acc[192:195], v7 offset:49408            \n" // 000000003F00: DBFEC100 C0000007
"  ds_read_b128  acc[196:199], v7 offset:50432            \n" // 000000003F08: DBFEC500 C4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]], acc[162:163], v[32:33], [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]]  \n" // 000000003F10: D3E1004C 0D3241A2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]], acc[164:165], v[32:33], [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]]  \n" // 000000003F18: D3E10050 0D4241A4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]], acc[166:167], v[32:33], [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]]  \n" // 000000003F20: D3E10054 0D5241A6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]], acc[168:169], v[32:33], [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]]  \n" // 000000003F28: D3E10058 0D6241A8
"  ds_read_b128  acc[200:203], v7 offset:51456            \n" // 000000003F30: DBFEC900 C8000007
"  ds_read_b128  acc[204:207], v7 offset:52480            \n" // 000000003F38: DBFECD00 CC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]], acc[170:171], v[32:33], [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]]  \n" // 000000003F40: D3E1005C 0D7241AA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]], acc[172:173], v[32:33], [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]]  \n" // 000000003F48: D3E10060 0D8241AC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]], acc[174:175], v[32:33], [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]]  \n" // 000000003F50: D3E10064 0D9241AE
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000003F58: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]], acc[176:177], v[32:33], [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]]  \n" // 000000003F5C: D3E10068 0DA241B0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]], acc[178:179], v[32:33], [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]]  \n" // 000000003F64: D3E1006C 0DB241B2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]], acc[180:181], v[32:33], [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]]  \n" // 000000003F6C: D3E10070 0DC241B4
"  s_waitcnt     vmcnt(18) & lgkmcnt(0)                   \n" // 000000003F74: BF8C4072
"  s_barrier                                              \n" // 000000003F78: BF8A0000
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]], acc[182:183], v[32:33], [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]]  \n" // 000000003F7C: D3E10074 0DD241B6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]], acc[184:185], v[32:33], [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]]  \n" // 000000003F84: D3E10078 0DE241B8
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]], acc[186:187], v[32:33], [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]]  \n" // 000000003F8C: D3E1007C 0DF241BA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]], acc[188:189], v[32:33], [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]]  \n" // 000000003F94: D3E10080 0E0241BC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]], acc[190:191], v[32:33], [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]]  \n" // 000000003F9C: D3E10084 0E1241BE
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]], acc[192:193], v[32:33], [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]]  \n" // 000000003FA4: D3E10088 0E2241C0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]], acc[194:195], v[32:33], [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]]  \n" // 000000003FAC: D3E1008C 0E3241C2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]], acc[196:197], v[32:33], [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]]  \n" // 000000003FB4: D3E10090 0E4241C4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]], acc[198:199], v[32:33], [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]]  \n" // 000000003FBC: D3E10094 0E5241C6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]], acc[200:201], v[32:33], [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]]  \n" // 000000003FC4: D3E10098 0E6241C8
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]], acc[202:203], v[32:33], [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]]  \n" // 000000003FCC: D3E1009C 0E7241CA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]], acc[204:205], v[32:33], [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]]  \n" // 000000003FD4: D3E100A0 0E8241CC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]], acc[206:207], v[32:33], [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]]  \n" // 000000003FDC: D3E100A4 0E9241CE
"  s_nop         0x0008                                   \n" // 000000003FE4: BF800008
"  s_branch      label_11F9%=                               \n" // 000000003FE8: BF8201FE 
"label_0FFB_%=: \n"
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000003FEC: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[144:145], acc[0:1], 0  \n" // 000000003FF0: D3E10020 1A020190
"  ds_read_b128  acc[176:179], v4 offset:19584            \n" // 000000003FF8: DBFE4C80 B0000004
"  ds_read_b128  acc[180:183], v4 offset:19648            \n" // 000000004000: DBFE4CC0 B4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[146:147], acc[2:3], v[32:35]  \n" // 000000004008: D3E10020 1C820592
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[148:149], acc[4:5], v[32:35]  \n" // 000000004010: D3E10020 1C820994
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[150:151], acc[6:7], v[32:35]  \n" // 000000004018: D3E10020 1C820D96
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[152:153], acc[8:9], v[32:35]  \n" // 000000004020: D3E10020 1C821198
"  ds_read_b128  acc[184:187], v4 offset:19840            \n" // 000000004028: DBFE4D80 B8000004
"  ds_read_b128  acc[188:191], v4 offset:19904            \n" // 000000004030: DBFE4DC0 BC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[154:155], acc[10:11], v[32:35]  \n" // 000000004038: D3E10020 1C82159A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[156:157], acc[12:13], v[32:35]  \n" // 000000004040: D3E10020 1C82199C
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[158:159], acc[14:15], v[32:35]  \n" // 000000004048: D3E10020 1C821D9E
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000004050: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[160:161], acc[16:17], v[32:35]  \n" // 000000004054: D3E10020 1C8221A0
"  ds_read_b128  acc[192:195], v4 offset:20096            \n" // 00000000405C: DBFE4E80 C0000004
"  ds_read_b128  acc[196:199], v4 offset:20160            \n" // 000000004064: DBFE4EC0 C4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[162:163], acc[18:19], v[32:35]  \n" // 00000000406C: D3E10020 1C8225A2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[164:165], acc[20:21], v[32:35]  \n" // 000000004074: D3E10020 1C8229A4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[166:167], acc[22:23], v[32:35]  \n" // 00000000407C: D3E10020 1C822DA6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[168:169], acc[24:25], v[32:35]  \n" // 000000004084: D3E10020 1C8231A8
"  ds_read_b128  acc[200:203], v4 offset:20352            \n" // 00000000408C: DBFE4F80 C8000004
"  ds_read_b128  acc[204:207], v4 offset:20416            \n" // 000000004094: DBFE4FC0 CC000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[170:171], acc[26:27], v[32:35]  \n" // 00000000409C: D3E10020 1C8235AA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[172:173], acc[28:29], v[32:35]  \n" // 0000000040A4: D3E10020 1C8239AC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[174:175], acc[30:31], v[32:35]  \n" // 0000000040AC: D3E10020 1C823DAE
"  s_waitcnt     lgkmcnt(4)                               \n" // 0000000040B4: BF8CC47F
"  s_barrier                                              \n" // 0000000040B8: BF8A0000
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[176:177], acc[32:33], v[32:35]  \n" // 0000000040BC: D3E10020 1C8241B0
"  ds_read_b128  acc[208:211], v4 offset:20608            \n" // 0000000040C4: DBFE5080 D0000004
"  ds_read_b128  acc[212:215], v4 offset:20672            \n" // 0000000040CC: DBFE50C0 D4000004
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[178:179], acc[34:35], v[32:35]  \n" // 0000000040D4: D3E10020 1C8245B2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[180:181], acc[36:37], v[32:35]  \n" // 0000000040DC: D3E10020 1C8249B4
"  v_perm_b32    v168, v22, v20, s53                      \n" // 0000000040E4: D1ED00A8 00D62916
"  v_perm_b32    v170, v22, v20, s52                      \n" // 0000000040EC: D1ED00AA 00D22916
"  v_perm_b32    v169, v26, v24, s53                      \n" // 0000000040F4: D1ED00A9 00D6311A
"  v_perm_b32    v171, v26, v24, s52                      \n" // 0000000040FC: D1ED00AB 00D2311A
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[182:183], acc[38:39], v[32:35]  \n" // 000000004104: D3E10020 1C824DB6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[184:185], acc[40:41], v[32:35]  \n" // 00000000410C: D3E10020 1C8251B8
"  ds_write_b128  v6, v[168:171] offset:45312             \n" // 000000004114: D9BEB100 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[186:187], acc[42:43], v[32:35]  \n" // 00000000411C: D3E10020 1C8255BA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[188:189], acc[44:45], v[32:35]  \n" // 000000004124: D3E10020 1C8259BC
"  v_perm_b32    v168, v23, v21, s53                      \n" // 00000000412C: D1ED00A8 00D62B17
"  v_perm_b32    v170, v23, v21, s52                      \n" // 000000004134: D1ED00AA 00D22B17
"  v_perm_b32    v169, v27, v25, s53                      \n" // 00000000413C: D1ED00A9 00D6331B
"  v_perm_b32    v171, v27, v25, s52                      \n" // 000000004144: D1ED00AB 00D2331B
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[190:191], acc[46:47], v[32:35]  \n" // 00000000414C: D3E10020 1C825DBE
"  s_waitcnt     lgkmcnt(1)                               \n" // 000000004154: BF8CC17F
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[192:193], acc[48:49], v[32:35]  \n" // 000000004158: D3E10020 1C8261C0
"  ds_write_b128  v6, v[168:171] offset:46336             \n" // 000000004160: D9BEB500 0000A806
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[194:195], acc[50:51], v[32:35]  \n" // 000000004168: D3E10020 1C8265C2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[196:197], acc[52:53], v[32:35]  \n" // 000000004170: D3E10020 1C8269C4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[198:199], acc[54:55], v[32:35]  \n" // 000000004178: D3E10020 1C826DC6
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[200:201], acc[56:57], v[32:35]  \n" // 000000004180: D3E10020 1C8271C8
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[202:203], acc[58:59], v[32:35]  \n" // 000000004188: D3E10020 1C8275CA
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[204:205], acc[60:61], v[32:35]  \n" // 000000004190: D3E10020 1C8279CC
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[206:207], acc[62:63], v[32:35]  \n" // 000000004198: D3E10020 1C827DCE
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[208:209], acc[64:65], v[32:35]  \n" // 0000000041A0: D3E10020 1C8281D0
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[210:211], acc[66:67], v[32:35]  \n" // 0000000041A8: D3E10020 1C8285D2
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[212:213], acc[68:69], v[32:35]  \n" // 0000000041B0: D3E10020 1C8289D4
"  v_mfma_f32_16x16x16_bf16  v[32:35], acc[214:215], acc[70:71], v[32:35]  \n" // 0000000041B8: D3E10020 1C828DD6
"  s_cmp_le_i32  s83, s82                                 \n" // 0000000041C0: BF055253
"  s_cbranch_scc1  label_1096_%=                             \n" // 0000000041C4: BF850024
"  v_mov_b32     v25, 0xff800000                          \n" // 0000000041C8: 7E3202FF FF800000
"  s_add_u32     s57, s82, 0                              \n" // 0000000041D0: 80398052
"  v_mov_b32     v24, s57                                 \n" // 0000000041D4: 7E300239
"  v_add_u32     v24, %[wave_id], v24                             \n" // 0000000041D8: 68303007
"  s_sub_u32     s56, s83, 15                             \n" // 0000000041DC: 80B88F53
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 0000000041E0: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 0000000041E4: 0C282884
"  v_add_u32     v20, s56, v20                            \n" // 0000000041E8: 68282838
"  v_add_u32     v21, 1, v20                              \n" // 0000000041EC: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 0000000041F0: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 0000000041F4: 682E2883
"  v_cmp_le_u32  s[38:39], v20, v24                       \n" // 0000000041F8: D0CB0026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000004200: 682828C0
"  s_nop         0x0000                                   \n" // 000000004204: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000004208: D1000020 009A4119
"  v_cmp_le_u32  s[38:39], v21, v24                       \n" // 000000004210: D0CB0026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 000000004218: 682A2AC0
"  s_nop         0x0000                                   \n" // 00000000421C: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 000000004220: D1000021 009A4319
"  v_cmp_le_u32  s[38:39], v22, v24                       \n" // 000000004228: D0CB0026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 000000004230: 682C2CC0
"  s_nop         0x0000                                   \n" // 000000004234: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 000000004238: D1000022 009A4519
"  v_cmp_le_u32  s[38:39], v23, v24                       \n" // 000000004240: D0CB0026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 000000004248: 682E2EC0
"  s_nop         0x0000                                   \n" // 00000000424C: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 000000004250: D1000023 009A4719
"label_1096_%=: \n"
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000004258: BF8CC07F
"  s_barrier                                              \n" // 00000000425C: BF8A0000
"  v_mov_b32     v25, 0xff800000                          \n" // 000000004260: 7E3202FF FF800000
"  s_and_b32     s56, s48, 0x000000ff                     \n" // 000000004268: 8638FF30 000000FF
"  v_mov_b32     v24, s56                                 \n" // 000000004270: 7E300238
"  v_lshrrev_b32  v20, 4, %[threadIdxx]                             \n" // 000000004274: 20280084
"  v_mul_i32_i24  v20, 4, v20                             \n" // 000000004278: 0C282884
"  v_add_u32     v21, 1, v20                              \n" // 00000000427C: 682A2881
"  v_add_u32     v22, 2, v20                              \n" // 000000004280: 682C2882
"  v_add_u32     v23, 3, v20                              \n" // 000000004284: 682E2883
"  v_cmp_lt_u32  s[38:39], v20, v24                       \n" // 000000004288: D0C90026 00023114
"  v_add_u32     v20, 64, v20                             \n" // 000000004290: 682828C0
"  s_nop         0x0000                                   \n" // 000000004294: BF800000
"  v_cndmask_b32  v32, v25, v32, s[38:39]                 \n" // 000000004298: D1000020 009A4119
"  v_cmp_lt_u32  s[38:39], v21, v24                       \n" // 0000000042A0: D0C90026 00023115
"  v_add_u32     v21, 64, v21                             \n" // 0000000042A8: 682A2AC0
"  s_nop         0x0000                                   \n" // 0000000042AC: BF800000
"  v_cndmask_b32  v33, v25, v33, s[38:39]                 \n" // 0000000042B0: D1000021 009A4319
"  v_cmp_lt_u32  s[38:39], v22, v24                       \n" // 0000000042B8: D0C90026 00023116
"  v_add_u32     v22, 64, v22                             \n" // 0000000042C0: 682C2CC0
"  s_nop         0x0000                                   \n" // 0000000042C4: BF800000
"  v_cndmask_b32  v34, v25, v34, s[38:39]                 \n" // 0000000042C8: D1000022 009A4519
"  v_cmp_lt_u32  s[38:39], v23, v24                       \n" // 0000000042D0: D0C90026 00023117
"  v_add_u32     v23, 64, v23                             \n" // 0000000042D8: 682E2EC0
"  s_nop         0x0000                                   \n" // 0000000042DC: BF800000
"  v_cndmask_b32  v35, v25, v35, s[38:39]                 \n" // 0000000042E0: D1000023 009A4719
"  v_max3_f32    v24, v32, v33, v32                       \n" // 0000000042E8: D1D30018 04824320
"  v_max3_f32    v24, v34, v35, v24                       \n" // 0000000042F0: D1D30018 04624722
"  ds_write_b32  v3, v24 offset:53504                     \n" // 0000000042F8: D81AD100 00001803
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000004300: BF8CC07F
"  ds_read_b32   v20, v2 offset:53504                     \n" // 000000004304: D86CD100 14000002
"  ds_read_b32   v21, v2 offset:53568                     \n" // 00000000430C: D86CD140 15000002
"  ds_read_b32   v22, v2 offset:53632                     \n" // 000000004314: D86CD180 16000002
"  ds_read_b32   v23, v2 offset:53696                     \n" // 00000000431C: D86CD1C0 17000002
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000004324: BF8CC07F
"  v_max3_f32    v24, v20, v21, v24                       \n" // 000000004328: D1D30018 04622B14
"  v_max3_f32    v24, v22, v23, v24                       \n" // 000000004330: D1D30018 04622F16
"  ds_read_b128  acc[144:147], v7 offset:37120            \n" // 000000004338: DBFE9100 90000007
"  ds_read_b128  acc[148:151], v7 offset:38144            \n" // 000000004340: DBFE9500 94000007
"  ds_read_b128  acc[152:155], v7 offset:39168            \n" // 000000004348: DBFE9900 98000007
"  ds_read_b128  acc[156:159], v7 offset:40192            \n" // 000000004350: DBFE9D00 9C000007
"  ds_read_b128  acc[160:163], v7 offset:41216            \n" // 000000004358: DBFEA100 A0000007
"  ds_read_b128  acc[164:167], v7 offset:42240            \n" // 000000004360: DBFEA500 A4000007
"  ds_read_b128  acc[168:171], v7 offset:43264            \n" // 000000004368: DBFEA900 A8000007
"  ds_read_b128  acc[172:175], v7 offset:44288            \n" // 000000004370: DBFEAD00 AC000007
"  v_mov_b32     v25, 0xff7fffff                          \n" // 000000004378: 7E3202FF FF7FFFFF
"  v_cmp_eq_u32  s[38:39], v25, v12                       \n" // 000000004380: D0CA0026 00021919
"  v_max_f32     v20, v24, v12                            \n" // 000000004388: 16281918
"  v_sub_f32     v16, v12, v20                            \n" // 00000000438C: 0420290C
"  v_cndmask_b32  v16, v16, 0, s[38:39]                   \n" // 000000004390: D1000010 00990110
"  v_mov_b32     v12, v20                                 \n" // 000000004398: 7E180314
"  v_mul_f32     v21, s5, v20                             \n" // 00000000439C: 0A2A2805
"  v_mul_f32     v16, s5, v16                             \n" // 0000000043A0: 0A202005
"  v_exp_f32     v16, v16                                 \n" // 0000000043A4: 7E204110
"  v_fma_f32     v32, v32, s5, -v21                       \n" // 0000000043A8: D1CB0020 84540B20
"  v_fma_f32     v33, v33, s5, -v21                       \n" // 0000000043B0: D1CB0021 84540B21
"  v_fma_f32     v34, v34, s5, -v21                       \n" // 0000000043B8: D1CB0022 84540B22
"  v_fma_f32     v35, v35, s5, -v21                       \n" // 0000000043C0: D1CB0023 84540B23
"  v_exp_f32     v32, v32                                 \n" // 0000000043C8: 7E404120
"  v_exp_f32     v33, v33                                 \n" // 0000000043CC: 7E424121
"  v_exp_f32     v34, v34                                 \n" // 0000000043D0: 7E444122
"  v_exp_f32     v35, v35                                 \n" // 0000000043D4: 7E464123
"  v_mul_f32     v14, v16, v14                            \n" // 0000000043D8: 0A1C1D10
"  v_mov_b32     v22, v32                                 \n" // 0000000043DC: 7E2C0320
"  v_add_f32     v22, v33, v22                            \n" // 0000000043E0: 022C2D21
"  v_add_f32     v22, v34, v22                            \n" // 0000000043E4: 022C2D22
"  v_add_f32     v22, v35, v22                            \n" // 0000000043E8: 022C2D23
"  v_add_f32     v14, v22, v14                            \n" // 0000000043EC: 021C1D16
"  v_mov_b32     v29, 0xffff0000                          \n" // 0000000043F0: 7E3A02FF FFFF0000
"  v_mov_b32     v30, 0x7fff0000                          \n" // 0000000043F8: 7E3C02FF 7FFF0000
"  v_mov_b32     v31, 0x00007fff                          \n" // 000000004400: 7E3E02FF 00007FFF
"  v_cmp_u_f32   s[38:39], v32, v32                       \n" // 000000004408: D0480026 00024120
"  v_add3_u32    v28, v32, v31, 1                         \n" // 000000004410: D1FF001C 02063F20
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000004418: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v33, v33                       \n" // 000000004420: D0480026 00024321
"  v_add3_u32    v28, v33, v31, 1                         \n" // 000000004428: D1FF001C 02063F21
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000004430: D1000015 009A3D1C
"  v_perm_b32    v32, v21, v20, s52                       \n" // 000000004438: D1ED0020 00D22915
"  v_cmp_u_f32   s[38:39], v34, v34                       \n" // 000000004440: D0480026 00024522
"  v_add3_u32    v28, v34, v31, 1                         \n" // 000000004448: D1FF001C 02063F22
"  v_cndmask_b32  v20, v28, v30, s[38:39]                 \n" // 000000004450: D1000014 009A3D1C
"  v_cmp_u_f32   s[38:39], v35, v35                       \n" // 000000004458: D0480026 00024723
"  v_add3_u32    v28, v35, v31, 1                         \n" // 000000004460: D1FF001C 02063F23
"  v_cndmask_b32  v21, v28, v30, s[38:39]                 \n" // 000000004468: D1000015 009A3D1C
"  v_perm_b32    v33, v21, v20, s52                       \n" // 000000004470: D1ED0021 00D22915
"  s_nop         0x0002                                   \n" // 000000004478: BF800002
"  s_add_u32     s83, s84, s83                            \n" // 00000000447C: 80535354
"  s_nop         0x0000                                   \n" // 000000004480: BF800000
"  v_mov_b32     v22, v16                                 \n" // 000000004484: 7E2C0310
"  v_mov_b32     v23, v16                                 \n" // 000000004488: 7E2E0310
"  v_pk_mul_f32  [%[o_regs_0], %[o_regs_1]], v[22:23], [%[o_regs_0], %[o_regs_1]]             \n" // 00000000448C: D3B14028 18025116
"  v_pk_mul_f32  [%[o_regs_2], %[o_regs_3]], v[22:23], [%[o_regs_2], %[o_regs_3]]             \n" // 000000004494: D3B1402A 18025516
"  v_pk_mul_f32  [%[o_regs_4], %[o_regs_5]], v[22:23], [%[o_regs_4], %[o_regs_5]]             \n" // 00000000449C: D3B1402C 18025916
"  v_pk_mul_f32  [%[o_regs_6], %[o_regs_7]], v[22:23], [%[o_regs_6], %[o_regs_7]]             \n" // 0000000044A4: D3B1402E 18025D16
"  v_pk_mul_f32  [%[o_regs_8], %[o_regs_9]], v[22:23], [%[o_regs_8], %[o_regs_9]]             \n" // 0000000044AC: D3B14030 18026116
"  v_pk_mul_f32  [%[o_regs_10], %[o_regs_11]], v[22:23], [%[o_regs_10], %[o_regs_11]]             \n" // 0000000044B4: D3B14032 18026516
"  v_pk_mul_f32  [%[o_regs_12], %[o_regs_13]], v[22:23], [%[o_regs_12], %[o_regs_13]]             \n" // 0000000044BC: D3B14034 18026916
"  v_pk_mul_f32  [%[o_regs_14], %[o_regs_15]], v[22:23], [%[o_regs_14], %[o_regs_15]]             \n" // 0000000044C4: D3B14036 18026D16
"  v_pk_mul_f32  [%[o_regs_16], %[o_regs_17]], v[22:23], [%[o_regs_16], %[o_regs_17]]             \n" // 0000000044CC: D3B14038 18027116
"  v_pk_mul_f32  [%[o_regs_18], %[o_regs_19]], v[22:23], [%[o_regs_18], %[o_regs_19]]             \n" // 0000000044D4: D3B1403A 18027516
"  v_pk_mul_f32  [%[o_regs_20], %[o_regs_21]], v[22:23], [%[o_regs_20], %[o_regs_21]]             \n" // 0000000044DC: D3B1403C 18027916
"  v_pk_mul_f32  [%[o_regs_22], %[o_regs_23]], v[22:23], [%[o_regs_22], %[o_regs_23]]             \n" // 0000000044E4: D3B1403E 18027D16
"  v_pk_mul_f32  [%[o_regs_24], %[o_regs_25]], v[22:23], [%[o_regs_24], %[o_regs_25]]             \n" // 0000000044EC: D3B14040 18028116
"  v_pk_mul_f32  [%[o_regs_26], %[o_regs_27]], v[22:23], [%[o_regs_26], %[o_regs_27]]             \n" // 0000000044F4: D3B14042 18028516
"  v_pk_mul_f32  [%[o_regs_28], %[o_regs_29]], v[22:23], [%[o_regs_28], %[o_regs_29]]             \n" // 0000000044FC: D3B14044 18028916
"  v_pk_mul_f32  [%[o_regs_30], %[o_regs_31]], v[22:23], [%[o_regs_30], %[o_regs_31]]             \n" // 000000004504: D3B14046 18028D16
"  v_pk_mul_f32  [%[o_regs_32], %[o_regs_33]], v[22:23], [%[o_regs_32], %[o_regs_33]]             \n" // 00000000450C: D3B14048 18029116
"  v_pk_mul_f32  [%[o_regs_34], %[o_regs_35]], v[22:23], [%[o_regs_34], %[o_regs_35]]             \n" // 000000004514: D3B1404A 18029516
"  v_pk_mul_f32  [%[o_regs_36], %[o_regs_37]], v[22:23], [%[o_regs_36], %[o_regs_37]]             \n" // 00000000451C: D3B1404C 18029916
"  v_pk_mul_f32  [%[o_regs_38], %[o_regs_39]], v[22:23], [%[o_regs_38], %[o_regs_39]]             \n" // 000000004524: D3B1404E 18029D16
"  v_pk_mul_f32  [%[o_regs_40], %[o_regs_41]], v[22:23], [%[o_regs_40], %[o_regs_41]]             \n" // 00000000452C: D3B14050 1802A116
"  v_pk_mul_f32  [%[o_regs_42], %[o_regs_43]], v[22:23], [%[o_regs_42], %[o_regs_43]]             \n" // 000000004534: D3B14052 1802A516
"  v_pk_mul_f32  [%[o_regs_44], %[o_regs_45]], v[22:23], [%[o_regs_44], %[o_regs_45]]             \n" // 00000000453C: D3B14054 1802A916
"  v_pk_mul_f32  [%[o_regs_46], %[o_regs_47]], v[22:23], [%[o_regs_46], %[o_regs_47]]             \n" // 000000004544: D3B14056 1802AD16
"  v_pk_mul_f32  [%[o_regs_48], %[o_regs_49]], v[22:23], [%[o_regs_48], %[o_regs_49]]             \n" // 00000000454C: D3B14058 1802B116
"  v_pk_mul_f32  [%[o_regs_50], %[o_regs_51]], v[22:23], [%[o_regs_50], %[o_regs_51]]             \n" // 000000004554: D3B1405A 1802B516
"  v_pk_mul_f32  [%[o_regs_52], %[o_regs_53]], v[22:23], [%[o_regs_52], %[o_regs_53]]             \n" // 00000000455C: D3B1405C 1802B916
"  v_pk_mul_f32  [%[o_regs_54], %[o_regs_55]], v[22:23], [%[o_regs_54], %[o_regs_55]]             \n" // 000000004564: D3B1405E 1802BD16
"  v_pk_mul_f32  [%[o_regs_56], %[o_regs_57]], v[22:23], [%[o_regs_56], %[o_regs_57]]             \n" // 00000000456C: D3B14060 1802C116
"  v_pk_mul_f32  [%[o_regs_58], %[o_regs_59]], v[22:23], [%[o_regs_58], %[o_regs_59]]             \n" // 000000004574: D3B14062 1802C516
"  v_pk_mul_f32  [%[o_regs_60], %[o_regs_61]], v[22:23], [%[o_regs_60], %[o_regs_61]]         \n" // 00000000457C: D3B14064 1802C916
"  v_pk_mul_f32  [%[o_regs_62], %[o_regs_63]], v[22:23], [%[o_regs_62], %[o_regs_63]]         \n" // 000000004584: D3B14066 1802CD16
"  v_pk_mul_f32  [%[o_regs_64], %[o_regs_65]], v[22:23], [%[o_regs_64], %[o_regs_65]]         \n" // 00000000458C: D3B14068 1802D116
"  v_pk_mul_f32  [%[o_regs_66], %[o_regs_67]], v[22:23], [%[o_regs_66], %[o_regs_67]]         \n" // 000000004594: D3B1406A 1802D516
"  v_pk_mul_f32  [%[o_regs_68], %[o_regs_69]], v[22:23], [%[o_regs_68], %[o_regs_69]]         \n" // 00000000459C: D3B1406C 1802D916
"  v_pk_mul_f32  [%[o_regs_70], %[o_regs_71]], v[22:23], [%[o_regs_70], %[o_regs_71]]         \n" // 0000000045A4: D3B1406E 1802DD16
"  v_pk_mul_f32  [%[o_regs_72], %[o_regs_73]], v[22:23], [%[o_regs_72], %[o_regs_73]]         \n" // 0000000045AC: D3B14070 1802E116
"  v_pk_mul_f32  [%[o_regs_74], %[o_regs_75]], v[22:23], [%[o_regs_74], %[o_regs_75]]         \n" // 0000000045B4: D3B14072 1802E516
"  v_pk_mul_f32  [%[o_regs_76], %[o_regs_77]], v[22:23], [%[o_regs_76], %[o_regs_77]]         \n" // 0000000045BC: D3B14074 1802E916
"  v_pk_mul_f32  [%[o_regs_78], %[o_regs_79]], v[22:23], [%[o_regs_78], %[o_regs_79]]         \n" // 0000000045C4: D3B14076 1802ED16
"  v_pk_mul_f32  [%[o_regs_80], %[o_regs_81]], v[22:23], [%[o_regs_80], %[o_regs_81]]         \n" // 0000000045CC: D3B14078 1802F116
"  v_pk_mul_f32  [%[o_regs_82], %[o_regs_83]], v[22:23], [%[o_regs_82], %[o_regs_83]]         \n" // 0000000045D4: D3B1407A 1802F516
"  v_pk_mul_f32  [%[o_regs_84], %[o_regs_85]], v[22:23], [%[o_regs_84], %[o_regs_85]]         \n" // 0000000045DC: D3B1407C 1802F916
"  v_pk_mul_f32  [%[o_regs_86], %[o_regs_87]], v[22:23], [%[o_regs_86], %[o_regs_87]]         \n" // 0000000045E4: D3B1407E 1802FD16
"  v_pk_mul_f32  [%[o_regs_88], %[o_regs_89]], v[22:23], [%[o_regs_88], %[o_regs_89]]         \n" // 0000000045EC: D3B14080 18030116
"  v_pk_mul_f32  [%[o_regs_90], %[o_regs_91]], v[22:23], [%[o_regs_90], %[o_regs_91]]         \n" // 0000000045F4: D3B14082 18030516
"  v_pk_mul_f32  [%[o_regs_92], %[o_regs_93]], v[22:23], [%[o_regs_92], %[o_regs_93]]         \n" // 0000000045FC: D3B14084 18030916
"  v_pk_mul_f32  [%[o_regs_94], %[o_regs_95]], v[22:23], [%[o_regs_94], %[o_regs_95]]         \n" // 000000004604: D3B14086 18030D16
"  v_pk_mul_f32  [%[o_regs_96], %[o_regs_97]], v[22:23], [%[o_regs_96], %[o_regs_97]]         \n" // 00000000460C: D3B14088 18031116
"  v_pk_mul_f32  [%[o_regs_98], %[o_regs_99]], v[22:23], [%[o_regs_98], %[o_regs_99]]         \n" // 000000004614: D3B1408A 18031516
"  v_pk_mul_f32  [%[o_regs_100], %[o_regs_101]], v[22:23], [%[o_regs_100], %[o_regs_101]]         \n" // 00000000461C: D3B1408C 18031916
"  v_pk_mul_f32  [%[o_regs_102], %[o_regs_103]], v[22:23], [%[o_regs_102], %[o_regs_103]]         \n" // 000000004624: D3B1408E 18031D16
"  v_pk_mul_f32  [%[o_regs_104], %[o_regs_105]], v[22:23], [%[o_regs_104], %[o_regs_105]]         \n" // 00000000462C: D3B14090 18032116
"  v_pk_mul_f32  [%[o_regs_106], %[o_regs_107]], v[22:23], [%[o_regs_106], %[o_regs_107]]         \n" // 000000004634: D3B14092 18032516
"  v_pk_mul_f32  [%[o_regs_108], %[o_regs_109]], v[22:23], [%[o_regs_108], %[o_regs_109]]         \n" // 00000000463C: D3B14094 18032916
"  v_pk_mul_f32  [%[o_regs_110], %[o_regs_111]], v[22:23], [%[o_regs_110], %[o_regs_111]]         \n" // 000000004644: D3B14096 18032D16
"  v_pk_mul_f32  [%[o_regs_112], %[o_regs_113]], v[22:23], [%[o_regs_112], %[o_regs_113]]         \n" // 00000000464C: D3B14098 18033116
"  v_pk_mul_f32  [%[o_regs_114], %[o_regs_115]], v[22:23], [%[o_regs_114], %[o_regs_115]]         \n" // 000000004654: D3B1409A 18033516
"  v_pk_mul_f32  [%[o_regs_116], %[o_regs_117]], v[22:23], [%[o_regs_116], %[o_regs_117]]         \n" // 00000000465C: D3B1409C 18033916
"  v_pk_mul_f32  [%[o_regs_118], %[o_regs_119]], v[22:23], [%[o_regs_118], %[o_regs_119]]         \n" // 000000004664: D3B1409E 18033D16
"  v_pk_mul_f32  [%[o_regs_120], %[o_regs_121]], v[22:23], [%[o_regs_120], %[o_regs_121]]         \n" // 00000000466C: D3B140A0 18034116
"  v_pk_mul_f32  [%[o_regs_122], %[o_regs_123]], v[22:23], [%[o_regs_122], %[o_regs_123]]         \n" // 000000004674: D3B140A2 18034516
"  v_pk_mul_f32  [%[o_regs_124], %[o_regs_125]], v[22:23], [%[o_regs_124], %[o_regs_125]]         \n" // 00000000467C: D3B140A4 18034916
"  v_pk_mul_f32  [%[o_regs_126], %[o_regs_127]], v[22:23], [%[o_regs_126], %[o_regs_127]]         \n" // 000000004684: D3B140A6 18034D16
"  s_waitcnt     lgkmcnt(0)                               \n" // 00000000468C: BF8CC07F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]], acc[144:145], v[32:33], [%[o_regs_0], %[o_regs_1], %[o_regs_2], %[o_regs_3]]  \n" // 000000004690: D3E10028 0CA24190
"  ds_read_b128  acc[176:179], v7 offset:45312            \n" // 000000004698: DBFEB100 B0000007
"  ds_read_b128  acc[180:183], v7 offset:46336            \n" // 0000000046A0: DBFEB500 B4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]], acc[146:147], v[32:33], [%[o_regs_4], %[o_regs_5], %[o_regs_6], %[o_regs_7]]  \n" // 0000000046A8: D3E1002C 0CB24192
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]], acc[148:149], v[32:33], [%[o_regs_8], %[o_regs_9], %[o_regs_10], %[o_regs_11]]  \n" // 0000000046B0: D3E10030 0CC24194
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]], acc[150:151], v[32:33], [%[o_regs_12], %[o_regs_13], %[o_regs_14], %[o_regs_15]]  \n" // 0000000046B8: D3E10034 0CD24196
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]], acc[152:153], v[32:33], [%[o_regs_16], %[o_regs_17], %[o_regs_18], %[o_regs_19]]  \n" // 0000000046C0: D3E10038 0CE24198
"  ds_read_b128  acc[184:187], v7 offset:47360            \n" // 0000000046C8: DBFEB900 B8000007
"  ds_read_b128  acc[188:191], v7 offset:48384            \n" // 0000000046D0: DBFEBD00 BC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]], acc[154:155], v[32:33], [%[o_regs_20], %[o_regs_21], %[o_regs_22], %[o_regs_23]]  \n" // 0000000046D8: D3E1003C 0CF2419A
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]], acc[156:157], v[32:33], [%[o_regs_24], %[o_regs_25], %[o_regs_26], %[o_regs_27]]  \n" // 0000000046E0: D3E10040 0D02419C
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]], acc[158:159], v[32:33], [%[o_regs_28], %[o_regs_29], %[o_regs_30], %[o_regs_31]]  \n" // 0000000046E8: D3E10044 0D12419E
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]], acc[160:161], v[32:33], [%[o_regs_32], %[o_regs_33], %[o_regs_34], %[o_regs_35]]  \n" // 0000000046F0: D3E10048 0D2241A0
"  ds_read_b128  acc[192:195], v7 offset:49408            \n" // 0000000046F8: DBFEC100 C0000007
"  ds_read_b128  acc[196:199], v7 offset:50432            \n" // 000000004700: DBFEC500 C4000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]], acc[162:163], v[32:33], [%[o_regs_36], %[o_regs_37], %[o_regs_38], %[o_regs_39]]  \n" // 000000004708: D3E1004C 0D3241A2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]], acc[164:165], v[32:33], [%[o_regs_40], %[o_regs_41], %[o_regs_42], %[o_regs_43]]  \n" // 000000004710: D3E10050 0D4241A4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]], acc[166:167], v[32:33], [%[o_regs_44], %[o_regs_45], %[o_regs_46], %[o_regs_47]]  \n" // 000000004718: D3E10054 0D5241A6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]], acc[168:169], v[32:33], [%[o_regs_48], %[o_regs_49], %[o_regs_50], %[o_regs_51]]  \n" // 000000004720: D3E10058 0D6241A8
"  ds_read_b128  acc[200:203], v7 offset:51456            \n" // 000000004728: DBFEC900 C8000007
"  ds_read_b128  acc[204:207], v7 offset:52480            \n" // 000000004730: DBFECD00 CC000007
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]], acc[170:171], v[32:33], [%[o_regs_52], %[o_regs_53], %[o_regs_54], %[o_regs_55]]  \n" // 000000004738: D3E1005C 0D7241AA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]], acc[172:173], v[32:33], [%[o_regs_56], %[o_regs_57], %[o_regs_58], %[o_regs_59]]  \n" // 000000004740: D3E10060 0D8241AC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]], acc[174:175], v[32:33], [%[o_regs_60], %[o_regs_61], %[o_regs_62], %[o_regs_63]]  \n" // 000000004748: D3E10064 0D9241AE
"  s_waitcnt     lgkmcnt(4)                               \n" // 000000004750: BF8CC47F
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]], acc[176:177], v[32:33], [%[o_regs_64], %[o_regs_65], %[o_regs_66], %[o_regs_67]]  \n" // 000000004754: D3E10068 0DA241B0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]], acc[178:179], v[32:33], [%[o_regs_68], %[o_regs_69], %[o_regs_70], %[o_regs_71]]  \n" // 00000000475C: D3E1006C 0DB241B2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]], acc[180:181], v[32:33], [%[o_regs_72], %[o_regs_73], %[o_regs_74], %[o_regs_75]]  \n" // 000000004764: D3E10070 0DC241B4
"  s_waitcnt     vmcnt(18) & lgkmcnt(0)                   \n" // 00000000476C: BF8C4072
"  s_barrier                                              \n" // 000000004770: BF8A0000
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]], acc[182:183], v[32:33], [%[o_regs_76], %[o_regs_77], %[o_regs_78], %[o_regs_79]]  \n" // 000000004774: D3E10074 0DD241B6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]], acc[184:185], v[32:33], [%[o_regs_80], %[o_regs_81], %[o_regs_82], %[o_regs_83]]  \n" // 00000000477C: D3E10078 0DE241B8
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]], acc[186:187], v[32:33], [%[o_regs_84], %[o_regs_85], %[o_regs_86], %[o_regs_87]]  \n" // 000000004784: D3E1007C 0DF241BA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]], acc[188:189], v[32:33], [%[o_regs_88], %[o_regs_89], %[o_regs_90], %[o_regs_91]]  \n" // 00000000478C: D3E10080 0E0241BC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]], acc[190:191], v[32:33], [%[o_regs_92], %[o_regs_93], %[o_regs_94], %[o_regs_95]]  \n" // 000000004794: D3E10084 0E1241BE
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]], acc[192:193], v[32:33], [%[o_regs_96], %[o_regs_97], %[o_regs_98], %[o_regs_99]]  \n" // 00000000479C: D3E10088 0E2241C0
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]], acc[194:195], v[32:33], [%[o_regs_100], %[o_regs_101], %[o_regs_102], %[o_regs_103]]  \n" // 0000000047A4: D3E1008C 0E3241C2
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]], acc[196:197], v[32:33], [%[o_regs_104], %[o_regs_105], %[o_regs_106], %[o_regs_107]]  \n" // 0000000047AC: D3E10090 0E4241C4
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]], acc[198:199], v[32:33], [%[o_regs_108], %[o_regs_109], %[o_regs_110], %[o_regs_111]]  \n" // 0000000047B4: D3E10094 0E5241C6
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]], acc[200:201], v[32:33], [%[o_regs_112], %[o_regs_113], %[o_regs_114], %[o_regs_115]]  \n" // 0000000047BC: D3E10098 0E6241C8
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]], acc[202:203], v[32:33], [%[o_regs_116], %[o_regs_117], %[o_regs_118], %[o_regs_119]]  \n" // 0000000047C4: D3E1009C 0E7241CA
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]], acc[204:205], v[32:33], [%[o_regs_120], %[o_regs_121], %[o_regs_122], %[o_regs_123]]  \n" // 0000000047CC: D3E100A0 0E8241CC
"  v_mfma_f32_16x16x16_bf16  [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]], acc[206:207], v[32:33], [%[o_regs_124], %[o_regs_125], %[o_regs_126], %[o_regs_127]]  \n" // 0000000047D4: D3E100A4 0E9241CE
"  s_nop         0x0008                                   \n" // 0000000047DC: BF800008
"  s_branch      label_11F9%=                               \n" // 0000000047E0: BF820000
"label_11F9%=: \n"
"  ds_write_b32  v3, v14 offset:55552                     \n" // 0000000047E4: D81AD900 00000E03
"  ds_write_b32  v3, v15 offset:56576                     \n" // 0000000047EC: D81ADD00 00000F03
"  s_waitcnt     lgkmcnt(0)                               \n" // 0000000047F4: BF8CC07F
"  ds_read_b32   v20, v2 offset:55552                     \n" // 0000000047F8: D86CD900 14000002
"  ds_read_b32   v21, v2 offset:55616                     \n" // 000000004800: D86CD940 15000002
"  ds_read_b32   v22, v2 offset:55680                     \n" // 000000004808: D86CD980 16000002
"  ds_read_b32   v23, v2 offset:55744                     \n" // 000000004810: D86CD9C0 17000002
"  ds_read_b32   v24, v2 offset:56576                     \n" // 000000004818: D86CDD00 18000002
"  ds_read_b32   v25, v2 offset:56640                     \n" // 000000004820: D86CDD40 19000002
"  ds_read_b32   v26, v2 offset:56704                     \n" // 000000004828: D86CDD80 1A000002
"  ds_read_b32   v27, v2 offset:56768                     \n" // 000000004830: D86CDDC0 1B000002
"  s_waitcnt     lgkmcnt(0)                               \n" // 000000004838: BF8CC07F
"  v_mov_b32     v14, 0                                   \n" // 00000000483C: 7E1C0280
"  v_mov_b32     v15, 0                                   \n" // 000000004840: 7E1E0280
"  v_add_f32     v14, v20, v14                            \n" // 000000004844: 021C1D14
"  v_add_f32     v15, v24, v15                            \n" // 000000004848: 021E1F18
"  v_add_f32     v14, v21, v14                            \n" // 00000000484C: 021C1D15
"  v_add_f32     v15, v25, v15                            \n" // 000000004850: 021E1F19
"  v_add_f32     v14, v22, v14                            \n" // 000000004854: 021C1D16
"  v_add_f32     v15, v26, v15                            \n" // 000000004858: 021E1F1A
"  v_add_f32     v14, v23, v14                            \n" // 00000000485C: 021C1D17
"  v_add_f32     v15, v27, v15                            \n" // 000000004860: 021E1F1B
"  v_mov_b32     v20, 0                                   \n" // 000000004864: 7E280280
"  v_cmp_eq_u32  s[38:39], v20, v14                       \n" // 000000004868: D0CA0026 00021D14
"  v_cmp_eq_u32  s[40:41], v20, v15                       \n" // 000000004870: D0CA0028 00021F14
"  v_mul_f32     v20, v12, s64                            \n" // 000000004878: D1050014 0000810C
"  v_mul_f32     v22, v13, s64                            \n" // 000000004880: D1050016 0000810D
"  v_log_f32     v21, v14                                 \n" // 000000004888: 7E2A430E
"  v_log_f32     v23, v15                                 \n" // 00000000488C: 7E2E430F
"  v_cndmask_b32  v14, v14, 1.0, s[38:39]                 \n" // 000000004890: D100000E 0099E50E
"  v_cndmask_b32  v15, v15, 1.0, s[40:41]                 \n" // 000000004898: D100000F 00A1E50F
"  s_nop         0x0001                                   \n" // 0000000048A0: BF800001
"  v_rcp_f32     v14, v14                                 \n" // 0000000048A4: 7E1C450E
"  v_rcp_f32     v15, v15                                 \n" // 0000000048A8: 7E1E450F
"  s_nop         0x0001                                   \n" // 0000000048AC: BF800001
"  v_fma_f32     %[lse_result], v21, s63, v20                       \n" // 0000000048B0: D1CB0018 04507F15
"  v_fma_f32     v25, v23, s63, v22                       \n" // 0000000048B8: D1CB0019 04587F17
"  v_mul_f32     %[o_regs_0], v14, %[o_regs_0]                            \n" // 0000000048C0: 0A50510E
"  v_mul_f32     %[o_regs_1], v14, %[o_regs_1]                            \n" // 0000000048C4: 0A52530E
"  v_mul_f32     %[o_regs_2], v14, %[o_regs_2]                            \n" // 0000000048C8: 0A54550E
"  v_mul_f32     %[o_regs_3], v14, %[o_regs_3]                            \n" // 0000000048CC: 0A56570E
"  v_mul_f32     %[o_regs_4], v14, %[o_regs_4]                            \n" // 0000000048D0: 0A58590E
"  v_mul_f32     %[o_regs_5], v14, %[o_regs_5]                            \n" // 0000000048D4: 0A5A5B0E
"  v_mul_f32     %[o_regs_6], v14, %[o_regs_6]                            \n" // 0000000048D8: 0A5C5D0E
"  v_mul_f32     %[o_regs_7], v14, %[o_regs_7]                            \n" // 0000000048DC: 0A5E5F0E
"  v_mul_f32     %[o_regs_8], v14, %[o_regs_8]                            \n" // 0000000048E0: 0A60610E
"  v_mul_f32     %[o_regs_9], v14, %[o_regs_9]                            \n" // 0000000048E4: 0A62630E
"  v_mul_f32     %[o_regs_10], v14, %[o_regs_10]                            \n" // 0000000048E8: 0A64650E
"  v_mul_f32     %[o_regs_11], v14, %[o_regs_11]                            \n" // 0000000048EC: 0A66670E
"  v_mul_f32     %[o_regs_12], v14, %[o_regs_12]                            \n" // 0000000048F0: 0A68690E
"  v_mul_f32     %[o_regs_13], v14, %[o_regs_13]                            \n" // 0000000048F4: 0A6A6B0E
"  v_mul_f32     %[o_regs_14], v14, %[o_regs_14]                            \n" // 0000000048F8: 0A6C6D0E
"  v_mul_f32     %[o_regs_15], v14, %[o_regs_15]                            \n" // 0000000048FC: 0A6E6F0E
"  v_mul_f32     %[o_regs_16], v14, %[o_regs_16]                            \n" // 000000004900: 0A70710E
"  v_mul_f32     %[o_regs_17], v14, %[o_regs_17]                            \n" // 000000004904: 0A72730E
"  v_mul_f32     %[o_regs_18], v14, %[o_regs_18]                            \n" // 000000004908: 0A74750E
"  v_mul_f32     %[o_regs_19], v14, %[o_regs_19]                            \n" // 00000000490C: 0A76770E
"  v_mul_f32     %[o_regs_20], v14, %[o_regs_20]                            \n" // 000000004910: 0A78790E
"  v_mul_f32     %[o_regs_21], v14, %[o_regs_21]                            \n" // 000000004914: 0A7A7B0E
"  v_mul_f32     %[o_regs_22], v14, %[o_regs_22]                            \n" // 000000004918: 0A7C7D0E
"  v_mul_f32     %[o_regs_23], v14, %[o_regs_23]                            \n" // 00000000491C: 0A7E7F0E
"  v_mul_f32     %[o_regs_24], v14, %[o_regs_24]                            \n" // 000000004920: 0A80810E
"  v_mul_f32     %[o_regs_25], v14, %[o_regs_25]                            \n" // 000000004924: 0A82830E
"  v_mul_f32     %[o_regs_26], v14, %[o_regs_26]                            \n" // 000000004928: 0A84850E
"  v_mul_f32     %[o_regs_27], v14, %[o_regs_27]                            \n" // 00000000492C: 0A86870E
"  v_mul_f32     %[o_regs_28], v14, %[o_regs_28]                            \n" // 000000004930: 0A88890E
"  v_mul_f32     %[o_regs_29], v14, %[o_regs_29]                            \n" // 000000004934: 0A8A8B0E
"  v_mul_f32     %[o_regs_30], v14, %[o_regs_30]                            \n" // 000000004938: 0A8C8D0E
"  v_mul_f32     %[o_regs_31], v14, %[o_regs_31]                            \n" // 00000000493C: 0A8E8F0E
"  v_mul_f32     %[o_regs_32], v14, %[o_regs_32]                            \n" // 000000004940: 0A90910E
"  v_mul_f32     %[o_regs_33], v14, %[o_regs_33]                            \n" // 000000004944: 0A92930E
"  v_mul_f32     %[o_regs_34], v14, %[o_regs_34]                            \n" // 000000004948: 0A94950E
"  v_mul_f32     %[o_regs_35], v14, %[o_regs_35]                            \n" // 00000000494C: 0A96970E
"  v_mul_f32     %[o_regs_36], v14, %[o_regs_36]                            \n" // 000000004950: 0A98990E
"  v_mul_f32     %[o_regs_37], v14, %[o_regs_37]                            \n" // 000000004954: 0A9A9B0E
"  v_mul_f32     %[o_regs_38], v14, %[o_regs_38]                            \n" // 000000004958: 0A9C9D0E
"  v_mul_f32     %[o_regs_39], v14, %[o_regs_39]                            \n" // 00000000495C: 0A9E9F0E
"  v_mul_f32     %[o_regs_40], v14, %[o_regs_40]                            \n" // 000000004960: 0AA0A10E
"  v_mul_f32     %[o_regs_41], v14, %[o_regs_41]                            \n" // 000000004964: 0AA2A30E
"  v_mul_f32     %[o_regs_42], v14, %[o_regs_42]                            \n" // 000000004968: 0AA4A50E
"  v_mul_f32     %[o_regs_43], v14, %[o_regs_43]                            \n" // 00000000496C: 0AA6A70E
"  v_mul_f32     %[o_regs_44], v14, %[o_regs_44]                            \n" // 000000004970: 0AA8A90E
"  v_mul_f32     %[o_regs_45], v14, %[o_regs_45]                            \n" // 000000004974: 0AAAAB0E
"  v_mul_f32     %[o_regs_46], v14, %[o_regs_46]                            \n" // 000000004978: 0AACAD0E
"  v_mul_f32     %[o_regs_47], v14, %[o_regs_47]                            \n" // 00000000497C: 0AAEAF0E
"  v_mul_f32     %[o_regs_48], v14, %[o_regs_48]                            \n" // 000000004980: 0AB0B10E
"  v_mul_f32     %[o_regs_49], v14, %[o_regs_49]                            \n" // 000000004984: 0AB2B30E
"  v_mul_f32     %[o_regs_50], v14, %[o_regs_50]                            \n" // 000000004988: 0AB4B50E
"  v_mul_f32     %[o_regs_51], v14, %[o_regs_51]                            \n" // 00000000498C: 0AB6B70E
"  v_mul_f32     %[o_regs_52], v14, %[o_regs_52]                            \n" // 000000004990: 0AB8B90E
"  v_mul_f32     %[o_regs_53], v14, %[o_regs_53]                            \n" // 000000004994: 0ABABB0E
"  v_mul_f32     %[o_regs_54], v14, %[o_regs_54]                            \n" // 000000004998: 0ABCBD0E
"  v_mul_f32     %[o_regs_55], v14, %[o_regs_55]                            \n" // 00000000499C: 0ABEBF0E
"  v_mul_f32     %[o_regs_56], v14, %[o_regs_56]                            \n" // 0000000049A0: 0AC0C10E
"  v_mul_f32     %[o_regs_57], v14, %[o_regs_57]                            \n" // 0000000049A4: 0AC2C30E
"  v_mul_f32     %[o_regs_58], v14, %[o_regs_58]                            \n" // 0000000049A8: 0AC4C50E
"  v_mul_f32     %[o_regs_59], v14, %[o_regs_59]                            \n" // 0000000049AC: 0AC6C70E
"  v_mul_f32     %[o_regs_60], v14, %[o_regs_60]                          \n" // 0000000049B0: 0AC8C90E
"  v_mul_f32     %[o_regs_61], v14, %[o_regs_61]                          \n" // 0000000049B4: 0ACACB0E
"  v_mul_f32     %[o_regs_62], v14, %[o_regs_62]                          \n" // 0000000049B8: 0ACCCD0E
"  v_mul_f32     %[o_regs_63], v14, %[o_regs_63]                          \n" // 0000000049BC: 0ACECF0E
"  v_mul_f32     %[o_regs_64], v14, %[o_regs_64]                          \n" // 0000000049C0: 0AD0D10E
"  v_mul_f32     %[o_regs_65], v14, %[o_regs_65]                          \n" // 0000000049C4: 0AD2D30E
"  v_mul_f32     %[o_regs_66], v14, %[o_regs_66]                          \n" // 0000000049C8: 0AD4D50E
"  v_mul_f32     %[o_regs_67], v14, %[o_regs_67]                          \n" // 0000000049CC: 0AD6D70E
"  v_mul_f32     %[o_regs_68], v14, %[o_regs_68]                          \n" // 0000000049D0: 0AD8D90E
"  v_mul_f32     %[o_regs_69], v14, %[o_regs_69]                          \n" // 0000000049D4: 0ADADB0E
"  v_mul_f32     %[o_regs_70], v14, %[o_regs_70]                          \n" // 0000000049D8: 0ADCDD0E
"  v_mul_f32     %[o_regs_71], v14, %[o_regs_71]                          \n" // 0000000049DC: 0ADEDF0E
"  v_mul_f32     %[o_regs_72], v14, %[o_regs_72]                          \n" // 0000000049E0: 0AE0E10E
"  v_mul_f32     %[o_regs_73], v14, %[o_regs_73]                          \n" // 0000000049E4: 0AE2E30E
"  v_mul_f32     %[o_regs_74], v14, %[o_regs_74]                          \n" // 0000000049E8: 0AE4E50E
"  v_mul_f32     %[o_regs_75], v14, %[o_regs_75]                          \n" // 0000000049EC: 0AE6E70E
"  v_mul_f32     %[o_regs_76], v14, %[o_regs_76]                          \n" // 0000000049F0: 0AE8E90E
"  v_mul_f32     %[o_regs_77], v14, %[o_regs_77]                          \n" // 0000000049F4: 0AEAEB0E
"  v_mul_f32     %[o_regs_78], v14, %[o_regs_78]                          \n" // 0000000049F8: 0AECED0E
"  v_mul_f32     %[o_regs_79], v14, %[o_regs_79]                          \n" // 0000000049FC: 0AEEEF0E
"  v_mul_f32     %[o_regs_80], v14, %[o_regs_80]                          \n" // 000000004A00: 0AF0F10E
"  v_mul_f32     %[o_regs_81], v14, %[o_regs_81]                          \n" // 000000004A04: 0AF2F30E
"  v_mul_f32     %[o_regs_82], v14, %[o_regs_82]                          \n" // 000000004A08: 0AF4F50E
"  v_mul_f32     %[o_regs_83], v14, %[o_regs_83]                          \n" // 000000004A0C: 0AF6F70E
"  v_mul_f32     %[o_regs_84], v14, %[o_regs_84]                          \n" // 000000004A10: 0AF8F90E
"  v_mul_f32     %[o_regs_85], v14, %[o_regs_85]                          \n" // 000000004A14: 0AFAFB0E
"  v_mul_f32     %[o_regs_86], v14, %[o_regs_86]                          \n" // 000000004A18: 0AFCFD0E
"  v_mul_f32     %[o_regs_87], v14, %[o_regs_87]                          \n" // 000000004A1C: 0AFEFF0E
"  v_mul_f32     %[o_regs_88], v14, %[o_regs_88]                          \n" // 000000004A20: 0B01010E
"  v_mul_f32     %[o_regs_89], v14, %[o_regs_89]                          \n" // 000000004A24: 0B03030E
"  v_mul_f32     %[o_regs_90], v14, %[o_regs_90]                          \n" // 000000004A28: 0B05050E
"  v_mul_f32     %[o_regs_91], v14, %[o_regs_91]                          \n" // 000000004A2C: 0B07070E
"  v_mul_f32     %[o_regs_92], v14, %[o_regs_92]                          \n" // 000000004A30: 0B09090E
"  v_mul_f32     %[o_regs_93], v14, %[o_regs_93]                          \n" // 000000004A34: 0B0B0B0E
"  v_mul_f32     %[o_regs_94], v14, %[o_regs_94]                          \n" // 000000004A38: 0B0D0D0E
"  v_mul_f32     %[o_regs_95], v14, %[o_regs_95]                          \n" // 000000004A3C: 0B0F0F0E
"  v_mul_f32     %[o_regs_96], v14, %[o_regs_96]                          \n" // 000000004A40: 0B11110E
"  v_mul_f32     %[o_regs_97], v14, %[o_regs_97]                          \n" // 000000004A44: 0B13130E
"  v_mul_f32     %[o_regs_98], v14, %[o_regs_98]                          \n" // 000000004A48: 0B15150E
"  v_mul_f32     %[o_regs_99], v14, %[o_regs_99]                          \n" // 000000004A4C: 0B17170E
"  v_mul_f32     %[o_regs_100], v14, %[o_regs_100]                          \n" // 000000004A50: 0B19190E
"  v_mul_f32     %[o_regs_101], v14, %[o_regs_101]                          \n" // 000000004A54: 0B1B1B0E
"  v_mul_f32     %[o_regs_102], v14, %[o_regs_102]                          \n" // 000000004A58: 0B1D1D0E
"  v_mul_f32     %[o_regs_103], v14, %[o_regs_103]                          \n" // 000000004A5C: 0B1F1F0E
"  v_mul_f32     %[o_regs_104], v14, %[o_regs_104]                          \n" // 000000004A60: 0B21210E
"  v_mul_f32     %[o_regs_105], v14, %[o_regs_105]                          \n" // 000000004A64: 0B23230E
"  v_mul_f32     %[o_regs_106], v14, %[o_regs_106]                          \n" // 000000004A68: 0B25250E
"  v_mul_f32     %[o_regs_107], v14, %[o_regs_107]                          \n" // 000000004A6C: 0B27270E
"  v_mul_f32     %[o_regs_108], v14, %[o_regs_108]                          \n" // 000000004A70: 0B29290E
"  v_mul_f32     %[o_regs_109], v14, %[o_regs_109]                          \n" // 000000004A74: 0B2B2B0E
"  v_mul_f32     %[o_regs_110], v14, %[o_regs_110]                          \n" // 000000004A78: 0B2D2D0E
"  v_mul_f32     %[o_regs_111], v14, %[o_regs_111]                          \n" // 000000004A7C: 0B2F2F0E
"  v_mul_f32     %[o_regs_112], v14, %[o_regs_112]                          \n" // 000000004A80: 0B31310E
"  v_mul_f32     %[o_regs_113], v14, %[o_regs_113]                          \n" // 000000004A84: 0B33330E
"  v_mul_f32     %[o_regs_114], v14, %[o_regs_114]                          \n" // 000000004A88: 0B35350E
"  v_mul_f32     %[o_regs_115], v14, %[o_regs_115]                          \n" // 000000004A8C: 0B37370E
"  v_mul_f32     %[o_regs_116], v14, %[o_regs_116]                          \n" // 000000004A90: 0B39390E
"  v_mul_f32     %[o_regs_117], v14, %[o_regs_117]                          \n" // 000000004A94: 0B3B3B0E
"  v_mul_f32     %[o_regs_118], v14, %[o_regs_118]                          \n" // 000000004A98: 0B3D3D0E
"  v_mul_f32     %[o_regs_119], v14, %[o_regs_119]                          \n" // 000000004A9C: 0B3F3F0E
"  v_mul_f32     %[o_regs_120], v14, %[o_regs_120]                          \n" // 000000004AA0: 0B41410E
"  v_mul_f32     %[o_regs_121], v14, %[o_regs_121]                          \n" // 000000004AA4: 0B43430E
"  v_mul_f32     %[o_regs_122], v14, %[o_regs_122]                          \n" // 000000004AA8: 0B45450E
"  v_mul_f32     %[o_regs_123], v14, %[o_regs_123]                          \n" // 000000004AAC: 0B47470E
"  v_mul_f32     %[o_regs_124], v14, %[o_regs_124]                          \n" // 000000004AB0: 0B49490E
"  v_mul_f32     %[o_regs_125], v14, %[o_regs_125]                          \n" // 000000004AB4: 0B4B4B0E
"  v_mul_f32     %[o_regs_126], v14, %[o_regs_126]                          \n" // 000000004AB8: 0B4D4D0E
"  v_mul_f32     %[o_regs_127], v14, %[o_regs_127]                          \n" // 000000004ABC: 0B4F4F0E
"  ;s_mov_b32     %[num_splits], %[num_splits]             \n" // 00000000206C: BEFC0022
"  s_mov_b32     %[max_seqlen_q], %[max_seqlen_q]         \n" // 00000000206C: BEFC0022
"  ;s_cmp_le_u32  %[num_splits], 1                                   \n" // 000000004AC0: BF0B8143
"  ;s_cbranch_scc0  label_17B3_%=                             \n" // 000000004AC4: BF840501
"  ;s_cbranch_scc0  label_1973_%=                             \n" // 000000004AC4: BF840501
"  s_branch      label_1973_%=                               \n" // 0000000024AC: BF82FB37
"label_1971_%=: \n"
"  s_waitcnt     0x0000                                   \n" // 0000000065C4: BF8C0000
"	s_endpgm \n"
"label_1973_%=: \n"
"  s_waitcnt     0x0000                                   \n" // 0000000065C4: BF8C0000

