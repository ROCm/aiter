
/data/yanguahe/code/wk_sp1/aiter/hsa/gfx1250/fmha_v3_bwd/bwd_hd128_odo_bf16.co:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001900 <_ZN5aiter23fmha_bwd_hd128_odo_bf16E>:
	s_version UC_VERSION_GFX12|UC_VERSION_W32_BIT              // 000000001900: B0804009
	s_and_b32 s1, s1, 0xffff                                   // 000000001904: 8B01FF01 0000FFFF
	s_mov_b32 s48, s4                                          // 00000000190C: BEB00004
	s_mov_b32 s49, s5                                          // 000000001910: BEB10005
	s_mov_b32 s40, s2                                          // 000000001914: BEA80002
	s_mov_b32 s41, s3                                          // 000000001918: BEA90003
	s_mov_b32 s44, s6                                          // 00000000191C: BEAC0006
	s_mov_b32 s45, s7                                          // 000000001920: BEAD0007
	v_and_b32_e32 v0, 31, v0                                   // 000000001924: 3600009F
	s_mov_b32 s25, ttmp9                                       // 000000001928: BE990075
	s_and_b32 s26, ttmp7, 0xffff                               // 00000000192C: 8B1AFF73 0000FFFF
	s_bfe_u32 s27, ttmp7, 0x100010                             // 000000001934: 931BFF73 00100010
	s_bfe_u32 s24, ttmp8, 0x50019                              // 00000000193C: 9318FF74 00050019
	s_and_b32 s41, s41, 0x1fffff                               // 000000001944: 8B29FF29 001FFFFF
	s_and_b32 s41, s41, 0x1fffffff                             // 00000000194C: 8B29FF29 1FFFFFFF
	s_mov_b32 s42, 0                                           // 000000001954: BEAA0080
	s_mov_b32 s43, 0x4000                                      // 000000001958: BEAB00FF 00004000
	s_and_b32 s49, s49, 0x1fffff                               // 000000001960: 8B31FF31 001FFFFF
	s_and_b32 s49, s49, 0x1fffffff                             // 000000001968: 8B31FF31 1FFFFFFF
	s_mov_b32 s50, 0                                           // 000000001970: BEB20080
	s_mov_b32 s51, 0x4000                                      // 000000001974: BEB300FF 00004000
	s_and_b32 s45, s45, 0x1fffff                               // 00000000197C: 8B2DFF2D 001FFFFF
	s_and_b32 s45, s45, 0x1fffffff                             // 000000001984: 8B2DFF2D 1FFFFFFF
	s_mov_b32 s46, 0                                           // 00000000198C: BEAE0080
	s_mov_b32 s47, 0x4000                                      // 000000001990: BEAF00FF 00004000
	v_mov_b32_e32 v35, 0                                       // 000000001998: 7E460280
	s_mul_i32 s28, s10, 0x80                                   // 00000000199C: 961CFF0A 00000080
	s_mul_i32 s28, s25, s28                                    // 0000000019A4: 961C1C19
	s_mul_i32 s29, s26, s8                                     // 0000000019A8: 961D081A
	s_mul_i32 s30, s27, s9                                     // 0000000019AC: 961E091B
	s_add_co_u32 s31, s28, s29                                 // 0000000019B0: 801F1D1C
	s_add_co_u32 s31, s31, s30                                 // 0000000019B4: 801F1E1F
	s_mul_i32 s29, s17, s10                                    // 0000000019B8: 961D0A11
	s_sub_co_i32 s29, s29, s28                                 // 0000000019BC: 819D1C1D
	s_mov_b32 s42, s29                                         // 0000000019C0: BEAA001D
	s_add_co_u32 s40, s40, s31                                 // 0000000019C4: 80281F28
	s_add_co_ci_u32 s41, s41, 0                                // 0000000019C8: 82298029
	s_mul_i32 s28, s13, 0x80                                   // 0000000019CC: 961CFF0D 00000080
	s_mul_i32 s28, s25, s28                                    // 0000000019D4: 961C1C19
	s_mul_i32 s29, s26, s11                                    // 0000000019D8: 961D0B1A
	s_mul_i32 s30, s27, s12                                    // 0000000019DC: 961E0C1B
	s_add_co_u32 s32, s28, s29                                 // 0000000019E0: 80201D1C
	s_add_co_u32 s32, s32, s30                                 // 0000000019E4: 80201E20
	s_mul_i32 s29, s17, s13                                    // 0000000019E8: 961D0D11
	s_sub_co_i32 s29, s29, s28                                 // 0000000019EC: 819D1C1D
	s_mov_b32 s50, s29                                         // 0000000019F0: BEB2001D
	s_add_co_u32 s48, s48, s32                                 // 0000000019F4: 80302030
	s_add_co_ci_u32 s49, s49, 0                                // 0000000019F8: 82318031
	s_mul_i32 s28, s16, 0x80                                   // 0000000019FC: 961CFF10 00000080
	s_mul_i32 s28, s25, s28                                    // 000000001A04: 961C1C19
	s_mul_i32 s29, s26, s14                                    // 000000001A08: 961D0E1A
	s_mul_i32 s30, s27, s15                                    // 000000001A0C: 961E0F1B
	s_add_co_u32 s33, s28, s29                                 // 000000001A10: 80211D1C
	s_add_co_u32 s33, s33, s30                                 // 000000001A14: 80211E21
	s_mul_i32 s29, s17, s16                                    // 000000001A18: 961D1011
	s_sub_co_i32 s29, s29, s28                                 // 000000001A1C: 819D1C1D
	s_mov_b32 s46, s29                                         // 000000001A20: BEAE001D
	s_add_co_u32 s44, s44, s33                                 // 000000001A24: 802C212C
	s_add_co_ci_u32 s45, s45, 0                                // 000000001A28: 822D802D
	s_mov_b32 s35, s10                                         // 000000001A2C: BEA3000A
	v_lshrrev_b32_e32 v24, 3, v0                               // 000000001A30: 32300083
	v_mul_u32_u24_e32 v24, s35, v24                            // 000000001A34: 16303023
	v_and_b32_e32 v25, 7, v0                                   // 000000001A38: 36320087
	v_mul_u32_u24_e32 v25, 16, v25                             // 000000001A3C: 16323290
	v_add_nc_u32_e32 v25, v24, v25                             // 000000001A40: 4A323318
	v_mov_b32_e32 v4, v25                                      // 000000001A44: 7E080319
	s_mul_i32 s28, 32, s35                                     // 000000001A48: 961C23A0
	s_mul_i32 s28, s24, s28                                    // 000000001A4C: 961C1C18
	v_add_nc_u32_e32 v4, s28, v4                               // 000000001A50: 4A08081C
	s_mov_b32 s35, s13                                         // 000000001A54: BEA3000D
	v_lshrrev_b32_e32 v24, 3, v0                               // 000000001A58: 32300083
	v_mul_u32_u24_e32 v24, s35, v24                            // 000000001A5C: 16303023
	v_and_b32_e32 v25, 7, v0                                   // 000000001A60: 36320087
	v_mul_u32_u24_e32 v25, 16, v25                             // 000000001A64: 16323290
	v_add_nc_u32_e32 v25, v24, v25                             // 000000001A68: 4A323318
	v_mov_b32_e32 v33, v25                                     // 000000001A6C: 7E420319
	s_mul_i32 s28, 32, s35                                     // 000000001A70: 961C23A0
	s_mul_i32 s28, s24, s28                                    // 000000001A74: 961C1C18
	v_add_nc_u32_e32 v33, s28, v33                             // 000000001A78: 4A42421C
	v_lshrrev_b32_e32 v5, 3, v0                                // 000000001A7C: 320A0083
	v_and_b32_e64 v24, v0, 7                                   // 000000001A80: D51B0018 00010F00
	v_mul_u32_u24_e32 v24, 4, v24                              // 000000001A88: 16303084
	v_add_nc_u32_e32 v5, v5, v24                               // 000000001A8C: 4A0A3105
	v_mul_u32_u24_e32 v5, s16, v5                              // 000000001A90: 160A0A10
	s_mul_i32 s28, 32, s16                                     // 000000001A94: 961C10A0
	s_mul_i32 s28, s24, s28                                    // 000000001A98: 961C1C18
	v_add_nc_u32_e32 v5, s28, v5                               // 000000001A9C: 4A0A0A1C
	v_and_b32_e32 v24, 7, v0                                   // 000000001AA0: 36300087
	v_lshlrev_b32_e32 v24, 3, v24                              // 000000001AA4: 30303083
	s_sub_co_i32 s28, s18, 64                                  // 000000001AA8: 819CC012
	v_cmp_lt_u32_e64 s34, v24, s28                             // 000000001AAC: D4490022 00003918
	v_mov_b32_e32 v22, 0                                       // 000000001AB4: 7E2C0280
	s_mul_i32 s28, s10, 0                                      // 000000001AB8: 961C800A
	s_lshl_b32 s28, s28, 2                                     // 000000001ABC: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000001AC0: 4A2E081C
	s_mul_i32 s28, s13, 0                                      // 000000001AC4: 961C800D
	s_lshl_b32 s28, s28, 2                                     // 000000001AC8: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000001ACC: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000001AD0: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 000000001ADC: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 000000001AE8: 36400087
	v_cmp_eq_u32_e64 s39, v32, 0                               // 000000001AEC: D44A0027 00010120
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000001AF4: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000001AFC: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 000000001B04: BFC00000
	v_mov_b32_e32 v18, 0                                       // 000000001B08: 7E240280
	v_mov_b32_e32 v10, 0                                       // 000000001B0C: 7E140280
	v_mov_b32_e32 v19, 0                                       // 000000001B10: 7E260280
	v_mov_b32_e32 v11, 0                                       // 000000001B14: 7E160280
	v_mov_b32_e32 v20, 0                                       // 000000001B18: 7E280280
	v_mov_b32_e32 v12, 0                                       // 000000001B1C: 7E180280
	v_mov_b32_e32 v21, 0                                       // 000000001B20: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 000000001B24: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 000000001B28: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 000000001B2C: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 000000001B38: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 000000001B44: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000001B48: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000001B50: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 000000001B58: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 000000001B5C: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 000000001B64: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 000000001B68: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 000000001B70: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001B74: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001B78: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001B7C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001B80: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001B84: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001B8C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000001B94: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000001B9C: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000001BA0: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000001BA8: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001BAC: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001BB0: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001BB4: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001BB8: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001BBC: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001BC4: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000001BCC: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000001BD4: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000001BD8: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 000000001BE0: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001BE4: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001BE8: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001BEC: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001BF0: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001BF4: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001BFC: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 000000001C04: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 000000001C0C: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 000000001C10: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 000000001C18: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001C1C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001C20: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001C24: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001C28: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001C2C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001C34: D613001C 04723719
	s_nop 1                                                    // 000000001C3C: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000001C40: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000001C48: 06383918
	s_nop 1                                                    // 000000001C4C: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000001C50: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000001C58: 06383918
	s_nop 1                                                    // 000000001C5C: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001C60: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000001C68: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000001C6C: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000001C70: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000001C74: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000001C78: BFC00000
	v_mov_b32_e32 v28, 0                                       // 000000001C7C: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 000000001C80: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000001C88: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 000000001C8C: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 000000001C94: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001C98: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001C9C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001CA0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001CA4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001CA8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001CB0: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000001CB8: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 000000001CC0: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000001CC4: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000001CCC: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001CD0: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001CD4: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001CD8: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001CDC: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001CE0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001CE8: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 000000001CF0: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 000000001CF8: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 000000001CFC: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 000000001D04: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001D08: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001D0C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001D10: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001D14: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001D18: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001D20: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 000000001D28: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 000000001D30: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 000000001D34: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 000000001D3C: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001D40: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001D44: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001D48: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001D4C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001D50: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001D58: D613001C 04723719
	s_nop 1                                                    // 000000001D60: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000001D64: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000001D6C: 06383918
	s_nop 1                                                    // 000000001D70: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000001D74: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000001D7C: 06383918
	s_nop 1                                                    // 000000001D80: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001D84: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000001D8C: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000001D90: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000001D94: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000001D98: BEFE00C1
	s_mul_i32 s28, s10, 1                                      // 000000001D9C: 961C810A
	s_lshl_b32 s28, s28, 2                                     // 000000001DA0: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000001DA4: 4A2E081C
	s_mul_i32 s28, s13, 1                                      // 000000001DA8: 961C810D
	s_lshl_b32 s28, s28, 2                                     // 000000001DAC: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000001DB0: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000001DB4: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 000000001DC0: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 000000001DCC: 36400087
	v_cmp_eq_u32_e64 s39, v32, 1                               // 000000001DD0: D44A0027 00010320
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000001DD8: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000001DE0: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 000000001DE8: BFC00000
	v_mov_b32_e32 v18, 0                                       // 000000001DEC: 7E240280
	v_mov_b32_e32 v10, 0                                       // 000000001DF0: 7E140280
	v_mov_b32_e32 v19, 0                                       // 000000001DF4: 7E260280
	v_mov_b32_e32 v11, 0                                       // 000000001DF8: 7E160280
	v_mov_b32_e32 v20, 0                                       // 000000001DFC: 7E280280
	v_mov_b32_e32 v12, 0                                       // 000000001E00: 7E180280
	v_mov_b32_e32 v21, 0                                       // 000000001E04: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 000000001E08: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 000000001E0C: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 000000001E10: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 000000001E1C: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 000000001E28: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000001E2C: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000001E34: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 000000001E3C: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 000000001E40: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 000000001E48: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 000000001E4C: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 000000001E54: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001E58: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001E5C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001E60: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001E64: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001E68: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001E70: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000001E78: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000001E80: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000001E84: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000001E8C: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001E90: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001E94: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001E98: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001E9C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001EA0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001EA8: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000001EB0: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000001EB8: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000001EBC: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 000000001EC4: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001EC8: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001ECC: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001ED0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001ED4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001ED8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001EE0: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 000000001EE8: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 000000001EF0: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 000000001EF4: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 000000001EFC: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001F00: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001F04: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001F08: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001F0C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001F10: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001F18: D613001C 04723719
	s_nop 1                                                    // 000000001F20: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000001F24: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000001F2C: 06383918
	s_nop 1                                                    // 000000001F30: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000001F34: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000001F3C: 06383918
	s_nop 1                                                    // 000000001F40: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000001F44: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000001F4C: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000001F50: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000001F54: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000001F58: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000001F5C: BFC00000
	v_mov_b32_e32 v28, 0                                       // 000000001F60: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 000000001F64: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000001F6C: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 000000001F70: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 000000001F78: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001F7C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001F80: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001F84: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001F88: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001F8C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001F94: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000001F9C: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 000000001FA4: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000001FA8: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000001FB0: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001FB4: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001FB8: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001FBC: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001FC0: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001FC4: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000001FCC: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 000000001FD4: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 000000001FDC: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 000000001FE0: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 000000001FE8: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 000000001FEC: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000001FF0: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000001FF4: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000001FF8: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000001FFC: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002004: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 00000000200C: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 000000002014: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 000000002018: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 000000002020: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002024: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002028: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 00000000202C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002030: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002034: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 00000000203C: D613001C 04723719
	s_nop 1                                                    // 000000002044: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002048: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002050: 06383918
	s_nop 1                                                    // 000000002054: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002058: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002060: 06383918
	s_nop 1                                                    // 000000002064: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002068: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000002070: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002074: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002078: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 00000000207C: BEFE00C1
	s_mul_i32 s28, s10, 2                                      // 000000002080: 961C820A
	s_lshl_b32 s28, s28, 2                                     // 000000002084: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000002088: 4A2E081C
	s_mul_i32 s28, s13, 2                                      // 00000000208C: 961C820D
	s_lshl_b32 s28, s28, 2                                     // 000000002090: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000002094: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000002098: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 0000000020A4: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 0000000020B0: 36400087
	v_cmp_eq_u32_e64 s39, v32, 2                               // 0000000020B4: D44A0027 00010520
	v_add_nc_u32_e32 v23, 0x80, v23                            // 0000000020BC: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 0000000020C4: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 0000000020CC: BFC00000
	v_mov_b32_e32 v18, 0                                       // 0000000020D0: 7E240280
	v_mov_b32_e32 v10, 0                                       // 0000000020D4: 7E140280
	v_mov_b32_e32 v19, 0                                       // 0000000020D8: 7E260280
	v_mov_b32_e32 v11, 0                                       // 0000000020DC: 7E160280
	v_mov_b32_e32 v20, 0                                       // 0000000020E0: 7E280280
	v_mov_b32_e32 v12, 0                                       // 0000000020E4: 7E180280
	v_mov_b32_e32 v21, 0                                       // 0000000020E8: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 0000000020EC: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 0000000020F0: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 0000000020F4: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 000000002100: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 00000000210C: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002110: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000002118: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 000000002120: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 000000002124: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 00000000212C: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 000000002130: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 000000002138: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 00000000213C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002140: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002144: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002148: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 00000000214C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002154: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 00000000215C: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000002164: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000002168: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000002170: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002174: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002178: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 00000000217C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002180: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002184: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 00000000218C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000002194: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 00000000219C: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 0000000021A0: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 0000000021A8: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000021AC: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000021B0: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000021B4: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000021B8: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000021BC: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000021C4: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 0000000021CC: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 0000000021D4: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 0000000021D8: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 0000000021E0: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000021E4: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000021E8: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000021EC: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000021F0: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000021F4: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000021FC: D613001C 04723719
	s_nop 1                                                    // 000000002204: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002208: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002210: 06383918
	s_nop 1                                                    // 000000002214: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002218: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002220: 06383918
	s_nop 1                                                    // 000000002224: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002228: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000002230: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002234: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002238: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 00000000223C: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000002240: BFC00000
	v_mov_b32_e32 v28, 0                                       // 000000002244: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 000000002248: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000002250: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 000000002254: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 00000000225C: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002260: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002264: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002268: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 00000000226C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002270: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002278: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000002280: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 000000002288: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 00000000228C: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000002294: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002298: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 00000000229C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000022A0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000022A4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000022A8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000022B0: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 0000000022B8: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 0000000022C0: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 0000000022C4: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 0000000022CC: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000022D0: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000022D4: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000022D8: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000022DC: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000022E0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000022E8: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 0000000022F0: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 0000000022F8: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 0000000022FC: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 000000002304: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002308: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 00000000230C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002310: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002314: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002318: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002320: D613001C 04723719
	s_nop 1                                                    // 000000002328: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 00000000232C: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002334: 06383918
	s_nop 1                                                    // 000000002338: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 00000000233C: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002344: 06383918
	s_nop 1                                                    // 000000002348: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000234C: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000002354: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002358: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 00000000235C: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002360: BEFE00C1
	s_mul_i32 s28, s10, 3                                      // 000000002364: 961C830A
	s_lshl_b32 s28, s28, 2                                     // 000000002368: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 00000000236C: 4A2E081C
	s_mul_i32 s28, s13, 3                                      // 000000002370: 961C830D
	s_lshl_b32 s28, s28, 2                                     // 000000002374: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000002378: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 00000000237C: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 000000002388: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 000000002394: 36400087
	v_cmp_eq_u32_e64 s39, v32, 3                               // 000000002398: D44A0027 00010720
	v_add_nc_u32_e32 v23, 0x80, v23                            // 0000000023A0: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 0000000023A8: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 0000000023B0: BFC00000
	v_mov_b32_e32 v18, 0                                       // 0000000023B4: 7E240280
	v_mov_b32_e32 v10, 0                                       // 0000000023B8: 7E140280
	v_mov_b32_e32 v19, 0                                       // 0000000023BC: 7E260280
	v_mov_b32_e32 v11, 0                                       // 0000000023C0: 7E160280
	v_mov_b32_e32 v20, 0                                       // 0000000023C4: 7E280280
	v_mov_b32_e32 v12, 0                                       // 0000000023C8: 7E180280
	v_mov_b32_e32 v21, 0                                       // 0000000023CC: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 0000000023D0: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 0000000023D4: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 0000000023D8: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 0000000023E4: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 0000000023F0: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 0000000023F4: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 0000000023FC: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 000000002404: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 000000002408: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 000000002410: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 000000002414: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 00000000241C: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002420: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002424: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002428: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 00000000242C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002430: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002438: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000002440: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000002448: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 00000000244C: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000002454: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002458: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 00000000245C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002460: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002464: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002468: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002470: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000002478: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000002480: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000002484: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 00000000248C: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002490: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002494: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002498: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 00000000249C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000024A0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000024A8: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 0000000024B0: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 0000000024B8: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 0000000024BC: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 0000000024C4: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000024C8: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000024CC: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000024D0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000024D4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000024D8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000024E0: D613001C 04723719
	s_nop 1                                                    // 0000000024E8: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 0000000024EC: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 0000000024F4: 06383918
	s_nop 1                                                    // 0000000024F8: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 0000000024FC: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002504: 06383918
	s_nop 1                                                    // 000000002508: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000250C: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000002514: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002518: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 00000000251C: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002520: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000002524: BFC00000
	v_mov_b32_e32 v28, 0                                       // 000000002528: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 00000000252C: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000002534: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 000000002538: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 000000002540: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002544: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002548: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 00000000254C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002550: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002554: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 00000000255C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000002564: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 00000000256C: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000002570: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000002578: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 00000000257C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002580: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002584: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002588: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 00000000258C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002594: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 00000000259C: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 0000000025A4: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 0000000025A8: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 0000000025B0: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000025B4: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000025B8: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000025BC: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000025C0: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000025C4: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000025CC: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 0000000025D4: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 0000000025DC: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 0000000025E0: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 0000000025E8: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000025EC: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000025F0: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000025F4: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000025F8: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000025FC: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002604: D613001C 04723719
	s_nop 1                                                    // 00000000260C: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002610: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002618: 06383918
	s_nop 1                                                    // 00000000261C: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002620: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002628: 06383918
	s_nop 1                                                    // 00000000262C: BF800001
	v_mov_b32_dpp v24, v28 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002630: 7E3002FA FF09041C
	s_mov_b32 exec_lo, s39                                     // 000000002638: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 00000000263C: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002640: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002644: BEFE00C1
	s_mul_i32 s28, s10, 4                                      // 000000002648: 961C840A
	s_lshl_b32 s28, s28, 2                                     // 00000000264C: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000002650: 4A2E081C
	s_mul_i32 s28, s13, 4                                      // 000000002654: 961C840D
	s_lshl_b32 s28, s28, 2                                     // 000000002658: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 00000000265C: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000002660: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 00000000266C: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 000000002678: 36400087
	v_cmp_eq_u32_e64 s39, v32, 4                               // 00000000267C: D44A0027 00010920
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002684: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 00000000268C: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 000000002694: BFC00000
	v_mov_b32_e32 v18, 0                                       // 000000002698: 7E240280
	v_mov_b32_e32 v10, 0                                       // 00000000269C: 7E140280
	v_mov_b32_e32 v19, 0                                       // 0000000026A0: 7E260280
	v_mov_b32_e32 v11, 0                                       // 0000000026A4: 7E160280
	v_mov_b32_e32 v20, 0                                       // 0000000026A8: 7E280280
	v_mov_b32_e32 v12, 0                                       // 0000000026AC: 7E180280
	v_mov_b32_e32 v21, 0                                       // 0000000026B0: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 0000000026B4: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 0000000026B8: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 0000000026BC: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 0000000026C8: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 0000000026D4: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 0000000026D8: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 0000000026E0: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 0000000026E8: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 0000000026EC: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 0000000026F4: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 0000000026F8: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 000000002700: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002704: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002708: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 00000000270C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002710: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002714: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 00000000271C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000002724: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 00000000272C: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000002730: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000002738: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 00000000273C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002740: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002744: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002748: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 00000000274C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002754: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 00000000275C: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000002764: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000002768: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 000000002770: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002774: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002778: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 00000000277C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002780: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002784: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 00000000278C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 000000002794: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 00000000279C: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 0000000027A0: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 0000000027A8: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000027AC: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000027B0: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000027B4: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000027B8: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000027BC: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000027C4: D613001C 04723719
	s_nop 1                                                    // 0000000027CC: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 0000000027D0: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 0000000027D8: 06383918
	s_nop 1                                                    // 0000000027DC: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 0000000027E0: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 0000000027E8: 06383918
	s_nop 1                                                    // 0000000027EC: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000027F0: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 0000000027F8: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 0000000027FC: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002800: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002804: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000002808: BFC00000
	v_mov_b32_e32 v28, 0                                       // 00000000280C: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 000000002810: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000002818: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 00000000281C: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 000000002824: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002828: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 00000000282C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002830: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002834: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002838: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002840: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000002848: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 000000002850: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000002854: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 00000000285C: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002860: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002864: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002868: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 00000000286C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002870: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002878: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 000000002880: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 000000002888: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 00000000288C: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 000000002894: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002898: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 00000000289C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000028A0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000028A4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000028A8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000028B0: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 0000000028B8: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 0000000028C0: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 0000000028C4: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 0000000028CC: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000028D0: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000028D4: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000028D8: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000028DC: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000028E0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000028E8: D613001C 04723719
	s_nop 1                                                    // 0000000028F0: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 0000000028F4: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 0000000028FC: 06383918
	s_nop 1                                                    // 000000002900: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002904: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 00000000290C: 06383918
	s_nop 1                                                    // 000000002910: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002914: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 00000000291C: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002920: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002924: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002928: BEFE00C1
	s_mul_i32 s28, s10, 5                                      // 00000000292C: 961C850A
	s_lshl_b32 s28, s28, 2                                     // 000000002930: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000002934: 4A2E081C
	s_mul_i32 s28, s13, 5                                      // 000000002938: 961C850D
	s_lshl_b32 s28, s28, 2                                     // 00000000293C: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000002940: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000002944: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 000000002950: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 00000000295C: 36400087
	v_cmp_eq_u32_e64 s39, v32, 5                               // 000000002960: D44A0027 00010B20
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002968: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000002970: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 000000002978: BFC00000
	v_mov_b32_e32 v18, 0                                       // 00000000297C: 7E240280
	v_mov_b32_e32 v10, 0                                       // 000000002980: 7E140280
	v_mov_b32_e32 v19, 0                                       // 000000002984: 7E260280
	v_mov_b32_e32 v11, 0                                       // 000000002988: 7E160280
	v_mov_b32_e32 v20, 0                                       // 00000000298C: 7E280280
	v_mov_b32_e32 v12, 0                                       // 000000002990: 7E180280
	v_mov_b32_e32 v21, 0                                       // 000000002994: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 000000002998: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 00000000299C: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 0000000029A0: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 0000000029AC: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 0000000029B8: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 0000000029BC: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 0000000029C4: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 0000000029CC: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 0000000029D0: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 0000000029D8: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 0000000029DC: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 0000000029E4: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000029E8: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000029EC: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000029F0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000029F4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000029F8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002A00: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000002A08: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000002A10: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000002A14: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000002A1C: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002A20: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002A24: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002A28: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002A2C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002A30: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002A38: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000002A40: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000002A48: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000002A4C: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 000000002A54: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002A58: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002A5C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002A60: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002A64: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002A68: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002A70: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 000000002A78: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 000000002A80: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 000000002A84: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 000000002A8C: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002A90: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002A94: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002A98: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002A9C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002AA0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002AA8: D613001C 04723719
	s_nop 1                                                    // 000000002AB0: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002AB4: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002ABC: 06383918
	s_nop 1                                                    // 000000002AC0: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002AC4: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002ACC: 06383918
	s_nop 1                                                    // 000000002AD0: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002AD4: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 000000002ADC: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002AE0: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002AE4: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002AE8: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000002AEC: BFC00000
	v_mov_b32_e32 v28, 0                                       // 000000002AF0: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 000000002AF4: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000002AFC: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 000000002B00: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 000000002B08: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002B0C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002B10: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002B14: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002B18: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002B1C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002B24: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000002B2C: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 000000002B34: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000002B38: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000002B40: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002B44: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002B48: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002B4C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002B50: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002B54: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002B5C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 000000002B64: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 000000002B6C: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 000000002B70: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 000000002B78: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002B7C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002B80: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002B84: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002B88: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002B8C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002B94: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 000000002B9C: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 000000002BA4: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 000000002BA8: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 000000002BB0: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002BB4: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002BB8: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002BBC: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002BC0: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002BC4: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002BCC: D613001C 04723719
	s_nop 1                                                    // 000000002BD4: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002BD8: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002BE0: 06383918
	s_nop 1                                                    // 000000002BE4: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002BE8: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002BF0: 06383918
	s_nop 1                                                    // 000000002BF4: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002BF8: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 000000002C00: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002C04: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002C08: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002C0C: BEFE00C1
	s_mul_i32 s28, s10, 6                                      // 000000002C10: 961C860A
	s_lshl_b32 s28, s28, 2                                     // 000000002C14: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000002C18: 4A2E081C
	s_mul_i32 s28, s13, 6                                      // 000000002C1C: 961C860D
	s_lshl_b32 s28, s28, 2                                     // 000000002C20: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000002C24: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000002C28: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 000000002C34: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 000000002C40: 36400087
	v_cmp_eq_u32_e64 s39, v32, 6                               // 000000002C44: D44A0027 00010D20
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002C4C: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000002C54: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 000000002C5C: BFC00000
	v_mov_b32_e32 v18, 0                                       // 000000002C60: 7E240280
	v_mov_b32_e32 v10, 0                                       // 000000002C64: 7E140280
	v_mov_b32_e32 v19, 0                                       // 000000002C68: 7E260280
	v_mov_b32_e32 v11, 0                                       // 000000002C6C: 7E160280
	v_mov_b32_e32 v20, 0                                       // 000000002C70: 7E280280
	v_mov_b32_e32 v12, 0                                       // 000000002C74: 7E180280
	v_mov_b32_e32 v21, 0                                       // 000000002C78: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 000000002C7C: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 000000002C80: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 000000002C84: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 000000002C90: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 000000002C9C: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002CA0: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000002CA8: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 000000002CB0: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 000000002CB4: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 000000002CBC: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 000000002CC0: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 000000002CC8: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002CCC: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002CD0: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002CD4: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002CD8: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002CDC: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002CE4: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000002CEC: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000002CF4: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000002CF8: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000002D00: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002D04: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002D08: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002D0C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002D10: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002D14: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002D1C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000002D24: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000002D2C: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000002D30: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 000000002D38: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002D3C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002D40: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002D44: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002D48: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002D4C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002D54: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 000000002D5C: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 000000002D64: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 000000002D68: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 000000002D70: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002D74: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002D78: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002D7C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002D80: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002D84: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002D8C: D613001C 04723719
	s_nop 1                                                    // 000000002D94: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002D98: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002DA0: 06383918
	s_nop 1                                                    // 000000002DA4: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002DA8: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002DB0: 06383918
	s_nop 1                                                    // 000000002DB4: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002DB8: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 000000002DC0: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002DC4: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002DC8: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002DCC: BEFE00C1
	s_wait_loadcnt 0x0                                         // 000000002DD0: BFC00000
	v_mov_b32_e32 v28, 0                                       // 000000002DD4: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 000000002DD8: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 000000002DE0: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 000000002DE4: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 000000002DEC: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002DF0: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002DF4: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002DF8: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002DFC: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002E00: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002E08: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 000000002E10: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 000000002E18: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000002E1C: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000002E24: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002E28: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002E2C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002E30: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002E34: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002E38: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002E40: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 000000002E48: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 000000002E50: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 000000002E54: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 000000002E5C: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002E60: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002E64: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002E68: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002E6C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002E70: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002E78: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 000000002E80: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 000000002E88: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 000000002E8C: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 000000002E94: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002E98: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002E9C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002EA0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002EA4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002EA8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002EB0: D613001C 04723719
	s_nop 1                                                    // 000000002EB8: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 000000002EBC: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000002EC4: 06383918
	s_nop 1                                                    // 000000002EC8: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 000000002ECC: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000002ED4: 06383918
	s_nop 1                                                    // 000000002ED8: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 000000002EDC: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 000000002EE4: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 000000002EE8: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 000000002EEC: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 000000002EF0: BEFE00C1
	s_mul_i32 s28, s10, 7                                      // 000000002EF4: 961C870A
	s_lshl_b32 s28, s28, 2                                     // 000000002EF8: 841C821C
	v_add_nc_u32_e32 v23, s28, v4                              // 000000002EFC: 4A2E081C
	s_mul_i32 s28, s13, 7                                      // 000000002F00: 961C870D
	s_lshl_b32 s28, s28, 2                                     // 000000002F04: 841C821C
	v_add_nc_u32_e32 v34, s28, v33                             // 000000002F08: 4A44421C
	buffer_load_b128 v[14:17], v34, s[48:51], null offen       // 000000002F0C: C405C07C 4080600E 00000022
	buffer_load_b128 v[6:9], v23, s[40:43], null offen         // 000000002F18: C405C07C 40805006 00000017
	v_and_b32_e32 v32, 7, v0                                   // 000000002F24: 36400087
	v_cmp_eq_u32_e64 s39, v32, 7                               // 000000002F28: D44A0027 00010F20
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002F30: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000002F38: 4A4444FF 00000080
	s_wait_loadcnt 0x0                                         // 000000002F40: BFC00000
	v_mov_b32_e32 v18, 0                                       // 000000002F44: 7E240280
	v_mov_b32_e32 v10, 0                                       // 000000002F48: 7E140280
	v_mov_b32_e32 v19, 0                                       // 000000002F4C: 7E260280
	v_mov_b32_e32 v11, 0                                       // 000000002F50: 7E160280
	v_mov_b32_e32 v20, 0                                       // 000000002F54: 7E280280
	v_mov_b32_e32 v12, 0                                       // 000000002F58: 7E180280
	v_mov_b32_e32 v21, 0                                       // 000000002F5C: 7E2A0280
	v_mov_b32_e32 v13, 0                                       // 000000002F60: 7E1A0280
	s_mov_b32 exec_lo, s34                                     // 000000002F64: BEFE0022
	buffer_load_b128 v[18:21], v34, s[48:51], null offen       // 000000002F68: C405C07C 40806012 00000022
	buffer_load_b128 v[10:13], v23, s[40:43], null offen       // 000000002F74: C405C07C 4080500A 00000017
	s_mov_b32 exec_lo, -1                                      // 000000002F80: BEFE00C1
	v_add_nc_u32_e32 v23, 0x80, v23                            // 000000002F84: 4A2E2EFF 00000080
	v_add_nc_u32_e32 v34, 0x80, v34                            // 000000002F8C: 4A4444FF 00000080
	v_mov_b32_e32 v28, 0                                       // 000000002F94: 7E380280
	v_and_b32_e32 v24, 0xffff, v6                              // 000000002F98: 36300CFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v6                              // 000000002FA0: 32320C90
	v_and_b32_e32 v26, 0xffff, v14                             // 000000002FA4: 36341CFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v14                             // 000000002FAC: 32361C90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002FB0: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002FB4: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002FB8: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002FBC: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002FC0: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000002FC8: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v7                              // 000000002FD0: 36300EFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v7                              // 000000002FD8: 32320E90
	v_and_b32_e32 v26, 0xffff, v15                             // 000000002FDC: 36341EFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v15                             // 000000002FE4: 32361E90
	v_cvt_f32_bf16_e32 v24, v24                                // 000000002FE8: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000002FEC: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000002FF0: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000002FF4: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000002FF8: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000003000: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v8                              // 000000003008: 363010FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v8                              // 000000003010: 32321090
	v_and_b32_e32 v26, 0xffff, v16                             // 000000003014: 363420FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v16                             // 00000000301C: 32362090
	v_cvt_f32_bf16_e32 v24, v24                                // 000000003020: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000003024: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000003028: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 00000000302C: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000003030: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000003038: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v9                              // 000000003040: 363012FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v9                              // 000000003048: 32321290
	v_and_b32_e32 v26, 0xffff, v17                             // 00000000304C: 363422FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v17                             // 000000003054: 32362290
	v_cvt_f32_bf16_e32 v24, v24                                // 000000003058: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 00000000305C: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000003060: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000003064: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000003068: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000003070: D613001C 04723719
	s_nop 1                                                    // 000000003078: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 00000000307C: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 000000003084: 06383918
	s_nop 1                                                    // 000000003088: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 00000000308C: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 000000003094: 06383918
	s_nop 1                                                    // 000000003098: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 00000000309C: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 0000000030A4: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 0000000030A8: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 0000000030AC: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 0000000030B0: BEFE00C1
	s_wait_loadcnt 0x0                                         // 0000000030B4: BFC00000
	v_mov_b32_e32 v28, 0                                       // 0000000030B8: 7E380280
	v_and_b32_e32 v24, 0xffff, v10                             // 0000000030BC: 363014FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v10                             // 0000000030C4: 32321490
	v_and_b32_e32 v26, 0xffff, v18                             // 0000000030C8: 363424FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v18                             // 0000000030D0: 32362490
	v_cvt_f32_bf16_e32 v24, v24                                // 0000000030D4: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 0000000030D8: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 0000000030DC: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 0000000030E0: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 0000000030E4: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 0000000030EC: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v11                             // 0000000030F4: 363016FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v11                             // 0000000030FC: 32321690
	v_and_b32_e32 v26, 0xffff, v19                             // 000000003100: 363426FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v19                             // 000000003108: 32362690
	v_cvt_f32_bf16_e32 v24, v24                                // 00000000310C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000003110: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000003114: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000003118: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 00000000311C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000003124: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v12                             // 00000000312C: 363018FF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v12                             // 000000003134: 32321890
	v_and_b32_e32 v26, 0xffff, v20                             // 000000003138: 363428FF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v20                             // 000000003140: 32362890
	v_cvt_f32_bf16_e32 v24, v24                                // 000000003144: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000003148: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 00000000314C: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000003150: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 000000003154: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 00000000315C: D613001C 04723719
	v_and_b32_e32 v24, 0xffff, v13                             // 000000003164: 36301AFF 0000FFFF
	v_lshrrev_b32_e32 v25, 16, v13                             // 00000000316C: 32321A90
	v_and_b32_e32 v26, 0xffff, v21                             // 000000003170: 36342AFF 0000FFFF
	v_lshrrev_b32_e32 v27, 16, v21                             // 000000003178: 32362A90
	v_cvt_f32_bf16_e32 v24, v24                                // 00000000317C: 7E30E518
	v_cvt_f32_bf16_e32 v26, v26                                // 000000003180: 7E34E51A
	v_cvt_f32_bf16_e32 v25, v25                                // 000000003184: 7E32E519
	v_cvt_f32_bf16_e32 v27, v27                                // 000000003188: 7E36E51B
	v_fma_f32 v28, v24, v26, v28                               // 00000000318C: D613001C 04723518
	v_fma_f32 v28, v25, v27, v28                               // 000000003194: D613001C 04723719
	s_nop 1                                                    // 00000000319C: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf// 0000000031A0: 7E3002FA FF00B11C
	v_add_f32_e32 v28, v24, v28                                // 0000000031A8: 06383918
	s_nop 1                                                    // 0000000031AC: BF800001
	v_mov_b32_dpp v24, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf// 0000000031B0: 7E3002FA FF004E1C
	v_add_f32_e32 v28, v24, v28                                // 0000000031B8: 06383918
	s_nop 1                                                    // 0000000031BC: BF800001
	v_mov_b32_dpp v24, v28 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1// 0000000031C0: 7E3002FA FF09141C
	s_mov_b32 exec_lo, s39                                     // 0000000031C8: BEFE0027
	v_add_f32_e32 v22, v28, v22                                // 0000000031CC: 062C2D1C
	v_add_f32_e32 v22, v24, v22                                // 0000000031D0: 062C2D18
	s_mov_b32 exec_lo, -1                                      // 0000000031D4: BEFE00C1
	buffer_store_b32 v22, v5, s[44:47], null offen             // 0000000031D8: C406807C 40805816 00000005
	s_wait_loadcnt 0x0                                         // 0000000031E4: BFC00000
	s_endpgm                                                   // 0000000031E8: BFB00000
	s_code_end                                                 // 0000000031EC: BF9F0000
	s_code_end                                                 // 0000000031F0: BF9F0000
	s_code_end                                                 // 0000000031F4: BF9F0000
	s_code_end                                                 // 0000000031F8: BF9F0000
	s_code_end                                                 // 0000000031FC: BF9F0000
	s_code_end                                                 // 000000003200: BF9F0000
	s_code_end                                                 // 000000003204: BF9F0000
	s_code_end                                                 // 000000003208: BF9F0000
	s_code_end                                                 // 00000000320C: BF9F0000
	s_code_end                                                 // 000000003210: BF9F0000
	s_code_end                                                 // 000000003214: BF9F0000
	s_code_end                                                 // 000000003218: BF9F0000
	s_code_end                                                 // 00000000321C: BF9F0000
	s_code_end                                                 // 000000003220: BF9F0000
	s_code_end                                                 // 000000003224: BF9F0000
	s_code_end                                                 // 000000003228: BF9F0000
	s_code_end                                                 // 00000000322C: BF9F0000
	s_code_end                                                 // 000000003230: BF9F0000
	s_code_end                                                 // 000000003234: BF9F0000
	s_code_end                                                 // 000000003238: BF9F0000
	s_code_end                                                 // 00000000323C: BF9F0000
	s_code_end                                                 // 000000003240: BF9F0000
	s_code_end                                                 // 000000003244: BF9F0000
	s_code_end                                                 // 000000003248: BF9F0000
	s_code_end                                                 // 00000000324C: BF9F0000
	s_code_end                                                 // 000000003250: BF9F0000
	s_code_end                                                 // 000000003254: BF9F0000
	s_code_end                                                 // 000000003258: BF9F0000
	s_code_end                                                 // 00000000325C: BF9F0000
	s_code_end                                                 // 000000003260: BF9F0000
	s_code_end                                                 // 000000003264: BF9F0000
	s_code_end                                                 // 000000003268: BF9F0000
	s_code_end                                                 // 00000000326C: BF9F0000
	s_code_end                                                 // 000000003270: BF9F0000
	s_code_end                                                 // 000000003274: BF9F0000
	s_code_end                                                 // 000000003278: BF9F0000
	s_code_end                                                 // 00000000327C: BF9F0000
	s_code_end                                                 // 000000003280: BF9F0000
	s_code_end                                                 // 000000003284: BF9F0000
	s_code_end                                                 // 000000003288: BF9F0000
	s_code_end                                                 // 00000000328C: BF9F0000
	s_code_end                                                 // 000000003290: BF9F0000
	s_code_end                                                 // 000000003294: BF9F0000
	s_code_end                                                 // 000000003298: BF9F0000
	s_code_end                                                 // 00000000329C: BF9F0000
	s_code_end                                                 // 0000000032A0: BF9F0000
	s_code_end                                                 // 0000000032A4: BF9F0000
	s_code_end                                                 // 0000000032A8: BF9F0000
	s_code_end                                                 // 0000000032AC: BF9F0000
	s_code_end                                                 // 0000000032B0: BF9F0000
	s_code_end                                                 // 0000000032B4: BF9F0000
	s_code_end                                                 // 0000000032B8: BF9F0000
	s_code_end                                                 // 0000000032BC: BF9F0000
	s_code_end                                                 // 0000000032C0: BF9F0000
	s_code_end                                                 // 0000000032C4: BF9F0000
	s_code_end                                                 // 0000000032C8: BF9F0000
	s_code_end                                                 // 0000000032CC: BF9F0000
	s_code_end                                                 // 0000000032D0: BF9F0000
	s_code_end                                                 // 0000000032D4: BF9F0000
	s_code_end                                                 // 0000000032D8: BF9F0000
	s_code_end                                                 // 0000000032DC: BF9F0000
	s_code_end                                                 // 0000000032E0: BF9F0000
	s_code_end                                                 // 0000000032E4: BF9F0000
	s_code_end                                                 // 0000000032E8: BF9F0000
	s_code_end                                                 // 0000000032EC: BF9F0000
	s_code_end                                                 // 0000000032F0: BF9F0000
	s_code_end                                                 // 0000000032F4: BF9F0000
	s_code_end                                                 // 0000000032F8: BF9F0000
	s_code_end                                                 // 0000000032FC: BF9F0000
	s_code_end                                                 // 000000003300: BF9F0000
	s_code_end                                                 // 000000003304: BF9F0000
	s_code_end                                                 // 000000003308: BF9F0000
	s_code_end                                                 // 00000000330C: BF9F0000
	s_code_end                                                 // 000000003310: BF9F0000
	s_code_end                                                 // 000000003314: BF9F0000
	s_code_end                                                 // 000000003318: BF9F0000
	s_code_end                                                 // 00000000331C: BF9F0000
	s_code_end                                                 // 000000003320: BF9F0000
	s_code_end                                                 // 000000003324: BF9F0000
	s_code_end                                                 // 000000003328: BF9F0000
	s_code_end                                                 // 00000000332C: BF9F0000
	s_code_end                                                 // 000000003330: BF9F0000
	s_code_end                                                 // 000000003334: BF9F0000
	s_code_end                                                 // 000000003338: BF9F0000
	s_code_end                                                 // 00000000333C: BF9F0000
	s_code_end                                                 // 000000003340: BF9F0000
	s_code_end                                                 // 000000003344: BF9F0000
	s_code_end                                                 // 000000003348: BF9F0000
	s_code_end                                                 // 00000000334C: BF9F0000
	s_code_end                                                 // 000000003350: BF9F0000
	s_code_end                                                 // 000000003354: BF9F0000
	s_code_end                                                 // 000000003358: BF9F0000
	s_code_end                                                 // 00000000335C: BF9F0000
	s_code_end                                                 // 000000003360: BF9F0000
	s_code_end                                                 // 000000003364: BF9F0000
	s_code_end                                                 // 000000003368: BF9F0000
	s_code_end                                                 // 00000000336C: BF9F0000
	s_code_end                                                 // 000000003370: BF9F0000
	s_code_end                                                 // 000000003374: BF9F0000
	s_code_end                                                 // 000000003378: BF9F0000
	s_code_end                                                 // 00000000337C: BF9F0000
