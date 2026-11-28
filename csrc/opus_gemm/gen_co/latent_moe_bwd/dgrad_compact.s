// SPDX-License-Identifier: MIT
.Lentry:
    .incbin "body.bin", 0, 0x50
    // Each lane counts its contiguous run of experts for the group search.
    // The mixed-height tiling uses exactly ceil(rows / 256) tiles.
    v_mul_lo_u32 v1, s49, v8
    s_sub_u32 s4, s33, 1
    s_mov_b32 s3, s49
.Lcount:
    v_min_u32_e32 v2, s4, v1
    v_lshlrev_b32_e32 v2, 2, v2
    global_load_dword v3, v2, s[42:43]
    v_cmp_lt_u32_e32 vcc, s4, v1
    s_waitcnt vmcnt(0)
    v_add_u32_e32 v3, 0xff, v3
    v_lshrrev_b32_e32 v3, 8, v3
    v_cndmask_b32_e64 v3, v3, 0, vcc
    v_add_u32_e32 v9, v9, v3
    v_add_u32_e32 v1, 1, v1
    s_sub_u32 s3, s3, 1
    s_cmp_lg_u32 s3, 0
    s_cbranch_scc1 .Lcount
    s_branch .Lprefix
    .org .Lentry+0xe1c
.Lprefix:
    .incbin "body.bin", 0xe1c, 0x14fc-0xe1c
    s_add_i32 s46, s15, s16
    .incbin "body.bin", 0x1500, 50272-0x1500
