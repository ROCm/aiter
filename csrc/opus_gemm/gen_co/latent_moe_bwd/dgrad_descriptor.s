// SPDX-License-Identifier: MIT
    s_branch .Ldescriptor
    s_nop 0
.Lpipeline:
    .incbin "body.bin", 8, 50272-8
.Ldescriptor:
    s_load_dwordx2 s[4:5], s[0:1], 0x18
    s_lshl_b32 s6, s3, 7
    s_waitcnt lgkmcnt(0)
    s_add_u32 s0, s4, s6
    s_addc_u32 s1, s5, 0
    s_load_dword s7, s[0:1], 0x58
    s_waitcnt lgkmcnt(0)
    s_cmp_lg_u32 s7, 0
    s_cbranch_scc1 .Lnonempty
    s_endpgm
.Lnonempty:
    .incbin "body.bin", 0, 8
    s_branch .Lpipeline
