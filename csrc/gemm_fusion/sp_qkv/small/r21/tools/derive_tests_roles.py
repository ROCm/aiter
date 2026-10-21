import sys
r20,r21=sys.argv[1],sys.argv[2]
t=open(r20+'/tests/check_r20_roles.py',encoding='utf-8').read()
def rep(old,new):
    global t
    assert t.count(old)==1,old[:70]
    t=t.replace(old,new)
rep('MP, NPASS, SK = 64, 4, 3','MP, NPASS, SK = 128, 8, 3')
rep("    push, selfw, bw = 24, upd * 2, 3 * (heads_src // NH) * NPASS","    UNIT, RG = 24, NPASS // 4\n    push, selfw, bw = UNIT * RG, upd * (NPASS // 2), 3 * (heads_src // NH) * NPASS")
rep("('indexer', 20, 19, 2, 2560, 9856, 154)","('indexer', 20, 19, 4, 2560, 9856, 154)")
rep("chk(grid == (304 if ik else 276) and grid <= 512,","chk(grid == (608 if ik else 552),")
rep("(<= 512 slots)' % (mode, grid, ik, selfw, bw))","(NOTE: > 256 CUs, WGs beyond residency start as earlier ones retire)' % (mode, grid, ik, selfw, bw))")
rep("    g0 = lambda w: (w * UNITS) // push","    u0 = lambda wu: (wu * UNITS) // UNIT\n    g0 = lambda w: u0(w % UNIT)")
t=t.replace("g0(w + 1)","u0(w % UNIT + 1)")
rep("key = (g // upd, g % upd, 2 * hf + q)","key = (g // upd, g % upd, 4 * (w // UNIT) + 2 * hf + q)")
rep("chk(K == [8 + ik] * 3, '%-8s K per (src,dst) = unit pushers 8 + IK_WGS %d = %s' % (mode, ik, K))","chk(K == [8 * RG + ik] * 3, '%-8s K per (src,dst) = unit pushers %d + IK_WGS %d = %s' % (mode, 8 * RG, ik, K))")
rep("for m in (33, 40, 48, 49, 63, 64):","for m in (65, 80, 96, 97, 113, 127, 128):")
rep("h, sp = k // 2, k % 2","h, sp = k // (NPASS // 2), k % (NPASS // 2)")
rep("exactly one of 2 row-group WGs","exactly one of 4 row-group WGs")
rep("rx <= 4 * 1024 * 1024 and rx == (2359296 if mode == 'dense' else 2621440)","4 * 1024 * 1024 < rx <= 8 * 1024 * 1024 and rx == (4718592 if mode == 'dense' else 5242880)")
rep("Rx window %d B <= 4 MiB (X-5 recomputed)","Rx window %d B: needs the 8 MiB arena view (> 4 MiB, <= 8 MiB)")
rep("stage = 64 * 128 + 64 * 128\nchk(4 * stage == 65536 and 2 * 4 * stage <= 163840, 'GEMM LDS: PFK4 x (A 8192 + B 8192) = 64 KiB per WG, 2 WG/CU = 128 KiB <= 160 KiB')","stage = 128 * 128 + 64 * 128\nchk(3 * stage == 73728 and 2 * 3 * stage <= 163840, 'GEMM LDS: PFK3 x (A 16384 + B 8192) = 72 KiB per WG, 2 WG/CU = 144 KiB <= 160 KiB')")
rep("dead passes acquire, live WGs >= 1)","dead passes acquire, live WGs >= 1)")
rep("chk(gemm_wgs <= 512,","chk(gemm_wgs <= 512,")
rep("%-8s GEMM grid %d x %d = %d WGs <= 512 slots; P = %d bytes","%-8s GEMM grid %d x %d = %d WGs <= 512 slots; P = %d bytes")
rep("for m in (65, 80, 96, 97, 113, 127, 128):","for m in (65, 80, 96, 97, 113, 127, 128):")
open(r21+'/tests/check_r21_roles.py','w',encoding='utf-8',newline='').write(t); print('written')