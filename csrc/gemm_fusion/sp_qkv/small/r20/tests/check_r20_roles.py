#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r20 (MP64) role-table model, independent of the C++ constexpr checks. Usage: check_r20_roles.py -> exit 0 = all pass.
Mirrors ran_kernel.cu: push WG w owns units [floor(U*w/24), floor(U*(w+1)/24)) of 3 destinations; half hf = tid>>8 owns row passes {2hf, 2hf+1}; IK owners o in {0,1} own passes {2o, 2o+1} and push to all 3 destinations;
self WG k = 2h + sp owns head h, passes {2sp, 2sp+1}; B WG bi -> (source r, pair = hp*4 + rsp) owns head group hp (NH heads) and ONE pass rsp. m = live rows (33..64)."""
import itertools
ok = True
def chk(c, msg):
    global ok
    print(('PASS ' if c else 'FAIL ') + msg)
    if not c: ok = False
MP, NPASS, SK = 64, 4, 3
for mode, heads_src, upd, ik, F, N, NT in (('dense', 18, 18, 0, 2304, 9216, 144), ('reuse', 18, 18, 0, 2304 if False else 2560, 9856, 154), ('indexer', 20, 19, 2, 2560, 9856, 154)):
    NH = 1
    push, selfw, bw = 24, upd * 2, 3 * (heads_src // NH) * NPASS
    grid = push + ik + selfw + bw
    chk(grid == (304 if ik else 276) and grid <= 512, '%-8s grid %d = push 24 + IK %d + self %d + B %d (<= 512 slots)' % (mode, grid, ik, selfw, bw))
    UNITS = 3 * upd
    g0 = lambda w: (w * UNITS) // push
    # push coverage: every (dst r, head h, pass) exactly once
    seen = {}
    for w in range(push):
        for g in range(g0(w), g0(w + 1)):
            for hf in range(2):
                for q in range(2):
                    key = (g // upd, g % upd, 2 * hf + q); seen[key] = seen.get(key, 0) + 1
    chk(all(v == 1 for v in seen.values()) and len(seen) == 3 * upd * NPASS, '%-8s push: each (dst, unit, pass) pushed exactly once by the two halves (%d tuples)' % (mode, len(seen)))
    # K per (src,dst)
    touch = lambda w, r: g0(w) < (r + 1) * upd and g0(w + 1) > r * upd
    K = [sum(touch(w, r) for w in range(push)) + ik for r in range(3)]
    chk(K == [8 + ik] * 3, '%-8s K per (src,dst) = unit pushers 8 + IK_WGS %d = %s' % (mode, ik, K))
    # IK owners: rows of the IK head covered once per pass, each owner releases once per destination
    if ik:
        passes_cov = sorted(p for o in range(ik) for p in (2 * o, 2 * o + 1))
        chk(passes_cov == list(range(NPASS)), '%-8s IK owners cover passes %s once, each releases once per destination' % (mode, passes_cov))
    # self: (head, pass) once
    sc = {}
    for k in range(selfw):
        h, sp = k // 2, k % 2
        for q in range(2): sc[(h, 2 * sp + q)] = sc.get((h, 2 * sp + q), 0) + 1
    chk(len(sc) == upd * NPASS and all(v == 1 for v in sc.values()), '%-8s self: each (head, pass) once by exactly one of 2 row-group WGs' % mode)
    # B: (src, head, pass) once on every rank; dead-pass WGs still acquire
    for m in (33, 40, 48, 49, 63, 64):
        passes = (m + 15) // 16
        for rank in range(4):
            seenb = {}; acq = {s: 0 for s in range(4) if s != rank}; live = {s: 0 for s in acq}
            for bi in range(bw):
                bpp = (heads_src // NH) * NPASS
                r, pair = bi // bpp, bi % bpp
                s = r + 1 if r >= rank else r
                hp, rsp = pair // NPASS, pair % NPASS
                for kk in range(NH): seenb[(s, hp * NH + kk, rsp)] = seenb.get((s, hp * NH + kk, rsp), 0) + 1
                acq[s] += 1                       # every B WG acquires flag[src] (dead or live)
                live[s] += (rsp < passes)         # live-pass WGs load/check/store
            want = {(s, h, p): 1 for s in acq for h in range(heads_src) for p in range(NPASS)}
            if seenb != want or any(v < 1 for v in acq.values()) or any(v < 1 for v in live.values()): chk(False, '%s m=%d rank %d B coverage/acquire' % (mode, m, rank))
    chk(True, '%-8s B coverage: every (src, head, pass) exactly once on all 4 ranks; every (src,dst) pair has >=1 acquiring WG for m in {33,40,48,49,63,64} (dead passes acquire, live WGs >= 1)' % mode)
    rx = 2 * 4 * MP * F * 2
    chk(rx <= 4 * 1024 * 1024 and rx == (2359296 if mode == 'dense' else 2621440), '%-8s Rx window %d B <= 4 MiB (X-5 recomputed)' % (mode, rx))
    mx = max(par * (4 * MP * F) + src * (MP * F) + row * F + col for par in (0, 1) for src in range(4) for row in (0, MP - 1) for col in (0, F - 8))
    chk(2 * mx + 16 <= rx, '%-8s X-5: highest Rx vector access %d B <= window %d' % (mode, 2 * mx + 16, rx))
    gemm_wgs = NT * SK
    chk(gemm_wgs <= 512, '%-8s GEMM grid %d x %d = %d WGs <= 512 slots; P = %d bytes' % (mode, NT, SK, gemm_wgs, SK * MP * N * 4))
stage = 64 * 128 + 64 * 128
chk(4 * stage == 65536 and 2 * 4 * stage <= 163840, 'GEMM LDS: PFK4 x (A 8192 + B 8192) = 64 KiB per WG, 2 WG/CU = 128 KiB <= 160 KiB')
chk(SK * 3 * 2 * 8 == 144, 'push partial registers per lane: SK3 x 3 units x 2 passes x 8 = 144 floats (same as r19: a half owns TWO passes)')
print('ALL PASS' if ok else 'FAILED'); raise SystemExit(0 if ok else 1)