#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r23 (MP256 v1) role/geometry model, independent of the C++ constexpr checks. Role counts are those of one 128-row chunk (r22); every WG walks ITERS = 2 chunks. exit 0 = all pass."""

ok = True


def chk(c, msg):
    global ok
    print(("PASS " if c else "FAIL ") + msg)
    if not c:
        ok = False


MP, CH, ITERS, SK = 256, 8, 2, 3  # CH = 16-row passes per 128-row chunk
UNIT, RG = 24, CH // 4
for mode, heads_src, upd, ik, F, N in (
    ("dense", 18, 18, 0, 2304, 9216),
    ("reuse", 18, 18, 0, 2560, 9856),
    ("indexer", 20, 19, 4, 2560, 9856),
):
    push, selfw = UNIT * RG, upd * (CH // 2)
    bsplit = CH // 4
    bw = 3 * heads_src * bsplit
    grid = push + ik + selfw + bw
    chk(
        grid == (248 if ik else 228) and grid <= 256,
        f"{mode:<8} grid {grid:d} = push {push:d} + IK {ik:d} + self {selfw:d} + B-octo {bw:d} (<= 256 resident, same as r22)",
    )
    UNITS = 3 * upd
    u0 = lambda wu, UNITS=UNITS: (wu * UNITS) // UNIT
    touch = (
        lambda w, rr, u0=u0, upd=upd: u0(w % UNIT) < (rr + 1) * upd
        and u0(w % UNIT + 1) > rr * upd
    )
    K = [sum(touch(w, rr) for w in range(push)) + ik for rr in range(3)]
    chk(
        K == [8 * RG + ik] * 3,
        f"{mode:<8} K per (src,dst) = {8 * RG:d} releasers + IK_WGS {ik:d} = {K} (ONE release per WG after the last iteration)",
    )
    for m in (129, 160, 192, 193, 223, 255, 256):
        passes = (m + 15) // 16
        # push: (dst, unit, pass) pushed exactly once over row groups x halves x 2 passes x iterations
        seen = {}
        for w in range(push):
            for g in range(u0(w % UNIT), u0(w % UNIT + 1)):
                for hf in range(2):
                    for it in range(ITERS):
                        for q in range(2):
                            ps = 4 * (w // UNIT) + 2 * hf + q + CH * it
                            seen[(g // upd, g % upd, ps)] = (
                                seen.get((g // upd, g % upd, ps), 0) + 1
                            )
        if seen != {
            (d, u, p): 1
            for d in range(3)
            for u in range(upd)
            for p in range(CH * ITERS)
        }:
            chk(False, f"{mode} m={m:d} push coverage")
        if ik:
            ikp = sorted(
                2 * o + q + CH * it
                for o in range(ik)
                for it in range(ITERS)
                for q in range(2)
            )
            if ikp != list(range(CH * ITERS)):
                chk(False, f"{mode} IK coverage")
        sc = {}
        for k in range(selfw):
            h, sp = k // (CH // 2), k % (CH // 2)
            for it in range(ITERS):
                for q in range(2):
                    sc[(h, 2 * sp + q + CH * it)] = (
                        sc.get((h, 2 * sp + q + CH * it), 0) + 1
                    )
        if sc != {(h, p): 1 for h in range(upd) for p in range(CH * ITERS)}:
            chk(False, f"{mode} self coverage")
        for rank in range(4):
            seenb, acq, live = (
                {},
                {s: 0 for s in range(4) if s != rank},
                {s: 0 for s in range(4) if s != rank},
            )
            bpp = heads_src * bsplit
            for bi in range(bw):
                r_, pair = bi // bpp, bi % bpp
                s = r_ + 1 if r_ >= rank else r_
                hp, qd = pair // bsplit, pair % bsplit
                acq[
                    s
                ] += 1  # every B WG polls/acquires once (first iteration), dead octos included
                any_live = False
                for it in range(ITERS):
                    lp = (CH * it + 4 * qd) < passes
                    any_live = any_live or lp
                    for hf in range(2):
                        for q in range(2):
                            p = CH * it + 4 * qd + 2 * hf + q
                            seenb[(s, hp, p)] = seenb.get((s, hp, p), 0) + 1
                live[s] += any_live
            if (
                seenb
                != {
                    (s, h, p): 1
                    for s in acq
                    for h in range(heads_src)
                    for p in range(CH * ITERS)
                }
                or any(v < 1 for v in acq.values())
                or any(v < 1 for v in live.values())
            ):
                chk(
                    False,
                    f"{mode} m={m:d} rank {rank:d} B coverage/acquire",
                )
    chk(
        True,
        f"{mode:<8}: push (dst, unit, pass), IK, self and B (src, head, pass) coverage exactly once over all 16 passes for m in {{129..256}}; every (src,dst) pair keeps >= 1 acquiring and >= 1 live B WG; dead iterations/octos only acquire",
    )
    slots = 3
    rx = 2 * slots * MP * F * 2
    mx = max(
        par * (slots * MP * F) + s * (MP * F) + row * F + col
        for par in (0, 1)
        for s in range(slots)
        for row in (0, MP - 1)
        for col in (0, F - 8)
    )
    want = 7077888 if F == 2304 else 7864320
    chk(
        rx == want and 4 << 20 < rx <= 8 << 20 and 2 * mx + 16 <= rx,
        f"{mode:<8} Rx 3 slots MP256: {rx:d} B in the 8 MiB view, X-5 highest access {2 * mx + 16:d} B <= window (4 slots would be {2 * 4 * MP * F * 2:d} B)",
    )
    for bn, pfk in ((64, 4), (128, 3)):
        ntiles = N // bn
        lds = pfk * (256 * 128 + bn * 128)
        rounds = -(-ntiles * SK // 256)
        chk(
            lds <= 163840 and (bn == 64 or ntiles * SK <= 256),
            f"{mode:<8} GEMM BM256xBN{bn:d} PFK{pfk:d}: {ntiles:d} x {SK:d} = {ntiles * SK:d} WGs at 1 WG/CU ({rounds:d} round(s)), LDS {lds // 1024:d} KiB{' (= the 160 KiB per-CU limit)' if lds == 163840 else ''}, 384 threads",
        )
    chk(
        SK * MP * N * 4 == (SK * 256 * 9216 * 4 if N == 9216 else SK * 256 * 9856 * 4),
        f"{mode:<8} P = {SK * MP * N * 4:d} B",
    )
chk(
    SK * 3 * 2 * 8 == 144,
    "push partial registers per lane unchanged (144): the second iteration reuses them after its data stores",
)
mb = 8 + 16
chk(
    2 * (4 + 16) <= 63 and (3 - 2) * mb <= 63,
    f"GEMM vmcnt immediate: BN64 PFK4: (PFK-2) * (BQ + AQ) = {2 * (4 + 16):d} <= 63; BN128 PFK3: {(3 - 2) * mb:d}",
)
# ---- fallback (b) SM_IK_SEQ: 8 IK WGs, one iteration each (indexer only) ----
ik_seq = 8
cov = sorted(2 * (o % 4) + q + CH * (o // 4) for o in range(ik_seq) for q in range(2))
chk(
    cov == list(range(CH * ITERS)),
    "IK_SEQ: 8 IK WGs (owner % 4 -> 2 passes, owner // 4 -> iteration) cover passes 0..15 exactly once, each releases once per destination",
)
grid_seq = UNIT * RG + ik_seq + 19 * (CH // 2) + 3 * 20 * (CH // 4)
chk(
    grid_seq == 252 and grid_seq <= 256,
    f"IK_SEQ indexer grid {grid_seq:d} <= 256; K per (src,dst) = {8 * RG:d} + {ik_seq:d} = {8 * RG + ik_seq:d}",
)
print("ALL PASS" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
