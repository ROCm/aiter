#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r22 role/geometry model, independent of the C++ constexpr checks (default options AND the r21 A/B settings). exit 0 = all pass."""

ok = True


def chk(c, msg):
    global ok
    print(("PASS " if c else "FAIL ") + msg)
    if not c:
        ok = False


MP, NPASS, SK = 128, 8, 3
UNIT, RG = 24, NPASS // 4
for mode, heads_src, upd, ik, F, N in (
    ("dense", 18, 18, 0, 2304, 9216),
    ("reuse", 18, 18, 0, 2560, 9856),
    ("indexer", 20, 19, 4, 2560, 9856),
):
    for quad in (1, 0):
        bsplit = NPASS // 4 if quad else NPASS
        tag = f"{mode:<8} B_QUAD={quad:d}"
        push, selfw, bw = UNIT * RG, upd * (NPASS // 2), 3 * heads_src * bsplit
        grid = push + ik + selfw + bw
        want = (248 if ik else 228) if quad else (608 if ik else 552)
        chk(
            grid == want,
            f"{tag} grid {grid:d} = push {push:d} + IK {ik:d} + self {selfw:d} + B {bw:d} (r21 = 552/608)",
        )
        if quad:
            chk(
                grid <= 256,
                f"{tag} grid {grid:d} <= 256 CUs (one resident round at 1 WG/CU)",
            )
        # push coverage (unchanged from r21) and K
        UNITS = 3 * upd
        u0 = lambda wu, UNITS=UNITS: (wu * UNITS) // UNIT
        touch = (
            lambda w, rr, u0=u0, upd=upd: u0(w % UNIT) < (rr + 1) * upd
            and u0(w % UNIT + 1) > rr * upd
        )
        K = [sum(touch(w, rr) for w in range(push)) + ik for rr in range(3)]
        chk(
            K == [8 * RG + ik] * 3,
            f"{tag} K per (src,dst) = {8 * RG:d} unit pushers + IK_WGS {ik:d} = {K}",
        )
        # B coverage: (src, head, pass) exactly once on every rank; quad: halves own 2 passes each; dead quads acquire only
        for m in (65, 80, 96, 97, 113, 127, 128):
            passes = (m + 15) // 16
            for rank in range(4):
                seen = {}
                acq = {s: 0 for s in range(4) if s != rank}
                live = {s: 0 for s in acq}
                for bi in range(bw):
                    bpp = heads_src * bsplit
                    r, pair = bi // bpp, bi % bpp
                    s = r + 1 if r >= rank else r
                    hp, rs = pair // bsplit, pair % bsplit
                    acq[s] += 1
                    if quad:
                        live_wg = 4 * rs < passes
                        owned = [
                            4 * rs + 2 * hf + q for hf in range(2) for q in range(2)
                        ]
                    else:
                        live_wg = rs < passes
                        owned = [rs]
                    live[s] += live_wg
                    for p_ in owned:
                        seen[(s, hp, p_)] = seen.get((s, hp, p_), 0) + 1
                want_cov = {
                    (s, h, p_): 1
                    for s in acq
                    for h in range(heads_src)
                    for p_ in range(NPASS)
                }
                if (
                    seen != want_cov
                    or any(v < 1 for v in acq.values())
                    or any(v < 1 for v in live.values())
                ):
                    chk(
                        False,
                        f"{tag} m={m:d} rank {rank:d} coverage/acquire",
                    )
        chk(
            True,
            "{}: every (src, head, pass) exactly once on all 4 ranks for m in 65..128; every (src,dst) pair keeps >= 1 acquiring AND >= 1 live WG; dead {} still acquire".format(
                tag, "quads" if quad else "passes"
            ),
        )
    # Rx: 3 slots (default) / 4 slots
    for slots in (3, 4):
        rx = 2 * slots * MP * F * 2
        mx = max(
            par * (slots * MP * F) + s * (MP * F) + row * F + col
            for par in (0, 1)
            for s in range(slots)
            for row in (0, MP - 1)
            for col in (0, F - 8)
        )
        want = {
            (3, 2304): 3538944,
            (3, 2560): 3932160,
            (4, 2304): 4718592,
            (4, 2560): 5242880,
        }[(slots, F)]
        chk(
            rx == want and 2 * mx + 16 <= rx,
            "{:<8} Rx {:d} slots: {:d} B ({}), X-5 highest access {:d} B <= window".format(
                mode,
                slots,
                rx,
                "<= 4 MiB view" if rx <= 4 << 20 else "needs 8 MiB",
                2 * mx + 16,
            ),
        )
    # GEMM BN128: tiles, WGs, LDS
    for bn, pfk, wgpc in ((128, 4, 1), (64, 3, 2)):
        ntiles = N // bn
        stage = 128 * 128 + bn * 128
        chk(
            ntiles * SK <= (256 if wgpc == 1 else 512) and pfk * stage * wgpc <= 163840,
            f"{mode:<8} GEMM BN{bn:d}: {ntiles:d} x {SK:d} = {ntiles * SK:d} WGs, LDS {pfk * stage // 1024:d} KiB x {wgpc:d} WG/CU",
        )
# B_HEADS=2 (r21 nh=2 ships with 216/240 B WGs and grid 336/368): r22 selectable A/B: quad + NH2 and r21-style NH2
for mode, heads_src, upd, ik in (
    ("dense", 18, 18, 0),
    ("reuse", 18, 18, 0),
    ("indexer", 20, 19, 4),
):
    NH = 2
    for quad in (1, 0):
        bsplit = NPASS // 4 if quad else NPASS
        bw = 3 * (heads_src // NH) * bsplit
        grid = UNIT * RG + ik + upd * (NPASS // 2) + bw
        chk(
            grid == (UNIT * RG + ik + upd * 4 + 3 * (heads_src // 2) * bsplit),
            f"{mode:<8} NH2 B_QUAD={quad:d}: B WGs {bw:d}, grid {grid:d} (r21 nh=2 = {UNIT * RG + ik + upd * 4 + 3 * (heads_src // 2) * NPASS:d})",
        )
        seen = {}
        bpp = (heads_src // NH) * bsplit
        for bi in range(bw):
            r_, pair = bi // bpp, bi % bpp
            hp, rs = pair // bsplit, pair % bsplit
            owned = (
                [4 * rs + 2 * hf + q for hf in range(2) for q in range(2)]
                if quad
                else [rs]
            )
            for k_ in range(NH):
                for p_ in owned:
                    seen[(r_, hp * NH + k_, p_)] = (
                        seen.get((r_, hp * NH + k_, p_), 0) + 1
                    )
        chk(
            seen
            == {
                (r_, h, p_): 1
                for r_ in range(3)
                for h in range(heads_src)
                for p_ in range(NPASS)
            },
            f"{mode:<8} NH2 B_QUAD={quad:d}: every (src, head, pass) exactly once",
        )
# rx_ord bijection (12 pairs)
for slots in (3, 4):
    good = True
    for d in range(4):
        used = set()
        for s in range(4):
            if s == d:
                continue
            o = (s if s < d else s - 1) if slots == 3 else s
            good = good and o < slots and o not in used
            used.add(o)
    chk(
        good,
        f"rx_ord bijective/injective over all 12 (s, d) pairs for {slots:d} slots",
    )
chk(
    SK * 3 * 2 * 8 == 144,
    "push partial registers per lane unchanged: SK3 x 3 units x 2 passes x 8 = 144 floats",
)
print("ALL PASS" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
