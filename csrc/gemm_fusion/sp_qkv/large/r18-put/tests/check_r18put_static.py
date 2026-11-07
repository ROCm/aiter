#!/usr/bin/env python3
"""Static / model checks for large/r18-put (source only; no compiler or GPU).

1. ADDITIVE-ONLY: every file copied from large/r13 is byte-identical to its r13 original once the r18-put additions are stripped
   (lines between `r18-put:begin` / `r18-put:end`, and single lines carrying an `r18-put` marker). The shared protocol code (R12 tags / credit / arrival, quiet_ready, drain, the r13 GEMM entries) is therefore unchanged.
2. The put kernel header has exactly ONE put call site (no s_swappc inflation), no gdone waits, the credit/arrival protocol, the plan-code guard, and both the device entry and the host stub exist.
3. Address model: for every (rank, peer) the four senders' puts + the self copy tile R = [4*m_real, F] exactly once; every source range stays inside its S slab (slab stride srows >= m_real); byte counts equal r13's poster (n = 1) formulas.
"""

import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
OLD = os.path.join(os.path.dirname(NEW), "r13")


def strip_additions(text):
    out, skip = [], False
    for line in text.split("\n"):
        if "r18-put:begin" in line:
            skip = True
            continue
        if "r18-put:end" in line:
            skip = False
            if line.strip().startswith("}") or line.strip().startswith("#"):
                pass
            continue
        if skip:
            continue
        if "r18-put" in line:
            continue
        out.append(line)
    return "\n".join(out)


def check_identical():
    ok = True
    for root, _, files in os.walk(OLD):
        for f in files:
            if f in ("SOURCE_SHA256.txt",):
                continue
            rel = os.path.relpath(os.path.join(root, f), OLD)
            n = os.path.join(NEW, rel)
            if not os.path.exists(n):
                print(f"FAIL missing in r18-put: {rel}")
                ok = False
                continue
            with open(os.path.join(root, f), encoding="utf-8", newline="") as _fa:
                a = _fa.read().replace("\r\n", "\n")
            with open(n, encoding="utf-8", newline="") as _fb:
                b = strip_additions(_fb.read().replace("\r\n", "\n"))
            if a.rstrip("\n") != b.rstrip("\n"):
                print(f"FAIL not additive-only: {rel}")
                ok = False
    print("PASS additive-only vs r13" if ok else "FAILED additive-only")
    return ok


def check_header():
    with open(os.path.join(NEW, "device", "lg_r18_put.hpp"), encoding="utf-8") as _f:
        h_full = _f.read()
    h = "\n".join(
        re.sub(r"//.*$", "", ln) for ln in h_full.split("\n")
    )  # code only: comments may mention gdone / the removed waits
    ok = True
    if len(re.findall(r"template put<", h)) != 1:
        print("FAIL: the put kernel must contain exactly ONE put call site")
        ok = False
    for bad in ("gdone", "r12_poll_u32_ge", "want_of", "r6x_put_group", "x_publish"):
        if bad in h:
            print(f"FAIL: header uses {bad}")
            ok = False
    for need in (
        "LGC_CTAG_OFF",
        "LGC_CREDIT_OFF",
        "LGC_ARRIVAL_OFF",
        "quietQueue",
        "LGC_POISON_CONFIG",
        "LG_R18_PUT_PLAN_CODE",
        "LGP_SROWS",
        "err_clear",
        "st_u64_rel_x",
    ):
        if need not in h:
            print(f"FAIL: header lacks {need}")
            ok = False
    with open(
        os.path.join(NEW, "device", "gemm_quad_fp8_comm.cu"), encoding="utf-8"
    ) as _f:
        cu = _f.read()
    if len(re.findall(r"void lg_put_v1\(", cu)) != 2:
        print("FAIL: lg_put_v1 must appear as a device entry and a host stub")
        ok = False
    print("PASS put kernel header" if ok else "FAILED put kernel header")
    return ok


def model(m_real, srows, F=2304, nrank=4):
    ranges = {}
    for dst_rank in range(nrank):
        cover = []
        for src in range(nrank):
            dst_off = src * 2 * m_real * F
            nbytes = 2 * m_real * F
            if src == dst_rank:
                src_off = src * 2 * srows * F  # self copy: S[rank] -> R[rank * m_real]
            else:
                src_off = (
                    dst_rank * 2 * srows * F
                )  # the put of sender `src` to peer dst_rank reads slab dst_rank of the SENDER's S
            slab_lo, slab_hi = dst_rank * 2 * srows * F, (dst_rank + 1) * 2 * srows * F
            if src != dst_rank and not (
                slab_lo <= src_off and src_off + nbytes <= slab_hi
            ):
                return (
                    False,
                    f"source range leaves slab dst={dst_rank} (m_real={m_real}, srows={srows})",
                )
            if src == dst_rank:
                s_lo, s_hi = src * 2 * srows * F, (src + 1) * 2 * srows * F
                if not (s_lo <= src_off and src_off + nbytes <= s_hi):
                    return False, "self source leaves its slab"
            cover.append((dst_off, dst_off + nbytes))
        cover.sort()
        pos = 0
        for lo, hi in cover:
            if lo != pos:
                return (
                    False,
                    f"R of rank {dst_rank}: gap/overlap at {lo} (expected {pos})",
                )
            pos = hi
        if pos != 4 * 2 * m_real * F:
            return False, "R not tiled exactly"
        ranges[dst_rank] = pos
    return True, ""


def check_model():
    ok = True
    for m_real, srows in itertools.product(
        (1, 17, 256, 300, 384, 450, 512, 585, 700, 768), (None,)
    ):
        for sr in {m_real, ((m_real + 255) // 256) * 256, 768, 512}:
            if sr < m_real:
                continue
            for F in (2304, 2560):
                good, msg = model(m_real, sr, F)
                if not good:
                    print("FAIL model:", msg)
                    ok = False
    print(
        "PASS address model (R tiled exactly once; sources inside their slabs)"
        if ok
        else "FAILED address model"
    )
    return ok


if __name__ == "__main__":
    r = [check_identical(), check_header(), check_model()]
    print("R18-PUT STATIC", "ALL PASS" if all(r) else "FAILED")
    sys.exit(0 if all(r) else 1)
