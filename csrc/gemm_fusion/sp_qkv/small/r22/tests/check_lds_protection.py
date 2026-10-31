#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""r11: source-level LDS hazard check for head_row (device/ran_kernel.cu). NOT a proof about codegen; a path-sensitive model of the SOURCE.
For each configuration (SM_RMS_ORDER x SM_AMAX_REG) and each head path h in {Q=5, K=16, V=17, IQ=18, IK=19} the head_row body is preprocessed (#if/#elif/#else on the config macros),
`if (h == N) {` blocks are resolved for the path, `return;` ends the path, and the sequence of LDS ops on s_row / s_part (R/W) and barriers (syncThreads) is extracted.
Two CONSECUTIVE calls (all ordered pairs of paths; s_row/s_part are only used inside head_row, callers sync between phases) are concatenated and checked per array, conservatively treating
every lane as distinct: no RAW, WAR or WAW between two ops of the same array without a barrier in between. Prints PASS/FAIL per configuration and the LDS op counts (the default order 3/4 + AMAX_REG=1 must have ZERO LDS ops).
--negative-control strips the trailing barrier for a legacy configuration and MUST report a hazard (proves the checker bites).
Usage: check_lds_protection.py [path/to/ran_kernel.cu] [--negative-control]"""

import itertools
import os
import re
import sys


def load_head_row(path):
    with open(path, encoding="utf-8") as _f:
        t = _f.read().replace("\r\n", "\n")
    i = t.index("SM_INL void head_row")
    j = t.index("// ---------------- SM-T1-r6 helpers")
    return t[i:j].split("\n")


def preprocess(lines, macros):
    out, stack = [], []  # stack of (active, taken, parent_active)

    def ev(expr):
        e = re.sub(r"//.*", "", expr).strip()
        e = e.replace("||", " or ").replace("&&", " and ")
        e = re.sub(r"!(?!=)", " not ", e)
        e = re.sub(
            r"\b([A-Z][A-Z0-9_]+)\b", lambda m: str(macros.get(m.group(1), 0)), e
        )
        return bool(eval(e))

    for ln in lines:
        s = ln.strip()
        m = re.match(r"#\s*(if|elif|else|endif)\b(.*)", s)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind == "if":
                par = all(a for a, _, _ in stack) if stack else True
                v = ev(rest) if par else False
                stack.append((v, v, par))
            elif kind == "elif":
                _a, taken, par = stack.pop()
                v = (not taken) and par and ev(rest)
                stack.append((v, taken or v, par))
            elif kind == "else":
                _a, taken, par = stack.pop()
                stack.append(((not taken) and par, True, par))
            else:
                stack.pop()
            continue
        if all(a for a, _, _ in stack) if stack else True:
            out.append(re.sub(r"//.*", "", ln))
    return out


def ops_for_path(lines, h):
    ops, active = [], [True]
    for ln in lines:
        m = re.search(r"if \(h == (\d+)u\)\s*\{\s*$", ln)
        opener_cond = (int(m.group(1)) == h) if m else True
        cur = active[-1]
        if cur:
            if "syncThreads()" in ln:
                ops.append(("B", None))
            for arr in ("s_row", "s_part"):
                for mm in re.finditer(r"\b" + arr + r"\[[^\]]*\](\s*=(?!=))?", ln):
                    # a store only if this occurrence is the LHS of '=' (the regex captured it) and not preceded by '&'
                    pre = ln[: mm.start()]
                    kind = (
                        "W" if mm.group(1) and not pre.rstrip().endswith("&") else "R"
                    )
                    ops.append((kind, arr))
            if re.search(r"\breturn\s*;", ln):
                return ops
        for _ in range(ln.count("{")):
            active.append(cur and opener_cond if m and _ == ln.count("{") - 1 else cur)
        for _ in range(ln.count("}")):
            if len(active) > 1:
                active.pop()
    return ops


def check_seq(seq):
    st = {"s_row": set(), "s_part": set()}
    for kind, arr in seq:
        if kind == "B":
            st = {"s_row": set(), "s_part": set()}
            continue
        seen = st[arr]
        if kind == "W" and ("R" in seen or "W" in seen):
            return f"WAR/WAW on {arr}"
        if kind == "R" and "W" in seen:
            return f"RAW on {arr}"
        seen.add(kind)
    return None


def run(path, negative):
    src = load_head_row(path)
    names = {5: "Q", 16: "K", 17: "V", 18: "IQ", 19: "IK"}
    bad_total = 0
    configs = [(o, a) for o in (0, 1, 2, 3, 4, 13, 14) for a in (0, 1)]
    if negative:
        configs = [(0, 0)]
    for order, amax_reg in configs:
        legacy = 0 if order in (3, 4) else 1
        tail = legacy or not amax_reg
        if negative:
            tail = 0
        macros = {
            "SM_RMS_ORDER": order,
            "SM_AMAX_REG": amax_reg,
            "SM_LDS_LEGACY": legacy,
            "SM_HEAD_TAIL_SYNC": int(bool(tail)),
            "SM_RSQ_BUILTIN": 1,
            "SM_ROT_NEG": 1,
            "SM_ROPE_FMA": 3,
            "SM_ZSIGN": 1,
        }
        lines = preprocess(src, macros)
        per = {h: ops_for_path(lines, h) for h in names}
        nops = sum(1 for h in per for k, _ in per[h] if k in "RW")
        nbar = sum(1 for h in per for k, _ in per[h] if k == "B")
        errs = []
        for h1, h2 in itertools.product(names, names):
            e = check_seq(per[h1] + per[h2])
            if e:
                errs.append(f"{names[h1]}->{names[h2]}: {e}")
        status = "PASS" if not errs else "FAIL"
        bad_total += bool(errs)
        print(
            "order {:<2d} amax_reg {:d}: LDS ops {:<3d} barriers {:<2d} (summed over the 5 paths) -> {} {}".format(
                order, amax_reg, nops, nbar, status, "; ".join(errs[:3]) if errs else ""
            )
        )
        if order in (3, 4) and amax_reg and (nops or nbar):
            print(
                "  UNEXPECTED: default configuration must have zero LDS ops and zero barriers"
            )
            bad_total += 1
    if negative:
        print(
            "NEGATIVE CONTROL: hazard {}".format(
                "REPORTED (checker bites)"
                if bad_total
                else "NOT reported (checker is blind!)"
            )
        )
        return 0 if bad_total else 1
    return 1 if bad_total else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    p = (
        args[0]
        if args
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "device", "ran_kernel.cu"
        )
    )
    sys.exit(run(p, "--negative-control" in sys.argv))
