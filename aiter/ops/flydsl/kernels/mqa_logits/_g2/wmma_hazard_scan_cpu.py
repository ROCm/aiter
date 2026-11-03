#!/usr/bin/env python3
"""ROUND-W64FIX: static WMMA -> VALU / WMMA -> WMMA data-hazard scan of a final ISA.

Kernel-wiki ``mi400-wmma-hazards`` (SPG §4.6.12.1, IMAGE-VERIFIED) -- MI450 software
requirements (*not* hardware interlocks) after a dense ``WMMA_*BF16``:

  B1  VALU whose vdst overlaps the WMMA's Matrix A/B (WAR), or whose vdst overlaps
      Matrix D (WAW), or that reads Matrix D (RAW): **4** V_NOP / independent VALU;
  A1  XDL WMMA whose Matrix A/B overlaps the previous WMMA's Matrix D (RAW): **5**.

"V_NOP" means a V_NOP *or another unrelated VALU instruction* in between. LLVM's
GCNHazardRecognizer pads compiler-generated instructions; instructions emitted by
inline asm (``;;#ASMSTART`` blocks) are opaque to it. This scan walks each WMMA's
straight-line successors (stops at a branch / s_endpgm), counts intervening VALU
instructions (v_* other than the second instruction itself), and reports every
second instruction found inside the window, tagging whether it is inline asm.

Usage: wmma_hazard_scan_cpu.py ISA.s [--json OUT] [--fail-on-any]
Exit 0 always unless --fail-on-any and a violation was found.
"""
import json
import re
import sys

REG = re.compile(r"\bv\[(\d+):(\d+)\]|\bv(\d+)\b")
B_WINDOW = 4          # B1: dense bf16 WMMA -> VALU
A_WINDOW = 5          # A1: dense bf16 WMMA -> WMMA (A/B RAW on D)


def regs(text):
    out = set()
    for m in REG.finditer(text):
        if m.group(3) is not None:
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def parse(path):
    insts, in_asm = [], False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for no, raw in enumerate(fh, 1):
            line = raw.split(";", 1)[0].strip() if not raw.strip().startswith(";;#ASM") else raw.strip()
            if line.startswith(";;#ASMSTART"):
                in_asm = True
                continue
            if line.startswith(";;#ASMEND"):
                in_asm = False
                continue
            if not line or line.startswith(".") or line.endswith(":"):
                continue
            op, _, rest = line.partition(" ")
            ops = [o.strip() for o in rest.split(",")] if rest else []
            insts.append({"line": no, "op": op, "ops": ops, "asm": in_asm, "text": line})
    return insts


def is_valu(i):
    return i["op"].startswith("v_")


def is_wmma(i):
    return i["op"].startswith("v_wmma") or i["op"].startswith("v_swmmac")


def defs_uses(i):
    ops = i["ops"]
    if not ops:
        return set(), set()
    if i["op"].startswith("v_cmp") or i["op"].startswith("v_readfirstlane") or \
            i["op"].startswith("v_readlane"):
        return set(), regs(",".join(ops))
    d = regs(ops[0])
    return d, regs(",".join(ops[1:]))


def scan(path):
    insts = parse(path)
    found = []
    nwmma = 0
    for k, w in enumerate(insts):
        if not is_wmma(w):
            continue
        nwmma += 1
        D, A, B = regs(w["ops"][0]), regs(w["ops"][1]), regs(w["ops"][2])
        between = 0
        for j in range(k + 1, len(insts)):
            n = insts[j]
            if n["op"].startswith(("s_cbranch", "s_branch", "s_endpgm", "s_setpc")):
                break
            if not is_valu(n):
                continue
            if between >= A_WINDOW:
                break
            dd, uu = defs_uses(n)
            kinds = []
            if is_wmma(n):
                nA, nB = regs(n["ops"][1]), regs(n["ops"][2])
                if (nA | nB) & D and between < A_WINDOW:
                    kinds.append("A1:wmma-AB-RAW-on-D")
            elif between < B_WINDOW:
                if dd & (A | B):
                    kinds.append("B1:WAR-vdst-on-AB")
                if dd & D:
                    kinds.append("B1:WAW-vdst-on-D")
                if uu & D:
                    kinds.append("B1:RAW-src-on-D")
            for kind in kinds:
                found.append({"wmma_line": w["line"], "wmma": w["text"], "line": n["line"],
                              "inst": n["text"], "asm": n["asm"], "valu_between": between,
                              "kind": kind})
            between += 1
    return {"isa": path, "wmma": nwmma, "violations": len(found),
            "asm_violations": sum(1 for f in found if f["asm"]),
            "compiler_violations": sum(1 for f in found if not f["asm"]),
            "detail": found}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    out = scan(argv[0])
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    print(f"WMMA_HAZARD_SCAN wmma={out['wmma']} violations={out['violations']} "
          f"asm={out['asm_violations']} compiler={out['compiler_violations']} isa={argv[0]}")
    for f in out["detail"][:12]:
        print(f"  L{f['wmma_line']} {f['wmma']}\n    -> L{f['line']} (+{f['valu_between']} VALU) "
              f"{'ASM ' if f['asm'] else ''}{f['inst']}  [{f['kind']}]")
    if "--fail-on-any" in argv and out["violations"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
