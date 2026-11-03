#!/usr/bin/env python3
"""ROUND-W64FIX2 ISA summary (no device). Usage: summarize_w64fix2_isa_cpu.py RESULTS/isa

Gates (W64FIX2_ISA_OK only if all hold):
 (a) every kernel that does not use the pk8 converter (all bf16, fp8 r32 raw,
     fp8 scalar, fp8 r16 legacy; legacy-env decode/draft; bf16 COLS=64) is identical to
     base a78c285c: raw sha, normalized sha, kernel name, resources;
 (b) pk8 kernels (fp8 r32 tdm cvtfast, draft fp8 tdm/raw, decode-group fp8 -- which the
     default env also selects for the fp8-r16-{tdm,raw} 16x4 specs) keep their
     names, WMMA / tensor_load_to_lds counts, LDS, zero scratch, and contain no inline
     asm; instruction / VGPR deltas vs base are reported;
 (c) WMMA hazard scan (kernel-wiki mi400-wmma-hazards B1/A1): 0 violations in every
     default, legacy-env and COLS=64 kernel of the final source;
 (d) COLS=64 fp8 builds the decode-group w64 kernel (no _dbg, no asm);
 (e) QSA_FP8_CVT_ASM=1 opt-out: names gain ``_cvtasm``; the instruction stream equals
     base modulo the kernel name; COLS=64 fp8 falls back to w32.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmma_hazard_scan_cpu import scan  # noqa: E402

# Specs whose default-env kernel uses the pk8 converter (asm opt-out compiled for these).
# fp8-r16-{tdm,raw} (16 rows x 4 requests) also route to the fp8 decode-group kernel in
# the default env, so classification for gates (a)/(b) is by the *base* kernel name.
PK8 = ("fp8-r32-tdm", "draft-fp8-tdm", "draft-fp8-raw", "decode-fp8-64x16")


def uses_pk8(name):
    return "_cvtfast" in name or "_fp8_draft_" in name or "_fp8_decg_" in name
INSTR = re.compile(r"^[a-z_][a-z0-9_.]*(\s|$)")


def rec(root, phase, case):
    return json.loads((root / phase / case / "dispatch.json").read_text())


def stream(r):
    """Instruction lines with the kernel name blanked (for name-only differences)."""
    out = []
    for ln in Path(r["isa"]["path"]).read_text().splitlines():
        ln = ln.strip()
        if ln and INSTR.match(ln) and not ln.startswith("."):
            out.append(ln.replace(r["kernel_name"], "<K>"))
    return out


def ident(b, d):
    return (b["isa"]["sha256"] == d["isa"]["sha256"]
            and b["isa"]["normalized_sha256"] == d["isa"]["normalized_sha256"]
            and b["kernel_name"] == d["kernel_name"] and b["resources"] == d["resources"])


def brief(r, s=None):
    mn = r["isa"]["mnemonics"]
    out = {"kernel": r["kernel_name"], "sha256": r["isa"]["sha256"][:16],
           "instructions": r["isa"]["instructions"], "vgpr": r["resources"]["next_free_vgpr"],
           "sgpr": r["resources"]["next_free_sgpr"], "lds": r["resources"]["group_segment_fixed_size"],
           "scratch": r["resources"]["private_segment_fixed_size"],
           "wmma": mn.get("v_wmma_f32_16x16x32_bf16", 0), "pk8": mn.get("v_cvt_scale_pk8_bf16_fp8", 0),
           "v_nop": mn.get("v_nop", 0), "tensor_load_to_lds": mn.get("tensor_load_to_lds", 0)}
    if s is not None:
        out["hazards"] = {k: s[k] for k in ("wmma", "violations", "asm_violations", "compiler_violations")}
    return out


def main(root):
    root = Path(root)
    specs = [ln.split("|")[0] for ln in (root / "specs.txt").read_text().split("\n") if ln.strip()]
    ok = True
    fails = []
    out = {"identity": {}, "pk8": {}, "hazard_scan": {}, "cols64": {}, "asm_optout": {}}

    def need(cond, what):
        nonlocal ok
        if not cond:
            ok = False
            fails.append(what)

    def hz(phase, case, r):
        s = scan(r["isa"]["path"])
        out["hazard_scan"][f"{phase}/{case}"] = {"kernel": r["kernel_name"], "wmma": s["wmma"],
                                                 "violations": s["violations"],
                                                 "asm_violations": s["asm_violations"]}
        need(s["violations"] == 0, f"hazard {phase}/{case}={s['violations']}")
        return s

    for spec in specs:
        for meta in ("i32", "qp64"):
            case = f"{spec}-{meta}"
            b, d = rec(root, "base", case), rec(root, "dflt", case)
            s = hz("dflt", case, d)
            if uses_pk8(b["kernel_name"]):
                sb = scan(b["isa"]["path"])
                bb, dd = brief(b, sb), brief(d, s)
                bb["asm_blocks"] = Path(b["isa"]["path"]).read_text().count(";;#ASMSTART")
                dd["asm_blocks"] = Path(d["isa"]["path"]).read_text().count(";;#ASMSTART")
                need(dd["asm_blocks"] == 0, f"inline asm left in {case}: {dd['asm_blocks']}")
                same_shape = all(bb[k] == dd[k] for k in ("kernel", "lds", "wmma", "tensor_load_to_lds", "pk8"))
                need(same_shape and dd["scratch"] == 0, f"pk8 shape {case}")
                out["pk8"][case] = {"base": bb, "final": dd,
                                    "delta": {"instructions": dd["instructions"] - bb["instructions"],
                                              "vgpr": dd["vgpr"] - bb["vgpr"],
                                              "sgpr": dd["sgpr"] - bb["sgpr"],
                                              "v_nop": dd["v_nop"] - bb["v_nop"]}}
            else:
                same = ident(b, d)
                out["identity"][case] = {"equal": same, "kernel": d["kernel_name"],
                                         "sha256": d["isa"]["sha256"][:16]}
                need(same, f"identity {case}")
    for spec in ("legacy-decode-bf16", "legacy-decode-fp8", "legacy-draft-bf16", "legacy-draft-fp8",
                 "cols64-bf16"):
        ph = "cols64" if spec.startswith("cols64") else "legacy"
        for meta in ("i32", "qp64"):
            case = f"{spec}-{meta}"
            b, d = rec(root, "base", case), rec(root, ph, case)
            same = ident(b, d)
            out["identity"][f"{ph}/{case}"] = {"equal": same, "kernel": d["kernel_name"],
                                               "sha256": d["isa"]["sha256"][:16]}
            need(same, f"identity {ph}/{case}")
            hz(ph, case, d)
    for meta in ("i32", "qp64"):
        case = f"cols64-fp8-{meta}"
        r = rec(root, "cols64", case)
        s = hz("cols64", case, r)
        name_ok = ("_fp8_decg_" in r["kernel_name"] and "_w64_" in r["kernel_name"]
                   and "_dbg" not in r["kernel_name"] and "_cvtasm" not in r["kernel_name"])
        need(name_ok and r["resources"]["private_segment_fixed_size"] == 0, f"cols64 fp8 {case}")
        w32 = brief(rec(root, "dflt", f"decode-fp8-64x16-{meta}"))
        out["cols64"][case] = {"final": brief(r, s), "vs_default_w32": w32, "name_ok": name_ok}
    for spec in PK8 + ("cols64-fp8",):
        for meta in ("i32", "qp64"):
            case = f"{spec}-{meta}"
            a = rec(root, "asm", case)
            s = scan(a["isa"]["path"])
            ref_case = "decode-fp8-64x16-" + meta if spec == "cols64-fp8" else case
            b = rec(root, "base", ref_case)
            tagged = "_cvtasm" in a["kernel_name"]
            w32 = "_w32_" in a["kernel_name"] if spec == "cols64-fp8" else True
            same_stream = stream(a) == stream(b)
            need(tagged and w32 and same_stream, f"asm opt-out {case}")
            out["asm_optout"][case] = {"kernel": a["kernel_name"], "tagged": tagged, "w32": w32,
                                       "instr_stream_equals_base": same_stream,
                                       "hazards": {k: s[k] for k in ("wmma", "violations", "asm_violations")}}
    out["fails"] = fails
    out["counts"] = {"identity_equal": sum(v["equal"] for v in out["identity"].values()),
                     "identity_total": len(out["identity"]),
                     "hazard_scanned": len(out["hazard_scan"]),
                     "hazard_violations": sum(v["violations"] for v in out["hazard_scan"].values())}
    print(json.dumps(out, indent=1))
    print("W64FIX2_ISA_OK" if ok else "W64FIX2_ISA_FAIL " + "; ".join(fails))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
