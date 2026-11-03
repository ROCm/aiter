#!/usr/bin/env python3
"""ROUND-W64FIX ISA summary (no device). Usage: summarize_w64fix_isa_cpu.py RESULTS/isa

(a) default env final == base a78c285c (raw sha, normalized sha, kernel name, resources)
    for 18 specs x {i32, qp64};
(b) QSA_DECODE_GROUP_COLS=64 without the debug knob still builds the fp8 *w32* kernel
    (== default decode-fp8-64x16);
(c) fp8 w64 bisect variants: kernel names ``..._w64_flydsl_tdm[_qp64]_dbg<v>``, resources,
    instruction counts and the WMMA hazard scan (kernel-wiki mi400-wmma-hazards B1/A1).
    nop4 and intrin must have 0 violations; asis / tdmwait are expected to keep them.
Prints a JSON document and W64FIX_ISA_OK / W64FIX_ISA_FAIL.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmma_hazard_scan_cpu import scan  # noqa: E402


def rec(root, phase, case):
    return json.loads((root / phase / case / "dispatch.json").read_text())


def main(root):
    root = Path(root)
    specs = [ln.split("|")[0] for ln in (root / "specs.txt").read_text().split("\n") if ln.strip()]
    ok, out = True, {"identity": {}, "variants": {}, "scans": {}}
    for spec in specs:
        for meta in ("i32", "qp64"):
            case = f"{spec}-{meta}"
            b, d = rec(root, "base", case), rec(root, "dflt", case)
            same = (b["isa"]["sha256"] == d["isa"]["sha256"]
                    and b["isa"]["normalized_sha256"] == d["isa"]["normalized_sha256"]
                    and b["kernel_name"] == d["kernel_name"] and b["resources"] == d["resources"])
            out["identity"][case] = {"equal": same, "kernel": d["kernel_name"],
                                     "sha256": d["isa"]["sha256"][:16]}
            ok &= same
    for meta in ("i32", "qp64"):
        off = rec(root, "dflt", f"cols64-fp8-off-{meta}")
        ref = rec(root, "dflt", f"decode-fp8-64x16-{meta}")
        same = off["isa"]["sha256"] == ref["isa"]["sha256"] and "_w32_" in off["kernel_name"]
        out["identity"][f"cols64-fp8-off-{meta}"] = {"equal_to_default_w32": same,
                                                    "kernel": off["kernel_name"]}
        ok &= same
    for v in ("asis", "nop4", "intrin", "tdmwait"):
        for meta in ("i32", "qp64"):
            r = rec(root, "var", f"w64-{v}-{meta}")
            s = scan(r["isa"]["path"])
            name_ok = r["kernel_name"].endswith(f"_dbg{v}") and "_fp8_decg_" in r["kernel_name"] \
                and "_w64_" in r["kernel_name"]
            mn = r["isa"]["mnemonics"]
            out["variants"][f"{v}-{meta}"] = {
                "kernel": r["kernel_name"], "sha256": r["isa"]["sha256"][:16],
                "instructions": r["isa"]["instructions"], "resources": r["resources"],
                "wmma": mn.get("v_wmma_f32_16x16x32_bf16", 0),
                "tensor_load_to_lds": mn.get("tensor_load_to_lds", 0),
                "s_wait_tensorcnt": mn.get("s_wait_tensorcnt", 0),
                "v_nop": mn.get("v_nop", 0), "pk8": mn.get("v_cvt_scale_pk8_bf16_fp8", 0),
                "hazards": {k: s[k] for k in ("violations", "asm_violations", "compiler_violations")},
                "name_ok": name_ok}
            ok &= name_ok and r["resources"]["private_segment_fixed_size"] == 0
            if v in ("nop4", "intrin"):
                ok &= s["violations"] == 0
    for case in ("decode-fp8-64x16", "decode-bf16-64x16", "draft-fp8-tdm", "draft-fp8-raw",
                 "fp8-r32-tdm"):
        r = rec(root, "dflt", f"{case}-i32")
        s = scan(r["isa"]["path"])
        out["scans"][f"dflt/{case}-i32"] = {"kernel": r["kernel_name"],
                                            **{k: s[k] for k in ("wmma", "violations", "asm_violations")}}
    for meta in ("i32",):
        r = rec(root, "var", f"bf16-w64-{meta}")
        s = scan(r["isa"]["path"])
        out["scans"][f"var/bf16-w64-{meta}"] = {"kernel": r["kernel_name"],
                                               **{k: s[k] for k in ("wmma", "violations", "asm_violations")}}
    print(json.dumps(out, indent=1))
    print("W64FIX_ISA_OK" if ok else "W64FIX_ISA_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
