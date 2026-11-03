# SPDX-License-Identifier: MIT
"""ROUND-DECG2 ISA summary (inputs: container_decg_cpu.sh isa).

(a) final with QSA_DECODE_GROUP=0 == champion c45b24e0 default env, every spec,
    int32 AND pos64 (raw + .loc-normalized ISA sha256, kernel name, resources).
(b) final default env == opt-in DECG bbe25c33 with QSA_DECODE_GROUP=all, every
    spec (the decode-routed specs run the decg w32 kernels; the others must also
    equal the base).
(c) COLS=64: bf16 final == bbe25c33 w64; fp8 final routes to the w32 kernel
    (== bbe25c33 fp8 w32) -- fp8 w64 GPU-incorrect (root run 2026-11-03), disabled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])


def specs(name):
    return [ln.split("|")[0] for ln in (root / name).read_text().splitlines() if ln.strip()]


def load(phase, case):
    return json.loads((root / phase / case / "dispatch.json").read_text())


def same(x, y):
    return (x["isa"]["sha256"] == y["isa"]["sha256"]
            and x["isa"]["normalized_sha256"] == y["isa"]["normalized_sha256"]
            and x["kernel_name"] == y["kernel_name"] and x["resources"] == y["resources"])


ok = True
rows = []
for spec in specs("specs.txt"):
    for meta in ("i32", "qp64"):
        case = f"{spec}-{meta}"
        b, o, p, d = (load(ph, case) for ph in ("base", "off", "prev", "dflt"))
        decg = "_decg_" in d["kernel_name"]
        r = {"spec": spec, "meta": meta, "off_eq_base": same(o, b), "dflt_eq_prev_all": same(d, p),
             "dflt_kernel": d["kernel_name"], "dflt_is_decg": decg,
             "dflt_isa_sha256": d["isa"]["sha256"], "base_kernel": b["kernel_name"]}
        r["dflt_eq_base"] = same(d, b)
        good = r["off_eq_base"] and r["dflt_eq_prev_all"] and (decg or r["dflt_eq_base"])
        if decg:
            good &= d["kernel_name"].split("_qp64")[0].endswith("_w32_flydsl_tdm")
            r.update(instr=d["isa"]["instructions"], vgpr=d["resources"]["next_free_vgpr"],
                     sgpr=d["resources"]["next_free_sgpr"],
                     lds=d["resources"]["group_segment_fixed_size"],
                     scratch=d["resources"]["private_segment_fixed_size"])
        r["ok"] = good
        ok &= good
        rows.append(r)
extra = []
for meta in ("i32", "qp64"):
    fb, pb = load("dflt", f"cols64-bf16-{meta}"), load("prev", f"cols64-bf16-{meta}")
    ff = load("dflt", f"cols64-fp8-{meta}")
    pf = load("prev", f"decode-fp8-64x16-{meta}")
    e = {"meta": meta,
         "bf16_cols64_kernel": fb["kernel_name"], "bf16_cols64_eq_prev_w64": same(fb, pb),
         "bf16_is_w64": "_w64_" in fb["kernel_name"],
         "fp8_cols64_kernel": ff["kernel_name"], "fp8_cols64_is_w32": "_w32_" in ff["kernel_name"],
         "fp8_cols64_eq_prev_fp8_w32": same(ff, pf)}
    e["ok"] = all(v for k, v in e.items() if isinstance(v, bool))
    ok &= e["ok"]
    extra.append(e)
out = {"all_ok": ok, "specs": rows, "cols64": extra}
(root / "isa-summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("spec meta off==base dflt==prev(all) dflt==base decg dflt_kernel")
for r in rows:
    print("  %-20s %-5s %-5s %-5s %-5s %-5s %s %s" % (
        r["spec"], r["meta"], r["off_eq_base"], r["dflt_eq_prev_all"], r["dflt_eq_base"],
        r["dflt_is_decg"], r["dflt_isa_sha256"][:16], r["dflt_kernel"]))
for e in extra:
    print("  cols64", json.dumps(e))
print("ALL_DECG2_ISA_OK" if ok else "DECG2_ISA_MISMATCH")
sys.exit(0 if ok else 1)
