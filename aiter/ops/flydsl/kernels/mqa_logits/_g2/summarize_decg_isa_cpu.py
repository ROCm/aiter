# SPDX-License-Identifier: MIT
"""ROUND-DECG ISA summary (inputs: container_decg_cpu.sh isa).

(a) default env (QSA_DECODE_GROUP unset) final == champion c45b24e0 for every
    specialization incl. the 64x16 decode routes, int32 AND pos64 (ISA sha256
    -- raw and .loc-normalized --, kernel name, resources).
(b) opt-in decode-group kernels: name, resources, mnemonic census, pos64 deltas;
    no s_barrier, TDM gathers present, LDS = 4 x wave_cols x pitch.
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
a_rows, b_rows = [], []
for spec in specs("specs.txt"):
    for meta in ("i32", "qp64"):
        b, d = load("base", f"{spec}-{meta}"), load("dflt", f"{spec}-{meta}")
        ident = same(b, d)
        ok &= ident
        a_rows.append({"spec": spec, "meta": meta, "default_eq_base": ident,
                       "kernel": d["kernel_name"], "isa_sha256": d["isa"]["sha256"]})
LDS = {("bf16", 32): 4 * 32 * 272, ("bf16", 64): 4 * 64 * 272,
       ("fp8", 32): 4 * 32 * 144, ("fp8", 64): 4 * 64 * 144}
for spec in specs("decg-specs.txt"):
    i32, p64 = load("decg", f"{spec}-i32"), load("decg", f"{spec}-qp64")
    dtype = "fp8" if "fp8" in spec else "bf16"
    wc = 64 if spec.endswith("w64") else 32
    mi, mp = i32["isa"]["mnemonics"], p64["isa"]["mnemonics"]
    dm = {m: mp.get(m, 0) - mi.get(m, 0) for m in set(mi) | set(mp)}
    row = {
        "spec": spec, "kernel_i32": i32["kernel_name"], "kernel_pos64": p64["kernel_name"],
        "name_ok": ("_decg_" in i32["kernel_name"] and i32["kernel_name"].endswith(f"_w{wc}_flydsl_tdm")
                    and p64["kernel_name"] == i32["kernel_name"] + "_qp64"),
        "pos64_zero_copy": p64["qp_launch"]["zero_copy"] and p64["qp_launch"]["dtype"] == "torch.int32",
        "grid_scalars_i32": i32["scalars"][:2],
        "instr": [i32["isa"]["instructions"], p64["isa"]["instructions"]],
        "vgpr": [i32["resources"]["next_free_vgpr"], p64["resources"]["next_free_vgpr"]],
        "sgpr": [i32["resources"]["next_free_sgpr"], p64["resources"]["next_free_sgpr"]],
        "lds": [i32["resources"]["group_segment_fixed_size"], p64["resources"]["group_segment_fixed_size"]],
        "scratch": [i32["resources"]["private_segment_fixed_size"], p64["resources"]["private_segment_fixed_size"]],
        "s_barrier": sum(v for m, v in mi.items() if m.startswith("s_barrier")),
        "tensor_load": sum(v for m, v in mi.items() if m.startswith("tensor_load")),
        "wmma": sum(v for m, v in mi.items() if "wmma" in m),
        "s_buffer_load": sum(v for m, v in mi.items() if m.startswith("s_buffer_load")),
        "s_wait_tensorcnt": mi.get("s_wait_tensorcnt", 0),
        "s_wait_dscnt": mi.get("s_wait_dscnt", 0),
        "v_readfirstlane": mi.get("v_readfirstlane_b32", 0),
        "pos64_mnemonic_delta": {m: d for m, d in sorted(dm.items()) if d},
    }
    good = (row["name_ok"] and row["pos64_zero_copy"] and row["s_barrier"] == 0
            and row["tensor_load"] > 0 and row["wmma"] > 0 and row["scratch"] == [0, 0]
            and row["lds"][0] == LDS[(dtype, wc)] and row["lds"][1] == LDS[(dtype, wc)])
    row["ok"] = good
    ok &= good
    b_rows.append(row)
out = {"all_ok": ok, "a_default_vs_base": a_rows, "b_decg": b_rows}
(root / "isa-summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("(a) default env == base c45b24e0")
for r in a_rows:
    print("  %-20s %-5s %-5s %s %s" % (r["spec"], r["meta"], r["default_eq_base"], r["isa_sha256"][:16], r["kernel"]))
print("(b) decode-group kernels (i32, qp64)")
for r in b_rows:
    print("  %-14s ok=%s instr %s vgpr %s sgpr %s lds %s scratch %s barrier=%d tensor_load=%d wmma=%d "
          "s_buffer_load=%d tcnt=%d dscnt=%d rfl=%d %s" % (
              r["spec"], r["ok"], r["instr"], r["vgpr"], r["sgpr"], r["lds"], r["scratch"],
              r["s_barrier"], r["tensor_load"], r["wmma"], r["s_buffer_load"], r["s_wait_tensorcnt"],
              r["s_wait_dscnt"], r["v_readfirstlane"], r["kernel_i32"]))
print("ALL_DECG_ISA_OK" if ok else "DECG_ISA_MISMATCH")
sys.exit(0 if ok else 1)
