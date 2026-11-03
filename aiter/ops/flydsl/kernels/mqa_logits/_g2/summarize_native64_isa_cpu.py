# SPDX-License-Identifier: MIT
"""ROUND-NATIVE64 ISA summary: int32 identity (base vs final) + POS64 deltas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
specs = [line.split("|")[0] for line in (root / "specs.txt").read_text().splitlines() if line.strip()]


def load(phase, case):
    return json.loads((root / phase / case / "dispatch.json").read_text())


rows, ok = [], True
for spec in specs:
    b, f, p = load("base", spec + "-i32"), load("final", spec + "-i32"), load("final", spec + "-qp64")
    ident = (b["isa"]["sha256"] == f["isa"]["sha256"] and b["kernel_name"] == f["kernel_name"]
             and b["resources"] == f["resources"])
    pos64_ok = (p["kernel_name"] == f["kernel_name"] + "_qp64" and p["qp_launch"]["zero_copy"]
                and p["qp_launch"]["dtype"] == "torch.int32")
    ok &= ident and pos64_ok
    dm = {m: p["isa"]["mnemonics"].get(m, 0) - f["isa"]["mnemonics"].get(m, 0)
          for m in set(p["isa"]["mnemonics"]) | set(f["isa"]["mnemonics"])}
    dm = {m: d for m, d in sorted(dm.items()) if d}
    rows.append({
        "spec": spec, "kernel": f["kernel_name"], "int32_isa_identical": ident,
        "isa_sha256": f["isa"]["sha256"], "loc_lines": f["isa"]["loc_lines"],
        "pos64_kernel": p["kernel_name"], "pos64_ok": pos64_ok,
        "instr_i32": f["isa"]["instructions"], "instr_pos64": p["isa"]["instructions"],
        "d_instr": p["isa"]["instructions"] - f["isa"]["instructions"],
        "vgpr_i32": f["resources"]["next_free_vgpr"], "vgpr_pos64": p["resources"]["next_free_vgpr"],
        "sgpr_i32": f["resources"]["next_free_sgpr"], "sgpr_pos64": p["resources"]["next_free_sgpr"],
        "lds_equal": f["resources"]["group_segment_fixed_size"] == p["resources"]["group_segment_fixed_size"],
        "scratch_pos64": p["resources"]["private_segment_fixed_size"],
        "mnemonic_delta": dm,
    })
(root / "isa-summary.json").write_text(json.dumps({"all_ok": ok, "rows": rows}, indent=2) + "\n")
print("%-18s %-5s %7s %7s %5s %9s %9s %s" % ("spec", "ident", "instr", "pos64", "d", "vgpr", "sgpr", "mnemonic delta"))
for r in rows:
    print("%-18s %-5s %7d %7d %+5d %4d->%-4d %4d->%-4d %s" % (
        r["spec"], r["int32_isa_identical"], r["instr_i32"], r["instr_pos64"], r["d_instr"],
        r["vgpr_i32"], r["vgpr_pos64"], r["sgpr_i32"], r["sgpr_pos64"],
        " ".join(f"{m}:{d:+d}" for m, d in r["mnemonic_delta"].items())))
print("ALL_NATIVE64_ISA_OK" if ok else "NATIVE64_ISA_MISMATCH")
sys.exit(0 if ok else 1)
