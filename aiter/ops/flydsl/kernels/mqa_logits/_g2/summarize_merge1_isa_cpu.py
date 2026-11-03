# SPDX-License-Identifier: MIT
"""ROUND-MERGE1 ISA summary (inputs: container_merge1_cpu.sh isa).

(a)  merge opt-out (QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0) == NATIVE64 for every
     specialization, int32 AND pos64 (ISA sha256, kernel name, resources).
(a2) merge default == NATIVE64 for every specialization except bf16 r16 TDM,
     which becomes the ``_nobar`` kernel by default.
(b)  merge default draft / nobar int32 kernels == DRAFT-branch opt-in ISA.
(c)  pos64 deltas (qp64 - int32) of the default draft and nobar kernels.
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
    return (x["isa"]["sha256"] == y["isa"]["sha256"] and x["kernel_name"] == y["kernel_name"]
            and x["resources"] == y["resources"])


def delta(i32, p64):
    mi, mp = i32["isa"]["mnemonics"], p64["isa"]["mnemonics"]
    dm = {m: mp.get(m, 0) - mi.get(m, 0) for m in set(mi) | set(mp)}
    return {
        "kernel_i32": i32["kernel_name"], "kernel_pos64": p64["kernel_name"],
        "pos64_name_ok": p64["kernel_name"] == i32["kernel_name"] + "_qp64",
        "pos64_zero_copy": p64["qp_launch"]["zero_copy"] and p64["qp_launch"]["dtype"] == "torch.int32"
        and p64["qp_launch"]["shape"] == [2 * i32["shape"]["rows"]],
        "instr_i32": i32["isa"]["instructions"], "instr_pos64": p64["isa"]["instructions"],
        "d_instr": p64["isa"]["instructions"] - i32["isa"]["instructions"],
        "vgpr": [i32["resources"]["next_free_vgpr"], p64["resources"]["next_free_vgpr"]],
        "sgpr": [i32["resources"]["next_free_sgpr"], p64["resources"]["next_free_sgpr"]],
        "lds": [i32["resources"]["group_segment_fixed_size"], p64["resources"]["group_segment_fixed_size"]],
        "scratch": [i32["resources"]["private_segment_fixed_size"], p64["resources"]["private_segment_fixed_size"]],
        "mnemonic_delta": {m: d for m, d in sorted(dm.items()) if d},
    }


ok = True
a_rows, a2_rows, b_rows, c_rows = [], [], [], []
for spec in specs("specs.txt"):
    for meta in ("i32", "qp64"):
        n, o, d = (load(p, f"{spec}-{meta}") for p in ("n64", "optout", "dflt"))
        ident = same(n, o)
        ok &= ident
        a_rows.append({"spec": spec, "meta": meta, "optout_eq_native64": ident,
                       "kernel": o["kernel_name"], "isa_sha256": o["isa"]["sha256"]})
        expect_change = spec == "bf16-r16-tdm"
        dflt_same = same(n, d)
        good = (not dflt_same and d["kernel_name"].replace("_nobar", "") == n["kernel_name"]) \
            if expect_change else dflt_same
        ok &= good
        a2_rows.append({"spec": spec, "meta": meta, "default_eq_native64": dflt_same,
                        "expected": "nobar" if expect_change else "identical", "ok": good,
                        "kernel_default": d["kernel_name"]})
for spec in specs("draft-specs.txt"):
    br, di, dq = load("draftbr", f"{spec}-i32"), load("dflt", f"{spec}-i32"), load("dflt", f"{spec}-qp64")
    ident = same(br, di)
    ok &= ident
    b_rows.append({"spec": spec, "default_eq_draft_branch_optin": ident, "kernel": di["kernel_name"],
                   "isa_sha256": di["isa"]["sha256"], "draft_branch_sha256": br["isa"]["sha256"]})
    row = {"spec": spec, **delta(di, dq)}
    ok &= row["pos64_name_ok"] and row["pos64_zero_copy"]
    c_rows.append(row)
# the bf16 r16 TDM nobar default (spec list) must equal the dedicated nobar case
ok &= load("dflt", "bf16-r16-tdm-i32")["isa"]["sha256"] == load("dflt", "nobar-bf16-r16-tdm-i32")["isa"]["sha256"]
out = {"all_ok": ok, "a_optout_vs_native64": a_rows, "a2_default_vs_native64": a2_rows,
       "b_default_vs_draft_branch": b_rows, "c_pos64_deltas": c_rows}
(root / "isa-summary.json").write_text(json.dumps(out, indent=2) + "\n")
print("(a) opt-out == NATIVE64")
for r in a_rows:
    print("  %-20s %-5s %-5s %s %s" % (r["spec"], r["meta"], r["optout_eq_native64"], r["isa_sha256"][:16], r["kernel"]))
print("(a2) default vs NATIVE64")
for r in a2_rows:
    print("  %-20s %-5s same=%-5s expected=%-9s ok=%s %s" % (r["spec"], r["meta"], r["default_eq_native64"], r["expected"], r["ok"], r["kernel_default"]))
print("(b) default int32 == DRAFT branch opt-in")
for r in b_rows:
    print("  %-20s %-5s %s %s" % (r["spec"], r["default_eq_draft_branch_optin"], r["isa_sha256"][:16], r["kernel"]))
print("(c) pos64 deltas (default draft/nobar)")
for r in c_rows:
    print("  %-20s instr %d->%d (%+d) vgpr %s sgpr %s lds %s scratch %s name_ok=%s zero_copy=%s %s" % (
        r["spec"], r["instr_i32"], r["instr_pos64"], r["d_instr"], r["vgpr"], r["sgpr"], r["lds"],
        r["scratch"], r["pos64_name_ok"], r["pos64_zero_copy"],
        " ".join(f"{m}:{d:+d}" for m, d in r["mnemonic_delta"].items())))
print("ALL_MERGE1_ISA_OK" if ok else "MERGE1_ISA_MISMATCH")
sys.exit(0 if ok else 1)
