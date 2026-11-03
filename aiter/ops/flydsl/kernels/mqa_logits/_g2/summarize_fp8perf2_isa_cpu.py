#!/usr/bin/env python3
"""ROUND-FP8PERF2 ISA summary (no device). Usage: summarize_fp8perf2_isa_cpu.py RESULTS/isa2

Base = 2e20f23b (ROUND-FP8PERF, defaults draft kpipe / decg kpipe). Final defaults:
draft fp8 qpre, decode-group fp8 base. Gates (FP8PERF2_ISA_OK only if all hold):
 (a) every default-env kernel that is neither fp8 draft nor fp8 decode-group (all bf16,
     fp8 r32 / scalar ...), legacy env, bf16 COLS=32/64: byte-identical to 2e20f23b
     (raw sha, normalized sha, name, resources);
 (b) default fp8 draft == 2e20f23b QSA_DRAFT_FP8_SCHED=qpre and default fp8 decode-group ==
     2e20f23b QSA_DECG_FP8_SCHED=base: full ISA text identical after replacing the kernel
     name, same resources; final name == ref name minus its _sqpre / _sbase tag (untagged);
 (c) old defaults now opt-in: QSA_DRAFT_FP8_SCHED=kpipe / QSA_DECG_FP8_SCHED=kpipe == the
     2e20f23b default kernels modulo the new _skpipe tag; kall/base draft opt-ins
     byte-identical to 2e20f23b (same names);
 (d) QSA_FP8_CVT_ASM=1 opt-out byte-identical to 2e20f23b's, schedule knobs ignored;
 (e) WMMA hazard scan (B1/A1, zero tolerance): 0 in every final-source kernel except asm.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmma_hazard_scan_cpu import scan  # noqa: E402

ROOT = Path(sys.argv[1])
fails = []
MS = ("i32", "qp64")


def rec(phase, case):
    return json.loads((ROOT / phase / case / "dispatch.json").read_text())


def text(r):
    return Path(r["isa"]["path"]).read_text().replace(r["kernel_name"], "<K>")


def ident(b, d):
    return (b["isa"]["sha256"] == d["isa"]["sha256"]
            and b["isa"]["normalized_sha256"] == d["isa"]["normalized_sha256"]
            and b["kernel_name"] == d["kernel_name"] and b["resources"] == d["resources"])


def same_mod_name(ref, d, tag):
    return (text(ref) == text(d) and ref["resources"] == d["resources"]
            and ref["kernel_name"] == d["kernel_name"] + tag
            and not re.search(r"_s(qpre|kall|kpipe|base)$", d["kernel_name"]))


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


def changed(name):
    return "_fp8_draft_" in name or "_fp8_decg_" in name


def brief(r):
    mn = r["isa"]["mnemonics"]
    return {"kernel": r["kernel_name"], "sha256": r["isa"]["sha256"][:16],
            "instr": r["isa"]["instructions"], "vgpr": r["resources"]["next_free_vgpr"],
            "sgpr": r["resources"]["next_free_sgpr"], "v_nop": mn.get("v_nop", 0),
            "s_delay_alu": mn.get("s_delay_alu", 0)}


specs = [ln.split("|")[0] for ln in (ROOT / "specs.txt").read_text().splitlines() if ln.strip()]
ev = {"identity": [], "new_default": [], "opt_in": [], "asm": [], "hazards": {}}

for spec in specs:
    for m in MS:
        b, d = rec("base", f"{spec}-{m}"), rec("dflt", f"{spec}-{m}")
        if changed(d["kernel_name"]) or changed(b["kernel_name"]):
            tag = "_sqpre" if "_draft_" in d["kernel_name"] else "_sbase"
            ok = same_mod_name(rec("baseref", f"{spec}-{m}"), d, tag)
            ev["new_default"].append({"case": f"{spec}-{m}", "final": brief(d),
                                      "ref_2e20f23b": rec("baseref", f"{spec}-{m}")["kernel_name"],
                                      "identical_mod_name": ok})
            check(ok, f"(b) {spec}-{m} != 2e20f23b{tag}")
        else:
            ok = ident(b, d)
            ev["identity"].append({"case": f"{spec}-{m}", "kernel": d["kernel_name"], "identical": ok})
            check(ok, f"(a) identity {spec}-{m}")
for m in MS:
    for case in ("draft-tdm-w64", "cols32-fp8"):
        d = rec("var", f"{case}-{m}")
        tag = "_sqpre" if "_draft_" in d["kernel_name"] else "_sbase"
        ok = same_mod_name(rec("baseref", f"{case}-{m}"), d, tag)
        ev["new_default"].append({"case": f"{case}-{m}", "final": brief(d), "identical_mod_name": ok})
        check(ok, f"(b) {case}-{m}")
    for case in ("legacy-decode-bf16", "legacy-decode-fp8", "legacy-draft-bf16", "legacy-draft-fp8"):
        ok = ident(rec("base", f"{case}-{m}"), rec("legacy", f"{case}-{m}"))
        ev["identity"].append({"case": f"legacy:{case}-{m}", "identical": ok})
        check(ok, f"(a) legacy {case}-{m}")
    for case in ("cols64-bf16", "cols32-bf16"):
        ok = ident(rec("base", f"{case}-{m}"), rec("var", f"{case}-{m}"))
        ev["identity"].append({"case": f"{case}-{m}", "identical": ok})
        check(ok, f"(a) {case}-{m}")
    # (c) old defaults -> _skpipe opt-ins
    for var, ref in (("draft-tdm-skpipe", "draft-fp8-tdm"), ("draft-raw-skpipe", "draft-fp8-raw"),
                     ("draft-tdm-w64-skpipe", "draft-tdm-w64"), ("decg-w64-skpipe", "decode-fp8-64x16"),
                     ("decg-w32-skpipe", "cols32-fp8")):
        v, b = rec("var", f"{var}-{m}"), rec("base", f"{ref}-{m}")
        ok = (text(v) == text(b) and v["resources"] == b["resources"]
              and v["kernel_name"] == b["kernel_name"] + "_skpipe")
        ev["opt_in"].append({"case": f"{var}-{m}", "kernel": v["kernel_name"],
                             "equals_2e20f23b_default_mod_name": ok})
        check(ok, f"(c) {var}-{m}")
    for var in ("draft-tdm-skall", "draft-tdm-sbase", "draft-raw-skall", "draft-raw-sbase"):
        v, b = rec("var", f"{var}-{m}"), rec("base", f"{var}-{m}")
        ok = ident(b, v) and v["kernel_name"].endswith(var[var.rindex("-"):].replace("-", "_"))
        ev["opt_in"].append({"case": f"{var}-{m}", "kernel": v["kernel_name"], "byte_identical": ok})
        check(ok, f"(c) {var}-{m}")
    # (d) asm opt-out
    for spec in ("fp8-r32-tdm", "draft-fp8-tdm", "draft-fp8-raw", "decode-fp8-64x16"):
        ok = ident(rec("baseasm", f"{spec}-{m}"), rec("asm", f"{spec}-{m}"))
        ev["asm"].append({"case": f"{spec}-{m}", "identical": ok})
        check(ok, f"(d) asm {spec}-{m}")
    check(ident(rec("asm", f"decode-fp8-64x16-{m}"), rec("asm", f"decode-fp8-sched-ignored-{m}")),
          f"(d) asm decode ignores sched {m}")
    check(ident(rec("asm", f"draft-fp8-tdm-{m}"), rec("asm", f"draft-fp8-sched-ignored-{m}")),
          f"(d) asm draft ignores sched {m}")
for ph in ("dflt", "legacy", "var"):
    for dj in sorted((ROOT / ph).glob("*/dispatch.json")):
        r = json.loads(dj.read_text())
        s = scan(r["isa"]["path"])
        ev["hazards"][f"{ph}/{r['case']}"] = {"wmma": s["wmma"], "violations": s["violations"]}
        check(s["violations"] == 0, f"(e) hazards {ph}/{r['case']}: {s['violations']}")

ev["fails"] = fails
(ROOT / "fp8perf2_isa_evidence.json").write_text(json.dumps(ev, indent=1) + "\n")
for x in ev["new_default"]:
    if x["case"].endswith("-i32"):
        f = x["final"]
        print(f"| {x['case']} | {f['kernel'][-44:]} | {f['instr']} | {f['vgpr']} | {f['v_nop']} | "
              f"{x['identical_mod_name']} |")
print(f"identity: {sum(x['identical'] for x in ev['identity'])}/{len(ev['identity'])}")
print(f"new defaults == 2e20f23b opt-in: {sum(x['identical_mod_name'] for x in ev['new_default'])}"
      f"/{len(ev['new_default'])}")
print(f"opt-ins ok: {len(ev['opt_in']) - sum(1 for f in fails if f.startswith('(c)'))}/{len(ev['opt_in'])}")
print(f"asm identical: {sum(x['identical'] for x in ev['asm'])}/{len(ev['asm'])}")
print(f"hazard-scanned final kernels: {len(ev['hazards'])}, violations: "
      f"{sum(v['violations'] for v in ev['hazards'].values())}")
for f in fails:
    print("FAIL", f)
print("FP8PERF2_ISA_OK" if not fails else "FP8PERF2_ISA_FAIL")
sys.exit(1 if fails else 0)
