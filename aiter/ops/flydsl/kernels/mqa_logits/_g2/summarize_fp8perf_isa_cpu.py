#!/usr/bin/env python3
"""ROUND-FP8PERF ISA summary (no device). Usage: summarize_fp8perf_isa_cpu.py RESULTS/isa

Gates (FP8PERF_ISA_OK only if all hold):
 (a) every kernel that is neither fp8 draft nor fp8 decode-group (all bf16 incl. COLS=32/64,
     fp8 r32 tdm cvtfast / raw, fp8 scalar, legacy-env kernels) is identical to base
     5fb8740e: raw sha, normalized sha, name, resources;
 (b) fp8 draft / decode-group kernels: no inline asm, no scratch, WMMA / tensor_load_to_lds /
     LDS equal to the same-shape base kernel; instr / VGPR / v_nop / s_delay_alu reported;
 (c) WMMA hazard scan (kernel-wiki mi400-wmma-hazards B1/A1): 0 violations in every
     kernel compiled from the final source except the QSA_FP8_CVT_ASM=1 opt-out;
 (d) routing / names: default decode fp8 = w64 (named as base COLS=64), bf16 = w32;
     COLS=32 forces fp8 w32; opt-in schedules carry _s<variant>; QSA_DRAFT_FP8_SCHED=base
     and QSA_DECG_FP8_SCHED=base reproduce the 5fb8740e instruction streams (modulo name);
 (e) QSA_FP8_CVT_ASM=1: byte-identical to base's asm opt-out (schedule knobs ignored).
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmma_hazard_scan_cpu import scan  # noqa: E402

INSTR = re.compile(r"^[a-z_][a-z0-9_.]*(\s|$)")
ROOT = Path(sys.argv[1])
fails = []


def rec(phase, case):
    return json.loads((ROOT / phase / case / "dispatch.json").read_text())


def lines(r):
    return Path(r["isa"]["path"]).read_text().splitlines()


def stream(r):
    out = []
    for ln in lines(r):
        ln = ln.strip()
        if ln and INSTR.match(ln) and not ln.startswith("."):
            out.append(ln.replace(r["kernel_name"], "<K>"))
    return out


def ident(b, d):
    return (b["isa"]["sha256"] == d["isa"]["sha256"]
            and b["isa"]["normalized_sha256"] == d["isa"]["normalized_sha256"]
            and b["kernel_name"] == d["kernel_name"] and b["resources"] == d["resources"])


def changed(name):
    return "_fp8_draft_" in name or "_fp8_decg_" in name


def brief(r, hz=True):
    mn = r["isa"]["mnemonics"]
    out = {"kernel": r["kernel_name"], "sha256": r["isa"]["sha256"][:16],
           "instr": r["isa"]["instructions"], "vgpr": r["resources"]["next_free_vgpr"],
           "sgpr": r["resources"]["next_free_sgpr"], "lds": r["resources"]["group_segment_fixed_size"],
           "scratch": r["resources"]["private_segment_fixed_size"],
           "v_nop": mn.get("v_nop", 0), "s_delay_alu": mn.get("s_delay_alu", 0),
           "s_nop": mn.get("s_nop", 0), "wmma": mn.get("v_wmma_f32_16x16x32_bf16", 0),
           "tensor_load_to_lds": mn.get("tensor_load_to_lds", 0),
           "s_wait_loadcnt": mn.get("s_wait_loadcnt", 0),
           "asm_blocks": sum(1 for ln in lines(r) if ";;#ASMSTART" in ln)}
    out["instr_excl_nop_delay"] = out["instr"] - out["v_nop"] - out["s_delay_alu"]
    if hz:
        s = scan(r["isa"]["path"])
        out["hazards"] = s["violations"]
        out["hazard_detail"] = s["detail"][:4]
    return out


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


specs = [ln.split("|")[0] for ln in (ROOT / "specs.txt").read_text().splitlines() if ln.strip()]
extra = [ln.split("|")[:2] for ln in (ROOT / "extra-specs.txt").read_text().splitlines() if ln.strip()]
ev = {"identity": [], "changed": [], "variants": {}, "hazards": {}, "asm": [], "routing": []}
final_phases = {"dflt", "legacy", "var"}

# (a) / (b) default env
for spec in specs:
    for m in ("i32", "qp64"):
        b, d = rec("base", f"{spec}-{m}"), rec("dflt", f"{spec}-{m}")
        if changed(b["kernel_name"]) or changed(d["kernel_name"]):
            bb, dd = brief(b), brief(d)
            ev["changed"].append({"case": f"{spec}-{m}", "base": bb, "final": dd})
            check(dd["asm_blocks"] == 0 and dd["scratch"] == 0, f"(b) asm/scratch {spec}-{m}")
            if "_decg_" in d["kernel_name"] and "_w64_" in d["kernel_name"]:
                ref = rec("base", f"cols64-fp8-{m}")      # same w64 shape in base
                check(ref["kernel_name"] == d["kernel_name"], f"(d) w64 name {spec}-{m}")
            else:
                ref = b
                check(ref["kernel_name"] == d["kernel_name"], f"(b) name {spec}-{m}")
            rb = brief(ref, hz=False)
            for key in ("wmma", "tensor_load_to_lds", "lds"):
                check(rb[key] == dd[key], f"(b) {key} {spec}-{m}: {rb[key]} vs {dd[key]}")
        else:
            ok = ident(b, d)
            ev["identity"].append({"case": f"{spec}-{m}", "kernel": d["kernel_name"], "identical": ok})
            check(ok, f"(a) identity {spec}-{m}")
# extra phases
for ph, spec in extra:
    for m in ("i32", "qp64"):
        case = f"{spec}-{m}"
        r = rec(ph, case)
        if ph == "legacy":
            ok = ident(rec("base", case), r)
            ev["identity"].append({"case": f"legacy:{case}", "kernel": r["kernel_name"], "identical": ok})
            check(ok, f"(a) legacy identity {case}")
        if ph == "var":
            ev["variants"][case] = brief(r)
# bf16 COLS pins
for m in ("i32", "qp64"):
    ok = ident(rec("base", f"cols64-bf16-{m}"), rec("var", f"cols64-bf16-{m}"))
    ev["identity"].append({"case": f"cols64-bf16-{m}", "identical": ok}); check(ok, f"(a) cols64-bf16-{m}")
    ok = ident(rec("base", f"decode-bf16-64x16-{m}"), rec("var", f"cols32-bf16-{m}"))
    ev["identity"].append({"case": f"cols32-bf16-{m}", "identical": ok}); check(ok, f"(a) cols32-bf16-{m}")
    # (d) routing + base-schedule reproduction
    dname = rec("dflt", f"decode-fp8-64x16-{m}")["kernel_name"]
    check("_w64_" in dname and not re.search(r"_s[a-z]+$", dname), f"(d) fp8 decode default w64 {dname}")
    check("_w32_" in rec("dflt", f"decode-bf16-64x16-{m}")["kernel_name"], "(d) bf16 decode w32")
    check("_w32_" in rec("var", f"cols32-fp8-{m}")["kernel_name"], "(d) COLS=32 fp8 w32")
    for var, ref, tag in (("draft-tdm-sbase", ("base", "draft-fp8-tdm"), "_sbase"),
                          ("draft-raw-sbase", ("base", "draft-fp8-raw"), "_sbase"),
                          ("draft-tdm-w64-sbase", ("base", "draft-tdm-w64"), "_sbase"),
                          ("decg-w64-sbase", ("base", "cols64-fp8"), "_sbase"),
                          ("decg-w32-sbase", ("base", "decode-fp8-64x16"), "_sbase")):
        v, b = rec("var", f"{var}-{m}"), rec(ref[0], f"{ref[1]}-{m}")
        same = stream(v) == stream(b) and v["resources"] == b["resources"]
        ev["routing"].append({"case": f"{var}-{m}", "name": v["kernel_name"],
                              "stream_equals_5fb8740e": same})
        check(v["kernel_name"] == b["kernel_name"] + tag, f"(d) name {var}-{m}")
        check(same, f"(d) base schedule reproduces 5fb8740e {var}-{m}")
    for var, tag in (("draft-tdm-sqpre", "_sqpre"), ("draft-tdm-skall", "_skall"),
                     ("draft-raw-sqpre", "_sqpre"), ("draft-raw-skall", "_skall")):
        check(rec("var", f"{var}-{m}")["kernel_name"].endswith(tag), f"(d) tag {var}-{m}")
    # (e) asm opt-out
    for spec in ("fp8-r32-tdm", "draft-fp8-tdm", "draft-fp8-raw", "decode-fp8-64x16"):
        ok = ident(rec("baseasm", f"{spec}-{m}"), rec("asm", f"{spec}-{m}"))
        ev["asm"].append({"case": f"{spec}-{m}", "identical_to_5fb8740e_asm": ok,
                          "hazards_latent": scan(rec("asm", f"{spec}-{m}")["isa"]["path"])["violations"]})
        check(ok, f"(e) asm identity {spec}-{m}")
    check(ident(rec("asm", f"decode-fp8-64x16-{m}"), rec("asm", f"decode-fp8-sched-ignored-{m}")),
          f"(e) asm decode ignores sched/COLS {m}")
    check(ident(rec("asm", f"draft-fp8-tdm-{m}"), rec("asm", f"draft-fp8-sched-ignored-{m}")),
          f"(e) asm draft ignores sched {m}")
# (c) hazards in every final-source kernel (except asm)
for ph in sorted(final_phases):
    for dj in sorted((ROOT / ph).glob("*/dispatch.json")):
        r = json.loads(dj.read_text())
        s = scan(r["isa"]["path"])
        ev["hazards"][f"{ph}/{r['case']}"] = {"wmma": s["wmma"], "violations": s["violations"]}
        check(s["violations"] == 0, f"(c) hazards {ph}/{r['case']}: {s['violations']}")

ev["fails"] = fails
(ROOT / "fp8perf_isa_evidence.json").write_text(json.dumps(ev, indent=1) + "\n")


def row(label, x):
    return (f"| {label} | {x['kernel'][-48:]} | {x['instr']} | {x['instr_excl_nop_delay']} | {x['vgpr']} | "
            f"{x['sgpr']} | {x['v_nop']} | {x['s_delay_alu']} | {x['s_wait_loadcnt']} | {x['wmma']} | "
            f"{x.get('hazards', '-')} |")


print("| case | kernel | instr | instr-nop-delay | VGPR | SGPR | v_nop | s_delay_alu | wait_loadcnt | WMMA | hazards |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c in ev["changed"]:
    if c["case"].endswith("-i32"):
        print(row("base " + c["case"], c["base"]))
        print(row("final " + c["case"], c["final"]))
for k, x in ev["variants"].items():
    if k.endswith("-i32"):
        print(row("var " + k, x))
print(f"identity: {sum(1 for x in ev['identity'] if x['identical'])}/{len(ev['identity'])}")
print(f"hazard-scanned final kernels: {len(ev['hazards'])}, violations: "
      f"{sum(v['violations'] for v in ev['hazards'].values())}")
print(f"asm opt-out identical: {sum(1 for x in ev['asm'] if x['identical_to_5fb8740e_asm'])}/{len(ev['asm'])}")
for f in fails:
    print("FAIL", f)
print("FP8PERF_ISA_OK" if not fails else "FP8PERF_ISA_FAIL")
sys.exit(1 if fails else 0)
