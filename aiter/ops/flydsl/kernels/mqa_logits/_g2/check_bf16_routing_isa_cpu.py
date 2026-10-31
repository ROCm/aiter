# SPDX-License-Identifier: MIT
"""Check bf16 routing in real no-device final ISA dumps.

Required directories: auto/raw/tdm. Optional: off/tdm64/r64/r64raw/r64tdm.
When off exists, require routed default=P8/off=single-tile and p2/p4/p8
production-wrapper AST evidence in compile JSON/log plus selected final ISA.
Each contains compile.log and isa/<kernel-name>/21_final_isa.s.
Auto r16 uses c32 TDM (34,816 B LDS, 97 VGPR); r32 TDM is unchanged.
Only r64* directories enable QSA_BF16_R64=1; raw overrides its runtime routing.
The r64raw directory may explicitly compile raw r64 for ISA evidence only.
--base compares unchanged layouts; new auto/tdm r16 c32 has no base counterpart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from check_tdm_isa_cpu import check, _resources

PREFIX = "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4"
MODES = ("auto", "raw", "tdm", "off", "tdm64", "r64", "r64raw", "r64tdm")
R64_MODES = ("r64", "r64raw", "r64tdm")


def _expected(mode, rows):
    assert mode in MODES and rows in (16, 32, 64), (mode, rows)
    assert rows != 64 or mode in R64_MODES, ("r64 requires QSA_BF16_R64=1", mode)
    tdm = mode not in ("raw", "r64raw")
    cols = 32 if (tdm and rows == 16 and mode != "tdm64") else 64
    return tdm, cols, f"{PREFIX}_r{rows}_c{cols}_flydsl" + ("_tdm" if tdm else "")


def _normalized_sha(path):
    """SHA256 of instructions/directives, ignoring comments and .loc/.file debug lines."""
    keep = []
    for line in path.read_text().splitlines():
        text = line.split(";", 1)[0].split("//", 1)[0].rstrip()
        if not text.strip() or text.strip().startswith((".loc", ".file")):
            continue
        keep.append(text)
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()


def _raw_record(path, mode, rows):
    source = path.read_text(encoding="utf-8")
    markers = {marker: len(re.findall(r"\b" + marker + r"\b", source))
               for marker in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                              "s_barrier_signal", "s_barrier_wait", "buffer_load_b128",
                              "v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid")}
    for marker in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                   "s_barrier_signal", "s_barrier_wait", "v_readfirstlane_b32",
                   "v_max_i64", "ds_load_b128_addtid"):
        assert markers[marker] == 0, (mode, rows, markers)
    lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", source)
    assert lds == ["0"], (mode, rows, lds)
    assert markers["buffer_load_b128"] > 0, (mode, rows, markers)
    return {"status": "PASS", "path": str(path), "lds_bytes": 0, "markers": markers,
            "resources": _resources(source),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_routed_selection(results, mode):
    """Cross-check actual production-wrapper AST selection, compilation and ISA."""
    directory = Path(results) / mode
    data = json.loads((directory / "compile_only.json").read_text(encoding="utf-8"))
    routed = data.get("routing", [])
    assert len(routed) == 1, (mode, "missing wrapper routing evidence")
    rec = routed[0]
    log = (directory / "compile.log").read_text(encoding="utf-8")
    logged = re.findall(r"^\[ROUTED\] (.+)$", log, re.MULTILINE)
    assert len(logged) == 1 and json.loads(logged[0]) == rec, (mode, "routing JSON/log mismatch")
    assert rec["compiled"] is True, (mode, rec)
    assert rec["evidence"] == "production wrapper AST via test_bf16_routing_cpu helpers"
    assert re.fullmatch(r"[0-9a-f]{64}", rec["source_sha256"])
    assert (rec["rows"], rec["requests"], rec["columns"], rec["pointer"]) == (512, 1, 1024, 0x1000)
    env = rec["environment"]
    if mode == "auto":
        assert all(value is None for value in env.values()), ("default not truly unset", env)
    elif mode == "off":
        assert env["QSA_BF16_PERSIST"] == "0", env
        assert all(value is None for key, value in env.items() if key != "QSA_BF16_PERSIST"), env
    elif mode in ("p2", "p4", "p8"):
        assert env["QSA_BF16_PERSIST"] == "1" and env["QSA_PERSIST_COLS"] == mode[1:], env
        assert env["QSA_BF16_K_LOAD"] is None and env["QSA_G2_ACCESS"] is None, env
    elif mode in ("raw", "tdm"):
        assert env["QSA_BF16_PERSIST"] == "0" and env["QSA_BF16_K_LOAD"] == mode, env
    else:
        raise AssertionError(("unknown routed mode", mode))
    chunks = 8 if mode == "auto" else int(mode[1:]) if mode.startswith("p") else 0
    selected = "persist" if chunks else "raw" if mode == "raw" else "tdm"
    name = f"{PREFIX}_r32_c64_flydsl" + ("" if selected == "raw" else "_tdm")
    if chunks:
        name += f"_persist{chunks}"
    assert rec["rows_per_block"] == 32 and rec["selected_k_load"] == selected, (mode, rec)
    assert rec["persist_chunks"] == chunks and rec["cols_per_block"] == 64 * (chunks or 1), (mode, rec)
    assert rec["grid"] == [16 // (chunks or 1), 16], (mode, rec)
    assert rec["kernel_name"] == name, (mode, rec, name)
    assert rec["kernel_function"] == ("kernel_r32_persist" if chunks else "kernel_r32"), (mode, rec)
    assert re.findall(r"^\[COMPILED\] bf16_wrapper_prefill kernel=(\S+)\s*$", log, re.MULTILINE) == [name]
    names = {p.parent.name for p in (directory / "isa").glob("*/21_final_isa.s")}
    assert name in names, (mode, "routed ISA missing", name)
    persistent = {n for n in names if "_persist" in n}
    assert persistent == ({name} if chunks else set()), (mode, persistent, name)
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--base", type=Path, help="base-commit results dir (same layout)")
    args = parser.parse_args()
    records = []
    modes = [m for m in MODES if (args.results / m / "compile.log").exists()]
    assert modes[:3] == ["auto", "raw", "tdm"], modes
    for mode in modes:
        log = (args.results / mode / "compile.log").read_text()
        assert "[FAILED]" not in log, (mode, "failed specialization")
        if mode not in R64_MODES:
            assert not re.search(r"\[COMPILED\] bf16_wmma_r64\b", log), (mode, "r64 without opt-in")
            assert not any((args.results / mode / "isa").glob(f"{PREFIX}_r64_*")), (mode, "r64 ISA without opt-in")
        specializations = [(16, "bf16_wmma"), (32, "bf16_wmma_r32")]
        if mode in R64_MODES:
            specializations.append((64, "bf16_wmma_r64"))
        for rows, dispatch in specializations:
            tdm, cols, name = _expected(mode, rows)
            compiled = re.findall(r"^\[COMPILED\] " + dispatch + r" kernel=(\S+)\s*$", log, re.MULTILINE)
            assert compiled == [name], (mode, dispatch, compiled, name)
            path = args.results / mode / "isa" / name / "21_final_isa.s"
            rec = check(path, rows, cols) if tdm else _raw_record(path, mode, rows)
            rec.update(mode=mode, rows_per_block=rows, cols_per_block=cols, dispatch=dispatch,
                       kernel_name=name, selected_k_load="tdm" if tdm else "raw",
                       r64_opt_in=mode in R64_MODES)
            records.append(rec)
    by = {(rec["mode"], rec["rows_per_block"]): rec for rec in records}
    assert by["auto", 16]["sha256"] == by["tdm", 16]["sha256"]
    assert by["auto", 32]["sha256"] == by["tdm", 32]["sha256"]
    out = {"scope": "CPU-only stock-FlyDSL compile routing; no GPU dispatch",
           "auto_r16_c32_equals_forced_tdm_r16_c32_isa": True,
           "auto_r32_equals_forced_tdm_r32_isa": True,
           "r64_absent_without_opt_in": True}
    for mode, reference in (("tdm64", "tdm"), ("r64", "auto"),
                            ("r64raw", "raw"), ("r64tdm", "tdm")):
        if mode in modes:
            for rows in (16, 32):
                if mode == "tdm64" and rows == 16:
                    continue
                assert by[mode, rows]["sha256"] == by[reference, rows]["sha256"], (mode, reference, rows)
    if "r64" in modes and "r64tdm" in modes:
        assert by["r64", 64]["sha256"] == by["r64tdm", 64]["sha256"]
        out["opt_in_auto_r64_equals_forced_tdm_r64_isa"] = True
    if "r64raw" in modes:
        out["raw_overrides_r64_for_r16_r32_isa"] = True
    if args.base:
        same = {}
        for (mode, rows), rec in sorted(by.items()):
            if rows == 16 and rec["cols_per_block"] == 32:
                continue  # New default TDM c32: no base counterpart.
            base_mode = {"tdm64": "tdm", "r64": "auto", "r64raw": "raw", "r64tdm": "tdm"}.get(mode, mode)
            base_name = f"{PREFIX}_r{rows}_c64_flydsl" + ("_tdm" if rec["selected_k_load"] == "tdm" else "")
            base_path = args.base / base_mode / "isa" / base_name / "21_final_isa.s"
            if rows == 64 and not base_path.exists():
                continue  # Pre-r64 base commits have no explicit r64 evidence.
            base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
            norm_base = _normalized_sha(base_path)
            norm_new = _normalized_sha(Path(rec["path"]))
            key = f"{mode}_r{rows}"
            same[key] = {"base": base_sha, "new": rec["sha256"],
                         "byte_identical": base_sha == rec["sha256"],
                         "identical": norm_base == norm_new, "normalized_sha256": norm_new}
            assert same[key]["identical"], (key, same[key])
        out["isa_identical_to_base"] = same
    if (args.results / "off" / "compile.log").exists():
        routed_modes = ("auto", "off", "raw", "tdm", "p2", "p4", "p8")
        out["routed_wrapper_prefill"] = [check_routed_selection(args.results, m) for m in routed_modes]
        out["default_auto_routes_persist8_off_routes_single_tile"] = True
    out["kernels"] = records
    text = json.dumps(out, indent=2)
    print(text)
    (args.results / "routing-isa.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
