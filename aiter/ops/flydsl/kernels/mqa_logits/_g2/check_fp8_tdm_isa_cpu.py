# SPDX-License-Identifier: MIT
"""Strict stock dev886 fp8 auto/raw/tdm ISA routing gate; no GPU imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_bf16_routing_isa_cpu import MODES
from check_tdm_isa_cpu import _resources


def check(path: Path, tdm: bool) -> dict:
    text = path.read_text(encoding="utf-8")
    names = ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
             "s_barrier_signal", "s_barrier_wait", "buffer_load_b64",
             "buffer_load_b128", "v_readfirstlane_b32", "v_max_i64",
             "ds_load_b128_addtid")
    counts = {n: len(re.findall(r"\b" + n + r"\b", text)) for n in names}
    lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", text)
    assert lds == ["9216" if tdm else "0"], (path, lds)
    for n in ("v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid"):
        assert counts[n] == 0, (path, n, counts)
    waits_zero = len(re.findall(r"\bs_wait_tensorcnt\s+(?:0x0|0)\b", text))
    if tdm:
        assert counts["tensor_load_to_lds"] == 2, (path, counts)
        assert counts["ds_load_b128"] >= 32, (path, counts)
        assert waits_zero == 1, (path, counts)
        assert counts["s_barrier_signal"] == counts["s_barrier_wait"] == 1, counts
        # Element size1, pad_enable1, interval4, amount3, multicast0.
        assert re.search(r"\bs_mov_b32\s+s\d+,\s*0x7100000\b", text), "fp8 config"
        assert re.search(r"\bv_[^\n]*\b0x90\b", text), "144B consumer pitch"
        tensor_lines = [line.strip() for line in text.splitlines()
                        if line.strip().startswith("tensor_load_to_lds ")]
        for line in tensor_lines:
            assert re.fullmatch(r"tensor_load_to_lds s\[\d+:\d+\], s\[\d+:\d+\], s\[\d+:\d+\], s\[\d+:\d+\]", line), line
        # Clauses carry the following count+1 memory instructions. LLVM emits
        # their members contiguously; assert no tensor in any emitted clause.
        instructions = [l.strip() for l in text.splitlines()
                        if re.match(r"^\s+(?:s_|v_|buffer_|global_|flat_|ds_|tensor_)", l)]
        for i, line in enumerate(instructions):
            match = re.match(r"s_clause\s+(0x[0-9a-f]+|\d+)", line)
            if match:
                n = int(match.group(1), 0) + 1
                assert not any("tensor_" in x for x in instructions[i + 1:i + n + 1]), (path, line)
    else:
        for n in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                  "s_barrier_signal", "s_barrier_wait"):
            assert counts[n] == 0, (path, n, counts)
        assert counts["buffer_load_b64"] > 0, (path, counts)
    resources = _resources(text)
    for name in ("next_free_vgpr", "next_free_sgpr"):
        values = re.findall(r"\.amdhsa_" + name + r"\s+(\d+)", text)
        assert len(values) == 1 and int(values[0]) > 0, (path, name, values)
        resources[name] = int(values[0])
    return {"status": "PASS", "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "lds_bytes": int(lds[0]), "stride_bytes": 144 if tdm else None,
            "descriptor_config_hex": "0x07100000" if tdm else None,
            "encoded_pad_interval": 4 if tdm else None,
            "encoded_pad_amount": 3 if tdm else None,
            "public_fp8_pad_interval": 128 if tdm else None,
            "public_fp8_pad_amount": 16 if tdm else None,
            "markers": counts, "wait_zero_count": waits_zero,
            "barrier_pairs": counts["s_barrier_signal"], "resources": resources,
            "max_tensor_issues_per_wave": 2 if tdm else 0,
            "tensor_operands_all_sgpr": True, "tensor_in_clauses": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--cvt", choices=("current", "fast"), default="fast",
                        help="conversion selection used for these dumps (default: fast)")
    args = parser.parse_args()
    records = []
    prefix = "qsa_paged_mqa_logits_page4_gfx1250_fp8_wmma_D128_H4_P4"
    modes = [mode for mode in MODES if (args.results / mode / "compile.log").exists()]
    assert modes[:3] == ["auto", "raw", "tdm"], modes
    for mode in modes:
        log = (args.results / mode / "compile.log").read_text()
        assert "[FAILED]" not in log, (mode, "failed specialization")
        for rows, dispatch in ((16, "fp8_wmma"), (32, "fp8_wmma_r32")):
            # Shared raw mode overrides both dtypes, including raw + R64=1.
            tdm = rows == 32 and mode not in ("raw", "r64raw")
            fast = tdm and args.cvt == "fast"
            name = (f"{prefix}_r{rows}_c64_flydsl" + ("_tdm" if tdm else "")
                    + ("_cvtfast" if fast else ""))
            compiled = re.findall(r"^\[COMPILED\] " + dispatch + r" kernel=(\S+)\s*$", log, re.MULTILINE)
            assert compiled == [name], (mode, dispatch, compiled, name)
            path = args.results / mode / "isa" / name / "21_final_isa.s"
            rec = check(path, tdm)
            native = len(re.findall(r"\bv_cvt_scale_pk8_bf16_fp8\b", path.read_text(encoding="utf-8")))
            assert native == (32 if fast else 0), (mode, rows, "conversion routing", native)
            rec.update(mode=mode, rows_per_block=rows, kernel_name=name,
                       selected_k_load="tdm" if tdm else "raw",
                       selected_fp8_cvt="fast" if fast else "current",
                       native_pk8_count=native)
            records.append(rec)
    by = {(r["mode"], r["rows_per_block"]): r for r in records}
    assert by["auto", 16]["sha256"] == by["raw", 16]["sha256"] == by["tdm", 16]["sha256"]
    assert by["auto", 32]["sha256"] == by["tdm", 32]["sha256"]
    for mode in modes:
        assert by[mode, 16]["sha256"] == by["raw", 16]["sha256"], mode
        reference = "raw" if mode in ("raw", "r64raw") else "tdm"
        assert by[mode, 32]["sha256"] == by[reference, 32]["sha256"], mode
    out = {"scope": "CPU-only stock dev886 final ISA; no GPU dispatch or timings",
           "auto_r16_equals_raw_and_forced_tdm_r16_isa": True,
           "auto_r32_equals_forced_tdm_r32_isa": True, "kernels": records}
    result = json.dumps(out, indent=2) + "\n"
    print(result)
    (args.results / "fp8-tdm-isa.json").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
