# SPDX-License-Identifier: MIT
"""Validate real stock-FlyDSL final ISA dumps, without importing GPU libraries.

Usage: python check_tdm_isa_cpu.py --r32 path/to/21_final_isa.s
       python check_tdm_isa_cpu.py --r16 r16.s --r32 r32.s --r64 r64.s --json evidence.json
r16 defaults to c32 (34,816 B LDS, 97 VGPR); --r16-cols 64 checks legacy c64.
This is a compile/layout gate, NOT GPU correctness or performance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def _resources(source: str) -> dict:
    """Require final physical register directives and complete YAML metadata.

    YAML counts may precede final allocation (e.g. c32 reports 80 VGPRs there
    but physically uses 97). SGPR spills to VGPRs are reported, not scratch;
    require zero VGPR spills and zero private memory, not zero SGPR spills.
    """
    resources = {}
    for name in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
                 "private_segment_fixed_size"):
        values = re.findall(r"\." + name + r":\s*(\d+)", source)
        assert len(values) == 1, ("missing/ambiguous metadata", name, values)
        key = "metadata_" + name if name in ("vgpr_count", "sgpr_count") else name
        resources[key] = int(values[0])
    for kind in ("vgpr", "sgpr"):
        values = re.findall(r"\.amdhsa_next_free_" + kind + r"\s+(\d+)", source)
        assert len(values) == 1 and int(values[0]) > 0, ("missing/ambiguous physical register count", kind, values)
        resources[kind + "_count"] = int(values[0])
        assert resources["metadata_" + kind + "_count"] > 0, resources
    for name in ("vgpr_spill_count", "private_segment_fixed_size"):
        assert resources[name] == 0, resources
    private = re.findall(r"\.amdhsa_private_segment_fixed_size\s+(\d+)", source)
    assert private == ["0"], ("private segment directive", private)
    return resources


def check(path: Path, rows: int, cols: int | None = None) -> dict:
    cols = (32 if rows == 16 else 64) if cols is None else cols
    assert rows in (16, 32, 64), rows
    assert cols == 64 or (rows == 16 and cols == 32), (rows, cols)
    source = path.read_text(encoding="utf-8")
    counts = {name: len(re.findall(r"\b" + re.escape(name) + r"\b", source))
              for name in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                           "s_barrier_signal", "s_barrier_wait", "v_readfirstlane_b32",
                           "v_max_i64", "ds_load_b128_addtid")}
    waits_zero = len(re.findall(r"\bs_wait_tensorcnt\s+(?:0x0|0)\b", source))
    # r16: four wave-private padded tiles; r32/r64 share one unchanged tile.
    expected_lds = 17408 if rows in (32, 64) else 4 * cols * 272
    min_tensor = 2 if rows in (32, 64) else 4 * (cols // 8)
    min_ds = 64 if rows == 64 else 32 if rows == 32 else 4 * (cols // 16) * 8
    lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", source)
    assert lds == [str(expected_lds)], f"r{rows}: LDS size {lds}, expected {expected_lds}"
    assert counts["tensor_load_to_lds"] >= min_tensor, counts
    assert counts["ds_load_b128"] >= min_ds, counts
    assert waits_zero >= (1 if rows in (32, 64) else 4), counts
    assert counts["s_barrier_signal"] >= (1 if rows in (32, 64) else 8), counts
    assert counts["s_barrier_wait"] >= (1 if rows in (32, 64) else 8), counts
    for marker in ("v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid"):
        assert counts[marker] == 0, f"r{rows}: forbidden marker {marker}"
    # Group1 bf16: data_size=1, pad_enable=1, encoded interval=5, amount=3.
    assert re.search(r"\bs_mov_b32\s+s\d+,\s*0x7510000\b", source), "missing padded descriptor config"
    assert re.search(r"\bv_[^\n]*\b0x110\b", source), "missing 272B consumer row stride"
    tensor_lines = [line.strip() for line in source.splitlines()
                    if line.strip().startswith("tensor_load_to_lds ")]
    for line in tensor_lines:
        assert re.fullmatch(r"tensor_load_to_lds s\[\d+:\d+\], s\[\d+:\d+\], s\[\d+:\d+\], s\[\d+:\d+\]", line), line
    instructions = [line.strip() for line in source.splitlines()
                    if re.match(r"^\s+(?:s_|v_|buffer_|global_|flat_|ds_|tensor_)", line)]
    for i, line in enumerate(instructions):
        match = re.match(r"s_clause\s+(0x[0-9a-f]+|\d+)", line)
        if match:
            n = int(match.group(1), 0) + 1
            assert not any("tensor_" in x for x in instructions[i + 1:i + n + 1]), (path, line)
    if rows in (32, 64):
        assert counts["tensor_load_to_lds"] == 2, counts
        assert waits_zero == counts["s_barrier_signal"] == counts["s_barrier_wait"] == 1, counts
    resources = _resources(source)
    if rows == 16 and cols == 32:
        assert resources["vgpr_count"] == 97, (path, resources)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows_per_block": rows, "cols_per_block": cols, "lds_bytes": expected_lds, "stride_bytes": 272,
            "descriptor_config_hex": "0x07510000", "pad_enable": 1,
            "encoded_pad_interval": 5, "encoded_pad_amount": 3,
            "public_bf16_pad_interval": 128, "public_bf16_pad_amount": 8,
            "markers": counts, "wait_zero_count": waits_zero, "status": "PASS",
            "barrier_pairs": counts["s_barrier_signal"], "resources": resources,
            "tensor_operands_all_sgpr": True, "tensor_in_clauses": 0,
            "max_tensor_issues_per_wave": 2 if rows in (32, 64) else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r16", type=Path)
    parser.add_argument("--r32", type=Path)
    parser.add_argument("--r64", type=Path)
    parser.add_argument("--r16-cols", type=int, default=32, choices=(32, 64))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    assert args.r16 or args.r32 or args.r64, "supply at least one final ISA dump"
    results = [check(path, rows, cols) for path, rows, cols in
               ((args.r16, 16, args.r16_cols), (args.r32, 32, 64), (args.r64, 64, 64)) if path]
    text = json.dumps({"scope": "CPU-only stock-FlyDSL ISA gate; no GPU dispatch", "kernels": results}, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
