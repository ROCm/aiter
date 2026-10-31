# SPDX-License-Identifier: MIT
"""Validate real stock-FlyDSL final ISA dumps, without importing GPU libraries.

Usage: python check_tdm_isa_cpu.py --r32 path/to/21_final_isa.s
       python check_tdm_isa_cpu.py --r16 r16.s --r32 r32.s --json evidence.json
This is a compile/layout gate, NOT GPU correctness or performance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def check(path: Path, rows: int) -> dict:
    source = path.read_text(encoding="utf-8")
    counts = {name: len(re.findall(r"\b" + re.escape(name) + r"\b", source))
              for name in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                           "s_barrier_signal", "s_barrier_wait", "v_readfirstlane_b32",
                           "v_max_i64", "ds_load_b128_addtid")}
    waits_zero = len(re.findall(r"\bs_wait_tensorcnt\s+(?:0x0|0)\b", source))
    expected_lds = 17408 if rows == 32 else 69632
    lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", source)
    assert lds == [str(expected_lds)], f"r{rows}: LDS size {lds}, expected {expected_lds}"
    assert counts["tensor_load_to_lds"] >= (2 if rows == 32 else 32), counts
    assert counts["ds_load_b128"] >= (32 if rows == 32 else 128), counts
    assert waits_zero >= (1 if rows == 32 else 4), counts
    assert counts["s_barrier_signal"] >= (1 if rows == 32 else 8), counts
    assert counts["s_barrier_wait"] >= (1 if rows == 32 else 8), counts
    for marker in ("v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid"):
        assert counts[marker] == 0, f"r{rows}: forbidden marker {marker}"
    # Group1 config: bf16 data_size=1, pad_enable=1, encoded interval=5,
    # encoded amount=3. All bits of the immediate, not only a textual pad flag.
    assert re.search(r"\bs_mov_b32\s+s\d+,\s*0x7510000\b", source), "missing padded descriptor config"
    assert re.search(r"\bv_[^\n]*\b0x110\b", source), "missing 272B consumer row stride"
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows_per_block": rows, "lds_bytes": expected_lds, "stride_bytes": 272,
            "descriptor_config_hex": "0x07510000", "pad_enable": 1,
            "encoded_pad_interval": 5, "encoded_pad_amount": 3,
            "public_bf16_pad_interval": 128, "public_bf16_pad_amount": 8,
            "markers": counts, "wait_zero_count": waits_zero, "status": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r16", type=Path)
    parser.add_argument("--r32", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    assert args.r16 or args.r32, "supply at least one final ISA dump"
    results = [check(path, rows) for path, rows in ((args.r16, 16), (args.r32, 32)) if path]
    text = json.dumps({"scope": "CPU-only stock-FlyDSL ISA gate; no GPU dispatch", "kernels": results}, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
