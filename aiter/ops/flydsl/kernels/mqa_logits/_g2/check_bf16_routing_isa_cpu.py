# SPDX-License-Identifier: MIT
"""Check auto/raw/tdm specialization routing in real no-device final ISA dumps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from check_tdm_isa_cpu import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    records = []
    prefix = "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4"
    for mode in ("auto", "raw", "tdm"):
        log = (args.results / mode / "compile.log").read_text()
        for rows, dispatch in ((16, "bf16_wmma"), (32, "bf16_wmma_r32")):
            tdm = mode == "tdm" or (mode == "auto" and rows == 32)
            name = f"{prefix}_r{rows}_c64_flydsl" + ("_tdm" if tdm else "")
            assert f"[COMPILED] {dispatch} kernel={name}" in log, (mode, dispatch, name)
            path = args.results / mode / "isa" / name / "21_final_isa.s"
            if tdm:
                rec = check(path, rows)
            else:
                source = path.read_text()
                markers = {marker: len(re.findall(r"\b" + marker + r"\b", source))
                           for marker in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                                          "s_barrier_signal", "s_barrier_wait", "buffer_load_b128")}
                for marker in ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                               "s_barrier_signal", "s_barrier_wait"):
                    assert markers[marker] == 0, (mode, rows, markers)
                lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", source)
                assert lds == ["0"], (mode, rows, lds)
                assert markers["buffer_load_b128"] > 0, (mode, rows, markers)
                rec = {"status": "PASS", "path": str(path), "lds_bytes": 0,
                       "markers": markers, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            rec.update(mode=mode, rows_per_block=rows, dispatch=dispatch, kernel_name=name,
                       selected_k_load="tdm" if tdm else "raw")
            records.append(rec)
    by_mode_rows = {(rec["mode"], rec["rows_per_block"]): rec for rec in records}
    assert by_mode_rows["auto", 16]["sha256"] == by_mode_rows["raw", 16]["sha256"]
    assert by_mode_rows["auto", 32]["sha256"] == by_mode_rows["tdm", 32]["sha256"]
    out = {"scope": "CPU-only stock-FlyDSL compile routing; no GPU dispatch",
           "auto_r16_equals_raw_r16_isa": True, "auto_r32_equals_forced_tdm_r32_isa": True,
           "kernels": records}
    text = json.dumps(out, indent=2)
    print(text)
    (args.results / "routing-isa.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
