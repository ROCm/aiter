# SPDX-License-Identifier: MIT
"""Summarize source-matched CPU logs and all 21 real-ISA trace specializations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("verify", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    kernel = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
    kernel_sha = hashlib.sha256(kernel.read_bytes()).hexdigest()
    assert kernel_sha in (args.verify / "source-match.txt").read_text()
    marker_names = ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128",
                    "s_barrier_signal", "s_barrier_wait", "v_readfirstlane_b32",
                    "v_max_i64", "ds_load_b128_addtid")
    records = []
    for mode in ("auto", "raw", "tdm"):
        log = (args.trace / mode / "compile.log").read_text()
        compiled = re.findall(r"\[COMPILED\]\s+(\S+)\s+kernel=(\S+)", log)
        assert len(compiled) == 7, (mode, compiled)
        for dispatch, name in compiled:
            path = args.trace / mode / "isa" / name / "21_final_isa.s"
            text = path.read_text()
            counts = {n: len(re.findall(r"\b" + n + r"\b", text)) for n in marker_names}
            assert all(counts[n] == 0 for n in marker_names[-3:]), (path, counts)
            lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", text)
            assert len(lds) == 1
            resources = {}
            for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
                        "private_segment_fixed_size"):
                match = re.search(r"\." + key + r":\s*(\d+)", text)
                resources[key] = int(match.group(1)) if match else None
            tdm = name.endswith("_tdm")
            pitch = (144 if "fp8" in name else 272) if tdm else None
            records.append({"mode": mode, "dispatch": dispatch, "kernel_name": name,
                            "path": str(path.relative_to(args.trace)).replace("\\", "/"),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "lds_bytes": int(lds[0]), "k_pitch_bytes": pitch,
                            "markers": counts,
                            "wait_zero_count": len(re.findall(r"\bs_wait_tensorcnt\s+(?:0x0|0)\b", text)),
                            "barrier_pairs": counts["s_barrier_signal"], "resources": resources})
    tests = []
    for name in ("test_contract_cpu", "audit_addressing_cpu", "test_q_lds_cpu",
                 "test_k_tdm_cpu", "test_bf16_routing_cpu", "test_fp8_k_tdm_cpu",
                 "test_bf16_bm64_cpu", "trace16", "trace32"):
        path = args.verify / f"{name}.log"
        text = path.read_text()
        assert "FAIL" not in text and ("PASS" in text or re.search(r"\bOK\b", text)), name
        ran = re.search(r"Ran (\d+) tests in ([0-9.]+)s", text)
        tests.append({"name": name, "status": "PASS",
                      "unittest_count": int(ran.group(1)) if ran else None,
                      "unittest_seconds": float(ran.group(2)) if ran else None,
                      "pass_case_count": text.count("[PASS]"),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    bf16_gate = json.loads((args.verify / "bf16-routing-isa.json").read_text())
    fp8_gate = json.loads((args.verify / "fp8-tdm-isa.json").read_text())
    assert all(r["status"] == "PASS" for r in bf16_gate["kernels"] + fp8_gate["kernels"])
    evidence = {
        "status": "PASS_NO_DEVICE",
        "base_commit": "3e3555b30d65249d856f8139142b879539f12e78",
        "kernel_sha256": kernel_sha,
        "scope": "CPU-only byte/address/control-flow tests and real final ISA; no GPU dispatch, hardware correctness or timings",
        "image": "rocm/fw-bringup-mlperf:gfx1250-atom-20261015", "flydsl": "0.3.3.dev886",
        "target_arch": "gfx1250", "no_device": True,
        "remote_stage": "/data/blyu/qsa/work/opt-tdm4-cpu-trace-20261031",
        "remote_trace": "/data/blyu/qsa/work/opt-tdm4-cpu-trace-20261031/results-tdm4.3w6xuI",
        "remote_final_cpu_verify": "/data/blyu/qsa/work/opt-tdm4-cpu-trace-20261031/verify-tdm4.yboNYs",
        "stock_padding_source": {
            "path": "/usr/local/lib/python3.12/dist-packages/flydsl/expr/rocdl/tdm_ops.py",
            "formula": "interval_dw=interval_elems*elem_bits//32; amount_dw=amount_elems*elem_bits//32; encoded_interval=log2(interval_dw)-1; encoded_amount=amount_dw-1",
            "gather_elem_bits": "elem_bytes*8",
            "fp8": {"element_bytes": 1, "interval_elements": 128, "amount_elements": 16,
                    "interval_dwords": 32, "amount_dwords": 4, "encoded_interval": 4,
                    "encoded_amount": 3, "config_hex": "0x07100000"},
            "bf16": {"element_bytes": 2, "interval_elements": 128, "amount_elements": 8,
                     "interval_dwords": 64, "amount_dwords": 4, "encoded_interval": 5,
                     "encoded_amount": 3, "config_hex": "0x07510000"}},
        "geometry": {
            "fp8_r32": {"query_rows": 32, "columns": 64, "heads": 4, "waves": 4,
                        "m16_tiles_per_wave": 2, "k_data_bytes": 8192, "k_pitch_bytes": 144,
                        "lds_bytes": 9216, "gathers_per_wave": 2,
                        "consumer": "aligned row*144+step*32 loads +0/+16, then each lane selects 8B halves before unchanged exact E4M3 conversion",
                        "k_global_bytes_vs_raw_r32": [8192, 32768]},
            "bf16_r64": {"query_rows": 64, "columns": 64, "heads": 4, "waves": 4,
                         "m16_tiles_per_wave": 4, "sequential_pairs_per_wave": 2,
                         "k_data_bytes": 16384, "k_pitch_bytes": 272, "lds_bytes": 17408,
                         "gathers_per_wave": 2, "k_tile_staged_once": True,
                         "k_global_bytes_vs_two_r32_workgroups": [16384, 32768]}},
        "tdm_rules": {"descriptor_and_row_ids": "uniform SGPRs; workgroup_mask=0",
                      "tensor_ops_in_clauses": 0, "completion_before_cross_wave_consume": "tensor_wait(0)+workgroup barrier",
                      "new_tile_overwrites": 0, "new_path_tensor_issues_per_wave": 2,
                      "issue_to_xack_limits": "hardware enforces 3/wave and 6/SIMD; unchanged two-gather producer schedule",
                      "new_path_outstanding_completion_per_wave_upper_bound": 2,
                      "completion_limit": 63, "invalid_pages": "physical page0 retaining c%4; never rely on random-index OOB semantics"},
        "routing": {"fp8_auto_r16_equals_raw_and_forced_tdm_r16_isa": fp8_gate["auto_r16_equals_raw_and_forced_tdm_r16_isa"],
                    "fp8_auto_r32_equals_tdm_r32_isa": fp8_gate["auto_r32_equals_forced_tdm_r32_isa"],
                    "bf16_auto_r16_equals_raw_r16_isa": bf16_gate["auto_r16_equals_raw_r16_isa"],
                    "bf16_auto_r32_equals_tdm_r32_isa": bf16_gate["auto_r32_equals_forced_tdm_r32_isa"],
                    "bf16_auto_r64_equals_tdm_r64_isa": bf16_gate["auto_r64_equals_forced_tdm_r64_isa"],
                    "bf16_bm64_shape": "rows>=64*num_requests in auto/tdm raw-access mode; full64 one-valid-request guard; raw/cop preserve r16/r32 shape routing",
                    "fp8_shape": "unchanged r16/r32 selector; full32 one-valid-request guard; decode/r16 raw",
                    "fallback": "mixed/partial request blocks and unaligned bases select actual raw path; wrapper API unchanged"},
        "cpu_tests": tests, "unittest_total": sum(t["unittest_count"] or 0 for t in tests),
        "real_isa_specializations": len(records), "zero_waterfall_markers_all_specializations": True,
        "isa_counts_scope": "static textual counts including fallback paths, not executed/dynamic instruction counts",
        "kernels": records,
        "risks": ["No hardware validation or performance measurement; candidate only.",
                  "BM64 uses235VGPR/105SGPR vs r32TDM183/81; occupancy/register pressure may offset reduced K stream. New paths have zero spills and scratch.",
                  "fp8 padded b128 reads duplicate lane half reads: extra LDS traffic, tensor/barrier cost and conversion overhead require measurement.",
                  "Inherited forced bf16r16TDM has69632B LDS and29SGPR spills; auto still keepsr16 raw, unchanged.",
                  "CPU symbolic GEMM control-flow/opaque operand models do not prove numerical WMMA lowering or real TDM destination behavior; FP16 denormal preserve needs GPU code sweep.",
                  "Mixed requests/full-block alignment can suppress TDM and lose expected prefill gains; correctness fallback is tested."],
        "expected_latency_reduction_percent_hypothesis_only": {"prefill_fp8": "0-15", "prefill_bf16_bmge64": "0-15", "bf16_bm16_style": 0, "decode_fp8": 0},
        "gpu_plan": ["On an authorized healthy idle gfx1250 host acquire cooperative GPU lock; run first-touch/dmesg watchdog protocol, never overlap another tenant.",
                     "Compare candidate to champion3e3555b3 and explicitraw controls: full/partial/mixed rows, invalid pages, column tails, aligned and 8/16B offset subviews; verify every logit/visible group.",
                     "Run fp8 all256-code sweep including subnormals and signed zeros against scalar, then per-token scale/negative-scale correctness before performance.",
                     "A/A then randomized A/B unprofiled timings for fp8prefill512 and bf16prefill512/bmge64; held-out rerun; separately preserve decode and bm16 regressions.",
                     "Diagnose any regression with authorized kernel trace/register/LDS/counter passes; do not interpret static counts as performance. NoGPU was used in this round."]}
    args.output.write_bytes((json.dumps(evidence, indent=2) + "\n").encode())
    print(json.dumps({"status": evidence["status"], "kernel_sha256": kernel_sha,
                      "cpu_unittests": evidence["unittest_total"], "isa_specializations": len(records),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
