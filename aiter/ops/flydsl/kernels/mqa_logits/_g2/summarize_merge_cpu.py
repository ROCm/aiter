# SPDX-License-Identifier: MIT
"""Summarize freshly compiled merge CPU gates and static final-ISA evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

MODES = ("auto", "raw", "tdm", "tdm64", "r64", "r64raw", "r64tdm")
TESTS = ("contract", "address", "q-lds", "trace16", "trace32", "k-tdm", "routing", "fp8-k-tdm", "bm64", "strict-tdm", "strict-legacy", "routing-isa", "fp8-isa")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
    assert sha(source) in (a.results / "source.sha256").read_text()
    tests, kernels = [], []
    for name in TESTS:
        path = a.results / (name + ".log")
        text = path.read_text()
        assert "FAIL" not in text and "Traceback" not in text, name
        if name not in ("strict-tdm", "strict-legacy", "routing-isa", "fp8-isa"):
            assert "PASS" in text or re.search(r"\bOK\b", text), name
        n = re.search(r"Ran (\d+) tests", text)
        tests.append(dict(name=name, status="PASS", unittest_count=int(n[1]) if n else None,
                          pass_cases=text.count("[PASS]"), log_sha256=sha(path)))
    markers = ("tensor_load_to_lds", "s_wait_tensorcnt", "ds_load_b128", "s_barrier_signal", "s_barrier_wait", "v_readfirstlane_b32", "v_max_i64", "ds_load_b128_addtid")
    for mode in MODES:
        compile_data = json.loads((a.results / mode / "compile_only.json").read_text())
        expected = 7 if mode.startswith("r64") else 6
        assert len(compile_data["specializations"]) == expected
        assert all(x["compiled"] for x in compile_data["specializations"])
        pairs = re.findall(r"^\[COMPILED\]\s+(\S+)\s+kernel=(\S+)", (a.results / mode / "compile.log").read_text(), re.M)
        assert len(pairs) == expected
        for dispatch, name in pairs:
            path = a.results / mode / "isa" / name / "21_final_isa.s"
            text = path.read_text()
            counts = {m: len(re.findall(r"\b" + m + r"\b", text)) for m in markers}
            assert all(counts[m] == 0 for m in markers[-3:]), (mode, name, counts)
            lds = re.findall(r"\.amdhsa_group_segment_fixed_size\s+(\d+)", text)
            assert len(lds) == 1
            resources = {}
            for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size"):
                match = re.search(r"\." + key + r":\s*(\d+)", text)
                assert match, (name, key)
                resources[key] = int(match[1])
            # Descriptor next-free counts include SGPR-to-VGPR spill carriers;
            # YAML logical counts do not describe the physical allocation.
            for kind in ("vgpr", "sgpr"):
                values = re.findall(r"\.amdhsa_next_free_" + kind + r"\s+(\d+)", text)
                assert len(values) == 1, (name, kind, values)
                resources["metadata_" + kind + "_count"] = resources[kind + "_count"]
                resources[kind + "_count"] = int(values[0])
            if "wmma" in name:
                assert resources["vgpr_spill_count"] == resources["private_segment_fixed_size"] == 0
            tdm = name.endswith("_tdm")
            kernels.append(dict(mode=mode, dispatch=dispatch, kernel_name=name,
                                sha256=sha(path), lds_bytes=int(lds[0]),
                                pitch_bytes=(144 if "fp8" in name else 272) if tdm else None,
                                tensor_wait_zero=len(re.findall(r"\bs_wait_tensorcnt\s+(?:0x0|0)\b", text)),
                                markers=counts, resources=resources))
    bf = json.loads((a.results / "routing-isa.json").read_text())
    fp = json.loads((a.results / "fp8-tdm-isa.json").read_text())
    assert all(x["status"] == "PASS" for x in bf["kernels"] + fp["kernels"])
    old = json.loads((Path(__file__).resolve().parent / "tdm4_isa_evidence.json").read_text())
    old_sha = {k["kernel_name"]: k["sha256"] for k in old["kernels"]}
    inherited = {}
    for k in kernels:
        if k["kernel_name"] in old_sha:
            assert k["sha256"] == old_sha[k["kernel_name"]], k["kernel_name"]
            inherited[k["kernel_name"]] = k["sha256"]
    r16_expected = "d04e3b64d5d7bcd06f8ef1dceb39d9aea91cd8d96721ac8dcf1b16e5f01ed666"
    r16 = next(k for k in kernels if k["mode"] == "tdm" and k["dispatch"] == "bf16_wmma")
    assert r16["sha256"] == r16_expected
    result = dict(inherited_tdm4_isa_byte_identical=inherited,
                  r16_c32_isa_byte_identical_to_measured_r16=True,
                  original_r16_c32_isa_sha256=r16_expected,
                  status="PASS_NO_DEVICE", kernel_sha256=sha(source),
                  image="rocm/fw-bringup-mlperf:gfx1250-atom-20261015", flydsl="0.3.3.dev886",
                  arch="gfx1250", no_device=True, real_isa_specializations=len(kernels),
                  scope="CPU/layout/control-flow/real compile ISA only; no merged GPU correctness or latency claim",
                  unittest_total=sum(t["unittest_count"] or 0 for t in tests), cpu_tests=tests,
                  zero_waterfall_markers_all_specializations=True,
                  static_counts_not_dynamic=True, kernels=kernels,
                  routing_equalities={k:v for k,v in bf.items() if k not in ("kernels", "scope")},
                  fp8_routing_equalities={k:v for k,v in fp.items() if k not in ("kernels", "scope")})
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=result["status"], cpu_unittests=result["unittest_total"],
                          real_isa_specializations=len(kernels), kernel_sha256=sha(source))))


if __name__ == "__main__":
    main()
