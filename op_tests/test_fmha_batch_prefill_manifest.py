# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "hsa" / "gfx950" / "fmha_v3_fwd" / "fmha_batch_prefill.csv"
DENSE_MANIFEST = ROOT / "hsa" / "gfx950" / "fmha_v3_fwd" / "fmha_fwd.csv"


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith(("//", "#", ";"))
        )
        return list(reader.fieldnames or ()), list(reader)


def test_batch_prefill_manifest_declares_static_page_size():
    dense_fields, _ = _read_manifest(DENSE_MANIFEST)
    fields, rows = _read_manifest(MANIFEST)

    assert "page_size" not in dense_fields
    assert "page_size" in fields
    assert rows
    assert {int(row["mask"]) for row in rows} == {0, 2}

    dispatch_keys = set()
    dispatch_ranges: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for row in rows:
        page_size = int(row["page_size"])
        min_seqlen_q = int(row["min_seqlen_q"])
        max_seqlen_q = int(row["max_seqlen_q"])
        assert page_size > 0 and page_size & (page_size - 1) == 0
        assert min_seqlen_q >= 0 and max_seqlen_q >= 0
        assert not (min_seqlen_q and max_seqlen_q) or min_seqlen_q <= max_seqlen_q
        assert row["kv_layout"] in {"linear", "vectorized"}
        assert row["lookup_table"] in {"sglang", "vllm"}
        assert row["qscale"] in {"no", "pertensor"}
        assert row["abi"] in {"paged_varlen_v3_ext", "paged_varlen_v3_reuse"}
        assert row["grid_layout"] in {"qtiles_heads_batch", "heads_batch_qtiles"}
        if row["dtype"] == "fp8bf16":
            assert row["qscale"] == "pertensor"
        else:
            assert row["dtype"] in {"bf16", "fp16"}
            assert row["qscale"] == "no"
        assert int(row["ts_qo"]) > 0
        assert int(row["bdx"]) > 0
        assert (MANIFEST.parent / row["co_name"]).is_file()

        key = tuple(
            row[name]
            for name in (
                "dtype",
                "hdim_q",
                "hdim_v",
                "mask",
                "page_size",
                "kv_layout",
                "lookup_table",
                "qscale",
                "min_seqlen_q",
                "max_seqlen_q",
            )
        )
        assert key not in dispatch_keys
        dispatch_keys.add(key)

        static_key = tuple(
            row[name]
            for name in (
                "dtype",
                "hdim_q",
                "hdim_v",
                "mask",
                "page_size",
                "kv_layout",
                "lookup_table",
                "qscale",
            )
        )
        dispatch_ranges.setdefault(static_key, []).append(
            (min_seqlen_q or 1, max_seqlen_q or 2**31 - 1)
        )

    for ranges in dispatch_ranges.values():
        ranges.sort()
        for previous, current in pairwise(ranges):
            assert previous[1] < current[0]

    wrapper = (ROOT / "csrc" / "cpp_itfs" / "mha_fwd_batch_prefill.cu").read_text()
    assert "a.page_block_size != 64" not in wrapper
    for row in rows:
        assert row["knl_name"] not in wrapper
        assert row["co_name"] not in wrapper


def test_batch_prefill_build_generates_asm_manifest_header():
    config = json.loads((ROOT / "aiter" / "jit" / "optCompilerConfig.json").read_text())
    for target in ("module_mha_batch_prefill", "libmha_fwd"):
        commands = config[target]["blob_gen_cmd"]
        assert any("hsa/codegen.py -m fmha_v3_fwd" in command for command in commands)


def test_optional_page_size_column_is_codegen_compatible(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    env = os.environ.copy()
    env["AITER_GPU_ARCHS"] = "gfx942;gfx950"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "hsa" / "codegen.py"),
            "-m",
            "fmha_v3_fwd",
            "--output_dir",
            str(tmp_path),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )

    generated = (tmp_path / "asm_fmha_v3_fwd_configs.hpp").read_text()
    assert "int page_size;" in generated
    assert "int min_seqlen_q;" in generated
    assert "int max_seqlen_q;" in generated
    assert "std::string kv_layout;" in generated
    assert "static CFG cfg_fmha_fwd" in generated
    assert "static CFG cfg_fmha_batch_prefill" in generated
