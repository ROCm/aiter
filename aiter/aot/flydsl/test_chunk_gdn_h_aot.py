# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the chunk-gated-delta-h AOT pre-compilation script.

These tests exercise the *parser + dedupe* layer only and do not require a
GPU or FlyDSL/MLIR compilation. Smoke + cache integration tests that touch
the FlyDSL JIT live in ``tests/...``.

Run::

    PYTHONPATH=./ pytest aiter/aot/flydsl/test_chunk_gdn_h_aot.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aiter.aot.flydsl.chunk_gdn_h import (
    DEFAULT_JSONLS,
    _DEFAULT_JSONL,
    parse_jsonl,
)
from aiter.aot.flydsl.common import job_identity


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    fp = tmp_path / "tuned.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return fp


_BASE_ROW = {
    "arch": "gfx950",
    "dtype": "torch.bfloat16",
    "K": 128,
    "V": 128,
    "BT": 64,
    "H": 16,
    "Hg": 4,
    "T_flat": 2500,
    "N": 1,
    "use_g": True,
    "use_gk": False,
    "use_h0": True,
    "store_fs": False,
    "save_vn": True,
    "is_varlen": False,
    "wu_contig": True,
    "config": {"BV": 16},
    "duration": 12.3,
}


def test_default_jsonl_exists():
    assert os.path.isfile(_DEFAULT_JSONL), (
        f"Default tuned jsonl missing: {_DEFAULT_JSONL}. "
        "AOT script will not find any work to do."
    )
    assert DEFAULT_JSONLS == [str(_DEFAULT_JSONL)]


def test_default_jsonl_parses_nonempty():
    jobs = parse_jsonl(str(_DEFAULT_JSONL))
    assert len(jobs) > 0, "Default jsonl produced zero AOT jobs"
    for job in jobs:
        for required in (
            "dtype",
            "arch",
            "K",
            "V",
            "BT",
            "BV",
            "H",
            "Hg",
            "use_g",
            "use_gk",
            "use_h0",
            "store_fs",
            "save_vn",
            "is_varlen",
            "wu_contig",
            "state_bf16",
        ):
            assert required in job, f"Missing field {required!r} in {job}"
        assert job["V"] % job["BV"] == 0
        assert job["BV"] <= job["V"]


def test_parse_dedupe_same_shape_different_T_flat(tmp_path):
    rows = []
    for tf in (2500, 60000, 128000):
        r = dict(_BASE_ROW)
        r["T_flat"] = tf
        rows.append(r)
    fp = _write_jsonl(tmp_path, rows)
    jobs = parse_jsonl(str(fp))
    assert len(jobs) == 1, (
        "T_flat does not affect the compiled artifact and should be "
        f"dedup'd; got {len(jobs)} jobs"
    )


def test_parse_dedupe_distinct_compile_keys(tmp_path):
    rows = [
        dict(_BASE_ROW, H=16, **{"config": {"BV": 16}}),
        dict(_BASE_ROW, H=32, **{"config": {"BV": 16}}),
        dict(_BASE_ROW, H=16, **{"config": {"BV": 32}}),
        dict(_BASE_ROW, H=16, use_gk=True, **{"config": {"BV": 16}}),
    ]
    fp = _write_jsonl(tmp_path, rows)
    jobs = parse_jsonl(str(fp))
    assert len(jobs) == 4
    keys = {job_identity(j) for j in jobs}
    assert len(keys) == 4


def test_parse_skip_unknown_dtype(tmp_path):
    fp = _write_jsonl(
        tmp_path,
        [
            dict(_BASE_ROW, dtype="torch.float64"),
            dict(_BASE_ROW),
        ],
    )
    jobs = parse_jsonl(str(fp))
    assert len(jobs) == 1
    assert jobs[0]["dtype"] == "torch.bfloat16"


def test_parse_skip_bad_bv(tmp_path):
    fp = _write_jsonl(
        tmp_path,
        [
            dict(_BASE_ROW, V=128, **{"config": {"BV": 24}}),
            dict(_BASE_ROW),
        ],
    )
    jobs = parse_jsonl(str(fp))
    assert len(jobs) == 1


def test_parse_skip_malformed_lines(tmp_path):
    fp = tmp_path / "tuned.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write("# comment\n")
        f.write("not-json\n")
        f.write(json.dumps(_BASE_ROW) + "\n")
    jobs = parse_jsonl(str(fp))
    assert len(jobs) == 1


def test_state_bf16_default_false(tmp_path):
    fp = _write_jsonl(tmp_path, [dict(_BASE_ROW)])
    jobs = parse_jsonl(str(fp))
    assert jobs[0]["state_bf16"] is False


@pytest.mark.parametrize("bad_v_bv", [(128, 96), (128, 256)])
def test_parse_rejects_bv_not_dividing_v(tmp_path, bad_v_bv):
    v, bv = bad_v_bv
    fp = _write_jsonl(
        tmp_path,
        [dict(_BASE_ROW, V=v, **{"config": {"BV": bv}})],
    )
    jobs = parse_jsonl(str(fp))
    assert len(jobs) == 0


def test_target_arch_override_changes_arch_field(tmp_path):
    """``--target-arch X`` should rewrite every job's ``arch`` to X."""
    from aiter.aot.flydsl.common import dedupe_jobs

    fp = _write_jsonl(
        tmp_path,
        [
            dict(_BASE_ROW, arch="gfx950"),
            dict(_BASE_ROW, arch="gfx950", H=32),
        ],
    )
    parsed = parse_jsonl(str(fp))
    assert {j["arch"] for j in parsed} == {"gfx950"}

    overridden = dedupe_jobs([dict(j, arch="gfx942") for j in parsed])
    assert {j["arch"] for j in overridden} == {"gfx942"}
    assert len(overridden) == len(parsed)


def test_target_arch_override_merges_cross_arch_dups(tmp_path):
    """Two jsonl entries differing only in ``arch`` collapse to one job
    after a ``--target-arch`` override (the rest of the compile key is
    identical)."""
    from aiter.aot.flydsl.common import dedupe_jobs

    fp = _write_jsonl(
        tmp_path,
        [
            dict(_BASE_ROW, arch="gfx950"),
            dict(_BASE_ROW, arch="gfx942"),
        ],
    )
    parsed = parse_jsonl(str(fp))
    assert len(parsed) == 2, "parser must keep distinct-arch entries separate"

    overridden = dedupe_jobs([dict(j, arch="gfx950") for j in parsed])
    assert len(overridden) == 1, (
        "after --target-arch rewrites arch, the two entries become "
        "identical and dedupe should collapse them"
    )
    assert overridden[0]["arch"] == "gfx950"
