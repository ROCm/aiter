# SPDX-License-Identifier: MIT
"""Contract/pruning tests; no GPU required."""

import csv
import json
import math

import pytest

from aiter.ops.flydsl.gemm_a16w16_tdm_policy import (
    KEYS,
    SCHEMA,
    TdmConfig,
    TdmTunedPolicy,
    candidates,
)


def row(config=None, **changes):
    r = dict(
        zip(
            KEYS,
            (
                "gfx1250",
                256,
                256,
                256,
                8192,
                SCHEMA,
                "torch.bfloat16",
                "nt",
                8192,
                8192,
                256,
                0,
                0,
                256 << 20,
                ".004",
            ),
        )
    )
    r.update(kernelName=(config or TdmConfig()).encode(), us=10.0, errRatio=0.001)
    r.update(changes)
    return r


def write(path, rows):
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def test_roundtrip_policy_key():
    a = TdmConfig(split_k=4, split_k_policy="atomic")
    b = TdmConfig(split_k=4, split_k_policy="workspace")
    assert a.encode() != b.encode()
    assert TdmConfig.decode(a.encode()) == a
    assert TdmConfig.decode(b.encode()) == b
    assert a.workspace_bytes(256, 256) == 0
    assert b.workspace_bytes(256, 256) == 4 * 256 * 256 * 4


def test_safe_and_optin_candidates():
    safe = candidates(256, 256, 8192)
    both = candidates(256, 256, 8192, allow_atomic=True)
    assert safe and all(c.split_k_policy == "workspace" for c in safe)
    assert set(safe) <= set(both)
    assert any(c.split_k_policy == "atomic" and c.split_k > 1 for c in both)
    assert any(c.split_k == 1 for c in candidates(256, 256, 8192, splits=(4,)))
    for c in both:
        c.validate(256, 256, 8192, "nt", True, 256 << 20)


def test_shape_grid_budget_and_odd_tail():
    assert all(c.split_k == 1 for c in candidates(8192, 8192, 8192))
    assert any(
        c.split_k > 1
        for c in candidates(
            8192, 8192, 8192, exhaustive=True, max_workspace_bytes=4 << 30
        )
    )
    assert all(
        c.workspace_bytes(128, 128) == 0
        for c in candidates(128, 128, 4096, max_workspace_bytes=0)
    )
    assert not any(
        c.split_k_policy == "atomic" and c.split_k > 1
        for c in candidates(137, 139, 4096, allow_atomic=True)
    )
    assert all(not c.interleave for c in candidates(128, 128, 4096, layout="nn"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_k", True),
        ("split_k", 2.5),
        ("stages", "2"),
        ("split_k_policy", "unknown"),
        ("interleave", 1),
    ],
)
def test_decode_rejects_types(field, value):
    d = TdmConfig().kwargs()
    d[field] = value
    with pytest.raises(ValueError):
        TdmConfig.decode(json.dumps(d))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "old"},
        {"gfx": "gfx950"},
        {"dtype": "torch.float32"},
        {"us": -1},
        {"us": math.nan},
        {"errRatio": math.inf},
        {"allow_atomic": 2},
        {"bias": 2},
        {"lda": 8191},
        {"ldc": 255},
        {"cu_num": 0},
        {"relative_l2_limit": ".1"},
        {"relative_l2_limit": "nan"},
        {"errRatio": 0.01},
        {"kernelName": TdmConfig(split_k=4, split_k_policy="atomic").encode()},
        {"kernelName": TdmConfig(split_k=4).encode(), "max_workspace_bytes": 0},
    ],
)
def test_loader_rejects_invalid_rows(tmp_path, changes):
    p = tmp_path / "bad.csv"
    write(p, [row(**changes)])
    with pytest.raises(ValueError):
        TdmTunedPolicy(p)


def test_loader_duplicate_and_distinct_precision_contracts(tmp_path):
    p = tmp_path / "rows.csv"
    write(p, [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        TdmTunedPolicy(p)
    write(p, [row(), row(allow_atomic=1)])
    assert len(TdmTunedPolicy(p).rows) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
