# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the nearest-tuned-M fallback in the a8w8 GEMM config lookup.

These exercise the pure lookup logic (no kernel launch, no GPU), so they run
anywhere aiter imports. The motivating case: a shape whose (N, K) is tuned but
whose M grid has a hole exactly where the padded_M candidates land, which used
to drop the shape to the untuned default config.
"""

import pandas as pd
import pytest

from aiter.ops import gemm_op_a8w8 as gemm_mod


def _ref_padded_m(M, N, K, gl):
    """Pure-python mirror of csrc/py_itfs_cu/gemm_common.cu::getPaddedM."""
    if gl == 0:
        if M <= 256:
            return (M + 15) // 16 * 16
        if M <= 1024:
            return (M + 31) // 32 * 32
        if M <= 4096:
            return (M + 63) // 64 * 64
        return (M + 127) // 128 * 128
    if gl == 1:
        if M > 8192 and N > 4096:
            return 8192
        n = 1
        while n < M:
            n <<= 1
        return n
    return M


# (N=5376, K=16384) is the Gemma-4-31B global-attention o_proj; tuned at
# power-of-2 M but with M=1024 MISSING -- the real gap observed at runtime.
_M_BUCKETS = [1, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 8192, 16384]


@pytest.fixture
def tuned_csv(tmp_path):
    rows = [
        {
            "gfx": "gfx942",
            "cu_num": 304,
            "M": m,
            "N": 5376,
            "K": 16384,
            "libtype": "ck",
            "kernelId": 0,
            "splitK": 0,
            "us": 1.0,
            "kernelName": f"k_{m}",
            "tflops": 0.0,
            "bw": 0.0,
            "errRatio": 0.0,
        }
        for m in _M_BUCKETS
    ]
    path = tmp_path / "tuned.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_nearest_tuned_m_floor():
    assert gemm_mod._nearest_tuned_m([1, 8, 512, 2048], 739) == 512
    assert gemm_mod._nearest_tuned_m([512, 2048], 512) == 512
    assert gemm_mod._nearest_tuned_m([512, 2048], 5000) == 2048
    # below the smallest bucket -> the smallest available bucket
    assert gemm_mod._nearest_tuned_m([512, 2048], 10) == 512
    assert gemm_mod._nearest_tuned_m([], 739) is None
    assert gemm_mod._nearest_tuned_m(None, 739) is None


def test_build_m_index_groups_by_shape():
    cache = {
        ("gfx942", 304, 512, 5376, 16384): {},
        ("gfx942", 304, 2048, 5376, 16384): {},
        ("gfx942", 304, 512, 5376, 8192): {},
    }
    idx = gemm_mod._build_m_index(cache, has_gfx=True)
    assert idx[("gfx942", 304, 5376, 16384)] == [512, 2048]
    assert idx[("gfx942", 304, 5376, 8192)] == [512]


def test_gap_in_m_grid_falls_back_to_nearest(monkeypatch, tuned_csv):
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: 304)
    monkeypatch.setattr(gemm_mod, "get_padded_m", _ref_padded_m)
    get = gemm_mod.get_CKGEMM_config
    get.cache_clear()

    # M in (512, 1024]: nextPow2(M) == 1024, the missing bucket; every padded_M
    # candidate misses, so the lookup must reuse the nearest tuned bucket (512)
    # instead of returning None (which would select the untuned default).
    for M in (513, 739, 1008):
        config = get(M, 5376, 16384, tuned_csv)
        assert config is not None, f"M={M} dropped to default despite a tuned (N,K)"
        assert config["kernelName"] == "k_512"


def test_exact_bucket_still_wins(monkeypatch, tuned_csv):
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: 304)
    monkeypatch.setattr(gemm_mod, "get_padded_m", _ref_padded_m)
    get = gemm_mod.get_CKGEMM_config
    get.cache_clear()

    assert get(512, 5376, 16384, tuned_csv)["kernelName"] == "k_512"
    assert get(2048, 5376, 16384, tuned_csv)["kernelName"] == "k_2048"


def test_untuned_shape_still_returns_none(monkeypatch, tuned_csv):
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: 304)
    monkeypatch.setattr(gemm_mod, "get_padded_m", _ref_padded_m)
    get = gemm_mod.get_CKGEMM_config
    get.cache_clear()

    # A genuinely untuned (N, K) has no bucket to borrow -> stays on the default.
    assert get(739, 1234, 5678, tuned_csv) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
