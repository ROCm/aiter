# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the PA-DECODE config loader, the v1/v2 dispatch rule and
the packaged gfx1250 config."""

import json
import os

import pytest
import torch

from aiter.ops.triton._triton_kernels.attention import pa_decode as pa_kernels
from aiter.ops.triton.attention import pa_decode as pa
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
)

_GFX1250_JSON = os.path.join(
    AITER_TRITON_CONFIGS_PATH,
    "gfx1250",
    "triton",
    "attention",
    "pa_decode",
    "DEFAULT.json",
)
_LAUNCH_KEYS = {"num_warps", "num_stages", "waves_per_eu"}


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Point the loader at a temp dir and return a writer for its DEFAULT.json."""
    monkeypatch.setattr(
        pa_kernels, "resolve_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    load_config_json.cache_clear()

    def write(config):
        (tmp_path / "DEFAULT.json").write_text(json.dumps(config))
        load_config_json.cache_clear()

    yield write
    load_config_json.cache_clear()


def test_launch_config_picks_bucket_by_page_size(config_dir):
    config_dir(
        {
            "launch": {
                "v1_w_dot": {
                    "BLK_LEQ_16": {"num_stages": 2},
                    "any": {"num_stages": 1},
                }
            }
        }
    )
    assert pa_kernels._get_launch_config("v1_w_dot", 16) == {"num_stages": 2}
    assert pa_kernels._get_launch_config("v1_w_dot", 8) == {"num_stages": 2}
    assert pa_kernels._get_launch_config("v1_w_dot", 128) == {"num_stages": 1}


def test_launch_config_returns_copy(config_dir):
    config_dir({"launch": {"v1_w_dot": {"any": {"num_stages": 1}}}})
    pa_kernels._get_launch_config("v1_w_dot", 16)["num_stages"] = 99
    assert pa_kernels._get_launch_config("v1_w_dot", 16) == {"num_stages": 1}


def test_launch_config_missing_returns_none(config_dir, tmp_path):
    # No file for this arch.
    assert pa_kernels._get_launch_config("v1_w_dot", 16) is None
    # File without a launch table, or without this kernel.
    config_dir({"dispatch": {}})
    assert pa_kernels._get_launch_config("v1_w_dot", 16) is None
    config_dir({"launch": {"v2_w_dot": {"any": {}}}})
    assert pa_kernels._get_launch_config("v1_w_dot", 16) is None


def test_dispatch_config_lookup(config_dir):
    config_dir({"dispatch": {"bf16": {"v1_max_partitions": 1}}})
    assert pa_kernels._get_dispatch_config("bf16") == {"v1_max_partitions": 1}
    assert pa_kernels._get_dispatch_config("fp16") is None


_DISPATCH = {
    "v1_max_partitions": 1,
    "v1_min_programs": 2048,
    "v1_busy_max_partitions": 8,
}


def _picked_kernel(monkeypatch, dispatch, num_seqs, seq_len, num_kv_heads=8):
    """Name of the paged_attn_decode_* entry point the dispatcher calls."""
    called = []
    monkeypatch.setattr(pa, "_get_dispatch_config", lambda kv_dtype: dispatch)
    monkeypatch.setattr(pa, "paged_attn_decode_v1", lambda *a, **k: called.append("v1"))
    monkeypatch.setattr(pa, "paged_attn_decode_v2", lambda *a, **k: called.append("v2"))
    pa.paged_attention_decode(
        output=None,
        query=torch.empty(num_seqs, 4 * num_kv_heads, 1, dtype=torch.float16),
        key_cache=torch.empty(1, num_kv_heads, 1, 1, dtype=torch.float16),
        value_cache=torch.empty(1, num_kv_heads, 1, 1, dtype=torch.float16),
        seq_lens=None,
        block_tables=None,
        attn_scale=1.0,
        max_seq_len=seq_len,
        compute_type=None,
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
    )
    assert len(called) == 1
    return called[0]


@pytest.mark.parametrize(
    "num_seqs, seq_len, expected",
    [
        (8, 1024, "v1"),  # one partition
        (8, 2048, "v2"),  # two partitions, GPU not busy
        (255, 8192, "v2"),  # 2040 programs, just under the busy threshold
        (256, 2048, "v1"),  # 2048 programs, partitions within the busy limit
        (256, 8192, "v1"),  # 8 partitions, at the busy limit
        (256, 9216, "v2"),  # 9 partitions, over the busy limit
    ],
)
def test_dispatch_rule(monkeypatch, num_seqs, seq_len, expected):
    assert _picked_kernel(monkeypatch, _DISPATCH, num_seqs, seq_len) == expected


def test_dispatch_without_busy_rule(monkeypatch):
    dispatch = {"v1_max_partitions": 1}
    assert _picked_kernel(monkeypatch, dispatch, 4096, 8192) == "v2"


def test_dispatch_without_config_keeps_legacy_rule(monkeypatch):
    # Legacy: v1 only for max_seq_len <= 8192 and one partition or many heads.
    assert _picked_kernel(monkeypatch, None, 8, 1024) == "v1"
    assert _picked_kernel(monkeypatch, None, 8, 8192) == "v2"
    assert _picked_kernel(monkeypatch, None, 64, 8192) == "v1"
    assert _picked_kernel(monkeypatch, None, 64, 16384) == "v2"


def test_gfx1250_config_schema():
    with open(_GFX1250_JSON) as f:
        config = json.load(f)

    for dtype in ("bf16", "fp16"):
        dispatch = config["dispatch"][dtype]
        assert set(dispatch) == {
            "v1_max_partitions",
            "v1_min_programs",
            "v1_busy_max_partitions",
        }
        assert all(isinstance(v, int) and v >= 1 for v in dispatch.values())
        # The busy rule only widens the v1 range.
        assert dispatch["v1_busy_max_partitions"] >= dispatch["v1_max_partitions"]

    for kernel in ("v1_w_dot", "v2_w_dot"):
        table = config["launch"][kernel]
        assert "any" in table
        for key, entry in table.items():
            assert key == "any" or key.startswith("BLK_LEQ_")
            assert set(entry) == _LAUNCH_KEYS
            assert entry["num_warps"] in (1, 2, 4, 8)
            # num_stages >= 3 hangs the GPU on gfx1250.
            assert entry["num_stages"] in (1, 2)
            assert entry["waves_per_eu"] >= 0
