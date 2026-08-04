# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.attention.mla import (
    _mla_split_policy_kernel,
    _resolve_num_kv_splits,
)
from aiter.ops.triton.attention import mla as mla_api


@pytest.mark.parametrize("auto_num_kv_splits", [1, 4, 32, 256])
def test_default_num_kv_splits_uses_auto_policy(auto_num_kv_splits):
    assert _resolve_num_kv_splits(auto_num_kv_splits, None) == auto_num_kv_splits


@pytest.mark.parametrize("num_kv_splits", [1, 16, 32, 256])
def test_num_kv_splits_override(num_kv_splits):
    assert _resolve_num_kv_splits(1, num_kv_splits) == num_kv_splits


@pytest.mark.parametrize("num_kv_splits", [0, 257, -1])
def test_num_kv_splits_override_rejects_out_of_range(num_kv_splits):
    with pytest.raises(ValueError, match=r"must be in \[1, 256\]"):
        _resolve_num_kv_splits(1, num_kv_splits)


@pytest.mark.parametrize("num_kv_splits", [True, 32.0, "32"])
def test_num_kv_splits_override_rejects_non_integer(num_kv_splits):
    with pytest.raises(TypeError, match="must be an int or None"):
        _resolve_num_kv_splits(1, num_kv_splits)


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (1, 1),
        (4096, 16),
        (4097, 48),
        (65536, 64),
        (100000, 96),
        (262144, 112),
    ],
)
def test_device_split_policy_2d(context, expected):
    seq_lens = torch.tensor([context], dtype=torch.int32, device="cuda")
    active_splits = torch.empty(1, dtype=torch.int32, device="cuda")

    _mla_split_policy_kernel[(1,)](
        seq_lens,
        active_splits,
        NUM_KV_SPLITS=112,
        BLOCK_N=64,
        USE_2D_VIEW=True,
    )

    assert active_splits.item() == expected


def test_device_split_policy_varlen():
    kv_indptr = torch.tensor([0, 100000], dtype=torch.int32, device="cuda")
    active_splits = torch.empty(1, dtype=torch.int32, device="cuda")

    _mla_split_policy_kernel[(1,)](
        kv_indptr,
        active_splits,
        NUM_KV_SPLITS=112,
        BLOCK_N=64,
        USE_2D_VIEW=False,
    )

    assert active_splits.item() == 96


def _gfx950_route_kwargs(num_query_heads):
    return {
        "num_query_heads": num_query_heads,
        "num_kv_heads": 1,
        "block_size": 1,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
        "out_dtype": torch.bfloat16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "shuffled_kv_cache": False,
        "q_descale": None,
        "kv_descale": None,
        "q_scales": None,
        "out_scale": None,
        "skip_reduce": False,
        "q_is_contiguous": True,
        "kv_buffer_is_contiguous": True,
        "out_is_contiguous": True,
    }


@pytest.mark.parametrize(
    ("num_query_heads", "expected"),
    [(12, True), (15, True), (16, False), (64, False), (128, False)],
)
def test_gfx950_gluon_head_guard(monkeypatch, num_query_heads, expected):
    monkeypatch.setattr(mla_api, "IS_DEVICE_ARCH_GFX950", True)
    monkeypatch.setattr(mla_api, "gfx950_mla_decode_fwd", object())

    assert (
        mla_api._use_gfx950_gluon_decode(**_gfx950_route_kwargs(num_query_heads))
        is expected
    )


def test_gfx950_gluon_unvalidated_layout_falls_back(monkeypatch):
    monkeypatch.setattr(mla_api, "IS_DEVICE_ARCH_GFX950", True)
    monkeypatch.setattr(mla_api, "gfx950_mla_decode_fwd", object())
    kwargs = _gfx950_route_kwargs(12)
    kwargs["block_size"] = 64

    assert not mla_api._use_gfx950_gluon_decode(**kwargs)
