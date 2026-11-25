# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-side dispatch checks for the packaged gfx950 attention configs."""

from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.utils import unified_attention_utils as config_utils


class _BlockTableMetadata:
    def __init__(self, width, row_stride=None, column_stride=1):
        self.shape = (1, width)
        self._strides = (width if row_stride is None else row_stride, column_stride)

    def stride(self, dimension):
        return self._strides[dimension]


def _params(q, k, **overrides):
    params = SimpleNamespace(
        num_seqs=1,
        num_queries_per_kv=1,
        num_tokens=q,
        max_seqlen_q=q,
        max_seqlen_k=k,
        causal=True,
        all_decode=False,
        shuffled_kv_cache=False,
        sliding_window=0,
        softcap=0,
        head_size=128,
        head_size_v=128,
        q_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        out=SimpleNamespace(dtype=torch.bfloat16),
        block_table=_BlockTableMetadata((k + 511) // 512),
        block_size=512,
        num_sms=256,
        num_query_heads=16,
        num_kv_heads=16,
        num_2d_prgms=(q // 16 + 1) * 16,
        sinks=None,
        q_scales=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_scale=None,
        use_alibi_slopes=False,
        use_qq_bias=False,
    )
    for name, value in overrides.items():
        setattr(params, name, value)
    return params


def _config(params):
    return config_utils.get_unified_attention_config(
        "attn_2d", params, backend="gluon", arch="gfx950"
    )


@pytest.fixture(scope="module")
def _packaged_table():
    table, axes, _ = config_utils._load("attn_2d", "gluon", "gfx950")
    return table, axes


@pytest.mark.parametrize(
    "q,k,preload_pages",
    [
        (462, 974, 2),
        (2510, 2510, 8),
        (257, 769, 2),
        (512, 1024, 2),
        (2049, 2049, 8),
        (4095, 4095, 8),
    ],
)
def test_recovery_config_selected(q, k, preload_pages, _packaged_table):
    params = _params(q, k)
    axes_values = config_utils._axis_values(
        params.head_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.block_size,
        params.q_dtype,
        params.kv_cache_dtype,
        config_utils._programs_per_cu(params),
        params.num_queries_per_kv,
        extra_axes=config_utils._prefill_axes(params),
    )
    key, leaf = config_utils._lookup(*_packaged_table, axes_values)
    config = _config(params)
    assert key.startswith("SIMPLE.")
    assert leaf["PRELOAD_PAGES"] == config["PRELOAD_PAGES"] == preload_pages
    assert config["USE_EXPLICIT_FMA"] is True


@pytest.mark.parametrize(
    "change",
    [
        "padded_table",
        "row_stride",
        "column_stride",
        "fp16",
        "v_fp16",
        "batch",
        "heads",
        "cus",
        "gqa",
        "noncausal",
        "window",
        "sink",
        "scale",
        "loose_query_bound",
    ],
)
def test_recovery_config_excluded_metadata(change):
    params = _params(2510, 2510)
    if change == "padded_table":
        params.block_table = _BlockTableMetadata(16)
    elif change == "row_stride":
        params.block_table = _BlockTableMetadata(5, row_stride=8)
    elif change == "column_stride":
        params.block_table = _BlockTableMetadata(5, column_stride=2)
    elif change == "fp16":
        params.q_dtype = params.kv_cache_dtype = torch.float16
        params.out = SimpleNamespace(dtype=torch.float16)
    elif change == "v_fp16":
        params.v_cache_dtype = torch.float16
    elif change == "batch":
        params.num_seqs = 2
    elif change == "heads":
        params.num_query_heads = params.num_kv_heads = 32
    elif change == "cus":
        params.num_sms = 128
    elif change == "gqa":
        params.num_queries_per_kv = 2
        params.num_kv_heads = 8
    elif change == "noncausal":
        params.causal = False
    elif change == "window":
        params.sliding_window = 256
    elif change == "sink":
        params.sinks = object()
    elif change == "scale":
        params.q_descale = object()
    else:
        params.num_tokens = 2509
    config = _config(params)
    assert config["PRELOAD_PAGES"] == 0
    assert config["USE_EXPLICIT_FMA"] is False


def test_recovery_config_cache_separates_optional_inputs():
    selected = _params(2510, 2510)
    excluded = _params(2510, 2510, sinks=object())
    assert config_utils._prefill_axes(selected) != config_utils._prefill_axes(excluded)
    assert _config(selected)["PRELOAD_PAGES"] > 0
    assert _config(excluded)["PRELOAD_PAGES"] == 0
    assert _config(selected)["PRELOAD_PAGES"] > 0


def test_recovery_config_returns_independent_values():
    params = _params(2510, 2510)
    first = _config(params)
    first.clear()
    assert _config(params)["PRELOAD_PAGES"] > 0


def test_metadata_only_params_keep_baseline_configuration():
    params = SimpleNamespace(
        head_size=128,
        head_size_v=128,
        max_seqlen_q=2510,
        max_seqlen_k=2510,
        sliding_window=0,
        shuffled_kv_cache=False,
        q_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        block_size=512,
        num_2d_prgms=2512,
        num_sms=256,
        num_queries_per_kv=1,
    )
    assert not dict(config_utils._prefill_axes(params))["SIMPLE"]
    assert _config(params)["PRELOAD_PAGES"] == 0
    explanation = config_utils.explain(
        "attn_2d", params, backend="gluon", arch="gfx950"
    )
    assert "matched: any" in explanation


@pytest.mark.parametrize(
    "op,backend",
    [
        ("attn_3d", "gluon"),
        ("kv_split", "gluon"),
        ("reduce", "triton"),
        ("attn_2d", "triton"),
    ],
)
def test_non_prefill_configs_skip_prefill_metadata(monkeypatch, op, backend):
    def unexpected_metadata(_params):
        pytest.fail("Non-prefill configuration inspected prefill metadata")

    monkeypatch.setattr(config_utils, "_prefill_axes", unexpected_metadata)
    params = _params(
        1, 5699, num_seqs=8, num_tokens=8, num_2d_prgms=128, all_decode=True
    )
    config = config_utils.get_unified_attention_config(
        op, params, backend=backend, arch="gfx950"
    )
    assert isinstance(config, dict)
    assert "matched:" in config_utils.explain(
        op, params, backend=backend, arch="gfx950"
    )
