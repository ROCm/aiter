# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU coverage for precision-specific PA tuning keys and synthetic layouts."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Loading this host-only utility directly keeps CPU coverage independent of
# AITER's optional GPU package imports and the FlyDSL SDK.
_spec = importlib.util.spec_from_file_location(
    "pa_decode_tuning_cpu",
    Path(__file__).resolve().parents[1] / "aiter/ops/flydsl/pa_decode_tuning.py",
)
tuning = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tuning)


def _shape(**overrides):
    arguments = {
        "batch_size": 2,
        "context_length": 129,
        "query_length": 4,
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 128,
        "page_size": 64,
        "per_token": False,
        "trans_v": True,
    }
    arguments.update(overrides)
    return tuning.make_shape(**arguments)


@pytest.mark.parametrize(
    "architecture,fp8_dtype",
    [("gfx942", "float8_e4m3fnuz"), ("gfx950", "float8_e4m3fn")],
)
def test_cache_key_separates_native_bf16_from_fp8(architecture, fp8_dtype):
    # Match BF16 Q, per_token=False and transposed V to isolate KV precision.
    fp8_shape = _shape(kv_dtype="fp8")
    bf16_shape = _shape(kv_dtype="bf16")
    fp8_key = tuning.make_key(fp8_shape, architecture, 304)
    bf16_key = tuning.make_key(bf16_shape, architecture, 304)
    assert fp8_key["kv_dtype"] == fp8_dtype
    assert bf16_key["kv_dtype"] == "bfloat16"
    assert fp8_key != bf16_key
    assert bf16_key == tuning.make_key(_shape(kv_dtype="bfloat16"), architecture, 304)
    assert tuning.unique_kv_bytes(bf16_shape) == 2 * tuning.unique_kv_bytes(fp8_shape)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dtype": "float16"}, "BF16 KV requires bfloat16 queries"),
        ({"per_token": True}, "BF16 KV is unscaled"),
        ({"trans_v": False}, "BF16 KV requires vectorized 5D V"),
    ],
    ids=["fp16-query", "per-token", "plain-v"],
)
def test_bf16_tuning_rejects_unsupported_inputs(overrides, match):
    with pytest.raises(ValueError, match=match):
        _shape(kv_dtype="bf16", **overrides)


def test_bf16_tuning_rejects_unsupported_architecture():
    with pytest.raises(ValueError, match="BF16 KV is supported on gfx942 and gfx950"):
        tuning.make_key(_shape(kv_dtype="bf16"), "gfx1250", 256)


@pytest.mark.parametrize("page_size", [16, 64, 128])
def test_bf16_tuning_inputs_use_vector8_without_scales(page_size):
    shape = _shape(kv_dtype="bf16", head_dim=192, page_size=page_size)
    inputs = tuning._make_inputs(torch, shape, torch.device("cpu"), torch.bfloat16)
    pages = shape["kv_pool_pages"]
    assert inputs["query"].dtype == torch.bfloat16
    assert inputs["key"].dtype == torch.bfloat16
    assert inputs["value"].dtype == torch.bfloat16
    assert inputs["key"].shape == (pages, 2, 24, page_size, 8)
    assert inputs["value"].shape == (pages, 2, page_size // 8, 192, 8)
    assert inputs["key"].is_contiguous()
    assert inputs["value"].is_contiguous()
    assert inputs["key_scale"] is None
    assert inputs["value_scale"] is None
    actual_key = tuning.storage_key(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["key_scale"],
        inputs["value_scale"],
    )
    assert actual_key == tuning._synthetic_storage_key(shape)
    assert actual_key[-2:] == (None, None)

    # Unpack both independent layouts and compare a dense causal reference.
    keys = inputs["key"].permute(0, 1, 3, 2, 4).reshape(pages, 2, page_size, 192)
    values = inputs["value"].permute(0, 1, 2, 4, 3).reshape(pages, 2, page_size, 192)
    query = inputs["query"].float().reshape(2, 4, 2, 4, 192)
    expected = torch.empty_like(query)
    for seq, length in enumerate(shape["lengths"]):
        tokens = torch.arange(length)
        page_indices = inputs["table"][seq, tokens // page_size].long()
        offsets = tokens % page_size
        key = keys[page_indices, :, offsets].float()
        value = values[page_indices, :, offsets].float()
        scores = (
            torch.einsum("qhgd,thd->qhgt", query[seq], key) * shape["softmax_scale"]
        )
        visible = length - 4 + 1 + torch.arange(4)
        scores.masked_fill_(
            tokens[None, None, None, :] >= visible[:, None, None, None], -torch.inf
        )
        expected[seq] = torch.einsum("qhgt,thd->qhgd", scores.softmax(-1), value)
    actual = tuning._reference(torch, inputs, shape, chunk_size=31)
    torch.testing.assert_close(
        actual, expected.reshape_as(inputs["query"]), atol=1e-6, rtol=1e-5
    )
