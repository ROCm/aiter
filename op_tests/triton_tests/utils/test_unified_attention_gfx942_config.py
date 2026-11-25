# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-side dispatch checks for the packaged gfx942 Triton attention configs."""

from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.utils import unified_attention_utils as config_utils

FP8 = config_utils.e4m3_dtype


def _config(q, head_size=64, q_dtype=FP8, kv_dtype=FP8, **overrides):
    params = SimpleNamespace(
        head_size=head_size,
        head_size_v=head_size,
        max_seqlen_q=q,
        max_seqlen_k=q + 8192,
        sliding_window=0,
        shuffled_kv_cache=False,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_dtype,
        block_size=16,
        num_queries_per_kv=8,
        num_sms=304,
        num_2d_prgms=2 * (q // 16 + 1),
    )
    for name, value in overrides.items():
        setattr(params, name, value)
    return config_utils.get_unified_attention_config(
        "attn_2d", params, backend="triton", arch="gfx942"
    )


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("q", [256, 8192])
def test_fp8_prefill_takes_split_unmasked_loop(head_size, q):
    config = _config(q, head_size=head_size)
    assert config["SPLIT_UNMASKED_LOOP"] is True
    assert config["TILE_SIZE"] == 128
    assert config["num_stages"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"sliding_window": 2048},
        {"shuffled_kv_cache": True, "block_size": 64},
        {"q_dtype": torch.bfloat16, "kv_dtype": torch.bfloat16},
        {"q_dtype": torch.bfloat16},
        {"head_size": 256},
    ],
)
def test_other_prefill_keeps_generic_row(overrides):
    assert not _config(8192, **overrides).get("SPLIT_UNMASKED_LOOP", False)


@pytest.mark.parametrize("q", [1, 16, 255])
def test_fp8_decode_and_short_prefill_unchanged(q):
    assert not _config(q).get("SPLIT_UNMASKED_LOOP", False)
