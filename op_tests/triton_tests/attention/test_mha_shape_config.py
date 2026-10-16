# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MHA forward resolver serves a tuned shape file before DEFAULT.json."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

from aiter.ops.triton._triton_kernels.attention import mha as kernels
from aiter.ops.triton.utils.config_utils import load_config_json

DEFAULT = (
    Path(kernels.__file__).parents[2]
    / "configs/gfx950/triton/attention/mha/DEFAULT.json"
)
TUNED = {
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "PRELOAD_V": False,
    "waves_per_eu": 1,
    "num_warps": 8,
    "num_ctas": 1,
    "num_stages": 2,
}


@pytest.fixture
def config_dir():
    with tempfile.TemporaryDirectory() as directory:
        shutil.copy(DEFAULT, directory)
        (Path(directory) / "MHA-DQ=128-DV=128-CAUSAL=1.json").write_text(
            json.dumps({"DEFAULT_FALLBACK": True, "SK_LEQ_4096": TUNED})
        )
        kernels._get_config.cache_clear()
        load_config_json.cache_clear()
        with mock.patch.object(kernels, "resolve_config_dir", return_value=directory):
            yield
        kernels._get_config.cache_clear()
        load_config_json.cache_clear()


def resolve(**overrides):
    arguments = {
        "enable_dropout": False,
        "dtype": torch.bfloat16,
        "has_pe": False,
        "head_dim_v": 128,
        "causal": True,
        "max_seqlen_q": 2048,
        "max_seqlen_k": 2048,
        "batch_heads": 64,
        "head_dim_qk": 128,
    }
    return kernels._get_config(**{**arguments, **overrides})


def default_bucket(**overrides):
    return resolve(**overrides, head_dim_qk=None)


def test_a_shape_inside_a_tuned_bucket_gets_its_tile(config_dir):
    assert resolve() == TUNED


def test_a_shape_past_the_largest_bucket_gets_default_json(config_dir):
    assert resolve(max_seqlen_k=8192) == default_bucket(max_seqlen_k=8192)


def test_another_head_dim_or_mask_has_no_file(config_dir):
    assert resolve(causal=False) == default_bucket(causal=False)
    assert resolve(head_dim_qk=192, has_pe=True) == default_bucket(has_pe=True)


def test_dropout_and_fp32_keep_their_default_json_buckets(config_dir):
    assert resolve(enable_dropout=True) == default_bucket(enable_dropout=True)
    assert resolve(dtype=torch.float32) == default_bucket(dtype=torch.float32)
