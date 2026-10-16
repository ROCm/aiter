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
FWD = json.loads(DEFAULT.read_text())["fwd"]
# Real DEFAULT.json entries that DEFAULT.json does not serve this shape, so
# a bucket's tile is a tile the kernel runs and can only come from the file.
SMALL, LARGE = dict(FWD["small_head"]), dict(FWD["pe"])
SHAPE_FILE = "MHA-DQ=128-DV=128-CAUSAL=1.json"


@pytest.fixture
def config_dir():
    with tempfile.TemporaryDirectory() as directory:
        shutil.copy(DEFAULT, directory)
        (Path(directory) / SHAPE_FILE).write_text(
            json.dumps(
                {
                    "DEFAULT_FALLBACK": True,
                    "SK_LEQ_4096": LARGE,
                    "SK_LEQ_1024": SMALL,
                    "SK_LEQ_4k": SMALL,
                }
            )
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


def test_the_tuned_tiles_are_not_what_default_json_serves(config_dir):
    assert SMALL != LARGE
    for max_seqlen_k in (1, 1024, 1025, 4096, 4097):
        assert default_bucket(max_seqlen_k=max_seqlen_k) not in (SMALL, LARGE)


@pytest.mark.parametrize(
    "max_seqlen_k, expected",
    [(1, SMALL), (1024, SMALL), (1025, LARGE), (4096, LARGE)],
)
def test_the_smallest_bucket_that_holds_the_shape_serves_it(
    config_dir, max_seqlen_k, expected
):
    assert resolve(max_seqlen_k=max_seqlen_k) == expected


def test_a_shape_past_the_largest_bucket_gets_default_json(config_dir):
    assert resolve(max_seqlen_k=4097) == default_bucket(max_seqlen_k=4097)


def test_a_bucket_whose_bound_is_not_an_integer_is_skipped(config_dir):
    with mock.patch.object(kernels._LOGGER, "warning") as warning:
        assert resolve(max_seqlen_k=2048) == LARGE
    assert "SK_LEQ_4k" in warning.call_args.args[0]


def test_another_head_dim_or_mask_has_no_file(config_dir):
    assert resolve(causal=False) == default_bucket(causal=False)
    assert resolve(head_dim_qk=192, has_pe=True) == default_bucket(has_pe=True)


def test_only_bf16_reads_the_shape_file(config_dir):
    # The harness tunes bf16; other dtypes keep the DEFAULT.json rules.
    for dtype in (torch.float16, torch.float32):
        assert resolve(dtype=dtype) == default_bucket(dtype=dtype)


def test_dropout_keeps_its_default_json_bucket(config_dir):
    assert resolve(enable_dropout=True) == default_bucket(enable_dropout=True)
