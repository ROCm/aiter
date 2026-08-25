# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage-1 fp8 quant fusion on get_2stage_cfgs' heuristic FlyDSL fallback.

The tuned path picks the fused "_fp8" stage-1 variant per shape (most rows in
tuned_fmoe.csv carry it), which folds stage 2's mxfp8 quant and scale-sort into
stage 1's CShuffle epilogue. The heuristic fallback did not consider it, so any
shape without a tuned row ran a standalone quant+sort kernel that the fused
epilogue makes unnecessary. These tests pin the fallback's choice and both
escape hatches.

The fallback is reached with a header-only config_file: pandas still sees the
shipped schema, but no row can match, so _lookup_cfg returns None. That uses
only the public parameter, rather than reaching into the config cache.
"""

import os

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes, fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available

TUNED_CSV = os.path.join(os.path.dirname(aiter.__file__), "configs", "tuned_fmoe.csv")

# DeepSeek-V4 shape: the fallback is what DSv4 at EP8 actually takes.
MODEL_DIM = 7168
INTER_DIM = 3072
EXPERTS = 256
TOPK = 8
TOKEN = 64

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm device"),
    pytest.mark.skipif(not is_flydsl_available(), reason="needs FlyDSL"),
    pytest.mark.skipif(get_gfx() not in ("gfx950",), reason="mxfp4 a8w4 is gfx950"),
]


@pytest.fixture
def no_tuned_rows(tmp_path):
    with open(TUNED_CSV) as f:
        header = f.readline()
    path = tmp_path / "no_tuned_rows.csv"
    path.write_text(header)
    return str(path)


def _fallback_metadata(config_file, inter_dim=INTER_DIM):
    return fused_moe.get_2stage_cfgs(
        TOKEN,
        MODEL_DIM,
        inter_dim,
        EXPERTS,
        TOPK,
        dtypes.bf16,
        dtypes.fp8,  # a8
        dtypes.fp4x2,  # w4
        QuantType.per_1x32,
        True,  # use_g1u1
        ActivationType.Silu,
        False,  # doweight_stage1
        0,  # hidden_pad
        0,  # intermediate_pad
        config_file=config_file,
    )


def test_fallback_fuses_stage1_fp8_quant(no_tuned_rows):
    """Without a tuned row, stage 1 is still promoted to the fused fp8 variant."""
    metadata = _fallback_metadata(no_tuned_rows)
    assert metadata.fuse_quant == "fp8"


def test_env_disables_fusion(no_tuned_rows, monkeypatch):
    """AITER_S1_FUSE_FP8Q=0 restores the unfused bf16 stage 1."""
    monkeypatch.setenv("AITER_S1_FUSE_FP8Q", "0")
    metadata = _fallback_metadata(no_tuned_rows)
    assert metadata.fuse_quant == ""


def test_min_inter_dim_floor_blocks_fusion(no_tuned_rows, monkeypatch):
    """A floor above inter_dim opts the shape out."""
    monkeypatch.setattr(fused_moe, "_S1_FUSE_FP8Q_MIN_INTER_DIM", INTER_DIM + 1)
    metadata = _fallback_metadata(no_tuned_rows)
    assert metadata.fuse_quant == ""


def test_env_forces_fusion_below_floor(no_tuned_rows, monkeypatch):
    """AITER_S1_FUSE_FP8Q=1 overrides the floor, which is how the sweep that
    set the default to always-fuse measured shapes below it."""
    monkeypatch.setattr(fused_moe, "_S1_FUSE_FP8Q_MIN_INTER_DIM", INTER_DIM + 1)
    monkeypatch.setenv("AITER_S1_FUSE_FP8Q", "1")
    metadata = _fallback_metadata(no_tuned_rows)
    assert metadata.fuse_quant == "fp8"
