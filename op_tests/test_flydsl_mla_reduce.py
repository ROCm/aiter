# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pytest correctness coverage for the FlyDSL MLA reduce kernel."""

import itertools

import pytest
import torch

from op_tests.flydsl_mla_reduce_common import (
    build_degenerate_inputs,
    build_inputs,
    hip_ref,
    make_runner,
    torch_ref,
)

_MATRIX_SHAPES = [(128, 512), (16, 512), (128, 128)]
_GLM_SHAPES = [(64, 256)]
_MATRIX_DTYPES = ["bf16", "fp16"]
_MATRIX_SPLITS = [2, 3, 8, 16, 64, 256]
_MATRIX_M = [1, 2, 4]
_MATRIX_TILES = 4

_DEGEN_TILES = [2, 4]

_MATRIX_CASES = list(
    itertools.product(_MATRIX_SHAPES, _MATRIX_DTYPES, _MATRIX_SPLITS, _MATRIX_M)
)
_MATRIX_IDS = [
    f"H{H}_Dv{Dv}_{dt}_s{S}_M{M}"
    for (H, Dv), dt, S, M in _MATRIX_CASES
]

_TORCH_REF_SHAPES = _MATRIX_SHAPES + _GLM_SHAPES
_TORCH_REF_CASES = list(
    itertools.product(_TORCH_REF_SHAPES, _MATRIX_DTYPES, _MATRIX_SPLITS, _MATRIX_M)
)
_TORCH_REF_IDS = [
    f"H{H}_Dv{Dv}_{dt}_s{S}_M{M}"
    for (H, Dv), dt, S, M in _TORCH_REF_CASES
]

_DEGEN_CASES = list(
    itertools.product(_TORCH_REF_SHAPES, _MATRIX_DTYPES, _DEGEN_TILES)
)
_DEGEN_IDS = [f"H{H}_Dv{Dv}_{dt}_tiles{T}" for (H, Dv), dt, T in _DEGEN_CASES]


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _out_atol(dtype_str: str) -> float:
    return 6.3e-2 if dtype_str == "bf16" else 2e-3


def _assert_close(fout, flse, ref_out, ref_lse, dt):
    atol = _out_atol(dt)
    out_err = (fout.float() - ref_out.float()).abs().max().item()
    lse_err = (flse - ref_lse).abs().max().item()
    assert out_err <= atol, f"out max_abs_err={out_err:.3e} > {atol}"
    assert lse_err <= 1e-3, f"lse max_abs_err={lse_err:.3e}"


@pytest.mark.parametrize(
    "shape,dt,S,M",
    _MATRIX_CASES,
    ids=_MATRIX_IDS,
)
def test_flydsl_mla_reduce_vs_hip(shape, dt, S, M):
    """FlyDSL output matches HIP kn_mla_reduce_v1 on DeepSeek-shaped (H, Dv) pairs."""
    _require_cuda()
    H, Dv = shape
    out_dtype = torch.bfloat16 if dt == "bf16" else torch.float16
    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
        _MATRIX_TILES, S, H, Dv, out_dtype, M=M
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M
    )
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref(po, pl, indptr, fmap, pmap, _MATRIX_TILES, H, Dv, out_dtype, M)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.parametrize(
    "shape,dt,S,M",
    _TORCH_REF_CASES,
    ids=_TORCH_REF_IDS,
)
def test_flydsl_mla_reduce_vs_torch_ref(shape, dt, S, M):
    """FlyDSL matches torch online-softmax ref (DeepSeek + GLM-5.2 anchor grid, tiles=4)."""
    _require_cuda()
    H, Dv = shape
    out_dtype = torch.bfloat16 if dt == "bf16" else torch.float16
    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
        _MATRIX_TILES, S, H, Dv, out_dtype, M=M
    )
    ref_out, ref_lse = torch_ref(po, pl, _MATRIX_TILES, S, H, Dv, out_dtype, M)
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M)
    run()
    torch.cuda.synchronize()
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.parametrize("shape,dt,num_tiles", _DEGEN_CASES, ids=_DEGEN_IDS)
def test_flydsl_mla_reduce_degenerate_empty_tile(shape, dt, num_tiles):
    """Empty-tile guard skips n_splits=0 tiles and never stores through garbage q-ranges."""
    _require_cuda()
    H, Dv = shape
    out_dtype = torch.bfloat16 if dt == "bf16" else torch.float16
    po, pl, indptr, fmap, pmap, fout, flse = build_degenerate_inputs(
        num_tiles, H, Dv, out_dtype
    )
    fout.fill_(12345.0)
    flse.fill_(12345.0)
    expected_out = fout.clone()
    expected_lse = flse.clone()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True)
    run()
    torch.cuda.synchronize()
    assert torch.equal(fout, expected_out)
    assert torch.equal(flse, expected_lse)
