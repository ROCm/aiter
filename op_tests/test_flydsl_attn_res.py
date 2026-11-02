# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test and benchmark for FlyDSL AttnRes on the ``attn_res_gate`` contract.

Usage:
    pytest -q op_tests/test_flydsl_attn_res.py
    python op_tests/test_flydsl_attn_res.py
    python op_tests/test_flydsl_attn_res.py --tokens 17 320 --blocks 7 8
    python op_tests/test_flydsl_attn_res.py --cache warm cold
"""

import argparse
import dataclasses
import itertools
import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_attn_res
from aiter.ops.flydsl.kernels import attn_res as attn_res_kernel
from aiter.ops.torch_ref.attn_res import attn_res as attn_res_reference
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

_D = 7168
_D_SMALL = 1024
_D_UNSUPPORTED = 520
_MAX_BLOCKS = 8
_EPS = 1e-5
_OUT_EPS = 1e-6
# One BF16 ulp is at most 2**-7 of the value. The floor, relative to the row
# peak, covers FP32 summation-order differences between two implementations.
_BF16_ULP_REL = 2.0**-7
_FP32_FLOOR_REL = 2.0**-12
# e4m3 keeps 3 mantissa bits, so one FP8 step is at most 2**-3 of the value.
_FP8_STEP_REL = 2.0**-3
# One FP8 step is relative only above e4m3's subnormals; near zero the slack is
# absolute, in units of the per-token scale.
_FP8_NEAR_ZERO_SLACK_SCALES = 1.0
# run_perftest num_rotate_args: 1 reuses one arg set (L2-resident), 0 rotates
# copies so each timed iteration reads cold HBM.
_ROTATE_ARGS = {"warm": 1, "cold": 0}
# A correctly rounded BF16 output is within 2**-8 of its row's peak magnitude;
# 1e-5 covers FP32 accumulation before that rounding.
_ORACLE_MAX_REL = 2.0**-8 + 1e-5
_SUPPORTED_GFX = ("gfx942", "gfx950")


def _supported_gfx() -> bool:
    try:
        return get_gfx() in _SUPPORTED_GFX
    except Exception:  # noqa: BLE001
        return False


_skip_unsupported_gfx = pytest.mark.skipif(
    not _supported_gfx(),
    reason="flydsl_attn_res requires gfx942 or gfx950",
)


@dataclasses.dataclass(frozen=True)
class _Config:
    add: bool = False
    add2: bool = False
    close: bool = False
    onorm: bool = False
    fp8: bool = False
    scale: float = 1.0


MIX = _Config()
ALL_ON = _Config(add=True, close=True, onorm=True)
ALL_ON_FP8 = dataclasses.replace(ALL_ON, fp8=True)
EVERY_FEATURE = _Config(add=True, add2=True, close=True, onorm=True, scale=0.7)
_BENCH_CONFIGS = {"mix": MIX, "all-on": ALL_ON, "all-on fp8": ALL_ON_FP8}


def _flag_surface():
    """Every valid combination: add2 needs add, and fp8 needs the output norm."""
    adds = ((False, False), (True, False), (True, True))
    norms = ((False, False), (True, False), (True, True))
    for (add, add2), close, (onorm, fp8) in itertools.product(
        adds, (False, True), norms
    ):
        yield _Config(add=add, add2=add2, close=close, onorm=onorm, fp8=fp8)


def _num_cus() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _make_inputs(
    tokens: int,
    hidden_size: int = _D,
    num_blocks: int = _MAX_BLOCKS,
    pad: int = 0,
    max_plus_one_parent: bool = False,
    leading: tuple[int, ...] = (),
):
    torch.manual_seed(20260804 + tokens + hidden_size + num_blocks + pad)
    rows = (*leading, tokens)
    width = hidden_size + pad
    parents = {
        name: torch.randn(*rows, width, dtype=torch.bfloat16)
        for name in ("prefix", "add_hidden", "add_hidden2")
    }
    parent_rows = _MAX_BLOCKS + 1 if max_plus_one_parent else num_blocks
    parents["block_residual"] = torch.randn(
        *rows, parent_rows, width, dtype=torch.bfloat16
    )
    views = {name: parent[..., :hidden_size] for name, parent in parents.items()}
    views["block_residual"] = parents["block_residual"][..., :num_blocks, :hidden_size]
    score_weight = (
        (1.0 + 0.1 * torch.randn(hidden_size))
        * torch.randn(hidden_size)
        / math.sqrt(hidden_size)
    )
    return SimpleNamespace(
        **views,
        score_weight=score_weight.to(torch.bfloat16),
        output_rms_weight=(1.0 + 0.1 * torch.randn(hidden_size)).to(torch.bfloat16),
        parents=parents,
    )


def _gate_args(inputs, cfg: _Config):
    args = (
        inputs.prefix,
        inputs.block_residual,
        inputs.score_weight,
        _EPS,
        inputs.add_hidden if cfg.add else None,
        inputs.add_hidden2 if cfg.add2 else None,
    )
    kwargs = {
        "output_rms_weight": inputs.output_rms_weight if cfg.onorm else None,
        "output_rms_eps": _OUT_EPS,
        "scale": cfg.scale,
        "close_block": cfg.close,
        "out_quant_dtype": dtypes.fp8 if cfg.fp8 else None,
    }
    return args, kwargs


def _call(fn, inputs, cfg: _Config, **extra):
    args, kwargs = _gate_args(inputs, cfg)
    return fn(*args, **kwargs, **extra)


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor, msg: str):
    actual, expected = actual.float(), expected.float()
    peak = expected.abs().amax(-1, keepdim=True)
    bound = _BF16_ULP_REL * expected.abs() + _FP32_FLOOR_REL * peak
    excess = ((actual - expected).abs() - bound).max().item()
    assert not torch.isnan(actual).any(), f"{msg}: NaN in output"
    assert excess <= 0, f"{msg}: error exceeds the BF16 bound by {excess:.3e}"


def _assert_fp8_close(actual, expected, msg: str):
    (y_fp8, y_scale), (ref_fp8, ref_scale) = actual, expected
    assert y_fp8.dtype == dtypes.fp8 and y_scale.dtype == torch.float32, msg
    assert y_scale.shape == y_fp8.shape[:-1] + (1,), msg
    torch.testing.assert_close(
        y_scale, ref_scale, atol=0.0, rtol=_FP32_FLOOR_REL, msg=msg
    )
    deq = y_fp8.float() * y_scale
    ref = ref_fp8.float() * ref_scale
    assert not torch.isnan(deq).any(), f"{msg}: NaN in FP8 output"
    bound = _FP8_STEP_REL * ref.abs() + _FP8_NEAR_ZERO_SLACK_SCALES * ref_scale
    assert ((deq - ref).abs() <= bound).all(), f"{msg}: FP8 values off by > 1 step"


def _assert_y_close(actual, expected, cfg: _Config, msg: str):
    if cfg.fp8:
        _assert_fp8_close(actual, expected, msg)
    else:
        _assert_bf16_close(actual, expected, msg)


def _row_peak_error(actual, expected, cfg: _Config) -> float:
    if cfg.fp8:
        actual = actual[0].float() * actual[1]
        expected = expected[0].float() * expected[1]
    actual, expected = actual.float(), expected.float()
    peak = expected.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    return ((actual - expected).abs() / peak).max().item()


def _run_and_check(
    tokens: int,
    num_blocks: int,
    cfg: _Config,
    pad: int = 0,
    max_plus_one_parent: bool = False,
    hidden_size: int = _D,
    rows_per_wg: int | None = None,
    leading: tuple[int, ...] = (),
) -> float:
    inputs = _make_inputs(
        tokens, hidden_size, num_blocks, pad, max_plus_one_parent, leading
    )
    parents_before = {k: v.clone() for k, v in inputs.parents.items()}
    msg = (
        f"AttnRes {cfg} T={tokens} D={hidden_size} B={num_blocks} pad={pad} "
        f"max_plus_one_parent={max_plus_one_parent} leading={leading} "
        f"rows_per_wg={rows_per_wg}"
    )

    expected = _call(attn_res_reference, inputs, cfg)
    out = _call(flydsl_attn_res, inputs, cfg, rows_per_wg=rows_per_wg)
    torch.cuda.synchronize()

    assert len(out) == (3 if cfg.close else 2), msg
    y, prefix_out = out[0], out[1]
    y_values = y[0] if cfg.fp8 else y
    assert y_values.shape == inputs.prefix.shape, msg
    assert y_values.is_contiguous(), msg
    _assert_y_close(y, expected[0], cfg, msg)

    if cfg.add:
        assert prefix_out.data_ptr() != inputs.prefix.data_ptr(), msg
        assert torch.equal(prefix_out, expected[1]), f"{msg}: prefix_out mismatch"
    else:
        assert prefix_out is inputs.prefix, f"{msg}: prefix_out must be prefix"
    if cfg.close:
        assert torch.equal(out[2], expected[2]), f"{msg}: block_out mismatch"

    for name, parent in inputs.parents.items():
        assert torch.equal(parent, parents_before[name]), f"{msg}: mutated {name}"
    return _row_peak_error(y, expected[0], cfg)


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", tuple(_flag_surface()), ids=str)
def test_flydsl_attn_res_flag_surface(cfg: _Config):
    _run_and_check(17, _MAX_BLOCKS - 1, cfg)


@_skip_unsupported_gfx
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
@pytest.mark.parametrize("num_blocks", range(_MAX_BLOCKS + 1))
def test_flydsl_attn_res_num_blocks(hidden_size: int, num_blocks: int):
    for cfg in (MIX, EVERY_FEATURE, ALL_ON_FP8):
        _run_and_check(17, num_blocks, cfg, hidden_size=hidden_size)


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", (MIX, EVERY_FEATURE, ALL_ON_FP8), ids=str)
def test_flydsl_attn_res_layouts(cfg: _Config):
    _run_and_check(17, _MAX_BLOCKS - 1, cfg, pad=8)
    _run_and_check(17, _MAX_BLOCKS, cfg, pad=16)
    _run_and_check(17, 5, cfg, max_plus_one_parent=True)
    _run_and_check(9, 3, cfg, leading=(2,))


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", (MIX, ALL_ON, ALL_ON_FP8), ids=str)
def test_flydsl_attn_res_token_counts(cfg: _Config):
    """T > num_cus switches D=7168 to the 256-thread wide grid; test both sides."""
    for tokens in (1, _num_cus(), _num_cus() + 1):
        _run_and_check(tokens, _MAX_BLOCKS, cfg)


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", (MIX, EVERY_FEATURE, ALL_ON_FP8), ids=str)
def test_flydsl_attn_res_rows_per_wg(cfg: _Config):
    """T=17 leaves three idle waves in the last workgroup."""
    _run_and_check(17, _MAX_BLOCKS, cfg, hidden_size=_D_SMALL, rows_per_wg=4)


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", (MIX, ALL_ON_FP8), ids=str)
def test_flydsl_attn_res_empty(cfg: _Config):
    inputs = _make_inputs(0)
    out = _call(flydsl_attn_res, inputs, cfg)
    expected = _call(attn_res_reference, inputs, cfg)

    def leaves(outputs):
        for item in outputs:
            yield from item if isinstance(item, tuple) else (item,)

    assert len(out) == len(expected)
    for got, want in zip(leaves(out), leaves(expected), strict=True):
        assert got.shape == want.shape
    y_dtypes = (dtypes.fp8, torch.float32) if cfg.fp8 else (torch.bfloat16,)
    assert tuple(t.dtype for t in leaves(out[:1])) == y_dtypes


@_skip_unsupported_gfx
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
def test_flydsl_attn_res_prefix_candidate_is_unrounded(hidden_size: int):
    """Mix the FP32 sum ``256 + 0.5``, not its BF16 rounding 256.

    A zero ``score_weight`` gives both candidates weight 1/2, so ``y`` is
    ``(-256 + 256.5) / 2 = 0.25`` exactly; a rounded candidate gives 0.
    """
    prefix = torch.full((17, hidden_size), 256.0, dtype=torch.bfloat16)
    add_hidden = torch.full_like(prefix, 0.5)
    block_residual = torch.full((17, 1, hidden_size), -256.0, dtype=torch.bfloat16)
    score_weight = torch.zeros(hidden_size, dtype=torch.bfloat16)

    y, prefix_out = flydsl_attn_res(
        prefix, block_residual, score_weight, _EPS, add_hidden
    )
    assert torch.equal(y, torch.full_like(prefix, 0.25))
    assert torch.equal(prefix_out, prefix)


@_skip_unsupported_gfx
@pytest.mark.parametrize(("eps", "output_rms_eps"), ((_EPS, 1.0), (1.0, _OUT_EPS)))
def test_flydsl_attn_res_eps_are_independent(eps: float, output_rms_eps: float):
    inputs = _make_inputs(17, num_blocks=3)
    args = (inputs.prefix, inputs.block_residual, inputs.score_weight, eps)
    kwargs = {
        "output_rms_weight": inputs.output_rms_weight,
        "output_rms_eps": output_rms_eps,
    }
    expected, _ = attn_res_reference(*args, **kwargs)
    y, _ = flydsl_attn_res(*args, **kwargs)
    _assert_bf16_close(y, expected, f"eps={eps} output_rms_eps={output_rms_eps}")


@_skip_unsupported_gfx
def test_flydsl_attn_res_fp8_all_zero_row():
    inputs = _make_inputs(17, num_blocks=3)
    inputs.prefix.zero_()
    inputs.block_residual.zero_()
    (y_fp8, y_scale), _ = flydsl_attn_res(
        inputs.prefix,
        inputs.block_residual,
        inputs.score_weight,
        _EPS,
        output_rms_weight=inputs.output_rms_weight,
        out_quant_dtype=dtypes.fp8,
    )
    assert torch.equal(y_scale, torch.zeros_like(y_scale))
    assert torch.equal(y_fp8.float(), torch.zeros_like(y_fp8, dtype=torch.float32))


@_skip_unsupported_gfx
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
@pytest.mark.parametrize("tokens", (17, 320))
@pytest.mark.parametrize("num_blocks", (1, _MAX_BLOCKS - 1, _MAX_BLOCKS))
@pytest.mark.parametrize("cfg", (MIX, EVERY_FEATURE, ALL_ON_FP8), ids=str)
def test_flydsl_attn_res_matches_triton_gate(
    hidden_size: int, tokens: int, num_blocks: int, cfg: _Config
):
    from aiter.ops.triton.fusions.attn_res import attn_res_gate

    inputs = _make_inputs(tokens, hidden_size, num_blocks, max_plus_one_parent=True)
    triton_out = _call(attn_res_gate, inputs, cfg)
    out = _call(flydsl_attn_res, inputs, cfg)
    torch.cuda.synchronize()

    msg = f"vs Triton {cfg} T={tokens} D={hidden_size} B={num_blocks}"
    assert len(out) == len(triton_out), msg
    _assert_y_close(out[0], triton_out[0], cfg, msg)
    assert torch.equal(out[1], triton_out[1]), f"{msg}: prefix_out differs"
    if cfg.close:
        assert torch.equal(out[2], triton_out[2]), f"{msg}: block_out differs"


@_skip_unsupported_gfx
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
@pytest.mark.parametrize(
    ("num_blocks", "cfg"), ((_MAX_BLOCKS, MIX), (_MAX_BLOCKS, EVERY_FEATURE))
)
def test_flydsl_attn_res_float64_oracle(hidden_size: int, num_blocks: int, cfg):
    inputs = _make_inputs(17, hidden_size, num_blocks)
    oracle = _call(attn_res_reference, inputs, cfg, dtype=torch.float64)[0]
    output = _call(flydsl_attn_res, inputs, cfg)[0]
    torch.cuda.synchronize()

    row_peak = oracle.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    max_rel = ((output.to(torch.float64) - oracle).abs() / row_peak).max().item()
    assert max_rel < _ORACLE_MAX_REL, (
        "float64 oracle max relative error vs row peak "
        f"{max_rel:.6e} exceeds {_ORACLE_MAX_REL:.6e} "
        f"(D={hidden_size}, B={num_blocks}, {cfg})"
    )


@_skip_unsupported_gfx
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
@pytest.mark.parametrize("cfg", (MIX, ALL_ON, ALL_ON_FP8), ids=str)
def test_flydsl_attn_res_without_permlane_swap(monkeypatch, hidden_size: int, cfg):
    """The shuffle fallback for targets without ``v_permlane*_swap`` gives the same results."""
    monkeypatch.setattr(attn_res_kernel, "_has_permlane_swap", lambda: False)
    attn_res_kernel._build_attn_res.cache_clear()
    try:
        for tokens in (1, 17, _num_cus() + 1):
            _run_and_check(tokens, _MAX_BLOCKS - 1, cfg, hidden_size=hidden_size)
    finally:
        attn_res_kernel._build_attn_res.cache_clear()


def _invalid_call(case: str):
    hidden_size = _D_UNSUPPORTED if case == "unsupported_hidden_size" else _D
    tokens = 0 if case == "unsupported_hidden_size" else 1
    num_blocks = _MAX_BLOCKS + 1 if case == "too_many_blocks" else 2
    inputs = _make_inputs(
        tokens, hidden_size, num_blocks, pad=7 if case == "pad7" else 0
    )
    kwargs = {}
    args = [inputs.prefix, inputs.block_residual, inputs.score_weight, _EPS]
    error = ValueError
    if case == "add2_without_add":
        args += [None, inputs.add_hidden2]
    elif case == "quant_without_output_norm":
        kwargs["out_quant_dtype"] = dtypes.fp8
    elif case == "unsupported_quant_dtype":
        kwargs["output_rms_weight"] = inputs.output_rms_weight
        kwargs["out_quant_dtype"] = torch.float8_e5m2
    elif case == "fp16_input":
        args[0] = inputs.prefix.half()
        error = TypeError
    elif case == "row_count_mismatch":
        args[1] = torch.randn(2, num_blocks, _D, dtype=torch.bfloat16)
    elif case == "non_unit_trailing_stride":
        args[0] = torch.randn(1, 2 * _D, dtype=torch.bfloat16)[:, ::2]
    elif case == "misaligned_base":
        args[0] = torch.randn(1, _D + 8, dtype=torch.bfloat16)[:, 1 : 1 + _D]
    elif case == "misaligned_weight":
        args[2] = torch.randn(_D + 8, dtype=torch.bfloat16)[1 : 1 + _D]
    elif case == "rows_per_wg_over_block_limit":
        inputs = _make_inputs(1, _D_SMALL, num_blocks)
        args = [inputs.prefix, inputs.block_residual, inputs.score_weight, _EPS]
        kwargs["rows_per_wg"] = 17
    return (lambda: flydsl_attn_res(*args, **kwargs)), error


@_skip_unsupported_gfx
@pytest.mark.parametrize(
    "case",
    (
        "add2_without_add",
        "quant_without_output_norm",
        "unsupported_quant_dtype",
        "fp16_input",
        "too_many_blocks",
        "row_count_mismatch",
        "unsupported_hidden_size",
        "pad7",
        "non_unit_trailing_stride",
        "misaligned_base",
        "misaligned_weight",
        "rows_per_wg_over_block_limit",
    ),
)
def test_flydsl_attn_res_rejects_invalid_inputs(case: str):
    call, error = _invalid_call(case)
    with pytest.raises(error):
        call()


@_skip_unsupported_gfx
def test_flydsl_attn_res_rejects_non_wave64_live_gpu(monkeypatch):
    monkeypatch.setattr(attn_res_kernel, "get_gfx_runtime", lambda: "gfx1250")
    inputs = _make_inputs(1, num_blocks=2)

    with pytest.raises(ValueError, match="wave64"):
        _call(flydsl_attn_res, inputs, MIX)


@_skip_unsupported_gfx
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_flydsl_attn_res_rejects_stream_on_another_device():
    inputs = _make_inputs(1, num_blocks=2)

    with pytest.raises(ValueError, match="stream must be on"):
        _call(flydsl_attn_res, inputs, MIX, stream=torch.cuda.Stream(device=1))


@_skip_unsupported_gfx
def test_flydsl_attn_res_rejects_tensors_over_4gib():
    inputs = _make_inputs(1)
    block_row_bytes = _MAX_BLOCKS * _D * torch.bfloat16.itemsize
    tokens = attn_res_kernel._BUFFER_LIMIT_BYTES // block_row_bytes + 1
    prefix = torch.empty(tokens, _D, dtype=torch.bfloat16)
    block_residual = torch.empty(tokens, _MAX_BLOCKS, _D, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="4 GiB"):
        flydsl_attn_res(prefix, block_residual, inputs.score_weight, _EPS)


def _plan_count() -> int:
    return len(attn_res_kernel._LAUNCH_PLANS)


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", (MIX, EVERY_FEATURE, ALL_ON_FP8), ids=str)
@pytest.mark.parametrize(
    "layout",
    ({}, {"pad": 8}, {"leading": (2,), "tokens": 9}, {"hidden_size": _D_SMALL}),
    ids=str,
)
def test_flydsl_attn_res_plan_reuse(cfg: _Config, layout: dict):
    attn_res_kernel._LAUNCH_PLANS.clear()
    layout = {"tokens": 17, **layout}
    _run_and_check(num_blocks=3, cfg=cfg, **layout)
    assert _plan_count() == 1
    _run_and_check(num_blocks=3, cfg=cfg, **layout)
    assert _plan_count() == 1


@_skip_unsupported_gfx
def test_flydsl_attn_res_plan_follows_every_option():
    inputs = _make_inputs(17, num_blocks=3)
    add_norm = dataclasses.replace(ALL_ON, close=False)
    for cfg in (MIX, ALL_ON, add_norm, ALL_ON, ALL_ON_FP8, ALL_ON, EVERY_FEATURE, MIX):
        expected = _call(attn_res_reference, inputs, cfg)
        out = _call(flydsl_attn_res, inputs, cfg)
        assert len(out) == len(expected), f"{cfg}"
        _assert_y_close(out[0], expected[0], cfg, f"{cfg}")
    args = (inputs.prefix, inputs.block_residual, inputs.score_weight)
    for eps, output_rms_eps, scale in (
        (_EPS, _OUT_EPS, 1.0),
        (1.0, _OUT_EPS, 1.0),
        (_EPS, 1.0, 1.0),
        (_EPS, _OUT_EPS, 0.5),
        (_EPS, _OUT_EPS, 1.0),
    ):
        kwargs = {
            "output_rms_weight": inputs.output_rms_weight,
            "output_rms_eps": output_rms_eps,
            "scale": scale,
        }
        expected, _ = attn_res_reference(*args, eps, **kwargs)
        y, _ = flydsl_attn_res(*args, eps, **kwargs)
        _assert_bf16_close(y, expected, f"{eps=} {output_rms_eps=} {scale=}")


@_skip_unsupported_gfx
@pytest.mark.parametrize(
    "case",
    (
        "add2_without_add",
        "quant_without_output_norm",
        "fp16_input",
        "row_count_mismatch",
        "non_unit_trailing_stride",
        "misaligned_base",
        "misaligned_weight",
        "pad7",
    ),
)
def test_flydsl_attn_res_failed_validation_is_not_cached(case: str):
    attn_res_kernel._LAUNCH_PLANS.clear()
    call, error = _invalid_call(case)
    for _ in range(2):
        with pytest.raises(error):
            call()
        assert _plan_count() == 0


@_skip_unsupported_gfx
def test_flydsl_attn_res_validates_after_the_geometry_is_cached():
    inputs = _make_inputs(1, _D_SMALL, 2)
    block_residual = inputs.block_residual
    row = torch.randn(1, _D_SMALL + 8, dtype=torch.bfloat16)
    weight = torch.randn(_D_SMALL + 8, dtype=torch.bfloat16)
    prefix, score_weight = row[:, :_D_SMALL], weight[:_D_SMALL]
    for _ in range(2):
        flydsl_attn_res(prefix, block_residual, score_weight, _EPS)

    strided = torch.randn(1, 2 * _D_SMALL, dtype=torch.bfloat16)[:, ::2]
    for args, error in (
        ((row[:, 1 : _D_SMALL + 1], block_residual, score_weight), ValueError),
        ((prefix, block_residual, weight[1 : _D_SMALL + 1]), ValueError),
        ((strided, block_residual, score_weight), ValueError),
        ((prefix.half(), block_residual, score_weight), TypeError),
        ((prefix.cpu(), block_residual, score_weight), ValueError),
    ):
        with pytest.raises(error):
            flydsl_attn_res(*args, _EPS)
    with pytest.raises(ValueError):
        flydsl_attn_res(prefix, block_residual, score_weight, _EPS, rows_per_wg=17)


@_skip_unsupported_gfx
def test_flydsl_attn_res_launches_on_the_current_stream(monkeypatch):
    inputs = _make_inputs(17, num_blocks=3)
    expected = _call(attn_res_reference, inputs, MIX)
    torch.cuda.synchronize()
    launched = []
    run_compiled = attn_res_kernel._run_compiled
    monkeypatch.setattr(
        attn_res_kernel,
        "_run_compiled",
        lambda *args: (launched.append(args[-1]), run_compiled(*args))[1],
    )
    side = torch.cuda.Stream()
    for _ in range(2):
        _call(flydsl_attn_res, inputs, MIX)
        with torch.cuda.stream(side):
            out = _call(flydsl_attn_res, inputs, MIX)
        side.synchronize()
        _assert_y_close(out[0], expected[0], MIX, "side stream")
        explicit = _call(flydsl_attn_res, inputs, MIX, stream=side)
        side.synchronize()
        _assert_y_close(explicit[0], expected[0], MIX, "explicit stream")
    default_raw = torch.cuda.default_stream().cuda_stream
    assert launched[0::3] == [default_raw] * 2
    assert launched[1::3] == [side.cuda_stream] * 2
    assert launched[2::3] == [side] * 2


@_skip_unsupported_gfx
def test_flydsl_attn_res_enters_the_device_only_when_it_differs(monkeypatch):
    inputs = _make_inputs(17, num_blocks=3)
    expected = _call(attn_res_reference, inputs, MIX)
    entered = []

    class SpyDevice(torch.cuda.device):
        def __init__(self, index):
            entered.append(index)
            super().__init__(index)

    monkeypatch.setattr(torch.cuda, "device", SpyDevice)

    _assert_y_close(_call(flydsl_attn_res, inputs, MIX)[0], expected[0], MIX, "same")
    assert not entered

    monkeypatch.setattr(torch.cuda, "current_device", lambda: -1)
    _assert_y_close(_call(flydsl_attn_res, inputs, MIX)[0], expected[0], MIX, "other")
    assert entered == [inputs.prefix.device]


@_skip_unsupported_gfx
def test_flydsl_attn_res_plan_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(attn_res_kernel, "_MAX_LAUNCH_PLANS", 3)
    attn_res_kernel._LAUNCH_PLANS.clear()
    for tokens in (1, 2, 3, 4, 5, 2):
        _run_and_check(tokens, 3, MIX, hidden_size=_D_SMALL)
        assert _plan_count() <= 3


@_skip_unsupported_gfx
def test_flydsl_attn_res_plan_cache_cap_holds():
    """Zero tokens builds a plan without compiling a kernel, so filling the cache past its cap stays cheap."""
    assert attn_res_kernel._MAX_LAUNCH_PLANS == 1024
    attn_res_kernel._LAUNCH_PLANS.clear()
    inputs = _make_inputs(0, _D, 1)
    for index in range(attn_res_kernel._MAX_LAUNCH_PLANS + 10):
        args, kwargs = _gate_args(inputs, MIX)
        flydsl_attn_res(*args[:3], 1e-6 * (index + 1), *args[4:], **kwargs)
        assert _plan_count() <= attn_res_kernel._MAX_LAUNCH_PLANS
    assert _plan_count() == 10


_BLOCK_OUT_CFGS = tuple(
    dataclasses.replace(cfg, close=False)
    for cfg in (MIX, ALL_ON, EVERY_FEATURE, ALL_ON_FP8)
)


def _cap_inputs(tokens: int, hidden_size: int, num_blocks: int, slots: int):
    inputs = _make_inputs(tokens, hidden_size, num_blocks)
    cap = torch.randn(tokens, slots, hidden_size, dtype=torch.bfloat16)
    inputs.block_residual = cap[:, :num_blocks]
    return inputs, cap


def _expected_closed(inputs, cfg: _Config):
    return _call(attn_res_reference, inputs, dataclasses.replace(cfg, close=True))


def _spy_block_alias(monkeypatch) -> list[bool]:
    seen: list[bool] = []
    build = attn_res_kernel._build_attn_res

    def spy(spec):
        seen.append(spec.block_alias)
        return build(spec)

    monkeypatch.setattr(attn_res_kernel, "_build_attn_res", spy)
    return seen


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", _BLOCK_OUT_CFGS, ids=str)
@pytest.mark.parametrize("num_blocks", (0, 1, _MAX_BLOCKS - 1))
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
@pytest.mark.parametrize("wide", (False, True), ids=("narrow", "wide"))
def test_flydsl_attn_res_block_out_aliased(
    monkeypatch, wide: bool, hidden_size: int, num_blocks: int, cfg: _Config
):
    tokens = _num_cus() + 1 if wide else 17
    inputs, cap = _cap_inputs(tokens, hidden_size, num_blocks, _MAX_BLOCKS + 1)
    cap_before = cap.clone()
    expected = _expected_closed(inputs, cfg)
    attn_res_kernel._LAUNCH_PLANS.clear()
    seen = _spy_block_alias(monkeypatch)

    out = _call(flydsl_attn_res, inputs, cfg, block_out=cap[:, : num_blocks + 1])
    torch.cuda.synchronize()

    assert seen == [num_blocks > 0]
    assert len(out) == 3
    assert out[2].data_ptr() == cap.data_ptr()
    assert out[2].shape == (tokens, num_blocks + 1, hidden_size)
    _assert_y_close(out[0], expected[0], cfg, f"aliased {cfg}")
    assert torch.equal(out[1], expected[1])
    assert torch.equal(cap[:, : num_blocks + 1], expected[2])
    assert torch.equal(cap[:, :num_blocks], cap_before[:, :num_blocks])
    assert torch.equal(cap[:, num_blocks + 1 :], cap_before[:, num_blocks + 1 :])


@_skip_unsupported_gfx
@pytest.mark.parametrize("cfg", _BLOCK_OUT_CFGS, ids=str)
@pytest.mark.parametrize("hidden_size", (_D, _D_SMALL))
@pytest.mark.parametrize("wide", (False, True), ids=("narrow", "wide"))
def test_flydsl_attn_res_block_out_separate_buffer(
    monkeypatch, wide: bool, hidden_size: int, cfg: _Config
):
    tokens = _num_cus() + 1 if wide else 17
    num_blocks = _MAX_BLOCKS - 1
    inputs = _make_inputs(tokens, hidden_size, num_blocks)
    closed = dataclasses.replace(cfg, close=True)
    expected = _call(attn_res_reference, inputs, closed)
    closed_out = _call(flydsl_attn_res, inputs, closed)
    attn_res_kernel._LAUNCH_PLANS.clear()
    seen = _spy_block_alias(monkeypatch)

    cap = torch.randn(tokens, _MAX_BLOCKS + 1, hidden_size, dtype=torch.bfloat16)
    cap_before = cap.clone()
    out = _call(flydsl_attn_res, inputs, cfg, block_out=cap[:, : num_blocks + 1])
    torch.cuda.synchronize()

    assert seen == [False]
    assert torch.equal(out[2], closed_out[2])
    assert torch.equal(cap[:, : num_blocks + 1], expected[2])
    assert torch.equal(cap[:, num_blocks + 1 :], cap_before[:, num_blocks + 1 :])
    _assert_y_close(out[0], closed_out[0], cfg, f"separate {cfg}")


@_skip_unsupported_gfx
def test_flydsl_attn_res_block_out_empty():
    inputs, cap = _cap_inputs(0, _D, 2, 4)
    out = _call(flydsl_attn_res, inputs, ALL_ON, block_out=cap[:, :3])
    assert out[2].shape == (0, 3, _D)


def _block_out_overlap_call(case: str):
    num_blocks = 3
    inputs, cap = _cap_inputs(4, _D, num_blocks, 6)
    block_out = cap[:, : num_blocks + 1]
    if case == "shifted_view_of_block_residual":
        block_out = cap[:, 1 : num_blocks + 2]
    elif case == "row_b_inside_block_residual":
        inputs.block_residual = cap[:, 1 : num_blocks + 1]
    elif case == "row_b_is_prefix":
        inputs.prefix = cap[:, num_blocks]
    elif case == "add_hidden_inside_block_out":
        inputs.block_residual = torch.randn(4, num_blocks, _D, dtype=torch.bfloat16)
        inputs.add_hidden = cap[:, 2]
    elif case == "prefix_rows_inside_block_out_other_stride":
        inputs.block_residual = torch.randn(4, num_blocks, _D, dtype=torch.bfloat16)
        inputs.prefix = cap.reshape(-1, _D)[:4]
    elif case == "interleaved_views":
        cap = torch.randn(4, 2 * (num_blocks + 1), _D, dtype=torch.bfloat16)
        inputs.block_residual = cap[:, 1::2][:, :num_blocks]
        block_out = cap[:, 0::2]
    elif case == "block_out_rows_overlap":
        base = torch.randn((4 + num_blocks) * _D, dtype=torch.bfloat16)
        block_out = base.as_strided((4, num_blocks + 1, _D), (_D, _D, 1))
        inputs.block_residual = torch.randn(4, num_blocks, _D, dtype=torch.bfloat16)
    return lambda: _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)


@_skip_unsupported_gfx
@pytest.mark.parametrize(
    "case",
    (
        "shifted_view_of_block_residual",
        "row_b_inside_block_residual",
        "row_b_is_prefix",
        "add_hidden_inside_block_out",
        "prefix_rows_inside_block_out_other_stride",
        "interleaved_views",
        "block_out_rows_overlap",
    ),
)
def test_flydsl_attn_res_block_out_rejects_overlap(case: str):
    with pytest.raises(ValueError, match="overlap"):
        _block_out_overlap_call(case)()


@_skip_unsupported_gfx
@pytest.mark.parametrize(
    ("case", "error"),
    (
        ("wrong_rows", ValueError),
        ("wrong_hidden_size", ValueError),
        ("fp16", TypeError),
        ("wrong_device", ValueError),
        ("not_flat_viewable", ValueError),
    ),
)
def test_flydsl_attn_res_block_out_rejects_invalid(case: str, error):
    num_blocks = 3
    inputs = _make_inputs(4, _D, num_blocks)
    shape = (4, num_blocks + 1, _D)
    block_out = torch.empty(shape, dtype=torch.bfloat16)
    if case == "wrong_rows":
        block_out = torch.empty(4, num_blocks, _D, dtype=torch.bfloat16)
    elif case == "wrong_hidden_size":
        block_out = torch.empty(4, num_blocks + 1, _D + 8, dtype=torch.bfloat16)
    elif case == "fp16":
        block_out = block_out.half()
    elif case == "wrong_device":
        block_out = block_out.cpu()
    elif case == "not_flat_viewable":
        inputs = _make_inputs(6, _D, num_blocks, leading=(2,))
        swapped = torch.empty(6, 2, num_blocks + 1, _D, dtype=torch.bfloat16)
        block_out = swapped.transpose(0, 1)
    with pytest.raises(error):
        _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)


@_skip_unsupported_gfx
def test_flydsl_attn_res_block_out_plan_follows_alias_and_geometry(monkeypatch):
    num_blocks = 3
    inputs, cap = _cap_inputs(17, _D, num_blocks, num_blocks + 1)
    separate = torch.empty_like(cap)
    wider = torch.randn(17, num_blocks + 3, _D, dtype=torch.bfloat16)
    expected = _expected_closed(inputs, ALL_ON)
    attn_res_kernel._LAUNCH_PLANS.clear()
    seen = _spy_block_alias(monkeypatch)

    for block_out, builds in (
        (cap[:, : num_blocks + 1], [True]),
        (cap[:, : num_blocks + 1], [True]),
        (separate, [True, False]),
        (cap[:, : num_blocks + 1], [True, False]),
        (wider[:, : num_blocks + 1], [True, False, False]),
        (separate, [True, False, False]),
    ):
        out = _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)
        torch.cuda.synchronize()
        assert seen == builds
        assert out[2] is block_out
        assert torch.equal(block_out, expected[2])
        _assert_bf16_close(out[0], expected[0], "block_out plan")
    assert _plan_count() == 3


@_skip_unsupported_gfx
def test_flydsl_attn_res_block_out_plan_is_not_served_to_an_overlap():
    num_blocks = 3
    inputs, cap = _cap_inputs(4, _D, num_blocks, num_blocks + 2)
    block_out = cap[:, : num_blocks + 1]
    for _ in range(2):
        _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)
    plans = _plan_count()

    inputs.block_residual = cap[:, 1 : num_blocks + 1]
    for _ in range(2):
        with pytest.raises(ValueError, match="overlap"):
            _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)
        assert _plan_count() == plans


@_skip_unsupported_gfx
def test_flydsl_attn_res_block_out_plan_is_not_served_to_interleaved_views():
    num_blocks = 3
    cap = torch.randn(4, 2 * (num_blocks + 1), _D, dtype=torch.bfloat16)
    inputs = _make_inputs(4, _D, num_blocks)
    block_out = cap[:, 0::2]
    disjoint = torch.randn(4, 2 * (num_blocks + 1), _D, dtype=torch.bfloat16)
    inputs.block_residual = disjoint[:, 1::2][:, :num_blocks]
    for _ in range(2):
        _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)
    plans = _plan_count()

    inputs.block_residual = cap[:, 1::2][:, :num_blocks]
    for _ in range(2):
        with pytest.raises(ValueError, match="overlap"):
            _call(flydsl_attn_res, inputs, ALL_ON, block_out=block_out)
        assert _plan_count() == plans


@_skip_unsupported_gfx
def test_flydsl_attn_res_default_rows_per_wg_keeps_its_own_plan(monkeypatch):
    rows_per_wg: list[int] = []
    build = attn_res_kernel._build_attn_res

    def spy(spec):
        rows_per_wg.append(spec.rows_per_wg)
        return build(spec)

    attn_res_kernel._LAUNCH_PLANS.clear()
    monkeypatch.setattr(attn_res_kernel, "_build_attn_res", spy)
    for tokens, explicit in ((8, None), (32, None), (8, None), (8, 1), (1, None)):
        extra = {} if explicit is None else {"rows_per_wg": explicit}
        _run_and_check(tokens, 3, MIX, hidden_size=_D_SMALL, **extra)
    assert rows_per_wg == [1, 1, 1, 1]
    assert _plan_count() == 4


def _moved_bytes(tokens: int, hidden_size: int, num_blocks: int, cfg: _Config):
    """HBM bytes moved, excluding the [D] weight vectors."""
    reads = num_blocks + 1 + int(cfg.add) + int(cfg.add2)
    writes = int(cfg.add) + (num_blocks + 1) * int(cfg.close)
    y_bytes = (1 if cfg.fp8 else 2) * hidden_size + 4 * int(cfg.fp8)
    return tokens * (2 * hidden_size * (reads + writes) + y_bytes)


@benchmark()
def benchmark_flydsl_attn_res(
    tokens: int,
    hidden_size: int,
    num_blocks: int,
    config: str,
    rows_per_wg: int = 1,
    cache: str = "warm",
):
    cfg = _BENCH_CONFIGS[config]
    err = _run_and_check(
        tokens, num_blocks, cfg, hidden_size=hidden_size, rows_per_wg=rows_per_wg
    )
    args, kwargs = _gate_args(_make_inputs(tokens, hidden_size, num_blocks), cfg)
    _, us = run_perftest(
        flydsl_attn_res,
        *args,
        **kwargs,
        rows_per_wg=rows_per_wg,
        num_rotate_args=_ROTATE_ARGS[cache],
    )

    k = num_blocks + 1
    # ~4D (sumsq + dot) + 2D mix per candidate; optional output RMS ~4D.
    flops = tokens * hidden_size * (6 * k + 4 * int(cfg.onorm))
    return {
        "gfx": get_gfx(),
        "us": round(us, 3),
        "TFLOPS": round(flops / us / 1e6, 2),
        "TB/s": round(_moved_bytes(tokens, hidden_size, num_blocks, cfg) / us / 1e6, 3),
        "err": err,
    }


def main():
    if get_gfx() not in _SUPPORTED_GFX:
        aiter.logger.warning("flydsl_attn_res unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 4, 17, 320],
        help="Prefill token counts to test and benchmark",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[_MAX_BLOCKS],
        help="Block counts B to benchmark",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        choices=tuple(_BENCH_CONFIGS),
        default=list(_BENCH_CONFIGS),
        help="mix: no add/close/norm; all-on: add_hidden + close_block + output norm",
    )
    parser.add_argument(
        "--cache",
        nargs="+",
        choices=tuple(_ROTATE_ARGS),
        default=["warm"],
        help="Cache states to time: warm reuses one arg set, cold reads HBM",
    )
    args = parser.parse_args()
    for hidden_size, num_blocks in itertools.product(
        (_D, _D_SMALL), range(_MAX_BLOCKS + 1)
    ):
        test_flydsl_attn_res_num_blocks(hidden_size, num_blocks)
    # R=4 is D=1024-only: packing a multi-wave D=7168 row is rejected.
    shapes = ((_D, 1), (_D_SMALL, 1), (_D_SMALL, 4))
    results = [
        benchmark_flydsl_attn_res(
            tokens,
            hidden_size,
            num_blocks,
            config,
            rows_per_wg=rows_per_wg,
            cache=cache,
        )
        for (hidden_size, rows_per_wg), num_blocks, config, tokens, cache in (
            itertools.product(shapes, args.blocks, args.config, args.tokens, args.cache)
        )
    ]
    aiter.logger.info(
        "flydsl_attn_res summary (markdown):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
