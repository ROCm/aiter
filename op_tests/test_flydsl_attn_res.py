# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test and benchmark for FlyDSL AttnRes on the ``attn_res_gate`` contract.

Usage:
    pytest -q op_tests/test_flydsl_attn_res.py
"""

import dataclasses
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_attn_res
from aiter.ops.flydsl.kernels import attn_res as attn_res_kernel
from aiter.ops.torch_ref.attn_res import attn_res as attn_res_reference

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


def _call(fn, inputs, cfg: _Config, **extra):
    return fn(
        inputs.prefix,
        inputs.block_residual,
        inputs.score_weight,
        _EPS,
        inputs.add_hidden if cfg.add else None,
        inputs.add_hidden2 if cfg.add2 else None,
        output_rms_weight=inputs.output_rms_weight if cfg.onorm else None,
        output_rms_eps=_OUT_EPS,
        scale=cfg.scale,
        close_block=cfg.close,
        out_quant_dtype=dtypes.fp8 if cfg.fp8 else None,
        **extra,
    )


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
