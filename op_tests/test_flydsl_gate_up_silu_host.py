# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the fused gate_up + SwiGLU + MXFP8 quant GEMM (gfx1250).

* ``gate_up_silu_supported(n2, k)`` is True only on the measured part and
  gate_up shape (no row count involved); ``gate_up_silu_kernel_for`` serves
  each M bucket of its table there and returns None elsewhere;
* every table tile pins its K loop with ``_pl`` and passes
  ``gate_up_silu_kernel_ok``;
* ``swiglu_limit`` follows aiter ``silu_and_mul`` on the f32 it receives
  (0, -0.0, f32 inf = no clamp; negative or NaN rejected), and the launch
  rejects bad limits, operand dtypes, strides, shapes, buffers, devices,
  misaligned bases and kernel names before it touches a GPU;
* with no kernel name the launch picks the tile from the runtime row count,
  and the functional registered op does that inside a graph compiled once
  for every row count;
* a CPU model of the clamp step documents the bf16-rounded gate clamp. Its
  epilogue model performs the same operations as its silu_and_mul model, so
  their agreement holds by construction: it shows what the f32 clamp got
  wrong, it does not test the kernel.

Runs on CPU; no case needs a GPU. The numerical check against GEMM +
silu_and_mul + quantize_fp8 is in op_tests/test_flydsl_batched_gemm.py
(gfx1250).
"""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from aiter.ops import batched_gemm_op_a8w8 as bop
from aiter.ops.flydsl import batched_gemm_a8w8_gfx1250 as bmm

N2, K = 4608, 5120
INTER = N2 // 2
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


@pytest.fixture
def measured(monkeypatch):
    """Pretend the running GPU is the measured gfx1250 part."""

    @torch.compiler.assume_constant_result
    def _yes() -> bool:
        return True

    monkeypatch.setattr(bmm, "_gate_up_silu_device_measured", _yes)


def _bucket(m):
    return next(
        (name for bound, name in bmm._GATE_UP_SILU_TILES if m <= bound),
        bmm._GATE_UP_SILU_TILES[-1][1],
    )


def test_picker_buckets(measured):
    bounds = [bound for bound, _ in bmm._GATE_UP_SILU_TILES]
    assert bounds == sorted(bounds) and bounds[-1] == bmm.GATE_UP_SILU_MAX_ROWS
    lo = 1
    for bound, name in bmm._GATE_UP_SILU_TILES:
        for m in {lo, (lo + bound) // 2, bound}:
            assert bmm.gate_up_silu_kernel_for(m, N2, K) == name
            assert bmm._gate_up_silu_tile(m) == name
        lo = bound + 1
    for m in (0, -1, bounds[-1] + 1, 16384):
        assert bmm.gate_up_silu_kernel_for(m, N2, K) is None
    # The launch's own pick past the table: the largest bucket's tile.
    assert bmm._gate_up_silu_tile(16384) == bmm._GATE_UP_SILU_TILES[-1][1]
    assert bmm.gate_up_silu_supported(N2, K)


@pytest.mark.parametrize(
    "n2,k",
    [(512, 3072), (3072, 7168), (N2, 7168), (2 * N2, K), (N2, 2 * K), (1152, K)],
)
def test_picker_unmeasured_shape(measured, n2, k):
    # Shapes the tiles may well divide but were never timed at.
    assert not bmm.gate_up_silu_supported(n2, k)
    for m in (1, 384, 1345, 4096):
        assert bmm.gate_up_silu_kernel_for(m, n2, k) is None


@pytest.mark.parametrize(
    "gfx,cu_num,expect",
    [
        ("gfx1250", 256, True),
        ("gfx1250", 192, False),
        ("gfx1250", 96, False),
        ("gfx950", 256, False),
        ("gfx942", 304, False),
    ],
)
def test_picker_device_gate(monkeypatch, gfx, cu_num, expect):
    monkeypatch.setattr(bmm, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(bmm, "get_cu_num", lambda: cu_num)
    assert bmm._gate_up_silu_device_measured() is expect
    assert bmm.gate_up_silu_supported(N2, K) is expect
    got = bmm.gate_up_silu_kernel_for(1345, N2, K)
    assert got == (_bucket(1345) if expect else None)


@pytest.mark.parametrize("error", [RuntimeError, KeyError, AssertionError, ValueError])
def test_picker_undetermined_device(monkeypatch, error):
    # No GPU, an unknown arch, mixed CU counts, a bad CU_NUM.
    def _fail():
        raise error("cannot tell")

    monkeypatch.setattr(bmm, "get_cu_num", _fail)
    monkeypatch.setattr(bmm, "get_gfx_runtime", lambda: "gfx1250")
    assert not bmm.gate_up_silu_supported(N2, K)
    assert bmm.gate_up_silu_kernel_for(7, N2, K) is None
    monkeypatch.setattr(bmm, "get_gfx_runtime", _fail)
    assert bmm.gate_up_silu_kernel_for(7, N2, K) is None


def test_table_tiles_pin_k_loop():
    for _, name in bmm._GATE_UP_SILU_TILES:
        assert name.endswith("_pl"), name
        cfg = bmm.parse_bmm_mx32_kernel_name(name)
        assert cfg is not None and cfg["pipelined"] == 1, name
        assert bmm.gate_up_silu_kernel_ok(name), name
        assert bmm._gate_up_silu_tiles_shape(cfg, N2, K), name


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "not_a_kernel",
        # split-K / cluster
        "flydsl_bmm_mxfp8_32_wmma_t64x128x256_mw2_nw2_nb4_sk2_cm1_cn1_pl",
        "flydsl_bmm_mxfp8_32_wmma_t64x128x256_mw2_nw2_nb4_sk1_cm2_cn1_pl",
        # tile_n not a multiple of 64
        "flydsl_bmm_mxfp8_32_wmma_t64x96x256_mw2_nw1_nb4_sk1_cm1_cn1_pl",
        # 32 columns per wave
        "flydsl_bmm_mxfp8_32_wmma_t64x128x256_mw1_nw4_nb4_sk1_cm1_cn1_pl",
        # 128 columns per wave
        "flydsl_bmm_mxfp8_32_wmma_t64x256x256_mw2_nw2_nb3_sk1_cm1_cn1_pl",
        # tile_n 192 (64 per wave, but not a tile the op test runs)
        "flydsl_bmm_mxfp8_32_wmma_t64x192x256_mw2_nw3_nb3_sk1_cm1_cn1_pl",
        # the mxfp8_128 family
        "flydsl_bmm_mxfp8_128_wmma_t64x128x256_mw2_nw2_nb4_sk1_cm1_cn1",
    ],
)
def test_kernel_ok_rejects(name):
    assert not bmm.gate_up_silu_kernel_ok(name)


@pytest.mark.parametrize(
    "limit,want",
    [
        (10.0, 10.0),
        (7, 7.0),
        (0.0, 0.0),
        (1e-3, 1e-3),
        (math.inf, 0.0),
        # positive doubles whose f32 value is 0: silu_and_mul gets 0.0f
        (1e-46, 0.0),
        (5e-324, 0.0),
        # -0.0f: silu_and_mul runs it as no clamp
        (-1e-46, 0.0),
        # inf as f32 (no overflow warning)
        (1e39, 0.0),
    ],
)
def test_limit_accepts(limit, want):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bmm.gate_up_silu_limit(limit) == want


@pytest.mark.parametrize("limit", [-1.0, -1e-6, -math.inf, math.nan])
def test_limit_rejects(limit):
    with pytest.raises(ValueError, match="swiglu_limit"):
        bmm.gate_up_silu_limit(limit)


# CPU model of the clamp + SiLU*up + group-32 e8m0 step, to check the clamp
# semantics without a gfx1250: a gate g past the limit and every bf16 up u in
# (0, L] (one element per 32-column group, the rest zero), e8m0 of the bf16
# activation.


def _bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def _e8m0(amax):
    # ceil pow2 of amax / 448 (amax floored at 1e-10), as an e8m0 exponent.
    q = torch.clamp(amax, min=1e-10) / 448.0
    mant, exp = torch.frexp(q)  # q = mant * 2**exp, mant in [0.5, 1)
    return exp - 1 + (mant > 0.5).to(exp.dtype) + 127


def _silu_and_mul_ref(g, u, limit):
    # csrc/kernels/activation_kernels.cu, silu with a limit: the gate is
    # clamped in f32 and cast back to bf16, the up clamped in f32 (fmed3).
    lim = torch.tensor(limit, dtype=torch.float32)
    gx = _bf16(torch.minimum(g, lim))
    uy = torch.clamp(u, -lim, lim)
    return _bf16(gx / (1 + torch.exp(-gx)) * uy)


def _epilogue_model(g, u, limit, round_gate):
    # The fused epilogue's `_act`: min / fmed3 clamp, SiLU, times up. The SiLU
    # is written as in the reference, so only the clamp differs (the exp / rcp
    # rounding difference is the op test's one-step allowance).
    lim = torch.tensor(limit, dtype=torch.float32)
    g = torch.minimum(g, lim)
    if round_gate:
        g = _bf16(g)
    u = torch.maximum(torch.minimum(u, lim), -lim)
    return _bf16(g / (1 + torch.exp(-g)) * u)


def _saturated_rows(limit):
    # every positive bf16 up to the limit as u, a gate well past the limit
    top = int(torch.tensor(limit).to(torch.bfloat16).view(torch.int16)) + 1
    u = torch.arange(1, top, dtype=torch.int16).view(torch.bfloat16).float()
    g = torch.full_like(u, 64.0)
    return g, u


@pytest.mark.parametrize("limit", [7.49, 5.02, 5.14])
def test_clamped_gate_rounding_matches_silu_and_mul(limit):
    """At limits bf16 cannot hold, an f32-clamped gate moves some groups' e8m0
    scale against silu_and_mul (the old epilogue). The bf16-rounded model
    performs the same operations as the silu_and_mul model, so the equality
    below holds by construction and documents the semantics; the kernel itself
    is checked by the gfx1250 op test."""
    assert float(torch.tensor(limit).to(torch.bfloat16)) != limit
    g, u = _saturated_rows(limit)
    ref = _silu_and_mul_ref(g, u, limit)
    new = _epilogue_model(g, u, limit, round_gate=True)
    old = _epilogue_model(g, u, limit, round_gate=False)
    assert torch.equal(new, ref)
    assert torch.equal(_e8m0(new.abs()), _e8m0(ref.abs()))
    # The f32 clamp the epilogue had reproduces the reported scale flips.
    assert bool((_e8m0(old.abs()) != _e8m0(ref.abs())).any())


@pytest.mark.parametrize("limit", [10.0, 7.5, 0.5])
def test_bf16_limits_unaffected_by_gate_rounding(limit):
    g, u = _saturated_rows(limit)
    old = _epilogue_model(g, u, limit, round_gate=False)
    assert torch.equal(old, _epilogue_model(g, u, limit, round_gate=True))


def _args(
    m=8,
    n2=N2,
    k=K,
    out_shape=None,
    scale_shape=None,
    device="meta",
    x_dtype=FP8,
    out_dtype=FP8,
):
    return {
        "XQ": torch.empty(m, k, dtype=x_dtype, device=device),
        "WQ": torch.empty(n2, k, dtype=FP8, device=device),
        "x_scale": torch.empty(m, k // 32, dtype=torch.uint8, device=device),
        "w_scale": torch.empty(n2 // 32, k // 32, dtype=torch.uint8, device=device),
        "Out": torch.empty(out_shape or (m, n2 // 2), dtype=out_dtype, device=device),
        "y_scale_out": torch.empty(
            scale_shape or (m, n2 // 64), dtype=E8M0, device=device
        ),
    }


@pytest.fixture
def other_arch(monkeypatch):
    monkeypatch.setattr(bmm, "get_gfx_runtime", lambda: "gfx950")


@pytest.mark.parametrize("limit", [-1.0, math.nan])
def test_run_rejects_limit(other_arch, limit):
    with pytest.raises(ValueError, match="swiglu_limit"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**_args(), swiglu_limit=limit)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"k": K + 32}, "bad shapes"),  # K not a multiple of 128
        ({"n2": N2 + 64}, "bad shapes"),  # 2*I not a multiple of 128
        ({"out_shape": (8, INTER + 32)}, "Out must be"),
        ({"scale_shape": (8, INTER // 32 + 1)}, "Out must be"),
        ({"x_dtype": torch.bfloat16}, "XQ/WQ must be e4m3"),
        ({"out_dtype": torch.int8}, "Out must be"),
        ({"out_dtype": torch.bfloat16}, "Out must be"),
    ],
)
def test_run_rejects_operands(other_arch, kw, match):
    with pytest.raises(RuntimeError, match=match):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**_args(**kw), swiglu_limit=10.0)


def test_run_rejects_layouts(other_arch):
    args = _args()
    wide = torch.empty(8, 2 * K, dtype=FP8, device="meta")
    with pytest.raises(RuntimeError, match="XQ must be row-contiguous"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **{**args, "XQ": wide[:, :K]}, swiglu_limit=10.0
        )
    wq_t = torch.empty(K, N2, dtype=FP8, device="meta").t()
    with pytest.raises(RuntimeError, match="WQ must be contiguous"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**{**args, "WQ": wq_t}, swiglu_limit=10.0)
    with pytest.raises(RuntimeError, match="must be 2-D"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **{**args, "XQ": args["XQ"].view(8, 1, K)}, swiglu_limit=10.0
        )


def test_run_rejects_kernel(other_arch):
    for bad in ("", "not_a_kernel"):
        with pytest.raises(ValueError, match="unsupported kernel"):
            bmm.run_gate_up_silu_mxfp8_32_gfx1250(
                **_args(), swiglu_limit=10.0, kernel_name=bad
            )
    # A valid fused tile that does not divide I.
    with pytest.raises(RuntimeError, match="does not tile"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **_args(n2=4480), swiglu_limit=10.0, kernel_name=_bucket(4096)
        )


def test_run_rejects_device_and_alignment(other_arch):
    args = _args(device="cpu")
    with pytest.raises(RuntimeError, match="must be on"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **{**args, "Out": torch.empty(8, INTER, dtype=FP8, device="meta")},
            swiglu_limit=10.0,
        )
    shifted = torch.empty(8 * INTER + 1, dtype=FP8)[1:].view(8, INTER)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **{**args, "Out": shifted}, swiglu_limit=10.0
        )


def test_run_rejects_other_arch(other_arch):
    with pytest.raises(RuntimeError, match="requires gfx1250"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**_args(), swiglu_limit=10.0)


@pytest.fixture
def launches(monkeypatch):
    """Run the launch path on CPU up to the kernel: record the tile of each
    launch instead of compiling it."""
    seen = []

    def run_compiled(exe, *args, specialization_key):
        seen.append(specialization_key)

    monkeypatch.setattr(bmm, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(bmm, "_lazy_import", lambda: None)
    monkeypatch.setattr(bmm, "_check_e8m0", lambda *a, **k: None)
    monkeypatch.setattr(bmm, "_ptr_arg", lambda t: None)
    monkeypatch.setattr(bmm, "_fx", SimpleNamespace(Stream=lambda s: None))
    monkeypatch.setattr(bmm, "_run_compiled", run_compiled)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    return seen


def _tile_of(spec):
    _, _, tm, tn, tk, mw, nw, nb, pl, _ = spec
    return (tm, tn, tk, mw, nw, nb, pl)


def _tile_of_name(name):
    c = bmm.parse_bmm_mx32_kernel_name(name)
    return tuple(
        c[f]
        for f in (
            "tile_m",
            "tile_n",
            "tile_k",
            "m_warp",
            "n_warp",
            "num_buffers",
            "pipelined",
        )
    )


def test_run_picks_tile_from_runtime_rows(launches):
    for m in (1, 500, 1345, 4096, 5000):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**_args(m=m), swiglu_limit=10.0)
        assert _tile_of(launches[-1]) == _tile_of_name(_bucket(m)), m
    name = bmm._GATE_UP_SILU_TILES[0][1]
    bmm.run_gate_up_silu_mxfp8_32_gfx1250(
        **_args(m=4096), swiglu_limit=10.0, kernel_name=name
    )
    assert _tile_of(launches[-1]) == _tile_of_name(name)


def test_registered_op_rejects_limit(other_arch):
    args = _args(device="cpu")
    with pytest.raises((ValueError, RuntimeError), match="swiglu_limit"):
        torch.ops.aiter.gate_up_silu_mxfp8_32(
            args["XQ"], args["WQ"], args["x_scale"], args["w_scale"], -1.0
        )


def test_traced_op_picks_per_runtime_rows(launches):
    """The functional op inside a fullgraph dynamic-shape compile: one graph
    serves every row count, and each call still launches its own row count's
    tile (the pick happens inside the op, not in the traced code)."""

    def fwd(xq, wq, xs, ws):
        act, act_scale = bop.gate_up_silu_mxfp8_32(xq, wq, xs, ws, 10.0)
        return act.view(torch.uint8).float().sum() + act_scale.view(torch.uint8).sum()

    torch._dynamo.reset()
    frames = []

    def backend(gm, example_inputs):
        frames.append(gm)
        return gm.forward

    compiled = torch.compile(fwd, fullgraph=True, dynamic=True, backend=backend)
    for m in (5000, 7, 1345, 2048):
        a = _args(m=m, device="cpu")
        compiled(a["XQ"], a["WQ"], a["x_scale"], a["w_scale"])
        assert _tile_of(launches[-1]) == _tile_of_name(_bucket(m)), m
    assert len(frames) == 1
    torch._dynamo.reset()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
