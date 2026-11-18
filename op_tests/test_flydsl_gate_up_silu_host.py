# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the fused gate_up + SwiGLU + MXFP8 quant GEMM (gfx1250).

* ``gate_up_silu_kernel_for`` serves each M bucket of its table only on the
  measured part and shapes, returns None elsewhere, and traces under
  ``torch.compile(fullgraph=True)`` with symbolic sizes;
* every table tile pins its K loop with ``_pl`` and passes
  ``gate_up_silu_kernel_ok``;
* ``swiglu_limit`` follows aiter ``silu_and_mul`` (0 = no clamp, negative or
  NaN rejected), and the launch rejects bad limits, shapes, buffers and kernel
  names before it touches a GPU;
* the registered op declares ``Out`` / ``y_scale_out`` as written.

Runs on CPU; no case needs a GPU. The numerical check against GEMM +
silu_and_mul + quantize_fp8 is in op_tests/test_flydsl_batched_gemm.py
(gfx1250).
"""

import math

import pytest
import torch

from aiter.ops import batched_gemm_op_a8w8 as bop
from aiter.ops.flydsl import batched_gemm_a8w8_gfx1250 as bmm

N2, K = 4608, 5120
INTER = N2 // 2


@pytest.fixture
def measured(monkeypatch):
    """Pretend the running GPU is the measured gfx1250 part."""

    @torch.compiler.assume_constant_result
    def _yes() -> bool:
        return True

    monkeypatch.setattr(bmm, "_gate_up_silu_device_measured", _yes)


def _bucket(m):
    return next(name for bound, name in bmm._GATE_UP_SILU_TILES if m <= bound)


def test_picker_buckets(measured):
    bounds = [bound for bound, _ in bmm._GATE_UP_SILU_TILES]
    assert bounds == sorted(bounds)
    lo = 1
    for bound, name in bmm._GATE_UP_SILU_TILES:
        for m in {lo, (lo + bound) // 2, bound}:
            assert bmm.gate_up_silu_kernel_for(m, N2, K) == name
        lo = bound + 1
    for m in (0, -1, bounds[-1] + 1, 16384):
        assert bmm.gate_up_silu_kernel_for(m, N2, K) is None


@pytest.mark.parametrize(
    "n2,k",
    [(512, 3072), (3072, 7168), (N2, 7168), (2 * N2, K), (N2, 2 * K), (4096, 4096)],
)
def test_picker_unmeasured_shape(measured, n2, k):
    # Shapes the tiles may well divide but were never timed at.
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
    got = bmm.gate_up_silu_kernel_for(1345, N2, K)
    assert got == (_bucket(1345) if expect else None)


def test_picker_no_gpu(monkeypatch):
    def _no_gpu():
        raise RuntimeError("no GPU")

    monkeypatch.setattr(bmm, "get_gfx_runtime", _no_gpu)
    assert bmm.gate_up_silu_kernel_for(7, N2, K) is None


def test_table_tiles_pin_k_loop():
    names = [name for _, name in bmm._GATE_UP_SILU_TILES]
    names.append(bmm.GATE_UP_SILU_DEFAULT_KERNEL)
    for name in names:
        assert name.endswith("_pl"), name
        cfg = bmm.parse_bmm_mx32_kernel_name(name)
        assert cfg is not None and cfg["pipelined"] == 1, name
        assert bmm.gate_up_silu_kernel_ok(name), name
        assert bmm._gate_up_silu_tiles_shape(cfg, N2, K), name
    assert bmm.GATE_UP_SILU_DEFAULT_KERNEL in names[:-1]


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
        # the mxfp8_128 family
        "flydsl_bmm_mxfp8_128_wmma_t64x128x256_mw2_nw2_nb4_sk1_cm1_cn1",
    ],
)
def test_kernel_ok_rejects(name):
    assert not bmm.gate_up_silu_kernel_ok(name)


@pytest.mark.parametrize(
    "limit,want",
    [(10.0, 10.0), (7, 7.0), (0.0, 0.0), (1e-3, 1e-3), (math.inf, 0.0)],
)
def test_limit_accepts(limit, want):
    assert bmm.gate_up_silu_limit(limit) == want


@pytest.mark.parametrize("limit", [-1.0, -1e-6, -math.inf, math.nan])
def test_limit_rejects(limit):
    with pytest.raises(ValueError, match="swiglu_limit"):
        bmm.gate_up_silu_limit(limit)


def _args(m=8, n2=N2, k=K, out_shape=None, scale_shape=None, device="meta"):
    fp8 = torch.float8_e4m3fn
    return {
        "XQ": torch.empty(m, k, dtype=fp8, device=device),
        "WQ": torch.empty(n2, k, dtype=fp8, device=device),
        "x_scale": torch.empty(m, k // 32, dtype=torch.uint8, device=device),
        "w_scale": torch.empty(n2 // 32, k // 32, dtype=torch.uint8, device=device),
        "Out": torch.empty(out_shape or (m, n2 // 2), dtype=fp8, device=device),
        "y_scale_out": torch.empty(
            scale_shape or (m, n2 // 64), dtype=torch.uint8, device=device
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
    ],
)
def test_run_rejects_shapes(other_arch, kw, match):
    with pytest.raises(RuntimeError, match=match):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**_args(**kw), swiglu_limit=10.0)


def test_run_rejects_kernel(other_arch):
    with pytest.raises(ValueError, match="unsupported kernel"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **_args(), swiglu_limit=10.0, kernel_name="not_a_kernel"
        )
    # A valid fused tile that does not divide I.
    with pytest.raises(RuntimeError, match="does not tile"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(
            **_args(n2=4480),
            swiglu_limit=10.0,
            kernel_name=_bucket(4096),
        )


def test_run_rejects_other_arch(other_arch):
    with pytest.raises(RuntimeError, match="requires gfx1250"):
        bmm.run_gate_up_silu_mxfp8_32_gfx1250(**_args(), swiglu_limit=10.0)


def test_registered_op_rejects_limit(other_arch):
    with pytest.raises((ValueError, RuntimeError), match="swiglu_limit"):
        torch.ops.aiter.gate_up_silu_mxfp8_32(*_args(device="cpu").values(), -1.0, None)


def test_traced_picker_and_op(measured, monkeypatch):
    """The picker and the registered op inside a fullgraph compile with
    symbolic sizes; the launch (stubbed here) runs once per call and its
    writes to Out / y_scale_out reach the caller's tensors."""
    calls = []

    def fake_run(XQ, WQ, x_scale, w_scale, Out, y_scale_out, limit, kernel_name):
        calls.append((XQ.shape[0], limit, kernel_name))
        Out.view(torch.uint8).fill_(1)
        y_scale_out.fill_(2)
        return Out

    monkeypatch.setattr(bmm, "run_gate_up_silu_mxfp8_32_gfx1250", fake_run)

    def fwd(xq, wq, xs, ws, out, ys):
        name = bmm.gate_up_silu_kernel_for(xq.shape[0], wq.shape[0], wq.shape[1])
        if name is None:
            return False
        bop.gate_up_silu_mxfp8_32(xq, wq, xs, ws, out, ys, 10.0, name)
        return True

    compiled = torch.compile(fwd, fullgraph=True, dynamic=True, backend="aot_eager")
    for m in (7, 1345, 5000):
        args = _args(m=m, device="cpu")
        took = compiled(*args.values())
        assert took == (m <= 4096)
        if took:
            assert calls[-1] == (m, 10.0, _bucket(m))
            assert bool((args["Out"].view(torch.uint8) == 1).all())
            assert bool((args["y_scale_out"] == 2).all())
        else:
            assert calls[-1][0] != m
    assert len(calls) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
