# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a8w8 bpreshuffle pad-K tests require a CUDA/HIP device",
)


def test_preshuffle_flat_buffers_stay_below_4gib():
    from aiter.ops.flydsl.gemm_kernels import (
        _check_preshuffle_flat_buffer_capacity,
    )

    # The output is just under 4 GiB; masking the padded M rows keeps this legal.
    _check_preshuffle_flat_buffer_capacity(8161, 262192, 512, 1, 1, 2)

    with pytest.raises(RuntimeError, match="output buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(8191, 262192, 512, 1, 1, 2)
    with pytest.raises(RuntimeError, match="A buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(65536, 16, 65536, 1, 1, 2)
    with pytest.raises(RuntimeError, match="B buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(16, 65536, 65536, 1, 1, 2)


def test_two_wave_vgpr_estimate_uses_128_threads():
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        _estimate_max_wpe,
    )

    # Both tiles hold eight accumulator VGPRs per thread: 32x32 / 128 threads
    # for the 2-wave path and 32x64 / 256 threads for the standard 4-wave path.
    assert _estimate_max_wpe(32, 32, 512, total_vgpr=12) == 1
    assert _estimate_max_wpe(32, 64, 512, total_vgpr=12) == 1


def test_tuner_small_m_candidates_by_arch():
    from aiter.ops.flydsl.gemm_tune import flydsl_gemm_a8w8_bpreshuffle_common as c

    for arch in ("gfx942", "gfx950"):
        pt = (
            [
                k
                for k in c.kernels_list_942.values()
                if c.kernel_fits_shape(k, 4, 2048, 2176, arch)
            ]
            if arch == "gfx942"
            else [
                k
                for k in c.kernels_list_950.values()
                if c.kernel_fits_shape(k, 4, 2048, 2176, arch)
            ]
        )
        bs = [
            k
            for k in getattr(c, f"kernels_list_blockscale_{arch[3:]}").values()
            if c.kernel_fits_shape(k, 4, 2048, 2176, arch)
        ]
        # K=2176 needs tile_k=128; M-overhanging 32-row tiles are admitted.
        assert pt and bs
        assert any(k.tile_m == 32 for k in pt)
        if arch == "gfx942":
            assert not any(k.use_async_copy for k in pt + bs)
    assert not c.async_copy_supported_on_arch("gfx942")
    assert c.async_copy_supported_on_arch("gfx950")


def test_shuffle_weight_pad_k_to_pads_last_dim():
    weight = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)

    shuffled = shuffle_weight(weight, layout=(16, 16), pad_k_to=128)

    assert shuffled.shape == (16, 128)
    assert shuffled.is_shuffled
    assert shuffled.aiter_original_k == 96
    assert shuffled.aiter_padded_k == 128


def test_gemm_a8w8_bpreshuffle_uses_logical_k_for_ck_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "ck", "splitK": 0}

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_uses_logical_k_for_cktile_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "cktile", "splitK": 0}

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_falls_back_to_padded_k_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))
        if k == 128:
            return {"libtype": "cktile", "splitK": 0}
        return None

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_uses_cktile_for_untuned_padded_k(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    def fake_ck(*args, **kwargs):
        raise AssertionError("padded-K untuned fallback should use CKTile")

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_pads_activation_for_flydsl(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "flydsl", "splitK": 0, "kernelName": "fake"}

    def fake_flydsl(XQ, WQ, x_scale, w_scale, Y, config):
        seen["x_shape"] = tuple(XQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_flydsl", fake_flydsl)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_rejects_short_weight_k():
    xq = torch.zeros((2, 128), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="WQ K >= XQ K"):
        gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)


# --- FlyDSL preshuffle GEMM scale-form / blockscale cases (moved from
# flydsl_tests/, which CI does not discover). Called through the FlyDSL op
# wrapper with production shuffle_weight / quantizer layouts.
requires_flydsl_gfx = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() not in ("gfx942", "gfx950"),
    reason="gfx942 or gfx950 required",
)

K = 2176
N = 2048
CONFIGS = [(1, 16, 64, 128, 1), (4, 32, 64, 128, 6)]


def _make(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(1)
    a = torch.randn(m, k, generator=g, device="cuda") * 0.5
    b = torch.randn(n, k, generator=g, device="cuda") * 0.5
    aq, xs = aiter.per_tensor_quant(a, quant_dtype=dtypes.fp8)
    bq, ws = aiter.per_tensor_quant(b, quant_dtype=dtypes.fp8)
    ref = (aq.float() * xs.float()) @ (bq.float() * ws.float()).T
    return aq, xs.float(), shuffle_weight(bq, layout=(16, 16)), ws.float(), ref


def _shape(s, rows):
    return {
        "expanded": lambda: s.reshape(1, 1).expand(rows, 1).contiguous(),
        "scalar1": lambda: s.reshape(1),
        "scalar11": lambda: s.reshape(1, 1),
    }


@requires_flydsl_gfx
@pytest.mark.parametrize(
    "form", ["expanded", "scalar1", "scalar11", "scalar_x", "scalar_w"]
)
@pytest.mark.parametrize("m,tm,tn,tk,sk", CONFIGS)
def test_per_tensor_scale_forms(m, tm, tn, tk, sk, form):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    aq, xs, wq, ws, ref = _make(m, N, K)
    out = torch.full((m, N), float("nan"), device="cuda", dtype=dtypes.bf16)
    x_form = (
        "scalar1" if form == "scalar_x" else "expanded" if form == "scalar_w" else form
    )
    w_form = (
        "scalar1" if form == "scalar_w" else "expanded" if form == "scalar_x" else form
    )
    flydsl_preshuffle_gemm_a8(
        aq,
        wq,
        _shape(xs, m)[x_form](),
        _shape(ws, N)[w_form](),
        out,
        tm,
        tn,
        tk,
        0,
        0,
        0,
        lds_stage=2,
        enable_scheduler=True,
        split_k=sk,
    )
    torch.cuda.synchronize()
    o = out.float()
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    assert torch.isfinite(o).all()
    assert (o - ref).abs().le(tol).all()


@requires_flydsl_gfx
@pytest.mark.parametrize("tile_n", [80, 96, 112])
def test_uncovered_tile_width_raises(tile_n):
    from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm

    with pytest.raises(ValueError, match="not fully covered by the MMA loop"):
        compile_preshuffle_gemm(
            N=256, K=256, tile_m=32, tile_n=tile_n, tile_k=128, in_dtype="fp8"
        )


@requires_flydsl_gfx
def test_wrong_length_scale_raises():
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    m = 4
    aq, _xs, wq, ws, _ = _make(m, N, K)
    out = torch.empty((m, N), device="cuda", dtype=dtypes.bf16)
    bad = torch.ones(m + 1, 1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="x_scale"):
        flydsl_preshuffle_gemm_a8(aq, wq, bad, ws.reshape(1), out, 32, 64, 128, 0, 0, 0)


BLOCKSCALE_CONFIGS = [
    (m, 2048, k, 32, 128, 1) for m in (1, 4, 16) for k in (2176, 7168)
] + [
    (1, 2048, 2176, 32, 128, 6),
    (4, 2048, 7168, 32, 256, 1),
    (1, 2048, 2176, 16, 128, 1),
    (16, 4096, 7168, 32, 128, 1),
    (4, 6144, 2176, 16, 128, 6),
]


@requires_flydsl_gfx
@pytest.mark.parametrize(
    "m,n,k,tm,tk,sk,tn",
    [(*config, 64) for config in BLOCKSCALE_CONFIGS]
    + [(4, 256, 256, 32, 128, 1, tn) for tn in (16, 32, 128)],
)
def test_blockscale(m, n, k, tm, tk, sk, tn):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    g = torch.Generator(device="cuda").manual_seed(123)
    aq = (torch.rand((m, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    bq = (torch.rand((n, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    # Distinct row/column and K-block scales expose transposition and pairing errors.
    kb = torch.arange(k // 128, device="cuda", dtype=torch.float32)
    rows = torch.arange(m, device="cuda", dtype=torch.float32)
    cols = torch.arange(n // 128, device="cuda", dtype=torch.float32)
    xs = (0.5 + (kb[:, None] % 7) * 0.125 + rows[None, :] * 0.0625).contiguous()
    ws = (0.75 + (cols[:, None] % 5) * 0.125 + (kb[None, :] % 11) * 0.0625).contiguous()
    a = aq.float() * xs.T.repeat_interleave(128, dim=1)
    b = bq.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    ref = a @ b.T
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtypes.bf16)
    wq = shuffle_weight(bq, layout=(16, 16))
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    for _ in range(5):
        out.fill_(float("nan"))
        flydsl_preshuffle_gemm_a8(
            aq,
            wq,
            xs,
            ws,
            out,
            tm,
            tn,
            tk,
            0,
            0,
            0,
            lds_stage=2,
            enable_scheduler=True,
            split_k=sk,
            scale_mode="blockscale",
        )
        torch.cuda.synchronize()
        o = out.float()
        nonfinite = (~torch.isfinite(o)).sum().item()
        bad = ((o - ref).abs() > tol).sum().item()
        assert nonfinite == 0, f"nonfinite={nonfinite}"
        assert bad == 0, f"bad={bad}"


@requires_flydsl_gfx
@pytest.mark.parametrize("layout", ["quantizer", "column_major", "k_major"])
def test_blockscale_quantizer_public_dispatch(monkeypatch, layout):
    from aiter.ops import gemm_op_a8w8 as gemm_mod

    m, n, k = 4, 256, 2176
    g = torch.Generator(device="cuda").manual_seed(19)
    a = torch.randn((m, k), generator=g, device="cuda", dtype=dtypes.bf16)
    aq, xs = aiter.per_group_quant_hip(
        a, quant_dtype=dtypes.fp8, group_size=128, transpose_scale=True
    )
    logical_xs = xs.view(k // 128, m).T
    bq = (torch.rand((n, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    ws = torch.linspace(0.5, 1.5, (n // 128) * (k // 128), device="cuda").reshape(
        n // 128, k // 128
    )
    ref = (aq.float() * logical_xs.repeat_interleave(128, dim=1)) @ (
        bq.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    ).T
    if layout == "column_major":
        xs = logical_xs
    elif layout == "k_major":
        xs = xs.view(k // 128, m)
    monkeypatch.setattr(
        gemm_mod,
        "get_CKGEMM_config",
        lambda *args: {
            "libtype": "flydsl",
            "kernelName": "flydsl_bpreshuflle_32x64x128_fp8_bf16_none_0x0x0x2_smbs",
        },
    )
    out = torch.full((m, n), float("nan"), device="cuda", dtype=dtypes.bf16)
    result = aiter.gemm_a8w8_blockscale_bpreshuffle(
        aq, shuffle_weight(bq, layout=(16, 16)), xs, ws, dtypes.bf16, out=out
    )
    torch.cuda.synchronize()
    assert result is out
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().le(tol).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx942",
    reason="gfx942 async-copy rejection",
)
def test_gfx942_async_copy_raises():
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    aq, xs, wq, ws, _ = _make(1, N, K)
    out = torch.empty((1, N), device="cuda", dtype=dtypes.bf16)
    with pytest.raises(ValueError, match="LDS-direct loads unavailable on gfx942"):
        flydsl_preshuffle_gemm_a8(aq, wq, xs, ws, out, 32, 64, 128, use_async_copy=1)


@requires_flydsl_gfx
@pytest.mark.parametrize(
    "m,k,tm,tn,sk",
    [
        (1, 2176, 32, 64, 6),
        (4, 7168, 32, 64, 1),
        (16, 2176, 32, 64, 1),
        (4, 2176, 16, 128, 3),
    ],
)
def test_per_row_scales(m, k, tm, tn, sk):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    g = torch.Generator(device="cuda").manual_seed(1)
    aq = (torch.rand((m, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    bq = (torch.rand((N, k), generator=g, device="cuda") - 0.5).to(dtypes.fp8)
    wq = shuffle_weight(bq, layout=(16, 16))
    xs = torch.linspace(0.5, 1.5, m, device="cuda").reshape(m, 1)
    ws = torch.linspace(0.75, 1.25, N, device="cuda").reshape(N, 1)
    ref = (aq.float() * xs) @ (bq.float() * ws).T
    out = torch.full((m, N), float("nan"), device="cuda", dtype=dtypes.bf16)
    tol = 5e-3 * ref.abs().max() + 1e-2 * ref.abs()
    for _ in range(5):
        out.fill_(float("nan"))
        flydsl_preshuffle_gemm_a8(
            aq, wq, xs, ws, out, tm, tn, 128, 0, 0, 0, lds_stage=2, split_k=sk
        )
        torch.cuda.synchronize()
        o = out.float()
        assert torch.isfinite(o).all()
        assert (o - ref).abs().le(tol).all()


if __name__ == "__main__":
    # CI runs op_tests/*.py with plain python3; hand off to pytest so the
    # test_* functions execute instead of being merely defined.
    import sys

    sys.exit(pytest.main(["-q", __file__]))
