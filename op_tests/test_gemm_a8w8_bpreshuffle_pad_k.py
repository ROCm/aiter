# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

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


@pytest.mark.parametrize(
    "m, n, k, kernel_name, backend",
    [
        # Kimi-K3's 6144x7168 M=8192 row, which padded-M lookup returns for
        # every M > 8192.
        (65536, 6144, 7168, "128x256x128_F8_F8_B16_1x2x4x2_default", "flydsl"),
        (65537, 6144, 7168, "128x256x128_F8_F8_B16_1x2x4x2_default", "ck"),
        # Past M=174592 the bf16 output, padded to 256 rows, reaches 2 GiB and
        # CK writes wrong rows, so the FlyDSL launcher raises instead.
        (174592, 6144, 7168, "128x256x128_F8_F8_B16_1x2x4x2_default", "ck"),
        (174593, 6144, 7168, "128x256x128_F8_F8_B16_1x2x4x2_default", "flydsl"),
        # The 57344x8192 M=8192 row on main: the output is past 2 GiB here.
        (65537, 57344, 8192, "256x128x128_F8_F8_B16_1x2x0x2_default", "flydsl"),
    ],
)
@pytest.mark.parametrize("stride", [1, 2], ids=["dense", "strided"])
def test_flydsl_bpreshuffle_row_falls_back_to_ck_above_m_max(
    monkeypatch, m, n, k, kernel_name, backend, stride
):
    from aiter.ops.flydsl import gemm_kernels

    row = {
        "libtype": "flydsl",
        "splitK": 0,
        "kernelName": f"flydsl_bpreshuflle_{kernel_name}",
    }
    # Meta tensors stand in for activations and outputs of several GiB. A
    # K-strided XQ (a view of every other column) and row-strided scale views
    # must reach CK made contiguous.
    xq = torch.empty((m, k * stride), device="meta", dtype=dtypes.fp8)[:, ::stride]
    wq = torch.empty((n, k), device="meta", dtype=dtypes.fp8)
    x_scale, w_scale = (
        torch.empty((rows, stride), device="meta", dtype=torch.float32)[:, :1]
        for rows in (m, n)
    )
    out = torch.empty((m, n), device="meta", dtype=torch.bfloat16)
    seen = []

    def fake_flydsl(*args, **kwargs):
        seen.append("flydsl")

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK=0):
        assert all(t.is_contiguous() for t in (XQ, WQ, x_scale, w_scale))
        seen.append("ck")
        return Y

    # gfx1250 routes every kernel name to its WMMA backend before the M-limit check.
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_kernels, "flydsl_preshuffle_gemm_a8", fake_flydsl)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    got = gemm_mod.gemm_a8w8_bpreshuffle_flydsl(xq, wq, x_scale, w_scale, out, row)

    assert gemm_kernels.PRESHUFFLE_M_MAX == 65536
    assert seen == [backend]
    assert got is out


@pytest.mark.parametrize(
    "m, n, k, kernel_name, backend",
    [
        # Kimi-K3's 2112x7168 M=1 row; 7168 = 7 * 1024 but 8192 is not.
        (1, 2112, 7168, "16x64x512_F8_F8_B16_1x2x0x2_default_ks7", "flydsl"),
        (1, 2112, 8192, "16x64x512_F8_F8_B16_1x2x0x2_default_ks7", "ck"),
        # K/split_k must also stay a multiple of tile_k.
        (1, 2112, 7040, "16x64x512_F8_F8_B16_1x2x0x2_default_ks2", "ck"),
        (1, 2112, 8192, "16x64x512_F8_F8_B16_1x2x0x2_default_ks4", "flydsl"),
        # Oversized M with an incompatible padded K must not reach CK, whose
        # output is wrong past 2 GiB: the FlyDSL error is raised instead.
        (175000, 6144, 7232, "128x256x128_F8_F8_B16_1x2x4x2_default", "flydsl"),
        (65537, 6144, 7232, "128x256x128_F8_F8_B16_1x2x4x2_default", "ck"),
    ],
)
@pytest.mark.parametrize("stride", [1, 2], ids=["dense", "strided"])
def test_flydsl_bpreshuffle_split_k_falls_back_to_ck_when_k_not_divisible(
    monkeypatch, m, n, k, kernel_name, backend, stride
):
    from aiter.ops.flydsl import gemm_kernels

    row = {"kernelName": f"flydsl_bpreshuflle_{kernel_name}"}
    xq = torch.empty((m, k), device="meta", dtype=dtypes.fp8)
    wq = torch.empty((n, k), device="meta", dtype=dtypes.fp8)
    # Row-strided scale views, e.g. backing[:, :1], must reach CK made contiguous.
    x_scale, w_scale = (
        torch.empty((rows, stride), device="meta", dtype=torch.float32)[:, :1]
        for rows in (m, n)
    )
    out = torch.empty((m, n), device="meta", dtype=torch.bfloat16)
    seen = []

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK=0):
        assert x_scale.is_contiguous() and w_scale.is_contiguous()
        seen.append("ck")
        return Y

    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        gemm_kernels,
        "flydsl_preshuffle_gemm_a8",
        lambda *args, **kwargs: seen.append("flydsl"),
    )
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    got = gemm_mod.gemm_a8w8_bpreshuffle_flydsl(xq, wq, x_scale, w_scale, out, row)

    assert seen == [backend]
    assert got is out


def test_two_wave_vgpr_estimate_uses_128_threads():
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        _estimate_max_wpe,
    )

    # Both tiles hold eight accumulator VGPRs per thread: 32x32 / 128 threads
    # for the 2-wave path and 32x64 / 256 threads for the standard 4-wave path.
    assert _estimate_max_wpe(32, 32, 512, total_vgpr=12) == 1
    assert _estimate_max_wpe(32, 64, 512, total_vgpr=12) == 1


def test_shuffle_weight_pad_k_to_pads_last_dim():
    weight = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)

    shuffled = shuffle_weight(weight, layout=(16, 16), pad_k_to=128)

    assert shuffled.shape == (16, 128)
    assert shuffled.is_shuffled
    assert shuffled.aiter_original_k == 96
    assert shuffled.aiter_padded_k == 128


def test_gemm_a8w8_bpreshuffle_pads_activation_for_ck(monkeypatch):
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
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_pads_activation_for_cktile(monkeypatch):
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
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


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
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_pads_activation_for_untuned_cktile(monkeypatch):
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
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
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
    assert seen["x_shape"] == (2, 128)
    assert seen["w_shape"] == (16, 128)
    assert seen["tail_is_zero"]


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


def _padded_k_backend(m, n, k, w_k):
    """(libtype, split-K?) the dispatcher picks for a K-padded weight, or None."""
    for kk in (k, w_k):
        config = gemm_mod.get_GEMM_config_with_quant_type(
            m, n, kk, dtypes.fp8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
        )
        if config is not None:
            return config["libtype"], int(config["splitK"]) > 0
    return None


def run_torch(xq, x_scale, wq, w_scale, dtype=dtypes.bf16):
    # Reference only: fp32 dequant GEMM over the logical K. Not timed, not in the table.
    out = (xq.to(dtypes.fp32) * x_scale) @ (wq.to(dtypes.fp32) * w_scale).t()
    return out.to(dtype)


@benchmark()
def test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtype):
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtypes.bf16)
    w = torch.randn((n, k), dtype=dtypes.bf16)
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    ref = run_torch(xq, x_scale, wq, w_scale, dtype)
    # As in the model: the weight is preshuffled once with K padded to w_k and
    # the logical-K activation goes straight into the dispatcher.
    wq_padded = shuffle_weight(wq, layout=(16, 16), pad_k_to=w_k)
    assert wq_padded.shape == (n, w_k)

    candidates = {
        "a8w8_bpreshuffle": lambda: gemm_mod.gemm_a8w8_bpreshuffle(
            xq, wq_padded, x_scale, w_scale, dtype=dtype
        ),
    }

    backend = _padded_k_backend(m, n, k, w_k)
    # Split-K sums bf16 partials, so small outputs pick up ~1 ulp of the row
    # magnitude; a misread weight is still off by O(|ref|).
    atol = 1.0 if backend is not None and backend[1] else 1e-2

    # Every backend runs at the padded K: XQ is zero-padded to w_k before launch.
    flops = 2 * m * n * w_k
    nbytes = (m * w_k + n * w_k) * xq.element_size() + m * n * ref.element_size()

    ret = {
        "gfx": get_gfx(),
        "backend": (
            "untuned"
            if backend is None
            else backend[0] + ("-splitk" if backend[1] else "")
        ),
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=atol,
            msg=f"{name}: M={m} N={n} K={k}->{w_k} ",
            catastrophic_check=True,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# Shape-arg sweep fn, not a pytest test: the parametrized wrapper below calls it.
test_gemm_a8w8_bpreshuffle_pad_k.__test__ = False


# (case id, expected (libtype, split-K?), (m, n, k, w_k)) per (gfx, cu_num) tuned
# table: rows reaching each CK/CKTile/FlyDSL path with a K-padded weight, either
# through the logical-K row or the padded-K fallback lookup, plus untuned shapes.
_UNTUNED_CASES = [
    ("untuned", None, (37, 1280, 448, 512)),
    ("untuned-half", None, (37, 1280, 256, 512)),
]
_PADDED_K_CASES = {
    ("gfx950", 256): _UNTUNED_CASES
    + [
        ("ck-logical-k", ("ck", False), (1, 64, 4096, 4224)),
        ("ck-splitk-logical-k", ("ck", True), (4, 64, 4096, 4224)),
        ("cktile-logical-k", ("cktile", False), (1, 1536, 128, 256)),
        ("ck-padded-k-row", ("ck", False), (32, 384, 7040, 7168)),
        ("cktile-padded-k-row", ("cktile", False), (2048, 384, 7040, 7168)),
        ("flydsl-padded-k-row", ("flydsl", False), (512, 384, 7040, 7168)),
    ],
    # The 304-CU gfx942 table has no fp8 rows, so only the untuned fallback runs.
    ("gfx942", 304): _UNTUNED_CASES,
    ("gfx942", 80): _UNTUNED_CASES
    + [
        ("ck-logical-k", ("ck", False), (1, 64, 7168, 7296)),
        ("cktile-logical-k", ("cktile", False), (1, 2048, 2048, 2176)),
        ("ck-padded-k-row", ("ck", False), (1, 64, 7040, 7168)),
        ("cktile-padded-k-row", ("cktile", False), (1, 2048, 1920, 2048)),
    ],
}

# Cards the CK/CKTile bpreshuffle paths are built for. gfx1250 has no CK path,
# and its untuned FlyDSL fallback is not covered by these shapes.
SUPPORTED_GFX = ["gfx942", "gfx950"]
# checkAllclose default: max fraction of mismatching elements per case.
TOL_ERR_RATIO = 0.05


@pytest.mark.parametrize(
    "device,backend,mnk",
    [(dev, *c[1:]) for dev, cases in _PADDED_K_CASES.items() for c in cases],
    ids=[
        f"{gfx}-cu{cu}-{c[0]}"
        for (gfx, cu), cases in _PADDED_K_CASES.items()
        for c in cases
    ],
)
def test_gemm_a8w8_bpreshuffle_padded_k_matches_reference(device, backend, mnk):
    # Dispatch keys the tuned CSV with (get_gfx(), get_cu_num()), and the
    # untuned fallback is arch-specific too, so a row only holds on the full
    # device its table was picked from, not on a partitioned one.
    if (get_gfx(), get_cu_num()) != device:
        pytest.skip(f"rows are picked from the {device[0]} cu_num={device[1]} table")
    m, n, k, w_k = mnk
    assert (
        _padded_k_backend(m, n, k, w_k) == backend
    ), f"M={m} N={n} K={k}->{w_k} no longer selects {backend}; pick another row"
    ret = test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtypes.bf16)
    assert ret["a8w8_bpreshuffle err"] <= TOL_ERR_RATIO, ret


# (case id, tuned FlyDSL kernelName, (m, n, k, w_k)) per (gfx, cu_num) table: Kimi-K3
# FlyDSL rows the launcher rejects, so gemm_a8w8_bpreshuffle_flydsl hands them to CK.
_FLYDSL_CK_FALLBACK_CASES = {
    ("gfx950", 256): [
        # M=65537 > PRESHUFFLE_M_MAX takes the M=8192 row through padded-M lookup.
        (
            "m-above-max",
            "flydsl_bpreshuflle_128x256x128_F8_F8_B16_1x2x4x2_default",
            (65537, 6144, 7168, 7168),
        ),
        # The logical-K _ks7 row on a weight padded to K=8192, which 7 does not divide.
        (
            "padded-k-split-k",
            "flydsl_bpreshuflle_16x64x512_F8_F8_B16_1x2x0x2_default_ks7",
            (1, 2112, 7168, 8192),
        ),
    ],
}


@pytest.mark.parametrize(
    "device,kernel_name,mnk",
    [(dev, *c[1:]) for dev, cases in _FLYDSL_CK_FALLBACK_CASES.items() for c in cases],
    ids=[
        f"{gfx}-cu{cu}-{c[0]}"
        for (gfx, cu), cases in _FLYDSL_CK_FALLBACK_CASES.items()
        for c in cases
    ],
)
def test_gemm_a8w8_bpreshuffle_flydsl_ck_fallback_matches_reference(
    monkeypatch, device, kernel_name, mnk
):
    if (get_gfx(), get_cu_num()) != device:
        pytest.skip(f"rows are picked from the {device[0]} cu_num={device[1]} table")
    from aiter.ops.flydsl import gemm_kernels

    m, n, k, w_k = mnk
    config = gemm_mod.get_GEMM_config_with_quant_type(
        m, n, k, dtypes.fp8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )
    assert (
        config is not None and config["kernelName"] == kernel_name
    ), f"M={m} N={n} K={k} no longer selects {kernel_name}; pick another row"

    seen = []
    ck, flydsl = (
        gemm_mod.gemm_a8w8_bpreshuffle_ck,
        gemm_kernels.flydsl_preshuffle_gemm_a8,
    )

    def record_ck(*args, **kwargs):
        seen.append("ck")
        return ck(*args, **kwargs)

    def record_flydsl(*args, **kwargs):
        seen.append("flydsl")
        return flydsl(*args, **kwargs)

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", record_ck)
    monkeypatch.setattr(gemm_kernels, "flydsl_preshuffle_gemm_a8", record_flydsl)

    ret = test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtypes.bf16)

    assert set(seen) == {"ck"}, seen
    assert ret["a8w8_bpreshuffle err"] <= TOL_ERR_RATIO, ret


@pytest.mark.parametrize(
    "device,kernel_name,mnk",
    [(dev, *c[1:]) for dev, cases in _FLYDSL_CK_FALLBACK_CASES.items() for c in cases],
    ids=[
        f"{gfx}-cu{cu}-{c[0]}"
        for (gfx, cu), cases in _FLYDSL_CK_FALLBACK_CASES.items()
        for c in cases
    ],
)
def test_gemm_a8w8_bpreshuffle_flydsl_ck_fallback_strided_scales(
    device, kernel_name, mnk
):
    """Row-strided scale views through the CK fallback match contiguous scales.

    CK reads the scales through raw pointers, so a view such as backing[:, :1]
    gave wrongly scaled rows unless the fallback makes it contiguous.
    """
    if (get_gfx(), get_cu_num()) != device:
        pytest.skip(f"rows are picked from the {device[0]} cu_num={device[1]} table")
    m, n, k, w_k = mnk
    config = gemm_mod.get_GEMM_config_with_quant_type(
        m, n, k, dtypes.fp8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )
    assert (
        config is not None and config["kernelName"] == kernel_name
    ), f"M={m} N={n} K={k} no longer selects {kernel_name}; pick another row"
    torch.manual_seed(0)
    xq = torch.randn((m, k), dtype=dtypes.bf16).to(dtypes.fp8)
    wq = shuffle_weight(
        torch.randn((n, k), dtype=dtypes.bf16).to(dtypes.fp8),
        layout=(16, 16),
        pad_k_to=w_k,
    )
    x_scale = (torch.rand((m, 2), dtype=torch.float32) + 0.5)[:, :1]
    w_scale = (torch.rand((n, 2), dtype=torch.float32) + 0.5)[:, :1]

    got = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=dtypes.bf16)
    want = gemm_mod.gemm_a8w8_bpreshuffle(
        xq, wq, x_scale.contiguous(), w_scale.contiguous(), dtype=dtypes.bf16
    )
    # Same kernel on the same values: the outputs must match bit for bit.
    err = checkAllclose(
        want.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=0,
        atol=0,
        tol_err_ratio=0,
        msg=f"strided scales: M={m} N={n} K={k}->{w_k} ",
    )
    assert err == 0, err


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a8w8_bpreshuffle padded K unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="""Output data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            c[2]
            for c in _PADDED_K_CASES.get((get_gfx(), get_cu_num()), _UNTUNED_CASES)
            + _FLYDSL_CK_FALLBACK_CASES.get((get_gfx(), get_cu_num()), [])
        ],
        help="""Shape m,n,k,w_k: k is the activation K, w_k the padded weight K.
    e.g.: -s 37,1280,448,512""",
    )
    args = parser.parse_args()

    df = []
    for dtype, (m, n, k, w_k) in itertools.product(args.dtype, args.mnk):
        df.append(test_gemm_a8w8_bpreshuffle_pad_k(m, n, k, w_k, dtype))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "gemm_a8w8_bpreshuffle padded K summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    bad = df[df["a8w8_bpreshuffle err"] > TOL_ERR_RATIO]
    assert bad.empty, f"padded-K mismatch above {TOL_ERR_RATIO}:\n{bad}"


if __name__ == "__main__":
    main()
    # CI runs this file with python3, which skips its pytest functions; run the
    # FlyDSL-to-CK fallback checks here so CI exercises them.
    sys.exit(pytest.main([__file__, "-q", "-k", "falls_back or ck_fallback"]))
