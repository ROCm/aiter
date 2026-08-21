# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Focused CPU regressions for the public exact-kid OPUS interfaces."""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance, kernels_list


def test_public_surface_and_registry_keep_global_kid_families():
    opus = importlib.import_module("aiter.ops.opus")

    assert opus.__all__ == ["opus_gemm", "opus_bmm"]
    assert tuple(inspect.signature(opus.opus_gemm).parameters) == (
        "XQ",
        "WQ",
        "Y",
        "kid",
        "layout",
        "x_scale",
        "w_scale",
        "bias",
        "split_k",
        "workspace",
    )
    assert kernels_list[200].kernel_tag.startswith("a16w16")
    assert kernels_list[2].kernel_tag == "a8w8"
    assert kernels_list[1].kernel_tag == "a8w8_scale"
    assert kernels_list[8000].kernel_tag == "a8w8_mxscale_bmm_flatmm_splitk"
    assert kernels_list[11000].kernel_tag == (
        "a8w8_blockscale_bpreshuffle_singlebuf"
    )
    assert kernels_list[20000].arch_prefix == "gfx1250"
    assert get_kernel_instance("gfx942", "a16w16", 10200) is kernels_list[10200]
    assert get_kernel_instance("gfx950", "a16w16", 10200) is None


@pytest.mark.parametrize(
    ("arch", "M", "N", "K", "expected"),
    [
        ("gfx950", 256, 256, 512, 1300),
        ("gfx942", 256, 1024, 7168, 10210),
        ("gfx1250", 32, 128, 4096, 20007),
    ],
)
def test_a16_caller_policy_selects_one_compiled_kid_per_arch(
    arch, M, N, K, expected
):
    policy = importlib.import_module("aiter.ops.opus.policy")
    assert policy.select_a16w16_heuristic_kid(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=1,
        has_bias=False,
        output_dtype=torch.bfloat16,
    ) == expected


@pytest.mark.parametrize(
    ("arch", "M", "N", "K", "cu_num", "kid", "dtype", "shape"),
    [
        ("gfx950", 64, 64, 512, 256, 200, torch.float32, (2, 1, 64, 64)),
        (
            "gfx942",
            128,
            384,
            4096,
            304,
            10210,
            torch.bfloat16,
            (2, 1, 128, 384),
        ),
        ("gfx1250", 16, 32, 512, 256, 20000, torch.bfloat16, (2, 16, 32)),
    ],
)
def test_a16_launch_plan_keeps_arch_workspace_contract(
    arch, M, N, K, cu_num, kid, dtype, shape
):
    plans = importlib.import_module("aiter.ops.opus.launch_plan")
    plan = plans._get_cached_a16w16_launch_plan(
        arch,
        M,
        N,
        K,
        1,
        cu_num,
        False,
        torch.bfloat16,
        torch.bfloat16,
        kid,
        2,
    )
    assert (plan.registry_arch, plan.resolved_kid, plan.abi_split_k) == (
        arch,
        kid,
        2,
    )
    assert plan.workspace_spec is not None
    assert (plan.workspace_spec.dtype, plan.workspace_spec.shape) == (dtype, shape)


@pytest.mark.parametrize(
    ("operation", "rank", "launcher"),
    [
        ("opus_gemm", 2, "_launch_a16w16_gemm"),
        ("opus_bmm", 3, "_launch_a16w16_bmm"),
    ],
)
def test_public_a16_routes_exact_kid_without_caller_policy(
    monkeypatch, operation, rank, launcher
):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    calls = []

    def fake(XQ, WQ, Y, bias, **kwargs):
        calls.append((XQ, WQ, Y, bias, kwargs))
        return Y

    monkeypatch.setattr(a16, launcher, fake)
    shape_x = (64, 512) if rank == 2 else (1, 64, 512)
    shape_y = (64, 64) if rank == 2 else (1, 64, 64)
    XQ = torch.empty(shape_x, dtype=torch.bfloat16)
    WQ = torch.empty(shape_x, dtype=torch.bfloat16)
    Y = torch.empty(shape_y, dtype=torch.bfloat16)

    assert getattr(opus, operation)(XQ, WQ, Y, kid=200, split_k=2) is Y
    assert len(calls) == 1
    assert calls[0][:4] == (XQ, WQ, Y, None)
    assert calls[0][4]["kid"] == 200
    assert calls[0][4]["split_k"] == 2
    assert calls[0][4]["route_arch"] == "gfx950"
    assert calls[0][4]["instance"] is kernels_list[200]


def test_a16_caller_workspace_reaches_checked_backend_unchanged(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    calls = []

    monkeypatch.setattr(
        a16,
        "_device_arch_and_cu",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("caller-workspace fast path queried the device")
        ),
    )
    monkeypatch.setattr(
        a16,
        "_launch_a16w16_backend",
        lambda *args: calls.append(args),
    )
    a16._get_cached_a16w16_launch_plan.cache_clear()
    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty_like(XQ)
    Y = torch.empty((1, 64, 64), dtype=torch.float32)
    workspace = torch.empty((2, 1, 64, 64), dtype=torch.float32)

    assert opus.opus_bmm(
        XQ, WQ, Y, kid=200, split_k=2, workspace=workspace
    ) is Y
    assert calls == [(XQ, WQ, Y, None, workspace, 200, 2)]


@pytest.mark.parametrize(
    ("kid", "layout", "dtype", "launcher", "scaled"),
    [
        (2, "plain", torch.float32, "_launch_a8w8_gemm", False),
        (1, "plain", torch.float32, "_launch_a8w8_blockscale_gemm", True),
        (
            11000,
            "bpreshuffle",
            torch.bfloat16,
            "_launch_a8w8_blockscale_bpreshuffle_gemm",
            True,
        ),
    ],
)
def test_public_a8_gemm_routes_three_production_families(
    monkeypatch, kid, layout, dtype, launcher, scaled
):
    opus = importlib.import_module("aiter.ops.opus")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return next(arg for arg in args if arg is Y)

    monkeypatch.setattr(a8, launcher, fake)
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((128, 256), dtype=fp8)
    WQ = torch.empty_like(XQ)
    Y = torch.empty((128, 128), dtype=dtype)
    kwargs = {"kid": kid, "layout": layout}
    if scaled:
        kwargs.update(
            x_scale=torch.empty((128, 2), dtype=torch.float32),
            w_scale=torch.empty((1, 2), dtype=torch.float32),
        )

    assert opus.opus_gemm(XQ, WQ, Y, **kwargs) is Y
    assert len(calls) == 1
    assert calls[0][1]["kid"] == kid
    assert calls[0][1]["instance"] is kernels_list[kid]


def test_public_mxscale_bmm_routes_exact_kid_and_workspace_plan(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    plans = importlib.import_module("aiter.ops.opus.launch_plan")
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(a8, "_launch_a8w8_mxscale_bmm", fake)
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((2, 17, 2048), dtype=fp8)
    WQ = torch.empty((2, 128, 2048), dtype=fp8)
    Y = torch.empty((2, 17, 128), dtype=torch.bfloat16)
    x_scale = torch.empty((2, 17, 16), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 16), dtype=torch.uint8)

    assert opus.opus_bmm(
        XQ,
        WQ,
        Y,
        kid=8000,
        layout="mxscale_bmm",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=2,
    ) is Y
    assert calls[0][1]["kid"] == 8000
    assert calls[0][1]["instance"] is kernels_list[8000]

    plan = plans._get_cached_a8w8_mxscale_bmm_plan(
        "gfx950", 8000, torch.bfloat16, 17, 2, 128, 2048, 2
    )
    assert plan.workspace_spec is not None
    assert plan.workspace_spec.dtype == torch.float32
    assert plan.workspace_spec.shape == (2 * 2 * 32 * 128,)


def test_high_level_mxscale_caller_keeps_split1_and_workspace_routes(monkeypatch):
    batched = importlib.import_module("aiter.ops.batched_gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    raw_calls, public_calls = [], []

    def fake_public(*args, **kwargs):
        public_calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(
        batched,
        "_get_mxscale_bmm_launchers",
        lambda: (lambda *args: raw_calls.append(args), fake_public),
    )
    monkeypatch.setattr(
        batched,
        "_resolve_a8w8_mxscale_bmm_plan",
        lambda _g, m, _n, _k: (8311, 1) if m == 1 else (8000, 2),
    )
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)

    for m in (1, 17):
        XQ = torch.empty((m, 2, 128), dtype=fp8)
        x_scale = torch.empty((m, 2, 1), dtype=torch.uint8)
        batched._batched_gemm_a8w8_mxscale_impl(
            XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16
        )

    assert len(raw_calls) == len(public_calls) == 1
    assert raw_calls[0][5:] == (None, 8311, 1)
    assert public_calls[0][1]["kid"] == 8000
    assert public_calls[0][1]["layout"] == "mxscale_bmm"
    assert public_calls[0][1]["split_k"] == 2
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()


def test_tuned_a16_and_deepgemm_shim_forward_to_public_gemm(monkeypatch):
    tuned = importlib.import_module("aiter.tuned_gemm")
    deepgemm = importlib.import_module("aiter.ops.deepgemm")
    calls = []

    def fake(XQ, WQ, Y, **kwargs):
        calls.append((XQ, WQ, Y, kwargs))
        return Y

    monkeypatch.setattr(tuned, "_opus_launch", fake)
    monkeypatch.setattr(deepgemm._opus, "opus_gemm", fake)
    XQ = torch.empty((32, 128), dtype=torch.bfloat16)
    WQ = torch.empty((64, 128), dtype=torch.bfloat16)
    tuned_y = tuned.opus_gemm(
        XQ,
        WQ,
        1206,
        otype=torch.bfloat16,
        config={"splitK": 3},
    )
    with pytest.warns(DeprecationWarning):
        assert deepgemm.opus_gemm_a16w16_tune(
            XQ, WQ, tuned_y, kernelId=1206, splitK=3
        ) is tuned_y

    assert len(calls) == 2
    assert calls[0][3] == {"kid": 1206, "bias": None, "split_k": 3}
    assert calls[1][3] == {"kid": 1206, "split_k": 3}


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda o, b, f, x, w: o.opus_gemm(b, b, b, kid=-1), "unknown OPUS kid"),
        (
            lambda o, b, f, x, w: o.opus_gemm(b, b, b, kid=200, split_k=-1),
            "split_k must be non-negative",
        ),
        (
            lambda o, b, f, x, w: o.opus_gemm(f, f, b, kid=200),
            "requires bf16 XQ/WQ",
        ),
        (
            lambda o, b, f, x, w: o.opus_gemm(x, w, f, kid=1),
            "requires x_scale and w_scale",
        ),
        (
            lambda o, b, f, x, w: o.opus_bmm(
                x.unsqueeze(0), w.unsqueeze(0), f[:, :64].unsqueeze(0), kid=2
            ),
            "GEMM-only",
        ),
    ],
)
def test_public_contract_rejects_representative_mismatches(call, message):
    opus = importlib.import_module("aiter.ops.opus")
    bf16 = torch.empty((64, 128), dtype=torch.bfloat16)
    fp32 = torch.empty((64, 128), dtype=torch.float32)
    fp8 = getattr(torch, "float8_e4m3fnuz")
    X8 = torch.empty((64, 128), dtype=fp8)
    W8 = torch.empty_like(X8)

    with pytest.raises(ValueError, match=message):
        call(opus, bf16, fp32, X8, W8)
