# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-side tests for the operation-specific exact-kid OPUS dispatcher."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
import weakref

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    GFX942_BF16WS_EXACT_N,
    get_kernel_instance,
    kernel_needs_external_workspace,
    kernels_list,
)

_ROOT = Path(__file__).resolve().parents[1]


def _a16_config(arch: str, kid: int, *, N=128, K=4096, split_k=2, batch=1):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    return gemm._resolve_exact_a16w16_config(
        arch=arch,
        M=128,
        N=N,
        K=K,
        batch=batch,
        cu_num=304,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        kid=kid,
        split_k=split_k,
    )


def test_final_registry_preserves_unique_arch_kid_bands_and_direct_lookup():
    assert len(kernels_list) == len(set(kernels_list)) == 706
    kids_by_arch = {
        arch: {
            kid
            for kid, instance in kernels_list.items()
            if (instance.arch_prefix or "gfx950").lower() == arch
        }
        for arch in ("gfx950", "gfx942", "gfx1250")
    }
    assert max(kids_by_arch["gfx950"]) < 10000
    assert min(kids_by_arch["gfx942"]) >= 10000
    assert max(kids_by_arch["gfx942"]) < 20000
    assert min(kids_by_arch["gfx1250"]) >= 20000
    assert kernels_list.get(1).kernel_tag == "a8w8_scale"
    assert kernels_list.get(2).kernel_tag == "a8w8"
    assert kernels_list.get(200).kernel_tag == "a16w16_flatmm_splitk"
    assert kernels_list.get(8000).kernel_tag == "a8w8_mxscale_bmm_flatmm_splitk"
    assert kernels_list.get(8653).kernel_tag == "a8w8_mxscale_bmm_flatmm_splitk"
    assert kernels_list.get(11000).kernel_tag == (
        "a8w8_blockscale_bpreshuffle_singlebuf"
    )
    assert kernels_list.get(20000).kernel_tag == "a16w16_cluster_tdm_splitk_ws"
    assert kernels_list.get(-1) is None


def test_common_queries_remain_arch_family_and_output_scoped():
    instance = get_kernel_instance("gfx942", "a16w16", 10200)
    assert instance is not None
    assert instance.name.startswith("opus_gemm_gfx942_splitk_legacy_")
    assert get_kernel_instance("gfx950", "a16w16", 10200) is None
    assert get_kernel_instance("gfx942", "a8w8", 11000) is None
    assert get_kernel_instance("gfx942", "a16w16", 11000) is None
    assert get_kernel_instance("gfx950", "a8w8", 2, torch.float32) is (
        kernels_list[2]
    )
    assert get_kernel_instance("gfx950", "a8w8", 2, torch.bfloat16) is None

    assert kernel_needs_external_workspace("gfx942", "a16w16", 10200)
    assert not kernel_needs_external_workspace("gfx950", "a16w16", 300)
    with pytest.raises(KeyError, match="unknown OPUS kernel"):
        kernel_needs_external_workspace("gfx942", "a16w16", 11000)


def test_a16_caller_policy_is_isolated_from_exact_execution():
    opus_dir = _ROOT / "aiter/ops/opus"
    policy_source = (opus_dir / "a16w16_policy.py").read_text()
    execution_source = (opus_dir / "gemm_op_a16w16.py").read_text()
    tuned_source = (_ROOT / "aiter/tuned_gemm.py").read_text()

    assert "_A16W16_HEURISTICS" in policy_source
    assert "def select_a16w16_heuristic_kid(" in policy_source
    assert "def resolve_a16w16_tuned_candidate(" in policy_source
    assert "def resolve_a16w16_heuristic_candidate(" in policy_source
    assert "def resolve_a16w16_caller_candidate(" in policy_source
    assert "def _heuristic_a16w16_kid_gfx950(" not in execution_source
    assert "def _select_a16w16_heuristic_kid(" not in execution_source
    assert "def _resolve_a16w16_caller_candidate(" not in execution_source
    assert "aiter.ops.opus.a16w16_policy" in tuned_source
    assert "resolve_a16w16_tuned_candidate" in tuned_source
    assert "resolve_a16w16_heuristic_candidate" not in tuned_source
    assert "resolve_a16w16_caller_candidate" not in tuned_source
    assert "aiter.ops.opus.gemm_op_a16w16 import (\n        _resolve" not in (
        tuned_source
    )

    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    execution = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    assert execution._select_a16w16_heuristic_kid is (
        policy.select_a16w16_heuristic_kid
    )
    assert execution._resolve_a16w16_caller_candidate is (
        policy.resolve_a16w16_caller_candidate
    )


def _gfx950_heuristic_golden(M, N, K, has_bias):
    split_barrier_ok = N % 16 == 0 and K % 64 == 0 and (K // 64) % 2 == 0
    if M <= 4:
        return 1208 if M % 64 == 0 and N % 64 == 0 and K % 128 == 0 else 208
    if M <= 64:
        return 1206 if M % 64 == 0 and N % 32 == 0 and K % 128 == 0 else 206
    if M <= 128:
        return 1200 if M % 64 == 0 and N % 64 == 0 and K % 64 == 0 else 200
    if split_barrier_ok and not has_bias:
        return 1300 if M % 256 == 0 and N % 256 == 0 else 300
    return 1200 if M % 64 == 0 and N % 64 == 0 and K % 64 == 0 else 200


def _gfx1250_heuristic_golden(M, N):
    if M % 32 == 0:
        if N % 128 == 0:
            return 20007
        if N % 64 == 0:
            return 20006
        if N % 32 == 0:
            return 20005
    if N % 128 == 0:
        return 20004
    if N % 64 == 0:
        return 20003
    return 20000


def test_gfx950_caller_heuristic_matches_baseline_boundary_policy():
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    for M in (1, 4, 5, 63, 64, 65, 127, 128, 129, 192, 255, 256, 257):
        for N in (31, 32, 63, 64, 65, 240, 256, 257):
            for K in (64, 127, 128, 192, 256):
                for has_bias in (False, True):
                    assert policy._heuristic_a16w16_kid_gfx950(
                        M, N, K, has_bias=has_bias
                    ) == _gfx950_heuristic_golden(M, N, K, has_bias)


def test_gfx1250_caller_heuristic_matches_baseline_boundary_policy():
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    for M in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65):
        for N in (31, 32, 33, 63, 64, 65, 127, 128, 129, 256):
            assert policy._heuristic_a16w16_kid_gfx1250(
                M, N, 4096
            ) == _gfx1250_heuristic_golden(M, N)


@pytest.mark.parametrize(
    ("M", "N", "K", "output_dtype", "has_bias", "expected"),
    [
        (48, 1024, 4096, "bf16", False, 10213),
        (512, 256, 4096, "bf16", False, 10203),
        (64, 1536, 4096, "bf16", False, 10205),
        (4, 4096, 1024, "bf16", False, 10300),
        (16, 1536, 4096, "bf16", False, 10305),
        (32, 1536, 2048, "bf16", False, 10303),
        (128, 512, 2048, "bf16", False, 10302),
        (256, 1024, 7168, "bf16", False, 10210),
        (160, 384, 4096, "bf16", False, 10201),
        (400, 384, 4096, "bf16", False, 10204),
        (128, 4096, 2048, "bf16", False, 10000),
        (128, 2048, 512, "bf16", False, 10000),
        (700, 256, 1024, "bf16", False, 10000),
        (700, 250, 1024, "bf16", False, 10201),
        (700, 512, 1000, "bf16", False, 10200),
        (32, 256, 1024, "fp32", False, 10201),
        (32, 512, 1024, "fp32", False, 10200),
        (32, 256, 1024, "bf16", True, 10201),
    ],
)
def test_gfx942_caller_heuristic_matches_baseline_policy(
    M, N, K, output_dtype, has_bias, expected
):
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    assert policy._heuristic_a16w16_kid_gfx942(
        M,
        N,
        K,
        has_bias=has_bias,
        output_dtype=output_dtype,
    ) == expected


def test_caller_heuristic_kids_are_in_default_compiled_floor():
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    from csrc.opus_gemm.opus_gemm_common import DEFAULT_COMPILED_KIDS_BY_ARCH

    cases = {
        "gfx950": (256, 256, 128),
        "gfx942": (256, 1024, 7168),
        "gfx1250": (32, 128, 4096),
    }
    for arch, (M, N, K) in cases.items():
        kid = policy.select_a16w16_heuristic_kid(
            arch=arch,
            M=M,
            N=N,
            K=K,
            batch=1,
            has_bias=False,
            output_dtype=torch.bfloat16,
        )
        assert kid in DEFAULT_COMPILED_KIDS_BY_ARCH[arch]


def test_caller_candidate_resolves_before_exact_public_launch():
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    common = dict(
        arch="gfx942",
        M=256,
        N=768,
        K=7168,
        batch=1,
        cu_num=304,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    tuned = policy.resolve_a16w16_tuned_candidate(
        **common, requested_kid=10210, requested_split_k=7
    )
    assert tuned is not None
    assert tuned.actual_kid == 10200
    assert tuned.allocation_split_k == 7

    heuristic = policy.resolve_a16w16_heuristic_candidate(
        **common, requested_split_k=0
    )
    assert heuristic is not None
    assert heuristic.actual_kid == 10200

    assert (
        policy.resolve_a16w16_tuned_candidate(
            **common, requested_kid=10216, requested_split_k=13
        )
        is None
    )

    # The former combined API remains behavior-compatible while production
    # callers migrate to the explicit tuned/heuristic policy operations.
    assert policy.resolve_a16w16_caller_candidate(
        **common, requested_kid=10210, requested_split_k=7
    ) == tuned
    assert policy.resolve_a16w16_caller_candidate(
        **common, requested_kid=None, requested_split_k=0
    ) == heuristic


def _tuned_row_key(gfx, cu_num, M, N, K, *, bias=False):
    return (
        gfx,
        cu_num,
        M,
        N,
        K,
        bias,
        str(torch.bfloat16),
        str(torch.bfloat16),
        False,
        False,
    )


def test_tuned_a16_row_does_not_invoke_heuristic(monkeypatch):
    tuned = importlib.import_module("aiter.tuned_gemm")
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    row = {
        "libtype": "opus",
        "solidx": 200,
        "splitK": 7,
        "kernelName": "",
    }
    monkeypatch.setattr(
        tuned,
        "get_GEMM_A16W16_config_",
        lambda: {_tuned_row_key("gfx950", 256, 128, 64, 4096): row},
    )
    monkeypatch.setattr(tuned, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "_opus_launch", object())

    def fail_heuristic(**_kwargs):
        raise AssertionError("heuristic ran before a valid tuned row")

    monkeypatch.setattr(policy, "select_a16w16_heuristic_kid", fail_heuristic)
    tuned.get_GEMM_A16W16_config.cache_clear()
    config = tuned.get_GEMM_A16W16_config(
        128,
        64,
        4096,
        False,
        str(torch.bfloat16),
        str(torch.bfloat16),
    )
    tuned.get_GEMM_A16W16_config.cache_clear()
    assert config["libtype"] == "opus"
    assert (config["solidx"], config["splitK"]) == (200, 7)


def test_invalid_tuned_pair_is_discarded_before_default_fallback(monkeypatch):
    tuned = importlib.import_module("aiter.tuned_gemm")
    stale = {
        "libtype": "opus",
        "solidx": 200,
        "splitK": 9,
        "kernelName": "",
    }
    monkeypatch.setattr(
        tuned,
        "get_GEMM_A16W16_config_",
        lambda: {_tuned_row_key("gfx942", 304, 256, 1024, 7168): stale},
    )
    monkeypatch.setattr(tuned, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: 304)
    monkeypatch.setattr(tuned, "_opus_launch", object())
    tuned.get_GEMM_A16W16_config.cache_clear()
    config = tuned.get_GEMM_A16W16_config(
        256,
        1024,
        7168,
        False,
        str(torch.bfloat16),
        str(torch.bfloat16),
    )
    tuned.get_GEMM_A16W16_config.cache_clear()
    assert config == {"libtype": "torch", "solidx": 0}


def test_tuned_adapter_passes_resolved_final_kid_to_public(monkeypatch):
    tuned = importlib.import_module("aiter.tuned_gemm")
    calls = []

    def fake_public(XQ, WQ, Y, **kwargs):
        calls.append((XQ.shape, WQ.shape, Y.shape, kwargs))
        return Y

    monkeypatch.setattr(tuned, "_opus_launch", fake_public)
    inp = torch.empty((32, 128), dtype=torch.bfloat16)
    weights = torch.empty((64, 128), dtype=torch.bfloat16)
    output = tuned.opus_gemm(
        inp,
        weights,
        1206,
        otype=torch.bfloat16,
        config={"splitK": 3},
    )
    assert output.shape == (32, 64)
    assert calls == [
        (
            torch.Size((32, 128)),
            torch.Size((64, 128)),
            torch.Size((32, 64)),
            {"kid": 1206, "bias": None, "split_k": 3},
        )
    ]


def test_mxscale_bmm_final_launch_plan_is_scalar_cached(monkeypatch):
    batched = importlib.import_module("aiter.ops.batched_gemm_op_a8w8")
    lookups = []
    row = {"libtype": "opus", "kernelId": 8311, "splitK": 1}

    monkeypatch.setattr(
        batched,
        "lookup_mxscale_bmm_config",
        lambda *shape: lookups.append(shape) or row,
    )
    monkeypatch.setattr(
        batched,
        "_mxscale_bmm_kid_runs_m",
        lambda kid, m: (kid, m) == (8311, 1),
    )
    batched._resolve_mxscale_bmm_launch.cache_clear()

    assert batched._resolve_mxscale_bmm_launch(2, 1, 1024, 4096) == (8311, 1)
    assert batched._resolve_mxscale_bmm_launch(2, 1, 1024, 4096) == (8311, 1)
    assert lookups == [(2, 1, 1024, 4096)]
    assert batched._resolve_mxscale_bmm_launch.cache_info().hits == 1
    assert batched._resolve_mxscale_bmm_launch.cache_info().maxsize == 1024
    batched._resolve_mxscale_bmm_launch.cache_clear()


def test_mxscale_bmm_high_level_split1_uses_checked_raw_launcher(monkeypatch):
    batched = importlib.import_module("aiter.ops.batched_gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((1, 2, 128), dtype=fp8)
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    x_scale = torch.empty((1, 2, 1), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    raw_calls = []
    public_calls = []

    def fake_raw(*args):
        raw_calls.append(args)

    def fake_public(*args, **kwargs):
        public_calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(
        batched, "_get_mxscale_bmm_launchers", lambda: (fake_raw, fake_public)
    )
    monkeypatch.setattr(
        batched, "_resolve_mxscale_bmm_launch", lambda *_shape: (8311, 1)
    )
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()

    Y = batched._batched_gemm_a8w8_mxscale_impl(
        XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16
    )
    assert Y.shape == (1, 2, 128)
    assert raw_calls == [(XQ, WQ, Y, x_scale, w_scale, None, 8311, 1)]
    assert public_calls == []
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()


def test_mxscale_bmm_high_level_workspace_keeps_unified_planner(monkeypatch):
    batched = importlib.import_module("aiter.ops.batched_gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((17, 2, 128), dtype=fp8)
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    x_scale = torch.empty((17, 2, 1), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    raw_calls = []
    public_calls = []

    def fake_public(*args, **kwargs):
        public_calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(
        batched,
        "_get_mxscale_bmm_launchers",
        lambda: (lambda *args: raw_calls.append(args), fake_public),
    )
    monkeypatch.setattr(
        batched, "_resolve_mxscale_bmm_launch", lambda *_shape: (8000, 2)
    )
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()

    Y = batched._batched_gemm_a8w8_mxscale_impl(
        XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16
    )
    assert raw_calls == []
    assert len(public_calls) == 1
    args, kwargs = public_calls[0]
    assert args[0].shape == (2, 17, 128)
    assert args[0].data_ptr() == XQ.data_ptr()
    assert args[1] is WQ
    assert args[2].shape == (2, 17, 128)
    assert args[2].data_ptr() == Y.data_ptr()
    assert kwargs["kid"] == 8000
    assert kwargs["layout"] == "mxscale_bmm"
    assert kwargs["x_scale"].shape == (2, 17, 1)
    assert kwargs["x_scale"].data_ptr() == x_scale.data_ptr()
    assert kwargs["w_scale"] is w_scale
    assert kwargs["split_k"] == 2
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()


def test_mxscale_bmm_high_level_reuses_one_scalar_launch_plan(monkeypatch):
    batched = importlib.import_module("aiter.ops.batched_gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((1, 2, 128), dtype=fp8)
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    x_scale = torch.empty((1, 2, 1), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    launcher_lookups = []
    plan_lookups = []
    raw_calls = []

    monkeypatch.setattr(
        batched,
        "_get_mxscale_bmm_launchers",
        lambda: launcher_lookups.append(True)
        or (lambda *args: raw_calls.append(args), lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        batched,
        "_resolve_mxscale_bmm_launch",
        lambda *shape: plan_lookups.append(shape) or (8311, 1),
    )
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()

    first = batched._batched_gemm_a8w8_mxscale_impl(
        XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16
    )
    second = batched._batched_gemm_a8w8_mxscale_impl(
        XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16
    )

    assert first is not second
    assert launcher_lookups == [True]
    assert plan_lookups == [(2, 1, 128, 128)]
    assert len(raw_calls) == 2
    assert all(call[5:] == (None, 8311, 1) for call in raw_calls)
    batched._MXSCALE_BMM_LAUNCH_PLANS.clear()


@pytest.mark.parametrize(
    ("gfx", "cu_num", "M", "N", "K", "bias", "otype", "expected_backend"),
    [
        ("gfx950", 256, 1, 64, 7168, False, torch.bfloat16, "skinny"),
        ("gfx950", 256, 16, 256, 256, False, torch.float32, "skinny"),
        ("gfx950", 256, 17, 128, 256, False, torch.bfloat16, "torch"),
        ("gfx942", 304, 32, 256, 1024, True, torch.bfloat16, "torch"),
        ("gfx90a", 120, 32, 256, 1024, False, torch.bfloat16, "torch"),
    ],
)
def test_no_tuned_row_uses_skinny_then_torch_without_heuristic(
    monkeypatch, gfx, cu_num, M, N, K, bias, otype, expected_backend
):
    tuned = importlib.import_module("aiter.tuned_gemm")
    policy = importlib.import_module("aiter.ops.opus.a16w16_policy")
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: {})
    monkeypatch.setattr(tuned, "get_gfx", lambda: gfx)
    monkeypatch.setattr(tuned, "get_cu_num", lambda: cu_num)
    monkeypatch.setattr(tuned, "_opus_launch", object())

    def fail_heuristic(**_kwargs):
        raise AssertionError("no-tuned fallback invoked the OPUS heuristic")

    monkeypatch.setattr(policy, "select_a16w16_heuristic_kid", fail_heuristic)
    tuned.get_GEMM_A16W16_config.cache_clear()
    config = tuned.get_GEMM_A16W16_config(
        M,
        N,
        K,
        bias,
        str(torch.bfloat16),
        str(otype),
    )
    tuned.get_GEMM_A16W16_config.cache_clear()
    assert config["libtype"] == expected_backend


def test_gfx942_exact_n_is_shared_and_contains_384():
    assert GFX942_BF16WS_EXACT_N == frozenset(
        {64, 128, 256, 384, 512, 1024, 2048}
    )


@pytest.mark.parametrize("requested", [0, 1, 3, 17])
@pytest.mark.parametrize("kid", [10200, 10201, 10203, 10204, 10210, 10213, 10216])
def test_gfx942_split_resolution_matches_launcher_constraints(kid, requested):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    instance = get_kernel_instance("gfx942", "a16w16", kid)
    assert instance is not None
    allocation, effective = gemm._resolve_gfx942_split_k(
        instance,
        M=257,
        N=769,
        K=4096,
        batch=2,
        cu_num=304,
        requested=requested,
    )
    assert allocation >= 1
    assert 1 <= effective <= allocation
    total_iters = (4096 + instance.B_K - 1) // instance.B_K
    iters_full = (total_iters + effective - 1) // effective
    last_loops = total_iters - (effective - 1) * iters_full
    assert iters_full >= 2 and last_loops >= 2
    if instance.kernel_tag in gemm._EVEN_LOOP_SPLITK_TAGS:
        assert iters_full % 2 == last_loops % 2 == 0


def test_gfx942_auto_and_explicit_split_k_keep_allocation_contract():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    p1 = get_kernel_instance("gfx942", "a16w16", 10201)
    legacy = get_kernel_instance("gfx942", "a16w16", 10200)
    assert p1 is not None and legacy is not None
    assert gemm._resolve_gfx942_split_k(
        p1, M=512, N=512, K=2048, batch=1, cu_num=304, requested=0
    ) == (5, 4)
    assert gemm._resolve_gfx942_split_k(
        legacy, M=128, N=128, K=4096, batch=1, cu_num=304, requested=17
    ) == (17, 16)


@pytest.mark.parametrize("kid", [10210, 10213, 10216])
def test_gfx942_bf16_workspace_kid_is_exact_not_redirected(kid):
    with pytest.raises(ValueError, match="exact kid.*requires N"):
        _a16_config("gfx942", kid, N=768)
    config = _a16_config("gfx942", kid, N=384)
    assert config.actual_kid == kid


def test_exact_a16_config_rejects_wrong_arch_and_batched_gfx1250():
    with pytest.raises(ValueError, match="not an a16w16 kernel"):
        _a16_config("gfx942", 200)
    with pytest.raises(ValueError, match="incompatible with shape"):
        _a16_config("gfx1250", 20000, batch=2)


def test_unified_dispatch_routes_a16_by_final_kid(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    calls = []

    def fake(XQ, WQ, Y, bias, **kwargs):
        calls.append((XQ, WQ, Y, bias, kwargs))
        return Y

    monkeypatch.setattr(a16, "_launch_a16w16_bmm", fake)
    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
    bias = torch.empty((64,), dtype=torch.bfloat16)
    assert opus.opus_bmm(
        XQ, WQ, Y, kid=200, bias=bias, split_k=2
    ) is Y
    assert len(calls) == 1
    assert calls[0][:4] == (XQ, WQ, Y, bias)
    assert calls[0][4]["kid"] == 200
    assert calls[0][4]["split_k"] == 2
    assert calls[0][4]["workspace"] is None
    assert calls[0][4]["route_arch"] == "gfx950"
    assert calls[0][4]["instance"] is kernels_list[200]


def test_unified_dispatch_routes_logical_a16_gemm_without_public_batch_axis(
    monkeypatch,
):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    calls = []

    def fake(XQ, WQ, Y, bias, **kwargs):
        calls.append((XQ, WQ, Y, bias, kwargs))
        return Y

    monkeypatch.setattr(a16, "_launch_a16w16_gemm", fake)
    XQ = torch.empty((64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((64, 512), dtype=torch.bfloat16)
    Y = torch.empty((64, 64), dtype=torch.bfloat16)
    assert opus.opus_gemm(XQ, WQ, Y, kid=200, split_k=2) is Y
    assert len(calls) == 1
    assert calls[0][:4] == (XQ, WQ, Y, None)
    assert calls[0][4]["kid"] == 200
    assert calls[0][4]["instance"] is kernels_list[200]


def test_public_gemm_and_bmm_reject_the_other_operation_rank():
    opus = importlib.import_module("aiter.ops.opus")
    gemm_tensor = torch.empty((64, 512), dtype=torch.bfloat16)
    bmm_tensor = torch.empty((1, 64, 512), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="opus_gemm expects logical 2D.*opus_bmm"):
        opus.opus_gemm(bmm_tensor, bmm_tensor, bmm_tensor, kid=200)
    with pytest.raises(
        ValueError, match="opus_bmm expects batch-first 3D.*opus_gemm"
    ):
        opus.opus_bmm(gemm_tensor, gemm_tensor, gemm_tensor, kid=200)


def test_unified_gfx950_a16_caller_workspace_uses_checked_cabi(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    calls = []

    def fail_generic_planner(*_args, **_kwargs):
        raise AssertionError("caller-workspace fast path re-entered planner")

    def fake_raw(XQ, WQ, Y, bias, workspace, kid, split_k):
        calls.append((XQ, WQ, Y, bias, workspace, kid, split_k))

    monkeypatch.setattr(a16, "_explicit_a16w16_launch", fail_generic_planner)
    monkeypatch.setattr(a16, "_opus_gemm_a16w16_launch_ctypes_raw", fake_raw)
    a16._cached_explicit_a16w16_plan.cache_clear()

    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.float32)
    workspace = torch.empty((2, 1, 64, 64), dtype=torch.float32)
    assert opus.opus_bmm(
        XQ,
        WQ,
        Y,
        kid=200,
        split_k=2,
        workspace=workspace,
    ) is Y
    assert calls == [(XQ, WQ, Y, None, workspace, 200, 2)]


def test_explicit_public_a16_kid_bypasses_tuned_and_heuristic_policy(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    tuned = importlib.import_module("aiter.tuned_gemm")

    def fail_policy(*_args, **_kwargs):
        raise AssertionError("explicit public kid entered caller selection")

    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config", fail_policy)
    monkeypatch.setattr(
        a16, "_launch_a16w16_bmm", lambda XQ, WQ, Y, *_a, **_k: Y
    )
    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
    assert opus.opus_bmm(XQ, WQ, Y, kid=200, split_k=2) is Y


def test_public_contract_cache_is_scalar_only(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")

    class CountingRegistry(dict):
        def __init__(self, values):
            super().__init__(values)
            self.lookups = 0

        def get(self, key, default=None):
            self.lookups += 1
            return super().get(key, default)

    registry = CountingRegistry(opus.kernels_list)
    monkeypatch.setattr(opus, "kernels_list", registry)
    monkeypatch.setattr(
        a16, "_launch_a16w16_bmm", lambda XQ, WQ, Y, *_a, **_k: Y
    )
    opus._cached_public_contract.cache_clear()

    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
    references = [weakref.ref(tensor) for tensor in (XQ, WQ, Y)]
    assert opus.opus_bmm(XQ, WQ, Y, kid=200, split_k=2) is Y
    assert opus.opus_bmm(XQ, WQ, Y, kid=200, split_k=2) is Y
    assert registry.lookups == 1
    assert opus._cached_public_contract.cache_info().maxsize == 4096
    assert opus._cached_public_contract.cache_info().hits == 1

    del XQ, WQ, Y
    gc.collect()
    assert all(reference() is None for reference in references)
    opus._cached_public_contract.cache_clear()


@pytest.mark.parametrize(
    ("kid", "arch", "raw_name", "layout", "with_scale"),
    [
        (2, "gfx950", "_opus_gemm_a8w8_launch_raw", "plain", False),
        (
            1,
            "gfx950",
            "_opus_gemm_a8w8_blockscale_launch_raw",
            "plain",
            True,
        ),
        (
            11000,
            "gfx942",
            "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
            "bpreshuffle",
            True,
        ),
    ],
)
def test_unified_dispatch_routes_a8_by_final_kid(
    monkeypatch, kid, arch, raw_name, layout, with_scale
):
    opus = importlib.import_module("aiter.ops.opus")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(opus, "_require_gpu_tensor", lambda _tensor: None)
    monkeypatch.setattr(a8, raw_name, fake)
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((128, 256), dtype=fp8)
    WQ = torch.empty((128, 256), dtype=fp8)
    Y = torch.empty(
        (128, 128),
        dtype=torch.bfloat16 if kid == 11000 else torch.float32,
    )
    kwargs = {"kid": kid, "layout": layout}
    if with_scale:
        kwargs.update(
            x_scale=torch.empty((128, 2), dtype=torch.float32),
            w_scale=torch.empty((1, 2), dtype=torch.float32),
        )
    assert opus.opus_gemm(XQ, WQ, Y, **kwargs) is Y
    assert len(calls) == 1
    raw_args, raw_kwargs = calls[0]
    assert raw_kwargs == {}
    assert raw_args[0].shape == (1, 128, 256)
    assert raw_args[0].data_ptr() == XQ.data_ptr()
    assert raw_args[1].shape == (1, 128, 256)
    assert raw_args[1].data_ptr() == WQ.data_ptr()
    raw_y = raw_args[4] if kid == 11000 else raw_args[2]
    assert raw_y.shape == (1, 128, 128)
    assert raw_y.data_ptr() == Y.data_ptr()


def test_unified_dispatch_routes_gfx942_bpreshuffle_batch_one_bmm(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    calls = []
    monkeypatch.setattr(opus, "_require_gpu_tensor", lambda _tensor: None)
    monkeypatch.setattr(
        a8,
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
        lambda *args: calls.append(args),
    )
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((1, 128, 256), dtype=fp8)
    WQ = torch.empty((1, 128, 256), dtype=fp8)
    Y = torch.empty((1, 128, 128), dtype=torch.bfloat16)
    x_scale = torch.empty((1, 128, 2), dtype=torch.float32)
    w_scale = torch.empty((1, 1, 2), dtype=torch.float32)

    assert opus.opus_bmm(
        XQ,
        WQ,
        Y,
        kid=11000,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
    ) is Y
    assert len(calls) == 1
    raw_x, raw_w, raw_x_scale, raw_w_scale, raw_y, raw_kid = calls[0]
    assert raw_x is XQ and raw_w is WQ and raw_y is Y
    assert raw_x_scale.shape == (128, 2)
    assert raw_x_scale.data_ptr() == x_scale.data_ptr()
    assert raw_w_scale.shape == (1, 2)
    assert raw_w_scale.data_ptr() == w_scale.data_ptr()
    assert raw_kid == 11000


def test_unified_dispatch_routes_mxscale_bmm_by_global_exact_kid(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    calls = []

    def fake(XQ, WQ, Y, x_scale, w_scale, **kwargs):
        calls.append((XQ, WQ, Y, x_scale, w_scale, kwargs))
        return Y

    monkeypatch.setattr(opus, "_require_gpu_tensor", lambda _tensor: None)
    monkeypatch.setattr(a8, "_launch_a8w8_mxscale_bmm", fake)
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((2, 17, 256), dtype=fp8)
    WQ = torch.empty((2, 128, 256), dtype=fp8)
    Y = torch.empty((2, 17, 128), dtype=torch.bfloat16)
    x_scale = torch.empty((2, 17, 2), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 2), dtype=torch.uint8)
    assert opus.opus_bmm(
        XQ,
        WQ,
        Y,
        kid=8312,
        layout="mxfp8_bmm",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=2,
    ) is Y
    assert calls == [
        (
            XQ,
            WQ,
            Y,
            x_scale,
            w_scale,
            {
                "kid": 8312,
                "split_k": 2,
                "workspace": None,
                "route_arch": "gfx950",
                "instance": kernels_list[8312],
            },
        )
    ]


def test_unified_dispatch_uses_checked_raw_mxscale_bmm_split1_hot_path(
    monkeypatch,
):
    opus = importlib.import_module("aiter.ops.opus")
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((2, 1, 128), dtype=fp8)
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    Y = torch.empty((2, 1, 128), dtype=torch.bfloat16)
    x_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    calls = []

    monkeypatch.setattr(opus, "_require_gpu_tensor", lambda _tensor: None)
    monkeypatch.setattr(
        a8,
        "_opus_gemm_a8w8_mxscale_bmm_launch_raw",
        lambda *args: calls.append(args),
    )
    assert opus.opus_bmm(
        XQ,
        WQ,
        Y,
        kid=8311,
        layout="mxscale_bmm",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=1,
    ) is Y
    assert len(calls) == 1
    raw_x, raw_w, raw_y, raw_x_scale, raw_w_scale, workspace, kid, split = calls[0]
    assert raw_x.shape == (1, 2, 128) and raw_x.data_ptr() == XQ.data_ptr()
    assert raw_w is WQ
    assert raw_y.shape == (1, 2, 128) and raw_y.data_ptr() == Y.data_ptr()
    assert raw_x_scale.shape == (1, 2, 1)
    assert raw_x_scale.data_ptr() == x_scale.data_ptr()
    assert raw_w_scale is w_scale
    assert workspace is None
    assert (kid, split) == (8311, 1)


def test_mxscale_bmm_plan_uses_registry_and_torch_workspace_sizes():
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    a8._cached_a8w8_mxscale_bmm_plan.cache_clear()
    assert a8._cached_a8w8_mxscale_bmm_plan(
        "gfx950", 8000, torch.bfloat16, 17, 2, 128, 2048, 2
    ) == (2, 2 * 2 * 32 * 128)
    # fused kid adds one aligned counter region after the FP32 partials.
    split, fused_numel = a8._cached_a8w8_mxscale_bmm_plan(
        "gfx950", 8100, torch.float32, 17, 2, 128, 2048, 2
    )
    assert split == 2
    assert fused_numel > 2 * 2 * 32 * 128
    with pytest.raises(ValueError, match="requires split_k <= 1"):
        a8._cached_a8w8_mxscale_bmm_plan(
            "gfx950", 8646, torch.bfloat16, 17, 2, 128, 2048, 2
        )
    assert a8._cached_a8w8_mxscale_bmm_plan.cache_info().maxsize == 4096
    a8._cached_a8w8_mxscale_bmm_plan.cache_clear()


def test_mxscale_bmm_split1_hot_path_defers_tensor_contract_to_cpp(monkeypatch):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((1, 2, 128), dtype=fp8)
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    Y = torch.empty((1, 2, 128), dtype=torch.bfloat16)
    x_scale = torch.empty((1, 2, 1), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    calls = []

    def fail_duplicate_python_validation(*_args, **_kwargs):
        raise AssertionError("split_k=1 hot path repeated the C++ Tensor contract")

    def fake_raw(*args):
        calls.append(args)

    monkeypatch.setattr(a8, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        a8,
        "_validate_a8w8_mxscale_bmm_tensors",
        fail_duplicate_python_validation,
    )
    monkeypatch.setattr(a8, "_opus_gemm_a8w8_mxscale_bmm_launch_raw", fake_raw)

    assert a8._launch_a8w8_mxscale_bmm_exact(
        XQ,
        WQ,
        Y,
        x_scale,
        w_scale,
        kid=8311,
        split_k=1,
        workspace=None,
    ) is Y
    assert calls == [(XQ, WQ, Y, x_scale, w_scale, None, 8311, 1)]


def test_mxscale_bmm_workspace_path_retains_python_contract_validation(monkeypatch):
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    fp8 = getattr(torch, "float8_e4m3fnuz")
    XQ = torch.empty((17, 2, 128), dtype=fp8)
    WQ = torch.empty((2, 128, 128), dtype=fp8)
    Y = torch.empty((17, 2, 128), dtype=torch.bfloat16)
    x_scale = torch.empty((17, 2, 1), dtype=torch.uint8)
    w_scale = torch.empty((2, 1, 1), dtype=torch.uint8)
    validation_calls = []
    raw_calls = []
    validate = a8._validate_a8w8_mxscale_bmm_tensors

    def tracking_validation(*args):
        validation_calls.append(args)
        return validate(*args)

    monkeypatch.setattr(a8, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        a8,
        "_validate_a8w8_mxscale_bmm_tensors",
        tracking_validation,
    )
    monkeypatch.setattr(
        a8,
        "_opus_gemm_a8w8_mxscale_bmm_launch_raw",
        lambda *args: raw_calls.append(args),
    )

    assert a8._launch_a8w8_mxscale_bmm_exact(
        XQ,
        WQ,
        Y,
        x_scale,
        w_scale,
        kid=8000,
        split_k=2,
        workspace=None,
    ) is Y
    assert validation_calls == [(XQ, WQ, Y, x_scale, w_scale)]
    assert len(raw_calls) == 1
    workspace = raw_calls[0][5]
    assert workspace.dtype == torch.float32
    assert workspace.numel() == 2 * 2 * 32 * 128


def test_mxscale_bmm_malformed_input_rank_still_fails_before_plan():
    a8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    with pytest.raises(ValueError, match="XQ and WQ must be 3D"):
        a8._mxscale_bmm_shape_for_plan(
            torch.empty((2, 128)),
            torch.empty((2, 128, 128)),
        )


def test_unified_dispatch_rejects_contract_mismatches_before_launch(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    bf16 = torch.empty((64, 64), dtype=torch.bfloat16)
    fp32 = torch.empty((64, 64), dtype=torch.float32)
    fp8 = getattr(torch, "float8_e4m3fnuz")
    X8 = torch.empty((64, 128), dtype=fp8)
    W8 = torch.empty((64, 128), dtype=fp8)

    with pytest.raises(ValueError, match="unknown OPUS kid"):
        opus.opus_gemm(bf16, bf16, bf16, kid=-1)
    with pytest.raises(ValueError, match="kid must be an integer"):
        opus.opus_gemm(bf16, bf16, bf16, kid=200.5)
    with pytest.raises(ValueError, match="split_k must be non-negative"):
        opus.opus_gemm(bf16, bf16, bf16, kid=200, split_k=-1)
    with pytest.raises(ValueError, match="does not support Y.dtype"):
        opus.opus_gemm(X8, W8, bf16, kid=2)
    with pytest.raises(ValueError, match="requires bf16 XQ/WQ"):
        opus.opus_gemm(fp32, fp32, bf16, kid=200)
    with pytest.raises(ValueError, match="requires x_scale and w_scale"):
        opus.opus_gemm(X8, W8, fp32, kid=1)
    with pytest.raises(ValueError, match="requires x_scale and w_scale together"):
        opus.opus_gemm(
            X8,
            W8,
            fp32,
            kid=1,
            x_scale=torch.empty((1, 1)),
        )
    with pytest.raises(ValueError, match="requires layout='bpreshuffle'"):
        opus.opus_gemm(
            X8,
            W8,
            bf16,
            kid=11000,
            x_scale=torch.empty((1, 1)),
            w_scale=torch.empty((1, 1)),
        )


def test_unified_dispatch_rejects_cross_arch_kid(monkeypatch):
    opus = importlib.import_module("aiter.ops.opus")
    a16 = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    monkeypatch.setattr(
        a16, "_device_arch_and_cu", lambda _device: ("gfx942", 304)
    )
    a16._cached_explicit_a16w16_plan.cache_clear()
    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="not an a16w16 kernel.*gfx942"):
        opus.opus_bmm(XQ, WQ, Y, kid=200, split_k=2)


def test_cpp_runtime_shape_policy_is_removed_but_physical_4g_safety_remains():
    generator = (_ROOT / "csrc/opus_gemm/gen_instances.py").read_text()
    common = (_ROOT / "csrc/opus_gemm/opus_gemm_common.py").read_text()

    assert "def gen_lookup_dict(" not in generator
    assert "def get_tune_dict(" not in generator
    assert "_combined_opus_tuned.csv" not in generator
    assert "codegen.gen_lookup_dict" not in generator
    assert "default_kernels_dict" not in common

    for arch in ("gfx942", "gfx950", "gfx1250"):
        include_dir = _ROOT / f"csrc/opus_gemm/include/{arch}"
        header = (include_dir / f"opus_gemm_arch_{arch}.cuh").read_text()
        assert '#include "opus_gemm_a16w16_kid_dispatch.h"' in header
        assert "find_kid(" in header
        assert "workspace_entry(" in header
        for removed in (
            "opus_gemm_lookup.h",
            "opus_gemm_a16w16_tune_lookup.h",
            "OpusA16W16Shape",
            "OpusA16W16RuntimeEntry",
            "find_shape_kid",
            "opus_select_a16w16_kid",
            "GENERATE_OPUS_LOOKUP_TABLE",
            "check_shape_4g",
        ):
            assert removed not in header
        assert not (include_dir / f"opus_gemm_heuristic_dispatch_{arch}.cuh").exists()

    tuner = (_ROOT / "csrc/opus_gemm/opus_gemm_tune.py").read_text()
    assert 'getattr(k_inst, "is_4g_safe", False)' in tuner
    assert "M * K * 2 > _UINT32_MAX_BYTES" in tuner


def test_gfx942_direct_workspace_pointer_keeps_wave_uniformization():
    include = _ROOT / "csrc/opus_gemm/include/gfx942/a16w16"
    traits = (include / "opus_gemm_traits_a16w16.cuh").read_text()
    assert "opus_gfx942_uniform_ws_ptr(Ptr ptr_ws)" in traits
    assert traits.count("__builtin_amdgcn_readfirstlane") >= 2
    assert "void*       __restrict__ ptr_ws" in traits
    assert "opus_splitk_ws_handle" not in traits

    for filename in (
        "opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh",
        "opus_gemm_pipeline_a16w16_kbuf1.cuh",
        "opus_gemm_pipeline_a16w16_kbuf2v.cuh",
        "opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh",
        "opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh",
    ):
        source = (include / filename).read_text()
        assert "opus_gfx942_uniform_ws_ptr<D_WS>(kargs.ptr_ws)" in source

    reduce = (include / "splitk_reduce_gfx942.cuh").read_text()
    assert reduce.count("opus_gfx942_uniform_ws_ptr<D_WS>(ws_ptr)") >= 2
    assert "const void*" in reduce
