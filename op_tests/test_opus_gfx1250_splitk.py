# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 reducer-grid admission and FP32 workspace integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.opus import gemm_op_a16w16, opus_gemm, policy
from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
from csrc.opus_gemm.opus_gemm_common import GFX1250_4WAVE_CO_KIDS, get_kernel_instance


def _plan_args(M=16, N=64, K=7168, output_dtype=torch.bfloat16):
    return {
        "arch": "gfx1250",
        "M": M,
        "N": N,
        "K": K,
        "batch": 1,
        "cu_num": 256,
        "has_bias": False,
        "input_dtype": torch.bfloat16,
        "output_dtype": output_dtype,
    }


@pytest.mark.parametrize("kid", (20000, 20100))
@pytest.mark.parametrize("M", (65535, 65536, 65537))
@pytest.mark.parametrize("split_k", (0, 1, 2))
@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float32))
def test_splitk_reduce_row_limit(kid, M, split_k, output_dtype):
    args = _plan_args(M=M, output_dtype=output_dtype)
    if M <= 65535:
        plan = _get_cached_a16w16_launch_plan(**args, kid=kid, split_k=split_k)
        assert plan.workspace_spec.dtype == torch.float32
        assert plan.workspace_spec.shape[0] == max(1, split_k)
    else:
        with pytest.raises(ValueError, match="requires M <= 65535"):
            _get_cached_a16w16_launch_plan(**args, kid=kid, split_k=split_k)
        assert (
            policy.resolve_a16w16_tuned_candidate(
                **args, requested_kid=kid, requested_split_k=split_k
            )
            is None
        )


@pytest.mark.parametrize("M", (65535, 65536, 65537))
def test_heuristic_reduce_row_limit(M):
    if M <= 65535:
        plan = policy.resolve_a16w16_heuristic_candidate(**_plan_args(M=M))
        assert plan is not None
        assert plan.workspace_spec.dtype == torch.float32
    else:
        with pytest.raises(RuntimeError, match="Tune this shape"):
            policy.resolve_a16w16_heuristic_candidate(**_plan_args(M=M))


@pytest.mark.parametrize("N", (64, 384, 1024, 2048))
def test_large_m_keeps_direct_co_candidates(N, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "csrc" / "opus_gemm")
    )
    from opus_gemm_tune import kid_rejects_shape

    for kid in (20000, 20100):
        instance = get_kernel_instance("gfx1250", "a16w16", kid)
        assert not kid_rejects_shape(instance, 65535, N, 7168)
        assert kid_rejects_shape(instance, 65536, N, 7168)

    kid = min(GFX1250_4WAVE_CO_KIDS)
    instance = get_kernel_instance("gfx1250", "a16w16", kid)
    assert not kid_rejects_shape(instance, 65536, N, 7168)
    plan = policy.resolve_a16w16_tuned_candidate(
        **_plan_args(M=65536, N=N), requested_kid=kid, requested_split_k=0
    )
    assert plan is not None
    assert plan.resolved_kid == kid
    assert plan.workspace_spec is None


@pytest.mark.parametrize("kid", (20000, 20100))
@pytest.mark.parametrize("caller_splits", (None, 2, 4))
@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float32))
def test_fp32_workspace_reaches_exact_launch(
    kid, caller_splits, output_dtype, monkeypatch
):
    calls = []

    def capture(_A, _B, _Y, _bias, workspace, selected_kid, split_k):
        calls.append((workspace, selected_kid, split_k))

    monkeypatch.setattr(
        gemm_op_a16w16, "_device_arch_and_cu", lambda _: ("gfx1250", 256)
    )
    monkeypatch.setattr(gemm_op_a16w16, "_opus_gemm_a16w16_launch_raw", capture)
    A = torch.empty((16, 512), device="meta", dtype=torch.bfloat16)
    B = torch.empty((64, 512), device="meta", dtype=torch.bfloat16)
    Y = torch.empty((16, 64), device="meta", dtype=output_dtype)
    workspace = (
        None
        if caller_splits is None
        else torch.empty((caller_splits, 16, 64), device="meta", dtype=torch.float32)
    )

    assert opus_gemm(A, B, Y, kid=kid, split_k=2, workspace=workspace) is Y
    assert len(calls) == 1
    actual_workspace, selected_kid, split_k = calls[0]
    assert (selected_kid, split_k) == (kid, 2)
    assert actual_workspace.dtype == torch.float32
    if workspace is None:
        assert actual_workspace.shape == (2, 16, 64)
    else:
        assert actual_workspace is workspace


@pytest.mark.parametrize("split_k", (0, 1, 2))
def test_large_m_rejects_before_exact_launch(split_k, monkeypatch):
    def unexpected_launch(*_args):
        pytest.fail("an invalid reduce grid reached the backend")

    monkeypatch.setattr(
        gemm_op_a16w16, "_device_arch_and_cu", lambda _: ("gfx1250", 256)
    )
    monkeypatch.setattr(
        gemm_op_a16w16, "_opus_gemm_a16w16_launch_raw", unexpected_launch
    )
    A = torch.empty((65536, 512), device="meta", dtype=torch.bfloat16)
    B = torch.empty((64, 512), device="meta", dtype=torch.bfloat16)
    Y = torch.empty((65536, 64), device="meta", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires M <= 65535"):
        opus_gemm(A, B, Y, kid=20000, split_k=split_k)


@pytest.mark.parametrize(
    "N,expected_kid", ((64, 21258), (384, 21173), (1024, 21179), (2048, 21179))
)
def test_large_m_production_csv_selects_direct_co(N, expected_kid, monkeypatch):
    import aiter.tuned_gemm as tuned

    csv_path = (
        Path(__file__).resolve().parents[1]
        / "aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv"
    )
    monkeypatch.setattr(
        tuned,
        "AITER_CONFIGS",
        SimpleNamespace(AITER_CONFIG_GEMM_BF16_FILE=str(csv_path)),
    )
    monkeypatch.setattr(tuned, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: 256)
    monkeypatch.setattr(
        gemm_op_a16w16, "_device_arch_and_cu", lambda _: ("gfx1250", 256)
    )
    calls = []

    def capture(_A, _B, _Y, _bias, workspace, selected_kid, split_k):
        calls.append((workspace, selected_kid, split_k))

    monkeypatch.setattr(gemm_op_a16w16, "_opus_gemm_a16w16_launch_raw", capture)
    tuned.get_GEMM_A16W16_config_.cache_clear()
    tuned.get_GEMM_A16W16_config.cache_clear()
    try:
        config = tuned.get_GEMM_A16W16_config(
            65536, N, 7168, False, str(torch.bfloat16), str(torch.bfloat16)
        )
        assert (config["libtype"], config["solidx"], config["splitK"]) == (
            "opus",
            expected_kid,
            0,
        )
        A = torch.empty((65536, 7168), device="meta", dtype=torch.bfloat16)
        B = torch.empty((N, 7168), device="meta", dtype=torch.bfloat16)
        result = tuned.opus_gemm(A, B, config["solidx"], config=config)
        assert result.shape == (65536, N)
        assert calls == [(None, expected_kid, 0)]
    finally:
        tuned.get_GEMM_A16W16_config_.cache_clear()
        tuned.get_GEMM_A16W16_config.cache_clear()


@pytest.mark.parametrize("kid", (20000, 20100))
@pytest.mark.parametrize("split_k", (1, 2))
@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float32))
def test_gfx1250_fp32_workspace_matches_torch(kid, split_k, output_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires gfx1250 hardware")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.gcnArchName.split(":", 1)[0] != "gfx1250":
        pytest.skip("requires gfx1250 hardware")

    torch.manual_seed(5162 + kid)
    M, N, K = 49, 96, 512
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    Y = torch.full((M, N), float("nan"), device="cuda", dtype=output_dtype)
    reference = A.float() @ B.float().T
    plan = _get_cached_a16w16_launch_plan(
        **_plan_args(M=M, N=N, K=K, output_dtype=output_dtype), kid=kid, split_k=split_k
    )
    assert plan.workspace_spec.dtype == torch.float32
    shape = plan.workspace_spec.shape

    opus_gemm(A, B, Y, kid=kid, split_k=split_k)
    automatic = Y.clone()
    workspace = torch.full(
        (shape[0] + 1, *shape[1:]), float("nan"), device="cuda", dtype=torch.float32
    )
    Y.fill_(float("nan"))
    opus_gemm(A, B, Y, kid=kid, split_k=split_k, workspace=workspace)
    torch.cuda.synchronize()
    torch.testing.assert_close(Y, automatic, rtol=0, atol=0)
    torch.testing.assert_close(
        Y.float(),
        reference,
        rtol=0.03 if output_dtype == torch.bfloat16 else 1e-3,
        atol=0.5 if output_dtype == torch.bfloat16 else 0.05,
    )

    invalid_workspaces = (
        torch.empty(shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(1, device="cuda", dtype=torch.float32),
        torch.empty((*shape, 2), device="cuda", dtype=torch.float32)[..., 0],
        torch.empty(shape, device="cpu", dtype=torch.float32),
    )
    for invalid in invalid_workspaces:
        with pytest.raises((ValueError, RuntimeError), match="workspace"):
            opus_gemm(A, B, Y, kid=kid, split_k=split_k, workspace=invalid)
