# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in MI300/MI308 acceptance sweep for every canonical gfx942 OPUS kid."""

from __future__ import annotations

import gc
import os
import weakref
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from aiter.jit.core import compile_ops
from aiter.ops.opus import gemm_op_a16w16 as gemm, opus_bmm
from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan
from csrc.opus_gemm.opus_gemm_common import (
    get_kernel_instance,
    kernel_needs_external_workspace,
    kernels_list,
)
from op_tests.opus_a16w16_test_utils import _init_a16w16_workspace


_ENABLED = os.getenv("OPUS_GFX942_EXHAUSTIVE", "0") == "1"
_ENDPOINT = os.getenv("OPUS_GFX942_ENDPOINT", "current").strip().lower()
if _ENDPOINT not in {"current", "baseline"}:
    raise ValueError(
        "OPUS_GFX942_ENDPOINT must be 'current' or 'baseline', "
        f"got {_ENDPOINT!r}"
    )

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set OPUS_GFX942_EXHAUSTIVE=1 to run the gfx942 acceptance sweep",
)


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    develop=True,
)
def _baseline_a16_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    bias: Tensor | None,
    kernelId: int,
    splitK: int,
) -> Tensor: ...


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_workspace_init", develop=True)
def _baseline_workspace_init() -> None: ...


@dataclass(frozen=True)
class _Case:
    kid: int
    instance: object
    needs_workspace: bool

    @property
    def id(self) -> str:
        kind = "workspace" if self.needs_workspace else "direct"
        return f"{kind}-kid{self.kid}"


def _canonical_cases() -> tuple[_Case, ...]:
    rows = []
    for kid, instance in sorted(kernels_list.items()):
        if get_kernel_instance("gfx942", "a16w16", kid) is not instance:
            continue
        rows.append(
            _Case(
                kid=int(kid),
                instance=instance,
                needs_workspace=kernel_needs_external_workspace(
                    "gfx942", "a16w16", kid
                ),
            )
        )
    return tuple(rows)


_ALL_CASES = _canonical_cases()
_WORKSPACE_CASES = tuple(case for case in _ALL_CASES if case.needs_workspace)
_DIRECT_CASES = tuple(case for case in _ALL_CASES if not case.needs_workspace)

assert len(_ALL_CASES) == 22
assert len(_WORKSPACE_CASES) == 8
assert len(_DIRECT_CASES) == 14
assert {case.kid for case in _WORKSPACE_CASES} == {
    10200,
    10201,
    10203,
    10204,
    10205,
    10210,
    10213,
    10216,
}


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


@pytest.fixture(scope="session", autouse=True)
def _require_gfx942_and_load_module():
    if not _ENABLED:
        return
    arch = _runtime_arch()
    if arch != "gfx942":
        pytest.fail(f"gfx942 exhaustive sweep requires gfx942 hardware, got {arch}")

    from aiter.jit.core import get_module

    get_module("module_deepgemm_opus")
    if _ENDPOINT == "baseline":
        _baseline_workspace_init()


def _make_problem(case: _Case):
    M = int(case.instance.B_M)
    N = int(case.instance.B_N)
    K = 32 * int(case.instance.B_K)
    torch.manual_seed(0x942000 + case.kid)
    XQ = torch.randn((1, M, K), device="cuda", dtype=torch.bfloat16)
    WQ = torch.randn((1, N, K), device="cuda", dtype=torch.bfloat16)
    golden = torch.bmm(XQ.float(), WQ.float().transpose(1, 2))
    return XQ, WQ, golden


def _plan(case: _Case, XQ: Tensor, out_dtype: torch.dtype):
    props = torch.cuda.get_device_properties(XQ.device)
    return _get_cached_a16w16_launch_plan(
        "gfx942",
        int(case.instance.B_M),
        int(case.instance.B_N),
        int(XQ.shape[-1]),
        1,
        int(props.multi_processor_count),
        False,
        torch.bfloat16,
        out_dtype,
        case.kid,
        2 if case.needs_workspace else 0,
    )


def _assert_result(actual: Tensor, golden: Tensor) -> float:
    assert torch.isfinite(actual).all()
    max_abs = float((actual.float() - golden).abs().max().item())
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual.float(), golden, rtol=0.03, atol=0.5)
    else:
        torch.testing.assert_close(actual, golden, rtol=1e-3, atol=0.05)
    return max_abs


def _workspace_dtype(case: _Case) -> torch.dtype:
    return {
        "bf16_t": torch.bfloat16,
        "fp32_t": torch.float32,
    }[case.instance.splitk_workspace_dtype]


def _workspace_output_dtypes(case: _Case) -> tuple[torch.dtype, ...]:
    if case.instance.splitk_workspace_dtype == "bf16_t":
        return (torch.bfloat16,)
    return (torch.bfloat16, torch.float32)


def _baseline_launch(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    case: _Case,
) -> Tensor:
    _baseline_a16_raw(XQ, WQ, Y, None, case.kid, 2)
    return Y


@pytest.mark.parametrize("case", _WORKSPACE_CASES, ids=lambda case: case.id)
def test_every_gfx942_workspace_kid(case: _Case):
    """Run both output dtypes and verify caller/automatic workspace contracts."""
    XQ, WQ, golden = _make_problem(case)
    M, N = int(case.instance.B_M), int(case.instance.B_N)
    expected_workspace_dtype = _workspace_dtype(case)
    errors: dict[str, float] = {}

    if _ENDPOINT == "baseline":
        for out_dtype in _workspace_output_dtypes(case):
            Y = torch.empty((1, M, N), device=XQ.device, dtype=out_dtype)
            _baseline_launch(XQ, WQ, Y, case)
            torch.cuda.synchronize(XQ.device)
            errors[str(out_dtype)] = _assert_result(Y, golden)
        print(
            "[gfx942 exhaustive baseline] "
            f"kid={case.kid} family={case.instance.kernel_tag} "
            f"shape=(1,{M},{N},{XQ.shape[-1]}) "
            f"internal_workspace={case.instance.splitk_workspace_dtype} "
            f"split_k=2 errors={errors}"
        )
        return

    plan = _plan(case, XQ, torch.bfloat16)
    assert plan.resolved_kid == case.kid
    assert plan.workspace_capacity_split_k == 2
    workspace = _init_a16w16_workspace(plan, XQ)
    assert workspace is not None
    assert tuple(workspace.shape) == (2, 1, M, N)
    assert workspace.dtype == expected_workspace_dtype
    assert workspace.device == XQ.device
    assert workspace.is_contiguous()
    assert workspace.data_ptr() % 16 == 0
    workspace_ptr = workspace.data_ptr()

    for out_dtype in _workspace_output_dtypes(case):
        workspace.zero_()
        Y = torch.empty((1, M, N), device=XQ.device, dtype=out_dtype)
        returned = opus_bmm(
            XQ,
            WQ,
            Y,
            kid=case.kid,
            split_k=2,
            workspace=workspace,
        )
        assert returned is Y
        torch.cuda.synchronize(XQ.device)
        assert workspace.data_ptr() == workspace_ptr
        errors[str(out_dtype)] = _assert_result(Y, golden)

    original_empty = gemm.torch.empty
    allocated_specs = []
    allocated_refs = []

    def tracked_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        allocated_specs.append((tuple(tensor.shape), tensor.dtype, tensor.device))
        allocated_refs.append(weakref.ref(tensor))
        return tensor

    auto_y = original_empty((1, M, N), device=XQ.device, dtype=torch.bfloat16)
    gemm.torch.empty = tracked_empty
    try:
        returned = opus_bmm(XQ, WQ, auto_y, kid=case.kid, split_k=2)
    finally:
        gemm.torch.empty = original_empty
    assert returned is auto_y
    torch.cuda.synchronize(XQ.device)
    assert allocated_specs == [((2, 1, M, N), expected_workspace_dtype, XQ.device)]
    gc.collect()
    assert allocated_refs[0]() is None
    auto_error = _assert_result(auto_y, golden)

    print(
        "[gfx942 exhaustive current] "
        f"kid={case.kid} family={case.instance.kernel_tag} "
        f"shape=(1,{M},{N},{XQ.shape[-1]}) "
        f"workspace_shape={(2, 1, M, N)} "
        f"workspace_dtype={expected_workspace_dtype} split_k=2 "
        f"errors={errors} "
        f"auto_bf16_max_abs={auto_error:.6g}"
    )


@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in _WORKSPACE_CASES
        if case.instance.splitk_workspace_dtype == "bf16_t"
    ),
    ids=lambda case: case.id,
)
def test_gfx942_bf16_workspace_kids_reject_fp32_y(case: _Case):
    """Record the three physical-contract exclusions from the 16-case proposal."""
    XQ, WQ, _golden = _make_problem(case)
    M, N = int(case.instance.B_M), int(case.instance.B_N)
    Y = torch.empty((1, M, N), device=XQ.device, dtype=torch.float32)
    if _ENDPOINT == "baseline":
        call = lambda: _baseline_launch(XQ, WQ, Y, case)
    else:
        workspace = torch.empty(
            (2, 1, M, N), device=XQ.device, dtype=torch.bfloat16
        )
        call = lambda: opus_bmm(
            XQ,
            WQ,
            Y,
            kid=case.kid,
            split_k=2,
            workspace=workspace,
        )
    with pytest.raises(
        RuntimeError, match="bf16 workspace currently supports only bf16 Y"
    ):
        call()


@pytest.mark.skipif(
    _ENDPOINT == "baseline",
    reason="baseline mode intentionally covers only the 8 workspace kids",
)
@pytest.mark.parametrize("case", _DIRECT_CASES, ids=lambda case: case.id)
def test_every_gfx942_non_workspace_kid(case: _Case):
    """Run every declared output dtype and prove no workspace allocation."""
    XQ, WQ, golden = _make_problem(case)
    M, N = int(case.instance.B_M), int(case.instance.B_N)
    dtype_map = {"bf16_t": torch.bfloat16, "fp32_t": torch.float32}
    output_dtypes = tuple(dtype_map[token] for token in case.instance.output_dtypes)
    assert output_dtypes

    outputs = {
        dtype: torch.empty((1, M, N), device=XQ.device, dtype=dtype)
        for dtype in output_dtypes
    }
    for out_dtype, Y in outputs.items():
        plan = _plan(case, XQ, out_dtype)
        assert plan.resolved_kid == case.kid
        assert _init_a16w16_workspace(plan, XQ) is None

    original_empty = gemm.torch.empty

    def unexpected_empty(*_args, **_kwargs):
        raise AssertionError(
            f"non-workspace gfx942 kid {case.kid} attempted torch.empty"
        )

    errors: dict[str, float] = {}
    gemm.torch.empty = unexpected_empty
    try:
        for out_dtype, Y in outputs.items():
            if case.instance.kernel_tag == "a16w16_wave_k_coop_accum":
                Y.zero_()
            returned = opus_bmm(XQ, WQ, Y, kid=case.kid, split_k=0)
            assert returned is Y
            torch.cuda.synchronize(XQ.device)
            errors[str(out_dtype)] = _assert_result(Y, golden)
    finally:
        gemm.torch.empty = original_empty

    print(
        "[gfx942 exhaustive current] "
        f"kid={case.kid} family={case.instance.kernel_tag} "
        f"shape=(1,{M},{N},{XQ.shape[-1]}) workspace=none split_k=0 "
        f"errors={errors}"
    )
