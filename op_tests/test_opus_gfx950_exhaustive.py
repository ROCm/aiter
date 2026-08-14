# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in, shardable gfx950 acceptance sweep for every canonical a16w16 kid.

The ordinary OPUS suite intentionally uses a small representative kernel set.
This file is the release/acceptance complement: it resolves the final
canonical registry, launches every gfx950 a16w16 kid with both BF16 and FP32
output, and checks the result against a float32 Torch golden.

The sweep is opt-in because its JIT module must contain all 140 gfx950 kids::

    OPUS_GFX950_EXHAUSTIVE=1 \
    OPUS_GFX950_SHARD_INDEX=0 OPUS_GFX950_SHARD_COUNT=4 \
    pytest -q -s op_tests/test_opus_gfx950_exhaustive.py

Use one isolated ``AITER_JIT_DIR`` shared by the four runtime shards, but build
that directory once in a single process before starting the shards.  The
module's ``build/compiled_kids_opus.json`` sidecar must contain the canonical
gfx950 kid ids; otherwise strict dispatch correctly reports an uncompiled id.
"""

from __future__ import annotations

import gc
import os
import weakref
from dataclasses import dataclass

import pytest
import torch

from aiter.ops.opus import gemm_op_a16w16 as gemm, opus_gemm
from csrc.opus_gemm.opus_gemm_common import (
    get_kernel_instance,
    kernel_needs_external_workspace,
    kernels_list,
)


_ENABLED = os.getenv("OPUS_GFX950_EXHAUSTIVE", "0") == "1"
_SHARD_COUNT = int(os.getenv("OPUS_GFX950_SHARD_COUNT", "1"))
_SHARD_INDEX = int(os.getenv("OPUS_GFX950_SHARD_INDEX", "0"))
if _SHARD_COUNT <= 0:
    raise ValueError("OPUS_GFX950_SHARD_COUNT must be positive")
if not 0 <= _SHARD_INDEX < _SHARD_COUNT:
    raise ValueError(
        "OPUS_GFX950_SHARD_INDEX must satisfy "
        f"0 <= index < {_SHARD_COUNT}, got {_SHARD_INDEX}"
    )

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set OPUS_GFX950_EXHAUSTIVE=1 to run the 140-kid gfx950 sweep",
)


@dataclass(frozen=True)
class _Case:
    ordinal: int
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
        if get_kernel_instance("gfx950", "a16w16", kid) is not instance:
            continue
        rows.append(
            _Case(
                ordinal=len(rows),
                kid=int(kid),
                instance=instance,
                needs_workspace=kernel_needs_external_workspace(
                    "gfx950", "a16w16", kid
                ),
            )
        )
    return tuple(rows)


_ALL_CASES = _canonical_cases()
_WORKSPACE_CASES = tuple(c for c in _ALL_CASES if c.needs_workspace)
_DIRECT_CASES = tuple(c for c in _ALL_CASES if not c.needs_workspace)

# These are acceptance invariants, not copied launch lists.  The rows above
# still come from the final registry, so a collision or accidental family
# removal fails collection instead of silently reducing coverage.
assert len(_ALL_CASES) == 140
assert len(_WORKSPACE_CASES) == 48
assert len(_DIRECT_CASES) == 92
assert {c.kid for c in _WORKSPACE_CASES} == (
    set(range(200, 224)) | set(range(1200, 1224))
)
assert all(c.instance.splitk_workspace_dtype == "fp32_t" for c in _WORKSPACE_CASES)


def _this_shard(cases: tuple[_Case, ...]) -> tuple[_Case, ...]:
    # Shard by the ordinal in the combined canonical list.  This keeps a stable
    # assignment even if test functions or family ordering change.
    return tuple(c for c in cases if c.ordinal % _SHARD_COUNT == _SHARD_INDEX)


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


@pytest.fixture(scope="session", autouse=True)
def _require_gfx950_and_load_module():
    if not _ENABLED:
        return
    arch = _runtime_arch()
    if arch != "gfx950":
        pytest.skip(f"gfx950 exhaustive sweep requires gfx950 hardware, got {arch}")

    # Import the prebuilt all-kid module before individual tests temporarily
    # replace torch.empty to prove allocation/no-allocation contracts.
    from aiter.jit.core import get_module

    get_module("module_deepgemm_opus")


def _make_problem(case: _Case):
    instance = case.instance
    M = int(instance.B_M)
    N = int(instance.B_N)
    # Two-stage kids need enough iterations that splitK=2 is never locally
    # down-clamped by the largest prefetch depth (currently <= 9).  Direct
    # families require an even number of K tiles and at least two iterations.
    K = (32 if case.needs_workspace else 2) * int(instance.B_K)

    torch.manual_seed(0x950000 + case.kid)
    XQ = torch.randn((1, M, K), device="cuda", dtype=torch.bfloat16)
    WQ = torch.randn((1, N, K), device="cuda", dtype=torch.bfloat16)
    golden = torch.bmm(XQ.float(), WQ.float().transpose(1, 2))
    return XQ, WQ, golden


def _config(case: _Case, XQ: torch.Tensor, out_dtype: torch.dtype):
    instance = case.instance
    device = XQ.device
    props = torch.cuda.get_device_properties(device)
    return gemm._resolve_exact_a16w16_config(
        arch="gfx950",
        M=int(instance.B_M),
        N=int(instance.B_N),
        K=int(XQ.shape[-1]),
        batch=1,
        cu_num=int(props.multi_processor_count),
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=out_dtype,
        kid=case.kid,
        split_k=2 if case.needs_workspace else 0,
    )


def _assert_result(actual: torch.Tensor, golden: torch.Tensor) -> float:
    assert torch.isfinite(actual).all()
    max_abs = float((actual.float() - golden).abs().max().item())
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual.float(), golden, rtol=0.03, atol=0.5)
    else:
        torch.testing.assert_close(actual, golden, rtol=1e-3, atol=0.05)
    return max_abs


@pytest.mark.parametrize(
    "case", _this_shard(_WORKSPACE_CASES), ids=lambda case: case.id
)
def test_every_gfx950_workspace_kid(case: _Case):
    """Run both output dtypes, caller reuse, and auto-workspace lifetime."""
    XQ, WQ, golden = _make_problem(case)
    M, N = int(case.instance.B_M), int(case.instance.B_N)
    config = _config(case, XQ, torch.bfloat16)
    assert config.actual_kid == case.kid
    assert config.allocation_split_k == 2

    probe_y = torch.empty((1, M, N), device=XQ.device, dtype=torch.bfloat16)
    workspace = gemm._init_a16w16_workspace(config, XQ, probe_y)
    assert workspace is not None
    assert tuple(workspace.shape) == (2, 1, M, N)
    assert workspace.dtype == torch.float32
    assert workspace.device == XQ.device
    assert workspace.is_contiguous()
    assert workspace.data_ptr() % 16 == 0
    workspace_ptr = workspace.data_ptr()

    errors: dict[str, float] = {}
    for out_dtype in (torch.bfloat16, torch.float32):
        # Poisoning proves every physical partial used by the exact-tile case
        # is freshly produced; reuse cannot accidentally consume stale data.
        workspace.fill_(float("nan"))
        Y = torch.empty((1, M, N), device=XQ.device, dtype=out_dtype)
        returned = opus_gemm(
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
        assert torch.isfinite(workspace).all()
        errors[str(out_dtype)] = _assert_result(Y, golden)

    # Exercise the normal per-call Torch allocation too.  Retain only a weak
    # reference to the intercepted tensor: after launch/synchronization there
    # must be no hidden Python/C++ registry keeping that Tensor object alive.
    original_empty = gemm.torch.empty
    allocated_specs = []
    allocated_refs = []

    def tracked_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        allocated_specs.append(
            (tuple(tensor.shape), tensor.dtype, tensor.device, tensor.data_ptr())
        )
        allocated_refs.append(weakref.ref(tensor))
        return tensor

    auto_y = original_empty((1, M, N), device=XQ.device, dtype=torch.bfloat16)
    gemm.torch.empty = tracked_empty
    try:
        returned = opus_gemm(
            XQ, WQ, auto_y, kid=case.kid, split_k=2
        )
    finally:
        gemm.torch.empty = original_empty
    assert returned is auto_y
    torch.cuda.synchronize(XQ.device)
    assert len(allocated_specs) == 1
    shape, dtype, device, _ptr = allocated_specs[0]
    assert shape == (2, 1, M, N)
    assert dtype == torch.float32
    assert device == XQ.device
    gc.collect()
    assert allocated_refs[0]() is None
    auto_error = _assert_result(auto_y, golden)

    print(
        "[gfx950 exhaustive] "
        f"kid={case.kid} family={case.instance.kernel_tag} "
        f"shape=(1,{M},{N},{XQ.shape[-1]}) workspace=fp32/reused+auto "
        f"bf16_max_abs={errors[str(torch.bfloat16)]:.6g} "
        f"fp32_max_abs={errors[str(torch.float32)]:.6g} "
        f"auto_bf16_max_abs={auto_error:.6g}"
    )


@pytest.mark.parametrize("case", _this_shard(_DIRECT_CASES), ids=lambda case: case.id)
def test_every_gfx950_non_workspace_kid(case: _Case):
    """Run both output dtypes and prove the direct path allocates no workspace."""
    XQ, WQ, golden = _make_problem(case)
    M, N = int(case.instance.B_M), int(case.instance.B_N)
    outputs = {
        dtype: torch.empty((1, M, N), device=XQ.device, dtype=dtype)
        for dtype in (torch.bfloat16, torch.float32)
    }

    for out_dtype, Y in outputs.items():
        config = _config(case, XQ, out_dtype)
        assert config.actual_kid == case.kid
        assert gemm._init_a16w16_workspace(config, XQ, Y) is None

    original_empty = gemm.torch.empty

    def unexpected_empty(*_args, **_kwargs):
        raise AssertionError(
            f"non-workspace gfx950 kid {case.kid} attempted torch.empty"
        )

    errors: dict[str, float] = {}
    gemm.torch.empty = unexpected_empty
    try:
        for out_dtype, Y in outputs.items():
            returned = opus_gemm(
                XQ, WQ, Y, kid=case.kid, split_k=0
            )
            assert returned is Y
            torch.cuda.synchronize(XQ.device)
            errors[str(out_dtype)] = _assert_result(Y, golden)
    finally:
        gemm.torch.empty = original_empty

    print(
        "[gfx950 exhaustive] "
        f"kid={case.kid} family={case.instance.kernel_tag} "
        f"shape=(1,{M},{N},{XQ.shape[-1]}) workspace=none "
        f"bf16_max_abs={errors[str(torch.bfloat16)]:.6g} "
        f"fp32_max_abs={errors[str(torch.float32)]:.6g}"
    )
