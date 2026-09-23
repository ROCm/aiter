# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests that runtime launch paths consume shared MoE AOT requests."""

from aiter.aot.flydsl.tests.moe_test_utils import (
    TARGET,
    stage1_metadata,
    stage2_metadata,
    stage2_runtime_metadata,
)
from aiter.ops.flydsl.aot_backend import create_compile_context
from aiter.ops.flydsl.compile_request import CompileContext
from aiter.ops.flydsl.moe_compile_requests import (
    INT4_STAGE2_GEMM_OP_ID,
    stage1_compile_requests,
    stage2_compile_requests,
)


class _RuntimeBackend:
    def __init__(self):
        self.requests = []
        self.launcher = object()

    def compile_aot(self, request, *, context):  # pragma: no cover - protocol stub
        raise AssertionError("runtime must not call compile_aot")

    def load_aot(  # pragma: no cover - protocol stub
        self, request, *, context, strict=True
    ):
        raise AssertionError("developer runtime must not call load_aot")

    def resolve_aot(self, request, *, context):
        self.requests.append(request)
        return type("Artifact", (), {"launcher": self.launcher})()


def _runtime_context(backend):
    registry = create_compile_context(TARGET).registry
    return CompileContext(TARGET, registry, backend)


def test_runtime_stage1_and_stage2_resolve_the_shared_requests(monkeypatch):
    import torch

    from aiter.ops.flydsl import moe_kernels

    backend = _RuntimeBackend()
    context = _runtime_context(backend)
    monkeypatch.setattr(moe_kernels, "_run_compiled", lambda _exe, _args: None)

    a = torch.empty((2, 64), dtype=torch.uint8)
    w1 = torch.empty((4, 128, 64), dtype=torch.uint8)
    sorted_ids = torch.zeros(64, dtype=torch.int32)
    expert_ids = torch.zeros(2, dtype=torch.int32)
    num_valid = torch.zeros(2, dtype=torch.int32)
    out1 = torch.empty((2, 2, 64), dtype=torch.bfloat16)
    moe_kernels._flydsl_moe_stage1_impl(
        a,
        w1,
        sorted_ids,
        expert_ids,
        num_valid,
        out=out1,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=64,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        _build_mx_args=lambda *_args, **_kwargs: (),
        _compile_context=context,
    )
    expected_stage1 = stage1_compile_requests(
        stage1_metadata(
            model_dim=64,
            inter_dim=64,
            experts=4,
            topk=2,
            tile_n=64,
            tile_k=64,
            b_nt=0,
            persist_m=1,
        ),
        TARGET,
        registry=context.registry,
    )[0]
    assert backend.requests == [expected_stage1]

    backend.requests.clear()
    inter = torch.empty((2, 2, 64), dtype=torch.uint8)
    w2 = torch.empty((4, 64, 64), dtype=torch.uint8)
    out2 = torch.zeros((2, 64), dtype=torch.bfloat16)
    moe_kernels._flydsl_moe_stage2_impl(
        inter,
        w2,
        sorted_ids,
        expert_ids,
        num_valid,
        out=out2,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=64,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        _build_mx_args=lambda *_args, **_kwargs: (),
        _compile_context=context,
    )
    expected_stage2 = stage2_compile_requests(
        stage2_metadata(
            model_dim=64,
            inter_dim=64,
            experts=4,
            topk=2,
            tile_n=64,
            tile_k=64,
            sort_block_m=0,
            b_nt=0,
            doweight_stage2=False,
        ),
        stage2_runtime_metadata(token_num=2, routing_block_count=2),
        TARGET,
        registry=context.registry,
    )[0]
    assert backend.requests == [expected_stage2]


def test_a16_stage2_runtime_launch_uses_the_resolved_request(monkeypatch):
    import torch

    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.kernels import moe_2stage_a16wmix

    backend = _RuntimeBackend()
    context = _runtime_context(backend)
    captured = {}
    monkeypatch.setattr(
        moe_2stage_a16wmix,
        "flydsl_a16w4_gemm2",
        lambda **kwargs: captured.update(kwargs),
    )

    inter = torch.empty((64, 64), dtype=torch.bfloat16)
    w2 = torch.empty((4, 64, 32), dtype=torch.uint8)
    sorted_ids = torch.zeros(64, dtype=torch.int32)
    expert_ids = torch.zeros(2, dtype=torch.int32)
    num_valid = torch.zeros(2, dtype=torch.int32)
    out = torch.zeros((2, 64), dtype=torch.bfloat16)
    moe_kernels._flydsl_moe_stage2_impl(
        inter,
        w2,
        sorted_ids,
        expert_ids,
        num_valid,
        out=out,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=64,
        a_dtype="bf16",
        b_dtype="int4",
        out_dtype="bf16",
        persist=True,
        _compile_context=context,
    )

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.op_id == INT4_STAGE2_GEMM_OP_ID
    assert captured["_compile_exe"] is backend.launcher
    assert captured["persist"] is True
    assert captured["tile_n"] == request.as_kwargs()["tile_n"]
    assert captured["tile_k"] == request.as_kwargs()["tile_k"]
