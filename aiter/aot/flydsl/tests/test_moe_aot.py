# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only behavior tests for the MoE AOT job adapter."""

from collections import Counter

from aiter.aot.flydsl import moe as aot_moe
from aiter.aot.flydsl.tests.moe_test_utils import (
    TARGET,
    stage1_metadata,
    stage2_metadata,
    stage2_runtime_metadata,
)
from aiter.ops.flydsl.aot_backend import create_compile_context
from aiter.ops.flydsl.moe_compile_requests import (
    A16_STAGE1_GEMM_OP_ID,
    A16_STAGE2_GEMM_OP_ID,
    CKTILE_SWIGLU_AND_MUL_OP_ID,
    FHMOE_STAGE1_GEMM_OP_ID,
    FHMOE_STAGE2_GEMM_OP_ID,
    FQ_ACTIVATION_OP_ID,
    INT4_STAGE1_GEMM_OP_ID,
    INT4_STAGE2_GEMM_OP_ID,
    MIXED_STAGE1_GEMM_OP_ID,
    MIXED_STAGE2_GEMM_OP_ID,
    PLAIN_REDUCTION_OP_ID,
    Stage2RuntimeMetadata,
    stage1_compile_requests,
    stage2_compile_requests,
)


def test_all_configured_moe_jobs_build_requests_without_tensors_or_a_gpu():
    context = create_compile_context(TARGET)
    stage_counts = Counter()
    op_counts = Counter()
    jobs = aot_moe.get_aot_jobs()

    for job in jobs:
        requests = aot_moe.build_moe_compile_requests(job, context)
        stage_counts[job["stage"]] += 1
        assert requests
        assert all(request.target == TARGET for request in requests)
        op_counts.update(request.op_id for request in requests)

    assert all(stage_counts[stage] > 0 for stage in (1, 2, "epilogue"))
    for op_id in (
        MIXED_STAGE1_GEMM_OP_ID,
        MIXED_STAGE2_GEMM_OP_ID,
        A16_STAGE1_GEMM_OP_ID,
        A16_STAGE2_GEMM_OP_ID,
        INT4_STAGE1_GEMM_OP_ID,
        INT4_STAGE2_GEMM_OP_ID,
        FHMOE_STAGE1_GEMM_OP_ID,
        FHMOE_STAGE2_GEMM_OP_ID,
        FQ_ACTIVATION_OP_ID,
        CKTILE_SWIGLU_AND_MUL_OP_ID,
        PLAIN_REDUCTION_OP_ID,
    ):
        assert op_counts[op_id] > 0
    assert not any(
        job.get("enable_bias")
        and job.get("a_dtype") == "bf16"
        and job.get("b_dtype") in ("fp4", "int4")
        for job in jobs
    )


def test_compile_one_config_delegates_to_the_request_based_job_compiler(monkeypatch):
    captured = {}

    def compile_job(job):
        captured.update(job)
        return (object(),)

    monkeypatch.setattr(aot_moe, "compile_moe_job", compile_job)
    result = aot_moe.compile_one_config(
        kernel_name="test",
        model_dim=7168,
        inter_dim=2048,
        experts=256,
        topk=8,
        cu_num=256,
        stage=1,
    )

    assert result["compile_requests"] == 1
    assert result["compile_time"] is not None
    assert captured["stage"] == 1
    assert captured["model_dim"] == 7168


def test_compile_moe_job_builds_then_compiles_each_shared_request(monkeypatch):
    context = create_compile_context(TARGET)
    request = stage1_compile_requests(stage1_metadata(), TARGET)[0]
    captured = {}
    compiled = []

    monkeypatch.setattr(aot_moe, "create_compile_context", lambda _target: context)

    def requests_for_job(cfg, received_context):
        captured.update(cfg)
        assert received_context is context
        return (request,)

    monkeypatch.setattr(aot_moe, "build_moe_compile_requests", requests_for_job)
    monkeypatch.setattr(
        aot_moe,
        "compile_aot",
        lambda value, *, context: compiled.append((value, context)) or value,
    )

    artifacts = aot_moe.compile_moe_job(
        {
            "stage": 1,
            "model_dim": 7168,
            "inter_dim": 2048,
            "experts": 256,
            "topk": 8,
            "tile_m": 32,
            "tile_n": 128,
            "tile_k": 256,
            "token_num": 0,
            "cu_num": 256,
            "shared_expert_id": 255,
        }
    )

    assert captured["token_num"] == 32
    assert captured["shared_expert_id"] == 255
    assert compiled == [(request, context)]
    assert artifacts == (request,)


def test_a16_aot_and_runtime_metadata_produce_identical_requests():
    context = create_compile_context(TARGET)
    stage1_config = {
        **stage1_metadata(a_dtype="bf16", b_dtype="fp4"),
        "stage": 1,
        "token_num": 1024,
        "block_m": 32,
    }
    assert aot_moe.build_moe_compile_requests(
        stage1_config, context
    ) == stage1_compile_requests(stage1_config, TARGET)

    stage2_config = {
        **stage2_metadata(a_dtype="bf16", b_dtype="int4", doweight_stage2=True),
        "stage": 2,
        "doweight_stage1": False,
        "token_num": 1024,
        "block_m": 32,
        "mode": "reduce",
        "persist": True,
    }
    aot_requests = aot_moe.build_moe_compile_requests(stage2_config, context)
    runtime_requests = stage2_compile_requests(
        {**stage2_config, "use_global_a": True},
        Stage2RuntimeMetadata(
            **stage2_runtime_metadata(
                mode="reduce",
                accumulate=False,
                persist=True,
                token_num=1024,
                routing_block_count=512,
            )
        ),
        TARGET,
    )
    assert aot_requests == runtime_requests
