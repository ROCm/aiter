# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure CPU contract tests for shared MoE compile-request factories."""

from dataclasses import replace

import pytest

from aiter.aot.flydsl.tests.moe_test_utils import (
    TARGET,
    request_argument_names,
    stage1_metadata,
    stage2_metadata,
    stage2_runtime_metadata,
)
from aiter.ops.flydsl.moe_compile_decisions import resolve_stage2_compile_decision
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
    cktile_epilogue_compile_requests,
    stage1_compile_requests,
    stage2_compile_requests,
)


def test_factories_are_pure_deterministic_and_runtime_metadata_is_not_bound():
    first = stage2_compile_requests(
        stage2_metadata(), stage2_runtime_metadata(), TARGET
    )
    second = stage2_compile_requests(
        stage2_metadata(), stage2_runtime_metadata(), TARGET
    )

    assert first == second
    assert len(first) == 1
    assert first[0].op_id == MIXED_STAGE2_GEMM_OP_ID
    assert "token_num" not in first[0].as_kwargs()
    assert "routing_block_count" not in first[0].as_kwargs()
    assert first[0].as_kwargs()["persist_m"] == 1


def test_stage2_runtime_metadata_and_injected_decision_fail_fast():
    with pytest.raises(ValueError, match="token_num"):
        Stage2RuntimeMetadata(**stage2_runtime_metadata(token_num=0))
    with pytest.raises(TypeError, match="use_weight"):
        Stage2RuntimeMetadata(**stage2_runtime_metadata(use_weight=1))

    metadata = stage2_metadata()
    runtime = Stage2RuntimeMetadata(**stage2_runtime_metadata())
    decision = resolve_stage2_compile_decision(
        metadata,
        mode=runtime.mode,
        accumulate=runtime.accumulate,
        return_per_slot=runtime.return_per_slot,
        persist=runtime.persist,
        token_num=runtime.token_num,
        routing_block_count=runtime.routing_block_count,
        dtype_str=runtime.dtype_str,
        use_mask=runtime.use_mask,
        topk_ids_available=runtime.topk_ids_available,
        num_experts=runtime.num_experts,
        fp8_intermediate=runtime.fp8_intermediate,
    )
    with pytest.raises(ValueError, match="decision disagrees"):
        stage2_compile_requests(
            metadata,
            runtime,
            TARGET,
            decision=replace(decision, persist_m=decision.persist_m + 1),
        )


@pytest.mark.parametrize(
    ("metadata", "op_id"),
    [
        (stage1_metadata(), MIXED_STAGE1_GEMM_OP_ID),
        (
            stage1_metadata(a_dtype="bf16", b_dtype="fp4"),
            A16_STAGE1_GEMM_OP_ID,
        ),
        (
            stage1_metadata(a_dtype="bf16", b_dtype="int4"),
            INT4_STAGE1_GEMM_OP_ID,
        ),
        (stage1_metadata(shared_expert_id=255), FHMOE_STAGE1_GEMM_OP_ID),
    ],
)
def test_stage1_selects_the_runtime_builder_family(metadata, op_id):
    assert stage1_compile_requests(metadata, TARGET)[0].op_id == op_id


@pytest.mark.parametrize(
    ("metadata", "op_id"),
    [
        (stage2_metadata(), MIXED_STAGE2_GEMM_OP_ID),
        (
            stage2_metadata(a_dtype="bf16", b_dtype="fp4"),
            A16_STAGE2_GEMM_OP_ID,
        ),
        (
            stage2_metadata(a_dtype="bf16", b_dtype="int4"),
            INT4_STAGE2_GEMM_OP_ID,
        ),
        (stage2_metadata(shared_expert_id=255), FHMOE_STAGE2_GEMM_OP_ID),
    ],
)
def test_stage2_selects_the_runtime_builder_family(metadata, op_id):
    request = stage2_compile_requests(metadata, stage2_runtime_metadata(), TARGET)[0]
    assert request.op_id == op_id


def test_a16_requests_bind_only_effective_family_compile_parameters():
    stage1 = stage1_compile_requests(
        stage1_metadata(a_dtype="bf16", b_dtype="fp4"), TARGET
    )[0]
    assert "persist_m" not in stage1.as_kwargs()
    assert "doweight_stage1" not in stage1.as_kwargs()
    assert "enable_bias" not in stage1.as_kwargs()

    stage2 = stage2_compile_requests(
        stage2_metadata(a_dtype="bf16", b_dtype="int4", use_global_a=False),
        stage2_runtime_metadata(mode="reduce", accumulate=False, persist=True),
        TARGET,
    )[0]
    assert stage2.as_kwargs()["persist_m"] == -1
    assert stage2.as_kwargs()["mode"] == "reduce"
    assert "use_global_a" not in stage2.as_kwargs()
    assert "accumulate" not in stage2.as_kwargs()
    assert "doweight_stage2" not in stage2.as_kwargs()


def test_a16_stage1_normalizes_the_runtime_situ_alias():
    alias = stage1_compile_requests(
        stage1_metadata(a_dtype="bf16", b_dtype="fp4", act="situ"), TARGET
    )
    canonical = stage1_compile_requests(
        stage1_metadata(a_dtype="bf16", b_dtype="fp4", act="situv2"), TARGET
    )

    assert alias == canonical
    assert alias[0].as_kwargs()["act"] == "situv2"


@pytest.mark.parametrize(
    ("stage", "metadata", "runtime", "message"),
    [
        (
            1,
            stage1_metadata(a_dtype="bf16", b_dtype="fp4", enable_bias=True),
            None,
            "bias",
        ),
        (
            1,
            stage1_metadata(a_dtype="bf16", b_dtype="int4", k_batch=2),
            None,
            "split-K",
        ),
        (
            1,
            stage1_metadata(a_dtype="bf16", b_dtype="fp4", act="gelu"),
            None,
            "activation",
        ),
        (
            2,
            stage2_metadata(a_dtype="bf16", b_dtype="fp4"),
            stage2_runtime_metadata(mode="reduce", accumulate=False),
            "atomic mode",
        ),
        (
            2,
            stage2_metadata(a_dtype="bf16", b_dtype="int4", enable_bias=True),
            stage2_runtime_metadata(),
            "bias",
        ),
    ],
)
def test_a16_unsupported_modes_fail_in_the_shared_factory(
    stage, metadata, runtime, message
):
    with pytest.raises(ValueError, match=message):
        if stage == 1:
            stage1_compile_requests(metadata, TARGET)
        else:
            stage2_compile_requests(metadata, runtime, TARGET)


def test_stage1_fq_requests_track_current_abis():
    requests = stage1_compile_requests(
        stage1_metadata(out_dtype="fp8", k_batch=4, gate_mode="interleave"),
        TARGET,
    )

    assert [request.op_id for request in requests] == [
        MIXED_STAGE1_GEMM_OP_ID,
        FQ_ACTIVATION_OP_ID,
    ]
    assert request_argument_names(requests[0])[-6:] == (
        "f32_situ_beta",
        "f32_situ_beta_rcp",
        "f32_situ_linear_beta",
        "f32_situ_linear_beta_rcp",
        "f32_swiglu_limit",
        "stream",
    )
    assert request_argument_names(requests[1])[-6:] == (
        "situ_beta_f",
        "situ_beta_rcp_f",
        "situ_linear_beta_f",
        "situ_linear_beta_rcp_f",
        "swiglu_limit_f",
        "stream",
    )


def test_reduction_request_tracks_current_fp8_and_weighted_abi():
    requests = stage2_compile_requests(
        stage2_metadata(),
        stage2_runtime_metadata(
            mode="reduce",
            accumulate=False,
            dtype_str="fp8",
            fp8_intermediate=True,
            out_dtype_str="bf16",
            use_weight=True,
            scale_blk=8,
            pitch_align=0,
        ),
        TARGET,
    )

    assert [request.op_id for request in requests] == [
        MIXED_STAGE2_GEMM_OP_ID,
        PLAIN_REDUCTION_OP_ID,
    ]
    reduction = requests[1]
    assert request_argument_names(reduction) == (
        "X",
        "Y",
        "expert_mask",
        "topk_ids",
        "topk_weights",
        "i32_m_tokens",
        "stream",
    )
    assert reduction.as_kwargs() == {
        "topk": 8,
        "model_dim": 7168,
        "dtype_str": "fp8",
        "use_mask": False,
        "num_experts": 0,
        "out_dtype_str": "bf16",
        "use_weight": True,
        "scale_blk": 8,
        "pitch_align": 0,
    }


def test_cktile_epilogue_request_uses_the_runtime_pointer_abi():
    requests = cktile_epilogue_compile_requests(
        {
            "act": "swiglu",
            "inter_dim": 2048,
            "topk": 8,
            "split_k": 2,
            "post_activation_layout": "interleaved",
            "enable_bias": False,
        },
        TARGET,
    )

    assert requests[0].op_id == CKTILE_SWIGLU_AND_MUL_OP_ID
    assert request_argument_names(requests[0]) == (
        "x",
        "out",
        "num_rows",
        "stream",
    )


def test_invalid_runtime_compile_metadata_fails_early():
    with pytest.raises(ValueError, match="accumulate disagrees"):
        stage2_compile_requests(
            stage2_metadata(),
            stage2_runtime_metadata(mode="reduce", accumulate=True),
            TARGET,
        )
