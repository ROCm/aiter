# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest

from aiter.aot.flydsl.mxfp4_moe import _compile_v2_stage2, parse_csv
from aiter.fused_moe import (
    AUX_SORT_OPUS,
    AUX_SORT_THREESTAGE,
    _validate_output_aux,
    fused_moe_2stages,
    get_inter_dim,
)
from aiter.ops.flydsl.mxfp4_gemm1_kernels import _assert_supported
from aiter.ops.flydsl.mxfp4_kname import native_scale_layout_for


def test_native_scale_layout_depends_on_output_dtype():
    assert native_scale_layout_for(16, "fp4")
    assert not native_scale_layout_for(32, "fp4")
    assert not native_scale_layout_for(16, "fp8")

    _assert_supported(
        NE=896,
        D_HIDDEN=3584,
        D_INTER=512,
        topk=16,
        BM=16,
        use_nt=True,
        inline_quant=True,
        BN=256,
        BK=256,
        a_dtype="fp8",
        out_dtype="fp8",
        native_scale_layout=native_scale_layout_for(16, "fp8"),
    )


def test_kimik3_aot_stage1_matches_runtime_inter_dim():
    root = Path(__file__).resolve().parents[2]
    jobs = parse_csv(
        str(root / "aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv")
    )
    stage1_jobs = [job for job in jobs if job["stage"] == 1]

    assert stage1_jobs
    assert {job["D_INTER"] for job in stage1_jobs} == {384}
    assert get_inter_dim((896, 768, 1792), (896, 3584, 192)) == (896, 3584, 384)


def test_layout_reduce_aot_does_not_require_removed_padding_fields(monkeypatch):
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher

    root = Path(__file__).resolve().parents[2]
    jobs = parse_csv(
        str(root / "aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv")
    )
    job = next(
        job for job in jobs if job.get("v2_stage2") and job["epilog"] == "reduce"
    )
    calls = []

    monkeypatch.setattr(
        mxmoe_dispatcher, "mxfp4_moe_gemm2", lambda **kwargs: calls.append("gemm2")
    )
    monkeypatch.setattr(
        moe_kernels,
        "_run_moe_reduction",
        lambda *args, **kwargs: calls.append("reduce"),
    )

    assert "model_dim_pad" not in job
    _compile_v2_stage2(job)
    assert calls == ["gemm2", "reduce"]


@pytest.mark.parametrize(
    "output_aux", [False, None, AUX_SORT_THREESTAGE, AUX_SORT_OPUS]
)
def test_validate_output_aux_accepts_supported_values(output_aux):
    _validate_output_aux(output_aux)


def test_fused_moe_2stages_rejects_unknown_output_aux_before_launch():
    with pytest.raises(ValueError, match="unknown output_aux"):
        fused_moe_2stages(*(None,) * 11, output_aux="opuss")
