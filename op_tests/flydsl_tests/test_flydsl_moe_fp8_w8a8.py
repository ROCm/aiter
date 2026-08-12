# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from pathlib import Path

import pandas as pd
import pytest

from aiter.fused_moe import _flydsl_stage2_wrapper, _stage2_sort_block_size
from aiter.ops.flydsl.moe_kernels import (
    get_flydsl_kernel_params,
    get_flydsl_stage1_kernels_fp8_w8a8,
    get_flydsl_stage2_kernels_fp8_w8a8,
    runtime_swiglu_limit,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "aiter" / "configs" / "model_configs"
TUNED_CONFIG = CONFIG_DIR / "minimax_m3_fp8_w8a8_tuned_fmoe.csv"
UNTUNED_CONFIG = CONFIG_DIR / "minimax_m3_fp8_w8a8_untuned_fmoe.csv"


def test_runtime_swiglu_limit_normalization():
    assert runtime_swiglu_limit(None, "swiglu") == 7.0
    assert runtime_swiglu_limit(0.0, "swiglu") == 7.0
    assert runtime_swiglu_limit(5.5, "swiglu") == 5.5
    assert runtime_swiglu_limit(None, "silu") == float("inf")


def test_fp8_w8a8_registry_is_limited_to_validated_tiles():
    stage1 = get_flydsl_stage1_kernels_fp8_w8a8("bf16")
    stage2 = get_flydsl_stage2_kernels_fp8_w8a8("bf16")

    assert len(stage1) == 9
    assert len(stage2) == 8
    assert all(params["k_batch"] == 1 for params in stage1.values())
    assert all(params["out_dtype"] == "bf16" for params in stage1.values())
    assert all(params["out_dtype"] == "bf16" for params in stage2.values())

    # Stage2 may consume a sorting layout whose block M differs from its GEMM
    # tile M. The suffix is parsed without registering another kernel binary.
    parsed = get_flydsl_kernel_params(
        "flydsl_moe2_afp8_wfp8_w8a8_bf16_" "t32x256x128_atomic_sbm96"
    )
    assert parsed is not None
    assert parsed["tile_m"] == 32
    assert parsed["sort_block_m"] == 96


def test_minimax_config_matches_aiter_token_buckets_and_registry():
    tuned = pd.read_csv(TUNED_CONFIG)
    untuned = pd.read_csv(UNTUNED_CONFIG)

    expected_tokens = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]
    assert tuned["token"].tolist() == expected_tokens
    assert untuned["token"].tolist() == expected_tokens

    assert set(tuned["gfx"]) == {"gfx942"}
    assert set(tuned["cu_num"]) == {304}
    assert set(tuned["model_dim"]) == {6144}
    assert set(tuned["inter_dim"]) == {768}
    assert set(tuned["expert"]) == {129}
    assert set(tuned["topk"]) == {5}
    assert set(tuned["q_type"]) == {"QuantType.per_Token"}
    assert set(tuned["act_type"]) == {"ActivationType.Swiglu"}

    for name in tuned["kernelName1"]:
        assert get_flydsl_kernel_params(name) is not None
    for name in tuned["kernelName2"]:
        assert get_flydsl_kernel_params(name) is not None

    # The measured 32K custom-image configuration requires independent sorting:
    # stage 1 uses tile-M 96 while stage 2 uses tile-M 64. Do not replace stage 2
    # with an unmeasured tile-M 96 kernel merely to share the first sort.
    largest = tuned.iloc[-1]
    assert largest["kernelName1"].endswith("t96x128x128")
    assert largest["kernelName2"].endswith("t64x256x256_atomic")


def test_minimax_config_produces_stage1_and_stage2_aot_jobs():
    from aiter.aot.flydsl.moe import parse_csv

    jobs = parse_csv(str(TUNED_CONFIG))
    assert len(jobs) == 32
    assert {job["stage"] for job in jobs} == {1, 2}
    assert all(job["a_dtype"] == "fp8_w8a8" for job in jobs)
    assert all(job["b_dtype"] == "fp8_w8a8" for job in jobs)
    assert all(job["act"] == "swiglu" for job in jobs)


def test_stage2_uses_independent_sort_for_measured_32k_tile():
    independent = functools.partial(
        _flydsl_stage2_wrapper,
        kernelName="flydsl_moe2_afp8_wfp8_w8a8_bf16_t64x256x256_atomic",
    )
    shared = functools.partial(
        _flydsl_stage2_wrapper,
        kernelName="flydsl_moe2_afp8_wfp8_w8a8_bf16_" "t32x256x128_atomic_sbm96",
    )

    assert _stage2_sort_block_size(independent, stage1_block_size=96) == 64
    assert _stage2_sort_block_size(shared, stage1_block_size=96) == 96


@pytest.mark.parametrize("act", ["silu", "swiglu"])
def test_fp8_w8a8_compile_contract_rejects_split_k(act):
    from aiter.ops.flydsl.moe_kernels import compile_flydsl_moe_stage1

    with pytest.raises(ValueError, match="does not support split-K"):
        compile_flydsl_moe_stage1(
            model_dim=6144,
            inter_dim=768,
            experts=129,
            topk=5,
            tile_m=16,
            tile_n=64,
            tile_k=512,
            doweight_stage1=False,
            a_dtype="fp8_w8a8",
            b_dtype="fp8_w8a8",
            out_dtype="bf16",
            act=act,
            k_batch=2,
        )


def test_fp8_w8a8_compile_contract_rejects_bias_and_padding():
    from aiter.ops.flydsl.moe_kernels import (
        compile_flydsl_moe_stage1,
        compile_flydsl_moe_stage2,
    )

    common = {
        "model_dim": 6144,
        "inter_dim": 768,
        "experts": 129,
        "topk": 5,
        "tile_m": 16,
        "tile_n": 128,
        "tile_k": 256,
        "a_dtype": "fp8_w8a8",
        "b_dtype": "fp8_w8a8",
        "out_dtype": "bf16",
    }
    with pytest.raises(ValueError, match="does not support bias"):
        compile_flydsl_moe_stage1(
            **common,
            doweight_stage1=False,
            act="swiglu",
            enable_bias=True,
        )
    with pytest.raises(ValueError, match="does not support padded dimensions"):
        compile_flydsl_moe_stage2(
            **common,
            doweight_stage2=True,
            accumulate=True,
            inter_dim_pad=128,
        )


def test_stage2_rejects_incompatible_sort_block_size():
    from aiter.ops.flydsl.kernels.moe_gemm_2stage import compile_moe_gemm2

    with pytest.raises(ValueError, match="must be a multiple of tile_m"):
        compile_moe_gemm2(
            model_dim=6144,
            inter_dim=768,
            experts=129,
            topk=5,
            tile_m=64,
            tile_n=256,
            tile_k=256,
            doweight_stage2=True,
            in_dtype="fp8",
            out_dtype="bf16",
            accumulate=True,
            sort_block_m=96,
        )
