# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate the same coupled MXMOE candidate boundary used by the tuner."""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import os
from typing import Any

import pandas as pd
import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _mxfp4_a4w4_stage1_fw,
    _mxfp4_a4w4_stage2_fw,
    moe_sorting,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
)
from aiter.ops.flydsl.mxfp4_kname import native_scale_layout_for
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from aiter.utility.dtypes import str2Dtype
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
    FmoeTuner,
    Mxfp4FlydslTuner,
    cosine_diff_compare,
)

SUPPORTED_GFX = ("gfx950",)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX,
    reason="Coupled MXMOE requires gfx950",
)


def run_torch(
    data: dict[str, Any],
    topk: int,
    dtype: torch.dtype,
    activation: ActivationType,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    return Mxfp4FlydslTuner._torch_ref(
        data, topk, dtype, activation, swiglu_limit=swiglu_limit
    )


def check_mxfp8_input(
    data: dict[str, Any],
    token: int,
    model_dim: int,
    expert: int,
    topk: int,
    block_m: int,
    dtype: torch.dtype,
) -> None:
    """Check payload and valid sorted scales against the public torch quantizer."""
    ref_q, ref_s = per_1x32_f8_scale_f8_quant(
        data["input"], scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    q = data["a1_qt"]
    scale = data["a1_scale"]
    assert q.dtype == dtypes.fp8
    assert scale.dtype == dtypes.fp8_e8m0
    assert q.shape == (token, model_dim)
    assert torch.equal(ref_q.view(torch.uint8), q.view(torch.uint8))
    nonzero_group = data["input"].reshape(token, -1, 32).abs().amax(-1) > 0
    # HIP floors amax at 1e-10; the torch helper keeps exponent 0 for zero
    # groups. Require exact positive-group scales and joint decoded values.
    assert torch.equal(
        ref_s.view(torch.uint8)[nonzero_group],
        scale.view(torch.uint8)[nonzero_group],
    )
    scale_f32 = fp4_utils.e8m0_to_f32(scale)
    assert torch.isfinite(scale_f32).all()
    assert (scale_f32 > 0).all()
    reconstructed = q.float() * scale_f32.repeat_interleave(32, 1)
    ref_reconstructed = ref_q.float() * fp4_utils.e8m0_to_f32(ref_s).repeat_interleave(
        32, 1
    )
    assert torch.equal(reconstructed, ref_reconstructed)
    assert torch.isfinite(reconstructed).all()
    assert torch.equal(
        reconstructed.reshape(token, -1, 32)[~nonzero_group],
        torch.zeros_like(reconstructed.reshape(token, -1, 32)[~nonzero_group]),
    )
    error = checkAllclose(
        data["input"].float(),
        reconstructed,
        # E4M3 rounds by up to half one 3-bit mantissa step (1/16).
        rtol=0.0625,
        atol=0.01,
        tol_err_ratio=0,
        msg="MXFP8 A payload + E8M0 scales",
    )
    assert error == 0
    input_error = cosine_diff_compare(
        data["input"], reconstructed, msg="MXFP8 input quantization"
    )
    assert math.isfinite(input_error) and input_error < 0.005
    assert torch.unique(scale.view(torch.uint8)).numel() > 1

    sti, sw, _, nvi, _, _, _ = moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        expert,
        model_dim,
        dtype,
        block_size=block_m,
        accumulate=False,
        output_aux="opus",
    )
    sorted_q, sorted_scale = aiter.fused_dynamic_mxfp8_quant_moe_sort(
        input=data["input"],
        sorted_ids=sti,
        num_valid_ids=nvi,
        token_num=token,
        topk=topk,
        block_size=block_m,
        sorted_weights=sw,
        num_experts_upper_bound=expert,
    )
    assert torch.equal(q.view(torch.uint8), sorted_q.view(torch.uint8))
    # Match only active scale addresses. Padding bytes have no consumer contract.
    valid_count = int(nvi[0].item())
    row = torch.arange(valid_count, device=sti.device)
    group = torch.arange(model_dim // 32, device=sti.device)
    token_id = sti[:valid_count].to(torch.int64) & 0xFFFFFF
    valid = token_id < token
    r = row[valid, None]
    g = group[None, :]
    cols_pad = sorted_scale.shape[1]
    offset = (
        (r // 32) * 32 * cols_pad
        + (g // 8) * 256
        + (g % 4) * 64
        + (r % 16) * 4
        + ((g % 8) // 4) * 2
        + (r % 32) // 16
    )
    actual = sorted_scale.view(torch.uint8).flatten()[offset]
    expected = scale.view(torch.uint8)[token_id[valid, None], g]
    assert torch.equal(actual, expected)


@benchmark()
def test_coupled_mxmoe(
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    dtype: torch.dtype,
    precision: str,
    activation: str,
    block_m: int,
    tile_k: int,
) -> dict[str, Any]:
    a_dtype = "fp8" if precision == "A8W4" else "fp4"
    data = Mxfp4FlydslTuner._prepare_case(
        token, model_dim, inter_dim, expert, topk, dtype, a_dtype=a_dtype
    )
    if precision == "A8W4":
        # Different group amplitudes and zero groups expose scale omissions;
        # regenerating the quantized reference is outside candidate timing.
        amplitude = torch.ones_like(data["input"])
        amplitude[: token // 2, :32] = 0
        amplitude[token // 2 :, 32:64] = 16
        data["input"].mul_(amplitude)
        from aiter.ops.quant import per_1x32_mx_quant_hip

        data["a1_qt"], data["a1_scale"] = per_1x32_mx_quant_hip(
            data["input"],
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )
        check_mxfp8_input(data, token, model_dim, expert, topk, block_m, dtype)

    act = {"Silu": "silu", "Situv2": "situv2", "Swiglu": "swiglu"}[activation]
    act_type = getattr(ActivationType, activation)
    limit_env = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT")
    limit = None if limit_env in (None, "") else float(limit_env)
    reference = run_torch(data, topk, dtype, act_type, swiglu_limit=limit)
    g1 = Mxfp4FlydslTuner._g1_kname(
        block_m,
        block_m == 16,
        block_m == 16,
        act=act,
        a_dtype=a_dtype,
        out_dtype=a_dtype,
    )
    modes = ("atomic", "reduce", "scatter") if block_m == 128 else ("atomic", "reduce")
    candidates = {}
    for mode in modes:
        g2 = (
            f"flydsl_moe2_layout_a{a_dtype}_wfp4_bf16_"
            f"t{block_m}x128x{tile_k}_{mode}_sbm{block_m}"
        )
        candidates[mode] = functools.partial(
            Mxfp4FlydslTuner._port_e2e,
            data,
            g1,
            g2,
            topk,
            expert,
            model_dim,
            dtype,
            swiglu_limit=limit,
        )
    flops = 6 * token * topk * model_dim * inter_dim
    nbytes = (
        sum(
            data[key].numel() * data[key].element_size()
            for key in (
                "input",
                "w1_a16",
                "w1s_a16",
                "w2_a16",
                "w2s_a16",
            )
        )
        + token * model_dim * torch.empty((), dtype=dtype).element_size()
    )
    ret = {"gfx": get_gfx(), "GEMM1": g1}
    for name, candidate in candidates.items():
        output, us = run_perftest(candidate, num_warmup=2, num_iters=5)
        assert math.isfinite(us) and us > 0
        assert torch.isfinite(output).all()
        error = cosine_diff_compare(reference, output, msg=f"{precision} {name}")
        assert math.isfinite(error) and error < 0.1
        # Also retain a position-sensitive check against the unquantized
        # intermediate reference; the tuner gate remains the amplitude metric.
        checkAllclose(
            reference.float(),
            output.float(),
            rtol=0.2,
            atol=0.2,
            msg=f"{precision} {name}",
            catastrophic_check=True,
        )
        first = output.clone()
        repeated = candidate()
        assert torch.isfinite(repeated).all()
        repeat_metric = cosine_diff_compare(
            first, repeated, msg=f"{precision} {name} repeat"
        )
        assert math.isfinite(repeat_metric) and repeat_metric < 1e-5
        repeat_error = checkAllclose(
            first.float(),
            repeated.float(),
            rtol=0.02,
            # BF16 atomic sums can round differently as expert writes arrive;
            # reduce/scatter sum routes without that atomic ordering variation.
            atol=0.0625 if name == "atomic" else 0.02,
            msg=f"{precision} {name} repeat",
            catastrophic_check=True,
        )
        # Atomic BF16 sums vary with expert arrival order, especially near
        # cancellation. Their repeat gate is the amplitude metric above plus
        # finite/catastrophic checks; deterministic reduce/scatter stay strict.
        if name != "atomic":
            assert repeat_error == 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = error
    return ret


test_coupled_mxmoe.__test__ = False  # The CLI owns the perf sweep axes.


def decode_bm16_intermediate(
    payload: torch.Tensor, scale: torch.Tensor, inter_dim: int, *, native: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the valid E8M0 addresses consumed by the SBM16 GEMM2 reader."""
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for

    rows = torch.arange(payload.shape[0], device=payload.device).reshape(-1, 1)
    groups = torch.arange(inter_dim // 32, device=payload.device).reshape(1, -1)
    chunk_bytes = kas_per_chunk_dw_for(inter_dim) * 4
    offsets = (
        (rows // (16 if native else 32)) * chunk_bytes
        + (groups // 8) * 256
        + (groups % 4) * 64
        + (rows % 16) * 4
        + ((groups // 4) % 2) * 2
    )
    if not native:
        offsets += (rows // 16) % 2
    valid_scale = scale.view(torch.uint8).flatten()[offsets].view(dtypes.fp8_e8m0)
    values = (
        payload.float()
        if payload.dtype == dtypes.fp8
        else fp4_utils.mxfp4_to_f32(payload)
    )
    decoded = values * fp4_utils.e8m0_to_f32(valid_scale).repeat_interleave(32, 1)
    return decoded, valid_scale


@benchmark()
def test_runtime_bias_stage1(
    precision: str, activation: str, block_m: int
) -> dict[str, Any]:
    """Zero A isolates expert bias, including cache switches on one CSV name."""
    token, model_dim, inter_dim, expert, topk = 33, 512, 256, 2, 2
    a_dtype = "fp8" if precision == "A8W4" else "fp4"
    data = Mxfp4FlydslTuner._prepare_case(
        token, model_dim, inter_dim, expert, topk, dtypes.bf16, a_dtype=a_dtype
    )
    data["input"].zero_()
    quant = per_1x32_f8_scale_f8_quant if a_dtype == "fp8" else per_1x32_f4_quant
    quant_kwargs = {"scale_type": dtypes.fp8_e8m0} if a_dtype == "fp8" else {}
    data["a1_qt"], data["a1_scale"] = quant(data["input"], **quant_kwargs)
    bias = torch.empty((expert, 2 * inter_dim), dtype=torch.float32)
    bias[:, :inter_dim] = torch.linspace(0.5, 1.5, inter_dim)
    bias[:, inter_dim:] = torch.linspace(-2.0, 2.0, inter_dim)
    bias[1] *= 2
    data["topk_ids"] = (
        torch.arange(expert, dtype=dtypes.i32).expand(token, topk).contiguous()
    )
    data["topk_weights"] = torch.full((token, topk), 0.5, dtype=dtypes.fp32)
    act = {"Silu": "silu", "Situv2": "situv2", "Swiglu": "swiglu"}[activation]
    inline = block_m == 16
    g1 = Mxfp4FlydslTuner._g1_kname(
        block_m, inline, inline, bn=128, act=act, a_dtype=a_dtype, out_dtype=a_dtype
    )
    assert "_bias" not in g1
    sti, sw, sei, nvi, moe_buf, indices, reverse = moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        expert,
        model_dim,
        dtypes.bf16,
        block_size=block_m,
        accumulate=False,
        output_aux="opus",
    )
    stage1_input, stage1_scale = data["input"], None
    if not inline:
        prequant = (
            aiter.fused_dynamic_mxfp8_quant_moe_sort
            if a_dtype == "fp8"
            else aiter.fused_dynamic_mxfp4_quant_moe_sort
        )
        stage1_input, stage1_scale = prequant(
            input=data["input"],
            sorted_ids=sti,
            num_valid_ids=nvi,
            token_num=token,
            topk=topk,
            block_size=block_m,
            sorted_weights=sw,
            num_experts_upper_bound=expert,
        )
    stage1 = functools.partial(
        _mxfp4_a4w4_stage1_fw,
        stage1_input,
        data["w1_a16"],
        data["w2_a16"],
        sti,
        sei,
        nvi,
        None,
        topk,
        block_m=block_m,
        a1_scale=stage1_scale,
        w1_scale=data["w1s_a16"],
        kernelName1=g1,
        m_indices=indices,
        interleave=a_dtype == "fp8",
        moe_buf=moe_buf,
        situ_beta=DEFAULT_SITUV2_BETA if act == "situv2" else 1.0,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA if act == "situv2" else 1.0,
    )
    references = {}
    for enabled, runtime_bias in ((True, bias), (False, None)):
        ref = FmoeTuner.run_torch_moe_stage1(
            data["a1_qt"],
            data["w1_qt"],
            data["w2_qt"],
            data["topk_weights"],
            data["topk_ids"],
            data["a1_scale"],
            data["w1_scale"],
            w1_bias=runtime_bias,
            # A4W4 quantizes FP32 activation directly; A8W4 materializes
            # BF16 before MXFP8 quantization.
            dtype=dtypes.bf16 if a_dtype == "fp8" else dtypes.fp32,
            activation=getattr(ActivationType, activation),
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            topk=topk,
        )
        ref_q, ref_s = quant(
            (ref if a_dtype == "fp8" else ref.float()).reshape(-1, inter_dim),
            **quant_kwargs,
        )
        ref_values = (
            ref_q.float() if a_dtype == "fp8" else fp4_utils.mxfp4_to_f32(ref_q)
        )
        references[enabled] = ref_values * fp4_utils.e8m0_to_f32(
            ref_s
        ).repeat_interleave(32, 1)
    snapshots = []
    for runtime_bias in (bias, None, bias):
        payload, scales = stage1(bias1=runtime_bias)
        decoded, _ = decode_bm16_intermediate(
            payload, scales, inter_dim, native=block_m == 16
        )
        selected = decoded[reverse.long()]
        assert torch.isfinite(selected).all()
        error = checkAllclose(
            references[runtime_bias is not None],
            selected,
            rtol=0.01,
            atol=0.01,
            tol_err_ratio=0,
            msg=f"{precision} {activation} BM{block_m} runtime bias",
        )
        assert error == 0
        snapshots.append(selected.clone())
    assert torch.equal(snapshots[0], snapshots[2])
    assert torch.count_nonzero(snapshots[1]) == 0
    assert torch.count_nonzero(snapshots[0]) > 0
    flops = 4 * token * topk * model_dim * inter_dim
    nbytes = sum(
        data[key].numel() * data[key].element_size()
        for key in ("input", "w1_a16", "w1s_a16")
    )
    ret = {"gfx": get_gfx(), "GEMM1": g1, "bias switch err": 0}
    candidates = {
        "bias": functools.partial(stage1, bias1=bias),
        "no_bias": functools.partial(stage1, bias1=None),
    }
    for name, candidate in candidates.items():
        (payload, scales), us = run_perftest(candidate, num_warmup=2, num_iters=5)
        decoded, _ = decode_bm16_intermediate(
            payload, scales, inter_dim, native=block_m == 16
        )
        error = checkAllclose(
            references[name == "bias"],
            decoded[reverse.long()],
            rtol=0.01,
            atol=0.01,
            tol_err_ratio=0,
            msg=f"{name} timed output",
        )
        assert error == 0 and math.isfinite(us) and us > 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = (
            (nbytes + (bias.numel() * bias.element_size() if name == "bias" else 0))
            / us
            / 1e6
        )
        ret[f"{name} err"] = error
    return ret


test_runtime_bias_stage1.__test__ = False


@pytest.mark.parametrize("precision", ["A4W4", "A8W4"])
@pytest.mark.parametrize("activation", ["Silu", "Situv2", "Swiglu"])
@pytest.mark.parametrize("block_m", [16, 32])
def test_mxmoe_runtime_bias_switch(
    precision: str, activation: str, block_m: int
) -> None:
    with torch.device("cuda"):
        result = test_runtime_bias_stage1(precision, activation, block_m)
    aiter.logger.info(
        "MXMOE runtime bias summary (markdown):\n%s",
        pd.DataFrame([result]).to_markdown(index=False),
    )


@benchmark()
def test_bm16_scale_pipeline(
    token: int,
    model_dim: int,
    inter_dim: int,
    dtype: torch.dtype,
    precision: str,
    activation: str,
    tile_k: int,
) -> dict[str, Any]:
    """Multiple expert blocks with different magnitudes and a partial tail."""
    expert, topk, bm = 2, 2, 16
    a_dtype = "fp8" if precision == "A8W4" else "fp4"
    data = Mxfp4FlydslTuner._prepare_case(
        token, model_dim, inter_dim, expert, topk, dtype, a_dtype=a_dtype
    )
    data["input"][16:] *= 16
    # Guarantee a zero intermediate group without making its neighbors zero.
    zero_columns = torch.cat([torch.arange(32), inter_dim + torch.arange(32)])
    data["w1_qt"].view(torch.uint8)[:, zero_columns] = 0
    data["w1_a16"] = shuffle_weight(
        data["w1_qt"], (16, 16), is_guinterleave=a_dtype == "fp8", gate_up=True
    )
    quant = per_1x32_f8_scale_f8_quant if a_dtype == "fp8" else per_1x32_f4_quant
    quant_kwargs = {"scale_type": dtypes.fp8_e8m0} if a_dtype == "fp8" else {}
    data["a1_qt"], data["a1_scale"] = quant(data["input"], **quant_kwargs)
    data["topk_ids"] = (
        torch.arange(expert, dtype=dtypes.i32).expand(token, topk).contiguous()
    )
    data["topk_weights"] = (
        torch.tensor([0.25, 0.75], dtype=dtypes.fp32).expand(token, topk).contiguous()
    )

    # Exact expert-major metadata isolates the producer/reader contract. The
    # separate coupled candidate sweep above times the actual production sort.
    padded_per_expert = (token + bm - 1) // bm * bm
    max_sorted = expert * padded_per_expert
    sorted_ids = torch.full((expert, padded_per_expert), token, dtype=dtypes.i32)
    token_ids = torch.arange(token, dtype=dtypes.i32)
    sorted_ids[:, :token] = token_ids[None, :] | (
        torch.arange(topk, dtype=dtypes.i32)[:, None] << 24
    )
    sorted_ids = sorted_ids.flatten()
    m_indices = sorted_ids & 0xFFFFFF
    sorted_experts = torch.arange(expert, dtype=dtypes.i32).repeat_interleave(
        padded_per_expert // bm
    )
    num_valid_ids = torch.tensor([max_sorted, token], dtype=dtypes.i32)
    sorted_weights = torch.zeros(expert, padded_per_expert, dtype=dtypes.fp32)
    sorted_weights[:, :token] = data["topk_weights"].T
    sorted_weights = sorted_weights.flatten()
    reverse_sorted = (
        token_ids[:, None]
        + torch.arange(topk, dtype=dtypes.i32)[None, :] * padded_per_expert
    ).flatten()
    act_type = getattr(ActivationType, activation)
    act = {"Silu": "silu", "Situv2": "situv2", "Swiglu": "swiglu"}[activation]
    limit_env = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT")
    limit = None if limit_env in (None, "") else float(limit_env)
    ref1 = FmoeTuner.run_torch_moe_stage1(
        data["a1_qt"],
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        data["a1_scale"],
        data["w1_scale"],
        dtype=dtype,
        activation=act_type,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        topk=topk,
        swiglu_limit=limit,
    )
    reference_quant_input = ref1 if a_dtype == "fp8" else ref1.float()
    ref_q, ref_s = quant(reference_quant_input.reshape(-1, inter_dim), **quant_kwargs)
    ref_values = ref_q.float() if a_dtype == "fp8" else fp4_utils.mxfp4_to_f32(ref_q)
    ref_inter = ref_values * fp4_utils.e8m0_to_f32(ref_s).repeat_interleave(32, 1)
    ref_out = FmoeTuner.run_torch_moe_stage2(
        ref_q.reshape(token, topk, -1),
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        ref_s,
        data["w2_scale"],
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
    )
    g1 = Mxfp4FlydslTuner._g1_kname(
        bm, True, True, act=act, bn=128, a_dtype=a_dtype, out_dtype=a_dtype
    )
    stage1 = functools.partial(
        _mxfp4_a4w4_stage1_fw,
        data["input"],
        data["w1_a16"],
        data["w2_a16"],
        sorted_ids,
        sorted_experts,
        num_valid_ids,
        None,
        topk,
        block_m=bm,
        w1_scale=data["w1s_a16"],
        kernelName1=g1,
        m_indices=m_indices,
        interleave=a_dtype == "fp8",
        situ_beta=DEFAULT_SITUV2_BETA if act == "situv2" else 1.0,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA if act == "situv2" else 1.0,
        swiglu_limit=limit,
    )
    payload, scale = stage1()
    decoded, reader_scale = decode_bm16_intermediate(
        payload, scale, inter_dim, native=True
    )
    selected = decoded[reverse_sorted.long()]
    assert torch.isfinite(selected).all()
    assert torch.isfinite(
        fp4_utils.e8m0_to_f32(reader_scale)[reverse_sorted.long()]
    ).all()
    assert torch.equal(selected[:, :32], torch.zeros_like(selected[:, :32]))
    intermediate_error = cosine_diff_compare(
        ref_inter, selected, msg="BM16 native reader"
    )
    assert math.isfinite(intermediate_error) and intermediate_error < 0.02
    per_block_error = []
    for start in range(0, token, bm):
        end = min(start + bm, token)
        err = cosine_diff_compare(
            ref_inter.reshape(token, topk, inter_dim)[start:end],
            selected.reshape(token, topk, inter_dim)[start:end],
            printLog=False,
        )
        assert math.isfinite(err) and err < 0.02
        per_block_error.append(err)
    # Effective scales must actually differ between the two complete blocks.
    scale_bytes = reader_scale.view(torch.uint8)
    assert not torch.equal(scale_bytes[:16, 1:], scale_bytes[16:32, 1:])

    # Direct callers may still ask for the regular producer layout. Check its
    # active decoded values; do not feed it to an SBM16 native reader.
    regular_q, regular_s = stage1(native_scale_layout=False)
    regular_decoded, _ = decode_bm16_intermediate(
        regular_q, regular_s, inter_dim, native=False
    )
    assert torch.equal(regular_decoded[reverse_sorted.long()], selected)
    assert native_scale_layout_for(bm, a_dtype)

    candidates = {}
    outputs = {}
    for mode in ("atomic", "reduce"):
        g2 = f"flydsl_moe2_layout_a{a_dtype}_wfp4_bf16_t16x128x{tile_k}_{mode}_sbm16"
        output = torch.full((token, model_dim), float("nan"), dtype=dtype)

        def pipeline(
            g2: str = g2, output: torch.Tensor = output, mode: str = mode
        ) -> torch.Tensor:
            # Atomic sort clears this buffer in the real coupled pipeline.
            if mode == "atomic":
                output.zero_()
            q, s = stage1()
            return _mxfp4_a4w4_stage2_fw(
                q,
                data["w1_a16"],
                data["w2_a16"],
                sorted_ids,
                sorted_experts,
                num_valid_ids,
                output,
                topk,
                block_m=bm,
                w2_scale=data["w2s_a16"],
                a2_scale=s,
                sorted_weights=sorted_weights,
                kernelName2=g2,
                reverse_sorted=reverse_sorted,
            )

        candidates[mode] = pipeline
        outputs[mode] = output
    flops = 6 * token * topk * model_dim * inter_dim
    nbytes = (
        sum(
            data[key].numel() * data[key].element_size()
            for key in (
                "input",
                "w1_a16",
                "w1s_a16",
                "w2_a16",
                "w2s_a16",
            )
        )
        + token * model_dim * torch.empty((), dtype=dtype).element_size()
    )
    ret = {
        "gfx": get_gfx(),
        "active_blocks": expert * ((token + bm - 1) // bm),
        "intermediate err": intermediate_error,
        "block errors": str(per_block_error),
    }
    for name, candidate in candidates.items():
        outputs[name].fill_(float("nan"))
        assert torch.isfinite(candidate()).all()
        result, us = run_perftest(candidate, num_warmup=2, num_iters=5)
        assert torch.isfinite(result).all()
        assert math.isfinite(us) and us > 0
        err = cosine_diff_compare(ref_out, result, msg=f"BM16 {precision} {name}")
        assert math.isfinite(err) and err < 0.02
        checkAllclose(
            ref_out.float(),
            result.float(),
            rtol=0.1,
            atol=0.1,
            msg=f"BM16 {precision} {name}",
            catastrophic_check=True,
        )
        first = result.clone()
        outputs[name].fill_(float("nan"))
        repeated = candidate()
        assert torch.isfinite(repeated).all()
        repeat_err = cosine_diff_compare(first, repeated, printLog=False)
        assert math.isfinite(repeat_err) and repeat_err < 1e-5
        per_element = checkAllclose(
            first.float(),
            repeated.float(),
            rtol=0.02,
            atol=0.0625 if name == "atomic" else 0.02,
            msg=f"BM16 {name} repeat",
            catastrophic_check=True,
        )
        if name != "atomic":
            assert per_element == 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


test_bm16_scale_pipeline.__test__ = False


@pytest.mark.parametrize("precision", ["A4W4", "A8W4"])
@pytest.mark.parametrize("activation", ["Silu", "Situv2", "Swiglu"])
@pytest.mark.parametrize("token", [32, 33])
@pytest.mark.parametrize("inter_dim,tile_k", [(384, 128), (512, 256)])
def test_bm16_producer_consumer(
    precision: str, activation: str, token: int, inter_dim: int, tile_k: int
) -> None:
    with torch.device("cuda"):
        result = test_bm16_scale_pipeline(
            token, 512, inter_dim, dtypes.bf16, precision, activation, tile_k
        )
    aiter.logger.info(
        "BM16 scale pipeline summary (markdown):\n%s",
        pd.DataFrame([result]).to_markdown(index=False),
    )


@pytest.mark.parametrize("precision", ["A4W4", "A8W4"])
@pytest.mark.parametrize("activation", ["Silu", "Situv2", "Swiglu"])
@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
@pytest.mark.parametrize("inter_dim,tile_k", [(384, 128), (512, 256)])
def test_coupled_candidate_numerics(
    precision: str, activation: str, block_m: int, inter_dim: int, tile_k: int
) -> None:
    with torch.device("cuda"):
        result = test_coupled_mxmoe(
            64,
            3072,
            inter_dim,
            4,
            4,
            dtypes.bf16,
            precision,
            activation,
            block_m,
            tile_k,
        )
    aiter.logger.info(
        "coupled MXMOE summary (markdown):\n%s",
        pd.DataFrame([result]).to_markdown(index=False),
    )


def main() -> None:
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("Coupled MXMOE is unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dtype", type=str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[64])
    # This Opus-routed workload exercises scale-N padding and reuses the
    # generated GPT-OSS H3072/topk4 scatter key without a test-only aux shape.
    parser.add_argument(
        "-s", "--mnk", type=dtypes.str2tuple, nargs="*", default=[(3072, 384)]
    )
    parser.add_argument("--expert", type=int, nargs="*", default=[4])
    parser.add_argument("--topk", type=int, nargs="*", default=[4])
    parser.add_argument(
        "--precision", choices=("A4W4", "A8W4"), nargs="*", default=["A4W4", "A8W4"]
    )
    parser.add_argument(
        "--activation",
        choices=("Silu", "Situv2", "Swiglu"),
        nargs="*",
        default=["Silu", "Situv2", "Swiglu"],
    )
    parser.add_argument(
        "--block-m",
        type=int,
        choices=(16, 32, 64, 128),
        nargs="*",
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--tile-k", type=int, choices=(128, 256), nargs="*", default=[128, 256]
    )
    args = parser.parse_args()
    torch.set_default_device("cuda")
    results = []
    for (
        dtype,
        token,
        shape,
        expert,
        topk,
        precision,
        activation,
        bm,
        bk,
    ) in itertools.product(
        args.dtype,
        args.batch,
        args.mnk,
        args.expert,
        args.topk,
        args.precision,
        args.activation,
        args.block_m,
        args.tile_k,
    ):
        h, inter = shape
        if inter % bk or dtype != dtypes.bf16 or topk > expert:
            continue
        results.append(
            test_coupled_mxmoe(
                token, h, inter, expert, topk, dtype, precision, activation, bm, bk
            )
        )
    aiter.logger.info(
        "coupled MXMOE summary (markdown):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
