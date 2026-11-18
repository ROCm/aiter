# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test per-tensor FP8 CK MoE shards whose inter_dim is not a multiple of 256."""

import argparse
import itertools
import os

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, run_perftest

SUPPORTED_ARCHS = ("gfx942", "gfx950")

pytestmark = pytest.mark.skipif(
    get_gfx_runtime() not in SUPPORTED_ARCHS,
    reason="Exercises the gfx942 and gfx950 per-tensor FP8 CK MoE dispatch",
)

# FP8 activations alone give about 0.05 relative error. A wrong kernel is far above it.
RELATIVE_ERROR_TOLERANCE = 0.1
# The global norm can miss a single wrong tile, so each row and element is bounded too.
# The element error is measured in units of the reference's RMS.
ROW_ERROR_TOLERANCE = 0.15
ELEMENT_ERROR_TOLERANCE = 1.0

# Padding inter_dim in HBM to the largest stage-2 K tile avoids in-kernel padding.
HBM_PAD_MULTIPLE = 256

# These shapes are multiples of 256, so they never needed padding.
ALIGNED_INTER_DIMS = [256, 512, 1024]
# None of these shapes is a multiple of 256, the K of the largest stage-2 tile.
# 88, 176, 352 and 384 are TP shards of DiffusionGemma-26B-A4B and Qwen3-235B-A22B.
UNALIGNED_INTER_DIMS = [64, 88, 176, 192, 352, 384, 640, 896]
# 2880 is a hidden size that is not a multiple of 128, as in gpt-oss.
UNALIGNED_MODEL_DIMS = [2880, 4096]
BLOCK_MS = [32, 64, 128]


@pytest.fixture(autouse=True)
def default_dispatch(monkeypatch):
    # Bypass the tuned CSV so that a tuned kernel cannot hide a dispatch bug.
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    get_2stage_cfgs.cache_clear()
    yield
    get_2stage_cfgs.cache_clear()


def make_inputs(tokens, model_dim, inter_dim, experts, topk):
    torch.manual_seed(42)
    x = torch.randn(tokens, model_dim, device="cuda", dtype=dtypes.bf16) / 10
    w1 = (
        torch.randn(experts, 2 * inter_dim, model_dim, device="cuda", dtype=dtypes.bf16)
        / 10
    )
    w2 = (
        torch.randn(experts, model_dim, inter_dim, device="cuda", dtype=dtypes.bf16)
        / 10
    )
    # One scale per expert is per-tensor quantization.
    w1_q, w1_scale = aiter.pertoken_quant(w1.view(experts, -1), quant_dtype=dtypes.fp8)
    w2_q, w2_scale = aiter.pertoken_quant(w2.view(experts, -1), quant_dtype=dtypes.fp8)
    weights, ids = torch.softmax(
        torch.randn(tokens, experts, device="cuda"), dim=-1
    ).topk(topk, dim=-1)
    return {
        "x": x,
        "w1_q": w1_q.view(w1.shape),
        "w1_scale": w1_scale,
        "w2_q": w2_q.view(w2.shape),
        "w2_scale": w2_scale,
        "weights": weights,
        "ids": ids.to(torch.int32),
    }


def compute_reference(inputs):
    # Compute the reference from the dequantized weights in FP32.
    x, ids, weights = inputs["x"], inputs["ids"], inputs["weights"]
    experts = inputs["w1_q"].shape[0]
    w1_dq = inputs["w1_q"].float() * inputs["w1_scale"].view(experts, 1, 1)
    w2_dq = inputs["w2_q"].float() * inputs["w2_scale"].view(experts, 1, 1)
    expected = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    for expert in range(experts):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1_dq[expert]).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, w2_dq[expert])
        expected.index_add_(0, rows, y * weights[rows, slots, None])
    return expected


def pad_inter_dim_in_hbm(inputs, multiple):
    # The zero padding leaves the per-expert scales and the result unchanged.
    w1_q, w2_q = inputs["w1_q"], inputs["w2_q"]
    experts, two_inter, model_dim = w1_q.shape
    inter_dim = two_inter // 2
    padded = -(-inter_dim // multiple) * multiple
    w1_p = torch.zeros(experts, 2 * padded, model_dim, device="cuda", dtype=w1_q.dtype)
    w1_p[:, :inter_dim] = w1_q[:, :inter_dim]
    w1_p[:, padded : padded + inter_dim] = w1_q[:, inter_dim:]
    w2_p = torch.zeros(experts, model_dim, padded, device="cuda", dtype=w2_q.dtype)
    w2_p[:, :, :inter_dim] = w2_q
    return {**inputs, "w1_q": w1_p, "w2_q": w2_p}


def make_fused_moe(inputs, block_m):
    # Shuffle outside the timed call, as a model loader does once at load time.
    w1 = shuffle_weight(inputs["w1_q"], layout=(16, 16))
    w2 = shuffle_weight(inputs["w2_q"], layout=(16, 16))
    return lambda: fused_moe(
        inputs["x"],
        w1,
        w2,
        inputs["weights"],
        inputs["ids"],
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Tensor,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        block_size_M=block_m,
    )


def check_accuracy(actual, expected, msg):
    assert torch.isfinite(actual).all(), f"{msg}: the output has NaN or Inf values."
    diff = actual.float() - expected
    rel_err = (diff.norm() / expected.norm()).item()
    row_err = (diff.norm(dim=1) / expected.norm(dim=1)).max().item()
    elem_err = (diff.abs().max() / expected.pow(2).mean().sqrt()).item()
    assert (
        rel_err < RELATIVE_ERROR_TOLERANCE
    ), f"{msg}: the relative error {rel_err:.4f} is above {RELATIVE_ERROR_TOLERANCE}."
    assert (
        row_err < ROW_ERROR_TOLERANCE
    ), f"{msg}: the worst row error {row_err:.4f} is above {ROW_ERROR_TOLERANCE}."
    assert (
        elem_err < ELEMENT_ERROR_TOLERANCE
    ), f"{msg}: the worst element error {elem_err:.4f} is above {ELEMENT_ERROR_TOLERANCE}."
    return rel_err, row_err, elem_err


def check_case(tokens, model_dim, inter_dim, experts, topk, block_m):
    inputs = make_inputs(tokens, model_dim, inter_dim, experts, topk)
    actual = make_fused_moe(inputs, block_m)()
    torch.accelerator.synchronize()
    check_accuracy(actual, compute_reference(inputs), "fused_moe")


@benchmark()
def bench_case(tokens, model_dim, inter_dim, experts, topk, block_m):
    inputs = make_inputs(tokens, model_dim, inter_dim, experts, topk)
    expected = compute_reference(inputs)
    candidates = {"in-kernel padding": make_fused_moe(inputs, block_m)}
    if inter_dim % HBM_PAD_MULTIPLE:
        candidates[f"HBM padding to {HBM_PAD_MULTIPLE}"] = make_fused_moe(
            pad_inter_dim_in_hbm(inputs, HBM_PAD_MULTIPLE), block_m
        )
    # Both candidates are measured against the FLOPs and bytes of the unpadded shape.
    flops = 6 * tokens * topk * model_dim * inter_dim
    active = inputs["ids"].unique().numel()
    nbytes = active * 3 * model_dim * inter_dim + 2 * 2 * tokens * model_dim
    ret = {}
    for name, fn in candidates.items():
        actual, us = run_perftest(fn)
        rel_err, row_err, elem_err = check_accuracy(actual, expected, name)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = rel_err
        ret[f"{name} row err"] = row_err
        ret[f"{name} element err"] = elem_err
    return ret


# The benchmark wrapper accepts CLI sweep arguments, not pytest fixtures.
bench_case.__test__ = False


@pytest.mark.parametrize("tokens", [33])
@pytest.mark.parametrize("model_dim", [4096])
@pytest.mark.parametrize("inter_dim", ALIGNED_INTER_DIMS)
@pytest.mark.parametrize("experts", [32])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("block_m", BLOCK_MS)
def test_per_tensor_fp8_aligned_inter_dim(
    tokens, model_dim, inter_dim, experts, topk, block_m
):
    check_case(tokens, model_dim, inter_dim, experts, topk, block_m)


@pytest.mark.parametrize("tokens", [33])
@pytest.mark.parametrize("model_dim", UNALIGNED_MODEL_DIMS)
@pytest.mark.parametrize("inter_dim", UNALIGNED_INTER_DIMS)
@pytest.mark.parametrize("experts", [32])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("block_m", BLOCK_MS)
def test_per_tensor_fp8_unaligned_inter_dim(
    tokens, model_dim, inter_dim, experts, topk, block_m
):
    check_case(tokens, model_dim, inter_dim, experts, topk, block_m)


def main():
    if get_gfx_runtime() not in SUPPORTED_ARCHS:
        aiter.logger.warning(
            "Per-tensor FP8 CK MoE is not supported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--tokens", nargs="+", type=int, default=[33])
    parser.add_argument("-e", "--experts", nargs="+", type=int, default=[32])
    parser.add_argument("-k", "--topk", nargs="+", type=int, default=[4])
    parser.add_argument("-m", "--block_m", nargs="+", type=int, default=BLOCK_MS)
    args = parser.parse_args()
    # Bypass the tuned CSV so that a tuned kernel cannot hide a dispatch bug.
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
    get_2stage_cfgs.cache_clear()
    shapes = [(4096, i) for i in ALIGNED_INTER_DIMS] + list(
        itertools.product(UNALIGNED_MODEL_DIMS, UNALIGNED_INTER_DIMS)
    )
    rows = [
        bench_case(tokens, model_dim, inter_dim, experts, topk, block_m)
        for tokens, (model_dim, inter_dim), experts, topk, block_m in itertools.product(
            args.tokens, shapes, args.experts, args.topk, args.block_m
        )
    ]
    aiter.logger.info(
        "Per-tensor FP8 MoE, unaligned inter_dim:\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
