# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
BLOCK = 128
ROUTED_SCALING_FACTOR = 2.5  # GLM-5.x config


def block_quant(w):
    """Per-128x128 FP8 block quant of [E, N, K]; returns fp8 weight, fp32 scale and the dequantized weight."""
    e, n, k = w.shape
    blocks = w.float().view(e, n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = (
        blocks.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12)
        / torch.finfo(dtypes.fp8).max
    )
    q = (blocks / scale).to(dtypes.fp8)
    return q.view(e, n, k), scale.view(e, -1), (q.float() * scale).view(e, n, k)


@functools.lru_cache(maxsize=1)
def make_weights(model_dim, inter_dim, expert):
    w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtypes.bf16) / 10
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16) / 10
    w1_q, w1_scale, w1_ref = block_quant(w1)
    w2_q, w2_scale, w2_ref = block_quant(w2)
    w1_q, w2_q = shuffle_weight(w1_q, (16, 16)), shuffle_weight(w2_q, (16, 16))
    return w1_q, w1_scale, w1_ref, w2_q, w2_scale, w2_ref


def shared_expert_routing(token, expert, topk):
    """vLLM's AITER fused-shared-expert layout: routed top-(k-1), then the shared expert (id expert-1) at weight 1."""
    score = torch.randn((token, expert - 1)).softmax(-1)
    w, ids = torch.topk(score, topk - 1, dim=-1)
    w = w / w.sum(-1, keepdim=True) * ROUTED_SCALING_FACTOR
    ids = torch.cat([ids, torch.full((token, 1), expert - 1)], dim=1).to(dtypes.i32)
    return torch.cat([w, torch.ones((token, 1))], dim=1).float(), ids


def run_torch(x, w1, w2, topk_weights, topk_ids):
    return torch_moe(x, w1, w2, topk_weights, topk_ids)


@benchmark()
def test_glm5_tp8_decode_moe(token, model_dim, inter_dim, expert, topk):
    w1_q, w1_scale, w1_ref, w2_q, w2_scale, w2_ref = make_weights(
        model_dim, inter_dim, expert
    )
    x = torch.randn((token, model_dim), dtype=dtypes.bf16)
    topk_weights, topk_ids = shared_expert_routing(token, expert, topk)
    ref = run_torch(x, w1_ref, w2_ref, topk_weights, topk_ids)

    candidates = {
        "fused_moe": lambda: fused_moe(
            x,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            quant_type=aiter.QuantType.per_128x128,
            activation=aiter.ActivationType.Silu,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        ),
    }
    n_used = int(topk_ids.unique().numel())
    flops = 2 * token * topk * 3 * inter_dim * model_dim
    nbytes = (
        n_used * 3 * inter_dim * model_dim * w1_q.element_size()
        + 2 * token * model_dim * x.element_size()
    )

    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        o, r = out.to(dtypes.fp32), ref.to(dtypes.fp32)
        err = checkAllclose(r, o, msg=f"{name}: GLM-5 TP8 decode MoE")
        logits_diff = float(
            1
            - 2
            * (o.double() * r.double()).sum()
            / (o.double().square() + r.double().square()).sum()
        )
        assert torch.isfinite(o).all(), f"{name}: non-finite output at token={token}"
        # Same gate as test_moe_2stage: an FP8 MoE routinely misses rtol/atol=1e-2 elementwise.
        assert not (
            err != 0 and logits_diff > 0.01
        ), f"{name}: token={token} err={err}, logits_diff={logits_diff}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} logits_diff"] = logits_diff
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "GLM-5 decode MoE test unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="GLM-5.x FP8 TP8 decode fused MoE (fused shared expert) against a torch reference",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 32],
        help="Token counts; covers each tuned bucket up to 32 and its lower edge.",
    )
    parser.add_argument(
        "-dim",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(6144, 256)],
        help="(model_dim, inter_dim) pairs; 6144,256 is GLM-5.x at TP8.",
    )
    parser.add_argument(
        "-e",
        "--expert",
        type=int,
        nargs="*",
        default=[257],
        help="Experts incl. the fused shared one.",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        nargs="*",
        default=[9],
        help="Top-k incl. the shared slot.",
    )
    args = parser.parse_args()

    df = []
    for (model_dim, inter_dim), expert, topk, token in itertools.product(
        args.dim, args.expert, args.topk, args.token
    ):
        df.append(test_glm5_tp8_decode_moe(token, model_dim, inter_dim, expert, topk))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "GLM-5 TP8 decode MoE summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
