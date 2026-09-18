# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Option A: aiter.topk_softmax with the shared-expert gate GEMV folded into the
# routed softmax top-k kernel (fuse-gate mode). Perf + correctness sweep.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.ops.moe_op import topk_softmax
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def sorted_pairs(ids, weights):
    order = torch.argsort(ids, dim=-1)
    return torch.gather(ids, -1, order), torch.gather(weights, -1, order)


def run_torch(gating, hidden, gate_weight, topk, num_shared, base, scale, renorm):
    # Reference only: fp32 math. Not timed, not in the table.
    probs = torch.softmax(gating.float(), dim=-1)
    routed_w, routed_i = torch.topk(probs, topk, dim=-1)
    if renorm:
        routed_w = routed_w / routed_w.sum(dim=-1, keepdim=True)
    shared_logit = hidden.float() @ gate_weight.float().t()  # [M, num_shared]
    shared_w = torch.sigmoid(shared_logit) * scale
    m = gating.shape[0]
    shared_i = (
        base + torch.arange(num_shared, device=gating.device, dtype=torch.int32)
    ).unsqueeze(0).expand(m, num_shared)
    return routed_w, routed_i.to(torch.int32), shared_w, shared_i


@benchmark()
def test_op(tokens, num_experts, hidden, topk, num_shared, scale, renorm, dtype):
    torch.manual_seed(0)
    base = num_experts  # shared experts appended at ids [num_experts .. +num_shared)

    gating = torch.randn(tokens, num_experts, dtype=dtype)
    hs = torch.randn(tokens, hidden, dtype=dtype) * 0.1
    gate_weight = torch.randn(num_shared, hidden, dtype=dtype) * 0.02

    total = topk + num_shared
    topk_weights = torch.empty(tokens, total, dtype=dtypes.fp32)
    topk_ids = torch.empty(tokens, total, dtype=dtypes.i32)
    token_expert_indices = torch.empty(tokens, total, dtype=dtypes.i32)

    def fn():
        topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating,
            renorm,
            num_shared_experts=num_shared,
            shared_expert_scoring_func="sigmoid",
            hidden_states=hs,
            gate_weight=gate_weight,
            shared_expert_scale=scale,
            shared_expert_base=base,
        )

    _, us = run_perftest(fn)

    ref_rw, ref_ri, ref_sw, ref_si = run_torch(
        gating, hs, gate_weight, topk, num_shared, base, scale, renorm
    )
    got_rw, got_ri = topk_weights[:, :topk], topk_ids[:, :topk]
    got_sw, got_si = topk_weights[:, topk:], topk_ids[:, topk:]

    # Shared columns are the new Option A logic -- ids exact, weights within tol.
    checkAllclose(
        got_si.to(dtypes.fp32), ref_si.to(dtypes.fp32), rtol=0, atol=0,
        msg="shared ids",
    )
    err = checkAllclose(
        got_sw.to(dtypes.fp32), ref_sw.to(dtypes.fp32), rtol=2e-2, atol=2e-2,
        msg="shared weights",
    )
    # Routed columns: compare as sets per row (kernel/topk order may differ).
    ref_ids, ref_w = sorted_pairs(ref_ri, ref_rw)
    got_ids, got_w = sorted_pairs(got_ri.to(dtypes.i32), got_rw)
    checkAllclose(
        got_ids.to(dtypes.fp32), ref_ids.to(dtypes.fp32), rtol=0, atol=0,
        msg="routed ids",
    )
    checkAllclose(
        got_w.to(dtypes.fp32), ref_w.to(dtypes.fp32), rtol=2e-2, atol=2e-2,
        msg="routed weights",
    )

    # Roofline: dominant work is the shared-gate GEMV (2*M*S*H); softmax traffic
    # over gating dominates the byte count.
    flops = 2 * tokens * num_shared * hidden
    nbytes = (
        gating.numel() * gating.element_size()
        + hs.numel() * hs.element_size()
        + gate_weight.numel() * gate_weight.element_size()
        + topk_weights.numel() * topk_weights.element_size()
        + topk_ids.numel() * topk_ids.element_size()
    )
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "topk_softmax shared-gate unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument("-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,")
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[1, 4, 17, 64])
    parser.add_argument("-e", "--experts", type=int, nargs="*", default=[512])
    parser.add_argument("--hidden", type=int, nargs="*", default=[4096])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[8])
    parser.add_argument("--num-shared", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--scale", type=float, nargs="*", default=[1.0, 0.5])
    parser.add_argument("--renorm", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for tokens, experts, hidden, topk, num_shared, scale, renorm in itertools.product(
            args.tokens, args.experts, args.hidden, args.topk,
            args.num_shared, args.scale, args.renorm,
        ):
            df.append(
                test_op(
                    tokens, experts, hidden, topk, num_shared,
                    scale, bool(renorm), dtype,
                )
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "topk_softmax shared-gate summary (%s):\n%s",
            dtype,
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
