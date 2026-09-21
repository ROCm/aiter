# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Benchmarks for aiter.topk_softmax_fused_shared_gate (Option A: routed softmax
# top-k + in-kernel shared-expert gate GEMV in a single launch).
#
# Two independent unit benchmarks, each emitting its own markdown table:
#   * bench_unfused : the NOT-fused baseline -- routed topk_softmax (existing op)
#                     + a separate shared-expert gate GEMV (sigmoid * scale) + append.
#   * bench_fused   : the fused op topk_softmax_fused_shared_gate (single launch).
# Both check the result against a torch reference (err == 0). Compare the two
# tables' `us` columns for the fused-vs-not-fused speedup.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.moe_op import topk_softmax, topk_softmax_fused_shared_gate
from aiter.test_common import benchmark, checkAllclose, run_perftest

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
    shared_w = torch.sigmoid(hidden.float() @ gate_weight.float().t()) * scale
    m = gating.shape[0]
    shared_i = (
        (base + torch.arange(num_shared, device=gating.device, dtype=torch.int32))
        .unsqueeze(0)
        .expand(m, num_shared)
    )
    return routed_w, routed_i.to(torch.int32), shared_w, shared_i


def _make_inputs(tokens, num_experts, hidden, num_shared, dtype):
    torch.manual_seed(0)
    gating = torch.randn(tokens, num_experts, dtype=dtype)
    hs = torch.randn(tokens, hidden, dtype=dtype) * 0.1
    gate_weight = torch.randn(num_shared, hidden, dtype=dtype) * 0.02
    return gating, hs, gate_weight


def _check(wbuf, ibuf, topk, ref, name):
    ref_rw, ref_ri, ref_sw, ref_si = ref
    got_rw, got_ri = wbuf[:, :topk], ibuf[:, :topk]
    got_sw, got_si = wbuf[:, topk:], ibuf[:, topk:]
    checkAllclose(
        got_si.to(dtypes.fp32),
        ref_si.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg=f"{name} shared ids",
    )
    err = checkAllclose(
        got_sw.to(dtypes.fp32),
        ref_sw.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{name} shared weights",
    )
    ref_ids, ref_w = sorted_pairs(ref_ri, ref_rw)
    got_ids, got_w = sorted_pairs(got_ri.to(dtypes.i32), got_rw)
    checkAllclose(
        got_ids.to(dtypes.fp32),
        ref_ids.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg=f"{name} routed ids",
    )
    checkAllclose(
        got_w.to(dtypes.fp32),
        ref_w.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{name} routed weights",
    )
    return err


def _roofline(tokens, num_shared, hidden, gating, hs, gate_weight, wbuf, ibuf):
    flops = 2 * tokens * num_shared * hidden  # dominant: shared-gate GEMV
    nbytes = (
        gating.numel() * gating.element_size()
        + hs.numel() * hs.element_size()
        + gate_weight.numel() * gate_weight.element_size()
        + wbuf.numel() * wbuf.element_size()
        + ibuf.numel() * ibuf.element_size()
    )
    return flops, nbytes


@benchmark()
def bench_unfused(tokens, num_experts, hidden, topk, num_shared, scale, renorm, dtype):
    """NOT-fused baseline: routed topk_softmax + separate gate GEMV + append."""
    base = num_experts
    total = topk + num_shared
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtype
    )

    w = torch.empty(tokens, total, dtype=dtypes.fp32)
    ids = torch.empty(tokens, total, dtype=dtypes.i32)
    r_w = torch.empty(tokens, topk, dtype=dtypes.fp32)
    r_i = torch.empty(tokens, topk, dtype=dtypes.i32)
    r_tei = torch.empty(tokens, topk, dtype=dtypes.i32)
    shared_ids = (
        (base + torch.arange(num_shared, dtype=dtypes.i32))
        .unsqueeze(0)
        .expand(tokens, num_shared)
    )

    def fn():
        topk_softmax(r_w, r_i, r_tei, gating, renorm)  # routed only, width topk
        shared_w = torch.sigmoid(hs.float() @ gate_weight.float().t()) * scale
        w[:, :topk] = r_w
        w[:, topk:] = shared_w
        ids[:, :topk] = r_i
        ids[:, topk:] = shared_ids

    _, us = run_perftest(fn)
    err = _check(
        w,
        ids,
        topk,
        run_torch(gating, hs, gate_weight, topk, num_shared, base, scale, renorm),
        "unfused",
    )
    flops, nbytes = _roofline(
        tokens, num_shared, hidden, gating, hs, gate_weight, w, ids
    )
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


@benchmark()
def bench_fused(tokens, num_experts, hidden, topk, num_shared, scale, renorm, dtype):
    """Fused op: topk_softmax_fused_shared_gate (single launch)."""
    base = num_experts
    total = topk + num_shared
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtype
    )

    w = torch.empty(tokens, total, dtype=dtypes.fp32)
    ids = torch.empty(tokens, total, dtype=dtypes.i32)
    tei = torch.empty(tokens, total, dtype=dtypes.i32)

    def fn():
        topk_softmax_fused_shared_gate(
            w,
            ids,
            tei,
            gating,
            renorm,
            num_shared,
            "sigmoid",
            hs,
            gate_weight,
            scale,
            base,
        )

    _, us = run_perftest(fn)
    err = _check(
        w,
        ids,
        topk,
        run_torch(gating, hs, gate_weight, topk, num_shared, base, scale, renorm),
        "fused",
    )
    flops, nbytes = _roofline(
        tokens, num_shared, hidden, gating, hs, gate_weight, w, ids
    )
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def _sweep(fn, args, dtype):
    df = []
    for tokens, experts, hidden, topk, num_shared, scale, renorm in itertools.product(
        args.tokens,
        args.experts,
        args.hidden,
        args.topk,
        args.num_shared,
        args.scale,
        args.renorm,
    ):
        df.append(
            fn(tokens, experts, hidden, topk, num_shared, scale, bool(renorm), dtype)
        )
    return pd.DataFrame(df)


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
    parser.add_argument(
        "--bench",
        choices=["fused", "unfused", "both"],
        default="both",
        help="which benchmark to run",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,"
    )
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[1, 4, 17, 64])
    parser.add_argument("-e", "--experts", type=int, nargs="*", default=[512])
    parser.add_argument("--hidden", type=int, nargs="*", default=[4096])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[10])
    parser.add_argument("--num-shared", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--scale", type=float, nargs="*", default=[1.0, 0.5])
    parser.add_argument("--renorm", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    for dtype in args.dtype:
        if args.bench in ("unfused", "both"):
            df = _sweep(bench_unfused, args, dtype)
            aiter.logger.info(
                "UNFUSED baseline (routed topk_softmax + separate gate GEMV + append) "
                "(%s):\n%s",
                dtype,
                df.to_markdown(index=False),
            )
        if args.bench in ("fused", "both"):
            df = _sweep(bench_fused, args, dtype)
            aiter.logger.info(
                "FUSED topk_softmax_fused_shared_gate (single launch) (%s):\n%s",
                dtype,
                df.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
