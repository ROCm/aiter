# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the HipKittens block-FP8 grouped MoE against an fp32 torch reference.

Random experts with power-of-two 128x128 block scales (the kernel's contract), top-8 routed or top-9 with the shared
expert fused as expert 288, with and without the SwiGLU clamp. Token counts cover the small-batch variant that skips
all-padding tile halves (< ~5k tokens) and the large-batch one.
"""

import argparse
import sys

import torch

import aiter
from aiter.ops.hk_fp8_moe import HIDDEN, INTER, hk_fp8_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import perftest

FP8 = torch.float8_e4m3fn
# power-of-two activation scales on top of the shared fp8 weights
TOL = 0.06


def requant_pow2(w):
    """[E, n, k] float -> fp8 with power-of-two 128x128 block scales [E, n/128, k/128]."""
    E, n, k = w.shape
    b = w.float().view(E, n // 128, 128, k // 128, 128)
    s = torch.exp2(
        torch.ceil(
            torch.log2(b.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0)
        )
    )
    return (b / s).to(FP8).view(E, n, k), s.view(E, n // 128, k // 128).contiguous()


def dequant(q, s):
    E, n, k = q.shape
    b = q.float().view(E, n // 128, 128, k // 128, 128)
    return (b * s.view(E, n // 128, 1, k // 128, 1)).view(E, n, k)


def reference(x, W13, W2, tw, ids, limit):
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for e in torch.unique(ids).tolist():
        if e < 0 or e >= W13.shape[0]:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        g, u = (x[tok].float() @ W13[e].t()).chunk(2, dim=-1)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        act = torch.nn.functional.silu(g) * u
        out.index_add_(0, tok, act @ W2[e].t() * tw[tok, slot, None])
    return out


def make_weights(E):
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13 = requant_pow2(
        torch.randn(E, 2 * INTER, HIDDEN, device="cuda", generator=gen) * 0.02
    )
    w2, s2 = requant_pow2(
        torch.randn(E, HIDDEN, INTER, device="cuda", generator=gen) * 0.02
    )
    return w13, s13, w2, s2


def make_routing(T, E):
    gen = torch.Generator(device="cuda").manual_seed(T)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen)
    tw, ids = torch.topk(
        torch.randn(T, 288, device="cuda", generator=gen).softmax(-1), 8, dim=-1
    )
    if E == 289:  # the fused shared expert, selected by every token
        ids = torch.cat([ids, torch.full_like(ids[:, :1], 288)], dim=1)
        tw = torch.cat([tw, torch.ones_like(tw[:, :1])], dim=1)
    return x, ids.to(torch.int32), tw


@perftest()
def run(x, ids, tw, w13, s13, w2, s2, limit):
    return hk_fp8_moe(x, ids, tw, w13, s13, w2, s2, swiglu_limit=limit)


def test_case(T, E, limit, weights):
    w13, s13, w2, s2 = weights
    x, ids, tw = make_routing(T, E)
    out, us = run(
        x,
        ids,
        tw,
        shuffle_weight(w13, (16, 16)),
        s13,
        shuffle_weight(w2, (16, 16)),
        s2,
        limit,
    )
    rows = torch.arange(0, T, max(1, T // 256), device="cuda")
    ref = reference(
        x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], limit
    )
    err = ((out[rows].float() - ref).norm() / ref.norm()).item()
    ok = err < TOL and not torch.isnan(out).any()
    aiter.logger.info(
        f"hk_fp8_moe T={T:6d} experts={E} topk={ids.shape[1]} limit={limit:4.1f}: rel err {err:.4f} "
        f"{us:8.1f} us  {'ok' if ok else 'FAIL'}"
    )
    return ok


def test_routing_replay(weights):
    """Replay changed routing through the same graph, including collisions and tails."""
    w13, s13, w2, s2 = weights
    E = w13.shape[0]
    packed = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    W13, W2 = dequant(w13, s13), dequant(w2, s2)
    ok = True
    # Straddle 4096 selections, with a partial wave and a partial routing block.
    topk = 9 if E == 289 else 8
    for T in (65, 4096 // topk, 4096 // topk + 1):
        x, ids, tw = make_routing(T, E)
        original_ids = ids.clone()
        args = (x, ids, tw, packed[0], s13, packed[1], s2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            hk_fp8_moe(*args, swiglu_limit=10.0)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = hk_fp8_moe(*args, swiglu_limit=10.0)
        rows = torch.arange(0, T, max(1, T // 32), device=x.device)
        # Revisit random routing after all-invalid routing to catch stale counters.
        for routing in (
            "random",
            "one_expert",
            "bank_aliases",
            "invalid",
            "empty",
            "random",
        ):
            if routing == "random":
                ids.copy_(original_ids)
            elif routing == "one_expert":
                ids.fill_(E - 1)
            elif routing == "bank_aliases":
                ids.copy_(
                    (torch.arange(ids.numel(), device=x.device).view_as(ids) * 64) % E
                )
            elif routing == "invalid":
                ids.view(-1)[::3] = -1
                ids.view(-1)[1::3] = 2147483647
            else:
                ids.fill_(-1)
            x.mul_(1.03125)
            tw.mul_(0.9375)
            graph.replay()
            first = out.clone()
            graph.replay()
            ref = reference(x[rows], W13, W2, tw[rows], ids[rows], 10.0)
            err = (
                (out[rows].float() - ref).norm() / ref.norm().clamp_min(1e-12)
            ).item()
            passed = (
                err < TOL
                and torch.isfinite(out).all().item()
                and torch.equal(first, out)
            )
            ok = ok and passed
            aiter.logger.info(
                f"hk_fp8_moe routing replay T={T} E={E} {routing}: "
                f"rel err {err:.4f} {'ok' if passed else 'FAIL'}"
            )
    return ok


def test_rejects(weights):
    w13, s13, w2, s2 = weights
    x, ids, tw = make_routing(8, w13.shape[0])
    args = [x, ids, tw, w13, s13, w2, s2]
    bad_args = []
    for index in range(len(args)):
        bad = args.copy()
        # Avoid copying the multi-GiB weight tensors to the CPU.
        bad[index] = torch.empty(
            args[index].shape, dtype=args[index].dtype, device="meta"
        )
        bad_args.append(bad)
    for index, tensor in (
        (0, x[:, :2048]),
        (1, ids[:, :4]),
        (1, ids[0, 0]),
        (1, ids.float()),
        (3, w13[0, 0, 0]),
        (3, w13[:0]),
    ):
        bad = args.copy()
        bad[index] = tensor
        bad_args.append(bad)
    ok = True
    for bad in bad_args:
        try:
            hk_fp8_moe(*bad)
            ok = False
        except ValueError:
            pass
    for out in (
        torch.empty_like(x, device="cpu"),
        torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x),
    ):
        try:
            hk_fp8_moe(*args, out=out)
            ok = False
        except ValueError:
            pass
    aiter.logger.info(f"hk_fp8_moe rejects invalid inputs: {'ok' if ok else 'FAIL'}")
    return ok


def test_int64_ids_and_empty(weights):
    w13, s13, w2, s2 = weights
    packed = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    x, ids, tw = make_routing(8, w13.shape[0])
    ids64 = ids.long()
    ids64[:, ::2] = 2**32
    ids64[:, 1::4] = -(2**32)
    masked = ids.clone()
    masked[:, ::2] = -1
    masked[:, 1::4] = -1
    args = (packed[0], s13, packed[1], s2)
    expected = hk_fp8_moe(x, masked, tw, *args)
    actual = hk_fp8_moe(x, ids64, tw, *args)
    empty = hk_fp8_moe(x[:0], ids[:0], tw[:0], *args)
    ok = torch.equal(expected, actual) and empty.shape == x[:0].shape
    aiter.logger.info(
        f"hk_fp8_moe int64 masking and empty input: {'ok' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="HipKittens block-FP8 grouped MoE: correctness vs fp32 and timing",
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=None, help="Token counts"
    )
    parser.add_argument(
        "--routing-only", action="store_true", help="Only routing replay regressions"
    )
    args = parser.parse_args()

    if aiter.get_gfx() != "gfx950":
        print(f"Skipping test_hk_fp8_moe.py: requires gfx950, got {aiter.get_gfx()}")
        sys.exit(0)

    tokens = args.tokens or [1, 1024, 4096, 6144, 16384]
    results = []
    for E in (288, 289):
        weights = make_weights(E)
        for T in ([] if args.routing_only else tokens):
            for limit in (0.0, 10.0):
                results.append(test_case(T, E, limit, weights))
        results.append(test_routing_replay(weights))
        results.append(test_rejects(weights))
        results.append(test_int64_ids_and_empty(weights))
        del weights
    sys.exit(0 if all(results) else 1)
