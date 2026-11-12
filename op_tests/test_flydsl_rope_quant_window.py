#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL rope_quant_window (DeepSeek V4.1's RoPE / MXFP8 / window seam) against the torch
port of ATOM's Triton kernel, byte for byte, in every query and KV mode."""

import argparse
import random

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import perftest
from op_tests.rope_quant_window_ref import (
    COS,
    E8,
    FP8,
    RD,
    SENTINEL,
    SIN,
    D,
    PaddingStep,
    Step,
    byte_diff,
    dev,
    rope_quant_window_ref,
    sink_kwargs,
    sinks,
    with_zeros,
)

MODES = [
    "qfp8",
    "rot",
    "qfp8+ringp",
    "rot+ringh",
    "rot+ringps",
    "qfp8+rows",
    "rot+qat+vs",
    "rot+qat",
    "qfp8+qat+vs",
]


def flydsl(query, kv, positions, kw):
    from aiter.ops.flydsl.kernels.rope_quant_window import rope_quant_window

    return rope_quant_window(query, kv, COS, SIN, positions, rope_dim=RD, **kw)


run_flydsl = perftest()(flydsl)


def test_flydsl_rope_quant_window(mode, tokens, heads, idx64, padding=False):
    step = PaddingStep(tokens, idx64) if padding else Step(tokens, idx64)
    query = with_zeros(
        (torch.randn(tokens, heads, D, device=dev) * 3).to(torch.bfloat16)
    )
    # An odd token count also takes kv as a strided slice.
    width = D + 64 if tokens % 2 else D
    kv_full = with_zeros(
        (torch.randn(tokens, width, device=dev) * 3).to(torch.bfloat16)
    )
    kv = kv_full[:, 32 : 32 + D] if tokens % 2 else kv_full
    ref = sinks(mode, tokens, heads)
    ref["query"] = query.clone()
    kw_ref = sink_kwargs(ref, step)
    has_kv = any(k in kw_ref for k in ("ring", "values", "qat"))
    rope_quant_window_ref(
        ref["query"], kv if has_kv else None, step.positions, rope_dim=RD, **kw_ref
    )
    # The timed calls write scratch sinks and a scratch query (the rotate mode works in
    # place); one call on fresh ones gives the bytes, so none can come from a timed call.
    scratch = sink_kwargs(sinks(mode, tokens, heads), step)
    _, us = run_flydsl(query.clone(), kv if has_kv else None, step.positions, scratch)
    got = sinks(mode, tokens, heads)
    got["query"] = query.clone()
    flydsl(got["query"], kv if has_kv else None, step.positions, sink_kwargs(got, step))
    torch.cuda.synchronize()
    bad = byte_diff(ref, got)
    tag = (
        f"{mode} T={tokens} H={heads} idx64={idx64}{' all-padding' if padding else ''}"
    )
    result = f"\033[31mbytes differ {bad}\033[0m" if bad else "byte exact"
    aiter.logger.info("[result] %s | %.2f us | vs reference: %s", tag, us, result)
    return not bad


def test_packed_ring_rejected():
    """A ring of rows packing values and codes together has no planar layout to write."""
    step = Step(5, False)
    o = sinks("ringp", 5, 0)
    kv = torch.randn(5, D, device=dev).to(torch.bfloat16)
    query = torch.empty(5, 0, D, device=dev, dtype=torch.bfloat16)
    packed = torch.zeros(o["pool"].shape[0], D + D // 32, device=dev, dtype=torch.uint8)
    try:
        flydsl(query, kv, step.positions, {"ring": (packed, step, o["w"])})
    except ValueError:
        return True
    return False


def test_tdm_gate_wide_rows(tokens=4096, heads=128, dim=1024):
    """Enough query rows for the gfx1250 TDM path, but rows wider than it takes: the rows
    path has to run them instead."""
    step = Step(tokens, False)
    query = (torch.randn(tokens, heads, dim, device=dev) * 3).to(torch.bfloat16)
    fp8 = [
        (
            torch.zeros(tokens, heads, dim, device=dev, dtype=torch.uint8),
            torch.zeros(tokens, heads, dim // 32, device=dev, dtype=torch.uint8),
        )
        for _ in range(2)
    ]
    for v, s in fp8:
        v.fill_(SENTINEL)
        s.fill_(SENTINEL)
    ref, got = [(v.view(FP8), s.view(E8)) for v, s in fp8]
    rope_quant_window_ref(query, None, step.positions, rope_dim=RD, query_fp8=ref)
    flydsl(query, None, step.positions, {"query_fp8": got})
    torch.cuda.synchronize()
    bad = sum(
        int((r.view(torch.uint8) != g.view(torch.uint8)).sum())
        for r, g in zip(ref, got)
    )
    aiter.logger.info(
        "[result] qfp8 T=%d H=%d D=%d | vs reference: %s",
        tokens,
        heads,
        dim,
        f"\033[31mbytes differ {bad}\033[0m" if bad else "byte exact",
    )
    return not bad


def test_bad_kv_args_rejected():
    """Arguments the kernel cannot honor raise instead of writing the wrong bytes: it takes
    raw pointers, so a missing scales (the scale stores would land on the query), a kv of the
    wrong width, dtype or inner stride, and a strided dense sink would all be misread or
    overrun."""
    step = Step(5, False)
    query = torch.zeros(5, 16, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(5, D, device=dev).to(torch.bfloat16)
    o = sinks("vs", 5, 16)
    cases = {
        "values without scales": (kv, {"values": o["v"]}),
        "scales without values": (kv, {"scales": o["s"]}),
        "kv inner stride 2": (
            torch.randn(5, 2 * D, device=dev).to(torch.bfloat16)[:, ::2],
            {"values": o["v"], "scales": o["s"]},
        ),
        "kv narrower than D": (kv[:, : D // 2], {"values": o["v"], "scales": o["s"]}),
        "fp32 kv": (kv.float(), {"values": o["v"], "scales": o["s"]}),
        "strided qat": (
            kv,
            {"qat": torch.zeros(5, 2 * D, device=dev, dtype=torch.bfloat16)[:, ::2]},
        ),
        "strided query_fp8": (
            None,
            {
                "query_fp8": (
                    torch.zeros(5, 16, 2 * D, device=dev, dtype=torch.uint8)[..., ::2],
                    torch.zeros(5, 16, D // 32, device=dev, dtype=torch.uint8),
                )
            },
        ),
    }
    accepted = []
    for name, (kv_in, kw) in cases.items():
        q = query.clone()
        try:
            flydsl(q, kv_in, step.positions, kw)
            accepted.append(name)
        except ValueError:
            if not torch.equal(q, query):
                accepted.append(f"{name} (query written)")
    return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    # T * H rows from 512K on take the gfx1250 TDM path.
    parser.add_argument(
        "-t", "--token", type=int, nargs="*", default=[1, 5, 13, 128, 1000, 4096, 9000]
    )
    parser.add_argument("--heads", type=int, nargs="*", default=[16, 128])
    parser.add_argument("--mode", type=str, nargs="*", default=MODES, choices=MODES)
    args = parser.parse_args()
    gfx = get_gfx_runtime()
    if gfx not in ("gfx950", "gfx1250"):
        aiter.logger.warning("rope_quant_window unsupported on %s; skipping", gfx)
        raise SystemExit(0)
    random.seed(1)
    torch.manual_seed(1)

    failed = [
        f"{mode} T={tokens} H={heads} idx64={idx64}"
        for mode in args.mode
        for tokens in args.token
        for heads in args.heads
        for idx64 in (False, True)
        if not test_flydsl_rope_quant_window(mode, tokens, heads, idx64)
    ]
    # heads == 0: the KV row alone.
    failed += [
        f"{mode} T=37 H=0"
        for mode in ("rows", "qat+vs", "ringp", "ringps")
        if not test_flydsl_rope_quant_window(mode, 37, 0, False)
    ]
    # An all-padding step: the ring is untouched and its metadata never read.
    failed += [
        f"{mode} T=5 H={heads} all-padding"
        for mode, heads in (
            ("qfp8+ringp", 16),
            ("rot+ringh", 16),
            ("rot+ringps", 16),
            ("qat+vs+ringp", 0),
        )
        for idx64 in (False, True)
        if not test_flydsl_rope_quant_window(mode, 5, heads, idx64, padding=True)
    ]
    failed += [] if test_tdm_gate_wide_rows() else ["wide-row TDM gate"]
    if not test_packed_ring_rejected():
        failed.append("packed byte ring accepted")
    failed += [f"{name} accepted" for name in test_bad_kv_args_rejected()]
    if failed:
        raise SystemExit(f"{len(failed)} cases differ: {failed}")
    aiter.logger.info("rope_quant_window: every case byte exact")
