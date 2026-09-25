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
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.moe_op import topk_softmax, topk_softmax_fused_shared_gate
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def _gfx_supported():
    return torch.cuda.is_available() and get_gfx() in SUPPORTED_GFX


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
    """NOT-fused baseline, as close to the fused op's work as possible so the speedup
    is not inflated by torch plumbing:
      * routed topk_softmax writes the routed columns in place through strided views
        (w[:, :topk] / ids[:, :topk]) -- no separate buffer + copy.
      * the shared-gate GEMV runs in the input dtype (bf16 mm accumulates in fp32 on
        MFMA, matching the fused kernel) -- no .float() upcast temporary, which would
        also make the baseline more accurate than the op under test.
      * the shared ids depend only on (base, num_shared), so they are precomputed once
        OUTSIDE the timed region.
    Timed region is thus ~2 kernels (routed top-k + gate mm) + sigmoid/scale store."""
    base = num_experts
    total = topk + num_shared
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtype
    )

    w = torch.empty(tokens, total, dtype=dtypes.fp32)
    ids = torch.empty(tokens, total, dtype=dtypes.i32)
    tei = torch.empty(tokens, topk, dtype=dtypes.i32)
    w_routed, ids_routed = w[:, :topk], ids[:, :topk]  # views, written in place
    # shared ids depend only on (base, num_shared) -> precompute once, outside timing.
    ids[:, topk:] = base + torch.arange(num_shared, dtype=dtypes.i32)
    logit = torch.empty(tokens, num_shared, dtype=dtype)  # reused, not reallocated

    def fn():
        topk_softmax(w_routed, ids_routed, tei, gating, renorm)
        torch.mm(hs, gate_weight.t(), out=logit)  # bf16 GEMV, no upcast
        torch.sigmoid(logit, out=logit)
        w[:, topk:] = logit * scale

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
    # token_expert_indices is write-only scratch, written with stride topk (routed only).
    tei = torch.empty(tokens, topk, dtype=dtypes.i32)

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


# Each case varies ONE axis from the baseline (tokens=64, experts=512, num_shared=1,
# bf16, hidden=4096, scale=1.0, renorm=True) so a failure points at the branch it
# exercises. Covered: shared-expert counts (incl. 8), fp32 gating, a larger hidden that
# exceeds the gate-LDS staging cap (global-read fallback), the need_renorm=False
# epilogue, a non-unit shared_expert_scale, num_experts=256 (BYTES_PER_LDG=32 GEMV
# vector width), and token-count edges M=1 and M=65 (tail block, not a multiple of
# ROWS_PER_CTA=8 at E=512).
# (tokens, num_experts, num_shared, dtype, hidden, scale, renorm)
_CORRECTNESS_CASES = [
    (64, 512, 1, dtypes.bf16, 4096, 1.0, True),  # baseline
    (64, 512, 2, dtypes.bf16, 4096, 1.0, True),  # num_shared=2
    (64, 512, 8, dtypes.bf16, 4096, 1.0, True),  # num_shared=8
    (64, 512, 1, dtypes.fp32, 4096, 1.0, True),  # fp32 gating
    (64, 512, 1, dtypes.bf16, 8192, 1.0, True),  # H=8192 gate-LDS global fallback
    (64, 512, 1, dtypes.bf16, 4096, 1.0, False),  # need_renorm=False epilogue
    (64, 512, 1, dtypes.bf16, 4096, 0.5, True),  # shared_expert_scale != 1.0
    (64, 256, 1, dtypes.bf16, 4096, 1.0, True),  # E=256 -> BYTES_PER_LDG=32 path
    (1, 512, 1, dtypes.bf16, 4096, 1.0, True),  # M=1
    (65, 512, 1, dtypes.bf16, 4096, 1.0, True),  # M=65 tail block (not mult of 8)
]


@pytest.mark.skipif(not _gfx_supported(), reason="requires an AMD GPU (gfx942/gfx950)")
@pytest.mark.parametrize(
    "tokens, num_experts, num_shared, dtype, hidden, scale, renorm", _CORRECTNESS_CASES
)
def test_correctness(tokens, num_experts, num_shared, dtype, hidden, scale, renorm):
    """pytest exercises real cases (not just bench_*): the fused op and the unfused
    baseline must both match the torch reference across shared-expert counts, dtypes,
    hidden sizes (incl. the larger-H gate-LDS global fallback), the renorm/scale
    epilogue variants, num_experts=256, and the M=1 / M=65 token-count edges."""
    topk = 10
    base = num_experts
    total = topk + num_shared
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtype
    )
    ref = run_torch(gating, hs, gate_weight, topk, num_shared, base, scale, renorm)

    w = torch.empty(tokens, total, dtype=dtypes.fp32)
    ids = torch.empty(tokens, total, dtype=dtypes.i32)
    tei = torch.empty(tokens, topk, dtype=dtypes.i32)  # scratch, stride topk
    topk_softmax_fused_shared_gate(
        w, ids, tei, gating, renorm, num_shared, "sigmoid", hs, gate_weight, scale, base
    )
    _check(w, ids, topk, ref, "fused")

    w_u = torch.empty(tokens, total, dtype=dtypes.fp32)
    ids_u = torch.empty(tokens, total, dtype=dtypes.i32)
    r_w = torch.empty(tokens, topk, dtype=dtypes.fp32)
    r_i = torch.empty(tokens, topk, dtype=dtypes.i32)
    r_tei = torch.empty(tokens, topk, dtype=dtypes.i32)
    shared_ids = (
        (base + torch.arange(num_shared, dtype=dtypes.i32))
        .unsqueeze(0)
        .expand(tokens, num_shared)
    )
    topk_softmax(r_w, r_i, r_tei, gating, renorm)
    shared_w = torch.sigmoid(hs.float() @ gate_weight.float().t()) * scale
    w_u[:, :topk], w_u[:, topk:] = r_w, shared_w
    ids_u[:, :topk], ids_u[:, topk:] = r_i, shared_ids
    _check(w_u, ids_u, topk, ref, "unfused")


def _make_fused_buffers(tokens, topk, num_shared):
    total = topk + num_shared
    return (
        torch.empty(tokens, total, dtype=dtypes.fp32),
        torch.empty(tokens, total, dtype=dtypes.i32),
        torch.empty(tokens, topk, dtype=dtypes.i32),  # token_expert_indices scratch
    )


@pytest.mark.skipif(not _gfx_supported(), reason="requires an AMD GPU (gfx942/gfx950)")
def test_misaligned_hidden_raises():
    """A hidden dim whose row byte-stride is not 64B-aligned must be rejected: otherwise
    the gate GEMV's vectorized loads are misaligned on every row after the first (UB).
    """
    tokens, num_experts, topk, num_shared = 64, 512, 10, 1
    hidden = 4008  # 4008 * 2B = 8016 B, not a multiple of 64
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtypes.bf16
    )
    w, ids, tei = _make_fused_buffers(tokens, topk, num_shared)
    with pytest.raises(RuntimeError, match="64B-aligned"):
        topk_softmax_fused_shared_gate(
            w,
            ids,
            tei,
            gating,
            True,
            num_shared,
            "sigmoid",
            hs,
            gate_weight,
            1.0,
            num_experts,
        )


@pytest.mark.skipif(not _gfx_supported(), reason="requires an AMD GPU (gfx942/gfx950)")
def test_non_power_of_2_experts_raises():
    """The fused op supports only power-of-2 num_experts (<= 512); a non-power-of-2
    routing-expert count must be rejected, not run the removed serial fallback."""
    tokens, num_experts, topk, num_shared = 64, 384, 10, 1
    hidden = 4096
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtypes.bf16
    )
    w, ids, tei = _make_fused_buffers(tokens, topk, num_shared)
    with pytest.raises(RuntimeError, match="power-of-2"):
        topk_softmax_fused_shared_gate(
            w,
            ids,
            tei,
            gating,
            True,
            num_shared,
            "sigmoid",
            hs,
            gate_weight,
            1.0,
            num_experts,
        )


@pytest.mark.skipif(not _gfx_supported(), reason="requires an AMD GPU (gfx942/gfx950)")
def test_zero_tokens_noop():
    """An empty batch (M=0) must be a guarded no-op: without the host-op guard the
    launcher computes num_blocks == 0 and the grid launch fails."""
    tokens, num_experts, topk, num_shared = 0, 512, 10, 1
    hidden = 4096
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtypes.bf16
    )
    w, ids, tei = _make_fused_buffers(tokens, topk, num_shared)
    topk_softmax_fused_shared_gate(
        w,
        ids,
        tei,
        gating,
        True,
        num_shared,
        "sigmoid",
        hs,
        gate_weight,
        1.0,
        num_experts,
    )
    assert w.shape[0] == 0 and ids.shape[0] == 0


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
    # Sweep decode -> prefill so the fused-vs-unfused crossover is visible: the
    # in-kernel gate GEMV adds a full [M, hidden] HBM pass, so the fused win
    # shrinks as M grows.
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 4, 16, 64, 256, 1024, 4096, 16384],
    )
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
