# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Op test for aiter.topk_softmax_fused_shared_gate (Option A: routed softmax
# top-k + in-kernel shared-expert gate GEMV in a single launch).
#
# Structure follows the aiter op_test standard (see op_tests/test_quant.py):
#   * ONE @benchmark function times BOTH paths as candidates in one table --
#       "unfused": routed topk_softmax + a separate bf16 gate GEMV + append
#       "fused"  : topk_softmax_fused_shared_gate (single launch)
#     so the fused-vs-unfused comparison is read straight off the `us` columns.
#   * Correctness IS the `err` column: each candidate is checked against a torch
#     reference on every swept shape. Because the file runs via `python3 <file>`
#     (which is how CI invokes op_tests -- NOT pytest), this correctness check
#     runs in CI. Extra edge configs and the negative/guard behaviors are also
#     driven from main() for the same reason.

import argparse
import itertools

import aiter
import pandas as pd
import torch
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


def _fused_buffers(tokens, topk, num_shared):
    total = topk + num_shared
    return (
        torch.empty(tokens, total, dtype=dtypes.fp32),
        torch.empty(tokens, total, dtype=dtypes.i32),
        torch.empty(tokens, topk, dtype=dtypes.i32),  # token_expert_indices scratch
    )


def _check(wbuf, ibuf, topk, ref, name):
    # Compare one candidate's output buffers against the torch reference and return
    # the WORST of the four checks, so a mismatch in ANY of them (ids to the bit,
    # weights within tol) surfaces as a non-zero err the caller can assert on.
    ref_rw, ref_ri, ref_sw, ref_si = ref
    got_rw, got_ri = wbuf[:, :topk], ibuf[:, :topk]
    got_sw, got_si = wbuf[:, topk:], ibuf[:, topk:]
    e_sid = checkAllclose(
        got_si.to(dtypes.fp32),
        ref_si.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg=f"{name} shared ids",
    )
    e_sw = checkAllclose(
        got_sw.to(dtypes.fp32),
        ref_sw.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{name} shared weights",
    )
    ref_ids, ref_w = sorted_pairs(ref_ri, ref_rw)
    got_ids, got_w = sorted_pairs(got_ri.to(dtypes.i32), got_rw)
    e_rid = checkAllclose(
        got_ids.to(dtypes.fp32),
        ref_ids.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg=f"{name} routed ids",
    )
    e_rw = checkAllclose(
        got_w.to(dtypes.fp32),
        ref_w.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{name} routed weights",
    )
    return max(e_sid, e_sw, e_rid, e_rw)


@benchmark()
def test_shared_gate(
    tokens, num_experts, hidden, topk, num_shared, scale, renorm, dtype
):
    """Fused vs unfused shared-expert gating in one table. Both candidates are
    checked against a torch reference (the `err` columns) -- that IS the
    correctness check, and it runs under `python3 <file>` (CI), not just pytest.

    unfused: routed topk_softmax writing routed columns in place via strided
             views + a separate bf16 gate GEMV (no fp32 upcast) + sigmoid/scale.
    fused  : topk_softmax_fused_shared_gate (single launch).
    """
    base = num_experts
    total = topk + num_shared
    gating, hs, gate_weight = _make_inputs(
        tokens, num_experts, hidden, num_shared, dtype
    )
    ref = run_torch(gating, hs, gate_weight, topk, num_shared, base, scale, renorm)

    # --- unfused candidate ---------------------------------------------------
    w_u = torch.empty(tokens, total, dtype=dtypes.fp32)
    ids_u = torch.empty(tokens, total, dtype=dtypes.i32)
    tei_u = torch.empty(tokens, topk, dtype=dtypes.i32)
    wr_u, ir_u = w_u[:, :topk], ids_u[:, :topk]  # views, written in place
    # shared ids depend only on (base, num_shared) -> precompute once.
    ids_u[:, topk:] = base + torch.arange(num_shared, dtype=dtypes.i32)
    logit = torch.empty(tokens, num_shared, dtype=dtype)  # reused

    def run_unfused():
        topk_softmax(wr_u, ir_u, tei_u, gating, renorm)
        torch.mm(hs, gate_weight.t(), out=logit)  # bf16 GEMV, no upcast
        torch.sigmoid(logit, out=logit)
        w_u[:, topk:] = logit * scale

    # --- fused candidate -----------------------------------------------------
    w_f, ids_f, tei_f = _fused_buffers(tokens, topk, num_shared)

    def run_fused():
        topk_softmax_fused_shared_gate(
            w_f,
            ids_f,
            tei_f,
            gating,
            renorm,
            num_shared,
            "sigmoid",
            hs,
            gate_weight,
            scale,
            base,
        )

    candidates = {
        "unfused": (run_unfused, w_u, ids_u),
        "fused": (run_fused, w_f, ids_f),
    }

    # Roofline: dominant work is the shared-gate GEMV over [M, hidden].
    flops = 2 * tokens * num_shared * hidden
    nbytes = (
        gating.numel() * gating.element_size()
        + hs.numel() * hs.element_size()
        + gate_weight.numel() * gate_weight.element_size()
        + w_f.numel() * w_f.element_size()
        + ids_f.numel() * ids_f.element_size()
    )

    ret = {"gfx": get_gfx()}
    for name, (fn, wbuf, ibuf) in candidates.items():
        _, us = run_perftest(fn)
        err = _check(wbuf, ibuf, topk, ref, name)
        # Gate CI on correctness: the runner judges pass/fail by this script's exit
        # code (checkAllclose only logs), so a wrong result must raise here.
        assert err == 0, (
            f"{name} correctness failed: err={err} "
            f"(tokens={tokens}, num_experts={num_experts}, hidden={hidden}, "
            f"num_shared={num_shared}, scale={scale}, renorm={renorm}, dtype={dtype})"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# Edge configs the Qwen perf sweep (E=512, H=4096, bf16) does not reach, each
# picked to exercise a distinct code path. Same @benchmark fn -> same err check.
# (tokens, num_experts, hidden, topk, num_shared, scale, renorm, dtype)
_CORRECTNESS_EDGES = [
    (64, 256, 4096, 10, 1, 1.0, True, dtypes.bf16),  # E=256 -> BYTES_PER_LDG=32
    (64, 512, 8192, 10, 1, 1.0, True, dtypes.bf16),  # H=8192 gate-LDS global fallback
    (64, 512, 4096, 10, 1, 1.0, True, dtypes.fp32),  # fp32 gating
    (64, 512, 4096, 10, 8, 1.0, True, dtypes.bf16),  # num_shared=8
    (65, 512, 4096, 10, 1, 1.0, True, dtypes.bf16),  # M=65 tail block (not mult of 8)
]


def _expect_raises(fn, needle, desc):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), f"{desc}: unexpected error message: {e}"
        return
    raise AssertionError(f"{desc}: expected RuntimeError containing {needle!r}")


def _sweep(args, dtype):
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
            test_shared_gate(
                tokens, experts, hidden, topk, num_shared, scale, bool(renorm), dtype
            )
        )
    return pd.DataFrame(df)


def _run_guard_checks():
    """Negative/guard behaviors. These raise or no-op rather than emit a table
    row, so they run here as asserts (under `python3 <file>`, i.e. in CI)."""
    topk, num_shared, num_experts = 10, 1, 512

    # A hidden dim whose row byte-stride is not 64B-aligned must be rejected.
    gating, hs, gw = _make_inputs(64, num_experts, 4008, num_shared, dtypes.bf16)
    w, ids, tei = _fused_buffers(64, topk, num_shared)
    _expect_raises(
        lambda: topk_softmax_fused_shared_gate(
            w, ids, tei, gating, True, num_shared, "sigmoid", hs, gw, 1.0, num_experts
        ),
        "64B-aligned",
        "misaligned hidden",
    )

    # Non-power-of-2 num_experts must be rejected (no serial fallback).
    gating, hs, gw = _make_inputs(64, 384, 4096, num_shared, dtypes.bf16)
    w, ids, tei = _fused_buffers(64, topk, num_shared)
    _expect_raises(
        lambda: topk_softmax_fused_shared_gate(
            w, ids, tei, gating, True, num_shared, "sigmoid", hs, gw, 1.0, 384
        ),
        "power-of-2",
        "non-power-of-2 experts",
    )

    # M=0 must be a guarded no-op (would otherwise launch a 0-block grid).
    gating, hs, gw = _make_inputs(0, num_experts, 4096, num_shared, dtypes.bf16)
    w, ids, tei = _fused_buffers(0, topk, num_shared)
    topk_softmax_fused_shared_gate(
        w, ids, tei, gating, True, num_shared, "sigmoid", hs, gw, 1.0, num_experts
    )
    assert w.shape[0] == 0 and ids.shape[0] == 0, "M=0 should be a no-op"

    aiter.logger.info(
        "guard checks passed: misaligned-raise, non-power-of-2-raise, M=0 no-op"
    )


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
        df = _sweep(args, dtype)
        aiter.logger.info(
            "fused vs unfused shared-gate (%s):\n%s",
            dtype,
            df.to_markdown(index=False),
        )

    # Extra correctness-only edge configs the perf sweep does not reach.
    edge_df = pd.DataFrame([test_shared_gate(*cfg) for cfg in _CORRECTNESS_EDGES])
    aiter.logger.info(
        "shared-gate correctness edges:\n%s", edge_df.to_markdown(index=False)
    )

    # Negative/guard behaviors (raise or no-op) -- asserts, run in CI too.
    _run_guard_checks()


if __name__ == "__main__":
    main()
