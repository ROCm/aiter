# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 PTPC bpreshuffle GEMM with its split-K reduction deferred.

gemm_a8w8_bpreshuffle_splitk_deferred skips the standalone split-K reduce of a
decode-size GEMM and leaves the fp32 partials (gemm_a8w8_bpreshuffle_splitk_partials,
`splitk_partials` here) to the kernel that reads the output. On gfx1250, outputs
are checked bitwise against gemm_a8w8_bpreshuffle (GEMM + reduce); the err column
is an allclose check against an fp32 torch reference:

* deferred_gemm: deferred GEMM + gemm_a8w8_bpreshuffle_splitk_reduce (`splitk_reduce`
  here) on MiniMax-M3 TP4 sparse QKV, dense QKV and dense gate/up.
* claim: the partials are handed out once; splitk_reduce claims them itself, and
  returns a tensor that holds its values as it is.
* handoff_table (host only, every arch; both FlyDSL launchers stubbed): claim once,
  a leading row slice claims, any other view of a deferred output raises, the
  oldest unclaimed output is reduced when the table is full, and off gfx1250
  the deferred GEMM returns gemm_a8w8_bpreshuffle's output.

Runs as `python3 <file>` (summary table) or under pytest (same cases and checks).
"""

import argparse
import contextlib
import itertools
from unittest import mock

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import gemm_op_a8w8
from aiter.ops.flydsl import bpreshuffle_gemm_gfx1250
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # the split-K PTPC GEMM is the gfx1250 WMMA backend
MAX_M = 64  # largest M the deferred path takes (its max_m default)
M_LIST = [1, 4, 8, 16, 32, 48, 64]  # decode tokens per TP4 rank
NK_LIST = [(2560, 6144), (2304, 6144), (6144, 6144)]  # sparse QKV, dense QKV, gate/up

splitk_partials = aiter.gemm_a8w8_bpreshuffle_splitk_partials
splitk_reduce = aiter.gemm_a8w8_bpreshuffle_splitk_reduce

gfx1250_only = pytest.mark.skipif(
    get_gfx_runtime() not in SUPPORTED_GFX, reason="gfx1250 only"
)


def bitwise_equal(a, b):
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def split_of(partials):
    return 1 if partials is None else partials.shape[0]


def gemm_inputs(m, n, k):
    """fp8 per-token operands (weight preshuffled) and an fp32 torch reference."""
    torch.manual_seed(0)
    xq, xs = aiter.pertoken_quant(
        torch.randn(m, k, dtype=dtypes.bf16), quant_dtype=dtypes.fp8
    )
    wq, ws = aiter.pertoken_quant(
        torch.randn(n, k, dtype=dtypes.bf16), quant_dtype=dtypes.fp8
    )
    ref = ((xq.float() * xs) @ (wq.float() * ws).t()).to(dtypes.bf16)
    return (xq, shuffle_weight(wq, layout=(16, 16)), xs, ws), ref


@benchmark()
def deferred_gemm(m, n, k):
    args, ref = gemm_inputs(m, n, k)
    base = aiter.gemm_a8w8_bpreshuffle(*args)
    y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
    partials = splitk_partials(y)
    ret = {"gfx": get_gfx_runtime(), "split_k": split_of(partials)}
    ret["bitwise"] = bitwise_equal(splitk_reduce(y, partials), base)
    assert ret["bitwise"], f"deferred GEMM differs from gemm_a8w8_bpreshuffle {m=} {n=}"
    candidates = {
        "gemm+reduce": lambda: aiter.gemm_a8w8_bpreshuffle(*args),
        "deferred+reduce": lambda: splitk_reduce(
            aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
        ),
        "deferred": lambda: aiter.gemm_a8w8_bpreshuffle_splitk_deferred(
            *args, max_m=MAX_M
        ),
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        if name == "deferred":
            splitk_reduce(out)  # the GEMM alone: its partials unread
            continue
        assert bitwise_equal(out, base), f"{name} differs from gemm+reduce {m=} {n=}"
        ret[f"{name} TFLOPS"] = 2 * m * n * k / us / 1e6
        ret[f"{name} err"] = checkAllclose(ref.float(), out.float(), msg=name)
    return ret


def run_suite(case, params, title=None):
    """Run case(*p) for every p; with a title, log the summary table. A split_k 1
    row compares gemm_a8w8_bpreshuffle with itself, so when some M is deferrable,
    require that the tuned configs still split K somewhere."""
    rows = [case(*p) for p in params]
    if title:
        table = pd.DataFrame(rows).to_markdown(index=False)
        aiter.logger.info("%s summary (markdown):\n%s", title, table)
    if any(r["m"] <= MAX_M for r in rows):
        assert any(r["split_k"] > 1 for r in rows), f"no case split K: {params}"
    return rows


@gfx1250_only
@pytest.mark.parametrize("n,k", NK_LIST)
def test_deferred_gemm(n, k):
    run_suite(deferred_gemm, [(m, n, k) for m in M_LIST])


@gfx1250_only
def test_claim():
    for m in M_LIST:
        args, _ = gemm_inputs(m, *NK_LIST[0])
        y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
        partials = splitk_partials(y)
        if partials is not None:
            break
    else:
        pytest.fail(f"no M in {M_LIST} splits K for {NK_LIST[0]}")
    base = aiter.gemm_a8w8_bpreshuffle(*args)
    assert splitk_partials(y) is None, "partials handed out twice"
    assert bitwise_equal(splitk_reduce(y, partials), base)
    y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
    assert splitk_reduce(y) is y and bitwise_equal(y, base)
    assert splitk_reduce(base) is base  # holds its values already


def test_handoff_table():
    """The hand-off table on the host: the GEMM stub writes partials s + 1 and the
    reduce stub sums them, so a finished [m, n] output holds split_k * (split_k + 1) / 2.
    """
    split_k, m, n = 4, 16, 64
    done = split_k * (split_k + 1) / 2

    def gemm(xq, wq, xs, ws, out, kernel_name, partials):
        partials.copy_(
            torch.arange(1, split_k + 1, device=partials.device)
            .view(-1, 1, 1)
            .expand_as(partials)
        )

    def reduce(partials, out):
        out.copy_(partials.sum(0))

    xq = torch.zeros(m, 128, dtype=dtypes.fp8, device="cpu")
    wq = torch.zeros(n, 128, dtype=dtypes.fp8, device="cpu")
    row = {"libtype": "flydsl"}
    row["kernelName"] = (
        f"flydsl_bpreshuffle_wmma_t16x64x512_mw1_nw4_nb3_sk{split_k}_cm1_cn1"
    )
    stubs = [
        mock.patch.dict(gemm_op_a8w8._SPLITK_DEFERRED, clear=True),
        mock.patch.object(gemm_op_a8w8, "get_gfx", return_value="gfx1250"),
        mock.patch.object(gemm_op_a8w8, "_bpreshuffle_config", return_value=row),
        mock.patch.object(
            bpreshuffle_gemm_gfx1250, "run_gemm_a8w8_bpreshuffle_gfx1250", gemm
        ),
        mock.patch.object(
            bpreshuffle_gemm_gfx1250, "reduce_splitk_partials_gfx1250", reduce
        ),
    ]

    def deferred():
        return aiter.gemm_a8w8_bpreshuffle_splitk_deferred(xq, wq, None, None)

    with contextlib.ExitStack() as stack:
        for stub in stubs:
            stack.enter_context(stub)
        y = deferred()
        partials = splitk_partials(y[: m // 2])  # leading row slice
        assert partials.shape == (split_k, m, n) and partials[-1].eq(split_k).all()
        assert splitk_partials(y) is None, "partials handed out twice"
        y = deferred()
        for view in (y.view(m, 2, n // 2), y[:, : n // 2], y[1:], y[:, 8:], y.t()):
            with pytest.raises(ValueError):
                splitk_partials(view)
        assert splitk_reduce(y) is y and y.eq(done).all()
        assert splitk_reduce(y) is y  # holds its values already
        ys = [deferred() for _ in range(gemm_op_a8w8._SPLITK_DEFERRED_MAX + 1)]
        assert ys[0].eq(done).all(), "evicted output left unreduced"
        assert splitk_partials(ys[0]) is None
        assert all(splitk_partials(y) is not None for y in ys[1:])
        plain = torch.zeros(m, n, dtype=dtypes.bf16, device="cpu")
        stack.enter_context(
            mock.patch.object(gemm_op_a8w8, "get_gfx", return_value="gfx950")
        )
        stack.enter_context(
            mock.patch.object(gemm_op_a8w8, "gemm_a8w8_bpreshuffle", return_value=plain)
        )
        assert deferred() is plain and splitk_partials(plain) is None


def main():
    test_handoff_table()
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("split-K deferral unsupported on %s", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(description="gfx1250 deferred split-K GEMM")
    parser.add_argument("-m", type=int, nargs="*", default=M_LIST)
    parser.add_argument("-s", "--nk", type=dtypes.str2tuple, nargs="*", default=NK_LIST)
    args = parser.parse_args()
    nk = [(m, n, k) for (n, k), m in itertools.product(args.nk, args.m)]
    run_suite(deferred_gemm, nk, "deferred GEMM + reduce")
    test_claim()


if __name__ == "__main__":
    main()
