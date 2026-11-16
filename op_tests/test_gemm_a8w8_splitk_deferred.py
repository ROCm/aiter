# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 PTPC bpreshuffle GEMM with its split-K reduction deferred.

gemm_a8w8_bpreshuffle_splitk_deferred skips the standalone split-K reduce of a
decode-size GEMM and leaves the fp32 partials (splitk_partials) to the kernel that
reads the output. Every check is bitwise against gemm_a8w8_bpreshuffle (GEMM + reduce):

* deferred_gemm: deferred GEMM + reduce_splitk_partials (MiniMax-M3 TP4 sparse QKV,
  dense QKV, dense gate/up).
* claim: the partials are handed out once; reduce_splitk_partials claims them itself,
  and returns a tensor that holds its values as it is.

Runs as `python3 <file>` (summary table) or under pytest (same cases and checks).
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # the split-K PTPC GEMM is the gfx1250 WMMA backend
MAX_M = 64  # largest M the deferred path takes (its max_m default)
M_LIST = [1, 4, 8, 16, 32, 48, 64]  # decode tokens per TP4 rank
NK_LIST = [(2560, 6144), (2304, 6144), (6144, 6144)]  # sparse QKV, dense QKV, gate/up

pytestmark = pytest.mark.skipif(
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
    partials = aiter.splitk_partials(y)
    ret = {"gfx": get_gfx_runtime(), "split_k": split_of(partials)}
    ret["bitwise"] = bitwise_equal(aiter.reduce_splitk_partials(y, partials), base)
    assert ret["bitwise"], f"deferred GEMM differs from gemm_a8w8_bpreshuffle {m=} {n=}"
    candidates = {
        "gemm+reduce": lambda: aiter.gemm_a8w8_bpreshuffle(*args),
        "deferred+reduce": lambda: aiter.reduce_splitk_partials(
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
            aiter.reduce_splitk_partials(out)  # the GEMM alone: its partials unread
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


@pytest.mark.parametrize("n,k", NK_LIST)
def test_deferred_gemm(n, k):
    run_suite(deferred_gemm, [(m, n, k) for m in M_LIST])


def test_claim():
    for m in M_LIST:
        args, _ = gemm_inputs(m, *NK_LIST[0])
        y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
        partials = aiter.splitk_partials(y)
        if partials is not None:
            break
    else:
        pytest.fail(f"no M in {M_LIST} splits K for {NK_LIST[0]}")
    base = aiter.gemm_a8w8_bpreshuffle(*args)
    assert aiter.splitk_partials(y) is None, "partials handed out twice"
    assert bitwise_equal(aiter.reduce_splitk_partials(y, partials), base)
    y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
    assert aiter.reduce_splitk_partials(y) is y and bitwise_equal(y, base)
    assert aiter.reduce_splitk_partials(base) is base  # holds its values already


def main():
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
