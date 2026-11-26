# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 PTPC bpreshuffle GEMM with its split-K reduction deferred.

gemm_a8w8_bpreshuffle_splitk_deferred skips the standalone split-K reduce of a
decode-size GEMM and leaves the fp32 partials to the kernel that reads the output.

* test_gemm_a8w8_splitk_deferred (gfx1250, summary table): on MiniMax-M3 TP4
  sparse QKV, dense QKV and dense gate/up, the deferred GEMM finished by
  gemm_a8w8_bpreshuffle_splitk_reduce equals gemm_a8w8_bpreshuffle bit for bit,
  and both match an fp32 torch reference.
* check_claim (gfx1250): the partials are handed out once; the reduce claims them
  itself, and returns a tensor that already holds its values as it is.
* check_handoff_table (every arch, host only; both FlyDSL launchers stubbed):
  claim once, a leading row slice claims, any other view of a deferred output
  raises, the oldest unclaimed output is reduced when the table is full, and off
  gfx1250 the deferred GEMM returns gemm_a8w8_bpreshuffle's output.
"""

import argparse
import contextlib
import itertools
from unittest import mock

import pandas as pd
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
# Against the fp32 reference, per element: 1% of the value (a bf16 output rounds
# within 0.4%) plus 0.1 absolute for outputs near zero, about 1e-3 of the output's
# RMS (sqrt(K) for unit-variance operands).
RTOL, ATOL = 1e-2, 0.1

deferred_gemm = aiter.gemm_a8w8_bpreshuffle_splitk_deferred
splitk_partials = aiter.gemm_a8w8_bpreshuffle_splitk_partials
splitk_reduce = aiter.gemm_a8w8_bpreshuffle_splitk_reduce


def run_torch(xq, wq, x_scale, w_scale):
    # Reference only: fp32 math on the dequantized operands. Not timed.
    return (xq.to(dtypes.fp32) * x_scale) @ (wq.to(dtypes.fp32) * w_scale).t()


def bitwise_equal(a, b):
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def gemm_inputs(m, n, k):
    """fp8 per-token operands, the weight preshuffled as the model stores it, and
    the fp32 reference. Distinct scales per row of x and of the weight."""
    torch.manual_seed(0)
    xq, x_scale = aiter.pertoken_quant(
        torch.randn(m, k, dtype=dtypes.bf16), quant_dtype=dtypes.fp8
    )
    wq, w_scale = aiter.pertoken_quant(
        torch.randn(n, k, dtype=dtypes.bf16), quant_dtype=dtypes.fp8
    )
    ref = run_torch(xq, wq, x_scale, w_scale)
    return (xq, shuffle_weight(wq, layout=(16, 16)), x_scale, w_scale), ref


@benchmark()
def test_gemm_a8w8_splitk_deferred(m, n, k):
    args, ref = gemm_inputs(m, n, k)
    base = aiter.gemm_a8w8_bpreshuffle(*args)
    partials = splitk_partials(deferred_gemm(*args, max_m=MAX_M))
    candidates = {
        "gemm+reduce": lambda: aiter.gemm_a8w8_bpreshuffle(*args),
        "deferred+reduce": lambda: splitk_reduce(deferred_gemm(*args, max_m=MAX_M)),
        # The GEMM alone, as the model launches it: its partials are left for the
        # consumer, so the output is finished below, outside the timing.
        "deferred": lambda: deferred_gemm(*args, max_m=MAX_M),
    }
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * args[0].element_size() + m * n * base.element_size()

    ret = {
        "gfx": get_gfx_runtime(),
        "split_k": 1 if partials is None else partials.shape[0],
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        out = splitk_reduce(out)  # a finished output is returned as it is
        assert bitwise_equal(out, base), f"{name} differs from gemm+reduce {m=} {n=}"
        err = checkAllclose(ref, out.to(dtypes.fp32), rtol=RTOL, atol=ATOL, msg=name)
        assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def check_claim(m, n, k):
    """A deferred output's partials are claimed once, by the caller or by the reduce."""
    args, _ = gemm_inputs(m, n, k)
    base = aiter.gemm_a8w8_bpreshuffle(*args)
    y = deferred_gemm(*args, max_m=MAX_M)
    partials = splitk_partials(y)
    assert partials is not None, f"no split K at {m=} {n=} {k=}"
    assert splitk_partials(y) is None, "partials handed out twice"
    assert bitwise_equal(splitk_reduce(y, partials), base)
    y = deferred_gemm(*args, max_m=MAX_M)
    assert splitk_reduce(y) is y and bitwise_equal(y, base)
    assert splitk_reduce(base) is base  # holds its values already


def check_handoff_table():
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

    def raises_value_error(view):
        try:
            splitk_partials(view)
        except ValueError:
            return True
        return False

    xq = torch.zeros(m, 128, dtype=dtypes.fp8, device="cpu")
    wq = torch.zeros(n, 128, dtype=dtypes.fp8, device="cpu")
    # An injected tuned row, so the check does not depend on the tuned CSVs.
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
        return deferred_gemm(xq, wq, None, None)

    with contextlib.ExitStack() as stack:
        for stub in stubs:
            stack.enter_context(stub)
        y = deferred()
        partials = splitk_partials(y[: m // 2])  # leading row slice
        assert partials.shape == (split_k, m, n) and partials[-1].eq(split_k).all()
        assert splitk_partials(y) is None, "partials handed out twice"
        y = deferred()
        for view in (y.view(m, 2, n // 2), y[:, : n // 2], y[1:], y[:, 8:], y.t()):
            assert raises_value_error(view), f"view {tuple(view.shape)} was claimed"
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
    check_handoff_table()
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "split-K deferral unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(m, n, k) for (n, k), m in itertools.product(NK_LIST, M_LIST)],
        help="shapes of the GEMM as m,n,k\n    e.g.: -s 16,2560,6144",
    )
    args = parser.parse_args()

    df = pd.DataFrame([test_gemm_a8w8_splitk_deferred(*mnk) for mnk in args.mnk])
    aiter.logger.info(
        "gemm_a8w8_bpreshuffle_splitk_deferred summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    # A split_k 1 row compares gemm_a8w8_bpreshuffle with itself: when some M can
    # defer, the tuned rows must still split K somewhere, and the claim check
    # runs on the first shape that does.
    split = df[(df["m"] <= MAX_M) & (df["split_k"] > 1)]
    if (df["m"] <= MAX_M).any():
        assert len(split), f"no shape split K: {args.mnk}"
        check_claim(*(int(v) for v in split.iloc[0][["m", "n", "k"]]))


if __name__ == "__main__":
    main()
