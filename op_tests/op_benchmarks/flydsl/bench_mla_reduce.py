# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA reduce benchmark.

Stage-2 of split-KV MLA decode: merges per-split partial outputs ``O_i`` weighted
by ``exp(LSE_i - LSE_max)`` (online softmax) into the final bf16/fp16 output.

The sweep is the GLM-5.2 serving decode grid: a sparse 16384-tile reduce grid
with active ``(active_tiles, splits)`` decode buckets. ``graph us`` is CUDA-graph
replay latency; ``TFLOPS`` / ``TB/s`` are derived from it.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.ops.flydsl import flydsl_mla_reduce_v1
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx

from op_tests.flydsl_mla_reduce_common import (
    MLA_REDUCE_SUPPORTED_GFX,
    bench_cudagraph,
    build_serving_decode_inputs,
    mla_reduce_out_atol,
    torch_ref_gather,
)

torch.set_default_device("cuda")

# (active_tiles, splits) serving decode buckets: 1 tile x 128 splits exercises
# the split-K path; 8 tiles x N splits exercises the sparse adaptive launch.
_SERVING_SCENARIOS = [
    (1, 128),
    (8, 32),
    (8, 26),
    (8, 13),
    (8, 6),
    (8, 5),
    (8, 3),
    (8, 2),
]


def _roofline(active, splits, H, Dv, out_dtype):
    """FLOPs (online-softmax weighted-sum FMA) and byte traffic for the reduce."""
    total_splits = active * splits
    out_bytes = torch.finfo(out_dtype).bits // 8
    flops = 2 * total_splits * H * Dv
    nbytes = (
        total_splits * H * Dv * 4  # partial_output fp32 read
        + total_splits * H * 4  # partial_lse   fp32 read
        + active * H * Dv * out_bytes  # final_output  write
        + active * H * 4  # final_lse     fp32 write
    )
    return flops, nbytes


@benchmark()
def test_mla_reduce(active, splits, H, Dv, dtype):
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_decode_inputs(
        active, splits, dtype, H=H, Dv=Dv
    )

    # torch online-softmax reference over the active prefix only (tail tiles are
    # empty and skipped by both the kernels and the ref).
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr[: active + 1], fmap[:active], pmap, H, Dv, dtype, M=1
    )

    candidates = {
        # mla_reduce_v1 signature: (..., max_seqlen_q, num_kv_splits, out, lse)
        "hip": lambda: aiter.mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, 0, fout, flse
        ),
        "wrapper": lambda: flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=splits
        ),
    }

    flops, nbytes = _roofline(active, splits, H, Dv, dtype)

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        fout.zero_()
        flse.zero_()
        _, us = run_perftest(fn, num_warmup=25, num_iters=100)
        out = fout.clone()
        lse = flse.clone()
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=mla_reduce_out_atol(dtype),
            msg=f"{name}: mla_reduce out",
            printLog=False,
        )
        checkAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-3,
            msg=f"{name}: mla_reduce lse",
            printLog=False,
        )
        # CUDA-graph replay µs (serving path): host dispatch captured once and
        # amortized away. TFLOPS/TB/s are derived from this, not eager us.
        graph_us = bench_cudagraph(fn) * 1e3
        ret[f"{name} us"] = us
        ret[f"{name} graph us"] = graph_us
        ret[f"{name} TFLOPS"] = flops / graph_us / 1e6
        ret[f"{name} TB/s"] = nbytes / graph_us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in MLA_REDUCE_SUPPORTED_GFX:
        aiter.logger.warning("mla_reduce unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="Output data type, e.g. -d bf16",
    )
    parser.add_argument(
        "--hdv",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(16, 512)],
        help="(H, Dv) shape, e.g. --hdv 16,512 128,512",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=dtypes.str2tuple,
        nargs="*",
        default=_SERVING_SCENARIOS,
        help="(active_tiles, splits) decode buckets, e.g. -s 1,128 8,32",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for (H, Dv), (active, splits) in itertools.product(args.hdv, args.scenario):
            df.append(test_mla_reduce(active, splits, H, Dv, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mla_reduce GLM-5.2 serving summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
