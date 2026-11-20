# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf and correctness for FlyDSL MXFP4 paged decode on gfx950.

The MXFP4 tile kernel (``single_query_tile_mxfp4``) widens V to bf16 and issues
one ``16x16x16bf16`` MFMA per 16-token chunk. This sweep times that kernel as
``pa_decode`` launches it: plan-free, one task per sequence, direct bf16 output.
Head dim 128, page 128 and query length 1 are the only geometry it accepts.

The pytest module ``test_flydsl_pa_decode.py`` is the correctness suite and is
run with pytest, so this sweep lives here where CI executes ``python3``.

    python op_tests/test_flydsl_pa_decode_mxfp4.py
    python op_tests/test_flydsl_pa_decode_mxfp4.py -b 1 -c 257 -d bf16 -g 16 -k 1
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.pa_decode import pa_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from op_tests.test_flydsl_pa_decode import (  # noqa: E402
    _make_inputs,
    _mxfp4_pages,
    _mxfp4_views,
    _reference,
    _round_query_e4m3,
)

torch.set_default_device("cuda")

HEAD_DIM = 128
PAGE = 128
TILE = 256
# One wave owns 4 chunks of a 256-token tile; each chunk is 8 K=16 PV MFMAs
# (head_dim / 16). ``pv_k16`` is that per-wave count for one sequence.
PV_K16_PER_TILE = 4 * (HEAD_DIM // 16)
SUPPORTED_GFX = ["gfx950"]
ATOL = 5e-3
RTOL = 5e-3


def run_torch(query, logical_key, logical_value, table, lengths):
    """fp32 attention on the dequantized pages. A bf16 query is rounded to e4m3,
    which is what the kernel does before QK."""
    seen = _round_query_e4m3(query) if query.dtype == torch.bfloat16 else query
    return _reference(seen, logical_key, logical_value, table, lengths, 1, 0, None)


@benchmark()
def test_pa_decode_mxfp4(batch, context, kv_heads, group, query_dtype):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    query, key, value, table, lengths = _make_inputs(
        1,
        kv_heads,
        group,
        HEAD_DIM,
        PAGE,
        torch.bfloat16,
        # Unequal 32-element blocks: a scale applied to the wrong block fails.
        "block_scales",
        device,
        (context,) * batch,
    )
    if query_dtype != torch.bfloat16:
        query = query.to(query_dtype)
    (key_pool, logical_key), (value_pool, logical_value) = (
        _mxfp4_pages(key),
        _mxfp4_pages(value),
    )
    del key, value
    (key_cache, key_scale), (value_cache, value_scale) = (
        _mxfp4_views(key_pool),
        _mxfp4_views(value_pool),
    )
    # NaN, not zeros: a row the kernel leaves unwritten fails the compare.
    output = torch.full(query.shape, float("nan"), dtype=torch.bfloat16)
    ref = run_torch(query, logical_key, logical_value, table, lengths)
    scale = HEAD_DIM**-0.5

    def launch():
        pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            lengths,
            table,
            scale,
            1,
            compute_type=torch.uint8,
            key_scale=key_scale,
            value_scale=value_scale,
            work_plan=None,
            max_context_length=context,
        )

    q_heads = kv_heads * group
    # QK and PV, each 2*tokens*heads*dim multiply-adds on the attended tokens.
    flops = 4 * batch * context * q_heads * HEAD_DIM
    # Pages the context touches, including the partial last page, plus Q and O.
    nbytes = (
        key_cache.nbytes
        + key_scale.nbytes
        + value_cache.nbytes
        + value_scale.nbytes
        + query.nbytes
        + output.nbytes
        + table.nbytes
        + lengths.nbytes
    )
    pv_k16 = ((context + TILE - 1) // TILE) * PV_K16_PER_TILE

    ret = {"gfx": get_gfx_runtime(), "pv_k16": pv_k16}
    for name, fn in {"flydsl": launch}.items():
        _, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            output.to(dtypes.fp32),
            rtol=RTOL,
            atol=ATOL,
            tol_err_ratio=0,
            msg=f"{name}: mxfp4 pa_decode",
        )
        assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "MXFP4 pa_decode unsupported on %s; skipping", get_gfx_runtime()
        )
        return 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MXFP4 paged-decode correctness and perf",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp8],
        help="""Query dtype. bf16 is quantized to e4m3 in the kernel; fp8 is e4m3 as stored.
    e.g.: -d bf16 fp8""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 32],
        help="""Sequences, one query token each.
    e.g.: -b 1 32""",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=int,
        nargs="*",
        default=[257, 4096, 32768],
        help="""Context tokens per sequence. 257 leaves a partial second tile.
    e.g.: -c 257 4096 32768""",
    )
    parser.add_argument(
        "-g",
        "--group",
        type=int,
        nargs="*",
        default=[16],
        help="""Query heads per KV head (1..16). 16 fills the 16-wide PV MFMA.
    e.g.: -g 1 16""",
    )
    parser.add_argument(
        "-k",
        "--kv-heads",
        type=int,
        nargs="*",
        default=[1],
        help="""KV heads.
    e.g.: -k 1""",
    )
    args = parser.parse_args()

    for query_dtype in args.dtype:
        rows = []
        for batch, context, kv_heads, group in itertools.product(
            args.batch, args.context, args.kv_heads, args.group
        ):
            rows.append(
                test_pa_decode_mxfp4(batch, context, kv_heads, group, query_dtype)
            )
        frame = pd.DataFrame(rows)
        aiter.logger.info(
            "mxfp4 pa_decode summary (markdown):\n%s",
            frame.to_markdown(index=False),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
