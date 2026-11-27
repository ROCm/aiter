# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rotating-buffer performance of QSA prep, excluding logits/top-k and projection."""

import functools

import pandas as pd
import torch

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.attention.qsa_indexer import (
    qsa_indexer_decode_prep,
    qsa_indexer_prefill_prep,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_qsa_indexer import make_prep_inputs


def make_pool_input(mode, rows, context):
    _, inputs = make_prep_inputs(mode, rows, context)
    inputs["q_out"] = torch.empty(
        (inputs["qk"].shape[0], 4, 128), dtype=torch.bfloat16, device="cuda"
    )
    return (inputs,)


def main():
    if get_gfx_runtime() != "gfx942":
        logger.warning("QSA indexer prep requires gfx942; skipping")
        return
    parser = get_parser(kernel_name="QSA indexer prep")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("prefill", "decode", "verify"),
        default=["prefill", "decode", "verify"],
    )
    args = parser.parse_args()
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    data = []
    for mode in args.modes:
        shapes = (
            ((12000, 12000), (16384, 262144))
            if mode == "prefill"
            else ((1, 12000), (32, 12000))
        )
        for rows, context in shapes:
            bencher.build_pool(
                [functools.partial(make_pool_input, mode, rows, context)], [["inputs"]]
            )
            op = (
                qsa_indexer_prefill_prep
                if mode == "prefill"
                else qsa_indexer_decode_prep
            )

            def call(inputs, op=op):
                return op(**inputs)

            ms = bencher.bench(call)
            data.append(
                {
                    "mode": mode,
                    "rows_or_requests": rows,
                    "context": context,
                    "us": ms * 1000,
                }
            )
    logger.info("QSA prep summary:\n%s", pd.DataFrame(data).to_markdown(index=False))


if __name__ == "__main__":
    main()
