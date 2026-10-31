# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode time of fused_qk_norm_rope_group_quant with the SWA write (us, H = 128)."""

import functools

import pandas as pd
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

from aiter import logger
from aiter.ops.triton.fusions.fused_qk_norm_rope_group_quant import (
    fused_qk_norm_rope_group_quant,
)
from op_tests.triton_tests.fusions.test_fused_qk_norm_rope_group_quant import (
    make_inputs,
)

TOKEN_COUNTS = [128, 256, 512, 1024, 2048]

_OUT_VAL_KEYS = [
    "q",
    "kv",
    "k_weight",
    "positions",
    "cos",
    "sin",
    "eps",
    "swa_nope",
    "swa_rope",
    "swa_dest_rows",
    "batch_ids",
]


def _generate_pool_inputs(num_tokens):
    inputs = make_inputs(num_tokens)
    return tuple(inputs[k] for k in _OUT_VAL_KEYS)


def main():
    results = []
    bencher = RotatingBufferBench()
    for num_tokens in TOKEN_COUNTS:
        _p_gen = functools.partial(_generate_pool_inputs, num_tokens)
        bencher.build_pool([_p_gen], [_OUT_VAL_KEYS])
        us = bencher.bench(fused_qk_norm_rope_group_quant) * 1000
        results.append({"T": num_tokens, "us": round(us, 2)})
    logger.info(
        "fused_qk_norm_rope_group_quant decode (us):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
