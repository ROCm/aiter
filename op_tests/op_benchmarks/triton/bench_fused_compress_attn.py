# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode time of fused_compress_attn_fp4 (us). "empty" means all plan rows are empty."""

import pandas as pd

from aiter import logger
from aiter.ops.triton.fusions.fused_compress_attn import fused_compress_attn_fp4
from aiter.test_common import run_perftest
from op_tests.triton_tests.fusions.test_fused_compress_attn import make_inputs

PLAN_ROW_COUNTS = [128, 256, 512, 1024, 2048]


def main():
    results = []
    for num_rows in PLAN_ROW_COUNTS:
        inputs = make_inputs(num_rows)
        # The first timing of new inputs is too high. Run once and drop it.
        run_perftest(fused_compress_attn_fp4, **inputs)
        _, compress_us = run_perftest(fused_compress_attn_fp4, **inputs)
        # Column 2 is last_pos. A negative last_pos makes every plan row empty.
        inputs["plan"][:, 2] = -1
        _, empty_us = run_perftest(fused_compress_attn_fp4, **inputs)
        results.append(
            {"M": num_rows, "compress_us": compress_us, "empty_us": empty_us}
        )
    logger.info(
        "fused_compress_attn_fp4 decode (us):\n%s",
        pd.DataFrame(results).round(2).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
