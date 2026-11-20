# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time of fused_qk_norm_mrope (us) for 4 Q + 1 K heads and 16 Q + 2 K heads."""

import pandas as pd

from aiter import logger
from aiter.ops.triton.fusions.fused_qk_norm_mrope import fused_qk_norm_mrope
from aiter.test_common import run_perftest
from op_tests.triton_tests.fusions.test_fused_qk_norm_mrope import make_inputs

TOKEN_COUNTS = [1, 16, 64, 256, 1024, 4096, 16384]
HEAD_COUNTS = [(4, 1), (16, 2)]


def main():
    results = []
    for q_heads, k_heads in HEAD_COUNTS:
        for num_tokens in TOKEN_COUNTS:
            inputs = make_inputs(num_tokens, q_heads, k_heads)
            # The first timing of new inputs is too high. Run once and drop it.
            run_perftest(fused_qk_norm_mrope, **inputs, num_rotate_args=4)
            _, us = run_perftest(fused_qk_norm_mrope, **inputs, num_rotate_args=4)
            results.append(
                {"Hq": q_heads, "Hk": k_heads, "T": num_tokens, "us": round(us, 2)}
            )
    logger.info(
        "fused_qk_norm_mrope (us):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
