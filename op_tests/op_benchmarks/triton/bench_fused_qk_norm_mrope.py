# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time of fused_qk_norm_mrope (us) for Qwen3.5 TP4: 4 Q heads and 1 K head."""

import pandas as pd

from aiter import logger
from aiter.ops.triton.fusions.fused_qk_norm_mrope import fused_qk_norm_mrope
from aiter.test_common import run_perftest
from op_tests.triton_tests.fusions.test_fused_qk_norm_mrope import make_inputs

TOKEN_COUNTS = [1, 16, 64, 256, 1024, 4096, 16384]


def main():
    results = []
    for num_tokens in TOKEN_COUNTS:
        inputs = make_inputs(num_tokens, 4, 1)
        # The first timing of new inputs is too high. Run once and drop it.
        run_perftest(fused_qk_norm_mrope, **inputs, num_rotate_args=4)
        _, us = run_perftest(fused_qk_norm_mrope, **inputs, num_rotate_args=4)
        results.append({"T": num_tokens, "us": round(us, 2)})
    logger.info(
        "fused_qk_norm_mrope (us):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
