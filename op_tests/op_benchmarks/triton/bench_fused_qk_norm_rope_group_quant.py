# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode time of fused_qk_norm_rope_group_quant with the SWA write (us, H = 128)."""

import pandas as pd

from aiter import logger
from aiter.ops.triton.fusions.fused_qk_norm_rope_group_quant import (
    fused_qk_norm_rope_group_quant,
)
from aiter.test_common import run_perftest
from op_tests.triton_tests.fusions.test_fused_qk_norm_rope_group_quant import (
    make_inputs,
)

TOKEN_COUNTS = [128, 256, 512, 1024, 2048]


def main():
    results = []
    for num_tokens in TOKEN_COUNTS:
        inputs = make_inputs(num_tokens)
        # The first timing of new inputs is too high. Run once and drop it.
        run_perftest(fused_qk_norm_rope_group_quant, **inputs, num_rotate_args=4)
        _, us = run_perftest(
            fused_qk_norm_rope_group_quant, **inputs, num_rotate_args=4
        )
        results.append({"T": num_tokens, "us": round(us, 2)})
    logger.info(
        "fused_qk_norm_rope_group_quant decode (us):\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
