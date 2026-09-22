# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shim. Canonical test: op_tests/test_flydsl_fp8_paged_mqa_logits.py."""

from op_tests.test_flydsl_fp8_paged_mqa_logits import (  # noqa: F401
    live_row_mask,
    main,
    padded_logits,
    ref_padded_ragged,
    sample_next_n_lens,
)

if __name__ == "__main__":
    main()
