# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compact or ragged-``next_n`` helpers. Implementation lives on the op test."""

from op_tests.test_flydsl_fp8_paged_mqa_logits import (  # noqa: F401
    MAX_NN,
    live_row_mask,
    padded_logits,
    ref_padded_ragged,
    sample_next_n_lens,
    scatter_seq_logits,
)
