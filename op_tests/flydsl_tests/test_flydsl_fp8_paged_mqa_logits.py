# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shim for WaveScope drivers. Canonical test: op_tests/test_flydsl_fp8_paged_mqa_logits.py."""

from op_tests.test_flydsl_fp8_paged_mqa_logits import (  # noqa: F401
    HEAD_DIM,
    HEADS,
    KV_BLOCK_SIZE,
    MAX_NN,
    Inputs,
    _build_inputs,
    _kernel_inputs,
    calc_diff,
    kv_cache_cast_to_fp8,
    live_row_mask,
    main,
    padded_logits,
    preshuffle_kv_data,
    ref_fp8_paged_mqa_logits,
    ref_padded_ragged,
    run_torch,
    sample_next_n_lens,
    scatter_seq_logits,
)

if __name__ == "__main__":
    main()
