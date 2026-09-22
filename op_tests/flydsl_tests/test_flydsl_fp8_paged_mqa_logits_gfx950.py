# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""WaveScope ATT driver. Canonical tables: op_tests/test_flydsl_fp8_paged_mqa_logits.py."""

from op_tests import test_flydsl_fp8_paged_mqa_logits as _op

if __name__ == "__main__":
    _op.test_fp8_paged_mqa_logits_ragged(16, _op.MAX_NN, 32768, 0)
