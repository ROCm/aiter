# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL K5 mfma16_hip BV tuner entry point (``K5BvTuner`` / ``TunerCommon``).

Usage
-----
Tune shapes declared in the untuned table (matched to K5 prefill test cases):

    python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \\
      -i aiter/configs/model_configs/qwen3_5_35b_chunk_gdn_h_mfma16_hip_untuned.csv \\
      -o /tmp/chunk_gdn_h_mfma16_hip_tuned.candidate.csv

Replay the checked-in tuned table and compare measured ``us``:

    python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \\
      --run_config aiter/configs/model_configs/qwen3_5_397b_chunk_gdn_h_mfma16_hip_tuned.csv
"""

from __future__ import annotations

from k5_bv_tuner import main

if __name__ == "__main__":
    main()
