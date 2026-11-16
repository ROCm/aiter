# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the bucketed config lookup and the Gluon quant config tables."""

import glob


from aiter.ops.triton.utils.config_utils import AITER_TRITON_CONFIGS_PATH


_QUANT_JSONS = sorted(
    glob.glob(f"{AITER_TRITON_CONFIGS_PATH}/*/gluon/quant/*/DEFAULT.json")
)


def test_quant_config_files_exist():
    assert _QUANT_JSONS, "no Gluon quant config files found"
