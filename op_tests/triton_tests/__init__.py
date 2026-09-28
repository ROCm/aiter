# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys


def _pin_triton_configs() -> None:
    """Make the unit tests pick one config instead of benchmarking for one.

    Rebinds prune_configs on the Autotuner class this process imported. Nothing
    is changed in Triton and nothing outside pytest is affected.

    Returning a single config puts Triton on its own single-config path: it
    caches that config, runs no benchmark and prints nothing. Each kernel's own
    pruning still runs first, so a config that cannot fit in LDS or registers
    is still discarded.

    Every kernel already routes its config list through autotune_configs(), so
    this is a backstop rather than the mechanism: it holds the invariant even
    if a family's flag reaches the environment or a new kernel forgets.
    """
    try:
        from triton.runtime import Autotuner
    except ImportError:
        return

    prune = Autotuner.prune_configs

    def prune_configs(self, kwargs):
        return prune(self, kwargs)[:1]

    Autotuner.prune_configs = prune_configs


# Under pytest, silence aiter's INFO chatter (e.g. checkAllclose "passed~") unless the caller
# set AITER_LOG_LEVEL; aiter/__init__.py reads it once, on the first import triggered below.
if "pytest" in sys.modules:
    if "AITER_LOG_LEVEL" not in os.environ:
        os.environ["AITER_LOG_LEVEL"] = "WARNING"
    # Unit tests run the Dao-AI flash-attention port on its fixed default configs, not autotuning.
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "0")
    # Same for every other kernel: a config picked by timing makes a test's
    # numerics depend on runner load.
    if os.getenv("AITER_UT_TRITON_AUTOTUNE", "0").lower() not in ("1", "true", "on"):
        _pin_triton_configs()

from op_tests.triton_tests.utils import *
