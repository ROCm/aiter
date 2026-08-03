# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
import sys

import torch

logger = logging.getLogger("aiter")


def getLogger():
    global logger
    if not logger.handlers:
        # Configure log level from environment variable
        # Valid values: DEBUG, INFO (default), WARNING, ERROR
        log_level_str = os.getenv("AITER_LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

        if log_level_str not in valid_levels:
            print(
                f"\033[93m[aiter] Warning: Invalid AITER_LOG_LEVEL '{log_level_str}', "
                f"using 'INFO'. Valid values: {', '.join(valid_levels)}\033[0m"
            )
            log_level_str = "INFO"

        log_level = getattr(logging, log_level_str)
        logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        if int(os.environ.get("AITER_LOG_MORE", 0)):
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)03d - %(processName)s:%(process)d - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = logging.Formatter(
                fmt="[%(name)s] %(message)s",
            )
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)

        logger.addHandler(console_handler)
        logger.propagate = False

        if hasattr(torch._dynamo.config, "ignore_logger_methods"):
            torch._dynamo.config.ignore_logger_methods = (
                logging.Logger.info,
                logging.Logger.warning,
                logging.Logger.debug,
                logger.warning,
                logger.info,
                logger.debug,
            )

    return logger


logger = getLogger()
AITER_AOT_IMPORT = os.getenv("AITER_AOT_IMPORT", "0") == "1"
# Triton-only: expose only the Triton ops, skipping the C++/CK/HIP ops and their
# JIT build. Always on for Windows (no CK/HIP there); elsewhere opt in via the
# env var, e.g. Triton-backend users with no C++ toolchain or CK.
AITER_TRITON_ONLY = (
    os.getenv("AITER_TRITON_ONLY", "0") == "1" or sys.platform == "win32"
)

# Use bundled pre-compiled FlyDSL cache unless the user overrides via env var.
_flydsl_cache = os.path.join(os.path.dirname(__file__), "jit", "flydsl_cache")
if os.path.isdir(_flydsl_cache) and "FLYDSL_RUNTIME_CACHE_DIR" not in os.environ:
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _flydsl_cache

if AITER_TRITON_ONLY:
    logger.info("Triton ops only: CK and HIP ops (and their JIT build) are skipped.")
elif AITER_AOT_IMPORT:
    from .jit import core as core
else:
    # NOTE: do NOT wrap this block in try/except.
    # Catching ImportError here silently truncates the top-level aiter
    # namespace whenever any single import fails, which has caused
    # downstream regressions (e.g. vLLM-ROCm losing rmsnorm2d_fwd_with_add).
    # Any real import failure on Linux must surface as a loud ImportError
    # on `import aiter` -- that is what 0.1.10.post3 and earlier did.
    # opus is gfx950-only but the package self-guards (warn + stubs on
    # non-gfx950) inside aiter/ops/opus/__init__.py, so its import line
    # is safe to put at top-level without try/except.
    from . import mla  # noqa: F401
    from .jit import core as core
    from .ops.activation import *
    from .ops.aiter_operator import *
    from .ops.attention import *
    from .ops.batched_gemm_op_a8w8 import *
    from .ops.batched_gemm_op_bf16 import *
    from .ops.cache import *
    from .ops.causal_conv1d_update import *
    from .ops.communication import *
    from .ops.custom import *
    from .ops.custom_all_reduce import *
    from .ops.deepgemm import *
    from .ops.enum import *
    from .ops.fused_qk_norm_mrope_cache_quant import *
    from .ops.fused_qk_norm_rope_cache_quant import *
    from .ops.fused_qk_rmsnorm_group_quant import *
    from .ops.fused_qknorm_idxrqknorm import (  # noqa: F401
        fused_qknorm_idxrqknorm,
    )
    from .ops.fused_split_gdr_update import *
    from .ops.gemm_op_a4w4 import *
    from .ops.gemm_op_a8w8 import *
    from .ops.gemm_op_a16w16 import *
    from .ops.gradlib import *
    from .ops.groupnorm import *
    from .ops.hisparse_swap import *
    from .ops.mha import *
    from .ops.mhc import *
    from .ops.moe_op import *
    from .ops.moe_sorting import *
    from .ops.moe_sorting_opus import *
    from .ops.norm import *
    from .ops.opus import *
    from .ops.pa_sparse_prefill_opus import *
    from .ops.pos_encoding import *
    from .ops.quant import *
    from .ops.quick_all_reduce import *
    from .ops.rmsnorm import *
    from .ops.rope import *
    from .ops.sample import *
    from .ops.topk import *
    from .ops.topk_plain import topk_plain  # noqa: F401
    from .ops.trans_ragged_layout import *
    from .utility import dtypes as dtypes

# Import Triton-based communication primitives from ops.triton.comms (optional, only if Iris is available)
try:
    from .ops.triton.comms import (
        IRIS_COMM_AVAILABLE,
        IrisCommContext,  # noqa: F401
        all_gather,  # noqa: F401
        calculate_heap_size,  # noqa: F401
        reduce_scatter_rmsnorm_quant_all_gather,  # noqa: F401
    )
    from .ops.triton.comms import (
        reduce_scatter as iris_reduce_scatter,  # noqa: F401  # avoid shadowing C++ reduce_scatter exported by custom_all_reduce.py
    )
except (ImportError, AttributeError):
    # Iris or triton not available, skip import
    IRIS_COMM_AVAILABLE = False
