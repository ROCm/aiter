# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import os
import logging

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

from .jit import core as core  # noqa: E402
from .utility import dtypes as dtypes  # noqa: E402
from .ops.enum import *  # noqa: F403,E402
from .ops.norm import *  # noqa: F403,E402
from .ops.quant import *  # noqa: F403,E402
from .ops.gemm_op_a8w8 import *  # noqa: F403,E402
from .ops.gemm_op_a16w16 import *  # noqa: F403,E402
from .ops.gemm_op_a4w4 import *  # noqa: F403,E402
from .ops.batched_gemm_op_a8w8 import *  # noqa: F403,E402
from .ops.batched_gemm_op_bf16 import *  # noqa: F403,E402
from .ops.deepgemm import *  # noqa: F403,E402
from .ops.aiter_operator import *  # noqa: F403,E402
from .ops.activation import *  # noqa: F403,E402
from .ops.attention import *  # noqa: F403,E402
from .ops.custom import *  # noqa: F403,E402
from .ops.custom_all_reduce import *  # noqa: F403,E402
from .ops.quick_all_reduce import *  # noqa: F403,E402
from .ops.moe_op import *  # noqa: F403,E402
from .ops.moe_sorting import *  # noqa: F403,E402
from .ops.moe_sorting_opus import *  # noqa: F403,E402
from .ops.pos_encoding import *  # noqa: F403,E402
from .ops.cache import *  # noqa: F403,E402
from .ops.rmsnorm import *  # noqa: F403,E402
from .ops.communication import *  # noqa: F403,E402
from .ops.rope import *  # noqa: F403,E402
from .ops.topk import *  # noqa: F403,E402
from .ops.topk_plain import topk_plain  # noqa: F403,F401,E402
from .ops.mha import *  # noqa: F403,E402
from .ops.gradlib import *  # noqa: F403,E402
from .ops.trans_ragged_layout import *  # noqa: F403,E402
from .ops.sample import *  # noqa: F403,E402
from .ops.fused_qk_norm_mrope_cache_quant import *  # noqa: F403,E402
from .ops.fused_qk_norm_rope_cache_quant import *  # noqa: F403,E402
from .ops.groupnorm import *  # noqa: F403,E402
from . import mla  # noqa: F403,F401,E402

# Import Triton-based communication primitives from ops.triton.comms (optional, only if Iris is available)
try:
    from .ops.triton.comms import (
        IrisCommContext,  # noqa: F401
        calculate_heap_size,  # noqa: F401
        reduce_scatter,  # noqa: F401
        all_gather,  # noqa: F401
        reduce_scatter_rmsnorm_quant_all_gather,  # noqa: F401
        IRIS_COMM_AVAILABLE,  # noqa: F401
    )
except ImportError:
    # Iris not available, skip import
    IRIS_COMM_AVAILABLE = False
