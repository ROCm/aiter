# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import os
import logging


logger = logging.getLogger("aiter")


def getLogger():
    global logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

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
        console_handler.setLevel(logging.INFO)

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


from .jit import core as core
from .utility import dtypes as dtypes
from .ops.enum import *
from .ops.norm import *
from .ops.quant import *
from .ops.gemm_op_a8w8 import *
from .ops.gemm_op_a16w16 import *
from .ops.gemm_op_a4w4 import *
from .ops.batched_gemm_op_a8w8 import *
from .ops.batched_gemm_op_bf16 import *
from .ops.deepgemm import *
from .ops.aiter_operator import *
from .ops.activation import *
from .ops.attention import *
from .ops.custom import *
from .ops.custom_all_reduce import *
from .ops.quick_all_reduce import *
from .ops.moe_op import *
from .ops.moe_sorting import *
from .ops.pos_encoding import *
from .ops.cache import *
from .ops.rmsnorm import *
from .ops.communication import *
from .ops.rope import *
from .ops.topk import *
from .ops.topk_plain import topk_plain
from .ops.mha import *
from .ops.gradlib import *
from .ops.trans_ragged_layout import *
from .ops.sample import *
from .ops.fused_mrope_rms import *
from . import mla

# Import Triton-based communication primitives from ops.triton.comms (optional, only if Iris is available)
try:
    from .ops.triton.comms import (
        IrisCommContext,
        calculate_heap_size,
        reduce_scatter,
        all_gather,
        reduce_scatter_rmsnorm_quant_all_gather,
        IRIS_COMM_AVAILABLE,
    )
except ImportError:
    # Iris not available, skip import
    IRIS_COMM_AVAILABLE = False
