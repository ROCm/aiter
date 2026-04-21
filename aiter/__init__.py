# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import os
import sys
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

# Use bundled pre-compiled FlyDSL cache unless the user overrides via env var.
_flydsl_cache = os.path.join(os.path.dirname(__file__), "jit", "flydsl_cache")
if os.path.isdir(_flydsl_cache) and "FLYDSL_RUNTIME_CACHE_DIR" not in os.environ:
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _flydsl_cache

if sys.platform == "win32":
    logger.info("Windows: CK and HIP ops are not available. Triton ops only.")
else:
    _HIP_INIT_EXCEPTIONS = (ImportError, RuntimeError, OSError, KeyError)

    _hip_runtime_ok = True
    try:
        from .jit import core as core  # noqa: E402
        from .utility import dtypes as dtypes  # noqa: E402
    except _HIP_INIT_EXCEPTIONS as _e:
        _hip_runtime_ok = False
        logger.warning(
            "ROCm/HIP JIT core unavailable: %s. "
            "CK and HIP ops are disabled. Triton ops remain available.",
            _e,
            exc_info=True,
        )

    if _hip_runtime_ok:
        # Ops loaded with `from .ops.X import *` semantics. Each is wrapped in its
        # own try/except so a single broken op (e.g. a dlopen failure inside one
        # .so) surfaces an actionable per-module traceback instead of silently
        # stripping every HIP/CK symbol from the `aiter` namespace.
        _OPS_IMPORT_STAR = (
            "enum",
            "norm",
            "quant",
            "gemm_op_a8w8",
            "gemm_op_a16w16",
            "gemm_op_a4w4",
            "batched_gemm_op_a8w8",
            "batched_gemm_op_bf16",
            "deepgemm",
            "aiter_operator",
            "activation",
            "attention",
            "custom",
            "custom_all_reduce",
            "quick_all_reduce",
            "moe_op",
            "moe_sorting",
            "moe_sorting_opus",
            "pos_encoding",
            "cache",
            "rmsnorm",
            "communication",
            "rope",
            "topk",
            "mha",
            "gradlib",
            "trans_ragged_layout",
            "sample",
            "fused_qk_norm_mrope_cache_quant",
            "fused_qk_norm_rope_cache_quant",
            "fused_qk_rmsnorm_group_quant",
            "groupnorm",
            "mhc",
            "causal_conv1d",
            "fused_split_gdr_update",
        )

        import importlib as _importlib  # noqa: E402

        for _ops_mod in _OPS_IMPORT_STAR:
            try:
                _m = _importlib.import_module(f".ops.{_ops_mod}", __name__)
            except _HIP_INIT_EXCEPTIONS as _e:
                logger.warning(
                    "Failed to import aiter.ops.%s: %s. "
                    "Symbols from this op will be unavailable.",
                    _ops_mod,
                    _e,
                    exc_info=True,
                )
                continue
            _all = getattr(_m, "__all__", None)
            if _all is None:
                _all = [n for n in dir(_m) if not n.startswith("_")]
            for _name in _all:
                globals()[_name] = getattr(_m, _name)

        try:
            from .ops.topk_plain import topk_plain  # noqa: F401,E402
        except _HIP_INIT_EXCEPTIONS as _e:
            logger.warning(
                "Failed to import aiter.ops.topk_plain.topk_plain: %s",
                _e,
                exc_info=True,
            )

        try:
            from . import mla  # noqa: F401,E402
        except _HIP_INIT_EXCEPTIONS as _e:
            logger.warning(
                "Failed to import aiter.mla: %s",
                _e,
                exc_info=True,
            )

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
except (ImportError, AttributeError):
    # Iris or triton not available, skip import
    IRIS_COMM_AVAILABLE = False
