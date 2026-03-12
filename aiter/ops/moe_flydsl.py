# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import torch
from aiter import logger

# Dynamic loading of FLIR based on FLIR_PATH
FLIR_PATH = os.getenv("FLIR_PATH")

# a8w4smooth kernel requires these env vars for packed4 qparams + K64 interleave.
os.environ.setdefault("FLIR_A8W4SMOOTH_QPARAM_FORMAT", "packed4")
os.environ.setdefault("FLIR_A8W4SMOOTH_INTERLEAVE_K64", "1")
os.environ.setdefault("FLIR_A8W4SMOOTH_OVERFLOW_GUARD", "1")


def _ensure_flir():
    if not FLIR_PATH:
        raise ImportError(
            "Please set FLIR_PATH environment variable to the root of the FLIR project."
        )
    if FLIR_PATH not in sys.path:
        sys.path.insert(0, FLIR_PATH)
    try:
        from kernels.moe_gemm_2stage import (
            compile_moe_gemm1,
            compile_moe_gemm2,
            compile_moe_gemm2_ex,
        )
        from tests.utils import shuffle_weight as flir_shuffle_weight

        return (
            compile_moe_gemm1,
            compile_moe_gemm2,
            compile_moe_gemm2_ex,
            flir_shuffle_weight,
        )
    except ImportError as e:
        logger.error(f"Failed to import FLIR kernels from FLIR_PATH={FLIR_PATH}: {e}")
        raise e


class FlyDSLManager:
    """Manages FlyDSL kernel compilation and caching."""

    def __init__(self):
        self.cache = {}

    def get_exe(self, compile_fn, **kwargs):
        key = (compile_fn.__name__, tuple(sorted(kwargs.items())))
        if key not in self.cache:
            logger.info(
                f"[FlyDSL] Compiling kernel {compile_fn.__name__} with config: {kwargs}"
            )
            self.cache[key] = compile_fn(**kwargs)
        return self.cache[key]


_manager = FlyDSLManager()

_ENV_TILE_N1 = "AITER_FLYDSL_MOE_TILE_N1"
_ENV_TILE_K1 = "AITER_FLYDSL_MOE_TILE_K1"
_ENV_TILE_N2 = "AITER_FLYDSL_MOE_TILE_N2"
_ENV_TILE_K2 = "AITER_FLYDSL_MOE_TILE_K2"


def flydsl_moe_stage1(
    input,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    *,
    block_m=16,
    a1_scale=None,
    w1_scale=None,
    w1_lqq_scale=None,
    w1_lqq_zero=None,
    sorted_weights=None,
    experts=0,
    model_dim=0,
    inter_dim=0,
):
    compile_moe_gemm1, _, _, _ = _ensure_flir()

    token_num = out.shape[0]
    tile_m = block_m if block_m else 16
    tile_n = int(os.environ.get(_ENV_TILE_N1, "64"))
    tile_k = int(os.environ.get(_ENV_TILE_K1, "256"))

    exe = _manager.get_exe(
        compile_moe_gemm1,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        doweight_stage1=sorted_weights is not None,
        in_dtype="a8w4smooth",
        out_dtype="bf16",
        use_cshuffle_epilog=False,
    )

    qs = w1_lqq_scale.view(torch.int32)
    qz = w1_lqq_zero.view(torch.int32)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    num_expert_blocks = sorted_expert_ids.shape[0]

    if sorted_weights is None:
        sorted_weights = torch.empty(0, device=out.device, dtype=torch.float32)

    exe(
        out,
        input,
        w1,
        a1_scale,
        w1_scale,
        qs,
        qz,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        inter_dim,
        model_dim,
        int(num_expert_blocks),
        stream_ptr,
    )

    return out


def flydsl_moe_stage2(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    *,
    block_m=16,
    a2_scale=None,
    w2_scale=None,
    w2_lqq_scale=None,
    w2_lqq_zero=None,
    sorted_weights=None,
    experts=0,
    model_dim=0,
    inter_dim=0,
    mode=None,
    valid_mask=None,
    intermediate=None,
):
    _, _, compile_moe_gemm2_ex, _ = _ensure_flir()

    token_num = out.shape[0]
    tile_m = block_m if block_m else 16
    tile_n = int(os.environ.get(_ENV_TILE_N2, "128"))
    tile_k = int(os.environ.get(_ENV_TILE_K2, "256"))
    if mode is None:
        mode = os.environ.get("AITER_FLYDSL_GEMM2_MODE", "ATOMIC")
    mode = str(mode).strip().lower()
    if mode not in ("atomic", "reduce"):
        raise ValueError(f"Unsupported moe gemm2 mode: {mode}")
    use_valid_mask = valid_mask is not None

    exe = _manager.get_exe(
        compile_moe_gemm2_ex,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        doweight_stage2=sorted_weights is not None,
        in_dtype="a8w4smooth",
        out_dtype="bf16",
        mode=mode,
        valid_mask=True if use_valid_mask else None,
        zero_intermediate=False,
    )

    qs = w2_lqq_scale.view(torch.int32)
    qz = w2_lqq_zero.view(torch.int32)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    num_expert_blocks = sorted_expert_ids.shape[0]

    if sorted_weights is None:
        sorted_weights = torch.empty(0, device=out.device, dtype=torch.float32)
    if valid_mask is not None:
        valid_mask = valid_mask.contiguous()

    if mode == "reduce":
        expected_intermediate_shape = (token_num * topk, model_dim)
        if intermediate is not None:
            if (
                intermediate.device != out.device
                or intermediate.dtype != out.dtype
                or intermediate.numel()
                != expected_intermediate_shape[0] * expected_intermediate_shape[1]
            ):
                intermediate = None
            else:
                intermediate = intermediate.view(expected_intermediate_shape)
        exe(
            out,
            inter_states,
            w2,
            a2_scale,
            w2_scale,
            qs,
            qz,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            token_num,
            model_dim,
            inter_dim,
            int(num_expert_blocks),
            intermediate=intermediate,
            valid_mask=valid_mask,
            stream_ptr=stream_ptr,
        )
    else:
        exe(
            out,
            inter_states,
            w2,
            a2_scale,
            w2_scale,
            qs,
            qz,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            token_num,
            model_dim,
            inter_dim,
            int(num_expert_blocks),
            stream_ptr,
        )

    return out
