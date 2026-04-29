# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 MoE HSACO: prebuilt ``.co`` via ``hsaco_tools.get_kernel`` (no pyhip package).

``fused_moe()`` calls ``run_moe_small_batch_hsaco`` only when ``AITER_MOE_SMALL_BATCH=1`` and
``1 <= B <= MOE_HSACO_SMALL_BATCH_MAX_B``. No HSACO hooks in ``get_2stage_cfgs``.

``fused_moe_`` may call ``fused_moe_hsaco_batch1_fwd`` for a second-chance B=1 path (no
``moe_sorting``) if the top-level fast path returned ``None``.

Requires matching artifacts under ``hsa/<gfx>/fmoe_pyhip/`` (README there).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from csrc.cpp_itfs.hsaco_tools import get_kernel
from csrc.cpp_itfs.utils import AITER_CORE_DIR

# Gated in ``fused_moe()``; must match the ``1 <= B <= …`` check there.
MOE_HSACO_SMALL_BATCH_MAX_B = 32


def _fmoe_pyhip_co_dir() -> str:
    return os.path.join(AITER_CORE_DIR, "hsa", get_gfx(), "fmoe_pyhip")


def _hsaco_prefix(co_basename: str, symbol_substring: str) -> str:
    """Path without ``.co`` plus ``:symbol`` — pyhip symbols are ``moe_*``, not full cache key names."""
    return os.path.join(_fmoe_pyhip_co_dir(), co_basename) + f":{symbol_substring}"


def _launchable_or_none(full_prefix_without_co_colon_symbol: str):
    try:
        return get_kernel(full_prefix_without_co_colon_symbol, ())
    except Exception:
        return None


# B=1 ``moe_gemm_batch1`` pair (see ``MOE_PYHIP_KERNELS.md`` / ``target_co``).
PYHIP_HSACO_CO_NAMES = {
    "moe_gemm_batch1_gate": "moe_gemm_batch1-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=True",
    "moe_gemm_batch1_down": "moe_gemm_batch1-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=False",
}


def fused_moe_hsaco_batch1_fwd(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """B=1 HSACO path: ``moe_gemm_batch1`` ×2 without ``moe_sorting``."""
    if get_gfx() != "gfx942":
        return None
    if hidden_states.shape[0] != 1:
        return None
    _, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    topk = topk_ids.shape[1]
    if N1 != 2 * K2 or K1 != 4096 or N2 != 4096 or K2 != 128 or topk != 10:
        return None
    if w1.dtype != dtypes.fp8:
        return None
    wd = str(w1.dtype)
    if wd != "torch.float8_e4m3fnuz":
        return None
    w1u = w1 if getattr(w1, "is_shuffled", False) else shuffle_weight(w1, layout=(16, 16))
    w2u = w2 if getattr(w2, "is_shuffled", False) else shuffle_weight(w2, layout=(16, 16))
    topk_w_f32 = (
        topk_weight
        if topk_weight.dtype == torch.float32
        else topk_weight.float()
    )
    k_gate = _launchable_or_none(
        _hsaco_prefix(PYHIP_HSACO_CO_NAMES["moe_gemm_batch1_gate"], "moe_gemm_batch1")
    )
    k_down = _launchable_or_none(
        _hsaco_prefix(PYHIP_HSACO_CO_NAMES["moe_gemm_batch1_down"], "moe_gemm_batch1")
    )
    if k_gate is None or k_down is None:
        return None
    device = hidden_states.device
    dtype_h = hidden_states.dtype
    gemm1_out = torch.empty((1, topk, N1 // 2), dtype=dtype_h, device=device)
    gemm2_out = torch.zeros((1, N2), dtype=dtype_h, device=device)
    k_gate(
        [N1 // 32, topk],
        [256],
        hidden_states,
        w1u,
        gemm1_out,
        topk_ids,
        topk_w_f32,
        w1_scale,
        1,
        N1,
        K1,
    )
    k_down(
        [N2 // 32, topk],
        [64],
        gemm1_out,
        w2u,
        gemm2_out,
        topk_ids,
        topk_w_f32,
        w2_scale,
        1,
        N2,
        K2,
    )
    logger.info(
        "[fused_moe] AITER_MOE_SMALL_BATCH=1: B=1 gfx942 HSACO (moe_gemm_batch1 gate + down)"
    )
    return gemm2_out


def _weight_dtype_tag(w: torch.Tensor) -> str:
    """Cache-key fragment as in pyhip ``.co`` basenames (e.g. ``torch.float8_e4m3fnuz``)."""
    return str(w.dtype)


def _maybe_shuffle_moe_w(w: torch.Tensor) -> torch.Tensor:
    if getattr(w, "is_shuffled", False):
        return w
    if w.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
        return shuffle_weight(w, layout=(16, 16))
    return w


def _co_basename_moe_gemm_batch1(weight_dtype_tag: str, with_silu: bool) -> str:
    silu = "True" if with_silu else "False"
    return f"moe_gemm_batch1-1-weight_dtype={weight_dtype_tag}-with_silu={silu}"


def _co_basename_moe_gemm_batch(weight_dtype_tag: str, with_silu: bool) -> str:
    silu = "True" if with_silu else "False"
    return f"moe_gemm_batch-1-weight_dtype={weight_dtype_tag}-with_silu={silu}"


def _co_basename_moe_2stage_splitk_down(
    weight_dtype_tag: str, topk: int, k: int, n: int
) -> str:
    return (
        f"moe_2stage_splitk-1-weight_dtype={weight_dtype_tag}-"
        f"TOPK={topk}-K={k}-N={n}-with_silu=False-"
        f"BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=64-fp8_ptpc=True"
    )


def run_moe_small_batch_hsaco(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
) -> Optional[torch.Tensor]:
    """Prebuilt ``.co`` MoE for ``1 <= B <= MOE_HSACO_SMALL_BATCH_MAX_B``; else ``None``."""
    b = int(hidden_states.shape[0])
    if not (1 <= b <= MOE_HSACO_SMALL_BATCH_MAX_B):
        return None
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation != ActivationType.Silu
    ):
        return None
    if not (
        (quant_type == QuantType.No and w1.dtype == torch.bfloat16)
        or (quant_type == QuantType.per_Token and w1.dtype == torch.float8_e4m3fnuz)
    ):
        return None

    from aiter.fused_moe import moe_sorting

    B = b
    E, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    TOPK = topk_ids.shape[1]
    if N1 != 2 * K2:
        return None

    wd = _weight_dtype_tag(w1)
    w1u = _maybe_shuffle_moe_w(w1)
    w2u = _maybe_shuffle_moe_w(w2)

    topk_w_f32 = (
        topk_weight
        if topk_weight.dtype == torch.float32
        else topk_weight.float()
    )

    gemm1_out = torch.empty(
        [B, TOPK, N1 // 2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if B == 1:
        k1 = _launchable_or_none(
            _hsaco_prefix(_co_basename_moe_gemm_batch1(wd, True), "moe_gemm_batch1")
        )
        k2 = _launchable_or_none(
            _hsaco_prefix(_co_basename_moe_gemm_batch1(wd, False), "moe_gemm_batch1")
        )
        if k1 is None or k2 is None:
            return None
        gemm2_out = torch.zeros(
            [1, N2], dtype=hidden_states.dtype, device=hidden_states.device
        )
        k1(
            [N1 // 32, TOPK],
            [256],
            hidden_states,
            w1u,
            gemm1_out,
            topk_ids,
            topk_w_f32,
            w1_scale,
            1,
            N1,
            K1,
        )
        k2(
            [N2 // 32, TOPK],
            [64],
            gemm1_out,
            w2u,
            gemm2_out,
            topk_ids,
            topk_w_f32,
            w2_scale,
            1,
            N2,
            K2,
        )
        return gemm2_out

    BLOCK_M = 16
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weight,
        E,
        K1,
        hidden_states.dtype,
        BLOCK_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    grid = int(sorted_expert_ids.shape[0])
    if B * TOPK <= E:
        grid = min(grid, B * TOPK)

    k_s1 = _launchable_or_none(
        _hsaco_prefix(_co_basename_moe_gemm_batch(wd, True), "moe_gemm_batch")
    )
    if k_s1 is None:
        return None
    k_s1(
        [N1 // 32, grid],
        [256],
        hidden_states,
        w1u,
        gemm1_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w1_scale,
        B,
        N1,
        K1,
        TOPK,
    )

    moe_buf.zero_()

    if quant_type == QuantType.per_Token and w1.dtype == torch.float8_e4m3fnuz:
        splitk_base = _co_basename_moe_2stage_splitk_down(wd, TOPK, K2, N2)
        k_down = _launchable_or_none(
            _hsaco_prefix(splitk_base, "moe_2stage_splitk")
        )
        if k_down is None:
            return None
        k_down(
            [N2 // 64, grid],
            [64],
            gemm1_out,
            w2u,
            moe_buf,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w2_scale,
            B,
        )
    else:
        k_s2 = _launchable_or_none(
            _hsaco_prefix(_co_basename_moe_gemm_batch(wd, False), "moe_gemm_batch")
        )
        if k_s2 is None:
            return None
        k_s2(
            [N2 // 32, grid],
            [64],
            gemm1_out,
            w2u,
            moe_buf,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w2_scale,
            B,
            N2,
            K2,
            TOPK,
        )
    return moe_buf
