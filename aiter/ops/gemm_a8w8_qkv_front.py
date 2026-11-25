# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MXFP8 QKV projection of a sparse-attention decode layer, as a tuned GEMM with a deferred fused epilogue.

A decode layer runs ``QKV = A @ W^T`` and then q / k / index RMSNorm + partial RoPE + NVFP4 K / V / index-K insert
(``aiter.fused_qknorm_idxrqknorm``). Per row count there are three ways to do that front, and only a measurement says
which is fastest, so the choice is a tuned-table row like every other GEMM's (``a8w8_mxfp8_qkv_front_tuned_gemm.csv``,
keyed ``gfx, cu_num, M, N, K, w_scale_block``, ``kernelId`` = ``qkvfront``):

* ``plain``  - activation quantize + ``gemm_a8w8_blockscale`` (its own tuned row) now, the insert later;
* ``coop``   - 17-64 rows: the activation is only quantized now; the QKV GEMM and the insert are ONE launch
  (``qkv_chain_cooperative_insert``) run by the insert call;
* ``split3`` - 65-128 rows: quantize now; split-3 GEMM and reduce + insert (two launches) run by the insert call.

``gemm_a8w8_qkv_front`` is the entry of the projection. It returns the ``[M, N]`` bf16 QKV; for a fused tier it returns
an UNCOMPUTED tensor and parks the quantized activation, and ``fused_qknorm_idxrqknorm`` -- called next on that same tensor,
on the same thread -- finds it (``take_deferred``) and runs the fused kernel instead of the insert alone. The caller says
it may accept a deferred result (``defer_ok``): only a caller whose consumer of the QKV is that insert can (a graph
that replays the quantizer without Python, or any other reader of the tensor, must pass ``defer_ok=False``).

The tier kernels and their numerics are in ``aiter.ops.flydsl.kernels.qkv_chain`` (technique source: amd-kernels
``mxfp8_qkv_norm_rope_nvfp4_insert``).
"""

import functools
import threading
from typing import NamedTuple

import pandas as pd
import torch
from torch import Tensor

from ..jit.core import AITER_CONFIGS
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..utility import dtypes
from .gemm_op_common import find_padded_m_row

N, K = (
    2560,
    6144,
)  # the MiniMax-M3.1 TP4 QKV projection (16 q heads + K + V + index q / k, 128 wide)
MIN_ROWS, COOP_MAX_ROWS, MAX_ROWS = 17, 64, 128
KERNEL_ID = "qkvfront"
TIER_PLAIN, TIER_COOP, TIER_SPLIT3 = "plain", "coop", "split3"
TIERS = (TIER_PLAIN, TIER_COOP, TIER_SPLIT3)


def tier_kernel_name(tier: str) -> str:
    return f"flydsl_qkvfront_{tier}"


_KEYS = ["gfx", "cu_num", "M", "N", "K", "w_scale_block"]


@functools.cache
def _load_tuned() -> dict:
    path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_QKV_FRONT_FILE
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df = df.sort_values("us", kind="stable").drop_duplicates(_KEYS)
    return df.set_index(_KEYS).to_dict("index")


def tier_applies(tier: str, m: int) -> bool:
    if tier == TIER_PLAIN:
        return True
    if tier == TIER_COOP:
        return MIN_ROWS <= m <= COOP_MAX_ROWS
    return tier == TIER_SPLIT3 and COOP_MAX_ROWS < m <= MAX_ROWS


@functools.lru_cache(maxsize=1024)
def front_tier(m: int, n: int = N, k: int = K, w_scale_block: str = "1x32") -> str:
    """The tuned tier of an ``m``-row front (``plain`` when untuned, off gfx950, or not this projection)."""
    if (n, k) != (N, K) or not (MIN_ROWS <= m <= MAX_ROWS) or get_gfx() != "gfx950":
        return TIER_PLAIN
    gfx, cu = get_gfx(), get_cu_num()
    row, _ = find_padded_m_row(
        _load_tuned(), lambda pm: (gfx, cu, pm, n, k, w_scale_block), m, n, k
    )
    if row is None or row["libtype"] != "flydsl" or str(row["kernelId"]) != KERNEL_ID:
        return TIER_PLAIN
    for tier in TIERS:
        if row["kernelName"] == tier_kernel_name(tier):
            # a row padded across the 64 / 65 boundary must still be a tier that serves this m
            return tier if tier_applies(tier, m) else TIER_PLAIN
    return TIER_PLAIN


def chain_weight_scale(weight_scale: Tensor) -> Tensor:
    """Row-major E8M0 weight scale [2560, 192] -> the shuffled layout the fused tiers' GEMM reads (once, at load)."""
    from .shuffle import shuffle_scale_a16w4

    return shuffle_scale_a16w4(
        weight_scale.view(dtypes.fp8_e8m0).contiguous(), 1, False
    )


class _Deferred(NamedTuple):
    m: int
    ptr: int  # data_ptr of the uncomputed QKV tensor handed out
    tier: str
    codes: Tensor
    scales: Tensor
    weight: Tensor
    chain_scale: Tensor


# One parked front per thread (TBO decode runs its two micro-batches on two threads, each projection -> insert in turn).
_STATE = threading.local()


def take_deferred(qkv: Tensor) -> "_Deferred | None":
    """The parked front of this thread if ``qkv`` is the tensor it handed out, else None. Clears the slot."""
    d = getattr(_STATE, "deferred", None)
    if d is None:
        return None
    _STATE.deferred = None
    if d.ptr != qkv.data_ptr() or d.m != qkv.shape[0]:
        raise RuntimeError(
            f"qkv front: a deferred {d.m}-row projection was parked but the insert got a different tensor "
            f"({qkv.shape[0]} rows, ptr match {d.ptr == qkv.data_ptr()}); its consumer must be "
            "fused_qknorm_idxrqknorm on the tensor gemm_a8w8_qkv_front returned"
        )
    return d


def gemm_a8w8_qkv_front(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    chain_scale: Tensor,
    *,
    defer_ok: bool,
    tier: str | None = None,
) -> Tensor:
    """bf16 ``x [M, 6144]`` @ the preshuffled (16, 16) MXFP8 ``weight [2560, 6144]`` (row-major E8M0 ``weight_scale``,
    ``chain_scale`` = :func:`chain_weight_scale` of it) -> bf16 ``[M, 2560]``, or an uncomputed tensor of that shape
    when the tuned tier is a fused one and ``defer_ok`` (see the module docstring). ``tier`` overrides the table (tuner).
    """
    from .flydsl.kernels import qkv_chain as _chain
    from .flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled
    from .gemm_op_a8w8 import gemm_a8w8_blockscale
    from .quant import per_group_quant_hip

    m = x.shape[0]
    _STATE.deferred = None
    if tier is None:
        tier = front_tier(m) if defer_ok else TIER_PLAIN
    if tier != TIER_PLAIN:
        if not tier_applies(tier, m):
            raise ValueError(f"qkv front tier {tier} does not serve {m} rows")
        codes, scales = mxfp8_quantize_shuffled(x, _chain.code_rows(m))
        qkv = torch.empty((m, N), dtype=dtypes.bf16, device=x.device)
        _STATE.deferred = _Deferred(
            m, qkv.data_ptr(), tier, codes, scales, weight, chain_scale
        )
        return qkv
    xq, xs = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    return gemm_a8w8_blockscale(
        xq,
        weight,
        xs,
        weight_scale.view(dtypes.fp8_e8m0),
        dtypes.bf16,
        isBpreshuffled=True,
    )


def run_deferred(
    d: _Deferred,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    index_q_norm_weight: Tensor,
    index_k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    slot_mapping: Tensor,
    index_slot_mapping: Tensor,
    kv_cache: Tensor,
    index_cache: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    q_out: Tensor,
    index_q_out: Tensor,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
) -> None:
    """The fused kernel of a parked front: GEMM + norm + RoPE + NVFP4 inserts (called by ``fused_qknorm_idxrqknorm``)."""
    from .flydsl.kernels import qkv_chain as _chain

    entry = (
        _chain.qkv_chain_cooperative_insert
        if d.tier == TIER_COOP
        else _chain.qkv_chain_split_insert
    )
    entry(
        d.codes.view(dtypes.fp8),
        d.scales,
        d.weight,
        d.chain_scale,
        d.m,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        k_scale,
        v_scale,
        q_out,
        index_q_out,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        eps=eps,
    )


def warm_qkv_front(weight: Tensor, chain_scale: Tensor) -> None:
    """Eager calls at load: compile every fused tier's row configs and create the counter pools outside graph capture
    (a first call inside capture raises). Every slot is -1, so no page is written."""
    dev = weight.device
    nw = torch.zeros(128, dtype=dtypes.bf16, device=dev)
    cs = torch.zeros((128, 64), dtype=dtypes.bf16, device=dev)
    kv = torch.zeros((1, 2, 128, 72), dtype=torch.uint8, device=dev)
    idx = torch.zeros((1, 128, 72), dtype=torch.uint8, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    # one row per config: the one-launch chain (17-64) and the three 65-128 row configs (80 / 96 / 128 rows of codes)
    for m in (17, 33, 49, 65, 81, 97):
        tier = TIER_COOP if m <= COOP_MAX_ROWS else TIER_SPLIT3
        x = torch.zeros((m, K), dtype=dtypes.bf16, device=dev)
        qkv = gemm_a8w8_qkv_front(
            x, weight, weight, chain_scale, defer_ok=True, tier=tier
        )
        d = take_deferred(qkv)
        skip = torch.full((m,), -1, dtype=torch.int64, device=dev)
        run_deferred(
            d, nw, nw, nw, nw, cs, torch.zeros(m, dtype=torch.int64, device=dev),
            skip, skip, kv, idx, one, one,
            torch.empty((m, 2048), dtype=dtypes.bf16, device=dev),
            torch.empty((m, 128), dtype=dtypes.fp8, device=dev),
            16, 1, 1, 1e-6,
        )  # fmt: skip
    torch.cuda.synchronize(dev)
