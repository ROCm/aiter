# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MXFP8 QKV projection of a MiniMax-M3.1 TP4 sparse-attention decode layer with its norm / RoPE / NVFP4 insert fused.

A decode layer runs ``QKV = A @ W^T`` (N 2560, K 6144, preshuffled (16, 16) MXFP8 weight, 1x32 E8M0 scales) and then q / k /
index RMSNorm + partial RoPE + NVFP4 K / V / index-K insert (``aiter.fused_qknorm_idxrqknorm``). This module is ONE op with
two row-range implementations, picked statically by the row count ``m``:

* 1-64 rows: ``qkv_chain_cooperative_insert``, the QKV GEMM and the insert in ONE launch;
* 65-128 rows: ``qkv_chain_split_insert``, a split-K GEMM (``ks`` K parts, default 3) and a reduce + insert launch;
* anything else (fewer / more rows, another arch or shape): the plain path, ``per_group_quant_hip`` +
  ``gemm_a8w8_blockscale`` now and the unfused insert later.

Both fused kernels read the plain quantizer's row-major codes and E8M0 scales (no relayout).

The fusion crosses two calls, because the QKV projection and the attention insert are separate layers of a model:

1. ``gemm_a8w8_qkv_front(x, ...)`` quantizes ``x`` and, on a fused row range, returns an UNCOMPUTED ``[m, 2560]`` bf16
   tensor and parks the quantized activation in a per-thread slot;
2. ``aiter.fused_qknorm_idxrqknorm`` -- called next, on that very tensor, on the same thread -- takes the slot
   (``take_deferred``) and runs the fused kernel, which computes the QKV and inserts in one go.

Failure modes of this hand-off, and what is checked (everything else is the caller's contract, ``defer_ok``):

* the parked front is never consumed (the insert is skipped, the forward aborts between the two calls): the next
  ``gemm_a8w8_qkv_front`` on that thread raises instead of silently dropping it;
* the insert gets another tensor or another row count than the one handed out: ``take_deferred`` raises;
* the insert cannot consume the parked activation (not the NVFP4 TP4 layout of the fused kernels): raises;
* ``qkv`` is read by anything else than that insert (a split / norm op between the two calls) or the projection is
  replayed from a CUDA graph without Python (the quantizer replays, the hand-off does not): the tensor holds garbage and
  nothing can notice. A caller whose consumer is not that insert, or whose step runs inside such a graph, MUST pass
  ``defer_ok=False``: the result is then always the computed plain QKV;
* ``slot_mapping`` / ``index_slot_mapping`` rows that share one 8-slot octet are packed as ONE run into consecutive
  slots starting at the first row's slot (the decode layout: one request's tokens take consecutive slots). A gapped
  mapping inside an octet (rows to slots 0 and 2) or one octet reached by two separate runs is NOT supported: the
  packed run lands in consecutive slots, so the gapped row goes to the wrong slot. Slot ``-1`` (skip) is supported. A
  caller that cannot guarantee this must pass ``defer_ok=False`` (checking it needs a device sync, so it is not done);
* two threads (TBO micro-batches) each have their own slot, so each must run projection -> insert in turn.

Technique source of the fused kernels: amd-kernels ``mxfp8_qkv_norm_rope_nvfp4_insert`` (credit only, no import).
"""

import threading
from typing import NamedTuple

import torch
from torch import Tensor

from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..utility import dtypes

N, K = 2560, 6144  # 16 q heads + K + V + index q / k of 128, hidden 6144
MIN_ROWS, COOP_MAX_ROWS, MAX_ROWS = 1, 64, 128
TIER_PLAIN, TIER_COOP, TIER_SPLIT = "plain", "coop", "split"
DEFAULT_KS = 3  # measured best split factor of the 65-128 row tier (ks 1 / 2 / 4 / 6 are within the repo tolerance)


def front_tier(m: int) -> str:
    """The implementation that serves an ``m``-row front on this device."""
    if get_gfx() != "gfx950" or not (MIN_ROWS <= m <= MAX_ROWS):
        return TIER_PLAIN
    return TIER_COOP if m <= COOP_MAX_ROWS else TIER_SPLIT


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
    ks: int
    cfg: "tuple[int, int, bool, bool] | None"
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
    ks: int = DEFAULT_KS,
    cfg: "tuple[int, int, bool, bool] | None" = None,
) -> Tensor:
    """bf16 ``x [M, 6144]`` @ the preshuffled (16, 16) MXFP8 ``weight [2560, 6144]`` (row-major E8M0 ``weight_scale``,
    ``chain_scale`` = :func:`chain_weight_scale` of it) -> bf16 ``[M, 2560]``; or, on a fused row range and ``defer_ok``, an
    uncomputed tensor of that shape that ``aiter.fused_qknorm_idxrqknorm`` computes and consumes (see the module docstring
    for the contract). ``tier`` / ``ks`` / ``cfg`` override the static choice (tests, benchmarks): ``ks`` in
    {1, 2, 3, 4, 6} K parts of the split tier, ``cfg`` its (RT, NI, non-temporal, interleave) tile.
    """
    from .gemm_op_a8w8 import gemm_a8w8_blockscale
    from .quant import per_group_quant_hip

    m = x.shape[0]
    if (
        x.dim() != 2
        or x.shape[1] != K
        or x.dtype != dtypes.bf16
        or tuple(weight.shape) != (N, K)
    ):
        raise ValueError(
            f"gemm_a8w8_qkv_front serves only bf16 x [M, {K}] @ weight [{N}, {K}] (the MiniMax-M3.1 TP4 QKV "
            f"projection), got x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)}"
        )
    if ks not in (1, 2, 3, 4, 6):
        raise ValueError(f"gemm_a8w8_qkv_front: ks {ks} not in (1, 2, 3, 4, 6)")
    if getattr(_STATE, "deferred", None) is not None:
        stale, _STATE.deferred = _STATE.deferred, None
        raise RuntimeError(
            f"gemm_a8w8_qkv_front: the deferred {stale.m}-row front parked by the previous call was never consumed by "
            "fused_qknorm_idxrqknorm (aborted forward, or a layer that skipped its insert)"
        )
    if tier is None:
        tier = front_tier(m) if defer_ok else TIER_PLAIN
    if tier == TIER_COOP and not MIN_ROWS <= m <= COOP_MAX_ROWS:
        raise ValueError(
            f"the one-launch tier serves {MIN_ROWS}-{COOP_MAX_ROWS} rows, got {m}"
        )
    if tier == TIER_SPLIT and not COOP_MAX_ROWS < m <= MAX_ROWS:
        raise ValueError(
            f"the split-K tier serves {COOP_MAX_ROWS + 1}-{MAX_ROWS} rows, got {m}"
        )
    # the plain quantizer's row-major codes / E8M0 scales feed every tier (no relayout)
    xq, xs = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    if tier != TIER_PLAIN:
        qkv = torch.empty((m, N), dtype=dtypes.bf16, device=x.device)
        _STATE.deferred = _Deferred(
            m, qkv.data_ptr(), tier, ks, cfg, xq, xs, weight, chain_scale
        )
        return qkv
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

    if d.tier == TIER_COOP:
        entry, extra = _chain.qkv_chain_cooperative_insert, {}
    else:
        entry = _chain.qkv_chain_split_insert
        extra = {"ks": d.ks, "cfg": d.cfg}
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
        **extra,
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
    # one row per compiled config: the one-launch chain's four row tiles, the split tier's row-table tiles
    plans = {
        1: TIER_COOP,
        17: TIER_COOP,
        33: TIER_COOP,
        49: TIER_COOP,
        65: TIER_SPLIT,
        97: TIER_SPLIT,
        121: TIER_SPLIT,
    }
    for m, tier in plans.items():
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
