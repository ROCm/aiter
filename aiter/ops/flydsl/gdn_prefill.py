# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated DeltaNet prefill front end on gfx942 (FlyDSL).

One launch reads the in_proj rows in the flat layout of Qwen3.5 / Qwen3.8
(``qkvz`` = [q | k | v | z]) and runs the varlen causal conv1d (width 4) + SiLU, the
q/k/v split and the q/k L2 norm. It replaces a conv + split kernel and two L2 norm
kernels, and updates the conv state pool in place. Callers pass the returned q/k to
the chunked delta rule with ``use_qk_l2norm_in_kernel=False``.

Callers gate on :func:`gdn_prefill_front_supported`. The interleaved Qwen3-Next
layout (``qkvz_layout="interleaved"``) has the same shapes and is rejected.
"""

from __future__ import annotations

import functools

import torch

from .kernels.gdn_prefill_front import CHUNK_TOKENS, create_gdn_prefill_front_kernel
from .kernels.tensor_shim import _run_compiled

__all__ = ["flydsl_gdn_prefill_front", "gdn_prefill_front_supported"]

_HEAD_DIM = 128
_CONV_WIDTH = 4
_QKVZ_LAYOUT = "flat"
# Per-rank (k heads, v heads) of Qwen3.8 GDN (16/48 heads) at TP1 and TP2: the only
# head counts this kernel is tested and timed for. TP4 (4/12) is left out on purpose:
# with its 8 KiB qkvz rows the time is 133 or 210 us on MI308X depending on where
# qkvz and q/k/v sit relative to each other; 16 and 32 KiB rows (TP2, TP1) are stable.
_QWEN38_HEADS = ((16, 48), (8, 24))


def gdn_prefill_front_supported(
    qkvz: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    qkvz_layout: str = "flat",
) -> bool:
    """True when :func:`flydsl_gdn_prefill_front` can serve these tensors.

    Requires gfx942, the Qwen3.8 per-rank head counts (``num_k_heads``,
    ``num_v_heads``) of ``(16, 48)`` or ``(8, 24)`` (TP1/TP2), the flat ``[q|k|v|z]``
    in_proj layout (``qkvz_layout="flat"``; the interleaved Qwen3-Next layout returns
    False), bf16
    activations, weights and conv state, 128-wide heads, width-4 conv weights
    ``[2HD + HVD, 4]``, a conv state pool ``[slots, conv_dim, 3]`` with contiguous
    slot rows, 16-byte aligned rows and less than 2 GiB of ``qkvz``. Callers keep their unfused path when
    this returns False.
    """
    if qkvz_layout != _QKVZ_LAYOUT:
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        if get_gfx_runtime() != "gfx942":
            return False
    except Exception:  # noqa: BLE001 - no device or unknown arch
        return False
    D, H, HV = _HEAD_DIM, num_k_heads, num_v_heads
    if (H, HV) not in _QWEN38_HEADS:
        return False
    C = 2 * H * D + HV * D
    return (
        qkvz.dim() == 2
        and qkvz.dtype == conv_weight.dtype == conv_state.dtype == torch.bfloat16
        and qkvz.stride(-1) == 1
        and qkvz.stride(0) % 8 == 0
        and qkvz.data_ptr() % 16 == 0
        and qkvz.shape[-1] >= C
        # 32-bit buffer offsets: the rows read must stay below 2 GiB
        and qkvz.shape[0] * qkvz.stride(0) * qkvz.element_size() < 2**31
        and conv_weight.is_contiguous()
        and conv_weight.data_ptr() % 16 == 0
        and tuple(conv_weight.shape) == (C, _CONV_WIDTH)
        and conv_state.dim() == 3
        and conv_state.shape[1:] == (C, _CONV_WIDTH - 1)
        and conv_state.stride()[1:] == (_CONV_WIDTH - 1, 1)
        and conv_state.data_ptr() % 16 == 0
        and conv_state.stride(0) % 8 == 0
    )


@functools.cache
def _launcher(device_index: int, num_k_heads: int, num_v_heads: int):
    """One launcher per GPU: a compiled handle built on one device does not launch
    on another, and ``_run_compiled`` caches the first handle it builds."""
    return create_gdn_prefill_front_kernel.__wrapped__(num_k_heads, num_v_heads)


def _require(ok: bool, msg: str):
    # Raised, not asserted: python -O strips an assert.
    if not ok:
        raise ValueError(f"flydsl_gdn_prefill_front: {msg}")


def flydsl_gdn_prefill_front(
    qkvz: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    max_seqlen: int,
    qkvz_layout: str = "flat",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conv + SiLU + split + q/k L2 norm of a varlen prefill batch.

    ``qkvz`` [T, >= 2HD + HVD] holds the packed sequences given by ``query_start_loc``
    [N + 1]; ``cache_indices`` [N] picks each sequence's conv state slot (-1 = pad, no
    output, no state update); ``has_initial_state`` [N] says whether the slot holds
    the state of earlier tokens. ``max_seqlen`` (host int) sizes the grid and must be
    at least the longest sequence.

    Returns q, k (L2-normalised) ``[1, T, H, 128]`` and v ``[1, T, HV, 128]``, bf16,
    contiguous. The conv state of each live slot is updated in place.
    """
    D, H, HV = _HEAD_DIM, num_k_heads, num_v_heads
    _require(
        qkvz_layout == _QKVZ_LAYOUT,
        f"reads the flat [q|k|v|z] layout (Qwen3.5/3.8), got qkvz_layout={qkvz_layout!r}",
    )
    _require(
        gdn_prefill_front_supported(qkvz, conv_weight, conv_state, H, HV),
        "unsupported arch or tensors; gate on gdn_prefill_front_supported()",
    )

    def _index(t: torch.Tensor, name: str) -> torch.Tensor:
        _require(
            t.dtype in (torch.int32, torch.int64) and t.dim() == 1,
            f"{name} must be a 1-D int32 or int64 tensor",
        )
        _require(t.device == qkvz.device, f"{name} must be on {qkvz.device}")
        return t.to(torch.int32).contiguous()

    cache_indices = _index(cache_indices, "cache_indices")
    query_start_loc = _index(query_start_loc, "query_start_loc")
    N = cache_indices.numel()
    _require(
        query_start_loc.numel() == N + 1, "query_start_loc must have N + 1 entries"
    )
    T = qkvz.shape[0]
    q = qkvz.new_empty(1, T, H, D)
    k = qkvz.new_empty(1, T, H, D)
    v = qkvz.new_empty(1, T, HV, D)
    if T == 0 or N == 0:
        return q, k, v
    if has_initial_state is None:
        has_init = torch.zeros(N, dtype=torch.int32, device=qkvz.device)
    else:
        _require(
            has_initial_state.numel() == N, "has_initial_state must have N entries"
        )
        _require(
            has_initial_state.device == qkvz.device,
            f"has_initial_state must be on {qkvz.device}",
        )
        has_init = has_initial_state.to(torch.int32).contiguous()
    max_chunks = max(1, (int(max_seqlen) + CHUNK_TOKENS - 1) // CHUNK_TOKENS)
    with torch.cuda.device(qkvz.device):
        exe = _launcher(torch.cuda.current_device(), H, HV)
        _run_compiled(
            exe,
            qkvz,
            conv_weight,
            conv_state,
            cache_indices,
            has_init,
            query_start_loc,
            q.view(-1),
            k.view(-1),
            v.view(-1),
            qkvz.stride(0) // 8,
            conv_state.stride(0) * conv_state.element_size(),
            max_chunks,
            N,
            torch.cuda.current_stream(),
        )
    return q, k, v
