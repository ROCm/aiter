# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Kimi-K3 KDA *decode* layer (whole rank-local linear-attention block).

A single Gluon launch replaces the entire Kimi-K3 Delta-Attention decode step:
the QKVG projection, the beta/forget-A projection, forget-B, the beta sigmoid,
the causal convolution, the single-token gated-delta recurrence, the gated
RMSNorm, and the output (O) projection -- all fused. The op reads the raw
hidden state ``(m, 7168)`` and writes the O-projection *partial* ``(m, 7168)``;
the conv/recurrent state pools are updated in place and returned unchanged
(same storage).

Because the O matmul is fused but tensor-parallel, the returned ``(m, 7168)`` is
the rank-local partial -- the caller is responsible for the O-projection
all-reduce across the attention TP group, exactly as the native
``RowParallelLinear`` o_proj would do. This op deliberately stops at the kernel
boundary and does not reduce.

Dispatch routes by batch range to two general schedules: ``kda_layer_decode_m64``
for ``m <= 64`` and ``kda_layer_decode_m256`` for ``64 < m <= 256``. An on-device
tile-vs-tile bake-off across every real decode tier found these two dominate --
each is correct over its range and within a few microseconds of the per-tier
best -- while the remaining per-size Kimi-K3 kernel-pack variants are never
fastest and the general serial catch-all is 2-8x slower at large ``m``. Decode
batches are exact (single tokens, no ragged lengths), so the range split is
unambiguous.

Support is deliberately narrow and reported by
:func:`kimi_kda_fused_layer_decode_supported` rather than asserted, so callers
fall back to the native KDA decode path instead of crashing. Two hard gates:

* gfx950 only -- the tiles use CDNA4 buffer addressing and explicit MFMA
  register layouts.
* Triton >= 3.8 -- the tiles are written in the Gluon dialect. On Triton 3.7 (or
  any build without ``triton.experimental.gluon``) the gate returns ``False`` so
  the caller falls back; importing this module never raises, and the Gluon/HIP
  imports only fire once a tile is actually dispatched on a supported device.
"""

import functools
import re

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

# NB: the Gluon tile kernels are imported lazily inside :func:`_load_tile`, not at
# module load, so this module -- and in particular the
# :func:`kimi_kda_fused_layer_decode_supported` gate a caller probes -- stays
# importable and callable on any platform (older Triton, non-gfx950, CUDA).

_HIDDEN = 7168
_CORE = 1536
_HEAD_DIM = 128
_NUM_HEADS = 12
_MAX_M = 256

# (m) -> tile key. An on-device tile-vs-tile bake-off across every real decode
# batch tier (see the bench) showed the two general schedules dominate: m64 is
# correct and near-optimal for the whole low range (m<=64, ~54us), and m256 is
# correct and optimal for the high range (m>64, 64..109us). The per-batch-size
# Artemis variants (m1/m2/m4/m32/m128) are never fastest at a real tier, and the
# m1_256 catch-all is a serial m=1-style kernel that is 2-8x slower at large m,
# so a dispatch keyed on exact m silently routes most tiers to it. Routing by
# range to {m64, m256} is both faster and simpler. Decode batches are exact
# (single tokens, no ragged lengths), so this range split is unambiguous.


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> tuple[bool, str]:
    """Cached probe: can this Triton compile the tiles' Gluon dialect?

    gfx950 alone is not enough. The tiles use the Gluon dialect as it stands in
    Triton **3.8**; ROCm backported an *earlier, incompatible* Gluon into some
    3.7 builds where ``triton.experimental.gluon`` imports fine but the tiles
    fail to compile. So the gate is an explicit ``>= 3.8`` version check, not
    just an import probe -- a 3.7 build returns ``(False, reason)`` and the
    caller falls back cleanly.
    """
    try:
        import triton
    except ImportError as exc:
        return False, f"triton not importable ({exc})"
    version = triton.__version__
    matched = re.match(r"(\d+)\.(\d+)", version or "")
    if matched is None or (int(matched.group(1)), int(matched.group(2))) < (3, 8):
        return False, f"Triton >= 3.8 required for this Gluon dialect, got {version}"
    try:
        import triton.experimental.gluon
    except ImportError:
        return False, f"triton.experimental.gluon unavailable (Triton {version})"
    return True, ""


def _select_tile_key(m: int) -> str | None:
    """Return the tile key for a covered decode batch ``m``, else ``None``.

    Pure -- no Gluon import -- so the coverage decision is shared cheaply by the
    support predicate and the dispatcher.
    """
    if not (1 <= m <= _MAX_M):
        return None
    return "m64" if m <= 64 else "m256"


def _load_tile(key: str):
    """Lazily import and return the Gluon decode entry point for ``key``.

    The ``triton.experimental.gluon`` + ROCm ``libdevice`` imports happen here,
    on the dispatch path, never at module load.
    """
    from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.fused_kda_layer_decode import (
        kda_layer_decode_m64,
        kda_layer_decode_m256,
    )

    return {
        "m64": kda_layer_decode_m64,
        "m256": kda_layer_decode_m256,
    }[key]


def _row_major_ok(t: torch.Tensor, shape: tuple[int, ...]) -> bool:
    """2-D, unit inner stride, outer stride >= row width (the tiles' contract)."""
    return (
        t.ndim == 2
        and tuple(t.shape) == tuple(shape)
        and t.stride(1) == 1
        and t.stride(0) >= shape[1]
    )


def kimi_kda_fused_layer_decode_supported(
    x: torch.Tensor,
    qkvg_weight: torch.Tensor,
    beta_forget_weight: torch.Tensor,
    output_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[bool, str]:
    """Report whether this decode call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``; the reason is meant to be
    logged once by the caller on its fallback path. Mirrors the Artemis
    ``_validate_kda_inputs`` metadata contract 1:1.
    """
    if get_arch() != "gfx950":
        return False, f"gfx950 only, got {get_arch()}"

    gluon_ok, gluon_reason = _gluon_supported()
    if not gluon_ok:
        return False, gluon_reason

    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim != 2:
        return False, "hidden state must be a 2-D CUDA matrix"
    m = x.shape[0]
    if _select_tile_key(m) is None:
        return False, f"decode batch m={m} outside covered range [1, {_MAX_M}]"
    device = x.device

    if not isinstance(state, torch.Tensor) or state.ndim != 4 or state.shape[0] <= 0:
        return False, "temporal cache must be rank-4 with positive capacity"
    slots = state.shape[0]

    row_major = (
        ("hidden", x, (m, _HIDDEN)),
        ("QKVG weight", qkvg_weight, (6144, _HIDDEN)),
        ("beta/forget-A weight", beta_forget_weight, (144, _HIDDEN)),
        ("O weight", output_weight, (_HIDDEN, _CORE)),
        ("forget-B weight", forget_weight, (_CORE, _HEAD_DIM)),
    )
    for name, t, shape in row_major:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if t.dtype != torch.bfloat16:
            return False, f"{name} must be bf16, got {t.dtype}"
        if not _row_major_ok(t, shape):
            return False, f"{name} must be row-major {shape}, got {tuple(t.shape)}"

    contiguous = (
        ("convolution weights", conv_weight, (4608, 4), torch.float32),
        ("A_log", a_log, (_NUM_HEADS,), torch.float32),
        ("dt_bias", dt_bias, (_CORE,), torch.float32),
        ("gated RMSNorm weight", norm_weight, (_HEAD_DIM,), torch.bfloat16),
        ("convolution cache", conv_state, (slots, 3, 4608), torch.bfloat16),
        (
            "temporal cache",
            state,
            (slots, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM),
            torch.float32,
        ),
    )
    for name, t, shape, dtype in contiguous:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
            return False, f"{name} must be contiguous {dtype} {shape}"

    if not isinstance(state_indices, torch.Tensor) or state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False, "cache indices must be int32/int64"
    if tuple(state_indices.shape) != (m,) or not state_indices.is_contiguous():
        return False, f"cache indices must be contiguous [{m}]"

    return True, ""


def kimi_kda_fused_layer_decode(
    x: torch.Tensor,
    qkvg_weight: torch.Tensor,
    beta_forget_weight: torch.Tensor,
    output_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    lower_bound: float = -5.0,
    norm_eps: float = 1e-5,
    output_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Kimi-K3 KDA decode layer (projections + conv + gated-delta
    recurrence + gated RMSNorm + O matmul) in a single Gluon launch.

    ``conv_state`` and ``state`` are updated in place. Returns
    ``(o_partial, conv_state, state)`` where ``o_partial`` is the rank-local
    ``(m, 7168)`` O-projection output (written into ``output_tensor`` when
    given); the caller must all-reduce it across the attention TP group.

    Narrow by design; guard with
    :func:`kimi_kda_fused_layer_decode_supported` and fall back to the native
    KDA decode path when it returns ``False``.
    """
    ok, reason = kimi_kda_fused_layer_decode_supported(
        x,
        qkvg_weight,
        beta_forget_weight,
        output_weight,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
    )
    if not ok:
        raise ValueError(
            f"kimi_kda_fused_layer_decode does not support this call: {reason}"
        )

    tile = _load_tile(_select_tile_key(x.shape[0]))
    return tile(
        x,
        qkvg_weight,
        beta_forget_weight,
        output_weight,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        output_tensor=output_tensor,
    )
