# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Kimi-K3 KDA *prefill* (ragged-batch gated-delta core).

A single Gluon launch replaces the Kimi-K3 Delta-Attention prefill chain
``forget-B -> beta sigmoid -> causal conv -> fp32 chunked gated-delta scan ->
gated RMSNorm``. Unlike the decode op, the QKVG / beta-forget-A projections and
the final O projection are *not* fused: the caller supplies the already-split
``qkv``/``gate`` and the ``forget_a``/``beta`` slices, and applies the native
o_proj to the ``(m, 1536)`` result (which also performs the O-projection TP
all-reduce). The conv/recurrent state pools are updated in place and returned
unchanged (same storage).

Only the final-state (no intermediate recurrent snapshot) path is supported --
the Kimi-K3 chunked-prefill contract, where aligned prefix snapshots reuse the
final states.

The op consumes variable-length ragged batches via ``cu_seqlens`` +
``has_initial_state`` and dispatches the single ``m1024_8192_seq1_32`` schedule
(tokens in ``[1024, 8192]``, sequences in ``[1, 32]``), mirroring the Kimi-K3
MI355X TP8 kernel-pack profile.

Support is deliberately narrow and reported by
:func:`kimi_kda_fused_layer_prefill_supported` rather than asserted, so callers
fall back to the native KDA prefill path instead of crashing. Two hard gates:

* gfx950 only -- the tiles use CDNA4 buffer addressing and explicit MFMA
  register layouts.
* Triton >= 3.8 -- the tiles are written in the Gluon dialect. On Triton 3.7 (or
  any build without ``triton.experimental.gluon``) the gate returns ``False`` so
  the caller falls back; importing this module never raises, and the Gluon/HIP
  imports only fire once the tile is actually dispatched on a supported device.
"""

import functools
import re

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

# NB: the Gluon tile kernel is imported lazily inside :func:`_load_tile`, not at
# module load, so this module -- and the
# :func:`kimi_kda_fused_layer_prefill_supported` gate a caller probes -- stays
# importable and callable on any platform (older Triton, non-gfx950, CUDA).

_CORE = 1536
_HEAD_DIM = 128
_NUM_HEADS = 12
_QKV = 4608
_BETA = 12
# Covered (tokens, sequences) range for the single prefill schedule.
_MIN_TOKENS = 1024
_MAX_TOKENS = 8192
_MAX_SEQ = 32


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


def _covered(tokens: int, sequences: int) -> bool:
    """Pure (tokens, sequences) coverage check for the prefill schedule."""
    return _MIN_TOKENS <= tokens <= _MAX_TOKENS and 1 <= sequences <= _MAX_SEQ


def _row_major_ok(t: torch.Tensor, shape: tuple[int, ...]) -> bool:
    """2-D, unit inner stride, outer stride >= row width. The projection inputs
    (qkv/gate/forget_a/beta) arrive as column slices of a wider packed GEMM
    output, so they are row-major but not contiguous; the kernel reads them by
    stride."""
    return (
        t.ndim == 2
        and tuple(t.shape) == tuple(shape)
        and t.stride(1) == 1
        and t.stride(0) >= shape[1]
    )


def _load_tile():
    """Lazily import and return the Gluon prefill entry point.

    The ``triton.experimental.gluon`` + ROCm ``libdevice`` imports happen here,
    on the dispatch path, never at module load.
    """
    from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.fused_kda_layer_prefill import (
        fused_kda_prefill,
    )

    return fused_kda_prefill


def kimi_kda_fused_layer_prefill_supported(
    qkv: torch.Tensor,
    gate: torch.Tensor,
    forget_a: torch.Tensor,
    beta: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[bool, str]:
    """Report whether this prefill call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``; the reason is meant to be
    logged once by the caller on its fallback path. Mirrors the Artemis prefill
    static/runtime contract.
    """
    if get_arch() != "gfx950":
        return False, f"gfx950 only, got {get_arch()}"

    gluon_ok, gluon_reason = _gluon_supported()
    if not gluon_ok:
        return False, gluon_reason

    if not isinstance(qkv, torch.Tensor) or not qkv.is_cuda or qkv.ndim != 2:
        return False, "qkv must be a 2-D CUDA matrix"
    m = qkv.shape[0]
    device = qkv.device

    if not isinstance(state, torch.Tensor) or state.ndim != 4 or state.shape[0] <= 0:
        return False, "temporal cache must be rank-4 with positive capacity"
    slots = state.shape[0]

    if (
        not isinstance(state_indices, torch.Tensor)
        or state_indices.dtype != torch.int32
    ):
        return False, "cache indices must be int32"
    sequences = state_indices.numel()
    if not _covered(m, sequences):
        return (
            False,
            f"(tokens={m}, sequences={sequences}) outside covered schedule "
            f"(tokens {_MIN_TOKENS}..{_MAX_TOKENS}, sequences 1..{_MAX_SEQ})",
        )

    # Projection inputs are row-major column slices of a packed GEMM output
    # (not contiguous); the kernel reads them by stride.
    row_major = (
        ("qkv", qkv, (m, _QKV)),
        ("gate", gate, (m, _CORE)),
        ("forget_a", forget_a, (m, _HEAD_DIM)),
        ("beta", beta, (m, _BETA)),
    )
    for name, t, shape in row_major:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if t.dtype != torch.bfloat16:
            return False, f"{name} must be bf16, got {t.dtype}"
        if not _row_major_ok(t, shape):
            return False, f"{name} must be row-major {shape}, got stride {t.stride()}"

    bf16 = (
        ("forget-B weight", forget_weight, (_CORE, _HEAD_DIM)),
        ("gated RMSNorm weight", norm_weight, (_HEAD_DIM,)),
    )
    for name, t, shape in bf16:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if (
            t.dtype != torch.bfloat16
            or tuple(t.shape) != shape
            or not t.is_contiguous()
        ):
            return False, f"{name} must be contiguous bf16 {shape}"

    f32 = (
        ("convolution weights", conv_weight, (_QKV, 4)),
        ("A_log", a_log, (_NUM_HEADS,)),
        ("dt_bias", dt_bias, (_CORE,)),
    )
    for name, t, shape in f32:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if t.dtype != torch.float32 or tuple(t.shape) != shape or not t.is_contiguous():
            return False, f"{name} must be contiguous fp32 {shape}"

    if (
        state.dtype != torch.float32
        or tuple(state.shape) != (slots, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM)
        or not state.is_contiguous()
    ):
        return False, "temporal cache must be contiguous fp32 [slots,12,128,128]"
    if (
        not isinstance(conv_state, torch.Tensor)
        or conv_state.dtype != torch.bfloat16
        or tuple(conv_state.shape) != (slots, 3, _QKV)
        or not conv_state.is_contiguous()
    ):
        return False, "convolution cache must be contiguous bf16 [slots,3,4608]"

    if tuple(state_indices.shape) != (sequences,) or not state_indices.is_contiguous():
        return False, f"cache indices must be contiguous [{sequences}]"
    if (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.dtype != torch.int32
        or tuple(cu_seqlens.shape) != (sequences + 1,)
        or not cu_seqlens.is_contiguous()
    ):
        return False, f"cu_seqlens must be contiguous int32 [{sequences + 1}]"
    if (
        not isinstance(has_initial_state, torch.Tensor)
        or has_initial_state.dtype != torch.bool
        or tuple(has_initial_state.shape) != (sequences,)
    ):
        return False, f"has_initial_state must be bool [{sequences}]"

    return True, ""


def kimi_kda_fused_layer_prefill(
    qkv: torch.Tensor,
    gate: torch.Tensor,
    forget_a: torch.Tensor,
    beta: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    lower_bound: float = -5.0,
    norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Kimi-K3 KDA prefill (forget-B + beta sigmoid + conv + fp32 chunked
    gated-delta scan + gated RMSNorm) in a single Gluon launch.

    ``conv_state`` and ``state`` are updated in place. Returns
    ``(normalized, conv_state, state)`` where ``normalized`` is the ``(m, 1536)``
    gated-RMSNorm output the caller feeds to the native o_proj (which performs
    the O-projection TP all-reduce).

    Narrow by design; guard with
    :func:`kimi_kda_fused_layer_prefill_supported` and fall back to the native
    KDA prefill path when it returns ``False``.
    """
    ok, reason = kimi_kda_fused_layer_prefill_supported(
        qkv,
        gate,
        forget_a,
        beta,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        cu_seqlens,
        has_initial_state,
    )
    if not ok:
        raise ValueError(
            f"kimi_kda_fused_layer_prefill does not support this call: {reason}"
        )

    tile = _load_tile()
    return tile(
        qkv,
        gate,
        forget_a,
        beta,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        cu_seqlens,
        has_initial_state,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
    )
