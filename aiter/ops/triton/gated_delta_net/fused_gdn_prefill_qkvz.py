# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Qwen3-Next Gated DeltaNet *prefill*, with an FP8 group-quant epilogue.

A fused Gluon op replaces the prefill chain ``causal_conv1d_split_qkv ->
fused_gdn_gating -> chunk_gated_delta_rule -> gated_rmsnorm_fp8_group_quant``
with six tight launches (``_prepare_inputs_tiled`` folds split + conv + SiLU +
QK-norm + gating prep; ``_update_conv_state``; ``_chunk_offsets``;
``_prepare_chunk_factors``; ``_propagate_chunks`` runs the fp32 chunked scan;
``_output_norm_quant`` folds the gated RMSNorm and the per-head group-128 FP8
quant). The win is the intra-launch fusion and dropped intermediate HBM traffic,
not a single launch; the FP8 activations a block-FP8 ``out_proj`` consumes are
emitted by the epilogue launch, so there is no separate quantization kernel.

This is the *prefill sibling* of :func:`fused_gdn_decode_qkvz`. It shares the
same Qwen3-Next GDN math -- a per-head ``exp(-exp(A_log) * softplus(a +
dt_bias))`` decay, a SiLU output gate, ``num_v_heads == 2 * num_k_heads`` and a
biased convolution -- but consumes variable-length ragged batches
(``cu_seqlens`` + ``has_initial_state``) and runs the FP32 chunked delta-rule
scan instead of the single-token recurrent update. Prefill spans a wide token
range, so it dispatches one of four autotuned M-tile schedules rather than a
single kernel (see :func:`_select_tile_key`).

Support is deliberately narrow and is reported by
:func:`fused_gdn_prefill_qkvz_supported` rather than asserted, so callers fall
back to the four-kernel chain instead of crashing. Two hard gates:

* gfx950 only -- the kernels use CDNA buffer addressing, hand-written inline asm
  and explicit register layouts.
* Triton >= 3.8 -- the tiles are written in the Gluon dialect. On Triton 3.7 (or
  any build without ``triton.experimental.gluon``) the gate returns ``False`` so
  the caller falls back; importing this module never raises, and the Gluon/HIP
  imports only fire once a tile is actually dispatched on a supported device.
"""

import functools
import re
from typing import Optional, Tuple

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

# NB: the Gluon tile kernels are imported lazily inside :func:`_load_tile`, not at
# module load. That keeps this module -- and in particular
# :func:`fused_gdn_prefill_qkvz_supported`, the gate a caller probes -- importable
# and callable on any platform (older Triton, non-gfx950, CUDA), so the caller can
# fall back to the four-kernel chain without the Gluon/HIP stack present.

# The only FP8 output dtype the gfx950 tiles emit (per-head group-128 quant).
_QUANT_DTYPE = torch.float8_e4m3fn
_HEAD_DIM = 128
_CONV_WIDTH = 4

# Covered (tokens, batch) ranges. Full coverage holds for tokens in
# [_MIN_TOKENS, _MAX_TOKENS] and batch in [1, _MAX_BATCH]; the ranges below only
# split that region between tile schedules.
_MIN_TOKENS = 1024
_MAX_TOKENS = 16384
_MAX_BATCH = 64


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> Tuple[bool, str]:
    """Cached probe: can this Triton compile the tiles' Gluon dialect?

    gfx950 alone is not enough. The tiles use the Gluon dialect as it stands in
    Triton **3.8**; ROCm backported an *earlier, incompatible* Gluon into some
    3.7 builds where ``triton.experimental.gluon`` imports fine but the tiles
    fail to compile (e.g. layout ops in ``_block_inverse``). So the gate is an
    explicit ``>= 3.8`` version check, not just an import probe -- a 3.7 build
    returns ``(False, reason)`` and the caller falls back cleanly.
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
        import triton.experimental.gluon  # noqa: F401
    except ImportError:
        return False, f"triton.experimental.gluon unavailable (Triton {version})"
    return True, ""


def _select_tile_key(tokens: int, batch: int) -> Optional[str]:
    """Return the tile key for a covered (tokens, batch), else ``None``.

    Pure -- no Gluon import -- so the coverage decision is shared cheaply by
    :func:`fused_gdn_prefill_qkvz_supported` and the dispatcher. Mirrors the
    Artemis MI355 v1 kernel-pack profile
    (qwen3_next.gdn_prefill_group_fp8_quant.mi355.v1).
    """
    if not (_MIN_TOKENS <= tokens <= _MAX_TOKENS and 1 <= batch <= _MAX_BATCH):
        return None
    if tokens <= 3071:
        return "m1024_3071"
    if tokens <= 12288:
        return "m3072_16384"
    # tokens in [12289, 16384]
    if batch <= 5:
        return "m12289_16384_b1_5"
    if batch <= 15:
        return "m12289_16384_b6_15"
    return "m3072_16384"


def _load_tile(key: str):
    """Lazily import and return the Gluon tile entry for ``key``.

    The ``triton.experimental.gluon`` + ROCm ``libdevice`` imports happen here,
    on the dispatch path, never at module load.
    """
    from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
        _prefill_m1024_3071,
        _prefill_m3072_16384,
        _prefill_m12289_16384_b1_5,
        _prefill_m12289_16384_b6_15,
    )

    return {
        "m1024_3071": _prefill_m1024_3071,
        "m3072_16384": _prefill_m3072_16384,
        "m12289_16384_b1_5": _prefill_m12289_16384_b1_5,
        "m12289_16384_b6_15": _prefill_m12289_16384_b6_15,
    }[key]


def fused_gdn_prefill_qkvz_supported(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    quant_dtype: Optional[torch.dtype] = None,
) -> Tuple[bool, str]:
    """Report whether this call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``. The reason is meant to be
    logged once by the caller on its fallback path.
    """
    if get_arch() != "gfx950":
        return False, f"gfx950 only, got {get_arch()}"

    gluon_ok, gluon_reason = _gluon_supported()
    if not gluon_ok:
        return False, gluon_reason

    tensors = (
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
    )
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
        return False, "all inputs must be CUDA tensors"

    if delta_state.ndim != 4 or conv_state.ndim != 3:
        return (
            False,
            f"expected delta_state rank-4 and conv_state rank-3, got "
            f"{delta_state.ndim}/{conv_state.ndim}",
        )

    tokens = projected_qkvz.shape[0]
    batch = cache_indices.numel()

    # --- (tokens, batch) must land on a tile schedule -------------------------
    if _select_tile_key(tokens, batch) is None:
        return (
            False,
            f"(tokens={tokens}, batch={batch}) outside covered tiles "
            f"(tokens {_MIN_TOKENS}..{_MAX_TOKENS}, batch 1..{_MAX_BATCH})",
        )

    # --- head topology --------------------------------------------------------
    _, v_heads, head_v_dim, head_k_dim = delta_state.shape
    if head_v_dim != _HEAD_DIM or head_k_dim != _HEAD_DIM:
        return False, f"head dims must be {_HEAD_DIM}, got {head_k_dim}/{head_v_dim}"

    channels = conv_state.shape[1]
    if (channels - v_heads * head_v_dim) % (2 * head_k_dim):
        return False, f"conv channel count {channels} is not 2*KH*KD + VH*VD"
    k_heads = (channels - v_heads * head_v_dim) // (2 * head_k_dim)
    if v_heads != 2 * k_heads:
        return (
            False,
            f"requires num_v_heads == 2 * num_k_heads, got {v_heads}/{k_heads}",
        )

    # --- convolution layout ---------------------------------------------------
    if conv_state.shape[2] != _CONV_WIDTH - 1 or conv_weight.shape != (
        channels,
        _CONV_WIDTH,
    ):
        return (
            False,
            f"conv width must be {_CONV_WIDTH}, got weight {tuple(conv_weight.shape)}",
        )
    if conv_bias is None:
        return False, "conv bias is required (pass zeros if the model has none)"

    # --- ragged-batch index tensors ------------------------------------------
    if cu_seqlens.dtype is not torch.int32 or cu_seqlens.shape != (batch + 1,):
        return (
            False,
            f"cu_seqlens must be int32 [{batch + 1}], got {cu_seqlens.dtype} "
            f"{tuple(cu_seqlens.shape)}",
        )
    if cache_indices.dtype is not torch.int32:
        return False, f"cache_indices must be int32, got {cache_indices.dtype}"
    if has_initial_state.shape != (batch,):
        return (
            False,
            f"has_initial_state must be [{batch}], got {tuple(has_initial_state.shape)}",
        )
    if has_initial_state.dtype is not torch.bool:
        return False, f"has_initial_state must be bool, got {has_initial_state.dtype}"

    # --- dtypes ---------------------------------------------------------------
    if delta_state.dtype is not torch.float32:
        return False, f"delta_state must be fp32, got {delta_state.dtype}"
    bf16_args = (projected_qkvz, projected_ba, conv_state, conv_weight, conv_bias)
    if not all(t.dtype is torch.bfloat16 for t in bf16_args):
        return False, "packed projections and conv state/weight/bias must be bf16"
    if not conv_state.is_contiguous() or not delta_state.is_contiguous():
        return False, "conv and recurrent state pools must be contiguous"

    if quant_dtype is not None and quant_dtype is not _QUANT_DTYPE:
        return False, f"only {_QUANT_DTYPE} output is supported, got {quant_dtype}"

    return True, ""


def fused_gdn_prefill_qkvz(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Qwen3-Next GDN prefill (conv + gating + chunked delta + gated
    RMSNorm + group-128 FP8 quant) in a single Gluon launch.

    ``conv_state`` and ``delta_state`` are updated in place. Returns
    ``(normalized_bf16, conv_state, delta_state, quantized_fp8, scales)`` where
    ``normalized_bf16`` is the pre-quant RMSNorm output and ``quantized_fp8`` (a
    ``torch.float8_e4m3fn`` tensor) + ``scales`` are the per-head group-128 FP8
    activations for a block-FP8 ``out_proj``.

    Narrow by design; guard with :func:`fused_gdn_prefill_qkvz_supported` and
    fall back to the four-kernel chain when it returns ``False``.
    """
    ok, reason = fused_gdn_prefill_qkvz_supported(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        _QUANT_DTYPE,
    )
    if not ok:
        raise ValueError(f"fused_gdn_prefill_qkvz does not support this call: {reason}")

    key = _select_tile_key(projected_qkvz.shape[0], cache_indices.numel())
    tile = _load_tile(key)
    return tile(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        norm_weight,
        scale=scale,
        eps=eps,
    )
