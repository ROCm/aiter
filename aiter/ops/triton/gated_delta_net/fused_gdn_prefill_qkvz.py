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

import torch

# NB: the Gluon tile kernels are imported lazily inside :func:`_load_tile`, not at
# module load. That keeps this module -- and in particular
# :func:`fused_gdn_prefill_qkvz_supported`, the gate a caller probes -- importable
# and callable on any platform (older Triton, non-gfx950, CUDA), so the caller can
# fall back to the four-kernel chain without the Gluon/HIP stack present.

# The only FP8 output dtype the gfx950 tiles emit (per-head group-128 quant).
_QUANT_DTYPE = torch.float8_e4m3fn

# Baked-in Qwen3-Next GDN topology. The gfx950 tiles hard-code this exact shape --
# packed ``[M, 3072]`` qkvz / ``[M, 16]`` ba addressing, 8-value-head launch grids
# and 2048 conv channels (see ``_gluon_kernels/.../fused_gdn_prefill_qkvz``). The
# support gate rejects anything else so the caller falls back rather than letting
# the fixed-stride tiles read out of bounds.
_K_HEADS = 4  # query/key heads
_V_HEADS = 8  # value heads (== 2 * _K_HEADS; also the z-gate head count)
_HEAD_DIM = 128
_CONV_WIDTH = 4
_CONV_CHANNELS = (2 * _K_HEADS + _V_HEADS) * _HEAD_DIM  # q + k + v conv = 2048
_QKVZ_WIDTH = (2 * _K_HEADS + 2 * _V_HEADS) * _HEAD_DIM  # q + k + v + z = 3072
_BA_WIDTH = 2 * _V_HEADS  # b + a gates = 16

# Covered (tokens, batch) ranges. Full coverage holds for tokens in
# [_MIN_TOKENS, _MAX_TOKENS] and batch in [1, _MAX_BATCH]; the ranges below only
# split that region between tile schedules.
_MIN_TOKENS = 1024
_MAX_TOKENS = 16384
_MAX_BATCH = 64


def _arch_supported() -> tuple[bool, str]:
    """gfx950 probe, guarded so architecture detection never raises at import.

    ``arch_info`` detection can touch the driver (and has an unguarded GPU
    fallback), so it is imported and called *here*, on the gate path, and any
    detection failure is reported as unsupported -- keeping this module import-safe
    on CPU-only / non-ROCm workers where the caller just takes the fallback chain.
    """
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001
        # Defensive: any detection failure (driver/subprocess/import) => unsupported.
        return False, f"architecture detection failed ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> tuple[bool, str]:
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
        import triton.experimental.gluon
    except ImportError:
        return False, f"triton.experimental.gluon unavailable (Triton {version})"
    return True, ""


def _select_tile_key(tokens: int, batch: int) -> str | None:
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
    """Lazily import and return the host launcher for tile ``key``.

    Only the dispatched tile's launcher is imported (and, transitively, its Gluon
    kernel module), so the ``triton.experimental.gluon`` + ROCm ``libdevice``
    imports happen here, on the dispatch path, never at module load. Each launcher
    lives in this package (``_gdn_prefill_launch_<key>``) and holds the torch/
    triton host orchestration; the kernel modules under ``_gluon_kernels`` stay
    torch-free.
    """
    import importlib

    launcher = importlib.import_module(
        f"aiter.ops.triton.gated_delta_net._gdn_prefill_launch_{key}"
    )
    return launcher.gdn_prefill_group_fp8_quant


def fused_gdn_prefill_qkvz_supported(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    quant_dtype: torch.dtype | None = None,
    *,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
) -> tuple[bool, str]:
    """Report whether this call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``. The reason is meant to be
    logged once by the caller on its fallback path.

    The gate is intentionally strict: the fixed-stride Gluon tiles assume the one
    baked topology (``_K_HEADS``/``_V_HEADS``/``_HEAD_DIM``) and exact packed
    widths, so every consumed tensor's shape, dtype, device and contiguity is
    validated here -- a permissive gate that returned ``True`` for a mismatched
    shape would let the tiles read out of bounds. ``a_log``/``dt_bias``/
    ``norm_weight`` are optional so the gate can also be used as a cheap coverage
    probe before those are materialized; when passed they are fully validated.
    """
    arch_ok, arch_reason = _arch_supported()
    if not arch_ok:
        return False, arch_reason

    gluon_ok, gluon_reason = _gluon_supported()
    if not gluon_ok:
        return False, gluon_reason

    # --- every consumed tensor must be a CUDA tensor, contiguous, one device ---
    required = {
        "projected_qkvz": projected_qkvz,
        "projected_ba": projected_ba,
        "conv_state": conv_state,
        "delta_state": delta_state,
        "cache_indices": cache_indices,
        "cu_seqlens": cu_seqlens,
        "has_initial_state": has_initial_state,
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
    }
    optional = {"a_log": a_log, "dt_bias": dt_bias, "norm_weight": norm_weight}
    named = {**required, **{k: v for k, v in optional.items() if v is not None}}
    if conv_bias is None:
        return False, "conv_bias is required (pass zeros if the model has none)"
    for name, t in named.items():
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return False, f"{name} must be a CUDA tensor"
        if not t.is_contiguous():
            return False, f"{name} must be contiguous"
    device = projected_qkvz.device
    if any(t.device != device for t in named.values()):
        return False, "all inputs must be on the same device"

    if delta_state.ndim != 4 or conv_state.ndim != 3:
        return (
            False,
            (
                f"expected delta_state rank-4 and conv_state rank-3, got "
                f"{delta_state.ndim}/{conv_state.ndim}"
            ),
        )

    tokens = projected_qkvz.shape[0]
    batch = cache_indices.numel()

    # --- (tokens, batch) must land on a tile schedule -------------------------
    if _select_tile_key(tokens, batch) is None:
        return (
            False,
            (
                f"(tokens={tokens}, batch={batch}) outside covered tiles "
                f"(tokens {_MIN_TOKENS}..{_MAX_TOKENS}, batch 1..{_MAX_BATCH})"
            ),
        )

    # --- baked head topology (tiles hard-code _K_HEADS/_V_HEADS/_HEAD_DIM) -----
    if tuple(delta_state.shape[1:]) != (_V_HEADS, _HEAD_DIM, _HEAD_DIM):
        return (
            False,
            (
                f"delta_state must be [N, {_V_HEADS}, {_HEAD_DIM}, {_HEAD_DIM}], got "
                f"{tuple(delta_state.shape)}"
            ),
        )
    if conv_state.shape[1] != _CONV_CHANNELS or conv_state.shape[2] != _CONV_WIDTH - 1:
        return (
            False,
            (
                f"conv_state must be [N, {_CONV_CHANNELS}, {_CONV_WIDTH - 1}], got "
                f"{tuple(conv_state.shape)}"
            ),
        )

    # --- packed projection widths (fixed-stride addressing) -------------------
    if projected_qkvz.shape != (tokens, _QKVZ_WIDTH):
        return (
            False,
            (
                f"projected_qkvz must be [{tokens}, {_QKVZ_WIDTH}], got "
                f"{tuple(projected_qkvz.shape)}"
            ),
        )
    if projected_ba.shape != (tokens, _BA_WIDTH):
        return (
            False,
            (
                f"projected_ba must be [{tokens}, {_BA_WIDTH}], got "
                f"{tuple(projected_ba.shape)}"
            ),
        )

    # --- convolution weight/bias layout ---------------------------------------
    if conv_weight.shape != (_CONV_CHANNELS, _CONV_WIDTH):
        return (
            False,
            (
                f"conv_weight must be [{_CONV_CHANNELS}, {_CONV_WIDTH}], got "
                f"{tuple(conv_weight.shape)}"
            ),
        )
    if conv_bias.shape != (_CONV_CHANNELS,):
        return (
            False,
            f"conv_bias must be [{_CONV_CHANNELS}], got {tuple(conv_bias.shape)}",
        )

    # --- ragged-batch index tensors ------------------------------------------
    if cu_seqlens.dtype is not torch.int32 or cu_seqlens.shape != (batch + 1,):
        return (
            False,
            (
                f"cu_seqlens must be int32 [{batch + 1}], got {cu_seqlens.dtype} "
                f"{tuple(cu_seqlens.shape)}"
            ),
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

    # --- gating / norm parameters (validated when supplied) -------------------
    if a_log is not None and (
        a_log.shape != (_V_HEADS,) or a_log.dtype is not torch.float32
    ):
        return (
            False,
            f"a_log must be fp32 [{_V_HEADS}], got {a_log.dtype} {tuple(a_log.shape)}",
        )
    if dt_bias is not None and (
        dt_bias.shape != (_V_HEADS,) or dt_bias.dtype is not torch.bfloat16
    ):
        return (
            False,
            (
                f"dt_bias must be bf16 [{_V_HEADS}], got {dt_bias.dtype} "
                f"{tuple(dt_bias.shape)}"
            ),
        )
    if norm_weight is not None and (
        norm_weight.shape != (_HEAD_DIM,) or norm_weight.dtype is not torch.bfloat16
    ):
        return (
            False,
            (
                f"norm_weight must be bf16 [{_HEAD_DIM}], got {norm_weight.dtype} "
                f"{tuple(norm_weight.shape)}"
            ),
        )

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Qwen3-Next GDN prefill (conv + gating + chunked delta + gated
    RMSNorm + group-128 FP8 quant) as a tight set of Gluon launches -- the win is
    the intra-launch fusion and the dropped intermediate HBM traffic, not a single
    launch.

    Narrow by design (gfx950 + Triton>=3.8 + the one baked topology); guard with
    :func:`fused_gdn_prefill_qkvz_supported` and fall back to the four-kernel chain
    when it returns ``False``.

    Args:
        projected_qkvz: bf16 ``[tokens, 3072]`` packed in_proj_qkvz output
            (``q|k|v|z`` interleaved per k-head).
        projected_ba: bf16 ``[tokens, 16]`` packed ``b|a`` delta-rule gates.
        conv_state: bf16 ``[N, 2048, 3]`` depthwise-conv history pool, updated
            in place.
        delta_state: fp32 ``[N, 8, 128, 128]`` recurrent state pool, updated in
            place.
        cache_indices: int32 ``[batch]`` slot index per sequence.
        cu_seqlens: int32 ``[batch + 1]`` ragged sequence offsets.
        has_initial_state: bool ``[batch]`` -- gates the conv history only (the
            recurrent state is always seeded from ``delta_state``; the caller
            zeroes fresh slots).
        conv_weight: bf16 ``[2048, 4]`` depthwise conv weights.
        conv_bias: bf16 ``[2048]`` conv bias (pass zeros if the model has none).
        a_log: fp32 ``[8]`` per-v-head decay log-rate.
        dt_bias: bf16 ``[8]`` per-v-head softplus bias.
        norm_weight: bf16 ``[128]`` gated-RMSNorm weight.
        scale: RMSNorm/quant scale (keyword-only).
        eps: RMSNorm epsilon (keyword-only, default ``1e-6``).

    Returns:
        ``(normalized_bf16, conv_state, delta_state, quantized_fp8, scales)``:
        ``normalized_bf16`` ``[tokens, 8, 128]`` is the pre-quant RMSNorm output;
        ``quantized_fp8`` (``torch.float8_e4m3fn`` ``[tokens, 1024]``) + ``scales``
        (``[tokens, 8]``) are the per-head group-128 FP8 activations a block-FP8
        ``out_proj`` consumes directly. ``conv_state`` and ``delta_state`` are the
        same (mutated) pool tensors passed in.

    Raises:
        ValueError: if the call is not supported (see
            :func:`fused_gdn_prefill_qkvz_supported` for the exact contract).
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
        a_log=a_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
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
