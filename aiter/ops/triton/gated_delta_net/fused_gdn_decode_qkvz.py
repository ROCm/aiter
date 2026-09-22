# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Qwen3-Next Gated DeltaNet decode, with an optional FP8 quant epilogue.

One Gluon kernel launch replaces the four-kernel decode chain
``fused_qkvzba_split_reshape_cat -> causal_conv1d_update ->
fused_recurrent_gated_delta_rule_packed_decode -> layer_norm_gated`` and can
additionally emit the per-head group-128 FP8 activations that a block-FP8
``out_proj`` consumes directly, removing a separate quantization launch.

This is a *sibling* of :func:`fused_kda_decode`, not a variant of it. KDA uses a
per-channel lower-bounded sigmoid gate, a sigmoid output gate, uniform head
counts and no conv bias. Qwen3-Next GDN uses a per-head
``exp(-exp(A_log) * softplus(a + dt_bias))`` decay, a SiLU output gate,
``num_v_heads == 2 * num_k_heads`` and a biased convolution. The head topology
alone (KDA's ``lp = H*K`` offsets) cannot express ``VH == 2*KH``.

Support is deliberately narrow and is reported by
:func:`fused_gdn_decode_qkvz_supported` rather than asserted, so callers fall
back instead of crashing. gfx950 only: the kernels use CDNA buffer addressing,
hand-written ``v_exp_f32``/``v_rcp_f32`` inline asm and explicit register
layouts.
"""

import torch
from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_decode_qkvz import (
    _decode_group,
    _fused_decode,
    _fused_decode_tiled,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

# SGLang's convention for a padded CUDA-graph row; kernels skip these rows.
PAD_SLOT_ID = -1

# gfx950 uses e4m3fn. e4m3fnuz is the gfx942 format and this op is gfx950-only,
# so advertising it would promise a path that can never run.
_FP8_QUANT_MAX = {torch.float8_e4m3fn: 448.0}
_HEAD_DIM = 128
# Batch at which the large-batch decomposition takes over. Measured crossover is
# between 64 and 128: below it the latency-hiding path wins, above it the
# register-resident tile does. See notes/gdn-perf-matrix.md.
_LARGE_BATCH = 128
_CONV_WIDTH = 4


def fused_gdn_decode_qkvz_supported(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    quant_dtype: torch.dtype | None = None,
) -> tuple[bool, str]:
    """Report whether this call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``. The reason is meant to be
    logged once by the caller on its fallback path.
    """
    if get_arch() != "gfx950":
        return False, f"gfx950 only, got {get_arch()}"

    tensors = (
        projected_qkvz,
        projected_ba,
        conv_state,
        ssm_state,
        ssm_state_indices,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        norm_weight,
    )
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors):
        return False, "every launch operand must be a CUDA tensor"
    if ssm_state.ndim != 4 or conv_state.ndim != 3:
        return (
            False,
            f"expected ssm_state rank-4 and conv_state rank-3, got {ssm_state.ndim}/{conv_state.ndim}",
        )

    tokens = projected_qkvz.shape[0]
    _, v_heads, head_v_dim, head_k_dim = ssm_state.shape
    if head_v_dim != _HEAD_DIM or head_k_dim != _HEAD_DIM:
        return False, f"head dims must be {_HEAD_DIM}, got {head_k_dim}/{head_v_dim}"

    if conv_state.shape[0] != ssm_state.shape[0]:
        return False, (
            "conv and recurrent pools must have the same slot count, got "
            f"{conv_state.shape[0]} and {ssm_state.shape[0]}"
        )

    channels = conv_state.shape[1]
    if (channels - v_heads * head_v_dim) % (2 * head_k_dim):
        return False, f"conv channel count {channels} is not 2*KH*KD + VH*VD"
    k_heads = (channels - v_heads * head_v_dim) // (2 * head_k_dim)
    if v_heads != 2 * k_heads:
        return (
            False,
            f"requires num_v_heads == 2 * num_k_heads, got {v_heads}/{k_heads}",
        )

    group_width = 2 * head_k_dim + 2 * (v_heads // k_heads) * head_v_dim
    if projected_qkvz.shape != (tokens, k_heads * group_width):
        return False, f"packed qkvz layout mismatch: {tuple(projected_qkvz.shape)}"
    if projected_ba.shape != (tokens, 2 * v_heads):
        return False, f"packed ba layout mismatch: {tuple(projected_ba.shape)}"
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

    if ssm_state.dtype is not torch.float32:
        return False, f"ssm_state must be fp32, got {ssm_state.dtype}"
    bf16_args = (projected_qkvz, projected_ba, conv_state, conv_weight, conv_bias)
    if not all(t.dtype is torch.bfloat16 for t in bf16_args):
        return False, "packed projections, conv state/weight/bias must be bf16"
    if A_log.shape != (v_heads,) or A_log.dtype is not torch.float32:
        return False, (
            f"A_log must be fp32 [{v_heads}], got {A_log.dtype} "
            f"{tuple(A_log.shape)}"
        )
    if dt_bias.shape != (v_heads,) or dt_bias.dtype is not torch.bfloat16:
        return False, (
            f"dt_bias must be bf16 [{v_heads}], got {dt_bias.dtype} "
            f"{tuple(dt_bias.shape)}"
        )
    if norm_weight.shape != (head_v_dim,) or norm_weight.dtype is not torch.bfloat16:
        return False, (
            f"norm_weight must be bf16 [{head_v_dim}], got {norm_weight.dtype} "
            f"{tuple(norm_weight.shape)}"
        )
    if ssm_state_indices.dtype is not torch.int32:
        return False, f"state indices must be int32, got {ssm_state_indices.dtype}"
    if ssm_state_indices.shape != (tokens,):
        return (
            False,
            f"state indices must be [{tokens}], got {tuple(ssm_state_indices.shape)}",
        )
    # The kernels address every one of these with flat pointer arithmetic, so a
    # same-shaped strided view would be read from the wrong locations.
    flat = {
        "projected_qkvz": projected_qkvz,
        "projected_ba": projected_ba,
        "conv_state": conv_state,
        "ssm_state": ssm_state,
        "ssm_state_indices": ssm_state_indices,
        "conv_weight": conv_weight,
        "conv_bias": conv_bias,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "norm_weight": norm_weight,
    }
    non_contiguous = [name for name, t in flat.items() if not t.is_contiguous()]
    if non_contiguous:
        return False, f"operands must be contiguous: {', '.join(non_contiguous)}"

    if quant_dtype is not None and quant_dtype not in _FP8_QUANT_MAX:
        return False, f"unsupported quant dtype {quant_dtype}"
    return True, ""


def fused_gdn_decode_qkvz(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    norm_eps: float,
    quant_dtype: torch.dtype | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused GDN decode: split + conv1d + recurrence + gated RMSNorm [+ FP8].

    Args:
        projected_qkvz: [T, KH*(2*KD + 2*ratio*VD)] bf16, the packed
            ``in_proj_qkvz`` output. Split internally; no separate split kernel.
        projected_ba: [T, 2*VH] bf16, packed ``in_proj_ba`` output (betas then a's
            per key-head group).
        conv_state: [N, channels, 3] bf16, rolling conv window. **Mutated in place.**
        ssm_state: [N, VH, VD, KD] fp32, delta-rule state. **Mutated in place.**
        ssm_state_indices: [T] int32, row-to-slot map. Rows equal to
            ``pad_slot_id`` are skipped entirely. **Every live row must name a
            distinct, in-range slot**: each performs a read-modify-write on its
            own state in a separate program, so duplicate live indices race and
            are not supported.
        conv_weight: [channels, 4] bf16. conv_bias: [channels] bf16.
        A_log: [VH] fp32. dt_bias: [VH] bf16. norm_weight: [VD] bf16.
        scale: query scaling, typically ``head_k_dim ** -0.5``.
        norm_eps: gated RMSNorm epsilon.
        quant_dtype: if given, also emit group-128 FP8 activations and scales.
            ``None`` returns ``(out, None, None)`` — use this when the consumer
            is an unquantized Linear, which cannot accept a ``(fp8, scales)``
            tuple.
        pad_slot_id: sentinel marking padded CUDA-graph rows.

    Returns:
        ``(out, quantized, scales)``. ``out`` is [T, VH, VD] bf16; when
        ``quant_dtype`` is set, ``quantized`` is [T, VH*VD] and ``scales`` is
        [T, VH] fp32 with one scale per (token, value head). Values at padded
        rows are undefined.
    """
    ok, reason = fused_gdn_decode_qkvz_supported(
        projected_qkvz,
        projected_ba,
        conv_state,
        ssm_state,
        ssm_state_indices,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        norm_weight,
        quant_dtype,
    )
    if not ok:
        raise ValueError(f"fused_gdn_decode_qkvz does not support this call: {reason}")

    tokens = projected_qkvz.shape[0]
    _, v_heads, head_v_dim, head_k_dim = ssm_state.shape
    channels = conv_state.shape[1]
    k_heads = (channels - v_heads * head_v_dim) // (2 * head_k_dim)
    device = projected_qkvz.device

    out = torch.empty(
        (tokens, v_heads, head_v_dim), dtype=torch.bfloat16, device=device
    )
    has_fp8 = quant_dtype is not None
    if has_fp8:
        quantized = torch.empty(
            (tokens, v_heads * head_v_dim), dtype=quant_dtype, device=device
        )
        scales = torch.empty((tokens, v_heads), dtype=torch.float32, device=device)
        quant_max = _FP8_QUANT_MAX[quant_dtype]
    else:
        # The kernel still takes the pointers; it never dereferences them.
        quantized, scales, quant_max = out, out, 0.0

    # The full recurrent pool bounds every state element offset, and its element
    # count also bounds the smaller convolution pool for this ABI.
    index64 = tokens > 64 or ssm_state.numel() >= 2**31
    launch_args = (
        projected_qkvz,
        projected_ba,
        conv_state,
        ssm_state,
        ssm_state_indices,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        norm_weight,
        out,
        quantized,
        scales,
        scale,
        norm_eps,
        quant_max,
        k_heads,
        v_heads,
    )
    if tokens >= _LARGE_BATCH:
        # Large-batch band: a separate decomposition that keeps the whole [V, K]
        # state tile in registers across both matrix-vector products. Above this
        # threshold the kernel is bandwidth-bound, where that reaches a better
        # fraction of peak than the latency-hiding decomposition below; under it
        # the reverse holds. A range check, so there are no uncovered sizes.
        _decode_group[(tokens * k_heads,)](
            projected_qkvz,
            projected_ba,
            conv_state,
            ssm_state,
            ssm_state_indices,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            norm_weight,
            out,
            quantized,
            scales,
            scale,
            norm_eps,
            quant_max,
            k_heads,
            v_heads,
            head_k_dim,
            head_v_dim,
            4 if tokens <= 128 else 8,
            16,
            4,
            PAD_SLOT_ID=pad_slot_id,
            HAS_FP8=has_fp8,
            num_warps=4 if tokens <= 128 else 8,
        )
    elif tokens == 32 or tokens > 64:
        grid = (tokens * k_heads,) if tokens == 32 else (tokens, k_heads)
        _fused_decode_tiled[grid](
            *launch_args,
            INDEX64=index64,
            B32=tokens == 32,
            PAD_SLOT_ID=pad_slot_id,
            HAS_FP8=has_fp8,
            num_warps=8,
        )
    else:
        grid = (k_heads, tokens) if tokens == 64 else (tokens, k_heads)
        _fused_decode[grid](
            *launch_args,
            BATCH=tokens,
            INDEX64=index64,
            PAD_SLOT_ID=pad_slot_id,
            HAS_FP8=has_fp8,
            num_warps=8,
        )
    if not has_fp8:
        return out, None, None
    return out, quantized, scales
