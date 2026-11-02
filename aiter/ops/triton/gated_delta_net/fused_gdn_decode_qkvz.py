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
``num_v_heads == ratio * num_k_heads`` (ratio 2 for Qwen3-Next, 3 for
Qwen3.5/3.8-27B) and a biased convolution. The head topology alone (KDA's
``lp = H*K`` offsets) cannot express ``VH == ratio*KH``.

Ratio 2 runs the generated, batch-specialised kernels; any other ratio runs the
hand-written ``_decode_group_ratio`` kernel, which walks a group's value heads
one state tile at a time.

Support is deliberately narrow and is reported by
:func:`fused_gdn_decode_qkvz_supported` rather than asserted, so callers fall
back instead of crashing. gfx950 only: the kernels use CDNA buffer addressing,
hand-written ``v_exp_f32``/``v_rcp_f32`` inline asm and explicit register
layouts.
"""

import functools
import re

import torch

# The Gluon tile kernels are imported lazily, at each launch site below, rather
# than here. This module has to stay importable on a build without
# ``triton.experimental.gluon`` so that :func:`fused_gdn_decode_qkvz_supported`
# can report the reason and the caller can take its fallback; a module-level
# import would raise before the probe is reachable, on exactly the platforms the
# probe exists for. Mirrors ``fused_gdn_prefill_qkvz.py``.

# SGLang's convention for a padded CUDA-graph row; kernels skip these rows.
PAD_SLOT_ID = -1

# gfx950 uses e4m3fn. e4m3fnuz is the gfx942 format and this op is gfx950-only,
# so advertising it would promise a path that can never run.
_FP8_QUANT_MAX = {torch.float8_e4m3fn: 448.0}
_HEAD_DIM = 128
# Value-heads-per-key-head ratios with a kernel: 2 = the generated pair kernels,
# anything else = the ratio-generic kernel. Both need the 128-wide heads.
_SUPPORTED_RATIOS = (1, 2, 3, 4)
# Batch at which the large-batch decomposition takes over. Measured crossover is
# between 64 and 128: below it the latency-hiding path wins, above it the
# register-resident tile does. See notes/gdn-perf-matrix.md.
_LARGE_BATCH = 128
_CONV_WIDTH = 4


def _uses_ratio_generic_kernel(ratio: int, flat: bool, has_qsl: bool) -> bool:
    """Ratio 2 in the interleaved layout without padding runs the generated pair
    kernels; every other call runs the ratio-generic kernel."""
    return ratio != 2 or flat or has_qsl


def _arch_supported() -> tuple[bool, str]:
    """gfx950 probe, guarded so architecture detection never raises at import.

    ``arch_info`` detection can touch the driver, so it is imported and called
    *here*, on the gate path, and any failure is reported as unsupported --
    keeping this module import-safe on CPU-only / non-ROCm workers where the
    caller simply takes the fallback chain.
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
    """Cached probe: can this Triton compile the kernels' Gluon dialect?

    gfx950 alone is not enough. These kernels use the Gluon dialect as it stands
    in Triton **3.8**; ROCm backported an *earlier, incompatible* Gluon into some
    3.7 builds where ``triton.experimental.gluon`` imports fine but the layout
    ops fail to compile. The decode kernels are built from exactly those
    primitives -- ``gl.BlockedLayout``, ``gl.convert_layout``, ``gl.amd.slice``
    and ``gl.amd.cdna3.buffer_load``/``store`` -- so the gate is an explicit
    ``>= 3.8`` version check, not just an import probe.
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
    qkvz_layout: str = "interleaved",
    query_start_loc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[bool, str]:
    """Report whether this call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``. The reason is meant to be
    logged once by the caller on its fallback path.
    """
    ok, reason = _arch_supported()
    if not ok:
        return False, reason

    ok, reason = _gluon_supported()
    if not ok:
        return False, reason

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
    device = projected_qkvz.device
    optional = tuple(t for t in (query_start_loc, out) if t is not None)
    if not all(
        isinstance(t, torch.Tensor) and t.is_cuda and t.device == device
        for t in tensors + optional
    ):
        return False, "every launch operand must be a CUDA tensor on one device"
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
    if qkvz_layout not in ("interleaved", "flat"):
        return (
            False,
            f"qkvz_layout must be 'interleaved' or 'flat', got {qkvz_layout!r}",
        )
    k_heads = (channels - v_heads * head_v_dim) // (2 * head_k_dim)
    if k_heads <= 0 or v_heads % k_heads:
        return (
            False,
            f"num_v_heads {v_heads} is not a multiple of num_k_heads {k_heads}",
        )
    if v_heads // k_heads not in _SUPPORTED_RATIOS:
        return (
            False,
            f"requires num_v_heads / num_k_heads in {_SUPPORTED_RATIOS}, got {v_heads}/{k_heads}",
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
            f"A_log must be fp32 [{v_heads}], got {A_log.dtype} {tuple(A_log.shape)}"
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
    # The recurrent state must be dense within a slot; the slot dimension of both
    # pools may be strided (a serving engine packs a block's conv and recurrent
    # states into one page) and the conv window may be stored channel-major or
    # width-major. Only the ratio-generic kernel takes strides.
    if ssm_state.stride()[1:] != (head_v_dim * head_k_dim, head_k_dim, 1):
        return False, (
            f"ssm_state must be dense within a slot, got strides {ssm_state.stride()}"
        )
    if 1 not in conv_state.stride()[1:]:
        return False, (
            f"conv_state must be dense along one inner dim, got strides "
            f"{conv_state.stride()}"
        )
    if out is not None:
        if out.dtype is not torch.bfloat16 or out.shape != (
            tokens,
            v_heads,
            head_v_dim,
        ):
            return False, (
                f"out must be bf16 [{tokens}, {v_heads}, {head_v_dim}], got {out.dtype} "
                f"{tuple(out.shape)}"
            )
        if not out.is_contiguous():
            return False, "out must be contiguous"
    if query_start_loc is not None:
        if query_start_loc.dtype is not torch.int32 or query_start_loc.shape != (
            tokens + 1,
        ):
            return False, (
                f"query_start_loc must be int32 [{tokens + 1}], got "
                f"{query_start_loc.dtype} {tuple(query_start_loc.shape)}"
            )
        if not query_start_loc.is_contiguous():
            return False, "query_start_loc must be contiguous"
    generic = _uses_ratio_generic_kernel(
        v_heads // k_heads, qkvz_layout == "flat", query_start_loc is not None
    )
    if not generic and not (conv_state.is_contiguous() and ssm_state.is_contiguous()):
        return False, (
            "strided state pools are only supported by the ratio-generic kernel "
            "(num_v_heads / num_k_heads != 2, the flat layout or query_start_loc); "
            f"got conv strides {conv_state.stride()}, ssm strides {ssm_state.stride()}"
        )

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
    qkvz_layout: str = "interleaved",
    query_start_loc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused GDN decode: split + conv1d + recurrence + gated RMSNorm [+ FP8].

    Args:
        projected_qkvz: [T, KH*(2*KD + 2*ratio*VD)] bf16, the packed
            ``in_proj_qkvz`` output (``ratio = VH // KH``, 1 to 4). Split
            internally; no separate split kernel.
        projected_ba: [T, 2*VH] bf16, packed ``in_proj_ba`` output (betas then a's
            per key-head group).
        conv_state: [N, channels, 3] bf16, rolling conv window. **Mutated in place.**
        ssm_state: [N, VH, VD, KD] fp32, delta-rule state. **Mutated in place.**
            On the ratio-generic path (``ratio != 2`` or the flat layout) both
            may be strided along N and ``conv_state`` may be any strided view
            (e.g. the transpose of a width-major [N, 3, channels] buffer);
            ``ssm_state`` stays dense within a slot.
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
        qkvz_layout: ``"interleaved"`` (Qwen3-Next: per key-head group
            ``[q | k | v*ratio | z*ratio]``, ``ba = [b*ratio | a*ratio]``) or
            ``"flat"`` (Qwen3.5/3.8: ``[q_all | k_all | v_all | z_all]``,
            ``ba = [b_all | a_all]``). Flat always runs the ratio-generic kernel.
        query_start_loc: optional [T + 1] int32 cumulative query lengths. Rows
            with an empty segment are padded CUDA-graph rows and are skipped
            even when their slot index is not ``pad_slot_id`` (serving engines
            pad the index buffer with the null block or leave stale entries).
            Ratio-generic kernel only.
        out: optional preallocated bf16 [T, VH, VD] buffer for the normalised
            output (e.g. a serving engine's persistent activation buffer); it is
            written in place and returned, saving a copy. Padded rows are left
            untouched.

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
        qkvz_layout,
        query_start_loc,
        out,
    )
    if not ok:
        raise ValueError(f"fused_gdn_decode_qkvz does not support this call: {reason}")
    flat = qkvz_layout == "flat"

    tokens = projected_qkvz.shape[0]
    _, v_heads, head_v_dim, head_k_dim = ssm_state.shape
    channels = conv_state.shape[1]
    k_heads = (channels - v_heads * head_v_dim) // (2 * head_k_dim)
    ratio = v_heads // k_heads
    device = projected_qkvz.device

    if out is None:
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
    if _uses_ratio_generic_kernel(ratio, flat, query_start_loc is not None):
        # Ratio-generic kernel at every batch size; the bands below hardcode
        # ratio 2, the interleaved layout and dense pools.
        from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_decode_qkvz_ratio import (
            _decode_group_ratio,
        )

        conv_width = 2 * head_k_dim + ratio * head_v_dim
        has_qsl = query_start_loc is not None
        _decode_group_ratio[(tokens * k_heads,)](
            *launch_args[:5],
            query_start_loc if has_qsl else ssm_state_indices,
            *launch_args[5:-2],
            *conv_state.stride(),
            ssm_state.stride(0),
            k_heads,
            v_heads,
            head_k_dim,
            head_v_dim,
            ratio,
            1 << (conv_width - 1).bit_length(),
            # NW, KL, KS: 16 lanes x 4 fp32 per tile row = one contiguous 16 B
            # access per lane (wider per-lane chunks cost 2-4x in bandwidth).
            8,
            16,
            4,
            PAD_SLOT_ID=pad_slot_id,
            HAS_FP8=has_fp8,
            FLAT=flat,
            HAS_QSL=has_qsl,
            num_warps=8,
        )
    elif tokens >= _LARGE_BATCH:
        # Large-batch band: a separate decomposition that keeps the whole [V, K]
        # state tile in registers across both matrix-vector products. Above this
        # threshold the kernel is bandwidth-bound, where that reaches a better
        # fraction of peak than the latency-hiding decomposition below; under it
        # the reverse holds. A range check, so there are no uncovered sizes.
        from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_decode_qkvz import (
            _decode_group,
        )

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
        from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_decode_qkvz import (
            _fused_decode_tiled,
        )

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
        from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_decode_qkvz import (
            _fused_decode,
        )

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
