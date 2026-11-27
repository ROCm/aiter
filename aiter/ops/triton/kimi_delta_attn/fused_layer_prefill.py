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
from typing import NamedTuple

import torch

# NB: the Gluon tile kernels are imported lazily inside the host launcher
# (appended at the end of this module), not at module load, so this module -- and
# the :func:`kimi_kda_fused_layer_prefill_supported` gate a caller probes -- stays
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
def _arch_supported() -> tuple[bool, str]:
    """Probe the GPU arch lazily and fail closed if it cannot be read.

    ``get_arch`` is imported here, not at module load, so this module stays
    importable on a CPU-only / non-ROCm host; a probe failure (e.g. no
    ``rocminfo``) returns ``(False, reason)`` rather than raising.
    """
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001 - any probe failure => unsupported
        return False, f"could not probe GPU arch ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


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
    """Return the prefill host launcher.

    The launcher lives at module scope (appended below) and imports its Gluon
    tile kernels lazily on first call (the ``triton.experimental.gluon`` + ROCm
    ``libdevice`` imports happen there, on the dispatch path, never at module
    load).
    """
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
    arch_ok, arch_reason = _arch_supported()
    if not arch_ok:
        return False, arch_reason

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
            (
                f"(tokens={m}, sequences={sequences}) outside covered schedule "
                f"(tokens {_MIN_TOKENS}..{_MAX_TOKENS}, sequences 1..{_MAX_SEQ})"
            ),
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
        or state.device != device
        or not state.is_contiguous()
    ):
        return (
            False,
            "temporal cache must be contiguous fp32 [slots,12,128,128] on device",
        )
    if (
        not isinstance(conv_state, torch.Tensor)
        or conv_state.dtype != torch.bfloat16
        or tuple(conv_state.shape) != (slots, 3, _QKV)
        or conv_state.device != device
        or not conv_state.is_contiguous()
    ):
        return (
            False,
            "convolution cache must be contiguous bf16 [slots,3,4608] on device",
        )

    if (
        state_indices.device != device
        or tuple(state_indices.shape) != (sequences,)
        or not state_indices.is_contiguous()
    ):
        return False, f"cache indices must be contiguous [{sequences}] on {device}"
    if (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != device
        or tuple(cu_seqlens.shape) != (sequences + 1,)
        or not cu_seqlens.is_contiguous()
    ):
        return False, f"cu_seqlens must be contiguous int32 [{sequences + 1}] on device"
    if (
        not isinstance(has_initial_state, torch.Tensor)
        or has_initial_state.dtype != torch.bool
        or has_initial_state.device != device
        or tuple(has_initial_state.shape) != (sequences,)
        or not has_initial_state.is_contiguous()
    ):
        return (
            False,
            f"has_initial_state must be contiguous bool [{sequences}] on device",
        )

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


# --- host launchers (moved from the Gluon tile so the tile module stays
# torch-free; each imports its tile kernels lazily on first call) ---
class _LaunchPolicy(NamedTuple):

    chunk: int
    prefix_rows: int
    project_rows: int
    output_rows: int
    direct_output: bool
    head_group: int
    blocked: bool
    zero_skip: bool
    sparse_maps: bool
    roundoff_skip: bool
    sparse_coeff: bool
    flag_chunks: int
    factor_unroll: int
    factor_prefetch: bool
    bulk_factors: bool
    bounded_recurrence: bool
    late_update_factor: bool
    mirror_prediction: bool
    early_decay: bool


def _launch_policy(m, sequences, heads):
    import triton

    blocked = m > 257
    if not blocked:
        chunk = 64
    elif m <= 2048:
        chunk = 96
    elif m <= 4096:
        chunk = 192
    elif m <= 6144:
        chunk = 320
    else:
        chunk = 320 if sequences >= 8 else 384
    prefix_rows = 8 if sequences == 1 else 16
    if blocked and 4 <= sequences < 8:
        prefix_rows = 32
    if m == 4096 and sequences == 4:
        prefix_rows = 16
    sparse_maps = blocked and sequences >= 8
    roundoff_skip = (m, sequences) in ((1024, 1), (4096, 1), (4096, 4), (6144, 2))
    zero_skip = (
        roundoff_skip
        or sparse_maps
        or (m > 6144 and sequences == 1)
        or (m == 4096 and sequences == 4)
    )
    short_factors = m <= 1024 and sequences == 1
    output_rows = 32 if chunk == 96 or m == 4096 or (m, sequences) == (8192, 1) else 64
    head_group = 4 if blocked and sequences > 1 and heads % 4 == 0 else 1
    if m == 4096 and sequences == 1 and heads % 12 == 0:
        head_group = 12
    if (m, sequences) == (8192, 1) and heads % 4 == 0:
        head_group = 4
    return _LaunchPolicy(
        chunk=chunk,
        prefix_rows=prefix_rows,
        project_rows=64 if m >= 2048 and m % 64 == 0 else 32,
        output_rows=output_rows,
        direct_output=output_rows == 32 or (m == 8192 and sequences == 1),
        head_group=head_group,
        blocked=blocked,
        zero_skip=zero_skip,
        sparse_maps=sparse_maps,
        roundoff_skip=roundoff_skip,
        sparse_coeff=m == 8192 and sequences == 1,
        flag_chunks=(
            triton.next_power_of_2(triton.cdiv(m, chunk))
            if zero_skip and not sparse_maps
            else 0
        ),
        factor_unroll=8 if short_factors else (16 if sequences >= 8 else 4),
        factor_prefetch=short_factors or (1024 < m <= 2048 and sequences == 2),
        bulk_factors=sequences >= 8,
        mirror_prediction=(m, sequences)
        in ((2048, 2), (4096, 1), (4096, 4), (6144, 2), (8192, 1), (8192, 8)),
        late_update_factor=(m, sequences) in ((2048, 2), (4096, 1)),
        early_decay=(m, sequences) == (1024, 1),
        bounded_recurrence=(m, sequences)
        in ((2048, 2), (4096, 1), (4096, 4), (6144, 2), (8192, 1), (8192, 8)),
    )


def fused_kda_prefill(
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
    *,
    lower_bound=-5.0,
    norm_eps=1e-5,
):

    import triton

    from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.fused_kda_layer_prefill import (
        _blocked_affine_chunks,
        _chunk_prefix,
        _fused_prepare,
        _normalize_output,
        _published_prefix,
        _response_factors,
        _serial_affine_chunks,
    )

    m, heads, sequences = qkv.shape[0], state.shape[1], state_indices.numel()
    policy = _launch_policy(m, sequences, heads)
    chunk = policy.chunk
    max_chunks = triton.cdiv(m, chunk)
    map_slots = max_chunks + sequences - 1
    decay_elements = m * heads * 128
    if policy.blocked:
        decay_elements = max(decay_elements, map_slots * heads * 32768)
    prepared = torch.empty(
        (m, 3 * heads * 128), dtype=torch.bfloat16, device=qkv.device
    )
    decay = torch.empty((decay_elements,), dtype=torch.float32, device=qkv.device)
    betas = torch.empty((m, heads), dtype=torch.float32, device=qkv.device)
    prepare_roles = 4 if policy.project_rows == 64 else 8
    _fused_prepare[(triton.cdiv(m, 32) * heads * prepare_roles,)](
        qkv,
        conv_weight,
        conv_state,
        state_indices,
        cu_seqlens,
        has_initial_state,
        beta,
        prepared,
        betas,
        forget_a,
        forget_weight,
        a_log,
        dt_bias,
        decay,
        lower_bound,
        heads,
        sequences,
        qkv.stride(0),
        beta.stride(0),
        state_indices.stride(0),
        *conv_state.stride(),
        m,
        forget_a.stride(0),
        forget_weight.stride(0),
        policy.project_rows,
        num_warps=4,
        enable_fp_fusion=False,
    )

    if policy.blocked:
        maps = decay[: map_slots * heads * 32768].view(map_slots, heads, 256, 128)
    else:
        maps = torch.empty(
            (map_slots, heads, 256, 128), dtype=torch.float32, device=qkv.device
        )
    if policy.roundoff_skip and chunk == 96:
        zero_flags = (
            betas.view(torch.int32)
            .view(-1)[: map_slots * heads * 8]
            .view(map_slots, heads, 8)
        )
    else:
        zero_flags = (
            torch.empty((map_slots, heads, 8), dtype=torch.int32, device=qkv.device)
            if policy.zero_skip
            else None
        )
    factor_slots = triton.cdiv(m, 16) + sequences - 1
    coeff_flags = (
        torch.empty((factor_slots, heads, 4), dtype=torch.int32, device=qkv.device)
        if policy.sparse_coeff
        else None
    )
    incoming_flags = (
        torch.empty(
            (map_slots, heads, 128 // policy.prefix_rows),
            dtype=torch.int32,
            device=qkv.device,
        )
        if policy.sparse_coeff
        else None
    )
    coeff = torch.empty((m, heads * 128), dtype=torch.bfloat16, device=qkv.device)
    if policy.blocked:
        local = prepared.view(torch.float32)[:, : heads * 128]
        out = coeff
        factor_elements = max(factor_slots * heads * 4224, map_slots * heads * 8192)
        factors = torch.empty(
            (factor_elements,), dtype=torch.float32, device=qkv.device
        )
        incoming = (
            torch.empty(
                (map_slots, heads, 16384), dtype=torch.bfloat16, device=qkv.device
            )
            if policy.roundoff_skip
            else factors.view(torch.bfloat16)
        )
        output_factors = torch.empty(
            (factor_slots, heads, 2304), dtype=torch.bfloat16, device=qkv.device
        )
        _response_factors[(heads, factor_slots)](
            prepared,
            decay,
            betas,
            factors,
            output_factors,
            state_indices,
            cu_seqlens,
            heads,
            state_indices.stride(0),
            sequences,
            policy.bulk_factors,
            policy.factor_unroll,
            policy.factor_prefetch,
            num_warps=1,
            enable_fp_fusion=True,
        )
        recurrence_grid = (sequences * heads, 8, max_chunks)
        _blocked_affine_chunks[recurrence_grid](
            prepared,
            factors,
            output_factors,
            maps,
            zero_flags,
            coeff_flags,
            coeff,
            local,
            state,
            has_initial_state,
            state_indices,
            cu_seqlens,
            qkv,
            conv_state,
            heads,
            state_indices.stride(0),
            state.stride(0),
            qkv.stride(0),
            *conv_state.stride(),
            chunk,
            local.stride(0),
            policy.zero_skip,
            policy.sparse_maps,
            policy.roundoff_skip,
            policy.bounded_recurrence,
            policy.sparse_coeff,
            policy.late_update_factor,
            policy.mirror_prediction,
            policy.early_decay,
            incoming,
            policy.roundoff_skip,
            MIRROR_UPDATE=(m, sequences) == (4096, 1),
            num_warps=1,
            enable_fp_fusion=True,
        )
    else:
        incoming = maps
        local = torch.empty((m, heads * 128), dtype=torch.float32, device=qkv.device)
        out = prepared.view(-1)[: m * heads * 128].view(m, heads * 128)
        _serial_affine_chunks[(16, sequences * heads, max_chunks)](
            prepared,
            decay,
            betas,
            maps,
            coeff,
            local,
            state,
            has_initial_state,
            state_indices,
            cu_seqlens,
            qkv,
            conv_state,
            heads,
            state_indices.stride(0),
            state.stride(0),
            qkv.stride(0),
            *conv_state.stride(),
            chunk,
            num_warps=1,
            enable_fp_fusion=True,
        )
    if policy.roundoff_skip:
        _published_prefix[(sequences * heads, 128 // policy.prefix_rows)](
            maps,
            incoming,
            zero_flags,
            state,
            state_indices,
            cu_seqlens,
            heads,
            state_indices.stride(0),
            state.stride(0),
            chunk,
            policy.prefix_rows,
            policy.flag_chunks,
            LOW_YIELD_DENSE=(m, sequences) in ((1024, 1), (6144, 2)),
            BITMAP_SCAN=(m, sequences) in ((4096, 1), (6144, 2)),
            num_warps=8,
        )
    else:
        _chunk_prefix[(sequences * heads, 128 // policy.prefix_rows)](
            maps,
            incoming,
            zero_flags,
            state,
            state_indices,
            cu_seqlens,
            heads,
            state_indices.stride(0),
            state.stride(0),
            chunk,
            policy.prefix_rows,
            policy.blocked,
            policy.zero_skip,
            policy.sparse_maps,
            policy.flag_chunks,
            policy.roundoff_skip,
            incoming_flags,
            policy.sparse_coeff,
            COPY_INITIAL=(m, sequences) == (4096, 1),
            PROPAGATE_BOUND=(m, sequences) in ((1024, 1), (4096, 1)),
            OMIT_FINAL_EXP=(m, sequences) == (4096, 4),
            num_warps=8,
            waves_per_eu=6 if (m, sequences) == (2048, 2) else 0,
        )
    output_rows = policy.output_rows
    _normalize_output[(triton.cdiv(chunk, output_rows), sequences * heads, max_chunks)](
        coeff,
        coeff_flags,
        local,
        maps,
        incoming,
        gate,
        norm_weight,
        out,
        state_indices,
        cu_seqlens,
        norm_eps,
        heads,
        state_indices.stride(0),
        gate.stride(0),
        chunk,
        output_rows,
        policy.blocked,
        local.stride(0),
        policy.direct_output,
        policy.head_group,
        policy.sparse_coeff,
        incoming_flags,
        128 // policy.prefix_rows,
        num_warps=4,
    )
    return out, conv_state, state
