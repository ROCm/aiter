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

Not adopted in-tree yet
-----------------------
aiter's own GDN prefill path does not call this op; it is offered for an external
caller (e.g. sglang) to opt into. The fallback is therefore the *caller's*
responsibility: probe :func:`fused_gdn_prefill_qkvz_supported` and, when it
returns ``(False, reason)``, run the existing four-kernel chain. The narrow
:class:`ValueError` in :func:`fused_gdn_prefill_qkvz` is only for a caller that
skipped that probe -- it flags misuse, it is not a shape-dispatch fallback.

Keeping the four tiles in sync
------------------------------
The four M-tile schedules are ported autotuner output (the Artemis MI355 v1
kernel-pack), specialized per token/batch band, not four hand-derived rewrites of
the math. They share one contract: for any covered shape the result must match
the single ``ref_gdn_prefill`` in ``test_fused_gdn_prefill_qkvz.py``, which every
tile is parametrized against. A GDN math or spec change lands in that reference;
CI then fails any tile that drifts, so the reference -- not four parallel edits --
is the source of truth the tiles are pinned to.
"""

import functools
import re
from dataclasses import dataclass

import torch
import triton

# NB: the Gluon tile kernels are imported lazily *inside each host launcher*
# (``gdn_prefill_group_fp8_quant_<key>`` below), not at module load. That keeps
# this module -- and in particular :func:`fused_gdn_prefill_qkvz_supported`, the
# gate a caller probes -- importable and callable on any platform (older Triton,
# non-gfx950, CUDA), so the caller can fall back to the four-kernel chain without
# the Gluon/HIP stack present. ``import triton`` itself is safe everywhere; only
# ``triton.experimental.gluon`` and the gfx950 kernel modules are gated.

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


##############################################################################
# Host launchers -- one per M-tile schedule, dispatched by _select_tile_key.
# Each imports its @gluon.jit kernels lazily inside the function body so this
# module stays importable on any platform (older Triton / non-gfx950 / CPU).
##############################################################################


@dataclass(frozen=True)
class FP8PrecisionConfig:
    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


# The m1024_3071 launcher names this dataclass ``_PrecisionConfig``; keep the
# alias so its body stays verbatim.
_PrecisionConfig = FP8PrecisionConfig

FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)

##############################################################################
# schedule m1024_3071
##############################################################################


def _run_chunked_delta(
    prepared,
    projected,
    ba,
    gates,
    states,
    indices,
    starts,
    a_log,
    dt_bias,
    norm_weight,
    normalized,
    quantized,
    scales,
    scale,
    eps,
    C=32,
    BV=16,
    S=0,
    WM=4,
):
    from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
        _chunk_offsets,
        _output_norm_quant,
        _prepare_chunk_factors,
        _prepare_segment_maps,
        _propagate_chunks,
        _propagate_segments,
        _render_segment_chunks,
    )

    m = prepared.shape[0]
    batch = indices.numel()
    chunks = triton.cdiv(m, C) + batch
    device = prepared.device
    offsets = torch.empty((batch + 1,), device=device, dtype=torch.int32)
    # Short, low-batch sequences benefit from parallel query/state products.
    # Larger workloads retain the smaller output-base buffer.
    history = S == 0 and batch <= 2 and C >= 32
    # Packing pays off only when the additional operands stay small.
    planes = (2 if history else 3) if S == 0 and m <= 4096 and batch <= 4 else 0
    w = torch.empty(
        (chunks, 8, max(1, planes), C, 128), device=device, dtype=torch.float32
    )
    u = torch.empty((chunks, 8, C, 128), device=device, dtype=torch.float32)
    updates = torch.empty_like(u)
    carry_buffer = torch.empty(
        (chunks, 8, 128 if history else C, 128), device=device, dtype=torch.float32
    )
    g = torch.empty((chunks, 8, C * C + C), device=device, dtype=torch.float32)
    _chunk_offsets[(batch + 1,)](
        starts, offsets, batch, triton.next_power_of_2(batch), C, num_warps=1
    )
    _prepare_chunk_factors[(chunks, 8)](
        prepared,
        ba,
        gates,
        a_log,
        dt_bias,
        starts,
        offsets,
        w,
        u,
        g,
        batch,
        C,
        planes,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if S:
        segments = triton.cdiv(m, C * S) + batch
        segment_offsets = torch.empty_like(offsets)
        transitions = torch.empty(
            (segments, 8, 128, 128), device=device, dtype=torch.float32
        )
        contributions = torch.empty_like(transitions)
        initial_states = torch.empty_like(transitions)
        _chunk_offsets[(batch + 1,)](
            starts,
            segment_offsets,
            batch,
            triton.next_power_of_2(batch),
            C * S,
            num_warps=1,
        )
        _prepare_segment_maps[(segments, 8, 4)](
            prepared,
            starts,
            offsets,
            segment_offsets,
            w,
            u,
            g,
            transitions,
            contributions,
            batch,
            C,
            S,
            32,
            2,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _propagate_segments[(batch, 8, 8)](
            states,
            indices,
            segment_offsets,
            transitions,
            contributions,
            initial_states,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _render_segment_chunks[(segments, 8, 4)](
            prepared,
            starts,
            offsets,
            segment_offsets,
            w,
            u,
            g,
            initial_states,
            updates,
            carry_buffer,
            batch,
            C,
            S,
            32,
            2,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _propagate_chunks[(batch, 8, 128 // BV)](
            prepared,
            states,
            indices,
            starts,
            offsets,
            w,
            u,
            g,
            updates,
            carry_buffer,
            C,
            BV,
            WM,
            history,
            planes,
            num_warps=4,
            enable_fp_fusion=False,
        )
    _output_norm_quant[(chunks, 8)](
        prepared,
        projected,
        norm_weight,
        starts,
        offsets,
        g,
        updates,
        carry_buffer,
        normalized,
        quantized,
        scales,
        scale,
        eps,
        batch,
        C,
        history,
        num_warps=4,
        enable_fp_fusion=False,
    )


def gdn_prefill_group_fp8_quant_m1024_3071(
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
    precision_config=FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
        _prepare_inputs_tiled,
        _update_conv_state,
    )

    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    prepared = torch.empty((m, 2048), device=device, dtype=torch.bfloat16)
    gates = torch.empty((m, 8), device=device, dtype=torch.float32)
    normalized = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    quantized = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    _prepare_inputs_tiled[(triton.cdiv(m, 16), 16)](
        projected_qkvz,
        projected_ba,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        prepared,
        gates,
        m,
        batch,
        16,
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _update_conv_state[(batch, 8)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        num_warps=4,
    )
    # Target about 16 or 32 independent temporal segments, rounded to an even
    # number of chunks. This balances parallel maps against the segment carry.
    target_segments = 16 if m <= 6144 else 32
    segment_chunks = (
        2 * triton.cdiv(m, 128 * target_segments) if batch <= 2 and m >= 4096 else 0
    )
    _run_chunked_delta(
        prepared,
        projected_qkvz,
        projected_ba,
        gates,
        delta_state,
        cache_indices,
        cu_seqlens,
        a_log,
        dt_bias,
        norm_weight,
        normalized,
        quantized,
        scales,
        scale,
        eps,
        C=32 if batch >= 16 or 5 <= batch <= 7 else 64,
        BV=64 if batch >= 16 else (8 if batch <= 2 else 16),
        WM=1 if batch >= 16 else 4,
        S=segment_chunks,
    )
    return normalized, conv_state, delta_state, quantized, scales


##############################################################################
# schedule m12289_16384_b1_5
##############################################################################


@dataclass(frozen=True)
class _Schedule:
    """Shape-only tuning; sequence boundaries and cache ownership stay on-device."""

    segment_tokens: int
    build_rows: int
    reverse_columns: int
    prefix_rows: int
    core_rows: int
    core_transposed: bool
    prep_warps: int
    prep_tokens: int
    norm_rows: int
    norm_warps: int
    fused_gates: bool
    fused_core: bool
    time_major: bool
    unit_major: bool
    preweight_key: bool
    prepare_bounds: bool


def _schedule(m: int, batch: int) -> _Schedule:
    wide = batch == 2 and m >= 16384
    if 2 < batch < 8 and m <= 4096:
        segment = 0
    elif batch <= 4 and m <= 4096:
        segment = 128
    elif m <= 2048 * batch:
        segment = 0
    elif batch == 1:
        if m >= 32768:
            segment = 1024
        elif m <= 12288:
            segment = ((m + 511) // 512) * 32
        else:
            segment = 512
    elif 3 <= batch <= 4 and m >= 24576:
        # Fewer prefix matrices, balanced by wider column-wise summaries.
        segment = 1024
    elif 3 <= batch <= 5 and 12288 <= m <= 24576:
        segment = 704
    elif wide or (batch > 4 and m > 4096 * batch):
        segment = 512
    else:
        segment = 256
    core_rows = (
        (64 if wide or m >= 32768 else 32) if segment else (32 if batch < 32 else 64)
    )
    fused_core = segment != 0 or batch >= 8
    build_rows = (
        128 if wide or (batch == 1 and m >= 32768) else (64 if m >= 16384 else 32)
    )
    reverse_columns = 0
    if segment and m // segment + batch >= 16:
        reverse_columns = (
            64 if build_rows == 128 or (3 <= batch <= 4 and m >= 24576) else 32
        )
    return _Schedule(
        segment_tokens=segment,
        build_rows=build_rows,
        reverse_columns=reverse_columns,
        prefix_rows=4 if batch == 1 else (8 if batch == 2 else 16),
        core_rows=core_rows,
        # Keep the established ownership of each recurrent path.
        core_transposed=(segment != 0 or batch >= 32)
        and not (batch == 1 and m <= 1024),
        prep_warps=4 if m <= 2048 else 1,
        prep_tokens=4 if m <= 12288 else 8,
        norm_rows=16 if m > 8192 else 32,
        norm_warps=1 if m > 8192 else 4,
        fused_gates=m <= 8192,
        fused_core=fused_core,
        time_major=batch >= 32,
        unit_major=segment != 0 and (m < 32768 or batch >= 3),
        preweight_key=(
            not fused_core or (batch == 1 and m <= 1024) or (batch == 8 and m <= 8192)
        ),
        prepare_bounds=batch >= 8 or (2 < batch < 8 and m <= 4096),
    )


def gdn_prefill_group_fp8_quant_m12289_16384_b1_5(
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
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
        _build_segments,
        _build_segments_reverse,
        _build_segments_stacked,
        _chunk_output_quant,
        _chunk_state_rows,
        _chunk_transform,
        _normalize_quantize,
        _prefix_segments,
        _prepare_gates,
        _prepare_qkv_window,
        _state_and_core,
    )

    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    schedule = _schedule(m, batch)
    bt = 32
    # A single sequence begins at zero, so no inter-sequence reserve is needed.
    # Multi-sequence bounds remain entirely on-device and use the safe capacity.
    chunks = (m + bt - 1) // bt if batch == 1 else m // bt + batch
    # Separate direct, preweighted head planes by eight unused rows. Consumers
    # address only initialized chunk rows; the gap changes physical head pitch.
    qkv_gap = (
        8
        if schedule.fused_core
        and not schedule.segment_tokens
        and schedule.preweight_key
        else 0
    )
    qkv_rows = chunks * bt + qkv_gap
    qkv = torch.empty((16, qkv_rows, 128), device=device, dtype=torch.bfloat16)
    bounds = (
        torch.empty((chunks, 2), device=device, dtype=torch.int32)
        if schedule.prepare_bounds
        else None
    )
    fused_gates = schedule.fused_gates
    gates = (
        projected_ba
        if fused_gates
        else torch.empty((8, 2, qkv_rows), device=device, dtype=torch.float32)
    )
    normalized = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    quantized = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    prep_warps = schedule.prep_warps
    prep_tokens = schedule.prep_tokens
    _prepare_qkv_window[(chunks * 32 // (prep_tokens * prep_warps), 16)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        qkv,
        qkv_rows,
        batch,
        prep_tokens,
        prep_warps,
        Bounds=bounds,
        num_warps=prep_warps,
        enable_fp_fusion=False,
    )
    if not fused_gates:
        _prepare_gates[(chunks,)](
            projected_ba,
            cu_seqlens,
            a_log,
            dt_bias,
            gates,
            qkv_rows,
            batch,
            bt,
            Bounds=bounds,
            num_warps=4,
            enable_fp_fusion=False,
        )
    seg = schedule.segment_tokens
    fused_core = schedule.fused_core
    time_major = schedule.time_major
    u = torch.empty((chunks, 8, 128, bt), device=device, dtype=torch.float32)
    w = torch.empty_like(u)
    scores = torch.empty((chunks, 8, bt, bt), device=device, dtype=torch.float32)
    coeff = torch.empty((chunks, 8, 2, bt), device=device, dtype=torch.float32)
    tail_key = (
        torch.empty((chunks, 8, bt, 128), device=device, dtype=torch.float32)
        if schedule.preweight_key
        else None
    )
    if fused_core:
        core = normalized
    else:
        chunk_state = torch.empty(
            (chunks, 8, 128, 128), device=device, dtype=torch.float32
        )
    # Adjacent heads share the tiny-input factorization traversal.
    head_major = m <= 1024 and batch == 1
    factor_grid = (8, chunks) if head_major else (chunks, 8)
    _chunk_transform[factor_grid](
        qkv,
        gates,
        u,
        w,
        scores,
        coeff,
        qkv_rows,
        bt,
        num_warps=4,
        enable_fp_fusion=False,
        TIME_MAJOR=time_major,
        BA=projected_ba,
        Starts=cu_seqlens,
        ALog=a_log,
        DTBias=dt_bias,
        BATCH=batch,
        FUSED_GATES=fused_gates,
        TailKey=tail_key,
        Bounds=bounds,
        HEAD_MAJOR=head_major,
    )
    if seg:
        segments = (m + seg - 1) // seg if batch == 1 else m // seg + batch
        affine = torch.empty(
            (segments, 8, 2, 128, 128), device=device, dtype=torch.float32
        )
        segment_state = affine
        build_bv = schedule.build_rows
        build_wm = 4 if build_bv == 128 else 2
        if schedule.reverse_columns:
            columns = schedule.reverse_columns
            build_kernel = (
                _build_segments_stacked if columns == 64 else _build_segments_reverse
            )
            build_kernel[(8, 128 // columns, segments)](
                qkv,
                u,
                w,
                coeff,
                cu_seqlens,
                affine,
                qkv_rows,
                batch,
                bt,
                seg,
                columns,
                4,
                MMA_SIZE=32 if columns == 32 else 16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _build_segments[(8, 2 * 128 // build_bv, segments)](
                qkv,
                u,
                w,
                coeff,
                cu_seqlens,
                affine,
                qkv_rows,
                batch,
                bt,
                seg,
                build_bv,
                4,
                build_wm,
                TailKey=tail_key,
                num_warps=4,
                enable_fp_fusion=False,
            )
        prefix_bv = schedule.prefix_rows
        _prefix_segments[(8, 128 // prefix_bv, batch)](
            affine,
            delta_state,
            cache_indices,
            cu_seqlens,
            seg,
            prefix_bv,
            num_warps=2 if prefix_bv == 4 else 4,
            enable_fp_fusion=False,
        )
    if fused_core:
        units = segments if seg else batch
        bv = schedule.core_rows
        core_grid = (
            (8, 128 // bv, units) if schedule.unit_major else (units, 8, 128 // bv)
        )
        _state_and_core[core_grid](
            qkv,
            u,
            w,
            scores,
            coeff,
            delta_state,
            cache_indices,
            cu_seqlens,
            segment_state if seg else delta_state,
            core,
            scale,
            qkv_rows,
            batch,
            bt,
            seg,
            bv,
            4,
            2,
            num_warps=4,
            enable_fp_fusion=False,
            TIME_MAJOR=time_major,
            UNIT_MAJOR=schedule.unit_major,
            TailKey=tail_key,
            TRANSPOSED=schedule.core_transposed,
        )
    else:
        bv, wm = 16, 1
        _chunk_state_rows[(batch, 8, 128 // bv)](
            qkv,
            u,
            w,
            coeff,
            delta_state,
            cache_indices,
            cu_seqlens,
            chunk_state,
            qkv_rows,
            bt,
            bv,
            4,
            wm,
            num_warps=4,
            enable_fp_fusion=False,
            TailKey=tail_key,
        )
    maximum = precision_config.group_quant_max
    inverse_maximum = 1.0 / maximum
    if fused_core:
        norm_rows = schedule.norm_rows
        _normalize_quantize[((m * 8 + norm_rows - 1) // norm_rows,)](
            core,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            eps,
            maximum,
            inverse_maximum,
            m,
            norm_rows,
            schedule.norm_warps,
            num_warps=schedule.norm_warps,
            enable_fp_fusion=False,
        )
    else:
        _chunk_output_quant[(chunks, 8)](
            qkv,
            u,
            scores,
            coeff,
            cu_seqlens,
            chunk_state,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            scale,
            eps,
            maximum,
            inverse_maximum,
            qkv_rows,
            batch,
            bt,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normalized, conv_state, delta_state, quantized, scales


##############################################################################
# schedule m12289_16384_b6_15
##############################################################################


def gdn_prefill_group_fp8_quant_m12289_16384_b6_15(
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
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 norm, updated state pools, FP8 values, and FP32 scales."""
    from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
        _build_group_maps,
        _evaluate_groups,
        _gated_rms_quant,
        _prepare_chunks_compact,
        _prepare_chunks_full,
        _prepare_tokens,
        _propagate_group_maps,
        _recurrence_compact,
        _recurrence_full,
        _update_conv_state_b6_15,
    )

    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    slots, heads, dim, key_dim = delta_state.shape
    assert dim == key_dim == 128 and heads == 8
    assert projected_qkvz.shape == (m, 3072) and projected_ba.shape == (m, 16)
    assert conv_state.shape == (slots, 2048, 3)
    assert conv_weight.shape == (2048, 4) and conv_bias.shape == (2048,)
    assert cu_seqlens.shape == (batch + 1,) and has_initial_state.shape == (batch,)
    assert cache_indices.dtype is cu_seqlens.dtype is torch.int32
    assert has_initial_state.dtype is torch.bool and delta_state.dtype is torch.float32
    assert a_log.shape == dt_bias.shape == (8,) and a_log.dtype is torch.float32
    assert norm_weight.shape == (128,) and m > 0
    bf16 = (
        projected_qkvz,
        projected_ba,
        conv_state,
        conv_weight,
        conv_bias,
        dt_bias,
        norm_weight,
    )
    assert all(t.dtype is torch.bfloat16 and t.is_contiguous() for t in bf16)

    device = projected_qkvz.device
    prepared = torch.empty((16, m, 128), device=device, dtype=torch.bfloat16)
    gates = torch.empty((8, m, 2), device=device, dtype=torch.float32)
    prep_rows, prep_warps = (16, 4) if m <= 4096 else (4, 1)
    _prepare_tokens[((m + prep_rows - 1) // prep_rows, 16)](
        projected_qkvz,
        projected_ba,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        prepared,
        gates,
        M=m,
        LOG_BATCH=(batch - 1).bit_length(),
        BATCH=batch,
        ROWS=prep_rows,
        LANES=16,
        PACK=4,
        num_warps=prep_warps,
        enable_fp_fusion=False,
    )
    # All readers of the old convolution history complete before this launch.
    _update_conv_state_b6_15[(batch, 16)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        num_warps=1,
    )
    core = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    use_groups = batch == 1 or (batch <= 4 and m // batch >= 4096)
    if use_groups:
        token_block, prep_warps = 32, 1
        value_block, recur_warps = (16, 2) if m <= 2048 else (32, 4)
    elif batch <= 4:
        token_block, prep_warps, value_block, recur_warps = 32, 1, 16, 4
    elif batch <= 8:
        token_block, prep_warps, value_block, recur_warps = 32, 1, 16, 2
    elif batch <= 16:
        token_block, prep_warps, value_block, recur_warps = 32, 1, 32, 2
    else:
        token_block, prep_warps, value_block, recur_warps = 16, 1, 32, 2
    num_chunks = m // token_block + batch
    chunk_c = torch.empty(
        (num_chunks, 8, token_block, token_block),
        device=device,
        dtype=torch.float32,
    )
    if batch <= 4:
        chunk_w = torch.empty(
            (num_chunks, 8, token_block, 128), device=device, dtype=torch.float32
        )
        chunk_u = torch.empty_like(chunk_w)
        chunk_q = torch.empty_like(chunk_w)
        chunk_k = torch.empty_like(chunk_w)
        chunk_g = torch.empty((num_chunks, 8), device=device, dtype=torch.float32)
        _prepare_chunks_full[(num_chunks, 8)](
            prepared,
            gates,
            cu_seqlens,
            chunk_w,
            chunk_u,
            chunk_q,
            chunk_k,
            chunk_c,
            chunk_g,
            M=m,
            BATCH=batch,
            BT=token_block,
            num_warps=prep_warps,
        )
        if use_groups:
            group_tokens = (
                128 if m <= 2048 else 512 if m <= 8192 else 1024 if m <= 16384 else 2048
            )
            group_row_warps = 1 if value_block == 16 else 2
            num_groups = m // group_tokens + batch
            maps = torch.empty(
                (num_groups, 8, 256, 128), device=device, dtype=torch.float32
            )
            boundaries = torch.empty(
                (num_groups, 8, 128, 128), device=device, dtype=torch.float32
            )
            _build_group_maps[(256 // value_block, 8, num_groups)](
                chunk_w,
                chunk_u,
                chunk_k,
                chunk_g,
                cu_seqlens,
                maps,
                BATCH=batch,
                BT=token_block,
                BV=value_block,
                GROUP=group_tokens,
                ROW_WARPS=group_row_warps,
                num_warps=recur_warps,
            )
            _propagate_group_maps[(batch, 8, 8)](
                maps,
                boundaries,
                delta_state,
                cache_indices,
                cu_seqlens,
                GROUP=group_tokens,
                BV=16,
                num_warps=4,
            )
            _evaluate_groups[(128 // value_block, 8, num_groups)](
                chunk_w,
                chunk_u,
                chunk_q,
                chunk_k,
                chunk_c,
                chunk_g,
                boundaries,
                cu_seqlens,
                core,
                scale,
                BATCH=batch,
                BT=token_block,
                BV=value_block,
                GROUP=group_tokens,
                ROW_WARPS=group_row_warps,
                num_warps=recur_warps,
            )
        else:
            _recurrence_full[(batch, 8, 128 // value_block)](
                chunk_w,
                chunk_u,
                chunk_q,
                chunk_k,
                chunk_c,
                chunk_g,
                delta_state,
                cache_indices,
                cu_seqlens,
                core,
                scale,
                BT=token_block,
                BV=value_block,
                num_warps=recur_warps,
            )
    else:
        chunk_inverse = torch.empty_like(chunk_c)
        chunk_decay = torch.empty(
            (num_chunks, 8, 2, token_block),
            device=device,
            dtype=torch.float32,
        )
        _prepare_chunks_compact[(num_chunks, 8)](
            prepared,
            gates,
            cu_seqlens,
            chunk_inverse,
            chunk_c,
            chunk_decay,
            M=m,
            BATCH=batch,
            BT=token_block,
            num_warps=prep_warps,
        )
        _recurrence_compact[(batch, 8, 128 // value_block)](
            prepared,
            gates,
            chunk_inverse,
            chunk_c,
            chunk_decay,
            delta_state,
            cache_indices,
            cu_seqlens,
            core,
            scale,
            M=m,
            BT=token_block,
            BV=value_block,
            num_warps=recur_warps,
            ROW_WARPS=2 if batch > 8 else 1,
            TRANSPOSED=batch <= 8,
        )
    normalized = torch.empty_like(core)
    values = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    epi_rows, epi_lanes, epi_pack = (4, 16, 4) if m < 4096 else (8, 8, 8)
    _gated_rms_quant[((m * 8 + epi_rows - 1) // epi_rows,)](
        core,
        projected_qkvz,
        norm_weight,
        normalized,
        values,
        scales,
        eps,
        precision_config.group_quant_max,
        1.0 / precision_config.group_quant_max,
        TOTAL=m * 8,
        ROWS=epi_rows,
        LANES=epi_lanes,
        PACK=epi_pack,
        num_warps=1,
    )
    return normalized, conv_state, delta_state, values, scales


##############################################################################
# schedule m3072_16384
##############################################################################


def gdn_prefill_group_fp8_quant_m3072_16384(
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
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 output, the mutated pools, FP8 output, and group scales."""
    from aiter.ops.triton._gluon_kernels.gfx950.gated_delta_net.fused_gdn_prefill_qkvz import (
        _block_transforms,
        _chunk_delta_recurrence,
        _convolve_sliding,
        _emit_chunk_outputs,
        _finish_blocks,
        _norm_and_quantize,
        _prepare_chunk_matrices,
        _propagate_block_states,
        _scan_chunk_states,
    )

    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    prepared = torch.empty((m, 2048), dtype=torch.bfloat16, device=device)
    core = (
        prepared.view(-1)[: m * 1024].view(m, 8, 128)
        if m >= 8192
        else torch.empty((m, 8, 128), dtype=torch.bfloat16, device=device)
    )
    normalized = core
    quantized = torch.empty((m, 1024), dtype=precision_config.dtype, device=device)
    scales = torch.empty((m, 8), dtype=torch.float32, device=device)
    bt = 32
    snapshot_outputs = 3 <= batch <= 4 and m <= 4096
    native_u = snapshot_outputs or (12288 < m <= 16384 and batch == 2)
    hierarchical = batch <= 5 and (batch < 4 or m > 4096) and not snapshot_outputs
    compact_qk = batch >= 16
    packed_rhs = hierarchical or snapshot_outputs or compact_qk
    chunks = triton.cdiv(m, bt) + batch
    chunk_offsets = torch.empty((batch + 1,), dtype=torch.int32, device=device)
    group = (
        4
        if m <= 2048
        else (
            12
            if 4096 < m <= 6144
            else 8 if m <= 8192 else 24 if m <= 12288 else 16 if m <= 16384 else 32
        )
    )
    wide_summary = 12288 < m <= 16384 and 3 <= batch <= 5
    if wide_summary:
        group = 20
    if hierarchical:
        block_offsets = torch.empty((batch + 1,), dtype=torch.int32, device=device)
    else:
        block_offsets = chunk_offsets
    matrices = torch.empty(
        (chunks, 8, 2 if compact_qk else 4, bt, 128), dtype=torch.float32, device=device
    )
    output_weights = torch.empty(
        (chunks, 8, bt, bt), dtype=torch.float32, device=device
    )
    chunk_decay = torch.empty((chunks, 8), dtype=torch.float32, device=device)
    chunk_begin = torch.empty((chunks,), dtype=torch.int32, device=device)
    raw_qk = (
        torch.empty((chunks, 4, 2, bt, 128), dtype=torch.bfloat16, device=device)
        if compact_qk
        else matrices
    )
    qk_factors = (
        torch.empty((chunks, 8, 2, bt), dtype=torch.float32, device=device)
        if compact_qk
        else chunk_decay
    )

    convolution_tile = 8 if m >= 16384 else 4
    _convolve_sliding[(triton.cdiv(m, convolution_tile), 4, 2)](
        projected_qkvz,
        conv_state,
        cu_seqlens,
        cache_indices,
        has_initial_state,
        conv_weight,
        conv_bias,
        prepared,
        chunk_offsets,
        block_offsets,
        m,
        batch,
        batch.bit_length(),
        triton.next_power_of_2(batch),
        bt,
        group,
        hierarchical,
        convolution_tile,
        num_warps=2,
        enable_fp_fusion=False,
    )
    _prepare_chunk_matrices[(8, chunks)](
        prepared,
        projected_ba,
        a_log,
        dt_bias,
        cu_seqlens,
        chunk_offsets,
        matrices,
        output_weights,
        chunk_decay,
        chunk_begin,
        batch,
        batch.bit_length(),
        bt,
        packed_rhs,
        num_warps=4,
        enable_fp_fusion=False,
        COMPACT_QK=compact_qk,
        raw_qk=raw_qk,
        qk_factors=qk_factors,
        NATIVE_U=native_u,
    )
    if snapshot_outputs:
        snapshots = torch.empty(
            (chunks, 8, 128, 128), dtype=torch.float32, device=device
        )
        _scan_chunk_states[(batch, 8, 8)](
            matrices,
            chunk_decay,
            chunk_offsets,
            delta_state,
            cache_indices,
            snapshots,
            bt,
            16,
            2,
            num_warps=2,
            enable_fp_fusion=False,
        )
        _emit_chunk_outputs[(chunks, 8)](
            matrices,
            output_weights,
            snapshots,
            chunk_begin,
            chunk_offsets,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            conv_state,
            cu_seqlens,
            cache_indices,
            has_initial_state,
            scale,
            eps,
            precision_config.group_quant_max,
            1.0 / precision_config.group_quant_max,
            batch,
            batch.bit_length(),
            bt,
            num_warps=4,
        )
    elif hierarchical:
        direct_first = m < 32768
        value_tile, warps, value_warps = (
            (16, 2, 1)
            if m < 4096
            else (
                (128, 4, 2)
                if m >= 32768
                else (64, 4, 2) if wide_summary else (32, 4, 2)
            )
        )
        finish_warps, finish_value_warps = warps, value_warps
        if 12288 < m <= 16384 and batch == 2:
            value_tile, warps, value_warps = 32, 2, 1
        blocks = triton.cdiv(m, bt * group) + batch
        transforms = torch.empty(
            (blocks, 8, 2, 128, 128), dtype=torch.float32, device=device
        )
        block_info = torch.empty((blocks, 3), dtype=torch.int32, device=device)
        _block_transforms[(blocks, 8, 2 * 128 // value_tile)](
            matrices,
            chunk_decay,
            chunk_offsets,
            block_offsets,
            transforms,
            block_info,
            delta_state,
            cache_indices,
            batch,
            batch.bit_length(),
            bt,
            group,
            value_tile,
            warps,
            value_warps,
            direct_first,
            num_warps=warps,
            enable_fp_fusion=False,
            NATIVE_U=native_u,
        )
        _propagate_block_states[(batch, 8, 8)](
            transforms,
            block_offsets,
            delta_state,
            cache_indices,
            16,
            4,
            direct_first,
            num_warps=4,
            enable_fp_fusion=False,
        )
        finish_tile = 16 if 8192 < m <= 12288 else 64 if m >= 8192 else value_tile
        if 8192 < m <= 12288:
            # More independent replay CTAs benefit this intermediate-length regime.
            finish_warps, finish_value_warps = 2, 1
        finish_unroll = 1 if (m <= 2048 and batch > 1) or 6144 < m <= 8192 else 4
        _finish_blocks[(blocks, 8, 128 // finish_tile)](
            matrices,
            output_weights,
            chunk_decay,
            chunk_begin,
            chunk_offsets,
            block_offsets,
            block_info,
            transforms,
            delta_state,
            cache_indices,
            cu_seqlens,
            core,
            scale,
            batch,
            bt,
            finish_tile,
            finish_warps,
            finish_value_warps,
            num_warps=finish_warps,
            enable_fp_fusion=False,
            UNROLL=finish_unroll,
            NATIVE_U=native_u,
        )
    else:
        value_tile, warps, value_warps = (64, 4, 2) if batch >= 32 else (32, 4, 2)
        transposed = not (8 <= batch < 16)
        preload = 1 if 8 <= batch < 16 else 3 if batch >= 16 else 0
        _chunk_delta_recurrence[(batch, 8, 128 // value_tile)](
            matrices,
            output_weights,
            chunk_decay,
            chunk_begin,
            chunk_offsets,
            delta_state,
            cache_indices,
            cu_seqlens,
            core,
            scale,
            bt,
            value_tile,
            warps,
            value_warps,
            packed_rhs,
            transposed,
            preload,
            num_warps=warps,
            enable_fp_fusion=False,
            COMPACT_QK=compact_qk,
            raw_qk=raw_qk,
            qk_factors=qk_factors,
        )
    if not snapshot_outputs:
        _norm_and_quantize[(max(triton.cdiv(m * 8, 16), batch * 8),)](
            core,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            conv_state,
            cu_seqlens,
            cache_indices,
            has_initial_state,
            eps,
            precision_config.group_quant_max,
            1.0 / precision_config.group_quant_max,
            m,
            16,
            batch,
            num_warps=4,
        )
    return normalized, conv_state, delta_state, quantized, scales


def _load_tile(key: str):
    """Return the host launcher for tile ``key`` -- a function defined in this
    module.

    All four schedules' host launchers live here as
    ``gdn_prefill_group_fp8_quant_<key>`` and each imports its Gluon kernels
    lazily inside its own body (mirroring ``gemm/basic/gemm_a16w16.py``, which
    lazy-imports its Gluon kernels inside the supported-arch branch). So
    importing this module -- and in particular probing
    :func:`fused_gdn_prefill_qkvz_supported` -- never pulls in the
    ``triton.experimental.gluon`` + ROCm ``libdevice`` stack; that only fires
    once a tile is dispatched on a supported device. The kernel modules under
    ``_gluon_kernels`` stay torch-free.
    """
    return globals()[f"gdn_prefill_group_fp8_quant_{key}"]


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
