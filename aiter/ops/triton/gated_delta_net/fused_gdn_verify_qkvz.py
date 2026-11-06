# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Gated DeltaNet speculative-verify step on ReplaySSM state, for gfx1250.

One launch replaces the verify chain ``causal_conv1d_update -> gating ->
ReplaySSM -> gated RMSNorm``. Two kernels: ``_fused_gdn_verify_chunk`` runs the
window in chunk form (the state's products on WMMA, never the state itself),
and ``_fused_gdn_verify`` runs it token by token on the rebuilt state, for the
shapes that the first does not take."""

import os

import torch
import triton

from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_CONFIG_NAME = "FUSED_GDN_VERIFY_QKVZ"
# The slot value that marks a padded sequence, as the other GDN ops define it.
PAD_SLOT_ID = -1


def _get_config(num_seqs: int) -> dict:
    config_dir = resolve_config_dir("fusions", _CONFIG_NAME, backend="gluon")
    configs = load_config_json(f"{config_dir}/DEFAULT.json")
    return select_leq_config(configs, num_seqs, prefix="S_LEQ_")


# LDS a chunk-form program may plan for (of 320 KB): the checkpoints, records,
# window rows and readouts of the group, before the compiler's scratch.
_CHUNK_LDS_BUDGET = 240 * 1024


def _chunk_kernel_fits(head_k_dim, head_v_dim, ratio, max_query_len, cap, num_warps):
    """Whether ``_fused_gdn_verify_chunk`` takes the shape: the window's q|k rows
    fill one 16-row WMMA tile, the record keys at least one 16-column tile, the
    waves split V, and the group's checkpoints and records fit in LDS."""
    t_p2 = triton.next_power_of_2(max_query_len)
    cap_p2 = triton.next_power_of_2(cap)
    k, v = head_k_dim, head_v_dim
    lds = (
        ratio * k * (v + 8) * 2  # checkpoints, bf16
        + 2 * ratio * cap_p2 * (k + 8) * 2  # k and u records, bf16
        + 16 * (k + 4) * 4  # the q|k rows, fp32
        + 16 * ratio * v * 4  # q S0 and k S0, fp32
        + 4 * t_p2 * ratio * v * 2  # v rows, residuals, readouts, bf16
    )
    return (
        k == v
        and k % 32 == 0
        and v % (16 * num_warps) == 0
        and ratio & (ratio - 1) == 0
        and 2 * t_p2 <= 16
        and cap_p2 >= 16
        and num_warps in (4, 8)
        and lds <= _CHUNK_LDS_BUDGET
    )


# The split form's tickets, one int32 counter per (conv slot, key head) for each
# split, never reset: each call adds SPLIT to the counters of its live sequences,
# so ticket % SPLIT == SPLIT - 1 marks the last program of the group to arrive.
_TICKETS: dict = {}
_TICKET_COUNT = 1 << 16


def _split_tickets(count, split, device):
    """The ticket counters, or None when they cannot be made (in a graph capture)."""
    if count > _TICKET_COUNT:
        return None
    tickets = _TICKETS.get((device, split))
    if tickets is None:
        if torch.cuda.is_current_stream_capturing():
            # Not in a graph's pool: the counters must outlive every graph.
            _LOGGER.info(
                "FUSED_GDN_VERIFY_QKVZ: split form skipped, first call in a capture"
            )
            return None
        tickets = torch.zeros(_TICKET_COUNT, dtype=torch.int32, device=device)
        _TICKETS[(device, split)] = tickets
    return tickets


def _chunk_warps(seqs: int) -> int:
    """The chunk kernel's waves (AITER_GDN_VERIFY_CHUNK_WARPS overrides), 0 for none."""
    return int(os.environ.get("AITER_GDN_VERIFY_CHUNK_WARPS", "0")) or _get_config(
        seqs
    ).get("chunk_num_warps", 0)


def fused_gdn_verify_mx32_supported(
    checkpoint: torch.Tensor,
    conv_state: torch.Tensor,
    record_k: torch.Tensor,
    max_query_len: int,
    seqs: int,
) -> bool:
    """Whether ``fused_gdn_verify_qkvz(..., out_scale=...)`` (the 1x32 MXFP8 output) takes
    this shape: it runs on ``_fused_gdn_verify_chunk`` only (not the token loop, nor the
    experimental compact / matrix forms). Host-side only; nothing is launched."""
    _, v_heads, head_k_dim, head_v_dim = checkpoint.shape
    if head_v_dim % 32 or (v_heads * head_v_dim) % 128:
        return False
    if os.environ.get("AITER_GDN_VERIFY_COMPACT", "0") == "1":
        return False
    if os.environ.get("AITER_GDN_VERIFY_MATRIX", "0") == "1":
        return False
    if seqs == 0:
        return True
    k_heads = (conv_state.shape[1] - v_heads * head_v_dim) // (2 * head_k_dim)
    warps = _chunk_warps(seqs)
    return bool(warps) and _chunk_kernel_fits(
        head_k_dim,
        head_v_dim,
        v_heads // k_heads,
        max_query_len,
        record_k.shape[2],
        warps,
    )


def fused_gdn_verify_qkvz(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    checkpoint: torch.Tensor,
    record_k: torch.Tensor,
    record_u: torch.Tensor,
    record_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    max_query_len: int,
    scale: float,
    norm_eps: float,
    out: torch.Tensor,
    pad_slot_id: int = PAD_SLOT_ID,
    out_scale: torch.Tensor | None = None,
    out_scale_rows: int | None = None,
) -> torch.Tensor:
    """One speculative-verify step of a Gated DeltaNet layer on ReplaySSM state.

    Each sequence verifies a window of up to ``max_query_len`` tokens. The op runs
    the causal conv1d over the window, the gating, the state rebuild from the
    checkpoint and the committed records, the delta-rule update of each token and
    the gated RMSNorm. It appends one record for each token. When the records cannot
    hold two more windows, it writes the rebuilt state as the new checkpoint and
    writes the records from position 0. The caller moves ``write_pos`` (the commit)
    once for each forward, before the first layer.

    Args:
        projected_qkvz: ``[tokens, q|k|v|z]`` bf16 rows (``q``, ``k``: ``KH * 128``;
            ``v``, ``z``: ``VH * 128``). The row stride can be larger than the row.
        projected_ba: ``[tokens, b|a]`` bf16 rows, ``VH`` each. The same rule applies.
        conv_state: ``[slots, channels, max_query_len + 2]`` bf16 conv windows, with
            any strides.
        conv_state_indices: ``[>= seqs]`` int32 conv slot of each sequence.
        num_accepted_tokens: ``[>= seqs]`` int32 tokens that the previous step
            accepted.
        query_start_loc: ``[seqs + 1]`` int32 first token row of each window.
        checkpoint: ``[slots, VH, K, V]`` bf16 checkpoints. Each slot is dense, with
            V contiguous. The slot stride can be any value.
        record_k: ``[slots, VH, CAP, K]`` bf16 k records.
        record_u: ``[slots, VH, CAP, V]`` bf16 u records.
        record_g: ``[slots, VH, CAP]`` fp32 g records, with any strides.
        write_pos: ``[slots]`` int32 count of the committed records. The op reads it.
        slot_idx: ``[seqs]`` int32 state slot of each sequence, or ``pad_slot_id``.
        conv_weight, conv_bias: bf16 ``[channels, 4]`` and ``[channels]`` (or ``None``).
        A_log, dt_bias, norm_weight: fp32 or bf16 ``[VH]``, ``[VH]`` and ``[V]``.
        max_query_len: the window length.
        scale: the scale of q, usually ``K ** -0.5``.
        norm_eps: the RMSNorm epsilon.
        out: ``[tokens, VH, V]`` bf16, contiguous. Receives ``norm(core) * silu(z)``.
        pad_slot_id: the slot value of a padded sequence. Padded sequences and
            sequences without tokens are not changed.
        out_scale: 1x32 MXFP8 output for the out projection (opt-in). With ``out`` a
            float8_e4m3fn ``[tokens, VH, V]`` and ``out_scale`` a contiguous uint8 buffer
            of ``pad32(R) * VH * V / 32`` bytes, ``R = out_scale_rows`` (default: tokens;
            the out projection GEMM's rows, which may exceed the rows written here), the
            op writes the gated RMSNorm as fp8 values with one e8m0 scale per 32 columns,
            in the 1x32 GEMM's x-scale layout: n32k4 for ``R >= 32`` (byte of row m,
            group g at ``(m // 32) * VH * V + (g // 4) * 128 + (m % 32) * 4 + g % 4``),
            row-major ``(R, VH * V / 32)`` below. Bit-identical to the bf16 ``out``
            followed by the standalone quant (per_group_quant_hip(group_size=32,
            scale_type=fp8_e8m0, scale_layout_m32k4=True), or per_1x32_mx_quant_hip for
            ``R < 32``). Rows of padded sequences, rows past ``tokens`` and the n32k4 pad
            rows are not written. Chunk-form kernel only: check
            ``fused_gdn_verify_mx32_supported`` first.

    Returns:
        ``out``.

    Limits: gfx1250 only. ``K = V = 128`` and ``VH`` is a multiple of ``KH``. ``CAP``
    must hold two windows. The rows of the projections and the records, ``out``,
    ``conv_weight`` and the 1-D operands other than ``conv_state_indices`` must be
    contiguous. The chunk-form kernel takes power-of-2 head ratios, windows of up to
    8 tokens and at least 9 records while the group's checkpoints and records fit
    in LDS; other shapes run token by token. The config is tuned for
    Qwen3.5-35B-A3B (head ratio 2, a window of 4 tokens, 16 records) at 1 to 512
    sequences. The chunk form sums in another order than the token loop, so the
    two agree to the bf16 rounding of the outputs, not bitwise.
    """
    from aiter.ops.triton._gluon_kernels.gfx1250.gated_delta_net.fused_gdn_verify_qkvz import (
        _fused_gdn_verify,
        _fused_gdn_verify_chunk,
        _fused_gdn_verify_chunk_compact,
        _fused_gdn_verify_chunk_mat,
    )

    _, v_heads, head_k_dim, head_v_dim = checkpoint.shape
    channels = conv_state.shape[1]
    k_heads = (channels - v_heads * head_v_dim) // (2 * head_k_dim)
    ratio = v_heads // k_heads
    seqs = slot_idx.shape[0]
    cap = record_k.shape[2]
    # The kernel computes these addresses without strides.
    assert checkpoint.stride()[1:] == (
        head_k_dim * head_v_dim,
        head_v_dim,
        1,
    ), "each checkpoint slot must be a dense [VH, K, V]"
    assert out.is_contiguous() and out.shape == (
        projected_qkvz.shape[0],
        v_heads,
        head_v_dim,
    ), "out must be a contiguous [tokens, VH, V]"
    mx32 = out_scale is not None
    scale_rows = out.shape[0] if out_scale_rows is None else out_scale_rows
    if mx32:
        assert out.dtype == torch.float8_e4m3fn, "out_scale needs an fp8 e4m3 out"
        assert scale_rows >= out.shape[0], "out_scale_rows must cover out's rows"
        assert (
            out_scale.is_contiguous()
            and out_scale.dtype == torch.uint8
            and out_scale.numel()
            == (scale_rows + 31) // 32 * 32 * (v_heads * head_v_dim // 32)
        ), "out_scale must be a contiguous uint8 buffer of pad32(rows) * VH * V / 32"
    assert (
        conv_weight.is_contiguous() and conv_weight.shape[1] == 4
    ), "conv_weight must be a contiguous [channels, 4]"
    assert (
        projected_qkvz.stride(1)
        == projected_ba.stride(1)
        == record_k.stride(3)
        == record_u.stride(3)
        == 1
    ), "the rows of the projections and the records must be contiguous"
    _LOGGER.info(
        "FUSED_GDN_VERIFY_QKVZ: seqs=%d k_heads=%d v_heads=%d max_query_len=%d cap=%d",
        seqs,
        k_heads,
        v_heads,
        max_query_len,
        cap,
    )
    if seqs == 0:
        return out
    config = _get_config(seqs)
    has_bias = conv_bias is not None
    rk, ru, rg = record_k.stride(), record_u.stride(), record_g.stride()
    num_warps, kl, ks = config["num_warps"], config["KL"], config["KS"]
    # The chunk-form kernel where it fits, else the token-loop kernel.
    kernel, tile_args = _fused_gdn_verify, {"KL": kl, "KS": ks}
    grid_split = 1
    # AITER_GDN_VERIFY_CHUNK_WARPS overrides the chunk kernel's waves,
    # AITER_GDN_VERIFY_COMPACT=1 selects its compact form and
    # AITER_GDN_VERIFY_MATRIX=1 its matrix form (experiments, off by default).
    chunk_warps = _chunk_warps(seqs)
    if chunk_warps and _chunk_kernel_fits(
        head_k_dim, head_v_dim, ratio, max_query_len, cap, chunk_warps
    ):
        chunk_kernel = _fused_gdn_verify_chunk
        if os.environ.get("AITER_GDN_VERIFY_COMPACT", "0") == "1":
            chunk_kernel = _fused_gdn_verify_chunk_compact
        if os.environ.get("AITER_GDN_VERIFY_MATRIX", "0") == "1":
            chunk_kernel = _fused_gdn_verify_chunk_mat
        kernel, num_warps, tile_args = chunk_kernel, chunk_warps, {}
        # AITER_GDN_VERIFY_UNROLL=1: the matrix form with its products of the
        # heads unrolled (the same bits; an experiment, off by default).
        if (
            kernel is _fused_gdn_verify_chunk_mat
            and os.environ.get("AITER_GDN_VERIFY_UNROLL", "0") == "1"
        ):
            tile_args = {"UNROLL_HEADS": True}
        # AITER_GDN_VERIFY_SPLIT=2: the matrix form with the value heads of a key
        # head over that many programs, at up to the config's split_max_seqs
        # sequences and with its split_num_warps waves (an experiment, off by
        # default). The q|k side runs in each program of the group.
        split = int(os.environ.get("AITER_GDN_VERIFY_SPLIT", "1"))
        split_warps = config.get("split_num_warps", num_warps)
        if (
            kernel is _fused_gdn_verify_chunk_mat
            and split > 1
            and ratio % split == 0
            and split & (split - 1) == 0
            and seqs <= config.get("split_max_seqs", 0)
            and _chunk_kernel_fits(
                head_k_dim, head_v_dim, ratio // split, max_query_len, cap, split_warps
            )
        ):
            tickets = _split_tickets(
                conv_state.shape[0] * k_heads, split, checkpoint.device
            )
            if tickets is not None:
                tile_args = {**tile_args, "SPLIT": split, "Tickets": tickets}
                grid_split, num_warps = split, split_warps
    if mx32:
        if kernel is not _fused_gdn_verify_chunk:
            raise ValueError(
                "the 1x32 MXFP8 out is implemented in the chunk-form kernel only "
                "(fused_gdn_verify_mx32_supported)"
            )
        tile_args = {"OUT_MX32": True, "N32K4": scale_rows >= 32, "OutScale": out_scale}
    kernel[(seqs * k_heads * grid_split,)](
        query_start_loc,
        slot_idx,
        conv_state_indices,
        num_accepted_tokens,
        write_pos,
        checkpoint,
        conv_state,
        projected_qkvz,
        conv_weight,
        projected_ba,
        A_log,
        dt_bias,
        record_g,
        record_k,
        record_u,
        norm_weight,
        out,
        # A placeholder pointer when there is no bias. The kernel does not read it.
        conv_bias if has_bias else conv_weight,
        scale,
        norm_eps,
        proj_token_stride=projected_qkvz.stride(0),
        ba_token_stride=projected_ba.stride(0),
        conv_index_stride=conv_state_indices.stride(0),
        conv_slot_stride=conv_state.stride(0),
        conv_channel_stride=conv_state.stride(1),
        conv_width_stride=conv_state.stride(2),
        ckpt_slot_stride=checkpoint.stride(0),
        rk_slot_stride=rk[0],
        rk_head_stride=rk[1],
        rk_pos_stride=rk[2],
        ru_slot_stride=ru[0],
        ru_head_stride=ru[1],
        ru_pos_stride=ru[2],
        rg_slot_stride=rg[0],
        rg_head_stride=rg[1],
        rg_pos_stride=rg[2],
        KH=k_heads,
        VH=v_heads,
        K=head_k_dim,
        V=head_v_dim,
        RATIO=ratio // grid_split,
        T_MAX=max_query_len,
        CAP=cap,
        NW=num_warps,
        **tile_args,
        PAD_SLOT_ID=pad_slot_id,
        HAS_BIAS=has_bias,
        num_warps=num_warps,
    )
    return out
