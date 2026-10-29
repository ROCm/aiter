# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Gated DeltaNet speculative-verify step on ReplaySSM state, for gfx1250.

One launch replaces the verify chain ``causal_conv1d_update -> gating ->
ReplaySSM -> gated RMSNorm``. Two kernels: ``_fused_gdn_verify_chunk`` runs the
window in chunk form (the state's products on WMMA, never the state itself),
and ``_fused_gdn_verify`` runs it token by token on the rebuilt state, for the
shapes that the first does not take."""

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
    chunk_warps = config.get("chunk_num_warps", 0)
    if chunk_warps and _chunk_kernel_fits(
        head_k_dim, head_v_dim, ratio, max_query_len, cap, chunk_warps
    ):
        kernel, num_warps, tile_args = _fused_gdn_verify_chunk, chunk_warps, {}
    kernel[(seqs * k_heads,)](
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
        RATIO=ratio,
        T_MAX=max_query_len,
        CAP=cap,
        NW=num_warps,
        **tile_args,
        PAD_SLOT_ID=pad_slot_id,
        HAS_BIAS=has_bias,
        num_warps=num_warps,
    )
    return out
