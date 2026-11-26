# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL stages of the KDA (``chunk_kimi_delta_attn``) chunk backward.

``kda_bwd_fwd_h_flydsl`` recomputes the forward's chunk-state recurrence for
``chunk_delta_attn_bwd`` on the GDN K5 kernel (``chunk_gated_delta_h``) with its
per-channel ``USE_GK`` decay: the same contract as the Triton
``chunk_delta_attn_bwd_kernel_fwd_h`` (k pre-decayed to the chunk end, log2
``g_cumsum`` as gk), so the rest of the Triton backward consumes its outputs
unchanged.
"""

from __future__ import annotations

import torch

from .linear_attention_prefill_kernels import (
    _get_or_compile_opt,
    _hipeq_select_bv,
    _is_gfx942,
    _placeholder_pair,
    _run_compiled,
)


def kda_bwd_fwd_h_supported(
    *, K: int, V: int, chunk_size: int, is_varlen: bool, dtype: torch.dtype
) -> bool:
    """Shapes the FlyDSL recurrence takes; the caller falls back to Triton otherwise."""
    # The K5 kernel bakes BT=64 and K=128 into its wave mapping; varlen needs
    # the GDR chunk schedule, which this wrapper does not build yet.
    return (
        K == 128
        and V == 128
        and chunk_size == 64
        and not is_varlen
        and dtype == torch.bfloat16
    )


def kda_bwd_fwd_h_flydsl(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk states ``h`` and ``v_new = u - w @ h`` for the KDA backward.

    Args:
        kg: ``[B, T, HV, K]`` bf16, k decayed to its chunk end.
        w, u: ``[B, T, HV, K]`` / ``[B, T, HV, V]`` bf16, token-major WY outputs.
        g_cumsum: ``[B, T, HV, K]`` chunk-local gate cumsum in log2 space.
        initial_state: ``[B, HV, K, V]`` (``[B, HV, V, K]`` if ``state_v_first``)
            or None.

    Returns:
        ``h`` as ``[B, NT, HV, K, V]`` bf16 and ``v_new`` as ``[B, T, HV, V]``
        bf16, the layouts the Triton backward reads.
    """
    B, T, HV, K = kg.shape
    V = u.shape[-1]
    BT = 64
    NT = (T + BT - 1) // BT
    kg, w, u = kg.contiguous(), w.contiguous(), u.contiguous()
    gk = g_cumsum.to(torch.float32).contiguous()

    h0 = None
    if initial_state is not None:
        # The kernel's state slot is V-first fp32.
        h0 = initial_state.to(torch.float32)
        if not state_v_first:
            h0 = h0.transpose(-1, -2)
        h0 = h0.contiguous()

    BV = _hipeq_select_bv(kg.device, HV, B * NT, NT)
    launch_fn = _get_or_compile_opt(
        K,
        V,
        BT,
        BV,
        HV,
        HV,
        False,  # use_g: KDA's decay is per channel only
        True,  # use_gk
        h0 is not None,
        False,  # store_fs: the backward does not need the final state
        True,  # save_vn
        False,  # is_varlen
        False,  # wu_contig: w/u are token-major [B, T, HV, *]
        state_bf16=False,
        g_log2_scaled=True,
        sched_gfx942=_is_gfx942(),
        bf16_convert_trunc=False,  # RNE, as the Triton recurrence's .to()
        snapshot_bf16=True,
    )

    dummy, int32_dummy = _placeholder_pair(kg.device)
    h_vk = kg.new_empty(B, NT, HV, V, K)
    v_new_hm = kg.new_empty(B, HV, T, V)
    _run_compiled(
        launch_fn,
        kg,
        u,
        w,
        v_new_hm,
        dummy,
        gk,
        h_vk,
        h0 if h0 is not None else dummy,
        dummy,
        int32_dummy,
        int32_dummy,
        int32_dummy,
        T,
        T,
        B,
        V // BV,
        B * HV,
        torch.cuda.current_stream(),
    )
    # The kernel writes the V-first snapshot and a head-major v_new.
    h = h_vk.transpose(-1, -2).contiguous()
    v_new = v_new_hm.transpose(1, 2).contiguous()
    return h, v_new
