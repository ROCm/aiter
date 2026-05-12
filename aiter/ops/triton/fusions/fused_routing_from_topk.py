# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# Fused replacement for the multi-kernel "topk → routing data" chain that
# bridges FusedMoE.select_experts to triton_kernels.matmul_ogs. See the
# accompanying _triton_kernels/fused_routing_from_topk.py for the kernel.
from typing import Tuple

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_routing_from_topk import (
    _fused_routing_from_topk_hist_kernel,
    _fused_routing_from_topk_offset_kernel,
    _fused_routing_from_topk_place_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


# Maximum NK supported by the single-CTA fused kernel is 4096. Above this the
# wrapper raises rather than degrading silently — callers should fall
# back to a multi-kernel reference path for prefill-shaped inputs (NK in
# the tens of thousands). Decode (num_tokens × top_k) at typical batch
# sizes is well within this budget.


def fused_routing_from_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expts_tot: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort (token, slot) pairs by their expert id via a single Triton kernel.

    Replaces the multi-kernel torch chain (per-row sort, gather, two stable
    argsorts, advanced indexing, fp32 histc) with a single counting-sort
    kernel launch. Intended for use in the
    triton_kernels.matmul_ogs MoE pipeline, where the launch overhead of
    the original chain is a measurable fraction of decode TPOT.

    Args:
        topk_weights: ``[n_tokens, n_expts_act]`` per-token routing weights.
        topk_ids: ``[n_tokens, n_expts_act]`` selected expert ids; values
            in ``[0, n_expts_tot)``.
        n_expts_tot: Total number of routed experts (= ``E``).

    Returns:
        Tuple ``(hist, topk_indx, gate_indx, gate_scal)``:

          - ``hist[E] int32``: tokens-per-expert histogram. Sums to
            ``n_tokens * n_expts_act``.
          - ``topk_indx[n_tokens * n_expts_act] int32``: the
            ``GatherIndx.src_indx`` — for each expert-sorted position, the
            original flat index ``token * K + slot`` it came from.
          - ``gate_indx[n_tokens * n_expts_act] int32``: the
            ``GatherIndx.dst_indx`` — inverse permutation of
            ``topk_indx``. Use as ``ScatterIndx.src_indx``.
          - ``gate_scal[n_tokens * n_expts_act]``: routing weights in
            expert-sorted order. Same dtype as ``topk_weights``.

    Notes:
        The kernel does NOT pre-sort each token's ``K`` expert ids. The
        resulting ``topk_indx`` / ``gate_indx`` differ from a
        stable-argsort reference at *intra-expert* ordering, but they form
        a valid inverse permutation pair and ``matmul_ogs`` produces the
        same per-token aggregation (gather + weighted scatter sum are
        both commutative over a per-expert slice).

        ``NK = n_tokens * n_expts_act`` must be ``<= 4096`` (single-CTA
        design budget). Callers should fall back to a multi-kernel
        reference implementation for larger NK; ``ValueError`` is raised
        when this limit is exceeded so the failure surfaces explicitly.
    """
    n_tokens, n_expts_act = topk_weights.shape
    n_gates_pad = n_tokens * n_expts_act

    assert n_gates_pad <= 4096, (
        f"fused_routing_from_topk: NK={n_gates_pad} exceeds the "
        f"single-CTA budget of 4096. Caller should "
        f"dispatch to a reference implementation for NK above this."
    )

    _LOGGER.info(
        f"FUSED_ROUTING_FROM_TOPK: n_tokens={n_tokens} K={n_expts_act} "
        f"E={n_expts_tot} NK={n_gates_pad}"
    )

    device = topk_weights.device
    weights_dtype = topk_weights.dtype

    # Triton kernel needs flat int32 inputs. .reshape on a contiguous tensor
    # is a view; .contiguous() / .to(int32) on already-canonical tensors
    # are no-ops.
    topk_ids_flat = topk_ids.contiguous().reshape(-1).to(torch.int32)
    topk_weights_flat = topk_weights.contiguous().reshape(-1)

    topk_indx = torch.empty(n_gates_pad, dtype=torch.int32, device=device)
    gate_indx = torch.empty(n_gates_pad, dtype=torch.int32, device=device)
    gate_scal = torch.empty(n_gates_pad, dtype=weights_dtype, device=device)
    hist = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
    offset_scratch = torch.empty(n_expts_tot, dtype=torch.int32, device=device)

    BLOCK_NK = max(triton.next_power_of_2(int(n_gates_pad)), 32)
    BLOCK_E = max(triton.next_power_of_2(int(n_expts_tot)), 32)

    # Kernel 1 (Phase A): histogram via tl.histogram (warp-local
    # shared-memory reduction). num_warps=1 keeps the reduction within a
    # single wave, matching the CTA-local design of the original kernel.
    _fused_routing_from_topk_hist_kernel[(1,)](
        topk_ids_flat,
        hist,
        n_gates_pad,
        E=n_expts_tot,
        BLOCK_NK=BLOCK_NK,
        BLOCK_E=BLOCK_E,
        num_warps=1,
    )

    # Kernel 2 (Phase B): exclusive prefix-sum hist → offset. The kernel
    # boundary above publishes hist without an explicit barrier.
    _fused_routing_from_topk_offset_kernel[(1,)](
        hist,
        offset_scratch,
        E=n_expts_tot,
        BLOCK_E=BLOCK_E,
        num_warps=1,
    )

    # Kernel 3 (Phase C): placement. The kernel boundary publishes the
    # prefix-sum offsets without an explicit barrier or atomic_xchg.
    _fused_routing_from_topk_place_kernel[(1,)](
        topk_ids_flat,
        topk_weights_flat,
        offset_scratch,
        topk_indx,
        gate_indx,
        gate_scal,
        n_gates_pad,
        BLOCK_NK=BLOCK_NK,
        num_warps=1,
    )

    return hist, topk_indx, gate_indx, gate_scal
