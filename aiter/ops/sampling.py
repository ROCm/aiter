# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional

from csrc.cpp_itfs.sampling.top_k_renorm_probs import (
    top_k_renorm_probs as top_k_renorm_probs_core,
)
from csrc.cpp_itfs.sampling.top_p_sampling_from_probs import (
    top_p_sampling_from_probs as top_p_sampling_from_probs_core,
)
from csrc.cpp_itfs.sampling.top_k_top_p_sampling_from_probs import (
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs_core,
)
from csrc.cpp_itfs.torch_utils import direct_register_custom_op


def top_k_renorm_probs(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    return top_k_renorm_probs_core(
        probs,
        maybe_top_k_arr,
        top_k_val,
    )


direct_register_custom_op(
    "top_k_renorm_probs",
    top_k_renorm_probs,
    [],
)


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool = False,
) -> torch.Tensor:
    return top_p_sampling_from_probs_core(
        probs,
        indices,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
    )


direct_register_custom_op(
    "top_p_sampling_from_probs",
    top_p_sampling_from_probs,
    [],
)


# ---------------------------------------------------------------------------
# "top-k first" fast path for top_k_top_p_sampling_from_probs.
#
# Mirrors flashinfer PR #3461: for a modest scalar top_k over a large vocab,
# selecting the top-k entries first and running top-p over only those k
# survivors is far cheaper than the fused full-vocab rejection kernel, which
# launches a single CTA per row and underutilizes the GPU at small batch.
#
# Semantic note (why this is NOT a verbatim port of flashinfer): aiter's fused
# TopKTopPSamplingFromProbKernel applies the top-p threshold on the ORIGINAL
# (un-renormalized) probability mass (`aggregate_gt_pivot.value < p`),
# intersected with the top-k count. flashinfer's `top_k_first` instead
# renormalizes after top-k and then applies top-p. To stay distribution-
# equivalent to aiter's existing kernel we renormalize the k survivors (mass S
# per row, needed so the top-p kernel's proportional draw u~U[0,1) is valid)
# and scale the threshold to p' = p / S (clamped to 1), so that
#     mass_k(x > pivot) < p'   <=>   mass_orig(x > pivot) < p .
# This makes the accepted set identical to the fused kernel's, modulo
# measure-zero floating-point ties at the boundary.
# ---------------------------------------------------------------------------
_TOPK_FIRST_FAST_MAX_K = 256
_TOPK_FIRST_FAST_MIN_VOCAB = 65536


def _topk_first_fast_path_applicable(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> bool:
    return (
        indices is None
        and maybe_top_k_arr is None
        and isinstance(top_k_val, int)
        and 0 < top_k_val <= _TOPK_FIRST_FAST_MAX_K
        and probs.dim() == 2
        and probs.size(-1) >= _TOPK_FIRST_FAST_MIN_VOCAB
        and top_k_val < probs.size(-1)
    )


def _select_topk(probs: torch.Tensor, k: int):
    """Top-k selection for the fast path.

    Uses aiter's native radix-select kernel `top_k_per_row_prefill`, which is the
    fastest top-k option in this repo for small k over a large vocab at small
    batch (the regime this fast path targets): measured ~1.5-7x faster than both
    `topk_plain` and `torch.topk`. Order of the survivors does not matter here -
    they are renormalized and fed to top-p as a set. The full row is selected by
    passing rowStarts=0 and rowEnds=vocab. Returns (values: float32,
    indices: int64).
    """
    from aiter.ops.topk import top_k_per_row_prefill

    bs, vocab = probs.size(0), probs.size(-1)
    row_starts = torch.zeros(bs, dtype=torch.int32, device=probs.device)
    row_ends = torch.full((bs,), vocab, dtype=torch.int32, device=probs.device)
    ids = torch.empty((bs, k), dtype=torch.int32, device=probs.device)
    vals = torch.empty((bs, k), dtype=torch.float32, device=probs.device)
    top_k_per_row_prefill(
        probs,
        row_starts,
        row_ends,
        ids,
        vals,
        bs,
        probs.stride(0),
        probs.stride(1),
        k=k,
    )
    return vals.float(), ids.long()


def _topk_first_fast_path(
    probs: torch.Tensor,
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
) -> torch.Tensor:
    # 1. parallel top-k selection: shrinks the row the top-p kernel must scan
    #    from `vocab` down to `k`.
    values, gathered_indices = _select_topk(probs, top_k_val)
    # 2. renormalize over the k survivors so the top-p kernel's proportional
    #    draw stays well-defined; clamp guards degenerate (all-zero) rows.
    denom = values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs_k = values / denom
    # 3. scale the top-p threshold to compensate for renormalization so the
    #    accepted set matches aiter's original-mass top-p semantics.
    s = denom.squeeze(-1)
    if maybe_top_p_arr is not None:
        top_p_eff = (maybe_top_p_arr.float() / s).clamp_(max=1.0)
    else:
        top_p_eff = (torch.full_like(s, float(top_p_val)) / s).clamp_(max=1.0)
    # 4. top-p sampling over the k-element distribution -> local index in [0, k).
    local = top_p_sampling_from_probs_core(
        probs_k,
        None,
        top_p_eff,
        0.0,
        deterministic,
    )
    # 5. map the local choice back to the global vocab index. The gathered
    #    indices are always valid in [0, vocab), so the output is in-range even
    #    on degenerate rows.
    return (
        gathered_indices.gather(1, local.view(-1, 1).long()).squeeze(1).to(torch.int32)
    )


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool = False,
) -> torch.Tensor:
    if _topk_first_fast_path_applicable(probs, indices, maybe_top_k_arr, top_k_val):
        return _topk_first_fast_path(
            probs,
            top_k_val,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
        )
    return top_k_top_p_sampling_from_probs_core(
        probs,
        indices,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
    )


direct_register_custom_op(
    "top_k_top_p_sampling_from_probs",
    top_k_top_p_sampling_from_probs,
    [],
)
