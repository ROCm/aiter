# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token sorting: route each token to its experts in blocks of `block_m` rows.

`moe_sorting` is the one implementation, moved here from aiter/fused_moe.py, which
imports it with its helpers: the backend comes from the same environment variables
(Opus by default, CK with AITER_USE_CK_MOE_SORTING=1, FlyDSL with
AITER_USE_FLYDSL_MOE_SORTING=1), and it also runs the local topk ids, the auxiliary
outputs (the adaptive sort) and FLAT's unsorted input. Segment `sort` runs the plain
sort through it; `hooks.set_sort_impl` replaces it there.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
`moe_sorting` and the helpers aiter.fused_moe imports from here are shared with the
legacy MoE path (re-exported from aiter.fused_moe, run by every legacy MoE call):
changes to them must keep legacy behavior
(`python tests/moe_dispatch/snapshot.py --check`). Please coordinate any change in this
module with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import functools
import os

import torch

import aiter
from aiter import dtypes, logger
from aiter.ops.moe_mxfp4_aux import _mxfp4_moe_sort_internal_is_supported
from aiter.presorted_moe import aux_sort_capacity

from .. import hooks
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..execute.metadata import AsmMoeMode
from .base import Cap, Segment
from .defaults import BLOCK_SIZE_M

_USE_CK_MOE_SORTING = os.environ.get("AITER_USE_CK_MOE_SORTING", "0") == "1"
_USE_FLYDSL_MOE_SORTING = os.environ.get("AITER_USE_FLYDSL_MOE_SORTING", "0") == "1"
# Adaptive sort has no shape fallback, so output_aux is limited to generated shapes.
_MOE_SORT_BACKEND = os.environ.get("AITER_MOE_SORT_BACKEND", "auto").lower()

AUX_SORT_OPUS = "opus"


def _aux_uses_opus(output_aux, block_size, routed_rows=None, num_experts=None):
    """Pick the auxiliary sort: Opus (one CTA per expert) or the fused one.

    block_size 16 has no three-stage sort instance -- codegen emits `aux_sort3s_*`
    only for MB in {32, 64, 128}, so it falls back to sort_quant_kernel_impl,
    whose whole sort runs under `blockIdx.x == 0`. That single CTA costs
    O(routed_rows); Opus spends one CTA per expert. So the fused sort wins while
    the routed rows are few relative to the experts and loses linearly after,
    which is why block_size 16 was originally pinned away from Opus wholesale.

    Switch on which side of that crossover we are: routed_rows >= num_experts is
    "at least one routed row per expert on average". Measured on kimi-k3 a4w4
    (NE=896, topk=16, gfx950) the prologue crosses between token 32 and 64 --
    fused is 2.2us faster at 32, Opus 0.9us faster at 64 and 10.6us faster at
    512 -- and the rule puts the boundary at token 56.

    Callers that cannot supply the shape keep the old conservative answer.
    """
    if output_aux != AUX_SORT_OPUS:
        return False
    if block_size != 16:
        return True
    if routed_rows is None or num_experts is None:
        return False
    return 3 * routed_rows >= 4 * num_experts  # crossover ~1.3 rows per expert


def _moe_prepare_unsorted_input(
    topk_ids, topk_weights, model_dim, moebuf_dtype, num_local_tokens=None
):
    device = topk_ids.device
    M = topk_ids.shape[0]
    # FLAT kernels zero their own output rows on-device and need an
    # extra M*8-byte per-token coordination region appended to moe_buf.
    # We over-allocate as a flat byte buffer and expose only the row part.
    #
    # The trailing region does not need initialisation; the kernel
    # tolerates arbitrary contents on the first reference per dispatch.
    elem_size = torch.empty(0, dtype=moebuf_dtype).element_size()
    row_bytes = M * model_dim * elem_size
    flag_bytes = 8
    flat_buf = torch.empty(row_bytes + flag_bytes, dtype=torch.uint8, device=device)
    moe_buf = flat_buf[:row_bytes].view(moebuf_dtype).view(M, model_dim)
    topk_ids_i32 = (
        topk_ids
        if topk_ids.dtype == dtypes.i32 and topk_ids.is_contiguous()
        else topk_ids.to(dtypes.i32).contiguous()
    )
    topk_weights_f32 = (
        topk_weights
        if topk_weights.dtype == dtypes.fp32 and topk_weights.is_contiguous()
        else topk_weights.to(dtypes.fp32).contiguous()
    )
    # sorted_expert_ids is unread; alias topk_ids so the pointer is valid.
    # XC is one dword. A CUDA num_local_tokens is that storage. Otherwise the
    # pointer is null (no allocation, no store) and the kernel keeps kernarg
    # token_cnt, which is M.
    if isinstance(num_local_tokens, torch.Tensor) and num_local_tokens.is_cuda:
        num_valid_ids = num_local_tokens
    else:
        num_valid_ids = torch.empty(0, dtype=dtypes.i32, device=device)
    return topk_ids_i32, topk_weights_f32, topk_ids_i32, num_valid_ids, moe_buf


def _adaptive_output_usable(output, M, model_dim, dtype, device):
    """Whether a caller ``output`` can be the adaptive sort's atomic moe_buf.

    The BM16 sort zeroes moe_buf with 4-byte stores, so the buffer has to be
    4-byte aligned on top of matching what fused_moe would have allocated.
    """
    return (
        output is not None
        and tuple(output.shape) == (M, model_dim)
        and output.dtype == dtype
        and output.device == device
        and output.is_contiguous()
        and output.data_ptr() % 4 == 0
    )


def _adaptive_moe_sort(
    topk_ids,
    topk_weights,
    num_experts,
    topk,
    block_size,
    model_dim,
    *,
    atomic=False,
    emit_aux=False,
    skip_quant=False,
    moebuf_dtype=dtypes.bf16,
    num_local_tokens=None,
    output=None,
):
    device = topk_ids.device
    M = topk_ids.shape[0]
    BM = block_size
    max_sorted = aux_sort_capacity(M, topk, num_experts, BM)

    sorted_token_ids = torch.empty(max_sorted, dtype=dtypes.i32, device=device)
    sorted_expert_ids = torch.empty(max_sorted // BM, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_sorted, dtype=dtypes.fp32, device=device)
    reverse_sorted = torch.empty(M * topk, dtype=dtypes.i32, device=device)
    m_indices = torch.empty(max_sorted, dtype=dtypes.i32, device=device)
    if atomic and _adaptive_output_usable(output, M, model_dim, moebuf_dtype, device):
        # The sort kernel zeroes (BM16) or we zero below (other BM) whatever buffer
        # it is handed, so a caller output can stand in for moe_buf: stage2
        # accumulates straight into it and _return_output skips the copy.
        moe_buf = output
    elif atomic:
        moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
    else:
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)
    # BM16 sort fuses output zeroing; three-stage sort only sorts.
    # Atomic GEMM2 needs a zeroed destination on every invocation.
    if atomic and BM != 16:
        moe_buf.zero_()
    empty_bf16 = _empty_bf16(device)
    bf16_zero = moe_buf if (atomic and (BM == 16 or skip_quant)) else empty_bf16

    # threestage-sort scratch (prologue==1). Previously allocated via
    # torch::empty inside the kernel; now passed in so the C++ TU is torch-free.
    # Size = NE*kSplitSortCtas + NE int32; kSplitSortCtas=16 mirrors
    # csrc/kernels/mxfp4_moe/moe_aux/codegen/mxfp4_moe_aux_dispatch.h.
    sort3stage_ws = (
        torch.empty(0, dtype=dtypes.i32, device=device)
        if BM == 16 or skip_quant
        else torch.empty(num_experts * 17, dtype=dtypes.i32, device=device)
    )

    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids,
        topk_weight=topk_weights,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        m_indices=m_indices,
        bf16_zero_out=bf16_zero,
        bf16_zero_workspace=empty_bf16,
        sort3stage_ws=sort3stage_ws,
        M_logical=M,
        NE=num_experts,
        TOPK=topk,
        D_HIDDEN=model_dim,
        D_INTER=1,  # (void)D_INTER in the sort path; unused
        MB=BM,
        prologue=0 if BM == 16 or skip_quant else 1,
        num_local_tokens=num_local_tokens,
    )
    std = (sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    if emit_aux:
        return (*std, m_indices, reverse_sorted)
    return std


@functools.cache
def _mxfp4_aux_instance_supported(experts, topk, hidden, block_m, zero_init):
    return _mxfp4_moe_sort_internal_is_supported(
        experts, topk, hidden, block_m, zero_init
    )


def _moe_sorting_impl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    dispatch_policy,
    use_opus,
    return_local_topk_ids=False,
    accumulate=True,
    output_aux=False,
    output=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape

    if (
        output_aux
        and expert_mask is None
        and not _aux_uses_opus(output_aux, block_size, M * topk, num_experts)
        and _mxfp4_aux_instance_supported(
            num_experts, topk, model_dim, block_size, accumulate
        )
        and _MOE_SORT_BACKEND not in ("opus", "ck")
        # only the BM16 (prologue 0) sort reads num_local_tokens
        and (num_local_tokens is None or block_size == 16)
    ):
        # adaptive (fused) sort emits the a4w4 extras (m_indices + reverse_sorted)
        # plus the atomic zero-init; opus single-pass aux is the env-gated fallback.
        # It has no expert_mask support, so EP always takes the Opus aux sort.
        # A suitable caller `output` becomes moe_buf (stage2 accumulates in place);
        # otherwise the sort allocates its own and _return_output copies.
        return _adaptive_moe_sort(
            topk_ids,
            topk_weights,
            num_experts,
            topk,
            block_size,
            model_dim,
            atomic=accumulate,
            emit_aux=True,
            moebuf_dtype=moebuf_dtype,
            num_local_tokens=num_local_tokens,
            output=output,
        )

    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    max_num_tokens_padded = max_num_m_blocks * block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    # moe_buf shape depends on the downstream stage2 path:
    #  - accumulate (or EP w/ expert_mask): stage2 atomically accumulates into [M, model_dim].
    #  - else (FlyDSL stage2 reduce mode without mask, or the non-atomic
    #    output_aux scatter, which writes every output row itself): caller owns
    #    the intermediate; allocate a placeholder here.
    # A caller buffer can stand in: the sort kernel zeroes what it is handed.
    if accumulate or (expert_mask is not None and not output_aux):
        moe_buf = (
            output
            if output is not None
            else torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        )
    else:
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)
    local_topk_ids = torch.empty_like(topk_ids) if return_local_topk_ids else None
    if return_local_topk_ids:
        # CK sorting does not emit local ids; use Opus so callers do not need a slow
        # Python-side remap or a hard failure when local expert ids are required.
        use_opus = True

    aux_m_indices = None
    aux_reverse_sorted = None
    if output_aux:
        use_opus = True
        dispatch_policy = (
            0 if _aux_uses_opus(output_aux, block_size, M * topk, num_experts) else 1
        )
        aux_m_indices = torch.empty(
            max_num_tokens_padded, dtype=dtypes.i32, device=device
        )
        aux_reverse_sorted = torch.empty(M * topk, dtype=dtypes.i32, device=device)

    if use_opus:
        ws_size = aiter.moe_sorting_opus_get_workspace_size(
            M, num_experts, topk, dispatch_policy, topk_ids.get_device()
        )
        workspace = (
            torch.empty(ws_size, dtype=torch.uint8, device=device)
            if ws_size > 0
            else None
        )
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            workspace,
            dispatch_policy,
            local_topk_ids,
            aux_m_indices,
            aux_reverse_sorted,
        )
    else:
        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            dispatch_policy,
        )
    ret = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    if output_aux:
        ret = (*ret, aux_m_indices, aux_reverse_sorted)
    if return_local_topk_ids:
        return (*ret, local_topk_ids)
    return ret


def _flydsl_moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    accumulate=True,
    output=None,
    herd_topk=0,
    herd_routed_scaling_factor=1.0,
):
    """FlyDSL sorting dispatch -- called outside torch_compile_guard."""
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd

    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = herd_topk or topk
    max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    max_num_tokens_padded = max_num_m_blocks * block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    # moe_buf shape mirrors _moe_sorting_impl: full [M, model_dim] when stage2
    # accumulates (or EP w/ expert_mask), else a (0,0) placeholder for FlyDSL
    # stage2 reduce mode. The kernel no-ops its zero pass on an empty buffer
    # (moe_buf_elems == 0), so reduce mode skips zeroing the [M, model_dim]
    # buffer entirely — the caller owns the [M, topk, model_dim] intermediate.
    # As in _moe_sorting_impl, a caller buffer can stand in.
    if (expert_mask is not None) or accumulate:
        moe_buf = (
            output
            if output is not None
            else torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        )
    else:
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)

    ret = (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    if herd_topk:
        from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_herd

        kept_ids, kept_weights = flydsl_moe_sorting_herd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            int(block_size),
            herd_routed_scaling_factor,
        )
        return (*ret, kept_ids, kept_weights)
    flydsl_moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        expert_mask,
        num_local_tokens,
    )
    return ret


def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
    return_local_topk_ids=False,
    accumulate=True,
    flat: AsmMoeMode | int = AsmMoeMode.SORTED,
    output_aux=False,
    output=None,
):
    asm_mode = AsmMoeMode(flat)
    if (
        not _USE_CK_MOE_SORTING
        and _USE_FLYDSL_MOE_SORTING
        and not return_local_topk_ids
        and asm_mode.requires_host_sort
        and not output_aux
        and dispatch_policy == 0
    ):
        return _flydsl_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            accumulate=accumulate,
            output=output,
        )
    # FLAT kernel: in-kernel routing (manifest flat=1 or flat=2); pass through unsorted topk.
    # `output` cannot be written in place: FLAT needs extra bytes appended to moe_buf.
    if not asm_mode.requires_host_sort:
        assert output is None, "FLAT over-allocates moe_buf; the caller copies"
        return _moe_prepare_unsorted_input(
            topk_ids,
            topk_weights,
            model_dim,
            moebuf_dtype,
            num_local_tokens,
        )
    try:
        return _moe_sorting_impl(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            use_opus=not _USE_CK_MOE_SORTING,
            return_local_topk_ids=return_local_topk_ids,
            accumulate=accumulate,
            output_aux=output_aux,
            output=output,
        )
    except Exception as e:
        logger.error(f"Error in moe_sorting: {e}")
        max_num_tokens_padded = int(
            topk_ids.numel() + num_experts * block_size - topk_ids.shape[1]
        )
        topk = topk_ids.shape[1]
        logger.error(
            f"Moe_sorting info: {max_num_tokens_padded=} {block_size=} {num_experts=} {topk=} {topk_ids.shape=}"
        )
        raise


def _empty_bf16(device):
    return torch.empty((0,), dtype=dtypes.bf16, device=device)


class Sort(Segment):
    name = "sort"
    covers = (Op.SORT, Op.SORT)
    structure_params = frozenset({"block_m"})
    capabilities = (
        Cap(
            features=frozenset(
                {
                    "ep_mask",
                    "num_local_tokens",
                    "doweight_stage1",
                    "act_scales",
                    "a1_scale",
                }
            )
        ),
    )

    def produces(self, p, params):
        return Buf("sorted", None, block_m=params.get("block_m"))

    def run(
        self,
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
        output,
    ):
        return hooks.sort_impl(moe_sorting)(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            accumulate=True,
            output=output,
        )
