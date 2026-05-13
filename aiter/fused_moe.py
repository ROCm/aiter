# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import aiter
import torch

# from aiter import get_torch_quant as get_quant
from aiter import ActivationType, QuantType, dtypes
from aiter import get_hip_quant as get_quant
from aiter import logger
from aiter.jit.core import AITER_CONFIGS, AITER_CSRC_DIR, PY, bd_dir, mp_lock
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter import fused_dynamic_mxfp4_quant_moe_sort, mxfp4_moe_sort_fwd

BLOCK_SIZE_M = 32

_USE_OPUS_MOE_SORTING = os.environ.get("AITER_USE_OPUS_MOE_SORTING", "0") == "1"


def _moe_sorting_torch_gfx1250(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
):
    """Pure-torch moe_sorting fallback for gfx1250 (vectorised, no host syncs).

    Both ``aiter.moe_sorting_opus_fwd`` (precompiled opus kernel) and
    ``aiter.moe_sorting_fwd`` (CK-tile) are broken on gfx1250 today:

      * The opus kernel submits to the HSA queue but never raises the
        completion signal, so any subsequent op that synchronises against
        that stream busy-waits inside ``rocr::core::InterruptSignal::WaitRelaxed``
        forever.
      * The CK-tile path calls into Composable Kernel which simply isn't
        compiled for gfx1250 (NULL kernel pointer -> segfault).

    The semantics match the reference ``moe_sorting_torch_native`` in
    ``aiter/op_tests/test_moe_sorting.py``:
      * tokens belonging to expert ``e`` are placed contiguously inside
        ``sorted_ids``, padded to a multiple of ``block_size``;
      * ``sorted_ids[i] = (topk_idx << 24) | token_idx`` for valid slots,
        ``init_val = (topk << 24) | M`` for padding slots;
      * ``sorted_expert_ids[b] = local_expert_idx`` (i.e. expert id with
        masked-out experts removed) for the b-th tile;
      * ``num_valid_ids = [num_valid_sorted_slots, num_valid_input_tokens]``.

    Implementation is fully vectorised on the GPU -- O(num_experts) python-side
    sync, but those are at most a few cheap shape lookups, not 256 ``.item()``
    + ``torch.where`` calls per forward.
    """
    assert topk_ids.is_cuda
    device = topk_ids.device
    M, topk = topk_ids.shape
    E = int(num_experts)
    BS = int(block_size)

    max_num_tokens_padded = int(topk_ids.numel() + E * BS - topk)
    max_num_m_blocks = int((max_num_tokens_padded + BS - 1) // BS)

    init_val = (int(topk) << 24) | int(M)
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=dtypes.i32, device=device
    )
    sorted_weights = torch.zeros(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.zeros(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    # ---- Build per-token (expert, token, topk, weight) flat arrays ------
    flat_expert = topk_ids.reshape(-1).to(torch.int64)            # (M*topk,)
    flat_weight = topk_weights.reshape(-1).to(dtypes.fp32)        # (M*topk,)
    arange_flat = torch.arange(
        flat_expert.numel(), device=device, dtype=torch.int64
    )
    flat_token = arange_flat // topk                              # (M*topk,)
    flat_topk_idx = arange_flat % topk                            # (M*topk,)

    # mask out tokens past the local DP slice (only applies if caller passes
    # a 1-elem tensor with num_local_tokens; matches opus semantics).
    if num_local_tokens is not None:
        # `num_local_tokens` is typically a 0-d int32 GPU tensor.
        n_local = num_local_tokens.reshape(-1)[0].to(torch.int64)
        valid_token_mask = flat_token < n_local
    else:
        valid_token_mask = torch.ones_like(flat_expert, dtype=torch.bool)

    # ---- Build local-expert mapping (skip masked-out experts) -----------
    if expert_mask is not None:
        keep = (expert_mask.to(torch.int32) != 0)
    else:
        keep = torch.ones(E, dtype=torch.bool, device=device)
    # `local_eid[i]` = position of expert i in the kept set (-1 if dropped)
    keep_i32 = keep.to(torch.int32)
    local_eid = torch.cumsum(keep_i32, dim=0).to(torch.int32) - 1
    local_eid = torch.where(
        keep, local_eid, torch.full_like(local_eid, -1)
    )                                                              # (E,)

    # token is "valid" iff its expert is kept *and* its token-id is < n_local
    # (we encode dropped experts by mapping them to expert id E for sorting,
    # so they end up at the end and never enter the valid prefix).
    keep_tok = valid_token_mask & keep[flat_expert]
    sort_key = torch.where(
        keep_tok,
        flat_expert,
        torch.full_like(flat_expert, E),       # send to the end
    )

    # ---- Stable sort by expert, then take valid prefix ------------------
    # `stable=True` keeps the natural (token, topk) order inside each bucket,
    # matching the reference loop which iterates (token, topk) row-major.
    perm = torch.argsort(sort_key, stable=True)
    sorted_expert = flat_expert[perm]                              # (M*topk,)
    sorted_token = flat_token[perm].to(torch.int32)
    sorted_tk_idx = flat_topk_idx[perm].to(torch.int32)
    sorted_w = flat_weight[perm]
    sorted_keep = keep_tok[perm]

    # ---- Per-expert counts and tile padding -----------------------------
    counts = torch.bincount(
        torch.where(keep_tok, flat_expert, torch.full_like(flat_expert, E)),
        minlength=E + 1,
    )[:E]                                                          # (E,)
    counts_kept = counts * keep.to(counts.dtype)                   # (E,)
    blocks_per_expert = (counts_kept + BS - 1) // BS               # (E,)
    pad_per_expert = blocks_per_expert * BS                        # (E,)

    # offset of each expert's first slot inside the *padded* layout
    pad_offsets = torch.zeros(E, dtype=torch.int64, device=device)
    pad_offsets[1:] = torch.cumsum(pad_per_expert[:-1], dim=0)

    # offset of each expert's first valid (un-padded) slot inside sorted_*
    valid_offsets = torch.zeros(E, dtype=torch.int64, device=device)
    valid_offsets[1:] = torch.cumsum(counts_kept[:-1], dim=0)

    n_valid_total = int(counts_kept.sum().item())                  # 1 sync
    # destination index in `sorted_ids` for each valid sorted entry.
    rank_within = (
        torch.arange(flat_expert.numel(), device=device, dtype=torch.int64)
        - valid_offsets[sorted_expert]
    )
    dest_idx = pad_offsets[sorted_expert] + rank_within

    valid_prefix = sorted_keep[:n_valid_total]
    if n_valid_total > 0:
        d = dest_idx[:n_valid_total][valid_prefix]
        packed = (
            (sorted_tk_idx[:n_valid_total][valid_prefix].to(torch.int32) << 24)
            | sorted_token[:n_valid_total][valid_prefix].to(torch.int32)
        )
        sorted_ids.index_copy_(0, d.to(torch.int64), packed)
        sorted_weights.index_copy_(
            0, d.to(torch.int64), sorted_w[:n_valid_total][valid_prefix]
        )

    # ---- Fill sorted_expert_ids with local expert id per tile ----------
    block_offsets = torch.zeros(E, dtype=torch.int64, device=device)
    block_offsets[1:] = torch.cumsum(blocks_per_expert[:-1], dim=0)
    n_block_total = int(blocks_per_expert.sum().item())            # 1 sync
    if n_block_total > 0:
        # blow up `local_eid` per expert by `blocks_per_expert`:
        block_expert = torch.repeat_interleave(local_eid, blocks_per_expert)
        sorted_expert_ids[:n_block_total] = block_expert.to(dtypes.i32)

    # `num_valid_ids[0]` is the number of *padded* valid slots (i.e. the
    # total padded length actually used), which equals `n_block_total * BS`.
    num_valid_ids[0] = int(n_block_total * BS)
    if num_local_tokens is not None:
        num_valid_ids[1] = num_local_tokens.reshape(-1)[0].to(dtypes.i32)
    else:
        num_valid_ids[1] = int(M)

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


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
):
    # gfx1250: prefer the prebuilt opus moe-sorting kernel when it works,
    # because the pure-torch fallback (_moe_sorting_torch_gfx1250) computes
    # padded slot counts that disagree with the FlyDSL stage2 kernel layout
    # (it over-counts blocks by ~1% which leaves stage2 atomic_add writing
    # to garbage rows -> output stays at zero).  Only fall back to torch if
    # the user explicitly disables opus (AITER_USE_OPUS_MOE_SORTING=0) or
    # the opus kernel is not loadable on this build.
    if get_gfx() == "gfx1250" and not use_opus:
        return _moe_sorting_torch_gfx1250(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
        )

    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)

    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
    fwd_fn(
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
    if int(os.environ.get("AITER_GFX1250_SORT_PROBE", "0")):
        torch.cuda.synchronize()
        logger.info(
            f"[sort_probe use_opus={use_opus}] "
            f"num_valid_ids={num_valid_ids.tolist()} "
            f"sorted_expert_ids={sorted_expert_ids.tolist()} "
            f"sorted_ids[:64]={sorted_ids[:64].tolist()} "
            f"sorted_weights[:32]={sorted_weights[:32].tolist()}"
        )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


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
):
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
            use_opus=_USE_OPUS_MOE_SORTING,
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
        raise e


# Lru cache will using hash to create key, which makes error when w1,w2 shape is symint.
# We can use torch.compile(dynamic=False) to avoid
@functools.lru_cache(maxsize=2048)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape

    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask: Optional[torch.tensor] = None,  # EP
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M=None,
    num_local_tokens: Optional[torch.tensor] = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
):
    if not block_size_M:
        block_size_M = -1
    return fused_moe_(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=activation.value,
        quant_type=quant_type.value,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
    )


def fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2: torch.Tensor,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,  # EP
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    # following for quant
    w1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M: int = -1,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = topk_ids.device
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype if dtype is None else dtype
    model_dim = w2.shape[1]
    moe_buf = torch.empty((M, model_dim), dtype=dtype, device=device)
    return moe_buf


@torch_compile_guard(gen_fake=fused_moe_fake)
def fused_moe_(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2: torch.Tensor,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,  # EP
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    # following for quant
    w1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M: int = -1,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # We do such convert since custom_op schema restriction on block_size_M, and Enum type
    activation = ActivationType(activation)
    quant_type = QuantType(quant_type)
    if block_size_M == -1:
        block_size_M = None
    """user API"""
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]
    isShuffled = getattr(w1, "is_shuffled", False)

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    dtype = hidden_states.dtype if dtype is None else dtype
    assert dtype in [
        dtypes.fp16,
        dtypes.bf16,
    ], f"Fused_moe unsupported out dtype: {dtype}"
    quant_type = quant_remap.get(quant_type, quant_type)
    q_dtype_w = w1.dtype
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else dtypes.fp8
    # If input is already FP8-quantized (e.g. from FP8 dispatch) with block scale,
    # use FP8 as activation dtype to skip redundant re-quantization
    if (
        quant_type == QuantType.per_1x128
        and hidden_states.dtype == dtypes.fp8
        and a1_scale is not None
    ):
        q_dtype_a = dtypes.fp8
    bf16_fp8_bound = 256
    if quant_type == QuantType.per_1x32:
        if get_gfx() == "gfx1250" and q_dtype_w == dtypes.fp8:
            q_dtype_a = dtypes.fp8
        elif activation == ActivationType.Swiglu:
            if get_gfx() == "gfx1250":
                if M >= bf16_fp8_bound:
                    q_dtype_a = dtypes.fp8
                else:
                    q_dtype_a = dtypes.fp4x2
            elif get_gfx() != "gfx950" or M < bf16_fp8_bound:
                q_dtype_a = dtypes.bf16
            elif M >= bf16_fp8_bound:
                q_dtype_a = dtypes.fp8
        else:
            q_dtype_a = dtypes.fp4x2

    _s1_tn, _s1_tk, _s1_bm, _s2_tn, _s2_tk = _gfx1250_tile_env_overrides()
    metadata = get_2stage_cfgs(
        get_padded_M(M),  # consider token_num > 1024 as prefill
        model_dim,
        inter_dim,
        E,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        quant_type,
        isG1U1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        isShuffled,
        stage1_tile_n=_s1_tn,
        stage1_tile_k=_s1_tk,
        stage1_block_m=_s1_bm,
        stage2_tile_n=_s2_tn,
        stage2_tile_k=_s2_tk,
    )

    block_size_M = metadata.block_m if block_size_M is None else block_size_M
    # Ensure block_size_M is int (metadata.block_m from CSV may be float)
    if block_size_M is not None:
        block_size_M = int(block_size_M)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )

    if metadata.run_1stage:
        return metadata.stage1(
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            isG1U1,
            block_size_M,
            # activation=activation,
            # quant_type=quant_type,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            M=M,
            device=topk_ids.device,
            doweight_stage1=doweight_stage1,
        )
    else:
        return fused_moe_2stages(
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            isG1U1,
            block_size_M,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            # following for cktile support
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
        )


def fused_moe_1stage(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    isG1U1,
    block_size_M=32,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    xbf16=False,
    kernelName: str = "",
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: Optional[torch.tensor] = None,
    M: int = None,
    device=None,
    doweight_stage1: bool = None,
):
    if quant_type == QuantType.No and activation == ActivationType.Silu and not isG1U1:
        # pure bf16
        aiter.fmoe(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
        )
    elif quant_type == QuantType.per_Token and doweight_stage1 and isG1U1:
        a8_type = w1.dtype
        _, model_dim, _ = w2.shape

        a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
        a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
        aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)

        aiter.fmoe_g1u1_tkw1(
            moe_buf,
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a8_scale,
            w1_scale,
            w2_scale,
            kernelName,
            a2_scale,
            activation,
        )
    else:
        if xbf16:
            # xquant happens inside the asm kernel for per_1x128
            a1 = hidden_states
            a1_scale = torch.empty(0, device="cuda")
        else:
            quant_func = get_quant(quant_type)
            if hidden_states.dtype != q_dtype_a:
                if quant_type == QuantType.per_1x128:
                    quant_func = functools.partial(quant_func, transpose_scale=True)
                a1, a1_scale = quant_func(
                    hidden_states,
                    scale=a1_scale,
                    quant_dtype=q_dtype_a,
                    num_rows=num_local_tokens,
                )
            else:
                assert (
                    a1_scale is not None or quant_type == QuantType.No
                ), "a1_scale must be provided for quantized input for fused_moe"
                a1 = hidden_states
                if quant_type == QuantType.per_1x128:
                    scale_t = torch.empty_like(a1_scale)
                    aiter.partial_transpose(
                        scale_t, a1_scale, num_rows=num_local_tokens
                    )
                    a1_scale = scale_t

        token_num = hidden_states.shape[0]
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        if quant_type == QuantType.per_1x32:
            a1_scale = mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
            w1_scale = w1_scale.view(E, -1)
            w2_scale = w2_scale.view(E, -1)

        if quant_type == QuantType.per_1x128:
            fmoe_func = functools.partial(
                aiter.fmoe_fp8_blockscale_g1u1,
                fc_scale_blkn=128,
                fc_scale_blkk=128,
                block_size_M=block_size_M,
            )
        elif isG1U1:
            fmoe_func = aiter.fmoe_g1u1
        else:
            aiter.fmoe_int8_g1u0(
                moe_buf,
                a1,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                a1_scale,
                w1_scale,
                w2_scale,
                fc2_smooth_scale=None,
                activation=activation,
            )
            return moe_buf

        fmoe_func(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernelName,
            fc2_smooth_scale=None,
            activation=activation,
        )
    return moe_buf


@functools.lru_cache(maxsize=2048)
def get_block_size_M(token, topk, expert, inter_dim):
    cu_num = get_cu_num()
    tileN = 128
    tgN = (inter_dim + tileN - 1) // tileN
    support_list = [32, 64, 128]

    tmp = []
    for el in support_list:
        max_num_tokens = token * topk + expert * el - topk
        tg_num = tgN * (max_num_tokens + el - 1) // el
        rnd = (tg_num + cu_num - 1) // cu_num
        empty = cu_num - tg_num % cu_num
        tmp.append((rnd, empty, el))
    return sorted(tmp, key=lambda x: x[:2])[0][-1]


@functools.lru_cache(maxsize=2048)
def use_nt(token, topk, e):
    use_nt = int(os.environ.get("AITER_USE_NT", "-1"))
    if use_nt != -1:
        return bool(use_nt)
    return (token * topk // e) < 64


@functools.lru_cache(maxsize=2048)
def get_ksplit(token, topk, expert, inter_dim, model_dim):
    aiter_ksplit = int(os.environ.get("AITER_KSPLIT", "0"))
    if aiter_ksplit != 0:
        return aiter_ksplit
    # only for moe_blk gemm1 a8w8 decode scenario
    if token * topk > expert:
        return 0
    cu_num = get_cu_num()
    tileN = 128

    tgM = token * topk  # decode tile num
    tgN = (inter_dim + tileN - 1) // tileN

    tg_num = tgN * tgM
    # if all cu already active
    if tg_num >= cu_num:
        return 0
    tilek = 256
    split_max = (cu_num + tg_num - 1) // tg_num
    # at least split = 2
    for i in reversed(range(2, split_max + 1)):
        if (model_dim % i == 0) and ((model_dim // i) % tilek == 0):
            return i
    return 0


cfg_2stages = None
# fmt: off
fused_moe_1stage_dict = {
    "gfx942":
    {
        # activation,                    quant_type,        dtype,    q_dtype_a,    q_dtype_w,   isG1U1,    doweight_stage1,      API
        (ActivationType.Silu,          QuantType.No,  dtypes.bf16,   dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,          QuantType.No,  dtypes.fp16,   dtypes.fp16,   dtypes.fp16,   False,   False) : aiter.fmoe,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,   dtypes.i4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,    QuantType.per_1x32,  dtypes.bf16,  dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False,   False) : aiter.fmoe_int8_g1u0,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False,   False) : aiter.fmoe_int8_g1u0,
    },
    "gfx950":
    {
        (ActivationType.Silu,    QuantType.per_1x32,   dtypes.bf16,   dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Gelu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,    dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   True)  : aiter.fmoe_g1u1_tkw1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
    },
    "gfx1250":
    {
    }
}
# fmt: on

quant_remap = {QuantType.per_128x128: QuantType.per_1x128}


def nextPow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def get_padded_M(M):
    padded_m = M
    if M < 32768:
        padded_m = nextPow2(padded_m)
    else:
        padded_m = 32768
    return padded_m


def _gfx1250_data_format(q_dtype_a, q_dtype_w, q_type, dtype):
    """Map aiter quant params to gfx1250 kernel data format string.

    Returns one of 'fp4', 'fp8', 'a8w4', 'fp16', 'bf16', or None if the
    combination is not directly supported by the gfx1250 FlyDSL kernels.
    """
    if q_type == QuantType.No:
        return "bf16" if dtype == dtypes.bf16 else "fp16"
    if q_type == QuantType.per_1x32:
        if q_dtype_a == dtypes.fp4x2 and q_dtype_w == dtypes.fp4x2:
            return "fp4"
        if q_dtype_a == dtypes.fp8 and q_dtype_w == dtypes.fp4x2:
            return "a8w4"
        if q_dtype_a == dtypes.fp8 and q_dtype_w == dtypes.fp8:
            return "fp8"
    return None


def _ensure_flydsl_kernels_path():
    """Ensure the FlyDSL kernels directory is importable as a top-level package.

    The gfx1250 kernel modules use bare ``from kernels.`` imports
    (matching the FlyDSL repo layout).  Adding the parent directory of
    ``kernels/`` to ``sys.path`` makes those imports resolve correctly
    inside the aiter package tree.
    """
    import sys, os
    flydsl_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ops", "flydsl"
    )
    if flydsl_dir not in sys.path:
        sys.path.insert(0, flydsl_dir)


def _gfx1250_moe_stage1(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m=32,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
    in_dtype="fp8",
    out_dtype_str="bf16",
    tile_n=128,
    tile_k=128,
    activation=ActivationType.Silu,
    # GPT-OSS style: per-expert bias (E, 2*inter_dim) f32, gate||up.
    # Kept None in the SiLU path; ``fused_moe_2stages`` only forwards
    # bias when ``activation == Swiglu`` (see ``MOEMetadata.has_bias``
    # guard).
    bias1=None,
    **_kwargs,
):
    """gfx1250 FlyDSL MOE stage1 wrapper (gate+up GEMM with activation)."""
    from aiter.ops.flydsl.moe_kernels import (
        _run_compiled, _view_safe,
        _MXSCALE_FORMAT_PACK, _mxscale_align_up, _mxscale_pick_tile_n,
        _mxscale_zero_pad_last, _mxscale_pad_weight_k,
    )
    _ensure_flydsl_kernels_path()

    token_num = hidden_states.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = hidden_states.shape[1]

    if in_dtype == "fp4":
        model_dim = model_dim * 2

    dev = hidden_states.device

    if out is None:
        torch_out_dtype = dtypes.bf16 if out_dtype_str == "bf16" else dtypes.fp16
        out = torch.zeros(
            (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
        )
    else:
        # The FlyDSL stage1 kernel only writes slots listed in sorted_token_ids;
        # any padding slot keeps whatever was in the caller's empty() buffer and
        # leaks into stage2 as garbage (→ inf/nan). Match FlyDSL UT semantics by
        # zero-initialising the output buffer up-front.
        out.zero_()

    # ------------------------------------------------------------------
    # K/N alignment: pick tile_n dividing both N (=2*inter_dim) and
    # inter_dim (required by _Stage1GateUpPackedWrapper), and zero-pad
    # K (=model_dim) up to a multiple of tile_k when the model shape is
    # not natively WMMA_K-aligned (e.g. GPT-OSS model_dim=2880).
    # ------------------------------------------------------------------
    if in_dtype in _MXSCALE_FORMAT_PACK:
        pack_a, pack_b, weight_shuffled = _MXSCALE_FORMAT_PACK[in_dtype]
        # Prefer the caller's default tile_n when it divides both N dims
        # (keeps DeepSeek on its validated tile_n=128 path). Only fall back
        # to a larger search when the default doesn't fit — e.g. GPT-OSS
        # (inter_dim=2880, 2*inter_dim=5760) where 128 leaves a 64-wide
        # remainder. In that case we match FlyDSL's ``bench_best_tile``
        # heuristic (largest multiple of align that divides both N dims,
        # capped at 256 — FlyDSL's bench target). We also require the
        # warp-tile alignment ``tile_n % 32 == 0`` so the kernel picker
        # can pick a multi-warp shape with ``warp_tile_n % 16 == 0``;
        # 240/nw never satisfies that except n_warp=1 which is buggy for
        # the Stage1GateUpPackedWrapper path on GPToss (NaN outputs).
        default_ok = (2 * inter_dim) % int(tile_n) == 0 and inter_dim % int(tile_n) == 0
        if default_ok:
            tile_n_eff = int(tile_n)
        else:
            tile_n_eff = _mxscale_pick_tile_n(
                256, 2 * inter_dim, inter_dim,
                in_dtype=in_dtype, align=32,
            )
        # The weight's effective K may already be larger than model_dim
        # (e.g. ATOM pads to multiples of 256 at load time: 2880→3072).
        # model_dim_padded must be at least the weight's effective K so the
        # kernel addresses both activation and weight with the same stride;
        # otherwise every weight row after the first is read at the wrong
        # offset, producing garbage and potential OOB access.
        weight_k_eff = w1.shape[2] * pack_b
        model_dim_padded = _mxscale_align_up(max(model_dim, weight_k_eff), tile_k)
        if model_dim_padded != model_dim:
            delta_k = model_dim_padded - model_dim
            # Activations (hidden_states, a1_scale) change every call -> skip
            # cache. Weights / weight scales are static across many calls -> cache
            # the padded copy so we don't redo a ~100MB memcpy on every fused_moe
            # invocation for shapes like GPT-OSS (model_dim=2880 -> 2944).
            hidden_states = _mxscale_zero_pad_last(hidden_states, delta_k // pack_a)
            if a1_scale is not None and a1_scale.numel() > 0:
                a1_scale = _mxscale_zero_pad_last(a1_scale, delta_k // 32,
                    0x7F if int(os.environ.get("AITER_GFX1250_SCALE_PAD_ONE", "0")) else 0)
        # Pad weight K only if the weight is still shorter than
        # model_dim_padded (i.e. when ATOM did NOT pad to 256 but tile_k
        # alignment added padding).
        weight_delta = model_dim_padded - weight_k_eff
        if weight_delta > 0:
            w1 = _mxscale_pad_weight_k(w1, weight_delta // pack_b, weight_shuffled, cache=True)
            _scale_pad_val = 0x7F if int(os.environ.get("AITER_GFX1250_SCALE_PAD_ONE", "0")) else 0
            if w1_scale is not None and w1_scale.numel() > 0:
                w1_scale = _mxscale_zero_pad_last(w1_scale, weight_delta // 32, _scale_pad_val, cache=True)
    else:
        tile_n_eff = tile_n
        model_dim_padded = model_dim

    # FlyDSL dev builds (e.g. 0.1.3.1.dev485) don't yet map DLPack dtype code 14
    # (fp8_e8m0) to an MLIR type, so view E8M0 scales as raw uint8. The bytes
    # are identical and kernels reinterpret the scale as i8 internally anyway.
    def _scale_as_uint8(t):
        return t.view(torch.uint8) if t is not None and t.dtype == dtypes.fp8_e8m0 else t

    a1_scale = _scale_as_uint8(a1_scale)
    w1_scale = _scale_as_uint8(w1_scale)

    flat_a_scale = (
        a1_scale.view(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w1_scale.view(-1) if w1_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _sort_block_m = max(16, block_m)
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    if in_dtype in ("fp4", "fp8", "a8w4"):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_mxscale_gfx1250 import (
            compile_moe_gemm1,
        )
    else:
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_wmma_gfx1250 import (
            compile_moe_gemm1,
        )

    if int(os.environ.get("AITER_GFX1250_PROBE", "0")):
        logger.info(
            f"[probe] stage1 in_dtype={in_dtype} model_dim={model_dim} "
            f"-> padded={model_dim_padded} delta_k={model_dim_padded-model_dim} "
            f"inter_dim={inter_dim} tile_n={tile_n_eff} tile_k={tile_k} "
            f"block_m={block_m} E={E} topk={topk} token_num={token_num} "
            f"a1_scale={'yes' if a1_scale is not None and a1_scale.numel() > 0 else 'no'} "
            f"w1_scale={'yes' if w1_scale is not None and w1_scale.numel() > 0 else 'no'}"
        )
    # FlyDSL kernel always takes an ``arg_bias`` tensor; pass an empty
    # f32 buffer when bias is disabled. SwiGLU + bias is the GPT-OSS
    # path; act='swiglu' is selected when the caller passes
    # ``activation == ActivationType.Swiglu``.
    _enable_bias = bias1 is not None and bias1.numel() > 0
    if _enable_bias:
        # Match the (E, 2*inter_dim) flat layout the kernel expects; in
        # the K-padding path inter_dim is unchanged (only model_dim is
        # padded), so bias dims stay valid. Cast to f32 since the kernel
        # reads f32. Caller is expected to already provide f32.
        flat_bias = bias1.contiguous().view(-1).to(dtypes.fp32)
    else:
        flat_bias = torch.empty(0, device=dev, dtype=dtypes.fp32)
    _act_str = (
        "swiglu" if activation == ActivationType.Swiglu else "silu"
    )

    _expert_sched = int(os.environ.get("AITER_GFX1250_EXPERT_SCHED", "1")) != 0
    _tdm_gather = int(os.environ.get("AITER_GFX1250_TDM_GATHER", "1")) != 0
    # K-pipeline depth. Default 1 == no pipelining (kernel default).
    # Caller is responsible for picking a value <= num_k_tiles
    # (=K_padded//tile_k); the FlyDSL kernel raises otherwise.
    _num_buffers = int(os.environ.get("AITER_GFX1250_NUM_BUFFERS", "1"))
    exe = compile_moe_gemm1(
        model_dim=model_dim_padded,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=block_m,
        tile_n=tile_n_eff,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        in_dtype=in_dtype,
        out_dtype=out_dtype_str,
        enable_bias=_enable_bias,
        act=_act_str,
        expert_sched_mode=_expert_sched,
        use_tdm_gather=_tdm_gather,
        use_tdm_gather_as=_tdm_gather,
        num_buffers=_num_buffers,
    )

    args = (
        _view_safe(out.view(-1)),
        _view_safe(hidden_states.view(-1)),
        _view_safe(w1.view(-1)),
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        flat_bias,
        token_num,
        inter_dim,
        model_dim_padded,
        _grid_y,
        torch.cuda.current_stream(),
    )

    _run_compiled(exe, args)
    return out


def _gfx1250_moe_stage2(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    w2_scale=None,
    a2_scale=None,
    block_m=32,
    sorted_weights=None,
    in_dtype="fp8",
    out_dtype_str="bf16",
    tile_n=128,
    tile_k=128,
    # Per-expert bias (E, model_dim) f32. In atomic-accumulate mode the
    # epilogue divides by ``topk`` so summing across the topk per-token
    # atomic_adds reproduces a single ``+ bias`` (matches torch ref).
    bias2=None,
    **_kwargs,
):
    """gfx1250 FlyDSL MOE stage2 wrapper (down-projection GEMM)."""
    from aiter.ops.flydsl.moe_kernels import (
        _run_compiled, _view_safe,
        _MXSCALE_FORMAT_PACK, _mxscale_align_up, _mxscale_pick_tile_n,
        _mxscale_zero_pad_last, _mxscale_pad_weight_k,
    )
    _ensure_flydsl_kernels_path()

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    if in_dtype == "fp4":
        inter_dim = inter_dim * 2

    dev = inter_states.device

    if out is None:
        torch_out_dtype = dtypes.bf16 if out_dtype_str == "bf16" else dtypes.fp16
        out = torch.zeros(
            (token_num, model_dim), dtype=torch_out_dtype, device=dev
        )
    else:
        # Stage2 kernel uses atomic_add (accumulate=True) into `out`; if the
        # caller passed an empty()-allocated buffer we must zero it first or
        # the accumulated result is garbage.
        out.zero_()

    # ------------------------------------------------------------------
    # K/N alignment: pick tile_n dividing N (=model_dim), and zero-pad
    # the K dim (=inter_dim) on activation/weight/scales to a multiple
    # of tile_k. See _gfx1250_moe_stage1 for rationale.
    # ------------------------------------------------------------------
    if in_dtype in _MXSCALE_FORMAT_PACK:
        pack_a, pack_b, weight_shuffled = _MXSCALE_FORMAT_PACK[in_dtype]
        default_ok = model_dim % int(tile_n) == 0
        if default_ok:
            tile_n_eff = int(tile_n)
        else:
            tile_n_eff = _mxscale_pick_tile_n(
                256, model_dim, in_dtype=in_dtype, align=32,
            )
        weight_k_eff = w2.shape[2] * pack_b
        inter_dim_padded = _mxscale_align_up(max(inter_dim, weight_k_eff), tile_k)
        if inter_dim_padded != inter_dim:
            delta_k = inter_dim_padded - inter_dim
            inter_states = _mxscale_zero_pad_last(inter_states, delta_k // pack_a)
            if a2_scale is not None and a2_scale.numel() > 0:
                a2_scale = _mxscale_zero_pad_last(a2_scale, delta_k // 32,
                    0x7F if int(os.environ.get("AITER_GFX1250_SCALE_PAD_ONE", "0")) else 0)
        weight_delta = inter_dim_padded - weight_k_eff
        if weight_delta > 0:
            w2 = _mxscale_pad_weight_k(w2, weight_delta // pack_b, weight_shuffled, cache=True)
            _scale_pad_val = 0x7F if int(os.environ.get("AITER_GFX1250_SCALE_PAD_ONE", "0")) else 0
            if w2_scale is not None and w2_scale.numel() > 0:
                w2_scale = _mxscale_zero_pad_last(w2_scale, weight_delta // 32, _scale_pad_val, cache=True)
    else:
        tile_n_eff = tile_n
        inter_dim_padded = inter_dim

    # See stage1 comment: view fp8_e8m0 scales as uint8 for FlyDSL DLPack.
    def _scale_as_uint8(t):
        return t.view(torch.uint8) if t is not None and t.dtype == dtypes.fp8_e8m0 else t

    a2_scale = _scale_as_uint8(a2_scale)
    w2_scale = _scale_as_uint8(w2_scale)

    flat_a_scale = (
        a2_scale.view(-1) if a2_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w2_scale.view(-1) if w2_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(
            sorted_token_ids.shape, dtype=torch.float32, device=dev
        )
    )

    m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)

    if in_dtype in ("fp4", "fp8", "a8w4"):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_mxscale_gfx1250 import (
            compile_moe_gemm2,
        )
    else:
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_wmma_gfx1250 import (
            compile_moe_gemm2,
        )

    if int(os.environ.get("AITER_GFX1250_PROBE", "0")):
        logger.info(
            f"[probe] stage2 in_dtype={in_dtype} inter_dim={inter_dim} "
            f"-> padded={inter_dim_padded} delta_k={inter_dim_padded-inter_dim} "
            f"model_dim={model_dim} tile_n={tile_n_eff} tile_k={tile_k} "
            f"block_m={block_m} E={E} topk={topk} token_num={token_num} "
            f"a2_scale={'yes' if a2_scale is not None and a2_scale.numel() > 0 else 'no'} "
            f"w2_scale={'yes' if w2_scale is not None and w2_scale.numel() > 0 else 'no'}"
        )
    _enable_bias_s2 = bias2 is not None and bias2.numel() > 0
    if _enable_bias_s2:
        flat_bias2 = bias2.contiguous().view(-1).to(dtypes.fp32)
    else:
        flat_bias2 = torch.empty(0, device=dev, dtype=dtypes.fp32)

    _expert_sched_s2 = int(os.environ.get("AITER_GFX1250_EXPERT_SCHED", "1")) != 0
    _tdm_gather_s2 = int(os.environ.get("AITER_GFX1250_TDM_GATHER", "1")) != 0
    _stage2_skip = int(os.environ.get("AITER_GFX1250_STAGE2_SKIP", "0")) != 0
    if _stage2_skip:
        # Diagnostic: skip the actual stage2 GEMM and return the zero-initialised
        # ``out`` buffer. Used to bisect whether stage1 or stage2 is the hang.
        if int(os.environ.get("AITER_GFX1250_PROBE", "0")):
            logger.info("[probe] stage2 SKIPPED (AITER_GFX1250_STAGE2_SKIP=1)")
        return out
    # K-pipeline depth: shared env with stage1 (AITER_GFX1250_NUM_BUFFERS,
    # default 1). Caller picks a value <= num_k_tiles for stage2's
    # K axis (=inter_dim_padded//tile_k), e.g. TP8 stage2 K=256 with
    # tile_k=128 only fits num_buffers<=2.
    _num_buffers_s2 = int(os.environ.get("AITER_GFX1250_NUM_BUFFERS", "1"))
    exe = compile_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim_padded,
        experts=E,
        topk=topk,
        tile_m=block_m,
        tile_n=tile_n_eff,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        in_dtype=in_dtype,
        out_dtype=out_dtype_str,
        accumulate=True,
        enable_bias=_enable_bias_s2,
        expert_sched_mode=_expert_sched_s2,
        use_tdm_gather=_tdm_gather_s2,
        use_tdm_gather_as=_tdm_gather_s2,
        num_buffers=_num_buffers_s2,
    )

    args = (
        _view_safe(out.view(-1)),
        _view_safe(inter_states.view(-1)),
        _view_safe(w2.view(-1)),
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        flat_bias2,
        token_num,
        model_dim,
        inter_dim_padded,
        m_blocks,
        torch.cuda.current_stream(),
    )

    if int(os.environ.get("AITER_GFX1250_PROBE", "0")):
        logger.info(
            f"[probe] stage2 launch: m_blocks={m_blocks} "
            f"sorted_token_ids.shape={tuple(sorted_token_ids.shape)} "
            f"sorted_expert_ids.shape={tuple(sorted_expert_ids.shape)} "
            f"num_valid={num_valid_ids.tolist() if num_valid_ids.numel() < 8 else num_valid_ids.shape} "
            f"a.dtype={inter_states.dtype} w2.dtype={w2.dtype} "
            f"as.numel={flat_a_scale.numel()} ws.numel={flat_w_scale.numel()}"
        )
    _run_compiled(exe, args)
    if int(os.environ.get("AITER_GFX1250_PROBE", "0")):
        torch.cuda.synchronize()
        nz = int((out != 0).sum())
        logger.info(
            f"[probe] stage2 done: out nonzero={nz}/{out.numel()} "
            f"absmax={float(out.float().abs().max()):.4f}"
        )
    return out


@dataclass
class MOEMetadata:
    stage1: Callable
    stage2: Callable
    block_m: int
    ksplit: int
    run_1stage: bool = False
    has_bias: bool = False
    use_non_temporal_load: bool = True
    fuse_quant: str = ""


def _flydsl_stage1_wrapper(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    activation=ActivationType.Silu,
    w1_scale=None,
    a1_scale=None,
    sorted_weights=None,
    out_scale=None,
    out_scale_sorted=None,
    bias1=None,
    **_kwargs,
):
    parsed = aiter.ops.flydsl.moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    act = "swiglu" if activation == ActivationType.Swiglu else "silu"
    _a_scale_one = parsed.get("a_scale_one", False)
    return aiter.ops.flydsl.flydsl_moe_stage1(
        a=hidden_states,
        w1=w1,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        act=act,
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        use_async_copy=True,
        k_batch=parsed.get("k_batch", 1),
        waves_per_eu=parsed.get("waves_per_eu", 3),
        b_nt=parsed.get("b_nt", 2),
        gate_mode=parsed.get("gate_mode", "separated"),
        bias=bias1,
        a_scale_one=_a_scale_one,
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
    )


def _flydsl_stage2_wrapper(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    bias2=None,
    **_kwargs,
):

    parsed = aiter.ops.flydsl.moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    return aiter.ops.flydsl.flydsl_moe_stage2(
        inter_states=inter_states,
        w2=w2,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        mode=parsed.get("mode", "atomic"),
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=parsed.get("sort_block_m", 0),
        b_nt=parsed.get("b_nt", 0),
        persist=parsed.get("persist", None),
        bias=bias2,
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
    )


def _gfx1250_tile_env_overrides():
    """Read ``AITER_GFX1250_*`` tile-config env vars.

    Returns a 5-tuple ``(stage1_tile_n, stage1_tile_k, stage1_block_m,
    stage2_tile_n, stage2_tile_k)`` where each entry is ``int`` if the
    env var is set to a positive integer, else ``None`` (fall back to
    the hardcoded default in ``get_2stage_cfgs``).

    Designed to be called at each ``get_2stage_cfgs`` call site so the
    resolved values become part of the ``lru_cache`` key; that way two
    runs with different env settings get distinct cache entries instead
    of silently sharing the first one's metadata.

    Env vars recognised:

    * ``AITER_GFX1250_STAGE1_TILE_N``
    * ``AITER_GFX1250_STAGE1_TILE_K``
    * ``AITER_GFX1250_BLOCK_M`` (== stage1/stage2 ``tile_m`` /
      ``route_tile_m`` -- shared by both stages in the current
      ``MOEMetadata.block_m`` plumbing)
    * ``AITER_GFX1250_STAGE2_TILE_N``
    * ``AITER_GFX1250_STAGE2_TILE_K``

    Unset / empty / non-positive values fall back to ``None``.
    """
    def _g(name):
        v = os.environ.get(name, "")
        if not v:
            return None
        try:
            iv = int(v)
        except ValueError:
            return None
        return iv if iv > 0 else None

    return (
        _g("AITER_GFX1250_STAGE1_TILE_N"),
        _g("AITER_GFX1250_STAGE1_TILE_K"),
        _g("AITER_GFX1250_BLOCK_M"),
        _g("AITER_GFX1250_STAGE2_TILE_N"),
        _g("AITER_GFX1250_STAGE2_TILE_K"),
    )


@functools.lru_cache(maxsize=2048)
def get_2stage_cfgs(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    activation,
    doweight_stage1,
    hidden_pad,
    intermediate_pad,
    is_shuffled=True,
    # gfx1250 FlyDSL tile overrides. ``None`` -> fall back to the
    # hardcoded default below. These are part of the lru_cache key so
    # callers can switch tile shape per-run without poisoning the
    # cache. Plumbed from env vars by the call sites; see
    # ``_gfx1250_tile_env_overrides``.
    stage1_tile_n=None,
    stage1_tile_k=None,
    stage1_block_m=None,
    stage2_tile_n=None,
    stage2_tile_k=None,
):
    _INDEX_COLS = [
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]

    # gfx1250: bypass tuning configs and route directly to FlyDSL kernels
    if get_gfx() == "gfx1250" and is_flydsl_available():
        gfx1250_fmt = _gfx1250_data_format(q_dtype_a, q_dtype_w, q_type, dtype)
        if gfx1250_fmt is not None:
            out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"
            is_mxscale = gfx1250_fmt in ("fp4", "fp8", "a8w4")
            # Hardcoded defaults; callers (e.g. op_tests/test_moe_2stage.py)
            # can override via ``stage1_tile_n`` / ``stage1_tile_k`` /
            # ``stage1_block_m`` / ``stage2_tile_n`` / ``stage2_tile_k``
            # kwargs (each plumbed in from ``AITER_GFX1250_*`` env vars at
            # the call site). ``None`` here keeps the historical defaults.
            _default_tile_n = 128 if is_mxscale else 64
            _default_tile_k = 128 if is_mxscale else 64
            # default_block_m = 32
            _default_block_m = 16
            default_tile_n = (
                stage1_tile_n if stage1_tile_n is not None else _default_tile_n
            )
            default_tile_k = (
                stage1_tile_k if stage1_tile_k is not None else _default_tile_k
            )
            default_block_m = (
                stage1_block_m if stage1_block_m is not None else _default_block_m
            )
            # FlyDSL UT (test_moe_gemm_mxscale_gfx1250.py) quantises the
            # stage-1 output for a8w4 with ``_per_1x32_fp8_quant`` (fp8
            # activation × fp4 weight, i.e. another ``a8w4`` GEMM), not
            # with fp4-quant.  Stage2's in_dtype must therefore equal the
            # caller's gfx1250_fmt for fp4/fp8/a8w4 alike.
            stage2_fmt = gfx1250_fmt
            stage2_is_mxscale = stage2_fmt in ("fp4", "fp8", "a8w4")
            _stage2_default_tile_n = 128 if stage2_is_mxscale else 64
            _stage2_default_tile_k = 128 if stage2_is_mxscale else 64
            # Local var names ``stage2_tile_n``/``stage2_tile_k`` are
            # already function kwargs holding the caller override (or
            # ``None``); resolve into the names downstream code expects.
            _s2_tile_n_override = stage2_tile_n
            _s2_tile_k_override = stage2_tile_k
            stage2_tile_n = (
                _s2_tile_n_override
                if _s2_tile_n_override is not None
                else _stage2_default_tile_n
            )
            stage2_tile_k = (
                _s2_tile_k_override
                if _s2_tile_k_override is not None
                else _stage2_default_tile_k
            )

            # Demoted to DEBUG (was INFO) -- per-call host log was adding
            # 50-200us of stdout overhead per fused_moe(), which dominated
            # measured fused time for small-M / TP4-TP8 cases where the
            # GEMM itself is only a few hundred us.  %-style formatting so
            # the args are not evaluated when DEBUG is disabled.
            logger.debug(
                "[fused_moe] gfx1250 FlyDSL dispatch: format=%s, %s kernel",
                gfx1250_fmt,
                "mxscale" if is_mxscale else "wmma",
            )
            logger.debug(
                "[fused_moe] input shapes: token=%s, model_dim=%s, "
                "inter_dim=%s, expert=%s, topk=%s",
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
            )
            
            # ``has_bias=True`` lets ``fused_moe_2stages`` forward
            # ``bias1``/``bias2`` into ``extra_stage1_args`` /
            # ``extra_stage2_args`` (the guard at the bias-forwarding
            # site additionally checks activation==Swiglu and
            # quant_type==per_1x32, matching GPT-OSS's path).
            _gfx1250_has_bias = (
                activation == ActivationType.Swiglu
                and dtype in [dtypes.bf16, dtypes.fp16]
                and is_mxscale
            )
            return MOEMetadata(
                stage1=functools.partial(
                    _gfx1250_moe_stage1,
                    in_dtype=gfx1250_fmt,
                    out_dtype_str=out_dtype_str,
                    tile_n=default_tile_n,
                    tile_k=default_tile_k,
                    activation=activation,
                ),
                stage2=functools.partial(
                    _gfx1250_moe_stage2,
                    in_dtype=stage2_fmt,
                    out_dtype_str=out_dtype_str,
                    tile_n=stage2_tile_n,
                    tile_k=stage2_tile_k,
                ),
                block_m=default_block_m,
                ksplit=0,
                run_1stage=False,
                has_bias=_gfx1250_has_bias,
            )

    def get_cfg_2stages(tune_file):
        import pandas as pd

        df = pd.read_csv(tune_file)
        if "_tag" in df.columns:
            df = df[df["_tag"].fillna("") == ""]
        df = df.set_index(_INDEX_COLS).to_dict("index")
        return df

    _flydsl_fallback_cache = {}

    def get_flydsl_fallback_cfgs(tune_file):
        """Return fallback configs (rows tagged ``flydsl_fallback``)."""
        if tune_file in _flydsl_fallback_cache:
            return _flydsl_fallback_cache[tune_file]
        import pandas as pd

        if not os.path.exists(tune_file):
            _flydsl_fallback_cache[tune_file] = {}
            return {}
        df = pd.read_csv(tune_file)
        if "_tag" not in df.columns:
            _flydsl_fallback_cache[tune_file] = {}
            return {}
        fb_df = df[df["_tag"] == "flydsl_fallback"]
        if fb_df.empty:
            _flydsl_fallback_cache[tune_file] = {}
            return {}
        result = fb_df.set_index(_INDEX_COLS).to_dict("index")
        _flydsl_fallback_cache[tune_file] = result
        return result

    global cfg_2stages
    config_path = os.path.dirname(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    untune_file = os.path.join(config_path, "untuned_fmoe.csv")
    profile_file = os.path.join(config_path, "profile_fmoe.csv")
    if cfg_2stages is None:
        cfg_2stages = get_cfg_2stages(tune_file)
    cu_num = get_cu_num()
    keys = (
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        str(activation),
        str(dtype),
        str(q_dtype_a),
        str(q_dtype_w),
        str(q_type),
        use_g1u1,
        doweight_stage1,
    )

    def MainFunc():
        with open(untune_file, "a") as f:
            if os.path.getsize(untune_file) == 0:
                f.write(
                    "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
                )
            q_dtype_ws = q_dtype_w if q_dtype_w != torch.uint32 else "torch.int4"
            f.write(
                f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},{dtype},{q_dtype_a},{q_dtype_ws},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
            )
        logger.info("\033[34m Start tuning fmoe")
        os.system(
            f"{PY} {AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i {untune_file} -o {tune_file} -o2 {profile_file} --last"
        )

    def FinalFunc():
        logger.info(
            f"[Hint] tuned configs are saved in {tune_file}, you can set AITER_CONFIG_FMOE to this file to use tuned configs"
        )
        logger.info("\033[0m")

    cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
    if cfg is None and os.environ.get("AITER_ONLINE_TUNE", "0") == "1":
        lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{keys}")
        mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
        cfg_2stages = get_cfg_2stages(tune_file)
        cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
        if cfg is None:
            logger.warning(f"Fmoe tuning not support for {keys}")
    if cfg is not None and not is_flydsl_available():
        kn1 = str(cfg.get("kernelName1", ""))
        kn2 = str(cfg.get("kernelName2", ""))
        if kn1.startswith("flydsl_") or kn2.startswith("flydsl_"):
            fallback_cfgs = get_flydsl_fallback_cfgs(tune_file)
            fallback = fallback_cfgs.get(keys)
            if fallback is not None:
                cfg = fallback
                logger.info(
                    f"[fused_moe] flydsl unavailable, using fallback config for {keys}"
                )
            else:
                cfg = None
                logger.warning(
                    f"[fused_moe] flydsl unavailable and no fallback for {keys}, "
                    "using default heuristics"
                )

    # gfx1250 safety net: a tuned cfg whose stage1 OR stage2 kernelName is a
    # CK kernel ("moe_ck2stages_*") will dispatch into ck_moe_stage1 /
    # ck_moe_stage2_fwd, but Composable Kernel is not built for gfx1250 -- the
    # call lands on a NULL kernel pointer and the process segfaults.  Drop any
    # such cfg on gfx1250 so we fall back to the default-heuristics path that
    # routes everything through the flydsl wrappers.
    if cfg is not None and get_gfx() == "gfx1250":
        kn1 = str(cfg.get("kernelName1", ""))
        kn2 = str(cfg.get("kernelName2", ""))
        if kn1.startswith("moe_ck2stages") or kn2.startswith("moe_ck2stages"):
            logger.warning(
                f"[fused_moe] gfx1250: tuned cfg for {keys} contains a CK "
                f"kernel ({kn1=}, {kn2=}); CK is unsupported on gfx1250, "
                "discarding cfg and falling back to default heuristics."
            )
            cfg = None

    use_non_temporal_load = False
    if cfg is None or int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")):
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        run_1stage_xbf16 = False
        if (
            activation,
            q_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            use_g1u1,
            doweight_stage1,
        ) in fused_moe_1stage_dict[get_gfx()]:
            if q_type == QuantType.per_1x128:
                # for fp8 blockscale, ck has better performance so disable assembly kernel
                run_1stage = token > 32 and (inter_dim % 128 == 0)
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.i8:
                run_1stage = token > 32
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:
                run_1stage = token > 16 or inter_dim % 128 != 0
            elif q_type != QuantType.per_1x32:
                run_1stage = token < 256

            if run_1stage and q_type == QuantType.per_1x128 and get_gfx() == "gfx950":
                run_1stage_xbf16 = int(os.environ.get("AITER_XBFLOAT16", "0")) == 1

        block_m = (
            BLOCK_SIZE_M
            if run_1stage
            else (
                (64 if token > 32 else 16)
                if q_type == QuantType.per_1x128
                else get_block_size_M(token, topk, expert, inter_dim)
            )
        )
        ksplit = (
            ksplit
            if (run_1stage)
            else (
                get_ksplit(token, topk, expert, inter_dim, model_dim)
                if q_type in [QuantType.per_1x128, QuantType.per_1x32]
                else ksplit
            )
        )
        use_non_temporal_load = use_nt(token, topk, expert)
        aiter.logger.info(
            f"run_1stage = {run_1stage}, xbf16 = {run_1stage_xbf16}, ksplit = {ksplit} q_type = {q_type} block_m = {block_m} use_nt = {use_non_temporal_load}, estimated_m_per_expert = {token * topk // expert}"
        )
    else:
        block_m = cfg["block_m"]
        if int(os.environ.get("AITER_KSPLIT", "0")) != -1:
            ksplit = cfg["ksplit"]
        else:
            ksplit = 0
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
        run_1stage = cfg.get("run_1stage", False)
        if not is_shuffled and not run_1stage:
            logger.warning(
                f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
                "Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
                "Running with preshuffle_off may produce incorrect results."
            )
        if "xbf16" in cfg:
            run_1stage_xbf16 = run_1stage and bool(int(cfg["xbf16"]))
        else:
            run_1stage_xbf16 = run_1stage and "blockscaleBf16" in str(kernelName1)

    tag = f"({kernelName1=}, {kernelName2=})"
    logger.info(
        f"[fused_moe] using {'1stage' if run_1stage else '2stage'}{' xbf16' if run_1stage_xbf16 else ''} {'default' if cfg is None else tag} for {keys} "
    )

    def get_block_m() -> int:
        if q_dtype_a == dtypes.fp8:
            return 32
        else:
            return 16 if token < 2048 else 32 if token < 16384 else 64

    if run_1stage:
        # never hard code block_m for 1-stage since it can be tuned by kernel itself, and we have different heuristics for different quant types
        # # TODO: enable this approach for other quant types and archs
        # if q_type == QuantType.per_1x128 and get_gfx() == "gfx950":
        #     tkn_per_epr = token * topk // expert
        #     block_m = 64 if tkn_per_epr > 32 else block_m
        return MOEMetadata(
            functools.partial(
                fused_moe_1stage,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                xbf16=run_1stage_xbf16,
            ),
            None,
            block_m,
            ksplit,
            run_1stage,
        )
    is_flydsl1 = bool(kernelName1) and kernelName1.startswith("flydsl_")
    is_flydsl2 = bool(kernelName2) and kernelName2.startswith("flydsl_")
    if (is_flydsl1 or is_flydsl2) and is_flydsl_available():
        _s1_fq = is_flydsl1 and "_fp4" in kernelName1.split("_t")[-1]
        if is_flydsl1:
            stage1_func = functools.partial(
                _flydsl_stage1_wrapper,
                kernelName=kernelName1,
                activation=activation,
            )
        else:
            stage1_func = functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dtype=dtype,
                splitk=ksplit,
                use_non_temporal_load=use_non_temporal_load,
            )

        if is_flydsl2:
            stage2_func = functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kernelName2,
            )
        else:
            stage2_func = functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type,
                use_non_temporal_load=use_non_temporal_load,
            )

        _has_bias = (
            activation == ActivationType.Swiglu
            and q_type == QuantType.per_1x32
            and dtype in [dtypes.bf16, dtypes.fp16]
        )
        _s1_fp8q = is_flydsl1 and "_fp8" in kernelName1.split("_t")[-1]
        _fuse_quant = "fp8" if _s1_fp8q else ("fp4" if _s1_fq else "")
        return MOEMetadata(
            stage1_func,
            stage2_func,
            block_m,
            int(ksplit),
            run_1stage,
            has_bias=_has_bias,
            fuse_quant=_fuse_quant,
        )
    if (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
    ):
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=max(ksplit, 1),
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            get_block_m(),
            ksplit,
            False,
            True,
        )
    elif (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and q_dtype_w in [dtypes.fp4x2]
        and ksplit > 1
        and is_shuffled
    ):
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=ksplit,
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            16 if token < 2048 else 32 if token < 16384 else 64,
            ksplit,
            run_1stage,
        )

    if (kernelName1 and "ck2stages" in kernelName1) or (
        not kernelName1
        and (
            (q_type == QuantType.per_1x128 and doweight_stage1)
            or q_dtype_w
            in [
                dtypes.bf16,
                dtypes.fp16,
                torch.uint32,
                dtypes.fp4x2,
                dtypes.fp8,
            ]
        )
    ):
        if kernelName2 and kernelName2.startswith("flydsl_") and is_flydsl_available():
            stage2_func = functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kernelName2,
            )
        else:
            stage2_func = functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type,
                use_non_temporal_load=use_non_temporal_load,
            )
        return MOEMetadata(
            functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dtype=dtype,
                splitk=ksplit,
                use_non_temporal_load=use_non_temporal_load,
            ),
            stage2_func,
            block_m,
            int(ksplit),
            run_1stage,
        )

    # TODO: remove when stage2 support more size
    tmpList = [16, 32, 64, 128]
    if block_m not in tmpList:
        tag = ""
        block_m = ([el for el in tmpList if block_m < el] + [128])[0]

    return MOEMetadata(
        functools.partial(
            asm_stage1,
            kernelName=kernelName1,
            activation=activation,
            quant_type=q_type,
        ),
        functools.partial(
            aiter.ck_moe_stage2_fwd,
            kernelName=kernelName2,
            activation=activation,
            quant_type=q_type,
        ),
        block_m,
        ksplit,
        run_1stage,
    )


def fused_moe_2stages(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_out,
    isG1U1,
    block_size_M,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: Optional[torch.tensor] = None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
):
    quant_func = get_quant(quant_type)
    token_num, _ = hidden_states.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = moe_out.dtype
    device = hidden_states.device
    is_shuffled = getattr(w1, "is_shuffled", False)
    _s1_tn, _s1_tk, _s1_bm, _s2_tn, _s2_tk = _gfx1250_tile_env_overrides()
    metadata = get_2stage_cfgs(
        get_padded_M(token_num),  # consider token_num > 1024 as prefill
        model_dim,
        inter_dim,
        E,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        quant_type,
        isG1U1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        is_shuffled,
        stage1_tile_n=_s1_tn,
        stage1_tile_k=_s1_tk,
        stage1_block_m=_s1_bm,
        stage2_tile_n=_s2_tn,
        stage2_tile_k=_s2_tk,
    )
    if (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and activation == ActivationType.Swiglu
            or (q_dtype_a in [dtypes.fp4x2] and metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a1 = hidden_states.to(dtype)
        a1_scale = None
    elif (
        quant_type == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype == dtypes.fp4x2
        and activation == aiter.ActivationType.Swiglu
        # gfx1250 must NOT take the e4m3fn / power-of-2 dtypeMax path here:
        # the FlyDSL a8w4 kernel decodes the activation byte stream as
        # e4m3fnuz (bias 8) and reads the scale as max_abs/finfo(fnuz).max,
        # so we let control fall through to the dedicated gfx1250 elif
        # below which mirrors the FlyDSL UT _per_1x32_fp8_quant exactly.
        and get_gfx() != "gfx1250"
    ):
        if get_gfx() == "gfx1250":
            # Dead code: the outer guard above (``get_gfx() != "gfx1250"``)
            # already excludes gfx1250 from this branch -- the gfx1250
            # stage1 fp8 quant lives in the dedicated elif below that
            # mirrors the FlyDSL UT exactly.  Kept for grep/back-compat.
            from aiter.ops.quant import _per_1x32_f8_e8m0_quant_triton

            a1, a1_scale = _per_1x32_f8_e8m0_quant_triton(
                hidden_states.to(dtypes.fp32)
            )
            a1_scale = a1_scale.view(torch.uint8).view(dtypes.fp8_e8m0)
        else:
            a1 = hidden_states.to(dtypes.fp8)
            M = sorted_ids.shape[0]
            N = a1.shape[-1]
            if metadata.fuse_quant == "fp8":
                a1_scale = torch.empty([1], dtype=dtypes.fp8_e8m0, device=a1.device)
            else:
                a1_scale = torch.ones(
                    [M, N // 32], dtype=dtypes.fp8_e8m0, device=a1.device
                )
    elif (
        quant_type == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype == dtypes.fp8
    ):
        from aiter.ops.quant import _per_1x32_f8_e8m0_quant_triton

        a1, a1_scale = _per_1x32_f8_e8m0_quant_triton(
            hidden_states.to(dtypes.fp32)
        )
        a1_scale = a1_scale.view(torch.uint8).view(dtypes.fp8_e8m0)

    elif quant_type == QuantType.per_1x32 and get_gfx() == "gfx1250":
        if int(os.environ.get("AITER_GFX1250_DEBUG", "0")):
            logger.info(
                f"[probe] stage1 quant entry: q_dtype_a={q_dtype_a} "
                f"q_dtype_w={q_dtype_w} hidden.dtype={hidden_states.dtype} "
                f"a1_scale_is_None={a1_scale is None}"
            )
        # FlyDSL gfx1250 MoE GEMM kernels address scale_x with buffer size
        # `tokens * K//32` in **source-token order** (see sx_nbytes in
        # moe_gemm_2stage_mxscale_gfx1250.py); the kernel gathers the
        # per-token scale via sorted_token_ids internally. The default
        # aiter path (fused_dynamic_mxfp4_quant_moe_sort / mxfp4_moe_sort_fwd)
        # returns a *pre-sorted* tile layout of shape (padded_sorted_size,
        # K//32), which the FlyDSL kernel cannot address correctly and
        # produces numerically uncorrelated output. Keep the quantization
        # output in source-token layout for gfx1250.
        #
        # Pick the quantization dtype to match the FlyDSL kernel's expected
        # activation format (gfx1250_fmt):
        #   * "fp4"   -> a1 must be fp4x2  (q_dtype_a == fp4x2)
        #   * "a8w4"  -> a1 must be fp8    (q_dtype_a == fp8, w1 dtype fp4x2)
        #   * "fp8"   -> a1 must be fp8    (q_dtype_a == fp8, w1 dtype fp8)
        # Quantizing to the wrong width here (the original branch assumed
        # fp4 for everything) makes the kernel read an fp8-stride buffer
        # as fp4, scaling the output by ~2^7 and producing 100% mismatch.
        if hidden_states.dtype == q_dtype_a and a1_scale is not None:
            a1 = hidden_states
            a1_scale = a1_scale.view(torch.uint8).view(dtypes.fp8_e8m0)
        elif q_dtype_a == dtypes.fp4x2:
            # Use the triton kernel (per_1x32_f4_quant_triton): it's
            # cuda-graph capturable (custom HIP ops with mutating kwargs
            # break capture) and fast enough at warmup-sized M (the
            # earlier 10-minute hang we saw with this path was the *pure
            # torch* reference, not the triton kernel).
            from aiter.ops.quant import per_1x32_f4_quant_triton
            a1, a1_scale = per_1x32_f4_quant_triton(hidden_states, quant_dtype=dtypes.fp4x2)
        elif q_dtype_a == dtypes.fp8:
            # a8w4 / a8w8 path on gfx1250.  Match FlyDSL UT exactly
            # (FlyDSL/tests/kernels/test_moe_gemm_mxscale_gfx1250.py::_per_1x32_fp8_quant):
            #
            #   * scale algo:  dtype_max = 240 (fnuz max), even though the
            #                  byte encoding is fn (bias 7).
            #   * byte encoding: bias-7 e4m3 (== float8_e4m3fn) via
            #                    fp4_utils._f32_to_floatx_unpacked(_, 4, 3).
            #                    NOTE: PyTorch's .to(e4m3fnuz) cast emits
            #                    0x80 (NaN sentinel) for tiny negatives and
            #                    poisons the whole next-stage GEMM K-sum;
            #                    the unpacked fn encoder clamps to ±240 and
            #                    never emits 0x80, matching what the FlyDSL
            #                    kernel decodes.
            #   * clamp before cast to ±240 (UT line 79).
            #   * layout:      a1 = (M, K) fn bytes (uint8 view-equivalent),
            #                  a1_scale = (M, K//32) e8m0 in src-token order.
            from aiter.utility import fp4_utils as _aiter_fp4u
            try:
                from FlyDSL.tests.kernels.utils import fp4_utils as _fly_fp4u
            except ImportError:
                import importlib, sys, os as _os
                for _root in ("/app/FlyDSL",):
                    if _root not in sys.path:
                        sys.path.insert(0, _root)
                _fly_fp4u = importlib.import_module(
                    "tests.kernels.utils.fp4_utils"
                )
            BLOCK = 32
            DTYPE_MAX = 240.0  # fnuz finfo.max -- matches UT
            a1_flat = hidden_states.view(-1, hidden_states.shape[-1]).to(
                dtypes.fp32
            )
            M_, K_ = a1_flat.shape
            assert K_ % BLOCK == 0, (
                f"per_1x32 fp8 quant on gfx1250 requires K%{BLOCK}==0 "
                f"(got K={K_})"
            )
            blk = a1_flat.view(-1, BLOCK)
            blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = blk.abs().amax(dim=1)
            scale_e8m0 = _aiter_fp4u.f32_to_e8m0(max_abs / DTYPE_MAX)
            scale_f32 = _aiter_fp4u.e8m0_to_f32(scale_e8m0)
            scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
            scale_f32[scale_f32 == 0] = 1.0
            y_f32 = blk.float() / scale_f32.unsqueeze(1)
            y_f32 = torch.clamp(y_f32, min=-DTYPE_MAX, max=DTYPE_MAX)
            a1 = _fly_fp4u._f32_to_floatx_unpacked(
                y_f32.contiguous().view(-1), 4, 3
            ).view(M_, K_)
            if int(os.environ.get("AITER_GFX1250_DEBUG", "0")):
                _ub2 = a1.view(torch.uint8).to(torch.int32)
                logger.info(
                    f"[probe] stage1 a1 fn-encoded: shape={tuple(a1.shape)} "
                    f"dtype={a1.dtype} 0x80_count={int((_ub2 == 0x80).sum())} "
                    f"byte_min={int(_ub2.min())} byte_max={int(_ub2.max())}"
                )
            a1 = a1.view(*hidden_states.shape[:-1], K_)
            a1_scale = scale_e8m0.view(M_, K_ // BLOCK).view(torch.uint8).view(
                dtypes.fp8_e8m0
            )
        else:
            raise NotImplementedError(
                f"gfx1250 fused_moe per_1x32 stage1 quant: unsupported "
                f"q_dtype_a={q_dtype_a}"
            )
    elif quant_type == QuantType.per_1x32:
        if hidden_states.dtype == dtypes.fp4x2 and a1_scale is not None:
            # Input is already quantized to fp4x2 (e.g., from FP4 dispatch),
            # skip re-quantization, only sort the scale
            a1 = hidden_states
            a1_scale = mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
        else:
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                topk=topk,
                block_size=block_size_M,
                num_rows=num_local_tokens,
            )
    elif hidden_states.dtype != q_dtype_a:
        if quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
            quant_func = functools.partial(quant_func, transpose_scale=True)
        a1, a1_scale = quant_func(
            hidden_states,
            scale=a1_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
        )
    else:
        assert (
            a1_scale is not None or quant_type == QuantType.No
        ), "a1_scale must be provided for quantized input for fused_moe"
        a1 = hidden_states
    if quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        ratio = a1_scale.element_size() // a1.element_size()
        a2 = torch.empty(
            (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
            dtype=q_dtype_a,
            device=device,
        )
    else:
        a2 = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=device,
        )
    extra_stage1_args = {}
    extra_stage2_args = {}
    if (
        not metadata.run_1stage
        and metadata.has_bias
        and dtype in [dtypes.bf16, dtypes.fp16]
        and quant_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
    ):
        extra_stage1_args["bias1"] = bias1
        extra_stage2_args["bias2"] = bias2
    a2 = metadata.stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        None if metadata.fuse_quant else a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=(
            w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale
        ),
        sorted_weights=sorted_weights if doweight_stage1 else None,
        **extra_stage1_args,
    )
    if metadata.fuse_quant == "fp4" and isinstance(a2, tuple):
        a2_raw, a2_scale = a2[0], a2[1]
        _fp4_bytes = token_num * topk * (inter_dim // 2)
        a2 = (
            a2_raw.view(-1)
            .view(torch.uint8)[:_fp4_bytes]
            .view(dtypes.fp4x2)
            .reshape(token_num, topk, -1)
        )
    elif metadata.fuse_quant == "fp8" and isinstance(a2, tuple):
        a2, a2_scale = a2[0], a2[1]
        a2 = a2.view(token_num, topk, -1)
    elif (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and activation == ActivationType.Swiglu
            or (metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a2_scale = None
    elif (
        quant_type == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype == dtypes.fp4x2
        and activation == aiter.ActivationType.Swiglu
        and get_gfx() != "gfx1250"
    ):
        a2 = a2.to(dtypes.fp8)
        a2_scale = a1_scale
    elif quant_type == QuantType.per_1x32 and w1.dtype == dtypes.fp8:
        from aiter.ops.quant import _per_1x32_f8_e8m0_quant_triton

        a2_flat = a2.view(-1, inter_dim)
        a2_flat, a2_scale = _per_1x32_f8_e8m0_quant_triton(
            a2_flat.to(dtypes.fp32)
        )
        a2_scale = a2_scale.view(torch.uint8).view(dtypes.fp8_e8m0)
        a2 = a2_flat.view(token_num, topk, -1)
    elif quant_type == QuantType.per_1x32 and get_gfx() == "gfx1250":
        # Stage2 FlyDSL kernel expects scale_x in source order of shape
        # (tokens*topk, inter_dim//32). See comment in stage1 path.
        # Pick the quant width to match the FlyDSL stage2 kernel (which
        # uses the same gfx1250 format string as stage1):
        #   * a8w4 / fp8 -> fp8 (e4m3fnuz)
        #   * fp4        -> fp4x2
        if q_dtype_a == dtypes.fp4x2:
            # Triton kernel: cuda-graph capturable + fast at warmup M
            # (see matching note in the stage1 fp4 branch above).
            from aiter.ops.quant import per_1x32_f4_quant_triton
            a2_flat = a2.view(-1, inter_dim)
            a2_flat, a2_scale = per_1x32_f4_quant_triton(a2_flat, quant_dtype=dtypes.fp4x2)
            a2_scale = a2_scale.view(torch.uint8).view(dtypes.fp8_e8m0)
            a2 = a2_flat.view(token_num, topk, -1)
        elif q_dtype_a == dtypes.fp8:
            # Mirror the FlyDSL UT's _per_1x32_fp8_quant exactly: scale
            # uses dtype_max=240 (fnuz max) but the byte encoding is fn
            # (bias 7, via _f32_to_floatx_unpacked).  See stage1 elif
            # above for full rationale.
            from aiter.utility import fp4_utils as _aiter_fp4u
            try:
                from FlyDSL.tests.kernels.utils import fp4_utils as _fly_fp4u
            except ImportError:
                import importlib, sys
                for _root in ("/app/FlyDSL",):
                    if _root not in sys.path:
                        sys.path.insert(0, _root)
                _fly_fp4u = importlib.import_module(
                    "tests.kernels.utils.fp4_utils"
                )
            BLOCK = 32
            DTYPE_MAX = 240.0
            if int(os.environ.get("AITER_GFX1250_DEBUG", "0")):
                _af = a2.float()
                logger.info(
                    f"[probe] stage2 a2 pre-quant: shape={tuple(a2.shape)} "
                    f"dtype={a2.dtype} min={float(_af.min()):.3f} "
                    f"max={float(_af.max()):.3f} "
                    f"absmax={float(_af.abs().max()):.3f} "
                    f"nan={int(torch.isnan(_af).sum())} "
                    f"inf={int(torch.isinf(_af).sum())}"
                )
            a2_flat = a2.view(-1, inter_dim).to(dtypes.fp32)
            assert inter_dim % BLOCK == 0
            blk = a2_flat.view(-1, BLOCK)
            blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = blk.abs().amax(dim=1)
            scale_e8m0 = _aiter_fp4u.f32_to_e8m0(max_abs / DTYPE_MAX)
            scale_f32 = _aiter_fp4u.e8m0_to_f32(scale_e8m0)
            scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
            scale_f32[scale_f32 == 0] = 1.0
            y_f32 = blk.float() / scale_f32.unsqueeze(1)
            y_f32 = torch.clamp(y_f32, min=-DTYPE_MAX, max=DTYPE_MAX)
            a2_q = _fly_fp4u._f32_to_floatx_unpacked(
                y_f32.contiguous().view(-1), 4, 3
            ).view(-1, inter_dim)
            if int(os.environ.get("AITER_GFX1250_DEBUG", "0")):
                _ub2 = a2_q.view(torch.uint8).to(torch.int32)
                logger.info(
                    f"[probe] stage2 a2 fn-encoded: 0x80_count="
                    f"{int((_ub2 == 0x80).sum())} "
                    f"byte_min={int(_ub2.min())} byte_max={int(_ub2.max())}"
                )
            a2 = a2_q.view(token_num, topk, -1)
            a2_scale = (
                scale_e8m0.view(token_num * topk, inter_dim // BLOCK)
                .view(torch.uint8)
                .view(dtypes.fp8_e8m0)
            )
            if int(os.environ.get("AITER_GFX1250_DEBUG", "0")):
                _af = a2.view(torch.uint8).to(torch.int32)
                _sf = a2_scale.view(torch.uint8).to(torch.int32)
                logger.info(
                    f"[probe] stage2 a2 post-quant: shape={tuple(a2.shape)} "
                    f"dtype={a2.dtype} byte_min={int(_af.min())} "
                    f"byte_max={int(_af.max())} "
                    f"scale_shape={tuple(a2_scale.shape)} "
                    f"scale_min={int(_sf.min())} scale_max={int(_sf.max())}"
                )
        else:
            raise NotImplementedError(
                f"gfx1250 fused_moe per_1x32 stage2 quant: unsupported "
                f"q_dtype_a={q_dtype_a}"
            )
    elif quant_type == QuantType.per_1x32:
        a2 = a2.view(-1, inter_dim)
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M,
            num_rows=num_local_tokens,
        )
        a2 = a2.view(token_num, topk, -1)
    elif quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        a2_v = a2[:token_num, :, :]
        a2_scale = (
            a2[token_num:, ...]
            .view(-1)[: token_num * topk * inter_dim * ratio // 128]
            .view(dtypes.fp32)
            .view(token_num, -1)
        )
        a2 = a2_v
    elif quant_type == QuantType.No:
        a2_scale = None
    else:
        a2, a2_scale = quant_func(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2 = a2.view(token_num, topk, inter_dim)

    metadata.stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=(
            w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale
        ),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
        **extra_stage2_args,
    )

    return moe_out


def torch_moe_act(act_input, torch_act, inter_dim):
    if act_input.shape[-1] == inter_dim:
        return torch_act(act_input)
    else:
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        return torch_act(gate) * up


def asm_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,  # [token_num, topk, inter_dim]
    topk,
    block_m: int,
    kernelName: str = "",
    ksplit: int = 0,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
):
    dtype = dtypes.bf16  # out.dtype, asm only support bf16
    if quant_type != QuantType.per_1x128:
        out = out.view(dtype)
    device = out.device
    token_num, _, _ = out.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type == QuantType.per_Tensor:
        a1_scale = a1_scale.view(1, 1).repeat(token_num, 1)
        w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[1])
        quant_type = QuantType.per_Token

    tmp_out = out
    if ksplit > 0:
        tmp_out = torch.zeros(
            (token_num, topk, w1.shape[1]),
            dtype=dtypes.fp32,
            device=device,
        ).view(dtype)

    aiter.moe_stage1_g1u1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        inter_dim,
        kernelName,
        block_m,
        ksplit=ksplit,
        activation=activation,
        quant_type=quant_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
    )
    if ksplit > 0:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32))
        else:
            aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32))
    return out


def torch_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    torch_act = aiter.get_torch_act(activation)
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            act_out = torch_moe_act(act_input, torch_act, inter_dim)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


# temp workaround for swiglu
def swiglu(x_glu, x_linear, alpha: float = 1.702, limit: float = 7.0):
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def torch_moe_stage1(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weight,
    topk_ids,
    dtype=dtypes.fp16,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    # following for quant
    a1_scale=None,  # [token, 1]
    w1_scale=None,  # [expert, inter_dim, 1]
    w1_bias=None,  # [expert, inter_dim, 1]
    doweight=False,
):
    quant_type = quant_remap.get(quant_type, quant_type)
    ctype = dtypes.fp32  # compute type
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if quant_type == QuantType.per_1x32:
        from aiter.utility import fp4_utils

        w1 = fp4_utils.mxfp4_to_f32(w1)
        w1_scale = fp4_utils.e8m0_to_f32(w1_scale)
        if a1_scale is not None:  # skip a16w4
            hidden_states = fp4_utils.mxfp4_to_f32(hidden_states)
            a1_scale = fp4_utils.e8m0_to_f32(a1_scale)
        else:  # a16w4
            hidden_states = hidden_states.to(ctype)

    else:
        hidden_states = hidden_states.to(ctype)
        w1 = w1.to(ctype)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        w1 = w1 * w1_scale.view(w1_scale.shape[0], -1, 1)
        hidden_states = hidden_states * a1_scale
    # per_128x128
    elif quant_type in [QuantType.per_128x128, QuantType.per_1x128]:
        w1_shape = w1.shape
        w1 = w1.view(
            w1.shape[0], w1.shape[1] // 128, 128, w1.shape[2] // 128, 128
        ) * w1_scale.view(
            w1_scale.shape[0], w1.shape[1] // 128, 1, w1.shape[2] // 128, 1
        )
        w1 = w1.view(w1_shape)

        a1_scale = a1_scale.view(hidden_states.shape[0], -1, 1)
        a1_scale = a1_scale.repeat(
            1, 1, hidden_states.shape[-1] // a1_scale.shape[1]
        ).view(hidden_states.shape[0], -1)
        hidden_states = hidden_states * a1_scale
    elif quant_type == QuantType.No:
        pass
    elif quant_type == QuantType.per_1x32:
        w1_shape = w1.shape
        w1 = w1.view(E, N, model_dim // 32, 32) * w1_scale.view(
            E, N, model_dim // 32, 1
        )
        w1 = w1.view(w1_shape)

        a1_shape = hidden_states.shape
        hidden_states = hidden_states.view(a1_shape[0], a1_shape[1] // 32, 32)
        if a1_scale is not None:
            a1_scale = a1_scale[: a1_shape[0]]
            hidden_states = hidden_states * a1_scale.view(
                a1_shape[0], a1_shape[1] // 32, 1
            )
        hidden_states = hidden_states.view(a1_shape)
    else:
        assert False, f"Unsupported quant_type: {quant_type}"

    hidden_states = hidden_states.view(B, -1, model_dim).repeat(1, topk, 1)

    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if doweight:
                act_input = act_input * topk_weight[mask].view(-1, 1)
            out[mask] = act_input
            if w1_bias is not None:
                out[mask] = out[mask] + w1_bias[E_id].view(1, -1)
    use_g1u1 = w1.shape[1] == (2 * inter_dim)
    use_swiglu = activation == aiter.ActivationType.Swiglu
    torch_act = aiter.get_torch_act(activation)
    if use_g1u1:
        gate, up = out.split([inter_dim, inter_dim], dim=-1)
        if use_swiglu:
            out = swiglu(gate, up)
        else:
            out = torch_act(gate) * up
    else:
        out = torch_act(out)
    return out.to(dtype)


def torch_moe_stage2(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weights,
    topk_ids,
    dtype=dtypes.fp16,
    quant_type=QuantType.No,
    w2_scale=None,  # [1]
    a2_scale=None,  # [expert]]'
    w2_bias=None,
    doweight=True,
):
    ctype = dtypes.fp32  # compute type
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if quant_type == QuantType.per_1x32:
        from aiter.utility import fp4_utils

        w2 = fp4_utils.mxfp4_to_f32(w2)
        w2_scale = fp4_utils.e8m0_to_f32(w2_scale)
        if a2_scale is not None:
            hidden_states = fp4_utils.mxfp4_to_f32(hidden_states)
            a2_scale = fp4_utils.e8m0_to_f32(a2_scale)
        else:  # a16w4
            hidden_states = hidden_states.to(ctype)
    else:
        hidden_states = hidden_states.to(ctype)
        w2 = w2.to(ctype)

    token_num, topk = topk_ids.shape
    hidden_states = hidden_states.view(token_num, topk, inter_dim)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        hidden_states = hidden_states * a2_scale.view(a2_scale.shape[0], -1, 1)
        w2 = w2 * w2_scale.view(w2_scale.shape[0], -1, 1)
    elif quant_type in [QuantType.per_128x128, QuantType.per_1x128]:
        a2_scale = a2_scale.view(hidden_states.shape[0], topk, -1, 1)
        a2_scale = a2_scale.repeat(1, 1, 1, 128).view(hidden_states.shape[0], topk, -1)
        hidden_states = hidden_states * a2_scale

        w2_shape = w2.shape
        w2 = w2.view(
            w2.shape[0], w2.shape[1] // 128, 128, w2.shape[2] // 128, 128
        ) * w2_scale.view(
            w2_scale.shape[0], w2.shape[1] // 128, 1, w2.shape[2] // 128, 1
        )
        w2 = w2.view(w2_shape)
    elif quant_type == QuantType.per_1x32:
        a2_shape = hidden_states.shape
        if a2_scale is not None:
            a2_scale = a2_scale[: a2_shape[0] * topk]
            a2_scale = a2_scale.view(token_num, topk, inter_dim // 32, 1)
            hidden_states = (
                hidden_states.view(token_num, topk, inter_dim // 32, 32) * a2_scale
            )
        hidden_states = hidden_states.view(a2_shape)

        w2_shape = w2.shape
        w2 = w2.view(E, model_dim, inter_dim // 32, 32) * w2_scale.view(
            E, model_dim, inter_dim // 32, 1
        )
        w2 = w2.view(w2_shape)

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w2[E_id].transpose(0, 1))
            out[mask] = act_input
            if w2_bias is not None:
                out[mask] = out[mask] + w2_bias[E_id].view(1, -1)
    if doweight:
        out = out * topk_weights.view(token_num, -1, 1)
    return out.sum(1).to(dtype)


def ck_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    kernelName="",
    sorted_weights=None,
    quant_type=aiter.QuantType.No,
    activation=ActivationType.Gelu,
    splitk=1,
    use_non_temporal_load=False,
    dtype=None,
):
    token_num = hidden_states.shape[0]
    is_splitk = quant_type is aiter.QuantType.per_1x128 and splitk > 1
    if is_splitk:
        # CK kernel zeros this buffer via hipMemsetAsync when KBatch > 1
        sorted_size = min(token_num * topk * block_m, sorted_token_ids.shape[0])
        tmp_out = torch.empty(
            (sorted_size, w1.shape[1]), dtype=dtypes.fp32, device=out.device
        )
    else:
        tmp_out = out
    aiter.ck_moe_stage1_fwd(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        block_m,
        sorted_weights,
        quant_type,
        activation,
        splitk if is_splitk else 0,
        use_non_temporal_load,
        out.dtype,
    )
    if is_splitk:
        valid_out = tmp_out[: token_num * topk, :]
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, valid_out.view(dtypes.fp32))
        else:
            aiter.gelu_and_mul(out, valid_out.view(dtypes.fp32))
    return out


def cktile_moe_stage1(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    sorted_weights=None,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias1=None,
    activation=ActivationType.Silu,
    split_k=1,
    dtype=torch.bfloat16,
    kernel_name="",
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2 * 2  # bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8

    out = torch.empty((token_num, topk, D), dtype=dtype, device=hidden_states.device)
    # WARNING: when split_k > 1, this allocation has the same undersized buffer
    # pattern fixed in ck_moe_stage1 (see ROCm/aiter#2508). If the CK tile
    # kernel calls hipMemsetAsync with sorted_size rows, this will overflow.
    # When fp32 splitk is enabled, apply the same fix: use sorted_size =
    # min(token_num * topk * block_m, sorted_token_ids.shape[0]) and slice
    # valid_out = tmp_out[:token_num * topk, :] before silu_and_mul/gelu_and_mul.
    tmp_out = (
        torch.zeros(
            (token_num, topk, w1.shape[1]), dtype=hidden_states.dtype, device=out.device
        )
        if split_k > 1
        else out
    )

    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        tmp_out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a1_scale,
        w1_scale,
        bias1,
        activation,
        block_m,
        split_k,
        kernel_name,
    )

    if split_k > 1:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, tmp_out)  # TODO: support fp32 splitk
        else:
            aiter.gelu_and_mul(out, tmp_out)
    return out


def cktile_moe_stage2(
    a2,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    w2_scale,
    a2_scale,
    block_m,
    activation=ActivationType.Swiglu,
    sorted_weights=None,
    zeros_out=False,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias2=None,
    kernel_name="",
):
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    # out = torch.empty(
    #     (token_num, D),
    #     dtype=a2.dtype,
    #     device=a2.device,
    # )
    # if zeros_out:
    #     out.fill_(0)
    # print("Run cktile_moe_stage2: M=%d, N=%d, K=%d, topk=%d, expert=%d"%(a2.shape[0]*a2.shape[1], w2.shape[1], a2.shape[2], topk, w2.shape[0]))
    aiter.moe_cktile2stages_gemm2(
        a2,
        w2,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a2_scale,
        w2_scale,
        bias2,
        activation,
        block_m,
        kernel_name=kernel_name,
    )
    return out


def _topk_softmax_torch_gfx1250(gating_output, topk, renormalize):
    """Pure-torch topk-softmax fallback for gfx1250.

    The HIP `aiter.topk_softmax` and asm `aiter.topk_softmax_asm` are
    compiled for gfx950 / gfx942; loading and dispatching them under the
    gfx1250 simulator segfaults inside `module_moe_asm.so` before any
    completion signal is raised. Mirrors `_moe_sorting_torch_gfx1250`.

    Semantics match `test_moeTopkSoftmax.test_nofuse` (the canonical
    reference): softmax along expert dim, then top-k, then optional
    renormalisation. Outputs are (fp32 weights, i32 ids), shape (M, topk).
    """
    probs = torch.nn.functional.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = probs.topk(k=topk, dim=-1, largest=True, sorted=True)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape
    expert = gating_output.shape[1]

    if get_gfx() == "gfx1250":
        tw_t, ti_t = _topk_softmax_torch_gfx1250(gating_output, topk, renormalize)
        if topk_weights is None:
            topk_weights = tw_t
        else:
            topk_weights.copy_(tw_t)
        if topk_ids is None:
            topk_ids = ti_t
        else:
            topk_ids.copy_(ti_t)
        return topk_weights, topk_ids

    token_expert_indicies = torch.empty(
        M, topk, dtype=dtypes.i32, device=hidden_states.device
    )

    if (
        get_gfx() in ["gfx942", "gfx950"]
        and (expert, topk)
        in [
            (128, 4),
            (128, 6),
            (128, 8),
            (256, 6),
            (256, 8),
            (384, 8),
        ]
        and gating_output.dtype in [dtypes.bf16, dtypes.fp32]
        and gating_output.is_contiguous()
    ):
        if topk_weights is None:
            topk_weights = torch.empty(
                (M + 3) // 4 * 4, topk, dtype=dtypes.fp32, device=hidden_states.device
            )
        if topk_ids is None:
            topk_ids = torch.empty(
                (M + 3) // 4 * 4, topk, dtype=dtypes.i32, device=hidden_states.device
            )
        aiter.topk_softmax_asm(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
        )
        topk_weights = topk_weights[:M, :]
        topk_ids = topk_ids[:M, :]
    else:
        if topk_weights is None:
            topk_weights = torch.empty(
                M, topk, dtype=dtypes.fp32, device=hidden_states.device
            )
        if topk_ids is None:
            topk_ids = torch.empty(
                M, topk, dtype=dtypes.i32, device=hidden_states.device
            )
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
        )

    del token_expert_indicies  # Not used. Will be used in the future.

    # if renormalize:
    #     topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
