# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unified perf + accuracy harness for MoE-sort + (optional) MX quant.

Covers two operations, dispatched by ``--quant-dtype``:

  * ``test_moe_mxfp4_sort``: standalone e8m0 byte sort/swizzle
    (``mxfp4_moe_sort_hip`` / Triton). The byte layout is dtype-agnostic
    (same swizzle for MXFP4 and MXFP8 scales), so this test always runs
    and is unaffected by ``--quant-dtype``.

  * ``test_moe_mx_quant_sort``: fused dynamic MX quant + sort. Selects:
      - ``--quant-dtype fp4x2`` (default): MXFP4, packed fp4 output +
        e8m0 scale. Compared paths: ref / split / HIP-fused / Triton.
      - ``--quant-dtype fp8``: MXFP8, fp8 e4m3 output + e8m0 scale.
        Compared paths: ref / split / HIP-fused (no Triton).

Also covers the MXFP4 aux kernels (``module_moe_mxfp4_aux``) behind
``fused_moe``'s A4W4 inline-sort path, gfx950 only, with invalid expert ids.
Callers mark padding rows with ``topk_ids = -1`` (e.g. vLLM's CUDA-graph
padding with ``VLLM_MOE_SKIP_PADDING=1``); the sorts must drop those routes
like ``moe_sorting`` does, and the scatter-reduce must add 0 for them:

  * ``test_moe_mxfp4_aux_sort``: ``_adaptive_moe_sort`` (one-CTA sort at
    block 16, three-stage sort at block 32/64/128).
  * ``test_moe_mxfp4_aux_sort_quant``: ``mxfp4_moe_sort_quant``.
  * ``test_moe_mxfp4_aux_scatter_reduce``: ``mxfp4_moe_scatter_reduce`` and
    ``mxfp4_moe_scatter_reduce_q``.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
import aiter.fused_moe as fm
from aiter import dtypes, get_torch_quant
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import (
    per_1x32_f8_scale_f8_quant,
    per_1x32_mx_quant_hip,
)
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(threshold=float("inf"))


def run_torch(scale, sorted_ids, num_valid_ids, token_num):
    topk = 1
    if len(scale.shape) == 3:
        topk = scale.shape[1]
        scale = scale.view(-1, scale.shape[-1])
    sorted_ids[num_valid_ids:] = token_num
    topk_ids = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xFFFFFF
    mask = sorted_ids == token_num
    if topk > 1:
        sorted_ids = sorted_ids * topk + topk_ids
    sorted_ids[mask] = 0  # set to 0 to avoid overflow
    scale = scale[sorted_ids]
    scale.view(torch.uint8)[mask] = 0
    sm, sn = scale.shape
    # Pad both dims to the tile boundaries the swizzle formula assumes:
    # rows -> multiple of 32, cols -> multiple of 8 (e.g. inter_dim=384
    # gives scaleN=12 which must be padded to 16).
    sm_pad = (sm + 31) // 32 * 32
    sn_pad = (sn + 7) // 8 * 8
    tmp = torch.zeros((sm_pad, sn_pad), dtype=scale.dtype, device=scale.device)
    tmp[:sm, :sn] = scale
    scale = tmp
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    ref = scale.view(-1, sn)
    return ref


def run_split_quant_sort(input, sorted_ids, num_valid_ids, token_num, quant_dtype):
    """Two-pass split path: per-token quant + e8m0 byte sort/swizzle.

    fp4 and fp8 share the same `mxfp4_moe_sort_hip` byte shuffle kernel
    (dtype-agnostic); only the per-token quant kernel differs.
    """
    model_dim = input.shape[-1]
    out, scale_per_token = per_1x32_mx_quant_hip(
        input,
        scale=None,
        quant_dtype=quant_dtype,
        scale_type=dtypes.fp8_e8m0 if quant_dtype == dtypes.fp8 else None,
        shuffle=False,
    )
    # Pad cols to multiple of 8 to match the kernel's swizzle stride.
    scaleN_pad = ((model_dim // 32) + 7) // 8 * 8
    out_scale_sorted = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, scaleN_pad),
        dtype=dtypes.fp8_e8m0,
        device=input.device,
    )
    aiter.mxfp4_moe_sort_hip(
        out_scale_sorted,
        scale_per_token,
        sorted_ids,
        num_valid_ids,
        token_num,
        model_dim,
    )
    return out, out_scale_sorted


def _ref_quant(input, quant_dtype):
    """Per-row MX quant reference (no sort). Returns (fp_out, per_token_scale)."""
    if quant_dtype == dtypes.fp4x2:
        return get_torch_quant(aiter.QuantType.per_1x32)(
            input, quant_dtype=dtypes.fp4x2
        )
    else:
        return per_1x32_f8_scale_f8_quant(
            input, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )


@benchmark()
def test_moe_mxfp4_sort(dtype, token_num, model_dim, E, topk, block_size, stage):
    input = torch.randn((token_num, model_dim), dtype=dtype)
    score = torch.randn((token_num, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    sorted_ids, _sorted_weights, _sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
        )
    )
    num_valid_ids = num_valid_ids[0]
    if stage == "stage1":
        scale = torch.arange(token_num * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, model_dim // 32)
        topk = 1
    else:
        scale = torch.arange(token_num * topk * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, topk, model_dim // 32)
    ref = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)
    triton_scale, triton_us = run_perftest(
        fp4_utils.moe_mxfp4_sort,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        block_size,
    )

    # Pad cols to multiple of 8: the swizzle formula `mx_scale_shuffle_idx`
    # uses `scaleN_pad = pad8(scaleN)`, so the destination buffer needs
    # `pad8(model_dim/32)` cols to avoid OOB writes when scaleN is not
    # already a multiple of 8 (e.g. inter_dim=384 -> scaleN=12 -> 16).
    scaleN_pad = ((model_dim // 32) + 7) // 8 * 8
    hip_scale = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, scaleN_pad),
        dtype=torch.uint8,
        device=input.device,
    )
    _, hip_us = run_perftest(
        aiter.mxfp4_moe_sort_hip,
        hip_scale,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        model_dim,
    )

    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    triton_err = checkAllclose(
        ref[:num_valid_ids],
        triton_scale[:num_valid_ids].view(torch.uint8),
        msg="sorted_mxfp4_scale",
    )

    hip_err = checkAllclose(
        ref[:num_valid_ids].view(torch.uint8),
        hip_scale[:num_valid_ids].view(torch.uint8),
        msg="hip sorted_mxfp4_scale",
    )
    return {
        "triton_us": triton_us,
        "triton_err": triton_err,
        "hip_us": hip_us,
        "hip_err": hip_err,
    }


@benchmark()
def test_moe_mx_quant_sort(
    dtype, token_num, model_dim, E, topk, block_size, stage, quant_dtype
):
    """Unified MXFP4 / MXFP8 quant + sort benchmark.

    Compares 4 paths (3 for fp8 -- Triton is fp4-only):
      * ref:    python ref quant + python sort
      * split:  per_1x32_mx_quant_hip + mxfp4_moe_sort_hip (2 kernels)
      * hip:    fused_dynamic_mx{fp4,fp8}_quant_moe_sort (1 kernel + dispatch)
      * triton: fused_dynamic_mxfp4_quant_moe_sort (Triton, fp4 only)
    """
    if get_gfx().startswith("gfx94"):
        return {}
    is_fp8 = quant_dtype == dtypes.fp8
    label = "mxfp8" if is_fp8 else "mxfp4"
    input = torch.randn((token_num, model_dim), dtype=dtype)
    score = torch.randn((token_num, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    sorted_ids, _sorted_weights, _sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
        )
    )
    num_valid_ids = num_valid_ids[0]
    topk_orig = topk  # keep before clobbering for stage1
    if stage != "stage1":
        input = torch.randn((token_num * topk, model_dim), dtype=dtype)
    else:
        topk = 1

    # Reference: per-row MX quant + python-side byte sort.
    ref_out, scale = _ref_quant(input, quant_dtype)
    # For stage2 the per-row scale is `(token_num * topk, scaleN)`. Reshape
    # to 3D `(token_num, topk, scaleN)` so `run_torch` sees the topk dim
    # and applies the proper `sorted_ids * topk + topk_ids` indexing
    # instead of treating it as 2D (which collapses every slot onto
    # token_idx 0..token_num-1 and produces ~3-5% byte mismatches versus
    # the HIP/split paths in stage2).
    if stage != "stage1":
        scale = scale.view(token_num, topk_orig, -1)
    ref_scale = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)

    # Split: per_1x32_mx_quant_hip + mxfp4_moe_sort_hip.
    (_split_out, split_scale), split_us = run_perftest(
        run_split_quant_sort,
        input,
        sorted_ids,
        num_valid_ids,
        token_num,
        quant_dtype,
    )

    # HIP fused: the Python wrapper internally dispatches by M (small M ->
    # single fused kernel; large M -> split path). For production-sized M
    # (e.g. 15472) this auto-selects split, matching the split column.
    hip_fn = (
        aiter.fused_dynamic_mxfp8_quant_moe_sort
        if is_fp8
        else aiter.fused_dynamic_mxfp4_quant_moe_sort
    )
    (hip_out, hip_scale), hip_us = run_perftest(
        hip_fn,
        input,
        sorted_ids,
        num_valid_ids,
        token_num,
        topk,
        block_size,
    )

    # Triton path: fp4 only. MUST run BEFORE the `num_valid_ids.item()`
    # below -- the Triton kernel takes `num_valid_ids` as a 0-d tensor
    # pointer (it does `tl.load(num_valid_ids_ptr)` internally), and
    # would otherwise see a Python int and fail at compile time with
    # "Unsupported ptr type triton.language.int32 in `tl.load`".
    triton_scale = None
    triton_us = None
    if not is_fp8:
        (_triton_out, triton_scale), triton_us = run_perftest(
            fused_dynamic_mxfp4_quant_moe_sort,
            input,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size,
        )

    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    checkAllclose(
        ref_out.view(torch.uint8), hip_out.view(torch.uint8), msg=f"hip {label} out"
    )
    # The wrapper allocates `scale` with `torch.empty` (no zero-init) for
    # perf, so positions the kernel does not write (padding rows within
    # expert blocks; padding cols when `model_dim/32` is not a multiple
    # of 8) contain allocator garbage. Production GEMM consumers never
    # read those positions, but a byte-level `checkAllclose` would
    # otherwise flag them. Mask out positions where ref is 0 (the
    # un-written / zero-init slots), matching the trick the original
    # mxfp4 test applied to the Triton path.
    hip_mask = ref_scale == 0
    hip_scale = hip_scale[: ref_scale.shape[0]]
    hip_scale.view(torch.uint8)[hip_mask] = 0
    split_scale = split_scale[: ref_scale.shape[0]]
    split_scale.view(torch.uint8)[hip_mask] = 0
    hip_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        hip_scale[:num_valid_ids].view(torch.uint8),
        msg=f"hip sorted_{label}_scale",
    )

    split_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        split_scale[:num_valid_ids].view(torch.uint8),
        msg=f"split sorted_{label}_scale",
    )

    result = {
        "hip_us": hip_us,
        "hip_err": hip_err,
        "split_us": split_us,
        "split_err": split_err,
    }

    if not is_fp8 and triton_scale is not None:
        mask = ref_scale == 0
        triton_scale = triton_scale[: ref_scale.shape[0]]
        triton_scale.view(torch.uint8)[mask] = 0
        triton_err = checkAllclose(
            ref_scale[:num_valid_ids].view(torch.uint8),
            triton_scale[:num_valid_ids].view(torch.uint8),
            msg=f"triton sorted_{label}_scale",
        )
        result["triton_us"] = triton_us
        result["triton_err"] = triton_err

    return result


# MXFP4 aux kernels with invalid expert ids (see the module docstring). The
# shapes are the DSv4.1 target and DSpark draft MoE, which have generated aux
# instances; block_size 16 runs the one-CTA sort, 32/64/128 the three-stage sort.
AUX_MODEL_DIM = 5120
AUX_SHAPES = [(384, 6), (128, 3)]
AUX_SORT_CASES = [(16, 8), (16, 64), (16, 256), (32, 512), (64, 2048), (128, 4096)]
AUX_ROUTING_CASES = ["valid", "mixed", "all-empty", "padded-rows"]


def _aux_routing(token_num, model_dim, E, topk, dtype, routing_case):
    input = torch.randn((token_num, model_dim), dtype=dtype)
    score = torch.randn((token_num, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    if routing_case == "mixed":  # every other route invalid
        invalid = torch.arange(token_num * topk).view(token_num, topk) % 2 == 1
        topk_ids.masked_fill_(invalid, -1)
        topk_weights.masked_fill_(invalid, 0)
    elif routing_case == "all-empty":
        topk_ids.fill_(-1)
        topk_weights.zero_()
    elif routing_case == "padded-rows":  # whole trailing rows, as vLLM pads
        topk_ids[token_num // 2 :] = -1
        topk_weights[token_num // 2 :] = 0
    elif routing_case != "valid":
        raise ValueError(f"unknown routing case: {routing_case}")
    return topk_ids, topk_weights


def _check_aux_sort(sort_out, topk_ids, topk_weights, E, block_size):
    """Mismatches of an aux sort against a torch reference.

    The reference drops routes outside [0, E). Within an expert the sort places
    routes in atomic order and pads with token id M, so each expert's routes
    are compared as a set; `reverse_sorted` must be -1 for a dropped route.
    """
    ids, w, eids, nv, m_indices, reverse_sorted = sort_out
    M, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).long()
    valid = (flat >= 0) & (flat < E)
    route = torch.arange(M * topk)[valid]
    eid = flat[valid]
    packed = ((route % topk) << 24) | (route // topk)
    counts = torch.bincount(eid, minlength=E)
    blocks = (counts + block_size - 1) // block_size
    total = int(blocks.sum()) * block_size
    errs = {}
    if int(nv[0]) != total or int(nv[1]) != M:
        return {"num_valid_ids": (nv.tolist(), [total, M])}
    block_eids = torch.repeat_interleave(torch.arange(E), blocks)
    if not torch.equal(eids[: block_eids.numel()].long(), block_eids):
        errs["sorted_expert_ids"] = True
    r = ids[:total].long()
    keep = (r & 0xFFFFFF) < M
    slot_e = torch.repeat_interleave(block_eids, block_size)
    got = ((slot_e[keep] << 32) | r[keep]).sort().values
    if not torch.equal(got, ((eid << 32) | packed).sort().values):
        errs["sorted_ids"] = True
    rs = reverse_sorted.long()
    if (rs[~valid] != -1).any():
        errs["reverse_sorted(dropped)"] = int((rs[~valid] != -1).sum())
    pos = rs[valid]
    if ((pos < 0) | (pos >= total)).any():
        errs["reverse_sorted(range)"] = int(((pos < 0) | (pos >= total)).sum())
        return errs
    if not torch.equal(ids[pos].long(), packed):
        errs["reverse_sorted(valid)"] = True
    if not torch.equal(m_indices[pos].long(), route // topk):
        errs["m_indices"] = True
    if not torch.equal(w[pos], topk_weights.reshape(-1)[valid]):
        errs["sorted_weights"] = True
    return errs


def test_moe_mxfp4_aux_sort(
    dtype, token_num, model_dim, E, topk, block_size, routing_case, atomic
):
    """MXFP4 aux sort via `_adaptive_moe_sort(emit_aux=True)`.

    block_size 16 runs the one-CTA sort (with or without the atomic zero-init),
    32/64/128 the three-stage sort.
    """
    topk_ids, topk_weights = _aux_routing(
        token_num, model_dim, E, topk, dtype, routing_case
    )
    out = fm._adaptive_moe_sort(
        topk_ids,
        topk_weights,
        E,
        topk,
        block_size,
        model_dim,
        atomic=atomic,
        emit_aux=True,
    )
    ids, w, eids, nv, _moe_buf, m_indices, reverse_sorted = out
    errs = _check_aux_sort(
        (ids, w, eids, nv, m_indices, reverse_sorted),
        topk_ids,
        topk_weights,
        E,
        block_size,
    )
    assert not errs, (
        f"MXFP4 aux sort mismatch for E={E}, topk={topk}, block_size={block_size}, "
        f"token={token_num}, atomic={atomic}, routing_case={routing_case}: {errs}"
    )


def test_moe_mxfp4_aux_sort_quant(dtype, token_num, model_dim, E, topk, routing_case):
    """One-CTA sort fused with the MXFP4 activation quant (block 32).

    The sort is checked like `test_moe_mxfp4_aux_sort`; the quant output of
    `mxfp4_moe_sort_quant` must match `mxfp4_moe_quant` byte for byte.
    """
    block_size = 32
    topk_ids, topk_weights = _aux_routing(
        token_num, model_dim, E, topk, dtype, routing_case
    )
    x = torch.randn((token_num, model_dim), dtype=dtype)
    active = min(E, token_num * topk)
    max_sorted = (
        (token_num * topk + active * (block_size - 1) + block_size - 1)
        // block_size
        * block_size
    )
    ids = torch.empty(max_sorted, dtype=dtypes.i32)
    eids = torch.empty(max_sorted // block_size, dtype=dtypes.i32)
    nv = torch.empty(2, dtype=dtypes.i32)
    reverse_sorted = torch.empty(token_num * topk, dtype=dtypes.i32)
    w = torch.empty(max_sorted, dtype=dtypes.fp32)
    m_indices = torch.empty(max_sorted, dtype=dtypes.i32)
    aq = torch.empty(token_num * model_dim // 2, dtype=torch.uint8)
    a_scale = torch.empty(token_num * model_dim // 32, dtype=torch.uint8)
    no_zero = torch.empty(0, dtype=dtypes.bf16)
    aiter.mxfp4_moe_sort_quant(
        a_input=x,
        topk_ids=topk_ids,
        topk_weight=topk_weights,
        sorted_token_ids=ids,
        sorted_expert_ids=eids,
        cumsum_tensor=nv,
        reverse_sorted=reverse_sorted,
        sorted_weights=w,
        a_quant=aq,
        a_scale=a_scale,
        m_indices=m_indices,
        bf16_zero_out=no_zero,
        NE=E,
        TOPK=topk,
        D_HIDDEN=model_dim,
        MB=block_size,
    )
    errs = _check_aux_sort(
        (ids, w, eids, nv, m_indices, reverse_sorted),
        topk_ids,
        topk_weights,
        E,
        block_size,
    )
    ref_q, ref_scale = torch.empty_like(aq), torch.empty_like(a_scale)
    aiter.mxfp4_moe_quant(
        a_input=x,
        a_quant=ref_q,
        a_scale=ref_scale,
        bf16_zero_out=no_zero,
        NE=E,
        TOPK=topk,
        D_HIDDEN=model_dim,
        MB=block_size,
    )
    if not (torch.equal(aq, ref_q) and torch.equal(a_scale, ref_scale)):
        errs["quant"] = True
    assert not errs, (
        f"mxfp4_moe_sort_quant mismatch for E={E}, topk={topk}, "
        f"token={token_num}, routing_case={routing_case}: {errs}"
    )


def test_moe_mxfp4_aux_scatter_reduce(
    dtype, token_num, model_dim, E, topk, routing_case, quant_dtype
):
    """Scatter-reduce of the per-route rows back to tokens.

    `mxfp4_moe_scatter_reduce` (bf16 rows) or `mxfp4_moe_scatter_reduce_q`
    (MXFP4 rows): out[t] = sum over t's routes of weight * row. A route the
    sort dropped (`reverse_sorted` -1) adds 0.
    """
    block_size = 32
    topk_ids, topk_weights = _aux_routing(
        token_num, model_dim, E, topk, dtype, routing_case
    )
    ids, w, _eids, _nv, _moe_buf, _m_indices, reverse_sorted = fm._adaptive_moe_sort(
        topk_ids, topk_weights, E, topk, block_size, model_dim, emit_aux=True
    )
    rows = ids.numel()
    out = torch.empty((token_num, model_dim), dtype=dtype)
    rs = reverse_sorted.view(token_num, topk).long()
    keep = (rs >= 0).unsqueeze(-1)
    pos = rs.clamp(min=0)
    if quant_dtype == dtypes.fp4x2:
        flat_q = torch.randint(0, 256, (rows, model_dim // 2), dtype=torch.uint8)
        flat_scale = torch.randint(118, 127, (rows, model_dim // 32), dtype=torch.uint8)
        aiter.mxfp4_moe_scatter_reduce_q(
            flat_out_q=flat_q,
            flat_out_scale=flat_scale,
            reverse_sorted=reverse_sorted,
            sorted_weights=w,
            out=out,
            NE=E,
            TOPK=topk,
            D_HIDDEN=model_dim,
            MB=block_size,
        )
        scale = (flat_scale[pos].to(torch.int32) << 23).view(torch.float32)
        rows_f = fp4_utils.mxfp4_to_f32(flat_q[pos]) * scale.repeat_interleave(32, -1)
    else:
        flat = torch.randn((rows, model_dim), dtype=dtype)
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=flat,
            reverse_sorted=reverse_sorted,
            sorted_weights=w,
            out=out,
            NE=E,
            TOPK=topk,
            D_HIDDEN=model_dim,
            MB=block_size,
        )
        rows_f = flat[pos].float()
    # `torch.where`, not `* keep`: a dropped route's `sorted_weights` slot is
    # never written and may hold NaN.
    ref = torch.where(keep, rows_f * w[pos].unsqueeze(-1), 0.0).sum(1)
    err = checkAllclose(
        ref,
        out.float(),
        atol=5e-2,
        rtol=1e-2,
        msg=f"scatter_reduce {quant_dtype} E={E} token={token_num} {routing_case}",
    )
    empty = (rs < 0).all(dim=1)
    assert err == 0 and not out[empty].any(), (
        f"mxfp4_moe_scatter_reduce mismatch for E={E}, topk={topk}, "
        f"token={token_num}, routing_case={routing_case}, quant={quant_dtype}"
    )


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-q",
    "--quant-dtype",
    type=str,
    choices=["fp4x2", "fp8"],
    default="fp4x2",
    help="""MX quant element format for the quant+sort test.
    fp4x2 (default): MXFP4, packed fp4 output. Compares ref / split / HIP / Triton.
    fp8:             MXFP8, fp8 e4m3 output.   Compares ref / split / HIP (no Triton).""",
)
parser.add_argument(
    "-dim1",
    type=int,
    nargs="*",
    default=[4096, 7168],
    help="""Model dimension for stage1.
    e.g.: -dim1 4096""",
)
parser.add_argument(
    "-dim2",
    type=int,
    nargs="*",
    default=[256, 2048],
    help="""Inter dimension for stage2.
    e.g.: -dim2 256""",
)
parser.add_argument(
    "-ek",
    "--expert_topk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[[32, 5], [256, 8], [512, 8]],
    help="""Number of experts.
    e.g.: -ek 32,5""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 64, 128, 256, 1024, 2050, 4200, 10000, 163840],
    help="""M of mnk.
    e.g.: -m 64""",
)
parser.add_argument(
    "-bm",
    "--block_m",
    type=int,
    default=32,
    choices=[16, 32, 64, 80, 128],
    help="""Block M.
    e.g.: -bm 64""",
)

args = parser.parse_args()
_quant_dtype = dtypes.fp4x2 if args.quant_dtype == "fp4x2" else dtypes.fp8
_label = args.quant_dtype  # for log msg

# Standalone byte sort/swizzle test. The underlying kernels
# (`mxfp4_moe_sort_hip`, `fp4_utils.moe_mxfp4_sort`) are dtype-agnostic
# (uint8 byte shuffle), but the Triton path is fp4-only by name, and the
# HIP path is implicitly exercised inside `test_moe_mx_quant_sort`'s split
# path. To avoid the misleading "triton_us" column when running with
# `-q fp8`, only run this test for the fp4 mode.
if _quant_dtype == dtypes.fp4x2:
    df = []
    for dtype in args.dtype:
        for (
            dim,
            (E, topk),
            m,
        ) in itertools.product(args.dim1, args.expert_topk, args.m):
            ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, args.block_m, "stage1")
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("moe_sorting_mxfp4_stage1 summary (markdown):\n%s", df_md)

    df = []
    for dtype in args.dtype:
        for (
            dim,
            (E, topk),
            m,
        ) in itertools.product(args.dim2, args.expert_topk, args.m):
            ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, args.block_m, "stage2")
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("moe_sorting_mxfp4_stage2 summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim1, args.expert_topk, args.m):
        ret = test_moe_mx_quant_sort(
            dtype, m, dim, E, topk, args.block_m, "stage1", _quant_dtype
        )
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_%s_quant_sort_stage1 summary (markdown):\n%s", _label, df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim2, args.expert_topk, args.m):
        ret = test_moe_mx_quant_sort(
            dtype, m, dim, E, topk, args.block_m, "stage2", _quant_dtype
        )
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_%s_quant_sort_stage2 summary (markdown):\n%s", _label, df_md)

# MXFP4 aux sorts and scatter-reduce with invalid expert ids: correctness only,
# each case asserts. gfx950 only, like the MXFP4 fused-MoE paths that use
# `module_moe_mxfp4_aux`.
if get_gfx() == "gfx950":
    for dtype in args.dtype:
        for (E, topk), (block_size, m), routing_case, atomic in itertools.product(
            AUX_SHAPES, AUX_SORT_CASES, AUX_ROUTING_CASES, [False, True]
        ):
            test_moe_mxfp4_aux_sort(
                dtype, m, AUX_MODEL_DIM, E, topk, block_size, routing_case, atomic
            )
        for (E, topk), m, routing_case in itertools.product(
            AUX_SHAPES, [32, 512], AUX_ROUTING_CASES
        ):
            test_moe_mxfp4_aux_sort_quant(
                dtype, m, AUX_MODEL_DIM, E, topk, routing_case
            )
        for (E, topk), m, routing_case, quant_dtype in itertools.product(
            AUX_SHAPES, [64, 512], AUX_ROUTING_CASES, [dtypes.bf16, dtypes.fp4x2]
        ):
            test_moe_mxfp4_aux_scatter_reduce(
                dtype, m, AUX_MODEL_DIM, E, topk, routing_case, quant_dtype
            )
    aiter.logger.info("moe_mxfp4_aux sort / sort_quant / scatter_reduce: all passed")
