# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

HEAD_DIM = 128


def make_cos_sin_cache(
    max_pos: int, rotary_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    base = 5_000_000.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_pos, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps) * (1.0 + weight.float())


def apply_rope_neox_partial(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    cos_sin = cos_sin_cache[positions].float()
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)

    rot = x[..., :rotary_dim]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def norm_rope_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    normed = gemma_rmsnorm(x.float(), weight, eps)
    return apply_rope_neox_partial(normed, positions, cos_sin_cache, rotary_dim).to(
        dtype
    )


def make_case(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    block_size: int = 16,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    num_index_heads: int = 4,
    rotary_dim: int = 64,
    seed: int = 123,
):
    torch.manual_seed(seed)
    eps = 1e-6
    max_pos = 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    iq_w = (
        torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
        if num_index_heads > 0
        else None
    )
    ik_w = (
        torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
        if num_index_heads > 0
        else None
    )
    cos_sin = make_cos_sin_cache(max_pos, rotary_dim, dtype)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda"
    )

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    iq_size = num_index_heads * HEAD_DIM
    ik_size = HEAD_DIM if num_index_heads > 0 else 0
    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size + iq_size + ik_size,
        dtype=dtype,
        device="cuda",
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]
    index_slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]

    return {
        "qkv": qkv,
        "q_norm_weight": q_w,
        "k_norm_weight": k_w,
        "index_q_norm_weight": iq_w,
        "index_k_norm_weight": ik_w,
        "cos_sin_cache": cos_sin,
        "positions": positions,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_index_heads": num_index_heads,
        "rotary_dim": rotary_dim,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "slot_mapping": slot_mapping,
        "index_slot_mapping": index_slot_mapping,
        "sizes": (q_size, kv_size, kv_size, iq_size, ik_size),
        "eps": eps,
        "dtype": dtype,
    }


def make_refs(case: dict, qkv_orig: torch.Tensor):
    q_size, kv_size, _, iq_size, ik_size = case["sizes"]
    split_sizes = [q_size, kv_size, kv_size]
    if case["num_index_heads"] > 0:
        split_sizes.extend([iq_size, ik_size])
    splits = qkv_orig.split(split_sizes, dim=-1)
    q_in, k_in, v_in = splits[:3]
    num_tokens = qkv_orig.size(0)

    q_ref = norm_rope_ref(
        q_in.view(num_tokens, case["num_heads"], HEAD_DIM),
        case["q_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["rotary_dim"],
        case["eps"],
        case["dtype"],
    ).view(num_tokens, q_size)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, case["num_kv_heads"], HEAD_DIM),
        case["k_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["rotary_dim"],
        case["eps"],
        case["dtype"],
    )
    v_ref = v_in.view(num_tokens, case["num_kv_heads"], HEAD_DIM)

    refs = {"q": q_ref, "k": k_ref, "v": v_ref}
    if case["num_index_heads"] > 0:
        iq_in, ik_in = splits[3:]
        refs["index_q"] = norm_rope_ref(
            iq_in.view(num_tokens, case["num_index_heads"], HEAD_DIM),
            case["index_q_norm_weight"],
            case["positions"],
            case["cos_sin_cache"],
            case["rotary_dim"],
            case["eps"],
            case["dtype"],
        ).view(num_tokens, iq_size)
        refs["index_k"] = norm_rope_ref(
            ik_in.view(num_tokens, 1, HEAD_DIM),
            case["index_k_norm_weight"],
            case["positions"],
            case["cos_sin_cache"],
            case["rotary_dim"],
            case["eps"],
            case["dtype"],
        ).view(num_tokens, HEAD_DIM)
    return refs


def split_qkv(case: dict, qkv: torch.Tensor):
    q_size, kv_size, _, iq_size, ik_size = case["sizes"]
    split_sizes = [q_size, kv_size, kv_size]
    if case["num_index_heads"] > 0:
        split_sizes.extend([iq_size, ik_size])
    return qkv.split(split_sizes, dim=-1)


def make_insert_outputs(
    case: dict,
    *,
    kv_cache_dtype: torch.dtype | None = None,
    index_cache_dtype: torch.dtype | None = None,
):
    q_size, _, _, iq_size, _ = case["sizes"]
    q_out = torch.empty(case["qkv"].size(0), q_size, dtype=case["dtype"], device="cuda")
    index_q_out = torch.empty(
        case["qkv"].size(0), iq_size, dtype=case["dtype"], device="cuda"
    )
    kv_cache = torch.zeros(
        case["num_blocks"],
        2,
        case["block_size"],
        case["num_kv_heads"],
        HEAD_DIM,
        dtype=kv_cache_dtype or case["dtype"],
        device="cuda",
    )
    index_cache = torch.zeros(
        case["num_blocks"],
        case["block_size"],
        HEAD_DIM,
        dtype=index_cache_dtype or case["dtype"],
        device="cuda",
    )
    return q_out, index_q_out, kv_cache, index_cache


def make_shuffle_caches(case: dict, *, kv_cache_dtype: torch.dtype | None = None):
    """Allocate page-`block_size` SHUFFLE (asm_layout) K/V caches.

    Matches reshape_and_cache(asm_layout=True):
      K [num_blocks, num_kv_heads, head_dim/x, block_size, x]
      V [num_blocks, num_kv_heads, block_size/x, head_dim, x]
    with x = 16 / cache_itemsize.
    """
    dtype = kv_cache_dtype or case["dtype"]
    itemsize = torch.empty(0, dtype=dtype).element_size()
    x = 16 // itemsize
    nkv = case["num_kv_heads"]
    bs = case["block_size"]
    nb = case["num_blocks"]
    assert HEAD_DIM % x == 0 and bs % x == 0
    kv_cache_k = torch.zeros(nb, nkv, HEAD_DIM // x, bs, x, dtype=dtype, device="cuda")
    kv_cache_v = torch.zeros(nb, nkv, bs // x, HEAD_DIM, x, dtype=dtype, device="cuda")
    return kv_cache_k, kv_cache_v


def make_pertoken_scales(case: dict, *, asm_layout: bool):
    """Allocate per-token dynamic-quant OUTPUT dequant-scale tensors.

    Layout mirrors reshape_and_cache_with_pertoken_quant:
      asm_layout : [num_blocks, num_kv_heads, block_size]
      page-128   : [num_kv_heads, max_kv_tokens]  (max_kv_tokens = num_blocks*block_size)
    """
    nkv = case["num_kv_heads"]
    nb = case["num_blocks"]
    bs = case["block_size"]
    if asm_layout:
        shape = (nb, nkv, bs)
    else:
        shape = (nkv, nb * bs)
    k_scale = torch.zeros(shape, dtype=torch.float32, device="cuda")
    v_scale = torch.zeros(shape, dtype=torch.float32, device="cuda")
    return k_scale, v_scale


def pertoken_scale_at(
    scale: torch.Tensor, *, asm_layout: bool, slot: int, head: int, block_size: int
) -> torch.Tensor:
    """Read one (token-slot, head) scalar from a per-token scale tensor."""
    if asm_layout:
        block, offset = divmod(slot, block_size)
        return scale[block, head, offset]
    return scale[head, slot]


def pertoken_quant_ref(x: torch.Tensor):
    """Per-token (per head-dim row) dynamic fp8 quant reference.

    x: [..., head_dim] float. Returns (dequant, scale) where
       scale = amax/fp8_max (arch fp8 max: 240 for e4m3fnuz on gfx942,
       448 for e4m3fn on gfx950+), dequant = round_to_fp8(x/scale)*scale.
    """
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    # fp8 max is arch-dependent and MUST match the kernel's fp8Max<cache_t>():
    # e4m3fnuz (gfx942) -> 240, e4m3fn (gfx950+) -> 448. Hardcoding 448 mis-scales
    # the e4m3fnuz cache on MI300X.
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    deq = (x.float() / scale).to(fp8_dtype).float() * scale
    return deq, scale.squeeze(-1)


def gather_shuffle_k_row(
    kv_cache_k: torch.Tensor, slot: int, head: int, block_size: int
) -> torch.Tensor:
    """Read one (token-slot, head) K head-dim row from a SHUFFLE K cache
    [num_blocks, num_kv_heads, head_dim/x, block_size, x]."""
    _nb, _nkv, hd_over_x, _bs, x = kv_cache_k.shape
    head_dim = hd_over_x * x
    block, offset = divmod(slot, block_size)
    row = torch.empty(head_dim, dtype=kv_cache_k.dtype, device=kv_cache_k.device)
    for d in range(head_dim):
        row[d] = kv_cache_k[block, head, d // x, offset, d % x]
    return row


def gather_shuffle_v_row(
    kv_cache_v: torch.Tensor, slot: int, head: int, block_size: int
) -> torch.Tensor:
    """Read one (token-slot, head) V head-dim row from a SHUFFLE V cache
    [num_blocks, num_kv_heads, block_size/x, head_dim, x]."""
    _nb, _nkv, _bs_over_x, head_dim, x = kv_cache_v.shape
    block, offset = divmod(slot, block_size)
    row = torch.empty(head_dim, dtype=kv_cache_v.dtype, device=kv_cache_v.device)
    for d in range(head_dim):
        row[d] = kv_cache_v[block, head, offset // x, d, offset % x]
    return row


def check_pertoken_fp8(
    case: dict,
    refs: dict,
    kv_cache: torch.Tensor,
    kv_cache_k: torch.Tensor,
    kv_cache_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    asm_layout: bool,
    msg: str,
):
    """Validate per-token dynamic-quant fp8 K/V caches + emitted dequant scales."""
    block_size = case["block_size"]
    num_kv_heads = case["num_kv_heads"]
    num_tokens = case["qkv"].size(0)

    k_ref = refs["k"]  # [tokens, nkv, hd] post norm+rope
    v_ref = refs["v"]  # [tokens, nkv, hd] raw

    cos_acc = []
    cos_min = 1.0
    for token in range(num_tokens):
        slot = case["slot_mapping"][token].item()
        for head in range(num_kv_heads):
            kref_row = k_ref[token, head].float()
            vref_row = v_ref[token, head].float()
            k_deq_ref, k_scale_ref = pertoken_quant_ref(kref_row)
            v_deq_ref, v_scale_ref = pertoken_quant_ref(vref_row)

            # emitted per-token scales must equal amax/fp8_max
            k_scale_act = pertoken_scale_at(
                k_scale,
                asm_layout=asm_layout,
                slot=slot,
                head=head,
                block_size=block_size,
            )
            v_scale_act = pertoken_scale_at(
                v_scale,
                asm_layout=asm_layout,
                slot=slot,
                head=head,
                block_size=block_size,
            )
            check_close(
                k_scale_act.reshape(1),
                k_scale_ref.reshape(1),
                msg=f"{msg}(k_scale tok{token} h{head})",
                rtol=1e-3,
                atol=1e-3,
            )
            check_close(
                v_scale_act.reshape(1),
                v_scale_ref.reshape(1),
                msg=f"{msg}(v_scale tok{token} h{head})",
                rtol=1e-3,
                atol=1e-3,
            )

            # read cache row + dequant by emitted per-token scale
            if asm_layout:
                k_raw = gather_shuffle_k_row(kv_cache_k, slot, head, block_size)
                v_raw = gather_shuffle_v_row(kv_cache_v, slot, head, block_size)
            else:
                block, offset = divmod(slot, block_size)
                k_raw = kv_cache[block, 0, offset, head]
                v_raw = kv_cache[block, 1, offset, head]
            k_deq_act = maybe_view_fp8(k_raw).float() * k_scale_act
            v_deq_act = maybe_view_fp8(v_raw).float() * v_scale_act

            check_close(
                k_deq_act,
                k_deq_ref,
                msg=f"{msg}(k_cache tok{token} h{head})",
                rtol=0.1,
                atol=0.1,
            )
            check_close(
                v_deq_act,
                v_deq_ref,
                msg=f"{msg}(v_cache tok{token} h{head})",
                rtol=0.1,
                atol=0.1,
            )
            for a, b in ((k_deq_act, kref_row), (v_deq_act, vref_row)):
                cos = torch.nn.functional.cosine_similarity(
                    a.reshape(1, -1), b.reshape(1, -1)
                ).item()
                cos_acc.append(cos)
                cos_min = min(cos_min, cos)
    aiter.logger.info("%s pertoken fp8 min cosine=%.5f", msg, cos_min)
    assert cos_min > 0.99, f"{msg} pertoken cosine too low: {cos_min}"


def gather_cache_outputs(
    case: dict,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor | None,
    *,
    index_slot_mapping: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
):
    index_slots = (
        index_slot_mapping if index_slot_mapping is not None else case["slot_mapping"]
    )
    k_outs = []
    v_outs = []
    index_k_outs = []

    for token in range(case["qkv"].size(0)):
        slot = case["slot_mapping"][token].item()
        block, offset = divmod(slot, case["block_size"])
        k_out = kv_cache[block, 0, offset]
        v_out = kv_cache[block, 1, offset]
        if k_scale is not None and v_scale is not None:
            k_out = maybe_view_fp8(k_out)
            v_out = maybe_view_fp8(v_out)
            k_out = k_out.float() * k_scale
            v_out = v_out.float() * v_scale
        k_outs.append(k_out)
        v_outs.append(v_out)

        if index_cache is not None:
            index_slot = index_slots[token].item()
            index_row = index_cache.view(-1, HEAD_DIM)[index_slot]
            if index_cache.dtype != case["dtype"]:
                index_row = maybe_view_fp8(index_row).float()
            index_k_outs.append(index_row)

    index_k = torch.stack(index_k_outs) if index_k_outs else None
    return torch.stack(k_outs), torch.stack(v_outs), index_k


def gather_index_cache(
    case: dict,
    index_cache: torch.Tensor,
    *,
    index_slot_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather the per-token index_k rows from the page-128-flat index cache."""
    index_slots = (
        index_slot_mapping if index_slot_mapping is not None else case["slot_mapping"]
    )
    rows = []
    flat = index_cache.view(-1, HEAD_DIM)
    for token in range(case["qkv"].size(0)):
        rows.append(maybe_view_fp8(flat[index_slots[token].item()]).float())
    return torch.stack(rows)


def check_close(actual, expected, *, msg: str, rtol: float, atol: float):
    err = checkAllclose(actual.float(), expected.float(), msg=msg, rtol=rtol, atol=atol)
    if err != 0:
        raise AssertionError(f"{msg} mismatch ratio: {err}")


def fp8_cache_dtype() -> torch.dtype | None:
    if dtypes.fp8 is not torch.uint8:
        return dtypes.fp8
    return getattr(torch, "float8_e4m3fnuz", getattr(torch, "float8_e4m3fn", None))


def maybe_view_fp8(x: torch.Tensor) -> torch.Tensor:
    if x.dtype is not torch.uint8:
        return x
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    return x.view(fp8_dtype)


def fp8_cache_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    return (x.float() / scale).to(fp8_dtype).float() * scale


def fp8_unit_scale_ref(x: torch.Tensor) -> torch.Tensor:
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    return x.float().to(fp8_dtype).float()


def run_minimax_tp4_focused_case(
    *, kv_cache_dtype: str, skip_index_branch: bool, supply_index_args: bool
):
    """MiniMax-M3 TP4 page-16 SHUFFLE correctness and non-mutation coverage."""
    use_static_fp8 = kv_cache_dtype == "fp8_e4m3_static"
    cache_dtype = dtypes.fp8 if use_static_fp8 else torch.bfloat16
    case = make_case(
        dtype=torch.bfloat16,
        num_tokens=3,
        block_size=16,
        num_heads=16,
        num_kv_heads=1,
        num_index_heads=1,
        rotary_dim=64,
    )
    case["slot_mapping"] = torch.tensor([0, -1, 17], dtype=torch.int64, device="cuda")
    case["index_slot_mapping"] = torch.tensor(
        [2, -1, 20], dtype=torch.int64, device="cuda"
    )
    qkv_before = case["qkv"].clone()
    refs = make_refs(case, qkv_before)
    q_out = torch.full((3, 16 * HEAD_DIM), 13, dtype=torch.bfloat16, device="cuda")
    index_q_out = torch.full((3, HEAD_DIM), 17, dtype=torch.bfloat16, device="cuda")
    index_cache = torch.full(
        (case["num_blocks"], 16, HEAD_DIM),
        19,
        dtype=torch.bfloat16,
        device="cuda",
    )
    index_q_before = index_q_out.clone()
    index_cache_before = index_cache.clone()
    kv_cache_k, kv_cache_v = make_shuffle_caches(case, kv_cache_dtype=cache_dtype)
    k_scale = (
        torch.tensor([0.25], dtype=torch.float32, device="cuda")
        if use_static_fp8
        else None
    )
    v_scale = (
        torch.tensor([0.5], dtype=torch.float32, device="cuda")
        if use_static_fp8
        else None
    )

    pass_index_args = not skip_index_branch or supply_index_args
    aiter.fused_qknorm_idxrqknorm(
        case["qkv"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        16,
        1,
        64,
        case["eps"],
        case["index_q_norm_weight"] if pass_index_args else None,
        case["index_k_norm_weight"] if pass_index_args else None,
        1,
        case["slot_mapping"],
        kv_cache_k,
        kv_cache_v,
        index_cache if pass_index_args else None,
        16,
        q_out,
        index_q_out if pass_index_args else None,
        case["index_slot_mapping"] if pass_index_args else None,
        kv_cache_dtype=kv_cache_dtype,
        index_cache_dtype="auto",
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=True,
        skip_index_branch=skip_index_branch,
    )

    check_close(q_out, refs["q"], msg="minimax tp4 q", rtol=1e-2, atol=1e-2)
    _, _, _, index_q_slice, index_k_slice = split_qkv(case, case["qkv"])
    _, _, _, index_q_before_slice, index_k_before_slice = split_qkv(case, qkv_before)
    check_close(
        index_q_slice,
        index_q_before_slice,
        msg="minimax tp4 packed index_q unchanged",
        rtol=0,
        atol=0,
    )
    check_close(
        index_k_slice,
        index_k_before_slice,
        msg="minimax tp4 packed index_k unchanged",
        rtol=0,
        atol=0,
    )

    for token, slot_tensor in enumerate(case["slot_mapping"]):
        slot = slot_tensor.item()
        if slot < 0:
            continue
        k_actual = maybe_view_fp8(gather_shuffle_k_row(kv_cache_k, slot, 0, 16)).float()
        v_actual = maybe_view_fp8(gather_shuffle_v_row(kv_cache_v, slot, 0, 16)).float()
        if use_static_fp8:
            k_expected = fp8_cache_ref(refs["k"][token, 0], k_scale)
            v_expected = fp8_cache_ref(refs["v"][token, 0], v_scale)
            k_actual = k_actual * k_scale
            v_actual = v_actual * v_scale
        else:
            k_expected = refs["k"][token, 0]
            v_expected = refs["v"][token, 0]
        check_close(
            k_actual, k_expected, msg=f"minimax tp4 k token {token}", rtol=0.1, atol=0.1
        )
        check_close(
            v_actual, v_expected, msg=f"minimax tp4 v token {token}", rtol=0.1, atol=0.1
        )

    # The padded -1 token must not write the cache location that would otherwise
    # correspond to its token ordinal.
    assert torch.count_nonzero(gather_shuffle_k_row(kv_cache_k, 1, 0, 16)) == 0
    assert torch.count_nonzero(gather_shuffle_v_row(kv_cache_v, 1, 0, 16)) == 0

    if skip_index_branch:
        if supply_index_args:
            assert torch.equal(index_q_out, index_q_before)
            assert torch.equal(index_cache, index_cache_before)
        else:
            # Also instantiate the non-insert skip path: only Q/K launch slots
            # run, while V and both packed index slices remain byte-identical.
            qkv_inplace = qkv_before.clone()
            aiter.fused_qknorm_idxrqknorm(
                qkv_inplace,
                case["q_norm_weight"],
                case["k_norm_weight"],
                case["cos_sin_cache"],
                case["positions"],
                16,
                1,
                64,
                case["eps"],
                num_index_heads=1,
                skip_index_branch=True,
            )
            q_actual, k_actual, v_actual, iq_actual, ik_actual = split_qkv(
                case, qkv_inplace
            )
            _, _, v_before, iq_before, ik_before = split_qkv(case, qkv_before)
            check_close(
                q_actual,
                refs["q"],
                msg="minimax tp4 inplace skip q",
                rtol=1e-2,
                atol=1e-2,
            )
            check_close(
                k_actual.view(3, 1, HEAD_DIM),
                refs["k"],
                msg="minimax tp4 inplace skip k",
                rtol=1e-2,
                atol=1e-2,
            )
            for name, actual, expected in (
                ("v", v_actual, v_before),
                ("index_q", iq_actual, iq_before),
                ("index_k", ik_actual, ik_before),
            ):
                check_close(
                    actual,
                    expected,
                    msg=f"minimax tp4 inplace skip {name} unchanged",
                    rtol=0,
                    atol=0,
                )
    else:
        check_close(
            index_q_out,
            refs["index_q"],
            msg="minimax tp4 index_q",
            rtol=1e-2,
            atol=1e-2,
        )
        for token in (0, 2):
            slot = case["index_slot_mapping"][token].item()
            actual = index_cache.view(-1, HEAD_DIM)[slot]
            check_close(
                actual,
                refs["index_k"][token],
                msg=f"minimax tp4 index_k token {token}",
                rtol=1e-2,
                atol=1e-2,
            )


def run_minimax_tp4_fp8_index_q_case(*, kv_cache_dtype: str, skip_index_branch: bool):
    """MiniMax TP4: unit-scale e4m3 index_q + index cache (4787 q_idx contract)."""
    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    use_static_fp8 = kv_cache_dtype == "fp8_e4m3_static"
    cache_dtype = dtypes.fp8 if use_static_fp8 else torch.bfloat16
    case = make_case(
        dtype=torch.bfloat16,
        num_tokens=3,
        block_size=16,
        num_heads=16,
        num_kv_heads=1,
        num_index_heads=1,
        rotary_dim=64,
    )
    case["slot_mapping"] = torch.tensor([0, -1, 17], dtype=torch.int64, device="cuda")
    case["index_slot_mapping"] = torch.tensor(
        [2, -1, 20], dtype=torch.int64, device="cuda"
    )
    qkv_before = case["qkv"].clone()
    refs = make_refs(case, qkv_before)
    q_out = torch.full((3, 16 * HEAD_DIM), 13, dtype=torch.bfloat16, device="cuda")
    index_q_out = torch.zeros((3, HEAD_DIM), dtype=fp8_dtype, device="cuda")
    index_cache = torch.zeros(
        (case["num_blocks"], 16, HEAD_DIM),
        dtype=fp8_dtype,
        device="cuda",
    )
    index_q_before = index_q_out.clone()
    index_cache_before = index_cache.clone()
    kv_cache_k, kv_cache_v = make_shuffle_caches(case, kv_cache_dtype=cache_dtype)
    k_scale = (
        torch.tensor([0.25], dtype=torch.float32, device="cuda")
        if use_static_fp8
        else None
    )
    v_scale = (
        torch.tensor([0.5], dtype=torch.float32, device="cuda")
        if use_static_fp8
        else None
    )
    tag = (
        f"minimax tp4 fp8-iq {kv_cache_dtype} {'skip' if skip_index_branch else 'full'}"
    )

    aiter.fused_qknorm_idxrqknorm(
        case["qkv"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        16,
        1,
        64,
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        1,
        case["slot_mapping"],
        kv_cache_k,
        kv_cache_v,
        index_cache,
        16,
        q_out,
        index_q_out,
        case["index_slot_mapping"],
        kv_cache_dtype=kv_cache_dtype,
        index_cache_dtype=None,
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=True,
        skip_index_branch=skip_index_branch,
    )

    check_close(q_out, refs["q"], msg=f"{tag} q", rtol=1e-2, atol=1e-2)
    if skip_index_branch:
        assert torch.equal(
            index_q_out.view(torch.uint8), index_q_before.view(torch.uint8)
        )
        assert torch.equal(
            index_cache.view(torch.uint8), index_cache_before.view(torch.uint8)
        )
        return

    check_close(
        maybe_view_fp8(index_q_out).float(),
        fp8_unit_scale_ref(refs["index_q"]),
        msg=f"{tag} index_q",
        rtol=1e-2,
        atol=1e-2,
    )
    for token in (0, 2):
        slot = case["index_slot_mapping"][token].item()
        actual = maybe_view_fp8(index_cache.view(-1, HEAD_DIM)[slot]).float()
        check_close(
            actual,
            fp8_unit_scale_ref(refs["index_k"][token]),
            msg=f"{tag} index_k token {token}",
            rtol=1e-2,
            atol=1e-2,
        )


def run_fp8_index_q_msa_score_decode_case():
    """Fused unit-scale e4m3 IQ/IK must be accepted by 4787's score_decode."""
    from aiter.ops.msa_block_select import pa_sparse_block_score_decode

    fp8_dtype = fp8_cache_dtype()
    assert fp8_dtype is not None
    case = make_case(
        dtype=torch.bfloat16,
        num_tokens=1,
        block_size=128,
        num_heads=16,
        num_kv_heads=1,
        num_index_heads=1,
        rotary_dim=64,
    )
    case["slot_mapping"] = torch.tensor([0], dtype=torch.int64, device="cuda")
    case["index_slot_mapping"] = torch.tensor([0], dtype=torch.int64, device="cuda")
    refs = make_refs(case, case["qkv"].clone())
    q_out, _, kv_cache, _ = make_insert_outputs(case)
    index_q_out = torch.zeros((1, HEAD_DIM), dtype=fp8_dtype, device="cuda")
    index_cache = torch.zeros(
        (case["num_blocks"], 128, HEAD_DIM),
        dtype=fp8_dtype,
        device="cuda",
    )
    aiter.fused_qknorm_idxrqknorm(
        case["qkv"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        16,
        1,
        64,
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        1,
        case["slot_mapping"],
        kv_cache[:, 0],
        kv_cache[:, 1],
        index_cache,
        128,
        q_out,
        index_q_out,
        case["index_slot_mapping"],
        kv_cache_dtype="auto",
        index_cache_dtype=None,
        asm_layout=False,
        skip_index_branch=False,
    )
    q_idx = index_q_out.view(1, 1, HEAD_DIM)
    score = torch.full((1, 1, 64), float("-inf"), device="cuda")
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
    pa_sparse_block_score_decode(
        q_idx,
        index_cache,
        score,
        block_table,
        seq_lens,
        query_len=1,
        max_seq_len=128,
    )
    ref_dot = (q_idx.float()[0, 0] * maybe_view_fp8(index_cache[0, 0]).float()).sum()
    check_close(
        score[0, 0, 0],
        ref_dot,
        msg="fused fp8 index_q into msa score_decode",
        rtol=1e-2,
        atol=1e-2,
    )
    check_close(
        maybe_view_fp8(index_q_out).float(),
        fp8_unit_scale_ref(refs["index_q"]),
        msg="msa cross-check index_q",
        rtol=1e-2,
        atol=1e-2,
    )


def make_packed_lbhnc_shuffle_views(
    *, dtype: torch.dtype, sentinel: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the runtime packed 3-block allocation and offset SHUFFLE views."""
    num_logical_blocks = 3
    logical_block_size = 128
    pages_per_side = logical_block_size // 16
    page_elements = 16 * HEAD_DIM
    packed = torch.full(
        (num_logical_blocks, 2, logical_block_size, HEAD_DIM),
        sentinel,
        dtype=dtype,
        device="cuda",
    )
    x = 16 // packed.element_size()
    num_physical_pages = num_logical_blocks * 2 * pages_per_side
    kv_cache_k = packed.view(num_physical_pages, 1, HEAD_DIM // x, 16, x)
    kv_cache_v = packed.view(num_physical_pages, 1, 16 // x, HEAD_DIM, x)[
        pages_per_side:
    ]
    assert packed.is_contiguous()
    assert kv_cache_k.storage_offset() == packed.storage_offset()
    assert kv_cache_v.storage_offset() == pages_per_side * page_elements
    return packed, kv_cache_k, kv_cache_v


def run_minimax_tp4_packed_lbhnc_case(*, kv_cache_dtype: str, skip_index_branch: bool):
    """Validate runtime-faithful packed LBHNC page spans and rebased slots."""
    use_static_fp8 = kv_cache_dtype == "fp8_e4m3_static"
    cache_dtype = dtypes.fp8 if use_static_fp8 else torch.bfloat16
    cache_sentinel = 24.0
    logical_slots = torch.tensor(
        [0, 15, 16, 127, 128, 129, 255, 256, 383, -1],
        dtype=torch.int64,
        device="cuda",
    )
    rebased_slots = torch.tensor(
        [0, 15, 16, 127, 256, 257, 383, 512, 639, -1],
        dtype=torch.int64,
        device="cuda",
    )
    rebased_ref = (
        logical_slots.clamp_min(0).div(128, rounding_mode="floor").mul(128)
        + logical_slots
    )
    assert torch.equal(rebased_slots, rebased_ref)
    index_slots = torch.tensor(
        [0, 1, 127, 128, 129, 255, 256, 383, 300, -1],
        dtype=torch.int64,
        device="cuda",
    )
    case = make_case(
        dtype=torch.bfloat16,
        num_tokens=logical_slots.numel(),
        block_size=16,
        num_heads=16,
        num_kv_heads=1,
        num_index_heads=1,
        rotary_dim=64,
        seed=20260831,
    )
    case["slot_mapping"] = rebased_slots
    case["index_slot_mapping"] = index_slots
    qkv_before = case["qkv"].clone()
    refs = make_refs(case, qkv_before)

    packed, kv_cache_k, kv_cache_v = make_packed_lbhnc_shuffle_views(
        dtype=cache_dtype, sentinel=cache_sentinel
    )
    x = 16 // packed.element_size()
    assert tuple(kv_cache_k.shape) == (48, 1, HEAD_DIM // x, 16, x)
    assert tuple(kv_cache_v.shape) == (40, 1, 16 // x, HEAD_DIM, x)
    if use_static_fp8:
        assert tuple(kv_cache_k.shape) == (48, 1, 8, 16, 16)
        assert tuple(kv_cache_v.shape) == (40, 1, 1, 128, 16)

    q_out = torch.full(
        (logical_slots.numel(), 16 * HEAD_DIM),
        25,
        dtype=case["dtype"],
        device="cuda",
    )
    index_q_out = torch.full(
        (logical_slots.numel(), HEAD_DIM),
        26,
        dtype=case["dtype"],
        device="cuda",
    )
    index_cache = torch.full((3, 128, HEAD_DIM), 27, dtype=case["dtype"], device="cuda")
    index_q_before = index_q_out.clone()
    index_cache_before = index_cache.clone()
    k_scale = (
        torch.tensor([0.125], dtype=torch.float32, device="cuda")
        if use_static_fp8
        else None
    )
    v_scale = (
        torch.tensor([0.25], dtype=torch.float32, device="cuda")
        if use_static_fp8
        else None
    )

    aiter.fused_qknorm_idxrqknorm(
        case["qkv"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        16,
        1,
        64,
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        1,
        rebased_slots,
        kv_cache_k,
        kv_cache_v,
        index_cache,
        16,
        q_out,
        index_q_out,
        index_slots,
        kv_cache_dtype=kv_cache_dtype,
        index_cache_dtype="auto",
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=True,
        skip_index_branch=skip_index_branch,
    )

    tag = f"packed {kv_cache_dtype} {'skip' if skip_index_branch else 'full'}"
    check_close(q_out, refs["q"], msg=f"{tag} q", rtol=1e-2, atol=1e-2)
    _, _, _, iq_actual, ik_actual = split_qkv(case, case["qkv"])
    _, _, _, iq_before, ik_before = split_qkv(case, qkv_before)
    check_close(
        iq_actual, iq_before, msg=f"{tag} packed index_q unchanged", rtol=0, atol=0
    )
    check_close(
        ik_actual, ik_before, msg=f"{tag} packed index_k unchanged", rtol=0, atol=0
    )

    page_elements = 16 * HEAD_DIM
    packed_flat = packed.view(-1)
    written_mask = torch.zeros(packed.numel(), dtype=torch.bool, device=packed.device)
    for token, slot_tensor in enumerate(rebased_slots):
        slot = slot_tensor.item()
        if slot < 0:
            continue
        page, offset = divmod(slot, 16)
        k_raw = maybe_view_fp8(gather_shuffle_k_row(kv_cache_k, slot, 0, 16)).float()
        v_raw = maybe_view_fp8(gather_shuffle_v_row(kv_cache_v, slot, 0, 16)).float()
        if use_static_fp8:
            k_actual = k_raw * k_scale
            v_actual = v_raw * v_scale
            k_expected = fp8_cache_ref(refs["k"][token, 0], k_scale)
            v_expected = fp8_cache_ref(refs["v"][token, 0], v_scale)
        else:
            k_actual = k_raw
            v_actual = v_raw
            k_expected = refs["k"][token, 0]
            v_expected = refs["v"][token, 0]
        check_close(
            k_actual, k_expected, msg=f"{tag} k token {token}", rtol=0.1, atol=0.1
        )
        check_close(
            v_actual, v_expected, msg=f"{tag} v token {token}", rtol=0.1, atol=0.1
        )

        for dim in range(HEAD_DIM):
            k_abs = page * page_elements + (dim // x) * (16 * x) + offset * x + dim % x
            # V's view starts eight physical pages into the packed allocation.
            v_abs = (
                (8 + page) * page_elements
                + (offset // x) * (HEAD_DIM * x)
                + dim * x
                + offset % x
            )
            written_mask[k_abs] = True
            written_mask[v_abs] = True
            assert maybe_view_fp8(packed_flat[k_abs]).float() == k_raw[dim]
            assert maybe_view_fp8(packed_flat[v_abs]).float() == v_raw[dim]

    untouched = maybe_view_fp8(packed_flat[~written_mask]).float()
    assert torch.all(untouched == cache_sentinel)

    if skip_index_branch:
        assert torch.equal(index_q_out, index_q_before)
        assert torch.equal(index_cache, index_cache_before)
    else:
        check_close(
            index_q_out,
            refs["index_q"],
            msg=f"{tag} index_q",
            rtol=1e-2,
            atol=1e-2,
        )
        written_index_rows = torch.zeros(
            3 * 128, dtype=torch.bool, device=index_cache.device
        )
        for token, index_slot_tensor in enumerate(index_slots):
            index_slot = index_slot_tensor.item()
            if index_slot < 0:
                continue
            written_index_rows[index_slot] = True
            check_close(
                index_cache.view(-1, HEAD_DIM)[index_slot],
                refs["index_k"][token],
                msg=f"{tag} index_k token {token}",
                rtol=1e-2,
                atol=1e-2,
            )
        assert torch.all(index_cache.view(-1, HEAD_DIM)[~written_index_rows] == 27)


def gpu_latency_stats_us(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    """Time an already-allocated/JIT-ready callable with per-iteration events."""
    for iteration in range(warmup):
        fn(iteration)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for iteration, (start, end) in enumerate(zip(starts, ends)):
        start.record()
        fn(warmup + iteration)
        end.record()
    torch.cuda.synchronize()
    samples = sorted(
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    )
    middle = len(samples) // 2
    median = (
        samples[middle]
        if len(samples) % 2
        else (samples[middle - 1] + samples[middle]) / 2.0
    )
    p95 = samples[max(0, math.ceil(0.95 * len(samples)) - 1)]
    return median, p95


def quant_quality_metrics(
    case: dict,
    refs: dict,
    kv_cache_k: torch.Tensor,
    kv_cache_v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    per_token: bool,
) -> list[dict]:
    """Measure dequantized SHUFFLE-cache quality against post-rounding BF16."""
    fp8_max = torch.finfo(fp8_cache_dtype()).max
    actual = {"k": [], "v": []}
    raw = {"k": [], "v": []}
    expected = {"k": [], "v": []}
    for token, slot_tensor in enumerate(slot_mapping):
        slot = slot_tensor.item()
        k_raw = maybe_view_fp8(
            gather_shuffle_k_row(kv_cache_k, slot, 0, case["block_size"])
        ).float()
        v_raw = maybe_view_fp8(
            gather_shuffle_v_row(kv_cache_v, slot, 0, case["block_size"])
        ).float()
        if per_token:
            ks = pertoken_scale_at(
                k_scale,
                asm_layout=True,
                slot=slot,
                head=0,
                block_size=case["block_size"],
            )
            vs = pertoken_scale_at(
                v_scale,
                asm_layout=True,
                slot=slot,
                head=0,
                block_size=case["block_size"],
            )
        else:
            ks, vs = k_scale, v_scale
        raw["k"].append(k_raw)
        raw["v"].append(v_raw)
        actual["k"].append(k_raw * ks)
        actual["v"].append(v_raw * vs)
        expected["k"].append(refs["k"][token, 0].float())
        expected["v"].append(refs["v"][token, 0].float())

    rows = []
    for component in ("k", "v"):
        act = torch.stack(actual[component]).float()
        ref = torch.stack(expected[component]).float()
        raw_values = torch.stack(raw[component]).float()
        error = act - ref
        rows.append(
            {
                "component": component,
                "max_error": error.abs().max().item(),
                "mae": error.abs().mean().item(),
                "rmse": error.square().mean().sqrt().item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    act.flatten().reshape(1, -1),
                    ref.flatten().reshape(1, -1),
                ).item(),
                "saturation_rate": (raw_values.abs() >= fp8_max).float().mean().item(),
            }
        )
    return rows


def make_minimax_benchmark_state(num_tokens: int) -> tuple[dict, dict]:
    """Create one deterministic TP4 source shared by all benchmark contracts."""
    case = make_case(
        dtype=torch.bfloat16,
        num_tokens=num_tokens,
        block_size=16,
        num_heads=16,
        num_kv_heads=1,
        num_index_heads=1,
        rotary_dim=64,
        seed=20260830 + num_tokens,
    )
    case["slot_mapping"] = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    case["index_slot_mapping"] = torch.roll(case["slot_mapping"], shifts=1)
    refs = make_refs(case, case["qkv"])
    return case, refs


def make_benchmark_layout(
    case: dict, *, layout: str, cache_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate one benchmark layout and return its physical page-16 mapping."""
    if layout == "separate":
        kv_cache_k, kv_cache_v = make_shuffle_caches(case, kv_cache_dtype=cache_dtype)
        return kv_cache_k, kv_cache_v, case["slot_mapping"]
    if layout != "packed":
        raise ValueError(f"unknown benchmark layout: {layout}")

    num_tokens = case["qkv"].size(0)
    num_logical_blocks = (num_tokens + 127) // 128
    packed = torch.empty(
        num_logical_blocks,
        2,
        128,
        HEAD_DIM,
        dtype=cache_dtype,
        device="cuda",
    )
    x = 16 // packed.element_size()
    num_physical_pages = num_logical_blocks * 16
    kv_cache_k = packed.view(num_physical_pages, 1, HEAD_DIM // x, 16, x)
    kv_cache_v = packed.view(num_physical_pages, 1, 16 // x, HEAD_DIM, x)[8:]
    logical_slots = case["slot_mapping"]
    rebased_slots = (
        logical_slots.clamp_min(0).div(128, rounding_mode="floor").mul(128)
        + logical_slots
    )
    return kv_cache_k, kv_cache_v, rebased_slots


def benchmark_fused_minimax_case(
    case: dict,
    refs: dict,
    *,
    layout: str,
    cache_mode: str,
    skip_index_branch: bool,
    warmup: int,
    iters: int,
) -> tuple[dict, list[dict]]:
    """Benchmark one preallocated fused full-index or skip-index configuration."""
    use_fp8 = cache_mode != "bf16"
    use_static = cache_mode == "fp8_e4m3_static"
    cache_dtype = dtypes.fp8 if use_fp8 else torch.bfloat16
    kv_cache_k, kv_cache_v, slot_mapping = make_benchmark_layout(
        case, layout=layout, cache_dtype=cache_dtype
    )
    q_out = torch.empty(
        case["qkv"].size(0),
        case["num_heads"] * HEAD_DIM,
        dtype=case["dtype"],
        device="cuda",
    )
    index_q_out = None
    index_cache = None
    if not skip_index_branch:
        index_q_out = torch.empty(
            case["qkv"].size(0), HEAD_DIM, dtype=case["dtype"], device="cuda"
        )
        index_cache = torch.empty(
            (case["qkv"].size(0) + 127) // 128,
            128,
            HEAD_DIM,
            dtype=case["dtype"],
            device="cuda",
        )

    k_scale = v_scale = None
    if use_static:
        # Synthetic calibration, not model scales: reserve 10% FP8 headroom using
        # this exact benchmark input's post-rounding BF16 K/V range.
        fp8_target = 0.9 * torch.finfo(fp8_cache_dtype()).max
        k_scale = (refs["k"].float().abs().max() / fp8_target).reshape(1)
        v_scale = (refs["v"].float().abs().max() / fp8_target).reshape(1)
    elif use_fp8:
        k_scale, v_scale = make_pertoken_scales(case, asm_layout=True)

    def invoke(_iteration: int):
        aiter.fused_qknorm_idxrqknorm(
            case["qkv"],
            case["q_norm_weight"],
            case["k_norm_weight"],
            case["cos_sin_cache"],
            case["positions"],
            16,
            1,
            64,
            case["eps"],
            None if skip_index_branch else case["index_q_norm_weight"],
            None if skip_index_branch else case["index_k_norm_weight"],
            1,
            slot_mapping,
            kv_cache_k,
            kv_cache_v,
            index_cache,
            16,
            q_out,
            index_q_out,
            None if skip_index_branch else case["index_slot_mapping"],
            kv_cache_dtype="auto" if cache_mode == "bf16" else cache_mode,
            index_cache_dtype="auto",
            k_scale=k_scale,
            v_scale=v_scale,
            asm_layout=True,
            skip_index_branch=skip_index_branch,
        )

    median_us, p95_us = gpu_latency_stats_us(invoke, warmup=warmup, iters=iters)
    row = {
        "implementation": "fused",
        "layout": layout,
        "index_mode": "skip" if skip_index_branch else "full",
        "cache_mode": cache_mode,
        "num_tokens": case["qkv"].size(0),
        "median_us": median_us,
        "p95_us": p95_us,
        "tokens_per_second": case["qkv"].size(0) * 1e6 / median_us,
        "synthetic_k_scale": k_scale.item() if use_static else None,
        "synthetic_v_scale": v_scale.item() if use_static else None,
    }
    quality = []
    if use_fp8:
        quality = quant_quality_metrics(
            case,
            refs,
            kv_cache_k,
            kv_cache_v,
            k_scale,
            v_scale,
            slot_mapping,
            per_token=not use_static,
        )
        for metrics in quality:
            metrics.update(
                {
                    "index_mode": row["index_mode"],
                    "layout": layout,
                    "cache_mode": cache_mode,
                    "num_tokens": row["num_tokens"],
                }
            )
    return row, quality


def benchmark_vllm_unfused_case(
    case: dict,
    refs: dict,
    *,
    layout: str,
    cache_mode: str,
    skip_index_branch: bool,
    warmup: int,
    iters: int,
) -> tuple[dict | None, str | None]:
    """Benchmark vLLM csrc norm/RoPE + AITER SHUFFLE insert + index scatter."""
    if cache_mode == "fp8_e4m3":
        return (
            None,
            (
                "vLLM's current sparse-PA fallback has fixed scalar FP8 scales, "
                "not per-token output scales"
            ),
        )
    try:
        from vllm import _custom_ops as vllm_ops
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_insert_index_cache,
        )
    except Exception as error:  # noqa: BLE001
        return None, f"vLLM fallback imports unavailable: {error}"
    if not hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"):
        return None, "vLLM csrc fused MiniMax norm/RoPE op is unavailable"

    use_static = cache_mode == "fp8_e4m3_static"
    cache_dtype = dtypes.fp8 if use_static else torch.bfloat16
    kv_cache_k, kv_cache_v, slot_mapping = make_benchmark_layout(
        case, layout=layout, cache_dtype=cache_dtype
    )
    q_out = torch.empty(
        case["qkv"].size(0),
        case["num_heads"] * HEAD_DIM,
        dtype=case["dtype"],
        device="cuda",
    )
    index_q_out = (
        None
        if skip_index_branch
        else torch.empty(
            case["qkv"].size(0), HEAD_DIM, dtype=case["dtype"], device="cuda"
        )
    )
    index_cache = (
        None
        if skip_index_branch
        else torch.empty(
            (case["qkv"].size(0) + 127) // 128,
            128,
            HEAD_DIM,
            dtype=case["dtype"],
            device="cuda",
        )
    )
    fp8_target = 0.9 * torch.finfo(fp8_cache_dtype()).max
    k_scale = (
        (refs["k"].float().abs().max() / fp8_target).reshape(1) if use_static else None
    )
    v_scale = (
        (refs["v"].float().abs().max() / fp8_target).reshape(1) if use_static else None
    )

    total_calls = warmup + iters
    qkv_work = [case["qkv"].clone() for _ in range(total_calls)]
    k_stage = torch.empty_like(refs["k"])
    v_stage = torch.empty_like(refs["v"])
    index_k_stage = (
        None
        if skip_index_branch
        else torch.empty(
            case["qkv"].size(0), HEAD_DIM, dtype=case["dtype"], device="cuda"
        )
    )
    q_size, kv_size, _, iq_size, _ = case["sizes"]

    def invoke(iteration: int):
        qkv = qkv_work[iteration]
        vllm_ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            case["q_norm_weight"],
            case["k_norm_weight"],
            case["cos_sin_cache"],
            case["positions"],
            16,
            1,
            64,
            case["eps"],
            None if skip_index_branch else case["index_q_norm_weight"],
            None if skip_index_branch else case["index_k_norm_weight"],
            1,
            q_out=q_out,
            index_q_out=index_q_out,
            skip_index_branch=skip_index_branch,
        )
        k_stage.copy_(qkv[:, q_size : q_size + kv_size].view_as(k_stage))
        v_stage.copy_(qkv[:, q_size + kv_size : q_size + 2 * kv_size].view_as(v_stage))
        aiter.reshape_and_cache(
            k_stage,
            v_stage,
            kv_cache_k,
            kv_cache_v,
            slot_mapping,
            "fp8_e4m3" if use_static else "auto",
            k_scale=k_scale,
            v_scale=v_scale,
            asm_layout=True,
        )
        if not skip_index_branch:
            index_k_begin = q_size + 2 * kv_size + iq_size
            index_k_stage.copy_(qkv[:, index_k_begin : index_k_begin + HEAD_DIM])
            minimax_m3_insert_index_cache(
                index_k_stage, index_cache, case["index_slot_mapping"]
            )

    try:
        median_us, p95_us = gpu_latency_stats_us(invoke, warmup=warmup, iters=iters)
    except Exception as error:  # noqa: BLE001
        return None, f"vLLM fallback execution failed: {error}"
    return (
        {
            "implementation": "vllm_unfused",
            "layout": layout,
            "index_mode": "skip" if skip_index_branch else "full",
            "cache_mode": cache_mode,
            "num_tokens": case["qkv"].size(0),
            "median_us": median_us,
            "p95_us": p95_us,
            "tokens_per_second": case["qkv"].size(0) * 1e6 / median_us,
            "synthetic_k_scale": k_scale.item() if use_static else None,
            "synthetic_v_scale": v_scale.item() if use_static else None,
        },
        None,
    )


def run_gluon_sensitivity(case: dict, refs: dict) -> tuple[list[dict], str | None]:
    """Compare static/per-token FP8 Gluon outputs with the BF16-cache output."""
    try:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import _run_gluon_decode
    except Exception as error:  # noqa: BLE001
        return [], f"Gluon reader import unavailable: {error}"

    built = {}
    for cache_mode in ("bf16", "fp8_e4m3_static", "fp8_e4m3"):
        use_fp8 = cache_mode != "bf16"
        use_static = cache_mode == "fp8_e4m3_static"
        cache_dtype = dtypes.fp8 if use_fp8 else torch.bfloat16
        kv_cache_k, kv_cache_v = make_shuffle_caches(case, kv_cache_dtype=cache_dtype)
        q_out = torch.empty(
            case["qkv"].size(0),
            case["num_heads"] * HEAD_DIM,
            dtype=case["dtype"],
            device="cuda",
        )
        k_scale = v_scale = None
        if use_static:
            fp8_target = 0.9 * torch.finfo(fp8_cache_dtype()).max
            k_scale = (refs["k"].float().abs().max() / fp8_target).reshape(1)
            v_scale = (refs["v"].float().abs().max() / fp8_target).reshape(1)
        elif use_fp8:
            k_scale, v_scale = make_pertoken_scales(case, asm_layout=True)

        aiter.fused_qknorm_idxrqknorm(
            case["qkv"],
            case["q_norm_weight"],
            case["k_norm_weight"],
            case["cos_sin_cache"],
            case["positions"],
            16,
            1,
            64,
            case["eps"],
            num_index_heads=1,
            slot_mapping=case["slot_mapping"],
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            block_size=16,
            q_out=q_out,
            kv_cache_dtype="auto" if cache_mode == "bf16" else cache_mode,
            index_cache_dtype="auto",
            k_scale=k_scale,
            v_scale=v_scale,
            asm_layout=True,
            skip_index_branch=True,
        )
        if cache_mode == "fp8_e4m3":
            # _run_gluon_decode accepts per-token scales as
            # [num_kv_heads, physical_pages * 16]. The consolidated writer emits
            # [physical_pages, num_kv_heads, 16] for SHUFFLE caches.
            k_scale = k_scale.permute(1, 0, 2).reshape(1, -1).contiguous()
            v_scale = v_scale.permute(1, 0, 2).reshape(1, -1).contiguous()
        built[cache_mode] = (kv_cache_k, kv_cache_v, q_out, k_scale, v_scale)

    num_tokens = case["qkv"].size(0)
    num_written_pages = (num_tokens + case["block_size"] - 1) // case["block_size"]
    # Benchmark slots are exactly [0, num_tokens), so the reader's physical
    # page-16 table is consecutive and its context is the number of written rows.
    sparse_bt = torch.arange(
        num_written_pages, dtype=torch.int32, device="cuda"
    ).reshape(1, -1)
    sparse_ctx = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    query = built["bf16"][2][-1:].view(1, 16, HEAD_DIM).contiguous()
    outputs = {
        cache_mode: torch.empty_like(query)
        for cache_mode in ("bf16", "fp8_e4m3_static", "fp8_e4m3")
    }

    try:
        for cache_mode, (k_cache, v_cache, _q_out, k_scale, v_scale) in built.items():
            _run_gluon_decode(
                query,
                k_cache,
                v_cache,
                sparse_bt,
                sparse_ctx,
                1,
                HEAD_DIM**-0.5,
                outputs[cache_mode],
                k_scale,
                v_scale,
            )
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        return [], f"Gluon sensitivity execution failed: {error}"

    reference = outputs["bf16"].float()
    rows = []
    for cache_mode in ("fp8_e4m3_static", "fp8_e4m3"):
        actual = outputs[cache_mode].float()
        error = actual - reference
        rows.append(
            {
                "num_tokens": num_tokens,
                "cache_mode": cache_mode,
                "max_error": error.abs().max().item(),
                "mae": error.abs().mean().item(),
                "rmse": error.square().mean().sqrt().item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    actual.flatten().reshape(1, -1),
                    reference.flatten().reshape(1, -1),
                ).item(),
            }
        )
    return rows, None


def run_minimax_benchmark(
    *, token_counts: list[int], warmup: int, iters: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    # Exercise the exact three-logical-block packed geometry, physical offsets,
    # padded slots, and full/skip index behavior once before collecting timings.
    for packed_kv_dtype in ("auto", "fp8_e4m3_static"):
        for packed_skip_index in (False, True):
            run_minimax_tp4_packed_lbhnc_case(
                kv_cache_dtype=packed_kv_dtype,
                skip_index_branch=packed_skip_index,
            )

    timing_rows = []
    quality_rows = []
    sensitivity_rows = []
    blockers = [
        "sibling #4813 specialized baseline unavailable: branch switching is prohibited"
    ]
    for num_tokens in token_counts:
        case, refs = make_minimax_benchmark_state(num_tokens)
        fused_by_key = {}
        for layout in ("separate", "packed"):
            cache_modes = (
                ("bf16", "fp8_e4m3_static", "fp8_e4m3")
                if layout == "separate"
                else ("bf16", "fp8_e4m3_static")
            )
            for skip_index_branch in (False, True):
                for cache_mode in cache_modes:
                    row, quality = benchmark_fused_minimax_case(
                        case,
                        refs,
                        layout=layout,
                        cache_mode=cache_mode,
                        skip_index_branch=skip_index_branch,
                        warmup=warmup,
                        iters=iters,
                    )
                    timing_rows.append(row)
                    quality_rows.extend(quality)
                    fused_by_key[(layout, row["index_mode"], cache_mode)] = row

                    baseline, blocker = benchmark_vllm_unfused_case(
                        case,
                        refs,
                        layout=layout,
                        cache_mode=cache_mode,
                        skip_index_branch=skip_index_branch,
                        warmup=warmup,
                        iters=iters,
                    )
                    if baseline is not None:
                        row["speedup_vs_unfused"] = (
                            baseline["median_us"] / row["median_us"]
                        )
                        timing_rows.append(baseline)
                    elif blocker is not None:
                        message = (
                            f"tokens={num_tokens}, layout={layout}, "
                            f"index={row['index_mode']}, cache={cache_mode}: {blocker}"
                        )
                        if message not in blockers:
                            blockers.append(message)

        for index_mode in ("full", "skip"):
            static_row = fused_by_key[("separate", index_mode, "fp8_e4m3_static")]
            pertoken_row = fused_by_key[("separate", index_mode, "fp8_e4m3")]
            static_row["fixed_over_pertoken_ratio"] = (
                static_row["median_us"] / pertoken_row["median_us"]
            )
            for cache_mode in ("bf16", "fp8_e4m3_static"):
                separate_row = fused_by_key[("separate", index_mode, cache_mode)]
                packed_row = fused_by_key[("packed", index_mode, cache_mode)]
                packed_row["packed_over_separate_ratio"] = (
                    packed_row["median_us"] / separate_row["median_us"]
                )

        sensitivity, blocker = run_gluon_sensitivity(case, refs)
        sensitivity_rows.extend(sensitivity)
        if blocker is not None:
            blockers.append(f"tokens={num_tokens}: {blocker}")

    return (
        pd.DataFrame(timing_rows),
        pd.DataFrame(quality_rows),
        pd.DataFrame(sensitivity_rows),
        blockers,
    )


@perftest(num_iters=10, num_warmup=1)
def run_fused_qknorm_idxrqknorm(
    case: dict,
    mode: str,
    use_index_slot_mapping: bool,
    use_fp8_kv_cache: bool,
):
    qkv = case["qkv"].clone()
    if mode == "inplace":
        aiter.fused_qknorm_idxrqknorm(
            qkv,
            case["q_norm_weight"],
            case["k_norm_weight"],
            case["cos_sin_cache"],
            case["positions"],
            case["num_heads"],
            case["num_kv_heads"],
            case["rotary_dim"],
            case["eps"],
            case["index_q_norm_weight"],
            case["index_k_norm_weight"],
            case["num_index_heads"],
        )
        return qkv

    use_asm_layout = mode.startswith("asm_layout")
    use_uint8_kv_cache = mode.endswith("_uint8")
    use_unit_scale_kv_cache = mode == "fp8_kv_cache_unit"
    kv_cache_dtype = None
    if use_fp8_kv_cache:
        kv_cache_dtype = torch.uint8 if use_uint8_kv_cache else dtypes.fp8
    use_fp8_index_cache = (
        use_fp8_kv_cache and not use_unit_scale_kv_cache
    ) or mode == "asm_layout_fp8_index"
    q_out, index_q_out, kv_cache, index_cache = make_insert_outputs(
        case,
        kv_cache_dtype=kv_cache_dtype,
        index_cache_dtype=dtypes.fp8 if use_fp8_index_cache else None,
    )
    if use_asm_layout:
        # SHUFFLE caches (separate K/V) for the page-16 asm layout.
        kv_cache_k, kv_cache_v = make_shuffle_caches(
            case, kv_cache_dtype=kv_cache_dtype
        )
    else:
        # page-128: the op takes separate K/V caches -> use the key/value slices
        # of the fused [nb, 2, bs, nkv, hd] tensor (views, so gather still works).
        kv_cache_k = kv_cache[:, 0]
        kv_cache_v = kv_cache[:, 1]
    index_slot_mapping = case["index_slot_mapping"] if use_index_slot_mapping else None
    index_q_out_arg = index_q_out
    index_cache_arg = index_cache
    if case["num_index_heads"] == 0:
        index_q_out_arg = None
        index_cache_arg = None
        index_slot_mapping = None
    k_scale = None
    v_scale = None
    use_pertoken = "pertoken" in mode
    kv_cache_dtype_arg = "auto"
    if use_fp8_kv_cache:
        if use_unit_scale_kv_cache:
            # SGLang MiniMax-M3 contract: direct BF16 -> FP8 cast, no scale buffer.
            kv_cache_dtype_arg = "fp8_e4m3_unit"
        elif use_pertoken:
            # Per-token dynamic quant: k_scale/v_scale are OUTPUT tensors the op fills.
            k_scale, v_scale = make_pertoken_scales(case, asm_layout=use_asm_layout)
            kv_cache_dtype_arg = "fp8_e4m3"
        else:
            raise AssertionError(f"unknown fp8 kv-cache contract for mode={mode}")
    index_cache_dtype_arg = "fp8" if use_fp8_index_cache else "auto"

    aiter.fused_qknorm_idxrqknorm(
        qkv,
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        case["num_heads"],
        case["num_kv_heads"],
        case["rotary_dim"],
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        case["num_index_heads"],
        case["slot_mapping"],
        kv_cache_k,
        kv_cache_v,
        index_cache_arg,
        case["block_size"],
        q_out,
        index_q_out_arg,
        index_slot_mapping,
        kv_cache_dtype=kv_cache_dtype_arg,
        index_cache_dtype=index_cache_dtype_arg,
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=use_asm_layout,
    )
    return (
        q_out,
        index_q_out_arg,
        kv_cache,
        index_cache_arg,
        index_slot_mapping,
        k_scale,
        v_scale,
        kv_cache_k,
        kv_cache_v,
    )


@benchmark()
def test_fused_qknorm_idxrqknorm(
    mode: str,
    dtype: torch.dtype,
    num_tokens: int,
    block_size: int,
    rotary_dim: int,
    num_index_heads: int = 4,
):
    use_fp8_kv_cache = mode.startswith("fp8_kv_cache") or (
        mode.startswith("asm_layout_fp8") and mode != "asm_layout_fp8_index"
    )
    use_unit_scale_kv_cache = mode == "fp8_kv_cache_unit"
    use_fp8_index_cache = (
        use_fp8_kv_cache and not use_unit_scale_kv_cache
    ) or mode == "asm_layout_fp8_index"
    if use_fp8_kv_cache and fp8_cache_dtype() is None:
        aiter.logger.info("Skip fp8_kv_cache: torch FP8 dtype is unavailable")
        return {
            "dtype": str(dtype),
            "num_tokens": num_tokens,
            "block_size": block_size,
            "rotary_dim": rotary_dim,
            "num_index_heads": num_index_heads,
            "status": "skipped",
        }

    case = make_case(
        dtype=dtype,
        num_tokens=num_tokens,
        block_size=block_size,
        num_index_heads=num_index_heads,
        rotary_dim=rotary_dim,
    )
    refs = make_refs(case, case["qkv"])
    rtol = 1e-2
    atol = 1e-2
    use_index_slot_mapping = mode != "slot_mapping_fallback"
    result, avg_opt = run_fused_qknorm_idxrqknorm(
        case,
        mode,
        use_index_slot_mapping,
        use_fp8_kv_cache,
    )

    info = (
        f"mode:{mode}, dtype:{dtype}, tokens:{num_tokens}, block:{block_size}, "
        f"rotary:{rotary_dim}, index_heads:{num_index_heads}"
    )
    msg = f"[perf] === {info} === fused_kernel avg: {avg_opt:<8.2f} us "

    if mode == "inplace":
        qkv_out = result
        q_out, k_out, v_out, *index_outs = split_qkv(case, qkv_out)
        _, _, v_orig, *_ = split_qkv(case, case["qkv"])
        check_close(q_out, refs["q"], msg=f"{msg}(q)", rtol=rtol, atol=atol)
        check_close(
            k_out.view(num_tokens, case["num_kv_heads"], HEAD_DIM),
            refs["k"],
            msg=f"{msg}(k)",
            rtol=rtol,
            atol=atol,
        )
        check_close(v_out, v_orig, msg=f"{msg}(v)", rtol=0, atol=0)
        if num_index_heads > 0:
            index_q_out, index_k_out = index_outs
            check_close(
                index_q_out,
                refs["index_q"],
                msg=f"{msg}(index_q)",
                rtol=rtol,
                atol=atol,
            )
            check_close(
                index_k_out,
                refs["index_k"],
                msg=f"{msg}(index_k)",
                rtol=rtol,
                atol=atol,
            )
    else:
        (
            q_out,
            index_q_out,
            kv_cache,
            index_cache,
            index_slot_mapping,
            k_scale,
            v_scale,
            kv_cache_k,
            kv_cache_v,
        ) = result
        check_close(q_out, refs["q"], msg=f"{msg}(q_out)", rtol=rtol, atol=atol)
        if num_index_heads > 0:
            check_close(
                index_q_out,
                refs["index_q"],
                msg=f"{msg}(index_q_out)",
                rtol=rtol,
                atol=atol,
            )

        if "pertoken" in mode:
            check_pertoken_fp8(
                case,
                refs,
                kv_cache,
                kv_cache_k,
                kv_cache_v,
                k_scale,
                v_scale,
                asm_layout=mode.startswith("asm_layout"),
                msg=msg,
            )
            if num_index_heads > 0:
                if mode.startswith("asm_layout"):
                    index_k_out = gather_index_cache(
                        case, index_cache, index_slot_mapping=index_slot_mapping
                    )
                else:
                    _, _, index_k_out = gather_cache_outputs(
                        case,
                        kv_cache,
                        index_cache,
                        index_slot_mapping=index_slot_mapping,
                    )
                check_close(
                    index_k_out,
                    fp8_unit_scale_ref(refs["index_k"]),
                    msg=f"{msg}(index_cache)",
                    rtol=rtol,
                    atol=atol,
                )
        elif mode.startswith("asm_layout"):
            # Ground truth: write the SAME normed/roped K and raw V into freshly
            # zeroed SHUFFLE caches via the PROVEN reshape_and_cache(asm_layout=True)
            # writer, then compare the fused-op caches against it element-wise. This
            # directly validates the new SHUFFLE layout offsets.
            ref_k_cache = torch.zeros_like(kv_cache_k)
            ref_v_cache = torch.zeros_like(kv_cache_v)
            kv_dtype_arg = "fp8_e4m3" if use_fp8_kv_cache else "auto"
            aiter.reshape_and_cache(
                refs["k"].contiguous(),
                refs["v"].contiguous(),
                ref_k_cache,
                ref_v_cache,
                case["slot_mapping"],
                kv_dtype_arg,
                k_scale=k_scale,
                v_scale=v_scale,
                asm_layout=True,
            )
            act_k = maybe_view_fp8(kv_cache_k).float()
            act_v = maybe_view_fp8(kv_cache_v).float()
            ref_k = maybe_view_fp8(ref_k_cache).float()
            ref_v = maybe_view_fp8(ref_v_cache).float()
            check_close(act_k, ref_k, msg=f"{msg}(k_shuffle)", rtol=rtol, atol=atol)
            check_close(act_v, ref_v, msg=f"{msg}(v_shuffle)", rtol=rtol, atol=atol)
            if num_index_heads > 0:
                index_k_out = gather_index_cache(
                    case, index_cache, index_slot_mapping=index_slot_mapping
                )
                index_k_ref = (
                    fp8_unit_scale_ref(refs["index_k"])
                    if use_fp8_index_cache
                    else refs["index_k"]
                )
                check_close(
                    index_k_out,
                    index_k_ref,
                    msg=f"{msg}(index_cache)",
                    rtol=rtol,
                    atol=atol,
                )
        else:
            k_out, v_out, index_k_out = gather_cache_outputs(
                case,
                kv_cache,
                index_cache,
                index_slot_mapping=index_slot_mapping,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            if use_unit_scale_kv_cache:
                k_ref = fp8_unit_scale_ref(refs["k"])
                v_ref = fp8_unit_scale_ref(refs["v"])
            elif use_fp8_kv_cache:
                k_ref = fp8_cache_ref(refs["k"], k_scale)
                v_ref = fp8_cache_ref(refs["v"], v_scale)
            else:
                k_ref = refs["k"]
                v_ref = refs["v"]
            check_close(k_out, k_ref, msg=f"{msg}(k_cache)", rtol=rtol, atol=atol)
            check_close(v_out, v_ref, msg=f"{msg}(v_cache)", rtol=rtol, atol=atol)
            if num_index_heads > 0:
                check_close(
                    index_k_out,
                    refs["index_k"],
                    msg=f"{msg}(index_cache)",
                    rtol=rtol,
                    atol=atol,
                )

    return {
        "mode": mode,
        "dtype": str(dtype),
        "num_tokens": num_tokens,
        "block_size": block_size,
        "rotary_dim": rotary_dim,
        "num_index_heads": num_index_heads,
        "fused_kernel_us": avg_opt,
        "status": "passed",
    }


DEFAULT_CASES = [
    ("insert", "bf16", 1, 16, 64, 4),
    ("insert", "bf16", 17, 16, 64, 4),
    ("insert", "bf16", 19, 16, 96, 4),
    ("insert", "fp16", 33, 8, 128, 4),
    ("dense_insert", "bf16", 13, 8, 96, 0),
    ("slot_mapping_fallback", "bf16", 9, 8, 64, 4),
    ("inplace", "bf16", 11, 16, 64, 0),
    ("inplace", "bf16", 11, 16, 64, 4),
    ("inplace", "fp16", 11, 16, 64, 4),
    ("asm_layout", "bf16", 17, 16, 64, 4),
    ("asm_layout", "fp16", 19, 16, 96, 4),
    ("asm_layout", "bf16", 13, 16, 64, 0),
    ("asm_layout_fp8_index", "bf16", 17, 16, 64, 4),
    # SGLang contract: page-1 NHD, unit-scale main FP8, BF16 index cache.
    ("fp8_kv_cache_unit", "bf16", 1, 1, 64, 1),
    ("fp8_kv_cache_unit", "bf16", 16, 1, 64, 1),
    ("fp8_kv_cache_unit", "bf16", 64, 1, 64, 1),
    # ATOM contract: per-token main FP8 scales and FP8 index cache.
    ("fp8_kv_cache_pertoken", "bf16", 17, 16, 64, 4),
    ("fp8_kv_cache_pertoken_uint8", "bf16", 17, 16, 64, 4),
    ("asm_layout_fp8_pertoken", "bf16", 17, 16, 64, 4),
    ("asm_layout_fp8_pertoken", "fp16", 19, 16, 96, 4),
    ("asm_layout_fp8_pertoken_uint8", "bf16", 17, 16, 64, 4),
]

l_mode = [
    "insert",
    "dense_insert",
    "slot_mapping_fallback",
    "inplace",
    "asm_layout",
    "asm_layout_fp8_index",
    "fp8_kv_cache_unit",
    "fp8_kv_cache_pertoken",
    "fp8_kv_cache_pertoken_uint8",
    "asm_layout_fp8_pertoken",
    "asm_layout_fp8_pertoken_uint8",
]
l_dtype = ["fp16", "bf16"]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test fused_qknorm_idxrqknorm op",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=l_mode,
    nargs="*",
    default=None,
    help="Mode(s) to test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="*",
    default=None,
    help="Data type(s). e.g. -d bf16 or -d bf16 fp16",
)
parser.add_argument("--num_tokens", type=int, nargs="*", default=None)
parser.add_argument("--block_size", type=int, nargs="*", default=None)
parser.add_argument("--rotary_dim", type=int, nargs="*", default=None)
parser.add_argument("--num_index_heads", type=int, nargs="*", default=None)
parser.add_argument(
    "--minimax_tp4",
    action="store_true",
    help="Run focused MiniMax-M3 TP4 full/skip BF16/static-FP8 cases",
)
parser.add_argument(
    "--minimax_benchmark",
    action="store_true",
    help="Benchmark TP4 full/skip BF16, static-FP8, and per-token-FP8 paths",
)
parser.add_argument(
    "--benchmark_tokens",
    type=int,
    nargs="*",
    default=[1, 16, 64, 128, 256],
)
parser.add_argument("--benchmark_warmup", type=int, default=10)
parser.add_argument("--benchmark_iters", type=int, default=50)
args = parser.parse_args()

selected_cases = []
for (
    mode,
    dtype_name,
    num_tokens,
    block_size,
    rotary_dim,
    num_index_heads,
) in DEFAULT_CASES:
    if args.minimax_tp4 or args.minimax_benchmark:
        continue
    if args.mode is not None and mode not in args.mode:
        continue
    if args.dtype is not None and dtype_name not in args.dtype:
        continue
    if args.num_tokens is not None and num_tokens not in args.num_tokens:
        continue
    if args.block_size is not None and block_size not in args.block_size:
        continue
    if args.rotary_dim is not None and rotary_dim not in args.rotary_dim:
        continue
    if args.num_index_heads is not None and num_index_heads not in args.num_index_heads:
        continue
    selected_cases.append(
        (mode, dtype_name, num_tokens, block_size, rotary_dim, num_index_heads)
    )

if args.minimax_tp4:
    if fp8_cache_dtype() is None:
        raise RuntimeError("MiniMax TP4 focused cases require an FP8 dtype")
    run_minimax_tp4_focused_case(
        kv_cache_dtype="auto",
        skip_index_branch=False,
        supply_index_args=True,
    )
    run_minimax_tp4_focused_case(
        kv_cache_dtype="auto",
        skip_index_branch=True,
        supply_index_args=False,
    )
    run_minimax_tp4_focused_case(
        kv_cache_dtype="fp8_e4m3_static",
        skip_index_branch=False,
        supply_index_args=True,
    )
    run_minimax_tp4_focused_case(
        kv_cache_dtype="fp8_e4m3_static",
        skip_index_branch=True,
        supply_index_args=True,
    )
    for packed_kv_dtype in ("auto", "fp8_e4m3_static"):
        for packed_skip_index in (False, True):
            run_minimax_tp4_packed_lbhnc_case(
                kv_cache_dtype=packed_kv_dtype,
                skip_index_branch=packed_skip_index,
            )
    for fp8_iq_kv_dtype in ("auto", "fp8_e4m3_static"):
        for fp8_iq_skip in (False, True):
            run_minimax_tp4_fp8_index_q_case(
                kv_cache_dtype=fp8_iq_kv_dtype,
                skip_index_branch=fp8_iq_skip,
            )
    run_fp8_index_q_msa_score_decode_case()

benchmark_timing = benchmark_quality = benchmark_sensitivity = None
benchmark_blockers = []
if args.minimax_benchmark:
    if fp8_cache_dtype() is None:
        raise RuntimeError("MiniMax benchmark requires an FP8 dtype")
    if (
        not args.benchmark_tokens
        or min(args.benchmark_tokens) <= 0
        or args.benchmark_warmup < 1
        or args.benchmark_iters < 1
    ):
        raise ValueError("benchmark tokens/warmup/iters must all be positive")
    (
        benchmark_timing,
        benchmark_quality,
        benchmark_sensitivity,
        benchmark_blockers,
    ) = run_minimax_benchmark(
        token_counts=args.benchmark_tokens,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
    )

df = []
for (
    mode,
    dtype_name,
    num_tokens,
    block_size,
    rotary_dim,
    num_index_heads,
) in selected_cases:
    ret = test_fused_qknorm_idxrqknorm(
        mode=mode,
        dtype=dtypes.d_dtypes[dtype_name],
        num_tokens=num_tokens,
        block_size=block_size,
        rotary_dim=rotary_dim,
        num_index_heads=num_index_heads,
    )
    df.append(ret)

df = pd.DataFrame(df)
if args.minimax_tp4:
    aiter.logger.info("focused MiniMax-M3 TP4 cases passed")
elif args.minimax_benchmark:
    aiter.logger.info(
        "MiniMax benchmark scale note: model artifacts are unavailable; "
        "fp8_e4m3_static uses synthetic non-unit scales "
        "amax(post-rounding BF16 reference) / (0.9 * fp8_max)"
    )
    aiter.logger.info(
        "MiniMax benchmark timing CSV:\n%s",
        benchmark_timing.to_csv(index=False),
    )
    aiter.logger.info(
        "MiniMax benchmark quantization-quality CSV:\n%s",
        benchmark_quality.to_csv(index=False),
    )
    aiter.logger.info(
        "MiniMax benchmark Gluon sensitivity CSV:\n%s",
        benchmark_sensitivity.to_csv(index=False),
    )
    for blocker in benchmark_blockers:
        aiter.logger.info("MiniMax benchmark unavailable baseline: %s", blocker)
else:
    df_md = df.to_markdown(index=False)
    aiter.logger.info("fused_qknorm_idxrqknorm summary (markdown):\n%s", df_md)
