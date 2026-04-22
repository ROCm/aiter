# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for mha_v3 APIs: flash_attn_with_kvcache (including paged), FP8 forward/backward,
torch.compile, and CUDA graph capture.

Inlines what was previously in test_flash_attn_kvcache.py.
"""

import math
import pytest
import torch
from einops import rearrange, repeat

from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS

arch = get_arch()
_supports_fp8 = arch in FP8_ARCHS

SEED = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_cosine_similarity(actual, expected, threshold=0.96, norm_floor=1e-3):
    a = actual.float().flatten()
    b = expected.float().flatten()
    if b.norm().item() > norm_floor:
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        assert cos_sim >= threshold, f"Cosine similarity {cos_sim:.6f} < {threshold}"


def fp8_assert_close(tensor_a, tensor_b, atol=1.0, cos_sim_threshold=0.96):
    a = tensor_a.float().flatten()
    b = tensor_b.float().flatten()
    max_abs = (a - b).abs().max().item()
    assert max_abs <= atol, f"Max absolute error {max_abs:.4f} > {atol}"
    assert_cosine_similarity(tensor_a, tensor_b, cos_sim_threshold)


def _attention_ref_with_tol(q, k, v, do, is_fp8=False, **kwargs):
    """Run attention reference and compute adaptive tolerances."""
    has_dropout = kwargs.get("dropout_p", 0.0) > 0.0

    def _run_ref(upcast, reorder_ops=False):
        q_ = q.detach().clone().requires_grad_(True)
        k_ = k.detach().clone().requires_grad_(True)
        v_ = v.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out, _, _ = attention_ref(
                q_, k_, v_, upcast=upcast, reorder_ops=reorder_ops, **kwargs
            )
        dq, dk, dv = torch.autograd.grad(out, (q_, k_, v_), do)
        return out, dq, dk, dv

    def _tol(ref_val, pt_val, is_forward=False):
        baseline = (pt_val - ref_val).abs().max().item()
        if is_fp8:
            mult = 4
            atol_floor = 5e-1 if is_forward else 1.0
            rtol_floor = 1e-1
        elif has_dropout:
            mult = 2
            atol_floor = 1e-1 if is_forward else 2.0
            rtol_floor = 1e-1
        else:
            mult = 2
            atol_floor = 1e-2 if is_forward else 1.5e-2
            rtol_floor = 1e-5
        atol = max(mult * baseline, atol_floor)
        return atol, rtol_floor

    out, dq, dk, dv = _run_ref(upcast=True)
    out_pt, dq_pt, dk_pt, dv_pt = _run_ref(upcast=False, reorder_ops=True)

    fwd_tol = _tol(out, out_pt, is_forward=True)
    bwd_tols = [_tol(dq, dq_pt), _tol(dk, dk_pt), _tol(dv, dv_pt)]

    return out, (dq, dk, dv), fwd_tol, bwd_tols


# ---------------------------------------------------------------------------
# KV cache helpers (from test_flash_attn_kvcache.py)
# ---------------------------------------------------------------------------


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def kvcache_attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def _generate_block_kvcache(
    seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks


# ===========================================================================
# FP8 forward tests
# ===========================================================================


@pytest.mark.parametrize("BATCH", [1, 30, 64])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (8192, 8192)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 8), (48, 8)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_fp8(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    dtype=torch.float16,
):
    if not _supports_fp8:
        pytest.skip(f"FP8 not supported on {arch}")

    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    triton_out = flash_attn_fp8_func(q, k, v, causal=CAUSAL)

    torch_out, _, _ = attention_ref(q, k, v, causal=CAUSAL)

    fp8_assert_close(triton_out, torch_out.to(triton_out.dtype))


@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (8192, 8192)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_varlen_fp8(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    dtype=torch.float16,
):
    if not _supports_fp8:
        pytest.skip(f"FP8 not supported on {arch}")

    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    triton_out = flash_attn_varlen_fp8_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=CAUSAL,
    )
    triton_out = output_pad_fn(triton_out)

    torch_out, _, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )

    fp8_assert_close(triton_out, torch_out.to(triton_out.dtype))


# ===========================================================================
# FP8 backward tests (extracted from test_mha.py)
# ===========================================================================


@pytest.mark.parametrize("SEQLEN_Q", [512, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_backward_fp8(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    CAUSAL: bool,
    dtype=torch.float16,
):
    BATCH = 3
    NUM_K_HEADS = 8
    HEAD_SZ = 128
    if not _supports_fp8:
        pytest.skip(f"FP8 not supported on {arch}")

    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    do = torch.randn_like(q)

    with torch.enable_grad():
        triton_out = flash_attn_fp8_func(q, k, v, causal=CAUSAL)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q, k, v), do.clone()
    )

    torch_out, torch_grads, fwd_tol, bwd_tols = _attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=True,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)
        assert_cosine_similarity(tri, ref)


@pytest.mark.parametrize("SEQLEN_Q", [512, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_backward_varlen_fp8(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    CAUSAL: bool,
    dtype=torch.float16,
):
    BATCH = 3
    NUM_K_HEADS = 8
    HEAD_SZ = 128
    if not _supports_fp8:
        pytest.skip(f"FP8 not supported on {arch}")

    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    do = torch.randn_like(q)

    with torch.enable_grad():
        triton_out = flash_attn_varlen_fp8_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=CAUSAL,
        )
    triton_out = output_pad_fn(triton_out)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q_unpad, k_unpad, v_unpad), do.clone()
    )
    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    torch_out, torch_grads, fwd_tol, bwd_tols = _attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=True,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)
        assert_cosine_similarity(tri, ref)


# ===========================================================================
# KV cache correctness tests (from test_flash_attn_kvcache.py)
# ===========================================================================


@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("paged_kv_block_size", [None, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 339),
        (3, 1024),
        (3, 799),
        (64, 2048),
        (128, 128),
        (8, 3131),
        (1, 1024),
    ],
)
@pytest.mark.parametrize("d", [64, 128])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    paged_kv_block_size,
    seqlen_new_eq_seqlen_q,
    causal,
    new_kv,
    mha_type,
):
    dtype = torch.bfloat16
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()

    device = "cuda"
    torch.random.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    batch_size = 2
    nheads = 6
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    seqlen_new = (
        seqlen_q
        if seqlen_new_eq_seqlen_q
        else torch.randint(1, seqlen_q + 1, (1,)).item()
    )
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None

    if paged_kv_block_size is None:
        k_cache = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype
        )
        v_cache = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype
        )
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )

    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        (seqlen_k - seqlen_new + 1) if new_kv else (seqlen_k + 1),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )

    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange,
            arange < cache_seqlens_expanded + seqlen_new,
        )
        k_cache_ref[update_mask] = rearrange(k, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")

    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)

    out = flash_attn_with_kvcache(
        q,
        k_cache if paged_kv_block_size is None else k_cache_paged,
        v_cache if paged_kv_block_size is None else v_cache_paged,
        k,
        v,
        cache_seqlens=cache_seqlens,
        page_table=block_table,
        causal=causal,
    )
    torch.cuda.synchronize()

    if isinstance(out, tuple):
        out = out[0]
    out = out.to(dtype)

    out_ref, _ = kvcache_attention_ref(
        q,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=(-1, -1),
    )
    out_pt, _ = kvcache_attention_ref(
        q,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        None,
        0.0,
        None,
        causal=causal,
        window_size=(-1, -1),
        upcast=False,
        reorder_ops=True,
    )

    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = k_cache
            v_cache_select = v_cache
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(
            k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3
        ), "k_cache was not updated correctly"
        assert torch.equal(
            v_cache_select, v_cache_ref
        ), "v_cache was not updated correctly"

    pt_max_diff = (out_pt - out_ref).abs().max().item()
    our_max_diff = (out - out_ref).abs().max().item()
    mult = 3
    assert our_max_diff <= mult * pt_max_diff + 1e-5, (
        f"Output max diff {our_max_diff:.6e} exceeds "
        f"{mult}x Pytorch baseline diff {pt_max_diff:.6e} + 1e-5"
    )


# ===========================================================================
# torch.compile tests (from test_flash_attn_kvcache.py)
# ===========================================================================


@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_flash_attn_kvcache_torch_compile(mha_type, causal, new_kv):
    d = 128
    device = "cuda"
    torch.random.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    batch_size = 2
    seqlen_q = 1
    seqlen_k = 1024
    nheads = 6
    nheads_k = nheads if mha_type == "mha" else 3
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    cache_seqlens = torch.randint(
        1, seqlen_k + 1, (batch_size,), dtype=torch.int32, device=device
    )

    if new_kv:
        k_new = torch.randn(
            batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype
        )
        v_new = torch.randn(
            batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype
        )
    else:
        k_new, v_new = None, None

    def fn(q, k_cache, v_cache, k_new, v_new, cache_seqlens):
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            causal=causal,
        )

    k_cache_eager = k_cache.clone()
    v_cache_eager = v_cache.clone()
    out_eager = fn(q, k_cache_eager, v_cache_eager, k_new, v_new, cache_seqlens.clone())
    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]
    torch.cuda.synchronize()

    compiled_fn = torch.compile(fn)
    k_cache_compiled = k_cache.clone()
    v_cache_compiled = v_cache.clone()
    out_compiled = compiled_fn(
        q, k_cache_compiled, v_cache_compiled, k_new, v_new, cache_seqlens.clone()
    )
    if isinstance(out_compiled, tuple):
        out_compiled = out_compiled[0]
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    diff = (out_eager - out_compiled).abs().max().item()
    assert diff < 1e-5, f"torch.compile vs eager max diff {diff:.6e} exceeds 1e-5"


# ===========================================================================
# CUDA graph capture tests
# ===========================================================================


@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_flash_attn_kvcache_cudagraph(mha_type, new_kv):
    """CUDA graph capture with non-paged KV cache."""
    d = 128
    device = "cuda"
    torch.random.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    batch_size = 2
    seqlen_q = 1
    seqlen_k = 4096
    initial_cache_len = 128
    nheads = 8
    nheads_k = nheads if mha_type == "mha" else 2
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    cache_seqlens = torch.full(
        (batch_size,), initial_cache_len, dtype=torch.int32, device=device
    )

    if new_kv:
        k_new = torch.randn(
            batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype
        )
        v_new = torch.randn(
            batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype
        )
    else:
        k_new, v_new = None, None

    q_orig = q.clone()
    k_cache_orig = k_cache.clone()
    v_cache_orig = v_cache.clone()
    k_new_orig = k_new.clone() if k_new is not None else None
    v_new_orig = v_new.clone() if v_new is not None else None

    # warmup (Triton JIT)
    for _ in range(3):
        _ = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
    torch.cuda.synchronize()

    # reset
    q.copy_(q_orig)
    k_cache.copy_(k_cache_orig)
    v_cache.copy_(v_cache_orig)
    cache_seqlens.fill_(initial_cache_len)
    if k_new is not None:
        k_new.copy_(k_new_orig)
        v_new.copy_(v_new_orig)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
        )

    # reset before replay
    q.copy_(q_orig)
    k_cache.copy_(k_cache_orig)
    v_cache.copy_(v_cache_orig)
    cache_seqlens.fill_(initial_cache_len)
    if k_new is not None:
        k_new.copy_(k_new_orig)
        v_new.copy_(v_new_orig)

    g.replay()
    torch.cuda.synchronize()

    if isinstance(out_graph, tuple):
        out_graph = out_graph[0]

    out_eager = flash_attn_with_kvcache(
        q_orig.clone(),
        k_cache_orig.clone(),
        v_cache_orig.clone(),
        k=k_new_orig,
        v=v_new_orig,
        cache_seqlens=torch.full(
            (batch_size,), initial_cache_len, dtype=torch.int32, device=device
        ),
        causal=True,
    )
    torch.cuda.synchronize()
    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]

    assert not torch.isnan(out_graph).any(), "Graph replay 1 produced NaN"
    diff1 = (out_eager - out_graph).abs().max().item()
    assert diff1 < 1e-5, f"Graph replay 1 vs eager max diff {diff1:.6e} exceeds 1e-5"

    # second replay with new data
    q_new_data = torch.randn_like(q)
    q.copy_(q_new_data)
    k_cache.copy_(k_cache_orig)
    v_cache.copy_(v_cache_orig)
    new_cache_len = 256
    cache_seqlens.fill_(new_cache_len)
    if k_new is not None:
        k_new_data = torch.randn_like(k_new)
        v_new_data = torch.randn_like(v_new)
        k_new.copy_(k_new_data)
        v_new.copy_(v_new_data)
    else:
        k_new_data, v_new_data = None, None

    g.replay()
    torch.cuda.synchronize()
    out_graph_2 = out_graph.clone()

    out_eager_2 = flash_attn_with_kvcache(
        q_new_data,
        k_cache_orig.clone(),
        v_cache_orig.clone(),
        k=k_new_data,
        v=v_new_data,
        cache_seqlens=torch.full(
            (batch_size,), new_cache_len, dtype=torch.int32, device=device
        ),
        causal=True,
    )
    torch.cuda.synchronize()
    if isinstance(out_eager_2, tuple):
        out_eager_2 = out_eager_2[0]

    assert not torch.isnan(out_graph_2).any(), "Graph replay 2 produced NaN"
    diff2 = (out_eager_2 - out_graph_2).abs().max().item()
    assert diff2 < 1e-5, f"Graph replay 2 vs eager max diff {diff2:.6e} exceeds 1e-5"


@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_flash_attn_kvcache_paged_cudagraph(mha_type):
    """CUDA graph capture with paged KV cache (block_table + max_seqlen_k)."""
    d = 128
    device = "cuda"
    torch.random.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    batch_size = 2
    seqlen_q = 1
    num_blocks_per_seq = 32
    block_size = 16
    max_cache_len = num_blocks_per_seq * block_size - 3  # 509
    nheads = 8
    nheads_k = nheads if mha_type == "mha" else 2
    dtype = torch.bfloat16

    num_blocks = batch_size * num_blocks_per_seq
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_blocks, block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .view(batch_size, num_blocks_per_seq)
        .contiguous()
    )
    cache_seqlens = torch.full(
        (batch_size,), max_cache_len, device=device, dtype=torch.int32
    )

    q_orig = q.clone()

    # warmup (Triton JIT)
    for _ in range(3):
        _ = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=d ** (-0.5),
            causal=False,
            page_table=block_table,
            max_seqlen_k=max_cache_len,
        )
    torch.cuda.synchronize()

    q.copy_(q_orig)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=d ** (-0.5),
            causal=False,
            page_table=block_table,
            max_seqlen_k=max_cache_len,
        )

    q.copy_(q_orig)
    g.replay()
    torch.cuda.synchronize()

    if isinstance(out_graph, tuple):
        out_graph = out_graph[0]

    # eager reference
    out_eager = flash_attn_with_kvcache(
        q_orig.clone(),
        k_cache.clone(),
        v_cache.clone(),
        cache_seqlens=cache_seqlens.clone(),
        softmax_scale=d ** (-0.5),
        causal=False,
        page_table=block_table.clone(),
        max_seqlen_k=max_cache_len,
    )
    torch.cuda.synchronize()
    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]

    assert not torch.isnan(out_graph).any(), "Paged graph replay produced NaN"
    diff = (out_eager - out_graph).abs().max().item()
    assert diff < 1e-5, f"Paged graph replay vs eager max diff {diff:.6e} exceeds 1e-5"


@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_flash_attn_func_cudagraph(mha_type):
    """CUDA graph capture for flash_attn_func (basic forward)."""
    d = 128
    device = "cuda"
    torch.manual_seed(SEED)
    batch_size = 2
    seqlen = 128
    nheads = 8
    nheads_k = nheads if mha_type == "mha" else 2
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype)

    q_orig = q.clone()
    k_orig = k.clone()
    v_orig = v.clone()

    # warmup
    for _ in range(3):
        _ = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    q.copy_(q_orig)
    k.copy_(k_orig)
    v.copy_(v_orig)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph = flash_attn_func(q, k, v, causal=True)

    q.copy_(q_orig)
    k.copy_(k_orig)
    v.copy_(v_orig)
    g.replay()
    torch.cuda.synchronize()

    out_eager = flash_attn_func(
        q_orig.clone(), k_orig.clone(), v_orig.clone(), causal=True
    )
    torch.cuda.synchronize()

    assert not torch.isnan(out_graph).any(), "Graph replay produced NaN"
    diff = (out_eager - out_graph).abs().max().item()
    assert diff < 1e-5, f"Graph replay vs eager max diff {diff:.6e} exceeds 1e-5"


@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_flash_attn_varlen_func_cudagraph(mha_type):
    """CUDA graph capture for flash_attn_varlen_func."""
    d = 128
    device = "cuda"
    torch.manual_seed(SEED)
    batch_size = 2
    seqlen = 128
    nheads = 8
    nheads_k = nheads if mha_type == "mha" else 2
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype)
    query_padding_mask = generate_random_padding_mask(
        seqlen, batch_size, device, mode="full"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen, batch_size, device, mode="full"
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    q_orig = q_unpad.clone()
    k_orig = k_unpad.clone()
    v_orig = v_unpad.clone()

    # warmup
    for _ in range(3):
        _ = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
        )
    torch.cuda.synchronize()

    q_unpad.copy_(q_orig)
    k_unpad.copy_(k_orig)
    v_unpad.copy_(v_orig)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
        )

    q_unpad.copy_(q_orig)
    k_unpad.copy_(k_orig)
    v_unpad.copy_(v_orig)
    g.replay()
    torch.cuda.synchronize()

    out_eager = flash_attn_varlen_func(
        q_orig.clone(),
        k_orig.clone(),
        v_orig.clone(),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=True,
    )
    torch.cuda.synchronize()

    if isinstance(out_graph, tuple):
        out_graph = out_graph[0]
    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]

    assert not torch.isnan(out_graph).any(), "Graph replay produced NaN"
    diff = (out_eager - out_graph).abs().max().item()
    assert diff < 1e-5, f"Graph replay vs eager max diff {diff:.6e} exceeds 1e-5"
