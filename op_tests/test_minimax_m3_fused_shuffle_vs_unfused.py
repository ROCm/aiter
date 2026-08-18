# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end equivalence: the fused MiniMax-M3 shuffle insert must reproduce
the three-kernel path vLLM's AMD sparse layer runs today.

Unfused path (3 kernels):
  1. vLLM  fused_minimax_m3_qknorm_rope_kv_insert (norm + RoPE only)
  2. aiter reshape_and_cache(..., asm_layout=True)
  3. vLLM  minimax_m3_insert_index_cache (Triton)

Fused path (1 kernel):
  aiter minimax_m3_qknorm_rope_cache_shuffle_insert

Also checks that pa_decode_gluon reads the fused cache to the same output as
the unfused one, i.e. the layout is what the decode kernel expects.
"""

import pytest
import torch

pytest.importorskip("vllm")

from aiter import (
    minimax_m3_qknorm_rope_cache_shuffle_insert,
    reshape_and_cache,
)
from aiter.utility.dtypes import get_dtype_fp8

HEAD_DIM = 128
PAGE_SIZE = 16
PAGES_PER_BLOCK = 8  # a 128-token vLLM block is 8 physical page-16 pages


def _vllm_ops():
    import vllm._custom_ops as ops
    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        minimax_m3_insert_index_cache,
    )

    return ops, minimax_m3_insert_index_cache


def build_case(num_tokens, nq, nkv, niq, rotary_dim, cache_dtype, idx_dtype, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    dtype = torch.bfloat16
    row = (nq + 2 * nkv + niq + 1) * HEAD_DIM
    qkv = torch.randn(num_tokens, row, device=dev, dtype=dtype)

    w = {
        n: torch.randn(HEAD_DIM, device=dev, dtype=dtype) * 0.1
        for n in ("q", "k", "iq", "ik")
    }
    max_pos = 8192
    cos_sin = torch.randn(max_pos, rotary_dim, device=dev, dtype=dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device=dev, dtype=torch.int64)

    num_blocks = 4
    num_pages = num_blocks * PAGES_PER_BLOCK
    x = 16 // torch.empty((), dtype=cache_dtype).element_size()

    def caches():
        k = torch.zeros(
            num_pages, nkv, HEAD_DIM // x, PAGE_SIZE, x, device=dev, dtype=cache_dtype
        )
        v = torch.zeros(
            num_pages, nkv, PAGE_SIZE // x, HEAD_DIM, x, device=dev, dtype=cache_dtype
        )
        idx = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM, device=dev, dtype=idx_dtype)
        return k, v, idx

    # vLLM slot_mapping is a global token index; page = slot // 16 works because
    # a logical 128-token block's 8 page-16 pages are contiguous.
    slots = torch.randperm(num_pages * PAGE_SIZE, device=dev)[:num_tokens]
    slot_mapping = slots.to(torch.int64)
    idx_slots = torch.randperm(num_pages * PAGE_SIZE, device=dev)[:num_tokens]
    index_slot_mapping = idx_slots.to(torch.int64)
    if num_tokens >= 4:
        slot_mapping[1] = -1  # padded token

    return {
        "qkv": qkv,
        "w": w,
        "cos_sin": cos_sin,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "index_slot_mapping": index_slot_mapping,
        "caches": caches,
        "x": x,
        "dtype": dtype,
        "num_tokens": num_tokens,
    }


def run_unfused(c, nq, nkv, niq, rotary_dim, eps, kv_cache_dtype, k_scale, v_scale):
    ops, insert_index_cache = _vllm_ops()
    qkv = c["qkv"].clone()
    n = c["num_tokens"]
    q = qkv.new_empty((n, nq * HEAD_DIM))
    k_cache, v_cache, index_cache = c["caches"]()
    index_q = torch.empty(
        (n, niq * HEAD_DIM), device=qkv.device, dtype=index_cache.dtype
    )

    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        c["w"]["q"],
        c["w"]["k"],
        c["cos_sin"],
        c["positions"],
        nq,
        nkv,
        rotary_dim,
        eps,
        c["w"]["iq"],
        c["w"]["ik"],
        niq,
        q_out=q,
        index_q_out=index_q,
        kv_cache_dtype=kv_cache_dtype,
    )

    k_start = nq * HEAD_DIM
    v_start = k_start + nkv * HEAD_DIM
    ik_start = v_start + nkv * HEAD_DIM + niq * HEAD_DIM
    k = qkv[:, k_start:v_start].view(n, nkv, HEAD_DIM)
    v = qkv[:, v_start : v_start + nkv * HEAD_DIM].view(n, nkv, HEAD_DIM)
    index_k = qkv[:, ik_start : ik_start + HEAD_DIM].view(n, HEAD_DIM)

    reshape_and_cache(
        k.contiguous(),
        v.contiguous(),
        k_cache,
        v_cache,
        c["slot_mapping"],
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=True,
    )
    insert_index_cache(index_k, index_cache, c["index_slot_mapping"])
    torch.cuda.synchronize()
    return q, index_q, k_cache, v_cache, index_cache


def run_fused(c, nq, nkv, niq, rotary_dim, eps, kv_cache_dtype, k_scale, v_scale):
    qkv = c["qkv"].clone()
    n = c["num_tokens"]
    q = qkv.new_empty((n, nq * HEAD_DIM))
    k_cache, v_cache, index_cache = c["caches"]()
    index_q = torch.empty(
        (n, niq * HEAD_DIM), device=qkv.device, dtype=index_cache.dtype
    )

    minimax_m3_qknorm_rope_cache_shuffle_insert(
        qkv,
        c["w"]["q"],
        c["w"]["k"],
        c["cos_sin"],
        c["positions"],
        nq,
        nkv,
        niq,
        rotary_dim,
        eps,
        c["slot_mapping"],
        k_cache,
        v_cache,
        q,
        index_q_norm_weight=c["w"]["iq"],
        index_k_norm_weight=c["w"]["ik"],
        index_slot_mapping=c["index_slot_mapping"],
        index_cache=index_cache,
        index_q_out=index_q,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    return q, index_q, k_cache, v_cache, index_cache


def assert_same(name, a, b, exact_frac=0.995):
    fa, fb = a.float(), b.float()
    torch.testing.assert_close(
        fa, fb, rtol=8e-3, atol=8e-3, msg=lambda m: f"{name}: {m}"
    )
    frac = (fa == fb).float().mean().item()
    assert frac >= exact_frac, f"{name}: only {frac:.3%} bit-identical"


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fp8_index", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 5, 33])
def test_fused_matches_unfused(quantized, fp8_index, num_tokens):
    nq, nkv, niq, rotary_dim, eps = 16, 1, 1, 64, 1e-6
    _check_matches_unfused(
        nq, nkv, niq, rotary_dim, eps, quantized, fp8_index, num_tokens
    )


@pytest.mark.parametrize("nkv", [2, 4])
def test_fused_matches_unfused_multi_kv_head(nkv):
    """Both writers must agree on page-major/head-minor placement.

    The AITER sparse-PA prototype only runs num_kv_heads == 1 per rank today, so
    this pins the layout contract that lifting that restriction depends on.
    """
    _check_matches_unfused(8 * nkv, nkv, 1, 64, 1e-6, True, True, 40)


def _check_matches_unfused(
    nq, nkv, niq, rotary_dim, eps, quantized, fp8_index, num_tokens
):
    fp8 = get_dtype_fp8()
    cache_dtype = fp8 if quantized else torch.bfloat16
    idx_dtype = fp8 if fp8_index else torch.bfloat16
    kv_cache_dtype = "fp8" if quantized else "auto"
    k_scale = torch.tensor([0.8], device="cuda") if quantized else None
    v_scale = torch.tensor([1.25], device="cuda") if quantized else None

    c = build_case(num_tokens, nq, nkv, niq, rotary_dim, cache_dtype, idx_dtype)
    args = (nq, nkv, niq, rotary_dim, eps, kv_cache_dtype, k_scale, v_scale)
    ref = run_unfused(c, *args)
    got = run_fused(c, *args)

    for name, a, b in zip(
        ("q_out", "index_q_out", "k_cache", "v_cache", "index_cache"), ref, got
    ):
        assert_same(name, a, b)


@pytest.mark.parametrize("nkv", [2, 4])
def test_gluon_head_folding_matches_per_head_decode(nkv):
    """Reader-side check for num_kv_heads > 1.

    ``_run_gluon_decode`` folds (page, kv_head) into gluon's page dim with the
    head minor, so a folded run with page ids ``page * nkv + head`` must equal
    running each head separately against its own slice of the cache. This is the
    other half of the layout contract needed to lift the one-KV-head limit.
    """
    sparse_pa = pytest.importorskip("vllm.models.minimax_m3.amd.ops.sparse_pa")

    group, num_tokens, pages_used = 8, 6, 3
    nq = group * nkv
    fp8 = get_dtype_fp8()
    c = build_case(num_tokens, nq, nkv, 1, 64, fp8, fp8, seed=11)
    k_scale = torch.tensor([1.0], device="cuda")
    v_scale = torch.tensor([1.0], device="cuda")
    q, _, k_cache, v_cache, _ = run_fused(
        c, nq, nkv, 1, 64, 1e-6, "fp8", k_scale, v_scale
    )
    q = q.view(num_tokens, nq, HEAD_DIM)

    num_pages = k_cache.shape[0]
    ctx = pages_used * PAGE_SIZE
    # Logical page choice per token, shared by every kv head (index_k is one head).
    pages = (
        torch.arange(num_tokens * pages_used, device="cuda", dtype=torch.int32)
        % num_pages
    ).view(num_tokens, pages_used)
    ctx_lens = torch.full((num_tokens,), ctx, device="cuda", dtype=torch.int32)

    # Folded: one row per (token, kv head), page ids scaled by nkv.
    folded_bt = pages.repeat_interleave(nkv, dim=0) * nkv + torch.arange(
        nkv, device="cuda", dtype=torch.int32
    ).repeat(num_tokens).unsqueeze(1)
    folded_out = torch.empty(
        num_tokens, nq, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    sparse_pa._run_gluon_decode(
        q,
        k_cache,
        v_cache,
        folded_bt,
        ctx_lens.repeat_interleave(nkv),
        nkv,
        HEAD_DIM**-0.5,
        folded_out,
        k_scale,
        v_scale,
    )
    torch.cuda.synchronize()

    for h in range(nkv):
        per_head_out = torch.empty(
            num_tokens, group, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        sparse_pa._run_gluon_decode(
            q[:, h * group : (h + 1) * group].contiguous(),
            k_cache[:, h : h + 1].contiguous(),
            v_cache[:, h : h + 1].contiguous(),
            pages,
            ctx_lens,
            1,
            HEAD_DIM**-0.5,
            per_head_out,
            k_scale,
            v_scale,
        )
        torch.cuda.synchronize()
        assert per_head_out.abs().sum().item() > 0, "per-head decode produced zeros"
        assert torch.equal(
            folded_out[:, h * group : (h + 1) * group], per_head_out
        ), f"folded gluon decode differs from per-head decode for kv head {h}"


@pytest.mark.parametrize("quantized", [False, True])
def test_fused_cache_feeds_gluon_decode_identically(quantized):
    """The fused cache must drive the production gluon decode to the same output."""
    sparse_pa = pytest.importorskip("vllm.models.minimax_m3.amd.ops.sparse_pa")

    nq, nkv, niq, rotary_dim, eps = 16, 1, 1, 64, 1e-6
    num_tokens = 8
    fp8 = get_dtype_fp8()
    cache_dtype = fp8 if quantized else torch.bfloat16
    k_scale = torch.tensor([1.0], device="cuda") if quantized else None
    v_scale = torch.tensor([1.0], device="cuda") if quantized else None
    c = build_case(num_tokens, nq, nkv, niq, rotary_dim, cache_dtype, fp8, seed=3)

    args = (
        nq,
        nkv,
        niq,
        rotary_dim,
        eps,
        "fp8" if quantized else "auto",
        k_scale,
        v_scale,
    )
    q_ref, _, k_ref, v_ref, _ = run_unfused(c, *args)
    q_got, _, k_got, v_got, _ = run_fused(c, *args)

    # Every token attends to a fixed window of physical pages, so the decode
    # touches slots both paths wrote.
    num_pages = k_ref.shape[0]
    pages_used = 4
    ctx = pages_used * PAGE_SIZE
    sparse_bt = (
        torch.arange(num_tokens * pages_used, device="cuda", dtype=torch.int32)
        % num_pages
    ).view(num_tokens, pages_used)
    sparse_ctx = torch.full((num_tokens,), ctx, device="cuda", dtype=torch.int32)

    def decode(q, k_cache, v_cache):
        out = torch.empty(num_tokens, nq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        sparse_pa._run_gluon_decode(
            q.view(num_tokens, nq, HEAD_DIM),
            k_cache,
            v_cache,
            sparse_bt,
            sparse_ctx,
            nkv,
            HEAD_DIM**-0.5,
            out,
            k_scale,
            v_scale,
        )
        torch.cuda.synchronize()
        return out

    out_ref = decode(q_ref, k_ref, v_ref)
    out_got = decode(q_got, k_got, v_got)
    assert out_ref.abs().sum().item() > 0, "decode produced all zeros"
    assert torch.equal(out_ref, out_got), "gluon decode differs on the fused cache"
