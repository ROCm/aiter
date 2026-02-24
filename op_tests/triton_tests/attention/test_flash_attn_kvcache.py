# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import pytest
import torch
from einops import rearrange, repeat

from aiter.ops.triton.attention.mha_v3 import flash_attn_with_kvcache

SEED = 0


# adopted from
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/test_flash_attn_triton_amd.py#L628-L959

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
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


def attention_ref(
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
    """
    q: (batch_size, seqlen_q, nheads, head_dim)
    k: (batch_size, seqlen_k, nheads_k, head_dim)
    v: (batch_size, seqlen_k, nheads_k, head_dim)
    """
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
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
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
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
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
    """Create a paged KV cache with a random block table, returning both the
    paged tensors and the equivalent dense view for reference comparison."""
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


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False])
@pytest.mark.parametrize("paged_kv_block_size", [None, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (3, 799),
        (64, 2048),
        (128, 128),
        (1, 3131),
        (8, 3131),
        (1, 1024),
        (3, 799),
        (1, 339),
    ],
)
@pytest.mark.parametrize("d", [64, 128, 256])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    paged_kv_block_size,
    seqlen_new_eq_seqlen_q,
    causal,
    new_kv,
    mha_type,
    dtype,
):
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
        k_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
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
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
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

    out_ref, _ = attention_ref(
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

    out_pt, _ = attention_ref(
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
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3), \
            "k_cache was not updated correctly"
        assert torch.equal(v_cache_select, v_cache_ref), \
            "v_cache was not updated correctly"

    pt_max_diff = (out_pt - out_ref).abs().max().item()
    our_max_diff = (out - out_ref).abs().max().item()
    mult = 3
    assert our_max_diff <= mult * pt_max_diff + 1e-5, (
        f"Output max diff {our_max_diff:.6e} exceeds "
        f"{mult}x Pytorch baseline diff {pt_max_diff:.6e} + 1e-5"
    )



# torch.compile tests
@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("d", [128])
def test_flash_attn_kvcache_torch_compile(
    d,
    mha_type,
    causal,
    new_kv,
):
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
    cache_seqlens = torch.randint(1, seqlen_k + 1, (batch_size,), dtype=torch.int32, device=device)

    if new_kv:
        k_new = torch.randn(batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype)
        v_new = torch.randn(batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype)
    else:
        k_new, v_new = None, None

    def fn(q, k_cache, v_cache, k_new, v_new, cache_seqlens):
        return flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k_new, v=v_new,
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


# Manual graph capture tests

@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("d", [128])
def test_flash_attn_kvcache_hipgraph_capture(d, mha_type, new_kv):
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
        k_new = torch.randn(batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype)
        v_new = torch.randn(batch_size, seqlen_q, nheads_k, d, device=device, dtype=dtype)
    else:
        k_new, v_new = None, None

    q_orig = q.clone()
    k_cache_orig = k_cache.clone()
    v_cache_orig = v_cache.clone()
    k_new_orig = k_new.clone() if k_new is not None else None
    v_new_orig = v_new.clone() if v_new is not None else None

    # warmup (Triton JIT compiles kernels on first invocation)
    for _ in range(3):
        _ = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k_new, v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
    torch.cuda.synchronize()

    # reset buffers
    q.copy_(q_orig)
    k_cache.copy_(k_cache_orig)
    v_cache.copy_(v_cache_orig)
    cache_seqlens.fill_(initial_cache_len)
    if k_new is not None:
        k_new.copy_(k_new_orig)
        v_new.copy_(v_new_orig)

    # capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k_new, v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
        )

    # reset again before first replay so state is identical to eager
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

    q_eager = q_orig.clone()
    k_cache_eager = k_cache_orig.clone()
    v_cache_eager = v_cache_orig.clone()
    cache_seqlens_eager = torch.full(
        (batch_size,), initial_cache_len, dtype=torch.int32, device=device
    )
    out_eager = flash_attn_with_kvcache(
        q_eager, k_cache_eager, v_cache_eager,
        k=k_new_orig, v=v_new_orig,
        cache_seqlens=cache_seqlens_eager,
        causal=True,
    )
    torch.cuda.synchronize()

    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]

    assert not torch.isnan(out_graph).any(), "HIP graph replay 1 produced NaN"
    diff1 = (out_eager - out_graph).abs().max().item()
    assert diff1 < 1e-5, f"HIP graph replay 1 vs eager max diff {diff1:.6e} exceeds 1e-5"

    # second replay with new data (simulates next decode step)
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
        q_new_data, k_cache_orig.clone(), v_cache_orig.clone(),
        k=k_new_data, v=v_new_data,
        cache_seqlens=torch.full(
            (batch_size,), new_cache_len, dtype=torch.int32, device=device
        ),
        causal=True,
    )
    torch.cuda.synchronize()

    if isinstance(out_eager_2, tuple):
        out_eager_2 = out_eager_2[0]

    assert not torch.isnan(out_graph_2).any(), "HIP graph replay 2 produced NaN"
    diff2 = (out_eager_2 - out_graph_2).abs().max().item()
    assert diff2 < 1e-5, f"HIP graph replay 2 vs eager max diff {diff2:.6e} exceeds 1e-5"