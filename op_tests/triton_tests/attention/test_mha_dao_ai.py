# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib

import torch
import pytest
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_impl,
)
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_random_padding_mask,
    generate_qkv,
)


@contextlib.contextmanager
def _with_default_device(device):
    """Run the capture under an explicit torch default device, restored on exit.

    Graph-capture safety is relative to global state: a sibling test that leaks
    ``torch.set_default_device("cuda")`` (some set it at module scope and never
    reset it) makes any tensor constructor in the op that omits ``device=``
    allocate on CUDA, an illegal H2D during capture. Parametrizing the graph
    tests over the default device makes both regimes deterministic and
    ordering-independent: the "cpu" case checks capture correctness; the "cuda"
    case reproduces the leaked-global-state regime that surfaces device-implicit
    allocations. (Host<->device syncs inside a captured region are already
    illegal under torch.cuda.graph, so capture catches those too.)
    """
    prev_device = torch.get_default_device()
    try:
        torch.set_default_device(device)
        yield
    finally:
        torch.set_default_device(prev_device)


@pytest.fixture
def dao_ai_impl():
    """Set mha impl to dao_ai for the test, restore to default on cleanup."""
    mha_set_impl("dao_ai")
    yield
    mha_set_impl("default")


@pytest.mark.parametrize(
    "BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, CAUSAL, VARLEN, BWD, WINDOW_SIZE",
    [
        (1, 128, 128, 8, 8, 64, True, False, False, (-1, -1)),  # fwd causal
        (2, 256, 256, 16, 4, 128, False, False, False, (-1, -1)),  # fwd GQA non-causal
        (1, 128, 128, 8, 8, 64, True, True, False, (-1, -1)),  # fwd_varlen causal
        (1, 128, 128, 8, 8, 64, True, False, True, (-1, -1)),  # bwd causal
        (1, 128, 128, 8, 8, 64, True, True, True, (-1, -1)),  # bwd_varlen causal
        (1, 128, 128, 8, 8, 64, True, False, True, (32, 0)),  # causal window
        (1, 128, 128, 8, 8, 64, True, True, True, (32, 0)),  # causal window varlen
        (1, 128, 128, 8, 8, 64, False, False, True, (16, 16)),  # symmetric window
        (1, 128, 128, 8, 8, 64, False, True, True, (16, 16)),  # symmetric win varlen
        (1, 128, 128, 8, 8, 64, False, False, True, (-1, 32)),  # infinite-left window
        # Infinite-RIGHT window (L, -1): the mirror of the infinite-left row above.
        # MUST be non-causal -- under causal the right edge is capped to the diagonal
        # (attention_ref forces window=(L, 0)), so (L, -1) would collapse to (L, 0)
        # and never exercise the new unbounded-right path.
        (1, 128, 128, 8, 8, 64, False, False, True, (32, -1)),  # infinite-right window
        (1, 128, 128, 8, 8, 64, False, True, True, (32, -1)),  # infinite-right varlen
        # seqlen_q != seqlen_k exercises the causal_offset / delta_qk arithmetic
        (1, 128, 256, 8, 8, 64, True, False, True, (32, 0)),  # causal, sq != sk
        (1, 128, 256, 8, 8, 64, True, True, True, (32, 0)),  # causal varlen, sq != sk
        (1, 128, 256, 8, 8, 64, False, False, True, (16, 16)),  # symmetric, sq != sk
        # infinite-right with sq != sk: exercises the sk-sq offset in the
        # unbounded-right per-element mask and the _sliding_window bounds.
        (
            1,
            128,
            256,
            8,
            8,
            64,
            False,
            False,
            True,
            (32, -1),
        ),  # infinite-right sq != sk
        # GQA (num_q_heads != num_k_heads) combined with sliding-window backward
        (2, 128, 128, 16, 4, 64, True, False, True, (32, 0)),  # GQA causal window
        (2, 128, 128, 16, 4, 64, False, True, True, (16, 16)),  # GQA symmetric varlen
        # Production-size sequence lengths (beyond the upstream 2048 cap) with
        # large windows (128/256). seqlen >> window means the window spans many
        # key blocks, exercising the bwd full/partial/skipped block
        # classification that the small (seqlen 128, window 16/32) cases barely
        # reach.
        (1, 4096, 4096, 8, 8, 64, True, False, True, (256, 0)),  # large causal window
        (1, 4096, 4096, 8, 8, 64, True, True, True, (256, 0)),  # large causal varlen
        (1, 4096, 4096, 8, 8, 64, False, False, True, (256, 256)),  # large symmetric
        (2, 4096, 4096, 16, 4, 64, True, False, True, (128, 0)),  # GQA large causal
        # seqlen_q != seqlen_k at production size exercises the causal_offset /
        # delta_qk arithmetic with a window spanning many blocks.
        (1, 4096, 8192, 8, 8, 64, True, False, True, (256, 0)),  # large causal sq != sk
        # Padded last block: seqlen NOT a multiple of the block size, with an active
        # window -> exercises handle_padded_last_block (the last partial-K-block bucket
        # move). Every multiple-of-block case above skips this branch. sq == sk keeps
        # every query row with at least the diagonal key (no fully-masked rows).
        (1, 1023, 1023, 8, 8, 64, True, False, True, (128, 0)),  # causal
        (1, 1025, 1025, 8, 8, 64, False, False, True, (64, 64)),  # symmetric
        (1, 4097, 4097, 8, 8, 64, True, False, True, (256, 0)),  # production size
        (1, 1024, 1024, 8, 8, 64, False, False, True, (0, 0)),  # (0,0) band
        # Broaden the production block-skip regime past hd64 / MHA: block sizes are
        # head-dim dependent and MQA changes the block-classification arithmetic.
        (1, 4096, 4096, 8, 8, 128, True, False, True, (256, 0)),  # hd128
        (2, 4096, 4096, 8, 1, 64, True, False, True, (256, 0)),  # MQA
    ],
)
def test_mha_dao_ai(
    dao_ai_impl,
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    VARLEN: bool,
    BWD: bool,
    WINDOW_SIZE: tuple[int, int],
    dtype=torch.bfloat16,
):
    """Test dao_ai impl dispatch for fwd/bwd x varlen against PyTorch reference."""
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn(
        BATCH,
        SEQLEN_Q,
        NUM_Q_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )
    k = torch.randn(
        BATCH,
        SEQLEN_K,
        NUM_K_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )
    v = torch.randn(
        BATCH,
        SEQLEN_K,
        NUM_K_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )

    if VARLEN:
        query_padding_mask = generate_random_padding_mask(
            SEQLEN_Q, BATCH, "cuda", mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            SEQLEN_K, BATCH, "cuda", mode="full"
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
        q_unpad.requires_grad_(BWD)
        k_unpad.requires_grad_(BWD)
        v_unpad.requires_grad_(BWD)

        with torch.set_grad_enabled(BWD):
            triton_out = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
            )
    else:
        query_padding_mask = None
        key_padding_mask = None
        with torch.set_grad_enabled(BWD):
            triton_out = flash_attn_func(
                q, k, v, causal=CAUSAL, window_size=WINDOW_SIZE
            )

    # Reference + tolerances. For backward, attention_ref_with_tol derives each
    # tensor's atol from the fp32-vs-bf16 reference gap (upstream FA pattern, as in
    # test_mha_v3): bf16 gradient reductions -- largest under MQA / long seqlen --
    # outrun a fixed 1e-2 atol.
    do = torch.randn_like(q) if BWD else None
    if BWD:
        torch_out, (torch_dq, torch_dk, torch_dv), fwd_tol, bwd_tols = (
            attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
            )
        )
    else:
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=WINDOW_SIZE,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
        )
        fwd_tol = (1e-2, 1e-2)

    # Forward check against PyTorch reference
    fwd_atol, fwd_rtol = fwd_tol
    triton_out_fwd = output_pad_fn(triton_out) if VARLEN else triton_out
    torch.testing.assert_close(triton_out_fwd, torch_out, atol=fwd_atol, rtol=fwd_rtol)

    # Backward check against PyTorch reference
    if BWD:
        if VARLEN:
            triton_out = output_pad_fn(triton_out)
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q_unpad, k_unpad, v_unpad), do
            )
            triton_dq = dq_pad_fn(triton_dq)
            triton_dk = dk_pad_fn(triton_dk)
            triton_dv = dk_pad_fn(triton_dv)
        else:
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q, k, v), do
            )

        for tri, ref, (atol, rtol), name in zip(
            (triton_dq, triton_dk, triton_dv),
            (torch_dq, torch_dk, torch_dv),
            bwd_tols,
            ("dq", "dk", "dv"),
        ):
            torch.testing.assert_close(
                tri,
                ref,
                atol=atol,
                rtol=rtol,
                msg=lambda m, name=name: f"dao_ai bwd {name} mismatch\n\n{m}\n",
            )


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_mha_dao_ai_graph(dao_ai_impl, mha_type, default_device):
    """graph capture for flash_attn_func with dao_ai impl."""
    d = 128
    device = "cuda"
    torch.manual_seed(20)
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

    # warmup (Triton JIT)
    for _ in range(3):
        _ = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    q.copy_(q_orig)
    k.copy_(k_orig)
    v.copy_(v_orig)

    g = torch.cuda.CUDAGraph()
    with _with_default_device(default_device), torch.cuda.graph(g):
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


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_mha_dao_ai_varlen_graph(dao_ai_impl, mha_type, default_device):
    """graph capture for flash_attn_varlen_func with dao_ai impl."""
    d = 128
    device = "cuda"
    torch.manual_seed(20)
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

    # warmup (Triton JIT)
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

    with torch.no_grad():
        q_unpad.copy_(q_orig)
        k_unpad.copy_(k_orig)
        v_unpad.copy_(v_orig)

    g = torch.cuda.CUDAGraph()
    with _with_default_device(default_device), torch.cuda.graph(g):
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

    with torch.no_grad():
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
