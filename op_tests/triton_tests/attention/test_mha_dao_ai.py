# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
from unittest.mock import Mock

import pytest
import torch

from aiter.ops.triton._triton_kernels.attention.dao_ai import flash_attn_2
from aiter.ops.triton.attention import mha
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    mha_set_impl,
)
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_qkv,
    generate_random_padding_mask,
)


@contextlib.contextmanager
def _with_default_device(device):
    """Run the capture under an explicit torch default device, restored on exit."""
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


# fmt: off
DENSE_CASES = [
    (1, 128, 128, 8, 8, 64, True, False, (-1, -1), 0.0),  # fwd causal
    (2, 256, 256, 16, 4, 128, False, False, (-1, -1), 0.0),  # fwd GQA non-causal
    (1, 128, 128, 8, 8, 64, True, True, (-1, -1), 0.0),  # bwd causal
    (1, 128, 128, 8, 8, 64, True, True, (32, 0), 0.0),  # causal window
    (1, 128, 128, 8, 8, 64, False, True, (16, 16), 0.0),  # symmetric window
    (1, 128, 128, 8, 8, 64, False, True, (-1, 32), 0.0),  # infinite-left window
    # Infinite-RIGHT window (L, -1): the mirror of the infinite-left row above.
    # MUST be non-causal -- under causal the right edge is capped to the diagonal
    # (attention_ref forces window=(L, 0)), so (L, -1) would collapse to (L, 0)
    # and never exercise the new unbounded-right path.
    (1, 128, 128, 8, 8, 64, False, True, (32, -1), 0.0),  # infinite-right window
    # seqlen_q != seqlen_k exercises the causal_offset / delta_qk arithmetic
    (1, 128, 256, 8, 8, 64, True, True, (32, 0), 0.0),  # causal, sq != sk
    (1, 128, 256, 8, 8, 64, False, True, (16, 16), 0.0),  # symmetric, sq != sk
    # infinite-right with sq != sk: exercises the sk-sq offset in the
    # unbounded-right per-element mask and the _sliding_window bounds.
    (1, 128, 256, 8, 8, 64, False, True, (32, -1), 0.0),  # infinite-right sq != sk
    # GQA (num_q_heads != num_k_heads) combined with sliding-window backward
    (2, 128, 128, 16, 4, 64, True, True, (32, 0), 0.0),  # GQA causal window
    # Production-size sequence lengths (beyond the upstream 2048 cap) with
    # large windows (128/256). seqlen >> window means the window spans many
    # key blocks, exercising the bwd full/partial/skipped block
    # classification that the small (seqlen 128, window 16/32) cases barely
    # reach.
    (1, 4096, 4096, 8, 8, 64, True, True, (256, 0), 0.0),  # large causal window
    (1, 4096, 4096, 8, 8, 64, False, True, (256, 256), 0.0),  # large symmetric
    (2, 4096, 4096, 16, 4, 64, True, True, (128, 0), 0.0),  # GQA large causal
    # seqlen_q != seqlen_k at production size exercises the causal_offset /
    # delta_qk arithmetic with a window spanning many blocks.
    (1, 4096, 8192, 8, 8, 64, True, True, (256, 0), 0.0),  # large causal sq != sk
    # Padded last block: seqlen NOT a multiple of the block size, with an active
    # window -> exercises handle_padded_last_block (the last partial-K-block bucket
    # move). Every multiple-of-block case above skips this branch. sq == sk keeps
    # every query row with at least the diagonal key (no fully-masked rows).
    (1, 1023, 1023, 8, 8, 64, True, True, (128, 0), 0.0),  # causal
    (1, 1025, 1025, 8, 8, 64, False, True, (64, 64), 0.0),  # symmetric
    (1, 4097, 4097, 8, 8, 64, True, True, (256, 0), 0.0),  # production size
    (1, 1024, 1024, 8, 8, 64, False, True, (0, 0), 0.0),  # (0,0) band
    # Broaden the production block-skip regime past hd64 / MHA: block sizes are
    # head-dim dependent and MQA changes the block-classification arithmetic.
    (1, 4096, 4096, 8, 8, 128, True, True, (256, 0), 0.0),  # hd128
    (2, 4096, 4096, 8, 1, 64, True, True, (256, 0), 0.0),  # MQA
    # Dropout: the reference applies the dropout record the forward returned.
    (2, 96, 96, 4, 4, 64, False, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, True, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, False, True, (16, 16), 0.17),
]
VARLEN_CASES = [
    (1, 128, 128, 8, 8, 64, True, False, (-1, -1), 0.0),  # fwd_varlen causal
    (1, 128, 128, 8, 8, 64, True, True, (-1, -1), 0.0),  # bwd_varlen causal
    (1, 128, 128, 8, 8, 64, True, True, (32, 0), 0.0),  # causal window varlen
    (1, 128, 128, 8, 8, 64, False, True, (16, 16), 0.0),  # symmetric win varlen
    (1, 128, 128, 8, 8, 64, False, True, (32, -1), 0.0),  # infinite-right varlen
    (1, 128, 256, 8, 8, 64, True, True, (32, 0), 0.0),  # causal varlen, sq != sk
    (2, 128, 128, 16, 4, 64, False, True, (16, 16), 0.0),  # GQA symmetric varlen
    (1, 4096, 4096, 8, 8, 64, True, True, (256, 0), 0.0),  # large causal varlen
    # Dropout, with random padding: the reference applies the forward's record.
    (2, 96, 96, 4, 4, 64, False, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, True, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, False, True, (16, 16), 0.17),
]
# fmt: on


@pytest.mark.parametrize(
    "BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, CAUSAL, BWD, WINDOW_SIZE, DROPOUT",
    DENSE_CASES,
)
def test_flash_attn_func_correctness(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    BWD: bool,
    WINDOW_SIZE: tuple[int, int],
    DROPOUT: float,
    dtype=torch.bfloat16,
):
    """flash_attn_func with dao_ai, forward and backward, matches the PyTorch reference.
    With dropout, the reference applies the dropout record that the forward returned."""
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

    with torch.set_grad_enabled(BWD):
        out, _, scores = flash_attn_func(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=WINDOW_SIZE,
            dropout_p=DROPOUT,
            return_lse=True,
            return_attn_probs=True,
            implementation="dao_ai",
        )
    mask = scores >= 0 if DROPOUT else None

    # For backward, attention_ref_with_tol derives each tensor's atol from the
    # fp32-vs-bf16 reference gap (upstream FA pattern).
    do = torch.randn_like(q) if BWD else None
    if BWD:
        ref, (ref_dq, ref_dk, ref_dv), (fwd_atol, fwd_rtol), bwd_tols = (
            attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                dropout_p=DROPOUT,
                dropout_mask=mask,
            )
        )
    else:
        ref, _, _ = attention_ref(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=WINDOW_SIZE,
            dropout_p=DROPOUT,
            dropout_mask=mask,
        )
        fwd_atol, fwd_rtol = 1e-2, 1e-2
    torch.testing.assert_close(out, ref, atol=fwd_atol, rtol=fwd_rtol)

    if BWD:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
        for tri, r, (atol, rtol), name in zip(
            (dq, dk, dv), (ref_dq, ref_dk, ref_dv), bwd_tols, ("dq", "dk", "dv")
        ):
            torch.testing.assert_close(
                tri,
                r,
                atol=atol,
                rtol=rtol,
                msg=lambda m, name=name: f"dao_ai bwd {name} mismatch\n\n{m}\n",
            )

        if DROPOUT:
            # The gradients are closer to the forward's dropout mask than to a shifted
            # one, as they would not be if backward lost the forward's RNG state.
            _, wrong_grads, _, _ = attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                dropout_p=DROPOUT,
                dropout_mask=torch.roll(mask, 1, dims=-1),
            )
            for tri, r, wrong, name in zip(
                (dq, dk, dv), (ref_dq, ref_dk, ref_dv), wrong_grads, ("dq", "dk", "dv")
            ):
                assert (tri - r).abs().max() < (tri - wrong).abs().max(), name


@pytest.mark.parametrize(
    "BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, CAUSAL, BWD, WINDOW_SIZE, DROPOUT",
    VARLEN_CASES,
)
def test_flash_attn_varlen_func_correctness(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    BWD: bool,
    WINDOW_SIZE: tuple[int, int],
    DROPOUT: float,
    dtype=torch.bfloat16,
):
    """flash_attn_varlen_func with dao_ai, forward and backward, matches the PyTorch
    reference. With dropout, the reference applies the dropout record that the forward
    returned."""
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    padding = "random" if DROPOUT else "full"
    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH, "cuda", padding)
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH, "cuda", padding)
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
        out_unpad, _, scores = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=CAUSAL,
            window_size=WINDOW_SIZE,
            dropout_p=DROPOUT,
            return_lse=True,
            return_attn_probs=True,
            implementation="dao_ai",
        )
    out = output_pad_fn(out_unpad)
    mask = None
    if DROPOUT:
        # The record covers the longest real sequence; the rest is dropped.
        scores = scores[:, :, :SEQLEN_Q, :SEQLEN_K]
        pad = (0, SEQLEN_K - scores.shape[3], 0, SEQLEN_Q - scores.shape[2])
        mask = torch.nn.functional.pad(scores, pad, value=-1.0) >= 0

    do = torch.randn_like(q) if BWD else None
    if BWD:
        ref, (ref_dq, ref_dk, ref_dv), (fwd_atol, fwd_rtol), bwd_tols = (
            attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                dropout_p=DROPOUT,
                dropout_mask=mask,
            )
        )
    else:
        ref, _, _ = attention_ref(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=WINDOW_SIZE,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=DROPOUT,
            dropout_mask=mask,
        )
        fwd_atol, fwd_rtol = 1e-2, 1e-2
    torch.testing.assert_close(out, ref, atol=fwd_atol, rtol=fwd_rtol)

    if BWD:
        dq, dk, dv = torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), do)
        dq, dk, dv = dq_pad_fn(dq), dk_pad_fn(dk), dk_pad_fn(dv)
        for tri, r, (atol, rtol), name in zip(
            (dq, dk, dv), (ref_dq, ref_dk, ref_dv), bwd_tols, ("dq", "dk", "dv")
        ):
            torch.testing.assert_close(
                tri,
                r,
                atol=atol,
                rtol=rtol,
                msg=lambda m, name=name: f"dao_ai varlen bwd {name} mismatch\n\n{m}\n",
            )

        if DROPOUT:
            # The gradients are closer to the forward's dropout mask than to a shifted
            # one, as they would not be if backward lost the forward's RNG state.
            _, wrong_grads, _, _ = attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                dropout_p=DROPOUT,
                dropout_mask=torch.roll(mask, 1, dims=-1),
            )
            for tri, r, wrong, name in zip(
                (dq, dk, dv), (ref_dq, ref_dk, ref_dv), wrong_grads, ("dq", "dk", "dv")
            ):
                assert (tri - r).abs().max() < (tri - wrong).abs().max(), name


@pytest.mark.parametrize("BWD_MODE", ["fused", "fused_atomic", "split"])
@pytest.mark.parametrize("SEQLEN, SCALE", [(3, 30.0), (11, 30.0), (3, 1.0)])
def test_mha_dao_ai_bwd_short_noncausal_varlen(
    monkeypatch,
    BWD_MODE: str,
    SEQLEN: int,
    SCALE: float,
    dtype=torch.bfloat16,
):
    """Non-causal varlen backward on sequences shorter than one block.

    With large logits a row's logsumexp can be very negative; the out-of-range
    slots of the partial block then give exp(-lse) = inf unless the bounds mask
    is applied, which turns dQ into NaN. Causal cases always masked, so only
    non-causal ones exposed it.
    """
    monkeypatch.setattr(flash_attn_2, "BWD_MODE", BWD_MODE)
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    BATCH, NUM_HEADS, HEAD_SZ = 8, 8, 64
    q, v = (
        (torch.randn(BATCH, SEQLEN, NUM_HEADS, HEAD_SZ, device="cuda") * SCALE)
        .to(dtype)
        .requires_grad_()
        for _ in range(2)
    )
    # k = -q makes each query's own key score -|q|^2 / sqrt(d), so the rows of
    # the shortest sequences get a very negative logsumexp.
    k = (-q).detach().requires_grad_()
    padding_mask = generate_random_padding_mask(SEQLEN, BATCH, "cuda", mode="random")
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
    ) = generate_qkv(q, k, v, padding_mask, padding_mask, kvpacked=False)
    q_unpad.requires_grad_()
    k_unpad.requires_grad_()
    v_unpad.requires_grad_()

    triton_out = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        implementation="dao_ai",
    )
    do = torch.randn_like(q)
    _, (torch_dq, torch_dk, torch_dv), _, bwd_tols = attention_ref_with_tol(
        q,
        k,
        v,
        do,
        causal=False,
        query_padding_mask=padding_mask,
        key_padding_mask=padding_mask,
    )
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        output_pad_fn(triton_out), (q_unpad, k_unpad, v_unpad), do
    )
    for tri, ref, (atol, rtol), name in zip(
        (dq_pad_fn(triton_dq), dk_pad_fn(triton_dk), dk_pad_fn(triton_dv)),
        (torch_dq, torch_dk, torch_dv),
        bwd_tols,
        ("dq", "dk", "dv"),
    ):
        assert torch.isfinite(tri).all(), f"non-finite {name}"
        torch.testing.assert_close(tri, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("new_kv", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_flash_attn_with_kvcache_correctness(mha_type, causal, new_kv):
    """flash_attn_with_kvcache matches attention over each batch's valid cache entries,
    and appends the new keys and values to the cache at cache_seqlens."""
    torch.manual_seed(20)
    batch, seqlen_q, seqlen_new, cache_len, nheads, d = 2, 4, 3, 128, 8, 64
    nheads_k = nheads if mha_type == "mha" else 2
    q = torch.randn(batch, seqlen_q, nheads, d, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(
        batch, cache_len, nheads_k, d, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([50, 97], device="cuda", dtype=torch.int32)
    k_new = v_new = None
    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    seqlens_k = cache_seqlens.clone()
    if new_kv:
        k_new = torch.randn(
            batch, seqlen_new, nheads_k, d, device="cuda", dtype=torch.bfloat16
        )
        v_new = torch.randn_like(k_new)
        for b, start in enumerate(cache_seqlens.tolist()):
            k_ref[b, start : start + seqlen_new] = k_new[b]
            v_ref[b, start : start + seqlen_new] = v_new[b]
        seqlens_k += seqlen_new

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, k_new, v_new, cache_seqlens=cache_seqlens, causal=causal
    )

    key_padding_mask = (
        torch.arange(cache_len, device="cuda")[None, :] < seqlens_k[:, None]
    )
    ref, _, _ = attention_ref(q, k_ref, v_ref, None, key_padding_mask, causal=causal)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_cache, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, v_ref, atol=0, rtol=0)


@pytest.mark.parametrize("return_softmax", [False, True])
def test_mha_dao_ai_contract(return_softmax):
    """Each function returns the shapes, dtypes and devices it declares:
    _flash_attn_forward and _flash_attn_backward as FlashAttention's fake ops declare,
    the public functions as their docstrings do."""
    torch.manual_seed(20)
    batch, seqlen, heads, d = 2, 128, 4, 64
    q = torch.randn(batch, seqlen, heads, d, device="cuda", dtype=torch.float16)
    q_unpad = q.flatten(0, 1)
    cu_seqlens = torch.arange(
        0, (batch + 1) * seqlen, seqlen, device="cuda", dtype=torch.int32
    )
    record = (batch, heads, seqlen, seqlen) if return_softmax else (0,)

    out, lse, s_dmask, _, _, rng_state = mha._flash_attn_forward(
        q,
        q,
        q,
        0.17,
        d**-0.5,
        True,
        -1,
        -1,
        None,
        None,
        True,
        return_softmax,
        seqlen,
        seqlen,
        implementation="dao_ai",
    )
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (batch, heads, seqlen) and lse.dtype == torch.float32
    assert s_dmask.shape == record and s_dmask.dtype == q.dtype
    assert rng_state.shape == (2,) and rng_state.dtype == torch.int64
    assert rng_state.device == q.device

    grads = (torch.empty_like(q), torch.empty_like(q), torch.empty_like(q))
    softmax_d = mha._flash_attn_backward(
        torch.ones_like(out),
        q,
        q,
        q,
        out,
        lse,
        *grads,
        None,
        0.17,
        d**-0.5,
        True,
        -1,
        -1,
        None,
        False,
        seqlen,
        seqlen,
        rng_state=rng_state,
        implementation="dao_ai",
    )
    assert softmax_d.shape == (batch, heads, seqlen)
    assert softmax_d.dtype == torch.float32

    out, lse, s_dmask, _, _, rng_state = mha._flash_attn_forward(
        q_unpad,
        q_unpad,
        q_unpad,
        0.17,
        d**-0.5,
        True,
        -1,
        -1,
        None,
        None,
        True,
        return_softmax,
        seqlen,
        seqlen,
        cu_seqlens,
        cu_seqlens,
        implementation="dao_ai",
    )
    assert out.shape == q_unpad.shape and out.dtype == q.dtype
    assert lse.shape == (heads, batch * seqlen) and lse.dtype == torch.float32
    assert s_dmask.shape == record and s_dmask.dtype == q.dtype
    assert rng_state.shape == (2,) and rng_state.device == q.device

    grads = (
        torch.empty_like(q_unpad),
        torch.empty_like(q_unpad),
        torch.empty_like(q_unpad),
    )
    softmax_d = mha._flash_attn_backward(
        torch.ones_like(out),
        q_unpad,
        q_unpad,
        q_unpad,
        out,
        lse,
        *grads,
        None,
        0.17,
        d**-0.5,
        True,
        -1,
        -1,
        None,
        False,
        seqlen,
        seqlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        rng_state=rng_state,
        implementation="dao_ai",
    )
    assert softmax_d.shape == (heads, batch * seqlen)
    assert softmax_d.dtype == torch.float32

    out, lse, probs = flash_attn_func(
        q,
        q,
        q,
        dropout_p=0.17,
        causal=True,
        return_lse=True,
        return_attn_probs=True,
        implementation="dao_ai",
    )
    assert out.shape == q.shape and lse.shape == (batch, heads, seqlen)
    assert probs.shape == (batch, heads, seqlen, seqlen)

    out, lse, probs = flash_attn_varlen_func(
        q_unpad,
        q_unpad,
        q_unpad,
        cu_seqlens,
        cu_seqlens,
        seqlen,
        seqlen,
        dropout_p=0.17,
        causal=True,
        return_lse=True,
        return_attn_probs=True,
        implementation="dao_ai",
    )
    assert out.shape == q_unpad.shape and lse.shape == (heads, batch * seqlen)
    assert probs.shape == (batch, heads, seqlen, seqlen)

    cache_seqlens = torch.full((batch,), seqlen // 2, device="cuda", dtype=torch.int32)
    out, lse = flash_attn_with_kvcache(
        q[:, :1], q, q, cache_seqlens=cache_seqlens, return_softmax_lse=True
    )
    assert out.shape == (batch, 1, heads, d) and lse.shape == (batch, heads, 1)
    assert lse.dtype == torch.float32


@pytest.mark.parametrize("head_size", [53, 64])
def test_mha_dao_ai_routing(monkeypatch, head_size):
    """Each forward's backward runs that forward's implementation, even after
    mha_set_impl changes the default between them: flash_attn_func, then
    flash_attn_varlen_func."""
    dao_bwd = Mock(wraps=mha.flash_attn_2.bwd)
    dao_varlen_bwd = Mock(wraps=mha.flash_attn_2.varlen_bwd)
    default_bwd = Mock(wraps=mha.flash_attn_onekernel_backward)
    monkeypatch.setattr(mha.flash_attn_2, "bwd", dao_bwd)
    monkeypatch.setattr(mha.flash_attn_2, "varlen_bwd", dao_varlen_bwd)
    monkeypatch.setattr(mha, "flash_attn_onekernel_backward", default_bwd)
    q, k, v = (
        torch.randn(
            2, 8, 4, head_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    q_unpad, k_unpad, v_unpad = (
        torch.randn(
            16, 4, head_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
    try:
        mha_set_impl("dao_ai")
        dao_out = flash_attn_func(q, k, v, deterministic=True)
        default_out = flash_attn_func(q, k, v, implementation="default")
        mha_set_impl("default")
        torch.autograd.grad(dao_out, (q, k, v), torch.ones_like(dao_out))
        assert dao_bwd.call_count == 1 and dao_bwd.call_args.kwargs["deterministic"]
        assert default_bwd.call_count == 0
        mha_set_impl("dao_ai")
        grads = torch.autograd.grad(
            default_out, (q, k, v), torch.ones_like(default_out)
        )
        assert default_bwd.call_count == 1 and dao_bwd.call_count == 1
        assert [g.shape for g in grads] == [q.shape, k.shape, v.shape]

        dao_out = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad, cu_seqlens, cu_seqlens, 8, 8, deterministic=True
        )
        default_out = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens,
            cu_seqlens,
            8,
            8,
            implementation="default",
        )
        mha_set_impl("default")
        torch.autograd.grad(
            dao_out, (q_unpad, k_unpad, v_unpad), torch.ones_like(dao_out)
        )
        assert dao_varlen_bwd.call_count == 1
        assert dao_varlen_bwd.call_args.kwargs["deterministic"]
        assert default_bwd.call_count == 1
        mha_set_impl("dao_ai")
        grads = torch.autograd.grad(
            default_out, (q_unpad, k_unpad, v_unpad), torch.ones_like(default_out)
        )
        assert default_bwd.call_count == 2 and dao_varlen_bwd.call_count == 1
        assert [g.shape for g in grads] == [q_unpad.shape, k_unpad.shape, v_unpad.shape]
    finally:
        mha_set_impl("default")


def test_mha_dao_ai_refusal(monkeypatch):
    """What the Dao-AI implementation does not support fails with a named error before
    any forward runs, and an unknown default leaves the default unchanged."""
    forward = Mock(wraps=mha._flash_attn_forward)
    monkeypatch.setattr(mha, "_flash_attn_forward", forward)
    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    q_unpad = q[0]
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    descale = torch.ones(1, 4, 2, device="cuda", dtype=torch.float32)
    buffer = torch.zeros(1, device="cuda", dtype=torch.int32)
    cache_seqlens = torch.full((1,), 64, device="cuda", dtype=torch.int32)
    try:
        mha_set_impl("dao_ai")
        with pytest.raises(ValueError, match="not supported by the dao_ai"):
            flash_attn_func(
                q, q, q, q_mx_scale=scale, k_mx_scale=scale, v_mx_scale=scale
            )
        with pytest.raises(ValueError, match="not supported by the dao_ai"):
            flash_attn_func(
                q, q, q, q_block_descale=descale, k_block_descale=descale, v_scale=scale
            )

        mha_set_impl("default")
        with pytest.raises(ValueError, match="not supported by the dao_ai"):
            flash_attn_func(
                q,
                q,
                q,
                q_mx_scale=scale,
                k_mx_scale=scale,
                v_mx_scale=scale,
                implementation="dao_ai",
            )
        with pytest.raises(ValueError, match="Unknown MHA implementation"):
            flash_attn_func(
                q,
                q,
                q,
                q_mx_scale=scale,
                k_mx_scale=scale,
                v_mx_scale=scale,
                implementation="unknown",
            )
        with pytest.raises(ValueError, match="not supported by the dao_ai"):
            flash_attn_func(
                q,
                q,
                q,
                q_block_descale=descale,
                k_block_descale=descale,
                v_scale=scale,
                implementation="dao_ai",
            )
        with pytest.raises(ValueError, match="Unknown MHA implementation"):
            flash_attn_func(
                q,
                q,
                q,
                q_block_descale=descale,
                k_block_descale=descale,
                v_scale=scale,
                implementation="unknown",
            )
        with pytest.raises(ValueError, match="Unknown MHA implementation"):
            flash_attn_func(q, q, q, implementation="unknown")
        with pytest.raises(ValueError, match="Unknown MHA implementation"):
            flash_attn_varlen_func(
                q_unpad,
                q_unpad,
                q_unpad,
                cu_seqlens,
                cu_seqlens,
                64,
                64,
                implementation="unknown",
            )
        with pytest.raises(NotImplementedError):
            flash_attn_varlen_func(
                q_unpad,
                q_unpad,
                q_unpad,
                cu_seqlens,
                cu_seqlens,
                64,
                64,
                implementation="dao_ai",
                block_table=buffer,
            )
        with pytest.raises(NotImplementedError):
            flash_attn_varlen_func(
                q_unpad,
                q_unpad,
                q_unpad,
                cu_seqlens,
                cu_seqlens,
                64,
                64,
                implementation="dao_ai",
                out=buffer,
            )
        with pytest.raises(NotImplementedError, match="softcap"):
            flash_attn_with_kvcache(q, q, q, cache_seqlens=cache_seqlens, softcap=1.0)
        with pytest.raises(NotImplementedError, match="num_splits"):
            flash_attn_with_kvcache(q, q, q, cache_seqlens=cache_seqlens, num_splits=2)
        with pytest.raises(ValueError, match="Unknown MHA implementation"):
            mha_set_impl("unknown")
        assert mha._MHA_IMPL == "default"
        forward.assert_not_called()
    finally:
        mha_set_impl("default")


@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_mha_dao_ai_capture(dao_ai_impl, mha_type, default_device):
    """Each function, captured in a CUDA graph under either default device, replays to
    the eager result: nothing allocates on the default device or syncs the host."""

    def captured(step):
        """Warm up on a side stream (Triton JIT), capture, replay; the replayed outputs."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with _with_default_device(default_device), torch.cuda.graph(graph):
            outputs = step()
        graph.replay()
        torch.cuda.synchronize()
        return outputs

    torch.manual_seed(20)
    batch, seqlen, nheads, d = 2, 128, 8, 128
    nheads_k = nheads if mha_type == "mha" else 2
    q = torch.randn(batch, seqlen, nheads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads_k, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads_k, d, device="cuda", dtype=torch.bfloat16)
    padding_mask = generate_random_padding_mask(seqlen, batch, "cuda", mode="full")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        *_,
    ) = generate_qkv(q, k, v, padding_mask, padding_mask, kvpacked=False)
    dout = torch.randn_like(q)
    dout_unpad = torch.randn_like(q_unpad)

    replayed = captured(lambda: flash_attn_func(q, k, v, causal=True))
    torch.testing.assert_close(
        replayed, flash_attn_func(q, k, v, causal=True), atol=1e-5, rtol=0
    )

    replayed = captured(
        lambda: flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
        )
    )
    eager = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=True,
    )
    torch.testing.assert_close(replayed, eager, atol=1e-5, rtol=0)

    cache_seqlens = torch.full((batch,), seqlen // 2, device="cuda", dtype=torch.int32)
    replayed = captured(
        lambda: flash_attn_with_kvcache(q[:, :1], k, v, cache_seqlens=cache_seqlens)
    )
    eager = flash_attn_with_kvcache(q[:, :1], k, v, cache_seqlens=cache_seqlens)
    torch.testing.assert_close(replayed, eager, atol=1e-5, rtol=0)

    inputs = tuple(x.detach().requires_grad_() for x in (q, k, v))
    replayed = captured(
        lambda: torch.autograd.grad(
            flash_attn_func(*inputs, dropout_p=0.17, causal=True), inputs, dout
        )
    )
    eager = torch.autograd.grad(
        flash_attn_func(*inputs, dropout_p=0.17, causal=True), inputs, dout
    )
    for r, e in zip(replayed, eager):
        torch.testing.assert_close(r, e)

    inputs = tuple(x.detach().requires_grad_() for x in (q_unpad, k_unpad, v_unpad))
    replayed = captured(
        lambda: torch.autograd.grad(
            flash_attn_varlen_func(
                *inputs,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.17,
                causal=True,
            ),
            inputs,
            dout_unpad,
        )
    )
    eager = torch.autograd.grad(
        flash_attn_varlen_func(
            *inputs,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.17,
            causal=True,
        ),
        inputs,
        dout_unpad,
    )
    for r, e in zip(replayed, eager):
        torch.testing.assert_close(r, e)
