# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse

DEVICE_ARCH = arch_info.get_arch()

# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------


def _sparse_prefill_attn_torch(
    q,
    unified_kv,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
):
    """Pure-torch reference for sparse prefill attention with two KV sources
    and per-head sink bias.

    Shapes:
        q:                 [T, H, D]
        unified_kv:        [total_pages, D]
        kv_indices_prefix: [total_prefix] int32
        kv_indptr_prefix:  [T+1] int32
        kv:                [total_tokens, D]
        kv_indices_extend: [total_extend] int32
        kv_indptr_extend:  [T+1] int32
        attn_sink:         [H] fp32
    Returns:
        [T, H, D]
    """
    T, H, D = q.shape
    device = q.device

    p_indptr = kv_indptr_prefix.to(torch.int64)
    e_indptr = kv_indptr_extend.to(torch.int64)

    out = torch.zeros(T, H, D, dtype=torch.float32, device=device)

    for t in range(T):
        # Gather prefix KV
        ps = int(p_indptr[t].item())
        pe = int(p_indptr[t + 1].item())
        p_slots = kv_indices_prefix[ps:pe].to(torch.int64)
        p_valid = p_slots >= 0
        p_safe = p_slots.clamp(min=0)
        p_kv = unified_kv[p_safe]  # [P, D]
        p_kv = torch.where(
            p_valid[:, None], p_kv.float(), torch.zeros_like(p_kv.float())
        )

        # Gather extend KV
        es = int(e_indptr[t].item())
        ee = int(e_indptr[t + 1].item())
        e_slots = kv_indices_extend[es:ee].to(torch.int64)
        e_valid = e_slots >= 0
        e_safe = e_slots.clamp(min=0)
        e_kv = kv[e_safe]  # [E, D]
        e_kv = torch.where(
            e_valid[:, None], e_kv.float(), torch.zeros_like(e_kv.float())
        )

        # Concatenate
        all_kv = torch.cat([p_kv, e_kv], dim=0)  # [K, D]
        all_valid = torch.cat([p_valid, e_valid], dim=0)  # [K]
        K = all_kv.shape[0]

        if K == 0:
            continue

        q_t = q[t].float()  # [H, D]
        scores = torch.einsum("hd,kd->hk", q_t, all_kv) * softmax_scale  # [H, K]
        scores = scores.masked_fill(~all_valid[None, :], float("-inf"))

        # Add sink as virtual K with V=0
        sink = attn_sink.float().unsqueeze(1)  # [H, 1]
        combined = torch.cat([scores, sink], dim=-1)  # [H, K+1]
        cmax = combined.amax(dim=-1, keepdim=True)
        cmax = torch.where(
            cmax == float("-inf"),
            torch.zeros_like(cmax),
            cmax,
        )
        weights = (combined - cmax).exp()
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        weights = weights / denom
        weights_kv = weights[:, :K]  # [H, K]
        out[t] = torch.einsum("hk,kd->hd", weights_kv, all_kv)

    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def _make_inputs(
    T: int,
    H: int,
    D: int,
    prefix_len_per_token: int,
    extend_len_per_token: int,
    total_pages: int,
    total_extend_tokens: int,
    dtype=torch.bfloat16,
    seed: int = 0,
    include_sentinels: bool = False,
    variable_len: bool = False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=device) * 0.5
    kv = torch.randn(total_extend_tokens, D, dtype=dtype, device=device) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    # Prefix per-token lengths
    if variable_len:
        p_lens = torch.randint(
            low=1,
            high=prefix_len_per_token + 1,
            size=(T,),
            device=device,
            dtype=torch.int64,
        )
    else:
        p_lens = torch.full(
            (T,), prefix_len_per_token, device=device, dtype=torch.int64
        )

    p_indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
    p_indptr[1:] = p_lens.cumsum(0)
    total_p = int(p_indptr[-1].item())

    p_indices = torch.randint(
        low=0,
        high=total_pages,
        size=(total_p,),
        device=device,
        dtype=torch.int32,
    )

    # Extend per-token lengths
    if variable_len:
        e_lens = torch.randint(
            low=1,
            high=extend_len_per_token + 1,
            size=(T,),
            device=device,
            dtype=torch.int64,
        )
    else:
        e_lens = torch.full(
            (T,), extend_len_per_token, device=device, dtype=torch.int64
        )

    e_indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
    e_indptr[1:] = e_lens.cumsum(0)
    total_e = int(e_indptr[-1].item())

    e_indices = torch.randint(
        low=0,
        high=total_extend_tokens,
        size=(total_e,),
        device=device,
        dtype=torch.int32,
    )

    if include_sentinels:
        if total_p > 0:
            n_s = max(1, total_p // 16)
            pos = torch.randperm(total_p, device=device)[:n_s]
            p_indices[pos] = -1
        if total_e > 0:
            n_s = max(1, total_e // 16)
            pos = torch.randperm(total_e, device=device)[:n_s]
            e_indices[pos] = -1

    p_indptr = p_indptr.to(torch.int32)
    e_indptr = e_indptr.to(torch.int32)
    softmax_scale = float(D) ** -0.5

    return (
        q,
        unified_kv,
        p_indices,
        p_indptr,
        kv,
        e_indices,
        e_indptr,
        attn_sink,
        softmax_scale,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# DSV4-Flash shapes: H=64, D=512 (MLA kv_head_dim), window_size=128,
# index_topk=512. First-chunk: prefix=0, extend=128. Chunked 2nd chunk:
# prefix up to 127 (SWA) + 128 (CSA topk) = 255, extend=128.
@pytest.mark.parametrize("T", [1024, 16384])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("prefix_len", [0, 128, 255])
@pytest.mark.parametrize("extend_len", [1, 128])
@pytest.mark.parametrize("sentinels", [True, False])
def test_pa_prefill_sparse_vs_reference(T, H, D, prefix_len, extend_len, sentinels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("pa_prefill_sparse requires gfx1250")

    total_pages = max(T * prefix_len, 1)
    total_ext = max(T * extend_len, 1)

    (
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    ) = _make_inputs(
        T,
        H,
        D,
        prefix_len,
        extend_len,
        total_pages,
        total_ext,
        include_sentinels=sentinels,
    )

    ref = _sparse_prefill_attn_torch(
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    )
    out = pa_prefill_sparse(
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    )

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("T", [1024, 16384])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("prefix_len", [128])
@pytest.mark.parametrize("extend_len", [0])
def test_pa_prefill_sparse_prefix_only(T, H, D, prefix_len, extend_len):
    """When extend region is empty, should match decode-style prefix-only."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("pa_prefill_sparse requires gfx1250")

    total_pages = max(T * prefix_len, 1)

    (
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    ) = _make_inputs(
        T,
        H,
        D,
        prefix_len,
        max(extend_len, 1),
        total_pages,
        1,
    )

    # Override extend to be empty
    e_indptr = torch.zeros(T + 1, dtype=torch.int32, device=q.device)
    e_idx = torch.empty(0, dtype=torch.int32, device=q.device)
    kv = torch.empty(1, D, dtype=q.dtype, device=q.device)

    ref = _sparse_prefill_attn_torch(
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    )
    out = pa_prefill_sparse(
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    )

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("T", [1024, 16384])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("prefix_len", [0])
@pytest.mark.parametrize("extend_len", [128])
def test_pa_prefill_sparse_extend_only(T, H, D, prefix_len, extend_len):
    """When prefix region is empty, should work from extend source alone."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("pa_prefill_sparse requires gfx1250")

    total_ext = max(T * extend_len, 1)

    (
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    ) = _make_inputs(
        T,
        H,
        D,
        max(prefix_len, 1),
        extend_len,
        1,
        total_ext,
    )

    # Override prefix to be empty
    p_indptr = torch.zeros(T + 1, dtype=torch.int32, device=q.device)
    p_idx = torch.empty(0, dtype=torch.int32, device=q.device)
    ukv = torch.empty(1, D, dtype=q.dtype, device=q.device)

    ref = _sparse_prefill_attn_torch(
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    )
    out = pa_prefill_sparse(
        q,
        ukv,
        p_idx,
        p_indptr,
        kv,
        e_idx,
        e_indptr,
        sink,
        scale,
    )

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
