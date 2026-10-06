# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse
from aiter.ops.triton.utils._triton import arch_info

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
        has_invalid=sentinels,
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


@pytest.mark.parametrize("T", [1024, 16384])
@pytest.mark.parametrize("H", [64])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("prefix_len", [255, 512])
def test_pa_prefill_sparse_gfx950(T, H, D, prefix_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    if DEVICE_ARCH not in ("gfx950",):
        pytest.skip("this case targets the gfx950 single-source branch")

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
        0,  # extend_len — single-source path
        total_pages,
        1,
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
        None,  # kv                — no extend source
        None,  # kv_indices_extend
        None,  # kv_indptr_extend
        sink,
        scale,
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def _sparse_prefill_single_source_torch(q, kv, indices, indptr, attn_sink, scale):
    """Vectorized fp32 reference for one KV pool: pads each row's ragged slot list
    to the longest row and masks padding and slots outside ``[0, len(kv))``.
    ``attn_sink=None`` means no sink."""
    T, H, _ = q.shape
    lens = (indptr[1:] - indptr[:-1]).long()
    K = max(int(lens.max().item()), 1)
    pos = torch.arange(K, device=q.device)
    in_row = pos[None, :] < lens[:, None]
    flat = (indptr[:-1].long()[:, None] + pos[None, :]).clamp(
        max=max(indices.numel() - 1, 0)
    )
    slots = torch.where(in_row, indices.long()[flat] if indices.numel() else -1, -1)
    valid = (slots >= 0) & (slots < kv.shape[0])
    kv_g = kv.float()[slots.clamp(0, max(kv.shape[0] - 1, 0))]  # [T, K, D]
    scores = torch.einsum("thd,tkd->thk", q.float(), kv_g) * scale
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.float()[None, :, None].expand(T, H, 1)
        scores = torch.cat([scores, sink], dim=-1)
    cmax = scores.amax(dim=-1, keepdim=True)
    cmax = torch.where(cmax == float("-inf"), torch.zeros_like(cmax), cmax)
    w = (scores - cmax).exp()
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    return torch.einsum("thk,tkd->thd", w[..., :K], kv_g).to(q.dtype)


def _inject_invalid_slots(indices, num_kv):
    """Mark slots invalid in place: 1/8 to -1, 1/32 to exactly num_kv (the
    first row past the pool) and 1/32 to far past it, up to the int32 max."""
    n = indices.numel()
    if not n:
        return
    perm = torch.randperm(n, device=indices.device)
    indices[perm[: n // 8]] = -1
    indices[perm[n // 8 : n // 8 + n // 32]] = num_kv
    far = perm[n // 8 + n // 32 : n // 8 + 2 * (n // 32)]
    indices[far] = torch.randint(
        num_kv + 1,
        2**31 - 1,
        (far.numel(),),
        dtype=indices.dtype,
        device=indices.device,
    )


# DSv4.1-Flash TP4: H=16 per rank, D=512, top-512 + 128 SWA = 640 slots.
# Lengths vary per row so the last BLOCK_K tile is usually partial.
@pytest.mark.parametrize("T", [37, 2048])
# H=32 exercises the autotuned launch; H<=16 the gfx942 fixed config.
@pytest.mark.parametrize("H", [8, 16, 32])
@pytest.mark.parametrize("max_len", [17, 640])
@pytest.mark.parametrize("sentinels", [True, False])
@pytest.mark.parametrize("with_sink", [True, False])
def test_pa_prefill_sparse_single_source(T, H, max_len, sentinels, with_sink):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if DEVICE_ARCH in ("gfx950", "gfx1250"):
        pytest.skip("covers the Triton single-source branch")

    D = 512
    torch.manual_seed(0)
    dev = "cuda"
    num_kv = 4096
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(num_kv, D, dtype=torch.bfloat16, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev) if with_sink else None
    lens = torch.randint(1, max_len + 1, (T,), device=dev)
    lens[::7] = 0  # empty rows: output is 0 (sink only)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    indptr[1:] = lens.cumsum(0)
    indices = torch.randint(
        0, num_kv, (int(indptr[-1]),), dtype=torch.int32, device=dev
    )
    if sentinels and indices.numel():
        _inject_invalid_slots(indices, num_kv)
        row = int(torch.nonzero(lens >= 32)[0]) if bool((lens >= 32).any()) else None
        if row is not None:  # a whole leading tile of -1
            s = int(indptr[row])
            indices[s : s + 16] = -1
    scale = D**-0.5

    ref = _sparse_prefill_single_source_torch(q, kv, indices, indptr, sink, scale)
    out = torch.full_like(q, float("nan"))
    ret = pa_prefill_sparse(
        q,
        kv,
        indices,
        indptr,
        None,
        None,
        None,
        sink,
        scale,
        out=out,
    )
    assert ret is out
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# Head dims other than 512 take the autotuned launch on every arch.
@pytest.mark.parametrize("D", [128, 576])
@pytest.mark.parametrize("sentinels", [True, False])
def test_pa_prefill_sparse_single_source_other_head_dims(D, sentinels):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if DEVICE_ARCH in ("gfx950", "gfx1250"):
        pytest.skip("covers the Triton single-source branch")

    torch.manual_seed(1)
    T, H, num_kv, dev = 257, 16, 2048, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(num_kv, D, dtype=torch.bfloat16, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    lens = torch.randint(1, 200, (T,), device=dev)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    indptr[1:] = lens.cumsum(0)
    indices = torch.randint(
        0, num_kv, (int(indptr[-1]),), dtype=torch.int32, device=dev
    )
    if sentinels:
        _inject_invalid_slots(indices, num_kv)
    scale = D**-0.5

    ref = _sparse_prefill_single_source_torch(q, kv, indices, indptr, sink, scale)
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def _triton_branch_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if DEVICE_ARCH in ("gfx950", "gfx1250"):
        pytest.skip("covers the Triton single-source branch")


# An empty pool leaves nothing to gather: every query gets the sink-only (zero) output.
@pytest.mark.parametrize("H", [8, 16])
def test_pa_prefill_sparse_empty_kv_pool(H):
    _triton_branch_only()
    T, D, dev = 64, 512, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.empty(0, D, dtype=torch.bfloat16, device=dev)
    indptr = torch.arange(0, (T + 1) * 16, 16, dtype=torch.int32, device=dev)
    indices = torch.full((T * 16,), -1, dtype=torch.int32, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, D**-0.5)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("scale", [0.0, -0.1])
def test_pa_prefill_sparse_rejects_non_positive_scale(scale):
    _triton_branch_only()
    q = torch.randn(4, 16, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
    indptr = torch.arange(0, 5, dtype=torch.int32, device="cuda")
    indices = torch.zeros(4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="softmax_scale"):
        pa_prefill_sparse(q, kv, indices, indptr, None, None, None, None, scale)


# Strided or overlapping out= buffers would be written wrongly (gfx950 ignores
# out.stride(2)) or raced on (an expand() over heads), so they are rejected.
@pytest.mark.parametrize("layout", ["transposed", "expanded_heads"])
def test_pa_prefill_sparse_rejects_non_contiguous_out(layout):
    _triton_branch_only()
    T, H, D = 4, 16, 512
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, D, dtype=torch.bfloat16, device="cuda")
    indptr = torch.arange(0, T + 1, dtype=torch.int32, device="cuda")
    indices = torch.zeros(T, dtype=torch.int32, device="cuda")
    if layout == "transposed":
        out = torch.empty(T, D, H, dtype=q.dtype, device="cuda").transpose(1, 2)
    else:
        out = torch.empty(T, 1, D, dtype=q.dtype, device="cuda").expand(T, H, D)
    with pytest.raises(AssertionError, match="contiguous"):
        pa_prefill_sparse(
            q, kv, indices, indptr, None, None, None, None, D**-0.5, out=out
        )


# out= on whichever branch this device takes (Triton fallback, gfx950 or
# gfx1250 Gluon): the supplied buffer is returned and holds the same result.
def test_pa_prefill_sparse_out_buffer_every_branch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    T, H, D = 512, 64, 512
    two_sources = DEVICE_ARCH == "gfx1250"
    q, ukv, p_idx, p_indptr, kv, e_idx, e_indptr, sink, scale = _make_inputs(
        T, H, D, 128, 128 if two_sources else 1, T * 128, T * 128 if two_sources else 1
    )
    extend = (kv, e_idx, e_indptr) if two_sources else (None, None, None)
    ref = pa_prefill_sparse(q, ukv, p_idx, p_indptr, *extend, sink, scale)
    out = torch.full_like(q, float("nan"))
    ret = pa_prefill_sparse(q, ukv, p_idx, p_indptr, *extend, sink, scale, out=out)
    assert ret is out
    assert torch.equal(out, ref)
