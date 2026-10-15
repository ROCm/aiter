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


def _gather_valid(pool, indices, start, end):
    """Rows ``indices[start:end]`` of ``pool`` in fp32, with slots outside
    ``[0, len(pool))`` zeroed and reported invalid."""
    slots = indices[start:end].to(torch.int64)
    valid = (slots >= 0) & (slots < pool.shape[0])
    rows = torch.zeros(slots.numel(), pool.shape[1], device=pool.device)
    rows[valid] = pool[slots[valid]].float()
    return rows, valid


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
        kv:                [total_tokens, D] — ``None`` for no extend source
        kv_indices_extend: [total_extend] int32 — ``None`` for no extend source
        kv_indptr_extend:  [T+1] int32 — ``None`` for no extend source
        attn_sink:         [H] fp32 — ``None`` for no sink
    Slots outside ``[0, len(pool))`` (``-1`` sentinels or past the pool) are
    masked out.
    Returns:
        [T, H, D]
    """
    T, H, D = q.shape
    device = q.device
    has_extend = kv is not None

    p_indptr = kv_indptr_prefix.to(torch.int64)
    if has_extend:
        e_indptr = kv_indptr_extend.to(torch.int64)

    out = torch.zeros(T, H, D, dtype=torch.float32, device=device)

    for t in range(T):
        all_kv, all_valid = _gather_valid(
            unified_kv, kv_indices_prefix, p_indptr[t], p_indptr[t + 1]
        )
        if has_extend:
            e_kv, e_valid = _gather_valid(
                kv, kv_indices_extend, e_indptr[t], e_indptr[t + 1]
            )
            all_kv = torch.cat([all_kv, e_kv], dim=0)  # [K, D]
            all_valid = torch.cat([all_valid, e_valid], dim=0)  # [K]
        K = all_kv.shape[0]

        if K == 0:
            continue

        q_t = q[t].float()  # [H, D]
        scores = torch.einsum("hd,kd->hk", q_t, all_kv) * softmax_scale  # [H, K]
        scores = scores.masked_fill(~all_valid[None, :], float("-inf"))

        # Add sink as virtual K with V=0
        if attn_sink is not None:
            sink = attn_sink.float().unsqueeze(1)  # [H, 1]
            scores = torch.cat([scores, sink], dim=-1)  # [H, K+1]
        cmax = scores.amax(dim=-1, keepdim=True)
        cmax = torch.where(
            cmax == float("-inf"),
            torch.zeros_like(cmax),
            cmax,
        )
        weights = (scores - cmax).exp()
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


def _triton_branch_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if DEVICE_ARCH in ("gfx950", "gfx1250"):
        pytest.skip("covers the Triton single-source branch")


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
# H=32 exercises the autotuned launch; H<=16 the gfx942 pinned config, in
# both dtypes since that config is keyed by shape only.
@pytest.mark.parametrize(
    "H, dtype",
    [
        (8, torch.float16),
        (16, torch.bfloat16),
        (16, torch.float16),
        (32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("with_sink", [True, False])
def test_pa_prefill_sparse_single_source(H, dtype, with_sink):
    _triton_branch_only()

    T, D, max_len = 300, 512, 640
    torch.manual_seed(0)
    dev = "cuda"
    num_kv = 4096
    q = torch.randn(T, H, D, dtype=dtype, device=dev)
    kv = torch.randn(num_kv, D, dtype=dtype, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev) if with_sink else None
    lens = torch.randint(1, max_len + 1, (T,), device=dev)
    lens[::7] = 0  # empty rows: output is 0 (sink only)
    lens[1::7] = torch.randint(1, 17, (lens[1::7].numel(),), device=dev)  # short rows
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    indptr[1:] = lens.cumsum(0)
    indices = torch.randint(
        0, num_kv, (int(indptr[-1]),), dtype=torch.int32, device=dev
    )
    # Every other row keeps only valid slots; the rest get invalid ones.
    flat_row = torch.repeat_interleave(torch.arange(T, device=dev), lens)
    with_invalid = indices.clone()
    _inject_invalid_slots(with_invalid, num_kv)
    indices = torch.where(flat_row % 2 == 0, with_invalid, indices)
    row = int(torch.nonzero((lens >= 32) & (torch.arange(T, device=dev) % 2 == 0))[0])
    indices[int(indptr[row]) : int(indptr[row]) + 16] = -1  # a whole leading tile
    scale = D**-0.5

    ref = _sparse_prefill_attn_torch(
        q, kv, indices, indptr, None, None, None, sink, scale
    )
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


# Head dims other than 512 take the autotuned launch on every arch; 576 also
# needs the dim mask (BLOCK_D=1024).
def test_pa_prefill_sparse_single_source_other_head_dims():
    _triton_branch_only()

    torch.manual_seed(1)
    D = 576
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
    _inject_invalid_slots(indices, num_kv)
    scale = D**-0.5

    ref = _sparse_prefill_attn_torch(
        q, kv, indices, indptr, None, None, None, sink, scale
    )
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# An empty pool leaves nothing to gather: every query gets the sink-only (zero) output.
def test_pa_prefill_sparse_empty_kv_pool():
    _triton_branch_only()
    T, H, D, dev = 64, 16, 512, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.empty(0, D, dtype=torch.bfloat16, device=dev)
    indptr = torch.arange(0, (T + 1) * 16, 16, dtype=torch.int32, device=dev)
    indices = torch.full((T * 16,), -1, dtype=torch.int32, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, D**-0.5)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.zeros_like(out))


# Strided or overlapping out= buffers would be written wrongly (gfx950 ignores
# out.stride(2)) or raced on (an expand() over heads), so they are rejected.
# An fp8 KV pool is rejected up front with a clear error instead of failing to
# compile inside tl.dot.
def test_pa_prefill_sparse_rejects_bad_inputs():
    _triton_branch_only()
    T, H, D = 4, 16, 512
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, D, dtype=torch.bfloat16, device="cuda")
    indptr = torch.arange(0, T + 1, dtype=torch.int32, device="cuda")
    indices = torch.zeros(T, dtype=torch.int32, device="cuda")
    outs = (
        torch.empty(T, D, H, dtype=q.dtype, device="cuda").transpose(1, 2),
        torch.empty(T, 1, D, dtype=q.dtype, device="cuda").expand(T, H, D),
    )
    for out in outs:
        with pytest.raises(AssertionError, match="contiguous"):
            pa_prefill_sparse(
                q, kv, indices, indptr, None, None, None, None, D**-0.5, out=out
            )
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        pa_prefill_sparse(
            q,
            kv.to(torch.float8_e4m3fnuz),
            indices,
            indptr,
            None,
            None,
            None,
            None,
            D**-0.5,
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


# int64 index buffers stay int64 on the Triton branch: slots past 2^31 - 1 must
# be rejected as out of pool, not wrapped into range by an int32 cast.
def test_pa_prefill_sparse_int64_indices():
    _triton_branch_only()
    torch.manual_seed(3)
    H, T, D, num_kv, dev = 16, 512, 512, 4096, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(num_kv, D, dtype=torch.bfloat16, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    lens = torch.randint(1, 640, (T,), device=dev)
    indptr = torch.zeros(T + 1, dtype=torch.int64, device=dev)
    indptr[1:] = lens.cumsum(0)
    indices = torch.randint(
        0, num_kv, (int(indptr[-1]),), dtype=torch.int64, device=dev
    )
    n = indices.numel()
    perm = torch.randperm(n, device=dev)
    indices[perm[: n // 8]] = -1
    # int32 casts of these would wrap to 0, 5 and num_kv - 1: valid-looking rows.
    indices[perm[n // 8 : n // 8 + n // 16]] = 2**32
    indices[perm[n // 8 + n // 16 : n // 8 + n // 8]] = 2**32 + 5
    indices[perm[n // 4 : n // 4 + n // 16]] = 2**32 + num_kv - 1
    scale = D**-0.5
    ref = _sparse_prefill_attn_torch(
        q, kv, indices, indptr, None, None, None, sink, scale
    )
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# The validity check must not rely on num_kv fitting int32. A real pool that
# large cannot be allocated in a test, so expand one KV row to the full row
# count (stride 0, no memory): every slot then reads the same row, and with a
# sink the output still depends on how many slots were accepted,
#   out = v * n e^s / (n e^s + e^sink),
# so a negative slot wrongly accepted as valid changes it.
def test_pa_prefill_sparse_huge_pool_rejects_negative_slots():
    _triton_branch_only()
    num_rows = 2**31 + 5
    torch.manual_seed(4)
    T, H, D, L, dev = 256, 16, 512, 64, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    row = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
    kv = row.expand(num_rows, D)
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    indptr = torch.arange(0, (T + 1) * L, L, dtype=torch.int32, device=dev)
    indices = torch.randint(0, 2**31 - 1, (T * L,), dtype=torch.int32, device=dev)
    indices[0::3] = -1
    indices[1::7] = -(2**31)
    scale = D**-0.5
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, scale)

    n = (indices.view(T, L) >= 0).sum(1).float()[:, None]  # accepted slots per query
    s_ = (q.float() @ row.float()[0]) * scale  # [T, H]: every key scores the same
    m = torch.maximum(s_, sink[None, :])
    w = n * torch.exp(s_ - m) / (n * torch.exp(s_ - m) + torch.exp(sink[None, :] - m))
    ref = (w[..., None] * row.float()[0]).to(q.dtype)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# A one-row pool: Triton specializes the integer 1 as a constant.
def test_pa_prefill_sparse_one_row_pool():
    _triton_branch_only()
    T, H, D, dev = 64, 16, 512, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    indptr = torch.arange(0, (T + 1) * 4, 4, dtype=torch.int32, device=dev)
    indices = torch.tensor([0, -1, 1, 0] * T, dtype=torch.int32, device=dev)
    ref = _sparse_prefill_attn_torch(
        q, kv, indices, indptr, None, None, None, sink, D**-0.5
    )
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, D**-0.5)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# Invalid slots are masked out of the gather, so a non-finite row the indices
# never select validly (row 0 here) must not reach the output. Loading it and
# giving it zero weight would not be enough: 0 * NaN is NaN in the PV dot.
def test_pa_prefill_sparse_nonfinite_row0_ignored_by_invalid_slots():
    _triton_branch_only()
    torch.manual_seed(5)
    H, T, D, num_kv, dev = 16, 256, 512, 1024, "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(num_kv, D, dtype=torch.bfloat16, device=dev)
    kv[0, ::2] = float("nan")
    kv[0, 1::2] = float("inf")
    sink = torch.randn(H, dtype=torch.float32, device=dev)
    indptr = torch.arange(0, (T + 1) * 48, 48, dtype=torch.int32, device=dev)
    indices = torch.randint(1, num_kv, (T * 48,), dtype=torch.int32, device=dev)
    indices[0::4] = -1
    indices[1::9] = num_kv + 3
    ref = _sparse_prefill_attn_torch(
        q, kv, indices, indptr, None, None, None, sink, D**-0.5
    )
    out = pa_prefill_sparse(q, kv, indices, indptr, None, None, None, sink, D**-0.5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
