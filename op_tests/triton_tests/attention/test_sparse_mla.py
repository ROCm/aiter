# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for sparse_mla_fwd (gluon MLA, gfx950 and gfx942).

Covers both geometries: separated rope (DSV3.2, GLM-5.1, GLM-5.2) and
rope-free (GLM-5.3-Flash), where the query is the latent alone.
"""

import pytest
import torch

import aiter.ops.triton.attention.sparse_mla as smd
from aiter.ops.triton.attention.sparse_mla import (
    FNUZ_ARCHS,
    FP8_ARCHS,
    FP8_SCALAR_ARCHS,
    SUPPORTED_ARCHS,
    sparse_mla_fwd,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.triton_tests.attention.test_pa_decode_sparse import _inv_rope_ref

# The arch-native fp8, as a producer on this machine writes it, which is the
# encoding the kernel reads.
FP8_DTYPE = get_fp8_e4m3_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max
KV_LORA, ROPE = 512, 64
D_QK = KV_LORA + ROPE


def _skip_unless_supported(dots="bf16", fmt="bf16"):
    arch = arch_info.get_arch()
    if arch not in SUPPORTED_ARCHS:
        pytest.skip(f"sparse_mla_fwd does not support {arch}")
    if dots == "fp8" and arch not in FP8_SCALAR_ARCHS:
        pytest.skip(f"fp8 dots are {'/'.join(FP8_SCALAR_ARCHS)}-only")
    if fmt == "tensor" and arch not in FP8_SCALAR_ARCHS:
        pytest.skip(f"the fp8_scalar cache is {'/'.join(FP8_SCALAR_ARCHS)}-only")
    if fmt == "dsmla" and arch not in FP8_ARCHS:
        pytest.skip(f"the fp8_dsv32_mla cache is {'/'.join(FP8_ARCHS)}-only")


def quantize_flat_fp8(kv):
    """vLLM's production layout: whole row fp8 with one per-tensor scale."""
    scale = (kv.float().abs().amax() / FP8_MAX).clamp_min(1e-30).reshape(1)
    q8 = (kv.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q8.view(torch.uint8), scale.to(torch.float32)


def pack_ds_mla(kv, block_size=64):
    """vLLM fp8_ds_mla: 656 B rows [512 fp8 | 4 x f32 per-128 | 64 bf16 rope]."""
    T = kv.shape[0]
    nb = (T + block_size - 1) // block_size
    cache = torch.zeros(nb * block_size, 656, dtype=torch.uint8, device=kv.device)
    x = kv[:, :KV_LORA].float().reshape(T, 4, 128)
    scale = (x.abs().amax(-1) / FP8_MAX).clamp_min(1e-30)
    q8 = (x / scale[..., None]).to(FP8_DTYPE).view(torch.uint8)
    cache[:T, :KV_LORA] = q8.reshape(T, KV_LORA)
    cache[:T, 512:528].view(torch.float32).copy_(scale)
    cache[:T, 528:].view(torch.bfloat16).copy_(kv[:, KV_LORA:])
    return cache.view(nb, block_size, 656)


def dequant_flat_fp8(u8, scale):
    return u8.view(FP8_DTYPE).float() * scale


def dequant_ds_mla(cache):
    flat = cache.reshape(-1, 656)
    x = flat[:, :KV_LORA].view(FP8_DTYPE).float().reshape(-1, 4, 128)
    sc = flat[:, 512:528].view(torch.float32)
    nope = (x * sc[..., None]).reshape(-1, KV_LORA)
    rope = flat[:, 528:].view(torch.bfloat16).float()
    return torch.cat([nope, rope], dim=-1)


def make_indices(C, pool, topk, device, seed, ragged):
    g = torch.Generator(device=device).manual_seed(seed)
    lens = (
        torch.randint(1, topk + 1, (C,), generator=g, device=device)
        if ragged
        else torch.full((C,), topk, device=device)
    )
    rows = [torch.randperm(pool, generator=g, device=device)[:n] for n in lens.tolist()]
    indptr = torch.zeros(C + 1, dtype=torch.int64, device=device)
    indptr[1:] = lens.cumsum(0)
    return (
        torch.cat(rows).to(torch.int32),
        indptr.to(torch.int32),
    )


def reference(q, kv_truth, indices, indptr, sm_scale, sink=None):
    """f32 attention over the dequantized cache. sink: [H] logits of a virtual key
    in the softmax denominator. A row with no key and no sink comes out 0."""
    outs = []
    for c in range(q.shape[0]):
        kvs = kv_truth[indices[indptr[c] : indptr[c + 1]].long()].float()
        s = torch.einsum("hd,td->ht", q[c].float(), kvs) * sm_scale
        if sink is not None:
            s = torch.cat([s, sink.float()[:, None]], dim=-1)
        if s.shape[1] == 0:
            outs.append(torch.zeros(q.shape[1], KV_LORA, device=q.device))
            continue
        p = torch.softmax(s, dim=-1)[:, : kvs.shape[0]]
        outs.append(torch.einsum("ht,td->hd", p, kvs[:, :KV_LORA]))
    return torch.stack(outs)


def rel_err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def _build(fmt, C, H, topk, pool, ragged, seed=0, rope=ROPE):
    torch.manual_seed(seed)
    d_qk = KV_LORA + rope
    q = torch.randn(C, H, d_qk, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(pool, d_qk, dtype=torch.bfloat16, device="cuda") * 0.125
    idx, ptr = make_indices(C, pool, topk, "cuda", seed + 7, ragged)
    ks = None
    if fmt == "bf16":
        cache, truth = kv, kv.float()
    elif fmt == "tensor":
        cache, ks = quantize_flat_fp8(kv)
        truth = dequant_flat_fp8(cache, ks)
    else:
        cache = pack_ds_mla(kv)
        truth = dequant_ds_mla(cache)[:pool]
    return q, cache, ks, idx, ptr, truth


def _run_and_check(
    fmt,
    C,
    H,
    topk,
    ragged,
    tol=2e-2,
    pool=1 << 16,
    rope=ROPE,
    q_fp8=False,
    sink=False,
    block=None,
    empty_row=False,
    check_lse=False,
    **kwargs,
):
    """q_fp8: pass q quantized with a per-tensor q_scale; sink: a per-head sink;
    block: read the cache as [nb, block, row] pages with a padded pitch;
    empty_row: the first query has no key (must come out 0); check_lse: also check
    the returned LSE."""
    sm = (KV_LORA + rope) ** -0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, C, H, topk, pool, ragged, rope=rope)
    if empty_row:
        idx, ptr = idx[ptr[1] :], (ptr - ptr[1]).clamp_min(0)
    attn_sink = torch.randn(H, device="cuda") if sink else None
    q_scale = None
    if q_fp8:
        q_scale = (q.float().abs().amax() / FP8_MAX).reshape(1)
        q = (q.float() / q_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    if block is not None:
        # a padded page pitch: packed pages would be read as one flat pool
        pages = cache.view(-1, block, cache.shape[-1])
        cache = pages.new_zeros(pages.shape[0], block + 1, pages.shape[2])[:, :block]
        cache.copy_(pages)
    q_ref = q.float() * q_scale if q_fp8 else q
    ref = reference(q_ref, truth.to(torch.bfloat16), idx, ptr, sm, attn_sink)
    out, lse = sparse_mla_fwd(
        q,
        cache,
        ptr,
        idx,
        sm,
        kv_scale=ks,
        qk_rope_head_dim=rope,
        q_scale=q_scale,
        attn_sink=attn_sink,
        return_lse=check_lse,
        **kwargs,
    )
    assert not out.isnan().any()
    if empty_row:
        assert (out[0] == 0).all(), "a query with no key must come out 0"
    e = rel_err(out, ref)
    assert e < tol, f"{fmt} C={C} H={H} topk={topk} ragged={ragged}: rel-err {e:.3e}"
    if check_lse:
        truth16 = truth.to(torch.bfloat16)
        ref_lse = reference_lse(q_ref, truth16, idx, ptr, sm, attn_sink)
        e = (lse - ref_lse).abs().max().item() / ref_lse.abs().max().item()
        assert e < 2e-2, f"{fmt} C={C} H={H}: lse rel-err {e:.3e}"


@pytest.mark.parametrize(
    "fmt,dots,tol",
    [("bf16", "bf16", 2e-2), ("tensor", "bf16", 2e-2), ("tensor", "fp8", 7e-2)],
    ids=["bf16", "tensor", "tensor_fp8"],
)
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize(
    "C,topk,ragged,pool",
    [(8, 2048, False, 1 << 16), (8, 500, True, 1 << 16), (2048, 256, True, 1 << 13)],
    ids=["topk2048", "ragged500", "prefill"],
)
def test_sparse_mla(fmt, dots, tol, H, C, topk, ragged, pool):
    _skip_unless_supported(dots, fmt)
    _run_and_check(
        fmt, C=C, H=H, topk=topk, ragged=ragged, pool=pool, tol=tol, dot_precision=dots
    )


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_dot_precision_arch_gate(arch):
    """The fp8-dot gate, for every arch, from any machine.

    The matrix above only runs its fp8 cases on the arch it is on.
    """
    assert smd._resolve_dot_precision("bf16", "fp8_scalar", arch) is False
    if arch in FP8_SCALAR_ARCHS:
        assert smd._resolve_dot_precision("fp8", "fp8_scalar", arch) is True
        with pytest.raises(ValueError, match="needs an fp8 cache"):
            smd._resolve_dot_precision("fp8", "bf16", arch)
    else:
        with pytest.raises(ValueError, match="not supported"):
            smd._resolve_dot_precision("fp8", "fp8_scalar", arch)


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_packed_cache_arch_gate(arch):
    """Packed caches are gfx950-only even where SUPPORTED_ARCHS is wider.

    They return to pa_decode_sparse before the kernel's own arch gate, so
    widening that gate does not reach them.
    """
    if arch in smd.PACKED_ARCHS:
        smd._check_packed_arch(arch)
    else:
        with pytest.raises(ValueError, match="fp8_dsv4_mla"):
            smd._check_packed_arch(arch)


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_fp8_arch_gate(arch):
    """fp8 q and fp8 caches, for every arch, from any machine."""
    smd._check_fp8_arch(arch, "bf16", torch.bfloat16)
    if arch in FP8_SCALAR_ARCHS:
        smd._check_fp8_arch(arch, "fp8_scalar", torch.bfloat16)
    for fmt, q_dtype in (
        ("fp8_dsv32_mla", torch.bfloat16),
        ("bf16", torch.float8_e4m3fn),
    ):
        if arch in FP8_ARCHS:
            smd._check_fp8_arch(arch, fmt, q_dtype)
        else:
            with pytest.raises(ValueError, match="takes bf16 q"):
                smd._check_fp8_arch(arch, fmt, q_dtype)


def test_native_dsmla_cache_rejected():
    """This arch's own fp8_dsv32_mla records behind a uint8 view, through the
    public wrapper, where only fp8_scalar is supported. Going through
    sparse_mla_fwd also catches the gate losing its one call site.
    """
    arch = arch_info.get_arch()
    if arch not in SUPPORTED_ARCHS or arch in FP8_ARCHS:
        pytest.skip(f"fp8_dsv32_mla runs on {arch}")
    q, cache, ks, idx, ptr, _ = _build("dsmla", 1, 16, 64, 1024, ragged=False)
    assert cache.dtype == torch.uint8
    with pytest.raises(ValueError, match="takes bf16 q"):
        sparse_mla_fwd(q, cache, ptr, idx, D_QK**-0.5, kv_scale=ks)


def test_foreign_fp8_cache_dtype_rejected():
    """A typed fp8 cache in the other arch's encoding is refused, not misread."""
    arch = arch_info.get_arch()
    if arch not in FP8_SCALAR_ARCHS:
        pytest.skip(f"no fp8 cache runs on {arch}")
    q, cache, ks, idx, ptr, _ = _build("tensor", 1, 16, 64, 1024, ragged=False)
    foreign = (
        torch.float8_e4m3fn
        if FP8_DTYPE == torch.float8_e4m3fnuz
        else torch.float8_e4m3fnuz
    )
    with pytest.raises(ValueError, match=f"is not {arch}'s fp8"):
        sparse_mla_fwd(q, cache.view(foreign), ptr, idx, D_QK**-0.5, kv_scale=ks)


@pytest.mark.parametrize(
    "dtype", [torch.float8_e5m2, torch.float8_e5m2fnuz, torch.int8, torch.bool]
)
def test_non_e4m3_byte_cache_dtype_rejected(dtype):
    """Other one-byte dtypes are refused rather than decoded as e4m3."""
    arch = arch_info.get_arch()
    if arch not in FP8_SCALAR_ARCHS:
        pytest.skip(f"no fp8 cache runs on {arch}")
    q, cache, ks, idx, ptr, _ = _build("tensor", 1, 16, 64, 1024, ragged=False)
    with pytest.raises(ValueError, match="unsupported cache dtype"):
        sparse_mla_fwd(q, cache.view(dtype), ptr, idx, D_QK**-0.5, kv_scale=ks)


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_launch_config_published(arch):
    """Every supported arch ships its launch config, checked from any machine."""
    assert {"BLOCK_K", "num_warps"} <= smd._get_config(arch).keys()
    assert {"BLOCK_K", "num_warps"} <= smd._get_config(arch, fp8_dots=True).keys()


def test_lds_budget_gfx950_is_unchecked():
    """gfx950 is left to the launcher, though arch_info lists its LDS too.

    The footprint model holds only for gfx942's non-async tiles.
    """
    smd._check_lds_budget("gfx950", 64, 2048, 64, kv_lds_pad=16)


@pytest.mark.parametrize(
    "kv_lds_pad, kv_lora_rank, rope, need",
    [
        # decode: the KV tile keeps the kernel's default pad
        (0, 512, 0, None),
        (0, 512, 64, None),
        (0, 512, 256, None),
        (0, 1000, 0, None),  # exactly 64 KB
        (0, 1008, 0, 66048),
        (0, 512, 512, 67584),
        (0, 1024, 0, 67072),
        (0, 1024, 64, 71680),
        (0, 2048, 0, 132608),
        # prefill: the launch pads the KV tile's rows by 16
        (16, 512, 0, None),
        (16, 512, 64, None),
        (16, 512, 256, None),
        (16, 992, 0, None),  # exactly 64 KB
        (16, 1000, 0, 66048),
        (16, 512, 512, 68096),
        (16, 1024, 0, 67584),
        (16, 1024, 64, 72192),
        (16, 2048, 0, 133120),
    ],
)
def test_lds_budget_gfx942_boundary(kv_lds_pad, kv_lora_rank, rope, need):
    """CPU-only: the gfx942 64 KB guard at the decode and prefill KV pads.

    The GPU suite only launches the default 512/64 geometry. The power-of-two
    rejects are the byte counts OutOfResources reported for them. 992, 1000 and
    1008 only pin the comparison at exactly 64 KB: the KV tile's shared layout
    takes a power-of-two width, so no launch reaches that point.
    """
    block_k = smd._get_config("gfx942")["BLOCK_K"]
    if need is None:
        smd._check_lds_budget("gfx942", block_k, kv_lora_rank, rope, kv_lds_pad)
        return
    with pytest.raises(ValueError, match=rf"needs {need} B of LDS"):
        smd._check_lds_budget("gfx942", block_k, kv_lora_rank, rope, kv_lds_pad)


@pytest.mark.parametrize(
    "kv_lora_rank, rope, need",
    [(512, 0, None), (512, 64, None), (1024, 0, 67584), (512, 512, 68608)],
)
def test_lds_budget_gfx942_fp8_tile(kv_lora_rank, rope, need):
    """CPU-only: fp8 dots' one-byte tiles, at their own BLOCK_K on gfx942.

    512/0 and 512/64 compile to the 34816 and 39936 B the model gives them.
    """
    block_k = smd._get_config("gfx942", fp8_dots=True)["BLOCK_K"]
    args = ("gfx942", block_k, kv_lora_rank, rope, 0)
    if need is None:
        smd._check_lds_budget(*args, fp8_dots=True)
        return
    with pytest.raises(ValueError, match=rf"needs {need} B of LDS"):
        smd._check_lds_budget(*args, fp8_dots=True)


def test_ds_mla_format():
    _skip_unless_supported(fmt="dsmla")
    _run_and_check("dsmla", C=8, H=16, topk=2048, ragged=True)


def reference_lse(q, kv_truth, indices, indptr, sm_scale, sink=None):
    """Natural-log LSE per (query, head), matching mla_decode_fwd's convention;
    sink as in reference."""
    rows = []
    for c in range(q.shape[0]):
        kvs = kv_truth[indices[indptr[c] : indptr[c + 1]].long()].float()
        s = torch.einsum("hd,td->ht", q[c].float(), kvs) * sm_scale
        if sink is not None:
            s = torch.cat([s, sink.float()[:, None]], dim=-1)
        rows.append(torch.logsumexp(s, dim=-1))
    return torch.stack(rows)


@pytest.mark.parametrize("fmt", ["bf16", "tensor"])
@pytest.mark.parametrize(
    "C,topk,ragged,splits",
    [(8, 2048, False, None), (8, 500, True, None), (2048, 256, True, 1)],
    ids=["split", "split_ragged", "nosplit"],
)
def test_return_lse(fmt, C, topk, ragged, splits):
    _skip_unless_supported(fmt=fmt)
    H, pool = 16, 1 << 16
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, C, H, topk, pool, ragged)
    kwargs = {} if splits is None else {"kv_splits": splits}
    out, lse = sparse_mla_fwd(
        q, cache, ptr, idx, sm, kv_scale=ks, return_lse=True, **kwargs
    )
    assert lse is not None
    assert lse.shape == (C, H) and lse.dtype == torch.float32
    ref = reference_lse(q, truth.to(torch.bfloat16), idx, ptr, sm)
    e = (lse - ref).abs().max().item() / ref.abs().max().item()
    assert e < 2e-2, f"{fmt} C={C} splits={splits}: lse rel-err {e:.3e}"
    # the output must not change just because the LSE was asked for
    out_no, lse_no = sparse_mla_fwd(q, cache, ptr, idx, sm, kv_scale=ks, **kwargs)
    assert lse_no is None
    assert torch.equal(out.view(torch.int16), out_no.view(torch.int16))


@pytest.mark.parametrize("fmt", ["bf16", "tensor"])
def test_global_load_path(fmt):
    """A pool whose addressable span passes buffer_load's 2 GB offset limit."""
    _skip_unless_supported(fmt=fmt)
    live = 1 << 16
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, 8, 16, 2048, live, ragged=False)
    rows = 3_800_000 // cache.element_size()
    big = torch.empty(rows, D_QK, dtype=cache.dtype, device=cache.device)
    big[:live] = cache
    assert smd.max_addressable_bytes(big) >= 2**31 - 1  # past buffer_load's offset
    out, _ = sparse_mla_fwd(q, big, ptr, idx, sm, kv_scale=ks)
    e = rel_err(out, reference(q, truth.to(torch.bfloat16), idx, ptr, sm))
    assert e < 2e-2, f"global load path: rel-err {e:.3e}"


@pytest.mark.parametrize(
    "fmt,dots,tol",
    [("bf16", "bf16", 2e-2), ("tensor", "bf16", 2e-2), ("tensor", "fp8", 7e-2)],
    ids=["bf16", "tensor", "tensor_fp8"],
)
@pytest.mark.parametrize("H", [8, 16])
@pytest.mark.parametrize(
    "C,topk,ragged,pool",
    [(8, 2048, False, 1 << 16), (8, 500, True, 1 << 16), (2048, 256, True, 1 << 13)],
    ids=["topk2048", "ragged500", "prefill"],
)
def test_sparse_mla_rope_free(fmt, dots, tol, H, C, topk, ragged, pool):
    _skip_unless_supported(dots, fmt)
    _run_and_check(
        fmt,
        C=C,
        H=H,
        topk=topk,
        ragged=ragged,
        pool=pool,
        tol=tol,
        rope=0,
        dot_precision=dots,
    )


@pytest.mark.parametrize("H", [8, 16, 32, 64, 128])
@pytest.mark.parametrize(
    "C,topk,ragged,pool",
    [(8, 2048, False, 1 << 16), (8, 500, True, 1 << 16), (512, 256, True, 1 << 13)],
    ids=["topk2048", "ragged500", "prefill"],
)
@pytest.mark.parametrize(
    "opts",
    [
        {},
        {"q_fp8": True, "sink": True, "check_lse": True},
        {"block": 64, "kv_splits": 3, "empty_row": True},
        {"rope": ROPE, "q_fp8": True},
    ],
    ids=["plain", "q_fp8_sink_lse", "paged_splits_empty", "rope"],
)
def test_sparse_mla_fp8_kernel(H, C, topk, ragged, pool, opts):
    """fp8_scalar with fp8 dots: gfx950's fp8 kernel (sparse_mla_fp8.py), at every
    head count it takes."""
    _skip_unless_supported("fp8", "tensor")
    _run_and_check(
        "tensor",
        C=C,
        H=H,
        topk=topk,
        ragged=ragged,
        pool=pool,
        tol=7e-2,
        dot_precision="fp8",
        **{"rope": 0, **opts},
    )


@pytest.mark.parametrize("one_alloc", [True, False], ids=["one_alloc", "two_allocs"])
@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("splits", [1, 3])
def test_sparse_mla_fp8_two_caches(one_alloc, mxfp8, splits):
    """DSv4's SWA + top-k caches with the fused inverse-RoPE (and MXFP8) output on
    the fp8 kernel: one allocation under one scale (one loop) or two with their own
    scales (two loops)."""
    _skip_unless_supported("fp8", "tensor")
    C, H, B, nb = 8, 64, 64, 512
    sm = KV_LORA**-0.5
    torch.manual_seed(0)
    q = torch.randn(C, H, KV_LORA, dtype=torch.bfloat16, device="cuda") * 0.125
    kv = torch.randn(2, nb * B, KV_LORA, dtype=torch.bfloat16, device="cuda") * 0.125
    if one_alloc:  # pages interleaved in one allocation, as vLLM allocates them
        u8, ks = quantize_flat_fp8(kv)
        pool = torch.stack([u8[0].view(nb, B, -1), u8[1].view(nb, B, -1)], dim=1)
        (main, extra), x_ks = (pool[:, 0], pool[:, 1]), ks
    else:
        (u8, ks), (x_u8, x_ks) = quantize_flat_fp8(kv[0]), quantize_flat_fp8(kv[1] * 4)
        main, extra = u8.view(nb, B, -1), x_u8.view(nb, B, -1)
    truth = torch.cat([dequant_flat_fp8(main, ks), dequant_flat_fp8(extra, x_ks)])
    m_idx, m_ptr = make_indices(C, nb * B, 128, "cuda", 1, ragged=False)
    x_idx, x_ptr = make_indices(C, nb * B, 256, "cuda", 2, ragged=False)
    idx = torch.cat([m_idx.view(C, -1), x_idx.view(C, -1) + nb * B], 1).flatten()
    sink = torch.randn(H, device="cuda")
    truth = truth.reshape(-1, KV_LORA).to(torch.bfloat16)
    ref = reference(q, truth, idx, m_ptr + x_ptr, sm, sink)
    ang = torch.rand(4096, 32, device="cuda") * 6.2831853
    cos_sin = torch.cat([ang.cos(), ang.sin()], dim=1)
    pos = torch.randint(0, 4096, (C,), device="cuda")
    ref = _inv_rope_ref(ref, pos, cos_sin)
    data = torch.empty(C, H * KV_LORA, dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty(C, H * KV_LORA // 32, dtype=torch.uint8, device="cuda")
    out, _ = sparse_mla_fwd(
        q,
        main,
        m_ptr,
        m_idx,
        sm,
        kv_scale=ks,
        qk_rope_head_dim=0,
        dot_precision="fp8",
        attn_sink=sink,
        kv_splits=splits,
        extra_kv=extra,
        extra_indptr=x_ptr,
        extra_indices=x_idx,
        extra_kv_scale=x_ks,
        inv_rope_positions=pos,
        inv_rope_cos_sin_cache=cos_sin,
        out_mxfp8=(data, scale) if mxfp8 else None,
    )
    if mxfp8:
        step = torch.exp2(scale.float() - 127.0).view(C, -1, 1)
        out = (data.float().view(C, -1, 32) * step).view(C, H, KV_LORA)
    e = rel_err(out, ref)
    assert e < 0.1, f"{one_alloc=} {mxfp8=} {splits=}: rel-err {e:.3e}"


def test_fp8_dots_keep_the_softmax_tail():
    """Each query's first key scores 9 above the other 2047, which carry a fifth
    of the output at p ~ 1e-4 each. That is below fnuz's smallest subnormal
    unless P_SCALE lifts p before the PV dot, and with split-K off a single
    program quantizes every tail p after the max.
    """
    arch = arch_info.get_arch()
    if arch not in FNUZ_ARCHS:
        pytest.skip(f"P_SCALE is 1 on {arch}")
    C, H, topk, pool = 4, 16, 2048, 1 << 13
    sm = KV_LORA**-0.5
    g = torch.Generator().manual_seed(0)
    # Head h reads axis h alone, where the dominant key is 1 (score 9) and the
    # tail keys are 0 (score 0). Elsewhere the tail keys share one direction.
    q = torch.zeros(C, H, KV_LORA)
    q[:, range(H), range(H)] = 9.0 / sm
    u = torch.randn(KV_LORA, generator=g)
    kv = torch.nn.functional.normalize(
        u + 0.3 * torch.randn(pool, KV_LORA, generator=g), dim=-1
    )
    kv = kv * KV_LORA**0.5
    kv[:, :H] = 0
    kv[-1] = 0
    kv[-1, :H] = 1
    idx = torch.stack(
        [
            torch.cat([torch.tensor([pool - 1]), torch.randperm(pool - 1, generator=g)])
            for _ in range(C)
        ]
    )[:, :topk]
    idx = idx.flatten().to(torch.int32).cuda()
    ptr = torch.arange(0, (C + 1) * topk, topk, dtype=torch.int32, device="cuda")
    q = q.to(torch.bfloat16).cuda()
    cache, ks = quantize_flat_fp8(kv.cuda())
    ref = reference(q, dequant_flat_fp8(cache, ks).to(torch.bfloat16), idx, ptr, sm)
    out, _ = sparse_mla_fwd(
        q,
        cache,
        ptr,
        idx,
        sm,
        kv_scale=ks,
        qk_rope_head_dim=0,
        dot_precision="fp8",
        kv_splits=1,
    )
    e = rel_err(out, ref)
    assert e < 5e-2, f"softmax tail: rel-err {e:.3e}"
