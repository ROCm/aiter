# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for sparse_mla_fwd (gluon MLA, gfx950 and gfx942).

Covers both geometries: separated rope (DSV3.2, GLM-5.1, GLM-5.2) and
rope-free (GLM-5.3-Flash), where the query is the latent alone.
"""

import pytest
import torch

from aiter.ops.triton.utils._triton import arch_info

import aiter.ops.triton.attention.sparse_mla as smd
from aiter.ops.triton.attention.sparse_mla import (
    FP8_DOT_ARCHS,
    SUPPORTED_ARCHS,
    sparse_mla_fwd,
)

# OCP e4m3, not the arch-native fp8: these records are OCP by definition and the
# kernel's dequant reads OCP. gfx942's native fnuz would decode to garbage.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
KV_LORA, ROPE = 512, 64
D_QK = KV_LORA + ROPE


def _skip_unless_supported(dots="bf16"):
    arch = arch_info.get_arch()
    if arch not in SUPPORTED_ARCHS:
        pytest.skip(f"sparse_mla_fwd does not support {arch}")
    if dots == "fp8" and arch not in FP8_DOT_ARCHS:
        pytest.skip(f"fp8 matrix-core dots need OCP e4m3, absent on {arch}")


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
    # Built on CPU
    g = torch.Generator(device="cpu").manual_seed(seed)
    lens = (
        torch.randint(1, topk + 1, (C,), generator=g, device="cpu")
        if ragged
        else torch.full((C,), topk, device="cpu")
    )
    rows = [
        torch.randperm(pool, generator=g, device="cpu")[: int(lens[c])]
        for c in range(C)
    ]
    indptr = torch.zeros(C + 1, dtype=torch.int64, device="cpu")
    indptr[1:] = lens.cumsum(0)
    return (
        torch.cat(rows).to(torch.int32).to(device),
        indptr.to(torch.int32).to(device),
    )


def reference(q, kv_truth, indices, indptr, sm_scale):
    """f32 attention over the dequantized cache."""
    outs = []
    for c in range(q.shape[0]):
        kvs = kv_truth[indices[indptr[c] : indptr[c + 1]].long()].float()
        s = torch.einsum("hd,td->ht", q[c].float(), kvs) * sm_scale
        p = torch.softmax(s, dim=-1)
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
    fmt, C, H, topk, ragged, tol=2e-2, pool=1 << 16, rope=ROPE, **kwargs
):
    sm = (KV_LORA + rope) ** -0.5
    q, cache, ks, idx, ptr, truth = _build(fmt, C, H, topk, pool, ragged, rope=rope)
    ref = reference(q, truth.to(torch.bfloat16), idx, ptr, sm)
    out, _ = sparse_mla_fwd(
        q, cache, ptr, idx, sm, kv_scale=ks, qk_rope_head_dim=rope, **kwargs
    )
    e = rel_err(out, ref)
    assert e < tol, f"{fmt} C={C} H={H} topk={topk} ragged={ragged}: rel-err {e:.3e}"


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
    _skip_unless_supported(dots)
    _run_and_check(
        fmt, C=C, H=H, topk=topk, ragged=ragged, pool=pool, tol=tol, dot_precision=dots
    )


@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_dot_precision_arch_gate(arch):
    """The fp8-dot gate, for every arch, from any machine.

    The matrix above skips its fp8 cases off gfx950, so the gate has no
    coverage on any arch without this.
    """
    assert smd._resolve_dot_precision("bf16", "fp8_scalar", arch) is False
    if arch in FP8_DOT_ARCHS:
        assert smd._resolve_dot_precision("fp8", "fp8_scalar", arch) is True
    else:
        with pytest.raises(ValueError, match="fnuz"):
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


def test_lds_budget_gfx950_is_unchecked():
    """gfx950 is left to the launcher; the 64 KB table is gfx942-only."""
    smd._check_lds_budget("gfx950", 64, 2048, 64)


@pytest.mark.parametrize(
    "kv_lora_rank, rope, need",
    [
        (512, 0, None),
        (512, 64, None),
        (1000, 0, None),  # exactly 64 KB
        (1008, 0, 66048),
        (1024, 0, 67072),
        (1024, 64, 71680),
        (2048, 0, 132608),
    ],
)
def test_lds_budget_gfx942_boundary(kv_lora_rank, rope, need):
    """CPU-only: the gfx942 64 KB guard, including the measured rejects.

    The GPU suite only launches the default 512/64 geometry. These are the
    rope-free and separated-rope points from the OutOfResources sweep.
    """
    block_k = smd._arch_block_k("gfx942")
    if need is None:
        smd._check_lds_budget("gfx942", block_k, kv_lora_rank, rope)
        return
    with pytest.raises(ValueError, match=rf"needs {need} B of LDS"):
        smd._check_lds_budget("gfx942", block_k, kv_lora_rank, rope)


def test_ds_mla_format():
    _skip_unless_supported()
    _run_and_check("dsmla", C=8, H=16, topk=2048, ragged=True)


def reference_lse(q, kv_truth, indices, indptr, sm_scale):
    """Natural-log LSE per (query, head), matching mla_decode_fwd's convention."""
    rows = []
    for c in range(q.shape[0]):
        kvs = kv_truth[indices[indptr[c] : indptr[c + 1]].long()].float()
        s = torch.einsum("hd,td->ht", q[c].float(), kvs) * sm_scale
        rows.append(torch.logsumexp(s, dim=-1))
    return torch.stack(rows)


@pytest.mark.parametrize("fmt", ["bf16", "tensor"])
@pytest.mark.parametrize(
    "C,topk,ragged,splits",
    [(8, 2048, False, None), (8, 500, True, None), (2048, 256, True, 1)],
    ids=["split", "split_ragged", "nosplit"],
)
def test_return_lse(fmt, C, topk, ragged, splits):
    _skip_unless_supported()
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


def test_global_load_path():
    """A pool whose addressable span passes buffer_load's 2 GB offset limit."""
    _skip_unless_supported()
    live = 1 << 16
    sm = D_QK**-0.5
    q, cache, ks, idx, ptr, truth = _build("tensor", 8, 16, 2048, live, ragged=False)
    big = torch.empty(3_800_000, D_QK, dtype=cache.dtype, device=cache.device)
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
    _skip_unless_supported(dots)
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
