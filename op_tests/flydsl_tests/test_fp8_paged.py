# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-KV correctness of the vendored fp8 attention kernel (gfx950).

Paged addressing is only meaningfully tested when the page order is SCRAMBLED.
With an identity block table (page i holds tile i) the page pool is bit-identical
to a contiguous cache, so a kernel that ignored page ids entirely would pass.
Every case here permutes the pages, and one case is run twice under different
permutations of the same logical KV to confirm the output does not move.

The reference is `ref_paged_attn`, the same fp32 paged reference the Triton
unified-attention suite uses, fed the identical scrambled page pool and block
table. Comparing against the pre-quantization bf16 input would measure
quantization error, not kernel error.

Also covers paged + varlen (the production prefill configuration) and paged +
varlen + non-causal.

Not comparable against upstream FlyDSL: it rejects fp8 + paged at the interface.

Clear ~/.flydsl/cache before trusting a run after editing kernel helper classes:
the JIT cache key walks the launcher's closure for function dependencies and does
not resolve methods reached through instance attributes, so an edit to a helper
method hits a stale binary under an unchanged key.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available
from op_tests.triton_tests.attention.test_unified_attention import (
    ref_paged_attn,
)

try:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    _ARCH = get_gfx_runtime()
except Exception:  # noqa: BLE001
    _ARCH = None

pytestmark = [
    pytest.mark.skipif(_ARCH != "gfx950", reason=f"gfx950 only, got {_ARCH}"),
    pytest.mark.skipif(not is_flydsl_available(), reason="flydsl not available"),
]

DEV = "cuda"
PAGE = 64  # structurally BLOCK_N; not a free parameter
HEAD_DIM = 128
SEED = 123

# (label, B, S, HKV, causal). H is fixed at 64 (GQA 16:1, the trace's ratio)
# unless HKV==H (the MHA case).
CASES = [
    ("aligned S=1024", 1, 1024, 4, True),
    ("unaligned S=1000", 1, 1000, 4, True),  # partial final page
    ("non-causal S=1024", 1, 1024, 4, False),  # every page read by every row
    ("batch=2 S=512", 2, 512, 4, True),  # per-batch block table rows
    ("batch=4 S=768", 4, 768, 4, True),
    ("MHA S=2048", 1, 2048, 8, True),  # HKV=H, different stride_kv_n
    ("short S=64", 1, 64, 4, True),  # single page
    ("S=4096", 1, 4096, 4, True),
]

# Paged + varlen: the production prefill shape. Page pools are per-sequence and
# the block table is [B, max_pages], so short sequences leave unused trailing
# columns -- the same padding a real serving allocator produces.
PAGED_VARLEN_CASES = [
    [1024],
    [512, 1536],
    [4023, 1384],  # trace-source shapes, packed
    [1, 3, 31, 33, 63, 65],  # all << BLOCK_M, sharpest active-guard test
    [2048, 512, 1024],
]


def _build_module():
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )

    return build


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def to_pages(kf, npages_per_seq, perm, hkv, d):
    """Scatter a contiguous [B, S, HKV, D] fp8 tensor into a permuted page pool.

    Returns the pool as [total_pages, PAGE, HKV, D]. `perm[b][t]` is the pool
    index holding batch b's logical tile t, so the pool is in no useful order --
    a kernel that read it linearly would get garbage.
    """
    b, s, _, _ = kf.shape
    total = b * npages_per_seq
    pool = torch.zeros(total, PAGE, hkv, d, device=DEV, dtype=kf.dtype)
    padded = torch.zeros(b, npages_per_seq * PAGE, hkv, d, device=DEV, dtype=kf.dtype)
    padded[:, :s] = kf
    for bi in range(b):
        for t in range(npages_per_seq):
            pool[perm[bi][t]] = padded[bi, t * PAGE : (t + 1) * PAGE]
    return pool


def _run_paged(build, b, s, h, hkv, d, causal, seed, perm=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda n: torch.randn(
        b, s, n, d, generator=g, device=DEV, dtype=torch.bfloat16
    )
    qf, qs = q8(mk(h))
    kf, ks = q8(mk(hkv))
    vf, vs = q8(mk(hkv))

    npages = (s + PAGE - 1) // PAGE
    if perm is None:
        pg = torch.Generator().manual_seed(seed + 7)
        flat = torch.randperm(b * npages, generator=pg).tolist()
        perm = [flat[bi * npages : (bi + 1) * npages] for bi in range(b)]

    kpool = to_pages(kf, npages, perm, hkv, d)
    vpool = to_pages(vf, npages, perm, hkv, d)
    block_table = torch.tensor(perm, device=DEV, dtype=torch.int32)

    mod = build(
        num_heads=h,
        head_dim=d,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=hkv,
        paged=True,
    )
    o = torch.empty(b, s, h, d, device=DEV, dtype=torch.bfloat16)
    mod(
        qf.contiguous().view(-1),
        kpool.contiguous().view(-1),
        vpool.contiguous().view(-1),
        o.contiguous().view(-1),
        b,
        s,
        block_table=block_table.contiguous().view(-1),
        block_table_stride=npages,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
    )
    torch.cuda.synchronize()

    want = ref_paged_attn(
        query=qf.reshape(b * s, h, d),
        key_cache=kpool,
        value_cache=vpool,
        query_lens=[s] * b,
        kv_lens=[s] * b,
        block_tables=block_table,
        scale=d**-0.5,
        out_dtype=torch.float32,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=1 if causal else 0,
    )
    of = o.float().reshape(b * s, h, d)
    err = (of - want).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), want.flatten(), dim=0
    ).item()
    # Per-row worst error: an aggregate cosine can stay high while whole
    # sequences or trailing pages are wrong.
    bad = int(((of - want).abs().amax(dim=(1, 2)) > 1e-1).sum().item())
    return o, err, cos, bad


def _run_paged_varlen(build, seqs, d, seed, causal=True):
    """Packed ragged batch over a scrambled page pool, vs `ref_paged_attn`."""
    h, hkv = 64, 4
    total, b = sum(seqs), len(seqs)
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda n: torch.randn(
        total, n, d, generator=g, device=DEV, dtype=torch.bfloat16
    )
    qf, qs = q8(mk(h))
    kf, ks = q8(mk(hkv))
    vf, vs = q8(mk(hkv))
    cu = torch.tensor(
        [0] + list(torch.tensor(seqs).cumsum(0)), device=DEV, dtype=torch.int32
    )

    npages_per = [(ln + PAGE - 1) // PAGE for ln in seqs]
    stride = max(npages_per)
    tot_pages = sum(npages_per)
    pg = torch.Generator().manual_seed(seed + 11)
    flat = torch.randperm(tot_pages, generator=pg).tolist()
    pk = torch.zeros(tot_pages, PAGE, hkv, d, device=DEV, dtype=kf.dtype)
    pv = torch.zeros_like(pk)
    bt = torch.zeros(b, stride, device=DEV, dtype=torch.int32)
    c = 0
    for bi, ln in enumerate(seqs):
        lo = cu[bi].item()
        for t in range(npages_per[bi]):
            pid = flat[c]
            c += 1
            s0, s1 = lo + t * PAGE, min(lo + (t + 1) * PAGE, lo + ln)
            n = s1 - s0
            pk[pid, :n] = kf[s0:s1]
            pv[pid, :n] = vf[s0:s1]
            bt[bi, t] = pid

    mod = build(
        num_heads=h,
        head_dim=d,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=hkv,
        varlen=True,
        paged=True,
    )
    o = torch.empty(total, h, d, device=DEV, dtype=torch.bfloat16)
    mod(
        qf.contiguous().view(-1),
        pk.contiguous().view(-1),
        pv.contiguous().view(-1),
        o.contiguous().view(-1),
        b,
        max(seqs),
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        block_table=bt.contiguous().view(-1),
        block_table_stride=stride,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
    )
    torch.cuda.synchronize()

    want = ref_paged_attn(
        query=qf,
        key_cache=pk,
        value_cache=pv,
        query_lens=list(seqs),
        kv_lens=list(seqs),
        block_tables=bt,
        scale=d**-0.5,
        out_dtype=torch.float32,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=1 if causal else 0,
    )
    of = o.float()
    err = (of - want).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), want.flatten(), dim=0
    ).item()
    bad = int(((of - want).abs().amax(dim=(1, 2)) > 1e-1).sum().item())
    return err, cos, bad


def _check(err, cos, bad):
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"


@pytest.mark.parametrize("label,b,s,hkv,causal", CASES, ids=[c[0] for c in CASES])
def test_paged(label, b, s, hkv, causal):
    build = _build_module()
    h = 64 if hkv == 4 else hkv
    _, err, cos, bad = _run_paged(build, b, s, h, hkv, HEAD_DIM, causal, SEED)
    _check(err, cos, bad)
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "seqs", PAGED_VARLEN_CASES, ids=[str(s) for s in PAGED_VARLEN_CASES]
)
def test_paged_varlen(seqs):
    """Paged + varlen: the production prefill configuration."""
    build = _build_module()
    err, cos, bad = _run_paged_varlen(build, seqs, HEAD_DIM, SEED)
    _check(err, cos, bad)
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "seqs", PAGED_VARLEN_CASES, ids=[str(s) for s in PAGED_VARLEN_CASES]
)
def test_paged_varlen_noncausal(seqs):
    """Paged + varlen + non-causal: the last uncovered combination once the
    adapter's causal-only gate is relaxed."""
    build = _build_module()
    err, cos, bad = _run_paged_varlen(build, seqs, HEAD_DIM, SEED, causal=False)
    _check(err, cos, bad)
    torch.cuda.empty_cache()


def test_permutation_invariance():
    """Same logical KV, two different page layouts. Isolates page-id handling
    from every other source of error -- if these differ, the kernel is reading
    the pool by position somewhere."""
    build = _build_module()
    npages = 1024 // PAGE
    ident = [list(range(npages))]
    rev = [list(reversed(range(npages)))]
    o1, _, _, _ = _run_paged(build, 1, 1024, 64, 4, HEAD_DIM, True, SEED, perm=ident)
    o2, _, _, _ = _run_paged(build, 1, 1024, 64, 4, HEAD_DIM, True, SEED, perm=rev)
    assert torch.equal(o1, o2), (
        "outputs differ across page permutations -- the kernel is reading the "
        "page pool by position somewhere"
    )
    torch.cuda.empty_cache()
