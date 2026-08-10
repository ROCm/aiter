# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed + split-K decode path of the vendored fp8 kernel (gfx950).

The committed split-K path was welded to the UNPACKED M factoring (gqa_pack_m
auto-off for num_kv_splits > 1). This test forces gqa_pack_m=True together with
num_kv_splits>1 -- the true Triton-3d analog the build refusal used to forbid --
and checks it against a torch reference.

Decode-only shapes: Sq=1 against a long KV context (cross_seqlen), GQA 16:1 so
packing is actually exercised (16 q-heads ride in M per kv head). Both non-paged
and paged (the production decode configuration).

Bad-row count is the signal, not aggregate cosine: the original split-K defect
left single-range rows exactly right and multi-range rows wrong, a binary split
the aggregate cosine hides. The dense cases feed contiguous KV with no block
table, which `ref_paged_attn` cannot express, so this file keeps its own fp32
reference for a single consistent oracle across dense and paged cases.

Clear ~/.flydsl/cache before trusting a run after a kernel edit -- the JIT cache
key does not resolve helper-class methods reached through instance attributes.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402

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
PAGE = 64
H, HKV, D = 64, 4, 128  # GQA 16:1
SEED = 3

# (batch, ctx, splits, paged)
CASES = []
for _b in (7, 8, 9):
    for _ctx in (4096, 16384):
        for _sp in (2, 4, 8):
            CASES.append((_b, _ctx, _sp, False))
            CASES.append((_b, _ctx, _sp, True))

# RAGGED decode: each sequence has its OWN kv length via cu_seqlens_kv, one query
# token each (Sq=1). This is the general decode case split-K must serve -- a batch
# where every sequence happens to be equal length does NOT exercise the per-
# sequence segment math. Lengths deliberately span short (<1 tile), single-page,
# odd, and long contexts so the segment [split_t0, split_t_end) differs per seq.
RAGGED_SEQS = {
    7: [4096, 8192, 16384, 1024, 6000, 300, 12000],
    8: [4096, 8192, 16384, 1024, 6000, 300, 12000, 2048],
    9: [4096, 8192, 16384, 1024, 6000, 300, 12000, 2048, 512],
}
RAGGED_CASES = []
for _b in (7, 8, 9):
    for _sp in (2, 4, 8):
        RAGGED_CASES.append((_b, _sp, False))
        RAGGED_CASES.append((_b, _sp, True))


def _build_mods():
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )
    from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import (
        dualwave_splitk_workspace_elems as ws_elems,
    )

    return build, ws_elems


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def reference(qf, qs, kf, ks, vf, vs, d):
    # decode: Sq=1, full (non-causal) attention over the KV context.
    q, k, v = qf.float() * qs, kf.float() * ks, vf.float() * vs
    b, sq, h, _ = q.shape
    rep = h // k.shape[2]
    kt = k.repeat_interleave(rep, 2).transpose(1, 2)   # [b,h,skv,d]
    vt = v.repeat_interleave(rep, 2).transpose(1, 2)
    att = q.transpose(1, 2) @ kt.transpose(-1, -2) / (d ** 0.5)  # [b,h,sq,skv]
    out = (att.softmax(-1) @ vt).transpose(1, 2)  # [b,sq,h,d]
    del att
    return out


def _run(build, ws_elems, b, ctx, splits, paged, d, seed, packed):
    sq = 1
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(b, sq, H, d, generator=g, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, ctx, HKV, d, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, ctx, HKV, d, generator=g, device=DEV, dtype=torch.bfloat16)
    qf, qs = q8(q)
    kf, ks = q8(k)
    vf, vs = q8(v)

    kwargs = {"seq_len_kv": ctx}
    if paged:
        npages = (ctx + PAGE - 1) // PAGE
        pg = torch.Generator().manual_seed(seed + 4)
        flat = torch.randperm(b * npages, generator=pg).tolist()
        perm = [flat[i * npages:(i + 1) * npages] for i in range(b)]
        pk = torch.zeros(b * npages, PAGE, HKV, d, device=DEV, dtype=kf.dtype)
        pv = torch.zeros_like(pk)
        for bi in range(b):
            for t in range(npages):
                pk[perm[bi][t]] = kf[bi, t * PAGE:(t + 1) * PAGE]
                pv[perm[bi][t]] = vf[bi, t * PAGE:(t + 1) * PAGE]
        bt = torch.tensor(perm, device=DEV, dtype=torch.int32)
        kv_k, kv_v = pk, pv
        kwargs["block_table"] = bt.contiguous().view(-1)
        kwargs["block_table_stride"] = npages
    else:
        kv_k, kv_v = kf, vf

    mod = build(num_heads=H, head_dim=d, causal=False, dtype_str="fp8",
                num_kv_heads=HKV, num_kv_splits=splits, paged=paged,
                cross_seqlen=True, gqa_pack_m=packed)
    o = torch.empty(b, sq, H, d, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(ws_elems(b, H, sq, splits, d), device=DEV, dtype=torch.float32)
    mod(qf.contiguous().view(-1), kv_k.contiguous().view(-1),
        kv_v.contiguous().view(-1), o.contiguous().view(-1), b, sq,
        workspace=ws, q_descale=qs, k_descale=ks, v_descale=vs, **kwargs)
    torch.cuda.synchronize()

    ref = reference(qf, qs, kf, ks, vf, vs, d)
    of = o.float()
    err = (of - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), ref.flatten(), dim=0).item()
    bad = int(((of - ref).abs().amax(dim=(2, 3)) > 1e-1).sum().item())
    del ref
    return err, cos, bad


def reference_ragged(qf, qs, kf, ks, vf, vs, cu_kv, d):
    """Per-sequence decode reference over a ragged KV pack.

    qf: [b, 1, H, D]; kf/vf: [total_kv, HKV, D] contiguous per sequence.
    Non-causal (decode attends the whole context). Returns [b, 1, H, D].
    """
    b = qf.shape[0]
    out = torch.empty(b, 1, H, d, device=DEV, dtype=torch.float32)
    for i in range(b):
        lo, hi = int(cu_kv[i]), int(cu_kv[i + 1])
        q = qf[i].float() * qs                       # [1, H, D]
        k = kf[lo:hi].float() * ks                   # [Lk, HKV, D]
        v = vf[lo:hi].float() * vs
        rep = H // k.shape[1]
        kt = k.repeat_interleave(rep, 1).transpose(0, 1)   # [H, Lk, D]
        vt = v.repeat_interleave(rep, 1).transpose(0, 1)
        att = q.transpose(0, 1) @ kt.transpose(-1, -2) / (d ** 0.5)  # [H, 1, Lk]
        out[i] = (att.softmax(-1) @ vt).transpose(0, 1)   # [1, H, D]
    return out


def _run_ragged(build, ws_elems, b, splits, paged, d, seed, packed):
    seqs_kv = RAGGED_SEQS[b][:b]
    total_kv = sum(seqs_kv)
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(b, 1, H, d, generator=g, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(total_kv, HKV, d, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(total_kv, HKV, d, generator=g, device=DEV, dtype=torch.bfloat16)
    qf, qs = q8(q)
    kf, ks = q8(k)
    vf, vs = q8(v)

    cu_q = torch.arange(b + 1, device=DEV, dtype=torch.int32)   # 1 q-token/seq
    cu_kv = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
    cu_kv[1:] = torch.tensor(seqs_kv, device=DEV).cumsum(0)

    kwargs = {"cu_seqlens_q": cu_q, "cu_seqlens_kv": cu_kv}
    if paged:
        # Per-sequence page pool with scrambled page order, stride = max pages.
        npages_per = [(ln + PAGE - 1) // PAGE for ln in seqs_kv]
        stride = max(npages_per)
        tot_pages = sum(npages_per)
        pg = torch.Generator().manual_seed(seed + 11)
        flat = torch.randperm(tot_pages, generator=pg).tolist()
        pk = torch.zeros(tot_pages, PAGE, HKV, d, device=DEV, dtype=kf.dtype)
        pv = torch.zeros_like(pk)
        bt = torch.zeros(b, stride, device=DEV, dtype=torch.int32)
        c = 0
        for bi, ln in enumerate(seqs_kv):
            lo = int(cu_kv[bi])
            for t in range(npages_per[bi]):
                pid = flat[c]
                c += 1
                s0 = lo + t * PAGE
                s1 = min(lo + (t + 1) * PAGE, lo + ln)
                n = s1 - s0
                pk[pid, :n] = kf[s0:s1]
                pv[pid, :n] = vf[s0:s1]
                bt[bi, t] = pid
        kv_k, kv_v = pk, pv
        kwargs["block_table"] = bt.contiguous().view(-1)
        kwargs["block_table_stride"] = stride
    else:
        kv_k, kv_v = kf, vf

    mod = build(num_heads=H, head_dim=d, causal=False, dtype_str="fp8",
                num_kv_heads=HKV, num_kv_splits=splits, paged=paged,
                varlen=True, gqa_pack_m=packed)
    o = torch.empty(b, 1, H, d, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(ws_elems(b, H, 1, splits, d), device=DEV, dtype=torch.float32)
    mod(qf.contiguous().view(-1), kv_k.contiguous().view(-1),
        kv_v.contiguous().view(-1), o.contiguous().view(-1), b, 1,
        workspace=ws, q_descale=qs, k_descale=ks, v_descale=vs, **kwargs)
    torch.cuda.synchronize()

    ref = reference_ragged(qf, qs, kf, ks, vf, vs, cu_kv, d)
    of = o.float()
    err = (of - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), ref.flatten(), dim=0).item()
    bad = int(((of - ref).abs().amax(dim=(2, 3)) > 1e-1).sum().item())
    del ref
    return err, cos, bad


def _check(err, cos, bad):
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"


@pytest.mark.parametrize(
    "b,ctx,splits,paged", CASES,
    ids=[f"b{b}_ctx{ctx}_sp{sp}_{'paged' if pg else 'dense'}"
         for b, ctx, sp, pg in CASES],
)
def test_packed_splitk_decode(b, ctx, splits, paged):
    """Equal-KV packed + split-K decode. gqa_pack_m and num_kv_splits are both
    forced -- the combination the build refusal used to forbid."""
    build, ws_elems = _build_mods()
    err, cos, bad = _run(build, ws_elems, b, ctx, splits, paged, D, SEED,
                         packed=True)
    _check(err, cos, bad)
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "b,splits,paged", RAGGED_CASES,
    ids=[f"b{b}_sp{sp}_{'paged' if pg else 'dense'}"
         for b, sp, pg in RAGGED_CASES],
)
def test_packed_splitk_decode_ragged(b, splits, paged):
    """Ragged decode: unequal seqused_k per sequence -- the general case that
    exercises per-sequence segment math the equal-KV loop does not."""
    build, ws_elems = _build_mods()
    err, cos, bad = _run_ragged(build, ws_elems, b, splits, paged, D, SEED,
                                packed=True)
    _check(err, cos, bad)
    torch.cuda.empty_cache()
