# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Low-chunk mixed split-K path of the vendored fp8 kernel (gfx950).

This is the case the old ``max_seqlen_q == 1`` split-K gate forbade: a batch
mixing a short PREFILL CHUNK (q > 1) with several DECODES (q == 1), so query
lengths are UNEQUAL. The dense-q combine addressed O at ``batch_idx *
max_seqlen_q`` and would miscombine such a batch; the combine now rebases its O
write on cu_seqlens_q (init_descriptors), so the ragged pack is written to the
right rows.

Shape: one seq q=128 kv=16384 plus 8 seqs q=1 kv=varying, GQA 16:1, splits 2/4/8,
paged and dense, vs a per-sequence torch reference. Bad-row count is the signal.

The workspace is dense (sized for max_seqlen_q q-slots per batch/head/split); the
decode sequences fill only slot 0 and the combine's per-sequence num_records
bound drops the padding rows the grid launches past their length. So this also
exercises that OOB guard, not just the ragged O offset.

The dense cases feed contiguous KV with no block table, which `ref_paged_attn`
cannot express, so this file keeps its own fp32 reference.

Clear ~/.flydsl/cache before trusting a run after a kernel edit.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available

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

# One prefill chunk (q=128) then 8 decodes (q=1). Per-sequence (q_len, kv_len).
# kv lengths span short (<1 tile), single-page, odd, and long so each segment
# [split_t0, split_t_end) differs per sequence.
MIXED_SEQS = [
    (128, 16384),
    (1, 4096),
    (1, 8192),
    (1, 1024),
    (1, 6000),
    (1, 300),
    (1, 12000),
    (1, 2048),
    (1, 512),
]
SPLITS = (2, 4, 8)


def _build_mods():
    from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import (
        dualwave_splitk_workspace_elems as ws_elems,
    )
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )

    return build, ws_elems


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def reference_mixed(qf, qs, kf, ks, vf, vs, cu_q, cu_kv, causal, d):
    """Per-sequence attention over a ragged q/kv pack. Returns [total_q, H, D]."""
    b = cu_q.numel() - 1
    total_q = int(cu_q[-1])
    out = torch.empty(total_q, H, d, device=DEV, dtype=torch.float32)
    for i in range(b):
        ql, qh = int(cu_q[i]), int(cu_q[i + 1])
        kl, kh = int(cu_kv[i]), int(cu_kv[i + 1])
        q = qf[ql:qh].float() * qs  # [Lq, H, D]
        k = kf[kl:kh].float() * ks  # [Lk, HKV, D]
        v = vf[kl:kh].float() * vs
        rep = H // k.shape[1]
        kt = k.repeat_interleave(rep, 1).transpose(0, 1)  # [H, Lk, D]
        vt = v.repeat_interleave(rep, 1).transpose(0, 1)
        att = q.transpose(0, 1) @ kt.transpose(-1, -2) / (d**0.5)  # [H, Lq, Lk]
        if causal:
            lq, lk = att.shape[1], att.shape[2]
            # right-aligned causal: query row r (0-based) sees kv <= lk-lq+r.
            rows = torch.arange(lq, device=DEV).view(lq, 1)
            cols = torch.arange(lk, device=DEV).view(1, lk)
            mask = cols > (lk - lq + rows)
            att = att.masked_fill(mask.view(1, lq, lk), float("-inf"))
        out[ql:qh] = (att.softmax(-1) @ vt).transpose(0, 1)  # [Lq, H, D]
    return out


def _run_mixed(build, ws_elems, splits, paged, causal, d, seed):
    seqs = MIXED_SEQS
    b = len(seqs)
    q_lens = [s[0] for s in seqs]
    kv_lens = [s[1] for s in seqs]
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    max_q = max(q_lens)

    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(total_q, H, d, generator=g, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(total_kv, HKV, d, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(total_kv, HKV, d, generator=g, device=DEV, dtype=torch.bfloat16)
    qf, qs = q8(q)
    kf, ks = q8(k)
    vf, vs = q8(v)

    cu_q = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
    cu_q[1:] = torch.tensor(q_lens, device=DEV).cumsum(0)
    cu_kv = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
    cu_kv[1:] = torch.tensor(kv_lens, device=DEV).cumsum(0)

    kwargs = {"cu_seqlens_q": cu_q, "cu_seqlens_kv": cu_kv}
    if paged:
        npages_per = [(ln + PAGE - 1) // PAGE for ln in kv_lens]
        stride = max(npages_per)
        tot_pages = sum(npages_per)
        pg = torch.Generator().manual_seed(seed + 11)
        flat = torch.randperm(tot_pages, generator=pg).tolist()
        pk = torch.zeros(tot_pages, PAGE, HKV, d, device=DEV, dtype=kf.dtype)
        pv = torch.zeros_like(pk)
        bt = torch.zeros(b, stride, device=DEV, dtype=torch.int32)
        c = 0
        for bi, ln in enumerate(kv_lens):
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

    mod = build(
        num_heads=H,
        head_dim=d,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=HKV,
        num_kv_splits=splits,
        paged=paged,
        varlen=True,
        gqa_pack_m=True,
    )
    o = torch.empty(total_q, H, d, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(ws_elems(b, H, max_q, splits, d), device=DEV, dtype=torch.float32)
    mod(
        qf.contiguous().view(-1),
        kv_k.contiguous().view(-1),
        kv_v.contiguous().view(-1),
        o.contiguous().view(-1),
        b,
        max_q,
        workspace=ws,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        **kwargs,
    )
    torch.cuda.synchronize()

    ref = reference_mixed(qf, qs, kf, ks, vf, vs, cu_q, cu_kv, causal, d)
    of = o.float()
    err = (of - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), ref.flatten(), dim=0
    ).item()
    bad = int(((of - ref).abs().amax(dim=(1, 2)) > 1e-1).sum().item())
    del ref
    return err, cos, bad


_CASES = [
    (causal, splits, paged)
    for causal in (False, True)
    for splits in SPLITS
    for paged in (False, True)
]


@pytest.mark.parametrize(
    "causal,splits,paged",
    _CASES,
    ids=[
        f"{'causal' if c else 'noncausal'}_sp{sp}_{'paged' if pg else 'dense'}"
        for c, sp, pg in _CASES
    ],
)
def test_packed_splitk_mixed(causal, splits, paged):
    """Low-chunk mixed prefill+decode batch, gqa_pack_m forced with split-K."""
    build, ws_elems = _build_mods()
    err, cos, bad = _run_mixed(build, ws_elems, splits, paged, causal, D, SEED)
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"
    torch.cuda.empty_cache()
