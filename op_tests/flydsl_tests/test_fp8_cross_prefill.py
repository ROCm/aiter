# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chunked-prefill (cross-attention) correctness of the vendored fp8 kernel.

The production unified-attention call is chunked prefill: a chunk of new Q
tokens attends against a KV cache that already holds the earlier context, so
Skv > Sq and the causal mask is BOTTOM-RIGHT aligned. aiter's own reference
`ref_paged_attn` encodes this as

    torch.triu(ones(q_len, kv_len), diagonal=kv_len - q_len + 1)

When kv_len == q_len that degenerates to diagonal=1, the square case the
benchmark used to measure exclusively -- which is why this path was never
exercised.

This gate pins the alignment. A kernel that top-left aligns instead (i.e.
ignores the kv_len - q_len delta) still produces finite, plausible output and
still passes every square test, so nothing else in the suite would catch it.

The reference is `ref_paged_attn`, fed the identical scrambled page pool and
block table the kernel is fed.

Clear ~/.flydsl/cache before trusting a run after a kernel edit.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available  # noqa: E402
from op_tests.triton_tests.attention.test_unified_attention import (  # noqa: E402
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
PAGE = 64
H, HKV, D = 64, 4, 128

# (q_lens, kv_lens, label). Small enough for an fp32 reference to be cheap, but
# covering: pure cross-attention, the degenerate square, a ragged mixed batch,
# and the production chunk+decodes packing.
CASES = [
    ([256], [1024], "single chunk, 4x cache"),
    ([256], [256], "square (self-attn, historical case)"),
    ([320], [1024], "non-page-aligned q against aligned kv"),
    ([256, 512], [1024, 2048], "ragged, both cross"),
    ([256] + [1] * 4, [1024] + [1024] * 4, "chunk + 4 decodes"),
]


def _build_module():
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )

    return build


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def make_pages(kf, vf, seqs, seed):
    hkv, d = kf.shape[-2], kf.shape[-1]
    npages_per = [(ln + PAGE - 1) // PAGE for ln in seqs]
    stride = max(npages_per)
    tot = sum(npages_per)
    g = torch.Generator().manual_seed(seed)
    flat = torch.randperm(tot, generator=g).tolist()
    pk = torch.zeros(tot, PAGE, hkv, d, device=DEV, dtype=kf.dtype)
    pv = torch.zeros_like(pk)
    bt = torch.zeros(len(seqs), stride, device=DEV, dtype=torch.int32)
    off, c = 0, 0
    for bi, ln in enumerate(seqs):
        for t in range(npages_per[bi]):
            pid = flat[c]
            c += 1
            s0, s1 = off + t * PAGE, min(off + (t + 1) * PAGE, off + ln)
            pk[pid, :s1 - s0] = kf[s0:s1]
            pv[pid, :s1 - s0] = vf[s0:s1]
            bt[bi, t] = pid
        off += ln
    return pk, pv, bt, stride


@pytest.mark.parametrize(
    "q_lens,kv_lens,label", CASES, ids=[c[2] for c in CASES],
)
def test_cross_prefill(q_lens, kv_lens, label):
    build = _build_module()
    total, kv_total, b = sum(q_lens), sum(kv_lens), len(q_lens)
    g = torch.Generator(device=DEV).manual_seed(7)
    mk = lambda rows, n: torch.randn(rows, n, D, generator=g,  # noqa: E731
                                     device=DEV, dtype=torch.bfloat16)
    qf, qs = q8(mk(total, H))
    kf, ks = q8(mk(kv_total, HKV))
    vf, vs = q8(mk(kv_total, HKV))
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)),
                        device=DEV, dtype=torch.int32)
    cu_kv = torch.tensor([0] + list(torch.tensor(kv_lens).cumsum(0)),
                         device=DEV, dtype=torch.int32)
    pk, pv, bt, stride = make_pages(kf, vf, kv_lens, 8)
    mod = build(num_heads=H, head_dim=D, causal=True, dtype_str="fp8",
                num_kv_heads=HKV, varlen=True, paged=True)
    o = torch.empty(total, H, D, device=DEV, dtype=torch.bfloat16)
    mod(qf.contiguous().view(-1), pk.contiguous().view(-1),
        pv.contiguous().view(-1), o.contiguous().view(-1), b, max(q_lens),
        cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv,
        q_descale=qs, k_descale=ks, v_descale=vs,
        block_table=bt.contiguous().view(-1), block_table_stride=stride)
    torch.cuda.synchronize()

    want = ref_paged_attn(
        query=qf, key_cache=pk, value_cache=pv, query_lens=list(q_lens),
        kv_lens=list(kv_lens), block_tables=bt, scale=D ** -0.5,
        out_dtype=torch.float32, q_descale=qs, k_descale=ks, v_descale=vs,
    )
    got = o.float()
    err = (got - want).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), want.flatten(), dim=0).item()
    # `bad` -- rows whose worst element exceeds the tolerance -- is the sharper
    # signal: a whole wrong sequence past the first BLOCK_M still read cos=0.96
    # before the active guard existed. A bottom-right/top-left mask confusion is
    # exactly that kind of failure.
    bad = int((got - want).abs().amax(dim=(1, 2)).gt(1e-1).sum().item())
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    assert err < 1e-1, f"max err {err:.4g}"
    assert cos > 0.99, f"cosine {cos:.6f}"
    torch.cuda.empty_cache()
