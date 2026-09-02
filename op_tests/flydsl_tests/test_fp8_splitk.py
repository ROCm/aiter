# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K path of the vendored fp8 attention kernel (gfx950).

Split-K partitions the KV axis across workgroups; each writes a partial plus
its (m, l) to a workspace, and a combine kernel forms sum(w*l*O)/sum(w*l) with
w = exp2(m_i - m_max).

The sharp signal is BAD ROWS, not aggregate cosine: the original defect
dropped every split but the largest-m one, so rows needing a single KV range
stayed exact while rows needing two or more were wrong (a clean 512/1024
split at S=1024 causal, yet aggregate cosine still read 0.94). Non-causal is
the strictest case since every row needs every split range.

The dense cases feed contiguous KV with no block table, which `ref_paged_attn`
cannot express, so this file keeps its own fp32 reference for a single
consistent oracle across dense and paged cases.

See _common for the ~/.flydsl/cache caveat.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aiter.ops.flydsl.utils import is_flydsl_available
from op_tests.flydsl_tests._common import assert_attn_close as _check
from op_tests.flydsl_tests._common import build_fp8_gfx950_with_ws_elems as _build_mods
from op_tests.flydsl_tests._common import dense_fp8_reference as reference
from op_tests.flydsl_tests._common import q8

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
HEAD_DIM = 128
SEED = 3

# (S, splits, causal)
CASES = [
    (1024, 1, True),  # baseline: no combine involved
    (1024, 2, True),
    (1024, 4, True),
    (1024, 8, True),
    (1024, 2, False),  # non-causal: every row needs every range
    (1024, 4, False),
    (512, 2, True),
    (512, 4, True),
    (4096, 2, True),
    (4096, 4, True),
]

# (S, splits, batch) -- paged + split-K is the production decode configuration.
PAGED_CASES = [
    (1024, 2, 1),
    (1024, 4, 1),
    (512, 4, 2),
    (768, 2, 4),
]


def _run(build, ws_elems, S, splits, causal, d, seed, b=1, paged=False):
    h, hkv = 64, 4
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda n: torch.randn(
        b, S, n, d, generator=g, device=DEV, dtype=torch.bfloat16
    )
    qf, qs = q8(mk(h))
    kf, ks = q8(mk(hkv))
    vf, vs = q8(mk(hkv))

    kwargs = {}
    if paged:
        npages = (S + PAGE - 1) // PAGE
        pg = torch.Generator().manual_seed(seed + 4)
        flat = torch.randperm(b * npages, generator=pg).tolist()
        perm = [flat[i * npages : (i + 1) * npages] for i in range(b)]
        pk = torch.zeros(b * npages, PAGE, hkv, d, device=DEV, dtype=kf.dtype)
        pv = torch.zeros_like(pk)
        for bi in range(b):
            for t in range(npages):
                pk[perm[bi][t]] = kf[bi, t * PAGE : (t + 1) * PAGE]
                pv[perm[bi][t]] = vf[bi, t * PAGE : (t + 1) * PAGE]
        bt = torch.tensor(perm, device=DEV, dtype=torch.int32)
        kv_k, kv_v = pk, pv
        kwargs = {"block_table": bt.contiguous().view(-1), "block_table_stride": npages}
    else:
        kv_k, kv_v = kf, vf

    mod = build(
        num_heads=h,
        head_dim=d,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=hkv,
        num_kv_splits=splits,
        paged=paged,
    )
    o = torch.empty(b, S, h, d, device=DEV, dtype=torch.bfloat16)
    ws = torch.zeros(ws_elems(b, h, S, splits, d), device=DEV, dtype=torch.float32)
    mod(
        qf.contiguous().view(-1),
        kv_k.contiguous().view(-1),
        kv_v.contiguous().view(-1),
        o.contiguous().view(-1),
        b,
        S,
        workspace=ws,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        **kwargs,
    )
    torch.cuda.synchronize()

    ref = reference(qf, qs, kf, ks, vf, vs, causal, d)
    of = o.float()
    err = (of - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), ref.flatten(), dim=0
    ).item()
    bad = int(((of - ref).abs().amax(dim=(2, 3)) > 1e-1).sum().item())
    del ref
    return err, cos, bad


@pytest.mark.parametrize(
    "S,splits,causal",
    CASES,
    ids=[f"S{S}_sp{sp}_{'c' if c else 'nc'}" for S, sp, c in CASES],
)
def test_splitk(S, splits, causal):
    build, ws_elems = _build_mods()
    err, cos, bad = _run(build, ws_elems, S, splits, causal, HEAD_DIM, SEED)
    _check(err, cos, bad)
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "S,splits,b",
    PAGED_CASES,
    ids=[f"S{S}_sp{sp}_b{b}" for S, sp, b in PAGED_CASES],
)
def test_paged_splitk(S, splits, b):
    """Paged + split-K: the production decode configuration."""
    build, ws_elems = _build_mods()
    err, cos, bad = _run(
        build, ws_elems, S, splits, True, HEAD_DIM, SEED, b=b, paged=True
    )
    _check(err, cos, bad)
    torch.cuda.empty_cache()
