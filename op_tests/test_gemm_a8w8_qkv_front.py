# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``gemm_a8w8_qkv_front`` + ``fused_qknorm_idxrqknorm``: the tuned-tier lookup (no GPU work) and the deferred fused tiers
against the plain tier through the same public calls (see test_flydsl_qkv_chain.py for the kernel-level checks).
"""

import inspect

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import gemm_a8w8_qkv_front as front

needs_gfx950 = pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 only")
NB = 64


def _row(m, tier):
    return {
        "libtype": "flydsl",
        "kernelId": front.KERNEL_ID,
        "kernelName": front.tier_kernel_name(tier),
        "us": 1.0,
    }


def test_front_tier_lookup(monkeypatch):
    rows = {
        ("gfx950", 256, 32, 2560, 6144, "1x32"): _row(32, "coop"),
        ("gfx950", 256, 64, 2560, 6144, "1x32"): _row(64, "coop"),
        ("gfx950", 256, 96, 2560, 6144, "1x32"): _row(96, "split3"),
        ("gfx950", 256, 128, 2560, 6144, "1x32"): _row(128, "plain"),
        ("gfx950", 256, 80, 2560, 6144, "1x32"): _row(
            80, "coop"
        ),  # a tier that cannot serve 80 rows
    }
    monkeypatch.setattr(front, "_load_tuned", lambda: rows)
    monkeypatch.setattr(front, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(front, "get_cu_num", lambda: 256)
    front.front_tier.cache_clear()
    try:
        assert front.front_tier(32) == "coop"
        assert front.front_tier(64) == "coop"
        assert front.front_tier(96) == "split3"
        assert front.front_tier(128) == "plain"
        assert front.front_tier(80) == "plain"  # the row's tier does not serve this m
        assert front.front_tier(16) == "plain"  # below the window
        assert front.front_tier(129) == "plain"  # above it
        assert front.front_tier(32, 2560, 4096) == "plain"  # another projection
    finally:
        front.front_tier.cache_clear()


def test_tier_applies():
    assert front.tier_applies("plain", 5)
    assert front.tier_applies("coop", 17) and front.tier_applies("coop", 64)
    assert not front.tier_applies("coop", 16) and not front.tier_applies("coop", 65)
    assert front.tier_applies("split3", 65) and front.tier_applies("split3", 128)
    assert not front.tier_applies("split3", 64) and not front.tier_applies(
        "split3", 129
    )


def _setup(seed=0):
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(front.N, front.K, generator=g, device="cuda") * 0.05).to(
        dtypes.fp8
    )
    w_scale = torch.randint(
        120,
        128,
        (front.N, front.K // 32),
        generator=g,
        dtype=torch.uint8,
        device="cuda",
    )
    weight = shuffle_weight(w, layout=(16, 16))
    aux = [
        (torch.randn(128, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
        for _ in range(4)
    ]
    cs = torch.randn(4096, 64, generator=g, device="cuda").to(torch.bfloat16)
    return weight, w_scale, front.chain_weight_scale(w_scale), aux, cs


def _insert(qkv, aux, cs, pos, slot, kv, ix, m, ks, vs):
    q = torch.empty(m, 2048, dtype=torch.bfloat16, device="cuda")
    iq = torch.empty(m, 128, dtype=dtypes.fp8, device="cuda")
    pages = kv.view(NB, -1)
    aiter.fused_qknorm_idxrqknorm(
        qkv, aux[0], aux[1], cs, pos, 16, 1, 64, 1e-6, aux[2], aux[3], 1, slot, pages, pages, ix, 128,
        q, iq, slot, kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4", k_scale=ks, v_scale=vs,
        asm_layout=False, skip_index_branch=False,
    )  # fmt: skip
    return q, iq


def _inputs(m):
    g = torch.Generator(device="cuda").manual_seed(m)
    x = (torch.randn(m, front.K, generator=g, device="cuda") * 2).to(torch.bfloat16)
    pos = torch.randint(0, 4000, (m,), generator=g, dtype=torch.int64, device="cuda")
    nreq = (m + 7) // 8
    bases = (
        torch.randperm(NB * 128 // 32, generator=g, device="cuda")[:nreq] * 32
    ).tolist()
    slot = torch.cat([torch.arange(b, b + 8) for b in bases])[:m].to(torch.int64).cuda()
    return x, pos, slot


@needs_gfx950
@pytest.mark.parametrize("m", [17, 33, 64, 65, 97, 128])
def test_fused_tier_matches_plain_tier(m):
    if m > 64:
        from aiter.ops.flydsl.kernels import mxfp8_gemm_decode_splitk as sk

        if not "cfg" in inspect.signature(sk.mxfp8_gemm_decode_splitk).parameters:
            pytest.skip(
                "split-3 row-major activation entry (A_ROW) not in this tree (#8610)"
            )
    weight, w_scale, chain_scale, aux, cs = _setup()
    x, pos, slot = _inputs(m)
    ks, vs = torch.tensor([0.5], device="cuda"), torch.tensor([0.7], device="cuda")
    tier = "coop" if m <= 64 else "split3"
    out = {}
    for t in ("plain", tier):
        kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8, device="cuda")
        ix = torch.zeros(NB, 128, 72, dtype=torch.uint8, device="cuda")
        qkv = front.gemm_a8w8_qkv_front(
            x, weight, w_scale, chain_scale, defer_ok=t != "plain", tier=t
        )
        assert (getattr(front._STATE, "deferred", None) is not None) == (t != "plain")
        q, iq = _insert(qkv, aux, cs, pos, slot, kv, ix, m, ks, vs)
        assert getattr(front._STATE, "deferred", None) is None  # consumed
        torch.cuda.synchronize()
        out[t] = (q, iq, kv, ix)
    (q0, iq0, kv0, ix0), (q1, iq1, kv1, ix1) = out["plain"], out[tier]
    assert (q0.float() - q1.float()).abs().max() <= 0.05 * q0.float().abs().max() + 1e-3
    assert (
        iq0.float() - iq1.float()
    ).abs().mean() < 0.05 * iq0.float().abs().mean() + 1e-3
    for ref, got in ((kv0, kv1), (ix0, ix1)):
        assert (ref != 0).any()
        assert ((got != ref).float().mean().item()) < 0.03


@needs_gfx950
def test_defer_not_ok_and_mismatch():
    weight, w_scale, chain_scale, aux, cs = _setup()
    x, pos, slot = _inputs(40)
    # defer_ok=False: always the plain tier, computed, nothing parked
    qkv = front.gemm_a8w8_qkv_front(x, weight, w_scale, chain_scale, defer_ok=False)
    assert (
        qkv.shape == (40, front.N) and getattr(front._STATE, "deferred", None) is None
    )
    # a parked front consumed by a different tensor is an error, not a silent wrong insert
    qkv = front.gemm_a8w8_qkv_front(
        x, weight, w_scale, chain_scale, defer_ok=True, tier="coop"
    )
    with pytest.raises(RuntimeError, match="deferred"):
        front.take_deferred(torch.empty_like(qkv))
    # a parked front the insert cannot consume (here: an mxfp4 cache) raises
    qkv = front.gemm_a8w8_qkv_front(
        x, weight, w_scale, chain_scale, defer_ok=True, tier="coop"
    )
    with pytest.raises(RuntimeError, match="deferred QKV front"):
        aiter.fused_qknorm_idxrqknorm(
            qkv, aux[0], aux[1], cs, pos, 16, 1, 64, 1e-6, aux[2], aux[3], 1, slot,
            torch.zeros(NB, 8704, dtype=torch.uint8, device="cuda"), torch.zeros(NB, 8704, dtype=torch.uint8, device="cuda"),
            None, 128, torch.empty(40, 2048, dtype=torch.bfloat16, device="cuda"),
            None, None, kv_cache_dtype="mxfp4", skip_index_branch=True,
        )  # fmt: skip


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
