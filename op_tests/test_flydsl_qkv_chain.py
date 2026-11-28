# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QKV front chain (MiniMax-M3.1 TP4, 17-128 decode rows: one launch to 64 rows, split-3 GEMM + reduce-insert above) against the
unfused public path.

Reference: the same bf16 activation through aiter's MXFP8 quantizer + ``gemm_a8w8_blockscale`` + ``fused_qknorm_idxrqknorm``
(NVFP4 K / V / index-K pages). The chain's GEMM sums the K in three fp32 thirds, the reference GEMM in another order, so
q / index-q / pages agree to bf16 rounding (q, index-q) and to the odd rounding-boundary flip in the NVFP4 codes (pages);
the checks bound both. Slots are decode-shaped (requests of 8 consecutive tokens at scattered offsets, some skipped with
-1). Also: the same call replayed in a CUDA graph with changed inputs gives the eager result (epoch counters), and a
second weight set / stream does not disturb the first.

This file also holds the ``gemm_a8w8_qkv_front`` + ``fused_qknorm_idxrqknorm`` tests (formerly
``test_gemm_a8w8_qkv_front.py``): the static tier lookup (no GPU work) and the deferred fused tiers against the plain
tier through the same public calls (the kernel-level checks are the chain tests above).
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import gemm_a8w8_qkv_front as front
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight

torch.set_default_device("cuda")
N, K, NB = 2560, 6144, 64


def make_inputs(m, seed, skip_every=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, K, generator=g) * 2).to(torch.bfloat16)
    pos = torch.randint(0, 4000, (m,), generator=g, dtype=torch.int64)
    nreq = (m + 7) // 8
    bases = (
        torch.randperm(NB * 128 // 32, generator=g)[:nreq] * 32
        + torch.randint(0, 24, (nreq,), generator=g)
    ).tolist()
    slot = torch.cat([torch.arange(b, b + 8) for b in bases])[:m].to(torch.int64).cuda()
    if skip_every:
        for r in range(0, nreq, skip_every):
            slot[r * 8 : (r + 1) * 8] = -1
    return x, pos, slot


def _weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, generator=g) * 0.05).to(dtypes.fp8)
    w_scale = torch.randint(120, 128, (N, K // 32), generator=g, dtype=torch.uint8)
    wsh = shuffle_weight(w, layout=(16, 16))
    chain_scale = shuffle_scale_a16w4(w_scale.view(dtypes.fp8_e8m0), 1, False)
    qw, kw, iqw, ikw = [
        (torch.randn(128, generator=g) * 0.1).to(torch.bfloat16) for _ in range(4)
    ]
    cs = torch.randn(4096, 64, generator=g).to(torch.bfloat16)
    return (
        wsh,
        w_scale,
        chain_scale,
        qw,
        kw,
        iqw,
        ikw,
        cs,
        torch.tensor([0.5]),
        torch.tensor([0.7]),
    )


needs_gfx950 = pytest.mark.skipif(get_gfx_runtime() != "gfx950", reason="gfx950 only")


@needs_gfx950
@pytest.mark.parametrize("skip", [0, 3])
@pytest.mark.parametrize(
    "m", [17, 24, 32, 40, 48, 56, 64, 65, 72, 80, 81, 96, 97, 112, 128]
)
def test_chain_matches_public_path(m, skip):
    from aiter.ops.flydsl.kernels.qkv_chain import qkv_chain_insert

    wsh, w_scale, chain_scale, qw, kw, iqw, ikw, cs, ks, vs = _weights()
    x, pos, slot = make_inputs(m, m + skip, skip)
    outs = []
    g = torch.Generator(device="cuda").manual_seed(7)
    # both implementations start from the same nonzero pages: bytes outside the inserted tokens must stay as they were
    kv_init = torch.randint(1, 256, (NB, 2, 128, 72), generator=g, dtype=torch.uint8)
    ix_init = torch.randint(1, 256, (NB, 128, 72), generator=g, dtype=torch.uint8)
    for impl in ("ref", "chain"):
        kv = kv_init.clone()
        ix = ix_init.clone()
        q = torch.empty(m, 2048, dtype=torch.bfloat16)
        iq = torch.empty(m, 128, dtype=dtypes.fp8)
        if impl == "ref":
            xq, xs = per_group_quant_hip(
                x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
            )
            qkv = gemm_a8w8_blockscale(
                xq,
                wsh,
                xs,
                w_scale.view(dtypes.fp8_e8m0),
                dtypes.bf16,
                isBpreshuffled=True,
            )
            kvf = kv.view(NB, -1)
            aiter.fused_qknorm_idxrqknorm(
                qkv,
                qw,
                kw,
                cs,
                pos,
                16,
                1,
                64,
                1e-6,
                iqw,
                ikw,
                1,
                slot,
                kvf,
                kvf,
                ix,
                128,
                q,
                iq,
                slot,
                kv_cache_dtype="nvfp4",
                index_cache_dtype="nvfp4",
                k_scale=ks,
                v_scale=vs,
                asm_layout=False,
                skip_index_branch=False,
            )
        else:
            codes, sc = per_group_quant_hip(
                x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
            )
            qkv_chain_insert(
                codes,
                sc,
                wsh,
                chain_scale,
                m,
                qw,
                kw,
                iqw,
                ikw,
                cs,
                pos,
                slot,
                slot,
                kv,
                ix,
                ks,
                vs,
                q,
                iq,
                eps=1e-6,
            )
        torch.cuda.synchronize()
        outs.append((q, iq, kv, ix))
    (q0, iq0, kv0, ix0), (q1, iq1, kv1, ix1) = outs
    dq = (q0.float() - q1.float()).abs().max()
    assert dq <= 0.05 * q0.float().abs().max() + 1e-3, f"q differs by {dq.item()}"
    miq = (iq0.float() - iq1.float()).abs().mean().item()
    assert (
        miq < 0.05 * iq0.float().abs().mean().item() + 1e-3
    ), f"index q mean diff {miq}"
    for name, ref, out, init in (
        ("kv", kv0, kv1, kv_init),
        ("index", ix0, ix1, ix_init),
    ):
        touched = (
            ref != init
        )  # bytes the reference insert wrote (skipped slots, -1, write none)
        assert touched.any(), f"{name}: the reference wrote nothing"
        stray = ((out != init) & ~touched).sum().item()
        assert stray == 0, f"{name}: {stray} bytes written outside the inserted tokens"
        frac = (out != ref)[touched].float().mean().item()
        assert frac < 0.03, f"{name} inserted bytes differ in {frac:.4f}"
        assert (out != init)[
            touched
        ].float().mean().item() > 0.97, f"{name}: inserts missing"


@needs_gfx950
@pytest.mark.parametrize("m", [40, 72, 112])
def test_chain_graph_replay_matches_eager(m):
    from aiter.ops.flydsl.kernels.qkv_chain import qkv_chain_insert

    wsh, _, chain_scale, qw, kw, iqw, ikw, cs, ks, vs = _weights()
    x, pos, slot = make_inputs(m, 123)
    xs_ = [x, make_inputs(m, 124)[0]]
    kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8)
    ix = torch.zeros(NB, 128, 72, dtype=torch.uint8)
    q = torch.empty(m, 2048, dtype=torch.bfloat16)
    iq = torch.empty(m, 128, dtype=dtypes.fp8)
    xin = x.clone()

    def call():
        codes, sc = per_group_quant_hip(
            xin, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
        )
        qkv_chain_insert(
            codes,
            sc,
            wsh,
            chain_scale,
            m,
            qw,
            kw,
            iqw,
            ikw,
            cs,
            pos,
            slot,
            slot,
            kv,
            ix,
            ks,
            vs,
            q,
            iq,
            eps=1e-6,
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for xi in xs_ * 3:  # changed inputs each replay: the epoch counters need no re-arm
        xin.copy_(xi)
        kv.zero_()
        ix.zero_()
        call()
        torch.cuda.synchronize()
        eager = (q.clone(), kv.clone(), ix.clone())
        kv.zero_()
        ix.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert (
            torch.equal(q, eager[0])
            and torch.equal(kv, eager[1])
            and torch.equal(ix, eager[2])
        )


# ---------------------------------------------------------------------------
# gemm_a8w8_qkv_front + fused_qknorm_idxrqknorm (absorbed from the former
# op_tests/test_gemm_a8w8_qkv_front.py): the static tier lookup (no GPU work)
# and the deferred fused tiers against the plain tier through the same public
# calls (the kernel-level checks are the chain tests above).
# ---------------------------------------------------------------------------


def test_front_tier_is_static(monkeypatch):
    monkeypatch.setattr(front, "get_gfx", lambda: "gfx950")
    tiers = {m: front.front_tier(m) for m in (0, 1, 16, 17, 40, 64, 65, 100, 128, 129)}
    assert tiers == {
        0: "plain",
        1: "coop",
        16: "coop",
        17: "coop",
        40: "coop",
        64: "coop",
        65: "split",
        100: "split",
        128: "split",
        129: "plain",
    }
    monkeypatch.setattr(front, "get_gfx", lambda: "gfx942")
    assert front.front_tier(40) == "plain"  # the fused kernels are gfx950 only


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
@pytest.mark.parametrize("m", [1, 4, 8, 12, 16, 17, 33, 64, 65, 97, 128])
@pytest.mark.parametrize("ks", [3, 1, 2, 4, 6])
def test_fused_tier_matches_plain_tier(m, ks):
    if ks != 3 and m <= 64:
        pytest.skip("the one-launch tier has a fixed K split")
    weight, w_scale, chain_scale, aux, cs = _setup()
    x, pos, slot = _inputs(m)
    kscale, vscale = torch.tensor([0.5], device="cuda"), torch.tensor(
        [0.7], device="cuda"
    )
    tier = "coop" if m <= 64 else "split"
    out = {}
    for t in ("plain", tier):
        kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8, device="cuda")
        ix = torch.zeros(NB, 128, 72, dtype=torch.uint8, device="cuda")
        qkv = front.gemm_a8w8_qkv_front(
            x, weight, w_scale, chain_scale, defer_ok=t != "plain", tier=t, ks=ks
        )
        assert (getattr(front._STATE, "deferred", None) is not None) == (t != "plain")
        q, iq = _insert(qkv, aux, cs, pos, slot, kv, ix, m, kscale, vscale)
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
    # a parked front never consumed: the next projection on this thread raises (and drops it)
    front.gemm_a8w8_qkv_front(
        x, weight, w_scale, chain_scale, defer_ok=True, tier="coop"
    )
    with pytest.raises(RuntimeError, match="never consumed"):
        front.gemm_a8w8_qkv_front(x, weight, w_scale, chain_scale, defer_ok=False)
    assert getattr(front._STATE, "deferred", None) is None
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


@needs_gfx950
@pytest.mark.parametrize("m", list(range(17, 129)))
def test_quantizer_buffers_not_overrun(m):
    """The quantizer call the front makes (``per_group_quant_hip`` -> ``dynamic_per_group_scaled_quant``: fp8 codes, row-major
    E8M0 scales, group 32, K 6144, no transpose) writes inside its [m, K] / [m, K / 32] buffers for every row count, m not a
    multiple of 32 included: guard bytes on both sides of both outputs stay intact."""
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    x = (torch.randn(m, front.K, device="cuda") * 2).to(torch.bfloat16)
    ns, nq = m * (front.K // 32), m * front.K
    g = 4096  # guard bytes either side

    def guarded(n):
        buf = torch.full((g + n + g,), 0xAB, dtype=torch.uint8, device="cuda")
        return buf, buf[g : g + n]

    sbuf, sv = guarded(ns)
    ybuf, yv = guarded(nq)
    dynamic_per_group_scaled_quant(
        yv.view(dtypes.fp8).view(m, front.K), x, sv.view(dtypes.fp8_e8m0).view(m, front.K // 32), 32,
        shuffle_scale=False, num_rows=None, num_rows_factor=1,
    )  # fmt: skip
    torch.cuda.synchronize()
    for buf, n in ((sbuf, ns), (ybuf, nq)):
        assert (buf[:g] == 0xAB).all() and (buf[g + n :] == 0xAB).all()
    assert (sv != 0xAB).any() and (yv != 0xAB).any()


@needs_gfx950
@pytest.mark.parametrize("m", [1, 4, 8, 12, 16, 17, 33, 64, 65, 97, 128])
@pytest.mark.parametrize("ks", [3, 1, 2, 4, 6])
def test_fused_front_is_bitwise_the_unfused_split_chain(m, ks):
    """The fused kernels (GEMM + insert) against the same GEMM's fp32 K-part slabs summed in order, rounded to bf16 and
    inserted by the unfused ``fused_qknorm_idxrqknorm``: q / index q bytes are identical, for every split factor.
    """
    from aiter.ops.flydsl.kernels import mxfp8_gemm_decode_splitk as sk
    from aiter.ops.quant import per_group_quant_hip

    if ks != 3 and m <= 64:
        pytest.skip("the one-launch tier has a fixed K split")
    weight, w_scale, chain_scale, aux, cs = _setup()
    x, pos, slot = _inputs(m)
    kscale, vscale = torch.tensor([0.5], device="cuda"), torch.tensor(
        [0.7], device="cuda"
    )
    outs = []
    for fused in (False, True):
        kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8, device="cuda")
        ix = torch.zeros(NB, 128, 72, dtype=torch.uint8, device="cuda")
        if fused:
            qkv = front.gemm_a8w8_qkv_front(
                x, weight, w_scale, chain_scale, defer_ok=True,
                tier="coop" if m <= 64 else "split", ks=ks,
            )  # fmt: skip
        else:
            xq, xs = per_group_quant_hip(
                x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
            )
            slab = sk.mxfp8_gemm_decode_splitk(
                xq, xs, weight, chain_scale, cfg=(3, 4, True, True), ks=ks
            )
            acc = torch.zeros_like(slab[0])
            for z in range(ks):
                acc = acc + slab[z]
            qkv = acc.to(torch.bfloat16)
        q, iq = _insert(qkv, aux, cs, pos, slot, kv, ix, m, kscale, vscale)
        torch.cuda.synchronize()
        outs.append((q, iq.view(torch.uint8)))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


if __name__ == "__main__":
    # CI runs op tests as `python3 <file>` (.github/scripts/aiter_test.sh): run the cases under pytest
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
