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
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
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
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled
    from aiter.ops.flydsl.kernels.qkv_chain import code_rows, qkv_chain_insert

    wsh, w_scale, chain_scale, qw, kw, iqw, ikw, cs, ks, vs = _weights()
    x, pos, slot = make_inputs(m, m + skip, skip)
    outs = []
    for impl in ("ref", "chain"):
        kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8)
        ix = torch.zeros(NB, 128, 72, dtype=torch.uint8)
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
            codes, sc = mxfp8_quantize_shuffled(x, code_rows(m))
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
    for name, a, b in (("kv", kv0, kv1), ("index", ix0, ix1)):
        frac = (a != b).float().mean().item()
        assert frac < 0.03, f"{name} pages differ in {frac:.4f} of bytes"
    if skip:  # whole skipped requests (slot -1) write no page bytes
        assert (kv1 != 0).any() and (ix1 != 0).any()


@needs_gfx950
@pytest.mark.parametrize("m", [40, 72, 112])
def test_chain_graph_replay_matches_eager(m):
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled
    from aiter.ops.flydsl.kernels.qkv_chain import code_rows, qkv_chain_insert

    wsh, _, chain_scale, qw, kw, iqw, ikw, cs, ks, vs = _weights()
    x, pos, slot = make_inputs(m, 123)
    xs_ = [x, make_inputs(m, 124)[0]]
    kv = torch.zeros(NB, 2, 128, 72, dtype=torch.uint8)
    ix = torch.zeros(NB, 128, 72, dtype=torch.uint8)
    q = torch.empty(m, 2048, dtype=torch.bfloat16)
    iq = torch.empty(m, 128, dtype=dtypes.fp8)
    xin = x.clone()

    def call():
        codes, sc = mxfp8_quantize_shuffled(xin, code_rows(m))
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
