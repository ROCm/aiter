# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run on gfx1250:

python op_tests/test_mla_nvfp4_decode_prep.py --batch 1536 2048 2200 2240 2300
python -m pytest op_tests/test_mla_nvfp4_decode_prep.py -q.

mla_nvfp4_decode_prep against the module's own producers, byte for byte, over
several decode steps from the same stale bytes: Q's RoPE bytes against
mla_nvfp4_quantize_q, the tail tiles and both cache planes against
mla_nvfp4_update_cache on the tiles, and the fused RoPE against aiter's
Triton RoPE (RotaryEmbedding.forward_triton, AITER_ROPE_TRITON_BACKEND=1).
"""

import pytest
import torch

from aiter.ops.mla_nvfp4 import (
    mla_nvfp4_decode_prep,
    mla_nvfp4_quantize_q,
    mla_nvfp4_update_cache,
)
from aiter.test_common import benchmark

H = 128
# DeepSeek-R1's YaRN RoPE.
YARN = {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250":
        pytest.skip("requires gfx1250")


def make_rope():
    from aiter.rotary_embedding import get_rope

    r = get_rope(
        64,
        rotary_dim=64,
        max_position=163840,
        base=10000,
        is_neox_style=False,
        rope_scaling=YARN,
        dtype=torch.bfloat16,
    ).to("cuda")
    assert r.cos_cache.dtype == torch.bfloat16 and r.cos_cache.shape[-1] == 32
    return r


@pytest.fixture(scope="module")
def rope():
    return make_rope()


def values(gen, *shape, s=1.0):
    """BF16 with outliers, zeros, signed zeros and blocks of tiny values."""
    x = torch.randn(shape, generator=gen, device="cuda") * s
    x = torch.where(torch.rand(shape, generator=gen, device="cuda") < 0.02, x * 30, x)
    x[torch.rand(shape, generator=gen, device="cuda") < 0.01] = 0.0
    x[torch.rand(shape, generator=gen, device="cuda") < 0.01] = -0.0
    tiny = torch.tensor([1e-35, -3e-38, 1e-40, 0.0, 2e-31], device="cuda")
    pick = tiny[torch.randint(0, len(tiny), shape, generator=gen, device="cuda")]
    rows = torch.rand((*shape[:-1], 1), generator=gen, device="cuda") < 0.03
    return torch.where(rows, pick, x).to(torch.bfloat16)


class Case:
    """B requests, one token each per step, their tail tiles in a ring of B + 3
    rows (permuted request ids), pages from a shuffled table; the rows from
    `live` on are padding. Each request starts part way into a tile, as a
    prefill leaves it (rows 0..rn-1 already in the tile and the cache)."""

    def __init__(self, b, live, seed):
        self.gen = gen = torch.Generator(device="cuda").manual_seed(seed)
        self.b, self.live = b, live
        self.pos = torch.randint(0, 400, (b,), generator=gen, device="cuda")
        self.pos[:6] = torch.tensor([0, 15, 16, 47, 63, 64], device="cuda")[: min(6, b)]
        self.table = torch.randperm(b * 8, generator=gen, device="cuda").view(b, 8)
        self.req = torch.randperm(b + 3, generator=gen, device="cuda")[:b].to(
            torch.int32
        )
        self.ring = values(gen, b + 3, 16, 576)
        planes = [
            torch.randint(
                0, 256, (b * 8, n), generator=gen, dtype=torch.uint8, device="cuda"
            )
            for n in (16384, 8192)
        ]
        self.cn, self.ca = planes
        rn = (self.pos % 16).to(torch.int32)
        part = rn > 0
        if part.any():
            mla_nvfp4_update_cache(
                self.ring[self.req.long()][part].contiguous(),
                self.page()[part],
                self.tile()[part],
                rn[part],
                self.cn,
                self.ca,
            )

    def page(self):
        return (
            self.table.gather(1, (self.pos // 64)[:, None]).squeeze(1).to(torch.int32)
        )

    def tile(self):
        return ((self.pos % 64) // 16).to(torch.int32)

    def slots(self):
        s = self.page().long() * 64 + self.pos % 64
        s[self.live :] = -1
        return s


def reference_step(case, ring, cn, ca, q, k, rope_mod, fused):
    """The step through the separate producers: the Triton RoPE on copies,
    mla_nvfp4_quantize_q, the tail row appended, mla_nvfp4_update_cache."""
    b, live = case.b, case.live
    q_pe, k_pe = q[..., 512:].clone(), k[:, 512:].clone().view(b, 1, 64)
    if fused:
        rope_mod.forward_triton(case.pos, q_pe, k_pe)
    q_full = q.clone()
    q_full[..., 512:] = q_pe
    _, qa = mla_nvfp4_quantize_q(q_full)
    if live:
        req = case.req[:live].long()
        rows = case.pos[:live] % 16
        ring[req, rows, :512] = k[:live, :512]
        ring[req, rows, 512:] = k_pe[:live, 0]
        mla_nvfp4_update_cache(
            ring[req].contiguous(),
            case.page()[:live],
            case.tile()[:live],
            (rows + 1).to(torch.int32),
            cn,
            ca,
        )
    return qa


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize(
    "b,live,steps,seed",
    [
        (1, 1, 3, 0),
        (37, 30, 6, 1),
        (1536, 1519, 3, 2),
        (2048, 2031, 3, 3),
        (2300, 2283, 3, 4),
    ],
)
def test_decode_prep_matches_the_producers(rope, fused, b, live, steps, seed):
    case = Case(b, live, seed)
    gen = case.gen
    ring, cn, ca = case.ring.clone(), case.cn.clone(), case.ca.clone()
    ref_ring, ref_cn, ref_ca = case.ring.clone(), case.cn.clone(), case.ca.clone()
    cos, sin = (rope.cos_cache, rope.sin_cache) if fused else (None, None)
    for step in range(steps):
        # Q and K as the projections leave them: q_rope and k_rope are views.
        q = values(gen, b, H, 576, s=2.0)
        k = values(gen, b, 576, s=0.5)
        stale = torch.randint(
            0, 256, (b, H, 128), generator=gen, dtype=torch.uint8, device="cuda"
        )
        qa = stale.clone()
        mla_nvfp4_decode_prep(
            q[..., 512:],
            qa,
            ring,
            case.req,
            k[:, :512],
            k[:, 512:],
            case.pos,
            case.slots(),
            cn,
            ca,
            cos,
            sin,
        )
        want = reference_step(case, ref_ring, ref_cn, ref_ca, q, k, rope, fused)
        torch.cuda.synchronize()
        rope_bytes = torch.cat((torch.arange(64), torch.arange(96, 128))).cuda()
        got_l, want_l = qa[:live][..., rope_bytes], want[:live][..., rope_bytes]
        assert torch.equal(got_l, want_l), f"step {step}: q_aux RoPE bytes differ"
        assert torch.equal(
            qa[..., 64:96], stale[..., 64:96]
        ), "q_aux NoPE scales touched"
        pad = qa[live:]
        assert (pad[..., :64] == 0).all() and (pad[..., 96:104] == 0).all()
        assert (pad[..., 104:] == 0x7F).all(), "padding Q scales"
        assert torch.equal(ring, ref_ring), f"step {step}: tail tiles differ"
        assert torch.equal(cn, ref_cn), f"step {step}: kv_nope differs"
        assert torch.equal(ca, ref_ca), f"step {step}: kv_aux differs"
        case.pos = case.pos + 1


def test_graph_replay(rope):
    b = 64
    case = Case(b, b, 7)
    q = values(case.gen, b, H, 576, s=2.0)
    k = values(case.gen, b, 576, s=0.5)
    slots = case.slots()
    out = [
        [
            torch.zeros((b, H, 128), dtype=torch.uint8, device="cuda"),
            case.ring.clone(),
            case.cn.clone(),
            case.ca.clone(),
        ]
        for _ in range(2)
    ]

    def run(qa, ring, cn, ca):
        mla_nvfp4_decode_prep(
            q[..., 512:],
            qa,
            ring,
            case.req,
            k[:, :512],
            k[:, 512:],
            case.pos,
            slots,
            cn,
            ca,
            rope.cos_cache,
            rope.sin_cache,
        )

    run(*out[0])
    snapshot = [t.clone() for t in out[1]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(*out[1])
    for t, s in zip(out[1], snapshot):
        t.copy_(s)
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(out[1], out[0]):
        assert torch.equal(got, want)


def test_contracts(rope):
    b = 4
    case = Case(b, b, 9)
    q = values(case.gen, b, H, 576)
    k = values(case.gen, b, 576)
    qa = torch.zeros((b, H, 128), dtype=torch.uint8, device="cuda")

    def call(q_rope=None, cos=rope.cos_cache, sin=rope.sin_cache, batch=b, ring=None):
        mla_nvfp4_decode_prep(
            q[:batch, :, 512:] if q_rope is None else q_rope,
            qa[:batch],
            case.ring if ring is None else ring,
            case.req[:batch],
            k[:batch, :512],
            k[:batch, 512:],
            case.pos[:batch],
            case.slots()[:batch],
            case.cn,
            case.ca,
            cos,
            sin,
        )

    call()
    call(batch=0)
    with pytest.raises(ValueError):
        call(sin=None)
    # Heads not 128 head strides apart from row to row.
    with pytest.raises(RuntimeError):
        call(q_rope=q.transpose(0, 1).contiguous().transpose(0, 1)[..., 512:])
    with pytest.raises(RuntimeError):
        call(q_rope=q[..., 512:].float())
    # Every request on one tile, or every row of a tile on one row.
    for ring in (
        case.ring[:1].expand(b + 3, 16, 576),
        case.ring[:, :1].expand(-1, 16, -1),
    ):
        with pytest.raises(RuntimeError, match="overlap"):
            call(ring=ring)


@pytest.mark.parametrize("b", [37, 1100])
def test_strided_rings(rope, b):
    """A ring with padded rows at the narrowest disjoint request stride
    (15 * 608 + 576) and one with the request and row axes swapped in memory
    take the same bytes as a contiguous ring."""
    case = Case(b, b - 5, 13)
    q = values(case.gen, b, H, 576, s=2.0)
    k = values(case.gen, b, 576, s=0.5)

    def padded(ring):
        rows = ring.shape[0]
        buf = torch.zeros(rows * 9696, dtype=ring.dtype, device=ring.device)
        view = buf.as_strided(ring.shape, (9696, 608, 1))
        view.copy_(ring)
        return view

    def swapped(ring):
        return ring.transpose(0, 1).contiguous().transpose(0, 1)

    def run(ring):
        qa = torch.zeros((b, H, 128), dtype=torch.uint8, device="cuda")
        cn, ca = case.cn.clone(), case.ca.clone()
        mla_nvfp4_decode_prep(
            q[..., 512:],
            qa,
            ring,
            case.req,
            k[:, :512],
            k[:, 512:],
            case.pos,
            case.slots(),
            cn,
            ca,
            rope.cos_cache,
            rope.sin_cache,
        )
        torch.cuda.synchronize()
        return qa, ring, cn, ca

    want = run(case.ring.clone())
    for layout in (padded, swapped):
        for got, ref in zip(run(layout(case.ring)), want):
            assert torch.equal(got, ref), layout.__name__


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_launches_on_the_input_device(rope):
    """Inputs on cuda:1 while cuda:0 is current: the same bytes as with cuda:1
    current, so the launch went to cuda:1's stream."""
    b = 37
    with torch.cuda.device(1):
        case = Case(b, 30, 11)
        q = values(case.gen, b, H, 576, s=2.0)
        k = values(case.gen, b, 576, s=0.5)
        cos, sin = rope.cos_cache.to("cuda:1"), rope.sin_cache.to("cuda:1")
        start = (case.ring, case.cn, case.ca)

    def run():
        ring, cn, ca = (t.clone() for t in start)
        qa = torch.zeros((b, H, 128), dtype=torch.uint8, device="cuda:1")
        mla_nvfp4_decode_prep(
            q[..., 512:],
            qa,
            ring,
            case.req,
            k[:, :512],
            k[:, 512:],
            case.pos,
            case.slots(),
            cn,
            ca,
            cos,
            sin,
        )
        torch.cuda.synchronize(1)
        return qa, ring, cn, ca

    with torch.cuda.device(1):
        want = run()
    with torch.cuda.device(0):
        got = run()
    for g, w in zip(got, want):
        assert torch.equal(g, w)


@pytest.mark.parametrize("qstride", [192, 200])
@pytest.mark.parametrize(
    "b", [1023, 1024, 1025, 1536, 1537, 2047, 2048, 2049, 2200, 2240, 2300]
)
@pytest.mark.parametrize("padding", [0, 17, None])
def test_dsr1_strides_eager_and_graph(rope, b, padding, qstride):
    live = b if padding is None else b - padding
    case = Case(b, live, 913 + live)
    backing = torch.full(
        (b + 3, 61, 16, 576), 0.75, dtype=torch.bfloat16, device="cuda"
    )
    ring = backing[:, 17]
    ring.copy_(case.ring)
    ref_ring = case.ring.clone()
    cn, ca = case.cn.clone(), case.ca.clone()
    ref_cn, ref_ca = case.cn.clone(), case.ca.clone()
    q_storage = torch.empty((b, 128, qstride), dtype=torch.bfloat16, device="cuda")
    q_pe = q_storage[..., -64:]
    kn = torch.empty((b, 512), dtype=torch.bfloat16, device="cuda")
    k_storage = torch.empty((b, 2112), dtype=torch.bfloat16, device="cuda")
    k_pe = k_storage[:, 2048:]
    qa = torch.empty((b, 128, 128), dtype=torch.uint8, device="cuda")
    slots = case.slots()
    graph = None
    assert (
        q_pe.stride(1),
        kn.stride(0),
        k_pe.stride(0),
        ring.stride(0),
        ring.stride(1),
        rope.cos_cache.stride(0),
    ) == (qstride, 512, 2112, 562176, 576, 32)

    def run():
        mla_nvfp4_decode_prep(
            q_pe,
            qa,
            ring,
            case.req,
            kn,
            k_pe,
            case.pos,
            slots,
            cn,
            ca,
            rope.cos_cache,
            rope.sin_cache,
        )

    for step in range(3):
        if padding is None:
            # Replay the same captured shape as the scheduler changes live rows.
            case.live = live = (b, b // 2, 0)[step]
        q = values(case.gen, b, 128, 576, s=2.0)
        k = values(case.gen, b, 576, s=0.5)
        q_pe.copy_(q[..., 512:])
        kn.copy_(k[:, :512])
        k_pe.copy_(k[:, 512:])
        stale = torch.randint(
            0, 256, qa.shape, generator=case.gen, dtype=torch.uint8, device="cuda"
        )
        qa.copy_(stale)
        slots.copy_(case.slots())
        if step == 0:
            run()
        else:
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
            graph.replay()
        want = reference_step(case, ref_ring, ref_cn, ref_ca, q, k, rope, True)
        torch.cuda.synchronize()
        assert torch.equal(qa[:live, :, :64], want[:live, :, :64])
        assert torch.equal(qa[:live, :, 96:], want[:live, :, 96:])
        assert torch.equal(qa[..., 64:96], stale[..., 64:96])
        assert (qa[live:, :, :64] == 0).all()
        assert (qa[live:, :, 96:104] == 0).all()
        assert (qa[live:, :, 104:] == 127).all()
        assert torch.equal(ring, ref_ring)
        assert torch.equal(cn, ref_cn), f"step {step}: kv_nope differs"
        assert torch.equal(ca, ref_ca), f"step {step}: kv_aux differs"
        assert (backing[:, :17] == 0.75).all() and (backing[:, 18:] == 0.75).all()
        case.pos.add_(1)


# The script benchmark uses normal finite inputs. The pytest cases above also
# cover tiny values/FTZ, stale cache bytes, changing positions and padding.
def run_torch(q, cos, sin, positions):
    """Independent BF16 RoPE + FP4 base/residual oracle, without native producers."""
    cs = cos.reshape(-1, 32)[positions].float()[:, None, :]
    sn = sin.reshape(-1, 32)[positions].float()[:, None, :]
    even, odd = q[..., 0::2].float(), q[..., 1::2].float()
    # BF16 multiply followed by a BF16 FMA, not FP32 RoPE. Double precision
    # retains the exact finite BF16 product/add before the final BF16 rounding.
    te = (-odd * sn).bfloat16().double()
    to = (even * sn).bfloat16().double()
    roped = torch.empty_like(q)
    roped[..., 0::2] = (even.double() * cs.double() + te).bfloat16()
    roped[..., 1::2] = (odd.double() * cs.double() + to).bfloat16()
    f4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    gain = torch.tensor(6.72, dtype=torch.float32, device=q.device)

    def quant(x):
        blocks = x.reshape(*x.shape[:-1], 4, 16)
        amax = blocks.abs().amax(-1)
        ideal = torch.where(amax == 0, 1.0, amax / 6.0)
        sf = (ideal * gain).clamp(1 / 512, 448).to(torch.float8_e4m3fn)
        divisor = sf.float() / gain
        v = blocks / divisor[..., None]
        hi = torch.bucketize(v.abs().contiguous(), f4).clamp_max(7)
        lo = (hi - 1).clamp_min(0)
        dl, dh = v.abs() - f4[lo], f4[hi] - v.abs()
        code = torch.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi)
        code = (code | (torch.signbit(v).long() << 3)).to(torch.uint8)
        vals = torch.where(
            (code & 8) != 0, -f4[(code & 7).long()], f4[(code & 7).long()]
        )
        deq = (vals * sf.float()[..., None]) / gain
        packed = code[..., 0::2] | (code[..., 1::2] << 4)
        return packed.flatten(-2), sf.view(torch.uint8), deq.reshape_as(x)

    bc, bs, deq = quant(roped.float())
    ec, es, _ = quant(roped.float() - deq)
    aux = torch.full((*q.shape[:2], 128), 127, dtype=torch.uint8, device=q.device)
    aux[..., :32], aux[..., 32:64] = bc, ec
    aux[..., 64:96] = 106
    aux[..., 96:100], aux[..., 100:104] = bs, es
    return roped, aux


@benchmark()
def bench_decode_prep(b, layout):
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.test_common import checkAllclose, run_perftest

    torch.manual_seed(174)
    # DSR1 q_b_proj gives 192 BF16 columns/head; ring has a 61-layer stride.
    stride = 192 if layout == "dsr1" else 200
    q = torch.randn((b, H, stride), dtype=dtypes.bf16, device="cuda")[..., -64:]
    kn = torch.randn((b, 512), dtype=dtypes.bf16, device="cuda")
    kr = torch.randn((b, 2112), dtype=dtypes.bf16, device="cuda")[:, 2048:]
    pos = torch.arange(b, device="cuda", dtype=torch.int64) % 64
    req = torch.arange(b, device="cuda", dtype=torch.int32)
    slots = req.long() * 64 + pos
    backing = torch.full((b, 61, 16, 576), 0.75, dtype=dtypes.bf16, device="cuda")
    ring = backing[:, 17]
    qa = torch.full((b, H, 128), 106, dtype=torch.uint8, device="cuda")
    cn = torch.full((b, 16384), 90, dtype=torch.uint8, device="cuda")
    ca = torch.full((b, 8192), 127, dtype=torch.uint8, device="cuda")
    r = make_rope()
    _, want = run_torch(q, r.cos_cache, r.sin_cache, pos)
    candidates = {
        "decode_prep": lambda: mla_nvfp4_decode_prep(
            q, qa, ring, req, kn, kr, pos, slots, cn, ca, r.cos_cache, r.sin_cache
        )
    }
    # Nominal RoPE FLOPs only (6/pair); quantization arithmetic is excluded.
    flops = 6 * 32 * (H + 1) * b
    # Logical global bytes, not measured HBM transactions. Count each input
    # once, including shared cos/sin and metadata, and all output writes:
    # Q: 128 BF16 bytes + 96 AUX bytes per head;
    # K: new BF16 row read/write, historical NoPE rows read, the full 16-row
    # NoPE code/scale tile written, and 72 new RoPE code/scale bytes.
    # At tile start, another 1464 AUX bytes initialize later rows/padding.
    # Excludes repeated lane loads, LDS traffic and cache-line amplification.
    rn = pos % 16
    nbytes = b * (H * (128 + 96) + 2 * 576 * 2 + 4608 + 72 + 128 + 20)
    nbytes += int(rn.sum()) * 512 * 2 + int((rn == 0).sum()) * 1464
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        err = checkAllclose(want.float(), qa.float(), rtol=0, atol=0, msg=name)
        assert err == 0, f"{name}: Q RoPE packed output differs"
        assert torch.equal(ring[req.long(), pos % 16, :512], kn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# Keep imports and GPU initialization inside main for collection on other hosts.
def main():
    import argparse
    import itertools

    import pandas as pd

    import aiter
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning(
            "decode_prep unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(description="NVFP4 decode-prep batch/layout sweep")
    # Current DSR1 deployment: about 2200-2300 live requests per rank.
    # This is the benchmark focus, not a kernel batch-size restriction.
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="+",
        default=[1536, 2048, 2176, 2200, 2240, 2300],
    )
    parser.add_argument(
        "--layout", nargs="+", choices=["dsr1", "generic"], default=["dsr1", "generic"]
    )
    args = parser.parse_args()
    rows = [
        bench_decode_prep(b, layout)
        for b, layout in itertools.product(args.batch, args.layout)
    ]
    aiter.logger.info(
        "decode_prep summary (RoPE-only FLOPs; logical bytes):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
