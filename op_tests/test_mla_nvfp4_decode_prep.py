# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run on gfx1250: python -m pytest op_tests/test_mla_nvfp4_decode_prep.py -q.

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
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx1250"):
        pytest.skip("requires gfx1250")


@pytest.fixture(scope="module")
def rope():
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
    "b,live,steps,seed", [(1, 1, 3, 0), (37, 30, 6, 1), (1536, 1519, 3, 2)]
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

    def call(q_rope=None, cos=rope.cos_cache, sin=rope.sin_cache, batch=b):
        mla_nvfp4_decode_prep(
            q[:batch, :, 512:] if q_rope is None else q_rope,
            qa[:batch],
            case.ring,
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
