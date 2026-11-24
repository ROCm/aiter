# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape MoE expert GEMM pair (``decode_expert_gemms``), E128 top4, H6144 / I768.

Two checks, both on a gfx950 GPU:

* ``test_decode_expert_gemms_matches_torch``: an independent fp32 torch reference (no
  amd-kernels, no aiter kernel code). Operands are built from logical tensors and packed into
  the kernels' preshuffled byte layouts with plain ``view`` / ``permute``; the reference
  dequantizes them, runs the routed expert (GEMM1, SwiGLU-OAI, MXFP4 requant, GEMM2, top-k
  weight) and the shared expert (GEMM1, SwiGLU-OAI, MXFP8 requant, GEMM2) in fp32 and compares
  with the BF16 kernel outputs. The intermediate requantization and the fp32 summation order
  differ from the kernels' (a code on a rounding tie can flip), so the gate is a relative
  Frobenius norm: ``TOL``. Weight scales are constant per expert (a scale-layout permutation
  inside an expert is only covered by the bit-identical test below); the activation scales
  vary per row and K group.
* ``test_decode_expert_gemms_matches_amd_kernels``: optional extra, skipped when amd-kernels
  is not installed. The id-rows host must reproduce amd-kernels' tiles bit for bit on identical
  random operands, for every tiling ``tiles_for`` selects. aiter's kernels do not import it.
"""

import pytest
import torch

pytest.importorskip("flydsl")
from aiter.jit.utils.chip_info import get_gfx_runtime

if not torch.cuda.is_available() or not get_gfx_runtime().startswith("gfx95"):
    pytest.skip("needs a gfx950 GPU", allow_module_level=True)

from aiter.ops.flydsl.kernels.mega_moe_tp import decode_expert_gemms as D

H, I, E, TOPK = 6144, 768, 128, 4

# Relative Frobenius error ||out - ref|| / ||ref|| allowed against the fp32 reference. Measured on
# gfx950 at 8..128 rows: 0.0016-0.0017 for both outputs (the BF16 output rounding, 2^-9 per
# element, dominates; the requantization codes agree with the kernels'). The gate is 6x above it
# and ~2 orders below the error of any layout, expert-offset or K-order mistake (O(1)).
TOL = 0.01

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _rand_bytes(shape, g, dev):
    return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).to(dev)


def _f8(shape, g, dev, std):
    """e4m3 bytes of N(0, std) values."""
    x = torch.randn(shape, generator=g) * std
    return x.to(torch.float8_e4m3fn).view(torch.uint8).to(dev)


def _fp4_to_f32(codes):
    """(..., K/2) packed MXFP4 bytes (low nibble first) -> (..., K) fp32 values."""
    tab = torch.tensor(FP4_VALUES + tuple(-v for v in FP4_VALUES), device=codes.device)
    lo, hi = (codes & 15).long(), (codes >> 4).long()
    return torch.stack((tab[lo], tab[hi]), dim=-1).flatten(-2)


def _e8m0(e):
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e.long() - 127)


def _fp4_quant_dequant(y):
    """The kernels' MXFP4 requantization contract, per 32 columns: e8m0 exponent
    ``(bits(amax) - 0xE00000) >> 23`` floored at 1 (amax / 2^(e-127) lands in (0, 7]), then
    round-to-nearest-even onto the e2m1 grid with saturation at 6."""
    yg = y.view(*y.shape[:-1], -1, 32)
    amax = yg.abs().amax(-1, keepdim=True)
    e = ((amax.view(torch.int32) - 0xE00000) >> 23).clamp(min=1)
    q = (yg / _e8m0(e)).abs().clamp(max=6.0)
    # ties go to the even code: 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2, 2.5->2, 3.5->4, 5->4
    edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=y.device)
    ties_up = torch.tensor([0, 1, 0, 1, 0, 1, 0], device=y.device, dtype=torch.bool)
    idx = torch.searchsorted(edges, q.contiguous())  # edges[idx - 1] < q <= edges[idx]
    at = idx.clamp(max=6)
    idx = idx + ((idx < 7) & (q == edges[at]) & ties_up[at]).long()
    grid = torch.tensor(FP4_VALUES, device=y.device)
    return (grid[idx] * torch.sign(yg) * _e8m0(e)).view_as(y)


def _fp8_quant_dequant(y):
    """MXFP8 requantization contract per 32 columns: e8m0 = ceil_pow2(amax / 448) floored at
    1, values / scale rounded to e4m3 (saturating)."""
    yg = y.view(*y.shape[:-1], -1, 32)
    amax = yg.abs().amax(-1, keepdim=True)
    u = (amax / 448.0).view(torch.int32)
    e = (u >> 23) & 255
    e = torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp(min=1)
    s = _e8m0(e)
    q = (yg / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s).view_as(y)


def _swiglu(g, u):
    """SwiGLU-OAI: clamp(g, max=7), clamp(u, -7, 7), g * sigmoid(1.702 g) * (u + 1)."""
    g = g.clamp(max=7.0)
    u = u.clamp(-7.0, 7.0)
    return g * torch.sigmoid(1.702 * g) * (u + 1.0)


def _pack_fp4(w, tiles_k):
    """(..., N, K/2) logical MXFP4 bytes -> kernel layout [N/16][kstep][q16][l16][16 B]
    (``tiles_k`` = K bytes / 64)."""
    *lead, n, _ = w.shape
    w = w.reshape(*lead, n // 16, 16, tiles_k, 4, 16)
    nl = len(lead)
    perm = tuple(range(nl)) + tuple(nl + i for i in (0, 2, 3, 1, 4))
    return w.permute(*perm).contiguous()


def _pack_fp8(w):
    """(N, K) logical MXFP8 bytes -> kernel layout [N/16][kstep][half][q16][l16][16 B]."""
    n, k = w.shape
    return (
        w.reshape(n // 16, 16, k // 128, 2, 4, 16)
        .permute(0, 2, 3, 4, 1, 5)
        .contiguous()
    )


def _build(n, dev, g):
    """Logical operands and the kernel-layout tensors packed from them."""
    # magnitudes chosen so gate/up pre-activations are O(1): the SwiGLU clamp stays rare
    x_e = torch.randint(122, 125, (n, H // 32), generator=g, dtype=torch.uint8)
    xq = _rand_bytes((n, H // 2), g, dev)
    xs = x_e.to(dev)
    q8 = _f8((n, H), g, dev, 0.15)
    s8 = torch.randint(125, 128, (n, H // 32), generator=g, dtype=torch.uint8).to(dev)

    w13 = _rand_bytes((E, 2, I, H // 2), g, dev)  # [gate rows, up rows] per expert
    w2 = _rand_bytes((E, H, I // 2), g, dev)
    e13 = torch.randint(121, 124, (E,), generator=g, dtype=torch.uint8)
    e2 = torch.randint(121, 124, (E,), generator=g, dtype=torch.uint8)
    sw13 = _f8((2 * I, H), g, dev, 0.15)
    sw2 = _f8((H, I), g, dev, 0.15)

    # kernel layouts: gate/up tiles interleave per 16 columns: [tile][gate, up]
    p13 = _pack_fp4(w13.view(E, 2, I, H // 2), H // 128)  # (E, 2, I/16, 48, 4, 16, 16)
    p13 = p13.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    packed = (
        p13.view(-1),
        e13.repeat_interleave(I * H // 16).to(dev),
        _pack_fp4(w2, I // 128).view(-1),
        e2.repeat_interleave(H * I // 32).to(dev),
    )
    shared = (
        _pack_fp8(sw13).view(-1),
        torch.full((I * (H // 16),), 127, dtype=torch.uint8, device=dev),
        _pack_fp8(sw2).view(-1),
        torch.full((H * (I // 32),), 127, dtype=torch.uint8, device=dev),
    )
    ids = (
        torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(n)])
        .to(torch.int32)
        .to(dev)
    )
    wt = torch.rand((n, TOPK), generator=g).to(dev)
    logical = (xq, xs, q8, s8, w13, e13.to(dev), w2, e2.to(dev), sw13, sw2)
    return logical, (xq, xs, q8, s8, packed, shared, ids, wt)


def _reference(n, logical, ids, wt):
    """fp32 routed + shared expert outputs from the logical operands."""
    xq, xs, q8, s8, w13, e13, w2, e2, sw13, sw2 = logical
    x = _fp4_to_f32(xq) * _e8m0(xs).repeat_interleave(32, dim=-1)
    x8 = q8.view(torch.float8_e4m3fn).float() * _e8m0(s8).repeat_interleave(32, dim=-1)
    slots = torch.zeros((n, TOPK, H), device=x.device)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        wg = _fp4_to_f32(w13[e]) * _e8m0(e13[e])  # (2, I, H)
        gate, up = x[tok] @ wg[0].T, x[tok] @ wg[1].T
        h = _fp4_quant_dequant(_swiglu(gate, up))
        w2e = _fp4_to_f32(w2[e]) * _e8m0(e2[e])  # (H, I)
        slots[tok, slot] = (h @ w2e.T) * wt[tok, slot].unsqueeze(-1)
    w13s = sw13.view(torch.float8_e4m3fn).float()
    gate, up = x8 @ w13s[:I].T, x8 @ w13s[I:].T
    h8 = _fp8_quant_dequant(_swiglu(gate, up))
    shared = h8 @ sw2.view(torch.float8_e4m3fn).float().T
    return slots, shared


def _rel(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("n", [8, 16, 24, 40, 64, 97, 128])
def test_decode_expert_gemms_matches_torch(n):
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(4321 + n)
    logical, args = _build(n, dev, g)
    plan = D.get_plan(n)
    out, sh_out = D.decode_expert_gemms(plan, *args)
    torch.cuda.synchronize()
    ids, wt = args[-2], args[-1]
    ref, ref_sh = _reference(n, logical, ids, wt)
    # non-degenerate: outputs of O(1) size, not an all-clamped or all-zero case
    assert 0.05 < ref.pow(2).mean().sqrt() < 500, ref.pow(2).mean().sqrt()
    assert 0.05 < ref_sh.pow(2).mean().sqrt() < 500, ref_sh.pow(2).mean().sqrt()
    err, err_sh = _rel(out, ref), _rel(sh_out, ref_sh)
    print(f"n={n}: routed rel err {err:.4f}, shared rel err {err_sh:.4f}")
    assert err < TOL, err
    assert err_sh < TOL, err_sh


def test_decode_expert_gemms_rejects_bad_operands():
    """The launch binds its buffers from ``plan.n``: a mismatched operand must raise."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(1)
    n = 8
    _, args = _build(n, dev, g)
    xq, xs, q8, s8, w, shared, ids, wt = args
    plan = D.get_plan(16)  # plan for 16 rows, operands for 8
    with pytest.raises(AssertionError):
        D.decode_expert_gemms(plan, xq, xs, q8, s8, w, shared, ids, wt)
    plan = D.get_plan(n)
    wide = torch.zeros((n, H), dtype=torch.uint8, device=dev)[:, ::2]
    with pytest.raises(AssertionError):
        D.decode_expert_gemms(plan, wide, xs, q8, s8, w, shared, ids, wt)
    with pytest.raises(AssertionError):
        D.decode_expert_gemms(plan, xq, xs, q8, s8, w, shared, ids.long(), wt)


# --- optional: bit-identical to amd-kernels' tiles -------------------------------------------


def _oracle(n, tiles, xq, xs, q8, s8, routed, shared, ids, wt):
    """amd-kernels' id-rows launches with the same tiling as ``DecodeExpertPlan``."""
    root = "amd_kernels.moe.mxfp4_h6144_i768_e128_top4"
    ak_sort = pytest.importorskip(f"{root}.sort_a16w4")
    ak_launch = pytest.importorskip(f"{root}.common.launch")
    ak_g1i = pytest.importorskip(f"{root}.gemm1_a4w4_id_rows")
    ak_g2i = pytest.importorskip(f"{root}.gemm2_a4w4_id_rows")
    ak_xs = pytest.importorskip(f"{root}.experts_a4w4_small")
    dev = xq.device
    kw, bm = ak_xs.small_config_for(n)
    sh_bm, sh_kw, sh_pf = ak_xs.shared_fold_tile_for(n)
    sh_bm, sh_pf = tiles.get("sh_bm", sh_bm), tiles.get("sh_pf", sh_pf)
    sh_tn = tiles.get("sh_tn", ak_xs.shared_g2_columns_for(n))
    g2_bm = ak_xs.shared_fold_down_rows_for(sh_bm, bm)
    rows = ak_sort.max_sorted_rows(n, E, TOPK, bm)
    k1 = ak_g1i.compile_gemm1_shared_first(
        kw=kw,
        pf=tiles.get("g1_pf", 3),
        bm=bm,
        sh_bm=sh_bm,
        sh_kw=sh_kw,
        sh_pf=sh_pf,
        pairs=ak_xs.id_pairs_for(n),
    )
    k2 = ak_g2i.compile_gemm2_shared_first(
        tn=tiles.get("g2_tn", 256),
        bm=bm,
        pf=tiles.get("g2_pf", 2),
        sh_tn=sh_tn,
        sh_bm=g2_bm,
    )
    nsh1 = (n + sh_bm - 1) // sh_bm * (24 // (4 // sh_kw))
    nsh2 = (n + g2_bm - 1) // g2_bm * (H // sh_tn)
    w13, s13, w2, s2 = routed
    sw13, ss13, sw2, ss2 = shared
    eids = torch.zeros(rows // 16 + 1, device=dev, dtype=torch.int32)
    nv = torch.zeros(rows, device=dev, dtype=torch.int32)
    q = torch.zeros((rows, 384), device=dev, dtype=torch.uint8)
    s = torch.zeros((rows, 24), device=dev, dtype=torch.uint8)
    out = torch.zeros((n, 4, H), device=dev, dtype=torch.bfloat16)
    hq = torch.zeros((n, 768), device=dev, dtype=torch.uint8)
    hs = torch.zeros((n, 24), device=dev, dtype=torch.uint8)
    sh_out = torch.zeros((n, H), device=dev, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dev)
    ak_launch._run_compiled(
        k1,
        xq.view(-1),
        xs.view(-1),
        w13,
        s13,
        q.view(-1),
        s.view(-1),
        ids.view(-1),
        eids,
        nv,
        q8.view(-1),
        s8.view(-1),
        sw13,
        ss13,
        hq.view(-1),
        hs.view(-1),
        n,
        rows,
        nsh1,
        stream,
    )
    ak_launch._run_compiled(
        k2,
        q.view(-1),
        s.view(-1),
        w2,
        s2,
        out.view(-1),
        nv,
        wt.view(-1),
        eids,
        nv,
        hq.view(-1),
        hs.view(-1),
        sw2,
        ss2,
        sh_out.view(-1),
        n,
        rows,
        nsh2,
        stream,
    )
    return out, sh_out


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _scales(shape, dev, g):
    return torch.randint(120, 128, shape, generator=g, dtype=torch.uint8).to(dev)


def _random_operands(n, dev, g):
    """Random bytes (the kernels' data path is layout-fixed, not value-aware): MXFP4 codes
    arbitrary, e8m0 scales near 2^0, MXFP8 codes below the e4m3 NaN encoding."""
    u8 = lambda *s: torch.randint(0, 256, s, generator=g, dtype=torch.uint8).to(dev)
    f8 = lambda *s: torch.randint(0, 0x70, s, generator=g, dtype=torch.uint8).to(dev)
    routed = (
        u8(E * I * H),
        _scales((E * I * (H // 16),), dev, g),
        u8(E * H * I // 2),
        _scales((E * H * (I // 32),), dev, g),
    )
    shared = (
        f8(2 * I * H),
        _scales((I * (H // 16),), dev, g),
        f8(H * I),
        _scales((H * (I // 32),), dev, g),
    )
    return routed, shared


def _random_inputs(n, dev, g):
    xq = torch.randint(0, 256, (n, H // 2), generator=g, dtype=torch.uint8).to(dev)
    xs = _scales((n, H // 32), dev, g)
    q8 = torch.randint(0, 0x70, (n, H), generator=g, dtype=torch.uint8).to(dev)
    s8 = _scales((n, H // 32), dev, g)
    ids = (
        torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(n)])
        .to(torch.int32)
        .to(dev)
    )
    wt = torch.rand((n, TOPK), generator=g).to(dev)
    return xq, xs, q8, s8, ids, wt


# every tiling band of ``tiles_for`` (8, 16, <=64, <=96, <=128 rows), with and without the 32-128 row
# prefetch / issue-priority tuning
CASES = [(n, False) for n in (8, 16, 24, 40, 64, 80, 96, 97, 112, 128)]
CASES += [(n, True) for n in (32, 48, 64, 96, 128)]


@pytest.mark.parametrize("n,asm", CASES)
def test_decode_expert_gemms_matches_amd_kernels(n, asm):
    pytest.importorskip("amd_kernels")
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(1234 + n)
    routed, shared = _random_operands(n, dev, g)
    tiles = D.tiles_for(n, asm)
    plan = D.DecodeExpertPlan(n, asm)
    for _ in range(2):
        xq, xs, q8, s8, ids, wt = _random_inputs(n, dev, g)
        out, sh_out = D.decode_expert_gemms(
            plan, xq, xs, q8, s8, routed, shared, ids, wt
        )
        torch.cuda.synchronize()
        ref_out, ref_sh = _oracle(n, tiles, xq, xs, q8, s8, routed, shared, ids, wt)
        torch.cuda.synchronize()
        assert ref_out.view(torch.int16).any(), "oracle output is all zero"
        assert torch.equal(_bits(out), _bits(ref_out))
        assert torch.equal(_bits(sh_out), _bits(ref_sh))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))


@pytest.mark.parametrize("n", [8, 24, 64, 128])
def test_decode_expert_gemms_w13_split_layout_is_bitwise_equal(n):
    """The routed w13 read in place in its preshuffled [E, 2, I/16, block] layout gives the same bits as the
    gate / up interleaved copy."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(777 + n)
    _, args = _build(n, dev, g)
    xq, xs, q8, s8, packed, shared, ids, wt = args
    w13, s13, w2, s2 = packed
    split = (
        w13.view(E, I // 16, 2, -1).transpose(1, 2).contiguous().view(-1),
        s13,
        w2,
        s2,
    )
    out, sh = D.decode_expert_gemms(D.get_plan(n), *args)
    out_s, sh_s = D.decode_expert_gemms(
        D.get_plan(n, w13_split=True), xq, xs, q8, s8, split, shared, ids, wt
    )
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), out_s.view(torch.int16))
    assert torch.equal(sh.view(torch.int16), sh_s.view(torch.int16))
