# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape MoE expert GEMM pair (``decode_expert_gemms``): the id-rows host must
reproduce amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4`` expert tiles bit for bit on identical
operands, for every tiling ``tiles_for`` selects.

amd-kernels is only the oracle here (skipped when it is not installed); aiter's kernels do not
import it. Operands are random bytes (the kernels' data path is layout-fixed, not value-aware):
MXFP4 codes arbitrary, e8m0 scales kept near 2^0, MXFP8 codes kept below the e4m3 NaN encoding.
"""

import pytest
import torch

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx95" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("needs a gfx950 GPU", allow_module_level=True)
ak_sort = pytest.importorskip("amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sort_a16w4")
ak_launch = pytest.importorskip(
    "amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch"
)
ak_g1i = pytest.importorskip(
    "amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm1_a4w4_id_rows"
)
ak_g2i = pytest.importorskip(
    "amd_kernels.moe.mxfp4_h6144_i768_e128_top4.gemm2_a4w4_id_rows"
)
ak_xs = pytest.importorskip(
    "amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_small"
)

from aiter.ops.flydsl.kernels.mega_moe_tp import decode_expert_gemms as D

H, I, E, TOPK = 6144, 768, 128, 4


def _scales(shape, dev, g):
    return torch.randint(120, 128, shape, generator=g, dtype=torch.uint8).to(dev)


def _weights(dev, g):
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


def _inputs(n, dev, g):
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


def _oracle(n, tiles, xq, xs, q8, s8, routed, shared, ids, wt):
    """amd-kernels' id-rows launches with the same tiling as ``DecodeExpertPlan``."""
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


# every tiling band of ``tiles_for`` (8, 16, <=64, <=96, <=128 rows), with and without the 32-128 row
# prefetch / issue-priority tuning
CASES = [(n, False) for n in (8, 16, 24, 40, 64, 80, 96, 97, 112, 128)]
CASES += [(n, True) for n in (32, 48, 64, 96, 128)]


@pytest.mark.parametrize("n,asm", CASES)
def test_decode_expert_gemms_matches_amd_kernels(n, asm):
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(1234 + n)
    routed, shared = _weights(dev, g)
    tiles = D.tiles_for(n, asm)
    plan = D.DecodeExpertPlan(n, asm)
    for _ in range(2):
        xq, xs, q8, s8, ids, wt = _inputs(n, dev, g)
        out, sh_out = D.decode_expert_gemms(
            plan, xq, xs, q8, s8, routed, shared, ids, wt
        )
        torch.cuda.synchronize()
        ref_out, ref_sh = _oracle(n, tiles, xq, xs, q8, s8, routed, shared, ids, wt)
        torch.cuda.synchronize()
        assert ref_out.view(torch.int16).any(), "oracle output is all zero"
        assert torch.equal(_bits(out), _bits(ref_out))
        assert torch.equal(_bits(sh_out), _bits(ref_sh))
