# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""flydsl_sparse_mla_decode_fp8_h64 against an fp64 torch reference (gfx950).

The DeepSeek-V4-style decode call of ``sparse_mla_fwd`` with a per-tensor e4m3
cache, fp8 dots and the inverse-RoPE MXFP8 store, as vLLM's DeepSeek-V4.1
DP-attention path makes it:

    q              [T, 64, 512]           e4m3, per-tensor q_scale ([1] f32)
    kv_buffer      [blocks, 2^k, 512]     e4m3 sliding-window cache
    extra_kv       [blocks, 2^k, 512]     e4m3 compressed (top-k) cache, same kv_scale
    kv/extra       indptr [T + 1], indices: int32, <= 128 + 512 keys per row
    attn_sink      [64] f32
    out_mxfp8      codes [T, 32768] e4m3, scales [T, 1024] E8M0 (inverse RoPE
                   on dims 448..511 at inv_rope_positions), or bf16 out [T, 64, 512]

Served shapes: 1..6 rows per request, 16..768 rows, cache blocks of 32 / 64 /
128 rows, a padded page pitch, the window alone, padding rows with empty lists,
rows without keys over empty index tensors. Every case checks the bf16 output
against the fp64 reference (per (row, head) relative L2 error and per-row
cosine), the MXFP8 bytes against the reference quantization (inverse RoPE +
E8M0 per 32 dims) of that bf16 output, repeat launches bit for bit, CUDA graph
replay into NaN-filled outputs. Every cache row no list names holds e4m3 NaN
bytes, so a wrong gather shows. The saturation tests drive scores of hundreds
to 1e5 log2 units (q codes at +-448, large scales, score jumps between tiles).

Run (the checks, then a perf sweep that ends in a markdown table):
    python3 op_tests/test_flydsl_sparse_mla_decode_fp8_h64.py
    python3 op_tests/test_flydsl_sparse_mla_decode_fp8_h64.py -b 16 768
    pytest -q op_tests/test_flydsl_sparse_mla_decode_fp8_h64.py
"""

import argparse
import itertools
import math
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

H, D, ROT = 64, 512, 64
FP8_MAX = 448.0
TINY_F32 = torch.finfo(torch.float32).tiny
SOFTMAX_SCALE = D**-0.5
NAN_BYTE = 0x7F
SAT = 0x7E  # e4m3 +-448
COS_MIN = 0.9999
REL = 0.02
WINDOW, TOPK = 128, 512
# CDNA4 scaled MFMAs, ds_read_b64_tr_b8 and 160 KB of LDS
SUPPORTED_GFX = ["gfx950"]


def _supported() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return get_gfx_runtime() in SUPPORTED_GFX


pytestmark = pytest.mark.skipif(not _supported(), reason="gfx950 required")


def _op():
    from aiter.ops.flydsl import flydsl_sparse_mla_decode_fp8_h64

    return flydsl_sparse_mla_decode_fp8_h64


def _can_use():
    from aiter.ops.flydsl import can_use_flydsl_sparse_mla_decode_fp8_h64

    return can_use_flydsl_sparse_mla_decode_fp8_h64


# ----------------------------------------------------------------------------
# fp64 reference
# ----------------------------------------------------------------------------


def _rows(cache, indices):
    if cache.dim() == 2:
        return cache.view(torch.float8_e4m3fn)[indices.long()]
    block, stride = cache.shape[1], cache.stride(0) // D
    idx = indices.long()
    row = (idx // block) * stride + idx % block
    flat = cache.view(torch.float8_e4m3fn).as_strided(
        (cache.shape[0] * stride, D), (D, 1)
    )
    return flat[row]


def attention_reference(case):
    """fp64 softmax over the listed keys and the sink, rounded to bf16."""
    q, rows = case["q"], case["q"].shape[0]
    out = torch.zeros((rows, H, D), dtype=torch.float64, device=q.device)
    qf = q.view(torch.float8_e4m3fn).double() * float(case["q_scale"])
    ks = float(case["kv_scale"])
    mp = case["kv_indptr"].tolist()
    xp = None if case["extra_kv"] is None else case["extra_indptr"].tolist()
    sink = case["attn_sink"].double()
    for t in range(rows):
        parts = [_rows(case["kv_buffer"], case["kv_indices"][mp[t] : mp[t + 1]])]
        if xp is not None:
            parts.append(
                _rows(case["extra_kv"], case["extra_indices"][xp[t] : xp[t + 1]])
            )
        kv = torch.cat(parts).double() * ks
        s = torch.einsum("hd,kd->hk", qf[t], kv) * case["softmax_scale"]
        p = torch.softmax(torch.cat([s, sink[:, None]], dim=1), dim=1)[:, :-1]
        out[t] = p @ kv
    return out.bfloat16()


def inverse_rope(x, positions, cos_sin):
    """fp32 rows of bf16 ``x`` with dims 448..511 rotated back (GPT-J pairs):
    ``fma(even, cos, odd * sin)``, ``fma(odd, cos, -(even * sin))``."""
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, None, : ROT // 2], cs[:, None, ROT // 2 :]
    y = x.float().clone()
    even, odd = y[..., D - ROT :: 2].clone(), y[..., D - ROT + 1 :: 2].clone()
    y[..., D - ROT :: 2] = (even.double() * cos.double() + (odd * sin).double()).float()
    y[..., D - ROT + 1 :: 2] = (
        odd.double() * cos.double() - (even * sin).double()
    ).float()
    return y


def mxfp8_reference(x):
    """fp32 [rows, K] -> (e4m3 codes as uint8, E8M0 scales uint8 per 32)."""
    rows, k = x.shape
    v = x.float().view(rows, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(TINY_F32)
    e = (torch.ceil(torch.log2((amax / FP8_MAX).double())) + 127.0).clamp(0.0, 254.0)
    codes = (v * torch.exp2(127.0 - e).float()[..., None]).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8).view(rows, k), e.to(torch.uint8)


def epilogue_reference(out_bf16, positions, cos_sin):
    rows = out_bf16.shape[0]
    return mxfp8_reference(inverse_rope(out_bf16, positions, cos_sin).view(rows, H * D))


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------


def e4m3(shape, std, gen):
    x = torch.randn(shape, generator=gen, device="cuda") * std
    return x.to(torch.float8_e4m3fn)


def row_lengths(qlens, window=128, topk=512, ctx=(300, 100000), seed=0):
    """(n_main, n_extra, position) per query row: request r has ``qlens[r]``
    rows at the last positions of a context of random length."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for ql in qlens:
        c = int(torch.randint(ctx[0], ctx[1], (1,), generator=gen))
        for j in range(ql):
            p = c + j
            out.append((min(window, p + 1), min(topk, p // 2), p))
    return out


def poison(pool, used, block):
    """e4m3 NaN in every row of ``pool`` [blocks, block + pitch, 512] that no
    slot id in ``used`` names."""
    keep = torch.zeros(pool.shape[0] * block, dtype=torch.bool)
    keep[used.long()] = True
    keep = keep.view(pool.shape[0], block).cuda()
    u8 = pool.view(torch.uint8)
    view = u8[:, :block]
    view[~keep] = NAN_BYTE
    if pool.shape[1] > block:
        u8[:, block:] = NAN_BYTE


def make_case(
    lens,
    *,
    pad=0,
    swa_block=32,
    comp_block=64,
    has_extra=True,
    q_std=1.5,
    k_std=1.0,
    seed=0,
    pitch=0,
    pool_rows=1 << 16,
):
    """``lens``: [(n_main, n_extra, position)] per query row, then ``pad`` rows
    with empty lists; ``pitch`` extra rows per cache block. Returns the op's
    arguments (a dict)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed + 1)
    rows = len(lens) + pad
    n_swa = max(sum(n for n, _, _ in lens), 1)
    swa_rows = -(-max(2 * n_swa, 4096) // swa_block) * swa_block
    comp_rows = -(-pool_rows // comp_block) * comp_block
    swa = e4m3((swa_rows // swa_block, swa_block + pitch, D), k_std, gen)
    comp = (
        e4m3((comp_rows // comp_block, comp_block + pitch, D), k_std, gen)
        if has_extra
        else None
    )
    mp, xp, mi, xi = [0], [0], [], []
    for n_main, n_x, _ in lens:
        mi.append(torch.randperm(swa_rows, generator=cpu)[:n_main])
        mp.append(mp[-1] + n_main)
        if has_extra:
            xi.append(torch.randperm(comp_rows, generator=cpu)[:n_x])
            xp.append(xp[-1] + n_x)
    mp += [mp[-1]] * pad
    xp += [xp[-1]] * pad
    empty = torch.zeros(0, dtype=torch.long)
    mi = torch.cat(mi) if mi else empty
    xi = torch.cat(xi) if xi else empty
    poison(swa, mi, swa_block)
    swa = swa[:, :swa_block]
    if has_extra:
        poison(comp, xi, comp_block)
        comp = comp[:, :comp_block]
    pos = [p for _, _, p in lens] + [0] * pad
    ang = torch.rand((max(pos) + 8, 32), generator=gen, device="cuda") * (2 * math.pi)

    def dev(x):
        return x.int().cuda()

    return {
        "q": e4m3((rows, H, D), q_std, gen),
        "kv_buffer": swa,
        "kv_indptr": dev(torch.tensor(mp)),
        "kv_indices": dev(mi),
        "softmax_scale": SOFTMAX_SCALE,
        "kv_scale": torch.full((1,), 1.0, device="cuda"),
        "q_scale": torch.full((1,), 1.0, device="cuda"),
        "attn_sink": torch.randn(H, generator=gen, device="cuda"),
        "extra_kv": comp,
        "extra_indptr": dev(torch.tensor(xp)) if has_extra else None,
        "extra_indices": dev(xi) if has_extra else None,
        "inv_rope_positions": dev(torch.tensor(pos)),
        "inv_rope_cos_sin_cache": torch.cat([ang.cos(), ang.sin()], dim=1).float(),
        # sparse_mla_fwd's arguments for this call, and the caller's key bound
        "kv_lora_rank": D,
        "qk_rope_head_dim": 0,
        "dot_precision": "fp8",
        "max_keys_per_row": WINDOW + TOPK,
    }


def mxfp8_outputs(rows):
    """NaN codes and 255 scales, so a byte the kernel does not write shows."""
    codes = torch.empty((rows, H * D), dtype=torch.float8_e4m3fn, device="cuda")
    codes.view(torch.uint8).fill_(NAN_BYTE)
    return codes, torch.full((rows, H * D // 32), 255, dtype=torch.uint8, device="cuda")


def launch(case, *, mxfp8, **kw):
    rows = case["q"].shape[0]
    args = dict(case)
    if mxfp8:
        out = mxfp8_outputs(rows)
        _op()(**args, out_mxfp8=out, **kw)
        return out
    args.pop("inv_rope_positions")
    args.pop("inv_rope_cos_sin_cache")
    out = torch.full((rows, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    _op()(**args, out=out, **kw)
    return out


def rel_error(out, ref):
    """Relative L2 error per (row, head); rows without keys compare to zero."""
    o, r = out.double(), ref.double()
    return ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).nan_to_num(nan=9.0)


def check_attention(out, ref, name, rel=REL):
    assert not torch.isnan(out.float()).any(), name
    empty = (ref.flatten(1) == 0).all(dim=1)
    assert (out[empty] == 0).all(), name  # a row without keys is zero
    live = ~empty
    if not live.any():
        return
    err = rel_error(out[live], ref[live])
    assert err.max() <= rel, (name, err.max().item())
    cos = torch.nn.functional.cosine_similarity(
        out[live].double().flatten(1), ref[live].double().flatten(1), dim=1
    )
    assert cos.min() >= COS_MIN, (name, cos.min().item())


def check_mxfp8(codes, scales, out_bf16, case, name):
    """The MXFP8 bytes equal the reference epilogue of the kernel's bf16 rows."""
    rc, rs = epilogue_reference(
        out_bf16, case["inv_rope_positions"], case["inv_rope_cos_sin_cache"]
    )
    for got, want, what in (
        (codes.view(torch.uint8), rc, "codes"),
        (scales, rs, "scales"),
    ):
        err = checkAllclose(
            want.float(),
            got.float(),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{name} {what}: ",
            printLog=False,
        )
        assert err == 0, (name, what, err)


# ----------------------------------------------------------------------------
# served shapes
# ----------------------------------------------------------------------------

EDGE = [
    (1, 0, 5),
    (1, 1, 7),
    (63, 0, 70),
    (64, 64, 300),
    (65, 63, 301),
    (128, 512, 4000),
    (128, 511, 4000),
    (5, 64, 9),
    (100, 2, 101),
    (128, 449, 2000),
    (2, 3, 3),
    (128, 130, 300),
    (64, 0, 64),
    (0, 0, 10),
    (0, 64, 128),
    (128, 512, 9000),
    (31, 33, 100),
    (1, 512, 1500),
]
# (id, requests' row counts, make_case kwargs)
CASES = [
    ("t16_mixed", [6, 6, 4], {}),
    ("t48_ql6", [6] * 8, {}),
    ("t96_mixed", [6, 1, 3, 2, 6, 5, 4, 6, 2, 6, 1, 3, 6, 5, 6, 6, 6], {}),
    ("t192_ql6", [6] * 32, {}),
    ("t30_pad", [6, 6, 6, 6, 6], {"pad": 3}),
    ("t384_ql6", [6] * 64, {"pool_rows": 1 << 17}),
    ("t100_blocks128", [6] * 16 + [4], {"swa_block": 128, "comp_block": 128}),
    ("t100_blocks64_32", [6] * 16 + [4], {"swa_block": 64, "comp_block": 32}),
    ("t60_pitch", [6] * 10, {"pitch": 8}),
    ("t60_swa_only", [6] * 10, {"has_extra": False}),
    ("t60_peaked", [6] * 10, {"q_std": 4.0}),
    ("t60_flat", [6] * 10, {"q_std": 0.25}),
    ("t768_ql6", [6] * 128, {"pool_rows": 1 << 17}),
]
CASE_KW = {c[0]: c for c in CASES}


def build(name):
    if name == "edge":
        return make_case(EDGE)
    _, spec, kw = CASE_KW[name]
    return make_case(row_lengths(spec), **kw)


@pytest.mark.parametrize("name", [c[0] for c in CASES] + ["edge"])
@torch.inference_mode()
def test_decode(name):
    case = build(name)
    out = launch(case, mxfp8=False)
    check_attention(out, attention_reference(case), name)
    codes, scales = launch(case, mxfp8=True)
    check_mxfp8(codes, scales, out, case, name)
    again = launch(case, mxfp8=True)
    assert torch.equal(again[0].view(torch.uint8), codes.view(torch.uint8)), name
    assert torch.equal(again[1], scales), name
    assert torch.equal(launch(case, mxfp8=False), out), name


@pytest.mark.parametrize("name", ["t48_ql6", "t30_pad", "t60_pitch", "t768_ql6"])
@torch.inference_mode()
def test_graph_replay(name):
    case = build(name)
    codes, scales = launch(case, mxfp8=True)
    ref_codes, ref_scales = codes.view(torch.uint8).clone(), scales.clone()
    _op()(**case, out_mxfp8=(codes, scales))  # compiled before the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _op()(**case, out_mxfp8=(codes, scales))
    for _ in range(3):
        codes.view(torch.uint8).fill_(NAN_BYTE)
        scales.fill_(255)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(codes.view(torch.uint8), ref_codes), name
        assert torch.equal(scales, ref_scales), name


# (heads per workgroup, workgroups per XCD): the three head-group kernels, and
# the persistent 64-head one with few workgroups, each running several rows
HEAD_GROUPS = [(16, None), (32, None), (64, None), (64, 1), (64, 3)]


@pytest.mark.parametrize(
    "hg,wgs", HEAD_GROUPS, ids=[f"hg{h}_wgs{w}" for h, w in HEAD_GROUPS]
)
@pytest.mark.parametrize("name", ["edge", "t60_pitch", "t96_mixed", "t100_blocks128"])
@torch.inference_mode()
def test_head_groups(name, hg, wgs):
    case = build(name)
    kw = {"heads_per_workgroup": hg, "workgroups_per_xcd": wgs}
    out = launch(case, mxfp8=False, **kw)
    check_attention(out, attention_reference(case), name)
    codes, scales = launch(case, mxfp8=True, **kw)
    check_mxfp8(codes, scales, out, case, name)
    again = launch(case, mxfp8=True, **kw)
    assert torch.equal(again[0].view(torch.uint8), codes.view(torch.uint8)), name


@torch.inference_mode()
def test_one_term_p():
    """p_terms=1: one e4m3 P, cosine ~0.9997 to the fp64 reference."""
    case = make_case(row_lengths([6] * 8), seed=5)
    out = launch(case, mxfp8=False, p_terms=1)
    ref = attention_reference(case)
    cos = torch.nn.functional.cosine_similarity(
        out.double().flatten(1), ref.double().flatten(1), dim=1
    )
    assert 0.999 <= cos.min() < COS_MIN
    codes, scales = launch(case, mxfp8=True, p_terms=1)
    check_mxfp8(codes, scales, out, case, "p1")


@pytest.mark.parametrize("rows_spec", [[6, 6, 4], [6] * 32])
@torch.inference_mode()
def test_matches_triton_sparse_mla_fwd(rows_spec):
    """The same call through aiter's Triton sparse_mla_fwd (fp8 dots, inverse
    RoPE + MXFP8 store): the dequantized outputs agree."""
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

    case = make_case(row_lengths(rows_spec), seed=7)
    codes, scales = launch(case, mxfp8=True)
    base = mxfp8_outputs(case["q"].shape[0])
    sparse_mla_fwd(
        case["q"],
        case["kv_buffer"],
        case["kv_indptr"],
        case["kv_indices"],
        case["softmax_scale"],
        kv_scale=case["kv_scale"],
        q_scale=case["q_scale"],
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        dot_precision="fp8",
        attn_sink=case["attn_sink"],
        extra_kv=case["extra_kv"],
        extra_indptr=case["extra_indptr"],
        extra_indices=case["extra_indices"],
        has_invalid=False,
        inv_rope_positions=case["inv_rope_positions"],
        inv_rope_cos_sin_cache=case["inv_rope_cos_sin_cache"],
        out_mxfp8=base,
    )

    def dequant(c, s):
        v = c.view(torch.float8_e4m3fn).double().view(c.shape[0], -1, 32)
        return (v * torch.exp2(s.double() - 127.0)[..., None]).view(c.shape[0], -1)

    cos = torch.nn.functional.cosine_similarity(
        dequant(codes, scales), dequant(*base), dim=1
    )
    assert cos.min() >= 0.998, cos.min().item()


# ----------------------------------------------------------------------------
# saturated and huge scores
# ----------------------------------------------------------------------------


def saturate(case, frac, seed=3):
    q = case["q"]
    if frac >= 1.0:
        sat = torch.ones(q.shape, dtype=torch.bool, device="cuda")
    else:
        g = torch.Generator(device="cuda").manual_seed(seed)
        sat = torch.rand(q.shape, generator=g, device="cuda") < frac
    qq = q.view(torch.uint8).clone()
    qq[sat] = (qq[sat] & 0x80) | SAT
    case["q"] = qq.view(q.dtype)
    return case


def check_finite_close(case, name, ties=False, rel_max=REL, **kw):
    out = launch(case, mxfp8=False, **kw)
    codes, scales = launch(case, mxfp8=True, **kw)
    assert torch.isfinite(out.float()).all(), name
    assert not ((codes.view(torch.uint8) & 0x7F) == 0x7F).any(), name
    assert not (scales == 255).any(), name
    err = rel_error(out, attention_reference(case))
    if ties:  # scores of ~1e4 log2 units: fp32 score noise decides near ties
        assert err.max() <= 0.1 and (err > REL).double().mean() <= 0.005, name
    else:
        assert err.max() <= rel_max, (name, err.max().item())
        assert (err > REL).double().mean() <= 0.005, name
    check_mxfp8(codes, scales, out, case, name)
    assert torch.equal(launch(case, mxfp8=False, **kw), out), name


VARIANTS = [
    {},
    {"heads_per_workgroup": 16},
    {"heads_per_workgroup": 32},
    {"heads_per_workgroup": 64},
    {"heads_per_workgroup": 64, "workgroups_per_xcd": 1},
]
VARIANT_IDS = ["default", "hg16", "hg32", "hg64", "hg64_wgs1"]


@pytest.mark.parametrize("kw", VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("frac", [0.01, 0.2, 1.0])
@torch.inference_mode()
def test_saturated_q(frac, kw):
    case = saturate(make_case(row_lengths([6] * 3)), frac)
    check_finite_close(case, f"sat{frac}", **kw)


@pytest.mark.parametrize("kw", VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("scale", [8.0, 1000.0])
@torch.inference_mode()
def test_large_scales(scale, kw):
    """Scores up to ~1e5 log2 units: the softmax is nearly one-hot."""
    case = saturate(make_case(row_lengths([6] * 3)), 0.05)
    case["q_scale"] = torch.full((1,), scale, device="cuda")
    case["kv_scale"] = torch.full((1,), 0.5, device="cuda")
    check_finite_close(case, f"scale{scale}", ties=True, **kw)


@pytest.mark.parametrize("kw", VARIANTS, ids=VARIANT_IDS)
@torch.inference_mode()
def test_masked_blocks_and_single_key(kw):
    """Last tiles with an all-masked 32-key half, single keys, the window
    alone, with saturated q."""
    lens = [
        (1, 0, 5),
        (0, 1, 7),
        (32, 0, 40),
        (33, 0, 40),
        (64, 0, 70),
        (65, 0, 70),
        (1, 64, 300),
        (64, 1, 300),
        (32, 33, 300),
        (128, 512, 4000),
        (127, 3, 4000),
        (0, 0, 10),
        (96, 1, 200),
        (2, 1, 3),
    ]
    check_finite_close(saturate(make_case(lens), 0.3), "masked", **kw)
    check_finite_close(saturate(make_case(EDGE), 1.0), "edge", **kw)


@pytest.mark.parametrize("kw", VARIANTS, ids=VARIANT_IDS)
@torch.inference_mode()
def test_score_jump_between_tiles(kw):
    """A few keys at +-448 among the top-k rows: the softmax reference moves by
    hundreds of log2 units at once."""
    case = make_case(row_lengths([6] * 4), q_std=1.0)
    gen = torch.Generator(device="cuda").manual_seed(5)
    flat = case["extra_kv"].view(torch.uint8).view(-1, D)
    used = case["extra_indices"].long()
    pick = used[torch.randint(0, used.numel(), (32,), generator=gen, device="cuda")]
    signs = (torch.rand((32, D), generator=gen, device="cuda") < 0.5).to(torch.uint8)
    flat[pick] = signs * 0x80 | SAT
    # a head dominated by the +-448 keys carries the P rounding: up to ~2.5%
    # here (sparse_mla_fwd's fp8 kernel: ~5% on the same case)
    check_finite_close(saturate(case, 0.1), "jump", rel_max=0.04, **kw)


@pytest.mark.parametrize("frac", [0.2, 1.0])
@torch.inference_mode()
def test_one_term_p_saturated(frac):
    case = saturate(make_case(row_lengths([6] * 3)), frac)
    out = launch(case, mxfp8=False, p_terms=1)
    codes, scales = launch(case, mxfp8=True, p_terms=1)
    assert torch.isfinite(out.float()).all()
    assert not ((codes.view(torch.uint8) & 0x7F) == 0x7F).any()
    assert not (scales == 255).any()
    assert rel_error(out, attention_reference(case)).max() <= 0.1


# ----------------------------------------------------------------------------
# eligibility: unsupported calls are declined, so a caller can fall back
# ----------------------------------------------------------------------------


def test_eligibility():
    case = make_case(row_lengths([6, 2]))
    rows = case["q"].shape[0]
    args = {k: v for k, v in case.items() if k != "softmax_scale"}
    can_use = _can_use()
    out = mxfp8_outputs(rows)

    def ok(**kw):
        return can_use(**dict(args, **kw))

    assert ok(out_mxfp8=out)
    assert ok(out_mxfp8=out, kv_splits=4)  # a split-K hint, ignored
    # the key bound must be stated: past 1024 keys the kernel would clamp
    assert not ok(out_mxfp8=out, max_keys_per_row=None)
    assert not ok(out_mxfp8=out, max_keys_per_row=2048)
    assert ok(out_mxfp8=out, max_keys_per_row=1024)
    # sparse_mla_fwd's arguments that change the result are declined
    assert not ok(out_mxfp8=out, dot_precision="bf16")
    assert not ok(out_mxfp8=out, return_lse=True)
    assert not ok(out_mxfp8=out, skip_reduce=True)
    assert not ok(out_mxfp8=out, qk_rope_head_dim=64)
    assert not ok(out_mxfp8=out, kv_lora_rank=448)
    defaults = {
        k: v
        for k, v in args.items()
        if k not in ("kv_lora_rank", "qk_rope_head_dim", "dot_precision")
    }
    assert not can_use(**defaults, out_mxfp8=out)  # sparse_mla_fwd's defaults
    assert not ok(out_mxfp8=out, has_invalid=True)
    assert not ok()  # no output
    other_scale = torch.ones(1, device="cuda")
    assert not ok(out_mxfp8=out, extra_kv_scale=other_scale)
    assert ok(out_mxfp8=out, extra_kv_scale=case["kv_scale"])
    assert not ok(out_mxfp8=out, attn_sink=None)
    no_rope = {"inv_rope_positions": None, "inv_rope_cos_sin_cache": None}
    assert not ok(out_mxfp8=out, **no_rope)
    bf16 = torch.empty((rows, H, D), dtype=torch.bfloat16, device="cuda")
    assert ok(out=bf16, **no_rope)
    assert not ok(out=bf16)  # bf16 rows carry no inverse RoPE
    assert not ok(q=case["q"][:, :32], out_mxfp8=out)
    assert not ok(q=case["q"].to(torch.bfloat16), out_mxfp8=out)
    assert not ok(kv_buffer=case["kv_buffer"][:, :24], out_mxfp8=out)
    assert not ok(kv_indices=case["kv_indices"].long(), out_mxfp8=out)
    strided = torch.empty((rows, 2 * H * D), dtype=torch.float8_e4m3fn, device="cuda")
    assert not ok(out_mxfp8=(strided[:, : H * D], out[1]))
    # 16-byte vector accesses: misaligned bases and cos / sin rows
    raw = torch.empty(rows * H * D + 16, dtype=torch.uint8, device="cuda")
    q_off = raw[1 : 1 + rows * H * D].view(torch.float8_e4m3fn).view(rows, H, D)
    assert q_off.is_contiguous() and not ok(q=q_off, out_mxfp8=out)
    codes_off = raw[1 : 1 + rows * H * D].view(torch.float8_e4m3fn).view(rows, H * D)
    assert not ok(out_mxfp8=(codes_off, out[1]))
    cs = case["inv_rope_cos_sin_cache"]
    cs_pitch = torch.empty((cs.shape[0], ROT + 2), device="cuda")[:, :ROT]
    assert not ok(out_mxfp8=out, inv_rope_cos_sin_cache=cs_pitch)
    cs_off = torch.empty(cs.numel() + 4, device="cuda")[1 : 1 + cs.numel()]
    assert not ok(out_mxfp8=out, inv_rope_cos_sin_cache=cs_off.view(cs.shape))
    kv_off = raw[1 : 1 + 32 * D].view(torch.float8_e4m3fn).view(32, D)
    assert not ok(kv_buffer=kv_off, out_mxfp8=out)
    # row count: output offsets stay 32-bit buffer offsets
    big = torch.empty((16385, H, D), dtype=torch.float8_e4m3fn, device="cuda")
    big_ptr = torch.zeros(16386, dtype=torch.int32, device="cuda")
    big_args = {
        "q": big,
        "kv_indptr": big_ptr,
        "extra_indptr": big_ptr,
        "out": torch.empty((16385, H, D), dtype=torch.bfloat16, device="cuda"),
        **no_rope,
    }
    assert not ok(**big_args)
    assert ok(**dict(big_args, q=big[:16384], out=big_args["out"][:16384]))
    with pytest.raises(ValueError):
        _op()(**dict(case, has_invalid=True), out_mxfp8=out)


@torch.inference_mode()
def test_zero_rows():
    case = make_case(row_lengths([1]))
    case["q"] = case["q"][:0]
    case["inv_rope_positions"] = case["inv_rope_positions"][:0]
    _op()(**case, out_mxfp8=mxfp8_outputs(0))


# rows without keys over empty index tensors: CUDA-graph padding rows, and the
# top-k lists of contexts shorter than the compression ratio. A fresh empty
# CUDA tensor has a null data pointer, so the kernel must not read it.
EMPTY_CASES = {
    "all_empty": [(0, 0, 10 + i) for i in range(40)],
    "no_topk": [((7 * i) % 129, 0, 200 + i) for i in range(40)],
}


@pytest.mark.parametrize(
    "hg,wgs", HEAD_GROUPS, ids=[f"hg{h}_wgs{w}" for h, w in HEAD_GROUPS]
)
@pytest.mark.parametrize("lists", list(EMPTY_CASES))
@torch.inference_mode()
def test_rows_without_keys(lists, hg, wgs):
    case = make_case(EMPTY_CASES[lists], pad=8)
    assert case["extra_indices"].numel() == 0
    if lists == "all_empty":
        assert case["kv_indices"].numel() == 0
    kw = {"heads_per_workgroup": hg, "workgroups_per_xcd": wgs}
    out = launch(case, mxfp8=False, **kw)
    torch.cuda.synchronize()
    check_attention(out, attention_reference(case), lists)
    codes, scales = launch(case, mxfp8=True, **kw)
    torch.cuda.synchronize()
    check_mxfp8(codes, scales, out, case, lists)


def test_warmup():
    """The warm-up compiles every head-group kernel of a configuration, so a
    later call (e.g. during CUDA graph capture) does not compile."""
    from aiter.ops.flydsl import warmup_flydsl_sparse_mla_decode_fp8_h64
    from aiter.ops.flydsl.sparse_mla_decode_fp8_h64 import HEAD_GROUPS as GROUPS
    from aiter.ops.flydsl.sparse_mla_decode_fp8_h64 import _launcher

    dev = torch.cuda.current_device()
    for p_terms, mxfp8, has_extra in ((1, True, True), (2, False, False)):
        warmup_flydsl_sparse_mla_decode_fp8_h64(
            dev, p_terms=p_terms, mxfp8=mxfp8, has_extra=has_extra
        )
        torch.cuda.synchronize()
        for hg in GROUPS:
            exe = _launcher(dev, hg, p_terms, mxfp8, has_extra)
            assert getattr(exe, "_cf", None) is not None, (hg, p_terms, mxfp8)


# ----------------------------------------------------------------------------
# perf sweep (python3 <this file>): served shapes, the MXFP8 call
# ----------------------------------------------------------------------------


def dequant_mxfp8(codes, scales):
    v = codes.view(torch.float8_e4m3fn).float().view(codes.shape[0], -1, 32)
    return (v * torch.exp2(scales.float() - 127.0)[..., None]).view(codes.shape[0], -1)


def run_sparse_mla_fwd(case, out_mxfp8):
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

    keys = ("q", "kv_buffer", "kv_indptr", "kv_indices", "softmax_scale")
    sparse_mla_fwd(
        *(case[k] for k in keys),
        **{k: v for k, v in case.items() if k not in keys and k != "max_keys_per_row"},
        has_invalid=False,
        out_mxfp8=out_mxfp8,
    )
    return out_mxfp8


@benchmark()
def test_sparse_mla_decode(rows, rows_per_request):
    """The MXFP8 call at ``rows`` query rows in requests of
    ``rows_per_request`` (a 128-key window + top-512 list per row): FlyDSL with
    two and one e4m3 P terms, and aiter's Triton ``sparse_mla_fwd`` (the path
    vLLM runs today). ``err``: dequantized MXFP8 against the inverse-RoPE'd
    fp64 reference."""
    q_lens = [rows_per_request] * (rows // rows_per_request)
    q_lens += [rows % rows_per_request] if rows % rows_per_request else []
    case = make_case(row_lengths(q_lens), pool_rows=1 << 17)
    ref = inverse_rope(
        attention_reference(case),
        case["inv_rope_positions"],
        case["inv_rope_cos_sin_cache"],
    ).view(rows, H * D)
    candidates = {
        "flydsl_p2": lambda o: _op()(**case, out_mxfp8=o, p_terms=2) or o,
        "flydsl_p1": lambda o: _op()(**case, out_mxfp8=o, p_terms=1) or o,
        "sparse_mla_fwd": lambda o: run_sparse_mla_fwd(case, o),
    }
    keys = int(case["kv_indptr"][-1]) + int(case["extra_indptr"][-1])
    flops = 4 * H * D * keys  # QK and PV, a multiply-add each per (key, head, dim)
    nbytes = (
        keys * (D + 4)  # gathered key rows and their ids
        + rows * H * D * 2  # q + MXFP8 codes
        + rows * H * D // 32  # E8M0 scales
    )
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        (codes, scales), us = run_perftest(fn, mxfp8_outputs(rows))
        err = checkAllclose(
            ref,
            dequant_mxfp8(codes, scales),
            rtol=0.1,
            atol=0.02,
            tol_err_ratio=1e-3,
            msg=f"{name}: rows={rows} ",
        )
        assert err <= 1e-3, (name, rows, err)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


test_sparse_mla_decode.__test__ = False


def main():
    if not _supported():
        aiter.logger.warning(
            "flydsl sparse MLA decode (fp8, 64 heads) unsupported on %s; skipping",
            get_gfx_runtime() if torch.cuda.is_available() else "no GPU",
        )
        return 0
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL sparse MLA decode (fp8 KV, 64 heads x 512) checks and perf",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[16, 64, 128, 256, 768],
        help="""Query rows (decode requests x verify rows).
    e.g.: -b 16 768""",
    )
    parser.add_argument(
        "-q",
        "--rows-per-request",
        type=int,
        nargs="*",
        default=[6],
        help="""Query rows per request (1 + speculative tokens).
    e.g.: -q 1 6""",
    )
    args = parser.parse_args()

    aiter.logger.info("sparse MLA decode: running the correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "sparse MLA decode: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1
    with torch.inference_mode():
        df = [
            test_sparse_mla_decode(rows, per_request)
            for per_request, rows in itertools.product(
                args.rows_per_request, args.batch
            )
        ]
    aiter.logger.info(
        "flydsl sparse MLA decode summary (markdown):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
