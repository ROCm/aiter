# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""flydsl_sparse_mla_decode_fp8_h64 against an fp64 torch reference.

Served shapes: 64 heads x 512 dims, e4m3 q and caches with per-tensor scales,
a sliding-window main list (<= 128 keys) and a top-k extra list (<= 512) per
query row, 1..6 rows per request, 16..768 rows, cache blocks of 32 / 64 / 128
rows, a padded page pitch, the window alone, padding rows with empty lists.

Every case checks the bf16 output against the fp64 reference (per (row, head)
relative L2 error and per-row cosine), the MXFP8 bytes against the reference
quantization (inverse RoPE + E8M0 per 32 dims) of that bf16 output, repeat
launches bit for bit, CUDA graph replay into NaN-filled outputs. Every cache row
no list names holds e4m3 NaN bytes, so a wrong gather shows. The saturation
tests drive scores of hundreds to 1e5 log2 units (q codes at +-448, large
scales, score jumps between tiles).

    pytest -q op_tests/flydsl_tests/test_flydsl_sparse_mla_decode_fp8_h64.py
"""

import math

import pytest
import torch

H, D, ROT = 64, 512, 64
FP8_MAX = 448.0
TINY_F32 = torch.finfo(torch.float32).tiny
SOFTMAX_SCALE = D**-0.5
NAN_BYTE = 0x7F
SAT = 0x7E  # e4m3 +-448
COS_MIN = 0.9999
REL = 0.02


def _on_gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="gfx950 required")


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
    err = rel_error(out[live], ref[live])
    assert err.max() <= rel, (name, err.max().item())
    cos = torch.nn.functional.cosine_similarity(
        out[live].double().flatten(1), ref[live].double().flatten(1), dim=1
    )
    assert cos.min() >= COS_MIN, (name, cos.min().item())


def check_mxfp8(codes, scales, out_bf16, case, name):
    rc, rs = epilogue_reference(
        out_bf16, case["inv_rope_positions"], case["inv_rope_cos_sin_cache"]
    )
    assert torch.equal(codes.view(torch.uint8), rc), name
    assert torch.equal(scales, rs), name


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
    assert can_use(**args, out_mxfp8=out)
    assert can_use(**args, out_mxfp8=out, max_keys_per_row=640)
    assert not can_use(**args, out_mxfp8=out, max_keys_per_row=2048)
    assert not can_use(**args, out_mxfp8=out, has_invalid=True)
    assert not can_use(**args)  # no output
    other_scale = torch.ones(1, device="cuda")
    assert not can_use(**args, out_mxfp8=out, extra_kv_scale=other_scale)
    assert can_use(**args, out_mxfp8=out, extra_kv_scale=case["kv_scale"])
    no_sink = dict(args, attn_sink=None)
    assert not can_use(**no_sink, out_mxfp8=out)
    no_rope = dict(args, inv_rope_positions=None, inv_rope_cos_sin_cache=None)
    assert not can_use(**no_rope, out_mxfp8=out)
    bf16 = torch.empty((rows, H, D), dtype=torch.bfloat16, device="cuda")
    assert can_use(**no_rope, out=bf16)
    assert not can_use(**args, out=bf16)  # bf16 rows carry no inverse RoPE
    q32 = case["q"][:, :32]
    assert not can_use(**dict(args, q=q32), out_mxfp8=out)
    bf16_q = case["q"].to(torch.bfloat16)
    assert not can_use(**dict(args, q=bf16_q), out_mxfp8=out)
    odd_block = case["kv_buffer"][:, :24]
    assert not can_use(**dict(args, kv_buffer=odd_block), out_mxfp8=out)
    i64 = case["kv_indices"].long()
    assert not can_use(**dict(args, kv_indices=i64), out_mxfp8=out)
    strided = torch.empty((rows, 2 * H * D), dtype=torch.float8_e4m3fn, device="cuda")
    assert not can_use(**args, out_mxfp8=(strided[:, : H * D], out[1]))
    with pytest.raises(ValueError):
        _op()(**case, out_mxfp8=out, has_invalid=True)


@torch.inference_mode()
def test_zero_rows():
    case = make_case(row_lengths([1]))
    case["q"] = case["q"][:0]
    case["inv_rope_positions"] = case["inv_rope_positions"][:0]
    _op()(**case, out_mxfp8=mxfp8_outputs(0))
