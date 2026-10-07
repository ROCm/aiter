# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tilescale GEMMs (aiter.ops.gemm_op_tilescale) against an fp64 reference, and A6W4 against A6W6 on the FP6
re-encoding of B (every E2M1 value is an E2M3 value and the MFMA treats them alike, so the two are bitwise equal)."""

import csv
import os

import pytest
import torch

import aiter.ops.tilescale as TS
from aiter.jit.core import AITER_META_DIR
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_tilescale import gemm_a6w4_tilescale, gemm_mx_tilescale, tilescale_supported


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="tilescale GEMMs require gfx950")


def _manifest():
    path = os.path.join(AITER_META_DIR, "hsa", "gfx950", "tsgemm", "tsgemm_bf16_per1x32.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: (v if k in ("knl_name", "co_name") else int(v or 0)) for k, v in r.items()}
                for r in csv.DictReader(f)]


ROWS = [r for r in _manifest() if r["abi"] not in (2, 3, 4)]  # exact-shape rows
SOFTMAX_D = [r for r in _manifest() if r["abi"] == 4]  # exact-shape rows emitting attention's softmax_d
GENERIC_ROWS = [r for r in _manifest() if r["abi"] in (2, 3)]
A6W4 = [r for r in ROWS if (r["a_fmt"], r["b_fmt"]) == (6, 4)]


def _operands(M, N, K, seed, bias):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a6 = torch.randint(0, 64, (M, K), dtype=torch.uint8, device="cuda", generator=g)
    b4 = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda", generator=g)
    sa = torch.randint(118, 133, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    sb = torch.randint(118, 133, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    bv = torch.randn(N, dtype=torch.bfloat16, device="cuda", generator=g) if bias else None
    return a6, b4, sa, sb, bv


def _small(r):
    return r["M"] * r["N"] * r["K"] <= 1 << 31


@requires_gfx950
def test_manifest_rows_current():
    assert A6W4, "no A6W4 tilescale rows"
    for r in ROWS:
        assert r["ts_ver"] == TS.TILESCALE_VERSION
        assert tilescale_supported(r["M"], r["N"], r["K"], r["a_fmt"], r["b_fmt"], bool(r["bias"]), r["b_codes"],
                                   r["b_ilv"])
    assert not tilescale_supported(256, 256, 256, 6, 4)


@requires_gfx950
@pytest.mark.parametrize("row", [r for r in A6W4 if _small(r)], ids=lambda r: r["knl_name"])
def test_a6w4_vs_fp64(row):
    M, N, K, bias = row["M"], row["N"], row["K"], bool(row["bias"])
    a6, b4, sa, sb, bv = _operands(M, N, K, M + N + K, bias)
    A = TS.pack_fp6_codes_ref(a6)
    B = TS.pack_fp4_codes_ref(b4, "k128")
    SA = TS.pack_scales_ref(sa, is_b=False)
    SB = TS.pack_scales_ref(sb, is_b=True)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    gemm_a6w4_tilescale(A, B, SA, SB, out, K, bv)
    ref = TS.dequant_ref(a6, sa, TS.FP6) @ TS.dequant_ref(b4, sb, TS.FP4).t()
    if bv is not None:
        ref = ref + bv.double()
    err = (out.double() - ref).abs()
    assert float(err.max()) <= float(ref.abs().max()) * 2**-7, float(err.max())


@requires_gfx950
@pytest.mark.parametrize("row", A6W4, ids=lambda r: r["knl_name"])
def test_a6w4_bitwise_vs_a6w6(row):
    M, N, K, bias = row["M"], row["N"], row["K"], bool(row["bias"])
    a6w6 = [r for r in ROWS if (r["a_fmt"], r["b_fmt"], r["bias"], r["M"], r["N"], r["K"]) == (6, 6, row["bias"], M, N, K)]
    try:
        from aiter.ops.gemm_op_a6w6_fly import gemm_a6w6_fly_asm
    except ImportError:
        gemm_a6w6_fly_asm = None
    a6, b4, sa, sb, bv = _operands(M, N, K, 7 + M + K, bias)
    A = TS.pack_fp6_codes_ref(a6)
    SA = TS.pack_scales_ref(sa, is_b=False)
    SB = TS.pack_scales_ref(sb, is_b=True)
    B6 = TS.pack_fp6_codes_ref(((b4 & 8) << 2) | ((b4 & 6) << 2) | ((b4 & 1) << 2))
    ref = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    if a6w6:
        gemm_mx_tilescale(A, B6, SA, SB, ref, 6, 6, K, bv)
    elif gemm_a6w6_fly_asm is not None:
        try:
            gemm_a6w6_fly_asm(A, B6, SA, SB, ref, K, bv)
        except Exception as e:  # noqa: BLE001 -- no A6W6 object for this shape
            pytest.skip(f"no A6W6 tilescale kernel for {M}x{N}x{K}: {e}")
    else:
        pytest.skip("no A6W6 tilescale kernels")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    gemm_a6w4_tilescale(A, TS.pack_fp4_codes_ref(b4, "k128"), SA, SB, out, K, bv)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@requires_gfx950
def test_missing_shape_raises():
    A = torch.zeros(256 * 1024 * 3 // 4, dtype=torch.uint8, device="cuda")
    B = torch.zeros(256 * 1024 // 2, dtype=torch.uint8, device="cuda")
    S = torch.zeros(256 * 1024 // 32, dtype=torch.uint8, device="cuda")
    out = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception):
        gemm_a6w4_tilescale(A, B, S, S, out, 1024)


def _codes(fmt, rows, K, g):
    return torch.randint(0, 16 if fmt == 4 else 64, (rows, K), dtype=torch.uint8, device="cuda", generator=g)


def _pack(fmt, codes, b_codes=0):
    if fmt == 6:
        return TS.pack_fp6_codes_ref(codes)
    return TS.pack_fp4_codes_ref(codes, ("row", "k128", "kouter")[b_codes])


@requires_gfx950
@pytest.mark.parametrize("row", [r for r in ROWS if _small(r)], ids=lambda r: r["knl_name"])
def test_row_vs_fp64(row):
    """Every small manifest row, whatever its format pair, against the fp64 product of its dequantized operands."""
    M, N, K, bias = row["M"], row["N"], row["K"], bool(row["bias"])
    af, bf = row["a_fmt"], row["b_fmt"]
    g = torch.Generator(device="cuda").manual_seed(M * 7 + N * 3 + K + af * 11 + bf)
    a, b = _codes(af, M, K, g), _codes(bf, N, K, g)
    sa = torch.randint(118, 133, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    sb = torch.randint(118, 133, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    bv = torch.randn(N, dtype=torch.bfloat16, device="cuda", generator=g) if bias else None
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    gemm_mx_tilescale(_pack(af, a), _pack(bf, b, row["b_codes"]), TS.pack_scales_ref(sa, is_b=False),
                      TS.pack_scales_ref(sb, is_b=True, ilv=row["b_ilv"], kouter=row["b_codes"] == 2), out, af, bf, K,
                      bv, row["b_codes"],
                      row["b_ilv"])
    fa, fb = (TS.FP4 if af == 4 else TS.FP6), (TS.FP4 if bf == 4 else TS.FP6)
    ref = TS.dequant_ref(a, sa, fa) @ TS.dequant_ref(b, sb, fb).t()
    if bv is not None:
        ref = ref + bv.double()
    err = (out.double() - ref).abs()
    assert float(err.max()) <= float(ref.abs().max()) * 2**-7, float(err.max())


GENERIC = [r for r in GENERIC_ROWS if r["abi"] == 2]


def _generic_shapes(row):
    k0 = row["kmin"]
    return [(768, 1280, k0), (512, 768, k0 + 1536)]


@requires_gfx950
@pytest.mark.parametrize("row", GENERIC, ids=lambda r: r["knl_name"])
def test_generic_row_vs_fp64(row):
    """A shape-generic row on shapes no exact-shape row covers, two K values of its class."""
    for M, N, K in _generic_shapes(row):
        assert (K // 128) % 12 == row["kcls"]
        assert tilescale_supported(M, N, K, row["a_fmt"], row["b_fmt"], bool(row["bias"]), row["b_codes"])
        r = dict(row, M=M, N=N, K=K)
        test_row_vs_fp64(r)


@requires_gfx950
@pytest.mark.parametrize("cls", sorted({r["kcls"] for r in GENERIC}))
@pytest.mark.parametrize("bias", [0, 1])
def test_generic_a6w4_bitwise_vs_a6w6(cls, bias):
    row = next(r for r in GENERIC if r["kcls"] == cls and r["bias"] == bias and r["b_fmt"] == 4)
    for M, N, K in _generic_shapes(row):
        a6, b4, sa, sb, bv = _operands(M, N, K, 5 + M + K, bool(bias))
        A, SA, SB = TS.pack_fp6_codes_ref(a6), TS.pack_scales_ref(sa, is_b=False), TS.pack_scales_ref(sb, is_b=True)
        B6 = TS.pack_fp6_codes_ref(((b4 & 8) << 2) | ((b4 & 6) << 2) | ((b4 & 1) << 2))
        ref = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        gemm_mx_tilescale(A, B6, SA, SB, ref, 6, 6, K, bv)
        out = torch.empty_like(ref)
        gemm_a6w4_tilescale(A, TS.pack_fp4_codes_ref(b4, "k128"), SA, SB, out, K, bv)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))



@requires_gfx950
@pytest.mark.parametrize("shape", [(512, 512, 1024), (768, 1280, 3072), (1024, 512, 2048), (512, 768, 4608)])
def test_a4w4_k_generic_vs_fp64(shape):
    """The K-generic A4W4 row (abi 3) on shapes no exact-shape row covers."""
    M, N, K = shape
    row = next(r for r in GENERIC_ROWS if (r["a_fmt"], r["b_fmt"]) == (4, 4))
    assert tilescale_supported(M, N, K, 4, 4, False, 0, row["b_ilv"])
    test_row_vs_fp64(dict(row, M=M, N=N, K=K))


@requires_gfx950
@pytest.mark.parametrize("bias", [False, True])
def test_a6w6_split_c1_plane(bias):
    """B's C1 plane in its own buffer gives the same bits as the one-buffer operand."""
    M, N, K = 16384, 9216, 3072
    if not tilescale_supported(M, N, K, 6, 6, bias):
        pytest.skip("no A6W6 kernel")
    g = torch.Generator(device="cuda").manual_seed(11)
    a6 = torch.randint(0, 64, (M, K), dtype=torch.uint8, device="cuda", generator=g)
    b6 = torch.randint(0, 64, (N, K), dtype=torch.uint8, device="cuda", generator=g)
    sa = torch.randint(118, 133, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    sb = torch.randint(118, 133, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    bv = torch.randn(N, dtype=torch.bfloat16, device="cuda", generator=g) if bias else None
    A, B = TS.pack_fp6_codes_ref(a6), TS.pack_fp6_codes_ref(b6)
    SA, SB = TS.pack_scales_ref(sa, is_b=False), TS.pack_scales_ref(sb, is_b=True)
    ref = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    gemm_mx_tilescale(A, B, SA, SB, ref, 6, 6, K, bv)
    c0, c1 = B[: N * K // 2].clone(), B[N * K // 2 :].clone()
    out = torch.empty_like(ref)
    gemm_mx_tilescale(A, c0, SA, SB, out, 6, 6, K, bv, b_c1=c1)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


KOUTER = [r for r in ROWS if r["b_codes"] == 2]


@requires_gfx950
@pytest.mark.parametrize("row", KOUTER, ids=lambda r: r["knl_name"])
def test_a4w4_kouter_bitwise_vs_row(row):
    """B in the K256-outer form (codes and scale slab) gives the same bits as the row-major form of the same B."""
    M, N, K, ilv = row["M"], row["N"], row["K"], row["b_ilv"]
    g = torch.Generator(device="cuda").manual_seed(M + N + K)
    a, b = _codes(4, M, K, g), _codes(4, N, K, g)
    sa = torch.randint(118, 133, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    sb = torch.randint(118, 133, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    A, SA = _pack(4, a), TS.pack_scales_ref(sa, is_b=False)
    outs = []
    for bc in (0, 2):
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        gemm_mx_tilescale(A, _pack(4, b, bc), SA, TS.pack_scales_ref(sb, is_b=True, ilv=ilv, kouter=bc == 2), out,
                          4, 4, K, None, bc, ilv)
        outs.append(out)
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


@requires_gfx950
@pytest.mark.parametrize("row", SOFTMAX_D, ids=lambda r: r["knl_name"])
def test_a4w4_softmax_d(row):
    """The softmax_d epilogue: out bitwise equal to the plain row's; softmax_d [B, N/128, S] = per row and head the
    fp32 sum of out * O (sbhd rows), within fp32 summation error of the fp64 sum. GEMMs covering one sequence in
    parts (a joint block's text and image rows) add into one softmax_d at their first sequence positions."""
    M, N, K, bc, ilv, B, S = (row[k] for k in ("M", "N", "K", "b_codes", "b_ilv", "epi_b", "epi_s"))
    H, parts = N // 128, S * B // M
    g = torch.Generator(device="cuda").manual_seed(M + N + K)
    o = torch.randn(S * B, N, dtype=torch.bfloat16, device="cuda", generator=g)
    d = torch.zeros(B, H, S, dtype=torch.float32, device="cuda")
    outs = []
    for i in range(parts):
        a, b = _codes(4, M, K, g), _codes(4, N, K, g)
        sa = torch.randint(118, 133, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
        sb = torch.randint(118, 133, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
        A, SA = _pack(4, a), TS.pack_scales_ref(sa, is_b=False)
        Bp, SB = _pack(4, b, bc), TS.pack_scales_ref(sb, is_b=True, ilv=ilv, kouter=bc == 2)
        plain = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        gemm_mx_tilescale(A, Bp, SA, SB, plain, 4, 4, K, None, bc, ilv)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        gemm_mx_tilescale(A, Bp, SA, SB, out, 4, 4, K, None, bc, ilv, epi_o=o[i * M : (i + 1) * M], epi_delta=d,
                          epi_s0=i * M // B)
        assert torch.equal(out.view(torch.int16), plain.view(torch.int16))
        outs.append(out)
    prod = (torch.cat(outs).double() * o.double()).view(S, B, H, 128)
    ref = prod.sum(-1).permute(1, 2, 0)
    mag = prod.abs().sum(-1).permute(1, 2, 0)
    assert ((d.double() - ref).abs() / mag).max().item() < 1e-5
