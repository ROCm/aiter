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
        return [{k: (v if k in ("knl_name", "co_name") else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]


ROWS = _manifest()
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
    return TS.pack_fp4_codes_ref(codes, "k128" if b_codes else "row")


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
                      TS.pack_scales_ref(sb, is_b=True, ilv=row["b_ilv"]), out, af, bf, K, bv, row["b_codes"],
                      row["b_ilv"])
    fa, fb = (TS.FP4 if af == 4 else TS.FP6), (TS.FP4 if bf == 4 else TS.FP6)
    ref = TS.dequant_ref(a, sa, fa) @ TS.dequant_ref(b, sb, fb).t()
    if bv is not None:
        ref = ref + bv.double()
    err = (out.double() - ref).abs()
    assert float(err.max()) <= float(ref.abs().max()) * 2**-7, float(err.max())
