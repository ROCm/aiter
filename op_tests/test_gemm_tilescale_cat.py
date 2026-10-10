# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two-segment A6W6 tilescale GEMM with the gated-residual epilogue (``gemm_a6w6_tilescale_cat``).

* tolerance: h and out against an fp32 reference (dequantised operands, fp32 matmul, fp32 epilogue);
* exact: with bias = 0, x = 0 and a gate of ones the kernel returns a = bf16(acc) in both h and out; a second call
  with random bias / x / a strided gate must then equal the documented epilogue applied to that a, bit for bit
  (h = bf16(a + bias), out = bf16(fma(g, a + bias, x)), emulated in fp64, where g * h is exact);
* a B whose C1 plane sits in its own buffer gives the same bits as the joined buffer;
* the sum of the two plain tilescale GEMMs (each rounded to bf16) agrees with a within tolerance.
"""

import pytest
import torch

import aiter.ops.tilescale as TS
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_tilescale import a6w6_tilescale_cat_table, gemm_a6w6_tilescale, gemm_a6w6_tilescale_cat


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="tilescale GEMMs require gfx950")
SHAPES = sorted(a6w6_tilescale_cat_table()) if _is_gfx950() else []


def _segment(M, N, K, g):
    a6 = torch.randint(0, 64, (M, K), dtype=torch.uint8, device="cuda", generator=g)
    b6 = torch.randint(0, 64, (N, K), dtype=torch.uint8, device="cuda", generator=g)
    sa = torch.randint(122, 130, (M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    sb = torch.randint(122, 130, (N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    packs = (TS.pack_fp6_codes_ref(a6), TS.pack_fp6_codes_ref(b6), TS.pack_scales_ref(sa, is_b=False),
             TS.pack_scales_ref(sb, is_b=True))
    ref = TS.dequant_ref(a6, sa, TS.FP6).float() @ TS.dequant_ref(b6, sb, TS.FP6).float().t()
    return packs, ref


def _split_b(b, N, K):
    return b[: N * K // 2].clone(), b[N * K // 2:].clone()


@requires_gfx950
def test_table():
    assert SHAPES, "no two-segment tilescale kernels in the manifest"
    for M, N, K, K2, Bt in SHAPES:
        assert M % 256 == 0 and N % 256 == 0 and K % 256 == 0 and K2 % 256 == 0 and M % Bt == 0


@requires_gfx950
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_cat(shape):
    M, N, K, K2, Bt = shape
    g = torch.Generator(device="cuda").manual_seed(M + N + K + K2)
    (A, B, SA, SB), ref0 = _segment(M, N, K, g)
    (A2, B2, SA2, SB2), ref1 = _segment(M, N, K2, g)
    acc = ref0 + ref1
    del ref0, ref1
    bf = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda", generator=g)  # noqa: E731
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    h = torch.empty_like(out)

    # a = bf16(acc): identity epilogue
    zx = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    zb = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    ones = torch.ones(Bt, N, dtype=torch.bfloat16, device="cuda")
    gemm_a6w6_tilescale_cat(A, B, SA, SB, A2, B2, SA2, SB2, out, h, K, K2, zb, zx, ones)
    a = out.clone()
    assert torch.equal(h, a)
    scale = acc.abs().max().item()
    assert (a.float() - acc).abs().max().item() <= 2 ** -7 * scale

    # the two plain GEMMs (each rounded to bf16) agree with a within tolerance
    p0, p1 = torch.empty_like(out), torch.empty_like(out)
    gemm_a6w6_tilescale(A, B, SA, SB, p0, K)
    gemm_a6w6_tilescale(A2, B2, SA2, SB2, p1, K2)
    assert (p0.float() + p1.float() - a.float()).abs().max().item() <= 2 ** -6 * scale

    # random epilogue, gate a chunk of a wider modulation table (row stride > N), B's C1 planes split off
    x, bias = bf(M, N), bf(N)
    table = bf(Bt, 6 * N) * 0.1
    gate = table[:, 4 * N:5 * N]
    b0, b1 = _split_b(B, N, K)
    b20, b21 = _split_b(B2, N, K2)
    gemm_a6w6_tilescale_cat(A, b0, SA, SB, A2, b20, SA2, SB2, out, h, K, K2, bias, x, gate, b_c1=b1, b2_c1=b21)
    gm = gate[torch.arange(M, device="cuda") % Bt].float()
    h32 = a.float() + bias.float()
    assert torch.equal(h, h32.to(torch.bfloat16))
    want = (x.double() + gm.double() * h32.double()).float().to(torch.bfloat16)
    assert torch.equal(out, want)

    # fp32 reference with tolerance
    hr = acc + bias.float()
    orf = x.float() + gm * hr
    assert (h.float() - hr).abs().max().item() <= 2 ** -7 * hr.abs().max().item()
    assert (out.float() - orf).abs().max().item() <= 2 ** -7 * orf.abs().max().item()

    # joined B buffers give the same bits
    out2, h2 = torch.empty_like(out), torch.empty_like(out)
    gemm_a6w6_tilescale_cat(A, B, SA, SB, A2, B2, SA2, SB2, out2, h2, K, K2, bias, x, gate)
    assert torch.equal(out2, out) and torch.equal(h2, h)


@requires_gfx950
def test_no_kernel():
    t = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
    u8 = torch.empty(256 * 256, dtype=torch.uint8, device="cuda")
    with pytest.raises(Exception):
        gemm_a6w6_tilescale_cat(u8[: 256 * 192], u8[: 256 * 192], u8[:2048], u8[:2048], u8[: 256 * 192],
                                u8[: 256 * 192], u8[:2048], u8[:2048], t, torch.empty_like(t), 256, 256,
                                t[0], t, t[:32])


@requires_gfx950
@pytest.mark.parametrize("alias", ["out_is_h", "out_is_x", "h_is_x"])
def test_aliasing_rejected(alias):
    t = torch.empty(256, 256, dtype=torch.bfloat16, device="cuda")
    u8 = torch.empty(256 * 256, dtype=torch.uint8, device="cuda")
    out, h, x = t, torch.empty_like(t), torch.empty_like(t)
    if alias == "out_is_h":
        h = out
    elif alias == "out_is_x":
        x = out
    else:
        x = h
    with pytest.raises(Exception, match="must not overlap"):
        gemm_a6w6_tilescale_cat(u8[: 256 * 192], u8[: 256 * 192], u8[:2048], u8[:2048], u8[: 256 * 192],
                                u8[: 256 * 192], u8[:2048], u8[:2048], out, h, 256, 256, t[0], x, t[:32])
