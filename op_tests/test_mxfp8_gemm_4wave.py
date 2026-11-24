# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Persistent 4-wave MXFP8 GEMM (aiter.ops.flydsl.kernels.mxfp8_gemm_4wave), gfx950.

The 4-wave kernel takes row-major fp8 A, ``shuffle_weight(.., (16, 16))`` W and both e8m0
scales in ``shuffle_scale_a16w4(.., 1, False)`` order (rows padded to 32); the 8-wave kernel
takes the same operands with raw scales. Same MFMA, same per-accumulator K order: the
outputs must be bit-identical, for every tile variant and for M that is not a multiple of
the tile.
"""

import pytest
import torch

from aiter import dtypes
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950" or not torch.cuda.is_available(), reason="gfx950 only"
)


def _operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    xq, xs = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    wq, ws = per_group_quant_hip(
        w, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    return x, w, xq, xs, shuffle_weight(wq, layout=(16, 16)), ws, wq


def _pad32(s):
    pad = -(-s.shape[0] // 32) * 32
    if pad == s.shape[0]:
        return s
    out = torch.full((pad, s.shape[1]), 127, dtype=torch.uint8, device=s.device)
    out[: s.shape[0]].copy_(s.view(torch.uint8))
    return out.view(s.dtype)


def _run_4w(xq, xs, wsh, ws, m, n, k, tiles, grid_max=256):
    from aiter.ops.flydsl.kernels.mxfp8_gemm_4wave import compile_mxfp8_gemm_4w
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, current_stream_arg

    exe = compile_mxfp8_gemm_4w(K=k, tiles=tiles, grid_max=grid_max)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    parents = -(-m // 256) * -(-n // 256)
    if exe.sub_split > 1:  # a sub-tile launch deals its parents out by XCD, 8 at a time
        parents = -(-parents // 8) * 8
    xs_s = shuffle_scale_a16w4(_pad32(xs), 1, False).view(torch.uint8).view(-1)
    ws_s = shuffle_scale_a16w4(_pad32(ws), 1, False).view(torch.uint8).view(-1)
    _run_compiled(
        exe,
        xq.view(torch.int8).view(-1),
        wsh.view(torch.int8).view(-1),
        out.view(-1),
        xs_s,
        ws_s,
        m,
        n,
        0,
        parents * exe.sub_split,
        current_stream_arg(xq.device),
        specialization_key=xq.device,
    )
    return out


def _run_8w(xq, xs, wsh, ws, m, n, k):
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import flydsl_mxfp8_8wave_gemm

    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    return flydsl_mxfp8_8wave_gemm(xq, wsh, xs, ws, out)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (256, 256, 1024),
        (300, 512, 1024),  # M not a multiple of the tile
        (2048, 6144, 8192),  # MiniMax-M3.1 o_proj_full, SP prefill rows
        (3072, 6144, 8192),
        (4100, 6144, 8192),  # partial last row of tiles
        (5000, 2560, 6144),
    ],
)
@pytest.mark.parametrize("tiles", [(4, 4), (2, 4), (2, 2)])
def test_4wave_bit_identical_to_8wave(m, n, k, tiles):
    _, _, xq, xs, wsh, ws, _ = _operands(m, n, k)
    ref = _run_8w(xq, xs, wsh, ws, m, n, k)
    got = _run_4w(xq, xs, wsh, ws, m, n, k, tiles)
    assert torch.equal(
        got, ref
    ), f"max abs diff {(got.float() - ref.float()).abs().max().item()}"


def test_4wave_plain_grid_matches_persistent():
    m, n, k = 2304, 2560, 4096
    _, _, xq, xs, wsh, ws, _ = _operands(m, n, k, seed=1)
    a = _run_4w(xq, xs, wsh, ws, m, n, k, (4, 4), grid_max=256)
    b = _run_4w(xq, xs, wsh, ws, m, n, k, (4, 4), grid_max=1 << 20)
    assert torch.equal(a, b)


def test_4wave_vs_fp32_reference():
    m, n, k = 384, 512, 2048
    _, _, xq, xs, wsh, ws, wq = _operands(m, n, k, seed=2)

    def deq(q, s):
        sc = torch.exp2(s.view(torch.uint8).float() - 127).repeat_interleave(32, dim=1)
        return q.float() * sc

    ref = deq(xq, xs) @ deq(wq, ws).t()
    got = _run_4w(xq, xs, wsh, ws, m, n, k, (4, 4)).float()
    rel = (got - ref).norm() / ref.norm()
    assert rel < 5e-3, rel.item()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x"]))
