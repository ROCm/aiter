# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness tests for the fp32 split-K preshuffle GEMM.

Run with ``python3 -m pytest op_tests/flydsl_tests/test_flydsl_preshuffle_gemm_splitk.py -rs``.
Clear ``~/.flydsl/cache`` (or export ``FLYDSL_RUNTIME_ENABLE_CACHE=0``) first when
validating a kernel edit — the cache key does not see helper-method changes.
"""

from __future__ import annotations

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

pytest.importorskip("flydsl")

from aiter.ops.flydsl.kernels.preshuffle_gemm_splitk_op import (
    flydsl_preshuffle_gemm_splitk_a8,
)

ARCH = get_gfx_runtime()
# Tolerance carried over from the fp8 decode GEMM tests (test_flydsl_decode_gemm.py):
# the oracle sees the same quantized operands, so only f32 reassociation and the
# single bf16 output rounding remain.
FP8_RTOL = 1e-2
FP8_ATOL_SCALE = 5e-3

pytestmark = pytest.mark.skipif(
    ARCH not in ("gfx950", "gfx942"),
    reason="split-K preshuffle GEMM is validated on gfx950/gfx942 only",
)


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 7 + red * 3) % 23) - 11).to(torch.float32).div_(16).bfloat16()
    b = (((col * 5 + red * 7) % 29) - 14).to(torch.float32).div_(16).bfloat16()
    return a, b


def _run_split_k(m: int, n: int, k: int, split_k: int, tile: tuple[int, int, int]):
    tile_m, tile_n, tile_k = tile
    a, b = _inputs(m, n, k)
    aq, x_scale = aiter.per_tensor_quant(a, quant_dtype=dtypes.fp8)
    bq, w_scale = aiter.per_tensor_quant(b, quant_dtype=dtypes.fp8)
    wq = shuffle_weight(bq, layout=(16, 16))
    # The preshuffle epilogue takes per-row/per-col scales; per-tensor quant is the
    # broadcast of its single scalar.
    xsb = x_scale.reshape(1, 1).expand(m, 1).contiguous().float()
    wsb = w_scale.reshape(1, 1).expand(n, 1).contiguous().float()

    reference = (aq.float() * x_scale.float()) @ (bq.float() * w_scale.float()).T

    out = torch.full((m, n), torch.nan, device="cuda", dtype=torch.bfloat16)
    returned = flydsl_preshuffle_gemm_splitk_a8(
        aq, wq, xsb, wsb, out, tile_m, tile_n, tile_k, split_k, waves_per_eu=2
    )
    torch.cuda.synchronize()
    assert returned is out
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(),
        reference,
        rtol=FP8_RTOL,
        atol=FP8_ATOL_SCALE * reference.abs().max().item(),
    )


@pytest.mark.skipif(
    ARCH != "gfx950",
    reason="epilogue split-K validated on gfx950 only; gfx942 port covers blockscale",
)
def test_split_k_14_decode_shape() -> None:
    # K=7168 over 14 splits is one 512-element tile per workgroup.
    _run_split_k(4, 2048, 7168, split_k=14, tile=(16, 64, 512))


def _run_blockscale(
    m: int,
    n: int,
    k: int,
    split_k: int,
    tile: tuple[int, int, int],
    lds_stage: int = 2,
):
    """128-block dequant folded into the K loop, checked against an fp32 oracle.

    ``lds_stage=2`` (the compile default) double-buffers the K tiles; with a split_k
    that leaves more than one tile per workgroup it exercises the ping-pong main loop,
    otherwise only the peeled final tile. ``lds_stage=1`` is the single-buffer path.
    """
    tile_m, tile_n, tile_k = tile
    block = 128
    scale_k, scale_n = k // block, n // block
    gen = torch.Generator(device="cuda").manual_seed(0)

    a = ((torch.rand((m, k), generator=gen, device="cuda") - 0.5) / 4).to(dtypes.fp8)
    b = ((torch.rand((n, k), generator=gen, device="cuda") - 0.5) / 4).to(dtypes.fp8)
    x_scale = torch.rand((m, scale_k), generator=gen, device="cuda") + 0.5
    w_scale = torch.rand((scale_n, scale_k), generator=gen, device="cuda") + 0.5

    a_deq = a.float().view(m, scale_k, block) * x_scale.unsqueeze(-1)
    b_deq = b.float().view(n, scale_k, block) * w_scale.repeat_interleave(
        block, dim=0
    ).unsqueeze(-1)
    reference = a_deq.view(m, k) @ b_deq.view(n, k).T

    wq = shuffle_weight(b, layout=(16, 16))
    # The kernel reads scale_a as [K/128, M] (transposed) and scale_b as [N/128, K/128].
    xsb = x_scale.transpose(0, 1).contiguous()

    out = torch.full((m, n), torch.nan, device="cuda", dtype=torch.bfloat16)
    flydsl_preshuffle_gemm_splitk_a8(
        a,
        wq,
        xsb,
        w_scale.contiguous(),
        out,
        tile_m,
        tile_n,
        tile_k,
        split_k,
        lds_stage=lds_stage,
        scale_mode="blockscale",
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(),
        reference,
        rtol=FP8_RTOL,
        atol=FP8_ATOL_SCALE * reference.abs().max().item(),
    )


@pytest.mark.parametrize("m, n", [(1, 2048), (4, 8192)])
def test_blockscale_decode_shapes(m: int, n: int) -> None:
    # Production decode shape: split_k=14 leaves one 512-K tile per workgroup
    # (num_tiles=1), so lds_stage=2 (default) runs only the peeled final tile.
    _run_blockscale(m, n, 7168, split_k=14, tile=(16, 64, 512))


@pytest.mark.parametrize("split_k", [7, 2])
def test_blockscale_multitile_double_buffer(split_k: int) -> None:
    # split_k < 14 leaves num_tiles > 1 per workgroup, exercising the lds_stage=2
    # ping-pong that split_k=14 never reaches: split_k=7 -> 2 tiles (tail + final
    # peel, one prefetch overlapping one MMA); split_k=2 -> 7 tiles (the two_tiles
    # main loop). The global scale-block index must stay correct across tiles.
    _run_blockscale(1, 2048, 7168, split_k=split_k, tile=(16, 64, 512))


def test_blockscale_single_buffer_lds1() -> None:
    # Single-buffer path: lds_stage=1 must stay correct alongside the
    # lds_stage=2 default.
    _run_blockscale(1, 2048, 7168, split_k=14, tile=(16, 64, 512), lds_stage=1)
