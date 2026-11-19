# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the FlyDSL 8-wave FP8 blockscale GEMM (gfx950).

Calls the kernel directly, without tuned-CSV dispatch, on shapes that hit
K=256, partial M and N tiles, and N below one tile, for both B layouts and
both output dtypes. C sits at the front of a sentinel-filled buffer, so the
test also catches unwritten elements and stores past the end of C.

Run:
    python op_tests/test_flydsl_gemm_a8w8_blockscale.py
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_blockscale import run_gemm_a8w8_blockscale
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
    kernelInstance,
)
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="FlyDSL blockscale GEMM requires gfx950",
)

# The kernel tile is 256x256x128; N stays a multiple of 16 for preshuffled B.
SHAPES = [
    (256, 512, 256),  # whole tiles, the shortest K loop
    (257, 576, 512),  # one-row M tail, 64-column N tail
    (1, 208, 1024),  # single row, N below one tile and one scale block
    (1000, 1344, 2048),  # M tail and N tail together
]

# Half an output ulp, relative.
RTOL = {torch.bfloat16: 2**-8, torch.float16: 2**-10}


def make_inputs(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
    x = (torch.randn(m, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    x_scale = torch.rand(m, k // 128, device="cuda", generator=g) * 0.02 + 0.001
    w_scale = torch.rand(-(-n // 128), k // 128, device="cuda", generator=g) * 0.02
    return x, w, x_scale, w_scale + 0.001


def reference(x, w, x_scale, w_scale):
    n = w.shape[0]
    a = x.float() * x_scale.repeat_interleave(128, 1)
    b = w.float() * w_scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return a @ b.T


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("preshuffle_b", [False, True])
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_flydsl_gemm_a8w8_blockscale(m, n, k, preshuffle_b, out_dtype):
    x, w, x_scale, w_scale = make_inputs(m, n, k)
    ref = reference(x, w, x_scale, w_scale)

    if preshuffle_b:
        wq = shuffle_weight(w, layout=(16, 16))
        # Preshuffled B takes x_scale as [K/128, M] memory under an (M, K/128) view.
        xs = x_scale.transpose(0, 1).contiguous().view(m, k // 128)
    else:
        wq, xs = w, x_scale

    guard = 256 * n
    buf = torch.full((m * n + guard,), float("nan"), dtype=out_dtype, device="cuda")
    out = buf[: m * n].view(m, n)
    name = kernelInstance(preshuffle_b).name
    run_gemm_a8w8_blockscale(x, wq, xs, w_scale, out, name, preshuffle_b)
    torch.cuda.synchronize()

    assert buf[m * n :].isnan().all(), "kernel stored past the end of C"
    torch.testing.assert_close(
        out.float(),
        ref,
        rtol=RTOL[out_dtype],
        atol=1e-3 * ref.abs().max().item(),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
