# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

requires_gfx950 = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 FlyDSL kernel"
)


def _run_4wave(x, weight_shuffled, x_scale, w_scale, out, num_cu):
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    m, k = x.shape
    n = weight_shuffled.shape[0]
    exe = compile_int8_gemm_4w(K=k, xcd_swizzle=8, num_cu=num_cu)
    _run_compiled(
        exe,
        x.view(-1),
        weight_shuffled.view(-1),
        out.view(-1),
        x_scale.view(-1),
        w_scale.view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )


@requires_gfx950
@pytest.mark.parametrize(
    "m,n,k",
    [
        # 2..9 K tiles: no loop trip, the loop with both tail parities.
        (256, 256, 256),
        (256, 256, 384),
        (256, 256, 512),
        (256, 256, 640),
        (256, 256, 1024),
        (256, 256, 1152),
        # M/N tails of the 256x256 workgroup tile.
        (1, 16, 256),
        (131, 272, 384),
        (513, 784, 1152),
        (1000, 1008, 5120),
        # More blocks than CUs.
        (4352, 4608, 1024),
    ],
)
# 0: one block per workgroup; 8: persistent workgroups take several blocks
# each on the small shapes; None: persistent on all CUs.
@pytest.mark.parametrize("num_cu", [0, 8, None])
@torch.no_grad()
def test_flydsl_int8_4wave_bit_exact(m, n, k, num_cu):
    if num_cu is None:
        num_cu = get_cu_num()
    generator = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16)).contiguous()
    actual = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")

    _run_4wave(x, weight_shuffled, x_scale, w_scale, actual, num_cu)

    # Small values keep every FP32 product and sum exact, so the kernel's
    # single post-reduction scale product must match bit for bit.
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16)
    assert torch.equal(actual, expected)
