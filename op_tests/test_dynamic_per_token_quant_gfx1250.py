# SPDX-License-Identifier: MIT
"""gfx1250 large-row per-token quant dispatch parity."""

import pytest
import torch

from aiter.ops.quant import dynamic_per_token_scaled_quant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("cols", [1024, 2048, 3072, 4096, 6144, 8192])
@pytest.mark.parametrize("rows", [1024, 2048, 4536])
def test_large_row_dispatch_matches_small_generic(rows, cols):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the large-row dispatch is gfx1250-only")

    torch.manual_seed(20261104)
    base = torch.randn((4, cols), dtype=torch.bfloat16, device="cuda")
    base[0].zero_()
    base[2, :4] = torch.tensor(
        [float("inf"), -float("inf"), 448.0, -448.0],
        dtype=torch.bfloat16,
        device="cuda",
    )
    base[3, 0] = float("nan")

    reference = torch.empty_like(base, dtype=torch.float8_e4m3fn)
    reference_scale = torch.empty(4, dtype=torch.float32, device="cuda")
    dynamic_per_token_scaled_quant(reference, base, reference_scale)
    assert int(reference[3, 0].view(torch.uint8)) == 0xFE

    source = base.repeat(rows // base.shape[0], 1)
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    dynamic_per_token_scaled_quant(output, source, scale)
    torch.cuda.synchronize()

    assert torch.equal(
        output.view(torch.uint8),
        reference.repeat(rows // base.shape[0], 1).view(torch.uint8),
    )
    assert torch.equal(
        scale.view(torch.uint8),
        reference_scale.repeat(rows // base.shape[0]).view(torch.uint8),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (1, 1024),
        (2, 2048),
        (3, 4096),
        (5, 6144),
        (7, 8192),
        (1025, 1024),
        (4097, 2048),
        (4537, 4096),
        (5611, 6144),
        (8193, 8192),
        (8192, 4096),
        (8193, 4096),
        (16384, 1024),
        (16385, 1024),
    ],
)
def test_row_tails_and_special_values_match_generic(rows, cols):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the large-row dispatch is gfx1250-only")

    torch.manual_seed(20261104)
    base = torch.randn((8, cols), dtype=torch.bfloat16, device="cuda")
    base[0].zero_()
    base[0, 0] = -0.0
    base[0, 2::2] = -0.0
    base[2, :6] = torch.tensor(
        [
            float("inf"),
            -float("inf"),
            448.0,
            -448.0,
            0.0,
            -0.0,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    base[3, 0] = float("nan")
    base[4, 0] = -float("nan")

    reference = torch.empty_like(base, dtype=torch.float8_e4m3fn)
    reference_scale = torch.empty(8, dtype=torch.float32, device="cuda")
    dynamic_per_token_scaled_quant(reference, base, reference_scale)

    indices = torch.arange(rows, device="cuda") % base.shape[0]
    source = base.index_select(0, indices)
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, dtype=torch.float32, device="cuda")
    dynamic_per_token_scaled_quant(output, source, scale)
    torch.cuda.synchronize()

    assert torch.equal(
        output.view(torch.uint8),
        reference.index_select(0, indices).view(torch.uint8),
    )
    assert torch.equal(
        scale.view(torch.uint8),
        reference_scale.index_select(0, indices).view(torch.uint8),
    )
