# SPDX-License-Identifier: MIT
"""Legacy fused RoPE query calls preserve their explicit FP4 layouts."""

import pytest
import torch

from aiter import dtypes, rope_rotate_activation
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="gfx950 required"
)


@pytest.mark.parametrize(
    "style,shuffle",
    [
        ("default", True),
        ("keyword", True),
        ("keyword", False),
        ("positional", True),
        ("positional", False),
    ],
)
def test_legacy_query_layout_matches_explicit(style, shuffle):
    torch.manual_seed(322)
    query = torch.randn(7, 64, 128, device="cuda").bfloat16()
    angles = torch.randn(16, 32, device="cpu")
    cos, sin = angles.cos().bfloat16().cuda(), angles.sin().bfloat16().cuda()
    positions = torch.arange(7, device="cuda")
    values = torch.empty(7, 64, 64, device="cuda", dtype=torch.uint8)
    shape = (7, 1, 4, 16, 4) if shuffle else (7, 64, 4)
    scales = torch.empty(shape, device="cuda", dtype=torch.uint8)
    ref_values, ref_scales = torch.empty_like(values), torch.empty_like(scales)
    common = (query, cos, sin, positions, 64)
    rope_rotate_activation(
        ref_values.view(dtypes.fp4x2),
        *common,
        out_scale=ref_scales,
        scale_layout="flydsl" if shuffle else "none",
        do_rotate_act=False,
        round_rope=True,
    )
    if style == "positional":
        rope_rotate_activation(
            values.view(dtypes.fp4x2), *common, scales, 32, shuffle, False, True
        )
    else:
        kwargs = {"shuffle_scale": shuffle} if style == "keyword" else {}
        rope_rotate_activation(
            values.view(dtypes.fp4x2),
            *common,
            out_scale=scales,
            do_rotate_act=False,
            round_rope=True,
            **kwargs,
        )
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)


if __name__ == "__main__":
    # The standard op-test runner invokes each file as a script.
    raise SystemExit(pytest.main([__file__, "-q"]))
