# SPDX-License-Identifier: MIT
"""The fused RoPE writer's OPUS scale layout for V4 and V4.1."""

import pytest
import torch

from aiter import dtypes, rope_rotate_activation
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="gfx950 required"
)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("round_rope", [False, True])
def test_opus_query_scales_match_natural_and_replay(heads, rotate, round_rope):
    torch.manual_seed(321)
    rows = 7
    query = torch.randn(rows, heads, 128, device="cuda").bfloat16()
    angles = torch.randn(32, 32, device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    positions = torch.tensor([3, 1, 7, 11, 2, 9, 0], device="cuda")
    values = torch.empty(rows, heads, 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(rows, 2, 32, heads // 16, device="cuda", dtype=torch.uint8)
    ref_values = torch.empty_like(values)
    natural = torch.empty(rows, heads, 4, device="cuda", dtype=torch.uint8)

    def run(out, scale, layout):
        rope_rotate_activation(
            out.view(dtypes.fp4x2),
            query,
            cos,
            sin,
            positions,
            64,
            out_scale=scale,
            group_size=32,
            scale_layout=layout,
            do_rotate_act=rotate,
            round_rope=round_rope,
        )

    def check():
        run(ref_values, natural, "none")
        expected = natural.reshape(rows, heads // 32, 32, 2, 2).permute(0, 4, 2, 3, 1)
        torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
        torch.testing.assert_close(scales.flatten(), expected.flatten(), rtol=0, atol=0)

    run(values, scales, "opus")
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(values, scales, "opus")
    query.normal_()
    graph.replay()
    check()
