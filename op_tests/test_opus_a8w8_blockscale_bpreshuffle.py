# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 OPUS A8W8 blockscale B-preshuffle GEMM matches an fp32 GEMM on every launch.

The single-buffer pipeline refills its A/B LDS tiles in place at every K step. A wave that
refills a tile before the other waves have read it corrupts their operands, and which
elements go wrong changes from launch to launch, so every case launches several times.
The race has only shown up with more workgroups than CUs and at least two K tiles, so the last
cases have more than 2 x 304 output tiles: two workgroups on every CU of a 304-CU part.
"""

import pytest
import torch

from aiter import dtypes
from aiter.ops.opus import opus_gemm
from aiter.ops.shuffle import shuffle_weight

GROUP = 128
LAUNCHES = 5


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(props.gcnArchName).split(":", 1)[0].lower()


def _quant_activation(x: torch.Tensor):
    """1 x 128 group quant. The kernel reads the scales column-major through a [M, K/128] view."""
    m, k = x.shape
    groups = x.float().view(m, k // GROUP, GROUP)
    scale = groups.abs().amax(dim=-1).clamp_min(1e-12) / torch.finfo(dtypes.fp8).max
    xq = (groups / scale.unsqueeze(-1)).view(m, k).to(dtypes.fp8)
    return xq, scale, scale.t().contiguous().view(m, k // GROUP)


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (1408, 1536, 4096),  # whole M tiles
        (1345, 1536, 4096),  # last M tile wider than half a tile
        (300, 4096, 1024),  # last M tile at most half a tile: the half-M pipeline
        (64, 12288, 4096),  # every tile on the half-M pipeline
        (1280, 4096, 2048),
        (2048, 8192, 1024),
        (4097, 8192, 1024),  # 2112 tiles; last M tile of one row: the half-M pipeline
        (4161, 8192, 1024),  # 2112 tiles; last M tile of 65 rows: masked A loads
    ],
)
def test_gfx942_bpreshuffle_matches_fp32_on_every_launch(M, N, K):
    if _runtime_arch() != "gfx942":
        pytest.skip("requires gfx942 hardware")
    torch.manual_seed(M * N + K)
    w = (torch.randn(N, K, device="cuda") * 0.5).to(dtypes.fp8)
    w_scale = torch.exp2(
        torch.randint(-10, -6, (N // GROUP, K // GROUP), device="cuda").float()
    )
    xq, x_scale, x_scale_kernel = _quant_activation(torch.randn(M, K, device="cuda"))
    golden = (xq.float() * x_scale.repeat_interleave(GROUP, 1)) @ (
        w.float() * w_scale.repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)
    ).T
    wq = shuffle_weight(w, layout=(16, 16))
    for launch in range(LAUNCHES):
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        out = opus_gemm(
            xq,
            wq,
            y,
            family="a8w8_blockscale_bpreshuffle",
            kid=0,
            layout="bpreshuffle",
            x_scale=x_scale_kernel,
            w_scale=w_scale,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(),
            golden,
            rtol=1e-2,
            atol=1e-2,
            msg=lambda m, launch=launch: f"launch {launch}: {m}",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
