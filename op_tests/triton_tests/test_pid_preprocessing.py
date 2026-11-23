# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd_aid


@triton.jit
def _remap_all_kernel(
    out_ptr,
    GRID_MN,
    NUM_AIDS: tl.constexpr,
    XCDS_PER_AID: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.store(
        out_ptr + pid,
        remap_xcd_aid(pid, GRID_MN, NUM_AIDS, XCDS_PER_AID),
    )


def _remap(grid_mn, num_aids, xcds_per_aid):
    out = torch.zeros(grid_mn, dtype=torch.int32, device="cuda")
    _remap_all_kernel[(grid_mn,)](out, grid_mn, num_aids, xcds_per_aid)
    return out


# A ragged GRID_MN is the interesting case: the permuted range must stop exactly where the
# tail begins, or a permuted pid collides with a tail pid that maps to itself.
@pytest.mark.parametrize("grid_mn", [5, 8, 12, 16, 20, 32, 64, 236, 256])
@pytest.mark.parametrize("num_aids, xcds_per_aid", [(1, 4), (2, 4), (2, 2), (4, 2)])
def test_remap_xcd_aid_is_a_bijection(grid_mn, num_aids, xcds_per_aid):
    out = _remap(grid_mn, num_aids, xcds_per_aid).cpu()
    expected = torch.arange(grid_mn, dtype=torch.int32)
    assert torch.equal(
        torch.sort(out).values, expected
    ), f"not a permutation of 0..{grid_mn - 1}"


@pytest.mark.parametrize("grid_mn", [64, 236, 256])
def test_remap_xcd_aid_gives_each_xcd_one_run_in_dispatch_order(grid_mn):
    """The pids of one XCD get consecutive tiles, and the runs follow the XCD order.

    The start of the round robin depends on the hardware queue, so the test checks that
    one rotation n places every pid: pid p is on XCD (p - n) mod W and holds a tile of
    that XCD's run.
    """
    W = 8
    blk = grid_mn // W
    out = _remap(grid_mn, 2, 4).cpu().tolist()
    for p in range(blk * W - W):
        assert out[p + W] == out[p] + 1, (p, out[p], out[p + W])
    rotations = {(p - out[p] // blk) % W for p in range(blk * W)}
    assert len(rotations) == 1, rotations
    assert out[blk * W :] == list(range(blk * W, grid_mn))
