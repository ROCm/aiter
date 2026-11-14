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
    AID_CONTIGUOUS: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.store(
        out_ptr + pid,
        remap_xcd_aid(pid, GRID_MN, NUM_AIDS, XCDS_PER_AID, AID_CONTIGUOUS),
    )


def _remap(grid_mn, num_aids, xcds_per_aid, aid_contiguous):
    out = torch.zeros(grid_mn, dtype=torch.int32, device="cuda")
    _remap_all_kernel[(grid_mn,)](out, grid_mn, num_aids, xcds_per_aid, aid_contiguous)
    return out


# A ragged GRID_MN is the interesting case: the permuted range must stop exactly where the
# tail begins, or a permuted pid collides with a tail pid that maps to itself.
@pytest.mark.parametrize("grid_mn", [5, 8, 12, 16, 20, 32, 64, 256])
@pytest.mark.parametrize("num_aids, xcds_per_aid", [(1, 4), (2, 4), (2, 2), (4, 2)])
@pytest.mark.parametrize("aid_contiguous", [True, False])
def test_remap_xcd_aid_is_a_bijection(grid_mn, num_aids, xcds_per_aid, aid_contiguous):
    out = _remap(grid_mn, num_aids, xcds_per_aid, aid_contiguous).cpu()
    expected = torch.arange(grid_mn, dtype=torch.int32)
    assert torch.equal(
        torch.sort(out).values, expected
    ), f"not a permutation of 0..{grid_mn - 1}"


def test_remap_xcd_aid_keeps_an_aid_on_adjacent_tiles():
    """An AID's XCDs must get adjacent tiles, not one run apart.

    The L2 is shared by the XCDs of one AID, so the four pids that land on one AID at the
    same dispatch step must map to four consecutive tiles.
    """
    num_aids, xcds_per_aid = 2, 4
    out = _remap(256, num_aids, xcds_per_aid, True).cpu().tolist()
    # pids 0..3 are the first dispatch step of AID 0, pids 4..7 of AID 1.
    assert out[0:4] == [0, 1, 2, 3], out[0:4]
    assert out[4:8] == [128, 129, 130, 131], out[4:8]
