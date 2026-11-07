# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-only tests for MegaMoE's stage-1 tile snap (AITER_S1F_SNAP_ALIGN).

A step whose grouped-GEMM CSV tile is not a stage-1 variant keeps dispatch
and GEMM1 as two launches; with the switch on, a step within
AITER_S1F_SNAP_MAX_RECV takes the widest variant below that tile instead.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import s1f_ctl

_TILES = {64, 128, 256}
_CONFIG = object()


def _align(align_m, recv_bound, *, tiles=_TILES, snap="1"):
    with mock.patch.dict(os.environ, {"AITER_S1F_SNAP_ALIGN": snap}), mock.patch.object(
        s1f_ctl, "_snap_tiles", return_value=set(tiles)
    ):
        return s1f_ctl.s1f_step_align(_CONFIG, align_m, recv_bound)


class TestS1fStepAlign(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AITER_S1F_SNAP_ALIGN", None)
            self.assertEqual(s1f_ctl.s1f_step_align(_CONFIG, 96, 2304), 96)

    def test_decode_step_snaps_to_the_widest_variant_below(self):
        # DSV4.1-Flash DEP4 at 576 tokens per rank: the CSV names 96.
        self.assertEqual(_align(96, 2304), 64)

    def test_stops_above_the_snap_bound(self):
        self.assertEqual(_align(96, s1f_ctl._S1F_SNAP_MAX_RECV), 64)
        self.assertEqual(_align(96, s1f_ctl._S1F_SNAP_MAX_RECV + 1), 96)

    def test_a_variant_tile_is_kept(self):
        for tile in _TILES:
            self.assertEqual(_align(tile, 2304), tile)

    def test_no_variant_below_keeps_the_tile(self):
        # Low-batch CSV tiles: padding them up would cost more than it saves.
        for tile in (16, 32):
            self.assertEqual(_align(tile, 512), tile)

    def test_arena_sizing_keeps_the_capacity_tile(self):
        self.assertEqual(_align(96, None), 96)

    def test_no_stage1_kernel_keeps_the_tile(self):
        self.assertEqual(_align(96, 2304, tiles=()), 96)


if __name__ == "__main__":
    unittest.main()
