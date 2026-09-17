# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The conv3d candidate sweep has to contain the config it is trying to beat.

`--compare --update_improved` only rejects a regression after measuring it, and
it can only measure the incumbent if the incumbent is in the candidate list. It
was not: `get_flydsl_conv3d_configs` pinned the baseline tiles at wgm=1 while
`_pick_wgm` decides independently and returns 4 or 8 on plenty of shapes. On
384->384 @48x70 the heuristic runs (32,32,1,2) at wgm=8, the sweep only offered
(32,32,1,2,1), and the winner measured 18.9% *slower* than the config it was
meant to replace.

Nothing about that failure is visible in a correctness test -- the answer stays
right, it just gets slower -- and the tuner's own report looked healthy, since
it compared candidates against each other rather than against the incumbent.
Hence a test on the candidate set itself, over the shapes the VAEs actually run
plus the small-M range where the heuristic switches ladders.
"""

from __future__ import annotations

import unittest

import torch

from aiter.ops.flydsl.conv3d_policy import (
    BASELINE_TILES,
    WGM_VALUES,
    get_flydsl_conv3d_configs,
    is_legal_tile,
)

# (npq, kg) pairs. The first eight are the Wan T>1 shapes at 480x832 and
# 368x544; the rest walk M down through the range where _pick_tile demotes.
SHAPES = [
    (1597440, 96),
    (399360, 192),
    (49920, 384),
    (6240, 384),
    (6240, 32),
    (800768, 96),
    (200192, 192),
    (3128, 384),
    (3128, 32),
    (1048576, 96),
    (262144, 192),
    (65536, 384),
    (16384, 384),
    (27556, 384),
    (1763584, 96),
    (1024, 128),
    (512, 64),
    (256, 384),
    (128, 16),
]


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU for the CU count")
class TestConv3dPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.ops.flydsl.kernels.conv3d_implicit import _pick_tile, _pick_wgm

        cls._pick_tile = staticmethod(_pick_tile)
        cls._pick_wgm = staticmethod(_pick_wgm)
        cls.dev = torch.device("cuda")
        cls.num_cu = torch.cuda.get_device_properties(cls.dev).multi_processor_count

    def test_incumbent_is_always_a_candidate(self):
        """What the production heuristic would pick must be in the sweep."""
        missing = []
        for npq, kg in SHAPES:
            tile = self._pick_tile(npq, kg, 1, self.dev)
            wgm = self._pick_wgm(npq, kg, 1, tile, self.dev)
            incumbent = (*tile, wgm)
            cands = get_flydsl_conv3d_configs(npq, kg, 1, self.num_cu)
            if incumbent not in cands:
                missing.append(f"npq={npq} kg={kg}: {incumbent} not among {len(cands)}")
        self.assertEqual(
            missing,
            [],
            "the heuristic's own config was not offered to the tuner, so "
            '"tuned is never worse than default" cannot hold:\n' + "\n".join(missing),
        )

    def test_baseline_tiles_cover_every_wgm(self):
        """Independently of the heuristic: each legal baseline tile, each WGM.

        The bug was one missing dimension of the cross product, so assert the
        cross product rather than only the shapes that happen to expose it.
        """
        for npq, kg in ((6240, 384), (3128, 384), (399360, 192)):
            cands = set(get_flydsl_conv3d_configs(npq, kg, 1, self.num_cu))
            for tile in BASELINE_TILES:
                if not is_legal_tile(*tile):
                    continue
                for wgm in WGM_VALUES:
                    self.assertIn(
                        (*tile, wgm),
                        cands,
                        f"npq={npq} kg={kg} missing baseline {tile} at wgm={wgm}",
                    )

    def test_baseline_contains_the_kernel_ladder(self):
        """BASELINE_TILES is spelled out, so nothing stops the ladder drifting off it.

        ``_pick_tile`` demotes through ``TILE_LADDER``; a rung that is not also a
        baseline tile is an incumbent the sweep would never measure, which is the
        same hole test_incumbent_is_always_a_candidate covers for these shapes and
        this one covers for every shape at once.
        """
        from aiter.ops.flydsl.kernels.conv3d_implicit import TILE_LADDER

        for tile in TILE_LADDER:
            self.assertIn(
                tile,
                BASELINE_TILES,
                f"kernel ladder rung {tile} is not a baseline candidate",
            )

    def test_every_candidate_is_legal(self):
        """A candidate that cannot compile wastes a sweep slot and logs a failure."""
        for npq, kg in SHAPES:
            for tm, tn, wm, wn, wgm in get_flydsl_conv3d_configs(
                npq, kg, 1, self.num_cu
            ):
                self.assertTrue(
                    is_legal_tile(tm, tn, wm, wn),
                    f"illegal candidate ({tm},{tn},{wm},{wn}) for npq={npq} kg={kg}",
                )
                self.assertIn(wgm, WGM_VALUES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
