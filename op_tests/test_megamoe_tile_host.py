# SPDX-License-Identifier: MIT
"""CPU checks for EP16 arena registration and benchmark statistics.

Run with ``python op_tests/test_megamoe_tile_host.py``. Loading the modules by
path keeps these host checks independent of AITER's GPU extension imports.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TILE = ROOT / "aiter/ops/flydsl/kernels/megamoe_tile"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = load_module(
    "_megamoe_tile_bench_host_test",
    ROOT / "op_tests/multigpu_tests/test_megamoe_tile_internode.py",
)


class WallStatsTest(unittest.TestCase):
    def test_even_samples_use_middle_pair(self):
        stats = bench.wall_stats([[1, 3, 5, 101], [2, 4, 6, 202]])
        self.assertEqual(stats, dict(min=1.5, median=4.5, pooled_min=1, worst_median=5))

    def test_odd_samples_and_rank_aggregation(self):
        stats = bench.wall_stats([[9, 1, 5], [30, 10, 20]])
        self.assertEqual(
            stats, dict(min=5.5, median=12.5, pooled_min=1, worst_median=20)
        )

    def test_single_sample(self):
        self.assertEqual(bench.wall_stats([[7]])["median"], 7)


class ArenaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("arena checks require PyTorch; no GPU is needed")
        package = load_module("_megamoe_tile_arena_host_test", TILE / "__init__.py")
        cls.s1 = package.Stage1ArenaLayout
        cls.s2 = package.Stage2ArenaLayout
        cls.compose = package.TwoKernelArenaLayout.compose
        cls.operator = package.MegaMoETileA4W4

    def make_layout(self, hidden, experts, topk, tokens, cap, group):
        s2 = self.s2.create(
            hidden=hidden,
            topk=topk,
            max_tokens=tokens,
            include_route_slots=False,
            include_rank_partials=True,
            include_rank_push=True,
            ready_granularity="group",
            ready_group_tiles=2,
            include_plane_slots=True,
        )
        s1 = self.s1.create(
            hidden=hidden,
            inter=3072,
            experts=experts,
            topk=topk,
            max_tokens=tokens,
            max_routes_per_token_per_rank=cap,
            tile_group=group,
            embed_stage2_bytes=s2.total_bytes,
        )
        return self.compose(s1, s2)

    def test_regions_and_registered_prefix_across_supported_capacities(self):
        for hidden, experts, topk in ((3584, 896, 16), (7168, 384, 6)):
            for tokens in (1, 3, 32, 128, 256, 512, 1024, 2048, 4096):
                for group in (1, 2, 4):
                    prefixes = []
                    for cap in (1, 4, topk):
                        with self.subTest(
                            hidden=hidden, tokens=tokens, group=group, cap=cap
                        ):
                            arena = self.make_layout(
                                hidden, experts, topk, tokens, cap, group
                            )
                            regions = [
                                (r.offset, r.nbytes)
                                for r in arena.stage1.regions
                                if r.name != "stage2_embed"
                            ]
                            regions += [
                                (arena.stage2_offset + r.offset, r.nbytes)
                                for r in arena.stage2.regions
                            ]
                            regions.sort()
                            for (start, size), (next_start, _) in zip(
                                regions, regions[1:]
                            ):
                                self.assertLessEqual(start + size, next_start)
                            self.assertLessEqual(
                                regions[-1][0] + regions[-1][1], arena.total_bytes
                            )
                            self.assertEqual(arena.stage2_offset % 4096, 0)
                            op = object.__new__(self.operator)
                            op.layout = arena
                            op.stage1_layout, op.stage2_layout = (
                                arena.stage1,
                                arena.stage2,
                            )
                            prefix = op._rdma_prefix_end()
                            self.assertLessEqual(prefix, arena.total_bytes)
                            self.assertLess(prefix, 1024**3)
                            prefixes.append(prefix)
                    # More local route capacity must never move an RDMA target.
                    self.assertEqual(len(set(prefixes)), 1)

    def test_reject_undersized_stage2_embedding(self):
        s2 = self.s2.create(hidden=3584, topk=16, max_tokens=128)
        s1 = self.s1.create(
            hidden=3584,
            inter=3072,
            experts=896,
            topk=16,
            max_tokens=128,
            embed_stage2_bytes=s2.total_bytes - 1,
        )
        with self.assertRaisesRegex(ValueError, "smaller than stage2"):
            self.compose(s1, s2)


if __name__ == "__main__":
    unittest.main()
