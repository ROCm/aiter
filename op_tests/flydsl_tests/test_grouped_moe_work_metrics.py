# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks of route padding, touched buffers and WMMA accounting."""

import unittest
from dataclasses import replace

from op_tests.flydsl_tests.grouped_moe_work_metrics import (
    GemmGeometry,
    grouped_moe_work_metrics,
)


class TestGroupedMoeWorkMetrics(unittest.TestCase):
    def setUp(self):
        self.g1 = GemmGeometry(1024, 512, 256, 256, 256, 2, 2, True, True, True, True)
        self.g2 = GemmGeometry(512, 512, 256, 256, 256, 2, 2, True, True)

    def test_partial_m16_and_inactive_expert(self):
        work = grouped_moe_work_metrics([1, 17, 0], self.g1, self.g2)
        self.assertEqual(work["gemm1"]["transfers"], {
            "A": 12288, "B": 524288, "ScaleA": 8192,
            "ScaleB": 32768, "output": 12288, "output_scale": 768,
        })
        self.assertEqual(work["gemm2"]["transfers"], {
            "A": 12288, "B": 262144, "ScaleA": 8192,
            "ScaleB": 16384, "output": 18432, "output_scale": 0,
        })
        self.assertEqual(work["gemm1"]["bytes"], 590592)
        self.assertEqual(work["gemm2"]["bytes"], 317440)
        self.assertEqual(work["gemm1"]["flops"], 536870912)
        self.assertEqual(work["gemm1"]["wmma_instructions"], 4096)
        self.assertEqual(work["gemm2"]["wmma_instructions"], 2048)
        self.assertEqual(work["gemm1"]["weight_experts"], [0, 1])

    def test_tile_m_changes_only_full_scale_rows_and_compute(self):
        old = grouped_moe_work_metrics([1, 17, 0], self.g1, self.g2)
        new = grouped_moe_work_metrics(
            [1, 17, 0], replace(self.g1, tile_m=288), replace(self.g2, tile_m=288))
        self.assertEqual(new["gemm1"]["compute_rows"], 576)
        self.assertEqual(new["gemm1"]["flops"], 603979776)
        self.assertEqual(new["gemm1"]["transfers"]["ScaleA"], 9216)
        for stage in ("gemm1", "gemm2"):
            for buffer in ("A", "B", "ScaleB", "output", "output_scale"):
                self.assertEqual(old[stage]["transfers"][buffer],
                                 new[stage]["transfers"][buffer])

    def test_n_partition_does_not_multiply_repeated_reads(self):
        reference = grouped_moe_work_metrics([1, 17], self.g1, self.g2)
        for tile_n in (128, 512):
            work = grouped_moe_work_metrics(
                [1, 17], replace(self.g1, tile_n=tile_n),
                replace(self.g2, tile_n=tile_n))
            for stage in ("gemm1", "gemm2"):
                self.assertEqual(work[stage]["bytes"], reference[stage]["bytes"])
                self.assertEqual(work[stage]["flops"], reference[stage]["flops"])

    def test_k_tail_uses_only_issued_k128_steps(self):
        g1 = replace(self.g1, k=640, a_preshuffled=False)
        work = grouped_moe_work_metrics([1, 17], g1, self.g2)
        self.assertEqual(work["gemm1"]["flops"], 671088640)
        self.assertEqual(work["gemm1"]["wmma_instructions"], 5120)
        self.assertEqual(work["gemm1"]["transfers"]["A"], 5760)
        self.assertEqual(work["gemm1"]["transfers"]["ScaleA"], 10240)

    def test_non_apre_quant_output_has_row_oob_and_m16_scales(self):
        work = grouped_moe_work_metrics(
            [1, 17], replace(self.g1, a_preshuffled=False),
            replace(self.g2, a_preshuffled=False))
        self.assertEqual(work["gemm1"]["transfers"]["A"], 4608)
        self.assertEqual(work["gemm1"]["transfers"]["output"], 4608)
        self.assertEqual(work["gemm1"]["transfers"]["output_scale"], 768)
        self.assertEqual(work["gemm2"]["transfers"]["A"], 4608)

    def test_fp8_and_bf16_fallback(self):
        fp8 = replace(self.g1, a_is_fp4=False, a_preshuffled=False)
        g2 = replace(self.g2, a_is_fp4=False, a_preshuffled=False)
        work = grouped_moe_work_metrics([1, 17], fp8, g2)
        self.assertEqual(work["gemm1"]["transfers"]["A"], 9216)
        self.assertEqual(work["gemm1"]["transfers"]["output"], 9216)
        self.assertEqual(work["gemm1"]["wmma_instructions"], 8192)
        fallback = grouped_moe_work_metrics(
            [1, 17], replace(fp8, stage1_quant_out=False), g2)
        self.assertEqual(fallback["gemm1"]["transfers"]["output"], 18432)
        self.assertEqual(fallback["gemm1"]["transfers"]["output_scale"], 0)

    def test_different_stage_tiles_use_actual_shared_row_map(self):
        work = grouped_moe_work_metrics(
            [1, 1], replace(self.g1, tile_m=128), self.g2,
            expert_ends=[1, 257], capacity_rows=512)
        self.assertEqual(work["expert_starts"], [0, 256])
        # The actual psum bisect assigns the intervening M128 padding tile
        # to expert 1, whose input/output descriptors also cover that gap.
        self.assertEqual(work["gemm1"]["compute_rows"], 384)
        self.assertEqual(work["gemm1"]["input_rows"], 160)
        self.assertEqual(work["gemm2"]["compute_rows"], 512)
        self.assertEqual(work["gemm2"]["input_rows"], 32)
        self.assertEqual(work["gemm2"]["output_rows"], 2)

    def test_all_dropped_routes_count_no_work(self):
        work = grouped_moe_work_metrics([0, 0, 0], self.g1, self.g2)
        self.assertEqual(work["total"], {"flops": 0, "bytes": 0, "wmma_instructions": 0})
        self.assertEqual(work["gemm1"]["weight_experts"], [])

    def test_histogram_must_match_same_launch_psum(self):
        with self.assertRaisesRegex(ValueError, "histogram"):
            grouped_moe_work_metrics([1, 17], self.g1, self.g2,
                                     expert_ends=[1, 272])
        with self.assertRaisesRegex(ValueError, "capacity"):
            grouped_moe_work_metrics([1, 17], self.g1, self.g2, capacity_rows=256)

    def test_unsupported_geometry_is_not_assigned_fake_bytes(self):
        for g1 in (replace(self.g1, n=1056), replace(self.g1, k=513),
                   replace(self.g1, tile_m=272)):
            with self.assertRaises(ValueError):
                grouped_moe_work_metrics([1, 17], g1, self.g2)


if __name__ == "__main__":
    unittest.main()
