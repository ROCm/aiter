# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import importlib.util
import os
import tempfile
import unittest
from unittest import mock

from aiter.aot.flydsl.common import aot_target_archs, collect_cli_jobs

GEMM_ROWS = [
    {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": "64",
        "N": "192",
        "K": "1024",
        "libtype": "flydsl",
        "kernelId": "989",
        "kernelName": "flydsl_bpreshuflle_16x64x512_F8_F8_B16_0x3x0x1_default",
    },
    {
        "gfx": "gfx1250",
        "cu_num": "256",
        "M": "16",
        "N": "2112",
        "K": "7168",
        "libtype": "flydsl",
        "kernelId": "21",
        "kernelName": "flydsl_bpreshuffle_wmma_t16x64x1024_mw1_nw4_nb3_sk1_cm1_cn1",
    },
]


CONV_ROW = {
    "gfx": "gfx950",
    "cu_num": "256",
    "N": "1",
    "C": "3",
    "D": "6",
    "H": "370",
    "W": "546",
    "K": "96",
    "kT": "3",
    "kH": "3",
    "kW": "3",
    "stride_d": "1",
    "stride_h": "1",
    "stride_w": "1",
    "pad_d": "0",
    "pad_h": "0",
    "pad_w": "0",
    "dil_d": "1",
    "dil_h": "1",
    "dil_w": "1",
    "groups": "1",
    "bias": "True",
    "libtype": "flydsl",
    "tile_m": "192",
    "tile_n": "96",
    "wave_m": "4",
    "wave_n": "1",
    "wgm": "1",
    "block_r": "1",
    "splitK": "1",
    "kernelName": "conv3d_implicit_t192x96_w4x1_g1",
}
CONV_ROWS = [CONV_ROW, {**CONV_ROW, "gfx": "gfx942", "cu_num": "304"}]


class TestCliJobs(unittest.TestCase):
    def test_cli_arch_replaces_gpu_archs(self):
        env = {"GPU_ARCHS": "gfx942", "ARCH": "gfx950"}
        with mock.patch.dict(os.environ, env, clear=True):
            jobs = collect_cli_jobs(
                [__file__],
                lambda _: [{"archs": tuple(sorted(aot_target_archs() or ()))}],
            )
            self.assertEqual(jobs, [{"archs": ("gfx950",)}])
            self.assertEqual(os.environ["GPU_ARCHS"], "gfx942")


@unittest.skipUnless(importlib.util.find_spec("flydsl"), "FlyDSL is not installed")
class TestParseCsvArchFilter(unittest.TestCase):
    def check_archs(self, parse_csv, job_arch, rows, cases):
        """Run parse_csv over rows under each env, and compare the job archs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "tuned.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            for env, expected in cases:
                with self.subTest(env=env):
                    with mock.patch.dict(os.environ, env, clear=True):
                        jobs = parse_csv(csv_path)
                    self.assertEqual(sorted({job_arch(j) for j in jobs}), expected)

    def test_gemm(self):
        from aiter.aot.flydsl import gemm

        self.check_archs(
            gemm.parse_csv,
            lambda j: gemm.job_arch(j["cu_num"], j["gfx"]),
            GEMM_ROWS,
            [
                ({}, ["gfx1250", "gfx950"]),
                ({"GPU_ARCHS": "gfx950"}, ["gfx950"]),
                ({"GPU_ARCHS": "gfx942"}, []),
            ],
        )

    def test_conv(self):
        from aiter.aot.flydsl import conv

        self.check_archs(
            conv.parse_csv,
            lambda j: conv.job_arch(j["cu_num"], j["gfx"]),
            CONV_ROWS,
            [
                ({}, ["gfx942", "gfx950"]),
                ({"GPU_ARCHS": "gfx950"}, ["gfx950"]),
                # parse_csv ignores ARCH, which build images set to the machine type.
                ({"ARCH": "x86_64"}, ["gfx942", "gfx950"]),
            ],
        )


@unittest.skipUnless(importlib.util.find_spec("flydsl"), "FlyDSL is not installed")
class TestEveryKindFiltersArch(unittest.TestCase):
    """A new OpKind must apply GPU_ARCHS to its jobs; see the AOT README."""

    def test_unlisted_arch_builds_nothing(self):
        from aiter.aot.flydsl.common import OpKind, _collect_aot_jobs_for

        # MoE and MegaMoE replay the runtime in run_aot and filter there.
        pooled_kinds = [k for k in OpKind if k not in (OpKind.MOE, OpKind.MEGA_MOE)]
        for kind in pooled_kinds:
            with self.subTest(kind=kind.name):
                with mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx1100"}):
                    filtered = _collect_aot_jobs_for(kind)
                self.assertEqual(filtered, [])
                with mock.patch.dict(os.environ):
                    os.environ.pop("GPU_ARCHS", None)
                    self.assertTrue(_collect_aot_jobs_for(kind))


if __name__ == "__main__":
    unittest.main()
