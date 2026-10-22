# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import importlib.util
import os
import tempfile
import unittest
from unittest import mock

from aiter.aot.flydsl.common import collect_cli_jobs, get_gfx_list_env

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


class TestGfxListEnv(unittest.TestCase):
    def test_gpu_archs_parsing(self):
        cases = [
            ({}, None),
            ({"GPU_ARCHS": "native"}, None),
            ({"ARCH": "x86_64"}, None),
            ({"GPU_ARCHS": "gfx942;gfx950"}, ["gfx942", "gfx950"]),
            ({"GPU_ARCHS": "gfx942,gfx950"}, ["gfx942", "gfx950"]),
        ]
        for env, expected in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(get_gfx_list_env(), expected)

    def test_cli_arch_replaces_gpu_archs(self):
        env = {"GPU_ARCHS": "gfx942", "ARCH": "gfx950"}
        with mock.patch.dict(os.environ, env, clear=True):
            jobs = collect_cli_jobs(
                [__file__], lambda _: [{"archs": tuple(get_gfx_list_env() or ())}]
            )
            self.assertEqual(jobs, [{"archs": ("gfx950",)}])
            self.assertEqual(os.environ["GPU_ARCHS"], "gfx942")


@unittest.skipUnless(importlib.util.find_spec("flydsl"), "FlyDSL is not installed")
class TestGemmParseCsvArchFilter(unittest.TestCase):
    def test_gpu_archs_keep_only_listed_archs(self):
        from aiter.aot.flydsl import gemm

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "gemm.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(GEMM_ROWS[0]))
                writer.writeheader()
                writer.writerows(GEMM_ROWS)

            cases = {"": ["gfx1250", "gfx950"], "gfx950": ["gfx950"], "gfx942": []}
            for gpu_archs, expected in cases.items():
                with self.subTest(gpu_archs=gpu_archs):
                    env = {"GPU_ARCHS": gpu_archs}
                    with mock.patch.dict(os.environ, env, clear=True):
                        jobs = gemm.parse_csv(csv_path)
                    archs = sorted({gemm.job_arch(j["cu_num"], j["gfx"]) for j in jobs})
                    self.assertEqual(archs, expected)


if __name__ == "__main__":
    unittest.main()
