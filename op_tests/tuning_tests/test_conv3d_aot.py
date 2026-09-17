# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Does the conv3d AOT pass actually cover what the runtime asks for?

``aiter.aot.flydsl.conv`` derives its compile keys from the tuned CSV, while the
runtime derives them inside ``_conv3d_impl``. The two derivations live in
different files, so agreeing by construction is not something to assume: a
change to the padding rules, the channel padding or the split-K heuristic could
desync them, and the only symptom would be a silent cache miss that falls back
to JIT -- slower, but not wrong, so nothing fails.

This test closes that hole by running the real op under
``run_only_env()``, where FlyDSL refuses to JIT and raises instead. A shape the
AOT pass did not cover, or covered under a different key, therefore fails here.

Needs a GPU, and runs the convolutions at their true sizes (the heaviest Wan
shape moves ~440 MiB), so it is a tuning test rather than part of the op sweep.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import pandas as pd
import torch

from aiter.aot.flydsl.common import OpKind, run_only_env
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl import flydsl_conv_implicit
from aiter.ops.flydsl.kernels.conv3d_implicit import _pad_channels

# Same gate as test_flydsl_conv_implicit.py: conv3d_implicit is gfx950-only.
SUPPORTED_GFX = ("gfx950",)


def _rows_for_this_device():
    """Tuned rows the runtime could actually hit on the device under test."""
    path = AITER_CONFIGS.AITER_CONFIG_CONV3D_BF16_FILE
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if df.empty:
        return df
    if "gfx" in df.columns:
        df = df[df["gfx"].astype(str) == str(get_gfx())]
    if "cu_num" in df.columns:
        df = df[df["cu_num"] == get_cu_num()]
    return df.reset_index(drop=True)


def _call_row(row, output_layout=None):
    """Run one tuned row through the production entry point."""
    n, c, d, h, w = (int(row[k]) for k in ("N", "C", "D", "H", "W"))
    k, kt, kh, kw = (int(row[x]) for x in ("K", "kT", "kH", "kW"))
    groups = int(row["groups"])
    has_bias = str(row["bias"]).strip().lower() == "true"
    stride = tuple(int(row[f"stride_{a}"]) for a in "dhw")
    padding = tuple(int(row[f"pad_{a}"]) for a in "dhw")
    dilation = tuple(int(row[f"dil_{a}"]) for a in "dhw")

    torch.manual_seed(0)
    dev = torch.device("cuda")
    # A tuned row is stored in the 3-D normalised form; a kT of 1 with D of 1 is
    # how a conv2d call is recorded, and it has to be replayed through the same
    # 2-D entry the model uses or the compile key will not match.
    if kt == 1 and d == 1:
        x = torch.randn((n, c, h, w), device=dev, dtype=torch.bfloat16)
        weight = torch.randn((k, c // groups, kh, kw), device=dev, dtype=torch.bfloat16)
        stride, padding, dilation = stride[1:], padding[1:], dilation[1:]
    else:
        x = torch.randn((n, c, d, h, w), device=dev, dtype=torch.bfloat16)
        weight = torch.randn(
            (k, c // groups, kt, kh, kw), device=dev, dtype=torch.bfloat16
        )
    bias = torch.randn((k,), device=dev, dtype=torch.bfloat16) if has_bias else None
    extra = {}
    if output_layout is not None:
        extra["output_layout"] = "NHWC" if (kt == 1 and d == 1) else "NDHWC"
    try:
        return flydsl_conv_implicit(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **extra,
        )
    finally:
        del x, weight, bias
        torch.cuda.empty_cache()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
@unittest.skipUnless(
    get_gfx() in SUPPORTED_GFX, f"flydsl conv3d unsupported on {get_gfx()}"
)
class TestConv3dAotCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _rows_for_this_device()
        if cls.rows.empty:
            raise unittest.SkipTest("no tuned conv3d rows for this device")

        cls.cache_dir = tempfile.mkdtemp(prefix="conv3d_aot_")
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cls.cache_dir
        # Same entry point setup.py drives, so the test exercises the shipped
        # path rather than a test-only reimplementation of it.
        from aiter.aot.flydsl.common import _collect_aot_jobs_for, run_jobs_parallel
        from aiter.aot.flydsl.conv import compile_one_config

        cls.jobs = _collect_aot_jobs_for(OpKind.CONV)
        cls.results = run_jobs_parallel(compile_one_config, cls.jobs)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("FLYDSL_RUNTIME_CACHE_DIR", None)
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_every_job_compiles(self):
        failed = [r["shape"] for r in self.results if r["compile_time"] is None]
        self.assertEqual(failed, [], f"{len(failed)} AOT compiles failed")

    def test_jobs_cover_both_kernels(self):
        kinds = {j["kind"] for j in self.jobs}
        self.assertIn("conv3d", kinds)
        # The NCDHW->NHWC pre-transpose is a separate lru_cache; leaving it out
        # would let it JIT at runtime while the convolution itself was covered.
        self.assertIn("transpose", kinds)

    def _aot_then_run_only(self, *, c, k, d, h, w, groups, tile, wgm, splitk, bias):
        """AOT one synthetic config, then make the runtime use it, run-only.

        The tuned CSV is all ``splitK=1``/``groups=1``, so the branches those
        select in this module would otherwise be dead code -- compiled by nobody
        and therefore never checked against the key the runtime derives. Forcing
        ``tile``/``wgm``/``splitk`` at the call site reaches them without
        inventing CSV rows for shapes no model runs.
        """
        from aiter.aot.flydsl.conv import compile_one_config

        cgp = _pad_channels(c // groups)
        c_padded = groups * cgp
        base = {"cu_num": get_cu_num(), "gfx": get_gfx()}
        shape = {
            "N": 1,
            "C": c,
            "D": d,
            "H": h,
            "W": w,
            "K": k,
            "kT": 3,
            "kH": 3,
            "kW": 3,
            "stride_d": 1,
            "stride_h": 1,
            "stride_w": 1,
            "pad_d": 1,
            "pad_h": 1,
            "pad_w": 1,
            "dil_d": 1,
            "dil_h": 1,
            "dil_w": 1,
            "groups": groups,
        }
        jobs = [
            {
                "kind": "conv3d",
                "kernel_name": "conv3d_implicit_kernel",
                "c_padded": c_padded,
                "has_bias": bias,
                "splitk": splitk,
                "tile_m": tile[0],
                "tile_n": tile[1],
                "wave_m": tile[2],
                "wave_n": tile[3],
                "wgm": wgm,
                **base,
                **shape,
            },
            {
                "kind": "transpose",
                "kernel_name": "transpose_ncdhw_ndhwc",
                "N": 1,
                "c_padded": c_padded,
                "s": d * h * w,
                **base,
            },
        ]
        for job in jobs:
            res = compile_one_config(**job)
            self.assertIsNotNone(
                res["compile_time"], f"AOT compile failed: {res['shape']}"
            )

        torch.manual_seed(0)
        dev = torch.device("cuda")
        x = torch.randn((1, c, d, h, w), device=dev, dtype=torch.bfloat16)
        weight = torch.randn(
            (k, c // groups, 3, 3, 3), device=dev, dtype=torch.bfloat16
        )
        b = torch.randn((k,), device=dev, dtype=torch.bfloat16) if bias else None
        ref = torch.nn.functional.conv3d(
            x, weight, b, stride=1, padding=1, groups=groups
        )
        with run_only_env():
            out = flydsl_conv_implicit(
                x,
                weight,
                b,
                stride=1,
                padding=1,
                groups=groups,
                tile=tile,
                wgm=wgm,
                splitk=splitk,
            )
        torch.cuda.synchronize()
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    def test_splitk_path_is_covered(self):
        """split-K swaps the output for an (npq, k) fp32 staging buffer.

        That drops the epilogue argument from rank 5 to rank 2, and rank is part
        of the cache key even though the extents are not -- the mistake this
        whole test file exists to catch.
        """
        self._aot_then_run_only(
            c=128,
            k=128,
            d=4,
            h=16,
            w=16,
            groups=1,
            tile=(128, 128, 2, 4),
            wgm=1,
            splitk=2,
            bias=True,
        )

    def test_groups_path_is_covered(self):
        self._aot_then_run_only(
            c=32,
            k=48,
            d=4,
            h=16,
            w=16,
            groups=4,
            tile=(32, 32, 1, 2),
            wgm=1,
            splitk=1,
            bias=True,
        )

    def _run_only_sweep(self, output_layout=None):
        missed = []
        with run_only_env():
            for i, row in self.rows.iterrows():
                try:
                    self.assertIsNotNone(_call_row(row, output_layout))
                except Exception as exc:  # noqa: BLE001
                    missed.append(
                        f"row {i} ({row['C']}->{row['K']} D{row['D']} "
                        f"{row['H']}x{row['W']} bias={row['bias']}): "
                        f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                    )
        return missed

    def test_runtime_never_jits(self):
        """The real op, run-only. An uncovered or mis-keyed shape raises here."""
        missed = self._run_only_sweep()
        self.assertEqual(
            missed,
            [],
            f"{len(missed)}/{len(self.rows)} tuned rows were not served from the "
            "AOT cache:\n" + "\n".join(missed),
        )

    def test_runtime_never_jits_channels_last_out(self):
        """Same sweep with a channels-last output.

        ``out_ndhwc`` flips the epilogue, so it is a second compile per row.
        Covering it now means a caller that later keeps a VAE stage in
        channels-last does not fall back to JIT -- and the measurement that
        motivates doing so at all is small (1.10x on the Wan shapes, against a
        boundary conversion in torch that costs ~18x the op's own transpose),
        so the JIT would have swamped the gain.
        """
        missed = self._run_only_sweep(output_layout="channels_last")
        self.assertEqual(
            missed,
            [],
            f"{len(missed)}/{len(self.rows)} tuned rows were not served from the "
            "AOT cache with a channels-last output:\n" + "\n".join(missed),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
