# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune plain OPUS FP8 GEMM; CSV scaleAB selects no-scale or blockscale."""

import argparse
import math
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch

from aiter import dtypes
from aiter.ops.opus import opus_gemm
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from csrc.opus_gemm.opus_gemm_common import (
    canonical_output_dtype,
    get_kernel_instance,
    kernels_list,
)

_SUPPORTED_GFX = "gfx950"


def candidate_kids_for_shape(gfx, m, n, k, scale_ab, outdtype="fp32"):
    """Return registered plain A8W8 kids whose launch constraints fit the shape."""
    if min(m, n, k) <= 0 or k % 2:
        return []
    family = "a8w8_blockscale" if scale_ab else "a8w8"
    candidates = []
    for kid in sorted(kernels_list):
        instance = get_kernel_instance(gfx, family, kid, outdtype)
        if instance is None:
            continue
        # These pipelines prime two K tiles and advance in pairs.
        loops = (k + instance.B_K - 1) // instance.B_K
        if loops < 2 or loops % 2:
            continue
        if not instance.has_oob and any(
            size % tile
            for size, tile in zip((m, n, k), (instance.B_M, instance.B_N, instance.B_K))
        ):
            continue
        if scale_ab and any(
            size % group
            for size, group in zip(
                (m, n, k), (instance.GROUP_M, instance.GROUP_N, instance.GROUP_K)
            )
        ):
            continue
        candidates.append(kid)
    return candidates


def generate_data(m, n, k, kid, *, device):
    instance = kernels_list[kid]
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((m, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    w = torch.randn((n, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    x_scale = w_scale = None
    if instance.GROUP_K:
        x_scale = 0.5 + torch.rand(
            (m // instance.GROUP_M, k // instance.GROUP_K),
            device=device,
            generator=generator,
        )
        w_scale = 0.5 + torch.rand(
            (n // instance.GROUP_N, k // instance.GROUP_K),
            device=device,
            generator=generator,
        )
    return {
        "x": x,
        "w": w,
        "out": torch.empty((m, n), device=device, dtype=torch.float32),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def run_torch(x, w, x_scale, w_scale):
    inputs = []
    for tensor, scale in ((x, x_scale), (w, w_scale)):
        value = tensor.float()
        if scale is not None:
            value = value * scale.repeat_interleave(
                tensor.shape[0] // scale.shape[0], dim=0
            ).repeat_interleave(tensor.shape[1] // scale.shape[1], dim=1)
        inputs.append(value)
    return inputs[0] @ inputs[1].T


def run_bench(x, w, out, x_scale, w_scale, kid):
    opus_gemm(x, w, out, kid=kid, x_scale=x_scale, w_scale=w_scale)
    return out


def compare_outputs(ref, out, **kwargs):
    from aiter.test_common import checkAllclose

    err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, **kwargs)
    # mp_tuner stores four decimals; a rare mismatch must not round down to zero.
    return math.ceil(err * 10000) / 10000


_BENCH_KEYS = ("x", "w", "out", "x_scale", "w_scale")
_REF_KEYS = ("x", "w", "x_scale", "w_scale")


class OpusA8W8Tuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": "/tmp/opus_a8w8_tuned.csv",
        "errRatio": 0.0,
    }

    def __init__(self):
        super().__init__(
            "opus_a8w8",
            key=["gfx", "cu_num", "M", "N", "K", "dtype", "outdtype", "scaleAB"],
            resultList=[
                "libtype",
                "kernelId",
                "splitK",
                "us",
                "kernelName",
                "tflops",
                "bw",
                "errRatio",
            ],
            description="Tune plain OPUS A8W8 GEMM (gfx950, FP8 inputs, FP32 output). "
            "CSV scaleAB=False selects no scales; True selects 1x128x128 blockscale.",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--input_file",
            dest="untune_file",
            default=argparse.SUPPRESS,
            help="Input shape CSV (alias for -i/--untune_file)",
        )
        self.parser.add_argument(
            "--tuned_file",
            dest="tune_file",
            default=argparse.SUPPRESS,
            help="Output tuned CSV (alias for -o/--tune_file)",
        )
        self.parser.add_argument("--libtype", choices=["opus"], default="opus")
        for action in self.parser._actions:
            if action.dest in {
                "splitK",
                "compare",
                "update_improved",
                "min_improvement_pct",
            }:
                action.help = argparse.SUPPRESS
            elif action.dest == "run_config":
                action.help = (
                    "Benchmark saved kids from TUNED_CSV (defaults to --tuned_file)"
                )
            elif action.dest == "errRatio":
                action.help = "Maximum mismatch fraction at rtol=atol=1e-2 (default: 0)"

    def _normalize_rows(self, df):
        df = df.copy()
        missing = {"M", "N", "K"}.difference(df.columns)
        if missing:
            raise ValueError(f"Shape CSV is missing columns: {sorted(missing)}")
        defaults = {
            "gfx": self.get_gfx(),
            "cu_num": self.get_cu_num(),
            "dtype": "fp8",
            "outdtype": "fp32",
            "scaleAB": False,
        }
        for column, default in defaults.items():
            if column not in df:
                df[column] = default
        for column in ("M", "N", "K", "cu_num", "kernelId", "splitK"):
            if column not in df:
                continue
            values = pd.to_numeric(df[column], errors="raise")
            minimum = 0 if column in ("kernelId", "splitK") else 1
            if (values.isna() | (values < minimum) | (values % 1 != 0)).any():
                raise ValueError(f"{column} must contain integers >= {minimum}")
            df[column] = values.astype("int64")
        for column in ("scaleAB", "bias", "bpreshuffle"):
            if column in df:
                try:
                    df[column] = df[column].map(lambda v: dtypes.str2bool(str(v)))
                except argparse.ArgumentTypeError as exc:
                    raise ValueError(f"{column} must be True or False") from exc
                if column != "scaleAB" and df[column].any():
                    raise ValueError(
                        f"Plain OPUS A8W8 tune does not support {column}=True"
                    )
        if (
            not df["dtype"]
            .isin(["fp8", "float8_e4m3fn", str(torch.float8_e4m3fn)])
            .all()
        ):
            raise ValueError("Plain OPUS A8W8 tune requires dtype=fp8 (float8_e4m3fn)")
        if not df["outdtype"].map(canonical_output_dtype).eq("fp32_t").all():
            raise ValueError("Plain OPUS A8W8 tune requires outdtype=fp32")
        if "libtype" in df and not df["libtype"].eq("opus").all():
            raise ValueError("Plain OPUS A8W8 tune requires libtype=opus")
        df["dtype"] = str(torch.float8_e4m3fn)
        df["outdtype"] = str(torch.float32)
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        return self._normalize_rows(df) if not df.empty else df

    def pre_process(self, args):
        gfx = self.get_gfx()
        if gfx != _SUPPORTED_GFX:
            self.parser.error(
                "Plain OPUS A8W8 tuning and --run_config only support "
                f"{_SUPPORTED_GFX}; current GPU is {gfx}"
            )
        if args.splitK:
            self.parser.error("These plain OPUS A8W8 kernels require splitK=0")
        if args.compare or args.update_improved:
            self.parser.error("Use --run_config TUNED_CSV to benchmark saved OPUS kids")
        if args.run_config:
            if args.run_config is True:
                args.run_config = args.tune_file
            if not Path(args.run_config).is_file():
                raise FileNotFoundError(args.run_config)
            self.tunedf = self.get_tuned_gemm_list(args.run_config)
            self.untunedf = self.tunedf
            return
        if not args.untune_file:
            self.parser.error("--input_file/-i is required for tuning")
        df = self._normalize_rows(self.get_untuned_gemm_list(args.untune_file))
        df = df[(df["gfx"] == gfx) & (df["cu_num"] == self.get_cu_num())]
        if df.empty:
            raise ValueError("No input shapes match the current GPU's gfx/cu_num")
        self.untunedf = df[self.keys].drop_duplicates().reset_index(drop=True)
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not args.all and not self.tunedf.empty:
            tuned_keys = self.tunedf[self.keys].apply(tuple, axis=1)
            self.untunedf = self.untunedf[
                ~self.untunedf.apply(tuple, axis=1).isin(tuned_keys)
            ].reset_index(drop=True)

    def tune(self, untunedf, tunedf, args):
        tasks, tasks_data = [], []
        for row in untunedf.itertuples(index=False):
            kids = candidate_kids_for_shape(
                row.gfx, row.M, row.N, row.K, row.scaleAB, row.outdtype
            )
            if not kids:
                raise ValueError(f"No OPUS A8W8 candidates for {row}")
            for kid in kids:
                tasks.append(
                    (
                        (tuple(row), kid, 0, kernels_list[kid].name),
                        generate_data,
                        (row.M, row.N, row.K, kid),
                        run_bench,
                        (_BENCH_KEYS, kid),
                        {"num_warmup": args.warmup, "num_iters": args.iters},
                        run_torch,
                        (_REF_KEYS,),
                        {},
                        None,
                        1e-2,
                        1e-2,
                        compare_outputs,
                        None,
                        ("out",),
                    )
                )
            tasks_data.append((len(kids), ()))
        return mp_tuner(
            tasks,
            tasks_data,
            mp_num=args.mp,
            shape_grouped=args.shape_grouped,
            err_ratio=args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def getKernelName(self, kernel_id):
        return kernels_list[kernel_id].name

    def calculate(self, results, bpes=(1, 1, 4)):
        return super().calculate(results, bpes)

    def result_to_df(self, results):
        df = super().result_to_df(results)
        df["libtype"] = "opus"
        return df[self.columns]

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        missing = {"kernelId", "splitK", "libtype"}.difference(self.untunedf.columns)
        if missing:
            raise ValueError(
                f"--run_config requires a tuned CSV with {sorted(missing)}"
            )
        results = []
        for row in self.untunedf.itertuples(index=False):
            kid = row.kernelId
            if row.splitK != 0:
                raise ValueError(f"OPUS A8W8 kid {kid} requires splitK=0")
            if kid not in candidate_kids_for_shape(
                row.gfx, row.M, row.N, row.K, row.scaleAB, row.outdtype
            ):
                raise ValueError(f"Saved kid {kid} is incompatible with {row}")
            data = generate_data(row.M, row.N, row.K, kid, device="cuda")
            ref = run_torch(*(data[key] for key in _REF_KEYS))
            data["out"].fill_(float("nan"))
            out, us = run_perftest(
                run_bench,
                *(data[key] for key in _BENCH_KEYS),
                kid,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            err = checkAllclose(
                ref,
                out,
                rtol=1e-2,
                atol=1e-2,
                tol_err_ratio=args.errRatio,
                printLog=args.verbose,
            )
            if (
                not math.isfinite(us)
                or us <= 0
                or not math.isfinite(err)
                or err > args.errRatio
            ):
                raise RuntimeError(f"Saved kid {kid} failed: {us=}, errRatio={err}")
            results.append(
                {
                    "shape": f"M={row.M},N={row.N},K={row.K},scaleAB={row.scaleAB},kid={kid}",
                    "e2e_us": us,
                    "errRatio": err,
                    "status": "ok",
                }
            )
        return results


def main():
    tuner = OpusA8W8Tuner()
    tuner.run(tuner.parse_args())


if __name__ == "__main__":
    main()
