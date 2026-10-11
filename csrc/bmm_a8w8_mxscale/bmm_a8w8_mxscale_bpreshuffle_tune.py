# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune preshuffled-weight MXFP8 BMM with OPUS and FlyDSL on gfx950.

Both backends use the same operands, timing settings and correctness checks.
OPUS candidates come from the default policy or all registered compatible
kernels. FlyDSL candidates combine shipped configurations with the heuristic
choice, filtered by shape compatibility and requested candidate scope.

The output records one backend/kernel selection per shape and scale group.
See README.md for CSV templates and the shared GEMM tuning workflow.
"""

import csv
import os
import sys
from typing import ClassVar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "opus_gemm"))

import opus_bmm_mxscale_tune as opus_tune

from aiter import dtypes

FLYDSL_KERNEL_ID = -1
_SPLITS = (1, 2, 4, 8, 16)
# The preshuffled kids each group's flydsl operands are generated with. Any
# plain-scale preshuffled kid of the group gives the same tensors; these are
# just ones every build has.
_DATA_KID = opus_tune.BMM_DATA_KIDS


# --- mp_tuner hooks (module level so the spawned workers import them) -------
def gen_flydsl_bmm_data(b, m, n, k, seed, out_dtype, group, device="cuda"):
    """Generate shared token-major operands and a preshuffled weight."""
    data = opus_tune.gen_bmm_mxscale_data(
        b, m, n, k, seed, out_dtype, _DATA_KID[group], 1, device=device
    )
    A_mx, _W_mx, Y, A_scale, ws_mx, _ws, ref, W_sh, _xs_sh, _ws_sh = data
    return A_mx, W_sh, A_scale, ws_mx, Y, ref


def run_flydsl_bmm_bench(x, w, x_scale, w_scale, y, kernel_name):
    from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

    return run_bmm_a8w8_mxfp8(x, w, x_scale, w_scale, y, kernel_name=kernel_name)


def _runs(b, n, k, group, cfg):
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import check_bmm_config

    try:
        check_bmm_config(
            n, k, b, **cfg, x_scale_k=group, w_scale_n=group, w_scale_k=group
        )
    except ValueError:
        return False
    return True


class BmmA8W8MxscaleBpreshuffleTuner(opus_tune.OpusBmmMxscaleTuner):
    ARG_DEFAULTS: ClassVar[dict] = {
        **opus_tune.OpusBmmMxscaleTuner.ARG_DEFAULTS,
        "tune_file": "",
        "config_env_name": "AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE",
    }

    def _setup_specific_arguments(self):
        super()._setup_specific_arguments()
        self.parser.add_argument(
            "--libtype",
            default="all",
            help="comma list of backends to tune: opus, flydsl, or all",
        )
        self.parser.add_argument(
            "--opus_candidates",
            choices=("policy", "all"),
            default="policy",
            help="Opus policy candidates, or all registered plain-scale "
            "preshuffled-B kids (off-policy kids use splitK=1)",
        )
        self.parser.add_argument(
            "--flydsl_candidates",
            choices=("near", "all"),
            default="near",
            help="flydsl configs to try per shape: the shipped table's at M "
            "within 2x (near), or every config it ships (all)",
        )

    def pre_process(self, args):
        libs = {s.strip() for s in args.libtype.split(",") if s.strip()}
        self.libs = {"opus", "flydsl"} if "all" in libs else libs
        unknown = self.libs - {"opus", "flydsl"}
        if unknown:
            raise SystemExit(f"--libtype: unknown backend(s) {sorted(unknown)}")
        # Seed flydsl from the table before tuning rewrites it.
        self.fly_seed = {}
        if "flydsl" in self.libs and os.path.exists(opus_tune.BPRESHUFFLE_CSV):
            from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name

            with open(opus_tune.BPRESHUFFLE_CSV) as source:
                seed_rows = list(csv.DictReader(source))
            for r in seed_rows:
                if r["gfx"] != "gfx950" or r["libtype"] != "flydsl":
                    continue
                if parse_bmm_kernel_name(r["kernelName"]) is None:
                    continue
                key = (int(r["b"]), r["w_scale_block"])
                self.fly_seed.setdefault(key, []).append((int(r["m"]), r["kernelName"]))
        args.bpreshuffle = True
        super().pre_process(args)
        if self.libs == {"flydsl"} and self.untunedf["w_scale_block"].eq("1x32").any():
            raise ValueError("1x32 is supported by OPUS; use --libtype opus or all")
        if args.opus_candidates == "all":
            if args.pool != "preb":
                raise ValueError("--opus_candidates all requires --pool preb")
            self.opus_policy = {
                kid: opus_tune._TUNE_POLICY.get(kid, [1])
                for family in opus_tune.a8w8_mxscale_bmm_kernel_lists
                for kid, inst in family.items()
                if inst.needs_preshuffled_b
                and inst.needs_shuffle_scale is None
                and inst.needs_mpacked_sfa is None
            }

    def _flydsl_names(self, b, m, n, k, block, how):
        if block == "1x32":
            return []
        from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
            bmm_kernel_name,
            parse_bmm_kernel_name,
            pick_bmm_kernel_name,
        )

        group = int(block.split("x")[1])
        seed = self.fly_seed.get((b, block), [])
        names = {r for mm, r in seed if how == "all" or m / 2 <= mm <= m * 2}
        try:
            names.add(pick_bmm_kernel_name(b, m, n, k, group, group, group))
        except ValueError:
            pass
        out = set()
        for name in names:
            cfg = parse_bmm_kernel_name(name)
            base = cfg["splits"]
            for sp in _SPLITS:
                if sp != base and not (base / 2 <= sp <= base * 2):
                    continue
                c = {**cfg, "splits": sp}
                if _runs(b, n, k, group, c):
                    out.add(bmm_kernel_name(**c))
        return sorted(out)

    def result_to_df(self, results):
        df = super().result_to_df(results)
        fly = df["kernelId"] == FLYDSL_KERNEL_ID
        df.loc[fly, "libtype"] = "flydsl"
        return df

    def _saved_benchmark(self, row, seed):
        if str(row["libtype"]).lower() == "opus":
            kid = int(row["kernelId"])
            if kid not in opus_tune._CODEGEN_BMM:
                kid = opus_tune.bmm_mxscale_global_kid(kid)
            inst = opus_tune._CODEGEN_BMM.get(kid)
            if inst is None or not inst.needs_preshuffled_b:
                raise ValueError(f"Kid {kid} does not read preshuffled B")
            return super()._saved_benchmark(row, seed)
        if str(row["libtype"]).lower() != "flydsl":
            raise ValueError(f"Unsupported BMM backend {row['libtype']!r}")
        if row["w_scale_block"] == "1x32":
            raise ValueError("FlyDSL BMM does not support 1x32 weight scales")
        from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name

        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        group = int(row["w_scale_block"].split("x")[0])
        name = str(row["kernelName"])
        cfg = parse_bmm_kernel_name(name)
        if cfg is None or not _runs(b, n, k, group, cfg):
            raise ValueError(f"FlyDSL config {name!r} cannot run this shape")
        if int(row["splitK"]) != cfg["splits"]:
            raise ValueError("FlyDSL splitK does not match kernelName")
        data = gen_flydsl_bmm_data(b, m, n, k, seed, dtypes.bf16, group)
        data[4].fill_(float("nan"))
        return run_flydsl_bmm_bench, data[:5] + (name,), data[5]

    def _iter_tuning_tasks(self, row, seed, args):
        if "opus" in self.libs:
            yield from self._iter_opus_tasks(row, seed, args)
        if "flydsl" not in self.libs:
            return
        from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name

        b, m, n, k = (int(row[name]) for name in ("b", "m", "n", "k"))
        block = row["w_scale_block"]
        group = int(block.split("x")[0])
        for name in self._flydsl_names(b, m, n, k, block, args.flydsl_candidates):
            splits = parse_bmm_kernel_name(name)["splits"]
            yield opus_tune.make_bmm_tuning_task(
                (
                    tuple(row[name] for name in self.keys),
                    FLYDSL_KERNEL_ID,
                    splits,
                    name,
                ),
                gen_flydsl_bmm_data,
                (b, m, n, k, seed, dtypes.bf16, group),
                run_flydsl_bmm_bench,
                ((0, 1, 2, 3, 4), name),
                self._perf_kwargs(args, m),
                ref_index=5,
                output_index=4,
            )


if __name__ == "__main__":
    tuner = BmmA8W8MxscaleBpreshuffleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
