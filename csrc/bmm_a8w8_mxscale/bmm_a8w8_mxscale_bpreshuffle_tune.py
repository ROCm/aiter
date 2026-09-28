# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the gfx950 preshuffled-B MXFP8 BMM across opus and flydsl in one sweep.

    python3 csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py \\
        --libtype all --groupSize 128,32 -g 2,8 -m 1,16,1024 --mp 8

Both backends read the same operands -- a (16, 16)-preshuffled weight and
row-major e8m0 scales -- so one shape's candidates from both run in the same
mp_tuner pass, under the same timing mode, and the fastest names the row's
libtype. That is the only way the two can be ranked: their own tuners time
differently, and a row picked across two measurement modes is a coin flip at
the few-percent gaps that separate them.

Candidates:
  * opus: every preshuffled-B kid of the row's group size the launch plan
    accepts (the opus tuner's pool "preb").
  * flydsl: the configs the preshuffle table already ships for this (batch,
    w_scale block) at M within a factor of two, each at its own split and the
    neighbouring powers of two, plus the heuristic's pick for the shape;
    ``--flydsl_candidates all`` takes every config the table ships at any M.
    A config is kept only where check_bmm_config accepts the shape.

Output schema is the preshuffle table's:
    gfx,b,m,n,k,w_scale_block,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
with kernelId -1 and kernelName the config on flydsl rows.
"""

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "opus_gemm"))

import opus_bmm_mxscale_tune as opus_tune  # noqa: E402
from aiter import dtypes, logger  # noqa: E402
from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8  # noqa: E402
from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (  # noqa: E402
    bmm_kernel_name,
    check_bmm_config,
    parse_bmm_kernel_name,
    pick_bmm_kernel_name,
)
from aiter.utility.mp_tuner import mp_tuner  # noqa: E402

FLYDSL_KERNEL_ID = -1
_SPLITS = (1, 2, 4, 8, 16)
# The preshuffled kids each group's flydsl operands are generated with. Any
# plain-scale preshuffled kid of the group gives the same tensors; these are
# just ones every build has.
_DATA_KID = {128: 8179, 32: 9179}


# --- mp_tuner hooks (module level so the spawned workers import them) -------
def gen_flydsl_bmm_data(b, m, n, k, seed, out_dtype, group, device="cuda"):
    """(x, w, x_scale, w_scale, y, ref): the opus tuner's operands, laid out as
    the flydsl entry takes them -- token-major and contiguous."""
    data = opus_tune.gen_bmm_mxscale_data(
        b, m, n, k, seed, out_dtype, _DATA_KID[group], 1, device=device
    )
    O_mx, _W_mx, Y, xs_mx, ws_mx, _ws, ref, W_sh, _xs_sh, _ws_sh = data
    x = O_mx.transpose(0, 1).contiguous()
    x_scale = xs_mx.transpose(0, 1).contiguous()
    return x, W_sh, x_scale, ws_mx, Y, ref


def run_flydsl_bmm_bench(x, w, x_scale, w_scale, y, kernel_name):
    return run_bmm_a8w8_mxfp8(x, w, x_scale, w_scale, y, kernel_name=kernel_name)


def _runs(b, n, k, group, cfg):
    try:
        check_bmm_config(
            n, k, b, **cfg, x_scale_k=group, w_scale_n=group, w_scale_k=group
        )
    except ValueError:
        return False
    return True


class BmmA8W8MxscaleBpreshuffleTuner(opus_tune.OpusBmmMxscaleTuner):
    ARG_DEFAULTS = {
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
            "--flydsl_candidates",
            choices=("near", "all"),
            default="near",
            help="flydsl configs to try per shape: the shipped table's at M "
            "within 2x (near), or every config it ships (all)",
        )

    def _shapes_from_shipped(self):
        return sorted(
            {
                (int(r["b"]), int(r["m"]), int(r["n"]), int(r["k"]))
                for r in csv.DictReader(open(opus_tune.BPRESHUFFLE_CSV))
                if r["gfx"] == "gfx950"
            }
        )

    def pre_process(self, args):
        libs = {s.strip() for s in args.libtype.split(",") if s.strip()}
        self.libs = {"opus", "flydsl"} if "all" in libs else libs
        unknown = self.libs - {"opus", "flydsl"}
        if unknown:
            raise SystemExit(f"--libtype: unknown backend(s) {sorted(unknown)}")
        # Seed flydsl from the table before tuning rewrites it.
        self.fly_seed = {}
        if os.path.exists(opus_tune.BPRESHUFFLE_CSV):
            for r in csv.DictReader(open(opus_tune.BPRESHUFFLE_CSV)):
                if r["gfx"] != "gfx950" or r["libtype"] != "flydsl":
                    continue
                if parse_bmm_kernel_name(r["kernelName"]) is None:
                    continue
                key = (int(r["b"]), r["w_scale_block"])
                self.fly_seed.setdefault(key, []).append((int(r["m"]), r["kernelName"]))
        args.bpreshuffle = True
        super().pre_process(args)

    def _flydsl_names(self, b, m, n, k, block, how):
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

    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        out_dtype = dtypes.bf16
        base_perf_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        graph_m_max = int(getattr(args, "graph_m_max", 0) or 0)

        task, tasks_data = [], []
        for seed, i in enumerate(range(len(untunedf)), start=1):
            b, m, n, k = (int(untunedf.loc[i, c]) for c in ("b", "m", "n", "k"))
            block = str(untunedf.loc[i, "w_scale_block"])
            group = int(block.split("x")[1])
            info_keys = (gfx, b, m, n, k, block)
            # One timing mode per shape, for every candidate of both backends.
            perf_kwargs = (
                {**base_perf_kwargs, "testGraph": True}
                if graph_m_max and m <= graph_m_max
                else base_perf_kwargs
            )
            n_cand = 0
            if "opus" in self.libs:
                for kid in opus_tune._CANDIDATE_KIDS:
                    if opus_tune._kid_group(kid) != group:
                        continue
                    for sk in opus_tune._applicable(kid, b, m, n, k, args.pool):
                        task.append(
                            (
                                (info_keys, kid, sk, ""),
                                opus_tune.gen_bmm_mxscale_data,
                                (b, m, n, k, seed, out_dtype, kid, sk),
                                opus_tune.run_bmm_mxscale_bench,
                                ([0, 1, 2, 3, 4, 5, 7, 8, 9], kid, sk),
                                perf_kwargs,
                                opus_tune._bmm_ref_passthrough,
                                ([6],),
                                {},
                                None,
                                1e-2,
                                1e-2,
                                None,
                                None,
                                [2],
                            )
                        )
                        n_cand += 1
            if "flydsl" in self.libs:
                for name in self._flydsl_names(b, m, n, k, block, args.flydsl_candidates):
                    splits = parse_bmm_kernel_name(name)["splits"]
                    task.append(
                        (
                            (info_keys, FLYDSL_KERNEL_ID, splits, name),
                            gen_flydsl_bmm_data,
                            (b, m, n, k, seed, out_dtype, group),
                            run_flydsl_bmm_bench,
                            ([0, 1, 2, 3, 4], name),
                            perf_kwargs,
                            opus_tune._bmm_ref_passthrough,
                            ([5],),
                            {},
                            None,
                            1e-2,
                            1e-2,
                            None,
                            None,
                            [4],
                        )
                    )
                    n_cand += 1
            if args.verbose:
                logger.info("B:%s M:%s %s: %d candidates", b, m, block, n_cand)
            tasks_data.append((n_cand, ()))

        if not task:
            return []
        return mp_tuner(
            task,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = BmmA8W8MxscaleBpreshuffleTuner()
    _args = tuner.parse_args()
    tuner.run(_args, False)
