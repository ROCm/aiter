# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner for ``flydsl_moe_router_topk``: per shape, the tuned router GEMM (tuned
into ``--gemm_tuned_file`` first if missing) + ``topk_gating`` against the fused
launch at each split_k of ``SPLIT_KS`` that divides K into 32-wide steps, and
each prefetch depth, in interleaved CUDA graphs with an A/A band. Candidates
must match the op test's fp32 reference (``check_routes``). ``fused`` = the
fastest beats the unfused path by more than the band."""

import importlib.util
import statistics
import subprocess
import sys
import tempfile
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import logger, tuned_gemm
from aiter.jit.core import AITER_CONFIG_MOE_ROUTER_TOPK, AITER_CONFIGS, AITER_ROOT_DIR
from aiter.ops.flydsl.moe_front_gfx1250 import flydsl_moe_router_topk as fused_topk
from aiter.utility.base_tuner import TunerCommon

KEYS = ["gfx", "cu_num", "M", "N", "K", "topk"]
CAPTURES = 3
SPLIT_KS = (1, 2, 4, 8, 16)  # the cluster holds 16 workgroups
PREFETCH = (2, 4, 8, 16)
BF16 = str(torch.bfloat16)


def _load_op_test():
    """Inputs, fp32 reference, route check and unfused path of the op test."""
    path = f"{AITER_ROOT_DIR}/op_tests/test_flydsl_moe_front.py"
    spec = importlib.util.spec_from_file_location("_moe_front_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_T = _load_op_test()


def _interleaved_us(fns, reps=10, replays=20, rounds=50):
    """Per-call us of each of ``fns``: a CUDA graph of ``reps`` calls each,
    replayed in turn (rotating the order) for ``rounds`` rounds; medians."""
    graphs = []
    for fn in fns:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        graphs.append(g)
    torch.cuda.synchronize()
    times = [[] for _ in fns]
    for r in range(rounds):
        for i in range(len(fns)):
            j = (i + r) % len(fns)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(replays):
                graphs[j].replay()
            b.record()
            b.synchronize()
            times[j].append(a.elapsed_time(b) * 1000 / (replays * reps))
    return [statistics.median(t) for t in times]


def _candidates(K):
    steps = {s: K // (s * 32) for s in SPLIT_KS if not K % (s * 32)}
    return sorted({(s, min(p, n)) for s, n in steps.items() for p in PREFETCH})


class MoeRouterTopkTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/moe_router_topk_untuned.csv",
        "tune_file": AITER_CONFIG_MOE_ROUTER_TOPK,
        "config_env_name": "AITER_CONFIG_MOE_ROUTER_TOPK",
    }

    def __init__(self):
        super().__init__(
            "moe_router_topk_tuned",
            KEYS,
            ["fused", "split_k", "prefetch", "us", "unfused_us"],
            "gfx1250 fused MoE router top-k tuner",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--gemm_tuned_file",
            default=f"{AITER_ROOT_DIR}/aiter/configs/model_configs/"
            "moe_router_bf16_tuned_gemm.csv",
            help="where router GEMM shapes missing from the GEMM tables are tuned to",
        )

    def pre_process(self, args):
        untunedf = self.get_untuned_gemm_list(args.untune_file)
        untunedf["gfx"], untunedf["cu_num"] = self.get_gfx(), self.get_cu_num()
        self.untunedf = untunedf[KEYS]
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        gemms = self.untunedf[["M", "N", "K"]].drop_duplicates()
        missing = gemms[
            [
                tuned_gemm.get_GEMM_A16W16_tuned_config(m, n, k, False, BF16, BF16)
                is None
                for m, n, k in gemms.itertuples(index=False)
            ]
        ]
        if missing.empty:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            missing.assign(
                bias=False, dtype=BF16, outdtype=BF16, scaleAB=False, bpreshuffle=False
            ).to_csv(f, index=False)
        tuner = f"{AITER_ROOT_DIR}/csrc/gemm_a16w16/gemm_a16w16_tune.py"
        cmd = [sys.executable, tuner, "-i", f.name, "-o", args.gemm_tuned_file]
        subprocess.run(cmd, check=True)
        type(AITER_CONFIGS).get_config_file.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config.cache_clear()

    def tune_shape(self, M, N, K, topk):
        torch.manual_seed(M)
        x, w, bias, _ = _T.make_inputs(M, N, K, "rand")
        nvr = torch.tensor([M], dtype=torch.int32, device="cuda")  # as callers pass it
        ref = _T.run_torch(x, w, topk, bias)
        fused = {}
        for sk, pf in _candidates(K):
            cfg = {"num_valid_rows": nvr, "split_k": sk, "prefetch": pf}
            tw, ti = fused_topk(x, w, topk, bias, **cfg)
            if _T.check_routes(ref, tw, ti, nvr):
                logger.warning(f"M={M} K={K} split_k={sk}: routes differ, skipped")
                continue
            fused[sk, pf] = lambda cfg=cfg: fused_topk(x, w, topk, bias, **cfg)

        def unfused():
            _T.run_unfused(x, w, topk, bias, nvr)

        arms = [unfused, *fused.values(), unfused]
        runs = [_interleaved_us(arms) for _ in range(CAPTURES)]
        base = statistics.median(t for r in runs for t in (r[0], r[-1]))
        band = max(abs(r[0] - r[-1]) for r in runs)
        us = {c: statistics.median(r[i] for r in runs) for i, c in enumerate(fused, 1)}
        best = min(us, key=us.get)
        logger.info(
            f"M={M} N={N} K={K} topk={topk}: unfused {base:.2f} us, A/A {band:.3f}, "
            f"fused {', '.join(f'{s}/{p}:{t:.2f}' for (s, p), t in us.items())}"
        )
        return {
            "fused": int(base - us[best] > band),
            "split_k": best[0],
            "prefetch": best[1],
            "us": round(us[best], 2),
            "unfused_us": round(base, 2),
        }

    def tune(self, untunedf, tunedf, args):
        return [
            {**r._asdict(), **self.tune_shape(*r[2:])}
            for r in untunedf.itertuples(index=False)
        ]

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        return pd.DataFrame(rets, columns=self.columns)

    def result_to_csv(self, results, file, concat=False):
        old = self.get_tuned_gemm_list(file)
        tuned = pd.MultiIndex.from_frame(results[KEYS])
        old = old[~pd.MultiIndex.from_frame(old[KEYS]).isin(tuned)]
        pd.concat([old, results]).to_csv(file, index=False)
        self.success = pd.concat([self.success, results], ignore_index=True)


if __name__ == "__main__":
    tuner = MoeRouterTopkTuner()
    tuner.run(tuner.parse_args())
