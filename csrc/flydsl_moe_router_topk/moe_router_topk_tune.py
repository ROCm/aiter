# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner for ``router_topk_config``: per shape, the tuned router GEMM (tuned into
``--gemm_tuned_file`` first if missing) + ``topk_gating`` against the fused
launch at each split_k of the FlyDSL GEMM catalog and prefetch depth, interleaved
with an A/A band. ``fused`` = the best (the GEMM's own split_k, ``bitexact``,
unless beaten by more than the band and ``--min_improvement_pct``) beats the
unfused path by more than the band."""

import functools
import statistics
import subprocess
import sys
import tempfile
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import logger, tuned_gemm
from aiter.jit.core import AITER_CONFIG_MOE_ROUTER_TOPK, AITER_CONFIGS, AITER_ROOT_DIR
from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_configs
from aiter.ops.flydsl.moe_front_gfx1250 import flydsl_moe_router_topk as fused_topk
from aiter.ops.flydsl.moe_front_gfx1250 import router_gemm_split_k
from aiter.ops.topk import topk_gating
from aiter.utility.base_tuner import TunerCommon

KEYS = ["gfx", "cu_num", "M", "N", "K", "topk"]
CAPTURES = 3
PREFETCH = (2, 4, 8, 16)
BF16 = str(torch.bfloat16)


def _unfused(x, w, bias, nvr, topk):
    """``tgemm.mm`` + ``topk_gating``, the path the fused launch replaces."""
    logits = tuned_gemm.tgemm.mm(x, w, None, otype=torch.bfloat16)
    tw = torch.empty(x.shape[0], topk, dtype=torch.float32, device="cuda")
    ti = torch.empty(x.shape[0], topk, dtype=torch.int32, device="cuda")
    topk_gating(
        tw, ti, logits, bias, True, 1.0, score_func="sigmoid", num_valid_rows=nvr
    )
    return tw, ti


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


def _matches(rw, ri, tw, ti):
    """Routes equal on >= 99% of rows, with weights there within 1e-2."""
    same = (ri == ti).all(dim=1)
    return same.float().mean() >= 0.99 and torch.allclose(rw[same], tw[same], atol=1e-2)


def _candidates(M, N, K, gemm_split_k):
    catalog = get_flydsl_a16w16_configs(M, N, K, torch.bfloat16, torch.bfloat16, 0)
    splits = {c["split_k"] for c in catalog if not c.get("split_k_atomic")}
    splits = {s for s in splits | {gemm_split_k} - {None} if not K % (s * 32)}
    steps = {s: K // (s * 32) for s in splits if s <= 16}
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
            ["fused", "split_k", "prefetch", "bitexact", "us", "unfused_us"],
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
        router_gemm_split_k.cache_clear()

    def tune_shape(self, M, N, K, topk, min_gain_pct):
        g = torch.Generator(device="cuda").manual_seed(M)
        x = torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16)
        w = (torch.randn(N, K, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        bias = torch.randn(N, generator=g, device="cuda") * 0.05
        nvr = torch.tensor([M], dtype=torch.int32, device="cuda")
        rw, ri = _unfused(x, w, bias, nvr, topk)
        gemm_split_k = router_gemm_split_k(M, N, K)
        fused = {}
        for sk, pf in _candidates(M, N, K, gemm_split_k):
            tw, ti = fused_topk(x, w, bias, topk, sk, num_valid_rows=nvr, prefetch=pf)
            if sk == gemm_split_k:
                assert torch.equal(rw.view(torch.int32), tw.view(torch.int32))
                assert torch.equal(ri, ti)
            elif not _matches(rw, ri, tw, ti):
                logger.warning(f"M={M} K={K} split_k={sk}: routes differ, skipped")
                continue
            out = {"num_valid_rows": nvr, "topk_weights": tw, "topk_ids": ti}
            fused[sk, pf] = functools.partial(
                fused_topk, x, w, bias, topk, sk, prefetch=pf, **out
            )

        unfused = functools.partial(_unfused, x, w, bias, nvr, topk)
        arms = [unfused, *fused.values(), unfused]
        runs = [_interleaved_us(arms) for _ in range(CAPTURES)]
        base = statistics.median(t for r in runs for t in (r[0], r[-1]))
        band = max(abs(r[0] - r[-1]) for r in runs)
        us = {c: statistics.median(r[i] for r in runs) for i, c in enumerate(fused, 1)}
        best = min(us, key=us.get)
        tol = max(band, us[best] * min_gain_pct / 100)
        own = [c for c in us if c[0] == gemm_split_k and us[c] - us[best] <= tol]
        best = min(own, key=us.get) if own else best
        logger.info(
            f"M={M} N={N} K={K} topk={topk}: unfused {base:.2f} us, A/A {band:.3f}, "
            f"fused {', '.join(f'{s}/{p}:{t:.2f}' for (s, p), t in us.items())}"
        )
        return {
            "fused": int(base - us[best] > band),
            "split_k": best[0],
            "prefetch": best[1],
            "bitexact": int(best[0] == gemm_split_k),
            "us": round(us[best], 2),
            "unfused_us": round(base, 2),
        }

    def tune(self, untunedf, tunedf, args):
        return [
            {**r._asdict(), **self.tune_shape(*r[2:], args.min_improvement_pct)}
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
