# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Backend tuner for ``aiter.mqa_logits`` (FP8 MQA logits over a contiguous KV).

Times the Triton/Gluon kernel and every FlyDSL variant the shape admits through
the same launcher ``aiter.mqa_logits`` uses, grades each against a torch
reference, and writes the fastest correct one per shape::

    python3 csrc/mqa_logits/mqa_logits_tune.py \
        -i aiter/configs/model_configs/glm5_untuned_mqa_logits.csv \
        -o aiter/configs/model_configs/glm5_tuned_mqa_logits.csv

``--run_config <tuned.csv>`` times the production op on every tuned row.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch

import aiter.mqa_logits as mql
from aiter import logger
from aiter.jit.core import AITER_CONFIG_MQA_LOGITS, AITER_ROOT_DIR
from aiter.test_common import run_perftest
from aiter.utility.base_tuner import TunerCommon

LOOKUP_KEYS = ["gfx", "cu_num", "heads", "head_dim", "seq_len", "seq_len_kv"]
RESULT_COLS = ["backend", "variant", "us", "triton_us", "flydsl_us"]
MAX_CALC_DIFF = 1e-3
RUN_CONFIG_TOL_PCT = 10.0
PICK_FINALISTS = 3
PICK_RETIMES = 3
_OP_TEST_PATH = Path(AITER_ROOT_DIR) / "op_tests" / "test_mqa_logits.py"


def _load_op_test():
    """Input builder and torch reference shared with the op test."""
    spec = importlib.util.spec_from_file_location("_mqa_logits_test", _OP_TEST_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_OP_TEST = _load_op_test()
build_inputs = _OP_TEST.build_inputs
reference = _OP_TEST.run_torch
_launch_args = _OP_TEST.launch_args


def _shape(row) -> tuple:
    return (
        int(row["seq_len"]),
        int(row["seq_len_kv"]),
        int(row["heads"]),
        int(row["head_dim"]),
    )


def _config_name(config) -> str:
    if config["backend"] == "flydsl":
        return f"flydsl {config['variant']}"
    return "triton"


def grade(ref: torch.Tensor, got: torch.Tensor) -> str | None:
    """None if ``got`` matches ``ref``, else the reason it does not."""
    ref_inf, got_inf = torch.isneginf(ref), torch.isneginf(got)
    if not torch.equal(ref_inf, got_inf):
        return "-inf mask mismatch"
    diff = _OP_TEST.calc_diff(ref.masked_fill(ref_inf, 0), got.masked_fill(got_inf, 0))
    if not diff < MAX_CALC_DIFF:
        return f"calc_diff {diff:.3g} >= {MAX_CALC_DIFF}"
    return None


def candidates(inp) -> list[dict]:
    configs = [dict(mql.DEFAULT_CONFIG)]
    mod = mql._flydsl_kernels()
    if mod is not None:
        configs += [
            {"backend": "flydsl", "variant": v}
            for v in mod.KERNEL_VARIANTS
            if mql.flydsl_supports(inp["Q"], inp["KV"], v)
        ]
    return configs


def time_us(fn, *args, warmup, iters) -> float:
    with torch.inference_mode():
        _, us = run_perftest(fn, *args, num_warmup=warmup, num_iters=iters)
    return float(us)


class MqaLogitsTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/untuned_mqa_logits.csv",
        "tune_file": AITER_CONFIG_MQA_LOGITS,
        "config_env_name": "AITER_CONFIG_MQA_LOGITS",
    }

    def __init__(self):
        super().__init__(
            "tuned_mqa_logits",
            LOOKUP_KEYS,
            RESULT_COLS,
            "aiter.mqa_logits backend tuner",
        )

    def _clear_op_caches(self):
        mql.reload_tuned_table()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        self._clear_op_caches()

    def pre_process(self, args):
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        untunedf = self.get_untuned_gemm_list(args.untune_file)
        untunedf["gfx"], untunedf["cu_num"] = gfx, cu_num
        self.untunedf = untunedf.drop_duplicates(subset=LOOKUP_KEYS).reset_index(
            drop=True
        )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if args.all or self.tunedf.empty:
            return
        tuned = {
            (str(r["gfx"]), int(r["cu_num"]), *_shape(r))
            for _, r in self.tunedf.iterrows()
        }
        keep = [
            (gfx, cu_num, *_shape(r)) not in tuned for _, r in self.untunedf.iterrows()
        ]
        self.untunedf = self.untunedf[keep].reset_index(drop=True)

    def measure(self, config, inp, ref, args):
        """``(us, None)`` for a correct candidate, else ``(None, reason)``."""
        launch = _launch_args(inp)
        seq_len_kv = inp["KV"].shape[0]
        try:
            out = mql.run_mqa_logits(config, *launch)
            reason = grade(ref, out[:, :seq_len_kv])
            del out
            if reason is not None:
                return None, reason
            us = time_us(
                mql.run_mqa_logits,
                config,
                *launch,
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as exc:  # noqa: BLE001  a candidate may not compile
            return None, f"{type(exc).__name__}: {exc}"
        return us, None

    def tune_shape(self, row, args) -> dict | None:
        """Pick the fastest correct config for one shape."""
        seq_len, seq_len_kv, heads, head_dim = _shape(row)
        inp = build_inputs(seq_len, seq_len_kv, heads, head_dim)
        ref = reference(inp)
        times: dict[str, float] = {}
        configs: dict[str, dict] = {}
        for config in candidates(inp):
            name = _config_name(config)
            us, reason = self.measure(config, inp, ref, args)
            if reason is not None:
                logger.warning(f"{_shape(row)} {name} rejected: {reason}")
                continue
            if args.verbose:
                print(f"{_shape(row)} {name} {us:.2f} us", flush=True)
            configs[name], times[name] = config, us
        if not times:
            return None
        # Single timings are noisy and their minimum is biased low, so the
        # closest candidates are re-timed interleaved and the best median kept.
        finalists = sorted(times, key=times.get)[:PICK_FINALISTS]
        retimed = {name: [] for name in finalists}
        for _ in range(PICK_RETIMES):
            for name in finalists:
                retimed[name].append(
                    time_us(
                        mql.run_mqa_logits,
                        configs[name],
                        *_launch_args(inp),
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                )
        medians = {name: statistics.median(ts) for name, ts in retimed.items()}
        pick = min(medians, key=medians.get)
        pick_us = medians[pick]
        del inp, ref
        torch.cuda.empty_cache()
        best = {
            b: min(
                (t for n, t in times.items() if configs[n]["backend"] == b),
                default=None,
            )
            for b in mql.BACKENDS
        }
        print(
            f"{_shape(row)}: {pick} {pick_us:.2f} us "
            f"(triton {best['triton']}, best flydsl {best['flydsl']})",
            flush=True,
        )
        return {
            **{k: row[k] for k in LOOKUP_KEYS},
            **configs[pick],
            "us": round(pick_us, 2),
            "triton_us": pd.NA if best["triton"] is None else round(best["triton"], 2),
            "flydsl_us": pd.NA if best["flydsl"] is None else round(best["flydsl"], 2),
        }

    def tune(self, untunedf, tunedf, args):
        rows = []
        for _, row in untunedf.iterrows():
            result = self.tune_shape(row, args)
            if result is None:
                self.failed = pd.concat(
                    [self.failed, pd.DataFrame([row.to_dict()])], ignore_index=True
                )
            else:
                rows.append(result)
        return rows

    def post_process(self, results, args, topk=-1, fast_mode=False):
        return pd.DataFrame(results, columns=self.columns)

    def result_to_csv(self, results, file, concat=False):
        old = self.get_tuned_gemm_list(file)
        merged = self.update_tunedf(old, results.loc[:, self.columns])
        merged = merged.drop_duplicates(subset=self.keys, keep="last")
        merged.to_csv(file, index=False)
        self.success = pd.concat([self.success, results], ignore_index=True)

    def run_config(self, args):
        results = []
        for _, row in self.untunedf.iterrows():
            seq_len, seq_len_kv, heads, head_dim = _shape(row)
            label = f"({seq_len}, {seq_len_kv}, {heads}, {head_dim})"
            try:
                inp = build_inputs(seq_len, seq_len_kv, heads, head_dim)
                us = time_us(
                    mql.mqa_logits,
                    *_launch_args(inp),
                    warmup=args.warmup,
                    iters=args.iters,
                )
                del inp
                csv_us = float(row.get("us", 0) or 0)
                drift = (us - csv_us) / csv_us * 100 if csv_us > 0 else 0.0
                status = (
                    "ok"
                    if abs(drift) <= RUN_CONFIG_TOL_PCT
                    else f"mismatch: {drift:+.1f}% vs tuned {csv_us:.2f} us"
                )
            except Exception as exc:  # noqa: BLE001
                us, status = -1.0, f"error: {exc}"
            results.append(
                {"shape": label, "e2e_us": us, "kernel_us": us, "status": status}
            )
            torch.cuda.empty_cache()
        return results

    def tune_summary(self, status):
        logger.info(
            f"Tuning {status}: {len(self.success)} shapes tuned, "
            f"{len(self.failed)} without a correct candidate"
        )
        if not self.success.empty:
            print(self.success.to_string(index=False), flush=True)
        if not self.failed.empty:
            print(self.failed.to_string(index=False), flush=True)
            sys.exit(1)

    def result_to_df(self, results):
        return pd.DataFrame(results, columns=self.columns)


if __name__ == "__main__":
    tuner = MqaLogitsTuner()
    tuner.run(tuner.parse_args())
