# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Backend tuner for ``aiter.paged_mqa_logits`` (FP8 paged MQA logits, decode).

Times every Gluon and FlyDSL candidate the shape admits through the same
launcher ``aiter.paged_mqa_logits`` uses, grades each against a torch
reference, and writes the fastest correct one per shape::

    python3 csrc/paged_mqa_logits/paged_mqa_logits_tune.py \
        -i aiter/configs/model_configs/paged_mqa_logits_untuned_glm5.csv \
        -o aiter/configs/model_configs/paged_mqa_logits_tuned_glm5.csv

``--run_config <tuned.csv>`` times the production op on every tuned row.

Context length is not a lookup key (the op cannot read it without a host
sync), so a shape listed at several ``context_len`` values in the untuned file
gets the config with the lowest geomean time across them. The tuned row records
the contexts (``8192;32768``) and that geomean.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch

import aiter.paged_mqa_logits as pmql
from aiter import logger
from aiter.jit.core import AITER_CONFIG_PAGED_MQA_LOGITS, AITER_ROOT_DIR
from aiter.test_common import run_perftest
from aiter.utility.base_tuner import TunerCommon

LOOKUP_KEYS = [
    "gfx",
    "cu_num",
    "batch_size",
    "next_n",
    "heads",
    "head_dim",
    "kv_block_size",
    "preshuffle",
]
RESULT_COLS = [
    "context_len",
    "backend",
    "ChunkK",
    "WavePerEU",
    "wg_per_cu",
    "us",
    "gluon_us",
    "flydsl_us",
]
GLUON_CHUNK_K = (128, 256)
GLUON_WAVE_PER_EU = (1, 2, 4)
FLYDSL_WG_PER_CU = (1, 2, 3, 4)
DEFAULT_MAX_MODEL_LEN = 131072
MAX_CALC_DIFF = 1e-3
RUN_CONFIG_TOL_PCT = 10.0
CONTEXT_SEP = ";"
_OP_TEST_PATH = Path(AITER_ROOT_DIR) / "op_tests" / "test_paged_mqa_logits.py"


def _load_op_test():
    """Input builder and torch reference shared with the op test."""
    spec = importlib.util.spec_from_file_location(
        "_paged_mqa_logits_test", _OP_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_OP_TEST = _load_op_test()
build_inputs = _OP_TEST.build_inputs
reference = _OP_TEST.run_torch
_launch_args = _OP_TEST.launch_args


def _bool(value) -> bool:
    return str(value).strip() in ("True", "true", "1")


def _contexts(value) -> list[int]:
    return [int(float(c)) for c in str(value).split(CONTEXT_SEP) if c.strip()]


def _join_contexts(contexts) -> str:
    return CONTEXT_SEP.join(str(c) for c in sorted(set(contexts)))


def _geomean(values) -> float:
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _config_name(config) -> str:
    if config["backend"] == "flydsl":
        return f"flydsl wg_per_cu={config['wg_per_cu']}"
    return f"gluon ChunkK={config['ChunkK']} WavePerEU={config['WavePerEU']}"


def _shape(row) -> tuple:
    return (
        int(row["batch_size"]),
        int(row["next_n"]),
        int(row["heads"]),
        int(row["head_dim"]),
        int(row["kv_block_size"]),
        _bool(row["preshuffle"]),
    )


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
    configs = [
        {"backend": "gluon", "ChunkK": c, "WavePerEU": w, "wg_per_cu": 0}
        for c, w in itertools.product(GLUON_CHUNK_K, GLUON_WAVE_PER_EU)
        if c % inp["KVBlockSize"] == 0 or inp["KVBlockSize"] % c == 0
    ]
    if pmql.flydsl_supports(
        inp["q_fp8"],
        inp["kv_cache"],
        inp["weights"],
        inp["Preshuffle"],
        inp["KVBlockSize"],
    ):
        configs += [
            {"backend": "flydsl", "ChunkK": 0, "WavePerEU": 0, "wg_per_cu": w}
            for w in FLYDSL_WG_PER_CU
        ]
    return configs


def time_us(fn, *args, warmup, iters) -> float:
    with torch.inference_mode():
        _, us = run_perftest(fn, *args, num_warmup=warmup, num_iters=iters)
    return float(us)


class PagedMqaLogitsTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/paged_mqa_logits_untuned.csv",
        "tune_file": AITER_CONFIG_PAGED_MQA_LOGITS,
        "config_env_name": "AITER_CONFIG_PAGED_MQA_LOGITS",
    }

    def __init__(self):
        super().__init__(
            "paged_mqa_logits_tuned",
            LOOKUP_KEYS,
            RESULT_COLS,
            "aiter.paged_mqa_logits backend tuner",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--max_model_len",
            type=int,
            default=DEFAULT_MAX_MODEL_LEN,
            help="Logits row width the shapes are timed with",
        )

    def _clear_op_caches(self):
        pmql.reload_tuned_table()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        self._clear_op_caches()

    def pre_process(self, args):
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        untunedf = self.get_untuned_gemm_list(args.untune_file)
        untunedf["gfx"], untunedf["cu_num"] = gfx, cu_num
        self.untunedf = (
            untunedf.groupby(LOOKUP_KEYS, sort=False)["context_len"]
            .agg(lambda s: _join_contexts(c for v in s for c in _contexts(v)))
            .reset_index()
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

    def measure(self, config, inp, ref, context_len, args):
        """``(us, None)`` for a correct candidate, else ``(None, reason)``."""
        launch = _launch_args(inp)
        try:
            inp["out_logits"].fill_(float("-inf"))
            pmql.run_paged_mqa_logits(config, *launch)
            reason = grade(ref, inp["out_logits"][:, :context_len])
            if reason is not None:
                return None, reason
            us = time_us(
                pmql.run_paged_mqa_logits,
                config,
                *launch,
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as exc:  # noqa: BLE001  a candidate may not compile
            return None, f"{type(exc).__name__}: {exc}"
        return us, None

    def tune_shape(self, row, args) -> dict | None:
        """Pick the config with the lowest geomean time over the row's contexts."""
        shape, contexts = _shape(row), _contexts(row["context_len"])
        configs: dict[str, dict] = {}
        times: dict[str, dict[int, float]] = {}
        for context_len in contexts:
            inp = build_inputs(shape, context_len, max(args.max_model_len, context_len))
            ref = reference(inp, context_len)
            for config in candidates(inp):
                name = _config_name(config)
                us, reason = self.measure(config, inp, ref, context_len, args)
                if reason is not None:
                    logger.warning(
                        f"{shape} ctx={context_len} {name} rejected: {reason}"
                    )
                    continue
                if args.verbose:
                    print(f"{shape} ctx={context_len} {name} {us:.2f} us", flush=True)
                configs[name] = config
                times.setdefault(name, {})[context_len] = us
            del inp, ref
            torch.cuda.empty_cache()

        scores = {
            name: _geomean(t.values())
            for name, t in times.items()
            if len(t) == len(contexts)
        }
        if not scores:
            return None
        best = {}
        for name, us in scores.items():
            backend = configs[name]["backend"]
            if backend not in best or us < scores[best[backend]]:
                best[backend] = name
        pick = min(best.values(), key=scores.get)
        per_backend = {b: round(scores[n], 2) for b, n in best.items()}
        print(
            f"{shape} ctx={row['context_len']}: {pick} {scores[pick]:.2f} us "
            f"(best per backend {per_backend})",
            flush=True,
        )
        return {
            **{k: row[k] for k in LOOKUP_KEYS},
            "context_len": row["context_len"],
            **configs[pick],
            "us": round(scores[pick], 2),
            "gluon_us": per_backend.get("gluon", pd.NA),
            "flydsl_us": per_backend.get("flydsl", pd.NA),
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
            shape = _shape(row)
            label = f"{shape} ctx={row['context_len']}"
            try:
                per_context = []
                for context_len in _contexts(row["context_len"]):
                    inp = build_inputs(
                        shape, context_len, max(args.max_model_len, context_len)
                    )
                    per_context.append(
                        time_us(
                            pmql.paged_mqa_logits,
                            *_launch_args(inp),
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                    )
                    del inp
                us = _geomean(per_context)
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
    tuner = PagedMqaLogitsTuner()
    tuner.run(tuner.parse_args())
