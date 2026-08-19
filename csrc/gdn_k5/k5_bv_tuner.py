# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TunerCommon wrapper for FlyDSL K5 mfma16_hip BV tuning."""

from __future__ import annotations

import sys
import time
from typing import Any, ClassVar

import pandas as pd
import torch
from k5_bv_tune_lib import (
    BV_CANDIDATES,
    LOOKUP_KEYS,
    TUNED_COLUMNS,
    bench_us,
    build_k5_inputs,
    case_snapshot_dtype,
    chunk_counts,
    dataframe_from_cases,
    find_case_for_row,
    load_k5_cases,
    lookup_key_from_case,
    read_csv_rows,
    select_cases,
    sweep_case_row,
)

from aiter import logger
from aiter.jit.core import AITER_ROOT_DIR
from aiter.utility.base_tuner import TunerCommon

_DEFAULT_UNTUNED = f"{AITER_ROOT_DIR}/aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv"
_DEFAULT_TUNED = f"{AITER_ROOT_DIR}/aiter/configs/chunk_gdn_h_mfma16_hip_tuned.csv"
_RESULT_COLS = [c for c in TUNED_COLUMNS if c not in LOOKUP_KEYS]


class K5BvTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": _DEFAULT_UNTUNED,
        "tune_file": _DEFAULT_TUNED,
        "config_env_name": "AITER_CONFIG_GDN_K5_MFMA16_HIP",
        "warmup": 5,
        "iters": 20,
        "batch": 100,
        "sort": False,
    }

    def __init__(self):
        super().__init__(
            "chunk_gdn_h_mfma16_hip_tuned",
            list(LOOKUP_KEYS),
            _RESULT_COLS,
            "FlyDSL K5 mfma16_hip BV tuner",
        )
        self._cases: list[tuple[str, Any]] = []
        self._case_by_id: dict[str, Any] = {}
        self.run_config_failed = False

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--case",
            nargs="+",
            default=[],
            help="optional regex filters on pytest case ids (after untuned shape filter)",
        )
        self.parser.add_argument(
            "--only-improvements",
            action="store_true",
            help="emit a row only when measured BV beats the rule's choice",
        )
        self.parser.add_argument(
            "--run_config_tol_pct",
            type=float,
            default=5.0,
            help="run_config pass threshold for live_us vs csv us drift (percent)",
        )
        self.parser.add_argument(
            "--list-cases",
            action="store_true",
            help="print PrefillGroup case ids and exit",
        )

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
            return

        untuned_rows = read_csv_rows(args.untune_file)
        self._cases = select_cases(load_k5_cases(), untuned_rows, args.case)
        self._case_by_id = {case_id: case for case_id, case in self._cases}

        if not self._cases:
            self.untunedf = pd.DataFrame(columns=list(TUNED_COLUMNS) + ["_case_id"])
            self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))
            return

        self.untunedf = dataframe_from_cases(self._cases)
        self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))

        if self.tunedf is not None and not self.tunedf.empty:
            dedup_cols = [c for c in self.keys if c in self.tunedf.columns]
            if len(dedup_cols) == len(self.keys):
                tuned_keys = set(self.tunedf[dedup_cols].apply(tuple, axis=1))
                mask = self.untunedf[dedup_cols].apply(tuple, axis=1).isin(tuned_keys)
                if mask.any() and args.verbose:
                    print(f"skip {mask.sum()} shapes already present in tuned csv")
                self.untunedf = self.untunedf[~mask].reset_index(drop=True)
                self._cases = [
                    (row["_case_id"], self._case_by_id[row["_case_id"]])
                    for _, row in self.untunedf.iterrows()
                ]

    def tune(self, untunedf, tunedf, args):
        if untunedf.empty:
            return []

        if not hasattr(self, "_printed_tune_header"):
            header = (
                f"{'case':58s} {'chunks':>7s} "
                + " ".join(f"BV{bv:<7d}" for bv in BV_CANDIDATES)
            ) + f" {'best':>5s} {'rule':>5s} {'gain%':>6s}"
            print(header)
            print("-" * len(header))
            self._printed_tune_header = True

        frames = []
        emitted: dict[tuple, dict[str, Any]] = {}
        for _, row in untunedf.iterrows():
            case_id = row["_case_id"]
            case = self._case_by_id[case_id]
            tuned_row = sweep_case_row(
                case_id,
                case,
                args.warmup,
                args.iters,
                args.only_improvements,
            )
            if tuned_row is None:
                continue
            snapshot_dtype = case_snapshot_dtype(case)
            batch = 1 if case.is_varlen else case.dense_batch
            total_chunks, max_seq_chunks = chunk_counts(
                case.resolve_context_lens(), batch
            )
            key = lookup_key_from_case(
                case, snapshot_dtype, total_chunks, max_seq_chunks
            )
            emitted[key] = tuned_row
            torch.cuda.empty_cache()

        if emitted:
            frames = list(emitted.values())
        if not frames:
            return []
        return pd.DataFrame(frames, columns=self.columns).to_dict("records")

    def post_process(self, results, args, topk=-1, fast_mode=False):
        if isinstance(results, list):
            results = pd.DataFrame(results, columns=self.columns)
        if isinstance(results, pd.DataFrame):
            if results.empty:
                return results
            return (
                results.sort_values("us")
                .drop_duplicates(subset=self.keys, keep="first")
                .reset_index(drop=True)
            )
        return pd.DataFrame(columns=self.columns)

    def result_to_csv(self, results, file, concat=False):
        old_tunedf = self.get_tuned_gemm_list(file)
        for col in self.columns:
            if col not in old_tunedf.columns:
                old_tunedf[col] = pd.NA
        resultdf = self.update_tunedf(old_tunedf, results.loc[:, self.columns])
        self.success = pd.concat([self.success, results], ignore_index=True)
        if results is not None and not results.empty:
            resultdf = resultdf.astype(str).drop_duplicates(
                subset=self.keys, keep="last"
            )
        ordered_cols = [c for c in self.columns if c in resultdf.columns]
        ordered_cols.extend(c for c in resultdf.columns if c not in ordered_cols)
        resultdf = resultdf[ordered_cols]
        resultdf.to_csv(file, index=False)

    def run_config(self, args):
        tol = float(getattr(args, "run_config_tol_pct", 5.0))
        cases = load_k5_cases()
        results = []
        print("Shape | e2e_us | Status")
        print("-" * 60)

        for _, row in self.untunedf.iterrows():
            row_dict = row.to_dict()
            case = find_case_for_row(cases, row_dict)
            label = f"H={row_dict['H']}/Hg={row_dict['Hg']}"
            shape = f"({label}, tc={row_dict['total_chunks']}, BV={row_dict['BV']})"
            if case is None:
                print(f"{shape} | {'-1':>10} | ERROR")
                print("reason: no matching K5 prefill case")
                results.append({"shape": shape, "us": -1.0, "status": "error:no case"})
                continue

            snapshot_dtype = case_snapshot_dtype(case)
            inputs, *_rest = build_k5_inputs(case, snapshot_dtype)
            bv = int(row_dict["BV"])
            csv_us = float(row_dict["us"])
            try:
                live_us = bench_us(inputs, bv, args.warmup, args.iters)
                delta = (live_us - csv_us) / csv_us * 100 if csv_us > 0 else 0.0
                if abs(delta) <= max(tol, 0.0):
                    status = "ok"
                    print(f"{shape} | {live_us:>10.1f} | OK")
                else:
                    status = (
                        f"mismatch: live_us drift {delta:.1f}% vs csv (tol {tol:.1f}%)"
                    )
                    print(f"{shape} | {live_us:>10.1f} | MISMATCH")
                    print(f"reason: {status[len('mismatch:'):].strip()}")
            except Exception as exc:  # noqa: BLE001
                live_us = -1.0
                status = f"error: {exc}"
                print(f"{shape} | {'-1':>10} | ERROR")
                print(f"reason: {exc}")
            results.append({"shape": shape, "us": live_us, "status": status})
            del inputs
            torch.cuda.empty_cache()
        return results

    def run(self, args, fast_mode=False):
        if args.list_cases:
            for case_id, _ in load_k5_cases():
                print(case_id)
            return pd.DataFrame()

        self.pre_process(args)

        run_config_file = args.run_config if isinstance(args.run_config, str) else None
        if args.run_config and run_config_file:
            tunedf = self.get_tuned_gemm_list(run_config_file)
            if not tunedf.empty and self.keys[0] in tunedf.columns:
                self.untunedf = tunedf.drop_duplicates(subset=self.keys).reset_index(
                    drop=True
                )

        if args.run_config:
            if self.untunedf.empty:
                print("No shapes to benchmark, nothing to run")
                return pd.DataFrame()
            results = self.run_config(args)
            self.run_config_failed = any(
                not str(r.get("status", "")).startswith("ok") for r in results
            )
            return self.tunedf if self.tunedf is not None else pd.DataFrame()

        if hasattr(self, "_printed_tune_header"):
            del self._printed_tune_header
        out = super().run(args, fast_mode=fast_mode)
        if not self.untunedf.empty:
            print(f"\n{len(self.success)} tuned rows")
        return out

    def tune_summary(self, status):
        tuning_time = round(time.time() - getattr(self, "tune_start_time", 0), 4)
        logger.info("============= Tuning results Summary: ==============")
        logger.info(
            f"Tuning {status}. tune {len(self.success)} shapes, "
            f"total tuning time is {tuning_time} seconds"
        )
        if not self.success.empty:
            logger.info("Successfully tuned shapes:")
            print(self.success, flush=True)
        if not self.failed.empty:
            logger.info("Failed shapes:")
            print(self.failed, flush=True)
            sys.exit(1)
        if self.success.empty and not self.untunedf.empty:
            logger.error("\033[91m[Tuning not Finished]\033[0m no shapes were tuned")
            sys.exit(1)

    def getKernelName(self, kernel_id):
        return f"BV{kernel_id}"

    def calculate(self, results, inbpe=2, outbpe=2):
        return 0, 0

    def result_to_df(self, rets):
        if isinstance(rets, pd.DataFrame):
            return rets
        return pd.DataFrame(columns=self.columns)


def main():
    tuner = K5BvTuner()
    args = tuner.parse_args()
    tuner.run(args, fast_mode=False)
    if args.run_config and tuner.run_config_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
