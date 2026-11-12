# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped-GEMM fused MoE tuner (--grouped-gemm)."""

import os
import re
import tempfile
from decimal import Decimal
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import (
    ActivationType,
    QuantType,
    dtypes,
)
from aiter.fused_moe import (
    fused_moe,
)
from aiter.jit.core import (
    AITER_CONFIG_GROUPED_FMOE,
    AITER_ROOT_DIR,
)
from aiter.ops.shuffle import (
    shuffle_weight,
)
from aiter.tuning.moe.fused_moe import (
    FmoeTuner,
    _frame_keeping_watts,
)


class GroupedFmoeTuner(FmoeTuner):
    WARP_TILE_N = 64
    TILE_K = 256
    TILE_M_CANDIDATES = (16, 32, 64, 128)

    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **FmoeTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GROUPED_FMOE}",
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/untuned_grouped_fmoe.csv",
        "config_env_name": "AITER_CONFIG_GROUPED_FMOE",
    }

    def _data_format(self, q_dtype_a: str) -> str:
        return "fp4" if ("fp4x2" in q_dtype_a or "float4" in q_dtype_a) else "a8w4"

    def _candidate_row(self, row, tile_m: int):
        token = int(row["token"])
        expert = int(row["expert"])
        topk = int(row["topk"])
        q_dtype_a = str(row["q_dtype_a"])
        gate_mode = str(row.get("gate_mode", "GateMode.SEPARATED"))
        layout = "gugu" if gate_mode.endswith("INTERLEAVE") else "gguu"
        max_m_raw = (token * topk + expert - 1) // expert
        max_m = max(tile_m, ((max_m_raw + tile_m - 1) // tile_m) * tile_m)
        data_format = self._data_format(q_dtype_a)
        n_warp = 4
        return {
            **{k: row[k] for k in self.keys if k in row},
            "gate_mode": gate_mode,
            "max_m": max_m,
            "tile_m": tile_m,
            "tile_n": self.WARP_TILE_N * n_warp,
            "tile_k": self.TILE_K,
            "m_warp": 1,
            "n_warp": n_warp,
            "num_buffers": 2,
            "grouped_persistent_m": 1,
            "persistent_workers": "",
            "kernelName1": f"grouped_gemm1_{data_format}_{layout}",
            "kernelName2": f"grouped_gemm2_{data_format}",
            "us": 0,
            "tflops": 0,
            "bw": 0,
        }

    def _candidate_rows(self, row):
        return [self._candidate_row(row, tile_m) for tile_m in self.TILE_M_CANDIDATES]

    @staticmethod
    def _balanced_topk(token_num: int, topk: int, experts: int):
        tok = torch.arange(token_num, device="cuda").view(token_num, 1)
        rk = torch.arange(topk, device="cuda").view(1, topk)
        ids = ((tok * topk + rk) % experts).to(torch.int32)
        weights = torch.full((token_num, topk), 1.0 / topk, device="cuda")
        return ids, weights.to(torch.bfloat16)

    @staticmethod
    def _full_scale(experts: int, rows: int, k_blocks: int):
        return torch.full((experts, rows, k_blocks), 127, dtype=torch.uint8)

    def _prepare_grouped_case(self, row, candidate):
        from aiter.ops.flydsl.grouped_moe_gfx1250 import (
            _grouped_a8w4_prepare_scale_batch,
        )

        token = int(row["token"])
        model_dim = int(row["model_dim"])
        inter_dim = int(row["inter_dim"])
        experts = int(row["expert"])
        topk = int(row["topk"])
        data_format = self._data_format(str(row["q_dtype_a"]))
        gen = torch.Generator(device="cpu").manual_seed(0)
        hidden = (
            torch.randn(
                (token, model_dim),
                generator=gen,
                dtype=torch.bfloat16,
                device="cpu",
            )
            / 10
        ).cuda()
        topk_ids, topk_weight = self._balanced_topk(token, topk, experts)
        w1 = torch.randint(
            0,
            256,
            (experts, 2 * inter_dim, model_dim // 2),
            dtype=torch.uint8,
            generator=gen,
            device="cpu",
        )
        w2 = torch.randint(
            0,
            256,
            (experts, model_dim, inter_dim // 2),
            dtype=torch.uint8,
            generator=gen,
            device="cpu",
        )
        w1 = shuffle_weight(w1, layout=(16, 16)).cuda()
        w2 = shuffle_weight(w2, layout=(16, 16)).cuda()
        w1_arg = w1.view(dtypes.fp4x2) if data_format == "fp4" else w1
        w2_arg = w2.view(dtypes.fp4x2) if data_format == "fp4" else w2
        warp_tile_n = int(candidate["tile_n"]) // int(candidate["n_warp"])
        w1_scale = self._full_scale(experts, 2 * inter_dim, model_dim // 32)
        w2_scale = self._full_scale(experts, model_dim, inter_dim // 32)
        w1_scale = _grouped_a8w4_prepare_scale_batch(
            w1_scale.view(dtypes.fp8_e8m0),
            experts=experts,
            rows=2 * inter_dim,
            k_dim=model_dim,
            warp_tile=warp_tile_n,
            tile_k=int(candidate["tile_k"]),
            device=hidden.device,
        )
        w2_scale = _grouped_a8w4_prepare_scale_batch(
            w2_scale.view(dtypes.fp8_e8m0),
            experts=experts,
            rows=model_dim,
            k_dim=inter_dim,
            warp_tile=warp_tile_n,
            tile_k=int(candidate["tile_k"]),
            device=hidden.device,
        )
        return hidden, w1_arg, w2_arg, topk_weight, topk_ids, w1_scale, w2_scale

    def _write_candidate_config(self, candidate):
        # delete=False on purpose: the path outlives the handle.
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            suffix=".csv",
            prefix="aiter_grouped_candidate_",
            delete=False,
            newline="",
        )
        try:
            pd.DataFrame([candidate], columns=self.columns).to_csv(
                tmp.name, index=False
            )
            return tmp.name
        finally:
            tmp.close()

    def _run_candidate(self, row, candidate, args):
        from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped_mod
        from aiter.ops.flydsl.moe_common import GateMode
        from aiter.test_common import run_perftest

        def _clear_grouped_config_cache():
            cache = getattr(grouped_mod, "_GROUPED_CONFIG_CACHE", None)
            if cache is not None:
                cache.clear()

        config_path = self._write_candidate_config(candidate)
        old_config = os.environ.get("AITER_CONFIG_GROUPED_FMOE")
        old_enable = os.environ.get("AITER_USE_GROUPED_GEMM")
        old_force = os.environ.get("AITER_FORCE_GFX1250")
        try:
            os.environ["AITER_CONFIG_GROUPED_FMOE"] = config_path
            os.environ["AITER_USE_GROUPED_GEMM"] = "1"
            os.environ["AITER_FORCE_GFX1250"] = "1"
            _clear_grouped_config_cache()
            case = self._prepare_grouped_case(row, candidate)
            activation = (
                ActivationType.Swiglu
                if str(row["act_type"]).endswith("Swiglu")
                else ActivationType.Silu
            )
            gate_mode = (
                GateMode.INTERLEAVE
                if str(row.get("gate_mode", "")).endswith("INTERLEAVE")
                else GateMode.SEPARATED
            )

            def _call():
                return fused_moe(
                    case[0],
                    case[1],
                    case[2],
                    case[3],
                    case[4],
                    activation=activation,
                    quant_type=QuantType.per_1x32,
                    w1_scale=case[5],
                    w2_scale=case[6],
                    dtype=torch.bfloat16,
                    gate_mode=gate_mode.value,
                )

            _call()
            torch.cuda.synchronize()
            _, us = run_perftest(
                _call, num_warmup=int(args.warmup), num_iters=int(args.iters)
            )
            us = round(float(us), 4)
            self.policy_for(args).record(candidate, "watts", _call, kernel_us=us)
            return us
        finally:
            if old_config is None:
                os.environ.pop("AITER_CONFIG_GROUPED_FMOE", None)
            else:
                os.environ["AITER_CONFIG_GROUPED_FMOE"] = old_config
            if old_enable is None:
                os.environ.pop("AITER_USE_GROUPED_GEMM", None)
            else:
                os.environ["AITER_USE_GROUPED_GEMM"] = old_enable
            if old_force is None:
                os.environ.pop("AITER_FORCE_GFX1250", None)
            else:
                os.environ["AITER_FORCE_GFX1250"] = old_force
            _clear_grouped_config_cache()
            try:
                os.unlink(config_path)
            except OSError:
                pass

    def tune(self, untunedf, tunedf, args):
        self._require_supported_fast_scan(args)
        del tunedf
        policy = self.policy_for(args)
        # Refuse an output file that the write-back cannot keep blank-free
        # before timing anything.
        self._write_layout(self.get_out_file(args.tune_file))
        rows = []
        for _, row in untunedf.iterrows():
            best = None
            failures = []
            for candidate in self._candidate_rows(row):
                try:
                    us = self._run_candidate(row, candidate, args)
                    candidate["us"] = us
                    watt_text = policy.watt_text(candidate.get("watts"))
                    print(
                        f"[grouped] token={row['token']} inter={row['inter_dim']} "
                        f"qa={row['q_dtype_a']} tile_m={candidate['tile_m']} "
                        f"us={us}{watt_text}",
                        flush=True,
                    )
                    if policy.beats(best, candidate):
                        best = candidate
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"tile_m={candidate['tile_m']}: {exc}")
                    print(f"[grouped] candidate failed: {failures[-1]}", flush=True)
            if best is None:
                best = self._candidate_row(row, self.TILE_M_CANDIDATES[0])
                best["us"] = self.INVALID_TIME
                failure_text = "; ".join(failures)
                best["kernelName1"] = ("FAILED: " + failure_text)[:240]
                print(
                    f"[grouped] all candidates failed for {tuple(row[k] for k in self.keys)}: "
                    + failure_text,
                    flush=True,
                )
            rows.append(best)
        return rows

    def post_process(self, results, args, topk=-1, fast_mode=False):
        del args, topk, fast_mode
        return _frame_keeping_watts(self.columns, results)

    # What grouped_moe_gfx1250 (and the AOT parser, for split_k*) reads a blank
    # cell as: the value of another column of the same row, or a constant.
    _BLANK_SAME_AS: ClassVar[dict[str, str]] = {
        "tile_m2": "tile_m",
        "tile_n2": "tile_n",
        "tile_k2": "tile_k",
        "m_warp2": "m_warp",
        "n_warp2": "n_warp",
        "num_buffer_stage2": "num_buffers",
        "cluster_n2": "cluster_n",
        "waves_per_tensor_tdm2": "waves_per_tensor_tdm",
    }
    _BLANK_VALUE: ClassVar[dict[str, int]] = {
        "split_k1": 1,
        "split_k2": 1,
        "cluster_m": -1,
        "cluster_n": -1,
        "cluster_m2": -1,
        "waves_per_tensor_tdm": -1,
        "next_stage_prefetch": 0,
        "tdm_as_in_prologue": 0,
        "tdm_b_th": 0,
        "lds_soa_load_interleave": 0,
        "rolled_drain": 0,
        "persistent_workers2": 0,
        "a_preshuffle": 0,
    }
    _NUMBER: ClassVar[re.Pattern[str]] = re.compile(
        r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"
    )

    @classmethod
    def _blank_means(cls, column, row):
        while column in cls._BLANK_SAME_AS:
            source = cls._BLANK_SAME_AS[column]
            if row.get(source, ""):
                return row[source]
            column = source
        value = cls._BLANK_VALUE.get(column)
        return "" if value is None else str(value)

    def _blank_is_known(self, column):
        while column in self._BLANK_SAME_AS:
            column = self._BLANK_SAME_AS[column]
        return column in self._BLANK_VALUE or column in self.columns

    @classmethod
    def _csv_text(cls, value):
        # pandas hands blank cells back as NaN, and the integers of a column
        # that has a blank cell as floats; write an integral number as the
        # integer, exactly ("16.0" and "1e5" become "16" and "100000").
        text = "" if value is None else str(value).strip()
        if text.lower() in ("nan", "none", "<na>"):
            return ""
        if cls._NUMBER.fullmatch(text):
            number = Decimal(text)
            # The bound keeps a stray exponent from building a huge integer.
            if number.adjusted() < 40 and number == number.to_integral_value():
                return str(int(number))
        return text

    def _write_layout(self, file):
        """Columns of a write to ``file``, its rows as text, and the columns in
        which a written row must not leave a blank cell."""
        # Read the file as text, so that a pandas round trip does not rewrite
        # the rows a run leaves alone ("16" becomes "16.0").
        if os.path.exists(file) and os.path.getsize(file) > 0:
            old_rows = pd.read_csv(file, dtype=str, keep_default_na=False)
            columns = list(old_rows.columns)
            # A row takes the file's columns, so a re-tuned row replaces the
            # old one and no other row gets blank cells. Only a missing key is
            # added, to keep rows apart; the lookup does not read gate_mode
            # (kernelName1 names the layout).
            columns += [c for c in self.keys if c not in columns and c != "gate_mode"]
            old_rows = old_rows.to_dict("records")
        else:
            columns = list(self.columns)
            old_rows = []
        # One blank cell makes the AITER config merge read the whole column as
        # floats, which the AOT parser's int() rejects; so in a column without
        # blanks, a re-tuned row gets the value a blank cell means instead.
        no_blanks = {
            c
            for c in columns
            if c not in self.columns
            and old_rows
            and all(str(row.get(c, "")).strip() for row in old_rows)
        }
        unknown = sorted(c for c in no_blanks if not self._blank_is_known(c))
        if unknown:
            raise ValueError(
                f"{file}: no blank-cell value for {', '.join(unknown)}; add it to "
                "GroupedFmoeTuner._BLANK_VALUE or _BLANK_SAME_AS"
            )
        return columns, old_rows, no_blanks

    def result_to_csv(self, results, file, concat=False):
        del concat
        columns, old_rows, no_blanks = self._write_layout(file)
        valid = results[
            (results["us"] != self.INVALID_TIME) & (results["us"] != self.INF_TIME)
        ]
        invalid = results[
            (results["us"] == self.INVALID_TIME) | (results["us"] == self.INF_TIME)
        ]
        self.success = pd.concat([self.success, valid], ignore_index=True)
        self.failed = pd.concat([self.failed, invalid], ignore_index=True)
        keys = [k for k in self.keys if k in columns]
        rows = [{c: row.get(c, "") for c in columns} for row in old_rows]
        where = {
            tuple(self._csv_text(row.get(k, "")) for k in keys): i
            for i, row in enumerate(old_rows)
        }
        for result in valid.to_dict("records"):
            # The columns the tuner does not set mean what a blank cell means:
            # it timed the candidate with them blank.
            new = {c: self._csv_text(result.get(c, "")) for c in columns}
            for c in no_blanks:
                new[c] = self._blank_means(c, new)
            key = tuple(new[k] for k in keys)
            if key in where:
                rows[where[key]] = new
            else:
                where[key] = len(rows)
                rows.append(new)
        pd.DataFrame(rows, columns=columns).to_csv(file, index=False)
