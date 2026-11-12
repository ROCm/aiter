# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-graph FlyDSL gfx942 fused MoE tuning (--e2e_tune)."""

import os

import pandas as pd

from aiter import ActivationType, QuantType
from aiter.fused_moe_registry import make_fused_moe_impl_kernel_name


def tune_flydsl_gfx942(self, args):
    """
    Choosing best kernels based on (stage1_us + stage2_us) or (single_stage_us)
    may overlook some overheads between stages, and this e2e tune is a complement.
    """
    from functools import partial

    from aiter.ops.flydsl.fused_moe_gfx942 import (
        Config,
        _Problem,
        get_tune_space,
        run_flydsl_moe_gfx942,
    )

    results_base = self._run_config_for_shapes(
        args,
        self.untunedf,
        config_file=self.get_out_file(args.tune_file),
    )
    better_kernels = {}

    for i in range(len(self.untunedf)):
        e2e_us = results_base[i]["e2e_us"]
        err_ratio = results_base[i].get("err_ratio", 0)
        status = results_base[i].get("status", "")
        row = self.untunedf.iloc[i]
        row_key = tuple(row[col] for col in self.keys)
        keyname = " ".join(map(str, row_key))
        baseline_valid = status == "ok" and e2e_us > 0
        better_kernels[i] = {
            "name": keyname,
            "row": row,
            "kernel_name": None,
            "e2e_us": e2e_us if baseline_valid else float("inf"),
            "err_ratio": err_ratio,
            "e2e_us_base": e2e_us,
            "err_ratio_base": err_ratio,
        }
        print(keyname, e2e_us, err_ratio)

    def target_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight_stage1=False,
        w1_scale=None,
        w2_scale=None,
        num_local_tokens=None,
        moe_sorting_dispatch_policy=0,
        dtype=None,
        config_string="",
        swiglu_limit=None,
        beta=None,
        linear_beta=None,
    ):
        del beta, linear_beta
        if doweight_stage1:
            raise NotImplementedError(
                "gfx942 FlyDSL whole-graph tuning does not support "
                "doweight_stage1=True"
            )
        return run_flydsl_moe_gfx942(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            activation,
            quant_type,
            w1_scale,
            w2_scale,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            config_string=config_string,
            swiglu_limit=swiglu_limit,
        )

    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    END = "\033[0m"
    for config_string in get_tune_space():
        config = Config.from_string(config_string)
        eligible_indices = [
            position
            for position, (_, row) in enumerate(self.untunedf.iterrows())
            if not bool(row["doweight_stage1"])
            and config.unsupported_reason(
                _Problem(
                    batch=int(row["token"]),
                    experts=int(row["expert"]),
                    gateup_dim=int(row["inter_dim"]) * 2,
                    hidden_dim=int(row["model_dim"]),
                    model_dim=int(row["model_dim"]),
                    inter_dim=int(row["inter_dim"]),
                    topk=int(row["topk"]),
                    quant_type=("no" if row["q_type"] == str(QuantType.No) else ""),
                )
            )
            is None
        ]
        if not eligible_indices:
            continue
        all_untunedf = self.untunedf
        self.untunedf = all_untunedf.iloc[eligible_indices].reset_index(drop=True)
        try:
            results_cur = self.run_config(
                args,
                target_fused_moe=partial(target_fused_moe, config_string=config_string),
                config_string=config_string,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{RED}Error with config {config_string}: {e}{END}")
            continue
        finally:
            self.untunedf = all_untunedf
        block_m = config.BLOCK_M
        ksplit = 0
        run_1stage = 0
        err1 = "0%"
        err2 = "0%"
        kernelName1 = make_fused_moe_impl_kernel_name("flydsl_gfx942", config_string)
        kernelName2 = ""
        xbf16 = 0
        for result_index, i in enumerate(eligible_indices):
            k = better_kernels[i]
            e2e_us = results_cur[result_index]["e2e_us"]
            status = results_cur[result_index]["status"]
            err_ratio = results_cur[result_index].get("err_ratio", 0)
            # skip invalid kernel
            if e2e_us < 0 or status != "ok":
                print(f"{k['name']} {RED} {e2e_us=:.3f} {status=} {END} {kernelName1}")
                continue
            print(
                f"{k['name']} {YELLOW} {float(k['e2e_us_base']):.3f}us -> {float(e2e_us):.3f}us (err: {err_ratio*100:.0f}%) {END} {kernelName1}"
            )
            if e2e_us < k["e2e_us"]:
                k["e2e_us"] = e2e_us
                k["err_ratio"] = err_ratio
                k["kernel_name"] = kernelName1
                row = self.untunedf.iloc[i]
                try:
                    key = tuple(row[self.keys].values)
                    tflops, bw = self.calculate(
                        (key, "stage1", kernelName1, block_m, e2e_us, err1)
                    )
                except Exception:  # noqa: BLE001
                    tflops, bw = 0, 0
                k["results"] = (
                    block_m,
                    ksplit,
                    e2e_us,
                    kernelName1,
                    f"{err_ratio*100:.2f}%",
                    0.0,
                    kernelName2,
                    err2,
                    e2e_us,
                    run_1stage,
                    xbf16,
                    0,  # flat
                    tflops,
                    bw,
                    0,  # nt: flydsl rows do not read the hint
                )

    tune_results = []

    for i, k in better_kernels.items():
        if k["kernel_name"] is None:
            continue
        tune_results.append([*[k["row"][col] for col in self.keys], *k["results"]])
        print(
            f"{k['name']} {GREEN} {float(k['e2e_us_base']):.3f}us -> {float(k['e2e_us']):.3f}us (err: {k['err_ratio_base']*100:.0f}% -> {k['err_ratio']*100:.0f}%) {END} {k['kernel_name']}"
        )

    if tune_results:
        new_tunedf = pd.DataFrame(tune_results, columns=self.columns)
        new_tunedf["_tag"] = ""
        output_file = self.get_out_file(args.tune_file)
        # Merge with existing tuned file: keep existing rows that are not
        # being updated, so repeated runs don't lose entries for shapes
        # where the new kernel doesn't beat the baseline.
        key_cols = []
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            # Build key for dedup: use the untuned input columns
            key_cols = [
                c
                for c in self.keys
                if c in new_tunedf.columns and c in existing_df.columns
            ]
            if key_cols:
                # Remove from existing any rows that will be replaced by new results
                new_keys = set(new_tunedf[key_cols].apply(tuple, axis=1))
                keep_mask = ~existing_df[key_cols].apply(tuple, axis=1).isin(new_keys)
                merged_df = pd.concat(
                    [existing_df[keep_mask], new_tunedf], ignore_index=True
                )
            else:
                merged_df = new_tunedf
        else:
            merged_df = new_tunedf
        # Sort by key columns before writing
        sort_cols = [c for c in key_cols if c in merged_df.columns] if key_cols else []
        if sort_cols:
            merged_df = merged_df.sort_values(sort_cols, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"{output_file} has been updated with {len(tune_results)} entries!")
    else:
        print("No improvements found during e2e tuning.")
