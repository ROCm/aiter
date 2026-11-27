# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MoE tuner: CLI, the loop over backends, stage pairing and the winner."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch

import aiter
from aiter import (
    ActivationType,
    QuantType,
    dtypes,
)
from aiter.fused_moe import (
    AsmMoeMode,
    swiglu_limit_tag,
    torch_moe_2stages,
    torch_moe_blockscale,
)
from aiter.fused_moe_bf16_asm import torch_moe_tkw1
from aiter.int4_utils import (
    convert_int8_to_uint32_int4,
    rearrange_4bit_elements,
)
from aiter.jit.core import (
    AITER_CONFIG_FMOE,
    AITER_ROOT_DIR,
)
from aiter.jit.utils.chip_info import get_gfx, gfx_from_cu_num
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
)
from aiter.ops.moe_limits import (
    RUNTIME_SILU_LIMIT_QUANT_TYPES,
    normalize_swiglu_limit,
)
from aiter.ops.shuffle import (
    pack_int8_to_packed_int4,
    shuffle_scale_a16w4,
    shuffle_scale_for_int4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import (
    _all_finite,
    tensor_compare_diagnostics,
)
from aiter.tuning.moe.backends.asm_1stage import (
    gen_1stage_asm_task,
    get_1stage_file_info,
    get_1stage_fmoe_func,
    run_1stage_fmoe_fp8_blockscale_g1u1,
    run_1stage_fmoe_g1u1,
    run_1stage_fmoe_g1u1_tkw1,
)
from aiter.tuning.moe.backends.asm_stage1 import (
    gen_2stages_asm1_task,
    run_asm_stage1,
)
from aiter.tuning.moe.backends.ck import (
    ck_moe_stage1_fwd_out,
    ck_moe_stage2_fwd_out,
    gen_2stages_task,
)
from aiter.tuning.moe.backends.cktile import (
    _gen_2stages_task_cktile,
    cktile_moe_stage1_out,
    cktile_moe_stage2_out,
)
from aiter.tuning.moe.backends.common import get_kernels_dict
from aiter.tuning.moe.backends.flydsl import (
    gen_flydsl_2stages_task,
    run_a16w_stage1_sorted_ref,
    run_flydsl_stage1_out,
    run_flydsl_stage2_out,
)
from aiter.tuning.moe.backends.flydsl_i4 import gen_flydsl_i4_2stages_task
from aiter.tuning.moe.backends.flydsl_v2 import (
    gen_flydsl_v2_2stages_task,
    generate_v2_stage1_data,
    generate_v2_stage2_data,
    run_flydsl_v2_stage1_out,
    run_flydsl_v2_stage2_out,
    run_v2_stage1_sorted_ref,
    run_v2_stage2_ref,
)
from aiter.tuning.moe.backends.opus import (
    gen_opus_2stages_task,
    run_opus_stage1_out,
    run_opus_stage2_out,
)
from aiter.tuning.moe.compare import (
    run_torch_moe_stage1,
    run_torch_moe_stage1_ref,
    run_torch_moe_stage2,
    torch_moe_test,
)
from aiter.tuning.moe.flydsl_gfx942 import tune_flydsl_gfx942
from aiter.tuning.moe.inputs import (
    _ep_route_stats,
    generate_asm_stage1,
    generate_data,
    generate_data_1stage,
    generate_data_2stages,
    generate_opus_sorted_stage2_data,
    weight_quant,
)
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import TunerCommon
from aiter.utility.mp_tuner import mp_tuner


def _candidate_dump_tag(
    gfx,
    cu_num,
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    act_type,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    doweight_stage1,
):
    return re.sub(
        r"[^0-9A-Za-z_.-]+",
        "_",
        (
            f"{gfx}_cu{cu_num}_t{token}_h{model_dim}_i{inter_dim}_e{expert}_k{topk}"
            f"_act{act_type}_dtype{dtype}_qa{q_dtype_a}_qw{q_dtype_w}_qt{q_type}"
            f"_g1u1{int(use_g1u1)}_dw{int(doweight_stage1)}"
        ),
    )


def _dump_prefilter_candidates(profile_df, args, stage_tag):
    root = os.environ.get("AITER_TUNE_CANDIDATE_DUMP_DIR")
    if not root:
        return
    out = profile_df.copy()
    out["passed_err_ratio"] = out["err"] <= args.errRatio
    out["filter_reason"] = ""
    invalid_time = out["us"].isin([float("inf"), float("-inf"), -1])
    out.loc[invalid_time, "filter_reason"] = "invalid_time"
    out.loc[(out["err"] > args.errRatio) & ~invalid_time, "filter_reason"] = "err_ratio"
    Path(root).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(root) / f"{stage_tag}.csv", index=False)


def _mxfp4_search_mode(args):
    """The MXFP4 search mode: an omitted one is staged, or prune under --fast-scan.

    prune is internal only; the CLI accepts full and staged.
    """
    mode = getattr(args, "mxfp4_search_mode", None)
    if mode:
        return mode
    return "prune" if getattr(args, "fast_scan", False) else "staged"


def _numeric_column(frame, name, default=float("nan")):
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def _frame_keeping_watts(columns, results):
    """Build the tuned frame without dropping a power measurement.

    ``DataFrame(..., columns=self.columns)`` keeps only the latency schema.
    Grouped and MXFP4 winners carry ``watts`` on the dict, and dropping it
    made the following stamp publish ``select=power`` with a null reading.
    """
    kept = list(columns)
    if (
        results
        and isinstance(results[0], dict)
        and "watts" in results[0]
        and "watts" not in kept
    ):
        kept.append("watts")
    return pd.DataFrame(results, columns=kept)


def rank_moe_pairs(profile_df, policy):
    """Rank paired MoE candidates with the tune's selection policy.

    The pair cost is the policy's stage energy ``us1*watts1 + us2*watts2``
    (stage 2 contributes only when it actually ran); its mean watts feed the
    same frame order every tuner uses. ``rank`` 1 is the winner.

    Latency mode returns the frame unchanged, so the ``-o2`` profile keeps
    main's columns and row order.
    """
    if not policy.measures_power:
        return profile_df
    frame = profile_df.copy()
    us1 = _numeric_column(frame, "us1")
    us2 = _numeric_column(frame, "us2", default=0.0).fillna(0.0)
    watts1 = _numeric_column(frame, "watts1")
    watts2 = _numeric_column(frame, "watts2")
    energy = [
        policy.pair_energy(a, b, c, d)
        for a, b, c, d in zip(us1, watts1, us2, watts2, strict=True)
    ]
    frame["energy"] = pd.Series(energy, index=frame.index).round(1)
    us = _numeric_column(frame, "us")
    if us.isna().all() and "us1" in frame.columns:
        us = us1
        frame["us"] = us
    mean_watts = frame["energy"] / us.replace(0, pd.NA)
    frame["watts"] = mean_watts.round(3)
    if "tflops" in frame.columns:
        tflops = pd.to_numeric(frame["tflops"], errors="coerce")
        frame["tflops_per_w"] = (tflops / mean_watts).round(6)
    frame = policy.sort_frame(frame).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


class FmoeTuner(TunerCommon):
    SUPPORTS_FAST_SCAN = False
    SUPPORTS_MXFP4_SEARCH_MODE = False
    SUPPORTS_POWER_SELECT = True
    swiglu_limit = None
    swiglu_limit_tag = ""
    KERNEL_NAME_COLUMNS: ClassVar[tuple[str, ...]] = ("kernelName1", "kernelName2")
    TAG_ROWS_DROPPED: ClassVar[str] = "all"

    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "verbose": False,
        "tune_file": f"{AITER_CONFIG_FMOE}",
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/untuned_fmoe.csv",
        # max accepted cosine_diff; 0.1 rejects broken kernels
        "errRatio": 0.1,
        "batch": 100,
        "profile_file": "",  # for all results
        "config_env_name": "AITER_CONFIG_FMOE",
    }

    def _clear_op_caches(self):
        import aiter.fused_moe as fmoe_module

        if hasattr(fmoe_module, "cfg_2stages"):
            fmoe_module.cfg_2stages = None
        # --run_config times the rows of the given table, also with
        # AITER_FLYDSL_STAGE2_FP8=1: no row of the stage2 FP8 table replaces them.
        if hasattr(fmoe_module, "cfg_2stages_stage2_fp8"):
            fmoe_module.cfg_2stages_stage2_fp8 = ({}, {})
        if hasattr(fmoe_module, "get_2stage_cfgs"):
            fmoe_module.get_2stage_cfgs.cache_clear()

    def _setup_specific_arguments(self):

        self.parser.add_argument(
            "--last",
            action="store_true",
            required=False,
            help="Only last kernel is tuned, if not, only kernels that are not in the tuned_fmoe.csv are tuned",
        )
        self.parser.add_argument(
            "--grouped-gemm",
            action="store_true",
            required=False,
            help="On gfx1250, tune the FlyDSL grouped-GEMM MoE path instead of the normal fmoe tuner.",
        )
        self.parser.add_argument(
            "--mxfp4-flydsl",
            action="store_true",
            required=False,
            help="Tune FlyDSL MXFP4 A4W4/A8W4 as a coupled (g1, g2) pipeline.",
        )
        self.parser.add_argument(
            "--mxfp4-search-mode",
            choices=("full", "staged"),
            help="screen the M_est-pruned stages (staged, the default; shapes with "
            "M_est < 4 validate every pruned pair unscreened) or exhaust all legal "
            "variants (full). Omitted with --fast-scan, every M_est-pruned pair is "
            "scanned. Requires --mxfp4-flydsl.",
        )
        self.parser.add_argument(
            "--mxfp4-screen-topk",
            type=int,
            help="positive screening top-K per stage context (default 2); "
            "staged search only; unused for M_est < 4; distinct from routing top-k.",
        )
        self.parser.add_argument(
            "--swiglu_limit",
            type=float,
            default=None,
            help="Tune FP8 per_1x128 / per_Token / per_Tensor SiLU MoE with this "
            "gate/up clamp. Only CK "
            "2-stage candidates can clamp; results are tagged in a _tag column "
            "and must go to a separate tuned file (-o).",
        )
        self.parser.add_argument(
            "--ep-size",
            type=int,
            default=0,
            help="Tune every row on EP data: token is the dispatch buffer row "
            "count (global tokens), expert the local expert count; routes span "
            "expert * ep-size global experts with the balanced routing of "
            "amd_moe_bench_share/bench_moe.py, and only rank 0's local routes "
            "are computed, laid out as op_tests/test_moe_ep.py --ep-mode real "
            "does. Tunes flydslv2 only, or with --mxfp4-flydsl MXMOE GEMM1 "
            "with atomic/reduce layout GEMM2. 0 (default) keeps the dense data.",
        )
        self.parser.add_argument(
            "--ep-seed",
            type=int,
            default=0,
            help="With --ep-size: routing seed, as bench_moe.py --seed. With "
            "the same seed, global expert count, topk and token, the routes "
            "equal the bench's when the global expert count is a multiple of topk.",
        )

    def parse_args(self) -> argparse.Namespace:
        args = super().parse_args()
        try:
            self.swiglu_limit = normalize_swiglu_limit(args.swiglu_limit)
        except ValueError as exc:
            self.parser.error(str(exc))
        self.swiglu_limit_tag = swiglu_limit_tag(self.swiglu_limit)
        if self.swiglu_limit is not None:
            if args.grouped_gemm or args.mxfp4_flydsl or args.e2e_tune:
                self.parser.error(
                    "--swiglu_limit is only supported by the default fmoe tuner "
                    "(MXFP4 uses AITER_MXFP4_TUNE_SWIGLU_LIMIT)"
                )
            if args.run_config or args.compare:
                self.parser.error(
                    "--swiglu_limit does not support --run_config or --compare; "
                    "their benchmark and reference would run unclamped"
                )
            if os.path.abspath(self.get_out_file(args.tune_file)) == os.path.abspath(
                self.get_out_file(self.ARG_DEFAULTS["tune_file"])
            ):
                self.parser.error(
                    "--swiglu_limit results must go to a separate tuned file (-o); "
                    "they would replace unclamped rows in the default one"
                )
            if "_tag" not in self.columns:
                self.columns = self.columns + ["_tag"]
        args.ep_size = getattr(args, "ep_size", 0)
        args.ep_seed = getattr(args, "ep_seed", 0)
        if args.ep_size < 0:
            self.parser.error("--ep-size must be >= 0")
        if args.ep_seed and not args.ep_size:
            self.parser.error("--ep-seed requires --ep-size")
        if args.ep_size and (args.grouped_gemm or args.e2e_tune):
            self.parser.error(
                "--ep-size applies to the default fmoe tuner and --mxfp4-flydsl "
                "only, not --grouped-gemm / --e2e_tune"
            )
        if getattr(args, "fast_scan", False) and not self.SUPPORTS_FAST_SCAN:
            self.parser.error("--fast-scan requires --mxfp4-flydsl")
        if not hasattr(args, "fast_scan"):
            args.fast_scan = False
        mxfp4_port = (
            args.mxfp4_flydsl
            and not args.grouped_gemm
            and self.SUPPORTS_MXFP4_SEARCH_MODE
        )
        # None distinguishes an omitted mode from an explicit one.
        if args.mxfp4_search_mode is not None and not mxfp4_port:
            self.parser.error(
                "--mxfp4-search-mode requires --mxfp4-flydsl without --grouped-gemm"
            )
        if args.mxfp4_screen_topk is not None:
            if args.mxfp4_screen_topk <= 0:
                self.parser.error("--mxfp4-screen-topk must be a positive integer")
            if not mxfp4_port or _mxfp4_search_mode(args) != "staged":
                self.parser.error(
                    "--mxfp4-screen-topk requires --mxfp4-flydsl staged search "
                    "(no --fast-scan, no --mxfp4-search-mode full)"
                )
        if args.fast_scan and args.mxfp4_search_mode == "staged":
            self.parser.error(
                "--fast-scan applies to pruned or full search, not "
                "--mxfp4-search-mode staged"
            )
        args.mxfp4_search_mode = _mxfp4_search_mode(args)
        args.mxfp4_screen_topk = args.mxfp4_screen_topk or 2
        return args

    def _require_supported_fast_scan(self, args):
        if getattr(args, "fast_scan", False) and not self.SUPPORTS_FAST_SCAN:
            raise ValueError("--fast-scan requires --mxfp4-flydsl")

    weight_quant = staticmethod(weight_quant)
    generate_data = staticmethod(generate_data)
    generate_asm_stage1 = staticmethod(generate_asm_stage1)
    generate_data_2stages = staticmethod(generate_data_2stages)
    generate_opus_sorted_stage2_data = staticmethod(generate_opus_sorted_stage2_data)
    generate_data_1stage = staticmethod(generate_data_1stage)
    run_torch_moe_stage1 = staticmethod(run_torch_moe_stage1)
    run_torch_moe_stage2 = staticmethod(run_torch_moe_stage2)
    run_torch_moe_stage1_ref = staticmethod(run_torch_moe_stage1_ref)
    torch_moe_test = staticmethod(torch_moe_test)
    get_kernels_dict = staticmethod(get_kernels_dict)
    run_asm_stage1 = staticmethod(run_asm_stage1)
    gen_2stages_asm1_task = staticmethod(gen_2stages_asm1_task)
    run_1stage_fmoe_g1u1_tkw1 = staticmethod(run_1stage_fmoe_g1u1_tkw1)
    run_1stage_fmoe_fp8_blockscale_g1u1 = staticmethod(
        run_1stage_fmoe_fp8_blockscale_g1u1
    )
    run_1stage_fmoe_g1u1 = staticmethod(run_1stage_fmoe_g1u1)
    get_1stage_fmoe_func = staticmethod(get_1stage_fmoe_func)
    get_1stage_file_info = staticmethod(get_1stage_file_info)
    gen_1stage_asm_task = staticmethod(gen_1stage_asm_task)
    cktile_moe_stage1_out = staticmethod(cktile_moe_stage1_out)
    cktile_moe_stage2_out = staticmethod(cktile_moe_stage2_out)
    _gen_2stages_task_cktile = staticmethod(_gen_2stages_task_cktile)
    ck_moe_stage1_fwd_out = staticmethod(ck_moe_stage1_fwd_out)
    ck_moe_stage2_fwd_out = staticmethod(ck_moe_stage2_fwd_out)
    run_flydsl_stage1_out = staticmethod(run_flydsl_stage1_out)
    run_flydsl_stage2_out = staticmethod(run_flydsl_stage2_out)
    run_a16w_stage1_sorted_ref = staticmethod(run_a16w_stage1_sorted_ref)
    gen_flydsl_2stages_task = staticmethod(gen_flydsl_2stages_task)
    generate_v2_stage1_data = staticmethod(generate_v2_stage1_data)
    run_flydsl_v2_stage1_out = staticmethod(run_flydsl_v2_stage1_out)
    generate_v2_stage2_data = staticmethod(generate_v2_stage2_data)
    run_flydsl_v2_stage2_out = staticmethod(run_flydsl_v2_stage2_out)
    run_v2_stage2_ref = staticmethod(run_v2_stage2_ref)
    run_v2_stage1_sorted_ref = staticmethod(run_v2_stage1_sorted_ref)
    gen_flydsl_v2_2stages_task = staticmethod(gen_flydsl_v2_2stages_task)
    gen_flydsl_i4_2stages_task = staticmethod(gen_flydsl_i4_2stages_task)
    run_opus_stage1_out = staticmethod(run_opus_stage1_out)
    run_opus_stage2_out = staticmethod(run_opus_stage2_out)
    gen_opus_2stages_task = staticmethod(gen_opus_2stages_task)
    torch_moe_tkw1 = staticmethod(torch_moe_tkw1)
    torch_moe_2stages = staticmethod(torch_moe_2stages)
    torch_moe_blockscale = staticmethod(torch_moe_blockscale)

    def gen_2stages_task(self, key, blockMs):
        # Not an alias: every call has to use this tuner's swiglu_limit.
        return gen_2stages_task(key, blockMs, self.swiglu_limit)

    def calculate(self, results, bpes=(1, 1, 2)):
        key, stage, _kernelName, _block_m, us, _err = results
        (
            _gfx,
            _cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            _act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            _q_type,
            use_g1u1,
            _doweight_stage1,
        ) = key
        if us == self.INVALID_TIME or us == self.INF_TIME:
            return 0, 0
        flop = 0
        data_bytes = 0
        if stage == "stage1":
            ## gemm1
            # input [token, topk, inter_dim]
            # weight [exprt, 2*inter_dim, model_dim]
            m = token
            k = model_dim
            if use_g1u1:
                n = inter_dim * 2
            else:
                n = inter_dim
            flop = m * n * k * topk * 2
            data_bytes = (
                m * k * self.get_bpe(q_dtype_a)
                + m * n * self.get_bpe(dtype)
                + k * n * self.get_bpe(q_dtype_w) * expert
            )
        elif stage == "stage2":
            ## gemm2
            m = token
            n = model_dim
            k = inter_dim
            b = topk
            # input [token, topk, inter_dim]
            # weight [exprt, dim, inter_dim]
            flop = b * m * n * k * 2
            data_bytes = (
                m * k * self.get_bpe(q_dtype_a) * topk
                + m * n * self.get_bpe(dtype)
                + k * n * self.get_bpe(q_dtype_w) * expert
            )
        else:
            if use_g1u1:
                n = inter_dim * 2
            else:
                n = inter_dim
            flop = (
                token * n * model_dim * topk * 2
                + topk * token * model_dim * inter_dim * 2
            )
            data_bytes = (
                token * model_dim * self.get_bpe(q_dtype_a)
                + n * model_dim * self.get_bpe(q_dtype_w) * expert
                + inter_dim * model_dim * self.get_bpe(q_dtype_w) * expert
                + token * model_dim * self.get_bpe(dtype)
            )  # Rough Estimate
        tflops = round(flop / (us * 1000000), 2)
        bw = round(data_bytes / (us * 1e-6) / 1e9, 2)
        return tflops, bw

    def run_config(self, args, target_fused_moe=None, config_string=""):
        from aiter.fused_moe import fused_moe, fused_topk
        from aiter.test_common import checkAllclose, run_perftest

        if target_fused_moe is None:
            target_fused_moe = fused_moe
        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            token = int(row["token"])
            model_dim = int(row["model_dim"])
            inter_dim = int(row["inter_dim"])
            expert = int(row["expert"])
            topk = int(row["topk"])
            act_type = eval(row["act_type"])
            dtype = eval(row["dtype"])
            q_dtype_a = dtypes.normalize_fp8_dtype(eval(row["q_dtype_a"]))
            q_dtype_w = dtypes.normalize_fp8_dtype(eval(row["q_dtype_w"]))
            q_type = eval(row["q_type"])
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
            use_g1u1 = bool(row["use_g1u1"])
            doweight_stage1 = bool(row["doweight_stage1"])
            limit_env = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT")
            swiglu_limit = None if limit_env in (None, "") else float(limit_env)
            # fused_moe overrides the activation quant dtype at runtime for
            # per_1x32 fp4-weight MoE (gate_mode defaults to SEPARATED, which
            # run_config does not override): Silu -> fp4, Swiglu -> bf16/fp4 by M.
            # The config's nominal q_dtype_a (e.g. a8w4=fp8) is ignored, so the
            # kernel actually runs a4w4. Mirror that here so the reference and the
            # kernel weight layout match (else outputs are uncorrelated).
            # Source of truth: aiter/fused_moe.py q_dtype_a selection.
            eff_q_dtype_a = q_dtype_a
            if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp4x2:
                if act_type == ActivationType.Swiglu:
                    eff_q_dtype_a = dtypes.bf16 if token < 256 else dtypes.fp4x2
                else:
                    eff_q_dtype_a = dtypes.fp4x2
            shape_str = (
                f"({token}, {model_dim}, {inter_dim}, E={expert}, topk={topk}, "
                f"{row['act_type']}, {row['dtype']}, {row['q_dtype_a']}, "
                f"{row['q_dtype_w']}, {row['q_type']}, g1u1={use_g1u1}, "
                f"dw_s1={doweight_stage1})"
            )
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            kernel_us = None
            if "us" in row and pd.notna(row["us"]):
                try:
                    kernel_us = float(row["us"])
                except (TypeError, ValueError):
                    kernel_us = None
            try:
                torch.manual_seed(0)
                hidden = (
                    torch.randn((token, model_dim), dtype=dtype, device="cuda") / 10
                )
                if use_g1u1:
                    w1 = (
                        torch.randn(
                            (expert, inter_dim * 2, model_dim),
                            dtype=dtype,
                            device="cuda",
                        )
                        / 10
                    )
                else:
                    w1 = (
                        torch.randn(
                            (expert, inter_dim, model_dim), dtype=dtype, device="cuda"
                        )
                        / 10
                    )
                w2 = torch.randn(
                    (expert, model_dim, inter_dim), dtype=dtype, device="cuda"
                )
                w1_qt, w1_scale = self.weight_quant(w1, q_type, quant_dtype=q_dtype_w)
                w2_qt, w2_scale = self.weight_quant(w2, q_type, quant_dtype=q_dtype_w)
                if q_dtype_w is not dtypes.fp4x2:
                    w1_qt = w1_qt.view(w1.shape)
                    w2_qt = w2_qt.view(w2.shape)
                else:
                    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
                    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

                # Match the production/test path used by op_tests/test_moe_2stage.py.
                w1_qt_fmoe = w1_qt
                w2_qt_fmoe = w2_qt
                w1_scale_fmoe = w1_scale
                w2_scale_fmoe = w2_scale
                if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
                    # a16wi4 OLD-kernel preshuffle -- match production/test_moe_2stage.py.
                    w1_qt_fmoe = (
                        pack_int8_to_packed_int4(
                            shuffle_weight(w1_qt_fmoe.view(dtypes.i8), (16, 16))
                        )
                        .view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
                        .view(dtypes.i4x2)
                    )
                    w2_qt_fmoe = (
                        pack_int8_to_packed_int4(
                            shuffle_weight(w2_qt_fmoe.view(dtypes.i8), (16, 16))
                        )
                        .view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
                        .view(dtypes.i4x2)
                    )
                    w1_scale_fmoe = (
                        shuffle_scale_for_int4(w1_scale, group_size=32)
                        .view(-1)
                        .contiguous()
                    )
                    w2_scale_fmoe = (
                        shuffle_scale_for_int4(w2_scale, group_size=32)
                        .view(-1)
                        .contiguous()
                    )
                elif q_dtype_w == torch.int4:
                    w1_qt_fmoe = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w1_qt_fmoe, (16, 16), use_int4=True)
                        )
                    )
                    w2_qt_fmoe = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w2_qt_fmoe, (16, 16), use_int4=True)
                        )
                    )
                    w1_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w1_scale)
                        if w1_scale is not None
                        else None
                    )
                    w2_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w2_scale)
                        if w2_scale is not None
                        else None
                    )
                elif (
                    q_type == QuantType.per_1x32
                    and q_dtype_a in [dtypes.bf16, dtypes.fp16, dtypes.fp8]
                    and q_dtype_w == dtypes.fp4x2
                ):
                    # a16w4 / a8w4 (16-bit or fp8 activation, fp4 weight): the
                    # weight layout follows the *config* activation dtype, not the
                    # runtime-effective one (mirror op_tests/test_moe_2stage.py):
                    # a16w4 takes the standard GGUU W1, a8w4 the gate/up-
                    # interleaved GUGU one, as in the tuning path.
                    w1_gu = q_dtype_a == dtypes.fp8
                    w1_qt_fmoe = shuffle_weight_a16w4(w1_qt_fmoe, 16, w1_gu)
                    w1_scale_fmoe = shuffle_scale_a16w4(w1_scale, expert, w1_gu)
                    w2_qt_fmoe = shuffle_weight_a16w4(w2_qt_fmoe, 16, False)
                    w2_scale_fmoe = shuffle_scale_a16w4(w2_scale, expert, False)
                elif (
                    q_type == QuantType.per_1x32
                    and q_dtype_a == dtypes.fp4x2
                    and q_dtype_w == dtypes.fp4x2
                ):
                    # a4w4 (fp4 activation/weight): standard (16,16) weight shuffle
                    # + e8m0 scale shuffle (mirror test_moe_2stage preshuffle path).
                    w1_qt_fmoe = shuffle_weight(w1_qt_fmoe, layout=(16, 16))
                    w2_qt_fmoe = shuffle_weight(w2_qt_fmoe, layout=(16, 16))
                    w1_scale_fmoe = fp4_utils.e8m0_shuffle(w1_scale)
                    w2_scale_fmoe = fp4_utils.e8m0_shuffle(w2_scale)
                elif (
                    q_type == QuantType.per_1x32
                    and q_dtype_a == dtypes.fp8
                    and q_dtype_w == dtypes.fp8
                ):
                    # mxfp8 (a8w8): gate-up interleaved fp8 weight; w1 scale uses the
                    # a16w4 interleave, w2 scale uses plain e8m0 (mirror
                    # op_tests/test_moe_2stage.py is_mxfp8).
                    w1_qt_fmoe = shuffle_weight_a16w4(w1_qt_fmoe, 16, True)
                    w1_scale_fmoe = shuffle_scale_a16w4(w1_scale, expert, True)
                    w2_qt_fmoe = shuffle_weight_a16w4(w2_qt_fmoe, 16, False)
                    w2_scale_fmoe = fp4_utils.e8m0_shuffle(w2_scale)
                elif q_dtype_w != dtypes.fp4x2:
                    w1_qt_fmoe = shuffle_weight(w1_qt_fmoe, (16, 16))
                    w2_qt_fmoe = shuffle_weight(w2_qt_fmoe, (16, 16))
                    w1_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w1_scale)
                        if w1_scale is not None
                        else None
                    )
                    w2_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w2_scale)
                        if w2_scale is not None
                        else None
                    )
                else:
                    w1_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w1_scale)
                        if w1_scale is not None
                        else None
                    )
                    w2_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w2_scale)
                        if w2_scale is not None
                        else None
                    )

                w1_qt_fmoe.is_shuffled = True
                w2_qt_fmoe.is_shuffled = True

                score = torch.randn((token, expert), dtype=dtype, device="cuda")
                topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
                if q_type == QuantType.per_1x128:
                    a1_qt, a1_scale = aiter.pertoken_quant(
                        hidden.view(token, -1, 128), quant_dtype=q_dtype_a
                    )
                    a1_qt = a1_qt.view(token, model_dim)
                    a1_scale = a1_scale.squeeze(-1)
                elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
                    a1_qt = hidden.to(dtypes.bf16)
                    a1_scale = None
                elif (
                    q_type == QuantType.per_1x32
                    and eff_q_dtype_a in [dtypes.bf16, dtypes.fp16]
                    and q_dtype_w == dtypes.fp4x2
                ):
                    # a16w4 (bf16 activation): reference runs activation in bf16.
                    a1_qt = hidden.to(dtype)
                    a1_scale = None
                elif (
                    q_type == QuantType.per_1x32
                    and q_dtype_a == dtypes.fp8
                    and q_dtype_w == dtypes.fp8
                ):
                    # mxfp8 (a8w8): kernel quantizes the activation internally; the
                    # reference runs activation in bf16 (mirror test_moe_2stage).
                    a1_qt = hidden.to(dtype)
                    a1_scale = None
                else:
                    # Use the *effective* activation dtype (what fused_moe runs),
                    # not the config's nominal q_dtype_a. For Silu+SEPARATED fp4
                    # weights this is fp4x2 (a4w4), so the reference quantizes the
                    # activation to fp4 to match the kernel.
                    torch_quant = aiter.get_torch_quant(q_type)
                    a1_qt, a1_scale = torch_quant(hidden, quant_dtype=eff_q_dtype_a)

                out, us = run_perftest(
                    target_fused_moe,
                    hidden,
                    w1_qt_fmoe,
                    w2_qt_fmoe,
                    topk_weights,
                    topk_ids,
                    activation=act_type,
                    quant_type=q_type,
                    doweight_stage1=doweight_stage1,
                    # The torch reference runs SiTUv2 with the model's betas;
                    # fused_moe defaults to 1.0. Without this every Situv2 shape
                    # reports logits_diff ~0.14 (out_norm/ref_norm ~0.61)
                    # regardless of which kernel it dispatched.
                    beta=(
                        DEFAULT_SITUV2_BETA
                        if act_type == ActivationType.Situv2
                        else None
                    ),
                    linear_beta=(
                        DEFAULT_SITUV2_LINEAR_BETA
                        if act_type == ActivationType.Situv2
                        else None
                    ),
                    swiglu_limit=swiglu_limit,
                    w1_scale=w1_scale_fmoe,
                    w2_scale=w2_scale_fmoe,
                    dtype=dtype,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                # a16wi4: per_1x32_i4_quant stores int4 in an int8 container.
                # The torch reference detects int4 weights by the i4x2 dtype, so
                # pass an i4x2-reinterpreted view to the reference only (the kernel
                # path above consumes the int8 w*_qt). Mirrors op_tests/test_moe_2stage.py.
                w1_qt_ref, w2_qt_ref = w1_qt, w2_qt
                if q_type == QuantType.per_1x32 and w1_qt.dtype == dtypes.i8:
                    w1_qt_ref = w1_qt.view(dtypes.i4x2)
                    w2_qt_ref = w2_qt.view(dtypes.i4x2)
                ref = self.torch_moe_2stages(
                    a1_qt,
                    w1_qt_ref,
                    w2_qt_ref,
                    topk_weights,
                    topk_ids,
                    a1_scale=a1_scale,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    dtype=dtype,
                    activation=act_type,
                    quant_type=q_type,
                    doweight_stage1=doweight_stage1,
                    swiglu_limit=swiglu_limit,
                )
                if not _all_finite(out) or not _all_finite(ref):
                    diag = tensor_compare_diagnostics(ref, out)
                    status = f"error:nonfinite output or reference; {diag}"
                    err_ratio = 1.0
                elif out.count_nonzero() == 0 and ref.count_nonzero() > 0:
                    diag = tensor_compare_diagnostics(ref, out)
                    status = (
                        "error:output is all zeros (kernel produced no output); "
                        f"{diag}"
                    )
                    err_ratio = 1.0
                else:
                    err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                    # Element-wise err_ratio (atol/rtol) is overly strict for lossy
                    # fp4/fp8 MoE: even a correct kernel differs on most elements
                    # vs the higher-precision reference. op_tests/test_moe_2stage.py
                    # judges these paths by cosine similarity (logits_diff) instead,
                    # so accept a shape if either metric passes. A genuinely wrong
                    # (uncorrelated) kernel still has large logits_diff and fails.
                    _x = out.to(dtypes.fp32)
                    _y = ref.to(dtypes.fp32)
                    _den = (_x * _x + _y * _y).sum()
                    logits_diff = (
                        float(1.0 - (2.0 * (_x * _y).sum() / _den).item())
                        if _den.item() != 0
                        else 0.0
                    )
                    cos_tol = float(os.environ.get("AITER_RUN_CONFIG_COS_TOL", "0.01"))
                    if err_ratio <= allowed_err_ratio or logits_diff <= cos_tol:
                        status = "ok"
                    else:
                        diag = tensor_compare_diagnostics(ref, out)
                        status = (
                            f"mismatch:err_ratio={err_ratio:.6g}"
                            f"(>{allowed_err_ratio_desc}),"
                            f"logits_diff={logits_diff:.6g}(>{cos_tol}),"
                            f"diagnostics={diag}"
                        )
                results.append(
                    {
                        "shape": shape_str,
                        "e2e_us": us,
                        "kernel_us": kernel_us,
                        "status": status,
                        "err_ratio": err_ratio,
                    }
                )
            except AssertionError as e:
                # checkAllclose raises on catastrophic error (max_delta exceeds threshold).
                # Demote to mismatch only when err_ratio also exceeds allowed_err_ratio,
                # i.e. it is genuinely a precision issue. Otherwise keep as error so that
                # rare outliers (e.g. a few NaN/zeros) are not silently swept under mismatch.
                msg = str(e)
                if "catastrophic" in msg:
                    import re

                    m = re.search(r"(\d+\.\d+)%", msg)
                    err_ratio = float(m.group(1)) / 100 if m else 1.0
                    if err_ratio > allowed_err_ratio:
                        results.append(
                            {
                                "shape": shape_str,
                                "e2e_us": us if "us" in locals() else -1,
                                "kernel_us": kernel_us,
                                "status": (
                                    f"mismatch:catastrophic err_ratio={err_ratio:.6g}"
                                    f"(>{allowed_err_ratio_desc})"
                                ),
                            }
                        )
                    else:
                        results.append(
                            {
                                "shape": shape_str,
                                "e2e_us": -1,
                                "kernel_us": kernel_us,
                                "status": f"error:{e}",
                            }
                        )
                else:
                    results.append(
                        {
                            "shape": shape_str,
                            "e2e_us": -1,
                            "kernel_us": kernel_us,
                            "status": f"error:{e}",
                        }
                    )
            except Exception as e:  # noqa: BLE001
                results.append(
                    {
                        "shape": shape_str,
                        "e2e_us": -1,
                        "kernel_us": kernel_us,
                        "status": f"error:{e}",
                    }
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        self._require_supported_fast_scan(args)
        mp_num = args.mp
        blockMs = [16, 32, 64, 128]
        keys = self.keys
        tasks = []
        tasks_ck = []
        task_1stage = []
        in_data = []
        for line in untunedf[keys].values:
            (
                gfx,
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            ) = line
            dtype = eval(dtype)
            q_dtype_a = dtypes.normalize_fp8_dtype(eval(q_dtype_a))
            q_dtype_w = dtypes.normalize_fp8_dtype(eval(q_dtype_w))
            q_type = eval(q_type)
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
            print("\nStart tuning", line)
            if get_gfx() not in ["gfx950"] and q_type in [aiter.QuantType.per_1x32]:
                print(f"{q_type} is not supported on {get_gfx()}")
                return []
            if not use_g1u1:
                print("no moe solution(g1u0) can tune for ", line)
                continue
            act_type = eval(act_type)
            info = (
                gfx,
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            )
            ep = args.ep_size
            if ep:
                if token < ep or token % ep:
                    print(
                        f"[ep] skip token={token}: EP{ep} dispatch buffer rows "
                        f"are a positive multiple of {ep}"
                    )
                    continue
                ep_seed = args.ep_seed
                recv, routes, active = _ep_route_stats(token, expert, topk, ep, ep_seed)
                print(
                    f"[ep] token={token} seed={ep_seed}: total_recv={recv} "
                    f"local_routes={routes} active_experts={active}/{expert}"
                )
                # Runtime EP rejects Opus stage2 configs (and the other families'
                # EP paths are not modeled here), so EP mode tunes flydslv2 only.
                ep_tasks = self.gen_flydsl_v2_2stages_task(info, blockMs, ep, ep_seed)
                print(
                    f"[ep] ep_size={ep}: {len(ep_tasks)} flydslv2 tasks, "
                    f"TUNE_ONLY/OPUS_ONLY ignored"
                )
                tasks_ck.extend(ep_tasks)
                continue
            _opus_only = os.environ.get("OPUS_ONLY", "0") == "1"
            # TUNE_ONLY: comma-list subset gate to isolate sources, e.g.
            #   TUNE_ONLY=flydsl  -> only gen_flydsl_2stages_task
            #   TUNE_ONLY=opus    -> only gen_opus_2stages_task
            # unset = flydslv2 if available, otherwise all (OPUS_ONLY applies);
            # empty = all (subject to OPUS_ONLY / OPUS_SKIP_CKTILE).
            tune_only = os.environ.get("TUNE_ONLY")
            v2_tasks = None
            if tune_only is None:
                v2_tasks = self.gen_flydsl_v2_2stages_task(info, blockMs)
                tune_only = "flydslv2" if v2_tasks else ""
            _tune_only = {s for s in tune_only.split(",") if s}
            if self.swiglu_limit is not None:
                if not (
                    q_type in RUNTIME_SILU_LIMIT_QUANT_TYPES
                    and act_type == ActivationType.Silu
                    and q_dtype_a == dtypes.fp8
                    and q_dtype_w == dtypes.fp8
                ):
                    raise ValueError(
                        "--swiglu_limit supports only FP8 per_1x128 / per_Token / "
                        f"per_Tensor SiLU, got {line}"
                    )
                # ASM, FlyDSL and Opus SiLU kernels for these quant types have no clamp.
                print(f"--swiglu_limit={self.swiglu_limit:g}: tuning CK 2-stage only")
                _opus_only, _tune_only = False, {"cktile"}

            def _want(name, _opus_only=_opus_only, _tune_only=_tune_only):
                if _tune_only:
                    return name in _tune_only
                if _opus_only:
                    return name == "opus"
                return True

            if _want("asm"):
                tasks.extend(self.gen_2stages_asm1_task(info, blockMs))
            if _want("cktile") and os.environ.get("OPUS_SKIP_CKTILE", "0") != "1":
                tasks_ck.extend(self.gen_2stages_task(info, blockMs))
            if _want("flydsl"):
                tasks_ck.extend(self.gen_flydsl_2stages_task(info, blockMs))
            if _want("flydslv2"):
                tasks_ck.extend(
                    v2_tasks
                    if v2_tasks is not None
                    else self.gen_flydsl_v2_2stages_task(info, blockMs)
                )
            if _want("flydsli4"):
                tasks_ck.extend(self.gen_flydsl_i4_2stages_task(info, blockMs))
            if _want("opus"):
                tasks_ck.extend(
                    self.gen_opus_2stages_task(
                        info,
                        blockMs,
                    )
                )
            if _want("asm"):
                task_1stage.extend(self.gen_1stage_asm_task(info))
            if tasks is None and tasks_ck is None and task_1stage is None:
                print("no moe solution can tune for ", line)
                continue
            print(
                f"stage1 asm tasks is {len(tasks)}, tasks_ck is {len(tasks_ck)}, task_1stage is {len(task_1stage)}"
            )
        all_tasks = tasks + tasks_ck + task_1stage
        kernel_regex = os.environ.get("TUNE_MOE_KERNEL_REGEX")
        if kernel_regex:
            import re

            pattern = re.compile(kernel_regex)
            all_tasks = [task for task in all_tasks if pattern.search(task[0][2])]
            print(f"filtered tuning tasks with {kernel_regex!r}: {len(all_tasks)}")
        # Record dispatched cases
        dispatched = {}
        for i, task in enumerate(all_tasks):
            tag = task[0]  # (info, stage, kname, blockM)
            dispatched[i] = tag
            if args.verbose:
                # task_1stage uses (info, stage, kname, blockM, flat_flag); others
                # use (info, stage, kname, blockM). Accept both lengths.
                _, stage, kname, blockM, *_extra = tag
                print(f"  [dispatch] task {i}: {stage} {kname} blockM={blockM}")

        in_data.append((len(all_tasks), ()))
        rets = []
        if len(all_tasks) > 0:
            ### shape_grouped should be False as multiple stages.
            # fast_mode here means "return every candidate"; it is the only
            # timing pass, so it picks the winner.
            rets = mp_tuner(
                all_tasks,
                in_data,
                mp_num,
                True,
                False,
                timeout=args.timeout,
                verbose=args.verbose,
                policy=self.policy_for(args),
                final_timing=True,
                default_device="cuda",
            )

        # Identify failed cases
        if rets:
            failed_cases = []
            for i, ret in enumerate(rets):
                info, us, err = ret
                if us == float("inf") or us == -1:
                    tag = dispatched.get(i, info)
                    failed_cases.append((i, tag, us, err))
            if failed_cases:
                print(f"\n[tune] {len(failed_cases)} of {len(rets)} tasks failed:")
                for i, tag, us, err in failed_cases:
                    # task_1stage tag is (info, stage, kname, blockM, flat_flag);
                    # 2-stage tag is (info, stage, kname, blockM). Accept both.
                    _, stage, kname, blockM, *_extra = tag
                    reason = "timeout/hang" if us == float("inf") else "crash/error"
                    print(f"  task {i}: {stage} {kname} blockM={blockM} -> {reason}")
            else:
                print(f"\n[tune] all {len(rets)} tasks completed successfully")

        if not rets:
            print("no shape to tune or no solution found")
            return []
        else:
            return rets

    def result_to_csv(self, results, file, concat=False):
        old_tunedf = self.get_tuned_gemm_list(file)
        other_tag_df = None
        if self.swiglu_limit_tag:
            # Rows tuned for another swiglu_limit share the keys; keep them as is.
            old_tags = (
                old_tunedf["_tag"].fillna("").astype(str)
                if "_tag" in old_tunedf.columns
                else pd.Series("", index=old_tunedf.index)
            )
            other_tag_df = old_tunedf[old_tags != self.swiglu_limit_tag]
            old_tunedf = old_tunedf[old_tags == self.swiglu_limit_tag]

        for col in self.columns:
            if col not in old_tunedf.columns:
                # Migrate legacy tuned files lacking a gfx column: infer from
                # cu_num so old rows stay matchable instead of collapsing to 0.
                if col == "gfx" and "cu_num" in old_tunedf.columns:
                    old_tunedf[col] = old_tunedf["cu_num"].map(gfx_from_cu_num)
                else:
                    old_tunedf[col] = 0

        # With a swiglu_limit every tagged row is kept: this tag's rows are
        # merged below, the others are other_tag_df.
        if not self.swiglu_limit_tag:
            old_tunedf = self._apply_tag_rule(old_tunedf)

        resultdf = self.update_tunedf(old_tunedf, results)
        self.success = pd.concat([self.success, results], ignore_index=True)
        resultdf["run_1stage"] = resultdf["run_1stage"].astype(int)
        if "xbf16" not in resultdf.columns:
            resultdf["xbf16"] = 0
        resultdf["xbf16"] = resultdf["xbf16"].fillna(0).astype(int)
        if "flat" not in resultdf.columns:
            resultdf["flat"] = AsmMoeMode.SORTED.value
        resultdf["flat"] = resultdf["flat"].fillna(AsmMoeMode.SORTED.value).astype(int)
        if "nt" not in resultdf.columns:
            resultdf["nt"] = 0
        resultdf["nt"] = resultdf["nt"].fillna(0).astype(int)
        if results is not None:
            resultdf = resultdf.astype(str).drop_duplicates(
                subset=self.keys,
                keep="last",
            )
        if other_tag_df is not None and not other_tag_df.empty:
            resultdf = pd.concat(
                [other_tag_df.astype(str), resultdf], ignore_index=True
            )
        # Canonical column order (self.columns, so gfx stays the first column).
        # Without this an incremental tune of a legacy file would append gfx at
        # the back.
        ordered_cols = [c for c in self.columns if c in resultdf.columns]
        ordered_cols += [c for c in resultdf.columns if c not in ordered_cols]
        resultdf = resultdf[ordered_cols]
        resultdf.to_csv(file, index=False)

    @staticmethod
    def _pair_nt_agnostic(profileDF, kernel_col):
        # Only the CK 2-stage instances read the hint; the rest are measured
        # once, so give them a copy on the nt=1 side of the merge. With the
        # sweep off there is no such side and the copy would survive as a tie.
        if not (profileDF["nt"] == 1).any():
            return profileDF
        agnostic = profileDF[
            ~profileDF[kernel_col].astype(str).str.startswith("moe_ck2stages")
        ]
        if agnostic.empty:
            return profileDF
        return pd.concat([profileDF, agnostic.assign(nt=1)], ignore_index=True)

    def post_process(self, results, args, topk=-1, fast_mode=False):
        profileDF = []
        profileDF = []
        prorfiles = []
        bests = []
        from collections import defaultdict

        policy = self.policy_for(args)
        if not getattr(args, "profile_file", ""):
            ranked_file = policy.profile_file(self.get_out_file(args.tune_file))
            if ranked_file:
                args.profile_file = ranked_file
                print(f"power ranking will be written to {args.profile_file}")

        ##group results by info[0](key)
        grouped_rets = defaultdict(list)
        for el in results:
            info, us, max_err_ratio = el
            grouped_rets[tuple(info[0])].append(
                (info[1:], us, max_err_ratio, getattr(el, "watts", float("nan")))
            )
        grouped_results = grouped_rets.items()
        for key, rets in grouped_results:
            us_qs_cache = {}
            (
                gfx,
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            ) = key
            import re

            profileDF = []
            for tail, us, err, watts in rets:
                stage = tail[0]
                kernelName = tail[1]
                block_m = tail[2]
                flat_flag = int(tail[3]) if len(tail) > 3 else AsmMoeMode.SORTED.value
                v2_flag = int(tail[4]) if len(tail) > 4 else 0
                nt_flag = int(tail[5]) if len(tail) > 5 else 0
                tflops, bw = self.calculate((key, stage, kernelName, block_m, us, err))
                row_ksplit = 0
                sk_match = re.search(r"_sk(\d+)$", str(kernelName))
                if sk_match:
                    row_ksplit = int(sk_match.group(1))
                    kernelName = re.sub(r"_sk\d+$", "", kernelName)
                profileDF.append(
                    [
                        stage,
                        gfx,
                        cu_num,
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        act_type,
                        dtype,
                        q_dtype_a,
                        q_dtype_w,
                        q_type,
                        use_g1u1,
                        doweight_stage1,
                        block_m,
                        row_ksplit,
                        us,
                        kernelName,
                        err,
                        tflops,
                        bw,
                        flat_flag,
                        v2_flag,
                        nt_flag,
                        watts,
                    ]
                )

            profileDF = pd.DataFrame(
                profileDF,
                columns=["stage"]
                # + ["cu_num"]
                + self.keys
                + [
                    "block_m",
                    "ksplit",
                    "us",
                    "kernelName",
                    "err",
                    "tflops",
                    "bw",
                    "flat",
                    "v2",
                    "nt",
                    "watts",
                ],
            )
            # Stage watts are a power-tune column; the latency profile files
            # keep main's schema.
            if not policy.measures_power:
                profileDF = profileDF.drop(columns="watts")
            prorfiles.append(profileDF)

            dump_tag = _candidate_dump_tag(
                gfx,
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            )
            _dump_prefilter_candidates(profileDF, args, dump_tag)
            if args.verbose:
                invalid_time = profileDF["us"].isin([float("inf"), float("-inf"), -1])
                passed = profileDF["err"] <= args.errRatio
                eligible = passed & ~invalid_time
                print(
                    "  candidate summary: "
                    f"generated={len(profileDF)} passed={int(passed.sum())} "
                    f"timed={int((~invalid_time).sum())} "
                    f"filtered={int((~eligible).sum())}"
                )

            ## remove invalid candidate
            profileDF = profileDF[
                (profileDF["us"] != float("-inf"))
                & (profileDF["us"] != float("inf"))
                & (profileDF["us"] != -1)
                & (profileDF["err"] <= args.errRatio)
            ]
            # a16w16 stage1 writes a SORTED intermediate that only the a16w16
            # stage2 reads, and CK stage2 cannot read it; both share BF16
            # QuantType.No keys, so dedup and pair them on separate sides.
            profileDF["a16w16"] = (
                profileDF["kernelName"]
                .astype(str)
                .str.startswith(
                    ("flydsl_moe1_abf16_wbf16_", "flydsl_moe2_abf16_wbf16_")
                )
                .astype(int)
            )
            # Dedup fused (_fp4/_fp8) and non-fused stage1 separately: the fairness
            # penalty below is applied only to non-fused, so dedup must not drop the
            # fused kernel by raw us before that penalty runs. Keep v2 and non-v2
            # candidates separate as well because they use different layouts.
            profileDF["_is_fused"] = (
                profileDF["kernelName"].astype(str).str.endswith(("_fp4", "_fp8"))
            )
            profileDF = policy.sort_frame(profileDF)
            profileDF = profileDF.drop_duplicates(
                ["stage", "block_m", "flat", "nt", "v2", "a16w16", "_is_fused"],
                keep="first",
            ).drop(columns=["_is_fused"])
            stage1_profileDF = profileDF[profileDF["stage"] == "stage1"].drop(
                columns=["stage", "flat"]
            )

            stage1_profileDF = stage1_profileDF.rename(
                columns={
                    "kernelName": "kernelName1",
                    "err": "err1",
                    "us": "us1",
                    "tflops": "tflops1",
                    "bw": "bw1",
                    "watts": "watts1",
                }
            )
            stage1_profileDF = self._pair_nt_agnostic(stage1_profileDF, "kernelName1")
            stage2_profileDF = profileDF[profileDF["stage"] == "stage2"].drop(
                columns=["stage", "ksplit", "flat"]
            )
            stage2_profileDF = stage2_profileDF.rename(
                columns={
                    "kernelName": "kernelName2",
                    "err": "err2",
                    "us": "us2",
                    "tflops": "tflops2",
                    "bw": "bw2",
                    "watts": "watts2",
                }
            )
            stage2_profileDF = self._pair_nt_agnostic(stage2_profileDF, "kernelName2")
            if (stage1_profileDF.shape[0] == 0 and stage2_profileDF.shape[0] != 0) or (
                stage1_profileDF.shape[0] != 0 and stage2_profileDF.shape[0] == 0
            ):
                print(
                    "Error: please check errRatio, stage1 and stage2 should be valid together!"
                )
            asm_1stage_mask = profileDF["stage"].isin(
                ["asm_1stage", "asm_1stage_xbf16"]
            )
            asm_1stage_profileDF = profileDF[asm_1stage_mask].copy()
            asm_1stage_profileDF["xbf16"] = (
                asm_1stage_profileDF["stage"] == "asm_1stage_xbf16"
            ).astype(int)
            asm_1stage_profileDF = asm_1stage_profileDF.drop(columns=["stage"])
            asm_1stage_profileDF = asm_1stage_profileDF.rename(
                columns={
                    "kernelName": "kernelName1",
                    "err": "err1",
                    "us": "us1",
                    "tflops": "tflops1",
                    "bw": "bw1",
                    "watts": "watts1",
                }
            )
            empty_1stage_profileDF = pd.DataFrame(index=asm_1stage_profileDF.index)

            empty_1stage_profileDF["kernelName2"] = None
            empty_1stage_profileDF["err2"] = 0
            empty_1stage_profileDF["us2"] = 0
            empty_1stage_profileDF["tflops2"] = 0
            empty_1stage_profileDF["bw2"] = 0
            if "watts1" in asm_1stage_profileDF.columns:
                empty_1stage_profileDF["watts2"] = float("nan")
            asm_1stage_profileDF = pd.concat(
                [asm_1stage_profileDF, empty_1stage_profileDF], axis=1
            )
            asm_1stage_profileDF["run_1stage"] = 1
            # v2 in the merge key isolates v2 gemm1<->v2 gemm2 pairing (and
            # normal<->normal); it is a per-row constant tag (0 for all
            # normal/asm rows, 1 for v2), so adding it to the key leaves
            # normal-path grouping unchanged.
            profileDF = pd.merge(
                stage1_profileDF,
                stage2_profileDF,
                on=[
                    "gfx",
                    "cu_num",
                    "token",
                    "model_dim",
                    "inter_dim",
                    "expert",
                    "topk",
                    "act_type",
                    "dtype",
                    "q_dtype_a",
                    "q_dtype_w",
                    "q_type",
                    "use_g1u1",
                    "doweight_stage1",
                    "block_m",
                    "nt",
                    "v2",
                    "a16w16",
                ],
                how="inner",
            )
            profileDF["run_1stage"] = 0
            profileDF["xbf16"] = 0
            profileDF["flat"] = AsmMoeMode.SORTED.value
            profileDF = pd.concat(
                [profileDF, asm_1stage_profileDF], axis=0, ignore_index=True
            )
            if len(profileDF) == 0:
                print(
                    f"no valid candidate found for {key}, please check the time or errRatio in all result file running with --profile_file"
                )
                ret = []
                ret.append(
                    [
                        gfx,
                        cu_num,
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        act_type,
                        dtype,
                        q_dtype_a,
                        q_dtype_w,
                        q_type,
                        use_g1u1,
                        doweight_stage1,
                        0,
                        0,
                        self.INVALID_TIME,
                        None,
                        1,
                        self.INVALID_TIME,
                        None,
                        1,
                        self.INVALID_TIME,
                        0,
                        0,
                        -1,
                        -1,
                        -1,
                        -1,
                    ]
                    + ([self.swiglu_limit_tag] if self.swiglu_limit_tag else [])
                )
                failedf = pd.DataFrame(ret, columns=self.columns)
                self.failed = pd.concat([self.failed, failedf], axis=0)
                continue
            if q_type == QuantType.per_1x32 and q_dtype_w != dtypes.i4x2:
                # For a4w4 (fp4 activation), a separate fp4-quant+sort step is needed
                # between stage1 (bf16 output) and stage2 (fp4 input).  Benchmark its
                # cost and add it to non-fused kernels so comparisons are fair.
                #
                # For a8w4 (fp8 activation), non-fused paths assume bf16 stage1 output
                # then a separate cast to fp8 before stage2; benchmark that cast
                # (simple .to(dtypes.fp8)) and add it to kernels whose stage1 name does
                # not end with _fp8 (those fuse the cast in stage1).
                if q_dtype_a == dtypes.fp4x2:
                    # Benchmark the same implementation fused_moe_ dispatches to
                    # (aiter.ops.quant, HIP); the triton one is never executed at
                    # runtime, so timing it skews the non-fused fairness penalty.
                    from aiter.ops.quant import (
                        fused_dynamic_mxfp4_quant_moe_sort,
                    )
                    from aiter.test_common import run_perftest

                    us_qs_cache = {}
                    for bm in profileDF["block_m"].unique():
                        bm_int = int(bm)
                        block_size = max(32, bm_int)
                        num_sorted = (
                            (token * topk + block_size - 1) // block_size
                        ) * block_size
                        dummy_act = torch.randn(
                            token * topk, inter_dim, dtype=dtype, device="cuda"
                        )
                        dummy_sorted_ids = torch.arange(
                            num_sorted, dtype=torch.int32, device="cuda"
                        )
                        dummy_num_valid = torch.tensor(
                            [token * topk], dtype=torch.int32, device="cuda"
                        )
                        _, us_qs = run_perftest(
                            fused_dynamic_mxfp4_quant_moe_sort,
                            dummy_act,
                            sorted_ids=dummy_sorted_ids,
                            num_valid_ids=dummy_num_valid,
                            token_num=token,
                            topk=topk,
                            block_size=block_size,
                        )
                        us_qs_cache[bm] = round(us_qs, 4)
                        print(
                            f"  quant_sort benchmark: blockM={bm_int}, us={us_qs_cache[bm]}"
                        )
                    profileDF["us_quant_sort"] = profileDF["block_m"].map(us_qs_cache)
                    # _fp4 kernels already fuse the fp4-quant+sort; skip cost addition.
                    # v2 pairs also fuse fp4-quant+sort into gemm1 -> skip.
                    is_fp4 = profileDF["kernelName1"].astype(str).str.endswith("_fp4")
                    is_v2 = profileDF["v2"].astype(int) == 1
                    add_qs = (
                        ~is_fp4
                        & ~is_v2
                        & (profileDF["flat"] != AsmMoeMode.POST_REDUCE.value)
                    )
                    profileDF.loc[add_qs, "us1"] = (
                        profileDF.loc[add_qs, "us1"]
                        + profileDF.loc[add_qs, "us_quant_sort"]
                    )
                    profileDF.drop(columns=["us_quant_sort"], inplace=True)
                elif q_dtype_a == dtypes.fp8:
                    from aiter.test_common import run_perftest

                    dummy_act = torch.randn(
                        token * topk, inter_dim, dtype=dtype, device="cuda"
                    )

                    def _act_to_fp8(x):
                        _scale_tmp = torch.ones(
                            [x.shape[0], x.shape[1] // 32],
                            dtype=dtypes.fp8_e8m0,
                            device=x.device,
                        )
                        return x.to(dtypes.fp8)

                    _, us_fp8_cast = run_perftest(_act_to_fp8, dummy_act)
                    us_fp8_cast = round(us_fp8_cast, 4)
                    print(f"  fp8 activation cast benchmark: us={us_fp8_cast}")
                    us_qs_cache = {}
                    for bm in profileDF["block_m"].unique():
                        us_qs_cache[bm] = us_fp8_cast
                    profileDF["us_quant_sort"] = profileDF["block_m"].map(us_qs_cache)
                    is_fp8 = profileDF["kernelName1"].astype(str).str.endswith("_fp8")
                    is_v2 = profileDF["v2"].astype(int) == 1
                    add_cast = ~is_fp8 & ~is_v2
                    profileDF.loc[add_cast, "us1"] = (
                        profileDF.loc[add_cast, "us1"]
                        + profileDF.loc[add_cast, "us_quant_sort"]
                    )
                    profileDF.drop(columns=["us_quant_sort"], inplace=True)

            has_xbf16 = "xbf16" in profileDF.columns and profileDF["xbf16"].any()
            if q_type == QuantType.per_1x128 and has_xbf16:
                from aiter.ops.quant import per_group_quant_hip
                from aiter.test_common import run_perftest

                dummy_act = torch.randn(token, model_dim, dtype=dtype, device="cuda")
                _, us_quant = run_perftest(
                    per_group_quant_hip,
                    dummy_act,
                    quant_dtype=dtypes.fp8,
                    group_size=128,
                    transpose_scale=True,
                )
                us_quant = round(us_quant, 4)
                print(f"  per_1x128 quant benchmark: us={us_quant}")
                non_xbf16 = profileDF.get("xbf16", 0) == 0
                profileDF.loc[non_xbf16, "us1"] = (
                    profileDF.loc[non_xbf16, "us1"] + us_quant
                )

            # Only no_sort/emsort avoid the host sort; post_reduce is sorted input.
            host_sort_modes = [
                mode.value for mode in AsmMoeMode if mode.requires_host_sort
            ]
            raw_route_modes = [
                mode.value for mode in AsmMoeMode if not mode.requires_host_sort
            ]
            has_flat = (
                "flat" in profileDF.columns
                and profileDF["flat"].isin(raw_route_modes).any()
            )
            if has_flat:
                from aiter.fused_moe import moe_sorting
                from aiter.test_common import run_perftest

                _topk_ids = torch.randint(
                    0, expert, (token, topk), dtype=torch.int32, device="cuda"
                )
                _topk_w = torch.rand((token, topk), dtype=dtypes.fp32, device="cuda")
                us_sort_cache = {}
                for bm in profileDF["block_m"].unique():
                    bm_int = int(bm)
                    try:
                        _, us_sort = run_perftest(
                            moe_sorting,
                            _topk_ids,
                            _topk_w,
                            expert,
                            model_dim,
                            dtype,
                            bm_int,
                        )
                        us_sort_cache[bm] = round(us_sort, 4)
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"  moe_sorting benchmark failed for block_m={bm_int}: {e}"
                        )
                        us_sort_cache[bm] = 0.0
                print(f"  moe_sorting benchmark per block_m: {us_sort_cache}")
                profileDF["us_moe_sort"] = profileDF["block_m"].map(us_sort_cache)
                non_flat = profileDF["flat"].isin(host_sort_modes)
                profileDF.loc[non_flat, "us1"] = (
                    profileDF.loc[non_flat, "us1"]
                    + profileDF.loc[non_flat, "us_moe_sort"]
                )
                profileDF.drop(columns=["us_moe_sort"], inplace=True)

            # Asymmetric head-to-head e2e to correct FLAT-vs-non-FLAT picks.
            # The additive moe_sorting fairness above already inflates non-FLAT
            # us1 (standalone benchmark over-estimates the in-context sort cost),
            # which biases the inter-class pick toward FLAT. We only intervene
            # when that bias could actually flip the outcome -- i.e. when FLAT
            # is currently the additive-fairness winner. Then we measure the
            # real sequences end-to-end and drop FLAT iff non-FLAT truly wins.
            # We never touch non-FLAT rows, so non-FLAT-vs-non-FLAT ranking
            # (additive fairness + idxmin) stays exactly as it was.
            if has_flat:
                try:
                    flat_mask = profileDF["flat"].isin(raw_route_modes)
                    nf_mask = profileDF["flat"].isin(host_sort_modes)
                    if not (flat_mask.any() and nf_mask.any()):
                        raise RuntimeError("missing FLAT or non-FLAT class")
                    # Rank candidates by us1+us2 (us2 is 0 for 1-stage rows,
                    # so this is equivalent to us1 there; for 2-stage rows
                    # we need the combined cost to know who would win idxmin).
                    flat_df = profileDF[flat_mask].copy()
                    nf_df = profileDF[nf_mask].copy()
                    flat_df["_us_total"] = flat_df["us1"] + flat_df["us2"]
                    nf_df["_us_total"] = nf_df["us1"] + nf_df["us2"]
                    best_flat_row = flat_df.sort_values("_us_total").iloc[0]
                    best_nf_row = nf_df.sort_values("_us_total").iloc[0]
                    if float(best_flat_row["_us_total"]) >= float(
                        best_nf_row["_us_total"]
                    ):
                        raise RuntimeError(
                            "non-FLAT already winning under additive fairness"
                        )
                    # Head-to-head currently supports only 1-stage non-FLAT
                    # candidates dispatched through fmoe_fp8_blockscale_g1u1
                    # (q_type==per_1x128). 2-stage non-FLAT requires a
                    # different launch sequence (ck_moe_stage1 + ck_moe_stage2)
                    # and is left to the additive-fairness ranking.
                    if int(best_nf_row.get("run_1stage", 0)) != 1:
                        raise RuntimeError(
                            "best non-FLAT is 2-stage; head-to-head needs"
                            " 1-stage non-FLAT"
                        )
                    if q_type != QuantType.per_1x128:
                        raise RuntimeError(
                            "head-to-head currently only supports per_1x128"
                        )
                    nf_block_m = int(best_nf_row["block_m"])
                    gen = FmoeTuner.generate_data_1stage(
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        act_type,
                        dtype,
                        q_dtype_a,
                        q_dtype_w,
                        q_type,
                        use_g1u1,
                        nf_block_m,
                    )
                    (
                        _input,
                        _a1_qt,
                        _w1_qt_s,
                        _w2_qt_s,
                        _sids,
                        _sws,
                        _seids,
                        _nv,
                        _moe_buf,
                        _a1_scale,
                        _w1_scale,
                        _w2_scale,
                        _w1_qt,
                        _w2_qt,
                        _topk_w,
                        _topk_ids,
                        _fc1_ss,
                        _fc2_ss,
                        _a1_scale_t,
                    ) = gen
                    _flat_kname = str(best_flat_row["kernelName1"])
                    _nf_kname = str(best_nf_row["kernelName1"])
                    nf_is_xbf16 = int(best_nf_row.get("xbf16", 0)) == 1
                    _nf_a1 = _input if nf_is_xbf16 else _a1_qt
                    _act_int = int(act_type.value)
                    # Call aiter.fmoe_fp8_blockscale_g1u1 directly with the
                    # moe_buf returned by moe_sorting -- this matches the
                    # production fused_moe path (the wrapper-based call would
                    # allocate a second moe_buf via torch.zeros, adding an
                    # aten::fill_ kernel that isn't on the production stream).

                    def _flat_seq(
                        _a1_scale_t=_a1_scale_t,
                        _act_int=_act_int,
                        _flat_kname=_flat_kname,
                        _input=_input,
                        _topk_ids=_topk_ids,
                        _topk_w=_topk_w,
                        _w1_qt_s=_w1_qt_s,
                        _w1_scale=_w1_scale,
                        _w2_qt_s=_w2_qt_s,
                        _w2_scale=_w2_scale,
                        dtype=dtype,
                        expert=expert,
                        model_dim=model_dim,
                        topk=topk,
                    ):
                        sids, sws, seids, nv, mbuf = moe_sorting(
                            _topk_ids,
                            _topk_w,
                            expert,
                            model_dim,
                            dtype,
                            1,
                            flat=AsmMoeMode.NO_SORT,
                        )
                        aiter.fmoe_fp8_blockscale_g1u1(
                            mbuf,
                            _input,
                            _w1_qt_s,
                            _w2_qt_s,
                            sids,
                            sws,
                            seids,
                            nv,
                            topk,
                            _a1_scale_t,
                            _w1_scale,
                            _w2_scale,
                            kernelName=_flat_kname,
                            activation=_act_int,
                        )
                        return mbuf

                    def _nf_seq(
                        _a1_scale_t=_a1_scale_t,
                        _act_int=_act_int,
                        _nf_a1=_nf_a1,
                        _nf_kname=_nf_kname,
                        _topk_ids=_topk_ids,
                        _topk_w=_topk_w,
                        _w1_qt_s=_w1_qt_s,
                        _w1_scale=_w1_scale,
                        _w2_qt_s=_w2_qt_s,
                        _w2_scale=_w2_scale,
                        dtype=dtype,
                        expert=expert,
                        model_dim=model_dim,
                        nf_block_m=nf_block_m,
                        topk=topk,
                    ):
                        sids, sws, seids, nv, mbuf = moe_sorting(
                            _topk_ids,
                            _topk_w,
                            expert,
                            model_dim,
                            dtype,
                            nf_block_m,
                        )
                        aiter.fmoe_fp8_blockscale_g1u1(
                            mbuf,
                            _nf_a1,
                            _w1_qt_s,
                            _w2_qt_s,
                            sids,
                            sws,
                            seids,
                            nv,
                            topk,
                            _a1_scale_t,
                            _w1_scale,
                            _w2_scale,
                            kernelName=_nf_kname,
                            activation=_act_int,
                        )
                        return mbuf

                    _, us_flat_e2e = run_perftest(_flat_seq)
                    _, us_nf_e2e = run_perftest(_nf_seq)
                    us_flat_e2e = round(us_flat_e2e, 4)
                    us_nf_e2e = round(us_nf_e2e, 4)
                    print(
                        f"  e2e head-to-head: FLAT '{_flat_kname}' "
                        f"us={us_flat_e2e}; non-FLAT '{_nf_kname}' "
                        f"(bm={nf_block_m}) us={us_nf_e2e}",
                        flush=True,
                    )
                    flat_e2e = {"us": us_flat_e2e}
                    nf_e2e = {"us": us_nf_e2e}
                    policy.record(flat_e2e, "watts", _flat_seq, kernel_us=us_flat_e2e)
                    policy.record(nf_e2e, "watts", _nf_seq, kernel_us=us_nf_e2e)
                    if "watts" in flat_e2e:
                        print(
                            f"  e2e power: FLAT{policy.watt_text(flat_e2e['watts'])}; "
                            f"non-FLAT{policy.watt_text(nf_e2e['watts'])}",
                            flush=True,
                        )
                    # Asymmetric override: only drop FLAT when non-FLAT truly
                    # wins (by a non-noise margin for latency). Otherwise leave
                    # profileDF untouched and let additive-fairness idxmin decide.
                    drop_flat = policy.beats(flat_e2e, nf_e2e, latency_margin_us=0.5)
                    if drop_flat:
                        dropped = int(flat_mask.sum())
                        profileDF = profileDF[~flat_mask].copy()
                        print(
                            f"  -> dropping {dropped} FLAT candidate(s); "
                            f"non-FLAT wins e2e by "
                            f"{us_flat_e2e - us_nf_e2e:.3f} us"
                        )
                    else:
                        print(
                            "  -> FLAT confirmed (or within noise); "
                            "keeping additive-fairness ranking"
                        )
                except Exception as _e2e_err:  # noqa: BLE001
                    print(
                        f"  e2e head-to-head skipped/failed ({_e2e_err}); "
                        "falling back to additive fairness"
                    )

            # v2 is a transient tuning-time pairing marker; the persistent layout
            # marker is kernelName2's ``_moe2_layout_`` segment. Drop it before the
            # best-row selection so it never reaches the tuned CSV. (It still
            # appears in the profile_fmoe.csv debug dump, since prorfiles
            # captured the pre-strip DataFrame by reference; that is fine -- the
            # dump already carries internal columns like stage/flat/ksplit.)
            if "v2" in profileDF.columns:
                profileDF = profileDF.drop(columns=["v2"])
            if "a16w16" in profileDF.columns:
                profileDF = profileDF.drop(columns=["a16w16"])
            profileDF["us"] = round(profileDF["us1"] + profileDF["us2"], 4)
            results = profileDF.apply(
                lambda row: self.calculate(
                    (
                        tuple(row[col] for col in self.keys),
                        "",
                        row["kernelName1"],
                        row["block_m"],
                        row["us"],
                        row["err1"],
                    )
                ),
                axis=1,
                result_type="expand",
            )
            profileDF["tflops"] = results[0]
            profileDF["bw"] = results[1]
            profileDF.drop(["tflops1", "tflops2", "bw1", "bw2"], axis=1, inplace=True)
            profileDF["err1"] = profileDF["err1"].apply(lambda x: f"{x:.1%}")
            profileDF["err2"] = profileDF["err2"].apply(lambda x: f"{x:.1%}")
            ranked = rank_moe_pairs(profileDF, policy)
            if args.profile_file != "":
                if os.path.exists(args.profile_file):
                    old_df = pd.read_csv(args.profile_file)
                else:
                    old_df = pd.DataFrame(columns=self.columns)
                tmpprofileDF = pd.concat([old_df, ranked], ignore_index=True)
                tmpprofileDF.to_csv(args.profile_file, index=False)
            # The first preferred row: lowest us in latency mode (same as
            # idxmin), lowest energy in power mode.
            best_one = policy.sort_frame(ranked).iloc[0].copy()
            latency_one = ranked.loc[
                pd.to_numeric(ranked["us"], errors="coerce").idxmin()
            ]
            energy_text = (
                f", rank={best_one.get('rank', 1)} energy={best_one['energy']} "
                f"tflops_per_w={best_one.get('tflops_per_w', float('nan'))}"
                if "energy" in ranked.columns
                else ""
            )
            print(
                f"Tuning result for {key} is {best_one['block_m'], best_one['kernelName1'], best_one['kernelName2'], best_one['err1'], best_one['err2'], best_one['run_1stage']} {best_one['us']} us, {best_one['tflops']} TFLOPS, {best_one['bw']} GB/s"
                + energy_text
            )
            if "energy" in ranked.columns:
                for _, cand in ranked.head(5).iterrows():
                    print(
                        f"  rank {int(cand['rank'])}: energy={cand['energy']} "
                        f"us={cand['us']} W={cand.get('watts', float('nan'))} "
                        f"{cand['kernelName1']} / {cand['kernelName2']}"
                    )
            if policy.measures_power and best_one.name != latency_one.name:
                print(
                    f"Latency winner differs: block_m={latency_one['block_m']} "
                    f"{latency_one['kernelName1']} / {latency_one['kernelName2']} "
                    f"{latency_one['us']} us, "
                    f"tflops_per_w={latency_one.get('tflops_per_w', float('nan'))}"
                )
            best_one["act_type"] = str(best_one["act_type"])
            best_one["q_type"] = str(best_one["q_type"])
            best_one["dtype"] = str(best_one["dtype"])
            best_one["q_dtype_a"] = str(best_one["q_dtype_a"])
            best_one["q_dtype_w"] = str(best_one["q_dtype_w"])
            if self.swiglu_limit_tag:
                best_one["_tag"] = self.swiglu_limit_tag
                if int(best_one.get("ksplit", 0) or 0) > 1:
                    raise ValueError("Runtime SiLU clamp winner cannot use split-K")
            bests.append(best_one)
        if len(prorfiles) > 0:
            profile_result = pd.concat(prorfiles)
            profile_result["err"] = profile_result["err"].apply(lambda x: f"{x:.1%}")
            profile_file = f"{AITER_ROOT_DIR}/aiter/configs/profile_fmoe.csv"
            old_profile = self.get_tuned_gemm_list(
                profile_file, profile_result.columns.tolist()
            )
            profile_result = pd.concat([old_profile, profile_result])
            profile_result.to_csv(profile_file, index=False)
        if len(bests) > 0:
            return pd.concat(bests, axis=1).T
        else:
            return pd.DataFrame()

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
        else:
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)

            if not args.all or args.last:
                self.tunedf = self.get_tuned_gemm_list(
                    self.get_out_file(args.tune_file)
                )
            else:
                self.tunedf = None
            self.untunedf["gfx"] = self.get_gfx()
            self.untunedf["cu_num"] = self.get_cu_num()
            # Migrate a legacy tuned file that predates the gfx column so the
            # untuned-vs-tuned dedup below (which now includes gfx) doesn't fail.
            if (
                self.tunedf is not None
                and "gfx" not in self.tunedf.columns
                and "cu_num" in self.tunedf.columns
            ):
                self.tunedf["gfx"] = self.tunedf["cu_num"].map(gfx_from_cu_num)
            if self.tunedf is not None and self.swiglu_limit_tag:
                # A shape only counts as tuned for rows with the same swiglu_limit.
                tags = (
                    self.tunedf["_tag"].fillna("").astype(str)
                    if "_tag" in self.tunedf.columns
                    else pd.Series("", index=self.tunedf.index)
                )
                self.tunedf = self.tunedf[tags == self.swiglu_limit_tag]
            if args.last:
                self.untunedf = self.untunedf.iloc[-1:]

            elif self.tunedf is not None:
                dedup_cols = [
                    col
                    for col in self.keys
                    if col in self.untunedf.columns and col in self.tunedf.columns
                ]
                if len(dedup_cols) == len(self.keys):
                    mask = self._already_tuned_mask(dedup_cols)
                    self.untunedf = self.untunedf[~mask]

    def e2e_tune(self, args):
        return tune_flydsl_gfx942(self, args)


def main():
    torch.set_default_device("cuda")
    key = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    # ep_fused is part of the shape identity (gemm2 doing the EP scatter shifts
    # the stage2 tile optimum), so it must be in the dedup key or the write-back
    # collapses an ep_fused row into the generic one for the same shape.
    grouped_key = key + ["gate_mode", "ep_fused"]
    resultList = [
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "xbf16",
        # Serialized AsmMoeMode value from the manifest.
        "flat",
        "tflops",
        "bw",
    ]
    grouped_result_list = [
        "max_m",
        "tile_m",
        "tile_n",
        "tile_k",
        "m_warp",
        "n_warp",
        "num_buffers",
        "grouped_persistent_m",
        "persistent_workers",
        "kernelName1",
        "kernelName2",
        "us",
        "tflops",
        "bw",
    ]
    from aiter.tuning.moe.grouped import GroupedFmoeTuner
    from aiter.tuning.moe.mxfp4 import Mxfp4FlydslTuner

    use_grouped = "--grouped-gemm" in sys.argv
    use_mxfp4_flydsl = "--mxfp4-flydsl" in sys.argv
    if use_grouped:
        if get_gfx() != "gfx1250":
            raise SystemExit("--grouped-gemm is only supported on gfx1250")
        tuner = GroupedFmoeTuner(
            "groupedFmoeTuner",
            grouped_key,
            grouped_result_list,
            "grouped fmoe tuner",
        )
    elif use_mxfp4_flydsl:
        tuner = Mxfp4FlydslTuner(
            "mxfp4FlydslTuner", key, resultList, "mxfp4 A4W4/A8W4 FlyDSL MoE tuner"
        )
    else:
        tuner = FmoeTuner("fmoeTuner", key, resultList + ["nt"], "fmoe tuner")
    args = tuner.parse_args()

    if args.e2e_tune:
        tuner.pre_process(args)
        tuner.e2e_tune(args)
    else:
        tuner.run(args, False)
