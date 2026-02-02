# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import torch
import aiter
import pandas as pd
import os
import sys
import re
from aiter import QuantType
from aiter.jit.core import get_asm_dir, AITER_CSRC_DIR, AITER_CONFIG_FMOE
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter import dtype2str_dict
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.mp_tuner import mp_tuner
from aiter import dtypes
from aiter import ActivationType as ActivationType
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility.base_tuner import TunerCommon
from aiter.utility import fp4_utils


torch.set_default_device("cuda")


class FmoeTuner(TunerCommon):

    ARG_DEFAULTS = {
        **TunerCommon.ARG_DEFAULTS,
        "verbose": False,
        "tune_file": f"{AITER_CONFIG_FMOE}",
        "untune_file": "aiter/configs/untuned_fmoe.csv",
        "errRatio": 0.5,
        "batch": 100,
        "profile_file": "",
    }

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--last",
            action="store_true",
            required=False,
            help="Only last kernel is tuned",
        )

        self.parser.add_argument(
            "--flydsl-candidate-csv",
            type=str,
            default="",
            required=False,
            help="Path to FlyDSL candidate configuration CSV file",
        )

    @staticmethod
    def weight_quant(weight, qType, quant_dtype):
        """Quantize weight tensor."""
        E, dim1, dim2 = weight.shape
        if qType == aiter.QuantType.per_Tensor and quant_dtype != torch.int4:
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight.view(E, -1), quant_dtype=quant_dtype
            )
        elif qType == QuantType.per_1x128:
            weight_qt = (
                weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(E, -1, 128 * 128)
            )
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_qt, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(E, -1)
            weight_qt = (
                weight_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(E, dim1, dim2)
            )
        else:
            torch_quant = aiter.get_torch_quant(qType)
            weight_qt, weight_scale = torch_quant(weight, quant_dtype=quant_dtype)
        return weight_qt, weight_scale

    @staticmethod
    def flydsl_moe_stage1_fwd_out(
        a1_qt,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w1_scale,
        a1_scale,
        dtype,
        topk,
        kernelName,
        blockM,
        q_type,
        act_type,
    ):
        """FlyDSL Stage1 forward."""
        from aiter.fused_moe import flydsl_moe_stage1

        inter_dim = w1_qt.shape[1] // 2
        token_num = a1_qt.shape[0]
        out = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=a1_qt.device,
        )

        out = flydsl_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            blockM,
            a1_scale,
            w1_scale,
            kernelName=kernelName,
            sorted_weights=sorted_weights,
            quant_type=q_type,
            activation=act_type,
            dtype=dtype,
        )

        return out

    @staticmethod
    def flydsl_moe_stage2_fwd_out(
        a2_qt,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w2_scale,
        a2_scale,
        dtype,
        topk,
        kernelName,
        blockM,
        q_type,
        act_type,
    ):
        """FlyDSL Stage2 forward."""
        from aiter.fused_moe import flydsl_moe_stage2

        model_dim = w2_qt.shape[1]
        token_num = a2_qt.shape[0]

        out = torch.zeros(
            (token_num, model_dim),
            dtype=dtype,
            device=a2_qt.device,
        )

        return flydsl_moe_stage2(
            a2_qt,
            w1_qt,
            w2_qt,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            w2_scale,
            a2_scale,
            blockM,
            kernelName=kernelName,
            sorted_weights=sorted_weights,
            quant_type=q_type,
            activation=act_type,
            dtype=dtype,
        )

    @staticmethod
    def generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device="cuda",
    ):
        torch.manual_seed(0)
        input = torch.randn((token, model_dim), dtype=dtype) / 10
        if use_g1u1:
            w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
        else:
            w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype) / 10
        w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)
        w1_qt, w1_scale = FmoeTuner.weight_quant(w1, q_type, quant_dtype=q_dtype_w)
        w2_qt, w2_scale = FmoeTuner.weight_quant(w2, q_type, quant_dtype=q_dtype_w)

        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)

        score = torch.randn((token, expert), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)

        torch_quant = aiter.get_torch_quant(q_type)
        a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)

        del w1, w2, score

        w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
        w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            moe_sorting(topk_ids, topk_weights, expert, model_dim, dtype, blockM)
        )
        return (
            input,
            a1_qt,
            w1_qt,
            w2_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk_weights,
            moe_buf,
            a1_scale,
            w1_scale,
            w2_scale,
        )

    @staticmethod
    def generate_data_2stages(
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
        blockM,
        stage=1,
        device="cuda",
    ):
        (
            input,
            a1_qt,
            w1_qt,
            w2_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk_weights,
            moe_buf,
            a1_scale,
            w1_scale,
            w2_scale,
        ) = FmoeTuner.generate_data(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            blockM,
            device,
        )

        w1_qt_shffle_ck = w1_qt_shffle
        w2_qt_shffle_ck = w2_qt_shffle
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

        if stage == 1:
            if not doweight_stage1:
                sorted_weights = None
            a1_scale_fp4_sort = a1_scale

            return (
                a1_qt,  # 0
                w1_qt_shffle_ck,  # 1
                w2_qt_shffle_ck,  # 2
                a1_scale,  # 3
                w1_scale,  # 4
                sorted_ids,  # 5
                sorted_expert_ids,  # 6
                sorted_weights,  # 7
                num_valid_ids,  # 8
                moe_buf,  # 9
                w1_qt,  # 10
                w2_qt,  # 11
                topk_weights,  # 12
                topk_ids,  # 13
                a1_scale_fp4_sort,  # 14
                w1_scale_aiter,  # 15
            )
        elif stage == 2:
            ref1 = FmoeTuner.run_torch_moe_stage1(
                a1_qt,
                w1_qt,
                w2_qt,
                topk_weights,
                topk_ids,
                a1_scale=a1_scale,
                w1_scale=w1_scale,
                dtype=dtype,
                activation=act_type,
                quant_type=q_type,
                doweight_stage1=doweight_stage1,
                topk=topk,
            )

            torch_quant = aiter.get_torch_quant(q_type)
            a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
            a2_qt = a2_qt.view(token, topk, -1)

            if doweight_stage1:
                sorted_weights = None
            a2_scale_mxfp4_sort = a2_scale

            return (
                a2_qt,  # 0
                w1_qt_shffle_ck,  # 1
                w2_qt_shffle_ck,  # 2
                a2_scale,  # 3
                w2_scale,  # 4
                sorted_ids,  # 5
                sorted_expert_ids,  # 6
                sorted_weights,  # 7
                num_valid_ids,  # 8
                moe_buf,  # 9
                w1_qt,  # 10
                w2_qt,  # 11
                topk_weights,  # 12
                topk_ids,  # 13
                a2_scale_mxfp4_sort,  # 14
                w2_scale_aiter,  # 15
            )

    @staticmethod
    def run_torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        a1_scale,
        w1_scale,
        dtype,
        activation,
        quant_type,
        doweight_stage1,
        topk,
    ):
        ref1 = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            activation=activation,
            quant_type=quant_type,
            dtype=dtype,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            doweight=doweight_stage1,
        )
        return ref1

    @staticmethod
    def run_torch_moe_stage2(
        a2_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        a2_scale,
        w2_scale,
        dtype,
        quant_type,
        doweight_stage1,
    ):
        return torch_moe_stage2(
            a2_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            dtype,
            quant_type,
            a2_scale=a2_scale,
            w2_scale=w2_scale,
            doweight=not doweight_stage1,
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        key, stage, kernelName, block_m, us, err = results
        (
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

        if us == self.INVALID_TIME or us == self.INF_TIME:
            return 0, 0

        # Calculate FLOP for 2-stage MOE
        if use_g1u1:
            n = inter_dim * 2
        else:
            n = inter_dim

        flop = (
            token * n * model_dim * topk * 2  # Stage1: gemm1
            + topk * token * model_dim * inter_dim * 2  # Stage2: gemm2
        )

        data_bytes = (
            token * model_dim * self.get_bpe(q_dtype_a)
            + n * model_dim * self.get_bpe(q_dtype_w) * expert
            + inter_dim * model_dim * self.get_bpe(q_dtype_w) * expert
            + token * model_dim * self.get_bpe(dtype)
        )

        tflops = round(flop / (us * 1000000), 2)
        bw = round(data_bytes / (us * 1e-6) / 1e9, 2)
        return tflops, bw

    def gen_2stages_flydsl_task(self, key, blockMs, flydsl_csv=None):
        info = key
        tasks_flydsl = []
        (
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
        ) = info

        if q_dtype_a == dtypes.fp8:
            quantDtype = "Fp8"
        elif q_dtype_a == dtypes.i8:
            quantDtype = "Int8"
        else:
            print(f"FlyDSL only supports fp8/int8 quantization, got {q_dtype_a}")
            return []

        # Use user-specified CSV or default path
        if flydsl_csv and os.path.exists(flydsl_csv):
            csv_path = flydsl_csv
        else:
            csv_path = f"{get_asm_dir()}/fmoe_2stages/flydsl_moe_bf16_pertoken{quantDtype}_g1u1.csv"

        if not os.path.exists(csv_path):
            print(f" FlyDSL config file not found: {csv_path}")
            return []

        flydsl_df = pd.read_csv(csv_path)
        print(f"Loaded {len(flydsl_df)} FlyDSL configurations from {csv_path}")

        for blockM in blockMs:
            if not use_g1u1:
                continue

            # FlyDSL Stage1 tasks
            for _, row in flydsl_df.iterrows():
                tile_m = int(row["tile_m"])
                tile_n = int(row["tile_n"])
                tile_k = int(row["tile_k"])
                kernelName1 = row["kernelName1"]

                if tile_m != blockM:
                    continue

                tasks_flydsl.append(
                    (
                        (info, "flydsl_stage1", kernelName1, blockM),
                        FmoeTuner.generate_data_2stages,
                        (
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
                            blockM,
                            1,  # stage=1
                        ),
                        FmoeTuner.flydsl_moe_stage1_fwd_out,
                        (
                            [0, 1, 2, 5, 6, 7, 8, 15, 14],
                            dtype,
                            topk,
                            kernelName1,
                            blockM,
                            q_type,
                            act_type,
                        ),
                        {},
                        FmoeTuner.run_torch_moe_stage1,
                        (
                            [0, 10, 11, 12, 13, 3, 4],
                            dtype,
                            act_type,
                            q_type,
                            doweight_stage1,
                            topk,
                        ),
                        {},
                        (None),
                        0.01,
                        0.01,
                        True,
                    )
                )

            # FlyDSL Stage2 tasks
            for _, row in flydsl_df.iterrows():
                kernelName2 = row["kernelName2"]

                match = re.search(r"gemm2_(\d+)x(\d+)x(\d+)", kernelName2)
                if match:
                    tile_m2 = int(match.group(1))
                else:
                    continue

                if tile_m2 != blockM:
                    continue

                tasks_flydsl.append(
                    (
                        (info, "flydsl_stage2", kernelName2, blockM),
                        FmoeTuner.generate_data_2stages,
                        (
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
                            blockM,
                            2,  # stage=2
                        ),
                        FmoeTuner.flydsl_moe_stage2_fwd_out,
                        (
                            [0, 1, 2, 5, 6, 7, 8, 15, 14],
                            dtype,
                            topk,
                            kernelName2,
                            blockM,
                            q_type,
                            act_type,
                        ),
                        {},
                        FmoeTuner.run_torch_moe_stage2,
                        (
                            [0, 10, 11, 12, 13, 3, 4],
                            dtype,
                            q_type,
                            doweight_stage1,
                        ),
                        {},
                        (None),
                        0.01,
                        0.01,
                        True,
                    )
                )

        return tasks_flydsl

    def tune(self, untunedf, tunedf, args):
        """Main tuning entry point - FlyDSL only."""
        mp_num = args.mp
        blockMs = [16, 32, 64, 128]
        keys = self.keys
        print(untunedf[keys])

        # Get FlyDSL candidate CSV path from args
        flydsl_csv = getattr(args, "flydsl_candidate_csv", None)
        if flydsl_csv:
            print(f"Using FlyDSL candidate CSV: {flydsl_csv}")

        tasks_flydsl = []

        for line in untunedf[keys].values:
            (
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

            eval_ns = {
                "torch": torch,
                "dtypes": dtypes,
                "fp8": dtypes.fp8,
                "i8": dtypes.i8,
                "bf16": dtypes.bf16,
                "fp16": dtypes.fp16,
                "fp32": dtypes.fp32,
                "QuantType": QuantType,
                "ActivationType": ActivationType,
                "aiter": aiter,
            }

            dtype = eval(dtype, eval_ns) if isinstance(dtype, str) else dtype
            q_dtype_a = (
                eval(q_dtype_a, eval_ns) if isinstance(q_dtype_a, str) else q_dtype_a
            )
            q_dtype_w = (
                eval(q_dtype_w, eval_ns) if isinstance(q_dtype_w, str) else q_dtype_w
            )
            q_type = eval(q_type, eval_ns) if isinstance(q_type, str) else q_type
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type

            print(f"\n Start FlyDSL tuning for: {line}")

            if not use_g1u1:
                print("FlyDSL requires use_g1u1=True, skipping...")
                continue

            if q_dtype_a not in [dtypes.fp8, dtypes.i8]:
                print(f"FlyDSL only supports fp8/int8 activation, got {q_dtype_a}")
                continue

            act_type = (
                eval(act_type, eval_ns) if isinstance(act_type, str) else act_type
            )
            info = (
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

            tasks_flydsl.extend(self.gen_2stages_flydsl_task(info, blockMs, flydsl_csv))

            if not tasks_flydsl:
                print("No FlyDSL tasks generated, skipping...")
                continue

            print(f"Generated {len(tasks_flydsl)} FlyDSL tasks")

        in_data = [(len(tasks_flydsl), ())]
        rets = []

        if len(tasks_flydsl) > 0:
            rets = mp_tuner(
                tasks_flydsl,
                in_data,
                mp_num,
                True,
                False,
                timeout=args.timeout,
                verbose=args.verbose,
            )

        if not rets:
            print("No FlyDSL results found")
            return []
        else:
            print(f"Got {len(rets)} FlyDSL results")
            return rets

    def result_to_csv(self, results, file, concat=False):
        """Save results to CSV."""
        old_tunedf = self.get_tuned_gemm_list(file)
        resultdf = self.update_tunedf(old_tunedf, results)
        self.success = pd.concat([self.success, results], ignore_index=True)
        resultdf["run_1stage"] = resultdf["run_1stage"].astype(int)
        if results is not None:
            resultdf = resultdf.astype(str).drop_duplicates(
                subset=self.keys,
                keep="last",
            )
        resultdf.to_csv(file, index=False)

    def post_process(self, results, args, topk=-1, fast_mode=False):
        """Post-process tuning results."""
        from collections import defaultdict

        prorfiles = []
        bests = []

        # Group results by key
        grouped_rets = defaultdict(list)
        for info, us, max_err_ratio in results:
            grouped_rets[tuple(info[0])].append((info[1:], us, max_err_ratio))

        for key, rets in grouped_rets.items():
            (
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

            profileDF = []
            for (stage, kernelName, block_m), us, err in rets:
                tflops, bw = self.calculate((key, stage, kernelName, block_m, us, err))
                profileDF.append(
                    [
                        stage,
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
                        0,
                        us,
                        kernelName,
                        err,
                        tflops,
                        bw,
                    ]
                )

            profileDF = pd.DataFrame(
                profileDF,
                columns=["stage"]
                + self.keys
                + ["block_m", "ksplit", "us", "kernelName", "err", "tflops", "bw"],
            )
            prorfiles.append(profileDF)

            # Remove invalid candidates
            profileDF = profileDF[
                (profileDF["err"] < args.errRatio)
                & (profileDF["us"] != float("-inf"))
                & (profileDF["us"] != -1)
            ]
            profileDF = profileDF.sort_values("us").drop_duplicates(
                ["stage", "block_m"], keep="first"
            )

            # FlyDSL Stage1 results
            stage1_profileDF = profileDF[profileDF["stage"] == "flydsl_stage1"].drop(
                columns=["stage"], axis=1
            )
            stage1_profileDF = stage1_profileDF.rename(
                columns={
                    "kernelName": "kernelName1",
                    "err": "err1",
                    "us": "us1",
                    "tflops": "tflops1",
                    "bw": "bw1",
                }
            )

            # FlyDSL Stage2 results
            stage2_profileDF = profileDF[profileDF["stage"] == "flydsl_stage2"].drop(
                columns=["stage", "ksplit"], axis=1, errors="ignore"
            )
            stage2_profileDF = stage2_profileDF.rename(
                columns={
                    "kernelName": "kernelName2",
                    "err": "err2",
                    "us": "us2",
                    "tflops": "tflops2",
                    "bw": "bw2",
                }
            )

            if stage1_profileDF.empty or stage2_profileDF.empty:
                print(f"⚠️  Missing Stage1 or Stage2 results for {key}")
                continue

            # Merge Stage1 and Stage2 by block_m
            profileDF = pd.merge(
                stage1_profileDF,
                stage2_profileDF,
                on=[
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
                ],
                how="inner",
            )
            profileDF["run_1stage"] = 0

            if len(profileDF) == 0:
                print(f"No valid FlyDSL configuration found for {key}")
                continue

            profileDF["us"] = round(profileDF["us1"] + profileDF["us2"], 4)
            results_calc = profileDF.apply(
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
            profileDF["tflops"] = results_calc[0]
            profileDF["bw"] = results_calc[1]
            profileDF.drop(["tflops1", "tflops2", "bw1", "bw2"], axis=1, inplace=True)
            profileDF["err1"] = profileDF["err1"].apply(lambda x: f"{x:.1%}")
            profileDF["err2"] = profileDF["err2"].apply(lambda x: f"{x:.1%}")

            # Save profile results
            if args.profile_file:
                if os.path.exists(args.profile_file):
                    old_df = pd.read_csv(args.profile_file)
                else:
                    old_df = pd.DataFrame(columns=self.columns)
                tmpprofileDF = pd.concat([old_df, profileDF], ignore_index=True)
                tmpprofileDF.to_csv(args.profile_file, index=False)

            # Select best configuration
            best_one = profileDF.loc[profileDF["us"].idxmin()].copy()
            print(
                f"\n🎯 Best FlyDSL config for {key}:\n"
                f"   block_m={best_one['block_m']}\n"
                f"   kernelName1={best_one['kernelName1']}\n"
                f"   kernelName2={best_one['kernelName2']}\n"
                f"   err1={best_one['err1']}, err2={best_one['err2']}\n"
                f"   us={best_one['us']}, tflops={best_one['tflops']}, bw={best_one['bw']} GB/s"
            )

            best_one["act_type"] = str(best_one["act_type"])
            best_one["q_type"] = str(best_one["q_type"])
            best_one["dtype"] = str(best_one["dtype"])
            best_one["q_dtype_a"] = str(best_one["q_dtype_a"])
            best_one["q_dtype_w"] = str(best_one["q_dtype_w"])
            bests.append(best_one)

        # Save all profile results
        if len(prorfiles) > 0:
            profile_result = pd.concat(prorfiles)
            profile_result["err"] = profile_result["err"].apply(lambda x: f"{x:.1%}")
            profile_file = args.profile_file or "aiter/configs/profile_fmoe_flydsl.csv"
            os.makedirs(os.path.dirname(profile_file), exist_ok=True)
            old_profile = self.get_tuned_gemm_list(
                profile_file, profile_result.columns.tolist()
            )
            profile_result = pd.concat([old_profile, profile_result])
            profile_result.to_csv(profile_file, index=False)
            print(f"\nProfile saved to: {profile_file}")

        if len(bests) > 0:
            return pd.concat(bests, axis=1).T
        else:
            return pd.DataFrame()

    def pre_process(self, args):
        """Pre-process: load untuned configurations."""
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

            self.untunedf["cu_num"] = self.get_cu_num()

            if args.last:
                self.untunedf = self.untunedf.iloc[-1:]
            elif self.tunedf is not None:
                untunedf_cols = self.untunedf.columns
                mask = self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[untunedf_cols].apply(tuple, axis=1)
                )
                self.untunedf = self.untunedf[~mask]


if __name__ == "__main__":
    key = [
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
        "tflops",
        "bw",
    ]

    tuner = FmoeTuner("FlyDSL MOE Tuner", key, resultList, "FlyDSL MOE tuner")
    args = tuner.parse_args()
    tuner.run(args, False)
