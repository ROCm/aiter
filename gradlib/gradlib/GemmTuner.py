"""hipblaslt-only bf16 GEMM tuner.

Non-hipblaslt backends (asm, opus, flydsl, triton, skinny, torch) have been
moved to ``csrc/gemm_a16w16/gemm_a16w16_tune.py``.  This file retains only the
hipblaslt search path so that ``gradlib/gradlib/gemm_tuner.py`` keeps working
as the dedicated hipblaslt tuning entry point.

Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
Copyright (C) 2024-2026, The vLLM team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from functools import lru_cache

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes, logger
from aiter.jit.core import AITER_CONFIG_GEMM_BF16
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner


def generate_data(
    m,
    n,
    k,
    indtype,
    outdtype,
    scaleAB,
    is_shuffle=False,
    seed=0,
    bias=False,
    device="cuda:0",
):
    torch.manual_seed(seed)
    if indtype == dtypes.fp8:
        randn_dtype = dtypes.bf16
    else:
        randn_dtype = indtype
    inp = torch.randn((m, k), device=device).to(randn_dtype)
    weights = torch.randn((n, k), device=device).to(randn_dtype)
    if indtype == dtypes.fp8:
        inp, x_scale = aiter.pertoken_quant(inp, quant_dtype=dtypes.fp8)
        weights, w_scale = aiter.pertoken_quant(weights, quant_dtype=dtypes.fp8)
    else:
        scale_half = torch.tensor(0.5, dtype=dtypes.fp32, device=device)
        w_scale = scale_half
        x_scale = scale_half
    if is_shuffle:
        from aiter.ops.shuffle import shuffle_weight

        shuffleweights = shuffle_weight(weights, layout=(16, 16))
    else:
        shuffleweights = weights

    bias = torch.randn(n, device=device).to(outdtype) if bias else None

    out_asm = torch.empty(m, n, dtype=outdtype, device=device)
    return {
        "inp": inp,
        "weights": weights,
        "weights_t": weights.t(),
        "bias": bias,
        "x_scale": x_scale,
        "out_asm": out_asm,
        "shuffleweights": shuffleweights,
        "w_scale": w_scale,
    }


def get_gemm_ref(inp, weights, bias, scaleA, scaleB, indtype, outdtype):
    scaleA = scaleA
    scaleB = scaleB
    if indtype == dtypes.fp8:
        x = inp.to(dtypes.fp32) * scaleA
        weight = weights.to(dtypes.fp32) * scaleB
        out = F.linear(x, weight)
        if bias is not None:
            out = out.to(bias) + bias
        return out.to(outdtype)
    else:
        ref = (
            (
                F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32))
                + bias.to(dtypes.fp32)
            ).to(outdtype)
            if bias is not None
            else F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32)).to(outdtype)
        )
    return ref


@lru_cache(maxsize=1)
def init_hipblas():
    """Lazy init: called after torch.cuda.set_device() so the hipBLASLt handle
    and workspace are allocated on the correct GPU."""
    aiter.hipb_create_extension()


def call_hipb_mm(
    input, weight, bias, scale_a, scale_b, solidx, out_dtype, bpreshuffle=False
):
    init_hipblas()
    if scale_b is not None:
        scale_b = scale_b.t()
    return aiter.hipb_mm(
        input,
        weight.t(),
        solidx,
        bias=bias,
        out_dtype=out_dtype,
        scaleA=scale_a,
        scaleB=scale_b,
        bpreshuffle=bpreshuffle,
    )


CACHE_INVALIDATE_BUFFERS = int(os.getenv("CACHE_INVALIDATE_BUFFERS", "37"))


class Gemm:
    """Per-shape hipblaslt solution scanner and timer."""

    def __init__(
        self,
        m,
        n,
        k,
        bias,
        indtype,
        outdtype,
        scaleAB=False,
        is_shuffle=False,
        mp=1,
        err_ratio=0.01,
        profile_file="",
        num_warmup=10,
        timeout=None,
        verbose=False,
        rtol=None,
        atol=None,
    ):
        torch.cuda.empty_cache()
        self.m = m
        self.k = k
        self.n = n
        self.bias = torch.randn(n, device="cuda").to(indtype) if bias else None
        self.indtype = indtype
        self.outdtype = outdtype
        self.scaleAB = scaleAB
        self.nb = CACHE_INVALIDATE_BUFFERS
        data = generate_data(m, n, k, indtype, outdtype, scaleAB, is_shuffle, 0, bias)
        self.inp = data["inp"]
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.x_scale = data["x_scale"]
        self.shuffleweights = data["shuffleweights"]
        self.w_scale = data["w_scale"]
        self.blob = torch.ones(128 * 1024 * 1024, dtype=dtypes.fp32, device="cuda")
        self.topn = 20
        self.hipb_sols = []
        _tol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        self.rtol = rtol if rtol is not None else _tol
        self.atol = atol if atol is not None else _tol
        self.check_err_ratio = err_ratio
        self.profile_file = profile_file
        self.hipb_prefer_ratio = 0.995
        self.mp = mp
        self.is_shuffle = is_shuffle
        self.has_bias = bias
        self.timeout = timeout
        self.verbose = verbose
        self.num_warmup = num_warmup

    def find_hipblas_sols(self):
        init_hipblas()
        if self.scaleAB and self.indtype == dtypes.fp8:
            scaleA = self.x_scale
            scaleB = self.w_scale.t()
        elif self.scaleAB:
            scaleA = torch.tensor(0.5, dtype=dtypes.fp32, device=self.inp.device)
            scaleB = scaleA
        else:
            scaleA = None
            scaleB = None
        sols = aiter.hipb_findallsols(
            self.inp,
            self.weights.t(),
            bias=self.bias,
            out_dtype=self.outdtype,
            scaleA=scaleA,
            scaleB=scaleB,
            bpreshuffle=self.is_shuffle,
        )
        print(
            "M N K bias dtype outdtype",
            self.m,
            self.n,
            self.k,
            self.bias is not None,
            self.indtype,
            self.outdtype,
            self.scaleAB,
            ">>> Total hipb solutions",
            len(sols),
            flush=True,
        )
        self.hipb_sols = sols

    def hipb_time_all_sols(self, fast_mode=0, top_sols=0):
        coldi = 50
        warmi = self.num_warmup
        if fast_mode:
            coldi = 2
            warmi = 5
        solutions = self.hipb_sols
        if top_sols:
            solutions = self.hipb_top_sols
        task = []
        for solidx in solutions:
            info = (
                (
                    self.m,
                    self.n,
                    self.k,
                    self.has_bias,
                    str(self.indtype),
                    str(self.outdtype),
                    self.scaleAB,
                    self.is_shuffle,
                ),
                solidx,
                0,  # splitK
                "",  # kernelName
                "hipblaslt",
                self.is_shuffle,
            )
            task.append(
                (
                    info,
                    generate_data,
                    (
                        self.m,
                        self.n,
                        self.k,
                        self.indtype,
                        self.outdtype,
                        self.scaleAB,
                        self.is_shuffle,
                        0,
                        self.has_bias,
                    ),
                    call_hipb_mm,
                    (
                        ["inp", "shuffleweights", "bias", "x_scale", "w_scale"],
                        solidx,
                        self.outdtype,
                        self.is_shuffle,
                    ),
                    {
                        "num_warmup": warmi,
                        "num_iters": coldi,
                    },
                    get_gemm_ref if fast_mode == 0 else None,
                    (
                        ["inp", "weights", "bias", "x_scale", "w_scale"],
                        self.indtype,
                        self.outdtype,
                    ),
                    {},
                    None,
                    self.rtol,
                    self.atol,
                )
            )
        in_data = [
            (
                len(solutions),
                (),
            )
        ]
        ret = mp_tuner(
            task,
            in_data,
            self.mp,
            fast_mode == 1,
            timeout=self.timeout,
            verbose=self.verbose,
        )
        if fast_mode == 1:
            self.hipb_gtimedf = self.save_topn_result(ret, fast_mode, "hipblaslt")
            return []
        print(f">>> hipblaslt top solutions, Fast Mode {fast_mode}")
        return ret

    def save_topn_result(self, rets, fast_mode, libtype):
        results = []
        if not rets:
            return pd.DataFrame(
                columns=["solidx", "gtimems", "splitK", "err_ratio", "kernelName"]
            )
        for info, us, err_ratio in rets:
            res_one = []
            solidx = info[1]
            splitK = info[2]
            kernelName = info[3]
            res_one.append(solidx)
            res_one.append(round(us / 1000.0, 4))
            res_one.append(splitK)
            res_one.append(err_ratio)
            res_one.append(kernelName)

            results.append(res_one)
        gtimedf = pd.DataFrame(
            results, columns=["solidx", "gtimems", "splitK", "err_ratio", "kernelName"]
        )

        gtimedf = gtimedf.sort_values(by="gtimems")
        gtimedf["libtype"] = libtype

        gtimedf.to_csv(f"/tmp/{libtype}_gtimedf.csv", index=False)
        print(f">>> {libtype} top solutions, Fast Mode {fast_mode}")
        print(gtimedf.head(self.topn), flush=True)
        return gtimedf

    def warmup(self, warmi=500):
        for i in range(warmi):
            self.blob = self.blob + 0.00001

    def functional_get_topn_fastest(self):
        hipb_topn = self.hipb_gtimedf["solidx"].head(self.topn).tolist()
        self.hipb_top_sols = hipb_topn

    def run_fast_solutions(self):
        self.find_hipblas_sols()
        self.warmup()
        self.hipb_time_all_sols(fast_mode=1)

    def run_best_solutions(self):
        self.warmup()
        rets_hipb = self.hipb_time_all_sols(fast_mode=0, top_sols=1)
        return rets_hipb

    def run_solutions(self):
        self.run_fast_solutions()
        self.functional_get_topn_fastest()
        rets = self.run_best_solutions()
        return rets

    def cleanup(self):
        if hasattr(self, "inp"):
            del self.inp
        if hasattr(self, "weights"):
            del self.weights
        if hasattr(self, "bias") and self.bias is not None:
            del self.bias
        if hasattr(self, "blob"):
            cpu_blob = self.blob.cpu()
            del cpu_blob


class GemmTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_BF16}",
        "untune_file": "aiter/configs/bf16_untuned_gemm.csv",
        "batch": 100,
        "config_env_name": "AITER_CONFIG_GEMM_BF16",
    }

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--tuned_file",
            type=str,
            default=os.getenv("GTUNE_TUNED", AITER_CONFIG_GEMM_BF16),
            dest="tune_file",
            help="output file for tuned gemm solutions",
        )
        self.parser.add_argument(
            "--input_file",
            type=str,
            default=os.getenv("GTUNE_INPUT", None),
            dest="untune_file",
            help="list of gemms to tune for, mutually exclusive with model_dir",
        )
        self.parser.add_argument(
            "--indtype",
            type=str,
            default=None,
            choices=["f32", "f16", "bf16", "fp8"],
            help="dtype: f32 f16 bf16 fp8. Use this to override the"
            " input_file or if no input_file provided",
        )
        self.parser.add_argument(
            "--outdtype",
            type=str,
            choices=["f32", "f16", "bf16", "fp8"],
            help="dtype: f32 f16 bf16 fp8. Use to override the default value,"
            " which is the same as indtype for each shape (see --indtype.)",
        )
        self.parser.add_argument(
            "--all_bias",
            action="store_true",
            help="Tune for both bias and non bias cases,"
            " regardless of what was used"
            " to collect the shapes",
        )

    def __init__(
        self,
        key=[
            "gfx",
            "cu_num",
            "M",
            "N",
            "K",
            "bias",
            "dtype",
            "outdtype",
            "scaleAB",
            "bpreshuffle",
        ],
        resultList=[
            "libtype",
            "solidx",
            "splitK",
            "us",
            "kernelName",
            "err_ratio",
            "tflops",
            "bw",
        ],
        description="GemmTuner (hipblaslt-only)",
    ):
        super().__init__(
            "GemmTuner",
            key=key,
            resultList=resultList,
            description=description,
        )

        self.hipb_prefer_ratio = 0.995
        self.cu_num = self.get_cu_num()
        self.gfx = self.get_gfx()
        self.gemmobj = None
        self.num_warmup = 10

    def _clear_op_caches(self):
        from aiter.tuned_gemm import get_GEMM_A16W16_config_, get_GEMM_A16W16_config

        get_GEMM_A16W16_config_.cache_clear()
        get_GEMM_A16W16_config.cache_clear()

    def run_config(self, args):
        from aiter.tuned_gemm import gemm_a16w16
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            bias = row["bias"]
            indtype = str(row["dtype"])
            outdtype = str(row["outdtype"])
            scaleAB = row["scaleAB"]
            bpreshuffle = row["bpreshuffle"]
            shape_str = f"({M}, {N}, {K}, {indtype}, bias={bias})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                data = generate_data(
                    M,
                    N,
                    K,
                    eval(indtype),
                    eval(outdtype),
                    scaleAB,
                    bpreshuffle,
                    0,
                    bias,
                )
                inp = data["inp"]
                weights = data["weights"]
                bias_tensor = data["bias"]
                x_scale = data["x_scale"]
                shuffleweights = data["shuffleweights"]
                w_scale = data["w_scale"]
                w = shuffleweights if bpreshuffle else weights
                scale_a = x_scale if scaleAB else None
                scale_b = w_scale if scaleAB else None
                out, us = run_perftest(
                    gemm_a16w16,
                    inp,
                    w,
                    bias=bias_tensor,
                    otype=eval(outdtype),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = get_gemm_ref(
                    inp,
                    weights,
                    bias_tensor,
                    x_scale,
                    w_scale,
                    eval(indtype),
                    eval(outdtype),
                )
                _tol = 5e-2 if eval(outdtype) == torch.bfloat16 else 1e-2
                _atol = _tol
                _rtol = _tol
                err_ratio = checkAllclose(
                    out, ref, atol=_atol, rtol=_rtol, msg=f"run_config {shape_str}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
        return results

    def calculate_perf(
        self,
        results,
        inbpe,
        outbpe,
    ):
        """calculate TFLOPS and bandwidth"""
        info, time, err_ratio = results
        if time <= 0:
            return -1, -1
        gfx, cu_num, m, n, k = info
        flops = m * n * k * 2
        tflops = round(flops / (time * 1000000), 2)

        bw = round(
            (m * k * inbpe + n * k * inbpe + m * n * outbpe) / (time * 1e-6) / 1e9,
            2,
        )
        return tflops, bw

    def get_untuned_gemm_list(self, untuned_gemm_file):
        assert os.path.exists(
            untuned_gemm_file
        ), f"Not exist untuned file: {untuned_gemm_file}"
        untunedf = pd.read_csv(untuned_gemm_file).fillna("")
        filtered_df = untunedf.drop_duplicates().reset_index(drop=True)

        return filtered_df

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
        else:
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)
            if "outdtype" not in self.untunedf.columns:
                self.untunedf["outdtype"] = self.untunedf["dtype"]
            if "scaleAB" not in self.untunedf.columns:
                self.untunedf["scaleAB"] = False
            _cli_to_dtypes = {
                "f16": "fp16",
                "f32": "fp32",
                "bf16": "bf16",
                "fp8": "fp8",
            }
            if args.indtype is not None:
                self.untunedf["dtype"] = f"dtypes.{_cli_to_dtypes[args.indtype]}"
            if args.outdtype is not None:
                self.untunedf["outdtype"] = f"dtypes.{_cli_to_dtypes[args.outdtype]}"

            if args.all_bias:
                for i in range(len(self.untunedf)):
                    ds = self.untunedf.iloc[i]
                    for bias in [True, False] if args.all_bias else [ds["bias"]]:
                        self.add_gemm(
                            ds["M"],
                            ds["N"],
                            ds["K"],
                            indtype=str(ds["dtype"]),
                            bias=bias,
                            outdtype=str(ds["outdtype"]),
                            scaleAB=ds["scaleAB"],
                            bpreshuffle=ds["bpreshuffle"],
                        )
            self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))
            self.untunedf["gfx"] = self.get_gfx()
            self.untunedf["cu_num"] = self.get_cu_num()
            self.untunedf = self.untunedf[self.keys]
            untunedf_cols = self.untunedf.columns
            if len(self.tunedf) != 0:
                mask = self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[untunedf_cols].apply(tuple, axis=1)
                )
                if args.verbose:
                    logger.info("skiped tuned shapes:")
                    print(self.untunedf[mask])
                self.untunedf = self.untunedf[~mask]
            self.untunedf = self.untunedf.drop_duplicates().reset_index(drop=True)
            print("untunedf is ", self.untunedf)

    def add_gemm(
        self,
        m,
        n,
        k,
        indtype,
        bias=False,
        outdtype=None,
        scaleAB=False,
        bpreshuffle=False,
    ):
        assert indtype is not None
        outdtype = outdtype if outdtype is not None else indtype
        assert outdtype is not None
        print(self.tunedf)
        if self.tunedf is None or (
            self.tunedf[
                (self.tunedf["gfx"] == self.gfx)
                & (self.tunedf["cu_num"] == self.cu_num)
                & (self.tunedf["M"] == m)
                & (self.tunedf["N"] == n)
                & (self.tunedf["K"] == k)
                & (self.tunedf["bias"] == bias)
                & (self.tunedf["dtype"] == str(indtype))
                & (self.tunedf["outdtype"] == str(outdtype))
                & (self.tunedf["bpreshuffle"] == str(bpreshuffle))
            ].empty
        ):
            entry = {
                "gfx": [self.gfx],
                "cu_num": [self.cu_num],
                "M": [m],
                "N": [n],
                "K": [k],
                "bias": [bias],
                "dtype": [indtype],
                "outdtype": [outdtype],
                "scaleAB": [scaleAB],
                "bpreshuffle": [bpreshuffle],
            }
            df = pd.DataFrame(entry)
            self.untunedf = pd.concat([self.untunedf, df], ignore_index=True)
        else:
            print(
                f">>>Info: Found Duplicate shape(M:{m},"
                f" N:{n}, K:{k} bias:{bias}), skipping"
            )

    def tune(self, untunedf, tunedf, args):
        df = untunedf
        ret = []
        for i in range(len(df)):
            ds = df.loc[i, :]
            indtype = ds["dtype"]
            outdtype = ds["outdtype"]
            outdtype = outdtype if outdtype is not None else indtype
            self.set_run_iters(
                (self.gfx, self.cu_num, ds["M"], ds["N"], ds["K"]), eval(indtype)
            )

            gemmobj = Gemm(
                ds["M"],
                ds["N"],
                ds["K"],
                ds["bias"],
                indtype=eval(indtype),
                outdtype=eval(outdtype),
                scaleAB=ds["scaleAB"],
                is_shuffle=ds["bpreshuffle"],
                mp=args.mp,
                err_ratio=args.errRatio,
                profile_file=args.profile_file,
                num_warmup=self.num_warmup,
                timeout=args.timeout,
                verbose=args.verbose,
            )

            ret.extend(gemmobj.run_solutions())
            gemmobj.cleanup()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            del gemmobj

        return ret

    def processResult(self, rets, fast_mode):
        results = []
        for info, us, err_ratio in rets:
            res_one = []
            solidx = info[1]
            splitK = info[2]
            kernelName = info[3]
            libtype = info[4]
            res_one.append(get_gfx())
            res_one.append(get_cu_num())
            for ele in info[0]:
                res_one.append(ele)

            res_one.append(libtype)
            res_one.append(int(solidx))
            res_one.append(int(splitK))
            res_one.append(round(us, 4))

            res_one.append(kernelName)
            res_one.append(err_ratio)
            ret = (
                (self.gfx, self.cu_num, info[0][0], info[0][1], info[0][2]),
                us,
                err_ratio,
            )
            tflops, bw = self.calculate_perf(
                ret,
                self.get_bpe(eval(info[0][4])),
                self.get_bpe(eval(info[0][5])),
            )
            res_one.append(tflops)
            res_one.append(bw)

            results.append(res_one)
        gtimedf = pd.DataFrame(results, columns=self.columns)
        gtimedf = gtimedf.sort_values(by="us")
        return gtimedf

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        from collections import defaultdict

        grouped_rets = defaultdict(list)

        for info, us, max_err_ratio in rets:
            grouped_rets[info[0]].append((info, us, max_err_ratio))

        grouped_results = list(grouped_rets.items())
        gtimedf_dic = {}
        for key, ret_info in grouped_results:
            gtimedf_dic[key] = self.processResult(ret_info, fast_mode)

        if args.profile_file != "":
            resultsdf = pd.concat(
                gtimedf_dic.values(),
                ignore_index=True,
            )
        else:
            resultsdf = pd.DataFrame(self.columns)
        self.save_profile(resultsdf, args.profile_file)

        best_gtimedfs = pd.DataFrame(columns=self.columns)
        for key, df in gtimedf_dic.items():
            gtimedf_dic[key] = df[df["err_ratio"] < args.errRatio]
            best_gtimedf = gtimedf_dic[key].sort_values(by="us")

            if len(gtimedf_dic[key]) == 0:
                print(">>> No valid hipblaslt solutions found!", flush=True)
                failedf = df.iloc[0:1]
                self.failed = pd.concat([self.failed, failedf], ignore_index=True)
                continue
            resultdf1 = best_gtimedf.head(1).reset_index(drop=True)
            kernal_name = aiter.getHipblasltKernelName(int(resultdf1.iloc[0]["solidx"]))
            resultdf1.loc[0, "kernelName"] = kernal_name
            if best_gtimedfs.empty:
                best_gtimedfs = resultdf1
            else:
                best_gtimedfs = pd.concat([best_gtimedfs, resultdf1], ignore_index=True)

            print(f"{key} >>> Fastest Solution is \n {resultdf1}", flush=True)
        return best_gtimedfs

    def save_profile(self, timedf, profile_file):
        if profile_file != "":
            if os.path.exists(profile_file):
                old_df = pd.read_csv(profile_file)
            else:
                old_df = pd.DataFrame(columns=self.columns)

            resultsdf = pd.concat([old_df, timedf], ignore_index=True)
            resultsdf.to_csv(profile_file, index=False)

    def set_run_iters(self, input, inputdtype):
        gfx, cu_num, m, n, k, *rest = input
        flops = m * n * k * 2
        if flops < 128 * 5120 * 256 * 2:
            self.num_warmup = 30
        elif flops < 256 * 5120 * 256 * 2:
            self.num_warmup = 20
        else:
            self.num_warmup = 10
