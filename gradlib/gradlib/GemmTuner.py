"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2025, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import functools
import os

import aiter
import pandas as pd
from aiter import dtypes
import torch
import torch.nn.functional as F
from aiter import logger

from aiter.utility.mp_tuner import mp_tuner
from functools import lru_cache
from aiter.jit.core import get_asm_dir
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.core import AITER_CONFIG_GEMM_BF16, get_asm_dir
from aiter.utility.base_tuner import GemmCommonTuner

aiter.hipb_create_extension()


@lru_cache(maxsize=1)
def init_hipblas():
    aiter.hipb_create_extension()


def call_hipb_mm(input, weight, bias, scale_a, scale_b, solidx, out_dtype):
    init_hipblas()
    return aiter.hipb_mm(
        input,
        weight,
        solidx,
        bias=bias,
        out_dtype=out_dtype,
        scaleA=scale_a,
        scaleB=scale_b,
    )


def run_gemm_bf16_asm(
    inp, w, out, bias=None, splitK=None, kernelName=None, bpreshuffle=False
):
    return aiter.gemm_a16w16_asm(
        inp,
        w,
        out,
        bias=bias,
        splitK=splitK,
        kernelName=kernelName,
        bpreshuffle=bpreshuffle,
    )


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    # cusPerTile = cu_num / tile_num
    splitK = 0
    if tile_num < cu_num:
        splitK = int(cu_num / tile_num)
    else:
        splitK = 4
    return splitK


def generate_data(
    m, n, k, indtype, outdtype, scaleAB, seed=0, bias=None, device="cuda:0"
):
    torch.manual_seed(seed)
    inp = torch.randn((m, k), device=device).to(indtype)
    weights = torch.randn((n, k), device=device).to(indtype)
    # blob = torch.ones(128 * 1024 * 1024, dtype=dtypes.fp32, device=device)
    bias = torch.randn(n, device=device).to(outdtype) if bias else None
    scale_half = torch.tensor(0.5, dtype=dtypes.fp32, device=device)
    scale_one = torch.tensor(1, dtype=dtypes.fp32, device=device)
    scale = scale_half if scaleAB else scale_one
    out_asm = torch.empty(m, n, dtype=outdtype, device=device)
    return (inp, weights, weights.t(), bias, scale, out_asm)


def get_gemm_ref(inp, weights, bias, scale, indtype, outdtype):
    scaleA = scale
    scaleB = scale
    if indtype == dtypes.fp8:
        try:
            ref = torch._scaled_mm(
                inp,
                weights.t(),
                bias=bias,
                scale_a=scaleA,
                scale_b=scaleB,
                out_dtype=outdtype,
            )
        except RuntimeError:
            ref = (
                F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32)) * scaleA * scaleB
            )
            ref = (ref.to(outdtype) + bias) if bias is not None else ref.to(outdtype)
        if type(ref) is tuple and len(ref) == 2:
            ref = ref[0]
    else:
        ref = F.linear(inp, weights, bias).to(outdtype)
    return ref


rtol = 1e-5
atol = 1

CACHE_INVALIDATE_BUFFERS = int(os.getenv("CACHE_INVALIDATE_BUFFERS", "37"))
ONE = torch.ones(1, dtype=dtypes.fp32, device="cuda")
HALF = torch.tensor(0.5, dtype=dtypes.fp32, device="cuda")


class Gemm:

    def __init__(
        self,
        m,
        n,
        k,
        bias,
        indtype,
        outdtype,
        scaleAB=False,
        mp=1,
        err_ratio=0.01,
        profile_file="",
        # splitK=None,
    ):
        torch.cuda.empty_cache()
        self.m = m
        self.k = k
        self.n = n
        self.bias = torch.randn(n, device="cuda").to(outdtype) if bias else None
        self.indtype = indtype
        self.outdtype = outdtype
        self.scaleAB = scaleAB
        self.nb = CACHE_INVALIDATE_BUFFERS
        (self.inp, self.weights, _, self.bias, _, scaleA) = generate_data(
            m, n, k, indtype, outdtype, scaleAB, 0, bias
        )
        self.blob = torch.ones(128 * 1024 * 1024, dtype=dtypes.fp32, device="cuda")
        self.topn = 20  # number of top solutions from each source
        self.hipb_sols = []
        self.rtol = 1e-2
        self.atol = 1e-2
        # self.ref = self.get_gemm_ref()
        self.check_err_ratio = err_ratio
        self.splitK = None
        self.profile_file = profile_file
        # self.start = torch.cuda.Event(enable_timing=True)
        # self.end = torch.cuda.Event(enable_timing=True)
        # prefer hipblaslt unless rocblas time is less than this
        # ratio of hipblaslt time
        self.hipb_prefer_ratio = 0.995
        self.mp = mp
        # self.inbpe = self.inp.element_size()
        # self.outbpe = self.ref.element_size()
        self.asm_map = {}

    def find_hipblas_sols(self):
        sols = aiter.hipb_findallsols(
            self.inp,
            self.weights.t(),
            bias=self.bias,
            out_dtype=self.outdtype,
            scaleA=HALF if self.scaleAB else None,
            scaleB=HALF if self.scaleAB else None,
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
        # print(sols)
        self.hipb_sols = sols

    def get_gemm_ref(self):
        scaleA = HALF if self.scaleAB else ONE
        scaleB = HALF if self.scaleAB else ONE
        if self.indtype == dtypes.fp8:
            try:
                ref = torch._scaled_mm(
                    self.inp,
                    self.weights.t(),
                    bias=self.bias,
                    scale_a=scaleA,
                    scale_b=scaleB,
                    out_dtype=self.outdtype,
                )
            except RuntimeError:
                ref = (
                    F.linear(self.inp.to(dtypes.fp32), self.weights.to(dtypes.fp32))
                    * scaleA
                    * scaleB
                )
                ref = (
                    (ref.to(self.outdtype) + self.bias)
                    if self.bias is not None
                    else ref.to(self.outdtype)
                )
            if type(ref) is tuple and len(ref) == 2:
                ref = ref[0]
        else:
            ref = F.linear(self.inp, self.weights, self.bias).to(self.outdtype)
        return ref

    def get_asm_kernels(self, file):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        if "bPreshuffle" in df.columns:
            df = df[df["bPreshuffle"] != 1]
        kernel_dict = (
            df.groupby(["tileM", "tileN", "pf"])["knl_name"].apply(list).to_dict()
        )
        return kernel_dict

    def asm_gemm_all_solutions(self):
        if (
            self.scaleAB
            or self.k % 64 != 0
            or (not (self.indtype == dtypes.bf16 and self.outdtype == dtypes.fp32))
        ):
            self.asm_gtimedf = pd.DataFrame(columns=["gtimems", "libtype"])
            return []
        asm_kernel_list_csv = f"{get_asm_dir()}/bf16gemm/bf16gemm_outf32.csv"
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
        asm_tiles = [key for key in asm_kernels.keys()]
        solidx = 0
        task_asm = []

        solutions = 0
        for key in asm_tiles:
            tile_m, tile_n, pf = key
            print(f"ASM Tile - M: {tile_m}, N: {tile_n}, PF: {pf}")
            kernelName = asm_kernels[key][0]
            maxSplitK = compute_gemm_SplitK(
                self.m, self.n, self.k, tile_m, tile_n, 256
            )  # if self.splitK else 1
            solidx = solidx + 1
            self.asm_map[solidx] = kernelName
            for splitK in range(1, maxSplitK + 1):
                info = (
                    (
                        self.m,
                        self.n,
                        self.k,
                        False,
                        str(self.indtype),
                        str(self.outdtype),
                        self.scaleAB,
                    ),
                    solidx,
                    splitK,
                    "asm",
                    kernelName,
                )
                task_asm.append(
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
                        ),
                        run_gemm_bf16_asm,
                        ([0, 1, 5, 3], splitK, kernelName),
                        {},
                        get_gemm_ref,
                        ([0, 1, 3, 4], self.indtype, self.outdtype),
                        {},
                        None,  # self.ref if fast_mode == 0 else None,
                        self.rtol,
                        self.atol,
                    )
                )

                solutions = solutions + 1
        in_data = [
            (
                solutions,
                (),
            )
        ]
        ret = mp_tuner(task_asm, in_data, self.mp, False)
        return ret

    def hipb_time_all_sols(self, fast_mode=0, top_sols=0):
        coldi = 20
        warmi = 20
        if fast_mode:
            coldi = 2
            warmi = 5
        solutions = self.hipb_sols
        if top_sols:
            solutions = self.hipb_top_sols
        task = []
        scaleA = HALF if self.scaleAB else None
        scaleB = HALF if self.scaleAB else None

        gtimes = {}
        for solidx in solutions:
            info = (
                (
                    self.m,
                    self.n,
                    self.k,
                    False,
                    str(self.indtype),
                    str(self.outdtype),
                    self.scaleAB,
                ),
                solidx,
                0,  # splitK
                "hipblaslt",
                "",
            )
            task.append(
                (
                    info,
                    generate_data,
                    (self.m, self.n, self.k, self.indtype, self.outdtype, self.scaleAB),
                    call_hipb_mm,
                    ([0, 2, 3, 4, 4], solidx, self.outdtype),
                    {
                        "num_warmup": warmi,
                        "num_iters": coldi,
                    },
                    get_gemm_ref if fast_mode == 0 else None,
                    ([0, 1, 3, 4], self.indtype, self.outdtype),
                    {},
                    None,  # self.ref if fast_mode == 0 else None,
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
        ret = mp_tuner(task, in_data, self.mp, fast_mode == 1)
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
            kernelName = info[4]
            # if fast_mode == 0:
            #    if err_ratio > self.check_err_ratio:
            #        continue
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
        rets_hipb_fast = self.hipb_time_all_sols(fast_mode=1)

    def run_best_solutions(self):
        self.warmup()
        rets_hipb = self.hipb_time_all_sols(fast_mode=0, top_sols=1)
        rets_asm = self.asm_gemm_all_solutions()
        return rets_hipb + rets_asm

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
        "batch": 1,
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
        key=["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "cu_num"],
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
        description="GemmTuner",
    ):
        super().__init__(
            "GemmTuner",
            key=key,
            resultList=resultList,
            description=description,
        )

        self.hipb_prefer_ratio = 0.995
        self.cu_num = self.get_cu_num()
        self.gemmobj = None

    def calculate_perf(
        self,
        results,
        inbpe,
        outbpe,
    ):
        """calculate TFLOPS and bandwidth"""
        ### gemm flops,bw
        info, time, err_ratio = results
        if time <= 0:
            return -1, -1
        cu_num, m, n, k = info
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
                self.untunedf["outdtype"] = ""
            if "scaleAB" not in self.untunedf.columns:
                self.untunedf["scaleAB"] = False
            if args.indtype is not None:
                self.untunedf["dtype"] = str(args.indtype)
            if args.outdtype is not None:
                self.untunedf["outdtype"] = str(args.outdtype)
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
                        )
            self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))
            self.untunedf["cu_num"] = self.get_cu_num()
            untunedf_cols = self.untunedf.columns
            if len(self.tunedf) != 0:
                mask = self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[untunedf_cols].apply(tuple, axis=1)
                )
                if args.verbose:
                    logger.info("skiped tuned shapes:")
                    print(self.untunedf[mask])
                self.untunedf = self.untunedf[~mask]
            self.untunedf.drop_duplicates().reset_index(drop=True)

    def add_gemm(self, m, n, k, indtype, bias=False, outdtype=None, scaleAB=False):
        assert indtype is not None
        outdtype = outdtype if outdtype is not None else indtype
        assert outdtype is not None
        print(self.tunedf)
        if self.tunedf is None or (
            self.tunedf[
                (self.tunedf["cu_num"] == self.cu_num)
                & (self.tunedf["M"] == m)
                & (self.tunedf["N"] == n)
                & (self.tunedf["K"] == k)
                & (self.tunedf["bias"] == bias)
                & (self.tunedf["dtype"] == str(indtype))
                & (self.tunedf["outdtype"] == str(outdtype))
            ].empty
        ):
            entry = {
                "M": [m],
                "N": [n],
                "K": [k],
                "bias": [bias],
                "dtype": [indtype],
                "outdtype": [outdtype],
                "scaleAB": [scaleAB],
            }
            df = pd.DataFrame(entry)
            self.untunedf = pd.concat([self.untunedf, df], ignore_index=True)
            print("self untunedf is ", self.untunedf)
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
            gemmobj = Gemm(
                ds["M"],
                ds["N"],
                ds["K"],
                ds["bias"],
                indtype=eval(indtype),
                outdtype=eval(outdtype),
                scaleAB=ds["scaleAB"],
                mp=args.mp,
                err_ratio=args.errRatio,
                profile_file=args.profile_file,
            )

            ret.extend(gemmobj.run_solutions())
            gemmobj.cleanup()
            del gemmobj

        return ret

    def processResult(self, rets, fast_mode):
        results = []
        for info, us, err_ratio in rets:
            res_one = []
            solidx = info[1]
            splitK = info[2]
            kernelName = info[4]
            libtype = info[3]
            for ele in info[0]:
                res_one.append(ele)
            res_one.append(get_cu_num())
            res_one.append(libtype)
            res_one.append(int(solidx))
            res_one.append(int(splitK))
            res_one.append(round(us, 4))

            res_one.append(kernelName)
            res_one.append(err_ratio)
            ret = (
                (self.cu_num, info[0][0], info[0][1], info[0][2]),
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
        import numpy as np

        best_gtimedfs = pd.DataFrame(columns=self.columns)
        for key, df in gtimedf_dic.items():
            gtimedf_dic[key] = df[df["err_ratio"] < args.errRatio]
            # get best solution
            best_gtimedf = gtimedf_dic[key].sort_values(by="us")

            if len(gtimedf_dic[key]) == 0:
                print(">>> No  hipblas or asm solutions found!", flush=True)
                continue
            asm_gtimedf = gtimedf_dic[key][gtimedf_dic[key]["libtype"] == "asm"]
            hibs_gtimedf = gtimedf_dic[key][gtimedf_dic[key]["libtype"] == "hipblaslt"]
            if len(hibs_gtimedf) == 0:
                print(">>>Only asm solutions found!", flush=True)
            elif len(asm_gtimedf) == 0:
                print(">>> no hipblas solutions found!", flush=True)
            resultdf1 = best_gtimedf.head(1).reset_index(drop=True)
            kernal_name = (
                aiter.getHipblasltKernelName(int(resultdf1.iloc[0]["solidx"]))
                if resultdf1.iloc[0]["libtype"] == "hipblaslt"
                else resultdf1.iloc[0]["kernelName"]
            )
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
