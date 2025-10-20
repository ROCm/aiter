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
import random
from pathlib import Path

import aiter
import pandas as pd
from aiter import dtypes
import torch
import torch.nn.functional as F

from aiter.utility.mp_tuner import mp_tuner
from functools import lru_cache
from aiter.jit.core import get_asm_dir
from aiter.jit.utils.chip_info import get_cu_num

aiter.rocb_create_extension()
aiter.hipb_create_extension()


@lru_cache(maxsize=1)
def init_hipblas():
    aiter.hipb_create_extension()


@lru_cache(maxsize=1)
def init_rocblas():
    aiter.rocb_create_extension()


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


def call_rocb_mm(inp, w, solidx):
    init_rocblas()
    return aiter.rocb_mm(inp, w, solidx)


def run_gemm_bf16_asm(inp, w, out, bias=None, splitK=None, kernelName=None):
    return aiter.gemm_a16w16_asm(
        inp, w, out, bias=bias, splitK=splitK, kernelName=kernelName
    )


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
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
        rocblas_decode=False,
        mp=1,
        err_ratio=0.01,
        profile_file="",
        # splitK=None,
    ):
        self.m = m
        self.k = k
        self.n = n
        self.bias = torch.randn(n, device="cuda").to(outdtype) if bias else None
        self.indtype = indtype
        self.outdtype = outdtype
        self.scaleAB = scaleAB
        self.use_rocblas = indtype == outdtype and str(indtype) != "dtypes.fp8"
        self.nb = CACHE_INVALIDATE_BUFFERS
        (self.inp, self.weights, _, self.bias, _, scaleA) = generate_data(
            m, n, k, indtype, outdtype, scaleAB, 0, bias
        )
        self.blob = torch.ones(128 * 1024 * 1024, dtype=dtypes.fp32, device="cuda")
        self.topn = 20  # number of top solutions from each source
        self.hipb_sols = []
        self.rocb_sols = []
        self.rtol = 1e-2
        self.atol = 1e-2
        self.ref = self.get_gemm_ref()
        self.check_err_ratio = err_ratio
        self.splitK = None
        self.profile_file = profile_file
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # prefer hipblaslt unless rocblas time is less than this
        # ratio of hipblaslt time
        self.hipb_prefer_ratio = 0.995
        self.rocblas_decode = rocblas_decode
        self.mp = mp
        self.inbpe = self.inp.element_size()
        self.outbpe = self.ref.element_size()
        self.asm_map = {}

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
        print("info is ", info)
        cu_num, m, n, k = info
        flops = m * n * k * 2
        tflops = round(flops / (time * 1000000), 2)

        bw = round(
            (m * k * inbpe + n * k * inbpe + m * n * outbpe) / (time * 1e-6) / 1e9,
            2,
        )
        return tflops, bw

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
        kernel_dict = (
            df.groupby(["tileM", "tileN", "pf"])["knl_name"].apply(list).to_dict()
        )
        return kernel_dict

    def asm_gemm_all_solutions(self):
        if (
            self.scaleAB
            # or self.k % 64 != 0
            or (not (self.indtype == dtypes.bf16 and self.outdtype == dtypes.fp32))
        ):
            self.asm_gtimedf = pd.DataFrame(columns=["gtimems", "libtype"])
            return
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
                        self.indtype,
                        self.outdtype,
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
        gtimes = {}
        ret = mp_tuner(task_asm, in_data, self.mp, False)

        results = []
        for info, us, err_ratio in ret:
            solidx = info[1]
            splitK = info[2]
            kernelName = info[4]
            res_one = []
            # if err_ratio > self.check_err_ratio:
            #    continue
            gtimes["solidx"] = solidx
            gtimes["gtimems"] = us / 1000.0
            gtimes["splitK"] = splitK
            res_one.append(solidx)
            res_one.append(round(us / 1000.0, 4))
            res_one.append(splitK)
            res_one.append(err_ratio)
            res_one.append(kernelName)
            results.append(res_one)

        self.asm_gtimedf = pd.DataFrame(
            results, columns=["solidx", "gtimems", "splitK", "err_ratio", "kernelName"]
        )
        self.asm_gtimedf["libtype"] = "asm"
        print(self.asm_gtimedf)
        self.asm_gtimedf.to_csv("/tmp/asm_gtimedf.csv", index=False)
        self.asm_gtimedf = self.asm_gtimedf.sort_values(by="gtimems")
        print(">>> asm top solutions, Fast Mode", 0)
        print(self.asm_gtimedf.head(self.topn))

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
                    self.indtype,
                    self.outdtype,
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
        results = []
        for info, us, err_ratio in ret:
            res_one = []
            solidx = info[1]
            kernelName = info[4]
            if fast_mode == 0:
                if err_ratio > self.check_err_ratio:
                    continue
            res_one.append(solidx)
            res_one.append(round(us / 1000.0, 4))
            res_one.append(err_ratio)
            res_one.append(kernelName)

            results.append(res_one)
        self.hipb_gtimedf = pd.DataFrame(
            results, columns=["solidx", "gtimems", "err_ratio", "kernelName"]
        )

        self.hipb_gtimedf = self.hipb_gtimedf.sort_values(by="gtimems")
        self.hipb_gtimedf["libtype"] = "hipblaslt"

        self.hipb_gtimedf.to_csv("/tmp/hipb_gtimedf.csv", index=False)
        print(">>> HipBlasLt top solutions, Fast Mode", fast_mode)
        print(self.hipb_gtimedf.head(self.topn))
        self.hipb_gtimedf["splitK"] = 0

    def find_rocblas_sols(self):
        if self.scaleAB or self.bias is not None:
            sols = []
        else:
            sols = aiter.rocb_findallsols(self.inp, self.weights.t())
        print(
            "M N K dtype",
            self.m,
            self.n,
            self.k,
            self.indtype,
            self.outdtype,
            ">>> Total rocb solutions",
            len(sols),
            flush=True,
        )
        # print(sols)
        self.rocb_sols = sols

    def rocb_time_all_sols(self, fast_mode=0, top_sols=0):
        coldi = 20
        warmi = 20
        if fast_mode:
            coldi = 2
            warmi = 5
        solutions = self.rocb_sols
        if top_sols:
            solutions = self.rocb_top_sols
        task = []
        gtimes = {}
        for solidx in solutions:
            info = (
                (self.m, self.n, self.k, False, self.indtype, self.outdtype, False),
                solidx,
                0,
                "rocblas",
                "rocblas",
            )
            task.append(
                (
                    info,
                    generate_data,
                    (self.m, self.n, self.k, self.indtype, self.outdtype, False),
                    call_rocb_mm,
                    (
                        [0, 2],
                        solidx,
                    ),
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
        in_data = [(len(solutions), ())]
        ret = mp_tuner(task, in_data, self.mp, fast_mode == 1)
        results = []
        for info, us, err_ratio in ret:
            solidx = info[1]
            kernelName = info[4]
            ret_one = []
            if fast_mode == 0:
                if err_ratio > self.check_err_ratio:
                    continue
            ret_one.append(solidx)
            ret_one.append(round(us / 1000.0, 4))
            ret_one.append(err_ratio)
            ret_one.append(kernelName)
            results.append(ret_one)
        self.rocb_gtimedf = pd.DataFrame(
            results, columns=["solidx", "gtimems", "err_ratio", "kernelName"]
        )
        self.rocb_gtimedf = self.rocb_gtimedf.sort_values(by="gtimems")
        self.rocb_gtimedf["libtype"] = "rocblas"
        self.rocb_gtimedf["splitK"] = 0
        self.rocb_gtimedf.to_csv("/tmp/rocb_gtimedf.csv", index=False)
        print(">>> Rocblas top solutions, Fast Mode", fast_mode, flush=True)
        print(self.rocb_gtimedf.head(self.topn), flush=True)

    def warmup(self, warmi=500):
        for i in range(warmi):
            self.blob = self.blob + 0.00001

    def functional_get_topn_fastest(self):
        rocb_topn = self.rocb_gtimedf["solidx"].head(self.topn).tolist()
        self.rocb_top_sols = rocb_topn
        hipb_topn = self.hipb_gtimedf["solidx"].head(self.topn).tolist()
        self.hipb_top_sols = hipb_topn

    def find_fastest_solution(self):
        if self.use_rocblas:
            self.find_rocblas_sols()
        if not (self.rocblas_decode and self.m == 1):
            self.find_hipblas_sols()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=1)
        self.functional_get_topn_fastest()
        self.warmup()
        self.rocb_time_all_sols(fast_mode=0, top_sols=1)
        self.warmup()
        self.hipb_time_all_sols(fast_mode=0, top_sols=1)
        self.asm_gemm_all_solutions()
        if self.profile_file != "":
            if os.path.exists(self.profile_file):
                old_df = pd.read_csv(self.profile_file)
            else:
                old_df = pd.DataFrame(
                    columns=[
                        "M",
                        "N",
                        "K",
                        "bias",
                        "dtype",
                        "outdtype",
                        "scaleAB",
                        "cu_num",
                        "libtype",
                        "solidx",
                        "splitK",
                        "soltimes",
                        "kernelName",
                        "err_ratio",
                    ]
                )

            resultsdf = pd.concat(
                [self.rocb_gtimedf, self.hipb_gtimedf, self.asm_gtimedf],
                ignore_index=True,
            )
            resultsdf = resultsdf.rename(
                columns={
                    "gtimems": "soltimes",
                }
            )
            resultsdf["soltimes"] = resultsdf["soltimes"].apply(
                lambda x: round(x * 1000, 3)
            )
            print(resultsdf)
            resultsdf["M"] = self.m
            resultsdf["N"] = self.n
            resultsdf["K"] = self.k
            resultsdf["bias"] = self.bias
            resultsdf["dtype"] = self.indtype
            resultsdf["outdtype"] = self.outdtype
            resultsdf["scaleAB"] = self.scaleAB
            resultsdf["cu_num"] = get_cu_num()
            keys = [
                "cu_num",
                "M",
                "N",
                "K",
            ]
            results = resultsdf.apply(
                lambda row: self.calculate_perf(
                    (
                        tuple(row[col] for col in keys),
                        row["soltimes"],
                        row["err_ratio"],
                    ),
                    self.inbpe,
                    self.outbpe,
                ),
                axis=1,
                result_type="expand",
            )
            resultsdf["tflops"] = results[0]
            resultsdf["bw"] = results[1]

            resultsdf = pd.concat([old_df, resultsdf], ignore_index=True)
            resultsdf.to_csv(self.profile_file, index=False)
        if len(self.asm_gtimedf) > 0:
            self.asm_gtimedf = self.asm_gtimedf[
                self.asm_gtimedf["err_ratio"] < self.check_err_ratio
            ]
            print(self.asm_gtimedf)
        if len(self.hipb_gtimedf) > 0 or len(self.asm_gtimedf) > 0:
            self.hipb_gtimedf = pd.concat(
                [self.hipb_gtimedf, self.asm_gtimedf], ignore_index=True
            )
        # get best solution
        self.hipb_gtimedf = self.hipb_gtimedf.sort_values(by="gtimems")
        print("rocb_gtimedf", self.rocb_gtimedf)
        if len(self.rocb_gtimedf) > 0 and len(self.hipb_gtimedf) > 0:
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            best_asm_time = 100  # self.asm_gtimedf.gtimems.iloc[0]
            self.best_kernelName = ""
            if best_rocb_time < best_hipb_time * self.hipb_prefer_ratio:
                self.best_libtype = self.rocb_gtimedf.libtype.iloc[0]
                self.best_solidx = self.rocb_gtimedf.solidx.iloc[0]
                self.best_soltime = best_rocb_time
                self.best_splitK = self.rocb_gtimedf.splitK.iloc[0]
                self.best_err_ratio = self.rocb_gtimedf.err_ratio.iloc[0]
                self.best_kernelName = self.rocb_gtimedf.kernelName.iloc[0]
            else:
                self.best_libtype = "hipblaslt"
                self.best_solidx = self.hipb_gtimedf.solidx.iloc[0]
                self.best_soltime = best_hipb_time
                self.best_splitK = self.hipb_gtimedf.splitK.iloc[0]
                self.best_err_ratio = self.hipb_gtimedf.err_ratio.iloc[0]
            # self.check_gemm_ref(self.best_libtype,self.best_solidx)
        elif len(self.hipb_gtimedf) > 0:
            print(">>> Only hipblas or asm solutions found!", flush=True)
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            self.best_libtype = self.hipb_gtimedf.libtype.iloc[0]
            self.best_solidx = self.hipb_gtimedf.solidx.iloc[0]
            self.best_soltime = best_hipb_time
            self.best_splitK = self.hipb_gtimedf.splitK.iloc[0]
            self.best_err_ratio = self.hipb_gtimedf.err_ratio.iloc[0]
            self.best_kernelName = self.hipb_gtimedf.kernelName.iloc[0]
        elif len(self.rocb_gtimedf) > 0:
            print(">>> Only rocblas solutions found!", flush=True)
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            self.best_libtype = self.rocb_gtimedf.libtype.iloc[0]
            self.best_solidx = self.rocb_gtimedf.solidx.iloc[0]
            self.best_soltime = best_rocb_time
            self.best_splitK = self.rocb_gtimedf.splitK.iloc[0]
            self.best_err_ratio = self.rocb_gtimedf.err_ratio.iloc[0]
            self.best_kernelName = self.rocb_gtimedf.kernelName.iloc[0]
        else:
            print(">>> No rocblas or hipblas or asm solutions found!", flush=True)
            self.best_libtype = "rocblas"
            self.best_solidx = 0
            self.best_soltime = 0
            self.best_splitK = 0
            self.best_err_ratio = 0
            self.best_kernelName = ""
        print(
            ">>> Fastest Solution is",
            self.best_libtype,
            self.best_solidx,
            self.best_soltime,
            self.best_splitK,
            self.best_err_ratio,
            flush=True,
        )


class GemmTuner:

    def __init__(
        self,
        indtype,
        outdtype,
        tuned_file=None,
        rocblas_decode=False,
        mp=1,
        err_ratio=0.01,
        profile_file="",
    ):
        self.gemm_problems = pd.DataFrame(columns=["M", "N", "K", "bias"])
        self.indtype = indtype
        self.outdtype = outdtype
        self.rocblas_decode = rocblas_decode
        self.tuned_file = tuned_file
        self.mp = mp
        self.err_ratio = err_ratio
        self.profile_file = profile_file

        tuned_file_path = Path(tuned_file)
        if tuned_file_path.exists():
            self.tuned_shapes = (
                pd.read_csv(tuned_file) if tuned_file_path.is_file() else None
            )
        else:
            self.tuned_shapes = None
            with open(tuned_file, "w") as tf:
                tf.write(
                    "M,N,K,bias,dtype,outdtype,scaleAB,cu_num,libtype,solidx,splitK,soltimes,kernelName,err_ratio,tflops,bw\n"
                )

        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        self.cu_num = cu_num

    def add_gemm(self, m, n, k, indtype, bias=False, outdtype=None, scaleAB=False):
        assert indtype is not None
        outdtype = outdtype if outdtype is not None else indtype
        assert outdtype is not None
        if self.tuned_shapes is None or (
            self.tuned_shapes[
                (self.tuned_shapes["cu_num"] == self.cu_num)
                & (self.tuned_shapes["M"] == m)
                & (self.tuned_shapes["N"] == n)
                & (self.tuned_shapes["K"] == k)
                & (self.tuned_shapes["bias"] == bias)
                & (self.tuned_shapes["dtype"] == str(indtype))
                & (self.tuned_shapes["outdtype"] == str(outdtype))
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
            self.gemm_problems = pd.concat([self.gemm_problems, df], ignore_index=True)
        else:
            print(
                f">>>Info: Found Duplicate shape(M:{m},"
                f" N:{n}, K:{k} bias:{bias}), skipping"
            )

    def find_best_sols(self):
        df = self.gemm_problems
        for i in range(len(df)):
            ds = df.loc[i, :]
            indtype = ds["dtype"]
            outdtype = ds["outdtype"]

            gemmobj = Gemm(
                ds["M"],
                ds["N"],
                ds["K"],
                ds["bias"],
                indtype=indtype,
                outdtype=outdtype,
                scaleAB=ds["scaleAB"],
                rocblas_decode=self.rocblas_decode,
                mp=self.mp,
                err_ratio=self.err_ratio,
                profile_file=self.profile_file,
            )
            gemmobj.find_fastest_solution()

            soltimes = round(gemmobj.best_soltime * 1000, 2)
            splitK = gemmobj.best_splitK
            err_ratio = gemmobj.best_err_ratio
            kernal_name = (
                aiter.getHipblasltKernelName(int(gemmobj.best_solidx))
                if gemmobj.best_libtype == "hipblaslt"
                else gemmobj.best_kernelName
            )
            ret = (
                (self.cu_num, ds["M"], ds["N"], ds["K"]),
                soltimes,
                gemmobj.check_err_ratio,
            )
            tflops, bw = gemmobj.calculate_perf(
                ret,
                gemmobj.inbpe,
                gemmobj.outbpe,
            )
            with open(self.tuned_file, "a") as tf:
                tf.write(
                    f"{ds['M']},{ds['N']},{ds['K']},{ds['bias']},{indtype},{outdtype},{ds['scaleAB']},"
                    f"{self.cu_num},{gemmobj.best_libtype},{int(gemmobj.best_solidx)},{int(splitK)},{(soltimes)},{kernal_name},{err_ratio},{tflops},{bw}\n"
                )

            del gemmobj
            torch.cuda.empty_cache()

        finaldf = pd.read_csv(self.tuned_file)
        print(finaldf)
