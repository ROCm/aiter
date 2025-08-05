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

import os
import random
from pathlib import Path

import aiter
import pandas as pd
from aiter import dtypes
import torch
import torch.nn.functional as F
from aiter.test_common import perftest
from aiter.utility.mp_tuner import mp_tuner
from functools import lru_cache

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
    return (inp, weights, weights.t(), bias, scale)


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
        (self.inp, self.weights, _, self.bias, scaleA) = generate_data(
            m, n, k, indtype, outdtype, scaleAB, 0, bias
        )
        self.blob = torch.ones(128 * 1024 * 1024, dtype=dtypes.fp32, device="cuda")
        self.topn = 20  # number of top solutions from each source
        self.hipb_sols = []
        self.rocb_sols = []
        self.rtol = 1e-2
        self.atol = 1e-2
        self.ref = self.get_gemm_ref()
        self.check_err_ratio = 0.01
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # prefer hipblaslt unless rocblas time is less than this
        # ratio of hipblaslt time
        self.hipb_prefer_ratio = 0.995
        self.rocblas_decode = rocblas_decode
        self.mp = mp

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
                    get_gemm_ref,
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
        for info, us, err_ratio in ret:
            solidx = info[1]
            if fast_mode == 0:
                if err_ratio > self.check_err_ratio:
                    continue
            gtimes[solidx] = us / 1000.0
        self.hipb_gtimedf = pd.DataFrame.from_dict(
            gtimes, orient="index", columns=["gtimems"]
        ).sort_values(by="gtimems")
        self.hipb_gtimedf.to_csv("/tmp/hipb_gtimedf.csv")
        print(">>> HipBlasLt top solutions, Fast Mode", fast_mode)
        print(self.hipb_gtimedf.head(self.topn))

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
                    get_gemm_ref,
                    ([0, 1, 3, 4], self.indtype, self.outdtype),
                    {},
                    None,  # self.ref if fast_mode == 0 else None,
                    self.rtol,
                    self.atol,
                )
            )
        in_data = [(len(solutions), ())]
        ret = mp_tuner(task, in_data, self.mp, fast_mode == 1)
        for info, us, err_ratio in ret:
            solidx = info[1]
            if fast_mode == 0:
                if err_ratio > self.check_err_ratio:
                    continue
            gtimes[solidx] = us / 1000.0
        self.rocb_gtimedf = pd.DataFrame.from_dict(
            gtimes, orient="index", columns=["gtimems"]
        ).sort_values(by="gtimems")
        self.rocb_gtimedf.to_csv("/tmp/rocb_gtimedf.csv")
        print(">>> Rocblas top solutions, Fast Mode", fast_mode, flush=True)
        print(self.rocb_gtimedf.head(self.topn), flush=True)

    def warmup(self, warmi=500):
        for i in range(warmi):
            self.blob = self.blob + 0.00001

    def functional_get_topn_fastest(self):
        rocb_topn = []
        for solidx in self.rocb_gtimedf.index[: self.topn]:
            rocb_topn.append(solidx)
        self.rocb_top_sols = rocb_topn
        hipb_topn = []
        for solidx in self.hipb_gtimedf.index[: self.topn]:
            hipb_topn.append(solidx)
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
        if len(self.rocb_gtimedf) > 0 and len(self.hipb_gtimedf) > 0:
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            if best_rocb_time < best_hipb_time * self.hipb_prefer_ratio:
                self.best_libtype = "rocblas"
                self.best_solidx = self.rocb_gtimedf.index[0]
                self.best_soltime = best_rocb_time
            else:
                self.best_libtype = "hipblaslt"
                self.best_solidx = self.hipb_gtimedf.index[0]
                self.best_soltime = best_hipb_time
            # self.check_gemm_ref(self.best_libtype,self.best_solidx)
        elif len(self.hipb_gtimedf) > 0:
            print(">>> Only hipblas solutions found!", flush=True)
            best_hipb_time = self.hipb_gtimedf.gtimems.iloc[0]
            self.best_libtype = "hipblaslt"
            self.best_solidx = self.hipb_gtimedf.index[0]
            self.best_soltime = best_hipb_time
        elif len(self.rocb_gtimedf) > 0:
            print(">>> Only rocblas solutions found!", flush=True)
            best_rocb_time = self.rocb_gtimedf.gtimems.iloc[0]
            self.best_libtype = "rocblas"
            self.best_solidx = self.rocb_gtimedf.index[0]
            self.best_soltime = best_rocb_time
        else:
            print(">>> No rocblas or hipblas solutions found!", flush=True)
            self.best_libtype = "rocblas"
            self.best_solidx = 0
            self.best_soltime = 0
        print(
            ">>> Fastest Solution is",
            self.best_libtype,
            self.best_solidx,
            self.best_soltime,
            flush=True,
        )


class GemmTuner:

    def __init__(self, indtype, outdtype, tuned_file=None, rocblas_decode=False, mp=1):
        self.gemm_problems = pd.DataFrame(columns=["M", "N", "K", "bias"])
        self.indtype = indtype
        self.outdtype = outdtype
        self.rocblas_decode = rocblas_decode
        self.tuned_file = tuned_file
        self.mp = mp
        if Path(tuned_file).is_file():
            self.tuned_shapes = pd.read_csv(tuned_file)
        else:
            self.tuned_shapes = None

    def add_gemm(self, m, n, k, indtype, bias=False, outdtype=None, scaleAB=False):
        assert indtype is not None
        outdtype = outdtype if outdtype is not None else indtype
        assert outdtype is not None
        if self.tuned_shapes is None or (
            self.tuned_shapes[
                (self.tuned_shapes["M"] == m)
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
        soldf = pd.DataFrame(columns=["libtype", "solidx", "soltimes", "kernelName"])
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
            )
            gemmobj.find_fastest_solution()
            soldf.loc[i, "libtype"] = gemmobj.best_libtype
            soldf.loc[i, "solidx"] = gemmobj.best_solidx
            soldf.loc[i, "soltimes"] = round(gemmobj.best_soltime * 1000, 2)
            soldf.loc[i, "kernelName"] = (
                aiter.getHipblasltKernelName(int(gemmobj.best_solidx))
                if gemmobj.best_libtype == "hipblaslt"
                else ""
            )

            del gemmobj
            torch.cuda.empty_cache()

        finaldf = pd.concat([self.gemm_problems, soldf], axis=1)
        if self.tuned_shapes is not None:
            finaldf = pd.concat([finaldf, self.tuned_shapes])
        finaldf["solidx"] = finaldf["solidx"].convert_dtypes("int64")
        finaldf.to_csv(self.tuned_file, index=False)
        print(finaldf)
