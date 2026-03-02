# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE, AITER_CSRC_DIR
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.ops.shuffle import shuffle_weight
from gemm_a8w8_bpreshuffle_common import kernels_list as kernels_list_ck

import argparse
from aiter.utility.mp_tuner import mp_tuner
from aiter.jit.core import get_asm_dir

sys.path.insert(0, f"{AITER_CSRC_DIR}/cktile_gemm_a8w8_bpreshuffle/")
from gemm_a8w8_bpreshuffle_cktile_common import kernels_list as kernels_list_cktile

sys.path.insert(0, f"{AITER_CSRC_DIR}/flydsl_gemm_a8w8_bpreshuffle/")
try:
    from flydsl_gemm_a8w8_bpreshuffle_common import kernels_list as kernels_list_flydsl
except ImportError:
    print(
        "[FlyDSL] flydsl_gemm_a8w8_bpreshuffle_common.py not found, flydsl tuning disabled"
    )
    kernels_list_flydsl = {}

DSL2_ROOT = os.environ.get("DSL2_ROOT", None)


def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel() / a.numel()
        if percent > 0.01:
            return False
        else:
            return True


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_bpreshuffle_cktile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK=0
):
    aiter.gemm_a8w8_bpreshuffle_cktile_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_asm(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    bias,
    kernelName,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    splitK=None,
):

    return aiter.gemm_a8w8_asm(
        x,
        weight,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        splitK=splitK,
    )


_flydsl_compile_fn = None
_flydsl_exe_cache = {}


def _get_flydsl_compile_fn():
    global _flydsl_compile_fn
    if _flydsl_compile_fn is not None:
        return _flydsl_compile_fn
    if not DSL2_ROOT:
        raise RuntimeError(
            "FlyDSL path not found. Please set environment variable, e.g. "
            "`export DSL2_ROOT=/path/to/FlyDSL`"
        )
    try:
        if DSL2_ROOT not in sys.path:
            sys.path.insert(0, DSL2_ROOT)
        from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

        _flydsl_compile_fn = compile_preshuffle_gemm_a8
    except Exception as e:
        print(f"[FlyDSL] compile function not available: {e}")
    return _flydsl_compile_fn


def run_gemm_flydsl(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out,
    tile_m,
    tile_n,
    tile_k,
    in_dtype,
    lds_stage,
    use_cshuffle_epilog,
    use_async_copy,
    waves_per_eu=0,
):
    compile_fn = _get_flydsl_compile_fn()
    if compile_fn is None:
        raise RuntimeError("[FlyDSL] compile function not available")

    m, k = x.shape
    n = out.shape[1]
    wpe = None if int(waves_per_eu) <= 0 else int(waves_per_eu)

    cache_key = (
        m,
        n,
        k,
        tile_m,
        tile_n,
        tile_k,
        in_dtype,
        lds_stage,
        use_cshuffle_epilog,
        use_async_copy,
        wpe,
    )
    if cache_key not in _flydsl_exe_cache:
        _flydsl_exe_cache[cache_key] = compile_fn(
            M=m,
            N=n,
            K=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype=in_dtype,
            lds_stage=lds_stage,
            use_cshuffle_epilog=bool(use_cshuffle_epilog),
            use_async_copy=bool(use_async_copy),
            waves_per_eu=wpe,
        )

    exe = _flydsl_exe_cache[cache_key]
    c_fp16 = torch.empty(m, n, dtype=torch.float16, device=x.device)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    exe(c_fp16, x, weight_shuffle, x_scale, w_scale, m, n, k, stream_ptr)
    out.copy_(c_fp16.to(out.dtype))
    return out


def generate_data(
    m, n, k, seed, dtype=dtypes.bf16, q_dtype_w=dtypes.fp8, is_asm=False, device="cuda"
):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=q_dtype_w)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=q_dtype_w)
    bias_f32 = None
    if is_asm:
        weight_shuffle = shuffle_weight(weight, layout=(32, 16))
        bias = torch.zeros(1, n, dtype=dtype, device=device)
        bias_f32 = bias.to(dtypes.fp32)
    else:
        weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtype, device=device)
    return x, weight_shuffle, x_scale, w_scale, out, weight, bias_f32


def generate_data_asm(
    m, n, k, seed, dtype=dtypes.bf16, q_dtype_w=dtypes.i8, device="cuda"
):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=q_dtype_w)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=q_dtype_w)
    weight_shuffle = shuffle_weight(weight, layout=(32, 16))
    bias = torch.rand([1, n], dtype=dtype, device=device)
    bias_f32 = bias.to(dtypes.fp32)
    out = torch.empty(m, n, dtype=dtype, device=device)
    return x, weight, weight_shuffle, x_scale, w_scale, out, bias_f32


def libtype_list(string):
    values = string.split(",")
    for value in values:
        if value not in ["all", "asm", "ck", "cktile", "flydsl"]:
            raise argparse.ArgumentTypeError(f"Invalid libtype: {value}")
    return values


class GemmA8W8BpreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE}",
        "untune_file": "aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv",
    }

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--libtype",
            # nargs='+',
            # choices=['all', 'asm', 'ck', 'cktile', 'flydsl'],
            type=libtype_list,
            default=["all"],
            required=False,
            help="choose libtype to be tuned, support ['all', 'asm', 'ck', 'cktile', 'flydsl']",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="ck"):
        if libtype == "ck":
            if kernelId < 0 or kernelId > len(kernels_list_ck):
                return None
            kernelList = kernels_list_ck
        elif libtype == "cktile":
            if kernelId < 0 or kernelId > len(kernels_list_cktile):
                return None
            kernelList = kernels_list_cktile
        elif libtype == "flydsl":
            if kernelId < 0 or kernelId > len(kernels_list_flydsl):
                return None
            kernelList = kernels_list_flydsl
        else:
            return None
        return kernelList[kernelId].name

    def get_asm_kernels(self, file):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        shuffle_df = (
            df[df["bpreshuffle"] == 1]
            .reset_index()
            .sort_values(by=["tile_m", "tile_n", "splitK"])
        )
        kernel_dict = (
            shuffle_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_asm_gemm_i8_tasks(self, info_keys, useSplitK, kernel_id_start, seed=0):
        task = []
        (cu_num, M, N, K, q_dtype_w) = info_keys
        if eval(q_dtype_w) != dtypes.i8:
            return task
        asm_kernel_list_csv = f"{get_asm_dir()}/i8gemm/i8gemm_bf16_perTokenI8.csv"
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
        asm_tiles = [key for key in asm_kernels.keys()]

        gemm_asm_data_idx = [0, 1, 2, 3, 4, 6]  # input index in generate_data
        torch_data_idx = [0, 5, 2, 3, 6]
        asm_kernels_id = kernel_id_start
        for key in asm_tiles:
            tile_m, tile_n, splitk = key
            maxsplitK = 8 if useSplitK else 1
            kernelName = asm_kernels.get((tile_m, tile_n, splitk), [])
            if len(kernelName) == 0:
                print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                continue
            if splitk == 0:
                maxsplitK = 1
            for splitK in range(1, maxsplitK + 1):
                kernel_name = kernelName[0]
                info = (info_keys, asm_kernels_id, splitK, kernel_name, "asm")
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w), True),
                        run_gemm_a8w8_asm,
                        (
                            gemm_asm_data_idx,
                            kernel_name,
                            dtypes.bf16,
                            True,
                            splitK,
                        ),
                        {
                            "num_warmup": 10,
                            "num_iters": 101,
                        },
                        run_torch,
                        (
                            torch_data_idx,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
            asm_kernels_id = asm_kernels_id + 1
        return task

    def get_flydsl_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        seed,
    ):
        (cu_num, M, N, K, q_dtype_w) = info_keys
        q_dtype_eval = eval(q_dtype_w)
        if q_dtype_eval == dtypes.fp8:
            in_dtype = "fp8"
        elif q_dtype_eval == dtypes.i8:
            in_dtype = "int8"
        else:
            print(f"[FlyDSL] unsupported q_dtype_w {q_dtype_w}, skipping")
            return []

        if not kernels_list_flydsl:
            return []

        gemm_flydsl_data_idx = [0, 1, 2, 3, 4]
        ref_data_idx = [0, 5, 2, 3, 6]
        tasks = []
        for i in sorted(kernels_list_flydsl.keys()):
            ki = kernels_list_flydsl[i]
            if N % ki.tile_n != 0 or K % ki.tile_k != 0:
                continue
            kernel_name = ki.name
            info = (info_keys, i, 0, kernel_name, "flydsl")
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed, dtypes.bf16, q_dtype_eval),
                    run_gemm_flydsl,
                    (
                        gemm_flydsl_data_idx,
                        ki.tile_m,
                        ki.tile_n,
                        ki.tile_k,
                        in_dtype,
                        ki.lds_stage,
                        ki.use_cshuffle_epilog,
                        ki.use_async_copy,
                        ki.waves_per_eu,
                    ),
                    {
                        "num_warmup": args.warmup,
                        "num_iters": args.iters,
                    },
                    run_torch,
                    (
                        ref_data_idx,
                        dtypes.bf16,
                    ),
                    {},
                    None,
                    1e-2,
                    0.01,
                )
            )
        return tasks

    def get_cktile_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        (cu_num, M, N, K, q_dtype_w) = info_keys
        if eval(q_dtype_w) != dtypes.fp8:
            print(
                f"Warning: q_dtype_w only support {dtypes.fp8}, actual q_dtype_w is {q_dtype_w}!"
            )
            return []
        kernels_num = len(kernels_list_cktile)
        gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 5, 2, 3, 6]
        tasks_ck = []
        for i in range(kernels_num):
            kernel = kernels_list_cktile[i]
            maxsplitK = (
                aiter.compute_gemm_SplitK(
                    M,
                    N,
                    K,
                    kernel.MPerBLOCK,
                    kernel.NPerBLOCK,
                    kernel.KPerBLOCK,
                )
                if useSplitK
                else 0
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "cktile")
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8_bpreshuffle_cktile,
                        (
                            gemm_a8w8_idx,
                            i,
                            splitK,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch,
                        (
                            ref_data_idx,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
        return tasks_ck

    def get_ck_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        (cu_num, M, N, K, q_dtype_w) = info_keys
        if eval(q_dtype_w) != dtypes.fp8:
            print(
                f"Warning: q_dtype_w only support {dtypes.fp8}, actual q_dtype_w is {q_dtype_w}!"
            )
            return []
        kernels_num = len(kernels_list_ck)
        gemm_a8w8_idx = [0, 1, 2, 3, 4]  # input index in generate_data
        ref_data_idx = [0, 5, 2, 3, 6]
        tasks_ck = []
        for i in range(kernels_num):
            kernel = kernels_list_ck[i]
            maxsplitK = (
                aiter.compute_gemm_SplitK(
                    M,
                    N,
                    K,
                    kernel.MPerBLOCK,
                    kernel.NPerBLOCK,
                    kernel.KPerBLOCK,
                )
                if useSplitK
                else 0
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "ck")
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8_bpreshuffle,
                        (
                            gemm_a8w8_idx,
                            i,
                            splitK,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch,
                        (
                            ref_data_idx,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                    )
                )
        return tasks_ck

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        issorted = args.sort
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 10000
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            q_dtype_w = untunedf.loc[i, "q_dtype_w"]
            seed = seed + 1
            total_kernel_nums = 0
            # kernels_num = len(kernels_list_ck)
            info_keys = (cu_num, M, N, K, q_dtype_w)
            if "all" in args.libtype or "ck" in args.libtype:
                task.extend(
                    self.get_ck_gemm_a8w8_bpreshuffle_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                    )
                )
            if "all" in args.libtype or "cktile" in args.libtype:
                task.extend(
                    self.get_cktile_gemm_a8w8_bpreshuffle_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                    )
                )
            if "all" in args.libtype or "asm" in args.libtype:
                task.extend(self.get_asm_gemm_i8_tasks(info_keys, useSplitK, 0, seed))
            if "all" in args.libtype or "flydsl" in args.libtype:
                task.extend(
                    self.get_flydsl_gemm_a8w8_bpreshuffle_tune_task(
                        info_keys,
                        seed,
                    )
                )

            total_kernel_nums = len(task)

            tasks_data.append((total_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )

        return ret

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME
                else (
                    self.getKernelName(kernelId, libtype)
                    if kernelName == ""
                    else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))

            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} {kernelName} {splitK=}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bw],
                }
            )
            temp = pd.DataFrame(key_dict)
            if resultdf.empty:
                resultdf = temp
            else:
                resultdf = pd.concat([resultdf, temp], ignore_index=True)
        return resultdf


if __name__ == "__main__":
    ## use default key and resultList
    key = ["cu_num", "M", "N", "K", "q_dtype_w"]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = GemmA8W8BpreShuffleTuner(
        "GemmA8W8BpreShuffleTuner",
        key=key,
        resultList=resultList,
        description="gen API for gemm a8w8 bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
