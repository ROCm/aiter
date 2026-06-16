# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8
from aiter.utility.base_tuner import GemmCommonTuner
from gemm_a8w8_common import (
    kernels_list,
    kernels_list_cktile,
    BLOCK_PER_CU_MAX,
)
from aiter.utility.mp_tuner import mp_tuner


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


def run_torch(
    x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16, quant_dtype=dtypes.i8
):
    if quant_dtype == dtypes.i8:
        x = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
        scale = torch.matmul(x_scale, w_scale)
        out = torch.mul(x, scale)
    else:
        x = x.to(dtypes.fp32) * x_scale
        weight = weight.to(dtypes.fp32) * w_scale
        out = F.linear(x, weight)

    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(
        untuned_gemm_file
    ), f"Not exist a8w8_untuned_gemm.csv file: {untuned_gemm_file}"
    untunedf = pd.read_csv(untuned_gemm_file)
    filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
    return filtered_df


def get_tuned_gemm_list(tuned_gemm_file):
    if os.path.exists(tuned_gemm_file):
        tunedf = pd.read_csv(tuned_gemm_file)
    else:
        tunedf = pd.DataFrame(
            columns=[
                "gfx",
                "cu_num",
                "M",
                "N",
                "K",
                "kernelId",
                "splitK",
                "us",
                "kernelName",
            ]
        )
    return tunedf


def generate_data(
    m, n, k, seed, dtype=dtypes.bf16, q_dtype_w=dtypes.fp8, device="cuda"
):
    torch.manual_seed(seed)

    if q_dtype_w == dtypes.i8:
        x = torch.randint(-20, 20, (m, k), dtype=dtypes.i8, device=device)
        weight = torch.randint(-20, 20, (n, k), dtype=dtypes.i8, device=device)
        x_scale = torch.rand([m, 1], dtype=dtypes.bf16, device=device)
        w_scale = torch.rand([1, n], dtype=dtypes.bf16, device=device)
    else:
        x_fp = torch.randn((m, k), dtype=dtype, device=device)
        weight_fp = torch.randn((n, k), dtype=dtype, device=device)
        x, x_scale = aiter.pertoken_quant(x_fp, quant_dtype=q_dtype_w)
        weight, w_scale = aiter.pertoken_quant(weight_fp, quant_dtype=q_dtype_w)

    out = torch.empty(m, n, dtype=dtype, device=device)
    return {
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "out": out,
    }


def gemm_a8w8_ref(x, weight, x_scale, w_scale, dtype=dtypes.bf16, q_dtype_w=dtypes.fp8):
    return run_torch(x, weight, x_scale, w_scale, dtype=dtype, quant_dtype=q_dtype_w)


def run_gemm_a8w8(x, weight, x_scale, w_scale, out, kernelId, splitK):
    aiter.gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernelId, splitK)
    return out


def run_gemm_a8w8_cktile(x, weight, x_scale, w_scale, out, kernelId, splitK):
    aiter.gemm_a8w8_cktile_tune(x, weight, x_scale, w_scale, out, kernelId, splitK)
    return out


class GemmA8W8Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8}",
        "untune_file": "aiter/configs/a8w8_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8",
    }

    def getKernelName(self, kernelId, libType):  # ="ck"
        """
        Get the kernel name based on the kernel ID for different types.
        """
        kernels = []
        if libType == "ck":
            kernels = kernels_list
        elif libType == "cktile":
            kernels = kernels_list_cktile
        else:
            return None

        if kernelId >= len(kernels) or kernelId < 0:
            return None
        return kernels[kernelId].name

    def get_gemm_a8w8_ck_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        run_kwargs,
    ):
        gfx, cu_num, M, N, K, q_dtype_w = info_keys

        gemm_keys = ["x", "weight", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_ck = []

        for j, kernel in enumerate(kernels_list):
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
                info = (info_keys, j, splitK, "", "ck")
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8,
                        (gemm_keys, j, splitK),
                        run_kwargs,
                        gemm_a8w8_ref,
                        (ref_keys, dtypes.bf16, eval(q_dtype_w)),
                        {},
                        None,
                        1e-2,
                        1e-2,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks_ck

    def get_gemm_a8w8_cktile_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        block_per_cu,
        run_kwargs,
    ):
        gfx, cu_num, M, N, K, q_dtype_w = info_keys
        kernel_list = {
            k: v for k, v in kernels_list_cktile.items() if v.BlockPerCu in block_per_cu
        }

        gemm_keys = ["x", "weight", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_cktile = []

        for j, kernel in enumerate(kernel_list):
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
                info = (info_keys, j, splitK, "", "cktile")
                tasks_cktile.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8_cktile,
                        (gemm_keys, j, splitK),
                        run_kwargs,
                        gemm_a8w8_ref,
                        (ref_keys, dtypes.bf16, eval(q_dtype_w)),
                        {},
                        None,
                        1e-2,
                        1e-2,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks_cktile

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_GEMM_config_with_quant_type.cache_clear()
        _op._GEMM_QUANT_TYPE_CACHE.clear()
        _op._GEMM_QUANT_TYPE_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        """
        Setup specific arguments for the tuner.
        """

        self.parser.add_argument(
            "--libtype",
            type=str,
            default="all",
            choices=["ck", "cktile", "all", "both"],
            required=False,
            help="CK gemm a8w8 type to tune: ck, cktile, both or all (covers all supported backends across standard/preshuffleB modes)",
        )

        self.parser.add_argument(
            "--blockPerCu",
            nargs="+",
            type=int,
            default=list(range(1, BLOCK_PER_CU_MAX + 1)),
            help="List of BlockPerCu values to tune (CKTile only)",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=(1, 1, 2))

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            q_dtype_w = row["q_dtype_w"]
            shape_str = f"({M}, {N}, {K}, {q_dtype_w})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(M, N, K, 0, dtypes.bf16, eval(q_dtype_w))
                x, weight, x_scale, w_scale, out = (
                    gd["x"],
                    gd["weight"],
                    gd["x_scale"],
                    gd["w_scale"],
                    gd["out"],
                )
                out, us = run_perftest(
                    gemm_a8w8,
                    x,
                    weight,
                    x_scale,
                    w_scale,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = gemm_a8w8_ref(
                    x,
                    weight,
                    x_scale,
                    w_scale,
                    dtype=dtypes.bf16,
                    q_dtype_w=eval(q_dtype_w),
                )
                err_ratio = checkAllclose(
                    out.to(dtypes.bf16), ref, msg=f"run_config {shape_str}"
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
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        block_per_cu = args.blockPerCu
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()

        task = []
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        tasks_data = []
        seed = 0

        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            q_dtype_w = untunedf.loc[i, "q_dtype_w"]
            info_keys = (gfx, cu_num, M, N, K, q_dtype_w)
            prev_task_count = len(task)

            lib = args.libtype
            if lib in ("ck", "both", "all"):
                task.extend(
                    self.get_gemm_a8w8_ck_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        run_kwargs,
                    )
                )
            if lib in ("cktile", "both", "all"):
                task.extend(
                    self.get_gemm_a8w8_cktile_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        block_per_cu,
                        run_kwargs,
                    )
                )

            shape_kernel_nums = len(task) - prev_task_count
            tasks_data.append((shape_kernel_nums, ()))
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
        """
        post-process the tuning results into a DataFrame.
        """

        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            print(el)
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME or time == self.INF_TIME
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
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
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

    ## use default key and resultList with q_dtype_w support
    key = ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
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
    tuner = GemmA8W8Tuner(
        "GemmA8W8Tuner",
        key=key,
        resultList=resultList,
        description="gen API for CK gemm a8w8 kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
