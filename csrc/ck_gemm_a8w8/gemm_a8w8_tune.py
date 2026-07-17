# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes, logger
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8
from aiter.utility.base_tuner import GemmCommonTuner
from gemm_a8w8_common import kernels_list
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.opus.gemm_op_a16w16 import _opus_gemm_bf16_dispatch as _opus_gemm_dispatch


# Descriptive labels for Opus a8w8 no-scale instances used by this tuner backend.
OPUS_A8W8_KERNEL_NAMES = {
    2: "opus_gemm_512x256x256x128_2x4_16x16x128_0x0x0<fp32_t>",
    10: "opus_gemm_512x256x256x128_2x4_16x16x128_0x0x0<bf16_t>",
}


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
    m,
    n,
    k,
    seed,
    dtype=dtypes.bf16,
    q_dtype_w=dtypes.fp8,
    backend="ck",
    device="cuda",
):
    torch.manual_seed(seed)

    if backend == "opus":
        # Opus a8w8 no-scale path consumes raw fp8 tensors (no x_scale/w_scale).
        x = torch.randn((m, k), dtype=dtypes.fp16, device=device).to(dtypes.fp8)
        weight = torch.randn((n, k), dtype=dtypes.fp16, device=device).to(dtypes.fp8)
        x_scale = None
        w_scale = None
    else:
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


def opus_a8w8_noscale_ref(x, weight, dtype=dtypes.bf16):
    return torch.matmul(x.float(), weight.float().t()).to(dtype)


def run_gemm_a8w8(x, weight, x_scale, w_scale, out, kernelId, splitK):

    aiter.gemm_a8w8_tune(x, weight, x_scale, w_scale, out, kernelId, splitK)
    return out


def run_opus_a8w8_noscale(x, weight, x_scale, w_scale, out, kernelId, splitK):
    # No-scale path: x_scale / w_scale / splitK / kernelId are not consumed by
    # Opus dispatch, but are kept for mp_tuner run-arg compatibility.
    _opus_gemm_dispatch(x, weight, out, None, None, None, None)
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

    def getKernelName(self, kernelId):
        if getattr(self, "backend", "ck") == "opus":
            return OPUS_A8W8_KERNEL_NAMES.get(
                kernelId, f"opus_a8w8_noscale_kid{kernelId}"
            )
        if kernelId >= len(kernels_list) or kernelId < 0:
            return None
        return kernels_list[kernelId].name

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_GEMM_config_with_quant_type.cache_clear()
        _op._GEMM_QUANT_TYPE_CACHE.clear()
        _op._GEMM_QUANT_TYPE_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--backend",
            choices=["ck", "opus"],
            default="ck",
            help="Backend to tune: 'ck' (existing path) or 'opus' (a8w8 no-scale).",
        )

    def pre_process(self, args):
        self.backend = args.backend
        if self.backend == "opus" and args.tune_file == AITER_CONFIG_GEMM_A8W8:
            args.tune_file = "/tmp/opus_a8w8_noscale_tuned_gemm.csv"
            logger.warning(
                "--backend opus selected; using /tmp/opus_a8w8_noscale_tuned_gemm.csv "
                "to avoid mixing with CK tuned file."
            )
        # Restrict module_deepgemm_opus codegen to a8w8-family instances only
        # (no a16w16 families) for this tuner backend.
        if self.backend == "opus":
            os.environ["OPUS_A8W8_ONLY"] = "1"
        else:
            os.environ.pop("OPUS_A8W8_ONLY", None)
        super().pre_process(args)

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=(1, 1, 2))

    def run_config(self, args):
        from aiter.test_common import run_perftest, checkAllclose
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8

        backend = getattr(args, "backend", "ck")
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
                q_dtype_w_eval = eval(q_dtype_w)
                if backend == "opus" and q_dtype_w_eval != dtypes.fp8:
                    results.append(
                        {
                            "shape": shape_str,
                            "e2e_us": -1,
                            "status": "skip:opus backend supports fp8 only",
                        }
                    )
                    continue

                gd = generate_data(
                    M,
                    N,
                    K,
                    0,
                    dtypes.bf16,
                    q_dtype_w_eval,
                    backend=backend,
                )
                x, weight, x_scale, w_scale, out = (
                    gd["x"],
                    gd["weight"],
                    gd["x_scale"],
                    gd["w_scale"],
                    gd["out"],
                )
                if backend == "opus":
                    out, us = run_perftest(
                        run_opus_a8w8_noscale,
                        x,
                        weight,
                        x_scale,
                        w_scale,
                        out,
                        10,
                        0,
                        num_warmup=args.warmup,
                        num_iters=args.iters,
                    )
                    ref = opus_a8w8_noscale_ref(x, weight, dtype=dtypes.bf16)
                else:
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
                        q_dtype_w=q_dtype_w_eval,
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
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()

        task = []
        tasks_data = []
        gemm_keys = ["x", "weight", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        seed = 0
        backend = getattr(args, "backend", "ck")

        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            q_dtype_w = untunedf.loc[i, "q_dtype_w"]
            q_dtype_w_eval = eval(q_dtype_w)

            total_kernel_nums = 0
            info_keys = (gfx, cu_num, M, N, K, q_dtype_w)

            if backend == "opus":
                if q_dtype_w_eval != dtypes.fp8:
                    tasks_data.append((0, ()))
                    continue

                kid = 10  # a8w8_noscale bf16-output variant (opus_gemm_common.py)
                splitK = 0
                info = (info_keys, kid, splitK, "")
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, q_dtype_w_eval, "opus"),
                        run_opus_a8w8_noscale,
                        (gemm_keys, kid, splitK),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        opus_a8w8_noscale_ref,
                        (["x", "weight"], dtypes.bf16),
                        {},
                        None,
                        1e-2,
                        1e-2,
                        None,
                        None,
                        ("out",),
                    )
                )
                total_kernel_nums = 1
            else:
                kernels_num = len(kernels_list)
                for j in range(kernels_num):
                    kernel = kernels_list[j]
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
                        info = (info_keys, j, splitK, "")
                        task.append(
                            (
                                info,
                                generate_data,
                                (M, N, K, seed, dtypes.bf16, q_dtype_w_eval),
                                run_gemm_a8w8,
                                (gemm_keys, j, splitK),
                                {
                                    "num_warmup": args.warmup,
                                    "num_iters": args.iters,
                                },
                                gemm_a8w8_ref,
                                (ref_keys, dtypes.bf16, q_dtype_w_eval),
                                {},
                                None,
                                1e-2,
                                1e-2,
                                None,
                                None,
                                ("out",),
                            )
                        )
                        total_kernel_nums = total_kernel_nums + 1

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


if __name__ == "__main__":

    ## use default key and resultList with q_dtype_w support
    key = ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
    resultList = [
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
