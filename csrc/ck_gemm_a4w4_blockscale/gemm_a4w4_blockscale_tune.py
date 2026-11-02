# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gemm_a4w4 on gfx950 and write the a4w4_blockscale table.

Candidates are the CK blockscale kernels, the asm f4gemm kernels and, with
--libtype triton/all, triton's raw or preshuffled MXFP4 kernel. Every row records the
winning backend in the ``libtype`` column (``ck``/``asm``/``triton``) so
aiter.ops.gemm_op_a4w4 can dispatch a ``triton`` row to the triton kernel.
"""

import os
from typing import Any, ClassVar

import pandas as pd
import torch
from gemm_a4w4_blockscale_common import kernels_list

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A4W4, get_asm_dir
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import perftest
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

# torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.random.manual_seed(0)
SCALE_GROUP_SIZE = 32
block_shape = (128, 128)


def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel() / a.numel()
        return not percent > 0.01


def run_torch(x, w, x_scales, w_scales, dtype):
    m, _k = x.shape
    n, _k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def kernel_instance_test(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a4w4_blockscale_tune(x, weight, x_scale, w_scale, out, kernel_id, splitK)
    return out


def run_gemm_a4w4_blockscale(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, scale_preshuffle=True
):
    m, _k = x.shape
    _n, _k = weight.shape
    if not scale_preshuffle:
        from aiter.ops.gemm_op_a4w4 import _a4w4_convert_scale

        x_scale = _a4w4_convert_scale(x_scale, m, _k * 2, False, True)
        w_scale = _a4w4_convert_scale(w_scale, _n, _k * 2, False, True)
    res = aiter.gemm_a4w4_blockscale_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return res[:m]


def run_gemm_a4w4_blockscale_asm(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out,
    bias,
    kernelName,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    splitK=None,
    scale_preshuffle=True,
):
    m, _k = x.shape
    if not scale_preshuffle:
        from aiter.ops.gemm_op_a4w4 import _a4w4_convert_scale

        x_scale = _a4w4_convert_scale(x_scale, m, _k * 2, False, True)
        w_scale = _a4w4_convert_scale(
            w_scale, weight_shuffle.shape[0], _k * 2, False, True
        )
    # if splitK is not None and splitK > 0:
    #    out_reset = torch.zeros(
    #        out.shape[0], out.shape[1], dtype=dtype, device=torch.cuda.current_device()
    #    )
    #    out = out_reset
    res = aiter.gemm_a4w4_asm(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=splitK,
    )
    return res[:m]


def run_gemm_a4w4_triton(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out=None,
    bpreshuffle=True,
    scale_preshuffle=True,
):
    # Identical to what gemm_a4w4 runs for a "triton" row: same input
    # operands and the same triton config selection (config=None).
    from aiter.ops.gemm_op_a4w4 import gemm_a4w4_triton

    return gemm_a4w4_triton(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out=out,
        bpreshuffle=bpreshuffle,
        scale_preshuffle=scale_preshuffle,
    )


def generate_data(m, n, k, seed, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(seed)
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtype, device=device)
    w = torch.randn((n, k), dtype=dtype, device=device)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    w_shuffle = shuffle_weight(w) if n % 16 == 0 and k % 64 == 0 else None
    out_ck = torch.empty((m + 255) // 256 * 256, n, dtype=dtype, device=device)
    # triton's kernel writes the exact M rows (no CK/asm padding).
    out_triton = torch.empty(m, n, dtype=dtype, device=device)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    bias_f32 = None
    return {
        "x": x,
        "w": w,
        "x_scales": x_scales,
        "w_scales": w_scales,
        "w_shuffle": w_shuffle,
        "x_scales_shuffle": x_scales_shuffle,
        "w_scales_shuffle": w_scales_shuffle,
        "out_ck": out_ck,
        "out_triton": out_triton,
        "bias_f32": bias_f32,
    }


class GemmA4W4BlockScaleTuner(GemmCommonTuner):
    # libtype names the backend of a row: gemm_a4w4 sends "triton" rows to the
    # matching Triton weight layout and keeps CK/ASM rows on kernelName.
    RESULT_LIST: ClassVar[list[str]] = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A4W4}",
        "untune_file": "aiter/configs/a4w4_blockscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A4W4",
    }

    def __init__(self, name, key=None, resultList=None, description=None):
        super().__init__(
            name,
            key=(
                ["gfx", "cu_num", "M", "N", "K", "bpreshuffle", "scale_preshuffle"]
                if key is None
                else key
            ),
            resultList=self.RESULT_LIST if resultList is None else resultList,
            description=description,
        )

    def _normalize_tuned_identity(self, tunedf):
        from aiter.ops.gemm_op_a4w4 import _a4w4_layout_flag

        tunedf = super()._normalize_tuned_identity(tunedf)
        for column in ("bpreshuffle", "scale_preshuffle"):
            if column not in tunedf:
                tunedf[column] = 1
            tunedf[column] = tunedf[column].map(_a4w4_layout_flag)
        return tunedf

    def get_untuned_gemm_list(self, untuned_gemm_file):
        from aiter.ops.gemm_op_a4w4 import _a4w4_layout_flag

        frame = super().get_untuned_gemm_list(untuned_gemm_file)
        for column in ("bpreshuffle", "scale_preshuffle"):
            option = getattr(self, "input_layouts", {}).get(column)
            if option == "both":
                frame = pd.concat(
                    [frame.assign(**{column: flag}) for flag in (0, 1)],
                    ignore_index=True,
                )
            elif option is not None:
                frame[column] = int(option)
            elif column not in frame:
                frame[column] = 1
            frame[column] = frame[column].map(_a4w4_layout_flag)
        return frame.drop_duplicates().reset_index(drop=True)

    def pre_process(self, args):
        self.input_layouts = {
            c: getattr(args, c, None) for c in ("bpreshuffle", "scale_preshuffle")
        }
        super().pre_process(args)

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config

        get_GEMM_config.cache_clear()
        if hasattr(get_GEMM_config, "gemm_dict"):
            del get_GEMM_config.gemm_dict

    def _setup_specific_arguments(self):
        for column in ("bpreshuffle", "scale_preshuffle"):
            self.parser.add_argument(
                "--" + column,
                choices=["0", "1", "both"],
                default=None,
                help=f"input {column}: 0=raw, 1=shuffled, both=compare each layout separately; default comes from CSV or 1 for legacy inputs",
            )
        self.parser.add_argument(
            "--libtype",
            type=str,
            default="all",
            choices=["ck", "asm", "triton", "all"],
            required=False,
            help="backends to time per shape: ck, asm, triton or all",
        )

    def run_config(self, args):
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4
        from aiter.test_common import checkAllclose, run_perftest

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            bpreshuffle = bool(int(row.get("bpreshuffle", 1)))
            scale_preshuffle = bool(int(row.get("scale_preshuffle", 1)))
            shape_str = (
                f"({M}, {N}, {K}), B={int(bpreshuffle)}, S={int(scale_preshuffle)}"
            )
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(M, N, K, 0)
                x, w = gd["x"], gd["w"]
                x_scales, w_scales = gd["x_scales"], gd["w_scales"]
                w_shuffle = gd["w_shuffle"] if bpreshuffle else w
                x_scales_shuffle = (
                    gd["x_scales_shuffle"] if scale_preshuffle else x_scales
                )
                w_scales_shuffle = (
                    gd["w_scales_shuffle"] if scale_preshuffle else w_scales
                )
                out, us = run_perftest(
                    gemm_a4w4,
                    x,
                    w_shuffle,
                    x_scales_shuffle,
                    w_scales_shuffle,
                    bpreshuffle=bpreshuffle,
                    scale_preshuffle=scale_preshuffle,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = run_torch(x, w, x_scales, w_scales, dtypes.bf16)
                err_ratio = checkAllclose(
                    out[:M].to(dtypes.bf16), ref, msg=f"run_config {shape_str}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def calculate(self, results, bpes=(1 / 2, 1 / 2, 2)):
        return super().calculate(results, bpes=bpes)

    def get_asm_kernels(self, file, bpreshuffle=True):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        shuffle_df = (
            df[df["bpreshuffle"] == int(bpreshuffle)]
            .reset_index()
            .sort_values(by=["tile_M", "tile_N", "splitK"])
        )
        kernel_dict = (
            shuffle_df.groupby(["tile_M", "tile_N", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def getKernelName(self, kernelId, libtype="ck"):
        if libtype == "triton":
            return "triton"
        # kernels_list is a dict keyed by kernel index; do not use len() bounds only.
        if kernelId is None or kernelId < 0 or kernelId not in kernels_list:
            return None
        return kernels_list[kernelId].name

    @staticmethod
    def _libtype_of(kernelName):
        name = str(kernelName)
        if name == "triton":
            return "triton"
        if name.startswith("_ZN"):
            return "asm"
        return "ck"

    def result_to_df(self, results):
        resultdf = super().result_to_df(results)
        if not resultdf.empty:
            resultdf["libtype"] = [
                self._libtype_of(kernelName) for kernelName in resultdf["kernelName"]
            ]
        return resultdf.reindex(columns=self.columns)

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
        lib = getattr(args, "libtype", "all")
        from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
        from aiter.ops.gemm_op_a4w4 import (
            a4w4_triton_shape_supported,
        )

        if get_gfx() not in ["gfx950"]:
            print(f"tuning is not supported in this chip {get_gfx()}")
            return []
        gfx = self.get_gfx()
        cu_num = self.get_cu_num()
        task = []
        tasks_in_data = []

        ck_kernels_num = len(kernels_list)
        ref_keys = ["x", "w", "x_scales", "w_scales"]
        seed = 0
        for shape_idx in range(len(untunedf)):
            row = untunedf.iloc[shape_idx]
            # Native int keys so post_process grouping matches single-shape runs (no np.int64 vs int split).
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            bpreshuffle = bool(int(row.get("bpreshuffle", 1)))
            scale_preshuffle = bool(int(row.get("scale_preshuffle", 1)))
            shape_key = (gfx, cu_num, M, N, K, int(bpreshuffle), int(scale_preshuffle))
            operand_keys = [
                "x",
                "w_shuffle" if bpreshuffle else "w",
                "x_scales_shuffle" if scale_preshuffle else "x_scales",
                "w_scales_shuffle" if scale_preshuffle else "w_scales",
            ]
            gemm_ck_keys = operand_keys + ["out_ck"]
            gemm_asm_keys = operand_keys + ["out_ck", "bias_f32"]
            gemm_triton_keys = operand_keys + ["out_triton"]

            total_kernel_nums = 0

            if lib in ("ck", "all") and bpreshuffle:
                for kernel_idx in range(ck_kernels_num):
                    kernel = kernels_list[kernel_idx]
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
                        info = (shape_key, kernel_idx, splitK, "")
                        task.append(
                            (
                                info,
                                generate_data,
                                (M, N, K, seed),
                                run_gemm_a4w4_blockscale,
                                (
                                    gemm_ck_keys,
                                    kernel_idx,
                                    splitK,
                                    scale_preshuffle,
                                ),
                                {
                                    "num_warmup": 10,
                                    "num_iters": 101,
                                },
                                run_torch,
                                (
                                    ref_keys,
                                    dtypes.bf16,
                                ),
                                {},
                                None,
                                1e-2,
                                0.01,
                                None,
                                None,
                                ("out_ck",),
                            )
                        )
                        total_kernel_nums = total_kernel_nums + 1
            ### asm kernels
            if lib in ("asm", "all"):
                asm_kernels_id = ck_kernels_num + 1
                asm_kernel_list_csv = (
                    f"{get_asm_dir()}/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
                )
                asm_kernels = self.get_asm_kernels(asm_kernel_list_csv, bpreshuffle)
                asm_tiles = [key for key in asm_kernels]
                for key in asm_tiles:
                    tile_m, tile_n, splitk = key
                    maxsplitK = (
                        aiter.compute_gemm_SplitK(M, N, K, tile_m, tile_n, 256)
                        if useSplitK
                        else 0
                    )
                    kernelName = asm_kernels.get((tile_m, tile_n, splitk), [])
                    if len(kernelName) == 0:
                        print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                        continue
                    if splitk == 0:
                        maxsplitK = 0
                    for splitK in range(maxsplitK + 1):
                        kernel_name = kernelName[0]
                        info = (
                            shape_key,
                            asm_kernels_id,
                            splitK,
                            kernel_name,
                        )
                        task.append(
                            (
                                info,
                                generate_data,
                                (M, N, K, seed),
                                run_gemm_a4w4_blockscale_asm,
                                (
                                    gemm_asm_keys,
                                    kernel_name,
                                    dtypes.bf16,
                                    bpreshuffle,
                                    splitK,
                                    scale_preshuffle,
                                ),
                                {
                                    "num_warmup": 10,
                                    "num_iters": 101,
                                },
                                run_torch,
                                (
                                    ref_keys,
                                    dtypes.bf16,
                                ),
                                {},
                                None,
                                1e-2,
                                0.01,
                                None,
                                None,
                                ("out_ck",),
                            )
                        )
                        asm_kernels_id = asm_kernels_id + 1

                        total_kernel_nums = total_kernel_nums + 1
            ### triton kernel: one candidate, exactly what gemm_a4w4 runs for a
            ### libtype="triton" row (triton picks its own config for the shape).
            if lib in ("triton", "all"):
                if a4w4_triton_shape_supported(M, N, K, bpreshuffle):
                    info = (shape_key, 0, 0, "triton")
                    task.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a4w4_triton,
                            (gemm_triton_keys, bpreshuffle, scale_preshuffle),
                            {
                                "num_warmup": 10,
                                "num_iters": 101,
                            },
                            run_torch,
                            (
                                ref_keys,
                                dtypes.bf16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.01,
                            None,
                            None,
                            ("out_triton",),
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1
                else:
                    print(
                        f"triton a4w4 unsupported shape/layout: "
                        f"({M}, {N}, {K}), bpreshuffle={bpreshuffle}"
                    )
            tasks_in_data.append((total_kernel_nums, ()))

        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_in_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )
        return ret


if __name__ == "__main__":
    # key = [
    #    "cu_num",
    #    "M",
    #    "N",
    #    "K",
    # ]
    # resultList = key + [
    #    "kernelId",
    #    "splitK",
    #    "us",
    #    "kernelName",
    #    "errRatio",
    #    "tflops",
    #    "bw",
    # ]
    ## use default key and resultList
    tuner = GemmA4W4BlockScaleTuner(
        "GemmA4W4BlockScaleTuner", description="gen API for CK gemm a4w4 kernel"
    )

    args = tuner.parse_args()

    tuner.run(args, False)
