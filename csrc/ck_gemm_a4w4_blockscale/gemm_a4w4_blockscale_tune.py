# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
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


def run_gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out, kernel_id, splitK):
    m, _k = x.shape
    _n, _k = weight.shape
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
):
    m, _k = x.shape
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


def run_gemm_a4w4_blockscale_flydsl(
    x,
    weight_shuffle,
    x_scale,
    w_scale,
    out,
    workspace,
    tile_n,
    k_waves,
    num_buffers,
    split_k,
):
    from aiter.ops.flydsl.decode_gemm_mxfp4 import flydsl_decode_gemm_mxfp4

    m = x.shape[0]
    return flydsl_decode_gemm_mxfp4(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out[:m],
        tile_n=tile_n,
        k_waves=k_waves,
        num_buffers=num_buffers,
        split_k=split_k,
        workspace=workspace,
    )


def libtype_list(string):
    values = string.split(",")
    for value in values:
        if value not in ["all", "asm", "ck", "flydsl"]:
            raise argparse.ArgumentTypeError(f"Invalid libtype: {value}")
    return values


def generate_data(m, n, k, seed, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(seed)
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtype, device=device)
    w = torch.randn((n, k), dtype=dtype, device=device)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    w_shuffle = shuffle_weight(w)
    out_ck = torch.empty((m + 255) // 256 * 256, n, dtype=dtype, device=device)
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
        "bias_f32": bias_f32,
    }


def generate_flydsl_data(m, n, k, seed, device="cuda"):
    data = generate_data(m, n, k, seed, device=device)
    # Keep split-K allocation out of the timed runner and rotate scratch with operands.
    for split_k in (1, 2, 4, 8):
        data[f"workspace_{split_k}"] = (
            None
            if split_k == 1
            else torch.empty((split_k, m, n), dtype=torch.float32, device=device)
        )
    return data


class GemmA4W4BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A4W4}",
        "untune_file": "aiter/configs/a4w4_blockscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A4W4",
    }

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config

        get_GEMM_config.cache_clear()
        if hasattr(get_GEMM_config, "gemm_dict"):
            del get_GEMM_config.gemm_dict

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--libtype",
            type=libtype_list,
            default=["all"],
            help="choose libtype to be tuned, support ['all', 'asm', 'ck', 'flydsl']",
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
            shape_str = f"({M}, {N}, {K})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(M, N, K, 0)
                x, w = gd["x"], gd["w"]
                x_scales, w_scales = gd["x_scales"], gd["w_scales"]
                w_shuffle = gd["w_shuffle"]
                x_scales_shuffle = gd["x_scales_shuffle"]
                w_scales_shuffle = gd["w_scales_shuffle"]
                out, us = run_perftest(
                    gemm_a4w4,
                    x,
                    w_shuffle,
                    x_scales_shuffle,
                    w_scales_shuffle,
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

    def get_asm_kernels(self, file):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        shuffle_df = (
            df[df["bpreshuffle"] == 1]
            .reset_index()
            .sort_values(by=["tile_M", "tile_N", "splitK"])
        )
        kernel_dict = (
            shuffle_df.groupby(["tile_M", "tile_N", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def getKernelName(self, kernelId):
        # kernels_list is a dict keyed by kernel index; do not use len() bounds only.
        if kernelId is None or kernelId < 0 or kernelId not in kernels_list:
            return None
        return kernels_list[kernelId].name

    def get_flydsl_decode_tasks(self, info_keys, seed, args):
        from aiter.jit.utils.chip_info import get_gfx_runtime

        _gfx, _cu_num, M, N, K = info_keys
        if get_gfx_runtime() != "gfx950" or not 1 <= M <= 16 or K % 64:
            return []
        from aiter.ops.flydsl.decode_gemm_mxfp4 import (
            DECODE_GEMM_MXFP4_LDS_LIMIT,
            decode_gemm_mxfp4_lds_bytes,
            flydsl_decode_name,
        )

        gemm_keys = ["x", "w_shuffle", "x_scales_shuffle", "w_scales_shuffle", "out_ck"]
        ref_keys = ["x", "w", "x_scales", "w_scales"]
        tasks = []
        for tile_n, k_waves in ((32, 2), (64, 1)):
            if N % tile_n:
                continue
            for num_buffers in (2, 4, 6, 8, 12):
                # Like CK/asm, partitions beyond 1 are opt-in via -k/--splitK.
                for split_k in (1, 2, 4, 8) if args.splitK else (1,):
                    try:
                        lds_bytes = decode_gemm_mxfp4_lds_bytes(
                            M, K, tile_n, k_waves, split_k
                        )
                    except ValueError:
                        continue
                    if lds_bytes > DECODE_GEMM_MXFP4_LDS_LIMIT:
                        continue
                    name = flydsl_decode_name(tile_n, k_waves, num_buffers, split_k)
                    tasks.append(
                        (
                            # splitK column is log2(partitions), as for CK/asm:
                            # 0 = no split. The kernel name carries the count.
                            (info_keys, -1, split_k.bit_length() - 1, name),
                            generate_flydsl_data,
                            (M, N, K, seed),
                            run_gemm_a4w4_blockscale_flydsl,
                            (
                                gemm_keys + [f"workspace_{split_k}"],
                                tile_n,
                                k_waves,
                                num_buffers,
                                split_k,
                            ),
                            {
                                "num_warmup": args.warmup,
                                "num_iters": args.iters,
                            },
                            run_torch,
                            (ref_keys, dtypes.bf16),
                            {},
                            None,
                            1e-2,
                            0.01,
                            None,
                            None,
                            ("out_ck",),
                        )
                    )
        return tasks

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
        from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

        if get_gfx() not in ["gfx950"]:
            print(f"tuning is not supported in this chip {get_gfx()}")
            return []
        gfx = get_gfx()
        cu_num = self.get_cu_num()
        task = []
        tasks_in_data = []

        ck_kernels_num = len(kernels_list)
        gemm_ck_keys = [
            "x",
            "w_shuffle",
            "x_scales_shuffle",
            "w_scales_shuffle",
            "out_ck",
        ]
        gemm_asm_keys = [
            "x",
            "w_shuffle",
            "x_scales_shuffle",
            "w_scales_shuffle",
            "out_ck",
            "bias_f32",
        ]
        ref_keys = ["x", "w", "x_scales", "w_scales"]
        seed = 0
        for shape_idx in range(len(untunedf)):
            row = untunedf.iloc[shape_idx]
            # Native int keys so post_process grouping matches single-shape runs (no np.int64 vs int split).
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])

            total_kernel_nums = 0

            if "all" in args.libtype or "ck" in args.libtype:
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
                        info = ((gfx, cu_num, M, N, K), kernel_idx, splitK, "")
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
            if "all" in args.libtype or "asm" in args.libtype:
                ### asm kernels
                asm_kernels_id = ck_kernels_num + 1
                asm_kernel_list_csv = (
                    f"{get_asm_dir()}/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
                )
                asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
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
                            (gfx, cu_num, M, N, K),
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
                                    True,
                                    splitK,
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
            if "all" in args.libtype or "flydsl" in args.libtype:
                flydsl_tasks = self.get_flydsl_decode_tasks(
                    (gfx, cu_num, M, N, K), seed, args
                )
                task.extend(flydsl_tasks)
                total_kernel_nums += len(flydsl_tasks)
            # A shape with no task (e.g. FlyDSL-only on an unsupported shape)
            # makes no group; mp_tuner asserts group count == bookkeeping count.
            if total_kernel_nums:
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
