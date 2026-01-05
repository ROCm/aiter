# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os

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
        if percent > 0.01:
            return False
        else:
            return True


def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
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
    m, k = x.shape
    n, k = weight.shape
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
    m, k = x.shape
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


def generate_data(m, n, k, seed, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(seed)
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtype, device=device)  #
    w = torch.randn((n, k), dtype=dtype, device=device)  #
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    w_shuffle = shuffle_weight(w)
    out_ck = torch.empty((m + 255) // 256 * 256, n, dtype=dtype, device=device)
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    bias_f32 = None
    return (
        x,
        w,
        x_scales,
        w_scales,
        w_shuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        out_ck,
        bias_f32,
    )


class GemmA4W4BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A4W4}",
        "untune_file": "aiter/configs/a4w4_blockscale_untuned_gemm.csv",
    }

    def _setup_specific_arguments(self):
        pass

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
        if kernelId < 0 or kernelId > len(kernels_list):
            return None
        return kernels_list[kernelId].name

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
        from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() not in ["gfx950"]:
            print(f"tuning is not supported in this chip {get_gfx()}")
            return []
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        task = []
        tasks_in_data = []

        ck_kernels_num = len(kernels_list)
        gemm_a4w4_data_idx = [0, 4, 5, 6, 7]  # index in generated data
        gemm_asm_data_idx = [0, 4, 5, 6, 7, 8]
        torch_data_idx = [0, 1, 2, 3]
        seed = 1000
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]

            total_kernel_nums = 0
            seed = seed + 1

            for i in range(ck_kernels_num):
                kernel = kernels_list[i]
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
                    info = ((cu_num, M, N, K), i, splitK, "")
                    task.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a4w4_blockscale,
                            (
                                gemm_a4w4_data_idx,
                                i,
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
                    total_kernel_nums = total_kernel_nums + 1
            ### asm kernels
            asm_kernels_id = ck_kernels_num + 1
            asm_kernel_list_csv = f"{get_asm_dir()}/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
            asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
            asm_tiles = [key for key in asm_kernels.keys()]
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
                    info = ((cu_num, M, N, K), asm_kernels_id, splitK, kernel_name)
                    task.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a4w4_blockscale_asm,
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

                    total_kernel_nums = total_kernel_nums + 1
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
