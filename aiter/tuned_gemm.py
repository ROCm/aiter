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
from pathlib import Path
import pandas as pd
import functools
import torch
import torch.nn.functional as F
from aiter import hipb_create_extension, hipb_mm, getHipblasltKernelName
from aiter import rocb_create_extension, rocb_mm
from aiter import logger, dtypes
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.jit.utils.chip_info import get_cu_num
from aiter import gemm_a16w16_asm
from aiter.ops.gemm_op_common import get_padded_m
from aiter.jit.core import (
    AITER_CONFIG_GEMM_BF16_FILE,
    AITER_LOG_TUNED_CONFIG,
)

this_dir = os.path.dirname(os.path.abspath(__file__))

bestsols = {}

solMap = ["torch", "hipblaslt", "rocblas", "skinny", "asm"]

# We need to set is 0 as default, None will error in torch.compile fakeTensor execution
soltype = 0


@torch_compile_guard()
def load_best_sols_custom(tune_path: str) -> bool:
    global bestsols
    cu_count = get_cu_num()
    if tune_path is not None and Path(tune_path).is_file():
        bestsols = pd.read_csv(tune_path)
        bestsols = bestsols[bestsols["cu_num"] == cu_count]
        if len(bestsols) > 0 and "kernelName" in bestsols.columns:
            hipblasltKernelNames = bestsols.apply(
                lambda s: (
                    getHipblasltKernelName(s.solidx)
                    if s.libtype == "hipblaslt"
                    else s.kernelName
                ),
                axis=1,
            )
            pd.set_option("display.max_colwidth", 100)
            assert hipblasltKernelNames.equals(bestsols["kernelName"]), (
                "error: gradlib tune gemm not match the current environment, need re-tune!!!\n"
                + f"differece:\n{pd.concat([bestsols[['solidx','kernelName']], hipblasltKernelNames], axis=1)[hipblasltKernelNames != bestsols['kernelName'].fillna('')]}"
            )
            return True

    return False


_GEMMA16W16_CONFIG_CACHE = None


@torch_compile_guard()
def get_GEMM_A16W16_config_(tuned_file: str = None) -> None:
    if tuned_file is None:
        tuned_file = AITER_CONFIG_GEMM_BF16_FILE
    global _GEMMA16W16_CONFIG_CACHE

    if _GEMMA16W16_CONFIG_CACHE is None:
        _GEMMA16W16_CONFIG_CACHE = {}
    if os.path.exists(tuned_file):
        gemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        _GEMMA16W16_CONFIG_CACHE = gemm_dict.set_index(
            ["cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB"]
        ).to_dict("index")
    return None


@functools.lru_cache(maxsize=4096)
def get_GEMM_A16W16_config(
    M: int, N: int, K: int, bias: bool, dtype: str, otype: str, scaleAB: bool = False
):
    get_GEMM_A16W16_config_(AITER_CONFIG_GEMM_BF16_FILE)
    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = _GEMMA16W16_CONFIG_CACHE.get(
            (cu_num, padded_M, N, K, bias, str(dtype), str(otype), scaleAB), None
        )
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                kernelName = config["kernelName"] if config["libtype"] == "asm" else ""
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIG_GEMM_BF16_FILE}, libtype is {config["libtype"]}, kernel name is {kernelName}"
                )
            break
    if config is None:
        default_config = {}
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, not found tuned config in {AITER_CONFIG_GEMM_BF16_FILE}, will use default config!"
        )
        if dtype in [dtypes.fp16, dtypes.bf16] and K % 8 == 0:
            if (
                ((M == 1 and N <= 2 * cu_num) or (M > 1 and M <= 4 and N <= cu_num))
                and K <= 9216
                or (M > 4 and M <= 8 and N <= cu_num)
                and K <= 5120
                or (M > 8 and M <= 16 and N <= cu_num)
                and K <= 256
            ):
                # soltype, solution_idx = 3, 2
                default_config["libtype"] = "skinny"
                default_config["solidx"] = 2
                default_config["kernelName"] = ""
                return {"libtype": "skinny", "solidx": 2, "kernelName": ""}
        else:
            default_config["libtype"] = "torch"
            default_config["solidx"] = 0
        logger.info(
            f"using {default_config["libtype"]} solution:{default_config["solidx"]} for {M=} {N=} {K=} {dtype=} {bias=}, {scaleAB=}"
        )
        return default_config

    return config


class TunedGemm:
    """bf16/fp16 with per tensor fp8 quant"""

    def __init__(self):
        self.extensions_created = False
        self.save_gemm = int(os.environ.get("AITER_TUNE_GEMM", 0))
        self.untune_path = f"{this_dir}/configs/untuned_gemm.csv"
        self.tune_path = AITER_CONFIG_GEMM_BF16_FILE
        self.bestsols = {}
        self.solMap = ["torch", "hipblaslt", "rocblas", "skinny", "asm"]
        self.cu_count = torch.cuda.get_device_properties(
            device="cuda"
        ).multi_processor_count

        # self.use_skinny = is_hip() and VLLM_USE_ROCM_SKINNY_GEMM and \
        #     "gfx1" not in torch.cuda.get_device_properties('cuda').gcnArchName
        self.use_skinny = True

        if self.save_gemm == 1:
            self.tuned_df = pd.DataFrame(
                columns=["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB"]
            )
        else:
            self.tuned_df = None

    def load_best_sols(self):
        if load_best_sols_custom(self.tune_path):
            global bestsols
            self.bestsols = bestsols

    def create_ds(self):
        self.solfuncs = [
            self.apply_torch_mm,
            self.apply_hipb_mm,
            self.apply_rocb_mm,
            self.apply_skinny,
            self.apply_asm_mm,
        ]

    def apply_skinny(
        self,
        inp,
        weights,
        solidx,
        bias=None,
        otype=None,
        scale_a=None,
        scale_b=None,
        scale_c=None,
    ):
        import aiter as ops

        if solidx == 0:
            out = torch.empty(
                inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
            )
            ops.wvSpltK(weights, inp, out, inp.shape[0], self.cu_count)
        elif solidx == 1:
            out = torch.empty(
                inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
            )
            ops.LLMM1(weights, inp, out, 4)
        if solidx == 2:
            out = torch.empty(
                inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
            )
            ops.wv_splitk_small_fp16_bf16(
                weights, inp, out, inp.shape[0], self.cu_count
            )
        if bias is not None:
            out += bias
        return out

    def apply_hipb_mm(
        self,
        inp,
        weights,
        solidx,
        bias=None,
        otype=None,
        scale_a=None,
        scale_b=None,
        scale_c=None,
    ):
        if otype is None:
            otype = inp.dtype
        return hipb_mm(inp, weights.t(), solidx, bias, otype, scale_a, scale_b, scale_c)

    def apply_rocb_mm(
        self,
        inp,
        weights,
        solidx,
        bias=None,
        otype=None,
        scale_a=None,
        scale_b=None,
        scale_c=None,
    ):
        assert (
            scale_a is None and scale_b is None and scale_c is None
        ), "scale_a, scale_b, scale_c must be None for rocblas"
        out = rocb_mm(inp, weights.t(), solidx)
        if bias is not None:
            out = out + bias
        return out

    def apply_torch_mm(
        self,
        inp,
        weights,
        solidx,
        bias=None,
        otype=None,
        scale_a=None,
        scale_b=None,
        scale_c=None,
    ):
        if self.save_gemm == 1:
            m, k = inp.shape
            n = weights.shape[0]
            self.tuned_df = pd.concat(
                [
                    self.tuned_df,
                    pd.DataFrame(
                        {
                            "M": [m],
                            "N": [n],
                            "K": [k],
                            "bias": [bias is not None],
                            "dtype": [inp.dtype],
                            "outdtype": [otype],
                            "scaleAB": [scale_a is not None or scale_b is not None],
                        }
                    ),
                ]
            ).drop_duplicates()
            self.tuned_df.to_csv(self.untune_path, index=False)
        if inp.dtype == dtypes.fp8:
            if scale_a is None:
                scale_a = torch.ones(1, dtype=dtypes.fp32, device=inp.device)
            if scale_b is None:
                scale_b = torch.ones(1, dtype=dtypes.fp32, device=inp.device)

            try:
                out = torch._scaled_mm(
                    inp,
                    weights.t(),
                    out_dtype=otype,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    bias=bias,
                )
            except RuntimeError:
                out = (
                    F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32))
                    * scale_a
                    * scale_b
                )
                out = (out.to(otype) + bias) if bias is not None else out.to(otype)
            return out
        out = F.linear(inp, weights, bias)
        if otype is not None:
            out = out.to(otype)
        return out

    def apply_asm_mm(
        self,
        inp,
        weights,
        bias=None,
        otype=None,
        splitK=None,
        KernelName=None,
    ):
        # just support bf16gemm_outFp32
        out_asm = torch.empty(
            inp.shape[0], weights.shape[0], dtype=otype, device=inp.device
        )
        return gemm_a16w16_asm(inp, weights, out_asm, bias, splitK, KernelName)

    def mm(
        self,
        inp,
        weights,
        bias=None,
        otype=None,
        scale_a=None,
        scale_b=None,
        scale_c=None,
    ):
        # F.Linear can take a 3 dimensional input. vllm
        # uses this for linear units. However, sampler
        # will use torch.matmul with 2 dimensions only
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
            self.load_best_sols()
            self.create_ds()
        if inp.dim() >= 3:
            try:
                inp_view = inp.view(-1, inp.size(-1))
                batched = True
            except RuntimeError:
                return F.linear(inp, weights, bias)
        else:
            inp_view = inp
            batched = False
        m, k = inp_view.shape
        n = weights.shape[0]
        use_bias = bias is not None
        config = get_GEMM_A16W16_config(
            M=m,
            N=n,
            K=k,
            bias=use_bias,
            dtype=str(inp.dtype),
            otype=str(otype) if otype is not None else str(inp.dtype),
            scaleAB=scale_a is not None or scale_b is not None,
        )

        if config is not None and config["libtype"] == "asm":
            kernelName = config["kernelName"]
            splitK = config["splitK"]
            out = self.apply_asm_mm(inp_view, weights, bias, otype, splitK, kernelName)
        else:
            soltype = solMap.index(config["libtype"])
            solution_idx = config["solidx"]
            out = self.solfuncs[soltype](
                inp_view, weights, solution_idx, bias, otype, scale_a, scale_b, scale_c
            )
        if batched:
            out = out.view(*inp.shape[:-1], weights.shape[0])
        if otype is not None:
            out = out.to(otype)
        return out


tgemm = TunedGemm()
