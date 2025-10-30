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
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from aiter import (
    dtypes,
    gemm_a16w16_asm,
    hipb_create_extension,
    hipb_mm,
    logger,
    rocb_create_extension,
    rocb_mm,
)
from aiter.jit.core import AITER_CONFIG_GEMM_BF16_FILE, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.gemm_op_common import get_padded_m

this_dir = os.path.dirname(os.path.abspath(__file__))


solMap = ["torch", "hipblaslt", "rocblas", "skinny", "asm"]


def get_solfunc(soltype: int):
    if soltype == 0:
        return torch_gemm
    elif soltype == 1:
        return hipb_gemm
    elif soltype == 2:
        return rocb_gemm
    elif soltype == 3:
        return skinny_gemm
    elif soltype == 4:
        return asm_gemm


@functools.lru_cache(maxsize=1)
def get_GEMM_A16W16_config_():
    tuned_file = AITER_CONFIG_GEMM_BF16_FILE
    gemm_dict = {}
    if os.path.exists(tuned_file):
        gemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        gemm_dict = gemm_dict.set_index(
            ["cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB"]
        ).to_dict("index")
    return gemm_dict


@functools.lru_cache(maxsize=4096)
def get_GEMM_A16W16_config(
    M: int, N: int, K: int, bias: bool, dtype: str, otype: str, scaleAB: bool = False
):
    cfg = get_GEMM_A16W16_config_()
    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = cfg.get(
            (cu_num, padded_M, N, K, bias, str(dtype), str(otype), scaleAB), None
        )
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                kernelName = config["kernelName"] if config["libtype"] == "asm" else ""
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIG_GEMM_BF16_FILE}, libtype is {config['libtype']}, kernel name is {kernelName}"
                )
            return config
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
            f"using {default_config['libtype']} solution:{default_config['solidx']} for {M=} {N=} {K=} {dtype=} {bias=}, {scaleAB=}"
        )
        return default_config

    return config


def gen_gemm_a16w16_fake_tensor(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
) -> Tensor:
    out = torch.empty(
        A.view(-1, A.size(-1)).shape[0], B.shape[0], dtype=A.dtype, device=A.device
    )
    if otype is not None:
        out = out.to(otype)
    return out.view(*A.shape[:-1], B.shape[0])


@torch_compile_guard(gen_fake=gen_gemm_a16w16_fake_tensor)
def gemm_a16w16(
    A: Tensor,
    B: Tensor,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
) -> Tensor:
    if A.dim() >= 3:
        try:
            inp_view = A.view(-1, A.size(-1))
            batched = True
        except RuntimeError:
            return F.linear(A, B, bias)
    else:
        inp_view = A
        batched = False
    m, k = inp_view.shape
    n = B.shape[0]
    use_bias = bias is not None
    config = get_GEMM_A16W16_config(
        M=m,
        N=n,
        K=k,
        bias=use_bias,
        dtype=str(inp_view.dtype),
        otype=str(otype) if otype is not None else str(inp_view.dtype),
        scaleAB=scale_a is not None or scale_b is not None,
    )

    if config is not None and config["libtype"] == "asm":
        kernelName = config["kernelName"]
        splitK = config["splitK"]
        out = asm_gemm(inp_view, B, bias, otype, splitK, kernelName)
    else:
        soltype = solMap.index(config["libtype"])
        solution_idx = config["solidx"]
        solfunc = get_solfunc(soltype)
        out = solfunc(inp_view, B, solution_idx, bias, otype, scale_a, scale_b, scale_c)
    if batched:
        out = out.view(*A.shape[:-1], B.shape[0])
    if otype is not None:
        out = out.to(otype)
    return out


def skinny_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
):
    import aiter as ops

    if solidx == 0:
        out = torch.empty(
            inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
        )
        ops.wvSpltK(weights, inp, out, inp.shape[0], get_cu_num())
    elif solidx == 1:
        out = torch.empty(
            inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
        )
        ops.LLMM1(weights, inp, out, 4)
    if solidx == 2:
        out = torch.empty(
            inp.shape[0], weights.shape[0], dtype=inp.dtype, device=inp.device
        )
        ops.wv_splitk_small_fp16_bf16(weights, inp, out, inp.shape[0], get_cu_num())
    if bias is not None:
        out += bias
    return out


def hipb_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
):
    if otype is None:
        otype = inp.dtype
    return hipb_mm(inp, weights.t(), solidx, bias, otype, scale_a, scale_b, scale_c)


def rocb_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
):
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "scale_a, scale_b, scale_c must be None for rocblas"
    out = rocb_mm(inp, weights.t(), solidx)
    if bias is not None:
        out = out + bias
    return out


def torch_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
):
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


def asm_gemm(
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


class TunedGemm:
    """bf16/fp16 with per tensor fp8 quant"""

    def __init__(self):
        self.extensions_created = False
        self.save_gemm = int(os.environ.get("AITER_TUNE_GEMM", 0))
        self.untune_path = f"{this_dir}/configs/untuned_gemm.csv"
        self.tune_path = AITER_CONFIG_GEMM_BF16_FILE

    def mm(
        self,
        inp: Tensor,
        weights: Tensor,
        bias: Optional[Tensor] = None,
        otype: Optional[torch.dtype] = None,
        scale_a: Optional[Tensor] = None,
        scale_b: Optional[Tensor] = None,
        scale_c: Optional[Tensor] = None,
    ):
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        out = gemm_a16w16(
            inp,
            weights,
            bias=bias,
            otype=otype,
            scale_a=scale_a,
            scale_b=scale_b,
            scale_c=scale_c,
        )
        return out


tgemm = TunedGemm()
