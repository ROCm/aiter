"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2026, The vLLM team.
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

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from aiter import dtypes, gemm_a16w16_asm, hipb_create_extension, hipb_mm, logger
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.gemm_op_common import get_padded_m

try:
    from aiter.ops.opus.gemm_op_a16w16 import opus_gemm_a16w16_tune as _opus_tune
except Exception:  # noqa: BLE001  blanket catch is intentional here
    _opus_tune = None


@functools.lru_cache(maxsize=1)
def _get_flydsl_gemm_kernels():
    from aiter.ops.flydsl import gemm_kernels

    return gemm_kernels


# NOTE: gfx1250 split-K kids allocate their partial-sum workspace as a plain
# torch.empty tensor (see aiter.ops.opus.gemm_op_a16w16._get_opus_workspace)
# passed explicitly to the launcher. torch's caching allocator is HIP graph-
# capture aware, so that single torch.empty path serves both eager and capture
# (a buffer first touched inside capture comes from the graph mempool with a
# replay-stable address) and no eager pre-warm of the shape is required. (The
# old per-stream hipMalloc registry -- opus_gemm_workspace_init /
# opus_splitk_ws_get -- used by the gfx942/gfx950 a16w16 split-K path still needs
# an eager warm before capture; if that path is ever exercised under cudagraphs,
# warm it via aiter.opus_gemm_workspace_init() on the capture stream. It fails
# loudly ("splitk workspace not initialized") rather than silently corrupting,
# so its absence here is safe to detect.)


this_dir = os.path.dirname(os.path.abspath(__file__))


extensions_created = False
untune_path = f"{this_dir}/configs/bf16_untuned_gemm.csv"
tune_path = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
tuned_df = pd.DataFrame(
    columns=[
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ]
)


@functools.lru_cache(maxsize=1)
def get_GEMM_A16W16_config_():
    tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
    gemm_dict = {}
    if os.path.exists(tuned_file):
        gemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        gemm_dict = gemm_dict.set_index(
            [
                "gfx",
                "cu_num",
                "M",
                "N",
                "K",
                "bias",
                "dtype",
                "outdtype",
                "scaleAB",
                "bpreshuffle",
            ]
        ).to_dict("index")
    return gemm_dict


def is_skinny_default_shape(
    M: int,
    N: int,
    K: int,
    dtype,
    cu_num: int | None = None,
):
    if isinstance(dtype, str):
        dtype = eval(dtype)
    cu_num = get_cu_num() if cu_num is None else cu_num
    return (
        dtype in [dtypes.fp16, dtypes.bf16]
        and K % 8 == 0
        and (
            (
                ((M == 1 and N <= 2 * cu_num) or (M > 1 and M <= 4 and N <= cu_num))
                and K <= 9216
            )
            or ((M > 4 and M <= 8 and N <= cu_num) and K <= 5120)
            or ((M > 8 and M <= 16 and N <= cu_num) and K <= 256)
        )
    )


@functools.lru_cache(maxsize=4096)
def get_GEMM_A16W16_config(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: str,
    otype: str,
    scaleAB: bool = False,
    bpreshuffle: bool = False,
):
    cfg = get_GEMM_A16W16_config_()
    cu_num = get_cu_num()
    padded_M = M
    config = None
    gfx = get_gfx()
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = cfg.get(
            (
                gfx,
                cu_num,
                padded_M,
                N,
                K,
                bias,
                str(dtype),
                str(otype),
                scaleAB,
                bpreshuffle,
            ),
            None,
        )
        if config is not None:
            if config["libtype"] == "flydsl":
                flydsl_config = (
                    _get_flydsl_gemm_kernels().get_flydsl_hgemm_kernel_params(
                        config["kernelName"]
                    )
                )
                # None means the tuned CSV names a kernel absent from this
                # catalog version; it is unrelated to FlyDSL import availability.
                if flydsl_config is None:
                    logger.warning(
                        f"FlyDSL kernel '{config['kernelName']}' from tuned config is not "
                        "recognized by the current catalog; falling back to next candidate."
                    )
                    config = None
            if config is None:
                continue
            if AITER_LOG_TUNED_CONFIG:
                kernelName = (
                    config["kernelName"] if config["libtype"] != "hipblaslt" else ""
                )
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, {bpreshuffle=} found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE}, libtype is {config['libtype']}, kernel name is {kernelName}"
                )
            return config

    if config is None:
        default_config = {}
        if bpreshuffle:
            default_config["bpreshuffle"] = True
            if gfx == "gfx942":
                default_config["libtype"] = "hipblaslt"
                default_config["solidx"] = -1
                default_config["kernelName"] = ""
            elif (
                eval(dtype) == dtypes.bf16
                and N % 64 == 0
                and K % 64 == 0
                and (eval(otype) == dtypes.bf16 or eval(otype) == dtypes.fp32)
            ):
                default_config["libtype"] = "asm"
                default_config["solidx"] = 0
                default_config["splitK"] = None
                default_config["kernelName"] = None
            else:
                assert (
                    False
                ), f"no solution for {M=} {N=} {K=} {dtype=} {bias=}, {scaleAB=}, {bpreshuffle=}"
        elif gfx in ("gfx90a", "gfx942", "gfx950") and is_skinny_default_shape(
            M, N, K, dtype, cu_num
        ):
            default_config["libtype"] = "skinny"
            default_config["solidx"] = 2
            default_config["kernelName"] = ""
        if not default_config:
            # gfx1250 has no tuned ASM/skinny/hipblaslt bf16 kernels, so the
            # torch fallback lands on hipBLASLt, which is markedly slower than
            # the Triton (gluon) a16w16 kernel for these shapes. Prefer Triton
            # for unscaled bf16/fp16 GEMMs; explicit tuned CSV entries still win
            # since they are matched before this fallback is reached.
            if (
                gfx == "gfx1250"
                and not scaleAB
                and eval(dtype) in (dtypes.bf16, dtypes.fp16)
            ):
                default_config["libtype"] = "triton"
                default_config["solidx"] = 0
            else:
                default_config["libtype"] = "torch"
                default_config["solidx"] = 0
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, {bpreshuffle=}, not found tuned config in {AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE}, will use default config! using {default_config['libtype']} solution:{default_config['solidx']}"
        )
        return default_config

    return config


def save_shapes(
    M,
    N,
    K,
    bias,
    dtype,
    otype,
    scaleAB,
    bpreshuffle,
):
    save_gemm = int(os.environ.get("AITER_TUNE_GEMM", "0"))
    global tuned_df
    if save_gemm:
        tuned_df = pd.concat(
            [
                tuned_df,
                pd.DataFrame(
                    {
                        "M": [M],
                        "N": [N],
                        "K": [K],
                        "bias": [bias is not None],
                        "dtype": [dtype],
                        "outdtype": [otype],
                        "scaleAB": [scaleAB],
                        "bpreshuffle": [bpreshuffle],
                    }
                ),
            ]
        ).drop_duplicates()
        tuned_df.to_csv(untune_path, index=False)


def gen_gemm_a16w16_fake_tensor(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> Tensor:
    return torch.empty(
        *A.shape[:-1],
        B.shape[0],
        dtype=otype or A.dtype,
        device=A.device,
    )


@torch_compile_guard(gen_fake=gen_gemm_a16w16_fake_tensor)
def gemm_a16w16(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
) -> Tensor:
    bpreshuffle = False
    if hasattr(B, "is_shuffled") and B.is_shuffled is True:
        bpreshuffle = True
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
    otype = otype if otype is not None else inp_view.dtype
    config = get_GEMM_A16W16_config(
        M=m,
        N=n,
        K=k,
        bias=use_bias,
        dtype=str(inp_view.dtype),
        otype=str(otype),
        scaleAB=scale_a is not None or scale_b is not None,
        bpreshuffle=bpreshuffle,
    )
    libtype = config["libtype"]
    solution_idx = config["solidx"]
    solfunc = solMap[libtype]
    out = solfunc(
        inp_view,
        B,
        solution_idx,
        bias,
        otype,
        scale_a,
        scale_b,
        scale_c,
        bpreshuffle,
        config=config,
    )
    if batched:
        out = out.view(*A.shape[:-1], B.shape[0])
    if otype is not None and out.dtype != otype:
        out = out.to(otype)
    save_shapes(
        m,
        n,
        k,
        bias,
        inp_view.dtype,
        otype,
        scale_a is not None or scale_b is not None,
        bpreshuffle,
    )
    return out


def skinny_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
):
    import aiter as ops

    assert not bpreshuffle, "bpreshuffle is not supported in skinny_gemm!"
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
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
):
    if otype is None:
        otype = inp.dtype
    global extensions_created
    if not extensions_created:
        hipb_create_extension()
        extensions_created = True
    return hipb_mm(
        inp, weights.t(), solidx, bias, otype, scale_a, scale_b, scale_c, bpreshuffle
    )


def torch_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
):
    assert not bpreshuffle, "bpreshuffle is not supported in torch_gemm!"
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
    return out


def asm_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
):
    kernelName = config.get("kernelName") if config else None
    splitK = config.get("splitK") if config else None
    out_asm = torch.empty(
        inp.shape[0], weights.shape[0], dtype=otype, device=inp.device
    )
    return gemm_a16w16_asm(inp, weights, out_asm, bias, splitK, kernelName, bpreshuffle)


def flydsl_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle=False,
    config: dict | None = None,
):
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "FlyDSL hgemm does not support scaling yet."
    flydsl_gemm_kernels = _get_flydsl_gemm_kernels()
    flydsl_config = flydsl_gemm_kernels.get_flydsl_hgemm_kernel_params(
        config["kernelName"]
    )
    fused_bias = None
    if (
        bias is not None
        and (otype is None or otype == inp.dtype)
        and bias.dtype == inp.dtype
    ):
        fused_bias = bias
    out = flydsl_gemm_kernels.flydsl_hgemm(
        inp,
        weights,
        bias=fused_bias,
        block_m=flydsl_config["block_m"],
        block_n=flydsl_config["block_n"],
        block_k=flydsl_config["block_k"],
        split_k=flydsl_config["split_k"],
        m_waves=flydsl_config["m_waves"],
        n_waves=flydsl_config["n_waves"],
        k_waves=flydsl_config["k_waves"],
        stages=flydsl_config["stages"],
        group_m=flydsl_config["group_m"],
        policy=("ht" if flydsl_config["use_half_tile_interleaved"] else "ft"),
        out_dtype=otype,
    )

    if bias is not None and fused_bias is None:
        out = out.to(bias.dtype) + bias
    if otype is not None and out.dtype != otype:
        out = out.to(otype)
    return out


def opus_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle: bool | None = False,
    config: dict | None = None,
):
    if _opus_tune is None:
        logger.warning(
            "opus tuned config found but opus is not available; falling back to torch"
        )
        return torch_gemm(
            inp,
            weights,
            solidx,
            bias,
            otype,
            scale_a,
            scale_b,
            scale_c,
            bpreshuffle,
            config,
        )
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "opus_gemm does not support scaling"
    assert not bpreshuffle, "opus_gemm does not support bpreshuffle"
    splitK = int(config.get("splitK", 0)) if config is not None else 0
    m, _k = inp.shape
    n = weights.shape[0]
    # The split-K workspace (if any) is allocated capture-safely inside
    # opus_gemm_a16w16_tune -> _get_opus_workspace; no eager pre-warm needed.
    Y = torch.empty(m, n, dtype=otype or inp.dtype, device=inp.device)
    _opus_tune(
        inp.unsqueeze(0),
        weights.unsqueeze(0),
        Y.unsqueeze(0),
        bias=bias,
        kernelId=int(solidx),
        splitK=splitK,
    )
    # NOTE: do NOT add bias again here -- the opus splitk reduce kernel already
    # folds `bias` into the fp32 accumulator before the bf16/fp32 cast (HAS_BIAS
    # path). The previous `Y = Y + bias` double-counted bias (output = A@B^T +
    # 2*bias), causing ~54% miscompare (maxabs ~= bias range) for every bias!=None
    # opus shape under tgemm (e.g. ATOM's bf16 linear).
    return Y


def triton_gemm(
    inp: Tensor,
    weights: Tensor,
    solidx: int,
    bias: Tensor | None = None,
    otype: torch.dtype | None = None,
    scale_a: Tensor | None = None,
    scale_b: Tensor | None = None,
    scale_c: Tensor | None = None,
    bpreshuffle: bool | None = False,
    config: dict | None = None,
):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "Triton gemm_a16w16 does not support scaling yet"
    assert not bpreshuffle, "Triton gemm_a16w16 does not support bpreshuffle yet."
    return gemm_a16w16(inp, weights, bias=bias, dtype=otype)


solMap = {
    "torch": torch_gemm,
    "hipblaslt": hipb_gemm,
    "skinny": skinny_gemm,
    "asm": asm_gemm,
    "triton": triton_gemm,
    "flydsl": flydsl_gemm,
    "opus": opus_gemm,
}


class TunedGemm:
    """bf16/fp16 with per tensor fp8 quant"""

    def __init__(self):
        # self.extensions_created = False
        self.save_gemm = int(os.environ.get("AITER_TUNE_GEMM", "0"))
        self.untune_path = f"{this_dir}/configs/bf16_untuned_gemm.csv"
        self.tune_path = AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE
        if self.save_gemm == 1:
            self.tuned_df = pd.DataFrame(
                columns=[
                    "M",
                    "N",
                    "K",
                    "bias",
                    "dtype",
                    "outdtype",
                    "scaleAB",
                    "bpreshuffle",
                ]
            )
        else:
            self.tuned_df = None

    def mm(
        self,
        inp: Tensor,
        weights: Tensor,
        bias: Tensor | None = None,
        otype: torch.dtype | None = None,
        scale_a: Tensor | None = None,
        scale_b: Tensor | None = None,
        scale_c: Tensor | None = None,
    ):

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
