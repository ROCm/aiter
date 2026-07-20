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
from typing import Optional

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes, gemm_a16w16_asm, hipb_create_extension, hipb_mm, logger
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard

try:
    from aiter.ops.flydsl.utils import is_flydsl_available
except ImportError:

    def is_flydsl_available():
        return False


from aiter.ops.gemm_op_common import get_padded_m
from torch import Tensor

try:
    from aiter.ops.opus.gemm_op_a16w16 import opus_gemm_a16w16_tune as _opus_tune
    from aiter.ops.opus.gemm_op_a16w16 import (
        opus_gemm_workspace_init as _opus_workspace_init,
    )
    from aiter.ops.opus.gemm_op_a16w16 import is_splitk_kid as _opus_is_splitk_kid
except Exception:
    _opus_tune = None
    _opus_workspace_init = None
    _opus_is_splitk_kid = None

# Every opus split-K arch (gfx950 / gfx942 / gfx1250) owns a per-stream fp32
# workspace (process-global `opus_splitk_ws_get` registry, backed by raw
# hipMalloc) that must be registered AND grown to the shape's size *eagerly*
# before HIP graph capture -- hipMalloc/hipFree are stream-capture-illegal, so a
# grow inside capture aborts the capture, leaving an empty graph whose replay
# silently writes zeros (garbage logits). torch.cuda.graph captures on a
# process-global stream (`torch.cuda.graphs.graph.default_capture_stream`) when
# no explicit stream is passed (the vLLM/ATOM CUDAGraphWrapper case); we warm
# that stream here during the eager pass so a later capture of the same shape
# finds a ready workspace. (The opus launcher reads a stable device-resident
# handle, so the captured graph stays valid across replays / post-capture grows
# -- which is exactly why opus keeps a persistent workspace instead of a
# per-call hipMallocAsync that would not survive capture; the only cost is this
# one-time warm.)
_OPUS_WS_ARCHS = {"gfx950", "gfx942", "gfx1250"}
_opus_ws_warmed_sigs = set()


@functools.lru_cache(maxsize=1)
def _opus_needs_ws_prewarm() -> bool:
    if _opus_tune is None or _opus_workspace_init is None:
        return False
    try:
        return get_gfx() in _OPUS_WS_ARCHS
    except Exception:
        return False


def _opus_graph_capture_stream():
    """The stream torch.cuda.graph captures on when called without `stream=`.

    Mirrors torch's own lazy-init so we register the opus workspace on the exact
    stream a later `with torch.cuda.graph(g):` will use.
    """
    g = torch.cuda.graphs.graph
    if getattr(g, "default_capture_stream", None) is None:
        g.default_capture_stream = torch.cuda.Stream()
    return g.default_capture_stream


def _opus_prewarm_capture_workspace(inp, weights, solidx, splitK, bias, otype):
    """Eagerly size the opus split-K workspace on the graph capture stream.

    No-op when already capturing (too late to allocate), on non-registry archs,
    for a non-split-K kid (never touches the workspace), or when this
    (shape, kid, splitK, bias) was already warmed.
    """
    if not _opus_needs_ws_prewarm():
        return
    # Only split-K kids allocate/read the fp32 workspace; every other kid family
    # (flatmm / persistent / mono_tile / nosplit) launches straight to its kernel
    # and never touches the registry, so warming it for them is pure waste.
    if _opus_is_splitk_kid is not None and not _opus_is_splitk_kid(solidx):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    m, k = inp.shape
    n = weights.shape[0]
    sig = (int(solidx), m, n, k, int(splitK), bias is not None, str(otype))
    if sig in _opus_ws_warmed_sigs:
        return
    try:
        s = _opus_graph_capture_stream()
        with torch.cuda.stream(s):
            _opus_workspace_init()
            Yw = torch.empty(m, n, dtype=otype or inp.dtype, device=inp.device)
            _opus_tune(
                inp.unsqueeze(0),
                weights.unsqueeze(0),
                Yw.unsqueeze(0),
                bias=bias,
                kernelId=int(solidx),
                splitK=int(splitK),
            )
        s.synchronize()
        _opus_ws_warmed_sigs.add(sig)
    except Exception as e:  # don't break eager callers; capture would re-surface it
        logger.warning(
            f"opus split-K workspace prewarm on the graph capture stream failed "
            f"({type(e).__name__}: {e}); HIP graph capture of this opus shape may "
            f"produce zeros. Call aiter.opus_gemm_workspace_init() on the capture "
            f"stream manually if you capture with a custom stream."
        )


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
    cu_num: Optional[int] = None,
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
                if is_flydsl_available():
                    flydsl_config = aiter.ops.flydsl.gemm_kernels.get_flydsl_splitk_hgemm_kernel_params(
                        config["kernelName"]
                    )
                    if flydsl_config is None:
                        logger.warning(
                            f"FlyDSL kernel '{config['kernelName']}' from tuned config is not "
                            "recognized by the current catalog; falling back to next candidate."
                        )
                        config = None
                else:
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
        elif is_skinny_default_shape(M, N, K, dtype, cu_num):
            # soltype, solution_idx = 3, 2
            default_config["libtype"] = "skinny"
            default_config["solidx"] = 2
            default_config["kernelName"] = ""
        if not default_config:
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
    save_gemm = int(os.environ.get("AITER_TUNE_GEMM", 0))
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle=False,
    config: Optional[dict] = None,
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle=False,
    config: Optional[dict] = None,
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle=False,
    config: Optional[dict] = None,
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle=False,
    config: Optional[dict] = None,
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle=False,
    config: Optional[dict] = None,
):
    assert (
        scale_a is None and scale_b is None and scale_c is None
    ), "FlyDSL hgemm does not support scaling yet."
    flydsl_config = aiter.ops.flydsl.gemm_kernels.get_flydsl_splitk_hgemm_kernel_params(
        config["kernelName"]
    )
    stages = flydsl_config.get("stages", flydsl_config.get("stage", 2))
    fused_bias = None
    if (
        bias is not None
        and (otype is None or otype == inp.dtype)
        and bias.dtype == inp.dtype
    ):
        fused_bias = bias
    out = aiter.ops.flydsl.gemm_kernels.flydsl_hgemm(
        inp,
        weights,
        bias=fused_bias,
        kernel_family=flydsl_config.get("kernel_family"),
        tile_m=flydsl_config["tile_m"],
        tile_n=flydsl_config["tile_n"],
        tile_k=flydsl_config["tile_k"],
        split_k=flydsl_config["split_k"],
        block_m_warps=flydsl_config["block_m_warps"],
        block_n_warps=flydsl_config["block_n_warps"],
        block_k_warps=flydsl_config.get("block_k_warps", 1),
        n_tile_repeat=flydsl_config.get("n_tile_repeat", 1),
        persistent_n_tiles=flydsl_config.get("persistent_n_tiles", 1),
        waves_per_eu=flydsl_config.get("waves_per_eu", 0),
        b_to_lds_unroll=flydsl_config.get("b_to_lds_unroll", 0),
        stages=stages,
        async_copy=flydsl_config.get("async_copy", False),
        b_to_lds=flydsl_config["b_to_lds"],
        b_preshuffle=flydsl_config.get("b_preshuffle", False),
        c_to_lds=flydsl_config.get("c_to_lds", False),
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle: Optional[bool] = False,
    config: Optional[dict] = None,
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
    m, k = inp.shape
    n = weights.shape[0]
    # Eagerly size the per-stream split-K workspace on torch's graph capture
    # stream so a later HIP graph capture of this shape doesn't abort (which
    # would leave the captured graph empty -> replay writes zeros). No-op when
    # already capturing, on gfx950, or for an already-warmed shape.
    _opus_prewarm_capture_workspace(inp, weights, solidx, splitK, bias, otype)
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
    bias: Optional[Tensor] = None,
    otype: Optional[torch.dtype] = None,
    scale_a: Optional[Tensor] = None,
    scale_b: Optional[Tensor] = None,
    scale_c: Optional[Tensor] = None,
    bpreshuffle: Optional[bool] = False,
    config: Optional[dict] = None,
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
        self.save_gemm = int(os.environ.get("AITER_TUNE_GEMM", 0))
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
        bias: Optional[Tensor] = None,
        otype: Optional[torch.dtype] = None,
        scale_a: Optional[Tensor] = None,
        scale_b: Optional[Tensor] = None,
        scale_c: Optional[Tensor] = None,
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
