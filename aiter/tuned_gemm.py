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
import sys
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from aiter import (
    dtypes,
    gemm_a16w16_asm,
    get_semaphore_workspace,
    hipb_create_extension,
    hipb_mm,
    logger,
)
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.gemm_op_common import get_padded_m

this_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------
# Optional FlyDSL preshuffle GEMM
# ---------------------------------
# Enable with:
#   export AITER_USE_FLYDSL_GEMM=1
#   export DSL2_ROOT=/home/gyu/zxe/FlyDSL
#
# Notes/constraints (current FlyDSL kernel):
# - Only supports B in bpreshuffle layout (layout=(16,16) shuffle_weight).
# - No N-tail support (N must be divisible by tile_n).
# - Output is fp16; we cast to requested `otype` in `gemm_a16w16`.
# - Scales are ignored for fp16/bf16 inputs; we pass empty tensors for ABI.
_flydsl_preshuffle_available: bool = False
_flydsl_preshuffle_failed_once: bool = False
_flydsl_compile_preshuffle_gemm_a8 = None
_flydsl_debug_seen: set[str] = set()


def _flydsl_debug_enabled() -> bool:
    return os.environ.get("AITER_FLYDSL_DEBUG", "0") in ("1", "true", "True", "YES", "yes")


def _flydsl_log_once(key: str, msg: str) -> None:
    if not _flydsl_debug_enabled():
        return
    if key in _flydsl_debug_seen:
        return
    _flydsl_debug_seen.add(key)
    logger.info(msg)


def _use_flydsl_gemm() -> bool:
    return os.environ.get("AITER_USE_FLYDSL_GEMM", "0") in ("1", "true", "True", "YES", "yes")


def _init_flydsl_preshuffle_backend() -> None:
    global _flydsl_preshuffle_available, _flydsl_preshuffle_failed_once, _flydsl_compile_preshuffle_gemm_a8
    if _flydsl_preshuffle_failed_once or _flydsl_preshuffle_available:
        return
    if not _use_flydsl_gemm():
        return

    dsl2_root = os.environ.get("DSL2_ROOT", None)
    if not dsl2_root:
        logger.warning("AITER_USE_FLYDSL_GEMM=1 but DSL2_ROOT is not set; falling back to existing backends.")
        _flydsl_preshuffle_failed_once = True
        return

    # Prefer the FlyDSL python sources from this checkout.
    flydsl_src = os.path.join(dsl2_root, "flydsl", "src")
    for p in (flydsl_src, dsl2_root):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8 as _compile  # type: ignore

        _flydsl_compile_preshuffle_gemm_a8 = _compile
        _flydsl_preshuffle_available = True
    except Exception as e:
        logger.warning(
            f"FlyDSL preshuffle GEMM backend not available ({type(e).__name__}: {e}); falling back to existing backends."
        )
        _flydsl_preshuffle_failed_once = True


def _flydsl_in_dtype(dt: torch.dtype) -> Optional[str]:
    if dt == torch.float16:
        return "fp16"
    if dt == torch.bfloat16:
        return "bf16"
    return None


def _flydsl_choose_tile_n(n: int) -> Optional[int]:
    for cand in (64, 128, 256):
        if n % cand == 0:
            return cand
    return None


def _flydsl_choose_tile_k(k: int, in_dtype: str) -> int:
    # tile_k is in element units; kernel requires tile_k_bytes % 64 == 0.
    if in_dtype in ("fp16", "bf16"):
        for cand in (128, 64, 32):
            if k % cand == 0:
                return cand
        return 32
    for cand in (256, 128, 64):
        if k % cand == 0:
            return cand
    return 64


@functools.lru_cache(maxsize=256)
def _flydsl_get_exe(n: int, k: int, in_dtype: str, tile_m: int, tile_n: int, tile_k: int, lds_stage: int):
    _init_flydsl_preshuffle_backend()
    if not _flydsl_preshuffle_available or _flydsl_compile_preshuffle_gemm_a8 is None:
        return None
    # NOTE: compile signature requires M,N,K though M is dynamic at runtime; pass M=1.
    return _flydsl_compile_preshuffle_gemm_a8(
        M=1,
        N=int(n),
        K=int(k),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        in_dtype=in_dtype,
        lds_stage=int(lds_stage),
        use_cshuffle_epilog=False,
    )


def _flydsl_preshuffle_gemm(
    inp: Tensor,
    weights_bpreshuffle: Tensor,
    bias: Optional[Tensor],
    otype: torch.dtype,
) -> Optional[Tensor]:
    """Best-effort FlyDSL preshuffle GEMM: out = inp @ W^T (+ bias).

    Returns None if FlyDSL is unavailable or constraints are not met.
    """
    global _flydsl_preshuffle_failed_once, _flydsl_preshuffle_available
    _init_flydsl_preshuffle_backend()
    if not _flydsl_preshuffle_available:
        _flydsl_log_once(
            "flydsl_a16w16_unavailable",
            "[flydsl][a16w16] backend unavailable -> fallback",
        )
        return None

    # FlyDSL preshuffle GEMM path is not CUDA-graph-capture safe today.
    if torch.cuda.is_current_stream_capturing():
        _flydsl_log_once(
            "flydsl_a16w16_skip_cudagraph",
            "[flydsl][a16w16] skip: stream is capturing (cudagraph)",
        )
        return None

    if inp.dim() != 2 or weights_bpreshuffle.dim() != 2:
        _flydsl_log_once(
            f"flydsl_a16w16_skip_rank_{inp.dim()}_{weights_bpreshuffle.dim()}",
            f"[flydsl][a16w16] skip: expect 2D tensors, got A.dim={inp.dim()} B.dim={weights_bpreshuffle.dim()}",
        )
        return None

    in_dtype = _flydsl_in_dtype(inp.dtype)
    if in_dtype is None:
        _flydsl_log_once(
            f"flydsl_a16w16_skip_dtype_{str(inp.dtype)}",
            f"[flydsl][a16w16] skip: unsupported inp dtype {inp.dtype}",
        )
        return None
    if weights_bpreshuffle.dtype != inp.dtype:
        _flydsl_log_once(
            f"flydsl_a16w16_skip_dtype_mismatch_{str(inp.dtype)}_{str(weights_bpreshuffle.dtype)}",
            f"[flydsl][a16w16] skip: dtype mismatch A={inp.dtype} B={weights_bpreshuffle.dtype}",
        )
        return None

    m, k = inp.shape
    n = weights_bpreshuffle.shape[0]
    if weights_bpreshuffle.shape[1] != k:
        _flydsl_log_once(
            f"flydsl_a16w16_skip_shape_{m}_{n}_{k}_{weights_bpreshuffle.shape[1]}",
            f"[flydsl][a16w16] skip: shape mismatch A[K]={k} B[K]={weights_bpreshuffle.shape[1]}",
        )
        return None

    tile_m = 16
    tile_n = _flydsl_choose_tile_n(n)
    if tile_n is None:
        _flydsl_log_once(
            f"flydsl_a16w16_skip_ntail_{n}",
            f"[flydsl][a16w16] skip: N={n} not divisible by supported tile_n (64/128/256)",
        )
        return None
    tile_k = _flydsl_choose_tile_k(k, in_dtype)
    if k % tile_k != 0:
        _flydsl_log_once(
            f"flydsl_a16w16_skip_kalign_{k}_{tile_k}",
            f"[flydsl][a16w16] skip: K={k} not divisible by tile_k={tile_k} (in_dtype={in_dtype})",
        )
        return None

    exe = _flydsl_get_exe(n=n, k=k, in_dtype=in_dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, lds_stage=2)
    if exe is None:
        _flydsl_log_once(
            f"flydsl_a16w16_noexe_{m}_{n}_{k}_{in_dtype}_{tile_m}_{tile_n}_{tile_k}",
            f"[flydsl][a16w16] skip: exe compile returned None for M={m} N={n} K={k} in={in_dtype} tile(m,n,k)=({tile_m},{tile_n},{tile_k})",
        )
        return None

    # FlyDSL preshuffle GEMM outputs fp16.
    out_f16 = torch.empty((m, n), dtype=torch.float16, device=inp.device)
    sa = torch.empty((0,), device=inp.device, dtype=torch.float32)  # ignored for fp16/bf16
    sb = torch.empty((0,), device=inp.device, dtype=torch.float32)  # ignored for fp16/bf16
    try:
        _flydsl_log_once(
            f"flydsl_a16w16_hit_{m}_{n}_{k}_{in_dtype}_{tile_m}_{tile_n}_{tile_k}",
            f"[flydsl][a16w16] HIT M={m} N={n} K={k} in={in_dtype} out=fp16 tile(m,n,k)=({tile_m},{tile_n},{tile_k}) lds_stage=2",
        )
        exe(out_f16, inp, weights_bpreshuffle, sa, sb, m, n, k)
    except Exception as e:
        _flydsl_log_once(
            f"flydsl_a16w16_fail_{m}_{n}_{k}_{in_dtype}_{tile_m}_{tile_n}_{tile_k}",
            f"[flydsl][a16w16] FAIL M={m} N={n} K={k} in={in_dtype} tile(m,n,k)=({tile_m},{tile_n},{tile_k}) err={type(e).__name__}: {e}",
        )
        _flydsl_preshuffle_failed_once = True
        _flydsl_preshuffle_available = False
        return None

    out = out_f16 if otype == torch.float16 else out_f16.to(otype)
    if bias is not None:
        out = out + bias
    return out


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

    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        config = cfg.get(
            (
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
            if AITER_LOG_TUNED_CONFIG:
                kernelName = config["kernelName"] if config["libtype"] == "asm" else ""
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, {bpreshuffle=} found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE}, libtype is {config['libtype']}, kernel name is {kernelName}"
                )
            return config

    if config is None:
        default_config = {}
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K} {dtype=} {otype=} {bias=}, {scaleAB=}, {bpreshuffle=} , not found tuned config in {AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE}, will use default config!"
        )
        if bpreshuffle:
            default_config["bpreshuflle"] = True
            if get_gfx() == "gfx942":
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
        elif eval(dtype) in [dtypes.fp16, dtypes.bf16] and K % 8 == 0:
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
        if not default_config:
            default_config["libtype"] = "torch"
            default_config["solidx"] = 0
        logger.info(
            f"using {default_config['libtype']} solution:{default_config['solidx']} for {M=} {N=} {K=} {dtype=} {bias=}, {scaleAB=}, {bpreshuffle=}"
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
    out = torch.empty(
        A.view(-1, A.size(-1)).shape[0],
        B.shape[0],
        dtype=otype or A.dtype,
        device=A.device,
    )
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

    # Optional FlyDSL preshuffle GEMM path (best-effort; falls back automatically).
    # Only enable when:
    # - explicit env var enabled
    # - B is already bpreshuffled (layout=(16,16) shuffle) and tagged via `is_shuffled`
    # - no scaling (A/B scales are for fp8/int8; FlyDSL preshuffle kernel ignores scales for fp16/bf16)
    # - fp16/bf16 input (current kernel outputs fp16; we cast to `otype` below)
    if (
        bpreshuffle
        and _use_flydsl_gemm()
        and scale_a is None
        and scale_b is None
        and scale_c is None
        and inp_view.dtype in (torch.float16, torch.bfloat16)
        and B.dtype == inp_view.dtype
        and otype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        out_flydsl = _flydsl_preshuffle_gemm(inp_view.contiguous(), B.contiguous(), bias, otype=torch.float16)
        if out_flydsl is not None:
            out = out_flydsl
            if batched:
                out = out.view(*A.shape[:-1], B.shape[0])
            if otype is not None:
                out = out.to(otype)
            save_shapes(
                m,
                n,
                k,
                bias,
                inp_view.dtype,
                otype,
                False,
                bpreshuffle,
            )
            return out

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
    if config is not None and config["libtype"] == "asm":
        kernelName = config["kernelName"]
        splitK = config["splitK"]
        out = asm_gemm(inp_view, B, bias, otype, splitK, kernelName, bpreshuffle)
    else:
        solution_idx = config["solidx"]
        solfunc = solMap[config["libtype"]]
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
        )
    if batched:
        out = out.view(*A.shape[:-1], B.shape[0])
    if otype is not None:
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
):
    if otype is None:
        otype = inp.dtype
    global extensions_created
    if extensions_created == False:
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
    bpreshuffle=False,
):
    # just support bf16gemm_outFp32
    out_asm = torch.empty(
        inp.shape[0], weights.shape[0], dtype=otype, device=inp.device
    )
    sema = get_semaphore_workspace(out_asm.device)
    return gemm_a16w16_asm(
        inp, weights, out_asm, sema, bias, splitK, KernelName, bpreshuffle
    )


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
):
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

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
