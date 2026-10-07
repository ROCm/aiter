# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math
import operator
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import Library

from aiter import logger

from ..jit.core import (
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
    compile_ops,
)
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..jit.utils.torch_guard import torch_compile_guard
from ..ops.gemm_op_common import (
    find_padded_m_row,
    get_padded_m,
    mxscale_w_scale_block,
    with_mxscale_w_scale_block,
)
from ..utility import dtypes
from ..utility.graph_alloc import persistent_alloc
from .mxfp8fp4gemm_common import (
    _MXFP8_GEMM_CONFIG_KEYS,
    get_mxfp8_asm_dir,
    get_mxfp8_config_file,
    mxfp8_compile_guard,
)

aiter_lib = Library("aiter", "FRAGMENT")


# Arches whose prebuilt HIP CK blockscale modules ship matching code objects.
# Other arches (e.g. gfx1201) SIGSEGV uncatchably at kernel launch, so gate
# before the HIP call rather than try/except. Extend when prebuilts add archs.
_BLOCKSCALE_HIP_PREBUILT_ARCHES = frozenset(
    {"gfx940", "gfx941", "gfx942", "gfx950", "gfx1250"}
)


def _hip_blockscale_supported() -> bool:
    """True if the prebuilt HIP CK blockscale module covers the running arch (else triton)."""
    try:
        return get_gfx() in _BLOCKSCALE_HIP_PREBUILT_ARCHES
    except Exception:  # noqa: BLE001
        return False


def _ck_a8w8_supported() -> bool:
    """The CK/asm INT8 a8w8 GEMM ships gfx9 (CDNA) code objects only; other
    arches (e.g. RDNA gfx11/gfx12) must fall back to the Triton kernel."""
    try:
        return get_gfx().startswith("gfx9")
    except Exception:  # noqa: BLE001
        return True


def gen_gemm_a8w8_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    bias: torch.Tensor | None = None,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8", fc_name="gemm_a8w8", gen_fake=gen_gemm_a8w8_ck_fake_tensors
)
def gemm_a8w8_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    bias: torch.Tensor | None = None,
    splitK: int = 0,
) -> torch.Tensor: ...


def gen_gemm_a8w8_bpreshuffle_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_bpreshuffle",
    fc_name="gemm_a8w8_bpreshuffle",
    gen_fake=gen_gemm_a8w8_bpreshuffle_ck_fake_tensors,
)
def gemm_a8w8_bpreshuffle_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor: ...


def gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_bpreshuffle_cktile",
    fc_name="gemm_a8w8_bpreshuffle_cktile",
    gen_fake=gen_gemm_a8w8_bpreshuffle_cktile_fake_tensors,
)
def gemm_a8w8_bpreshuffle_cktile(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    splitK: int = 0,
) -> Tensor: ...


def _parse_flydsl_kernel_name(kernel_name: str):
    """Parse a flydsl kernelName into ``(tile_m, tile_n, tile_k, async_copy,
    waves_per_eu, xcd_swizzle, lds_stage, scheduler, k_split)``, or None on
    failure. Legacy names lacking the xcd/lds/scheduler tokens default them to
    ``0``/``2``/``"Default"``; the ``_ksN`` split-K suffix is only emitted for
    k_split > 1, so every previously tuned name still parses to k_split=1.
    """
    import re

    m = re.match(
        r"flydsl_bpreshuflle_(\d+)x(\d+)x(\d+)_\w+_\w+_\w+_(\d+)x(\d+)(?:x(\d+))?(?:x(\d+))?"
        r"(?:_(?!ks\d+$)([A-Za-z][A-Za-z0-9]*))?(?:_ks(\d+))?$",
        kernel_name,
    )
    if m is None:
        return None
    tm, tn, tk, acp, wpe = (int(m.group(i)) for i in range(1, 6))
    xcd_swizzle = int(m.group(6)) if m.group(6) else 0
    lds_stage = int(m.group(7)) if m.group(7) else 2
    scheduler = m.group(8) if m.group(8) else "Default"
    k_split = int(m.group(9)) if m.group(9) else 1
    return (tm, tn, tk, acp, wpe, xcd_swizzle, lds_stage, scheduler, k_split)


def gemm_a8w8_bpreshuffle_flydsl(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    config: dict,
) -> Tensor:
    kernel_name = str(config.get("kernelName", ""))
    # gfx1250 runs the WMMA ptpc backend; other archs use the MFMA preshuffle path.
    if get_gfx() == "gfx1250":
        from .flydsl.bpreshuffle_gemm_gfx1250 import run_gemm_a8w8_bpreshuffle_gfx1250

        return run_gemm_a8w8_bpreshuffle_gfx1250(
            XQ, WQ, x_scale, w_scale, Out, kernel_name
        )

    if kernel_name.startswith("flydsl_bpreshuffle_8w_"):
        from .flydsl.gemm_a8w8_bpreshuffle_8wave import run_gemm_a8w8_bpreshuffle_8wave

        return run_gemm_a8w8_bpreshuffle_8wave(
            XQ, WQ, x_scale, w_scale, Out, kernel_name
        )

    from .flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    parsed = _parse_flydsl_kernel_name(kernel_name)
    if parsed is None:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Out)
    tm, tn, tk, acp, wpe, xcd_swizzle, lds_stage, scheduler, k_split = parsed

    flydsl_preshuffle_gemm_a8(
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale,
        w_scale,
        Out,
        tm,
        tn,
        tk,
        acp,
        wpe,
        xcd_swizzle,
        lds_stage=lds_stage,
        enable_scheduler=str(scheduler).lower() != "off",
        split_k=k_split,
    )
    return Out


@functools.cache
def _warn_untuned_flydsl_fallback(gfx: str, op: str, n: int, k: int) -> None:
    """Warn once per (arch, op, N, K) that a shape has no tuned row."""
    logger.warning(
        f"[{gfx}] {op}: no tuned row for N={n}, K={k}; falling back to a "
        f"heuristic flydsl kernel. Tune this shape to remove the guess. "
        f"(logged once per N/K; AITER_LOG_TUNED_CONFIG=1 for per-call detail)"
    )


def gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    config: dict,
    a_is_preshuffled: bool = False,
) -> Tensor:
    kernel_name = str(config.get("kernelName", ""))
    if get_gfx() != "gfx1250":
        raise RuntimeError(
            "gemm_a8w8_mxfp8_128_bpreshuffle_flydsl is only supported on gfx1250"
        )
    from .flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250,
    )

    return run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        kernel_name,
        a_is_preshuffled=a_is_preshuffled,
    )


def gemm_a8w8_mxscale_preshuffle_flydsl(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    config: dict,
    w_scale_block: str | None = None,
) -> Tensor:
    """gfx950 FlyDSL MX-microscale preshuffle GEMM (CDNA4 scaled-MFMA).

    ``w_scale_block`` is the scale contract of the flat ``x_scale``/``w_scale``
    buffers, as the tuned table spells it ("128x128" or "1x32"); None reads it
    off their sizes.
    """
    kernel_name = str(config.get("kernelName", ""))
    if get_gfx() != "gfx950":
        raise RuntimeError(
            "gemm_a8w8_mxscale_preshuffle_flydsl is only supported on gfx950"
        )
    from .flydsl.mxscale_preshuffle_kernels import (
        run_gemm_a8w8_mxscale_preshuffle_gfx950,
    )

    return run_gemm_a8w8_mxscale_preshuffle_gfx950(
        XQ, WQ, x_scale, w_scale, Out, kernel_name, w_scale_block
    )


@compile_ops(
    "module_gemm_a8w8_asm",
    fc_name="gemm_a8w8_asm",
    ffi_type="ctypes",
)
def _gemm_a8w8_asm(
    XQ: Tensor,  # A:[M, K] i8
    WQ: Tensor,  # B:[N, K] i8 -> shuffle layout(32,16)
    x_scale: Tensor,  # A_scale:[M, 1] f32
    w_scale: Tensor,  # B_scale:[1, N] f32
    Out: Tensor,  # Out:[M, N] bf16
    kernelName: str | None = None,
    bias: Tensor | None = None,  # bias:[1, N] f32
    bpreshuffle: bool = True,
    splitK: int = -1,
) -> None: ...


def gemm_a8w8_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernelName: str = "",
    bias: Tensor | None = None,
    bpreshuffle: bool | None = True,
    splitK: int | None = None,
) -> Tensor:
    _gemm_a8w8_asm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        kernelName if kernelName else None,
        bias,
        bool(bpreshuffle) if bpreshuffle is not None else True,
        splitK if splitK is not None else -1,
    )
    return Out


def gen_gemm_a8w8_blockscale_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
) -> Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_blockscale",
    fc_name="gemm_a8w8_blockscale",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
    kernelName: str = "",
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_cktile",
    fc_name="gemm_a8w8_blockscale_cktile",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_cktile(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    isBpreshuffled: bool = False,
    splitK: int = 0,
    kernelName: str = "",
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle",
    fc_name="gemm_a8w8_blockscale_bpreshuffle",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_ck(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelName: str = "",
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_cktile",
    gen_fake=gen_gemm_a8w8_blockscale_ck_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_cktile(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    isBpreshuffled: bool = True,
    kernelName: str = "",
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_asm",
    fc_name="flatmm_a8w8_blockscale_asm",
    ffi_type="ctypes",
)
def _flatmm_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> None: ...
def flatmm_a8w8_blockscale_asm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
) -> Tensor:
    _flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, out)
    return out


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_asm",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_asm",
    ffi_type="ctypes",
)
def _gemm_a8w8_blockscale_bpreshuffle_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Tensor | None = None,
    splitK: int = -1,
    kernelName: str | None = None,
    bpreshuffle: int = 1,
    zero_bias_buf: Tensor | None = None,
) -> None: ...


# Ref on https://github.com/ROCm/aiter/blob/1be4ee9f70a7a7de5e9f57de2c0ecb9d13ed5983/aiter/ops/gemm_op_a16w16.py#L37-L57
@functools.lru_cache(maxsize=1024)
def get_zero_bias_buf_keyed(
    device: torch.device, stream_id: int, out_shape: int
) -> Tensor:
    with persistent_alloc(device):
        return torch.zeros(1, out_shape, dtype=torch.float32, device=device)


def get_zero_bias_buf(B: Tensor) -> Tensor:
    stream = torch.cuda.current_stream(B.device)
    return get_zero_bias_buf_keyed(B.device, stream.cuda_stream, B.shape[0])


def gemm_a8w8_blockscale_bpreshuffle_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool | None = True,
    zero_bias_buf: Tensor | None = None,
) -> Tensor:
    if bias is None and zero_bias_buf is None:
        zero_bias_buf = get_zero_bias_buf(B)
    _gemm_a8w8_blockscale_bpreshuffle_asm(
        A,
        B,
        out,
        A_scale,
        B_scale,
        bias,
        splitK if splitK is not None else -1,
        kernelName,
        int(bpreshuffle) if bpreshuffle is not None else 1,
        zero_bias_buf,
    )
    return out


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    return splitK


_CKGEMM_CONFIG_CACHE: dict = {}
_CKGEMM_HAS_GFX: dict = {}


@functools.lru_cache(maxsize=1024)
def get_CKGEMM_config(M: int, N: int, K: int, tuned_file=None):
    if tuned_file is None:
        tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
    if tuned_file not in _CKGEMM_CONFIG_CACHE:
        ckgemm_dict = pd.read_csv(f"{tuned_file}").drop_duplicates()
        # Use (gfx, cu_num, M, N, K) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, M, N, K) for old CSVs that pre-date the gfx column.
        if "gfx" in ckgemm_dict.columns:
            _CKGEMM_CONFIG_CACHE[tuned_file] = ckgemm_dict.set_index(
                ["gfx", "cu_num", "M", "N", "K"]
            ).to_dict("index")
            _CKGEMM_HAS_GFX[tuned_file] = True
        else:
            logger.warning(
                f"{tuned_file} has no 'gfx' column -- falling back to cu_num-only key. "
                "Re-run the tuner or migrate the CSV to add a gfx column."
            )
            _CKGEMM_CONFIG_CACHE[tuned_file] = ckgemm_dict.set_index(
                ["cu_num", "M", "N", "K"]
            ).to_dict("index")
            _CKGEMM_HAS_GFX[tuned_file] = False

    gfx = get_gfx()
    cu_num = get_cu_num()
    has_gfx = _CKGEMM_HAS_GFX[tuned_file]
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        key = (gfx, cu_num, padded_M, N, K) if has_gfx else (cu_num, padded_M, N, K)
        config = _CKGEMM_CONFIG_CACHE[tuned_file].get(key, None)
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {tuned_file} , kernel name is {config['kernelName']}!"
                )
            break
    if config is None:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, not found tuned config in {tuned_file}, will use default config!"
        )
    return config


_GEMM_QUANT_TYPE_CACHE: dict = {}
_GEMM_QUANT_TYPE_HAS_GFX: dict = {}


@functools.lru_cache(maxsize=1024)
def get_GEMM_config_with_quant_type(
    M: int,
    N: int,
    K: int,
    q_dtype_w: torch.dtype,
    tuned_file=None,
):
    if tuned_file is None:
        tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    # Load file if not cached
    if tuned_file not in _GEMM_QUANT_TYPE_CACHE:
        asmGemmDictDf = pd.read_csv(tuned_file).drop_duplicates()
        # Use (gfx, cu_num, M, N, K, q_dtype_w) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, M, N, K, q_dtype_w) for old CSVs that pre-date the gfx column.
        if "gfx" in asmGemmDictDf.columns:
            _GEMM_QUANT_TYPE_CACHE[tuned_file] = asmGemmDictDf.set_index(
                ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
            ).to_dict("index")
            _GEMM_QUANT_TYPE_HAS_GFX[tuned_file] = True
        else:
            logger.warning(
                f"{tuned_file} has no 'gfx' column -- falling back to cu_num-only key. "
                "Re-run the tuner or migrate the CSV to add a gfx column."
            )
            _GEMM_QUANT_TYPE_CACHE[tuned_file] = asmGemmDictDf.set_index(
                ["cu_num", "M", "N", "K", "q_dtype_w"]
            ).to_dict("index")
            _GEMM_QUANT_TYPE_HAS_GFX[tuned_file] = False

    gfx = get_gfx()
    cu_num = get_cu_num()
    has_gfx = _GEMM_QUANT_TYPE_HAS_GFX[tuned_file]
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        key = (
            (gfx, cu_num, padded_M, N, K, str(q_dtype_w))
            if has_gfx
            else (cu_num, padded_M, N, K, str(q_dtype_w))
        )
        config = _GEMM_QUANT_TYPE_CACHE[tuned_file].get(key, None)
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                msg = f"shape M:{M}, N:{N}, K:{K} q_dtype_w:{q_dtype_w}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned, in {tuned_file}!"
                if "libtype" in config:
                    msg += f" libtype is {config['libtype']}!"
                if "kernelName" in config:
                    msg += f" kernelName is {config['kernelName']} (kernelId {config.get('kernelId')})!"
                logger.info(msg)
            break
    if config is None:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, q_dtype_w:{q_dtype_w}, not found tuned config in {tuned_file}, will use default config!"
        )
    return config


def gemm_a8w8_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: int | None = None,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_fake)
def gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: int | None = None,
) -> Tensor:
    # assert dtype in [
    #     dtypes.bf16,
    #     dtypes.fp16,
    # ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    if not _ck_a8w8_supported():
        # RDNA (gfx11/gfx12): the CK/asm a8w8 kernel is unavailable; route to the
        # portable Triton kernel. Registered/faked via @torch_compile_guard above,
        # so callers stay torch.compile- and graph-capture-safe.
        from ..ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as gemm_a8w8_triton

        return gemm_a8w8_triton(XQ, WQ, x_scale, w_scale, bias, dtype=dtype)
    return gemm_a8w8_CK(XQ, WQ, x_scale, w_scale, bias, dtype, splitK)


def gemm_a8w8_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor,
    dtype=dtypes.bf16,
    check=False,
):
    """
    Notes for use gemm_a8w8_ASM:
    1. WQ(weight) must be shuffle, you can use \
        'weightshuffle = shuffle_weight(weight,layout=(32,16))'
    2. Use asm gemm must give bias, if not have bias, please give  \
        'bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')'
    """
    if check:
        assert dtype in [
            dtypes.bf16,
        ], f"Output {dtype=} is currently not supported in gemm_a8w8_ASM"
        assert (
            x_scale.dtype == dtypes.fp32 and w_scale.dtype == dtypes.fp32
        ), f"{x_scale.dtype=} or {w_scale.dtype=} must be dtypes.fp32"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    kernelName = ""
    if (
        x_scale.dtype == dtypes.fp32
        and w_scale.dtype == dtypes.fp32
        and (
            asm_config := get_GEMM_config_with_quant_type(
                m,
                n,
                k,
                dtypes.i8,
                AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
            )
        )
        is not None
    ):
        assert (
            bias is not None
        ), "Use asm gemm must give bias, please give a bias=torch.zeros(n,dtype=dtypes.fp32,device='cuda')"
        splitK = asm_config["splitK"]
        kernelName = asm_config["kernelName"]
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
        return gemm_a8w8_asm(
            XQ, WQ, x_scale, w_scale, Y, kernelName, bias, splitK=splitK
        )
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8_asm(XQ, WQ, x_scale, w_scale, Y, kernelName, bias, splitK=1)


def gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: int | None = None,
) -> Tensor:
    # assert dtype in [
    #     dtypes.bf16,
    #     dtypes.fp16,
    # ], f"Output {dtype=} is currently not supported in gemm_a8w8 CK"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]

    q_dtype_w = WQ.dtype if WQ.dtype in [dtypes.fp8, dtypes.i8] else dtypes.i8
    ck_config = get_GEMM_config_with_quant_type(
        m, n, k, q_dtype_w, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
    )
    if splitK is None:
        if ck_config is not None:
            splitK = ck_config["splitK"]
        else:
            splitK = 0
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    try:
        return gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Y, bias, splitK)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_CK failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, {splitK=}, config={ck_config}: {e}"
        ) from e


def gemm_a8w8_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    check: bool = False,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_bpreshuffle_fake)
def gemm_a8w8_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = dtypes.bf16,
    check: bool = False,
) -> Tensor:
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    w_k = WQ.shape[-1]
    if w_k < k:
        raise RuntimeError(
            f"gemm_a8w8_bpreshuffle requires WQ K >= XQ K, got WQ K={w_k}, " f"XQ K={k}"
        )

    # if (
    #     ck_config is None
    #     and dtype == dtypes.bf16
    #     and bias is not None
    #     and WQ.dtype != dtypes.i8
    # ):
    #     res = gemm_a8w8_ASM(XQ, WQ, x_scale, w_scale, bias, dtype=dtype, check=check)
    #     if res is not None:
    #         return res
    assert WQ.dtype == dtypes.fp8, "gemm_a8w8_bpreshuffle only support fp8 now"
    assert bias is None, "gemm_a8w8_bpreshuffle does not support bias now"
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)

    # CKTile only supports bf16 dtype
    config = get_GEMM_config_with_quant_type(
        m,
        n,
        k,
        dtypes.fp8,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    )
    if config is None and w_k > k:
        config = get_GEMM_config_with_quant_type(
            m,
            n,
            w_k,
            dtypes.fp8,
            AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
        )
    if config is not None:
        libtype = config["libtype"]
        splitK = int(config["splitK"])
        if libtype == "ck":
            return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y, splitK)
        elif libtype == "cktile":
            return gemm_a8w8_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Y, splitK)
        elif libtype == "flydsl":
            if w_k > k:
                XQ = F.pad(XQ.contiguous(), (0, w_k - k), value=0)
            return gemm_a8w8_bpreshuffle_flydsl(XQ, WQ, x_scale, w_scale, Y, config)

    if get_gfx() == "gfx1250":
        from ..ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
            kernel_fits_shape,
            kernels_list,
        )

        fits = [ki for ki in kernels_list.values() if kernel_fits_shape(ki, m, n, k)]
        if fits:
            want_tm = min(256, max(16, 1 << (m - 1).bit_length()))
            ki = min(
                fits, key=lambda x: (abs(x.tile_m - want_tm), -x.tile_n, -x.tile_k)
            )
            logger.warning(
                f"[gfx1250] gemm_a8w8_bpreshuffle untuned M={m}, N={n}, K={k}; "
                f"falling back to flydsl kernel '{ki.name}'."
            )
            if w_k > k:
                XQ = F.pad(XQ.contiguous(), (0, w_k - k), value=0)
            return gemm_a8w8_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, {"kernelName": ki.name}
            )
    try:
        if w_k > k:
            return gemm_a8w8_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Y, 0)
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y, 0)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


# M at or above which the triton kernel beats the untuned CK fallback, which
# runs one fixed tile shape at every M. Measured per arch; do not extrapolate.
_BLOCKSCALE_TRITON_FALLBACK_MIN_M = {"gfx942": 2048, "gfx950": 384}


def _blockscale_triton(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype,
    *,
    group32: bool = False,
    split_k: int | None = None,
) -> Tensor:
    """Run Triton on unshuffled weights with the selected scale format."""
    if group32:
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import (
            gemm_a8w8_blockscale_group32,
        )

        assert WQ.ndim == w_scale.ndim == 2, "Expected matrix weights and scales"
        group_n = 32 if w_scale.shape[0] == -(-WQ.shape[0] // 32) else 1
        return gemm_a8w8_blockscale_group32(
            XQ,
            WQ,
            x_scale,
            w_scale,
            dtype=dtype,
            weight_group_rows=group_n,
            split_k=split_k,
        )

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as _gemm_a8w8_blockscale_triton,
    )

    xq = XQ if XQ.dtype != torch.uint8 else XQ.view(dtypes.fp8)
    wq = WQ if WQ.dtype != torch.uint8 else WQ.view(dtypes.fp8)
    return _gemm_a8w8_blockscale_triton(xq, wq, x_scale, w_scale, dtype=dtype)


def gemm_a8w8_blockscale_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled=False,
    split_k: int | None = None,
) -> torch.Tensor:
    m = XQ.shape[0]
    n = WQ.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return Y


def _group32_w_scale_block(XQ, WQ, x_scale, w_scale) -> str | None:
    """ "1x32" or "32x32" for native E8M0 group32 operands (typed or uint8
    scales), else None."""
    if not (
        x_scale.dtype in (dtypes.fp8_e8m0, torch.uint8)
        and w_scale.dtype in (dtypes.fp8_e8m0, torch.uint8)
        and XQ.ndim == WQ.ndim == 2
        and x_scale.shape == (XQ.shape[0], XQ.shape[1] // 32)
    ):
        return None
    n, k = WQ.shape
    if w_scale.shape == (n, k // 32):
        return "1x32"
    if w_scale.shape == (-(-n // 32), k // 32):
        return "32x32"
    return None


_MXSCALE_BPRESHUFFLE_KEYS = ["gfx", "cu_num", "M", "N", "K", "w_scale_block"]

MXSCALE_BMM_KERNEL_ID = "bmm"


@functools.cache
def _load_mxscale_bpreshuffle_tuned(bmm: bool) -> dict:
    """{(gfx, cu_num, M, N, K, w_scale_block): row} of the e8m0 block-scale
    table for preshuffled weights, restricted to the batched-GEMM rows
    (``bmm``) or to the shuffled-scale 2D-GEMM rows."""
    path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
    df = pd.read_csv(path).drop_duplicates()
    df = with_mxscale_w_scale_block(df, path)
    is_bmm = (df["libtype"].astype(str) == "flydsl") & (
        df["kernelId"].astype(str) == MXSCALE_BMM_KERNEL_ID
    )
    return (
        df[is_bmm if bmm else ~is_bmm]
        .set_index(_MXSCALE_BPRESHUFFLE_KEYS)
        .to_dict("index")
    )


@functools.lru_cache(maxsize=1024)
def get_mxscale_bpreshuffle_config(
    m: int, n: int, k: int, w_scale_block: str, bmm: bool
):
    # bmm is passed positionally at every call site: lru_cache's _make_key takes
    # a slower path for keyword arguments (+37ns on every hit).
    """Tuned row of an e8m0 block-scale GEMM on preshuffled weights, at M or a
    padded M; None when untuned. ``bmm`` selects the operand contract: the
    batched GEMM run with B = 1, or the shuffled-scale 2D GEMM. Cached per
    shape, so a miss is reported once."""
    gfx, cu_num = get_gfx(), get_cu_num()
    kind = "bmm" if bmm else "2d"
    row, padded_m = find_padded_m_row(
        _load_mxscale_bpreshuffle_tuned(bmm),
        lambda pm: (gfx, cu_num, pm, n, k, w_scale_block),
        m,
        n,
        k,
    )
    if row is None:
        logger.warning(
            f"mxscale bpreshuffle ({kind}) M:{m}, N:{n}, K:{k}, w_scale "
            f"{w_scale_block} is untuned on {gfx}; a heuristic picks the kernel."
        )
    elif AITER_LOG_TUNED_CONFIG:
        logger.info(
            f"mxscale bpreshuffle ({kind}) M:{m}, N:{n}, K:{k}, w_scale "
            f"{w_scale_block} is tuned at padded_M {padded_m} on {gfx}: "
            f"{row['kernelName']}"
        )
    return row


# The blockscale x_scale block: a 1x128 x_scale has column-major bytes.
_BLOCKSCALE_X_BLOCK = 128


def _gemm_mxscale_bpreshuffle(XQ, WQ, x_scale, w_scale, Y):
    """E8M0 block-scale GEMM on (16, 16)-preshuffled weights. A 128-wide
    x_scale has column-major bytes (blockscale), a 32-wide one is row-major
    (group32/MX). Rows are selected by kernelId == "bmm": the flydsl batched
    GEMM (flydsl.batched_gemm_a8w8) run with B = 1, which also serves untuned
    shapes. The non-"bmm" rows at the same key belong to the shuffled-scale 2D
    GEMMs, which cannot read these raw scales."""
    m, k = XQ.shape
    n = WQ.shape[0]
    w_scale_block = mxscale_w_scale_block(tuple(w_scale.shape), n, k)
    config = get_mxscale_bpreshuffle_config(m, n, k, w_scale_block, True)  # bmm
    if config is not None and (config["libtype"], str(config["kernelId"])) != (
        "flydsl",
        MXSCALE_BMM_KERNEL_ID,
    ):
        raise NotImplementedError(
            f"mxscale bpreshuffle row {config['libtype']}/{config['kernelId']} "
            f"has no implementation"
        )
    from .flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

    run_bmm_a8w8_mxfp8(
        XQ.view(m, 1, k),
        WQ.view(1, n, k),
        x_scale.view(m, 1, x_scale.shape[-1]),
        w_scale.view(1, *w_scale.shape),
        Y.view(m, 1, n),
        kernel_name=None if config is None else config["kernelName"],
        x_scale_transposed=k // x_scale.shape[-1] == _BLOCKSCALE_X_BLOCK,
    )
    return Y


@torch_compile_guard(mutates_args=[], gen_fake=gemm_a8w8_blockscale_fake)
def gemm_a8w8_blockscale(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled: bool = False,
    split_k: int | None = None,
) -> torch.Tensor:
    """Blockscaled A8W8 GEMM with configuration-first backend dispatch.

    Native E8M0 group32 scales (typed tensors or uint8 views) use
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_GROUP32;
    FP32 128x128 scales use AITER_CONFIG_GEMM_A8W8_BLOCKSCALE. Both tables are
    queried through get_CKGEMM_config before choosing a fallback. Group32
    currently supports libtype="triton", also its default on a config miss.
    Triton tile and split-K parameters come from its own tuning tables.
    For native group32 operands, split_k optionally overrides the positive
    partition count without changing configured backend selection.
    Group32 with isBpreshuffled (weights shuffled (16, 16)) goes to
    gemm_a8w8_blockscale_bpreshuffle's E8M0 route and its
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE table.
    """
    is_group32 = _group32_w_scale_block(XQ, WQ, x_scale, w_scale) is not None
    # A malformed byte-scale layout must not fall through to FP32 CK dispatch.
    assert (
        is_group32 or x_scale.dtype == w_scale.dtype == dtypes.fp32
    ), "Expected E8M0 group32 scale shapes (typed or uint8), or FP32 128x128 scales"
    assert (
        split_k is None or is_group32
    ), "split_k override requires native group32 operands"
    assert dtype in (dtypes.bf16, dtypes.fp16) or (
        is_group32 and dtype == dtypes.fp32
    ), f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    if isBpreshuffled:
        if is_group32:
            assert (
                split_k is None
            ), "preshuffled group32 GEMM takes its split from the tuned table"
            e8m0 = dtypes.fp8_e8m0
            return gemm_a8w8_blockscale_bpreshuffle(
                XQ, WQ, x_scale.view(e8m0), w_scale.view(e8m0), dtype
            )
        if get_gfx() in ["gfx950"] and m >= 16 and k >= 512 and dtype == dtypes.bf16:
            Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
            return gfx950_a8w8_blockscale_ASM(XQ, WQ, x_scale, w_scale, Y)
        else:
            assert 0, "asm kernel only support B preshuffle and m >= 16"

    tuned_file = (
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_GROUP32_FILE
        if is_group32
        else AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
    )
    config = get_CKGEMM_config(m, n, k, tuned_file)
    if config is not None:
        libtype = config["libtype"]
        if libtype == "triton":
            return _blockscale_triton(
                XQ, WQ, x_scale, w_scale, dtype, group32=is_group32, split_k=split_k
            )
        # CK/CKTile currently consume FP32 128x128 scales. A misconfigured
        # group32 row must fail instead of reinterpreting its E8M0 bytes.
        assert not is_group32, f"Unsupported libtype {libtype} for group32 GEMM"
        splitK = int(config.get("splitK", 0))
        kernelName = str(config.get("kernelName", ""))
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
        if libtype == "ck":
            return gemm_a8w8_blockscale_ck(
                XQ,
                WQ,
                x_scale,
                w_scale,
                Y,
                splitK=splitK,
                kernelName=kernelName,
            )
        elif libtype == "cktile":
            return gemm_a8w8_blockscale_cktile(
                XQ,
                WQ,
                x_scale,
                w_scale,
                Y,
                splitK=splitK,
                kernelName=kernelName,
            )
        else:
            assert 0, f"Unsupported libtype {libtype} for gemm_a8w8_blockscale"

    if is_group32 or not _hip_blockscale_supported():
        return _blockscale_triton(
            XQ, WQ, x_scale, w_scale, dtype, group32=is_group32, split_k=split_k
        )
    min_m = _BLOCKSCALE_TRITON_FALLBACK_MIN_M.get(get_gfx())
    if min_m is not None and m >= min_m:
        return _blockscale_triton(XQ, WQ, x_scale, w_scale, dtype)
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    try:
        return gemm_a8w8_blockscale_ck(XQ, WQ, x_scale, w_scale, Y)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_blockscale failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


def flatmm_a8w8_blockscale_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype=dtypes.fp16,
):
    assert dtype in [
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    # k = XQ.shape[-1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return flatmm_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, Y)


@functools.lru_cache(maxsize=1024)
def _flydsl_mxfp8_fallback_kernel(
    m: int, n: int, k: int, a_preshuffle: bool = False, mx32: bool = False
):
    """Best-effort gfx1250 mxfp8 kernel for a shape with no tuned row.

    ``mx32``: the kernel runs on 1x32 scales, which take one n32k4 super-row
    per 32 columns, so tile_n must be a multiple of 32.
    """
    from ..ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        is_compute_kernel,
        kernel_fits_shape,
        kernels_list,
    )

    # A-preshuffle packs adjacent row pairs, so an odd M has no valid pairing.
    if a_preshuffle and m % 2:
        return None
    fits = [
        ki
        for ki in kernels_list.values()
        if kernel_fits_shape(ki, m, n, k)
        and not (mx32 and (ki.a_preshuffle or ki.tile_n % 32))
    ]
    if not fits:
        return None
    want_tm = min(256, max(16, 1 << (m - 1).bit_length()))
    compute = [ki for ki in fits if is_compute_kernel(ki)]
    if compute:
        return min(
            compute,
            key=lambda x: (
                abs(x.tile_m - want_tm),
                -(x.cluster_m * x.cluster_n),
                x.split_k,
                x.persistent_n_tiles,
                -x.tile_n,
                -x.num_buffers,
            ),
        )
    return min(fits, key=lambda x: (abs(x.tile_m - want_tm), -x.tile_n, -x.tile_k))


def gemm_a8w8_blockscale_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Tensor | None = None,
) -> Tensor:
    # Must mirror the real signature (incl. out=): the abstract_impl forwards the
    # out= kwarg here at trace time. When out is given the real fn returns it, so
    # the fake returns it too (preserves tensor identity for mutation/aliasing).
    if out is not None:
        return out
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_bpreshuffle_fake)
def gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Tensor | None = None,
) -> Tensor:
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    # `out`: optional caller-owned output buffer so the result lands at a FIXED
    # address (needed to capture this GEMM's consumer into a cudagraph without a
    # per-step input copy). The ck/cktile/asm/opus/flydsl paths below take Y
    # positionally and write into it, so honoring `out` there is free. The
    # triton-fallback branches self-allocate their output and CANNOT honor `out`
    # -> assert loud rather than silently returning a different address.
    if out is not None:
        assert out.shape == (m, n) and out.dtype == dtype and out.device == XQ.device, (
            f"gemm_a8w8_blockscale_bpreshuffle: out buffer {tuple(out.shape)}/"
            f"{out.dtype} != expected ({m},{n})/{dtype}"
        )
        Y = out
    else:
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)

    use_gfx1250_flydsl_or_triton_mxfp8_128 = (
        get_gfx() == "gfx1250"
        and x_scale.dtype == dtypes.fp8_e8m0
        and w_scale.dtype == dtypes.fp8_e8m0
    )
    if use_gfx1250_flydsl_or_triton_mxfp8_128:
        config = get_mxscale_bpreshuffle_config(
            m, n, k, mxscale_w_scale_block(tuple(w_scale.shape), n, k), False  # not bmm
        )
        # A tuned triton/gluon row wins over flydsl on the SAME mxfp8_128
        # operands: gemm_afp8wfp8_preshuffle consumes the e8m0 scales natively
        # (wmma_scaled on gluon / dot_scaled on triton), so unlike the fp32
        # blockscale path further down there is no scale widening, and the (N, K)
        # shuffled weight + column-major x_scale layout is a direct fit.
        # Deliberately checked BEFORE the FlyDSL availability gate so a
        # triton-tuned shape does not require FlyDSL to be installed.
        if config is not None and config["libtype"] == "triton":
            from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
                gemm_afp8wfp8_preshuffle as _gemm_afp8wfp8_preshuffle_triton,
            )

            # Same convention as the fp32 blockscale triton branch below:
            # kernelName optionally carries the backend hint ("triton"/"gluon"),
            # anything else -> None (auto gluon->triton detection).
            kernelName = str(config.get("kernelName", ""))
            backend = kernelName if kernelName in ("triton", "gluon") else None
            return _gemm_afp8wfp8_preshuffle_triton(
                XQ,
                WQ,
                # wmma_scaled/dot_scaled take uint8-typed scale operands; e8m0 is
                # bit-identical, so a view is the whole conversion.
                x_scale.view(torch.uint8),
                w_scale.view(torch.uint8),
                dtype=dtype,
                y=Y,  # honor caller out= (zero-copy); Y = out or fresh empty
                x_scale_group_size=128,
                # The mxfp8_128 contract, not a guess: x_scale bytes are
                # column-major (K // 128, M) -- what per_group_quant_hip(
                # transpose_scale=True) emits, and what the flydsl runner
                # hard-codes as x_scale_transposed=True. It is NOT inferable from
                # strides here: that buffer is a contiguous (M, K // 128) tensor
                # whose *bytes* are transposed, so the stride(0) != 1 probe the
                # fp32 branches use would read it as row-major.
                is_x_scale_transposed=True,
                backend=backend,
            )
        if config is not None and config["libtype"] == "flydsl":
            return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, config
            )

        ki = _flydsl_mxfp8_fallback_kernel(m, n, k)
        if ki is not None:
            logger.warning(
                f"[gfx1250] gemm_a8w8_blockscale_bpreshuffle untuned "
                f"M={m}, N={n}, K={k}; falling back to flydsl kernel '{ki.name}'."
            )
            return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, {"kernelName": ki.name}
            )

    # gfx950 e8m0 has two implementations; the scale layout tells them apart.
    #   1-D flat -> the caller pre-shuffled with shuffle_scale_blockscale_a/_b
    #               (128x128) or shuffle_scale_a16w4(., 1, False).flatten()
    #               (1x32, A rows padded to 32 first), layouts only the mxpsh
    #               2D GEMM reads. Which of the two is read off their size.
    #   2-D raw  -> the batched-B=1 path, which also covers group32 (1x32).
    if (
        get_gfx() == "gfx950"
        and x_scale.dtype == dtypes.fp8_e8m0
        and w_scale.dtype == dtypes.fp8_e8m0
    ):
        if x_scale.dim() == 1 and w_scale.dim() == 1:
            from ..ops.flydsl.mxscale_preshuffle_kernels import (
                _heuristic_tile,
                mxpsh_w_scale_block,
            )

            w_scale_block = mxpsh_w_scale_block(m, n, k, x_scale, w_scale)
            config = get_mxscale_bpreshuffle_config(
                m, n, k, w_scale_block, False  # not bmm
            )
            if config is not None and config["libtype"] == "flydsl":
                return gemm_a8w8_mxscale_preshuffle_flydsl(
                    XQ, WQ, x_scale, w_scale, Y, config, w_scale_block
                )

            ki = _heuristic_tile("fp8", "fp8", m, n, k, w_scale_block)
            if ki is None:
                # Cannot fall through: the scales are the MX kernel's shuffled
                # flat buffers, which the fp32 backends below would read as
                # garbage.
                from ..ops.flydsl.gemm_tune import (
                    flydsl_gemm_mxscale_preshuffle_common as _mxpsh_cat,
                )

                b_rows = _mxpsh_cat.W_SCALE_BLOCKS[w_scale_block].b_rows
                raise RuntimeError(
                    f"gemm_a8w8_blockscale_bpreshuffle: no legal gfx950 MX tile "
                    f"for M={m}, N={n}, K={k} at w_scale_block {w_scale_block} "
                    f"(needs N%{b_rows}==0 and K%128==0)"
                )
            _warn_untuned_flydsl_fallback(
                "gfx950", "gemm_a8w8_blockscale_bpreshuffle", n, k
            )
            return gemm_a8w8_mxscale_preshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, {"kernelName": ki.name}, w_scale_block
            )
        return _gemm_mxscale_bpreshuffle(XQ, WQ, x_scale, w_scale, Y)

    # temporarily guard scale that are not fp32.
    if x_scale.dtype == dtypes.fp8_e8m0:
        x_scale = x_scale.to(dtypes.fp32)
    if w_scale.dtype == dtypes.fp8_e8m0:
        w_scale = w_scale.to(dtypes.fp32)

    if not _hip_blockscale_supported():
        # No CK code object for this arch -> triton preshuffle. WQ is already
        # (16,16)-shuffled (the only blockscale layout) == triton's (N//16, K*16)
        # view; x_scale is column-major. Direct fit.
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle as _gemm_a8w8_blockscale_preshuffle_triton,
        )

        xq = XQ if XQ.dtype != torch.uint8 else XQ.view(dtypes.fp8)
        wq = WQ if WQ.dtype != torch.uint8 else WQ.view(dtypes.fp8)
        # Explicit config (no PRESHUFFLED tuning file on main yet); mirrors the
        # gfx1201 non-preshuffle M_LEQ_8 default.
        _fallback_cfg = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 8,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
            "kpack": 2,
        }
        # triton impl accepts a pre-allocated `y=` -> forward `out` (zero-copy).
        return _gemm_a8w8_blockscale_preshuffle_triton(
            xq,
            wq.reshape(n // 16, k * 16),
            x_scale,
            w_scale,
            dtype=dtype,
            y=Y,  # honor caller out= (zero-copy); Y = out or fresh empty
            config=_fallback_cfg,
            is_x_scale_tranposed=x_scale.stride(0) != 1,
        )

    config = get_CKGEMM_config(
        m,
        n,
        k,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE,
    )
    # Triton path first: it allocates its own output, so skip the Y buffer the
    # ck/asm paths below need.
    if (config is not None and config["libtype"] == "triton") or get_gfx() == "gfx1250":
        # kernelName optionally carries the backend hint ("triton"/"gluon");
        # anything else -> None (auto gluon->triton detection). config=None lets
        # the triton impl load its own tuned config internally. WQ is already
        # (16,16)-shuffled == triton's (N//16, K*16) view; x_scale is
        # column-major -> direct fit.
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle as _gemm_a8w8_blockscale_preshuffle_triton,
        )

        kernelName = str(config.get("kernelName", "")) if config is not None else ""
        backend = kernelName if kernelName in ("triton", "gluon") else None
        xq = XQ if XQ.dtype != torch.uint8 else XQ.view(dtypes.fp8)
        wq = WQ if WQ.dtype != torch.uint8 else WQ.view(dtypes.fp8)
        # The triton impl accepts a pre-allocated output via `y=`; forward `out`
        # so the result lands at the caller's fixed address (zero-copy path).
        return _gemm_a8w8_blockscale_preshuffle_triton(
            xq,
            wq.reshape(n // 16, k * 16),
            x_scale,
            w_scale,
            dtype=dtype,
            y=Y,  # honor caller out= (zero-copy); Y = out or fresh empty
            backend=backend,
            is_x_scale_tranposed=x_scale.stride(0) != 1,
        )
    if config is not None:
        libtype = config["libtype"]
        kernelName = str(config.get("kernelName", ""))
        if libtype == "cktile":
            return gemm_a8w8_blockscale_bpreshuffle_cktile(
                XQ, WQ, x_scale, w_scale, Y, kernelName=kernelName
            )
        elif libtype == "ck":
            return gemm_a8w8_blockscale_bpreshuffle_ck(
                XQ, WQ, x_scale, w_scale, Y, kernelName=kernelName
            )
        elif libtype == "asm":
            splitK = config["splitK"]
            return gemm_a8w8_blockscale_bpreshuffle_asm(
                XQ, WQ, Y, x_scale, w_scale, splitK=splitK, kernelName=kernelName
            )
        elif libtype == "opus":
            kernelId = int(config["kernelId"])
            from aiter.ops.opus import opus_gemm

            return opus_gemm(
                XQ,
                WQ,
                Y,
                kid=kernelId,
                layout="bpreshuffle",
                x_scale=x_scale,
                w_scale=w_scale,
            )
        elif libtype == "flydsl":
            return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
                XQ, WQ, x_scale, w_scale, Y, config
            )
    try:
        return gemm_a8w8_blockscale_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_blockscale_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


def _abpreshuffle_config_from_bpreshuffle(m: int, n: int, k: int, w_scale) -> dict:
    """Fall back to the bpreshuffle winner for a shape with no A-preshuffle row.

    The two kernel families differ only by an ``_apre`` marker, which sits before
    any ``_ps<n>`` persistent-tile suffix. Those bpreshuffle rows are the e8m0
    2D-GEMM rows of the mxscale table -- the very ones the non-A-preshuffled
    gfx1250 path reads -- not the fp32 blockscale-bpreshuffle table.
    """
    from .flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import is_compute_wmma_kernel_name

    config = get_mxscale_bpreshuffle_config(
        m, n, k, mxscale_w_scale_block(tuple(w_scale.shape), n, k), False  # not bmm
    )
    borrowed = None
    if config is not None and config.get("libtype") == "flydsl":
        name = str(config["kernelName"])
        head, sep, tail = name.partition("_ps")
        borrowed = dict(config, kernelName=head + "_apre" + sep + tail)
        if is_compute_wmma_kernel_name(borrowed["kernelName"]):
            return borrowed
    ki = _flydsl_mxfp8_fallback_kernel(m, n, k, a_preshuffle=True)
    if ki is not None and (
        borrowed is None or is_compute_wmma_kernel_name(ki.name_for(True))
    ):
        logger.warning(
            f"[gfx1250] gemm_a8w8_blockscale_abpreshuffle untuned "
            f"M={m}, N={n}, K={k}; falling back to flydsl kernel "
            f"'{ki.name_for(True)}'."
        )
        return {"kernelName": ki.name_for(True), "libtype": "flydsl"}
    if borrowed is not None:
        logger.info(
            f"[gfx1250] gemm_a8w8_blockscale_abpreshuffle untuned "
            f"M={m}, N={n}, K={k}; no compute-bound kernel fits, borrowing the "
            f"generic bpreshuffle winner '{borrowed['kernelName']}'."
        )
        return borrowed
    raise RuntimeError(
        f"gemm_a8w8_blockscale_abpreshuffle: no FlyDSL config for M={m}, N={n}, K={k}"
    )


def gemm_a8w8_blockscale_abpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Tensor | None = None,
) -> Tensor:
    # A is padded to an even row count, so XQ.shape[0] is M+1 for odd M.
    if out is not None:
        return out
    return torch.empty(x_scale.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_abpreshuffle_fake)
def gemm_a8w8_blockscale_abpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Tensor | None = None,
) -> Tensor:
    """Blockscale GEMM taking an A-preshuffled activation.

    ``XQ`` must already be laid out by :func:`aiter.ops.shuffle.shuffle_mxfp8fp4_a`
    and ``WQ`` by :func:`shuffle_weight`; everything else matches
    :func:`gemm_a8w8_blockscale_bpreshuffle`.  Only the gfx1250 mxfp8_128 FlyDSL
    path implements it, so an unsupported operand set raises rather than silently
    computing a row-major result.

    Odd M: ``shuffle_mxfp8fp4_a`` packs adjacent A row pairs, so the caller must
    pad A to ``M + 1`` rows before shuffling (the kernel reads the last pair
    whole).  A is the ONLY operand that is padded -- ``x_scale`` stays ``(M,
    K//128)`` and the result stays ``(M, N)``, both with the true, odd M, which
    is read from ``x_scale``.  The padded A row is loaded but never contributes:
    its C row, its A row bound and its A-scale super are all clamped to M::

        m_pad = m + (m & 1)
        a = torch.zeros((m_pad, k), dtype=x.dtype, device=x.device)
        a[:m] = x
        y = gemm_a8w8_blockscale_abpreshuffle(
            shuffle_mxfp8fp4_a(a), wq, x_scale, w_scale)   # y is (m, n)
    """
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = x_scale.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    # The kernel reads A as row-major [rows, K] off lda=stride(0); a non-unit
    # inner stride would be read as if it were packed and silently miscompute.
    if XQ.stride(1) != 1 or not WQ.is_contiguous():
        raise RuntimeError(
            "gemm_a8w8_blockscale_abpreshuffle: XQ rows must be contiguous and WQ "
            f"fully contiguous, got XQ.stride={tuple(XQ.stride())}, "
            f"WQ.stride={tuple(WQ.stride())}"
        )
    if XQ.shape[0] != m + (m & 1):
        raise RuntimeError(
            f"gemm_a8w8_blockscale_abpreshuffle: x_scale gives M={m}, so the "
            f"preshuffled A must have {m + (m & 1)} rows, got {XQ.shape[0]}. "
            "shuffle_mxfp8fp4_a packs adjacent row pairs, so an odd M must be "
            "padded to M+1 rows before shuffling."
        )
    if out is not None:
        assert out.shape == (m, n) and out.dtype == dtype and out.device == XQ.device, (
            f"gemm_a8w8_blockscale_abpreshuffle: out buffer {tuple(out.shape)}/"
            f"{out.dtype} != expected ({m},{n})/{dtype}"
        )
        Y = out
    else:
        Y = torch.empty(m, n, dtype=dtype, device=XQ.device)

    if not (
        get_gfx() == "gfx1250"
        and x_scale.dtype == dtypes.fp8_e8m0
        and w_scale.dtype == dtypes.fp8_e8m0
    ):
        raise RuntimeError(
            "gemm_a8w8_blockscale_abpreshuffle needs gfx1250 with e8m0 scales, got "
            f"gfx={get_gfx()}, {x_scale.dtype=}, {w_scale.dtype=}"
        )

    try:
        config = get_CKGEMM_config(
            m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_ABPRESHUFFLE_FILE
        )
    except FileNotFoundError:
        config = None
    if config is None or config.get("libtype") != "flydsl":
        config = _abpreshuffle_config_from_bpreshuffle(m, n, k, w_scale)
    return gemm_a8w8_mxfp8_128_bpreshuffle_flydsl(
        XQ, WQ, x_scale, w_scale, Y, config, a_is_preshuffled=True
    )


def gfx950_a8w8_blockscale_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    dtype=dtypes.bf16,
):
    assert dtype in [
        dtypes.bf16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    return gfx950_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, Y)  # noqa: F821


def gen_gemm_a8w8_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_tune",
    fc_name="gemm_a8w8_tune",
    gen_fake=gen_gemm_a8w8_tune_fake_tensors,
)
def gemm_a8w8_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


def gen_gemm_a8w8_blockscale_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a8w8_blockscale_tune",
    fc_name="gemm_a8w8_blockscale_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_cktile_tune",
    fc_name="gemm_a8w8_blockscale_cktile_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_cktile_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
    preshuffleB: bool = False,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_bpreshuffle_tune",
    fc_name="gemm_a8w8_bpreshuffle_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_bpreshuffle_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
    preshuffleB: bool = True,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_blockscale_bpreshuffle_tune",
    fc_name="gemm_a8w8_blockscale_bpreshuffle_tune",
    gen_fake=gen_gemm_a8w8_blockscale_tune_fake_tensors,
)
def gemm_a8w8_blockscale_bpreshuffle_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


@compile_ops(
    "module_gemm_a8w8_bpreshuffle_cktile_tune",
    fc_name="gemm_a8w8_bpreshuffle_cktile_tune",
)
def gemm_a8w8_bpreshuffle_cktile_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...


# ---------------------------------------------------------------------------
# gfx1250 MXFP8 x MXFP8 ASM GEMM with e8m0 block scales.
# CSV selects kernel/split; misses use the .cu heuristic.
# ---------------------------------------------------------------------------
@compile_ops(
    "module_mxfp8fp4gemm_asm",
    fc_name="mxfp8_mxfp8_gemm_asm",
    ffi_type="ctypes",
)
def _mxfp8_mxfp8_gemm_asm(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
    B: Tensor,  # B:[N, K]   mxfp8 e4m3 (always preshuffled)
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0 (shuffled)
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0 (shuffled)
    out: Tensor,  # BF16 [M, N], or [splitk, M, N] for partial outputs
    kernelName: str | None = None,
    a_preshuffle: int = 1,
    splitk: int = 1,
) -> None:
    """Write compact BF16 [splitk,M,N] partials, or [M,N] for splitk=1.

    out must be contiguous; only its first splitk*M*N elements are written.
    Storage offsets are supported; padded row/plane strides are not.
    """


@compile_ops(
    "module_mxfp8fp4gemm_asm",
    fc_name="mxfp8fp4_gemm_validate",
    ffi_type="ctypes",
)
def _mxfp8fp4_gemm_validate(
    A: Tensor,
    B: Tensor,
    kernelName: str | None,
    b_intype: str,
    a_preshuffle: int,
    splitk: int,
) -> None:
    """Validate the native kernel/count selection before allocating partials."""


def _reduce_mxfp8_partials(partials: Tensor, out: Tensor | None = None) -> Tensor:
    """Reduce compact BF16 [splitk, M, N] partials from either ASM GEMM.

    ``out``: optional contiguous BF16 [M, N] destination.
    """
    import flydsl.expr as fx

    from .flydsl.kernels.gemm_a8w8_splitk_reduce_gfx1250 import (
        compile_gemm_a8w8_splitk_reduce,
    )
    from .flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    splitk, M, N = partials.shape
    if out is None:
        out = torch.empty((M, N), dtype=partials.dtype, device=partials.device)
    _run_compiled(
        compile_gemm_a8w8_splitk_reduce(split_k=splitk, out_dtype_str="bf16"),
        ptr_arg(partials),
        ptr_arg(out),
        M * N,
        1,
        N,
        partials.stride(0) * partials.element_size(),
        fx.Stream(torch.cuda.current_stream(partials.device)),
    )
    return out


_MXFP8_GEMM_CONFIG_CACHE: dict = {}
_MXFP8_128_MIN_K = 8 * 128  # K128/PF8 prologue


def _mxfp8_config_int(value, name, minimum=1):
    """Parse a saved integer without truncation or overflow at the C++ ABI."""
    number = float(value)
    if (
        isinstance(value, bool)
        or not math.isfinite(number)
        or not number.is_integer()
        or not minimum <= number <= (1 << 31) - 1
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, 2147483647]")
    return int(number)


def _load_mxfp8_gemm_configs(tuned_file):
    try:
        table = pd.read_csv(tuned_file).drop_duplicates()
        missing = {*_MXFP8_GEMM_CONFIG_KEYS, "kernelName", "splitK"} - set(
            table.columns
        )
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        logger.warning(
            "Ignoring MXFP8 GEMM tuned CSV %r; using default dispatch: %s",
            tuned_file,
            exc,
        )
        return {}

    configs, duplicates = {}, set()
    for row in table.to_dict("records"):
        try:
            for name in ("gfx", "b_intype", "outdtype", "kernelName"):
                value = row[name]
                if not isinstance(value, str) or value.strip() in ("", "0"):
                    raise ValueError(f"{name} must be a nonempty string")
            if not row["gfx"].startswith("gfx"):
                raise ValueError(f"invalid gfx {row['gfx']!r}")
            if row["b_intype"] not in ("mxfp8", "mxfp4"):
                raise ValueError(f"invalid b_intype {row['b_intype']!r}")
            if row["outdtype"] not in (
                "torch.bfloat16",
                "torch.float16",
                "torch.float32",
            ):
                raise ValueError(f"invalid outdtype {row['outdtype']!r}")
            for name in ("M", "N", "K"):
                row[name] = _mxfp8_config_int(row[name], name)
            row["a_preshuffle"] = _mxfp8_config_int(
                row["a_preshuffle"], "a_preshuffle", minimum=0
            )
            if row["a_preshuffle"] not in (0, 1):
                raise ValueError("a_preshuffle must be 0 or 1")
            key = tuple(row[name] for name in _MXFP8_GEMM_CONFIG_KEYS)
            if key in configs:
                duplicates.add(key)
            configs[key] = row
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning("Skipping invalid MXFP8 GEMM row in %r: %s", tuned_file, exc)
    for key in duplicates:
        del configs[key]
        logger.warning(
            "Ignoring conflicting MXFP8 GEMM rows for %s in %r; using default dispatch",
            key,
            tuned_file,
        )
    groups = {}
    for (gfx, M, N, K, b_intype, apre, dtype), row in configs.items():
        groups.setdefault((gfx, b_intype, apre, dtype), {})[(M, N, K)] = row
    return groups


@functools.lru_cache(maxsize=8)
def _mxfp8_kernel_configs(asm_dir):
    manifest = Path(asm_dir) / "mxfp8fp4gemm" / "mxfp8fp4gemm.csv"
    return pd.read_csv(manifest).set_index("knl_name").to_dict("index")


def _normalize_mxfp8_splitk(splitk):
    """Accept integer-like counts without truncating floats at the C++ ABI."""
    message = "splitk must be a positive C++ int"
    if isinstance(splitk, bool):
        raise TypeError(message)
    try:
        splitk = operator.index(splitk)
    except TypeError as exc:
        raise TypeError(message) from exc
    if not 1 <= splitk <= (1 << 31) - 1:
        raise ValueError(message)
    return splitk


def _validate_mxfp8_splitk(M, N, K, splitk, kernel=None):
    """Match native splitk_is_valid; omit kernel-specific checks if kernel=None."""
    context = f"splitk={splitk} for M={M}, N={N}, K={K}"
    try:
        splitk = _normalize_mxfp8_splitk(splitk)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{context}: {exc}") from exc
    if splitk == 1:
        return
    if splitk & (splitk - 1):
        raise ValueError(f"{context}: splitk must be a power of two")
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"{context}: M, N and K must be positive")
    if K % splitk:
        raise ValueError(f"{context}: splitk must divide K")
    part_k = K // splitk
    if part_k % 128:
        raise ValueError(f"{context}: K/splitk={part_k} must be a multiple of 128")
    if kernel is None:
        return
    tile_m, tile_n = int(kernel["tile_m"]), int(kernel["tile_n"])
    cluster_x, cluster_y = int(kernel["cluster_x"]), int(kernel["cluster_y"])
    b_intype = kernel["b_intype"]
    supports_splitk = (
        kernel["outtype"] == "bf16"
        and cluster_x == 4
        and b_intype in ("mxfp8", "mxfp4")
        and (
            (tile_m, tile_n) == (256, 256)
            and (cluster_y == 4 or (cluster_y == 2 and b_intype == "mxfp8"))
            or (tile_m, tile_n, cluster_y) == (64, 512, 1)
        )
    )
    label = f"{b_intype} {tile_m}x{tile_n}_{cluster_x}x{cluster_y}"
    if not supports_splitk:
        raise ValueError(f"{context}: {label} requires splitk=1")
    min_k = 768 if tile_m == 64 and b_intype == "mxfp4" else 512
    if part_k < min_k:
        raise ValueError(f"{context}: K/splitk={part_k} must be >= {min_k} for {label}")
    tiles = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    if tiles * splitk > 256:
        raise ValueError(
            f"{context}: {label} needs splitk * tiles = {splitk} * {tiles} = "
            f"{tiles * splitk} work units, exceeding 256"
        )


def _validate_mxfp8_tuned_config(
    config, M, N, K, a_preshuffle, dtype, b_intype, *, gfx
):
    """Check the saved launch before allocating partials; match the native guards."""
    splitk = _mxfp8_config_int(config["splitK"], "splitK")
    asm_dir = get_mxfp8_asm_dir(gfx)
    kernel = _mxfp8_kernel_configs(asm_dir)[config["kernelName"]]
    if (
        dtype != dtypes.bf16
        or kernel["outtype"] != "bf16"
        or kernel["b_intype"] != b_intype
        or kernel["a_preshuffle"] != int(bool(a_preshuffle))
    ):
        raise ValueError("kernel does not match b_intype/a_preshuffle/outdtype")
    if not (Path(asm_dir) / "mxfp8fp4gemm" / kernel["co_name"]).is_file():
        raise ValueError(f"kernel code object is missing: {kernel['co_name']}")
    if M <= 0 or N <= 0 or K <= 0 or N % 16 or K % 128 or (a_preshuffle and M % 2):
        raise ValueError("shape does not satisfy the kernel alignment requirements")
    tile_m, tile_n = int(kernel["tile_m"]), int(kernel["tile_n"])
    cluster_x, cluster_y = int(kernel["cluster_x"]), int(kernel["cluster_y"])
    if (tile_m, tile_n) == (128, 128):
        # Both A layouts require full clusters for safe page-table prefetches.
        m_align, n_align = tile_m * cluster_y, tile_n * cluster_x
        if M % m_align or N % n_align or K < _MXFP8_128_MIN_K:
            raise ValueError(
                f"128x128 requires complete clusters: M%{m_align}==0, "
                f"N%{n_align}==0 and K>={_MXFP8_128_MIN_K}"
            )
    _validate_mxfp8_splitk(M, N, K, splitk, kernel)
    return dict(config, splitK=splitk)


@functools.lru_cache(maxsize=1024)
def get_mxfp8_gemm_config(
    M,
    N,
    K,
    a_preshuffle,
    dtype=dtypes.bf16,
    tuned_file=None,
    *,
    b_intype="mxfp8",
    splitk=None,
):
    """Try exact/fine/coarse M keys without padding inputs.

    Validate saved/actual shapes and explicit splits; skip unusable rows.
    Cross-file collisions follow the shared resolve-and-rerun policy.
    """
    if splitk is not None:
        splitk = _normalize_mxfp8_splitk(splitk)
        _validate_mxfp8_splitk(M, N, K, splitk)
    if tuned_file is None:
        try:
            tuned_file = get_mxfp8_config_file()
        except (OSError, UnicodeError, ValueError) as exc:
            # Preserve the shared merger's RuntimeError/rerun protocol.
            logger.warning(
                "Ignoring MXFP8 GEMM tuned config resolution; using default dispatch: %s",
                exc,
            )
            return None
    if tuned_file not in _MXFP8_GEMM_CONFIG_CACHE:
        _MXFP8_GEMM_CONFIG_CACHE[tuned_file] = _load_mxfp8_gemm_configs(tuned_file)
    gfx = get_gfx()
    configs = _MXFP8_GEMM_CONFIG_CACHE[tuned_file].get(
        (gfx, b_intype, int(bool(a_preshuffle)), str(dtype)), {}
    )
    tried_m = set()
    for gl in (None, 0, 1):
        if not configs:
            break
        try:
            lookup_m = M if gl is None else get_padded_m(M, N, K, gl)
        except (OSError, RuntimeError, OverflowError) as exc:
            logger.warning(
                "Cannot resolve MXFP8 GEMM padded M for M:%s N:%s K:%s; "
                "using default dispatch: %s",
                M,
                N,
                K,
                exc,
            )
            break
        if lookup_m in tried_m:
            continue
        tried_m.add(lookup_m)
        config = configs.get((lookup_m, N, K))
        if config is None:
            continue
        try:
            # Validate the saved shape even when the actual M is smaller.
            config = _validate_mxfp8_tuned_config(
                config, lookup_m, N, K, a_preshuffle, dtype, b_intype, gfx=gfx
            )
            if lookup_m != M:
                config = _validate_mxfp8_tuned_config(
                    config, M, N, K, a_preshuffle, dtype, b_intype, gfx=gfx
                )
            if splitk is not None:
                config = _validate_mxfp8_tuned_config(
                    dict(config, splitK=splitk),
                    M,
                    N,
                    K,
                    a_preshuffle,
                    dtype,
                    b_intype,
                    gfx=gfx,
                )
        except (
            OSError,
            UnicodeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            logger.warning(
                "Ignoring unusable MXFP8 GEMM tuned row in %r with lookup M:%s "
                "for actual M:%s N:%s K:%s b_intype:%s a_preshuffle:%s "
                "(kernelName=%r, splitK=%r); trying remaining candidates "
                "before default dispatch: %s",
                tuned_file,
                lookup_m,
                M,
                N,
                K,
                b_intype,
                a_preshuffle,
                config.get("kernelName"),
                config.get("splitK"),
                exc,
            )
            continue
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"shape is M:{M}, N:{N}, K:{K}, b_intype:{b_intype}, "
                f"a_preshuffle:{a_preshuffle}, found lookup M:{lookup_m} in "
                f"{tuned_file}, kernel name is {config['kernelName']}, "
                f"splitK is {config['splitK']}!"
            )
        return config
    logger.info(
        f"shape is M:{M}, N:{N}, K:{K}, b_intype:{b_intype}, a_preshuffle:{a_preshuffle}, "
        f"not found usable exact/padded-M config in {tuned_file}, will use default config!"
    )
    return None


def _resolve_mxfp8_gemm_config(
    M,
    N,
    K,
    a_preshuffle,
    dtype=dtypes.bf16,
    kernelName="",
    splitk=None,
    *,
    b_intype="mxfp8",
):
    # Explicit kernels bypass tuning; an explicit split count overrides the CSV.
    if splitk is not None:
        splitk = _normalize_mxfp8_splitk(splitk)
        _validate_mxfp8_splitk(M, N, K, splitk)
    if not kernelName:
        config = get_mxfp8_gemm_config(
            M, N, K, a_preshuffle, dtype, b_intype=b_intype, splitk=splitk
        )
        if config is not None:
            kernelName = config["kernelName"]
            splitk = config["splitK"]
    return kernelName, 1 if splitk is None else splitk


def _gemm_a8w8_mxfp8_fake(
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,
    ScaleB: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    a_preshuffle: bool = True,
    kernelName: str = "",
    splitk: int | None = None,
) -> Tensor:
    return torch.empty((A.shape[0], B.shape[0]), dtype=dtype, device=A.device)


@mxfp8_compile_guard(mutates_args=[], gen_fake=_gemm_a8w8_mxfp8_fake)
def gemm_a8w8_mxfp8(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3
    B: Tensor,  # B:[N, K]   mxfp8 e4m3
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,
    a_preshuffle: bool = True,
    kernelName: str = "",
    splitk: int | None = None,
) -> Tensor:
    """gfx1250 MXFP8 x MXFP8 GEMM. Return BF16 [M,N] with e8m0 block scales.

    CSV lookup tries exact/padded M without padding inputs; unusable rows fall
    back to the native heuristic. Explicit kernels bypass CSV; kernel and split
    arguments override tuning. Without a saved or explicit split, use splitk=1.
    Split counts are literal powers of two, validated before partial allocation.
    For splitk>1, K/splitk is a multiple of 128 and >=512, with splitk*tiles <=256.
    The 128x128 kernels require splitk=1, K>=1024, and positive M/N multiples
    of 512 for both A layouts. BF16 partials use FP32 FlyDSL reduction, so their
    rounding can differ from splitk=1. Cross-file collisions require a rerun.
    """
    require_gfx1250_asm("gemm_a8w8_mxfp8")
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    if dtype != dtypes.bf16:
        raise NotImplementedError(
            f"gfx1250 a8w8 MXFP8 GEMM: unsupported output dtype {dtype}"
        )
    if K % 128 != 0:  # A (m/2,k/128) preshuffle
        raise NotImplementedError(
            f"gfx1250 a8w8 MXFP8 GEMM requires K%128==0, got K={K}"
        )
    if N % 16 != 0:  # B 16x16 preshuffle
        raise NotImplementedError(
            f"gfx1250 a8w8 MXFP8 GEMM requires N%16==0, got N={N}"
        )
    if a_preshuffle and M % 2 != 0:  # A (m/2,k/128) preshuffle
        raise NotImplementedError(
            f"gfx1250 a8w8 MXFP8 GEMM a_preshuffle requires M%2==0, got M={M}"
        )
    kernelName, splitk = _resolve_mxfp8_gemm_config(
        M, N, K, a_preshuffle, dtype, kernelName, splitk
    )
    if splitk > 1:
        _mxfp8fp4_gemm_validate(
            A, B, kernelName or None, "mxfp8", int(bool(a_preshuffle)), splitk
        )
    out = torch.empty(
        (splitk, M, N) if splitk > 1 else (M, N), dtype=dtype, device=A.device
    )
    _mxfp8_mxfp8_gemm_asm(
        A,
        B,
        ScaleA,
        ScaleB,
        out,
        kernelName if kernelName else None,
        int(bool(a_preshuffle)),
        splitk,
    )
    return _reduce_mxfp8_partials(out) if splitk > 1 else out


# ---------------------------------------------------------------------------
# gfx1250 MXFP8 (1x32 e8m0) bpreshuffle GEMM.
# One operand contract for every backend: row-major FP8 A with its m32k4
# scale (pad32(M), K/32), 16x16-preshuffled FP8 B with its n32k4 scale
# (N, K/32) -- the shuffle_mxfp8fp4_scale layout. Its tuned CSV routes each
# (M, N, K) to the ASM or the FlyDSL mxfp8_32 kernel; callers see neither.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _mxfp8_bpreshuffle_tuned_nk(tuned_file: str) -> frozenset:
    try:
        table = pd.read_csv(tuned_file)
    except (OSError, pd.errors.EmptyDataError):
        return frozenset()  # no tuned shapes on this install
    return frozenset(zip(table["gfx"], table["cu_num"], table["N"], table["K"]))


def mxfp8_bpreshuffle_tuned(N: int, K: int) -> bool:
    """Whether this arch has tuned MXFP8 1x32 GEMM configs for (N, K)."""
    if get_gfx() != "gfx1250":
        return False
    tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE
    return (get_gfx(), get_cu_num(), N, K) in _mxfp8_bpreshuffle_tuned_nk(tuned_file)


def _mxfp8_32_fallback_kernel_name(M: int, N: int, K: int) -> str | None:
    """The FlyDSL heuristic kernel for this shape, as its mxfp8_32 name."""
    ki = _flydsl_mxfp8_fallback_kernel(M, N, K, mx32=True)
    if ki is None:
        return None
    from .flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        COMPUTE_WMMA_NAME_PREFIX,
        MX32_COMPUTE_WMMA_NAME_PREFIX,
        MX32_WMMA_NAME_PREFIX,
        WMMA_NAME_PREFIX,
    )

    for mx128, mx32 in (
        (COMPUTE_WMMA_NAME_PREFIX, MX32_COMPUTE_WMMA_NAME_PREFIX),
        (WMMA_NAME_PREFIX, MX32_WMMA_NAME_PREFIX),
    ):
        if ki.name.startswith(mx128 + "_"):
            return mx32 + ki.name[len(mx128) :]
    return None


@functools.lru_cache(maxsize=1024)
def _get_mxfp8_bpreshuffle_config(M: int, N: int, K: int):
    """(libtype, kernelName, splitK) serving this shape.

    A tuned row wins. An ASM row that does not fit this M (a padded-M row serves
    smaller M too) falls back to the ASM kernel's own heuristic; an untuned shape
    to the FlyDSL heuristic, else the ASM one. kernelName None = ASM heuristic.
    """
    config = get_CKGEMM_config(
        M, N, K, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE
    )
    if config is not None:
        libtype, kernel_name = config["libtype"], str(config["kernelName"])
        if libtype == "flydsl":
            return "flydsl", kernel_name, 1
        if libtype == "asm":
            try:
                config = _validate_mxfp8_tuned_config(
                    config, M, N, K, False, dtypes.bf16, "mxfp8", gfx=get_gfx()
                )
                return "asm", kernel_name, int(config["splitK"])
            except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
                logger.warning(
                    f"[gfx1250] gemm_a8w8_mxfp8_bpreshuffle: ASM row {kernel_name!r} "
                    f"does not fit M={M}, N={N}, K={K} ({exc}); using the ASM "
                    "heuristic."
                )
                return "asm", None, 1
        logger.warning(
            f"[gfx1250] gemm_a8w8_mxfp8_bpreshuffle: ignoring {libtype} row "
            f"{kernel_name!r} for M={M}, N={N}, K={K}"
        )
    name = _mxfp8_32_fallback_kernel_name(M, N, K)
    if name is not None:
        logger.warning(
            f"[gfx1250] gemm_a8w8_mxfp8_bpreshuffle untuned M={M}, N={N}, K={K}; "
            f"falling back to flydsl kernel '{name}'."
        )
        return "flydsl", name, 1
    logger.warning(
        f"[gfx1250] gemm_a8w8_mxfp8_bpreshuffle untuned M={M}, N={N}, K={K}; "
        "falling back to the ASM heuristic."
    )
    return "asm", None, 1


def gemm_a8w8_mxfp8_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Tensor | None = None,
) -> Tensor:
    if out is not None:
        return out
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_mxfp8_bpreshuffle_fake)
def gemm_a8w8_mxfp8_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    out: Tensor | None = None,
) -> Tensor:
    """gfx1250 FP8 GEMM with 1x32 e8m0 scales; returns ``out`` or a new [M, N].

    XQ: [M, K] FP8 row-major. WQ: [N, K] FP8, 16x16 preshuffled.
    x_scale: [pad32(M), K/32] e8m0, m32k4. w_scale: [N, K/32] e8m0, n32k4.
    The tuned CSV picks the ASM or FlyDSL kernel per (M, N, K); a shape or M
    it does not cover runs a heuristic kernel (see _get_mxfp8_bpreshuffle_config).
    """
    M, K = XQ.shape
    N = WQ.shape[0]
    Y = torch.empty(M, N, dtype=dtype, device=XQ.device) if out is None else out
    if M == 0:
        return Y
    libtype, kernel_name, splitk = _get_mxfp8_bpreshuffle_config(M, N, K)
    if libtype == "asm" and (dtype != dtypes.bf16 or not Y.is_contiguous()):
        # The ASM kernels write a compact BF16 [M, N] only.
        fallback = _mxfp8_32_fallback_kernel_name(M, N, K)
        if fallback is None:
            raise RuntimeError(
                f"gemm_a8w8_mxfp8_bpreshuffle: no kernel for M={M}, N={N}, K={K} "
                f"with dtype={dtype} and out strides {tuple(Y.stride())}"
            )
        libtype, kernel_name = "flydsl", fallback
    if libtype == "flydsl":
        from .flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
            run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
        )

        return run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
            XQ, WQ, x_scale, w_scale, Y, kernel_name
        )
    partials = (
        Y if splitk == 1 else torch.empty(splitk, M, N, dtype=dtype, device=Y.device)
    )
    _mxfp8_mxfp8_gemm_asm(XQ, WQ, x_scale, w_scale, partials, kernel_name, 0, splitk)
    if splitk > 1:
        _reduce_mxfp8_partials(partials, out=Y)
    return Y
