# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
from typing import Optional

import pandas as pd
import torch
from aiter import logger
from torch import Tensor
from torch.library import Library

from ..jit.core import (
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
    compile_ops,
)
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx
from ..jit.utils.torch_guard import torch_compile_guard
from ..ops.gemm_op_common import get_padded_m
from ..utility import dtypes
from ..ops.flydsl.utils import is_flydsl_available

aiter_lib = Library("aiter", "FRAGMENT")


def gen_gemm_a8w8_ck_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
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
    bias: Optional[torch.Tensor] = None,
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
    """Parse tile config from flydsl kernelName.
    Returns ``(tile_m, tile_n, tile_k, lds_stage, cshuffle, async_copy,
    waves_per_eu, xcd_swizzle)`` or None on parse failure.
    """
    import re

    m = re.match(
        r"flydsl_bpreshuflle_(\d+)x(\d+)x(\d+)_\w+_\w+_\w+_"
        r"(\d+)x(\d+)x(\d+)x(\d+)x(\d+)",
        kernel_name,
    )
    if m is None:
        return None
    return tuple(int(m.group(i)) for i in range(1, 9))


def gemm_a8w8_bpreshuffle_flydsl(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    config: dict,
) -> Tensor:
    from .flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    kernel_name = config.get("kernelName", "")
    parsed = _parse_flydsl_kernel_name(str(kernel_name))
    if parsed is None:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Out)
    tm, tn, tk, lds, csh, acp, wpe, xcd = parsed

    flydsl_preshuffle_gemm_a8(
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale,
        w_scale,
        Out,
        tm,
        tn,
        tk,
        lds,
        csh,
        acp,
        wpe,
        xcd,
    )
    return Out


def _parse_flydsl_fp8_kernel_name(kernel_name: str):
    """Parse tile config from a FlyDSL FP8 row-scale GEMM kernelName.

    Format:
      4-wave: ``flydsl_fp8_4w_<tm>x<tn>_<xcd|noxcd>``
      8-wave: ``flydsl_fp8_8w_<tm>x<tn>``

    Returns ``(wave_variant, tile_m, tile_n, xcd_remap)`` or ``None`` on
    parse failure. ``xcd_remap`` is always 1 for 8-wave. B is always
    delivered to the kernel in the ``preshuffle_b`` layout.
    """
    import re

    m8 = re.match(r"^flydsl_fp8_8w_(\d+)x(\d+)$", kernel_name)
    if m8 is not None:
        return (8, int(m8.group(1)), int(m8.group(2)), 1)

    m4 = re.match(r"^flydsl_fp8_4w_(\d+)x(\d+)_(xcd|noxcd)$", kernel_name)
    if m4 is not None:
        tm, tn, xcd = m4.group(1), m4.group(2), m4.group(3)
        return (4, int(tm), int(tn), 1 if xcd == "xcd" else 0)

    return None


def _unshuffle_weight_16x16_fp8(w: Tensor) -> Tensor:
    """Inverse of ``shuffle_weight(layout=(16, 16))`` for fp8 (1 byte/elem).

    ``shuffle_weight`` reshapes (N, K) to (1, N/16, 16, K/32, 2, 16), permutes
    (0, 1, 3, 4, 2, 5), and views back as (N, K). The inverse re-views the
    shuffled storage as (1, N/16, K/32, 2, 16, 16) and applies permutation
    (0, 1, 4, 2, 3, 5).
    """
    N, K_total = w.shape
    assert N % 16 == 0 and K_total % 32 == 0, (
        f"fp8 (16,16) un-shuffle needs N%16==0 and K%32==0, got N={N}, K={K_total}"
    )
    x = w.view(1, N // 16, K_total // 32, 2, 16, 16)
    x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
    return x.view(N, K_total)


# Per-WQ-identity cache of the (shuffle_weight((16,16)) -> row-major ->
# preshuffle_b) round-trip. The transform is ~35 us for an N=7168 K=2048 FP8
# weight on gfx950 and the result is consumed unmodified by the FlyDSL FP8
# row-scale 4-wave / 8-wave kernels, so a per-tensor memoization eliminates the
# repeated cost in the typical inference setting where ``WQ`` is a stable
# ``nn.Module`` parameter.
#
# Key:   (data_ptr, shape, dtype-str) -- light, hashable, and stable while the
#        underlying storage stays alive.
# Value: (preshuffled_tensor, untyped_storage_ref) -- the held storage ref
#        (i) keeps the allocator from re-using the data_ptr behind our back, and
#        (ii) lets us cheaply re-check the storage identity on each lookup.
#
# Eviction is FIFO (dict insertion-order) with a max bounded by
# ``AITER_FLYDSL_FP8_B_CACHE_SIZE`` (default 64). The cached tensor itself is
# roughly N*K bytes (FP8 = 1 byte/elem), so 64 * 14 MB ~= 900 MB worst-case.
_FLYDSL_FP8_B_CACHE: "dict" = {}


def _flydsl_fp8_b_cache_max() -> int:
    raw = os.environ.get("AITER_FLYDSL_FP8_B_CACHE_SIZE")
    if not raw:
        return 64
    try:
        return max(1, int(raw))
    except ValueError:
        return 64


def _get_flydsl_fp8_b(WQ: Tensor) -> Tensor:
    """Return the FlyDSL ``preshuffle_b`` view of ``WQ``, cached per identity.

    On hit the round-trip ``_unshuffle_weight_16x16_fp8 -> preshuffle_b`` is
    skipped entirely; the held storage reference both prevents allocator reuse
    of the same ``data_ptr`` for an unrelated tensor and is double-checked on
    every lookup.
    """
    from .flydsl.kernels.fp8_gemm_utils import preshuffle_b

    key = (WQ.data_ptr(), tuple(WQ.shape), str(WQ.dtype))
    storage = WQ.untyped_storage()
    hit = _FLYDSL_FP8_B_CACHE.get(key)
    if hit is not None and hit[1].data_ptr() == storage.data_ptr():
        return hit[0]

    weight = _unshuffle_weight_16x16_fp8(WQ.contiguous())
    weight = preshuffle_b(weight.contiguous()).contiguous()

    cap = _flydsl_fp8_b_cache_max()
    while len(_FLYDSL_FP8_B_CACHE) >= cap:
        _FLYDSL_FP8_B_CACHE.pop(next(iter(_FLYDSL_FP8_B_CACHE)))
    _FLYDSL_FP8_B_CACHE[key] = (weight, storage)
    return weight


def gemm_a8w8_rowscale_flydsl(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    config: dict,
) -> Tensor:
    """Dispatch FP8 row-scale GEMM through a tuned FlyDSL 4-wave / 8-wave kernel.

    ``WQ`` from the ``gemm_a8w8_bpreshuffle`` entry point is pre-shuffled via
    ``shuffle_weight(layout=(16,16))``. The row-scale kernel always consumes
    the ``preshuffle_b`` layout, so reconcile by inverse-shuffling back to
    row-major and then applying ``preshuffle_b``. The per-WQ-identity cache in
    ``_get_flydsl_fp8_b`` memoizes the result so the round-trip only runs on
    the first call for each weight tensor.

    Falls back to the CK preshuffle path on parse failure so callers always
    get a result.
    """
    from .flydsl.gemm_kernels import flydsl_fp8_gemm_rowscale

    kernel_name = config.get("kernelName", "")
    parsed = _parse_flydsl_fp8_kernel_name(str(kernel_name))
    if parsed is None:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Out)
    wave_variant, tm, tn, xcd_remap = parsed

    weight = _get_flydsl_fp8_b(WQ)

    x_scale_v = x_scale.squeeze().contiguous()
    w_scale_v = w_scale.squeeze().contiguous()

    flydsl_fp8_gemm_rowscale(
        XQ.contiguous(),
        weight,
        x_scale_v,
        w_scale_v,
        Out,
        tile_m=tm,
        tile_n=tn,
        wave_variant=wave_variant,
        b_preshuffled=1,
        xcd_remap=xcd_remap,
    )
    return Out


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
    kernelName: Optional[str] = None,
    bias: Optional[Tensor] = None,  # bias:[1, N] f32
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
    bias: Optional[Tensor] = None,
    bpreshuffle: Optional[bool] = True,
    splitK: Optional[int] = None,
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
    bias: Optional[Tensor] = None,
    splitK: int = -1,
    kernelName: Optional[str] = None,
    bpreshuffle: int = 1,
    zero_bias_buf: Optional[Tensor] = None,
) -> None: ...


def gemm_a8w8_blockscale_bpreshuffle_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: Optional[bool] = True,
    zero_bias_buf: Optional[Tensor] = None,
) -> Tensor:
    if bias is None and zero_bias_buf is None:
        zero_bias_buf = torch.zeros(1, B.shape[0], dtype=torch.float32, device=A.device)
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
                f"{tuned_file} has no 'gfx' column — falling back to cu_num-only key. "
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
                f"{tuned_file} has no 'gfx' column — falling back to cu_num-only key. "
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
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: Optional[int] = None,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_fake)
def gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: Optional[int] = None,
) -> Tensor:
    # assert dtype in [
    #     dtypes.bf16,
    #     dtypes.fp16,
    # ], f"Output {dtype=} is currently not supported in gemm_a8w8"
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
    bias: Optional[Tensor] = None,
    dtype: torch.dtype = dtypes.bf16,
    splitK: Optional[int] = None,
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
    bias: Optional[Tensor] = None,
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
    bias: Optional[Tensor] = None,
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
    if config is not None:
        libtype = config["libtype"]
        splitK = int(config["splitK"])
        if libtype == "ck":
            return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y, splitK)
        elif libtype == "cktile":
            return gemm_a8w8_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Y, splitK)
        elif libtype == "flydsl" and is_flydsl_available():
            return gemm_a8w8_bpreshuffle_flydsl(XQ, WQ, x_scale, w_scale, Y, config)
        elif libtype == "flydsl_fp8" and is_flydsl_available():
            return gemm_a8w8_rowscale_flydsl(XQ, WQ, x_scale, w_scale, Y, config)
    try:
        return gemm_a8w8_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y, 0)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


def gemm_a8w8_blockscale_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled=False,
) -> torch.Tensor:
    m = XQ.shape[0]
    n = WQ.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return Y


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_fake)
def gemm_a8w8_blockscale(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
    isBpreshuffled: bool = False,
) -> torch.Tensor:
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    if isBpreshuffled:
        if get_gfx() in ["gfx950"] and m >= 16 and k >= 512 and dtype == dtypes.bf16:
            return gfx950_a8w8_blockscale_ASM(XQ, WQ, x_scale, w_scale, Y)
        else:
            assert 0, "asm kernel only support B preshuffle and m >= 16"
    else:
        config = get_CKGEMM_config(
            m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
        )
        if config is not None:
            libtype = config["libtype"]
            splitK = int(config.get("splitK", 0))
            kernelName = str(config.get("kernelName", ""))
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


def gemm_a8w8_blockscale_bpreshuffle_fake(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    return torch.empty(XQ.shape[0], WQ.shape[0], dtype=dtype, device=XQ.device)


@torch_compile_guard(gen_fake=gemm_a8w8_blockscale_bpreshuffle_fake)
def gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    assert dtype in [
        dtypes.bf16,
        dtypes.fp16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[1]
    config = get_CKGEMM_config(
        m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
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
    try:
        return gemm_a8w8_blockscale_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)
    except RuntimeError as e:
        raise RuntimeError(
            f"gemm_a8w8_blockscale_bpreshuffle failed for shape M={m}, N={n}, K={k}, "
            f"{dtype=}, config={config}: {e}"
        ) from e


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
    return gfx950_a8w8_blockscale_asm(XQ, WQ, x_scale, w_scale, Y)


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
