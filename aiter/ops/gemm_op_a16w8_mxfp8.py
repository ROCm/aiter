# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# GEMM with BF16 activations and MXFP8 weights for decode on gfx942: e4m3
# bytes with one UE8M0 exponent per 32 weights along K (the OCP MX layout),
# BF16 output, M from 1 to 64. ASM kernels in hsa/gfx942/a16w8gemm/, loader
# csrc/py_itfs_cu/asm_gemm_a16w8_mxfp8.cu.

import csv
import functools
import os
from typing import Optional

import torch
from torch import Tensor

from .. import logger
from ..jit.core import (
    AITER_CONFIGS,
    AITER_LOG_TUNED_CONFIG,
    AITER_META_DIR,
    compile_ops,
)
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes
from ..utility.graph_alloc import persistent_alloc

__all__ = [
    "clear_gemm_a16w8_mxfp8_config_cache",
    "gemm_a16w8_mxfp8_asm",
    "gemm_a16w8_mxfp8_prepare_weight",
    "get_gemm_a16w8_mxfp8_config",
    "is_gemm_a16w8_mxfp8_tuned",
]

_MX_GROUP = 32
_MX_MAX_M = 64
# Split-K scratch per (device, stream): fp32 partial tiles and arrival counters.
# The tuned kernels need at most 12.6 MB and 6060 counters (N = 32320, K = 5120).
_MX_WORKSPACE_BYTES = 16 << 20
_MX_COUNTERS = 1 << 16
_MX_REGISTRY = "hsa/gfx942/a16w8gemm/a16w8gemm_mxfp8.csv"
_MX_TUNED_COLUMNS = ("gfx", "cu_num", "M", "N", "K", "kernelName", "splitK")


@compile_ops(
    "module_gemm_a16w8_mxfp8_asm",
    fc_name="gemm_a16w8_mxfp8_asm",
    ffi_type="ctypes",
)
def _gemm_a16w8_mxfp8_asm(
    A: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Tensor,
    workspace: Tensor,
    counters: Tensor,
    kernelName: str,
    splitK: int,
) -> None: ...


def gemm_a16w8_mxfp8_prepare_weight(
    weight: Tensor, weight_scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Convert OCP MXFP8 weights to the layout that gemm_a16w8_mxfp8_asm reads.

    weight: [N, K] torch.float8_e4m3fn (or its uint8 bytes), K % 32 == 0.
    weight_scale: [N, K / 32] UE8M0 exponents (torch.float8_e8m0fnu or uint8).

    gfx942 converts fp8 as e4m3fnuz. The same bits have half the e4m3fn value,
    and 0x80, which is -0 in e4m3fn, is NaN. The result therefore maps 0x80 to
    0x00 and adds 1 to every exponent; both steps are exact. Weights that are
    NaN in e4m3fn (0x7F, 0xFF) and exponents of 254 or 255 (2^127, NaN) have no
    exact equivalent and raise. Returns (B, B_scale): B is
    torch.float8_e4m3fnuz [N, K] and B_scale uint8 [N, K / 32], both contiguous.
    The conversion runs once, at weight load time.
    """
    if weight.dim() != 2 or weight.element_size() != 1:
        raise RuntimeError(
            "gemm_a16w8_mxfp8_prepare_weight: weight must be a 2-D fp8 tensor"
        )
    n, k = weight.shape
    if k % _MX_GROUP or tuple(weight_scale.shape) != (n, k // _MX_GROUP):
        raise RuntimeError(
            "gemm_a16w8_mxfp8_prepare_weight: weight_scale must be "
            f"[{n}, {k // _MX_GROUP}], got {tuple(weight_scale.shape)}"
        )
    if weight_scale.element_size() != 1:
        raise RuntimeError(
            "gemm_a16w8_mxfp8_prepare_weight: weight_scale must hold 1-byte "
            "UE8M0 exponents (torch.float8_e8m0fnu or uint8)"
        )
    w = weight.contiguous().view(torch.uint8)
    if bool(((w & 0x7F) == 0x7F).any()):
        raise RuntimeError(
            "gemm_a16w8_mxfp8_prepare_weight: weight holds e4m3fn NaN (0x7F/0xFF)"
        )
    s = weight_scale.contiguous().view(torch.uint8)
    if bool((s >= 254).any()):
        raise RuntimeError(
            "gemm_a16w8_mxfp8_prepare_weight: weight_scale holds an exponent of "
            "254 or 255 (2^127 or NaN), which the gfx942 layout cannot represent"
        )
    w = torch.where(w == 0x80, torch.zeros_like(w), w)
    return w.view(torch.float8_e4m3fnuz).contiguous(), (s + 1).contiguous()


@functools.lru_cache(maxsize=1)
def _mx_registry() -> dict[str, dict[str, int]]:
    """Kernel name -> symbol and tile parameters, keyed by the symbol and by the
    code object name without .co."""
    path = os.path.join(AITER_META_DIR, _MX_REGISTRY)
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            params = {c: int(row[c]) for c in ("tr", "tn", "nw", "sk", "threads")}
            params["knl_name"] = row["knl_name"]
            out[row["knl_name"]] = params
            out[row["co_name"].removesuffix(".co")] = params
    return out


@functools.lru_cache(maxsize=4)
def _mx_tuned_table(path: str) -> dict[tuple, list[tuple[int, str, int]]]:
    """(gfx, cu_num, N, K) -> [(M, kernelName, splitK)] sorted by M.

    A missing file or column raises: the path comes from
    AITER_CONFIGS.AITER_CONFIG_GEMM_A16W8_MXFP8_ASM_FILE, so a wrong
    AITER_CONFIG_GEMM_A16W8_MXFP8_ASM must not silently disable the table.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"gemm_a16w8_mxfp8_asm: tuned config {path} does not exist "
            "(check AITER_CONFIG_GEMM_A16W8_MXFP8_ASM)"
        )
    table: dict[tuple, list[tuple[int, str, int]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(_MX_TUNED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        for row in reader:
            key = (row["gfx"], int(row["cu_num"]), int(row["N"]), int(row["K"]))
            table.setdefault(key, []).append(
                (int(row["M"]), row["kernelName"], int(row["splitK"]))
            )
    for v in table.values():
        v.sort()
    return table


@functools.lru_cache(maxsize=1024)
def _get_gemm_a16w8_mxfp8_config_cached(
    M: int, N: int, K: int, tuned_file: str, gfx: str, cu_num: int
) -> tuple[str, int] | None:
    for m, name, splitk in _mx_tuned_table(tuned_file).get((gfx, cu_num, N, K), ()):
        if M <= m:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    "A16W8 MXFP8 shape M:%s N:%s K:%s uses the M:%s entry of %s: "
                    "%s splitK %s",
                    M,
                    N,
                    K,
                    m,
                    tuned_file,
                    name,
                    splitk,
                )
            return name, splitk
    if AITER_LOG_TUNED_CONFIG:
        logger.info(
            "A16W8 MXFP8 shape M:%s N:%s K:%s has no tuned config for %s/%s in %s; "
            "using the default kernel",
            M,
            N,
            K,
            gfx,
            cu_num,
            tuned_file,
        )
    return None


def get_gemm_a16w8_mxfp8_config(
    M: int, N: int, K: int
) -> Optional[tuple[str, int]]:  # noqa: UP045
    """The tuned (kernelName, splitK) for M rows of an N x K weight on this GPU.

    Rows are bucketed: the entry with the smallest tuned M >= M is used. None
    when the shape has no tuned entry for this GPU (gfx and CU count), or M > 64.
    The table is AITER_CONFIGS.AITER_CONFIG_GEMM_A16W8_MXFP8_ASM_FILE: by
    default aiter/configs/a16w8_mxfp8_asm_tuned_gemm.csv merged with the
    model files aiter/configs/model_configs/*a16w8_mxfp8_asm_tuned_gemm*.csv.
    """
    if not 0 < M <= _MX_MAX_M:
        return None
    tuned_file = os.path.abspath(AITER_CONFIGS.AITER_CONFIG_GEMM_A16W8_MXFP8_ASM_FILE)
    return _get_gemm_a16w8_mxfp8_config_cached(
        M, N, K, tuned_file, get_gfx(), get_cu_num()
    )


def clear_gemm_a16w8_mxfp8_config_cache() -> None:
    """Drop the cached tuned table, for example after a tuner rewrote the CSV."""
    _mx_tuned_table.cache_clear()
    _get_gemm_a16w8_mxfp8_config_cached.cache_clear()


def is_gemm_a16w8_mxfp8_tuned(M: int, N: int, K: int) -> bool:
    """True when gemm_a16w8_mxfp8_asm has a tuned kernel for this shape on this GPU.

    Engines should gate on this: an untuned shape still computes correctly with
    a default kernel, but is not timed against the BF16 path it would replace.
    """
    return get_gemm_a16w8_mxfp8_config(M, N, K) is not None


def _mx_default_kernel(M: int, N: int) -> str:
    tr = (M + 15) // 16
    name = f"a16w8gemm_bf16_mxfp8_tr{tr}_tn1_nw4"
    if N % 16 or name not in _mx_registry():
        raise RuntimeError(
            f"gemm_a16w8_mxfp8_asm: no kernel for M={M} N={N} "
            "(needs 1 <= M <= 64 and N % 16 == 0)"
        )
    return _mx_registry()[name]["knl_name"]


@functools.lru_cache(maxsize=64)
def _mx_workspace_keyed(device: torch.device, stream_id: int) -> tuple[Tensor, Tensor]:
    with persistent_alloc(device):
        ws = torch.empty(_MX_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        cnt = torch.zeros(_MX_COUNTERS, dtype=torch.int32, device=device)
    return ws, cnt


def _mx_workspace(device: torch.device) -> tuple[Tensor, Tensor]:
    """Split-K partials and arrival counters for the current (device, stream).

    The last split of each tile resets its counter, so the counters stay zero
    between calls. Launches on different streams must not share counters, so
    each stream gets its own pair.
    """
    stream = torch.cuda.current_stream(device)
    return _mx_workspace_keyed(device, stream.cuda_stream)


def gemm_a16w8_mxfp8_asm(
    A: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Optional[Tensor] = None,  # noqa: UP045
    kernelName: Optional[str] = None,  # noqa: UP045
    splitK: Optional[int] = None,  # noqa: UP045
) -> Tensor:
    """out = A @ dequant(B, B_scale).T with BF16 activations and MXFP8 weights, on gfx942.

    A: [M, K] bf16, contiguous, M from 1 to 64.
    B: [N, K] torch.float8_e4m3fnuz and B_scale: [N, K / 32] uint8, both from
       gemm_a16w8_mxfp8_prepare_weight. K % 32 == 0 and N % 16 == 0.
    out: [M, N] bf16, contiguous. A new tensor is returned when out is None.

    The weights are dequantized in registers, exactly, and the products are
    summed in fp32, so the result is A @ W.T rounded to BF16 up to the fp32
    summation order. The activations are not quantized.

    The kernel comes from the tuned table (AITER_CONFIG_GEMM_A16W8_MXFP8_ASM,
    see get_gemm_a16w8_mxfp8_config) when it has the shape; otherwise a default
    kernel for M rows is used. is_gemm_a16w8_mxfp8_tuned() tells which.
    kernelName / splitK override the choice.

    Works under torch.compile (fullgraph) and in CUDA graphs: the launch is the
    custom op aiter::gemm_a16w8_mxfp8_asm_out.
    """
    if B.dtype != torch.float8_e4m3fnuz or B_scale.dtype != torch.uint8:
        raise RuntimeError(
            "gemm_a16w8_mxfp8_asm: B must be torch.float8_e4m3fnuz and B_scale "
            f"uint8 (got {B.dtype} and {B_scale.dtype}); convert the checkpoint "
            "once with gemm_a16w8_mxfp8_prepare_weight (gfx942 reads fp8 as "
            "e4m3fnuz)"
        )
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError(
            f"gemm_a16w8_mxfp8_asm: A and B must be 2-D, got {A.dim()}-D and {B.dim()}-D"
        )
    M, N = A.shape[0], B.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=dtypes.bf16, device=A.device)
    if M == 0:
        return out
    gemm_a16w8_mxfp8_asm_out(A, B, B_scale, out, kernelName, splitK)
    return out


def _gemm_a16w8_mxfp8_asm_out_fake(
    A: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Tensor,
    kernelName: Optional[str] = None,  # noqa: UP045
    splitK: Optional[int] = None,  # noqa: UP045
) -> None:
    return None


@torch_compile_guard(mutates_args=["out"], gen_fake=_gemm_a16w8_mxfp8_asm_out_fake)
def gemm_a16w8_mxfp8_asm_out(
    A: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Tensor,
    kernelName: Optional[str] = None,  # noqa: UP045
    splitK: Optional[int] = None,  # noqa: UP045
) -> None:
    """Kernel choice and launch of gemm_a16w8_mxfp8_asm, writing into out.

    Registered as the torch custom op aiter::gemm_a16w8_mxfp8_asm_out (mutates
    out), so the table lookups, the workspace and the launch stay opaque to
    torch.compile and gemm_a16w8_mxfp8_asm traces into one op.
    """
    M, K = A.shape
    N = B.shape[0]
    if kernelName is None:
        cfg = get_gemm_a16w8_mxfp8_config(M, N, K)
        if cfg is None:
            cfg = (_mx_default_kernel(M, N), 1)
        kernelName = cfg[0]
        if splitK is None:
            splitK = cfg[1]
    params = _mx_registry().get(kernelName)
    if params is None:
        raise RuntimeError(f"gemm_a16w8_mxfp8_asm: unknown kernel {kernelName}")
    kernelName = params["knl_name"]
    if splitK is None:
        splitK = 4 if params["sk"] else 0
    if params["sk"]:
        ws, cnt = _mx_workspace(A.device)
    else:  # only the split-K kernels read the workspace and counters
        ws = torch.empty((0,), dtype=torch.uint8, device=A.device)
        cnt = torch.empty((0,), dtype=torch.int32, device=A.device)
    _gemm_a16w8_mxfp8_asm(
        A,
        B.view(torch.uint8),
        B_scale.view(torch.uint8),
        out,
        ws,
        cnt,
        kernelName,
        splitK,
    )
