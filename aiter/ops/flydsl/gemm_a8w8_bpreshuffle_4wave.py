# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime wrapper for the gfx950 4-wave signed-INT8 bpreshuffle GEMM."""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import (
    get_cdna4_num_xcds,
    get_cu_num,
    get_gfx_runtime,
)

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
MIN_K = 2 * BLOCK_K
DEFAULT_XCD_SWIZZLE = 8
_I32_MAX = 2**31
# Full-range signed INT8 products reach 128 * 128, and all K of them accumulate
# in INT32.
MAX_K = (_I32_MAX - 1) // (128 * 128) // BLOCK_K * BLOCK_K

_compile_int8_gemm_4w = None
_run_compiled = None
_fx = None


def _lazy_import() -> None:
    global _compile_int8_gemm_4w, _run_compiled, _fx
    if _compile_int8_gemm_4w is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from .kernels.tensor_shim import _run_compiled as run_compiled

    _compile_int8_gemm_4w = compile_int8_gemm_4w
    _run_compiled = run_compiled
    _fx = fx_mod


_NAME_PREFIX = "flydsl_bpreshuffle_4w_"


def is_4wave_kernel_name(name: str) -> bool:
    """Whether a tuned-table ``kernelName`` names a 4-wave INT8 kernel."""
    return name.startswith(_NAME_PREFIX)


def make_4wave_kernel_name(
    *,
    K: int,
    xcd_swizzle: int = DEFAULT_XCD_SWIZZLE,
    num_cu: int = 0,
    num_xcds: int = 8,
) -> str:
    """Return the canonical tuned-table name for one 4-wave specialization.

    The kernel's symbol is this name with its epilogue's parts after it
    (``compile_int8_gemm_4w``).
    """
    name = (
        f"{_NAME_PREFIX}{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_"
        f"I8_I8_B16_1x{int(xcd_swizzle)}_k{int(K)}"
    )
    if num_cu > 0:
        name += f"_p{int(num_cu)}"
    if num_xcds != 8:
        name += f"_xcd{int(num_xcds)}"
    return name


_KERNEL_NAME_RE = re.compile(
    r"^flydsl_bpreshuffle_4w_"
    r"(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"1x(?P<xcd_swizzle>\d+)_k(?P<k>\d+)"
    r"(?:_p(?P<num_cu>\d+))?"
    r"(?:_xcd(?P<num_xcds>\d+))?$"
)


def parse_4wave_kernel_name(name: str) -> dict[str, int | str] | None:
    """Parse a canonical 4-wave kernel name, or return ``None``."""
    match = _KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "block_m": int(values["block_m"]),
        "block_n": int(values["block_n"]),
        "block_k": int(values["block_k"]),
        "q_dtype_a": values["qa"],
        "q_dtype_w": values["qw"],
        "dtype": values["out"],
        "xcd_swizzle": int(values["xcd_swizzle"]),
        "k": int(values["k"]),
        "num_cu": int(values["num_cu"] or 0),
        "num_xcds": int(values["num_xcds"] or 8),
    }


def validate_4wave_config(
    *, K: int, xcd_swizzle: int, num_cu: int, num_xcds: int
) -> None:
    """Raise for a 4-wave configuration the kernel cannot run; every compile checks it.

    A persistent grid (``num_cu > 0``) is a multiple of ``num_xcds`` workgroups, so each
    block stays on its workgroup's XCD. The op path also requires a tuned row's
    ``num_cu`` to be the device's (``run_gemm_a8w8_bpreshuffle_4wave``).
    """
    if K < MIN_K or K > MAX_K or K % BLOCK_K:
        raise ValueError(
            f"[FlyDSL INT8 4wave] K must be in [{MIN_K}, {MAX_K}] and divisible "
            f"by {BLOCK_K}, got {K}"
        )
    if xcd_swizzle < 0:
        raise ValueError(
            f"[FlyDSL INT8 4wave] xcd_swizzle must be non-negative, got "
            f"{xcd_swizzle}"
        )
    if num_xcds <= 0 or num_xcds > 8:
        raise ValueError(
            f"[FlyDSL INT8 4wave] num_xcds must be in [1, 8], got {num_xcds}"
        )
    if num_cu < 0 or num_cu % num_xcds:
        raise ValueError(
            "[FlyDSL INT8 4wave] persistent num_cu must be zero or a multiple of "
            f"num_xcds, got num_cu={num_cu}, num_xcds={num_xcds}"
        )
    if num_cu and (K // BLOCK_K) % 2:
        raise ValueError(
            "[FlyDSL INT8 4wave] persistent mode requires an even number of "
            f"K128 tiles, got K={K}"
        )


def compile_4wave_gemm(
    *,
    K: int,
    xcd_swizzle: int = DEFAULT_XCD_SWIZZLE,
    num_cu: int = 0,
    num_xcds: int | None = None,
    bias: bool = False,
):
    """The current device's 4-wave launcher of the op path (``compile_int8_gemm_4w``,
    which compiles each configuration once per device); this cache keeps the op's
    per-call lookup short.

    With ``bias`` the epilogue adds a per-column FP32 bias before the BF16 rounding, and
    the launcher's column operand holds the N column scales followed by the N biases.
    """
    return _compile_4wave_gemm(
        torch.cuda.current_device(), K, xcd_swizzle, num_cu, num_xcds, bias
    )


@functools.lru_cache(maxsize=1024)
def _compile_4wave_gemm(device, K, xcd_swizzle, num_cu, num_xcds, bias):
    # device only keys the cache: a launcher runs on the device that first ran it.
    _lazy_import()
    if num_xcds is None:
        num_xcds = get_cdna4_num_xcds() if int(xcd_swizzle) > 0 else 8
    return _compile_int8_gemm_4w(
        K=int(K),
        xcd_swizzle=int(xcd_swizzle),
        num_cu=int(num_cu),
        num_xcds=int(num_xcds),
        bias=bool(bias),
    )


def default_4wave_kernel_name(
    M: int,
    N: int,
    K: int,
    *,
    cu_num: int | None = None,
    num_xcds: int | None = None,
) -> str:
    """Select the validated generic variant for an untuned supported shape.

    Persistent workgroups are enabled only for the measured K=5120 path and
    only when the output tile grid exceeds the active CU count.
    """
    cu_num = int(get_cu_num() if cu_num is None else cu_num)
    num_xcds = int(get_cdna4_num_xcds(cu_num) if num_xcds is None else num_xcds)
    grid = (int(M) + BLOCK_M - 1) // BLOCK_M
    grid *= (int(N) + BLOCK_N - 1) // BLOCK_N
    persistent_cu = cu_num if int(K) == 5120 and grid > cu_num else 0
    return make_4wave_kernel_name(
        K=int(K),
        xcd_swizzle=DEFAULT_XCD_SWIZZLE,
        num_cu=persistent_cu,
        num_xcds=num_xcds,
    )


def supports_4wave_int8_shape(M: int, N: int, K: int) -> bool:
    """Whether a logical shape satisfies the public 4-wave INT8 contract."""
    return (
        M >= 0
        and N > 0
        and N % 16 == 0
        and MIN_K <= K <= MAX_K
        and K % BLOCK_K == 0
        and M * N * 2 < _I32_MAX
        and M * K < _I32_MAX
        and N * K < _I32_MAX
    )


def _as_scale(scale: Tensor, length: int, name: str) -> Tensor:
    flat = scale.reshape(-1)
    if flat.numel() != length:
        raise ValueError(
            f"[FlyDSL INT8 4wave] {name} must have {length} elements, got "
            f"{tuple(scale.shape)}"
        )
    if flat.dtype != torch.float32:
        raise ValueError(f"[FlyDSL INT8 4wave] {name} must be FP32, got {flat.dtype}")
    return flat.contiguous()


def check_bias(bias: Tensor, length: int, device: torch.device) -> None:
    """Raise unless ``bias`` is an FP32, BF16 or FP16 ``[N]`` or ``[1, N]`` on ``device``."""
    if tuple(bias.shape) not in ((length,), (1, length)):
        raise ValueError(
            f"[FlyDSL INT8 4wave] bias must be [{length}] or [1, {length}], got "
            f"{tuple(bias.shape)}"
        )
    if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(
            f"[FlyDSL INT8 4wave] bias must be FP32, BF16 or FP16, got {bias.dtype}"
        )
    if bias.device != device:
        raise ValueError(
            f"[FlyDSL INT8 4wave] bias must be on {device}, got {bias.device}"
        )


def _as_bias(bias: Tensor, length: int, device: torch.device) -> Tensor:
    """The bias as FP32 ``[N]``; BF16 and FP16 biases are converted."""
    check_bias(bias, length, device)
    return bias.reshape(-1).to(torch.float32).contiguous()


def _validate_runtime(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
) -> tuple[int, int, int]:
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError(
            "[FlyDSL INT8 4wave] kernel is supported only on gfx950"
        )
    if XQ.ndim != 2 or WQ.ndim != 2:
        raise ValueError(
            "[FlyDSL INT8 4wave] A/B must be 2-D, got "
            f"{tuple(XQ.shape)} and {tuple(WQ.shape)}"
        )
    if XQ.dtype != torch.int8 or WQ.dtype != torch.int8:
        raise ValueError(
            "[FlyDSL INT8 4wave] A/B must both be torch.int8, got "
            f"{XQ.dtype} and {WQ.dtype}"
        )
    if not XQ.is_cuda or not WQ.is_cuda or not Out.is_cuda:
        raise ValueError("[FlyDSL INT8 4wave] A/B/Out must be GPU tensors")
    devices = {XQ.device, WQ.device, x_scale.device, w_scale.device, Out.device}
    if len(devices) != 1:
        raise ValueError(
            f"[FlyDSL INT8 4wave] all tensors must share one device, got {devices}"
        )
    if not XQ.is_contiguous() or not WQ.is_contiguous():
        raise ValueError("[FlyDSL INT8 4wave] A and preshuffled B must be contiguous")
    M, K = XQ.shape
    N, weight_k = WQ.shape
    if weight_k != K:
        raise ValueError(f"[FlyDSL INT8 4wave] K mismatch: A.K={K}, B.K={weight_k}")
    if N <= 0 or N % 16:
        raise ValueError(
            f"[FlyDSL INT8 4wave] N must be a positive multiple of 16, got {N}"
        )
    validate_4wave_config(
        K=K,
        xcd_swizzle=DEFAULT_XCD_SWIZZLE,
        num_cu=0,
        num_xcds=get_cdna4_num_xcds(),
    )
    if Out.dtype != torch.bfloat16 or tuple(Out.shape) != (M, N):
        raise ValueError(
            f"[FlyDSL INT8 4wave] Out must be BF16 with shape {(M, N)}, got "
            f"{Out.dtype} {tuple(Out.shape)}"
        )
    if not Out.is_contiguous():
        raise ValueError("[FlyDSL INT8 4wave] Out must be contiguous")
    if M * N * 2 >= _I32_MAX or M * K >= _I32_MAX or N * K >= _I32_MAX:
        raise ValueError(
            f"[FlyDSL INT8 4wave] M={M}, N={N}, K={K} exceeds 32-bit indexing"
        )
    return M, N, K


def run_gemm_a8w8_bpreshuffle_4wave(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
    bias: Tensor | None,
) -> Tensor:
    """Decode ``kernel_name`` and execute the matching cached specialization.

    ``bias`` (``[N]`` or ``[1, N]``, FP32, BF16 or FP16), or None, is added in FP32 to
    each column before the BF16 rounding: ``Out = bf16(acc * x_scale * w_scale + bias)``.
    """
    config = parse_4wave_kernel_name(kernel_name)
    if config is None:
        raise ValueError(
            f"[FlyDSL INT8 4wave] unrecognized kernelName: {kernel_name!r}"
        )
    if (
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["q_dtype_a"],
        config["q_dtype_w"],
        config["dtype"],
    ) != (BLOCK_M, BLOCK_N, BLOCK_K, "I8", "I8", "B16"):
        raise ValueError(
            f"[FlyDSL INT8 4wave] unsupported kernel contract: {kernel_name!r}"
        )
    M, N, K = _validate_runtime(XQ, WQ, x_scale, w_scale, Out)
    device = XQ.get_device()
    if device != torch.cuda.current_device():
        with torch.cuda.device(device):
            return run_gemm_a8w8_bpreshuffle_4wave(
                XQ, WQ, x_scale, w_scale, Out, kernel_name, bias
            )
    if int(config["k"]) != K:
        raise ValueError(
            f"[FlyDSL INT8 4wave] kernelName K={config['k']} does not match K={K}"
        )
    active_cu = int(get_cu_num())
    active_xcds = int(get_cdna4_num_xcds(active_cu))
    if int(config["num_xcds"]) != active_xcds:
        raise ValueError(
            f"[FlyDSL INT8 4wave] kernelName targets {config['num_xcds']} XCDs, "
            f"but the active device exposes {active_xcds}"
        )
    if int(config["num_cu"]) not in (0, active_cu):
        raise ValueError(
            f"[FlyDSL INT8 4wave] persistent kernel targets {config['num_cu']} "
            f"CUs, but the active device exposes {active_cu}"
        )
    if M == 0:
        return Out

    sa = _as_scale(x_scale, M, "x_scale")
    # The column operand: the N column scales, then with a bias the N biases.
    columns = _as_scale(w_scale, N, "w_scale")
    if bias is not None:
        columns = torch.cat((columns, _as_bias(bias, N, XQ.device)))
    executable = _compile_4wave_gemm(
        device,
        K,
        int(config["xcd_swizzle"]),
        int(config["num_cu"]),
        int(config["num_xcds"]),
        bias is not None,
    )
    _run_compiled(
        executable,
        XQ.view(-1),
        WQ.view(-1),
        Out.view(-1),
        sa,
        columns,
        M,
        N,
        _fx.Stream(torch.cuda.current_stream(device)),
    )
    return Out


__all__ = [
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N",
    "MIN_K",
    "check_bias",
    "compile_4wave_gemm",
    "default_4wave_kernel_name",
    "is_4wave_kernel_name",
    "make_4wave_kernel_name",
    "parse_4wave_kernel_name",
    "run_gemm_a8w8_bpreshuffle_4wave",
    "supports_4wave_int8_shape",
    "validate_4wave_config",
]
