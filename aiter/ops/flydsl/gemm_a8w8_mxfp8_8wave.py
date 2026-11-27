# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8-wave CDNA4 (gfx950) backend for the FlyDSL mxfp8 (1x32 e8m0) bpreshuffle GEMM."""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .aot_launch import aot_factory

# Fixed by the kernel: MFMA_Scale(16, 16, 128) over a 128-deep K tile, one
# 256x256 block tile, and scales staged a 256-K pair at a time.
BLOCK_K = 128
BLOCK_M = BLOCK_N = 256
K_ALIGN = 256
SCALE_GROUP = 32

# A/B stages: 4 buffers each of (BLOCK/2) x 128 fp8; the row layout adds two
# 8 KB scale-chunk slots.
_STAGE_LDS_BYTES = 4 * (BLOCK_M // 2 + BLOCK_N // 2) * BLOCK_K
_ROW_SCALE_LDS_BYTES = 2 * 2 * 256 * 16
_I32_MAX = 2**31

# Lazily bound flydsl symbols (kept out of the import path when flydsl is absent).
_compile_mxfp8_gemm_8w = None
_current_stream_arg = None
_run_compiled = None


def _lazy_import() -> None:
    global _compile_mxfp8_gemm_8w, _current_stream_arg, _run_compiled
    if _compile_mxfp8_gemm_8w is not None:
        return
    from .kernels.mxfp8_gemm_8wave import compile_mxfp8_gemm_8w
    from .kernels.tensor_shim import _run_compiled as run_compiled
    from .kernels.tensor_shim import current_stream_arg

    _compile_mxfp8_gemm_8w = compile_mxfp8_gemm_8w
    _current_stream_arg = current_stream_arg
    _run_compiled = run_compiled


def lds_bytes(scale_preshuffled: bool = False) -> int:
    """Exact LDS footprint of the 256x256 kernel."""
    return _STAGE_LDS_BYTES + (0 if scale_preshuffled else _ROW_SCALE_LDS_BYTES)


def _validate(
    *,
    K: int,
    waves_per_eu: int,
    scale_preshuffled: bool,
    M: int | None = None,
    N: int | None = None,
) -> None:
    """Check every kernel precondition, raising ValueError.

    The kernel itself uses bare ``assert`` for its constraints, and
    ``aiter.utility.mp_tuner`` only catches ``(RuntimeError, ValueError)`` -- an
    AssertionError escaping here would kill the tuner worker process rather than
    just failing the candidate.
    """
    if K <= 0 or K % K_ALIGN != 0:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] K must be a positive multiple of {K_ALIGN}, got {K}"
        )
    if int(waves_per_eu) < 1:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] waves_per_eu must be >=1 (the kernel always emits "
            f"the rocdl.waves_per_eu attribute), got {waves_per_eu}"
        )
    # Local import: this module stays importable without flydsl so the tuner can
    # read K_ALIGN/lds_bytes on hosts that cannot compile.
    from flydsl.runtime.device import get_rocm_arch

    arch = get_rocm_arch().split(":", 1)[0]
    if arch != "gfx950":
        raise ValueError(f"[FlyDSL mxfp8 8wave] gfx950 only, got {arch}")
    need = lds_bytes(scale_preshuffled)
    have = get_lds_capacity_bytes(arch)
    if need > have:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] needs {need} B of LDS, limit is {have} B"
        )
    if M is not None and N is not None:
        # Buffer descriptors index in 32-bit; C is bf16 so its byte count doubles.
        if M <= 0 or N <= 0:
            raise ValueError(
                f"[FlyDSL mxfp8 8wave] M and N must be positive, got M={M}, N={N}"
            )
        if M * N * 2 >= _I32_MAX or M * K >= _I32_MAX or N * K >= _I32_MAX:
            raise ValueError(
                f"[FlyDSL mxfp8 8wave] M={M} N={N} K={K} overflows 32-bit buffer indexing"
            )


@aot_factory("gemm_a8w8_mxfp8_8wave")
@functools.lru_cache(maxsize=1024)
def compile_mxfp8_8wave_gemm(
    *,
    K: int,
    waves_per_eu: int,
    xcd_swizzle: int,
    scale_preshuffled: bool = False,
    w_scale_block: str = "1x32",
    k_split: int = 1,
):
    """Compile (and memoize) an 8-wave mxfp8 launcher (``k_split > 1``: the
    split-K tail kernel, see compile_mxfp8_gemm_8w)."""
    _lazy_import()
    _validate(K=K, waves_per_eu=waves_per_eu, scale_preshuffled=scale_preshuffled)
    return _compile_mxfp8_gemm_8w(
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        b_preshuffled=True,
        scale_preshuffled=bool(scale_preshuffled),
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
        b_scale_block=w_scale_block,
        a_scale_block="1x128" if w_scale_block == "128x128" else "1x32",
        k_split=int(k_split),
    )


def _as_u8(t: Tensor) -> Tensor:
    """Bitcast 1-byte fp8 / e8m0 storage to uint8 for the buffer descriptors."""
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def flydsl_mxfp8_8wave_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    *,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
    w_scale_block: str = "1x32",
    n_split: int = 1,
) -> Tensor:
    """Run the 8-wave mxfp8 GEMM; writes into ``Out`` and returns it.

    XQ: ``(M, K)`` fp8 e4m3. WQ: ``(N, K)`` fp8 e4m3 in ``shuffle_weight``
    ``(16, 16)`` layout. x_scale ``[M, K/32]`` / w_scale ``[N, K/32]`` e8m0,
    row-major and unshuffled; a 1-D scale pair is the ``shuffle_scale_w4``
    layout over rows padded to 256. ``w_scale_block="32x32"`` takes a row-major
    w_scale ``[N/32, K/32]`` (N a multiple of 32); ``"128x128"`` the blockscale
    pair, w_scale ``[N/128, K/128]`` (N a multiple of 128, K >= 512) and x_scale
    a contiguous ``(M, K/128)`` holding the column-major ``[K/128, M]`` bytes, as
    bmm's transposed x_scale. Out: ``(M, N)`` bf16; a row-major one (unit column
    stride) may have a row stride above N -- a column slice of a wider output --
    and is written in place, any other layout through a contiguous copy.

    ``n_split > 1``: when the 256x256 tiles overrun the CUs by a partial wave,
    the columns that fill whole waves run as usual and the rest (the last
    wave) runs K-split ``n_split`` ways into an fp32 workspace that a reduce
    sums into Out (see _run_tail_split). Shapes it does not apply to run
    unsplit.

    M need not be a multiple of 256 nor N: out-of-range A/B/scale reads return
    zero via the buffer descriptors, C row overflow is dropped the same way, and
    C column overflow is redirected out of bounds by the kernel.
    """
    _lazy_import()

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] A/B must be 2-D, got {tuple(XQ.shape)}, "
            f"{tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise ValueError("[FlyDSL mxfp8 8wave] A/B must be 1-byte fp8 storage")
    if Out.dtype != torch.bfloat16:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] only bf16 output is supported, got {Out.dtype}"
        )

    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] K mismatch: A.K={K} vs B.K={WQ.shape[1]}"
        )
    if tuple(Out.shape) != (M, N):
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] Out must be ({M}, {N}), got {tuple(Out.shape)}"
        )
    if M == 0 or N == 0:
        return Out
    # A row-major Out (unit column stride, row stride >= N) is written in
    # place; any other layout goes through a contiguous copy.
    strided = (
        Out.stride(1) == 1
        and Out.stride(0) >= N
        and (n_split <= 1 or Out.stride(0) % 2 == 0)
    )
    out_c = Out if strided else Out.contiguous()
    ldc = N if M == 1 else int(out_c.stride(0))

    scale_preshuffled = x_scale.dim() == 1
    if (w_scale.dim() == 1) != scale_preshuffled:
        raise ValueError(
            "[FlyDSL mxfp8 8wave] x_scale and w_scale must both be row-major 2-D "
            "or both pre-shuffled 1-D"
        )
    # w_scale rows per scale row, x_scale K per scale column.
    blocks = {"1x32": (1, 32), "32x32": (32, 32), "128x128": (128, 128)}
    if w_scale_block not in blocks:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] w_scale_block must be one of {sorted(blocks)}, "
            f"got {w_scale_block!r}"
        )
    w_rows, x_k = blocks[w_scale_block]
    if w_rows > 1 and (scale_preshuffled or N % w_rows != 0):
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] a {w_scale_block} w_scale is row-major 2-D with "
            f"N a multiple of {w_rows}, got N={N}"
        )
    if w_scale_block == "128x128" and K < 512:
        raise ValueError("[FlyDSL mxfp8 8wave] a 128x128 w_scale needs K >= 512")
    if scale_preshuffled:
        want = {
            "x_scale": -(-M // 256) * 256 * K // 32,
            "w_scale": -(-N // 256) * 256 * K // 32,
        }
        got = {"x_scale": x_scale.numel(), "w_scale": w_scale.numel()}
    else:
        want = {
            "x_scale": (M, K // x_k),
            "w_scale": (N // w_rows, K // x_k),
        }
        got = {"x_scale": tuple(x_scale.shape), "w_scale": tuple(w_scale.shape)}
    for name, t in (("x_scale", x_scale), ("w_scale", w_scale)):
        if got[name] != want[name]:
            raise ValueError(
                f"[FlyDSL mxfp8 8wave] {name} must be {want[name]}, got {got[name]}"
            )
        if t.element_size() != 1 or not t.is_contiguous():
            raise ValueError(
                f"[FlyDSL mxfp8 8wave] {name} must be contiguous 1-byte e8m0"
            )

    _validate(
        K=K,
        waves_per_eu=waves_per_eu,
        scale_preshuffled=scale_preshuffled,
        M=M,
        N=ldc,
    )

    if n_split > 1:
        n1 = tail_split_n1(M, N, K, n_split, scale_preshuffled, w_scale_block)
        if n1 is not None:
            _run_tail_split(
                XQ,
                WQ,
                x_scale,
                w_scale,
                out_c,
                n1,
                waves_per_eu=waves_per_eu,
                xcd_swizzle=xcd_swizzle,
                w_scale_block=w_scale_block,
                n_split=n_split,
            )
            if out_c is not Out:
                Out.copy_(out_c)
            return Out

    compile_kwargs = {
        "K": K,
        "waves_per_eu": int(waves_per_eu),
        "xcd_swizzle": int(xcd_swizzle),
        "scale_preshuffled": scale_preshuffled,
        "w_scale_block": w_scale_block,
    }

    # Every tensor is a flat unit-stride uint8 or bf16 view, so the compile
    # arguments determine the launcher's argument signature. The device keeps
    # the JIT path's handles apart; entries on disk are shared.
    # The 8-wave launcher takes (A, B_T, C, A_scale, B_scale, c_m, c_n, ldc,
    # stream) -- A first and C *third*, as gemm_a8w8_bpreshuffle_8wave. C is
    # passed as the flat span from Out's first element to its last.
    _run_compiled(
        compile_mxfp8_8wave_gemm.lazy(**compile_kwargs),
        _as_u8(XQ.contiguous()).view(-1),
        _as_u8(WQ.contiguous()).view(-1),
        out_c.as_strided(((M - 1) * ldc + N,), (1,)),
        _as_u8(x_scale).view(-1),
        _as_u8(w_scale).view(-1),
        M,
        N,
        ldc,
        _current_stream_arg(XQ.device),
        specialization_key=XQ.device,
    )
    if out_c is not Out:
        Out.copy_(out_c)
    return Out


def tail_split_n1(M, N, K, n_split, scale_preshuffled, w_scale_block):
    """Columns of the whole-wave head of a tail split, or None when the shape
    has no partial last wave to split (or the split kernel cannot run it)."""
    from aiter.jit.utils.chip_info import get_cu_num

    if (
        scale_preshuffled
        or w_scale_block == "128x128"
        or N % BLOCK_N
        or K % (n_split * K_ALIGN)
    ):
        return None
    m_tiles = -(-M // BLOCK_M)
    n_tiles = N // BLOCK_N
    cus = get_cu_num()
    head_n = (m_tiles * n_tiles // cus) * cus // m_tiles  # N tiles in whole waves
    if head_n == 0 or head_n == n_tiles:
        return None
    return head_n * BLOCK_N


def _run_tail_split(
    XQ,
    WQ,
    x_scale,
    w_scale,
    Out,
    n1,
    *,
    waves_per_eu,
    xcd_swizzle,
    w_scale_block,
    n_split,
):
    """Head = columns [0, n1) unsplit, in place; tail = the rest, K-split
    ``n_split`` ways into an fp32 [n_split, M, N - n1] workspace, then summed
    into Out[:, n1:] by the mxpsh split-K reduce (in split order)."""
    from .kernels.tensor_shim import ptr_arg
    from .mxscale_preshuffle_kernels import _bound_reduce_exe

    M, K = XQ.shape
    N = WQ.shape[0]
    rows = 32 if w_scale_block == "32x32" else 1  # N-rows per w_scale row
    n2 = N - n1
    if n_split * M * n2 * 4 >= _I32_MAX:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] split workspace {n_split}x{M}x{n2} overflows "
            "32-bit indexing"
        )
    flydsl_mxfp8_8wave_gemm(
        XQ,
        WQ[:n1],
        x_scale,
        w_scale[: n1 // rows],
        Out[:, :n1],
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
        w_scale_block=w_scale_block,
    )
    ws = torch.empty((n_split, M, n2), dtype=torch.float32, device=XQ.device)
    compile_kwargs = {
        "K": K,
        "waves_per_eu": int(waves_per_eu),
        "xcd_swizzle": 0,
        "w_scale_block": w_scale_block,
        "k_split": n_split,
    }
    stream = _current_stream_arg(XQ.device)
    _run_compiled(
        compile_mxfp8_8wave_gemm.lazy(**compile_kwargs),
        _as_u8(XQ.contiguous()).view(-1),
        _as_u8(WQ[n1:].contiguous()).view(-1),
        ws.view(-1),
        _as_u8(x_scale).view(-1),
        _as_u8(w_scale[n1 // rows :]).view(-1),
        M,
        n2,
        n2,  # ldc: unused by the split kernel
        stream,
        specialization_key=XQ.device,
    )
    tail = Out[:, n1:]
    ldc = int(tail.stride(0))
    _run_compiled(
        _bound_reduce_exe.lazy(n_split, "bf16", True),
        ptr_arg(ws),
        ptr_arg(tail),
        M * n2 // 2,  # n_out_dw
        M * n2,  # slab_stride_dw
        n2 // 2,  # row_dw
        ldc // 2,  # ldc_dw
        stream,
        specialization_key=XQ.device,
    )
    return Out


NAME_PREFIX = "flydsl_mxfp8_8w"
# flydsl_mxfp8_8w_{BM}x{BN}x{BK}_{QA}_{QW}_{OUT}_{wpe}x{xcd}[_ns{n_split}]
_KERNEL_NAME_RE = re.compile(
    rf"^{NAME_PREFIX}_"
    r"(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"(?P<waves_per_eu>\d+)x(?P<xcd_swizzle>\d+)(?:_ns(?P<n_split>\d+))?$"
)


def make_mxfp8_8wave_kernel_name(
    waves_per_eu: int, xcd_swizzle: int, n_split: int = 1
) -> str:
    return (
        f"{NAME_PREFIX}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_B16_"
        f"{waves_per_eu}x{xcd_swizzle}{f'_ns{n_split}' if n_split > 1 else ''}"
    )


def parse_mxfp8_8wave_kernel_name(name: str) -> dict | None:
    """Parse a ``flydsl_mxfp8_8w_`` kernelName into its config dict, or None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    if m is None:
        return None
    g = m.groupdict()
    return {
        "block_m": int(g["block_m"]),
        "block_n": int(g["block_n"]),
        "block_k": int(g["block_k"]),
        "q_dtype_a": g["qa"],
        "q_dtype_w": g["qw"],
        "dtype": g["out"],
        "waves_per_eu": int(g["waves_per_eu"]),
        "xcd_swizzle": int(g["xcd_swizzle"]),
        "n_split": int(g["n_split"] or 1),
    }


def run_gemm_a8w8_mxfp8_8wave(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
    w_scale_block: str = "1x32",
) -> Tensor:
    """Dispatch entry: decode a tuned mxfp8 8wave kernelName and run the kernel."""
    cfg = parse_mxfp8_8wave_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] unrecognised kernelName: {kernel_name!r}"
        )
    if (cfg["block_m"], cfg["block_n"], cfg["block_k"]) != (BLOCK_M, BLOCK_N, BLOCK_K):
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] the kernel fixes a {BLOCK_M}x{BLOCK_N}x{BLOCK_K} "
            f"tile: {kernel_name!r}"
        )
    return flydsl_mxfp8_8wave_gemm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        waves_per_eu=cfg["waves_per_eu"],
        xcd_swizzle=cfg["xcd_swizzle"],
        w_scale_block=w_scale_block,
        n_split=cfg["n_split"],
    )
