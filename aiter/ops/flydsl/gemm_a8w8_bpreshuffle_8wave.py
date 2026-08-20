# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8-wave CDNA4 (gfx950) backend for the FlyDSL a8w8 ptpc bpreshuffle GEMM."""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

from .utils import get_shared_memory_per_block

# Fixed by the kernel: MFMA_Scale(16, 16, 128) over a 128-deep K tile.
BLOCK_K = 128
# The main loop is ``range_constexpr(K_ITERS - 2)`` plus two peeled tail steps,
# so K_ITERS must be >= 2 or the loop count goes negative at trace time.
MIN_K = 2 * BLOCK_K

_LDS_BYTES_PER_BLOCK_UNIT = 256  # LDS = 256 * (BLOCK_M + BLOCK_N), see _lds_bytes
_I32_MAX = 2**31

# Lazily bound flydsl symbols (kept out of the import path when flydsl is absent).
_compile_fp8_gemm_8w = None
_run_compiled = None
_fx = None


def _lazy_import() -> None:
    global _compile_fp8_gemm_8w, _run_compiled, _fx
    if _compile_fp8_gemm_8w is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_8wave import compile_fp8_gemm_8w
    from .kernels.tensor_shim import _run_compiled as run_compiled

    _compile_fp8_gemm_8w = compile_fp8_gemm_8w
    _run_compiled = run_compiled
    _fx = fx_mod


def lds_bytes(block_m: int, block_n: int) -> int:
    """Exact LDS footprint: 4 A buffers of (BM/2)x128 plus 4 B buffers of (BN/2)x128."""
    return _LDS_BYTES_PER_BLOCK_UNIT * (int(block_m) + int(block_n))


def _validate(
    *,
    K: int,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    M: int | None = None,
    N: int | None = None,
) -> None:
    """Check every kernel precondition, raising ValueError.

    The kernel itself uses bare ``assert`` for its tile constraints, and
    ``aiter.utility.mp_tuner`` only catches ``(RuntimeError, ValueError)`` -- an
    AssertionError escaping here would kill the tuner worker process rather than
    just failing the candidate. So everything is re-checked as ValueError before
    ``compile_fp8_gemm_8w`` is reached.
    """
    if block_m < 128 or block_m % 128 != 0:
        raise ValueError(
            f"[FlyDSL 8wave] BLOCK_M must be >=128 and %128==0, got {block_m}"
        )
    if block_n < 256 or block_n % 256 != 0:
        raise ValueError(
            f"[FlyDSL 8wave] BLOCK_N must be >=256 and %256==0, got {block_n}"
        )
    if K % BLOCK_K != 0:
        raise ValueError(f"[FlyDSL 8wave] K must be a multiple of {BLOCK_K}, got {K}")
    if K < MIN_K:
        raise ValueError(
            f"[FlyDSL 8wave] K must be >= {MIN_K} (the pipeline peels 2 K-tiles), got {K}"
        )
    if int(waves_per_eu) < 1:
        raise ValueError(
            f"[FlyDSL 8wave] waves_per_eu must be >=1 (the kernel always emits the "
            f"rocdl.waves_per_eu attribute; 'no hint' is not expressible), got {waves_per_eu}"
        )
    need = lds_bytes(block_m, block_n)
    have = get_shared_memory_per_block(fallback_gfx="gfx950")
    if need > have:
        raise ValueError(
            f"[FlyDSL 8wave] {block_m}x{block_n} needs {need} B of LDS, limit is {have} B"
        )
    if M is not None and N is not None:
        # Buffer descriptors index in 32-bit; C is bf16 so its byte count doubles.
        if M <= 0 or N <= 0:
            raise ValueError(
                f"[FlyDSL 8wave] M and N must be positive, got M={M}, N={N}"
            )
        if M * N * 2 >= _I32_MAX or M * K >= _I32_MAX or N * K >= _I32_MAX:
            raise ValueError(
                f"[FlyDSL 8wave] M={M} N={N} K={K} overflows 32-bit buffer indexing"
            )


@functools.lru_cache(maxsize=1024)
def compile_8wave_gemm(
    *,
    K: int,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    xcd_swizzle: int,
    b_preshuffled: bool = True,
):
    """Compile (and memoize) an 8-wave launcher."""
    _lazy_import()
    _validate(K=K, block_m=block_m, block_n=block_n, waves_per_eu=waves_per_eu)
    return _compile_fp8_gemm_8w(
        K=K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        b_preshuffled=bool(b_preshuffled),
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
    )


def _as_1d_fp32(scale: Tensor, length: int, name: str) -> Tensor:
    """Flatten a ``(L,)`` / ``(L, 1)`` / ``(1, L)`` scale to contiguous 1-D fp32.

    aiter passes per-token scales as ``(M, 1)`` and per-channel as ``(1, N)``;
    the kernel wants flat ``(M,)`` / ``(N,)``. The element-count check matters:
    an undersized scale array would be read through a buffer descriptor, return
    zeros out of bounds, and silently produce a partially-zero result.
    """
    flat = scale.reshape(-1)
    if flat.numel() != length:
        raise ValueError(
            f"[FlyDSL 8wave] {name} must have {length} elements, got {tuple(scale.shape)}"
        )
    if flat.dtype != torch.float32:
        flat = flat.to(torch.float32)
    return flat.contiguous()


def _as_i8(t: Tensor) -> Tensor:
    """Bitcast fp8 storage to int8; ``make_fp8_buffer_tensor`` recasts the iterator."""
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def flydsl_8wave_gemm_a8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    block_m: int,
    block_n: int,
    *,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
) -> Tensor:
    """Run the 8-wave a8w8 ptpc GEMM; writes into ``Out`` and returns it.

    XQ: ``(M, K)`` fp8 e4m3. WQ: ``(N, K)`` fp8 e4m3 in ``shuffle_weight``
    ``(16, 16)`` layout. x_scale: per-token fp32. w_scale: per-channel fp32.
    Out: ``(M, N)`` bf16.

    M need not be a multiple of ``block_m`` nor N of ``block_n``: out-of-range A
    and B reads return zero via the buffer descriptor's num_records, C row
    overflow is dropped the same way, and C column overflow is redirected to an
    explicit out-of-bounds index by the kernel's ``col_valid`` select.
    """
    _lazy_import()

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise ValueError(
            f"[FlyDSL 8wave] A/B must be 2-D, got {tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise ValueError("[FlyDSL 8wave] A/B must be 1-byte fp8 storage")
    if Out.dtype != torch.bfloat16:
        raise ValueError(
            f"[FlyDSL 8wave] only bf16 output is supported, got {Out.dtype}"
        )

    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise ValueError(f"[FlyDSL 8wave] K mismatch: A.K={K} vs B.K={WQ.shape[1]}")
    if tuple(Out.shape) != (M, N):
        raise ValueError(
            f"[FlyDSL 8wave] Out must be ({M}, {N}), got {tuple(Out.shape)}"
        )
    if M == 0 or N == 0:
        return Out

    _validate(
        K=K,
        block_m=block_m,
        block_n=block_n,
        waves_per_eu=waves_per_eu,
        M=M,
        N=N,
    )

    sa = _as_1d_fp32(x_scale, M, "x_scale")
    sb = _as_1d_fp32(w_scale, N, "w_scale")

    exe = compile_8wave_gemm(
        K=K,
        block_m=int(block_m),
        block_n=int(block_n),
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
    )

    out_contig = Out.contiguous()
    # NOTE: the 8-wave launcher takes (A, B_T, C, A_scale, B_scale, c_m, c_n,
    # stream) -- A first and C *third*. This is the opposite of the preshuffle
    # launcher's (C, A, B, ...), and getting it wrong yields a kernel that runs
    # happily and writes garbage.
    _run_compiled(
        exe,
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        out_contig.view(-1),
        sa,
        sb,
        M,
        N,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)
    return Out


# flydsl_bpreshuffle_8w_{BM}x{BN}x{BK}_{QA}_{QW}_{OUT}_{wpe}x{xcd}
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_bpreshuffle_8w_"
    r"(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"(?P<waves_per_eu>\d+)x(?P<xcd_swizzle>\d+)$"
)


def parse_8wave_kernel_name(name: str) -> dict | None:
    """Parse a ``flydsl_bpreshuffle_8w_`` kernelName into its config dict, or None."""
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
    }


def run_gemm_a8w8_bpreshuffle_8wave(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
) -> Tensor:
    """Dispatch entry: decode a tuned 8wave kernelName and run the kernel."""
    cfg = parse_8wave_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(f"[FlyDSL 8wave] unrecognised kernelName: {kernel_name!r}")
    if cfg["block_k"] != BLOCK_K:
        raise ValueError(
            f"[FlyDSL 8wave] kernelName says BLOCK_K={cfg['block_k']}, "
            f"but the kernel fixes it at {BLOCK_K}: {kernel_name!r}"
        )
    return flydsl_8wave_gemm_a8(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        cfg["block_m"],
        cfg["block_n"],
        waves_per_eu=cfg["waves_per_eu"],
        xcd_swizzle=cfg["xcd_swizzle"],
    )
