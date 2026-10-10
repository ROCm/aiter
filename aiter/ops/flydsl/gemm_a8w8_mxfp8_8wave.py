# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8-wave CDNA4 (gfx950) backend for the FlyDSL mxfp8 (1x32 ue8m0) GEMM.

Operands, all preshuffling caller-side as for the ptpc kernel next door:

    XQ      : [M, K] fp8 e4m3, row-major, not preshuffled
    WQ      : [N, K] fp8 e4m3 via ``aiter.ops.shuffle.shuffle_weight(., (16, 16))``
              -- the same layout the ptpc 8-wave kernel reads, so a server
              shuffles the weight once for both
    x_scale : ``shuffle_mxfp8_a_scale`` of the quantiser's [M, K/32] e8m0 bytes
    w_scale : ``shuffle_mxfp8_b_scale`` of the checkpoint's [N/32, K/32] bytes
    Out     : [M, N] bf16

The two scale helpers are not interchangeable with the ones the 4-wave
``mxscale_preshuffle`` GEMM reads, and the buffers are indistinguishable by
shape and dtype -- handing one kernel the other's layout computes garbage
rather than raising. See ``shuffle_mxfp8_a_scale`` for why this layout.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

# Fixed by the kernel: MFMA_Scale(16, 16, 128) over a 128-deep K tile.
BLOCK_K = 128
# The main loop is ``range_constexpr(K_ITERS - 2)`` plus two peeled tail steps,
# so K_ITERS must be >= 2 or the loop count goes negative at trace time.
MIN_K = 2 * BLOCK_K
# ue8m0 block size along K, and along N for the weight.
MXFP8_BLOCK = 32
# The packed A scale puts a lane's four 16-row M tiles in one dword.
BLOCK_M = 256
A_SCALE_GROUP_M = 64

_LDS_BYTES_PER_BLOCK_UNIT = 256  # LDS = 256 * (BLOCK_M + BLOCK_N)
_I32_MAX = 2**31

#: Workgroups the 256-wide tile's grid needs before it beats the 128-wide one.
#:
#: The narrow tile exists to double the grid when the wide one is launch
#: starved, so what decides between them is the wide grid's *size*, not M --
#: and the grid is ``ceildiv(M, BLOCK_M) * (N / BLOCK_N)``, which depends on N
#: as much as on M. Gating on M alone is only ever right for the N it was
#: fitted to: mori measured a bare ``M < 2048`` costing 48% at M=1280 on one
#: shape, because two shapes turn over at different M (between 1024 and 1280 on
#: one, between 1536 and 1792 on the other) and land at 128->160 and 120->140
#: wide-grid workgroups.
WIDE_TILE_MIN_GRID = 140
#: What the narrow tile costs to get: its store cannot use the permlane lane
#: transpose, which needs exactly two N-tiles.
NARROW_BLOCK_N = 128
DEFAULT_BLOCK_N = 256

# Lazily bound flydsl symbols (kept out of the import path when flydsl is absent).
_compile_mxfp8_gemm_8w = None
_run_compiled = None
_fx = None


def _lazy_import() -> None:
    global _compile_mxfp8_gemm_8w, _run_compiled, _fx
    if _compile_mxfp8_gemm_8w is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_mxfp8_8wave import compile_mxfp8_gemm_8w
    from .kernels.tensor_shim import _run_compiled as run_compiled

    _compile_mxfp8_gemm_8w = compile_mxfp8_gemm_8w
    _run_compiled = run_compiled
    _fx = fx_mod


def lds_bytes(block_n: int) -> int:
    """Exact LDS footprint: 4 A buffers of (BM/2)x128 plus 4 B of (BN/2)x128."""
    return _LDS_BYTES_PER_BLOCK_UNIT * (BLOCK_M + int(block_n))


def supports_shape(N: int, K: int, block_n: int = DEFAULT_BLOCK_N) -> bool:
    """Whether (N, K) is expressible by this kernel. M is deliberately absent.

    Any M is *servable* -- it is zero-extended to a multiple of 64 and the tail
    block masks its stores. Whether an M is *profitable* is the caller's call,
    and is decided by the grid rather than by M; see ``pick_block_n``.
    """
    try:
        _validate(K=K, block_n=block_n, N=N)
    except ValueError:
        return False
    return True


def pick_block_n(M: int, N: int) -> int:
    """The N tile to compile for this shape, by wide-grid size."""
    if N % DEFAULT_BLOCK_N:
        return NARROW_BLOCK_N
    wide_grid = -(-M // BLOCK_M) * (N // DEFAULT_BLOCK_N)
    return DEFAULT_BLOCK_N if wide_grid >= WIDE_TILE_MIN_GRID else NARROW_BLOCK_N


def padded_m(M: int) -> int:
    """M rounded up to the packed A scale's 64-row group."""
    return -(-int(M) // A_SCALE_GROUP_M) * A_SCALE_GROUP_M


def _validate(
    *,
    K: int,
    block_n: int,
    waves_per_eu: int = 2,
    M: int | None = None,
    N: int | None = None,
) -> None:
    """Check every kernel precondition, raising ValueError."""
    if K % BLOCK_K:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] K={K} must be a multiple of {BLOCK_K} "
            "(the scaled MFMA's K step)"
        )
    if K < MIN_K:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] K={K} is below the minimum {MIN_K}: the "
            f"main loop prefetches a second K block and runs two tail steps"
        )
    if block_n not in (NARROW_BLOCK_N, DEFAULT_BLOCK_N):
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] block_n={block_n} must be "
            f"{NARROW_BLOCK_N} or {DEFAULT_BLOCK_N}"
        )
    if waves_per_eu < 0:
        raise ValueError(f"[FlyDSL mxfp8 8wave] waves_per_eu={waves_per_eu} < 0")
    need = lds_bytes(block_n)
    cap = get_lds_capacity_bytes()
    if need > cap:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] BLOCK_N={block_n} needs {need} B of LDS, "
            f"over the {cap} B available"
        )
    if N is not None and N % block_n:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] N={N} must be a multiple of block_n={block_n}"
        )
    if N is not None and N % MXFP8_BLOCK:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] N={N} must be a multiple of {MXFP8_BLOCK} "
            "(the weight's ue8m0 block along N)"
        )
    if M is not None and N is not None:
        if max(M * K, N * K, M * N) >= _I32_MAX:
            raise ValueError(
                f"[FlyDSL mxfp8 8wave] M={M} N={N} K={K} overflows 32-bit "
                "buffer indexing"
            )


@functools.lru_cache(maxsize=1024)
def compile_mxfp8_8wave_gemm(
    *,
    K: int,
    block_n: int,
    waves_per_eu: int,
    xcd_swizzle: int,
):
    """Compile (and memoize) an 8-wave mxfp8 launcher.

    Every value that changes codegen must appear in this signature: the cache
    key is the arguments, so one left out silently serves a kernel built for a
    different configuration.
    """
    _lazy_import()
    _validate(K=K, block_n=block_n, waves_per_eu=waves_per_eu)
    # The permlane store pairs exactly two N-tiles, so the narrow tile gives it
    # up -- and with it the lane transpose that builds on it.
    permlane = block_n == DEFAULT_BLOCK_N
    return _compile_mxfp8_gemm_8w(
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=int(block_n),
        permlane=permlane,
        lane_transpose=permlane,
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
    )


def _as_i8(t: Tensor) -> Tensor:
    """Bitcast fp8 storage to int8; ``make_fp8_buffer_tensor`` recasts the iterator."""
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _as_flat_i32(scale: Tensor, name: str) -> Tensor:
    """Flatten a shuffled scale buffer to contiguous 1-D int32.

    The kernel reads these through a buffer descriptor sized from the shape
    arguments, so a buffer that is merely the wrong *layout* reads in range and
    computes a wrong answer. Only the element type is checkable here.
    """
    if scale.dtype not in (torch.int32, torch.uint32):
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] {name} must be int32 (use "
            f"shuffle_mxfp8_a_scale / shuffle_mxfp8_b_scale), got {scale.dtype}"
        )
    return scale.reshape(-1).contiguous().view(torch.int32)


def flydsl_8wave_gemm_mxfp8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    block_n: int | None = None,
    *,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
) -> Tensor:
    """Run the 8-wave mxfp8 GEMM; writes into ``Out`` and returns it.

    ``block_n`` defaults to ``pick_block_n(M, N)``.

    M must be a multiple of 64 -- the packed A scale's group -- and need not be
    a multiple of ``BLOCK_M``: the grid is ``ceildiv(M, BLOCK_M)`` and the tail
    block masks its stores. Pad *before* quantising, so the A scale is built
    over the padded M and its packed layout needs no repair.
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
    if M % A_SCALE_GROUP_M:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] M={M} must be a multiple of "
            f"{A_SCALE_GROUP_M} (the packed A scale's group); pad before "
            "quantising"
        )

    if block_n is None:
        block_n = pick_block_n(M, N)
    _validate(K=K, block_n=block_n, waves_per_eu=waves_per_eu, M=M, N=N)

    sa = _as_flat_i32(x_scale, "x_scale")
    sb = _as_flat_i32(w_scale, "w_scale")
    want_a = M * (K // MXFP8_BLOCK) // 4  # packed: 4 bytes to a dword
    want_b = (N // MXFP8_BLOCK) * (K // MXFP8_BLOCK)
    if sa.numel() != want_a:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] x_scale must have {want_a} int32 after "
            f"shuffle_mxfp8_a_scale, got {sa.numel()}"
        )
    if sb.numel() != want_b:
        raise ValueError(
            f"[FlyDSL mxfp8 8wave] w_scale must have {want_b} int32 after "
            f"shuffle_mxfp8_b_scale, got {sb.numel()}"
        )

    exe = compile_mxfp8_8wave_gemm(
        K=K,
        block_n=int(block_n),
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
    )

    out_contig = Out.contiguous()
    # The launcher takes (A, B_T, C, A_scale, B_scale, c_m, c_n, stream) -- A
    # first and C *third*, the opposite of the preshuffle launcher's
    # (C, A, B, ...). Getting it wrong yields a kernel that runs and writes
    # garbage.
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
