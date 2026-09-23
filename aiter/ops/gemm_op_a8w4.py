# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 MXFP8 x MXFP4 GEMM (a8w4) -- ASM, kernarg preload mode.
# A (activation) is mxfp8 (e4m3, 1 byte/elem); B (weight) is mxfp4 (e2m1,
# 2 elems/byte). Both operands carry OCP MX e8m0 block scales (block=32). The
# tuned CSV selects kernel and split count; misses use the .cu heuristic.
# See csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu.


import torch
from torch import Tensor

from ..jit.core import compile_ops, torch_compile_guard
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..utility import dtypes
from .gemm_op_a8w8 import (
    _mxfp8fp4_gemm_validate,
    _reduce_mxfp8_partials,
    _resolve_mxfp8_gemm_config,
)


@compile_ops(
    "module_mxfp8fp4gemm_asm",
    fc_name="mxfp8_mxfp4_gemm_asm",
    ffi_type="ctypes",
)
def _mxfp8_mxfp4_gemm_asm(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
    B: Tensor,  # B:[N, K/2] mxfp4 e2m1 (always preshuffled)
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0 (shuffled)
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0 (shuffled)
    out: Tensor,  # BF16 [M, N], or [splitk, M, N] for partial outputs
    kernelName: str | None = None,
    a_preshuffle: int = 1,
    splitk: int = 1,
) -> None:
    """Write compact BF16 [splitk, M, N] partials (or [M, N] for splitk=1).

    ``out`` must be contiguous with at least splitk*M*N elements. A workspace
    slice may have a nonzero storage offset. Only the first splitk*M*N elements
    from out.data_ptr() are written; extra capacity is untouched. The output
    shape does not set row/plane strides, and padded/strided views are unsupported.
    """


def _gemm_a8w4_mxfp8_fake(
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


@torch_compile_guard(mutates_args=[], gen_fake=_gemm_a8w4_mxfp8_fake)
def gemm_a8w4_mxfp8(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3
    B: Tensor,  # B:[N, K/2] mxfp4 e2m1
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,
    a_preshuffle: bool = True,
    kernelName: str = "",
    splitk: int | None = None,
) -> Tensor:
    """gfx1250 MXFP8 (activation) x MXFP4 (weight) GEMM (a8w4). D[M,N] bf16 =
    A @ B^T with e8m0 block scales. Select kernel and split count from
    ``b_intype=mxfp4`` rows in the shared tuned CSV unless overridden.
    Try exact M first, then get_padded_m at granularity levels 0 and 1; all other
    key fields must match exactly. Reuse a row only if its saved and actual shapes
    satisfy the kernel guards. Lookup does not pad inputs or change output shape.
    Missing CSVs and invalid tuned rows are logged and discarded as config misses.
    A miss uses the native heuristic and preserves an explicit ``splitk``;
    otherwise it defaults to one split. Explicit kernels bypass the CSV and
    default to one split. A CSV kernel incompatible with an explicit split count
    is skipped; an invalid explicit kernel/count combination raises before
    output allocation.

    Split counts are literal counts, not log2 values. The 256x256_4x4 and
    64x512_4x1 kernels support power-of-two splits with K/splitk a multiple
    of 128 and at least 512 or 768, respectively, subject to
    splitk * ceil(M/tile_m) * ceil(N/tile_n) <= 256. The shared FlyDSL reducer
    accumulates BF16 partials in FP32 and writes the public BF16 [M,N] output.
    Each partial is rounded before reduction, so results can differ from ``splitk=1``.

    K is taken from A (mxfp8, ``A.shape[1] == K``); B is packed mxfp4 with
    ``B.shape == [N, K/2]``."""
    require_gfx1250_asm("gemm_a8w4_mxfp8")
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    if dtype != dtypes.bf16:
        raise NotImplementedError(
            f"gfx1250 a8w4 MXFP8xMXFP4 GEMM: unsupported output dtype {dtype}"
        )
    if K % 128 != 0:  # A (m/2,k/128) preshuffle
        raise NotImplementedError(
            f"gfx1250 a8w4 MXFP8xMXFP4 GEMM requires K%128==0, got K={K}"
        )
    if N % 16 != 0:  # B 16x16 preshuffle
        raise NotImplementedError(
            f"gfx1250 a8w4 MXFP8xMXFP4 GEMM requires N%16==0, got N={N}"
        )
    if a_preshuffle and M % 2 != 0:  # A (m/2,k/128) preshuffle
        raise NotImplementedError(
            f"gfx1250 a8w4 MXFP8xMXFP4 GEMM a_preshuffle requires M%2==0, got M={M}"
        )
    kernelName, splitk = _resolve_mxfp8_gemm_config(
        M, N, K, a_preshuffle, dtype, kernelName, splitk, b_intype="mxfp4"
    )
    if splitk > 1:
        _mxfp8fp4_gemm_validate(
            A, B, kernelName or None, "mxfp4", int(bool(a_preshuffle)), splitk
        )
    allocate = torch.zeros if splitk > 1 else torch.empty
    out = allocate(
        (splitk, M, N) if splitk > 1 else (M, N), dtype=dtype, device=A.device
    )
    _mxfp8_mxfp4_gemm_asm(
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
