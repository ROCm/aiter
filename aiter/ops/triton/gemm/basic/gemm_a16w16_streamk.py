# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._gluon_kernels.gfx1250.layouts import (
    WMMA_INSTR_SHAPE,
    operand_bundle_cga,
    wmma_layout,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.common_utils import deserialize_str, serialize_dict
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_SUPPORTED_ARCHS = ("gfx1250",)

# wmma takes 16-bit operands and accumulates in fp32.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_OPERAND_BYTES = 2
#: wmma K step; a drain narrower than this cannot be sliced.
_WMMA_K = WMMA_INSTR_SHAPE[2]

# Floors for a remainder sub-tile. These are hardware widths, not tuned values, so they stay
# here and not in the config, like TDM_PAD_INTERVAL_MAX in _gluon_kernels/gfx1250/layouts.py.
# A TDM transfer and the aliased C store want a 128-byte row, which is 64 elements of a 16-bit
# operand. Below that the row halves to 64 bytes: a per-CTA N of 32 measured -11.2% against 64
# at an equal sub-tile area and an equal sub-tile count, while 128 against 64 measured -0.3%.
_SPLIT_FLOOR_N = 128 // _OPERAND_BYTES
# M only has to fill the warp grid, and a per-CTA M of 32 measures well.
_SPLIT_FLOOR_M = 32


def _plan_k_pipeline(K, BLOCK_K, num_buffers):
    """Depth of the K pipeline, and whether its steady loop runs at all.

    Returns `(num_k_tiles, num_buffers, steady_nonempty)`. The steady loop needs a full ring
    before it runs, so the depth clamps to the K tile count, but it keeps a floor of 2. One K
    tile therefore leaves the steady loop empty, and the kernel must not then state that its
    trip count is above 0: that is UB, and it miscompiles the whole kernel. The unit test calls
    this too, so a shape meant to cover the empty loop cannot quietly stop covering it.
    """
    num_k_tiles = triton.cdiv(K, BLOCK_K)
    num_buffers = max(2, min(num_buffers, num_k_tiles))
    steady_nonempty = 1 if num_k_tiles - (num_buffers - 1) > 0 else 0
    return num_k_tiles, num_buffers, steady_nonempty


def _plan_tiles(M, N, BLOCK_M, BLOCK_N, CTAS_M, CTAS_N):
    """Map a shape onto the workers: how many tiles, how many workers, what the remainder gets.

    Returns `(num_tiles, num_sms, streamk_tiles, split_m, split_n)`. `streamk_tiles` is 0 when
    phase 2 does not run. The unit test calls this too, so the test cannot drift from the
    dispatch that the launch really uses.
    """
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # get_num_sms honors the CU_NUM override, which is also what the config dispatch keys on,
    # so the worker count stays consistent with the selected config.
    num_sms = min(get_num_sms() // (CTAS_M * CTAS_N), num_tiles)

    # The tiles left over after the whole waves. There is nothing to balance until the machine
    # holds at least one full wave, so a shape with too few tiles gets zero.
    streamk_tiles = 0 if num_tiles <= num_sms else num_tiles % num_sms
    split_m, split_n = _pick_split(
        streamk_tiles, num_sms, BLOCK_M, BLOCK_N, CTAS_M, CTAS_N
    )
    if split_m * split_n == 1:
        streamk_tiles = (
            0  # one sub-tile for each tile is the phase-1 path, keep it there
        )
    return num_tiles, num_sms, streamk_tiles, split_m, split_n


def _pick_split(streamk_tiles, num_sms, BLOCK_M, BLOCK_N, CTAS_M, CTAS_N):
    """Smallest split that fills the machine with the remainder sub-tiles.

    Split M before N: narrowing N shrinks the per-CTA N width, which feeds the TDM pad interval
    and the aliased C store. A sub-tile must stay divisible by the cluster and keep enough
    elements for each CTA on the axis that it splits.

    The two axes have different floors, `_SPLIT_FLOOR_M` and `_SPLIT_FLOOR_N`. Both come from a
    hardware width; see the comment where they are set.
    """
    sm, sn = 1, 1
    if not streamk_tiles:
        return sm, sn

    def ok(blk, split, ctas, floor):
        return blk % (split * ctas) == 0 and (blk // split) // ctas >= floor

    # Split only while the sub-tiles still fit ONE wave. Going past num_sms turns one partly
    # idle wave into two full ones of half-size tiles, which is slower: at 12 left-over tiles on
    # 16 workers a 2x split gives 24 sub-tiles, so 1 wave becomes 2 and the shape loses 6%.
    while streamk_tiles * sm * sn * 2 <= num_sms and ok(
        BLOCK_M, sm * 2, CTAS_M, _SPLIT_FLOOR_M
    ):
        sm *= 2
    while streamk_tiles * sm * sn * 2 <= num_sms and ok(
        BLOCK_N, sn * 2, CTAS_N, _SPLIT_FLOOR_N
    ):
        sn *= 2
    return sm, sn


@functools.lru_cache(maxsize=128)
def _build_layouts(BLOCK_M, BLOCK_N, BLOCK_K, ctas_per_cga, num_warps, TRANSPOSE_B):
    """Build the WMMA, operand and shared layouts for one cluster-wide tile.

    Cached: the layout objects cost more to build than the launch itself, and they depend only on
    the config, not on the shape.
    """
    per_cta_m = BLOCK_M // ctas_per_cga[0]
    per_cta_n = BLOCK_N // ctas_per_cga[1]

    wmma = wmma_layout((per_cta_m, per_cta_n), ctas_per_cga, num_warps)
    dot_a, shared_a = operand_bundle_cga(wmma, 0, (per_cta_m, BLOCK_K), ctas_per_cga)
    if TRANSPOSE_B:
        dot_b, shared_b = operand_bundle_cga(
            wmma, 1, (per_cta_n, BLOCK_K), ctas_per_cga, transpose_b=True
        )
    else:
        dot_b, shared_b = operand_bundle_cga(
            wmma, 1, (BLOCK_K, per_cta_n), ctas_per_cga
        )
    return wmma, dot_a, dot_b, shared_a, shared_b


def gemm_a16w16_streamk_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    if y is not None:
        return y
    return torch.empty(
        (x.shape[0], w.shape[0]), dtype=dtype or x.dtype, device=x.device
    )


@torch_compile_guard(gen_fake=gemm_a16w16_streamk_fake_tensor)
def gemm_a16w16_streamk_(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    """WG cluster StreamK GEMM on gfx1250: y = x @ w.T, one tile per workgroup cluster.

    x is (M, K) row major and w is (N, K). A K-major w (w.stride(1) == 1) keeps B in its
    transposed form; an N-major w takes the native path, which also unlocks the aliased C store.
    """
    assert any(
        a in get_arch() for a in _SUPPORTED_ARCHS
    ), f"gemm_a16w16_streamk requires one of {_SUPPORTED_ARCHS}, got '{get_arch()}'"
    assert x.shape[1] == w.shape[1], f"incompatible shapes {x.shape} and {w.shape}"
    assert x.stride(1) == 1, f"x must be row major (M, K), got strides {x.stride()}"

    # wmma is built for 16-bit operands, so an unsupported dtype must stop here. Without this
    # check it reaches the layout builders and fails with a compiler message about tt.dot.
    assert x.dtype in _SUPPORTED_DTYPES, f"x must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in _SUPPORTED_DTYPES, f"w must be fp16 or bf16, got {w.dtype}"
    assert (
        x.dtype == w.dtype
    ), f"x and w must share a dtype, got {x.dtype} and {w.dtype}"

    M, K = x.shape
    N = w.shape[0]
    if dtype is None:
        dtype = x.dtype
    assert dtype in _SUPPORTED_DTYPES + (
        torch.float32,
    ), f"the output must be fp16, bf16 or fp32, got {dtype}"

    cfg = deserialize_str(config) if config else None
    if cfg is None:
        cfg, _ = get_gemm_config("GEMM-A16W16-STREAMK", M, N, K, backend="gluon")

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    NUM_BUFFERS = cfg["NUM_BUFFERS"]
    GROUP_SIZE_M = cfg["GROUP_SIZE_M"]
    CTAS_M = cfg["CTAS_M"]
    CTAS_N = cfg["CTAS_N"]
    num_warps = cfg["num_warps"]
    waves_per_eu = cfg["waves_per_eu"]

    # BLOCK_M and BLOCK_N are cluster-wide, so each CTA must get a whole sub-tile.
    assert (
        BLOCK_M % CTAS_M == 0 and BLOCK_N % CTAS_N == 0
    ), f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} must divide by the cluster {CTAS_M}x{CTAS_N}"
    num_ctas = CTAS_M * CTAS_N

    # The kernel swaps the two strides itself for a transposed B, so it always gets the (K, N)
    # pair; only the orientation flag changes.
    w_t = w.T
    if w_t.stride(1) == 1:
        TRANSPOSE_B = False
    elif w_t.stride(0) == 1:
        TRANSPOSE_B = True
    else:
        raise ValueError(
            f"w must be contiguous in one dimension, got strides {w.stride()}"
        )
    stride_bk, stride_bn = w_t.stride(0), w_t.stride(1)

    _, NUM_BUFFERS, STEADY_NONEMPTY = _plan_k_pipeline(K, BLOCK_K, NUM_BUFFERS)

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    else:
        # The kernel writes a whole (M, N) tile grid, so a y that is too small is overrun. The
        # TDM store also addresses C by row, which needs a unit stride on N.
        assert tuple(y.shape) == (
            M,
            N,
        ), f"y must be ({M}, {N}), got {tuple(y.shape)}"
        assert y.dtype == dtype, f"y must be {dtype}, got {y.dtype}"
        assert y.device == x.device, f"y must be on {x.device}, got {y.device}"
        assert y.stride(1) == 1, f"y must be row major (M, N), got strides {y.stride()}"
        # Rows must not alias. The TDM store takes the innermost stride as 1 and addresses C
        # by row, so a view whose row stride is below N (an expand, or an as_strided overlap)
        # writes several rows onto the same memory and gives a wrong result instead of failing.
        assert (
            M == 1 or y.stride(0) >= N
        ), f"y rows must not overlap, got shape ({M}, {N}) strides {y.stride()}"

    wmma, dot_a, dot_b, shared_a, shared_b = _build_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, (CTAS_M, CTAS_N), num_warps, TRANSPOSE_B
    )

    num_tiles, NUM_SMS, STREAMK_TILES, SPLIT_M, SPLIT_N = _plan_tiles(
        M, N, BLOCK_M, BLOCK_N, CTAS_M, CTAS_N
    )
    rem = (
        _build_layouts(
            BLOCK_M // SPLIT_M,
            BLOCK_N // SPLIT_N,
            BLOCK_K,
            (CTAS_M, CTAS_N),
            num_warps,
            TRANSPOSE_B,
        )
        if STREAMK_TILES
        else (None,) * 5
    )
    wmma_r, dot_r_a, dot_r_b, shared_r_a, shared_r_b = rem

    # KTAIL is the width of the last K step when K does not fill BLOCK_K, rounded up to the
    # wmma K step. The TDM load zero-fills the rest, so a full-width drain wmma multiplies the
    # fill: at K=2880 with BLOCK_K=256 the tail is 64, and a full 256-wide drain throws away 75
    # percent of one step of 12. It is decided here, from the real K, so the compiler builds
    # only one drain path; testing K inside the kernel compiles both and costs even the shapes
    # that do not take the tail.
    _k_rem = K % BLOCK_K
    KTAIL = triton.cdiv(_k_rem, _WMMA_K) * _WMMA_K if _k_rem else 0

    # The masked C store indexes C by an absolute element offset, which overflows int32 past
    # 2**31 elements. Switch that store to a 64-bit base only when C can reach it: the 64-bit
    # address maths costs up to 18% on this path.
    C_OFFSET_64 = 1 if (M - 1) * y.stride(0) + (N - 1) * y.stride(1) >= 2**31 else 0

    _LOGGER.info("GEMM_A16W16_STREAMK [gluon/gfx1250]: x=%s w=%s", x.shape, w.shape)

    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16_streamk import (
        gemm_a16w16_streamk_kernel_,
    )

    gemm_a16w16_streamk_kernel_[(NUM_SMS,)](
        x,
        w_t,
        y,
        M,
        N,
        K,
        num_tiles,
        x.stride(0),
        x.stride(1),
        stride_bk,
        stride_bn,
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        TRANSPOSE_B=TRANSPOSE_B,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        OPERAND_LAYOUT_A=dot_a,
        OPERAND_LAYOUT_B=dot_b,
        WMMA_LAYOUT=wmma,
        NUM_SMS=NUM_SMS,
        CTAS_N=CTAS_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        KTAIL=KTAIL,
        STEADY_NONEMPTY=STEADY_NONEMPTY,
        C_OFFSET_64=C_OFFSET_64,
        STREAMK_TILES=STREAMK_TILES,
        SPLIT_M=SPLIT_M,
        SPLIT_N=SPLIT_N,
        SHARED_REM_A=shared_r_a,
        SHARED_REM_B=shared_r_b,
        OPERAND_REM_A=dot_r_a,
        OPERAND_REM_B=dot_r_b,
        WMMA_REM=wmma_r,
        num_warps=num_warps,
        num_ctas=num_ctas,
        waves_per_eu=waves_per_eu,
    )
    return y


def gemm_a16w16_streamk(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Public entry point. See gemm_a16w16_streamk_ for the argument contract."""
    return gemm_a16w16_streamk_(
        x, w, dtype, y, serialize_dict(config) if config else None
    )
