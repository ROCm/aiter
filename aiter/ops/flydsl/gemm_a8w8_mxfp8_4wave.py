# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""4-wave CDNA4 (gfx950) backend for the FlyDSL mxfp8 (1x32 e8m0) bpreshuffle GEMM.

The persistent 256x256 4-wave kernel (``kernels/mxfp8_gemm_4wave.py``) behind the same
operands as the 8-wave backend (row-major fp8 A, ``shuffle_weight(.., (16, 16))`` W, raw
row-major e8m0 scales): the scales are brought into the kernel's ``shuffle_scale_a16w4``
order here (A's per call over rows padded to 32, W's once per weight scale tensor). Rows
and launch plan follow amd-kernels' ``mxfp8_common/gemm.py`` (``_fourwave_plan``): the
remainder of a partial last round of 256x256 tiles runs as one round of 128x128 / 128x256
sub-tiles.
"""

from __future__ import annotations

import functools
import re
from collections import OrderedDict

import torch
import triton
import triton.language as tl
from torch import Tensor

# Fixed by the kernel: MFMA_Scale(16, 16, 128) over a 128-deep K step, a 256x256 parent
# tile, scales gathered a 256-K block at a time (K >= 4 steps for the pipelined K loop).
BLOCK_K = 128
BLOCK_M = BLOCK_N = 256
K_ALIGN = 256
K_MIN = 512
SCALE_GROUP = 32
_I32_MAX = 2**31

NAME_PREFIX = "flydsl_mxfp8_4w"
# flydsl_mxfp8_4w_{BM}x{BN}x{BK}_{QA}_{QW}_{OUT}_xcd{0|1}[_nt]
_KERNEL_NAME_RE = re.compile(
    rf"^{NAME_PREFIX}_"
    r"(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"xcd(?P<xcd>[01])(?P<nt>_nt)?$"
)


def make_mxfp8_4wave_kernel_name(xcd_remap: bool, store_nt: bool = False) -> str:
    return (
        f"{NAME_PREFIX}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_B16_"
        f"xcd{int(bool(xcd_remap))}{'_nt' if store_nt else ''}"
    )


def parse_mxfp8_4wave_kernel_name(name: str) -> dict | None:
    """Parse a ``flydsl_mxfp8_4w_`` kernelName into its config dict, or None."""
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
        "xcd_remap": g["xcd"] == "1",
        "store_nt": g["nt"] is not None,
    }


def fits_shape(M: int, N: int, K: int) -> bool:
    """The kernel's shape contract: M ragged (rows beyond M read zero through the buffer
    bounds and their stores are dropped), N whole 256-column tiles (the output store has no
    column guard), K whole 256-K scale blocks with at least four 128-K steps, and 32-bit
    buffer indexing (C is bf16, so its byte count doubles)."""
    if M <= 0 or N <= 0 or N % BLOCK_N != 0:
        return False
    if K < K_MIN or K % K_ALIGN != 0:
        return False
    return M * N * 2 < _I32_MAX and M * K < _I32_MAX and N * K < _I32_MAX


@functools.lru_cache(maxsize=256)
def compile_mxfp8_4wave_gemm(
    *, K: int, xcd_remap: bool, store_nt: bool, tiles: tuple, grid_max: int
):
    """Compile (and memoize) a 4-wave launcher; ``tiles`` = (n_a, n_b) 16-row / 16-column
    MFMA tiles per wave quadrant: (4, 4) the 256x256 tile, (2, 4) / (2, 2) the sub-tiles of
    the remainder round."""
    from .kernels.mxfp8_gemm_4wave import compile_mxfp8_gemm_4w

    return compile_mxfp8_gemm_4w(
        K=K,
        use_xcd_remap=bool(xcd_remap),
        store_nt=bool(store_nt),
        tiles=tuple(tiles),
        grid_max=int(grid_max),
    )


def launch_plan(
    M: int, N: int, cus: int, xcd_remap: bool
) -> list[tuple[tuple, int, int]]:
    """The launches of one GEMM: (tiles, tile_off, n_work) over the 256x256 parent grid's
    index space. One round of ``cus`` parent tiles per launch; a remainder of R <= cus / 2
    parents (a round with idle CUs) runs as one round of sub-tiles instead: 128x128 for
    R <= cus / 4, 128x256 for R <= cus / 2 (amd-kernels measured qkv M 8192: 127 -> 109 us,
    16384: 205 -> 195)."""
    parents = -(-M // BLOCK_M) * -(-N // BLOCK_N)
    r = parents % cus
    if parents > cus and 0 < r <= cus // 2:
        tail = (2, 2) if r <= cus // 4 else (2, 4)
        tail_split = (BLOCK_M // (64 * tail[0])) * (BLOCK_N // (64 * tail[1]))
        # a sub-tile launch deals its parents out by XCD, 8 at a time; the parents beyond
        # the grid are dropped by the buffer bounds
        n_tail = (r + 7) // 8 * 8 * tail_split if xcd_remap else r * tail_split
        return [((4, 4), 0, parents - r), (tail, parents - r, n_tail)]
    return [((4, 4), 0, parents)]


# W-scale shuffles, one per scale tensor: (device, ptr, shape, stride, version) -> (the
# tensor, its shuffle). The entry holds the tensor, so its storage cannot be freed and
# reused under the key; an in-place update changes the version and misses.
_W_SCALE_CACHE: "OrderedDict[tuple, tuple[Tensor, Tensor]]" = OrderedDict()
_W_SCALE_CACHE_MAX = 512


@triton.jit
def _shuffle_scale_kernel(src_ptr, dst_ptr, rows, cols, k1, tiles, TILES: tl.constexpr):
    """dst[N1, K1, 4 (k lane), 16 (n lane), 2 (k pack), 2 (n pack)] = src[N1, 2, 16, K1, 2, 4]
    over 32-row groups (``shuffle_scale_a16w4``'s order for one expert, not gate/up); rows
    >= ``rows`` read as 127 (2^0), the padding of the last 32-row group. One program moves
    TILES tiles of 32 rows x 8 columns (256 contiguous output bytes each)."""
    e = tl.arange(0, 256)
    # the element's place in the tile: (k_l, n_l, k_p, n_p) are the output's fastest 4 digits
    n_p = e % 2
    k_p = (e // 2) % 2
    n_l = (e // 4) % 16
    k_l = e // 64
    r = n_p * 16 + n_l
    c = k_p * 4 + k_l
    for t in range(TILES):
        tile = (tl.program_id(0) * TILES + t).to(tl.int64)
        n1 = tile // k1
        kk = tile % k1
        row = n1 * 32 + r
        live = tile < tiles
        v = tl.load(
            src_ptr + row * cols + kk * 8 + c, mask=live & (row < rows), other=127
        )
        tl.store(dst_ptr + tile * 256 + e, v, mask=live)


def _shuffled_scale(scale: Tensor) -> Tensor:
    """Row-major e8m0 [rows, K / 32] -> the kernel's ``shuffle_scale_a16w4`` order, the rows
    padded to 32 with 127 (2^0), flattened to bytes. One Triton kernel (the torch version is
    a padded 7-D permute copy, 5-13 us at M 1024-16384, about 3x slower)."""
    rows, cols = scale.shape
    padded = -(-rows // 32) * 32
    assert cols % 8 == 0, cols  # K % 256 == 0
    out = torch.empty(padded * cols, dtype=torch.uint8, device=scale.device)
    tiles = out.numel() // 256
    _shuffle_scale_kernel[(triton.cdiv(tiles, 4),)](
        scale.view(torch.uint8), out, rows, cols, cols // 8, tiles, TILES=4
    )
    return out


def _w_scale_shuffled(w_scale: Tensor) -> Tensor:
    key = (
        w_scale.device,
        w_scale.data_ptr(),
        tuple(w_scale.shape),
        tuple(w_scale.stride()),
        # inference tensors (created under torch.inference_mode(), e.g. weights loaded there)
        # have no version counter; they cannot be updated in place outside inference mode
        # either, so the pointer + the held tensor identify the content
        -1 if w_scale.is_inference() else w_scale._version,
    )
    hit = _W_SCALE_CACHE.get(key)
    if hit is not None:
        _W_SCALE_CACHE.move_to_end(key)
        return hit[1]
    shuffled = _shuffled_scale(w_scale)
    _W_SCALE_CACHE[key] = (w_scale, shuffled)
    if len(_W_SCALE_CACHE) > _W_SCALE_CACHE_MAX:
        _W_SCALE_CACHE.popitem(last=False)
    return shuffled


def flydsl_mxfp8_4wave_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    *,
    xcd_remap: bool = True,
    store_nt: bool = False,
) -> Tensor:
    """Run the 4-wave mxfp8 GEMM; writes into ``Out`` and returns it.

    XQ ``(M, K)`` fp8 e4m3; WQ ``(N, K)`` fp8 e4m3 in ``shuffle_weight`` ``(16, 16)``
    layout; x_scale ``[M, K/32]`` / w_scale ``[N, K/32]`` e8m0, row-major and unshuffled
    (the 8-wave backend's operands); Out ``(M, N)`` contiguous bf16. N a multiple of 256,
    K a multiple of 256 and >= 512, M any. The first call with a given w_scale tensor
    shuffles it (a few kernels): warm it before a CUDA-graph capture.
    """
    if XQ.dim() != 2 or WQ.dim() != 2:
        raise ValueError(
            f"[FlyDSL mxfp8 4wave] A/B must be 2-D, got {tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise ValueError("[FlyDSL mxfp8 4wave] A/B must be 1-byte fp8 storage")
    if Out.dtype != torch.bfloat16 or not Out.is_contiguous():
        raise ValueError("[FlyDSL mxfp8 4wave] Out must be a contiguous bf16 tensor")
    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1] or tuple(Out.shape) != (M, N):
        raise ValueError(
            f"[FlyDSL mxfp8 4wave] shapes A {tuple(XQ.shape)}, B {tuple(WQ.shape)}, "
            f"Out {tuple(Out.shape)} do not agree"
        )
    if not fits_shape(M, N, K):
        raise ValueError(
            f"[FlyDSL mxfp8 4wave] needs N % {BLOCK_N} == 0, K % {K_ALIGN} == 0, "
            f"K >= {K_MIN}, 32-bit buffer sizes; got M={M} N={N} K={K}"
        )
    for name, t, rows in (("x_scale", x_scale, M), ("w_scale", w_scale, N)):
        if tuple(t.shape) != (rows, K // SCALE_GROUP):
            raise ValueError(
                f"[FlyDSL mxfp8 4wave] {name} must be ({rows}, {K // SCALE_GROUP}) "
                f"row-major e8m0, got {tuple(t.shape)}"
            )
        if t.element_size() != 1 or not t.is_contiguous():
            raise ValueError(
                f"[FlyDSL mxfp8 4wave] {name} must be contiguous 1-byte e8m0"
            )
    if any(t.device != XQ.device for t in (WQ, x_scale, w_scale, Out)):
        raise ValueError("[FlyDSL mxfp8 4wave] every operand must be on A's device")
    if M == 0:
        return Out

    with torch.cuda.device(XQ.device):
        return launch_mxfp8_4wave_shuffled(
            XQ,
            WQ,
            _shuffled_scale(x_scale),
            _w_scale_shuffled(w_scale),
            Out,
            xcd_remap=xcd_remap,
            store_nt=store_nt,
        )


def launch_mxfp8_4wave_shuffled(
    XQ: Tensor,
    WQ: Tensor,
    x_scale_shuffled: Tensor,
    w_scale_shuffled: Tensor,
    Out: Tensor,
    *,
    xcd_remap: bool = True,
    store_nt: bool = False,
) -> Tensor:
    """The launches of one GEMM on scales already in the kernel's order (flat uint8, rows
    padded to 32): for a producer that emits them (a quantizer) and for the wrappers above.
    Operands are not re-validated beyond ``fits_shape``."""
    from aiter.jit.utils.chip_info import get_cu_num

    from .kernels.tensor_shim import _run_compiled, current_stream_arg

    M, K = XQ.shape
    N = WQ.shape[0]
    cus = get_cu_num()
    a = XQ.contiguous().view(torch.int8).view(-1)
    w = WQ.contiguous().view(torch.int8).view(-1)
    c = Out.view(-1)
    xs = x_scale_shuffled.view(torch.uint8).view(-1)
    ws = w_scale_shuffled.view(torch.uint8).view(-1)
    stream = current_stream_arg(XQ.device)
    for tiles, tile_off, n_work in launch_plan(M, N, cus, xcd_remap):
        _run_compiled(
            compile_mxfp8_4wave_gemm(
                K=K, xcd_remap=xcd_remap, store_nt=store_nt, tiles=tiles, grid_max=cus
            ),
            a, w, c, xs, ws, M, N, tile_off, n_work, stream,
            specialization_key=XQ.device,
        )  # fmt: skip
    return Out


def run_gemm_a8w8_mxfp8_4wave(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
    w_scale_block: str = "1x32",
) -> Tensor:
    """Dispatch entry: decode a tuned mxfp8 4wave kernelName and run the kernel."""
    cfg = parse_mxfp8_4wave_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL mxfp8 4wave] unrecognised kernelName: {kernel_name!r}"
        )
    if (cfg["block_m"], cfg["block_n"], cfg["block_k"]) != (BLOCK_M, BLOCK_N, BLOCK_K):
        raise ValueError(
            f"[FlyDSL mxfp8 4wave] the kernel fixes a {BLOCK_M}x{BLOCK_N}x{BLOCK_K} tile: "
            f"{kernel_name!r}"
        )
    if w_scale_block != "1x32":
        raise ValueError(
            f"[FlyDSL mxfp8 4wave] reads a 1x32 w_scale only, got {w_scale_block!r}"
        )
    return flydsl_mxfp8_4wave_gemm(
        XQ, WQ, x_scale, w_scale, Out,
        xcd_remap=cfg["xcd_remap"], store_nt=cfg["store_nt"],
    )  # fmt: skip
