# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatcher for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA).

Operand convention (the weight is preshuffled caller-side):
    A       : [M, K]   row-major, NOT preshuffled  (fp8/fp6 = 1 byte/code, fp4 = 2 codes/byte)
    B       : preshuffled via aiter.ops.shuffle.shuffle_weight(., (16, 16))  (fp4 or fp8 weight)
    a_scale / b_scale : E8M0, each a block and a layout (flydsl_mxscale_preshuffle_gemm).
              "row" -- raw, as quantized, staged by the kernel itself (a8w8): x_scale
              1x32 [M, K//32] or the blockscale 1x128 column-major bytes of [K//128, M];
              w_scale 1x32 [N, K//32], 32x32 [N//32, K//32] or 128x128 [N//128, K//128].
              "mfma_tile" -- pre-shuffled into the MFMA operand order by the caller
              (per-1x32 shuffle_scale_a16w4'd; the coarse 1x128 / 128x128 blocks in the
              compact broadcast layout the kernel reads). No aiter op takes these any more.
    Out     : [M, N]   bf16 / fp16

``run_gemm_a8w8_mxfp8_gfx950`` is the entry point that gemm_a8w8_mxfp8_bpreshuffle,
gemm_a8w8_blockscale_bpreshuffle's "mxfp8" rows and the tuner share: raw scales,
a tuned kernelName (mxpsh or 8-wave) or the heuristic.
"""

from __future__ import annotations

import functools

import torch

# gemm_a8w8_mxfp8_bpreshuffle's gfx950 scales per w_scale block (row-major 1x32
# x_scale; 1x32 or 32x32 w_scale), defined next to its tune pipelines;
# re-exported for the runner and AOT.
from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK,
)

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}


@functools.cache
def _gemm_exe(fused):
    import flydsl.compiler as flyc

    from .kernels.gemm.mxscale_preshuffle import launch_gemm, launch_gemm_fused

    launcher = launch_gemm_fused if fused else launch_gemm
    return flyc.jit(launcher.func)


@functools.cache
def _reduce_exe():
    import flydsl.compiler as flyc

    from .kernels.gemm.mxscale_preshuffle import launch_splitk_reduce

    return flyc.jit(launch_splitk_reduce.func)


@functools.lru_cache(maxsize=1024)
def _fused_splitk_buffers(
    device: torch.device,
    stream_id: int,
    split_k: int,
    M: int,
    N: int,
    num_tiles: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream-private partials (BF16 for the coarse small-M path, FP32 for 1x32
    row scales) and per-output-tile arrival counters."""
    del stream_id
    workspace = torch.empty((split_k, M, N), dtype=dtype, device=device)
    semaphore = torch.zeros((num_tiles,), dtype=torch.int32, device=device)
    return workspace, semaphore


def flydsl_mxscale_preshuffle_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str = "fp4",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    split_k: int = 1,
    stream=None,
    a_scale_block: str,
    b_scale_block: str,
    a_scale_layout: str,
    b_scale_layout: str,
    fused_split_k: bool = False,
) -> torch.Tensor:
    """Run the gfx950 MXFP4/6/8 preshuffle GEMM. a8w8 = a_dtype="fp8", b_dtype="fp8".

    A is [M, K]; N is taken from Out ([M, N]); K from A. Returns Out.

    split_k>1 splits the K reduction across grid.z: fp32 partial slabs and a
    separate reduce kernel, or, with fused_split_k (row layouts, bf16 Out), an
    in-kernel reduce by each output tile's last-arriving block -- the same FP32
    numbers, a tunable (kernelName ``_fsk``). A mfma_tile coarse-block M<=16 tile
    reduces BF16 partials in-kernel instead.

    Each operand's scale is a block and a layout, always named by the caller (see
    flydsl_gemm_mxscale_preshuffle_common.SCALE_LAYOUTS); a default would read one
    contract's bytes as the other's:

    * a_scale_block "1x32" | "1x128", b_scale_block "1x32" | "128x128" | "32x32".
      The coarse blocks are a8w8-only; a 128x128 b_scale needs N%128==0.
    * "row" (a8w8): the UNSHUFFLED E8M0 as quantized, staged by the kernel
      itself: a 1x32 A [M, K//32] (no row padding) / B [N, K//32]; a 32x32 B
      [N//32, K//32]; a 1x128 A the blockscale column-major bytes [K//128, M]
      (a contiguous (M, K//128) tensor holding them, as bmm's transposed
      x_scale); a 128x128 B [N//128, K//128].
    * "mfma_tile": pre-shuffled into the MFMA operand order by the caller -- the
      per-1x32 scales through ``shuffle_scale_a16w4``, the coarse 1x128 / 128x128
      blocks in the compact broadcast layout the kernel reads (no aiter helper
      builds those; no aiter op passes them). The op does NOT repack them.
    """
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        SCALE_BLOCKS_A,
        SCALE_BLOCKS_B,
        SCALE_LAYOUTS,
    )
    from .kernels.tensor_shim import _run_compiled, ptr_arg

    # Logical K: fp4 A packs 2 codes/byte (A last dim = K//2); fp6/fp8 A = 1 byte/code.
    if a_dtype not in ("fp4", "fp6", "fp8"):
        raise ValueError(
            f"unsupported a_dtype {a_dtype!r}; expected 'fp4', 'fp6', or 'fp8'"
        )
    if b_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported b_dtype {b_dtype!r}; expected 'fp4' or 'fp8'")

    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    if N % int(tile_n) != 0:
        raise ValueError(f"N ({N}) is not a multiple of tile_n ({tile_n})")
    if K % int(tile_k) != 0:
        raise ValueError(f"K ({K}) is not a multiple of tile_k ({tile_k})")
    if K % 128 != 0:
        raise ValueError(
            f"K ({K}) must be a multiple of 128 for MXFP microscale; got {K}"
        )
    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(
            f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16"
        )

    if a_scale_block not in SCALE_BLOCKS_A or b_scale_block not in SCALE_BLOCKS_B:
        raise ValueError(
            f"unsupported scale blocks a={a_scale_block!r} b={b_scale_block!r}; "
            f"expected a in {SCALE_BLOCKS_A}, b in {SCALE_BLOCKS_B}"
        )
    for layout in (a_scale_layout, b_scale_layout):
        if layout not in SCALE_LAYOUTS:
            raise ValueError(f"unsupported scale layout {layout!r}; {SCALE_LAYOUTS}")
    # The mfma_tile 1x128 / 128x128 blocks broadcast to the MFMA through the
    # load address; a row (unshuffled) scale of any block is not coarse (nor
    # does it take the kernel's fused small-M blockscale path).
    coarse = (a_scale_block == "1x128" and a_scale_layout != "row") or (
        b_scale_block == "128x128" and b_scale_layout != "row"
    )
    # An unsupported shape is an error rather than a silent downgrade.
    if coarse and (a_dtype != "fp8" or b_dtype != "fp8"):
        raise ValueError(
            f"a coarse scale block is a8w8-only; got a_dtype={a_dtype!r} "
            f"b_dtype={b_dtype!r}"
        )
    if b_scale_block == "128x128" and N % 128 != 0:
        raise ValueError(f"a 128x128 b_scale needs N ({N}) a multiple of 128")
    for name, t, block, layout, rows in (
        ("a_scale", a_scale, a_scale_block, a_scale_layout, M),
        ("b_scale", b_scale, b_scale_block, b_scale_layout, N),
    ):
        if layout != "row":
            if block == "32x32":
                raise ValueError(f"a 32x32 {name} is row layout only")
            continue
        if (a_dtype, b_dtype) != ("fp8", "fp8"):
            raise ValueError(f"a row {name} is a8w8 only")
        # (rows per scale row, K per scale column) of each row block.
        rows_per, k_per = {"1x32": (1, 32), "32x32": (32, 32)}.get(block, (128, 128))
        if block == "1x128":
            rows_per = 1  # column-major bytes of a (rows, K // 128) tensor
        if rows % rows_per != 0:
            raise ValueError(
                f"a {block} {name} needs {rows} rows a multiple of {rows_per}"
            )
        want = (rows // rows_per, K // k_per)
        if tuple(t.shape) != want or not t.is_contiguous():
            raise ValueError(
                f"row {name} must be a contiguous {want}, got "
                f"{tuple(t.shape)} stride {tuple(t.stride())}"
            )
    split_k = int(split_k)
    # For the latency-sensitive M<=16 path, keep the split partials in BF16 and
    # let the last arriving GEMM block reduce them. This avoids a second launch
    # without introducing output atomics or a grid-wide spin wait.
    splitk_fused = (
        coarse
        and split_k > 1
        and M <= 16
        and int(tile_m) == 16
        and out_dtype == "bf16"
        and int(tile_n) <= 128
    )
    # fused_split_k: row-layout scales reduce in-kernel at any tile, from FP32
    # partials (the kernel's fused_f32; it must agree, as the fused launch ABI
    # differs).
    fused_f32 = (
        bool(fused_split_k)
        and split_k > 1
        and (a_scale_layout, b_scale_layout) == ("row", "row")
        and out_dtype == "bf16"
    )
    splitk_fused = splitk_fused or fused_f32

    st = stream if stream is not None else torch.cuda.current_stream()

    if split_k > 1:
        # split-K legality (same constraints the tuner enforces in fits_shape):
        # per-split K must be a whole number of tile_k tiles AND 256-K scale chunks.
        k_per_split = K // split_k
        if K % split_k != 0 or k_per_split % int(tile_k) != 0 or k_per_split % 256 != 0:
            raise ValueError(
                f"illegal split_k={split_k} for K={K}, tile_k={tile_k}: "
                f"K/split_k ({k_per_split}) must be a multiple of tile_k and 256"
            )

    # Constexpr tail of launch_gemm -- the same for the direct and the split-K launch
    # apart from k_batch, and it doubles as the per-config compiled-kernel cache key.
    cfg = (
        N,
        K,
        int(tile_m),
        int(tile_n),
        int(tile_k),
        a_dtype,
        out_dtype,
        b_dtype,
        1,  # batch
        -1,  # a_row_stride     ) each <0 keeps the contiguous
        -1,  # a_batch_stride   ) [B,M,*] default; this op never
        -1,  # sca_row_stride   ) overrides them, the batched
        -1,  # sca_batch_stride ) callers do
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
        int(xcd_swizzle),
        split_k,  # k_batch
        a_scale_block,
        b_scale_block,
    )
    multi_row = int(tile_m) == 16 and M > 1
    layouts = (a_scale_layout, b_scale_layout)  # launch_gemm's last Constexprs
    gemm_exe = _gemm_exe(splitk_fused)
    # launch_gemm's Constexpr tail but fused_split_k, which each launch appends.
    spec = (*cfg, multi_row, *layouts)
    # Build each runtime pointer wrapper once per op.  The fused ABI carries two
    # additional pointer slots; re-wrapping Out (or the split workspace) for
    # those aliases measurably increases launch gaps on very short decode GEMMs.
    a_ptr = ptr_arg(A)
    b_ptr = ptr_arg(B)
    a_scale_ptr = ptr_arg(a_scale)
    b_scale_ptr = ptr_arg(b_scale)
    out_ptr = ptr_arg(Out)

    if split_k == 1:
        _run_compiled(
            gemm_exe,
            out_ptr,
            a_ptr,
            b_ptr,
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            st,
            *spec,
            False,  # fused_split_k
            specialization_key=(*spec, False),
        )
        return Out

    if splitk_fused:
        num_tiles = ((M + int(tile_m) - 1) // int(tile_m)) * (N // int(tile_n))
        if isinstance(st, torch.cuda.Stream):
            with torch.cuda.stream(st):
                workspace, semaphore = _fused_splitk_buffers(
                    A.device,
                    int(st.cuda_stream),
                    split_k,
                    M,
                    N,
                    num_tiles,
                    torch.float32 if fused_f32 else torch.bfloat16,
                )
        else:
            # Compile-only/AOT uses an fx.Stream placeholder and CPU/Fake tensors.
            workspace, semaphore = _fused_splitk_buffers(
                A.device,
                0,
                split_k,
                M,
                N,
                num_tiles,
                torch.float32 if fused_f32 else torch.bfloat16,
            )
        _run_compiled(
            gemm_exe,
            ptr_arg(workspace),
            a_ptr,
            b_ptr,
            a_scale_ptr,
            b_scale_ptr,
            out_ptr,
            ptr_arg(semaphore),
            M,
            N,
            st,
            *spec,
            fused_f32,  # fused_split_k
            specialization_key=(*spec, fused_f32),
        )
        return Out

    # split-K: GEMM -> fp32 partial slabs tmp[split_k, M, N] -> fused fp32 reduce -> Out.
    tmp = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    if isinstance(st, torch.cuda.Stream):
        tmp.record_stream(st)
    tmp_ptr = ptr_arg(tmp)
    _run_compiled(
        gemm_exe,
        tmp_ptr,
        a_ptr,
        b_ptr,
        a_scale_ptr,
        b_scale_ptr,
        M,
        N,
        st,
        *spec,
        False,  # fused_split_k
        specialization_key=(*spec, False),
    )
    _run_compiled(
        _reduce_exe(),
        tmp_ptr,
        out_ptr,
        (M * N) // 2,  # n_out_dw (2 out elems per dword)
        M * N,  # slab_stride_dw (fp32: 1 dword/elem)
        st,
        split_k,
        out_dtype,
        specialization_key=(split_k, out_dtype),
    )
    return Out


_HEURISTIC_MIN_TILE_N = 32


@functools.lru_cache(maxsize=1024)
def _warn_untuned(M, N, K, a_dtype, b_dtype, kernel_name):
    from aiter import logger

    logger.warning(
        f"[flydsl mxpsh] no tuned row for M={M} N={N} K={K} {a_dtype}/{b_dtype}; "
        f"falling back to the heuristic tile '{kernel_name}'. It picks a grid that "
        f"fills the CUs, but nothing about this shape was measured -- tune it with "
        f"csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py."
    )


@functools.lru_cache(maxsize=1024)
def _heuristic_tile(a_dtype, b_dtype, M, N, K, scales):
    from aiter.jit.utils.chip_info import get_cu_num

    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import candidates_for

    # scales: candidates_for's scale kwargs as sorted (name, value) pairs, which
    # keep the call hashable for the cache.
    # tile_k=512 and _fsk are tuned-only options: the heuristic keeps its
    # untuned picks.
    cands = [
        ki
        for _, ki in candidates_for(a_dtype, b_dtype, M, N, K, **dict(scales))
        if ki.tile_k <= 256 and not ki.fused_split_k
    ]
    if not cands:
        return None

    def _wg(ki):
        return -(-M // ki.tile_m) * (N // ki.tile_n) * ki.split_k

    pool = [ki for ki in cands if ki.tile_n >= _HEURISTIC_MIN_TILE_N] or cands
    filled = [ki for ki in pool if _wg(ki) >= get_cu_num()]
    if filled:
        pool = filled
    else:  # N too narrow, or split-K legality leaves nothing that fills the CUs
        best_wg = max(_wg(ki) for ki in pool)
        pool = [ki for ki in pool if _wg(ki) == best_wg]

    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        pool,
        key=lambda ki: (
            -ki.split_k,
            ki.tile_k,
            ki.tile_n,
            -abs(ki.tile_m - target_m),
            -ki.waves_per_eu,
        ),
    )


def run_gemm_a8w8_mxfp8_gfx950(
    XQ, WQ, x_scale, w_scale, Out, kernel_name=None, w_scale_block=None
):
    """gfx950 FP8 GEMM with unshuffled e8m0 scales, into ``Out``.

    XQ [M, K] fp8 row-major; WQ [N, K] fp8, shuffle_weight(., (16, 16)).
    x_scale [M, K // 32] and w_scale e8m0, row-major and NOT shuffled (no row
    padding): the kernel stages them through LDS itself. w_scale is
    [N, K // 32] for a 1x32 ``w_scale_block`` or [N // 32, K // 32] for 32x32
    (bmm's); None reads the block off its shape. ``kernel_name`` is a tuned
    row's mxpsh or 8-wave kernelName, else the heuristic picks an mxpsh tile.
    """
    from ..gemm_op_common import mxscale_w_scale_block
    from .gemm_a8w8_mxfp8_8wave import NAME_PREFIX as MXFP8_8WAVE_PREFIX
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import parse_kernel_name

    if w_scale_block is None:
        w_scale_block = mxscale_w_scale_block(
            tuple(w_scale.shape), int(WQ.shape[0]), int(WQ.shape[-1])
        )
    scales = MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK.get(w_scale_block)
    if scales is None:
        raise NotImplementedError(
            f"[FlyDSL gfx950 mxfp8] no kernel reads a {w_scale_block} w_scale; "
            f"supported: {sorted(MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK)}"
        )

    if kernel_name is not None and kernel_name.startswith(f"{MXFP8_8WAVE_PREFIX}_"):
        if Out.dtype == torch.bfloat16:
            from .gemm_a8w8_mxfp8_8wave import run_gemm_a8w8_mxfp8_8wave

            return run_gemm_a8w8_mxfp8_8wave(
                XQ, WQ, x_scale, w_scale, Out, kernel_name, w_scale_block
            )
        # The 8-wave kernel writes bf16 only; any other Out runs the mxpsh
        # heuristic tile for this (tuned) shape instead.
        kernel_name = None
        warn_untuned = False
    else:
        warn_untuned = kernel_name is None

    for name, t in (("XQ", XQ), ("WQ", WQ), ("Out", Out)):
        if not t.is_contiguous():
            raise RuntimeError(
                f"[FlyDSL gfx950 mxfp8] {name} must be contiguous, got shape "
                f"{tuple(t.shape)} stride {tuple(t.stride())}"
            )
    M, K = int(XQ.shape[0]), int(XQ.shape[-1])
    N = int(Out.shape[-1])
    if kernel_name is None:
        ki = _heuristic_tile("fp8", "fp8", M, N, K, scales)
        if ki is None:
            raise NotImplementedError(
                f"[FlyDSL gfx950 mxfp8] no kernel for M={M}, N={N}, K={K} with "
                f"a {w_scale_block} w_scale (needs K%128==0 and N a multiple of a "
                f"tile_n)"
            )
        kernel_name = ki.name
        if warn_untuned:
            _warn_untuned(M, N, K, "fp8", "fp8", kernel_name)
    p = parse_kernel_name(kernel_name)
    if p is None or (p["a_dtype"], p["b_dtype"]) != ("fp8", "fp8"):
        raise ValueError(
            f"[FlyDSL gfx950 mxfp8] {kernel_name!r} is not an a8w8 mxpsh kernel"
        )
    # flydsl_mxscale_preshuffle_gemm checks the row scale shapes.
    return flydsl_mxscale_preshuffle_gemm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=p["tile_m"],
        tile_n=p["tile_n"],
        tile_k=p["tile_k"],
        waves_per_eu=p["waves_per_eu"],
        xcd_swizzle=p["xcd_swizzle"],
        split_k=p["split_k"],
        fused_split_k=p["fused_split_k"],
        **dict(scales),
    )


def pick_mxfp8_kernel_name_gfx950(M, N, K, w_scale_block="1x32"):
    """The kernelName run_gemm_a8w8_mxfp8_gfx950 runs for an untuned shape."""
    ki = _heuristic_tile(
        "fp8", "fp8", M, N, K, MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK[w_scale_block]
    )
    return None if ki is None else ki.name
