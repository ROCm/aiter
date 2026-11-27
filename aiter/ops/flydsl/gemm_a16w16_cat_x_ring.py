# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host op: ``out = concat(xs, dim=-1) @ w^T`` for five equal-width activation taps and a small M (1..128) on gfx950,
without materializing the concat (the DSpark draft ``fc``: five taps of 6144, N=1536, K=30720).

    w_ring = shuffle_weight_ring(w)                       # once, before CUDA-graph capture
    if cat_x_ring_supported(m, n, seg_k, device):
        out = gemm_a16w16_cat_x_ring(xs, w_ring)

K is split ``ks`` ways (default 10: two splits per tap), each split's fp32 partial is stored, and the partials are
summed in ascending split order by aiter's FlyDSL split-K reduce (the one of the mxscale preshuffle GEMM, so this op
stays on one backend and its AOT replay is complete) with one bf16 rounding, so the result is deterministic. It is not bit-identical to a single-pass K order (or to hipBLASLt), only to bf16 rounding.
"""

import torch

from .gemm_a16w16_ring import MAX_ROWS, _device_ok
from .kernels.gemm_a16w16_cat_x_ring_gfx950 import (
    LAYOUT_AITER,
    LAYOUT_RING,
    LAYOUT_ROW,
    NSEG,
    compile_cat_ring,
)
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

MAX_SPLIT_K = 40  # tuner bound on ks: the fp32 partials are [ks, M, N]
NWAVE_FRAGS = 4  # 16-row weight fragments per workgroup column tile (one per wave)


def _ring_depth(stages: int) -> int:
    return 6 if stages >= 6 else 4


def _layout(shuffled) -> int:
    """``shuffled``: True (ring tiles), False (row-major) or ``"aiter"`` (``shuffle_weight(w, (16, 16))``)."""
    if shuffled == "aiter":
        return LAYOUT_AITER
    if isinstance(shuffled, bool):
        return LAYOUT_RING if shuffled else LAYOUT_ROW
    raise ValueError(f"unknown weight layout {shuffled!r}")


def cat_x_ring_supported(
    m: int,
    n: int,
    seg_k: int,
    device=None,
    ks: int = 10,
    r: int | None = None,
    mt: int | None = None,
) -> bool:
    """Whether ``gemm_a16w16_cat_x_ring`` takes M rows of five ``seg_k``-wide taps and an ``[n, 5 * seg_k]`` weight
    with ``ks`` K splits, ring depth ``r`` (default: ``_ring_depth``) and ``mt`` 16-row M fragments per tile (default
    ``ceil(m / 16)``)."""
    # ``is_available`` only gates the device-less (tuner catalog) callers; with a device the AOT dry run (no GPU
    # runtime) and a launch (CUDA tensors) both reach ``_device_ok``, which asks the arch and the CU count.
    if not 1 <= m <= MAX_ROWS or (device is None and not torch.cuda.is_available()):
        return False
    index = (
        torch.cuda.current_device() if device is None else torch.device(device).index
    )
    if index is None:
        index = torch.cuda.current_device()
    stages = NSEG * seg_k // 64
    if ks < 1 or ks % NSEG or stages % ks:
        return False
    r = r or _ring_depth(stages // ks)
    mt = mt or (m + 15) // 16
    return (
        mt <= 8
        and (m + 15) // 16 <= mt
        and _device_ok(index)
        and n % (16 * NWAVE_FRAGS) == 0
        and seg_k % 64 == 0
        and 2 <= r
        and stages // ks >= r
        # LDS ring: r slots of (weight fragments + M fragments) 2 KiB tiles; <= 160 KiB
        and r * (NWAVE_FRAGS + 1 + mt) * 2048 <= 160 * 1024
        # byte offsets into the weight and the fp32 partials are i32 inside the kernel
        and n * NSEG * seg_k * 2 < 2**31
        and ks * m * n * 4 < 2**31
    )


def ring_splitk_config(mt: int, ks: int, r: int) -> dict:
    """The HGEMM-name config of a split-K ring tiling (policy ``rk``): ``block_m`` is the rows of one M tile,
    ``block_n=64`` the four 16-column weight fragments of a workgroup, ``stages`` the ring depth, ``split_k`` the K
    splits (a multiple of 5: the K chunks of ``gemm_a16w16_ring_splitk``)."""
    return {
        "block_m": 16 * mt,
        "block_n": 16 * NWAVE_FRAGS,
        "block_k": 64,
        "stages": r,
        "split_k": ks,
        "m_waves": 1,
        "n_waves": NWAVE_FRAGS,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
        "policy": "rk",
    }


def ring_splitk_name_tiling(params: dict) -> tuple[int, int, int] | None:
    """``(mt, ks, r)`` of a parsed ``prk`` kernel name, or None when the fields are not a split-K ring config."""
    if (
        params["dtype"] != "bf16"
        or params["out_dtype"] != "bf16"
        or params["target_gfx"] != "gfx950"
        or params["has_bias"]
        or params["k_waves"] != 1
        or params["m_waves"] != 1
        or params["n_waves"] != NWAVE_FRAGS
        or params["block_n"] != 16 * NWAVE_FRAGS
        or params["block_k"] != 64
        or params["group_m"] != 0
        or params["k_stagger"]
        or params["b_non_temporal"]
        or params["block_m"] % 16
        or not 1 <= params["block_m"] // 16 <= 8
        or params["split_k"] % NSEG
        or not 2 <= params["stages"] <= 6
    ):
        return None
    return params["block_m"] // 16, params["split_k"], params["stages"]


def ring_splitk_tilings(m: int, k: int, n: int, device=None):
    """Every supported ``(mt, ks, r)`` for ``M x N x K``: the search space of the tuner. ``mt`` is the smallest M
    tile that holds ``m`` rows; the ring depth is 3..6 and ``ks`` every K split that keeps >= ``r`` stages per split
    and gives at least 64 workgroups (``ks`` <= ``MAX_SPLIT_K``)."""
    if k % NSEG:
        return []
    mt = (m + 15) // 16
    out = []
    for ks in range(NSEG, min(k // 64, MAX_SPLIT_K) + 1, NSEG):
        for r in (3, 4, 5, 6):
            if n // (16 * NWAVE_FRAGS) * ks >= 64 and cat_x_ring_supported(
                m, n, k // NSEG, device, ks, r, mt
            ):
                out.append((mt, ks, r))
    return out


def _launch(
    ptrs, m, seg_k, stride_bytes, w, out, ks, r, layout, workspace, tune_mt=None
):
    from .mxscale_preshuffle_kernels import _bound_reduce_exe

    if w.ndim != 2:
        raise ValueError(
            f"ring GEMM takes a 2-D weight [N, K], got shape {tuple(w.shape)}"
        )
    n = w.shape[0]
    # The kernel reads row ``i`` of a tap at byte ``i * stride_bytes`` through 32-bit offsets and a descriptor that
    # spans the last row's ``seg_k`` elements: a zero (expanded) or padded row stride is fine, a span past 2**31 is not.
    if stride_bytes < 0 or (m - 1) * stride_bytes + seg_k * 2 >= 2**31:
        raise ValueError(
            f"tap row stride {stride_bytes} B x {m} rows exceeds the kernel's 32-bit address range"
        )
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
    elif (
        out.device != w.device
        or out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
        # rows must not overlap (an expanded [1, N] buffer would take every result row at the same addresses); the
        # reduce kernel stores dwords of two bf16, so the row stride is even and the base 4-byte aligned
        or (m > 1 and (out.stride(0) < n or out.stride(0) % 2))
        or out.data_ptr() % 4
    ):
        raise ValueError(
            "out must be a bf16 [M, N] with unit column stride, a non-overlapping even row stride and 4-byte alignment"
        )
    elif m > 1 and ((m - 1) * out.stride(0) + n) * 2 >= 2**31:
        # the reduce kernel counts the output's dwords in 32-bit arithmetic: bound the physical byte span
        raise ValueError(
            f"out spans {((m - 1) * out.stride(0) + n) * 2} B, past the kernel's 32-bit address range"
        )
    if workspace is None:
        workspace = torch.empty(ks, m, n, dtype=torch.float32, device=w.device)
    elif (
        workspace.device != w.device
        or workspace.shape != (ks, m, n)
        or workspace.dtype != torch.float32
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be a contiguous fp32 [ks, M, N]")
    ldc = out.stride(0) if m > 1 else n
    with torch.cuda.device(w.device):
        stream = torch.cuda.current_stream()
        exe = compile_cat_ring(
            n,
            seg_k,
            tune_mt or (m + 15) // 16,
            ks,
            r or _ring_depth(NSEG * seg_k // 64 // ks),
            w.device.index,
            layout=layout,
        )
        _run_compiled(
            exe,
            *ptrs,
            w.data_ptr(),
            workspace.data_ptr(),
            m,
            stride_bytes,
            stream,
        )
        _run_compiled(
            _bound_reduce_exe.lazy(ks, "bf16", ldc != n),
            ptr_arg(workspace),
            ptr_arg(out),
            m * n // 2,  # n_out_dw (2 bf16 per dword)
            m * n,  # slab_stride_dw (fp32: 1 dword per element)
            n // 2,  # row_dw
            ldc // 2,  # ldc_dw
            current_stream_arg(w.device),
            specialization_key=w.device,
        )
    return out


def gemm_a16w16_cat_x_ring(
    xs: list[torch.Tensor],
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    ks: int = 10,
    workspace: torch.Tensor | None = None,
    shuffled: bool | str = True,
    ring_depth: int | None = None,
) -> torch.Tensor:
    """``out [M, N] bf16 = concat(xs, -1) @ w^T``; ``xs`` are five bf16 ``[M, seg_k]`` taps (unit column stride, one
    common 16-byte-aligned row stride) and ``w`` is ``shuffle_weight_ring(weight [N, 5 * seg_k])`` (``shuffled=True``),
    AITER's ``shuffle_weight(w, (16, 16))`` read in place (``"aiter"``) or the row-major weight (``False``).
    ``workspace`` is an optional fp32 ``[ks, M, N]`` buffer."""
    if len(xs) != NSEG:
        raise ValueError(f"expected {NSEG} taps, got {len(xs)}")
    m, seg_k = xs[0].shape
    n = w.shape[0]
    if (
        not cat_x_ring_supported(m, n, seg_k, xs[0].device, ks, ring_depth)
        or w.shape[1] != NSEG * seg_k
    ):
        raise ValueError(
            f"cat-x ring GEMM does not take M={m} N={n} seg_k={seg_k} ks={ks} on this device"
        )
    stride = xs[0].stride(0)
    if w.device.type != "cuda" or any(x.device != w.device for x in xs):
        raise ValueError("taps and weight must be on one CUDA device")
    for x in xs:
        if (
            x.dtype != torch.bfloat16
            or x.shape != (m, seg_k)
            or x.stride(1) != 1
            or x.stride(0) != stride
            or (stride * 2) % 16
            or x.data_ptr() % 16
        ):
            raise ValueError(
                "taps must be bf16 [M, seg_k] with unit column stride, one aligned row stride"
            )
    layout = _layout(shuffled)
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.data_ptr() % 16:
        raise ValueError("w must be a contiguous aligned bf16 weight")
    return _launch(
        [x.data_ptr() for x in xs],
        m,
        seg_k,
        stride * 2,
        w,
        out,
        ks,
        ring_depth,
        layout,
        workspace,
    )


def gemm_a16w16_ring_splitk(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    ks: int = 10,
    ring_depth: int | None = None,
    shuffled: bool | str = False,
    workspace: torch.Tensor | None = None,
    mt: int | None = None,
) -> torch.Tensor:
    """``out [M, N] bf16 = x [M, K] @ w^T`` with the split-K ring: the plain (K-contiguous) form of
    ``gemm_a16w16_cat_x_ring``. ``x`` is read as five ``K / 5`` chunks whose base pointers are ``x`` plus a chunk
    offset (``K`` a multiple of 320), so a caller that stores the five taps side by side in one ``[M, 5 * seg_k]``
    buffer (or ``torch.cat``s them) needs no tap list. ``mt`` is the 16-row M fragments of a tile (default
    ``ceil(M / 16)``)."""
    m, k = x.shape
    n = w.shape[0]
    if k % NSEG or w.shape[1] != k:
        raise ValueError(
            f"ring split-K GEMM does not take K={k} for w {tuple(w.shape)}"
        )
    seg_k = k // NSEG
    if not cat_x_ring_supported(m, n, seg_k, x.device, ks, ring_depth, mt):
        raise ValueError(
            f"ring split-K GEMM does not take M={m} N={n} K={k} ks={ks} on this device"
        )
    if x.device.type != "cuda" or w.device != x.device:
        raise ValueError("x and w must be on one CUDA device")
    if (
        x.dtype != torch.bfloat16
        or x.stride(1) != 1
        or (x.stride(0) * 2) % 16
        or x.data_ptr() % 16
    ):
        raise ValueError("x must be bf16 [M, K] with unit column stride, aligned rows")
    layout = _layout(shuffled)
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.data_ptr() % 16:
        raise ValueError("w must be a contiguous aligned bf16 weight")
    base = x.data_ptr()
    return _launch(
        [base + i * seg_k * 2 for i in range(NSEG)],
        m,
        seg_k,
        x.stride(0) * 2,
        w,
        out,
        ks,
        ring_depth,
        layout,
        workspace,
        mt,
    )


def warmup_cat_x_ring(
    m_max: int, n: int, seg_k: int, w: torch.Tensor, ks: int = 10
) -> None:
    """Compile every row tier up to and including ``m_max`` (call before CUDA-graph capture)."""
    x = [
        torch.zeros(m_max, seg_k, dtype=torch.bfloat16, device=w.device)
        for _ in range(NSEG)
    ]
    for m in range(16, m_max + 1, 16):
        gemm_a16w16_cat_x_ring([t[:m] for t in x], w, ks=ks)
    if m_max % 16:
        # The last tier is a partial 16-row tile (e.g. m_max=17 needs mt=2, m_max=8 needs mt=1):
        # the loop above only reaches multiples of 16 and would otherwise leave it uncompiled.
        gemm_a16w16_cat_x_ring([t[:m_max] for t in x], w, ks=ks)
    torch.cuda.synchronize(w.device)
