# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host op for the gfx950 small-M BF16 ring GEMM (``out = x @ w^T``, 1 <= M <= 128), e.g. a vocabulary-shard
``lm_head`` where the weight is read once per call and the GEMM is HBM-bound.

Typical use for a weight that is fixed for the life of the model::

    w_ring = shuffle_weight_ring(w)            # once, before CUDA-graph capture
    ...
    if ring_supported(x.shape[0], *w.shape):
        out = gemm_a16w16_ring(x, w_ring)      # per call, graph safe
"""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.gemm_a16w16_ring_gfx950 import (
    LAYOUT_AITER,
    LAYOUT_RING,
    LAYOUT_ROW,
    NWV,
    compile_ring,
    fits,
    ring_depth,
)
from .kernels.tensor_shim import _run_compiled

MAX_ROWS = 128
_MH1_MAX_ROWS = 64


def ring_tiling(m: int):
    """``(mt, mh, r)`` for ``m`` rows: one M half (4 waves) up to 64 rows at the deepest ring that fits (amd-kernels'
    own dispatch), two M halves (8 waves, ring depth 3) for 65-128 rows."""
    if m <= _MH1_MAX_ROWS:
        mt = (m + 15) // 16
        return mt, 1, ring_depth(mt)
    return (m + 31) // 32, 2, 3


def _layout(shuffled) -> int:
    """``shuffled``: True (ring tiles), False (row-major) or ``"aiter"`` (``shuffle_weight(w, (16, 16))``)."""
    if shuffled == "aiter":
        return LAYOUT_AITER
    if isinstance(shuffled, bool):
        return LAYOUT_RING if shuffled else LAYOUT_ROW
    raise ValueError(f"unknown ring weight layout {shuffled!r}")


def ring_tilings(m: int):
    """Every legal ``(mt, mh, r)`` that covers ``m`` rows in one M tile: the search space of the tuner."""
    out = []
    for mh in (1, 2):
        for mt in range(1, 9):
            if 16 * mt * mh < m or 16 * mt * mh > MAX_ROWS:
                continue
            out.extend((mt, mh, r) for r in (2, 3, 4))
    return out


def ring_config(mt: int, mh: int, r: int) -> dict:
    """The HGEMM-name config of a ring tiling: ``block_m`` is the rows of one M tile, ``block_n=16`` one weight
    fragment, ``stages`` the ring depth, ``m_waves`` the M halves and ``n_waves`` the waves splitting the fragments.
    """
    return {
        "block_m": 16 * mt * mh,
        "block_n": 16,
        "block_k": 64,
        "stages": r,
        "split_k": 1,
        "m_waves": mh,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
        "policy": "rg",
    }


def ring_name_tiling(params: dict) -> tuple[int, int, int] | None:
    """``(mt, mh, r)`` of a parsed ``prg`` kernel name, or None when the name fields are not a ring config."""
    mh = params["m_waves"]
    if (
        params["dtype"] != "bf16"
        or params["out_dtype"] != "bf16"
        or params["target_gfx"] != "gfx950"
        or params["has_bias"]
        or params["split_k"] != 1
        or params["k_waves"] != 1
        or params["n_waves"] != NWV
        or params["block_n"] != 16
        or params["block_k"] != 64
        or params["group_m"] != 0
        or params["k_stagger"]
        or params["b_non_temporal"]
        or mh not in (1, 2)
        or params["block_m"] % (16 * mh)
        or not 1 <= params["block_m"] // (16 * mh) <= 8
        or not 2 <= params["stages"] <= 4
    ):
        return None
    return params["block_m"] // (16 * mh), mh, params["stages"]


@functools.cache
def _device_ok(index: int) -> bool:
    return (
        get_gfx_runtime() == "gfx950"
        and torch.cuda.get_device_properties(index).multi_processor_count == 256
    )


def ring_supported(m: int, n: int, k: int, device=None, tiling=None) -> bool:
    """Whether ``gemm_a16w16_ring`` takes this M/N/K (and ``tiling``, default ``ring_tiling(m)``) on this GPU
    (gfx950, 256 CUs)."""
    # ``is_available`` only gates the device-less (tuner catalog) callers; with a device the AOT dry run (no GPU
    # runtime) and a launch (CUDA tensors) both reach ``_device_ok``, which asks the arch and the CU count.
    if not 1 <= m <= MAX_ROWS or (device is None and not torch.cuda.is_available()):
        return False
    index = (
        torch.cuda.current_device() if device is None else torch.device(device).index
    )
    if index is None:
        index = torch.cuda.current_device()
    mt, mh, r = tiling or ring_tiling(m)
    # Reject a tile too small to hold M, as ``gemm_a16w16_ring`` does below, so this dispatch gate and the launch
    # agree: ``fits`` only checks the N/K resources, not the row capacity. Mirrors ``cat_x_ring_supported``.
    if mt * mh * 16 < m:
        return False
    return _device_ok(index) and fits(n, k, mt, mh, r)


def shuffle_weight_ring(w: torch.Tensor) -> torch.Tensor:
    """The ``[N/16, K/64, 16, 64]`` tile layout of the row-major ``w [N, K]``, returned as ``[N, K]`` (a bit-exact
    permutation, new storage). ``gemm_a16w16_ring`` reads it with ``shuffled=True``."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0
    return w.view(n // 16, 16, k // 64, 64).permute(0, 2, 1, 3).contiguous().view(n, k)


def unshuffle_weight_ring(ws: torch.Tensor) -> torch.Tensor:
    """The inverse of ``shuffle_weight_ring``."""
    n, k = ws.shape
    return ws.view(n // 16, k // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(n, k)


def gemm_a16w16_ring(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    shuffled: bool | str = True,
    tiling: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """``out [M, N] bf16 = x [M, K] bf16 @ w^T`` with fp32 accumulation; ``w`` is ``shuffle_weight_ring``'s layout
    (``shuffled=True``), AITER's ``shuffle_weight(w, (16, 16))`` layout read in place (``shuffled="aiter"``, no second
    weight copy) or the row-major weight (``False``). ``tiling`` is ``(mt, mh, r)`` (16-row fragments per wave, M halves,
    ring depth), default ``ring_tiling(m)``. Deterministic, no allocation when ``out`` is given, graph safe
    once the first call of each ``(M tier, N, K, layout)`` has compiled (warm up before capture).
    """
    m, k = x.shape
    n = w.shape[0]
    if x.device.type != "cuda" or w.device != x.device:
        raise ValueError("ring GEMM takes CUDA x and w on the same device")
    layout = _layout(shuffled)
    mt, mh, r = tiling or ring_tiling(m)
    if tiling and mt * mh * 16 < m:
        raise ValueError(f"ring tiling {tiling} holds {mt * mh * 16} rows, not M={m}")
    if not ring_supported(m, n, k, x.device, tiling):
        raise ValueError(f"ring GEMM does not take M={m} N={n} K={k} on this device")
    if (
        w.ndim != 2
        or x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or w.shape[1] != k
        or not x.is_contiguous()
        or not w.is_contiguous()
        or x.data_ptr() % 16
        or w.data_ptr() % 16
    ):
        raise ValueError(
            "ring GEMM takes contiguous 16-byte-aligned BF16 x [M, K] and w [N, K]"
        )
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    elif (
        out.device != x.device
        or out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError(
            "out must be a contiguous 16-byte-aligned BF16 [M, N] on the device of x"
        )
    # The compiled handle is device bound: key it (and launch) per device.
    with torch.cuda.device(x.device):
        exe = compile_ring(n, k, mt, mh, r, layout, torch.cuda.current_device())
        _run_compiled(
            exe,
            x.data_ptr(),
            w.data_ptr(),
            out.data_ptr(),
            m,
            1,
            torch.cuda.current_stream(x.device),
        )
    return out


def warmup_ring(
    n: int, k: int, device, w: torch.Tensor, shuffled: bool | str = True
) -> None:
    """Compile every row tier (call before CUDA-graph capture)."""
    for m in (16, 32, 48, 64, 96, 128):
        if ring_supported(m, n, k, device):
            gemm_a16w16_ring(
                torch.zeros(m, k, dtype=torch.bfloat16, device=device),
                w,
                shuffled=shuffled,
            )
    torch.cuda.synchronize(device)
