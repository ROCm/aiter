# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch layer for jagged_dense_bmm_broadcast_add.

Owns every launch default. Picks a config in this order: explicit override,
then the row of ``AITER_CONFIG_JAGGED_DENSE_BMM`` keyed by
``(gfx, cu_num, K, N, tiles_log2)``, then the analytic default. The skew path
takes only ``block_k`` from the row; its XCD remap comes from
``_skew_compact_xcd`` unless overridden.
"""

from __future__ import annotations

import functools

import pandas as pd
import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_gfx_runtime

from .kernels.jagged_dense_bmm_gen import (
    ARCH_TRAITS,
    BLOCK_M,
    BLOCK_N,
    jagged_dense_bmm,
)

__all__ = ["jagged_dense_bmm_dispatched"]

SUPPORTED_ARCHS = tuple(ARCH_TRAITS)

DEFAULT_BLOCK_K = 64
DEFAULT_THREADS = 256
DEFAULT_XCD_W = 8
DEFAULT_XCD_C_SMALL_K = 32
DEFAULT_XCD_C_LARGE_K = 120

SKEW_COMPACT_XCD_C = 32
SKEW_COMPACT_XCD_W = 8
SKEW_COMPACT_KN_XCD_C_SMALL = 64
SKEW_COMPACT_KN_XCD_W_SMALL = 16
_SKEW_XCD_REMAP_SMALL_B_THRESHOLD = 120

_KEY_COLS = ["gfx", "cu_num", "K", "N", "tiles_log2"]
_CFG_KEYS = ("xcd_c", "xcd_w", "block_k", "threads")


@functools.cache
def _tuned_table() -> dict:
    df = pd.read_csv(
        AITER_CONFIGS.AITER_CONFIG_JAGGED_DENSE_BMM_FILE,
        dtype=str,
        keep_default_na=False,
    )
    return {
        (r["gfx"], int(r["cu_num"]), int(r["K"]), int(r["N"]), int(r["tiles_log2"])): {
            k: int(r[k]) if r[k] else None for k in _CFG_KEYS
        }
        for r in df.to_dict("records")
    }


def tiles_log2(*, n_groups: int, output_n: int, max_seq_len: int) -> int:
    tiles = n_groups * -(-max_seq_len // BLOCK_M) * (output_n // BLOCK_N)
    return max(tiles, 1).bit_length() - 1


def _skew_compact_xcd(
    n_groups: int, reduction_k: int, dense_kn: bool = False
) -> tuple[int | None, int | None]:
    if dense_kn and n_groups <= _SKEW_XCD_REMAP_SMALL_B_THRESHOLD and reduction_k < 512:
        return SKEW_COMPACT_KN_XCD_C_SMALL, SKEW_COMPACT_KN_XCD_W_SMALL
    if n_groups > _SKEW_XCD_REMAP_SMALL_B_THRESHOLD or reduction_k >= 512:
        return SKEW_COMPACT_XCD_C, SKEW_COMPACT_XCD_W
    return None, None


def _config_valid(cfg: dict, *, reduction_k: int) -> bool:
    bk = cfg["block_k"] or DEFAULT_BLOCK_K
    return reduction_k % bk == 0 and reduction_k // bk >= 2


def resolve_config(
    *,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    uniform_seqlen: bool,
    dense_kn: bool,
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
) -> dict:
    key = (
        get_gfx(),
        get_cu_num(),
        reduction_k,
        output_n,
        tiles_log2(n_groups=n_groups, output_n=output_n, max_seq_len=max_seq_len),
    )
    cfg = dict(_tuned_table().get(key) or dict.fromkeys(_CFG_KEYS))
    if not _config_valid(cfg, reduction_k=reduction_k):
        cfg = dict.fromkeys(_CFG_KEYS)
    if block_k is not None:
        cfg["block_k"] = int(block_k)
    if not uniform_seqlen:
        cfg["xcd_c"], cfg["xcd_w"] = _skew_compact_xcd(n_groups, reduction_k, dense_kn)
    # Thread counts were tuned on the uniform 2-D layout; 3-D is faster at the default.
    if not uniform_seqlen or dense_kn:
        cfg["threads"] = None
    if xcd_c is not None:
        cfg["xcd_c"] = int(xcd_c)
    if xcd_w is not None:
        cfg["xcd_w"] = int(xcd_w)
    if cfg["xcd_c"] is None:
        if not uniform_seqlen:
            cfg["xcd_c"] = 1
        elif reduction_k > 256:
            cfg["xcd_c"] = DEFAULT_XCD_C_LARGE_K
        else:
            cfg["xcd_c"] = DEFAULT_XCD_C_SMALL_K
    if cfg["xcd_w"] is None:
        cfg["xcd_w"] = 1 if cfg["xcd_c"] == 1 else DEFAULT_XCD_W
    if cfg["block_k"] is None:
        cfg["block_k"] = DEFAULT_BLOCK_K
    if cfg["threads"] is None:
        cfg["threads"] = DEFAULT_THREADS
    return cfg


def _validate(C, A, B, BIAS, SEQ_OFFSETS, n_groups: int) -> tuple[int, int]:
    tensors = {"C": C, "A": A, "B": B, "BIAS": BIAS, "SEQ_OFFSETS": SEQ_OFFSETS}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.device != A.device or t.device.type != "cuda":
            raise ValueError(f"{name} must be on A's GPU device {A.device}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name in ("C", "A", "B", "BIAS"):
        if tensors[name].dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {tensors[name].dtype}")
    if SEQ_OFFSETS.dtype != torch.int32:
        raise TypeError(f"SEQ_OFFSETS must be int32, got {SEQ_OFFSETS.dtype}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if A.dim() != 2 or C.dim() != 2:
        raise ValueError("A and C must be 2-D")
    if B.dim() == 3:
        if B.shape[0] != n_groups:
            raise ValueError(f"B has {B.shape[0]} groups, n_groups={n_groups}")
        K, N = B.shape[1], B.shape[2]
    elif B.dim() == 2:
        if B.shape[0] % n_groups:
            raise ValueError(
                f"B rows {B.shape[0]} are not a multiple of n_groups={n_groups}"
            )
        N, K = B.shape[0] // n_groups, B.shape[1]
    else:
        raise ValueError(f"B must be 2-D or 3-D, got {B.dim()}-D")
    if A.shape[1] != K:
        raise ValueError(f"A has K={A.shape[1]}, B has K={K}")
    if tuple(C.shape) != (A.shape[0], N):
        raise ValueError(f"C must be {(A.shape[0], N)}, got {tuple(C.shape)}")
    if tuple(BIAS.shape) not in ((n_groups * N,), (n_groups, N)):
        raise ValueError(
            f"BIAS must be {(n_groups * N,)} or {(n_groups, N)}, "
            f"got {tuple(BIAS.shape)}"
        )
    if tuple(SEQ_OFFSETS.shape) != (n_groups + 1,):
        raise ValueError(
            f"SEQ_OFFSETS must be {(n_groups + 1,)}, got {tuple(SEQ_OFFSETS.shape)}"
        )
    return N, K


def jagged_dense_bmm_dispatched(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream=None,
    uniform_seqlen: bool = True,
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
):
    """``C[s:e] = A[s:e] @ dense[b] + bias[b]`` for each group ``b`` in place.

    Args:
        C: ``(L, N)`` bf16 output. Rows of a group are all written; rows past
            ``SEQ_OFFSETS[-1]`` in an over-allocated ``C`` are left untouched.
        A: ``(L, K)`` bf16 packed jagged rows. Requires
            ``L >= SEQ_OFFSETS[-1]``, which is not checked; extra rows are not
            read. One group's rows of ``A`` and of ``C`` must each stay under
            4 GiB, the 32-bit buffer bound.
        B: bf16 dense weights, either ``(n_groups, K, N)`` or
            ``(n_groups * N, K)``, the latter being
            ``dense.transpose(1, 2).reshape(-1, K)`` of the former.
        BIAS: ``(n_groups, N)`` or ``(n_groups * N,)`` bf16.
        SEQ_OFFSETS: ``(n_groups + 1,)`` int32 prefix sum of group lengths,
            read on the device only.
        max_seq_len: Uniform path: must be >= the longest group, which is
            not checked; rows past it are not computed. Skew path: only a
            tuning-table key.
        stream: Defaults to the current torch stream. One launch and no host
            sync, so the call can be captured in a graph.
        uniform_seqlen: True selects the dense ``(group, M-tile, N-tile)``
            grid; False the compact grid that skips empty tiles.
        xcd_c, xcd_w, block_k: Override the tuned config.

    Eager-only: no FakeTensor or ``torch.compile`` registration.
    """
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_ARCHS:
        raise RuntimeError(f"jagged_dense_bmm supports {SUPPORTED_ARCHS}, not {arch}")
    output_n, reduction_k = _validate(C, A, B, BIAS, SEQ_OFFSETS, n_groups)
    BIAS = BIAS.view(-1)
    if A.shape[0] == 0:
        return
    if uniform_seqlen and max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    cfg = resolve_config(
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
        uniform_seqlen=uniform_seqlen,
        dense_kn=B.dim() == 3,
        xcd_c=xcd_c,
        xcd_w=xcd_w,
        block_k=block_k,
    )
    if stream is None:
        stream = torch.cuda.current_stream(A.device)
    return jagged_dense_bmm(
        C,
        A,
        B,
        BIAS,
        SEQ_OFFSETS,
        n_groups,
        max_seq_len,
        stream=stream,
        uniform_seqlen=uniform_seqlen,
        traits=ARCH_TRAITS[arch],
        **cfg,
    )
