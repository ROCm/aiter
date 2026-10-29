# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch layer for jagged_dense_bmm_broadcast_add.

Picks a per-shape config from the live arch's section of
``jagged_dense_bmm_dispatch.json``: explicit override, then table winner, then
kernel default. The skew path takes only ``block_k`` from the table; its XCD
remap comes from ``_skew_compact_xcd`` unless overridden.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.jagged_dense_bmm_gen import jagged_dense_bmm

__all__ = ["jagged_dense_bmm_dispatched"]

SUPPORTED_ARCHS = ("gfx942",)

SKEW_COMPACT_XCD_C = 32
SKEW_COMPACT_XCD_W = 8
SKEW_COMPACT_KN_XCD_C_SMALL = 64
SKEW_COMPACT_KN_XCD_W_SMALL = 16
_SKEW_XCD_REMAP_SMALL_B_THRESHOLD = 120

_CFG_KEYS = ("xcd_c", "xcd_w", "block_k", "threads")


def _dispatch_json_path() -> Path:
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_DISPATCH_JSON")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "jagged_dense_bmm_dispatch.json"


@functools.cache
def _load_winners(arch: str) -> dict:
    path = _dispatch_json_path()
    if not path.is_file():
        return {}
    section = json.loads(path.read_text()).get("by_arch", {}).get(arch, {})
    return dict(section.get("winners") or {})


def shape_id(
    *, n_groups: int, reduction_k: int, output_n: int, max_seq_len: int
) -> str:
    return f"B{n_groups}D{reduction_k}K{output_n}N{max_seq_len}"


def _skew_compact_xcd(
    n_groups: int, reduction_k: int, dense_kn: bool = False
) -> tuple[int | None, int | None]:
    if dense_kn and n_groups <= _SKEW_XCD_REMAP_SMALL_B_THRESHOLD and reduction_k < 512:
        return SKEW_COMPACT_KN_XCD_C_SMALL, SKEW_COMPACT_KN_XCD_W_SMALL
    if n_groups > _SKEW_XCD_REMAP_SMALL_B_THRESHOLD or reduction_k >= 512:
        return SKEW_COMPACT_XCD_C, SKEW_COMPACT_XCD_W
    return None, None


def _config_valid(cfg: dict, *, reduction_k: int) -> bool:
    bk = cfg["block_k"] or 64
    return reduction_k % bk == 0 and reduction_k // bk >= 2


def resolve_config(
    *,
    arch: str,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
) -> dict:
    sid = shape_id(
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
    )
    entry = _load_winners(arch).get(sid) or {}
    cfg = {k: None if entry.get(k) is None else int(entry[k]) for k in _CFG_KEYS}
    if not _config_valid(cfg, reduction_k=reduction_k):
        cfg = dict.fromkeys(_CFG_KEYS)
    for k, v in (("xcd_c", xcd_c), ("xcd_w", xcd_w), ("block_k", block_k)):
        if v is not None:
            cfg[k] = int(v)
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
        arch=arch,
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
        xcd_c=xcd_c,
        xcd_w=xcd_w,
        block_k=block_k,
    )
    if stream is None:
        stream = torch.cuda.current_stream(A.device)

    if not uniform_seqlen:
        sc_xcd_c, sc_xcd_w = _skew_compact_xcd(n_groups, reduction_k, B.dim() == 3)
        return jagged_dense_bmm(
            C,
            A,
            B,
            BIAS,
            SEQ_OFFSETS,
            n_groups,
            max_seq_len,
            stream=stream,
            xcd_c=sc_xcd_c if xcd_c is None else xcd_c,
            xcd_w=sc_xcd_w if xcd_w is None else xcd_w,
            uniform_seqlen=False,
            block_k=cfg["block_k"],
        )

    return jagged_dense_bmm(
        C,
        A,
        B,
        BIAS,
        SEQ_OFFSETS,
        n_groups,
        max_seq_len,
        stream=stream,
        xcd_c=cfg["xcd_c"],
        xcd_w=cfg["xcd_w"],
        block_k=cfg["block_k"],
        # Thread counts were tuned on the 2-D layout; 3-D is faster at the default.
        threads=cfg["threads"] if B.dim() == 2 else None,
    )
