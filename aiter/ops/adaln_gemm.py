# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Skinny bf16 GEMMs of a DiT's AdaLN modulation linear for a 32-row micro-batch (asm kernels, ``hsa/<arch>/adalngemm``):

* ``adaln_fwd(x, w, bias, out)``:  ``out[32, N] = x[32, K] @ w[N, K]^T + bias``
* ``adaln_dgrad(dy, w, out)``:     ``out[32, K] = dy[32, N] @ w[N, K]``

  (both one launch with a deterministic split-K reduce; the fp32 workspace and int32 counters they need are allocated
  here per (pass, device, N, K) and reused -- the counters stay zero between calls)
* ``adaln_wgrad(dy, x, out)``:     ``out[N, K] = dy[32, N]^T @ x[32, K]`` (``out`` may be a parameter's main_grad)

Kernels exist for exact (pass, N, K); ``adaln_gemm_supported`` says whether one does (pure Python on ints, safe inside
compiled regions).
"""

import csv
import functools
import os

import torch
from torch import Tensor

from ..jit.core import AITER_META_DIR, compile_ops

PASS = {"fwd": 0, "dgrad": 1, "wgrad": 2}


@compile_ops("module_adaln_gemm_asm", fc_name="adaln_gemm_asm", ffi_type="ctypes")
def _adaln_gemm_asm(
    pass_: int,
    a: Tensor,
    b: Tensor,
    bias: Tensor | None,
    out: Tensor,
    ws: Tensor | None,
    cnt: Tensor | None,
) -> None: ...


@functools.lru_cache(maxsize=None)
def _manifest(arch: str = "gfx950"):
    path = os.path.join(AITER_META_DIR, "hsa", arch, "adalngemm", "adalngemm_bf16.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(int(r["op"]), int(r["N"]), int(r["K"])): (int(r["ws"]), int(r["cnt"])) for r in csv.DictReader(f)}


def adaln_gemm_supported(pass_: str, N: int, K: int) -> bool:
    """Whether a kernel exists for ``pass_`` ("fwd" / "dgrad" / "wgrad") at (N, K) with a 32-row micro-batch."""
    return (PASS[pass_], N, K) in _manifest()


_WS: dict = {}


def _workspace(pass_: int, device, N: int, K: int):
    key = (pass_, device, N, K)
    if key not in _WS:
        ws, cnt = _manifest()[(pass_, N, K)]
        _WS[key] = (torch.empty(ws, dtype=torch.float32, device=device),
                    torch.zeros(cnt, dtype=torch.int32, device=device))
    return _WS[key]


def adaln_fwd(x: Tensor, w: Tensor, bias: Tensor, out: Tensor) -> Tensor:
    ws, cnt = _workspace(0, x.device, w.shape[0], w.shape[1])
    _adaln_gemm_asm(0, x, w, bias, out, ws, cnt)
    return out


def adaln_dgrad(dy: Tensor, w: Tensor, out: Tensor) -> Tensor:
    ws, cnt = _workspace(1, dy.device, w.shape[0], w.shape[1])
    _adaln_gemm_asm(1, dy, w, None, out, ws, cnt)
    return out


def adaln_wgrad(dy: Tensor, x: Tensor, out: Tensor) -> Tensor:
    _adaln_gemm_asm(2, dy, x, None, out, None, None)
    return out
