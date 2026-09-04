# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-ASM AttnResidual score and combine kernels for Kimi-K3 on gfx950.

These are hand-optimised AMDGCN assembly kernels produced by Evolve-Kernel
island search (2026-08-05).  They replace the Triton _score_kernel and
_combine_kernel used by the RadixLinearAttention AttnResidual path.

Performance on MI355X at the Kimi-K3 CONC=64 decode shape
(T=64, NVB=8, H=7168, BLOCK_H=1024):
  score:   34.78 us -> 5.53 us  (-84 %)
  combine: 104.02 us -> 3.32 us (-97 %)

The kernels are gfx950-only; callers must guard with
``is_kimi_k3_attnres_asm_supported()``.  If this returns False,
fall back to the Triton reference implementation.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import Optional

import torch

_HSA_DIR = Path(__file__).resolve().parents[4] / "hsa" / "gfx950" / "attnres"
_SCORE_CO = _HSA_DIR / "kimik3_attnres_score_gfx950.co"
_COMBINE_CO = _HSA_DIR / "kimik3_attnres_combine_gfx950.co"

# Fixed Kimi-K3 shapes (kernels are not parametric)
_T = 64
_NVB = 8
_H = 7168
_BLOCK_H = 1024
_MAX_ROWS = 16
_EPS = 1e-6


@functools.lru_cache(maxsize=None)
def _rocm_arch(device_idx: int) -> str:
    props = torch.cuda.get_device_properties(device_idx)
    arch = getattr(props, "gcnArchName", "")
    return arch.split(":", 1)[0]


def is_kimi_k3_attnres_asm_supported(device: Optional[torch.device] = None) -> bool:
    """Return True when the current GPU can run the gfx950-only ASM kernels."""
    if not torch.cuda.is_available():
        return False
    if not _SCORE_CO.exists() or not _COMBINE_CO.exists():
        return False
    try:
        idx = torch.cuda.current_device() if device is None else device.index or 0
        return _rocm_arch(idx) == "gfx950"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Low-level loader: hipModuleLoadData + hipModuleGetFunction, cached per co
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _load_fn(co_path: str, kernel_name: str):
    """Load a .co file and return (hip_lib, module_p, fn_p).

    Cached per (co_path, kernel_name) -- the module stays loaded for the
    lifetime of the process.
    """
    hip_lib = ctypes.CDLL("libamdhip64.so")

    code = Path(co_path).read_bytes()
    module_p = ctypes.c_void_p()
    err = hip_lib.hipModuleLoadData(ctypes.byref(module_p), ctypes.c_char_p(code))
    if err:
        raise RuntimeError(f"hipModuleLoadData failed ({err}) for {co_path}")

    fn_p = ctypes.c_void_p()
    err = hip_lib.hipModuleGetFunction(
        ctypes.byref(fn_p), module_p, kernel_name.encode()
    )
    if err:
        raise RuntimeError(
            f"hipModuleGetFunction failed ({err}): kernel '{kernel_name}' "
            f"not found in {co_path}"
        )
    return hip_lib, module_p, fn_p


def _launch(
    hip_lib, fn_p, args, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared=0
):
    arg_ptrs = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
    )
    err = hip_lib.hipModuleLaunchKernel(
        fn_p,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared,
        None,
        arg_ptrs,
        None,
    )
    if err:
        raise RuntimeError(f"hipModuleLaunchKernel failed ({err})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attnres_score_asm(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """In-place AttnResidual score kernel (ASM, gfx950 only).

    Args:
        prefix: [T, H] bf16 -- prefix hidden states.
        bank:   [T, NVB, H] bf16 -- bank hidden states.
        cw:     [H] fp32 -- combined weight vector.
        scores: [T, MAX_ROWS] fp32 -- output (written in-place, zeroed first).

    Shape contract: T=64, NVB=8, H=7168, MAX_ROWS=16 (Kimi-K3 serving config).
    """
    T, H = prefix.shape
    assert (T, H) == (
        _T,
        _H,
    ), f"score: expected prefix [{_T},{_H}], got {list(prefix.shape)}"
    assert bank.shape == (T, _NVB, H), f"score: expected bank [{T},{_NVB},{H}]"
    assert cw.shape == (H,) and cw.dtype == torch.float32
    assert scores.shape == (T, _MAX_ROWS) and scores.dtype == torch.float32

    hip_lib, _, fn_p = _load_fn(str(_SCORE_CO), "kimik3_attnres_score")

    args = [
        ctypes.c_void_p(prefix.data_ptr()),
        ctypes.c_void_p(bank.data_ptr()),
        ctypes.c_void_p(cw.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_int(_NVB),
        ctypes.c_float(_EPS),
        ctypes.c_int(int(prefix.stride(0))),
        ctypes.c_int(int(bank.stride(0))),
        ctypes.c_int(int(bank.stride(1))),
        ctypes.c_int(int(scores.stride(0))),
    ]
    # Grid: [T, NVB+1] -- one CTA per (token, snapshot_or_prefix)
    _launch(hip_lib, fn_p, args, T, _NVB + 1, 1, _BLOCK_H, 1, 1, shared=128)


def attnres_combine_asm(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    scores: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """In-place AttnResidual combine kernel (ASM, gfx950 only).

    Args:
        prefix: [T, H] bf16.
        bank:   [T, NVB, H] bf16.
        scores: [T, MAX_ROWS] fp32 -- raw (unnormalised) logits.
        out:    [T, H] bf16 -- softmax-weighted sum (written in-place).

    Shape contract: T=64, NVB=8, H=7168, BLOCK_H=1024 (Kimi-K3 serving config).
    """
    T, H = prefix.shape
    n_h_blocks = H // _BLOCK_H
    assert (T, H) == (
        _T,
        _H,
    ), f"combine: expected prefix [{_T},{_H}], got {list(prefix.shape)}"
    assert bank.shape == (T, _NVB, H)
    assert scores.shape == (T, _MAX_ROWS) and scores.dtype == torch.float32
    assert out.shape == (T, H) and out.dtype == torch.bfloat16

    hip_lib, _, fn_p = _load_fn(str(_COMBINE_CO), "kimik3_attnres_combine")

    args = [
        ctypes.c_void_p(prefix.data_ptr()),
        ctypes.c_void_p(bank.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(_NVB),
        ctypes.c_int(int(prefix.stride(0))),
        ctypes.c_int(int(bank.stride(0))),
        ctypes.c_int(int(bank.stride(1))),
        ctypes.c_int(int(scores.stride(0))),
        ctypes.c_int(int(out.stride(0))),
    ]
    # Grid: [T, H//BLOCK_H] -- one CTA per (token, H-tile)
    _launch(hip_lib, fn_p, args, T, n_h_blocks, 1, _BLOCK_H, 1, 1, shared=256)
