# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Python launcher for the gfx950 MXFP4 MoE stage-2 ``.co`` kernels in
``hsa/gfx950/moe_stage2_mxfp4_t1`` (made by ``gen_t1g2.py`` in that directory).

The kernels do the stage 2 (down projection) of AITER's MXFP4 a4w4 MoE for one
shape: 128 experts, top-4, model_dim 6144, inter_dim 1536 (MiniMax-M3.1-Flash at
TP2). They read stage 1's output and AITER's sort as they are, with no host sync
(CUDA-graph safe):
  * v2ao: bf16 atomic epilogue into ``moe_out`` in AITER's gemm2 tile order,
    for an atomic ``kernelName2`` (the sort zeroed ``moe_out``);
  * v2n, v4g8: unweighted bf16 rows into a flat buffer, then
    ``aiter.mxfp4_moe_scatter_reduce``, as AITER's non-atomic path.

``fused_moe._mxfp4_a4w4_stage2_fw`` calls :func:`moe_stage2_mxfp4_t1` when
``AITER_MXFP4_MOE_STAGE2_T1=1``. Calls outside the shape or the token ranges
below return False and AITER's own stage 2 serves them.
"""

import ctypes
import functools
import os

import torch

import aiter
from aiter.jit.core import AITER_ASM_DIR
from aiter.ops.asm.asm_utils import (
    current_raw_stream,
    device_guard,
    get_function,
    get_gfx_from_device,
    launch_co,
    load_asm_cfg_csv,
)

_SUBDIR = "moe_stage2_mxfp4_t1"
_CSV = "moe_stage2_mxfp4_t1.csv"
_NE, _TOPK, _D_HIDDEN, _D_INTER = 128, 4, 6144, 1536
_BMS = (32, 64, 128)
# Token counts (M = moe_out.shape[0]) where each kernel was measured faster than
# AITER's stage 2 on MI350X (gfx950) and checked for accuracy.
_ATOMIC_MIN_M, _ATOMIC_MAX_M = 1024, 2285  # v2ao, atomic kernelName2
_ATOMIC_BMS = (64, 128)
_FLAT_MIN_M = 1536  # v2n
_DENSE_MIN_M = 12288  # v4g8, block_m >= 64


class _KernelArgs(ctypes.Structure):
    """88-byte kernarg of the flat-output kernels (v2n, v4g8)."""

    _fields_ = [
        ("w2", ctypes.c_void_p),
        ("s2", ctypes.c_void_p),
        ("x", ctypes.c_void_p),
        ("xs", ctypes.c_void_p),
        ("eids", ctypes.c_void_p),
        ("sids", ctypes.c_void_p),
        ("nvalid", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("x_bytes", ctypes.c_uint32),
        ("xs_bytes", ctypes.c_uint32),
        ("out_bytes", ctypes.c_uint32),
        ("eids_bytes", ctypes.c_uint32),
        ("sids_bytes", ctypes.c_uint32),
        ("M", ctypes.c_uint32),
    ]


class _AtomicKernelArgs(ctypes.Structure):
    """96-byte kernarg of the atomic kernel (v2ao): adds sorted_weights."""

    _fields_ = _KernelArgs._fields_ + [("sw", ctypes.c_void_p)]


assert ctypes.sizeof(_KernelArgs) == 88 and ctypes.sizeof(_AtomicKernelArgs) == 96


def pick_kernel(M, block_m, atomic):
    """(atomic, dense) of the kernel that serves M tokens, or None when AITER
    keeps the call. ``atomic``: ``kernelName2`` is an atomic gemm2."""
    if atomic and block_m in _ATOMIC_BMS and _ATOMIC_MIN_M <= M <= _ATOMIC_MAX_M:
        return 1, 0
    if M < _FLAT_MIN_M:
        return None
    return 0, int(M >= _DENSE_MIN_M and block_m >= 64)


@functools.cache
def _device_arch(device_index):
    return get_gfx_from_device(device_index)


@functools.cache
def _kernel(arch, block_m, atomic, dense):
    """(registry row, .co path) of the kernel for (block_m, atomic, dense)."""
    d = os.path.join(AITER_ASM_DIR, arch, _SUBDIR)
    for cfg in load_asm_cfg_csv(os.path.join(d, _CSV)):
        if (cfg["BM"], cfg["atomic"], cfg["dense"]) == (block_m, atomic, dense):
            return cfg, os.path.join(d, cfg["co_name"])
    raise RuntimeError(
        f"moe_stage2_mxfp4_t1: no kernel for block_m {block_m} atomic {atomic} "
        f"dense {dense} on {arch}"
    )


def _grid(cfg, neid):
    """CTAs for neid sorted blocks, a multiple of the 8 XCDs: NSL feature slices
    per block, or per group of up to Gb blocks of one expert (dense)."""
    nsl, gb = cfg["NSL"], cfg["Gb"]
    if cfg["dense"]:
        return 8 * nsl * (-(-((neid + min(neid, cfg["NE"]) * (gb - 1)) // gb) // 8))
    return 8 * (-(-neid * nsl // 8))


def moe_stage2_mxfp4_t1(
    inter_states,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    moe_out,
    topk,
    *,
    w2_scale,
    a2_scale,
    block_m,
    sorted_weights,
    topk_weights,
    bias2,
    reverse_sorted,
    use_valid_token_count,
    atomic,
) -> bool:
    """Write the stage 2 of one ``_mxfp4_a4w4_stage2_fw`` call into ``moe_out``
    and return True, or return False, having written nothing, for a call these
    kernels do not serve."""
    if (
        block_m is None
        or int(block_m) not in _BMS
        or bias2 is not None
        or topk_weights is not None
        or sorted_weights is None
        or a2_scale is None
        or w2_scale is None
        or use_valid_token_count
        or topk != _TOPK
        or inter_states.element_size() != 1
        or tuple(inter_states.shape[1:]) != (_D_INTER // 2,)
        or tuple(w2.shape) != (_NE, _D_HIDDEN, _D_INTER // 2)
        or getattr(w2, "inter_real", _D_INTER) != _D_INTER
        or moe_out.dtype != torch.bfloat16
        or tuple(moe_out.shape[1:]) != (_D_HIDDEN,)
    ):
        return False
    block_m = int(block_m)
    M = moe_out.shape[0]
    pick = pick_kernel(M, block_m, atomic)
    if pick is None or (not pick[0] and reverse_sorted is None):
        return False
    out_rows = M if pick[0] else inter_states.shape[0]
    sizes = {
        "x_bytes": inter_states.numel(),
        "xs_bytes": a2_scale.numel(),
        "out_bytes": out_rows * _D_HIDDEN * 2,
        "eids_bytes": sorted_expert_ids.numel() * 4,
        "sids_bytes": sorted_token_ids.numel() * 4,
    }
    if max(sizes.values()) >= 1 << 32:
        return False
    dev = inter_states.device.index
    arch = _device_arch(dev)
    if arch != "gfx950":
        return False
    cfg, co_path = _kernel(arch, block_m, *pick)

    if pick[0]:
        out, Args, extra = moe_out, _AtomicKernelArgs, {"sw": sorted_weights.data_ptr()}
    else:
        out = torch.empty(
            (out_rows, _D_HIDDEN), dtype=torch.bfloat16, device=moe_out.device
        )
        Args, extra = _KernelArgs, {}
    args = Args(
        w2=w2.data_ptr(),
        s2=w2_scale.data_ptr(),
        x=inter_states.data_ptr(),
        xs=a2_scale.data_ptr(),
        eids=sorted_expert_ids.data_ptr(),
        sids=sorted_token_ids.data_ptr(),
        nvalid=num_valid_ids.data_ptr(),
        out=out.data_ptr(),
        M=M,
        **sizes,
        **extra,
    )
    grid = (_grid(cfg, sorted_expert_ids.numel()), 1, 1)
    with device_guard(dev):
        func = get_function(co_path, cfg["knl_name"], dev)
        launch_co(func, grid, (256, 1, 1), args, stream=current_raw_stream(dev))
    if not pick[0]:
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=out,
            reverse_sorted=reverse_sorted,
            sorted_weights=sorted_weights,
            out=moe_out,
            NE=_NE,
            TOPK=topk,
            D_HIDDEN=_D_HIDDEN,
            MB=block_m,
        )
    return True
