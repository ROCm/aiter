# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Python launcher for the gfx950 MXFP4 MoE stage-2 ``.co`` kernels in
``hsa/gfx950/moe_stage2_mxfp4_t1`` (made by ``gen_t1g2.py`` in that directory).

The kernels do the stage 2 (down projection) of AITER's MXFP4 a4w4 MoE for one
shape: 128 experts, top-4, model_dim 6144, inter_dim 1536 (MiniMax-M3.1-Flash at
TP2), for a large prefill. They replace the FlyDSL scatter gemm2
(``..._scatter_sbm128``): unweighted bf16 rows into a flat buffer, then
``aiter.mxfp4_moe_scatter_reduce``, as AITER's own non-atomic path, with the same
output bits. They read stage 1's output and AITER's sort as they are, with no
host sync (CUDA-graph safe).

``fused_moe._mxfp4_a4w4_stage2_fw`` calls :func:`moe_stage2_mxfp4_t1` when
``AITER_MXFP4_MOE_STAGE2_T1=1``. Calls outside the shape or the token range
below, and atomic ``kernelName2`` picks, return False and AITER's own stage 2
serves them.
"""

import ctypes
import functools
import os

import torch

import aiter
from aiter import dtypes
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
_BM = 128  # the block_m of AITER's FlyDSL scatter epilogue
# Token counts (M = moe_out.shape[0]) from which the kernels were measured faster than
# AITER's scatter stage 2 on gfx950, under uniform and under skewed routing (1.07x-1.37x
# against AITER's best gemm2 for the shape). Below 4096 AITER's atomic gemm2 picks win
# under uniform routing (the kernels reach 0.8x-1.0x) and lose under skewed routing, so
# those calls stay with AITER.
_MIN_M = 4096  # v2n
_DENSE_MIN_M = 12288  # v4g8: dense groups of blocks of one expert
_FP4_DTYPES = (torch.uint8, dtypes.fp4x2)  # packed MXFP4 data, as AITER passes it


class _KernelArgs(ctypes.Structure):
    """88-byte kernarg of the kernels."""

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


assert ctypes.sizeof(_KernelArgs) == 88


def pick_kernel(M, block_m, atomic):
    """``dense`` (0: v2n, 1: v4g8) of the kernel that serves M tokens, or None
    when AITER keeps the call. ``atomic``: ``kernelName2`` is an atomic gemm2."""
    if atomic or block_m != _BM or M < _MIN_M:
        return None
    return int(M >= _DENSE_MIN_M)


@functools.cache
def _device_arch(device_index):
    return get_gfx_from_device(device_index)


@functools.cache
def _kernel(arch, dense):
    """(registry row, .co path) of the kernel for ``dense``."""
    d = os.path.join(AITER_ASM_DIR, arch, _SUBDIR)
    for cfg in load_asm_cfg_csv(os.path.join(d, _CSV)):
        if (cfg["BM"], cfg["dense"]) == (_BM, dense):
            return cfg, os.path.join(d, cfg["co_name"])
    raise RuntimeError(
        f"moe_stage2_mxfp4_t1: no kernel for block_m {_BM} dense {dense} on {arch}"
    )


def _grid(cfg, neid):
    """CTAs for neid sorted blocks, a multiple of the 8 XCDs: NSL feature slices
    per block, or per group of up to Gb blocks of one expert (dense)."""
    nsl, gb = cfg["NSL"], cfg["Gb"]
    if cfg["dense"]:
        return 8 * nsl * (-(-((neid + min(neid, cfg["NE"]) * (gb - 1)) // gb) // 8))
    return 8 * (-(-neid * nsl // 8))


def check_installed():
    """Raise if this install lacks the code objects (e.g. a Triton-only wheel ships no ``hsa/``). Called when
    ``AITER_MXFP4_MOE_STAGE2_T1=1`` is read, so a missing file fails at import, not at the first long prefill.
    """
    d = os.path.join(AITER_ASM_DIR, "gfx950", _SUBDIR)
    csv = os.path.join(d, _CSV)
    if not os.path.isfile(csv):
        raise RuntimeError(
            f"AITER_MXFP4_MOE_STAGE2_T1=1, but {csv} is missing: this install ships no gfx950 "
            "moe_stage2_mxfp4_t1 code objects (a Triton-only wheel has an empty hsa/)"
        )
    for cfg in load_asm_cfg_csv(csv):
        if not os.path.isfile(os.path.join(d, cfg["co_name"])):
            raise RuntimeError(
                f"AITER_MXFP4_MOE_STAGE2_T1=1, but {cfg['co_name']} is missing in {d}"
            )


def _eligible(
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
):
    """``dense`` of the kernel that serves this call, or None: every flag, shape, dtype and layout check the
    kernels' ABI depends on. No device query, so it runs on CPU or meta tensors."""
    M = moe_out.shape[0]
    # Decode and short prefill, the common case, leave on one comparison.
    if M < _MIN_M or atomic:
        return None
    if (
        block_m is None
        or int(block_m) != _BM
        or bias2 is not None
        or topk_weights is not None
        or sorted_weights is None
        or reverse_sorted is None
        or a2_scale is None
        or w2_scale is None
        or use_valid_token_count
        or topk != _TOPK
    ):
        return None
    if (
        inter_states.dtype not in _FP4_DTYPES
        or tuple(inter_states.shape[1:]) != (_D_INTER // 2,)
        or w2.dtype not in _FP4_DTYPES
        or tuple(w2.shape) != (_NE, _D_HIDDEN, _D_INTER // 2)
        or getattr(w2, "inter_real", _D_INTER) != _D_INTER
        or a2_scale.element_size() != 1
        or w2_scale.element_size() != 1
        or moe_out.dtype != torch.bfloat16
        or tuple(moe_out.shape[1:]) != (_D_HIDDEN,)
        or sorted_weights.dtype != torch.float32
        or any(
            t.dtype != torch.int32
            for t in (sorted_token_ids, sorted_expert_ids, num_valid_ids)
        )
        or not all(
            t.is_contiguous()
            for t in (inter_states, w2, a2_scale, w2_scale, moe_out, sorted_weights)
        )
    ):
        return None
    return pick_kernel(M, int(block_m), atomic)


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
    dense = _eligible(
        inter_states,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_m=block_m,
        sorted_weights=sorted_weights,
        topk_weights=topk_weights,
        bias2=bias2,
        reverse_sorted=reverse_sorted,
        use_valid_token_count=use_valid_token_count,
        atomic=atomic,
    )
    if dense is None or not inter_states.is_cuda:
        return False
    rows = inter_states.shape[0]
    sizes = {
        "x_bytes": inter_states.numel(),
        "xs_bytes": a2_scale.numel(),
        "out_bytes": rows * _D_HIDDEN * 2,
        "eids_bytes": sorted_expert_ids.numel() * 4,
        "sids_bytes": sorted_token_ids.numel() * 4,
    }
    if max(sizes.values()) >= 1 << 32:
        return False
    dev = inter_states.device.index
    arch = _device_arch(dev)
    if arch != "gfx950":
        return False
    cfg, co_path = _kernel(arch, dense)

    flat_out = torch.empty(
        (rows, _D_HIDDEN), dtype=torch.bfloat16, device=moe_out.device
    )
    args = _KernelArgs(
        w2=w2.data_ptr(),
        s2=w2_scale.data_ptr(),
        x=inter_states.data_ptr(),
        xs=a2_scale.data_ptr(),
        eids=sorted_expert_ids.data_ptr(),
        sids=sorted_token_ids.data_ptr(),
        nvalid=num_valid_ids.data_ptr(),
        out=flat_out.data_ptr(),
        M=moe_out.shape[0],
        **sizes,
    )
    grid = (_grid(cfg, sorted_expert_ids.numel()), 1, 1)
    with device_guard(dev):
        func = get_function(co_path, cfg["knl_name"], dev)
        launch_co(func, grid, (256, 1, 1), args, stream=current_raw_stream(dev))
    aiter.mxfp4_moe_scatter_reduce(
        flat_out=flat_out,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        out=moe_out,
        NE=_NE,
        TOPK=topk,
        D_HIDDEN=_D_HIDDEN,
        MB=_BM,
    )
    return True
