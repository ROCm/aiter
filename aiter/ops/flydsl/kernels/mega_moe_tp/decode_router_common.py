# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared helpers: bf16x3 weight split of the decode-shape router GEMM.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4`` (``router_gemm.py``, ``router_topk.py``,
``common/loaders.py``, ``i3072/gemm1_prefill_h6144_i3072_persistent.py``, ``mxfp8/gemm1_prefill_h6144_i3072.py``),
Apache-2.0, "Copyright contributors to the amd-kernels project".
"""

import functools

import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from aiter.jit.utils.chip_info import get_gfx_runtime_for_device
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.kernels.topk.topk_launch import _launcher_on

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


# The kernels issue ``mfma_f32_16x16x32_bf16`` (gfx950 only, gfx942 has the 16x16x16 form) on 64-lane waves, and the
# 128-row tier's LDS (69632 B) exceeds gfx942's 64 KiB.
SUPPORTED_ARCH = "gfx950"
# a byte extent the kernels address through a 32-bit buffer descriptor
MAX_BUFFER_BYTES = 2**31 - 1


@functools.cache
def _arch_of(index: int) -> str:
    return get_gfx_runtime_for_device(index)


def on_supported_device(*tensors) -> bool:
    """True when every tensor is CUDA storage on one device whose arch the kernels target. A launch on anything else
    would hand a host or foreign-device address to the GPU, or issue an instruction the arch lacks.
    """
    dev = tensors[0].device
    return (
        dev.type == "cuda"
        and all(t.device == dev for t in tensors)
        and _arch_of(dev.index) == SUPPORTED_ARCH
    )


def run_on_device(exe, device: torch.device, *args):
    """``_run_compiled`` on ``device`` with that device's stream: one launcher (and compiled handle) per GPU, since
    a handle built on one GPU does not launch on another (see aiter/ops/flydsl/AGENTS.md, compiled-handle caches).
    """
    with torch.cuda.device(device):
        _run_compiled(
            _launcher_on(exe, device.index), *args, torch.cuda.current_stream(device)
        )


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def _split3(w):
    """4 fp32 (i32 bits) -> three terms of 2 bf16-pair dwords: RN(w), RN(w - w0), RN(w - w0 - w1), exact remainders."""
    parts = [[], [], []]
    for i in range_constexpr(2):
        a = fx.Int32(w[2 * i]).bitcast(fx.Float32)
        b = fx.Int32(w[2 * i + 1]).bitcast(fx.Float32)
        for p in range_constexpr(3):
            pk = _pk_bf16(a, b)
            parts[p].append(pk)
            if p < 2:
                a = a - (pk << 16).bitcast(fx.Float32)
                b = b - (pk & -65536).bitcast(fx.Float32)
    return parts
