# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared helpers: bf16x3 weight split of the decode-shape router GEMM.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4`` (``router_gemm.py``, ``router_topk.py``,
``common/loaders.py``, ``i3072/gemm1_prefill_h6144_i3072_persistent.py``, ``mxfp8/gemm1_prefill_h6144_i3072.py``),
Apache-2.0, "Copyright contributors to the amd-kernels project".
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


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
