# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# AIESW-32176: thin Python wrapper for the CK WMMA W4A16 b_scale GEMM.

from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops


def _gemm_w4a16_fake(
    in_a: Tensor,
    in_b: Tensor,
    in_s: Tensor,
    Y: Tensor,
    group_size: int,
    scaled_zp: Optional[Tensor] = None,
    pre_dequant_to_lds: Optional[bool] = None,
    truncate_bf16_round: Optional[bool] = None,
) -> Tensor:
    return Y


@compile_ops("module_gemm_w4a16", gen_fake=_gemm_w4a16_fake)
def gemm_w4a16(
    in_a: Tensor,
    in_b: Tensor,
    in_s: Tensor,
    Y: Tensor,
    group_size: int,
    scaled_zp: Optional[Tensor] = None,
    pre_dequant_to_lds: Optional[bool] = None,
    truncate_bf16_round: Optional[bool] = None,
) -> Tensor:
    """CK WMMA W4A16 b_scale GEMM (gfx1151 / RDNA 3.5).

    Single dispatch covering the symmetric (uint4b8) and asymmetric (AWQ
    per-group zero-point) variants of the W4A16 prefill GEMM tuned for
    Qwen3-4B (AIESW-32176).

    Args:
        in_a:        [M, K] activation, fp16 or bf16, row-major contiguous.
        in_b:        [K0, N, K1/2] int8 in CK pk_i4_v3 b_scale layout
                     (K0 = K/KPerBlock=K/32, K1 = KPerBlock=32).
        in_s:        [N, K/G] activation-dtype scales, contiguous row-major.
        Y:           [M, N] activation-dtype, caller-allocated output.
                     Output dtype determines the kernel template instantiation.
        group_size:  AWQ per-group quantization granularity. Must be one of
                     ``{32, 128}`` — selects the matching ScaleBlockK template
                     instantiation of the CK kernel.
        scaled_zp:   Optional [N, K/G] activation-dtype, ``None`` for symmetric
                     (uint4b8). For asymmetric AWQ pass ``(zp - 8) * scale``
                     precomputed at weight load time. Costs one extra
                     activation-dtype vector subtract per dequant pack vs the
                     symmetric path.
        pre_dequant_to_lds:
                     Optional bool, defaults to ``False``. ``False`` selects the
                     existing fused-dequant baseline (CK dequants int4 inside
                     the WMMA inner loop). ``True`` selects the pre-dequant-to
                     -LDS variant (dequant once per K-block into activation-
                     dtype B in LDS scratch, WMMA reads activation-dtype B
                     from LDS). The ``True`` path is currently **STUBBED** and
                     will raise at runtime — see TODO(AIESW-32282) in
                     ``csrc/ck_w4a16/include/gemm_w4a16_common.cuh``.
        truncate_bf16_round:
                     Optional bool, defaults to ``False``. ``False`` keeps the
                     existing IEEE round-to-nearest-even fp32->bf16 step in
                     the bf16 dequant. ``True`` switches to a bit-cast
                     truncate (drops the upper 16 bits of fp32 directly,
                     saving ~3 RDNA3.5 VALU instructions per nibble at
                     <0.5 ULP of bf16 error vs RTE). Silently ignored on the
                     fp16 path (fp16 already uses the optimal bit-trick).
                     True runtime template flag — both flavors live in the
                     same JIT-built ``.so`` as distinct template-mangled
                     symbols (no rebuild required to flip).

    Returns:
        ``Y``, populated in place with ``in_a @ dequant(in_b, in_s, scaled_zp).T``.
    """
    ...
