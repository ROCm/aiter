# SPDX-License-Identifier: MIT
"""Minimal Stage-1-shaped BF16 -> MXFP4 quantization diagnostic.

This is intentionally not a production entry point.  It preserves the fused
Stage-1 mapping of one 256-thread CTA per token while removing ticketing, LDS,
route metadata, ready publication, and communication.  The kernel therefore
isolates the quantization arithmetic and the 128-CTA geometry from the larger
persistent Stage-1 code object.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from aiter.ops.flydsl.kernels import buffer_ops


BLOCK = 256
GROUP = 32
_FP4_INV_MAX_POS_BITS = 0x3E2AAAAB
_LAUNCHER_CACHE = {}


def build_megamoe_tile_quant_core(n: int):
    """Build the one-CTA-per-token diagnostic for BF16 ``[m, n]`` input."""

    n = int(n)
    if n <= 0 or n % GROUP:
        raise ValueError(f"n={n} must be positive and divisible by {GROUP}")
    scale_n = n // GROUP
    if scale_n > BLOCK:
        raise ValueError(
            f"one-CTA quant core requires n/32 <= {BLOCK}, got {scale_n}"
        )

    @flyc.kernel(
        name=f"megamoe_tile_quant_core_fp4_n{n}",
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(x: fx.Tensor, y: fx.Tensor, scale: fx.Tensor, m: fx.Int32):
        token = fx.block_idx.x
        group = fx.thread_idx.x
        if (token < m) & (group < fx.Int32(scale_n)):
            x_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
            y_rsrc = buffer_ops.create_buffer_resource(y, max_size=True)
            scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

            in_dw = token * fx.Int32(n // 2) + group * fx.Int32(16)
            values = []
            local_max = fx.Float32(1e-10)
            for chunk in range_constexpr(4):
                raw = buffer_ops.buffer_load(
                    x_rsrc,
                    in_dw + fx.Int32(chunk * 4),
                    vec_width=4,
                    dtype=T.i32,
                )
                vals = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                local_max = local_max.maximumf(
                    fmath.absf(vals).reduce(ReductionOp.MAX)
                )
                for elem in range_constexpr(8):
                    values.append(vals[elem])

            working = (
                local_max
                * fx.Int32(_FP4_INV_MAX_POS_BITS).bitcast(fx.Float32)
            ).bitcast(fx.Int32)
            mantissa = working & fx.Int32(0x7FFFFF)
            biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
            e8m0 = (mantissa != fx.Int32(0)).select(
                biased_exp + fx.Int32(1), biased_exp
            )
            e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
            qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)

            words = []
            for word in range_constexpr(4):
                packed = fx.Int32(0)
                for pair in range_constexpr(4):
                    idx = word * 8 + pair * 2
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        packed,
                        values[idx],
                        values[idx + 1],
                        qscale,
                        pair,
                    )
                words.append(packed)
            out_dw = token * fx.Int32(n // 8) + group * fx.Int32(4)
            buffer_ops.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                y_rsrc,
                out_dw,
            )
            buffer_ops.buffer_store(
                e8m0.to(fx.Uint8),
                scale_rsrc,
                token * fx.Int32(scale_n) + group,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(x, y, scale, m).launch(
            grid=(fx.Int64(m), 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    launch.kernel_name = f"megamoe_tile_quant_core_fp4_n{n}"
    launch.grid_contract = "one_block_per_token"
    launch.block_threads = BLOCK
    launch.writes_metadata = False
    launch.publishes_ready = False
    launch.uses_ticket = False
    return launch


def _get_launcher(n: int):
    n = int(n)
    launcher = _LAUNCHER_CACHE.get(n)
    if launcher is None:
        launcher = build_megamoe_tile_quant_core(n)
        _LAUNCHER_CACHE[n] = launcher
    return launcher


def megamoe_tile_quant_core(x: torch.Tensor, stream=None):
    """Run the minimal Stage-1-shaped quant core and return FP4/E8M0 tensors."""

    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError(f"x must be a 2D BF16 tensor, got {x.dtype} {tuple(x.shape)}")
    x = x.contiguous()
    m, n = x.shape
    if n % GROUP:
        raise ValueError(f"n={n} must be divisible by {GROUP}")
    y = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    scale = torch.empty((m, n // GROUP), dtype=torch.uint8, device=x.device)
    fx_stream = fx.Stream(
        stream if stream is not None else torch.cuda.current_stream().cuda_stream
    )
    _get_launcher(n)(x, y, scale, int(m), stream=fx_stream)
    return y.view(torch.float4_e2m1fn_x2), scale


__all__ = ["build_megamoe_tile_quant_core", "megamoe_tile_quant_core"]
