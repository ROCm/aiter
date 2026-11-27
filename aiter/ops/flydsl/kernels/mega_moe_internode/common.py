# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""E8M0 helpers MegaMoEInternode does not share with ``mxfp4_gemm_common``.

Everything else GMM1 / GMM2 need comes from ``aiter.ops.flydsl.kernels.mxfp4_gemm_common``
directly; ``_e8m0_*`` here compare against an unsigned constant.
"""

import flydsl.expr as fx


def _e8m0_roundup(amax_f32, max_norm=6.0):
    wi = (amax_f32 * fx.Float32(1.0 / float(max_norm))).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = fx.Uint32(bexp) < fx.Uint32(254)
    return lt.select(bexp, fx.Int32(254))


def _e8m0_from_amax(amax_f32, max_norm=6.0):
    e8m0 = _e8m0_roundup(amax_f32, max_norm=max_norm)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale
