# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM helpers MegaMoEInternode does not share with ``mxfp4_gemm_common``.

Everything else GMM1 / GMM2 need comes from ``aiter.ops.flydsl.kernels.mxfp4_gemm_common``
directly; only the definitions that differ live here:

* ``kbs_c_k1_for`` pads the B-scale K groups to eight, as ``shuffle_scale_a16w4`` /
  ``e8m0_shuffle`` do (the shared one truncates; equal on every shape whose
  ``K / 32`` is a multiple of 8), and the strides built on it;
* ``_e8m0_*`` compare against an unsigned constant.
"""

import flydsl.expr as fx

from ..mxfp4_gemm_common import kbs_c_n1_for


def _e8m0_roundup(amax_f32):
    wi = (amax_f32 * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = fx.Uint32(bexp) < fx.Uint32(254)
    return lt.select(bexp, fx.Int32(254))


def _e8m0_from_amax(amax_f32):
    e8m0 = _e8m0_roundup(amax_f32)
    qscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, qscale


def kbs_c_k1_for(k):
    # Match shuffle_scale_a16w4/e8m0_shuffle's eight-group K padding.
    return ((k // 32) + 7) // 8


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64


def kbs_per_expert_dw_for(n, k):
    return kbs_c_n1_for(n) * kbs_stride_n0_dw_for(k)


def bscale_bytes_for(ne, n, k):
    return ne * kbs_per_expert_dw_for(n, k) * 4
