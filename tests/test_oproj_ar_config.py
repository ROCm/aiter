# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The o_proj + all-reduce launch's tile is a valid decode-GEMM tile that its receivers can split."""

import pytest

oproj_ar = pytest.importorskip("aiter.ops.flydsl.oproj_ar")
kern = pytest.importorskip("aiter.ops.flydsl.kernels.oproj_ar_gfx1250")
gemm = pytest.importorskip("aiter.ops.flydsl.kernels.mxfp8_decode_gemm_gfx1250")


def test_oproj_ar_tile():
    tile_m, nb, kw, wn, wdepth, wcpol, atdm = oproj_ar._CFG
    err = gemm.decode_gemm_config_error(
        kern.HIDDEN, 2048, tile_m, nb, 1, kw, wn, wdepth, atdm
    )
    assert err is None, err
    assert kw * wn == 2 * kern.RWAVES  # two receiver rows a workgroup
    assert kern.SLAB % (nb * 16 // 8) == 0  # a workgroup's packs stay in one slab
