# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every row of the MXFP8 decode GEMM table is a valid kernel config; decode rows take 16-row tiles."""

import pytest

table = pytest.importorskip("aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250")
kern = pytest.importorskip("aiter.ops.flydsl.kernels.mxfp8_decode_gemm_gfx1250")


@pytest.mark.parametrize("nk", sorted(table._TABLE))
def test_decode_gemm_table(nk):
    n, k = nk
    for _, (tile_m, nb, ks, kw, wn, wdepth, _wcpol, atdm) in table._TABLE[nk]:
        err = kern.decode_gemm_config_error(n, k, tile_m, nb, ks, kw, wn, wdepth, atdm)
        assert err is None, err
    assert table.mxfp8_decode_gemm_config(8, 6144, 2048)[:2] == (16, 2)
    assert table.mxfp8_decode_gemm_config(17, 6144, 2048)[:2] == (32, 3)
