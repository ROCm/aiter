# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

pytest.importorskip("flydsl")

from aiter.aot.flydsl.gemm import parse_csv


def test_decode_mxfp4_jobs(tmp_path):
    csv_path = tmp_path / "a4w4.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,1,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
        "gfx950,256,2,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
    )
    jobs = parse_csv(str(csv_path))
    assert [job["m"] for job in jobs] == [1, 2]
    for job in jobs:
        assert job["kind"] == "decode"
        assert (job["n"], job["k"]) == (8192, 5120)
        assert (job["tile_n"], job["k_waves"], job["num_buffers"], job["split_k"]) == (
            32,
            2,
            4,
            1,
        )


def test_decode_mxfp4_arch_filter(tmp_path):
    csv_path = tmp_path / "a4w4.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelName\n"
        "gfx942,304,1,8192,5120,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
    )
    assert parse_csv(str(csv_path)) == []


def test_decode_mxfp4_padded_m_expansion(tmp_path):
    # Runtime pads M=3 to the M=4 row and M=5..7 to the M=8 row; an exact row
    # takes precedence over padded resolution.
    csv_path = tmp_path / "a4w4.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,2,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
        "gfx950,256,4,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
        "gfx950,256,8,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb6_sk1\n"
        "gfx950,256,16,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb6_sk1\n"
    )
    jobs = parse_csv(str(csv_path))
    # M=1 has no row at any padding level, so it is not enrolled.
    assert sorted(j["m"] for j in jobs if j["num_buffers"] == 4) == [2, 3, 4]
    assert sorted(j["m"] for j in jobs if j["num_buffers"] == 6) == list(range(5, 17))
    assert len({j["m"] for j in jobs}) == len(jobs) == 15


def test_decode_mxfp4_exact_row_shadows_padded(tmp_path):
    # A non-FlyDSL row at M=3 wins for M=3, so the M=4 FlyDSL row does not serve it.
    csv_path = tmp_path / "a4w4.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,3,8192,5120,1,0,1,_ZN5aiter_asm\n"
        "gfx950,256,4,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
    )
    assert sorted(j["m"] for j in parse_csv(str(csv_path))) == [4]
