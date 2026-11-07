# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

pytest.importorskip("flydsl")

from aiter.aot.flydsl.gemm import parse_csv


def test_decode_mxfp4_jobs(tmp_path):
    csv_path = tmp_path / "a4w4_blockscale_tuned_gemm.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,1,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
        "gfx950,256,2,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
    )
    jobs = parse_csv(str(csv_path))
    assert [job["m"] for job in jobs] == [1, 2]
    for job in jobs:
        assert job["table"] == "a4w4_blockscale_tuned_gemm"
        assert (job["n"], job["k"]) == (8192, 5120)
        assert job["kernel_name"] == "flydsl_decode_t16x32x512_kw2_nb4_sk1"
        assert job["m_low"] == job["m"]


def test_decode_mxfp4_arch_filter(tmp_path):
    csv_path = tmp_path / "a4w4_blockscale_tuned_gemm.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelName\n"
        "gfx942,304,1,8192,5120,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
    )
    assert parse_csv(str(csv_path)) == []


def test_decode_mxfp4_exact_m_only(tmp_path):
    # Padded-M FlyDSL lookups fall back to asm, so only exact rows need AOT.
    csv_path = tmp_path / "a4w4_blockscale_tuned_gemm.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,2,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
        "gfx950,256,4,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
        "gfx950,256,8,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb6_sk1\n"
        "gfx950,256,16,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb6_sk1\n"
    )
    jobs = parse_csv(str(csv_path))
    assert sorted(j["m"] for j in jobs if "_nb4_" in j["kernel_name"]) == [2, 4]
    assert sorted(j["m"] for j in jobs if "_nb6_" in j["kernel_name"]) == [8, 16]
    assert len({j["m"] for j in jobs}) == len(jobs) == 4
    assert all(j["m_low"] == j["m"] for j in jobs)


def test_decode_mxfp4_exact_row_shadows_padded(tmp_path):
    # A non-FlyDSL row at M=3 wins for M=3, so the M=4 FlyDSL row does not serve it.
    csv_path = tmp_path / "a4w4_blockscale_tuned_gemm.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,3,8192,5120,1,0,1,_ZN5aiter_asm\n"
        "gfx950,256,4,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk1\n"
    )
    assert sorted(j["m"] for j in parse_csv(str(csv_path))) == [4]


def test_decode_mxfp4_public_replay(tmp_path, monkeypatch):
    from aiter.aot.flydsl import gemm, replay
    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.jit.core import AITER_CONFIG
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config

    csv_path = tmp_path / "a4w4_blockscale_tuned_gemm.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName\n"
        "gfx950,256,4,8192,5120,-1,0,1,flydsl_decode_t16x32x512_kw2_nb4_sk2\n"
    )
    monkeypatch.setattr(
        AITER_CONFIG,
        "AITER_CONFIG_GEMM_A4W4_FILE",
        property(lambda self: str(csv_path)),
    )
    monkeypatch.delattr(get_GEMM_config, "gemm_dict", raising=False)
    monkeypatch.delattr(get_GEMM_config, "has_gfx", raising=False)
    get_GEMM_config.cache_clear()
    try:
        with dry_run("gfx950", 256, python_ops=True):
            for job in parse_csv(str(csv_path)):
                with replay.identified_kernels() as kernels:
                    gemm.replay(job)
                # Split-K must reach both the decode and reduction launchers.
                assert len(kernels) == 2
    finally:
        get_GEMM_config.cache_clear()
