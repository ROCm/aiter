# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks for the boundary between general A8W8 APIs and OPUS."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest
import torch

from aiter import dtypes
from aiter.ops import batched_gemm_op_a8w8 as batched_a8w8
from aiter.ops import gemm_op_a8w8 as general_a8w8
from aiter.ops.opus import gemm_op_a8w8 as opus_a8w8
from aiter.ops.opus import opus_gemm


def test_general_a8w8_restores_required_scales_and_blockscale_dtypes():
    fake_parameters = inspect.signature(general_a8w8.gemm_a8w8_fake).parameters
    assert fake_parameters["x_scale"].default is inspect.Parameter.empty
    assert fake_parameters["w_scale"].default is inspect.Parameter.empty

    schema = str(torch.ops.aiter.gemm_a8w8.default._schema)
    assert "x_scale=None" not in schema
    assert "w_scale=None" not in schema

    XQ = torch.empty((1, 1))
    WQ = torch.empty((1, 1))
    with pytest.raises(RuntimeError, match="x_scale"):
        general_a8w8.gemm_a8w8(XQ, WQ)

    scale = torch.empty((1, 1))
    with pytest.raises(
        AssertionError,
        match="Output dtype=torch.float32 is currently not supported",
    ):
        general_a8w8.gemm_a8w8_blockscale(
            XQ,
            WQ,
            scale,
            scale,
            dtype=torch.float32,
        )


def test_general_scaled_a8w8_keeps_legacy_backend_route(monkeypatch):
    calls = []
    result = torch.empty((2, 3), dtype=torch.bfloat16)

    def fake_ck(XQ, WQ, x_scale, w_scale, bias, dtype, splitK):
        calls.append((XQ, WQ, x_scale, w_scale, bias, dtype, splitK))
        return result

    monkeypatch.setattr(general_a8w8, "_ck_a8w8_supported", lambda: True)
    monkeypatch.setattr(general_a8w8, "gemm_a8w8_CK", fake_ck)

    XQ = torch.empty((2, 4), dtype=torch.int8)
    WQ = torch.empty((3, 4), dtype=torch.int8)
    x_scale = torch.empty((2, 1), dtype=torch.float32)
    w_scale = torch.empty((1, 3), dtype=torch.float32)
    actual = general_a8w8.gemm_a8w8(
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        splitK=3,
    )

    assert actual.data_ptr() == result.data_ptr()
    assert len(calls) == 1
    call = calls[0]
    assert call[0] is XQ
    assert call[1] is WQ
    assert call[2] is x_scale
    assert call[3] is w_scale
    assert call[4:] == (None, torch.bfloat16, 3)


@pytest.mark.parametrize(
    ("kid", "raw_name", "layout", "with_scale", "output_dtype"),
    [
        pytest.param(
            2,
            "_opus_gemm_a8w8_launch_raw",
            "plain",
            False,
            torch.float32,
            id="noscale",
        ),
        pytest.param(
            1,
            "_opus_gemm_a8w8_blockscale_launch_raw",
            "plain",
            True,
            torch.float32,
            id="blockscale",
        ),
        pytest.param(
            11000,
            "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
            "bpreshuffle",
            True,
            torch.bfloat16,
            id="blockscale-bpreshuffle",
        ),
    ],
)
def test_a8w8_opus_families_use_explicit_exact_kids(
    monkeypatch,
    kid,
    raw_name,
    layout,
    with_scale,
    output_dtype,
):
    calls = []
    raw_names = (
        "_opus_gemm_a8w8_launch_raw",
        "_opus_gemm_a8w8_blockscale_launch_raw",
        "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
    )

    for candidate in raw_names:

        def fake(*args, _name=candidate, **kwargs):
            calls.append((_name, args, kwargs))

        monkeypatch.setattr(opus_a8w8, candidate, fake)

    XQ = torch.empty((128, 256), dtype=dtypes.fp8)
    WQ = torch.empty((128, 256), dtype=dtypes.fp8)
    Y = torch.empty((128, 128), dtype=output_dtype)
    x_scale = torch.empty((128, 2), dtype=torch.float32)
    w_scale = torch.empty((1, 2), dtype=torch.float32)
    kwargs = {"kid": kid, "layout": layout}
    if with_scale:
        kwargs.update(x_scale=x_scale, w_scale=w_scale)

    assert opus_gemm(XQ, WQ, Y, **kwargs) is Y
    assert len(calls) == 1

    called_name, raw_args, raw_kwargs = calls[0]
    assert called_name == raw_name
    assert raw_kwargs == {}
    assert raw_args[-1] == kid

    raw_xq, raw_wq = raw_args[:2]
    assert raw_xq.shape == (1, *XQ.shape)
    assert raw_xq.data_ptr() == XQ.data_ptr()
    assert raw_wq.shape == (1, *WQ.shape)
    assert raw_wq.data_ptr() == WQ.data_ptr()

    if layout == "bpreshuffle":
        raw_x_scale, raw_w_scale, raw_y = raw_args[2:5]
    elif with_scale:
        raw_y, raw_x_scale, raw_w_scale = raw_args[2:5]
    else:
        assert len(raw_args) == 4
        raw_y = raw_args[2]

    assert raw_y.shape == (1, *Y.shape)
    assert raw_y.data_ptr() == Y.data_ptr()
    if with_scale:
        assert len(raw_args) == 6
        assert raw_x_scale is x_scale
        assert raw_w_scale is w_scale


def test_bpreshuffle_uses_opus_for_tuned_row(monkeypatch):
    opus_calls = []
    ck_calls = []
    config = {"libtype": "opus", "kernelId": 11000}

    def fake_opus_gemm(XQ, WQ, Y, **kwargs):
        opus_calls.append(kwargs)
        return Y

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, kernelName=""):
        ck_calls.append(kernelName)
        return Y

    from aiter.ops import opus

    monkeypatch.setattr(general_a8w8, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(general_a8w8, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(general_a8w8, "get_CKGEMM_config", lambda *_args: config)
    monkeypatch.setattr(
        general_a8w8,
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        fake_ck,
    )
    monkeypatch.setattr(opus, "opus_gemm", fake_opus_gemm)

    XQ = torch.empty((2, 128), dtype=dtypes.fp8)
    WQ = torch.empty((128, 128), dtype=dtypes.fp8)
    x_scale = torch.empty((2, 1), dtype=torch.float32)
    w_scale = torch.empty((1, 1), dtype=torch.float32)

    bf16_result = general_a8w8.gemm_a8w8_blockscale_bpreshuffle(
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
    )
    assert bf16_result.dtype == torch.bfloat16
    assert len(opus_calls) == 1
    assert opus_calls[0]["kid"] == 11000
    assert opus_calls[0]["layout"] == "bpreshuffle"
    assert opus_calls[0]["x_scale"] is x_scale
    assert opus_calls[0]["w_scale"] is w_scale
    assert ck_calls == []


def test_mxscale_launch_plan_cache_is_bounded(monkeypatch):
    calls = []

    def resolve(g, m, n, k, *, w_scale_block="128x128"):
        calls.append((g, m, n, k, w_scale_block))
        return 8000, 1

    monkeypatch.setattr(
        batched_a8w8,
        "_resolve_a8w8_mxscale_bmm_plan",
        resolve,
    )
    batched_a8w8._get_mxscale_bmm_launch_plan.cache_clear()
    try:
        for m in range(1025):
            assert batched_a8w8._get_mxscale_bmm_launch_plan(2, m, 1024, 4096) == (
                8000,
                1,
            )

        cache = batched_a8w8._get_mxscale_bmm_launch_plan.cache_info()
        assert cache.maxsize == 1024
        assert cache.currsize == 1024
        assert len(calls) == 1025
    finally:
        batched_a8w8._get_mxscale_bmm_launch_plan.cache_clear()


def test_mxscale_invalid_tuned_kid_warns_and_uses_heuristic(
    monkeypatch,
    tmp_path,
):
    from aiter.ops.opus import policy

    config_path = tmp_path / "mxscale.csv"
    config_path.write_text(
        "gfx,b,m,n,k,w_scale_block,libtype,kernelId,splitK\n"
        "gfx950,2,1,1024,4096,128x128,opus,8001,1\n"
        "gfx950,3,1,1024,4096,128x128,other,42,1\n"
    )
    warnings = []
    monkeypatch.setattr(
        policy,
        "AITER_CONFIGS",
        SimpleNamespace(
            AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE=str(config_path)
        ),
    )
    monkeypatch.setattr(policy, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(policy, "get_cu_num", lambda: 256)
    monkeypatch.setattr(policy, "get_padded_m", lambda m, _n, _k, _gl: m)
    monkeypatch.setattr(
        policy.logger,
        "warning",
        lambda *args, **_kwargs: warnings.append(args),
    )
    policy._load_mxscale_bmm_tuned.cache_clear()
    policy.lookup_mxscale_bmm_config.cache_clear()
    try:
        rows = policy._load_mxscale_bmm_tuned(None, False)
        assert rows[("gfx950", 3, 1, 1024, 4096, "128x128")]["kernelId"] == 42
        assert policy.resolve_a8w8_mxscale_bmm_plan(2, 1, 1024, 4096) == (
            8640,
            1,
        )
        assert len(warnings) == 1
        assert warnings[0][0].startswith("Skipping %d invalid OPUS row")
    finally:
        policy.lookup_mxscale_bmm_config.cache_clear()
        policy._load_mxscale_bmm_tuned.cache_clear()


def test_mxscale_tuned_lookup_keys_on_cu_count(monkeypatch, tmp_path):
    from aiter.ops.opus import policy

    config_path = tmp_path / "mxscale.csv"
    monkeypatch.setattr(
        policy,
        "AITER_CONFIGS",
        SimpleNamespace(
            AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE=str(config_path)
        ),
    )
    monkeypatch.setattr(policy, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        policy,
        "_get_cached_a8w8_mxscale_bmm_plan",
        lambda *_args, **_kwargs: object(),
    )

    def lookup_kid(table: str, cu_num: int) -> int:
        config_path.write_text(table)
        monkeypatch.setattr(policy, "get_cu_num", lambda: cu_num)
        policy._load_mxscale_bmm_tuned.cache_clear()
        policy.lookup_mxscale_bmm_config.cache_clear()
        return policy.lookup_mxscale_bmm_config(2, 1, 1024, 4096)["kernelId"]

    header = "gfx,cu_num,b,m,n,k,w_scale_block,libtype,kernelId,splitK\n"
    shape = "2,1,1024,4096,128x128,opus"
    try:
        # A bound row serves its own CU count; a blank cu_num serves every other.
        mixed = (
            header
            + f"gfx950,128,{shape},8311,1\n"
            + f"gfx950,256,{shape},8312,1\n"
            + f"gfx950,,{shape},8653,1\n"
        )
        assert lookup_kid(mixed, 128) == 8311
        assert lookup_kid(mixed, 256) == 8312
        assert lookup_kid(mixed, 304) == 8653

        # A table without the column serves every CU count.
        legacy = (
            "gfx,b,m,n,k,libtype,kernelId,splitK\ngfx950,2,1,1024,4096,opus,8311,1\n"
        )
        assert lookup_kid(legacy, 128) == 8311

        # Blank and negative cu_num count as 0 in the duplicate check.
        for first, second in (("128", "128"), ("", "0"), ("-1", "0")):
            duplicates = (
                header
                + f"gfx950,{first},{shape},8311,1\n"
                + f"gfx950,{second},{shape},8312,1\n"
            )
            with pytest.raises(RuntimeError, match="duplicate"):
                lookup_kid(duplicates, 128)
    finally:
        policy.lookup_mxscale_bmm_config.cache_clear()
        policy._load_mxscale_bmm_tuned.cache_clear()


def test_mxscale_tuner_keys_on_device_cu_count(monkeypatch, tmp_path):
    from csrc.opus_gemm.opus_bmm_mxscale_tune import OpusBmmMxscaleTuner

    tuner = OpusBmmMxscaleTuner()
    monkeypatch.setattr(tuner, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(tuner, "get_cu_num", lambda: 256)
    header = "gfx,cu_num,b,m,n,k,w_scale_block,libtype,kernelId,splitK\n"

    # Shapes tuned for any CU count (0 or -1) are skipped; one tuned only for
    # 128 CUs is retuned, retargeted to the device's 256 like the new shape.
    untuned_path = tmp_path / "untuned.csv"
    untuned_path.write_text(
        "b,m,n,k,w_scale_block\n"
        "2,1,1024,4096,128x128\n"
        "2,16,1024,4096,128x128\n"
        "2,64,1024,4096,128x128\n"
        "2,128,1024,4096,128x128\n"
    )
    tuned_path = tmp_path / "tuned.csv"
    tuned_path.write_text(
        header
        + "gfx950,0,2,1,1024,4096,128x128,opus,8311,1\n"
        + "gfx950,128,2,16,1024,4096,128x128,opus,8311,1\n"
        + "gfx950,-1,2,128,1024,4096,128x128,opus,8311,1\n"
    )
    tuner.pre_process(
        tuner.parser.parse_args(["-i", str(untuned_path), "-o", str(tuned_path)])
    )
    assert tuner.untunedf[["cu_num", "m"]].values.tolist() == [[256, 16], [256, 64]]

    # --run_config benchmarks the device's rows and the unbound ones, not rows
    # for another CU count or arch.
    saved_path = tmp_path / "saved.csv"
    saved_path.write_text(
        header
        + "gfx950,0,2,1,1024,4096,128x128,opus,8311,1\n"
        + "gfx950,256,2,16,1024,4096,128x128,opus,8311,1\n"
        + "gfx950,128,2,64,1024,4096,128x128,opus,8311,1\n"
        + "gfx1250,0,2,128,1024,4096,128x128,flydsl,-1,1\n"
    )
    benchmarked = []
    monkeypatch.setattr(
        tuner, "run_config", lambda _args: benchmarked.extend(tuner.untunedf["m"])
    )
    tuner.run(
        tuner.parser.parse_args(
            ["--run_config", str(saved_path), "-o", str(tmp_path / "out.csv")]
        )
    )
    assert sorted(benchmarked) == [1, 16]

    # Merging tables that all lack cu_num leaves every row unbound.
    first_path, second_path = tmp_path / "first.csv", tmp_path / "second.csv"
    first_path.write_text(
        "gfx,b,m,n,k,w_scale_block,libtype,kernelId,splitK,us\n"
        "gfx950,2,1,1024,4096,128x128,opus,8311,1,10.0\n"
    )
    second_path.write_text(
        "gfx,b,m,n,k,w_scale_block,libtype,kernelId,splitK,us\n"
        "gfx950,2,16,1024,4096,128x128,opus,8311,1,9.0\n"
    )
    merged = tuner.get_tuned_gemm_list(f"{first_path}{os.pathsep}{second_path}")
    assert sorted(merged[["cu_num", "m"]].values.tolist()) == [[0, 1], [0, 16]]

    # Writing results into a table without cu_num leaves its rows unbound and
    # adds the column after gfx. The first sort is what a run with nothing to
    # tune ends with.
    legacy_path = tmp_path / "legacy.csv"
    legacy_path.write_text(
        "gfx,b,m,n,k,w_scale_block,libtype,kernelId,splitK,us\n"
        "gfx950,2,1,1024,4096,128x128,opus,8311,1,10.0\n"
        "gfx1250,2,1,1024,4096,128x128,flydsl,-1,1,10.0\n"
    )
    tuner.sortResults(str(legacy_path), True, tuner.sort_keys)
    for concat in (True, False):
        result = tuner.result_to_df(
            [((("gfx950", 256, 2, 16, 1024, 4096, "128x128"), 8311, 1, ""), 9.0, 0.0)]
        )
        tuner.result_to_csv(result, str(legacy_path), concat=concat)
        tuner.sortResults(str(legacy_path), True, tuner.sort_keys)

    assert legacy_path.read_text().startswith("gfx,cu_num,b,m,n,k,w_scale_block,")
    rows = tuner.get_tuned_gemm_list(str(legacy_path))
    assert sorted(rows[["gfx", "cu_num", "m"]].values.tolist()) == [
        ["gfx1250", 0, 1],
        ["gfx950", 0, 1],
        ["gfx950", 256, 16],
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
