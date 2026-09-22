# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contracts for A8W8 format dispatch and JSON-defined GEMM bounds."""

import json

import pytest
import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.ops import gemm_op_a8w8
from aiter.ops.triton.gemm.basic import gemm_a8w8_blockscale_group32 as group32
from aiter.ops.triton.utils import gemm_config_utils
from aiter.ops.triton.utils.config_utils import load_config_json


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gemm_config_utils, "resolve_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    load_config_json.cache_clear()
    yield tmp_path
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    load_config_json.cache_clear()


def test_file_bounds_and_explicit_override(config_dir):
    table = {
        "M_BOUNDS": [3, 7],
        "M_LEQ_3": {"value": 3},
        "M_LEQ_7": {"value": 7},
        "any": {"value": 99},
    }
    (config_dir / "DEFAULT.json").write_text(json.dumps(table))
    assert gemm_config_utils.get_gemm_config("GEMM-TEST", 2)[0]["value"] == 3
    assert gemm_config_utils.get_gemm_config("GEMM-TEST", 4)[0]["value"] == 7
    assert gemm_config_utils.get_gemm_config("GEMM-TEST", 8)[0]["value"] == 99
    assert (
        gemm_config_utils.get_gemm_config("GEMM-TEST", 2, bounds=(7,))[0]["value"] == 7
    )


def test_legacy_bounds_and_specialized_nested_copy(config_dir):
    (config_dir / "DEFAULT.json").write_text(
        json.dumps(
            {"M_LEQ_3": {"value": 3}, "M_LEQ_4": {"value": 4}, "any": {"value": 99}}
        )
    )
    assert gemm_config_utils.get_gemm_config("GEMM-TEST", 2)[0]["value"] == 4
    (config_dir / "GEMM-TEST-N=64-K=32.json").write_text(
        json.dumps(
            {
                "M_BOUNDS": [3],
                "M_LEQ_3": {"packed": {"K_PACK": 4}, "nested": [{"value": 1}]},
                "any": {"value": 99},
            }
        )
    )
    first, tuned = gemm_config_utils.get_gemm_config("GEMM-TEST", 2, 64, 32)
    assert tuned
    first["packed"]["K_PACK"] = 1
    first["nested"][0]["value"] = 2
    assert gemm_config_utils.get_gemm_config("GEMM-TEST", 2, 64, 32)[0]["nested"] == [
        {"value": 1}
    ]
    assert (
        gemm_config_utils.get_gemm_config("GEMM-TEST", 2, 64, 32)[0]["packed"]["K_PACK"]
        == 4
    )


@pytest.mark.parametrize("bounds", [[], [4, 3], [3, 3], [0, 4], [1.5, 4], [True, 4]])
def test_invalid_file_bounds(config_dir, bounds):
    (config_dir / "DEFAULT.json").write_text(
        json.dumps({"M_BOUNDS": bounds, "any": {"value": 1}})
    )
    with pytest.raises(AssertionError, match="M_BOUNDS"):
        gemm_config_utils.get_gemm_config("GEMM-TEST", 2)


@pytest.mark.parametrize("group_n", [1, 32])
@pytest.mark.parametrize("libtype", [None, "triton", "ck", "cktile", "unknown"])
def test_public_native_group32_route(monkeypatch, group_n, libtype):
    calls = []
    expected = torch.empty((3, 65), dtype=torch.float32)

    def backend(x, w, xs, ws, **kwargs):
        calls.append(kwargs)
        return expected

    def lookup(m, n, k, tuned_file):
        assert (m, n, k) == (3, 65, 64)
        assert tuned_file.endswith("a8w8_blockscale_group32_tuned_gemm.csv")
        calls.append("lookup")
        return None if libtype is None else {"libtype": libtype}

    monkeypatch.setattr(gemm_op_a8w8, "get_CKGEMM_config", lookup)
    monkeypatch.setattr(group32, "gemm_a8w8_blockscale_group32", backend)
    x = torch.empty((3, 64), dtype=torch.float8_e4m3fn)
    w = torch.empty((65, 64), dtype=torch.float8_e4m3fn)
    xs = torch.empty((3, 2), dtype=torch.float8_e8m0fnu)
    ws = torch.empty(((65 + group_n - 1) // group_n, 2), dtype=torch.float8_e8m0fnu)
    if libtype not in (None, "triton"):
        with pytest.raises(AssertionError, match="Unsupported libtype"):
            gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws, dtype=torch.float32)
        assert calls == ["lookup"]
        return
    actual = gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws, dtype=torch.float32)
    assert actual is expected
    assert calls == ["lookup", {"dtype": torch.float32, "weight_group_rows": group_n}]


@pytest.mark.parametrize("libtype", ["ck", "cktile"])
def test_public_legacy_ck_route_and_readonly_schema(monkeypatch, libtype):
    calls = []

    def ck(x, w, xs, ws, out, **kwargs):
        calls.append(kwargs)
        return out

    monkeypatch.setattr(gemm_op_a8w8, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(
        gemm_op_a8w8,
        "get_CKGEMM_config",
        lambda *args: {"libtype": libtype, "splitK": 2, "kernelName": "existing"},
    )
    monkeypatch.setattr(gemm_op_a8w8, f"gemm_a8w8_blockscale_{libtype}", ck)
    x = torch.empty((3, 128), dtype=torch.float8_e4m3fn)
    w = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
    xs = torch.empty((3, 1), dtype=torch.float32)
    ws = torch.empty((1, 1), dtype=torch.float32)
    actual = gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws)
    assert actual.shape == (3, 128)
    assert calls == [{"splitK": 2, "kernelName": "existing"}]
    schema = torch.ops.aiter.gemm_a8w8_blockscale.default._schema
    assert all(arg.alias_info is None for arg in schema.arguments)


@pytest.fixture
def backend_config_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a8w8, "get_cu_num", lambda: 256)
    monkeypatch.setattr(gemm_op_a8w8, "_CKGEMM_CONFIG_CACHE", {})
    monkeypatch.setattr(gemm_op_a8w8, "_CKGEMM_HAS_GFX", {})
    gemm_op_a8w8.get_CKGEMM_config.cache_clear()
    AITER_CONFIGS.get_config_file.cache_clear()
    files = {}
    for suffix in ("", "_GROUP32"):
        path = tmp_path / f"blockscale{suffix.lower()}.csv"
        monkeypatch.setenv(f"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE{suffix}", str(path))
        files[suffix] = path
    yield files
    gemm_op_a8w8.get_CKGEMM_config.cache_clear()
    AITER_CONFIGS.get_config_file.cache_clear()


@pytest.mark.parametrize("libtype", [None, "triton", "unknown"])
def test_scale_formats_have_independent_backend_configs(
    backend_config_files, monkeypatch, libtype
):
    # Identical M/N/K must not make the FP32 128x128 winner apply to E8M0.
    header = "gfx,cu_num,M,N,K,libtype,splitK,kernelName\n"
    backend_config_files[""].write_text(header + "gfx950,256,4,128,128,ck,2,legacy\n")
    backend_config_files["_GROUP32"].write_text(
        header
        + ("" if libtype is None else f"gfx950,256,4,128,128,{libtype},0,native\n")
    )
    calls = []

    def ck(x, w, xs, ws, out, **kwargs):
        assert xs.dtype == ws.dtype == torch.float32
        assert kwargs == {"splitK": 2, "kernelName": "legacy"}
        calls.append("ck")
        return out

    def native(x, w, xs, ws, **kwargs):
        assert xs.dtype == ws.dtype == torch.float8_e8m0fnu
        calls.append("group32")
        return torch.empty((4, 128), dtype=kwargs["dtype"])

    monkeypatch.setattr(gemm_op_a8w8, "gemm_a8w8_blockscale_ck", ck)
    monkeypatch.setattr(group32, "gemm_a8w8_blockscale_group32", native)
    x = torch.empty((4, 128), dtype=torch.float8_e4m3fn)
    w = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
    gemm_op_a8w8.gemm_a8w8_blockscale(x, w, torch.empty((4, 1)), torch.empty((1, 1)))
    assert calls == ["ck"]
    xs = torch.empty((4, 4), dtype=torch.float8_e8m0fnu)
    ws = torch.empty((4, 4), dtype=torch.float8_e8m0fnu)
    if libtype == "unknown":
        with pytest.raises(AssertionError, match="Unsupported libtype unknown"):
            gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws)
        assert calls == ["ck"]
    else:
        gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws)
        assert calls == ["ck", "group32"]


@pytest.mark.parametrize("configured", [False, True])
def test_legacy_triton_config_and_fallback(monkeypatch, configured):
    calls = []

    def lookup(*args):
        calls.append("lookup")
        return {"libtype": "triton"} if configured else None

    def triton(x, w, xs, ws, dtype, **kwargs):
        calls.append("triton")
        return torch.empty((3, 128), dtype=dtype)

    monkeypatch.setattr(gemm_op_a8w8, "get_CKGEMM_config", lookup)
    monkeypatch.setattr(gemm_op_a8w8, "_hip_blockscale_supported", lambda: False)
    monkeypatch.setattr(gemm_op_a8w8, "_blockscale_triton", triton)
    gemm_op_a8w8.gemm_a8w8_blockscale(
        torch.empty((3, 128), dtype=torch.float8_e4m3fn),
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        torch.empty((3, 1)),
        torch.empty((1, 1)),
    )
    assert calls == ["lookup", "triton"]
