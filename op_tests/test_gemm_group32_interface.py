# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU contracts for A8W8 format dispatch and JSON-defined GEMM bounds."""

import functools
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
@pytest.mark.parametrize("split_k", [None, 3])
@pytest.mark.parametrize(
    "scale_dtypes",
    [
        (torch.float8_e8m0fnu, torch.float8_e8m0fnu),
        (torch.uint8, torch.uint8),
        (torch.float8_e8m0fnu, torch.uint8),
        (torch.uint8, torch.float8_e8m0fnu),
    ],
)
def test_public_native_group32_route(
    monkeypatch, group_n, libtype, split_k, scale_dtypes
):
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
    xs = torch.empty((3, 2), dtype=scale_dtypes[0])
    ws = torch.empty(((65 + group_n - 1) // group_n, 2), dtype=scale_dtypes[1])
    if libtype not in (None, "triton"):
        with pytest.raises(AssertionError, match="Unsupported libtype"):
            gemm_op_a8w8.gemm_a8w8_blockscale(
                x, w, xs, ws, dtype=torch.float32, split_k=split_k
            )
        assert calls == ["lookup"]
        return
    actual = gemm_op_a8w8.gemm_a8w8_blockscale(
        x, w, xs, ws, dtype=torch.float32, split_k=split_k
    )
    assert actual is expected
    assert calls == [
        "lookup",
        {"dtype": torch.float32, "weight_group_rows": group_n, "split_k": split_k},
    ]


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


def test_legacy_scale_format_rejects_native_split_override():
    with pytest.raises(
        AssertionError, match="split_k override requires native group32"
    ):
        gemm_op_a8w8.gemm_a8w8_blockscale(
            torch.empty((3, 128), dtype=torch.float8_e4m3fn),
            torch.empty((128, 128), dtype=torch.float8_e4m3fn),
            torch.empty((3, 1)),
            torch.empty((1, 1)),
            split_k=3,
        )


@pytest.mark.parametrize(
    "invalid", ["activation_groups", "weight_groups", "mixed_formats"]
)
def test_invalid_byte_scales_never_reach_legacy_dispatch(monkeypatch, invalid):
    def lookup(*args):
        pytest.fail("Invalid byte scales reached backend configuration lookup")

    monkeypatch.setattr(gemm_op_a8w8, "get_CKGEMM_config", lookup)
    x = torch.empty((3, 128), dtype=torch.float8_e4m3fn)
    w = torch.empty((65, 128), dtype=torch.float8_e4m3fn)
    xs = torch.empty((3, 4), dtype=torch.uint8)
    ws = torch.empty((3, 4), dtype=torch.uint8)
    if invalid == "activation_groups":
        xs = torch.empty((3, 1), dtype=torch.uint8)
    elif invalid == "weight_groups":
        ws = torch.empty((3, 1), dtype=torch.uint8)
    else:
        xs = xs.float()
    with pytest.raises(AssertionError, match="Expected E8M0 group32 scale shapes"):
        gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws)


BMM_ROW = {
    "libtype": "flydsl",
    "kernelId": "bmm",
    "kernelName": "flydsl_bmm_mxfp8_mfma_t16x32x128_w1x2_nb4_sk1",
}


def _mxscale_operands(
    m=3, n=256, k=512, x_block=32, w_rows=32, w_block=None, scale_dtype=None
):
    """CPU operands of an e8m0 GEMM; w_block defaults to x_block."""
    e8m0 = scale_dtype or torch.float8_e8m0fnu
    x = torch.empty((m, k), dtype=torch.float8_e4m3fn)
    w = torch.empty((n, k), dtype=torch.float8_e4m3fn)
    xs = torch.empty((m, k // x_block), dtype=e8m0)
    ws = torch.empty((n // w_rows, k // (w_block or x_block)), dtype=e8m0)
    return x, w, xs, ws


MXFP8_ROW = {"libtype": "flydsl", "kernelId": "mxfp8", "kernelName": "unused"}


@pytest.mark.parametrize("row", [None, BMM_ROW, MXFP8_ROW])
@pytest.mark.parametrize(
    "block, w_rows, w_block",
    [
        (32, 32, 32),
        (32, 1, 32),
        (128, 128, 128),
        (128, 1, 32),
        (128, 32, 32),
        (32, 128, 128),
    ],
)
def test_mxscale_bpreshuffle_route(monkeypatch, row, block, w_rows, w_block):
    """B = 1 bmm runs every pairing; a tuned row only with the x_scale block it
    was tuned with (1x128 with 128x128, else 1x32), else the heuristic."""
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm

    calls = []

    def lookup(m, n, k, w_scale_block):
        calls.append(("lookup", m, n, k, w_scale_block))
        return row

    def run(x, w, xs, ws, out, kernel_name=None, x_scale_transposed=False):
        calls.append(("bmm", x.shape, w.shape, xs.shape, ws.shape, out.shape))
        calls.append((kernel_name, x_scale_transposed))
        return out

    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(bmm, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lookup)
    monkeypatch.setattr(bmm, "run_bmm_a8w8_mxfp8", run)
    x, w, xs, ws = _mxscale_operands(x_block=block, w_rows=w_rows, w_block=w_block)
    paired = block == (128 if w_rows == 128 else 32)
    if row is MXFP8_ROW and paired:
        pytest.skip("runs the mxfp8 kernels (test_mxscale_bpreshuffle_mxfp8_rows)")
    y = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws)
    assert y.shape == (3, 256) and y.dtype == torch.bfloat16
    tuned = row["kernelName"] if row is BMM_ROW and paired else None
    # Only a paired x_scale reads the table.
    lookups = [("lookup", 3, 256, 512, f"{w_rows}x{w_block}")] if paired else []
    assert calls == lookups + [
        (
            "bmm",
            (3, 1, 512),
            (1, 256, 512),
            (3, 1, 512 // block),
            (1, 256 // w_rows, 512 // w_block),
            (3, 1, 256),
        ),
        (tuned, block == 128),
    ]


def test_mxscale_bpreshuffle_rejects_unknown_implementation(monkeypatch):
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm

    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(bmm, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        gemm_op_a8w8,
        "get_mxscale_bpreshuffle_config",
        lambda *args: {**BMM_ROW, "kernelId": "other"},
    )
    with pytest.raises(NotImplementedError, match="flydsl/other"):
        gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(*_mxscale_operands())


def test_mxscale_bpreshuffle_rejects_wide_row_scales(monkeypatch):
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm

    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(bmm, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: None)
    with pytest.raises(NotImplementedError, match="a 1x128 w_scale"):
        gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(
            *_mxscale_operands(x_block=128, w_rows=1, w_block=128)
        )


@pytest.mark.parametrize("w_rows", [32, 1])
def test_gfx1250_group32_bpreshuffle_route(monkeypatch, w_rows):
    """gfx1250 group32: a 32x32 (raw) or 1x32 (n32k4) w_scale goes to the
    flydsl batched GEMM at B = 1; flydsl, not gemm_op_a8w8, picks the arch's
    kernel, and the tuned table is still consulted."""
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm

    calls = []

    def lookup(m, n, k, w_scale_block):
        calls.append(("lookup", w_scale_block))

    def run_bmm(x, w, xs, ws, out, kernel_name=None, x_scale_transposed=False):
        calls.append(("bmm", ws.shape, x_scale_transposed))
        return out

    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(bmm, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lookup)
    monkeypatch.setattr(bmm, "run_bmm_a8w8_mxfp8", run_bmm)
    y = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(*_mxscale_operands(w_rows=w_rows))
    assert y.shape == (3, 256)
    block = f"{w_rows}x32"
    assert calls == [("lookup", block), ("bmm", (1, 256 // w_rows, 16), False)]


@pytest.mark.parametrize("w_rows", [1, 32])
@pytest.mark.parametrize("m", [1, 3])
def test_mxscale_bpreshuffle_32_wide_w_reaches_blockscale_x_kernel(
    monkeypatch, m, w_rows
):
    """1x32 / 32x32 w_scales with the column-major 1x128 x_scale run on
    gfx950's kernel, which the heuristic has a config for."""
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm
    from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as gfx950

    calls = []

    def run(x, w, xs, ws, out, kernel_name=None, x_scale_transposed=False):
        calls.append((xs.shape, ws.shape, kernel_name, x_scale_transposed))
        return out

    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(bmm, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(bmm, "_runner", functools.cache(bmm._runner.__wrapped__))
    monkeypatch.setattr(gfx950, "run_bmm_a8w8_mxfp8_gfx950", run)
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: None)
    x, w, xs, ws = _mxscale_operands(m=m, x_block=128, w_rows=w_rows, w_block=32)
    gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws)
    assert calls == [((m, 1, 4), (1, 256 // w_rows, 16), None, True)]
    assert gfx950.pick_bmm_kernel_name(1, m, 256, 512, 128, w_rows, 32, m > 1)


@pytest.mark.parametrize("k, backend", [(512, "opus"), (576, "flydsl")])
def test_batched_untuned_1x32_runs_flydsl_where_opus_cannot(monkeypatch, k, backend):
    """Off the table, 1x32 runs OPUS's heuristic kid, or flydsl's when no
    OPUS kid runs the shape (K not a multiple of 128)."""
    from aiter.ops import batched_gemm_op_a8w8 as batched
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm
    from aiter.ops.opus import _arch, policy
    from aiter.ops.opus import gemm_op_a8w8 as opus

    calls = []

    def flydsl(x, w, xs, ws, out, kernel_name=None):
        calls.append(("flydsl", kernel_name))
        return out

    def opus_bmm(x, w, xs, ws, out, dtype, kernelId, splitK, group_size, group_n):
        calls.append(("opus", kernelId))
        assert (group_n, group_size) == (1, 32) and kernelId is not None

    heuristic = policy._heuristic_mxscale_bmm_bpreshuffle_kid

    def counted_heuristic(*args, **kwargs):
        calls.append(("heuristic",))
        return heuristic(*args, **kwargs)

    for module in (batched, bmm, policy):
        monkeypatch.setattr(module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(_arch, "_device_arch_and_cu", lambda _: ("gfx950", 256))
    monkeypatch.setattr(batched, "lookup_mxscale_bmm_config", lambda *a, **kw: None)
    monkeypatch.setattr(bmm, "run_bmm_a8w8_mxfp8", flydsl)
    monkeypatch.setattr(opus, "bmm_a8w8_mxscale_opus_bpreshuffle", opus_bmm)
    monkeypatch.setattr(
        policy, "_heuristic_mxscale_bmm_bpreshuffle_kid", counted_heuristic
    )
    batched._opus_bpreshuffle_kid.cache_clear()
    g, m, n = 2, 40, 256
    operands = (
        torch.empty((m, g, k), dtype=torch.float8_e4m3fn),
        torch.empty((g, n, k), dtype=torch.float8_e4m3fn),
        torch.empty((m, g, k // 32), dtype=torch.uint8),
        torch.empty((g, n, k // 32), dtype=torch.uint8),
    )
    try:
        for _ in range(2):
            batched._batched_gemm_a8w8_mxscale_bpreshuffle_impl(*operands)
        # The OPUS kid search runs once per shape.
        assert [c[0] for c in calls] == ["heuristic", backend, backend]
        if backend == "flydsl":
            with pytest.raises(NotImplementedError, match="flydsl writes bf16 only"):
                batched._batched_gemm_a8w8_mxscale_bpreshuffle_impl(
                    *operands, dtype=torch.float16
                )
    finally:
        batched._opus_bpreshuffle_kid.cache_clear()


@pytest.mark.parametrize("block", ["1x32", "32x32"])
@pytest.mark.parametrize(
    "k, dtype, backend",
    [
        (512, torch.float32, "opus"),
        (576, torch.bfloat16, "flydsl"),
        (576, torch.float32, None),
        (576, torch.float16, None),
    ],
)
def test_batched_tuned_flydsl_row_non_bf16_out(monkeypatch, block, k, dtype, backend):
    """gfx950's flydsl writes bf16 only: a tuned flydsl row with another
    output runs OPUS's kid where one runs the shape, else raises."""
    from aiter.ops import batched_gemm_op_a8w8 as batched
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm
    from aiter.ops.opus import _arch, policy
    from aiter.ops.opus import gemm_op_a8w8 as opus

    calls = []
    row = {**BMM_ROW, "kernelId": 0, "splitK": 1}

    def flydsl(x, w, xs, ws, out, kernel_name=None):
        calls.append(("flydsl", kernel_name))
        return out

    def opus_bmm(x, w, xs, ws, out, dtype, kernelId, splitK, group_size, group_n):
        calls.append(("opus", kernelId, splitK))

    for module in (batched, bmm, policy):
        monkeypatch.setattr(module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(_arch, "_device_arch_and_cu", lambda _: ("gfx950", 256))
    monkeypatch.setattr(batched, "lookup_mxscale_bmm_config", lambda *a, **kw: row)
    monkeypatch.setattr(bmm, "run_bmm_a8w8_mxfp8", flydsl)
    monkeypatch.setattr(opus, "bmm_a8w8_mxscale_opus_bpreshuffle", opus_bmm)
    g, m, n = 2, 40, 256
    rows = 1 if block == "1x32" else 32
    operands = (
        torch.empty((m, g, k), dtype=torch.float8_e4m3fn),
        torch.empty((g, n, k), dtype=torch.float8_e4m3fn),
        torch.empty((m, g, k // 32), dtype=torch.uint8),
        torch.empty((g, n // rows, k // 32), dtype=torch.uint8),
    )
    batched._opus_bpreshuffle_kid.cache_clear()
    try:
        if backend is None:
            with pytest.raises(NotImplementedError, match="flydsl writes bf16 only"):
                batched._batched_gemm_a8w8_mxscale_bpreshuffle_impl(
                    *operands, dtype=dtype
                )
            return
        batched._batched_gemm_a8w8_mxscale_bpreshuffle_impl(*operands, dtype=dtype)
    finally:
        batched._opus_bpreshuffle_kid.cache_clear()
    if backend == "flydsl":
        assert calls == [("flydsl", BMM_ROW["kernelName"])]
    else:
        # The heuristic's kid, searched without the flydsl row's splitK.
        assert len(calls) == 1 and calls[0][0] == "opus"
        assert calls[0][1] is not None and calls[0][2] is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx950",
    reason="the preshuffled e8m0 batched GEMM's gfx950 route",
)
@pytest.mark.parametrize("k, backend", [(512, "opus"), (576, "flydsl")])
@pytest.mark.parametrize("m", [1, 40])
def test_batched_untuned_1x32_matches_dequant(monkeypatch, m, k, backend):
    """The public batched entry's untuned 1x32 route, OPUS's kid or flydsl's
    heuristic kernel where no kid runs the shape, on a 1x32 w_scale whose
    bytes differ row to row."""
    from aiter.ops import batched_gemm_op_a8w8 as batched
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm
    from aiter.ops.opus import gemm_op_a8w8 as opus
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import dtypes

    ran = []

    def spy(name, fn):
        def run(*args, **kwargs):
            ran.append(name)
            return fn(*args, **kwargs)

        return run

    monkeypatch.setattr(batched, "lookup_mxscale_bmm_config", lambda *a, **kw: None)
    monkeypatch.setattr(
        bmm, "run_bmm_a8w8_mxfp8", spy("flydsl", bmm.run_bmm_a8w8_mxfp8)
    )
    monkeypatch.setattr(
        opus,
        "bmm_a8w8_mxscale_opus_bpreshuffle",
        spy("opus", opus.bmm_a8w8_mxscale_opus_bpreshuffle),
    )
    g, n = 2, 512
    x = (torch.randn(m, g, k, device="cuda") * 2).to(dtypes.fp8)
    w = (torch.randn(g, n, k, device="cuda") * 2).to(dtypes.fp8)
    xs = torch.randint(118, 136, (m, g, k // 32), dtype=torch.uint8, device="cuda")
    ws = torch.randint(118, 136, (g, n, k // 32), dtype=torch.uint8, device="cuda")
    e8 = lambda s: torch.exp2(s.float() - 127).repeat_interleave(32, -1)
    ref = torch.einsum("mgk,gnk->mgn", x.float() * e8(xs), w.float() * e8(ws))
    out = batched.batched_gemm_a8w8_mxscale_bpreshuffle(
        x,
        shuffle_weight(w, layout=(16, 16)),
        xs.view(dtypes.fp8_e8m0),
        ws.view(dtypes.fp8_e8m0),
    )
    assert ran == [backend]
    _assert_matches(out.float(), ref)


def test_mxscale_bpreshuffle_table_reads_blank_and_merged_cells(monkeypatch, tmp_path):
    """A blank w_scale_block cell, or the 0 a config merge fills a missing
    column with, reads as 128x128, in AOT and at runtime; a blank row that
    repeats a 128x128 row's shape loads, and the faster of the two is used."""
    from aiter.aot.flydsl import gemm as aot
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import make_mxfp8_8wave_kernel_name

    mxfp8 = make_mxfp8_8wave_kernel_name(2, 0)
    csv_path = tmp_path / "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
    lines = [
        "gfx,cu_num,M,N,K,w_scale_block,libtype,kernelId,us,kernelName",
        f"gfx950,256,1,256,512,0,flydsl,mxfp8,5.0,{mxfp8}",
        f"gfx950,256,1,512,512,,flydsl,mxfp8,5.0,{mxfp8}",
        f"gfx950,256,1,512,512,128x128,flydsl,bmm,4.0,{BMM_ROW['kernelName']}",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    jobs = aot.parse_csv(str(csv_path))
    blocks = [job["w_scale_block"] for job in jobs if job["kind"] == "mxfp8_8wave"]
    assert blocks == ["128x128", "128x128"]

    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a8w8, "get_cu_num", lambda: 256)
    monkeypatch.setenv(
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE", str(csv_path)
    )
    caches = (
        AITER_CONFIGS.get_config_file,
        gemm_op_a8w8._load_mxscale_bpreshuffle_tuned,
        gemm_op_a8w8.get_mxscale_bpreshuffle_config,
    )
    for cache in caches:
        cache.cache_clear()
    try:
        for n, kernel_id in ((256, "mxfp8"), (512, "bmm")):
            config = gemm_op_a8w8.get_mxscale_bpreshuffle_config(1, n, 512, "128x128")
            assert config["kernelId"] == kernel_id
    finally:
        for cache in caches:
            cache.cache_clear()


def test_blockscale_preshuffled_group32_forwards_e8m0_views(monkeypatch):
    calls = []
    forwarded = torch.full((3, 256), 7.0, dtype=torch.bfloat16)

    def bpreshuffle(x, w, xs, ws, dtype):
        calls.append((xs.dtype, ws.dtype, dtype))
        return forwarded

    monkeypatch.setattr(gemm_op_a8w8, "gemm_a8w8_blockscale_bpreshuffle", bpreshuffle)
    out = gemm_op_a8w8.gemm_a8w8_blockscale(
        *_mxscale_operands(scale_dtype=torch.uint8),
        dtype=torch.bfloat16,
        isBpreshuffled=True,
    )
    e8m0 = torch.float8_e8m0fnu
    assert torch.equal(out, forwarded) and calls == [(e8m0, e8m0, torch.bfloat16)]


def _mxscale_gemm_vs_dequant(m, n, k, block, w_rows=None, x_block=None):
    """The public e8m0 GEMM on random operands, and its dequantized FP32
    reference. w_scale blocks are w_rows x block (default square), x_scale
    blocks 1 x x_block (default block): a group32 x_scale is row-major, a
    128-wide blockscale one column-major bytes."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import dtypes

    w_rows = w_rows or block
    x_block = x_block or block
    x = (torch.randn(m, k, device="cuda") * 2).to(dtypes.fp8)
    w = (torch.randn(n, k, device="cuda") * 2).to(dtypes.fp8)
    xs = torch.randint(118, 136, (m, k // x_block), dtype=torch.uint8, device="cuda")
    ws = torch.randint(
        118, 136, (n // w_rows, k // block), dtype=torch.uint8, device="cuda"
    )
    e8 = lambda s: torch.exp2(s.float() - 127)
    ref = (x.float() * e8(xs).repeat_interleave(x_block, 1)) @ (
        w.float() * e8(ws).repeat_interleave(w_rows, 0).repeat_interleave(block, 1)
    ).T
    xs_arg = xs if x_block == 32 else xs.t().contiguous().view(-1).view(m, k // 128)
    out = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(
        x,
        shuffle_weight(w, layout=(16, 16)),
        xs_arg.view(dtypes.fp8_e8m0),
        ws.view(dtypes.fp8_e8m0),
    )
    return out.float(), ref


def _assert_matches(out, ref):
    """Elementwise, so one wrong scale byte fails however few outputs it
    touches: BF16 rounding stays well inside rtol."""
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item())


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx950",
    reason="the preshuffled e8m0 GEMM runs on gfx950",
)
@pytest.mark.parametrize(
    "m,n,k,block",
    [
        (1, 1152, 5120, 32),
        (5, 5120, 576, 32),
        (64, 512, 5120, 32),
        (1, 6144, 7168, 128),
        (5, 6144, 7168, 128),
        (300, 1536, 1024, 128),
        # 128-wide scale rows ending mid dword (DeepSeek-V4 shared-expert down
        # at TP4 / TP8); M = 37 leaves a partial column-major x_scale dword.
        (1, 7168, 768, 128),
        (37, 7168, 768, 128),
        (1, 7168, 384, 128),
        (300, 7168, 384, 128),
    ],
)
@pytest.mark.parametrize("row", [None, BMM_ROW])
def test_mxscale_bpreshuffle_matches_dequant(monkeypatch, m, n, k, block, row):
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: row)
    _assert_matches(*_mxscale_gemm_vs_dequant(m, n, k, block))


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx950",
    reason="the preshuffled e8m0 GEMM runs on gfx950",
)
@pytest.mark.parametrize(
    "config",
    [
        "t16x32x128_w1x2_nb2_sk1",  # scales preloaded
        "t16x32x128_w1x1_nb3_sk1_sps",  # a byte of scales per row per stage
        "t32x64x128_w1x4_nb2_sk3",  # split K
        "t64x64x128_w2x2_nb3_sk1_bd2",  # B straight to registers
    ],
)
@pytest.mark.parametrize("k", [384, 768])
@pytest.mark.parametrize("m", [1, 37])
def test_mxscale_bpreshuffle_mid_dword_scale_rows(monkeypatch, m, k, config):
    """128-wide scale rows of 3 or 6 bytes start and end mid dword; every
    kernel layout must read the last row's bytes, not the zeroed straddle."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name
    from aiter.ops.flydsl.kernels.bmm_a8w8_mxscale_gfx950 import check_bmm_config

    name = f"flydsl_bmm_mxfp8_mfma_{config}"
    blocks = dict.fromkeys(("x_scale_k", "w_scale_n", "w_scale_k"), 128)
    check_bmm_config(
        7168, k, 1, **parse_bmm_kernel_name(name), **blocks, x_scale_transposed=m > 1
    )
    row = {**BMM_ROW, "kernelName": name}
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: row)
    _assert_matches(*_mxscale_gemm_vs_dequant(m, 7168, k, 128))


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx950",
    reason="the preshuffled e8m0 GEMM runs on gfx950",
)
@pytest.mark.parametrize(
    "m,n,k,block",
    [
        (1, 1152, 5120, 32),
        (37, 5120, 1536, 32),
        (300, 512, 5120, 32),
        (1, 6144, 7168, 128),
        (37, 7168, 768, 128),
        (300, 1536, 1024, 128),
        (300, 7168, 384, 128),
    ],
)
@pytest.mark.parametrize("kernel", ["mxpsh", "8wave"])
def test_mxscale_bpreshuffle_mxfp8_rows(monkeypatch, m, n, k, block, kernel):
    """A kernelId "mxfp8" row runs gfx950's mxpsh / 8-wave GEMM on the raw
    scales bmm reads (32x32, or 128x128 with the column-major x_scale)."""
    from aiter.ops.flydsl import batched_gemm_a8w8
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import make_mxfp8_8wave_kernel_name
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        pick_mxfp8_kernel_name_gfx950,
    )

    if kernel == "8wave":
        if k % 256 or (block == 128 and k < 512):
            pytest.skip("the 8-wave kernel needs K % 256 == 0 (and K >= 512 at 128)")
        name = make_mxfp8_8wave_kernel_name(2, 0)
    else:
        name = pick_mxfp8_kernel_name_gfx950(m, n, k, f"{block}x{block}")
    row = {"libtype": "flydsl", "kernelId": "mxfp8", "kernelName": name}
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: row)

    def no_bmm(*a, **kw):
        raise AssertionError("an mxfp8 row must not run bmm")

    monkeypatch.setattr(batched_gemm_a8w8, "run_bmm_a8w8_mxfp8", no_bmm)
    _assert_matches(*_mxscale_gemm_vs_dequant(m, n, k, block))


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx1250",
    reason="gfx1250's group32 route",
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        # DeepSeek-V4.1-Flash FP8 linears per rank at TP4 (32x32 weight scale).
        (1, 1792, 5120),  # wqkv_a
        (7, 8192, 1280),  # wq_b
        (64, 4096, 1280),  # indexer wq_b
        (512, 5120, 2048),  # wo_b
        (4096, 1152, 5120),  # shared expert w13
        # Shared expert w2: K = 576 is not a multiple of 128, a ragged k-tile.
        (1, 5120, 576),
        (37, 5120, 576),
        (4096, 5120, 576),
    ],
)
def test_gfx1250_group32_matches_dequant(m, n, k):
    """gemm_a8w8_blockscale_bpreshuffle's gfx1250 group32 route: the 32x32
    w_scale is read as stored by the batched mxfp8_32 GEMM at B = 1."""
    _assert_matches(*_mxscale_gemm_vs_dequant(m, n, k, 32))


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx1250",
    reason="gfx1250's batched mxfp8_32 GEMM",
)
def test_gfx1250_ragged_k_refuses_n32k4_scale():
    """A ragged K runs with the row-major 32x32 w_scale only: an n32k4 scale
    does not keep its leading K bytes contiguous, so it must be refused."""
    from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
    from aiter.utility import dtypes

    m, b, n, k = 7, 1, 512, 576
    x = torch.zeros(m, b, k, device="cuda").to(dtypes.fp8)
    w = torch.zeros(b, n, k, device="cuda").to(dtypes.fp8)
    xs = torch.full((m, b, k // 32), 127, dtype=torch.uint8, device="cuda")
    ws = torch.full((b, n, k // 32), 127, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="ragged"):
        batched_gemm_a8w8_mxscale_bpreshuffle(
            x, w, xs.view(dtypes.fp8_e8m0), ws.view(dtypes.fp8_e8m0)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx950",
    reason="the preshuffled e8m0 GEMM runs on gfx950",
)
@pytest.mark.parametrize(
    "m,n,k,config",
    [
        (300, 1536, 1024, "t256x192x128_w1x4_nb3_sk1_bd2_xcd_sps"),
        (5, 1536, 6144, "t16x32x256_w1x2_nb4_sk4"),
        (1024, 1024, 1024, "t64x128x128_w1x4_nb4_sk1_bd2_xcd4"),
        # 5 M tiles leave a 1-tile swizzle group; 12 N tiles pad the grid.
        (300, 1536, 1024, "t64x128x128_w1x4_nb4_sk1_xcd4"),
    ],
)
def test_mxscale_bpreshuffle_1x32_weight_scales(monkeypatch, m, n, k, config):
    """Each weight row has its own scale, so every kernel layout must read
    the scale of the row it computes."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name
    from aiter.ops.flydsl.kernels.bmm_a8w8_mxscale_gfx950 import check_bmm_config

    name = f"flydsl_bmm_mxfp8_mfma_{config}"
    check_bmm_config(
        n, k, 1, **parse_bmm_kernel_name(name), x_scale_k=32, w_scale_n=1, w_scale_k=32
    )
    row = {**BMM_ROW, "kernelName": name}
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: row)
    _assert_matches(*_mxscale_gemm_vs_dequant(m, n, k, 32, w_rows=1))


@pytest.mark.skipif(
    not torch.cuda.is_available() or gemm_op_a8w8.get_gfx() != "gfx950",
    reason="the preshuffled e8m0 GEMM runs on gfx950",
)
@pytest.mark.parametrize("m,n,k", [(1, 1536, 1024), (37, 5120, 1536), (300, 512, 5120)])
@pytest.mark.parametrize("w_rows", [1, 32])
def test_mxscale_bpreshuffle_32_wide_w_blockscale_x(m, n, k, w_rows):
    """1x32 / 32x32 w_scales with the column-major 1x128 blockscale x_scale,
    which no tuned row pairs with, so dispatch runs the heuristic kernel."""
    _assert_matches(*_mxscale_gemm_vs_dequant(m, n, k, 32, w_rows=w_rows, x_block=128))


def _bmm_legal(n, k, block, **cfg):
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import bmm_config_runs

    base = {
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 128,
        "m_warp": 1,
        "n_warp": 4,
        "num_buffers": 3,
    }
    blocks = dict.fromkeys(("x_scale_k", "w_scale_n", "w_scale_k"), block)
    return bmm_config_runs(1, n, k, blocks, {**base, **cfg})


def test_bmm_config_rejects_partial_wave_tiles():
    """A wave's share of the tile must be whole 16-wide MFMA tiles: 112 over
    four N waves is 28 columns, of which the kernel would compute 16."""
    assert _bmm_legal(7168, 4096, 32, tile_n=128, n_warp=4)
    assert not _bmm_legal(7168, 4096, 32, tile_n=112, n_warp=4)
    assert not _bmm_legal(7168, 4096, 32, tile_m=96, m_warp=4, n_warp=1)


def test_bmm_config_rejects_tiles_straddling_w_scale_blocks():
    """A tile reads whole w_scale blocks or fits in one: 96 rows straddle a
    128-row block, while 32-row blocks tile them."""
    assert not _bmm_legal(6144, 4096, 128, tile_n=96, n_warp=2)
    assert _bmm_legal(6144, 4096, 32, tile_n=96, n_warp=2)
    assert _bmm_legal(6144, 4096, 128, tile_n=64, n_warp=2)


def test_bmm_xcd_swizzle_names():
    """_xcd<g> names round-trip; only _xcd, or _xcd<g> with g >= 2 and no
    split K, runs."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        BMM_MFMA_NAME_PREFIX,
        bmm_kernel_name,
        parse_bmm_kernel_name,
    )

    base = f"{BMM_MFMA_NAME_PREFIX}_t64x128x128_w1x4_nb4_sk1_bd2_xcd"
    assert parse_bmm_kernel_name(base)["xcd_swizzle"] == 0
    assert parse_bmm_kernel_name(f"{base}12")["xcd_swizzle"] == 12
    for swizzle in ("0", "04", "1"):
        assert parse_bmm_kernel_name(base + swizzle) is None
    # Split K keeps its own tile order: the plain _xcd parses, a swizzle not.
    assert parse_bmm_kernel_name(base.replace("_sk1", "_sk2"))["splits"] == 2
    assert parse_bmm_kernel_name(base.replace("_sk1", "_sk2") + "4") is None
    cfg = parse_bmm_kernel_name(f"{base}4")
    assert bmm_kernel_name(**cfg) == f"{base}4"
    swizzled = {"xcd_order": True, "xcd_swizzle": 4}
    assert _bmm_legal(1536, 4096, 32, **swizzled)
    for bad in ({"xcd_order": False}, {"xcd_swizzle": 1}, {"splits": 2}):
        assert not _bmm_legal(1536, 4096, 32, **{**swizzled, **bad})
    # The tuned DSv4 rows name the plain order with split K.
    assert _bmm_legal(1536, 4096, 32, splits=2, xcd_order=True)


def test_bmm_candidates_split_swizzled_seeds():
    """A swizzled seed's split-K neighbours drop the XCD order and swizzle
    rather than failing the config check (or timing the same kernel twice)."""
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        BMM_MFMA_NAME_PREFIX,
        bmm_candidate_names,
        parse_bmm_kernel_name,
    )

    seed = f"{BMM_MFMA_NAME_PREFIX}_t64x128x128_w1x4_nb4_sk1_bd2_xcd4"
    scales = {"x_scale_k": 32, "w_scale_n": 1, "w_scale_k": 32}
    names = bmm_candidate_names(1, 4096, 6144, 6144, [seed], **scales)
    split = f"{BMM_MFMA_NAME_PREFIX}_t64x128x128_w1x4_nb4_sk2_bd2"
    assert seed in names and split in names
    for cfg in map(parse_bmm_kernel_name, names):
        assert cfg["splits"] == 1 or not (cfg["xcd_order"] or cfg["xcd_swizzle"])


@pytest.mark.parametrize("block", ["1x32", "32x32", "128x128"])
def test_mxscale_table_tuner_times_bmm(block):
    """--mxscale_table times bmm on every w_scale block, 1x32's included; a
    128x128 candidate also runs the row-major x_scale M = 1 reads."""
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "csrc",
        "gemm_a8w8_mxfp8_bpreshuffle",
        "gemm_a8w8_mxfp8_bpreshuffle_tune.py",
    )
    spec = importlib.util.spec_from_file_location("mxfp8_tune", path)
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name
    from aiter.ops.flydsl.kernels.bmm_a8w8_mxscale_gfx950 import check_bmm_config

    names = tune.bmm_candidates(1, 4608, 6144, block)
    assert names
    rows, cols = (int(v) for v in block.split("x"))
    for cfg in map(parse_bmm_kernel_name, names):
        check_bmm_config(
            4608, 6144, 1, **cfg, x_scale_k=cols, w_scale_n=rows, w_scale_k=cols
        )


@pytest.mark.parametrize("w_scale_n", [1, 32, 128])
def test_bmm_heuristic_swizzles_1x32_only(w_scale_n):
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        parse_bmm_kernel_name,
        pick_bmm_kernel_name,
    )

    block = 32 if w_scale_n == 1 else w_scale_n
    name = pick_bmm_kernel_name(1, 4096, 6144, 6144, block, w_scale_n, block)
    cfg = parse_bmm_kernel_name(name)
    assert cfg["xcd_order"] and cfg["splits"] == 1
    assert cfg["xcd_swizzle"] == (4 if w_scale_n == 1 else 0)


def test_bmm_rejects_wide_1_row_w_scale(monkeypatch):
    from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as bmm

    monkeypatch.setattr(bmm, "get_gfx", lambda: "gfx950")
    x, w, xs, ws = _mxscale_operands(x_block=128, w_rows=1)
    m, k = x.shape
    with pytest.raises(ValueError, match="1x128 / 1x128 scale blocks"):
        bmm.run_bmm_a8w8_mxfp8_gfx950(
            x.view(m, 1, k),
            w.view(1, *w.shape),
            xs.view(m, 1, -1),
            ws.view(1, *ws.shape),
            torch.empty((m, 1, w.shape[0]), dtype=torch.bfloat16),
        )


def test_bmm_unrecognised_tuned_name_warns_and_runs_heuristic(
    monkeypatch, tmp_path, capsys
):
    """A tuned kernelName that breaks a naming rule is skipped by AOT and runs
    the heuristic's pick at runtime, with a warning on both sides."""
    from aiter.aot.flydsl import gemm as aot
    from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as gfx950

    bad = "flydsl_bmm_mxfp8_mfma_t16x32x128_w1x2_nb4_sk2_xcd4"
    assert gfx950.parse_bmm_kernel_name(bad) is None
    csv_path = tmp_path / "batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv"
    csv_path.write_text(
        "gfx,b,m,n,k,w_scale_block,libtype,kernelId,kernelName\n"
        f"gfx950,1,3,256,512,1x32,flydsl,0,{bad}\n"
    )
    assert aot.parse_csv(str(csv_path)) == []
    assert f"Unknown FlyDSL BMM kernel name: {bad}" in capsys.readouterr().out

    launched, warned = [], []
    monkeypatch.setattr(gfx950, "_launch", lambda t, m, s, key: launched.append(key))
    monkeypatch.setattr(gfx950.torch.cuda, "current_stream", lambda *a: None)
    monkeypatch.setattr(gfx950, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gfx950.logger, "warning", warned.append)
    gfx950._warn_unrecognised.cache_clear()
    x, w, xs, ws = _mxscale_operands(w_rows=1)
    m, k = x.shape
    operands = (x.view(m, 1, k), w.view(1, *w.shape), xs.view(m, 1, -1))
    operands += (ws.view(1, *ws.shape),)
    for name in (bad, None):
        out = torch.empty((m, 1, w.shape[0]), dtype=torch.bfloat16)
        gfx950.run_bmm_a8w8_mxfp8_gfx950(*operands, out, kernel_name=name)
    assert launched[0] == launched[1]
    assert len(warned) == 1 and bad in warned[0]


def test_aot_bmm_rows_cover_their_dispatch(monkeypatch, tmp_path):
    """AOT compiles the launcher every tuned GEMM bmm row dispatches, at M = 1
    and M > 1; both read a blank w_scale_block cell (left by merging a table
    from before the column) as 128x128."""
    from aiter.aot.flydsl import gemm as aot
    from aiter.ops.flydsl import batched_gemm_a8w8 as bmm
    from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as gfx950

    launched = []
    monkeypatch.setattr(gfx950, "_launch", lambda t, m, s, key: launched.append(key))
    monkeypatch.setattr(gfx950.torch.cuda, "current_stream", lambda *a: None)
    for module in (gemm_op_a8w8, bmm, gfx950):
        monkeypatch.setattr(module, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a8w8, "get_cu_num", lambda: 256)
    monkeypatch.setattr(bmm, "_runner", functools.cache(bmm._runner.__wrapped__))
    k = 512
    # (w_scale_block cell, N, the scale block it reads as)
    rows = [("1x32", 256, (1, 32)), ("32x32", 256, (32, 32))]
    rows += [("128x128", 256, (128, 128)), ("", 512, (128, 128))]
    csv_path = tmp_path / "a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
    lines = ["gfx,cu_num,M,N,K,w_scale_block,libtype,kernelId,kernelName"]
    lines += [
        f"gfx950,256,{m},{n},{k},{cell},flydsl,bmm,{BMM_ROW['kernelName']}"
        for cell, n, _ in rows
        for m in (1, 37)
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    jobs = aot.parse_csv(str(csv_path))
    assert {job["w_scale_block"] for job in jobs} == {"1x32", "32x32", "128x128"}
    for job in jobs:
        aot._compile_bmm_mfma_to_cache(**job)
    compiled, launched[:] = set(launched), []

    monkeypatch.setenv(
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE", str(csv_path)
    )
    caches = (
        AITER_CONFIGS.get_config_file,
        gemm_op_a8w8._load_mxscale_bpreshuffle_tuned,
        gemm_op_a8w8.get_mxscale_bpreshuffle_config,
    )
    for cache in caches:
        cache.cache_clear()
    try:
        for _, n, (w_rows, w_block) in rows:
            for m in (1, 37):
                block = f"{w_rows}x{w_block}"
                assert gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, n, k, block)
                x, w, xs, ws = _mxscale_operands(
                    m=m, n=n, k=k, x_block=w_block, w_rows=w_rows, w_block=w_block
                )
                gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws)
    finally:
        for cache in caches:
            cache.cache_clear()
    assert len(launched) == 8 and set(launched) <= compiled
