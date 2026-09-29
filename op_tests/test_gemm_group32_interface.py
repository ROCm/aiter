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


def _mxscale_operands(m=3, n=256, k=512, x_block=32, w_rows=32, scale_dtype=None):
    """CPU operands of an e8m0 GEMM."""
    e8m0 = scale_dtype or torch.float8_e8m0fnu
    x = torch.empty((m, k), dtype=torch.float8_e4m3fn)
    w = torch.empty((n, k), dtype=torch.float8_e4m3fn)
    xs = torch.empty((m, k // x_block), dtype=e8m0)
    ws = torch.empty((n // w_rows, k // x_block), dtype=e8m0)
    return x, w, xs, ws


@pytest.mark.parametrize("row", [None, BMM_ROW])
@pytest.mark.parametrize("block", [32, 128])
def test_mxscale_bpreshuffle_route(monkeypatch, row, block):
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
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lookup)
    monkeypatch.setattr(bmm, "run_bmm_a8w8_mxfp8", run)
    x, w, xs, ws = _mxscale_operands(x_block=block, w_rows=block)
    y = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws)
    kb = 512 // block
    assert y.shape == (3, 256) and y.dtype == torch.bfloat16
    assert calls == [
        ("lookup", 3, 256, 512, f"{block}x{block}"),
        (
            "bmm",
            (3, 1, 512),
            (1, 256, 512),
            (3, 1, kb),
            (1, 256 // block, kb),
            (3, 1, 256),
        ),
        (None if row is None else row["kernelName"], block == 128),
    ]


def test_mxscale_bpreshuffle_rejects_unknown_implementation(monkeypatch):
    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        gemm_op_a8w8,
        "get_mxscale_bpreshuffle_config",
        lambda *args: {**BMM_ROW, "kernelId": "other"},
    )
    with pytest.raises(NotImplementedError, match="flydsl/other"):
        gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(*_mxscale_operands())


def test_mxscale_bpreshuffle_rejects_row_scales(monkeypatch):
    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a8w8, "get_mxscale_bpreshuffle_config", lambda *a: None)
    with pytest.raises(NotImplementedError, match="1x32"):
        gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle(*_mxscale_operands(w_rows=1))


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


def _mxscale_gemm_vs_dequant(m, n, k, block):
    """The public e8m0 GEMM on random operands, and its dequantized FP32
    reference. A group32 x_scale is row-major, a 128-wide blockscale one
    column-major bytes."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import dtypes

    x = (torch.randn(m, k, device="cuda") * 2).to(dtypes.fp8)
    w = (torch.randn(n, k, device="cuda") * 2).to(dtypes.fp8)
    xs = torch.randint(118, 136, (m, k // block), dtype=torch.uint8, device="cuda")
    ws = torch.randint(
        118, 136, (n // block, k // block), dtype=torch.uint8, device="cuda"
    )
    e8 = lambda s: torch.exp2(s.float() - 127)
    ref = (x.float() * e8(xs).repeat_interleave(block, 1)) @ (
        w.float() * e8(ws).repeat_interleave(block, 0).repeat_interleave(block, 1)
    ).T
    xs_arg = xs if block == 32 else xs.t().contiguous().view(-1).view(m, k // block)
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


def _bmm_legal(n, k, block, **cfg):
    from aiter.ops.flydsl.kernels.bmm_a8w8_mxscale_gfx950 import check_bmm_config

    base = {
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 128,
        "m_warp": 1,
        "n_warp": 4,
        "num_buffers": 3,
    }
    blocks = dict.fromkeys(("x_scale_k", "w_scale_n", "w_scale_k"), block)
    try:
        check_bmm_config(n, k, 1, **{**base, **cfg}, **blocks)
    except ValueError:
        return False
    return True


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
