# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Specialized correctness tests for the native gfx942 (CDNA3) bf16 GEMM.

Compares the kernel against an f32 matmul oracle (bf16 inputs upcast to f32)
to within one bf16 ulp, or f32 summation-order noise for fp32 output. The
standard HGEMM harness covers representative shape dispatch; these tests cover
explicit pipelines, split-K modes, boundary stores, streams, and invalid input.

Skips unless running on real gfx942 hardware.
"""

import pytest
import torch

if __package__:
    from .gfx942_gemm_oracle import error_ratio
else:
    from gfx942_gemm_oracle import error_ratio

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 import gemm_a16w16


def _require_gfx942():
    gfx = get_gfx()
    if gfx != "gfx942":
        pytest.skip(f"requires gfx942, got {gfx!r}")


def _inputs(m, n, k):
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)  # [N, K]
    ref = x.float() @ w.float().t()  # f32 oracle
    return x, w, ref


def _assert_close(y, ref, msg):
    worst = error_ratio(y, ref)
    assert worst <= 1.0, f"{msg}: error is {worst:.2f}x the {y.dtype} tolerance"


OUT_DTYPES = [torch.bfloat16, torch.float32]
OUT_DTYPE_IDS = ["bf16", "fp32"]


def _guarded_out(m, n, dtype):
    """Output with NaN guard rows past M; the caller passes ``out[:m]``."""
    return torch.full((m + 64, n), float("nan"), dtype=dtype, device="cuda")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"k_waves": 2},
        {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "stages": 2,
            "split_k": 1,
            "m_waves": 2,
            "n_waves": 2,
            "group_m": 0,
        },
    ],
    ids=["unset", "k_waves-only", "explicit-default-tile"],
)
def test_flydsl_hgemm_gfx942_forwards_overrides(overrides, monkeypatch):
    """flydsl_hgemm passes exactly the tile arguments it was given to the kernel."""
    _require_gfx942()
    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    compiled = {}
    make_param = g.make_gemm_a16w16_gfx942_param

    def recording_make_param(**kwargs):
        compiled.update(kwargs)
        return make_param(**kwargs)

    monkeypatch.setattr(g, "make_gemm_a16w16_gfx942_param", recording_make_param)
    m, n, k = 1024, 1024, 1024
    x, w, ref = _inputs(m, n, k)
    y = flydsl_hgemm(x, w, **overrides)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"flydsl_hgemm {overrides}")
    if overrides.keys() & g._STRUCTURAL_KEYS:
        assert {key: compiled[key] for key in overrides} == overrides
        assert not compiled.get("buffered_load", False)
    else:
        expected = {**g.gfx942_default_config(m, n, k), **overrides}
        assert {key: compiled[key] for key in expected} == expected


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_gemm_a16w16_gfx942_split_k(out_dtype):
    """Explicit async split-K: fp32 output uses native atomics; bf16 goes via f32 accum."""
    _require_gfx942()
    m, n, k = 512, 384, 10240
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(
        x,
        w.t(),
        layout="nt",
        out_dtype=out_dtype,
        user_kwargs={
            "block_m": 32,
            "block_n": 32,
            "block_k": 64,
            "stages": 2,
            "split_k": 4,
            "m_waves": 2,
            "n_waves": 2,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"split-K out={out_dtype}")


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("split_k", [2, 3, 4])
def test_gemm_a16w16_gfx942_buffered_split_k(split_k, stages):
    """The 128-bit global-to-register load path honors each split's K offset."""
    _require_gfx942()
    m, n, k = 128, 128, 1536
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(
        x,
        w.t(),
        layout="nt",
        out_dtype=torch.bfloat16,
        user_kwargs={
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "stages": stages,
            "split_k": split_k,
            "m_waves": 2,
            "n_waves": 2,
            "buffered_load": True,
            "async_load_bytes": 16,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"buffered stages={stages} split-K={split_k}")


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
@pytest.mark.parametrize("split_k", [1, 3])
@pytest.mark.parametrize("m,n", [(1000, 384), (1000, 336), (960, 344)])
def test_gemm_a16w16_gfx942_pingpong_edge_tiles(m, n, split_k, out_dtype):
    """Pingpong edge tiles: bounded buffered loads, and direct stores past M."""
    _require_gfx942()
    k = 4096
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    y = gemm_a16w16(
        x,
        w.t(),
        out=out[:m],
        layout="nt",
        user_kwargs={
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "stages": 1,
            "split_k": split_k,
            "m_waves": 2,
            "n_waves": 2,
            "buffered_load": True,
            "async_load_bytes": 16,
            "pingpong": True,
            "split_k_reduce": split_k > 1,
            "direct_store": n % 64 == 0,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"pingpong edge M{m} N{n} split_k={split_k} {out_dtype}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


@pytest.mark.parametrize(
    "m,n,k,tile",
    [
        (
            1000,
            336,
            10240,
            {"block_m": 64, "block_n": 64, "block_k": 128, "split_k": 3},
        ),
        (960, 344, 4096, {"block_m": 128, "block_n": 96, "block_k": 128, "split_k": 2}),
        (2048, 4096, 1024, {"block_m": 256, "block_n": 128, "block_k": 64}),
        (1024, 768, 128, {"block_m": 128, "block_n": 192, "block_k": 64}),
        (
            1000,
            1024,
            2048,
            {"block_m": 64, "block_n": 128, "block_k": 128, "k_waves": 2},
        ),
        (1000, 10240, 320, {"block_m": 128, "block_n": 128, "block_k": 64}),
        (
            1000,
            336,
            10240,
            {
                "block_m": 128,
                "block_n": 112,
                "block_k": 128,
                "m_waves": 2,
                "n_waves": 1,
                "k_waves": 2,
                "split_k": 3,
                "direct_store": True,
            },
        ),
        (250, 384, 10240, {"block_m": 32, "block_n": 96, "block_k": 128, "split_k": 8}),
        (
            500,
            336,
            10240,
            {
                "block_m": 32,
                "block_n": 128,
                "block_k": 128,
                "m_waves": 1,
                "n_waves": 4,
                "split_k": 6,
            },
        ),
        (
            1000,
            336,
            10240,
            {
                "block_m": 64,
                "block_n": 64,
                "block_k": 128,
                "m_waves": 2,
                "n_waves": 1,
                "k_waves": 2,
                "split_k": 3,
            },
        ),
        # C tile over 64 KB: staged through LDS in row chunks.
        (
            1000,
            10240,
            320,
            {"block_m": 256, "block_n": 160, "block_k": 64, "m_waves": 4, "n_waves": 1},
        ),
        (
            1000,
            1000,
            1024,
            {
                "block_m": 256,
                "block_n": 160,
                "block_k": 64,
                "m_waves": 4,
                "n_waves": 1,
                "split_k": 2,
            },
        ),
    ],
)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
def test_gemm_a16w16_gfx942_pingpong(m, n, k, tile, out_dtype):
    """Pingpong: edge tiles, split-K, k_waves and two-K-tile shapes."""
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    split_k = tile.get("split_k", 1)
    cfg = {
        "stages": 1,
        "m_waves": 2,
        "n_waves": 2,
        "buffered_load": True,
        "async_load_bytes": 16,
        "pingpong": True,
        "split_k_reduce": split_k > 1,
        "direct_store": False,
    }
    cfg.update(tile)
    if cfg.get("k_waves", 1) > 1 and "m_waves" not in tile:
        cfg["m_waves"] = 1
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"pingpong M{m} N{n} K{k} {tile} {out_dtype}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize(
    "m,n,k,split_k,split_k_reduce",
    [
        (768, 768, 128, 1, False),  # fewer K tiles than stages
        (768, 768, 192, 1, False),
        (1000, 336, 1024, 1, False),  # M and N edge tiles
        (500, 384, 2048, 2, False),  # atomic split-K
        (1000, 336, 4096, 3, True),  # workspace split-K
    ],
)
def test_gemm_a16w16_gfx942_async_stages(
    m, n, k, split_k, split_k_reduce, stages, out_dtype
):
    """Async DMA pipeline; stages >= 3 adds the register double buffer."""
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    cfg = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": stages,
        "split_k": split_k,
        "split_k_reduce": split_k_reduce,
        "m_waves": 2,
        "n_waves": 2,
        "async_load_bytes": 4,
    }
    if k // 64 // split_k < stages - 1:
        with pytest.raises(ValueError, match="needs at least"):
            gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
        return
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"async stages={stages} M{m} N{n} K{k} split_k={split_k}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


def test_gemm_a16w16_gfx942_tuned_table(tmp_path, monkeypatch):
    """A tuned-table row overrides the fallback; misses and the env switch use it."""
    _require_gfx942()
    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    m, n, k = 1000, 344, 4096
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    row = {
        "gfx": props.gcnArchName.split(":")[0],
        "cu_num": props.multi_processor_count,
        "M": m,
        "N": n,
        "K": k,
        "out_dtype": "bf16",
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "stages": 1,
        "split_k": 2,
        "m_waves": 2,
        "n_waves": 1,
        "k_waves": 2,
        "group_m": 4,
        "async_load_bytes": 16,
        "buffered_load": True,
        "direct_store": False,
        "pingpong": True,
        "split_k_reduce": True,
    }
    table = tmp_path / "table.csv"
    table.write_text(
        ",".join(row) + "\n" + ",".join(str(v) for v in row.values()) + "\n"
    )
    monkeypatch.setenv("AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942", str(table))
    g.reload_tuned_table()
    try:
        expected = {key: row[key] for key in g.TUNED_CONFIG_COLUMNS}
        assert g.gfx942_default_config(m, n, k) == expected
        assert g.gfx942_tuned_config(m + 8, n, k) is None
        assert g.gfx942_tuned_config(m, n, k, torch.float32) is None
        monkeypatch.setenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "1")
        assert g.gfx942_default_config(m, n, k) == g.gfx942_fallback_config(m, n, k)
        monkeypatch.delenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE")
        x, w, ref = _inputs(m, n, k)
        y = gemm_a16w16(x, w.t(), layout="nt")
        torch.cuda.synchronize()
        _assert_close(y, ref, f"tuned-table M{m} N{n} K{k}")
    finally:
        monkeypatch.delenv("AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942")
        g.reload_tuned_table()


def test_gemm_a16w16_gfx942_fp32_out():
    """fp32 output path on a compute-bound square shape."""
    _require_gfx942()
    m, n, k = 4096, 4096, 4096
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(x, w.t(), layout="nt", out_dtype=torch.float32)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"fp32-out M{m} N{n} K{k}")


@pytest.mark.parametrize(
    "store",
    [
        {"direct_store": True},
        {"direct_store": False},
        {"split_k": 4, "split_k_reduce": True},
        {"split_k": 4, "pingpong": False},  # atomic
    ],
    ids=["direct", "cshuffle", "workspace-split-k", "atomic-split-k"],
)
def test_gemm_a16w16_gfx942_fp32_out_precision(store):
    """fp32 output keeps fp32 accuracy on every store path (no bf16 rounding)."""
    _require_gfx942()
    m, n, k = 512, 512, 4096
    x, w, ref = _inputs(m, n, k)
    cfg = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "stages": 1,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "buffered_load": True,
        "async_load_bytes": 16,
        "pingpong": True,
        **store,
    }
    y = gemm_a16w16(x, w.t(), layout="nt", out_dtype=torch.float32, user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"fp32 output {store}")


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 384, 320),
        (300, 1024, 320),
        (100, 7168, 1024),
        (6000, 2560, 320),
        (77, 200, 64),
    ],
)
def test_gemm_a16w16_gfx942_fallback(m, n, k, monkeypatch):
    """Cost-model fallback on untuned shapes, down to the K=64 async last resort."""
    _require_gfx942()
    monkeypatch.setenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "1")
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(x, w.t(), layout="nt")
    torch.cuda.synchronize()
    _assert_close(y, ref, f"fallback M{m} N{n} K{k}")


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
@pytest.mark.parametrize(
    "split_k",
    [{}, {"split_k": 2, "split_k_reduce": True}],
    ids=["default", "workspace-split-k"],
)
@pytest.mark.parametrize("n", [337, 339])
def test_gemm_a16w16_gfx942_ragged_n(n, split_k, out_dtype):
    """N not a multiple of the store vector: the last vector of each row stops at N."""
    _require_gfx942()
    m, k = 200, 512
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    cfg = {"block_m": 64, "block_n": 64, "block_k": 64, **split_k} if split_k else {}
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"ragged N{n} {split_k} {out_dtype}")
    assert torch.isnan(out[m:]).all(), "stores past N overwrote the guard rows"


@pytest.mark.parametrize(
    "n,cfg,match",
    [
        (337, {"split_k": 2, "split_k_reduce": False}, "multiple of 4"),
        (337, {"block_k": 16}, "multiple of 32"),
        (337, {"block_k": 32}, "multiple of 64"),
        (
            336,
            {
                "block_m": 32,
                "block_n": 48,
                "block_k": 128,
                "stages": 1,
                "m_waves": 2,
                "n_waves": 1,
                "k_waves": 2,
                "buffered_load": True,
                "async_load_bytes": 16,
                "pingpong": True,
                "split_k": 2,
            },
            "vectors per thread",
        ),
    ],
    ids=[
        "atomic-split-k-ragged-n",
        "block_k-16",
        "async-block_k-32",
        "atomic-split-k-partial-vectors",
    ],
)
def test_gemm_a16w16_gfx942_rejects_unsupported(n, cfg, match):
    _require_gfx942()
    x, w, _ = _inputs(64, n, 512)
    tile = {"block_m": 32, "block_n": 32, "block_k": 64, "m_waves": 1, "n_waves": 1}
    with pytest.raises(ValueError, match=match):
        gemm_a16w16(x, w.t(), layout="nt", user_kwargs={**tile, **cfg})


def test_gemm_a16w16_gfx942_tile_override_drops_split_k(monkeypatch):
    """A caller tile does not inherit the default config's split-K: with
    split_k_reduce dropped, it would turn into atomic split-K."""
    _require_gfx942()
    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    monkeypatch.setenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "1")
    m, n, k = 250, 338, 10240
    assert g.gfx942_default_config(m, n, k)["split_k"] > 1
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(x, w.t(), layout="nt", user_kwargs={"block_m": 64})
    torch.cuda.synchronize()
    _assert_close(y, ref, "block_m override of a split-K default")


def test_flydsl_hgemm_gfx942_aligned_row_strided_inputs():
    _require_gfx942()
    m, n, k = 128, 192, 512
    x, w, ref = _inputs(m, n, k)
    x_padded = torch.empty((m, k + 8), dtype=x.dtype, device=x.device)
    w_padded = torch.empty((n, k + 8), dtype=w.dtype, device=w.device)
    x_strided, w_strided = x_padded[:, :k], w_padded[:, :k]
    x_strided.copy_(x)
    w_strided.copy_(w)
    assert not x_strided.is_contiguous() and not w_strided.is_contiguous()

    y = flydsl_hgemm(x_strided, w_strided)
    _assert_close(y, ref, "aligned row-strided flydsl_hgemm")


@pytest.mark.parametrize("case", ["misaligned-a-base", "misaligned-b-row"])
def test_flydsl_hgemm_gfx942_rejects_misaligned_inputs(case):
    _require_gfx942()
    m, n, k = 64, 128, 512
    x, w, _ = _inputs(m, n, k)
    if case == "misaligned-a-base":
        x = torch.empty((m, k + 8), dtype=x.dtype, device=x.device)[:, 1 : k + 1]
        operand = "a"
    else:
        w = torch.empty((n, k + 1), dtype=w.dtype, device=w.device)[:, :k]
        operand = "b"

    with pytest.raises(ValueError, match=rf"{operand} must have a 16-byte-aligned"):
        flydsl_hgemm(x, w)


def test_gemm_a16w16_gfx942_split_k_buffers_do_not_retain_streams():
    """Split-K semaphores are cached by stream handle: a transient stream is
    freed after its launch, and a stream on the same handle reuses the set."""
    _require_gfx942()
    import gc
    import weakref

    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    x, w, ref = _inputs(256, 256, 2048)
    cfg = {"block_m": 64, "block_n": 64, "block_k": 64, "stages": 2, "split_k": 4}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    y = gemm_a16w16(x, w.t(), layout="nt", user_kwargs=cfg, stream=stream)
    stream.synchronize()
    _assert_close(y, ref, "split-K on a transient stream")
    handle, alive = stream.cuda_stream, weakref.ref(stream)
    del stream
    gc.collect()
    assert alive() is None, "the split-K buffer cache kept the stream alive"
    same_handle = torch.cuda.ExternalStream(handle)
    device = torch.device("cuda", torch.cuda.current_device())
    assert g.get_split_k_buffers(
        same_handle.cuda_stream, device, 4, False
    ) is g.get_split_k_buffers(handle, device, 4, False)


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
def test_error_ratio_zero_reference(out_dtype):
    """An exact output of an all-zero reference is correct, a wrong one is not."""
    ref = torch.zeros(4, 4)
    assert error_ratio(torch.zeros(4, 4, dtype=out_dtype), ref) == 0.0
    assert error_ratio(torch.full((4, 4), 1e-3, dtype=out_dtype), ref) > 1.0


def _invalid_inputs(case):
    x, w, _ = _inputs(64, 128, 256)
    b = w.t()
    if case == "3d-a":
        return {"a": x[None], "b": b}
    if case == "fp16":
        return {"a": x.half(), "b": b.half()}
    if case == "k-mismatch":
        return {"a": x[:, :128], "b": b}
    if case == "col-major-a":
        return {"a": x.t().contiguous().t(), "b": b}
    if case == "row-major-b":
        return {"a": x, "b": b.contiguous()}
    if case == "out-dtype":
        return {"a": x, "b": b, "out": torch.empty(64, 128, device="cuda")}
    bf16_out = {"dtype": torch.bfloat16, "device": "cuda"}
    if case == "out-shape":
        return {"a": x, "b": b, "out": torch.empty(64, 64, **bf16_out)}
    return {"a": x, "b": b, "out": torch.empty(128, 64, **bf16_out).t()}


@pytest.mark.parametrize(
    "case,error",
    [
        ("3d-a", ValueError),
        ("fp16", TypeError),
        ("k-mismatch", ValueError),
        ("col-major-a", ValueError),
        ("row-major-b", ValueError),
        ("out-dtype", TypeError),
        ("out-shape", ValueError),
        ("out-strided", ValueError),
    ],
)
def test_gemm_a16w16_gfx942_rejects_invalid_inputs(case, error):
    """Inputs are validated with exceptions, which ``python -O`` keeps."""
    _require_gfx942()
    with pytest.raises(error):
        gemm_a16w16(layout="nt", out_dtype=torch.bfloat16, **_invalid_inputs(case))


def test_gemm_a16w16_gfx942_other_device():
    """Operands on a device that is not the current one, on its current stream
    or on an explicit one."""
    _require_gfx942()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    with torch.cuda.device(1):
        x, w, ref = _inputs(256, 256, 1024)
    assert torch.cuda.current_device() == 0
    _assert_close(gemm_a16w16(x, w.t()), ref, "operands on cuda:1")
    side = torch.cuda.Stream(device=1)
    side.wait_stream(torch.cuda.current_stream(1))
    y = gemm_a16w16(x, w.t(), stream=side)
    side.synchronize()
    _assert_close(y, ref, "operands and stream on cuda:1")
    with pytest.raises(ValueError, match="stream is on"):
        gemm_a16w16(x, w.t(), stream=torch.cuda.current_stream(0))


@pytest.mark.parametrize("layout", ["nn", "tn", "tt"])
def test_gemm_a16w16_gfx942_rejects_transposed_layouts(layout):
    _require_gfx942()
    x, w, _ = _inputs(32, 64, 128)
    with pytest.raises(ValueError, match="'nt' layout only"):
        gemm_a16w16(x, w.t(), layout=layout)


def test_gemm_a16w16_gfx942_split_k_launch_sequence():
    """Atomic and workspace split-K launches with one split_k, interleaved on a
    stream: each keeps its own semaphores, so no launch sees a stale tag."""
    _require_gfx942()
    m, n, k = 256, 256, 2048
    x, w, ref = _inputs(m, n, k)
    base = {"block_m": 64, "block_n": 64, "block_k": 64, "stages": 2, "split_k": 4}
    for i in range(3):
        for reduce in (False, True):
            y = gemm_a16w16(
                x, w.t(), layout="nt", user_kwargs={**base, "split_k_reduce": reduce}
            )
            torch.cuda.synchronize()
            _assert_close(y, ref, f"launch {i} split_k_reduce={reduce}")


@pytest.mark.parametrize(
    "split",
    [
        {"split_k": 1},
        {"split_k": 4, "split_k_reduce": True},
        {"split_k": 4, "split_k_reduce": False},
    ],
    ids=["no-split-k", "workspace-split-k", "atomic-split-k"],
)
def test_gemm_a16w16_gfx942_explicit_stream(split):
    """A launch on a delayed side stream is complete once that stream is:
    no host-side allocation or copy runs ahead on the current stream."""
    _require_gfx942()
    m, n, k = 256, 256, 2048
    x, w, ref = _inputs(m, n, k)
    cfg = {"block_m": 64, "block_n": 64, "block_k": 64, "stages": 2, **split}
    # Compile first, so the delay below is not spent compiling. Negated input:
    # a buffer the allocator hands back from this call holds -ref, not ref.
    gemm_a16w16(-x, w.t(), layout="nt", user_kwargs=cfg)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
    y = gemm_a16w16(x, w.t(), layout="nt", user_kwargs=cfg, stream=stream)
    stream.synchronize()
    _assert_close(y, ref, f"explicit stream {split}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
