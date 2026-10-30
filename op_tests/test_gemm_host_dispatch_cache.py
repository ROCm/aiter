# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side dispatch caches of the GEMM wrappers must not change any answer.

* the batched mxfp8_32 heuristic memoised per M class picks exactly what the
  per-M heuristic picks;
* the mxscale bpreshuffle lookup's N/K pre-check answers like the padded-M walk
  and reports an untuned weight shape once;
* the nested custom op called through its undecorated body (gemm_a8w8_blockscale
  -> gemm_a8w8_blockscale_bpreshuffle) returns what the registered torch.ops
  dispatch returns (torch_compile_guard's _eager_body), and a rebound module
  attribute still wins.

The first two and the routing contract run on CPU; the numerical checks run
on any GPU whose arch has the kernel and skip elsewhere.
"""

import itertools
import logging

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8
from aiter.ops.flydsl import batched_gemm_a8w8_gfx1250 as bmm
from aiter.ops.gemm_op_common import find_padded_m_row

# DeepSeek-V4.1 projection shapes at TP4/TP8 plus generic ones; K = 576 is the
# ragged shared-expert down projection.
_MX32_NK = [
    (512, 7168),
    (1024, 7168),
    (2048, 7168),
    (7168, 576),
    (7168, 1024),
    (7168, 2048),
    (4096, 4096),
    (1536, 512),
]


def _mx32_ms():
    """Every M up to 1024 and, above it, each tile boundary +-1 of every
    candidate tile_m up to well past the M-block limit."""
    ms = set(range(1, 1025))
    params, _ = bmm._mx32_m_class_params()
    for tm, _sat in params:
        for j in range(1, 80):
            ms.update((tm * j - 1, tm * j, tm * j + 1))
    ms.update(range(1025, 20000, 97))
    return sorted(m for m in ms if m > 0)


@pytest.fixture
def mx32_pick(monkeypatch):
    def setup(cu_num):
        monkeypatch.setattr(bmm, "get_cu_num", lambda: cu_num)
        bmm.pick_bmm_mx32_kernel_name.cache_clear()

    yield setup
    bmm.pick_bmm_mx32_kernel_name.cache_clear()


def _exact_or_error(b, m, n, k, w32):
    try:
        return bmm._pick_bmm_mx32_kernel_name_exact.__wrapped__(b, m, n, k, w32)
    except RuntimeError as e:
        return type(e)


def _cached_or_error(b, m, n, k, w32):
    try:
        return bmm.pick_bmm_mx32_kernel_name(b, m, n, k, w32)
    except RuntimeError as e:
        return type(e)


@pytest.mark.parametrize("cu_num", [256, 64])
@pytest.mark.parametrize("b", [1, 2, 8])
@pytest.mark.parametrize("w32", [False, True])
def test_mx32_pick_by_m_class_matches_exact(mx32_pick, cu_num, b, w32):
    mx32_pick(cu_num)
    ms = _mx32_ms()
    for n, k in _MX32_NK:
        # Ascending and descending M so each M class is first filled from both
        # of its ends.
        for m in itertools.chain(ms, reversed(ms)):
            assert _cached_or_error(b, m, n, k, w32) == _exact_or_error(
                b, m, n, k, w32
            ), (b, m, n, k, w32, cu_num)
    # A few hundred classes per weight shape, not one entry per M.
    assert len(bmm._MX32_PICK_BY_M_CLASS) < len(_MX32_NK) * len(ms) // 4


def test_mx32_parse_cache_returns_private_copies():
    name = bmm.bmm_mx32_kernel_name(64, 128, 256, 2, 2, 3)
    first = bmm.parse_bmm_mx32_kernel_name(name)
    first["tile_m"] = -1
    assert bmm.parse_bmm_mx32_kernel_name(name)["tile_m"] == 64
    assert bmm.parse_bmm_mx32_kernel_name("not_a_kernel") is None


@pytest.fixture
def mxscale_table(monkeypatch):
    table = {
        ("gfx1250", 256, 64, 7168, 1024, "32x32"): {"kernelName": "m64"},
        ("gfx1250", 256, 256, 7168, 1024, "32x32"): {"kernelName": "m256"},
        ("gfx1250", 256, 128, 2048, 7168, "1x32"): {"kernelName": "m128"},
    }
    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(gemm_op_a8w8, "get_cu_num", lambda: 256)
    monkeypatch.setattr(gemm_op_a8w8, "_load_mxscale_bpreshuffle_tuned", lambda: table)
    monkeypatch.setattr(gemm_op_a8w8, "_MXSCALE_UNTUNED_REPORTED", set())
    gemm_op_a8w8._mxscale_bpreshuffle_tuned_nk.cache_clear()
    gemm_op_a8w8.get_mxscale_bpreshuffle_config.cache_clear()
    yield table
    gemm_op_a8w8._mxscale_bpreshuffle_tuned_nk.cache_clear()
    gemm_op_a8w8.get_mxscale_bpreshuffle_config.cache_clear()


def test_mxscale_config_precheck_matches_padded_walk(mxscale_table):
    shapes = [(7168, 1024, "32x32"), (2048, 7168, "1x32"), (2048, 7168, "32x32")]
    shapes += [(4096, 4096, "32x32")]
    for (n, k, block), m in itertools.product(shapes, range(1, 1100)):
        want, _ = find_padded_m_row(
            mxscale_table,
            lambda pm, n=n, k=k, block=block: ("gfx1250", 256, pm, n, k, block),
            m,
            n,
            k,
        )
        assert gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, n, k, block) == want


def test_mxscale_untuned_reported_once_per_weight_shape(mxscale_table, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        gemm_op_a8w8.logger, "warning", lambda msg, *a, **kw: warnings.append(msg)
    )
    for m in range(1, 300):
        assert (
            gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, 4096, 4096, "32x32") is None
        )
        assert (
            gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, 4096, 4096, "1x32") is None
        )
    assert len(warnings) == 2
    assert all("untuned" in w for w in warnings)


def test_blockscale_bpreshuffle_nested_body_and_rebind(monkeypatch):
    calls = []

    def body(XQ, WQ, x_scale, w_scale, dtype=dtypes.bf16, out=None):
        calls.append((x_scale.dtype, w_scale.dtype))
        return torch.zeros(XQ.shape[0], WQ.shape[0], dtype=dtype)

    op = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle
    assert op._eager_body.__name__ == "gemm_a8w8_blockscale_bpreshuffle"
    monkeypatch.setattr(op, "_eager_body", body)
    x = torch.zeros(4, 64, dtype=dtypes.fp8)
    w = torch.zeros(32, 64, dtype=dtypes.fp8)
    xs = torch.zeros(4, 2, dtype=torch.uint8)
    ws = torch.zeros(1, 2, dtype=torch.uint8)
    y = gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws, isBpreshuffled=True)
    assert y.shape == (4, 32)
    assert calls == [(dtypes.fp8_e8m0, dtypes.fp8_e8m0)]

    rebound = []

    def patched(XQ, WQ, x_scale, w_scale, dtype=dtypes.bf16, out=None):
        rebound.append(True)
        return torch.ones(XQ.shape[0], WQ.shape[0], dtype=dtype)

    monkeypatch.setattr(gemm_op_a8w8, "gemm_a8w8_blockscale_bpreshuffle", patched)
    y = gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws, isBpreshuffled=True)
    assert rebound == [True] and len(calls) == 1
    assert bool((y == 1).all())


def _gpu():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu(), reason="needs a GPU")
@pytest.mark.parametrize("m", [1, 37, 512, 1758])
@pytest.mark.parametrize("n,k", [(7168, 1024), (2048, 7168)])
@pytest.mark.parametrize("w_rows", [1, 32])
def test_blockscale_bpreshuffle_nested_matches_op(m, n, k, w_rows):
    from aiter import benchmark_data_init as bench_init
    from aiter.ops.flydsl.batched_gemm_a8w8 import bmm_a8w8_mxfp8_supported
    from aiter.ops.shuffle import shuffle_weight

    if not bmm_a8w8_mxfp8_supported(f"{w_rows}x32"):
        pytest.skip(f"no group32 {w_rows}x32 bpreshuffle GEMM on this arch")
    gen = bench_init.make_generator(0)
    x = bench_init.fill_fp8((m, k), "uniform", gen, dtype=dtypes.fp8)
    w = bench_init.fill_fp8((n, k), "uniform", gen, dtype=dtypes.fp8)
    xs = bench_init.fill_scale_e8m0((m, k // 32), "auto", gen).view(dtypes.fp8_e8m0)
    ws = bench_init.fill_scale_e8m0((-(-n // w_rows), k // 32), "auto", gen).view(
        dtypes.fp8_e8m0
    )
    w_shuf = shuffle_weight(w, layout=(16, 16))
    nested = gemm_op_a8w8.gemm_a8w8_blockscale(
        x, w_shuf, xs, ws, dtypes.bf16, isBpreshuffled=True
    )
    via_op = torch.ops.aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, w_shuf, xs, ws, dtypes.bf16
    )
    assert torch.equal(nested, via_op)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(pytest.main([__file__, "-v"]))
