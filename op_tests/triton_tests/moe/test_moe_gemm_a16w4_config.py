# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Config resolution for the a16w4 MOE GEMM.

`test_moe_gemm_a16w4.py` skips unless `arch_info.is_fp4_avail()`, which is
gfx950/gfx1250 only, so nothing there covers the gfx942 tuned table. Resolution
is pure Python, so these run anywhere.

The expected tiles below are written out rather than read from the shipped
JSON: a test that reads the table can only prove it resolves to itself, and
would not notice an entry being edited to a tile that was never measured.
"""

import json
from pathlib import Path
from unittest import mock

import pytest
import triton

from aiter.ops.triton.moe import moe_op_gemm_a16w4
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import get_kernel_config_triton
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch

_TABLE = (
    Path(moe_op_gemm_a16w4.__file__).resolve().parents[1]
    / "configs/gfx942/triton/moe/a16w4/DEFAULT.json"
)

# An arch that ships no a16w4 table, so get_moe_dispatch returns {}.
_UNTUNED_ARCH = "gfx000"

# (block_m, N, K) -> (block_n, block_k, num_warps, num_stages, waves_per_eu,
# matrix_instr_nonkdim), as measured on MI325X. Keep in sync with DEFAULT.json.
_MEASURED = {
    (128, 1024, 5120): (256, 128, 8, 2, 2, 32),
    (128, 1536, 5120): (256, 64, 4, 1, 2, 32),
    (128, 5120, 512): (256, 64, 8, 2, 0, 32),
    (128, 5120, 768): (256, 64, 4, 2, 2, 32),
    (64, 1024, 5120): (256, 256, 8, 2, 0, 32),
    (64, 1536, 5120): (256, 256, 8, 2, 0, 32),
    (64, 5120, 512): (256, 128, 4, 1, 0, 16),
    (64, 5120, 768): (256, 128, 4, 2, 0, 16),
}


class _Routing:
    """Minimal RoutingData: block_m plus the n_blocks the block_m=16 path calls."""

    def __init__(self, block_m):
        self.block_m = block_m

    def n_blocks(self, m, block_m):
        return triton.cdiv(m, block_m)


def _stock_config(block_m, n, k):
    """What get_kernel_config_triton returns when no table is shipped."""
    get_moe_dispatch.cache_clear()
    try:
        with mock.patch.object(moe_op_gemm_a16w4, "get_arch", lambda: _UNTUNED_ARCH):
            return get_kernel_config_triton(
                m=block_m, n=n, k=k, routing_data=_Routing(block_m)
            )
    finally:
        get_moe_dispatch.cache_clear()


@pytest.fixture
def on_gfx942(monkeypatch):
    """Resolve as gfx942 regardless of the host the suite runs on.

    ``get_moe_dispatch`` is lru_cached on ``(config_name, arch, backend)``;
    clear it either side so a table resolved under another arch -- including
    the untuned probe in ``_stock_config`` -- cannot leak into these tests.
    """
    get_moe_dispatch.cache_clear()
    monkeypatch.setattr(moe_op_gemm_a16w4, "get_arch", lambda: "gfx942")
    yield
    get_moe_dispatch.cache_clear()


def test_table_matches_measured_tiles():
    """The shipped JSON carries exactly the keys and tiles that were measured."""
    with _TABLE.open() as f:
        table = json.load(f)

    shipped = {}
    for key, entry in table.items():
        bm, n, k = key.split("_")
        shipped[(int(bm[2:]), int(n[1:]), int(k[1:]))] = (
            entry["BLOCK_SIZE_N"],
            entry["BLOCK_SIZE_K"],
            entry["num_warps"],
            entry["num_stages"],
            entry["waves_per_eu"],
            entry["matrix_instr_nonkdim"],
        )

    assert shipped == _MEASURED


@pytest.mark.parametrize(
    "shape,want",
    sorted(_MEASURED.items()),
    ids=lambda v: str(v) if isinstance(v, tuple) else None,
)
def test_measured_tile_reaches_the_kernel(on_gfx942, shape, want):
    """Each measured shape resolves to the tile it was tuned with."""
    block_m, n, k = shape
    block_n, block_k, num_warps, num_stages, waves_per_eu, nonkdim = want

    cfg = get_kernel_config_triton(m=block_m, n=n, k=k, routing_data=_Routing(block_m))

    assert cfg["block_m"] == block_m, "block_m is the dispatch key, not a tunable"
    assert cfg["block_n"] == block_n
    assert cfg["block_k"] == block_k
    assert cfg["num_warps"] == num_warps
    assert cfg["num_stages"] == num_stages
    assert cfg["waves_per_eu"] == waves_per_eu
    assert cfg["matrix_instr_nonkdim"] == nonkdim


@pytest.mark.parametrize("shape", sorted(k for k, v in _MEASURED.items() if v[1] == 64))
def test_block_k_64_pairs_with_nonkdim_32(on_gfx942, shape):
    """block_k=64 only compiles with matrix_instr_nonkdim=32."""
    block_m, n, k = shape
    cfg = get_kernel_config_triton(m=block_m, n=n, k=k, routing_data=_Routing(block_m))
    assert cfg["block_k"] == 64
    assert cfg["matrix_instr_nonkdim"] == 32


@pytest.mark.parametrize("block_m", [16, 32])
@pytest.mark.parametrize("n,k", [(1024, 5120), (5120, 512)])
def test_decode_block_m_untouched(on_gfx942, block_m, n, k):
    """Decode ships no entries, so it resolves exactly as the stock path does."""
    cfg = get_kernel_config_triton(m=block_m, n=n, k=k, routing_data=_Routing(block_m))
    assert cfg == _stock_config(block_m, n, k)


def test_unlisted_prefill_shape_falls_back(on_gfx942):
    """A prefill block_m with no entry resolves as the stock path does."""
    cfg = get_kernel_config_triton(m=128, n=999, k=777, routing_data=_Routing(128))
    assert cfg == _stock_config(128, 999, 777)


@pytest.mark.parametrize("shape", sorted(_MEASURED))
def test_other_arch_unaffected(monkeypatch, shape):
    """An arch with no table keeps the stock tile even on a key gfx942 tunes."""
    monkeypatch.setattr(moe_op_gemm_a16w4, "get_arch", lambda: "gfx950")
    block_m, n, k = shape
    cfg = get_kernel_config_triton(m=block_m, n=n, k=k, routing_data=_Routing(block_m))
    assert cfg == _stock_config(block_m, n, k)
