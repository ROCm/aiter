# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-sparse MHA v4: LUT handling, work-table ordering, and tile-count scaling.

Split out of test_mha_v4.py, which had grown past 2600 lines with the sparse cases forming
one contiguous block of it. Absorbs the former test_mha_v4_sparse_tile_scaling.py.
"""

import os
import subprocess
import sys
from typing import NamedTuple

import pytest
import torch
import torch._dynamo

from aiter.jit.core import AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    mha_v4_kv_tile,
    mha_v4_packed,
    mha_v4_sparse_work_table,
    native_fp8_format,
)
from aiter.ops.mha_v4_quant import (
    quantize_fp8,
    quantize_fp8_rotated,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut


@pytest.fixture(autouse=True)
def isolate_dynamo_cache():
    """Keep each test's ``torch.compile`` behaviour independent of the tests that ran before it.

    Dynamo caches compiled entries per code object, and this file compiles ``mha_v4`` under many
    format combinations. Sharing that cache across tests means the suite drifts toward the recompile
    limit and whichever test compiles last fails under ``fullgraph=True`` -- a failure that reports
    against a kernel while actually depending on how many earlier tests got far enough to compile.
    """
    torch._dynamo.reset()
    yield


def _mha_v4_sparse_co_available() -> bool:
    gfx = get_gfx()
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    if gfx == "gfx942":
        return os.path.isfile(
            os.path.join(
                asm_dir, "gfx942", "fmha_v4_fwd", "MI300", "fwd_hd128_fp8_sparse.co"
            )
        )
    return os.path.isfile(
        os.path.join(asm_dir, "gfx950", "fmha_v4_fwd", "fwd_hd128_fp8_sparse.co")
    )


_MHA_V4_SPARSE_ARCH = get_gfx() in ("gfx942", "gfx950")


def _mha_v4_mxfp6_sparse_co_available() -> bool:
    """MXFP6 Q/K/V sparse is a gfx950-only row."""
    if get_gfx() != "gfx950":
        return False
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    return os.path.isfile(
        os.path.join(asm_dir, "gfx950", "fmha_v4_fwd", "fwd_hd128_mxfp6_sparse.co")
    )


def test_mha_v4_packed_rejects_partial_lut():
    dummy = torch.empty(0)
    with pytest.raises(ValueError, match="all be set or all omitted"):
        mha_v4_packed(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=dummy,
        )


def test_mha_v4_rejects_wrong_block_mask_shape():
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="block_mask must have shape"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse schema")
def test_mha_v4_sparse_schema_mutates_only_out():
    dense = str(torch.ops.aiter.mha_v4_fwd_launch.default._schema)
    assert "Tensor kv_block_indices" not in dense
    assert "Tensor(a6!) out" in dense

    sparse = str(torch.ops.aiter.mha_v4_fwd_sparse_launch.default._schema)
    assert "Tensor kv_block_indices" in sparse
    assert "Tensor lut_start" in sparse
    assert "Tensor lut_count" in sparse
    assert "Tensor(a6!) out" in sparse
    assert sparse.endswith("-> ()")


def _work_table_counts(total, pattern):
    """LUT lengths covering the tie structures the ordering has to get right."""
    if pattern == "uniform":
        return torch.full((total,), 7, device="cuda", dtype=torch.int32)
    if pattern == "zeros":
        return torch.zeros((total,), device="cuda", dtype=torch.int32)
    if pattern == "random":
        return torch.randint(0, 64, (total,), device="cuda", dtype=torch.int32)
    if pattern == "wide_random":
        return torch.randint(0, 8192, (total,), device="cuda", dtype=torch.int32)
    if pattern == "two_values":
        alternating = torch.arange(total, device="cuda") % 3 == 0
        return torch.where(alternating, 9, 4).to(torch.int32)
    if pattern == "descending":
        return torch.arange(total, 0, -1, device="cuda", dtype=torch.int32)
    return torch.arange(1, total + 1, device="cuda", dtype=torch.int32)


def _unpack_work_table(table, nhead, q_tiles):
    q_idx = (table & 0xFFFF).long()
    h_idx = ((table >> 16) & 0xFF).long()
    b_idx = ((table >> 24) & 0xFF).long()
    return (b_idx * nhead + h_idx) * q_tiles + q_idx


# The table has one entry per (batch, head, query tile). 8192 is the point where the builder hands
# the sort to ATen, so straddle it, and include sizes that are not multiples of a wave or workgroup.
@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    ("batch", "nhead", "q_tiles"),
    [
        (1, 1, 1),
        (1, 8, 3),
        (2, 5, 7),
        (1, 16, 32),
        (1, 32, 32),
        (1, 5, 296),
        (4, 16, 64),
        (8, 16, 64),
        (8, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "pattern",
    [
        "uniform",
        "zeros",
        "random",
        "wide_random",
        "two_values",
        "descending",
        "ascending",
    ],
)
def test_mha_v4_sparse_work_table_is_longest_lut_first(batch, nhead, q_tiles, pattern):
    torch.manual_seed(7)
    total = batch * nhead * q_tiles
    counts = _work_table_counts(total, pattern)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)
    visited = _unpack_work_table(table, nhead, q_tiles)

    # Every tile exactly once. This is the part a wrong table would turn into a wrong result.
    assert torch.equal(visited.sort().values, torch.arange(total, device="cuda"))

    # Longest LUT first, so no heavy tile straggles behind the rest.
    ordered = counts[visited]
    assert bool((ordered[:-1] >= ordered[1:]).all())

    # Ties keep raster order, which is what leaves uniform counts spatially coherent. A stable
    # reference sort pins the whole permutation, not just the two properties above.
    expected = torch.argsort(counts, descending=True, stable=True)
    assert torch.equal(visited, expected)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    "batch,nhead,q_tiles", [(1, 16, 32), (1, 5, 296), (8, 16, 64), (8, 32, 64)]
)
def test_mha_v4_sparse_work_table_leaves_uniform_counts_in_raster_order(
    batch, nhead, q_tiles
):
    """Top-k sparsity gives every tile the same LUT length, and that case must not be shuffled."""
    total = batch * nhead * q_tiles
    counts = torch.full((total,), 5, device="cuda", dtype=torch.int32)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)

    visited = _unpack_work_table(table, nhead, q_tiles)
    assert torch.equal(visited, torch.arange(total, device="cuda"))


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(
    "launch",
    [
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                native_fp8_format(),
                native_fp8_format(),
                native_fp8_format(),
                block_mask=mask,
            ),
            id="fp8",
        ),
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                AttentionFormat.INT8,
                AttentionFormat.INT8,
                native_fp8_format(),
                block_mask=mask,
            ),
            id="i8fp8",
        ),
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                native_fp8_format(),
                native_fp8_format(),
                native_fp8_format(),
                block_mask=mask,
                q_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
                k_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
                v_scale_mode=AttentionScaleMode.F32_PER_TENSOR,
            ),
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
            id="mxfp8",
        ),
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                native_fp8_format(),
                native_fp8_format(),
                AttentionFormat.MXFP6,
                block_mask=mask,
            ),
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
            id="f8f6",
        ),
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                AttentionFormat.MXFP6,
                AttentionFormat.MXFP6,
                native_fp8_format(),
                block_mask=mask,
            ),
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
            id="f6f8",
        ),
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                AttentionFormat.MXFP6,
                AttentionFormat.MXFP6,
                AttentionFormat.MXFP4,
                block_mask=mask,
            ),
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
            id="f6f4",
        ),
        pytest.param(
            lambda q, k, v, mask: mha_v4(
                q,
                k,
                v,
                AttentionFormat.MXFP4,
                AttentionFormat.MXFP4,
                AttentionFormat.MXFP4,
                block_mask=mask,
            ),
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
            id="mxfp4",
        ),
    ],
)
def test_mha_v4_sparse_all_true_mask_matches_dense(launch):
    torch.manual_seed(41)
    q = torch.randn((1, 511, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(
        (1, 5, 2, 512 // mha_v4_kv_tile()), device="cuda", dtype=torch.bool
    )
    dense = launch(q, k, v, None)
    sparse = launch(q, k, v, mask)
    torch.cuda.synchronize()
    _assert_sparse_matches_dense(sparse, dense)


def _assert_sparse_matches_dense(sparse, dense, message=None):
    """Compare code objects that use different softmax reduction schedules."""
    cosine = torch.nn.functional.cosine_similarity(
        sparse.float().flatten(), dense.float().flatten(), dim=0
    )
    assert cosine > 0.99, message
    assert torch.isfinite(sparse).all()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_f4f4_sparse_all_true_mask_matches_dense():
    """Retained FP8-P sparse F4F4 remains close to dense FP6-P on an all-true mask."""
    torch.manual_seed(41)
    q = torch.randn((1, 511, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(
        (1, 5, 2, 512 // mha_v4_kv_tile()), device="cuda", dtype=torch.bool
    )
    args = (AttentionFormat.MXFP4,) * 3
    dense = mha_v4(q, k, v, *args)
    sparse = mha_v4(q, k, v, *args, block_mask=mask)
    torch.cuda.synchronize()

    _assert_sparse_matches_dense(sparse, dense)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    "v_format",
    [
        pytest.param(AttentionFormat.BF16, id="bf16"),
        pytest.param(native_fp8_format(), id="bf16fp8"),
    ],
)
def test_mha_v4_sparse_dense_only_formats_reject_block_mask(v_format):
    q = torch.zeros((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(
        (1, 2, 1, 256 // mha_v4_kv_tile()), device="cuda", dtype=torch.bool
    )
    with pytest.raises(NotImplementedError, match="does not have a BF16 manifest row"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.BF16,
            AttentionFormat.BF16,
            v_format,
            block_mask=mask,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 validation")
@pytest.mark.skipif(
    not _mha_v4_mxfp6_sparse_co_available(),
    reason="sorted-sparse MXFP6 code object is not deployed",
)
def test_mha_v4_mxfp6_accepts_block_mask():
    """MXFP6 Q/K/V has a sorted-sparse row, so a block mask is dispatched, not rejected."""
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(
        (1, 2, 1, 256 // mha_v4_kv_tile()), device="cuda", dtype=torch.bool
    )
    out = mha_v4(
        q,
        q,
        q,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        block_mask=mask,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_block_mask_compiles_without_graph_breaks():
    """The mask path derives its geometry from host state, which Dynamo cannot trace.

    mha_v4_kv_tile() reads the manifest and get_gfx() shells out to rocminfo, so both sit behind
    torch_compile_guard. Without that the sparse mask path costs graph breaks per trace and fails
    under fullgraph, which no other test in this file would notice.
    """
    torch.manual_seed(41)
    kv_tile = mha_v4_kv_tile()
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 4 * kv_tile, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones((1, 2, 1, 4), device="cuda", dtype=torch.bool)
    fp8_format = native_fp8_format()

    def call():
        return mha_v4(q, k, v, fp8_format, fp8_format, fp8_format, block_mask=mask)

    explained = torch._dynamo.explain(call)()
    assert explained.break_reasons == [], [
        str(reason.reason) for reason in explained.break_reasons
    ]

    eager = call()
    compiled = torch.compile(call, fullgraph=True)()
    torch.cuda.synchronize()

    assert torch.equal(eager, compiled)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        pytest.param(
            native_fp8_format(),
            native_fp8_format(),
            id="fp8",
        ),
        pytest.param(
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            marks=pytest.mark.skipif(
                get_gfx() != "gfx950", reason="gfx950 MXFP4 sparse"
            ),
            id="mxfp4",
        ),
    ],
)
def test_mha_v4_sparse_gqa_all_true_mask_matches_repeated_kv(q_format, v_format):
    torch.manual_seed(41)
    query_heads = 8
    kv_heads = 2
    gqa_ratio = query_heads // kv_heads
    q = torch.randn((1, 256, query_heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, kv_heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv_tiles = 256 // mha_v4_kv_tile()
    mask = torch.ones((1, query_heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    k_repeated = k.repeat_interleave(gqa_ratio, dim=2)
    v_repeated = v.repeat_interleave(gqa_ratio, dim=2)

    gqa_sparse = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask)
    mha_sparse = mha_v4(
        q,
        k_repeated,
        v_repeated,
        q_format,
        q_format,
        v_format,
        block_mask=mask,
    )
    torch.cuda.synchronize()

    assert torch.equal(gqa_sparse, mha_sparse)
    if q_format != AttentionFormat.MXFP4:
        gqa_dense = mha_v4(q, k, v, q_format, q_format, v_format)
        mha_dense = mha_v4(q, k_repeated, v_repeated, q_format, q_format, v_format)
        assert torch.equal(gqa_dense, mha_dense)
        _assert_sparse_matches_dense(gqa_sparse, gqa_dense)
    assert torch.equal(gqa_sparse, mha_sparse)


class _Operand(NamedTuple):
    """A quantized MHA v4 operand and the descale it was produced with."""

    quantized: torch.Tensor
    descale: torch.Tensor


def _sparse_fp8_operands(sequence_k, heads=2, sequence_q=256, batch=1, seed=0):
    """Quantize once so sparse and reference runs share descales exactly.

    Re-quantizing a KV slice would pick a different per-tensor amax, which shifts every
    value and hides whether the kernel read the KV blocks the LUT named.
    """
    torch.manual_seed(seed)
    q = torch.randn(
        (batch, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    return (
        _Operand(*quantize_fp8_rotated(q)),
        _Operand(*quantize_fp8_rotated(k)),
        _Operand(*quantize_fp8(v)),
    )


def _sparse_fp8_launch(q, k, v, block_mask=None):
    lut = {}
    if block_mask is not None:
        indices, start, count = block_attn_mask_to_ragged_lut(
            block_mask,
            num_heads=block_mask.shape[1],
            return_none_if_dense=False,
        )
        lut = {
            "kv_block_indices": indices,
            "lut_start": start,
            "lut_count": count,
        }
    fp8_format = native_fp8_format()
    return mha_v4_packed(
        q.quantized,
        k.quantized,
        v.quantized,
        q.descale,
        k.descale,
        v.descale,
        fp8_format,
        fp8_format,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        **lut,
    )


def _gather_kv_tiles(operand, tiles):
    """Concatenate the named KV tiles, leaving the quantized bytes and descale untouched."""
    kv_tile = mha_v4_kv_tile()
    gathered = torch.cat(
        [operand.quantized[:, tile * kv_tile : (tile + 1) * kv_tile] for tile in tiles],
        dim=1,
    )
    return _Operand(gathered.contiguous(), operand.descale)


def _tile_mask(heads, kv_tiles, tiles, q_tiles=1, batch=1):
    mask = torch.zeros(
        (batch, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool
    )
    for tile in tiles:
        mask[:, :, :, tile] = True
    return mask


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("tiles", [(0,), (1,), (3,), (0, 2), (1, 2, 3)])
def test_mha_v4_sparse_reads_only_the_kv_tiles_the_lut_names(tiles):
    """A kernel that ignored kv_block_indices would pass every all-True test."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    mask = _tile_mask(heads, kv_tiles, tiles)
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    # Dense over exactly the selected tiles: same quantized bytes, same descales, so the
    # only difference is which KV blocks take part.
    dense = _sparse_fp8_launch(
        q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
    )
    torch.cuda.synchronize()

    _assert_sparse_matches_dense(sparse, dense)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_distinct_kv_tiles_give_distinct_results():
    """Guards the reference itself: selecting different tiles must change the output."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    outputs = [
        _sparse_fp8_launch(q, k, v, block_mask=_tile_mask(heads, kv_tiles, (tile,)))
        for tile in range(kv_tiles)
    ]
    torch.cuda.synchronize()

    for tile in range(1, kv_tiles):
        assert not torch.equal(outputs[0], outputs[tile]), (
            f"kv tile 0 and kv tile {tile} produced identical output, so the kernel is "
            "not reading kv_block_indices"
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_gives_each_head_its_own_kv_tiles():
    """4-D masks may give heads different KV lists; each head must follow its own row."""
    heads = 3
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    per_head = ((0,), (3,), (1, 2))
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    mask = torch.zeros((1, heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    for head, tiles in enumerate(per_head):
        for tile in tiles:
            mask[:, head, :, tile] = True
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for head, tiles in enumerate(per_head):
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        _assert_sparse_matches_dense(
            sparse[:, :, head],
            dense[:, :, head],
            f"head {head} did not attend to tiles {tiles}",
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_follows_the_lut_across_query_tiles():
    """Multiple query tiles exercise the work table on a real launch, not just its ordering."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q_tiles = 2
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, sequence_q=256 * q_tiles
    )

    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    mask[:, :, 0, 0] = True
    mask[:, :, 1, 3] = True
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for q_tile, tiles in ((0, (0,)), (1, (3,))):
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        rows = slice(q_tile * 256, (q_tile + 1) * 256)
        _assert_sparse_matches_dense(
            sparse[:, rows],
            dense[:, rows],
            f"query tile {q_tile} did not attend to tiles {tiles}",
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("tail_rows", [64, 128, 200])
def test_mha_v4_sparse_partial_query_tile_follows_the_lut(tail_rows):
    """A trailing partial query tile has to select and zero the same way a full one does.

    Production sequence lengths are not multiples of the 256-row query tile, and the empty-row
    no-op is built on the same tail masking the partial tile uses, so the two belong in one
    case: the short tile must still read the KV blocks its row names, and an all-False row on
    that tile must come back zero rather than reading the masked-off remainder.
    """
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q_tiles = 3
    per_tile = ((0, (0,)), (1, (1, 2)), (2, (3,)))
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile,
        heads=heads,
        sequence_q=256 * (q_tiles - 1) + tail_rows,
    )

    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    for q_tile, tiles in per_tile:
        for tile in tiles:
            mask[:, :, q_tile, tile] = True
    mask[:, 1, q_tiles - 1, :] = False  # head 1's partial-tile row selects nothing
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for q_tile, tiles in per_tile:
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        rows = slice(q_tile * 256, min((q_tile + 1) * 256, sparse.shape[1]))
        live_heads = 1 if q_tile == q_tiles - 1 else heads
        _assert_sparse_matches_dense(
            sparse[:, rows, :live_heads],
            dense[:, rows, :live_heads],
            f"query tile {q_tile} did not attend to tiles {tiles}",
        )

    tail = slice((q_tiles - 1) * 256, sparse.shape[1])
    assert torch.equal(
        sparse[:, tail, 1], torch.zeros_like(sparse[:, tail, 1])
    ), "empty row on the partial query tile is not zero"
    # Without this the case would also pass on a kernel that skipped the short tile entirely,
    # since both sides of the comparison above would then be zero.
    assert (
        sparse[:, tail, 0].abs().max() > 0
    ), "live partial-tile row came back degenerate"
    assert torch.isfinite(sparse).all(), "partial query tile leaked NaN or infinity"


def _gfx950_only(launch, label):
    return pytest.param(
        launch,
        id=label,
        marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
    )


_EMPTY_ROW_LAUNCHES = [
    pytest.param(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            native_fp8_format(),
            native_fp8_format(),
            native_fp8_format(),
            block_mask=m,
        ),
        id="fp8",
    ),
    pytest.param(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            native_fp8_format(),
            block_mask=m,
        ),
        id="i8fp8",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            native_fp8_format(),
            native_fp8_format(),
            native_fp8_format(),
            block_mask=m,
            q_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
            k_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
            v_scale_mode=AttentionScaleMode.F32_PER_TENSOR,
        ),
        "mxfp8",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            native_fp8_format(),
            native_fp8_format(),
            AttentionFormat.MXFP6,
            block_mask=m,
        ),
        "f8f6",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            native_fp8_format(),
            block_mask=m,
        ),
        "f6f8",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP4,
            block_mask=m,
        ),
        "f6f4",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            block_mask=m,
        ),
        "mxfp4",
    ),
]


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("launch", _EMPTY_ROW_LAUNCHES)
def test_mha_v4_sparse_empty_row_writes_zeros(launch):
    """An all-False row selects no KV block, so its output tile must be zero, not garbage.

    Its lut_start also sits one past the last kv_block_indices entry, which is what used to walk
    the prologue's unguarded reads into a multi-gigabyte scalar offset and fault the kernel.
    """
    heads = 2
    kv_tiles = 4
    selected = (0, 1)
    torch.manual_seed(0)
    q = torch.randn((1, 256, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        (1, kv_tiles * mha_v4_kv_tile(), heads, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    mask = torch.zeros((1, heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    for tile in selected:
        mask[:, 0, :, tile] = True  # head 0 selects two tiles; head 1 stays all-False
    out = launch(q, k, v, mask)
    torch.cuda.synchronize()

    assert torch.equal(
        out[:, :, 1], torch.zeros_like(out[:, :, 1])
    ), "empty row is not zero"
    assert torch.isfinite(out).all(), "empty row leaked NaN or infinity"
    assert out[:, :, 0].abs().max() > 0, "live row came back degenerate"

    # The empty row must not perturb the row that does select tiles: give head 1 a tile and head 0's
    # output has to stay bit-identical, since each workgroup owns one (batch, head, query tile).
    mask[:, 1, :, 0] = True
    populated = launch(q, k, v, mask)
    torch.cuda.synchronize()
    assert torch.equal(
        out[:, :, 0], populated[:, :, 0]
    ), "empty row disturbed the live row"


def test_mha_v4_rejects_non_bool_block_mask():
    """Counts come from a sum but the fill uses truthiness, so non-bool masks disagree."""
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 256 // mha_v4_kv_tile()), dtype=torch.int32)
    with pytest.raises(ValueError, match="block_mask must be a bool tensor"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs to mismatch devices"
)
def test_mha_v4_rejects_block_mask_on_another_device():
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16, device="cuda:0")
    mask = torch.ones(
        (1, 2, 1, 256 // mha_v4_kv_tile()), dtype=torch.bool, device="cuda:1"
    )
    with pytest.raises(ValueError, match="block_mask must be on the same device"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_rejects_empty_kv_block_indices():
    """Rows may be empty, but the ASM still dereferences the row base, so the buffer cannot be."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)
    rows = heads  # batch 1, one query tile
    device = q.quantized.device
    fp8_format = native_fp8_format()
    with pytest.raises(RuntimeError, match="must be non-empty"):
        mha_v4_packed(
            q.quantized,
            k.quantized,
            v.quantized,
            q.descale,
            k.descale,
            v.descale,
            fp8_format,
            fp8_format,
            fp8_format,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=torch.zeros(0, dtype=torch.int32, device=device),
            lut_start=torch.zeros(rows, dtype=torch.int32, device=device),
            lut_count=torch.ones(rows, dtype=torch.int32, device=device),
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(
    "mutation,message",
    [
        pytest.param(
            "indices.fill_(9999)",
            "outside",
            id="index_out_of_range",
        ),
        pytest.param(
            "start.fill_(-1)",
            "negative",
            id="negative_start",
        ),
    ],
)
def test_mha_v4_sparse_validation_rejects_malformed_lut(mutation, message):
    """Enable opt-in validation before AITER loads, without slowing the parent test process."""
    probe = f"""
from op_tests.test_mha_v4_sparse import (
    AttentionFormat,
    AttentionScaleMode,
    _sparse_fp8_operands,
    _tile_mask,
    block_attn_mask_to_ragged_lut,
    mha_v4_kv_tile,
    mha_v4_packed,
    native_fp8_format,
)

heads = 2
kv_tiles = 4
q, k, v = _sparse_fp8_operands(
    sequence_k=kv_tiles * mha_v4_kv_tile(), heads=heads
)
mask = _tile_mask(heads, kv_tiles, (0, 1))
indices, start, count = block_attn_mask_to_ragged_lut(
    mask, num_heads=heads, return_none_if_dense=False
)
{mutation}
fp8_format = native_fp8_format()
mha_v4_packed(
    q.quantized,
    k.quantized,
    v.quantized,
    q.descale,
    k.descale,
    v.descale,
    fp8_format,
    fp8_format,
    fp8_format,
    AttentionScaleMode.F32_PER_TENSOR,
    AttentionScaleMode.F32_PER_TENSOR,
    AttentionScaleMode.F32_PER_TENSOR,
    kv_block_indices=indices,
    lut_start=start,
    lut_count=count,
)
"""
    env = {**os.environ, "AITER_MHA_V4_VALIDATE_LUT": "1"}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=AITER_ROOT_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr


FP8 = native_fp8_format()
_MX_SCALES = {
    "q_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
    "k_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
    "v_scale_mode": AttentionScaleMode.F32_PER_TENSOR,
}

# (q/k format, v format, kwargs). K takes Q's format, as the manifest rows do.
SPARSE_RECIPES = {
    "i8fp8": (AttentionFormat.INT8, FP8, {}),
    "fp8": (FP8, FP8, {}),
    "mxfp8": (FP8, FP8, _MX_SCALES),
    "f8f6": (FP8, AttentionFormat.MXFP6, {}),
    "f6f8": (AttentionFormat.MXFP6, FP8, {}),
    "f6f4": (AttentionFormat.MXFP6, AttentionFormat.MXFP4, {}),
    "mxfp4": (AttentionFormat.MXFP4, AttentionFormat.MXFP4, {}),
}

# Two tiles is the baseline the growing counts are judged against; the rest span the region the
# suite never covered. 32 tiles is where the shipped mxfp8 object had lost a quarter of its cosine.
BASELINE_TILES = 2
GROWN_TILES = (4, 7, 8, 12, 16, 32)

# Each row carries its own quantization error, so the bar is that accuracy does not DEGRADE with
# tile count. Only a modest absolute floor is applied on top, to catch a row that is broken outright.
MAX_DEGRADATION = 0.01
ABSOLUTE_FLOOR = 0.95

requires_sparse = pytest.mark.skipif(
    get_gfx() != "gfx950" or not _mha_v4_sparse_co_available(),
    reason="gfx950 sorted-sparse MHA v4 code object is not deployed",
)


def _reference(q, k, v):
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (qf.shape[-1] ** -0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), vf).transpose(1, 2)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _run(recipe_name, tiles, heads=5, sequence_q=256, all_true=True):
    q_format, v_format, kwargs = SPARSE_RECIPES[recipe_name]
    sequence_k = tiles * mha_v4_kv_tile()
    torch.manual_seed(41)
    q = torch.randn((1, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.zeros(
        (1, heads, -(-sequence_q // 256), tiles), device="cuda", dtype=torch.bool
    )
    mask[:] = all_true
    if not all_true:
        mask[:, :, :, ::2] = True
    out = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs)
    torch.cuda.synchronize()
    return out, _reference(q, k, v), mask


@requires_sparse
@pytest.mark.parametrize("recipe_name", sorted(SPARSE_RECIPES))
def test_mha_v4_sparse_accuracy_holds_as_tile_count_grows(recipe_name):
    """A row's own small-tile-count accuracy is its baseline; growing the count must not erode it."""
    baseline_out, baseline_ref, _ = _run(recipe_name, BASELINE_TILES)
    baseline = _cosine(baseline_out, baseline_ref)
    assert torch.isfinite(
        baseline_out
    ).all(), f"{recipe_name}: non-finite at {BASELINE_TILES} tiles"
    assert baseline > ABSOLUTE_FLOOR, f"{recipe_name}: baseline cosine {baseline:.5f}"

    for tiles in GROWN_TILES:
        out, ref, _ = _run(recipe_name, tiles)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        cosine = _cosine(out, ref)
        assert (
            cosine > ABSOLUTE_FLOOR
        ), f"{recipe_name}: cosine {cosine:.5f} at {tiles} tiles"
        assert cosine > baseline - MAX_DEGRADATION, (
            f"{recipe_name}: accuracy degrades with tile count -- "
            f"{baseline:.5f} at {BASELINE_TILES} tiles, {cosine:.5f} at {tiles}"
        )


@requires_sparse
@pytest.mark.parametrize("recipe_name", sorted(SPARSE_RECIPES))
def test_mha_v4_sparse_skipping_lut_holds_as_tile_count_grows(recipe_name):
    """Same sweep with a LUT that actually skips, so the walk has to jump rather than run affine."""
    for tiles in (4, 8, 16, 32):
        out, _, mask = _run(recipe_name, tiles, all_true=False)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        assert mask.sum() > 0
        assert not bool(
            (out == 0).all()
        ), f"{recipe_name}: all-zero output at {tiles} tiles"


@requires_sparse
@pytest.mark.parametrize("recipe_name", ["i8fp8", "fp8", "mxfp8", "f6f8"])
def test_mha_v4_sparse_all_true_lut_matches_dense_bitwise(recipe_name):
    """An all-true LUT selects every tile, so the walk must reduce exactly to the dense one.

    Restricted to the rows whose V stays FP8. The MX-V rows resolve to a different V packing on
    their dense manifest row than on their sparse one, so they cannot be compared bitwise.
    """
    q_format, v_format, kwargs = SPARSE_RECIPES[recipe_name]
    for tiles in (2, 8, 16):
        sequence_k = tiles * mha_v4_kv_tile()
        torch.manual_seed(41)
        q = torch.randn((1, 256, 5, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, sequence_k, 5, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = torch.ones((1, 5, 1, tiles), device="cuda", dtype=torch.bool)
        sparse = mha_v4(
            q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs
        )
        dense = mha_v4(q, k, v, q_format, q_format, v_format, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(sparse, dense), (
            f"{recipe_name}: all-true LUT differs from dense at {tiles} tiles "
            f"(cosine {_cosine(sparse, dense):.7f})"
        )
