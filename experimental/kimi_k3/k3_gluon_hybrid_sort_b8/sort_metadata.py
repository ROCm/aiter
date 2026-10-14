"""AITER BM16 metadata for eight tokens, each with 16 distinct experts."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from .coherent_memory import coherent_load


@g.jit
def prepare_routes(
    IDs,
    Weights,
    SortIDs,
    SortWeights,
    SortExperts,
    SortCounts,
    SortM,
    SortReverse,
    B: gl.constexpr,
):
    gl.static_assert(B == 8)
    flat: gl.constexpr = gl.BlockedLayout([1], [64], [8], [0])
    route = gl.arange(0, 128, layout=flat)
    ids = coherent_load(IDs + route)
    weights = coherent_load(Weights + route)
    # Stable within-expert row numbers. Uniqueness within each token bounds
    # these at eight, so no expert can require a second BM16 block.
    pairs: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [8, 1], [1, 0])
    ir = gl.arange(0, 128, layout=gl.SliceLayout(1, pairs))
    jr = gl.arange(0, 128, layout=gl.SliceLayout(0, pairs))
    ei = gl.convert_layout(ids, gl.SliceLayout(1, pairs))
    ej = gl.convert_layout(ids, gl.SliceLayout(0, pairs))
    same = ei[:, None] == ej[None, :]
    row = gl.sum((same & (jr[None, :] < ir[:, None])).to(gl.int32), 1)
    count = gl.sum(same.to(gl.int32), 1)
    first = row == 0
    first_j = gl.convert_layout(first, gl.SliceLayout(0, pairs))
    # Count only the first occurrence of every smaller expert. This orders
    # blocks by expert ID without a 1024-bin histogram or cross-warp scan.
    block = gl.sum(((ej[None, :] < ei[:, None]) & first_j[None, :]).to(gl.int32), 1)
    num_blocks = gl.sum(first.to(gl.int32), 0)
    pad = gl.arange(0, 16, layout=gl.SliceLayout(0, pairs))
    padding = first[:, None] & (pad[None, :] >= count[:, None])
    padding_offset = block[:, None] * 16 + pad[None, :]
    # Padding and valid rows are disjoint; no clear-then-scatter barrier.
    gl.store(SortIDs + padding_offset, B, padding)
    gl.store(SortWeights + padding_offset, 0.0, padding)
    gl.store(SortM + padding_offset, B, padding)
    gl.store(SortExperts + block, ei, first)
    row = gl.convert_layout(row, flat)
    block = gl.convert_layout(block, flat)
    destination = block * 16 + row
    gl.store(SortCounts, num_blocks * 16)
    gl.store(SortCounts + 1, B)
    gl.store(SortIDs + destination, (route // 16) | ((route % 16) << 24))
    gl.store(SortWeights + destination, weights)
    gl.store(SortM + destination, route // 16)
    gl.store(SortReverse + route, destination)
