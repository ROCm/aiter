# SPDX-License-Identifier: MIT
"""Fused TP route/psum must drop expert -1 pad routes like the split-kernel route."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    _tdm_align_up,
    fused_route_psum_remap,
)
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3), (513, 11), (1024, 8)])
@pytest.mark.parametrize("token_num", [1, 5, 26, 64, 186, 341])
@pytest.mark.parametrize("pad_frac", [0.0, 0.5, 1.0])
def test_route_psum_fused_pad_routes(experts, topk, token_num, pad_frac):
    torch.manual_seed(token_num)
    align = 64
    if token_num * topk > 2048:
        pytest.skip("beyond the fused route's single-workgroup bound")
    ids = torch.stack([torch.randperm(experts)[:topk] for _ in range(token_num)])
    n_pad = int(token_num * pad_frac)
    if n_pad:
        # The router leaves unrouted pad rows at expert -1.
        ids[token_num - n_pad :] = -1
    ids = ids.to("cuda", torch.int32)
    numel = token_num * topk
    max_m = max(align, _tdm_align_up(numel, align))
    contiguous_m = max(align, _tdm_align_up(numel + experts * align - topk, align))

    masked_m, rows, psum = fused_route_psum_remap(ids, experts, max_m, align)
    torch.cuda.synchronize()

    ids = ids.reshape(-1).cpu().long()
    rows = rows.reshape(-1).cpu().long()
    kept = ids >= 0
    counts = torch.bincount(ids[kept], minlength=experts)
    starts = torch.cumsum((counts + align - 1) // align * align, 0) - (
        (counts + align - 1) // align * align
    )

    assert (rows[~kept] == DROPPED_ROUTE_ROW).all()
    assert torch.equal(masked_m.cpu().long(), counts)
    assert torch.equal(psum.cpu().long(), starts + counts)
    if kept.any():
        kept_rows = rows[kept]
        base = starts[ids[kept]]
        assert ((kept_rows >= base) & (kept_rows < base + counts[ids[kept]])).all()
        assert kept_rows.unique().numel() == kept_rows.numel()
        assert int(kept_rows.max()) < contiguous_m


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3), (513, 6), (1024, 6)])
@pytest.mark.parametrize("token_num", [1, 8])
def test_route_psum_fused_mixed_pad_and_out_of_range(experts, topk, token_num):
    """One row can mix a valid id, expert -1, and an id >= E."""
    align = 64
    pattern = torch.tensor(
        [0, -1, experts, experts - 1, experts + 5, 2], dtype=torch.int32
    )
    ids = pattern[:topk].repeat(token_num).reshape(token_num, topk)
    ids = ids.to("cuda")
    numel = token_num * topk
    max_m = max(align, _tdm_align_up(numel, align))

    masked_m, rows, psum = fused_route_psum_remap(ids, experts, max_m, align)
    torch.cuda.synchronize()

    ids = ids.reshape(-1).cpu().long()
    rows = rows.reshape(-1).cpu().long()
    kept = (ids >= 0) & (ids < experts)
    counts = torch.bincount(ids[kept], minlength=experts)
    starts = torch.cumsum((counts + align - 1) // align * align, 0) - (
        (counts + align - 1) // align * align
    )

    assert (rows[~kept] == DROPPED_ROUTE_ROW).all()
    assert torch.equal(masked_m.cpu().long(), counts)
    assert torch.equal(psum.cpu().long(), starts + counts)
    kept_rows = rows[kept]
    base = starts[ids[kept]]
    assert ((kept_rows >= base) & (kept_rows < base + counts[ids[kept]])).all()
    assert kept_rows.unique().numel() == kept_rows.numel()


def _check_routes(ids, masked_m, rows, psum, experts, align):
    ids = ids.reshape(-1).cpu().long()
    rows = rows.reshape(-1).cpu().long()
    kept = (ids >= 0) & (ids < experts)
    counts = torch.bincount(ids[kept], minlength=experts)
    padded = (counts + align - 1) // align * align
    starts = torch.cumsum(padded, 0) - padded
    assert (rows[~kept] == DROPPED_ROUTE_ROW).all()
    assert torch.equal(masked_m.cpu().long(), counts)
    assert torch.equal(psum.cpu().long(), starts + counts)
    kept_rows = rows[kept]
    base = starts[ids[kept]]
    assert ((kept_rows >= base) & (kept_rows < base + counts[ids[kept]])).all()
    assert kept_rows.unique().numel() == kept_rows.numel()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)
@pytest.mark.parametrize("experts,topk", [(513, 11), (384, 6)])
@pytest.mark.parametrize("token_num", [4, 40, 128])
def test_route_psum_fused_graph_replay(experts, topk, token_num):
    """A captured launch stays correct as the ids under it change, pad rows included."""
    align = 16
    numel = token_num * topk
    max_m = max(align, _tdm_align_up(numel, align))
    ids = torch.zeros((token_num, topk), dtype=torch.int32, device="cuda")

    fused_route_psum_remap(ids, experts, max_m, align)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        masked_m, rows, psum = fused_route_psum_remap(ids, experts, max_m, align)

    gen = torch.Generator().manual_seed(experts * 1000 + token_num)
    for step in range(8):
        new = torch.stack(
            [torch.randperm(experts, generator=gen)[:topk] for _ in range(token_num)]
        ).to(torch.int32)
        if step % 2:
            # Graph-padded rows: unrouted (-1) or left with ids past E.
            n_pad = (step * token_num) // 8
            new[token_num - n_pad :] = -1 if step % 4 == 1 else experts + step
        ids.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        _check_routes(new, masked_m, rows, psum, experts, align)
