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
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
@pytest.mark.parametrize("token_num", [1, 5, 26, 64, 341, 682])
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
@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
@pytest.mark.parametrize("token_num", [1, 8])
def test_route_psum_fused_mixed_pad_and_out_of_range(experts, topk, token_num):
    """One row can mix a valid id, expert -1, and an id >= E."""
    align = 64
    pattern = torch.tensor([0, -1, experts, 1, experts + 5, 2], dtype=torch.int32)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)
@pytest.mark.parametrize("experts,topk,shared", [(128, 4, False), (129, 5, True)])
@pytest.mark.parametrize("token_num", [1, 5, 26, 64, 341, 819])
@pytest.mark.parametrize("pad_frac", [0.0, 0.5, 1.0])
def test_route_quant_front_matches_split_kernels(
    experts, topk, shared, token_num, pad_frac
):
    """Route + stage-1 quant in one launch: the route table of the fused route
    (rows in token order within an expert) and the bytes the quant kernels
    write for those rows. With ``shared`` the last column is the shared expert:
    rows in token order, activations MXFP8 in per-token buffers."""
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_moe_fused_quant_preshuffle,
        flydsl_moe_route_quant_front,
    )

    torch.manual_seed(token_num)
    align, wmma_rep, model_dim = 32, 2, 512
    routed = experts - 1 if shared else experts
    ids = torch.stack(
        [
            torch.randperm(routed, device="cpu")[: topk - int(shared)]
            for _ in range(token_num)
        ]
    )
    if shared:
        ids = torch.cat(
            [ids, torch.full((token_num, 1), experts - 1, device="cpu")], dim=1
        )
    n_pad = int(token_num * pad_frac)
    if n_pad:
        # The router leaves unrouted pad rows at expert -1.
        ids[token_num - n_pad :] = -1
    ids = ids.to("cuda", torch.int32)
    hidden = torch.randn(token_num, model_dim, device="cuda", dtype=torch.bfloat16)
    numel = token_num * topk
    contiguous_m = _tdm_align_up(numel + experts * align, align)
    shared_rows = _tdm_align_up(token_num, 128) if shared else 0

    masked_m, rows, psum, payload, scale, sh_payload, sh_scale = (
        flydsl_moe_route_quant_front(
            hidden,
            ids,
            experts,
            contiguous_m,
            align,
            wmma_rep=wmma_rep,
            shared_rows=shared_rows,
        )
    )
    torch.cuda.synchronize()

    flat_ids = ids.reshape(-1).cpu().long()
    flat_rows = rows.reshape(-1).cpu().long()
    kept = flat_ids >= 0
    counts = torch.bincount(flat_ids[kept], minlength=experts)
    padded = (counts + align - 1) // align * align
    starts = torch.cumsum(padded, 0) - padded
    assert (flat_rows[~kept] == DROPPED_ROUTE_ROW).all()
    assert torch.equal(masked_m.cpu().long(), counts)
    assert torch.equal(psum.cpu().long(), starts + counts)
    # Slot = the number of lower tokens routed to the same expert.
    seen = torch.zeros(experts, dtype=torch.long, device="cpu")
    expect = torch.full_like(flat_rows, DROPPED_ROUTE_ROW)
    for route in kept.nonzero().flatten().tolist():
        expert = int(flat_ids[route])
        expect[route] = starts[expert] + seen[expert]
        seen[expert] += 1
    assert torch.equal(flat_rows, expect)

    # The split quant kernels, run over the front kernel's own buffers, must
    # leave them as they are.
    quant_rows = rows.clone()
    if shared:
        quant_rows[:, -1] = DROPPED_ROUTE_ROW
    ref_payload, ref_scale = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, token_num, model_dim),
        1,
        contiguous_m,
        wmma_rep=wmma_rep,
        quant_mode="fp4",
        topids_to_rows=quant_rows,
        source_topk=topk,
        out_payload=payload.clone(),
        out_scale=scale.clone(),
    )
    torch.cuda.synchronize()
    assert torch.equal(ref_payload, payload)
    assert torch.equal(ref_scale, scale)
    if shared:
        token_rows = torch.arange(token_num, dtype=torch.int32, device="cuda")
        token_rows[token_num - n_pad :] = DROPPED_ROUTE_ROW
        ref_payload, ref_scale = flydsl_moe_fused_quant_preshuffle(
            hidden.reshape(1, token_num, model_dim),
            1,
            shared_rows,
            wmma_rep=wmma_rep,
            quant_mode="fp8",
            topids_to_rows=token_rows,
            source_topk=1,
            out_payload=sh_payload.clone(),
            out_scale=sh_scale.clone(),
        )
        torch.cuda.synchronize()
        assert torch.equal(ref_payload, sh_payload)
        assert torch.equal(ref_scale, sh_scale)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="Requires gfx1250",
)
def test_route_quant_front_duplicate_and_out_of_range_ids():
    """A token routed twice to one expert takes one row for both routes; an
    id at or past E, like expert -1, takes none."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_route_quant_front

    experts, align, token_num = 64, 16, 9
    pattern = torch.tensor(
        [3, -1, experts, 3, experts + 5, 7], dtype=torch.int32, device="cpu"
    )
    ids = pattern.repeat(token_num).reshape(token_num, -1).to("cuda")
    hidden = torch.randn(token_num, 256, device="cuda", dtype=torch.bfloat16)
    masked_m, rows, psum, *_ = flydsl_moe_route_quant_front(
        hidden, ids, experts, 256, align, wmma_rep=1
    )
    torch.cuda.synchronize()

    rows = rows.cpu().long()
    counts = torch.zeros(experts, dtype=torch.long, device="cpu")
    counts[3] = counts[7] = token_num
    padded = (counts + align - 1) // align * align
    starts = torch.cumsum(padded, 0) - padded
    token = torch.arange(token_num, device="cpu")
    assert torch.equal(masked_m.cpu().long(), counts)
    assert torch.equal(psum.cpu().long(), starts + counts)
    assert (rows[:, [1, 2, 4]] == DROPPED_ROUTE_ROW).all()
    assert torch.equal(rows[:, 0], starts[3] + token)
    assert torch.equal(rows[:, 3], starts[3] + token)
    assert torch.equal(rows[:, 5], starts[7] + token)
