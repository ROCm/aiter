# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression test for the NaN / empty-split guards in ``mla_reduce_v1``.

``mla_reduce_v1`` (csrc/kernels/mla/reduce.cu) merges the per-split partials of
the persistent MLA split-KV decode. Without guards it trusts every partial
unconditionally -- it takes max/expf on the partial LSE, accumulates the partial
output, and (in ``_impl_simple``) divides by ``sum_e_lse`` -- so one poisoned
partial corrupts the entire reduce tile.

The reference reduction in ``test_mla_persistent.py`` (``torch_mla_reduce_v1``)
is defensive: NaN LSE -> -inf, NaN output -> 0, ``sum_e_lse <= 0`` -> 0. The
non-persistent decode path likewise never reads invalid splits
(``valid_split_count``). These tests drive ``mla_reduce_v1`` directly with
pathological partials and require it to agree with that reference.

Contract under test: the kernel must not introduce non-finite values that the
reference does not produce, and finite entries must match. Note ``+/-inf``
partials are deliberately propagated by the reference, so an equal non-finite
count there is correct behaviour, not a defect.

Run:  pytest -q op_tests/test_mla_reduce_nan_guard.py
"""

import math

import pytest
import torch

import aiter

torch.set_default_device("cuda")

# (nhead, v_head_dim, max_seqlen_q, num_tiles, splits_per_tile)
SHAPES = [
    (128, 512, 1, 8, 8),   # dp-attention decode, one query token
    (128, 512, 4, 8, 8),   # dp-attention decode with MTP (4 draft tokens)
    (16, 512, 1, 8, 32),   # tp8 decode, high split count
]
SHAPE_IDS = [f"nh{n}_dv{d}_msq{m}_tiles{t}_splits{s}" for n, d, m, t, s in SHAPES]


def torch_mla_reduce_v1(
    partial_output,      # [rows, nhead, dv] fp32
    partial_lse,         # [rows, nhead]     fp32
    reduce_indptr,       # [num_tile + 1]
    reduce_final_map,    # [num_tile, 2] or None
    reduce_partial_map,  # [reduce_indptr[-1]]
    max_seqlen_q,
    final_output,        # [total_q, nhead, dv]
    final_lse,           # [total_q, nhead] or None
):
    """Reference online-softmax merge (mirrors test_mla_persistent.py)."""
    device = partial_output.device
    dtype = partial_output.dtype
    num_reduce_tile = reduce_indptr.shape[0] - 1
    num_heads = partial_output.shape[1]
    head_dim = final_output.shape[2]

    for tile_idx in range(num_reduce_tile):
        start = reduce_indptr[tile_idx].item()
        end = reduce_indptr[tile_idx + 1].item()
        if start == end:
            continue
        num_splits = end - start
        tile_map = reduce_partial_map[start:end]

        if reduce_final_map is not None:
            q_start = reduce_final_map[tile_idx, 0].item()
            q_end = reduce_final_map[tile_idx, 1].item()
        else:
            q_start = tile_idx * max_seqlen_q
            q_end = (tile_idx + 1) * max_seqlen_q

        for seq_idx in range(q_start, q_end):
            for head_idx in range(num_heads):
                local = seq_idx - q_start
                lses, outs = [], []
                for s in range(num_splits):
                    loc = tile_map[s].item() + local
                    if loc < partial_lse.shape[0]:
                        v = partial_lse[loc, head_idx].item()
                        if math.isnan(v):
                            v = float("-inf")
                    else:
                        v = float("-inf")
                    if loc < partial_output.shape[0]:
                        o = partial_output[loc, head_idx, :].clone()
                        o = torch.where(torch.isnan(o), torch.zeros_like(o), o)
                    else:
                        o = torch.zeros(head_dim, dtype=dtype, device=device)
                    lses.append(v)
                    outs.append(o)

                max_lse = lses[0]
                reg_out = outs[0].clone()
                sum_e = 1.0
                for s in range(1, num_splits):
                    lse = lses[s]
                    new_max = max(max_lse, lse)
                    if new_max == float("-inf"):
                        old_scale = new_scale = 0.0
                    else:
                        old_scale = math.exp(max_lse - new_max)
                        new_scale = math.exp(lse - new_max)
                    reg_out = old_scale * reg_out + new_scale * outs[s]
                    max_lse = new_max
                    sum_e = sum_e * old_scale + new_scale

                if sum_e > 0 and not math.isnan(sum_e):
                    reg_out = reg_out / sum_e
                else:
                    reg_out = torch.zeros_like(reg_out)

                final_output[seq_idx, head_idx, :] = reg_out.to(final_output.dtype)
                if final_lse is not None:
                    if sum_e > 0 and not math.isnan(sum_e):
                        final_lse[seq_idx, head_idx] = max_lse + math.log(sum_e)
                    else:
                        final_lse[seq_idx, head_idx] = float("inf")


def build_problem(nhead, dv, msq, num_tile, splits):
    """Uniform reduce problem: `num_tile` tiles, `splits` partials each."""
    torch.manual_seed(20260729)

    reduce_indptr = torch.zeros(num_tile + 1, dtype=torch.int32)
    reduce_indptr[1:] = torch.cumsum(
        torch.full((num_tile,), splits, dtype=torch.int32), 0
    )
    total_partials = int(reduce_indptr[-1].item())

    # each partial tile occupies msq consecutive rows
    reduce_partial_map = torch.arange(total_partials, dtype=torch.int32) * msq
    rows = total_partials * msq

    reduce_final_map = torch.zeros((num_tile, 2), dtype=torch.int32)
    for t in range(num_tile):
        reduce_final_map[t, 0] = t * msq
        reduce_final_map[t, 1] = (t + 1) * msq

    return dict(
        partial_output=torch.randn(rows, nhead, dv, dtype=torch.float32),
        # plausible logsumexp range
        partial_lse=torch.rand(rows, nhead, dtype=torch.float32) * 20.0 - 5.0,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        rows=rows,
        total_q=num_tile * msq,
        nhead=nhead,
        dv=dv,
        msq=msq,
        splits=splits,
    )


def check_against_reference(p, partial_output, partial_lse):
    """Run kernel + reference on the given partials and assert they agree."""
    nhead, dv, msq, total_q = p["nhead"], p["dv"], p["msq"], p["total_q"]

    out_ker = torch.zeros(total_q, nhead, dv, dtype=torch.bfloat16)
    lse_ker = torch.zeros(total_q, nhead, dtype=torch.float32)
    aiter.mla_reduce_v1(
        partial_output.view(p["rows"], 1, nhead, dv),
        partial_lse.view(p["rows"], 1, nhead, 1),
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        msq,
        p["splits"],
        out_ker,
        lse_ker,
    )
    torch.cuda.synchronize()

    out_ref = torch.zeros(total_q, nhead, dv, dtype=torch.bfloat16)
    lse_ref = torch.zeros(total_q, nhead, dtype=torch.float32)
    torch_mla_reduce_v1(
        partial_output,
        partial_lse,
        p["reduce_indptr"],
        p["reduce_final_map"],
        p["reduce_partial_map"],
        msq,
        out_ref,
        lse_ref,
    )

    a = out_ker.float()
    b = out_ref.float()
    fin_a = torch.isfinite(a)
    fin_b = torch.isfinite(b)

    extra = int((~fin_a & fin_b).sum().item())
    assert extra == 0, (
        f"kernel produced {extra} non-finite outputs where the reference is "
        f"finite (a poisoned partial leaked into the reduce tile)"
    )
    assert bool((fin_a == fin_b).all().item()), (
        "kernel and reference disagree on which outputs are finite"
    )

    both = fin_a & fin_b
    if both.any():
        max_abs = float((a[both] - b[both]).abs().max().item())
        assert max_abs < 2e-2, f"finite outputs differ: max_abs_diff={max_abs:.4g}"


@pytest.mark.parametrize("nhead,dv,msq,num_tile,splits", SHAPES, ids=SHAPE_IDS)
def test_all_finite_partials(nhead, dv, msq, num_tile, splits):
    """Sanity: with well-formed partials the guards must be a no-op."""
    p = build_problem(nhead, dv, msq, num_tile, splits)
    check_against_reference(p, p["partial_output"], p["partial_lse"])


@pytest.mark.parametrize("nhead,dv,msq,num_tile,splits", SHAPES, ids=SHAPE_IDS)
def test_nan_in_partial_output(nhead, dv, msq, num_tile, splits):
    """A NaN partial output must not leak into the reduced tile.

    Without the guards this is the failing case: the kernel emits NaN for every
    element of the affected tile while the reference (which zeroes NaN partials)
    stays finite.
    """
    p = build_problem(nhead, dv, msq, num_tile, splits)
    po = p["partial_output"].clone()
    loc = int(p["reduce_partial_map"][1].item())
    po[loc, :, : dv // 4] = float("nan")
    check_against_reference(p, po, p["partial_lse"])


@pytest.mark.parametrize("nhead,dv,msq,num_tile,splits", SHAPES, ids=SHAPE_IDS)
def test_nan_in_partial_lse(nhead, dv, msq, num_tile, splits):
    """A NaN partial LSE must be treated as an empty split (weight 0)."""
    p = build_problem(nhead, dv, msq, num_tile, splits)
    pl = p["partial_lse"].clone()
    loc = int(p["reduce_partial_map"][1].item())
    pl[loc, :] = float("nan")
    check_against_reference(p, p["partial_output"], pl)


@pytest.mark.parametrize("nhead,dv,msq,num_tile,splits", SHAPES, ids=SHAPE_IDS)
def test_empty_tile_all_splits_masked(nhead, dv, msq, num_tile, splits):
    """Every split of a tile empty (-inf LSE) => sum_e_lse == 0.

    The unguarded kernel divides by zero here; the reference returns 0.
    """
    p = build_problem(nhead, dv, msq, num_tile, splits)
    pl = p["partial_lse"].clone()
    s = int(p["reduce_indptr"][1].item())
    e = int(p["reduce_indptr"][2].item())
    for k in range(s, e):
        loc = int(p["reduce_partial_map"][k].item())
        pl[loc : loc + msq, :] = float("-inf")
    check_against_reference(p, p["partial_output"], pl)


@pytest.mark.parametrize("nhead,dv,msq,num_tile,splits", SHAPES, ids=SHAPE_IDS)
def test_huge_partial_lse(nhead, dv, msq, num_tile, splits):
    """Partial LSE close to the fp32 limit must not blow up the merge."""
    p = build_problem(nhead, dv, msq, num_tile, splits)
    pl = p["partial_lse"].clone()
    loc = int(p["reduce_partial_map"][2].item())
    pl[loc, :] = 3.4e38
    check_against_reference(p, p["partial_output"], pl)


@pytest.mark.parametrize("nhead,dv,msq,num_tile,splits", SHAPES, ids=SHAPE_IDS)
def test_inf_in_partial_output(nhead, dv, msq, num_tile, splits):
    """+inf partial output: kernel must match the reference's propagation."""
    p = build_problem(nhead, dv, msq, num_tile, splits)
    po = p["partial_output"].clone()
    loc = int(p["reduce_partial_map"][3].item())
    po[loc, :, :16] = float("inf")
    check_against_reference(p, po, p["partial_lse"])
