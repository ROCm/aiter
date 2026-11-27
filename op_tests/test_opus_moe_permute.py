# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE token permute / unpermute (aiter.ops.opus.moe_permute): plan, forward and backward.

The torch reference follows the numeric rules of the kernels (see the module docstring of
aiter/ops/opus/moe_permute.py), so every kernel output is required to match it exactly (err == 0).
The reference backward is also checked against autograd of a float64 forward, with a tolerance:
autograd adds in its own order.
"""

import argparse
import itertools
from collections import namedtuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus import moe_permute as mp
from aiter.ops.opus.moe_permute import (
    DW_THREADS,
    DW_VEC,
    GROUP_ALIGN,
    moe_permute,
    moe_permute_plan,
    moe_permute_rows,
    moe_unpermute,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# The reference's own plan: the row of every token, written the direct way.
RefPlan = namedtuple("RefPlan", "row_of_token token_of_row grouped_layout group_rows")

SUPPORTED_GFX = ["gfx950"]
DISTS = ["uniform", "skewed", "empty", "one"]
# stages of the forward / backward chain; the last two are moe_unpermute without weights
STAGES = [
    "permute_fwd",
    "permute_bwd",
    "unpermute_fwd",
    "unpermute_bwd",
    "unpermute_fwd_noweight",
    "unpermute_bwd_noweight",
]

GRID_N = [1, 2, 31, 255, 256, 257, 1023, 1024, 1025, 4097, 50000, 69632, 70001, 139264]
GRID_E = [1, 2, 31, 32, 33, 64, 255, 256]
GRID_H = [
    8,
    16,
    64,
    1000,
    2048,
    2056,
    4104,
    6144,
    8192,
]  # below, at and above one chunk per thread (H = 2048)


# ---- inputs ------------------------------------------------------------------


def make_expert_ids(n, experts, dist, seed=0):
    """expert_ids [N] int32.
    uniform: every expert equally likely; skewed: expert e with weight 1 / (e + 1);
    empty: the odd experts get no token; one: every token goes to one expert;
    invalid: uniform over -3 .. E + 4 plus some INT_MAX / INT_MIN (ids that are not experts; plan only).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "uniform":
        ids = torch.randint(0, experts, (n,), generator=g)
    elif dist == "skewed":
        p = 1.0 / torch.arange(1, experts + 1, dtype=torch.float32)
        ids = torch.multinomial(p, n, replacement=True, generator=g)
    elif dist == "empty":
        ids = torch.randint(0, (experts + 1) // 2, (n,), generator=g) * 2
    elif dist == "one":
        ids = torch.full((n,), experts // 2)
    elif dist == "invalid":
        ids = torch.randint(-3, experts + 5, (n,), generator=g)
        ids[torch.rand(n, generator=g) < 0.02] = 2**31 - 1
        ids[torch.rand(n, generator=g) < 0.02] = -(2**31)
    else:
        raise ValueError(dist)
    return ids.to(torch.int32)


def make_rows(rows, hidden):
    return torch.randn(rows, hidden).to(dtypes.bf16)


# ---- torch reference (not timed) -------------------------------------------


def run_torch_plan(expert_ids, num_experts, table_rows=None):
    """table_rows: rows of the [R] tables; None: moe_permute_rows(N, E), the tables of exact=False."""
    ids = expert_ids.flatten().long()
    n = ids.numel()
    tok = torch.nonzero(
        (ids >= 0) & (ids < num_experts)
    ).flatten()  # the tokens that get a row
    e = ids[tok]
    order = torch.sort(e, stable=True).indices  # by expert, then token
    cnt = torch.bincount(e, minlength=num_experts)
    padded = (cnt + GROUP_ALIGN - 1) // GROUP_ALIGN * GROUP_ALIGN
    start = padded.cumsum(0) - padded
    e_sorted = e[order]
    rows = start[e_sorted] + torch.arange(e.numel()) - (cnt.cumsum(0) - cnt)[e_sorted]
    n_rows = moe_permute_rows(n, num_experts) if table_rows is None else table_rows
    row_of_token = torch.full((n,), -1, dtype=torch.long)
    row_of_token[tok[order]] = rows
    token_of_row = torch.full((n_rows,), -1, dtype=torch.long)
    token_of_row[rows] = tok[order]
    grouped_layout = torch.full((n_rows,), -1, dtype=torch.long)
    grouped_layout[rows] = e_sorted
    return RefPlan(
        row_of_token.int(), token_of_row.int(), grouped_layout.int(), cnt.int()
    )


def _scatter_rows(src, plan, scale=None):
    """[R, H]: row r = src[token of r] (times scale[token]), padding rows 0."""
    tok = plan.token_of_row.long()
    valid = tok >= 0
    dst = torch.zeros((tok.numel(), src.shape[1]), dtype=src.dtype)
    if scale is None:
        dst[valid] = src[tok[valid]]
    else:  # the product is an fp32 tensor, rounded to bf16 once
        dst[valid] = (
            scale.flatten()[tok[valid]][:, None] * src[tok[valid]].to(dtypes.fp32)
        ).to(src.dtype)
    return dst


def run_torch_permute_fwd(x, plan):
    return _scatter_rows(x, plan)


def run_torch_permute_bwd(d_perm, plan):
    return d_perm[plan.row_of_token.long()]


def run_torch_unpermute_fwd(y_perm, plan, weights=None):
    rows = y_perm[plan.row_of_token.long()]
    if weights is None:
        return rows
    return (weights.flatten()[:, None] * rows.to(dtypes.fp32)).to(y_perm.dtype)


def run_torch_unpermute_bwd(d_out, y_perm, plan, weights=None, batch=1024):
    """(d_perm, d_weights); weights None: d_perm only (the copy)."""
    if weights is None:
        return _scatter_rows(d_out, plan)
    d_perm = _scatter_rows(d_out, plan, weights)

    # d_weights: thread i of DW_THREADS adds the products of chunks i, i + DW_THREADS, ... (DW_VEC
    # elements each) in ascending element order; the partial sums are added in ascending thread order.
    n, hidden = d_out.shape
    chunks = hidden // DW_VEC
    rounds = -(-chunks // DW_THREADS)
    d_weights = torch.empty(n, dtype=dtypes.fp32)
    for t0 in range(0, n, batch):
        t1 = min(t0 + batch, n)
        prod = d_out[t0:t1].to(dtypes.fp32) * y_perm[
            plan.row_of_token[t0:t1].long()
        ].to(dtypes.fp32)
        prod = prod.view(t1 - t0, chunks, DW_VEC)
        partial = None
        for j in range(rounds):
            blk = prod[:, j * DW_THREADS : (j + 1) * DW_THREADS]
            m = blk.shape[1]  # the last round may reach fewer threads
            for e in range(DW_VEC):
                if partial is None:
                    partial = blk[..., e].clone()
                else:
                    partial[:, :m] = partial[:, :m] + blk[..., e]
        acc = partial[:, 0]
        for i in range(1, partial.shape[1]):
            acc = acc + partial[:, i]
        d_weights[t0:t1] = acc
    return d_perm, d_weights.view(weights.shape)


def check_plan(plan, expert_ids, num_experts):
    """Properties every valid plan has, whatever produced it."""
    ids = expert_ids.flatten().long()
    n = ids.numel()
    rows = plan.row_of_token.long()
    tok = plan.token_of_row.long()
    layout = plan.grouped_layout.long()
    has_row = (ids >= 0) & (ids < num_experts)
    cnt = torch.bincount(ids[has_row], minlength=num_experts)
    assert tok.numel() == layout.numel() <= moe_permute_rows(n, num_experts)
    assert torch.equal(plan.group_rows.long(), cnt)
    assert torch.equal(rows >= 0, has_row)  # an id that is not an expert gets no row
    assert torch.equal(
        tok[rows[has_row]], torch.nonzero(has_row).flatten()
    )  # a bijection onto the used rows
    assert int((tok >= 0).sum()) == int(has_row.sum())
    assert torch.equal(layout[rows[has_row]], ids[has_row])
    assert torch.equal(layout >= 0, tok >= 0)
    padded = (cnt + GROUP_ALIGN - 1) // GROUP_ALIGN * GROUP_ALIGN
    start = padded.cumsum(0) - padded
    used = torch.nonzero(layout >= 0).flatten()
    assert bool(
        (used - start[layout[used]] < cnt[layout[used]]).all()
    )  # packed at the group's first row
    assert bool(
        (tok[used][1:] > tok[used][:-1])[layout[used][1:] == layout[used][:-1]].all()
    )  # token order


def check_kernel_plan(got, ref, tail=True):
    """The plan kernels against the reference plan: every field, every element (padding rows included).
    tail False (ids that are not experts, outside the precondition of the op): the [R] tables are compared up
    to the rows in use only. Such ids take no row, so the tail is longer than the 128 E rows the plan fills,
    and the rows behind that are not written (the same at 84371a7f2)."""
    # rows in use: every group with its padding; the rows from there on are the tail
    used = int(
        ((ref.group_rows.long() + GROUP_ALIGN - 1) // GROUP_ALIGN * GROUP_ALIGN).sum()
    )
    end = None if tail else used
    assert got.reverse_sorted.shape == ref.row_of_token.shape and torch.equal(
        got.reverse_sorted, ref.row_of_token
    ), "reverse_sorted"
    assert (
        got.sorted_token_ids.shape == ref.token_of_row.shape
        and got.grouped_layout.shape == ref.grouped_layout.shape
    )
    assert torch.equal(
        got.sorted_token_ids[:end], ref.token_of_row[:end]
    ), "sorted_token_ids"
    assert torch.equal(
        got.grouped_layout[:end], ref.grouped_layout[:end]
    ), "grouped_layout"
    assert torch.equal(got.group_rows, ref.group_rows), "group_rows"
    assert (
        got.num_valid_ids.shape == (1,) and got.num_valid_ids.dtype == torch.int32
    ), "num_valid_ids"
    assert int(got.num_valid_ids) == used, "num_valid_ids"
    for f in got:
        assert f.dtype == torch.int32


def make_reference(x, y_perm, d_perm, d_out, plan, weights):
    return {
        "permute_fwd": run_torch_permute_fwd(x, plan),
        "permute_bwd": run_torch_permute_bwd(d_perm, plan),
        "unpermute_fwd": run_torch_unpermute_fwd(y_perm, plan, weights),
        "unpermute_bwd": run_torch_unpermute_bwd(d_out, y_perm, plan, weights),
        "unpermute_fwd_noweight": run_torch_unpermute_fwd(y_perm, plan),
        "unpermute_bwd_noweight": run_torch_unpermute_bwd(d_out, y_perm, plan),
    }


def check_grad_parity(x, y_perm, d_perm, d_out, plan, weights, ref):
    """Reference backward vs autograd of a float64 forward (tolerance: different summation order)."""
    tok = plan.token_of_row.long()
    valid = tok >= 0
    rows = plan.row_of_token.long()
    xd = x.double().requires_grad_()
    x_perm = torch.zeros((tok.numel(), x.shape[1]), dtype=torch.float64)
    x_perm[valid] = xd[tok[valid]]
    (dx,) = torch.autograd.grad(x_perm, xd, d_perm.double())
    yd, wd = y_perm.double().requires_grad_(), weights.double().requires_grad_()
    dy, dw = torch.autograd.grad(wd[:, None] * yd[rows], (yd, wd), d_out.double())
    yd = y_perm.double().requires_grad_()
    (dy_noweight,) = torch.autograd.grad(yd[rows], yd, d_out.double())
    errs = {
        "dx": checkAllclose(dx.float(), ref["permute_bwd"].float(), msg="autograd dx"),
        "d_perm": checkAllclose(
            dy.float(), ref["unpermute_bwd"][0].float(), msg="autograd d_perm"
        ),
        # H terms of mixed sign: compare against the magnitude of the largest sum
        "d_weights": checkAllclose(
            dw.float(),
            ref["unpermute_bwd"][1],
            rtol=1e-3,
            atol=1e-3 * float(dw.abs().max()),
            msg="autograd d_weights",
        ),
        "d_perm noweight": checkAllclose(
            dy_noweight.float(),
            ref["unpermute_bwd_noweight"].float(),
            msg="autograd d_perm (no weights)",
        ),
    }
    assert all(e == 0 for e in errs.values()), errs


# ---- candidates --------------------------------------------------------------


def _torch_index_candidates(x, y_perm, d_perm, d_out, plan, weights):
    """Baseline: plain torch indexing (index_select). Compared with a tolerance."""
    n = x.shape[0]
    tok = plan.token_of_row.long()
    token_of_row = torch.where(
        tok >= 0, tok, n
    )  # padding rows read an appended zero row
    rows = plan.row_of_token.long()
    w_row = torch.cat([weights, torch.zeros(1)])[token_of_row][:, None]
    x_pad = torch.cat([x, torch.zeros_like(x[:1])])
    d_out_pad = torch.cat([d_out, torch.zeros_like(d_out[:1])])
    return {
        "permute_fwd": lambda: x_pad.index_select(0, token_of_row),
        "permute_bwd": lambda: d_perm.index_select(0, rows),
        "unpermute_fwd": lambda: (y_perm.index_select(0, rows) * weights[:, None]).to(
            y_perm.dtype
        ),
        "unpermute_bwd": lambda: (
            (d_out_pad.index_select(0, token_of_row) * w_row).to(y_perm.dtype),
            torch.einsum(
                "nh,nh->n", d_out.float(), y_perm.index_select(0, rows).float()
            ),
        ),
        "unpermute_fwd_noweight": lambda: y_perm.index_select(0, rows),
        "unpermute_bwd_noweight": lambda: d_out_pad.index_select(0, token_of_row),
    }


def _opus_candidates(x, y_perm, d_perm, d_out, plan, weights):
    """The kernels K2 .. K5, launched directly into preallocated outputs (what the autograd.Functions call).
    plan: a MoEPermutePlan. moe_unpermute without weights is the copy kernels of K3 (forward) / K2 (backward).
    """
    x_perm, dx, out = torch.empty_like(y_perm), torch.empty_like(x), torch.empty_like(x)
    dp, dw = torch.empty_like(y_perm), torch.empty_like(weights)

    def run(launch, *args, ret):
        launch(*args)
        return ret

    return {
        "permute_fwd": lambda: run(
            mp._permute_fwd_launch, x, plan.sorted_token_ids, x_perm, ret=x_perm
        ),
        "permute_bwd": lambda: run(
            mp._permute_bwd_launch, d_perm, plan.reverse_sorted, dx, ret=dx
        ),
        "unpermute_fwd": lambda: run(
            mp._unpermute_fwd_launch, y_perm, plan.reverse_sorted, weights, out, ret=out
        ),
        "unpermute_bwd": lambda: run(
            mp._unpermute_bwd_launch,
            d_out,
            y_perm,
            weights,
            plan.sorted_token_ids,
            dp,
            dw,
            ret=(dp, dw),
        ),
        "unpermute_fwd_noweight": lambda: run(
            mp._permute_bwd_launch, y_perm, plan.reverse_sorted, out, ret=out
        ),
        "unpermute_bwd_noweight": lambda: run(
            mp._permute_fwd_launch, d_out, plan.sorted_token_ids, dp, ret=dp
        ),
    }


def _same_bits(a, b):
    """The same bytes (K2 / K3 and the unweighted moe_unpermute are copies)."""
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.view(torch.int16), b.view(torch.int16))
    )


def _exact(a, b):
    """Same values everywhere (0.0 == -0.0); no tolerance."""
    return a.shape == b.shape and bool((a.to(dtypes.fp32) == b.to(dtypes.fp32)).all())


COPIES = (
    "permute_fwd",
    "permute_bwd",
    "unpermute_fwd_noweight",
    "unpermute_bwd_noweight",
)


def check_stage(stage, got, ref, what):
    pairs = zip(got, ref) if isinstance(ref, tuple) else [(got, ref)]
    same = _same_bits if stage in COPIES else _exact
    assert all(
        same(a, b) for a, b in pairs
    ), f"{what}: {stage} differs from the reference"


def check_public_api(x, y_perm, d_perm, d_out, expert_ids, experts, weights, plan, ref):
    """moe_permute / moe_unpermute and their autograd backward against the reference, with ids / weights
    given as [N] and as [N, 1], and moe_unpermute without weights."""
    for column in (False, True):
        ids = expert_ids.view(-1, 1) if column else expert_ids
        w = weights.view(-1, 1) if column else weights
        xr = x.detach().requires_grad_()
        x_perm, got_plan = moe_permute(xr, ids, experts, exact=False)
        check_kernel_plan(got_plan, plan)
        (dx,) = torch.autograd.grad(x_perm, xr, d_perm)
        yr, wr = y_perm.detach().requires_grad_(), w.detach().requires_grad_()
        out = moe_unpermute(yr, got_plan, wr)
        dp, dw = torch.autograd.grad(out, (yr, wr), d_out)
        assert dw.shape == w.shape, "d_weights has the shape of weights"
        yr = y_perm.detach().requires_grad_()
        out_noweight = moe_unpermute(yr, got_plan)
        (dp_noweight,) = torch.autograd.grad(out_noweight, yr, d_out)
        got = {
            "permute_fwd": x_perm,
            "permute_bwd": dx,
            "unpermute_fwd": out,
            "unpermute_bwd": (dp, dw.view(weights.shape)),
            "unpermute_fwd_noweight": out_noweight,
            "unpermute_bwd_noweight": dp_noweight,
        }
        for stage in STAGES:
            check_stage(
                stage,
                got[stage],
                ref[stage],
                "moe_permute API" + (" ([N, 1])" if column else ""),
            )
    # a plan computed before is reused; the weights argument of moe_permute is not read
    x_perm, same_plan = moe_permute(x, None, experts, weights, plan=got_plan)
    assert same_plan is got_plan
    check_stage("permute_fwd", x_perm, ref["permute_fwd"], "moe_permute(plan=)")


@benchmark()
def test_moe_permute(n, experts, hidden, dist):
    expert_ids = make_expert_ids(n, experts, dist)
    n_rows = moe_permute_rows(n, experts)
    x, d_out = make_rows(n, hidden), make_rows(n, hidden)
    y_perm, d_perm = make_rows(n_rows, hidden), make_rows(n_rows, hidden)
    weights = torch.rand(n) + 0.5

    plan = run_torch_plan(expert_ids, experts)
    check_plan(plan, expert_ids, experts)
    ref = make_reference(x, y_perm, d_perm, d_out, plan, weights)
    check_grad_parity(x, y_perm, d_perm, d_out, plan, weights, ref)
    check_public_api(x, y_perm, d_perm, d_out, expert_ids, experts, weights, plan, ref)
    kplan = moe_permute_plan(expert_ids, experts, exact=False)

    row_bytes = hidden * x.element_size()
    moved = {  # bytes read + written (padding rows are written, not read)
        "permute_fwd": (n + n_rows) * row_bytes,
        "permute_bwd": 2 * n * row_bytes,
        "unpermute_fwd": 2 * n * row_bytes,
        "unpermute_bwd": (2 * n + n_rows) * row_bytes,
        "unpermute_fwd_noweight": 2 * n * row_bytes,
        "unpermute_bwd_noweight": (n + n_rows) * row_bytes,
    }
    ret = {"gfx": get_gfx_runtime(), "rows": n_rows}

    # plan: the plan kernels against the torch sort
    got, us = run_perftest(lambda: moe_permute_plan(expert_ids, experts, exact=False))
    check_kernel_plan(got, plan)
    ret["opus plan us"] = us
    _, ret["torch plan us"] = run_perftest(lambda: run_torch_plan(expert_ids, experts))

    candidates = {
        "opus": (_opus_candidates(x, y_perm, d_perm, d_out, kplan, weights), True),
        "torch": (
            _torch_index_candidates(x, y_perm, d_perm, d_out, plan, weights),
            False,
        ),
    }
    for (name, (fns, exact)), stage in itertools.product(candidates.items(), STAGES):
        out, us = run_perftest(fns[stage])
        outs = out if isinstance(out, tuple) else (out,)
        refs = ref[stage] if isinstance(ref[stage], tuple) else (ref[stage],)
        err = 0.0
        for o, r in zip(outs, refs):
            if exact:  # the same values everywhere: no tolerance (0.0 == -0.0)
                err = max(
                    err, float((o.to(dtypes.fp32) != r.to(dtypes.fp32)).float().mean())
                )
            else:
                tol = 2e-2 * max(float(r.to(dtypes.fp32).abs().max()), 1.0)
                err = max(
                    err,
                    checkAllclose(
                        r.to(dtypes.fp32),
                        o.to(dtypes.fp32),
                        rtol=2e-2,
                        atol=tol,
                        msg=f"{name}: {stage}",
                    ),
                )
        assert (
            err == 0
        ), f"{name} {stage}: {err:.2%} of elements differ from the reference"
        ret[f"{name} {stage} us"] = us
        ret[f"{name} {stage} TB/s"] = moved[stage] / us / 1e6
        ret[f"{name} {stage} err"] = err
    return ret


def _check_shape(n, experts, hidden, dist, seed=0):
    """Plan and K2 .. K5 through the public API against the reference, for one shape (not timed)."""
    expert_ids = make_expert_ids(n, experts, dist, seed)
    n_rows = moe_permute_rows(n, experts)
    x, d_out = make_rows(n, hidden), make_rows(n, hidden)
    y_perm, d_perm = make_rows(n_rows, hidden), make_rows(n_rows, hidden)
    w = torch.randn(n)
    plan = run_torch_plan(expert_ids, experts)
    ref = make_reference(x, y_perm, d_perm, d_out, plan, w)
    check_public_api(x, y_perm, d_perm, d_out, expert_ids, experts, w, plan, ref)


def test_plan_grid():
    """The plan for every (N, E) of the grid and every distribution, ids that are not experts included:
    N across bitmap words and token blocks (one block, several, more than one batch of block counts), E below
    and above the 256-token-block limit of the plan and not a multiple of its fill group.
    """
    for i, (n, experts) in enumerate(itertools.product(GRID_N, GRID_E)):
        for dist in DISTS + ["invalid"]:
            expert_ids = make_expert_ids(n, experts, dist, seed=i)
            ref = run_torch_plan(expert_ids, experts)
            check_plan(ref, expert_ids, experts)
            for ids in (expert_ids, expert_ids.view(-1, 1)):
                check_kernel_plan(
                    moe_permute_plan(ids, experts, exact=False),
                    ref,
                    tail=dist != "invalid",
                )


def test_small_shapes(max_elements=1 << 26):
    """K2 .. K5 for every (N, E) of the grid; the distributions and the H values that keep [R, H] below
    max_elements in turn (H below, at and above one chunk per thread)."""
    for i, (n, experts) in enumerate(itertools.product(GRID_N, GRID_E)):
        hs = [h for h in GRID_H if moe_permute_rows(n, experts) * h <= max_elements]
        _check_shape(n, experts, hs[i % len(hs)], DISTS[i % len(DISTS)], seed=i)
    for i, hidden in enumerate(GRID_H):  # every H with few rows
        _check_shape(
            [1, 257, 1025][i % 3],
            [1, 2, 31][i % 3],
            hidden,
            DISTS[i % len(DISTS)],
            seed=i,
        )


def test_exact_rows(hidden=64):
    """The default (exact): the [R] tables end at the rows in use, and every output is the one of the
    upper-bound plan (checked against the reference above) on those rows."""
    for i, (n, experts) in enumerate(itertools.product(GRID_N, [1, 2, 31, 32, 256])):
        dist = DISTS[i % len(DISTS)]
        ids = make_expert_ids(n, experts, dist, seed=i)
        bound, plan = moe_permute_plan(ids, experts, exact=False), moe_permute_plan(
            ids, experts
        )
        rows = plan.sorted_token_ids.numel()
        assert (
            rows == max(int(bound.num_valid_ids), GROUP_ALIGN)
            and rows % GROUP_ALIGN == 0
        )
        assert bound.sorted_token_ids.numel() == moe_permute_rows(n, experts) >= rows
        assert torch.equal(plan.sorted_token_ids, bound.sorted_token_ids[:rows])
        assert torch.equal(plan.grouped_layout, bound.grouped_layout[:rows])
        assert bool((bound.sorted_token_ids[rows:] == -1).all()) and bool(
            (bound.grouped_layout[rows:] == -1).all()
        )
        assert torch.equal(plan.reverse_sorted, bound.reverse_sorted) and torch.equal(
            plan.group_rows, bound.group_rows
        )
        if n * hidden > 1 << 24:
            continue
        x, d_out, w = make_rows(n, hidden), make_rows(n, hidden), torch.randn(n)
        y_perm, d_perm = make_rows(bound.sorted_token_ids.numel(), hidden), make_rows(
            bound.sorted_token_ids.numel(), hidden
        )
        got = []
        for p in (bound, None):
            r = rows if p is None else y_perm.shape[0]
            xr = x.detach().requires_grad_()
            x_perm, p = moe_permute(xr, ids, experts, plan=p)
            (dx,) = torch.autograd.grad(x_perm, xr, d_perm[:r].contiguous())
            yr, wr = (
                y_perm[:r].contiguous().requires_grad_(),
                w.detach().requires_grad_(),
            )
            out = moe_unpermute(yr, p, wr)
            dp, dw = torch.autograd.grad(out, (yr, wr), d_out)
            got.append((x_perm.detach()[:rows], dx, out.detach(), dp[:rows], dw))
            assert x_perm.shape[0] == r
        assert all(
            _same_bits(a, b) for a, b in zip(*got)
        ), f"exact rows differ from the upper-bound plan: N {n} E {experts} {dist}"


# A source larger than this (OPUS_MOE_PERMUTE_LLC_BYTES) is loaded non-temporal: different code from there on,
# and the small shapes do not reach it.
STREAM_BYTES = 256 << 20


def test_stream_shapes(hidden=6144):
    """H = 6144 with sources past the last-level cache (non-temporal loads), every distribution."""
    for i, (n, experts) in enumerate(
        [(50000, 32), (69632, 32), (70001, 256), (139264, 32), (69632, 64), (50000, 31)]
    ):
        assert n * hidden * 2 > STREAM_BYTES
        _check_shape(n, experts, hidden, DISTS[i % len(DISTS)], seed=i)


def test_large_buffers(hidden=6144, n=350000, experts=32):
    """x / y_perm of 4 GiB and more (rows past the 32-bit byte offsets of one view). Not in the default run:
    about 60 GiB of device memory with the reference (--large)."""
    assert n * hidden * 2 >= 1 << 32
    _check_shape(n, experts, hidden, "uniform")


def test_api_rejects_bad_inputs():
    rows = moe_permute_rows(64, 32)
    plan = moe_permute_plan(make_expert_ids(64, 32, "uniform"), 32, exact=False)
    for bad in (
        lambda: moe_permute(torch.zeros(64, 64), None, 32, plan=plan),  # fp32 x
        lambda: moe_permute(
            torch.zeros(63, 64, dtype=dtypes.bf16), None, 32, plan=plan
        ),  # tokens
        lambda: moe_permute(
            torch.zeros(64, 60, dtype=dtypes.bf16), None, 32, plan=plan
        ),  # H % 8
        lambda: moe_unpermute(
            torch.zeros(rows - 1, 64, dtype=dtypes.bf16), plan, torch.zeros(64)
        ),
        lambda: moe_unpermute(
            torch.zeros(rows, 64, dtype=dtypes.bf16),
            plan,
            torch.zeros(64, dtype=dtypes.bf16),
        ),
        lambda: moe_unpermute(
            torch.zeros(rows, 64, dtype=dtypes.bf16), plan, torch.zeros(63)
        ),  # weights per token
        lambda: moe_unpermute(
            torch.zeros(rows, 64, dtype=dtypes.bf16), plan, torch.zeros(32, 2)
        ),  # top-k weights
        lambda: moe_permute_plan(torch.zeros(64, dtype=torch.int64), 32),
        lambda: moe_permute_plan(
            torch.zeros(64, 8, dtype=torch.int32), 32
        ),  # top-k ids
        lambda: moe_permute_plan(torch.zeros(64, dtype=torch.int32), 0),
        lambda: moe_permute_plan(
            torch.zeros(64, dtype=torch.int32), mp.PLAN_MAX_EXPERTS + 1
        ),  # experts
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a malformed input was accepted")


def main():
    parser = argparse.ArgumentParser(description="MoE token permute / unpermute (OPUS)")
    parser.add_argument("-n", "--tokens", type=int, nargs="*", default=[69632, 8704])
    parser.add_argument("-e", "--experts", type=int, nargs="*", default=[32])
    parser.add_argument("--hidden", type=int, nargs="*", default=[6144])
    parser.add_argument("--dist", type=str, nargs="*", default=DISTS, choices=DISTS)
    parser.add_argument(
        "--large", action="store_true", help="also run the 4 GiB buffer test"
    )
    args = parser.parse_args()
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.info(
            "moe_permute: %s is not supported, nothing to run", get_gfx_runtime()
        )
        return
    test_api_rejects_bad_inputs()
    test_plan_grid()
    test_exact_rows()
    test_small_shapes()
    test_stream_shapes()
    if args.large:
        test_large_buffers()
    df = [
        test_moe_permute(n, e, h, d)
        for n, e, h, d in itertools.product(
            args.tokens, args.experts, args.hidden, args.dist
        )
    ]
    aiter.logger.info(
        "moe_permute summary (markdown):\n%s", pd.DataFrame(df).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
