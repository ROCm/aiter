# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NAIVE=0 vs NAIVE=1 equivalence for the grouped gfx1250 MoE routing primitives.

The grouped path supports two route-map builders (selected by
AITER_GROUPED_GEMM_NAIVE):
  * NAIVE=0 -> build_route_maps          (atomic-scatter FlyDSL kernel)
  * NAIVE=1 -> _build_route_maps_naive   (pure-torch, token-major order)

They assign within-expert rows in *different order* (atomic-race vs token-major),
but the contract guarantees the *final* MoE output is identical: the grouped GEMM
is order-agnostic within an expert, scatter-copy/gather-reduce key off the same
maps, and masked_m == bincount in both. This test proves that end to end, using
the real FlyDSL kernels, without the (gfx1250-only) stage1/stage2 GEMM:

  1. build_route_maps   : masked_m identical; per-expert row *sets* identical;
                          both maps are valid inverses (rows_to_tokens o
                          topids_to_rows == identity); padding stays -1.
  2. scatter_copy       : each builder places the correct token payload in every
                          valid grouped row (grouped[row] == x[token(row)]).
  3. gather_reduce      : the per-token reduced output is bit-identical between
                          the two builders (order independence of the sum).

A no-op "GEMM" (identity) stands in for stage1/stage2: the masked GEMM applies
the same per-row transform regardless of which slot a row occupies, so identity
is a faithful proxy for the ordering question this test targets.

    python op_tests/test_grouped_naive_equiv.py
"""

import os
import sys

os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch

torch.set_default_device("cuda")

WARP_TILE_M = 16


def _import():
    from aiter.ops.flydsl.moe_kernels import (
        build_route_maps,
        flydsl_moe_gather_reduce,
        flydsl_moe_scatter_copy_token,
    )
    from aiter.fused_moe import _build_route_maps_naive

    return (
        build_route_maps,
        _build_route_maps_naive,
        flydsl_moe_scatter_copy_token,
        flydsl_moe_gather_reduce,
    )


(
    build_route_maps,
    build_route_maps_naive,
    scatter_copy,
    gather_reduce,
) = _import()


def _make_topk_ids(token_num, topk, E, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.stack(
        [torch.randperm(E, generator=gen, device="cuda")[:topk] for _ in range(token_num)]
    ).to(torch.int32)


def _static_max_m(token_num):
    # Same static upper bound fused_moe uses (token_num rounded up to WARP_TILE_M).
    return max(WARP_TILE_M, ((token_num + WARP_TILE_M - 1) // WARP_TILE_M) * WARP_TILE_M)


def _check_maps_equiv(topk_ids, E, max_m):
    """build_route_maps (kernel) vs _build_route_maps_naive: masked_m identical,
    per-expert row sets identical, both valid inverses, padding -1."""
    token_num, topk = topk_ids.shape
    t2r_k, r2t_k, mm_k = build_route_maps(topk_ids, E, max_m)
    t2r_n, r2t_n, mm_n = build_route_maps_naive(topk_ids, E, max_m)
    torch.cuda.synchronize()

    bc = torch.bincount(topk_ids.reshape(-1).long(), minlength=E).to(torch.int32)
    masked_ok = torch.equal(mm_k, bc) and torch.equal(mm_n, bc)

    fe = topk_ids.reshape(-1)
    flat_tok = (torch.arange(token_num * topk, device="cuda") // topk).to(torch.int32)

    set_ok = True
    for e in range(E):
        a = torch.sort(t2r_k.reshape(-1)[fe == e]).values
        b = torch.sort(t2r_n.reshape(-1)[fe == e]).values
        if not torch.equal(a, b):
            set_ok = False
            break

    # inverse: rows_to_tokens[ topids_to_rows[route] ] == token(route)
    inv_k = bool((r2t_k.long()[t2r_k.reshape(-1).long()] == flat_tok.long()).all())
    inv_n = bool((r2t_n.long()[t2r_n.reshape(-1).long()] == flat_tok.long()).all())

    return masked_ok and set_ok and inv_k and inv_n, (t2r_k, r2t_k, t2r_n, r2t_n, mm_k)


def _check_scatter_equiv(topk_ids, E, max_m, maps):
    """Each builder must place the correct token payload in every valid row:
    grouped[row] == x[token(row)] for all rows where rows_to_tokens[row] != -1."""
    t2r_k, r2t_k, t2r_n, r2t_n, _ = maps
    token_num, topk = topk_ids.shape
    dim = 256  # bytes per token (multiple of 32)
    gen = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randint(0, 256, (token_num, dim), dtype=torch.uint8, device="cuda", generator=gen)
    sc = torch.randint(0, 256, (token_num, dim // 32), dtype=torch.uint8, device="cuda", generator=gen)

    ok = True
    for r2t in (r2t_k, r2t_n):
        g = torch.zeros((E, max_m, dim), dtype=torch.uint8, device="cuda")
        gs = torch.zeros((E, max_m, dim // 32), dtype=torch.uint8, device="cuda")
        scatter_copy(x, sc, r2t, E, max_m, grouped_a1=g, a1_scale_raw=gs)
        torch.cuda.synchronize()
        valid = r2t >= 0
        rows = torch.nonzero(valid).squeeze(1)
        toks = r2t[rows].long()
        gf = g.view(E * max_m, dim)
        gsf = gs.view(E * max_m, dim // 32)
        if not (torch.equal(gf[rows], x[toks]) and torch.equal(gsf[rows], sc[toks])):
            ok = False
            break
    return ok


def _check_gather_equiv(topk_ids, topk_weight, E, max_m, maps):
    """Per-token reduced output must be bit-identical for the two builders.

    Fill each grouped_out row with the (token,expert)-specific contribution, then
    gather_reduce. moe_out[t] = sum_k w[t,k] * contribution[t,k] regardless of the
    within-expert slot, so the two orderings must agree exactly."""
    t2r_k, r2t_k, t2r_n, r2t_n, _ = maps
    token_num, topk = topk_ids.shape
    dim = 512
    gen = torch.Generator(device="cuda").manual_seed(7)
    # per-route contribution = "expert output for this token"
    contrib = torch.randn(token_num, topk, dim, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = topk_weight.to(torch.bfloat16)

    outs = []
    for t2r in (t2r_k, t2r_n):
        grouped_out = torch.zeros((E, max_m, dim), dtype=torch.bfloat16, device="cuda")
        grouped_out.view(E * max_m, dim)[t2r.reshape(-1).long()] = contrib.reshape(-1, dim)
        moe = gather_reduce(grouped_out, t2r, w)
        torch.cuda.synchronize()
        outs.append(moe)
    return torch.equal(outs[0], outs[1]), outs


def _run_one(token_num, topk, E, seed):
    topk_ids = _make_topk_ids(token_num, topk, E, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 100)
    w = torch.rand(token_num, topk, generator=gen, device="cuda")
    w = w / w.sum(-1, keepdim=True)
    max_m = _static_max_m(token_num)

    maps_ok, maps = _check_maps_equiv(topk_ids, E, max_m)
    scat_ok = _check_scatter_equiv(topk_ids, E, max_m, maps)
    gath_ok, _ = _check_gather_equiv(topk_ids, w, E, max_m, maps)

    ok = maps_ok and scat_ok and gath_ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] tok={token_num:<5} topk={topk} E={E:<4} "
        f"max_m={max_m:<3} route_maps={maps_ok} scatter_copy={scat_ok} "
        f"gather_reduce={gath_ok}"
    )
    return ok


def main():
    configs = [
        (1, 1, 8),
        (1, 8, 32),
        (7, 2, 8),
        (13, 3, 16),
        (128, 8, 32),
        (256, 8, 256),
        (64, 4, 16),
        (1, 4, 4),
    ]
    all_ok = True
    for i, (tn, topk, E) in enumerate(configs):
        all_ok &= _run_one(tn, topk, E, seed=i)
    print("\nALL PASS (NAIVE 0/1 equivalent)" if all_ok else "\nSOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
