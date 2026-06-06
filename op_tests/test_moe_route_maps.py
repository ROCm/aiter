# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate the atomic-kernel route map (build_route_maps) used by fused_moe.

build_route_maps (SGLang-style atomic-scatter argsort) produces topids_to_rows
[t,k] = topk_ids[t,k]*max_m + slot with the within-expert slot in atomic-race
order. We check it is a *valid* mapping (each expert's assigned rows are exactly
its block [e*max_m, e*max_m+counts[e]), no collisions), and that it is
set-equivalent to the deterministic build_topids_to_rows (same rows per
expert, possibly different within-expert order).

Usage: python op_tests/test_moe_route_maps.py
"""

import argparse
import importlib.util
import os
import sys
import types

os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch

torch.set_default_device("cuda")

WARP_TILE_M = 16


def _import():
    try:
        from aiter.ops.flydsl.moe_kernels import (
            build_route_maps,
            build_topids_to_rows,
        )

        return build_route_maps, build_topids_to_rows
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(root, "aiter", "ops", "flydsl")

        def _load(path, name):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        for n in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
            sys.modules.setdefault(n, types.ModuleType(n))
        kern = _load(
            os.path.join(base, "kernels", "moe_route_maps.py"),
            "aiter.ops.flydsl.kernels.moe_route_maps",
        )
        sys.modules["aiter.ops.flydsl.kernels.moe_route_maps"] = kern
        mk = _load(os.path.join(base, "moe_kernels.py"), "aiter.ops.flydsl.moe_kernels")
        return mk.build_route_maps, mk.build_topids_to_rows


build_route_maps, build_topids_to_rows = _import()


def _run_one(token_num, topk, E, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    topk_ids = torch.stack(
        [torch.randperm(E, generator=gen, device="cuda")[:topk] for _ in range(token_num)]
    ).to(torch.long)
    counts = torch.bincount(topk_ids.reshape(-1), minlength=E)
    max_m = int(counts.max().item()) if counts.numel() else 0
    max_m = max(WARP_TILE_M, ((max_m + WARP_TILE_M - 1) // WARP_TILE_M) * WARP_TILE_M)

    src, dst, masked_m = build_route_maps(topk_ids, E, max_m)
    torch.cuda.synchronize()
    # masked_m (from the atomic counters) must equal bincount(topk_ids).
    masked_m_ok = bool(torch.equal(masked_m, counts.to(torch.int32)))
    srf = src.reshape(-1).to(torch.long)
    fe = topk_ids.reshape(-1)
    flat_tokens = (torch.arange(token_num * topk, device="cuda") // topk)

    # 1) every route's row is in its expert's block
    in_block = bool(((srf >= fe * max_m) & (srf < fe * max_m + counts[fe])).all().item())

    # 2) each expert's rows are exactly {e*max_m .. e*max_m+counts[e]-1} (valid perm)
    ref = build_topids_to_rows(topk_ids, max_m, E).reshape(-1).to(torch.long)
    perm_ok = True
    for e in range(E):
        got = torch.sort(srf[fe == e]).values
        exp = torch.sort(ref[fe == e]).values  # same expected set as the deterministic build
        if not torch.equal(got, exp):
            perm_ok = False
            break

    # 3) rows_to_tokens is the exact inverse of topids_to_rows: dst[src[route]] == token(route),
    #    and padding rows (not targeted by any route) stay -1.
    inv_ok = bool((dst.long()[srf] == flat_tokens).all().item())
    valid_mask = torch.zeros(E * max_m, dtype=torch.bool, device="cuda")
    valid_mask[srf] = True
    pad_ok = bool((dst[~valid_mask] == -1).all().item())

    ok = in_block and perm_ok and inv_ok and pad_ok and masked_m_ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] tok={token_num:<5} topk={topk} E={E:<3} "
        f"max_m={max_m} in_block={in_block} set_eq={perm_ok} "
        f"dst_inverse={inv_ok} pad-1={pad_ok} masked_m={masked_m_ok}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    configs = [
        (1, 1, 8), (1, 8, 32), (7, 2, 8), (13, 3, 8), (128, 8, 32),
        (256, 8, 256), (32, 4, 16), (1, 4, 4),
    ]
    all_ok = True
    for i, (tn, topk, E) in enumerate(configs):
        all_ok &= _run_one(tn, topk, E, seed=i)
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
