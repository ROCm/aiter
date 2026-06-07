# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate the FlyDSL m-tile-map kernel against the original host reference.

The kernel-backed ``_make_m_tile_map`` packs the grouped-persistent M-tile
schedule on-device (no .cpu() sync). This test checks, for several masked_m
distributions, that ``m_tile_map[0:total]`` (total = prefix[E]) matches the
original Python packing exactly:

    packed = [e*max_m_tiles + j
              for e in range(E) for j in range(ceil(clamp(masked_m[e],0,max_m)/tile_m))]

    python op_tests/test_moe_m_tile_map.py
"""

import os
import sys
import types

os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch

torch.set_default_device("cuda")


def _ref_packed(masked_m, experts, max_m, tile_m):
    valid_m = masked_m[:experts].to(torch.int32).clamp(min=0, max=max_m)
    valid_tiles = ((valid_m + (tile_m - 1)) // tile_m).cpu().tolist()
    max_m_tiles = (max_m + tile_m - 1) // tile_m
    packed = [
        e * max_m_tiles + j for e, c in enumerate(valid_tiles) for j in range(int(c))
    ]
    return packed, max_m_tiles


def _run_one(experts, max_m, tile_m, dist, seed=0):
    from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
        _make_m_tile_map,
        _make_m_tile_prefix,
    )

    cfg = types.SimpleNamespace(experts=experts, max_m=max_m, tile_m=tile_m)

    gen = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "rand":
        masked_m = torch.randint(
            0, max_m + 1, (experts,), generator=gen, device="cuda"
        ).to(torch.int32)
    elif dist == "empty":
        masked_m = torch.zeros(experts, dtype=torch.int32, device="cuda")
    elif dist == "full":
        masked_m = torch.full((experts,), max_m, dtype=torch.int32, device="cuda")
    elif dist == "sparse":  # only a few experts active
        masked_m = torch.zeros(experts, dtype=torch.int32, device="cuda")
        masked_m[0] = max_m
        if experts > 3:
            masked_m[3] = 1
        masked_m[-1] = max_m // 2 + 1
    else:
        raise ValueError(dist)

    ref_packed, max_m_tiles = _ref_packed(masked_m, experts, max_m, tile_m)
    total = len(ref_packed)

    prefix = _make_m_tile_prefix(masked_m, cfg)
    prefix_total = int(prefix[experts].item())  # device-side total the GEMM uses

    # NAIVE=1: original host packing (exactly-sized tensor).
    os.environ["AITER_GROUPED_GEMM_NAIVE"] = "1"
    naive_map = _make_m_tile_map(masked_m, cfg, prefix)
    # NAIVE=0: FlyDSL kernel (max-sized buffer, valid prefix [0:total]).
    os.environ["AITER_GROUPED_GEMM_NAIVE"] = "0"
    kernel_map = _make_m_tile_map(masked_m, cfg, prefix)
    torch.cuda.synchronize()

    total_ok = prefix_total == total
    # kernel buffer sized to the max; naive tensor sized to total (or 1 if empty)
    size_ok = kernel_map.numel() == experts * max_m_tiles
    # both must reproduce the reference packing on their valid prefix
    naive_ok = naive_map[:total].cpu().tolist() == ref_packed
    kernel_ok = kernel_map[:total].cpu().tolist() == ref_packed
    # naive empty-case returns [0]; kernel reads nothing (total=0) -> both fine
    naive_empty_ok = (total > 0) or (naive_map.cpu().tolist() == [0])

    ok = total_ok and size_ok and naive_ok and kernel_ok and naive_empty_ok
    print(
        f"[{'PASS' if ok else 'FAIL'}] E={experts:<4} max_m={max_m:<4} tile_m={tile_m:<3} "
        f"dist={dist:<6} total={total:<5} prefix_total={prefix_total:<5} "
        f"size={size_ok} naive={naive_ok} kernel={kernel_ok}"
    )
    return ok


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA/ROCm GPU")
    configs = [
        (32, 16, 16),
        (32, 256, 16),
        (256, 16, 16),
        (8, 128, 32),
        (128, 64, 16),
        (1, 16, 16),
    ]
    all_ok = True
    seed = 0
    for E, max_m, tile_m in configs:
        for dist in ("rand", "empty", "full", "sparse"):
            all_ok &= _run_one(E, max_m, tile_m, dist, seed=seed)
            seed += 1
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
