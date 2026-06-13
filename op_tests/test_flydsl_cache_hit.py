#!/usr/bin/env python3
"""Validate FlyDSL disk cache hit for grouped MoE GEMM kernels.

Usage:
    # Must set GPU_ARCHS and FLYDSL env for cross-compile on non-gfx1250:
    GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_COMPILE_ARCH=gfx1250 \
        python op_tests/test_flydsl_cache_hit.py

    # On real gfx1250 just:
    python op_tests/test_flydsl_cache_hit.py
"""

import argparse
import glob
import os
import shutil
import sys
import time

_LOCAL_DEPS = ("/root/data/aiter", "/root/data/triton/python")
for _dep in reversed(_LOCAL_DEPS):
    if os.path.exists(_dep) and _dep not in sys.path:
        sys.path.insert(0, _dep)

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
    compile_moe_grouped_gemm1_a8w4_masked,
)
from flydsl.compiler.jit_function import JitFunction


def _get_launcher_info(stage1_launch):
    """Extract the 3 JitFunction launchers from the stage1 launch closure."""
    cells = {
        name: cell.cell_contents
        for name, cell in zip(
            stage1_launch.__code__.co_freevars, stage1_launch.__closure__
        )
    }
    result = {}
    for name in ("fused_base", "fused_base_bias", "raw_base"):
        fn = cells.get(name)
        if fn is None or not isinstance(fn, JitFunction):
            continue
        fn._ensure_cache_manager()
        cache_dir_name = f"{fn.func.__name__}_{fn.manager_key}"
        result[name] = {
            "func_name": fn.func.__name__,
            "manager_key": fn.manager_key,
            "cache_dir_name": cache_dir_name,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache2", default="cache2",
        help="Path to the online cache pulled from gfx1250 (default: ./cache2)",
    )
    parser.add_argument("--experts", type=int, default=257)
    parser.add_argument("--max-m", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=32)
    args = parser.parse_args()

    cache2 = os.path.abspath(args.cache2)
    cache_home = os.path.expanduser("~/.flydsl/cache")
    E = args.experts
    max_m = args.max_m
    model_dim = args.model_dim
    inter_dim = args.inter_dim
    tile_m = args.tile_m
    K_scale = model_dim // 32
    wmma_m_rep = (tile_m // 1) // 16  # m_warp=1
    _wmma_rep = (256 // 4) // 16  # warp_tile_n // WMMA_N

    print(f"experts={E} max_m={max_m} model_dim={model_dim} inter_dim={inter_dim} tile_m={tile_m}")
    print(f"cache2: {cache2}")
    print(f"cache_home: {cache_home}")

    # --- Step 1: compile (Python-level only, returns closures) ---
    stage1 = compile_moe_grouped_gemm1_a8w4_masked(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, max_m=max_m,
        tile_m=tile_m, tile_n=256, tile_k=256, m_warp=1, n_warp=4,
        out_dtype="bf16", num_buffers=2, split_k=1,
        expert_sched_mode=False, grouped_persistent_m=False,
        grouped_contiguous_m=False, persistent_workers=None,
        act="silu", stage1_weight_layout="gguu", data_format="a8w4",
    )

    info = _get_launcher_info(stage1)
    fused_info = info["fused_base"]
    print(f"\nfused_base manager_key: {fused_info['manager_key']}")
    print(f"fused_base cache dir:   {fused_info['cache_dir_name']}")

    # --- Step 2: check cache2 for matching pkl ---
    local_dir = os.path.join(cache_home, fused_info["cache_dir_name"])
    already_cached = bool(glob.glob(f"{local_dir}/*.pkl"))
    print(f"local cache exists: {already_cached}")

    if not already_cached and os.path.isdir(cache2):
        # Search cache2 for dirs with the same func name, any manager_key
        func_name = fused_info["func_name"]
        c2_dirs = glob.glob(f"{cache2}/{func_name}_*")
        print(f"cache2 has {len(c2_dirs)} dirs for {func_name}_*")

        # Find all unique pkl names across cache2 dirs
        c2_pkls = {}
        for d in c2_dirs:
            for p in glob.glob(f"{d}/*.pkl"):
                pkl_name = os.path.basename(p)
                c2_pkls.setdefault(pkl_name, []).append(p)

        print(f"cache2 unique pkl names: {list(c2_pkls.keys())}")
    else:
        c2_pkls = {}

    # --- Step 3: trigger flyc.compile and get the cache_key pkl name ---
    # Clean local cache for this dir so we compile fresh
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir)
        print(f"cleaned local dir: {local_dir}")

    device = "cuda"
    y = torch.empty((E, max_m, inter_dim), dtype=torch.bfloat16, device=device)
    x = torch.empty((E, max_m, model_dim), dtype=torch.uint8, device=device)
    w = torch.empty((E, 2 * inter_dim, model_dim // 2), dtype=torch.uint8, device=device)
    sx = torch.empty((E, max_m // wmma_m_rep, K_scale * wmma_m_rep), dtype=torch.uint8, device=device)
    sw = torch.empty((E, 2 * inter_dim // _wmma_rep, K_scale * _wmma_rep), dtype=torch.uint8, device=device)
    mm = torch.full((E,), max_m, dtype=torch.int32, device=device)

    print(f"\n=== Compile from scratch (no cache) ===")
    t0 = time.time()
    try:
        stage1(y, x, w, sx, sw, mm, max_m, inter_dim, model_dim, E,
               stream=torch.cuda.current_stream())
    except Exception:
        pass
    compile_time = time.time() - t0
    print(f"fused_base compile: {compile_time:.2f}s")

    # Find the pkl that was just written
    new_pkls = glob.glob(f"{local_dir}/*.pkl")
    if not new_pkls:
        print("ERROR: no pkl written to local cache!")
        sys.exit(1)

    pkl_name = os.path.basename(new_pkls[0])
    pkl_size = os.path.getsize(new_pkls[0])
    print(f"cache_key pkl: {pkl_name} ({pkl_size} bytes)")

    # --- Step 4: check if this pkl exists in cache2 ---
    if pkl_name in c2_pkls:
        c2_sources = c2_pkls[pkl_name]
        c2_size = os.path.getsize(c2_sources[0])
        print(f"\nMATCH: {pkl_name} found in {len(c2_sources)} cache2 dirs")
        print(f"  cache2 size: {c2_size} bytes, local size: {pkl_size} bytes")

        # --- Step 5: validate cache hit using cache2 pkl ---
        # Remove local compiled pkl, replace with cache2 copy
        os.remove(new_pkls[0])
        shutil.copy2(c2_sources[0], new_pkls[0])
        print(f"  replaced local pkl with cache2 copy")

        # Force new process state: need fresh JitFunction instances
        # Easiest: re-import and re-call
        # But we can't re-import in same process. Instead, clear the in-process caches.
        compile_moe_grouped_gemm1_a8w4_masked.cache_clear()

        stage1_v2 = compile_moe_grouped_gemm1_a8w4_masked(
            model_dim=model_dim, inter_dim=inter_dim, experts=E, max_m=max_m,
            tile_m=tile_m, tile_n=256, tile_k=256, m_warp=1, n_warp=4,
            out_dtype="bf16", num_buffers=2, split_k=1,
            expert_sched_mode=False, grouped_persistent_m=False,
            grouped_contiguous_m=False, persistent_workers=None,
            act="silu", stage1_weight_layout="gguu", data_format="a8w4",
        )

        print(f"\n=== Cache hit test (using cache2 pkl) ===")
        t0 = time.time()
        try:
            stage1_v2(y, x, w, sx, sw, mm, max_m, inter_dim, model_dim, E,
                      stream=torch.cuda.current_stream())
        except Exception:
            pass
        hit_time = time.time() - t0
        print(f"fused_base with cache2 pkl: {hit_time:.3f}s")

        if hit_time < compile_time * 0.5 and hit_time < 1.0:
            print(f"PASS: cache2 pkl is a valid cache hit ({hit_time:.3f}s vs {compile_time:.2f}s compile)")
        else:
            print(f"FAIL: cache2 pkl did NOT produce a cache hit")
            sys.exit(1)
    else:
        print(f"\nNO MATCH: {pkl_name} not found in cache2")
        print(f"  cache2 has: {list(c2_pkls.keys()) if c2_pkls else '(empty or not found)'}")
        print(f"  The online cache was built with different runtime args.")
        sys.exit(1)


if __name__ == "__main__":
    main()
