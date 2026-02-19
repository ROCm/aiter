# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Main orchestrator for Triton kernel precompilation.

Cross-compiles all registered Triton kernels for target GPU architectures
using triton.compile(ASTSource(...), target=GPUTarget("hip", arch, 64)).
No GPU is required at build time.

Usage:
    # From setup.py (automatic):
    PREBUILD_TRITON=1 pip install -e .

    # Standalone:
    python -m aiter.prebuild_triton.warmup --archs gfx942,gfx950

Compiled .hsaco + .json are stored in:
    aiter/prebuild_triton_cache/<arch>/<hash>/
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def prebuild_triton_kernels(cache_dir, archs=None, max_workers=4, verbose=True):
    """
    Cross-compile all registered Triton kernels for target architectures.

    Args:
        cache_dir: Root directory for the prebuild cache
                   (e.g., "aiter/prebuild_triton_cache")
        archs: List of GPU architecture strings (e.g., ["gfx942", "gfx950"])
        max_workers: Number of parallel compilation threads
        verbose: Print progress information
    """
    if archs is None:
        archs = ["gfx942", "gfx950"]

    import triton
    from triton.compiler import ASTSource

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:
        from triton.compiler import GPUTarget

    from .registry import get_kernel_specs

    specs = get_kernel_specs()
    if not specs:
        print("[prebuild_triton] Warning: no kernel specs registered")
        return

    total_compiled = 0
    total_failed = 0
    total_skipped = 0
    t0 = time.time()

    for arch in archs:
        target = GPUTarget("hip", arch, 64)
        arch_cache = os.path.join(cache_dir, arch)
        os.makedirs(arch_cache, exist_ok=True)

        if verbose:
            print(f"\n[prebuild_triton] Compiling for {arch} -> {arch_cache}")

        # Collect all (spec, variant) pairs
        work_items = []
        for spec in specs:
            try:
                variants = spec.get_variants()
                for variant in variants:
                    work_items.append((spec, variant))
            except Exception as e:
                print(
                    f"[prebuild_triton] Warning: failed to get variants "
                    f"for {spec.name}: {e}"
                )

        if verbose:
            print(
                f"[prebuild_triton] {len(work_items)} kernel variants "
                f"to compile for {arch}"
            )

        arch_compiled = 0
        arch_failed = 0
        arch_skipped = 0

        def compile_one(item):
            spec, variant = item
            # Set per-thread cache dir
            os.environ["TRITON_CACHE_DIR"] = arch_cache

            # Build constexprs and signature
            constexprs = variant.constexprs
            # Build full signature: non-constexpr params get their type,
            # constexpr params get "constexpr"
            signature = dict(variant.signature)
            for k in constexprs:
                signature[k] = "constexpr"

            # Build attrs (divisibility hints for pointer args)
            attrs = {}
            for i, (k, v) in enumerate(variant.signature.items()):
                if v.startswith("*"):
                    attrs[(i,)] = [["tt.divisibility", 16]]

            try:
                src = ASTSource(
                    fn=spec.kernel_fn,
                    constexprs=constexprs,
                    signature=signature,
                    attrs=attrs,
                )

                backend = triton.compiler.make_backend(target)
                options_dict = {
                    "num_warps": variant.num_warps,
                    "num_stages": variant.num_stages,
                    "waves_per_eu": variant.waves_per_eu,
                }
                options = backend.parse_options(options_dict)

                triton.compile(src, target=target, options=options.__dict__)
                return ("ok", spec.name)
            except Exception as e:
                return ("fail", spec.name, str(e))

        # Compile with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compile_one, item): item for item in work_items}
            for future in as_completed(futures):
                result = future.result()
                if result[0] == "ok":
                    arch_compiled += 1
                elif result[0] == "fail":
                    arch_failed += 1
                    if verbose:
                        print(f"  FAIL: {result[1]}: {result[2][:120]}")

        total_compiled += arch_compiled
        total_failed += arch_failed
        total_skipped += arch_skipped

        if verbose:
            print(
                f"[prebuild_triton] {arch}: "
                f"{arch_compiled} compiled, "
                f"{arch_failed} failed, "
                f"{arch_skipped} skipped"
            )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"\n[prebuild_triton] Done in {elapsed:.1f}s: "
            f"{total_compiled} compiled, "
            f"{total_failed} failed across {len(archs)} arch(es)"
        )

    # Count .hsaco files
    hsaco_count = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".hsaco"):
                hsaco_count += 1
    if verbose:
        print(f"[prebuild_triton] Cache contains {hsaco_count} .hsaco files")

    return total_compiled, total_failed


def main():
    """CLI entry point for standalone precompilation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompile AITER Triton kernels for target GPU architectures"
    )
    parser.add_argument(
        "--archs",
        type=str,
        default="gfx942,gfx950",
        help="Comma-separated list of GPU architectures (default: gfx942,gfx950)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: aiter/prebuild_triton_cache/)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel compilation threads (default: 4)",
    )
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="List registered kernels and exit",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate variant count and exit",
    )
    args = parser.parse_args()

    if args.list_kernels:
        from .registry import list_registered_kernels

        print("Registered kernels:")
        list_registered_kernels()
        return

    if args.estimate:
        from .registry import estimate_variant_count

        print("Variant count estimate:")
        estimate_variant_count()
        return

    archs = [a.strip() for a in args.archs.split(",")]
    cache_dir = args.cache_dir
    if cache_dir is None:
        # Default: relative to aiter package
        aiter_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(aiter_dir, "prebuild_triton_cache")

    prebuild_triton_kernels(
        cache_dir=cache_dir,
        archs=archs,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
