# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cross-arch shared codegen helpers + emit registry.

Each arch module under codegen/ self-registers its emit functions at import
time via register_emit(arch, kernel_tag, fn).  The entry-point gen_instances.py
imports each arch module (triggering registration) and dispatches via
dispatch_emit(cg, k, **kwargs).  Adding a new arch (e.g. gfx1250) = one new
file + one new import; entry point itself is arch-agnostic.
"""

EMIT_REGISTRY = {}


def kid_arch(k):
    """Resolve a kid's target arch_prefix (defaults to gfx950 for legacy kids)."""
    return (getattr(k, "arch_prefix", "") or "gfx950").lower()


def register_emit(arch, kernel_tag, fn):
    """Register a per-(arch, kernel_tag) emit function. Called at arch-module import."""
    key = (arch, kernel_tag)
    if key in EMIT_REGISTRY:
        raise RuntimeError(f"emit already registered for {key}")
    EMIT_REGISTRY[key] = fn


def dispatch_emit(cg, k, **kwargs):
    """Lookup (kid_arch(k), k.kernel_tag) -> call registered emit."""
    key = (kid_arch(k), k.kernel_tag)
    fn = EMIT_REGISTRY.get(key)
    if fn is None:
        raise KeyError(
            f"No emit registered for {key}. "
            f"Available: {sorted(EMIT_REGISTRY.keys())}"
        )
    return fn(cg, k, **kwargs)
