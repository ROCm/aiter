# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cross-arch shared codegen helpers + emit registry.

Each arch module under codegen/ self-registers its emit functions at import
time via register_emit(arch, kernel_tag, fn).  The entry-point gen_instances.py
imports each arch module (triggering registration) and dispatches via
dispatch_emit(cg, k, **kwargs).  Adding a new arch (e.g. gfx1250) = one new
file + one new import; entry point itself is arch-agnostic.
"""

WARP_SIZE = 64

# Paired gfx942 kernels (nosplit_tag -> splitk_tag) share one <Traits, Kargs> template.
W3_KERNEL_PAIRS = {
    "a16w16_kbuf2v": "a16w16_kbuf2v_sk",
    "a16w16_kbuf2v_bk128": "a16w16_kbuf2v_bk128_sk",
    "a16w16_kbuf1": "a16w16_kbuf1_sk",
    "a16w16_quad_mfma32_kbuf1": "a16w16_quad_mfma32_kbuf1_sk",
}
_NOSPLIT = tuple(W3_KERNEL_PAIRS.keys())
_SPLITK = tuple(W3_KERNEL_PAIRS.values())
_GFX942_A16W16_TAGS = (
    _SPLITK
    + (
        "a16w16_em3en4_lds1_pgr2_sk",
        "a16w16_kbuf1_large_tile",
        "a16w16_wave_k_coop",
        "a16w16_wave_k_coop_accum",
    )
    + _NOSPLIT
)
_A16W16_TAGS = (
    "a16w16",
    "a16w16_flatmm",
    "a16w16_flatmm_splitk",
    "a16w16_persistent",
    "a16w16_mono_tile",
    # gfx1250 cluster/TDM split-K (fp32 workspace + reduce kernel).
    "a16w16_cluster_tdm_splitk_ws",
    # gfx1250 CLUSTER-LAUNCH (multicast) TDM split-K (fp32 workspace + reduce).
    "a16w16_clusterlaunch_tdm_splitk_ws",
) + _GFX942_A16W16_TAGS

EMIT_REGISTRY = {}

# Per-arch map registry: {(arch, map_name): dict}. Each arch module registers
# its overrides at import time; gen_instances merges them into the cross-arch
# default maps.
ARCH_MAP_REGISTRY = {}


def register_arch_map(arch, map_name, mapping):
    key = (arch, map_name)
    if key in ARCH_MAP_REGISTRY:
        raise RuntimeError(f"arch map already registered for {key}")
    ARCH_MAP_REGISTRY[key] = mapping


def get_arch_map(arch, map_name):
    """Return the registered map, or {} if none."""
    return ARCH_MAP_REGISTRY.get((arch, map_name), {})


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
