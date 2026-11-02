# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Synthetic tuning problems with known optima, for tests and demos.

:func:`gemm_like` is a GEMM-tuning-shaped space: ordinal tiles, a
categorical pipeline and warp layout, a gated prefetch with a depth
sub-knob, an occupancy knob, resource constraints that make part of the
space invalid, interactions between knobs, and (``rugged``) deterministic
multiplicative bumps so greedy local search can get stuck. The objective is
a "time" (lower is better); :func:`brute_force_optimum` finds the true best
by enumeration, which is cheap because no measurement is involved.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping

from .params import Param, SearchSpace

__all__ = ["brute_force_optimum", "gemm_like", "sphere"]


def _bump(config: Mapping[str, object], salt: str, amp: float) -> float:
    blob = repr(sorted(config.items())) + salt
    h = int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 1.0 + amp * h


def gemm_like(
    *, m: int = 4096, n: int = 4096, k: int = 4096, rugged: float = 0.0, salt: str = ""
) -> tuple[SearchSpace, Callable[[Mapping[str, object]], float]]:
    """``(space, objective)`` for a synthetic GEMM tuning problem."""
    params = (
        Param.ordinal("tile_m", (32, 64, 128, 256)),
        Param.ordinal("tile_n", (32, 64, 128, 256)),
        Param.ordinal("tile_k", (16, 32, 64)),
        Param.categorical("warps", ("1x4", "2x2", "4x1")),
        Param.categorical("pipeline", ("intrawave", "interwave", "compv4")),
        Param.boolean("prefetch"),
        Param.ordinal("prefetch_depth", (1, 2, 3), gate="prefetch"),
        Param.boolean("swizzle"),
        Param.ordinal("waves_per_eu", (1, 2, 3, 4)),
    )

    def is_valid(c: Mapping[str, object]) -> bool:
        tm, tn, tk = int(c["tile_m"]), int(c["tile_n"]), int(c["tile_k"])
        lds = (tm + tn) * tk * 2 * (2 if c["pipeline"] == "compv4" else 1)
        if lds > 64 * 1024:
            return False
        if tm * tn > 128 * 256:
            return False
        if c.get("prefetch_depth", 1) == 3 and tk > 32:
            return False
        return not (c["warps"] == "4x1" and tm < 64)

    flops = 2.0 * m * n * k

    def objective(c: Mapping[str, object]) -> float:
        tm, tn, tk = int(c["tile_m"]), int(c["tile_n"]), int(c["tile_k"])
        tiles = math.ceil(m / tm) * math.ceil(n / tn)
        waves = math.ceil(tiles / 256.0)
        util = tiles / (waves * 256.0)
        intensity = (tm * tn) / (tm + tn)
        eff = min(1.0, intensity / 64.0) * (0.55 + 0.45 * util)
        eff *= {16: 0.86, 32: 1.0, 64: 0.95}[tk]
        eff *= {"1x4": 0.92, "2x2": 1.0, "4x1": 0.95}[str(c["warps"])]
        pipe = str(c["pipeline"])
        eff *= {"intrawave": 0.93, "interwave": 0.9, "compv4": 1.0}[pipe]
        if c.get("prefetch"):
            depth = int(c.get("prefetch_depth", 1))
            gain = {"intrawave": 1.05, "interwave": 1.08, "compv4": 0.97}[pipe]
            eff *= gain if depth == 2 else (gain * 0.99 if depth == 1 else gain * 0.97)
        if c.get("swizzle"):
            eff *= 1.03 if tiles > 256 else 0.99
        wpe = int(c["waves_per_eu"])
        eff *= {1: 0.94, 2: 1.0, 3: 0.97, 4: 0.9}[wpe] if tm * tn >= 128 * 128 else 1.0
        t = flops / (eff * 1e15) * 1e3
        if rugged > 0:
            t *= _bump(dict(c), salt, rugged)
        return t

    space = SearchSpace(params, is_valid=is_valid, name=f"gemm_like_{m}x{n}x{k}")
    return space, objective


def sphere(levels: int = 9, dims: int = 4, *, optimum: tuple[int, ...] | None = None):
    """An ordinal bowl: ``sum (x_i - opt_i)^2 + 1``; optimum value 1."""
    opt = optimum or tuple((i * 3 + 2) % levels for i in range(dims))
    params = tuple(Param.ordinal(f"x{i}", tuple(range(levels))) for i in range(dims))

    def objective(c: Mapping[str, object]) -> float:
        return 1.0 + sum((int(c[f"x{i}"]) - opt[i]) ** 2 for i in range(dims))

    return SearchSpace(params, name=f"sphere{dims}x{levels}"), objective


def brute_force_optimum(space: SearchSpace, objective) -> tuple[str, float, int]:
    """``(key, value, valid count)`` of the true optimum by enumeration."""
    best_key, best, count, seen = None, math.inf, 0, set()
    for g in space.enumerate():
        key = space.key(g)
        if key is None or key in seen:
            continue
        seen.add(key)
        count += 1
        v = objective(space.decode(g))
        if v < best or (v == best and key < best_key):
            best_key, best = key, v
    return best_key, best, count
