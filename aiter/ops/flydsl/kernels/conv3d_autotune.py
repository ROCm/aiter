# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# ruff: noqa: B023, BLE001, S110
# B023: the do_bench lambda closes over the loop's `tile`, but do_bench consumes
# it inside the same iteration, so the late binding is never observed.
# BLE001/S110: every blind except here is a deliberate degradation path -- a
# corrupt cache file, an absent flydsl attribute, or a config that fails to
# compile must all fall through to "no cached answer" rather than propagate.

"""In-process tile autotuner for the implicit-GEMM conv3d kernels.

Upstream: FlyDSL ``kernels/conv/conv3d_autotune.py``. Vendored here because
``conv3d_implicit.py`` imported it as ``kernels.conv.conv3d_autotune`` -- a
FlyDSL-repo-relative path with no counterpart in aiter -- so both ``autotune=True``
and ``FLYDSL_CONV3D_AUTOTUNE=1`` raised ImportError instead of tuning.

This is the opt-in, per-process path: it benchmarks candidates on first call and
caches the winner in memory and in JSON under ``~/.flydsl/autotune``. The offline
counterpart that writes a checked-in CSV is ``csrc/flydsl_conv3d/conv3d_tune.py``;
prefer that one for anything that ships.
"""

import json
import os
from pathlib import Path

from flydsl.autotune import do_bench

__all__ = [
    "BF16_CANDIDATES",
    "FP8_CANDIDATES",
    "WGM_VALUES",
    "autotune_conv3d",
]


def _device_fingerprint():
    """GPU arch string (e.g. 'gfx950'), or '' if unavailable."""
    try:
        from flydsl.runtime.device import get_rocm_arch

        return str(get_rocm_arch())
    except Exception:
        return ""


def _toolchain_fingerprint():
    """FlyDSL version, so a rebuild invalidates stale cached configs."""
    try:
        import flydsl

        return str(getattr(flydsl, "__version__", ""))
    except Exception:
        return ""


# Legacy fixed candidate table, kept so that callers passing an explicit
# candidate list keep working. The shape-aware enumeration in
# ``aiter/ops/flydsl/conv3d_policy.py`` supersedes it: this table has no N tile
# that is not a power of two, so it cannot express an exact fit for the 96- and
# 192-channel rungs of the Qwen-Image / Wan VAE ladders.
BF16_CANDIDATES = [
    (128, 128, 2, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 4),
    (256, 256, 2, 4),
    (256, 256, 4, 4),
    (128, 128, 4, 2),
    (64, 128, 1, 4),
    (64, 64, 2, 2),
]

FP8_CANDIDATES = [
    (128, 128, 2, 4),
    (128, 256, 2, 4),
    (256, 128, 2, 4),
    (256, 256, 2, 4),
    (128, 128, 4, 2),
    (64, 128, 1, 4),
]

WGM_VALUES = [1, 4, 8]

_MEM_CACHE = {}
_CACHE_SCHEMA_VERSION = 3


def _cache_dir():
    return Path(
        os.environ.get(
            "FLYDSL_AUTOTUNE_CACHE_DIR", os.path.expanduser("~/.flydsl/autotune")
        )
    )


def _cache_file(kind):
    return _cache_dir() / f"conv3d_{kind}.json"


def _make_key(kind, shape, dtype_str, candidates):
    def _canon(c):
        if isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], tuple):
            return (tuple(c[0]), c[1])
        return tuple(c)

    return (
        kind,
        tuple(shape),
        dtype_str,
        _device_fingerprint(),
        _toolchain_fingerprint(),
        _CACHE_SCHEMA_VERSION,
        tuple(_canon(c) for c in candidates),
    )


def _load_disk(kind, key):
    f = _cache_file(kind)
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except Exception:
        return None
    ent = data.get(json.dumps(list(key)))
    if ent is None:
        return None
    if isinstance(ent[0], list):
        return (tuple(ent[0]), ent[1])
    return tuple(ent)


def _save_disk(kind, key, best):
    f = _cache_file(kind)
    f.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if f.exists():
        try:
            data = json.loads(f.read_text())
        except Exception:
            data = {}
    if isinstance(best, tuple) and len(best) == 2 and isinstance(best[0], tuple):
        data[json.dumps(list(key))] = [list(best[0]), best[1]]
    else:
        data[json.dumps(list(key))] = list(best)
    f.write_text(json.dumps(data, indent=2))


def autotune_conv3d(
    kind, shape, dtype_str, candidates, device, run_tile, warmup=5, rep=20
):
    """Return the best candidate for this problem shape.

    Each element of ``candidates`` is either a tile tuple (TILE_M, TILE_N, WAVE_M, WAVE_N)
    or a (tile_tuple, wgm) pair when wgm sweep is enabled. ``run_tile(candidate)``
    must launch one full conv for the given candidate and return the output tensor.
    The winning candidate is cached (in-memory + JSON on disk).

    No correctness check is performed here: a candidate that compiles and runs is
    timed, and the fastest wins. The offline tuner gates on ``errRatio`` instead.
    """
    key = _make_key(kind, shape, dtype_str, candidates)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    disk = _load_disk(kind, key)
    if disk is not None:
        _MEM_CACHE[key] = disk
        return disk

    results = []
    for tile in candidates:
        try:
            run_tile(
                tile
            )  # dry run: triggers compile (lru_cache miss) outside do_bench
            t = do_bench(lambda: run_tile(tile), warmup=warmup, rep=rep)
            results.append((tile, t))
        except Exception:
            pass  # skip illegal / register-spilling / OOM configs
    if not results:
        raise RuntimeError(
            f"all conv3d {kind} autotune configs failed for shape {shape}"
        )

    best = min(results, key=lambda x: x[1])[0]
    _MEM_CACHE[key] = best
    _save_disk(kind, key, best)
    return best
