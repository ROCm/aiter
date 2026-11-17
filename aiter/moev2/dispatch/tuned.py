# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned-table lookup (design doc §3.5, §5).

Every cache in v2 dispatch that depends on tuned tables, environment variables or the
arch is declared with `@cached`, so `clear_cache()` resets all of them at once; tests call
it after changing AITER_CONFIG_FMOE or patching the arch.
"""

import functools

_CACHES: list = []


def cached(fn=None, *, maxsize=None):
    """`functools.lru_cache` that `clear_cache()` also clears."""

    def wrap(f):
        wrapped = functools.lru_cache(maxsize=maxsize)(f)
        _CACHES.append(wrapped)
        return wrapped

    return wrap(fn) if fn is not None else wrap


def clear_cache():
    """Clear every v2 dispatch cache and re-read the moev2 environment variables."""
    from .. import envs

    for cache in _CACHES:
        cache.cache_clear()
    envs.reload()
