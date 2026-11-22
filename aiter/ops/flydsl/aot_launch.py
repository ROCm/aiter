# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route ``_run_compiled`` launches through ``FlyDSLAotCache`` without
changing their call sites.

A launcher factory opts in once, and every ``_run_compiled(exe, ...)`` of the
launcher it returns then goes through the cache::

    @aot_factory("moe_gfx942_gemm")
    @functools.cache
    def _get_compiled_kernel(N, K, ...):
        return compile_gemm(...)

``factory.lazy(...)`` takes the same arguments but does not call the factory:
it returns a handle that ``_run_compiled`` launches from the cache, so a hit
never runs the factory (whose ``@flyc.kernel`` definitions alone take a few
hundred ms in a fresh process). The factory runs only to build an entry or to
fall back to the JIT. Pass the handle to ``_run_compiled`` only.

A factory whose launcher is built inside a wrapper binds it where it is built::

    @functools.cache
    def sorted_sum(TOPK, N, output_padding_bytes=0):
        ...
        bind_aot(launch, _SORTED_SUM_AOT_CACHE, (TOPK, N, output_padding_bytes))

The cache key is the factory's arguments, so the factory's arguments must
determine the launcher, which ``functools.cache`` on the factory already
requires. The in-memory key is that key and ``_run_compiled``'s
``specialization_key``, as on the JIT path, which keeps one compiled function
per launcher.

Under ``AITER_FLYDSL_AOT_CACHE=0`` the launch takes ``_run_compiled``'s JIT
path unchanged, and calling the launcher directly is unaffected.
"""

from __future__ import annotations

import functools
import inspect
import threading

import flydsl.expr as fx
import torch

from aiter import logger

from .aot_cache import FlyDSLAotCache, _normalize_key, _UsageError, aot_cache_mode

__all__ = ["aot_factory", "bind_aot"]

_ATTR = "_aiter_aot"


class _Inside(threading.local):
    # Set while FlyDSLAotCache runs a launch. Its JIT fallback calls
    # _run_compiled on the launcher make_exe returns, which may be bound.
    active = False


_inside = _Inside()


class _AotBinding:
    __slots__ = ("cache", "key", "launcher", "make_exe")

    def __init__(self, cache: FlyDSLAotCache, key, make_exe, launcher):
        self.cache = cache
        self.key = key
        # Builds the launcher for the cache to trace.
        self.make_exe = make_exe
        # The launcher _run_compiled's JIT path runs when this returns False.
        self.launcher = launcher

    def __call__(self, args, specialization_key) -> bool:
        """Launch through the cache. False leaves the launch to the JIT path."""
        if _inside.active:
            return False
        key = self.key if specialization_key is None else (self.key, specialization_key)
        # FlyDSLAotCache keys launch arguments by their FlyDSL type; a torch
        # stream has none. aiter launchers take the stream last, so only that
        # argument is checked (a stream elsewhere falls back to the JIT).
        if args and isinstance(args[-1], torch.cuda.Stream):
            args = (*args[:-1], fx.Stream(args[-1]))
        _inside.active = True
        try:
            self.cache.launch(key, self.make_exe, *args)
        finally:
            _inside.active = False
        return True


def _storable(cache: FlyDSLAotCache, key) -> bool:
    """Whether the cache can key ``key`` on disk. A key it cannot (e.g. a numpy
    scalar or an Enum among the factory's arguments) stays on the JIT path
    instead of failing every launch."""
    try:
        _normalize_key(key)
    except _UsageError as e:
        logger.info(f"[FlyDSL AOT] {cache.name}: {e}; launching through the JIT path")
        return False
    return True


def bind_aot(exe, cache: FlyDSLAotCache, key, make_exe=None):
    """Launch ``exe`` through ``cache`` under ``key`` from ``_run_compiled``.

    ``key`` must determine ``exe`` within ``cache``. ``make_exe`` builds the
    launcher when the cache traces it (by default ``exe`` itself). Returns
    ``exe``.
    """
    if aot_cache_mode() == "0":
        return exe
    current = getattr(exe, _ATTR, None)
    if current is None:
        if not _storable(cache, key):
            setattr(exe, _ATTR, False)
            return exe

        def launcher():
            return exe

        setattr(exe, _ATTR, _AotBinding(cache, key, make_exe or launcher, launcher))
    elif current is not False and (current.cache is not cache or current.key != key):
        # One launcher object under two keys: the key does not determine it.
        logger.warning(
            f"[FlyDSL AOT] {cache.name}: one launcher bound to keys "
            f"{current.key!r} and {key!r}; launching it through the JIT path"
        )
        setattr(exe, _ATTR, False)
    return exe


class _LazyLauncher:
    """A launcher not built yet, for ``_run_compiled`` only."""

    __slots__ = (_ATTR,)

    def __init__(self, binding: _AotBinding):
        setattr(self, _ATTR, binding)


def aot_factory(name: str, *, extra_sources=()):
    """Decorate a launcher factory so that the launchers it returns launch
    through a ``FlyDSLAotCache`` named ``name``, keyed by its arguments.

    Place it above ``functools.cache``. Under ``AITER_FLYDSL_AOT_CACHE=0`` it
    returns the factory itself, with ``lazy`` the factory.
    """
    cache = FlyDSLAotCache(name, extra_sources)

    def decorate(factory):
        if aot_cache_mode() == "0":
            factory.aot_cache = cache
            factory.lazy = factory
            return factory
        signature = inspect.signature(inspect.unwrap(factory))
        var_keyword = {
            name
            for name, p in signature.parameters.items()
            if p.kind is inspect.Parameter.VAR_KEYWORD
        }
        # Calls whose key is checked, and lazy handles, by call: the factory
        # runs on every launch of some callers, so a key is built once per
        # call. Calls with and without keyword arguments are kept apart.
        seen, seen_kw = set(), set()
        handles, handles_kw = {}, {}

        def key_of(bound):
            return tuple(
                (name, tuple(sorted(value.items())) if name in var_keyword else value)
                for name, value in bound.arguments.items()
            )

        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            exe = factory(*args, **kwargs)
            if kwargs:
                calls, call = seen_kw, (args, tuple(kwargs.items()))
            else:
                calls, call = seen, args
            # A bounded factory cache may hand out a new launcher for a call
            # already seen: bind every launcher not yet bound.
            if call not in calls or getattr(exe, _ATTR, None) is None:
                calls.add(call)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                bind_aot(exe, cache, key_of(bound))
            return exe

        def lazy(*args, **kwargs):
            if kwargs:
                calls, call = handles_kw, (args, tuple(kwargs.items()))
            else:
                calls, call = handles, args
            handle = calls.get(call)
            if handle is None:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                key = key_of(bound)
                if not _storable(cache, key):
                    handle = calls[call] = wrapper(*args, **kwargs)
                    return handle
                # The cached factory, as the JIT path would call it: an AOT
                # replay that only identifies kernels relies on one launcher
                # per key to skip repeated traces.
                make_exe = functools.partial(factory, *args, **kwargs)
                handle = calls[call] = _LazyLauncher(
                    _AotBinding(cache, key, make_exe, make_exe)
                )
            return handle

        wrapper.aot_cache = cache
        wrapper.lazy = lazy
        if hasattr(factory, "cache_info"):
            wrapper.cache_info = factory.cache_info
        if hasattr(factory, "cache_clear"):

            def cache_clear():
                # Resets the factory only: an entry already launched, like one
                # on disk, stays keyed by the factory's arguments, so state a
                # launcher bakes in beyond them must be part of the key.
                for calls in (seen, seen_kw, handles, handles_kw):
                    calls.clear()
                factory.cache_clear()

            wrapper.cache_clear = cache_clear
        return wrapper

    return decorate
