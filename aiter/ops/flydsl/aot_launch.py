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
fall back to the JIT. Pass the handle to ``tensor_shim``'s ``_run_compiled``,
``_preload_compiled`` or ``_compile_and_keep`` only.

A factory whose launcher is built inside a wrapper binds it where it is built::

    @functools.cache
    def sorted_sum(TOPK, N, output_padding_bytes=0):
        ...
        bind_aot(launch, _SORTED_SUM_AOT_CACHE, (TOPK, N, output_padding_bytes))

The cache key is the factory's arguments, so the factory's arguments must
determine the launcher, which ``functools.cache`` on the factory already
requires. The in-memory key is that key and ``_run_compiled``'s
``specialization_key``, as on the JIT path, which keeps one compiled function
per launcher. A ``specialization_key`` the cache cannot key on disk (an
Enum, a numpy scalar) launches through the JIT path.

A launcher's ``Constexpr`` launch arguments are compiled into the object, so
each must be in ``specialization_key``, as on the JIT path. An entry on disk
records their values and serves only calls that pass them; one key launched
with two values raises.

``factory.cache_clear()`` also drops the launchers the cache loaded and keeps
the factory on the JIT path for the rest of the process: a caller that resets
its factory wants a new trace, of state the cache may not track.

Under ``AITER_FLYDSL_AOT_CACHE=0`` the launch takes ``_run_compiled``'s JIT
path unchanged, and calling the launcher directly is unaffected.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading

import flydsl.expr as fx
import torch

from aiter import logger

from .aot_cache import (
    FlyDSLAotCache,
    _JitKernel,
    _normalize_key,
    _prebuilding,
    _UsageError,
    aot_cache_mode,
)

__all__ = ["aot_factory", "bind_aot", "preload_bound"]

_ATTR = "_aiter_aot"


class _Inside(threading.local):
    # Set while FlyDSLAotCache runs a launch. Its JIT fallback calls
    # _run_compiled on the launcher make_exe returns, which may be bound.
    active = False


_inside = _Inside()


class _AotBinding:
    __slots__ = (
        "cache",
        "epoch",
        "kernels",
        "key",
        "launcher",
        "make_exe",
        "unkeyable",
    )

    def __init__(self, cache: FlyDSLAotCache, key, make_exe, launcher):
        self.cache = cache
        self.key = key
        # Builds the launcher for the cache to trace.
        self.make_exe = make_exe
        # The launcher _run_compiled's JIT path runs when this returns False.
        self.launcher = launcher
        # Specialization keys the cache cannot key on disk; they launch
        # through the JIT path, like a factory argument it cannot key.
        self.unkeyable = set()
        # The objects the cache loaded for this launcher, by specialization
        # key, while the cache's epoch is the one they were loaded in: a
        # launch then hashes the specialization key, not the whole key.
        self.kernels = {}
        self.epoch = cache.epoch

    def __call__(self, args, specialization_key) -> bool:
        """Launch through the cache. False leaves the launch to the JIT path."""
        if _inside.active:
            return False
        # FlyDSLAotCache keys launch arguments by their FlyDSL type; a torch
        # stream has none. aiter launchers take the stream last, so only that
        # argument is checked (a stream elsewhere falls back to the JIT).
        if args and isinstance(args[-1], torch.cuda.Stream):
            args = (*args[:-1], fx.Stream(args[-1]))
        cache = self.cache
        kernel = self.kernels.get(specialization_key)
        if (
            kernel is not None
            and self.epoch == cache.epoch
            and "COMPILE_ONLY" not in os.environ
        ):
            if kernel.__class__ is _JitKernel:
                # No entry for this key: it calls _run_compiled on the
                # launcher, which comes back here.
                _inside.active = True
                try:
                    kernel(*args)
                finally:
                    _inside.active = False
            else:
                kernel(*args)
            return True
        if specialization_key is None:
            key = self.key
        elif self.unkeyable and specialization_key in self.unkeyable:
            return False
        else:
            key = (self.key, specialization_key)
        # Read before the launch: what is kept below belongs to this epoch
        # even if another thread invalidates the cache meanwhile.
        epoch = cache.epoch
        if self.epoch != epoch:
            self.kernels.clear()
            self.epoch = epoch
        _inside.active = True
        try:
            cache.launch(key, self.make_exe, *args)
        except _UsageError as e:
            # Raised when the key is hashed for the disk, before any build or
            # launch. self.key was checked when it was bound.
            if specialization_key is None:
                raise
            logger.error(
                f"[FlyDSL AOT] {self.cache.name}: {e} in specialization_key; "
                f"launching it through the JIT path"
            )
            self.unkeyable.add(specialization_key)
            return False
        finally:
            _inside.active = False
        kernel = cache._mem.get(key)
        # Also the JIT kernel of a key with no entry, so that its launches
        # skip the lookup too; compile-only leaves nothing to keep.
        if kernel is not None and self.epoch == epoch == cache.epoch:
            self.kernels[specialization_key] = kernel
        return True


def _storable(cache: FlyDSLAotCache, key) -> bool:
    """Whether the cache can key ``key`` on disk. A key it cannot (e.g. a numpy
    scalar or an Enum among the factory's arguments) stays on the JIT path
    instead of failing every launch."""
    try:
        _normalize_key(key)
    except _UsageError as e:
        logger.error(f"[FlyDSL AOT] {cache.name}: {e}; launching through the JIT path")
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
        try:
            setattr(exe, _ATTR, False)
        except (AttributeError, TypeError):
            # Not one launcher (a factory returning a tuple or a dict of them):
            # nothing to bind, so its launchers take the JIT path. Bind each
            # with bind_aot where it is built.
            logger.error(
                f"[FlyDSL AOT] {cache.name}: cannot bind a "
                f"{type(exe).__name__}; launching through the JIT path"
            )
            return exe
        if not _storable(cache, key):
            return exe

        def launcher():
            return exe

        setattr(exe, _ATTR, _AotBinding(cache, key, make_exe or launcher, launcher))
    elif current is not False and (current.cache is not cache or current.key != key):
        # One launcher object under two keys: the key does not determine it.
        logger.error(
            f"[FlyDSL AOT] {cache.name}: one launcher bound to keys "
            f"{current.key!r} and {key!r}; launching it through the JIT path"
        )
        setattr(exe, _ATTR, False)
    return exe


def preload_bound(exe, args, specialization_key=None):
    """Load the entry of a bound launcher into this process without launching
    it, for ``tensor_shim._preload_compiled`` and a caller that keeps the
    function it gets and calls it directly.

    Returns the loaded function, which takes the launcher's arguments (a
    torch stream too); later launches of the key through ``_run_compiled``
    use it as well. None if ``exe`` is not bound, in a prebuild (which builds
    entries by launching: ``_run_compiled``), while ``COMPILE_ONLY`` is on
    for another reason (a preload of another thread: nothing is loaded
    then), or when the key has no entry: the cache then logs one error and
    the key takes the JIT path, where the caller preloads or compiles it.
    """
    binding = getattr(exe, _ATTR, None)
    if not binding or _prebuilding():
        return None
    # FlyDSLAotCache keys launch arguments by their FlyDSL type (see
    # _AotBinding); the function it returns takes a torch stream as is.
    if args and isinstance(args[-1], torch.cuda.Stream):
        args = (*args[:-1], fx.Stream(args[-1]))
    if specialization_key is None:
        key = binding.key
    elif specialization_key in binding.unkeyable:
        return None
    else:
        key = (binding.key, specialization_key)
    try:
        return binding.cache.preload(key, binding.make_exe, *args)
    except _UsageError as e:
        if specialization_key is None:
            raise
        logger.error(
            f"[FlyDSL AOT] {binding.cache.name}: {e} in specialization_key; "
            f"launching it through the JIT path"
        )
        binding.unkeyable.add(specialization_key)
        return None


class _LazyLauncher:
    """A launcher not built yet, for ``_run_compiled`` only."""

    __slots__ = (_ATTR,)

    def __init__(self, binding: _AotBinding):
        setattr(self, _ATTR, binding)


def aot_factory(name: str, *, sources=None, env=None):
    """Decorate a launcher factory so that the launchers it returns launch
    through a ``FlyDSLAotCache`` named ``name``, keyed by its arguments.

    What the cache is built from (source files, environment variables) is
    declared under ``name`` in ``aiter/aot/flydsl/sources.py``; ``sources``
    and ``env`` are for a cache that is not in that table (see
    ``FlyDSLAotCache``).

    Place it above ``functools.cache``. Under ``AITER_FLYDSL_AOT_CACHE=0`` it
    returns the factory itself, with ``lazy`` the factory.
    """
    cache = FlyDSLAotCache(name, sources, env)

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
        # Calls whose result cannot be bound (not one launcher).
        unbound = set()

        def key_of(args, kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return tuple(
                (name, tuple(sorted(value.items())) if name in var_keyword else value)
                for name, value in bound.arguments.items()
            )

        def bind(exe, args, kwargs):
            call = (args, tuple(kwargs.items()))
            if call not in unbound:
                bind_aot(exe, cache, key_of(args, kwargs))
                if getattr(exe, _ATTR, None) is None:
                    unbound.add(call)

        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            exe = factory(*args, **kwargs)
            # Some callers run the factory on every launch: a launcher already
            # bound costs one attribute read. A launcher is what it is under
            # any arguments that return it, so its first key stands.
            if getattr(exe, _ATTR, None) is None:
                bind(exe, args, kwargs)
            return exe

        # Handles by call, built as the factory's own cache keys its calls.
        @functools.cache
        def lazy(*args, **kwargs):
            key = key_of(args, kwargs)
            if not _storable(cache, key):
                return wrapper(*args, **kwargs)
            # The cached factory, as the JIT path would call it: an AOT
            # replay that only identifies kernels relies on one launcher
            # per key to skip repeated traces.
            make_exe = functools.partial(factory, *args, **kwargs)
            return _LazyLauncher(_AotBinding(cache, key, make_exe, make_exe))

        wrapper.aot_cache = cache
        wrapper.lazy = lazy
        if hasattr(factory, "cache_info"):
            wrapper.cache_info = factory.cache_info
        if hasattr(factory, "cache_clear"):

            def cache_clear():
                # The caller wants its launchers traced again, so state the
                # cache does not track may have changed: drop the launchers
                # it loaded and keep this factory on the JIT path.
                unbound.clear()
                lazy.cache_clear()
                factory.cache_clear()
                cache.invalidate()

            wrapper.cache_clear = cache_clear
        return wrapper

    return decorate
