# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch a Triton kernel without redoing the part that only depends on shapes.

``JITFunction.run`` binds the arguments, specializes each one, builds a cache
key and looks the kernel up, and it does all of that on every call. On the
flash_kda pipeline that is several times the cost of the launch itself, which
is enough to decide a design question the wrong way: an eager caller waits on
the host, and a faster pipeline of more launches measures as the slower one.

Wrap a kernel once and call it as before::

    _kernel = fast_launch(_kernel)
    _kernel[grid](x, y, n=..., BLOCK=..., num_warps=4)

The first launch of a key goes through ``kernel[grid]`` and keeps the compiled
kernel. A later launch with the same key calls its launcher directly. The key
holds the current device, the knobs that enter Triton's compile options or
specialization (``runtime.debug``, ``compilation.instrumentation_mode``,
``compilation.fpsan_homomorphic_casts`` on the Triton builds that have it,
``amd.use_buffer_ops``), the number of positional arguments, the keyword names,
and for each argument:

* a tensor: its dtype, whether its data pointer is 16-byte aligned, and whether
  its storage fits in 2 GiB (``HIPBackend.get_tensor_specialization``); and,
  for a kernel under ``@triton.autotune`` or ``@triton.heuristics``, its shape
  and strides, which a config choice can read;
* any other value: its type and value.

With ``fast_launch(kernel, by_specialization=True)``, a plain ``@triton.jit``
kernel keys its non-constexpr parameters the way Triton specializes them
instead of by value: an int by whether it is 1, else whether it is a multiple
of 16, and by its i32 / i64 / u64 range; a float or bool by its type. A caller
whose sizes change on every call then hits without a miss per size.

These launches go through ``kernel[grid]`` and are not cached: launches with a
launch-enter or launch-exit hook, a pre-run hook, or a stages-inspection hook;
launches under torch.compile tracing; launches whose checked global values
changed since the compile (Triton raises for them); launches of an autotuned
config that has a ``pre_hook``; launches with an argument that cannot be
hashed, after which the wrapper stays on the ordinary path.
"""

import contextlib
from collections import ChainMap, OrderedDict

import torch
from triton import knobs
from triton.compiler import CompiledKernel
from triton.runtime import driver
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction

# `HIPBackend.is_within_2gb`: the storage size up to which buffer ops apply,
# and the i32 range of an int argument.
_MAX_INT32 = 2**31 - 1
_MIN_INT32 = -(2**31)

# Entries per wrapped kernel, past which the least recently used one goes. A
# miss is an ordinary Triton dispatch that still hits Triton's own compile cache,
# so eviction costs the binding this file skips and never a recompile.
_MAX_ENTRIES = 1024

# Set to record every key miss, which a test uses to prove the guard is what
# separates two shapes rather than luck.
_MISSES: list | None = None

# Set while a test wants the ordinary path, to compare this one against.
_BYPASS = False

# Not every supported Triton build has this knob, and a build without it does
# not key on it.
_FPSAN = "fpsan_homomorphic_casts" in knobs.compilation.knob_descriptors

_MISSING = object()
_Tensor = torch.Tensor


@contextlib.contextmanager
def bypassed():
    """Route every launch the ordinary way, for tests to compare against."""
    global _BYPASS
    _BYPASS, was = True, _BYPASS
    try:
        yield
    finally:
        _BYPASS = was


@contextlib.contextmanager
def recording():
    """Collect ``(kernel, key)`` for each key this block compiles for."""
    global _MISSES
    _MISSES, was = [], _MISSES
    try:
        yield _MISSES
    finally:
        _MISSES = was


def _hooked(hook) -> bool:
    """Whether a launch hook would run. An empty ``HookChain`` does not."""
    if hook is None:
        return False
    calls = getattr(hook, "calls", None)
    return calls is None or len(calls) > 0


def _wide_int_spec(v: int) -> str:
    """The class Triton specializes a non-constexpr int outside i32 on."""
    if -(2**63) <= v < 2**63:
        width = "i64"
    elif 0 <= v < 2**64:
        width = "u64"
    else:
        raise TypeError("int out of range for a Triton argument")
    return width if v & 15 else width + "D"


class _FastLaunch:
    """A kernel that remembers what it compiled for a given key."""

    def __init__(self, kernel, by_specialization: bool = False):
        self._kernel = kernel
        self._cache = OrderedDict()
        # The autotuner and any other decorator wrap the JITFunction that owns
        # the argument binder and the hooks.
        jit = kernel
        while not isinstance(jit, JITFunction):
            jit = jit.fn
        self._jit = jit
        self._wrapped = jit is not kernel
        if by_specialization and self._wrapped:
            raise ValueError(
                "by_specialization needs a plain @triton.jit kernel: a config "
                "chosen by @triton.autotune or @triton.heuristics can read the "
                "value of any argument"
            )
        self._by_spec = by_specialization
        runtime = [not p.is_constexpr for p in jit.params]
        self._runtime_at = tuple(runtime)
        self._runtime_named = {p.name: r for p, r in zip(jit.params, runtime)}
        self._disabled = False

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # Before anything else: Dynamo traces this function, and branching
            # on the hooks, the cache or any other state of this wrapper is an
            # error under fullgraph=True.
            if torch.compiler.is_compiling():
                return self._kernel[grid](*args, **kwargs)
            if self._disabled or _BYPASS or not self._plain():
                return self._kernel[grid](*args, **kwargs)
            device = driver.active.get_current_device()
            try:
                key = self._key(device, args, kwargs)
                entry = self._cache.get(key)
            except TypeError:
                # An argument that cannot be hashed cannot be guarded, and a
                # guard that silently omits an argument is the failure mode
                # this whole file exists to avoid.
                self._disabled = True
                return self._kernel[grid](*args, **kwargs)

            if entry is None or self._globals_changed():
                return self._capture(grid, args, kwargs, key)
            self._cache.move_to_end(key)

            compiled, names, frozen = entry
            rest = names[len(args) :]
            # Rebuilt per call rather than captured: keeping the bound arguments
            # pinned every tensor of the call that created the entry, and handed
            # a shape-dependent grid the extents of a call already returned.
            if callable(grid):
                g = grid(ChainMap(dict(zip(names, args)), kwargs, frozen))
            else:
                g = grid
            # Not hoisted: capture redirects work onto its own stream, so the
            # answer changes under torch.cuda.graph.
            stream = driver.active.get_current_stream(device)
            compiled.run(
                g[0],
                g[1] if len(g) > 1 else 1,
                g[2] if len(g) > 2 else 1,
                stream,
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                *[kwargs[n] if n in kwargs else frozen[n] for n in rest],
            )
            return compiled

        return launch

    def _plain(self) -> bool:
        """Whether nothing needs the ordinary path's side effects."""
        return not (
            self._jit.pre_run_hooks
            or knobs.runtime.add_stages_inspection_hook is not None
            or _hooked(knobs.runtime.launch_enter_hook)
            or _hooked(knobs.runtime.launch_exit_hook)
        )

    def _globals_changed(self) -> bool:
        """Whether a global the kernel read at compile time has changed."""
        for (name, _), (val, globals_dict) in self._jit.used_global_vals.items():
            if globals_dict.get(name, _MISSING) != val:
                return True
        return False

    def _key(self, device, args, kwargs) -> tuple:
        """Flat: a tensor adds three or five items, any other argument two (a
        type and a value or class), so the layout is unambiguous."""
        buffer_ops = knobs.amd.use_buffer_ops
        key = [
            device,
            knobs.runtime.debug,
            knobs.compilation.instrumentation_mode,
            knobs.compilation.fpsan_homomorphic_casts if _FPSAN else None,
            buffer_ops,
            len(args),
            *kwargs,
        ]
        append = key.append
        wrapped = self._wrapped
        runtime = self._runtime(len(args), kwargs) if self._by_spec else None
        for i, v in enumerate(args + tuple(kwargs.values()) if kwargs else args):
            if isinstance(v, _Tensor):
                append(v.dtype)
                append(v.data_ptr() % 16 == 0)
                append(v.untyped_storage().nbytes() <= _MAX_INT32)
                if wrapped:
                    append(tuple(v.shape))
                    append(v.stride())
                continue
            cls = type(v)
            append(cls)
            if runtime is not None and runtime[i]:
                if cls is int:
                    # The common i32 case inline: is it 1, else is it aligned.
                    if v == 1:
                        append("1")
                    elif _MIN_INT32 <= v <= _MAX_INT32:
                        append(not v & 15)
                    else:
                        append(_wide_int_spec(v))
                    continue
                if cls is float or cls is bool:
                    append(None)
                    continue
            append(v)
        return tuple(key)

    def _runtime(self, n_args: int, kwargs) -> tuple:
        """Per argument of a launch, whether it is a non-constexpr parameter."""
        at = self._runtime_at[:n_args]
        at += (False,) * (n_args - len(at))
        if not kwargs:
            return at
        return at + tuple(self._runtime_named.get(n, False) for n in kwargs)

    def _capture(self, grid, args, kwargs, key):
        """Launch the ordinary way, and keep what it worked out."""
        if _MISSES is not None:
            _MISSES.append((self._jit.__name__, key))

        seen = []

        def hook(*a, **kw):
            seen.append((a, kw))

        self._jit.pre_run_hooks.append(hook)
        try:
            compiled = self._kernel[grid](*args, **kwargs)
        finally:
            self._jit.pre_run_hooks.remove(hook)
        if not isinstance(compiled, CompiledKernel) or not seen:
            return compiled
        if self._config_pre_hook():
            return compiled

        # Autotuning launches once per candidate config; the last one is the
        # launch that used the config it settled on.
        a, kw = seen[-1]
        device = driver.active.get_current_device()
        binder = self._jit.device_caches[device][4]
        bound_args, _, _ = binder(*a, **kw)

        names = tuple(bound_args.keys())
        # Arguments the call site does not pass -- defaults, and whatever the
        # autotuner added. The key pins every non-tensor value, so these are
        # constant for as long as this entry is the one selected.
        frozen = {
            n: v for n, v in list(bound_args.items())[len(args) :] if n not in kwargs
        }
        self._cache[key] = (compiled, names, frozen)
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)
        return compiled

    def _config_pre_hook(self) -> bool:
        """Whether an autotuner ran a config ``pre_hook`` the cached launch
        would skip."""
        k = self._kernel
        while k is not self._jit:
            if isinstance(k, Autotuner) and k.best_config.pre_hook is not None:
                return True
            k = k.fn
        return False

    def clear(self):
        """Drop every entry, for a caller done with a device or a shape mix."""
        self._cache.clear()


def fast_launch(kernel, *, by_specialization: bool = False) -> _FastLaunch:
    """Wrap ``kernel`` so ``wrapped[grid](...)`` skips Triton's per-call
    dispatch for a key it has launched before.

    ``kernel`` is a ``@triton.jit`` or ``@gluon.jit`` kernel, optionally under
    ``@triton.autotune`` / ``@triton.heuristics``. ``by_specialization=True``
    keys the non-constexpr arguments of a plain kernel by Triton's
    specialization classes instead of by value; it raises ``ValueError`` for a
    wrapped kernel. Launches return what ``kernel[grid](...)`` returns.
    """
    return _FastLaunch(kernel, by_specialization)
