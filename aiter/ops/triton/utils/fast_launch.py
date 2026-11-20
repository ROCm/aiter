# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch a Triton kernel without redoing the part that only depends on shapes.

``JITFunction.run`` binds the arguments, specializes each one, builds a cache
key and looks the kernel up, and it does all of that on every call. For a
short kernel that can cost more host time than the launch itself, and an eager
caller then waits on the host.

Wrap a kernel once and call it as before::

    _kernel = fast_launch(_kernel)
    _kernel[grid](x, y, n=..., BLOCK=..., num_warps=4)

The first launch of a key goes through ``kernel[grid]`` and keeps the compiled
kernel. A later launch with the same key calls its launcher directly. The key
holds the current device, the knobs ``JITFunction.run`` puts in its cache key
(``runtime.debug``, ``compilation.instrumentation_mode``,
``compilation.fpsan_homomorphic_casts`` on the Triton builds that have it), the
``amd.use_buffer_ops`` knob the HIP backend specializes tensors on, the number
of positional arguments, the keyword names, and for each argument:

* a tensor: its dtype, whether its data pointer is 16-byte aligned, and whether
  its storage fits in 2 GiB (``HIPBackend.get_tensor_specialization``); and,
  for a kernel under ``@triton.autotune`` or ``@triton.heuristics``, its shape
  and strides, which a ``@triton.heuristics`` function can read. An autotuner
  keys on the values of its ``key`` arguments and on dtypes, which the key
  already holds. ``fast_launch(kernel, key_shapes=False)`` leaves shapes and
  strides out, for a kernel whose heuristics read none. A heuristic may also
  read a tensor's dtype or 16-byte alignment; other properties (the full
  ``data_ptr()``, ``storage_offset()``, the values) are not keyed or supported;
* any other value: its type and value.

With ``fast_launch(kernel, by_specialization=True)``, a plain ``@triton.jit``
kernel keys its non-constexpr parameters the way Triton specializes them
instead of by value: an int by whether it is 1, else whether it is a multiple
of 16, and by its i32 / i64 / u64 range; a float or bool by its type. A caller
whose sizes change on every call then hits without a miss per size.

These launches go through ``kernel[grid]`` and are not cached: launches with a
launch-enter or launch-exit hook, a pre-run hook, or a stages-inspection hook;
launches under torch.compile tracing; launches after a global the kernel reads
was rebound to an unequal value (Triton raises for them); launches of an
autotuned config that has a ``pre_hook``; launches of a kernel built under
``TRITON_INTERPRET=1``; launches with a tuple argument or with an argument that
cannot be hashed, after which the wrapper stays on the ordinary path.

A hit checks those globals by identity against the objects seen at the last
full check, and runs Triton's equality check only after a rebind. A global
changed in place, which Triton's own check would catch, is therefore not
caught on a hit; that includes a list field of a Gluon layout.
"""

import contextlib
from collections import ChainMap, OrderedDict
from itertools import repeat
from operator import itemgetter
from typing import Any, NamedTuple

import torch
from triton import knobs
from triton.compiler import CompiledKernel
from triton.knobs import HookChain
from triton.runtime import driver
from triton.runtime.autotuner import Autotuner
from triton.runtime.interpreter import InterpretedFunction
from triton.runtime.jit import JITFunction

# 2 GiB - 1: the largest storage HIPBackend.is_within_2gb allows buffer ops on;
# with _MIN_INT32, also the i32 range Triton specializes a non-constexpr int on.
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


def _wide_int_spec(v: int) -> str:
    """The class Triton specializes a non-constexpr int outside i32 on."""
    if -(2**63) <= v < 2**63:
        width = "i64"
    elif 0 <= v < 2**64:
        width = "u64"
    else:
        raise OverflowError("int out of range for a Triton argument")
    return width if v & 15 else width + "D"


def _tensor_spec(v) -> tuple:
    """What `HIPBackend.get_tensor_specialization` reads off a tensor."""
    return (v.dtype, v.data_ptr() % 16 == 0, v.untyped_storage().nbytes() <= _MAX_INT32)


def _same_objects(groups) -> bool:
    """Whether every `globals_dict[name]` is still `obj`, for
    `groups = ((globals_dict, ((name, obj), ...)), ...)`."""
    try:
        for globals_dict, objs in groups:
            for name, obj in objs:
                if globals_dict[name] is not obj:
                    return False
    except KeyError:
        return False
    return True


def _picker(positions):
    """`values -> tuple(values[i] for i in positions)`."""
    if len(positions) > 1:
        return itemgetter(*positions)
    if positions:
        (i,) = positions
        return lambda values: (values[i],)
    return lambda values: ()


class _Entry(NamedTuple):
    """What a launch with a key it has seen needs."""

    compiled: CompiledKernel
    names: tuple
    # Parameters the call site leaves out (defaults, heuristic and autotuner choices).
    frozen: dict
    run: Any
    function: Any
    metadata: Any
    # The parameters after the call site's positional arguments.
    rest: tuple


class _FastLaunch:
    """A kernel that remembers what it compiled for a given key."""

    def __init__(
        self, kernel, by_specialization: bool = False, key_shapes: bool = True
    ):
        self._kernel = kernel
        self._cache = OrderedDict()
        self._disabled = False
        # The autotuner and any other decorator wrap the JITFunction that owns
        # the signature, the pre-run hooks and the globals the kernel reads.
        jit = kernel
        autotuners = []
        while not isinstance(jit, (JITFunction, InterpretedFunction)):
            if isinstance(jit, Autotuner):
                autotuners.append(jit)
            jit = jit.fn
        self._jit = jit
        self._autotuners = tuple(autotuners)
        self._wrapped = jit is not kernel
        if by_specialization and self._wrapped:
            raise ValueError(
                "by_specialization needs a plain @triton.jit kernel: a config "
                "chosen by @triton.autotune or @triton.heuristics can read the "
                "value of any argument"
            )
        if isinstance(jit, InterpretedFunction):
            # TRITON_INTERPRET=1: no compiled kernel to keep.
            self._disabled = True
            return
        self._names = tuple(jit.signature.parameters)
        self._defaults = tuple(
            (n, () if p.kind is p.VAR_POSITIONAL else p.default)
            for n, p in jit.signature.parameters.items()
        )
        self._by_spec = by_specialization
        self._key_shapes = self._wrapped and key_shapes
        runtime = [not p.is_constexpr for p in jit.params]
        self._runtime_at = tuple(runtime)
        self._runtime_named = {p.name: r for p, r in zip(jit.params, runtime)}
        # (number of positional arguments, *keyword names) -> `_pickers`.
        self._layouts = {}
        # Types of the constexpr arguments of a launch, seen without a tensor.
        self._tensorless = set()
        # Bound on the first launch, which assumes the active driver is not
        # swapped afterwards. The device and the stream are still asked for on
        # every launch.
        self._get_device = None
        self._get_stream = None
        # (`used_global_vals`, ((globals dict, ((name, object), ...)), ...)) as
        # of the last full check that passed.
        self._globals_seen = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # Before anything else: Dynamo traces this function, and branching
            # on the hooks, the cache or any other state of this wrapper is an
            # error under fullgraph=True.
            if torch.compiler.is_compiling():
                return self._kernel[grid](*args, **kwargs)
            # The ordinary path runs the hooks. An empty HookChain is no hook.
            runtime = knobs.runtime
            if (
                self._disabled
                or _BYPASS
                or self._jit.pre_run_hooks
                or runtime.add_stages_inspection_hook is not None
                or (
                    h.calls
                    if isinstance(h := runtime.launch_enter_hook, HookChain)
                    else h is not None
                )
                or (
                    h.calls
                    if isinstance(h := runtime.launch_exit_hook, HookChain)
                    else h is not None
                )
            ):
                return self._kernel[grid](*args, **kwargs)
            get_device = self._get_device
            if get_device is None:
                get_device = self._get_device = driver.active.get_current_device
                self._get_stream = driver.active.get_current_stream
            device = get_device()
            try:
                key = self._key(device, args, kwargs)
                entry = self._cache.get(key)
            except TypeError:
                # An argument that cannot be hashed cannot be guarded, and a
                # guard that silently omits an argument is the failure mode
                # this whole file exists to avoid.
                self._disabled = True
                return self._kernel[grid](*args, **kwargs)

            if entry is None or (
                self._jit.used_global_vals and self._globals_changed()
            ):
                return self._capture(grid, args, kwargs, key)
            self._cache.move_to_end(key)

            compiled, names, frozen, run, function, metadata, rest = entry
            # Rebuilt per call rather than captured: keeping the bound arguments
            # pinned every tensor of the call that created the entry, and handed
            # a shape-dependent grid the extents of a call already returned.
            if type(grid) is not tuple and callable(grid):
                g = grid(
                    ChainMap(dict(zip(names, args)), kwargs, frozen)
                    if args
                    else ChainMap(kwargs, frozen)
                )
            else:
                g = grid
            run(
                g[0],
                g[1] if len(g) > 1 else 1,
                g[2] if len(g) > 2 else 1,
                # Not hoisted: capture redirects work onto its own stream, so
                # the answer changes under torch.cuda.graph.
                self._get_stream(device),
                function,
                metadata,
                None,
                None,
                None,
                *args,
                *[kwargs[n] if n in kwargs else frozen[n] for n in rest],
            )
            return compiled

        return launch

    def _globals_changed(self) -> bool:
        """Whether a global the kernel read at compile time was rebound to an
        unequal value. Triton keeps deep copies, and comparing a Gluon layout
        with its copy costs far more than the rest of a hit, so the comparison
        runs only when a global is no longer the object seen last time."""
        used = self._jit.used_global_vals
        seen = self._globals_seen
        if seen is not None and seen[0] is used and _same_objects(seen[1]):
            return False
        groups = {}
        for (name, _), (val, globals_dict) in used.items():
            obj = globals_dict.get(name, _MISSING)
            if obj != val:
                return True
            groups.setdefault(id(globals_dict), (globals_dict, []))[1].append(
                (name, obj)
            )
        self._globals_seen = (
            used,
            tuple(
                (globals_dict, tuple(objs)) for globals_dict, objs in groups.values()
            ),
        )
        return False

    def _key(self, device, args, kwargs) -> tuple:
        """Flat: a tensor adds three or five items, any other argument two (a
        type and a value or class), so the layout is unambiguous. With
        by_specialization the constexpr arguments come right after the header,
        as one tuple of values (a tensor's `_tensor_spec` in place of the
        tensor) and one of their types; the runtime arguments follow."""
        n_args = len(args)
        key = [
            device,
            knobs.runtime.debug,
            knobs.compilation.instrumentation_mode,
            knobs.compilation.fpsan_homomorphic_casts if _FPSAN else None,
            knobs.amd.use_buffer_ops,
            n_args,
            *kwargs,
        ]
        append = key.append
        values = (*args, *kwargs.values()) if kwargs else args
        if not self._by_spec:
            key_shapes = self._key_shapes
            for v in values:
                if isinstance(v, _Tensor):
                    append(v.dtype)
                    append(v.data_ptr() % 16 == 0)
                    append(v.untyped_storage().nbytes() <= _MAX_INT32)
                    if key_shapes:
                        append(tuple(v.shape))
                        append(v.stride())
                    continue
                append(type(v))
                append(v)
            return tuple(key)

        layout = (n_args, *kwargs)
        pickers = self._layouts.get(layout)
        if pickers is None:
            pickers = self._layouts[layout] = self._pickers(n_args, kwargs)
        pick_runtime, pick_constexpr = pickers
        constexprs = pick_constexpr(values)
        types = tuple(map(type, constexprs))
        if types not in self._tensorless:
            if any(issubclass(t, _Tensor) for t in types):
                # By value a tensor would key on its identity and stay alive in
                # the cache.
                constexprs = tuple(
                    _tensor_spec(v) if isinstance(v, _Tensor) else v for v in constexprs
                )
            else:
                self._tensorless.add(types)
        append(constexprs)
        append(types)
        for v in pick_runtime(values):
            if isinstance(v, _Tensor):
                append(v.dtype)
                append(v.data_ptr() % 16 == 0)
                append(v.untyped_storage().nbytes() <= _MAX_INT32)
                continue
            cls = type(v)
            append(cls)
            if cls is int:
                # The common i32 case inline: is it 1, else is it aligned.
                if v == 1:
                    append("1")
                elif _MIN_INT32 <= v <= _MAX_INT32:
                    append(not v & 15)
                else:
                    append(_wide_int_spec(v))
            elif cls is float or cls is bool:
                append(None)
            else:
                append(v)
        return tuple(key)

    def _pickers(self, n_args: int, kwargs) -> tuple:
        """For a call-site layout, a picker of its runtime arguments and one of
        its constexpr arguments."""
        runtime = self._runtime(n_args, kwargs)
        return (
            _picker([i for i, r in enumerate(runtime) if r]),
            _picker([i for i, r in enumerate(runtime) if not r]),
        )

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
        if any(map(isinstance, (*args, *kwargs.values()), repeat(tuple))):
            # Triton specializes each item of a tuple, but the key holds the
            # tuple by value, and (2,) == (2.0,).
            self._disabled = True
            return self._kernel[grid](*args, **kwargs)

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
        _, kw = seen[-1]
        names = self._names
        # Defaults are fixed; autotuner and heuristic choices depend on the key
        # (by_specialization allows neither), so omitted arguments stay constant.
        frozen = {
            n: kw.get(n, default)
            for n, default in self._defaults[len(args) :]
            if n not in kwargs
        }
        self._cache[key] = _Entry(
            compiled,
            names,
            frozen,
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
            names[len(args) :],
        )
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)
        return compiled

    def _config_pre_hook(self) -> bool:
        """Whether an autotuner ran a config ``pre_hook`` the cached launch
        would skip."""
        return any(t.best_config.pre_hook is not None for t in self._autotuners)

    def clear(self):
        """Drop every entry, for a caller done with a device or a shape mix."""
        self._cache.clear()


def fast_launch(
    kernel, *, by_specialization: bool = False, key_shapes: bool = True
) -> _FastLaunch:
    """Wrap ``kernel`` so ``wrapped[grid](...)`` skips Triton's per-call
    dispatch for a key it has launched before.

    ``kernel`` is a ``@triton.jit`` or ``@gluon.jit`` kernel, optionally under
    ``@triton.autotune`` / ``@triton.heuristics``. ``by_specialization=True``
    keys the non-constexpr arguments of a plain kernel by Triton's
    specialization classes instead of by value; it raises ``ValueError`` for a
    wrapped kernel. ``key_shapes=False`` leaves tensor shapes and strides out
    of the key of a wrapped kernel; pass it only when no ``@triton.heuristics``
    function of the kernel reads a shape or stride. A heuristic may also read a
    tensor's dtype or 16-byte alignment; other properties (the full data pointer,
    storage offset, values) are not keyed or supported. Launches return what
    ``kernel[grid](...)`` returns.
    """
    return _FastLaunch(kernel, by_specialization, key_shapes)
