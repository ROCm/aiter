# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Low-overhead relaunch of Triton / Gluon kernels.

JITFunction.run re-derives every argument's specialization on each launch
(alignment, the AMD buffer-ops "within 2 GB" storage check, int flags) before
it finds the compiled kernel: ~25-35 us of host time for a 30-40 argument
kernel on MI355X, several times the ~7 us HIP launch itself. That dominates
eager decode / prefill steps that launch a few small kernels per layer.

FastLaunch keys its own cache on the same properties, computed cheaply:
  tensor  (dtype, data_ptr % 16 == 0, view bytes <= 2**31 - 1)
  int     (== 1, % 16 == 0, fits int32)
  bool / float / None   their type (Triton does not specialize them)
plus the keyword constexprs / launch options and the device. A new key goes
through JITFunction.run (compiling if needed); later calls with that key launch
the cached CompiledKernel directly. The 2 GB flag uses the view's bytes rather
than its storage's: a key that claims "within 2 GB" then only ever runs on
views that are, and a kernel addresses only its views, so the kernel run is
always valid (at worst a non-buffer-ops variant where Triton would have used
buffer ops).

    launch = FastLaunch(my_kernel)
    launch(grid, *positional_args, CONST=..., num_warps=4)
"""

import torch
import triton
from triton import knobs
from triton.runtime.driver import driver

_MAX_I32 = 2**31 - 1


class FastLaunch:
    def __init__(self, fn):
        self.fn = fn
        self._params = fn.params
        self._cache = {}

    def _key(self, args, kwargs):
        T, M = torch.Tensor, _MAX_I32
        return (
            tuple([
                (a.dtype, not (a.data_ptr() & 15), a.nbytes <= M) if isinstance(a, T)
                else (a == 1, not (a % 16), -M - 1 <= a <= M) if type(a) is int
                else type(a) if a is None or type(a) in (bool, float)
                else (type(a), a)
                for a in args
            ]),
            tuple(kwargs.items()),
            torch.cuda.current_device(),
        )  # fmt: skip

    def __call__(self, grid, *args, **kwargs):
        key = self._key(args, kwargs)
        hit = self._cache.get(key)
        if hit is None:
            kernel = self.fn[grid](*args, **kwargs)
            # the launcher takes every parameter in signature order (constexprs
            # included); the trailing ones were passed by keyword or defaulted
            tail = tuple(p.name for p in self._params[len(args):])
            defaults = {p.name: p.default for p in self._params[len(args):]}
            self._cache[key] = (kernel, tail, defaults)
            return kernel
        kernel, tail, defaults = hit
        full = args + tuple(kwargs[n] if n in kwargs else defaults[n] for n in tail)
        g = grid if len(grid) == 3 else tuple(grid) + (1,) * (3 - len(grid))
        stream = driver.active.get_current_stream(driver.active.get_current_device())
        enter, exit_ = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
        meta = kernel.launch_metadata(g, stream, *full) if enter is not None else None
        kernel.run(g[0], g[1], g[2], stream, kernel.function, kernel.packed_metadata, meta,
                   enter, exit_, *full)  # fmt: skip
        return kernel


__all__ = ["FastLaunch", "triton"]
