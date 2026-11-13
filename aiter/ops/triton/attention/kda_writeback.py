# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatch for the gfx1250 one-token KDA write-back path."""

import torch
from triton.runtime import driver

try:
    from triton import knobs
except ImportError:
    knobs = None

from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_decode import (
    get_kda_config,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_decode import (
    kda_writeback_decode_kernel as kernel,
)

_launcher = None
_I32 = 2**31


def _integer_key(value):
    return (
        type(value),
        value == 1,
        value % 16 == 0,
        -_I32 <= value < _I32,
        value < 2**63,
    )


class _WritebackLauncher:
    def __init__(self):
        self.config = get_kda_config(1, 72, 12, 128, 128, writeback_decode=True)
        self.kernels = {}
        c = self.config
        self.constants = (
            128**-0.5,
            12,
            128,
            128,
            c["BV"],
            c["CH"],
            c["num_warps"],
            c["SK"],
        )

    def _jit(self, args, grid):
        return kernel[(grid,)](
            *args,
            *self.constants,
            num_warps=self.config["num_warps"],
            enable_fp_fusion=True
        )

    def launch(self, tensors, storages):
        _q, _k, _v, g, beta, _alog, _bias, out, state, _starts, slots = tensors
        stride = state.stride(0) // 128
        rows = (state.shape[0] - 1) * stride + 12 * 128
        args = (
            *tensors,
            slots.stride(0),
            g.stride(1),
            beta.stride(1),
            out.stride(1),
            stride,
            rows,
            -5.0,
        )
        grid = (128 // self.config["BV"]) * 12 * slots.numel()
        # Hooks must see each launch. Unsupported runtime versions use normal JIT.
        try:
            rt, comp = knobs.runtime, knobs.compilation
            hooks = rt.launch_enter_hook.calls or rt.launch_exit_hook.calls
            runtime_key = (
                rt.debug,
                kernel.debug,
                comp.instrumentation_mode,
                comp.fpsan_homomorphic_casts,
                knobs.amd.use_buffer_ops,
            )
        except AttributeError:
            hooks, runtime_key = True, ()
        if grid == 0 or hooks:
            self._jit(args, grid)
            return
        device = driver.active.get_current_device()
        key = (
            device,
            runtime_key,
            tuple(
                (x.dtype, x.data_ptr() % 16 == 0, st.nbytes() < _I32)
                for x, st in zip(tensors, storages)
            ),
            tuple(_integer_key(x) for x in args[11:17]),
        )
        compiled = self.kernels.get(key)
        if compiled is None:
            self.kernels[key] = self._jit(args, grid)
        else:
            compiled.run(
                grid,
                1,
                1,
                driver.active.get_current_stream(device),
                compiled.function,
                compiled.packed_metadata,
                None,
                None,
                None,
                *args,
                *self.constants
            )


def try_writeback(q, k, v, g, beta, alog, bias, state, starts, slots, out):
    """Return output for supported dense inputs, or None for generic dispatch."""
    if state is None or starts is None or slots is None or alog is None or bias is None:
        return None
    device = q.device
    if not q.is_cuda or q.ndim != 4 or device.index != torch.cuda.current_device():
        return None
    shape = q.shape
    n = shape[1]
    if not (
        shape == (1, n, 12, 128)
        and k.shape == shape
        and v.shape == shape
        and g.shape == shape
        and beta.shape == (1, n, 12)
        and state.shape[1:] == (12, 128, 128)
        and state.stride()[1:] == (16384, 128, 1)
        and state.stride(0) % 128 == 0
        and state.dtype == torch.float32
        and all(x.dtype == torch.bfloat16 for x in (q, k, v, g, beta))
        and all(x.is_contiguous() for x in (q, k, v, alog, bias, starts))
        and g.stride()[1:] == (1536, 128, 1)
        and beta.stride(-1) == 1
        and alog.numel() == 12
        and bias.numel() == 1536
        and alog.dtype == bias.dtype == torch.float32
        and starts.dtype == slots.dtype == torch.int32
        and starts.ndim == slots.ndim == 1
        and starts.numel() == n + 1
        and slots.numel() == n
    ):
        return None
    tensors = [q, k, v, g, beta, alog, bias, out, state, starts, slots]
    if not all(x.device == device for x in tensors if x is not None):
        return None
    if any(x.data_ptr() % 16 for x in (q, k, v, g, bias, state)):
        return None
    if out is not None and not (
        out.shape == v.shape
        and out.is_contiguous()
        and out.dtype == v.dtype
        and out.data_ptr() % 16 == 0
    ):
        return None
    storages = [x.untyped_storage() if x is not None else None for x in tensors]
    if out is not None:
        optr = storages[7].data_ptr()
        if any(s.data_ptr() == optr for s in storages[:7] + storages[8:]):
            return None
    else:
        out = tensors[7] = torch.empty_like(v)
        storages[7] = out.untyped_storage()
    global _launcher
    if _launcher is None:
        _launcher = _WritebackLauncher()
    _launcher.launch(tensors, storages)
    return out
