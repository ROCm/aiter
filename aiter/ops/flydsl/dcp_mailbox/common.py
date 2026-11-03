# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d.
# Source files: atom/mono/plan/build_key.py and atom/mono/runtime/abi.py.

"""Small build-time and device helpers for the DCP mailbox kernels."""

from __future__ import annotations

import enum
import functools
import hashlib
import inspect
from dataclasses import dataclass, fields
from pathlib import Path

import flydsl.expr as fx

PAIR_BYTES = 8

_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parent
_SHARED_SOURCES = (
    _ROOT.parent / "kernels" / "communication_ops_utils.py",
    _ROOT.parent / "kernels" / "kernels_common.py",
    _ROOT.parent / "kernels" / "quant_utils.py",
)


@functools.cache
def source_digest() -> str:
    """Hash the mailbox sources into FlyDSL's JIT key."""
    h = hashlib.sha256()
    for path in [*sorted(_ROOT.glob("*.py")), *_SHARED_SOURCES]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def key_tuple(key) -> tuple:
    """Return all build fields and the mailbox source digest."""
    out = []
    for field in fields(key):
        value = getattr(key, field.name)
        if not isinstance(value, _SCALARS):
            raise TypeError(
                f"{type(key).__name__}.{field.name}={value!r} must be a scalar"
            )
        out.append((field.name, value))
    out.append(("sources", source_digest()))
    return tuple(out)


def pair_layout(items) -> dict:
    """Lay out named arrays of tagged 32-bit values."""
    out, offset = {}, 0
    for name, pairs in items:
        offset = (offset + PAIR_BYTES - 1) // PAIR_BYTES * PAIR_BYTES
        out[name] = (offset, pairs * PAIR_BYTES)
        offset += pairs * PAIR_BYTES
    return out


def _python_params(fn) -> tuple[str, ...]:
    py = getattr(fn, "func", None) or getattr(fn, "_func", None)
    if py is None:
        raise TypeError(f"no Python function behind {fn!r}")
    return tuple(inspect.signature(py).parameters)


@dataclass(frozen=True)
class KernelAbi:
    """Check and pack one FlyDSL kernel ABI."""

    names: tuple[str, ...]

    def check(self, kernel, launcher) -> None:
        got = _python_params(kernel)
        if got != self.names:
            raise TypeError(f"kernel parameters {got} != ABI {self.names}")
        got = _python_params(launcher)
        if got != (*self.names, "stream"):
            raise TypeError(f"launcher parameters {got} != ABI {self.names} + stream")


def bf16_pair(lo, hi):
    return (
        fx.Vector.from_elements([lo, hi], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf2_f32(word):
    return (word << 16).bitcast(fx.Float32), (word & fx.Int32(-65536)).bitcast(
        fx.Float32
    )
