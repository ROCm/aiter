# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d.
# Source files: atom/mono/plan/build_key.py and atom/mono/runtime/abi.py.

"""Small build-time and device helpers for the DCP mailbox kernels."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import flydsl.expr as fx

PAIR_BYTES = 8
EPOCH_BYTES = 4


def pair_layout(items) -> dict:
    """Lay out named arrays of naturally aligned 8-byte tagged pairs."""
    out, offset = {}, 0
    for name, pairs in items:
        offset = (offset + PAIR_BYTES - 1) // PAIR_BYTES * PAIR_BYTES
        out[name] = (offset, pairs * PAIR_BYTES)
        offset += pairs * PAIR_BYTES
    return out


def pair_span(layout) -> int:
    """Return the byte span of one parity of tagged-pair regions."""
    return max(offset + size for offset, size in layout.values())


def epoch_offset(layout) -> int:
    """Return the untagged epoch region offset after both pair parities."""
    return 2 * pair_span(layout)


def peer_bytes(layout, ranks) -> int:
    """Return two pair parities plus one i32 epoch slot per rank."""
    return epoch_offset(layout) + ranks * EPOCH_BYTES


def token_tile(tokens: int) -> int:
    """Select tokens per CTA; ``AITER_DCP_MAILBOX_TOKEN_TILE`` overrides it."""
    override = int(os.environ.get("AITER_DCP_MAILBOX_TOKEN_TILE", "0"))
    if override > 0:
        return override
    if tokens >= 192:
        return 8
    if tokens >= 96:
        return 4
    return 2 if tokens >= 48 else 1


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
