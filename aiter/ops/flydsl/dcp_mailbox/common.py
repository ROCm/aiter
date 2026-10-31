# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM at a526f0d under the MIT License.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small build-time and device helpers for the DCP mailbox kernels."""

from __future__ import annotations

import enum
import functools
import hashlib
import inspect
from dataclasses import dataclass, fields
from pathlib import Path

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops as bo

CM_DEV = 16
CM_SYS = 17
PAIR_BYTES = 8

_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parent
_COMMUNICATION_UTILS = _ROOT.parent / "kernels" / "communication_ops_utils.py"


@functools.cache
def source_digest() -> str:
    """Hash the mailbox sources into FlyDSL's JIT key."""
    h = hashlib.sha256()
    for path in [*sorted(_ROOT.glob("*.py")), _COMMUNICATION_UTILS]:
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


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def uniform(value):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value).ir_value()))


def traced(fn):
    """Rewrite control flow over traced values in a device helper."""
    return ASTRewriter.transform(fn)


def hw_f32(name, value):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(value).ir_value()], [], [])
    )


def hw_exp2(value):
    return hw_f32("llvm.amdgcn.exp2.f32", value)


def ld_i32(ptr, index):
    return fx.Int32(bo.buffer_load(rsrc(ptr), index, vec_width=1, dtype=T.i32))


def ld_dev(ptr, index, words=1):
    return bo.buffer_load(
        rsrc(ptr), index, vec_width=words, dtype=T.i32, cache_modifier=CM_DEV
    )


def load_ptr64(table, index):
    words = fx.Vector(bo.buffer_load(rsrc(table), index * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(words[1])) << 32) | fx.Int64(fx.Uint32(uniform(words[0])))


def peer_bases(peers, ranks):
    return [load_ptr64(peers, rank) for rank in range(ranks)]


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


def fp8_pack4(a, b, c, d):
    """Pack four in-range f32 values as E4M3 bytes."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))
