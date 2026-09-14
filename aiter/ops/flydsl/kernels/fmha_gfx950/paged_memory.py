# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Plain scalar/vector memory copies on AITER's FlyDSL 0.3.2 runtime."""

import flydsl.expr as fx


def load(ptr, *, dtype, count):
    # Dynamic FP8 offsets can leave only byte alignment in the pointer type.
    # Copy bytes and bitcast the registers instead of asserting word alignment.
    byte_count = count * dtype.width // 8
    view = fx.make_view(fx.recast_iter(fx.Uint8, ptr), fx.make_layout(byte_count, 1))
    fragment = fx.make_rmem_tensor(byte_count, fx.Uint8)
    atom = fx.make_copy_atom(fx.UniversalCopy(byte_count * 8), fx.Uint8)
    fx.copy(atom, view, fragment)
    result = fx.Vector(fragment.load()).bitcast(dtype)
    return result[0] if count == 1 else result


def store(ptr, value):
    vector = (
        value
        if isinstance(value, fx.Vector)
        else fx.Vector.from_elements([value], type(value))
    )
    packed = vector.bitcast(fx.Uint8)
    view = fx.make_view(fx.recast_iter(fx.Uint8, ptr), fx.make_layout(packed.numel, 1))
    fragment = fx.make_rmem_tensor(packed.numel, fx.Uint8)
    fragment.store(packed)
    atom = fx.make_copy_atom(fx.UniversalCopy(packed.numel * 8), fx.Uint8)
    fx.copy(atom, fragment, view)
