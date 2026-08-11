# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Predicated memory access helpers shared by the chunk_delta_attn Gluon kernels.

Variable-length batches leave a partial chunk at the end of every sequence, so its
rows have to be zero-filled on load and skipped on store. Fixed-length batches never
have a partial chunk, and a per-lane predicate there would cost the async-copy fast
path these kernels are built around. Each helper therefore takes the layout decision
as a ``gl.constexpr`` and the unused side folds away at compile time.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def async_load(dest, ptr, offsets, mask, MASKED: gl.constexpr, CACHE: gl.constexpr):
    """``buffer_load_to_shared`` + ``commit_group``, predicated only when ``MASKED``."""
    if MASKED:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest, ptr, offsets, mask=mask, other=0.0, cache_modifier=CACHE
        )
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest, ptr, offsets, cache_modifier=CACHE
        )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def reg_load(ptr, offsets, mask, MASKED: gl.constexpr, CACHE: gl.constexpr):
    """Register-direct ``buffer_load``, predicated only when ``MASKED``."""
    if MASKED:
        return gl.amd.cdna4.buffer_load(ptr, offsets, mask=mask, other=0.0, cache=CACHE)
    return gl.amd.cdna4.buffer_load(ptr, offsets, cache=CACHE)


@gluon.jit
def reg_store(value, ptr, offsets, mask, MASKED: gl.constexpr, CACHE: gl.constexpr):
    """``buffer_store``, predicated only when ``MASKED``.

    Under varlen the predicate is load-bearing rather than an optimization: the rows
    past a sequence's end belong to the next sequence, whose own CTA writes them.
    """
    if MASKED:
        gl.amd.cdna4.buffer_store(
            value, ptr=ptr, offsets=offsets, mask=mask, cache=CACHE
        )
    else:
        gl.amd.cdna4.buffer_store(value, ptr=ptr, offsets=offsets, cache=CACHE)
