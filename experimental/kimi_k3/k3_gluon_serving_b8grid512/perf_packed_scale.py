# Frozen JIT snapshot: retain original imports and tuning overrides.
# ruff: noqa: E402, F401
from .coherent_memory import coherent_load

"""Preserve exact zero-scale behavior without unpacking converted BF16 pairs."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def mask_zero_scale_packed(raw, scales):
    shape: gl.constexpr = scales.shape
    paired = gl.reshape(scales, shape[:-1] + [shape[-1] // 2, 2])
    raw_scales, unused = gl.split(paired)
    raw_scales = gl.convert_layout(raw_scales, raw.type.layout)
    return gl.where(raw_scales == 0, 0, raw).to(gl.uint8)


@g.jit
def mask_zero_scale_words(raw, scales):
    shape: gl.constexpr = raw.shape
    groups = gl.reshape(raw, shape[:-1] + [shape[-1] // 4, 2, 2])
    even, odd = gl.split(groups)
    b0, b2 = gl.split(even)
    b1, b3 = gl.split(odd)
    words = (
        b0.to(gl.uint32)
        | b1.to(gl.uint32) << 8
        | b2.to(gl.uint32) << 16
        | b3.to(gl.uint32) << 24
    )
    scale_groups = gl.reshape(scales, shape[:-1] + [shape[-1] // 4, 2, 2, 2])
    selected, unused = gl.split(scale_groups)
    selected, unused = gl.split(selected)
    selected, unused = gl.split(selected)
    selected = gl.convert_layout(selected, words.type.layout)
    words = gl.where(selected == 0, 0, words)
    even = gl.join((words & 255).to(gl.uint8), (words >> 16 & 255).to(gl.uint8))
    odd = gl.join((words >> 8 & 255).to(gl.uint8), (words >> 24).to(gl.uint8))
    return gl.reshape(gl.join(even, odd), shape)
