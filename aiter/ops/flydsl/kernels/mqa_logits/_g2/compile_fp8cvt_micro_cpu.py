# SPDX-License-Identifier: MIT
"""No-device micro ISA: actual converter helpers, eight codes per lane.

This is instruction-count evidence only, NOT a launch or performance benchmark.
"""
import os
os.environ.setdefault("COMPILE_ONLY", "1")
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from _load import load_kernel_module
K, _ = load_kernel_module()
Vec = fx.Vector
_plane = K._plane
_plane_bytes = K._plane_bytes
_ld_vec = K._ld_vec
_st_i32 = K._st_i32
legacy = K._e4m3x4_to_bf16x2
native = K._e4m3x8_to_bf16x4_fast


def current8(x0, x1):
    a, b = legacy(x0)
    c, d = legacy(x1)
    return Vec.from_elements([a, b, c, d], dtype=fx.Int32)


def build(fast):
    name = 'qsa_fp8cvt_micro_' + ('fast' if fast else 'current')
    convert = native if fast else current8

    @flyc.kernel(name=name, known_block_size=[32, 1, 1])
    def kernel(SRC, DST):
        lane = fx.Int32(fx.thread_idx.x)
        src = _plane(SRC, _plane_bytes(fx.Int32(32), 8))
        dst = _plane(DST, _plane_bytes(fx.Int32(32), 16))
        x = _ld_vec(src, lane * 8, 2, fx.Int32)
        words = convert(fx.Int32(x[0]), fx.Int32(x[1]))
        for j in fx.range_constexpr(4):
            _st_i32(dst, lane * 16 + j * 4, fx.Int32(words[j]))

    @flyc.jit
    def launch(SRC, DST, stream: fx.Stream = fx.Stream(None)):
        kernel(SRC, DST).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    return launch


fast = K._FP8_CVT == 'fast'
flyc.compile(build(fast), torch.zeros(256, dtype=torch.uint8),
             torch.zeros(512, dtype=torch.uint8), fx.Stream(None))
print('[COMPILED] actual converter micro', 'fast' if fast else 'current')
