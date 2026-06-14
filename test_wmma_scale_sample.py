#!/usr/bin/env python3
"""V_WMMA_SCALE_F32_16X16X128_F8F6F4 sample: constant data + opsel verification.

SRC0: FP4 all-1.0, SRC1: FP8 all-1.0, scaleA=scaleB=[1.0,0.5,0.25,0.125].
Base expected = 32+8+2+0.5 = 42.5

opsel test: scale VGPR is per-lane. opsel=0 should read from lanes 0-15,
opsel=1 from lanes 16-31. We init the "unused" half-wave's scale to zero
and verify the result is unchanged (real half used) vs near-zero (wrong half).
"""

import os, sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch, pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _arith_d
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

WAVE_SIZE = 32
FP4_ONE = 0x22222222
FP8_ONE = 0x38383838
SCALE_REAL = 124 << 24 | 125 << 16 | 126 << 8 | 127  # [1.0, 0.5, 0.25, 0.125]
SCALE_ZERO = 0  # E8M0 byte=0 → 2^-127 ≈ 0


def _store_vec8(result, rsrc, base_off):
    for h in range_constexpr(2):
        vals = [vector.extract(result, static_position=[h * 4 + i], dynamic_position=[]) for i in range_constexpr(4)]
        buffer_ops.buffer_store(vector.from_elements(T.vec(4, T.f32), vals), rsrc, base_off + arith.constant(h * 4))


def _wmma_body(C, scale_a_val, scale_b_val, opselA=0, opselB=0):
    """Shared kernel body: constant A/B data, parameterized scales + opsel."""
    tid = fx.thread_idx.x
    a = arith.constant_vector(FP4_ONE, T.vec(8, T.i32))
    b = arith.constant_vector(FP8_ONE, T.vec(16, T.i32))
    acc = arith.constant_vector(0.0, T.vec(8, T.f32))
    result = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
        T.vec(8, T.f32), a, b, acc, scale_a_val, scale_b_val,
        fmtA=4, fmtB=0, scaleAType=opselA, scaleBType=opselB,
    )
    _store_vec8(result, buffer_ops.create_buffer_resource(C), tid * arith.constant(8))


# ── Case 1: base ──
@flyc.kernel
def _k_base(C: fx.Tensor):
    _wmma_body(C, arith.constant(SCALE_REAL), arith.constant(SCALE_REAL))


@flyc.jit
def launch_base(C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _k_base(C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)


# ── Case 2: opsel=0 — lanes 0-15 get real scale, lanes 16-31 get zero ──
@flyc.kernel
def _k_opsel0(C: fx.Tensor):
    tid = fx.thread_idx.x
    is_lo = arith.cmpi(_arith_d.CmpIPredicate.slt, tid, arith.constant(16))
    s = arith.select(is_lo, arith.constant(SCALE_REAL), arith.constant(SCALE_ZERO))
    _wmma_body(C, s, s, opselA=0, opselB=0)


@flyc.jit
def launch_opsel0(C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _k_opsel0(C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)


# ── Case 3: opsel=1 — lanes 16-31 get real scale, lanes 0-15 get zero ──
@flyc.kernel
def _k_opsel1(C: fx.Tensor):
    tid = fx.thread_idx.x
    is_hi = arith.cmpi(_arith_d.CmpIPredicate.sge, tid, arith.constant(16))
    s = arith.select(is_hi, arith.constant(SCALE_REAL), arith.constant(SCALE_ZERO))
    _wmma_body(C, s, s, opselA=1, opselB=1)


@flyc.jit
def launch_opsel1(C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _k_opsel1(C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)


# ── Runner ──
def _run(label, fn):
    c = torch.zeros(256, dtype=torch.float32, device="cuda")
    fn(c)
    torch.cuda.synchronize()
    c = c.cpu()
    active = c[:128]
    print(f"\n  {label}")
    for lane in range(16):
        v = c[lane * 8: lane * 8 + 8].tolist()
        print(f"    Lane {lane:2d}: [{', '.join(f'{x:.2f}' for x in v)}]")
    print(f"    Unique: {sorted(active.unique().tolist())}")
    return active


def _skip_if_not_gfx1250():
    if str(get_rocm_arch()) != "gfx1250":
        pytest.skip("requires gfx1250")


def test_base():
    _skip_if_not_gfx1250()
    active = _run("Base: all scales real, opsel=0", launch_base)
    assert torch.allclose(active, torch.full_like(active, 42.5), atol=1e-2)


def test_opsel0():
    """opsel=0: HW reads scales from lanes 0-15. Those lanes have real scale → expect 42.5."""
    _skip_if_not_gfx1250()
    active = _run("opsel=0: lanes 0-15=real, 16-31=zero", launch_opsel0)
    assert torch.allclose(active, torch.full_like(active, 42.5), atol=1e-2), (
        f"opsel=0 should use lanes 0-15 scales. Got {sorted(active.unique().tolist())}"
    )


def test_opsel1():
    """opsel=1: HW reads scales from lanes 16-31. Those lanes have real scale → expect 42.5."""
    _skip_if_not_gfx1250()
    active = _run("opsel=1: lanes 0-15=zero, 16-31=real", launch_opsel1)
    assert torch.allclose(active, torch.full_like(active, 42.5), atol=1e-2), (
        f"opsel=1 should use lanes 16-31 scales. Got {sorted(active.unique().tolist())}"
    )


if __name__ == "__main__":
    test_base()
    test_opsel0()
    test_opsel1()
    print("\nAll passed.")
