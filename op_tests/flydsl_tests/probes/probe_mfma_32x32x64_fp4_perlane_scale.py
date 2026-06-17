"""Probe: with K=64 active per call, can we pack 2 scale groups by giving
klane=0 lanes a different scaleA i32 than klane=1 lanes?

Setup: A operand carries K=64 active (klane=0 → K[0:32], klane=1 → K[32:64]).
       Scale i32 is loaded as a per-lane VGPR whose contents depend on klane:
       - klane=0 lanes: scaleA = pack([0x7F, ..]) (×1 at byte 0)
       - klane=1 lanes: scaleA = pack([0x80, ..]) (×2 at byte 0)
       opselA=0 → byte 0 selected.

If per-lane scale works:
  Output = AB_g[0] * 1 + AB_g[1] * 2  (klane=0 contributes ×1, klane=1 contributes ×2)

If scale is broadcast across lanes (e.g. coalesced into one value):
  Output would be either ×1 or ×2 across the whole call.
"""
import torch
import numpy as np

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly, llvm as _llvm
from flydsl._mlir.dialects._rocdl_ops_gen import (
    mfma_scale_f32_32x32x64_f8f6f4 as _mfma_op,
)


E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def f32_to_e2m1(x):
    return int(np.argmin(np.abs(E2M1_LUT - x)))


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty(): return ir.Type.parse("!llvm.ptr")
def _extract_ptr(t): return _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), _llvm_value(t))
def _ld(rt, p): return _llvm.LoadOp(rt, _llvm_value(p)).result
def _st(v, p): return _llvm.StoreOp(_llvm_value(v), _llvm_value(p))


def make_kernel(scale_a_klane0, scale_a_klane1, scale_b_val, opsel_a, opsel_b):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(A_buf: fx.Tensor, B_buf: fx.Tensor, Out: fx.Tensor):
        a_ptr = _extract_ptr(A_buf); b_ptr = _extract_ptr(B_buf); o_ptr = _extract_ptr(Out)
        v16f32_type = Vec.make_type(16, fx.Float32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v32i8_type = Vec.make_type(32, fx.Int8)
        c_zero = Vec.filled(16, 0.0, fx.Float32)
        tid = fx.Index(gpu.thread_idx.x); lane = tid

        a_off = lane * 32
        a_v8 = vector.bitcast(v8i32_type, _ld(v32i8_type,
            buffer_ops.get_element_ptr(a_ptr, fx.Int64(a_off), elem_type=T.i8)))
        b_off = lane * 32
        b_v8 = vector.bitcast(v8i32_type, _ld(v32i8_type,
            buffer_ops.get_element_ptr(b_ptr, fx.Int64(b_off), elem_type=T.i8)))

        # Per-lane scaleA: klane=0 lanes get scale_a_klane0, klane=1 get scale_a_klane1.
        # lane // 32 picks klane.
        klane = lane // fx.Index(32)
        klane_i32 = arith.unwrap(arith.index_cast(T.i32, _raw(klane)))
        sa0_const = arith.constant(scale_a_klane0, type=T.i32)
        sa1_const = arith.constant(scale_a_klane1, type=T.i32)
        # select(klane==0, sa0_const, sa1_const)
        zero_i32 = arith.constant(0, type=T.i32)
        is_klane0 = arith.cmpi(arith.CmpIPredicate.eq, klane_i32, zero_i32)
        sa = ArithValue_pick(is_klane0, sa0_const, sa1_const)

        sb = arith.constant(scale_b_val, type=T.i32)
        d = _mfma_op(res=v16f32_type, a=a_v8, b=b_v8, c=_raw(c_zero),
            cbsz=4, blgp=4, opselA=opsel_a, scaleA=sa, opselB=opsel_b, scaleB=sb).result
        for i in range_constexpr(16):
            v = vector.extract(d, static_position=[i], dynamic_position=[])
            g = buffer_ops.get_element_ptr(o_ptr, fx.Int64(lane * 16 + i), elem_type=T.f32)
            _st(v, g)

    @flyc.jit
    def launch(A, B, Out, stream: fx.Stream = fx.Stream(None)):
        k(A, B, Out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)
    return launch


# Helper for arith.select
def ArithValue_pick(cond, a, b):
    from flydsl.expr.utils.arith import _to_raw
    return _llvm.SelectOp(cond, a, b).result


def gather(out_dev):
    a = out_dev.cpu().numpy()
    D = np.zeros((32, 32), dtype=np.float32)
    for lane in range(64):
        klane, lane32 = lane // 32, lane % 32
        for i in range(16):
            sub, elem = i // 4, i % 4
            D[sub * 8 + klane * 4 + elem, lane32] = a[lane, i]
    return D


def build(seed=0):
    rng = np.random.default_rng(seed)
    M, K, N = 32, 64, 32
    A_f = rng.choice(E2M1_LUT[:8], size=(M, K)).astype(np.float32)
    B_f = rng.choice(E2M1_LUT[:8], size=(K, N)).astype(np.float32)
    A_buf = np.zeros((64, 32), dtype=np.uint8)
    B_buf = np.zeros((64, 32), dtype=np.uint8)
    for lane in range(64):
        klane, lane32 = lane // 32, lane % 32
        for byte_i in range(16):
            kl, kh = klane * 32 + 2 * byte_i, klane * 32 + 2 * byte_i + 1
            A_buf[lane, byte_i] = f32_to_e2m1(A_f[lane32, kl]) | (f32_to_e2m1(A_f[lane32, kh]) << 4)
            B_buf[lane, byte_i] = f32_to_e2m1(B_f[kl, lane32]) | (f32_to_e2m1(B_f[kh, lane32]) << 4)
    return A_f, B_f, A_buf, B_buf


def main():
    device = "cuda"
    A_f, B_f, A_buf, B_buf = build(0)
    A_dev = torch.from_numpy(A_buf.flatten()).to(device).contiguous()
    B_dev = torch.from_numpy(B_buf.flatten()).to(device).contiguous()
    AB_g0 = A_f[:, 0:32] @ B_f[0:32, :]
    AB_g1 = A_f[:, 32:64] @ B_f[32:64, :]

    print("Per-lane scaleA test: klane=0 has byte0=0x7F (×1), klane=1 has byte0=0x80 (×2)")
    print("Expected if per-lane scale works: D = AB_g0 * 1 + AB_g1 * 2")
    print()

    sa_k0 = 0x7F  # ×1 at byte 0
    sa_k1 = 0x80  # ×2 at byte 0
    sb = 0x7F7F7F7F

    launch = make_kernel(sa_k0, sa_k1, sb, 0, 0)
    out = torch.zeros((64, 16), dtype=torch.float32, device=device)
    launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    D = gather(out)

    expected_per_lane = AB_g0 * 1.0 + AB_g1 * 2.0
    expected_all_x1 = AB_g0 * 1.0 + AB_g1 * 1.0
    expected_all_x2 = AB_g0 * 2.0 + AB_g1 * 2.0

    candidates = {
        "per_lane (klane0=×1, klane1=×2)": expected_per_lane,
        "all ×1 (klane0 broadcast)":       expected_all_x1,
        "all ×2 (klane1 broadcast)":       expected_all_x2,
        "zero":                            np.zeros_like(D),
    }
    print(f"|D|.max = {np.abs(D).max():.3f}")
    for name, exp in candidates.items():
        r = float(np.abs(D - exp).max())
        marker = "  <-- MATCH" if r < 1e-3 else ""
        print(f"  resid vs {name}: {r:.4f}{marker}")


if __name__ == "__main__":
    main()
