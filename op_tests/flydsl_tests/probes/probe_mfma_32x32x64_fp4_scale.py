"""Sweep scaleA bytes and opselA to determine scale-group mapping.

K=64 = 2 scale groups (group g covers K=g*32..g*32+31).
scaleA is i32 (4 bytes). Sweep one byte to 0x80 (=128 → ×2.0) and
observe which group's contribution doubles.
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


def make_kernel(scale_a_val=0x7F7F7F7F, scale_b_val=0x7F7F7F7F,
                opsel_a=0, opsel_b=0):
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

        sa = arith.constant(scale_a_val, type=T.i32)
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
    D_ref = (A_f @ B_f).astype(np.float32)
    AB_g = [(A_f[:, g*32:(g+1)*32] @ B_f[g*32:(g+1)*32, :]) for g in range(2)]
    A_dev = torch.from_numpy(A_buf.flatten()).to(device).contiguous()
    B_dev = torch.from_numpy(B_buf.flatten()).to(device).contiguous()

    print("Sweep: scaleA byte position with byte=0x80 (others=0x7F), opselA=0..3")
    print("=" * 70)
    for opsel in range(4):
        for b_pos in range(4):
            scale = 0
            for byte_i in range(4):
                v = 0x80 if byte_i == b_pos else 0x7F
                scale |= v << (byte_i * 8)
            launch = make_kernel(scale_a_val=scale, scale_b_val=0x7F7F7F7F,
                                 opsel_a=opsel, opsel_b=0)
            out = torch.zeros((64, 16), dtype=torch.float32, device=device)
            launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            D = gather(out)
            # Expected: D = D_ref + extra*AB_g[scaled_group_idx]
            # where extra = (×2 - ×1) = 1
            deltas = [float(np.abs(D - (D_ref + AB_g[g])).max()) for g in range(2)]
            best_g = int(np.argmin(deltas))
            tag = "PASS" if min(deltas) < 1e-2 else "----"
            print(f"  [{tag}] opselA={opsel} byte_pos={b_pos}: best_group={best_g}, "
                  f"residuals={[f'{d:.3f}' for d in deltas]}")


if __name__ == "__main__":
    main()
