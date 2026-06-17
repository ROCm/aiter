"""Test if 32x32x64 in FP4 mode actually means K=64 elements (not K=128)."""

import torch
import numpy as np

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    arith, buffer_ops, gpu, range_constexpr, vector,
)
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
    diffs = np.abs(E2M1_LUT - x)
    return int(np.argmin(diffs))


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty(): return ir.Type.parse("!llvm.ptr")
def _extract_ptr(t): return _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), _llvm_value(t))
def _ld(rt, p): return _llvm.LoadOp(rt, _llvm_value(p)).result
def _st(v, p): return _llvm.StoreOp(_llvm_value(v), _llvm_value(p))


def make_kernel():
    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(A_buf: fx.Tensor, B_buf: fx.Tensor, Out: fx.Tensor):
        a_ptr = _extract_ptr(A_buf)
        b_ptr = _extract_ptr(B_buf)
        o_ptr = _extract_ptr(Out)

        v16f32_type = Vec.make_type(16, fx.Float32)
        v8i32_type = Vec.make_type(8, fx.Int32)
        v32i8_type = Vec.make_type(32, fx.Int8)
        c_zero = Vec.filled(16, 0.0, fx.Float32)

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid

        a_off = lane * 32
        a_gep = buffer_ops.get_element_ptr(a_ptr, fx.Int64(a_off), elem_type=T.i8)
        a_v8i32 = vector.bitcast(v8i32_type, _ld(v32i8_type, a_gep))
        b_off = lane * 32
        b_gep = buffer_ops.get_element_ptr(b_ptr, fx.Int64(b_off), elem_type=T.i8)
        b_v8i32 = vector.bitcast(v8i32_type, _ld(v32i8_type, b_gep))

        scale = arith.constant(0x7F7F7F7F, type=T.i32)
        d = _mfma_op(
            res=v16f32_type, a=a_v8i32, b=b_v8i32, c=_raw(c_zero),
            cbsz=4, blgp=4, opselA=0, scaleA=scale, opselB=0, scaleB=scale,
        ).result
        for i in range_constexpr(16):
            v = vector.extract(d, static_position=[i], dynamic_position=[])
            o_g = buffer_ops.get_element_ptr(o_ptr, fx.Int64(lane * 16 + i), elem_type=T.f32)
            _st(v, o_g)

    @flyc.jit
    def launch(A, B, Out, stream: fx.Stream = fx.Stream(None)):
        k(A, B, Out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def gather(out_dev):
    out_np = out_dev.cpu().numpy()
    D = np.zeros((32, 32), dtype=np.float32)
    for lane in range(64):
        klane = lane // 32
        lane32 = lane % 32
        for i in range(16):
            sub = i // 4; elem = i % 4
            m = sub * 8 + klane * 4 + elem
            n = lane32
            D[m, n] = out_np[lane, i]
    return D


# Hypothesis: K=64 elements (32 nibbles per lane = 16 bytes per lane).
# Per lane: row m=lane32 (A) or col n=lane32 (B), with K=klane*32..+31.
# Bytes 0..15 of v8i32 hold real data; bytes 16..31 unused.

# OR: K=128 with NIBBLES laid out specifically. Maybe nibble-per-byte
# packing is along K dim such that byte_b in v8i32 holds K-nibbles
# (klane*64 + 2*b) and (klane*64 + 2*b + 1).

# Test K=64 hypothesis:
def build_K64(seed):
    rng = np.random.default_rng(seed)
    M, K, N = 32, 64, 32
    A_f = rng.choice(E2M1_LUT[:8], size=(M, K)).astype(np.float32)
    B_f = rng.choice(E2M1_LUT[:8], size=(K, N)).astype(np.float32)
    # Pack: per lane (klane, lane32), 16 bytes = 32 nibbles for K=klane*32..+31
    A_buf = np.zeros((64, 32), dtype=np.uint8)
    B_buf = np.zeros((64, 32), dtype=np.uint8)
    for lane in range(64):
        klane = lane // 32
        lane32 = lane % 32
        for byte_i in range(16):  # 16 bytes per lane
            k_lo = klane * 32 + 2 * byte_i
            k_hi = klane * 32 + 2 * byte_i + 1
            v_lo = f32_to_e2m1(float(A_f[lane32, k_lo]))
            v_hi = f32_to_e2m1(float(A_f[lane32, k_hi]))
            A_buf[lane, byte_i] = v_lo | (v_hi << 4)
            v_lo = f32_to_e2m1(float(B_f[k_lo, lane32]))
            v_hi = f32_to_e2m1(float(B_f[k_hi, lane32]))
            B_buf[lane, byte_i] = v_lo | (v_hi << 4)
    return A_f, B_f, A_buf, B_buf


def main():
    device = "cuda"
    print("Hypothesis: 32x32x64 FP4 mode means K=64 elements")
    print("=" * 70)
    A_f, B_f, A_buf, B_buf = build_K64(seed=0)
    D_ref = (A_f @ B_f).astype(np.float32)
    print(f"D_ref mean: {D_ref.mean():.2f}, max: {D_ref.max():.2f}")
    A_dev = torch.from_numpy(A_buf.flatten()).to(device).contiguous()
    B_dev = torch.from_numpy(B_buf.flatten()).to(device).contiguous()
    out = torch.zeros((64, 16), dtype=torch.float32, device=device)
    launch = make_kernel()
    launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    D = gather(out)
    diff = float(np.abs(D - D_ref).max())
    print(f"D mean: {D.mean():.2f}, max: {D.max():.2f}")
    print(f"max|diff|: {diff:.4f}")
    print(f"D[0:2, 0:4]:\n{D[:2, :4]}")
    print(f"D_ref[0:2, 0:4]:\n{D_ref[:2, :4]}")
    if diff < 1e-3:
        print("\n*** PASS: K=64 elements per call in FP4 mode of 32x32x64. ***")
        print("Per lane: 16 bytes = 32 nibbles for K=klane*32..+31.")
        print("(v8i32 = 32 bytes per lane; only first 16 bytes hold data?)")
    else:
        print("\nFAIL with K=64 hypothesis. Try other layouts.")
        # Maybe bytes 0-15 are unused and 16-31 hold the data?
        print("\nTrying: bytes 16..31 hold data instead of 0..15")
        A_buf2 = np.zeros((64, 32), dtype=np.uint8)
        B_buf2 = np.zeros((64, 32), dtype=np.uint8)
        A_buf2[:, 16:] = A_buf[:, :16]
        B_buf2[:, 16:] = B_buf[:, :16]
        A_dev = torch.from_numpy(A_buf2.flatten()).to(device).contiguous()
        B_dev = torch.from_numpy(B_buf2.flatten()).to(device).contiguous()
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)
        launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        D2 = gather(out)
        diff2 = float(np.abs(D2 - D_ref).max())
        print(f"max|diff|: {diff2:.4f}")
        if diff2 < 1e-3:
            print("PASS: data in upper 16 bytes")


if __name__ == "__main__":
    main()
