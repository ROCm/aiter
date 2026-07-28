"""Definitive probe: does opsel select 1 scale byte for all K=64, or does it
select different bytes for K[0:32] vs K[32:64]?

Triton's tl.dot_scaled emits 8 QK MFMA per inner iter for head_dim=128,
which can only be correct under per-32-group MXFP4 if ONE call applies
TWO different scale bytes to the two K-halves. This probe pins that down.

Method: zero one scale byte (×0) and leave others ×1. Run with each
opselA. Observe which K-group is zeroed in the output.

Hypotheses:
  H_call_wide: opselA selects 1 scale byte applied to all K=64.
               Setting byte X to 0 zeros the WHOLE call when opselA=X,
               and has no effect when opselA != X.
  H_per_half:  opselA selects byte X for K[0:32] and byte X^1 (or some
               function of X) for K[32:64]. Setting byte X to 0 with
               opselA=X zeros HALF the K (one specific group).

The prior probe (probe_mfma_32x32x64_fp4_scale.py) used byte=×2 and
produced ambiguous residuals. This one uses ×0 to make the signal sharp.
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


def make_kernel(scale_a_val, scale_b_val, opsel_a, opsel_b):
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
    """Reconstruct 32x32 D from 64-lane output."""
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


def pack_scale(bytes4):
    """Pack 4 e8m0 bytes (LSB first) into i32."""
    s = 0
    for i, b in enumerate(bytes4):
        s |= (b & 0xFF) << (i * 8)
    return s


def main():
    device = "cuda"
    A_f, B_f, A_buf, B_buf = build(0)
    A_dev = torch.from_numpy(A_buf.flatten()).to(device).contiguous()
    B_dev = torch.from_numpy(B_buf.flatten()).to(device).contiguous()

    # Per-group products: AB_g[g] = A[:, g*32:(g+1)*32] @ B[g*32:(g+1)*32, :]
    AB_g = [(A_f[:, g*32:(g+1)*32] @ B_f[g*32:(g+1)*32, :]) for g in range(2)]
    D_full = AB_g[0] + AB_g[1]
    print(f"|AB_g[0]|.max = {np.abs(AB_g[0]).max():.3f}")
    print(f"|AB_g[1]|.max = {np.abs(AB_g[1]).max():.3f}")
    print(f"|D_full|.max  = {np.abs(D_full).max():.3f}")
    print()

    # Test 1: scaleA bytes [×1, ×0, ×0, ×0], opselA=0..3.
    # Under H_call_wide: opselA=0 picks ×1 → output = D_full;
    #                    opselA=1,2,3 picks ×0 → output = 0.
    # Under H_per_half:  opselA=X applies bytes[X] to one K-half AND
    #                    bytes[X^?] to the other half. Need to see pattern.
    print("=" * 70)
    print("TEST 1: scaleA=[0x7F, 0x00, 0x00, 0x00] (only byte 0 = ×1)")
    print("        scaleB=0x7F7F7F7F (all ×1)")
    print("        Sweep opselA = 0..3")
    print("=" * 70)
    sa = pack_scale([0x7F, 0x00, 0x00, 0x00])
    sb = 0x7F7F7F7F
    for opsel_a in range(4):
        launch = make_kernel(sa, sb, opsel_a, 0)
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)
        launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        D = gather(out)
        # Compare against candidate hypotheses
        candidates = {
            "0": np.zeros_like(D_full),                # full zero (call killed)
            "g0": AB_g[0],                              # group 0 only
            "g1": AB_g[1],                              # group 1 only
            "full": D_full,                             # full (no scaling change)
        }
        residuals = {name: float(np.abs(D - cand).max()) for name, cand in candidates.items()}
        match = min(residuals, key=residuals.get)
        print(f"  opselA={opsel_a}: matches '{match}' (resid={residuals[match]:.4f})")
        for name, r in residuals.items():
            print(f"      {name}: {r:.4f}")

    # Test 2: scaleA=[×0, ×1, ×0, ×0] — only byte 1 alive.
    print()
    print("=" * 70)
    print("TEST 2: scaleA=[0x00, 0x7F, 0x00, 0x00] (only byte 1 = ×1)")
    print("=" * 70)
    sa = pack_scale([0x00, 0x7F, 0x00, 0x00])
    for opsel_a in range(4):
        launch = make_kernel(sa, sb, opsel_a, 0)
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)
        launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        D = gather(out)
        candidates = {
            "0": np.zeros_like(D_full),
            "g0": AB_g[0],
            "g1": AB_g[1],
            "full": D_full,
        }
        residuals = {name: float(np.abs(D - cand).max()) for name, cand in candidates.items()}
        match = min(residuals, key=residuals.get)
        print(f"  opselA={opsel_a}: matches '{match}' (resid={residuals[match]:.4f})")

    # Test 3: scaleA=[×1, ×1, ×0, ×0] — bytes 0 AND 1 alive (the Triton pattern!).
    # If Triton's "K=64 per call covers 2 scale groups" is correct AND
    # opselA=0 means byte 0→g0, byte 1→g1 (per-half), then this should
    # produce D_full for opselA=0 (both groups scaled ×1).
    print()
    print("=" * 70)
    print("TEST 3: scaleA=[0x7F, 0x7F, 0x00, 0x00] (bytes 0,1 = ×1)")
    print("        Critical: under per-half hypothesis, opselA=0 → both groups ×1")
    print("=" * 70)
    sa = pack_scale([0x7F, 0x7F, 0x00, 0x00])
    for opsel_a in range(4):
        launch = make_kernel(sa, sb, opsel_a, 0)
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)
        launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        D = gather(out)
        candidates = {
            "0": np.zeros_like(D_full),
            "g0": AB_g[0],
            "g1": AB_g[1],
            "full": D_full,
        }
        residuals = {name: float(np.abs(D - cand).max()) for name, cand in candidates.items()}
        match = min(residuals, key=residuals.get)
        print(f"  opselA={opsel_a}: matches '{match}' (resid={residuals[match]:.4f})")

    # Test 4: distinct multipliers per byte — definitive.
    # bytes = [×1, ×2, ×4, ×8] = [0x7F, 0x80, 0x81, 0x82]
    # Run each opselA and check what scale was applied to each K-group.
    print()
    print("=" * 70)
    print("TEST 4: scaleA bytes=[0x7F, 0x80, 0x81, 0x82] = [×1, ×2, ×4, ×8]")
    print("        Decode (alpha_g0, alpha_g1) from D = alpha_g0 * AB_g[0] + alpha_g1 * AB_g[1]")
    print("=" * 70)
    sa = pack_scale([0x7F, 0x80, 0x81, 0x82])
    for opsel_a in range(4):
        launch = make_kernel(sa, sb, opsel_a, 0)
        out = torch.zeros((64, 16), dtype=torch.float32, device=device)
        launch(A_dev, B_dev, out.flatten(), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        D = gather(out)
        # Solve for alpha_g0, alpha_g1 in least-squares sense:
        # D = alpha_g0 * AB_g[0] + alpha_g1 * AB_g[1]
        Y = D.flatten()
        X = np.stack([AB_g[0].flatten(), AB_g[1].flatten()], axis=1)
        alpha, *_ = np.linalg.lstsq(X, Y, rcond=None)
        recon = (alpha[0] * AB_g[0] + alpha[1] * AB_g[1])
        resid = float(np.abs(D - recon).max())
        print(f"  opselA={opsel_a}: alpha_g0={alpha[0]:.4f}, alpha_g1={alpha[1]:.4f}, resid={resid:.4f}")


if __name__ == "__main__":
    main()
