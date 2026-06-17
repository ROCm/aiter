#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Minimal single-MFMA validation for CDNA4 32x32x64 fp8 scaled-MFMA.

Computes ``C[32,32] = A[32,64] @ B[64,32]`` with one workgroup, one
wave (block=(64,1,1)), one ``mma_atom_call_ssa``.  A/B are supplied to
the kernel pre-packed into lane-major fragment buffers (see
``aiter.ops.flydsl.rocdl_mfma_fp8.pack_a`` / ``pack_b``); C is read back
lane-major and unpacked.  Validated against a torch f32 matmul of the
same fp8 inputs.

Run:
    HIP_VISIBLE_DEVICES=0 python op_tests/test_flydsl_mfma_fp8_32x32x64.py
"""

import os
import sys

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import range_constexpr

_HERE = os.path.dirname(os.path.abspath(__file__))
_AITER_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

from aiter.ops.flydsl.rocdl_mfma_fp8 import (  # noqa: E402
    A_FP8_PER_LANE,
    B_FP8_PER_LANE,
    C_F32_PER_LANE,
    K,
    M,
    N,
    NUM_LANES,
    Mfma32x32x64,
    pack_a,
    pack_b,
    unpack_c,
)

FP8_DTYPE = torch.float8_e4m3fn


def _build_kernel():
    """Compile the single-MFMA launch fn.

    Args (flattened, fp8 viewed as int8):
        a_frag  : int8[NUM_LANES * A_FP8_PER_LANE]  (lane-major)
        b_frag  : int8[NUM_LANES * B_FP8_PER_LANE]  (lane-major)
        c_frag  : float32[NUM_LANES * C_F32_PER_LANE] (lane-major out)
    """

    @flyc.kernel
    def kernel(a_in: fx.Tensor, b_in: fx.Tensor, c_out: fx.Tensor):
        mfma = Mfma32x32x64()
        f8_t = fx.Float8E4M3FN.ir_type
        lane = fx.thread_idx.x % 64

        # --- A fragment: 32 fp8 / lane == vec<8xi32>, 16B chunks ---
        gA = fx.rocdl.make_buffer_tensor(a_in, max_size=False)
        gB = fx.rocdl.make_buffer_tensor(b_in, max_size=False)
        gC = fx.rocdl.make_buffer_tensor(c_out, max_size=False)

        # Recast int8 buffers to fp8 element type for A/B.
        def _f8_view(buf):
            it = fx.get_iter(buf)
            f8_ptr_ty = fx.PointerType.get(
                elem_ty=f8_t,
                address_space=TargetAddressSpace.BufferDesc,
                alignment=fx.PointerType(it.type).alignment,
            )
            it_f8 = fx.recast_iter(f8_ptr_ty, it)
            return fx.Tensor(fx.make_view(it_f8, fx.get_layout(buf)))

        gA8 = _f8_view(gA)
        gB8 = _f8_view(gB)

        a_div = fx.logical_divide(gA8, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB8, fx.make_layout(1, 1))
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))

        copy16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
        store128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        def load_frag_i32x8(div, base_elem):
            # Two 16-byte (16 fp8) chunks -> vec<8xi32>.
            r0 = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FN)
            r1 = fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FN)
            fx.copy_atom_call(copy16, fx.slice(div, (None, base_elem)), r0)
            fx.copy_atom_call(
                copy16, fx.slice(div, (None, base_elem + fx.Int32(16))), r1
            )
            v0 = fx.memref_load_vec(r0).bitcast(fx.Int32)
            v1 = fx.memref_load_vec(r1).bitcast(fx.Int32)
            return v0.shuffle(v1, list(range(8)))

        a_base = lane * fx.Int32(A_FP8_PER_LANE)
        b_base = lane * fx.Int32(B_FP8_PER_LANE)
        a_frag = load_frag_i32x8(a_div, a_base)
        b_frag = load_frag_i32x8(b_div, b_base)

        c_frag = mfma.call(a_frag, b_frag, mfma.zero_value)

        # Store vec<16xf32> as 4 x 128b chunks, lane-major.
        c_base = lane * fx.Int32(C_F32_PER_LANE)
        for chunk in range_constexpr(4):
            rc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
            sub = fx.Vector(c_frag).shuffle(
                fx.Vector(c_frag), [chunk * 4 + i for i in range(4)]
            )
            fx.memref_store_vec(sub, rc)
            fx.copy_atom_call(
                store128, rc, fx.slice(c_div, (None, c_base + fx.Int32(chunk * 4)))
            )

    @flyc.jit
    def launch(a_in: fx.Tensor, b_in: fx.Tensor, c_out: fx.Tensor, stream: fx.Stream):
        kernel(a_in, b_in, c_out).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


def main():
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available")
        return 1, 0.0, float("inf")
    torch.manual_seed(0)
    device = "cuda"

    # Logical tiles.
    a_f = (torch.randn(M, K, device=device) * 0.5).clamp(-4, 4)
    b_f = (torch.randn(K, N, device=device) * 0.5).clamp(-4, 4)
    a_q = a_f.to(FP8_DTYPE)
    b_q = b_f.to(FP8_DTYPE)

    a_np = a_q.view(torch.int8).cpu().numpy()
    b_np = b_q.view(torch.int8).cpu().numpy()

    # Pack to lane-major fragments on host.
    a_frag_np = pack_a(a_np)  # (64, 32) int8
    b_frag_np = pack_b(b_np)  # (64, 32) int8

    a_frag = torch.from_numpy(a_frag_np).to(device).contiguous().view(-1)
    b_frag = torch.from_numpy(b_frag_np).to(device).contiguous().view(-1)
    c_frag = torch.zeros(NUM_LANES * C_F32_PER_LANE, dtype=torch.float32, device=device)

    launch = _build_kernel()

    def _args(a, b, c):
        return (a, b, c, torch.cuda.current_stream())

    compiled = flyc.compile(launch, *_args(a_frag, b_frag, c_frag))
    compiled(*_args(a_frag, b_frag, c_frag))
    torch.cuda.synchronize()

    c_frag_np = c_frag.cpu().numpy().reshape(NUM_LANES, C_F32_PER_LANE)
    c_kernel = unpack_c(c_frag_np)  # (32, 32) f32

    # Reference: f32 matmul of the fp8 inputs.
    a_ref = a_q.to(torch.float32).cpu().numpy()
    b_ref = b_q.to(torch.float32).cpu().numpy()
    c_ref = a_ref @ b_ref

    c_k = c_kernel.astype(np.float32)
    cos = float(
        (c_k.reshape(-1) @ c_ref.reshape(-1))
        / (np.linalg.norm(c_k) * np.linalg.norm(c_ref) + 1e-12)
    )
    max_err = float(np.max(np.abs(c_k - c_ref)))
    rel_err = max_err / (float(np.max(np.abs(c_ref))) + 1e-12)
    print(f"cosine      = {cos:.6f}")
    print(f"max abs err = {max_err:.6f}")
    print(f"rel err     = {rel_err:.6f}")
    print(f"sample kernel[0,:4] = {c_k[0, :4]}")
    print(f"sample ref   [0,:4] = {c_ref[0, :4]}")

    ok = cos > 0.999 and rel_err < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2, cos, max_err


def test_flydsl_mfma_fp8_32x32x64():
    """Pytest entry: single 32x32x64 fp8 MFMA matches torch f32 matmul."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA/ROCm not available")
    rc, cos, max_err = main()
    assert cos > 0.999, f"cosine {cos} too low"
    assert max_err < 0.05, f"max abs err {max_err} too high (fp8 rounding only)"


if __name__ == "__main__":
    sys.exit(main()[0])
