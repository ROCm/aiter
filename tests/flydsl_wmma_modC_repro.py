#!/usr/bin/env python3
"""Minimal reproducer: fx.mma_atom_call vs raw rocdl.wmma precision mismatch.

Run on gfx1250:
    python tests/flydsl_wmma_modC_repro.py

Expected: both methods produce identical output.
Actual:   mma_atom_call output differs (precision regression from missing modC=0).
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import arith, rocdl
from flydsl.expr.typing import T, Vector as Vec

BLOCK = 32


@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def wmma_raw(A: fx.Tensor, B: fx.Tensor, out: fx.Tensor):
    """Use raw rocdl_dialect.wmma_f32_16x16x32_bf16 with explicit modC=0."""
    tid = fx.Int32(fx.thread_idx.x)
    # Load A (v16bf16) and B (v16bf16) from global
    a_vec = A[(tid,)].load()  # assumes A is rmem-compatible
    b_vec = B[(tid,)].load()

    zero = fx.constant_vector(0.0, T.vec(8, T.f32))
    result = rocdl_dialect.wmma_f32_16x16x32_bf16(
        T.vec(8, T.f32),
        a_vec.ir_value(),
        b_vec.ir_value(),
        zero,
        signA=False,
        signB=False,
        modC=0,       # <-- explicit
        reuseA=False,
        reuseB=False,
    )
    # Store result
    out_rmem = fx.make_rmem_tensor(8, fx.Float32)
    out_rmem.store(result.result)
    fx.memref_store_vec(out_rmem.load(), out[(tid,)])


@flyc.kernel(known_block_size=[BLOCK, 1, 1])
def wmma_atom(A: fx.Tensor, B: fx.Tensor, out: fx.Tensor):
    """Use fx.make_mma_atom + fx.mma_atom_call (modC NOT forwarded)."""
    tid = fx.Int32(fx.thread_idx.x)
    a_vec = A[(tid,)].load()
    b_vec = B[(tid,)].load()

    atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))

    a_rmem = fx.make_rmem_tensor(16, fx.BFloat16)
    a_rmem.store(a_vec)
    b_rmem = fx.make_rmem_tensor(16, fx.BFloat16)
    b_rmem.store(b_vec)
    c_rmem = fx.make_rmem_tensor(8, fx.Float32)
    c_rmem.store(fx.constant_vector(0.0, T.vec(8, T.f32)))

    fx.mma_atom_call(atom, c_rmem, a_rmem, b_rmem, c_rmem)

    fx.memref_store_vec(c_rmem.load(), out[(tid,)])


@flyc.jit
def launch_raw(A: fx.Tensor, B: fx.Tensor, out: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
    wmma_raw(A, B, out).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)


@flyc.jit
def launch_atom(A: fx.Tensor, B: fx.Tensor, out: fx.Tensor,
                stream: fx.Stream = fx.Stream(None)):
    wmma_atom(A, B, out).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)


def main():
    torch.manual_seed(42)
    device = "cuda"

    # Random bf16 inputs shaped for v16bf16 per thread
    A = torch.randn(BLOCK, 16, dtype=torch.bfloat16, device=device)
    B = torch.randn(BLOCK, 16, dtype=torch.bfloat16, device=device)
    out_raw = torch.zeros(BLOCK, 8, dtype=torch.float32, device=device)
    out_atom = torch.zeros(BLOCK, 8, dtype=torch.float32, device=device)

    stream = torch.cuda.current_stream()

    launch_raw(A, B, out_raw, stream=stream)
    launch_atom(A, B, out_atom, stream=stream)
    torch.cuda.synchronize()

    max_diff = (out_raw - out_atom).abs().max().item()
    print(f"Max abs diff: {max_diff:.6e}")

    if max_diff == 0.0:
        print("PASS: Both methods produce identical output")
    elif max_diff < 1e-5:
        print(f"WARN: Small diff ({max_diff:.2e}) — likely rounding, not modC issue")
    else:
        print(f"FAIL: Significant diff ({max_diff:.2e}) — likely modC not forwarded")
        print(f"  out_raw[:4]:  {out_raw[:4].tolist()}")
        print(f"  out_atom[:4]: {out_atom[:4].tolist()}")


if __name__ == "__main__":
    main()
