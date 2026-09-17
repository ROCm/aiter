"""Pin down the gfx950 MFMA 16x16x32 bf16 lane mapping.

The index-score kernel's entire design rests on this mapping, so it is asserted
here rather than assumed:

    A operand[v] = A[u, 8*g + v]      v = 0..7    (A is [M=16, K=32])
    B operand[v] = B[u, 8*g + v]      v = 0..7    (B is [N=16, K=32])
    C accum[r]   = C[4*g + r, u]      r = 0..3    (C is [M=16, N=16])

with lane = tid % 64, g = lane // 16, u = lane % 16.

Only the C equation had direct evidence in the repo (mfma_epilogues.py:65,71);
A and B were inferred. That matters because the whole point of choosing
A=K (M=token) is that the token axis -- the one being reduced -- lands in the
*4 accumulators of one lane* rather than being spread across lanes. If the
mapping is not what is claimed above, the reduction strategy that motivates the
rewrite collapses. So verify it before building on it.

Method: fill A and B with positional codes rather than random data, so a
mismatch says *which* mapping is real instead of just "wrong".

    A[m, k] = (m + 1) + (k + 1)/4        B[n, k] = 2*(n + 1) + (k + 1)/4

Both vary along m/n *and* k, so a wrong k-gather is caught as well as a wrong
row. Both are exactly representable in bf16 (quarter steps below 64), and the
fp32 dot of 32 such products is itself exact -- so the comparison runs at
atol=rtol=0 and any disagreement is a layout bug, never rounding.

The codes are deliberately asymmetric: an earlier version used
A[m,k] = m*32+k and B[n,k] = (n*32+k)*1024, which makes A @ B.T a symmetric
matrix, so a transposed C mapping would have passed unnoticed. The assertions
below check that each of the four plausible mis-mappings actually produces a
different matrix.

Run:
    python op_tests/test_flydsl_mfma_lane_mapping.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg, ptr_buf_tensor

M = N = 16
K = 32
WAVE = 64


def build_probe():
    """One wave, one 16x16x32 MFMA, operands assembled per the equations above."""

    @flyc.kernel(name="mfma_lane_probe", known_block_size=[WAVE, 1, 1])
    def probe_kernel(arg_a: fx.Pointer, arg_b: fx.Pointer, arg_c: fx.Pointer):
        lane = fx.Int32(gpu.thread_id("x"))
        g = lane // fx.Int32(16)
        u = lane % fx.Int32(16)

        a_buf = ptr_buf_tensor(arg_a, fx.BFloat16)
        b_buf = ptr_buf_tensor(arg_b, fx.BFloat16)
        c_buf = ptr_buf_tensor(arg_c, fx.Float32)

        def frag8(buf, row, k_base):
            """Gather the 8 elements the mapping says this lane holds: [row, k_base + v]."""
            t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            vals = [
                fx.BFloat16(
                    fx.add_offset(
                        fx.get_iter(buf), row * fx.Int32(K) + k_base + fx.Int32(v)
                    ).load(T.bf16)
                )
                for v in range_constexpr(8)
            ]
            t.store(fx.Vector.from_elements(vals, fx.BFloat16))
            return t

        k_base = g * fx.Int32(8)  # 8*g
        a_frag = frag8(a_buf, u, k_base)  # A[u, 8g+v]
        b_frag = frag8(b_buf, u, k_base)  # B[u, 8g+v]

        acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        acc.store(fx.Vector.filled(4, 0.0, fx.Float32))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        fx.gemm(mma_atom, acc, a_frag, b_frag, acc)

        # Write back under the claimed C mapping: acc[r] -> C[4g + r, u].
        accv = fx.Vector(fx.memref_load_vec(acc))
        for r in range_constexpr(4):
            row = g * fx.Int32(4) + fx.Int32(r)
            fx.add_offset(fx.get_iter(c_buf), row * fx.Int32(N) + u).store(
                fx.Float32(accv[r])
            )

    @flyc.jit
    def launch(a: fx.Pointer, b: fx.Pointer, c: fx.Pointer, stream: fx.Stream):
        probe_kernel(a, b, c).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    return launch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"# arch: {arch}")
    if "gfx950" not in arch:
        print("SKIP: MFMA(16,16,32) is gfx950-only")
        return 0

    dev = "cuda"
    # Positional codes (see module docstring): vary along both axes, exact in
    # bf16, and asymmetric so a transposed C mapping cannot hide.
    mm = torch.arange(M, dtype=torch.float32, device=dev)[:, None]
    nn = torch.arange(N, dtype=torch.float32, device=dev)[:, None]
    kk = torch.arange(K, dtype=torch.float32, device=dev)[None, :]
    a_f32 = (mm + 1) + (kk + 1) / 4
    b_f32 = 2 * (nn + 1) + (kk + 1) / 4
    a, b = a_f32.to(torch.bfloat16), b_f32.to(torch.bfloat16)
    assert torch.equal(a.float(), a_f32) and torch.equal(b.float(), b_f32)
    c = torch.zeros(M, N, dtype=torch.float32, device=dev)

    launch = build_probe()
    _run_compiled(
        launch,
        ptr_arg(a, fx.BFloat16),
        ptr_arg(b, fx.BFloat16),
        ptr_arg(c, fx.Float32),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    # The claim under test: with A=[M,K] and B=[N,K] both K-major, one MFMA
    # computes C[m,n] = sum_k A[m,k] * B[n,k], i.e. A @ B.T.
    ref = (a.float() @ b.float().T).float()

    # A test that only ever passes proves nothing. Confirm the chosen codes
    # actually separate the mappings we could have gotten wrong, so a PASS
    # below is evidence rather than coincidence.
    a_dup, b_dup = a.float().clone(), b.float().clone()
    a_dup[:, 8:16] = a_dup[:, 0:8]  # lane re-reads k=0..7 for g=1
    b_dup[:, 8:16] = b_dup[:, 0:8]
    for tag, wrong in (
        ("C transposed", ref.T),
        ("operands swapped", b.float() @ a.float().T),
        ("A k-gather wrong", a_dup @ b.float().T),
        ("B k-gather wrong", a.float() @ b_dup.T),
    ):
        assert not torch.equal(wrong, ref), f"codes cannot distinguish {tag}"

    ok = torch.allclose(c, ref, atol=0, rtol=0)
    print(f"exact match to A @ B.T : {ok}")
    if not ok:
        err = (c - ref).abs()
        print(f"  mismatched slots : {int((err > 0).sum())} / {M * N}")
        print(f"  max abs err      : {err.max().item():.6g}")
        # Which permutation did we actually get? Naming it turns a bare
        # failure into a pointer at the line to fix.
        for name, cand in (
            ("B @ A.T (operands swapped)", b.float() @ a.float().T),
            ("C transposed (row/col mapping swapped)", ref.T),
            ("A k-gather wrong (g not scaling k by 8)", a_dup @ b.float().T),
            ("B k-gather wrong (g not scaling k by 8)", a.float() @ b_dup.T),
        ):
            if torch.equal(c, cand):
                print(f"  -> result actually equals: {name}")
                break
        else:
            print("  -> no known mis-mapping matches; inspect the dump below")
        print("\n  got[0, :8]:", c[0, :8].tolist())
        print("  ref[0, :8]:", ref[0, :8].tolist())
        print("\nFAIL: the assumed lane mapping is wrong -- fix it before trusting\n      any kernel built on it.")
        return 1

    # The property the whole design depends on: a lane's 4 accumulators are
    # 4 *different M rows* (tokens) at one N column (feature). If instead they
    # were 4 N columns at one M row, the token reduction would need 4 cross-lane
    # shuffles rather than a register-local fold.
    print("\nchecking the reduction-critical property:")
    lane0_rows = [0, 1, 2, 3]  # g=0 -> C[0..3, u=0]
    vals = c[lane0_rows, 0]
    ref_vals = ref[lane0_rows, 0]
    print(f"  lane 0 holds C[0..3, 0] = {vals.tolist()}")
    print(f"  expected                = {ref_vals.tolist()}")
    assert torch.equal(vals, ref_vals)
    print("  -> one lane holds 4 consecutive M rows at a single N column: CONFIRMED")
    print("  -> with A=K (M=token), token reduction is register-local. Design holds.")

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
