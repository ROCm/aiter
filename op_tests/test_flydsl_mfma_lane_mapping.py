# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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

import argparse
import itertools

import flydsl.compiler as flyc
import flydsl.expr as fx
import pandas as pd
import pytest
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg, ptr_buf_tensor
from aiter.test_common import benchmark, checkAllclose, run_perftest

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


def run_torch(a, b):
    return a.float() @ b.float().T


@benchmark()
def test_mapping(dtype=torch.bfloat16):
    if dtype != torch.bfloat16:
        raise ValueError("MFMA 16x16x32 probe supports only BF16 operands")
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("MFMA mapping probe requires gfx950")
    dev = "cuda"
    # Positional codes (see module docstring): vary along both axes, exact in
    # bf16, and asymmetric so a transposed C mapping cannot hide.
    mm = torch.arange(M, dtype=torch.float32, device=dev)[:, None]
    nn = torch.arange(N, dtype=torch.float32, device=dev)[:, None]
    kk = torch.arange(K, dtype=torch.float32, device=dev)[None, :]
    a_f32 = (mm + 1) + (kk + 1) / 4
    b_f32 = 2 * (nn + 1) + (kk + 1) / 4
    a, b = a_f32.to(dtype), b_f32.to(dtype)
    assert torch.equal(a.float(), a_f32) and torch.equal(b.float(), b_f32)
    c = torch.zeros(M, N, dtype=torch.float32, device=dev)

    launch = build_probe()

    def run():
        _run_compiled(
            launch,
            ptr_arg(a, fx.BFloat16),
            ptr_arg(b, fx.BFloat16),
            ptr_arg(c, fx.Float32),
            torch.cuda.current_stream().cuda_stream,
        )
        return c

    candidates = {"flydsl": run}
    ret = {"gfx": get_gfx(), "timing": "repeated-buffer"}
    ref = run_torch(a, b)
    for name, fn in candidates.items():
        got, us = run_perftest(fn, num_rotate_args=1)
        err = checkAllclose(ref.float(), got.float(), atol=0, rtol=0, tol_err_ratio=0)
        assert err == 0
        ret.update(
            {
                f"{name} us": us,
                f"{name} TFLOPS": 2 * M * N * K / us / 1e6,
                f"{name} TB/s": (a.numel() * 2 + b.numel() * 2 + c.numel() * 4)
                / us
                / 1e6,
                f"{name} err": err,
            }
        )

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

    assert torch.equal(c, ref)
    assert torch.equal(c[[0, 1, 2, 3], 0], ref[[0, 1, 2, 3], 0])
    return ret


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Single BF16 MFMA lane-mapping probe (fixed M=N=16, K=32; no batch axis)"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[torch.bfloat16],
        default=[torch.bfloat16],
        help="operand dtype list (only bf16 is supported by this instruction)",
    )
    return parser.parse_args(argv)


def test_cli_dtype():
    assert parse_args([]).dtype == [torch.bfloat16]
    assert parse_args(["-d", "bf16"]).dtype == [torch.bfloat16]
    assert parse_args(["--dtype"]).dtype == []


@pytest.mark.parametrize(
    "argv",
    [["-d", "fp16"], ["-d", "fp8"], ["-d", "unknown"], ["-b", "2"], ["-s", "16,16,32"]],
)
def test_cli_invalid_args(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() not in ["gfx950"]:
        aiter.logger.warning("MFMA mapping probe requires gfx950; skipping")
        return
    rows = [test_mapping(dtype) for (dtype,) in itertools.product(args.dtype)]
    if not rows:
        aiter.logger.warning("Empty dtype selection; no MFMA tests executed")
        return
    aiter.logger.info(
        "MFMA mapping summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
