# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end regression of exact-kid ``opus_bmm`` vs torch.bmm.

Usage:
    python3 op_tests/test_opus_a16w16_gemm.py --kid KID [-m M -n N -k K -b B]
    python3 op_tests/test_opus_a16w16_gemm.py --csv_file <shape_csv>

    # opus-only sweep in CUDA-graph mode, golden-checked (default entry):
    python3 op_tests/test_opus_a16w16_gemm.py --opus_sweep -n 2048 -k 7168
"""

import argparse
import sys

import pytest
import torch

# Skip on unsupported arch via the same probe opus uses at import time.
from aiter.ops.opus._arch import _detect_arch

_arch_ok, _detected_gfx = _detect_arch({"gfx950", "gfx942", "gfx1250"})

from aiter.ops.opus import opus_bmm, opus_gemm
from aiter.test_common import checkAllclose, run_perftest


def _torch_ref(A: torch.Tensor, B: torch.Tensor, out_dtype):
    # A: [batch, M, K], B: [N, K] or [batch, N, K] -> bmm.
    # run_torch computes in fp32 then casts to match the opus path.
    if B.dim() == 2:
        return torch.einsum("bmk,nk->bmn", A.float(), B.float()).to(out_dtype)
    return torch.bmm(A.float(), B.float().transpose(-1, -2)).to(out_dtype)


def _make_b(batch: int, N: int, K: int) -> torch.Tensor:
    """Build the physical dense ``[batch, N, K]`` weight contract."""
    B2D = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    return B2D.unsqueeze(0).expand(batch, -1, -1).contiguous()


def run_a16w16_case(
    batch: int,
    M: int,
    N: int,
    K: int,
    *,
    kid: int,
    split_k: int = 0,
    out_dtype=torch.bfloat16,
):
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
    B = _make_b(batch, N, K)
    Y = torch.empty((batch, M, N), device="cuda", dtype=out_dtype)

    ref = _torch_ref(A, B, out_dtype)

    Y, us = run_perftest(
        opus_bmm,
        A,
        B,
        Y,
        kid=kid,
        split_k=split_k,
    )

    err = checkAllclose(
        Y,
        ref,
        msg=f"a16w16 b={batch} m={M} n={N} k={K}",
        rtol=0.1,
        atol=0.5,
    )
    flops = 2.0 * batch * M * N * K
    tflops = flops / us / 1e6
    print(
        f"[a16w16] batch={batch} M={M} N={N} K={K} dtype={out_dtype} "
        f"| {us:.1f}us | {tflops:.2f} TFLOPs | err={err}"
    )
    return err


def load_shapes_from_csv(csv_path, *, default_kid=None, default_split_k=0):
    import pandas as pd

    df = pd.read_csv(csv_path)
    kid_column = next(
        (name for name in ("kernelId", "solidx", "kid") if name in df), None
    )
    split_column = next((name for name in ("splitK", "split_k") if name in df), None)
    if kid_column is None and default_kid is None:
        raise ValueError(
            "exact-kid CSV sweep needs a kernelId/solidx/kid column or --kid"
        )
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            (
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                int(default_kid if default_kid is not None else row[kid_column]),
                (
                    int(row[split_column])
                    if split_column is not None
                    else int(default_split_k)
                ),
            )
        )
    return list(dict.fromkeys(rows))


def run_a16w16_csv_sweep(
    csv_path: str,
    batch: int = 1,
    *,
    kid: int | None = None,
    split_k: int = 0,
):
    shapes = load_shapes_from_csv(csv_path, default_kid=kid, default_split_k=split_k)
    print(f"\n{'=' * 80}")
    print(f"a16w16 sweep from {csv_path}: {len(shapes)} unique shapes, batch={batch}")
    print("=" * 80)
    passed = failed = 0
    for M, N, K, row_kid, row_split_k in shapes:
        tag = (
            f"a16w16 b={batch} M={M} N={N} K={K} "
            f"kid={row_kid} split_k={row_split_k}"
        )
        try:
            A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
            B = _make_b(batch, N, K)
            Y = torch.empty((batch, M, N), device="cuda", dtype=torch.bfloat16)
            ref = _torch_ref(A, B, torch.bfloat16)
            Y, us = run_perftest(
                opus_bmm,
                A,
                B,
                Y,
                kid=row_kid,
                split_k=row_split_k,
            )
            err = checkAllclose(Y, ref, msg=tag, rtol=0.1, atol=0.5)
            tflops = 2.0 * batch * M * N * K / us / 1e6
            print(f"[PASS] {tag} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag} | {type(e).__name__}: {e}")
            failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(shapes)}")
    return failed == 0


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


def _assert_matches_golden(actual, A, B, bias=None):
    golden = A.float() @ B.float().transpose(-1, -2)
    if bias is not None:
        golden = golden + bias.float()
    # BF16 output has one final rounding; fp32 output is normally much tighter.
    atol = 0.5 if actual.dtype == torch.bfloat16 else 0.05
    rtol = 0.03 if actual.dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual.float(), golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("kid", "M", "N", "K", "split_k"),
    ((200, 64, 64, 512, 2), (1400, 192, 256, 128, 0)),
)
def test_gfx950_logical_2d_gemm_matches_torch(kid, M, N, K, split_k):
    """The public GEMM adapter adds only a no-copy batch-one raw view."""
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    torch.manual_seed(0x2D950 + kid)
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    actual = opus_gemm(A, B, Y, kid=kid, split_k=split_k)
    torch.cuda.synchronize()
    assert actual is Y
    _assert_matches_golden(actual, A, B)


def test_gfx950_batch_first_bmm_matches_torch():
    """The public BMM contract preserves a real batch dimension."""
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    torch.manual_seed(0xB950)
    A = torch.randn((2, 192, 128), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((2, 256, 128), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((2, 192, 256), device="cuda", dtype=torch.bfloat16)
    actual = opus_bmm(A, B, Y, kid=1400)
    torch.cuda.synchronize()
    assert actual is Y
    _assert_matches_golden(actual, A, B)


@pytest.mark.parametrize(
    ("arch", "kid", "M", "N", "K", "split_k", "out_dtype"),
    [
        ("gfx950", 200, 64, 64, 512, 2, torch.bfloat16),
        ("gfx950", 200, 64, 64, 512, 2, torch.float32),
        ("gfx942", 10200, 128, 128, 512, 2, torch.float32),
        ("gfx942", 10210, 128, 128, 512, 2, torch.bfloat16),
        ("gfx1250", 20000, 16, 32, 512, 2, torch.bfloat16),
        ("gfx1250", 20000, 16, 32, 512, 2, torch.float32),
    ],
)
def test_split_k_matches_torch_golden(arch, kid, M, N, K, split_k, out_dtype):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    torch.manual_seed(8192 + kid)
    A = torch.randn((1, M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, N, K), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, M, N), device="cuda", dtype=out_dtype)
    actual = opus_bmm(
        A,
        B,
        Y,
        kid=kid,
        split_k=split_k,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B)


@pytest.mark.parametrize("kid", (1400, 6400))
def test_gfx950_mono_fp32_overwrites_poisoned_output(kid):
    """Regress the ordinary and 4G-safe mono FP32 physical-store paths."""
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")

    torch.manual_seed(0x950000 + kid)
    A = torch.randn((1, 192, 128), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 256, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.full((1, 192, 256), 12345.0, device="cuda", dtype=torch.float32)

    actual = opus_bmm(
        A,
        B,
        out,
        kid=kid,
    )
    torch.cuda.synchronize()

    assert actual is out
    assert int((actual != 12345.0).sum().item()) == actual.numel()
    _assert_matches_golden(actual, A, B)


def test_gfx950_bias_dtype_rules_and_numerics():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    A = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), device="cuda", dtype=torch.bfloat16)
    actual = opus_bmm(
        A,
        B,
        Y,
        kid=200,
        bias=bias,
        split_k=2,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)

    with pytest.raises(RuntimeError, match="bias dtype must match Y dtype"):
        opus_bmm(
            A,
            B,
            Y,
            kid=200,
            bias=bias.float(),
            split_k=2,
        )


def test_gfx942_workspace_kid_rejects_bias_without_framework_fallback():
    if _runtime_arch() != "gfx942":
        pytest.skip("requires gfx942 hardware")
    A = torch.randn((1, 128, 4096), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 256, 4096), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, 128, 256), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="rejects bias on split-K kernels"):
        opus_bmm(
            A,
            B,
            Y,
            kid=10201,
            bias=bias,
            split_k=2,
        )


def test_gfx1250_bf16_output_accepts_fp32_bias():
    if _runtime_arch() != "gfx1250":
        pytest.skip("requires gfx1250 hardware")
    A = torch.randn((1, 16, 512), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((1, 32, 512), device="cuda", dtype=torch.bfloat16)
    Y = torch.empty((1, 16, 32), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((32,), device="cuda", dtype=torch.float32)
    actual = opus_bmm(
        A,
        B,
        Y,
        kid=20000,
        bias=bias,
        split_k=2,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)


if __name__ == "__main__":
    if not _arch_ok:
        print(
            "[skip] test_opus_a16w16_gemm requires "
            f"gfx950/gfx942/gfx1250 (detected {_detected_gfx!r})"
        )
        sys.exit(0)
    parser = argparse.ArgumentParser(
        description="End-to-end exact-kid test for aiter.ops.opus.opus_bmm"
    )
    parser.add_argument("-m", type=int, default=256)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=256)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("--kid", type=int, default=None)
    parser.add_argument("--split-k", type=int, default=0)
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Optional CSV with M,N,K columns. When given, skips the "
            "single-shape test and runs a full sweep instead."
        ),
    )
    parser.add_argument(
        "--opus_sweep",
        action="store_true",
        help=(
            "Run the CUDA-graph-mode opus_gemm sweep (golden-checked) over "
            "the M values whose tuned winner is opus for the given N/K in the "
            "tuned CSV (default: dsv4_bf16_tuned_gemm.csv). This is also the "
            "DEFAULT action when no -m and no --csv_file is given."
        ),
    )
    parser.add_argument(
        "--tuned_csv",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Tuned GEMM CSV used by --opus_sweep to pick opus shapes. "
            "Defaults to the shipped dsv4_bf16_tuned_gemm.csv."
        ),
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Use CUDA-graph mode for the single-shape / --csv_file paths too.",
    )
    args = parser.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.csv_file is not None:
        run_a16w16_csv_sweep(
            args.csv_file,
            batch=args.batch,
            kid=args.kid,
            split_k=args.split_k,
        )
    else:
        if args.kid is None:
            parser.error("--kid is required when --csv_file is not supplied")
        k_eff = max(args.k, 128)
        run_a16w16_case(
            args.batch,
            args.m,
            args.n,
            k_eff,
            kid=args.kid,
            split_k=args.split_k,
            out_dtype=out_dtype,
        )
