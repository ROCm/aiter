# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end regression of gemm_a16w16_opus vs torch.bmm; prints TFLOPs.

Usage:
    python3 op_tests/test_opus_a16w16_gemm.py [-m M -n N -k K -b B]
    python3 op_tests/test_opus_a16w16_gemm.py --csv_file <shape_csv>

    # opus-only sweep in CUDA-graph mode, golden-checked (default entry):
    python3 op_tests/test_opus_a16w16_gemm.py --opus_sweep -n 2048 -k 7168
"""

import argparse
import os
import sys

import torch

# Skip on unsupported arch via the same probe opus uses at import time.
from aiter.ops.opus._arch import _detect_arch

_arch_ok, _detected_gfx = _detect_arch({"gfx950", "gfx942", "gfx1250"})
if not _arch_ok:
    print(
        f"[skip] test_opus_a16w16_gemm requires gfx950/gfx942/gfx1250 (detected {_detected_gfx!r})"
    )
    sys.exit(0)

from aiter.ops.opus import gemm_a16w16_opus
from aiter.test_common import checkAllclose, run_perftest

try:
    from aiter.ops.opus import opus_gemm_workspace_init
except Exception:  # noqa: BLE001
    opus_gemm_workspace_init = None


def _graph_capture_stream():
    """The stream torch.cuda.graph captures on when no `stream=` is passed.

    torch lazily creates a single process-global `default_capture_stream`; we
    mirror that here so the opus split-K workspace is registered/grown on the
    exact stream a later `with torch.cuda.graph(g):` (as used by run_perftest's
    graph mode) will capture on.
    """
    g = torch.cuda.graphs.graph
    if getattr(g, "default_capture_stream", None) is None:
        g.default_capture_stream = torch.cuda.Stream()
    return g.default_capture_stream


def _prewarm_opus_graph_workspace(A, B, out_dtype):
    """Eagerly register + size the opus split-K workspace on the capture stream.

    opus split-K kernels keep a per-stream fp32 workspace backed by raw
    hipMalloc; growing it is stream-capture-illegal, so it must be registered
    and grown to the shape's size *eagerly* before HIP graph capture. Without
    this, capturing an opus split-K shape aborts with "splitk workspace not
    initialized for the current CUDA stream". No-op on archs without the
    registry (opus_gemm_workspace_init unavailable) or while already capturing.
    """
    if opus_gemm_workspace_init is None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    s = _graph_capture_stream()
    with torch.cuda.stream(s):
        opus_gemm_workspace_init()
        # Warm the exact shape so the workspace buffer reaches its final size.
        _ = gemm_a16w16_opus(A, B, None, out_dtype)
    s.synchronize()


def _torch_ref(A: torch.Tensor, B: torch.Tensor, out_dtype):
    # A: [batch, M, K], B: [N, K] or [batch, N, K] -> bmm.
    # run_torch computes in fp32 then casts to match the opus path.
    if B.dim() == 2:
        return torch.einsum("bmk,nk->bmn", A.float(), B.float()).to(out_dtype)
    return torch.bmm(A.float(), B.float().transpose(-1, -2)).to(out_dtype)


# --------------------------------------------------------------------------- #
# Data initialization (bf16 operands) + seed
# --------------------------------------------------------------------------- #
DATA_INITS = ("zero", "constant", "uniform", "norm")


def _make_generator(seed):
    """Seeded CUDA generator, or None for the default (unseeded) RNG.

    A fixed seed reproduces both operands bit-for-bit; None keeps the previous
    behavior where every call draws fresh values.
    """
    if seed is None:
        return None
    return torch.Generator(device="cuda").manual_seed(int(seed))


def _make_tensor(shape, dist="norm", gen=None, const_val=1.0):
    """Build a bf16 operand under the requested distribution.

    zero     : all zeros
    constant : filled with ``const_val``
    uniform  : U(-1, 1)
    norm     : N(0, 1)   [default; matches the original torch.randn path]

    ``gen`` seeds the sampled dists (uniform/norm); zero/constant ignore it.
    """
    if dist == "zero":
        return torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    if dist == "constant":
        return torch.full(shape, const_val, device="cuda", dtype=torch.bfloat16)
    if dist == "uniform":
        return torch.empty(shape, device="cuda", dtype=torch.bfloat16).uniform_(
            -1.0, 1.0, generator=gen
        )
    if dist == "norm":
        return torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_(
            0.0, 1.0, generator=gen
        )
    raise ValueError(f"data-init {dist!r}; choose from {DATA_INITS}")


def _make_b(
    batch: int, N: int, K: int, dist: str = "norm", gen=None, const_val: float = 1.0
) -> torch.Tensor:
    """Build a B that gemm_a16w16_opus accepts for both batch=1 and batch>1.

    The wrapper rejects 2D B + batch>1 because the opus launcher hardcodes
    stride_b_batch == N*K (a broadcast view would silently fault). For the
    common "shared weight across batch" case, materialize an explicit
    `[batch, N, K]` tensor via the contiguous broadcast pattern.
    """
    B2D = _make_tensor((N, K), dist, gen, const_val)
    if batch == 1:
        return B2D
    return B2D.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _make_a(
    batch: int, M: int, K: int, dist: str = "norm", gen=None, const_val: float = 1.0
) -> torch.Tensor:
    """Build the [batch, M, K] bf16 activation under the requested dist."""
    return _make_tensor((batch, M, K), dist, gen, const_val)


# --------------------------------------------------------------------------- #
# FLOPS + Bandwidth
# --------------------------------------------------------------------------- #
def _tflops(batch, M, N, K, us):
    """GEMM TFLOPS (2*b*M*N*K FLOP) from microseconds (None-safe)."""
    return (2.0 * batch * M * N * K / us / 1e6) if us else None


def _tbs(batch, M, N, K, us, out_bytes=2):
    """Operand traffic in TB/s (None-safe).

    Bytes moved = A[b,M,K]@bf16 + B@bf16 + out[b,M,N]@out_bytes. B is read once
    per batch when materialized (batch>1) and once as a shared weight (batch=1).
    bytes / (us*1e-6) / 1e12 == bytes / us / 1e6.
    """
    if not us:
        return None
    a = batch * M * K * 2
    b = (batch * N * K if batch > 1 else N * K) * 2
    o = batch * M * N * out_bytes
    return (a + b + o) / us / 1e6


def test_a16w16(
    batch: int,
    M: int,
    N: int,
    K: int,
    out_dtype=torch.bfloat16,
    use_graph=False,
    *,
    dist="norm",
    gen=None,
    const_val=1.0,
    iters=101,
    warmup=2,
    rotate=0,
):
    # gemm_a16w16_opus accepts either 2D or 3D A; test 3D to exercise the
    # batched reshape path. B is 2D when batch==1, 3D contiguous otherwise.
    A = _make_a(batch, M, K, dist, gen, const_val)
    B = _make_b(batch, N, K, dist, gen, const_val)

    ref = _torch_ref(A, B, out_dtype)

    Y, us = run_perftest(
        gemm_a16w16_opus,
        A,
        B,
        None,
        out_dtype,
        testGraph=use_graph,
        num_iters=iters,
        num_warmup=warmup,
        num_rotate_args=rotate,
    )

    err = checkAllclose(
        Y,
        ref,
        msg=f"a16w16 b={batch} m={M} n={N} k={K}",
        rtol=0.1,
        atol=0.5,
    )
    tflops = _tflops(batch, M, N, K, us)
    tbs = _tbs(batch, M, N, K, us, out_bytes=Y.element_size())
    print(
        f"[a16w16] batch={batch} M={M} N={N} K={K} dtype={out_dtype} "
        f"| {us:.1f}us | {tflops:.2f} TFLOPs | {tbs:.3f} TB/s | err={err}"
    )
    return err


def load_shapes_from_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    shapes = list(zip(df["M"].astype(int), df["N"].astype(int), df["K"].astype(int)))
    return list(dict.fromkeys(shapes))


def _default_tuned_csv():
    """Locate the shipped dsv4 tuned GEMM CSV inside the aiter package."""
    import aiter

    return os.path.join(
        os.path.dirname(aiter.__file__),
        "configs",
        "model_configs",
        "dsv4_bf16_tuned_gemm.csv",
    )


def load_opus_shapes(csv_path, gfx, N=None, K=None):
    """Return the opus_gemm rows (with their tuned reference timing).

    Filters the tuned CSV to rows matching the current arch (``gfx``) and,
    optionally, a fixed ``N``/``K``, keeping only rows where ``libtype ==
    'opus'`` (i.e. the shapes for which opus_gemm was selected). Returns a
    list of dicts sorted by M, each with keys: ``M``, ``csv_us`` (the tuned
    latency recorded in the CSV) and ``kernelName``.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    mask = (df["gfx"] == gfx) & (df["libtype"] == "opus")
    if N is not None:
        mask &= df["N"].astype(int) == N
    if K is not None:
        mask &= df["K"].astype(int) == K
    sub = df.loc[mask].copy()
    sub["M"] = sub["M"].astype(int)
    # Keep the first row per M (CSV holds one tuned winner per shape).
    sub = sub.sort_values("M").drop_duplicates(subset="M", keep="first")
    rows = []
    for _, r in sub.iterrows():
        # splitK may be blank/NaN in some rows; keep it as an int when present.
        try:
            splitk = int(r["splitK"]) if not pd.isna(r.get("splitK")) else None
        except (KeyError, ValueError, TypeError):
            splitk = None
        rows.append(
            {
                "M": int(r["M"]),
                "csv_us": float(r["us"]),
                "kernelName": str(r.get("kernelName", "")),
                "splitK": splitk,
            }
        )
    return rows


def test_opus_shapes_graph(
    csv_path,
    gfx,
    N=2048,
    K=7168,
    batch=1,
    out_dtype=torch.bfloat16,
    *,
    dist="norm",
    gen=None,
    const_val=1.0,
    iters=101,
    warmup=2,
    rotate=0,
):
    """CUDA-graph-mode opus_gemm sweep with golden check.

    Selects the M values from the tuned CSV where opus_gemm is the winner
    (for the given arch / N / K), then times each in CUDA-graph mode and
    validates against the torch fp32 reference.
    """
    rows = load_opus_shapes(csv_path, gfx, N=N, K=K)
    ms = [r["M"] for r in rows]
    print(f"\n{'=' * 80}")
    print(
        f"opus graph-mode sweep [{gfx}] N={N} K={K} batch={batch}: "
        f"{len(rows)} opus shapes -> M={ms}"
    )
    print("=" * 80)
    if not rows:
        print(f"[skip] no opus_gemm rows in {csv_path} for gfx={gfx} N={N} K={K}")
        return True

    passed = failed = 0
    perf_rows = []
    out_bytes = torch.empty((), dtype=out_dtype).element_size()
    for r in rows:
        M = r["M"]
        csv_us = r["csv_us"]
        kid = r.get("kernelName") or ""
        splitk = r.get("splitK")
        tag = f"a16w16-graph b={batch} M={M} N={N} K={K}"
        try:
            A = _make_a(batch, M, K, dist, gen, const_val)
            B = _make_b(batch, N, K, dist, gen, const_val)
            ref = _torch_ref(A, B, out_dtype)
            # opus split-K workspace must be grown on the capture stream before
            # run_perftest's graph mode captures this shape.
            _prewarm_opus_graph_workspace(A, B, out_dtype)
            Y, us = run_perftest(
                gemm_a16w16_opus,
                A,
                B,
                None,
                out_dtype,
                testGraph=True,
                num_iters=iters,
                num_warmup=warmup,
                num_rotate_args=rotate,
            )
            err = checkAllclose(Y, ref, msg=tag, rtol=0.1, atol=0.5)
            tflops = _tflops(batch, M, N, K, us)
            tbs = _tbs(batch, M, N, K, us, out_bytes=out_bytes)
            # Ratio of measured graph latency to the tuned CSV reference.
            ratio = (us / csv_us) if csv_us else float("nan")
            splitk_str = "" if splitk in (None, "") else str(splitk)
            print(
                f"[PASS] {tag} | {us:.1f}us (csv {csv_us:.1f}us, "
                f"{ratio:.2f}x) | {tflops:.2f} TFLOPs | {tbs:.3f} TB/s | err={err} "
                f"| splitK={splitk_str or '-'} | kid={kid or '-'}"
            )
            perf_rows.append((M, us, csv_us, ratio, tflops, tbs, splitk, kid))
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag} | {type(e).__name__}: {e}")
            failed += 1

    if perf_rows:
        print(f"\n{'-' * 88}")
        print(f"latency vs tuned CSV [{gfx}] N={N} K={K} batch={batch}")
        print(f"{'-' * 88}")
        print(
            f"{'M':>6} | {'graph us':>10} | {'csv us':>10} | {'ratio':>7} | "
            f"{'TFLOPs':>9} | {'TB/s':>7} | {'splitK':>6} | note | kernel(kid)"
        )
        for M, us, csv_us, ratio, tflops, tbs, splitk, kid in perf_rows:
            # >20% slower than the tuned reference is flagged for a closer look.
            note = "" if ratio <= 1.20 else "SLOW >1.20x"
            splitk_str = "-" if splitk in (None, "") else str(splitk)
            print(
                f"{M:>6} | {us:>10.2f} | {csv_us:>10.2f} | {ratio:>6.2f}x | "
                f"{tflops:>9.2f} | {tbs:>7.3f} | {splitk_str:>6} | "
                f"{note or '':<10} | {kid or '-'}"
            )

    print(f"\nSummary: {passed} passed, {failed} failed out of {len(rows)}")
    return failed == 0


def test_a16w16_csv_sweep(
    csv_path: str,
    batch: int = 1,
    *,
    out_dtype=torch.bfloat16,
    dist="norm",
    gen=None,
    const_val=1.0,
    iters=101,
    warmup=2,
    rotate=0,
    use_graph=False,
):
    shapes = load_shapes_from_csv(csv_path)
    print(f"\n{'=' * 80}")
    print(f"a16w16 sweep from {csv_path}: {len(shapes)} unique shapes, batch={batch}")
    print("=" * 80)
    passed = failed = 0
    out_bytes = torch.empty((), dtype=out_dtype).element_size()
    for M, N, K in shapes:
        tag = f"a16w16 b={batch} M={M} N={N} K={K}"
        try:
            A = _make_a(batch, M, K, dist, gen, const_val)
            B = _make_b(batch, N, K, dist, gen, const_val)
            ref = _torch_ref(A, B, out_dtype)
            if use_graph:
                _prewarm_opus_graph_workspace(A, B, out_dtype)
            Y, us = run_perftest(
                gemm_a16w16_opus,
                A,
                B,
                None,
                out_dtype,
                testGraph=use_graph,
                num_iters=iters,
                num_warmup=warmup,
                num_rotate_args=rotate,
            )
            err = checkAllclose(Y, ref, msg=tag, rtol=0.1, atol=0.5)
            tflops = _tflops(batch, M, N, K, us)
            tbs = _tbs(batch, M, N, K, us, out_bytes=out_bytes)
            print(
                f"[PASS] {tag} | {us:.1f}us | {tflops:.2f} TFLOPs | "
                f"{tbs:.3f} TB/s | err={err}"
            )
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag} | {type(e).__name__}: {e}")
            failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(shapes)}")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end test for aiter.ops.opus.gemm_a16w16_opus"
    )
    parser.add_argument(
        "-m",
        type=int,
        default=None,
        help="Single-shape M. Passing -m forces the single-shape test.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="N (default: 2048 for the opus sweep, 512 for single-shape).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=None,
        help="K (default: 7168 for the opus sweep, 256 for single-shape).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=None,
        help="Batch size. Defaults to 1 for --opus_sweep, else 8.",
    )
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
    # --- warmup / iteration controls ---
    parser.add_argument(
        "--iters",
        type=int,
        default=101,
        help="Timed iterations passed to run_perftest (default: 101).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations passed to run_perftest (default: 2).",
    )
    # --- rotating tensors ---
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        help=(
            "num_rotate_args for run_perftest: number of rotated input copies "
            "used to defeat L2 caching. 0 (default) lets the framework auto-size "
            "the rotation from the L2 cache; 1 disables rotation."
        ),
    )
    # --- data initialization + seed ---
    parser.add_argument(
        "--data-init",
        type=str,
        default="norm",
        choices=list(DATA_INITS),
        help="Operand init distribution: zero/constant/uniform/norm (default: norm).",
    )
    parser.add_argument(
        "--const-val",
        type=float,
        default=1.0,
        help="Fill value used by --data-init constant (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed; fixed => bit-identical operands. Unset => fresh each run.",
    )
    args = parser.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    gen = _make_generator(args.seed)
    init_kwargs = {
        "dist": args.data_init,
        "gen": gen,
        "const_val": args.const_val,
        "iters": args.iters,
        "warmup": args.warmup,
        "rotate": args.rotate,
    }

    # Default action (no -m and no --csv_file): auto-sweep the opus shapes in
    # CUDA-graph mode and print the vs-CSV latency table. So a bare
    # `python3 op_tests/test_opus_a16w16_gemm.py` reproduces the table.
    run_opus_sweep = args.opus_sweep or (args.m is None and args.csv_file is None)

    if run_opus_sweep:
        tuned_csv = args.tuned_csv or _default_tuned_csv()
        batch = args.batch if args.batch is not None else 1
        ok = test_opus_shapes_graph(
            tuned_csv,
            _detected_gfx,
            N=args.n if args.n is not None else 2048,
            K=args.k if args.k is not None else 7168,
            batch=batch,
            out_dtype=out_dtype,
            **init_kwargs,
        )
        sys.exit(0 if ok else 1)
    elif args.csv_file is not None:
        test_a16w16_csv_sweep(
            args.csv_file,
            batch=(args.batch or 8),
            out_dtype=out_dtype,
            use_graph=args.graph,
            **init_kwargs,
        )
    else:
        # Clamp K>=128 so every kid the heuristic picks has K>=B_K (smallest is 128).
        k_eff = max(args.k if args.k is not None else 256, 128)
        test_a16w16(
            args.batch or 8,
            args.m,
            args.n if args.n is not None else 512,
            k_eff,
            out_dtype=out_dtype,
            use_graph=args.graph,
            **init_kwargs,
        )
