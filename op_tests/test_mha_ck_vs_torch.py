#!/usr/bin/env python3
"""CK flash_attn_func correctness test: run_ck vs run_torch reference.

Modeled after FlyDSL's test_flash_attn_func.py framework with detailed
error comparison, MD5 matching, and threshold distribution.
"""

import argparse
import hashlib
import itertools
import os
import sys

import numpy as np
import torch

import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
)

DEFAULT_SEED = 123
UNIFORM_RANGE = (-1, 1)


def setup_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_md5(tensor: torch.Tensor) -> str:
    return hashlib.md5(
        tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 5,
    thresholds: list = None,
) -> dict:
    """Compare two numpy arrays and compute detailed difference metrics."""
    if thresholds is None:
        thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: arr1 {arr1.shape} vs arr2 {arr2.shape}")

    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)
    result = {"top_k_diff": [], "threshold_stats": [], "nan_info": {}}

    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    if np.any(nan_mask1):
        result["nan_info"]["arr1_nan_count"] = int(np.sum(nan_mask1))
        print(f"  Warning: result contains {result['nan_info']['arr1_nan_count']} NaN values")
    if np.any(nan_mask2):
        result["nan_info"]["arr2_nan_count"] = int(np.sum(nan_mask2))
        print(f"  Warning: reference contains {result['nan_info']['arr2_nan_count']} NaN values")

    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = (diff / (1.0 + np.abs(arr2))).max()
    result["max_diff"] = float(diff.max())
    result["max_diff_thr"] = float(max_diff_thr)

    print(f"  diff.abs.max  = {diff.max():.6f}")
    print(f"  diff.abs.mean = {diff.mean():.6f}")
    print(f"  max_diff_thr (rel) = {max_diff_thr:.6e}")

    flat_diff = diff.flatten()
    actual_k = min(k, len(flat_diff))
    top_k_indices = np.argpartition(flat_diff, -actual_k)[-actual_k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    print(f"  Top-{actual_k} differences:")
    for i in range(actual_k):
        idx = tuple(dim[i] for dim in orig_indices)
        entry = {
            "value": float(diff[idx]),
            "position": idx,
            "arr1_value": float(arr1[idx]),
            "arr2_value": float(arr2[idx]),
        }
        result["top_k_diff"].append(entry)
        print(f"    [{idx}] ck={arr1[idx]:.6f}, ref={arr2[idx]:.6f}, diff={diff[idx]:.6f}")

    print(f"  Threshold distribution ({total_elements} elements):")
    for i in range(len(thresholds) - 1):
        lower, upper = thresholds[i], thresholds[i + 1]
        count = int(np.sum((diff >= lower) & (diff < upper)))
        pct = 100.0 * count / total_elements
        result["threshold_stats"].append(
            {"range": f"[{lower:.0e}, {upper:.0e})", "count": count, "percentage": pct}
        )
        print(f"    [{lower:.0e}, {upper:.0e}): {count:>8d} ({pct:6.2f}%)")

    count = int(np.sum(diff >= thresholds[-1]))
    pct = 100.0 * count / total_elements
    result["threshold_stats"].append(
        {"range": f">={thresholds[-1]:.0e}", "count": count, "percentage": pct}
    )
    print(f"    >={thresholds[-1]:.0e}       : {count:>8d} ({pct:6.2f}%)")

    return result


# ---------------------------------------------------------------------------
# CK kernel under test
# ---------------------------------------------------------------------------

def run_ck(
    q, k, v,
    bias=None, alibi_slopes=None, dout=None,
    dropout_p=0.0, causal=False, window_size=(-1, -1),
    deterministic=False, return_lse=True, return_attn_probs=False,
):
    (out, softmax_lse, S_dmask), us_fwd = run_perftest(
        aiter.flash_attn_func,
        q, k, v,
        dropout_p,
        None,  # softmax_scale
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
        how_v3_bf16_cvt=2,
        num_rotate_args=1,
    )

    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, _) = k.shape
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask, seqlen_q, seqlen_k, None, None, d,
            dropout_p > 0.0, causal=causal, window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    return out, softmax_lse, dropout_mask, us_fwd


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def run_torch(
    q, k, v,
    bias=None, alibi_slopes=None, dout=None,
    dropout_p=0.0, dropout_mask=None, causal=False,
    window_size=(-1, -1), upcast=True, reorder_ops=False,
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, causal=causal
        )
    else:
        attn_bias = None

    out, _, softmax_lse = attention_ref(
        q, k, v, None, None,
        attn_bias, dropout_p, dropout_mask,
        causal=causal, window_size=window_size,
        upcast=upcast, reorder_ops=reorder_ops,
    )

    return out, softmax_lse


# ---------------------------------------------------------------------------
# Single-config test
# ---------------------------------------------------------------------------

def run_config(
    batch_size, nheads, seqlen_q, seqlen_k, d, d_v,
    dtype, causal, local, bias_type, deterministic,
    mha_type, input_layout, dropout_p=0.0,
    seed=DEFAULT_SEED,
):
    results = {}
    setup_seed(seed)
    torch.cuda.empty_cache()

    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    if nheads % nheads_k != 0:
        results["err"] = f"nheads ({nheads}) not divisible by nheads_k ({nheads_k})"
        return results

    window_size = (-1, -1) if not local else tuple(torch.randint(0, seqlen_k, (2,)).tolist())

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d_v, device="cuda", dtype=dtype, requires_grad=True)

    (_, _, _, _, _, _, _, q, k, v, _, _, _) = generate_qkv(
        q, k, v, None, None,
        kvpacked=(input_layout == "KVPACKED"),
        qkvpacked=(input_layout == "QKVPACKED"),
        input_layout=input_layout,
    )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(seqlen_q, seqlen_k, device="cuda", dtype=dtype, requires_grad=True)
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=dtypes.fp32)

    tag = f"B={batch_size} Sq={seqlen_q} Sk={seqlen_k} H={nheads} D={d} Dv={d_v}"
    tag += f" {mha_type} {'causal' if causal else 'nocausal'}"
    tag += f" {'local' if local else ''} bias={bias_type} {dtype}".rstrip()
    results["tag"] = tag

    # --- Run CK ---
    try:
        out, softmax_lse, dropout_mask, us_fwd = run_ck(
            q, k, v, attn_bias, alibi_slopes, None,
            dropout_p, causal, window_size, deterministic,
            return_lse=True, return_attn_probs=True,
        )
    except Exception as e:
        results["err"] = f"run_ck: {e}"
        import traceback; traceback.print_exc()
        return results

    # --- Run reference (upcast fp32) ---
    try:
        out_ref, softmax_lse_ref = run_torch(
            q, k, v, attn_bias, alibi_slopes, None,
            dropout_p, dropout_mask, causal, window_size,
        )
    except Exception as e:
        results["err"] = f"run_torch: {e}"
        import traceback; traceback.print_exc()
        return results

    # --- Run pytorch (non-upcast, for tolerance baseline) ---
    out_pt, _ = run_torch(
        q, k, v, attn_bias, alibi_slopes, None,
        dropout_p, dropout_mask, causal, window_size,
        upcast=False, reorder_ops=True,
    )

    # --- Comparison ---
    out_diff = (out - out_ref).abs().max().item()
    pt_diff = (out_pt - out_ref).abs().max().item()
    out_tol = max(2 * pt_diff, 0.01)

    lse_diff = (softmax_lse - softmax_lse_ref).abs().max().item()

    results["max_err"] = out_diff
    results["pt_diff"] = pt_diff
    results["tol"] = out_tol
    results["lse_diff"] = lse_diff
    results["passed"] = out_diff <= out_tol
    results["us_fwd"] = us_fwd

    # Compute TFLOPS
    s_eff = seqlen_k / 2.0 if causal else float(seqlen_k)
    flops = 4.0 * seqlen_q * s_eff * d * nheads * batch_size
    results["tflops"] = flops / (us_fwd * 1e-6) / 1e12

    # Detailed comparison
    print(f"\n  [{tag}] --- detailed comparison ---")
    result_md5 = compute_md5(out)
    ref_md5 = compute_md5(out_ref)
    print(f"  result_md5 = {result_md5}")
    print(f"  ref_md5    = {ref_md5}")
    if result_md5 == ref_md5:
        print(f"  MD5 match: EXACT (bit-identical)")
    else:
        print(f"  MD5 match: DIFFER")

    compare_arrays(
        out.to(torch.float32).detach().cpu().numpy(),
        out_ref.to(torch.float32).detach().cpu().numpy(),
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="CK flash_attn_func correctness test (run_ck vs run_torch)",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-n", "--nheads", type=int, default=6)
    parser.add_argument("-q", "--seqlen_q", type=int, default=512)
    parser.add_argument("-k", "--seqlen_k", type=int, default=512)
    parser.add_argument(
        "-d_qk_v", type=dtypes.str2tuple, nargs="+",
        default=[(64, 64), (128, 128)],
        help="head_dim pairs, e.g. -d_qk_v 128,128",
    )
    parser.add_argument("-p", "--dropout_p", type=float, default=0.0)
    parser.add_argument(
        "-c", "--causal", action=argparse.BooleanOptionalAction, default=None,
        help="-c for causal, --no-causal for non-causal, default=both",
    )
    parser.add_argument(
        "-l", "--local", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument("-bt", "--bias_type", type=str, default="no", choices=["no", "bias", "alibi"])
    parser.add_argument(
        "-det", "--deterministic", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "-m", "--mha_type", type=str, nargs="+",
        choices=["mha", "mqa", "gqa"], default=["mha"],
    )
    parser.add_argument(
        "-d", "--dtype", type=str, nargs="+",
        choices=["bf16", "fp16"], default=["bf16"],
    )
    parser.add_argument(
        "-i", "--input_layout", type=str,
        choices=["BSHD", "BHSD", "SBHD", "KVPACKED"], default="BSHD",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bf16-fwd-backend", type=str, choices=["auto", "asm", "ck"], default="auto",
    )
    args = parser.parse_args()

    os.environ["AITER_BF16_FWD_BACKEND"] = args.bf16_fwd_backend

    l_causal = [False, True] if args.causal is None else [args.causal]
    l_local = [False, True] if args.local is None else [args.local]
    l_deterministic = [False, True] if args.deterministic is None else [args.deterministic]

    print("=" * 130)
    print("CK flash_attn_func correctness test  (run_ck vs run_torch)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  backend: {args.bf16_fwd_backend}")
    print("=" * 130)

    hdr = (
        f"{'Config':>72s} | {'Status':>6s} | "
        f"{'MaxErr':>9s} {'PtDiff':>9s} {'Tol':>9s} | "
        f"{'LSE_diff':>9s} | {'us':>10s} {'TFLOPS':>8s}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))

    all_passed = True
    n_pass = 0
    n_fail = 0
    n_error = 0

    for dtype_str, (dim_qk, dim_v), mha_type, causal, local, deterministic in itertools.product(
        args.dtype, args.d_qk_v, args.mha_type, l_causal, l_local, l_deterministic,
    ):
        dtype = dtypes.d_dtypes[dtype_str]
        r = run_config(
            args.batch_size, args.nheads,
            args.seqlen_q, args.seqlen_k,
            dim_qk, dim_v, dtype,
            causal, local, args.bias_type,
            deterministic, mha_type, args.input_layout,
            dropout_p=args.dropout_p, seed=args.seed,
        )

        tag = r.get("tag", "?")
        if "err" in r:
            print(f"{tag:>72s} | {'ERROR':>6s} | {r['err'][:60]}")
            all_passed = False
            n_error += 1
            continue

        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
            n_fail += 1
        else:
            n_pass += 1

        us_s = f"{r['us_fwd']:>10.1f}"
        tf_s = f"{r['tflops']:>8.3f}"
        print(
            f"{tag:>72s} | {status:>6s} | "
            f"{r['max_err']:>9.2e} {r['pt_diff']:>9.2e} {r['tol']:>9.2e} | "
            f"{r['lse_diff']:>9.2e} | {us_s} {tf_s}"
        )

    print("=" * 130)
    print(f"Total: {n_pass + n_fail + n_error}  PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_error}")
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
