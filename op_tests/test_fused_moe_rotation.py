# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Precision comparison: fused per-1x32 MXFP4 MoE with vs. without
per-block rotation of weights / activations.

What is the rotation matrix R?
------------------------------
``R`` has shape ``(B, g, g)`` with ``g == 32`` (E8M0 / MXFP4 scale block
size) and ``B == K // g``, where ``K`` is the K (input) axis of the
weight being rotated. Each block ``R[b]`` must be **orthogonal**
(``R[b].T @ R[b] == I_g``) so that rotating both the activation and the
weight along K leaves the matmul exactly invariant in fp32:

    y[n] = sum_k x[k] * w[n, k]                            (original)
    y'[n] = sum_k (R x)[k] * (R w)[n, k]                   (rotated)
    y'[n] = sum_b sum_{g1,g2} x[bg+g1] w[n,bg+g2]
                              * (sum_h R[b,h,g1] R[b,h,g2])
          = y[n]   iff R[b].T @ R[b] = I_g                  (orthogonal)

Sampling is the standard Haar-uniform construction: take ``QR`` of a
Gaussian matrix and use the orthogonal factor ``Q``.

In bf16 / MXFP4 *quantised* execution, the rotation usually improves
output precision because it spreads activation energy more uniformly
within each 32-element block, so the per-block amax (and hence the
E8M0 scale) becomes a tighter fit.

Test
----
Reference          fp32 ``torch_moe`` on the original (un-rotated,
                   un-quantised) weights.
Baseline           ``fused_moe`` with MXFP4 quant of the original
                   weights and **no** rotation.
Rotated            apply ``R1`` to ``w1``'s K axis (= ``model_dim``)
                   and ``R2`` to ``w2``'s K axis (= ``inter_dim``)
                   **before** weight MXFP4 quant; call ``fused_moe``
                   with ``W1_R=R1, W2_R=R2`` so the kernel does the
                   matching activation rotation inline with the fp4
                   quant.

We report the L1 / L∞ / cosine error of both baseline and rotated
paths against the fp32 reference at several
``(token, model_dim, inter_dim)`` configs.

Usage::

    python op_tests/test_fused_moe_rotation.py
    python op_tests/test_fused_moe_rotation.py -t 64 1024 -dim 4096,768
"""
import argparse
import logging

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, fused_moe, torch_moe
from aiter.test_common import benchmark
from aiter.utility import fp4_utils

# Surface the [fused_moe] stage1/stage2 rotation-quant logs to stdout so the
# test output proves the FlyDSL rotation+MXFP4 quant kernel is on the a1/a2
# code path when W1_R / W2_R are supplied (and which `rot_transposed`
# build it picked up).
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
aiter.logger.setLevel(logging.INFO)

torch.set_default_device("cuda")


def _orthogonal_block_R(num_blocks: int, g: int, dtype: torch.dtype) -> torch.Tensor:
    """Return a Haar-uniform per-block orthogonal ``(num_blocks, g, g)``."""
    M = torch.randn(num_blocks, g, g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(M)
    return Q.to(dtype).contiguous()


def _maybe_transpose_storage(R: torch.Tensor, rot_transposed: bool) -> torch.Tensor:
    """Optionally store ``R`` in the ``[b, g, h]`` (transposed) layout and
    tag it with ``R_out.transpose = True`` so ``fused_moe`` knows to
    compile the FlyDSL kernel with ``rot_transposed=True``.

    Math invariance check: the kernel applies, per block,
    ``Y = X @ R.T`` when ``rot_transposed=False`` and
    ``Y = X @ R``   when ``rot_transposed=True``.  Passing
    ``R.transpose(-1, -2).contiguous()`` with the flag set therefore
    yields the same activation as passing ``R`` with the flag unset.
    """
    if not rot_transposed:
        # Default: stored as [b, h, g], kernel applies X @ R.T.
        R_out = R
        R_out.transpose = False
        return R_out
    # Transposed-storage variant: pass R.T contiguous, flag the kernel
    # to load it in [b, g, h] layout.  Mathematically equivalent to the
    # default branch.
    R_T = R.transpose(-1, -2).contiguous()
    R_T.transpose = True
    return R_T


def _apply_block_rotation(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """``y = einsum("...bg, bhg -> ...bh", x.view(*lead, B, g), R)``.

    Rotates the last dim of ``x`` block-wise (block size ``g == R.shape[-1]``).
    Same math as ``aiter.fused_moe.apply_block_rotation``-style references.
    """
    *lead, N = x.shape
    B, g, _ = R.shape
    assert B * g == N, (
        f"_apply_block_rotation: last dim {N} != B*g ({B}*{g}={B*g})"
    )
    return torch.einsum(
        "...bg,bhg->...bh", x.reshape(*lead, B, g), R
    ).reshape(*lead, N)


def _fp4_per_1x32_weight_quant(w: torch.Tensor):
    """Per-row-32-col MXFP4 quant of ``w`` -> ``(qt_uint8, scale_e8m0)``."""
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    return quant(w, quant_dtype=dtypes.fp4x2)


def _err_metrics(y: torch.Tensor, y_ref: torch.Tensor):
    y = y.float()
    y_ref = y_ref.float()
    abs_err = (y - y_ref).abs()
    cos_diff = (
        1.0
        - (y * y_ref).sum()
        / (y.norm() * y_ref.norm() + 1e-12)
    )
    return {
        "mae":      float(abs_err.mean().item()),
        "max":      float(abs_err.max().item()),
        "rmse":     float(abs_err.pow(2).mean().sqrt().item()),
        "cos_diff": float(cos_diff.item()),
    }


def _make_weight(shape, dtype, pattern: str, seed: int):
    """Generate weight tensor following ``pattern``:

    * ``gauss``    -- random N(0, sigma) with small sigma (uniform energy
      across blocks; rotation only helps marginally).
    * ``outlier``  -- N(0, sigma) with ~1% of entries replaced by large
      ±20·sigma outliers (heavy-tailed; rotation's amax-smoothing payoff
      is visible).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sigma = 0.05
    w = torch.randn(*shape, dtype=torch.float32, generator=gen, device="cpu") * sigma
    if pattern == "outlier":
        mask = torch.rand(*shape, generator=gen, device="cpu") < 0.01
        signs = torch.where(
            torch.rand(*shape, generator=gen, device="cpu") < 0.5,
            torch.full_like(w, -1.0),
            torch.full_like(w,  1.0),
        )
        outliers = signs * (sigma * 20.0)
        w = torch.where(mask, outliers, w)
    return w.to(dtype).to("cuda").contiguous()


@benchmark()
def test_fused_moe_rotation(token, model_dim, inter_dim, E, topk, dtype,
                            weight_pattern: str = "gauss", seed: int = 0,
                            rot_transposed: bool = False):
    """Run baseline (no rotation) and rotated paths; return both errors.

    ``rot_transposed`` toggles the ``R.transpose`` attribute convention:
    when ``True`` we pass ``R.transpose(-1, -2).contiguous()`` and tag it
    with ``R.transpose = True``, exercising the FlyDSL kernel's
    ``rot_transposed=True`` build.  Output should match the default
    branch exactly (up to fp4 quantisation noise).
    """
    assert model_dim % 32 == 0, f"model_dim {model_dim} must be % 32 == 0"
    assert inter_dim % 32 == 0, f"inter_dim {inter_dim} must be % 32 == 0"

    torch.manual_seed(seed)
    g = 32

    # --- input + unquantized fp32-precision weights ---
    # x stays standard-normal so token activation entering stage1 is
    # generic; only the weight distribution is varied to expose / hide
    # the rotation benefit.
    x = torch.randn((token, model_dim), dtype=dtype) * 1.0
    w1_fp = _make_weight(
        (E, inter_dim * 2, model_dim), dtype, weight_pattern, seed=seed + 1
    )
    w2_fp = _make_weight(
        (E, model_dim, inter_dim), dtype, weight_pattern, seed=seed + 2
    )
    score = torch.randn((token, E), dtype=dtype)
    topk_weight, topk_ids = fused_topk(x, score, topk, True)
    act_type = ActivationType.Silu  # Silu hits the per_1x32 fp4x2 a path
                                    # which is the branch wired to W1_R/W2_R

    # --- (1) fp32 reference: torch_moe on unrotated, unquantised weights ---
    y_ref = torch_moe(
        x, w1_fp, w2_fp, topk_weight, topk_ids, activation=act_type
    )

    # --- (2) baseline path: MXFP4 quant of unrotated weights, no W*_R ---
    w1_qt_b, w1_scale_b = _fp4_per_1x32_weight_quant(w1_fp)
    w2_qt_b, w2_scale_b = _fp4_per_1x32_weight_quant(w2_fp)
    # fp4x2 weights ship as uint8 buffer with last dim K // 2.
    w1_qt_b = w1_qt_b.view(E, inter_dim * 2, model_dim // 2)
    w2_qt_b = w2_qt_b.view(E, model_dim,     inter_dim // 2)
    w1_scale_b_sh = fp4_utils.e8m0_shuffle(w1_scale_b)
    w2_scale_b_sh = fp4_utils.e8m0_shuffle(w2_scale_b)
    print("===== fused_moe BASELINE (no rotation; expect plain MXFP4 quant logs) =====")
    y_baseline = fused_moe(
        x, w1_qt_b, w2_qt_b, topk_weight, topk_ids,
        quant_type=QuantType.per_1x32,
        activation=act_type,
        w1_scale=w1_scale_b_sh, w2_scale=w2_scale_b_sh,
    )

    # --- (3) rotated path: per-block orthogonal R1, R2; rotate weights
    #         along K BEFORE MXFP4 quant; call fused_moe with W*_R so the
    #         kernel rotates activations the matching way inline. ---
    R1 = _orthogonal_block_R(model_dim // g, g, dtype)
    R2 = _orthogonal_block_R(inter_dim  // g, g, dtype)
    w1_rot_fp = _apply_block_rotation(w1_fp, R1)
    w2_rot_fp = _apply_block_rotation(w2_fp, R2)
    w1_qt_r, w1_scale_r = _fp4_per_1x32_weight_quant(w1_rot_fp)
    w2_qt_r, w2_scale_r = _fp4_per_1x32_weight_quant(w2_rot_fp)
    w1_qt_r = w1_qt_r.view(E, inter_dim * 2, model_dim // 2)
    w2_qt_r = w2_qt_r.view(E, model_dim,     inter_dim // 2)
    w1_scale_r_sh = fp4_utils.e8m0_shuffle(w1_scale_r)
    w2_scale_r_sh = fp4_utils.e8m0_shuffle(w2_scale_r)
    # Hand R off to fused_moe in either default [b,h,g] or transposed
    # [b,g,h] storage; the `.transpose` attribute tells the kernel which
    # build to compile.
    R1_in = _maybe_transpose_storage(R1, rot_transposed)
    R2_in = _maybe_transpose_storage(R2, rot_transposed)
    print(
        f"===== fused_moe ROTATED (W1_R/W2_R set, rot_transposed={rot_transposed}; "
        "expect FlyDSL rotation+MXFP4 quant logs with matching transpose=...) ====="
    )
    y_rotated = fused_moe(
        x, w1_qt_r, w2_qt_r, topk_weight, topk_ids,
        quant_type=QuantType.per_1x32,
        activation=act_type,
        w1_scale=w1_scale_r_sh, w2_scale=w2_scale_r_sh,
        W1_R=R1_in,
        W2_R=R2_in,
    )

    err_b = _err_metrics(y_baseline, y_ref)
    err_r = _err_metrics(y_rotated,  y_ref)

    ret = {
        "token":     token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E":         E,
        "topk":      topk,
        "dtype":     str(dtype).split(".")[-1],
        "rot_T":     rot_transposed,
    }
    # The ``@benchmark`` decorator already records ``weight_pattern`` and
    # ``seed`` from the call-site kwargs, so don't duplicate them here.
    for k, v in err_b.items():
        ret[f"base_{k}"] = v
    for k, v in err_r.items():
        ret[f"rot_{k}"] = v
    ret["rmse_ratio (rot/base)"] = (
        err_r["rmse"] / err_b["rmse"] if err_b["rmse"] > 0 else float("nan")
    )
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "Precision comparison: per_1x32 MXFP4 fused MoE with vs. without "
        "per-block rotation of W1 / W2 via the FlyDSL rotation+quant kernel."
    ),
)
parser.add_argument(
    "-t", "--token", type=int, nargs="*",
    default=[64, 256, 1024],
    help="Token (M) counts to test.  e.g.: -t 64 1024",
)
parser.add_argument(
    "-dim", "--dim", type=str, nargs="*",
    default=["256,128", "4096,768", "7168,2560"],
    help="`model_dim,inter_dim` pairs. e.g.: -dim 4096,768 7168,2560",
)
parser.add_argument(
    "-E", "--experts", type=int, default=8,
    help="Total experts (the test always activates all of them in topk).",
)
parser.add_argument(
    "-topk", "--topk", type=int, default=2,
    help="topk routing.",
)
parser.add_argument(
    "-d", "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16, fp16}",
    help="Activation/weight dtype(s).  e.g.: -d bf16",
)
parser.add_argument(
    "-w", "--weight-pattern",
    nargs="*",
    default=["gauss", "outlier"],
    choices=["gauss", "outlier"],
    help=(
        "Weight init pattern(s) to test. `gauss` is Gaussian (rotation "
        "helps marginally); `outlier` is Gaussian with ~1%% large "
        "outliers (rotation's amax-smoothing payoff is much more "
        "visible)."
    ),
)
parser.add_argument(
    "--rot-transposed",
    nargs="*",
    default=["both"],
    choices=["false", "true", "both"],
    help=(
        "How the rotation tensor is laid out before handoff to fused_moe.\n"
        "  false: pass R as-is in [b, h, g], leave R.transpose = False.\n"
        "         Kernel applies X @ R.T per block (default).\n"
        "  true : pass R.transpose(-1, -2).contiguous() with\n"
        "         R.transpose = True. Kernel applies X @ R per block;\n"
        "         numerically equivalent to `false`.\n"
        "  both : run both `false` and `true`; lets you sanity-check that\n"
        "         the two storage layouts agree."
    ),
)


def main():
    args = parser.parse_args()
    rot_T_vals = []
    for v in args.rot_transposed:
        if v == "both":
            rot_T_vals = [False, True]
            break
        rot_T_vals.append(v == "true")
    # Deduplicate while preserving order.
    rot_T_vals = list(dict.fromkeys(rot_T_vals))

    rows = []
    for dtype in args.dtype:
        for dim in args.dim:
            model_dim, inter_dim = (int(s) for s in dim.split(","))
            for t in args.token:
                for pat in args.weight_pattern:
                    for rot_T in rot_T_vals:
                        rows.append(
                            test_fused_moe_rotation(
                                token=t,
                                model_dim=model_dim,
                                inter_dim=inter_dim,
                                E=args.experts,
                                topk=args.topk,
                                dtype=dtype,
                                weight_pattern=pat,
                                rot_transposed=rot_T,
                            )
                        )
    import pandas as pd

    df = pd.DataFrame(rows)
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        table = df.to_string(index=False)
    aiter.logger.info(
        "fused_moe block-rotation precision comparison:\n%s", table
    )


if __name__ == "__main__":
    main()
