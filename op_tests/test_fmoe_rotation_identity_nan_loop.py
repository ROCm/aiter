# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NaN soak test for fused_moe with an *identity* per-block rotation.

Builds an a4w4 (per_1x32 MXFP4 activation + fp4 weight) MoE once for
``expert=400, topk=20, model_dim=4096, inter_dim=1536`` and then repeatedly
calls :func:`aiter.fused_moe` with a *randomly chosen* token count ``M`` each
iteration, asserting the output contains no NaN / Inf.

Identity rotation is a numerical no-op, so a finite, sane output is expected
for every ``M``. The random ``M`` range deliberately straddles the point
where the stage2 activation gather offset ``M * topk * inter_dim`` crosses
2**31 (here ``M ~= 69905``), exercising both the 32-bit ``buffer_load`` fast
path and the 64-bit GEP path of the fused rotation+quant+sort kernel.

Usage::

    HIP_VISIBLE_DEVICES=0 python op_tests/test_fmoe_rotation_identity_nan_loop.py
    HIP_VISIBLE_DEVICES=0 python op_tests/test_fmoe_rotation_identity_nan_loop.py \
        --iters 100 --min-m 1 --max-m 131072 --seed 0
"""
import argparse
import random

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility import fp4_utils

torch.set_default_device("cuda")


def _identity_rot_R(cols, dtype):
    """Per-block identity rotation ``(cols // 32, 32, 32)``."""
    g = 32
    nblk = cols // g
    return (
        torch.eye(g, dtype=dtype, device="cuda")
        .unsqueeze(0)
        .expand(nblk, g, g)
        .contiguous()
    )


def build_moe(E, model_dim, inter_dim, dtype, g1u1=True):
    """Quantize a4w4 (per_1x32 fp4) expert weights once (M-independent)."""
    qType = aiter.QuantType.per_1x32
    torch_quant = aiter.get_torch_quant(qType)

    # use_g1u1=True -> gated FFN, w1 is (E, inter_dim*2, model_dim).
    # use_g1u1=False -> plain FFN, w1 is (E, inter_dim, model_dim).
    w1_rows = inter_dim * 2 if g1u1 else inter_dim
    w1 = torch.randn((E, w1_rows, model_dim), dtype=dtype)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # a4w4 path (preshuffle=False): weights unshuffled, scales e8m0-shuffled.
    w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    return w1_qt, w2_qt, w1_scale_aiter, w2_scale_aiter


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--iters", type=int, default=500, help="loop iterations")
    p.add_argument("--min-m", type=int, default=1, help="min random token count")
    p.add_argument("--max-m", type=int, default=131072, help="max random token count")
    p.add_argument("--expert", type=int, default=400)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=1536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--act", choices=["silu", "gelu"], default="silu",
        help="activation: silu (a4w4 fp4 act) or gelu",
    )
    p.add_argument(
        "--g1u1", type=int, choices=[0, 1], default=1,
        help="1 -> gated FFN (w1 has 2*inter rows); 0 -> plain FFN",
    )
    p.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        default=dtypes.d_dtypes["bf16"],
    )
    args = p.parse_args()

    if get_gfx() not in ["gfx950"]:
        print(f"[skip] per_1x32 fp4 requires gfx950, got {get_gfx()}")
        return 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    E, topk = args.expert, args.topk
    model_dim, inter_dim = args.model_dim, args.inter_dim
    dtype = args.dtype
    qType = aiter.QuantType.per_1x32
    g1u1 = bool(args.g1u1)
    activation = (
        aiter.ActivationType.Gelu if args.act == "gelu"
        else aiter.ActivationType.Silu
    )

    print(
        f"building a4w4 MoE: E={E} topk={topk} model_dim={model_dim} "
        f"inter_dim={inter_dim} dtype={str(dtype).split('.')[-1]} "
        f"act={args.act} g1u1={int(g1u1)}"
    )
    w1_qt, w2_qt, w1_scale, w2_scale = build_moe(
        E, model_dim, inter_dim, dtype, g1u1=g1u1
    )

    # Identity per-block rotation -> numerical no-op (must stay finite).
    W1_R = _identity_rot_R(model_dim, dtype)  # rotates hidden_states (stage1)
    W2_R = _identity_rot_R(inter_dim, dtype)  # rotates a2 (stage2)

    ptr64_boundary = (1 << 31) // (topk * inter_dim)  # M where a2 gather hits 2**31
    print(
        f"looping {args.iters}x, M in [{args.min_m}, {args.max_m}] "
        f"(64-bit gather path engages at M >= {ptr64_boundary + 1})\n"
    )

    failures = []
    for it in range(args.iters):
        M = random.randint(args.min_m, args.max_m)
        input = torch.randn((M, model_dim), dtype=dtype)
        score = torch.randn((M, E), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)

        # NOTE: per_1x32 + ActivationType.Silu selects the fp4 activation
        # (a4w4) path; Swiglu would instead route to a8w4/a16w4 (see
        # fused_moe_: q_dtype_a selection). The fp4 activation path is the
        # one that folds W1_R/W2_R into the rotation+quant+sort kernel.
        out = fused_moe(
            input,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            quant_type=qType,
            activation=activation,
            doweight_stage1=False,
            W1_R=W1_R,
            W2_R=W2_R,
        )
        torch.cuda.synchronize()

        bad = torch.isnan(out) | torch.isinf(out)
        n_nan = int(torch.isnan(out).sum().item())
        n_inf = int(torch.isinf(out).sum().item())
        bad_rows = bad.any(dim=1).nonzero().flatten()
        finite = out[~bad.any(dim=1)]
        amax = finite.abs().float().amax().item() if finite.numel() else float("nan")
        ptr64 = M > ptr64_boundary
        ok = bad_rows.numel() == 0
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] iter={it:3d} M={M:7d} ptr64={int(ptr64)} "
            f"out={tuple(out.shape)} nan={n_nan} inf={n_inf} "
            f"bad_rows={bad_rows.numel()} amax(finite)={amax:.4g}"
        )
        if not ok:
            r0 = int(bad_rows[0])
            full = int(bad[r0].sum()) == model_dim
            print(
                f"         -> bad rows {bad_rows.tolist()[:8]}"
                f"{' ...' if bad_rows.numel() > 8 else ''}; "
                f"first row {r0}: {int(bad[r0].sum())}/{model_dim} bad "
                f"(full_row={full})"
            )
            failures.append((it, M, n_nan, n_inf))

        del input, score, topk_weights, topk_ids, out
        torch.cuda.empty_cache()

    print()
    if failures:
        print(f"SOME FAILED: {len(failures)}/{args.iters} iters produced NaN/Inf")
        for it, M, n_nan, n_inf in failures:
            print(f"  iter={it} M={M} nan={n_nan} inf={n_inf}")
        raise SystemExit(1)
    print(f"ALL PASS: {args.iters}/{args.iters} iters finite (no NaN/Inf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
