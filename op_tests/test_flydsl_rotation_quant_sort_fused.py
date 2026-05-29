# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Before/after precision check for the ``_flydsl_rotation_mxfp4_quant_moe_sort``
swap (two-kernel chain -> single-kernel fused).

This validates the exact change made inside ``aiter.fused_moe``: the
``_flydsl_rotation_mxfp4_quant_moe_sort`` helper used to run a two-kernel
chain, and now runs a single fused kernel. Both are compared here:

  A. BEFORE (``_chained_reference``) -- a verbatim copy of the original
     helper body:
       flydsl_per_1x32_fp4_quant_block_rotation_mfma   (rotation + quant,
                                                         per-source-row scale)
       + mxfp4_moe_sort_fwd                            (scale scatter into the
                                                         MoE-sorted shuffle layout)

  B. AFTER (``_flydsl_rotation_mxfp4_quant_moe_sort``, imported from
     ``aiter.fused_moe``) -- the live function, which dispatches to the
     single fused kernel
     ``build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module``
     (rotation + quant + shuffled scale store in one launch over the
     sorted-row grid).

Both paths share identical MFMA / amax / E8M0 / cvt_fp4 arithmetic, so the
packed fp4x2 bytes and the MoE-sorted E8M0 scale bytes should match
**bit-exactly**.

Test harness
------------
``sorted_ids`` is a full permutation of the (topk-expanded) source rows so
every row is routed exactly once; that makes the fused path (which only
writes routed ``out`` rows) cover every row, so the two ``out`` tensors are
directly comparable. The two ``out_scale`` buffers are ``torch.empty``-
allocated by their respective high-level functions, so only the addresses
the MoE scale-scatter actually wrote (computed via ``_valid_shuffle_addrs``)
are compared.

Default ``cols`` are multiples of 256 so ``scale_n = cols/32`` is a multiple
of 8 (``scaleN_pad == scale_n``); the shuffle-address comparison assumes
this.

Usage::

    python op_tests/test_flydsl_rotation_quant_sort_fused.py
    python op_tests/test_flydsl_rotation_quant_sort_fused.py \
        -t 256 -dim 4096 7168 -topk 1 2 -d bf16 --rot-transposed both
"""
import argparse

import torch

import aiter
from aiter import dtypes, mxfp4_moe_sort_fwd
from aiter.ops.flydsl import flydsl_per_1x32_fp4_quant_block_rotation_mfma
# The exact function fused_moe uses for the rotation+quant+sort step
# (the "after" of this change), plus its rot_R.transpose attr reader.
from aiter.fused_moe import (
    _flydsl_rotation_mxfp4_quant_moe_sort,
    _read_rot_transposed_attr,
)

torch.set_default_device("cuda")

G = 32  # FP4 / E8M0 group size


def _chained_reference(x, rot_R, sorted_ids, num_valid_ids, token_num, cols):
    """Faithful copy of the ORIGINAL ``_flydsl_rotation_mxfp4_quant_moe_sort``
    body (the "before" of this change): the standalone rotation+quant MFMA
    kernel, then a separate ``mxfp4_moe_sort_fwd`` scale-scatter. Returns
    ``(fp4x2, sorted_scale)`` exactly like the function it replaced.
    """
    rot_transposed = _read_rot_transposed_attr(rot_R)
    rows = int(x.shape[0])
    if not x.is_contiguous():
        x = x.contiguous()
    if not rot_R.is_contiguous():
        rot_R = rot_R.contiguous()
    a = torch.empty(rows, cols // 2, dtype=torch.uint8, device=x.device)
    a_scale = torch.empty(rows, cols // 32, dtype=torch.uint8, device=x.device)
    flydsl_per_1x32_fp4_quant_block_rotation_mfma(
        a, x, rot_R, a_scale, rot_transposed=rot_transposed,
    )
    a_scale = mxfp4_moe_sort_fwd(
        a_scale.view(dtypes.fp8_e8m0),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        cols=cols,
    )
    return a.view(dtypes.fp4x2), a_scale


def _valid_shuffle_addrs(num_valid: int, scale_n: int) -> torch.Tensor:
    """Flat byte addresses written by the MoE scale-scatter, i.e.
    ``fp4_scale_shuffle_id(scaleN_pad, sorted_row, b)`` for every
    ``sorted_row in [0, num_valid)`` and ``b in [0, scale_n)``.

    Both implementations allocate their ``out_scale`` with ``torch.empty``,
    so untouched positions hold differing garbage; comparing only these
    valid addresses isolates the bytes both paths actually wrote.
    """
    scaleN_pad = ((scale_n + 7) // 8) * 8
    x = torch.arange(num_valid, device="cuda").view(-1, 1)   # sorted_row
    y = torch.arange(scale_n, device="cuda").view(1, -1)     # b
    addr = (
        (x // 32 * scaleN_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )
    return addr.flatten().to(torch.long)


def _make_sorted_ids(token_num: int, topk: int, seed: int):
    """Full-coverage ``sorted_ids`` (permutation of expanded rows).

    Returns ``(sorted_ids_i32, num_valid_ids_i32, rows)`` where each
    expanded source row ``src = token_idx * topk + topk_id`` appears
    exactly once, encoded as ``(topk_id << 24) | token_idx``. The
    ``sorted_ids`` buffer is padded up to a multiple of 32 with a
    sentinel (``token_num``) and ``num_valid_ids`` records the real
    count, mirroring ``moe_sort_block_fwd`` output.
    """
    rows = token_num * topk
    gen = torch.Generator(device="cpu").manual_seed(seed)
    src = torch.randperm(rows, generator=gen, device="cpu")  # permutation 0..rows-1
    token_idx = src // topk
    topk_id = src % topk
    info = (topk_id << 24) | token_idx  # matches the CUDA decode

    sorted_len = (rows + 31) // 32 * 32
    sorted_ids = torch.full((sorted_len,), token_num, dtype=torch.int32)
    sorted_ids[:rows] = info.to(torch.int32)
    sorted_ids = sorted_ids.to("cuda")
    num_valid_ids = torch.tensor([rows], dtype=torch.int32, device="cuda")
    return sorted_ids, num_valid_ids, rows


def _orthonormal_R(scale_n: int, dtype, seed: int) -> torch.Tensor:
    """Per-block ``(scale_n, 32, 32)`` matrix. Bit-exactness does not
    need orthogonality, but use QR so values stay O(1)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(scale_n, G, G, generator=gen, device="cpu")
    Q, _ = torch.linalg.qr(M)
    return Q.to(dtype).to("cuda").contiguous()


def _run_one(token_num, topk, cols, dtype, rot_transposed, seed=0):
    assert cols % G == 0
    scale_n = cols // G

    sorted_ids, num_valid_ids, rows = _make_sorted_ids(token_num, topk, seed)

    x = (
        torch.randn(rows, cols, generator=torch.Generator(device="cpu").manual_seed(seed + 1), device="cpu")
        .to(dtype)
        .to("cuda")
        .contiguous()
    )
    R = _orthonormal_R(scale_n, dtype, seed + 2)
    # Storage layout for R: default [b, h, g], or transposed [b, g, h]
    # with the ``transpose`` attribute set (both impls read it the same
    # way, so the comparison stays apples-to-apples).
    if rot_transposed:
        R_in = R.transpose(-1, -2).contiguous()
        R_in.transpose = True
    else:
        R_in = R
        R_in.transpose = False

    # ---- A. BEFORE: original chained body (rotation+quant kernel then
    #         mxfp4_moe_sort_fwd) reconstructed verbatim. ----
    out_a, scale_a = _chained_reference(
        x, R_in, sorted_ids, num_valid_ids, token_num, cols,
    )

    # ---- B. AFTER: the real function fused_moe now calls (single fused
    #         rotation+quant+sort kernel). ----
    out_b, scale_b = _flydsl_rotation_mxfp4_quant_moe_sort(
        x, R_in, sorted_ids, num_valid_ids, token_num, cols,
    )

    torch.cuda.synchronize()

    # ---- compare ----
    # fp4 bytes: the chained path writes every source row; the fused path
    # writes only routed rows. ``sorted_ids`` is a full permutation here,
    # so every row is covered and a full compare is valid.
    oa = out_a.view(torch.uint8)
    ob = out_b.view(torch.uint8)
    out_eq = torch.equal(oa, ob)
    out_mism = int((oa != ob).sum().item())

    # scale bytes: both buffers are torch.empty-allocated, so only compare
    # the addresses the MoE scale-scatter actually wrote.
    addrs = _valid_shuffle_addrs(rows, scale_n)
    sa = scale_a.reshape(-1).view(torch.uint8)[addrs]
    sb = scale_b.reshape(-1).view(torch.uint8)[addrs]
    scale_eq = torch.equal(sa, sb)
    scale_mism = int((sa != sb).sum().item())
    scale_max_exp_diff = (
        int((sa.int() - sb.int()).abs().max().item()) if sa.numel() else 0
    )

    status = "PASS" if (out_eq and scale_eq) else "FAIL"
    print(
        f"[{status}] token_num={token_num:5d} topk={topk} cols={cols:5d} "
        f"dtype={str(dtype).split('.')[-1]:5s} rot_T={int(rot_transposed)} | "
        f"fp4 bytes mismatch={out_mism:6d}  scale bytes mismatch={scale_mism:5d} "
        f"(max e8m0 exp diff={scale_max_exp_diff})"
    )
    return out_eq and scale_eq


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "Bit-exact comparison of the two-kernel chain vs. the single-kernel "
        "fused rotation + per-1x32 MXFP4 quant + MoE scale-sort."
    ),
)
parser.add_argument("-t", "--token", type=int, nargs="*", default=[64, 256, 1024],
                    help="token_num values to test.")
parser.add_argument("-dim", "--dim", type=int, nargs="*", default=[256, 4096, 7168],
                    help="cols (model_dim / inter_dim) values to test.")
parser.add_argument("-topk", "--topk", type=int, nargs="*", default=[1, 2],
                    help="topk values (1 -> stage1-like, >1 -> stage2-like).")
parser.add_argument("-d", "--dtype", type=dtypes.str2Dtype,
                    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
                    nargs="*", default=[dtypes.d_dtypes["bf16"]], metavar="{bf16, fp16}",
                    help="activation/rotation dtype(s).")
parser.add_argument("--rot-transposed", nargs="*", default=["false"],
                    choices=["false", "true", "both"],
                    help="rotation tensor storage layout to exercise.")


def main():
    args = parser.parse_args()
    rt_vals = []
    for v in args.rot_transposed:
        if v == "both":
            rt_vals = [False, True]
            break
        rt_vals.append(v == "true")
    rt_vals = list(dict.fromkeys(rt_vals))

    all_pass = True
    for dtype in args.dtype:
        for cols in args.dim:
            for topk in args.topk:
                for t in args.token:
                    for rt in rt_vals:
                        ok = _run_one(t, topk, cols, dtype, rt)
                        all_pass = all_pass and ok

    print("\n" + ("ALL PASS" if all_pass else "SOME FAILED"))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
