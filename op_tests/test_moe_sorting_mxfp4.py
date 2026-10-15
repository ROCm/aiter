# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unified perf + accuracy harness for MoE-sort + (optional) MX quant.

Covers two operations, dispatched by ``--quant-dtype``:

  * ``test_moe_mxfp4_sort``: standalone e8m0 byte sort/swizzle
    (``mxfp4_moe_sort_hip`` / Triton). The byte layout is dtype-agnostic
    (same swizzle for MXFP4 and MXFP8 scales), so this test always runs
    and is unaffected by ``--quant-dtype``.

  * ``test_moe_mx_quant_sort``: fused dynamic MX quant + sort. Selects:
      - ``--quant-dtype fp4x2`` (default): MXFP4, packed fp4 output +
        e8m0 scale. Compared paths: ref / split / HIP-fused / Triton.
      - ``--quant-dtype fp8``: MXFP8, fp8 e4m3 output + e8m0 scale.
        Compared paths: ref / split / HIP-fused (no Triton).
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes, fused_moe, get_torch_quant
from aiter.fused_moe import (
    _fused_decode_sort_quant,
    _is_fused_decode_sort_quant_enabled,
    fused_topk,
    moe_sorting,
)
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.quant import (
    fused_dynamic_mxfp4_quant_moe_sort as hip_fused_mxfp4_quant_moe_sort,
)
from aiter.ops.quant import (
    per_1x32_f4_quant_hip,
    per_1x32_f8_scale_f8_quant,
    per_1x32_mx_quant_hip,
)
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(threshold=float("inf"))


def run_torch(scale, sorted_ids, num_valid_ids, token_num):
    topk = 1
    if len(scale.shape) == 3:
        topk = scale.shape[1]
        scale = scale.view(-1, scale.shape[-1])
    sorted_ids[num_valid_ids:] = token_num
    topk_ids = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xFFFFFF
    mask = sorted_ids == token_num
    if topk > 1:
        sorted_ids = sorted_ids * topk + topk_ids
    sorted_ids[mask] = 0  # set to 0 to avoid overflow
    scale = scale[sorted_ids]
    scale.view(torch.uint8)[mask] = 0
    sm, sn = scale.shape
    # Pad both dims to the tile boundaries the swizzle formula assumes:
    # rows -> multiple of 32, cols -> multiple of 8 (e.g. inter_dim=384
    # gives scaleN=12 which must be padded to 16).
    sm_pad = (sm + 31) // 32 * 32
    sn_pad = (sn + 7) // 8 * 8
    tmp = torch.zeros((sm_pad, sn_pad), dtype=scale.dtype, device=scale.device)
    tmp[:sm, :sn] = scale
    scale = tmp
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    ref = scale.view(-1, sn)
    return ref


def run_split_quant_sort(input, sorted_ids, num_valid_ids, token_num, quant_dtype):
    """Two-pass split path: per-token quant + e8m0 byte sort/swizzle.

    fp4 and fp8 share the same `mxfp4_moe_sort_hip` byte shuffle kernel
    (dtype-agnostic); only the per-token quant kernel differs.
    """
    model_dim = input.shape[-1]
    out, scale_per_token = per_1x32_mx_quant_hip(
        input,
        scale=None,
        quant_dtype=quant_dtype,
        scale_type=dtypes.fp8_e8m0 if quant_dtype == dtypes.fp8 else None,
        shuffle=False,
    )
    # Pad cols to multiple of 8 to match the kernel's swizzle stride.
    scaleN_pad = ((model_dim // 32) + 7) // 8 * 8
    out_scale_sorted = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, scaleN_pad),
        dtype=dtypes.fp8_e8m0,
        device=input.device,
    )
    aiter.mxfp4_moe_sort_hip(
        out_scale_sorted,
        scale_per_token,
        sorted_ids,
        num_valid_ids,
        token_num,
        model_dim,
    )
    return out, out_scale_sorted


def _ref_quant(input, quant_dtype):
    """Per-row MX quant reference (no sort). Returns (fp_out, per_token_scale)."""
    if quant_dtype == dtypes.fp4x2:
        return get_torch_quant(aiter.QuantType.per_1x32)(
            input, quant_dtype=dtypes.fp4x2
        )
    else:
        return per_1x32_f8_scale_f8_quant(
            input, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )


@benchmark()
def test_moe_mxfp4_sort(dtype, token_num, model_dim, E, topk, block_size, stage):
    input = torch.randn((token_num, model_dim), dtype=dtype)
    score = torch.randn((token_num, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    sorted_ids, _sorted_weights, _sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
        )
    )
    num_valid_ids = num_valid_ids[0]
    if stage == "stage1":
        scale = torch.arange(token_num * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, model_dim // 32)
        topk = 1
    else:
        scale = torch.arange(token_num * topk * model_dim // 32, dtype=torch.uint8)
        scale = scale.view(token_num, topk, model_dim // 32)
    ref = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)
    triton_scale, triton_us = run_perftest(
        fp4_utils.moe_mxfp4_sort,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        block_size,
    )

    # Pad cols to multiple of 8: the swizzle formula `mx_scale_shuffle_idx`
    # uses `scaleN_pad = pad8(scaleN)`, so the destination buffer needs
    # `pad8(model_dim/32)` cols to avoid OOB writes when scaleN is not
    # already a multiple of 8 (e.g. inter_dim=384 -> scaleN=12 -> 16).
    scaleN_pad = ((model_dim // 32) + 7) // 8 * 8
    hip_scale = torch.zeros(
        ((sorted_ids.shape[0] + 31) // 32 * 32, scaleN_pad),
        dtype=torch.uint8,
        device=input.device,
    )
    _, hip_us = run_perftest(
        aiter.mxfp4_moe_sort_hip,
        hip_scale,
        scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        model_dim,
    )

    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    triton_err = checkAllclose(
        ref[:num_valid_ids],
        triton_scale[:num_valid_ids].view(torch.uint8),
        msg="sorted_mxfp4_scale",
    )

    hip_err = checkAllclose(
        ref[:num_valid_ids].view(torch.uint8),
        hip_scale[:num_valid_ids].view(torch.uint8),
        msg="hip sorted_mxfp4_scale",
    )
    return {
        "triton_us": triton_us,
        "triton_err": triton_err,
        "hip_us": hip_us,
        "hip_err": hip_err,
    }


@benchmark()
def test_moe_mx_quant_sort(
    dtype, token_num, model_dim, E, topk, block_size, stage, quant_dtype
):
    """Unified MXFP4 / MXFP8 quant + sort benchmark.

    Compares 4 paths (3 for fp8 -- Triton is fp4-only):
      * ref:    python ref quant + python sort
      * split:  per_1x32_mx_quant_hip + mxfp4_moe_sort_hip (2 kernels)
      * hip:    fused_dynamic_mx{fp4,fp8}_quant_moe_sort (1 kernel + dispatch)
      * triton: fused_dynamic_mxfp4_quant_moe_sort (Triton, fp4 only)
    """
    if get_gfx().startswith("gfx94"):
        return {}
    is_fp8 = quant_dtype == dtypes.fp8
    label = "mxfp8" if is_fp8 else "mxfp4"
    input = torch.randn((token_num, model_dim), dtype=dtype)
    score = torch.randn((token_num, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    sorted_ids, _sorted_weights, _sorted_expert_ids, num_valid_ids, _moe_buf = (
        moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
        )
    )
    num_valid_ids = num_valid_ids[0]
    topk_orig = topk  # keep before clobbering for stage1
    if stage != "stage1":
        input = torch.randn((token_num * topk, model_dim), dtype=dtype)
    else:
        topk = 1

    # Reference: per-row MX quant + python-side byte sort.
    ref_out, scale = _ref_quant(input, quant_dtype)
    # For stage2 the per-row scale is `(token_num * topk, scaleN)`. Reshape
    # to 3D `(token_num, topk, scaleN)` so `run_torch` sees the topk dim
    # and applies the proper `sorted_ids * topk + topk_ids` indexing
    # instead of treating it as 2D (which collapses every slot onto
    # token_idx 0..token_num-1 and produces ~3-5% byte mismatches versus
    # the HIP/split paths in stage2).
    if stage != "stage1":
        scale = scale.view(token_num, topk_orig, -1)
    ref_scale = run_torch(scale.clone(), sorted_ids.clone(), num_valid_ids, token_num)

    # Split: per_1x32_mx_quant_hip + mxfp4_moe_sort_hip.
    (_split_out, split_scale), split_us = run_perftest(
        run_split_quant_sort,
        input,
        sorted_ids,
        num_valid_ids,
        token_num,
        quant_dtype,
    )

    # HIP fused: the Python wrapper internally dispatches by M (small M ->
    # single fused kernel; large M -> split path). For production-sized M
    # (e.g. 15472) this auto-selects split, matching the split column.
    hip_fn = (
        aiter.fused_dynamic_mxfp8_quant_moe_sort
        if is_fp8
        else aiter.fused_dynamic_mxfp4_quant_moe_sort
    )
    (hip_out, hip_scale), hip_us = run_perftest(
        hip_fn,
        input,
        sorted_ids,
        num_valid_ids,
        token_num,
        topk,
        block_size,
    )

    # Triton path: fp4 only. MUST run BEFORE the `num_valid_ids.item()`
    # below -- the Triton kernel takes `num_valid_ids` as a 0-d tensor
    # pointer (it does `tl.load(num_valid_ids_ptr)` internally), and
    # would otherwise see a Python int and fail at compile time with
    # "Unsupported ptr type triton.language.int32 in `tl.load`".
    triton_scale = None
    triton_us = None
    if not is_fp8:
        (_triton_out, triton_scale), triton_us = run_perftest(
            fused_dynamic_mxfp4_quant_moe_sort,
            input,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size,
        )

    num_valid_ids = num_valid_ids.item()
    num_valid_ids = (num_valid_ids + block_size - 1) // block_size * block_size

    checkAllclose(
        ref_out.view(torch.uint8), hip_out.view(torch.uint8), msg=f"hip {label} out"
    )
    # The wrapper allocates `scale` with `torch.empty` (no zero-init) for
    # perf, so positions the kernel does not write (padding rows within
    # expert blocks; padding cols when `model_dim/32` is not a multiple
    # of 8) contain allocator garbage. Production GEMM consumers never
    # read those positions, but a byte-level `checkAllclose` would
    # otherwise flag them. Mask out positions where ref is 0 (the
    # un-written / zero-init slots), matching the trick the original
    # mxfp4 test applied to the Triton path.
    hip_mask = ref_scale == 0
    hip_scale = hip_scale[: ref_scale.shape[0]]
    hip_scale.view(torch.uint8)[hip_mask] = 0
    split_scale = split_scale[: ref_scale.shape[0]]
    split_scale.view(torch.uint8)[hip_mask] = 0
    hip_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        hip_scale[:num_valid_ids].view(torch.uint8),
        msg=f"hip sorted_{label}_scale",
    )

    split_err = checkAllclose(
        ref_scale[:num_valid_ids].view(torch.uint8),
        split_scale[:num_valid_ids].view(torch.uint8),
        msg=f"split sorted_{label}_scale",
    )

    result = {
        "hip_us": hip_us,
        "hip_err": hip_err,
        "split_us": split_us,
        "split_err": split_err,
    }

    if not is_fp8 and triton_scale is not None:
        mask = ref_scale == 0
        triton_scale = triton_scale[: ref_scale.shape[0]]
        triton_scale.view(torch.uint8)[mask] = 0
        triton_err = checkAllclose(
            ref_scale[:num_valid_ids].view(torch.uint8),
            triton_scale[:num_valid_ids].view(torch.uint8),
            msg=f"triton sorted_{label}_scale",
        )
        result["triton_us"] = triton_us
        result["triton_err"] = triton_err

    return result


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-q",
    "--quant-dtype",
    type=str,
    choices=["fp4x2", "fp8"],
    default="fp4x2",
    help="""MX quant element format for the quant+sort test.
    fp4x2 (default): MXFP4, packed fp4 output. Compares ref / split / HIP / Triton.
    fp8:             MXFP8, fp8 e4m3 output.   Compares ref / split / HIP (no Triton).""",
)
parser.add_argument(
    "-dim1",
    type=int,
    nargs="*",
    default=[4096, 7168],
    help="""Model dimension for stage1.
    e.g.: -dim1 4096""",
)
parser.add_argument(
    "-dim2",
    type=int,
    nargs="*",
    default=[256, 2048],
    help="""Inter dimension for stage2.
    e.g.: -dim2 256""",
)
parser.add_argument(
    "-ek",
    "--expert_topk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[[32, 5], [256, 8], [512, 8]],
    help="""Number of experts.
    e.g.: -ek 32,5""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 64, 128, 256, 1024, 2050, 4200, 10000, 163840],
    help="""M of mnk.
    e.g.: -m 64""",
)
parser.add_argument(
    "-bm",
    "--block_m",
    type=int,
    default=32,
    choices=[16, 32, 64, 80, 128],
    help="""Block M.
    e.g.: -bm 64""",
)

args = parser.parse_args()
_quant_dtype = dtypes.fp4x2 if args.quant_dtype == "fp4x2" else dtypes.fp8
_label = args.quant_dtype  # for log msg

# Standalone byte sort/swizzle test. The underlying kernels
# (`mxfp4_moe_sort_hip`, `fp4_utils.moe_mxfp4_sort`) are dtype-agnostic
# (uint8 byte shuffle), but the Triton path is fp4-only by name, and the
# HIP path is implicitly exercised inside `test_moe_mx_quant_sort`'s split
# path. To avoid the misleading "triton_us" column when running with
# `-q fp8`, only run this test for the fp4 mode.
if _quant_dtype == dtypes.fp4x2:
    df = []
    for dtype in args.dtype:
        for (
            dim,
            (E, topk),
            m,
        ) in itertools.product(args.dim1, args.expert_topk, args.m):
            ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, args.block_m, "stage1")
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("moe_sorting_mxfp4_stage1 summary (markdown):\n%s", df_md)

    df = []
    for dtype in args.dtype:
        for (
            dim,
            (E, topk),
            m,
        ) in itertools.product(args.dim2, args.expert_topk, args.m):
            ret = test_moe_mxfp4_sort(dtype, m, dim, E, topk, args.block_m, "stage2")
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("moe_sorting_mxfp4_stage2 summary (markdown):\n%s", df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim1, args.expert_topk, args.m):
        ret = test_moe_mx_quant_sort(
            dtype, m, dim, E, topk, args.block_m, "stage1", _quant_dtype
        )
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_%s_quant_sort_stage1 summary (markdown):\n%s", _label, df_md)

df = []
for dtype in args.dtype:
    for (
        dim,
        (E, topk),
        m,
    ) in itertools.product(args.dim2, args.expert_topk, args.m):
        ret = test_moe_mx_quant_sort(
            dtype, m, dim, E, topk, args.block_m, "stage2", _quant_dtype
        )
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moe_%s_quant_sort_stage2 summary (markdown):\n%s", _label, df_md)


# ---------------------------------------------------------------------------
# Fused route-sort + MXFP4 activation-quant kernel (gfx950 decode).
#
# `mxfp4_moe_sort_quant_fwd` fuses Opus route sorting, BF16->MXFP4 activation
# quant (per-token E8M0 scales), and routed-output zeroing into one launch.
# It is instantiated only for model_dim=4096, block_m=32, M in [1, 128], and
# the two routes (num_experts, topk) in {(512, 10), (513, 11)}, and is selected
# automatically by the fused_moe gate -- there is no enable switch.
# ---------------------------------------------------------------------------
_FUSED_HIDDEN = 4096
# The last expert of (513, 11) is a fused shared expert every token routes to;
# (512, 10) dispatches its shared expert separately.
_FUSED_SHARED_ROUTE = (513, 11)
_FUSED_ROUTES = ((513, 11), (512, 10))
# Non-power-of-2 counts included on purpose: the contract is M in [1, 128], not
# a whitelist, and partial 32-row tail blocks take a different path.
_FUSED_TOKEN_COUNTS = [1, 2, 3, 4, 8, 16, 17, 32, 64, 100, 127, 128]
# Keyword gate inputs shared by the contract checks below.
_FUSED_GATE_KWARGS = {
    "block_m": 32,
    "quant_type": aiter.QuantType.per_1x32,
    "q_dtype_a": dtypes.fp4x2,
    "q_dtype_w": dtypes.fp4x2,
    "activation": aiter.ActivationType.Silu,
    "is_g1u1": True,
    "expert_mask": None,
    "num_local_tokens": None,
    "need_local_topk_ids": False,
    "run_1stage": False,
    "stage1_is_flydsl": True,
    "hidden_pad": 0,
    "ksplit": 0,
}


def _running_on_gfx950():
    """True only on a live ROCm gfx950 device.

    The kernel lowers to scalef32 fp4-cvt instructions that exist only on
    gfx950 and trap elsewhere. `get_gfx_runtime` (not `get_gfx`) because the
    latter honours GPU_ARCHS and can name an arch that is not installed.
    """
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (KeyError, OSError, RuntimeError):
        return False


def _make_fused_inputs(tokens, experts, topk):
    torch.manual_seed(1000 + tokens)
    hidden = torch.randn((tokens, _FUSED_HIDDEN), dtype=torch.bfloat16, device="cuda")
    if (experts, topk) != _FUSED_SHARED_ROUTE:
        topk_ids = torch.stack(
            [
                torch.randperm(experts, dtype=torch.int32, device="cuda")[:topk]
                for _ in range(tokens)
            ]
        )
        topk_weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    else:
        shared = experts - 1
        routed_ids = torch.stack(
            [
                torch.randperm(shared, dtype=torch.int32, device="cuda")[: topk - 1]
                for _ in range(tokens)
            ]
        )
        topk_ids = torch.cat(
            (
                routed_ids,
                torch.full((tokens, 1), shared, dtype=torch.int32, device="cuda"),
            ),
            dim=1,
        )
        topk_weights = torch.cat(
            (
                torch.rand((tokens, topk - 1), dtype=torch.float32, device="cuda"),
                torch.ones((tokens, 1), dtype=torch.float32, device="cuda"),
            ),
            dim=1,
        )
    return hidden, topk_ids, topk_weights


def _canonical_entries(sorted_ids, sorted_weights, sorted_experts, num_valid_ids, bm):
    valid, tokens = [int(v) for v in num_valid_ids.cpu().tolist()]
    entries = []
    for block, expert in enumerate(sorted_experts[: valid // bm].cpu().tolist()):
        for row in range(block * bm, block * bm + bm):
            packed = int(sorted_ids[row].item())
            token = packed & 0x00FFFFFF
            if token != tokens:
                entries.append(
                    (
                        expert,
                        token,
                        (packed >> 24) & 0xFF,
                        float(sorted_weights[row].item()),
                    )
                )
    return sorted(entries)


def _run_fused(hidden, topk_ids, topk_weights, experts, bm):
    tokens = hidden.shape[0]
    route_count = topk_ids.numel()
    active_experts = min(experts, route_count)
    max_sorted = (route_count + active_experts * (bm - 1) + bm - 1) // bm * bm
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device="cuda")
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device="cuda")
    sorted_experts = torch.empty(
        (max_sorted + bm - 1) // bm, dtype=torch.int32, device="cuda"
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device="cuda")
    moe_buf = torch.empty((tokens, _FUSED_HIDDEN), dtype=torch.bfloat16, device="cuda")
    activation_quant = torch.empty(
        (tokens, _FUSED_HIDDEN // 2), dtype=dtypes.fp4x2, device="cuda"
    )
    activation_scale_token = torch.empty(
        (tokens, _FUSED_HIDDEN // 32), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    aiter.mxfp4_moe_sort_quant_fwd(
        hidden,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid_ids,
        moe_buf,
        activation_quant,
        activation_scale_token,
        experts,
    )
    return (
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid_ids,
        moe_buf,
        activation_quant,
        activation_scale_token,
    )


@benchmark()
def test_mxfp4_fused_sort_quant(dtype, token_num, E, topk, block_size):
    """Fused sort+quant vs Opus sort + generic MXFP4 quant, plus its latency.

    Records the kernel latency and the byte-level quant/scale diffs; the sort
    layout, valid counts, and output zeroing are exact contracts and assert.
    """
    hidden, topk_ids, topk_weights = _make_fused_inputs(token_num, E, topk)

    ref_ids, ref_weights, ref_experts, ref_valid, _ = moe_sorting(
        topk_ids, topk_weights, E, _FUSED_HIDDEN, dtypes.bf16, block_size
    )
    ref_quant, _ = hip_fused_mxfp4_quant_moe_sort(
        hidden,
        sorted_ids=ref_ids,
        num_valid_ids=ref_valid,
        token_num=token_num,
        topk=topk,
        block_size=block_size,
        sorted_weights=ref_weights,
    )
    compact_quant, compact_scale = per_1x32_f4_quant_hip(hidden, shuffle=False)

    fused, fused_us = run_perftest(
        _run_fused, hidden, topk_ids, topk_weights, E, block_size
    )
    f_ids, f_weights, f_experts, f_valid, f_buf, f_quant, f_scale = fused

    assert _canonical_entries(
        ref_ids, ref_weights, ref_experts, ref_valid, block_size
    ) == _canonical_entries(f_ids, f_weights, f_experts, f_valid, block_size)
    torch.testing.assert_close(f_valid, ref_valid, rtol=0, atol=0)
    torch.testing.assert_close(f_buf, torch.zeros_like(f_buf))

    quant_err = checkAllclose(
        f_quant.view(torch.uint8),
        ref_quant.view(torch.uint8),
        msg="fused vs generic mxfp4 quant",
    )
    checkAllclose(
        f_quant.view(torch.uint8),
        compact_quant.view(torch.uint8),
        msg="fused vs compact mxfp4 quant",
    )
    scale_err = checkAllclose(
        f_scale.view(torch.uint8),
        compact_scale.view(torch.uint8),
        msg="fused per-token e8m0 scale",
    )
    return {"fused_us": fused_us, "quant_err": quant_err, "scale_err": scale_err}


def _check_fused_decode_sort_quant_wrapper():
    """`_fused_decode_sort_quant` returns sorted (default) and compact scales."""
    for tokens in (2, 4):
        hidden, topk_ids, topk_weights = _make_fused_inputs(tokens, 512, 10)
        result = _fused_decode_sort_quant(
            hidden,
            topk_ids,
            topk_weights,
            model_dim=_FUSED_HIDDEN,
            global_experts=512,
            block_m=32,
            moebuf_dtype=torch.bfloat16,
            accumulate=True,
        )
        sorted_scale = result[-1]
        assert sorted_scale.shape[0] == result[0].shape[0]
        assert sorted_scale.shape[1] == _FUSED_HIDDEN // 32
        compact = _fused_decode_sort_quant(
            hidden,
            topk_ids,
            topk_weights,
            model_dim=_FUSED_HIDDEN,
            global_experts=512,
            block_m=32,
            moebuf_dtype=torch.bfloat16,
            accumulate=True,
            compact_scale=True,
        )
        assert compact[-1].shape == (tokens, _FUSED_HIDDEN // 32)


def _check_fused_gate():
    """The gate must reject any shape/route the kernel is not instantiated for.

    AITER_CHECK aborts the process instead of raising, so an unsupported shape
    reaching the op is fatal -- this gate is the only guard.
    """
    table = [
        (512, 10, 100, True),  # non power-of-2 M accepted
        (513, 11, 8, True),  # fused-shared route accepted
        (512, 8, 8, False),  # uninstantiated topk
        (256, 10, 8, False),  # uninstantiated experts
        (512, 10, 129, False),  # M above kernel contract
    ]
    for experts, topk, tokens, expected in table:
        pool = min(experts, 512)
        topk_ids = torch.stack(
            [
                torch.randperm(pool, dtype=torch.int32, device="cuda")[:topk]
                for _ in range(tokens)
            ]
        )
        topk_weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
        hidden = torch.randn(
            (tokens, _FUSED_HIDDEN), dtype=torch.bfloat16, device="cuda"
        )
        assert (
            _is_fused_decode_sort_quant_enabled(
                hidden,
                topk_ids,
                topk_weights,
                global_experts=experts,
                **_FUSED_GATE_KWARGS,
            )
            is expected
        ), (experts, topk, tokens)


def _check_fused_gate_uses_runtime_arch():
    """Runtime dispatch must ignore a potentially different build target."""
    hidden, topk_ids, topk_weights = _make_fused_inputs(8, 512, 10)
    original = fused_moe.get_gfx_runtime
    for arch in ("gfx942", None):

        def detect(_arch=arch):
            if _arch is None:
                raise RuntimeError("runtime architecture unavailable")
            return _arch

        fused_moe.get_gfx_runtime = detect
        try:
            assert not _is_fused_decode_sort_quant_enabled(
                hidden,
                topk_ids,
                topk_weights,
                global_experts=512,
                **_FUSED_GATE_KWARGS,
            )
        finally:
            fused_moe.get_gfx_runtime = original


def _check_fused_disable_toggle():
    """The internal base-vs-head hook forces the generic path when set."""
    hidden, topk_ids, topk_weights = _make_fused_inputs(8, 512, 10)
    assert _is_fused_decode_sort_quant_enabled(
        hidden, topk_ids, topk_weights, global_experts=512, **_FUSED_GATE_KWARGS
    )
    fused_moe._fused_decode_sort_quant_disabled = True
    try:
        assert not _is_fused_decode_sort_quant_enabled(
            hidden, topk_ids, topk_weights, global_experts=512, **_FUSED_GATE_KWARGS
        )
    finally:
        fused_moe._fused_decode_sort_quant_disabled = False


def _check_fused_cuda_graph_replay_safe(experts, topk, block_size):
    hidden, topk_ids, topk_weights = _make_fused_inputs(4, experts, topk)
    # Compile/warm up before capture: the graph only records the fixed-address
    # launch; every buffer is preallocated.
    _run_fused(hidden, topk_ids, topk_weights, experts, block_size)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused = _run_fused(hidden, topk_ids, topk_weights, experts, block_size)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    assert int(fused[3][1].item()) == 4
    torch.testing.assert_close(fused[4], torch.zeros_like(fused[4]))


if _running_on_gfx950():
    _check_fused_gate()
    _check_fused_gate_uses_runtime_arch()
    _check_fused_disable_toggle()
    _check_fused_decode_sort_quant_wrapper()

    df = []
    for dtype in args.dtype:
        for (E, topk), m in itertools.product(_FUSED_ROUTES, _FUSED_TOKEN_COUNTS):
            df.append(test_mxfp4_fused_sort_quant(dtype, m, E, topk, 32))
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mxfp4_fused_sort_quant summary (markdown):\n%s", df_md)

    for E, topk in _FUSED_ROUTES:
        _check_fused_cuda_graph_replay_safe(E, topk, 32)
