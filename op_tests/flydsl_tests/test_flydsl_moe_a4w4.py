# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for FlyDSL MOE a4w4 kernels.

Tests:
  - Stage1 (gate+up GEMM): flydsl_moe_stage1 with a_dtype="fp4", b_dtype="fp4"
  - Stage2 (down-proj GEMM): flydsl_moe_stage2 with a_dtype="fp4", b_dtype="fp4"
  - Stage2 per-slot output: flydsl_moe_stage2(return_per_slot=True)
  - End-to-end: stage1, activation quantization, and stage2

Usage:
    python op_tests/flydsl_tests/test_flydsl_moe_a4w4.py
    python op_tests/flydsl_tests/test_flydsl_moe_a4w4.py -t 16 128
    python op_tests/flydsl_tests/test_flydsl_moe_a4w4.py --block-m 16 32 64
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import (
    e8m0_shuffle,
    moe_mxfp4_sort,
)

SUPPORTED_GFX = ["gfx942", "gfx950"]
Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
Q_DTYPE_W = dtypes.fp4x2
ATOL = 1.0
RTOL = 0.05
MAX_ERR_RATIO = 0.05
E2E_MAX_ERR_RATIO = 0.10


# ---------------------------------------------------------------------------
# Shared data generation
# ---------------------------------------------------------------------------


def _generate_a4w4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    dtype=torch.bfloat16,
    doweight_stage1: bool = False,
):
    """Generate quantised a4w4 data with torch reference outputs for stage1 and stage2."""
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype, device="cuda") / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda") / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda") / 10
    score = torch.randn((token, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # Quantize weights
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Quantize activation (stage1 input)
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=Q_DTYPE_A)

    # Torch reference: stage1
    ref1 = torch_moe_stage1(
        a1_qt,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=doweight_stage1,
    )

    # Quantize stage2 activation (stage1 output)
    a2_qt, a2_scale = torch_quant(ref1, quant_dtype=Q_DTYPE_A)
    a2_qt = a2_qt.view(token, topk, -1)

    # Torch reference: stage2
    ref2 = torch_moe_stage2(
        a2_qt,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=not doweight_stage1,
    )

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    if doweight_stage1:
        sorted_weights_s1 = sorted_weights
        sorted_weights_s2 = None
    else:
        sorted_weights_s1 = None
        sorted_weights_s2 = sorted_weights

    # Stage1 now follows CK preshuffle for fp4 weights/scales.
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    # Sort activation scales for MoE dispatch
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )
    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[: token * topk, :].view(token, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )

    return {
        # References
        "ref_stage1": ref1,
        "ref_stage2": ref2,
        # Quantised tensors
        "a1_qt": a1_qt,
        "a1_scale": a1_scale,
        "a1_scale_sort": a1_scale_sort,
        "a2_qt": a2_qt,
        "a2_scale": a2_scale,
        "a2_scale_sort": a2_scale_sort,
        "w1_qt": w1_qt,
        "w1_qt_shuf": w1_qt_shuf,
        "w1_scale": w1_scale,
        "w1_scale_shuf": w1_scale_shuf,
        "w2_qt": w2_qt,
        "w2_qt_shuf": w2_qt_shuf,
        "w2_scale": w2_scale,
        "w2_scale_shuf": w2_scale_shuf,
        # Sorting results
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_weights_s1": sorted_weights_s1,
        "sorted_weights_s2": sorted_weights_s2,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        # Shape info
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "dtype": dtype,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E": E,
        "topk": topk,
    }


def _tensor_nbytes(*tensors):
    """Return the total storage bytes represented by the supplied tensors."""
    return sum(t.nbytes for t in tensors if t is not None)


def _out_dtype_name(dtype):
    if dtype == dtypes.bf16:
        return "bf16"
    if dtype == dtypes.fp16:
        return "f16"
    raise ValueError(f"unsupported output dtype: {dtype}")


def _check_output(ref, out, label, max_err_ratio=MAX_ERR_RATIO):
    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        tol_err_ratio=max_err_ratio,
        msg=label,
    )
    assert (
        err < max_err_ratio
    ), f"{label}: mismatch ratio {err:.2%} must be below {max_err_ratio:.2%}"
    return err


# ---------------------------------------------------------------------------
# Stage1 test: FlyDSL flydsl_moe_stage1 a4w4
# ---------------------------------------------------------------------------


@benchmark()
def test_flydsl_stage1_a4w4(
    token, model_dim, inter_dim, num_experts, topk, block_m, dtype
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=num_experts,
        topk=topk,
        block_m=block_m,
        dtype=dtype,
    )
    ref = data["ref_stage1"]
    candidates = {
        "flydsl": lambda: flydsl_moe_stage1(
            a=data["a1_qt"],
            w1=data["w1_qt_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=_out_dtype_name(dtype),
            w1_scale=data["w1_scale_shuf"],
            a1_scale=data["a1_scale_sort"],
            sorted_weights=data["sorted_weights_s1"],
        )
    }

    # Two gate/up GEMMs each cost 2*M*N*K:
    # 2 * 2 * token * topk * inter_dim * model_dim.
    flops = 4 * token * topk * inter_dim * model_dim
    # Lower-bound traffic: actual quantized activation/weight inputs, scales,
    # routing metadata, and the stage1 output tensor.
    input_nbytes = _tensor_nbytes(
        data["a1_qt"],
        data["a1_scale_sort"],
        data["w1_qt_shuf"],
        data["w1_scale_shuf"],
        data["sorted_ids"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = _check_output(ref, out, f"{name}: stage1 a4w4")
        nbytes = input_nbytes + out.nbytes
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# ---------------------------------------------------------------------------
# Stage2 test: FlyDSL flydsl_moe_stage2 a4w4
# ---------------------------------------------------------------------------


@benchmark()
def test_flydsl_stage2_a4w4(
    token, model_dim, inter_dim, num_experts, topk, block_m, mode, dtype
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=num_experts,
        topk=topk,
        block_m=block_m,
        dtype=dtype,
    )
    ref = data["ref_stage2"]
    candidates = {
        "flydsl": lambda: flydsl_moe_stage2(
            inter_states=data["a2_qt"],
            w2=data["w2_qt_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=_out_dtype_name(dtype),
            mode=mode,
            w2_scale=data["w2_scale_shuf"],
            a2_scale=data["a2_scale_sort"],
            sorted_weights=data["sorted_weights_s2"],
        )
    }

    # One down-projection GEMM costs
    # 2 * token * topk * model_dim * inter_dim.
    flops = 2 * token * topk * model_dim * inter_dim
    # Lower-bound traffic: actual quantized activation/weight inputs, scales,
    # routing/weight metadata, and the reduced stage2 output tensor.
    input_nbytes = _tensor_nbytes(
        data["a2_qt"],
        data["a2_scale_sort"],
        data["w2_qt_shuf"],
        data["w2_scale_shuf"],
        data["sorted_ids"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        data["sorted_weights_s2"],
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = _check_output(ref, out, f"{name}: stage2 a4w4 ({mode})")
        nbytes = input_nbytes + out.nbytes
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_flydsl_stage2_a4w4_return_per_slot(
    token, model_dim, inter_dim, num_experts, topk, block_m, mode, dtype
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=num_experts,
        topk=topk,
        block_m=block_m,
        dtype=dtype,
    )
    ref = data["ref_stage2"]
    candidates = {
        "flydsl": lambda: flydsl_moe_stage2(
            inter_states=data["a2_qt"],
            w2=data["w2_qt_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=_out_dtype_name(dtype),
            mode=mode,
            w2_scale=data["w2_scale_shuf"],
            a2_scale=data["a2_scale_sort"],
            return_per_slot=True,
        )
    }

    # The per-slot epilogue changes only the output layout; GEMM work remains
    # 2 * token * topk * model_dim * inter_dim.
    flops = 2 * token * topk * model_dim * inter_dim
    # Lower-bound traffic uses the actual kernel inputs and the larger
    # (token, topk, model_dim) per-slot output. Host-side weighting is not timed.
    input_nbytes = _tensor_nbytes(
        data["a2_qt"],
        data["a2_scale_sort"],
        data["w2_qt_shuf"],
        data["w2_scale_shuf"],
        data["sorted_ids"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        per_slot, us = run_perftest(fn)
        assert per_slot.shape == (token, topk, model_dim), (
            f"{name}: expected shape {(token, topk, model_dim)}, "
            f"got {tuple(per_slot.shape)}"
        )
        assert (
            per_slot.dtype == dtype
        ), f"{name}: expected dtype {dtype}, got {per_slot.dtype}"
        assert per_slot.is_contiguous(), f"{name}: per-slot output must be contiguous"

        out = (
            per_slot.to(dtypes.fp32)
            * data["topk_weights"].to(dtypes.fp32).unsqueeze(-1)
        ).sum(dim=1)
        err = _check_output(ref, out, f"{name}: stage2 a4w4 per-slot ({mode})")
        nbytes = input_nbytes + per_slot.nbytes
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# ---------------------------------------------------------------------------
# End-to-end test: FlyDSL stage1 + stage2 combined
# ---------------------------------------------------------------------------


@benchmark()
def test_flydsl_e2e_a4w4(
    token, model_dim, inter_dim, num_experts, topk, block_m, mode, dtype
):
    """Benchmark stage1, requantization, scale sorting, and stage2 together."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    torch_quant = aiter.get_torch_quant(Q_TYPE)
    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=num_experts,
        topk=topk,
        block_m=block_m,
        dtype=dtype,
    )
    ref = data["ref_stage2"]
    out_dtype = _out_dtype_name(dtype)

    def run_flydsl():
        stage1_out = flydsl_moe_stage1(
            a=data["a1_qt"],
            w1=data["w1_qt_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=out_dtype,
            w1_scale=data["w1_scale_shuf"],
            a1_scale=data["a1_scale_sort"],
            sorted_weights=data["sorted_weights_s1"],
        )

        stage1_flat = stage1_out.view(-1, inter_dim)
        a2_qt_e2e, a2_scale_e2e = torch_quant(stage1_flat, quant_dtype=Q_DTYPE_A)
        a2_qt_e2e = a2_qt_e2e.view(token, topk, -1)
        a2_scale_sort_e2e = moe_mxfp4_sort(
            a2_scale_e2e[: token * topk, :].view(token, topk, -1),
            sorted_ids=data["sorted_ids"],
            num_valid_ids=data["num_valid_ids"],
            token_num=token,
            block_size=block_m,
        )

        return flydsl_moe_stage2(
            inter_states=a2_qt_e2e,
            w2=data["w2_qt_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=out_dtype,
            mode=mode,
            w2_scale=data["w2_scale_shuf"],
            a2_scale=a2_scale_sort_e2e,
            sorted_weights=data["sorted_weights_s2"],
        )

    candidates = {"flydsl": run_flydsl}

    # E2E GEMM work is stage1 + stage2:
    # (4 + 2) * token * topk * model_dim * inter_dim.
    # Quantization and scale-sorting arithmetic is intentionally not called GEMM FLOPs.
    flops = (
        4 * token * topk * inter_dim * model_dim
        + 2 * token * topk * model_dim * inter_dim
    )
    # Lower-bound E2E traffic counts each major read/write at the phase where it
    # occurs: stage1 inputs/output; quantization input/outputs; scale-sort
    # input/output; and stage2 inputs/output. Repeated entries are real traffic.
    input_nbytes = _tensor_nbytes(
        data["a1_qt"],
        data["a1_scale_sort"],
        data["w1_qt_shuf"],
        data["w1_scale_shuf"],
        data["sorted_ids"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        data["ref_stage1"],
        data["ref_stage1"],
        data["a2_qt"],
        data["a2_scale"],
        data["a2_scale"],
        data["sorted_ids"],
        data["num_valid_ids"],
        data["a2_scale_sort"],
        data["a2_qt"],
        data["a2_scale_sort"],
        data["w2_qt_shuf"],
        data["w2_scale_shuf"],
        data["sorted_ids"],
        data["sorted_expert_ids"],
        data["num_valid_ids"],
        data["sorted_weights_s2"],
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = _check_output(ref, out, f"{name}: e2e a4w4 ({mode})", E2E_MAX_ERR_RATIO)
        nbytes = input_nbytes + out.nbytes
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


test_flydsl_stage1_a4w4.__test__ = False
test_flydsl_stage2_a4w4.__test__ = False
test_flydsl_stage2_a4w4_return_per_slot.__test__ = False
test_flydsl_e2e_a4w4.__test__ = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL MOE a4w4 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        description="FlyDSL MOE A4W4 correctness + performance sweep",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[16, 64, 256],
        help="Token counts (default: 16 64 256)",
    )
    parser.add_argument("--model-dim", type=int, nargs="*", default=[7168])
    parser.add_argument("--inter-dim", type=int, nargs="*", default=[256])
    parser.add_argument("-E", "--experts", type=int, nargs="*", default=[256])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[8])
    parser.add_argument("--block-m", type=int, nargs="*", default=[32])
    parser.add_argument(
        "--mode",
        type=str,
        nargs="*",
        default=["atomic"],
        choices=["atomic", "reduce"],
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="Output/reference dtype (default: bf16)",
    )
    args = parser.parse_args()

    stage1_rows = [
        test_flydsl_stage1_a4w4(*case)
        for case in itertools.product(
            args.tokens,
            args.model_dim,
            args.inter_dim,
            args.experts,
            args.topk,
            args.block_m,
            args.dtype,
        )
    ]
    _summarize("FlyDSL MOE a4w4 stage1", stage1_rows)

    stage2_axes = itertools.product(
        args.tokens,
        args.model_dim,
        args.inter_dim,
        args.experts,
        args.topk,
        args.block_m,
        args.mode,
        args.dtype,
    )
    stage2_rows = [test_flydsl_stage2_a4w4(*case) for case in stage2_axes]
    _summarize("FlyDSL MOE a4w4 stage2", stage2_rows)

    per_slot_axes = itertools.product(
        args.tokens,
        args.model_dim,
        args.inter_dim,
        args.experts,
        args.topk,
        args.block_m,
        args.mode,
        args.dtype,
    )
    per_slot_rows = [
        test_flydsl_stage2_a4w4_return_per_slot(*case) for case in per_slot_axes
    ]
    _summarize("FlyDSL MOE a4w4 stage2 per-slot", per_slot_rows)

    e2e_axes = itertools.product(
        args.tokens,
        args.model_dim,
        args.inter_dim,
        args.experts,
        args.topk,
        args.block_m,
        args.mode,
        args.dtype,
    )
    e2e_rows = [test_flydsl_e2e_a4w4(*case) for case in e2e_axes]
    _summarize("FlyDSL MOE a4w4 e2e", e2e_rows)


if __name__ == "__main__":
    main()
