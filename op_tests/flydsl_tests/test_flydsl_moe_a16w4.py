# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL MOE a16w4 (bf16 activations, fp4 weights, per_1x32) kernels.

Tests:
  - Stage1 (gate+up GEMM): flydsl_moe_stage1 with a_dtype="bf16", b_dtype="fp4"
  - Stage2 (down-proj GEMM): flydsl_moe_stage2 with a_dtype="bf16", b_dtype="fp4"
  - End-to-end (stage1 -> stage2 combined)

Usage:
    python op_tests/flydsl_tests/test_flydsl_moe_a16w4.py                   # all tests
    python op_tests/flydsl_tests/test_flydsl_moe_a16w4.py --stage stage1    # stage1 only
    python op_tests/flydsl_tests/test_flydsl_moe_a16w4.py --stage stage2    # stage2 only
    python op_tests/flydsl_tests/test_flydsl_moe_a16w4.py --stage e2e       # end-to-end only
    python op_tests/flydsl_tests/test_flydsl_moe_a16w4.py -t 16 -t 128     # specific token counts
"""

import argparse
import sys

import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import (
    e8m0_shuffle,
)

_CUDA = torch.device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_W = dtypes.fp4x2


# ---------------------------------------------------------------------------
# Shared data generation
# ---------------------------------------------------------------------------


def _generate_a16w4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    dtype=torch.bfloat16,
    doweight_stage1: bool = False,
):
    """Generate a16w4 data: bf16 activations, fp4 weights with per_1x32 scales."""
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype, device=_CUDA) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device=_CUDA) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device=_CUDA) / 10
    score = torch.randn((token, E), dtype=dtype, device=_CUDA)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # Quantize weights only (a16w4: activations stay in bf16)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Torch reference: stage1.
    # a1_scale=None signals a16w4 path (bf16 activations, no activation quant).
    ref1 = torch_moe_stage1(
        inp,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=None,  # a16w4: no activation quantization
        w1_scale=w1_scale,
        doweight=doweight_stage1,
    )
    # ref1: [token, topk, inter_dim] post-SwiGLU

    # Stage2 input: ref1 output (bf16, no re-quantization for a16w4 stage2)
    # For a16w4 stage2, activation is bf16, weights are fp4.
    # We pass ref1 directly as the stage2 activation.
    a2 = ref1  # [token, topk, inter_dim] bf16

    # Torch reference: stage2
    ref2 = torch_moe_stage2(
        a2,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,  # a16w4: no activation scale
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

    # Stage1 uses shuffle_weight_a16w4 (gate_up=False: gate+up are separate B-streams,
    # not interleaved — so shuffle as a plain weight matrix, not gate_up layout).
    w1_qt_shuf = shuffle_weight_a16w4(w1_qt, 16, False)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, False)

    # Stage2: w2 uses shuffle_weight_a16w4(gate_up=False) and e8m0_shuffle for scales.
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_shuf = e8m0_shuffle(w2_scale)

    return dict(
        # References
        ref_stage1=ref1,
        ref_stage2=ref2,
        # Activations (bf16 — no quantization)
        inp=inp,
        a2=a2,
        # Weight tensors (shuffled)
        w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_qt_shuf=w2_qt_shuf,
        w2_scale_shuf=w2_scale_shuf,
        # Sorting results
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_weights_s1=sorted_weights_s1,
        sorted_weights_s2=sorted_weights_s2,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        # Shape info
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        dtype=dtype,
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
    )


def _check_result(ref_out, test_out, label, atol=1.0, rtol=0.05, pass_pct=95.0):
    """Compare outputs and print result. Returns (passed, max_delta, pct_close)."""
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct

    print(
        f"  max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    print(f"  ref  sample: {ref_out.reshape(-1)[:8]}")
    print(f"  test sample: {test_out.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta, pct_close


# ---------------------------------------------------------------------------
# Stage1 test: FlyDSL flydsl_moe_stage1 a16w4
# ---------------------------------------------------------------------------


def test_flydsl_stage1_a16w4(
    token: int = 16,
    model_dim: int = 512,
    inter_dim: int = 256,
    E: int = 64,
    topk: int = 4,
    block_m: int = 32,
    k_batch_intra_block: int = 1,
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage1 A16W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, k_batch_intra_block={k_batch_intra_block}"
    )
    print(f"{'='*70}")

    data = _generate_a16w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
        k_batch_intra_block=k_batch_intra_block,
    )
    torch.cuda.synchronize()

    ref = data["ref_stage1"]
    print(f"  ref shape: {ref.shape}, out shape: {out.shape}")
    passed, max_delta, pct_close = _check_result(
        ref, out, "stage1_a16w4", atol=atol, rtol=rtol
    )
    assert (
        passed
    ), f"stage1_a16w4 FAIL: max_delta={max_delta:.4f}, {pct_close:.1f}% close"


# ---------------------------------------------------------------------------
# Stage2 test: FlyDSL flydsl_moe_stage2 a16w4
# ---------------------------------------------------------------------------


def test_flydsl_stage2_a16w4(
    token: int = 16,
    model_dim: int = 512,
    inter_dim: int = 256,
    E: int = 64,
    topk: int = 4,
    block_m: int = 32,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage2 A16W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage2(
        inter_states=data["a2"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage2"]
    print(f"  ref shape: {ref.shape}, out shape: {out.shape}")
    passed, max_delta, pct_close = _check_result(
        ref, out, f"stage2_a16w4_{mode}", atol=atol, rtol=rtol
    )
    assert (
        passed
    ), f"stage2_a16w4_{mode} FAIL: max_delta={max_delta:.4f}, {pct_close:.1f}% close"


# ---------------------------------------------------------------------------
# End-to-end test: FlyDSL stage1 + stage2 combined
# ---------------------------------------------------------------------------


def test_flydsl_e2e_a16w4(
    token: int = 16,
    model_dim: int = 512,
    inter_dim: int = 256,
    E: int = 64,
    topk: int = 4,
    block_m: int = 32,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    """End-to-end test: FlyDSL a16w4 stage1 -> FlyDSL a16w4 stage2."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL E2E A16W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    # Stage1: FlyDSL a16w4
    stage1_out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=None,
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()

    # Stage2: FlyDSL a16w4 using stage1 output directly (no re-quantization)
    e2e_out = flydsl_moe_stage2(
        inter_states=stage1_out,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=256,
        a_dtype="bf16",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=None,
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage2"]
    print(f"  ref shape: {ref.shape}, e2e_out shape: {e2e_out.shape}")
    passed, max_delta, pct_close = _check_result(
        ref, e2e_out, f"e2e_a16w4_{mode}", atol=atol, rtol=rtol, pass_pct=90.0
    )
    assert (
        passed
    ), f"e2e_a16w4_{mode} FAIL: max_delta={max_delta:.4f}, {pct_close:.1f}% close"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL MOE A16W4 (bf16 activations, fp4 weights) unit tests",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=[16, 64, 256],
        help="Token counts to test (default: 16 64 256)",
    )
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=64)
    parser.add_argument("-k", "--topk", type=int, default=4)
    parser.add_argument("--block-m", type=int, nargs="+", default=[32])
    parser.add_argument(
        "--k-batch",
        type=int,
        nargs="+",
        default=[1],
        help="k_batch_intra_block (split-K) values for stage1 sweep (default: 1)",
    )
    parser.add_argument(
        "--mode", type=str, nargs="+", default=["atomic"], choices=["atomic", "reduce"]
    )
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["stage1", "stage2", "e2e"],
        choices=["stage1", "stage2", "e2e"],
        help="Which tests to run (default: all)",
    )
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL is not available. Install flydsl package first.")
        sys.exit(0)

    results = []

    for token in args.tokens:
        for bm in args.block_m:
            if "stage1" in args.stage:
                for kb in args.k_batch:
                    name = f"stage1_a16w4_t{token}_bm{bm}_kb{kb}"
                    try:
                        test_flydsl_stage1_a16w4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            k_batch_intra_block=kb,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append((name, "PASS"))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append((name, "ERROR"))

            if "stage2" in args.stage:
                for mode in args.mode:
                    name = f"stage2_a16w4_t{token}_bm{bm}_{mode}"
                    try:
                        test_flydsl_stage2_a16w4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            mode=mode,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append((name, "PASS"))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append((name, "ERROR"))

            if "e2e" in args.stage:
                for mode in args.mode:
                    name = f"e2e_a16w4_t{token}_bm{bm}_{mode}"
                    try:
                        test_flydsl_e2e_a16w4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            mode=mode,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append((name, "PASS"))
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append((name, "ERROR"))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, status in results:
        print(f"  {status:>5s}  {name}")

    n_pass = sum(1 for _, s in results if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")

    if any(s in ("FAIL", "ERROR") for _, s in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
