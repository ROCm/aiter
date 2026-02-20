import argparse
from dataclasses import dataclass

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import run_perftest


@dataclass(frozen=True)
class MoeShape:
    token: int = 8192
    topk: int = 4
    expert: int = 128
    model_dim: int = 3072
    stage1_n: int = 1024
    stage2_k: int = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark cktile MoeFlatmm stage1/stage2 with fixed MI355 shapes.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Activation dtype for hidden/a2/moe_out.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def pick_dtype(dtype_name: str):
    return dtypes.bf16 if dtype_name == "bf16" else dtypes.fp16


def tflops(m: int, n: int, k: int, us: float) -> float:
    if us <= 0:
        return 0.0
    flop = 2.0 * m * n * k
    return flop / (us * 1e-6) / 1e12


def main():
    args = parse_args()
    shape = MoeShape()
    dtype = pick_dtype(args.dtype)
    device = "cuda"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    hidden_states = torch.randn(
        (shape.token, shape.model_dim), dtype=dtype, device=device
    )
    w1_fp = torch.randn(
        (shape.expert, shape.stage1_n, shape.model_dim), dtype=dtype, device=device
    )
    w2_fp = torch.randn(
        (shape.expert, shape.model_dim, shape.stage2_k), dtype=dtype, device=device
    )

    topk_ids = torch.randint(
        low=0,
        high=shape.expert,
        size=(shape.token, shape.topk),
        dtype=dtypes.i32,
        device=device,
    )
    topk_weights = torch.rand(
        (shape.token, shape.topk), dtype=dtypes.fp32, device=device
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Keep benchmark behavior aligned with fused_moe cktile auto-selection.
    block_m = 16 if shape.token < 2048 else 32 if shape.token < 16384 else 64

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=shape.expert,
        model_dim=shape.model_dim,
        moebuf_dtype=dtype,
        block_size=block_m,
    )

    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_qt, w1_scale = quant(w1_fp, quant_dtype=dtypes.fp4x2) # quantize to mxfp4
    w2_qt, w2_scale = quant(w2_fp, quant_dtype=dtypes.fp4x2) # quantize to mxfp4
    w1_qt = w1_qt.view(shape.expert, shape.stage1_n, shape.model_dim // 2) # reshape to (expert, stage1_n, model_dim // 2)
    w2_qt = w2_qt.view(shape.expert, shape.model_dim, shape.stage2_k // 2) # reshape to (expert, model_dim, stage2_k // 2)
    w1_qt = shuffle_weight_a16w4(w1_qt, 16, True) # shuffle weight
    w2_qt = shuffle_weight_a16w4(w2_qt, 16, False) # shuffle weight
    w1_scale = shuffle_scale_a16w4(w1_scale, shape.expert, True) # shuffle scale
    w2_scale = shuffle_scale_a16w4(w2_scale, shape.expert, False) # shuffle scale
    bias1 = torch.zeros((shape.expert, shape.stage1_n), dtype=dtypes.fp32, device=device)
    bias2 = torch.zeros((shape.expert, shape.model_dim), dtype=dtypes.fp32, device=device)

    stage1_out = torch.empty(
        (shape.token, shape.topk, shape.stage2_k), dtype=dtype, device=device
    )

    _, stage1_us = run_perftest(
        cktile_moe_stage1,
        hidden_states,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        stage1_out,
        shape.topk,
        block_m=block_m,
        a1_scale=None,
        w1_scale=w1_scale,
        sorted_weights=None,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias1=bias1,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
    )

    stage2_in = cktile_moe_stage1(
        hidden_states,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        stage1_out,
        shape.topk,
        block_m=block_m,
        a1_scale=None,
        w1_scale=w1_scale,
        sorted_weights=None,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias1=bias1,
    )

    moe_out = torch.empty((shape.token, shape.model_dim), dtype=dtype, device=device)
    _, stage2_us = run_perftest(
        cktile_moe_stage2,
        stage2_in,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        shape.topk,
        w2_scale=w2_scale,
        a2_scale=None,
        block_m=block_m,
        sorted_weights=sorted_weights,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias2=bias2,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
    )

    stage1_m = shape.token
    stage1_n = shape.stage1_n
    stage1_k = shape.model_dim
    stage2_m = shape.token * shape.topk
    stage2_n = shape.model_dim
    stage2_k = shape.stage2_k

    stage1_tflops = tflops(stage1_m, stage1_n, stage1_k, stage1_us)
    stage2_tflops = tflops(stage2_m, stage2_n, stage2_k, stage2_us)
    total_us = stage1_us + stage2_us

    print("==== MoeFlatmmKernel benchmark (cktile 2-stage) ====")
    print(
        "config: "
        f"dtype={args.dtype} (quantized to mxfp4), token={shape.token}, topk={shape.topk}, expert={shape.expert}, "
        f"block_m={block_m}, num_iters={args.num_iters}, num_warmup={args.num_warmup}, seed={args.seed}"
    )
    print(
        f"stage1 shape: M={stage1_m}, N={stage1_n}, K={stage1_k} "
        f"(expected N(N*2)={shape.stage1_n})"
    )
    print(f"stage2 shape: M={stage2_m}, N={stage2_n}, K={stage2_k}")
    print(f"stage1: {stage1_us:.2f} us, {stage1_tflops:.2f} TFLOPS")
    print(f"stage2: {stage2_us:.2f} us, {stage2_tflops:.2f} TFLOPS")
    print(f"total : {total_us:.2f} us")


if __name__ == "__main__":
    main()
