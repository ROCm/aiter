import argparse
from dataclasses import dataclass

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2, moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import run_perftest


@dataclass(frozen=True)
class MoeCase:
    stage1_m: int
    stage1_n: int
    stage1_k: int
    topk: int
    expert: int
    stage2_m: int
    stage2_n: int
    stage2_k: int


STAGE1_SHAPES = [
    (16384, 6144, 3072, 4, 128),
    (256, 6144, 3072, 4, 128),
    (128, 6144, 3072, 4, 128),
    (64, 6144, 3072, 4, 128),
    (48, 6144, 3072, 4, 128),
    (32, 6144, 3072, 4, 128),
    (16, 6144, 3072, 4, 128),
    (8, 6144, 3072, 4, 128),
    (4, 6144, 3072, 4, 128),
    (2, 6144, 3072, 4, 128),
    (1, 6144, 3072, 4, 128),
    (8192, 6144, 3072, 4, 128),
]

STAGE2_SHAPES = [
    (65536, 3072, 3072, 4, 128),
    (1024, 3072, 3072, 4, 128),
    (512, 3072, 3072, 4, 128),
    (256, 3072, 3072, 4, 128),
    (192, 3072, 3072, 4, 128),
    (128, 3072, 3072, 4, 128),
    (64, 3072, 3072, 4, 128),
    (32, 3072, 3072, 4, 128),
    (16, 3072, 3072, 4, 128),
    (8, 3072, 3072, 4, 128),
    (4, 3072, 3072, 4, 128),
    (32768, 3072, 3072, 4, 128),
]


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
    parser.add_argument(
        "--case-index",
        type=int,
        default=-1,
        help="Run one shape case by index (0-based). Default: -1 (run all).",
    )
    return parser.parse_args()


def pick_dtype(dtype_name: str):
    return dtypes.bf16 if dtype_name == "bf16" else dtypes.fp16


def tflops(m: int, n: int, k: int, us: float) -> float:
    if us <= 0:
        return 0.0
    flop = 2.0 * m * n * k
    return flop / (us * 1e-6) / 1e12


def build_cases():
    if len(STAGE1_SHAPES) != len(STAGE2_SHAPES):
        raise ValueError("Stage1/Stage2 shape list lengths must match.")
    cases = []
    for s1, s2 in zip(STAGE1_SHAPES, STAGE2_SHAPES):
        s1_m, s1_n, s1_k, topk1, expert1 = s1
        s2_m, s2_n, s2_k, topk2, expert2 = s2
        if topk1 != topk2 or expert1 != expert2:
            raise ValueError(f"Mismatched topk/expert between {s1} and {s2}.")
        if s2_m != s1_m * topk1:
            raise ValueError(f"Expected stage2 M == stage1 M * topk, got {s1} and {s2}.")
        if s2_n != s1_k:
            raise ValueError(f"Expected stage2 N == stage1 K, got {s1} and {s2}.")
        cases.append(MoeCase(s1_m, s1_n, s1_k, topk1, expert1, s2_m, s2_n, s2_k))
    return cases


def run_case(case: MoeCase, dtype, num_iters: int, num_warmup: int, seed: int):
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    token = case.stage1_m
    model_dim = case.stage1_k
    inter_dim = case.stage2_k

    hidden_states = torch.randn((token, model_dim), dtype=dtype, device=device)
    w1_fp = torch.randn((case.expert, case.stage1_n, model_dim), dtype=dtype, device=device)
    w2_fp = torch.randn((case.expert, model_dim, inter_dim), dtype=dtype, device=device)

    topk_ids = torch.randint(
        low=0,
        high=case.expert,
        size=(token, case.topk),
        dtype=dtypes.i32,
        device=device,
    )
    topk_weights = torch.rand((token, case.topk), dtype=dtypes.fp32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    block_m = 16 if token < 2048 else 32 if token < 16384 else 64

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=case.expert,
        model_dim=model_dim,
        moebuf_dtype=dtype,
        block_size=block_m,
    )

    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_qt, w1_scale = quant(w1_fp, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = quant(w2_fp, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(case.expert, case.stage1_n, model_dim // 2)
    w2_qt = w2_qt.view(case.expert, model_dim, inter_dim // 2)
    w1_qt = shuffle_weight_a16w4(w1_qt, 16, True)
    w2_qt = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale = shuffle_scale_a16w4(w1_scale, case.expert, True)
    w2_scale = shuffle_scale_a16w4(w2_scale, case.expert, False)

    # per_1x32 cktile kernels dereference exp_bias optional in generated code.
    bias1 = torch.zeros((case.expert, case.stage1_n), dtype=dtypes.fp32, device=device)
    bias2 = torch.zeros((case.expert, model_dim), dtype=dtypes.fp32, device=device)

    stage1_out = torch.empty((token, case.topk, inter_dim), dtype=dtype, device=device)
    _, stage1_us = run_perftest(
        cktile_moe_stage1,
        hidden_states,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        stage1_out,
        case.topk,
        block_m=block_m,
        a1_scale=None,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias1=bias1,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )

    stage2_in = cktile_moe_stage1(
        hidden_states,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        stage1_out,
        case.topk,
        block_m=block_m,
        a1_scale=None,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias1=bias1,
    )

    moe_out = torch.empty((token, model_dim), dtype=dtype, device=device)
    _, stage2_us = run_perftest(
        cktile_moe_stage2,
        stage2_in,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        case.topk,
        w2_scale=w2_scale,
        a2_scale=None,
        block_m=block_m,
        sorted_weights=sorted_weights,
        n_pad_zeros=0,
        k_pad_zeros=0,
        bias2=bias2,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    return stage1_us, stage2_us, block_m


def main():
    args = parse_args()
    dtype = pick_dtype(args.dtype)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    cases = build_cases()
    if args.case_index >= len(cases):
        raise ValueError(f"--case-index {args.case_index} out of range [0, {len(cases)-1}].")
    if args.case_index >= 0:
        selected = [(args.case_index, cases[args.case_index])]
    else:
        selected = list(enumerate(cases))

    print("==== MoeFlatmmKernel benchmark (cktile 2-stage, shape sweep) ====")
    print(
        "config: "
        f"dtype={args.dtype} (quantized to mxfp4), num_iters={args.num_iters}, "
        f"num_warmup={args.num_warmup}, seed={args.seed}, cases={len(selected)}"
    )
    print(
        "idx | block_m | stage1(M,N,K) | stage1_us | stage1_tflops | "
        "stage2(M,N,K) | stage2_us | stage2_tflops | total_us"
    )

    for idx, case in selected:
        stage1_us, stage2_us, block_m = run_case(
            case,
            dtype=dtype,
            num_iters=args.num_iters,
            num_warmup=args.num_warmup,
            seed=args.seed + idx,
        )
        stage1_tflops = tflops(case.stage1_m, case.stage1_n, case.stage1_k, stage1_us)
        stage2_tflops = tflops(case.stage2_m, case.stage2_n, case.stage2_k, stage2_us)
        total_us = stage1_us + stage2_us
        print(
            f"{idx:>3} | {block_m:>7} | "
            f"({case.stage1_m},{case.stage1_n},{case.stage1_k}) | "
            f"{stage1_us:>9.2f} | {stage1_tflops:>13.2f} | "
            f"({case.stage2_m},{case.stage2_n},{case.stage2_k}) | "
            f"{stage2_us:>9.2f} | {stage2_tflops:>13.2f} | {total_us:>8.2f}"
        )


if __name__ == "__main__":
    main()
