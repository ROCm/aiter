import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from itertools import combinations

sys.path.insert(0, os.environ.get("AITER_REPO", "/tmp/aiter-pr3470"))

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.test_common import run_perftest
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.utility.fp4_utils import e8m0_shuffle


@dataclass(frozen=True)
class Shape:
    NE: int
    H: int
    INTER: int
    TOPK: int


KIMI = Shape(NE=385, H=7168, INTER=512, TOPK=9)


def build_weights(shape: Shape, device, seed=0):
    torch.manual_seed(seed)
    ne, h, inter = shape.NE, shape.H, shape.INTER
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((ne, 2 * inter, h), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((ne, h, inter), dtype=dtypes.bf16, device=device) / 10
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    # Default tuned FP4/FlyDSL rows use the legacy preshuffle layout from
    # op_tests/test_moe_2stage.py for (q_dtype_a=fp4x2, q_dtype_w=fp4x2).
    fly_w1 = shuffle_weight(w1_qt, layout=(16, 16))
    fly_w2 = shuffle_weight(w2_qt, layout=(16, 16))
    fly = dict(
        w1=fly_w1,
        w2=fly_w2,
        w1_scale=e8m0_shuffle(w1_scale),
        w2_scale=e8m0_shuffle(w2_scale),
    )

    # PR #3470 mxfp4_moe rows are selected by the shuffle_kind tag and use the
    # a16w4 gate/up-interleaved weight/scale layout.
    mx_w1 = shuffle_weight_a16w4(w1_qt, 16, True)
    mx_w1.shuffle_kind = "mxfp4_moe"
    mx = dict(
        w1=mx_w1,
        w2=shuffle_weight_a16w4(w2_qt, 16, False),
        w1_scale=shuffle_scale_a16w4(w1_scale, ne, True),
        w2_scale=shuffle_scale_a16w4(w2_scale, ne, False),
    )
    return fly, mx


def build_inputs(shape: Shape, M: int, device, seed=1):
    torch.manual_seed(seed)
    ne, h, topk = shape.NE, shape.H, shape.TOPK
    hidden = torch.randn((M, h), dtype=dtypes.bf16, device=device) / 10
    n_routed = ne - 1
    shared_id = ne - 1
    n_topk_routed = topk - 1
    g = torch.Generator(device=device).manual_seed(seed)
    bias = torch.randn(n_routed, generator=g, device=device) * 0.5
    scores = torch.randn(M, n_routed, generator=g, device=device) + bias
    routed_w, routed_ids = torch.topk(scores.softmax(-1), n_topk_routed, dim=-1)
    shared_ids = torch.full((M, 1), shared_id, device=device, dtype=routed_ids.dtype)
    shared_w = torch.ones((M, 1), device=device, dtype=routed_w.dtype)
    topk_ids = torch.cat([shared_ids, routed_ids], dim=1).to(torch.int32)
    topk_weight = torch.cat([shared_w, routed_w], dim=1).to(torch.float32)
    return hidden, topk_ids, topk_weight


def make_fn(hidden, topk_ids, topk_weight, w):
    def fn():
        return fused_moe(
            hidden,
            w["w1"],
            w["w2"],
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
        )

    return fn


def make_mx_fn(hidden, topk_ids, topk_weight, w, fly_gemms):
    """make_fn variant that pins the mutable ``gemm{1,2}_backend`` flags on every
    call, so the same ``mx_w`` tensors can back both the pure-HIP and FlyDSL
    gemm paths without cross-iteration state leaking between benchmarks.

    ``fly_gemms`` is the set of gemm indices (subset of {1, 2}) routed through
    FlyDSL; the rest fall back to the pure-HIP backend.
    """
    g1 = "flydsl" if 1 in fly_gemms else None
    g2 = "flydsl" if 2 in fly_gemms else None

    def fn():
        w["w1"].gemm1_backend = g1
        w["w2"].gemm2_backend = g2
        return fused_moe(
            hidden,
            w["w1"],
            w["w2"],
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
        )

    return fn


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def tensor_hash(*tensors):
    """Stable content hash of one or more tensors (first 16 hex chars)."""
    hh = hashlib.sha256()
    for t in tensors:
        b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hh.update(b)
    return hh.hexdigest()[:16]


BENCHMARK_LABELS = {
    "hip": "hip gemm",
    "flyg": "flydsl-gemm",
    "fly": "fly-2stage",
}


def parse_benchmarks(values):
    selected = []
    valid = set(BENCHMARK_LABELS)
    for value in values:
        for item in value.split(","):
            name = item.strip()
            if not name:
                continue
            if name == "all":
                for benchmark in BENCHMARK_LABELS:
                    if benchmark not in selected:
                        selected.append(benchmark)
                continue
            if name not in valid:
                choices = ", ".join([*sorted(valid), "all"])
                raise argparse.ArgumentTypeError(
                    f"unknown benchmark '{name}', choose from: {choices}"
                )
            if name not in selected:
                selected.append(name)
    if len(selected) not in (2, 3):
        raise argparse.ArgumentTypeError("select exactly 2 or 3 benchmarks")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--M-list", default="4,8,16,32,64,128,256")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["hip", "flyg", "fly"],
        help=(
            "benchmarks to run: choose any 2 or all 3 from hip, flyg, fly, all; "
            "accepts either space-separated or comma-separated values"
        ),
    )
    parser.add_argument(
        "--fly-gemm",
        default="1",
        help=(
            "which gemm(s) the 'flyg' benchmark routes through FlyDSL: "
            "1, 2, or 1,2 (the rest stay on the pure-HIP backend)"
        ),
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="print stable content hashes of weights/inputs/outputs (for "
        "reproducibility checks across runs)",
    )
    args = parser.parse_args()
    try:
        selected_benchmarks = parse_benchmarks(args.benchmarks)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    fly_gemms = set()
    for tok in args.fly_gemm.replace(",", " ").split():
        if tok not in ("1", "2"):
            parser.error(f"--fly-gemm expects 1, 2, or 1,2; got '{tok}'")
        fly_gemms.add(int(tok))
    if not fly_gemms:
        parser.error("--fly-gemm must select at least one of 1, 2")

    # gemm1/gemm2 experiments share the single 'flyg' label; the displayed
    # column just records which gemm(s) were routed through FlyDSL.
    labels = dict(BENCHMARK_LABELS)
    labels["flyg"] = "flydsl-g" + "".join(str(g) for g in sorted(fly_gemms))

    device = torch.device("cuda")
    shape = KIMI
    fly_w, mx_w = build_weights(shape, device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.hash:
        print(
            "weights hash: "
            f"mx_w1={tensor_hash(mx_w['w1'])} mx_w2={tensor_hash(mx_w['w2'])} "
            f"fly_w1={tensor_hash(fly_w['w1'])} fly_w2={tensor_hash(fly_w['w2'])}"
        )
    print(
        f"Shape Kimi-K2.5 TP=4: NE={shape.NE} H={shape.H} "
        f"INTER={shape.INTER} TOPK={shape.TOPK}"
    )
    print("Selected benchmarks: " + ", ".join(selected_benchmarks))

    time_headers = [f"{labels[name]} us" for name in selected_benchmarks]
    pair_headers = []
    for left, right in combinations(selected_benchmarks, 2):
        pair_headers.append(f"{right}/{left}")
        pair_headers.append(f"cos {right}/{left}")
    header = f"\n{'M':>6}"
    for name in time_headers:
        header += f" | {name:>18}"
    for name in pair_headers:
        header += f" | {name:>13}"
    print(header)
    print("-" * len(header))
    for M in [int(x) for x in args.M_list.split(",")]:
        hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
        if args.hash:
            print(
                f"[M={M}] input hash: hidden={tensor_hash(hidden)} "
                f"topk_ids={tensor_hash(topk_ids)} "
                f"topk_weight={tensor_hash(topk_weight)}"
            )
        benchmark_fns = {
            "hip": make_mx_fn(hidden, topk_ids, topk_weight, mx_w, set()),
            "flyg": make_mx_fn(hidden, topk_ids, topk_weight, mx_w, fly_gemms),
            "fly": make_fn(hidden, topk_ids, topk_weight, fly_w),
        }
        outputs = {name: benchmark_fns[name]().clone() for name in selected_benchmarks}
        torch.cuda.synchronize()
        if args.hash:
            hashes = " ".join(
                f"{name}={tensor_hash(outputs[name])}" for name in selected_benchmarks
            )
            print(f"[M={M}] output hash: {hashes}")
        timings = {}
        for name in selected_benchmarks:
            _, timings[name] = run_perftest(
                benchmark_fns[name], num_warmup=args.warmup, num_iters=args.iters
            )
        row = f"{M:>6}"
        for name in selected_benchmarks:
            row += f" | {timings[name]:>18.1f}"
        for left, right in combinations(selected_benchmarks, 2):
            row += f" | {timings[right] / timings[left]:>12.2f}x"
            row += f" | {cosine(outputs[left], outputs[right]):>13.4f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
