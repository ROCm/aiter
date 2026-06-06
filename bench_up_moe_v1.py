import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from itertools import combinations
 
# sys.path.insert(0, os.environ.get("AITER_REPO", "/tmp/aiter-pr3470"))
 
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

from mx_sort_fly_gemm import (
    mx_sort_fly_gemm1_gemm2,
    _fly_gemm1,
    _fly_gemm2,
    _mxfn_regime,
)
 
 
@dataclass(frozen=True)
class Shape:
    NE: int
    H: int
    INTER: int
    TOPK: int
 
 
KIMI = Shape(NE=385, H=7168, INTER=512, TOPK=9)
 
 
def build_weights(shape: Shape, device, seed=0):
    # Fully deterministic across runs: drive every randn from an explicit
    # torch.Generator (isolated from the global RNG state) seeded with `seed`.
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    ne, h, inter = shape.NE, shape.H, shape.INTER
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn(
        (ne, 2 * inter, h), generator=g, dtype=dtypes.bf16, device=device
    ) / 10
    w2 = torch.randn(
        (ne, h, inter), generator=g, dtype=dtypes.bf16, device=device
    ) / 10
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
    # Fully deterministic across runs: every randn is driven by an explicit
    # torch.Generator seeded with `seed` (hidden and routing use separate
    # streams, both seeded identically, isolated from the global RNG state).
    torch.manual_seed(seed)
    ne, h, topk = shape.NE, shape.H, shape.TOPK
    gh = torch.Generator(device=device).manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn((M, h), generator=gh, dtype=dtypes.bf16, device=device) / 10
    n_routed = ne - 1
    shared_id = ne - 1
    n_topk_routed = topk - 1
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


def make_mx_sort_fly_fn(hidden, topk_ids, topk_weight, w):
    """mx_fn sort (+ BM=16 inline-quant / BM=32 separate quant) prologue + fresh
    FlyDSL a4w4 gemm1/gemm2 drop-ins.

    Reuses mx_fn's sort/quant kernels verbatim; gemm1/gemm2 are the new
    ``compile_mxfp4_gemm{1,2}_a4w4`` kernels reading the SAME a16w4 ``mx_w``
    layout (drop-in). BM regime mirrors mx_fn (BM=16 small-M, BM=32 large-M),
    so the kernel trace matches mx_fn except the two gemm names.
    """
    M = hidden.shape[0]
    # mirror mx_fn's exact CSV regime (BM + gemm2 mode) for this shape.
    BM, g2_mode = _mxfn_regime(M)

    def fn():
        return mx_sort_fly_gemm1_gemm2(
            hidden,
            w["w1"],
            w["w2"],
            topk_ids,
            topk_weight,
            topk=topk_ids.shape[1],
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
            BM=BM,
            g2_mode=g2_mode,
            gemm1_backend=_fly_gemm1,
            gemm2_backend=_fly_gemm2,
        )

    return fn
 
 
def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    mask = torch.isfinite(a) & torch.isfinite(b)
    if not bool(mask.all()):
        a = a[mask]
        b = b[mask]
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def tensor_hash(*tensors):
    """Stable content hash of one or more tensors (first 16 hex chars)."""
    hh = hashlib.sha256()
    for t in tensors:
        b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hh.update(b)
    return hh.hexdigest()[:16]


BENCHMARK_LABELS = {
    "mx": "mxfp4 tuned",
    "fly": "flydsl tuned",
    "msfg": "mxsort+flygemm",
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mx", "fly", "msfg"],
        help=(
            "benchmarks to run: choose any 2 or all 3 from mx, fly, msfg, all; "
            "accepts either space-separated or comma-separated values"
        ),
    )
    parser.add_argument(
        "--hash", action="store_true",
        help="print stable content hashes of weights/inputs/outputs (for "
             "reproducibility checks across runs)",
    )
    args = parser.parse_args()
    try:
        selected_benchmarks = parse_benchmarks(args.benchmarks)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
 
    device = torch.device("cuda")
    shape = KIMI
    fly_w, mx_w = build_weights(shape, device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.hash:
        print(
            "weights hash: "
            f"mx_w1={tensor_hash(mx_w['w1'])} mx_w2={tensor_hash(mx_w['w2'])} "
            f"mx_w1s={tensor_hash(mx_w['w1_scale'])} "
            f"mx_w2s={tensor_hash(mx_w['w2_scale'])}"
        )
    print(
        f"Shape Kimi-K2.5 TP=4: NE={shape.NE} H={shape.H} "
        f"INTER={shape.INTER} TOPK={shape.TOPK}"
    )
    print("Selected benchmarks: " + ", ".join(selected_benchmarks))

    time_headers = [f"{BENCHMARK_LABELS[name]} us" for name in selected_benchmarks]
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
        fly_fn = make_fn(hidden, topk_ids, topk_weight, fly_w)
        mx_fn = make_fn(hidden, topk_ids, topk_weight, mx_w)
        mx_sort_fly_gemm1_gemm2_fn = make_mx_sort_fly_fn(
            hidden, topk_ids, topk_weight, mx_w
        )
        benchmark_fns = {
            "mx": mx_fn,
            "fly": fly_fn,
            "msfg": mx_sort_fly_gemm1_gemm2_fn,
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
 
 